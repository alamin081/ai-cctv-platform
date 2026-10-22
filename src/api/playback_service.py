"""
AI CCTV Platform - Real-Time Playback Analysis Service
Manages real-time AI frame processing sessions synchronized with HTML5 video playback.
Includes shared singleton YOLO detector, session-isolated tracking & zone analytics,
instant seek re-synchronization, and adaptive frame sampling.
"""

import asyncio
import json
import logging
import math
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

import cv2
import yaml

from src.analytics import (
    EntryExitEngine,
    EntryExitResult,
    FootfallAnalyticsEngine,
)
from src.detection import DetectionResult, YOLOPersonDetector
from src.tracking import ByteTrackPersonTracker, TrackStateManager
from src.video_input import Frame, VideoFileReader
from src.zones import ZoneCountingEngine, ZoneEngine

logger = logging.getLogger("api.playback_service")


class SharedDetectorManager:
    """
    Singleton wrapper for YOLOPersonDetector.
    Ensures model weights and CUDA/CPU resources are loaded once into memory
    and shared safely across multiple concurrent playback sessions.
    """

    _instance: Optional["SharedDetectorManager"] = None
    _init_lock: threading.Lock = threading.Lock()

    def __init__(self, project_root: Path, config: Dict[str, Any]) -> None:
        self.project_root = project_root
        self.config = config
        self._inference_lock = threading.Lock()

        detection_cfg = config.get("detection", {})
        model_name = detection_cfg.get("model_name", "data/models/yolov8n.pt")
        model_path = project_root / model_name if not Path(model_name).is_absolute() else Path(model_name)
        if not model_path.exists():
            model_path = project_root / "yolov8n.pt"
            if not model_path.exists():
                model_path = Path("yolov8n.pt")

        conf = float(detection_cfg.get("confidence", 0.40))
        iou = float(detection_cfg.get("iou", 0.45))
        device = str(detection_cfg.get("device", "auto"))

        logger.info("Initializing SharedDetectorManager with model: %s on device: %s", model_path, device)
        self.detector = YOLOPersonDetector(
            model_name_or_path=model_path,
            confidence_threshold=conf,
            iou_threshold=iou,
            device=device,
        )
        logger.info("SharedDetectorManager initialized successfully [%s]", self.detector.device_display_name)

    @classmethod
    def get_instance(cls, project_root: Path, config: Dict[str, Any]) -> "SharedDetectorManager":
        """Get or initialize singleton instance."""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(project_root, config)
            return cls._instance

    def detect(self, frame: Frame) -> DetectionResult:
        """Thread-safe inference using the shared YOLO model."""
        with self._inference_lock:
            return self.detector.detect(frame)

    @property
    def device_display_name(self) -> str:
        return self.detector.device_display_name


class PlaybackSession:
    """
    Stateful real-time analysis session for a specific video playback instance.
    Keeps session-isolated tracking, state management, and spatial zones.
    """

    def __init__(
        self,
        session_id: str,
        video_filename: str,
        video_path: Path,
        config: Dict[str, Any],
        detector_manager: SharedDetectorManager,
        zone_store: Optional[Any] = None,
    ) -> None:
        self.session_id = session_id
        self.video_filename = video_filename
        self.video_path = video_path
        self.config = config
        self.detector_manager = detector_manager
        self._zone_store = zone_store

        self.session_lock = threading.Lock()
        self.last_activity = time.time()

        # Playback states
        self.is_playing: bool = False
        self.playback_speed: float = 1.0
        self.stride: int = 1
        self.dropped_frames: int = 0
        self.processed_frames: int = 0
        self.last_latency_ms: float = 0.0
        self.avg_latency_ms: float = 0.0

        # Video Reader
        self.reader = VideoFileReader(self.video_path)
        self.reader.open()
        self.fps: float = self.reader.fps if self.reader.fps > 0 else 30.0
        self.width: int = self.reader.width
        self.height: int = self.reader.height
        self.total_frames: int = self.reader.total_frames
        self.duration_seconds: float = self.reader.duration

        # Session-isolated Analytics Engines
        tracking_cfg = self.config.get("tracking", {})
        state_cfg = self.config.get("track_state", {})
        zones_cfg = self.config.get("zones", {})
        counting_cfg = self.config.get("zone_counting", {})
        entry_exit_cfg = self.config.get("entry_exit", {})
        footfall_cfg = self.config.get("footfall", {})

        self.tracker = ByteTrackPersonTracker(
            track_high_thresh=float(tracking_cfg.get("track_high_thresh", 0.5)),
            track_low_thresh=float(tracking_cfg.get("track_low_thresh", 0.1)),
            new_track_thresh=float(tracking_cfg.get("new_track_thresh", 0.6)),
            track_buffer=int(tracking_cfg.get("track_buffer", 30)),
            match_thresh=float(tracking_cfg.get("match_thresh", 0.8)),
            fuse_score=bool(tracking_cfg.get("fuse_score", True)),
        )

        self.state_manager = TrackStateManager(
            max_missed_frames=int(state_cfg.get("max_missed_frames", 30))
        )

        # Load zone config from persistent store if available, else from config
        if self._zone_store is not None:
            zones_engine_cfg = self._zone_store.load_as_engine_config(enabled_only=True)
        else:
            zones_engine_cfg = zones_cfg

        self.zone_engine: Optional[ZoneEngine] = None
        if zones_cfg.get("enabled", True):
            self.zone_engine = ZoneEngine.from_config(zones_engine_cfg)
            # Set frame dimensions for normalized → pixel coordinate conversion
            self.zone_engine.set_frame_size(self.width, self.height)

        self.counting_engine: Optional[ZoneCountingEngine] = None
        if counting_cfg.get("enabled", True) and self.zone_engine:
            self.counting_engine = ZoneCountingEngine.from_zone_engine(self.zone_engine)

        self.entry_exit_engine: Optional[EntryExitEngine] = None
        if entry_exit_cfg.get("enabled", True):
            self.entry_exit_engine = EntryExitEngine.from_config(entry_exit_cfg)

        self.footfall_engine: Optional[FootfallAnalyticsEngine] = None
        if footfall_cfg.get("enabled", True):
            self.footfall_engine = FootfallAnalyticsEngine.from_config(footfall_cfg)

        # Cache of recently analyzed frames: { frame_number: frame_data }
        self.frames_cache: Dict[int, Dict[str, Any]] = {}
        self.recent_events: List[Dict[str, Any]] = []

        self.current_frame_number: int = 0
        self.current_timestamp: float = 0.0

        # Pre-analyze frame 1 so initial overlay is immediately available
        self._analyze_initial_frame()

    def _analyze_initial_frame(self) -> None:
        """Analyze first frame upon session creation."""
        with self.session_lock:
            frame = self.reader.read()
            if frame:
                data = self._process_frame_unlocked(frame)
                self.current_frame_number = frame.frame_number
                self.current_timestamp = frame.timestamp
                self.frames_cache[frame.frame_number] = data

    def reset_state_engines(self) -> None:
        """Reset internal tracking and spatial states."""
        self.tracker.reset()
        self.state_manager.reset()
        if self.zone_engine:
            self.zone_engine.reset()
        if self.counting_engine:
            self.counting_engine.reset()
        if self.entry_exit_engine:
            self.entry_exit_engine.reset()
        if self.footfall_engine:
            self.footfall_engine.reset()
        logger.info("Session %s engines reset.", self.session_id)

    def reset_spatial_engines(self) -> None:
        """Reset only zone/counting/entry-exit/footfall state, NOT tracker or detector."""
        if self.zone_engine:
            self.zone_engine.reset()
        if self.counting_engine:
            self.counting_engine.reset()
        if self.entry_exit_engine:
            self.entry_exit_engine.reset()
        if self.footfall_engine:
            self.footfall_engine.reset()
        self.recent_events.clear()
        logger.info("Session %s spatial engines reset.", self.session_id)

    def reload_zones(self, zone_store: Any) -> None:
        """
        Hot-reload zones from persistent store.
        Replaces ZoneEngine and ZoneCountingEngine with new zone definitions,
        resets spatial analytics state, but preserves tracker and video position.
        """
        with self.session_lock:
            zones_engine_cfg = zone_store.load_as_engine_config(enabled_only=True)
            counting_cfg = self.config.get("zone_counting", {})
            entry_exit_cfg = self.config.get("entry_exit", {})
            footfall_cfg = self.config.get("footfall", {})

            self.zone_engine = ZoneEngine.from_config(zones_engine_cfg)
            self.zone_engine.set_frame_size(self.width, self.height)

            if counting_cfg.get("enabled", True):
                self.counting_engine = ZoneCountingEngine.from_zone_engine(self.zone_engine)
            else:
                self.counting_engine = None

            if entry_exit_cfg.get("enabled", True):
                self.entry_exit_engine = EntryExitEngine.from_config(entry_exit_cfg)
            else:
                self.entry_exit_engine = None

            if footfall_cfg.get("enabled", True):
                self.footfall_engine = FootfallAnalyticsEngine.from_config(footfall_cfg)
            else:
                self.footfall_engine = None

            self.recent_events.clear()
            logger.info("Session %s zones hot-reloaded.", self.session_id)

    def set_speed(self, speed: float) -> None:
        """Update playback speed and adjust frame sampling stride accordingly."""
        with self.session_lock:
            self.last_activity = time.time()
            self.playback_speed = max(0.25, min(4.0, float(speed)))
            if self.playback_speed <= 0.75:
                self.stride = 1
            elif self.playback_speed <= 1.25:
                # 1.0x: stride 1 unless latency is high (> 45ms)
                self.stride = 2 if self.avg_latency_ms > 45.0 else 1
            else:
                # 2.0x or higher: stride 2 to easily keep pace with 2x video
                self.stride = 2 if self.avg_latency_ms < 50.0 else 3
            logger.info("Session %s speed set to %.2fx (stride: %d)", self.session_id, self.playback_speed, self.stride)

    def set_playing(self, is_playing: bool, timestamp: Optional[float] = None) -> None:
        """Set play/pause status and update current playback timestamp."""
        with self.session_lock:
            self.last_activity = time.time()
            self.is_playing = bool(is_playing)
            if timestamp is not None:
                self.current_timestamp = float(timestamp)
                self.current_frame_number = max(1, int(round(self.current_timestamp * self.fps)) + 1)

    def process_frame_at(self, timestamp: float, frame_number: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Directly process the frame presented at given timestamp / frame_number.
        Maintains ByteTrack tracking continuity for sequential forward playback,
        while handling intentional jumps/seeks cleanly.
        """
        with self.session_lock:
            self.last_activity = time.time()
            if not self.reader.is_opened():
                return None

            ts = max(0.0, float(timestamp))
            target_frame = frame_number if frame_number is not None else max(1, int(round(ts * self.fps)) + 1)
            target_frame = min(self.total_frames, max(1, target_frame))

            curr_pos = self.reader.current_frame_number

            # If client jumped backwards, or jumped far ahead (> 2.0s / 30 frames):
            # This indicates an unannounced scrub/jump: reset tracking state to avoid corrupting tracker
            if target_frame < curr_pos or (target_frame - curr_pos) > int(self.fps * 2.0):
                self.reader.seek(target_frame)
                self.reset_state_engines()
            elif target_frame != curr_pos + 1:
                # Fast forward sequential grab if gap is small (1-15 frames) for ultra-low latency (3ms vs 120ms seek)
                skip_count = target_frame - (curr_pos + 1)
                if 1 <= skip_count <= 15 and hasattr(self.reader, '_cap') and self.reader._cap is not None:
                    for _ in range(skip_count):
                        self.reader._cap.grab()
                    self.reader._current_frame_number = target_frame - 1
                    self.reader._current_timestamp = (self.reader._current_frame_number / self.fps) if self.fps > 0 else 0.0
                else:
                    self.reader.seek(target_frame)

            frame = self.reader.read()
            if frame is None:
                return None

            try:
                data = self._process_frame_unlocked(frame)
            except Exception as e:
                logger.error("Error processing frame %d: %s", target_frame, e)
                return {
                    "frame_number": target_frame,
                    "timestamp": ts,
                    "detections_count": 0,
                    "tracks": [],
                    "occupancy": {},
                    "metrics": self.get_current_metrics(),
                    "new_events": [],
                }

            self.current_frame_number = frame.frame_number
            self.current_timestamp = frame.timestamp
            self.frames_cache[frame.frame_number] = data

            if len(self.frames_cache) > 300:
                keys_to_del = sorted(self.frames_cache.keys())[:-300]
                for k in keys_to_del:
                    self.frames_cache.pop(k, None)

            return data

    def seek(self, timestamp: float, frame_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Seek to specified timestamp / frame number.
        Resets tracking & zone state to prevent state carry-over.
        Immediately analyzes and returns the seeked frame.
        """
        with self.session_lock:
            self.last_activity = time.time()
            ts = max(0.0, float(timestamp))
            target_frame = frame_number if frame_number is not None else max(1, int(round(ts * self.fps)) + 1)
            target_frame = min(self.total_frames, max(1, target_frame))

            logger.info("Session %s seek to frame %d (%.3fs)", self.session_id, target_frame, ts)

            # 1. Seek the video file reader
            self.reader.seek(target_frame)

            # 2. Reset tracking and zone states
            self.reset_state_engines()

            # 3. Read and analyze the seeked frame immediately
            frame = self.reader.read()
            if frame is not None:
                data = self._process_frame_unlocked(frame)
                self.current_frame_number = frame.frame_number
                self.current_timestamp = frame.timestamp
                self.frames_cache[frame.frame_number] = data
                return data
            else:
                # End of video fallback
                return {
                    "frame_number": target_frame,
                    "timestamp": ts,
                    "detections_count": 0,
                    "tracks": [],
                    "occupancy": {},
                    "metrics": self.get_current_metrics(),
                    "new_events": [],
                }

    def _process_frame_unlocked(self, frame: Frame) -> Dict[str, Any]:
        """Process a single frame through the full AI analytics pipeline."""
        t_start = time.perf_counter()

        # Step 1: YOLO Person Detection (Thread-safe shared inference)
        det_result = self.detector_manager.detect(frame)

        # Step 2: ByteTrack Multi-Object Tracking
        track_result = self.tracker.update(
            detections=det_result,
            frame_number=frame.frame_number,
            timestamp=frame.timestamp,
            frame_bgr=frame.data,
        )

        # Step 3: Track State Manager
        state_result = self.state_manager.update(
            tracks=track_result,
            frame_number=frame.frame_number,
            timestamp=frame.timestamp,
        )

        removed_ids = [t.track_id for t in self.state_manager.removed_tracks]

        # Step 4: Spatial Zone Engine
        zone_result = None
        if self.zone_engine:
            zone_result = self.zone_engine.update(
                tracks=state_result,
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                removed_track_ids=removed_ids,
            )

        # Step 5: Zone Occupancy Counting
        counting_result = None
        if self.counting_engine and zone_result:
            counting_result = self.counting_engine.update(
                zone_result=zone_result,
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                removed_track_ids=removed_ids,
            )

        # Step 6: Entry / Exit Engine
        entry_exit_result = None
        if self.entry_exit_engine and zone_result:
            entry_exit_result = self.entry_exit_engine.update(
                zone_result=zone_result,
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                removed_track_ids=removed_ids,
            )

        # Step 7: Footfall Analytics Engine
        footfall_result = None
        if self.footfall_engine and entry_exit_result:
            footfall_result = self.footfall_engine.update(
                entry_exit_input=entry_exit_result,
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
            )

        t_elapsed = (time.perf_counter() - t_start) * 1000.0  # ms
        self.last_latency_ms = t_elapsed
        self.processed_frames += 1
        self.avg_latency_ms = (
            (self.avg_latency_ms * 0.9 + t_elapsed * 0.1)
            if self.avg_latency_ms > 0
            else t_elapsed
        )

        # Extract new events
        new_events = []
        if entry_exit_result:
            for ev in entry_exit_result.events:
                ev_dict = {
                    "event_id": ev.event_id,
                    "type": ev.movement_type,
                    "track_id": ev.track_id,
                    "zone_id": ev.zone_id or ev.current_zone_id,
                    "previous_zone_id": ev.previous_zone_id,
                    "frame_number": frame.frame_number,
                    "timestamp": round(frame.timestamp, 2),
                }
                new_events.append(ev_dict)
                self.recent_events.append(ev_dict)

        # Format tracks for canvas overlay
        active_tracks = [
            {
                "track_id": t.track_id,
                "state": t.state,
                "confidence": round(t.confidence, 3),
                "bbox": [round(c, 1) for c in t.bbox],
            }
            for t in state_result.tracks
        ]

        # Format occupancy snapshot
        occ_snapshot = {}
        if counting_result:
            for zid, occ in counting_result.occupancy.items():
                occ_snapshot[zid] = {
                    "zone_name": occ.zone_name,
                    "count": occ.count,
                    "track_ids": occ.track_ids,
                }

        calc_fps = 1000.0 / self.avg_latency_ms if self.avg_latency_ms > 0 else 30.0

        metrics = {
            "detected_persons": det_result.count,
            "active_track_ids": [t.track_id for t in state_result.tracks if t.state in ("ACTIVE", "NEW")],
            "total_unique_tracks": len(self.tracker.unique_track_ids),
            "total_entries": self.entry_exit_engine.total_entries if self.entry_exit_engine else 0,
            "total_exits": self.entry_exit_engine.total_exits if self.entry_exit_engine else 0,
            "net_flow": (
                (self.entry_exit_engine.total_entries - self.entry_exit_engine.total_exits)
                if self.entry_exit_engine
                else 0
            ),
            "total_zone_transitions": (
                self.entry_exit_engine.total_zone_transitions
                if self.entry_exit_engine
                else 0
            ),
            "footfall_total": (
                self.footfall_engine.total_movement_events
                if self.footfall_engine
                else 0
            ),
            "pipeline_fps": round(calc_fps, 1),
            "latency_ms": round(self.last_latency_ms, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "dropped_frames": self.dropped_frames,
            "stride": self.stride,
            "playback_speed": self.playback_speed,
            "device": self.detector_manager.device_display_name,
        }

        frame_data = {
            "frame_number": frame.frame_number,
            "timestamp": round(frame.timestamp, 3),
            "detections_count": det_result.count,
            "tracks": active_tracks,
            "occupancy": occ_snapshot,
            "metrics": metrics,
            "new_events": new_events,
        }

        return frame_data

    def process_next_frame(self, client_time: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Advance video reader and process the next frame respecting stride.
        If client_time is provided and reader has fallen too far behind (> 0.4s),
        skips frames directly to keep in tight synchronization with playback.
        """
        with self.session_lock:
            self.last_activity = time.time()
            if not self.reader.is_opened():
                return None

            # Synchronization check: if server lags significantly behind client video
            if client_time is not None and client_time > 0:
                expected_frame = int(round(client_time * self.fps)) + 1
                curr_frame = self.reader.current_frame_number
                frame_lag = expected_frame - curr_frame

                if frame_lag > 5:  # more than 5 frames (~160ms) behind
                    logger.debug("Playback sync lag detected (%d frames). Fast-forwarding to %d", frame_lag, expected_frame)
                    self.dropped_frames += (frame_lag - 1)
                    self.reader.seek(expected_frame)

            # Apply stride (skip intermediate frames if stride > 1)
            if self.stride > 1:
                for _ in range(self.stride - 1):
                    skipped = self.reader.read()
                    if skipped is None:
                        return None
                    self.dropped_frames += 1

            frame = self.reader.read()
            if frame is None:
                return None

            data = self._process_frame_unlocked(frame)
            self.current_frame_number = frame.frame_number
            self.current_timestamp = frame.timestamp
            self.frames_cache[frame.frame_number] = data

            # Cap frame cache size to last 200 frames
            if len(self.frames_cache) > 200:
                keys_to_del = sorted(self.frames_cache.keys())[:-200]
                for k in keys_to_del:
                    self.frames_cache.pop(k, None)

            return data

    def get_current_metrics(self) -> Dict[str, Any]:
        """Return current cumulative and live metrics."""
        calc_fps = 1000.0 / self.avg_latency_ms if self.avg_latency_ms > 0 else 30.0
        return {
            "detected_persons": 0,
            "active_track_ids": [],
            "total_unique_tracks": len(self.tracker.unique_track_ids),
            "total_entries": self.entry_exit_engine.total_entries if self.entry_exit_engine else 0,
            "total_exits": self.entry_exit_engine.total_exits if self.entry_exit_engine else 0,
            "net_flow": (
                (self.entry_exit_engine.total_entries - self.entry_exit_engine.total_exits)
                if self.entry_exit_engine
                else 0
            ),
            "total_zone_transitions": (
                self.entry_exit_engine.total_zone_transitions
                if self.entry_exit_engine
                else 0
            ),
            "footfall_total": (
                self.footfall_engine.total_movement_events
                if self.footfall_engine
                else 0
            ),
            "pipeline_fps": round(calc_fps, 1),
            "latency_ms": round(self.last_latency_ms, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "dropped_frames": self.dropped_frames,
            "stride": self.stride,
            "playback_speed": self.playback_speed,
            "device": self.detector_manager.device_display_name,
        }

    def close(self) -> None:
        """Release session resources."""
        with self.session_lock:
            self.is_playing = False
            self.reader.release()
            self.frames_cache.clear()
            logger.info("Session %s closed.", self.session_id)


class PlaybackService:
    """
    Manages active playback sessions, handles SSE streaming generators,
    and coordinates real-time AI analytics.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.videos_dir = project_root / "data" / "videos"
        self.config_path = project_root / "config.yaml"

        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()

        self.detector_manager = SharedDetectorManager.get_instance(self.project_root, self.config)
        self.sessions: Dict[str, PlaybackSession] = {}
        self._lock = threading.Lock()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return {}

    def create_or_get_session(
        self, video_filename: str, zone_store: Optional[Any] = None,
    ) -> PlaybackSession:
        """Create a new playback session or reuse active session for the given video."""
        decoded_name = urllib.parse.unquote(video_filename).strip()
        safe_name = Path(decoded_name).name
        video_path = self.videos_dir / safe_name

        if not video_path.exists():
            matched = None
            for f in self.videos_dir.iterdir():
                if f.is_file() and (f.name.lower() == safe_name.lower() or f.stem.lower() == Path(safe_name).stem.lower()):
                    matched = f
                    break
            if matched:
                video_path = matched
            else:
                raise FileNotFoundError(f"Video file '{video_filename}' not found.")

        session_id = f"play_{Path(video_path).stem}_{uuid.uuid4().hex[:6]}"

        session = PlaybackSession(
            session_id=session_id,
            video_filename=video_path.name,
            video_path=video_path,
            config=self.config,
            detector_manager=self.detector_manager,
            zone_store=zone_store,
        )

        with self._lock:
            # Clean up old sessions (> 10 minutes inactive)
            now = time.time()
            stale = [sid for sid, s in self.sessions.items() if now - s.last_activity > 600]
            for sid in stale:
                try:
                    self.sessions[sid].close()
                except Exception:
                    pass
                self.sessions.pop(sid, None)

            self.sessions[session_id] = session

        return session

    def reload_all_sessions_zones(self, zone_store: Any) -> int:
        """
        Hot-reload zones for all active playback sessions.
        Returns the number of sessions reloaded.
        """
        count = 0
        with self._lock:
            for sid, session in self.sessions.items():
                try:
                    session.reload_zones(zone_store)
                    count += 1
                except Exception as e:
                    logger.error("Failed to reload zones for session %s: %s", sid, e)
        return count

    def get_session(self, session_id: str) -> Optional[PlaybackSession]:
        with self._lock:
            return self.sessions.get(session_id)

    def process_frame_for_session(
        self, session_id: str, timestamp: float, frame_number: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Process a specific presented frame for an active session."""
        session = self.get_session(session_id)
        if not session:
            return None
        return session.process_frame_at(timestamp, frame_number)

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            session = self.sessions.pop(session_id, None)
            if session:
                session.close()
                return True
            return False

    async def sse_frame_generator(
        self, session: PlaybackSession
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronous generator yielding Server-Sent Events (SSE) for analyzed frames.
        Synchronizes generation speed with video playback pacing.
        """
        logger.info("Starting SSE stream for session %s", session.session_id)

        try:
            while True:
                # If session is paused, idle wait briefly and send keep-alive
                if not session.is_playing:
                    await asyncio.sleep(0.1)
                    continue

                # Process next frame in worker thread to prevent blocking event loop
                data = await asyncio.to_thread(session.process_next_frame)

                if data is None:
                    # End of video reached or paused
                    yield f"event: end_of_stream\ndata: {json.dumps({'message': 'End of video reached'})}\n\n"
                    session.set_playing(False)
                    await asyncio.sleep(0.5)
                    continue

                # Format SSE payload
                payload = json.dumps(data)
                yield f"data: {payload}\n\n"

                # Calculate pacing delay: subtract inference latency so net rate matches video FPS
                frame_duration = 1.0 / (session.fps * session.playback_speed)
                ideal_interval = frame_duration * session.stride
                sleep_time = max(0.001, ideal_interval - (session.last_latency_ms / 1000.0))
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("SSE client disconnected for session %s", session.session_id)
        except Exception as e:
            logger.exception("Error in SSE frame generator for session %s: %s", session.session_id, e)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

