"""
AI CCTV Platform - Background Video Pipeline Service
Manages processing jobs, executes the AI CCTV pipeline, and aggregates live metrics.
"""

import json
import logging
import os
import subprocess
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yaml

from src.analytics import (
    EntryExitEngine,
    EntryExitResult,
    FootfallAnalyticsEngine,
)
from src.detection import YOLOPersonDetector
from src.tracking import ByteTrackPersonTracker, TrackStateManager
from src.video_input import VideoFileReader
from src.zones import ZoneConfigStore, ZoneCountingEngine, ZoneEngine

logger = logging.getLogger("api.pipeline_service")


class PipelineJob:
    """Represents a background video analysis job with real-time status and metrics."""

    def __init__(
        self,
        job_id: str,
        video_filename: str,
        video_path: Path,
        output_dir: Path,
        config: Dict[str, Any],
    ):
        self.job_id = job_id
        self.video_filename = video_filename
        self.video_path = video_path
        self.output_dir = output_dir
        self.config = config
        self.output_video_path = output_dir / f"{job_id}_annotated.mp4"

        # Status tracking
        self.status: str = "QUEUED"  # QUEUED, PROCESSING, COMPLETED, FAILED
        self.progress: float = 0.0
        self.current_frame: int = 0
        self.total_frames: int = 0
        self.current_timestamp: float = 0.0
        self.fps: float = 0.0
        self.pipeline_fps: float = 0.0
        self.start_time: float = 0.0
        self.elapsed_seconds: float = 0.0
        self.error_message: Optional[str] = None

        # Live and cumulative metrics
        self.current_metrics: Dict[str, Any] = {
            "detected_persons": 0,
            "active_track_ids": [],
            "total_unique_tracks": 0,
            "occupancy": {},
            "total_entries": 0,
            "total_exits": 0,
            "total_zone_transitions": 0,
            "net_flow": 0,
            "footfall_total": 0,
        }
        self.recent_events: List[Dict[str, Any]] = []
        self.final_summary: Optional[Dict[str, Any]] = None

        # Per-frame metadata for high-precision synchronized UI playback
        self.frames_data: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert job state to JSON-serializable dictionary."""
        return {
            "job_id": self.job_id,
            "video_filename": self.video_filename,
            "video_url": f"/media/videos/{urllib.parse.quote(self.video_filename)}",
            "status": self.status,
            "progress": round(self.progress, 1),
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "current_timestamp": round(self.current_timestamp, 3),
            "fps": round(self.fps, 2),
            "pipeline_fps": round(self.pipeline_fps, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "error_message": self.error_message,
            "current_metrics": self.current_metrics,
            "recent_events": self.recent_events[-50:],  # Return up to 50 recent events
            "has_annotated_video": self.output_video_path.exists(),
            "annotated_video_filename": self.output_video_path.name if self.output_video_path.exists() else None,
            "annotated_video_url": f"/media/outputs/{urllib.parse.quote(self.output_video_path.name)}" if self.output_video_path.exists() else None,
            "final_summary": self.final_summary,
        }


class PipelineService:
    """Singleton service to submit, monitor, and execute AI pipeline jobs."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.videos_dir = project_root / "data" / "videos"
        self.outputs_dir = project_root / "data" / "outputs"
        self.config_path = project_root / "config.yaml"

        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.jobs: Dict[str, PipelineJob] = {}
        self._lock = threading.Lock()

    def get_config(self) -> Dict[str, Any]:
        """Load YAML configuration with defaults."""
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return {}

    def list_available_videos(self) -> List[Dict[str, Any]]:
        """List all video files available in data/videos/ with metadata and streaming URLs."""
        videos = []
        valid_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        for f in sorted(self.videos_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in valid_extensions:
                size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                video_url = f"/media/videos/{urllib.parse.quote(f.name)}"
                v_info: Dict[str, Any] = {
                    "filename": f.name,
                    "video_url": video_url,
                    "size_mb": size_mb,
                    "resolution": "Unknown",
                    "fps": 0.0,
                    "duration_seconds": 0.0,
                    "total_frames": 0,
                }
                # Probe video metadata using OpenCV
                try:
                    cap = cv2.VideoCapture(str(f))
                    if cap.isOpened():
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = float(cap.get(cv2.CAP_PROP_FPS))
                        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        dur = round(frames / fps, 2) if fps > 0 else 0.0
                        v_info.update({
                            "resolution": f"{w}x{h}",
                            "width": w,
                            "height": h,
                            "fps": round(fps, 2),
                            "duration_seconds": dur,
                            "total_frames": frames,
                        })
                        cap.release()
                except Exception as e:
                    logger.debug("Could not probe video %s: %s", f.name, e)
                videos.append(v_info)
        return videos

    def create_job(self, video_filename: str) -> PipelineJob:
        """Create and queue a new processing job with robust filename lookup."""
        decoded_name = urllib.parse.unquote(video_filename).strip()
        safe_name = Path(decoded_name).name
        video_path = self.videos_dir / safe_name

        if not video_path.exists():
            # Fallback search
            video_path = self.videos_dir / video_filename
        if not video_path.exists():
            matched = None
            for f in self.videos_dir.iterdir():
                if f.is_file() and (f.name.lower() == safe_name.lower() or f.stem.lower() == Path(safe_name).stem.lower()):
                    matched = f
                    break
            if matched:
                video_path = matched
            else:
                raise FileNotFoundError(f"Video file '{video_filename}' not found in {self.videos_dir}")

        job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        config = self.get_config()

        job = PipelineJob(
            job_id=job_id,
            video_filename=video_path.name,
            video_path=video_path,
            output_dir=self.outputs_dir,
            config=config,
        )

        with self._lock:
            self.jobs[job_id] = job

        # Launch background thread
        thread = threading.Thread(
            target=self._run_job_worker,
            args=(job,),
            daemon=True,
            name=f"Worker-{job_id}",
        )
        thread.start()

        return job

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        """Retrieve job by ID."""
        with self._lock:
            return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Retrieve list of all jobs."""
        with self._lock:
            return [job.to_dict() for job in reversed(list(self.jobs.values()))]

    def _run_job_worker(self, job: PipelineJob) -> None:
        """Execute full Phase 1-9 pipeline for the given job."""
        logger.info("Starting processing job %s for video %s", job.job_id, job.video_filename)
        job.status = "PROCESSING"
        job.start_time = time.perf_counter()

        reader: Optional[VideoFileReader] = None
        video_writer: Optional[cv2.VideoWriter] = None

        try:
            config = job.config
            detection_cfg = config.get("detection", {})
            tracking_cfg = config.get("tracking", {})
            state_cfg = config.get("track_state", {})
            zones_cfg = config.get("zones", {})
            counting_cfg = config.get("zone_counting", {})
            entry_exit_cfg = config.get("entry_exit", {})
            footfall_cfg = config.get("footfall", {})

            # 1. Initialize Reader
            reader = VideoFileReader(job.video_path)
            reader.open()
            job.total_frames = reader.total_frames
            job.fps = reader.fps

            # 2. Initialize YOLO Detector
            model_name = detection_cfg.get("model_name", "data/models/yolov8n.pt")
            model_path = self.project_root / model_name if not Path(model_name).is_absolute() else Path(model_name)
            if not model_path.exists():
                model_path = Path("yolov8n.pt")

            detector = YOLOPersonDetector(
                model_name_or_path=model_path,
                confidence_threshold=float(detection_cfg.get("confidence", 0.40)),
                iou_threshold=float(detection_cfg.get("iou", 0.45)),
                device=str(detection_cfg.get("device", "auto")),
            )

            # 3. Initialize ByteTrack Tracker
            tracker = ByteTrackPersonTracker(
                track_high_thresh=float(tracking_cfg.get("track_high_thresh", 0.5)),
                track_low_thresh=float(tracking_cfg.get("track_low_thresh", 0.1)),
                new_track_thresh=float(tracking_cfg.get("new_track_thresh", 0.6)),
                track_buffer=int(tracking_cfg.get("track_buffer", 30)),
                match_thresh=float(tracking_cfg.get("match_thresh", 0.8)),
                fuse_score=bool(tracking_cfg.get("fuse_score", True)),
            )

            # 4. Initialize Track State Manager
            state_manager = TrackStateManager(
                max_missed_frames=int(state_cfg.get("max_missed_frames", 30))
            )

            # 5. Initialize Zone Engine from persistent zones.json
            zone_store = ZoneConfigStore(self.project_root)
            zones_engine_cfg = zone_store.load_as_engine_config(enabled_only=True)
            zone_engine = None
            if zones_cfg.get("enabled", True):
                zone_engine = ZoneEngine.from_config(zones_engine_cfg)
                # Set frame dimensions for normalized → pixel coordinate conversion
                zone_engine.set_frame_size(reader.width, reader.height)

            # 6. Initialize Zone Counting Engine
            counting_engine = None
            if counting_cfg.get("enabled", True) and zone_engine:
                counting_engine = ZoneCountingEngine.from_zone_engine(zone_engine)

            # 7. Initialize Entry / Exit Engine
            entry_exit_engine = None
            if entry_exit_cfg.get("enabled", True):
                entry_exit_engine = EntryExitEngine.from_config(entry_exit_cfg)

            # 8. Initialize Footfall Analytics Engine
            footfall_engine = None
            if footfall_cfg.get("enabled", True):
                footfall_engine = FootfallAnalyticsEngine.from_config(footfall_cfg)

            # 9. Initialize Video Writer for Visual Annotation
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(job.output_video_path),
                fourcc,
                reader.fps if reader.fps > 0 else 30.0,
                (reader.width, reader.height),
            )

            # Frame processing loop
            while True:
                frame = reader.read()
                if frame is None:
                    break

                job.current_frame = frame.frame_number
                job.current_timestamp = frame.timestamp
                if job.total_frames > 0:
                    job.progress = min(99.0, (job.current_frame / job.total_frames) * 100.0)

                # Pipeline steps
                det_result = detector.detect(frame)
                track_result = tracker.update(
                    detections=det_result,
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                    frame_bgr=frame.data,
                )
                state_result = state_manager.update(
                    tracks=track_result,
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                )

                removed_ids = [t.track_id for t in state_manager.removed_tracks]

                zone_result = None
                if zone_engine:
                    zone_result = zone_engine.update(
                        tracks=state_result,
                        frame_number=frame.frame_number,
                        timestamp=frame.timestamp,
                        removed_track_ids=removed_ids,
                    )

                counting_result = None
                if counting_engine and zone_result:
                    counting_result = counting_engine.update(
                        zone_result=zone_result,
                        frame_number=frame.frame_number,
                        timestamp=frame.timestamp,
                        removed_track_ids=removed_ids,
                    )

                entry_exit_result = None
                if entry_exit_engine and zone_result:
                    entry_exit_result = entry_exit_engine.update(
                        zone_result=zone_result,
                        frame_number=frame.frame_number,
                        timestamp=frame.timestamp,
                        removed_track_ids=removed_ids,
                    )

                footfall_result = None
                if footfall_engine and entry_exit_result:
                    footfall_result = footfall_engine.update(
                        entry_exit_input=entry_exit_result,
                        frame_number=frame.frame_number,
                        timestamp=frame.timestamp,
                    )

                # Write annotated frame
                if video_writer:
                    vis_frame = TrackStateManager.draw_state_tracks(frame.data, state_result.tracks)
                    if zone_engine and zone_result:
                        vis_frame = zone_engine.draw_zones(vis_frame, zone_result)
                    if counting_engine and counting_result:
                        vis_frame = counting_engine.draw_occupancy(vis_frame, counting_result)
                    if entry_exit_engine and entry_exit_result:
                        vis_frame = entry_exit_engine.draw_entry_exit(vis_frame, entry_exit_result)
                    if footfall_engine and footfall_result:
                        vis_frame = footfall_engine.draw_footfall(vis_frame, footfall_result)
                    video_writer.write(vis_frame)

                # Record any new events
                if entry_exit_result:
                    for ev in entry_exit_result.events:
                        job.recent_events.append({
                            "event_id": ev.event_id,
                            "type": ev.movement_type,
                            "track_id": ev.track_id,
                            "zone_id": ev.zone_id or ev.current_zone_id,
                            "previous_zone_id": ev.previous_zone_id,
                            "frame_number": frame.frame_number,
                            "timestamp": round(frame.timestamp, 2),
                        })

                # Build frame data snapshot for frontend player synchronization
                active_tracks_snapshot = [
                    {
                        "track_id": t.track_id,
                        "state": t.state,
                        "confidence": round(t.confidence, 3),
                        "bbox": [round(c, 1) for c in t.bbox],
                    }
                    for t in state_result.tracks
                ]

                occ_snapshot = {}
                if counting_result:
                    for zid, occ in counting_result.occupancy.items():
                        occ_snapshot[zid] = {
                            "zone_name": occ.zone_name,
                            "count": occ.count,
                            "track_ids": occ.track_ids,
                        }

                job.frames_data.append({
                    "frame_number": frame.frame_number,
                    "timestamp": round(frame.timestamp, 3),
                    "detections_count": det_result.count,
                    "tracks": active_tracks_snapshot,
                    "occupancy": occ_snapshot,
                })

                # Update live metrics
                elapsed = time.perf_counter() - job.start_time
                job.elapsed_seconds = elapsed
                job.pipeline_fps = job.current_frame / elapsed if elapsed > 0 else 0.0

                active_ids = [t.track_id for t in state_result.tracks if t.state in ("ACTIVE", "NEW")]
                job.current_metrics = {
                    "detected_persons": det_result.count,
                    "active_track_ids": active_ids,
                    "total_unique_tracks": len(tracker.unique_track_ids),
                    "occupancy": occ_snapshot,
                    "total_entries": entry_exit_engine.total_entries if entry_exit_engine else 0,
                    "total_exits": entry_exit_engine.total_exits if entry_exit_engine else 0,
                    "total_zone_transitions": entry_exit_engine.total_zone_transitions if entry_exit_engine else 0,
                    "net_flow": (
                        (entry_exit_engine.total_entries - entry_exit_engine.total_exits)
                        if entry_exit_engine
                        else 0
                    ),
                    "footfall_total": footfall_engine.total_movement_events if footfall_engine else 0,
                }

            # Finalize results
            job.progress = 100.0
            job.status = "COMPLETED"
            total_time = time.perf_counter() - job.start_time
            job.elapsed_seconds = total_time
            job.pipeline_fps = (job.total_frames / total_time) if total_time > 0 else 0.0

            ff_summary = footfall_engine.get_summary() if footfall_engine else None
            job.final_summary = {
                "total_frames_processed": job.current_frame,
                "unique_track_ids": sorted(list(tracker.unique_track_ids)),
                "total_unique_tracks": len(tracker.unique_track_ids),
                "total_entries": entry_exit_engine.total_entries if entry_exit_engine else 0,
                "total_exits": entry_exit_engine.total_exits if entry_exit_engine else 0,
                "total_zone_transitions": entry_exit_engine.total_zone_transitions if entry_exit_engine else 0,
                "net_flow": (
                    (entry_exit_engine.total_entries - entry_exit_engine.total_exits)
                    if entry_exit_engine
                    else 0
                ),
                "occupancy": occ_snapshot,
                "footfall": ff_summary.to_dict() if ff_summary else None,
                "runtime_seconds": round(total_time, 2),
                "avg_pipeline_fps": round(job.pipeline_fps, 2),
            }
            logger.info("Job %s completed successfully in %.2fs (%.2f FPS)", job.job_id, total_time, job.pipeline_fps)

        except Exception as e:
            logger.exception("Error executing pipeline job %s: %s", job.job_id, e)
            job.status = "FAILED"
            job.error_message = str(e)

        finally:
            if reader:
                reader.release()
            if video_writer:
                video_writer.release()
                video_writer = None

            # Convert annotated video to standard H.264 for universal browser playback
            if job.output_video_path.exists() and job.output_video_path.stat().st_size > 0:
                try:
                    import imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    temp_output = job.output_video_path.with_name(f"{job.output_video_path.stem}_browser.mp4")
                    cmd = [
                        ffmpeg_exe, "-y", "-i", str(job.output_video_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(temp_output)
                    ]
                    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                    if res.returncode == 0 and temp_output.exists() and temp_output.stat().st_size > 0:
                        temp_output.replace(job.output_video_path)
                        logger.info("Successfully converted %s to H.264 for web playback", job.output_video_path.name)
                except Exception as ffmpeg_err:
                    logger.debug("H.264 conversion step skipped or failed: %s", ffmpeg_err)
