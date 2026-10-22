"""
Track State Manager for maintaining temporal lifecycle states of tracked persons.
Phase: Phase 5 - Track State Manager
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .models import (
    InvalidStateManagerInputError,
    Track,
    TrackState,
    TrackStateResult,
    TrackingResult,
)

logger = logging.getLogger(__name__)

TARGET_PERSON_CLASS_ID = 0


class TrackStateManager:
    """
    Manages the temporal lifecycle and state transitions for persistent track IDs.
    Transitions tracks across NEW -> ACTIVE -> LOST -> REMOVED.
    """

    def __init__(self, max_missed_frames: int = 30) -> None:
        """
        Initialize the TrackStateManager.

        Args:
            max_missed_frames: Number of consecutive missed frames before marking track as REMOVED.
        """
        self.max_missed_frames = max(1, int(max_missed_frames))
        self._tracks: Dict[int, Track] = {}
        self._removed_tracks: Dict[int, Track] = {}
        self._total_tracks_created: int = 0
        self._max_concurrent_tracks: int = 0

        logger.info(
            "TrackStateManager initialized (max_missed_frames=%d)",
            self.max_missed_frames,
        )

    @property
    def total_tracks_created(self) -> int:
        """Total number of unique tracks registered over the session."""
        return self._total_tracks_created

    @property
    def currently_active_tracks(self) -> int:
        """Count of tracks currently in NEW or ACTIVE state."""
        return sum(
            1 for t in self._tracks.values() if t.state in (TrackState.NEW.value, TrackState.ACTIVE.value)
        )

    @property
    def currently_lost_tracks(self) -> int:
        """Count of tracks currently in LOST state."""
        return sum(1 for t in self._tracks.values() if t.state == TrackState.LOST.value)

    @property
    def currently_removed_tracks(self) -> int:
        """Count of tracks currently in REMOVED state."""
        return len(self._removed_tracks)

    @property
    def max_concurrent_tracks(self) -> int:
        """Maximum simultaneous active tracks observed in a single frame."""
        return self._max_concurrent_tracks

    @property
    def active_tracks(self) -> List[Track]:
        """List of tracks currently in NEW or ACTIVE state."""
        return [
            t for t in self._tracks.values() if t.state in (TrackState.NEW.value, TrackState.ACTIVE.value)
        ]

    @property
    def lost_tracks(self) -> List[Track]:
        """List of tracks currently in LOST state."""
        return [t for t in self._tracks.values() if t.state == TrackState.LOST.value]

    @property
    def removed_tracks(self) -> List[Track]:
        """List of tracks that transitioned to REMOVED state."""
        return list(self._removed_tracks.values())

    @property
    def all_tracks(self) -> Dict[int, Track]:
        """Union mapping of active, lost, and removed tracks."""
        merged = dict(self._tracks)
        merged.update(self._removed_tracks)
        return merged

    @property
    def average_track_age(self) -> float:
        """Average age in frames across all registered tracks."""
        all_trks = self.all_tracks.values()
        if not all_trks:
            return 0.0
        return sum(t.age_frames for t in all_trks) / len(all_trks)

    @property
    def average_active_frames(self) -> float:
        """Average number of actively tracked frames across all registered tracks."""
        all_trks = self.all_tracks.values()
        if not all_trks:
            return 0.0
        return sum(t.active_frames for t in all_trks) / len(all_trks)

    @property
    def total_track_lifetime(self) -> int:
        """Sum of age_frames across all registered tracks."""
        return sum(t.age_frames for t in self.all_tracks.values())

    def get_track(self, track_id: int) -> Optional[Track]:
        """Query a track by track_id from active or removed tracks."""
        if track_id in self._tracks:
            return self._tracks[track_id]
        return self._removed_tracks.get(track_id)

    def reset(self) -> None:
        """Reset internal state and metrics."""
        self._tracks.clear()
        self._removed_tracks.clear()
        self._total_tracks_created = 0
        self._max_concurrent_tracks = 0
        logger.info("TrackStateManager reset completed.")

    def update(
        self,
        tracks: Union[TrackingResult, List[Track], Any],
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> TrackStateResult:
        """
        Update temporal lifecycle state for all tracks based on the current frame.

        Args:
            tracks: TrackingResult from ByteTrack, or list of Track objects.
            frame_number: Sequence index of the current frame.
            timestamp: Playback timestamp of the current frame in seconds.

        Returns:
            TrackStateResult: Container with currently active/visible tracks and state counts.

        Raises:
            InvalidStateManagerInputError: If input format is invalid.
        """
        start_time = time.perf_counter()

        # Parse and validate incoming tracks
        if isinstance(tracks, TrackingResult):
            fn = tracks.frame_number if tracks.frame_number is not None else frame_number
            ts = tracks.timestamp if tracks.timestamp is not None else timestamp
            incoming_list = tracks.tracks
        elif isinstance(tracks, list):
            fn = frame_number
            ts = timestamp
            incoming_list = tracks
        elif tracks is None:
            fn = frame_number
            ts = timestamp
            incoming_list = []
        else:
            raise InvalidStateManagerInputError(
                f"Unsupported tracks input type: {type(tracks).__name__}. Expected TrackingResult or list of Track."
            )

        fn = int(fn) if fn is not None else 1
        ts = float(ts) if ts is not None else 0.0

        seen_ids = set()
        active_in_frame: List[Track] = []

        # 1. Process visible incoming tracks
        for incoming in incoming_list:
            if not hasattr(incoming, "track_id"):
                raise InvalidStateManagerInputError(
                    f"Track object missing 'track_id': {incoming}"
                )

            # Strictly filter to person class (COCO 0)
            cls_id = getattr(incoming, "class_id", TARGET_PERSON_CLASS_ID)
            if cls_id != TARGET_PERSON_CLASS_ID:
                continue

            tid = int(incoming.track_id)
            seen_ids.add(tid)
            bbox = list(incoming.bbox)
            conf = float(incoming.confidence)

            if tid not in self._tracks:
                # Brand new track: State = NEW
                self._total_tracks_created += 1
                managed_track = Track(
                    track_id=tid,
                    class_id=cls_id,
                    class_name="person",
                    confidence=conf,
                    bbox=bbox,
                    frame_number=fn,
                    timestamp=ts,
                    state=TrackState.NEW.value,
                    age=1,
                    hits=1,
                    time_since_update=0,
                    first_seen_frame=fn,
                    last_seen_frame=fn,
                    first_seen_timestamp=ts,
                    last_seen_timestamp=ts,
                    active_frames=1,
                    missed_frames=0,
                    age_frames=1,
                )
                self._tracks[tid] = managed_track
                active_in_frame.append(managed_track)
                logger.info("Track %d registered (state: NEW) at frame %d", tid, fn)

            else:
                # Existing track: State = ACTIVE
                existing = self._tracks[tid]
                old_state = existing.state

                existing.state = TrackState.ACTIVE.value
                existing.last_seen_frame = fn
                existing.last_seen_timestamp = ts
                existing.active_frames += 1
                existing.hits += 1
                existing.missed_frames = 0
                existing.time_since_update = 0
                existing.age_frames = fn - existing.first_seen_frame + 1
                existing.age = existing.age_frames
                existing.bbox = bbox
                existing.confidence = conf
                existing.frame_number = fn
                existing.timestamp = ts

                active_in_frame.append(existing)

                # Log meaningful state transition
                if old_state != TrackState.ACTIVE.value:
                    if old_state == TrackState.LOST.value:
                        logger.info("Track %d recovered and became ACTIVE again at frame %d", tid, fn)
                    elif old_state == TrackState.NEW.value:
                        logger.info("Track %d confirmed (state: ACTIVE) at frame %d", tid, fn)

        # 2. Process tracks missing in this frame
        to_remove = []
        for tid, trk in self._tracks.items():
            if tid not in seen_ids:
                trk.missed_frames += 1
                trk.time_since_update += 1
                trk.age_frames = fn - trk.first_seen_frame + 1
                trk.age = trk.age_frames
                old_state = trk.state

                if trk.missed_frames >= self.max_missed_frames:
                    trk.state = TrackState.REMOVED.value
                    to_remove.append(tid)
                    logger.info(
                        "Track %d REMOVED after exceeding %d missed frames at frame %d",
                        tid,
                        self.max_missed_frames,
                        fn,
                    )
                else:
                    trk.state = TrackState.LOST.value
                    if old_state != TrackState.LOST.value:
                        logger.info(
                            "Track %d became LOST (missed %d frames) at frame %d",
                            tid,
                            trk.missed_frames,
                            fn,
                        )

        # Move removed tracks to history archive
        for tid in to_remove:
            self._removed_tracks[tid] = self._tracks.pop(tid)

        # Update peak concurrent count
        current_active = len(active_in_frame)
        if current_active > self._max_concurrent_tracks:
            self._max_concurrent_tracks = current_active

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return TrackStateResult(
            frame_number=fn,
            timestamp=ts,
            tracks=active_in_frame,
            active_count=current_active,
            lost_count=self.currently_lost_tracks,
            removed_count=self.currently_removed_tracks,
            state_processing_time_ms=elapsed_ms,
        )

    @staticmethod
    def draw_state_tracks(
        frame_bgr: np.ndarray,
        tracks: List[Track],
    ) -> np.ndarray:
        """
        Draw clean bounding boxes and Track ID + State labels on an image frame.

        Args:
            frame_bgr: Source OpenCV image matrix.
            tracks: List of active Track objects to draw.

        Returns:
            np.ndarray: Annotated frame matrix.
        """
        output = frame_bgr.copy()
        state_colors = {
            TrackState.NEW.value: (255, 200, 0),     # Cyan / Sky Blue
            TrackState.ACTIVE.value: (0, 255, 128),  # Bright Surveillance Green
            TrackState.LOST.value: (0, 165, 255),    # Amber / Orange
            TrackState.REMOVED.value: (80, 80, 80),  # Gray
        }

        for t in tracks:
            x1, y1, x2, y2 = t.as_int_bbox()
            color = state_colors.get(str(t.state).upper(), (0, 255, 128))

            # Draw person bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label banner: "ID: 1 | ACTIVE"
            label = f"ID: {t.track_id} | {str(t.state).upper()}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            banner_y1 = max(0, y1 - text_h - 8)
            banner_y2 = y1
            banner_x2 = min(output.shape[1], x1 + text_w + 8)

            cv2.rectangle(output, (x1, banner_y1), (banner_x2, banner_y2), color, -1)
            cv2.putText(
                output,
                label,
                (x1 + 4, banner_y2 - 4),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        return output
