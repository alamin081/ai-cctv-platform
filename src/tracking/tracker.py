"""
ByteTrack-based Multi-Object Person Tracker.
Phase: Phase 4 - ByteTrack Multi-Object Tracking
"""

import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from src.detection.models import Detection, DetectionResult
from .models import (
    InvalidTrackerInputError,
    Track,
    TrackerConfigError,
    TrackingResult,
    TrackState,
)

logger = logging.getLogger(__name__)

# COCO dataset class index for 'person'
TARGET_PERSON_CLASS_ID = 0


class _DetectionsAdapter:
    """Internal adapter to present structured detections to Ultralytics BYTETracker."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __getitem__(self, idx: Any) -> "_DetectionsAdapter":
        return _DetectionsAdapter(self.xywh[idx], self.conf[idx], self.cls[idx])

    def __len__(self) -> int:
        return len(self.conf)


class ByteTrackPersonTracker:
    """
    Multi-Object Tracker utilizing the ByteTrack association algorithm.
    Specialized strictly for tracking persons across consecutive video frames.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
    ) -> None:
        """
        Initialize the ByteTrack tracker with validated parameters.

        Args:
            track_high_thresh: Threshold for first association stage (high-confidence).
            track_low_thresh: Threshold for second association stage (low-confidence).
            new_track_thresh: Minimum detection score to initialize a new track candidate.
            track_buffer: Maximum frames to retain lost tracks before removing.
            match_thresh: IoU matching threshold for Kalman-to-detection assignment.
            fuse_score: Whether to fuse detection score into IoU similarity metric.
        """
        if track_high_thresh <= track_low_thresh:
            raise TrackerConfigError(
                f"track_high_thresh ({track_high_thresh}) must be greater than track_low_thresh ({track_low_thresh})"
            )

        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.fuse_score = bool(fuse_score)

        # Build ByteTrack args namespace
        self._args = SimpleNamespace(
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            fuse_score=self.fuse_score,
            gmc_method="none",
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,
        )

        # Initialize core ByteTracker instance
        self._tracker = BYTETracker(self._args)

        # Track history & statistics
        self._track_history: Dict[int, Dict[str, Any]] = {}
        self._unique_track_ids: Set[int] = set()
        self._frames_processed: int = 0
        self._frames_with_tracks: int = 0
        self._total_track_instances: int = 0
        self._max_active_tracks: int = 0
        self._active_tracks_sum: int = 0

        logger.info(
            "ByteTrackPersonTracker initialized [high_thresh=%.2f, low_thresh=%.2f, buffer=%d, match_thresh=%.2f]",
            self.track_high_thresh,
            self.track_low_thresh,
            self.track_buffer,
            self.match_thresh,
        )

    @property
    def unique_track_ids(self) -> Set[int]:
        """Set of all unique track IDs observed over the tracker lifetime."""
        return set(self._unique_track_ids)

    @property
    def total_frames_processed(self) -> int:
        return self._frames_processed

    @property
    def frames_with_tracks(self) -> int:
        return self._frames_with_tracks

    @property
    def total_track_instances(self) -> int:
        return self._total_track_instances

    @property
    def max_simultaneous_tracks(self) -> int:
        return self._max_active_tracks

    @property
    def average_active_tracks(self) -> float:
        return (
            (self._active_tracks_sum / self._frames_processed)
            if self._frames_processed > 0
            else 0.0
        )

    def reset(self) -> None:
        """Reset internal tracker state and all aggregated metrics."""
        self._tracker = BYTETracker(self._args)
        self._track_history.clear()
        self._unique_track_ids.clear()
        self._frames_processed = 0
        self._frames_with_tracks = 0
        self._total_track_instances = 0
        self._max_active_tracks = 0
        self._active_tracks_sum = 0
        logger.info("ByteTrackPersonTracker state reset.")

    def update(
        self,
        detections: Union[DetectionResult, List[Detection], Any],
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> TrackingResult:
        """
        Update tracker with detections for the current frame.

        Args:
            detections: DetectionResult from Phase 3 or list of Detection objects.
            frame_number: Optional frame sequence index (inferred from DetectionResult if present).
            timestamp: Optional frame timestamp in seconds (inferred from DetectionResult if present).
            frame_bgr: Optional image frame matrix.

        Returns:
            TrackingResult: Container with active tracks and performance metrics.

        Raises:
            InvalidTrackerInputError: If detections object is malformed or invalid.
        """
        start_time = time.perf_counter()

        # Extract frame index and timestamp if available
        if isinstance(detections, DetectionResult):
            fn = detections.frame_number if detections.frame_number is not None else frame_number
            ts = detections.timestamp if detections.timestamp is not None else timestamp
            det_list = detections.detections
            det_count = detections.count
        elif isinstance(detections, list):
            fn = frame_number
            ts = timestamp
            det_list = detections
            det_count = len(detections)
        elif detections is None:
            fn = frame_number
            ts = timestamp
            det_list = []
            det_count = 0
        else:
            raise InvalidTrackerInputError(
                f"Unsupported detections input type: {type(detections).__name__}. Expected DetectionResult or list of Detection."
            )

        # Filter strictly to person class (COCO 0) and prepare xywh format
        person_xywh: List[List[float]] = []
        person_conf: List[float] = []
        person_cls: List[int] = []

        for d in det_list:
            if not isinstance(d, Detection):
                # Fallback dictionary or duck-typed object check
                cls_id = getattr(d, "class_id", None) if not isinstance(d, dict) else d.get("class_id")
                bbox = getattr(d, "bbox", None) if not isinstance(d, dict) else d.get("bbox")
                conf = getattr(d, "confidence", None) if not isinstance(d, dict) else d.get("confidence")
            else:
                cls_id = d.class_id
                bbox = d.bbox
                conf = d.confidence

            # Strictly person class
            if cls_id != TARGET_PERSON_CLASS_ID:
                continue

            if bbox is None or len(bbox) != 4:
                raise InvalidTrackerInputError(f"Malformed bounding box: {bbox}")

            x1, y1, x2, y2 = bbox
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            if w <= 0 or h <= 0:
                continue

            cx = float(x1) + (w / 2.0)
            cy = float(y1) + (h / 2.0)

            person_xywh.append([cx, cy, w, h])
            person_conf.append(float(conf))
            person_cls.append(int(cls_id))

        # Build ByteTrack-compatible adapter
        adapter = _DetectionsAdapter(
            xywh=np.array(person_xywh, dtype=np.float32) if person_xywh else np.empty((0, 4), dtype=np.float32),
            conf=np.array(person_conf, dtype=np.float32) if person_conf else np.empty((0,), dtype=np.float32),
            cls=np.array(person_cls, dtype=np.float32) if person_cls else np.empty((0,), dtype=np.float32),
        )

        # Run ByteTrack update step
        raw_tracks = self._tracker.update(adapter, img=frame_bgr)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Parse active tracks
        current_tracks: List[Track] = []
        if len(raw_tracks) > 0:
            for row in raw_tracks:
                # row format: [x1, y1, x2, y2, track_id, score, cls, idx]
                x1 = float(row[0])
                y1 = float(row[1])
                x2 = float(row[2])
                y2 = float(row[3])
                track_id = int(row[4])
                score = float(row[5])
                cls_id = int(row[6])

                # Update track history
                if track_id not in self._track_history:
                    self._track_history[track_id] = {
                        "first_frame": fn,
                        "hits": 1,
                        "age": 1,
                    }
                else:
                    self._track_history[track_id]["hits"] += 1
                    self._track_history[track_id]["age"] += 1

                hist = self._track_history[track_id]

                track_obj = Track(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name="person",
                    confidence=score,
                    bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    frame_number=fn,
                    timestamp=ts,
                    state=TrackState.ACTIVE.value,
                    age=hist["age"],
                    hits=hist["hits"],
                    time_since_update=0,
                )
                current_tracks.append(track_obj)
                self._unique_track_ids.add(track_id)

        # Update lifetime statistics
        self._frames_processed += 1
        active_count = len(current_tracks)
        if active_count > 0:
            self._frames_with_tracks += 1
            self._total_track_instances += active_count

        self._active_tracks_sum += active_count
        if active_count > self._max_active_tracks:
            self._max_active_tracks = active_count

        return TrackingResult(
            frame_number=fn,
            timestamp=ts,
            detections=det_count,
            tracks=current_tracks,
            tracking_time_ms=elapsed_ms,
        )

    @staticmethod
    def draw_tracks(
        frame_bgr: np.ndarray,
        tracks: List[Track],
        color: Tuple[int, int, int] = (0, 255, 128),
    ) -> np.ndarray:
        """
        Draw clean bounding boxes and Track ID labels on an image frame.

        Args:
            frame_bgr: Source OpenCV image matrix.
            tracks: List of active Track objects to draw.
            color: BGR box color (default: bright surveillance green).

        Returns:
            np.ndarray: Annotated frame matrix.
        """
        output = frame_bgr.copy()
        for t in tracks:
            x1, y1, x2, y2 = t.as_int_bbox()

            # Draw person bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label banner
            label = f"ID: {t.track_id} | PERSON {t.confidence:.2f}"
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
