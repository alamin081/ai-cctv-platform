"""
Data models and exceptions for Multi-Object Tracking and Track State Management.
Phase: Phase 4 & Phase 5 - ByteTrack and Track State Manager
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TrackStateStr(str):
    """Case-insensitive string representation for track lifecycle states."""

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.upper() == other.upper()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.upper())


class TrackState(str, Enum):
    """Lifecycle state of an individual track."""
    NEW = TrackStateStr("NEW")
    ACTIVE = TrackStateStr("ACTIVE")
    LOST = TrackStateStr("LOST")
    REMOVED = TrackStateStr("REMOVED")


class TrackingError(Exception):
    """Base exception for tracking module errors."""
    pass


class InvalidTrackerInputError(TrackingError):
    """Raised when tracker receives malformed or invalid detections."""
    pass


class TrackerConfigError(TrackingError):
    """Raised when tracker configuration parameters are invalid."""
    pass


class InvalidStateManagerInputError(TrackingError):
    """Raised when TrackStateManager receives malformed or invalid input."""
    pass


@dataclass
class Track:
    """Represents a tracked person across video frames with temporal lifecycle state."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    frame_number: Optional[int] = None
    timestamp: Optional[float] = None
    state: str = "ACTIVE"
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    # Phase 5 temporal metadata
    first_seen_frame: int = 1
    last_seen_frame: int = 1
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    active_frames: int = 1
    missed_frames: int = 0
    age_frames: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.state, TrackStateStr):
            self.state = TrackStateStr(str(self.state).upper())

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0

    def as_int_bbox(self) -> Tuple[int, int, int, int]:
        """Returns integer bounding box coordinates (x1, y1, x2, y2)."""
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def to_dict(self) -> Dict[str, Any]:
        """Convert track to JSON-serializable dictionary matching Phase 5 schema."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "state": str(self.state).upper(),
            "confidence": round(self.confidence, 4),
            "bbox": [round(c, 2) for c in self.bbox],
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "first_seen_timestamp": round(self.first_seen_timestamp, 4),
            "last_seen_timestamp": round(self.last_seen_timestamp, 4),
            "active_frames": self.active_frames,
            "missed_frames": self.missed_frames,
            "age_frames": self.age_frames,
        }


@dataclass
class TrackingResult:
    """Represents the tracking output for a single video frame from ByteTrack."""
    frame_number: Optional[int]
    timestamp: Optional[float]
    detections: int
    tracks: List[Track]
    tracking_time_ms: float = 0.0

    @property
    def count(self) -> int:
        """Total number of active tracks in this frame."""
        return len(self.tracks)

    @property
    def active_track_ids(self) -> List[int]:
        """List of active track IDs in this frame."""
        return [t.track_id for t in self.tracks]

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracking result to JSON-serializable dictionary."""
        return {
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
            "detections": self.detections,
            "tracks": [t.to_dict() for t in self.tracks],
        }


@dataclass
class TrackStateResult:
    """Represents the structured state output for a single video frame from TrackStateManager."""
    frame_number: int
    timestamp: float
    tracks: List[Track]
    active_count: int = 0
    lost_count: int = 0
    removed_count: int = 0
    state_processing_time_ms: float = 0.0

    @property
    def count(self) -> int:
        """Number of active visible tracks."""
        return len(self.tracks)

    @property
    def active_track_ids(self) -> List[int]:
        """List of active track IDs in this frame."""
        return [t.track_id for t in self.tracks]

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracking state result to JSON-serializable dictionary matching Section 13 schema."""
        return {
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
            "tracks": [t.to_dict() for t in self.tracks],
        }
