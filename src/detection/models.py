"""
Detection models, data containers, and exceptions.
Phase: Phase 3 - YOLO Person Detection
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class DetectionError(Exception):
    """Base exception for detection module."""
    pass


class ModelLoadError(DetectionError):
    """Raised when the detection model fails to load."""
    pass


class InferenceError(DetectionError):
    """Raised when an error occurs during model inference."""
    pass


class InvalidFrameError(DetectionError):
    """Raised when an invalid, empty, or incompatible frame is supplied."""
    pass


@dataclass
class Detection:
    """Represents a single detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

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
        """Convert detection to JSON-serializable dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(c, 2) for c in self.bbox],
        }


@dataclass
class DetectionResult:
    """Represents the detection output for a single video frame."""
    frame_number: Optional[int]
    timestamp: Optional[float]
    detections: List[Detection]
    inference_time_ms: float

    @property
    def count(self) -> int:
        """Total number of detected objects."""
        return len(self.detections)

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame detection result to JSON-serializable dictionary."""
        return {
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
            "count": self.count,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "detections": [d.to_dict() for d in self.detections],
        }
