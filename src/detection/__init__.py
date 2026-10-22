"""
Object detection module.
Specialized for lightweight YOLO-based person detection.
"""

from .detector import COCO_PERSON_CLASS_ID, YOLOPersonDetector
from .models import (
    Detection,
    DetectionError,
    DetectionResult,
    InferenceError,
    InvalidFrameError,
    ModelLoadError,
)

__all__ = [
    "YOLOPersonDetector",
    "Detection",
    "DetectionResult",
    "DetectionError",
    "ModelLoadError",
    "InferenceError",
    "InvalidFrameError",
    "COCO_PERSON_CLASS_ID",
]
