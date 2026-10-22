"""
Unit tests for YOLOPersonDetector and detection data structures.
Phase: Phase 3 - YOLO Person Detection
"""

import cv2
import numpy as np
import pytest
from ultralytics.utils import ASSETS

from src.detection import (
    COCO_PERSON_CLASS_ID,
    Detection,
    DetectionResult,
    InvalidFrameError,
    ModelLoadError,
    YOLOPersonDetector,
)
from src.video_input import Frame


@pytest.fixture(scope="module")
def detector() -> YOLOPersonDetector:
    """Fixture initializing a reusable lightweight YOLOPersonDetector."""
    return YOLOPersonDetector(
        model_name_or_path="data/models/yolov8n.pt",
        confidence_threshold=0.35,
        device="auto",
    )


@pytest.fixture(scope="module")
def sample_person_image() -> np.ndarray:
    """Fixture providing a verified real image containing persons."""
    img_path = ASSETS / "zidane.jpg"
    assert img_path.exists(), f"Sample image must exist at {img_path}"
    img = cv2.imread(str(img_path))
    assert img is not None and isinstance(img, np.ndarray)
    return img


def test_model_initialization(detector: YOLOPersonDetector):
    """Test detector initialization, model loading, and device selection."""
    assert detector.model is not None
    assert detector.target_classes == [COCO_PERSON_CLASS_ID]
    assert detector.confidence_threshold == 0.35
    assert detector.device in ["cuda:0", "cuda", "cpu"]
    assert "CUDA" in detector.device_display_name or "CPU" in detector.device_display_name


def test_invalid_model_path():
    """Test that specifying an invalid model path raises ModelLoadError."""
    with pytest.raises(ModelLoadError):
        YOLOPersonDetector(model_name_or_path="non_existent_weights_xyz.pt")


def test_empty_and_invalid_frame_handling(detector: YOLOPersonDetector):
    """Test detector rejection of None, invalid type, and empty frames."""
    # None input
    with pytest.raises(InvalidFrameError) as exc_none:
        detector.detect(None)
    assert "Supplied frame is None" in str(exc_none.value)

    # Invalid type
    with pytest.raises(InvalidFrameError) as exc_type:
        detector.detect("not_an_image")
    assert "Expected numpy.ndarray" in str(exc_type.value)

    # Empty 0-size array
    with pytest.raises(InvalidFrameError) as exc_empty:
        detector.detect(np.empty((0, 0, 3), dtype=np.uint8))
    assert "invalid shape" in str(exc_empty.value)


def test_detection_result_structure(detector: YOLOPersonDetector):
    """Test DetectionResult container and dictionary serialization on blank image."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(blank_frame)

    assert isinstance(result, DetectionResult)
    assert result.frame_number is None
    assert result.timestamp is None
    assert isinstance(result.detections, list)
    assert result.count == len(result.detections)
    assert result.inference_time_ms > 0.0

    d_dict = result.to_dict()
    assert "detections" in d_dict
    assert "count" in d_dict
    assert "inference_time_ms" in d_dict


def test_person_detection_format_and_confidence(
    detector: YOLOPersonDetector, sample_person_image: np.ndarray
):
    """Test detection on a real image and verify person class, bbox, and confidence."""
    result = detector.detect(sample_person_image)

    assert result.count > 0, "Expected at least one detected person in sample image"

    for det in result.detections:
        assert isinstance(det, Detection)
        # Verify strict person class filtering
        assert det.class_id == COCO_PERSON_CLASS_ID
        assert det.class_name == "person"

        # Verify confidence range
        assert 0.0 <= det.confidence <= 1.0
        assert det.confidence >= detector.confidence_threshold

        # Verify bounding box format [x1, y1, x2, y2]
        assert len(det.bbox) == 4
        x1, y1, x2, y2 = det.bbox
        assert x1 < x2, f"Expected x1 < x2, got {x1} >= {x2}"
        assert y1 < y2, f"Expected y1 < y2, got {y1} >= {y2}"
        assert det.width > 0
        assert det.height > 0
        assert det.area > 0

        # Integer bbox
        ix1, iy1, ix2, iy2 = det.as_int_bbox()
        assert isinstance(ix1, int) and isinstance(iy1, int)

        # Dictionary format
        det_dict = det.to_dict()
        assert det_dict["class_id"] == 0
        assert det_dict["class_name"] == "person"
        assert isinstance(det_dict["confidence"], float)
        assert len(det_dict["bbox"]) == 4


def test_frame_object_compatibility(detector: YOLOPersonDetector, sample_person_image: np.ndarray):
    """Test detector accepts a Frame object from video_input module and preserves metadata."""
    frame_obj = Frame(frame_number=105, timestamp=3.5, data=sample_person_image)
    result = detector.detect(frame_obj)

    assert result.frame_number == 105
    assert result.timestamp == 3.5
    assert result.count > 0


def test_device_selection_cpu():
    """Test explicitly forcing CPU device."""
    cpu_detector = YOLOPersonDetector(
        model_name_or_path="data/models/yolov8n.pt",
        device="cpu",
    )
    assert cpu_detector.device == "cpu"
    assert "CPU" in cpu_detector.device_display_name
