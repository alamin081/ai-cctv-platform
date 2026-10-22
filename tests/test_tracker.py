"""
Unit tests for ByteTrackPersonTracker and tracking data models.
Phase: Phase 4 - ByteTrack Multi-Object Tracking
"""

from pathlib import Path
from typing import List
import pytest

from src.detection import Detection, DetectionResult, YOLOPersonDetector
from src.tracking import (
    ByteTrackPersonTracker,
    InvalidTrackerInputError,
    Track,
    TrackerConfigError,
    TrackingResult,
)
from src.video_input import VideoFileReader


@pytest.fixture
def tracker() -> ByteTrackPersonTracker:
    """Fixture initializing a fresh ByteTrackPersonTracker."""
    return ByteTrackPersonTracker(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )


def test_tracker_initialization():
    """Test 1: Tracker initializes correctly with validated parameters."""
    t = ByteTrackPersonTracker(
        track_high_thresh=0.6,
        track_low_thresh=0.2,
        new_track_thresh=0.7,
        track_buffer=25,
        match_thresh=0.75,
    )
    assert t.track_high_thresh == 0.6
    assert t.track_low_thresh == 0.2
    assert t.track_buffer == 25
    assert len(t.unique_track_ids) == 0

    # Invalid thresholds where high <= low
    with pytest.raises(TrackerConfigError):
        ByteTrackPersonTracker(track_high_thresh=0.2, track_low_thresh=0.5)


def test_empty_detection_handling(tracker: ByteTrackPersonTracker):
    """Test 2: Tracker does not crash when passed empty detections."""
    result = tracker.update([], frame_number=1, timestamp=0.03)
    assert isinstance(result, TrackingResult)
    assert result.count == 0
    assert result.tracks == []
    assert result.detections == 0

    # Passing empty DetectionResult
    empty_det_res = DetectionResult(frame_number=2, timestamp=0.06, detections=[], inference_time_ms=5.0)
    result2 = tracker.update(empty_det_res)
    assert result2.count == 0
    assert result2.tracks == []


def test_person_filtering(tracker: ByteTrackPersonTracker):
    """Test 3: Non-person classes (class_id != 0) are strictly ignored."""
    mixed_detections = [
        Detection(class_id=2, class_name="car", confidence=0.95, bbox=[100.0, 100.0, 250.0, 200.0]),
        Detection(class_id=56, class_name="chair", confidence=0.88, bbox=[300.0, 200.0, 350.0, 260.0]),
        Detection(class_id=0, class_name="person", confidence=0.92, bbox=[50.0, 50.0, 120.0, 200.0]),
    ]
    result = tracker.update(mixed_detections, frame_number=1, timestamp=0.03)
    assert result.count == 1
    assert result.tracks[0].class_id == 0
    assert result.tracks[0].class_name == "person"


def test_track_id_creation(tracker: ByteTrackPersonTracker):
    """Test 4: Valid person detection produces a valid track with positive integer ID."""
    det = [Detection(class_id=0, class_name="person", confidence=0.90, bbox=[100.0, 100.0, 180.0, 300.0])]
    result = tracker.update(det, frame_number=1, timestamp=0.03)

    assert result.count == 1
    track = result.tracks[0]
    assert isinstance(track.track_id, int)
    assert track.track_id > 0
    assert track.confidence > 0.0
    assert track.state == "active"


def test_track_persistence_across_frames(tracker: ByteTrackPersonTracker):
    """Test 5: Consecutive frames maintain the same persistent track ID for a moving person."""
    # Person moving slightly to the right across 5 consecutive frames
    boxes = [
        [100.0, 120.0, 170.0, 300.0],
        [104.0, 120.0, 174.0, 300.0],
        [108.0, 121.0, 178.0, 301.0],
        [112.0, 121.0, 182.0, 301.0],
        [116.0, 122.0, 186.0, 302.0],
    ]

    assigned_ids = []
    for i, bbox in enumerate(boxes):
        det = [Detection(class_id=0, class_name="person", confidence=0.92, bbox=bbox)]
        result = tracker.update(det, frame_number=i + 1, timestamp=(i + 1) * 0.033)
        assert result.count == 1
        assigned_ids.append(result.tracks[0].track_id)

    # All frames must share the exact same track ID
    initial_id = assigned_ids[0]
    assert all(tid == initial_id for tid in assigned_ids), f"Track ID changed unexpectedly: {assigned_ids}"


def test_multiple_persons_distinct_ids(tracker: ByteTrackPersonTracker):
    """Test 6: Distinct persons in the same frame receive separate track IDs."""
    det_multi = [
        Detection(class_id=0, class_name="person", confidence=0.91, bbox=[50.0, 100.0, 120.0, 280.0]),
        Detection(class_id=0, class_name="person", confidence=0.89, bbox=[350.0, 110.0, 420.0, 290.0]),
    ]
    result = tracker.update(det_multi, frame_number=1, timestamp=0.033)
    assert result.count == 2

    id_1 = result.tracks[0].track_id
    id_2 = result.tracks[1].track_id
    assert id_1 != id_2, f"Expected distinct IDs for two separate persons, got {id_1} and {id_2}"


def test_detection_loss_graceful_handling(tracker: ByteTrackPersonTracker):
    """Test 7: Tracker handles temporary detection loss without crashing and maintains state."""
    # Frame 1: Person visible
    det_f1 = [Detection(class_id=0, class_name="person", confidence=0.92, bbox=[100.0, 100.0, 160.0, 280.0])]
    res_f1 = tracker.update(det_f1, frame_number=1, timestamp=0.03)
    assert res_f1.count == 1
    original_id = res_f1.tracks[0].track_id

    # Frame 2: Occlusion / detection miss (empty detections)
    res_f2 = tracker.update([], frame_number=2, timestamp=0.06)
    assert res_f2.count == 0  # No active detections reported

    # Frame 3: Person reappears nearby
    det_f3 = [Detection(class_id=0, class_name="person", confidence=0.91, bbox=[106.0, 101.0, 166.0, 281.0])]
    res_f3 = tracker.update(det_f3, frame_number=3, timestamp=0.09)

    # Reappearance should be handled gracefully (active track produced)
    assert res_f3.count == 1
    assert res_f3.tracks[0].track_id == original_id


def test_invalid_tracker_input(tracker: ByteTrackPersonTracker):
    """Test 8: Malformed detections trigger InvalidTrackerInputError."""
    with pytest.raises(InvalidTrackerInputError):
        tracker.update("invalid_string_input")

    malformed_bbox = [Detection(class_id=0, class_name="person", confidence=0.9, bbox=[10.0, 20.0])]  # only 2 elements
    with pytest.raises(InvalidTrackerInputError):
        tracker.update(malformed_bbox)


def test_track_output_structure(tracker: ByteTrackPersonTracker):
    """Test 9: Verify output schema and dictionary serialization."""
    det = [Detection(class_id=0, class_name="person", confidence=0.9324, bbox=[152.4, 150.0, 312.8, 390.0])]
    result = tracker.update(det, frame_number=30, timestamp=1.0)

    res_dict = result.to_dict()
    assert res_dict["frame_number"] == 30
    assert res_dict["timestamp"] == 1.0
    assert res_dict["detections"] == 1
    assert len(res_dict["tracks"]) == 1

    t_dict = res_dict["tracks"][0]
    assert "track_id" in t_dict
    assert t_dict["class_id"] == 0
    assert t_dict["class_name"] == "person"
    assert t_dict["confidence"] == 0.9324
    assert t_dict["bbox"] == [152.4, 150.0, 312.8, 390.0]


def test_video_integration_pipeline():
    """Test 10: End-to-end integration test (VideoFileReader -> YOLOPersonDetector -> ByteTrack)."""
    video_path = Path(__file__).resolve().parent.parent / "data" / "videos" / "test.mp4"
    assert video_path.exists(), f"Test video must exist at {video_path}"

    reader = VideoFileReader(video_path)
    detector = YOLOPersonDetector("data/models/yolov8n.pt", confidence_threshold=0.35, device="auto")
    tracker = ByteTrackPersonTracker()

    tracked_frames = 0
    all_observed_ids = set()

    with reader:
        # Run on first 25 frames
        for i, frame in enumerate(reader):
            if i >= 25:
                break
            det_res = detector.detect(frame)
            track_res = tracker.update(det_res)
            tracked_frames += 1
            all_observed_ids.update(track_res.active_track_ids)

    assert tracked_frames == 25
    assert len(all_observed_ids) >= 1
    assert tracker.frames_with_tracks > 0
