"""
Unit tests for TrackStateManager and temporal track lifecycle management.
Phase: Phase 5 - Track State Manager
"""

from pathlib import Path
import pytest

from src.detection import Detection, YOLOPersonDetector
from src.tracking import (
    ByteTrackPersonTracker,
    InvalidStateManagerInputError,
    Track,
    TrackState,
    TrackStateManager,
    TrackStateResult,
    TrackingResult,
)
from src.video_input import VideoFileReader


@pytest.fixture
def state_manager() -> TrackStateManager:
    """Fixture providing a clean TrackStateManager instance."""
    return TrackStateManager(max_missed_frames=5)


def test_state_manager_initialization():
    """Test 1: TrackStateManager initializes correctly with custom parameters."""
    sm = TrackStateManager(max_missed_frames=10)
    assert sm.max_missed_frames == 10
    assert sm.total_tracks_created == 0
    assert sm.currently_active_tracks == 0
    assert sm.currently_lost_tracks == 0
    assert sm.currently_removed_tracks == 0
    assert sm.max_concurrent_tracks == 0
    assert sm.average_track_age == 0.0
    assert sm.average_active_frames == 0.0
    assert sm.total_track_lifetime == 0


def test_new_track_registration(state_manager: TrackStateManager):
    """Test 2: A new track from tracker is registered with state NEW and initial temporal metadata."""
    incoming_track = Track(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=[100.0, 100.0, 200.0, 300.0],
        frame_number=1,
        timestamp=0.033,
    )
    result = state_manager.update([incoming_track], frame_number=1, timestamp=0.033)

    assert isinstance(result, TrackStateResult)
    assert result.count == 1
    t = result.tracks[0]
    assert t.track_id == 1
    assert t.state == TrackState.NEW.value
    assert t.first_seen_frame == 1
    assert t.last_seen_frame == 1
    assert t.first_seen_timestamp == 0.033
    assert t.last_seen_timestamp == 0.033
    assert t.active_frames == 1
    assert t.missed_frames == 0
    assert t.age_frames == 1
    assert state_manager.total_tracks_created == 1


def test_active_track_update(state_manager: TrackStateManager):
    """Test 3: Track present in consecutive frames transitions from NEW to ACTIVE."""
    track_f1 = Track(track_id=1, class_id=0, class_name="person", confidence=0.92, bbox=[100.0, 100.0, 200.0, 300.0])
    state_manager.update([track_f1], frame_number=1, timestamp=0.033)

    track_f2 = Track(track_id=1, class_id=0, class_name="person", confidence=0.93, bbox=[102.0, 100.0, 202.0, 300.0])
    result_f2 = state_manager.update([track_f2], frame_number=2, timestamp=0.066)

    assert result_f2.count == 1
    t = result_f2.tracks[0]
    assert t.track_id == 1
    assert t.state == TrackState.ACTIVE.value
    assert t.first_seen_frame == 1
    assert t.last_seen_frame == 2
    assert t.active_frames == 2
    assert t.missed_frames == 0
    assert t.age_frames == 2


def test_track_state_persistence(state_manager: TrackStateManager):
    """Test 4: Continuous track remains ACTIVE and correctly computes age across 10 frames."""
    for fn in range(1, 11):
        trk = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[100.0 + fn, 100.0, 200.0, 300.0])
        res = state_manager.update([trk], frame_number=fn, timestamp=fn * 0.033)

    t = state_manager.get_track(1)
    assert t is not None
    assert t.state == TrackState.ACTIVE.value
    assert t.first_seen_frame == 1
    assert t.last_seen_frame == 10
    assert t.active_frames == 10
    assert t.missed_frames == 0
    assert t.age_frames == 10


def test_track_becomes_lost_after_missing_frames(state_manager: TrackStateManager):
    """Test 5: Track not returned in a frame transitions to LOST and increments missed_frames."""
    # Frame 1: Track present
    trk = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[100.0, 100.0, 200.0, 300.0])
    state_manager.update([trk], frame_number=1, timestamp=0.033)

    # Frame 2: Missing
    res_f2 = state_manager.update([], frame_number=2, timestamp=0.066)
    assert res_f2.count == 0  # No active visible tracks in this frame

    t = state_manager.get_track(1)
    assert t is not None
    assert t.state == TrackState.LOST.value
    assert t.missed_frames == 1
    assert t.active_frames == 1
    assert t.age_frames == 2
    assert state_manager.currently_lost_tracks == 1


def test_lost_track_recovery(state_manager: TrackStateManager):
    """Test 6: A LOST track that reappears recovers to ACTIVE state and resets missed_frames."""
    # Frame 1: Present (NEW)
    state_manager.update([Track(1, 0, "person", 0.9, [100.0, 100.0, 200.0, 300.0])], frame_number=1, timestamp=0.033)
    # Frame 2: Present (ACTIVE)
    state_manager.update([Track(1, 0, "person", 0.9, [102.0, 100.0, 202.0, 300.0])], frame_number=2, timestamp=0.066)
    # Frame 3: Missing (LOST, missed_frames=1)
    state_manager.update([], frame_number=3, timestamp=0.099)
    # Frame 4: Missing (LOST, missed_frames=2)
    state_manager.update([], frame_number=4, timestamp=0.132)

    lost_t = state_manager.get_track(1)
    assert lost_t.state == TrackState.LOST.value
    assert lost_t.missed_frames == 2

    # Frame 5: Reappears!
    res_f5 = state_manager.update(
        [Track(1, 0, "person", 0.91, [105.0, 100.0, 205.0, 300.0])],
        frame_number=5,
        timestamp=0.165,
    )

    assert res_f5.count == 1
    recovered_t = res_f5.tracks[0]
    assert recovered_t.state == TrackState.ACTIVE.value
    assert recovered_t.missed_frames == 0
    assert recovered_t.active_frames == 3
    assert recovered_t.age_frames == 5


def test_track_removed_after_timeout():
    """Test 7: Track missing for max_missed_frames transitions to REMOVED."""
    sm = TrackStateManager(max_missed_frames=3)

    # Frame 1: Present
    sm.update([Track(1, 0, "person", 0.9, [100.0, 100.0, 200.0, 300.0])], frame_number=1, timestamp=0.033)

    # Frames 2, 3: Missing (LOST, missed_frames = 1, 2)
    sm.update([], frame_number=2, timestamp=0.066)
    sm.update([], frame_number=3, timestamp=0.099)
    assert sm.currently_lost_tracks == 1

    # Frame 4: Missing (missed_frames = 3 >= max_missed_frames -> REMOVED)
    sm.update([], frame_number=4, timestamp=0.132)
    assert sm.currently_lost_tracks == 0
    assert sm.currently_removed_tracks == 1

    removed_t = sm.get_track(1)
    assert removed_t is not None
    assert removed_t.state == TrackState.REMOVED.value
    assert removed_t.missed_frames >= 3


def test_multiple_track_states(state_manager: TrackStateManager):
    """Test 8: Multiple tracks coexist in different lifecycle states simultaneously."""
    # Frame 1: Track 1 and Track 2 appear
    t1 = Track(1, 0, "person", 0.9, [50.0, 100.0, 150.0, 300.0])
    t2 = Track(2, 0, "person", 0.88, [300.0, 100.0, 400.0, 300.0])
    state_manager.update([t1, t2], frame_number=1, timestamp=0.033)

    # Frame 2: Track 1 continues, Track 2 missing, Track 3 appears
    t1_f2 = Track(1, 0, "person", 0.92, [52.0, 100.0, 152.0, 300.0])
    t3 = Track(3, 0, "person", 0.95, [500.0, 100.0, 600.0, 300.0])
    res_f2 = state_manager.update([t1_f2, t3], frame_number=2, timestamp=0.066)

    assert state_manager.get_track(1).state == TrackState.ACTIVE.value
    assert state_manager.get_track(2).state == TrackState.LOST.value
    assert state_manager.get_track(3).state == TrackState.NEW.value
    assert state_manager.currently_active_tracks == 2
    assert state_manager.currently_lost_tracks == 1


def test_non_person_tracks_are_ignored(state_manager: TrackStateManager):
    """Test 9: Tracks with class_id != 0 are strictly ignored by TrackStateManager."""
    car_track = Track(10, 2, "car", 0.95, [100.0, 100.0, 300.0, 300.0])
    chair_track = Track(11, 56, "chair", 0.89, [400.0, 100.0, 450.0, 200.0])

    res = state_manager.update([car_track, chair_track], frame_number=1, timestamp=0.033)
    assert res.count == 0
    assert state_manager.total_tracks_created == 0


def test_state_output_structure(state_manager: TrackStateManager):
    """Test 10: Validate structured output schema and dictionary serialization matching Section 13."""
    trk = Track(1, 0, "person", 0.9324, [152.4, 150.0, 312.8, 390.0])
    state_manager.update([trk], frame_number=1, timestamp=0.033)

    trk_f30 = Track(1, 0, "person", 0.9324, [152.4, 150.0, 312.8, 390.0])
    res = state_manager.update([trk_f30], frame_number=30, timestamp=1.0)

    d = res.to_dict()
    assert d["frame_number"] == 30
    assert d["timestamp"] == 1.0
    assert "tracks" in d
    assert len(d["tracks"]) == 1

    t_dict = d["tracks"][0]
    expected_keys = {
        "track_id", "class_id", "class_name", "state", "confidence",
        "bbox", "first_seen_frame", "last_seen_frame", "first_seen_timestamp",
        "last_seen_timestamp", "active_frames", "missed_frames", "age_frames",
    }
    assert expected_keys.issubset(t_dict.keys())
    assert t_dict["track_id"] == 1
    assert t_dict["state"] == "ACTIVE"
    assert t_dict["first_seen_frame"] == 1
    assert t_dict["last_seen_frame"] == 30


def test_invalid_state_manager_input(state_manager: TrackStateManager):
    """Test 11: Invalid input types raise InvalidStateManagerInputError."""
    with pytest.raises(InvalidStateManagerInputError):
        state_manager.update("invalid_string_input")

    with pytest.raises(InvalidStateManagerInputError):
        state_manager.update(12345)


def test_video_tracking_state_integration():
    """Test 12: End-to-end integration: VideoFileReader -> YOLOPersonDetector -> ByteTrack -> TrackStateManager."""
    video_path = Path(__file__).resolve().parent.parent / "data" / "videos" / "test.mp4"
    assert video_path.exists(), f"Test video must exist at {video_path}"

    reader = VideoFileReader(video_path)
    detector = YOLOPersonDetector("data/models/yolov8n.pt", confidence_threshold=0.35, device="auto")
    tracker = ByteTrackPersonTracker()
    state_manager = TrackStateManager(max_missed_frames=15)

    processed_frames = 0
    with reader:
        for i, frame in enumerate(reader):
            if i >= 20:
                break
            det_res = detector.detect(frame)
            track_res = tracker.update(det_res)
            state_res = state_manager.update(track_res, frame_number=frame.frame_number, timestamp=frame.timestamp)

            processed_frames += 1
            assert isinstance(state_res, TrackStateResult)

    assert processed_frames == 20
    assert state_manager.total_tracks_created >= 1
    assert state_manager.currently_active_tracks >= 1
