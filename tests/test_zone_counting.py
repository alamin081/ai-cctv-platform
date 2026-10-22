"""
Unit and integration tests for ZoneCountingEngine and frame-level zone occupancy.
Phase: Phase 7 - Zone Counting / Occupancy Engine
"""

from pathlib import Path
import pytest

from src.detection import YOLOPersonDetector
from src.tracking import ByteTrackPersonTracker, Track, TrackStateManager
from src.video_input import VideoFileReader
from src.zones import (
    TrackZoneAssignment,
    Zone,
    ZoneCountingEngine,
    ZoneCountingResult,
    ZoneEngine,
    ZoneEvent,
    ZoneOccupancy,
    ZoneResult,
)


@pytest.fixture
def two_zones_counting_engine() -> ZoneCountingEngine:
    """Fixture providing a ZoneCountingEngine with entrance and waiting_area zones."""
    engine = ZoneCountingEngine()
    engine.register_zone("entrance", "Entrance")
    engine.register_zone("waiting_area", "Waiting Area")
    return engine


def test_empty_zones(two_zones_counting_engine: ZoneCountingEngine):
    """Test 1: When no tracks are present, all configured zone occupancy counts are 0."""
    result = two_zones_counting_engine.update([], frame_number=1, timestamp=0.033)

    assert isinstance(result, ZoneCountingResult)
    assert result.total_occupancy == 0
    assert len(result.events) == 0
    assert "entrance" in result.occupancy
    assert "waiting_area" in result.occupancy

    assert result.get_count("entrance") == 0
    assert result.get_track_ids("entrance") == []
    assert result.get_count("waiting_area") == 0
    assert result.get_track_ids("waiting_area") == []

    # Also test with None input
    res_none = two_zones_counting_engine.update(None, frame_number=2, timestamp=0.066)
    assert res_none.total_occupancy == 0
    assert res_none.get_count("entrance") == 0


def test_one_person(two_zones_counting_engine: ZoneCountingEngine):
    """Test 2: A single track assigned to entrance sets entrance count to 1 and track_ids to [1]."""
    t1 = TrackZoneAssignment(
        track_id=1,
        position=[150.0, 200.0],
        previous_zone_id=None,
        zone_id="entrance",
        zone_name="Entrance",
        transition="ENTERED",
    )
    result = two_zones_counting_engine.update([t1], frame_number=1, timestamp=0.033)

    assert result.total_occupancy == 1
    assert result.get_count("entrance") == 1
    assert result.get_track_ids("entrance") == [1]
    assert result.get_count("waiting_area") == 0
    assert result.get_track_ids("waiting_area") == []

    occ_ent = result.get_occupancy("entrance")
    assert occ_ent is not None
    assert occ_ent.count == len(occ_ent.track_ids) == 1


def test_multiple_people(two_zones_counting_engine: ZoneCountingEngine):
    """Test 3: Multiple tracks across zones: entrance = [1, 3], waiting_area = [2, 4]."""
    assignments = [
        TrackZoneAssignment(track_id=1, position=[100.0, 100.0], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None),
        TrackZoneAssignment(track_id=3, position=[120.0, 150.0], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None),
        TrackZoneAssignment(track_id=2, position=[400.0, 100.0], previous_zone_id=None, zone_id="waiting_area", zone_name="Waiting Area", transition=None),
        TrackZoneAssignment(track_id=4, position=[450.0, 150.0], previous_zone_id=None, zone_id="waiting_area", zone_name="Waiting Area", transition=None),
    ]
    result = two_zones_counting_engine.update(assignments, frame_number=1)

    assert result.total_occupancy == 4
    assert result.get_count("entrance") == 2
    assert result.get_track_ids("entrance") == [1, 3]
    assert result.get_count("waiting_area") == 2
    assert result.get_track_ids("waiting_area") == [2, 4]


def test_duplicate_frame_detection(two_zones_counting_engine: ZoneCountingEngine):
    """Test 4: Same Track ID remaining in same zone across multiple frames does not accumulate counts."""
    t1 = TrackZoneAssignment(track_id=1, position=[150.0, 200.0], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)

    for fn in range(1, 10):
        res = two_zones_counting_engine.update([t1], frame_number=fn)
        assert res.get_count("entrance") == 1
        assert res.get_track_ids("entrance") == [1]
        assert res.total_occupancy == 1


def test_zone_entry(two_zones_counting_engine: ZoneCountingEngine):
    """Test 5: Track moving outside -> entrance generates exactly one ZONE_ENTER event."""
    # Frame 1: Track 1 is outside
    t_f1 = TrackZoneAssignment(track_id=1, position=[800.0, 200.0], previous_zone_id=None, zone_id=None, zone_name=None, transition=None)
    res_f1 = two_zones_counting_engine.update([t_f1], frame_number=1)
    assert len(res_f1.events) == 0
    assert res_f1.get_count("entrance") == 0

    # Frame 2: Track 1 moves into entrance
    t_f2 = TrackZoneAssignment(track_id=1, position=[150.0, 200.0], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition="ENTERED")
    res_f2 = two_zones_counting_engine.update([t_f2], frame_number=2)
    assert len(res_f2.events) == 1
    ev = res_f2.events[0]
    assert ev.event_type == "ZONE_ENTER"
    assert ev.track_id == 1
    assert ev.zone == "entrance"
    assert ev.zone_id == "entrance"
    assert ev.frame_number == 2

    # Frame 3: Track 1 stays in entrance (deduplication check)
    t_f3 = TrackZoneAssignment(track_id=1, position=[155.0, 205.0], previous_zone_id="entrance", zone_id="entrance", zone_name="Entrance", transition=None)
    res_f3 = two_zones_counting_engine.update([t_f3], frame_number=3)
    assert len(res_f3.events) == 0
    assert two_zones_counting_engine.total_enter_events == 1


def test_zone_exit(two_zones_counting_engine: ZoneCountingEngine):
    """Test 6: Track moving entrance -> outside generates exactly one ZONE_EXIT event."""
    # Frame 1: In entrance
    t_f1 = TrackZoneAssignment(track_id=1, position=[150.0, 200.0], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    res_f1 = two_zones_counting_engine.update([t_f1], frame_number=1)
    assert res_f1.get_count("entrance") == 1

    # Frame 2: Leaves to outside
    t_f2 = TrackZoneAssignment(track_id=1, position=[800.0, 200.0], previous_zone_id="entrance", zone_id=None, zone_name=None, transition="LEFT")
    res_f2 = two_zones_counting_engine.update([t_f2], frame_number=2)
    assert len(res_f2.events) == 1
    ev = res_f2.events[0]
    assert ev.event_type == "ZONE_EXIT"
    assert ev.track_id == 1
    assert ev.zone_id == "entrance"
    assert res_f2.get_count("entrance") == 0

    # Frame 3: Remains outside -> no further events
    t_f3 = TrackZoneAssignment(track_id=1, position=[810.0, 210.0], previous_zone_id=None, zone_id=None, zone_name=None, transition=None)
    res_f3 = two_zones_counting_engine.update([t_f3], frame_number=3)
    assert len(res_f3.events) == 0
    assert two_zones_counting_engine.total_exit_events == 1


def test_zone_transition(two_zones_counting_engine: ZoneCountingEngine):
    """Test 7: Track moving entrance -> waiting_area generates exactly one ZONE_TRANSITION event."""
    # Frame 1: In entrance
    t_f1 = TrackZoneAssignment(track_id=1, position=[150.0, 200.0], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    two_zones_counting_engine.update([t_f1], frame_number=1)

    # Frame 2: Moves into waiting_area
    t_f2 = TrackZoneAssignment(track_id=1, position=[450.0, 200.0], previous_zone_id="entrance", zone_id="waiting_area", zone_name="Waiting Area", transition="ENTERED")
    res_f2 = two_zones_counting_engine.update([t_f2], frame_number=2)

    assert len(res_f2.events) == 1
    ev = res_f2.events[0]
    assert ev.event_type == "ZONE_TRANSITION"
    assert ev.track_id == 1
    assert ev.from_zone == "entrance"
    assert ev.to_zone == "waiting_area"
    assert res_f2.get_count("entrance") == 0
    assert res_f2.get_count("waiting_area") == 1
    assert two_zones_counting_engine.total_transition_events == 1


def test_no_duplicate_transition(two_zones_counting_engine: ZoneCountingEngine):
    """Test 8: Person remains in waiting_area for many frames without repeated transition events."""
    # Frame 1: Initial transition into waiting_area
    t_f1 = TrackZoneAssignment(track_id=1, position=[450.0, 200.0], previous_zone_id=None, zone_id="waiting_area", zone_name="Waiting Area", transition=None)
    res_f1 = two_zones_counting_engine.update([t_f1], frame_number=1)
    assert len(res_f1.events) == 1
    assert res_f1.events[0].event_type == "ZONE_ENTER"

    # Frames 2 to 20: Track 1 stays in waiting_area
    for fn in range(2, 21):
        t_fn = TrackZoneAssignment(track_id=1, position=[450.0 + fn, 200.0], previous_zone_id="waiting_area", zone_id="waiting_area", zone_name="Waiting Area", transition=None)
        res_fn = two_zones_counting_engine.update([t_fn], frame_number=fn)
        assert len(res_fn.events) == 0  # Deduplicated!
        assert res_fn.get_count("waiting_area") == 1

    assert two_zones_counting_engine.total_enter_events == 1
    assert two_zones_counting_engine.total_transition_events == 0


def test_multiple_track_ids(two_zones_counting_engine: ZoneCountingEngine):
    """Test 9: Multiple Track IDs are tracked independently with correct individual events."""
    # Frame 1: Track 1 enters entrance, Track 2 enters waiting_area
    t1 = TrackZoneAssignment(track_id=1, position=[100, 100], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    t2 = TrackZoneAssignment(track_id=2, position=[400, 100], previous_zone_id=None, zone_id="waiting_area", zone_name="Waiting Area", transition=None)
    res_f1 = two_zones_counting_engine.update([t1, t2], frame_number=1)

    assert len(res_f1.events) == 2
    event_types = {e.track_id: e.event_type for e in res_f1.events}
    assert event_types[1] == "ZONE_ENTER"
    assert event_types[2] == "ZONE_ENTER"

    # Frame 2: Track 1 transitions to waiting_area, Track 2 leaves to outside
    t1_f2 = TrackZoneAssignment(track_id=1, position=[420, 120], previous_zone_id="entrance", zone_id="waiting_area", zone_name="Waiting Area", transition=None)
    t2_f2 = TrackZoneAssignment(track_id=2, position=[800, 100], previous_zone_id="waiting_area", zone_id=None, zone_name=None, transition=None)
    res_f2 = two_zones_counting_engine.update([t1_f2, t2_f2], frame_number=2)

    assert len(res_f2.events) == 2
    ev_map = {e.track_id: e for e in res_f2.events}
    assert ev_map[1].event_type == "ZONE_TRANSITION"
    assert ev_map[1].from_zone == "entrance" and ev_map[1].to_zone == "waiting_area"
    assert ev_map[2].event_type == "ZONE_EXIT"
    assert ev_map[2].zone_id == "waiting_area"


def test_track_disappearance(two_zones_counting_engine: ZoneCountingEngine):
    """Test 10: When a track disappears from detections, it is removed from current occupancy immediately."""
    # Frame 100: Track 7 in entrance
    t7 = TrackZoneAssignment(track_id=7, position=[150, 200], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    res_100 = two_zones_counting_engine.update([t7], frame_number=100)
    assert res_100.get_count("entrance") == 1
    assert res_100.get_track_ids("entrance") == [7]

    # Frame 101: Track 7 still in entrance
    res_101 = two_zones_counting_engine.update([t7], frame_number=101)
    assert res_101.get_count("entrance") == 1

    # Frame 102: Track 7 disappears (empty frame)
    res_102 = two_zones_counting_engine.update([], frame_number=102)
    assert res_102.get_count("entrance") == 0
    assert res_102.get_track_ids("entrance") == []
    assert res_102.total_occupancy == 0


def test_unknown_non_zone_assignment(two_zones_counting_engine: ZoneCountingEngine):
    """Test 11: Track outside all configured zones does not appear in any zone occupancy."""
    t_outside = TrackZoneAssignment(track_id=99, position=[900, 900], previous_zone_id=None, zone_id=None, zone_name=None, transition=None)
    res = two_zones_counting_engine.update([t_outside], frame_number=1)

    assert res.total_occupancy == 0
    assert res.get_count("entrance") == 0
    assert res.get_count("waiting_area") == 0
    assert 99 not in res.get_track_ids("entrance")
    assert 99 not in res.get_track_ids("waiting_area")


def test_multiple_configured_zones():
    """Test 12: ZoneCountingEngine works with arbitrary number of configured zones (5 zones)."""
    zones_list = [
        {"id": "entrance", "name": "Entrance"},
        {"id": "waiting_area", "name": "Waiting Area"},
        {"id": "counter_01", "name": "Counter 01"},
        {"id": "counter_02", "name": "Counter 02"},
        {"id": "exit", "name": "Exit Door"},
    ]
    engine = ZoneCountingEngine(zones=zones_list)
    assert len(engine.configured_zones) == 5

    assignments = [
        TrackZoneAssignment(track_id=1, position=[0, 0], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None),
        TrackZoneAssignment(track_id=2, position=[0, 0], previous_zone_id=None, zone_id="counter_01", zone_name="Counter 01", transition=None),
        TrackZoneAssignment(track_id=3, position=[0, 0], previous_zone_id=None, zone_id="exit", zone_name="Exit Door", transition=None),
    ]
    res = engine.update(assignments, frame_number=1)

    assert res.total_occupancy == 3
    assert res.get_count("entrance") == 1
    assert res.get_count("counter_01") == 1
    assert res.get_count("exit") == 1
    assert res.get_count("counter_02") == 0
    assert res.get_count("waiting_area") == 0


def test_memory_safety_cleanup(two_zones_counting_engine: ZoneCountingEngine):
    """Test 13: cleanup_track and cleanup_tracks purge internal state without memory leaks."""
    t1 = TrackZoneAssignment(track_id=1, position=[100, 100], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    two_zones_counting_engine.update([t1], frame_number=1)
    assert 1 in two_zones_counting_engine._track_last_zone

    two_zones_counting_engine.cleanup_track(1)
    assert 1 not in two_zones_counting_engine._track_last_zone

    # Bulk cleanup
    two_zones_counting_engine._track_last_zone[2] = "entrance"
    two_zones_counting_engine._track_last_zone[3] = "waiting_area"
    two_zones_counting_engine.cleanup_tracks([2, 3])
    assert 2 not in two_zones_counting_engine._track_last_zone
    assert 3 not in two_zones_counting_engine._track_last_zone


def test_zone_engine_to_counting_engine_coupling():
    """Test 14: Direct coupling of ZoneEngine output into ZoneCountingEngine."""
    zones_cfg = [
        {"id": "entrance", "name": "Entrance", "polygon": [[0, 0], [320, 0], [320, 480], [0, 480]]},
        {"id": "waiting_area", "name": "Waiting Area", "polygon": [[320, 0], [640, 0], [640, 480], [320, 480]]},
    ]
    zone_engine = ZoneEngine(zones=zones_cfg)
    counting_engine = ZoneCountingEngine.from_zone_engine(zone_engine)

    assert "entrance" in counting_engine.configured_zones
    assert "waiting_area" in counting_engine.configured_zones

    # Create dummy tracks
    trk1 = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[100, 100, 200, 200])  # center: (150, 150) -> entrance
    trk2 = Track(track_id=2, class_id=0, class_name="person", confidence=0.9, bbox=[400, 100, 500, 200])  # center: (450, 150) -> waiting_area

    zone_res = zone_engine.update([trk1, trk2], frame_number=1, timestamp=0.033)
    count_res = counting_engine.update(zone_res)

    assert count_res.get_count("entrance") == 1
    assert count_res.get_track_ids("entrance") == [1]
    assert count_res.get_count("waiting_area") == 1
    assert count_res.get_track_ids("waiting_area") == [2]
    assert count_res.total_occupancy == 2
    assert len(count_res.events) == 2


def test_video_integration_full_pipeline():
    """Test 15: Full pipeline VideoReader -> YOLO -> ByteTrack -> StateManager -> ZoneEngine -> ZoneCountingEngine."""
    video_path = Path(__file__).resolve().parent.parent / "data" / "videos" / "test.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found at {video_path}")

    model_path = Path(__file__).resolve().parent.parent / "data" / "models" / "yolov8n.pt"
    if not model_path.exists():
        model_path = Path("yolov8n.pt")

    reader = VideoFileReader(video_path)
    reader.open()

    detector = YOLOPersonDetector(model_name_or_path=model_path, confidence_threshold=0.35)
    tracker = ByteTrackPersonTracker(track_high_thresh=0.4, new_track_thresh=0.4)
    state_manager = TrackStateManager(max_missed_frames=15)

    zones_cfg = [
        {"id": "entrance", "name": "Entrance", "polygon": [[0, 0], [320, 0], [320, 480], [0, 480]]},
        {"id": "waiting_area", "name": "Waiting Area", "polygon": [[320, 0], [640, 0], [640, 480], [320, 480]]},
    ]
    zone_engine = ZoneEngine(zones=zones_cfg)
    counting_engine = ZoneCountingEngine.from_zone_engine(zone_engine)

    frames_tested = 0
    total_events_collected = 0

    try:
        for _ in range(35):
            frame = reader.read()
            if frame is None:
                break
            frames_tested += 1

            det = detector.detect(frame)
            trk = tracker.update(det, frame.frame_number, frame.timestamp)
            state = state_manager.update(trk, frame.frame_number, frame.timestamp)
            z_res = zone_engine.update(state, frame.frame_number, frame.timestamp)
            c_res = counting_engine.update(z_res, removed_track_ids=[t.track_id for t in state_manager.removed_tracks])

            assert isinstance(c_res, ZoneCountingResult)
            assert c_res.frame_number == frame.frame_number
            assert "entrance" in c_res.occupancy
            assert "waiting_area" in c_res.occupancy

            total_events_collected += len(c_res.events)

    finally:
        reader.release()

    assert frames_tested == 35
    assert total_events_collected >= 1  # At least track 1 entered entrance and transitioned
