"""
Unit and integration tests for EntryExitEngine and movement event analytics.
Phase: Phase 8 - Entry / Exit Analytics Engine
"""

from pathlib import Path
import pytest

from src.analytics import (
    EntryExitEngine,
    EntryExitEvent,
    EntryExitResult,
    InvalidEntryExitConfigError,
    MovementType,
)
from src.detection import YOLOPersonDetector
from src.tracking import ByteTrackPersonTracker, Track, TrackStateManager
from src.video_input import VideoFileReader
from src.zones import (
    TrackZoneAssignment,
    Zone,
    ZoneCountingEngine,
    ZoneEngine,
    ZoneResult,
)


@pytest.fixture
def default_entry_exit_engine() -> EntryExitEngine:
    """Fixture providing an EntryExitEngine with entrance configured for entry and exit."""
    definitions = [
        {"id": "main_entrance", "from": None, "to": "entrance", "type": "ENTRY"},
        {"id": "main_exit", "from": "entrance", "to": None, "type": "EXIT"},
    ]
    return EntryExitEngine(definitions=definitions, allow_initial_entry=False)


def test_engine_initialization():
    """Test 1: Engine initializes correctly with valid configuration schema."""
    definitions = [
        {"id": "gate_in", "from": None, "to": "lobby", "type": "ENTRY"},
        {"id": "gate_out", "from": "lobby", "to": None, "type": "EXIT"},
    ]
    engine = EntryExitEngine(definitions=definitions)
    assert engine.total_entries == 0
    assert engine.total_exits == 0
    assert engine.total_zone_transitions == 0
    assert engine.frames_evaluated == 0

    # Also test from_config factory
    cfg = {
        "definitions": definitions,
        "allow_initial_entry": True,
    }
    engine_from_cfg = EntryExitEngine.from_config(cfg)
    assert engine_from_cfg.allow_initial_entry is True


def test_invalid_configuration():
    """Test 2: Invalid configuration raises InvalidEntryExitConfigError."""
    # Definitions must be a list
    with pytest.raises(InvalidEntryExitConfigError):
        EntryExitEngine(definitions="invalid_string")

    # Item must be dict
    with pytest.raises(InvalidEntryExitConfigError):
        EntryExitEngine(definitions=["not_a_dict"])

    # Both from and to are outside/null
    with pytest.raises(InvalidEntryExitConfigError):
        EntryExitEngine(definitions=[{"id": "bad", "from": None, "to": None}])

    # from and to are the same zone
    with pytest.raises(InvalidEntryExitConfigError):
        EntryExitEngine(definitions=[{"id": "bad", "from": "entrance", "to": "entrance"}])

    # Entry target cannot be outside/null
    with pytest.raises(InvalidEntryExitConfigError):
        EntryExitEngine(definitions=[{"id": "bad", "from": "entrance", "to": None, "type": "ENTRY"}])

    # Exit origin cannot be outside/null
    with pytest.raises(InvalidEntryExitConfigError):
        EntryExitEngine(definitions=[{"id": "bad", "from": None, "to": "entrance", "type": "EXIT"}])


def test_entry_detection(default_entry_exit_engine: EntryExitEngine):
    """Test 3: Movement from null -> entrance generates an ENTRY event."""
    # Frame 1: Track 1 observed outside (null)
    t_f1 = TrackZoneAssignment(track_id=1, position=[800.0, 200.0], previous_zone_id=None, zone_id=None, zone_name=None, transition=None)
    res_f1 = default_entry_exit_engine.update([t_f1], frame_number=1)
    assert len(res_f1.events) == 0

    # Frame 2: Track 1 moves into entrance
    t_f2 = TrackZoneAssignment(track_id=1, position=[150.0, 200.0], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    res_f2 = default_entry_exit_engine.update([t_f2], frame_number=2, timestamp=0.066)

    assert len(res_f2.events) == 1
    ev = res_f2.events[0]
    assert ev.type == "ENTRY"
    assert ev.movement_type == MovementType.ENTRY.value
    assert ev.track_id == 1
    assert ev.zone_id == "entrance"
    assert ev.previous_zone_id is None
    assert ev.current_zone_id == "entrance"
    assert ev.frame_number == 2
    assert default_entry_exit_engine.total_entries == 1


def test_exit_detection(default_entry_exit_engine: EntryExitEngine):
    """Test 4: Movement from entrance -> null generates an EXIT event."""
    # Frame 1: Track 1 in entrance (baseline established)
    t_f1 = TrackZoneAssignment(track_id=1, position=[150.0, 200.0], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    default_entry_exit_engine.update([t_f1], frame_number=1)

    # Frame 2: Track 1 moves outside
    t_f2 = TrackZoneAssignment(track_id=1, position=[800.0, 200.0], previous_zone_id="entrance", zone_id=None, zone_name=None, transition=None)
    res_f2 = default_entry_exit_engine.update([t_f2], frame_number=2, timestamp=0.066)

    assert len(res_f2.events) == 1
    ev = res_f2.events[0]
    assert ev.type == "EXIT"
    assert ev.movement_type == MovementType.EXIT.value
    assert ev.track_id == 1
    assert ev.zone_id == "entrance"
    assert ev.previous_zone_id == "entrance"
    assert ev.current_zone_id is None
    assert default_entry_exit_engine.total_exits == 1


def test_zone_to_zone_transition(default_entry_exit_engine: EntryExitEngine):
    """Test 5: Movement entrance -> waiting_area is ZONE_TRANSITION (NOT ENTRY, NOT EXIT)."""
    # Frame 1: In entrance
    t_f1 = TrackZoneAssignment(track_id=1, position=[150.0, 200.0], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    default_entry_exit_engine.update([t_f1], frame_number=1)

    # Frame 2: Moves into waiting_area
    t_f2 = TrackZoneAssignment(track_id=1, position=[450.0, 200.0], previous_zone_id="entrance", zone_id="waiting_area", zone_name="Waiting Area", transition=None)
    res = default_entry_exit_engine.update([t_f2], frame_number=2)

    assert len(res.events) == 1
    ev = res.events[0]
    assert ev.type == "ZONE_TRANSITION"
    assert ev.movement_type == MovementType.ZONE_TRANSITION.value
    assert ev.previous_zone_id == "entrance"
    assert ev.current_zone_id == "waiting_area"
    # Verify strict non-classification as entry or exit
    assert default_entry_exit_engine.total_entries == 0
    assert default_entry_exit_engine.total_exits == 0
    assert default_entry_exit_engine.total_zone_transitions == 1


def test_no_duplicate_entry(default_entry_exit_engine: EntryExitEngine):
    """Test 6: Repeated frames inside entrance after entry emit only 1 ENTRY event."""
    # Frame 1: Outside
    t_out = TrackZoneAssignment(track_id=1, position=[800, 200], previous_zone_id=None, zone_id=None, zone_name=None, transition=None)
    default_entry_exit_engine.update([t_out], frame_number=1)

    # Frame 2: Enters entrance
    t_in = TrackZoneAssignment(track_id=1, position=[150, 200], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    res_f2 = default_entry_exit_engine.update([t_in], frame_number=2)
    assert len(res_f2.events) == 1
    assert res_f2.events[0].type == "ENTRY"

    # Frames 3 to 10: Still in entrance
    for fn in range(3, 11):
        res_fn = default_entry_exit_engine.update([t_in], frame_number=fn)
        assert len(res_fn.events) == 0

    assert default_entry_exit_engine.total_entries == 1


def test_no_duplicate_exit(default_entry_exit_engine: EntryExitEngine):
    """Test 7: Repeated frames outside after exit emit only 1 EXIT event."""
    # Frame 1: In entrance
    t_in = TrackZoneAssignment(track_id=1, position=[150, 200], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    default_entry_exit_engine.update([t_in], frame_number=1)

    # Frame 2: Leaves to outside
    t_out = TrackZoneAssignment(track_id=1, position=[800, 200], previous_zone_id="entrance", zone_id=None, zone_name=None, transition=None)
    res_f2 = default_entry_exit_engine.update([t_out], frame_number=2)
    assert len(res_f2.events) == 1
    assert res_f2.events[0].type == "EXIT"

    # Frames 3 to 10: Still outside
    for fn in range(3, 11):
        res_fn = default_entry_exit_engine.update([t_out], frame_number=fn)
        assert len(res_fn.events) == 0

    assert default_entry_exit_engine.total_exits == 1


def test_re_entry(default_entry_exit_engine: EntryExitEngine):
    """Test 8: ENTRY -> EXIT -> ENTRY sequence produces 2 ENTRY and 1 EXIT events."""
    t_out = TrackZoneAssignment(track_id=1, position=[800, 200], previous_zone_id=None, zone_id=None, zone_name=None, transition=None)
    t_in = TrackZoneAssignment(track_id=1, position=[150, 200], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)

    # Frame 1: Outside
    default_entry_exit_engine.update([t_out], frame_number=1)

    # Frame 2: First Entry
    res2 = default_entry_exit_engine.update([t_in], frame_number=2)
    assert len(res2.events) == 1 and res2.events[0].type == "ENTRY"

    # Frame 3: Exit
    res3 = default_entry_exit_engine.update([t_out], frame_number=3)
    assert len(res3.events) == 1 and res3.events[0].type == "EXIT"

    # Frame 4: Second Entry (Re-entry)
    res4 = default_entry_exit_engine.update([t_in], frame_number=4)
    assert len(res4.events) == 1 and res4.events[0].type == "ENTRY"

    assert default_entry_exit_engine.total_entries == 2
    assert default_entry_exit_engine.total_exits == 1


def test_multiple_tracks(default_entry_exit_engine: EntryExitEngine):
    """Test 9: Multiple tracks processed independently: ID 1 -> ENTRY, ID 2 -> EXIT, ID 3 -> ZONE_TRANSITION."""
    # Frame 1: Setup baseline for tracks
    t1_f1 = TrackZoneAssignment(track_id=1, position=[800, 200], previous_zone_id=None, zone_id=None, zone_name=None, transition=None)
    t2_f1 = TrackZoneAssignment(track_id=2, position=[150, 200], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    t3_f1 = TrackZoneAssignment(track_id=3, position=[160, 200], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    default_entry_exit_engine.update([t1_f1, t2_f1, t3_f1], frame_number=1)

    # Frame 2:
    # ID 1 moves outside -> entrance (ENTRY)
    # ID 2 moves entrance -> outside (EXIT)
    # ID 3 moves entrance -> waiting_area (ZONE_TRANSITION)
    t1_f2 = TrackZoneAssignment(track_id=1, position=[150, 200], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    t2_f2 = TrackZoneAssignment(track_id=2, position=[800, 200], previous_zone_id=None, zone_id=None, zone_name=None, transition=None)
    t3_f2 = TrackZoneAssignment(track_id=3, position=[450, 200], previous_zone_id=None, zone_id="waiting_area", zone_name="Waiting Area", transition=None)

    res = default_entry_exit_engine.update([t1_f2, t2_f2, t3_f2], frame_number=2)
    assert len(res.events) == 3

    event_map = {e.track_id: e.type for e in res.events}
    assert event_map[1] == "ENTRY"
    assert event_map[2] == "EXIT"
    assert event_map[3] == "ZONE_TRANSITION"


def test_lost_track(default_entry_exit_engine: EntryExitEngine):
    """Test 10: Track becoming LOST does not emit an EXIT event."""
    # Frame 1: In entrance
    t_in = TrackZoneAssignment(track_id=7, position=[150, 200], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    default_entry_exit_engine.update([t_in], frame_number=1)

    # Frame 2: Track 7 is missing / LOST (empty frame)
    res = default_entry_exit_engine.update([], frame_number=2)
    assert len(res.events) == 0
    assert default_entry_exit_engine.total_exits == 0


def test_removed_track(default_entry_exit_engine: EntryExitEngine):
    """Test 11: Track transitioning to REMOVED cleans up state without emitting false EXIT."""
    t_in = TrackZoneAssignment(track_id=7, position=[150, 200], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    default_entry_exit_engine.update([t_in], frame_number=1)
    assert 7 in default_entry_exit_engine._track_states

    # Notify removal of track 7
    res = default_entry_exit_engine.update([], frame_number=30, removed_track_ids=[7])
    assert len(res.events) == 0
    assert default_entry_exit_engine.total_exits == 0
    assert 7 not in default_entry_exit_engine._track_states  # Cleaned up!


def test_new_track_already_inside_zone(default_entry_exit_engine: EntryExitEngine):
    """Test 12: First observation of a new track already inside entrance does not claim historical ENTRY."""
    # Frame 1: Track 10 first observed inside entrance
    t10 = TrackZoneAssignment(track_id=10, position=[150, 200], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    res_f1 = default_entry_exit_engine.update([t10], frame_number=1)
    assert len(res_f1.events) == 0
    assert default_entry_exit_engine.total_entries == 0

    # Frame 2: Track 10 still inside entrance
    res_f2 = default_entry_exit_engine.update([t10], frame_number=2)
    assert len(res_f2.events) == 0
    assert default_entry_exit_engine.total_entries == 0


def test_multiple_entry_exit_definitions():
    """Test 13: Multiple configured entry and exit transitions operate independently."""
    definitions = [
        {"id": "main_in", "from": None, "to": "main_entrance", "type": "ENTRY"},
        {"id": "side_in", "from": None, "to": "side_entrance", "type": "ENTRY"},
        {"id": "main_out", "from": "main_entrance", "to": None, "type": "EXIT"},
    ]
    engine = EntryExitEngine(definitions=definitions)

    # Establish baseline outside
    t1 = TrackZoneAssignment(track_id=1, position=[0, 0], previous_zone_id=None, zone_id=None, zone_name=None, transition=None)
    t2 = TrackZoneAssignment(track_id=2, position=[0, 0], previous_zone_id=None, zone_id=None, zone_name=None, transition=None)
    engine.update([t1, t2], frame_number=1)

    # Frame 2: Track 1 -> main_entrance, Track 2 -> side_entrance
    t1_f2 = TrackZoneAssignment(track_id=1, position=[0, 0], previous_zone_id=None, zone_id="main_entrance", zone_name="Main", transition=None)
    t2_f2 = TrackZoneAssignment(track_id=2, position=[0, 0], previous_zone_id=None, zone_id="side_entrance", zone_name="Side", transition=None)
    res_f2 = engine.update([t1_f2, t2_f2], frame_number=2)

    assert len(res_f2.events) == 2
    assert res_f2.events[0].type == "ENTRY" and res_f2.events[0].transition_id == "main_in"
    assert res_f2.events[1].type == "ENTRY" and res_f2.events[1].transition_id == "side_in"
    assert engine.total_entries == 2


def test_empty_tracks(default_entry_exit_engine: EntryExitEngine):
    """Test 14: Empty track inputs return valid empty EntryExitResult without errors."""
    res_empty = default_entry_exit_engine.update([], frame_number=10)
    assert isinstance(res_empty, EntryExitResult)
    assert res_empty.events == []
    assert res_empty.event_count == 0
    assert res_empty.frame_number == 10

    res_none = default_entry_exit_engine.update(None, frame_number=11)
    assert res_none.events == []
    assert res_none.event_count == 0


def test_metrics(default_entry_exit_engine: EntryExitEngine):
    """Test 15: Aggregate metrics total_entries, total_exits, and total_zone_transitions update accurately."""
    # Track 1: outside -> entrance (ENTRY)
    default_entry_exit_engine.record_movement(track_id=1, from_zone=None, to_zone="entrance", frame_number=1)
    # Track 1: entrance -> waiting_area (ZONE_TRANSITION)
    default_entry_exit_engine.record_movement(track_id=1, from_zone="entrance", to_zone="waiting_area", frame_number=2)
    # Track 2: entrance -> outside (EXIT)
    default_entry_exit_engine.record_movement(track_id=2, from_zone="entrance", to_zone=None, frame_number=3)

    assert default_entry_exit_engine.total_entries == 1
    assert default_entry_exit_engine.total_exits == 1
    assert default_entry_exit_engine.total_zone_transitions == 1


def test_video_integration_full_pipeline():
    """Test 16: Complete integration test on test.mp4: VideoReader -> YOLO -> ByteTrack -> StateManager -> ZoneEngine -> ZoneCounting -> EntryExitEngine."""
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

    definitions = [
        {"id": "main_in", "from": None, "to": "entrance", "type": "ENTRY"},
        {"id": "main_out", "from": "entrance", "to": None, "type": "EXIT"},
    ]
    # Set allow_initial_entry=True for pipeline test to track entry of person entering in initial video frame
    entry_exit_engine = EntryExitEngine(definitions=definitions, allow_initial_entry=True)

    frames_tested = 0
    total_events = 0

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
            ee_res = entry_exit_engine.update(z_res, frame.frame_number, frame.timestamp, removed_track_ids=[t.track_id for t in state_manager.removed_tracks])

            assert isinstance(ee_res, EntryExitResult)
            assert ee_res.frame_number == frame.frame_number
            total_events += len(ee_res.events)

    finally:
        reader.release()

    assert frames_tested == 35
    assert total_events >= 1
