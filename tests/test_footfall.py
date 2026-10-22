"""
Unit and integration tests for FootfallAnalyticsEngine and time-bucket metrics.
Phase: Phase 9 - Footfall Analytics Engine
"""

from pathlib import Path
import pytest

from src.analytics import (
    EntryExitEngine,
    EntryExitEvent,
    EntryExitResult,
    FootfallAnalyticsEngine,
    FootfallBucket,
    FootfallResult,
    FootfallSummary,
    InvalidFootfallConfigError,
    InvalidFootfallEventError,
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
def default_footfall_engine() -> FootfallAnalyticsEngine:
    """Fixture providing a default FootfallAnalyticsEngine with 60s buckets for testing."""
    return FootfallAnalyticsEngine(bucket_seconds=60.0, deduplicate_events=True)


def test_initialization():
    """Test 1: Engine initializes correctly with valid configuration."""
    engine = FootfallAnalyticsEngine(bucket_minutes=15, deduplicate_events=True)
    assert engine.bucket_seconds == 900.0
    assert engine.deduplicate_events is True
    assert engine.total_entries == 0
    assert engine.total_exits == 0
    assert engine.total_movement_events == 0
    assert engine.net_flow == 0

    # Also test from_config
    cfg = {"bucket_minutes": 10, "deduplicate_events": False}
    from_cfg = FootfallAnalyticsEngine.from_config(cfg)
    assert from_cfg.bucket_seconds == 600.0
    assert from_cfg.deduplicate_events is False


def test_empty_input(default_footfall_engine: FootfallAnalyticsEngine):
    """Test 2: Empty input results in zero totals and empty buckets."""
    res = default_footfall_engine.update([], frame_number=1, timestamp=0.0)
    assert isinstance(res, FootfallResult)
    assert res.total_entries == 0
    assert res.total_exits == 0
    assert res.total_movement_events == 0
    assert res.net_flow == 0
    assert len(res.summary.buckets) == 0

    res_none = default_footfall_engine.update(None, frame_number=2, timestamp=1.0)
    assert res_none.total_movement_events == 0


def test_entry_aggregation(default_footfall_engine: FootfallAnalyticsEngine):
    """Test 3: Three valid ENTRY events produce total_entries = 3."""
    events = [
        EntryExitEvent(event_id=1, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=1, timestamp=5.0),
        EntryExitEvent(event_id=2, track_id=2, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=2, timestamp=10.0),
        EntryExitEvent(event_id=3, track_id=3, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=3, timestamp=15.0),
    ]
    res = default_footfall_engine.update(events)
    assert res.total_entries == 3
    assert res.total_exits == 0
    assert res.total_movement_events == 3
    assert res.net_flow == 3


def test_exit_aggregation(default_footfall_engine: FootfallAnalyticsEngine):
    """Test 4: Two valid EXIT events produce total_exits = 2."""
    events = [
        EntryExitEvent(event_id=1, track_id=1, movement_type="EXIT", previous_zone_id="entrance", current_zone_id=None, frame_number=1, timestamp=5.0),
        EntryExitEvent(event_id=2, track_id=2, movement_type="EXIT", previous_zone_id="entrance", current_zone_id=None, frame_number=2, timestamp=10.0),
    ]
    res = default_footfall_engine.update(events)
    assert res.total_entries == 0
    assert res.total_exits == 2
    assert res.total_movement_events == 2
    assert res.net_flow == -2


def test_net_flow(default_footfall_engine: FootfallAnalyticsEngine):
    """Test 5: Three entries and two exits produce net_flow = 1."""
    events = [
        EntryExitEvent(event_id=1, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=1, timestamp=5.0),
        EntryExitEvent(event_id=2, track_id=2, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=2, timestamp=10.0),
        EntryExitEvent(event_id=3, track_id=3, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=3, timestamp=15.0),
        EntryExitEvent(event_id=4, track_id=1, movement_type="EXIT", previous_zone_id="entrance", current_zone_id=None, frame_number=4, timestamp=20.0),
        EntryExitEvent(event_id=5, track_id=2, movement_type="EXIT", previous_zone_id="entrance", current_zone_id=None, frame_number=5, timestamp=25.0),
    ]
    res = default_footfall_engine.update(events)
    assert res.total_entries == 3
    assert res.total_exits == 2
    assert res.total_movement_events == 5
    assert res.net_flow == 1


def test_zone_transition_exclusion(default_footfall_engine: FootfallAnalyticsEngine):
    """Test 6: ZONE_TRANSITION is excluded from footfall entries and exits."""
    events = [
        EntryExitEvent(event_id=1, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=1, timestamp=5.0),
        EntryExitEvent(event_id=2, track_id=1, movement_type="ZONE_TRANSITION", previous_zone_id="entrance", current_zone_id="waiting_area", frame_number=2, timestamp=10.0),
        EntryExitEvent(event_id=3, track_id=2, movement_type="ZONE_TRANSITION", previous_zone_id="entrance", current_zone_id="waiting_area", frame_number=3, timestamp=15.0),
    ]
    res = default_footfall_engine.update(events)
    assert res.total_entries == 1
    assert res.total_exits == 0
    assert res.total_movement_events == 1
    assert res.net_flow == 1


def test_duplicate_event(default_footfall_engine: FootfallAnalyticsEngine):
    """Test 7: Duplicate event with same event_id is counted only once."""
    ev = EntryExitEvent(event_id=101, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=1, timestamp=5.0)

    # First presentation
    res1 = default_footfall_engine.update([ev], frame_number=1)
    assert res1.total_entries == 1

    # Second presentation of same event_id
    res2 = default_footfall_engine.update([ev], frame_number=2)
    assert res2.total_entries == 1  # Deduplicated!
    assert res2.new_entries == 0


def test_legitimate_re_entry(default_footfall_engine: FootfallAnalyticsEngine):
    """Test 8: Same track with distinct event IDs is counted legitimately twice."""
    ev1 = EntryExitEvent(event_id=201, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=10, timestamp=5.0)
    ev2 = EntryExitEvent(event_id=202, track_id=1, movement_type="EXIT", previous_zone_id="entrance", current_zone_id=None, frame_number=20, timestamp=15.0)
    ev3 = EntryExitEvent(event_id=203, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=30, timestamp=25.0)

    res = default_footfall_engine.update([ev1, ev2, ev3])
    assert res.total_entries == 2
    assert res.total_exits == 1
    assert res.total_movement_events == 3
    assert res.net_flow == 1


def test_time_bucket_assignment():
    """Test 9: Bucket boundary assignment: start, middle, and end boundary."""
    # 60-second bucket: [0.0, 60.0), [60.0, 120.0)
    engine = FootfallAnalyticsEngine(bucket_seconds=60.0)

    ev_start = EntryExitEvent(event_id=1, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=1, timestamp=0.0)
    ev_mid = EntryExitEvent(event_id=2, track_id=2, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=2, timestamp=30.0)
    ev_edge = EntryExitEvent(event_id=3, track_id=3, movement_type="EXIT", previous_zone_id="entrance", current_zone_id=None, frame_number=3, timestamp=59.99)
    ev_next = EntryExitEvent(event_id=4, track_id=4, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=4, timestamp=60.0)

    res = engine.update([ev_start, ev_mid, ev_edge, ev_next])
    summary = res.summary
    assert len(summary.buckets) == 2

    # Bucket 0: [0.0, 60.0)
    b0 = summary.buckets[0]
    assert b0.bucket_start == 0.0 and b0.bucket_end == 60.0
    assert b0.entries == 2  # ev_start, ev_mid
    assert b0.exits == 1    # ev_edge
    assert b0.total_movement_events == 3

    # Bucket 1: [60.0, 120.0)
    b1 = summary.buckets[1]
    assert b1.bucket_start == 60.0 and b1.bucket_end == 120.0
    assert b1.entries == 1  # ev_next
    assert b1.exits == 0


def test_multiple_buckets():
    """Test 10: Events across multiple intervals map to their respective buckets."""
    engine = FootfallAnalyticsEngine(bucket_seconds=10.0)
    events = [
        EntryExitEvent(event_id=1, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=1, timestamp=2.0),
        EntryExitEvent(event_id=2, track_id=2, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=2, timestamp=12.0),
        EntryExitEvent(event_id=3, track_id=3, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=3, timestamp=25.0),
    ]
    res = engine.update(events)
    buckets = res.summary.buckets
    assert len(buckets) == 3
    assert buckets[0].bucket_start == 0.0
    assert buckets[1].bucket_start == 10.0
    assert buckets[2].bucket_start == 20.0


def test_invalid_timestamp(default_footfall_engine: FootfallAnalyticsEngine):
    """Test 11: Invalid or missing timestamps do not crash or create erroneous buckets."""
    events = [
        EntryExitEvent(event_id=1, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=1, timestamp=None),
        EntryExitEvent(event_id=2, track_id=2, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=2, timestamp=-10.0),
        EntryExitEvent(event_id=3, track_id=3, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=3, timestamp=float("nan")),
        EntryExitEvent(event_id=4, track_id=4, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=4, timestamp=15.0),
    ]
    res = default_footfall_engine.update(events)
    # All 4 are accepted in totals
    assert res.total_entries == 4
    # Only valid timestamp (15.0) creates a bucket
    assert len(res.summary.buckets) == 1
    assert res.summary.buckets[0].entries == 1


def test_invalid_event_type():
    """Test 12: Invalid movement type is rejected cleanly."""
    # With non-strict validation (default)
    engine = FootfallAnalyticsEngine(strict_validation=False)
    ev_bad = {"event_id": 999, "track_id": 1, "movement_type": "UNKNOWN_ACTION", "timestamp": 10.0}
    res = engine.update([ev_bad])
    assert res.total_movement_events == 0

    # With strict validation
    engine_strict = FootfallAnalyticsEngine(strict_validation=True)
    with pytest.raises(InvalidFootfallEventError):
        engine_strict.update([ev_bad])


def test_out_of_order_events():
    """Test 13: Out-of-order event timestamps still yield accurate totals and bucket assignments."""
    engine = FootfallAnalyticsEngine(bucket_seconds=50.0)
    # Event at t=80 arrives before event at t=20
    ev_later = EntryExitEvent(event_id=1, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=10, timestamp=80.0)
    ev_earlier = EntryExitEvent(event_id=2, track_id=2, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=11, timestamp=20.0)

    res = engine.update([ev_later, ev_earlier])
    assert res.total_entries == 2
    buckets = res.summary.buckets
    assert len(buckets) == 2
    # Buckets remain sorted deterministically by time
    assert buckets[0].bucket_start == 0.0 and buckets[0].entries == 1
    assert buckets[1].bucket_start == 50.0 and buckets[1].entries == 1


def test_config_validation():
    """Test 14: Invalid bucket size (<= 0, non-numeric) is rejected."""
    with pytest.raises(InvalidFootfallConfigError):
        FootfallAnalyticsEngine(bucket_seconds=0)

    with pytest.raises(InvalidFootfallConfigError):
        FootfallAnalyticsEngine(bucket_seconds=-15)

    with pytest.raises(InvalidFootfallConfigError):
        FootfallAnalyticsEngine(bucket_minutes="not_a_number")

    with pytest.raises(InvalidFootfallConfigError):
        FootfallAnalyticsEngine.from_config("not_a_dict")


def test_multiple_entrances():
    """Test 15: Breakdown is tracked accurately by entrance/transition ID."""
    engine = FootfallAnalyticsEngine(bucket_seconds=60.0)
    events = [
        EntryExitEvent(event_id=1, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=1, timestamp=5.0, transition_id="main_entrance"),
        EntryExitEvent(event_id=2, track_id=2, movement_type="ENTRY", previous_zone_id=None, current_zone_id="side_entrance", frame_number=2, timestamp=10.0, transition_id="side_entrance"),
        EntryExitEvent(event_id=3, track_id=1, movement_type="EXIT", previous_zone_id="entrance", current_zone_id=None, frame_number=3, timestamp=20.0, transition_id="main_entrance"),
    ]
    res = engine.update(events)
    breakdown = res.summary.entrance_breakdown

    assert "main_entrance" in breakdown
    assert breakdown["main_entrance"]["entries"] == 1
    assert breakdown["main_entrance"]["exits"] == 1
    assert breakdown["main_entrance"]["net_flow"] == 0

    assert "side_entrance" in breakdown
    assert breakdown["side_entrance"]["entries"] == 1
    assert breakdown["side_entrance"]["exits"] == 0
    assert breakdown["side_entrance"]["net_flow"] == 1


def test_reset_behavior(default_footfall_engine: FootfallAnalyticsEngine):
    """Test 16: reset() clears totals, buckets, and deduplication state."""
    ev = EntryExitEvent(event_id=1, track_id=1, movement_type="ENTRY", previous_zone_id=None, current_zone_id="entrance", frame_number=1, timestamp=5.0)
    default_footfall_engine.update([ev])
    assert default_footfall_engine.total_entries == 1

    default_footfall_engine.reset()
    assert default_footfall_engine.total_entries == 0
    assert default_footfall_engine.total_exits == 0
    assert default_footfall_engine.net_flow == 0
    assert len(default_footfall_engine.get_summary().buckets) == 0

    # Can accept event_id 1 again after reset
    res = default_footfall_engine.update([ev])
    assert res.total_entries == 1


def test_integration_with_entry_exit_engine():
    """Test 17: Phase 8 EntryExitResult fed directly into FootfallAnalyticsEngine."""
    ee_engine = EntryExitEngine(allow_initial_entry=True)
    ff_engine = FootfallAnalyticsEngine(bucket_seconds=60.0)

    # Frame 1: Track 1 at entrance (initial entry allowed in this test fixture)
    t1 = TrackZoneAssignment(track_id=1, position=[150, 200], previous_zone_id=None, zone_id="entrance", zone_name="Entrance", transition=None)
    ee_res1 = ee_engine.update([t1], frame_number=1, timestamp=0.033)
    ff_res1 = ff_engine.update(ee_res1)

    assert ff_res1.total_entries == 1
    assert ff_res1.total_exits == 0

    # Frame 2: Track 1 leaves entrance -> outside
    t1_out = TrackZoneAssignment(track_id=1, position=[800, 200], previous_zone_id="entrance", zone_id=None, zone_name=None, transition=None)
    ee_res2 = ee_engine.update([t1_out], frame_number=2, timestamp=0.066)
    ff_res2 = ff_engine.update(ee_res2)

    assert ff_res2.total_entries == 1
    assert ff_res2.total_exits == 1
    assert ff_res2.net_flow == 0
    assert ff_res2.total_movement_events == 2


def test_video_integration_full_pipeline():
    """Test 18: Full pipeline VideoReader -> YOLO -> ByteTrack -> StateManager -> ZoneEngine -> ZoneCounting -> EntryExit -> Footfall."""
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
    entry_exit_engine = EntryExitEngine(definitions=definitions, allow_initial_entry=True)
    footfall_engine = FootfallAnalyticsEngine(bucket_seconds=60.0)

    frames_tested = 0

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
            ff_res = footfall_engine.update(ee_res, frame.frame_number, frame.timestamp)

            assert isinstance(ff_res, FootfallResult)
            assert ff_res.frame_number == frame.frame_number

    finally:
        reader.release()

    assert frames_tested == 35
    summary = footfall_engine.get_summary()
    assert summary.total_entries >= 1
