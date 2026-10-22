"""
Unit tests for ZoneEngine and spatial track-to-zone membership and transitions.
Phase: Phase 6 - Zone Engine
"""

from pathlib import Path
import pytest

from src.detection import Detection, YOLOPersonDetector
from src.tracking import (
    ByteTrackPersonTracker,
    Track,
    TrackState,
    TrackStateManager,
    TrackStateResult,
)
from src.video_input import VideoFileReader
from src.zones import (
    DuplicateZoneIdError,
    InvalidPolygonError,
    InvalidZoneConfigError,
    TrackZoneAssignment,
    Zone,
    ZoneEngine,
    ZoneResult,
)


@pytest.fixture
def two_zones_config() -> list:
    """Fixture returning definitions for two adjacent rectangular zones."""
    return [
        {
            "id": "entrance",
            "name": "Entrance",
            "polygon": [
                [0, 0],
                [320, 0],
                [320, 480],
                [0, 480],
            ],
        },
        {
            "id": "waiting_area",
            "name": "Waiting Area",
            "polygon": [
                [320, 0],
                [640, 0],
                [640, 480],
                [320, 480],
            ],
        },
    ]


@pytest.fixture
def zone_engine(two_zones_config: list) -> ZoneEngine:
    """Fixture providing an initialized ZoneEngine with two configured zones."""
    return ZoneEngine(zones=two_zones_config, assignment_mode="center", boundary_policy="inside")


def test_zone_initialization(two_zones_config: list):
    """Test 1: Valid zone configuration initializes successfully via constructor and from_config."""
    ze = ZoneEngine(zones=two_zones_config, assignment_mode="center", boundary_policy="inside")
    assert len(ze.zones) == 2
    assert ze.assignment_mode == "center"
    assert ze.boundary_policy == "inside"

    z1 = ze.get_zone("entrance")
    assert z1 is not None
    assert z1.zone_id == "entrance"
    assert z1.zone_name == "Entrance"
    assert len(z1.polygon) == 4
    assert z1.contour.shape == (4, 1, 2)

    z2 = ze.get_zone("waiting_area")
    assert z2 is not None
    assert z2.zone_id == "waiting_area"

    # Test initialization via from_config helper
    cfg = {
        "assignment_mode": "bottom_center",
        "boundary_policy": "outside",
        "definitions": two_zones_config,
    }
    ze2 = ZoneEngine.from_config(cfg)
    assert ze2.assignment_mode == "bottom_center"
    assert ze2.boundary_policy == "outside"
    assert len(ze2.zones) == 2


def test_invalid_zone_id():
    """Test 2: Missing or empty zone_id raises InvalidZoneConfigError."""
    # Empty string
    with pytest.raises(InvalidZoneConfigError):
        Zone(zone_id="", zone_name="Entrance", polygon=[[0, 0], [10, 0], [10, 10]])

    # Whitespace only
    with pytest.raises(InvalidZoneConfigError):
        Zone(zone_id="   ", zone_name="Entrance", polygon=[[0, 0], [10, 0], [10, 10]])

    # None or non-string
    with pytest.raises(InvalidZoneConfigError):
        Zone(zone_id=None, zone_name="Entrance", polygon=[[0, 0], [10, 0], [10, 10]])

    # Empty ID via ZoneEngine dict
    with pytest.raises(InvalidZoneConfigError):
        ZoneEngine(zones=[{"id": "", "name": "Invalid", "polygon": [[0, 0], [10, 0], [10, 10]]}])


def test_duplicate_zone_id():
    """Test 3: Duplicate zone IDs raise DuplicateZoneIdError."""
    duplicates = [
        {"id": "zone_a", "name": "Zone A1", "polygon": [[0, 0], [100, 0], [100, 100]]},
        {"id": "zone_a", "name": "Zone A2", "polygon": [[100, 100], [200, 100], [200, 200]]},
    ]
    with pytest.raises(DuplicateZoneIdError):
        ZoneEngine(zones=duplicates)


def test_invalid_polygon():
    """Test 4: Polygons with < 3 points or non-numeric/boolean coordinates raise InvalidPolygonError."""
    # Less than 3 points
    with pytest.raises(InvalidPolygonError):
        Zone(zone_id="bad", zone_name="Bad", polygon=[])

    with pytest.raises(InvalidPolygonError):
        Zone(zone_id="bad", zone_name="Bad", polygon=[[10, 10], [20, 20]])

    # Malformed point dimension (single value instead of [x, y])
    with pytest.raises(InvalidPolygonError):
        Zone(zone_id="bad", zone_name="Bad", polygon=[[10], [20, 30], [30, 40]])

    # Non-numeric string coordinate
    with pytest.raises(InvalidPolygonError):
        Zone(zone_id="bad", zone_name="Bad", polygon=[["abc", 10], [20, 30], [30, 40]])

    # Boolean coordinate (bool is subclass of int in Python)
    with pytest.raises(InvalidPolygonError):
        Zone(zone_id="bad", zone_name="Bad", polygon=[[True, 10], [20, 30], [30, 40]])


def test_point_inside_polygon(zone_engine: ZoneEngine):
    """Test 5: Track position strictly inside a zone polygon assigns correct zone."""
    # Track center: (160, 240) - inside entrance [0,0] to [320,480]
    track = Track(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.90,
        bbox=[110.0, 190.0, 210.0, 290.0],
        state="ACTIVE",
    )
    result = zone_engine.update([track], frame_number=1, timestamp=0.033)

    assert isinstance(result, ZoneResult)
    assert result.count == 1
    t = result.tracks[0]
    assert t.track_id == 1
    assert t.zone_id == "entrance"
    assert t.zone_name == "Entrance"
    assert t.position == [160.0, 240.0]


def test_point_outside_polygon(zone_engine: ZoneEngine):
    """Test 6: Track position outside all zone polygons assigns zone_id = None."""
    # Track center: (800, 240) - outside both 0..320 and 320..640
    track = Track(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.90,
        bbox=[750.0, 190.0, 850.0, 290.0],
        state="ACTIVE",
    )
    result = zone_engine.update([track], frame_number=1, timestamp=0.033)

    assert result.count == 1
    t = result.tracks[0]
    assert t.zone_id is None
    assert t.zone_name is None
    assert t.transition is None


def test_boundary_point():
    """Test 7: Track on polygon boundary behaves according to configured boundary_policy."""
    polygon = [[0, 0], [320, 0], [320, 480], [0, 480]]
    # Point exactly on boundary x=320: center is (320, 240)
    boundary_track = Track(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.90,
        bbox=[310.0, 230.0, 330.0, 250.0],
        state="ACTIVE",
    )

    # 1. boundary_policy = 'inside'
    ze_inside = ZoneEngine(
        zones=[{"id": "entrance", "name": "Entrance", "polygon": polygon}],
        boundary_policy="inside",
    )
    res_inside = ze_inside.update([boundary_track], frame_number=1)
    assert res_inside.tracks[0].zone_id == "entrance"

    # 2. boundary_policy = 'outside'
    ze_outside = ZoneEngine(
        zones=[{"id": "entrance", "name": "Entrance", "polygon": polygon}],
        boundary_policy="outside",
    )
    res_outside = ze_outside.update([boundary_track], frame_number=1)
    assert res_outside.tracks[0].zone_id is None


def test_multiple_zones(zone_engine: ZoneEngine):
    """Test 8: Tracks in different zones receive appropriate zone memberships simultaneously."""
    # Track 1 inside entrance (center: 150, 200)
    # Track 2 inside waiting_area (center: 450, 200)
    # Track 3 outside (center: 750, 200)
    t1 = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[100, 150, 200, 250])
    t2 = Track(track_id=2, class_id=0, class_name="person", confidence=0.9, bbox=[400, 150, 500, 250])
    t3 = Track(track_id=3, class_id=0, class_name="person", confidence=0.9, bbox=[700, 150, 800, 250])

    result = zone_engine.update([t1, t2, t3], frame_number=1)
    assert result.count == 3

    track_map = {t.track_id: t for t in result.tracks}
    assert track_map[1].zone_id == "entrance"
    assert track_map[2].zone_id == "waiting_area"
    assert track_map[3].zone_id is None


def test_zone_transition_entered(zone_engine: ZoneEngine):
    """Test 9: Track moving from entrance to waiting_area detects ENTERED transition."""
    # Frame 1: Track 1 in entrance
    t_f1 = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[100, 150, 200, 250])
    res_f1 = zone_engine.update([t_f1], frame_number=1)
    assert res_f1.tracks[0].zone_id == "entrance"
    assert res_f1.tracks[0].previous_zone_id is None
    assert res_f1.tracks[0].transition == "ENTERED"

    # Frame 2: Track 1 moves into waiting_area
    t_f2 = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[400, 150, 500, 250])
    res_f2 = zone_engine.update([t_f2], frame_number=2)
    assert res_f2.tracks[0].previous_zone_id == "entrance"
    assert res_f2.tracks[0].zone_id == "waiting_area"
    assert res_f2.tracks[0].transition == "ENTERED"


def test_zone_leave(zone_engine: ZoneEngine):
    """Test 10: Track moving from waiting_area to outside detects LEFT transition."""
    # Frame 1: In waiting_area
    t_f1 = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[400, 150, 500, 250])
    res_f1 = zone_engine.update([t_f1], frame_number=1)
    assert res_f1.tracks[0].zone_id == "waiting_area"

    # Frame 2: Moves outside (center: 750, 200)
    t_f2 = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[700, 150, 800, 250])
    res_f2 = zone_engine.update([t_f2], frame_number=2)
    assert res_f2.tracks[0].previous_zone_id == "waiting_area"
    assert res_f2.tracks[0].zone_id is None
    assert res_f2.tracks[0].transition == "LEFT"


def test_same_zone_no_transition(zone_engine: ZoneEngine):
    """Test 11: Track remaining in the same zone has transition = None."""
    # Frame 1: In entrance
    t_f1 = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[100, 150, 200, 250])
    res_f1 = zone_engine.update([t_f1], frame_number=1)
    assert res_f1.tracks[0].transition == "ENTERED"

    # Frame 2: Still in entrance (slightly moved: center 160, 210)
    t_f2 = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[110, 160, 210, 260])
    res_f2 = zone_engine.update([t_f2], frame_number=2)
    assert res_f2.tracks[0].previous_zone_id == "entrance"
    assert res_f2.tracks[0].zone_id == "entrance"
    assert res_f2.tracks[0].transition is None


def test_track_removed_cleanup(zone_engine: ZoneEngine):
    """Test 12: Track transitions ACTIVE -> LOST (preserved) -> REMOVED (cleaned up without memory leak)."""
    # Frame 1: ACTIVE in entrance
    t_active = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[100, 150, 200, 250], state="ACTIVE")
    zone_engine.update([t_active], frame_number=1)
    assert zone_engine.get_track_zone(1) == "entrance"

    # Frame 2: LOST - preserves last known zone state
    t_lost = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[100, 150, 200, 250], state="LOST")
    zone_engine.update([t_lost], frame_number=2)
    assert zone_engine.get_track_zone(1) == "entrance"

    # Frame 3: REMOVED - purges internal state
    t_removed = Track(track_id=1, class_id=0, class_name="person", confidence=0.9, bbox=[100, 150, 200, 250], state="REMOVED")
    zone_engine.update([t_removed], frame_number=3)
    assert zone_engine.get_track_zone(1) is None
    assert 1 not in zone_engine.active_track_zones


def test_multiple_tracks(zone_engine: ZoneEngine):
    """Test 13: Multiple tracks processed independently: ID 1 -> Zone A, ID 2 -> Zone B, ID 3 -> outside."""
    t1 = Track(track_id=1, class_id=0, class_name="person", confidence=0.88, bbox=[50, 100, 150, 200])    # center: (100, 150) -> entrance
    t2 = Track(track_id=2, class_id=0, class_name="person", confidence=0.91, bbox=[400, 100, 500, 200])  # center: (450, 150) -> waiting_area
    t3 = Track(track_id=3, class_id=0, class_name="person", confidence=0.79, bbox=[800, 100, 900, 200])  # center: (850, 150) -> outside

    res = zone_engine.update([t1, t2, t3], frame_number=10)
    assert res.count == 3
    mapping = {t.track_id: t.zone_id for t in res.tracks}
    assert mapping[1] == "entrance"
    assert mapping[2] == "waiting_area"
    assert mapping[3] is None


def test_empty_tracks(zone_engine: ZoneEngine):
    """Test 14: Empty track list returns valid empty ZoneResult without errors."""
    res_empty = zone_engine.update([], frame_number=5, timestamp=0.16)
    assert isinstance(res_empty, ZoneResult)
    assert res_empty.count == 0
    assert res_empty.tracks == []
    assert len(res_empty.zones) == 2
    assert res_empty.frame_number == 5

    res_none = zone_engine.update(None, frame_number=6, timestamp=0.20)
    assert res_none.count == 0
    assert res_none.tracks == []


def test_video_integration_pipeline():
    """Test 15: Full pipeline integration on test video: VideoReader -> YOLO -> ByteTrack -> StateManager -> ZoneEngine."""
    test_video_path = Path(__file__).resolve().parent.parent / "data" / "videos" / "test.mp4"
    if not test_video_path.exists():
        pytest.skip(f"Test video not found at {test_video_path}")

    model_path = Path(__file__).resolve().parent.parent / "data" / "models" / "yolov8n.pt"
    if not model_path.exists():
        model_path = Path("yolov8n.pt")

    reader = VideoFileReader(test_video_path)
    reader.open()

    detector = YOLOPersonDetector(model_name_or_path=model_path, confidence_threshold=0.35)
    tracker = ByteTrackPersonTracker(track_high_thresh=0.4, new_track_thresh=0.4)
    state_manager = TrackStateManager(max_missed_frames=15)

    zones_cfg = [
        {"id": "left_half", "name": "Left Half", "polygon": [[0, 0], [320, 0], [320, 480], [0, 480]]},
        {"id": "right_half", "name": "Right Half", "polygon": [[320, 0], [640, 0], [640, 480], [320, 480]]},
    ]
    zone_engine = ZoneEngine(zones=zones_cfg)

    frames_tested = 0
    total_zone_assignments = 0

    try:
        # Run across first 30 frames
        for _ in range(30):
            frame = reader.read()
            if frame is None:
                break
            frames_tested += 1

            det_result = detector.detect(frame)
            track_result = tracker.update(det_result, frame.frame_number, frame.timestamp)
            state_result = state_manager.update(track_result, frame.frame_number, frame.timestamp)
            zone_result = zone_engine.update(state_result, frame.frame_number, frame.timestamp)

            assert isinstance(zone_result, ZoneResult)
            assert zone_result.frame_number == frame.frame_number
            assert len(zone_result.zones) == 2

            total_zone_assignments += zone_result.count
            for zt in zone_result.tracks:
                assert zt.track_id in [t.track_id for t in state_result.tracks]
                assert zt.zone_id in ("left_half", "right_half", None)

    finally:
        reader.release()

    assert frames_tested == 30
    assert total_zone_assignments > 0
