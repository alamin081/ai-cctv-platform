"""
Tests for Real-Time Playback Analysis Service and Playback API Endpoints.
Verifies session creation, frame processing, instant seeking with state reset,
playback speed stride adjustments, and FastAPI playback routes.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.playback_service import PlaybackService, SharedDetectorManager


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def test_client() -> TestClient:
    return TestClient(app)


def test_shared_detector_singleton(project_root: Path):
    """Verify SharedDetectorManager returns singleton instance and doesn't reload model."""
    config = {"detection": {"confidence": 0.4, "iou": 0.45, "device": "cpu"}}
    det1 = SharedDetectorManager.get_instance(project_root, config)
    det2 = SharedDetectorManager.get_instance(project_root, config)
    assert det1 is det2
    assert det1.detector is not None


def test_playback_session_lifecycle(project_root: Path):
    """Verify session creation, frame processing, and release."""
    service = PlaybackService(project_root)
    session = service.create_or_get_session("test.mp4")

    try:
        assert session.session_id.startswith("play_test_")
        assert session.fps == 30.0
        assert session.total_frames == 90
        assert session.width == 640
        assert session.height == 480
        assert session.current_frame_number == 1

        # Process next frame
        session.set_playing(True)
        frame_data = session.process_next_frame()
        assert frame_data is not None
        assert frame_data["frame_number"] == 2
        assert "tracks" in frame_data
        assert "occupancy" in frame_data
        assert "metrics" in frame_data

    finally:
        service.close_session(session.session_id)


def test_playback_session_seek_resets_state(project_root: Path):
    """
    Verify seek resets tracking & zone state to avoid state carry-over,
    and returns immediate frame analysis at the target position.
    """
    service = PlaybackService(project_root)
    session = service.create_or_get_session("test.mp4")

    try:
        # Advance a few frames
        session.set_playing(True)
        session.process_next_frame()
        session.process_next_frame()

        # Seek to frame 45 (1.5 seconds)
        seek_result = session.seek(timestamp=1.5, frame_number=45)
        assert seek_result is not None
        assert seek_result["frame_number"] == 45
        assert round(seek_result["timestamp"], 2) == 1.5
        assert session.current_frame_number == 45

        # Tracker unique IDs should be fresh for the seeked position (reset called)
        assert len(session.tracker.unique_track_ids) <= 2

    finally:
        service.close_session(session.session_id)


def test_playback_speed_and_stride(project_root: Path):
    """Verify that playback speed dynamically updates stride."""
    service = PlaybackService(project_root)
    session = service.create_or_get_session("test.mp4")

    try:
        session.set_speed(0.5)
        assert session.playback_speed == 0.5
        assert session.stride == 1

        session.set_speed(1.0)
        assert session.playback_speed == 1.0

        session.set_speed(2.0)
        assert session.playback_speed == 2.0
        assert session.stride >= 2

    finally:
        service.close_session(session.session_id)


def test_api_playback_session_endpoints(test_client: TestClient):
    """Test full HTTP API lifecycle for playback sessions."""
    # 1. Create session
    res = test_client.post("/api/playback/session", json={"video_filename": "test.mp4"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    session_id = data["session_id"]
    assert data["fps"] == 30.0
    assert data["total_frames"] == 90
    assert "zones" in data
    assert "initial_frame" in data

    # 2. Play
    res = test_client.post(f"/api/playback/{session_id}/play", json={"timestamp": 0.0, "speed": 1.0})
    assert res.status_code == 200
    assert res.json()["status"] == "playing"

    # 3. Pause
    res = test_client.post(f"/api/playback/{session_id}/pause", json={"timestamp": 0.5})
    assert res.status_code == 200
    assert res.json()["status"] == "paused"

    # 4. Seek
    res = test_client.post(f"/api/playback/{session_id}/seek", json={"timestamp": 1.0, "frame_number": 30})
    assert res.status_code == 200
    seek_data = res.json()
    assert seek_data["status"] == "seeked"
    assert seek_data["frame_number"] == 30
    assert "frame_data" in seek_data

    # 5. Process frame endpoint
    res = test_client.post(
        f"/api/playback/{session_id}/process_frame",
        json={"timestamp": 1.033, "frame_number": 31},
    )
    assert res.status_code == 200
    pdata = res.json()
    assert pdata["status"] == "success"
    assert pdata["frame_number"] == 31
    assert "frame_data" in pdata
    assert "tracks" in pdata["frame_data"]
    assert "metrics" in pdata

    # 6. Sync
    res = test_client.post(f"/api/playback/{session_id}/sync", json={"current_time": 1.0, "is_playing": True, "speed": 1.0})
    assert res.status_code == 200
    assert res.json()["status"] == "synced"

    # 7. Delete
    res = test_client.delete(f"/api/playback/{session_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "closed"


def test_tracking_continuity_across_frames(project_root: Path):
    """Verify that sequential process_frame_at calls maintain tracking continuity and stable track IDs."""
    service = PlaybackService(project_root)
    session = service.create_or_get_session("test.mp4")

    try:
        # Seek to start
        session.seek(timestamp=0.0, frame_number=1)

        # Process frames 2, 3, 4 sequentially
        f2 = session.process_frame_at(timestamp=0.033, frame_number=2)
        f3 = session.process_frame_at(timestamp=0.066, frame_number=3)
        f4 = session.process_frame_at(timestamp=0.100, frame_number=4)

        assert f2 is not None and f3 is not None and f4 is not None
        assert f2["frame_number"] == 2
        assert f3["frame_number"] == 3
        assert f4["frame_number"] == 4

        # Track IDs should not be reset between adjacent frames
        if f3["tracks"] and f4["tracks"]:
            f3_ids = {t["track_id"] for t in f3["tracks"]}
            f4_ids = {t["track_id"] for t in f4["tracks"]}
            # At least one track should persist across consecutive frames
            assert len(f3_ids.intersection(f4_ids)) > 0 or len(f3_ids) == 0

    finally:
        service.close_session(session.session_id)



def test_api_upload_does_not_block_for_ai(test_client: TestClient, tmp_path: Path):
    """Verify that video upload immediately returns metadata and does not execute full AI analysis."""
    import numpy as np
    import cv2

    test_file = tmp_path / "quick_test_upload.mp4"
    writer = cv2.VideoWriter(str(test_file), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for _ in range(15):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with open(test_file, "rb") as f:
        res = test_client.post("/api/upload", files={"file": ("quick_test_upload.mp4", f, "video/mp4")})

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["filename"].startswith("quick_test_upload")
    assert "video_url" in data
    assert data["width"] == 320
    assert data["height"] == 240
    assert data["total_frames"] == 15
