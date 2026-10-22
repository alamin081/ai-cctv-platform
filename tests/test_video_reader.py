"""
Unit tests for VideoFileReader and video input models.
"""

import tempfile
from pathlib import Path
import pytest

from src.video_input import (
    VideoFileReader,
    VideoNotFoundError,
    VideoOpenError,
    VideoMetadata,
    Frame,
)


@pytest.fixture
def sample_video_path() -> Path:
    """Fixture returning path to the sample test video."""
    video_path = Path(__file__).resolve().parent.parent / "data" / "videos" / "test.mp4"
    assert video_path.exists(), f"Sample test video must exist at {video_path}"
    return video_path


def test_invalid_path_raises_video_not_found():
    """Test that a non-existent video path raises VideoNotFoundError."""
    invalid_path = Path("data/videos/non_existent_file_xyz_123.mp4")
    reader = VideoFileReader(invalid_path)
    with pytest.raises(VideoNotFoundError) as exc_info:
        reader.open()
    assert "Video file not found" in str(exc_info.value)


def test_directory_path_raises_video_not_found():
    """Test that specifying a directory instead of a file raises VideoNotFoundError."""
    dir_path = Path(__file__).resolve().parent
    reader = VideoFileReader(dir_path)
    with pytest.raises(VideoNotFoundError) as exc_info:
        reader.open()
    assert "Path is not a regular file" in str(exc_info.value)


def test_read_before_open_raises_video_open_error():
    """Test that calling read() before open() raises VideoOpenError."""
    fake_path = Path("dummy.mp4")
    reader = VideoFileReader(fake_path)
    with pytest.raises(VideoOpenError) as exc_info:
        reader.read()
    assert "Call open() first" in str(exc_info.value)


def test_corrupted_file_raises_video_open_error():
    """Test that opening an empty/corrupted file raises VideoOpenError."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_file.write(b"NOT A REAL VIDEO CONTENT JUNK DATA")
        tmp_path = Path(tmp_file.name)

    try:
        reader = VideoFileReader(tmp_path)
        with pytest.raises(VideoOpenError):
            reader.open()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_video_initialization_and_metadata(sample_video_path: Path):
    """Test opening a valid video and parsing metadata."""
    reader = VideoFileReader(sample_video_path)
    try:
        reader.open()
        assert reader.is_opened() is True

        metadata = reader.metadata
        assert isinstance(metadata, VideoMetadata)
        assert metadata.width == 640
        assert metadata.height == 480
        assert metadata.fps == 30.0
        assert metadata.total_frames == 90
        assert round(metadata.duration_seconds, 2) == 3.0
        assert metadata.resolution == (640, 480)
    finally:
        reader.release()
        assert reader.is_opened() is False


def test_frame_reading_and_timestamps(sample_video_path: Path):
    """Test reading frames sequentially and checking incrementing counter/timestamps."""
    reader = VideoFileReader(sample_video_path)
    reader.open()

    read_count = 0
    previous_timestamp = -1.0

    while True:
        frame = reader.read()
        if frame is None:
            break

        read_count += 1
        assert isinstance(frame, Frame)
        assert frame.frame_number == read_count
        assert frame.timestamp > previous_timestamp
        assert frame.width == 640
        assert frame.height == 480
        assert frame.shape == (480, 640, 3)
        assert frame.data is not None

        previous_timestamp = frame.timestamp

    assert read_count == 90
    assert reader.current_frame_number == 90
    reader.release()


def test_end_of_video_handling(sample_video_path: Path):
    """Test that end-of-video returns None gracefully on subsequent reads."""
    with VideoFileReader(sample_video_path) as reader:
        # Exhaust all frames
        frames = list(reader)
        assert len(frames) == 90

        # Further read calls should safely return None without errors
        assert reader.read() is None
        assert reader.read() is None


def test_context_manager_behavior(sample_video_path: Path):
    """Test that using 'with VideoFileReader(...)' properly opens and releases."""
    reader = VideoFileReader(sample_video_path)
    assert reader.is_opened() is False

    with reader as r:
        assert r.is_opened() is True
        frame = r.read()
        assert frame is not None
        assert frame.frame_number == 1

    assert reader.is_opened() is False


def test_video_reader_seek(sample_video_path: Path):
    """Test seeking to specific frame positions."""
    reader = VideoFileReader(sample_video_path)
    reader.open()
    try:
        # Seek to frame 45
        assert reader.seek(45) is True
        frame = reader.read()
        assert frame is not None
        assert frame.frame_number == 45
        assert round(frame.timestamp, 2) == round(45 / 30.0, 2)

        # Seek backwards to frame 10
        assert reader.seek(10) is True
        frame = reader.read()
        assert frame is not None
        assert frame.frame_number == 10

        # Seek to beginning
        assert reader.seek(1) is True
        frame = reader.read()
        assert frame is not None
        assert frame.frame_number == 1
    finally:
        reader.release()


def test_video_reader_seek_timestamp(sample_video_path: Path):
    """Test seeking by timestamp in seconds."""
    reader = VideoFileReader(sample_video_path)
    reader.open()
    try:
        # 1.5 seconds at 30 fps corresponds to frame 46
        assert reader.seek_timestamp(1.5) is True
        frame = reader.read()
        assert frame is not None
        assert frame.frame_number == 46
    finally:
        reader.release()

