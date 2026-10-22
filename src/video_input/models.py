"""
Models and exceptions for video input handling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Optional


class VideoInputError(Exception):
    """Base exception for all video input related errors."""
    pass


class VideoNotFoundError(VideoInputError):
    """Raised when the specified video file path does not exist."""
    pass


class VideoOpenError(VideoInputError):
    """Raised when the video file cannot be opened (corrupted, unsupported codec, etc.)."""
    pass


class VideoReadError(VideoInputError):
    """Raised when an error occurs while reading a frame from the video stream."""
    pass


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable metadata representing properties of a video source."""
    source_path: Path
    fps: float
    width: int
    height: int
    total_frames: int
    duration_seconds: float

    @property
    def resolution(self) -> Tuple[int, int]:
        """Returns resolution as (width, height)."""
        return self.width, self.height

    def __str__(self) -> str:
        minutes, seconds = divmod(self.duration_seconds, 60)
        return (
            f"VideoMetadata(source='{self.source_path}', "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.fps:.2f}, "
            f"frames={self.total_frames}, "
            f"duration={int(minutes):02d}:{seconds:05.2f})"
        )


@dataclass
class Frame:
    """Represents a single video frame with metadata and raw image data."""
    frame_number: int
    timestamp: float
    data: Any  # numpy.ndarray (typed Any to allow clean decoupling before cv2 loads)

    @property
    def height(self) -> int:
        """Frame pixel height."""
        return self.data.shape[0] if self.data is not None else 0

    @property
    def width(self) -> int:
        """Frame pixel width."""
        return self.data.shape[1] if self.data is not None else 0

    @property
    def shape(self) -> Tuple[int, ...]:
        """Frame dimension shape (H, W, C)."""
        return self.data.shape if self.data is not None else ()
