"""
Video file reader implementation for local CCTV video files.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2

from .base import BaseVideoReader
from .models import (
    Frame,
    VideoMetadata,
    VideoNotFoundError,
    VideoOpenError,
    VideoReadError,
)

logger = logging.getLogger(__name__)


class VideoFileReader(BaseVideoReader):
    """
    Reader for local video files (.mp4, .avi, etc.).
    Extracts frame-by-frame image matrices and maintains stream state.
    """

    def __init__(self, file_path: Union[str, Path]) -> None:
        """
        Initialize the video file reader.

        Args:
            file_path: Path to the target video file.
        """
        self._file_path = Path(file_path).resolve()
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._current_frame_number: int = 0
        self._current_timestamp: float = 0.0
        self._is_opened: bool = False

    @property
    def file_path(self) -> Path:
        """Absolute path to the video file."""
        return self._file_path

    @property
    def metadata(self) -> Optional[VideoMetadata]:
        """Video metadata container if open, else None."""
        return self._metadata

    @property
    def fps(self) -> float:
        """Frames per second of the video source."""
        return self._metadata.fps if self._metadata else 0.0

    @property
    def width(self) -> int:
        """Frame pixel width."""
        return self._metadata.width if self._metadata else 0

    @property
    def height(self) -> int:
        """Frame pixel height."""
        return self._metadata.height if self._metadata else 0

    @property
    def total_frames(self) -> int:
        """Total frame count in the source."""
        return self._metadata.total_frames if self._metadata else 0

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self._metadata.duration_seconds if self._metadata else 0.0

    @property
    def current_frame_number(self) -> int:
        """Current frame counter (1-indexed)."""
        return self._current_frame_number

    @property
    def current_timestamp(self) -> float:
        """Current timestamp in seconds."""
        return self._current_timestamp

    def is_opened(self) -> bool:
        """Return True if video stream is currently open and valid."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    def open(self) -> None:
        """
        Open the video file and parse video metadata.

        Raises:
            VideoNotFoundError: If the file does not exist or is not a file.
            VideoOpenError: If OpenCV fails to open the video file or codec is invalid.
        """
        if not self._file_path.exists():
            error_msg = f"Video file not found at path: {self._file_path}"
            logger.error(error_msg)
            raise VideoNotFoundError(error_msg)

        if not self._file_path.is_file():
            error_msg = f"Path is not a regular file: {self._file_path}"
            logger.error(error_msg)
            raise VideoNotFoundError(error_msg)

        logger.info("Opening video file: %s", self._file_path)
        self._cap = cv2.VideoCapture(str(self._file_path))

        if not self._cap.isOpened():
            error_msg = f"Failed to open video file (unsupported codec or corrupted): {self._file_path}"
            logger.error(error_msg)
            self.release()
            raise VideoOpenError(error_msg)

        # Parse properties
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            error_msg = f"Invalid video dimensions ({width}x{height}) in file: {self._file_path}"
            logger.error(error_msg)
            self.release()
            raise VideoOpenError(error_msg)

        # Handle zero or negative FPS gracefully (fallback to default 30.0 if corrupted header)
        if fps <= 0:
            logger.warning("Invalid FPS (%.2f) detected in header. Falling back to 30.0 FPS.", fps)
            fps = 30.0

        duration_seconds = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0

        self._metadata = VideoMetadata(
            source_path=self._file_path,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration_seconds=duration_seconds,
        )

        self._is_opened = True
        self._current_frame_number = 0
        self._current_timestamp = 0.0
        logger.info("Successfully opened video: %s", self._metadata)

    def read(self) -> Optional[Frame]:
        """
        Read the next frame from the video file.

        Returns:
            Frame: Next frame object containing frame data and timestamps.
            None: When end of stream is reached or video is closed.

        Raises:
            VideoOpenError: If read is attempted before calling open().
        """
        if not self.is_opened():
            raise VideoOpenError("Cannot read from video file that is not opened. Call open() first.")

        try:
            ret, frame_bgr = self._cap.read()
        except cv2.error as e:
            logger.error("OpenCV internal exception while reading frame: %s", e)
            raise VideoReadError(f"OpenCV error during frame read: {e}") from e

        if not ret or frame_bgr is None:
            logger.info(
                "End of video stream reached at frame %d of %d.",
                self._current_frame_number,
                self.total_frames,
            )
            return None

        self._current_frame_number += 1
        self._current_timestamp = (
            self._current_frame_number / self.fps if self.fps > 0 else 0.0
        )

        return Frame(
            frame_number=self._current_frame_number,
            timestamp=self._current_timestamp,
            data=frame_bgr,
        )

    def seek(self, frame_number: int) -> bool:
        """
        Seek video reader position to a specific frame number (1-indexed).

        Args:
            frame_number: 1-indexed target frame number.

        Returns:
            bool: True if seek command was accepted by OpenCV.

        Raises:
            VideoOpenError: If seek is attempted on an unopened reader.
        """
        if not self.is_opened():
            raise VideoOpenError("Cannot seek in video file that is not opened. Call open() first.")

        target_idx = max(0, min(max(0, self.total_frames - 1), frame_number - 1))
        success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_idx))
        self._current_frame_number = target_idx
        self._current_timestamp = (target_idx / self.fps) if self.fps > 0 else 0.0
        return bool(success)

    def seek_timestamp(self, timestamp: float) -> bool:
        """
        Seek video reader position to a specific timestamp in seconds.

        Args:
            timestamp: Target timestamp in seconds.

        Returns:
            bool: True if seek was successful.
        """
        if self.fps <= 0:
            return False
        target_frame = max(1, int(round(max(0.0, timestamp) * self.fps)) + 1)
        return self.seek(target_frame)

    def release(self) -> None:
        """Release the video file capture resource cleanly."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing cv2.VideoCapture: %s", e)
            self._cap = None

        self._is_opened = False
        logger.info("Video capture released for: %s", self._file_path)
