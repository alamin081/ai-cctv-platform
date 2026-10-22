"""
Base abstract class for video readers.
Enforces interface consistency across file, RTSP, and future input sources.
"""

from abc import ABC, abstractmethod
from typing import Optional, Iterator
from .models import Frame, VideoMetadata


class BaseVideoReader(ABC):
    """Abstract Base Class for all video input readers."""

    @abstractmethod
    def open(self) -> None:
        """Open and initialize the video stream."""
        pass

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """
        Read the next frame from the video stream.

        Returns:
            Frame: The next frame container if available.
            None: If the stream has ended or no frame could be read.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release allocated video capture resources."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if the video capture stream is currently open."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Optional[VideoMetadata]:
        """Return video metadata if stream is open, otherwise None."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Frames per second of the video source."""
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """Frame pixel width."""
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """Frame pixel height."""
        pass

    @property
    @abstractmethod
    def total_frames(self) -> int:
        """Total frame count in the source (or -1 if live stream)."""
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """Total duration in seconds (or -1.0 if live stream)."""
        pass

    @property
    @abstractmethod
    def current_frame_number(self) -> int:
        """Current frame index counter (1-indexed)."""
        pass

    @property
    @abstractmethod
    def current_timestamp(self) -> float:
        """Current playback timestamp in seconds."""
        pass

    def __enter__(self) -> "BaseVideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __iter__(self) -> Iterator[Frame]:
        """Allows iterating through frames using a for loop."""
        while self.is_opened():
            frame = self.read()
            if frame is None:
                break
            yield frame
