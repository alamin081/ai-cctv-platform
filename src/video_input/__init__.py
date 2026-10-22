"""
Video input ingestion module.
Provides abstract base interfaces and concrete implementations for video streams.
"""

from .base import BaseVideoReader
from .file_reader import VideoFileReader
from .models import (
    Frame,
    VideoMetadata,
    VideoInputError,
    VideoNotFoundError,
    VideoOpenError,
    VideoReadError,
)

__all__ = [
    "BaseVideoReader",
    "VideoFileReader",
    "Frame",
    "VideoMetadata",
    "VideoInputError",
    "VideoNotFoundError",
    "VideoOpenError",
    "VideoReadError",
]
