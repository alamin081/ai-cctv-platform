"""
Multi-Object Tracking and Track State Management module.
Phase 4: ByteTrack Multi-Object Tracking
Phase 5: Track State Manager
"""

from .models import (
    InvalidStateManagerInputError,
    InvalidTrackerInputError,
    Track,
    TrackerConfigError,
    TrackingError,
    TrackingResult,
    TrackState,
    TrackStateResult,
)
from .state_manager import TrackStateManager
from .tracker import ByteTrackPersonTracker

__all__ = [
    "ByteTrackPersonTracker",
    "TrackStateManager",
    "Track",
    "TrackingResult",
    "TrackStateResult",
    "TrackState",
    "TrackingError",
    "InvalidTrackerInputError",
    "TrackerConfigError",
    "InvalidStateManagerInputError",
]
