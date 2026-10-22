"""
Video Analytics Module.
Phases:
- Phase 6: Zone Engine
- Phase 7: Zone Counting / Occupancy Engine
- Phase 8: Entry / Exit Analytics Engine
- Phase 9: Footfall Analytics Engine
"""

from src.zones import (
    CountingEngineError,
    DuplicateZoneIdError,
    InvalidPolygonError,
    InvalidZoneConfigError,
    TrackZoneAssignment,
    Zone,
    ZoneCountingEngine,
    ZoneCountingResult,
    ZoneEngine,
    ZoneError,
    ZoneEvent,
    ZoneOccupancy,
    ZoneResult,
)
from .entry_exit_engine import EntryExitEngine
from .footfall_engine import FootfallAnalyticsEngine
from .models import (
    EntryExitError,
    EntryExitEvent,
    EntryExitResult,
    FootfallBucket,
    FootfallError,
    FootfallResult,
    FootfallSummary,
    InvalidEntryExitConfigError,
    InvalidFootfallConfigError,
    InvalidFootfallEventError,
    MovementType,
    TrackMovementState,
)

__all__ = [
    # Phase 6 & 7 Exports
    "Zone",
    "ZoneEngine",
    "ZoneCountingEngine",
    "TrackZoneAssignment",
    "ZoneResult",
    "ZoneOccupancy",
    "ZoneEvent",
    "ZoneCountingResult",
    "ZoneError",
    "InvalidZoneConfigError",
    "InvalidPolygonError",
    "DuplicateZoneIdError",
    "CountingEngineError",
    # Phase 8 Exports
    "EntryExitEngine",
    "EntryExitEvent",
    "EntryExitResult",
    "MovementType",
    "TrackMovementState",
    "EntryExitError",
    "InvalidEntryExitConfigError",
    # Phase 9 Exports
    "FootfallAnalyticsEngine",
    "FootfallBucket",
    "FootfallSummary",
    "FootfallResult",
    "FootfallError",
    "InvalidFootfallConfigError",
    "InvalidFootfallEventError",
]
