"""
Zone Engine Module - Spatial Region, Zone Membership, Zone Occupancy Counting, and Zone Config Store.
Phases: Phase 6 - Zone Engine & Phase 7 - Zone Counting / Occupancy Engine
"""

from .counting_engine import ZoneCountingEngine
from .models import (
    CountingEngineError,
    DuplicateZoneIdError,
    InvalidPolygonError,
    InvalidZoneConfigError,
    TrackZoneAssignment,
    Zone,
    ZoneCountingResult,
    ZoneError,
    ZoneEvent,
    ZoneOccupancy,
    ZoneResult,
)
from .zone_engine import ZoneEngine
from .zone_store import ZoneConfigStore

__all__ = [
    "Zone",
    "ZoneEngine",
    "ZoneCountingEngine",
    "ZoneConfigStore",
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
]
