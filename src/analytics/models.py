"""
Data models and exceptions for Movement and Analytics Engines (Entry/Exit, Footfall).
Phases: Phase 8 - Entry / Exit Analytics Engine & Phase 9 - Footfall Analytics Engine
"""

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional


# ==============================================================================
# Phase 8 — Entry / Exit Models
# ==============================================================================

class MovementType(str, Enum):
    """Classification of spatial track movements."""
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_TRANSITION = "ZONE_TRANSITION"


class EntryExitError(Exception):
    """Base exception for all entry/exit analytics engine errors."""
    pass


class InvalidEntryExitConfigError(EntryExitError):
    """Raised when entry/exit configuration schema or transitions are invalid."""
    pass


@dataclass
class EntryExitEvent:
    """
    Represents a discrete entry, exit, or zone-to-zone movement event.
    Matches Section 9, 10, 20, 21 specification.
    """
    event_id: int
    track_id: int
    movement_type: str  # "ENTRY", "EXIT", "ZONE_TRANSITION"
    previous_zone_id: Optional[str]
    current_zone_id: Optional[str]
    frame_number: int
    timestamp: Optional[float] = None
    transition_id: Optional[str] = None

    @property
    def type(self) -> str:
        """Alias for movement_type matching Section 9-10 schema."""
        return self.movement_type

    @property
    def zone_id(self) -> Optional[str]:
        """Convenience property for zone_id (for ENTRY: current_zone, for EXIT: previous_zone)."""
        if self.movement_type == "ENTRY":
            return self.current_zone_id
        elif self.movement_type == "EXIT":
            return self.previous_zone_id
        return self.current_zone_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "type": self.movement_type,
            "movement_type": self.movement_type,
            "zone_id": self.zone_id,
            "previous_zone_id": self.previous_zone_id,
            "current_zone_id": self.current_zone_id,
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
            "transition_id": self.transition_id,
        }

    def __repr__(self) -> str:
        return (
            f"EntryExitEvent(id={self.event_id}, track={self.track_id}, type={self.movement_type}, "
            f"'{self.previous_zone_id}' -> '{self.current_zone_id}' @ f{self.frame_number})"
        )


@dataclass
class TrackMovementState:
    """Per-track spatial movement and history state."""
    track_id: int
    previous_zone_id: Optional[str]
    current_zone_id: Optional[str]
    last_transition_frame: int
    last_transition_timestamp: float
    total_entries: int = 0
    total_exits: int = 0
    total_transitions: int = 0


@dataclass
class EntryExitResult:
    """Structured output container for frame-level entry/exit analytics."""
    frame_number: int
    timestamp: float
    events: List[EntryExitEvent]
    active_track_movements: Dict[int, Optional[str]]  # track_id -> current_zone_id
    total_entries: int
    total_exits: int
    total_zone_transitions: int
    processing_time_ms: float = 0.0

    @property
    def event_count(self) -> int:
        """Number of events generated in this frame."""
        return len(self.events)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return {
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
            "events": [e.to_dict() for e in self.events],
            "active_tracks": dict(self.active_track_movements),
            "metrics": {
                "total_entries": self.total_entries,
                "total_exits": self.total_exits,
                "total_zone_transitions": self.total_zone_transitions,
            },
        }


# ==============================================================================
# Phase 9 — Footfall Analytics Models
# ==============================================================================

class FootfallError(Exception):
    """Base exception for all footfall analytics engine errors."""
    pass


class InvalidFootfallConfigError(FootfallError):
    """Raised when footfall configuration parameters (e.g. bucket duration) are invalid."""
    pass


class InvalidFootfallEventError(FootfallError):
    """Raised when an event provided to Footfall engine is malformed or invalid."""
    pass


@dataclass
class FootfallBucket:
    """
    Represents aggregated footfall metrics for a discrete time interval bucket.
    Timestamps are video-relative seconds.
    """
    bucket_start: float
    bucket_end: float
    entries: int = 0
    exits: int = 0
    total_movement_events: int = 0
    net_flow: int = 0
    entry_rate: float = 0.0  # entries per minute
    exit_rate: float = 0.0   # exits per minute
    breakdown: Optional[Dict[str, Dict[str, int]]] = None

    def __post_init__(self) -> None:
        if self.breakdown is None:
            self.breakdown = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert bucket metrics to dictionary."""
        return {
            "bucket_start": round(self.bucket_start, 2),
            "bucket_end": round(self.bucket_end, 2),
            "entries": self.entries,
            "exits": self.exits,
            "total_movement_events": self.total_movement_events,
            "net_flow": self.net_flow,
            "entry_rate": round(self.entry_rate, 4),
            "exit_rate": round(self.exit_rate, 4),
            "breakdown": self.breakdown or {},
        }


@dataclass
class FootfallSummary:
    """
    Cumulative in-memory Footfall analytics summary across video playback.
    """
    total_entries: int = 0
    total_exits: int = 0
    total_movement_events: int = 0
    net_flow: int = 0
    buckets: List[FootfallBucket] = None
    entrance_breakdown: Dict[str, Dict[str, int]] = None

    def __post_init__(self) -> None:
        if self.buckets is None:
            self.buckets = []
        if self.entrance_breakdown is None:
            self.entrance_breakdown = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "total_movement_events": self.total_movement_events,
            "net_flow": self.net_flow,
            "buckets": [b.to_dict() for b in self.buckets],
            "entrance_breakdown": self.entrance_breakdown,
        }


@dataclass
class FootfallResult:
    """
    Per-frame output from FootfallAnalyticsEngine.
    """
    frame_number: int
    timestamp: float
    new_entries: int
    new_exits: int
    summary: FootfallSummary
    processing_time_ms: float = 0.0

    @property
    def total_entries(self) -> int:
        return self.summary.total_entries

    @property
    def total_exits(self) -> int:
        return self.summary.total_exits

    @property
    def net_flow(self) -> int:
        return self.summary.net_flow

    @property
    def total_movement_events(self) -> int:
        return self.summary.total_movement_events

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
            "new_entries": self.new_entries,
            "new_exits": self.new_exits,
            "summary": self.summary.to_dict(),
        }
