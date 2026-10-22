"""
Data models and exceptions for Zone Engine, spatial geometry, and Zone Counting/Occupancy.
Phases: Phase 6 - Zone Engine & Phase 7 - Zone Counting / Occupancy Engine
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class ZoneError(Exception):
    """Base exception for all zone engine errors."""
    pass


class InvalidZoneConfigError(ZoneError):
    """Raised when zone configuration schema or parameters are invalid."""
    pass


class InvalidPolygonError(ZoneError):
    """Raised when a zone polygon has fewer than 3 vertices or non-numeric coordinates."""
    pass


class DuplicateZoneIdError(ZoneError):
    """Raised when two or more configured zones share the same zone_id."""
    pass


class CountingEngineError(ZoneError):
    """Raised when counting or occupancy processing encounters an invalid state or input."""
    pass


class Zone:
    """
    Represents a spatial region of interest defined by a 2D polygon.
    Supports both pixel-space coordinates and normalized (0.0–1.0) coordinates.
    When using normalized coordinates, pixel contours are computed on-demand
    via set_frame_size() for resolution-independent operation.
    """

    def __init__(
        self,
        zone_id: str,
        zone_name: str,
        polygon: List[Union[List[float], Tuple[float, float]]],
        color: Optional[Tuple[int, int, int]] = None,
        is_normalized: bool = False,
    ) -> None:
        """
        Initialize and validate a spatial Zone.

        Args:
            zone_id: Unique string identifier for the zone.
            zone_name: Human-readable display name.
            polygon: List of at least 3 [x, y] vertex coordinate pairs.
            color: Optional BGR color tuple for visual rendering.
            is_normalized: If True, polygon coordinates are in 0.0–1.0 range.
        """
        # Validate zone_id
        if not zone_id or not isinstance(zone_id, str) or not zone_id.strip():
            raise InvalidZoneConfigError(f"zone_id must be a non-empty string. Got: {zone_id!r}")
        self.zone_id = zone_id.strip()

        # Validate zone_name
        if not zone_name or not isinstance(zone_name, str) or not zone_name.strip():
            raise InvalidZoneConfigError(f"zone_name must be a non-empty string for zone '{self.zone_id}'.")
        self.zone_name = zone_name.strip()

        # Validate polygon vertices
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
            pts_count = len(polygon) if isinstance(polygon, (list, tuple)) else type(polygon).__name__
            raise InvalidPolygonError(
                f"Zone '{self.zone_id}' polygon must contain at least 3 vertices. Got {pts_count} points."
            )

        validated_points: List[Tuple[float, float]] = []
        for i, pt in enumerate(polygon):
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                raise InvalidPolygonError(
                    f"Zone '{self.zone_id}' point {i} must be a 2-element [x, y] pair. Got: {pt!r}"
                )
            x, y = pt
            # Explicit check against booleans (bool is subclass of int) and non-numeric
            if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise InvalidPolygonError(
                    f"Zone '{self.zone_id}' point {i} coordinates must be numeric. Got: ({x!r}, {y!r})"
                )
            if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                raise InvalidPolygonError(
                    f"Zone '{self.zone_id}' point {i} coordinates must be finite numbers. Got: ({x!r}, {y!r})"
                )
            validated_points.append((float(x), float(y)))

        self.polygon = validated_points
        self.is_normalized = bool(is_normalized)

        # Frame dimensions for normalized → pixel conversion
        self._frame_width: int = 0
        self._frame_height: int = 0

        # Precompute OpenCV contour matrices for pixel-space zones;
        # for normalized zones these will be recomputed when set_frame_size() is called.
        if not self.is_normalized:
            self._contour = np.array(self.polygon, dtype=np.int32).reshape((-1, 1, 2))
            self._contour_f32 = np.array(self.polygon, dtype=np.float32).reshape((-1, 1, 2))
        else:
            # Placeholder contours; will be set by set_frame_size()
            self._contour = np.array(self.polygon, dtype=np.int32).reshape((-1, 1, 2))
            self._contour_f32 = np.array(self.polygon, dtype=np.float32).reshape((-1, 1, 2))

        # Default display color (distinct hues or cyan default)
        self.color = color if color is not None else (255, 180, 0)

    def set_frame_size(self, width: int, height: int) -> None:
        """
        Set the video frame dimensions for normalized → pixel coordinate conversion.
        Recomputes the internal OpenCV contour arrays for accurate pointPolygonTest.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        if width <= 0 or height <= 0:
            return
        self._frame_width = int(width)
        self._frame_height = int(height)

        if self.is_normalized:
            pixel_pts = [
                (x * self._frame_width, y * self._frame_height)
                for x, y in self.polygon
            ]
            self._contour = np.array(pixel_pts, dtype=np.int32).reshape((-1, 1, 2))
            self._contour_f32 = np.array(pixel_pts, dtype=np.float32).reshape((-1, 1, 2))

    @property
    def contour(self) -> np.ndarray:
        """Cached contour array for OpenCV drawing operations."""
        return self._contour

    def contains_point(self, point: Tuple[float, float], boundary_policy: str = "inside") -> bool:
        """
        Check if a 2D coordinate point lies within the zone polygon.

        Args:
            point: (x, y) coordinates in pixel space.
            boundary_policy: 'inside' treats boundary points as inside; 'outside' treats boundary as outside.

        Returns:
            bool: True if point is inside according to the boundary policy.
        """
        # cv2.pointPolygonTest returns:
        # > 0 for strictly inside
        # == 0 for on the boundary
        # < 0 for strictly outside
        val = cv2.pointPolygonTest(self._contour_f32, (float(point[0]), float(point[1])), measureDist=False)
        if val > 0:
            return True
        elif abs(val) < 1e-5:
            return boundary_policy.lower().strip() == "inside"
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert zone definition to dictionary with normalized polygon."""
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "polygon": [[round(x, 6), round(y, 6)] for x, y in self.polygon],
            "is_normalized": self.is_normalized,
            "color": list(self.color) if self.color else [255, 180, 0],
        }

    def to_normalized_dict(self) -> Dict[str, Any]:
        """Convert zone definition to dictionary, ensuring normalized polygon coordinates."""
        if self.is_normalized:
            poly = [[round(x, 6), round(y, 6)] for x, y in self.polygon]
        elif self._frame_width > 0 and self._frame_height > 0:
            poly = [
                [round(x / self._frame_width, 6), round(y / self._frame_height, 6)]
                for x, y in self.polygon
            ]
        else:
            poly = [[round(x, 1), round(y, 1)] for x, y in self.polygon]
        return {
            "id": self.zone_id,
            "name": self.zone_name,
            "polygon": poly,
            "color": list(self.color) if self.color else [255, 180, 0],
        }

    def __repr__(self) -> str:
        coord_type = "normalized" if self.is_normalized else "pixel"
        return f"Zone(id='{self.zone_id}', name='{self.zone_name}', vertices={len(self.polygon)}, {coord_type})"


@dataclass
class TrackZoneAssignment:
    """Represents a tracked person's spatial zone membership and transition state in a frame."""
    track_id: int
    position: List[float]  # [x, y] representative point
    previous_zone_id: Optional[str]
    zone_id: Optional[str]
    zone_name: Optional[str]
    transition: Optional[str]  # 'ENTERED', 'LEFT', or None

    def to_dict(self) -> Dict[str, Any]:
        """Convert assignment to JSON-serializable dictionary."""
        return {
            "track_id": self.track_id,
            "position": [round(c, 2) for c in self.position],
            "previous_zone_id": self.previous_zone_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "transition": self.transition,
        }


@dataclass
class ZoneResult:
    """Structured output container for a single video frame from ZoneEngine."""
    frame_number: int
    timestamp: float
    zones: List[Dict[str, str]]
    tracks: List[TrackZoneAssignment]
    processing_time_ms: float = 0.0

    @property
    def count(self) -> int:
        """Count of tracks with zone assignments in this frame."""
        return len(self.tracks)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
            "zones": self.zones,
            "tracks": [t.to_dict() for t in self.tracks],
        }


# ==============================================================================
# Phase 7 — Zone Counting & Occupancy Models
# ==============================================================================

@dataclass
class ZoneOccupancy:
    """
    Represents the instantaneous occupancy count and assigned Track IDs of a zone in a frame.
    Guarantees count == len(track_ids) and deterministic track_ids ordering without duplicates.
    """
    zone_id: str
    zone_name: str
    count: int
    track_ids: List[int]

    def __post_init__(self) -> None:
        """Enforce deduplication, sorting, and count consistency."""
        # Deduplicate while preserving unique IDs, then sort deterministically
        deduped = sorted(list({int(tid) for tid in self.track_ids}))
        self.track_ids = deduped
        self.count = len(self.track_ids)

    def to_dict(self) -> Dict[str, Any]:
        """Convert occupancy state to JSON-serializable dictionary."""
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "count": self.count,
            "track_ids": list(self.track_ids),
        }

    def __repr__(self) -> str:
        return f"ZoneOccupancy(zone='{self.zone_id}', count={self.count}, track_ids={self.track_ids})"


@dataclass
class ZoneEvent:
    """
    Represents a discrete spatial boundary crossing event for a track.
    Event types: 'ZONE_ENTER', 'ZONE_EXIT', 'ZONE_TRANSITION'.
    """
    event_type: str
    track_id: int
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    from_zone: Optional[str] = None
    from_zone_name: Optional[str] = None
    to_zone: Optional[str] = None
    to_zone_name: Optional[str] = None
    frame_number: int = 0
    timestamp: Optional[float] = None

    @property
    def zone(self) -> Optional[str]:
        """Convenience property for zone name / zone id."""
        return self.zone_id

    @property
    def frame_id(self) -> int:
        """Convenience property for frame ID."""
        return self.frame_number

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to JSON-serializable dictionary matching Section 7-9 schema."""
        d: Dict[str, Any] = {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "frame_id": self.frame_number,
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
        }
        if self.event_type == "ZONE_ENTER":
            d["zone"] = self.zone_id
            d["zone_id"] = self.zone_id
            d["zone_name"] = self.zone_name
        elif self.event_type == "ZONE_EXIT":
            d["zone"] = self.zone_id
            d["zone_id"] = self.zone_id
            d["zone_name"] = self.zone_name
        elif self.event_type == "ZONE_TRANSITION":
            d["from_zone"] = self.from_zone
            d["from_zone_name"] = self.from_zone_name
            d["to_zone"] = self.to_zone
            d["to_zone_name"] = self.to_zone_name
        return d

    def __repr__(self) -> str:
        if self.event_type == "ZONE_TRANSITION":
            return f"ZoneEvent({self.event_type}: Track {self.track_id} '{self.from_zone}' -> '{self.to_zone}' @ f{self.frame_number})"
        return f"ZoneEvent({self.event_type}: Track {self.track_id} in '{self.zone_id}' @ f{self.frame_number})"


@dataclass
class ZoneCountingResult:
    """
    Structured output container for frame-level zone occupancy and boundary events.
    """
    frame_number: int
    timestamp: float
    occupancy: Dict[str, ZoneOccupancy]
    events: List[ZoneEvent]
    total_occupancy: int
    processing_time_ms: float = 0.0

    def get_occupancy(self, zone_id: str) -> Optional[ZoneOccupancy]:
        """Query occupancy object for a given zone_id."""
        return self.occupancy.get(zone_id)

    def get_count(self, zone_id: str) -> int:
        """Query current occupancy count for a given zone_id."""
        occ = self.occupancy.get(zone_id)
        return occ.count if occ else 0

    def get_track_ids(self, zone_id: str) -> List[int]:
        """Query list of active track IDs in a given zone_id."""
        occ = self.occupancy.get(zone_id)
        return list(occ.track_ids) if occ else []

    def to_dict(self) -> Dict[str, Any]:
        """Convert counting result to JSON-serializable dictionary."""
        return {
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
            "total_occupancy": self.total_occupancy,
            "occupancy": {zid: occ.to_dict() for zid, occ in self.occupancy.items()},
            "events": [e.to_dict() for e in self.events],
        }
