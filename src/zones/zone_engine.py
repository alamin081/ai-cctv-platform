"""
Zone Engine implementation for spatial zone definition and track membership/transition detection.
Phase: Phase 6 - Zone Engine
"""

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.tracking.models import (
    Track,
    TrackState,
    TrackStateResult,
    TrackingResult,
)
from .models import (
    DuplicateZoneIdError,
    InvalidZoneConfigError,
    TrackZoneAssignment,
    Zone,
    ZoneResult,
)

logger = logging.getLogger(__name__)

TARGET_PERSON_CLASS_ID = 0


class ZoneEngine:
    """
    Manages spatial zones of interest and evaluates track-to-zone membership and transitions.
    Determines whether tracked persons are inside zones and detects ENTERED / LEFT transitions.
    """

    def __init__(
        self,
        zones: Optional[List[Union[Zone, Dict[str, Any]]]] = None,
        assignment_mode: str = "center",
        boundary_policy: str = "inside",
        is_normalized: bool = False,
    ) -> None:
        """
        Initialize the ZoneEngine with configured spatial zones.

        Args:
            zones: List of Zone objects or configuration dictionaries defining zones.
            assignment_mode: Track reference point strategy ('center' or 'bottom_center').
            boundary_policy: How points lying on the polygon boundary are evaluated ('inside' or 'outside').
            is_normalized: If True, zone polygons use normalized (0.0-1.0) coordinates.

        Raises:
            InvalidZoneConfigError: If configuration schema or parameters are invalid.
            DuplicateZoneIdError: If duplicate zone IDs are provided.
        """
        if not isinstance(assignment_mode, str):
            raise InvalidZoneConfigError(f"assignment_mode must be a string. Got: {type(assignment_mode).__name__}")
        self.assignment_mode = assignment_mode.lower().strip()
        if self.assignment_mode not in ("center", "bottom_center"):
            raise InvalidZoneConfigError(
                f"Unsupported assignment_mode '{assignment_mode}'. Expected 'center' or 'bottom_center'."
            )

        if not isinstance(boundary_policy, str):
            raise InvalidZoneConfigError(f"boundary_policy must be a string. Got: {type(boundary_policy).__name__}")
        self.boundary_policy = boundary_policy.lower().strip()
        if self.boundary_policy not in ("inside", "outside"):
            raise InvalidZoneConfigError(
                f"Unsupported boundary_policy '{boundary_policy}'. Expected 'inside' or 'outside'."
            )

        self._zones: List[Zone] = []
        self._zone_map: Dict[str, Zone] = {}
        self._track_current_zone: Dict[int, Optional[str]] = {}

        # Metrics counters
        self._total_zone_enter_events: int = 0
        self._total_zone_leave_events: int = 0
        self._frames_evaluated: int = 0

        # Colors for automatic zone visualization (BGR)
        palette = [
            (255, 128, 0),   # Cyan / Blue
            (0, 200, 255),   # Amber / Orange
            (180, 0, 255),   # Magenta
            (0, 255, 128),   # Light Green
            (255, 0, 128),   # Purple
        ]

        # Whether zones use normalized (0.0-1.0) coordinates
        self._is_normalized = is_normalized
        self._frame_width: int = 0
        self._frame_height: int = 0

        if zones:
            if not isinstance(zones, (list, tuple)):
                raise InvalidZoneConfigError(f"zones must be a list. Got: {type(zones).__name__}")

            for i, z in enumerate(zones):
                if isinstance(z, Zone):
                    zone_obj = z
                elif isinstance(z, dict):
                    zone_id = z.get("id") or z.get("zone_id")
                    zone_name = z.get("name") or z.get("zone_name") or zone_id
                    polygon = z.get("polygon")
                    color = z.get("color") or palette[i % len(palette)]
                    zone_obj = Zone(
                        zone_id=zone_id,
                        zone_name=zone_name,
                        polygon=polygon,
                        color=color,
                        is_normalized=self._is_normalized,
                    )
                else:
                    raise InvalidZoneConfigError(f"Invalid zone definition type: {type(z).__name__}")

                if zone_obj.zone_id in self._zone_map:
                    raise DuplicateZoneIdError(f"Duplicate zone_id detected: '{zone_obj.zone_id}'")

                self._zones.append(zone_obj)
                self._zone_map[zone_obj.zone_id] = zone_obj

        logger.info(
            "ZoneEngine initialized with %d zones (mode='%s', boundary='%s')",
            len(self._zones),
            self.assignment_mode,
            self.boundary_policy,
        )

    def set_frame_size(self, width: int, height: int) -> None:
        """
        Set video frame dimensions for all normalized zones.
        This propagates resolution to each Zone for accurate pixel-space operations.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self._frame_width = int(width)
        self._frame_height = int(height)
        for zone in self._zones:
            zone.set_frame_size(width, height)
        logger.info(
            "ZoneEngine frame size set to %dx%d for %d zones (normalized=%s)",
            width, height, len(self._zones), self._is_normalized,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ZoneEngine":
        """
        Create a ZoneEngine instance from a configuration dictionary.

        Expected schema:
            assignment_mode: "center" | "bottom_center"
            boundary_policy: "inside" | "outside"
            coordinate_system: "normalized" | "pixel"
            definitions: list of zone dicts
        """
        if not isinstance(config, dict):
            raise InvalidZoneConfigError(f"Expected dict for zone configuration, got {type(config).__name__}")

        assignment_mode = str(config.get("assignment_mode", "center"))
        boundary_policy = str(config.get("boundary_policy", "inside"))
        coordinate_system = str(config.get("coordinate_system", "pixel")).lower().strip()
        is_normalized = coordinate_system == "normalized"
        definitions = config.get("definitions", [])

        if definitions is None:
            definitions = []
        elif not isinstance(definitions, list):
            raise InvalidZoneConfigError(f"'definitions' must be a list, got {type(definitions).__name__}")

        return cls(
            zones=definitions,
            assignment_mode=assignment_mode,
            boundary_policy=boundary_policy,
            is_normalized=is_normalized,
        )

    @property
    def zones(self) -> List[Zone]:
        """List of configured Zone objects."""
        return list(self._zones)

    @property
    def total_zone_enter_events(self) -> int:
        return self._total_zone_enter_events

    @property
    def total_zone_leave_events(self) -> int:
        return self._total_zone_leave_events

    @property
    def frames_evaluated(self) -> int:
        return self._frames_evaluated

    @property
    def active_track_zones(self) -> Dict[int, Optional[str]]:
        """Snapshot of current track-to-zone mappings."""
        return dict(self._track_current_zone)

    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Query a configured zone by ID."""
        return self._zone_map.get(zone_id)

    def get_track_zone(self, track_id: int) -> Optional[str]:
        """Get the current assigned zone_id for a track, or None if outside/unknown."""
        return self._track_current_zone.get(track_id)

    def cleanup_track(self, track_id: int) -> None:
        """Explicitly purge internal state for a removed or expired track."""
        if track_id in self._track_current_zone:
            self._track_current_zone.pop(track_id, None)

    def cleanup_tracks(self, track_ids: Iterable[int]) -> None:
        """Purge internal state for multiple track IDs."""
        for tid in track_ids:
            self.cleanup_track(tid)

    def reset(self) -> None:
        """Reset all runtime track memberships and metrics."""
        self._track_current_zone.clear()
        self._total_zone_enter_events = 0
        self._total_zone_leave_events = 0
        self._frames_evaluated = 0
        logger.info("ZoneEngine state reset completed.")

    def _extract_position(self, bbox: List[float]) -> Tuple[float, float]:
        """Extract representative reference point from bounding box based on assignment_mode."""
        x1, y1, x2, y2 = bbox
        if self.assignment_mode == "bottom_center":
            return (float(x1 + x2) / 2.0, float(y2))
        # Default: center point
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)

    def update(
        self,
        tracks: Union[TrackStateResult, TrackingResult, List[Track], Any],
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
        removed_track_ids: Optional[Iterable[int]] = None,
    ) -> ZoneResult:
        """
        Evaluate spatial zone memberships and transitions for tracks in the current frame.

        Args:
            tracks: TrackStateResult (Phase 5), TrackingResult (Phase 4), or list of Track objects.
            frame_number: Optional frame index.
            timestamp: Optional playback timestamp in seconds.
            removed_track_ids: Optional collection of track IDs that have been REMOVED in this frame.

        Returns:
            ZoneResult: Structured container with per-track zone memberships and transitions.
        """
        start_time = time.perf_counter()
        self._frames_evaluated += 1

        # Purge explicitly provided removed track IDs to avoid memory leaks
        if removed_track_ids is not None:
            self.cleanup_tracks(removed_track_ids)

        # Extract frame number, timestamp, and track list
        if isinstance(tracks, TrackStateResult):
            fn = tracks.frame_number if tracks.frame_number is not None else frame_number
            ts = tracks.timestamp if tracks.timestamp is not None else timestamp
            incoming_tracks = tracks.tracks
        elif isinstance(tracks, TrackingResult):
            fn = tracks.frame_number if tracks.frame_number is not None else frame_number
            ts = tracks.timestamp if tracks.timestamp is not None else timestamp
            incoming_tracks = tracks.tracks
        elif isinstance(tracks, list):
            fn = frame_number
            ts = timestamp
            incoming_tracks = tracks
        elif tracks is None:
            fn = frame_number
            ts = timestamp
            incoming_tracks = []
        else:
            raise InvalidZoneConfigError(
                f"Unsupported tracks input type: {type(tracks).__name__}. Expected TrackStateResult, TrackingResult, or list of Track."
            )

        fn = int(fn) if fn is not None else self._frames_evaluated
        ts = float(ts) if ts is not None else 0.0

        assignments: List[TrackZoneAssignment] = []

        # Process each incoming track
        for t in incoming_tracks:
            # Filter strictly to person class (COCO 0)
            cls_id = getattr(t, "class_id", TARGET_PERSON_CLASS_ID)
            if cls_id != TARGET_PERSON_CLASS_ID:
                continue

            tid = int(t.track_id)
            state_val = getattr(t, "state", "ACTIVE")
            state_str = str(state_val).upper()

            # Handle REMOVED tracks: purge internal memory state
            if state_str == TrackState.REMOVED.value:
                self.cleanup_track(tid)
                continue

            # Only evaluate active/visible tracks
            if state_str not in (TrackState.ACTIVE.value, TrackState.NEW.value):
                # LOST tracks retain their previous zone state without new assignments
                continue

            pos_x, pos_y = self._extract_position(t.bbox)

            # Determine matching zone (first match policy for deterministic overlap resolution)
            matched_zone: Optional[Zone] = None
            for zone in self._zones:
                if zone.contains_point((pos_x, pos_y), boundary_policy=self.boundary_policy):
                    matched_zone = zone
                    break

            prev_zone_id = self._track_current_zone.get(tid, None)
            curr_zone_id = matched_zone.zone_id if matched_zone else None
            curr_zone_name = matched_zone.zone_name if matched_zone else None

            # Detect transition events
            transition: Optional[str] = None
            if prev_zone_id is None and curr_zone_id is not None:
                # Track entered a zone from outside / initial registration
                transition = "ENTERED"
                self._total_zone_enter_events += 1
                logger.info("Track %d ENTERED zone '%s' at frame %d", tid, curr_zone_id, fn)

            elif prev_zone_id is not None and curr_zone_id is not None and prev_zone_id != curr_zone_id:
                # Track moved directly from one zone into another
                transition = "ENTERED"
                self._total_zone_enter_events += 1
                logger.info(
                    "Track %d transitioned: '%s' -> '%s' at frame %d",
                    tid,
                    prev_zone_id,
                    curr_zone_id,
                    fn,
                )

            elif prev_zone_id is not None and curr_zone_id is None:
                # Track left zone to outside area
                transition = "LEFT"
                self._total_zone_leave_events += 1
                logger.info("Track %d LEFT zone '%s' at frame %d", tid, prev_zone_id, fn)

            elif prev_zone_id == curr_zone_id:
                # Still in the same zone or still outside
                transition = None

            # Update current zone state
            self._track_current_zone[tid] = curr_zone_id

            assignment = TrackZoneAssignment(
                track_id=tid,
                position=[round(pos_x, 2), round(pos_y, 2)],
                previous_zone_id=prev_zone_id,
                zone_id=curr_zone_id,
                zone_name=curr_zone_name,
                transition=transition,
            )
            assignments.append(assignment)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ZoneResult(
            frame_number=fn,
            timestamp=ts,
            zones=[{"zone_id": z.zone_id, "zone_name": z.zone_name} for z in self._zones],
            tracks=assignments,
            processing_time_ms=elapsed_ms,
        )

    def draw_zones(
        self,
        frame_bgr: np.ndarray,
        zone_result: Optional[ZoneResult] = None,
    ) -> np.ndarray:
        """
        Draw configured zones, boundary polygons, and track zone membership annotations.

        Args:
            frame_bgr: Source OpenCV BGR image matrix.
            zone_result: Optional ZoneResult for annotating track positions and zone tags.

        Returns:
            np.ndarray: Annotated frame matrix.
        """
        output = frame_bgr.copy()
        overlay = frame_bgr.copy()

        # 1. Render zone polygons with semi-transparent fill and thick outlines
        for zone in self._zones:
            cv2.fillPoly(overlay, [zone.contour], zone.color)
            cv2.polylines(output, [zone.contour], isClosed=True, color=zone.color, thickness=2)

            # Zone label at polygon centroid/first vertex
            pts = np.array(zone.polygon, dtype=np.int32)
            lbl_x = int(pts[:, 0].mean())
            lbl_y = int(pts[:, 1].min() + 25)

            font = cv2.FONT_HERSHEY_SIMPLEX
            label = f"[ZONE: {zone.zone_name.upper()}]"
            cv2.putText(output, label, (lbl_x - 60, lbl_y), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(output, label, (lbl_x - 60, lbl_y), font, 0.55, zone.color, 1, cv2.LINE_AA)

        # Blend semi-transparent polygons
        cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)

        # 2. Render track positions and zone tags if zone_result is provided
        if zone_result:
            for t in zone_result.tracks:
                px, py = int(round(t.position[0])), int(round(t.position[1]))

                # Draw track position dot
                cv2.circle(output, (px, py), 5, (0, 255, 255), -1)
                cv2.circle(output, (px, py), 7, (0, 0, 0), 1)

                # Track zone tag
                zone_tag = f"ID: {t.track_id} | Zone: {t.zone_name if t.zone_name else 'Outside'}"
                if t.transition:
                    zone_tag += f" ({t.transition})"

                cv2.putText(
                    output,
                    zone_tag,
                    (px - 40, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    zone_tag,
                    (px - 40, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return output
