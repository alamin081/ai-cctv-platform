"""
Zone Counting / Occupancy Engine.
Evaluates per-zone instantaneous occupancy and discrete spatial boundary events (ENTER, EXIT, TRANSITION).
Phase: Phase 7 - Zone Counting / Occupancy Engine
"""

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

from .models import (
    CountingEngineError,
    TrackZoneAssignment,
    Zone,
    ZoneCountingResult,
    ZoneEvent,
    ZoneOccupancy,
    ZoneResult,
)
from .zone_engine import ZoneEngine

logger = logging.getLogger(__name__)


class ZoneCountingEngine:
    """
    Manages zone occupancy calculations and discrete boundary crossing events.
    Computes per-zone track occupancy and emits ZONE_ENTER, ZONE_EXIT, and ZONE_TRANSITION events.
    """

    def __init__(
        self,
        zones: Optional[List[Union[Zone, Dict[str, Any]]]] = None,
        zone_engine: Optional[ZoneEngine] = None,
        zone_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the ZoneCountingEngine.

        Args:
            zones: Optional list of Zone objects or configuration dictionaries.
            zone_engine: Optional ZoneEngine instance to adopt configured zones from.
            zone_ids: Optional list of zone ID strings to track.
        """
        self._configured_zones: Dict[str, str] = {}  # zone_id -> zone_name

        # Adopt zones from ZoneEngine if provided
        if zone_engine is not None:
            for z in zone_engine.zones:
                self.register_zone(z.zone_id, z.zone_name)

        # Adopt zones from explicit zone list if provided
        if zones:
            for z in zones:
                if isinstance(z, Zone):
                    self.register_zone(z.zone_id, z.zone_name)
                elif isinstance(z, dict):
                    zid = z.get("id") or z.get("zone_id")
                    zname = z.get("name") or z.get("zone_name") or zid
                    if zid:
                        self.register_zone(str(zid), str(zname))

        # Adopt zones from string list if provided
        if zone_ids:
            for zid in zone_ids:
                if zid:
                    self.register_zone(str(zid), str(zid))

        # Runtime track state: track_id -> last_known_zone_id
        self._track_last_zone: Dict[int, Optional[str]] = {}
        self._track_last_zone_name: Dict[int, Optional[str]] = {}

        # Aggregate metrics
        self._frames_evaluated: int = 0
        self._total_enter_events: int = 0
        self._total_exit_events: int = 0
        self._total_transition_events: int = 0

        logger.info(
            "ZoneCountingEngine initialized with %d configured zones: %s",
            len(self._configured_zones),
            list(self._configured_zones.keys()),
        )

    @classmethod
    def from_zone_engine(cls, zone_engine: ZoneEngine) -> "ZoneCountingEngine":
        """Factory method to construct ZoneCountingEngine directly from a ZoneEngine."""
        return cls(zone_engine=zone_engine)

    @property
    def configured_zones(self) -> Dict[str, str]:
        """Dictionary of configured zone IDs and display names."""
        return dict(self._configured_zones)

    @property
    def total_enter_events(self) -> int:
        """Cumulative count of ZONE_ENTER events detected."""
        return self._total_enter_events

    @property
    def total_exit_events(self) -> int:
        """Cumulative count of ZONE_EXIT events detected."""
        return self._total_exit_events

    @property
    def total_transition_events(self) -> int:
        """Cumulative count of ZONE_TRANSITION events detected."""
        return self._total_transition_events

    @property
    def frames_evaluated(self) -> int:
        """Total number of frames processed by the counting engine."""
        return self._frames_evaluated

    def register_zone(self, zone_id: str, zone_name: Optional[str] = None) -> None:
        """
        Register a zone so it is consistently tracked in occupancy results even when empty.

        Args:
            zone_id: Unique string identifier of the zone.
            zone_name: Human-readable display name.
        """
        zid = str(zone_id).strip()
        if not zid:
            return
        zname = str(zone_name).strip() if zone_name else zid
        self._configured_zones[zid] = zname

    def cleanup_track(self, track_id: int) -> None:
        """Purge internal state for a removed or expired track to prevent memory leaks."""
        self._track_last_zone.pop(track_id, None)
        self._track_last_zone_name.pop(track_id, None)

    def cleanup_tracks(self, track_ids: Iterable[int]) -> None:
        """Purge internal state for multiple removed track IDs."""
        for tid in track_ids:
            self.cleanup_track(tid)

    def reset(self) -> None:
        """Reset internal runtime state and metrics."""
        self._track_last_zone.clear()
        self._track_last_zone_name.clear()
        self._frames_evaluated = 0
        self._total_enter_events = 0
        self._total_exit_events = 0
        self._total_transition_events = 0
        logger.info("ZoneCountingEngine runtime state reset completed.")

    def update(
        self,
        zone_result: Union[ZoneResult, List[TrackZoneAssignment], Any],
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
        removed_track_ids: Optional[Iterable[int]] = None,
    ) -> ZoneCountingResult:
        """
        Evaluate frame-level zone occupancy and emit boundary crossing events.

        Args:
            zone_result: ZoneResult from ZoneEngine, or a list of TrackZoneAssignment objects.
            frame_number: Optional frame sequence number.
            timestamp: Optional playback timestamp in seconds.
            removed_track_ids: Optional collection of track IDs that have been REMOVED in this frame.

        Returns:
            ZoneCountingResult: Structured container holding per-zone occupancy and frame events.
        """
        start_time = time.perf_counter()
        self._frames_evaluated += 1

        # Purge explicitly provided removed tracks to prevent memory leaks
        if removed_track_ids is not None:
            self.cleanup_tracks(removed_track_ids)

        # Extract assignments, frame_number, and timestamp
        assignments: List[TrackZoneAssignment] = []
        if isinstance(zone_result, ZoneResult):
            fn = zone_result.frame_number if zone_result.frame_number is not None else frame_number
            ts = zone_result.timestamp if zone_result.timestamp is not None else timestamp
            assignments = zone_result.tracks
            # Auto-register any zones present in zone_result if not already known
            for zinfo in zone_result.zones:
                zid = zinfo.get("zone_id")
                zname = zinfo.get("zone_name")
                if zid and zid not in self._configured_zones:
                    self.register_zone(zid, zname)
        elif isinstance(zone_result, list):
            fn = frame_number
            ts = timestamp
            assignments = zone_result
        elif zone_result is None:
            fn = frame_number
            ts = timestamp
            assignments = []
        else:
            raise CountingEngineError(
                f"Unsupported zone_result type: {type(zone_result).__name__}. Expected ZoneResult or list."
            )

        fn = int(fn) if fn is not None else self._frames_evaluated
        ts = float(ts) if ts is not None else 0.0

        # Step 1: Collect active track IDs per zone for current frame occupancy
        zone_tracks_map: Dict[str, List[int]] = {zid: [] for zid in self._configured_zones}
        events: List[ZoneEvent] = []

        # Step 2: Evaluate track assignments and detect boundary events
        for assignment in assignments:
            if not hasattr(assignment, "track_id"):
                continue

            tid = int(assignment.track_id)
            curr_zid: Optional[str] = assignment.zone_id
            curr_zname: Optional[str] = getattr(assignment, "zone_name", None) or (
                self._configured_zones.get(curr_zid, curr_zid) if curr_zid else None
            )

            # Auto-register newly encountered zone if active
            if curr_zid and curr_zid not in self._configured_zones:
                self.register_zone(curr_zid, curr_zname)
                zone_tracks_map[curr_zid] = []

            # Populate occupancy mapping if track is inside a configured zone
            if curr_zid and curr_zid in zone_tracks_map:
                if tid not in zone_tracks_map[curr_zid]:
                    zone_tracks_map[curr_zid].append(tid)

            # Boundary Event Detection & Deduplication
            prev_zid = self._track_last_zone.get(tid, None)
            prev_zname = self._track_last_zone_name.get(tid, None)

            if prev_zid is None and curr_zid is not None:
                # OUTSIDE -> ZONE: Emits ZONE_ENTER
                event = ZoneEvent(
                    event_type="ZONE_ENTER",
                    track_id=tid,
                    zone_id=curr_zid,
                    zone_name=curr_zname,
                    frame_number=fn,
                    timestamp=ts,
                )
                events.append(event)
                self._total_enter_events += 1
                logger.info("ZONE_ENTER: Track %d into '%s' at frame %d", tid, curr_zid, fn)

            elif prev_zid is not None and curr_zid is not None and prev_zid != curr_zid:
                # ZONE A -> ZONE B: Emits ZONE_TRANSITION
                event = ZoneEvent(
                    event_type="ZONE_TRANSITION",
                    track_id=tid,
                    from_zone=prev_zid,
                    from_zone_name=prev_zname,
                    to_zone=curr_zid,
                    to_zone_name=curr_zname,
                    frame_number=fn,
                    timestamp=ts,
                )
                events.append(event)
                self._total_transition_events += 1
                logger.info(
                    "ZONE_TRANSITION: Track %d from '%s' to '%s' at frame %d",
                    tid,
                    prev_zid,
                    curr_zid,
                    fn,
                )

            elif prev_zid is not None and curr_zid is None:
                # ZONE -> OUTSIDE: Emits ZONE_EXIT
                event = ZoneEvent(
                    event_type="ZONE_EXIT",
                    track_id=tid,
                    zone_id=prev_zid,
                    zone_name=prev_zname,
                    frame_number=fn,
                    timestamp=ts,
                )
                events.append(event)
                self._total_exit_events += 1
                logger.info("ZONE_EXIT: Track %d from '%s' at frame %d", tid, prev_zid, fn)

            # prev_zid == curr_zid: no event emitted (deduplication)

            # Update last known state
            self._track_last_zone[tid] = curr_zid
            self._track_last_zone_name[tid] = curr_zname

        # Step 3: Build structured ZoneOccupancy for all configured zones
        occupancy_dict: Dict[str, ZoneOccupancy] = {}
        total_occupancy = 0

        for zid, zname in self._configured_zones.items():
            tids = sorted(zone_tracks_map.get(zid, []))
            occ = ZoneOccupancy(
                zone_id=zid,
                zone_name=zname,
                count=len(tids),
                track_ids=tids,
            )
            occupancy_dict[zid] = occ
            total_occupancy += occ.count

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ZoneCountingResult(
            frame_number=fn,
            timestamp=ts,
            occupancy=occupancy_dict,
            events=events,
            total_occupancy=total_occupancy,
            processing_time_ms=elapsed_ms,
        )

    def draw_occupancy(
        self,
        frame_bgr: np.ndarray,
        counting_result: ZoneCountingResult,
    ) -> np.ndarray:
        """
        Draw an occupancy dashboard and event feed overlay onto the frame.

        Args:
            frame_bgr: Source OpenCV BGR image matrix.
            counting_result: ZoneCountingResult from the current frame.

        Returns:
            np.ndarray: Annotated frame matrix.
        """
        output = frame_bgr.copy()
        h, w = output.shape[:2]

        # Draw a translucent HUD card in the top-left or top-right corner
        card_w, card_h = 240, 30 + (len(counting_result.occupancy) * 22) + (len(counting_result.events) * 18)
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (10 + card_w, 10 + card_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        cv2.rectangle(output, (10, 10), (10 + card_w, 10 + card_h), (80, 80, 80), 1)

        # Header
        cv2.putText(
            output,
            f"ZONE OCCUPANCY (Total: {counting_result.total_occupancy})",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y = 50
        for zid, occ in counting_result.occupancy.items():
            txt = f"{occ.zone_name}: {occ.count} {occ.track_ids if occ.count > 0 else ''}"
            cv2.putText(
                output,
                txt,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255) if occ.count > 0 else (160, 160, 160),
                1,
                cv2.LINE_AA,
            )
            y += 22

        # Draw any events triggered in this frame
        for ev in counting_result.events:
            if ev.event_type == "ZONE_TRANSITION":
                ev_txt = f"[{ev.event_type}] ID {ev.track_id}: {ev.from_zone}->{ev.to_zone}"
                color = (0, 180, 255)
            elif ev.event_type == "ZONE_ENTER":
                ev_txt = f"[{ev.event_type}] ID {ev.track_id} -> {ev.zone_id}"
                color = (0, 255, 128)
            else:
                ev_txt = f"[{ev.event_type}] ID {ev.track_id} from {ev.zone_id}"
                color = (0, 128, 255)

            cv2.putText(
                output,
                ev_txt,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 18

        return output
