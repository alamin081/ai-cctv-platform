"""
Entry / Exit Analytics Engine.
Evaluates physical entry, exit, and zone-to-zone movements of tracked persons across configured boundaries.
Phase: Phase 8 - Entry / Exit Analytics Engine
"""

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.zones.models import TrackZoneAssignment, ZoneResult
from .models import (
    EntryExitEvent,
    EntryExitResult,
    InvalidEntryExitConfigError,
    MovementType,
    TrackMovementState,
)

logger = logging.getLogger(__name__)


def _normalize_zone_id(zone: Optional[Union[str, Any]]) -> Optional[str]:
    """Normalize zone name: None, 'outside', 'null', or empty string evaluates to None (outside)."""
    if zone is None:
        return None
    s = str(zone).strip()
    if not s or s.lower() in ("outside", "null", "none"):
        return None
    return s


class EntryExitEngine:
    """
    Engine to detect and classify entry, exit, and zone-to-zone transitions
    for tracked persons across configured spatial boundaries.
    """

    def __init__(
        self,
        entry_transitions: Optional[List[Dict[str, Any]]] = None,
        exit_transitions: Optional[List[Dict[str, Any]]] = None,
        definitions: Optional[List[Dict[str, Any]]] = None,
        allow_initial_entry: bool = False,
    ) -> None:
        """
        Initialize the EntryExitEngine.

        Args:
            entry_transitions: List of allowed entry transition dicts (e.g. [{'from': None, 'to': 'entrance'}]).
            exit_transitions: List of allowed exit transition dicts (e.g. [{'from': 'entrance', 'to': None}]).
            definitions: Optional unified list of transition definitions.
            allow_initial_entry: If False (recommended), tracks already inside a zone on first observation
                                do not produce a retrospective ENTRY event.
        """
        self.allow_initial_entry = bool(allow_initial_entry)

        # Transition rules stored as:
        # entry_rules: List of (from_zone, to_zone, rule_id)
        # exit_rules: List of (from_zone, to_zone, rule_id)
        self._entry_rules: List[Tuple[Optional[str], str, str]] = []
        self._exit_rules: List[Tuple[str, Optional[str], str]] = []

        # Parse and validate transition definitions
        self._parse_configuration(
            entry_transitions=entry_transitions,
            exit_transitions=exit_transitions,
            definitions=definitions,
        )

        # Per-track runtime movement states: track_id -> TrackMovementState
        self._track_states: Dict[int, TrackMovementState] = {}

        # Aggregate metrics
        self._total_entries: int = 0
        self._total_exits: int = 0
        self._total_zone_transitions: int = 0
        self._frames_evaluated: int = 0
        self._event_counter: int = 0

        logger.info(
            "EntryExitEngine initialized with %d entry rules and %d exit rules (allow_initial_entry=%s)",
            len(self._entry_rules),
            len(self._exit_rules),
            self.allow_initial_entry,
        )

    def _parse_configuration(
        self,
        entry_transitions: Optional[List[Dict[str, Any]]] = None,
        exit_transitions: Optional[List[Dict[str, Any]]] = None,
        definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Validate and parse configured entry/exit rules."""
        # 1. Parse unified definitions if provided
        if definitions is not None:
            if not isinstance(definitions, list):
                raise InvalidEntryExitConfigError(
                    f"'definitions' must be a list, got: {type(definitions).__name__}"
                )
            for i, d in enumerate(definitions):
                if not isinstance(d, dict):
                    raise InvalidEntryExitConfigError(
                        f"Definition {i} must be a dictionary, got: {type(d).__name__}"
                    )

                rule_id = str(d.get("id") or f"def_{i}")
                from_raw = d.get("from") if "from" in d else (d.get("entry_zone") if "entry_zone" in d else d.get("from_zone"))
                to_raw = d.get("to") if "to" in d else (d.get("target_zone") if "target_zone" in d else d.get("to_zone"))

                from_zone = _normalize_zone_id(from_raw)
                to_zone = _normalize_zone_id(to_raw)

                if from_zone is None and to_zone is None:
                    raise InvalidEntryExitConfigError(
                        f"Definition '{rule_id}' has both from and to as outside/null."
                    )
                if from_zone == to_zone and from_zone is not None:
                    raise InvalidEntryExitConfigError(
                        f"Definition '{rule_id}' has identical from and to zone: '{from_zone}'."
                    )

                rule_type = str(d.get("type") or "").upper().strip()
                if not rule_type:
                    # Inferred type
                    if from_zone is None and to_zone is not None:
                        rule_type = "ENTRY"
                    elif from_zone is not None and to_zone is None:
                        rule_type = "EXIT"
                    else:
                        rule_type = "ZONE_TRANSITION"

                if rule_type == "ENTRY":
                    if to_zone is None:
                        raise InvalidEntryExitConfigError(
                            f"Entry definition '{rule_id}' must have a valid target zone (cannot enter outside)."
                        )
                    self._entry_rules.append((from_zone, to_zone, rule_id))
                elif rule_type == "EXIT":
                    if from_zone is None:
                        raise InvalidEntryExitConfigError(
                            f"Exit definition '{rule_id}' must have a valid origin zone (cannot exit from outside)."
                        )
                    self._exit_rules.append((from_zone, to_zone, rule_id))
                elif rule_type == "ZONE_TRANSITION":
                    # Stored as directional transition rule
                    pass
                else:
                    raise InvalidEntryExitConfigError(
                        f"Unsupported transition rule type '{rule_type}' in definition '{rule_id}'."
                    )

        # 2. Parse dedicated entry_transitions list if provided
        if entry_transitions is not None:
            if not isinstance(entry_transitions, list):
                raise InvalidEntryExitConfigError(
                    f"'entry_transitions' must be a list, got: {type(entry_transitions).__name__}"
                )
            for i, d in enumerate(entry_transitions):
                if not isinstance(d, dict):
                    raise InvalidEntryExitConfigError(
                        f"entry_transitions[{i}] must be a dict, got: {type(d).__name__}"
                    )
                rule_id = str(d.get("id") or f"entry_{i}")
                from_raw = d.get("from") if "from" in d else d.get("from_zone")
                to_raw = d.get("to") if "to" in d else d.get("to_zone")

                from_zone = _normalize_zone_id(from_raw)
                to_zone = _normalize_zone_id(to_raw)

                if to_zone is None:
                    raise InvalidEntryExitConfigError(
                        f"Entry transition '{rule_id}' must have a target zone (got None/outside)."
                    )
                self._entry_rules.append((from_zone, to_zone, rule_id))

        # 3. Parse dedicated exit_transitions list if provided
        if exit_transitions is not None:
            if not isinstance(exit_transitions, list):
                raise InvalidEntryExitConfigError(
                    f"'exit_transitions' must be a list, got: {type(exit_transitions).__name__}"
                )
            for i, d in enumerate(exit_transitions):
                if not isinstance(d, dict):
                    raise InvalidEntryExitConfigError(
                        f"exit_transitions[{i}] must be a dict, got: {type(d).__name__}"
                    )
                rule_id = str(d.get("id") or f"exit_{i}")
                from_raw = d.get("from") if "from" in d else d.get("from_zone")
                to_raw = d.get("to") if "to" in d else d.get("to_zone")

                from_zone = _normalize_zone_id(from_raw)
                to_zone = _normalize_zone_id(to_raw)

                if from_zone is None:
                    raise InvalidEntryExitConfigError(
                        f"Exit transition '{rule_id}' must have an origin zone (got None/outside)."
                    )
                self._exit_rules.append((from_zone, to_zone, rule_id))

        # Fallback defaults if no rules specified at all
        if not self._entry_rules and not self._exit_rules and definitions is None and entry_transitions is None and exit_transitions is None:
            self._entry_rules.append((None, "entrance", "default_entry"))
            self._exit_rules.append(("entrance", None, "default_exit"))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EntryExitEngine":
        """Factory method to construct EntryExitEngine from dictionary configuration."""
        if not isinstance(config, dict):
            raise InvalidEntryExitConfigError(
                f"Expected dictionary for entry_exit config, got {type(config).__name__}"
            )
        return cls(
            entry_transitions=config.get("entry_transitions"),
            exit_transitions=config.get("exit_transitions"),
            definitions=config.get("definitions"),
            allow_initial_entry=bool(config.get("allow_initial_entry", False)),
        )

    @property
    def total_entries(self) -> int:
        """Cumulative count of ENTRY events."""
        return self._total_entries

    @property
    def total_exits(self) -> int:
        """Cumulative count of EXIT events."""
        return self._total_exits

    @property
    def total_zone_transitions(self) -> int:
        """Cumulative count of ZONE_TRANSITION events."""
        return self._total_zone_transitions

    @property
    def frames_evaluated(self) -> int:
        """Total frames evaluated."""
        return self._frames_evaluated

    @property
    def active_track_states(self) -> Dict[int, TrackMovementState]:
        """Snapshot of current track movement states."""
        return dict(self._track_states)

    def cleanup_track(self, track_id: int) -> None:
        """Purge internal state for a removed track without emitting an EXIT event."""
        self._track_states.pop(track_id, None)

    def cleanup_tracks(self, track_ids: Iterable[int]) -> None:
        """Purge internal state for multiple removed track IDs."""
        for tid in track_ids:
            self.cleanup_track(tid)

    def reset(self) -> None:
        """Reset all runtime track movement states and aggregate metrics."""
        self._track_states.clear()
        self._total_entries = 0
        self._total_exits = 0
        self._total_zone_transitions = 0
        self._frames_evaluated = 0
        self._event_counter = 0
        logger.info("EntryExitEngine runtime state reset completed.")

    def _matches_entry_rule(self, from_zone: Optional[str], to_zone: Optional[str]) -> Optional[str]:
        """Check if (from_zone, to_zone) satisfies an entry rule."""
        if to_zone is None:
            return None
        for rule_from, rule_to, rule_id in self._entry_rules:
            # Matches if origin matches rule_from (None for outside) and destination matches rule_to
            if rule_from == from_zone and rule_to == to_zone:
                return rule_id
            # Also support generic wildcard outside-to-zone if rule_from is None
            if rule_from is None and from_zone is None and rule_to == to_zone:
                return rule_id
        return None

    def _matches_exit_rule(self, from_zone: Optional[str], to_zone: Optional[str]) -> Optional[str]:
        """Check if (from_zone, to_zone) satisfies an exit rule."""
        if from_zone is None:
            return None
        for rule_from, rule_to, rule_id in self._exit_rules:
            # Matches if origin matches rule_from and destination matches rule_to (None for outside)
            if rule_from == from_zone and rule_to == to_zone:
                return rule_id
            if rule_from == from_zone and rule_to is None and to_zone is None:
                return rule_id
        return None

    def record_movement(
        self,
        track_id: int,
        from_zone: Optional[str],
        to_zone: Optional[str],
        frame_number: int,
        timestamp: Optional[float] = None,
    ) -> Optional[EntryExitEvent]:
        """
        Directly evaluate and record a movement between two zones for a track.

        Args:
            track_id: ID of the tracked person.
            from_zone: Starting zone or None if outside.
            to_zone: Destination zone or None if outside.
            frame_number: Sequence number of the frame.
            timestamp: Playback timestamp.

        Returns:
            EntryExitEvent if a valid movement occurred, otherwise None.
        """
        from_zid = _normalize_zone_id(from_zone)
        to_zid = _normalize_zone_id(to_zone)

        # Same zone: no movement event (deduplication)
        if from_zid == to_zid:
            return None

        event: Optional[EntryExitEvent] = None

        # 1. Check ENTRY
        entry_rule_id = self._matches_entry_rule(from_zid, to_zid)
        if entry_rule_id:
            self._event_counter += 1
            self._total_entries += 1
            event = EntryExitEvent(
                event_id=self._event_counter,
                track_id=track_id,
                movement_type=MovementType.ENTRY.value,
                previous_zone_id=from_zid,
                current_zone_id=to_zid,
                frame_number=frame_number,
                timestamp=timestamp,
                transition_id=entry_rule_id,
            )
            logger.info(
                "ENTRY: Track %d into '%s' (from '%s') at frame %d",
                track_id,
                to_zid,
                from_zid,
                frame_number,
            )
            return event

        # 2. Check EXIT
        exit_rule_id = self._matches_exit_rule(from_zid, to_zid)
        if exit_rule_id:
            self._event_counter += 1
            self._total_exits += 1
            event = EntryExitEvent(
                event_id=self._event_counter,
                track_id=track_id,
                movement_type=MovementType.EXIT.value,
                previous_zone_id=from_zid,
                current_zone_id=to_zid,
                frame_number=frame_number,
                timestamp=timestamp,
                transition_id=exit_rule_id,
            )
            logger.info(
                "EXIT: Track %d left '%s' (to '%s') at frame %d",
                track_id,
                from_zid,
                to_zid,
                frame_number,
            )
            return event

        # 3. Check ZONE-TO-ZONE TRANSITION (Both are zones, neither is ENTRY nor EXIT)
        if from_zid is not None and to_zid is not None:
            self._event_counter += 1
            self._total_zone_transitions += 1
            event = EntryExitEvent(
                event_id=self._event_counter,
                track_id=track_id,
                movement_type=MovementType.ZONE_TRANSITION.value,
                previous_zone_id=from_zid,
                current_zone_id=to_zid,
                frame_number=frame_number,
                timestamp=timestamp,
                transition_id=f"{from_zid}_to_{to_zid}",
            )
            logger.info(
                "ZONE_TRANSITION: Track %d from '%s' to '%s' at frame %d",
                track_id,
                from_zid,
                to_zid,
                frame_number,
            )
            return event

        return None

    def update(
        self,
        zone_result: Union[ZoneResult, List[TrackZoneAssignment], Any],
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
        removed_track_ids: Optional[Iterable[int]] = None,
    ) -> EntryExitResult:
        """
        Evaluate entry, exit, and zone-to-zone transitions for tracks in the current frame.

        Args:
            zone_result: ZoneResult from ZoneEngine, or list of TrackZoneAssignment objects.
            frame_number: Optional frame sequence number.
            timestamp: Optional playback timestamp in seconds.
            removed_track_ids: Optional collection of track IDs that have been REMOVED in this frame.

        Returns:
            EntryExitResult: Structured container holding movement events and current metrics.
        """
        start_time = time.perf_counter()
        self._frames_evaluated += 1

        # Purge explicitly removed tracks (no false EXIT events generated)
        if removed_track_ids is not None:
            self.cleanup_tracks(removed_track_ids)

        # Extract track assignments
        assignments: List[TrackZoneAssignment] = []
        if isinstance(zone_result, ZoneResult):
            fn = zone_result.frame_number if zone_result.frame_number is not None else frame_number
            ts = zone_result.timestamp if zone_result.timestamp is not None else timestamp
            assignments = zone_result.tracks
        elif isinstance(zone_result, list):
            fn = frame_number
            ts = timestamp
            assignments = zone_result
        elif zone_result is None:
            fn = frame_number
            ts = timestamp
            assignments = []
        else:
            raise InvalidEntryExitConfigError(
                f"Unsupported zone_result type: {type(zone_result).__name__}. Expected ZoneResult or list."
            )

        fn = int(fn) if fn is not None else self._frames_evaluated
        ts = float(ts) if ts is not None else 0.0

        events: List[EntryExitEvent] = []
        active_track_movements: Dict[int, Optional[str]] = {}

        for assignment in assignments:
            if not hasattr(assignment, "track_id"):
                continue

            tid = int(assignment.track_id)
            curr_zone = _normalize_zone_id(getattr(assignment, "zone_id", None))
            active_track_movements[tid] = curr_zone

            if tid not in self._track_states:
                # First observation of this track ID
                if not self.allow_initial_entry:
                    # Section 19: Establish baseline state without claiming retrospective entry
                    self._track_states[tid] = TrackMovementState(
                        track_id=tid,
                        previous_zone_id=curr_zone,
                        current_zone_id=curr_zone,
                        last_transition_frame=fn,
                        last_transition_timestamp=ts,
                    )
                    continue
                else:
                    # Allow entry on initial observation if inside an entry zone
                    prev_zone = None
            else:
                prev_zone = self._track_states[tid].current_zone_id

            # Evaluate movement
            ev = self.record_movement(
                track_id=tid,
                from_zone=prev_zone,
                to_zone=curr_zone,
                frame_number=fn,
                timestamp=ts,
            )

            if ev is not None:
                events.append(ev)
                state = self._track_states.get(tid)
                if state:
                    state.previous_zone_id = prev_zone
                    state.current_zone_id = curr_zone
                    state.last_transition_frame = fn
                    state.last_transition_timestamp = ts
                    if ev.movement_type == MovementType.ENTRY.value:
                        state.total_entries += 1
                    elif ev.movement_type == MovementType.EXIT.value:
                        state.total_exits += 1
                    elif ev.movement_type == MovementType.ZONE_TRANSITION.value:
                        state.total_transitions += 1
                else:
                    self._track_states[tid] = TrackMovementState(
                        track_id=tid,
                        previous_zone_id=prev_zone,
                        current_zone_id=curr_zone,
                        last_transition_frame=fn,
                        last_transition_timestamp=ts,
                        total_entries=1 if ev.movement_type == MovementType.ENTRY.value else 0,
                        total_exits=1 if ev.movement_type == MovementType.EXIT.value else 0,
                        total_transitions=1 if ev.movement_type == MovementType.ZONE_TRANSITION.value else 0,
                    )
            else:
                # Update current zone state even if no boundary crossing occurred
                if tid in self._track_states:
                    self._track_states[tid].current_zone_id = curr_zone

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return EntryExitResult(
            frame_number=fn,
            timestamp=ts,
            events=events,
            active_track_movements=active_track_movements,
            total_entries=self._total_entries,
            total_exits=self._total_exits,
            total_zone_transitions=self._total_zone_transitions,
            processing_time_ms=elapsed_ms,
        )

    def draw_entry_exit(
        self,
        frame_bgr: np.ndarray,
        result: EntryExitResult,
    ) -> np.ndarray:
        """
        Draw an entry/exit metrics badge and movement event ticker on the frame.

        Args:
            frame_bgr: Source OpenCV BGR image matrix.
            result: EntryExitResult from the current frame.

        Returns:
            np.ndarray: Annotated frame matrix.
        """
        output = frame_bgr.copy()
        h, w = output.shape[:2]

        # Draw a semi-transparent card on the bottom-left corner
        card_w, card_h = 240, 60 + (len(result.events) * 20)
        overlay = output.copy()
        cv2.rectangle(overlay, (10, h - card_h - 10), (10 + card_w, h - 10), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)
        cv2.rectangle(output, (10, h - card_h - 10), (10 + card_w, h - 10), (70, 70, 70), 1)

        # Header
        cv2.putText(
            output,
            "ENTRY / EXIT ANALYTICS",
            (18, h - card_h + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        # Metrics
        metric_str = f"Entries: {result.total_entries} | Exits: {result.total_exits} | Trans: {result.total_zone_transitions}"
        cv2.putText(
            output,
            metric_str,
            (18, h - card_h + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Events
        y = h - card_h + 52
        for ev in result.events:
            color = (0, 255, 128) if ev.movement_type == "ENTRY" else ((0, 100, 255) if ev.movement_type == "EXIT" else (255, 200, 0))
            txt = f"[{ev.movement_type}] ID {ev.track_id}: {ev.previous_zone_id or 'outside'}->{ev.current_zone_id or 'outside'}"
            cv2.putText(
                output,
                txt,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 20

        return output
