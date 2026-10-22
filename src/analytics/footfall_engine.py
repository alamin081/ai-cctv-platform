"""
Footfall Analytics Engine.
Performs in-memory footfall aggregation, entry/exit totals, net flow, and time-bucket analytics.
Phase: Phase 9 - Footfall Analytics Engine
"""

import logging
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import cv2
import numpy as np

from .models import (
    EntryExitEvent,
    EntryExitResult,
    FootfallBucket,
    FootfallResult,
    FootfallSummary,
    InvalidFootfallConfigError,
    InvalidFootfallEventError,
    MovementType,
)

logger = logging.getLogger(__name__)


class FootfallAnalyticsEngine:
    """
    Engine for in-memory Footfall aggregation, net flow calculation, and time-bucket summaries.
    Consumes validated Entry/Exit events from EntryExitEngine.
    """

    def __init__(
        self,
        bucket_minutes: Optional[float] = None,
        bucket_seconds: Optional[float] = None,
        deduplicate_events: bool = True,
        strict_validation: bool = False,
    ) -> None:
        """
        Initialize FootfallAnalyticsEngine.

        Args:
            bucket_minutes: Bucket duration in minutes (e.g. 15 for 15-minute intervals).
            bucket_seconds: Bucket duration in seconds (takes precedence if provided).
            deduplicate_events: If True, uses event_id to prevent double-counting duplicate events.
            strict_validation: If True, malformed events raise InvalidFootfallEventError.
        """
        self.deduplicate_events = bool(deduplicate_events)
        self.strict_validation = bool(strict_validation)

        # Determine and validate bucket size in seconds
        if bucket_seconds is not None:
            if not isinstance(bucket_seconds, (int, float)) or isinstance(bucket_seconds, bool) or bucket_seconds <= 0 or math.isnan(bucket_seconds) or math.isinf(bucket_seconds):
                raise InvalidFootfallConfigError(
                    f"bucket_seconds must be a positive number, got: {bucket_seconds!r}"
                )
            self.bucket_seconds = float(bucket_seconds)
        elif bucket_minutes is not None:
            if not isinstance(bucket_minutes, (int, float)) or isinstance(bucket_minutes, bool) or bucket_minutes <= 0 or math.isnan(bucket_minutes) or math.isinf(bucket_minutes):
                raise InvalidFootfallConfigError(
                    f"bucket_minutes must be a positive number, got: {bucket_minutes!r}"
                )
            self.bucket_seconds = float(bucket_minutes) * 60.0
        else:
            self.bucket_seconds = 900.0  # Default: 15 minutes = 900 seconds

        # Cumulative totals
        self._total_entries: int = 0
        self._total_exits: int = 0
        self._duplicate_events_rejected: int = 0
        self._seen_event_ids: Set[Any] = set()

        # Time-bucket mapping: bucket_index -> FootfallBucket
        self._buckets: Dict[int, FootfallBucket] = {}

        # Entrance breakdown mapping: entrance_id -> {"entries": int, "exits": int, "net_flow": int}
        self._entrance_breakdown: Dict[str, Dict[str, int]] = {}

        self._frames_evaluated: int = 0

        logger.info(
            "FootfallAnalyticsEngine initialized (bucket_seconds=%.1fs, deduplicate=%s)",
            self.bucket_seconds,
            self.deduplicate_events,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FootfallAnalyticsEngine":
        """Factory method to construct engine from a configuration dictionary."""
        if not isinstance(config, dict):
            raise InvalidFootfallConfigError(
                f"Expected dictionary for footfall config, got: {type(config).__name__}"
            )
        return cls(
            bucket_minutes=config.get("bucket_minutes"),
            bucket_seconds=config.get("bucket_seconds"),
            deduplicate_events=bool(config.get("deduplicate_events", True)),
            strict_validation=bool(config.get("strict_validation", False)),
        )

    @property
    def total_entries(self) -> int:
        """Cumulative count of accepted ENTRY events."""
        return self._total_entries

    @property
    def total_exits(self) -> int:
        """Cumulative count of accepted EXIT events."""
        return self._total_exits

    @property
    def total_movement_events(self) -> int:
        """Total movement events (entries + exits)."""
        return self._total_entries + self._total_exits

    @property
    def net_flow(self) -> int:
        """Net flow (entries - exits)."""
        return self._total_entries - self._total_exits

    @property
    def frames_evaluated(self) -> int:
        """Total frames evaluated."""
        return self._frames_evaluated

    @property
    def duplicate_events_rejected(self) -> int:
        """Total duplicate events rejected."""
        return self._duplicate_events_rejected

    @property
    def entrance_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Dictionary of per-entrance breakdown totals."""
        return {k: dict(v) for k, v in self._entrance_breakdown.items()}

    def reset(self) -> None:
        """Reset internal metrics, buckets, and deduplication history."""
        self._total_entries = 0
        self._total_exits = 0
        self._duplicate_events_rejected = 0
        self._seen_event_ids.clear()
        self._buckets.clear()
        self._entrance_breakdown.clear()
        self._frames_evaluated = 0
        logger.info("FootfallAnalyticsEngine reset completed.")

    def process_event(self, event: Union[EntryExitEvent, Dict[str, Any]]) -> bool:
        """
        Process a single movement event, updating totals, deduplication, breakdown, and buckets.

        Args:
            event: EntryExitEvent object or compatible dictionary.

        Returns:
            bool: True if the event was accepted as a footfall entry or exit, False otherwise.
        """
        # Extract fields whether event is an EntryExitEvent or dict
        if isinstance(event, dict):
            event_id = event.get("event_id")
            m_type = str(event.get("movement_type") or event.get("type") or "").upper().strip()
            ts = event.get("timestamp")
            trans_id = event.get("transition_id") or event.get("zone_id") or event.get("zone")
        elif hasattr(event, "movement_type"):
            event_id = getattr(event, "event_id", None)
            m_type = str(getattr(event, "movement_type", "")).upper().strip()
            ts = getattr(event, "timestamp", None)
            trans_id = getattr(event, "transition_id", None) or getattr(event, "zone_id", None)
        else:
            if self.strict_validation:
                raise InvalidFootfallEventError(f"Unsupported event type: {type(event).__name__}")
            logger.warning("Unsupported event object ignored: %r", event)
            return False

        # Validate movement_type
        if m_type not in (MovementType.ENTRY.value, MovementType.EXIT.value, MovementType.ZONE_TRANSITION.value):
            if self.strict_validation:
                raise InvalidFootfallEventError(f"Invalid movement_type: '{m_type}'")
            logger.warning("Invalid movement_type '%s' ignored in event %s", m_type, event_id)
            return False

        # ZONE_TRANSITION is strictly excluded from Footfall counting
        if m_type == MovementType.ZONE_TRANSITION.value:
            return False

        # Validate event_id for deduplication
        if event_id is None:
            if self.strict_validation:
                raise InvalidFootfallEventError("Event missing mandatory event_id")
            # If no event_id, allow processing but cannot deduplicate
            event_id = f"anon_{time.perf_counter()}"

        # Deduplication check
        if self.deduplicate_events:
            if event_id in self._seen_event_ids:
                self._duplicate_events_rejected += 1
                logger.debug("Duplicate event_id '%s' ignored", event_id)
                return False
            self._seen_event_ids.add(event_id)

        # Determine entry vs exit
        is_entry = (m_type == MovementType.ENTRY.value)
        if is_entry:
            self._total_entries += 1
        else:
            self._total_exits += 1

        # Optional entrance breakdown
        entrance_key = str(trans_id).strip() if trans_id else None
        if entrance_key:
            if entrance_key not in self._entrance_breakdown:
                self._entrance_breakdown[entrance_key] = {"entries": 0, "exits": 0, "net_flow": 0}
            if is_entry:
                self._entrance_breakdown[entrance_key]["entries"] += 1
            else:
                self._entrance_breakdown[entrance_key]["exits"] += 1
            self._entrance_breakdown[entrance_key]["net_flow"] = (
                self._entrance_breakdown[entrance_key]["entries"] - self._entrance_breakdown[entrance_key]["exits"]
            )

        # Time-bucket assignment (handles valid video-relative timestamps)
        if ts is not None and isinstance(ts, (int, float)) and not isinstance(ts, bool):
            ts_val = float(ts)
            if not math.isnan(ts_val) and not math.isinf(ts_val) and ts_val >= 0:
                b_idx = int(ts_val // self.bucket_seconds)
                b_start = b_idx * self.bucket_seconds
                b_end = b_start + self.bucket_seconds

                if b_idx not in self._buckets:
                    self._buckets[b_idx] = FootfallBucket(
                        bucket_start=b_start,
                        bucket_end=b_end,
                        entries=0,
                        exits=0,
                        total_movement_events=0,
                        net_flow=0,
                        entry_rate=0.0,
                        exit_rate=0.0,
                        breakdown={},
                    )

                bucket = self._buckets[b_idx]
                if is_entry:
                    bucket.entries += 1
                else:
                    bucket.exits += 1

                bucket.total_movement_events = bucket.entries + bucket.exits
                bucket.net_flow = bucket.entries - bucket.exits

                duration_mins = (bucket.bucket_end - bucket.bucket_start) / 60.0
                if duration_mins > 0:
                    bucket.entry_rate = bucket.entries / duration_mins
                    bucket.exit_rate = bucket.exits / duration_mins

                if entrance_key:
                    if entrance_key not in bucket.breakdown:
                        bucket.breakdown[entrance_key] = {"entries": 0, "exits": 0, "net_flow": 0}
                    if is_entry:
                        bucket.breakdown[entrance_key]["entries"] += 1
                    else:
                        bucket.breakdown[entrance_key]["exits"] += 1
                    bucket.breakdown[entrance_key]["net_flow"] = (
                        bucket.breakdown[entrance_key]["entries"] - bucket.breakdown[entrance_key]["exits"]
                    )
            else:
                logger.warning("Event %s has invalid timestamp value %s, skipping bucket assignment", event_id, ts)
        else:
            logger.warning("Event %s missing valid timestamp, skipping bucket assignment", event_id)

        return True

    def get_summary(self) -> FootfallSummary:
        """
        Generate a snapshot of aggregated footfall totals, sorted time buckets, and breakdowns.
        """
        sorted_buckets = [self._buckets[k] for k in sorted(self._buckets.keys())]
        return FootfallSummary(
            total_entries=self._total_entries,
            total_exits=self._total_exits,
            total_movement_events=self.total_movement_events,
            net_flow=self.net_flow,
            buckets=sorted_buckets,
            entrance_breakdown=self.entrance_breakdown,
        )

    def update(
        self,
        entry_exit_input: Optional[Union[EntryExitResult, List[EntryExitEvent], Dict[str, Any], Any]] = None,
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
        events: Optional[Union[EntryExitResult, List[EntryExitEvent], Dict[str, Any], Any]] = None,
    ) -> FootfallResult:
        """
        Process events for the current frame and return an updated FootfallResult.

        Args:
            entry_exit_input: EntryExitResult, list of EntryExitEvent, or compatible dictionary.
            frame_number: Optional sequence frame number.
            timestamp: Optional playback timestamp in seconds.
            events: Alias for entry_exit_input for backward-compatible naming.

        Returns:
            FootfallResult: Per-frame result container with cumulative footfall summary.
        """
        start_time = time.perf_counter()
        self._frames_evaluated += 1

        target_input = entry_exit_input if entry_exit_input is not None else events
        fn = frame_number
        ts = timestamp
        events_to_process: List[Any] = []

        if isinstance(target_input, EntryExitResult):
            fn = target_input.frame_number if fn is None else fn
            ts = target_input.timestamp if ts is None else ts
            events_to_process = target_input.events
        elif isinstance(target_input, list):
            events_to_process = target_input
        elif isinstance(target_input, dict):
            fn = target_input.get("frame_number", fn)
            ts = target_input.get("timestamp", ts)
            events_to_process = target_input.get("events", [])
        elif target_input is None:
            events_to_process = []
        else:
            if self.strict_validation:
                raise InvalidFootfallEventError(
                    f"Unsupported entry_exit_input type: {type(target_input).__name__}"
                )
            logger.warning("Unsupported entry_exit_input ignored: %r", target_input)
            events_to_process = []

        fn = int(fn) if fn is not None else self._frames_evaluated
        ts = float(ts) if ts is not None else 0.0

        new_entries = 0
        new_exits = 0

        # Process each incoming event
        for ev in events_to_process:
            m_type = ""
            if isinstance(ev, dict):
                m_type = str(ev.get("movement_type") or ev.get("type") or "").upper().strip()
            elif hasattr(ev, "movement_type"):
                m_type = str(getattr(ev, "movement_type", "")).upper().strip()

            accepted = self.process_event(ev)
            if accepted:
                if m_type == MovementType.ENTRY.value:
                    new_entries += 1
                elif m_type == MovementType.EXIT.value:
                    new_exits += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return FootfallResult(
            frame_number=fn,
            timestamp=ts,
            new_entries=new_entries,
            new_exits=new_exits,
            summary=self.get_summary(),
            processing_time_ms=elapsed_ms,
        )

    def draw_footfall(
        self,
        frame_bgr: np.ndarray,
        result: FootfallResult,
    ) -> np.ndarray:
        """
        Draw a Footfall summary badge overlay on the top-right corner of the video frame.

        Args:
            frame_bgr: Source OpenCV BGR image matrix.
            result: FootfallResult from current frame.

        Returns:
            np.ndarray: Annotated frame matrix.
        """
        output = frame_bgr.copy()
        h, w = output.shape[:2]

        card_w, card_h = 240, 75
        x1 = w - card_w - 10
        y1 = 10

        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + card_w, y1 + card_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)
        cv2.rectangle(output, (x1, y1), (x1 + card_w, y1 + card_h), (80, 80, 80), 1)

        # Header
        cv2.putText(
            output,
            "FOOTFALL ANALYTICS",
            (x1 + 10, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Metrics
        flow_sign = "+" if result.net_flow >= 0 else ""
        txt1 = f"Entries: {result.total_entries} | Exits: {result.total_exits}"
        txt2 = f"Net Flow: {flow_sign}{result.net_flow} | Total: {result.total_movement_events}"

        cv2.putText(
            output,
            txt1,
            (x1 + 10, y1 + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            txt2,
            (x1 + 10, y1 + 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 128) if result.net_flow >= 0 else (0, 100, 255),
            1,
            cv2.LINE_AA,
        )

        return output
