"""
Persistent Zone Configuration Store.
Thread-safe load / validate / save / reset of zone definitions in data/configs/zones.json.
Uses atomic writes (temp file + rename) and a threading lock for concurrent safety.
"""

import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Validation constants
MAX_ZONES = 20
MAX_NAME_LENGTH = 64
VALID_ID_PATTERN = re.compile(r"^[a-z0-9_\-]+$")
VALID_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
VALID_ASSIGNMENT_MODES = {"center", "bottom_center"}
MIN_POLYGON_AREA = 1e-6  # Minimum normalized area to reject degenerate polygons


def _segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float],
                        p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """
    Check if line segment (p1→p2) properly intersects segment (p3→p4).
    Uses the CCW orientation test. Shared endpoints are NOT considered an intersection.
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear overlap checks omitted: for polygon self-intersection
    # in zone configs, proper crossing is the primary concern.
    return False


def _polygon_self_intersects(polygon: List[List[float]]) -> bool:
    """Check if a polygon's edges properly intersect (self-crossing)."""
    n = len(polygon)
    if n < 4:
        return False  # Triangle cannot self-intersect

    edges = []
    for i in range(n):
        edges.append((tuple(polygon[i]), tuple(polygon[(i + 1) % n])))

    for i in range(n):
        for j in range(i + 2, n):
            # Skip adjacent edges (they share a vertex)
            if i == 0 and j == n - 1:
                continue
            if _segments_intersect(edges[i][0], edges[i][1],
                                   edges[j][0], edges[j][1]):
                return True
    return False


def _polygon_area(polygon: List[List[float]]) -> float:
    """Compute the area of a polygon using the Shoelace formula. Returns absolute area."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


class ZoneConfigStore:
    """
    Thread-safe persistent storage for zone configurations.
    Zones are stored in data/configs/zones.json with normalized (0.0-1.0) coordinates.
    Falls back to config.yaml defaults when the JSON file is missing or corrupt.
    """

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._zones_path = project_root / "data" / "configs" / "zones.json"
        self._config_yaml_path = project_root / "config.yaml"
        self._lock = threading.Lock()

    @property
    def zones_path(self) -> Path:
        return self._zones_path

    # ──────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────

    def load(self) -> Dict[str, Any]:
        """
        Load zone configuration. Priority:
        1. data/configs/zones.json (if valid)
        2. config.yaml defaults (fallback)
        """
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> Dict[str, Any]:
        if self._zones_path.exists():
            try:
                with open(self._zones_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "definitions" in data and isinstance(data["definitions"], list):
                    return data
                else:
                    logger.warning("zones.json has invalid structure, falling back to defaults.")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("zones.json is corrupt or unreadable (%s), falling back to defaults.", e)

        return self._get_defaults_unlocked()

    # ──────────────────────────────────────────────
    # Defaults from config.yaml
    # ──────────────────────────────────────────────

    def get_defaults(self) -> Dict[str, Any]:
        """Get default zone config from config.yaml."""
        with self._lock:
            return self._get_defaults_unlocked()

    def _get_defaults_unlocked(self) -> Dict[str, Any]:
        zones_cfg: Dict[str, Any] = {}
        if self._config_yaml_path.exists():
            try:
                with open(self._config_yaml_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                zones_cfg = cfg.get("zones", {})
            except Exception as e:
                logger.error("Failed to read config.yaml for zone defaults: %s", e)

        # Convert to canonical store format
        raw_defs = zones_cfg.get("definitions", [])
        if not isinstance(raw_defs, list):
            raw_defs = []

        default_colors = ["#FF8000", "#00C8FF", "#B400FF", "#00FF80", "#FF0080"]
        definitions = []
        for i, z in enumerate(raw_defs):
            definitions.append({
                "id": z.get("id") or z.get("zone_id", f"zone_{i}"),
                "name": z.get("name") or z.get("zone_name", f"Zone {i}"),
                "polygon": z.get("polygon", []),
                "color": default_colors[i % len(default_colors)],
                "assignment_mode": zones_cfg.get("assignment_mode", "center"),
                "enabled": True,
            })

        return {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "coordinate_system": zones_cfg.get("coordinate_system", "normalized"),
            "assignment_mode": zones_cfg.get("assignment_mode", "center"),
            "boundary_policy": zones_cfg.get("boundary_policy", "inside"),
            "definitions": definitions,
        }

    # ──────────────────────────────────────────────
    # Validate
    # ──────────────────────────────────────────────

    def validate(self, zones_data: Dict[str, Any]) -> List[str]:
        """
        Validate a zone configuration payload.
        Returns a list of human-readable error strings. Empty list = valid.
        """
        errors: List[str] = []

        if not isinstance(zones_data, dict):
            return ["Request body must be a JSON object."]

        definitions = zones_data.get("definitions")
        if definitions is None:
            return ["Missing required field 'definitions'."]
        if not isinstance(definitions, list):
            return ["'definitions' must be an array of zone objects."]

        if len(definitions) > MAX_ZONES:
            errors.append(f"Too many zones: {len(definitions)} exceeds maximum of {MAX_ZONES}.")

        seen_ids: set = set()

        for i, z in enumerate(definitions):
            prefix = f"Zone #{i + 1}"
            if not isinstance(z, dict):
                errors.append(f"{prefix}: must be a JSON object.")
                continue

            # ─── id ───
            zid = z.get("id")
            if not zid or not isinstance(zid, str):
                errors.append(f"{prefix}: 'id' is required and must be a non-empty string.")
                continue
            zid = zid.strip()
            if not VALID_ID_PATTERN.match(zid):
                errors.append(
                    f"Zone '{zid}': 'id' must contain only lowercase letters, digits, "
                    f"underscores or hyphens. Got: '{zid}'."
                )
            if zid in seen_ids:
                errors.append(f"Zone '{zid}': duplicate zone id.")
            seen_ids.add(zid)
            prefix = f"Zone '{zid}'"

            # ─── name ───
            name = z.get("name")
            if not name or not isinstance(name, str) or not name.strip():
                errors.append(f"{prefix}: 'name' is required and must be a non-empty string.")
            elif len(name.strip()) > MAX_NAME_LENGTH:
                errors.append(f"{prefix}: 'name' exceeds maximum length of {MAX_NAME_LENGTH} characters.")

            # ─── color ───
            color = z.get("color")
            if color is not None:
                if not isinstance(color, str) or not VALID_HEX_COLOR.match(color):
                    errors.append(f"{prefix}: 'color' must be a valid hex color (#RRGGBB). Got: '{color}'.")

            # ─── assignment_mode ───
            mode = z.get("assignment_mode")
            if mode is not None:
                if not isinstance(mode, str) or mode.lower().strip() not in VALID_ASSIGNMENT_MODES:
                    errors.append(
                        f"{prefix}: 'assignment_mode' must be 'center' or 'bottom_center'. Got: '{mode}'."
                    )

            # ─── polygon ───
            polygon = z.get("polygon")
            if not polygon or not isinstance(polygon, list):
                errors.append(f"{prefix}: 'polygon' is required and must be an array of [x, y] pairs.")
                continue
            if len(polygon) < 3:
                errors.append(f"{prefix}: polygon must have at least 3 vertices (got {len(polygon)}).")
                continue

            poly_valid = True
            for vi, pt in enumerate(polygon):
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    errors.append(f"{prefix}: vertex {vi} must be a [x, y] pair.")
                    poly_valid = False
                    continue
                x, y = pt
                if isinstance(x, bool) or isinstance(y, bool):
                    errors.append(f"{prefix}: vertex {vi} coordinates must be numbers, not booleans.")
                    poly_valid = False
                    continue
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    errors.append(f"{prefix}: vertex {vi} coordinates must be numbers. Got: ({x!r}, {y!r}).")
                    poly_valid = False
                    continue
                if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                    errors.append(f"{prefix}: vertex {vi} has NaN or Infinity value.")
                    poly_valid = False
                    continue
                if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                    errors.append(
                        f"{prefix}: vertex {vi} coordinates must be in range 0.0–1.0 "
                        f"(normalized). Got: ({x}, {y})."
                    )
                    poly_valid = False

            if poly_valid:
                # Check self-intersection
                if _polygon_self_intersects(polygon):
                    errors.append(f"{prefix}: polygon edges cross each other (self-intersecting).")

                # Check degenerate area
                area = _polygon_area(polygon)
                if area < MIN_POLYGON_AREA:
                    errors.append(
                        f"{prefix}: polygon area is zero or near-zero ({area:.8f}). "
                        f"Vertices may be collinear or duplicate."
                    )

        # ─── entry_exit dependency check ───
        if not errors:
            self._validate_entry_exit_deps(seen_ids, errors)

        return errors

    def _validate_entry_exit_deps(self, zone_ids: set, errors: List[str]) -> None:
        """Check that entry_exit rules don't reference zones being removed."""
        try:
            if not self._config_yaml_path.exists():
                return
            with open(self._config_yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            ee_cfg = cfg.get("entry_exit", {})
            if not ee_cfg.get("enabled", False):
                return
            ee_defs = ee_cfg.get("definitions", [])
            if not isinstance(ee_defs, list):
                return

            for rule in ee_defs:
                rule_id = rule.get("id", "unknown")
                from_zone = rule.get("from")
                to_zone = rule.get("to")
                if from_zone and from_zone not in zone_ids:
                    errors.append(
                        f"Cannot remove zone '{from_zone}': it is referenced by "
                        f"entry_exit rule '{rule_id}' (from field)."
                    )
                if to_zone and to_zone not in zone_ids:
                    errors.append(
                        f"Cannot remove zone '{to_zone}': it is referenced by "
                        f"entry_exit rule '{rule_id}' (to field)."
                    )
        except Exception as e:
            logger.warning("Could not validate entry_exit dependencies: %s", e)

    # ──────────────────────────────────────────────
    # Save (atomic write)
    # ──────────────────────────────────────────────

    def save(self, zones_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and atomically save zone configuration.
        Raises ValueError with combined error messages on validation failure.
        Returns the saved data (with version and updated_at applied).
        """
        errors = self.validate(zones_data)
        if errors:
            raise ValueError("\n".join(errors))

        with self._lock:
            # Read current version
            current = self._load_unlocked()
            current_version = current.get("version", 0)
            if not isinstance(current_version, int):
                current_version = 0

            new_version = current_version + 1
            now_iso = datetime.now(timezone.utc).isoformat()

            # Normalize definitions
            normalized_defs = []
            for z in zones_data.get("definitions", []):
                normalized_defs.append({
                    "id": z["id"].strip(),
                    "name": z.get("name", z["id"]).strip(),
                    "polygon": [[round(float(x), 6), round(float(y), 6)] for x, y in z["polygon"]],
                    "color": z.get("color", "#FF8000"),
                    "assignment_mode": z.get("assignment_mode", "center").lower().strip(),
                    "enabled": z.get("enabled", True),
                })

            save_payload = {
                "version": new_version,
                "updated_at": now_iso,
                "coordinate_system": zones_data.get("coordinate_system", "normalized"),
                "assignment_mode": zones_data.get("assignment_mode", "center"),
                "boundary_policy": zones_data.get("boundary_policy", "inside"),
                "definitions": normalized_defs,
            }

            self._atomic_write(save_payload)
            logger.info(
                "Zones saved: version=%d, zones=%d, at=%s",
                new_version, len(normalized_defs), now_iso,
            )
            return save_payload

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        """Write JSON to a temp file then atomically replace the target."""
        self._zones_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._zones_path.parent),
            prefix=".zones_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            # Atomic rename (works on Windows if target doesn't exist, so remove first)
            if self._zones_path.exists():
                os.replace(tmp_path, str(self._zones_path))
            else:
                os.rename(tmp_path, str(self._zones_path))
        except Exception:
            # Cleanup temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ──────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        """Reset zone configuration to config.yaml defaults and persist."""
        with self._lock:
            defaults = self._get_defaults_unlocked()
            defaults["version"] = 1
            defaults["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._atomic_write(defaults)
            logger.info("Zones reset to config.yaml defaults.")
            return defaults

    # ──────────────────────────────────────────────
    # Helpers for engine construction
    # ──────────────────────────────────────────────

    def load_as_engine_config(self, enabled_only: bool = True) -> Dict[str, Any]:
        """
        Load zones and return a dict suitable for ZoneEngine.from_config().
        If enabled_only=True, only zones with enabled=True are included.
        """
        data = self.load()
        definitions = data.get("definitions", [])
        if enabled_only:
            definitions = [d for d in definitions if d.get("enabled", True)]
        return {
            "coordinate_system": data.get("coordinate_system", "normalized"),
            "assignment_mode": data.get("assignment_mode", "center"),
            "boundary_policy": data.get("boundary_policy", "inside"),
            "definitions": definitions,
        }
