"""
Comprehensive tests for ZoneConfigStore.
Tests: save/load round-trip, corrupt fallback, validation rules,
entry_exit dependency check, atomic write, reset, and multi-resolution correctness.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from src.zones.zone_store import (
    ZoneConfigStore,
    _polygon_area,
    _polygon_self_intersects,
)


# ─── Fixtures ───

@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project structure with config.yaml and data dirs."""
    # config.yaml with default zones
    config = {
        "zones": {
            "enabled": True,
            "assignment_mode": "center",
            "boundary_policy": "inside",
            "coordinate_system": "normalized",
            "definitions": [
                {"id": "entrance", "name": "Entrance", "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]},
                {"id": "waiting_area", "name": "Waiting Area", "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]},
            ],
        },
        "entry_exit": {
            "enabled": True,
            "definitions": [
                {"id": "main_entrance", "from": None, "to": "entrance", "type": "ENTRY"},
                {"id": "main_exit", "from": "entrance", "to": None, "type": "EXIT"},
            ],
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Create data/configs directory
    (tmp_path / "data" / "configs").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def store(tmp_project):
    """Return a ZoneConfigStore instance for testing."""
    return ZoneConfigStore(tmp_project)


def _valid_payload(**overrides):
    """Create a minimal valid zone payload."""
    base = {
        "coordinate_system": "normalized",
        "assignment_mode": "center",
        "boundary_policy": "inside",
        "definitions": [
            {
                "id": "entrance",
                "name": "Entrance",
                "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
                "color": "#FF8000",
                "assignment_mode": "center",
                "enabled": True,
            },
            {
                "id": "waiting_area",
                "name": "Waiting Area",
                "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
                "color": "#00C8FF",
                "assignment_mode": "center",
                "enabled": True,
            },
        ],
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════
# 1. Save → Load round-trip
# ═══════════════════════════════════════════════════════

class TestSaveLoadRoundTrip:

    def test_save_and_load_returns_same_definitions(self, store):
        payload = _valid_payload()
        saved = store.save(payload)
        loaded = store.load()

        assert loaded["version"] == saved["version"]
        assert len(loaded["definitions"]) == 2
        assert loaded["definitions"][0]["id"] == "entrance"
        assert loaded["definitions"][1]["id"] == "waiting_area"

    def test_version_increments_on_each_save(self, store):
        payload = _valid_payload()
        first = store.save(payload)
        second = store.save(payload)
        assert second["version"] == first["version"] + 1

    def test_updated_at_changes_on_save(self, store):
        payload = _valid_payload()
        first = store.save(payload)
        second = store.save(payload)
        assert second["updated_at"] != first["updated_at"]

    def test_polygon_coordinates_preserved(self, store):
        payload = _valid_payload()
        store.save(payload)
        loaded = store.load()
        poly = loaded["definitions"][0]["polygon"]
        assert poly == [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]

    def test_enabled_flag_preserved(self, store):
        payload = _valid_payload()
        payload["definitions"][1]["enabled"] = False
        store.save(payload)
        loaded = store.load()
        assert loaded["definitions"][0]["enabled"] is True
        assert loaded["definitions"][1]["enabled"] is False


# ═══════════════════════════════════════════════════════
# 2. Fallback behavior
# ═══════════════════════════════════════════════════════

class TestFallback:

    def test_load_without_json_file_returns_defaults(self, store):
        # No zones.json exists yet
        data = store.load()
        assert isinstance(data, dict)
        assert "definitions" in data
        assert len(data["definitions"]) == 2
        assert data["definitions"][0]["id"] == "entrance"

    def test_corrupt_json_falls_back_to_defaults(self, store, tmp_project):
        # Write invalid JSON
        zones_path = tmp_project / "data" / "configs" / "zones.json"
        with open(zones_path, "w") as f:
            f.write("{corrupt json data!!!}")

        data = store.load()
        assert isinstance(data, dict)
        assert len(data["definitions"]) == 2
        assert data["definitions"][0]["id"] == "entrance"

    def test_empty_json_falls_back_to_defaults(self, store, tmp_project):
        zones_path = tmp_project / "data" / "configs" / "zones.json"
        with open(zones_path, "w") as f:
            json.dump({"something": "else"}, f)

        data = store.load()
        # Missing "definitions" key → fallback
        assert len(data["definitions"]) == 2

    def test_json_with_non_list_definitions_falls_back(self, store, tmp_project):
        zones_path = tmp_project / "data" / "configs" / "zones.json"
        with open(zones_path, "w") as f:
            json.dump({"definitions": "not_a_list"}, f)

        data = store.load()
        assert isinstance(data["definitions"], list)
        assert len(data["definitions"]) == 2


# ═══════════════════════════════════════════════════════
# 3. Validation — each rule tested
# ═══════════════════════════════════════════════════════

class TestValidation:

    def test_valid_payload_passes(self, store):
        errors = store.validate(_valid_payload())
        assert errors == []

    # ─── Polygon vertex count ───
    def test_fewer_than_3_vertices_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[0.0, 0.0], [0.5, 0.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("at least 3" in e for e in errors)

    # ─── Out of range ───
    def test_coordinates_outside_0_1_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[0.0, 0.0], [1.5, 0.0], [1.5, 1.0], [0.0, 1.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("0.0–1.0" in e for e in errors)

    def test_negative_coordinate_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[-0.1, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("0.0–1.0" in e for e in errors)

    # ─── NaN / Infinity ───
    def test_nan_coordinate_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[float('nan'), 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("NaN" in e for e in errors)

    def test_infinity_coordinate_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[float('inf'), 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("NaN or Infinity" in e for e in errors)

    # ─── Boolean coordinates ───
    def test_boolean_coordinate_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[True, False], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("boolean" in e.lower() for e in errors)

    # ─── Duplicate ID ───
    def test_duplicate_id_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"].append({
            "id": "entrance", "name": "Duplicate",
            "polygon": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3]],
            "color": "#FF0000",
        })
        errors = store.validate(payload)
        assert any("duplicate" in e.lower() for e in errors)

    # ─── Invalid ID format ───
    def test_uppercase_id_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "BadZone", "name": "Bad",
            "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("lowercase" in e.lower() for e in errors)

    def test_special_chars_in_id_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad zone!", "name": "Bad",
            "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("lowercase" in e.lower() for e in errors)

    # ─── Empty name ───
    def test_empty_name_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "",
            "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("name" in e.lower() for e in errors)

    def test_too_long_name_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "A" * 100,
            "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("length" in e.lower() for e in errors)

    # ─── Invalid color ───
    def test_invalid_hex_color_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "red",
        }]
        errors = store.validate(payload)
        assert any("hex color" in e.lower() for e in errors)

    def test_short_hex_color_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "#FFF",
        }]
        errors = store.validate(payload)
        assert any("hex color" in e.lower() for e in errors)

    # ─── Invalid assignment_mode ───
    def test_invalid_assignment_mode_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "#FF0000",
            "assignment_mode": "top_left",
        }]
        errors = store.validate(payload)
        assert any("assignment_mode" in e for e in errors)

    # ─── Self-intersecting polygon ───
    def test_self_intersecting_polygon_rejected(self, store):
        payload = _valid_payload()
        # A bowtie shape: edges cross
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[0.0, 0.0], [0.5, 0.5], [0.5, 0.0], [0.0, 0.5]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("self-intersecting" in e.lower() for e in errors)

    # ─── Zero-area polygon ───
    def test_collinear_polygon_rejected(self, store):
        payload = _valid_payload()
        # All points on a line
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("zero" in e.lower() for e in errors)

    # ─── Too many zones ───
    def test_too_many_zones_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [
            {
                "id": f"zone_{i}", "name": f"Zone {i}",
                "polygon": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]],
                "color": "#FF0000",
            }
            for i in range(25)
        ]
        errors = store.validate(payload)
        assert any("maximum" in e.lower() for e in errors)

    # ─── Missing definitions ───
    def test_missing_definitions_rejected(self, store):
        errors = store.validate({"coordinate_system": "normalized"})
        assert any("definitions" in e.lower() for e in errors)

    def test_non_list_definitions_rejected(self, store):
        errors = store.validate({"definitions": "not_a_list"})
        assert any("array" in e.lower() for e in errors)

    # ─── Missing id ───
    def test_missing_id_rejected(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "name": "No ID",
            "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            "color": "#FF0000",
        }]
        errors = store.validate(payload)
        assert any("'id'" in e for e in errors)


# ═══════════════════════════════════════════════════════
# 4. Entry/Exit dependency check
# ═══════════════════════════════════════════════════════

class TestEntryExitDependency:

    def test_removing_referenced_zone_rejected(self, store):
        """Removing 'entrance' (used in entry_exit rules) should fail."""
        payload = _valid_payload()
        # Remove entrance, keep only waiting_area
        payload["definitions"] = [{
            "id": "waiting_area", "name": "Waiting Area",
            "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
            "color": "#00C8FF",
        }]
        errors = store.validate(payload)
        assert any("entrance" in e and "entry_exit" in e for e in errors)

    def test_keeping_referenced_zone_passes(self, store):
        """Keeping 'entrance' (used in entry_exit rules) should pass."""
        errors = store.validate(_valid_payload())
        assert errors == []

    def test_save_with_missing_referenced_zone_raises(self, store):
        """Save should raise ValueError when referenced zone is missing."""
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "waiting_area", "name": "Waiting Area",
            "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
            "color": "#00C8FF",
        }]
        with pytest.raises(ValueError, match="entrance"):
            store.save(payload)


# ═══════════════════════════════════════════════════════
# 5. Reset
# ═══════════════════════════════════════════════════════

class TestReset:

    def test_reset_returns_defaults(self, store):
        # First save something different
        payload = _valid_payload()
        payload["definitions"].append({
            "id": "new_zone", "name": "New Zone",
            "polygon": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
            "color": "#00FF00",
        })
        store.save(payload)

        # Reset
        defaults = store.reset()
        assert len(defaults["definitions"]) == 2
        assert defaults["definitions"][0]["id"] == "entrance"
        assert defaults["version"] == 1

    def test_reset_persists_to_file(self, store, tmp_project):
        store.save(_valid_payload())
        store.reset()
        # Load from file directly
        with open(tmp_project / "data" / "configs" / "zones.json") as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["definitions"]) == 2


# ═══════════════════════════════════════════════════════
# 6. Atomic write safety
# ═══════════════════════════════════════════════════════

class TestAtomicWrite:

    def test_file_is_valid_json_after_save(self, store, tmp_project):
        store.save(_valid_payload())
        with open(tmp_project / "data" / "configs" / "zones.json") as f:
            data = json.load(f)
        assert data["version"] >= 1
        assert isinstance(data["definitions"], list)

    def test_no_temp_files_left_after_save(self, store, tmp_project):
        store.save(_valid_payload())
        config_dir = tmp_project / "data" / "configs"
        tmp_files = [f for f in config_dir.iterdir() if f.suffix == ".tmp"]
        assert len(tmp_files) == 0


# ═══════════════════════════════════════════════════════
# 7. load_as_engine_config
# ═══════════════════════════════════════════════════════

class TestLoadAsEngineConfig:

    def test_returns_engine_compatible_dict(self, store):
        store.save(_valid_payload())
        cfg = store.load_as_engine_config()
        assert "coordinate_system" in cfg
        assert "assignment_mode" in cfg
        assert "boundary_policy" in cfg
        assert isinstance(cfg["definitions"], list)

    def test_filters_disabled_zones(self, store):
        payload = _valid_payload()
        payload["definitions"][1]["enabled"] = False
        store.save(payload)

        cfg = store.load_as_engine_config(enabled_only=True)
        assert len(cfg["definitions"]) == 1
        assert cfg["definitions"][0]["id"] == "entrance"

    def test_includes_all_when_enabled_only_false(self, store):
        payload = _valid_payload()
        payload["definitions"][1]["enabled"] = False
        store.save(payload)

        cfg = store.load_as_engine_config(enabled_only=False)
        assert len(cfg["definitions"]) == 2


# ═══════════════════════════════════════════════════════
# 8. Multi-resolution correctness
# ═══════════════════════════════════════════════════════

class TestMultiResolution:

    def test_same_relative_position_640x480(self, store):
        """Saved normalized zones work at 640x480."""
        from src.zones import ZoneEngine

        store.save(_valid_payload())
        cfg = store.load_as_engine_config()
        engine = ZoneEngine.from_config(cfg)
        engine.set_frame_size(640, 480)

        # Point at center of entrance zone (0.25, 0.5) → pixel (160, 240)
        entrance = engine.get_zone("entrance")
        assert entrance is not None
        assert entrance.contains_point((160, 240))
        assert not entrance.contains_point((480, 240))

    def test_same_relative_position_1920x1080(self, store):
        """Saved normalized zones work at 1920x1080."""
        from src.zones import ZoneEngine

        store.save(_valid_payload())
        cfg = store.load_as_engine_config()
        engine = ZoneEngine.from_config(cfg)
        engine.set_frame_size(1920, 1080)

        # Point at center of entrance zone (0.25, 0.5) → pixel (480, 540)
        entrance = engine.get_zone("entrance")
        assert entrance is not None
        assert entrance.contains_point((480, 540))
        assert not entrance.contains_point((1440, 540))

    def test_zones_consistent_across_resolutions(self, store):
        """A normalized point inside a zone works identically at both resolutions."""
        from src.zones import ZoneEngine

        store.save(_valid_payload())
        cfg = store.load_as_engine_config()

        # 640x480
        engine_640 = ZoneEngine.from_config(cfg)
        engine_640.set_frame_size(640, 480)
        assert engine_640.get_zone("entrance").contains_point((160, 240))  # (0.25, 0.5) of 640x480

        # 1920x1080
        engine_1920 = ZoneEngine.from_config(cfg)
        engine_1920.set_frame_size(1920, 1080)
        assert engine_1920.get_zone("entrance").contains_point((480, 540))  # (0.25, 0.5) of 1920x1080


# ═══════════════════════════════════════════════════════
# 9. Polygon geometry helpers
# ═══════════════════════════════════════════════════════

class TestPolygonHelpers:

    def test_polygon_area_square(self):
        """Unit square should have area 1.0."""
        assert abs(_polygon_area([[0, 0], [1, 0], [1, 1], [0, 1]]) - 1.0) < 1e-6

    def test_polygon_area_triangle(self):
        assert abs(_polygon_area([[0, 0], [1, 0], [0.5, 1]]) - 0.5) < 1e-6

    def test_polygon_area_collinear_is_zero(self):
        assert _polygon_area([[0, 0], [0.5, 0], [1, 0]]) < 1e-6

    def test_self_intersection_bowtie(self):
        # Bowtie: edges definitely cross
        assert _polygon_self_intersects([[0, 0], [1, 1], [1, 0], [0, 1]])

    def test_no_self_intersection_square(self):
        assert not _polygon_self_intersects([[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_no_self_intersection_triangle(self):
        assert not _polygon_self_intersects([[0, 0], [1, 0], [0.5, 1]])

    def test_no_self_intersection_pentagon(self):
        import math as m
        pts = [[0.5 + 0.4 * m.cos(2 * m.pi * i / 5), 0.5 + 0.4 * m.sin(2 * m.pi * i / 5)] for i in range(5)]
        assert not _polygon_self_intersects(pts)


# ═══════════════════════════════════════════════════════
# 10. Save validation rejects (HTTP 422 path)
# ═══════════════════════════════════════════════════════

class TestSaveValidationRejects:

    def test_save_with_2_vertices_raises(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[0.0, 0.0], [0.5, 0.0]],
            "color": "#FF0000",
        }]
        with pytest.raises(ValueError, match="at least 3"):
            store.save(payload)

    def test_save_with_duplicate_id_raises(self, store):
        payload = _valid_payload()
        payload["definitions"].append({
            "id": "entrance", "name": "Dup",
            "polygon": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3]],
            "color": "#FF0000",
        })
        with pytest.raises(ValueError, match="duplicate"):
            store.save(payload)

    def test_save_with_self_intersecting_raises(self, store):
        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[0.0, 0.0], [0.5, 0.5], [0.5, 0.0], [0.0, 0.5]],
            "color": "#FF0000",
        }]
        with pytest.raises(ValueError, match="self-intersecting"):
            store.save(payload)

    def test_file_not_written_on_validation_failure(self, store, tmp_project):
        zones_path = tmp_project / "data" / "configs" / "zones.json"
        assert not zones_path.exists()

        payload = _valid_payload()
        payload["definitions"] = [{
            "id": "bad", "name": "Bad",
            "polygon": [[0.0, 0.0], [0.5, 0.0]],
            "color": "#FF0000",
        }]
        with pytest.raises(ValueError):
            store.save(payload)

        # File should not have been created
        assert not zones_path.exists()
