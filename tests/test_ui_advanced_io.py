"""Tests for Advanced page import/export logic."""

from osmose.config.reader import OsmoseConfigReader
from osmose.config.writer import OsmoseConfigWriter


def test_import_config_merges(tmp_path):
    """Importing a config should merge keys into existing state."""
    config_file = tmp_path / "test.csv"
    config_file.write_text("simulation.nspecies ; 5\nspecies.k.sp0 ; 0.3\n")

    reader = OsmoseConfigReader()
    loaded = reader.read_file(config_file)
    assert loaded["simulation.nspecies"] == "5"
    assert loaded["species.k.sp0"] == "0.3"


def test_export_config_roundtrip(tmp_path):
    """Export should produce files that can be read back."""
    config = {"simulation.nspecies": "3", "species.k.sp0": "0.2"}
    writer = OsmoseConfigWriter()
    writer.write(config, tmp_path)

    reader = OsmoseConfigReader()
    loaded = reader.read(tmp_path / "osm_all-parameters.csv")
    assert loaded["simulation.nspecies"] == "3"


def test_preview_import_diff():
    """compute_import_diff should return only changed/new keys with old and new values."""
    from ui.pages.advanced import compute_import_diff

    current = {"a": "1", "b": "2", "c": "3"}
    incoming = {"a": "1", "b": "99", "d": "4"}
    diff = compute_import_diff(current, incoming)
    assert len(diff) == 2
    changed = {d["key"]: d for d in diff}
    assert changed["b"]["old"] == "2"
    assert changed["b"]["new"] == "99"
    assert changed["d"]["old"] is None
    assert changed["d"]["new"] == "4"


def test_preview_import_diff_empty():
    """compute_import_diff should return empty list when configs are identical."""
    from ui.pages.advanced import compute_import_diff

    current = {"a": "1", "b": "2"}
    incoming = {"a": "1", "b": "2"}
    diff = compute_import_diff(current, incoming)
    assert diff == []


def test_preview_import_diff_all_new():
    """compute_import_diff should show all keys as new when current is empty."""
    from ui.pages.advanced import compute_import_diff

    current = {}
    incoming = {"x": "10", "y": "20"}
    diff = compute_import_diff(current, incoming)
    assert len(diff) == 2
    for d in diff:
        assert d["old"] is None
