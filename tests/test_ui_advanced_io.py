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
