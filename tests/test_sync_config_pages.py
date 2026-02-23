"""Tests for config page input syncing (Grid, Forcing, Fishing, Movement)."""

from osmose.schema.grid import GRID_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from osmose.schema.fishing import FISHING_FIELDS
from osmose.schema.movement import MOVEMENT_FIELDS


def test_grid_global_keys():
    from ui.pages.grid import GRID_GLOBAL_KEYS

    expected = [f.key_pattern for f in GRID_FIELDS if not f.indexed]
    for key in expected:
        assert key in GRID_GLOBAL_KEYS


def test_forcing_global_keys():
    from ui.pages.forcing import FORCING_GLOBAL_KEYS

    expected = [f.key_pattern for f in LTL_FIELDS if not f.indexed]
    for key in expected:
        assert key in FORCING_GLOBAL_KEYS


def test_fishing_global_keys():
    from ui.pages.fishing import FISHING_GLOBAL_KEYS

    expected = [f.key_pattern for f in FISHING_FIELDS if not f.indexed]
    for key in expected:
        assert key in FISHING_GLOBAL_KEYS


def test_movement_global_keys():
    from ui.pages.movement import MOVEMENT_GLOBAL_KEYS

    expected = [f.key_pattern for f in MOVEMENT_FIELDS if not f.indexed]
    for key in expected:
        assert key in MOVEMENT_GLOBAL_KEYS


def test_movement_uses_dynamic_species_count():
    """Movement page should read species count from state, not hardcode 3."""
    import inspect
    from ui.pages.movement import movement_server

    source = inspect.getsource(movement_server)
    assert "range(3)" not in source, "Movement page still hardcodes 3 species"
