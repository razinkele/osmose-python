"""Tests for setup page input syncing to state."""

from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS


def test_setup_global_keys():
    """Setup sync should cover all non-advanced simulation fields."""
    from ui.pages.setup import SETUP_GLOBAL_KEYS

    expected_patterns = [f.key_pattern for f in SIMULATION_FIELDS if not f.advanced]
    for pattern in expected_patterns:
        assert pattern in SETUP_GLOBAL_KEYS, f"Missing key: {pattern}"


def test_setup_species_sync_keys():
    """Species fields should resolve correctly for a given species index."""
    from ui.pages.setup import get_species_keys

    keys = get_species_keys(species_idx=0, show_advanced=False)
    # Should include growth K for species 0
    assert any("species.k.sp0" == k for k in keys)
    # Should NOT include advanced fields
    adv_keys_patterns = [f.key_pattern for f in SPECIES_FIELDS if f.advanced]
    for pattern in adv_keys_patterns:
        resolved = pattern.replace("{idx}", "0")
        assert resolved not in keys


def test_setup_species_sync_keys_with_advanced():
    from ui.pages.setup import get_species_keys

    keys = get_species_keys(species_idx=1, show_advanced=True)
    # Should include all species fields for index 1
    assert any("sp1" in k for k in keys)
