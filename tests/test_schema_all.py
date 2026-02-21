from osmose.schema.registry import ParameterRegistry
from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from osmose.schema.grid import GRID_FIELDS
from osmose.schema.predation import PREDATION_FIELDS
from osmose.schema.fishing import FISHING_FIELDS
from osmose.schema.movement import MOVEMENT_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from osmose.schema.output import OUTPUT_FIELDS
from osmose.schema.bioenergetics import BIOENERGETICS_FIELDS
from osmose.schema.economics import ECONOMICS_FIELDS


def build_full_registry() -> ParameterRegistry:
    reg = ParameterRegistry()
    for fields in [
        SIMULATION_FIELDS, SPECIES_FIELDS, GRID_FIELDS,
        PREDATION_FIELDS, FISHING_FIELDS, MOVEMENT_FIELDS,
        LTL_FIELDS, OUTPUT_FIELDS, BIOENERGETICS_FIELDS,
        ECONOMICS_FIELDS,
    ]:
        for f in fields:
            reg.register(f)
    return reg


def test_full_registry_has_all_categories():
    reg = build_full_registry()
    cats = set(reg.categories())
    expected = {"simulation", "growth", "reproduction", "predation",
                "mortality", "fishing", "grid", "movement", "ltl",
                "output", "bioenergetics", "economics"}
    assert expected.issubset(cats), f"Missing categories: {expected - cats}"


def test_full_registry_param_count():
    reg = build_full_registry()
    total = len(reg.all_fields())
    assert total >= 150, f"Only {total} params registered, expected >= 150"


def test_grid_fields_present():
    assert len(GRID_FIELDS) >= 8


def test_output_fields_present():
    assert len(OUTPUT_FIELDS) >= 50


def test_ltl_fields_present():
    assert len(LTL_FIELDS) >= 5


def test_fishing_fields_present():
    assert len(FISHING_FIELDS) >= 10


def test_movement_fields_present():
    assert len(MOVEMENT_FIELDS) >= 5


def test_no_duplicate_key_patterns():
    reg = build_full_registry()
    patterns = [f.key_pattern for f in reg.all_fields()]
    duplicates = [p for p in patterns if patterns.count(p) > 1]
    assert duplicates == [], f"Duplicate key patterns: {set(duplicates)}"
