# OSMOSE Python Port — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python orchestration layer and Shiny web interface for the OSMOSE marine ecosystem simulator, replacing the R package while keeping the Java engine unchanged.

**Architecture:** Monolithic Shiny for Python app with a schema-driven parameter system. The `osmose/` core library handles config I/O, Java execution, NetCDF output reading, and calibration. The `ui/` layer auto-generates forms from schema metadata. Shinyswatch (superhero theme) provides styling.

**Tech Stack:** Python 3.12, Shiny for Python, shinyswatch, xarray, plotly, pymoo, scikit-learn, pandas, jinja2, asyncio, Docker

**Design doc:** `docs/plans/2026-02-21-osmose-python-port-design.md`

---

## Phase 1: Project Scaffolding & Schema Foundation

### Task 1: Project setup and dependencies

**Files:**
- Create: `osmose-python/pyproject.toml`
- Create: `osmose-python/osmose/__init__.py`
- Create: `osmose-python/ui/__init__.py`
- Create: `osmose-python/tests/__init__.py`

**Step 1: Create project directory structure**

```bash
mkdir -p osmose-python/{osmose/{schema,config,calibration},ui/{pages,components},data/{examples,defaults,scenarios},osmose-java,tests}
```

**Step 2: Write pyproject.toml**

```toml
[project]
name = "osmose-python"
version = "0.1.0"
description = "Python orchestration layer and Shiny web interface for the OSMOSE marine ecosystem simulator"
requires-python = ">=3.12"
dependencies = [
    "shiny>=1.3.0",
    "shinyswatch>=0.7.0",
    "pandas>=2.2",
    "xarray>=2024.1",
    "netCDF4>=1.6",
    "plotly>=5.18",
    "jinja2>=3.1",
    "pymoo>=0.6",
    "scikit-learn>=1.4",
    "SALib>=1.5",
    "numpy>=1.26",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.1",
    "ruff>=0.3",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py312"
line-length = 100
```

**Step 3: Create package init files**

```python
# osmose-python/osmose/__init__.py
"""OSMOSE Python - orchestration layer for the OSMOSE marine ecosystem simulator."""
__version__ = "0.1.0"
```

```python
# osmose-python/ui/__init__.py
# osmose-python/tests/__init__.py
# (empty init files)
```

**Step 4: Install in development mode**

```bash
cd osmose-python && pip install -e ".[dev]"
```

**Step 5: Verify installation**

```bash
python -c "import osmose; print(osmose.__version__)"
```
Expected: `0.1.0`

**Step 6: Commit**

```bash
git init
git add pyproject.toml osmose/ ui/ tests/ data/
git commit -m "feat: scaffold osmose-python project with dependencies"
```

---

### Task 2: Schema base classes and field types

**Files:**
- Create: `osmose-python/osmose/schema/__init__.py`
- Create: `osmose-python/osmose/schema/base.py`
- Create: `osmose-python/tests/test_schema.py`

**Step 1: Write the failing test**

```python
# tests/test_schema.py
from osmose.schema.base import OsmoseField, ParamType


def test_osmose_field_creation():
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        default=100.0,
        min_val=1.0,
        max_val=500.0,
        description="L-infinity (asymptotic length)",
        category="growth",
        unit="cm",
        indexed=True,
    )
    assert field.key_pattern == "species.linf.sp{idx}"
    assert field.param_type == ParamType.FLOAT
    assert field.default == 100.0
    assert field.indexed is True
    assert field.required is True  # default
    assert field.advanced is False  # default


def test_osmose_field_resolve_key():
    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        indexed=True,
    )
    assert field.resolve_key(3) == "species.linf.sp3"


def test_osmose_field_resolve_key_non_indexed():
    field = OsmoseField(
        key_pattern="simulation.time.ndtperyear",
        param_type=ParamType.INT,
        indexed=False,
    )
    assert field.resolve_key() == "simulation.time.ndtperyear"


def test_osmose_field_validate_in_range():
    field = OsmoseField(
        key_pattern="species.k.sp{idx}",
        param_type=ParamType.FLOAT,
        min_val=0.01,
        max_val=2.0,
    )
    assert field.validate_value(0.5) == []
    errors = field.validate_value(5.0)
    assert len(errors) == 1
    assert "max" in errors[0].lower()


def test_osmose_field_validate_enum():
    field = OsmoseField(
        key_pattern="grid.java.classname",
        param_type=ParamType.ENUM,
        choices=["fr.ird.osmose.grid.OriginalGrid", "fr.ird.osmose.grid.NcGrid"],
    )
    assert field.validate_value("fr.ird.osmose.grid.OriginalGrid") == []
    errors = field.validate_value("InvalidGrid")
    assert len(errors) == 1


def test_param_type_enum():
    assert ParamType.FLOAT.value == "float"
    assert ParamType.MATRIX.value == "matrix"
    assert ParamType.FILE_PATH.value == "file_path"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_schema.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'osmose.schema.base'`

**Step 3: Write implementation**

```python
# osmose/schema/__init__.py
from osmose.schema.base import OsmoseField, ParamType

__all__ = ["OsmoseField", "ParamType"]
```

```python
# osmose/schema/base.py
"""Schema base classes for OSMOSE parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParamType(Enum):
    """Types of OSMOSE configuration parameters."""
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    FILE_PATH = "file_path"
    MATRIX = "matrix"
    ENUM = "enum"


@dataclass
class OsmoseField:
    """Metadata for a single OSMOSE parameter.

    Attributes:
        key_pattern: OSMOSE property key, e.g. "species.linf.sp{idx}".
            Use {idx} placeholder for species-indexed parameters.
        param_type: The data type of this parameter.
        default: Default value if not specified.
        min_val: Minimum allowed value (numeric types only).
        max_val: Maximum allowed value (numeric types only).
        description: Human-readable description for UI tooltips.
        category: UI grouping category (e.g. "growth", "reproduction").
        unit: Physical unit (e.g. "cm", "year^-1").
        choices: Valid values for ENUM type.
        indexed: True if this parameter is per-species (uses sp{idx}).
        required: Whether this parameter must be specified.
        advanced: If True, shown only in the advanced config panel.
    """
    key_pattern: str
    param_type: ParamType
    default: Any = None
    min_val: float | None = None
    max_val: float | None = None
    description: str = ""
    category: str = ""
    unit: str = ""
    choices: list[str] | None = None
    indexed: bool = False
    required: bool = True
    advanced: bool = False

    def resolve_key(self, idx: int | None = None) -> str:
        """Resolve the key pattern to a concrete OSMOSE property key.

        Args:
            idx: Species index (required if self.indexed is True).

        Returns:
            Concrete key string, e.g. "species.linf.sp3".
        """
        if self.indexed:
            if idx is None:
                raise ValueError(f"Index required for indexed field: {self.key_pattern}")
            return self.key_pattern.replace("{idx}", str(idx))
        return self.key_pattern

    def validate_value(self, value: Any) -> list[str]:
        """Validate a value against this field's constraints.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []
        if self.param_type in (ParamType.FLOAT, ParamType.INT):
            if self.min_val is not None and value < self.min_val:
                errors.append(f"Value {value} below min {self.min_val}")
            if self.max_val is not None and value > self.max_val:
                errors.append(f"Value {value} above max {self.max_val}")
        if self.param_type == ParamType.ENUM and self.choices:
            if value not in self.choices:
                errors.append(f"Value '{value}' not in choices: {self.choices}")
        return errors
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_schema.py -v
```
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add osmose/schema/ tests/test_schema.py
git commit -m "feat: add schema base classes (OsmoseField, ParamType)"
```

---

### Task 3: Parameter registry

**Files:**
- Create: `osmose-python/osmose/schema/registry.py`
- Create: `osmose-python/tests/test_registry.py`

**Step 1: Write the failing test**

```python
# tests/test_registry.py
from osmose.schema.base import OsmoseField, ParamType
from osmose.schema.registry import ParameterRegistry


def test_registry_register_and_retrieve():
    reg = ParameterRegistry()
    f = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        category="growth",
        indexed=True,
    )
    reg.register(f)
    assert len(reg.all_fields()) == 1
    assert reg.all_fields()[0] is f


def test_registry_fields_by_category():
    reg = ParameterRegistry()
    f1 = OsmoseField(key_pattern="species.linf.sp{idx}", param_type=ParamType.FLOAT, category="growth", indexed=True)
    f2 = OsmoseField(key_pattern="species.k.sp{idx}", param_type=ParamType.FLOAT, category="growth", indexed=True)
    f3 = OsmoseField(key_pattern="simulation.time.ndtperyear", param_type=ParamType.INT, category="simulation")
    reg.register(f1)
    reg.register(f2)
    reg.register(f3)
    growth = reg.fields_by_category("growth")
    assert len(growth) == 2
    sim = reg.fields_by_category("simulation")
    assert len(sim) == 1


def test_registry_get_field_by_pattern():
    reg = ParameterRegistry()
    f = OsmoseField(key_pattern="species.linf.sp{idx}", param_type=ParamType.FLOAT, category="growth", indexed=True)
    reg.register(f)
    result = reg.get_field("species.linf.sp{idx}")
    assert result is f


def test_registry_get_field_not_found():
    reg = ParameterRegistry()
    assert reg.get_field("nonexistent") is None


def test_registry_categories():
    reg = ParameterRegistry()
    reg.register(OsmoseField(key_pattern="a", param_type=ParamType.FLOAT, category="growth"))
    reg.register(OsmoseField(key_pattern="b", param_type=ParamType.FLOAT, category="simulation"))
    reg.register(OsmoseField(key_pattern="c", param_type=ParamType.FLOAT, category="growth"))
    cats = reg.categories()
    assert set(cats) == {"growth", "simulation"}


def test_registry_validate_config():
    reg = ParameterRegistry()
    reg.register(OsmoseField(
        key_pattern="species.k.sp{idx}",
        param_type=ParamType.FLOAT,
        min_val=0.01,
        max_val=2.0,
        indexed=True,
        category="growth",
    ))
    # Valid config
    errors = reg.validate({"species.k.sp0": 0.5})
    assert errors == []
    # Invalid config
    errors = reg.validate({"species.k.sp0": 5.0})
    assert len(errors) == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_registry.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# osmose/schema/registry.py
"""Central registry for all OSMOSE parameters."""

from __future__ import annotations

import re
from osmose.schema.base import OsmoseField


class ParameterRegistry:
    """Collects all OSMOSE parameter definitions and provides lookup/validation."""

    def __init__(self):
        self._fields: list[OsmoseField] = []
        self._by_pattern: dict[str, OsmoseField] = {}

    def register(self, field: OsmoseField) -> None:
        self._fields.append(field)
        self._by_pattern[field.key_pattern] = field

    def all_fields(self) -> list[OsmoseField]:
        return list(self._fields)

    def fields_by_category(self, category: str) -> list[OsmoseField]:
        return [f for f in self._fields if f.category == category]

    def get_field(self, key_pattern: str) -> OsmoseField | None:
        return self._by_pattern.get(key_pattern)

    def categories(self) -> list[str]:
        seen = []
        for f in self._fields:
            if f.category not in seen:
                seen.append(f.category)
        return seen

    def match_field(self, concrete_key: str) -> OsmoseField | None:
        """Match a concrete key like 'species.k.sp0' to its field pattern."""
        for pattern, field in self._by_pattern.items():
            regex = re.escape(pattern).replace(r"\{idx\}", r"\d+")
            if re.fullmatch(regex, concrete_key):
                return field
        return None

    def validate(self, config: dict[str, object]) -> list[str]:
        """Validate a flat config dict against registered field constraints."""
        errors = []
        for key, value in config.items():
            field = self.match_field(key)
            if field:
                field_errors = field.validate_value(value)
                for e in field_errors:
                    errors.append(f"{key}: {e}")
        return errors
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_registry.py -v
```
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add osmose/schema/registry.py tests/test_registry.py
git commit -m "feat: add parameter registry with validation and category lookup"
```

---

### Task 4: Simulation parameter schema

**Files:**
- Create: `osmose-python/osmose/schema/simulation.py`
- Modify: `osmose-python/tests/test_schema.py` (append)

**Step 1: Write the failing test**

```python
# Append to tests/test_schema.py

from osmose.schema.simulation import SIMULATION_FIELDS


def test_simulation_fields_registered():
    assert len(SIMULATION_FIELDS) >= 10  # at least 10 simulation params


def test_simulation_ndtperyear_field():
    field = next(f for f in SIMULATION_FIELDS if "ndtperyear" in f.key_pattern)
    assert field.param_type == ParamType.INT
    assert field.default == 24
    assert field.category == "simulation"
    assert not field.indexed


def test_simulation_nspecies_field():
    field = next(f for f in SIMULATION_FIELDS if "nspecies" in f.key_pattern)
    assert field.param_type == ParamType.INT
    assert field.min_val >= 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_schema.py::test_simulation_fields_registered -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# osmose/schema/simulation.py
"""Simulation-level OSMOSE parameters."""

from osmose.schema.base import OsmoseField, ParamType

SIMULATION_FIELDS: list[OsmoseField] = [
    OsmoseField(
        key_pattern="simulation.time.ndtperyear",
        param_type=ParamType.INT,
        default=24,
        min_val=1,
        max_val=365,
        description="Number of time steps per year",
        category="simulation",
        unit="steps/year",
    ),
    OsmoseField(
        key_pattern="simulation.time.nyear",
        param_type=ParamType.INT,
        default=100,
        min_val=1,
        max_val=1000,
        description="Total number of simulation years",
        category="simulation",
        unit="years",
    ),
    OsmoseField(
        key_pattern="simulation.nspecies",
        param_type=ParamType.INT,
        default=3,
        min_val=1,
        max_val=50,
        description="Number of focal species",
        category="simulation",
    ),
    OsmoseField(
        key_pattern="simulation.nresource",
        param_type=ParamType.INT,
        default=0,
        min_val=0,
        max_val=50,
        description="Number of resource (plankton) groups",
        category="simulation",
    ),
    OsmoseField(
        key_pattern="simulation.nbackground",
        param_type=ParamType.INT,
        default=0,
        min_val=0,
        max_val=50,
        description="Number of background species",
        category="simulation",
    ),
    OsmoseField(
        key_pattern="simulation.nschool",
        param_type=ParamType.INT,
        default=20,
        min_val=1,
        max_val=200,
        description="Default number of schools per species",
        category="simulation",
    ),
    OsmoseField(
        key_pattern="simulation.ncpu",
        param_type=ParamType.INT,
        default=1,
        min_val=1,
        max_val=128,
        description="Number of CPUs for parallel runs",
        category="simulation",
    ),
    OsmoseField(
        key_pattern="simulation.nsimulation",
        param_type=ParamType.INT,
        default=1,
        min_val=1,
        max_val=1000,
        description="Number of replicate simulations",
        category="simulation",
    ),
    OsmoseField(
        key_pattern="simulation.restart.file",
        param_type=ParamType.FILE_PATH,
        default="null",
        description="Path to restart file (null = no restart)",
        category="simulation",
        required=False,
    ),
    OsmoseField(
        key_pattern="simulation.bioen.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description="Enable bioenergetics module",
        category="simulation",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="simulation.genetic.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description="Enable genetic/evolutionary module",
        category="simulation",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="simulation.incoming.flux.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description="Enable immigration fluxes",
        category="simulation",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="simulation.fishing.mortality.enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable fishing mortality",
        category="simulation",
    ),
    OsmoseField(
        key_pattern="simulation.nfisheries",
        param_type=ParamType.INT,
        default=0,
        min_val=0,
        max_val=50,
        description="Number of fisheries (v4 fisheries module)",
        category="simulation",
    ),
    OsmoseField(
        key_pattern="mortality.subdt",
        param_type=ParamType.INT,
        default=10,
        min_val=1,
        max_val=100,
        description="Sub-timesteps for mortality algorithm",
        category="simulation",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="stochastic.mortality.randomseed.fixed",
        param_type=ParamType.BOOL,
        default=False,
        description="Fix random seed for mortality (reproducibility)",
        category="simulation",
        advanced=True,
    ),
]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_schema.py -v
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add osmose/schema/simulation.py tests/test_schema.py
git commit -m "feat: add simulation parameter schema (16 params)"
```

---

### Task 5: Species parameter schema

**Files:**
- Create: `osmose-python/osmose/schema/species.py`
- Create: `osmose-python/tests/test_schema_species.py`

**Step 1: Write the failing test**

```python
# tests/test_schema_species.py
from osmose.schema.species import SPECIES_FIELDS
from osmose.schema.base import ParamType


def test_species_fields_count():
    # Should have 30+ parameters per species
    assert len(SPECIES_FIELDS) >= 25


def test_all_species_fields_indexed():
    for f in SPECIES_FIELDS:
        assert f.indexed, f"Species field {f.key_pattern} should be indexed"
        assert "{idx}" in f.key_pattern


def test_species_growth_params_present():
    patterns = [f.key_pattern for f in SPECIES_FIELDS]
    assert "species.linf.sp{idx}" in patterns
    assert "species.k.sp{idx}" in patterns
    assert "species.t0.sp{idx}" in patterns


def test_species_reproduction_params_present():
    patterns = [f.key_pattern for f in SPECIES_FIELDS]
    assert "species.maturity.size.sp{idx}" in patterns
    assert "species.relativefecundity.sp{idx}" in patterns
    assert "species.sexratio.sp{idx}" in patterns


def test_species_linf_has_correct_metadata():
    linf = next(f for f in SPECIES_FIELDS if f.key_pattern == "species.linf.sp{idx}")
    assert linf.param_type == ParamType.FLOAT
    assert linf.unit == "cm"
    assert linf.min_val is not None
    assert linf.category == "growth"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_schema_species.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# osmose/schema/species.py
"""Focal species OSMOSE parameters."""

from osmose.schema.base import OsmoseField, ParamType

SPECIES_FIELDS: list[OsmoseField] = [
    # ── Identity ──
    OsmoseField(
        key_pattern="species.name.sp{idx}", param_type=ParamType.STRING,
        description="Species name", category="identity", indexed=True,
    ),
    OsmoseField(
        key_pattern="species.type.sp{idx}", param_type=ParamType.ENUM,
        default="focal", choices=["focal", "resource", "background"],
        description="Species type", category="identity", indexed=True,
    ),

    # ── Von Bertalanffy Growth ──
    OsmoseField(
        key_pattern="species.linf.sp{idx}", param_type=ParamType.FLOAT,
        default=100.0, min_val=1.0, max_val=1000.0,
        description="L-infinity (asymptotic length)", category="growth",
        unit="cm", indexed=True,
    ),
    OsmoseField(
        key_pattern="species.k.sp{idx}", param_type=ParamType.FLOAT,
        default=0.2, min_val=0.01, max_val=3.0,
        description="Von Bertalanffy growth coefficient K", category="growth",
        unit="year^-1", indexed=True,
    ),
    OsmoseField(
        key_pattern="species.t0.sp{idx}", param_type=ParamType.FLOAT,
        default=-0.5, min_val=-10.0, max_val=0.0,
        description="Theoretical age at length 0", category="growth",
        unit="year", indexed=True,
    ),
    OsmoseField(
        key_pattern="species.vonbertalanffy.threshold.age.sp{idx}", param_type=ParamType.FLOAT,
        default=1.0, min_val=0.0, max_val=10.0,
        description="Age threshold for switching to VB growth", category="growth",
        unit="year", indexed=True, advanced=True,
    ),
    OsmoseField(
        key_pattern="species.lmax.sp{idx}", param_type=ParamType.FLOAT,
        default=None, min_val=1.0, max_val=2000.0,
        description="Maximum length cap", category="growth",
        unit="cm", indexed=True, required=False,
    ),
    OsmoseField(
        key_pattern="growth.java.classname.sp{idx}", param_type=ParamType.ENUM,
        default="VonBertalanffyGrowth",
        choices=["VonBertalanffyGrowth", "GompertzGrowth"],
        description="Growth model class", category="growth", indexed=True, advanced=True,
    ),

    # ── Length-Weight Relationship ──
    OsmoseField(
        key_pattern="species.length2weight.condition.factor.sp{idx}", param_type=ParamType.FLOAT,
        default=0.005, min_val=0.0001, max_val=1.0,
        description="Length-weight condition factor (a)", category="growth",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="species.length2weight.allometric.power.sp{idx}", param_type=ParamType.FLOAT,
        default=3.0, min_val=2.0, max_val=4.0,
        description="Length-weight allometric power (b)", category="growth",
        indexed=True,
    ),

    # ── Reproduction ──
    OsmoseField(
        key_pattern="species.egg.size.sp{idx}", param_type=ParamType.FLOAT,
        default=0.1, min_val=0.01, max_val=5.0,
        description="Egg diameter", category="reproduction",
        unit="cm", indexed=True,
    ),
    OsmoseField(
        key_pattern="species.egg.weight.sp{idx}", param_type=ParamType.FLOAT,
        default=0.001, min_val=0.00001, max_val=1.0,
        description="Egg weight", category="reproduction",
        unit="g", indexed=True,
    ),
    OsmoseField(
        key_pattern="species.maturity.size.sp{idx}", param_type=ParamType.FLOAT,
        default=30.0, min_val=0.1, max_val=500.0,
        description="Size at maturity", category="reproduction",
        unit="cm", indexed=True,
    ),
    OsmoseField(
        key_pattern="species.maturity.age.sp{idx}", param_type=ParamType.FLOAT,
        default=2.0, min_val=0.0, max_val=50.0,
        description="Age at maturity", category="reproduction",
        unit="year", indexed=True, required=False,
    ),
    OsmoseField(
        key_pattern="species.relativefecundity.sp{idx}", param_type=ParamType.FLOAT,
        default=50.0, min_val=0.1, max_val=10000.0,
        description="Eggs per gram of mature female per year", category="reproduction",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="species.sexratio.sp{idx}", param_type=ParamType.FLOAT,
        default=0.5, min_val=0.0, max_val=1.0,
        description="Female proportion (sex ratio)", category="reproduction",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="reproduction.season.file.sp{idx}", param_type=ParamType.FILE_PATH,
        description="Spawning seasonality CSV file", category="reproduction",
        indexed=True, required=False,
    ),

    # ── Life History ──
    OsmoseField(
        key_pattern="species.lifespan.sp{idx}", param_type=ParamType.FLOAT,
        default=10.0, min_val=0.5, max_val=200.0,
        description="Maximum age (lifespan)", category="life_history",
        unit="year", indexed=True,
    ),
    OsmoseField(
        key_pattern="species.first.feeding.age.sp{idx}", param_type=ParamType.FLOAT,
        default=1.0, min_val=0.0, max_val=24.0,
        description="Age at first feeding (in timesteps)", category="life_history",
        indexed=True, advanced=True,
    ),

    # ── Predation (per-species) ──
    OsmoseField(
        key_pattern="predation.ingestion.rate.max.sp{idx}", param_type=ParamType.FLOAT,
        default=3.5, min_val=0.1, max_val=20.0,
        description="Maximum ingestion rate", category="predation",
        unit="g/g/year", indexed=True,
    ),
    OsmoseField(
        key_pattern="predation.efficiency.critical.sp{idx}", param_type=ParamType.FLOAT,
        default=0.57, min_val=0.0, max_val=1.0,
        description="Critical predation efficiency (triggers starvation)", category="predation",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="predation.predprey.sizeratio.max.sp{idx}", param_type=ParamType.FLOAT,
        default=5.0, min_val=1.0, max_val=100.0,
        description="Maximum predator/prey size ratio", category="predation",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="predation.predprey.sizeratio.min.sp{idx}", param_type=ParamType.FLOAT,
        default=9.0, min_val=1.0, max_val=100.0,
        description="Minimum predator/prey size ratio", category="predation",
        indexed=True,
    ),

    # ── Natural Mortality ──
    OsmoseField(
        key_pattern="mortality.natural.rate.sp{idx}", param_type=ParamType.FLOAT,
        default=0.2, min_val=0.0, max_val=10.0,
        description="Annual natural mortality rate (adults)", category="mortality",
        unit="year^-1", indexed=True,
    ),
    OsmoseField(
        key_pattern="mortality.natural.larva.rate.sp{idx}", param_type=ParamType.FLOAT,
        default=5.0, min_val=0.0, max_val=100.0,
        description="Larval natural mortality rate", category="mortality",
        unit="year^-1", indexed=True,
    ),
    OsmoseField(
        key_pattern="mortality.starvation.rate.max.sp{idx}", param_type=ParamType.FLOAT,
        default=0.3, min_val=0.0, max_val=5.0,
        description="Maximum starvation mortality rate", category="mortality",
        unit="year^-1", indexed=True,
    ),

    # ── Fishing (per-species legacy) ──
    OsmoseField(
        key_pattern="mortality.fishing.rate.sp{idx}", param_type=ParamType.FLOAT,
        default=0.0, min_val=0.0, max_val=5.0,
        description="Annual fishing mortality rate", category="fishing",
        unit="year^-1", indexed=True,
    ),
    OsmoseField(
        key_pattern="mortality.fishing.recruitment.age.sp{idx}", param_type=ParamType.FLOAT,
        default=0.5, min_val=0.0, max_val=50.0,
        description="Age at recruitment to fishery", category="fishing",
        unit="year", indexed=True, required=False,
    ),
    OsmoseField(
        key_pattern="mortality.fishing.recruitment.size.sp{idx}", param_type=ParamType.FLOAT,
        default=None, min_val=0.0, max_val=500.0,
        description="Size at recruitment to fishery", category="fishing",
        unit="cm", indexed=True, required=False,
    ),

    # ── Population Initialization ──
    OsmoseField(
        key_pattern="population.seeding.biomass.sp{idx}", param_type=ParamType.FLOAT,
        default=1000.0, min_val=0.0, max_val=1e9,
        description="Initial seeding biomass", category="initialization",
        unit="tons", indexed=True,
    ),
]
```

**Step 4: Run tests**

```bash
pytest tests/test_schema_species.py -v
```
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add osmose/schema/species.py tests/test_schema_species.py
git commit -m "feat: add species parameter schema (30 params covering growth, repro, mortality, predation)"
```

---

### Task 6: Grid, LTL, output, and remaining schema modules

**Files:**
- Create: `osmose-python/osmose/schema/grid.py`
- Create: `osmose-python/osmose/schema/predation.py`
- Create: `osmose-python/osmose/schema/fishing.py`
- Create: `osmose-python/osmose/schema/movement.py`
- Create: `osmose-python/osmose/schema/ltl.py`
- Create: `osmose-python/osmose/schema/output.py`
- Create: `osmose-python/osmose/schema/bioenergetics.py`
- Create: `osmose-python/osmose/schema/economics.py`
- Create: `osmose-python/tests/test_schema_all.py`

This task is large — implement each schema module following the same pattern as simulation.py and species.py. Each module exports a `*_FIELDS` list. The test verifies all modules register their fields and that a global registry can be built.

**Step 1: Write the failing test**

```python
# tests/test_schema_all.py
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
    # Should have at least 150 parameters across all modules
    assert total >= 150, f"Only {total} params registered, expected >= 150"


def test_grid_fields_present():
    assert len(GRID_FIELDS) >= 8


def test_output_fields_present():
    # Output has 115+ boolean flags
    assert len(OUTPUT_FIELDS) >= 50


def test_ltl_fields_present():
    assert len(LTL_FIELDS) >= 5
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_schema_all.py -v
```
Expected: FAIL — missing modules

**Step 3: Implement all remaining schema modules**

Each module follows the same pattern — a list of `OsmoseField` objects. The key references for parameter names are in the design doc section "OSMOSE Configuration Reference" and the detailed research output.

- `grid.py`: ~10 fields (grid.ncolumn, grid.nline, grid.upleft.lat, grid.upleft.lon, grid.lowright.lat, grid.lowright.lon, grid.java.classname, grid.netcdf.file, etc.)
- `predation.py`: ~10 fields (accessibility matrix file, stage structure, etc.) — global predation params (per-species predation is in species.py)
- `fishing.py`: ~20 fields (fisheries module: fisheries.name.fsh{idx}, selectivity, seasonality, catchability, MPAs)
- `movement.py`: ~10 fields (movement.distribution.method, map file patterns)
- `ltl.py`: ~10 fields per resource (species.name.sp{idx} for resources, size, TL, biomass file, accessibility)
- `output.py`: 115+ boolean flags + output settings (output.dir.path, output.biomass.enabled, etc.)
- `bioenergetics.py`: ~15 fields (temperature forcing, species bioen params)
- `economics.py`: ~5 fields (economy.enabled, economic output stages)

For output.py, generate the boolean flags programmatically:

```python
# osmose/schema/output.py (excerpt showing pattern)
OUTPUT_FIELDS: list[OsmoseField] = [
    OsmoseField(
        key_pattern="output.dir.path", param_type=ParamType.STRING,
        default="output", description="Output directory path", category="output",
    ),
    OsmoseField(
        key_pattern="output.file.prefix", param_type=ParamType.STRING,
        default="osm", description="Output file prefix", category="output",
    ),
    OsmoseField(
        key_pattern="output.start.year", param_type=ParamType.INT,
        default=0, min_val=0, description="First year to write output", category="output",
    ),
    OsmoseField(
        key_pattern="output.recordfrequency.ndt", param_type=ParamType.INT,
        default=12, min_val=1, description="Recording frequency (timesteps)", category="output",
    ),
    # ... then all 115+ boolean enable flags:
]

# Programmatically generate output enable flags
_OUTPUT_ENABLE_FLAGS = [
    "output.biomass.enabled", "output.abundance.enabled",
    "output.abundance.age1.enabled", "output.ssb.enabled",
    "output.biomass.bysize.enabled", "output.biomass.byage.enabled",
    # ... (full list from design doc research)
]
for flag in _OUTPUT_ENABLE_FLAGS:
    OUTPUT_FIELDS.append(OsmoseField(
        key_pattern=flag, param_type=ParamType.BOOL,
        default=False, description=flag.replace("output.", "").replace(".", " ").replace("enabled", "").strip(),
        category="output", advanced=True,
    ))
```

**Step 4: Run tests**

```bash
pytest tests/test_schema_all.py -v
```
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add osmose/schema/ tests/test_schema_all.py
git commit -m "feat: add all schema modules (grid, predation, fishing, movement, LTL, output, bioen, economics)"
```

---

## Phase 2: Config I/O

### Task 7: Config reader — parse OSMOSE .properties files

**Files:**
- Create: `osmose-python/osmose/config/__init__.py`
- Create: `osmose-python/osmose/config/reader.py`
- Create: `osmose-python/tests/test_config_reader.py`
- Create: `osmose-python/tests/fixtures/` (test config files)

**Step 1: Create test fixtures**

Create minimal OSMOSE config files for testing:

```
# tests/fixtures/osm_all-parameters.csv
simulation.time.ndtperyear ; 12
simulation.time.nyear ; 50
simulation.nspecies ; 2
osmose.configuration.species ; osm_param-species.csv
```

```
# tests/fixtures/osm_param-species.csv
species.name.sp0 ; Anchovy
species.linf.sp0 ; 19.5
species.k.sp0 ; 0.364
species.name.sp1 ; Sardine
species.linf.sp1 ; 23.0
species.k.sp1 ; 0.28
```

**Step 2: Write the failing test**

```python
# tests/test_config_reader.py
from pathlib import Path
from osmose.config.reader import OsmoseConfigReader

FIXTURES = Path(__file__).parent / "fixtures"


def test_read_single_file():
    reader = OsmoseConfigReader()
    result = reader.read_file(FIXTURES / "osm_all-parameters.csv")
    assert result["simulation.time.ndtperyear"] == "12"
    assert result["simulation.time.nyear"] == "50"


def test_read_recursive():
    reader = OsmoseConfigReader()
    result = reader.read(FIXTURES / "osm_all-parameters.csv")
    # Should include params from both master and species sub-file
    assert result["simulation.nspecies"] == "2"
    assert result["species.name.sp0"] == "Anchovy"
    assert result["species.linf.sp0"] == "19.5"
    assert result["species.name.sp1"] == "Sardine"


def test_keys_are_lowercase():
    reader = OsmoseConfigReader()
    result = reader.read_file(FIXTURES / "osm_all-parameters.csv")
    for key in result:
        assert key == key.lower()


def test_auto_detect_separator_equals():
    reader = OsmoseConfigReader()
    # Create a temp file with = separator
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("key1 = value1\nkey2 = value2\n")
        path = Path(f.name)
    result = reader.read_file(path)
    assert result["key1"] == "value1"
    path.unlink()


def test_skip_comments():
    reader = OsmoseConfigReader()
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("# This is a comment\nkey1 ; value1\n")
        path = Path(f.name)
    result = reader.read_file(path)
    assert "# this is a comment" not in result
    assert result["key1"] == "value1"
    path.unlink()
```

**Step 3: Run tests to verify they fail, then implement**

```python
# osmose/config/reader.py
"""Parse OSMOSE .properties/.csv configuration files."""

from __future__ import annotations

import re
from pathlib import Path


class OsmoseConfigReader:
    """Read OSMOSE configuration files with recursive sub-file loading."""

    SEPARATORS = re.compile(r"\s*[=;,:\t]\s*")
    COMMENT_CHARS = {"#", "!"}

    def read(self, master_file: Path) -> dict[str, str]:
        """Recursively read a master config and all referenced sub-configs."""
        flat = {}
        self._read_recursive(master_file, flat)
        return flat

    def _read_recursive(self, filepath: Path, flat: dict[str, str]) -> None:
        file_params = self.read_file(filepath)
        flat.update(file_params)
        # Follow osmose.configuration.* references
        for key, value in file_params.items():
            if key.startswith("osmose.configuration."):
                sub_path = filepath.parent / value.strip()
                if sub_path.exists():
                    self._read_recursive(sub_path, flat)

    def read_file(self, filepath: Path) -> dict[str, str]:
        """Parse a single OSMOSE config file."""
        result = {}
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] in self.COMMENT_CHARS:
                    continue
                parts = self.SEPARATORS.split(line, maxsplit=1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    result[key] = value
        return result
```

**Step 4: Run tests**

```bash
pytest tests/test_config_reader.py -v
```
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add osmose/config/ tests/test_config_reader.py tests/fixtures/
git commit -m "feat: add config reader with recursive sub-file loading and auto separator detection"
```

---

### Task 8: Config writer — generate OSMOSE config files from schema

**Files:**
- Create: `osmose-python/osmose/config/writer.py`
- Create: `osmose-python/tests/test_config_writer.py`

**Step 1: Write the failing test**

```python
# tests/test_config_writer.py
import tempfile
from pathlib import Path
from osmose.config.writer import OsmoseConfigWriter
from osmose.config.reader import OsmoseConfigReader


def test_write_master_file():
    config = {
        "simulation.time.ndtperyear": 12,
        "simulation.time.nyear": 50,
        "simulation.nspecies": 2,
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": 19.5,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))
        master = Path(tmpdir) / "osm_all-parameters.csv"
        assert master.exists()


def test_roundtrip_read_write():
    config = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "50",
        "simulation.nspecies": "2",
        "species.name.sp0": "Anchovy",
        "species.linf.sp0": "19.5",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))
        reader = OsmoseConfigReader()
        result = reader.read(Path(tmpdir) / "osm_all-parameters.csv")
        for key, value in config.items():
            assert result[key] == str(value), f"Mismatch for {key}: {result.get(key)} != {value}"


def test_write_splits_by_category():
    config = {
        "simulation.time.ndtperyear": "12",
        "species.name.sp0": "Anchovy",
        "grid.ncolumn": "30",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OsmoseConfigWriter()
        writer.write(config, Path(tmpdir))
        # Should create sub-files referenced from master
        master_content = (Path(tmpdir) / "osm_all-parameters.csv").read_text()
        assert "osmose.configuration" in master_content
```

**Step 2: Run tests, implement, verify, commit**

Implementation writes a master file + category sub-files. Each sub-file gets parameters that match its category prefix. The master file includes `osmose.configuration.*` references to all sub-files.

```bash
git commit -m "feat: add config writer with category-based file splitting"
```

---

### Task 9: Config roundtrip test

**Files:**
- Create: `osmose-python/tests/test_roundtrip.py`

A comprehensive test that creates a full config (all categories), writes it, reads it back, and verifies equality. This is the key regression test for the entire config system.

```bash
git commit -m "test: add full config roundtrip test"
```

---

## Phase 3: Runner & Results

### Task 10: Java engine runner (async subprocess)

**Files:**
- Create: `osmose-python/osmose/runner.py`
- Create: `osmose-python/tests/test_runner.py`

Implements `OsmoseRunner` with:
- `run()` — async subprocess execution of `java -jar osmose.jar config.csv`
- `cancel()` — terminate running process
- `get_java_version()` — check Java is installed
- Progress streaming via callback

Test with a mock Java process (since we don't have the actual JAR in tests).

```bash
git commit -m "feat: add async OSMOSE Java runner with progress streaming"
```

---

### Task 11: NetCDF results reader

**Files:**
- Create: `osmose-python/osmose/results.py`
- Create: `osmose-python/tests/test_results.py`
- Create: `osmose-python/tests/fixtures/sample_output.nc` (synthetic test NetCDF)

Implements `OsmoseResults` with methods for biomass, abundance, diet matrix, spatial maps, etc. Tests use synthetic NetCDF files created with xarray.

```bash
git commit -m "feat: add NetCDF results reader with biomass, diet, spatial output parsing"
```

---

## Phase 4: Scenario Management

### Task 12: Scenario save/load/compare/fork

**Files:**
- Create: `osmose-python/osmose/scenarios.py`
- Create: `osmose-python/tests/test_scenarios.py`

Implements `Scenario` dataclass and `ScenarioManager` with save, load, list, compare (diff), fork operations. Storage is JSON + file copying.

```bash
git commit -m "feat: add scenario management (save/load/compare/fork)"
```

---

## Phase 5: Shiny UI

### Task 13: App shell with shinyswatch theme

**Files:**
- Create: `osmose-python/app.py`
- Create: `osmose-python/ui/theme.py`

**Step 1: Write app shell**

```python
# app.py
from shiny import App, ui
import shinyswatch

from ui.pages.setup import setup_ui, setup_server
from ui.pages.run import run_ui, run_server
from ui.pages.results import results_ui, results_server

app_ui = ui.page_navbar(
    ui.nav_panel("Setup", setup_ui()),
    ui.nav_panel("Grid & Maps", ui.div("Grid configuration - TODO")),
    ui.nav_panel("Forcing", ui.div("Forcing configuration - TODO")),
    ui.nav_panel("Fishing", ui.div("Fishing configuration - TODO")),
    ui.nav_panel("Run", run_ui()),
    ui.nav_panel("Results", results_ui()),
    ui.nav_panel("Calibration", ui.div("Calibration - TODO")),
    ui.nav_panel("Scenarios", ui.div("Scenario management - TODO")),
    ui.nav_panel("Advanced", ui.div("Advanced config editor - TODO")),
    title="OSMOSE | Python Interface",
    theme=shinyswatch.theme.superhero,
)

def server(input, output, session):
    setup_server(input, output, session)
    run_server(input, output, session)
    results_server(input, output, session)

app = App(app_ui, server)
```

**Step 2: Verify app starts**

```bash
cd osmose-python && shiny run app.py --port 8000
```
Expected: App loads in browser with superhero theme and 9 tabs

**Step 3: Commit**

```bash
git commit -m "feat: add Shiny app shell with shinyswatch superhero theme and 9-tab navbar"
```

---

### Task 14: Auto-generated parameter form component

**Files:**
- Create: `osmose-python/ui/components/param_form.py`
- Create: `osmose-python/tests/test_param_form.py`

Implements `render_field()` that generates a Shiny input widget from an `OsmoseField`:
- FLOAT/INT → `ui.input_numeric()`
- BOOL → `ui.input_switch()`
- STRING → `ui.input_text()`
- ENUM → `ui.input_select()`
- FILE_PATH → `ui.input_file()`

And `render_category()` that generates a full form section for a category.

```bash
git commit -m "feat: add auto-generated parameter form component from schema"
```

---

### Task 15: Species setup page

**Files:**
- Create: `osmose-python/ui/pages/setup.py`
- Create: `osmose-python/ui/components/species_table.py`

Implements the Setup tab:
- Species count selector
- Species name inputs
- Per-species parameter forms (auto-generated from schema)
- Editable data table for bulk parameter editing

```bash
git commit -m "feat: add species setup page with auto-generated parameter forms"
```

---

### Task 16: Grid configuration page

**Files:**
- Create: `osmose-python/ui/pages/grid.py`
- Create: `osmose-python/ui/components/map_viewer.py`

Grid tab with:
- Grid type selector (regular vs NetCDF)
- Dimension/coordinate inputs
- Grid mask CSV upload
- Plotly map preview showing grid extent and mask

```bash
git commit -m "feat: add grid configuration page with spatial preview"
```

---

### Task 17: Forcing (LTL) page

**Files:**
- Create: `osmose-python/ui/pages/forcing.py`
- Create: `osmose-python/ui/components/file_upload.py`

Forcing tab with:
- Resource/plankton group configuration
- NetCDF file upload for biomass forcing
- Temperature/oxygen forcing upload
- Preview of uploaded forcing data (time series plot)

```bash
git commit -m "feat: add forcing/LTL configuration page with NetCDF upload"
```

---

### Task 18: Fishing configuration page

**Files:**
- Create: `osmose-python/ui/pages/fishing.py`

Fishing tab with:
- Legacy fishing mortality per species
- Fisheries module configuration
- Selectivity curve editor (sigmoid/knife-edge/Gaussian)
- MPA definition interface

```bash
git commit -m "feat: add fishing configuration page with fisheries module support"
```

---

### Task 19: Movement maps page

**Files:**
- Create: `osmose-python/ui/pages/movement.py`

Movement tab with:
- Species distribution maps (CSV upload or draw on map)
- Age/season/year range selectors per map
- Map preview with plotly heatmap

```bash
git commit -m "feat: add movement/spatial distribution page"
```

---

### Task 20: Run control page

**Files:**
- Create: `osmose-python/ui/pages/run.py`

Run tab with:
- Start/cancel buttons
- Java version check indicator
- Progress console (streaming stdout/stderr)
- Run history list
- Config validation before run (highlight errors)

```bash
git commit -m "feat: add run control page with progress streaming and validation"
```

---

### Task 21: Results visualization page

**Files:**
- Create: `osmose-python/ui/pages/results.py`

Results tab with:
- Biomass time series (plotly line chart, species selector)
- Diet composition matrix (plotly heatmap)
- Size distribution histograms
- Spatial biomass maps (plotly heatmap)
- Mortality breakdown (stacked bar chart)
- Export to CSV/PNG buttons

```bash
git commit -m "feat: add results visualization page with plotly charts"
```

---

### Task 22: Scenarios page

**Files:**
- Create: `osmose-python/ui/pages/scenarios.py`

Scenarios tab with:
- Save current config as named scenario
- Load scenario (replaces current config)
- Scenario list with metadata
- Compare two scenarios (diff table)
- Fork from existing scenario

```bash
git commit -m "feat: add scenario management page"
```

---

### Task 23: Advanced raw config editor

**Files:**
- Create: `osmose-python/ui/pages/advanced.py`

Advanced tab with:
- Searchable/filterable table of ALL parameters
- Inline editing for any parameter value
- Import existing OSMOSE config (upload .csv)
- Export current config (download .csv)

```bash
git commit -m "feat: add advanced raw config editor with search and import/export"
```

---

## Phase 6: Calibration

### Task 24: Objective functions

**Files:**
- Create: `osmose-python/osmose/calibration/__init__.py`
- Create: `osmose-python/osmose/calibration/objectives.py`
- Create: `osmose-python/tests/test_objectives.py`

Implements:
- `biomass_rmse(simulated, observed)` — root mean square error of biomass time series
- `diet_distance(simulated_matrix, observed_matrix)` — Frobenius norm of diet matrix difference
- `abundance_rmse(simulated, observed)` — RMSE for abundance

```bash
git commit -m "feat: add calibration objective functions (biomass RMSE, diet distance)"
```

---

### Task 25: OSMOSE as pymoo optimization problem

**Files:**
- Create: `osmose-python/osmose/calibration/problem.py`
- Create: `osmose-python/tests/test_calibration_problem.py`

Implements `OsmoseCalibrationProblem(pymoo.Problem)`:
- Takes base config, free parameters with bounds, objectives
- `_evaluate()` runs OSMOSE in parallel (isolated temp dirs per candidate)
- Returns objective values for pymoo's optimizer

```bash
git commit -m "feat: add OSMOSE calibration problem for pymoo multi-objective optimization"
```

---

### Task 26: GP surrogate model

**Files:**
- Create: `osmose-python/osmose/calibration/surrogate.py`
- Create: `osmose-python/tests/test_surrogate.py`

Implements `SurrogateCalibrator`:
- Latin hypercube sampling for training data generation
- Gaussian Process fitting (scikit-learn)
- Optimization on surrogate
- Validation against real OSMOSE runs

```bash
git commit -m "feat: add GP surrogate model for fast OSMOSE calibration"
```

---

### Task 27: Sensitivity analysis

**Files:**
- Create: `osmose-python/osmose/calibration/sensitivity.py`
- Create: `osmose-python/tests/test_sensitivity.py`

Implements SALib-based Sobol sensitivity analysis:
- Generate parameter samples
- Run OSMOSE for each sample
- Compute first-order and total-order sensitivity indices

```bash
git commit -m "feat: add Sobol sensitivity analysis using SALib"
```

---

### Task 28: Calibration UI page

**Files:**
- Create: `osmose-python/ui/pages/calibration.py`

Calibration tab with:
- Free parameter selector (checkbox list from schema)
- Bounds editor per free parameter
- Objective function selector (upload observed data)
- Algorithm choice (NSGA-II direct / GP surrogate)
- Population size, generations, parallel workers controls
- Run calibration button with progress
- Pareto front viewer (plotly scatter)
- Best candidate table with parameter values

```bash
git commit -m "feat: add calibration UI page with Pareto front viewer"
```

---

## Phase 7: Deployment & Polish

### Task 29: Dockerfile

**Files:**
- Create: `osmose-python/Dockerfile`
- Create: `osmose-python/.dockerignore`

Multi-stage build: Java JRE + Python 3.12. Expose port 8000.

```bash
git commit -m "feat: add Dockerfile for containerized deployment"
```

---

### Task 30: Integration test with example config

**Files:**
- Create: `osmose-python/tests/test_integration.py`
- Create: `osmose-python/data/examples/` (example OSMOSE configs from public repos)

End-to-end test: load example config → verify schema mapping → write config → compare with original. Does NOT require Java (tests config I/O only).

```bash
git commit -m "test: add integration test with public OSMOSE example configuration"
```

---

### Task 31: Wire all UI pages into app.py

**Files:**
- Modify: `osmose-python/app.py`

Replace all TODO placeholders with actual page imports. Verify all 9 tabs work end-to-end.

```bash
git commit -m "feat: wire all UI pages into app shell"
```

---

## Task Dependency Graph

```
Task 1 (scaffold)
  └─> Task 2 (schema base)
       └─> Task 3 (registry)
            ├─> Task 4 (simulation schema)
            ├─> Task 5 (species schema)
            └─> Task 6 (remaining schemas)
                 └─> Task 7 (config reader)
                      └─> Task 8 (config writer)
                           └─> Task 9 (roundtrip test)
                                ├─> Task 10 (runner)
                                │    └─> Task 20 (run page)
                                ├─> Task 11 (results reader)
                                │    └─> Task 21 (results page)
                                ├─> Task 12 (scenarios)
                                │    └─> Task 22 (scenarios page)
                                └─> Task 13 (app shell)
                                     ├─> Task 14 (param form component)
                                     │    ├─> Task 15 (setup page)
                                     │    ├─> Task 16 (grid page)
                                     │    ├─> Task 17 (forcing page)
                                     │    ├─> Task 18 (fishing page)
                                     │    ├─> Task 19 (movement page)
                                     │    └─> Task 23 (advanced page)
                                     └─> Tasks 24-28 (calibration)

Task 29 (Docker) — independent
Task 30 (integration test) — depends on Tasks 7-9
Task 31 (wire pages) — depends on all UI tasks
```

## Summary

- **31 tasks** across 7 phases
- **Phase 1** (Tasks 1-6): Foundation — project setup, schema system
- **Phase 2** (Tasks 7-9): Config I/O — read/write OSMOSE files
- **Phase 3** (Tasks 10-11): Engine — run Java, read results
- **Phase 4** (Task 12): Scenarios — save/load/compare
- **Phase 5** (Tasks 13-23): UI — all 9 Shiny pages
- **Phase 6** (Tasks 24-28): Calibration — objectives, pymoo, GP, sensitivity, UI
- **Phase 7** (Tasks 29-31): Polish — Docker, integration tests, final wiring
