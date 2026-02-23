# Phase 3: Complete Phase 2 Gaps — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire remaining input syncing for 5 config pages, calibration Start/Stop/Sensitivity handlers, Advanced import/export, and fix movement page species count.

**Architecture:** Add `sync_inputs()` utility to `ui/state.py` that auto-syncs non-indexed fields. Each page server adds explicit `@reactive.effect` handlers for indexed fields. Calibration uses `threading.Thread` to run pymoo in background. Advanced import/export uses `OsmoseConfigReader` and `OsmoseConfigWriter`.

**Tech Stack:** Shiny for Python, pymoo, SALib, plotly, shinywidgets

---

### Task 1: Add `jar_path` and `sync_inputs` to AppState

**Files:**
- Modify: `ui/state.py`
- Test: `tests/test_state.py`

**Step 1: Write the failing tests**

Append to `tests/test_state.py`:

```python
def test_appstate_jar_path_default():
    state = AppState()
    assert state.jar_path.get() == "osmose-java/osmose.jar"


def test_appstate_jar_path_set():
    state = AppState()
    state.jar_path.set("/path/to/osmose.jar")
    assert state.jar_path.get() == "/path/to/osmose.jar"


def test_sync_inputs_updates_config():
    """sync_inputs should update state.config for non-indexed fields with matching input values."""
    from ui.state import sync_inputs

    state = AppState()
    state.reset_to_defaults()

    # Simulate an input namespace with attribute access
    class FakeInput:
        def __getattr__(self, name):
            def getter():
                if name == "simulation_nspecies":
                    return 5
                if name == "simulation_time_ndtperyear":
                    return 12
                return None
            return getter

    changed = sync_inputs(FakeInput(), state, ["simulation.nspecies", "simulation.time.ndtperyear"])
    assert changed["simulation.nspecies"] == "5"
    assert changed["simulation.time.ndtperyear"] == "12"
    assert state.config.get()["simulation.nspecies"] == "5"


def test_sync_inputs_skips_none():
    """sync_inputs should skip keys where the input value is None."""
    from ui.state import sync_inputs

    state = AppState()
    state.reset_to_defaults()

    class FakeInput:
        def __getattr__(self, name):
            return lambda: None

    changed = sync_inputs(FakeInput(), state, ["simulation.nspecies"])
    assert changed == {}
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_state.py -v -k "jar_path or sync_inputs"`
Expected: FAIL (AttributeError: jar_path / ImportError: sync_inputs)

**Step 3: Write the implementation**

In `ui/state.py`, add `jar_path` to `AppState.__init__` and add the `sync_inputs` function:

After the existing imports, the `AppState.__init__` method should have this added after `self.registry = REGISTRY`:

```python
        self.jar_path: reactive.Value[str] = reactive.Value("osmose-java/osmose.jar")
```

Add this module-level function after the `AppState` class:

```python
def sync_inputs(
    input: object,
    state: AppState,
    keys: list[str],
) -> dict[str, str]:
    """Read Shiny inputs for the given OSMOSE keys and update state.config.

    For each key, computes the input ID via key.replace(".", "_"), reads the
    value from input, and calls state.update_config() if non-None.

    Returns:
        Dict of keys that were actually updated with their new values.
    """
    changed: dict[str, str] = {}
    for key in keys:
        input_id = key.replace(".", "_")
        try:
            val = getattr(input, input_id)()
        except (AttributeError, TypeError):
            continue
        if val is not None:
            str_val = str(val)
            changed[key] = str_val
            state.update_config(key, str_val)
    return changed
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_state.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add ui/state.py tests/test_state.py
git commit -m "feat: add jar_path to AppState and sync_inputs utility"
```

---

### Task 2: Wire Setup page input syncing

**Files:**
- Modify: `ui/pages/setup.py`
- Test: `tests/test_sync_setup.py`

**Step 1: Write the failing test**

Create `tests/test_sync_setup.py`:

```python
"""Tests for setup page input syncing to state."""

from ui.state import AppState, sync_inputs
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
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sync_setup.py -v`
Expected: FAIL (ImportError: cannot import name 'SETUP_GLOBAL_KEYS')

**Step 3: Write the implementation**

Rewrite `ui/pages/setup.py`:

```python
"""Species & Simulation setup page."""

from shiny import ui, reactive, render

from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from ui.components.param_form import render_category, render_species_params
from ui.state import sync_inputs

# Keys for non-indexed simulation fields (synced automatically)
SETUP_GLOBAL_KEYS: list[str] = [f.key_pattern for f in SIMULATION_FIELDS if not f.advanced]


def get_species_keys(species_idx: int, show_advanced: bool = False) -> list[str]:
    """Return resolved OSMOSE keys for one species."""
    keys = []
    for f in SPECIES_FIELDS:
        if f.advanced and not show_advanced:
            continue
        keys.append(f.resolve_key(species_idx))
    return keys


def setup_ui():
    return ui.page_fluid(
        ui.layout_columns(
            # Left column: Simulation settings
            ui.card(
                ui.card_header("Simulation Settings"),
                render_category(
                    [f for f in SIMULATION_FIELDS if not f.advanced],
                ),
            ),
            # Right column: Species configuration (dynamic)
            ui.card(
                ui.card_header("Species Configuration"),
                ui.input_numeric("n_species", "Number of focal species", value=3, min=1, max=20),
                ui.input_switch("show_advanced_species", "Show advanced parameters", value=False),
                ui.output_ui("species_panels"),
            ),
            col_widths=[4, 8],
        ),
    )


def setup_server(input, output, session, state):
    @render.ui
    def species_panels():
        n = input.n_species()
        show_adv = input.show_advanced_species()
        panels = []
        for i in range(n):
            name = f"Species {i}"
            panels.append(
                render_species_params(
                    SPECIES_FIELDS,
                    species_idx=i,
                    species_name=name,
                    show_advanced=show_adv,
                )
            )
        return ui.div(*panels)

    @reactive.effect
    def sync_simulation_inputs():
        """Auto-sync simulation fields to state.config."""
        sync_inputs(input, state, SETUP_GLOBAL_KEYS)

    @reactive.effect
    def sync_species_inputs():
        """Auto-sync species fields to state.config."""
        n = input.n_species()
        show_adv = input.show_advanced_species()
        # Update nspecies in config
        state.update_config("simulation.nspecies", str(n))
        # Sync each species' fields
        for i in range(n):
            keys = get_species_keys(i, show_adv)
            sync_inputs(input, state, keys)
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sync_setup.py -v`
Expected: All 3 tests PASS

**Step 5: Run full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add ui/pages/setup.py tests/test_sync_setup.py
git commit -m "feat: wire Setup page input syncing to AppState"
```

---

### Task 3: Wire Grid, Forcing, Fishing, Movement page input syncing

**Files:**
- Modify: `ui/pages/grid.py`
- Modify: `ui/pages/forcing.py`
- Modify: `ui/pages/fishing.py`
- Modify: `ui/pages/movement.py`
- Test: `tests/test_sync_config_pages.py`

**Step 1: Write the failing tests**

Create `tests/test_sync_config_pages.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sync_config_pages.py -v`
Expected: FAIL (ImportError)

**Step 3: Implement Grid page sync**

Rewrite `ui/pages/grid.py` — add `GRID_GLOBAL_KEYS` and sync effect to `grid_server`:

```python
"""Grid configuration page."""

import plotly.graph_objects as go
from shiny import ui, reactive
from shinywidgets import output_widget, render_plotly

from osmose.schema.grid import GRID_FIELDS
from ui.components.param_form import render_field
from ui.state import sync_inputs

GRID_GLOBAL_KEYS: list[str] = [f.key_pattern for f in GRID_FIELDS if not f.indexed]


def make_grid_preview(
    ul_lat: float,
    ul_lon: float,
    lr_lat: float,
    lr_lon: float,
    nx: int = 0,
    ny: int = 0,
) -> go.Figure:
    """Create a plotly figure showing the grid extent."""
    fig = go.Figure()
    lats = [ul_lat, ul_lat, lr_lat, lr_lat, ul_lat]
    lons = [ul_lon, lr_lon, lr_lon, ul_lon, ul_lon]
    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="lines",
            line=dict(width=3, color="#e67e22"),
            name="Grid extent",
        )
    )
    fig.update_geos(
        projection_type="natural earth",
        showcoastlines=True,
        showland=True,
        landcolor="#2c3e50",
        oceancolor="#1a252f",
    )
    title = f"Grid: {ny}x{nx}" if nx and ny else "Grid Extent"
    fig.update_layout(title=title, template="plotly_dark", height=400)
    return fig


def grid_ui():
    grid_type_field = next((f for f in GRID_FIELDS if "classname" in f.key_pattern), None)
    regular_fields = [
        f
        for f in GRID_FIELDS
        if (
            f.key_pattern.startswith("grid.n")
            or f.key_pattern.startswith("grid.up")
            or f.key_pattern.startswith("grid.low")
        )
        and "netcdf" not in f.key_pattern
    ]
    netcdf_fields = [
        f for f in GRID_FIELDS if "netcdf" in f.key_pattern or f.key_pattern.startswith("grid.var")
    ]

    return ui.page_fluid(
        ui.layout_columns(
            ui.card(
                ui.card_header("Grid Type"),
                render_field(grid_type_field) if grid_type_field else ui.div(),
                ui.hr(),
                ui.h5("Regular Grid Settings"),
                *[render_field(f) for f in regular_fields],
                ui.hr(),
                ui.h5("NetCDF Grid Settings"),
                *[render_field(f) for f in netcdf_fields if not f.advanced],
            ),
            ui.card(
                ui.card_header("Grid Preview"),
                ui.p("Upload a grid mask or configure coordinates to see a preview."),
                output_widget("grid_preview"),
            ),
            col_widths=[6, 6],
        ),
    )


def grid_server(input, output, session, state):
    @render_plotly
    def grid_preview():
        ul_lat = float(input.grid_upleft_lat() or 0)
        ul_lon = float(input.grid_upleft_lon() or 0)
        lr_lat = float(input.grid_lowright_lat() or 0)
        lr_lon = float(input.grid_lowright_lon() or 0)
        nx = int(input.grid_ncolumn() or 0)
        ny = int(input.grid_nline() or 0)
        return make_grid_preview(ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)

    @reactive.effect
    def sync_grid_inputs():
        sync_inputs(input, state, GRID_GLOBAL_KEYS)
```

**Step 4: Implement Forcing page sync**

Rewrite `ui/pages/forcing.py`:

```python
"""Environmental forcing / LTL configuration page."""

from shiny import ui, reactive, render

from osmose.schema.bioenergetics import BIOENERGETICS_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from ui.components.param_form import render_field
from ui.state import sync_inputs

FORCING_GLOBAL_KEYS: list[str] = [f.key_pattern for f in LTL_FIELDS if not f.indexed]
_TEMP_KEYS: list[str] = [
    f.key_pattern
    for f in BIOENERGETICS_FIELDS
    if f.key_pattern.startswith("temperature.") and not f.indexed
]


def forcing_ui():
    global_ltl = [f for f in LTL_FIELDS if not f.indexed]
    temp_fields = [f for f in BIOENERGETICS_FIELDS if f.key_pattern.startswith("temperature.")]

    return ui.page_fluid(
        ui.layout_columns(
            ui.card(
                ui.card_header("Lower Trophic Level (Plankton)"),
                ui.h5("Global LTL Settings"),
                *[render_field(f) for f in global_ltl],
                ui.hr(),
                ui.input_numeric(
                    "n_resources", "Number of resource groups", value=3, min=0, max=20
                ),
                ui.output_ui("resource_panels"),
            ),
            ui.card(
                ui.card_header("Environmental Forcing"),
                ui.h5("Temperature"),
                *[render_field(f) for f in temp_fields if not f.advanced],
                ui.hr(),
                ui.p(
                    "Upload NetCDF forcing data for spatially-varying temperature, "
                    "oxygen, or other environmental variables."
                ),
            ),
            col_widths=[7, 5],
        ),
    )


def forcing_server(input, output, session, state):
    @render.ui
    def resource_panels():
        n = input.n_resources()
        panels = []
        for i in range(n):
            resource_fields = [f for f in LTL_FIELDS if f.indexed]
            card = ui.card(
                ui.card_header(f"Resource Group {i}"),
                *[render_field(f, species_idx=i) for f in resource_fields],
            )
            panels.append(card)
        return ui.div(*panels)

    @reactive.effect
    def sync_forcing_inputs():
        sync_inputs(input, state, FORCING_GLOBAL_KEYS + _TEMP_KEYS)

    @reactive.effect
    def sync_resource_inputs():
        n = input.n_resources()
        indexed_fields = [f for f in LTL_FIELDS if f.indexed]
        for i in range(n):
            keys = [f.resolve_key(i) for f in indexed_fields]
            sync_inputs(input, state, keys)
```

**Step 5: Implement Fishing page sync**

Rewrite `ui/pages/fishing.py`:

```python
"""Fishing configuration page."""

from shiny import ui, reactive, render

from osmose.schema.fishing import FISHING_FIELDS
from ui.components.param_form import render_field
from ui.state import sync_inputs

FISHING_GLOBAL_KEYS: list[str] = [f.key_pattern for f in FISHING_FIELDS if not f.indexed]


def fishing_ui():
    global_fields = [f for f in FISHING_FIELDS if not f.indexed]

    return ui.page_fluid(
        ui.layout_columns(
            ui.card(
                ui.card_header("Fisheries Module"),
                *[render_field(f) for f in global_fields],
                ui.hr(),
                ui.input_numeric("n_fisheries", "Number of fisheries", value=1, min=0, max=20),
                ui.output_ui("fishery_panels"),
            ),
            ui.card(
                ui.card_header("Marine Protected Areas"),
                ui.input_numeric("n_mpas", "Number of MPAs", value=0, min=0, max=10),
                ui.output_ui("mpa_panels"),
            ),
            col_widths=[8, 4],
        ),
    )


def fishing_server(input, output, session, state):
    @render.ui
    def fishery_panels():
        n = input.n_fisheries()
        fishery_fields = [f for f in FISHING_FIELDS if f.indexed and "fsh" in f.key_pattern]
        panels = []
        for i in range(n):
            card = ui.card(
                ui.card_header(f"Fishery {i}"),
                *[render_field(f, species_idx=i) for f in fishery_fields],
            )
            panels.append(card)
        return ui.div(*panels)

    @render.ui
    def mpa_panels():
        n = input.n_mpas()
        mpa_fields = [f for f in FISHING_FIELDS if f.indexed and "mpa" in f.key_pattern]
        panels = []
        for i in range(n):
            card = ui.card(
                ui.card_header(f"MPA {i}"),
                *[render_field(f, species_idx=i) for f in mpa_fields],
            )
            panels.append(card)
        return ui.div(*panels)

    @reactive.effect
    def sync_fishing_inputs():
        sync_inputs(input, state, FISHING_GLOBAL_KEYS)

    @reactive.effect
    def sync_fishery_inputs():
        n = input.n_fisheries()
        fishery_fields = [f for f in FISHING_FIELDS if f.indexed and "fsh" in f.key_pattern]
        for i in range(n):
            keys = [f.resolve_key(i) for f in fishery_fields]
            sync_inputs(input, state, keys)

    @reactive.effect
    def sync_mpa_inputs():
        n = input.n_mpas()
        mpa_fields = [f for f in FISHING_FIELDS if f.indexed and "mpa" in f.key_pattern]
        for i in range(n):
            keys = [f.resolve_key(i) for f in mpa_fields]
            sync_inputs(input, state, keys)
```

**Step 6: Implement Movement page sync + fix hardcoded species**

Rewrite `ui/pages/movement.py`:

```python
"""Movement / spatial distribution page."""

from shiny import ui, reactive, render

from osmose.schema.movement import MOVEMENT_FIELDS
from ui.components.param_form import render_field
from ui.state import sync_inputs

MOVEMENT_GLOBAL_KEYS: list[str] = [f.key_pattern for f in MOVEMENT_FIELDS if not f.indexed]


def movement_ui():
    global_fields = [f for f in MOVEMENT_FIELDS if not f.indexed]

    return ui.page_fluid(
        ui.layout_columns(
            ui.card(
                ui.card_header("Movement Settings"),
                *[render_field(f) for f in global_fields if not f.advanced],
                ui.hr(),
                ui.h5("Per-Species Distribution Method"),
                ui.output_ui("species_movement_panels"),
            ),
            ui.card(
                ui.card_header("Distribution Maps"),
                ui.input_numeric("n_maps", "Number of distribution maps", value=1, min=0, max=50),
                ui.output_ui("map_panels"),
            ),
            col_widths=[5, 7],
        ),
    )


def movement_server(input, output, session, state):
    @render.ui
    def species_movement_panels():
        per_species = [f for f in MOVEMENT_FIELDS if f.indexed and "map" not in f.key_pattern]
        n_species = int(state.config.get().get("simulation.nspecies", "3"))
        panels = []
        for i in range(n_species):
            panels.extend([render_field(f, species_idx=i) for f in per_species])
        return ui.div(*panels)

    @render.ui
    def map_panels():
        n = input.n_maps()
        map_fields = [f for f in MOVEMENT_FIELDS if f.indexed and "map" in f.key_pattern]
        panels = []
        for i in range(n):
            card = ui.card(
                ui.card_header(f"Map {i}"),
                *[render_field(f, species_idx=i) for f in map_fields],
            )
            panels.append(card)
        return ui.div(*panels)

    @reactive.effect
    def sync_movement_inputs():
        sync_inputs(input, state, MOVEMENT_GLOBAL_KEYS)

    @reactive.effect
    def sync_species_movement_inputs():
        per_species = [f for f in MOVEMENT_FIELDS if f.indexed and "map" not in f.key_pattern]
        n_species = int(state.config.get().get("simulation.nspecies", "3"))
        for i in range(n_species):
            keys = [f.resolve_key(i) for f in per_species]
            sync_inputs(input, state, keys)

    @reactive.effect
    def sync_map_inputs():
        n = input.n_maps()
        map_fields = [f for f in MOVEMENT_FIELDS if f.indexed and "map" in f.key_pattern]
        for i in range(n):
            keys = [f.resolve_key(i) for f in map_fields]
            sync_inputs(input, state, keys)
```

**Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sync_config_pages.py -v`
Expected: All 5 tests PASS

**Step 8: Run full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All tests PASS

**Step 9: Commit**

```bash
git add ui/pages/grid.py ui/pages/forcing.py ui/pages/fishing.py ui/pages/movement.py tests/test_sync_config_pages.py
git commit -m "feat: wire Grid, Forcing, Fishing, Movement input syncing"
```

---

### Task 4: Wire Run page to read JAR path from AppState

**Files:**
- Modify: `ui/pages/run.py`

**Step 1: Modify run_server to sync JAR path to AppState**

In `ui/pages/run.py`, add a `@reactive.effect` that syncs the JAR path input to `state.jar_path`, and modify `handle_run` to read from `state.jar_path`:

Add `from shiny import reactive` to imports (already present). Add this to `run_server()` before the existing `handle_run`:

```python
    @reactive.effect
    def sync_jar_path():
        state.jar_path.set(input.jar_path())
```

And change line 91 in `handle_run` from:
```python
        jar_path = Path(input.jar_path())
```
to:
```python
        jar_path = Path(state.jar_path.get())
```

**Step 2: Run full suite to verify nothing breaks**

Run: `.venv/bin/python -m pytest -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add ui/pages/run.py
git commit -m "feat: sync Run page JAR path to AppState"
```

---

### Task 5: Wire Calibration Start/Stop/Sensitivity handlers

**Files:**
- Modify: `ui/pages/calibration.py`
- Test: `tests/test_ui_calibration_handlers.py`

**Step 1: Write the failing tests**

Create `tests/test_ui_calibration_handlers.py`:

```python
"""Tests for calibration page handler helper functions."""

from ui.pages.calibration import collect_selected_params, build_free_params


def test_collect_selected_params():
    """Should return keys where the corresponding checkbox is True."""
    from ui.state import AppState

    state = AppState()
    state.reset_to_defaults()

    # Simulate checkboxes: only species.k.sp0 is checked
    class FakeInput:
        def __getattr__(self, name):
            if name == "cal_param_species_k_sp0":
                return lambda: True
            return lambda: False

    params = collect_selected_params(FakeInput(), state)
    keys = [p["key"] for p in params]
    assert "species.k.sp0" in keys


def test_collect_selected_params_empty():
    """Should return empty list when nothing is checked."""
    from ui.state import AppState

    state = AppState()
    state.reset_to_defaults()

    class FakeInput:
        def __getattr__(self, name):
            return lambda: False

    params = collect_selected_params(FakeInput(), state)
    assert params == []


def test_build_free_params():
    """Should create FreeParameter objects from selected param dicts."""
    from osmose.calibration.problem import FreeParameter

    selected = [
        {"key": "species.k.sp0", "lower": 0.1, "upper": 1.0},
        {"key": "species.k.sp1", "lower": 0.1, "upper": 1.0},
    ]
    free_params = build_free_params(selected)
    assert len(free_params) == 2
    assert isinstance(free_params[0], FreeParameter)
    assert free_params[0].key == "species.k.sp0"
    assert free_params[0].lower_bound == 0.1
    assert free_params[0].upper_bound == 1.0
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration_handlers.py -v`
Expected: FAIL (ImportError)

**Step 3: Write the implementation**

Rewrite `ui/pages/calibration.py` with Start/Stop/Sensitivity handlers:

```python
# ui/pages/calibration.py
"""Calibration page - multi-objective optimization and sensitivity analysis."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_plotly

from osmose.calibration.objectives import biomass_rmse, diet_distance
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem
from osmose.calibration.sensitivity import SensitivityAnalyzer
from osmose.config.writer import OsmoseConfigWriter
from osmose.schema.base import ParamType
from osmose.schema.registry import ParameterRegistry


# ── Helper functions (module-level, testable without Shiny) ──────────────────


def get_calibratable_params(registry: ParameterRegistry, n_species: int) -> list[dict]:
    """Get all numeric species-indexed params suitable for calibration."""
    params = []
    for field in registry.all_fields():
        if not field.indexed:
            continue
        if field.param_type not in (ParamType.FLOAT, ParamType.INT):
            continue
        if field.min_val is None or field.max_val is None:
            continue
        for i in range(n_species):
            key = field.resolve_key(i)
            params.append(
                {
                    "key": key,
                    "label": f"{field.description} (sp{i})",
                    "category": field.category,
                    "lower": field.min_val,
                    "upper": field.max_val,
                }
            )
    return params


def collect_selected_params(input: object, state) -> list[dict]:
    """Return calibratable param dicts where the checkbox is checked."""
    cfg = state.config.get()
    n_species = int(cfg.get("simulation.nspecies", "3"))
    all_params = get_calibratable_params(state.registry, n_species)
    selected = []
    for p in all_params:
        input_id = f"cal_param_{p['key'].replace('.', '_')}"
        try:
            if getattr(input, input_id)():
                selected.append(p)
        except (AttributeError, TypeError):
            continue
    return selected


def build_free_params(selected: list[dict]) -> list[FreeParameter]:
    """Convert selected param dicts to FreeParameter objects."""
    return [
        FreeParameter(key=p["key"], lower_bound=p["lower"], upper_bound=p["upper"])
        for p in selected
    ]


def make_convergence_chart(history: list[float]) -> go.Figure:
    """Line chart of best objective value per generation."""
    if not history:
        return go.Figure().update_layout(title="Convergence", template="plotly_dark")
    fig = px.line(x=list(range(len(history))), y=history, title="Convergence")
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Best Objective",
        template="plotly_dark",
    )
    return fig


def make_pareto_chart(F: np.ndarray, obj_names: list[str]) -> go.Figure:
    """Scatter plot of Pareto front (2 objectives)."""
    fig = px.scatter(x=F[:, 0], y=F[:, 1], title="Pareto Front")
    fig.update_layout(
        xaxis_title=obj_names[0] if len(obj_names) > 0 else "Obj 1",
        yaxis_title=obj_names[1] if len(obj_names) > 1 else "Obj 2",
        template="plotly_dark",
    )
    return fig


def make_sensitivity_chart(result: dict) -> go.Figure:
    """Bar chart of Sobol sensitivity indices."""
    names = result["param_names"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="S1 (First-order)", x=names, y=result["S1"]))
    fig.add_trace(go.Bar(name="ST (Total-order)", x=names, y=result["ST"]))
    fig.update_layout(
        title="Sobol Sensitivity Indices",
        barmode="group",
        template="plotly_dark",
    )
    return fig


# ── Shiny UI / Server ───────────────────────────────────────────────────────


def calibration_ui():
    return ui.page_fluid(
        ui.layout_columns(
            # Left: Configuration
            ui.card(
                ui.card_header("Calibration Setup"),
                ui.input_select(
                    "cal_algorithm",
                    "Algorithm",
                    choices={
                        "nsga2": "NSGA-II (Direct)",
                        "surrogate": "GP Surrogate",
                    },
                ),
                ui.input_numeric("cal_pop_size", "Population size", value=50, min=10, max=500),
                ui.input_numeric("cal_generations", "Generations", value=100, min=10, max=1000),
                ui.input_numeric("cal_n_parallel", "Parallel workers", value=4, min=1, max=32),
                ui.hr(),
                ui.h5("Free Parameters"),
                ui.p("Select parameters to optimize:", style="color: #999; font-size: 13px;"),
                ui.output_ui("free_param_selector"),
                ui.hr(),
                ui.h5("Objectives"),
                ui.input_file("observed_biomass", "Upload observed biomass CSV", accept=[".csv"]),
                ui.input_file("observed_diet", "Upload observed diet matrix CSV", accept=[".csv"]),
                ui.hr(),
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_start_cal", "Start Calibration", class_="btn-success w-100"
                    ),
                    ui.input_action_button("btn_stop_cal", "Stop", class_="btn-danger w-100"),
                    col_widths=[8, 4],
                ),
            ),
            # Right: Results
            ui.navset_card_tab(
                ui.nav_panel(
                    "Progress",
                    ui.div(
                        ui.output_text("cal_status"),
                        output_widget("convergence_chart"),
                    ),
                ),
                ui.nav_panel(
                    "Pareto Front",
                    ui.div(
                        output_widget("pareto_chart"),
                    ),
                ),
                ui.nav_panel(
                    "Best Parameters",
                    ui.div(
                        ui.output_ui("best_params_table"),
                    ),
                ),
                ui.nav_panel(
                    "Sensitivity",
                    ui.div(
                        ui.input_action_button(
                            "btn_sensitivity",
                            "Run Sensitivity Analysis",
                            class_="btn-info w-100",
                        ),
                        output_widget("sensitivity_chart"),
                    ),
                ),
            ),
            col_widths=[4, 8],
        ),
    )


def calibration_server(input, output, session, state):
    cal_history = reactive.value([])
    cal_F = reactive.value(None)
    cal_X = reactive.value(None)
    sensitivity_result = reactive.value(None)
    cal_thread = reactive.value(None)
    cancel_flag = reactive.value(False)

    @render.text
    def cal_status():
        hist = cal_history.get()
        if not hist:
            return "Ready. Configure parameters and objectives, then click Start."
        return f"Generation {len(hist)} — Best: {hist[-1]:.4f}"

    @render.ui
    def free_param_selector():
        cfg = state.config.get()
        n_str = cfg.get("simulation.nspecies", "3")
        n_species = int(n_str) if n_str else 3
        params = get_calibratable_params(state.registry, n_species)
        checkboxes = [
            ui.input_checkbox(
                f"cal_param_{p['key'].replace('.', '_')}",
                p["label"],
                value=False,
            )
            for p in params
        ]
        return ui.div(*checkboxes)

    @reactive.effect
    @reactive.event(input.btn_start_cal)
    def handle_start_cal():
        # Collect selected params
        selected = collect_selected_params(input, state)
        if not selected:
            cal_history.set([])
            return

        # Validate JAR
        jar_path = Path(state.jar_path.get())
        if not jar_path.exists():
            return

        # Validate at least one objective file
        obs_bio = input.observed_biomass()
        obs_diet = input.observed_diet()
        if not obs_bio and not obs_diet:
            return

        # Build problem
        free_params = build_free_params(selected)
        work_dir = Path(tempfile.mkdtemp(prefix="osmose_cal_"))

        # Write base config
        writer = OsmoseConfigWriter()
        config_dir = work_dir / "config"
        writer.write(state.config.get(), config_dir)
        base_config = config_dir / "osm_all-parameters.csv"

        # Build objectives
        objective_fns = []
        if obs_bio:
            obs_bio_df = pd.read_csv(obs_bio[0]["datapath"])
            objective_fns.append(lambda r, df=obs_bio_df: biomass_rmse(r.biomass(), df))
        if obs_diet:
            obs_diet_df = pd.read_csv(obs_diet[0]["datapath"])
            objective_fns.append(lambda r, df=obs_diet_df: diet_distance(r.diet_matrix(), df))

        if not objective_fns:
            return

        problem = OsmoseCalibrationProblem(
            free_params=free_params,
            objective_fns=objective_fns,
            base_config_path=base_config,
            jar_path=jar_path,
            work_dir=work_dir,
        )

        # Reset state
        cancel_flag.set(False)
        cal_history.set([])
        cal_F.set(None)
        cal_X.set(None)

        # Run in background thread
        def run_optimization():
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination

            algorithm = NSGA2(pop_size=int(input.cal_pop_size()))
            termination = get_termination("n_gen", int(input.cal_generations()))

            res = minimize(
                problem,
                algorithm,
                termination,
                seed=42,
                verbose=False,
            )

            if res.F is not None:
                cal_F.set(res.F)
                cal_X.set(res.X)
                # Build convergence history from result
                if hasattr(res, "history") and res.history:
                    history = [
                        float(np.min(gen.opt.get("F").sum(axis=1)))
                        for gen in res.history
                        if gen.opt is not None
                    ]
                    cal_history.set(history)

        thread = threading.Thread(target=run_optimization, daemon=True)
        thread.start()
        cal_thread.set(thread)

    @reactive.effect
    @reactive.event(input.btn_stop_cal)
    def handle_stop_cal():
        cancel_flag.set(True)

    @reactive.effect
    @reactive.event(input.btn_sensitivity)
    def handle_sensitivity():
        selected = collect_selected_params(input, state)
        if not selected:
            return

        jar_path = Path(state.jar_path.get())
        if not jar_path.exists():
            return

        param_names = [p["key"] for p in selected]
        param_bounds = [(p["lower"], p["upper"]) for p in selected]
        analyzer = SensitivityAnalyzer(param_names, param_bounds)

        obs_bio = input.observed_biomass()
        if not obs_bio:
            return
        obs_bio_df = pd.read_csv(obs_bio[0]["datapath"])

        def run_sensitivity():
            samples = analyzer.generate_samples(n_base=64)
            work_dir = Path(tempfile.mkdtemp(prefix="osmose_sens_"))
            writer = OsmoseConfigWriter()
            config_dir = work_dir / "config"
            writer.write(state.config.get(), config_dir)
            base_config = config_dir / "osm_all-parameters.csv"

            Y = np.zeros(samples.shape[0])
            for idx, row in enumerate(samples):
                overrides = {
                    selected[j]["key"]: str(row[j]) for j in range(len(selected))
                }
                try:
                    problem = OsmoseCalibrationProblem(
                        free_params=build_free_params(selected),
                        objective_fns=[lambda r, df=obs_bio_df: biomass_rmse(r.biomass(), df)],
                        base_config_path=base_config,
                        jar_path=jar_path,
                        work_dir=work_dir / f"sens_{idx}",
                    )
                    result = problem._run_single(overrides, run_id=idx)
                    Y[idx] = result[0]
                except Exception:
                    Y[idx] = float("inf")

            result = analyzer.analyze(Y)
            sensitivity_result.set(result)

        thread = threading.Thread(target=run_sensitivity, daemon=True)
        thread.start()

    @render_plotly
    def convergence_chart():
        return make_convergence_chart(cal_history.get())

    @render_plotly
    def pareto_chart():
        F = cal_F.get()
        if F is None:
            return go.Figure().update_layout(
                title="Pareto Front (run calibration first)", template="plotly_dark"
            )
        return make_pareto_chart(F, ["Biomass RMSE", "Diet Distance"])

    @render.ui
    def best_params_table():
        X = cal_X.get()
        F = cal_F.get()
        if X is None or F is None:
            return ui.div(
                "Run calibration to see best parameters.",
                style="padding: 20px; text-align: center; color: #999;",
            )
        order = np.argsort(F.sum(axis=1))[:10]
        selected = collect_selected_params(input, state)
        headers = [ui.tags.th(p["key"].split(".")[-1]) for p in selected]
        headers.append(ui.tags.th("Total Obj"))
        rows = []
        for idx in order:
            cells = [ui.tags.td(f"{v:.4f}") for v in X[idx]]
            cells.append(ui.tags.td(f"{F[idx].sum():.4f}"))
            rows.append(ui.tags.tr(*cells))
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(*headers)),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
        )

    @render_plotly
    def sensitivity_chart():
        result = sensitivity_result.get()
        if result is None:
            return go.Figure().update_layout(
                title="Sensitivity (click Run)", template="plotly_dark"
            )
        return make_sensitivity_chart(result)
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration_handlers.py tests/test_ui_calibration.py -v`
Expected: All tests PASS

**Step 5: Run full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add ui/pages/calibration.py tests/test_ui_calibration_handlers.py
git commit -m "feat: wire Calibration Start/Stop/Sensitivity handlers"
```

---

### Task 6: Wire Advanced page import/export

**Files:**
- Modify: `ui/pages/advanced.py`
- Test: `tests/test_ui_advanced_io.py`

**Step 1: Write the failing tests**

Create `tests/test_ui_advanced_io.py`:

```python
"""Tests for Advanced page import/export logic."""

from pathlib import Path

from osmose.config.reader import OsmoseConfigReader
from osmose.config.writer import OsmoseConfigWriter


def test_import_config_merges(tmp_path):
    """Importing a config should merge keys into existing state."""
    # Write a small config file
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
```

**Step 2: Run tests to verify they pass (these test the backend, should already pass)**

Run: `.venv/bin/python -m pytest tests/test_ui_advanced_io.py -v`
Expected: All 2 tests PASS

**Step 3: Write the implementation**

Rewrite `ui/pages/advanced.py` with import/export handlers:

```python
"""Advanced raw config editor page."""

import tempfile
from pathlib import Path

from shiny import render, ui

from osmose.config.reader import OsmoseConfigReader
from osmose.config.writer import OsmoseConfigWriter
from osmose.schema.registry import ParameterRegistry
from ui.state import REGISTRY


def advanced_ui():
    categories = ["all"] + REGISTRY.categories()

    return ui.page_fluid(
        ui.layout_columns(
            # Controls
            ui.card(
                ui.card_header("Config I/O"),
                ui.input_file(
                    "import_config", "Import OSMOSE config", accept=[".csv", ".properties"]
                ),
                ui.download_button(
                    "export_config", "Export Current Config", class_="btn-primary w-100"
                ),
            ),
            ui.card(
                ui.card_header("Filters"),
                ui.input_select(
                    "adv_category", "Category", choices={c: c.title() for c in categories}
                ),
                ui.input_text("adv_search", "Search parameters", placeholder="Type to filter..."),
                ui.p(
                    f"Total parameters in registry: {len(REGISTRY.all_fields())}",
                    style="color: #999; font-size: 12px;",
                ),
            ),
            col_widths=[4, 8],
        ),
        ui.card(
            ui.card_header("All Parameters"),
            ui.output_ui("param_table"),
        ),
    )


def advanced_server(input, output, session, state):
    @render.ui
    def param_table():
        category = input.adv_category()
        search = input.adv_search().lower() if input.adv_search() else ""

        if category == "all":
            fields = REGISTRY.all_fields()
        else:
            fields = REGISTRY.fields_by_category(category)

        if search:
            fields = [
                f
                for f in fields
                if search in f.key_pattern.lower() or search in f.description.lower()
            ]

        if not fields:
            return ui.div("No parameters match your filter.", style="padding: 20px; color: #999;")

        # Show current config values
        cfg = state.config.get()

        rows = []
        for f in fields[:100]:
            current_val = cfg.get(f.key_pattern, "-")
            rows.append(
                ui.tags.tr(
                    ui.tags.td(f.key_pattern, style="font-family: monospace; font-size: 12px;"),
                    ui.tags.td(f.param_type.value),
                    ui.tags.td(str(current_val)),
                    ui.tags.td(f.category),
                    ui.tags.td(
                        f.description[:60] + "..." if len(f.description) > 60 else f.description
                    ),
                )
            )

        return ui.tags.div(
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Key"),
                        ui.tags.th("Type"),
                        ui.tags.th("Current Value"),
                        ui.tags.th("Category"),
                        ui.tags.th("Description"),
                    )
                ),
                ui.tags.tbody(*rows),
                class_="table table-striped table-hover table-sm",
            ),
            style="max-height: 600px; overflow-y: auto;",
        )

    @render.ui
    @render.event(input.import_config)
    def handle_import():
        file_info = input.import_config()
        if not file_info:
            return
        filepath = Path(file_info[0]["datapath"])
        reader = OsmoseConfigReader()
        loaded = reader.read_file(filepath)
        # Merge into current config
        cfg = dict(state.config.get())
        cfg.update(loaded)
        state.config.set(cfg)

    @render.download(filename="osm_all-parameters.csv")
    def export_config():
        work_dir = Path(tempfile.mkdtemp(prefix="osmose_export_"))
        writer = OsmoseConfigWriter()
        writer.write(state.config.get(), work_dir)
        master = work_dir / "osm_all-parameters.csv"
        return str(master)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ui_advanced_io.py -v`
Expected: All PASS

**Step 5: Run full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add ui/pages/advanced.py tests/test_ui_advanced_io.py
git commit -m "feat: wire Advanced page import/export handlers"
```

---

### Task 7: Full test suite, lint, format, verify

**Step 1: Run all tests**

Run: `.venv/bin/python -m pytest --cov=osmose --cov=ui -v`
Expected: All tests PASS, coverage stays at or above 65%

**Step 2: Lint**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`
Expected: No errors (or fix any that appear)

**Step 3: Format**

Run: `.venv/bin/ruff format osmose/ ui/ tests/`

**Step 4: Start the app and verify**

Run: `.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000`

Verify:
- Setup tab: Change species count → config updates
- Grid tab: Change coordinates → preview updates AND config updates
- Forcing tab: Change LTL settings → config updates
- Fishing tab: Add fisheries/MPAs → config updates
- Movement tab: Species count follows Setup page value (not hardcoded 3)
- Run tab: JAR path syncs to AppState
- Calibration tab: Start button validates inputs, checkboxes reflect selected params
- Advanced tab: Import loads config file, Export downloads config, table shows current values
- Scenarios tab: Save captures full config from all pages

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: format and lint Phase 3 code"
```

---

### Task 8: Push and verify CI

**Step 1: Push**

Run: `git push origin master`

**Step 2: Check CI**

Run: `gh run list --limit 1`
Expected: CI passes (lint + test)

**Step 3: Update memory**

Update MEMORY.md with Phase 3 status, new test count, and commit history.

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | Add jar_path + sync_inputs to AppState | ui/state.py | test_state.py (+4) |
| 2 | Wire Setup page syncing | ui/pages/setup.py | test_sync_setup.py (3) |
| 3 | Wire Grid/Forcing/Fishing/Movement syncing | 4 page files | test_sync_config_pages.py (5) |
| 4 | Wire Run page JAR to AppState | ui/pages/run.py | existing suite |
| 5 | Wire Calibration handlers | ui/pages/calibration.py | test_ui_calibration_handlers.py (3) |
| 6 | Wire Advanced import/export | ui/pages/advanced.py | test_ui_advanced_io.py (2) |
| 7 | Full test + lint + verify | all | all |
| 8 | Push + CI | - | CI |

**Total new tests: ~17**
**Estimated test count after: ~207**
