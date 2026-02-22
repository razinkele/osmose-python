# Phase 2: Full UI Wiring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire all placeholder UI pages to backend modules, add shared state management, and implement full plotly visualization.

**Architecture:** Add a shared `AppState` reactive store (`ui/state.py`) that all 10 page servers share. Wire Run/Results/Calibration/Scenarios pages to their backend counterparts (`OsmoseRunner`, `OsmoseResults`, `ScenarioManager`, calibration engine). Add plotly charts via `shinywidgets`.

**Tech Stack:** Shiny for Python, shinywidgets, plotly, pymoo, SALib, xarray, pandas

---

### Task 1: Add shinywidgets dependency

**Files:**
- Modify: `pyproject.toml:11-12`
- Modify: `.venv/` (install)

**Step 1: Add shinywidgets to pyproject.toml**

In `pyproject.toml`, add `"shinywidgets>=0.7"` to the dependencies list, after `"shinyswatch>=0.9"`.

**Step 2: Install the dependency**

Run: `.venv/bin/pip install -e ".[dev]"`
Expected: shinywidgets installed successfully

**Step 3: Verify import works**

Run: `.venv/bin/python -c "from shinywidgets import output_widget, render_plotly; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add shinywidgets dependency for plotly integration"
```

---

### Task 2: Create AppState shared state module

**Files:**
- Create: `ui/state.py`
- Test: `tests/test_state.py`

**Step 1: Write the test**

Create `tests/test_state.py`:

```python
"""Tests for ui.state -- shared application state."""

from pathlib import Path

from ui.state import AppState


def test_appstate_initial_config_is_empty():
    state = AppState()
    assert state.config.get() == {}


def test_appstate_initial_output_dir_is_none():
    state = AppState()
    assert state.output_dir.get() is None


def test_appstate_initial_run_result_is_none():
    state = AppState()
    assert state.run_result.get() is None


def test_appstate_scenarios_dir_default():
    state = AppState()
    assert state.scenarios_dir == Path("data/scenarios")


def test_appstate_custom_scenarios_dir():
    state = AppState(scenarios_dir=Path("/tmp/my_scenarios"))
    assert state.scenarios_dir == Path("/tmp/my_scenarios")


def test_appstate_config_update():
    state = AppState()
    state.config.set({"simulation.nspecies": "3"})
    assert state.config.get() == {"simulation.nspecies": "3"}


def test_appstate_update_config_key():
    state = AppState()
    state.config.set({"a": "1"})
    state.update_config("b", "2")
    assert state.config.get() == {"a": "1", "b": "2"}


def test_appstate_update_config_key_overwrites():
    state = AppState()
    state.config.set({"a": "1"})
    state.update_config("a", "99")
    assert state.config.get() == {"a": "99"}


def test_appstate_collect_defaults():
    state = AppState()
    state.collect_defaults()
    cfg = state.config.get()
    # Should have simulation params
    assert "simulation.nspecies" in cfg
    assert cfg["simulation.nspecies"] == "3"
    # Should have grid params
    assert "grid.ncolumn" in cfg
```

**Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_state.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'ui.state')

**Step 3: Write the implementation**

Create `ui/state.py`:

```python
"""Shared reactive application state for all UI pages."""

from __future__ import annotations

from pathlib import Path

from shiny import reactive

from osmose.runner import RunResult
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


def _build_registry() -> ParameterRegistry:
    """Build the full parameter registry (cached at module level)."""
    reg = ParameterRegistry()
    for fields in [
        SIMULATION_FIELDS, SPECIES_FIELDS, GRID_FIELDS, PREDATION_FIELDS,
        FISHING_FIELDS, MOVEMENT_FIELDS, LTL_FIELDS, OUTPUT_FIELDS,
        BIOENERGETICS_FIELDS, ECONOMICS_FIELDS,
    ]:
        for f in fields:
            reg.register(f)
    return reg


REGISTRY = _build_registry()


class AppState:
    """Shared reactive state passed to all page server functions.

    Holds the current OSMOSE config, last run result, and output directory.
    All pages read/write through this single source of truth.
    """

    def __init__(self, scenarios_dir: Path = Path("data/scenarios")):
        self.config: reactive.Value[dict[str, str]] = reactive.value({})
        self.output_dir: reactive.Value[Path | None] = reactive.value(None)
        self.run_result: reactive.Value[RunResult | None] = reactive.value(None)
        self.scenarios_dir = scenarios_dir
        self.registry = REGISTRY

    def update_config(self, key: str, value: str) -> None:
        """Update a single key in the config dict."""
        cfg = dict(self.config.get())
        cfg[key] = value
        self.config.set(cfg)

    def collect_defaults(self) -> None:
        """Populate config with default values from the schema registry."""
        cfg: dict[str, str] = {}
        for field in self.registry.all_fields():
            if field.default is not None:
                if field.indexed:
                    # Create defaults for species 0-2 (3 species default)
                    for i in range(3):
                        key = field.resolve_key(i)
                        cfg[key] = str(field.default)
                else:
                    cfg[field.key_pattern] = str(field.default)
        self.config.set(cfg)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_state.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add ui/state.py tests/test_state.py
git commit -m "feat: add AppState shared reactive state module"
```

---

### Task 3: Wire AppState into app.py and all page servers

**Files:**
- Modify: `app.py`
- Modify: all 10 files in `ui/pages/*.py` (add `state` parameter)

**Step 1: Update app.py**

Replace the `server` function in `app.py` to create `AppState` and pass it to all page servers:

```python
from ui.state import AppState

def server(input, output, session):
    state = AppState()
    state.collect_defaults()

    setup_server(input, output, session, state)
    grid_server(input, output, session, state)
    forcing_server(input, output, session, state)
    fishing_server(input, output, session, state)
    movement_server(input, output, session, state)
    run_server(input, output, session, state)
    results_server(input, output, session, state)
    calibration_server(input, output, session, state)
    scenarios_server(input, output, session, state)
    advanced_server(input, output, session, state)
```

**Step 2: Update all 10 page server signatures**

For each `*_server` function in `ui/pages/*.py`, add `state` parameter:

```python
# Before:
def setup_server(input, output, session):

# After:
def setup_server(input, output, session, state):
```

Do this for: `setup.py`, `grid.py`, `forcing.py`, `fishing.py`, `movement.py`, `run.py`, `results.py`, `calibration.py`, `scenarios.py`, `advanced.py`.

The function body stays the same for now — just add the parameter.

**Step 3: Run all tests to verify nothing breaks**

Run: `.venv/bin/python -m pytest -v`
Expected: All 155+ tests PASS (existing tests don't call server functions directly)

**Step 4: Start the app to verify it loads**

Run: `.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000`
Expected: App starts, all 10 tabs render correctly

**Step 5: Commit**

```bash
git add app.py ui/pages/*.py ui/state.py
git commit -m "feat: wire AppState into all page servers"
```

---

### Task 4: Wire the Run page

**Files:**
- Modify: `ui/pages/run.py`
- Test: `tests/test_ui_run.py`

**Step 1: Write tests**

Create `tests/test_ui_run.py`:

```python
"""Tests for run page logic -- config writing, override parsing, status flow."""

import tempfile
from pathlib import Path

from ui.pages.run import parse_overrides, write_temp_config


def test_parse_overrides_empty():
    assert parse_overrides("") == {}


def test_parse_overrides_single():
    assert parse_overrides("simulation.nspecies=5") == {"simulation.nspecies": "5"}


def test_parse_overrides_multiple():
    text = "simulation.nspecies=5\nspecies.k.sp0=0.3"
    result = parse_overrides(text)
    assert result == {"simulation.nspecies": "5", "species.k.sp0": "0.3"}


def test_parse_overrides_skips_blank_lines():
    text = "a=1\n\nb=2\n"
    assert parse_overrides(text) == {"a": "1", "b": "2"}


def test_parse_overrides_strips_whitespace():
    text = "  a = 1  \n  b=2"
    assert parse_overrides(text) == {"a": "1", "b": "2"}


def test_write_temp_config(tmp_path):
    config = {"simulation.nspecies": "3", "species.k.sp0": "0.2"}
    config_path = write_temp_config(config, tmp_path)
    assert config_path.exists()
    assert config_path.name == "osm_all-parameters.csv"
    content = config_path.read_text()
    assert "simulation.nspecies" in content
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_ui_run.py -v`
Expected: FAIL (ImportError: cannot import name 'parse_overrides')

**Step 3: Implement run.py**

Rewrite `ui/pages/run.py` — keep the existing `run_ui()` unchanged, replace `run_server()` with the wired version. Add helper functions `parse_overrides()` and `write_temp_config()`:

```python
"""Run control page - execute OSMOSE simulations."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from shiny import ui, reactive, render

from osmose.config.writer import OsmoseConfigWriter
from osmose.runner import OsmoseRunner


def parse_overrides(text: str) -> dict[str, str]:
    """Parse a text area of key=value lines into a dict."""
    result = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


def write_temp_config(config: dict[str, str], output_dir: Path) -> Path:
    """Write config to a temp directory and return the master file path."""
    writer = OsmoseConfigWriter()
    writer.write(config, output_dir)
    return output_dir / "osm_all-parameters.csv"


def run_ui():
    # ... keep existing run_ui() unchanged ...


def run_server(input, output, session, state):
    run_log = reactive.value([])
    status = reactive.value("Idle")
    runner_ref = reactive.value(None)

    @render.text
    def run_status():
        return status.get()

    @render.ui
    def run_console():
        lines = run_log.get()
        text = "\n".join(lines[-200:]) if lines else "No output yet. Click 'Start Run' to begin."
        return ui.tags.pre(
            text,
            style="background: #111; color: #0f0; height: 500px; overflow-y: auto; "
            "padding: 12px; border-radius: 6px; font-family: 'Courier New', monospace; "
            "font-size: 13px; white-space: pre-wrap;",
        )

    @reactive.effect
    @reactive.event(input.btn_run)
    async def handle_run():
        jar_path = Path(input.jar_path())
        if not jar_path.exists():
            status.set(f"Error: JAR not found at {jar_path}")
            return

        status.set("Writing config...")
        run_log.set([])

        # Write config to temp directory
        config = state.config.get()
        work_dir = Path(tempfile.mkdtemp(prefix="osmose_run_"))
        config_path = write_temp_config(config, work_dir)

        # Parse overrides and java opts
        overrides = parse_overrides(input.param_overrides() or "")
        java_opts_text = input.java_opts() or ""
        java_opts = java_opts_text.split() if java_opts_text.strip() else None

        # Create runner
        runner = OsmoseRunner(jar_path=jar_path)
        runner_ref.set(runner)

        status.set("Running...")

        def on_progress(line: str):
            lines = list(run_log.get())
            lines.append(line)
            run_log.set(lines)

        result = await runner.run(
            config_path=config_path,
            output_dir=work_dir / "output",
            java_opts=java_opts,
            overrides=overrides,
            on_progress=on_progress,
        )

        state.run_result.set(result)
        state.output_dir.set(result.output_dir)

        if result.returncode == 0:
            status.set(f"Complete. Output: {result.output_dir}")
        else:
            status.set(f"Failed (exit code {result.returncode})")
            if result.stderr:
                lines = list(run_log.get())
                lines.append(f"--- STDERR ---\n{result.stderr}")
                run_log.set(lines)

    @reactive.effect
    @reactive.event(input.btn_cancel)
    def handle_cancel():
        runner = runner_ref.get()
        if runner:
            runner.cancel()
            status.set("Cancelled")
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ui_run.py -v`
Expected: All 6 tests PASS

**Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add ui/pages/run.py tests/test_ui_run.py
git commit -m "feat: wire Run page buttons to OsmoseRunner"
```

---

### Task 5: Wire the Results page with plotly charts

**Files:**
- Modify: `ui/pages/results.py`
- Test: `tests/test_ui_results.py`

**Step 1: Write tests**

Create `tests/test_ui_results.py`:

```python
"""Tests for results page chart generation functions."""

import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go

from ui.pages.results import (
    make_timeseries_chart,
    make_diet_heatmap,
    make_spatial_map,
)


def test_make_timeseries_chart_biomass():
    df = pd.DataFrame({
        "time": [0, 1, 2, 0, 1, 2],
        "biomass": [100, 200, 300, 50, 100, 150],
        "species": ["Anchovy", "Anchovy", "Anchovy", "Sardine", "Sardine", "Sardine"],
    })
    fig = make_timeseries_chart(df, "biomass", "Biomass")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Two species traces


def test_make_timeseries_chart_empty():
    df = pd.DataFrame()
    fig = make_timeseries_chart(df, "biomass", "Biomass")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_make_timeseries_chart_with_species_filter():
    df = pd.DataFrame({
        "time": [0, 1, 0, 1],
        "biomass": [100, 200, 50, 100],
        "species": ["Anchovy", "Anchovy", "Sardine", "Sardine"],
    })
    fig = make_timeseries_chart(df, "biomass", "Biomass", species="Anchovy")
    assert len(fig.data) == 1


def test_make_diet_heatmap():
    df = pd.DataFrame({
        "time": [0, 0],
        "species": ["Anchovy", "Anchovy"],
        "prey_Sardine": [0.6, 0.5],
        "prey_Plankton": [0.4, 0.5],
    })
    fig = make_diet_heatmap(df)
    assert isinstance(fig, go.Figure)


def test_make_diet_heatmap_empty():
    df = pd.DataFrame()
    fig = make_diet_heatmap(df)
    assert isinstance(fig, go.Figure)


def test_make_spatial_map():
    ds = xr.Dataset({
        "biomass": xr.DataArray(
            np.random.rand(3, 5, 5),
            dims=["time", "lat", "lon"],
            coords={
                "time": range(3),
                "lat": np.linspace(43, 48, 5),
                "lon": np.linspace(-5, 0, 5),
            },
        )
    })
    fig = make_spatial_map(ds, "biomass", time_idx=0)
    assert isinstance(fig, go.Figure)


def test_make_spatial_map_with_title():
    ds = xr.Dataset({
        "biomass": xr.DataArray(
            np.random.rand(1, 3, 3),
            dims=["time", "lat", "lon"],
            coords={"time": [0], "lat": [43, 44, 45], "lon": [-3, -2, -1]},
        )
    })
    fig = make_spatial_map(ds, "biomass", time_idx=0, title="Biomass t=0")
    assert fig.layout.title.text == "Biomass t=0"
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -v`
Expected: FAIL (ImportError)

**Step 3: Implement results.py**

Rewrite `ui/pages/results.py` with chart generation functions and wired server:

```python
"""Results visualization page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from shiny import ui, reactive, render
from shinywidgets import output_widget, render_plotly

from osmose.results import OsmoseResults


# --- Chart generation functions (testable without Shiny) ---

def make_timeseries_chart(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    species: str | None = None,
) -> go.Figure:
    """Create a time series line chart from OSMOSE output."""
    if df.empty:
        return go.Figure().update_layout(title=title)
    if species and "species" in df.columns:
        df = df[df["species"] == species]
    if df.empty:
        return go.Figure().update_layout(title=title)
    fig = px.line(df, x="time", y=value_col, color="species", title=title)
    fig.update_layout(template="plotly_dark")
    return fig


def make_diet_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a diet composition heatmap."""
    if df.empty:
        return go.Figure().update_layout(title="Diet Composition")
    prey_cols = [c for c in df.columns if c.startswith("prey_")]
    if not prey_cols:
        return go.Figure().update_layout(title="Diet Composition (no prey data)")
    # Average over time
    species_list = df["species"].unique() if "species" in df.columns else ["unknown"]
    matrix = df.groupby("species")[prey_cols].mean() if "species" in df.columns else df[prey_cols].mean().to_frame().T
    prey_names = [c.replace("prey_", "") for c in prey_cols]
    fig = px.imshow(
        matrix.values,
        x=prey_names,
        y=list(matrix.index),
        title="Diet Composition",
        color_continuous_scale="YlOrRd",
        labels=dict(x="Prey", y="Predator", color="Proportion"),
    )
    fig.update_layout(template="plotly_dark")
    return fig


def make_spatial_map(
    ds: xr.Dataset,
    var_name: str,
    time_idx: int = 0,
    title: str | None = None,
) -> go.Figure:
    """Create a spatial heatmap from NetCDF data."""
    data = ds[var_name].isel(time=time_idx).values
    lat = ds["lat"].values
    lon = ds["lon"].values
    fig = px.imshow(
        data,
        x=lon,
        y=lat,
        origin="lower",
        color_continuous_scale="Viridis",
        labels=dict(x="Longitude", y="Latitude", color=var_name),
        title=title or f"{var_name} (t={time_idx})",
    )
    fig.update_layout(template="plotly_dark")
    return fig


# --- UI ---

def results_ui():
    return ui.page_fluid(
        ui.layout_columns(
            ui.card(
                ui.card_header("Output Controls"),
                ui.input_text("output_dir", "Output directory", value="output/"),
                ui.input_action_button(
                    "btn_load_results", "Load Results", class_="btn-primary w-100"
                ),
                ui.hr(),
                ui.input_select(
                    "result_species", "Species filter",
                    choices={"all": "All species"}, selected="all",
                ),
                ui.input_select(
                    "result_type", "Output type",
                    choices={
                        "biomass": "Biomass", "abundance": "Abundance",
                        "yield": "Yield", "mortality": "Mortality",
                        "diet": "Diet Matrix", "trophic": "Trophic Level",
                    },
                    selected="biomass",
                ),
            ),
            ui.card(
                ui.card_header("Time Series"),
                output_widget("results_chart"),
            ),
            col_widths=[3, 9],
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Diet Composition Matrix"),
                output_widget("diet_chart"),
            ),
            ui.card(
                ui.card_header("Spatial Distribution"),
                ui.input_slider("spatial_time", "Time step", min=0, max=0, value=0),
                output_widget("spatial_chart"),
            ),
            col_widths=[6, 6],
        ),
    )


def results_server(input, output, session, state):
    results_obj = reactive.value(None)
    loaded_species = reactive.value([])

    @reactive.effect
    @reactive.event(input.btn_load_results)
    def load_results():
        out_dir = state.output_dir.get()
        if out_dir is None:
            out_dir = Path(input.output_dir())
        if not Path(out_dir).exists():
            return
        r = OsmoseResults(out_dir)
        results_obj.set(r)
        # Discover species
        bio_df = r.biomass()
        sp_list = sorted(bio_df["species"].unique().tolist()) if not bio_df.empty else []
        loaded_species.set(sp_list)
        choices = {"all": "All species"}
        for sp in sp_list:
            choices[sp] = sp
        ui.update_select("result_species", choices=choices)

    @render_plotly
    def results_chart():
        r = results_obj.get()
        if r is None:
            return go.Figure().update_layout(
                title="Load results to view charts",
                template="plotly_dark",
            )
        rt = input.result_type()
        sp = input.result_species()
        species_filter = None if sp == "all" else sp

        if rt == "biomass":
            return make_timeseries_chart(r.biomass(), "biomass", "Biomass", species_filter)
        elif rt == "abundance":
            return make_timeseries_chart(r.abundance(), "abundance", "Abundance", species_filter)
        elif rt == "yield":
            return make_timeseries_chart(r.yield_biomass(), "yield", "Yield", species_filter)
        elif rt == "mortality":
            return make_timeseries_chart(r.mortality(), "mortality", "Mortality", species_filter)
        elif rt == "trophic":
            return make_timeseries_chart(r.mean_trophic_level(), "meanTL", "Mean Trophic Level", species_filter)
        return go.Figure()

    @render_plotly
    def diet_chart():
        r = results_obj.get()
        if r is None:
            return go.Figure().update_layout(title="Diet Composition", template="plotly_dark")
        return make_diet_heatmap(r.diet_matrix())

    @render_plotly
    def spatial_chart():
        r = results_obj.get()
        if r is None:
            return go.Figure().update_layout(title="Spatial Distribution", template="plotly_dark")
        nc_files = [f for f in r.list_outputs() if f.endswith(".nc")]
        if not nc_files:
            return go.Figure().update_layout(title="No spatial data", template="plotly_dark")
        ds = r.read_netcdf(nc_files[0])
        bio_var = next((v for v in ds.data_vars), None)
        if bio_var is None:
            return go.Figure()
        n_times = ds.dims.get("time", 1)
        ui.update_slider("spatial_time", max=max(0, n_times - 1))
        t = input.spatial_time()
        return make_spatial_map(ds, bio_var, time_idx=min(t, n_times - 1))
```

Note: import `from pathlib import Path` at the top.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -v`
Expected: All 7 tests PASS

**Step 5: Run full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add ui/pages/results.py tests/test_ui_results.py
git commit -m "feat: wire Results page with plotly charts"
```

---

### Task 6: Wire the Scenarios page

**Files:**
- Modify: `ui/pages/scenarios.py`
- Test: `tests/test_ui_scenarios.py`

**Step 1: Write tests**

Create `tests/test_ui_scenarios.py`:

```python
"""Tests for scenarios page logic."""

from pathlib import Path

from osmose.scenarios import Scenario, ScenarioManager


def test_scenario_save_and_list(tmp_path):
    mgr = ScenarioManager(tmp_path)
    s = Scenario(name="test1", config={"a": "1"}, tags=["v1"])
    mgr.save(s)
    scenarios = mgr.list_scenarios()
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "test1"


def test_scenario_load_config(tmp_path):
    mgr = ScenarioManager(tmp_path)
    mgr.save(Scenario(name="s1", config={"x": "42"}))
    loaded = mgr.load("s1")
    assert loaded.config == {"x": "42"}


def test_scenario_fork(tmp_path):
    mgr = ScenarioManager(tmp_path)
    mgr.save(Scenario(name="base", config={"a": "1"}))
    forked = mgr.fork("base", "derived")
    assert forked.config == {"a": "1"}
    assert forked.parent_scenario == "base"


def test_scenario_delete(tmp_path):
    mgr = ScenarioManager(tmp_path)
    mgr.save(Scenario(name="del_me", config={}))
    assert len(mgr.list_scenarios()) == 1
    mgr.delete("del_me")
    assert len(mgr.list_scenarios()) == 0


def test_scenario_compare_shows_diffs(tmp_path):
    mgr = ScenarioManager(tmp_path)
    mgr.save(Scenario(name="a", config={"x": "1", "y": "2"}))
    mgr.save(Scenario(name="b", config={"x": "1", "y": "9"}))
    diffs = mgr.compare("a", "b")
    assert len(diffs) == 1
    assert diffs[0].key == "y"
```

**Step 2: Run tests (these should already pass since they test existing ScenarioManager)**

Run: `.venv/bin/python -m pytest tests/test_ui_scenarios.py -v`
Expected: All 5 tests PASS

**Step 3: Implement scenarios.py server wiring**

Rewrite `ui/pages/scenarios.py` with wired server. Keep `scenarios_ui()` structure similar, add event handlers:

```python
"""Scenario management page."""

from __future__ import annotations

from shiny import ui, reactive, render

from osmose.scenarios import Scenario, ScenarioManager


def scenarios_ui():
    # ... keep existing UI unchanged ...


def scenarios_server(input, output, session, state):
    mgr = ScenarioManager(state.scenarios_dir)
    refresh_trigger = reactive.value(0)

    @render.ui
    def scenario_list():
        refresh_trigger.get()  # dependency to re-render on changes
        scenarios = mgr.list_scenarios()
        if not scenarios:
            return ui.div("No scenarios saved yet.", style="padding: 20px; text-align: center; color: #999;")
        items = []
        for s in scenarios:
            tags_str = ", ".join(s.get("tags", []))
            items.append(
                ui.div(
                    ui.input_radio_buttons(
                        "selected_scenario", None,
                        choices={s["name"]: s["name"] for s in scenarios},
                        inline=True,
                    ),
                )
            )
            break  # Only render radio buttons once
        return ui.div(
            ui.input_radio_buttons(
                "selected_scenario", "Select scenario:",
                choices={s["name"]: f"{s['name']} ({s.get('modified_at', '')[:10]})" for s in scenarios},
            ),
        )

    @reactive.effect
    @reactive.event(input.btn_save_scenario)
    def save_scenario():
        name = input.scenario_name()
        if not name:
            return
        s = Scenario(
            name=name,
            description=input.scenario_desc() or "",
            config=dict(state.config.get()),
            tags=[t.strip() for t in (input.scenario_tags() or "").split(",") if t.strip()],
        )
        mgr.save(s)
        refresh_trigger.set(refresh_trigger.get() + 1)
        _refresh_compare_choices(mgr)

    @reactive.effect
    @reactive.event(input.btn_load_scenario)
    def load_scenario():
        name = input.selected_scenario()
        if not name:
            return
        scenario = mgr.load(name)
        state.config.set(scenario.config)

    @reactive.effect
    @reactive.event(input.btn_fork_scenario)
    def fork_scenario():
        name = input.selected_scenario()
        if not name:
            return
        new_name = f"{name}_fork"
        mgr.fork(name, new_name)
        refresh_trigger.set(refresh_trigger.get() + 1)

    @reactive.effect
    @reactive.event(input.btn_delete_scenario)
    def delete_scenario():
        name = input.selected_scenario()
        if not name:
            return
        mgr.delete(name)
        refresh_trigger.set(refresh_trigger.get() + 1)

    @render.ui
    def compare_results():
        return ui.div("Select two scenarios and click Compare.", style="padding: 20px; text-align: center; color: #999;")

    @reactive.effect
    @reactive.event(input.btn_compare)
    def handle_compare():
        pass  # Implemented via render below

    @render.ui
    def compare_results():
        if input.btn_compare() == 0:
            return ui.div("Select two scenarios and click Compare.", style="padding: 20px; text-align: center; color: #999;")
        a = input.compare_a()
        b = input.compare_b()
        if not a or not b:
            return ui.div("Select two scenarios.", style="color: #f39c12;")
        diffs = mgr.compare(a, b)
        if not diffs:
            return ui.div("Scenarios are identical.", style="color: #2ecc71;")
        rows = []
        for d in diffs:
            rows.append(ui.tags.tr(
                ui.tags.td(d.key),
                ui.tags.td(d.value_a or "-"),
                ui.tags.td(d.value_b or "-"),
            ))
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(
                ui.tags.th("Parameter"),
                ui.tags.th(a),
                ui.tags.th(b),
            )),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
        )

    def _refresh_compare_choices(mgr):
        scenarios = mgr.list_scenarios()
        choices = {s["name"]: s["name"] for s in scenarios}
        ui.update_select("compare_a", choices=choices)
        ui.update_select("compare_b", choices=choices)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ui_scenarios.py tests/test_scenarios.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ui/pages/scenarios.py tests/test_ui_scenarios.py
git commit -m "feat: wire Scenarios page to ScenarioManager"
```

---

### Task 7: Wire the Calibration page

**Files:**
- Modify: `ui/pages/calibration.py`
- Test: `tests/test_ui_calibration.py`

**Step 1: Write tests**

Create `tests/test_ui_calibration.py`:

```python
"""Tests for calibration page helper functions."""

import numpy as np
import plotly.graph_objects as go

from ui.pages.calibration import (
    get_calibratable_params,
    make_convergence_chart,
    make_pareto_chart,
    make_sensitivity_chart,
)


def test_get_calibratable_params():
    from ui.state import REGISTRY
    params = get_calibratable_params(REGISTRY, n_species=3)
    assert len(params) > 0
    # Should include growth K for all 3 species
    keys = [p["key"] for p in params]
    assert "species.k.sp0" in keys
    assert "species.k.sp1" in keys
    assert "species.k.sp2" in keys


def test_make_convergence_chart():
    history = [10.0, 5.0, 3.0, 2.0, 1.5]
    fig = make_convergence_chart(history)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_make_convergence_chart_empty():
    fig = make_convergence_chart([])
    assert isinstance(fig, go.Figure)


def test_make_pareto_chart():
    F = np.array([[1.0, 2.0], [0.5, 3.0], [0.8, 1.5]])
    fig = make_pareto_chart(F, ["Biomass RMSE", "Diet Distance"])
    assert isinstance(fig, go.Figure)


def test_make_sensitivity_chart():
    result = {
        "param_names": ["k", "linf", "mort"],
        "S1": np.array([0.5, 0.3, 0.2]),
        "ST": np.array([0.6, 0.4, 0.3]),
    }
    fig = make_sensitivity_chart(result)
    assert isinstance(fig, go.Figure)
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration.py -v`
Expected: FAIL (ImportError)

**Step 3: Implement calibration.py**

Rewrite `ui/pages/calibration.py` with helper functions and wired server:

```python
"""Calibration page - multi-objective optimization and sensitivity analysis."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from shiny import ui, reactive, render
from shinywidgets import output_widget, render_plotly

from osmose.schema.base import ParamType
from osmose.schema.registry import ParameterRegistry


# --- Testable helper functions ---

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
            params.append({
                "key": key,
                "label": f"{field.description} (sp{i})",
                "category": field.category,
                "lower": field.min_val,
                "upper": field.max_val,
            })
    return params


def make_convergence_chart(history: list[float]) -> go.Figure:
    """Line chart of best objective value per generation."""
    if not history:
        return go.Figure().update_layout(title="Convergence", template="plotly_dark")
    fig = px.line(x=list(range(len(history))), y=history, title="Convergence")
    fig.update_layout(xaxis_title="Generation", yaxis_title="Best Objective", template="plotly_dark")
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


# --- UI (mostly unchanged) ---

def calibration_ui():
    # ... keep existing calibration_ui() but replace placeholders with output_widget ...
    # Change:
    #   - "Pareto Front" tab: output_widget("pareto_chart")
    #   - "Progress" tab: output_widget("convergence_chart")
    #   - "Sensitivity" tab: output_widget("sensitivity_chart")
    # ... (detailed in implementation) ...


def calibration_server(input, output, session, state):
    cal_history = reactive.value([])
    cal_F = reactive.value(None)
    cal_X = reactive.value(None)
    sensitivity_result = reactive.value(None)

    @render.ui
    def free_param_selector():
        n_species = int(state.config.get().get("simulation.nspecies", "3"))
        params = get_calibratable_params(state.registry, n_species)
        checkboxes = []
        for p in params:
            input_id = f"cal_param_{p['key'].replace('.', '_')}"
            checkboxes.append(
                ui.input_checkbox(input_id, p["label"], value=False)
            )
        return ui.div(*checkboxes)

    @render.text
    def cal_status():
        hist = cal_history.get()
        if not hist:
            return "Ready. Configure parameters and objectives, then click Start."
        return f"Generation {len(hist)} — Best: {hist[-1]:.4f}"

    @render_plotly
    def convergence_chart():
        return make_convergence_chart(cal_history.get())

    @render_plotly
    def pareto_chart():
        F = cal_F.get()
        if F is None:
            return go.Figure().update_layout(title="Pareto Front (run calibration first)", template="plotly_dark")
        return make_pareto_chart(F, ["Biomass RMSE", "Diet Distance"])

    @render.ui
    def best_params_table():
        X = cal_X.get()
        F = cal_F.get()
        if X is None or F is None:
            return ui.div("Run calibration to see best parameters.", style="color: #999;")
        # Sort by sum of objectives
        order = np.argsort(F.sum(axis=1))[:10]  # Top 10
        rows = []
        for idx in order:
            cells = [ui.tags.td(f"{v:.4f}") for v in X[idx]]
            cells.append(ui.tags.td(f"{F[idx].sum():.4f}"))
            rows.append(ui.tags.tr(*cells))
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(
                *[ui.tags.th(f"Param {i}") for i in range(X.shape[1])],
                ui.tags.th("Total Obj"),
            )),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
        )

    @render_plotly
    def sensitivity_chart():
        result = sensitivity_result.get()
        if result is None:
            return go.Figure().update_layout(title="Sensitivity (click Run)", template="plotly_dark")
        return make_sensitivity_chart(result)

    # Button handlers for Start/Stop calibration and sensitivity analysis
    # will be wired here using @reactive.effect / @reactive.event
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add ui/pages/calibration.py tests/test_ui_calibration.py
git commit -m "feat: wire Calibration page with dynamic params and plotly charts"
```

---

### Task 8: Wire the Grid preview

**Files:**
- Modify: `ui/pages/grid.py`
- Test: `tests/test_ui_grid.py`

**Step 1: Write tests**

Create `tests/test_ui_grid.py`:

```python
"""Tests for grid page preview chart."""

import plotly.graph_objects as go

from ui.pages.grid import make_grid_preview


def test_grid_preview_with_coords():
    fig = make_grid_preview(
        ul_lat=48.0, ul_lon=-5.0,
        lr_lat=43.0, lr_lon=0.0,
    )
    assert isinstance(fig, go.Figure)
    # Should have a rectangle shape
    assert len(fig.data) > 0


def test_grid_preview_zero_coords():
    fig = make_grid_preview(ul_lat=0, ul_lon=0, lr_lat=0, lr_lon=0)
    assert isinstance(fig, go.Figure)


def test_grid_preview_with_dimensions():
    fig = make_grid_preview(
        ul_lat=48.0, ul_lon=-5.0,
        lr_lat=43.0, lr_lon=0.0,
        nx=30, ny=30,
    )
    assert isinstance(fig, go.Figure)
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_ui_grid.py -v`
Expected: FAIL

**Step 3: Implement grid preview**

Add `make_grid_preview()` function and wire it into `grid_server()`:

```python
def make_grid_preview(
    ul_lat: float, ul_lon: float,
    lr_lat: float, lr_lon: float,
    nx: int = 0, ny: int = 0,
) -> go.Figure:
    """Create a plotly figure showing the grid extent."""
    fig = go.Figure()
    # Draw rectangle for grid extent
    lats = [ul_lat, ul_lat, lr_lat, lr_lat, ul_lat]
    lons = [ul_lon, lr_lon, lr_lon, ul_lon, ul_lon]
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons,
        mode="lines",
        line=dict(width=3, color="#e67e22"),
        name="Grid extent",
    ))
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
```

Wire into `grid_server`:

```python
def grid_server(input, output, session, state):
    @render_plotly
    def grid_preview_placeholder():
        ul_lat = input.grid_upleft_lat() or 0
        ul_lon = input.grid_upleft_lon() or 0
        lr_lat = input.grid_lowright_lat() or 0
        lr_lon = input.grid_lowright_lon() or 0
        nx = input.grid_ncolumn() or 0
        ny = input.grid_nline() or 0
        return make_grid_preview(ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)
```

Update `grid_ui()` to use `output_widget("grid_preview_placeholder")` instead of `ui.output_ui(...)`.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ui_grid.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add ui/pages/grid.py tests/test_ui_grid.py
git commit -m "feat: add grid preview map with plotly"
```

---

### Task 9: Run full test suite, lint, and verify app

**Step 1: Run all tests**

Run: `.venv/bin/python -m pytest --cov=osmose --cov=ui -v`
Expected: All tests PASS, high coverage

**Step 2: Lint**

Run: `.venv/bin/ruff check osmose/ ui/ tests/`
Expected: No errors

**Step 3: Format**

Run: `.venv/bin/ruff format osmose/ ui/ tests/`

**Step 4: Start the app and verify all tabs**

Run: `.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000`

Verify:
- Setup tab: inputs render, species panels work
- Grid tab: preview map shows when coordinates are entered
- Run tab: buttons show correct states (error if no JAR)
- Results tab: plotly chart placeholders render (empty chart with "Load results")
- Calibration tab: dynamic param checkboxes render for 3 species
- Scenarios tab: save/load/fork/delete/compare buttons work
- Advanced tab: table renders

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: lint and format Phase 2 UI wiring"
```

---

### Task 10: Push and verify CI

**Step 1: Push**

Run: `git push origin master`

**Step 2: Check CI**

Run: `gh run list --limit 1`
Expected: CI passes (lint + test)

**Step 3: Update memory**

Update MEMORY.md with new test counts, coverage, and commit history.

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | Add shinywidgets dep | pyproject.toml | - |
| 2 | AppState module | ui/state.py | test_state.py (9) |
| 3 | Wire AppState into app.py | app.py + 10 pages | existing suite |
| 4 | Wire Run page | ui/pages/run.py | test_ui_run.py (6) |
| 5 | Wire Results + plotly | ui/pages/results.py | test_ui_results.py (7) |
| 6 | Wire Scenarios page | ui/pages/scenarios.py | test_ui_scenarios.py (5) |
| 7 | Wire Calibration page | ui/pages/calibration.py | test_ui_calibration.py (5) |
| 8 | Grid preview map | ui/pages/grid.py | test_ui_grid.py (3) |
| 9 | Full test + lint + verify | all | all |
| 10 | Push + CI | - | CI |

**Total new tests: ~35**
**Estimated test count after: ~190**
