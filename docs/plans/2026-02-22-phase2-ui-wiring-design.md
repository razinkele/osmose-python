# Phase 2: Full UI Wiring + Visualization

**Date:** 2026-02-22
**Status:** Approved
**Scope:** Wire all placeholder UI pages to backend modules, add shared state management, implement full plotly visualization suite.

## Context

Phase 1 delivered a solid foundation: schema-driven parameter system (181 params), config I/O, async runner, results reader, calibration engine, scenario management, and a 10-tab Shiny UI. However, the UI is ~60% complete — page layouts exist but most button handlers and visualizations are placeholder-only. This phase connects everything.

## Architecture Change: Shared AppState

### Problem
Each page has isolated `reactive.value()` objects. Config entered on Setup doesn't flow to Run. Results page can't find output files. Scenarios can't save the current config.

### Solution
New file `ui/state.py` with an `AppState` class:

```python
class AppState:
    config: reactive.Value[dict[str, str]]      # Full OSMOSE key→value config
    output_dir: reactive.Value[Path | None]      # Last run output directory
    run_result: reactive.Value[RunResult | None]  # Last run result
    scenarios_dir: Path                           # data/scenarios/
    runner: OsmoseRunner | None                   # Active runner instance
```

- Created once in `app.py`'s `server()` function
- Passed to all 10 `*_server(input, output, session, state)` functions
- Config pages (Setup, Grid, Forcing, Fishing, Movement) write inputs back to `state.config`
- Consumer pages (Run, Results, Calibration, Scenarios) read from `state.config`

## Run Page

### Current State
Buttons exist but have no handlers. Console shows static text.

### Design
- **Start Run** handler:
  1. Validate JAR path exists
  2. Write `state.config` to temp dir via `OsmoseConfigWriter`
  3. Parse overrides textarea (one `key=value` per line)
  4. Call `OsmoseRunner.run()` async with `on_progress` callback
  5. Callback appends lines to `run_log` reactive, updating console in real-time
  6. On completion, set `state.output_dir` and `state.run_result`
- **Cancel** handler: calls `runner.cancel()`
- **Status** lifecycle: Idle → Writing config... → Running → Complete / Failed / Cancelled
- **Error handling:** JAR not found → inline error. Java failure → stderr in console (red).

## Results Page — Full Visualization

### Current State
All 3 output areas are placeholders.

### Design
- **Load Results** button: creates `OsmoseResults(state.output_dir)`, stores in state
- **Species filter** auto-populates from discovered species in output files
- **Output type** selector switches displayed chart

### Charts (all plotly via shinywidgets)

| Chart | Plotly Type | Data Source |
|-------|------------|-------------|
| Biomass time series | `px.line` (one trace/species) | `results.biomass()` |
| Abundance time series | `px.line` | `results.abundance()` |
| Yield time series | `px.line` | `results.yield_biomass()` |
| Mortality | `px.area` (stacked by cause) | `results.mortality()` |
| Trophic level | `px.line` | `results.mean_trophic_level()` |
| Diet composition | `px.imshow` heatmap (predator × prey) | `results.diet_matrix()` |
| Spatial biomass | `px.density_mapbox` or `px.imshow` on lat/lon | `results.spatial_biomass()` |

- Species filter applies to all time series charts
- Spatial map includes a time slider for animation
- Empty data → graceful "No data available" message (not crash)

## Calibration Page

### Current State
Hardcoded 5-parameter checkbox list. All buttons do nothing. All result tabs are placeholders.

### Design

**Free Parameter Selector:**
- Dynamic: reads all numeric species-indexed fields from registry
- Grouped by category (growth, predation, mortality, initialization)
- Generates checkboxes per species (e.g., "Growth K (sp0)", "Growth K (sp1)")

**Start Calibration:**
1. Validate: at least 1 free param + 1 objective file selected
2. Write current config to temp directory
3. Create `OsmoseCalibrationProblem` with selected params + schema bounds
4. Run pymoo NSGA-II or GP surrogate (based on algorithm dropdown) in background thread
5. Update progress reactive per generation

**Stop:** Cancels optimization loop.

**Result Tabs:**
- **Progress:** Generation counter + convergence line chart (best objective/generation)
- **Pareto Front:** Scatter plot of objective 1 vs objective 2
- **Best Parameters:** Table of top-N Pareto-optimal parameter sets
- **Sensitivity:** "Run" button → SALib Sobol analysis → bar chart of S1/ST indices

## Scenarios Page

### Current State
Save/Load/Fork/Delete/Compare buttons all do nothing.

### Design
- `ScenarioManager` initialized with `data/scenarios/`
- **Save:** reads `state.config`, calls `ScenarioManager.save()` with name/description/tags
- **Scenario List:** auto-refreshes after mutations, shows name + timestamp + tags
- **Selection:** radio buttons or clickable list
- **Load:** loads scenario config into `state.config` (all pages update reactively)
- **Fork:** creates copy with new name via modal dialog
- **Delete:** confirms via modal, then deletes
- **Compare:** runs `ScenarioManager.compare()`, shows diff table (key, value_A, value_B) with highlighted differences

## Grid Preview

### Current State
Static placeholder.

### Design
- When lat/lon coordinates are set, render a plotly figure showing the grid extent as a rectangle on a world map (scatter_mapbox or choropleth)
- If mask file uploaded, overlay active cells
- Reactive: updates when coordinates change

## Testing Strategy

- Unit tests for `AppState` initialization and config flow
- Mock-based tests for Run page handler (mock `OsmoseRunner.run()`)
- Mock-based tests for Results page (mock `OsmoseResults` methods with fixture DataFrames)
- Mock-based tests for Scenarios page (mock `ScenarioManager` filesystem ops)
- Integration test: Setup → Run → Results flow with fake JAR
- All plotly chart functions tested with sample data (verify figure is created, correct traces)

## Files to Create/Modify

### New Files
- `ui/state.py` — AppState class

### Modified Files
- `app.py` — create AppState, pass to all servers
- `ui/pages/run.py` — wire Start/Cancel buttons to OsmoseRunner
- `ui/pages/results.py` — wire Load Results, add 7 plotly charts
- `ui/pages/calibration.py` — wire Start/Stop, dynamic param selector, 4 result tabs
- `ui/pages/scenarios.py` — wire Save/Load/Fork/Delete/Compare to ScenarioManager
- `ui/pages/grid.py` — add grid preview map
- `ui/pages/setup.py` — write inputs back to state.config
- `ui/pages/forcing.py` — write inputs back to state.config
- `ui/pages/fishing.py` — write inputs back to state.config
- `ui/pages/movement.py` — write inputs back to state.config

### New Test Files
- `tests/test_state.py`
- `tests/test_ui_run.py`
- `tests/test_ui_results.py`
- `tests/test_ui_calibration.py`
- `tests/test_ui_scenarios.py`
- `tests/test_ui_grid.py`

## Dependencies
- `shinywidgets` — already in pyproject.toml for plotly integration
- No new external dependencies required
