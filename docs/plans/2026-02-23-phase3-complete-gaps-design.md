# Phase 3: Complete Phase 2 Gaps

**Date:** 2026-02-23
**Status:** Approved
**Scope:** Wire remaining input syncing (5 config pages), calibration Start/Stop handlers, Advanced import/export, and fix movement page species count.

## Context

Phase 2 delivered shared `AppState`, fully wired Run/Results/Scenarios/Grid pages, and chart helper functions. However, 5 config pages don't sync inputs back to `state.config`, calibration Start/Stop buttons have no handlers, and Advanced import/export buttons are no-ops. This phase closes all remaining gaps.

## 1. Input Syncing Architecture

### Problem
Setup, Grid, Forcing, Fishing, and Movement pages render input widgets via `render_field()` but never write changed values back to `state.config`. The Run page always uses defaults.

### Solution: Hybrid Auto-Sync

**Global fields (non-indexed):** Add a utility function `sync_global_inputs()` in `ui/state.py`:
- Iterates all non-indexed fields in the registry
- Computes the input ID from `key_pattern.replace(".", "_")`
- Reads the current input value
- Calls `state.update_config(key, str(value))`

**Indexed fields (species, fisheries, MPAs, resources):** Each page server adds explicit `@reactive.effect` handlers that loop over the current count and sync each indexed field.

**Trigger:** Auto-sync on input change (no manual save button). Each page server's sync effect lists its input dependencies reactively.

### Pages Affected

| Page | Global Fields | Indexed Fields |
|------|--------------|----------------|
| Setup | simulation.* (11) | species.* per n_species |
| Grid | grid.* (~15) | None |
| Forcing | Global LTL | ltl.* per n_resources |
| Fishing | Global fishing | fishery.* per n_fisheries, mpa.* per n_mpas |
| Movement | Global movement | movement.* per n_species (dynamic) |

## 2. Calibration Start/Stop Handlers

### Problem
Start/Stop buttons and Sensitivity "Run" button exist but have no `@reactive.event` handlers. Backend (`OsmoseCalibrationProblem`, `SensitivityAnalyzer`) is fully implemented and tested.

### Design

**JAR path:** Add `jar_path: reactive.Value[str]` to `AppState`. Both Run and Calibration pages read/write it. Run page sets it from its JAR path input. Calibration page reads it.

**Start Calibration:**
1. Collect selected free params from dynamic checkboxes
2. Validate: >= 1 param selected, >= 1 objective file uploaded, JAR path set
3. Write `state.config` to temp dir via `OsmoseConfigWriter`
4. Build `FreeParameter` list with bounds from registry
5. Build objective functions from uploaded observed data
6. Create `OsmoseCalibrationProblem`
7. Run pymoo NSGA-II in `threading.Thread` (pymoo is synchronous)
8. Callback updates `cal_history`, `cal_F`, `cal_X` reactives per generation
9. Charts update reactively

**Stop:** Set `cancel_flag` checked between generations.

**Sensitivity:**
1. Collect selected params
2. Create `SensitivityAnalyzer` with names/bounds
3. Generate Sobol samples in background thread
4. Evaluate OSMOSE per sample
5. `analyzer.analyze(Y)` → update `sensitivity_result` reactive

## 3. Advanced Page Import/Export

### Import
1. User uploads `.csv`/`.properties` via `input.import_config()`
2. `OsmoseConfigReader().read(path)` parses the file
3. Loaded config merged into `state.config` (overwriting matching keys)
4. All pages reactively update

### Export
1. `@render.download` on `export_config` button
2. Writes `state.config` to temp dir via `OsmoseConfigWriter`
3. Returns master file as download

## 4. Movement Page Fix

Replace hardcoded `range(3)` with dynamic species count from `state.config.get().get("simulation.nspecies", "3")`.

## Files to Modify

| File | Change |
|------|--------|
| `ui/state.py` | Add `jar_path` reactive, `sync_global_inputs()` utility |
| `ui/pages/setup.py` | Add sync handlers for simulation + species fields |
| `ui/pages/grid.py` | Add sync for grid fields |
| `ui/pages/forcing.py` | Add sync for LTL + temp fields |
| `ui/pages/fishing.py` | Add sync for fishery + MPA fields |
| `ui/pages/movement.py` | Fix species count, add sync |
| `ui/pages/calibration.py` | Add Start/Stop/Sensitivity handlers |
| `ui/pages/advanced.py` | Add import/export handlers |
| `ui/pages/run.py` | Read JAR path from AppState |

## New Test Files

| File | Tests |
|------|-------|
| `tests/test_sync.py` | Input sync utility, global + indexed sync |
| `tests/test_ui_calibration_handlers.py` | Calibration Start/Stop/Sensitivity |
| `tests/test_ui_advanced_io.py` | Import/Export config |

## Estimated Scope

~310 lines of new code, 9 files modified, ~60 lines of new tests.
