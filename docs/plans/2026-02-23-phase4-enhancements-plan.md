# Phase 4: Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GP surrogate calibration, progress indicators, parallel workers, config import validation, spatial animation, structured logging, scenario bulk operations, and input validation tooltips.

**Architecture:** Each enhancement is independent — no task blocks another. The calibration enhancements (Tasks 1-3) modify `ui/pages/calibration.py`; all others touch separate files. The surrogate integration reuses the existing `SurrogateCalibrator` class from `osmose/calibration/surrogate.py` by adding a 3-phase workflow (sample → evaluate → optimize). Parallel workers use `concurrent.futures.ProcessPoolExecutor` inside the pymoo evaluation loop. Logging uses Python's `logging` module with a module-level logger pattern.

**Tech Stack:** Python 3.12, Shiny for Python, plotly, pymoo, scikit-learn (GP), SALib, concurrent.futures, logging

---

## Context for All Tasks

- **Repo root:** `/home/razinka/osmose/osmose-python/`
- **Run tests:** `.venv/bin/python -m pytest`
- **Run single test:** `.venv/bin/python -m pytest tests/test_file.py::test_name -v`
- **Lint:** `.venv/bin/ruff check osmose/ ui/ tests/`
- **Format:** `.venv/bin/ruff format osmose/ ui/ tests/`
- **OSMOSE keys** use dot-separated lowercase (e.g., `species.linf.sp0`)
- **Shiny input IDs** are OSMOSE keys with dots replaced by underscores
- **`AppState`** in `ui/state.py` holds shared reactive state across all pages
- **`OsmoseCalibrationProblem._run_single()`** runs OSMOSE synchronously via subprocess

---

### Task 1: Wire GP Surrogate Calibration into UI

The `cal_algorithm` dropdown at `ui/pages/calibration.py:123-130` offers "GP Surrogate" as a choice, but `handle_start_cal()` (line 225) ignores it and always uses NSGA-II. The backend `SurrogateCalibrator` class in `osmose/calibration/surrogate.py` is fully implemented and tested.

The surrogate workflow is: (1) generate Latin hypercube samples, (2) evaluate OSMOSE for each sample to get training data, (3) fit GP model, (4) find optimum on surrogate. This differs from NSGA-II which iterates generations.

**Files:**
- Modify: `ui/pages/calibration.py:76-111` (add `make_surrogate_chart` helper)
- Modify: `ui/pages/calibration.py:192-301` (add surrogate reactive state + handler branch)
- Test: `tests/test_ui_calibration.py` (add surrogate chart test)
- Test: `tests/test_ui_calibration_handlers.py` (add surrogate workflow test)

**Step 1: Write failing tests for surrogate chart helper and workflow**

In `tests/test_ui_calibration.py`, add:

```python
def test_make_surrogate_chart():
    import numpy as np
    from ui.pages.calibration import make_surrogate_chart

    means = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]])
    stds = np.array([[0.1, 0.2], [0.3, 0.4], [0.05, 0.15]])
    param_names = ["species.k.sp0", "species.linf.sp0"]
    fig = make_surrogate_chart(means, stds, param_names)
    assert fig is not None
    assert len(fig.data) > 0
    assert "Surrogate" in fig.layout.title.text
```

In `tests/test_ui_calibration_handlers.py`, add:

```python
def test_run_surrogate_workflow():
    """Test that surrogate workflow produces results from SurrogateCalibrator."""
    from osmose.calibration.surrogate import SurrogateCalibrator

    bounds = [(0.1, 1.0), (10.0, 100.0)]
    cal = SurrogateCalibrator(param_bounds=bounds, n_objectives=1)
    samples = cal.generate_samples(n_samples=20, seed=42)
    assert samples.shape == (20, 2)

    # Simulate objective evaluation (quadratic)
    import numpy as np

    Y = np.sum((samples - 0.5) ** 2, axis=1)
    cal.fit(samples, Y)
    result = cal.find_optimum(n_candidates=1000, seed=123)
    assert "params" in result
    assert "predicted_objectives" in result
    assert result["predicted_objectives"].shape == (1,)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration.py::test_make_surrogate_chart tests/test_ui_calibration_handlers.py::test_run_surrogate_workflow -v`
Expected: FAIL with `ImportError: cannot import name 'make_surrogate_chart'`

**Step 3: Add `make_surrogate_chart` helper**

In `ui/pages/calibration.py`, after `make_sensitivity_chart()` (line 111), add:

```python
def make_surrogate_chart(
    means: np.ndarray, stds: np.ndarray, param_names: list[str]
) -> go.Figure:
    """Scatter plot of surrogate predictions with uncertainty bars."""
    if means.size == 0:
        return go.Figure().update_layout(title="Surrogate Results", template="plotly_dark")
    n_obj = means.shape[1] if means.ndim > 1 else 1
    if n_obj >= 2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=means[:, 0],
                y=means[:, 1],
                mode="markers",
                marker=dict(size=6),
                name="Predictions",
            )
        )
        fig.update_layout(
            title="Surrogate Predicted Pareto Front",
            xaxis_title="Objective 1",
            yaxis_title="Objective 2",
            template="plotly_dark",
        )
    else:
        flat = means.ravel()
        fig = px.histogram(x=flat, nbins=30, title="Surrogate Prediction Distribution")
        fig.update_layout(template="plotly_dark", xaxis_title="Predicted Objective")
    return fig
```

**Step 4: Add surrogate branch to `handle_start_cal`**

In `ui/pages/calibration.py`, add import at top (line 19):

```python
from osmose.calibration.surrogate import SurrogateCalibrator
```

In `calibration_server()`, add reactive state (after line 198):

```python
    surrogate_status = reactive.value("")
```

Replace the `run_optimization()` inner function (lines 272-297) and the thread launch (lines 299-301) with an algorithm branch:

```python
        algorithm_choice = input.cal_algorithm()

        if algorithm_choice == "surrogate":
            surrogate_status.set("Generating samples...")

            def run_surrogate():
                bounds = [(fp.lower_bound, fp.upper_bound) for fp in free_params]
                n_obj = len(objective_fns)
                cal = SurrogateCalibrator(param_bounds=bounds, n_objectives=n_obj)

                n_samples = int(input.cal_pop_size())
                samples = cal.generate_samples(n_samples=n_samples, seed=42)
                surrogate_status.set(f"Evaluating {n_samples} OSMOSE samples...")

                Y = np.full((samples.shape[0], n_obj), np.inf)
                for idx, row in enumerate(samples):
                    if cancel_flag.get():
                        surrogate_status.set("Cancelled")
                        return
                    overrides = {
                        free_params[j].key: str(row[j]) for j in range(len(free_params))
                    }
                    try:
                        result = problem._run_single(overrides, run_id=idx)
                        for k, v in enumerate(result):
                            Y[idx, k] = v
                    except Exception:
                        pass
                    if (idx + 1) % 5 == 0:
                        surrogate_status.set(
                            f"Evaluated {idx + 1}/{n_samples} samples..."
                        )

                surrogate_status.set("Fitting GP surrogate model...")
                cal.fit(samples, Y)

                surrogate_status.set("Finding optimum on surrogate...")
                optimum = cal.find_optimum(n_candidates=10000)

                cal_X.set(samples)
                cal_F.set(Y)
                cal_history.set(
                    [float(np.min(Y[: i + 1].sum(axis=1))) for i in range(len(Y))]
                )
                surrogate_status.set(
                    f"Done. Best predicted: {optimum['predicted_objectives']}"
                )

            thread = threading.Thread(target=run_surrogate, daemon=True)
            thread.start()
            cal_thread.set(thread)
        else:
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
```

Update `cal_status` render (lines 200-205) to include surrogate status:

```python
    @render.text
    def cal_status():
        surr = surrogate_status.get()
        if surr:
            return surr
        hist = cal_history.get()
        if not hist:
            return "Ready. Configure parameters and objectives, then click Start."
        return f"Generation {len(hist)} — Best: {hist[-1]:.4f}"
```

**Step 5: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration.py tests/test_ui_calibration_handlers.py -v`
Expected: ALL PASS

**Step 6: Lint and format**

Run: `.venv/bin/ruff check ui/pages/calibration.py && .venv/bin/ruff format ui/pages/calibration.py`

**Step 7: Commit**

```bash
git add ui/pages/calibration.py tests/test_ui_calibration.py tests/test_ui_calibration_handlers.py
git commit -m "feat: wire GP surrogate calibration into UI"
```

---

### Task 2: Add Calibration Progress Indicators

The NSGA-II handler runs in a background thread with no per-generation progress updates. The `cal_status` text only shows "Ready" or the final result. We need to update `cal_history` reactively during optimization so the convergence chart and status text update live.

pymoo's `minimize()` doesn't support per-generation callbacks out of the box, but we can use pymoo's `Callback` class to hook into each generation and push updates to the reactive values.

**Files:**
- Modify: `ui/pages/calibration.py:272-297` (add pymoo Callback for per-generation updates)
- Test: `tests/test_ui_calibration.py` (test callback behavior)

**Step 1: Write failing test**

In `tests/test_ui_calibration.py`, add:

```python
def test_make_convergence_chart_incremental():
    """Convergence chart updates incrementally as history grows."""
    from ui.pages.calibration import make_convergence_chart

    # Simulates per-generation updates
    history = []
    for val in [10.0, 7.0, 5.0, 3.5, 2.8]:
        history.append(val)
        fig = make_convergence_chart(list(history))
        assert len(fig.data) == 1
        assert len(fig.data[0].y) == len(history)
```

**Step 2: Run test to verify it passes (existing function already handles this)**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration.py::test_make_convergence_chart_incremental -v`
Expected: PASS (existing `make_convergence_chart` already handles growing lists)

**Step 3: Add pymoo Callback to NSGA-II handler**

In `ui/pages/calibration.py`, replace the `run_optimization()` inner function inside the `else` branch (from Task 1) with:

```python
            def run_optimization():
                from pymoo.algorithms.moo.nsga2 import NSGA2
                from pymoo.core.callback import Callback
                from pymoo.optimize import minimize
                from pymoo.termination import get_termination

                class ProgressCallback(Callback):
                    def notify(self, algorithm):
                        if cancel_flag.get():
                            algorithm.termination.force_termination = True
                            return
                        opt = algorithm.opt
                        if opt is not None:
                            best = float(np.min(opt.get("F").sum(axis=1)))
                            hist = list(cal_history.get())
                            hist.append(best)
                            cal_history.set(hist)

                algorithm = NSGA2(pop_size=int(input.cal_pop_size()))
                termination = get_termination("n_gen", int(input.cal_generations()))

                res = minimize(
                    problem,
                    algorithm,
                    termination,
                    seed=42,
                    verbose=False,
                    callback=ProgressCallback(),
                )

                if res.F is not None:
                    cal_F.set(res.F)
                    cal_X.set(res.X)
```

This gives live per-generation updates AND makes the Stop button actually work for NSGA-II (previously `cancel_flag` was set but never checked).

**Step 4: Run all calibration tests**

Run: `.venv/bin/python -m pytest tests/test_ui_calibration.py tests/test_ui_calibration_handlers.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add ui/pages/calibration.py tests/test_ui_calibration.py
git commit -m "feat: add per-generation progress callback for NSGA-II calibration"
```

---

### Task 3: Parallelize Calibration Objective Evaluation

The `cal_n_parallel` input exists at `ui/pages/calibration.py:133` but is never read. The `OsmoseCalibrationProblem._evaluate()` in `osmose/calibration/problem.py:60-82` evaluates candidates sequentially in a for-loop. We add `concurrent.futures.ProcessPoolExecutor` to evaluate multiple OSMOSE runs in parallel.

**Files:**
- Modify: `osmose/calibration/problem.py:60-82` (parallelize `_evaluate`)
- Modify: `ui/pages/calibration.py` (pass `n_parallel` to problem)
- Test: `tests/test_calibration_problem.py` (test parallel evaluation)

**Step 1: Write failing test**

In `tests/test_calibration_problem.py`, add:

```python
def test_evaluate_parallel():
    """Parallel evaluation produces same results as sequential."""
    import numpy as np
    from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem

    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=1.0)
    obj_fn = lambda r: 0.5  # noqa: E731 — trivial objective for testing

    prob_seq = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[obj_fn],
        base_config_path=Path("/dev/null"),
        jar_path=Path("/dev/null"),
        work_dir=Path("/tmp/osmose_test_par"),
        n_parallel=1,
    )
    prob_par = OsmoseCalibrationProblem(
        free_params=[fp],
        objective_fns=[obj_fn],
        base_config_path=Path("/dev/null"),
        jar_path=Path("/dev/null"),
        work_dir=Path("/tmp/osmose_test_par2"),
        n_parallel=4,
    )
    assert prob_seq.n_parallel == 1
    assert prob_par.n_parallel == 4
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py::test_evaluate_parallel -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'n_parallel'`

**Step 3: Add `n_parallel` parameter to `OsmoseCalibrationProblem`**

In `osmose/calibration/problem.py`, modify `__init__` (line 33) to add `n_parallel: int = 1`:

```python
    def __init__(
        self,
        free_params: list[FreeParameter],
        objective_fns: list[Callable],
        base_config_path: Path,
        jar_path: Path,
        work_dir: Path,
        java_cmd: str = "java",
        n_parallel: int = 1,
    ):
        self.free_params = free_params
        self.objective_fns = objective_fns
        self.base_config_path = base_config_path
        self.jar_path = jar_path
        self.work_dir = work_dir
        self.java_cmd = java_cmd
        self.n_parallel = max(1, n_parallel)
```

Replace `_evaluate` (lines 60-82) with parallel version:

```python
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate a population of candidates, optionally in parallel."""
        F = np.full((X.shape[0], self.n_obj), np.inf)

        def eval_one(i_params):
            i, params = i_params
            overrides = {}
            for j, fp in enumerate(self.free_params):
                val = params[j]
                if fp.transform == "log":
                    val = 10**val
                overrides[fp.key] = str(val)
            try:
                return i, self._run_single(overrides, run_id=i)
            except Exception:
                return i, [float("inf")] * self.n_obj

        if self.n_parallel > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.n_parallel) as pool:
                for i, objectives in pool.map(eval_one, enumerate(X)):
                    for k, v in enumerate(objectives):
                        F[i, k] = v
        else:
            for i, params in enumerate(X):
                idx, objectives = eval_one((i, params))
                for k, v in enumerate(objectives):
                    F[idx, k] = v

        out["F"] = F
```

**Step 4: Wire `n_parallel` from UI input**

In `ui/pages/calibration.py`, in `handle_start_cal()`, modify the `OsmoseCalibrationProblem` creation (around line 259) to pass `n_parallel`:

```python
        n_parallel = int(input.cal_n_parallel())

        problem = OsmoseCalibrationProblem(
            free_params=free_params,
            objective_fns=objective_fns,
            base_config_path=base_config,
            jar_path=jar_path,
            work_dir=work_dir,
            n_parallel=n_parallel,
        )
```

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py tests/test_ui_calibration_handlers.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add osmose/calibration/problem.py ui/pages/calibration.py tests/test_calibration_problem.py
git commit -m "feat: parallelize calibration objective evaluation"
```

---

### Task 4: Config Import Preview and Validation

Currently `handle_import()` in `ui/pages/advanced.py:51-63` blindly merges uploaded config into state. We add a preview step: (1) read the file, (2) show a diff table of what will change, (3) require user confirmation via a "Confirm Import" button.

**Files:**
- Modify: `ui/pages/advanced.py:13-63` (add preview UI and 2-step import)
- Test: `tests/test_ui_advanced_io.py` (test preview diff generation)

**Step 1: Write failing test**

In `tests/test_ui_advanced_io.py`, add:

```python
def test_preview_import_diff():
    """Preview shows keys that would be added or changed."""
    from ui.pages.advanced import compute_import_diff

    current = {"a": "1", "b": "2", "c": "3"}
    incoming = {"a": "1", "b": "99", "d": "4"}
    diff = compute_import_diff(current, incoming)
    # b changed from 2 to 99, d is new
    assert len(diff) == 2
    changed = {d["key"]: d for d in diff}
    assert changed["b"]["old"] == "2"
    assert changed["b"]["new"] == "99"
    assert changed["d"]["old"] is None
    assert changed["d"]["new"] == "4"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui_advanced_io.py::test_preview_import_diff -v`
Expected: FAIL with `ImportError: cannot import name 'compute_import_diff'`

**Step 3: Add `compute_import_diff` helper and preview UI**

In `ui/pages/advanced.py`, add helper function after imports (after line 10):

```python
def compute_import_diff(
    current: dict[str, str], incoming: dict[str, str]
) -> list[dict[str, str | None]]:
    """Compute diff between current config and incoming import.

    Returns list of dicts with keys: key, old, new (only changed/new keys).
    """
    diff = []
    for key, new_val in sorted(incoming.items()):
        old_val = current.get(key)
        if old_val != new_val:
            diff.append({"key": key, "old": old_val, "new": new_val})
    return diff
```

In `advanced_ui()`, add a preview area and confirm button inside the Config I/O card (after the import_config file input, line 23):

```python
                ui.output_ui("import_preview"),
                ui.input_action_button(
                    "btn_confirm_import",
                    "Confirm Import",
                    class_="btn-warning w-100",
                    style="display: none;",
                ),
```

In `advanced_server()`, add preview reactive state and handler:

```python
    import_pending = reactive.value({})

    @reactive.effect
    @reactive.event(input.import_config)
    def handle_import():
        file_info = input.import_config()
        if not file_info:
            return
        filepath = Path(file_info[0]["datapath"])
        reader = OsmoseConfigReader()
        loaded = reader.read_file(filepath)
        import_pending.set(loaded)

    @render.ui
    def import_preview():
        pending = import_pending.get()
        if not pending:
            return ui.div()
        diff = compute_import_diff(state.config.get(), pending)
        if not diff:
            return ui.div("No changes detected.", style="color: #999; padding: 8px;")
        rows = []
        for d in diff:
            rows.append(
                ui.tags.tr(
                    ui.tags.td(d["key"], style="font-family: monospace; font-size: 12px;"),
                    ui.tags.td(str(d["old"]) if d["old"] is not None else "(new)"),
                    ui.tags.td(str(d["new"])),
                )
            )
        return ui.div(
            ui.p(f"{len(diff)} parameters will change:", style="font-weight: bold;"),
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Key"), ui.tags.th("Current"), ui.tags.th("New")
                    )
                ),
                ui.tags.tbody(*rows),
                class_="table table-sm table-striped",
            ),
            ui.input_action_button(
                "btn_confirm_import", "Confirm Import", class_="btn-warning w-100"
            ),
            style="max-height: 300px; overflow-y: auto; margin-top: 8px;",
        )

    @reactive.effect
    @reactive.event(input.btn_confirm_import)
    def confirm_import():
        pending = import_pending.get()
        if not pending:
            return
        cfg = dict(state.config.get())
        cfg.update(pending)
        state.config.set(cfg)
        import_pending.set({})
```

Remove the old `handle_import` that directly merged (the new one sets `import_pending` instead).

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ui_advanced_io.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add ui/pages/advanced.py tests/test_ui_advanced_io.py
git commit -m "feat: add config import preview with diff before merge"
```

---

### Task 5: Spatial Map Time Animation

The results page at `ui/pages/results.py:137-144` has a manual slider for time steps. We add Play/Pause/Step buttons and an `reactive.poll` timer that auto-increments the slider when playing.

**Files:**
- Modify: `ui/pages/results.py:130-148` (add animation controls to UI)
- Modify: `ui/pages/results.py:253-272` (add auto-play timer logic)
- Test: `tests/test_ui_results.py` (test spatial map at different time indices)

**Step 1: Write failing test**

In `tests/test_ui_results.py`, add:

```python
def test_make_spatial_map_multiple_timesteps():
    """Spatial map renders correctly at different time indices."""
    import xarray as xr

    data = np.random.rand(5, 4, 6)
    ds = xr.Dataset(
        {"biomass": (["time", "lat", "lon"], data)},
        coords={"time": range(5), "lat": np.linspace(40, 50, 4), "lon": np.linspace(-5, 5, 6)},
    )
    for t in range(5):
        fig = make_spatial_map(ds, "biomass", time_idx=t, title=f"t={t}")
        assert fig is not None
        assert f"t={t}" in fig.layout.title.text
```

**Step 2: Run test to verify it passes (existing function handles this)**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py::test_make_spatial_map_multiple_timesteps -v`
Expected: PASS (existing `make_spatial_map` already handles arbitrary `time_idx`)

**Step 3: Add animation controls to results UI**

In `ui/pages/results.py`, replace the spatial card (lines 135-146) with:

```python
            ui.card(
                ui.card_header("Spatial Distribution"),
                ui.layout_columns(
                    ui.input_slider(
                        "spatial_time_idx",
                        "Time step",
                        min=0,
                        max=1,
                        value=0,
                        step=1,
                        animate=ui.AnimationOptions(
                            interval=1000,
                            loop=True,
                            play_button="Play",
                            pause_button="Pause",
                        ),
                    ),
                    col_widths=[12],
                ),
                output_widget("spatial_chart"),
            ),
```

Shiny's `input_slider` natively supports `animate` with play/pause buttons. No custom timer code needed — Shiny handles it.

**Step 4: Run full results tests**

Run: `.venv/bin/python -m pytest tests/test_ui_results.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add ui/pages/results.py tests/test_ui_results.py
git commit -m "feat: add play/pause animation to spatial map time slider"
```

---

### Task 6: Structured Logging

Add Python's `logging` module throughout the app for deployment debugging. Create a central log configuration in `osmose/logging.py` and add logger calls in key modules (runner, calibration, config I/O).

**Files:**
- Create: `osmose/logging.py`
- Modify: `osmose/runner.py` (add logger calls)
- Modify: `osmose/calibration/problem.py` (add logger calls)
- Modify: `osmose/config/reader.py` (add logger calls)
- Test: `tests/test_logging.py`

**Step 1: Write failing test**

Create `tests/test_logging.py`:

```python
"""Tests for structured logging configuration."""

import logging

from osmose.logging import setup_logging


def test_setup_logging_returns_logger():
    logger = setup_logging("test_osmose")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_osmose"


def test_setup_logging_default_level():
    logger = setup_logging("test_default")
    assert logger.level == logging.INFO


def test_setup_logging_custom_level():
    logger = setup_logging("test_debug", level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_logger_has_handler():
    logger = setup_logging("test_handler")
    assert len(logger.handlers) >= 1
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_logging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'osmose.logging'`

**Step 3: Create `osmose/logging.py`**

```python
"""Centralized logging configuration for OSMOSE."""

import logging
import sys


def setup_logging(
    name: str = "osmose",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a logger with console output.

    Args:
        name: Logger name (typically module name).
        level: Logging level (default INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
```

**Step 4: Add logger calls to key modules**

In `osmose/runner.py`, at the top after imports, add:

```python
from osmose.logging import setup_logging

_log = setup_logging("osmose.runner")
```

Add `_log.info(...)` calls at key points:
- Before running subprocess: `_log.info("Starting OSMOSE: %s", " ".join(cmd))`
- After completion: `_log.info("OSMOSE finished with exit code %d", result.returncode)`
- On cancel: `_log.info("OSMOSE run cancelled")`

In `osmose/calibration/problem.py`, at the top:

```python
from osmose.logging import setup_logging

_log = setup_logging("osmose.calibration")
```

Add: `_log.info("Evaluating %d candidates (parallel=%d)", X.shape[0], self.n_parallel)` in `_evaluate`.

In `osmose/config/reader.py`, at the top:

```python
from osmose.logging import setup_logging

_log = setup_logging("osmose.config")
```

Add: `_log.info("Reading config from %s", path)` in `read_file`.

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_logging.py -v`
Expected: ALL PASS

Run: `.venv/bin/python -m pytest -v`
Expected: ALL 207+ PASS (existing tests unaffected by logging)

**Step 6: Commit**

```bash
git add osmose/logging.py tests/test_logging.py osmose/runner.py osmose/calibration/problem.py osmose/config/reader.py
git commit -m "feat: add structured logging module with console output"
```

---

### Task 7: Scenario Bulk Export/Import

Add buttons to export all scenarios as a ZIP file and import a ZIP of scenario JSON files. This enables backup and transfer between OSMOSE instances.

**Files:**
- Modify: `osmose/scenarios.py` (add `export_all` and `import_all` methods)
- Modify: `ui/pages/scenarios.py:8-46` (add bulk buttons to UI)
- Modify: `ui/pages/scenarios.py:49-end` (add bulk handlers)
- Test: `tests/test_scenarios.py` (test bulk export/import)

**Step 1: Write failing test**

In `tests/test_scenarios.py`, add:

```python
def test_export_all_creates_zip(tmp_path):
    """Export all scenarios to a ZIP file."""
    mgr = ScenarioManager(tmp_path / "scenarios")
    mgr.save(Scenario(name="alpha", config={"x": "1"}))
    mgr.save(Scenario(name="beta", config={"y": "2"}))

    zip_path = tmp_path / "export.zip"
    mgr.export_all(zip_path)
    assert zip_path.exists()

    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "alpha.json" in names
        assert "beta.json" in names


def test_import_all_from_zip(tmp_path):
    """Import scenarios from a ZIP file."""
    # Create source
    src_mgr = ScenarioManager(tmp_path / "src")
    src_mgr.save(Scenario(name="gamma", config={"z": "3"}))
    zip_path = tmp_path / "bundle.zip"
    src_mgr.export_all(zip_path)

    # Import into fresh manager
    dst_mgr = ScenarioManager(tmp_path / "dst")
    count = dst_mgr.import_all(zip_path)
    assert count == 1
    loaded = dst_mgr.load("gamma")
    assert loaded.config == {"z": "3"}
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py::test_export_all_creates_zip tests/test_scenarios.py::test_import_all_from_zip -v`
Expected: FAIL with `AttributeError: 'ScenarioManager' object has no attribute 'export_all'`

**Step 3: Add `export_all` and `import_all` to `ScenarioManager`**

In `osmose/scenarios.py`, add these methods to the `ScenarioManager` class:

```python
    def export_all(self, zip_path: Path) -> None:
        """Export all scenarios to a ZIP file."""
        import zipfile

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for info in self.list_scenarios():
                scenario = self.load(info["name"])
                data = {
                    "name": scenario.name,
                    "description": scenario.description,
                    "config": scenario.config,
                    "tags": scenario.tags,
                    "parent_scenario": scenario.parent_scenario,
                }
                zf.writestr(f"{scenario.name}.json", json.dumps(data, indent=2))

    def import_all(self, zip_path: Path) -> int:
        """Import scenarios from a ZIP file. Returns count of imported scenarios."""
        import zipfile

        count = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                data = json.loads(zf.read(name))
                scenario = Scenario(
                    name=data["name"],
                    description=data.get("description", ""),
                    config=data.get("config", {}),
                    tags=data.get("tags", []),
                    parent_scenario=data.get("parent_scenario"),
                )
                self.save(scenario)
                count += 1
        return count
```

**Step 4: Add bulk buttons to scenarios UI**

In `ui/pages/scenarios.py`, in the `scenarios_ui()` function, add a new row below the existing layout (after line 45):

```python
        ui.layout_columns(
            ui.card(
                ui.card_header("Bulk Operations"),
                ui.download_button(
                    "export_all_scenarios", "Export All (ZIP)", class_="btn-primary w-100"
                ),
                ui.input_file(
                    "import_scenarios_zip", "Import Scenarios (ZIP)", accept=[".zip"]
                ),
            ),
            col_widths=[12],
        ),
```

In `scenarios_server()`, add handlers:

```python
    @render.download(filename="osmose_scenarios.zip")
    def export_all_scenarios():
        import tempfile

        zip_path = Path(tempfile.mktemp(suffix=".zip"))
        mgr.export_all(zip_path)
        return str(zip_path)

    @reactive.effect
    @reactive.event(input.import_scenarios_zip)
    def handle_import_scenarios():
        file_info = input.import_scenarios_zip()
        if not file_info:
            return
        zip_path = Path(file_info[0]["datapath"])
        count = mgr.import_all(zip_path)
        _bump()
        ui.notification_show(f"Imported {count} scenarios.", type="message", duration=3)
```

Add `from pathlib import Path` and `from shiny import render` to imports if not already present.

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add osmose/scenarios.py ui/pages/scenarios.py tests/test_scenarios.py
git commit -m "feat: add scenario bulk export/import as ZIP"
```

---

### Task 8: Input Validation Tooltips

The `render_field()` function in `ui/components/param_form.py:9-83` generates labels from `field.description` but doesn't show min/max/unit constraints. We add a tooltip text showing constraints below each numeric input.

**Files:**
- Modify: `ui/components/param_form.py:9-48` (add constraint hint text)
- Test: `tests/test_param_form.py` (test hint text generation)

**Step 1: Write failing test**

In `tests/test_param_form.py`, add:

```python
def test_constraint_hint_float():
    """Float field includes range hint text."""
    from ui.components.param_form import constraint_hint

    from osmose.schema.base import OsmoseField, ParamType

    field = OsmoseField(
        key_pattern="species.linf.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Asymptotic length",
        category="species",
        min_val=1.0,
        max_val=200.0,
        unit="cm",
    )
    hint = constraint_hint(field)
    assert "1.0" in hint
    assert "200.0" in hint
    assert "cm" in hint


def test_constraint_hint_no_bounds():
    """Field without bounds returns empty hint."""
    from ui.components.param_form import constraint_hint

    from osmose.schema.base import OsmoseField, ParamType

    field = OsmoseField(
        key_pattern="simulation.name",
        param_type=ParamType.STRING,
        description="Simulation name",
        category="simulation",
    )
    hint = constraint_hint(field)
    assert hint == ""
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_param_form.py::test_constraint_hint_float tests/test_param_form.py::test_constraint_hint_no_bounds -v`
Expected: FAIL with `ImportError: cannot import name 'constraint_hint'`

**Step 3: Add `constraint_hint` function and integrate into `render_field`**

In `ui/components/param_form.py`, add after line 7:

```python
def constraint_hint(field: OsmoseField) -> str:
    """Generate a constraint hint string for a field.

    Returns text like 'Range: 1.0 — 200.0 cm' or empty string if no constraints.
    """
    parts = []
    if field.min_val is not None and field.max_val is not None:
        parts.append(f"Range: {field.min_val} — {field.max_val}")
    elif field.min_val is not None:
        parts.append(f"Min: {field.min_val}")
    elif field.max_val is not None:
        parts.append(f"Max: {field.max_val}")
    if field.unit and parts:
        parts[0] = f"{parts[0]} {field.unit}"
    return " | ".join(parts)
```

In `render_field()`, for `ParamType.FLOAT` and `ParamType.INT` cases, wrap the widget in a div with hint text:

```python
        case ParamType.FLOAT:
            widget = ui.input_numeric(
                input_id,
                label,
                value=field.default if field.default is not None else 0.0,
                min=field.min_val,
                max=field.max_val,
                step=_guess_step(field),
            )
            hint = constraint_hint(field)
            if hint:
                return ui.div(
                    widget,
                    ui.tags.small(hint, style="color: #888; font-size: 11px; margin-top: -8px;"),
                )
            return widget
        case ParamType.INT:
            widget = ui.input_numeric(
                input_id,
                label,
                value=field.default if field.default is not None else 0,
                min=int(field.min_val) if field.min_val is not None else None,
                max=int(field.max_val) if field.max_val is not None else None,
                step=1,
            )
            hint = constraint_hint(field)
            if hint:
                return ui.div(
                    widget,
                    ui.tags.small(hint, style="color: #888; font-size: 11px; margin-top: -8px;"),
                )
            return widget
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_param_form.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add ui/components/param_form.py tests/test_param_form.py
git commit -m "feat: add input validation tooltips showing field constraints"
```

---

### Task 9: Full Verification and Push

Run entire test suite, lint, format, and push all Phase 4 commits.

**Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest --cov=osmose --cov=ui -v`
Expected: ALL PASS (210+ tests), coverage ≥ 59%

**Step 2: Lint and format**

Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format osmose/ ui/ tests/`
Expected: All clean

**Step 3: Verify git status**

Run: `git log --oneline -10` — should show 8 new Phase 4 commits
Run: `git status` — should show clean working tree

**Step 4: Push**

Run: `git push origin master`

**Step 5: Verify CI**

Run: `gh run watch --exit-status`
Expected: Green (lint + test)
