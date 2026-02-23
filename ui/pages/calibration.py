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

from pymoo.core.callback import Callback

from osmose.calibration.objectives import biomass_rmse, diet_distance
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem
from osmose.calibration.sensitivity import SensitivityAnalyzer
from osmose.calibration.surrogate import SurrogateCalibrator
from osmose.config.writer import OsmoseConfigWriter
from osmose.schema.base import ParamType
from osmose.schema.registry import ParameterRegistry


class ProgressCallback(Callback):
    """pymoo callback that reports per-generation progress and supports cancellation.

    Args:
        cal_history_append: Callable to append a float (best objective value) each generation.
        cancel_check: Callable returning True if optimization should be cancelled.
    """

    def __init__(self, cal_history_append, cancel_check):
        super().__init__()
        self._cal_history_append = cal_history_append
        self._cancel_check = cancel_check

    def notify(self, algorithm):
        if self._cancel_check():
            algorithm.termination.force_termination = True
            return
        F = algorithm.opt.get("F")
        best = float(np.min(F.sum(axis=1)))
        self._cal_history_append(best)


# -- Helper functions (module-level, testable without Shiny) -------------------


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


# -- Shiny UI / Server --------------------------------------------------------


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
                ui.p(
                    "Select parameters to optimize:",
                    style="color: #999; font-size: 13px;",
                ),
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
    surrogate_status = reactive.value("")

    @render.text
    def cal_status():
        surr = surrogate_status.get()
        if surr:
            return surr
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
        selected = collect_selected_params(input, state)
        if not selected:
            cal_history.set([])
            return

        jar_path = Path(state.jar_path.get())
        if not jar_path.exists():
            return

        obs_bio = input.observed_biomass()
        obs_diet = input.observed_diet()
        if not obs_bio and not obs_diet:
            return

        free_params = build_free_params(selected)
        work_dir = Path(tempfile.mkdtemp(prefix="osmose_cal_"))

        writer = OsmoseConfigWriter()
        config_dir = work_dir / "config"
        writer.write(state.config.get(), config_dir)
        base_config = config_dir / "osm_all-parameters.csv"

        objective_fns = []
        if obs_bio:
            obs_bio_df = pd.read_csv(obs_bio[0]["datapath"])
            objective_fns.append(lambda r, df=obs_bio_df: biomass_rmse(r.biomass(), df))
        if obs_diet:
            obs_diet_df = pd.read_csv(obs_diet[0]["datapath"])
            objective_fns.append(lambda r, df=obs_diet_df: diet_distance(r.diet_matrix(), df))

        if not objective_fns:
            return

        n_parallel = int(input.cal_n_parallel())

        problem = OsmoseCalibrationProblem(
            free_params=free_params,
            objective_fns=objective_fns,
            base_config_path=base_config,
            jar_path=jar_path,
            work_dir=work_dir,
            n_parallel=n_parallel,
        )

        cancel_flag.set(False)
        cal_history.set([])
        cal_F.set(None)
        cal_X.set(None)
        surrogate_status.set("")

        algorithm_choice = input.cal_algorithm()
        pop_size = int(input.cal_pop_size())
        generations = int(input.cal_generations())

        if algorithm_choice == "surrogate":

            def run_surrogate():
                bounds = [(fp.lower_bound, fp.upper_bound) for fp in free_params]
                n_obj = len(objective_fns)
                calibrator = SurrogateCalibrator(param_bounds=bounds, n_objectives=n_obj)

                n_samples = pop_size
                surrogate_status.set(f"Generating {n_samples} Latin hypercube samples...")
                samples = calibrator.generate_samples(n_samples=n_samples)

                # Evaluate OSMOSE for each sample
                Y = np.zeros((n_samples, n_obj))
                for idx in range(n_samples):
                    if cancel_flag.get():
                        surrogate_status.set("Cancelled.")
                        return

                    surrogate_status.set(f"Evaluating sample {idx + 1}/{n_samples}...")
                    overrides = {fp.key: str(samples[idx, j]) for j, fp in enumerate(free_params)}
                    try:
                        result = problem._run_single(overrides, run_id=idx)
                        for k in range(n_obj):
                            Y[idx, k] = result[k]
                    except Exception as exc:
                        Y[idx, :] = float("inf")
                        surrogate_status.set(f"Sample {idx + 1}/{n_samples} failed: {exc}")

                if cancel_flag.get():
                    surrogate_status.set("Cancelled.")
                    return

                surrogate_status.set("Fitting GP model...")
                calibrator.fit(samples, Y)

                surrogate_status.set("Finding optimum on surrogate...")
                optimum = calibrator.find_optimum()

                # Set results for the UI
                cal_X.set(samples)
                cal_F.set(Y)
                history = [float(np.min(Y[: i + 1].sum(axis=1))) for i in range(n_samples)]
                cal_history.set(history)
                surrogate_status.set(
                    f"Done. Best predicted objective: {optimum['predicted_objectives']}"
                )

            thread = threading.Thread(target=run_surrogate, daemon=True)
            thread.start()
            cal_thread.set(thread)

        else:
            # NSGA-II (default)
            def run_optimization():
                from pymoo.algorithms.moo.nsga2 import NSGA2
                from pymoo.optimize import minimize
                from pymoo.termination import get_termination

                algorithm = NSGA2(pop_size=pop_size)
                termination = get_termination("n_gen", generations)

                def append_history(val):
                    current = cal_history.get()
                    cal_history.set(current + [val])

                callback = ProgressCallback(
                    cal_history_append=append_history,
                    cancel_check=cancel_flag.get,
                )

                res = minimize(
                    problem,
                    algorithm,
                    termination,
                    seed=42,
                    verbose=False,
                    callback=callback,
                )

                if res.F is not None:
                    cal_F.set(res.F)
                    cal_X.set(res.X)

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
            sens_work_dir = Path(tempfile.mkdtemp(prefix="osmose_sens_"))
            writer = OsmoseConfigWriter()
            config_dir = sens_work_dir / "config"
            writer.write(state.config.get(), config_dir)
            base_config = config_dir / "osm_all-parameters.csv"

            Y = np.zeros(samples.shape[0])
            for idx, row in enumerate(samples):
                overrides = {selected[j]["key"]: str(row[j]) for j in range(len(selected))}
                try:
                    prob = OsmoseCalibrationProblem(
                        free_params=build_free_params(selected),
                        objective_fns=[lambda r, df=obs_bio_df: biomass_rmse(r.biomass(), df)],
                        base_config_path=base_config,
                        jar_path=jar_path,
                        work_dir=sens_work_dir / f"sens_{idx}",
                    )
                    result = prob._run_single(overrides, run_id=idx)
                    Y[idx] = result[0]
                except Exception:
                    Y[idx] = float("inf")

            sens_result = analyzer.analyze(Y)
            sensitivity_result.set(sens_result)

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
