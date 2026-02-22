# ui/pages/calibration.py
"""Calibration page - multi-objective optimization and sensitivity analysis."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from shiny import render, ui
from shinywidgets import output_widget, render_plotly

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
                ui.input_numeric(
                    "cal_generations", "Generations", value=100, min=10, max=1000
                ),
                ui.input_numeric(
                    "cal_n_parallel", "Parallel workers", value=4, min=1, max=32
                ),
                ui.hr(),
                ui.h5("Free Parameters"),
                ui.p("Select parameters to optimize:", style="color: #999; font-size: 13px;"),
                ui.output_ui("free_param_selector"),
                ui.hr(),
                ui.h5("Objectives"),
                ui.input_file(
                    "observed_biomass", "Upload observed biomass CSV", accept=[".csv"]
                ),
                ui.input_file(
                    "observed_diet", "Upload observed diet matrix CSV", accept=[".csv"]
                ),
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
    @render.text
    def cal_status():
        return "Ready. Configure parameters and objectives, then click Start."

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

    @render_plotly
    def convergence_chart():
        # Placeholder: no history yet
        return make_convergence_chart([])

    @render_plotly
    def pareto_chart():
        # Placeholder: no calibration result yet
        return make_pareto_chart(
            np.array([[0.0, 0.0]]),
            ["Biomass RMSE", "Diet Distance"],
        )

    @render.ui
    def best_params_table():
        return ui.div(
            "Best parameter values will appear here after calibration.",
            style="padding: 20px; text-align: center; color: #999;",
        )

    @render_plotly
    def sensitivity_chart():
        # Placeholder: no sensitivity result yet
        return make_sensitivity_chart({
            "param_names": ["(none)"],
            "S1": np.array([0.0]),
            "ST": np.array([0.0]),
        })
