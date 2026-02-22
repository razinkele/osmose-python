# ui/pages/calibration.py
"""Calibration page - multi-objective optimization and sensitivity analysis."""

from shiny import ui, render


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
                        ui.output_ui("cal_progress"),
                    ),
                ),
                ui.nav_panel(
                    "Pareto Front",
                    ui.div(
                        ui.output_ui("pareto_plot_placeholder"),
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
                            "btn_sensitivity", "Run Sensitivity Analysis", class_="btn-info w-100"
                        ),
                        ui.output_ui("sensitivity_plot_placeholder"),
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
        # Common calibration parameters
        common_params = [
            ("species.k.sp0", "Growth K (sp0)"),
            ("species.linf.sp0", "L-infinity (sp0)"),
            ("mortality.natural.rate.sp0", "Natural mortality (sp0)"),
            ("predation.ingestion.rate.max.sp0", "Max ingestion rate (sp0)"),
            ("population.seeding.biomass.sp0", "Seeding biomass (sp0)"),
        ]
        checkboxes = [
            ui.input_checkbox(f"cal_param_{key.replace('.', '_')}", label, value=False)
            for key, label in common_params
        ]
        return ui.div(*checkboxes)

    @render.ui
    def cal_progress():
        return ui.div(
            "Calibration progress will appear here.",
            style="height: 300px; display: flex; align-items: center; justify-content: center; "
            "border: 1px dashed #4e5d6c; border-radius: 8px; color: #999;",
        )

    @render.ui
    def pareto_plot_placeholder():
        return ui.div(
            "Pareto front plot will appear here after calibration.",
            style="height: 400px; display: flex; align-items: center; justify-content: center; "
            "border: 1px dashed #4e5d6c; border-radius: 8px; color: #999;",
        )

    @render.ui
    def best_params_table():
        return ui.div(
            "Best parameter values will appear here after calibration.",
            style="padding: 20px; text-align: center; color: #999;",
        )

    @render.ui
    def sensitivity_plot_placeholder():
        return ui.div(
            "Sensitivity analysis results will appear here.",
            style="height: 400px; display: flex; align-items: center; justify-content: center; "
            "border: 1px dashed #4e5d6c; border-radius: 8px; color: #999;",
        )
