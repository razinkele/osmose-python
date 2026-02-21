"""Scenario management page."""

from shiny import ui, reactive, render


def scenarios_ui():
    return ui.page_fluid(
        ui.layout_columns(
            # Left: Save & manage
            ui.card(
                ui.card_header("Save Scenario"),
                ui.input_text("scenario_name", "Scenario name"),
                ui.input_text("scenario_desc", "Description"),
                ui.input_text("scenario_tags", "Tags (comma-separated)"),
                ui.input_action_button("btn_save_scenario", "Save Current Config", class_="btn-success w-100"),
            ),
            # Middle: Scenario list
            ui.card(
                ui.card_header("Saved Scenarios"),
                ui.output_ui("scenario_list"),
                ui.hr(),
                ui.layout_columns(
                    ui.input_action_button("btn_load_scenario", "Load", class_="btn-primary w-100"),
                    ui.input_action_button("btn_fork_scenario", "Fork", class_="btn-info w-100"),
                    ui.input_action_button("btn_delete_scenario", "Delete", class_="btn-danger w-100"),
                    col_widths=[4, 4, 4],
                ),
            ),
            # Right: Compare
            ui.card(
                ui.card_header("Compare Scenarios"),
                ui.input_select("compare_a", "Scenario A", choices={}),
                ui.input_select("compare_b", "Scenario B", choices={}),
                ui.input_action_button("btn_compare", "Compare", class_="btn-warning w-100"),
                ui.hr(),
                ui.output_ui("compare_results"),
            ),
            col_widths=[3, 5, 4],
        ),
    )


def scenarios_server(input, output, session):
    @render.ui
    def scenario_list():
        return ui.div(
            "No scenarios saved yet.",
            style="padding: 20px; text-align: center; color: #999;",
        )

    @render.ui
    def compare_results():
        return ui.div(
            "Select two scenarios and click Compare.",
            style="padding: 20px; text-align: center; color: #999;",
        )
