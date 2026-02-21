"""Results visualization page."""

from shiny import ui, reactive, render


def results_ui():
    return ui.page_fluid(
        ui.layout_columns(
            # Sidebar: Controls
            ui.card(
                ui.card_header("Output Controls"),
                ui.input_text("output_dir", "Output directory", value="output/"),
                ui.input_action_button("btn_load_results", "Load Results", class_="btn-primary w-100"),
                ui.hr(),
                ui.input_select("result_species", "Species filter", choices={"all": "All species"}, selected="all"),
                ui.input_select("result_type", "Output type", choices={
                    "biomass": "Biomass",
                    "abundance": "Abundance",
                    "yield": "Yield",
                    "mortality": "Mortality",
                    "diet": "Diet Matrix",
                    "trophic": "Trophic Level",
                }, selected="biomass"),
            ),
            # Main: Visualization
            ui.card(
                ui.card_header("Time Series"),
                ui.output_ui("results_plot_placeholder"),
            ),
            col_widths=[3, 9],
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Diet Composition Matrix"),
                ui.output_ui("diet_placeholder"),
            ),
            ui.card(
                ui.card_header("Spatial Distribution"),
                ui.output_ui("spatial_placeholder"),
            ),
            col_widths=[6, 6],
        ),
    )


def results_server(input, output, session):
    @render.ui
    def results_plot_placeholder():
        return ui.div(
            "Load results to view time series plots.",
            style="height: 350px; display: flex; align-items: center; justify-content: center; "
                  "border: 1px dashed #4e5d6c; border-radius: 8px; color: #999;",
        )

    @render.ui
    def diet_placeholder():
        return ui.div(
            "Diet composition heatmap will appear here.",
            style="height: 300px; display: flex; align-items: center; justify-content: center; "
                  "border: 1px dashed #4e5d6c; border-radius: 8px; color: #999;",
        )

    @render.ui
    def spatial_placeholder():
        return ui.div(
            "Spatial biomass maps will appear here.",
            style="height: 300px; display: flex; align-items: center; justify-content: center; "
                  "border: 1px dashed #4e5d6c; border-radius: 8px; color: #999;",
        )
