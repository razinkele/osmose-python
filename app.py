"""OSMOSE Python Interface - main Shiny application."""

from shiny import App, ui, reactive
from ui.theme import THEME

app_ui = ui.page_navbar(
    ui.nav_panel("Setup", ui.div(
        ui.h3("Species & Simulation Setup"),
        ui.p("Configure species parameters, simulation settings, and initial conditions."),
        ui.output_ui("setup_content"),
    )),
    ui.nav_panel("Grid & Maps", ui.div(
        ui.h3("Grid Configuration"),
        ui.p("Define the spatial domain, grid dimensions, and land/sea mask."),
        ui.output_ui("grid_content"),
    )),
    ui.nav_panel("Forcing", ui.div(
        ui.h3("Environmental Forcing"),
        ui.p("Configure plankton/LTL groups and environmental forcing data."),
        ui.output_ui("forcing_content"),
    )),
    ui.nav_panel("Fishing", ui.div(
        ui.h3("Fishing Configuration"),
        ui.p("Set up fishing mortality, fisheries, selectivity, and MPAs."),
        ui.output_ui("fishing_content"),
    )),
    ui.nav_panel("Run", ui.div(
        ui.h3("Run Control"),
        ui.p("Execute OSMOSE simulations and monitor progress."),
        ui.output_ui("run_content"),
    )),
    ui.nav_panel("Results", ui.div(
        ui.h3("Results Visualization"),
        ui.p("Explore simulation outputs: biomass, diet, spatial maps, mortality."),
        ui.output_ui("results_content"),
    )),
    ui.nav_panel("Calibration", ui.div(
        ui.h3("Model Calibration"),
        ui.p("Multi-objective optimization and sensitivity analysis."),
        ui.output_ui("calibration_content"),
    )),
    ui.nav_panel("Scenarios", ui.div(
        ui.h3("Scenario Management"),
        ui.p("Save, load, compare, and fork named configurations."),
        ui.output_ui("scenarios_content"),
    )),
    ui.nav_panel("Advanced", ui.div(
        ui.h3("Advanced Configuration"),
        ui.p("Raw parameter editor for all 200+ OSMOSE parameters."),
        ui.output_ui("advanced_content"),
    )),
    title="OSMOSE | Python Interface",
    theme=THEME,
)


def server(input, output, session):
    pass  # Server logic will be added as pages are implemented


app = App(app_ui, server)
