"""OSMOSE Python Interface - main Shiny application."""

from shiny import App, ui
from ui.theme import THEME

from ui.pages.setup import setup_ui, setup_server
from ui.pages.grid import grid_ui, grid_server
from ui.pages.forcing import forcing_ui, forcing_server
from ui.pages.fishing import fishing_ui, fishing_server
from ui.pages.movement import movement_ui, movement_server
from ui.pages.run import run_ui, run_server
from ui.pages.results import results_ui, results_server
from ui.pages.calibration import calibration_ui, calibration_server
from ui.pages.scenarios import scenarios_ui, scenarios_server
from ui.pages.advanced import advanced_ui, advanced_server

app_ui = ui.page_navbar(
    ui.nav_panel("Setup", setup_ui()),
    ui.nav_panel("Grid & Maps", grid_ui()),
    ui.nav_panel("Forcing", forcing_ui()),
    ui.nav_panel("Fishing", fishing_ui()),
    ui.nav_panel("Movement", movement_ui()),
    ui.nav_panel("Run", run_ui()),
    ui.nav_panel("Results", results_ui()),
    ui.nav_panel("Calibration", calibration_ui()),
    ui.nav_panel("Scenarios", scenarios_ui()),
    ui.nav_panel("Advanced", advanced_ui()),
    title="OSMOSE | Python Interface",
    theme=THEME,
)


def server(input, output, session):
    setup_server(input, output, session)
    grid_server(input, output, session)
    forcing_server(input, output, session)
    fishing_server(input, output, session)
    movement_server(input, output, session)
    run_server(input, output, session)
    results_server(input, output, session)
    calibration_server(input, output, session)
    scenarios_server(input, output, session)
    advanced_server(input, output, session)


app = App(app_ui, server)
