"""OSMOSE Python Interface - main Shiny application."""

from pathlib import Path

from shiny import App, ui

from ui.state import AppState
from ui.theme import THEME
import ui.charts as _charts  # noqa: F401 — registers custom plotly template

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

_WWW = Path(__file__).parent / "www"


def _nav_section(label: str):
    """Render a section header in the pill list sidebar."""
    return ui.nav_control(
        ui.tags.span(label, class_="osmose-section-label"),
    )


app_ui = ui.page_fillable(
    # ── Custom CSS ──────────────────────────────────────────────
    ui.head_content(ui.include_css(_WWW / "osmose.css")),
    # ── App header ──────────────────────────────────────────────
    ui.div(
        ui.tags.h4(
            "OSMOSE",
            ui.tags.span(" | Python Interface", class_="subtitle"),
            class_="osmose-logo",
        ),
        ui.tags.span(
            ui.tags.span(class_="dot"),
            "Marine Ecosystem Simulator",
            class_="osmose-badge",
        ),
        class_="osmose-header",
    ),
    # ── Left pill navigation with grouped sections ──────────────
    ui.navset_pill_list(
        # Configure
        _nav_section("Configure"),
        ui.nav_panel("Setup", setup_ui(), value="setup"),
        ui.nav_panel("Grid & Maps", grid_ui(), value="grid"),
        ui.nav_panel("Forcing", forcing_ui(), value="forcing"),
        ui.nav_panel("Fishing", fishing_ui(), value="fishing"),
        ui.nav_panel("Movement", movement_ui(), value="movement"),
        # Execute
        _nav_section("Execute"),
        ui.nav_panel("Run", run_ui(), value="run"),
        ui.nav_panel("Results", results_ui(), value="results"),
        # Optimize
        _nav_section("Optimize"),
        ui.nav_panel("Calibration", calibration_ui(), value="calibration"),
        # Manage
        _nav_section("Manage"),
        ui.nav_panel("Scenarios", scenarios_ui(), value="scenarios"),
        ui.nav_panel("Advanced", advanced_ui(), value="advanced"),
        id="main_nav",
        selected="setup",
        widths=(2, 10),
        well=False,
    ),
    theme=THEME,
)


def server(input, output, session):
    state = AppState()
    state.reset_to_defaults()

    setup_server(input, output, session, state)
    grid_server(input, output, session, state)
    forcing_server(input, output, session, state)
    fishing_server(input, output, session, state)
    movement_server(input, output, session, state)
    run_server(input, output, session, state)
    results_server(input, output, session, state)
    calibration_server(input, output, session, state)
    scenarios_server(input, output, session, state)
    advanced_server(input, output, session, state)


app = App(app_ui, server)
