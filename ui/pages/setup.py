"""Species & Simulation setup page."""

from shiny import ui, render
from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from ui.components.param_form import render_category, render_species_params


def setup_ui():
    return ui.page_fluid(
        ui.layout_columns(
            # Left column: Simulation settings
            ui.card(
                ui.card_header("Simulation Settings"),
                render_category(
                    [f for f in SIMULATION_FIELDS if not f.advanced],
                ),
            ),
            # Right column: Species configuration (dynamic)
            ui.card(
                ui.card_header("Species Configuration"),
                ui.input_numeric("n_species", "Number of focal species", value=3, min=1, max=20),
                ui.input_switch("show_advanced_species", "Show advanced parameters", value=False),
                ui.output_ui("species_panels"),
            ),
            col_widths=[4, 8],
        ),
    )


def setup_server(input, output, session, state):
    @render.ui
    def species_panels():
        n = input.n_species()
        show_adv = input.show_advanced_species()
        panels = []
        for i in range(n):
            name = f"Species {i}"  # Will be editable
            panels.append(
                render_species_params(
                    SPECIES_FIELDS,
                    species_idx=i,
                    species_name=name,
                    show_advanced=show_adv,
                )
            )
        return ui.div(*panels)
