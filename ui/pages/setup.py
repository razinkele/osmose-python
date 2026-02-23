"""Species & Simulation setup page."""

from shiny import ui, reactive, render

from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from ui.components.param_form import render_category, render_species_params
from ui.state import sync_inputs

# Keys for non-indexed simulation fields (synced automatically)
SETUP_GLOBAL_KEYS: list[str] = [f.key_pattern for f in SIMULATION_FIELDS if not f.advanced]


def get_species_keys(species_idx: int, show_advanced: bool = False) -> list[str]:
    """Return resolved OSMOSE keys for one species."""
    keys = []
    for f in SPECIES_FIELDS:
        if f.advanced and not show_advanced:
            continue
        keys.append(f.resolve_key(species_idx))
    return keys


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
            name = f"Species {i}"
            panels.append(
                render_species_params(
                    SPECIES_FIELDS,
                    species_idx=i,
                    species_name=name,
                    show_advanced=show_adv,
                )
            )
        return ui.div(*panels)

    @reactive.effect
    def sync_simulation_inputs():
        """Auto-sync simulation fields to state.config."""
        sync_inputs(input, state, SETUP_GLOBAL_KEYS)

    @reactive.effect
    def sync_species_inputs():
        """Auto-sync species fields to state.config."""
        n = input.n_species()
        show_adv = input.show_advanced_species()
        # Update nspecies in config
        state.update_config("simulation.nspecies", str(n))
        # Sync each species' fields
        for i in range(n):
            keys = get_species_keys(i, show_adv)
            sync_inputs(input, state, keys)
