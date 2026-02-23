"""Environmental forcing / LTL configuration page."""

from shiny import ui, reactive, render

from osmose.schema.bioenergetics import BIOENERGETICS_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from ui.components.param_form import render_field
from ui.state import sync_inputs

FORCING_GLOBAL_KEYS: list[str] = [f.key_pattern for f in LTL_FIELDS if not f.indexed]
_TEMP_KEYS: list[str] = [
    f.key_pattern
    for f in BIOENERGETICS_FIELDS
    if f.key_pattern.startswith("temperature.") and not f.indexed
]


def forcing_ui():
    global_ltl = [f for f in LTL_FIELDS if not f.indexed]
    temp_fields = [f for f in BIOENERGETICS_FIELDS if f.key_pattern.startswith("temperature.")]

    return ui.page_fluid(
        ui.layout_columns(
            ui.card(
                ui.card_header("Lower Trophic Level (Plankton)"),
                ui.h5("Global LTL Settings"),
                *[render_field(f) for f in global_ltl],
                ui.hr(),
                ui.input_numeric(
                    "n_resources", "Number of resource groups", value=3, min=0, max=20
                ),
                ui.output_ui("resource_panels"),
            ),
            ui.card(
                ui.card_header("Environmental Forcing"),
                ui.h5("Temperature"),
                *[render_field(f) for f in temp_fields if not f.advanced],
                ui.hr(),
                ui.p(
                    "Upload NetCDF forcing data for spatially-varying temperature, "
                    "oxygen, or other environmental variables."
                ),
            ),
            col_widths=[7, 5],
        ),
    )


def forcing_server(input, output, session, state):
    @render.ui
    def resource_panels():
        n = input.n_resources()
        panels = []
        for i in range(n):
            resource_fields = [f for f in LTL_FIELDS if f.indexed]
            card = ui.card(
                ui.card_header(f"Resource Group {i}"),
                *[render_field(f, species_idx=i) for f in resource_fields],
            )
            panels.append(card)
        return ui.div(*panels)

    @reactive.effect
    def sync_forcing_inputs():
        sync_inputs(input, state, FORCING_GLOBAL_KEYS + _TEMP_KEYS)

    @reactive.effect
    def sync_resource_inputs():
        n = input.n_resources()
        indexed_fields = [f for f in LTL_FIELDS if f.indexed]
        for i in range(n):
            keys = [f.resolve_key(i) for f in indexed_fields]
            sync_inputs(input, state, keys)
