"""Environmental forcing / LTL configuration page."""

from shiny import ui, reactive, render
from osmose.schema.ltl import LTL_FIELDS
from osmose.schema.bioenergetics import BIOENERGETICS_FIELDS
from ui.components.param_form import render_field, render_category


def forcing_ui():
    # Separate global LTL settings from per-resource fields
    global_ltl = [f for f in LTL_FIELDS if not f.indexed]
    resource_fields = [f for f in LTL_FIELDS if f.indexed]

    # Temperature/environmental fields from bioenergetics
    temp_fields = [f for f in BIOENERGETICS_FIELDS if f.key_pattern.startswith("temperature.")]

    return ui.page_fluid(
        ui.layout_columns(
            ui.card(
                ui.card_header("Lower Trophic Level (Plankton)"),
                ui.h5("Global LTL Settings"),
                *[render_field(f) for f in global_ltl],
                ui.hr(),
                ui.input_numeric("n_resources", "Number of resource groups", value=3, min=0, max=20),
                ui.output_ui("resource_panels"),
            ),
            ui.card(
                ui.card_header("Environmental Forcing"),
                ui.h5("Temperature"),
                *[render_field(f) for f in temp_fields if not f.advanced],
                ui.hr(),
                ui.p("Upload NetCDF forcing data for spatially-varying temperature, oxygen, or other environmental variables."),
            ),
            col_widths=[7, 5],
        ),
    )


def forcing_server(input, output, session):
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
