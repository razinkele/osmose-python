"""Fishing configuration page."""

from shiny import ui, reactive, render
from osmose.schema.fishing import FISHING_FIELDS
from ui.components.param_form import render_field


def fishing_ui():
    global_fields = [f for f in FISHING_FIELDS if not f.indexed]
    fishery_fields = [f for f in FISHING_FIELDS if f.indexed and "fsh" in f.key_pattern]
    mpa_fields = [f for f in FISHING_FIELDS if f.indexed and "mpa" in f.key_pattern]

    return ui.page_fluid(
        ui.layout_columns(
            ui.card(
                ui.card_header("Fisheries Module"),
                *[render_field(f) for f in global_fields],
                ui.hr(),
                ui.input_numeric("n_fisheries", "Number of fisheries", value=1, min=0, max=20),
                ui.output_ui("fishery_panels"),
            ),
            ui.card(
                ui.card_header("Marine Protected Areas"),
                ui.input_numeric("n_mpas", "Number of MPAs", value=0, min=0, max=10),
                ui.output_ui("mpa_panels"),
            ),
            col_widths=[8, 4],
        ),
    )


def fishing_server(input, output, session):
    @render.ui
    def fishery_panels():
        n = input.n_fisheries()
        fishery_fields = [f for f in FISHING_FIELDS if f.indexed and "fsh" in f.key_pattern]
        panels = []
        for i in range(n):
            card = ui.card(
                ui.card_header(f"Fishery {i}"),
                *[render_field(f, species_idx=i) for f in fishery_fields],
            )
            panels.append(card)
        return ui.div(*panels)

    @render.ui
    def mpa_panels():
        n = input.n_mpas()
        mpa_fields = [f for f in FISHING_FIELDS if f.indexed and "mpa" in f.key_pattern]
        panels = []
        for i in range(n):
            card = ui.card(
                ui.card_header(f"MPA {i}"),
                *[render_field(f, species_idx=i) for f in mpa_fields],
            )
            panels.append(card)
        return ui.div(*panels)
