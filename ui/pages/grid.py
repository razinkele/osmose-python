"""Grid configuration page."""

from shiny import ui, render
from osmose.schema.grid import GRID_FIELDS
from ui.components.param_form import render_field


def grid_ui():
    # Separate grid type selector from other fields
    grid_type_field = next((f for f in GRID_FIELDS if "classname" in f.key_pattern), None)
    regular_fields = [
        f
        for f in GRID_FIELDS
        if (
            f.key_pattern.startswith("grid.n")
            or f.key_pattern.startswith("grid.up")
            or f.key_pattern.startswith("grid.low")
        )
        and "netcdf" not in f.key_pattern
    ]
    netcdf_fields = [
        f for f in GRID_FIELDS if "netcdf" in f.key_pattern or f.key_pattern.startswith("grid.var")
    ]

    return ui.page_fluid(
        ui.layout_columns(
            ui.card(
                ui.card_header("Grid Type"),
                render_field(grid_type_field) if grid_type_field else ui.div(),
                ui.hr(),
                ui.h5("Regular Grid Settings"),
                *[render_field(f) for f in regular_fields],
                ui.hr(),
                ui.h5("NetCDF Grid Settings"),
                *[render_field(f) for f in netcdf_fields if not f.advanced],
            ),
            ui.card(
                ui.card_header("Grid Preview"),
                ui.p("Upload a grid mask or configure coordinates to see a preview."),
                ui.output_ui("grid_preview_placeholder"),
            ),
            col_widths=[6, 6],
        ),
    )


def grid_server(input, output, session):
    @render.ui
    def grid_preview_placeholder():
        return ui.div(
            ui.tags.div(
                "Grid visualization will appear here once coordinates are set.",
                style="height: 400px; display: flex; align-items: center; justify-content: center; border: 1px dashed #4e5d6c; border-radius: 8px; color: #999;",
            )
        )
