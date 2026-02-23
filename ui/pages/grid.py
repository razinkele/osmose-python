"""Grid configuration page."""

import plotly.graph_objects as go
from shiny import ui, reactive
from shinywidgets import output_widget, render_plotly

from osmose.schema.grid import GRID_FIELDS
from ui.components.param_form import render_field
from ui.state import sync_inputs

GRID_GLOBAL_KEYS: list[str] = [f.key_pattern for f in GRID_FIELDS if not f.indexed]


def make_grid_preview(
    ul_lat: float,
    ul_lon: float,
    lr_lat: float,
    lr_lon: float,
    nx: int = 0,
    ny: int = 0,
) -> go.Figure:
    """Create a plotly figure showing the grid extent.

    Args:
        ul_lat: Upper-left latitude.
        ul_lon: Upper-left longitude.
        lr_lat: Lower-right latitude.
        lr_lon: Lower-right longitude.
        nx: Number of columns (optional).
        ny: Number of rows (optional).

    Returns:
        A Plotly Figure with the grid extent drawn on a world map.
    """
    fig = go.Figure()
    # Draw rectangle for grid extent
    lats = [ul_lat, ul_lat, lr_lat, lr_lat, ul_lat]
    lons = [ul_lon, lr_lon, lr_lon, ul_lon, ul_lon]
    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="lines",
            line=dict(width=3, color="#e67e22"),
            name="Grid extent",
        )
    )
    fig.update_geos(
        projection_type="natural earth",
        showcoastlines=True,
        showland=True,
        landcolor="#2c3e50",
        oceancolor="#1a252f",
    )
    title = f"Grid: {ny}x{nx}" if nx and ny else "Grid Extent"
    fig.update_layout(title=title, template="plotly_dark", height=400)
    return fig


def grid_ui():
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

    return ui.layout_columns(
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
            output_widget("grid_preview"),
        ),
        col_widths=[6, 6],
    )


def grid_server(input, output, session, state):
    @render_plotly
    def grid_preview():
        ul_lat = float(input.grid_upleft_lat() or 0)
        ul_lon = float(input.grid_upleft_lon() or 0)
        lr_lat = float(input.grid_lowright_lat() or 0)
        lr_lon = float(input.grid_lowright_lon() or 0)
        nx = int(input.grid_ncolumn() or 0)
        ny = int(input.grid_nline() or 0)
        return make_grid_preview(ul_lat, ul_lon, lr_lat, lr_lon, nx, ny)

    @reactive.effect
    def sync_grid_inputs():
        sync_inputs(input, state, GRID_GLOBAL_KEYS)
