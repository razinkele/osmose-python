"""Tests for grid page preview chart."""

import plotly.graph_objects as go

from ui.pages.grid import make_grid_preview


def test_grid_preview_with_coords():
    fig = make_grid_preview(
        ul_lat=48.0,
        ul_lon=-5.0,
        lr_lat=43.0,
        lr_lon=0.0,
    )
    assert isinstance(fig, go.Figure)
    # Should have a rectangle shape
    assert len(fig.data) > 0


def test_grid_preview_zero_coords():
    fig = make_grid_preview(ul_lat=0, ul_lon=0, lr_lat=0, lr_lon=0)
    assert isinstance(fig, go.Figure)


def test_grid_preview_with_dimensions():
    fig = make_grid_preview(
        ul_lat=48.0,
        ul_lon=-5.0,
        lr_lat=43.0,
        lr_lon=0.0,
        nx=30,
        ny=30,
    )
    assert isinstance(fig, go.Figure)
