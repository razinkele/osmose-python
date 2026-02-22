"""Tests for calibration page helper functions."""

import numpy as np
import plotly.graph_objects as go

from ui.pages.calibration import (
    get_calibratable_params,
    make_convergence_chart,
    make_pareto_chart,
    make_sensitivity_chart,
)


def test_get_calibratable_params():
    from ui.state import REGISTRY

    params = get_calibratable_params(REGISTRY, n_species=3)
    assert len(params) > 0
    # Should include growth K for all 3 species
    keys = [p["key"] for p in params]
    assert "species.k.sp0" in keys
    assert "species.k.sp1" in keys
    assert "species.k.sp2" in keys


def test_make_convergence_chart():
    history = [10.0, 5.0, 3.0, 2.0, 1.5]
    fig = make_convergence_chart(history)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_make_convergence_chart_empty():
    fig = make_convergence_chart([])
    assert isinstance(fig, go.Figure)


def test_make_pareto_chart():
    F = np.array([[1.0, 2.0], [0.5, 3.0], [0.8, 1.5]])
    fig = make_pareto_chart(F, ["Biomass RMSE", "Diet Distance"])
    assert isinstance(fig, go.Figure)


def test_make_sensitivity_chart():
    result = {
        "param_names": ["k", "linf", "mort"],
        "S1": np.array([0.5, 0.3, 0.2]),
        "ST": np.array([0.6, 0.4, 0.3]),
    }
    fig = make_sensitivity_chart(result)
    assert isinstance(fig, go.Figure)
