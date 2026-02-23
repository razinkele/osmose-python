"""Tests for calibration page handler helper functions."""

from shiny import reactive

from ui.pages.calibration import collect_selected_params, build_free_params


def test_collect_selected_params():
    """Should return keys where the corresponding checkbox is True."""
    from ui.state import AppState

    state = AppState()
    with reactive.isolate():
        state.reset_to_defaults()

        # Simulate checkboxes: only species.k.sp0 is checked
        class FakeInput:
            def __getattr__(self, name):
                if name == "cal_param_species_k_sp0":
                    return lambda: True
                return lambda: False

        params = collect_selected_params(FakeInput(), state)
        keys = [p["key"] for p in params]
        assert "species.k.sp0" in keys


def test_collect_selected_params_empty():
    """Should return empty list when nothing is checked."""
    from ui.state import AppState

    state = AppState()
    with reactive.isolate():
        state.reset_to_defaults()

        class FakeInput:
            def __getattr__(self, name):
                return lambda: False

        params = collect_selected_params(FakeInput(), state)
        assert params == []


def test_build_free_params():
    """Should create FreeParameter objects from selected param dicts."""
    from osmose.calibration.problem import FreeParameter

    selected = [
        {"key": "species.k.sp0", "lower": 0.1, "upper": 1.0},
        {"key": "species.k.sp1", "lower": 0.1, "upper": 1.0},
    ]
    free_params = build_free_params(selected)
    assert len(free_params) == 2
    assert isinstance(free_params[0], FreeParameter)
    assert free_params[0].key == "species.k.sp0"
    assert free_params[0].lower_bound == 0.1
    assert free_params[0].upper_bound == 1.0
