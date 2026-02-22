"""Tests for ui.state -- shared application state."""

from pathlib import Path

from shiny import reactive

from ui.state import AppState


def test_appstate_initial_config_is_empty():
    state = AppState()
    with reactive.isolate():
        assert state.config.get() == {}


def test_appstate_initial_output_dir_is_none():
    state = AppState()
    with reactive.isolate():
        assert state.output_dir.get() is None


def test_appstate_initial_run_result_is_none():
    state = AppState()
    with reactive.isolate():
        assert state.run_result.get() is None


def test_appstate_scenarios_dir_default():
    state = AppState()
    assert state.scenarios_dir == Path("data/scenarios")


def test_appstate_custom_scenarios_dir():
    state = AppState(scenarios_dir=Path("/tmp/my_scenarios"))
    assert state.scenarios_dir == Path("/tmp/my_scenarios")


def test_appstate_config_update():
    state = AppState()
    with reactive.isolate():
        state.config.set({"simulation.nspecies": "3"})
        assert state.config.get() == {"simulation.nspecies": "3"}


def test_appstate_update_config_key():
    state = AppState()
    with reactive.isolate():
        state.config.set({"a": "1"})
        state.update_config("b", "2")
        assert state.config.get() == {"a": "1", "b": "2"}


def test_appstate_update_config_key_overwrites():
    state = AppState()
    with reactive.isolate():
        state.config.set({"a": "1"})
        state.update_config("a", "99")
        assert state.config.get() == {"a": "99"}


def test_appstate_reset_to_defaults():
    state = AppState()
    with reactive.isolate():
        state.reset_to_defaults()
        cfg = state.config.get()
        # Should have simulation params
        assert "simulation.nspecies" in cfg
        assert cfg["simulation.nspecies"] == "3"
        # Should have grid params
        assert "grid.ncolumn" in cfg
        # Should have species-indexed params expanded for 3 species
        assert "species.linf.sp0" in cfg
