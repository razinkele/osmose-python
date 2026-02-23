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


def test_run_surrogate_workflow():
    """Test SurrogateCalibrator end-to-end: generate samples, fit, predict, find_optimum."""
    import numpy as np

    from osmose.calibration.surrogate import SurrogateCalibrator

    # Two free params, 1 objective
    bounds = [(0.1, 1.0), (10.0, 100.0)]
    cal = SurrogateCalibrator(param_bounds=bounds, n_objectives=1)

    # Step 1: generate samples
    n_samples = 20
    samples = cal.generate_samples(n_samples=n_samples)
    assert samples.shape == (n_samples, 2)
    # All samples within bounds
    assert np.all(samples[:, 0] >= 0.1) and np.all(samples[:, 0] <= 1.0)
    assert np.all(samples[:, 1] >= 10.0) and np.all(samples[:, 1] <= 100.0)

    # Step 2: simulate OSMOSE evaluations (use a known function)
    Y = np.sum(samples**2, axis=1)  # simple quadratic

    # Step 3: fit GP model
    cal.fit(samples, Y)
    assert cal._is_fitted

    # Step 4: predict on new points
    test_X = cal.generate_samples(n_samples=5, seed=99)
    means, stds = cal.predict(test_X)
    assert means.shape == (5, 1)
    assert stds.shape == (5, 1)
    assert np.all(stds >= 0)

    # Step 5: find optimum
    result = cal.find_optimum(n_candidates=500)
    assert "params" in result
    assert "predicted_objectives" in result
    assert "predicted_uncertainty" in result
    assert result["params"].shape == (2,)


def test_run_surrogate_workflow_multi_objective():
    """Test SurrogateCalibrator with multiple objectives."""
    import numpy as np

    from osmose.calibration.surrogate import SurrogateCalibrator

    bounds = [(0.0, 5.0), (0.0, 5.0)]
    cal = SurrogateCalibrator(param_bounds=bounds, n_objectives=2)

    samples = cal.generate_samples(n_samples=30)
    Y = np.column_stack(
        [
            np.sum(samples**2, axis=1),
            np.sum((samples - 3) ** 2, axis=1),
        ]
    )

    cal.fit(samples, Y)
    assert cal.n_objectives == 2

    means, stds = cal.predict(samples[:5])
    assert means.shape == (5, 2)

    result = cal.find_optimum(n_candidates=500)
    assert result["predicted_objectives"].shape == (2,)
