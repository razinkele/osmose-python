import numpy as np
import pandas as pd
import pytest
from osmose.calibration.objectives import biomass_rmse, diet_distance, normalized_rmse


def test_biomass_rmse_identical():
    df = pd.DataFrame({"time": range(10), "biomass": [100.0] * 10})
    assert biomass_rmse(df, df) == 0.0


def test_biomass_rmse_different():
    sim = pd.DataFrame({"time": range(10), "biomass": [100.0] * 10})
    obs = pd.DataFrame({"time": range(10), "biomass": [110.0] * 10})
    assert biomass_rmse(sim, obs) == pytest.approx(10.0)


def test_biomass_rmse_with_species_filter():
    sim = pd.DataFrame({"time": [0, 0], "biomass": [100, 200], "species": ["A", "B"]})
    obs = pd.DataFrame({"time": [0, 0], "biomass": [110, 200], "species": ["A", "B"]})
    assert biomass_rmse(sim, obs, species="A") == pytest.approx(10.0)
    assert biomass_rmse(sim, obs, species="B") == pytest.approx(0.0)


def test_biomass_rmse_no_overlap():
    sim = pd.DataFrame({"time": [0, 1], "biomass": [100, 200]})
    obs = pd.DataFrame({"time": [5, 6], "biomass": [100, 200]})
    assert biomass_rmse(sim, obs) == float("inf")


def test_diet_distance_identical():
    df = pd.DataFrame({"prey1": [0.5, 0.3], "prey2": [0.5, 0.7]})
    assert diet_distance(df, df) == 0.0


def test_diet_distance_different():
    sim = pd.DataFrame({"prey1": [1.0, 0.0], "prey2": [0.0, 1.0]})
    obs = pd.DataFrame({"prey1": [0.0, 0.0], "prey2": [0.0, 0.0]})
    result = diet_distance(sim, obs)
    assert result == pytest.approx(np.sqrt(2.0))


def test_normalized_rmse():
    sim = np.array([100, 110, 90])
    obs = np.array([100, 100, 100])
    result = normalized_rmse(sim, obs)
    expected = np.sqrt(np.mean([0, 100, 100])) / 100
    assert result == pytest.approx(expected)
