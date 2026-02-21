# osmose/calibration/objectives.py
"""Objective functions for OSMOSE calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def biomass_rmse(
    simulated: pd.DataFrame, observed: pd.DataFrame, species: str | None = None
) -> float:
    """Root mean square error of biomass time series.

    Args:
        simulated: DataFrame with 'time' and 'biomass' columns (and optionally 'species').
        observed: DataFrame with 'time' and 'biomass' columns (and optionally 'species').
        species: If specified, filter to this species.

    Returns:
        RMSE value.
    """
    if species:
        simulated = simulated[simulated["species"] == species]
        observed = observed[observed["species"] == species]

    # Merge on time to align
    merged = pd.merge(simulated, observed, on="time", suffixes=("_sim", "_obs"))
    if merged.empty:
        return float("inf")

    diff = merged["biomass_sim"] - merged["biomass_obs"]
    return float(np.sqrt(np.mean(diff**2)))


def abundance_rmse(
    simulated: pd.DataFrame, observed: pd.DataFrame, species: str | None = None
) -> float:
    """RMSE for abundance time series."""
    if species:
        simulated = simulated[simulated["species"] == species]
        observed = observed[observed["species"] == species]

    merged = pd.merge(simulated, observed, on="time", suffixes=("_sim", "_obs"))
    if merged.empty:
        return float("inf")

    diff = merged["abundance_sim"] - merged["abundance_obs"]
    return float(np.sqrt(np.mean(diff**2)))


def diet_distance(simulated: pd.DataFrame, observed: pd.DataFrame) -> float:
    """Frobenius norm distance between diet composition matrices.

    Both DataFrames should be square matrices with predator rows and prey columns.
    """
    sim_vals = simulated.select_dtypes(include=[np.number]).values
    obs_vals = observed.select_dtypes(include=[np.number]).values

    if sim_vals.shape != obs_vals.shape:
        return float("inf")

    return float(np.linalg.norm(sim_vals - obs_vals, "fro"))


def normalized_rmse(simulated: np.ndarray, observed: np.ndarray) -> float:
    """RMSE normalized by the mean of observed values."""
    obs_mean = np.mean(observed)
    if obs_mean == 0:
        return float("inf")
    rmse = float(np.sqrt(np.mean((simulated - observed) ** 2)))
    return rmse / obs_mean
