"""Tests for osmose.results – OSMOSE output file reader."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osmose.results import OsmoseResults


@pytest.fixture
def output_dir(tmp_path):
    """Create a fake OSMOSE output directory with test files."""
    # Create biomass CSVs
    for sp_name in ["Anchovy", "Sardine"]:
        df = pd.DataFrame(
            {
                "time": range(10),
                "biomass": np.random.rand(10) * 1000,
            }
        )
        df.to_csv(tmp_path / f"osm_biomass_{sp_name}.csv", index=False)

    # Create abundance CSVs
    for sp_name in ["Anchovy", "Sardine"]:
        df = pd.DataFrame(
            {
                "time": range(10),
                "abundance": np.random.randint(1000, 100000, 10),
            }
        )
        df.to_csv(tmp_path / f"osm_abundance_{sp_name}.csv", index=False)

    # Create yield CSV
    df = pd.DataFrame({"time": range(10), "yield": np.random.rand(10) * 100})
    df.to_csv(tmp_path / "osm_yield_Anchovy.csv", index=False)

    # Create a spatial NetCDF
    ds = xr.Dataset(
        {
            "biomass": xr.DataArray(
                np.random.rand(10, 5, 5),
                dims=["time", "lat", "lon"],
                coords={
                    "time": range(10),
                    "lat": np.linspace(55, 65, 5),
                    "lon": np.linspace(20, 30, 5),
                },
            )
        }
    )
    ds.to_netcdf(tmp_path / "osm_spatial_biomass.nc")
    ds.close()

    return tmp_path


def test_list_outputs(output_dir):
    results = OsmoseResults(output_dir)
    files = results.list_outputs()
    assert len(files) > 0
    assert any("biomass" in f for f in files)


def test_biomass_all_species(output_dir):
    results = OsmoseResults(output_dir)
    df = results.biomass()
    assert not df.empty
    assert "species" in df.columns
    assert set(df["species"].unique()) == {"Anchovy", "Sardine"}


def test_biomass_single_species(output_dir):
    results = OsmoseResults(output_dir)
    df = results.biomass("Anchovy")
    assert not df.empty
    assert set(df["species"].unique()) == {"Anchovy"}


def test_abundance(output_dir):
    results = OsmoseResults(output_dir)
    df = results.abundance()
    assert not df.empty
    assert "abundance" in df.columns


def test_yield_biomass(output_dir):
    results = OsmoseResults(output_dir)
    df = results.yield_biomass()
    assert not df.empty


def test_missing_output_returns_empty(output_dir):
    results = OsmoseResults(output_dir)
    df = results.diet_matrix()
    assert df.empty


def test_spatial_netcdf(output_dir):
    results = OsmoseResults(output_dir)
    ds = results.spatial_biomass("osm_spatial_biomass.nc")
    assert "biomass" in ds.data_vars
    assert ds["biomass"].dims == ("time", "lat", "lon")
    results.close()


def test_read_csv_pattern(output_dir):
    results = OsmoseResults(output_dir)
    csvs = results.read_csv("osm_biomass_*.csv")
    assert len(csvs) == 2


def test_close_clears_cache(output_dir):
    results = OsmoseResults(output_dir)
    results.read_netcdf("osm_spatial_biomass.nc")
    assert len(results._nc_cache) == 1
    results.close()
    assert len(results._nc_cache) == 0
