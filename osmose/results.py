"""Read OSMOSE simulation output files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


class OsmoseResults:
    """Read and query OSMOSE simulation outputs.

    OSMOSE writes output as CSV and/or NetCDF files. This class provides
    a unified interface to access biomass, abundance, yield, diet, and
    mortality data.
    """

    def __init__(self, output_dir: Path, prefix: str = "osm"):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self._nc_cache: dict[str, xr.Dataset] = {}

    def list_outputs(self) -> list[str]:
        """List all output files in the output directory."""
        files = []
        for f in sorted(self.output_dir.iterdir()):
            if f.suffix in (".csv", ".nc"):
                files.append(f.name)
        return files

    def read_csv(self, pattern: str) -> dict[str, pd.DataFrame]:
        """Read CSV output files matching a glob pattern.

        Returns dict mapping filename to DataFrame.
        """
        result = {}
        for f in sorted(self.output_dir.glob(pattern)):
            result[f.stem] = pd.read_csv(f)
        return result

    def read_netcdf(self, filename: str) -> xr.Dataset:
        """Read a NetCDF output file, with caching."""
        if filename not in self._nc_cache:
            path = self.output_dir / filename
            self._nc_cache[filename] = xr.open_dataset(path)
        return self._nc_cache[filename]

    def biomass(self, species: str | None = None) -> pd.DataFrame:
        """Read biomass time series.

        Returns DataFrame with columns: time, species, biomass.
        Reads from CSV files matching *biomass*.csv pattern.
        """
        return self._read_species_output("biomass", species)

    def abundance(self, species: str | None = None) -> pd.DataFrame:
        """Read abundance time series."""
        return self._read_species_output("abundance", species)

    def yield_biomass(self, species: str | None = None) -> pd.DataFrame:
        """Read yield/catch biomass time series."""
        return self._read_species_output("yield", species)

    def mortality(self, species: str | None = None) -> pd.DataFrame:
        """Read mortality breakdown."""
        return self._read_species_output("mortality", species)

    def diet_matrix(self, species: str | None = None) -> pd.DataFrame:
        """Read diet composition matrix."""
        return self._read_species_output("dietMatrix", species)

    def mean_size(self, species: str | None = None) -> pd.DataFrame:
        """Read mean size time series."""
        return self._read_species_output("meanSize", species)

    def mean_trophic_level(self, species: str | None = None) -> pd.DataFrame:
        """Read mean trophic level time series."""
        return self._read_species_output("meanTL", species)

    def spatial_biomass(self, filename: str) -> xr.Dataset:
        """Read spatial (gridded) biomass output from NetCDF."""
        return self.read_netcdf(filename)

    def _read_species_output(self, output_type: str, species: str | None) -> pd.DataFrame:
        """Read CSV output files for a given output type.

        Files are expected to match: {prefix}_{output_type}*.csv
        Each file's data gets a 'species' column derived from the filename.
        """
        pattern = f"{self.prefix}_{output_type}*.csv"
        frames = []
        for filepath in sorted(self.output_dir.glob(pattern)):
            df = pd.read_csv(filepath)
            # Extract species name from filename (e.g., osm_biomass_Anchovy.csv -> Anchovy)
            parts = filepath.stem.split("_", 2)
            sp_name = parts[2] if len(parts) > 2 else filepath.stem
            df["species"] = sp_name
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        if species:
            combined = combined[combined["species"] == species]
        return combined

    def close(self) -> None:
        """Close any cached NetCDF datasets."""
        for ds in self._nc_cache.values():
            ds.close()
        self._nc_cache.clear()
