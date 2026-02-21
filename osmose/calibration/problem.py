# osmose/calibration/problem.py
"""OSMOSE calibration as a pymoo optimization problem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from pymoo.core.problem import Problem


@dataclass
class FreeParameter:
    """A parameter to optimize during calibration."""

    key: str  # OSMOSE parameter key
    lower_bound: float
    upper_bound: float
    transform: str = "linear"  # "linear" or "log"


class OsmoseCalibrationProblem(Problem):
    """Multi-objective optimization problem for OSMOSE.

    Each evaluation:
    1. Maps candidate parameter vector to OSMOSE config overrides
    2. Runs OSMOSE with those overrides
    3. Reads results and computes objective values
    """

    def __init__(
        self,
        free_params: list[FreeParameter],
        objective_fns: list[Callable],
        base_config_path: Path,
        jar_path: Path,
        work_dir: Path,
        java_cmd: str = "java",
    ):
        self.free_params = free_params
        self.objective_fns = objective_fns
        self.base_config_path = base_config_path
        self.jar_path = jar_path
        self.work_dir = work_dir
        self.java_cmd = java_cmd

        xl = np.array([fp.lower_bound for fp in free_params])
        xu = np.array([fp.upper_bound for fp in free_params])

        super().__init__(
            n_var=len(free_params),
            n_obj=len(objective_fns),
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate a population of candidates.

        X has shape (pop_size, n_var). Each row is a candidate.
        """
        F = np.full((X.shape[0], self.n_obj), np.inf)

        for i, params in enumerate(X):
            overrides = {}
            for j, fp in enumerate(self.free_params):
                val = params[j]
                if fp.transform == "log":
                    val = 10**val
                overrides[fp.key] = str(val)

            try:
                objectives = self._run_single(overrides, run_id=i)
                for k, obj_val in enumerate(objectives):
                    F[i, k] = obj_val
            except Exception:
                pass  # Leave as inf

        out["F"] = F

    def _run_single(self, overrides: dict[str, str], run_id: int) -> list[float]:
        """Run OSMOSE synchronously with overrides and return objective values.

        Uses subprocess (synchronous) since pymoo evaluates in a loop.
        """
        import subprocess

        # Create isolated output directory
        run_dir = self.work_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = run_dir / "output"

        cmd = [self.java_cmd, "-jar", str(self.jar_path), str(self.base_config_path)]
        cmd.append(f"-Poutput.dir.path={output_dir}")
        for key, value in overrides.items():
            cmd.append(f"-P{key}={value}")

        result = subprocess.run(cmd, capture_output=True, timeout=3600)

        if result.returncode != 0:
            return [float("inf")] * self.n_obj

        # Compute objectives
        from osmose.results import OsmoseResults

        results = OsmoseResults(output_dir)
        obj_values = []
        for fn in self.objective_fns:
            obj_values.append(fn(results))

        return obj_values
