from pathlib import Path

import numpy as np
from osmose.calibration.problem import FreeParameter, OsmoseCalibrationProblem


def test_free_parameter():
    fp = FreeParameter(key="species.k.sp0", lower_bound=0.1, upper_bound=0.5)
    assert fp.key == "species.k.sp0"
    assert fp.transform == "linear"


def test_problem_dimensions():
    params = [
        FreeParameter("species.k.sp0", 0.1, 0.5),
        FreeParameter("species.linf.sp0", 10, 200),
    ]
    problem = OsmoseCalibrationProblem(
        free_params=params,
        objective_fns=[lambda r: 0.0, lambda r: 0.0],
        base_config_path=Path("/tmp/fake"),
        jar_path=Path("/tmp/fake.jar"),
        work_dir=Path("/tmp/work"),
    )
    assert problem.n_var == 2
    assert problem.n_obj == 2
    assert np.array_equal(problem.xl, np.array([0.1, 10]))
    assert np.array_equal(problem.xu, np.array([0.5, 200]))
