"""The vendored Powell evaluates exactly what scipy 1.7.3's Powell evaluates.

``l2co_optimizers._src.scipy17_powell`` copies scipy 1.7.3's bounded Powell
for ``"mlsl"`` (its module docstring says why). The fixture holds the points
the real scipy 1.7.3 evaluated, recorded by
``tests/reference/scipy173_powell/make_fixture.py``; it includes calls with
``maxfev=0``, which current scipy refuses.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import Bounds

from l2co_optimizers._src.scipy17_powell import minimize_powell

__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"

FIXTURE = json.loads(
    (
        Path(__file__).parent / "data" / "scipy173_powell_points.json"
    ).read_text()
)


def _rastrigin(x):
    return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def _rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


FUNCTIONS = {"rastrigin": _rastrigin, "rosenbrock": _rosenbrock}


@pytest.mark.parametrize(
    "case",
    FIXTURE["cases"],
    ids=[
        f"{c['function']}-{len(c['x0'])}d-maxfev{c['maxfev']}"
        for c in FIXTURE["cases"]
    ],
)
def test_evaluates_the_points_scipy_1_7_3_evaluates(case):
    f = FUNCTIONS[case["function"]]
    points = []
    result = minimize_powell(
        lambda x: (points.append(x.copy()), f(x))[1],
        np.asarray(case["x0"]),
        Bounds(np.asarray(case["lb"]), np.asarray(case["ub"])),
        ftol=1e-8,
        maxfev=case["maxfev"],
    )
    np.testing.assert_array_equal(np.asarray(points), case["points"])
    assert result.nfev == case["nfev"]
    np.testing.assert_array_equal(result.x, case["x"])
    assert result.fun == case["fun"]


def test_a_zero_budget_still_runs_one_sweep():
    # The behaviour "mlsl" depends on: scipy >= 1.8 raises here.
    points = []
    minimize_powell(
        lambda x: (points.append(x.copy()), _rastrigin(x))[1],
        np.array([1.3, -2.2]),
        Bounds(-5 * np.ones(2), 5 * np.ones(2)),
        ftol=1e-8,
        maxfev=0,
    )
    assert len(points) > 1
