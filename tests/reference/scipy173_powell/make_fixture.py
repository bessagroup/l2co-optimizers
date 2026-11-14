"""Record the points scipy 1.7.3's Powell evaluates, for ``test_scipy17_powell``.

Run it in a throwaway environment with the real scipy 1.7.3, from the repo
root (Python 3.9; scipy 1.7.3 has no wheels for newer Pythons)::

    uv run --no-project --python 3.9 --with scipy==1.7.3 --with numpy==1.21.6 \
        python tests/reference/scipy173_powell/make_fixture.py

It writes ``tests/data/scipy173_powell_points.json``.
"""

import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, minimize

assert scipy.__version__ == "1.7.3", scipy.__version__


def rastrigin(x):
    return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


FUNCTIONS = {"rastrigin": rastrigin, "rosenbrock": rosenbrock}

cases = []
rng = np.random.RandomState(0)
for name, f in FUNCTIONS.items():
    for dim in (2, 5):
        lb, ub = -5 * np.ones(dim), 5 * np.ones(dim)
        for maxfev in (0, 3, 26.0, None):
            x0 = rng.uniform(-5, 5, dim)
            points = []
            result = minimize(
                lambda x: (points.append(x.copy()), f(x))[1],
                x0,
                method="Powell",
                bounds=Bounds(lb, ub),
                options={"ftol": 1e-8, "maxfev": maxfev},
            )
            cases.append(
                {
                    "function": name,
                    "x0": x0.tolist(),
                    "lb": lb.tolist(),
                    "ub": ub.tolist(),
                    "maxfev": maxfev,
                    "points": np.asarray(points).tolist(),
                    "nfev": int(result.nfev),
                    "x": np.asarray(result.x).tolist(),
                    "fun": float(result.fun),
                }
            )

out = Path("tests/data/scipy173_powell_points.json")
out.write_text(json.dumps({"scipy": scipy.__version__, "cases": cases}))
print(f"wrote {len(cases)} cases to {out}")
