"""Write ``tests/data/nl_shade_rsp_metabox_fixture.json``.

Runs the vendored MetaBox v2 ``NL_SHADE_RSP`` (``rldas_optimizer.py``
beside this file) the way RL-DAS drives it -- one shared ``Population``,
``step`` called for one 2500-evaluation period at a time -- on sphere,
Rastrigin and Rosenbrock in 10 D, box ``[-5, 5]``, budget 20000 (MetaBox's
bbob ``maxFEs``). Each seed seeds both MetaBox's ``RandomState`` and the
global ``np.random`` stream (its Cauchy draws use the latter). The
fixture stores every seed's best cost and the medians; the slow tests in
``tests/test_nl_shade_rsp.py`` compare ``nl_shade_rsp_metabox`` (and
``nl_shade_rsp``) with them.

Run from the repository root::

    uv run --no-sync python tests/reference/metabox_rldas/make_fixture_nl_shade_rsp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rldas_optimizer import NL_SHADE_RSP, Population  # noqa: E402

OUT = HERE.parents[1] / "data" / "nl_shade_rsp_metabox_fixture.json"

DIM = 10
MAX_FEVALS = 20000
PERIOD = 2500  # RL-DAS switch interval
BOUNDS = (-5.0, 5.0)
SEEDS = list(range(20))


def sphere(x):
    return np.sum(x**2, axis=-1)


def rastrigin(x):
    return 10.0 * x.shape[-1] + np.sum(
        x**2 - 10.0 * np.cos(2 * np.pi * x), axis=-1
    )


def rosenbrock(x):
    return np.sum(
        100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1.0 - x[..., :-1]) ** 2,
        axis=-1,
    )


FUNCTIONS = {"sphere": sphere, "rastrigin": rastrigin, "rosenbrock": rosenbrock}


class Problem:
    """The attributes MetaBox's ``Population`` and DEs read."""

    def __init__(self, fn):
        self.fn = fn
        self.lb, self.ub = BOUNDS
        self.optimum = 0.0  # all three minima are 0

    def eval(self, x):
        return self.fn(np.atleast_2d(x))


def best_cost(fn, seed: int) -> float:
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    problem = Problem(fn)
    population = Population(DIM, rng, problem)
    population.initialize_costs(problem)
    optimizer = NL_SHADE_RSP(DIM, rng)
    fevals = population.NP  # the starting population is billed
    while fevals < MAX_FEVALS and population.gbest >= optimizer.error:
        population, fevals = optimizer.step(
            population, problem, fevals, fevals + PERIOD, MAX_FEVALS
        )
    return float(population.gbest)


def main() -> None:
    functions = {}
    for name, fn in FUNCTIONS.items():
        finals = [best_cost(fn, seed) for seed in SEEDS]
        functions[name] = {"best": finals, "median": float(np.median(finals))}
        print(name, functions[name]["median"])
    OUT.write_text(
        json.dumps(
            {
                "reference": "MetaBox v2 rldas_optimizer.NL_SHADE_RSP",
                "dim": DIM,
                "max_fevals": MAX_FEVALS,
                "period": PERIOD,
                "bounds": list(BOUNDS),
                "seeds": SEEDS,
                "functions": functions,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
