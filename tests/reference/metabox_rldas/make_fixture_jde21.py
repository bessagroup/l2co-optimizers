"""Write ``tests/data/jde21_metabox_fixture.json`` from MetaBox's ``JDE21``.

Runs the vendored MetaBox v2 ``JDE21`` (``rldas_optimizer.py``) on
three 10-D functions in the box ``[-5, 5]`` for ``SEEDS`` seeds each and
records every run's final best cost, plus the medians. One MetaBox
``step`` call spans the whole budget (``FEs_end = MaxFEs``), which is a
standalone run: one opening reverse sort, ``age`` counted over the run.
Each run seeds both the ``RandomState`` MetaBox is given and the global
``np.random`` stream.

Run from the repository root::

    uv run --no-sync python tests/reference/metabox_rldas/make_fixture_jde21.py

``tests/test_jde21.py`` defines the same functions in JAX; keep the two
in step.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rldas_optimizer import JDE21, Population  # noqa: E402

DIM = 10
MAX_FEVALS = 10_000
SEEDS = 20
LOW, HIGH = -5.0, 5.0
OUT = HERE.parents[1] / "data" / "jde21_metabox_fixture.json"


def sphere(x):
    return np.sum((x - 0.5) ** 2, axis=-1)


def rastrigin(x):
    z = x - 0.5
    return 10.0 * z.shape[-1] + np.sum(
        z**2 - 10.0 * np.cos(2 * np.pi * z), axis=-1
    )


def rosenbrock(x):
    return np.sum(
        100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1.0 - x[..., :-1]) ** 2,
        axis=-1,
    )


FUNCTIONS = {"sphere": sphere, "rastrigin": rastrigin, "rosenbrock": rosenbrock}


class Problem:
    """The attributes MetaBox's ``Population`` and ``JDE21`` read."""

    def __init__(self, fn):
        self.fn = fn
        self.lb = LOW
        self.ub = HIGH
        self.optimum = None

    def eval(self, x):
        return self.fn(np.atleast_2d(x))


def run(fn, seed: int) -> float:
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    problem = Problem(fn)
    population = Population(DIM, rng, problem)
    population.initialize_costs(problem)
    fevals = population.NP
    JDE21(DIM, rng).step(population, problem, fevals, MAX_FEVALS, MAX_FEVALS)
    return float(min(population.gbest, np.min(population.cost)))


def main() -> None:
    start = time.time()
    results = {}
    for name, fn in FUNCTIONS.items():
        finals = [run(fn, seed) for seed in range(SEEDS)]
        results[name] = {"median": float(np.median(finals)), "finals": finals}
        print(f"{name}: median {results[name]['median']:.4g}")
    payload = {
        "source": "MetaBox v2 JDE21 (tests/reference/metabox_rldas)",
        "generator": "tests/reference/metabox_rldas/make_fixture_jde21.py",
        "dim": DIM,
        "max_fevals": MAX_FEVALS,
        "bounds": [LOW, HIGH],
        "seeds": SEEDS,
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
