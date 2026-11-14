"""Write ``tests/data/madde_metabox_fixture.json`` from the vendored MadDE.

Runs MetaBox v2's RL-DAS ``MadDE`` on its shared ``Population`` (non-linear
170 -> 30 reduction) for 20 seeds on the 10-D sphere and Rastrigin
(optimum shifted to 1.5) and Rosenbrock functions in [-5, 5]^10, a budget of 10 000 evaluations, and
stores each run's final best cost and the medians. The slow parity test
in ``tests/test_madde.py`` compares ``madde_metabox`` with these numbers.

Run from the repository root::

    uv run --no-sync python -m tests.reference.metabox_rldas.make_fixture_madde

Both random streams are seeded: the ``RandomState`` handed to
``Population`` and ``MadDE``, and the global ``np.random`` that
``scipy.stats.cauchy.rvs`` in ``Population.choose_F_Cr`` draws from.
"""

#                                                                       Modules
# =============================================================================

# Standard
import json
import warnings
from pathlib import Path

# Third-party
import numpy as np

# Local
from tests.reference.metabox_rldas.rldas_optimizer import MadDE, Population

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

DIM = 10
MAX_FEVALS = 10_000
SEEDS = list(range(20))
BOUND = 5.0
#: Sphere and Rastrigin are shifted off the origin: MadDE's weighted
#: rand-to-qbest mutation ``F x_r1 + ...`` pulls towards the origin, which
#: would make an origin optimum unrepresentatively easy.
SHIFT = 1.5
OUT = Path(__file__).resolve().parents[2] / "data" / "madde_metabox_fixture.json"


def sphere(x: np.ndarray) -> np.ndarray:
    z = x - SHIFT
    return np.sum(z**2, axis=-1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    z = x - SHIFT
    return 10.0 * z.shape[-1] + np.sum(
        z**2 - 10.0 * np.cos(2.0 * np.pi * z), axis=-1
    )


def rosenbrock(x: np.ndarray) -> np.ndarray:
    return np.sum(
        100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1.0 - x[..., :-1]) ** 2,
        axis=-1,
    )


FUNCTIONS = {"sphere": sphere, "rastrigin": rastrigin, "rosenbrock": rosenbrock}


class Problem:
    """The attributes MetaBox's ``Population`` and ``MadDE`` read."""

    def __init__(self, fn):
        self.fn = fn
        self.lb = -BOUND
        self.ub = BOUND
        self.optimum = None
        self.fevals = 0

    def eval(self, x):
        x = np.asarray(x)
        self.fevals += len(np.atleast_2d(x))
        return self.fn(x)


def run(fn, seed: int) -> tuple[float, int]:
    """One MetaBox MadDE run over the whole budget.

    Returns its best cost and the evaluations it used. MetaBox's loop
    (``while FEs < MaxFEs: ... FEs += NP``) lets the last generation run
    past ``MaxFEs``, so the count can exceed ``MAX_FEVALS``; the parity
    test gives the port the same count.
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    problem = Problem(fn)
    population = Population(DIM, rng, problem)
    population.initialize_costs(problem)
    optimizer = MadDE(DIM, rng)
    fevals = population.NP
    population, _ = optimizer.step(
        population, problem, fevals, MAX_FEVALS, MAX_FEVALS
    )
    return float(population.gbest), problem.fevals


def main() -> None:
    results = {}
    with warnings.catch_warnings():
        # np.mean of an empty strategy group in the p_m update.
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, fn in FUNCTIONS.items():
            finals, used = zip(*(run(fn, seed) for seed in SEEDS))
            results[name] = {
                "final_best": list(finals),
                "median": float(np.median(finals)),
                "fevals_used": list(used),
            }
    fixture = {
        "source": "tests/reference/metabox_rldas/rldas_optimizer.py MadDE",
        "dim": DIM,
        "max_fevals": MAX_FEVALS,
        "bounds": [-BOUND, BOUND],
        "shift": SHIFT,
        "seeds": SEEDS,
        "error_stop": 1e-8,
        "functions": results,
    }
    OUT.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
