"""Tests for the ``"mlsl"`` registry entry, a port of the Zenodo MLSL.

The entry ports ``dynas/mlsl.py`` from Zenodo 10.5281/zenodo.6458266,
vendored under ``tests/reference/zenodo_mlsl``. Two properties are checked
at the registry seam:

1. driven with the same random stream, the entry evaluates exactly the
   points the vendored reference evaluates, in the same order;
2. a run makes exactly ``n_iterations`` evaluations, each billed one.
"""

#                                                                       Modules
# =============================================================================

# Standard
from types import SimpleNamespace

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# Local
from l2co_optimizers import BatchState
from l2co_optimizers._src.mapping import optimizer_mapping
from l2co_optimizers._src.mlsl import mlsl_rng
from l2co_optimizers._src.scipy17_powell import minimize_powell

from .reference.zenodo_mlsl import mlsl as reference_mlsl
from .toy_problems import quadratic_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

BOX = (-5.0, 5.0)


def _sphere(x, **_):
    return jnp.sum((x - 1.3) ** 2)


def _rastrigin(x, **_):
    z = x - 0.7
    return 10.0 * z.size + jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z))


def _counting(loss_fn, calls):
    """``loss_fn`` that appends every point it is evaluated at."""

    def counted(x, **kwargs):
        jax.debug.callback(lambda v: calls.append(np.asarray(v)), x)
        return loss_fn(x, **kwargs)

    return counted


def _run_entry(
    loss_fn, dim, n_iterations, key, bounded=BOX, **hyperparameters
):
    """One ``"mlsl"`` realization through ``run``, jitted."""
    update_class = optimizer_mapping("mlsl")(
        **hyperparameters,
        **quadratic_problem(jnp.zeros(dim), loss_fn),
        opt_hash=7,
        bounded=bounded,
        stop_fn=None,
    )
    params = jnp.zeros((1, dim))
    dataset = {}
    batch_state = BatchState.init(dataset=dataset, batch_size=None, key=key)
    opt_state = update_class.init_state(params, batch_state, dataset, key)
    return eqx.filter_jit(update_class.run)(
        opt_state=opt_state,
        params=params,
        batch_state=batch_state,
        dataset=dataset,
        key=key,
        n_iterations=n_iterations,
        verbose=False,
    )


class _ReferenceFunction:
    """The new-IOH function interface the vendored reference reads.

    It evaluates the same jitted loss the entry evaluates, on the same
    dtype, and records every point it is called at.
    """

    def __init__(self, loss_fn, dim):
        self._loss = jax.jit(loss_fn)
        self.meta_data = SimpleNamespace(n_variables=dim)
        self.constraint = SimpleNamespace(
            lb=np.full(dim, BOX[0]), ub=np.full(dim, BOX[1])
        )
        self.state = SimpleNamespace(evaluations=0)
        self.points = []

    def __call__(self, x):
        # Like the entry's driver, evaluate a point outside the box at its
        # projection (``mlsl`` module docstring, deviation 7).
        x = np.clip(x, self.constraint.lb, self.constraint.ub)
        self.points.append(np.array(x, dtype=np.float64))
        self.state.evaluations += 1
        return float(self._loss(jnp.asarray(x)))


def _scipy17_minimize(fun, x0, method, bounds, options):
    """``scipy.optimize.minimize`` as the Zenodo runs saw it (scipy 1.7)."""
    assert method == "Powell"
    return minimize_powell(fun, x0, bounds, **options)


def _run_reference(loss_fn, dim, budget, key, **hyperparameters):
    func = _ReferenceFunction(loss_fn, dim)
    algorithm = reference_mlsl.MLSL(func, seed=mlsl_rng(key).get_state())
    algorithm.set_hyperparams(hyperparameters)
    algorithm.set_params({"budget": budget})
    algorithm.set_stopping_criteria(lambda: func.state.evaluations >= budget)
    algorithm.run()
    return np.asarray(func.points)


@pytest.fixture(autouse=True)
def _reference_on_scipy17_powell(monkeypatch):
    # The reference imports ``minimize`` from scipy; the Zenodo runs used
    # scipy 1.7, whose Powell runs one sweep even with ``maxfev=0``, where
    # current scipy raises (``mlsl`` module docstring, deviation 6).
    monkeypatch.setattr(reference_mlsl, "minimize", _scipy17_minimize)


# =============================================================================
#                                              1. parity with the reference


# With the default zeta the local-search pool runs out in the first
# iteration, so the default cases cover many spent-pool local searches
# (one Powell sweep each). A large zeta starts only one local search per
# iteration, so those runs cover more iterations.
@pytest.mark.parametrize(
    ("loss_fn", "dim", "budget", "hyperparameters"),
    [
        (_sphere, 2, 600, {}),
        (_rastrigin, 2, 600, {}),
        (_rastrigin, 5, 1500, {}),
        (_sphere, 2, 1000, {"zeta": 1000.0}),
        (_rastrigin, 2, 1000, {"zeta": 1000.0}),
    ],
    ids=[
        "sphere-2d",
        "rastrigin-2d",
        "rastrigin-5d",
        "sphere-2d-5-iterations",
        "rastrigin-2d-5-iterations",
    ],
)
def test_evaluates_the_points_the_zenodo_reference_evaluates(
    loss_fn, dim, budget, hyperparameters
):
    key = jr.key(3)
    reference = _run_reference(loss_fn, dim, budget, key, **hyperparameters)
    calls = []
    _run_entry(_counting(loss_fn, calls), dim, budget, key, **hyperparameters)
    jax.effects_barrier()
    ours = np.asarray(calls)
    assert ours.shape == (budget, dim)
    # The reference checks its budget only between iterations, so it may
    # overrun it; the entry stops at the budget. Over the whole budget the
    # points must agree.
    assert len(reference) >= budget
    np.testing.assert_allclose(ours, reference[:budget], rtol=1e-6, atol=1e-6)


# =============================================================================
#                                                               2. billing


@pytest.mark.parametrize(
    ("n_iterations", "hyperparameters"),
    [(700, {}), (250, {"zeta": 1000.0})],
    ids=["past-the-local-pool", "mid-iteration"],
)
def test_makes_exactly_the_budget_and_bills_each_evaluation(
    n_iterations, hyperparameters
):
    calls = []
    out = _run_entry(
        _counting(_rastrigin, calls),
        2,
        n_iterations,
        jr.key(5),
        **hyperparameters,
    )
    jax.effects_barrier()
    history = out[5]
    assert len(calls) == n_iterations
    assert int(history.cursor) == n_iterations
    np.testing.assert_array_equal(np.asarray(history.fevals), 1)
    np.testing.assert_array_equal(np.asarray(history.iterations), 1)
    np.testing.assert_allclose(
        np.asarray(history.output_min),
        [float(_rastrigin(jnp.asarray(c))) for c in calls],
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "bounded",
    [(None, None), (-5.0, None), (-jnp.inf, 5.0)],
    ids=["unbounded", "no-upper", "infinite-lower"],
)
def test_refuses_a_task_without_a_finite_box(bounded):
    with pytest.raises(ValueError, match="needs finite bounds"):
        _run_entry(_sphere, 2, 10, jr.key(0), bounded=bounded)
