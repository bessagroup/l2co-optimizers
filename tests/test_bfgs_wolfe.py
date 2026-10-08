"""``bfgs`` with ``linesearch="wolfe"``: scipy's BFGS, stepped (ADR 0005).

The properties checked:

1. in float64 it evaluates the points scipy's BFGS evaluates, so it
   needs scipy's number of evaluations (issue #27: 63 instead of 473 on
   a 40-D ellipsoid of condition 1e6);
2. one evaluation per step, billed as one, every point inside the box,
   and the gradient recorded at every point;
3. where scipy would stop, the run goes on without a NaN: past
   convergence, from a stationary start, across a region where the loss
   is NaN, on a noisy loss and in float32;
4. the Armijo settings are refused with it, and the default stays
   Armijo;
5. it runs through the fused multi-realization driver.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optimistix as optx
import pytest

# Local
from l2co_optimizers import BatchState
from l2co_optimizers._src.mapping import optimizer_mapping
from l2co_optimizers._src.optimistix_implementations import (
    WolfeBFGS,
    WolfeSearchState,
)

from .toy_problems import quadratic_problem

scipy_optimize = pytest.importorskip("scipy.optimize")

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def _rosenbrock(x, **_):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def _ellipsoid(x, **_):
    """Condition number 1e6."""
    w = 1e6 ** (jnp.arange(x.shape[0]) / (x.shape[0] - 1))
    return jnp.sum(w * x**2)


def _wavy(x, **_):
    return jnp.sum((x - 0.5) ** 2) + 0.3 * jnp.sum(jnp.sin(3.0 * x))


def _build(loss_fn, model, bounded=(None, None), pass_rng=False, **hp):
    hp = {"linesearch": "wolfe", **hp}
    return optimizer_mapping("bfgs")(
        **hp,
        **quadratic_problem(model, loss_fn, pass_rng),
        opt_hash=1,
        bounded=bounded,
        stop_fn=None,
    )


def _drive(update_class, x0, n_steps, sample=None):
    """Step from ``x0``; return the carries and histories."""
    params, key = x0[None], jr.key(0)
    carry = (params, update_class.init_fn(params, key), key)
    step = eqx.filter_jit(update_class.step_fn)
    carries, histories = [], []
    for _ in range(n_steps):
        carry, history = step(carry, sample or {})
        carries.append(carry)
        histories.append(history)
    return carries, histories


def _scipy_points(loss_fn, x0):
    """Every point scipy's BFGS evaluates, value and gradient together."""
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    points = []

    def fun(x):
        points.append(np.array(x))
        f, g = value_and_grad(jnp.asarray(x))
        return float(f), np.asarray(g)

    scipy_optimize.minimize(
        fun,
        np.asarray(x0),
        jac=True,
        method="BFGS",
        options={"gtol": 0.0, "maxiter": 10**6},
    )
    return np.array(points)


def _evals_to(histories, target):
    best = np.minimum.accumulate([float(h.loss[0]) for h in histories])
    hit = np.nonzero(best <= target)[0]
    return int(hit[0]) + 1 if len(hit) else None


# =============================================================================
#                                                   1. scipy's evaluations


@pytest.mark.parametrize(
    ("loss_fn", "x0"),
    [
        (_rosenbrock, [-1.2, 1.0]),
        (_ellipsoid, np.linspace(1.0, 2.0, 10)),
        (_wavy, [2.0, -1.0, 0.5]),
    ],
    ids=["rosenbrock-2d", "ellipsoid-10d", "wavy-3d"],
)
def test_evaluates_the_points_scipy_evaluates(loss_fn, x0):
    """Until scipy stops; rounding in the BFGS update is the only gap."""
    with jax.enable_x64(True):
        x0 = jnp.asarray(x0, jnp.float64)
        reference = _scipy_points(loss_fn, x0)
        _, histories = _drive(_build(loss_fn, x0), x0, len(reference))
        points = np.array([np.asarray(h.params[0]) for h in histories])
    np.testing.assert_allclose(points, reference, rtol=1e-6, atol=1e-6)


def test_closes_the_ellipsoid_gap_of_issue_27():
    """473 evaluations with Armijo; scipy's 63 with Wolfe."""
    with jax.enable_x64(True):
        x0 = jnp.linspace(1.0, 2.0, 40)
        reference = _scipy_points(_ellipsoid, x0)
        best = np.minimum.accumulate(
            [float(_ellipsoid(jnp.asarray(p))) for p in reference]
        )
        scipy_evals = int(np.nonzero(best <= 1e-8)[0][0]) + 1
        _, wolfe = _drive(_build(_ellipsoid, x0), x0, 100)
        _, armijo = _drive(
            _build(_ellipsoid, x0, linesearch="armijo"), x0, 100
        )
    assert _evals_to(wolfe, 1e-8) == scipy_evals
    assert _evals_to(armijo, 1e-8) is None  # not within 100


# =============================================================================
#                                     2. billing, the box, recorded gradients


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_one_loss_call_per_step(jit):
    calls = []

    def counted(x, **_):
        jax.debug.callback(lambda v: calls.append(1), x)
        return _wavy(x)

    update_class = _build(counted, jnp.zeros(3))
    params, key = jnp.full((1, 3), 2.0), jr.key(0)
    carry = (params, update_class.init_fn(params, key), key)
    step = (
        eqx.filter_jit(update_class.step_fn) if jit else update_class.step_fn
    )
    for _ in range(25):
        calls.clear()
        carry, history = step(carry, {})
        jax.effects_barrier()
        assert int(history.fevals) == len(calls) == 1
    assert update_class.fevals_bound == 1


def test_every_point_lies_in_the_box():
    bounded, seen = (-1.0, 1.0), []

    def recorded(x, **_):
        jax.debug.callback(lambda v: seen.append(np.asarray(v)), x)
        return _wavy(x)

    carries, histories = _drive(
        _build(recorded, jnp.zeros(3), bounded=bounded),
        jnp.array([3.0, -2.5, 4.0]),
        30,
    )
    jax.effects_barrier()
    for points in (
        np.stack(seen),
        np.concatenate([np.asarray(h.params) for h in histories]),
        np.concatenate([np.asarray(c[0]) for c in carries]),
    ):
        assert points.min() >= bounded[0] and points.max() <= bounded[1]


def test_records_the_gradient_at_every_point():
    """Rejected trials included: the search needs it there anyway."""
    carries, histories = _drive(
        _build(_wavy, jnp.zeros(3)), jnp.array([4.0, -3.0, 2.0]), 25
    )
    accepted = [bool(c[1].solver_state.search_state.accept) for c in carries]
    assert any(accepted) and not all(accepted)
    for history in histories:
        np.testing.assert_allclose(
            history.grads[0],
            jax.grad(_wavy)(history.params[0]),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            history.loss[0], _wavy(history.params[0]), rtol=1e-6
        )


# =============================================================================
#                                          3. where scipy would have stopped


def test_keeps_evaluating_past_convergence():
    with jax.enable_x64(True):
        x0 = jnp.array([-1.2, 1.0])
        carries, histories = _drive(_build(_rosenbrock, x0), x0, 300)
    losses = np.array([float(h.loss[0]) for h in histories])
    assert np.isfinite(losses).all()
    assert all(np.isfinite(np.asarray(c[0])).all() for c in carries)
    assert losses.min() < 1e-20
    np.testing.assert_allclose(carries[-1][0][0], [1.0, 1.0], atol=1e-10)


def test_a_stationary_start_is_re_evaluated():
    """A zero gradient: not even steepest descent goes downhill."""
    x0 = jnp.full(3, 0.5)
    carries, histories = _drive(
        _build(lambda x, **_: jnp.sum((x - 0.5) ** 2), x0), x0, 5
    )
    for carry, history in zip(carries, histories, strict=True):
        np.testing.assert_array_equal(history.params[0], x0)
        assert float(history.loss[0]) == 0.0
        assert bool(carry[1].solver_state.search_state.idle)


def test_steps_back_from_a_region_where_the_loss_is_nan():
    """The minimum at 3 lies beyond a NaN hole on (0.5, 1.5)."""

    def holed(x, **_):
        hole = (x[0] > 0.5) & (x[0] < 1.5)
        return jnp.where(hole, jnp.nan, (x[0] - 3.0) ** 2)

    with jax.enable_x64(True):
        x0 = jnp.zeros(1)
        carries, histories = _drive(_build(holed, x0), x0, 40)
    losses = np.array([float(h.loss[0]) for h in histories])
    assert np.isnan(losses).any()  # the hole was hit ...
    assert all(np.isfinite(np.asarray(c[0])).all() for c in carries)
    assert np.nanmin(losses) < 1e-12  # ... and stepped over


def test_a_noisy_loss_runs_without_nans():
    def noisy(x, *, key, **_):
        return _wavy(x) + 0.05 * jr.normal(key)

    x0 = jnp.array([3.0, -2.0, 1.0])
    carries, histories = _drive(_build(noisy, x0, pass_rng=True), x0, 100)
    losses = np.array([float(h.loss[0]) for h in histories])
    assert np.isfinite(losses).all()
    assert all(np.isfinite(np.asarray(c[0])).all() for c in carries)
    assert float(_wavy(carries[-1][0][0])) < float(_wavy(x0)) - 1.0


def test_float32_rosenbrock():
    """scipy's step bounds overflow float32; they are narrowed."""
    x0 = jnp.array([-1.2, 1.0], jnp.float32)
    _, histories = _drive(_build(_rosenbrock, x0), x0, 200)
    losses = np.array([float(h.loss[0]) for h in histories])
    assert np.isfinite(losses).all()
    assert losses.min() < 1e-6


@pytest.mark.parametrize("use_inverse", [True, False])
def test_both_forms_of_the_estimate_converge(use_inverse):
    with jax.enable_x64(True):
        x0 = jnp.array([-1.2, 1.0])
        _, histories = _drive(
            _build(_rosenbrock, x0, use_inverse=use_inverse), x0, 80
        )
    assert _evals_to(histories, 1e-10) is not None


# =============================================================================
#                                                       4. hyperparameters


@pytest.mark.parametrize(
    "armijo", [{"step_init": 2.0}, {"slope": 0.2}, {"decrease_factor": 0.3}]
)
def test_armijo_settings_are_refused(armijo):
    with pytest.raises(ValueError, match="Armijo"):
        _build(_wavy, jnp.zeros(3), **armijo)


def test_unknown_linesearch_is_refused():
    with pytest.raises(ValueError, match="linesearch"):
        _build(_wavy, jnp.zeros(3), linesearch="zoom")


def test_the_default_is_still_armijo():
    x0 = jnp.array([2.0, -1.0, 0.5])
    default = optimizer_mapping("bfgs")(
        **quadratic_problem(x0, _wavy), opt_hash=1, bounded=None, stop_fn=None
    )
    _, by_default = _drive(default, x0, 20)
    _, armijo = _drive(_build(_wavy, x0, linesearch="armijo"), x0, 20)
    for a, b in zip(by_default, armijo, strict=True):
        np.testing.assert_array_equal(a.params, b.params)


# =============================================================================
#                                                    5. driver, upstream pin


def test_batch_run_end_to_end():
    dim, n_realizations, n_iterations = 4, 3, 40
    update_class = _build(_wavy, jnp.zeros(dim), bounded=(-2.0, 2.0))
    assert not update_class.sequential_realizations
    key = jr.key(0)
    batch_state = BatchState.init(dataset={}, batch_size=None, key=key)
    params = jr.normal(key, (n_realizations, 1, dim))
    keys = jr.split(key, n_realizations)
    opt_state = eqx.filter_vmap(
        update_class.init_state, in_axes=(0, None, None, 0)
    )(params, batch_state, {}, keys)
    out = eqx.filter_jit(update_class.batch_run)(
        opt_state=opt_state,
        params=params,
        batch_state=batch_state,
        dataset={},
        key=keys,
        n_iterations=n_iterations,
        verbose=False,
    )
    _, best_params, best_loss, *_, history = out
    assert np.isfinite(best_loss).all()
    np.testing.assert_allclose(
        best_loss, jax.vmap(_wavy)(best_params), rtol=1e-5
    )
    assert history.output_min.shape == (n_realizations, n_iterations)


def test_upstream_internals_it_relies_on():
    """optimistix 0.1 internals ``WolfeBFGS.step`` reads or replaces."""
    y = jnp.zeros(3)
    solver = WolfeBFGS()
    state = solver.init(
        lambda x, args: (_wavy(x), None),
        y,
        None,
        {},
        jax.ShapeDtypeStruct((), y.dtype),
        None,
        frozenset(),
    )
    for field in (
        "first_step",
        "y_eval",
        "search_state",
        "f_info",
        "aux",
        "descent_state",
        "num_accepted_steps",
    ):
        assert hasattr(state, field), field
    assert isinstance(state.search_state, WolfeSearchState)
    assert isinstance(solver.descent, optx.NewtonDescent)
    assert hasattr(state.descent_state, "newton")
    assert hasattr(state.f_info.hessian_inv, "pytree")
