"""Unit tests for the scipy registry entries (ADRs 0002, 0003, 0006).

``"cobyqa"``, ``"powell"``, ``"tnc"``, ``"trustkrylov"``, ``"slsqp"``,
``"trustconstr"`` and ``"lbfgsb"`` hand a whole run to
:func:`scipy.optimize.minimize` inside one host callback. The properties
checked:

1. the wrapper is transparent -- its evaluations are the ones a direct
   ``scipy.optimize.minimize`` call makes, until scipy stops;
2. every method improves on its starting point (a scipy failure would
   otherwise hide behind "stay at the best point");
3. one iteration is one real evaluation, billed one, for exactly
   ``n_iterations``;
4. after scipy stops, the final point is re-evaluated to the budget;
   after a failure, the best point is; a method that stops evaluating
   is stopped;
5. every evaluated point lies in ``bounded``;
6. stochastic and minibatched losses get a fresh key and batch per
   evaluation, drawn as the run loop draws them;
7. the hyperparameters reach scipy;
8. ``stop_fn`` raises, ``step`` raises, none of them unpacks into
   switching parts, and realizations run one at a time;
9. the family and its (absent) transfer spec.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import scipy.optimize as so

# Local
from l2co_optimizers import (
    FAMILY_DERIVATIVE_FREE,
    FAMILY_GRADIENT,
    BatchState,
    OptimizationStep,
    optimizer_parts,
)
from l2co_optimizers._src.core.state_transfer import transfer_spec_for
from l2co_optimizers._src.mapping import optimizer_mapping, optimizers
from l2co_optimizers._src.scipy_implementations import (
    _COBYQA_FINAL_TR_RADIUS,
    _LBFGSB_FTOL,
    _LBFGSB_GTOL,
    _POWELL_FTOL,
    _POWELL_XTOL,
    _SLSQP_FTOL,
    _TNC_FTOL,
    _TNC_GTOL,
    _TNC_XTOL,
    _TRUSTCONSTR_GTOL,
    _TRUSTCONSTR_XTOL,
    _TRUSTKRYLOV_GTOL,
    _UNREACHABLE,
    SCIPY_OPTIMIZERS,
    ScipyUpdateClass,
    _scipy_update,
    cobyqa_update,
    lbfgsb_update,
    powell_update,
    slsqp_update,
    tnc_update,
    trustconstr_update,
    trustkrylov_update,
)

from .toy_problems import quadratic_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

NAMES = (
    "cobyqa",
    "powell",
    "tnc",
    "trustkrylov",
    "slsqp",
    "trustconstr",
    "lbfgsb",
)
NATIVE_BOUNDS = ("cobyqa", "powell", "tnc", "slsqp", "trustconstr", "lbfgsb")
DIM = 3


def _wavy(x, **_):
    return jnp.sum((x - 0.5) ** 2) + 0.3 * jnp.sum(jnp.sin(3.0 * x))


def _build(name, loss_fn=_wavy, dim=DIM, bounded=(None, None), **hp):
    return optimizer_mapping(name)(
        **hp,
        **quadratic_problem(jnp.zeros(dim), loss_fn),
        opt_hash=7,
        bounded=bounded,
        stop_fn=None,
    )


def _x0(dim=DIM, seed=1):
    return jr.normal(jr.key(seed), (1, dim))


def _run(
    update_class,
    n_iterations,
    params=None,
    dataset=None,
    batch_size=None,
    key=None,
):
    """One realization through ``run``, jitted like a databank run."""
    key = jr.key(0) if key is None else key
    dataset = {} if dataset is None else dataset
    params = _x0() if params is None else params
    batch_state = BatchState.init(
        dataset=dataset, batch_size=batch_size, key=key
    )
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


def _losses(out):
    return np.asarray(out[5].output_min)


def _counting(loss_fn, calls):
    """``loss_fn`` that appends every point it is evaluated at."""

    def counted(x, **kwargs):
        jax.debug.callback(lambda v: calls.append(np.asarray(v)), x)
        return loss_fn(x, **kwargs)

    return counted


# =============================================================================
#                                                                      Registry


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("cobyqa", cobyqa_update),
        ("powell", powell_update),
        ("tnc", tnc_update),
        ("trustkrylov", trustkrylov_update),
        ("slsqp", slsqp_update),
        ("trustconstr", trustconstr_update),
        ("lbfgsb", lbfgsb_update),
    ],
)
def test_registered_under_its_bare_name(name, factory):
    assert name in SCIPY_OPTIMIZERS
    assert optimizers[name].func is factory


@pytest.mark.parametrize(
    ("spelling", "name"),
    [
        ("trust-krylov", "trustkrylov"),
        ("trust-constr", "trustconstr"),
        ("L-BFGS-B", "lbfgsb"),
    ],
)
def test_resolves_from_its_scipy_spelling(spelling, name):
    assert optimizer_mapping(spelling) is optimizers[name]


# =============================================================================
#                                                1. transparent wrapper


def _direct_scipy(name, loss_fn, x0):
    """The evaluations a direct ``minimize`` call makes, with our options."""
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    value = jax.jit(loss_fn)
    seen = []

    def f(x):
        v = float(value(jnp.asarray(x)))
        seen.append(v)
        return v

    def fg(x):
        v, g = value_and_grad(jnp.asarray(x))
        seen.append(float(v))
        return float(v), np.asarray(g, dtype=np.float64)

    eval_eps = float(np.finfo(np.asarray(x0).dtype).eps)
    x0 = np.asarray(x0, dtype=np.float64)
    if name == "cobyqa":
        so.minimize(
            f,
            x0,
            method="COBYQA",
            options=dict(
                final_tr_radius=_COBYQA_FINAL_TR_RADIUS,
                maxfev=_UNREACHABLE,
                maxiter=_UNREACHABLE,
            ),
        )
    elif name == "powell":
        so.minimize(
            f,
            x0,
            method="Powell",
            options=dict(
                xtol=_POWELL_XTOL,
                ftol=_POWELL_FTOL,
                maxfev=_UNREACHABLE,
                maxiter=_UNREACHABLE,
            ),
        )
    elif name == "slsqp":
        so.minimize(
            fg,
            x0,
            jac=True,
            method="SLSQP",
            options=dict(ftol=_SLSQP_FTOL, maxiter=_UNREACHABLE),
        )
    elif name == "lbfgsb":
        so.minimize(
            fg,
            x0,
            jac=True,
            method="L-BFGS-B",
            options=dict(
                ftol=_LBFGSB_FTOL,
                gtol=_LBFGSB_GTOL,
                maxfun=_UNREACHABLE,
                maxiter=_UNREACHABLE,
            ),
        )
    elif name == "trustconstr":
        so.minimize(
            fg,
            x0,
            jac=True,
            hess=so.BFGS(),
            method="trust-constr",
            options=dict(
                gtol=_TRUSTCONSTR_GTOL,
                xtol=_TRUSTCONSTR_XTOL,
                maxiter=_UNREACHABLE,
            ),
        )
    elif name == "tnc":
        so.minimize(
            fg,
            x0,
            jac=True,
            method="TNC",
            options=dict(
                accuracy=eval_eps,
                ftol=_TNC_FTOL,
                xtol=_TNC_XTOL,
                gtol=_TNC_GTOL,
                maxfun=_UNREACHABLE,
            ),
        )
    else:
        # An independent re-statement of the finite-difference product.
        grads = {}

        def fg_cached(x):
            v, g = fg(x)
            grads[np.asarray(x, dtype=np.float64).tobytes()] = g
            return v, g

        def hessp(x, p):
            eps = (
                np.sqrt(eval_eps)
                * max(1.0, np.linalg.norm(x))
                / np.linalg.norm(p)
            )
            g0 = grads[np.asarray(x, dtype=np.float64).tobytes()]
            _, g1 = fg(x + eps * p)
            return (g1 - g0) / eps

        try:
            so.minimize(
                fg_cached,
                x0,
                jac=True,
                hessp=hessp,
                method="trust-krylov",
                options=dict(gtol=_TRUSTKRYLOV_GTOL, maxiter=_UNREACHABLE),
            )
        except Exception:  # noqa: BLE001 -- the wrapper stops here too
            pass
    return np.asarray(seen)


@pytest.mark.parametrize("name", NAMES)
def test_evaluations_match_a_direct_scipy_minimize(name):
    """Until scipy stops, the history is scipy's own evaluation sequence."""
    x0 = _x0()
    direct = _direct_scipy(name, _wavy, x0[0])
    n = 400
    ours = _losses(_run(_build(name), n, params=x0))
    k = min(len(direct), n)
    assert k > 10, f"{name} stopped after {k} evaluations"
    np.testing.assert_allclose(ours[:k], direct[:k], rtol=1e-6)


# =============================================================================
#                                                        2. makes progress


@pytest.mark.parametrize("name", NAMES)
def test_improves_on_its_starting_point(name):
    # Start far from the minimum, so a run that fails at once and stays
    # at its starting point cannot pass.
    out = _run(_build(name), 300, params=3.0 + _x0())
    first = _losses(out)[0]
    best = float(out[2])
    assert best < 0.2 * first


# =============================================================================
#                                                               3. billing


@pytest.mark.parametrize("name", NAMES)
def test_each_iteration_is_one_real_evaluation(name):
    calls = []
    n = 120
    out = _run(_build(name, _counting(_wavy, calls)), n)
    jax.effects_barrier()
    history = out[5]
    assert len(calls) == n
    assert int(history.cursor) == n
    np.testing.assert_array_equal(np.asarray(history.fevals), 1)
    np.testing.assert_array_equal(np.asarray(history.iterations), 1)
    np.testing.assert_array_equal(np.asarray(history.update_step), 7)
    # The recorded losses are the losses at the evaluated points.
    np.testing.assert_allclose(
        _losses(out),
        [float(_wavy(jnp.asarray(c))) for c in calls],
        rtol=1e-6,
    )


@pytest.mark.parametrize("name", NAMES)
def test_best_is_the_lowest_evaluation(name):
    out = _run(_build(name), 150)
    losses = _losses(out)
    assert float(out[2]) == pytest.approx(np.nanmin(losses), rel=1e-6)
    assert float(out[2]) == pytest.approx(float(_wavy(out[1])), rel=1e-5)
    assert out[0].shape == (1, DIM)
    assert out[1].shape == (DIM,)


# =============================================================================
#                                                       4. after scipy stops


@pytest.mark.parametrize(
    "name", ["cobyqa", "tnc", "trustkrylov", "trustconstr", "lbfgsb"]
)
def test_final_point_is_re_evaluated_until_the_budget(name):
    calls = []
    n = 2000
    out = _run(_build(name, _counting(_wavy, calls)), n)
    jax.effects_barrier()
    losses = _losses(out)
    # Every iteration was a real evaluation; the last run of identical
    # points is the idle phase at scipy's final point ...
    assert len(calls) == n
    moved = [i for i in range(n) if not np.array_equal(calls[i], calls[-1])]
    stopped = moved[-1] + 1
    # ... which began well before the budget ran out ...
    assert stopped < n // 2
    # ... and recorded that point's loss every time.
    np.testing.assert_array_equal(losses[stopped:], losses[-1])
    np.testing.assert_array_equal(np.asarray(out[0])[0], calls[-1])


def test_a_method_that_stops_evaluating_is_stopped(monkeypatch):
    """trust-constr at ``xtol = 0`` iterates forever without evaluating.

    Its trust radius reaches exactly zero, so every trial point is the
    current one, which scipy has already evaluated. The stall guard ends
    the run there, and the rest of the budget re-evaluates that point.
    """
    monkeypatch.setattr(
        "l2co_optimizers._src.scipy_implementations._TRUSTCONSTR_XTOL", 0.0
    )

    def sphere(x, **_):
        return jnp.sum((x - 0.5) ** 2)

    calls = []
    n = 300
    out = _run(_build("trustconstr", _counting(sphere, calls)), n)
    jax.effects_barrier()
    assert len(calls) == n
    moved = [i for i in range(n) if not np.array_equal(calls[i], calls[-1])]
    assert moved[-1] + 1 < n // 2
    assert float(out[2]) < 1e-10


def _failing(after, how):
    """A minimizer that evaluates ``after`` points, then fails."""

    def minimize(driver, x0, bounds):
        rng = np.random.default_rng(0)
        for _ in range(after):
            driver.value(x0 + rng.normal(size=x0.size))
        if how == "raise":
            raise RuntimeError("scipy blew up")
        if how == "nonfinite":
            driver.value(np.full(x0.size, np.nan))
        return np.full(x0.size, np.inf)  # "returned a non-finite x"

    return minimize


def _failing_update(minimize, loss_fn=_wavy):
    return _scipy_update(
        minimize,
        name="failing",
        family=FAMILY_GRADIENT,
        with_grad=False,
        **quadratic_problem(jnp.zeros(DIM), loss_fn),
        opt_hash=7,
        bounded=(None, None),
        stop_fn=None,
    )


@pytest.mark.parametrize("how", ["raise", "nonfinite", "returns_nonfinite"])
def test_failure_stays_at_the_best_point(how):
    calls = []
    n, after = 60, 12
    update_class = _failing_update(
        _failing(after, how), _counting(_wavy, calls)
    )
    out = _run(update_class, n)
    jax.effects_barrier()
    losses = _losses(out)
    best = np.argmin(losses[:after])
    # Nothing non-finite was ever evaluated; the rest re-evaluates the best.
    assert len(calls) == n
    assert all(np.all(np.isfinite(c)) for c in calls)
    np.testing.assert_array_equal(calls[after], calls[best])
    np.testing.assert_allclose(losses[after:], losses[best], rtol=1e-6)
    assert float(out[2]) == pytest.approx(losses[best], rel=1e-6)


def test_nan_loss_is_recorded_and_never_best():
    def sometimes_nan(x, **_):
        return jnp.where(x[0] > 0.0, jnp.nan, _wavy(x))

    def minimize(driver, x0, bounds):
        driver.value(np.array([-1.0, 0.0, 0.0]))
        driver.value(np.array([1.0, 0.0, 0.0]))  # NaN
        return np.array([-1.0, 0.0, 0.0])

    out = _run(_failing_update(minimize, sometimes_nan), 5)
    losses = _losses(out)
    assert np.isnan(losses[1])
    assert np.isfinite(float(out[2]))
    np.testing.assert_allclose(np.asarray(out[1]), [-1.0, 0.0, 0.0])


# =============================================================================
#                                                                 5. bounds


@pytest.mark.parametrize("name", NAMES)
def test_every_evaluated_point_lies_in_the_box(name):
    # The unconstrained minimum (near 0.5) lies outside the box.
    bounded = (-1.0, 0.2)
    calls = []
    params = 2.0 + _x0()  # start outside the box
    out = _run(
        _build(name, _counting(_wavy, calls), bounded=bounded),
        200,
        params=params,
    )
    jax.effects_barrier()
    for points in (np.stack(calls), np.asarray(out[0]), np.asarray(out[1])):
        assert points.min() >= bounded[0]
        assert points.max() <= bounded[1]


@pytest.mark.parametrize("name", NATIVE_BOUNDS)
def test_native_bounds_reach_the_boundary_optimum(name):
    # The box optimum of ``(x - 0.5)^2`` on [-1, 0.2] is x = 0.2.
    def sphere(x, **_):
        return jnp.sum((x - 0.5) ** 2)

    out = _run(_build(name, sphere, bounded=(-1.0, 0.2)), 300)
    np.testing.assert_allclose(np.asarray(out[1]), 0.2, atol=1e-3)


# =============================================================================
#                                       6. stochastic and minibatched losses


def test_each_evaluation_gets_a_fresh_key():
    """A noisy loss re-evaluated at one point varies: fresh keys."""

    def noisy(x, *, key, **_):
        return jnp.sum((x - 0.5) ** 2) + jr.normal(key)

    def minimize(driver, x0, bounds):
        return x0  # stop at once; idle re-evaluates x0 every time

    update_class = _scipy_update(
        minimize,
        name="noisy",
        family=FAMILY_GRADIENT,
        with_grad=False,
        **quadratic_problem(jnp.zeros(DIM), noisy, pass_rng=True),
        opt_hash=7,
        bounded=(None, None),
        stop_fn=None,
    )
    n = 50
    key = jr.key(5)
    params = _x0()
    losses = _losses(_run(update_class, n, params=params, key=key))

    # The keys UpdateClass.step would hand the loss: split, use the
    # second half, carry the first.
    expected = []
    for _ in range(n):
        key, eval_key = jr.split(key)
        expected.append(float(noisy(params[0], key=eval_key)))
    np.testing.assert_allclose(losses, expected, rtol=1e-6)
    assert np.unique(losses).size == n


def test_batches_are_drawn_as_the_run_loop_draws_them():
    """Each evaluation's batch is ``batch_state.next(key)`` on its key."""
    n_rows, batch_size, n = 10, 3, 25
    dataset = {"w": jnp.arange(n_rows, dtype=float)}

    def batch_loss(x, w):
        return jnp.sum((x - 0.5) ** 2) + jnp.sum(w)

    def minimize(driver, x0, bounds):
        return x0  # stop at once; idle re-evaluates x0 every time

    update_class = _scipy_update(
        minimize,
        name="batched",
        family=FAMILY_GRADIENT,
        with_grad=False,
        **quadratic_problem(jnp.zeros(DIM), batch_loss),
        opt_hash=7,
        bounded=(None, None),
        stop_fn=None,
    )
    key = jr.key(3)
    params = _x0()
    out = _run(
        update_class,
        n,
        params=params,
        dataset=dataset,
        batch_size=batch_size,
        key=key,
    )

    # The same draws, made the way UpdateClass.step makes them.
    batch_state = BatchState.init(
        dataset=dataset, batch_size=batch_size, key=key
    )
    expected = []
    for _ in range(n):
        idx, batch_state = batch_state.next(key)
        key, _ = jr.split(key)
        expected.append(float(batch_loss(params[0], dataset["w"][idx])))
    np.testing.assert_allclose(_losses(out), expected, rtol=1e-6)
    np.testing.assert_array_equal(out[4].pos, batch_state.pos)
    np.testing.assert_array_equal(out[4].perm, batch_state.perm)


# =============================================================================
#                                                        7. hyperparameters


@pytest.mark.parametrize(
    ("name", "hp"),
    [
        ("cobyqa", {"initial_tr_radius": 0.1}),
        # scipy's default is max(1, min(50, n / 2)), i.e. 1 at n = 3.
        ("tnc", {"max_cg_iterations": 3}),
        ("tnc", {"eta": 0.9}),
        ("tnc", {"stepmx": 0.01}),
        ("trustkrylov", {"initial_trust_radius": 0.05}),
        ("trustkrylov", {"max_trust_radius": 0.1}),
        ("trustconstr", {"initial_tr_radius": 0.05}),
        ("lbfgsb", {"maxcor": 1}),
        ("lbfgsb", {"maxls": 1}),
    ],
)
def test_hyperparameters_change_the_run(name, hp):
    # Far from the minimum, where steps and line searches matter.
    params = 3.0 + _x0()
    default = _losses(_run(_build(name), 60, params=params))
    changed = _losses(_run(_build(name, **hp), 60, params=params))
    assert not np.array_equal(default, changed)


@pytest.mark.parametrize(
    ("name", "hp", "option", "value"),
    [
        ("cobyqa", {"initial_tr_radius": 0.3}, "initial_tr_radius", 0.3),
        ("tnc", {"eta": 0.9}, "eta", 0.9),
        ("tnc", {"stepmx": 2.0}, "stepmx", 2.0),
        ("tnc", {"max_cg_iterations": 2}, "maxCGit", 2),
        ("tnc", {}, "maxCGit", -1),
        (
            "trustkrylov",
            {"initial_trust_radius": 0.3},
            "initial_trust_radius",
            0.3,
        ),
        ("trustkrylov", {"max_trust_radius": 5.0}, "max_trust_radius", 5.0),
        # Whether 0.2 changes a run depends on whether some step's
        # reduction ratio falls in [0.15, 0.2); here it only must arrive.
        ("trustkrylov", {"eta": 0.2}, "eta", 0.2),
        (
            "trustconstr",
            {"initial_tr_radius": 0.3},
            "initial_tr_radius",
            0.3,
        ),
        # The two barrier settings act only with a box; they must arrive.
        (
            "trustconstr",
            {"initial_barrier_parameter": 0.5},
            "initial_barrier_parameter",
            0.5,
        ),
        (
            "trustconstr",
            {"initial_barrier_tolerance": 0.5},
            "initial_barrier_tolerance",
            0.5,
        ),
        ("lbfgsb", {"maxcor": 3}, "maxcor", 3),
        ("lbfgsb", {"maxls": 7}, "maxls", 7),
        ("lbfgsb", {}, "maxcor", 10),
        ("lbfgsb", {}, "maxls", 20),
    ],
)
def test_hyperparameters_reach_scipy(monkeypatch, name, hp, option, value):
    seen = []
    real = so.minimize

    def spy(*args, **kwargs):
        seen.append(kwargs["options"])
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "l2co_optimizers._src.scipy_implementations.so.minimize", spy
    )
    _run(_build(name, **hp), 20)
    assert seen and seen[0][option] == value


@pytest.mark.parametrize("hp", [{"maxcor": 0}, {"maxls": 0}])
def test_lbfgsb_refuses_settings_scipy_would_raise_on(hp):
    """Raised at construction, not mid-run, where it would look like a
    failed run."""
    with pytest.raises(ValueError, match="maxcor"):
        _build("lbfgsb", **hp)


@pytest.mark.parametrize("name", NAMES)
def test_unknown_hyperparameters_are_rejected(name):
    with pytest.raises(TypeError):
        _build(name, not_a_knob=1.0)


# =============================================================================
#                                              8. what is refused, and how


@pytest.mark.parametrize("name", NAMES)
def test_stop_fn_raises(name):
    with pytest.raises(ValueError, match="stopping criterion"):
        optimizer_mapping(name)(
            **quadratic_problem(jnp.zeros(DIM), _wavy),
            opt_hash=7,
            bounded=(None, None),
            stop_fn=lambda history, state: jnp.array(False),
        )


@pytest.mark.parametrize("name", NAMES)
def test_has_no_single_step(name):
    update_class = _build(name)
    with pytest.raises(NotImplementedError, match="scipy owns the loop"):
        update_class.step(None, None, {})


@pytest.mark.parametrize("name", NAMES)
def test_cannot_be_unpacked_into_switching_parts(name):
    with pytest.raises(ValueError, match="scipy minimiser"):
        optimizer_parts(
            OptimizationStep(optimizer=name),
            **quadratic_problem(jnp.zeros(DIM), _wavy),
        )


@pytest.mark.parametrize("name", NAMES)
def test_batch_run_maps_its_own_run_over_realizations(name):
    update_class = _build(name)
    assert isinstance(update_class, ScipyUpdateClass)
    assert update_class.sequential_realizations
    assert update_class.popsize == 1

    n_realizations, n = 3, 80
    key = jr.key(0)
    batch_state = BatchState.init(dataset={}, batch_size=None, key=key)
    params = jr.normal(key, (n_realizations, 1, DIM))
    keys = jr.split(key, n_realizations)
    opt_state = eqx.filter_vmap(
        update_class.init_state, in_axes=(0, None, None, 0)
    )(params, batch_state, {}, keys)
    batched = eqx.filter_jit(update_class.batch_run)(
        opt_state=opt_state,
        params=params,
        batch_state=batch_state,
        dataset={},
        key=keys,
        n_iterations=n,
        verbose=False,
    )
    assert batched[5].output_min.shape == (n_realizations, n)
    for r in range(n_realizations):
        single = _run(update_class, n, params=params[r], key=keys[r])
        np.testing.assert_allclose(
            np.asarray(batched[5].output_min[r]), _losses(single), rtol=1e-6
        )


# =============================================================================
#                                                                9. family


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("cobyqa", FAMILY_DERIVATIVE_FREE),
        ("powell", FAMILY_DERIVATIVE_FREE),
        ("tnc", FAMILY_GRADIENT),
        ("trustkrylov", FAMILY_GRADIENT),
        ("slsqp", FAMILY_GRADIENT),
        ("trustconstr", FAMILY_GRADIENT),
        ("lbfgsb", FAMILY_GRADIENT),
    ],
)
def test_family(name, family):
    update_class = _build(name)
    assert update_class.family == family
    assert update_class.transfer_read_fn is None
    assert update_class.transfer_write_fn is None


def test_derivative_free_family_has_no_transfer_spec():
    with pytest.raises(ValueError, match="plain-run"):
        transfer_spec_for("cobyqa", FAMILY_DERIVATIVE_FREE)
