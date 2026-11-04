"""Unit tests for the stepped scipy L-BFGS-B (ADR 0011).

``"lbfgsb"`` drives scipy's reverse-communication Fortran routine one
evaluation per step, from one transform shared by the registry entry and
the switching parts. It replaced ADR 0002's whole-run driver, so it is
held to that driver in two levels:

A. **Replay.** Handed the evaluations scipy's own ``minimize`` loop
   makes, the transform asks for exactly the points that loop asks for,
   bitwise -- the plumbing (workspace packing, the request loop, the
   evaluation cache, the idle phase) adds nothing.
B. **End to end.** A run evaluates the points the whole-run driver
   (:mod:`tests.whole_run_reference`) evaluates, bitwise. The run loop's
   first step is its unrecorded sentinel, as for every gradient entry,
   so the stepped history is the whole-run one shifted by one
   evaluation.

Then the behaviour ADR 0002 fixed, which the stepped entry keeps
(billing, the idle phase, failure, NaN, bounds), what stepping adds
(``stop_fn``, ``vmap``), and the switching parts.
"""

#                                                                       Modules
# =============================================================================

# Standard
import os
import subprocess
import sys

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
    CONF_ABSENT,
    CONF_ESTIMATED,
    FAMILY_GRADIENT,
    PLAIN_RUN_OPTIMIZERS,
    BatchState,
    GradientParts,
    OptimizationStep,
    constraint_handling,
    optimizer_parts,
)
from l2co_optimizers._src import scipy_stepped
from l2co_optimizers._src.mapping import optimizer_mapping, optimizers
from l2co_optimizers._src.scipy_implementations import (
    _LBFGSB_FTOL,
    _LBFGSB_GTOL,
    _UNREACHABLE,
    SCIPY_OPTIMIZERS,
)
from l2co_optimizers._src.scipy_stepped import (
    _IDLE,
    _RUNNING,
    STEPPED_SCIPY_OPTIMIZERS,
    _host_map,
    _lbfgsb_host_init,
    _lbfgsb_host_update,
    _lbfgsb_layout,
    lbfgsb_transform,
    lbfgsb_update,
)

from .toy_problems import quadratic_problem
from .whole_run_reference import whole_run_lbfgsb_update

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

DIM = 3
X64 = bool(jax.config.jax_enable_x64)


def _wavy(x, **_):
    return jnp.sum((x - 0.5) ** 2) + 0.3 * jnp.sum(jnp.sin(3.0 * x))


def _sphere(x, **_):
    return jnp.sum((x - 0.5) ** 2)


def _ellipsoid(x, **_):
    weights = 10.0 ** (6.0 * jnp.arange(x.size) / max(x.size - 1, 1))
    return jnp.sum(weights * (x - 0.5) ** 2)


def _rosenbrock(x, **_):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def _rastrigin(x, **_):
    z = x - 0.5
    return 10.0 * z.size + jnp.sum(z**2 - 10.0 * jnp.cos(2 * jnp.pi * z))


def _build(loss_fn=_wavy, dim=DIM, bounded=(None, None), stop_fn=None, **hp):
    return optimizer_mapping("lbfgsb")(
        **hp,
        **quadratic_problem(jnp.zeros(dim), loss_fn),
        opt_hash=7,
        bounded=bounded,
        stop_fn=stop_fn,
    )


def _x0(dim=DIM, seed=1):
    return jr.normal(jr.key(seed), (1, dim))


def _run(update_class, n_iterations, params=None, key=None):
    """One realization through ``run``, jitted like a databank run."""
    key = jr.key(0) if key is None else key
    params = _x0() if params is None else params
    batch_state = BatchState.init(dataset={}, batch_size=None, key=key)
    opt_state = update_class.init_state(params, batch_state, {}, key)
    return eqx.filter_jit(update_class.run)(
        opt_state=opt_state,
        params=params,
        batch_state=batch_state,
        dataset={},
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


def test_registered_under_its_bare_name():
    assert "lbfgsb" in STEPPED_SCIPY_OPTIMIZERS
    assert optimizers["lbfgsb"].func is lbfgsb_update
    assert optimizer_mapping("L-BFGS-B") is optimizers["lbfgsb"]


def test_is_switchable_and_keeps_its_box():
    assert "lbfgsb" not in PLAIN_RUN_OPTIMIZERS
    assert "lbfgsb" not in SCIPY_OPTIMIZERS
    assert constraint_handling("lbfgsb") == "box"


# =============================================================================
#                                           A. replay into scipy's own loop


def _scipy_evaluations(fg, x0, bounds, n):
    """``(x, f, g)`` of the first ``n`` evaluations scipy's loop makes."""
    seen = []

    class Enough(Exception):
        pass

    def recorded(x):
        if len(seen) == n:
            raise Enough
        f, g = fg(x)
        seen.append((np.array(x), f, g))
        return f, g

    try:
        result = so.minimize(
            recorded,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(
                ftol=_LBFGSB_FTOL,
                gtol=_LBFGSB_GTOL,
                maxfun=_UNREACHABLE,
                maxiter=_UNREACHABLE,
            ),
        )
        final = result.x
    except Enough:
        final = None
    return seen, final


def _numpy_wavy(dtype):
    """``_wavy``'s value and gradient, in ``dtype`` as a loop computes them."""

    def fg(x):
        x = np.asarray(x, dtype=dtype)
        f = np.sum((x - 0.5) ** 2) + dtype(0.3) * np.sum(np.sin(3 * x))
        g = 2 * (x - 0.5) + dtype(0.9) * np.cos(3 * x)
        return float(f), np.asarray(g, dtype=np.float64)

    return fg


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("bounded", [(None, None), (-1.0, 0.2)])
@pytest.mark.parametrize("dim", [2, 10])
def test_replay_asks_for_the_points_scipy_asks_for(x64, bounded, dim):
    """Level A: the transform's requests are scipy's, bitwise."""
    dtype = np.float64 if x64 else np.float32
    fg = _numpy_wavy(dtype)
    x0 = np.asarray(
        np.random.default_rng(dim).normal(size=dim), dtype=dtype
    ).astype(np.float64)
    lo = -np.inf if bounded[0] is None else bounded[0]
    hi = np.inf if bounded[1] is None else bounded[1]
    bounds = None if bounded == (None, None) else so.Bounds(lo, hi)
    seen, final = _scipy_evaluations(fg, x0, bounds, 400)
    assert final is not None, "scipy should stop within the budget"

    transform = lbfgsb_transform(bounded=bounded)
    with jax.enable_x64(x64):
        state = transform.init(jnp.asarray(x0, dtype=dtype))
        requested = []
        for x, f, g in seen:
            requested.append(np.asarray(state.x))
            np.testing.assert_array_equal(requested[-1], x.astype(dtype))
            _, state = transform.update(
                jnp.asarray(g, dtype=dtype),
                state,
                state.x,
                value=jnp.asarray(f, dtype=dtype),
            )
        # Once scipy stops, every step re-evaluates its final point.
        for _ in range(3):
            np.testing.assert_array_equal(
                np.asarray(state.x), final.astype(dtype)
            )
            f, g = fg(np.asarray(state.x, dtype=np.float64))
            _, state = transform.update(
                jnp.asarray(g, dtype=dtype),
                state,
                state.x,
                value=jnp.asarray(f, dtype=dtype),
            )
    assert len(requested) > 10


# =============================================================================
#                                        B. end to end, against ADR 0002


LEVEL_B_PROBLEMS = [
    (loss_fn, dim)
    for loss_fn in (_sphere, _ellipsoid, _rosenbrock, _rastrigin)
    for dim in (2, 10, 40)
]


def _level_b(loss_fn, dim, bounded=(None, None), n=600, **hp):
    params = _x0(dim)
    problem = quadratic_problem(jnp.zeros(dim, params.dtype), loss_fn)
    stepped = _run(_build(loss_fn, dim, bounded, **hp), n, params=params)
    whole = _run(
        whole_run_lbfgsb_update(**problem, opt_hash=7, bounded=bounded, **hp),
        n,
        params=params,
    )
    # The run loop's sentinel step evaluates x0 unrecorded, so stepped
    # entry i is whole-run entry i + 1.
    np.testing.assert_array_equal(_losses(stepped)[:-1], _losses(whole)[1:])
    assert float(stepped[2]) <= float(whole[2])


@pytest.mark.parametrize(
    ("loss_fn", "dim"),
    LEVEL_B_PROBLEMS,
    ids=[f"{f.__name__[1:]}-{d}" for f, d in LEVEL_B_PROBLEMS],
)
def test_runs_the_whole_run_trajectory(loss_fn, dim):
    """Level B, in the process's own precision (see the x64 rerun)."""
    _level_b(loss_fn, dim)


@pytest.mark.parametrize("hp", [{}, {"maxcor": 3, "maxls": 5}])
def test_runs_the_whole_run_trajectory_in_a_box(hp):
    _level_b(_rastrigin, 10, bounded=(-1.0, 0.2), **hp)


@pytest.mark.slow
@pytest.mark.skipif(X64, reason="already running under x64")
def test_runs_the_whole_run_trajectory_under_x64():
    """Level B again, in a process with x64 on from the start.

    A scoped ``jax.enable_x64`` does not reach the thread the whole-run
    reference evaluates the loss in, so only a process-wide setting
    gives it a float64 loss -- as every databank run sets it.
    """
    env = dict(os.environ, JAX_ENABLE_X64="1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-q",
            "-p",
            "no:cacheprovider",
            "-k",
            "runs_the_whole_run_trajectory and not under_x64",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert " passed" in result.stdout


# =============================================================================
#                                                       billing and phases


def test_each_step_is_one_real_evaluation():
    calls = []
    n = 120
    out = _run(_build(_counting(_wavy, calls)), n)
    jax.effects_barrier()
    history = out[5]
    # n recorded steps after the unrecorded sentinel.
    assert len(calls) == n + 1
    assert int(history.cursor) == n
    np.testing.assert_array_equal(np.asarray(history.fevals), 1)
    np.testing.assert_array_equal(np.asarray(history.update_step), 7)
    np.testing.assert_allclose(
        _losses(out),
        [float(_wavy(jnp.asarray(c))) for c in calls[1:]],
        rtol=1e-6,
    )


def test_final_point_is_re_evaluated_until_the_budget():
    calls = []
    n = 2000
    out = _run(_build(_counting(_wavy, calls)), n)
    jax.effects_barrier()
    moved = [i for i in range(n) if not np.array_equal(calls[i], calls[-1])]
    stopped = moved[-1] + 1
    assert stopped < n // 2
    np.testing.assert_array_equal(_losses(out)[stopped:], _losses(out)[-1])
    np.testing.assert_array_equal(np.asarray(out[0])[0], calls[-1])


def test_improves_on_its_starting_point():
    out = _run(_build(), 300, params=3.0 + _x0())
    assert float(out[2]) < 0.2 * float(_wavy((3.0 + _x0())[0]))


# =============================================================================
#                                                     failure, on the host


def _started(n=DIM, m=10):
    """A host state one evaluation in: ``(packed, req)``."""
    x0 = np.linspace(-1.0, 1.0, n)
    req, packed = _lbfgsb_host_init(x0, m=m, bounded=(None, None))
    return packed, req


def _fg(x):
    return float(np.sum((x - 0.5) ** 2)), 2 * (x - 0.5)


def _host_step(packed, req, **kw):
    f, g = _fg(req)
    return _lbfgsb_host_update(
        packed, np.float64(f), g, m=10, maxls=20, bounded=(None, None), **kw
    )


@pytest.mark.parametrize("how", ["raise", "nonfinite", "returns_nonfinite"])
def test_failure_idles_at_the_best_point(monkeypatch, how):
    packed, req = _started()
    points, losses = [], []
    for _ in range(4):
        points.append(req)
        losses.append(_fg(req)[0])
        req, packed = _host_step(packed, req)
    best = points[int(np.argmin(losses))]

    def broken(m, x, *args):
        task = args[10]
        if how == "raise":
            raise RuntimeError("scipy blew up")
        x[:] = np.nan
        task[0] = 3 if how == "nonfinite" else 4

    monkeypatch.setattr(scipy_stepped._lbfgsb, "setulb", broken)
    req, packed = _host_step(packed, req)
    np.testing.assert_array_equal(req, best)
    state = _lbfgsb_layout(DIM, 10).unpack(packed)
    assert state["status"][0] == _IDLE
    # Idle: scipy is never called again, and the point stays put.
    monkeypatch.setattr(
        scipy_stepped._lbfgsb, "setulb", lambda *a: pytest.fail("called")
    )
    again, _ = _host_step(packed, req)
    np.testing.assert_array_equal(again, best)


def test_nan_loss_is_handed_on_and_never_best():
    packed, req = _started()
    first = req
    req, packed = _host_step(packed, req)
    _, packed = _lbfgsb_host_update(
        packed,
        np.float64(np.nan),
        np.zeros(DIM),
        m=10,
        maxls=20,
        bounded=(None, None),
    )
    state = _lbfgsb_layout(DIM, 10).unpack(packed)
    assert state["best_loss"][0] == _fg(first)[0]
    np.testing.assert_array_equal(state["best_x"], first)
    assert state["status"][0] in (_RUNNING, _IDLE)


def test_layout_round_trips_every_field():
    layout = _lbfgsb_layout(5, 3)
    rng = np.random.default_rng(0)
    state = {
        name: (rng.normal(size=length) * 1e300).astype(dtype)
        if dtype.kind == "f"
        else rng.integers(-(2**30), 2**30, size=length).astype(dtype)
        for name, dtype, length in layout.fields
    }
    packed = layout.pack(state)
    assert packed.dtype == np.uint32 and packed.size == layout.size
    for name, value in layout.unpack(packed).items():
        np.testing.assert_array_equal(value, state[name])


# =============================================================================
#                                                                   bounds


def test_every_evaluated_point_lies_in_the_box():
    bounded = (-1.0, 0.2)
    calls = []
    out = _run(
        _build(_counting(_wavy, calls), bounded=bounded),
        200,
        params=2.0 + _x0(),  # starts outside the box
    )
    jax.effects_barrier()
    for points in (np.stack(calls), np.asarray(out[0]), np.asarray(out[1])):
        assert points.min() >= bounded[0]
        assert points.max() <= bounded[1]


def test_reaches_the_boundary_optimum():
    out = _run(_build(_sphere, bounded=(-1.0, 0.2)), 300)
    np.testing.assert_allclose(np.asarray(out[1]), 0.2, atol=1e-3)


# =============================================================================
#                                                       stop_fn and vmap


def test_honours_a_stopping_criterion():
    def below(history, state):
        return jnp.nanmin(history.loss) < 1e-3

    out = _run(_build(_sphere, stop_fn=below), 200)
    losses = _losses(out)
    stopped = np.flatnonzero(np.isnan(losses))
    assert stopped.size > 0, "the criterion should have fired"
    assert np.nanmin(losses) < 1e-3
    assert np.all(np.isnan(losses[stopped[0] :]))


def test_batch_run_vmaps_and_matches_single_runs():
    update_class = _build()
    assert not update_class.sequential_realizations
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
    for r in range(n_realizations):
        single = _run(update_class, n, params=params[r], key=keys[r])
        np.testing.assert_array_equal(
            np.asarray(batched[5].output_min[r]), _losses(single)
        )


def test_host_map_runs_every_element_of_a_nested_batch():
    def fn(a, b):
        return (a * b.sum(), np.array([a, a]))

    shapes = (
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((2,), jnp.float32),
    )
    host = _host_map(fn, (0, 1), shapes)
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.ones((1, 1, 4), np.float32)  # unbatched under both vmaps
    out = host(a, b)
    assert out[0].shape == (2, 3) and out[1].shape == (2, 3, 2)
    np.testing.assert_array_equal(out[0], 4 * a)
    np.testing.assert_array_equal(out[1][..., 1], a)


# =============================================================================
#                                                        hyperparameters


def test_hyperparameters_reach_scipy(monkeypatch):
    seen = []
    real = scipy_stepped._lbfgsb.setulb

    def spy(m, *args):
        seen.append((m, args[14]))
        return real(m, *args)

    monkeypatch.setattr(scipy_stepped._lbfgsb, "setulb", spy)
    _run(_build(maxcor=3, maxls=7), 10)
    assert seen and set(seen) == {(3, 7)}


@pytest.mark.parametrize("hp", [{"maxcor": 1}, {"maxls": 1}])
def test_hyperparameters_change_the_run(hp):
    params = 3.0 + _x0()
    default = _losses(_run(_build(), 60, params=params))
    changed = _losses(_run(_build(**hp), 60, params=params))
    assert not np.array_equal(default, changed)


@pytest.mark.parametrize("hp", [{"maxcor": 0}, {"maxls": 0}])
def test_refuses_settings_scipy_would_raise_on(hp):
    with pytest.raises(ValueError, match="maxcor"):
        _build(**hp)


def test_unknown_hyperparameters_are_rejected():
    with pytest.raises(TypeError):
        _build(not_a_knob=1.0)


# =============================================================================
#                                                           switching parts


def _parts(**hyperparameters):
    return optimizer_parts(
        OptimizationStep(optimizer="lbfgsb", hyperparameters=hyperparameters),
        **quadratic_problem(jnp.zeros(DIM), _wavy),
    )


def _drive_parts(parts, loss_fn, x0, n_steps):
    """A loop as a switching consumer runs it: evaluate, then update."""
    update = eqx.filter_jit(
        lambda g, s, p, v: parts.transform.update(g, s, p, value=v, grad=g)
    )
    params, state = x0, parts.transform.init(x0)
    points, losses = [], []
    for _ in range(n_steps):
        value, grad = jax.value_and_grad(loss_fn)(params)
        points.append(params)
        losses.append(value)
        step, state = update(grad, state, params, value)
        params = params + step
    return jnp.stack(points), jnp.stack(losses), state


def test_unpacks_into_gradient_parts_billed_one_per_step():
    parts = _parts()
    assert isinstance(parts, GradientParts)
    assert parts.popsize == 1
    assert parts.max_fevals_per_step is None
    assert parts.family == FAMILY_GRADIENT


def test_parts_evaluate_the_points_the_registry_path_evaluates():
    """One transform behind both paths.

    The parts path applies ``params + (x - params)``, which can differ
    from ``x`` in the last bit, and Rosenbrock amplifies that: by step
    ~45 the two drift apart at ~1e-7. So 40 steps at a tight tolerance,
    in float64.
    """
    with jax.enable_x64(True):
        x0 = jnp.array([-1.2, 1.0, -0.5, 0.3], dtype=jnp.float64)
        update_class = optimizer_mapping("lbfgsb")(
            **quadratic_problem(x0, _rosenbrock),
            opt_hash=1,
            bounded=(None, None),
            stop_fn=None,
        )
        carry = (x0[None], update_class.init_fn(x0[None]), jr.key(0))
        registry_points, registry_losses = [], []
        step = jax.jit(update_class.step_fn)
        for _ in range(40):
            carry, history = step(carry, {})
            registry_points.append(history.params[0])
            registry_losses.append(history.loss[0])
        parts = optimizer_parts(
            OptimizationStep(optimizer="lbfgsb"),
            **quadratic_problem(x0, _rosenbrock),
        )
        points, losses, _ = _drive_parts(parts, _rosenbrock, x0, 40)
    np.testing.assert_allclose(
        points, jnp.stack(registry_points), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        losses, jnp.stack(registry_losses), rtol=1e-8, atol=1e-8
    )


def test_parts_refuse_an_update_without_the_loss():
    parts = _parts()
    x0 = jnp.zeros(DIM)
    state = parts.transform.init(x0)
    with pytest.raises(ValueError, match="value="):
        parts.transform.update(jax.grad(_wavy)(x0), state, x0)


def test_parts_report_their_last_move_as_scale():
    parts = _parts()
    x0 = jnp.zeros(DIM)
    state = parts.transform.init(x0)
    _, conf = parts.transfer_read_fn(state)
    assert float(conf) == CONF_ABSENT
    _, _, state = _drive_parts(parts, _wavy, x0, 2)
    sigma, conf = parts.transfer_read_fn(state)
    step = state.x - state.x_prev
    assert float(conf) == CONF_ESTIMATED
    np.testing.assert_allclose(sigma, jnp.sqrt(jnp.mean(step**2)), rtol=1e-6)
    assert parts.transfer_write_fn.__name__ == "write_none"


def test_registry_path_carries_the_same_ports():
    update_class = _build()
    parts = _parts()
    assert update_class.transfer_read_fn is parts.transfer_read_fn
    assert (
        update_class.transfer_write_fn.__name__
        == parts.transfer_write_fn.__name__
    )


def test_parts_hyperparameters_reach_the_transform(monkeypatch):
    seen = []
    real = scipy_stepped._lbfgsb.setulb

    def spy(m, *args):
        seen.append((m, args[14]))
        return real(m, *args)

    monkeypatch.setattr(scipy_stepped._lbfgsb, "setulb", spy)
    _drive_parts(_parts(maxcor=4, maxls=9), _wavy, jnp.zeros(DIM), 3)
    assert seen and set(seen) == {(4, 9)}
