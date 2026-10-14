"""Unit tests for the stepped scipy L-BFGS-B and SLSQP (ADR 0011).

``"lbfgsb"`` and ``"slsqp"`` drive scipy's reverse-communication
routines one evaluation per step, each from one transform shared by the
registry entry and the switching parts. They replaced ADR 0002's
whole-run driver, so each is held to that driver in two levels:

A. **Replay.** Handed the evaluations scipy's own ``minimize`` loop
   makes, the transform asks for exactly the points that loop asks for,
   bitwise -- the plumbing (workspace packing, the request loop, the
   evaluation cache, the idle phase) adds nothing.
B. **End to end.** A run evaluates the points the whole-run driver
   (:mod:`tests.whole_run_reference`) evaluates, bitwise. The run loop's
   first step is its unrecorded sentinel, as for every gradient entry,
   so the stepped history is the whole-run one shifted by one
   evaluation.

Then the behaviour ADR 0002 fixed, which the stepped entries keep
(billing, the idle phase, failure, NaN, bounds), what stepping adds
(``stop_fn``, ``vmap``), and the switching parts.
"""

#                                                                       Modules
# =============================================================================

# Standard
import functools
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
    _SLSQP_FTOL,
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
    _slsqp_host_init,
    _slsqp_host_update,
    _slsqp_layout,
    lbfgsb_transform,
    lbfgsb_update,
    slsqp_transform,
    slsqp_update,
)

from .toy_problems import quadratic_problem
from .whole_run_reference import (
    whole_run_lbfgsb_update,
    whole_run_slsqp_update,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

DIM = 3
X64 = bool(jax.config.jax_enable_x64)
NAMES = ("lbfgsb", "slsqp")

#: Per entry: its registry factory, transform, whole-run reference, and
#: scipy's own ``minimize`` method and options.
ENTRIES = {
    "lbfgsb": dict(
        factory=lbfgsb_update,
        transform=lbfgsb_transform,
        reference=whole_run_lbfgsb_update,
        method="L-BFGS-B",
        options=dict(
            ftol=_LBFGSB_FTOL,
            gtol=_LBFGSB_GTOL,
            maxfun=_UNREACHABLE,
            maxiter=_UNREACHABLE,
        ),
    ),
    "slsqp": dict(
        factory=slsqp_update,
        transform=slsqp_transform,
        reference=whole_run_slsqp_update,
        method="SLSQP",
        options=dict(ftol=_SLSQP_FTOL, maxiter=_UNREACHABLE),
    ),
}


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


def _build(
    name,
    loss_fn=_wavy,
    dim=DIM,
    bounded=(None, None),
    stop_fn=None,
    **hp,
):
    return optimizer_mapping(name)(
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


@pytest.mark.parametrize("name", NAMES)
def test_registered_under_its_bare_name(name):
    assert name in STEPPED_SCIPY_OPTIMIZERS
    assert optimizers[name].func is ENTRIES[name]["factory"]
    assert optimizer_mapping(ENTRIES[name]["method"]) is optimizers[name]


@pytest.mark.parametrize(
    ("name", "level"), [("lbfgsb", "box"), ("slsqp", "general")]
)
def test_is_switchable_and_keeps_its_constraint_level(name, level):
    # Unchanged by stepping (ADR 0008). SLSQP's solver takes general
    # constraints; nothing passes it any yet (l2co ADR 0020).
    assert name not in PLAIN_RUN_OPTIMIZERS
    assert name not in SCIPY_OPTIMIZERS
    assert constraint_handling(name) == level


# =============================================================================
#                                           A. replay into scipy's own loop


def _scipy_evaluations(name, fg, x0, bounds, n):
    """``(x, f, g)`` of the first ``n`` evaluations scipy's loop makes,
    and its final point (``None`` if it had not stopped by then)."""
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
            method=ENTRIES[name]["method"],
            bounds=bounds,
            options=ENTRIES[name]["options"],
        )
        final = result.x
    except Enough:
        final = None
    return seen, final


def _numpy_rosenbrock(dtype):
    """Rosenbrock's value and gradient in ``dtype``, as a loop computes."""
    hundred, one = dtype(100), dtype(1)

    def fg(x):
        x = np.asarray(x, dtype=dtype)
        r = x[1:] - x[:-1] ** 2
        f = np.sum(hundred * r**2 + (one - x[:-1]) ** 2)
        g = np.zeros_like(x)
        g[:-1] = -4 * hundred * x[:-1] * r - 2 * (one - x[:-1])
        g[1:] += 2 * hundred * r
        return float(f), np.asarray(g, dtype=np.float64)

    return fg


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("bounded", [(None, None), (-1.0, 0.2)])
@pytest.mark.parametrize("dim", [2, 10])
def test_replay_asks_for_the_points_scipy_asks_for(name, x64, bounded, dim):
    """Level A: the transform's requests are scipy's, bitwise.

    SLSQP at ``ftol = 0`` can grind on for a million evaluations, so
    the idle phase is checked only where scipy stopped within the 400.
    """
    dtype = np.float64 if x64 else np.float32
    fg = _numpy_rosenbrock(dtype)
    x0 = np.asarray(
        np.random.default_rng(dim).normal(size=dim), dtype=dtype
    ).astype(np.float64)
    lo = -np.inf if bounded[0] is None else bounded[0]
    hi = np.inf if bounded[1] is None else bounded[1]
    bounds = None if bounded == (None, None) else so.Bounds(lo, hi)
    seen, final = _scipy_evaluations(name, fg, x0, bounds, 400)

    transform = ENTRIES[name]["transform"](bounded=bounded)
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
        for _ in range(0 if final is None else 3):
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
    assert len(requested) >= 5


# =============================================================================
#                                        B. end to end, against ADR 0002


LEVEL_B_PROBLEMS = [
    (loss_fn, dim)
    for loss_fn in (_sphere, _ellipsoid, _rosenbrock, _rastrigin)
    for dim in (2, 10, 40)
]


def _level_b(name, loss_fn, dim, bounded=(None, None), n=600, **hp):
    params = _x0(dim)
    problem = quadratic_problem(jnp.zeros(dim, params.dtype), loss_fn)
    stepped = _run(_build(name, loss_fn, dim, bounded, **hp), n, params=params)
    whole = _run(
        ENTRIES[name]["reference"](
            **problem, opt_hash=7, bounded=bounded, **hp
        ),
        n,
        params=params,
    )
    # The run loop's sentinel step evaluates x0 unrecorded, so stepped
    # entry i is whole-run entry i + 1.
    np.testing.assert_array_equal(_losses(stepped)[:-1], _losses(whole)[1:])
    assert float(stepped[2]) <= float(whole[2])


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    ("loss_fn", "dim"),
    LEVEL_B_PROBLEMS,
    ids=[f"{f.__name__[1:]}-{d}" for f, d in LEVEL_B_PROBLEMS],
)
def test_runs_the_whole_run_trajectory(name, loss_fn, dim):
    """Level B, in the process's own precision (see the x64 rerun)."""
    _level_b(name, loss_fn, dim)


@pytest.mark.parametrize(
    ("name", "hp"),
    [("lbfgsb", {}), ("lbfgsb", {"maxcor": 3, "maxls": 5}), ("slsqp", {})],
)
def test_runs_the_whole_run_trajectory_in_a_box(name, hp):
    _level_b(name, _rastrigin, 10, bounded=(-1.0, 0.2), **hp)


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


@pytest.mark.parametrize("name", NAMES)
def test_each_step_is_one_real_evaluation(name):
    calls = []
    n = 120
    out = _run(_build(name, _counting(_wavy, calls)), n)
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


@pytest.mark.parametrize("name", NAMES)
def test_final_point_is_re_evaluated_until_the_budget(name):
    calls = []
    n = 2000
    out = _run(_build(name, _counting(_rosenbrock, calls)), n)
    jax.effects_barrier()
    moved = [i for i in range(n) if not np.array_equal(calls[i], calls[-1])]
    stopped = moved[-1] + 1
    assert stopped < n // 2
    np.testing.assert_array_equal(_losses(out)[stopped:], _losses(out)[-1])
    np.testing.assert_array_equal(np.asarray(out[0])[0], calls[-1])


@pytest.mark.parametrize("name", NAMES)
def test_improves_on_its_starting_point(name):
    out = _run(_build(name), 300, params=3.0 + _x0())
    assert float(out[2]) < 0.2 * float(_wavy((3.0 + _x0())[0]))


# =============================================================================
#                                                     failure, on the host


#: Per entry: host ``init``, ``update`` and layout, unbounded, at
#: scipy's default settings.
HOST = {
    "lbfgsb": (
        functools.partial(_lbfgsb_host_init, m=10, bounded=(None, None)),
        functools.partial(
            _lbfgsb_host_update, m=10, maxls=20, bounded=(None, None)
        ),
        lambda n: _lbfgsb_layout(n, 10),
    ),
    "slsqp": (
        functools.partial(_slsqp_host_init, bounded=(None, None)),
        functools.partial(_slsqp_host_update, bounded=(None, None)),
        _slsqp_layout,
    ),
}


def _started(name, n=DIM):
    """A fresh host state: ``(packed, req)``."""
    req, packed = HOST[name][0](np.linspace(-1.0, 1.0, n))
    return packed, req


def _fg(x):
    return float(np.sum((x - 0.5) ** 2)), 2 * (x - 0.5)


def _host_step(name, packed, req, f=None, g=None):
    if f is None:
        f, g = _fg(req)
    return HOST[name][1](packed, np.float64(f), g)


def _break_scipy(monkeypatch, name, how):
    """Make the next scipy call raise, propose or return a non-finite x."""
    if name == "lbfgsb":

        def broken(m, x, *args):
            if how == "raise":
                raise RuntimeError("scipy blew up")
            x[:] = np.nan
            args[10][0] = 3 if how == "nonfinite" else 4  # task: FG / CONV

        monkeypatch.setattr(scipy_stepped._lbfgsb, "setulb", broken)
    else:

        def broken(state, fx, g, c, d, x, *args):
            if how == "raise":
                raise RuntimeError("scipy blew up")
            x[:] = np.nan
            state["mode"] = 1 if how == "nonfinite" else 0

        monkeypatch.setattr(scipy_stepped._slsqplib, "slsqp", broken)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("how", ["raise", "nonfinite", "returns_nonfinite"])
def test_failure_idles_at_the_best_point(monkeypatch, name, how):
    packed, req = _started(name)
    points, losses = [], []
    for _ in range(4):
        points.append(req)
        losses.append(_fg(req)[0])
        req, packed = _host_step(name, packed, req)
    best = points[int(np.argmin(losses))]

    _break_scipy(monkeypatch, name, how)
    req, packed = _host_step(name, packed, req)
    np.testing.assert_array_equal(req, best)
    assert HOST[name][2](DIM).unpack(packed)["status"][0] == _IDLE
    # Idle: scipy is never called again, and the point stays put.
    _break_scipy(monkeypatch, name, "raise")
    again, _ = _host_step(name, packed, req)
    np.testing.assert_array_equal(again, best)


@pytest.mark.parametrize("name", NAMES)
def test_nan_loss_is_handed_on_and_never_best(name):
    packed, req = _started(name)
    first = req
    req, packed = _host_step(name, packed, req)
    _, packed = _host_step(name, packed, req, np.nan, np.zeros(DIM))
    state = HOST[name][2](DIM).unpack(packed)
    assert state["best_loss"][0] == _fg(first)[0]
    np.testing.assert_array_equal(state["best_x"], first)
    assert state["status"][0] in (_RUNNING, _IDLE)


@pytest.mark.parametrize(
    "layout", [_lbfgsb_layout(5, 3), _slsqp_layout(5)], ids=NAMES
)
def test_layout_round_trips_every_field(layout):
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


@pytest.mark.parametrize("name", NAMES)
def test_every_evaluated_point_lies_in_the_box(name):
    bounded = (-1.0, 0.2)
    calls = []
    out = _run(
        _build(name, _counting(_wavy, calls), bounded=bounded),
        200,
        params=2.0 + _x0(),  # starts outside the box
    )
    jax.effects_barrier()
    for points in (np.stack(calls), np.asarray(out[0]), np.asarray(out[1])):
        assert points.min() >= bounded[0]
        assert points.max() <= bounded[1]


@pytest.mark.parametrize("name", NAMES)
def test_reaches_the_boundary_optimum(name):
    out = _run(_build(name, _sphere, bounded=(-1.0, 0.2)), 300)
    np.testing.assert_allclose(np.asarray(out[1]), 0.2, atol=1e-3)


# =============================================================================
#                                                       stop_fn and vmap


@pytest.mark.parametrize("name", NAMES)
def test_honours_a_stopping_criterion(name):
    def below(history, state):
        return jnp.nanmin(history.loss) < 1e-3

    out = _run(_build(name, _sphere, stop_fn=below), 200)
    losses = _losses(out)
    stopped = np.flatnonzero(np.isnan(losses))
    assert stopped.size > 0, "the criterion should have fired"
    assert np.nanmin(losses) < 1e-3
    assert np.all(np.isnan(losses[stopped[0] :]))


@pytest.mark.parametrize("name", NAMES)
def test_batch_run_vmaps_and_matches_single_runs(name):
    update_class = _build(name)
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


def test_lbfgsb_hyperparameters_reach_scipy(monkeypatch):
    seen = []
    real = scipy_stepped._lbfgsb.setulb

    def spy(m, *args):
        seen.append((m, args[14]))
        return real(m, *args)

    monkeypatch.setattr(scipy_stepped._lbfgsb, "setulb", spy)
    _run(_build("lbfgsb", maxcor=3, maxls=7), 10)
    assert seen and set(seen) == {(3, 7)}


@pytest.mark.parametrize("hp", [{"maxcor": 1}, {"maxls": 1}])
def test_lbfgsb_hyperparameters_change_the_run(hp):
    params = 3.0 + _x0()
    default = _losses(_run(_build("lbfgsb"), 60, params=params))
    changed = _losses(_run(_build("lbfgsb", **hp), 60, params=params))
    assert not np.array_equal(default, changed)


@pytest.mark.parametrize("hp", [{"maxcor": 0}, {"maxls": 0}])
def test_lbfgsb_refuses_settings_scipy_would_raise_on(hp):
    with pytest.raises(ValueError, match="maxcor"):
        _build("lbfgsb", **hp)


@pytest.mark.parametrize("name", NAMES)
def test_unknown_hyperparameters_are_rejected(name):
    with pytest.raises(TypeError):
        _build(name, not_a_knob=1.0)


# =============================================================================
#                                                           switching parts


def _parts(name, **hyperparameters):
    return optimizer_parts(
        OptimizationStep(optimizer=name, hyperparameters=hyperparameters),
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


@pytest.mark.parametrize("name", NAMES)
def test_unpacks_into_gradient_parts_billed_one_per_step(name):
    parts = _parts(name)
    assert isinstance(parts, GradientParts)
    assert parts.popsize == 1
    assert parts.max_fevals_per_step is None
    assert parts.family == FAMILY_GRADIENT


@pytest.mark.parametrize("name", NAMES)
def test_parts_evaluate_the_points_the_registry_path_evaluates(name):
    """One transform behind both paths.

    The parts path applies ``params + (x - params)``, which can differ
    from ``x`` in the last bit, and Rosenbrock amplifies that: by step
    ~45 the two drift apart at ~1e-7. So 40 steps at a tight tolerance,
    in float64.
    """
    with jax.enable_x64(True):
        x0 = jnp.array([-1.2, 1.0, -0.5, 0.3], dtype=jnp.float64)
        update_class = optimizer_mapping(name)(
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
            OptimizationStep(optimizer=name),
            **quadratic_problem(x0, _rosenbrock),
        )
        points, losses, _ = _drive_parts(parts, _rosenbrock, x0, 40)
    np.testing.assert_allclose(
        points, jnp.stack(registry_points), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        losses, jnp.stack(registry_losses), rtol=1e-8, atol=1e-8
    )


@pytest.mark.parametrize("name", NAMES)
def test_parts_refuse_an_update_without_the_loss(name):
    parts = _parts(name)
    x0 = jnp.zeros(DIM)
    state = parts.transform.init(x0)
    with pytest.raises(ValueError, match="value="):
        parts.transform.update(jax.grad(_wavy)(x0), state, x0)


@pytest.mark.parametrize("name", NAMES)
def test_parts_report_their_last_move_as_scale(name):
    parts = _parts(name)
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


@pytest.mark.parametrize("name", NAMES)
def test_registry_path_carries_the_same_ports(name):
    update_class = _build(name)
    parts = _parts(name)
    assert update_class.transfer_read_fn is parts.transfer_read_fn
    assert (
        update_class.transfer_write_fn.__name__
        == parts.transfer_write_fn.__name__
    )


def test_lbfgsb_parts_hyperparameters_reach_the_transform(monkeypatch):
    seen = []
    real = scipy_stepped._lbfgsb.setulb

    def spy(m, *args):
        seen.append((m, args[14]))
        return real(m, *args)

    monkeypatch.setattr(scipy_stepped._lbfgsb, "setulb", spy)
    _drive_parts(_parts("lbfgsb", maxcor=4, maxls=9), _wavy, jnp.zeros(DIM), 3)
    assert seen and set(seen) == {(4, 9)}
