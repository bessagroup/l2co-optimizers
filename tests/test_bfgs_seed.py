"""Seeding ``bfgs``'s inverse Hessian and reading ``cmaes``'s covariance.

The hooks behind issue #64 (CMA-ES -> BFGS warm starts). The properties
checked:

1. on an ill-conditioned convex quadratic, a ``bfgs`` run seeded with
   the exact inverse Hessian needs fewer evaluations than one seeded
   with the identity, with either line search;
2. without a seed, a run evaluates the points it evaluated before the
   hooks existed (a trajectory recorded at commit c88801a, to 1e-12),
   and seeded with the identity it runs bitwise as without a seed;
2b. with ``linesearch="wolfe"``, the steepest-descent fallback and the
   restart after a failed line search both return to the seed, not to
   the identity;
3. on a real ``cmaes`` state, :func:`read_cmaes` returns its mean,
   step size and ``C``, and :func:`seed_cmaes_covariance` replaces ``C``
   (with ``B`` and ``D`` such that ``B diag(D**2) B^T = C``) and keeps
   the mean and step size.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# Local
from l2co_optimizers import (
    read_cmaes,
    read_inverse_hessian,
    seed_cmaes_covariance,
    seed_inverse_hessian,
)
from l2co_optimizers._src.mapping import optimizer_mapping
from l2co_optimizers._src.optimistix_implementations import _WOLFE_MAX_EVALS

from .toy_problems import quadratic_problem, run_steps, sphere_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def _ill_conditioned(n=6, condition=1e4):
    """A rotated SPD matrix of the given condition number."""
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(n, n)))
    return q @ np.diag(np.logspace(0.0, np.log10(condition), n)) @ q.T


def _bfgs(loss_fn, n, linesearch):
    return optimizer_mapping("bfgs")(
        linesearch=linesearch,
        **quadratic_problem(jnp.zeros(n), loss_fn),
        opt_hash=1,
        bounded=(None, None),
        stop_fn=None,
    )


def _losses(update_class, x0, n_steps, seed=None):
    """The loss at every evaluated point, seeding the state with ``seed``."""
    params, key = jnp.asarray(x0)[None], jr.key(0)
    state = update_class.init_fn(params, key)
    if seed is not None:
        state = seed_inverse_hessian(state, seed)
    carry = (params, state, key)
    step = eqx.filter_jit(update_class.step_fn)
    losses = []
    for _ in range(n_steps):
        carry, history = step(carry, {})
        losses.append(float(history.loss[0]))
    return np.array(losses)


def _evals_to(losses, target):
    hit = np.nonzero(np.minimum.accumulate(losses) <= target)[0]
    return int(hit[0]) + 1 if len(hit) else np.inf


# =============================================================================
#                                    1. the exact inverse Hessian pays off


@pytest.mark.parametrize("linesearch", ["armijo", "wolfe"])
def test_exact_inverse_hessian_needs_fewer_evaluations(linesearch):
    with jax.enable_x64(True):
        a = _ill_conditioned()
        n = a.shape[0]

        def loss_fn(x, **_):
            return 0.5 * x @ (a @ x)

        update_class = _bfgs(loss_fn, n, linesearch)
        x0 = np.linspace(1.0, 2.0, n)
        target = 1e-10 * float(loss_fn(x0))

        exact = _losses(update_class, x0, 60, seed=np.linalg.inv(a))
        identity = _losses(update_class, x0, 60, seed=np.eye(n))

        assert _evals_to(exact, target) < _evals_to(identity, target)

        params = jnp.asarray(x0)[None]
        seeded = seed_inverse_hessian(
            update_class.init_fn(params, jr.key(0)), np.linalg.inv(a)
        )
        np.testing.assert_allclose(
            read_inverse_hessian(seeded), np.linalg.inv(a), rtol=1e-12
        )


# =============================================================================
#                                     2. the default path is unchanged


_RECORDED = Path(__file__).parent / "data" / "bfgs_unseeded_c88801a.json"


def _recorded_problem(x, **_):
    w = 10.0 ** jnp.arange(x.shape[0])
    return jnp.sum(w * x**2) + 0.1 * jnp.sum(jnp.sin(3.0 * x))


def _trajectory(linesearch, n_steps, seed=None):
    """The losses and evaluated points of a run on the recorded problem."""
    with jax.enable_x64(True):
        update_class = _bfgs(_recorded_problem, 4, linesearch)
        params, key = jnp.linspace(1.0, 2.0, 4)[None], jr.key(0)
        state = update_class.init_fn(params, key)
        if seed is not None:
            state = seed_inverse_hessian(state, seed)
        carry = (params, state, key)
        step = eqx.filter_jit(update_class.step_fn)
        losses, points = [], []
        for _ in range(n_steps):
            carry, history = step(carry, {})
            losses.append(float(history.loss[0]))
            points.append([float(v) for v in history.params[0]])
    return np.array(losses), np.array(points)


#: Cross-host tolerance for the recorded trajectory (see the test).
_HOST_RTOL = 1e-9


@pytest.mark.parametrize("linesearch", ["armijo", "wolfe"])
def test_unseeded_run_matches_the_pre_change_trajectory(linesearch):
    """Against the trajectory recorded before the hooks existed.

    The recording holds exact float64 values, but it was made on one CPU
    and XLA build (Oscar, x86-64); another host differs in the last bits,
    and the difference grows along the run: macOS arm64 in CI was off by
    a relative 1.4e-12 after 40 steps. ``_HOST_RTOL = 1e-9`` absorbs that
    with room to spare and still catches any change in behaviour: starting
    from ``1.001 * I`` instead of ``I`` moves the run by 1e-2 or more
    (relative).
    """
    recorded = json.loads(_RECORDED.read_text())[linesearch]
    losses, points = _trajectory(linesearch, len(recorded["loss"]))
    np.testing.assert_allclose(
        losses, [float.fromhex(v) for v in recorded["loss"]], rtol=_HOST_RTOL
    )
    np.testing.assert_allclose(
        points,
        [[float.fromhex(v) for v in p] for p in recorded["params"]],
        rtol=_HOST_RTOL,
    )


@pytest.mark.parametrize("linesearch", ["armijo", "wolfe"])
def test_identity_seed_runs_exactly_as_no_seed(linesearch):
    """Bitwise, on the same host: seeding the identity changes nothing."""
    unseeded = _trajectory(linesearch, 40)
    identity = _trajectory(linesearch, 40, seed=np.eye(4))
    np.testing.assert_array_equal(identity[0], unseeded[0])
    np.testing.assert_array_equal(identity[1], unseeded[1])


# =============================================================================
#                                  2b. WolfeBFGS resets to the seed


def _drifted_wolfe_state(update_class, params, key, seed, current):
    """A seeded ``"wolfe"`` state whose estimate has moved to ``current``.

    It stands in for a run that has already updated its estimate away
    from the seed it started from: the stored seed is ``seed``, the
    estimate the next direction uses is ``current``.
    """
    state = update_class.init_fn(params, key)
    seeded = seed_inverse_hessian(state, seed)
    moved = seed_inverse_hessian(state, current)
    return eqx.tree_at(
        lambda s: s.solver_state.f_info, seeded, moved.solver_state.f_info
    )


def _spd_seed(n):
    """A symmetric positive definite matrix far from the identity."""
    return np.diag(np.linspace(0.5, 3.0, n)) + 0.1 * np.ones((n, n))


def test_wolfe_fallback_returns_to_the_seed():
    """When the estimate does not go downhill, the step falls back to
    the seed, not the identity."""
    with jax.enable_x64(True):
        n = 4
        h0 = _spd_seed(n)
        update_class = _bfgs(lambda x, **_: 0.5 * x @ x, n, "wolfe")
        params, key = jnp.linspace(1.0, 2.0, n)[None], jr.key(0)
        # -I sends the quasi-Newton direction uphill.
        state = _drifted_wolfe_state(update_class, params, key, h0, -np.eye(n))
        (_, state, _), _ = eqx.filter_jit(update_class.step_fn)(
            (params, state, key), {}
        )
        np.testing.assert_array_equal(read_inverse_hessian(state), h0)


def test_wolfe_restart_returns_to_the_seed():
    """When a line search fails without a better point, the estimate
    restarts from the seed, not the identity."""
    with jax.enable_x64(True):
        n = 4
        h0 = _spd_seed(n)
        x0 = jnp.linspace(1.0, 2.0, n)

        def loss_fn(x, **_):
            # Finite only at the start, so every trial point fails and
            # the search runs out of evaluations.
            return jnp.where(jnp.all(x == x0), 0.5 * x @ x, jnp.nan)

        update_class = _bfgs(loss_fn, n, "wolfe")
        params, key = x0[None], jr.key(0)
        # 2 I goes downhill, so only the restart can bring back the seed.
        state = _drifted_wolfe_state(
            update_class, params, key, h0, 2.0 * np.eye(n)
        )
        carry = (params, state, key)
        step = eqx.filter_jit(update_class.step_fn)
        for _ in range(_WOLFE_MAX_EVALS + 2):
            carry, _ = step(carry, {})
        np.testing.assert_array_equal(read_inverse_hessian(carry[1]), h0)


# =============================================================================
#                                    3. reading and seeding cmaes's C


def test_cmaes_covariance_round_trip():
    n = 4
    update_class = optimizer_mapping("cmaes")(
        **sphere_problem(n),
        popsize=8,
        opt_hash=1,
        bounded=(None, None),
        stop_fn=None,
    )
    _, state, _ = run_steps(update_class, jnp.zeros(n), 5)

    mean, sigma, c = read_cmaes(state)
    np.testing.assert_array_equal(mean, state.mean)
    assert float(sigma) == float(state.std)
    np.testing.assert_array_equal(c, state.C)
    assert not np.allclose(c, np.eye(n))  # five generations adapted it

    rng = np.random.default_rng(1)
    m = rng.normal(size=(n, n))
    new_c = jnp.asarray(m @ m.T + n * np.eye(n), dtype=c.dtype)
    seeded = seed_cmaes_covariance(state, new_c)

    seeded_mean, seeded_sigma, seeded_c = read_cmaes(seeded)
    np.testing.assert_array_equal(seeded_c, new_c)
    np.testing.assert_array_equal(seeded_mean, mean)
    assert float(seeded_sigma) == float(sigma)
    np.testing.assert_allclose(
        seeded.B @ jnp.diag(seeded.D**2) @ seeded.B.T,
        new_c,
        atol=1e-5 * float(jnp.max(jnp.abs(new_c))),  # float32 round-off
    )
