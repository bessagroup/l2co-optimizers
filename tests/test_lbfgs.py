"""Unit tests for the ``"lbfgs"`` registry entry.

Covers the key-flow contract on stochastic tasks (every linesearch
evaluation receives its own freshly split PRNG key — the property the
stock optax wiring cannot provide), the deterministic-task path
(bit-identical trajectories to the previous ``optax.lbfgs`` registry
wiring, no key ever handed to the loss), the registry wiring, and the
feval accounting (one evaluation per linesearch trial —
``docs/adr/0010``).
"""

#                                                                       Modules
# =============================================================================

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

# Local
from l2co_optimizers._src.lbfgs import (
    DEFAULT_MAX_LINESEARCH_STEPS,
    lbfgs_update,
)
from l2co_optimizers._src.mapping import optimizer_mapping, optimizers
from l2co_optimizers._src.optax_implementations import (
    optax_update_extra_kwargs,
    step_fevals,
)
from l2co_optimizers._src.update_class import UpdateClass

from .toy_problems import quadratic_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def _noisy_quadratic(beta: float):
    """Multiplicative log-normal noise on a quadratic bowl."""

    def loss_fn(model, key):
        return jnp.sum(model**2) * jnp.exp(beta * jr.normal(key, ()))

    return loss_fn


def _quadratic(model):
    return jnp.sum(model**2)


def _build(problem: dict):
    return optimizer_mapping("lbfgs")(
        **problem,
        opt_hash=1,
        bounded=(-5.0, 5.0),
        stop_fn=None,
    )


def _run_steps(update_class, n_steps: int, dim: int = 4, jit: bool = True):
    params = jnp.full((1, dim), 2.0)
    state = update_class.init_fn(params, jr.key(0))
    carry = (params, state, jr.key(0))
    step = (
        eqx.filter_jit(update_class.step_fn) if jit else (update_class.step_fn)
    )
    losses = []
    for _ in range(n_steps):
        carry, history = step(carry, {})
        losses.append(history.loss)
    return carry[0], jnp.stack(losses)


# =============================================================================
#                                                                      Registry


def test_registry_routes_lbfgs_to_lbfgs_update():
    assert "lbfgs" in optimizers
    assert optimizers["lbfgs"].func is lbfgs_update


# =============================================================================
#                                                                      Key flow


def test_every_linesearch_evaluation_gets_a_fresh_key():
    """Within one outer step, no two evaluations share a realization.

    A callback inside the loss function records the key of every
    single evaluation. Severe multiplicative noise forces the zoom
    linesearch through several trial evaluations per step; each must
    arrive with a distinct key (the stock optax wiring would show one
    key repeated for the entire step).
    """
    received = []

    def logging_loss(model, key):
        jax.debug.callback(
            lambda kd: received.append(np.asarray(kd).copy()),
            jr.key_data(key),
        )
        return jnp.sum(model**2) * jnp.exp(1.0 * jr.normal(key, ()))

    problem = quadratic_problem(
        model=jnp.zeros(4),
        loss_fn=logging_loss,
        pass_rng=True,
    )
    update_class = _build(problem)

    params = jnp.full((1, 4), 2.0)
    state = update_class.init_fn(params, jr.key(0))
    carry = (params, state, jr.key(0))

    per_step_keys = []
    for _ in range(3):
        received.clear()
        carry, _ = update_class.step_fn(carry, {})
        jax.block_until_ready(carry[0])
        per_step_keys.append([tuple(k.ravel().tolist()) for k in received])

    all_keys = [k for step_keys in per_step_keys for k in step_keys]
    # Several evaluations happened, and every single one used its own
    # key — within a step and across steps.
    assert len(all_keys) >= 3
    assert len(set(all_keys)) == len(all_keys)
    # At least one step exercised multiple linesearch evaluations.
    assert max(len(step_keys) for step_keys in per_step_keys) >= 2


# =============================================================================
#                                                            Deterministic path


def test_deterministic_problem_matches_previous_registry_wiring():
    """Without ``pass_rng`` the factory uses stock ``optax.lbfgs``.

    Fresh keys per evaluation of a deterministic loss are a
    mathematical no-op, so the ``"lbfgs"`` entry must produce
    trajectories bit-identical to the previous registry wiring
    (``optax_update_extra_kwargs`` with ``optax.lbfgs``)
    under the same seeds — and must never hand the loss a key.
    """
    problem = quadratic_problem(
        model=jnp.zeros(4),
        loss_fn=_quadratic,
        pass_rng=False,
    )
    new_wiring = _build(problem)
    old_wiring = optax_update_extra_kwargs(
        **problem,
        optimizer=optax.lbfgs,
        opt_hash=1,
        bounded=(-5.0, 5.0),
        stop_fn=None,
    )
    params_new, losses_new = _run_steps(new_wiring, n_steps=5)
    params_old, losses_old = _run_steps(old_wiring, n_steps=5)
    assert jnp.array_equal(params_new, params_old)
    assert jnp.array_equal(losses_new, losses_old)


# =============================================================================
#                                                                   Convergence


def test_converges_on_mildly_noisy_quadratic():
    """Sanity: the machinery still optimizes when noise is small."""
    problem = quadratic_problem(
        model=jnp.zeros(4),
        loss_fn=_noisy_quadratic(beta=1e-3),
        pass_rng=True,
    )
    update_class = _build(problem)
    final_params, losses = _run_steps(update_class, n_steps=15)
    assert bool(jnp.isfinite(losses).all())
    assert float(jnp.linalg.norm(final_params)) < 1e-3


def test_stays_finite_under_severe_noise():
    """Severe noise may stall progress but must never produce NaNs."""
    problem = quadratic_problem(
        model=jnp.zeros(4),
        loss_fn=_noisy_quadratic(beta=1.0),
        pass_rng=True,
    )
    update_class = _build(problem)
    final_params, losses = _run_steps(update_class, n_steps=15)
    assert bool(jnp.isfinite(losses).all())
    assert bool(jnp.isfinite(final_params).all())


# =============================================================================
#                                                             Feval accounting


def _run_collect(update_class, n_steps: int, dim: int = 4):
    """Run ``n_steps`` and return the billed and actual trial counts.

    Returns ``(fevals, num_linesearch_steps)`` as plain int lists, the
    second read straight off the optimizer state so the assertions
    compare the billed number against the linesearch's own bookkeeping
    rather than against a hard-coded expectation.
    """
    params = jnp.full((1, dim), 2.0)
    carry = (params, update_class.init_fn(params, jr.key(0)), jr.key(0))
    fevals, trials = [], []
    for _ in range(n_steps):
        carry, history = update_class.step_fn(carry, {})
        fevals.append(int(history.fevals))
        trials.append(int(optax.tree.get(carry[1], "num_linesearch_steps")))
    return fevals, trials


def _deterministic_problem():
    return quadratic_problem(
        model=jnp.zeros(4),
        loss_fn=_quadratic,
        pass_rng=False,
    )


def _stochastic_problem():
    return quadratic_problem(
        model=jnp.zeros(4),
        loss_fn=_noisy_quadratic(beta=1.0),
        pass_rng=True,
    )


def test_fevals_bill_every_linesearch_trial():
    """``fevals`` tracks the linesearch, not the iteration count.

    Each zoom-linesearch trial is one objective evaluation, so the
    billed count must equal ``num_linesearch_steps`` — plus one on the
    first step only, where ``optax.value_and_grad_from_state`` has no
    cached value and recomputes. Guards against the flat ``1`` that
    ``docs/adr/0006`` deferred and ``docs/adr/0010`` replaced.

    Both wirings are checked: the stock linesearch on deterministic
    tasks and the per-eval-key linesearch on stochastic ones.
    """
    for problem in (_deterministic_problem(), _stochastic_problem()):
        fevals, trials = _run_collect(_build(problem), n_steps=8)
        expected = [trials[0] + 1] + trials[1:]
        assert fevals == expected, (
            f"pass_rng={problem['pass_rng']}: billed {fevals}, "
            f"linesearch spent {trials}"
        )


def test_fevals_vary_across_steps():
    """The count is dynamic, not a constant.

    A step that accepts its first trial bills 1; a step whose search
    struggles bills more. Several distinct values must appear,
    otherwise the accounting has collapsed back to a constant.

    Uses the stochastic task deliberately: on a *deterministic*
    quadratic the linesearch accepts its first trial on essentially
    every step, so the count is nearly constant there and would make a
    vacuous assertion.
    """
    fevals, _ = _run_collect(_build(_stochastic_problem()), n_steps=15)
    assert len(set(fevals)) >= 3, fevals


def test_exhausted_linesearch_bills_the_cap():
    """A search that runs out of trials bills every one of them.

    When the Wolfe conditions cannot be met the loop runs to
    ``max_linesearch_steps`` and fails out. Those really are
    evaluations spent, and hiding them behind a flat ``1`` is what
    ``docs/adr/0010`` corrects. The bound must also hold: no step may
    bill more than ``fevals_bound``.
    """
    cap = 5
    update_class = optimizer_mapping("lbfgs")(
        **_stochastic_problem(),
        opt_hash=1,
        bounded=(-5.0, 5.0),
        stop_fn=None,
        max_linesearch_steps=cap,
    )
    fevals, _ = _run_collect(update_class, n_steps=15)
    assert cap in fevals, fevals
    assert max(fevals) <= update_class.fevals_bound


def test_fevals_bound_tracks_max_linesearch_steps():
    """``fevals_bound`` is the cap plus the first-step recompute."""
    default = _build(_deterministic_problem())
    assert default.fevals_bound == DEFAULT_MAX_LINESEARCH_STEPS + 1

    override = optimizer_mapping("lbfgs")(
        **_deterministic_problem(),
        opt_hash=1,
        bounded=(-5.0, 5.0),
        stop_fn=None,
        max_linesearch_steps=7,
    )
    assert override.fevals_bound == 8


def test_non_linesearch_optimizers_still_bill_one():
    """Optimizers without a linesearch are unaffected.

    ``optax.tree.get`` returns ``None`` for states carrying no
    ``num_linesearch_steps``, so ``step_fevals`` falls back statically
    to ``1`` and every non-L-BFGS entry keeps its accounting.
    """
    params = jnp.zeros(3)
    for optimizer in (optax.adam(1e-3), optax.sgd(1e-3)):
        state = optimizer.init(params)
        _, new_state = optimizer.update(jnp.ones(3), state, params)
        assert int(step_fevals(state, new_state)) == 1


def test_fevals_bound_defaults_to_popsize():
    """Optimizers that never set the bound report ``popsize``."""
    update_class = UpdateClass(
        init_fn=lambda *a, **k: None,
        step_fn=lambda *a, **k: None,
        popsize=7,
        hash=0,
    )
    assert update_class.max_fevals_per_step is None
    assert update_class.fevals_bound == 7
