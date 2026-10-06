"""``RunState`` and its rollout entry points (moved from l2co, ADR 0017).

These drive :class:`RunState` on the toy problems, so they need neither
l2co nor ``l2co_tasks``:

1. ``RunState.init`` builds a population-shaped state from an
   already-built ``UpdateClass``, a model and a dataset.
2. ``evaluate`` scores a population through the loss, with and
   without a ``key``.
3. ``reset`` keeps the best of a fresh sample as the running best.
4. ``run`` / ``batch_evaluate`` are thin rewraps of the ``UpdateClass``
   run loop: same numbers, a realization axis on every array field.

Resolving an ``OptimizationStep`` against a real ``Task`` is l2co's
``init_run_state`` and is tested there.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# Local
from l2co_optimizers import (
    BatchState,
    HistoryState,
    OptimizationStep,
    RunState,
    batch_evaluate,
    evaluate,
    normal_sampling,
    optimizer_mapping,
    reset,
    run,
)

from .toy_problems import noisy_sphere_problem, sphere_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

DIM = 4
N_ITERATIONS = 5
N_REALIZATIONS = 3

_STEPS = [
    OptimizationStep(
        optimizer="adam", hyperparameters={"learning_rate": 0.05}
    ),
    OptimizationStep(optimizer="cmaes", hyperparameters={"popsize": 4}),
]
_STEP_IDS = [step.optimizer for step in _STEPS]


def _init(step: OptimizationStep, problem=None, key=None) -> RunState:
    problem = sphere_problem(DIM) if problem is None else problem
    update_class = optimizer_mapping(step.optimizer)(
        **step.hyperparameters,
        **problem,
        opt_hash=step.hash,
        bounded=(None, None),
        stop_fn=step.stopping_fn,
    )
    return RunState.init(
        update_class,
        model=problem["model"],
        dataset={},
        batch_size=None,
        key=jr.key(0) if key is None else key,
    )


def _batch_state() -> BatchState:
    return BatchState.init(dataset={}, batch_size=None, key=jr.key(0))


#                                                                     init
# =============================================================================


@pytest.mark.parametrize("step", _STEPS, ids=_STEP_IDS)
def test_init_builds_a_population_shaped_state(step):
    problem = sphere_problem(DIM)
    run_state = _init(step, problem)

    popsize = run_state.update_class.popsize
    assert run_state.params.shape == (popsize, DIM)
    # Every member starts at the (placeholder) model.
    assert np.array_equal(
        np.asarray(run_state.params),
        np.broadcast_to(np.asarray(problem["model"]), (popsize, DIM)),
    )
    assert np.array_equal(
        np.asarray(run_state.best_params), np.asarray(problem["model"])
    )
    assert float(run_state.best_loss) == float("inf")


#                                                                 evaluate
# =============================================================================


def test_evaluate_scores_every_member():
    problem = sphere_problem(DIM)
    population = jr.normal(jr.key(1), (6, DIM))
    key = jr.key(2)

    loss = evaluate(
        population,
        _batch_state(),
        {},
        problem["loss_fn"],
        problem["pass_rng"],
        key,
        jr.split(key, 6),
    )

    expected = jax.vmap(problem["loss_fn"])(population)
    assert loss.shape == (6,)
    assert np.allclose(np.asarray(loss), np.asarray(expected))


def test_evaluate_hands_each_member_its_own_key():
    problem = noisy_sphere_problem(DIM)
    # Identical members: only the per-member loss key can tell them apart.
    population = jnp.zeros((6, DIM))
    key = jr.key(3)

    loss = evaluate(
        population,
        _batch_state(),
        {},
        problem["loss_fn"],
        problem["pass_rng"],
        key,
        jr.split(key, 6),
    )

    assert np.unique(np.asarray(loss)).size == 6


#                                                                    reset
# =============================================================================


@pytest.mark.parametrize("step", _STEPS, ids=_STEP_IDS)
def test_reset_keeps_the_best_of_the_fresh_sample(step):
    problem = sphere_problem(DIM)
    run_state = _init(step, problem)
    static = eqx.filter(problem["model"], eqx.is_inexact_array, inverse=True)

    reset_state = reset(
        run_state,
        _batch_state(),
        static,
        {},
        problem["loss_fn"],
        problem["pass_rng"],
        normal_sampling,
        jr.key(4),
    )

    losses = jax.vmap(problem["loss_fn"])(reset_state.params)
    assert float(reset_state.best_loss) == pytest.approx(
        float(jnp.min(losses))
    )
    assert float(reset_state.best_loss) == pytest.approx(
        float(problem["loss_fn"](reset_state.best_params))
    )
    assert reset_state.update_class is run_state.update_class


#                                                       run / batch_evaluate
# =============================================================================


@pytest.mark.parametrize("step", _STEPS, ids=_STEP_IDS)
def test_run_rewraps_the_update_class_run(step):
    problem = sphere_problem(DIM)
    run_state = _init(step, problem)
    batch_state = _batch_state()
    kwargs = dict(
        batch_state=batch_state,
        dataset={},
        n_iterations=N_ITERATIONS,
        key=jr.key(5),
        verbose=False,
    )

    new_state, _, history = run(run_state, **kwargs)
    params, best_params, best_loss, opt_state, _, _ = (
        run_state.update_class.run(
            opt_state=run_state.opt_state, params=run_state.params, **kwargs
        )
    )

    assert isinstance(new_state, RunState)
    assert isinstance(history, HistoryState)
    for a, b in zip(
        jax.tree.leaves((new_state.params, new_state.best_params)),
        jax.tree.leaves((params, best_params)),
        strict=True,
    ):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert float(new_state.best_loss) == float(best_loss)
    assert jax.tree.structure(new_state.opt_state) == jax.tree.structure(
        opt_state
    )


@pytest.mark.parametrize("step", _STEPS, ids=_STEP_IDS)
def test_batch_evaluate_adds_a_realization_axis(step):
    problem = sphere_problem(DIM)
    run_state = _init(step, problem)
    static = eqx.filter(problem["model"], eqx.is_inexact_array, inverse=True)

    new_state, _, history = batch_evaluate(
        run_state=run_state,
        batch_state=_batch_state(),
        static=static,
        dataset={},
        loss_fn=problem["loss_fn"],
        sampler=normal_sampling,
        n_iterations=N_ITERATIONS,
        pass_rng=problem["pass_rng"],
        key=jr.split(jr.key(6), N_REALIZATIONS),
        verbose=False,
    )

    popsize = run_state.update_class.popsize
    assert new_state.params.shape == (N_REALIZATIONS, popsize, DIM)
    assert new_state.best_params.shape == (N_REALIZATIONS, DIM)
    assert new_state.best_loss.shape == (N_REALIZATIONS,)
    assert history.output_min.shape == (N_REALIZATIONS, N_ITERATIONS)
    # ``init`` handed the step's hash to the factory as ``opt_hash``.
    assert bool(jnp.all(history.update_step == step.hash))
    # The running best is the realization's own best evaluation.
    assert np.allclose(
        np.asarray(new_state.best_loss),
        np.asarray(jax.vmap(problem["loss_fn"])(new_state.best_params)),
        rtol=1e-5,
    )
