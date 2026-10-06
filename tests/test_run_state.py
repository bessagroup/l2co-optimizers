"""``RunState`` and its rollout entry points (moved from l2co, ADR 0017).

These drive :class:`RunState` on the toy tasks, so they need neither
l2co nor ``l2co_tasks``:

1. ``RunState.init`` resolves the optimizer through the registry and
   builds a population-shaped state from a :class:`RunnableTaskLike`.
2. ``evaluate`` scores a population through the task's loss, with and
   without a ``key``.
3. ``reset`` keeps the best of a fresh sample as the running best.
4. ``run`` / ``batch_evaluate`` are thin rewraps of the ``UpdateClass``
   run loop: same numbers, a realization axis on every array field.

The real-``Task`` path is in ``test_l2co_tasks_integration.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from l2co_optimizers import (
    BatchState,
    HistoryState,
    OptimizationStep,
    RunnableTaskLike,
    RunState,
    batch_evaluate,
    evaluate,
    normal_sampling,
    reset,
    run,
)

from .toy_tasks import noisy_sphere_task, sphere_task

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


def _init(step: OptimizationStep, task=None, key=None) -> RunState:
    return RunState.init(
        optimizer=step,
        task=sphere_task(DIM) if task is None else task,
        bounded=(None, None),
        key=jr.key(0) if key is None else key,
    )


def _batch_state(task) -> BatchState:
    return BatchState.init(
        dataset=task.loaded_dataset, batch_size=task.batch_size, key=jr.key(0)
    )


#                                                                     init
# =============================================================================


def test_toy_task_is_runnable_tasklike():
    assert isinstance(sphere_task(DIM), RunnableTaskLike)


@pytest.mark.parametrize("step", _STEPS, ids=_STEP_IDS)
def test_init_builds_a_population_shaped_state(step):
    task = sphere_task(DIM)
    run_state = _init(step, task)

    popsize = run_state.update_class.popsize
    assert run_state.params.shape == (popsize, DIM)
    # Every member starts at the task's (placeholder) model.
    assert np.array_equal(
        np.asarray(run_state.params),
        np.broadcast_to(np.asarray(task.model), (popsize, DIM)),
    )
    assert np.array_equal(
        np.asarray(run_state.best_params), np.asarray(task.model)
    )
    assert float(run_state.best_loss) == float("inf")


#                                                                 evaluate
# =============================================================================


def test_evaluate_scores_every_member():
    task = sphere_task(DIM)
    population = jr.normal(jr.key(1), (6, DIM))
    key = jr.key(2)

    loss = evaluate(
        population,
        _batch_state(task),
        task.loaded_dataset,
        task.loss_fn,
        task.pass_rng,
        key,
        jr.split(key, 6),
    )

    expected = jax.vmap(task.loss_fn)(population)
    assert loss.shape == (6,)
    assert np.allclose(np.asarray(loss), np.asarray(expected))


def test_evaluate_hands_each_member_its_own_key():
    task = noisy_sphere_task(DIM)
    # Identical members: only the per-member loss key can tell them apart.
    population = jnp.zeros((6, DIM))
    key = jr.key(3)

    loss = evaluate(
        population,
        _batch_state(task),
        task.loaded_dataset,
        task.loss_fn,
        task.pass_rng,
        key,
        jr.split(key, 6),
    )

    assert np.unique(np.asarray(loss)).size == 6


#                                                                    reset
# =============================================================================


@pytest.mark.parametrize("step", _STEPS, ids=_STEP_IDS)
def test_reset_keeps_the_best_of_the_fresh_sample(step):
    task = sphere_task(DIM)
    run_state = _init(step, task)
    static = eqx.filter(task.model, eqx.is_inexact_array, inverse=True)

    reset_state = reset(
        run_state,
        _batch_state(task),
        static,
        task.loaded_dataset,
        task.loss_fn,
        task.pass_rng,
        normal_sampling,
        jr.key(4),
    )

    losses = jax.vmap(task.loss_fn)(reset_state.params)
    assert float(reset_state.best_loss) == pytest.approx(
        float(jnp.min(losses))
    )
    assert float(reset_state.best_loss) == pytest.approx(
        float(task.loss_fn(reset_state.best_params))
    )
    assert reset_state.update_class is run_state.update_class


#                                                       run / batch_evaluate
# =============================================================================


@pytest.mark.parametrize("step", _STEPS, ids=_STEP_IDS)
def test_run_rewraps_the_update_class_run(step):
    task = sphere_task(DIM)
    run_state = _init(step, task)
    batch_state = _batch_state(task)
    kwargs = dict(
        batch_state=batch_state,
        dataset=task.loaded_dataset,
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
    task = sphere_task(DIM)
    run_state = _init(step, task)
    static = eqx.filter(task.model, eqx.is_inexact_array, inverse=True)

    new_state, _, history = batch_evaluate(
        run_state=run_state,
        batch_state=_batch_state(task),
        static=static,
        dataset=task.loaded_dataset,
        loss_fn=task.loss_fn,
        sampler=normal_sampling,
        n_iterations=N_ITERATIONS,
        pass_rng=task.pass_rng,
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
        np.asarray(jax.vmap(task.loss_fn)(new_state.best_params)),
        rtol=1e-5,
    )
