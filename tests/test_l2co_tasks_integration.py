"""Integration battery: real ``l2co_tasks.Task`` objects through every factory.

``l2co_tasks`` is intentionally **not** a dependency of
``l2co-optimizers`` -- tasks reach the optimizers only through the
structural :class:`l2co_optimizers.TaskLike` protocol -- so this module
skips entirely unless ``l2co_tasks`` imports. Run it against a sibling
checkout with::

    uv run --with-editable ../l2co-tasks \
        pytest tests/test_l2co_tasks_integration.py

It checks the bet the protocol makes: that a real ``Task`` satisfies it
unchanged, and that every registered optimizer builds and steps on one.
"""

import jax.numpy as jnp
import pytest

import l2co_optimizers as lo

from .toy_tasks import run_steps

l2co_tasks = pytest.importorskip("l2co_tasks")

pytestmark = pytest.mark.requires_l2co_tasks

# Hyperparameters a factory cannot default (optax constructors take a
# required learning rate); everything else builds from its defaults.
_REQUIRED = {"learning_rate": 1e-2}
_NEEDS_BOUNDS = {"shade", "turbo", "rbftrustregion"}


@pytest.fixture(scope="module", params=[False, True], ids=["det", "noisy"])
def task(request):
    if request.param:
        return l2co_tasks.create_bbob_noisy_task(
            fn_name="bbob_noisy_f101", seed=0, dimensionality=4
        )
    return l2co_tasks.create_bbob_task(
        fn_name="sphere", seed=0, dimensionality=4
    )


def test_task_is_tasklike(task):
    assert isinstance(task, lo.TaskLike)
    assert isinstance(task, lo.RunnableTaskLike)


def test_run_state_batch_evaluates_a_real_task(task):
    """The rollout path l2co's ``RolloutWrapper`` drives, minus l2co."""
    import equinox as eqx
    import jax.random as jr

    step = lo.OptimizationStep(
        optimizer="adam", hyperparameters={"learning_rate": 0.05}
    )
    run_state = lo.RunState.init(
        optimizer=step, task=task, bounded=(None, None), key=jr.key(0)
    )
    batch_state = lo.BatchState.init(
        dataset=task.loaded_dataset, batch_size=task.batch_size, key=jr.key(0)
    )
    run_state, _, history = lo.batch_evaluate(
        run_state=run_state,
        batch_state=batch_state,
        static=eqx.filter(task.model, eqx.is_inexact_array, inverse=True),
        dataset=task.loaded_dataset,
        loss_fn=task.loss_fn,
        sampler=lo.normal_sampling,
        n_iterations=3,
        pass_rng=task.pass_rng,
        key=jr.split(jr.key(1), 2),
        verbose=False,
    )
    assert run_state.best_loss.shape == (2,)
    assert jnp.isfinite(run_state.best_loss).all()
    assert history.output_min.shape == (2, 3)


def test_count_parameters_matches_task_dimensionality(task):
    assert lo.count_parameters(task.model) == task.dimensionality


@pytest.mark.parametrize("name", sorted(lo.optimizers))
def test_every_registered_factory_builds_and_steps(name, task):
    factory = lo.optimizer_mapping(name)
    kwargs = dict(task=task, opt_hash=1)
    if name in _NEEDS_BOUNDS:
        kwargs.update(bounded=(-5.0, 5.0), x_min=-5.0, x_max=5.0)
    try:
        update_class = factory(**kwargs)
    except TypeError:
        update_class = factory(**kwargs, **_REQUIRED)
    if isinstance(update_class, lo.UpdateClass) and not hasattr(
        update_class, "sampling_fn"
    ):
        _, _, histories = run_steps(update_class, task, n_steps=2)
        assert jnp.isfinite(histories[-1].loss).any()
