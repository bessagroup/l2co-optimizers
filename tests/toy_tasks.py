"""Toy ``TaskLike`` objectives and a minimal driver, for tests only.

l2co-optimizers depends on neither ``l2co`` nor ``l2co_tasks``, so its
tests cannot borrow their tasks or rollout loop. These stand-ins are the
smallest objects that satisfy :class:`l2co_optimizers.TaskLike`: a model
(a flat parameter vector), a loss and a ``pass_rng`` flag. Building every
optimizer from them is itself the check that the protocol is enough.

:func:`run_steps` is a bare ``init_fn`` then ``step_fn`` loop in Python,
for tests that inspect every generation's ``OptHistory``. The real loop
is :meth:`UpdateClass.run` (``tests/test_run_loop.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


@dataclass(frozen=True)
class ToyTask:
    """Smallest object satisfying :class:`l2co_optimizers.TaskLike`.

    Also satisfies :class:`l2co_optimizers.RunnableTaskLike`: the toy
    objectives are data-free, so the dataset is empty and full-batch.
    """

    model: jax.Array
    loss_fn: Callable
    pass_rng: bool = False
    global_min: float | None = None  # unused by optimizers; kept for parity
    loaded_dataset: dict[str, jax.Array] = field(default_factory=dict)
    batch_size: int | None = None


def _sphere(x, **_):
    return jnp.sum((x - 0.5) ** 2)


def _rastrigin(x, **_):
    z = x - 0.5
    return 10.0 * z.size + jnp.sum(z**2 - 10.0 * jnp.cos(2 * jnp.pi * z))


def _noisy_sphere(x, *, key, **_):
    return _sphere(x) + 0.01 * jr.normal(key)


def sphere_task(dimensionality: int = 4) -> ToyTask:
    """Shifted sphere, minimum ``0`` at ``x = 0.5``; starts at the origin."""
    return ToyTask(model=jnp.zeros(dimensionality), loss_fn=_sphere)


def rastrigin_task(dimensionality: int = 6) -> ToyTask:
    """Shifted Rastrigin, minimum ``0`` at ``x = 0.5``; multimodal."""
    return ToyTask(model=jnp.zeros(dimensionality), loss_fn=_rastrigin)


def noisy_sphere_task(dimensionality: int = 4) -> ToyTask:
    """Stochastic sphere: ``loss_fn`` takes a ``key`` (``pass_rng=True``)."""
    return ToyTask(
        model=jnp.zeros(dimensionality),
        loss_fn=_noisy_sphere,
        pass_rng=True,
    )


def quadratic_task(model: jax.Array, loss_fn: Callable, pass_rng=False):
    """A ``ToyTask`` around a caller-supplied model and loss."""
    return ToyTask(model=model, loss_fn=loss_fn, pass_rng=pass_rng)


def run_steps(update_class, task, n_steps: int, key=None):
    """Drive ``update_class`` for ``n_steps`` generations; test-only.

    Mirrors the shape contract of :meth:`UpdateClass.run`: the
    population carries a leading ``popsize`` axis and ``step_fn`` takes
    ``(params, opt_state, key)`` plus the (empty) data batch.

    Returns
    -------
    tuple
        ``(params, opt_state, histories)`` with one ``OptHistory`` per
        generation.
    """
    key = jr.key(0) if key is None else key
    params, _ = eqx.partition(task.model, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda x: jnp.repeat(x[None], update_class.popsize, axis=0), params
    )
    opt_state = update_class.init_fn(params, key)
    carry = (params, opt_state, key)
    histories = []
    for _ in range(n_steps):
        carry, history = update_class.step_fn(carry, sample={})
        histories.append(history)
    params, opt_state, _ = carry
    return params, opt_state, histories
