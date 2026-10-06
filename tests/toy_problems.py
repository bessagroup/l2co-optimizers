"""Toy objectives and a minimal driver, for tests only.

l2co-optimizers depends on neither ``l2co`` nor ``l2co_tasks``, so its
tests cannot borrow their tasks or rollout loop. Each helper here
returns the three keywords a factory takes -- ``model`` (a flat
parameter vector), ``loss_fn`` and ``pass_rng`` -- as a dict, so a test
builds an optimizer with ``factory(**sphere_problem(), opt_hash=0, ...)``.
There is deliberately no task-shaped object: the package's contract is
the keywords, and the tests use exactly that.

:func:`run_steps` is a bare ``init_fn`` then ``step_fn`` loop in Python,
for tests that inspect every generation's ``OptHistory``. The real loop
is :meth:`UpdateClass.run` (``tests/test_run_loop.py``).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def _sphere(x, **_):
    return jnp.sum((x - 0.5) ** 2)


def _rastrigin(x, **_):
    z = x - 0.5
    return 10.0 * z.size + jnp.sum(z**2 - 10.0 * jnp.cos(2 * jnp.pi * z))


def _noisy_sphere(x, *, key, **_):
    return _sphere(x) + 0.01 * jr.normal(key)


def quadratic_problem(
    model: jax.Array, loss_fn: Callable, pass_rng: bool = False
) -> dict:
    """The factory keywords for a caller-supplied model and loss."""
    return {"model": model, "loss_fn": loss_fn, "pass_rng": pass_rng}


def sphere_problem(dimensionality: int = 4) -> dict:
    """Shifted sphere, minimum ``0`` at ``x = 0.5``; starts at the origin."""
    return quadratic_problem(jnp.zeros(dimensionality), _sphere)


def rastrigin_problem(dimensionality: int = 6) -> dict:
    """Shifted Rastrigin, minimum ``0`` at ``x = 0.5``; multimodal."""
    return quadratic_problem(jnp.zeros(dimensionality), _rastrigin)


def noisy_sphere_problem(dimensionality: int = 4) -> dict:
    """Stochastic sphere: ``loss_fn`` takes a ``key`` (``pass_rng=True``)."""
    return quadratic_problem(
        jnp.zeros(dimensionality), _noisy_sphere, pass_rng=True
    )


def run_steps(update_class, model, n_steps: int, key=None):
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
    params, _ = eqx.partition(model, eqx.is_inexact_array)
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
