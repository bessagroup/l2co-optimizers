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


def _rosenbrock(x, **_):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def rosenbrock_problem(dimensionality: int = 2) -> dict:
    """Rosenbrock, minimum ``0`` at ``x = 1``; starts at the origin."""
    return quadratic_problem(jnp.zeros(dimensionality), _rosenbrock)


def realization_best_losses(
    name: str,
    problem: dict,
    n_realizations: int,
    n_iterations: int,
    **hyperparameters,
):
    """Per-step best loss of every realization of a batched run.

    Builds the registry entry ``name`` and runs ``n_realizations``
    independent realizations together through :func:`batch_evaluate`,
    the fused, vectorized path a databank cell takes. Realization
    ``i`` gets key ``jr.split(jr.key(1), n)[i]``, which with
    partitionable threefry does not depend on ``n``, so realization 0
    must reproduce whatever ``n`` is (l2co-optimizers#24). Runs in
    float64, as databank runs do.

    Returns
    -------
    jax.Array
        ``history.output_min``, shape ``(n_realizations, n_iterations)``.
    """
    import functools

    import l2co_optimizers as lo

    with jax.enable_x64(True):
        return _realization_best_losses(
            lo,
            functools,
            name,
            problem,
            n_realizations,
            n_iterations,
            **hyperparameters,
        )


def _realization_best_losses(
    lo,
    functools,
    name,
    problem,
    n_realizations,
    n_iterations,
    **hyperparameters,
):
    model = jnp.asarray(problem["model"], dtype=jnp.float64)
    loss_fn = problem["loss_fn"]
    update_class = lo.optimizer_mapping(name)(
        model=model,
        loss_fn=loss_fn,
        pass_rng=problem["pass_rng"],
        opt_hash=0,
        bounded=(None, None),
        stop_fn=None,
        **hyperparameters,
    )
    run_state = lo.RunState.init(
        update_class, model=model, dataset={}, batch_size=None, key=jr.key(0)
    )
    batch_state = lo.BatchState.init(
        dataset={}, batch_size=None, key=jr.key(0)
    )
    _, static = eqx.partition(model, eqx.is_inexact_array)
    sampler = functools.partial(lo.normal_sampling, std=0.1)
    keys = jr.split(jr.key(1), n_realizations)
    _, _, history = eqx.filter_jit(lo.batch_evaluate)(
        run_state,
        batch_state,
        static,
        {},
        loss_fn,
        sampler,
        n_iterations,
        problem["pass_rng"],
        keys,
        False,
    )
    return history.output_min


def opaque_problem(problem: dict) -> dict:
    """The same objective hidden from XLA behind a host callback.

    XLA cannot fuse or re-vectorize a ``pure_callback``, so the loss of
    a point is computed the same way however the run is compiled: the
    reference that a native run should match.
    """
    import numpy as np

    loss_fn = jax.jit(problem["loss_fn"])

    def opaque(x, **_):
        return jax.pure_callback(
            lambda y: np.asarray(loss_fn(y)),
            jax.ShapeDtypeStruct((), x.dtype),
            x,
            vmap_method="sequential",
        )

    return {**problem, "loss_fn": opaque}
