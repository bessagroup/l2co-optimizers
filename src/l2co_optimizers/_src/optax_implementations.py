"""
Optax ``UpdateClass`` factories and optimizer name registry.

The closure factories ``optax_fn`` / ``optax_extra_kwargs_fn`` build the
``(init_fn, step_fn)`` pairs for an optax optimizer, and ``step_fevals``
is the feval count they bill. The ``UpdateClass`` factories
``optax_update`` / ``optax_update_extra_kwargs`` wrap them into an
:class:`~l2co_optimizers._src.update_class.UpdateClass`. The registry
binds each optimizer name to the appropriate factory via
:class:`jax.tree_util.Partial`. The caller is responsible for supplying
every hyperparameter the underlying optax constructor needs (e.g.
``learning_rate`` for ``adam``); no defaults are injected here.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

# Third-party
import optax
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray, PyTree
from optax import OptState as OptaxState

from l2co_optimizers._src.loss import (
    vmapped_loss_and_grad,
    vmapped_loss_and_grad_with_rng,
)
from l2co_optimizers._src.opt_history import OptHistory
from l2co_optimizers._src.state_transfer import (
    FAMILY_GRADIENT,
    build_transfer_fns,
)
from l2co_optimizers._src.typing import (
    InitFunction,
    InputParameters,
    LossFunction,
    StepFunction,
    StopFunction,
    TaskLike,
)

# Local
from l2co_optimizers._src.update_class import UpdateClass
from l2co_optimizers._src.utils import (
    normalize_key,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#                                                             Closure factories
# =============================================================================


def optax_fn(
    static: PyTree,
    optimizer: optax.GradientTransformationExtraArgs,
    loss_fn: LossFunction,
    bounded: tuple[float | None, float | None] | None,
    opt_hash: int,
    pass_rng: bool,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for Optax optimizers.

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : optax.GradientTransformationExtraArgs
        Optax optimizer instance.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float | None, float | None] or None
        Parameter bounds for clipping. ``None``, or ``None`` on one
        side, leaves that side unbounded.
    opt_hash : int
        Hash for optimizer step tracking.
    pass_rng : bool
        Whether to pass RNG key to loss function.
    **kwargs
        Additional arguments.

    Returns
    -------
    tuple[InitFunction, StepFunction]
        Initialization and step functions for optimization.
    """
    if bounded is None:
        bounded = (None, None)

    def init_fn(
        params: InputParameters, key: PRNGKeyArray | None = None, **sample
    ) -> OptaxState:
        """Initialise the Optax optimizer state."""
        opt_state = optimizer.init(params)
        return opt_state

    def step_fn(
        carry: tuple[InputParameters, OptaxState, PRNGKeyArray],
        sample: dict,
    ) -> tuple[tuple[InputParameters, OptaxState, PRNGKeyArray], OptHistory]:
        """Perform one gradient-based Optax step."""
        input_params, opt_state, key = carry

        if pass_rng:
            loss, grads = vmapped_loss_and_grad_with_rng(
                eqx.combine(input_params, static),
                loss_fn,
                sample,
                jr.split(key, 1),
            )

        else:
            loss, grads = vmapped_loss_and_grad(
                eqx.combine(input_params, static), loss_fn, sample
            )

        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            input_params,
            value=loss,
        )
        new_params = eqx.apply_updates(input_params, updates)

        history = OptHistory(
            loss=loss,
            params=input_params,
            grads=grads,
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=jnp.array(1, dtype=int),
            iterations=jnp.array(1, dtype=int),
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


def step_fevals(opt_state: OptaxState, new_opt_state: OptaxState) -> jax.Array:
    """Function evaluations one optax step actually spent.

    Optimizers that search along a direction before committing to a
    step -- L-BFGS and anything else built on
    ``optax.scale_by_zoom_linesearch`` -- evaluate the objective once
    per linesearch trial, not once per iteration. Billing them a flat
    ``1`` under-reports their cost, and most severely *after* they
    converge: once the gradient bottoms out at a small but non-zero
    residual, the Wolfe curvature criterion (``|slope_step|`` against
    ``0.9 * |slope_init|``, with ``tol=0.0``) can no longer be
    satisfied exactly, so the search exhausts ``max_linesearch_steps``
    and fails out on most subsequent iterations. That tail is
    landscape-dependent -- an objective that converges to an *exactly*
    zero gradient, like a textbook quadratic, keeps costing one trial
    per step -- but on realistic problems it dominates the run. See
    ``l2co ADR 0010``.

    The count comes from the linesearch's own bookkeeping
    (``ZoomLinesearchInfo.num_linesearch_steps``, reset per outer
    update), which is exactly the number of objective evaluations:
    optax's linesearch ``step_fn`` dispatches to *either* the interval
    search *or* the zoom, each of which evaluates once and increments
    the count once. Optimizers with no linesearch carry no such field
    and keep billing ``1``.

    The extra evaluation is the cache miss in
    ``optax.value_and_grad_from_state``, which recomputes only when
    the value cached by the previous step is non-finite -- in practice
    the first step alone. This mirrors optax's own branch condition.
    It is billed *logically*: under ``eqx.filter_vmap`` the underlying
    ``lax.cond`` degrades to a ``select`` and both branches execute, so
    a batched rollout really pays for it on every step, but billing the
    logical cost keeps single and batched runs reporting the same
    number.

    Parameters
    ----------
    opt_state : OptaxState
        Optimizer state *before* the update, holding the value cached
        by the previous step.
    new_opt_state : OptaxState
        Optimizer state returned by ``optimizer.update``, holding this
        step's linesearch info.

    Returns
    -------
    jax.Array
        Scalar integer count for ``OptHistory.fevals``.
    """
    n_linesearch = optax.tree.get(new_opt_state, "num_linesearch_steps")

    # Absent for every optimizer without a linesearch (adam, sgd, ...);
    # ``optax.tree.get`` returns a Python ``None`` at trace time, so
    # this branch is static.
    if n_linesearch is None:
        return jnp.array(1, dtype=int)

    cached_value = optax.tree.get(opt_state, "value")
    if cached_value is None:
        return jnp.asarray(n_linesearch, dtype=int)

    recompute = jnp.isinf(cached_value) | jnp.isnan(cached_value)
    return jnp.asarray(n_linesearch, dtype=int) + recompute.astype(int)


def optax_extra_kwargs_fn(
    static: PyTree,
    optimizer: optax.GradientTransformationExtraArgs,
    loss_fn: LossFunction,
    bounded: tuple[float | None, float | None] | None,
    opt_hash: int,
    pass_rng: bool,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for Optax optimizers with extra
    ``update`` keyword arguments (``value``, ``grad``, ``value_fn``).

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : optax.GradientTransformationExtraArgs
        Optax optimizer instance supporting extra arguments.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float | None, float | None] or None
        Parameter bounds for clipping. ``None``, or ``None`` on one
        side, leaves that side unbounded.
    opt_hash : int
        Hash for optimizer step tracking.
    pass_rng : bool
        Whether to pass RNG key to loss function.
    **kwargs
        Additional arguments.

    Returns
    -------
    tuple[InitFunction, StepFunction]
        Initialization and step functions for optimization.
    """
    if bounded is None:
        bounded = (None, None)

    def init_fn(
        params: InputParameters, key: PRNGKeyArray = None, **sample
    ) -> OptaxState:
        """Initialise the extra-kwargs Optax optimizer state."""
        params = jax.tree.map(lambda x: x[0], params)
        opt_state = optimizer.init(params)
        return opt_state

    def step_fn(
        carry: tuple[InputParameters, OptaxState, PRNGKeyArray],
        sample: dict,
    ) -> tuple[tuple[InputParameters, OptaxState, PRNGKeyArray], OptHistory]:
        """Perform one extra-kwargs Optax step (e.g. L-BFGS)."""
        input_params, opt_state, key = carry

        input_params = jax.tree.map(
            lambda x: jnp.squeeze(x, axis=0), input_params
        )

        model = eqx.combine(input_params, static)
        if pass_rng:
            fn = Partial(loss_fn, key=key, **sample)
        else:
            fn = Partial(loss_fn, **sample)

        loss, grads = optax.value_and_grad_from_state(fn)(
            model, state=opt_state
        )

        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            input_params,
            value=loss,
            grad=grads,
            value_fn=fn,
        )

        new_params = eqx.apply_updates(input_params, updates)

        history = OptHistory(
            loss=jax.tree.map(lambda x: jnp.expand_dims(x, 0), loss),
            params=jax.tree.map(lambda x: jnp.expand_dims(x, 0), input_params),
            grads=jax.tree.map(lambda x: jnp.expand_dims(x, 0), grads),
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=step_fevals(opt_state, new_opt_state),
            iterations=jnp.array(1, dtype=int),
        )

        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        # Expand dimensions
        new_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


#                                                         UpdateClass factories
# =============================================================================


def optax_update(
    task: TaskLike,
    optimizer: Callable[..., optax.GradientTransformation],
    opt_hash: int,
    popsize: int = 1,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    name: str = "",
    **hyperparameters,
) -> UpdateClass:
    """Construct an ``UpdateClass`` for a standard optax optimizer.

    Parameters
    ----------
    task : TaskLike
        Task providing the model and loss function.
    optimizer : Callable[..., optax.GradientTransformation]
        Optax optimizer constructor (e.g. ``optax.adam``).
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    popsize : int, optional
        Population size; defaults to 1 (gradient-based optimizers
        operate on a single point).
    bounded : tuple[float | None, float | None] or None, optional
        Box bounds applied to parameters after each step. Defaults to
        ``(None, None)``; ``None``, or ``None`` on one side, leaves that
        side unbounded.
    stop_fn : StopFunction or None, optional
        Stopping function.
    name : str, optional
        Registry name, used only to resolve this optimizer's
        state-transfer overrides
        (:data:`~l2co_optimizers._src.state_transfer.TRANSFER_OVERRIDES`).
        Declared explicitly rather than left to
        ``**hyperparameters`` so it is never forwarded to the
        underlying constructor. Defaults to ``""``, which resolves
        to the family default.
    **hyperparameters
        Forwarded to the optimizer constructor.

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.
    """
    _, static = eqx.partition(task.model, eqx.is_inexact_array)

    init_fn, step_fn = optax_fn(
        static=static,
        optimizer=optimizer(**hyperparameters),
        loss_fn=task.loss_fn,
        bounded=bounded,
        opt_hash=opt_hash,
        pass_rng=task.pass_rng,
    )
    read_fn, write_fn = build_transfer_fns(
        name, FAMILY_GRADIENT, hyperparameters
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
        family=FAMILY_GRADIENT,
        transfer_read_fn=read_fn,
        transfer_write_fn=write_fn,
    )


def optax_update_extra_kwargs(
    task: TaskLike,
    optimizer: Callable[..., optax.GradientTransformationExtraArgs],
    opt_hash: int,
    popsize: int = 1,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    max_fevals_per_step: int | None = None,
    name: str = "",
    **hyperparameters,
) -> UpdateClass:
    """Construct an ``UpdateClass`` for optax optimizers with extra
    ``update`` kwargs (``value``, ``grad``, ``value_fn`` -- e.g.
    L-BFGS).

    Parameters
    ----------
    task : TaskLike
        Task providing the model and loss function.
    optimizer : Callable[..., optax.GradientTransformationExtraArgs]
        Optax optimizer constructor supporting extra update kwargs.
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    popsize : int, optional
        Population size; defaults to 1.
    bounded : tuple[float | None, float | None] or None, optional
        Box bounds applied to parameters after each step. Defaults to
        ``(None, None)``; ``None``, or ``None`` on one side, leaves that
        side unbounded.
    stop_fn : StopFunction or None, optional
        Stopping function.
    name : str, optional
        Registry name, used only to resolve this optimizer's
        state-transfer overrides
        (:data:`~l2co_optimizers._src.state_transfer.TRANSFER_OVERRIDES`).
        Declared explicitly rather than left to
        ``**hyperparameters`` so it is never forwarded to the
        underlying constructor. Defaults to ``""``, which resolves
        to the family default.
    max_fevals_per_step : int or None, optional
        Forwarded to :attr:`UpdateClass.max_fevals_per_step`. Set
        it for linesearch optimizers, whose per-iteration cost is
        not ``popsize``; ``None`` (default) leaves the bound at
        ``popsize``.
    **hyperparameters
        Forwarded to the optimizer constructor.

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.
    """
    _, static = eqx.partition(task.model, eqx.is_inexact_array)

    init_fn, step_fn = optax_extra_kwargs_fn(
        static=static,
        optimizer=optimizer(**hyperparameters),
        loss_fn=task.loss_fn,
        bounded=bounded,
        opt_hash=opt_hash,
        pass_rng=task.pass_rng,
    )
    read_fn, write_fn = build_transfer_fns(
        name, FAMILY_GRADIENT, hyperparameters
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
        family=FAMILY_GRADIENT,
        transfer_read_fn=read_fn,
        transfer_write_fn=write_fn,
        max_fevals_per_step=max_fevals_per_step,
    )


#                                                                      Registry
# =============================================================================


opt_names_normal = [
    "adabelief",
    "adadelta",
    "adan",
    "adafactor",
    "adagrad",
    "adam",
    "adamw",
    "adamax",
    "adamaxw",
    "amsgrad",
    "fromage",
    "lamb",
    "lars",
    "lion",
    "nadam",
    "nadamw",
    "noisy_sgd",
    "novograd",
    "optimistic_gradient_descent",
    "optimistic_adam",
    # "polyak_sgd", # function call signature is different, ignored!
    "radam",
    "rmsprop",
    "rprop",
    "sgd",
    "sign_sgd",
    # "signum", # not available in this version of optax!
    "sm3",
    "yogi",
]

# Optimizers needing the extra ``update`` kwargs (``value``, ``grad``,
# ``value_fn``). L-BFGS used to live here; it now has its own module
# (``l2co_optimizers._src.lbfgs``) so the ``"lbfgs"`` entry can switch
# to per-eval-key noise semantics on stochastic tasks.
opt_names_from_state: list[str] = []

normalized_optax_normal = {
    normalize_key(name): getattr(optax, name) for name in opt_names_normal
}

normalized_optax_from_state = {
    normalize_key(name): getattr(optax, name) for name in opt_names_from_state
}

optax_mapping = {
    **{
        name: Partial(optax_update, optimizer=cls, name=name)
        for name, cls in normalized_optax_normal.items()
    },
    **{
        name: Partial(
            optax_update_extra_kwargs,
            optimizer=cls,
            name=name,
        )
        for name, cls in normalized_optax_from_state.items()
    },
}
