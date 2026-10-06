"""
Per-optimizer run state, and the rollout entry points built on it.

:class:`RunState` bundles one optimizer's population, running best,
optimizer state and :class:`UpdateClass`. :meth:`RunState.init` builds
it from an already-resolved :class:`UpdateClass`, a model and a dataset;
resolving an :class:`OptimizationStep` against a task is l2co's job
(``l2co.init_run_state``). :func:`reset`, :func:`run`, :func:`batch_run` and
:func:`batch_evaluate` hand the state to the ``UpdateClass`` run loop
and rewrap what it returns. Moved here from l2co (ADR 0017 there).
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
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

# Local
from l2co_optimizers._src.core.batching import BatchState
from l2co_optimizers._src.core.history_state import HistoryState
from l2co_optimizers._src.core.model_evaluation import evaluate
from l2co_optimizers._src.core.typing import (
    InputParameters,
    SamplerFunction,
)
from l2co_optimizers._src.core.update_class import UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class RunState(eqx.Module):
    """
    State container for running optimization processes.
    This class encapsulates the state required to run optimization processes,
    including the update state, history state, optimizer state, batch state,
    and update class. It provides methods for initializing and running the
    optimization process in a batched manner for multiple realizations.

    Parameters
    ----------
    params : InputParameters
        The current input parameters being optimized, shaped as (popsize, dim).
    best_params : Float[PyTree, " dimensionality"]
        The best input parameters found so far, shaped as (dim,): a
        single point, without the popsize axis of ``params``.
    best_loss : Array[Float, ""]
        The best loss value found so far.
    opt_state : PyTree
        The state of the optimizer.
    update_class : UpdateClass
        The class defining the optimizer update logic.

    Notes
    -----
    The ``RunState`` returned by :func:`batch_evaluate` / :func:`batch_run`
    carries a leading ``n_realizations`` axis on every array field.
    """

    params: InputParameters
    best_params: Float[PyTree, " dimensionality"]
    best_loss: Array[Float, ""]
    opt_state: PyTree
    update_class: UpdateClass = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        update_class: UpdateClass,
        *,
        model: PyTree,
        dataset: dict[str, Array],
        batch_size: int | None,
        key: PRNGKeyArray,
    ) -> RunState:
        """Initialize a new RunState.

        Parameters
        ----------
        update_class : UpdateClass
            The optimizer to run, already built by its factory (bounds,
            stopping and hyperparameters are baked in there).
        model : PyTree
            Model whose inexact-array leaves are optimized; they seed
            every population member and ``best_params``.
        dataset : dict[str, Array]
            The dataset the loss is evaluated on (empty for a
            data-free problem).
        batch_size : int or None
            Mini-batch size drawn from ``dataset`` per evaluation;
            ``None`` for the full batch.
        key : PRNGKeyArray
            PRNG key for random operations.

        Returns
        -------
        RunState
            Initialized run state.
        """
        model_params = eqx.filter(model, eqx.is_inexact_array)

        # Initialize batch state
        batch_state = BatchState.init(
            dataset=dataset,
            batch_size=batch_size,
            key=key,
        )

        params = jax.tree.map(
            lambda x: jnp.repeat(x[None, ...], update_class.popsize, axis=0),
            model_params,
        )

        # Initialize the optimizer state. RandomSearchUpdateClass carries
        # a no-op ``init_fn`` that returns a scalar placeholder, so the
        # inherited ``init_state`` works for it too -- no dispatch needed.
        opt_state = update_class.init_state(
            params=params,
            batch_state=batch_state,
            dataset=dataset,
            key=key,
        )

        return cls(
            params=params,
            best_params=model_params,
            best_loss=jnp.array(jnp.inf, dtype=float),
            opt_state=opt_state,
            update_class=update_class,
        )


def reset(
    run_state: RunState,
    batch_state: BatchState,
    static: PyTree,
    dataset: dict[str, jax.Array],
    loss_fn: Callable,
    pass_rng: bool,
    sampler: SamplerFunction,
    key: PRNGKeyArray,
) -> RunState:
    """Reset the run state with new samples.

    Parameters
    ----------
    run_state : RunState
        Current run state.
    batch_state : BatchState
        Current batch state.
    static : PyTree
        Static (non-inexact-array) partition of the model.
    dataset : dict[str, jax.Array]
        Dataset for evaluation.
    loss_fn : Callable
        Loss function for evaluation.
    pass_rng : bool
        Whether to pass RNG keys to the loss function.
    sampler : SamplerFunction
        Sampling function for generating new parameters.
    key : PRNGKeyArray
        PRNG key for random operations.

    Returns
    -------
    RunState
        Reset run state with new samples.
    """
    samples = sampler(
        key,
        run_state.best_params,
        run_state.update_class.popsize,
    )

    loss = evaluate(
        eqx.combine(static, samples),
        batch_state,
        dataset,
        loss_fn,
        pass_rng,
        key,
        jr.split(key, run_state.update_class.popsize),
    )

    # calculate the best params and best loss
    best_idx = jnp.argmin(loss)
    best_params = jax.tree.map(lambda x: x[best_idx], samples)
    best_loss = loss[best_idx]

    opt_state = run_state.update_class.init_state(
        params=samples,
        batch_state=batch_state,
        dataset=dataset,
        key=key,
    )

    return RunState(
        params=samples,
        best_params=best_params,
        best_loss=best_loss,
        opt_state=opt_state,
        update_class=run_state.update_class,
    )


batch_reset = eqx.filter_vmap(
    reset,
    in_axes=(None, None, None, None, None, None, None, 0),
)


# =============================================================================


def run(
    run_state: RunState,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    n_iterations: int,
    key: PRNGKeyArray,
    verbose: bool,
) -> tuple[RunState, BatchState, HistoryState]:
    """Run optimization for a specified number of iterations.

    Hands off to :meth:`UpdateClass.run`, which a subclass may override
    (``RandomSearchUpdateClass`` does).

    Parameters
    ----------
    run_state : RunState
        Current run state.
    batch_state : BatchState
        Current batch state.
    dataset : dict[str, jax.Array]
        Dataset for evaluation.
    n_iterations : int
        Number of iterations to run.
    key : PRNGKeyArray
        PRNG key for random operations.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    tuple[RunState, BatchState, HistoryState]
        Updated RunState, BatchState and history state after optimization.
    """
    (params, best_params, best_loss, opt_state, batch_state, history_state) = (
        run_state.update_class.run(
            opt_state=run_state.opt_state,
            params=run_state.params,
            batch_state=batch_state,
            dataset=dataset,
            key=key,
            n_iterations=n_iterations,
            verbose=verbose,
        )
    )

    run_state = RunState(
        params=params,
        best_params=best_params,
        best_loss=best_loss,
        opt_state=opt_state,
        update_class=run_state.update_class,
    )

    return run_state, batch_state, history_state


def batch_run(
    run_state: RunState,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    n_iterations: int,
    key: PRNGKeyArray,
    verbose: bool,
) -> tuple[RunState, BatchState, HistoryState]:
    """Run ``n_realizations`` trajectories, on whichever driver fits.

    Hands off to :meth:`UpdateClass.batch_run`, which picks the driver:
    one fused scan for a plain optimizer, one realization at a time for
    an optimizer marked :attr:`UpdateClass.sequential_realizations` (the
    meta-optimizers) or for random search. That method's docstring
    records why, and what the two drivers do and do not agree on.

    Parameters
    ----------
    run_state : RunState
        Run state carrying a leading ``n_realizations`` axis on every
        array field.
    batch_state : BatchState
        Single (un-batched) batch state.
    dataset : dict[str, jax.Array]
        Dataset for evaluation.
    n_iterations : int
        Number of iterations to run.
    key : PRNGKeyArray
        Per-realization PRNG keys, shape ``(n_realizations,)``.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    tuple[RunState, BatchState, HistoryState]
        Updated RunState, BatchState and history state, each carrying
        the realization axis.
    """
    (params, best_params, best_loss, opt_state, batch_state, history_state) = (
        run_state.update_class.batch_run(
            opt_state=run_state.opt_state,
            params=run_state.params,
            batch_state=batch_state,
            dataset=dataset,
            key=key,
            n_iterations=n_iterations,
            verbose=verbose,
        )
    )

    run_state = RunState(
        params=params,
        best_params=best_params,
        best_loss=best_loss,
        opt_state=opt_state,
        update_class=run_state.update_class,
    )

    return run_state, batch_state, history_state


# =============================================================================


def batch_evaluate(
    run_state: RunState,
    batch_state: BatchState,
    static: PyTree,
    dataset: dict[str, jax.Array],
    loss_fn: Callable,
    sampler: Callable,
    n_iterations: int,
    pass_rng: bool,
    key: PRNGKeyArray,
    verbose: bool,
) -> tuple[RunState, BatchState, HistoryState]:
    """Evaluate optimization by resetting and running to completion.

    Parameters
    ----------
    run_state : RunState
        Current run state.
    batch_state : BatchState
        Current batch state.
    static : PyTree
        Static (non-inexact-array) partition of the model.
    dataset : dict[str, jax.Array]
        Dataset for evaluation.
    loss_fn : Callable
        Loss function for evaluation.
    sampler : Callable
        Sampling function for generating new parameters.
    n_iterations : int
        Number of iterations to run.
    pass_rng : bool
        Whether to pass RNG keys to the loss function.
    key : PRNGKeyArray
        PRNG key for random operations.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    tuple[RunState, BatchState, HistoryState]
        Tuple of updated RunState, BatchState and History state after
        complete optimization run.
    """
    run_state = batch_reset(
        run_state,
        batch_state,
        static,
        dataset,
        loss_fn,
        pass_rng,
        sampler,
        key,
    )
    return batch_run(
        run_state=run_state,
        batch_state=batch_state,
        dataset=dataset,
        n_iterations=n_iterations,
        key=key,
        verbose=verbose,
    )
