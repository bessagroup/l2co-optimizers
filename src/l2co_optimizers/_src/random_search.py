"""
Module for the one-shot random search optimizer.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray, PyTree

from l2co_optimizers._src.popsize import resolve_popsize
from l2co_optimizers._src.sampler import get_sampler

# Local
from l2co_optimizers._src.state_transfer import FAMILY_POPULATION
from l2co_optimizers._src.typing import (
    InputParameters,
    PopSize,
    StopFunction,
    TaskLike,
)
from l2co_optimizers._src.update_class import UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class RandomSearchUpdateClass(UpdateClass):
    """One-shot random-search update class.

    Subclasses :class:`UpdateClass` but stores the extra fields needed
    by the dedicated runners
    :func:`l2co._src.update_class.run_randomsearch` and
    :func:`l2co._src.update_class.batch_run_randomsearch`, which
    evaluate the entire ``n_iterations * popsize`` candidate set in a
    single batched call rather than iterating through the scan loop.
    ``run_state.run`` / ``run_state.batch_run`` dispatch to those
    runners when they see an instance of this class. ``init_fn`` is a
    no-op that returns a scalar placeholder -- it is invoked through
    the generic :func:`l2co._src.update_class.init_` so callers do not
    need a special initialization path. ``step_fn`` is a no-op as well
    and is never invoked, since the dedicated runners bypass the scan
    loop entirely.

    The runner shapes its result to mimic an iterative trajectory:
    ``output_min[i]`` / ``output_mean[i]`` / ``output_std[i]`` are
    statistics over the ``popsize`` samples drawn for iteration ``i``
    (independent samples per iteration, not cumulative across earlier
    iterations).

    Attributes
    ----------
    sampling_fn : Callable
        Closure ``(key, n_samples) -> InputParameters`` that draws
        ``n_samples`` candidates with leaves shaped like the task's
        model parameters.
    static_model : PyTree
        The non-inexact-array partition of ``task.model``; recombined
        with sampled candidates via ``eqx.combine`` before evaluation.
    loss_fn : Callable
        Loss function taken from ``task.loss_fn``.
    pass_rng : bool
        Whether the loss function accepts a ``key`` keyword argument.
    bounded : tuple[float | None, float | None]
        Box bounds applied to every sampled candidate.
    popsize : int
        Number of candidates drawn per logical iteration. Total
        evaluations per ``run`` call equal ``popsize * n_iterations``.
    hash : int
        Stable optimizer hash from the originating ``OptimizationStep``.
    """

    sampling_fn: Callable = eqx.field(static=True)
    static_model: PyTree
    loss_fn: Callable = eqx.field(static=True)
    pass_rng: bool = eqx.field(static=True)
    bounded: tuple = eqx.field(static=True)

    def __init__(
        self,
        sampling_fn: Callable,
        static_model: PyTree,
        loss_fn: Callable,
        pass_rng: bool,
        bounded: tuple[float | None, float | None],
        popsize: int,
        hash: int,
        family: str = FAMILY_POPULATION,
    ):
        """Initialise the random-search update class.

        ``init_fn`` is a no-op returning a scalar placeholder; it is
        invoked through the generic :func:`l2co._src.update_class.init_`
        so the random-search code path needs no special initialization
        helper. ``step_fn`` is a no-op as well and is never invoked --
        ``run_state.run`` / ``run_state.batch_run`` dispatch to the
        dedicated runners :func:`l2co._src.update_class.run_randomsearch`
        / :func:`l2co._src.update_class.batch_run_randomsearch`, which
        do all the work in a single batched evaluation.

        Parameters
        ----------
        sampling_fn : Callable
            ``(key, n_samples) -> InputParameters`` candidate sampler.
        static_model : PyTree
            Non-inexact-array partition of the task model.
        loss_fn : Callable
            Loss function from the task.
        pass_rng : bool
            Whether ``loss_fn`` takes a ``key`` keyword argument.
        bounded : tuple[float | None, float | None]
            Box bounds for sampled candidates.
        popsize : int
            Candidates per logical iteration.
        hash : int
            Optimizer hash stamped into ``OptHistory.update_step``.
        family : str, optional
            Which optimizer family this is, by default
            :data:`~l2co.optimizers.FAMILY_POPULATION`. Declared
            explicitly because the ``UpdateClass`` default is
            ``gradient``, which would misdescribe it -- and because this
            subclass defines its own ``__init__``, the field has to be
            threaded through by hand rather than inherited.
        """

        def _noop_init_fn(
            params: InputParameters,
            key: PRNGKeyArray | None = None,
            **sample: Any,
        ) -> jax.Array:
            return jnp.array(0, dtype=int)

        def _noop_step_fn(carry, sample):
            return carry, None

        super().__init__(
            init_fn=_noop_init_fn,
            step_fn=_noop_step_fn,
            popsize=popsize,
            hash=hash,
            family=family,
        )
        self.sampling_fn = sampling_fn
        self.static_model = static_model
        self.loss_fn = loss_fn
        self.pass_rng = pass_rng
        self.bounded = bounded


def random_search_update(
    task: TaskLike,
    opt_hash: int,
    bounded: tuple[float | None, float | None] = (None, None),
    stop_fn: StopFunction | None = None,
    sampler: str = "random",
    popsize: PopSize = 1,
    **sampler_kwargs: Any,
) -> UpdateClass:
    """Factory for the one-shot random-search optimizer.

    The returned :class:`RandomSearchUpdateClass` evaluates all
    ``n_iterations * popsize`` candidates in a single batched call,
    bypassing the scan loop used by the Optax / EvoSax wrappers -- safe
    here because every random-search candidate is independent.

    Parameters
    ----------
    task : TaskLike
        Task to optimize (provides ``model``, ``loss_fn``, ``pass_rng``).
    opt_hash : int
        Hash from the originating ``OptimizationStep``; stamped into
        ``OptHistory.update_step`` for downstream tracking.
    bounded : tuple[float | None, float | None], optional
        Box bounds applied to every sampled candidate, by default
        ``(None, None)`` (no clipping).
    stop_fn : StopFunction or None, optional
        Accepted for interface compatibility but **ignored** -- there is
        no inner loop to short-circuit. Present so callers building this
        through ``optimizer_mapping`` and ``OptimizationStep.stopping_fn``
        need no special case.
    sampler : str, optional
        Name of the sampler used to draw candidates, resolved via
        :func:`l2co_optimizers._src.sampler.get_sampler`. Must accept ``(key,
        params, n_samples, ...)`` -- i.e. ``"random"``, ``"normal"``,
        ``"xavier"``, or ``"constant"``. Default ``"random"``.
    popsize : int or Callable[[TaskLike], int], optional
        Candidates per logical iteration; total evaluations per
        ``step`` call equal ``popsize * n_iterations``. Either a
        literal integer or a callable taking the ``Task`` (e.g.
        :func:`l2co_optimizers._src.popsize.variable_popsize`).
        Default 1.
    **sampler_kwargs : Any
        Forwarded to the underlying sampler (e.g. ``lower_bound`` /
        ``upper_bound`` for ``"random"``; ``mean`` / ``std`` for
        ``"normal"``).

    Returns
    -------
    UpdateClass
        A :class:`RandomSearchUpdateClass` ready to slot into
        :class:`l2co._src.run_state.RunState`.
    """
    del stop_fn  # accepted but unused; see docstring
    popsize = resolve_popsize(popsize, task)
    model_params, static_model = eqx.partition(
        task.model, eqx.is_inexact_array
    )
    sampler_fn = get_sampler(sampler)

    def sampling_fn(key: PRNGKeyArray, n_samples: int) -> InputParameters:
        """Draw ``n_samples`` candidates shaped like ``task.model``."""
        return sampler_fn(
            key,
            params=model_params,
            n_samples=n_samples,
            **sampler_kwargs,
        )

    # Random search carries no state between iterations, so there is
    # nothing for a switch to hand over or receive: ``transfer_read_fn``
    # and ``transfer_write_fn`` stay ``None``. ``family`` is still set
    # explicitly rather than left to the ``UpdateClass`` default, which
    # is ``gradient`` and would misdescribe it.
    return RandomSearchUpdateClass(
        sampling_fn=sampling_fn,
        static_model=static_model,
        loss_fn=task.loss_fn,
        pass_rng=task.pass_rng,
        bounded=bounded,
        popsize=popsize,
        hash=opt_hash,
        family=FAMILY_POPULATION,
    )


# =============================================================================

random_search_mapping = {"randomsearch": Partial(random_search_update)}
