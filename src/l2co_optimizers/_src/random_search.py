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
import jax.random as jr
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray, PyTree

# Local
from l2co_optimizers._src.batching import BatchState
from l2co_optimizers._src.history_state import HistoryState
from l2co_optimizers._src.loss import vmapped_loss, vmapped_loss_with_rng
from l2co_optimizers._src.popsize import resolve_popsize
from l2co_optimizers._src.sampler import get_sampler
from l2co_optimizers._src.state_transfer import FAMILY_POPULATION
from l2co_optimizers._src.typing import (
    InputParameters,
    LossFunction,
    PopSize,
    StopFunction,
)
from l2co_optimizers._src.update_class import RunResult, UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# Target number of candidates per random-search chunk. Peak memory for a
# chunk is roughly this many rows times the parameter dimension, times a
# small constant for the RNG generation. ~1e5 keeps a 1024-dim chunk well
# under 1 GiB while staying large enough that the per-chunk vectorised
# evaluation is efficient.
_RS_CHUNK_TARGET_CANDIDATES = 100_000


def _rs_chunking(n_iterations: int, popsize: int) -> tuple[int, int]:
    """Pick ``(n_chunks, chunk_iters)`` for chunked random search.

    Splits the ``n_iterations`` logical iterations into ``n_chunks``
    balanced chunks of ``chunk_iters`` iterations each such that a chunk
    holds about :data:`_RS_CHUNK_TARGET_CANDIDATES` candidates. Both
    values are static (``n_iterations``/``popsize`` are Python ints), so
    they fix shapes at trace time. ``n_chunks * chunk_iters`` may slightly
    exceed ``n_iterations`` (by fewer than ``n_chunks`` iterations); the
    caller masks/drops that padding.
    """
    target_iters = max(1, _RS_CHUNK_TARGET_CANDIDATES // max(popsize, 1))
    n_chunks = max(1, -(-n_iterations // target_iters))  # ceil div
    chunk_iters = -(-n_iterations // n_chunks)  # ceil div -> balanced
    return n_chunks, chunk_iters


class RandomSearchUpdateClass(UpdateClass):
    """One-shot random-search update class.

    Subclasses :class:`UpdateClass`, overriding :meth:`run` with a
    chunked runner that evaluates the ``n_iterations * popsize``
    candidate set in vectorised chunks rather than iterating through the
    scan loop, and :meth:`batch_run` to map that runner over
    realizations one at a time. ``init_fn`` is a no-op that returns a
    scalar placeholder -- it is invoked through the inherited
    :meth:`~UpdateClass.init_state` so callers do not need a special
    initialization path. ``step_fn`` is a no-op as well and is never
    invoked, since :meth:`run` bypasses the scan loop entirely.

    The runner shapes its result to mimic an iterative trajectory:
    ``output_min[i]`` / ``output_mean[i]`` / ``output_std[i]`` are
    statistics over the ``popsize`` samples drawn for iteration ``i``
    (independent samples per iteration, not cumulative across earlier
    iterations).

    Attributes
    ----------
    sampling_fn : Callable
        Closure ``(key, n_samples) -> InputParameters`` that draws
        ``n_samples`` candidates with leaves shaped like the
        model parameters.
    static_model : PyTree
        The non-inexact-array partition of ``model``; recombined
        with sampled candidates via ``eqx.combine`` before evaluation.
    loss_fn : Callable
        The loss the factory was handed.
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
        invoked through the inherited :meth:`~UpdateClass.init_state`
        so the random-search code path needs no special initialization
        helper. ``step_fn`` is a no-op as well and is never invoked --
        the overridden :meth:`run` does all the work in chunked batched
        evaluations.

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

    def run(
        self,
        opt_state: PyTree,
        params: InputParameters,
        batch_state: BatchState,
        dataset: dict[str, jax.Array],
        key: PRNGKeyArray,
        n_iterations: int,
        verbose: bool,
    ) -> RunResult:
        """Run random search over the candidate set, one chunk per scan step.

        Draws ``n_iterations * popsize`` independent candidates from
        :attr:`sampling_fn`, evaluates them via :func:`vmapped_loss` (or
        :func:`vmapped_loss_with_rng` for stochastic losses), and reports
        one ``output_min/mean/std`` per logical iteration plus the global
        best. Rather than materialise and evaluate the whole set at once
        -- which peaks at several times its own size (``jax.random``
        uniform generation alone spikes to ~4x the result while producing
        threefry bits), the cause of out-of-memory failures on
        high-dimensional tasks -- the candidates are processed in chunks
        of :func:`_rs_chunking` iterations under a :func:`jax.lax.scan`.
        Each chunk's candidate array is transient (not part of the scan
        carry) and freed before the next step, so peak memory is
        ``O(chunk_iters * popsize * dim)`` independent of the total
        budget. The scan-loop machinery of :meth:`UpdateClass.run` is
        still bypassed within a chunk: candidates are independent and
        evaluated in one vectorised call. Per-iteration reductions and
        the global best are identical to a single-shot evaluation of the
        same per-chunk draws.

        Parameters
        ----------
        opt_state : PyTree
            Ignored (random search keeps no state); returned unchanged.
        params : InputParameters
            Ignored (each iteration re-samples from scratch).
        batch_state : BatchState
            Used to draw one batch; advanced once.
        dataset : dict[str, jax.Array]
            Task dataset.
        key : PRNGKeyArray
            PRNG key, split into batch / sampling / per-candidate-loss
            sub-keys.
        n_iterations : int
            Number of logical iterations to emit in the trajectory.
        verbose : bool
            Ignored.

        Returns
        -------
        RunResult
            ``(final_params, best_params, best_loss, opt_state,
            batch_state, history_state)`` matching
            :meth:`UpdateClass.run`.
        """
        del params, verbose
        popsize = self.popsize
        n_chunks, chunk_iters = _rs_chunking(n_iterations, popsize)
        padded_iters = n_chunks * chunk_iters  # >= n_iterations
        chunk_size = chunk_iters * popsize

        batch_key, sample_key, loss_key = jr.split(key, 3)

        # One batch reused for all evaluations.
        batch_idxs, batch_state = batch_state.next(batch_key)
        sample = jax.tree.map(lambda x: x[batch_idxs], dataset)

        # Independent per-chunk sampling / loss keys.
        sample_keys = jr.split(sample_key, n_chunks)
        loss_keys = jr.split(loss_key, n_chunks)

        # ``bounded`` is a static field, so this is resolved at trace time.
        # With ``(None, None)`` -- the default for the random-search
        # baseline -- ``jnp.clip`` is a no-op that still allocates a full
        # copy of the chunk, so skip it unless a finite bound is set.
        lo, hi = self.bounded
        do_clip = lo is not None or hi is not None

        def _chunk(carry, xs):
            c_idx, s_key, l_key = xs

            candidates = self.sampling_fn(s_key, chunk_size)
            if do_clip:
                candidates = jax.tree.map(
                    lambda p: jnp.clip(p, lo, hi), candidates
                )
            combined = eqx.combine(candidates, self.static_model)
            if self.pass_rng:
                losses = vmapped_loss_with_rng(
                    combined,
                    self.loss_fn,
                    sample,
                    jr.split(l_key, chunk_size),
                )
            else:
                losses = vmapped_loss(combined, self.loss_fn, sample)

            # Mask candidates in padded (out-of-range) iterations so they
            # can neither win the global best nor pollute the trajectory.
            valid_iter = (
                c_idx * chunk_iters + jnp.arange(chunk_iters)
            ) < n_iterations
            valid_cand = jnp.repeat(valid_iter, popsize)
            masked = jnp.where(valid_cand, losses, jnp.inf)

            # Per-chunk best (``argmin`` keeps the earliest candidate on
            # ties).
            best_idx = jnp.argmin(masked)
            chunk_best_loss = masked[best_idx]
            chunk_best_params = jax.tree.map(lambda x: x[best_idx], candidates)

            losses_2d = losses.reshape(chunk_iters, popsize)

            # Population at this chunk's last in-range iteration. Only the
            # final chunk's value is used downstream, giving the last
            # logical iteration's ``popsize`` block (matches
            # ``candidates[-popsize:]`` of the previous one-shot
            # implementation).
            last_valid_local = jnp.clip(
                n_iterations - 1 - c_idx * chunk_iters, 0, chunk_iters - 1
            )
            last_pop = jax.tree.map(
                lambda x: jax.lax.dynamic_slice_in_dim(
                    x, last_valid_local * popsize, popsize, axis=0
                ),
                candidates,
            )
            return carry, (
                losses_2d,
                last_pop,
                chunk_best_loss,
                chunk_best_params,
            )

        _, (losses_2d_all, last_pops, chunk_best_losses, chunk_best_params) = (
            jax.lax.scan(
                _chunk,
                (),
                (jnp.arange(n_chunks), sample_keys, loss_keys),
            )
        )

        # Reduce the flat candidate losses per logical iteration directly
        # into the HistoryState (dropping the padded tail). Building an
        # intermediate ``(n_iter, popsize, dim)`` ``OptHistory`` -- and an
        # all-NaN ``grads`` array of the same shape -- only to collapse it
        # via ``from_history`` wastes memory scaling with ``dim``;
        # random-search candidates carry no NaN padding, so per-iteration
        # ``nanmin/nanmean/nanstd`` match exactly.
        losses_2d = losses_2d_all.reshape(padded_iters, popsize)[:n_iterations]
        history_state = HistoryState.from_reduced(
            jnp.nanmin(losses_2d, axis=1),
            jnp.nanmean(losses_2d, axis=1),
            jnp.nanstd(losses_2d, axis=1),
            jnp.full((n_iterations,), self.hash, dtype=int),
            jnp.full((n_iterations,), popsize, dtype=int),
            jnp.ones((n_iterations,), dtype=int),
        )

        # Global best across all chunks (earliest chunk wins ties).
        best_chunk = jnp.argmin(chunk_best_losses)
        best_loss = chunk_best_losses[best_chunk]
        best_params = jax.tree.map(lambda x: x[best_chunk], chunk_best_params)

        # "Current population" returned to RunState -- the last logical
        # iteration's ``popsize`` candidates, matching ``UpdateClass.run``.
        final_params = jax.tree.map(lambda x: x[-1], last_pops)

        return (
            final_params,
            best_params,
            best_loss,
            opt_state,
            batch_state,
            history_state,
        )

    def batch_run(
        self,
        opt_state: PyTree,
        params: InputParameters,
        batch_state: BatchState,
        dataset: dict[str, jax.Array],
        key: PRNGKeyArray,
        n_iterations: int,
        verbose: bool,
    ) -> RunResult:
        """Run ``n_realizations`` random searches, one at a time.

        Always :meth:`~UpdateClass.batch_run_sequential`, which maps the
        overridden :meth:`run` over realizations with
        :func:`jax.lax.map`, whatever :attr:`sequential_realizations`
        says. The reason is peak memory, not branching: a vmap would
        materialise every realization's ``(n_iterations * popsize, dim)``
        candidate chunk concurrently, so peak memory would scale with
        ``n_realizations``, while ``lax.map`` keeps one realization's
        chunk live at a time. Results match a vmap because the
        per-realization ``key`` slices -- and therefore the drawn
        candidates -- are unchanged; only execution order differs (up to
        float32 reduction-ordering noise XLA does not pin across the two
        lowerings).

        Parameters and returns are those of
        :meth:`UpdateClass.batch_run`. ``opt_state`` is a
        per-realization scalar placeholder, shape ``(n_realizations,)``,
        returned unchanged, and ``params`` is read only for its
        realization axis.
        """
        return self.batch_run_sequential(
            opt_state=opt_state,
            params=params,
            batch_state=batch_state,
            dataset=dataset,
            key=key,
            n_iterations=n_iterations,
            verbose=verbose,
        )


def random_search_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
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
    model : PyTree
        Model whose inexact-array leaves are optimized; the rest is
        recombined as static structure.
    loss_fn : LossFunction
        ``loss_fn(model, **sample)`` -- or ``loss_fn(model, key=key,
        **sample)`` when ``pass_rng`` -- returning a scalar loss.
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword (a stochastic loss).
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
    popsize : int or Callable[[int], int], optional
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
        A :class:`RandomSearchUpdateClass` ready to slot into a
        :class:`~l2co_optimizers.RunState`.
    """
    del stop_fn  # accepted but unused; see docstring
    popsize = resolve_popsize(popsize, model)
    model_params, static_model = eqx.partition(model, eqx.is_inexact_array)
    sampler_fn = get_sampler(sampler)

    def sampling_fn(key: PRNGKeyArray, n_samples: int) -> InputParameters:
        """Draw ``n_samples`` candidates shaped like ``model``."""
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
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        bounded=bounded,
        popsize=popsize,
        hash=opt_hash,
        family=FAMILY_POPULATION,
    )


# =============================================================================

random_search_mapping = {"randomsearch": Partial(random_search_update)}
