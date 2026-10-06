"""
The ``HistoryState`` buffer: the per-iteration record of a run.

``UpdateClass.run`` / ``batch_run`` build one; rl2co appends to one
across an episode with :meth:`HistoryState.add`. Turning it into an
xarray ``Dataset`` or a ``DataLoader`` is l2co's job
(``l2co.history_to_xarray`` and friends, l2co ADR 0017).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

# Local
from l2co_optimizers._src.core.typing import OptHistoryType

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class HistoryState(eqx.Module):
    """State container for optimization history tracking.

    This class stores the optimization history including input parameters,
    output statistics, and evaluation metrics across iterations. Entries
    are written front-to-back: ``cursor`` points at the next slot to fill,
    so valid data lives in ``[0, cursor)`` and uninitialized padding (NaN
    floats / zero ints) lives in ``[cursor, n_iterations)``.

    This is the pure-JAX buffer only. Its exports (to an xarray
    ``Dataset`` or an ``l2co.DataLoader``) are free functions in l2co,
    which owns xarray, the ERT and the task labels.

    Attributes
    ----------
    output_min : Float[Array, " n_iterations"]
        Minimum output value at each iteration.
    output_mean : Float[Array, " n_iterations"]
        Mean output value at each iteration.
    output_std : Float[Array, " n_iterations"]
        Standard deviation of output values at each iteration.
    update_step : Int[Array, " n_iterations"]
        Update step number at each iteration.
    fevals : Int[Array, " n_iterations"]
        Number of function evaluations at each iteration.
    iterations : Int[Array, " n_iterations"]
        Iteration counter.
    cursor : Int[Array, ""]
        Index of the next write position; equals the number of filled
        slots. The most-recent valid entry lives at ``cursor - 1``.
    """

    output_min: Float[Array, " n_iterations"]
    output_mean: Float[Array, " n_iterations"]
    output_std: Float[Array, " n_iterations"]
    update_step: Int[Array, " n_iterations"]
    fevals: Int[Array, " n_iterations"]
    iterations: Int[Array, " n_iterations"]
    cursor: Int[Array, ""]

    @classmethod
    def init(cls, max_iterations: int) -> HistoryState:
        """Initialize a new HistoryState with NaN values.

        Parameters
        ----------
        max_iterations : int
            Number of iterations to track in history.

        Returns
        -------
        HistoryState
            Initialized history state with NaN floats, zero ints, and
            ``cursor`` set to 0 (empty buffer, next write at index 0).
        """
        nans_output = jnp.full((max_iterations,), jnp.nan, dtype=float)
        zeros_int = jnp.zeros((max_iterations,), dtype=int)

        return cls(
            output_min=nans_output,
            output_mean=nans_output,
            output_std=nans_output,
            update_step=zeros_int,
            fevals=zeros_int,
            iterations=zeros_int,
            cursor=jnp.array(0, dtype=int),
        )

    def reset(self) -> HistoryState:
        """Reset the history state to initial NaN values and cursor 0."""
        return self.init(max_iterations=self.output_min.shape[0])

    @classmethod
    def from_history(cls, history: OptHistoryType) -> HistoryState:
        """Create HistoryState from optimization history.

        Parameters
        ----------
        history : OptHistoryType
            Optimization history containing parameters and losses.

        Returns
        -------
        HistoryState
            History state populated with statistics from optimization history.
            ``cursor`` is set to ``history.loss.shape[0]`` since the buffer
            produced by the scan is fully filled.

        Notes
        -----
        The per-generation reductions are NaN-aware
        (:func:`jnp.nanmin` / :func:`jnp.nanmean` / :func:`jnp.nanstd`).
        Optimizers whose population is sized exactly to their popsize
        carry no ``NaN`` and so are unaffected (``nanmin == min``); the
        NaN-awareness matters for callers that emit a NaN-padded
        population -- e.g. the rl2co policy wrapper, whose generations
        pad slots ``[sub.popsize:]`` with ``NaN`` when the active
        sub-optimizer's popsize is below the buffer width. ``nanstd``
        of a single valid entry is ``0.0``.
        """
        # History has shape (n_iter, pop)
        # Compute statistics across P dimension for each time step
        # (NaN-aware: padded candidate slots are ignored, see Notes).
        return cls(
            output_min=jnp.nanmin(history.loss, axis=1),
            output_mean=jnp.nanmean(history.loss, axis=1),
            output_std=jnp.nanstd(history.loss, axis=1),
            update_step=history.update_step,
            fevals=history.fevals,
            iterations=history.iterations,
            cursor=jnp.array(history.loss.shape[0], dtype=int),
        )

    @classmethod
    def from_reduced(
        cls,
        output_min: Float[Array, " n_iterations"],
        output_mean: Float[Array, " n_iterations"],
        output_std: Float[Array, " n_iterations"],
        update_step: Int[Array, " n_iterations"],
        fevals: Int[Array, " n_iterations"],
        iterations: Int[Array, " n_iterations"],
    ) -> HistoryState:
        """Build a HistoryState from already-reduced per-iteration fields.

        Companion to :meth:`from_history` for the reduce-in-scan rollout
        path (:meth:`UpdateClass.run` /
        :meth:`UpdateClass.batch_run_fused`): those drivers reduce
        each generation's population *inside* the scan and emit the
        ``output_min / output_mean / output_std`` statistics directly,
        rather than stacking the full ``(n_iter, popsize, dim)``
        :class:`OptHistory` and reducing it afterwards. The per-iteration
        NaN-aware reductions are identical to :meth:`from_history`; this
        constructor just skips the intermediate stacked trajectory.

        Each argument is a fully-filled ``(n_iterations,)`` buffer, so
        ``cursor`` is set to ``output_min.shape[0]`` exactly as
        :meth:`from_history` does.

        Parameters
        ----------
        output_min, output_mean, output_std : Float[Array, " n_iterations"]
            Per-iteration NaN-aware min / mean / std of the population
            loss.
        update_step, fevals, iterations : Int[Array, " n_iterations"]
            Per-iteration integer counters (non-cumulative; the cumulative
            sums are applied on export, see ``l2co.history_to_xarray``).

        Returns
        -------
        HistoryState
            History state populated with the given per-iteration fields.
        """
        return cls(
            output_min=output_min,
            output_mean=output_mean,
            output_std=output_std,
            update_step=update_step,
            fevals=fevals,
            iterations=iterations,
            cursor=jnp.array(output_min.shape[0], dtype=int),
        )

    def add(self, other: HistoryState) -> HistoryState:
        """
        Insert ``other``'s entries front-to-back at ``self.cursor``.

        Each field of ``other`` (length ``K``) is written into ``self`` at
        positions ``[cursor, cursor + K)`` via ``dynamic_update_slice``;
        the cursor is advanced by ``K``. Entries before ``cursor`` are
        preserved unchanged. Existing padding past ``cursor + K`` is left
        untouched (NaN floats, zero ints).

        Precondition: ``self.cursor + K <= n_iterations``. The buffer is
        sized to fit the full episode at the call site, so this is
        guaranteed by construction in current callers (rl2co allocates
        ``HistoryState.init(max_iterations + 1)``).

        Parameters
        ----------
        other : HistoryState
            HistoryState whose entries are appended at the cursor.

        Returns
        -------
        HistoryState
            New HistoryState with ``other``'s entries written at the cursor
            and ``cursor`` advanced by ``other.output_min.shape[0]``.
        """
        cursor = self.cursor

        def write(a, b):
            return jax.lax.dynamic_update_slice(a, b, (cursor,))

        return HistoryState(
            output_min=write(self.output_min, other.output_min),
            output_mean=write(self.output_mean, other.output_mean),
            output_std=write(self.output_std, other.output_std),
            update_step=write(self.update_step, other.update_step),
            fevals=write(self.fevals, other.fevals),
            iterations=write(self.iterations, other.iterations),
            cursor=cursor + other.output_min.shape[0],
        )
