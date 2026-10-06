"""
The ``UpdateClass`` adapter and the run loop that drives it.

``UpdateClass`` is the static, JIT-friendly container for one optimizer,
and its methods are the loop that runs it on a dataset: ``init_state``,
``step`` / ``batch_step``, ``run``, and the multi-realization drivers
``batch_run`` / ``batch_run_fused`` / ``batch_run_sequential`` (l2co ADR
0017). The factories that build one from a model and loss plus an optimizer
specification live with their library -- ``optax_update`` /
``optax_update_extra_kwargs`` in
:mod:`~l2co_optimizers._src.optax_implementations`,
``evosax_distribution_update`` / ``evosax_population_update`` in
:mod:`~l2co_optimizers._src.evosax_implementations`, and the other
built-ins (``lbfgs_update``, ``shade_update``, ``random_search_update``,
...) in their own modules.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from typing import ClassVar

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

# Local
from l2co_optimizers._src.core.batching import BatchState
from l2co_optimizers._src.core.history_state import HistoryState
from l2co_optimizers._src.core.opt_history import OptHistory, RecentHistory
from l2co_optimizers._src.core.state_transfer import (
    FAMILY_GRADIENT,
)
from l2co_optimizers._src.core.typing import (
    AskFunction,
    Carry,
    InitFunction,
    InputParameters,
    StepFunction,
    StopFunction,
    TransferReadFunction,
    TransferWriteFunction,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: What every run driver returns: ``(params, best_params, best_loss,
#: opt_state, batch_state, history_state)``. ``params`` is the last
#: evaluated population; ``best_*`` is the best member seen.
RunResult = tuple[
    InputParameters,
    InputParameters,
    Float[Array, ""],
    PyTree,
    BatchState,
    HistoryState,
]

#: Width of the rolling ``RecentHistory`` window ``stop_fn`` sees.
_RECENT_HISTORY_SIZE = 20


#                                                                   UpdateClass
# =============================================================================


class UpdateClass(eqx.Module):
    """One optimizer step, and the loop that runs it.

    Encapsulates the initialization function, step function, population
    size, optimizer hash, and stopping criterion for a single optimizer.
    Constructed via a factory function such as ``optax_update`` /
    ``evosax_*_update`` (or a subclass like ``RandomSearchUpdateClass``).
    The methods run it on a dataset: :meth:`init_state` builds the
    optimizer state, :meth:`run` drives one trajectory, and
    :meth:`batch_run` drives ``n_realizations`` of them.

    Attributes
    ----------
    init_fn : InitFunction
        Function to initialize optimizer state.
    step_fn : StepFunction
        Function to perform a single optimization step.
    popsize : int
        Population size for the optimizer (1 for gradient-based).
    hash : int
        Stable hash identifier for the optimizer configuration.
    stop_fn : StopFunction or None
        ``stop_fn(recent_history, opt_state) -> bool`` -- early-stopping
        check, evaluated before every step. ``None`` (default) never
        stops, and is resolved at trace time in :meth:`step`, so it
        costs nothing.
    max_fevals_per_step : int or None
        Most function evaluations this optimizer can bill in one
        iteration, or ``None`` (default) when that is just ``popsize``.
        Only linesearch optimizers set it: L-BFGS bills one evaluation
        per linesearch trial, so it can spend far more than its
        population of one (see :func:`step_fevals` and
        ``l2co ADR 0010``). Read it through :attr:`fevals_bound`, which
        resolves the ``None``. Consumers that size a budget or
        normalize a per-evaluation cost must use that bound rather than
        ``popsize`` -- notably rl2co's ``AUCEnv``, whose reward
        normalization assumes no optimizer bills more than
        ``max_iterations * max(popsize)``.
    family : str
        Which optimizer family this is, one of
        :data:`~l2co_optimizers._src.core.state_transfer.FAMILIES`. Set by
        the constructing factory, which knows what it is building --
        *not* inferred from evosax registry membership, which
        ``FEEDBACK_evosax.md`` records as a source of friction. Says
        what kind of search state this optimizer holds, and nothing
        more: the population half of a handshake is decided by
        :attr:`ask_fn`'s presence, not by this.
    ask_fn : AskFunction or None
        ``ask_fn(opt_state, key) -> (params, opt_state)`` -- the
        optimizer's own sampling step, bounds-clipped like
        :attr:`step_fn`. Present when
        :attr:`~l2co_optimizers._src.core.state_transfer.TransferSpec.own_ask`
        is set, which is every ``"distribution"`` optimizer and the
        mutation GAs; ``None`` (default) otherwise. A switching caller
        that finds it set must draw the post-switch population from it
        rather than handing over a layout. Calling it *after* writing
        the handshake into the state is what keeps the optimizer's next
        update consistent with the population it is told about:
        CR-FM-NES and seven relatives cache their draws in state and
        consume them on the next ``tell`` (``l2co ADR 0012``), and the
        mutation GAs score the next generation against a baseline the
        writer just repaired (``l2co ADR 0014``).
    transfer_read_fn : TransferReadFunction or None
        ``transfer_read_fn(opt_state) -> (sigma, conf)`` -- projects
        this optimizer's own state onto the canonical scale carried
        across a switch. Always callable when the optimizer was built
        through a factory: an optimizer holding no scale gets a reader
        returning ``CONF_ABSENT`` rather than ``None``, so callers need
        no presence check. ``None`` only for a hand-constructed
        ``UpdateClass``.
    transfer_write_fn : TransferWriteFunction or None
        ``transfer_write_fn(bundle, opt_state) -> opt_state`` -- folds a
        :class:`~l2co_optimizers._src.core.state_transfer.TransferBundle`
        into this optimizer's own state. Same convention: a
        non-receiver such as L-BFGS gets
        :func:`~l2co_optimizers._src.core.state_transfer.write_none`, an
        identity, rather than ``None``. Both halves are bound at
        construction because they need hyperparameters the state does
        not retain -- the learning rate, Rprop's clip bounds, the ravel
        function evosax uses to flatten a solution.
    sequential_realizations : bool
        Class-level marker (not a field): whether a multi-realization
        run must execute one realization at a time rather than with all
        realizations vectorized together. ``False`` for every plain
        optimizer, which is the fast choice. A subclass sets it to
        ``True`` when its ``step_fn`` branches on a value that differs
        per realization -- the meta-optimizers, whose step picks a
        sub-optimizer from a menu. Read by :meth:`batch_run`, which is
        where the consequences are spelled out.
    """

    #: See the class docstring. A ``ClassVar`` annotation keeps this out
    #: of the dataclass fields, so subclasses override it with a plain
    #: assignment and no constructor signature changes.
    sequential_realizations: ClassVar[bool] = False

    init_fn: InitFunction = eqx.field(static=True)
    step_fn: StepFunction = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    hash: int = eqx.field(static=True)
    stop_fn: StopFunction | None = eqx.field(static=True, default=None)
    max_fevals_per_step: int | None = eqx.field(static=True, default=None)
    family: str = eqx.field(static=True, default=FAMILY_GRADIENT)
    ask_fn: AskFunction | None = eqx.field(static=True, default=None)
    transfer_read_fn: TransferReadFunction | None = eqx.field(
        static=True, default=None
    )
    transfer_write_fn: TransferWriteFunction | None = eqx.field(
        static=True, default=None
    )

    @property
    def fevals_bound(self) -> int:
        """Upper bound on ``OptHistory.fevals`` for one iteration.

        Returns
        -------
        int
            :attr:`max_fevals_per_step` when set, else :attr:`popsize`.
        """
        if self.max_fevals_per_step is None:
            return self.popsize
        return self.max_fevals_per_step

    def init_state(
        self,
        params: InputParameters,
        batch_state: BatchState,
        dataset: dict[str, jax.Array],
        key: PRNGKeyArray,
    ) -> PyTree:
        """Initialize the optimizer state on one batch of ``dataset``.

        Parameters
        ----------
        params : InputParameters
            Initial population, shape ``(popsize, ...)``.
        batch_state : BatchState
            Batch state the initialising batch is drawn from. It is not
            advanced: the returned state does not include it.
        dataset : dict[str, jax.Array]
            Dataset for evaluation.
        key : PRNGKeyArray
            PRNG key, used both to draw the batch and by ``init_fn``.

        Returns
        -------
        PyTree
            Initialized optimizer state.
        """
        batch_idxs, _ = batch_state.next(key)
        sample = jax.tree.map(lambda x: x[batch_idxs], dataset)
        return self.init_fn(params, key, **sample)

    def step(
        self,
        carry: Carry,
        xs,
        dataset: dict[str, jax.Array],
    ) -> tuple[Carry, OptHistory]:
        """Advance one generation, or skip it once ``stop_fn`` fires.

        Parameters
        ----------
        carry : Carry
            ``(params, opt_state, key, batch_state, done,
            recent_history)``.
        xs : Any
            The scan's per-step input; unused.
        dataset : dict[str, jax.Array]
            Dataset the next batch is drawn from.

        Returns
        -------
        tuple[Carry, OptHistory]
            The advanced carry and this generation's history entry. A
            skipped generation returns the carry unchanged and an
            all-NaN ``OptHistory.init`` entry.
        """

        def do_step(carry: Carry) -> tuple[Carry, OptHistory]:
            (params, opt_state, key, batch_state, done, recent_history) = carry
            batch_idxs, batch_state = batch_state.next(key)
            sample = jax.tree.map(lambda x: x[batch_idxs], dataset)
            new_carry, history = self.step_fn(
                (params, opt_state, key), sample=sample
            )

            new_recent_history = recent_history.update(history)

            return (
                *new_carry,
                batch_state,
                done,
                new_recent_history,
            ), history

        def skip_step(carry: Carry) -> tuple[Carry, OptHistory]:
            return carry, OptHistory.init(carry[0], self.popsize, self.hash)

        _, opt_state, _, _, done, recent_history = carry

        # ``stop_fn`` is static, so ``None`` (never stop) is resolved here at
        # trace time rather than traced as a call.
        stop = done
        if self.stop_fn is not None:
            stop = stop | self.stop_fn(recent_history, opt_state)

        return jax.lax.cond(stop, skip_step, do_step, carry)

    def batch_step(
        self,
        carry: Carry,
        xs,
        dataset: dict[str, jax.Array],
    ) -> tuple[Carry, OptHistory]:
        """:meth:`step` vmapped over a leading realization axis.

        Every array leaf of ``carry`` carries the realization axis;
        ``xs`` and ``dataset`` are shared.

        Parameters
        ----------
        carry : Carry
            Per-realization carry, shape ``(n_realizations, ...)`` on
            every array leaf.
        xs : Any
            The scan's per-step input; unused.
        dataset : dict[str, jax.Array]
            Dataset shared across realizations.

        Returns
        -------
        tuple[Carry, OptHistory]
            As :meth:`step`, with the realization axis leading.
        """
        return eqx.filter_vmap(
            self.step, in_axes=(eqx.if_array(axis=0), None, None)
        )(carry, xs, dataset)

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
        """Run one trajectory of ``n_iterations`` generations.

        Parameters
        ----------
        opt_state : PyTree
            Current optimizer state.
        params : InputParameters
            Current population, shape ``(popsize, ...)``.
        batch_state : BatchState
            Current batch state.
        dataset : dict[str, jax.Array]
            Dataset for evaluation.
        key : PRNGKeyArray
            PRNG key for random operations.
        n_iterations : int
            Number of iterations to perform.
        verbose : bool
            Whether to wrap the scan body with
            :func:`jax_tqdm.scan_tqdm`.

        Returns
        -------
        RunResult
            ``(params, best_params, best_loss, opt_state, batch_state,
            history_state)``.
        """
        recent_history_init = RecentHistory.init(
            params=params,
            memory_size=_RECENT_HISTORY_SIZE,
        )
        init_done = jnp.array(False)

        n = n_iterations + 1

        # Reduce each generation in place (``_reduce_entry``) instead of
        # stacking the full ``(n, popsize, dim)`` population/gradient
        # trajectory as scan output and reducing afterwards: the trajectory
        # stays out of the scan output, so peak memory no longer scales with
        # ``dim`` or ``popsize``. The running best and the last evaluated
        # population ride in the carry. ``params[0]`` seeds the running best
        # with a correctly shaped/typed member (overwritten on the first,
        # always-executed step); ``best_loss`` is +inf in the loss dtype so
        # the carry type is stable across the scan.
        float_dtype = jax.tree.leaves(params)[0].dtype
        best_params_init = jax.tree.map(lambda p: p[0], params)
        best_loss_init = jnp.array(jnp.inf, dtype=float_dtype)
        eval_params_init = jax.tree.map(
            lambda p: jnp.full_like(p, jnp.nan), params
        )

        def scan_step(carry_ext, xs):
            carry, best_params, best_loss, _eval = carry_ext
            new_carry, entry = self.step(carry, xs, dataset)
            stats, cand_params, cand_loss = _reduce_entry(entry)
            best_params, best_loss = _update_best(
                best_params, best_loss, cand_params, cand_loss
            )
            # ``entry.params`` is the evaluated population
            # (``input_params``, NaN on skipped steps) -- what the legacy
            # path returned as ``final_params`` via ``history.params[-1]``.
            # Thread it forward so the last step's value survives without
            # stacking the trajectory.
            return (new_carry, best_params, best_loss, entry.params), stats

        init_carry = (
            params,
            opt_state,
            key,
            batch_state,
            init_done,
            recent_history_init,
        )
        final_ext, stats = jax.lax.scan(
            f=_maybe_progress_bar(scan_step, n, verbose),
            init=(
                init_carry,
                best_params_init,
                best_loss_init,
                eval_params_init,
            ),
            xs=jnp.arange(n),
            length=n,
        )

        carry, best_params, best_loss, params = final_ext
        _, opt_state, _, batch_state, _, _ = carry

        # Drop the leading sentinel iteration (parity with the legacy
        # ``history[1:]`` slice) before assembling the reduced HistoryState.
        stats = tuple(s[1:] for s in stats)
        history_state = HistoryState.from_reduced(*stats)

        return (
            params,
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
        """Run ``n_realizations`` trajectories, on whichever driver fits.

        Two drivers, same signature and same output contract; the
        optimizer decides which one it needs.

        :meth:`batch_run_fused` is the default and the fastest: all
        realizations advance together in one fused scan.

        :meth:`batch_run_sequential` runs one realization at a time, for
        an optimizer whose step branches on a choice that differs per
        realization -- the meta-optimizers, which pick a sub-optimizer
        from a menu each generation. Vectorizing those realizations
        together would force every menu entry's evaluation on every
        generation and discard all but one, because ``vmap`` cannot take
        a different branch per lane. That waste is per generation and it
        scales with the task's loss, so it bites hardest where it can
        least be afforded: on the 2241-dimensional ``convection`` PINN at
        25 realizations, the two drivers measure 496 ms against 57 ms per
        iteration -- 5.7 h against 0.65 h over a 41000-iteration budget,
        against a 2.5 h SLURM wall limit on the run step.

        :attr:`sequential_realizations` is what marks such an optimizer.
        Read :meth:`batch_run_sequential`'s docstring before trusting a
        recomputed cell to match a stored one: the two agree on every
        discrete choice but not on the resulting numbers, and it also
        records where the sequential/fused crossover sits.

        A subclass with a different reason to run sequentially overrides
        this method instead: ``RandomSearchUpdateClass`` does, to bound
        peak memory.

        Parameters
        ----------
        opt_state : PyTree
            Per-realization optimizer state, shape
            ``(n_realizations, ...)`` on every array leaf.
        params : InputParameters
            Per-realization populations, shape
            ``(n_realizations, popsize, ...)``.
        batch_state : BatchState
            Single (un-batched) batch state; broadcast across
            realizations internally.
        dataset : dict[str, jax.Array]
            Dataset for evaluation (shared across realizations).
        key : PRNGKeyArray
            Per-realization PRNG keys, shape ``(n_realizations,)``.
        n_iterations : int
            Number of iterations to perform.
        verbose : bool
            Progress bar; honoured by the fused driver only.

        Returns
        -------
        RunResult
            ``(params, best_params, best_loss, opt_state, batch_state,
            history_state)`` -- each output carries a leading
            ``n_realizations`` axis.
        """
        driver = (
            self.batch_run_sequential
            if self.sequential_realizations
            else self.batch_run_fused
        )
        return driver(
            opt_state=opt_state,
            params=params,
            batch_state=batch_state,
            dataset=dataset,
            key=key,
            n_iterations=n_iterations,
            verbose=verbose,
        )

    def batch_run_fused(
        self,
        opt_state: PyTree,
        params: InputParameters,
        batch_state: BatchState,
        dataset: dict[str, jax.Array],
        key: PRNGKeyArray,
        n_iterations: int,
        verbose: bool,
    ) -> RunResult:
        """Run ``n_realizations`` trajectories in one fused scan.

        A single :func:`jax.lax.scan` walks the time axis, and each scan
        iteration invokes :meth:`batch_step` -- :meth:`step` vmapped over
        a leading ``n_realizations`` axis. This produces one compiled
        scan body that does all realizations' time steps together,
        rather than ``n_realizations`` independent scans.

        Inputs ``opt_state``, ``params`` and ``key`` are expected to
        carry a leading realization axis (size ``n_realizations``); the
        auxiliary scan carry leaves (``batch_state``, ``init_done``,
        ``recent_history_init``) are broadcast to that same axis here so
        every leaf of the carry is mappable on axis 0.

        Parameters
        ----------
        opt_state : PyTree
            Per-realization optimizer state with shape
            ``(n_realizations, ...)`` on every array leaf.
        params : InputParameters
            Per-realization input parameters with shape
            ``(n_realizations, popsize, ...)``.
        batch_state : BatchState
            Single (un-batched) batch state; broadcast across
            realizations internally.
        dataset : dict[str, jax.Array]
            Dataset for evaluation (shared across realizations).
        key : PRNGKeyArray
            Per-realization PRNG keys, shape ``(n_realizations,)``.
        n_iterations : int
            Number of iterations to perform.
        verbose : bool
            Whether to wrap the scan body with
            :func:`jax_tqdm.scan_tqdm`.

        Returns
        -------
        RunResult
            ``(params, best_params, best_loss, opt_state, batch_state,
            history_state)`` -- each output carries a leading
            ``n_realizations`` axis.
        """
        n_realizations = jax.tree.leaves(params)[0].shape[0]

        recent_history_init = eqx.filter_vmap(
            RecentHistory.init,
            in_axes=(0, None),
        )(params, _RECENT_HISTORY_SIZE)

        # Per-realization done flag.
        init_done = jnp.zeros((n_realizations,), dtype=jnp.bool_)

        batch_state = _broadcast_batch_state(batch_state, n_realizations)

        n = n_iterations + 1

        # Per-realization running-best / last-evaluated-population seeds;
        # see :meth:`run` for the reduce-in-scan rationale. ``params[:, 0]``
        # seeds one correctly-shaped best member per realization;
        # ``best_loss`` is +inf in the loss dtype so the carry type is
        # stable across the scan.
        float_dtype = jax.tree.leaves(params)[0].dtype
        best_params_init = jax.tree.map(lambda p: p[:, 0], params)
        best_loss_init = jnp.full(
            (n_realizations,), jnp.inf, dtype=float_dtype
        )
        eval_params_init = jax.tree.map(
            lambda p: jnp.full_like(p, jnp.nan), params
        )

        reduce_batch = jax.vmap(_reduce_entry)
        update_best_batch = jax.vmap(_update_best)

        def scan_step(carry_ext, xs):
            carry, best_params, best_loss, _eval = carry_ext
            new_carry, entry = self.batch_step(carry, xs, dataset)
            stats, cand_params, cand_loss = reduce_batch(entry)
            best_params, best_loss = update_best_batch(
                best_params, best_loss, cand_params, cand_loss
            )
            return (new_carry, best_params, best_loss, entry.params), stats

        init_carry = (
            params,
            opt_state,
            key,
            batch_state,
            init_done,
            recent_history_init,
        )
        final_ext, stats = jax.lax.scan(
            f=_maybe_progress_bar(scan_step, n, verbose),
            init=(
                init_carry,
                best_params_init,
                best_loss_init,
                eval_params_init,
            ),
            xs=jnp.arange(n),
            length=n,
        )

        carry, best_params, best_loss, params = final_ext
        _, opt_state, _, batch_state, _, _ = carry

        # ``stats`` leaves are (n_iter, n_realizations); drop the leading
        # sentinel iteration and move realization to the front so
        # ``from_reduced`` (vmapped over realizations) sees canonical
        # (n_iter,) fields. This transpose is on scalar-per-step arrays,
        # not the (n, popsize, dim) trajectory the legacy swapaxes copied.
        stats = tuple(jnp.swapaxes(s[1:], 0, 1) for s in stats)
        history_state = jax.vmap(HistoryState.from_reduced)(*stats)

        return (
            params,
            best_params,
            best_loss,
            opt_state,
            batch_state,
            history_state,
        )

    def batch_run_sequential(
        self,
        opt_state: PyTree,
        params: InputParameters,
        batch_state: BatchState,
        dataset: dict[str, jax.Array],
        key: PRNGKeyArray,
        n_iterations: int,
        verbose: bool,
    ) -> RunResult:
        """Run ``n_realizations`` trajectories one at a time.

        Same inputs, same outputs and same per-realization arithmetic as
        :meth:`batch_run_fused` -- it maps :meth:`run` over the
        realization axis with :func:`jax.lax.map` (sequentially) instead
        of vmapping :meth:`step` inside one fused scan. Because it maps
        ``self.run``, a subclass that overrides :meth:`run` gets its own
        runner mapped here too; ``RandomSearchUpdateClass`` relies on
        that.

        Why a sequential map is ever worth having
        -----------------------------------------
        A meta-optimizer's step chooses a sub-optimizer and runs only that
        one, expressed as a :func:`jax.lax.switch` on the chosen index.
        ``vmap`` cannot do per-lane control flow: with a batched index the
        switch has to evaluate **every** branch and select from the
        results. So under :meth:`batch_run_fused` a menu of ``(lbfgs,
        rprop, crfmnes, mr15ga)`` pays, on every single generation, a
        width-1 gradient evaluation *and* a width-28 population evaluation
        *and* a width-27 one *and* L-BFGS's line search -- then throws all
        but one away. The waste is per-generation and it scales with how
        expensive the task's loss is, so it hurts most exactly where it
        can least be afforded. One realization at a time keeps the index
        a scalar, so only the chosen branch runs.

        The trade is the SIMD width across realizations, and it is a real
        trade rather than a free win. The saving grows with what a branch
        costs -- so with the task's loss and with the menu's population
        sizes -- while the lost vectorization is roughly fixed. Below the
        crossover a menu-based optimizer is genuinely better off
        vectorized. Measured on ``bbob`` ``sphere`` at 25 realizations
        with the four-entry ``(lbfgs, rprop, crfmnes, mr15ga)`` menu,
        sequential against fused:

        ==========  =============
        ``d``       speedup
        ==========  =============
        2           0.72x
        10          0.64x
        40          1.03x
        64          1.02x
        256         1.83x
        1024        2.89x
        ==========  =============

        and 8.7x on the 2241-dimensional ``convection`` PINN. So this
        driver is taken unconditionally for a menu-based optimizer: it
        gives back about 1.5x on the small-``d`` cells, which are seconds
        of work, and it is break-even to 8.7x everywhere the runtime
        actually is (``conf/budgets/based_on_dim.yaml`` records that
        ``d >= 64`` is 96.6% of BBOB runtime). A plain optimizer has no
        switch to collapse and stays on :meth:`batch_run_fused`, which is
        strictly faster for it. :meth:`batch_run` picks between them on
        :attr:`sequential_realizations`.

        Equivalence to :meth:`batch_run_fused`
        --------------------------------------
        Each realization sees the same ``opt_state`` / ``params`` /
        ``batch_state`` / ``key`` slice it would as a ``vmap`` lane, and
        :meth:`run` applies the same per-lane initialisation and
        reductions :meth:`batch_run_fused` applies through its vmaps.
        Same inputs, same arithmetic, same output contract -- but not the
        same bits, and over a long run not even the same third
        significant figure.

        What holds exactly is the sequence of choices: ``update_step``
        agrees generation for generation, so both drivers run the same
        sub-optimizer at the same time on the same realization.

        What drifts is everything continuous, and it drifts because an
        optimizer trajectory amplifies. ``vmap``-of-``switch`` compiles a
        different (all-branches-then-select) program than a single-branch
        scan, so XLA does not pin reduction order across the two, and the
        starting disagreement is a few ulp. A line-search sub-optimizer
        then turns that into a discrete difference -- how many trials
        L-BFGS bills depends on a continuous comparison, so ``fevals``
        can differ by one -- and the slightly different step feeds the
        next generation. Measured on the ``convection`` PINN over 120
        generations: median relative difference in ``output_min`` 2e-15,
        but 3e-2 at the worst entry, and the divergence appears only in
        the realizations that got far enough to be in a sensitive region
        (the ones still far from a minimum agree to round-off
        throughout).

        So: a cell recomputed on this driver is the same experiment, not
        the same numbers as one stored from :meth:`batch_run_fused`.
        rl2co's ``rl2co._src.rollout.unroll.map_unroll`` documents the
        same trade for the same reason on the training side.

        Parameters
        ----------
        opt_state : PyTree
            Per-realization optimizer state with shape
            ``(n_realizations, ...)`` on every array leaf.
        params : InputParameters
            Per-realization input parameters with shape
            ``(n_realizations, popsize, ...)``.
        batch_state : BatchState
            Single (un-batched) batch state; broadcast across
            realizations internally so each realization advances its own
            copy.
        dataset : dict[str, jax.Array]
            Dataset for evaluation (shared across realizations).
        key : PRNGKeyArray
            Per-realization PRNG keys, shape ``(n_realizations,)`` --
            already split by the caller (see l2co's
            ``RolloutWrapper.batch_evaluate``).
        n_iterations : int
            Number of iterations to perform.
        verbose : bool
            Ignored. A per-realization :func:`jax_tqdm.scan_tqdm` bar
            would emit ``n_realizations`` interleaved progress bars from
            inside the map, which is worse than none.

        Returns
        -------
        RunResult
            ``(params, best_params, best_loss, opt_state, batch_state,
            history_state)`` -- each output carries a leading
            ``n_realizations`` axis, matching :meth:`batch_run_fused`.
        """
        del verbose
        n_realizations = jax.tree.leaves(params)[0].shape[0]
        batch_state = _broadcast_batch_state(batch_state, n_realizations)

        def _single(slices):
            opt_state, params, batch_state, key = slices
            return self.run(
                opt_state=opt_state,
                params=params,
                batch_state=batch_state,
                dataset=dataset,
                key=key,
                n_iterations=n_iterations,
                verbose=False,
            )

        return jax.lax.map(_single, (opt_state, params, batch_state, key))


#                                                                    Helpers
# =============================================================================


def _broadcast_batch_state(
    batch_state: BatchState, n_realizations: int
) -> BatchState:
    """Give every leaf of ``batch_state`` a leading realization axis."""
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n_realizations, *x.shape)),
        batch_state,
    )


def _maybe_progress_bar(scan_step, n: int, verbose: bool):
    """Wrap ``scan_step`` in a :func:`jax_tqdm.scan_tqdm` bar if verbose."""
    return scan_tqdm(n)(scan_step) if verbose else scan_step


def _reduce_entry(
    entry: OptHistory,
) -> tuple[tuple[Array, ...], PyTree, Array]:
    """Reduce one per-step ``OptHistory`` to the fields a run needs.

    Returns ``(stats, step_best_params, step_best_loss)`` where ``stats``
    is ``(output_min, output_mean, output_std, update_step, fevals,
    iterations)`` for this single generation -- the same NaN-aware
    reductions :meth:`HistoryState.from_history` computes over the
    ``popsize`` axis, applied per step so the full ``(n_iter, popsize,
    dim)`` population/gradient trajectory never has to be stacked as scan
    output. ``step_best_params`` / ``step_best_loss`` are the best
    (lowest-loss) member of this generation, threaded through the scan
    carry to maintain a running best in place of a post-hoc
    ``OptHistory.retrieve_best`` over the stacked trajectory.

    Operates on an unbatched entry (``loss`` shape ``(popsize,)``); the
    batched driver vmaps it over the realization axis.
    """
    loss = entry.loss
    output_min = jnp.nanmin(loss)
    stats = (
        output_min,
        jnp.nanmean(loss),
        jnp.nanstd(loss),
        entry.update_step,
        entry.fevals,
        entry.iterations,
    )
    # NaN-safe argmin: NaN slots (padding / skipped steps) are pushed to
    # +inf so they are never selected, matching ``jnp.nanargmin`` on the
    # flattened trajectory used by ``OptHistory.retrieve_best``.
    safe = jnp.where(jnp.isnan(loss), jnp.inf, loss)
    best_idx = jnp.argmin(safe)
    step_best_params = jax.tree.map(lambda p: p[best_idx], entry.params)
    return stats, step_best_params, output_min


def _update_best(
    best_params: PyTree,
    best_loss: Array,
    cand_params: PyTree,
    cand_loss: Array,
) -> tuple[PyTree, Array]:
    """Keep the lower-loss of the running best and a new candidate.

    Strict ``<`` so the *earliest* generation wins ties, matching the
    row-major first-occurrence of ``jnp.nanargmin`` over the flattened
    trajectory. A ``NaN`` candidate loss compares ``False`` and leaves the
    running best untouched. Unbatched (scalar ``cand_loss``); the batched
    driver vmaps it over the realization axis.
    """
    take = cand_loss < best_loss
    best_loss = jnp.where(take, cand_loss, best_loss)
    best_params = jax.tree.map(
        lambda c, b: jnp.where(take, c, b), cand_params, best_params
    )
    return best_params, best_loss
