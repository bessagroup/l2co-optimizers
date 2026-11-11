"""
Shared state, schedules and operators for differential-evolution variants.

The DE variants of the RL-DAS menu (JDE21, MadDE, NL-SHADE-RSP; Guo et
al. 2024, MetaBox v2 ``src/environment/optimizer/rldas_optimizer.py``)
are built on this module. It holds what they share, so each variant
module holds only its own operators and adaptation:

* :class:`DEPopulation` -- population, costs, live-slot mask,
  per-individual ``F`` / ``Cr``, external archive, success-history
  memories ``M_F`` / ``M_Cr`` with write index ``k``, and the current
  generation's successes ``S_F`` / ``S_Cr`` with their weights. Shapes
  are padded to the starting population size ``n_max`` (and the largest
  archive), with ``NaN`` in inactive slots, so the state stays one JAX
  carry while the population shrinks.
* Population-size schedules as pure functions of
  ``(fevals, max_fevals)``, and :func:`generations_for_budget`.
* Operators taking explicit draws or a key: current-to-pbest/1 with
  archive (donor sampling restricted to live slots and valid archive
  rows), binomial and exponential crossover, the weighted-Lehmer memory
  update, archive insertion with random replacement, truncated-Cauchy
  ``F`` / normal ``Cr`` sampling, sorting, shrinking, j21's pairwise
  halving and archive resizing.
* The step protocol (:class:`AbstractDEVariant`, :class:`DEPrivateState`,
  :class:`DEProgress`) and a registry adapter
  (:func:`de_variant_update`) that turns a variant into an
  ``UpdateClass`` billing only the rows it evaluates.

**Why the population is not the variant's.** Under RL-DAS an agent
switches between DE variants every few thousand evaluations, and all of
them work on one population (MetaBox's single ``Population`` object: the
group, costs, F, Cr, archive, memories, ``k``). So a variant's step is
``ask(key, pop, private, progress)`` / ``tell(pop, private, trials,
trial_cost, aux, progress)``: the :class:`DEPopulation` is passed in and
out, and only what one variant alone understands sits in its private
state (e.g. MadDE's operator probabilities). The RL-DAS wrapper in
metabbo-jax keeps one ``DEPopulation`` and one private state per
variant; switching hands the population to the chosen variant together
with *its* private state, and the others' private states are kept as
they are (frozen) until their next turn. Standalone, the adapter owns
the population and the variant's private state alike.

**Who owns the schedule.** Whoever bills evaluations: the adapter for a
standalone run (planned against ``max_fevals``), metabbo-jax under
RL-DAS. Either way the variant receives ``(fevals, max_fevals)`` as
:class:`DEProgress` and shrinks itself in ``tell``, because how it
shrinks is its own (keep the best, keep the tail, halve pairwise).

**Population-size reduction formulas.** With ``p = fevals / max_fevals``:

* non-linear (NLPSR): ``round(n_max + (n_min - n_max) * p ** (1 - p))``
  -- Stanovov, Akhmedova and Semenkin, "NL-SHADE-RSP Algorithm with
  Adaptive Archive and Selective Pressure for CEC 2021 Numerical
  Optimization", CEC 2021; MetaBox ``Population.cal_NP_next_gen``
  (``n_max = 170``, ``n_min = 30`` for RL-DAS);
* linear (LPSR): ``round(n_max + (n_min - n_max) * p)`` -- Tanabe and
  Fukunaga, "Improving the search performance of SHADE using linear
  population size reduction", CEC 2014 (L-SHADE); used by MadDE (Biswas
  et al., "Improving Differential Evolution through Bayesian
  Hyperparameter Optimization", CEC 2021);
* j21: the big population is halved by pairwise competition when
  ``fevals`` reaches ``MaxFEs/4``, ``2/4`` and ``3/4``; the small one is
  kept -- Brest, Maučec and Bošković, "Self-adaptive Differential
  Evolution Algorithm with Population Size Reduction for Single
  Objective Bound-Constrained Optimization: Algorithm j21", CEC 2021,
  Sec. III, halving after Brest and Maučec, Appl. Intell. 29 (2008).

The reduction runs at the end of a generation, never after the starting
population is evaluated, so the first generation has the starting size
(:func:`bill_live_population`).

Deviations from the references, all at the level of the shared pieces
(variant-level quirks are listed in each variant's module):

1. **Exact donor exclusion** -- ``r1``, ``r2`` and ``pbest`` are drawn
   among valid candidates only (:func:`sample_index`); MetaBox re-draws
   duplicates at most 25 times (``pbest``: once) and may keep one.
2. **Synchronous generations** -- trials are built from the
   generation's parents and archive, as in every vectorized DE.
3. **Unevaluated or NaN cost is ``inf``** -- a row not evaluated this
   step, and a ``NaN`` loss, enter ``tell`` as ``inf`` and so never win
   a selection.
4. **Every evaluation is billed in a step** -- the adapter's ``init_fn``
   evaluates nothing; its first step evaluates and bills the starting
   population (the evosax adapter evaluates it in ``init_fn`` unbilled).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import abc
from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PRNGKeyArray, PyTree

# Local
from l2co_optimizers._src.core.loss import vmapped_loss, vmapped_loss_with_rng
from l2co_optimizers._src.core.opt_history import OptHistory
from l2co_optimizers._src.core.state_transfer import FAMILY_POPULATION
from l2co_optimizers._src.core.typing import (
    InitFunction,
    InputParameters,
    LossFunction,
    StepFunction,
    StopFunction,
)
from l2co_optimizers._src.core.update_class import UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: ``schedule(fevals, max_fevals) -> population size``.
PopulationSchedule = Callable[[ArrayLike, ArrayLike], Int[Array, ""]]

#: ``bill(step, fevals, max_fevals) -> evaluations billed by that step``.
StepBill = Callable[[int, int, int], ArrayLike]


#                                                             Shared population
# =============================================================================


class DEPopulation(eqx.Module):
    """The state every DE variant shares: population, archive, memories.

    Every array has a fixed, padded shape so the state is a valid JAX
    carry while the population shrinks: the population axis is the
    starting size ``n_max``, the archive axis its largest capacity
    ``archive_max``. An inactive slot holds ``NaN`` in ``x``, ``f`` and
    ``cr`` and ``inf`` in ``cost``; an unused archive or success row holds
    ``NaN``. The operators below keep the live slots a prefix
    ``[0, size)`` whenever they reorder or shrink.

    Attributes
    ----------
    x : Float[Array, "n_max d"]
        Individuals.
    cost : Float[Array, " n_max"]
        Their costs; ``inf`` for an inactive slot and for an individual
        not evaluated yet.
    live : Bool[Array, " n_max"]
        Which slots hold an individual.
    f, cr : Float[Array, " n_max"]
        Per-individual scale factor and crossover rate (jDE-style
        self-adaptation keeps them; SHADE-style variants overwrite them
        each generation with fresh draws from the memory).
    archive : Float[Array, "archive_max d"]
        Defeated parents; rows ``[0, archive_count)`` are valid.
    archive_count : Int[Array, ""]
        Number of valid archive rows.
    archive_capacity : Int[Array, ""]
        Current capacity, at most ``archive_max``; variants that tie it
        to the population size shrink it with :func:`resize_archive`.
    memory_f, memory_cr : Float[Array, " memory_size"]
        Success-history memories ``M_F``, ``M_Cr``.
    memory_index : Int[Array, ""]
        Next memory entry to write, ``k``.
    success_f, success_cr : Float[Array, " success_capacity"]
        This generation's successful parameters ``S_F``, ``S_Cr``;
        entries ``[0, success_count)`` are valid.
    success_weight : Float[Array, " success_capacity"]
        Improvement of each success, the weight in the memory update.
    success_count : Int[Array, ""]
        Number of recorded successes.
    """

    x: Float[Array, "n_max d"]
    cost: Float[Array, " n_max"]
    live: Bool[Array, " n_max"]
    f: Float[Array, " n_max"]
    cr: Float[Array, " n_max"]
    archive: Float[Array, "archive_max d"]
    archive_count: Int[Array, ""]
    archive_capacity: Int[Array, ""]
    memory_f: Float[Array, " memory_size"]
    memory_cr: Float[Array, " memory_size"]
    memory_index: Int[Array, ""]
    success_f: Float[Array, " success_capacity"]
    success_cr: Float[Array, " success_capacity"]
    success_weight: Float[Array, " success_capacity"]
    success_count: Int[Array, ""]

    @property
    def size(self) -> Int[Array, ""]:
        """Number of live individuals."""
        return jnp.sum(self.live).astype(jnp.int32)


def init_population(
    key: PRNGKeyArray,
    bounds: tuple[ArrayLike, ArrayLike],
    *,
    n_max: int,
    d: int,
    memory_size: int,
    archive_max: int,
    archive_capacity: int | None = None,
    success_capacity: int | None = None,
    memory_f: float = 0.5,
    memory_cr: float = 0.5,
    f_init: float = 0.5,
    cr_init: float = 0.9,
    x: Float[Array, "n_max d"] | None = None,
) -> DEPopulation:
    """A full, not yet evaluated population.

    Parameters
    ----------
    key : PRNGKeyArray
        Key for the uniform seeding.
    bounds : tuple[ArrayLike, ArrayLike]
        ``(low, high)``, scalars or shape ``(d,)``; the seeding box.
    n_max : int
        Starting population size, the padded population axis.
    d : int
        Dimensionality.
    memory_size : int
        Number of memory entries ``H``.
    archive_max : int
        Largest archive capacity, the padded archive axis.
    archive_capacity : int or None, optional
        Starting capacity; ``None`` uses ``archive_max``.
    success_capacity : int or None, optional
        Room for successes between two memory updates; ``None`` uses
        ``n_max`` (one generation). A variant that updates the memory
        after several sub-generations (j21) needs more.
    memory_f, memory_cr : float, optional
        Initial memory entries. Defaults 0.5 (SHADE); MetaBox's RL-DAS
        population uses 0.2.
    f_init, cr_init : float, optional
        Initial per-individual ``F`` and ``Cr``. Defaults 0.5 and 0.9
        (jDE, MetaBox).
    x : Float[Array, "n_max d"] or None, optional
        Starting individuals; ``None`` draws them uniformly in
        ``bounds``.

    Returns
    -------
    DEPopulation
        Every slot live, every cost ``inf`` (not evaluated).
    """
    low, high = bounds
    if x is None:
        x = jr.uniform(
            key, (n_max, d), minval=jnp.asarray(low), maxval=jnp.asarray(high)
        )
    if archive_capacity is None:
        archive_capacity = archive_max
    if success_capacity is None:
        success_capacity = n_max
    nan_s = jnp.full((success_capacity,), jnp.nan)
    return DEPopulation(
        x=jnp.asarray(x, dtype=float),
        cost=jnp.full((n_max,), jnp.inf),
        live=jnp.ones((n_max,), dtype=bool),
        f=jnp.full((n_max,), f_init),
        cr=jnp.full((n_max,), cr_init),
        archive=jnp.full((archive_max, d), jnp.nan),
        archive_count=jnp.array(0, dtype=jnp.int32),
        archive_capacity=jnp.array(archive_capacity, dtype=jnp.int32),
        memory_f=jnp.full((memory_size,), memory_f),
        memory_cr=jnp.full((memory_size,), memory_cr),
        memory_index=jnp.array(0, dtype=jnp.int32),
        success_f=nan_s,
        success_cr=nan_s,
        success_weight=nan_s,
        success_count=jnp.array(0, dtype=jnp.int32),
    )


#                                                     Population-size schedules
# =============================================================================


def _progress(fevals: ArrayLike, max_fevals: ArrayLike) -> Array:
    """``fevals / max_fevals`` clipped to ``[0, 1]``."""
    return jnp.clip(jnp.asarray(fevals) / max_fevals, 0.0, 1.0)


def nonlinear_population_size(
    fevals: ArrayLike, max_fevals: ArrayLike, *, n_max: int, n_min: int
) -> Int[Array, ""]:
    """Non-linear population-size reduction (NLPSR).

    ``round(n_max + (n_min - n_max) * p ** (1 - p))`` with
    ``p = fevals / max_fevals`` clipped to ``[0, 1]``. Stanovov,
    Akhmedova and Semenkin (2021), NL-SHADE-RSP, CEC 2021; MetaBox v2
    ``Population.cal_NP_next_gen`` (``rldas_optimizer.py``), which RL-DAS
    applies to its shared population (170 -> 30).

    Parameters
    ----------
    fevals : ArrayLike
        Evaluations billed so far.
    max_fevals : ArrayLike
        The run's budget ``MaxFEs``.
    n_max, n_min : int
        Starting and final population size.

    Returns
    -------
    Int[Array, ""]
        Population size for the next generation.
    """
    p = _progress(fevals, max_fevals)
    size = n_max + (n_min - n_max) * jnp.power(p, 1.0 - p)
    return jnp.round(size).astype(jnp.int32)


def linear_population_size(
    fevals: ArrayLike, max_fevals: ArrayLike, *, n_max: int, n_min: int
) -> Int[Array, ""]:
    """Linear population-size reduction (LPSR).

    ``round((n_min - n_max) / max_fevals * fevals + n_max)``, with
    ``fevals`` clipped to ``max_fevals``. Tanabe and Fukunaga (2014),
    L-SHADE, CEC 2014, eq. (10); MadDE (Biswas et al., 2021) uses it.

    Parameters
    ----------
    fevals : ArrayLike
        Evaluations billed so far.
    max_fevals : ArrayLike
        The run's budget ``MaxFEs``.
    n_max, n_min : int
        Starting and final population size.

    Returns
    -------
    Int[Array, ""]
        Population size for the next generation.
    """
    p = _progress(fevals, max_fevals)
    return jnp.round(n_max + (n_min - n_max) * p).astype(jnp.int32)


def j21_population_size(
    fevals: ArrayLike,
    max_fevals: ArrayLike,
    *,
    n_big: int,
    n_small: int,
    n_reductions: int = 3,
) -> Int[Array, ""]:
    """j21's population size: the big population halves at fixed points.

    The big population (``n_big`` at the start) is halved each time
    ``fevals`` reaches ``k / (n_reductions + 1) * max_fevals``,
    ``k = 1 .. n_reductions``; the small population ``n_small`` is never
    reduced. Brest, Maučec and Bošković (2021), "Self-adaptive
    Differential Evolution Algorithm with Population Size Reduction for
    Single Objective Bound-Constrained Optimization: Algorithm j21", CEC
    2021, Sec. III: three reductions, at MaxFEs/4, 2/4 and 3/4 (their
    example: 400 -> 200 -> 100 -> 50). Halving is integer division,
    ``n_big // 2 ** k``.

    Parameters
    ----------
    fevals : ArrayLike
        Evaluations billed so far.
    max_fevals : ArrayLike
        The run's budget ``MaxFEs``.
    n_big, n_small : int
        Starting sizes of the big and the small population.
    n_reductions : int, optional
        How many halvings over the run. Defaults to 3 (the paper).

    Returns
    -------
    Int[Array, ""]
        Total population size (big + small) for the next generation.
    """
    p = _progress(fevals, max_fevals)
    k = jnp.minimum(jnp.floor((n_reductions + 1) * p), n_reductions)
    big = n_big // jnp.power(2, k.astype(jnp.int32))
    return (big + n_small).astype(jnp.int32)


#                                                         Budget -> generations
# =============================================================================


def bill_live_population(schedule: PopulationSchedule) -> StepBill:
    """The bill of a variant whose every step evaluates its whole population.

    Step 0 evaluates the starting population, ``schedule(0,
    max_fevals)``. Every later step is a generation evaluating the live
    individuals, whose number the previous generation's reduction set
    from the evaluations billed so far. Evaluating the starting
    population is not a generation and reduces nothing, so step 1 bills
    the starting size again (L-SHADE, NL-SHADE-RSP, MetaBox: the
    reduction runs at the end of each generation).

    Parameters
    ----------
    schedule : PopulationSchedule
        ``schedule(fevals, max_fevals) -> population size``.

    Returns
    -------
    StepBill
        ``bill(step, fevals, max_fevals)`` for
        :func:`generations_for_budget`.
    """

    def bill(step: int, fevals: int, max_fevals: int) -> ArrayLike:
        if step <= 1:
            return schedule(0, max_fevals)
        return schedule(fevals, max_fevals)

    return bill


def generations_for_budget(bill: StepBill, budget: int) -> int:
    """Number of steps whose cumulative bill stays within ``budget``.

    The run plans its population against ``max_fevals = budget`` and
    every step bills only the individuals it evaluates, so the bill per
    step varies. This counts steps until the next one would exceed the
    budget: the cumulative evaluations come as close to ``budget`` as
    whole steps allow, never above it.

    Parameters
    ----------
    bill : StepBill
        ``bill(step, fevals, max_fevals)``: evaluations step ``step``
        bills when ``fevals`` were billed before it. Build it with
        :func:`bill_live_population` for a variant that evaluates its
        whole live population every step.
    budget : int
        Total evaluation budget ``B``, also the ``max_fevals`` the
        schedule is planned against.

    Returns
    -------
    int
        Number of steps.

    Raises
    ------
    ValueError
        If a step would bill no evaluation (the count would not end).
    """
    steps, fevals = 0, 0
    while True:
        cost = int(bill(steps, fevals, budget))
        if cost <= 0:
            raise ValueError(
                f"step {steps} bills {cost} evaluations; every step must "
                "bill at least one"
            )
        if fevals + cost > budget:
            return steps
        fevals += cost
        steps += 1


#                                                  Parameter adaptation (SHADE)
# =============================================================================


def record_successes(
    pop: DEPopulation,
    f: Float[Array, " n"],
    cr: Float[Array, " n"],
    weight: Float[Array, " n"],
    mask: Bool[Array, " n"],
) -> DEPopulation:
    """Append the masked ``(F, Cr, weight)`` triples to ``S_F`` / ``S_Cr``.

    Entries are appended in slot order after the ones already recorded.
    The weight is the improvement the variant measures (absolute,
    ``f(x) - f(u)``, in SHADE; relative in MetaBox). Entries beyond the
    success capacity are dropped.

    Parameters
    ----------
    pop : DEPopulation
        Population to record into.
    f, cr, weight : Float[Array, " n"]
        Candidate parameters and improvements.
    mask : Bool[Array, " n"]
        Which candidates succeeded.

    Returns
    -------
    DEPopulation
        ``pop`` with the successes appended.
    """
    capacity = pop.success_f.shape[0]
    pos = pop.success_count + jnp.cumsum(mask) - 1
    pos = jnp.where(mask & (pos < capacity), pos, capacity)

    def put(buffer, values):
        return buffer.at[pos].set(values, mode="drop")

    count = jnp.minimum(pop.success_count + jnp.sum(mask), capacity)
    return eqx.tree_at(
        lambda p: (
            p.success_f,
            p.success_cr,
            p.success_weight,
            p.success_count,
        ),
        pop,
        (
            put(pop.success_f, f),
            put(pop.success_cr, cr),
            put(pop.success_weight, weight),
            count.astype(jnp.int32),
        ),
    )


def update_memory(
    pop: DEPopulation,
    *,
    cr_mean: str = "lehmer",
    old_weight: float = 0.0,
    on_empty: float | None = None,
    small_sum_threshold: float = 0.0,
    small_sum_value: float | None = None,
) -> DEPopulation:
    """Write the weighted means of ``S_F`` / ``S_Cr`` at ``k``; clear them.

    ``M_F[k]`` becomes the weighted Lehmer mean
    ``sum(w s^2) / sum(w s)`` of ``S_F`` with ``w`` the normalized
    improvements (Tanabe and Fukunaga 2013, SHADE, eq. 7-8; L-SHADE 2014
    uses the Lehmer mean for ``Cr`` too, as do MadDE, NL-SHADE-RSP and
    MetaBox's ``Population.update_M_F_Cr``), then ``k`` advances
    cyclically. ``S_F``, ``S_Cr`` and the weights are cleared.

    Parameters
    ----------
    pop : DEPopulation
        Population whose successes are folded in.
    cr_mean : {"lehmer", "arithmetic"}, optional
        Mean for ``Cr``. ``"arithmetic"`` is SHADE 2013's ``mean_WA``.
    old_weight : float, optional
        Blend with the old entry: ``new = (1 - old_weight) * mean +
        old_weight * old``. ``0.5`` is jSO's rule. Defaults to 0.
    on_empty : float or None, optional
        No success this generation: ``None`` (default, the papers)
        leaves the memory and ``k`` alone; a number is written at ``k``
        for both memories without advancing ``k`` (MetaBox writes 0.5).
    small_sum_threshold : float, optional
        A Lehmer mean whose denominator ``sum(w s)`` is at or below
        this is degenerate. Defaults to 0 (only an exact zero).
    small_sum_value : float or None, optional
        Value written for a degenerate Lehmer mean; ``None`` (default)
        keeps the old entry. MetaBox's ``mean_wL`` uses ``1e-6`` / 0.5.

    Returns
    -------
    DEPopulation
        Updated memories and index, empty success buffers.
    """
    capacity = pop.success_f.shape[0]
    valid = jnp.arange(capacity) < pop.success_count
    raw_w = jnp.where(valid, pop.success_weight, 0.0)
    total = jnp.sum(raw_w)
    # All-zero improvements (ties) weigh the successes equally.
    w = jnp.where(total > 0, raw_w / jnp.where(total > 0, total, 1.0), valid)
    w = w / jnp.maximum(jnp.sum(w), 1.0)
    k = pop.memory_index

    def lehmer(s, old):
        s = jnp.where(valid, s, 0.0)
        den = jnp.sum(w * s)
        degenerate = den <= small_sum_threshold
        mean = jnp.sum(w * s**2) / jnp.where(degenerate, 1.0, den)
        fallback = old if small_sum_value is None else small_sum_value
        return jnp.where(degenerate, fallback, mean), degenerate

    def blend(mean, old, degenerate):
        new = (1.0 - old_weight) * mean + old_weight * old
        # A kept entry stays exactly as it was.
        if small_sum_value is None:
            return jnp.where(degenerate, old, new)
        return new

    old_f, old_cr = pop.memory_f[k], pop.memory_cr[k]
    mean_f, deg_f = lehmer(pop.success_f, old_f)
    if cr_mean == "lehmer":
        mean_cr, deg_cr = lehmer(pop.success_cr, old_cr)
    elif cr_mean == "arithmetic":
        mean_cr = jnp.sum(w * jnp.where(valid, pop.success_cr, 0.0))
        deg_cr = jnp.array(False)
    else:
        raise ValueError(
            f"cr_mean must be 'lehmer' or 'arithmetic', got {cr_mean!r}"
        )
    new_f = blend(mean_f, old_f, deg_f)
    new_cr = blend(mean_cr, old_cr, deg_cr)

    has = pop.success_count > 0
    if on_empty is None:
        empty_f, empty_cr = old_f, old_cr
    else:
        empty_f = empty_cr = jnp.asarray(on_empty, dtype=pop.memory_f.dtype)
    memory_f = pop.memory_f.at[k].set(jnp.where(has, new_f, empty_f))
    memory_cr = pop.memory_cr.at[k].set(jnp.where(has, new_cr, empty_cr))
    size = pop.memory_f.shape[0]
    index = jnp.where(has, (k + 1) % size, k).astype(jnp.int32)

    nan_s = jnp.full_like(pop.success_f, jnp.nan)
    return eqx.tree_at(
        lambda p: (
            p.memory_f,
            p.memory_cr,
            p.memory_index,
            p.success_f,
            p.success_cr,
            p.success_weight,
            p.success_count,
        ),
        pop,
        (
            memory_f,
            memory_cr,
            index,
            nan_s,
            nan_s,
            nan_s,
            jnp.array(0, dtype=jnp.int32),
        ),
    )


def f_cr_from_draws(
    pop: DEPopulation,
    slot: Int[Array, " n"],
    z: Float[Array, " n"],
    u: Float[Array, " n"],
    *,
    scale_f: float = 0.1,
    scale_cr: float = 0.1,
    negative_f: str = "truncate",
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """``F`` and ``Cr`` from the memory, given the random draws.

    ``Cr = clip(M_Cr[r] + scale_cr * z, 0, 1)`` (a normal draw) and
    ``F = M_F[r] + scale_f * tan(pi * (u - 0.5))`` (a Cauchy draw by
    inverse CDF), capped at 1 (SHADE).

    Parameters
    ----------
    pop : DEPopulation
        Supplies ``M_F`` and ``M_Cr``.
    slot : Int[Array, " n"]
        Memory entry ``r`` per individual.
    z : Float[Array, " n"]
        Standard-normal draws for ``Cr``.
    u : Float[Array, " n"]
        Uniform ``[0, 1)`` draws for ``F``.
    scale_f, scale_cr : float, optional
        Cauchy and normal scales, 0.1 (SHADE).
    negative_f : {"truncate", "reflect"}, optional
        ``"truncate"`` (default) draws from the Cauchy truncated to
        ``(0, inf)``, ``u`` mapped onto its CDF range; this is the
        papers' resample-while-``F <= 0`` loop in distribution.
        ``"reflect"`` maps a negative ``F`` to ``2 * M_F[r] - F``
        (MetaBox's ``Population.choose_F_Cr``).

    Returns
    -------
    tuple[Float[Array, " n"], Float[Array, " n"]]
        ``(F, Cr)``.
    """
    loc_f = pop.memory_f[slot]
    cr = jnp.clip(pop.memory_cr[slot] + scale_cr * z, 0.0, 1.0)
    if negative_f == "truncate":
        cdf_zero = 0.5 + jnp.arctan(-loc_f / scale_f) / jnp.pi
        u = cdf_zero + u * (1.0 - cdf_zero)
        f = loc_f + scale_f * jnp.tan(jnp.pi * (u - 0.5))
    elif negative_f == "reflect":
        f = loc_f + scale_f * jnp.tan(jnp.pi * (u - 0.5))
        f = jnp.where(f < 0, 2.0 * loc_f - f, f)
    else:
        raise ValueError(
            f"negative_f must be 'truncate' or 'reflect', got {negative_f!r}"
        )
    return jnp.minimum(f, 1.0), cr


def sample_f_cr(
    key: PRNGKeyArray, pop: DEPopulation, **kwargs
) -> tuple[Float[Array, " n_max"], Float[Array, " n_max"]]:
    """:func:`f_cr_from_draws` with the draws taken from ``key``.

    One memory entry per slot, uniformly over all ``H`` entries.
    Keyword arguments go to :func:`f_cr_from_draws`.

    Returns
    -------
    tuple[Float[Array, " n_max"], Float[Array, " n_max"]]
        ``(F, Cr)`` for every slot, dead ones included.
    """
    n = pop.x.shape[0]
    k_slot, k_z, k_u = jr.split(key, 3)
    slot = jr.randint(k_slot, (n,), 0, pop.memory_f.shape[0])
    z = jr.normal(k_z, (n,))
    u = jr.uniform(k_u, (n,))
    return f_cr_from_draws(pop, slot, z, u, **kwargs)


#                                                       Mutation and crossover
# =============================================================================


def _cost_order(pop: DEPopulation) -> Int[Array, " n_max"]:
    """Slot indices, live first by ascending cost (stable), dead last."""
    cost = jnp.where(jnp.isnan(pop.cost), jnp.inf, pop.cost)
    return jnp.lexsort((cost, ~pop.live))


def sample_index(
    key: PRNGKeyArray,
    valid: Bool[Array, "n m"],
    *,
    weights: Float[Array, " m"] | None = None,
    fallback: Int[Array, " n"] | None = None,
) -> Int[Array, " n"]:
    """One column per row, among that row's valid columns.

    Exact exclusion, where the references re-draw a bounded number of
    times (MetaBox: up to 25) and may keep a duplicate; in JAX a mask is
    cheaper than a re-draw loop.

    Parameters
    ----------
    key : PRNGKeyArray
        Key for the draw.
    valid : Bool[Array, "n m"]
        Which columns each row may pick.
    weights : Float[Array, " m"] or None, optional
        Unnormalized column weights (e.g. NL-SHADE-RSP's rank-based
        selection); ``None`` picks uniformly. A zero weight is never
        picked.
    fallback : Int[Array, " n"] or None, optional
        Returned for a row with no valid column; ``None`` uses 0.

    Returns
    -------
    Int[Array, " n"]
        Picked column per row.
    """
    if weights is not None:
        valid = valid & (weights > 0)[None, :]
        log_w = jnp.log(jnp.where(weights > 0, weights, 1.0))[None, :]
    else:
        log_w = 0.0
    logits = jnp.where(valid, log_w, -jnp.inf)
    pick = jr.categorical(key, logits, axis=-1)
    if fallback is None:
        fallback = jnp.zeros(valid.shape[0], dtype=pick.dtype)
    return jnp.where(jnp.any(valid, axis=-1), pick, fallback)


def sample_current_to_pbest(
    key: PRNGKeyArray,
    pop: DEPopulation,
    n_pbest: ArrayLike,
    *,
    exclude_pbest: bool = False,
    use_archive: bool = True,
) -> tuple[Int[Array, " n_max"], Int[Array, " n_max"], Int[Array, " n_max"]]:
    """Donor indices for current-to-pbest/1, restricted to valid rows.

    ``pbest`` is uniform among the ``n_pbest`` best live individuals;
    ``r1`` among live individuals other than ``i``; ``r2`` among live
    individuals and valid archive rows other than ``i`` and ``r1``
    (SHADE: ``r1 != r2 != i``, ``r2`` from ``P U A``). The rounding of
    ``n_pbest`` from ``p * NP`` differs between variants, so the caller
    computes it.

    Parameters
    ----------
    key : PRNGKeyArray
        Key for the draws.
    pop : DEPopulation
        Population and archive.
    n_pbest : ArrayLike
        Size of the pbest pool (at least 1; at least 2 with
        ``exclude_pbest``).
    exclude_pbest : bool, optional
        Also keep ``pbest`` away from ``i``, ``r1`` and ``r2`` (MetaBox's
        MadDE and NL-SHADE-RSP). Defaults to False (SHADE).
    use_archive : bool, optional
        Draw ``r2`` from the archive too. Defaults to True.

    Returns
    -------
    tuple[Int[Array, " n_max"], Int[Array, " n_max"], Int[Array, " n_max"]]
        ``(pbest, r1, r2)`` per slot. ``pbest`` and ``r1`` index the
        population; ``r2`` indexes ``concat(x, archive)``, so a value
        ``>= n_max`` is archive row ``r2 - n_max``. A row with no valid
        candidate gets its own index (only possible for a population too
        small for the operator); a dead slot's indices are meaningless.
    """
    n = pop.x.shape[0]
    n_archive = pop.archive.shape[0]
    idx = jnp.arange(n)
    rank = jnp.argsort(_cost_order(pop))
    top = pop.live & (rank < n_pbest)
    not_self = idx[None, :] != idx[:, None]
    k_pbest, k_r1, k_r2 = jr.split(key, 3)

    valid_pbest = jnp.broadcast_to(top[None, :], (n, n))
    if exclude_pbest:
        valid_pbest = valid_pbest & not_self
    pbest = sample_index(k_pbest, valid_pbest, fallback=idx)

    valid_r1 = pop.live[None, :] & not_self
    if exclude_pbest:
        valid_r1 = valid_r1 & (idx[None, :] != pbest[:, None])
    r1 = sample_index(k_r1, valid_r1, fallback=idx)

    archive_valid = jnp.arange(n_archive) < pop.archive_count
    if not use_archive:
        archive_valid = jnp.zeros_like(archive_valid)
    union_valid = jnp.concatenate([pop.live, archive_valid])
    col = jnp.arange(n + n_archive)[None, :]
    valid_r2 = (
        union_valid[None, :] & (col != idx[:, None]) & (col != r1[:, None])
    )
    if exclude_pbest:
        valid_r2 = valid_r2 & (col != pbest[:, None])
    r2 = sample_index(k_r2, valid_r2, fallback=idx)
    return pbest, r1, r2


def current_to_pbest_1(
    pop: DEPopulation,
    f: Float[Array, " n_max"],
    pbest: Int[Array, " n_max"],
    r1: Int[Array, " n_max"],
    r2: Int[Array, " n_max"],
) -> Float[Array, "n_max d"]:
    """DE/current-to-pbest/1 mutants (JADE, SHADE).

    ``v_i = x_i + F_i (x_pbest - x_i) + F_i (x_r1 - z_r2)`` with
    ``z = concat(x, archive)``. Dead slots give ``NaN`` rows.

    Parameters
    ----------
    pop : DEPopulation
        Population and archive.
    f : Float[Array, " n_max"]
        Scale factor per slot.
    pbest, r1, r2 : Int[Array, " n_max"]
        Donor indices, as :func:`sample_current_to_pbest` returns them.

    Returns
    -------
    Float[Array, "n_max d"]
        Mutant vectors.
    """
    union = jnp.concatenate([pop.x, pop.archive])
    f = f[:, None]
    v = pop.x + f * (pop.x[pbest] - pop.x) + f * (pop.x[r1] - union[r2])
    return jnp.where(pop.live[:, None], v, jnp.nan)


def binomial_crossover(
    x: Float[Array, "n d"],
    v: Float[Array, "n d"],
    cr: Float[Array, " n"],
    u: Float[Array, "n d"],
    jrand: Int[Array, " n"],
) -> Float[Array, "n d"]:
    """Binomial crossover with fixed draws.

    Component ``j`` comes from the mutant when ``u[i, j] < cr[i]`` or
    ``j == jrand[i]``, else from the parent (Storn and Price 1997; the
    references compare with ``<``).

    Parameters
    ----------
    x, v : Float[Array, "n d"]
        Parents and mutants.
    cr : Float[Array, " n"]
        Crossover rate per row.
    u : Float[Array, "n d"]
        Uniform draws.
    jrand : Int[Array, " n"]
        Component always taken from the mutant.

    Returns
    -------
    Float[Array, "n d"]
        Trial vectors.
    """
    j = jnp.arange(x.shape[1])[None, :]
    take = (u < cr[:, None]) | (j == jrand[:, None])
    return jnp.where(take, v, x)


def exponential_crossover(
    x: Float[Array, "n d"],
    v: Float[Array, "n d"],
    cr: Float[Array, " n"],
    start: Int[Array, " n"],
    u: Float[Array, "n d"],
) -> Float[Array, "n d"]:
    """Exponential crossover with fixed draws.

    A cyclic run of components starting at ``start[i]`` comes from the
    mutant: the start always, then each next component while its draw
    stays below ``cr[i]`` (Storn and Price 1997; NL-SHADE-RSP). Draw
    ``u[i, k]`` decides the ``k``-th component after the start, so
    ``u[i, 0]`` is unused.

    Parameters
    ----------
    x, v : Float[Array, "n d"]
        Parents and mutants.
    cr : Float[Array, " n"]
        Crossover rate per row.
    start : Int[Array, " n"]
        First component taken from the mutant.
    u : Float[Array, "n d"]
        Uniform draws, indexed by offset from ``start``.

    Returns
    -------
    Float[Array, "n d"]
        Trial vectors.
    """
    d = x.shape[1]
    goes_on = u[:, 1:] < cr[:, None]
    by_offset = jnp.concatenate(
        [
            jnp.ones((x.shape[0], 1), dtype=bool),
            jnp.cumprod(goes_on, axis=1) > 0,
        ],
        axis=1,
    )
    offset = (jnp.arange(d)[None, :] - start[:, None]) % d
    take = jnp.take_along_axis(by_offset, offset, axis=1)
    return jnp.where(take, v, x)


#                                               Archive, ordering and shrinking
# =============================================================================


def insert_archive(
    pop: DEPopulation,
    rows: Float[Array, "n d"],
    mask: Bool[Array, " n"],
    u: Float[Array, " n"],
) -> DEPopulation:
    """Insert the masked rows into the archive, one at a time, in order.

    A row is appended while the archive holds fewer than
    ``archive_capacity`` rows; once full, it replaces row
    ``floor(u * count)``, a uniformly random one (JADE; MetaBox's
    ``Population.update_archive``).

    Parameters
    ----------
    pop : DEPopulation
        Population whose archive grows.
    rows : Float[Array, "n d"]
        Candidates, usually the defeated parents.
    mask : Bool[Array, " n"]
        Which candidates to insert.
    u : Float[Array, " n"]
        Uniform ``[0, 1)`` draws, used only when the archive is full.

    Returns
    -------
    DEPopulation
        ``pop`` with the archive updated.
    """

    def insert(carry, item):
        archive, count = carry
        row, take, draw = item
        full = count >= pop.archive_capacity
        target = jnp.where(
            full, jnp.floor(draw * count).astype(jnp.int32), count
        )
        target = jnp.where(
            take & (pop.archive_capacity > 0), target, archive.shape[0]
        )
        archive = archive.at[target].set(row, mode="drop")
        count = jnp.where(take & ~full, count + 1, count)
        return (archive, count), None

    (archive, count), _ = jax.lax.scan(
        insert, (pop.archive, pop.archive_count), (rows, mask, u)
    )
    return eqx.tree_at(
        lambda p: (p.archive, p.archive_count),
        pop,
        (archive, count.astype(jnp.int32)),
    )


def resize_archive(
    pop: DEPopulation,
    capacity: ArrayLike,
    *,
    key: PRNGKeyArray | None = None,
) -> DEPopulation:
    """Set the archive capacity, dropping rows that no longer fit.

    Parameters
    ----------
    pop : DEPopulation
        Population whose archive is resized.
    capacity : ArrayLike
        New capacity, at most ``archive_max``.
    key : PRNGKeyArray or None, optional
        ``None`` (default) keeps the leading rows, as MetaBox's
        ``Population.NLPSR`` does (``archive[:A]``); a key keeps a
        uniformly random subset (L-SHADE).

    Returns
    -------
    DEPopulation
        ``pop`` with the new capacity and at most that many valid rows.
    """
    n_archive = pop.archive.shape[0]
    capacity = jnp.minimum(jnp.asarray(capacity, dtype=jnp.int32), n_archive)
    valid = jnp.arange(n_archive) < pop.archive_count
    archive = pop.archive
    if key is not None:
        score = jnp.where(valid, jr.uniform(key, (n_archive,)), jnp.inf)
        archive = archive[jnp.argsort(score)]
    count = jnp.minimum(pop.archive_count, capacity)
    archive = jnp.where(
        (jnp.arange(n_archive) < count)[:, None], archive, jnp.nan
    )
    return eqx.tree_at(
        lambda p: (p.archive, p.archive_count, p.archive_capacity),
        pop,
        (archive, count.astype(jnp.int32), capacity),
    )


def _gather(
    pop: DEPopulation, src: Int[Array, " n_max"], live: Bool[Array, " n_max"]
) -> DEPopulation:
    """Rebuild from slots ``src``; ``live`` says which slots survive."""
    keep = live[:, None]
    return eqx.tree_at(
        lambda p: (p.x, p.cost, p.live, p.f, p.cr),
        pop,
        (
            jnp.where(keep, pop.x[src], jnp.nan),
            jnp.where(live, pop.cost[src], jnp.inf),
            live,
            jnp.where(live, pop.f[src], jnp.nan),
            jnp.where(live, pop.cr[src], jnp.nan),
        ),
    )


def sort_by_cost(pop: DEPopulation) -> DEPopulation:
    """Reorder slots: live individuals by ascending cost, then dead slots.

    Stable, so equal costs keep their order. MetaBox's DEs sort before
    taking ``pbest`` from the leading rows and before shrinking.

    Parameters
    ----------
    pop : DEPopulation
        Population to reorder.

    Returns
    -------
    DEPopulation
        The same individuals, live ones a sorted prefix.
    """
    order = _cost_order(pop)
    return _gather(pop, order, pop.live[order])


def shrink(
    pop: DEPopulation, n_new: ArrayLike, *, mode: str = "best"
) -> DEPopulation:
    """Keep ``n_new`` live individuals and NaN-pad the freed slots.

    Never grows the population: ``n_new >= size`` keeps everyone. The
    archive and memories are untouched (see :func:`resize_archive`).

    Parameters
    ----------
    pop : DEPopulation
        Population to shrink; its live slots must be a prefix.
    n_new : ArrayLike
        Target size, typically a schedule's value.
    mode : {"best", "tail"}, optional
        ``"best"`` (default) sorts by cost and keeps the best (L-SHADE,
        NL-SHADE-RSP, MadDE; MetaBox's ``Population.NLPSR`` = sort then
        slice); the result is always sorted. ``"tail"`` keeps the last
        ``n_new`` live slots in slot order (MetaBox's JDE21:
        ``group[-NP:]``).

    Returns
    -------
    DEPopulation
        The survivors in slots ``[0, min(n_new, size))``.
    """
    n = pop.x.shape[0]
    size = pop.size
    n_keep = jnp.minimum(jnp.asarray(n_new, dtype=jnp.int32), size)
    slots = jnp.arange(n)
    if mode == "best":
        src = _cost_order(pop)
    elif mode == "tail":
        src = jnp.clip(slots + (size - n_keep), 0, n - 1)
    else:
        raise ValueError(f"mode must be 'best' or 'tail', got {mode!r}")
    return _gather(pop, src, slots < n_keep)


def halve_pairwise(pop: DEPopulation, n_region: ArrayLike) -> DEPopulation:
    """Halve the leading ``n_region`` slots by pairwise competition.

    Slot ``j`` competes with slot ``j + n_region // 2`` and the better
    one (``j`` on a tie) survives in slot ``j`` (Brest and Maučec 2008,
    "Population size reduction for the differential evolution
    algorithm", Appl. Intell. 29; j21 applies it to its big
    population). The live slots after the region move up behind the
    survivors. ``n_region`` should be even; for an odd one the last
    region slot has no partner and is dropped.

    Parameters
    ----------
    pop : DEPopulation
        Population whose live slots are a prefix.
    n_region : ArrayLike
        Size of the leading region to halve (j21: the big population).

    Returns
    -------
    DEPopulation
        Population of ``size - (n_region - n_region // 2)``.
    """
    n = pop.x.shape[0]
    n_region = jnp.asarray(n_region, dtype=jnp.int32)
    half = n_region // 2
    shift = n_region - half
    slots = jnp.arange(n)
    rival = jnp.clip(slots + half, 0, n - 1)
    winner = jnp.where(pop.cost[rival] < pop.cost, rival, slots)
    rest = jnp.clip(slots + shift, 0, n - 1)
    src = jnp.where(slots < half, winner, rest)
    return _gather(pop, src, slots < pop.size - shift)


#                                                          Step protocol
# =============================================================================


class DEProgress(eqx.Module):
    """Where the run stands: evaluations billed and the budget.

    Whoever bills the evaluations owns the population-size schedule: the
    registry adapter (:func:`de_variant_update`) for a standalone run,
    the RL-DAS wrapper in metabbo-jax when several variants take turns
    on one population.

    Attributes
    ----------
    fevals : Array
        Evaluations billed so far. ``ask`` sees the count before its
        trials are billed, ``tell`` the count after.
    max_fevals : Array
        The run's budget ``MaxFEs``.
    """

    fevals: Array
    max_fevals: Array

    @property
    def ratio(self) -> Array:
        """``fevals / max_fevals`` clipped to ``[0, 1]``."""
        return _progress(self.fevals, self.max_fevals)


class DEPrivateState(eqx.Module):
    """Base of a variant's own adaptive state; empty is allowed.

    What a variant carries here is what the reference carries from one
    generation to the next and is not part of :class:`DEPopulation`
    (e.g. MadDE's operator probabilities, j21's age counter). Freezing a
    variant is keeping this pytree; nothing else is needed.
    """


class AbstractDEVariant(eqx.Module):
    """The step protocol every DE variant follows.

    A generation is ``ask`` (build trials), evaluation by the caller,
    then ``tell`` (selection, archive, adaptation, shrinking for the
    next generation). The population lives in :class:`DEPopulation`,
    which the caller owns and may hand to another variant between
    generations; everything only this variant understands lives in its
    :class:`DEPrivateState`.

    Contract:

    * ``ask`` returns trials of shape ``(n_max, d)`` with ``NaN`` rows at
      the slots it does not want evaluated (inactive slots, and for
      j21 the sub-population not stepping this time). The caller bills
      exactly the rows that are not all ``NaN``.
    * ``tell`` receives ``inf`` cost for every row not evaluated (and
      for a ``NaN`` loss), and leaves the live slots a prefix.
    * After ``tell`` at ``progress.fevals = F``, the next ``ask`` bills
      ``bill()(step, F, max_fevals)`` rows; for a variant evaluating its
      whole population that is ``population_size(F, max_fevals)``.
    """

    @abc.abstractmethod
    def init(
        self, key: PRNGKeyArray, x: Float[Array, "n_max d"]
    ) -> tuple[DEPopulation, DEPrivateState]:
        """Population (from the starting points ``x``) and private state."""

    @abc.abstractmethod
    def ask(
        self,
        key: PRNGKeyArray,
        pop: DEPopulation,
        private: DEPrivateState,
        progress: DEProgress,
    ) -> tuple[Float[Array, "n_max d"], PyTree]:
        """Trials (``NaN`` at rows not to evaluate) and auxiliary data.

        The auxiliary pytree (e.g. which strategy built each trial) goes
        to the matching ``tell`` unchanged; its structure must not
        depend on the inputs' values.
        """

    @abc.abstractmethod
    def tell(
        self,
        pop: DEPopulation,
        private: DEPrivateState,
        trials: Float[Array, "n_max d"],
        trial_cost: Float[Array, " n_max"],
        aux: PyTree,
        progress: DEProgress,
    ) -> tuple[DEPopulation, DEPrivateState]:
        """Fold the evaluated trials in and shrink for the next generation."""

    @abc.abstractmethod
    def population_size(
        self, fevals: ArrayLike, max_fevals: ArrayLike
    ) -> Int[Array, ""]:
        """The variant's schedule; ``population_size(0, B)`` is ``n_max``."""

    def bill(self) -> StepBill:
        """Evaluations per step; override when a step evaluates a subset."""
        return bill_live_population(self.population_size)

    def generations_for_budget(self, budget: int) -> int:
        """Steps that fit in ``budget``; see :func:`generations_for_budget`."""
        return generations_for_budget(self.bill(), budget)


#                                                             Registry adapter
# =============================================================================


class DEAdapterState(eqx.Module):
    """What :func:`de_variant_update` threads between steps.

    Attributes
    ----------
    pop : DEPopulation
        The shared population.
    private : DEPrivateState
        The variant's own state.
    aux : PyTree
        ``ask``'s auxiliary output for the trials now being evaluated.
    fevals : Int[Array, ""]
        Evaluations billed so far.
    evaluated : Bool[Array, ""]
        False until the starting population has been evaluated.
    """

    pop: DEPopulation
    private: DEPrivateState
    aux: PyTree
    fevals: Int[Array, ""]
    evaluated: Bool[Array, ""]


def de_variant_fn(
    static: PyTree,
    variant: AbstractDEVariant,
    loss_fn: LossFunction,
    pass_rng: bool,
    max_fevals: float,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None,
) -> tuple[InitFunction, StepFunction]:
    """``(init_fn, step_fn)`` driving ``variant`` with live-only billing.

    The carried parameters are the rows to evaluate next, shape
    ``(n_max, ...)``: the starting population first, then each ``ask``'s
    trials with ``NaN`` rows where nothing is evaluated. A step
    evaluates them, bills the rows that are not all ``NaN``, folds the
    costs in (the first step just records the starting population's
    costs; nothing is evaluated outside a step, so every evaluation is
    billed), and asks for the next trials. The history row carries the
    billed losses and ``NaN`` elsewhere, which ``HistoryState`` reduces
    NaN-aware.

    Returns
    -------
    tuple[InitFunction, StepFunction]
        Initialization and step functions.
    """
    low, high = (None, None) if bounded is None else bounded
    max_fevals_arr = jnp.asarray(max_fevals, dtype=float)

    def ravel_population(params):
        member = jax.tree.map(lambda a: a[0], params)
        _, unravel = ravel_pytree(member)
        flat = jax.vmap(lambda p: ravel_pytree(p)[0])(params)
        return flat, jax.vmap(unravel)

    def init_fn(params: InputParameters, key: PRNGKeyArray, **sample):
        del sample
        x, _ = ravel_population(params)
        pop, private = variant.init(key, x)
        progress = DEProgress(jnp.array(0), max_fevals_arr)
        # Only the structure of ``aux`` matters before the first ask.
        aux_shape = jax.eval_shape(
            lambda: variant.ask(key, pop, private, progress)[1]
        )
        aux = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), aux_shape)
        return DEAdapterState(
            pop=pop,
            private=private,
            aux=aux,
            fevals=jnp.array(0, dtype=jnp.int32),
            evaluated=jnp.array(False),
        )

    def step_fn(carry, sample: dict):
        params, state, key = carry
        k_loss, k_ask, k_next = jr.split(key, 3)
        n_rows = jax.tree.leaves(params)[0].shape[0]
        if pass_rng:
            loss = vmapped_loss_with_rng(
                eqx.combine(params, static),
                loss_fn,
                sample,
                jr.split(k_loss, n_rows),
            )
        else:
            loss = vmapped_loss(eqx.combine(params, static), loss_fn, sample)

        x, unravel = ravel_population(params)
        billed = jnp.any(~jnp.isnan(x), axis=1)
        n_billed = jnp.sum(billed).astype(jnp.int32)
        cost = jnp.where(billed & ~jnp.isnan(loss), loss, jnp.inf)
        fevals = state.fevals + n_billed
        progress = DEProgress(fevals, max_fevals_arr)

        def first(_):
            pop = eqx.tree_at(
                lambda p: p.cost,
                state.pop,
                jnp.where(state.pop.live, cost, jnp.inf),
            )
            return pop, state.private

        def later(_):
            return variant.tell(
                state.pop, state.private, x, cost, state.aux, progress
            )

        pop, private = jax.lax.cond(state.evaluated, later, first, None)
        trials, aux = variant.ask(k_ask, pop, private, progress)
        if low is not None or high is not None:
            trials = jnp.clip(trials, low, high)

        history = OptHistory(
            loss=jnp.where(billed, loss, jnp.nan),
            params=params,
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=n_billed.astype(int),
            grads=jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), params),
            iterations=jnp.array(1, dtype=int),
        )
        new_state = DEAdapterState(
            pop=pop,
            private=private,
            aux=aux,
            fevals=fevals,
            evaluated=jnp.array(True),
        )
        return (unravel(trials), new_state, k_next), history

    return init_fn, step_fn


def de_variant_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    variant: AbstractDEVariant,
    max_fevals: float,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
) -> UpdateClass:
    """Turn a DE variant into an ``UpdateClass`` (standalone mode).

    The adapter owns the schedule: it counts billed evaluations and
    hands ``(fevals, max_fevals)`` to the variant, which shrinks against
    them. The population size is the variant's starting size,
    ``population_size(0, max_fevals)``; each step bills only the rows it
    evaluates (see :func:`de_variant_fn`), so run the variant for
    ``variant.generations_for_budget(max_fevals)`` steps to spend the
    budget without exceeding it.

    Parameters
    ----------
    model : PyTree
        Model whose inexact-array leaves are optimized.
    loss_fn : LossFunction
        Scalar loss, ``loss_fn(model, **sample)`` (with ``key=`` when
        ``pass_rng``).
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword.
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    variant : AbstractDEVariant
        The DE variant.
    max_fevals : float
        Run budget ``MaxFEs`` the schedule is planned against.
    bounded : tuple[float | None, float | None] or None, optional
        Box the trials are clipped to after ``ask`` (a variant repairs
        bounds itself; this is the same safety net as the evosax
        adapter's). Defaults to no bounds.
    stop_fn : StopFunction or None, optional
        Stopping function.

    Returns
    -------
    UpdateClass
        Population-family update class. It declares no state-transfer
        or switch ports: under RL-DAS the population is shared through
        :class:`DEPopulation` instead (metabbo-jax).
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)
    del params
    popsize = int(variant.population_size(0, max_fevals))
    init_fn, step_fn = de_variant_fn(
        static=static,
        variant=variant,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        max_fevals=max_fevals,
        opt_hash=opt_hash,
        bounded=bounded,
    )
    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
        family=FAMILY_POPULATION,
    )
