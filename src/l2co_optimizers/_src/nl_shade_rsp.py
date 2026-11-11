"""
NL-SHADE-RSP -- SHADE with non-linear population size reduction, an
adaptive archive and rank-based selective pressure.

Two registry entries on the shared DE state of
:mod:`~l2co_optimizers._src.de_common`:

* ``nl_shade_rsp`` (:class:`NLSHADERSP`) -- the published algorithm:
  Stanovov, Akhmedova and Semenkin, "NL-SHADE-RSP Algorithm with
  Adaptive Archive and Selective Pressure for CEC 2021 Numerical
  Optimization", IEEE CEC 2021, pp. 809-816,
  doi:10.1109/CEC45853.2021.9504959.
* ``nl_shade_rsp_metabox`` (:class:`NLSHADERSPMetaBox`) -- MetaBox v2's
  ``NL_SHADE_RSP`` and its shared ``Population``
  (``src/environment/optimizer/rldas_optimizer.py``, the RL-DAS DE menu;
  vendored in ``tests/reference/metabox_rldas``), quirks included.

**The algorithm.** A generation of L-SHADE's current-to-pbest/1 with an
external archive, plus: the population shrinks non-linearly,
``NP = round(NP_max + (NP_min - NP_max) r^(1 - r))`` with ``r =
NFE / NFE_max`` (eq. 13, after AGSK), and the archive with it
(``N_A = 2.1 NP``); ``r2`` is taken from the archive with an adapted
probability ``p_A`` (eq. 14), else from the population with rank-based
selective pressure ``R_i = exp(-i / NP)`` (eq. 11); each individual uses
binomial crossover with a scheduled ``Cr_b`` (eq. 15) or exponential
crossover with its memory-sampled ``Cr_i`` with probability 0.5, and the
``Cr_i`` are sorted so the better individuals get the smaller ones;
``pb`` falls linearly from 0.4 to 0.2.

**Published values** (paper Alg. 1 and Sec. III unless noted):
``NP_max = 30 D``; memory ``H = 20 D`` entries initialised to
``M_F = M_Cr = 0.2``; ``N_A = 2.1 NP`` (rounded); ``p_A`` starts at 0.5
and is clipped to ``[0.1, 0.9]``; ``pb`` from 0.4 to 0.2; crossover
choice 0.5; ``F ~ Cauchy(M_F, 0.1)`` re-drawn while negative and capped
at 1, ``Cr ~ N(M_Cr, 0.1)`` clipped to ``[0, 1]`` (eq. 6); weighted
Lehmer means for ``F`` and ``Cr`` with weights ``|f(u) - f(x)|``
(eq. 7); out-of-box components redrawn uniformly in the box (eq. 4).

**Values the paper does not state, and their source:**

* ``NP_min = 4`` -- not given for NL-SHADE-RSP; the value the paper gives
  for L-SHADE (Sec. II.B, eq. 10), which it builds on.
* Memory blend ``c = 0.5``, ``M_k <- c M_k + (1 - c) mean_wL`` -- the
  paper's own description of the success-history update it uses
  (Sec. II.B, eq. 8-9); Alg. 1 l.50 only says "update".
* pbest pool ``int(max(2, NP * pb))`` -- MetaBox's rounding; the paper
  writes ``randInt(1, NP * p)``.
* ``p_A`` when eq. (14) is undefined (no offspring used the archive, or
  none improved) -- reset to 0.5, as MetaBox's ``update_Pa`` does.
* ``Cr_b`` uses the evaluation count at the start of the generation
  (see deviation 1).

**Deviations from the paper:**

1. **Synchronous generations** -- all trials are built from the
   generation's parents and archive; ``Cr_b`` and ``pb`` use the
   evaluation count at the start of the generation, not per individual.
2. **Exact donor exclusion** -- ``pbest``, ``r1``, ``r2`` are drawn among
   valid candidates only (:func:`~l2co_optimizers._src.de_common.
   sample_index`), not by a repeat-until loop.
3. **Eq. (14)'s denominator** -- printed ``1 - n_A``, read as ``NP -
   n_A``, the offspring built without the archive.
4. **Alg. 1 l.22-26 against the text** -- the pseudo-code takes ``r2``
   from the population when ``rand < p_A`` and uniformly; the text
   (Sec. III) says ``p_A`` is the probability of using the archive and
   that ``r2`` from the population is rank-selected. The text is
   followed.
5. **Replacement on ties** -- a trial replaces its parent on ``f(u) <=
   f(x)`` (eq. 5); only ``f(u) < f(x)`` is a success that archives the
   parent and feeds the memory (Alg. 1 l.38-40).
6. **Truncated Cauchy** -- ``F`` is drawn from the Cauchy truncated to
   ``(0, inf)`` by inverse CDF, the re-draw loop's distribution
   (as ``shade`` does).
7. **Memory index** -- ``k`` advances every generation, success or not
   (Alg. 1 l.51); without success the memory is left as it is.
8. **Archive overflow** -- a full archive replaces a uniformly random
   row (L-SHADE, Sec. II.B); Alg. 1 l.39 writes ``A_randInt[1, N_A]``.
9. **Non-finite parent cost** -- a parent with a ``NaN`` loss (``inf``
   cost) gives its successful trial weight 0 in the memory and ``p_A``
   updates, so the weights stay finite.
10. **Box** -- without finite ``bounded`` there is nothing to repair.
    In a half-open box (one side ``None``) eq. (4)'s uniform redraw is
    undefined, so a component past the finite side is clipped onto it.

**Differences from ``shade``** (``_src/shade.py``; ``shade`` itself is
unchanged except for the #24 fix): the population shrinks (NLPSR) and
its padded slots are NaN; ``H = 20 D`` and ``N_A = 2.1 NP`` instead of
``NP``; memories start at 0.2, not 0.5; Lehmer mean for ``Cr`` too and a
0.5 blend with the old entry; ``pb`` falls from 0.4 to 0.2 instead of
``p ~ U[2 / NP, 0.2]`` per individual; ``r2`` from the archive with
probability ``p_A``, else rank-selected, instead of uniform over ``P u
A``; ``pbest`` is kept away from ``i``, ``r1`` and ``r2``; the
binomial/exponential crossover choice and the sorted ``Cr``; uniform
re-draw out of the box instead of midpoint repair; no turning-based
mutation. Shared with ``shade``: current-to-pbest/1 with archive, the
truncated Cauchy and clipped normal draws, greedy selection with a
strict success test.

**MetaBox quirks reproduced by ``nl_shade_rsp_metabox``:**

1. ``Population`` settings: ``NP`` 170 -> 30 by NLPSR whatever ``D``,
   archive ``int(2.1 NP)`` (357), memory ``20 D`` at 0.2, per-individual
   ``F`` / ``Cr`` start 0.5 / 0.9, ``pb`` 0.4 -> 0.2.
2. One crossover draw for the whole generation, and its branches are
   swapped: "exponential" runs *binomial* crossover with ``Crb``.
3. ``Crb = 2 ((FEs + i) / MaxFEs - 0.5)`` in the first half of the
   budget and 0 after (the paper's condition turned around), so
   ``Crb <= 0`` and the binomial branch takes only ``jrand``; ``Crb`` is
   also what that generation records as ``S_Cr`` and ``Cr``
   (:func:`metabox_binomial_crossover_rate`).
4. "Exponential" crossover is not a run: every ``j >= start`` (no wrap)
   whose draw is *above* ``Cr`` crosses (:func:`metabox_exponential_
   crossover`).
5. ``F`` negative values are reflected, ``2 M_F - F``
   (``choose_F_Cr``), then capped at 1.
6. The archive receives ``group[i]`` for ``i in range(len(optim))``: the
   leading rows of the sorted population, not the defeated parents.
7. Selection and success are both strict (``cost < parent``).
8. Memory weights are relative improvements ``(f(x) - f(u)) / (f(x) +
   1e-9)``; ``mean_wL`` returns 0.5 when ``sum(w s) <= 1e-6``; a
   generation without success writes 0.5 at ``k`` and does not advance
   it; no blending.
9. Archive use needs at least 25 rows; out-of-range components are
   redrawn uniformly in ``[-100, 100]``.
10. ``pa`` adapts after every generation with ``fa`` / ``fp`` swapped
    and ``na`` counting only successful archive users
    (:func:`metabox_archive_probability`), and restarts at 0.5 with each
    RL-DAS turn (:class:`NLSHADERSPMetaBoxState`).
10. At reduction the archive is cut to its leading ``A = int(max(2.1 N,
    N_min))`` rows, and ``NA`` changes only when the archive holds more
    than ``A``.

Not reproduced: the ``1e-8`` error floor and early stop (they need the
known optimum; MetaBox's costs are errors to it -- the relative weights
of quirk 8 assume non-negative costs); MetaBox's re-draw loops (25
tries, ``pbest`` once, deviation 2 of :mod:`de_common`); ``pb`` at
the start of a ``step`` is the value the instance last computed, here
always recomputed from the evaluation count; the step-start archive cut
to ``int(2.1 NP)``, a no-op once ``NP >= 15``.

**Private state** (what RL-DAS freezes and restores per variant, decision
7). The issue lists ``{n_A, p_A}``. ``nl_shade_rsp`` carries ``p_A``
only (:class:`NLSHADERSPState`): ``n_A`` restarts at 0 every generation
(Alg. 1 l.6), so it never outlives one. ``nl_shade_rsp_metabox`` carries
``pa`` (:class:`NLSHADERSPMetaBoxState`) between the generations of one
turn only: MetaBox resets it to 0.5 with every ``step`` call, so the
RL-DAS wrapper starts each turn from ``variant.init``'s fresh state and
nothing survives a switch (the same rule as ``jde21_metabox``). The
shared state
``{S_F, S_Cr, M_F, M_Cr, archive}`` is the
:class:`~l2co_optimizers._src.de_common.DEPopulation`.

**Budget.** Both entries need ``max_fevals``, the run's budget, which
the population schedule and ``Cr_b`` are planned against; they build
without it (the registry builds every name to read its ``popsize``) and
raise when a run starts. Each step bills only the rows it evaluates, and
``variant.generations_for_budget(max_fevals)`` gives the number of steps
that spends the budget without exceeding it. l2co ADR 0026's
``budget_fn`` does not use ``generations_for_budget`` yet
(bessagroup/l2co#121), so a standalone run through l2co is cut short
of the budget until it does. Under
RL-DAS metabbo-jax drives the steps and this does not apply.

**Size.** The donor draws build ``(n_max, n_max)`` masks, and the paper
variant's ``n_max = 30 D``: memory grows with ``D^2``.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import dataclasses

# Third-party
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial
from jaxtyping import (
    Array,
    ArrayLike,
    Bool,
    Float,
    Int,
    PRNGKeyArray,
    PyTree,
)

# Local
from l2co_optimizers._src.core.popsize import count_parameters
from l2co_optimizers._src.core.typing import LossFunction, StopFunction
from l2co_optimizers._src.core.update_class import UpdateClass
from l2co_optimizers._src.de_common import (
    AbstractDEVariant,
    DEPopulation,
    DEPrivateState,
    binomial_crossover,
    current_to_pbest_1,
    de_variant_update,
    exponential_crossover,
    init_population,
    insert_archive,
    nonlinear_population_size,
    record_successes,
    resize_archive,
    sample_f_cr,
    sample_index,
    shrink,
    update_memory,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


#                                                                     Operators
# =============================================================================


def cost_ranks(pop: DEPopulation) -> Int[Array, " n_max"]:
    """Rank of every slot by cost, 0 for the best live one.

    Live slots get ranks ``0 .. size - 1`` (ties by slot order); dead
    slots rank after them. The references sort the population before a
    generation; ranking instead keeps the slots where they are, so
    ``ask`` and ``tell`` see the same slot order.
    """
    cost = jnp.where(jnp.isnan(pop.cost), jnp.inf, pop.cost)
    order = jnp.lexsort((cost, ~pop.live))
    return jnp.argsort(order)


def rank_selection_weights(pop: DEPopulation) -> Float[Array, " n_max"]:
    """Rank-based selective pressure for ``r2`` taken from the population.

    ``R = exp(-i / NP)`` for the individual of rank ``i = 1 .. NP`` (1 the
    best, ``NP`` the live size): paper eq. (11), after L-SHADE-RSP;
    MetaBox ``pr = exp(-(arange(NP) + 1) / NP)`` on its sorted
    population is the same. Dead slots weigh 0. Unnormalized, for
    :func:`~l2co_optimizers._src.de_common.sample_index`.
    """
    rank = cost_ranks(pop) + 1
    weights = jnp.exp(-rank / pop.size)
    return jnp.where(pop.live, weights, 0.0)


def uses_archive(
    u: Float[Array, " n"],
    p_archive: ArrayLike,
    archive_count: ArrayLike,
    *,
    min_archive: int = 1,
) -> Bool[Array, " n"]:
    """Which individuals take ``r2`` from the archive.

    With probability ``p_archive`` (draw ``u < p_archive``), and only
    when the archive holds at least ``min_archive`` rows: the paper uses
    the archive "unless it is empty" (``min_archive = 1``); MetaBox
    cancels archive use while it holds fewer than 25 rows.
    """
    enough = jnp.asarray(archive_count) >= min_archive
    return (u < p_archive) & enough


def archive_probability(
    gain_archive: ArrayLike,
    gain_population: ArrayLike,
    n_archive: ArrayLike,
    n: ArrayLike,
    *,
    p_min: float = 0.1,
    p_max: float = 0.9,
    p_reset: float = 0.5,
) -> Float[Array, ""]:
    """Archive usage probability ``p_A`` for the next generation.

    Paper eq. (14), the mean improvement per offspring built with the
    archive against the mean per offspring built without it::

        p_A = (dF_A / n_A) / (dF_A / n_A + dF_P / (NP - n_A))

    clipped to ``[p_min, p_max]`` (after IMODE). The printed
    denominator ``1 - n_A`` is read as ``NP - n_A``, the number of
    offspring built without the archive (MetaBox reads it so too).

    Parameters
    ----------
    gain_archive, gain_population : ArrayLike
        Summed improvements ``f(x) - f(u)`` of the successful offspring
        built with and without the archive.
    n_archive : ArrayLike
        Offspring built with the archive this generation, ``n_A``.
    n : ArrayLike
        Offspring built this generation, ``NP``.
    p_min, p_max : float, optional
        Clipping range, 0.1 and 0.9 (paper).
    p_reset : float, optional
        Value when eq. (14) is undefined: no offspring used the archive,
        or no offspring improved. The paper does not say; 0.5, its
        starting value, as MetaBox's ``update_Pa`` does.

    Returns
    -------
    Float[Array, ""]
        The new ``p_A``.
    """
    n_archive = jnp.asarray(n_archive, dtype=float)
    n_population = jnp.asarray(n, dtype=float) - n_archive
    mean_archive = gain_archive / jnp.maximum(n_archive, 1.0)
    mean_population = jnp.where(
        n_population > 0, gain_population / jnp.maximum(n_population, 1.0), 0.0
    )
    total = mean_archive + mean_population
    p = mean_archive / jnp.where(total > 0, total, 1.0)
    p = jnp.clip(p, p_min, p_max)
    return jnp.where((n_archive > 0) & (total > 0), p, p_reset)


def metabox_archive_probability(
    gain_archive: ArrayLike,
    gain_population: ArrayLike,
    n_archive: ArrayLike,
    n: ArrayLike,
) -> Float[Array, ""]:
    """MetaBox's ``update_Pa``, with its argument swap.

    MetaBox computes ``fp`` as the summed relative gains of the
    successful offspring built *with* the archive and ``fa`` as those
    built *without* it, then calls ``update_Pa(fa, fp, na, NP)``::

        pa = (fa / (na + 1e-15)) / (fa / (na + 1e-15) + fp / (NP - na + 1e-15))

    clipped to ``[0.1, 0.9]``, and 0.5 when ``na == 0`` or ``fa == 0``.
    ``na`` counts only the *successful* archive users. So the archive
    probability grows with the population offspring's gains (quirk 10).

    Parameters
    ----------
    gain_archive, gain_population : ArrayLike
        Summed relative improvements of the successful offspring built
        with and without the archive.
    n_archive : ArrayLike
        Successful offspring built with the archive (``na``).
    n : ArrayLike
        The generation's population size ``NP``.

    Returns
    -------
    Float[Array, ""]
        The next generation's ``pa``.
    """
    fa = jnp.asarray(gain_population, dtype=float)
    fp = jnp.asarray(gain_archive, dtype=float)
    na = jnp.asarray(n_archive, dtype=float)
    np_ = jnp.asarray(n, dtype=float)
    mean_a = fa / (na + 1e-15)
    pa = mean_a / (mean_a + fp / (np_ - na + 1e-15))
    pa = jnp.minimum(0.9, jnp.maximum(pa, 0.1))
    return jnp.where((na == 0) | (fa == 0), 0.5, pa)


def sort_by_rank(
    values: Float[Array, " n_max"], pop: DEPopulation
) -> Float[Array, " n_max"]:
    """Hand the live slots' values out in ascending order of cost rank.

    The best individual gets the smallest value, the next best the next
    one: paper Alg. 1 l.17 ("sort Cr according to fitness"); MetaBox's
    ``Cr = np.sort(Cr)`` on its sorted population. Dead slots get NaN.
    """
    ranked = jnp.sort(jnp.where(pop.live, values, jnp.inf))
    got = ranked[cost_ranks(pop)]
    return jnp.where(pop.live, got, jnp.nan)


def binomial_crossover_rate(
    fevals: ArrayLike, max_fevals: ArrayLike
) -> Float[Array, ""]:
    """Crossover rate of the binomial branch, ``Cr_b`` (paper eq. 15).

    0 while ``NFE < 0.5 NFE_max``, then ``2 (NFE / NFE_max - 0.5)``,
    rising to 1 at the end of the budget.
    """
    ratio = jnp.clip(jnp.asarray(fevals) / max_fevals, 0.0, 1.0)
    return jnp.where(ratio < 0.5, 0.0, 2.0 * (ratio - 0.5))


def metabox_binomial_crossover_rate(
    fevals: ArrayLike, max_fevals: ArrayLike, position: Int[Array, " n"]
) -> Float[Array, " n"]:
    """MetaBox's ``Crb``: the paper's condition turned around.

    ``Crb[i] = 2 ((FEs + i) / MaxFEs - 0.5)`` while ``FEs + i < 0.5
    MaxFEs``, else 0, with ``i`` the individual's position in the sorted
    population. So ``Crb <= 0`` throughout, and the binomial branch
    takes only the ``jrand`` component from the mutant.
    """
    nfe = jnp.asarray(fevals) + position
    return jnp.where(
        nfe < 0.5 * max_fevals, 2.0 * (nfe / max_fevals - 0.5), 0.0
    )


def crossover(
    x: Float[Array, "n d"],
    v: Float[Array, "n d"],
    cr: Float[Array, " n"],
    crb: ArrayLike,
    branch_u: Float[Array, " n"],
    u: Float[Array, "n d"],
    jrand: Int[Array, " n"],
    start: Int[Array, " n"],
) -> Float[Array, "n d"]:
    """The paper's crossover: binomial or exponential per individual.

    With probability 0.5 (``branch_u < 0.5``) binomial crossover with
    ``Cr_b`` (:func:`binomial_crossover_rate`), else exponential
    crossover with the individual's ``Cr_i`` (paper Alg. 1 l.32-36).
    Both draw from the same uniforms ``u``; only one is used per row.
    """
    binomial = binomial_crossover(
        x, v, jnp.broadcast_to(crb, cr.shape), u, jrand
    )
    exponential = exponential_crossover(x, v, cr, start, u)
    return jnp.where((branch_u < 0.5)[:, None], binomial, exponential)


def metabox_exponential_crossover(
    x: Float[Array, "n d"],
    v: Float[Array, "n d"],
    cr: Float[Array, " n"],
    start: Int[Array, " n"],
    u: Float[Array, "n d"],
) -> Float[Array, "n d"]:
    """MetaBox's ``Exponential_``: not a contiguous run.

    Component ``j`` comes from the mutant when ``j >= start`` (no
    wrap-around) and its own draw is *above* ``Cr``, so each such
    component crosses independently with probability ``1 - Cr``, and a
    trial may equal its parent.
    """
    j = jnp.arange(x.shape[1])[None, :]
    take = (j >= start[:, None]) & (u > cr[:, None])
    return jnp.where(take, v, x)


def metabox_crossover(
    x: Float[Array, "n d"],
    v: Float[Array, "n d"],
    cr: Float[Array, " n"],
    crb: Float[Array, " n"],
    branch_u: ArrayLike,
    u: Float[Array, "n d"],
    jrand: Int[Array, " n"],
    start: Int[Array, " n"],
) -> Float[Array, "n d"]:
    """MetaBox's crossover: one branch for the whole population.

    A single draw ``branch_u < 0.5`` sets ``CrossExponential``, which
    then runs *binomial* crossover with ``Crb``
    (:func:`metabox_binomial_crossover_rate`, never positive, so only
    ``jrand`` crosses); otherwise :func:`metabox_exponential_crossover`
    with ``Cr``.
    """
    binomial = binomial_crossover(x, v, crb, u, jrand)
    exponential = metabox_exponential_crossover(x, v, cr, start, u)
    return jnp.where(branch_u < 0.5, binomial, exponential)


def resample_out_of_bounds(
    trials: Float[Array, "n d"],
    low: ArrayLike,
    high: ArrayLike,
    u: Float[Array, "n d"],
) -> Float[Array, "n d"]:
    """Redraw each component outside ``[low, high]`` uniformly inside.

    Paper eq. (4); MetaBox does the same with a hard-coded box of
    ``[-100, 100]``. A uniform redraw needs both sides finite: in a
    half-open box (one side ``inf``) a component past the finite side
    is clipped onto it instead, and an infinite side never triggers.
    """
    outside = (trials < low) | (trials > high)
    closed = jnp.isfinite(low) & jnp.isfinite(high)
    redraw = low + (high - low) * jnp.where(closed, u, 0.0)
    fresh = jnp.where(closed, redraw, jnp.clip(trials, low, high))
    return jnp.where(outside, fresh, trials)


def draw_donors(
    key: PRNGKeyArray,
    pop: DEPopulation,
    n_pbest: ArrayLike,
    from_archive: Bool[Array, " n_max"],
) -> tuple[Int[Array, " n_max"], Int[Array, " n_max"], Int[Array, " n_max"]]:
    """Donors for current-to-pbest/1 with rank-based ``r2``.

    ``pbest`` is uniform among the ``n_pbest`` best live individuals
    other than ``i``; ``r1`` uniform among the live ones other than
    ``i`` and ``pbest``. ``r2`` is a uniformly drawn valid archive row
    where ``from_archive``, else a live individual other than ``i``,
    ``pbest`` and ``r1`` drawn with :func:`rank_selection_weights`
    (paper Sec. III; MetaBox the same, with re-draws instead of exact
    exclusion).

    Returns
    -------
    tuple
        ``(pbest, r1, r2)``; ``r2`` indexes ``concat(x, archive)``, as
        :func:`~l2co_optimizers._src.de_common.current_to_pbest_1`
        expects.
    """
    n = pop.x.shape[0]
    idx = jnp.arange(n)
    col, row = idx[None, :], idx[:, None]
    k_pbest, k_r1, k_r2, k_arc = jr.split(key, 4)
    live = pop.live[None, :] & (col != row)
    top = (cost_ranks(pop) < n_pbest)[None, :]
    pbest = sample_index(k_pbest, live & top, fallback=idx)
    live = live & (col != pbest[:, None])
    r1 = sample_index(k_r1, live, fallback=idx)
    live = live & (col != r1[:, None])
    r2 = sample_index(
        k_r2, live, weights=rank_selection_weights(pop), fallback=idx
    )
    row_arc = jnp.floor(jr.uniform(k_arc, (n,)) * pop.archive_count)
    r2 = jnp.where(from_archive, n + row_arc.astype(r2.dtype), r2)
    return pbest, r1, r2


def _pbest_pool(pop: DEPopulation, pb: ArrayLike) -> Int[Array, ""]:
    """``int(max(2, NP * pb))``: the pbest pool (MetaBox's rounding)."""
    return jnp.maximum(2, jnp.floor(pop.size * pb)).astype(jnp.int32)


def _resize_archive_randomly(
    pop: DEPopulation, capacity: ArrayLike, score: Float[Array, " archive_max"]
) -> DEPopulation:
    """Set the capacity; drop random rows when the archive is over it.

    :func:`~l2co_optimizers._src.de_common.resize_archive` with a key,
    taking the uniform draws instead (``tell`` has no key).
    """
    n_archive = pop.archive.shape[0]
    valid = jnp.arange(n_archive) < pop.archive_count
    order = jnp.argsort(jnp.where(valid, score, jnp.inf))
    over = pop.archive_count > capacity
    archive = jnp.where(over, pop.archive[order], pop.archive)
    return resize_archive(
        eqx.tree_at(lambda p: p.archive, pop, archive), capacity
    )


def _replace(
    pop: DEPopulation,
    trials: Float[Array, "n_max d"],
    cost: Float[Array, " n_max"],
    aux: dict,
    replace: Bool[Array, " n_max"],
    success: Bool[Array, " n_max"],
) -> DEPopulation:
    """Trials take their parents' slots where ``replace``.

    A successful trial also keeps the ``F`` and ``Cr`` that built it
    (MetaBox's ``population.F[optim] = F[optim]``).
    """
    return eqx.tree_at(
        lambda p: (p.x, p.cost, p.f, p.cr),
        pop,
        (
            jnp.where(replace[:, None], trials, pop.x),
            jnp.where(replace, cost, pop.cost),
            jnp.where(success, aux["f"], pop.f),
            jnp.where(success, aux["cr"], pop.cr),
        ),
    )


#                                                                      Variants
# =============================================================================


class NLSHADERSPState(DEPrivateState):
    """The paper variant's own adaptive state.

    Attributes
    ----------
    p_archive : Float[Array, ""]
        Archive usage probability ``p_A`` (paper eq. 14), adapted at the
        end of every generation. ``n_A``, the count of offspring built
        with the archive, restarts at 0 each generation (Alg. 1 l.6), so
        it is not carried.
    """

    p_archive: Float[Array, ""]


class NLSHADERSPMetaBoxState(DEPrivateState):
    """MetaBox's ``pa``, adapted between the generations of one turn.

    MetaBox resets ``pa`` to 0.5 at the start of every ``step`` call (one
    RL-DAS turn of many generations) and adapts it after each generation
    with :func:`metabox_archive_probability`. Nothing survives a switch:
    the RL-DAS wrapper starts each turn from ``variant.init``'s fresh
    state rather than restoring a frozen one. ``pb`` is recomputed from
    the evaluation count.

    Attributes
    ----------
    p_archive : Float[Array, ""]
        ``pa``, the probability that ``r2`` comes from the archive.
    """

    p_archive: Float[Array, ""]


class NLSHADERSP(AbstractDEVariant):
    """NL-SHADE-RSP as published (Stanovov et al., CEC 2021).

    See the module docstring for the algorithm, its parameters and the
    deviations.
    """

    d: int = eqx.field(static=True)
    n_max: int = eqx.field(static=True)
    n_min: int = eqx.field(static=True)
    memory_size: int = eqx.field(static=True)
    archive_rate: float = eqx.field(static=True)
    memory_init: float = eqx.field(static=True)
    pb_max: float = eqx.field(static=True)
    pb_min: float = eqx.field(static=True)
    p_archive_init: float = eqx.field(static=True)
    p_archive_min: float = eqx.field(static=True)
    p_archive_max: float = eqx.field(static=True)
    memory_blend: float = eqx.field(static=True)
    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)

    def __init__(
        self,
        d: int,
        *,
        n_max: int | None = None,
        n_min: int = 4,
        memory_size: int | None = None,
        archive_rate: float = 2.1,
        memory_init: float = 0.2,
        pb_max: float = 0.4,
        pb_min: float = 0.2,
        p_archive_init: float = 0.5,
        p_archive_min: float = 0.1,
        p_archive_max: float = 0.9,
        memory_blend: float = 0.5,
        low: float = -jnp.inf,
        high: float = jnp.inf,
    ):
        self.d = d
        self.n_max = 30 * d if n_max is None else n_max
        self.n_min = n_min
        self.memory_size = 20 * d if memory_size is None else memory_size
        self.archive_rate = archive_rate
        self.memory_init = memory_init
        self.pb_max = pb_max
        self.pb_min = pb_min
        self.p_archive_init = p_archive_init
        self.p_archive_min = p_archive_min
        self.p_archive_max = p_archive_max
        self.memory_blend = memory_blend
        self.low = low
        self.high = high

    def _archive_size(self, n: ArrayLike) -> Int[Array, ""]:
        return jnp.round(self.archive_rate * n).astype(jnp.int32)

    def init(self, key, x):  # noqa: D102
        archive_max = round(self.archive_rate * self.n_max)
        pop = init_population(
            key,
            (self.low, self.high),
            n_max=self.n_max,
            d=self.d,
            memory_size=self.memory_size,
            archive_max=archive_max,
            memory_f=self.memory_init,
            memory_cr=self.memory_init,
            x=x,
        )
        private = NLSHADERSPState(jnp.asarray(self.p_archive_init))
        return pop, private

    def population_size(self, fevals, max_fevals):  # noqa: D102
        return nonlinear_population_size(
            fevals, max_fevals, n_max=self.n_max, n_min=self.n_min
        )

    def ask(self, key, pop, private, progress):  # noqa: D102
        n, d = pop.x.shape
        k_fcr, k_arc, k_don, k_br, k_u, k_j, k_s, k_b, k_in, k_rs = jr.split(
            key, 10
        )
        f, cr = sample_f_cr(k_fcr, pop)
        cr = sort_by_rank(cr, pop)
        pb = self.pb_max + (self.pb_min - self.pb_max) * progress.ratio
        from_archive = pop.live & uses_archive(
            jr.uniform(k_arc, (n,)), private.p_archive, pop.archive_count
        )
        pbest, r1, r2 = draw_donors(
            k_don, pop, _pbest_pool(pop, pb), from_archive
        )
        v = current_to_pbest_1(pop, f, pbest, r1, r2)
        trials = crossover(
            pop.x,
            v,
            cr,
            binomial_crossover_rate(progress.fevals, progress.max_fevals),
            jr.uniform(k_br, (n,)),
            jr.uniform(k_u, (n, d)),
            jr.randint(k_j, (n,), 0, d),
            jr.randint(k_s, (n,), 0, d),
        )
        trials = resample_out_of_bounds(
            trials, self.low, self.high, jr.uniform(k_b, (n, d))
        )
        trials = jnp.where(pop.live[:, None], trials, jnp.nan)
        aux = {
            "f": f,
            "cr": cr,
            "from_archive": from_archive,
            "u_insert": jr.uniform(k_in, (n,)),
            "u_resize": jr.uniform(k_rs, (pop.archive.shape[0],)),
        }
        return trials, aux

    def tell(self, pop, private, trials, trial_cost, aux, progress):  # noqa: D102
        cost = jnp.where(pop.live, trial_cost, jnp.inf)
        replace = pop.live & (cost <= pop.cost)
        success = pop.live & (cost < pop.cost)
        # Improvement f(x) - f(u); a parent without a finite cost (a NaN
        # loss) weighs 0, so the memory weights stay finite.
        finite = jnp.isfinite(pop.cost)
        gain = jnp.where(success & finite, pop.cost - cost, 0.0)
        used = aux["from_archive"] & pop.live
        private = NLSHADERSPState(
            archive_probability(
                jnp.sum(jnp.where(used, gain, 0.0)),
                jnp.sum(jnp.where(used, 0.0, gain)),
                jnp.sum(used),
                pop.size,
                p_min=self.p_archive_min,
                p_max=self.p_archive_max,
                p_reset=self.p_archive_init,
            )
        )
        pop = insert_archive(pop, pop.x, success, aux["u_insert"])
        pop = record_successes(pop, aux["f"], aux["cr"], gain, success)
        pop = _replace(pop, trials, cost, aux, replace, success)
        n_next = self.population_size(progress.fevals, progress.max_fevals)
        pop = shrink(pop, n_next, mode="best")
        pop = _resize_archive_randomly(
            pop, self._archive_size(n_next), aux["u_resize"]
        )
        # Eq. (7)-(9); Alg. 1 l.51 advances k even without a success.
        k = pop.memory_index
        pop = update_memory(pop, old_weight=self.memory_blend)
        pop = eqx.tree_at(
            lambda p: p.memory_index,
            pop,
            ((k + 1) % self.memory_size).astype(jnp.int32),
        )
        return pop, private


class NLSHADERSPMetaBox(AbstractDEVariant):
    """MetaBox v2's ``NL_SHADE_RSP``, quirks included.

    See the module docstring for the quirks it reproduces.
    """

    d: int = eqx.field(static=True)
    n_max: int = eqx.field(static=True)
    n_min: int = eqx.field(static=True)
    memory_size: int = eqx.field(static=True)
    archive_rate: float = eqx.field(static=True)
    memory_init: float = eqx.field(static=True)
    f_init: float = eqx.field(static=True)
    cr_init: float = eqx.field(static=True)
    pb_max: float = eqx.field(static=True)
    pb_min: float = eqx.field(static=True)
    p_archive: float = eqx.field(static=True)
    min_archive: int = eqx.field(static=True)
    box: float = eqx.field(static=True)

    def __init__(
        self,
        d: int,
        *,
        n_max: int = 170,
        n_min: int = 30,
        memory_size: int | None = None,
        archive_rate: float = 2.1,
        memory_init: float = 0.2,
        f_init: float = 0.5,
        cr_init: float = 0.9,
        pb_max: float = 0.4,
        pb_min: float = 0.2,
        p_archive: float = 0.5,
        min_archive: int = 25,
        box: float = 100.0,
    ):
        self.d = d
        self.n_max = n_max
        self.n_min = n_min
        self.memory_size = 20 * d if memory_size is None else memory_size
        self.archive_rate = archive_rate
        self.memory_init = memory_init
        self.f_init = f_init
        self.cr_init = cr_init
        self.pb_max = pb_max
        self.pb_min = pb_min
        self.p_archive = p_archive
        self.min_archive = min_archive
        self.box = box

    def init(self, key, x):  # noqa: D102
        pop = init_population(
            key,
            (-self.box, self.box),
            n_max=self.n_max,
            d=self.d,
            memory_size=self.memory_size,
            archive_max=int(self.n_max * self.archive_rate),
            memory_f=self.memory_init,
            memory_cr=self.memory_init,
            f_init=self.f_init,
            cr_init=self.cr_init,
            x=x,
        )
        return pop, NLSHADERSPMetaBoxState(
            jnp.asarray(self.p_archive, dtype=float)
        )

    def population_size(self, fevals, max_fevals):  # noqa: D102
        return nonlinear_population_size(
            fevals, max_fevals, n_max=self.n_max, n_min=self.n_min
        )

    def ask(self, key, pop, private, progress):  # noqa: D102
        n, d = pop.x.shape
        k_fcr, k_arc, k_don, k_br, k_u, k_j, k_s, k_b, k_in = jr.split(key, 9)
        f, cr = sample_f_cr(k_fcr, pop, negative_f="reflect")
        cr = sort_by_rank(cr, pop)
        pb = self.pb_max + (self.pb_min - self.pb_max) * progress.ratio
        from_archive = pop.live & uses_archive(
            jr.uniform(k_arc, (n,)),
            private.p_archive,
            pop.archive_count,
            min_archive=self.min_archive,
        )
        pbest, r1, r2 = draw_donors(
            k_don, pop, _pbest_pool(pop, pb), from_archive
        )
        v = current_to_pbest_1(pop, f, pbest, r1, r2)
        crb = metabox_binomial_crossover_rate(
            progress.fevals, progress.max_fevals, cost_ranks(pop)
        )
        branch_u = jr.uniform(k_br)
        trials = metabox_crossover(
            pop.x,
            v,
            cr,
            crb,
            branch_u,
            jr.uniform(k_u, (n, d)),
            jr.randint(k_j, (n,), 0, d),
            jr.randint(k_s, (n,), 0, d),
        )
        trials = resample_out_of_bounds(
            trials, -self.box, self.box, jr.uniform(k_b, (n, d))
        )
        trials = jnp.where(pop.live[:, None], trials, jnp.nan)
        aux = {
            "f": f,
            "cr": jnp.where(branch_u < 0.5, crb, cr),
            "u_insert": jr.uniform(k_in, (n,)),
            "from_archive": from_archive,
        }
        return trials, aux

    def tell(self, pop, private, trials, trial_cost, aux, progress):  # noqa: D102
        cost = jnp.where(pop.live, trial_cost, jnp.inf)
        success = pop.live & (cost < pop.cost)
        # ``for i in range(len(optim)): update_archive(i)``: the leading
        # rows of the sorted population, before the group is updated.
        by_rank = jnp.argsort(cost_ranks(pop))
        leading = jnp.arange(pop.x.shape[0]) < jnp.sum(success)
        pop = insert_archive(pop, pop.x[by_rank], leading, aux["u_insert"])
        finite = jnp.isfinite(pop.cost)
        relative = (pop.cost - cost) / (pop.cost + 1e-9)
        weight = jnp.where(success & finite, relative, 0.0)
        # ``update_Pa(fa, fp, na, NP)`` after the generation (quirk 10).
        used = success & aux["from_archive"]
        private = NLSHADERSPMetaBoxState(
            metabox_archive_probability(
                jnp.sum(jnp.where(used, weight, 0.0)),
                jnp.sum(jnp.where(used, 0.0, weight)),
                jnp.sum(used),
                pop.size,
            )
        )
        pop = record_successes(pop, aux["f"], aux["cr"], weight, success)
        pop = _replace(pop, trials, cost, aux, success, success)
        n_next = self.population_size(progress.fevals, progress.max_fevals)
        pop = shrink(pop, n_next, mode="best")
        limit = jnp.floor(
            jnp.maximum(n_next * self.archive_rate, self.n_min)
        ).astype(jnp.int32)
        capacity = jnp.where(
            limit < pop.archive_count, limit, pop.archive_capacity
        )
        pop = resize_archive(pop, capacity)
        # ``update_M_F_Cr`` / ``mean_wL``: no blending, 0.5 for a
        # weighted sum at or below 1e-6, 0.5 at k (not advanced) on a
        # generation without success.
        pop = update_memory(
            pop,
            on_empty=0.5,
            small_sum_threshold=1e-6,
            small_sum_value=0.5,
        )
        return pop, private


#                                                         UpdateClass factories
# =============================================================================


def _update(
    name: str,
    variant: AbstractDEVariant,
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None,
    stop_fn: StopFunction | None,
    max_fevals: float | None,
) -> UpdateClass:
    """:func:`de_variant_update`, buildable before the budget is known.

    The registry builds every name without a budget (to read its
    ``popsize``, to check configs), so a missing ``max_fevals`` is an
    error only when a run starts: ``init_fn`` raises.
    """
    update_class = de_variant_update(
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        variant=variant,
        max_fevals=jnp.inf if max_fevals is None else max_fevals,
        bounded=bounded,
        stop_fn=stop_fn,
    )
    if max_fevals is not None:
        return update_class

    def init_fn(params, key=None, **sample):
        raise ValueError(
            f"{name} needs max_fevals, the run's evaluation budget: its "
            "population shrinks against it. Pass it as a hyperparameter, "
            "e.g. max_fevals=20000."
        )

    return dataclasses.replace(update_class, init_fn=init_fn)


def _box(bounded) -> tuple[float, float]:
    low, high = (None, None) if bounded is None else bounded
    return (
        -jnp.inf if low is None else float(low),
        jnp.inf if high is None else float(high),
    )


def nl_shade_rsp_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    max_fevals: float | None = None,
    **hyperparameters,
) -> UpdateClass:
    """``UpdateClass`` for :class:`NLSHADERSP`, the published algorithm.

    Parameters
    ----------
    model : PyTree
        Model whose inexact-array leaves are optimized; their count is
        the dimensionality ``D``.
    loss_fn : LossFunction
        Scalar loss, ``loss_fn(model, **sample)`` (with ``key=`` when
        ``pass_rng``).
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword.
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    bounded : tuple[float | None, float | None] or None, optional
        Box; out-of-box trial components are redrawn uniformly inside it
        (paper eq. 4). With one side ``None`` a component past the
        finite side is clipped onto it (deviation 10). Unbounded by
        default.
    stop_fn : StopFunction or None, optional
        Stopping function.
    max_fevals : float or None, optional
        The run's evaluation budget, which the population-size and
        ``Cr_b`` schedules are planned against. Required to run; the
        factory builds without it.
    **hyperparameters
        Keyword arguments of :class:`NLSHADERSP` (``n_max``, ``n_min``,
        ``memory_size``, ``archive_rate``, ...).

    Returns
    -------
    UpdateClass
        Population-family update class of ``popsize = n_max``.
    """
    low, high = _box(bounded)
    variant = NLSHADERSP(
        count_parameters(model), low=low, high=high, **hyperparameters
    )
    return _update(
        "nl_shade_rsp",
        variant,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
        max_fevals=max_fevals,
    )


def nl_shade_rsp_metabox_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    max_fevals: float | None = None,
    **hyperparameters,
) -> UpdateClass:
    """``UpdateClass`` for :class:`NLSHADERSPMetaBox`.

    Parameters are those of :func:`nl_shade_rsp_update`, except that
    ``bounded`` only clips the trials (the adapter's safety net):
    MetaBox repairs against its own ``[-100, 100]``. ``**hyperparameters``
    go to :class:`NLSHADERSPMetaBox`.
    """
    variant = NLSHADERSPMetaBox(count_parameters(model), **hyperparameters)
    return _update(
        "nl_shade_rsp_metabox",
        variant,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
        max_fevals=max_fevals,
    )


nl_shade_rsp_mapping = {
    "nl_shade_rsp": Partial(nl_shade_rsp_update),
    "nl_shade_rsp_metabox": Partial(nl_shade_rsp_metabox_update),
}
