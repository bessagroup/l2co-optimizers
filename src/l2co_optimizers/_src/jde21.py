"""
j21 (jDE with two populations) and MetaBox v2's JDE21.

Two registry entries built on :mod:`~l2co_optimizers._src.de_common`:

* ``jde21`` -- :class:`JDE21`, the published algorithm: Brest, Maučec
  and Bošković, "Self-adaptive Differential Evolution Algorithm with
  Population Size Reduction for Single Objective Bound-Constrained
  Optimization: Algorithm j21", CEC 2021, pp. 817-824,
  doi:10.1109/CEC45853.2021.9504782.
* ``jde21_metabox`` -- :class:`JDE21Metabox`, MetaBox v2's ``JDE21``
  (``src/environment/optimizer/rldas_optimizer.py``, vendored at
  ``tests/reference/metabox_rldas``) with its quirks, the version the
  RL-DAS agent (Guo et al. 2024) switches to.

**Algorithm.** Two populations: a big one ``Pb`` (``bNP``) and a small
one ``Ps`` (``sNP``). A generation is one jDE generation on ``Pb``
(DE/rand/1/bin, ``r1`` from ``Pb``, ``r2``, ``r3`` from ``Pb`` and the
first ``|Ms|`` individuals of ``Ps``; each trial competes with the
``Pb`` individual nearest to it, crowding), then the best individual is
copied into ``Ps`` if it is in ``Pb``, then ``m = bNP // sNP`` jDE
generations on ``Ps``. ``F`` and ``CR`` self-adapt per individual (jDE,
Brest et al. 2006): with probability ``tau1`` a new ``F = F_l + rand *
F_u``, with probability ``tau2`` a new ``CR = CR_l + rand * CR_u``, the
limits differing between ``Pb`` and ``Ps``. ``Pb`` restarts when its
best has not improved for ``ageLmt`` evaluations or ``myEqs`` of it lie
within ``eps`` of its best; ``Ps`` restarts (keeping its best) when
``myEqs`` of it lie within ``eps``.

**Steps.** A generation spans several adapter steps, one per evaluated
batch: the big step evaluates ``bNP`` rows, each small round ``sNP``
rows, and the other rows are ``NaN``. So :meth:`_J21Base.bill`
overrides
:func:`~l2co_optimizers._src.de_common.bill_live_population`, and
``generations_for_budget(B)`` counts steps whose billed rows stay
within ``B``.

**Parameters** (Table II unless stated): ``bNP = 160``, ``sNP = 10``,
``F_l,b = 0.1``, ``F_l,s = 0.17``, ``F_u = 1.1``, ``CR_l,b = 0.0``,
``CR_l,s = 0.1``, ``CR_u,b = 1.1``, ``CR_u,s = 0.8``, ``F_init = 0.5``,
``CR_init = 0.9``, ``tau1 = tau2 = 0.1``, ``ageLmt = MaxFEs / 10``
(``age_limit = 0.1``), ``eps = 1e-12``, ``myEqs = 25 %``; ``|Ms| = 1, 2,
3`` in the budget's thirds (Sec. III); the big population halves by
pairwise competition at ``MaxFEs/4``, ``2/4``, ``3/4`` (Sec. III;
:func:`~l2co_optimizers._src.de_common.j21_population_size`);
replacement on ``f(u) <= f(x)`` (Alg. 1, step 17); trials leaving the
box are reflected into it (Sec. III). Values the paper does not state,
and where they come from:

1. **eps** -- Table II says ``1e-12``, the text ``1e-16``; we use the
   table, as MetaBox does.
2. **The stagnation count** -- the paper says "``myEqs`` percent";
   MetaBox's ``prevecEnakih`` (the name of the routine in Brest's C
   code) also requires more than two; we follow MetaBox.
3. **CR formula** -- the paper gives ``F_l + rand * F_u`` for ``F`` and
   names ``CR_l``, ``CR_u`` without a formula; we use the same form,
   ``CR_l + rand * CR_u``, as MetaBox does.
4. **Age** -- counted over all evaluations (both populations) since the
   big population's best last improved; MetaBox counts the same
   evaluations, reset on any improvement of the overall best.
5. **Which individuals form Ms, and where the best is copied** -- the
   first ``|Ms|`` slots of ``Ps``, and over ``Ps``'s first slot
   (MetaBox).
6. **Restarted individuals are not evaluated** -- their cost is ``inf``
   until a trial replaces them (MetaBox: ``1e15``); no evaluation is
   billed for a restart.
7. **Crowding replaces with the trial's F and CR** -- the winner brings
   the parameters that built it to the slot it takes (MetaBox).
8. **Reflection formula** -- ``2 * low - x`` below, ``2 * high - x``
   above, then clipped.
9. **The population at the start** is ``bNP + sNP`` uniform points
   (the adapter's starting rows), ``Pb`` first.

Deviations from the paper's per-individual pseudo-code:

1. **Synchronous donors** -- a generation's trials are built from its
   parents, not from individuals replaced earlier in the same
   generation. Selection still runs in row order: when two trials reach
   the same individual, the second competes with the winner of the
   first.
2. **Exact donor exclusion** (de_common deviation 1).
3. **At least one small round** -- ``m = max(bNP // sNP, 1)``, so a big
   population halved below ``sNP`` still gets one.
4. **No early stop** -- MetaBox stops when a cost falls below ``1e-8``
   and reports it as 0; l2co stops through ``stop_fn``.
5. **Budget** -- the last steps that would pass ``MaxFEs`` are not
   taken (``generations_for_budget``); MetaBox finishes the batch it is
   on.

**jde21_metabox quirks** (kept on purpose, from MetaBox ``JDE21.step``
and the shared ``Population``):

1. Mutants are clipped to ``[Xmin, Xmax]`` (no reflection).
2. The small population draws ``CR`` with ``CRu_b = 1.1`` instead of
   ``CRu_s = 0.8``.
3. ``r1`` is re-drawn only when it equals ``i`` *and* the tracked best
   (``*`` used where ``+`` was meant).
4. Replacement on ``<`` only.
5. The improvement recorded for the memory update is ``(cost[i] - c) /
   (cost[i] + 1e-9)`` with the loop index ``i`` rather than the replaced
   slot, read after the cost was overwritten; ``update_M_F_Cr`` then
   divides by the signed sum of these (``mean_wL``).
6. Every success archives the individual it displaces (archive
   ``int(2.1 * NP)``, random replacement when full), though JDE21 never
   reads the archive.
7. Each ``step`` call opens with a descending sort, so the best ``sNP``
   individuals start as the small population.
8. The size after a generation is the shared population's NLPSR
   schedule, ``170 -> 30`` (``n_min``), and the reduction keeps
   ``group[-NP:]`` -- the last slots, not the best.
9. ``Pb`` restarts on stagnation against the global best or when
   ``age > MaxFEs / 10``; ``Ps`` restarts only when the tracked best is
   in it, after the big step; the tracked best (``cbest_id``) then
   points at the copy in ``Ps``.
10. Memories of ``20 * D`` entries at 0.2, ``F`` / ``Cr`` start at 0.5
    / 0.9 (MetaBox ``Population``).
11. The global best ``gbest`` is read as the population's best cost.
    In MetaBox they are equal: the best is in ``Ps`` whenever ``Pb``
    restarts, and the reduction keeps ``Ps``.

**Private state** (decision 7 of the wave plan). :class:`JDE21State`
carries the paper's real adaptive state: ``age`` and the generation's
phase. :class:`JDE21MetaboxState` carries nothing MetaBox keeps across
``step`` calls; its fields (phase, age, tracked best, whether the
opening sort ran) are bookkeeping of one call spread over several
steps, and an RL-DAS wrapper reproduces MetaBox by starting each turn
from :meth:`JDE21Metabox.initial_private`. Issue #60 listed ``tau1,
tau2, ageLmt, eps, myEqs``: these are constants and are hyperparameters
here (fields of the variant), not state.

.. important::
   l2co ADR 0026's ``budget_fn`` does not yet use
   ``generations_for_budget`` (bessagroup/l2co#121), so a standalone run
   is cut short until it does. Under RL-DAS the wrapper drives the
   steps and this does not apply.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import functools
from typing import ClassVar

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PRNGKeyArray, PyTree

# Local
from l2co_optimizers._src import de_common as dc
from l2co_optimizers._src.core.state_transfer import FAMILY_POPULATION
from l2co_optimizers._src.core.typing import LossFunction, StopFunction
from l2co_optimizers._src.core.update_class import UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class JDE21State(dc.DEPrivateState):
    """What j21 carries from one step to the next.

    Attributes
    ----------
    phase : Int[Array, ""]
        Next step of the generation: ``0`` the big population, ``k >= 1``
        the small population's ``k``-th of ``m = bNP // sNP`` rounds.
    age : Int[Array, ""]
        Evaluations since the big population's best last improved; the
        big population restarts once it exceeds ``ageLmt``.
    """

    phase: Int[Array, ""]
    age: Int[Array, ""]


class JDE21MetaboxState(dc.DEPrivateState):
    """Bookkeeping inside one MetaBox ``JDE21.step`` call.

    MetaBox's ``JDE21`` keeps nothing between two ``step`` calls: ``age``
    is a local, the best-slot index lives on the shared ``Population``
    and every call starts a fresh generation on a reverse-sorted
    population. Nothing here is adaptive state that should survive an
    RL-DAS switch; it exists because one MetaBox generation spans several
    of our steps (one per evaluated batch). To reproduce MetaBox under
    RL-DAS, start each turn from :meth:`JDE21Metabox.initial_private`
    instead of restoring the frozen value.

    Attributes
    ----------
    phase : Int[Array, ""]
        As in :class:`JDE21State`.
    age : Int[Array, ""]
        Evaluations since the population's best (``cbest``) improved.
    best_slot : Int[Array, ""]
        MetaBox's ``cbest_id``: the tracked best slot, which after the
        best is copied into the small population points at the copy.
    started : Bool[Array, ""]
        Whether this call's opening reverse sort has been done.
    """

    phase: Int[Array, ""]
    age: Int[Array, ""]
    best_slot: Int[Array, ""]
    started: Bool[Array, ""]


def _reflect(x: Array, low: float, high: float) -> Array:
    """Mirror components that left ``[low, high]`` back into it."""
    x = jnp.where(x < low, 2 * low - x, x)
    x = jnp.where(x > high, 2 * high - x, x)
    return jnp.clip(x, low, high)


def _nearest(
    x: Float[Array, "n d"],
    valid: Bool[Array, " n"],
    trials: Float[Array, "m d"],
) -> Int[Array, " m"]:
    """Index of the valid row of ``x`` nearest to each trial (Euclidean)."""
    d2 = jnp.sum((trials[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    d2 = jnp.where(valid[None, :] & ~jnp.isnan(d2), d2, jnp.inf)
    return jnp.argmin(d2, axis=1)


def _metabox_update_memory(pop: dc.DEPopulation) -> dc.DEPopulation:
    """MetaBox ``Population.update_M_F_Cr`` with ``mean_wL``, signs kept.

    ``w = df / sum(df)`` without clipping: JDE21's improvements can be
    negative (they read another slot's cost), and MetaBox divides by
    their signed sum. A mean whose ``sum(w * s)`` is at most ``1e-6``
    (or not a number) is 0.5; no success writes 0.5 at ``k`` without
    advancing it. :func:`~l2co_optimizers._src.de_common.update_memory`
    weighs successes equally when the sum is not positive, hence this
    private copy.
    """
    capacity = pop.success_f.shape[0]
    valid = jnp.arange(capacity) < pop.success_count
    df = jnp.where(valid, pop.success_weight, 0.0)
    w = df / jnp.sum(df)

    def mean_wl(s):
        s = jnp.where(valid, s, 0.0)
        den = jnp.sum(w * s)
        ok = den > 1e-6
        return jnp.where(ok, jnp.sum(w * s**2) / jnp.where(ok, den, 1.0), 0.5)

    has = pop.success_count > 0
    k = pop.memory_index
    new_f = jnp.where(has, mean_wl(pop.success_f), 0.5)
    new_cr = jnp.where(has, mean_wl(pop.success_cr), 0.5)
    size = pop.memory_f.shape[0]
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
            pop.memory_f.at[k].set(new_f),
            pop.memory_cr.at[k].set(new_cr),
            jnp.where(has, (k + 1) % size, k).astype(jnp.int32),
            nan_s,
            nan_s,
            nan_s,
            jnp.array(0, dtype=jnp.int32),
        ),
    )


class JDE21Aux(eqx.Module):
    """What a step's ``ask`` hands to its ``tell``.

    ``ask`` restarts and migrates before it builds trials
    (:meth:`_J21Base.prepare`). It does not hand its prepared copy of the
    population on: ``tell`` replays :meth:`~_J21Base.prepare` on the
    population it is handed, with the same restart rows and evaluation
    count, so edits the caller made in between (under RL-DAS the wrapper
    owns the population) are kept. The replay is deterministic, so on
    an unedited population it rebuilds exactly what ``ask`` built from.

    Attributes
    ----------
    fresh : Float[Array, "n_max d"]
        Restart rows ``ask`` drew.
    fevals : Array
        Evaluations billed when ``ask`` ran.
    f, cr : Float[Array, " n_max"]
        Parameters each trial was built with.
    u_archive : Float[Array, " n_max"]
        Draws for archive replacement (MetaBox).
    """

    fresh: Float[Array, "n_max d"]
    fevals: Array
    f: Float[Array, " n_max"]
    cr: Float[Array, " n_max"]
    u_archive: Float[Array, " n_max"]


def _small_rounds(n_big_now, n_small):
    """``m = bNP // sNP`` small rounds, at least one."""
    return jnp.maximum(n_big_now // n_small, 1)


@functools.lru_cache(maxsize=64)
def _plan(variant: _J21Base, max_fevals: int) -> tuple[int, ...]:
    """Rows billed by each step of a run, until the budget is passed.

    Step 0 evaluates the starting population; then each generation is a
    big step (``bNP`` rows) and ``m`` small rounds (``sNP`` rows each),
    after which the population shrinks as :meth:`end_generation` does.
    """
    n_small = variant.n_small
    size = variant.n_big + n_small
    bills, fevals = [size], size
    while fevals <= max_fevals:
        n_big_now = size - n_small
        m = max(n_big_now // n_small, 1)
        for rows in [n_big_now] + [n_small] * m:
            bills.append(rows)
            fevals += rows
        target = int(variant.population_size(fevals, max_fevals))
        if variant._metabox:
            size = min(size, target)
        else:
            for _ in range(3):
                if size > target:
                    region = size - n_small
                    size -= region - region // 2
    return tuple(bills)


class _J21Base(dc.AbstractDEVariant):
    """Shared body of :class:`JDE21` and :class:`JDE21Metabox`."""

    n_big: int = eqx.field(static=True, default=160)
    n_small: int = eqx.field(static=True, default=10)
    tau1: float = 0.1
    tau2: float = 0.1
    f_init: float = 0.5
    cr_init: float = 0.9
    f_low_big: float = 0.1
    f_low_small: float = 0.17
    f_up: float = 1.1
    cr_low_big: float = 0.0
    cr_low_small: float = 0.1
    cr_up_big: float = 1.1
    cr_up_small: float = 0.8
    eps: float = 1e-12
    my_eqs: float = 0.25
    age_limit: float = 0.1
    n_min: int = eqx.field(static=True, default=30)
    low: float | None = eqx.field(static=True, default=None)
    high: float | None = eqx.field(static=True, default=None)

    #: MetaBox draws the small population's CR with ``CRu_b``.
    _small_cr_up_is_big: ClassVar[bool] = False
    #: jDE replaces on ``<=``; MetaBox on ``<``.
    _replace_on_tie: ClassVar[bool] = True
    #: MetaBox archives displaced individuals and records successes.
    _metabox: ClassVar[bool] = False

    def adapt(
        self,
        f: Float[Array, " n"],
        cr: Float[Array, " n"],
        rand_f: Float[Array, " n"],
        tau_f: Float[Array, " n"],
        rand_cr: Float[Array, " n"],
        tau_cr: Float[Array, " n"],
        *,
        big: bool | Bool[Array, ""],
    ) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
        """jDE self-adaptation of ``F`` and ``CR`` with fixed draws.

        ``F = F_l + rand_f * F_u`` where ``tau_f < tau1``, else the
        individual's ``F``; ``CR = CR_l + rand_cr * CR_u`` where
        ``tau_cr < tau2``, else its ``CR``. The limits are the big or the
        small population's (Brest et al. 2021, Table II).
        """
        f_low = jnp.where(big, self.f_low_big, self.f_low_small)
        cr_low = jnp.where(big, self.cr_low_big, self.cr_low_small)
        cr_up_small = (
            self.cr_up_big if self._small_cr_up_is_big else self.cr_up_small
        )
        cr_up = jnp.where(big, self.cr_up_big, cr_up_small)
        new_f = jnp.where(tau_f < self.tau1, f_low + rand_f * self.f_up, f)
        new_cr = jnp.where(tau_cr < self.tau2, cr_low + rand_cr * cr_up, cr)
        return new_f, new_cr

    def select(
        self,
        pop: dc.DEPopulation,
        private: JDE21State | JDE21MetaboxState,
        trials: Float[Array, "n d"],
        trial_cost: Float[Array, " n"],
        f: Float[Array, " n"],
        cr: Float[Array, " n"],
        u_archive: Float[Array, " n"],
        *,
        big: bool | Bool[Array, ""],
    ) -> tuple[dc.DEPopulation, JDE21State | JDE21MetaboxState]:
        """Fold evaluated trials in: crowding (big step) and jDE selection.

        In a big step each trial competes with the big-population
        individual nearest to it (crowding, Brest et al. 2021, Sec. III);
        in a small step trial ``i`` competes with slot ``i``. Trials are
        processed in row order as the reference loops do: when two trials
        reach the same individual the second competes with whichever won
        the first contest. A winner brings the ``F`` and ``CR`` that built
        it. Rows of ``trials`` that are all ``NaN`` were not evaluated.

        Parameters
        ----------
        pop : DEPopulation
            The population the trials were built from.
        private : JDE21State or JDE21MetaboxState
            The variant's state; its age (and MetaBox's best slot) is
            updated.
        trials : Float[Array, "n d"]
            Trial vectors, ``NaN`` rows where nothing was evaluated.
        trial_cost : Float[Array, " n"]
            Their costs.
        f, cr : Float[Array, " n"]
            The parameters each trial was built with.
        u_archive : Float[Array, " n"]
            Uniform draws for archive replacement (MetaBox only).
        big : bool or Bool[Array, ""]
            Whether this is the big population's step.

        Returns
        -------
        tuple[DEPopulation, JDE21State | JDE21MetaboxState]
            Population and state after selection.
        """
        n = trials.shape[0]
        slots = jnp.arange(n)
        rows = jnp.any(~jnp.isnan(trials), axis=1)
        c = jnp.where(rows & ~jnp.isnan(trial_cost), trial_cost, jnp.inf)
        n_big_now = pop.size - self.n_small
        in_big = pop.live & (slots < n_big_now)
        target = jnp.where(big, _nearest(pop.x, in_big, trials), slots)

        # [i, j]: trial j came before trial i.
        earlier = slots[None, :] < slots[:, None]
        same = target[:, None] == target[None, :]
        before = same & earlier & rows[None, :]
        # What trial i meets: the running minimum at its target.
        current = jnp.minimum(
            pop.cost[target],
            jnp.min(jnp.where(before, c[None, :], jnp.inf), axis=1),
        )
        if self._replace_on_tie:
            success = rows & (c <= current)
        else:
            success = rows & (c < current)

        # [s, j]: trial j won at slot s; the last one stays.
        hit = (target[None, :] == slots[:, None]) & success[None, :]
        last = jnp.max(jnp.where(hit, slots[None, :], -1), axis=1)
        won = last >= 0
        src = jnp.maximum(last, 0)
        new = eqx.tree_at(
            lambda p: (p.x, p.cost, p.f, p.cr),
            pop,
            (
                jnp.where(won[:, None], trials[src], pop.x),
                jnp.where(won, c[src], pop.cost),
                jnp.where(won, f[src], pop.f),
                jnp.where(won, cr[src], pop.cr),
            ),
        )
        n_rows = jnp.sum(rows).astype(jnp.int32)

        if not self._metabox:
            old_best = jnp.min(jnp.where(in_big, pop.cost, jnp.inf))
            new_best = jnp.min(jnp.where(in_big, new.cost, jnp.inf))
            improved = big & (new_best < old_best)
            age = jnp.where(improved, 0, private.age + n_rows)
            return new, eqx.tree_at(lambda s: s.age, private, age)

        # The individual each success displaces: the previous winner at
        # its target, else the parent (MetaBox ``update_archive(id)``).
        prev = jnp.max(
            jnp.where(before & success[None, :], slots[None, :], -1), axis=1
        )
        displaced = jnp.where(
            (prev >= 0)[:, None], trials[jnp.maximum(prev, 0)], pop.x[target]
        )
        new = dc.insert_archive(new, displaced, success, u_archive)

        # MetaBox's improvement reads ``cost[i]`` with the loop index i
        # (slot i of the whole population) after the write.
        idx = jnp.clip(jnp.where(big, slots, slots - n_big_now), 0, n - 1)
        upto = (
            (slots[None, :] <= slots[:, None])
            & success[None, :]
            & (target[None, :] == idx[:, None])
        )
        last_at_idx = jnp.max(jnp.where(upto, slots[None, :], -1), axis=1)
        now = jnp.where(
            last_at_idx >= 0, c[jnp.maximum(last_at_idx, 0)], pop.cost[idx]
        )
        df = (now - c) / (now + 1e-9)
        new = dc.record_successes(new, f, cr, df, success)

        # ``age += xNP``, reset when a trial beats the tracked best.
        best0 = pop.cost[private.best_slot]
        running = jnp.minimum(
            best0,
            jnp.min(
                jnp.where(earlier & success[None, :], c[None, :], jnp.inf),
                axis=1,
            ),
        )
        improve = success & (c < running)
        any_improve = jnp.any(improve)
        last_improve = jnp.max(jnp.where(improve, slots, -1))
        best_slot = jnp.where(
            any_improve,
            target[jnp.maximum(last_improve, 0)],
            private.best_slot,
        )
        age = jnp.where(any_improve, 0, private.age + n_rows)
        return new, eqx.tree_at(
            lambda s: (s.age, s.best_slot),
            private,
            (age.astype(jnp.int32), best_slot.astype(jnp.int32)),
        )

    def _stagnant(
        self, cost: Float[Array, " n"], mask: Bool[Array, " n"], best: Array
    ) -> Bool[Array, ""]:
        """More than 2, and more than ``myEqs`` of ``mask``, lie within eps.

        MetaBox's ``prevecEnakih`` (the name of the routine in Brest's
        code); the paper states the ``myEqs`` share only.
        """
        eqs = jnp.sum(mask & (jnp.abs(cost - best) < self.eps))
        return (eqs > 2) & (eqs > self.my_eqs * jnp.sum(mask))

    def _reinit(
        self,
        pop: dc.DEPopulation,
        mask: Bool[Array, " n"],
        fresh: Float[Array, "n d"],
        cost: float,
    ) -> dc.DEPopulation:
        """Re-draw the masked slots: ``fresh`` rows, initial F and CR."""
        return eqx.tree_at(
            lambda p: (p.x, p.cost, p.f, p.cr),
            pop,
            (
                jnp.where(mask[:, None], fresh, pop.x),
                jnp.where(mask, cost, pop.cost),
                jnp.where(mask, self.f_init, pop.f),
                jnp.where(mask, self.cr_init, pop.cr),
            ),
        )

    def prepare(
        self,
        pop: dc.DEPopulation,
        private: JDE21State | JDE21MetaboxState,
        progress: dc.DEProgress,
        fresh: Float[Array, "n d"],
    ) -> tuple[dc.DEPopulation, JDE21State | JDE21MetaboxState]:
        """Restarts and the best's migration before a step builds trials.

        Before a big step (``phase == 0``) the big population restarts
        when it stagnates or its age exceeds ``ageLmt = age_limit *
        MaxFEs``. The paper also checks the small population there (Alg.
        2, steps 5-6) and, before the first small round, copies the best
        individual into the small population when it is in the big one
        (steps 16-18). MetaBox checks the small population only before
        the first small round, and only when its tracked best is there.
        A restart re-draws individuals from ``fresh`` (rows already in the
        box) without evaluating them: their cost is ``inf`` (MetaBox:
        ``1e15``).

        Parameters
        ----------
        pop : DEPopulation
            Population before the step.
        private : JDE21State or JDE21MetaboxState
            The variant's state.
        progress : DEProgress
            Evaluations so far and the budget.
        fresh : Float[Array, "n d"]
            Uniform rows in the box, one per slot.

        Returns
        -------
        tuple[DEPopulation, JDE21State | JDE21MetaboxState]
            Population and state the step's trials are built from.
        """
        if self._metabox:
            return self._prepare_metabox(pop, private, progress, fresh)
        slots = jnp.arange(pop.x.shape[0])
        n_big_now = pop.size - self.n_small
        in_big = pop.live & (slots < n_big_now)
        in_small = pop.live & (slots >= n_big_now)
        before_big = private.phase == 0

        big_best = jnp.min(jnp.where(in_big, pop.cost, jnp.inf))
        restart_big = before_big & (
            self._stagnant(pop.cost, in_big, big_best)
            | (private.age > self.age_limit * progress.max_fevals)
        )
        pop = self._reinit(pop, in_big & restart_big, fresh, jnp.inf)
        age = jnp.where(restart_big, 0, private.age)

        small_best = jnp.argmin(jnp.where(in_small, pop.cost, jnp.inf))
        restart_small = before_big & self._stagnant(
            pop.cost, in_small, pop.cost[small_best]
        )
        keep = slots != small_best
        pop = self._reinit(
            pop, in_small & restart_small & keep, fresh, jnp.inf
        )

        big_best_slot = jnp.argmin(jnp.where(in_big, pop.cost, jnp.inf))
        copy = (private.phase == 1) & (
            pop.cost[big_best_slot]
            < jnp.min(jnp.where(in_small, pop.cost, jnp.inf))
        )
        pop = self._copy_best(pop, big_best_slot, n_big_now, copy)
        return pop, eqx.tree_at(
            lambda s: s.age, private, age.astype(jnp.int32)
        )

    @staticmethod
    def _copy_best(
        pop: dc.DEPopulation, src: Array, dst: Array, do: Bool[Array, ""]
    ) -> dc.DEPopulation:
        """Copy individual ``src`` (position and cost) into slot ``dst``."""
        return eqx.tree_at(
            lambda p: (p.x, p.cost),
            pop,
            (
                pop.x.at[dst].set(jnp.where(do, pop.x[src], pop.x[dst])),
                pop.cost.at[dst].set(
                    jnp.where(do, pop.cost[src], pop.cost[dst])
                ),
            ),
        )

    def _prepare_metabox(self, pop, private, progress, fresh):
        """:meth:`prepare` as MetaBox's ``JDE21.step`` orders it."""
        slots = jnp.arange(pop.x.shape[0])
        before_big = private.phase == 0

        # ``population.sort(NP, True)`` opens every ``step`` call:
        # descending cost, so the best individuals form the small
        # population at the tail.
        do_sort = before_big & ~private.started
        order = jnp.argsort(jnp.where(pop.live, -pop.cost, jnp.inf))
        pop = jax.tree.map(
            lambda a, b: jnp.where(do_sort, a, b),
            eqx.tree_at(
                lambda p: (p.x, p.cost, p.live, p.f, p.cr),
                pop,
                (
                    pop.x[order],
                    pop.cost[order],
                    pop.live[order],
                    pop.f[order],
                    pop.cr[order],
                ),
            ),
            pop,
        )
        live_cost = jnp.where(pop.live, pop.cost, jnp.inf)
        best_slot = jnp.where(
            do_sort, jnp.argmin(live_cost), private.best_slot
        )
        started = private.started | before_big

        n_big_now = pop.size - self.n_small
        in_big = pop.live & (slots < n_big_now)
        in_small = pop.live & (slots >= n_big_now)

        gbest = jnp.min(live_cost)
        restart_big = before_big & (
            self._stagnant(pop.cost, in_big, gbest)
            | (private.age > self.age_limit * progress.max_fevals)
        )
        pop = self._reinit(pop, in_big & restart_big, fresh, 1e15)
        age = jnp.where(restart_big, 0, private.age)
        best_slot = jnp.where(
            restart_big,
            jnp.argmin(jnp.where(pop.live, pop.cost, jnp.inf)),
            best_slot,
        )

        before_small = private.phase == 1
        cbest = pop.cost[best_slot]
        restart_small = (
            before_small
            & (best_slot >= n_big_now)
            & self._stagnant(pop.cost, in_small, cbest)
        )
        mask = in_small & restart_small
        kept_x = pop.x[best_slot]
        pop = self._reinit(pop, mask, fresh, 1e15)
        pop = eqx.tree_at(
            lambda p: (p.x, p.cost),
            pop,
            (
                pop.x.at[best_slot].set(
                    jnp.where(restart_small, kept_x, pop.x[best_slot])
                ),
                pop.cost.at[best_slot].set(cbest),
            ),
        )

        copy = before_small & (best_slot < n_big_now)
        pop = self._copy_best(pop, best_slot, n_big_now, copy)
        best_slot = jnp.where(copy, n_big_now, best_slot)
        return pop, eqx.tree_at(
            lambda s: (s.age, s.best_slot, s.started),
            private,
            (age.astype(jnp.int32), best_slot.astype(jnp.int32), started),
        )

    def end_generation(
        self,
        pop: dc.DEPopulation,
        private: JDE21State | JDE21MetaboxState,
        progress: dc.DEProgress,
    ) -> tuple[dc.DEPopulation, JDE21State | JDE21MetaboxState]:
        """Population-size reduction after the last small round.

        The paper halves the big population by pairwise competition
        (:func:`~l2co_optimizers._src.de_common.halve_pairwise`) each time
        ``fevals`` passes ``MaxFEs/4``, ``2/4``, ``3/4`` (Sec. III).
        MetaBox writes the generation's successes into ``M_F`` / ``M_Cr``
        (``update_M_F_Cr``) and keeps the last ``NP`` slots, ``NP`` from
        the shared population's NLPSR schedule; its tracked best becomes
        the cheapest survivor.
        """
        target = self.population_size(progress.fevals, progress.max_fevals)
        if not self._metabox:
            # Several halvings at once only when one generation spans
            # more than a quarter of the budget.
            for _ in range(3):
                halved = dc.halve_pairwise(pop, pop.size - self.n_small)
                shrink = pop.size > target
                pop = jax.tree.map(
                    lambda a, b, shrink=shrink: jnp.where(shrink, a, b),
                    halved,
                    pop,
                )
            return pop, private
        pop = _metabox_update_memory(pop)
        pop = dc.shrink(pop, target, mode="tail")
        best_slot = jnp.argmin(jnp.where(pop.live, pop.cost, jnp.inf))
        return pop, eqx.tree_at(
            lambda s: s.best_slot, private, best_slot.astype(jnp.int32)
        )

    def initial_private(self) -> JDE21State | JDE21MetaboxState:
        """State at the start of a run (MetaBox: of a ``step`` call)."""
        zero = jnp.array(0, dtype=jnp.int32)
        if self._metabox:
            return JDE21MetaboxState(
                phase=zero, age=zero, best_slot=zero, started=jnp.array(False)
            )
        return JDE21State(phase=zero, age=zero)

    def init(
        self, key: PRNGKeyArray, x: Float[Array, "n_max d"]
    ) -> tuple[dc.DEPopulation, JDE21State | JDE21MetaboxState]:
        """Population from the starting points ``x`` and a fresh state.

        MetaBox's shared ``Population``: memories of ``20 * D`` entries at
        0.2, archive ``int(2.1 * NP)``, per-individual ``F`` / ``Cr``
        0.5 / 0.9. The paper variant uses neither archive nor memories
        and gets the smallest ones.
        """
        n_max, d = x.shape
        if self._metabox:
            sizes = dict(
                memory_size=20 * d,
                archive_max=int(2.1 * n_max),
                success_capacity=2 * self.n_big,
                memory_f=0.2,
                memory_cr=0.2,
            )
        else:
            sizes = dict(memory_size=1, archive_max=1, archive_capacity=0)
        pop = dc.init_population(
            key,
            (self.low, self.high),
            n_max=n_max,
            d=d,
            f_init=self.f_init,
            cr_init=self.cr_init,
            x=x,
            **sizes,
        )
        return pop, self.initial_private()

    def _donors(
        self,
        keys: tuple[PRNGKeyArray, PRNGKeyArray, PRNGKeyArray],
        pop: dc.DEPopulation,
        private: JDE21State | JDE21MetaboxState,
        progress: dc.DEProgress,
        big: Bool[Array, ""],
    ) -> tuple[Int[Array, " n"], Int[Array, " n"], Int[Array, " n"]]:
        """``r1, r2, r3`` for DE/rand/1, exactly excluded.

        Big step: ``r1`` from the big population, ``r2``, ``r3`` from the
        big population and the first ``|Ms|`` small-population slots,
        ``|Ms| = 1, 2, 3`` in the budget's thirds. Small step: all three
        from the small population. All differ from each other and from
        ``i``, except MetaBox's ``r1``, which re-draws only when it is
        ``i`` *and* the tracked best (``*`` where ``+`` was meant).
        """
        n = pop.x.shape[0]
        slots = jnp.arange(n)
        row, col = slots[:, None], slots[None, :]
        n_big_now = pop.size - self.n_small
        third = progress.max_fevals / 3
        if self._metabox:
            mig = (
                1 + (progress.fevals >= third) + (progress.fevals >= 2 * third)
            )
        else:
            mig = 1 + (progress.fevals > third) + (progress.fevals > 2 * third)
        small = (col >= n_big_now) & pop.live[None, :]
        pool1 = jnp.where(big, col < n_big_now, small)
        pool23 = jnp.where(big, col < n_big_now + mig, small)
        if self._metabox:
            quirk = ~((row == private.best_slot) & (col == row))
            not_self = jnp.where(big, quirk, col != row)
        else:
            not_self = col != row
        k1, k2, k3 = keys
        r1 = dc.sample_index(k1, pool1 & not_self)
        distinct = (col != row) & (col != r1[:, None])
        r2 = dc.sample_index(k2, pool23 & distinct)
        r3 = dc.sample_index(k3, pool23 & distinct & (col != r2[:, None]))
        return r1, r2, r3

    def ask(
        self,
        key: PRNGKeyArray,
        pop: dc.DEPopulation,
        private: JDE21State | JDE21MetaboxState,
        progress: dc.DEProgress,
    ) -> tuple[Float[Array, "n_max d"], JDE21Aux]:
        """Trials of the step: the big population, or one small round.

        Restarts and migration first (:meth:`prepare`), then jDE
        self-adaptation (:meth:`adapt`), DE/rand/1 and binomial
        crossover. The paper reflects a trial leaving the box back into
        it; MetaBox clips the mutant. Rows outside the stepping
        population are ``NaN``.
        """
        n, d = pop.x.shape
        keys = jr.split(key, 8)
        fresh = jr.uniform(keys[0], (n, d), minval=self.low, maxval=self.high)
        pop, private = self.prepare(pop, private, progress, fresh)
        big = private.phase == 0
        slots = jnp.arange(n)
        n_big_now = pop.size - self.n_small
        rows = jnp.where(
            big, slots < n_big_now, (slots >= n_big_now) & pop.live
        )

        draws = jr.uniform(keys[1], (4, n))
        f, cr = self.adapt(pop.f, pop.cr, *draws, big=big)
        r1, r2, r3 = self._donors(
            (keys[2], keys[3], keys[4]), pop, private, progress, big
        )
        v = pop.x[r1] + f[:, None] * (pop.x[r2] - pop.x[r3])
        if self._metabox:
            v = jnp.clip(v, self.low, self.high)
        jrand = jr.randint(keys[5], (n,), 0, d)
        u = jr.uniform(keys[6], (n, d))
        trial = dc.binomial_crossover(pop.x, v, cr, u, jrand)
        if not self._metabox:
            trial = _reflect(trial, self.low, self.high)
        trials = jnp.where(rows[:, None], trial, jnp.nan)
        aux = JDE21Aux(
            fresh=fresh,
            fevals=jnp.asarray(progress.fevals),
            f=f,
            cr=cr,
            u_archive=jr.uniform(keys[7], (n,)),
        )
        return trials, aux

    def tell(
        self,
        pop: dc.DEPopulation,
        private: JDE21State | JDE21MetaboxState,
        trials: Float[Array, "n_max d"],
        trial_cost: Float[Array, " n_max"],
        aux: JDE21Aux,
        progress: dc.DEProgress,
    ) -> tuple[dc.DEPopulation, JDE21State | JDE21MetaboxState]:
        """Selection, then the next phase; the reduction after the last.

        Replays ``ask``'s restarts and migration on ``pop`` / ``private``
        (the caller's population, edits since ``ask`` included) with the
        restart rows and evaluation count ``ask`` used, then folds the
        costs in.
        """
        asked = dc.DEProgress(aux.fevals, progress.max_fevals)
        pop, private = self.prepare(pop, private, asked, aux.fresh)
        phase = private.phase
        pop, private = self.select(
            pop,
            private,
            trials,
            trial_cost,
            aux.f,
            aux.cr,
            aux.u_archive,
            big=phase == 0,
        )
        m = _small_rounds(pop.size - self.n_small, self.n_small)
        last = (phase != 0) & (phase >= m)
        ended = self.end_generation(pop, private, progress)
        pop, private = jax.tree.map(
            lambda a, b: jnp.where(last, a, b), ended, (pop, private)
        )
        next_phase = jnp.where(last, 0, phase + 1).astype(jnp.int32)
        return pop, eqx.tree_at(lambda s: s.phase, private, next_phase)

    def bill(self) -> dc.StepBill:
        """Rows each step evaluates: big step ``bNP``, small rounds ``sNP``.

        :func:`~l2co_optimizers._src.de_common.bill_live_population`
        does not fit: a big step and a small round evaluate different
        subsets. The bill replays the run's plan against ``max_fevals``.
        """

        def bill(step: int, fevals: int, max_fevals: int) -> int:
            del fevals
            plan = _plan(self, int(max_fevals))
            return plan[min(step, len(plan) - 1)]

        return bill

    def population_size(
        self, fevals: ArrayLike, max_fevals: ArrayLike
    ) -> Int[Array, ""]:
        """Size of the next generation (big + small population)."""
        if self._metabox:
            return dc.nonlinear_population_size(
                fevals,
                max_fevals,
                n_max=self.n_big + self.n_small,
                n_min=self.n_min,
            )
        return dc.j21_population_size(
            fevals, max_fevals, n_big=self.n_big, n_small=self.n_small
        )


class JDE21(_J21Base):
    """j21, as published (Brest, Maučec and Bošković, CEC 2021)."""


class JDE21Metabox(_J21Base):
    """MetaBox v2's ``JDE21``, quirks included."""

    _small_cr_up_is_big: ClassVar[bool] = True
    _replace_on_tie: ClassVar[bool] = False
    _metabox: ClassVar[bool] = True


#                                                             Registry entries
# =============================================================================


def _update(
    cls: type[_J21Base],
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
    """Build the ``UpdateClass``; a run without budget or box fails."""
    low, high = (None, None) if bounded is None else bounded
    variant = cls(
        low=None if low is None else float(low),
        high=None if high is None else float(high),
        **hyperparameters,
    )
    missing = []
    if max_fevals is None:
        missing.append(
            "the max_fevals hyperparameter (the run's evaluation budget, "
            "which the population-size reduction is planned against)"
        )
    if low is None or high is None:
        missing.append("a box: run with bounded=(low, high)")
    if not missing:
        return dc.de_variant_update(
            model=model,
            loss_fn=loss_fn,
            pass_rng=pass_rng,
            opt_hash=opt_hash,
            variant=variant,
            max_fevals=max_fevals,
            bounded=bounded,
            stop_fn=stop_fn,
        )

    # Config checks and popsize resolution build every registry name
    # without a budget, so only running fails.
    message = f"{cls.__name__} needs " + " and ".join(missing) + "."

    def init_fn(params, key, **sample):
        raise ValueError(message)

    def step_fn(carry, sample):
        raise ValueError(message)

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=variant.n_big + variant.n_small,
        hash=opt_hash,
        stop_fn=stop_fn,
        family=FAMILY_POPULATION,
    )


def jde21_update(**kwargs) -> UpdateClass:
    """``UpdateClass`` for :class:`JDE21` (registry name ``jde21``).

    Parameters
    ----------
    **kwargs
        The registry keywords (``model``, ``loss_fn``, ``pass_rng``,
        ``opt_hash``, ``bounded``, ``stop_fn``), ``max_fevals`` (the
        run's budget; required to run, not to build) and any
        :class:`JDE21` field as a hyperparameter. ``bounded`` must be a
        scalar box ``(low, high)`` to run.

    Returns
    -------
    UpdateClass
        Population-family update class; run it for
        ``JDE21(...).generations_for_budget(max_fevals)`` steps.
    """
    return _update(JDE21, **kwargs)


def jde21_metabox_update(**kwargs) -> UpdateClass:
    """``UpdateClass`` for :class:`JDE21Metabox` (``jde21_metabox``).

    Parameters
    ----------
    **kwargs
        As :func:`jde21_update`, with :class:`JDE21Metabox` fields.

    Returns
    -------
    UpdateClass
        Population-family update class.
    """
    return _update(JDE21Metabox, **kwargs)


jde21_mapping = {
    "jde21": Partial(jde21_update),
    "jde21_metabox": Partial(jde21_metabox_update),
}
