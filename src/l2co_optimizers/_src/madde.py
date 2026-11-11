"""
MadDE -- multiple adaptation differential evolution.

Biswas, Saha, De, Cobb, Das and Jalaian, "Improving Differential
Evolution through Bayesian Hyperparameter Optimization", IEEE CEC 2021,
pp. 832-840, doi:10.1109/CEC45853.2021.9504792. The authors' MATLAB code,
github.com/subhodipbiswas/MadDE (``MadDE/MadDE.m``, ``gnR1R2.m``,
``boundConstraint.m``, ``updateArchive.m``, commit ``13f0582``), settles
what the paper leaves open; it is cited below as "the authors' code".

Each generation, every individual picks one of three mutations with the
adaptive probabilities ``p_m`` (Sec. II-B, eq. 4-6):

* DE/current-to-pbest/1 + archive:
  ``v = x + F (x_pbest - x + x_r1 - z_r2)``, ``z_r2`` from ``P U A``;
* DE/current-to-rand/1 + archive: ``v = x + F (x_r1 - z_r2)``;
* DE/weighted-rand-to-qbest/1: ``v = F x_r1 + F Fa (x_qbest - x_r2)``
  with ``q = 2p - p FEs/MaxFEs`` (eq. 7) and attraction
  ``Fa = 0.5 + 0.5 FEs/MaxFEs`` (eq. 8);

repairs a component outside the box halfway between the parent and the
bound (eq. 9), and crosses over binomially (eq. 10), against a q-best row
of ``P U A`` instead of the parent with probability ``p_qBX`` (qBX,
eq. 11). ``F`` and ``Cr`` come from SHADE-style memories (eq. 13-18,
with the terminal ``M_Cr`` value of eq. 16); ``p_m`` is each strategy's
mean relative improvement, normalized and truncated to ``[0.1, 0.9]``
(eq. 20-21); the population shrinks linearly from ``NP_max`` to
``NP_min`` (eq. 19) and the archive with it (``round(A_rate NP)``).

Two registry entries:

* ``madde`` -- the paper (Sec. II-F, III-A): ``NP_max = round(2 D^2)``,
  ``NP_min = 4``, ``H = round(10 D)``, ``A_rate = 2.3``, ``p = 0.18``,
  ``p_qBX = 0.01``, ``F0 = Cr0 = 0.2``. Values the paper does not state
  are taken from the authors' code: the q-best and p-best pool sizes
  ``max(round(q NP), 2)``; the qBX pool is the leading
  ``max(round(q |P U A|), 2)`` rows of ``[P sorted; A]``; a strategy no
  individual used scores 0 in eq. 21 (MATLAB ``max(0, mean([]))``); no
  improvement at all resets ``p_m`` to uniform; the relative improvement
  divides by ``|f(x)|``; ``r2`` of the weighted mutation is drawn from
  ``P`` only (``gnR1R2``'s ``r3``); bound repair uses the run's box.
* ``madde_metabox`` -- MadDE as MetaBox v2 runs it inside RL-DAS
  (``src/environment/optimizer/rldas_optimizer.py``, classes ``MadDE``
  and ``Population``; vendored in ``tests/reference/metabox_rldas``),
  quirks included (listed below). This is the parity-tested entry.

Private state (decision 7 of the wave plan): both variants carry only
``p_m`` (:class:`MadDEState`), which is what MetaBox's ``MadDE`` keeps
from one ``step`` to the next. ``p_best`` and ``p_qBX`` are constants of
the algorithm and so hyperparameters, not state; issue #61 listed
``{p_m, p_best, p_qBX}``. ``S_F``, ``S_Cr``, ``M_F``, ``M_Cr``, ``k`` and
the archive live in the shared :class:`~l2co_optimizers._src.de_common.
DEPopulation`.

Deviations of ``madde`` from the paper and the authors' code:

1. **Exact donor exclusion and synchronous generations** -- as for every
   DE on ``de_common`` (its deviations 1-2).
2. **Archive insertion** -- each defeated parent is appended, or replaces
   a random row once the archive is full (``de_common.insert_archive``);
   the authors' ``updateArchive`` appends the generation's batch, drops
   duplicate rows and keeps a random subset. The authors' code also
   archives ``popold(I == 1)`` with ``I`` in sorted order but ``popold``
   unsorted (a misalignment); the port archives the defeated parents.
3. **Strategy probabilities are renormalized** after the truncation to
   ``[0.1, 0.9]``, so they sum to 1; the authors' code samples with the
   truncated values as cumulative thresholds (the third strategy gets
   what is left of 1, possibly nothing).
4. **Reset of ``p_m``** -- to uniform only when no strategy improved;
   the authors' ``if count_S ~= 0`` is true only when *all three*
   improved (MATLAB semantics for a vector), resetting far more often.
5. **Selection** -- replacement on ``f(u) <= f(x)`` (Algorithm 1); the
   authors' code replaces only on ``<``. Successes and archiving use
   ``<`` in both.
6. **Zero or non-finite parent cost** -- its relative improvement counts
   as 0 in eq. 21 (the paper divides by ``f(x)``, which is 0 at an
   optimum and negative for a negative loss).
7. **Crossover compares ``u < Cr``** (``de_common.binomial_crossover``);
   the paper uses ``<=``, which differs on a set of measure zero.

MetaBox quirks ``madde_metabox`` reproduces:

1. **Bound repair toward a hard-coded +-100**, whatever the problem's
   box: ``v < -100 -> (x - 100) / 2``, ``v > 100 -> (x + 100) / 2``
   (hyperparameter ``repair_bound``). On a [-5, 5] problem it never
   fires; the adapter's clip to ``bounded`` is the only box handling.
2. **Shared-population schedule** -- non-linear reduction
   ``round(170 - 140 p^(1-p))`` (``Population.NLPSR``), not MadDE's
   linear one, independent of ``D``; archive ``int(2.1 NP)`` (357 at the
   start), memory ``20 D`` entries initialised at 0.2, per-individual
   ``F`` / ``Cr`` starting at 0.5 / 0.9.
3. **Archive capacity shrinks late** -- ``NLPSR`` sets the capacity to
   ``int(max(2.1 N, 30))`` and cuts to the leading rows only once the
   archive holds more than that; until then the old capacity stands.
4. **Archive written before the group update**, one defeated parent at a
   time with random replacement when full (``update_archive``).
5. **Donors from the strategy's own group** -- each mutation runs on
   ``group[mu == m]`` only, so ``x_r1`` and the population part of
   ``z_r2`` come from individuals that drew the same strategy; pbest and
   qbest come from the whole sorted population. MetaBox's duplicate
   checks between the best index and the group index compare positions
   in different arrays and exclude nothing in particular; the port
   leaves them out (pbest and qbest may be ``x`` itself, as in the
   paper).
6. **Pool sizes truncate**: ``max(int(p NP), 2)``, ``max(int(q NP), 2)``,
   and the qBX pool ``max(int(q (NP + |A|)), 2)`` leading rows of
   ``[P sorted; A]``.
7. **F / Cr sampling** -- ``Population.choose_F_Cr``: a negative Cauchy
   ``F`` is reflected (``2 M_F - F``), not redrawn; Lehmer means with
   MetaBox's ``1e-6`` / 0.5 fallback; no success writes 0.5 at ``k``
   without advancing it.
8. **p_m update** -- ``mean(df / cost)`` per group with
   ``df = max(0, cost - ncost)``, the signed cost in the denominator; an
   unused strategy's mean is NaN, which makes the sum NaN and resets
   ``p_m`` to uniform, as does a non-positive sum (e.g. negative losses).
   Then truncation to ``[0.1, 0.9]`` and renormalization. The port also
   resets on a non-finite result, where MetaBox would crash in
   ``rng.choice``.
9. **Strict selection** -- ``ncost < cost`` replaces.

MetaBox deviations that are not reproduced: its exact random streams
(numpy vs JAX; parity is statistical, ``tests/test_madde.py``), its
duplicate re-draws (de_common deviation 1), and the early stop at a cost
below ``1e-8`` (a stopping rule, which belongs to ``stop_fn``).

**Budget.** Both entries plan their reduction against ``max_fevals``
(the run's budget ``B``, a required hyperparameter for a run; the
factory builds without it so population sizes can be resolved) and
bill only the live individuals each step. Run them for
``variant.generations_for_budget(B)`` steps to use the budget without
exceeding it. l2co ADR 0026's ``budget_fn`` does not use
``generations_for_budget`` yet (bessagroup/l2co#121): until it does, a
standalone run gets a step count sized for a fixed population and is cut
short. Under RL-DAS, metabbo-jax drives the steps and owns the schedule.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import dataclasses
import math

# Third-party
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PRNGKeyArray, PyTree

# Local
from l2co_optimizers._src import de_common as dc
from l2co_optimizers._src.core.typing import LossFunction, StopFunction
from l2co_optimizers._src.core.update_class import UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def repair_bounds(
    v: Float[Array, "n d"],
    x: Float[Array, "n d"],
    low: float | None,
    high: float | None,
) -> Float[Array, "n d"]:
    """Midpoint bound repair (paper eq. 9; MetaBox with ``+-100``).

    A component below ``low`` becomes ``(x + low) / 2``, then one above
    ``high`` becomes ``(x + high) / 2``, ``x`` being the parent.

    Parameters
    ----------
    v : Float[Array, "n d"]
        Mutants.
    x : Float[Array, "n d"]
        Their parents.
    low, high : float or None
        The box; ``None`` leaves that side unrepaired.

    Returns
    -------
    Float[Array, "n d"]
        Repaired mutants.
    """
    if low is not None:
        v = jnp.where(v < low, (x + low) / 2.0, v)
    if high is not None:
        v = jnp.where(v > high, (x + high) / 2.0, v)
    return v


def _rank(pop):
    """Rank of each slot by cost, live slots first (0 = best)."""
    cost = jnp.where(jnp.isnan(pop.cost), jnp.inf, pop.cost)
    return jnp.argsort(jnp.lexsort((cost, ~pop.live)))


def sample_donors(
    key: PRNGKeyArray,
    pop: dc.DEPopulation,
    strategy: Int[Array, " n_max"],
    n_pbest: ArrayLike,
    n_qbest: ArrayLike,
    *,
    within_strategy: bool = False,
) -> tuple[Int[Array, " n_max"], Int[Array, " n_max"], Int[Array, " n_max"]]:
    """Donor indices ``(best, r1, r2)`` for the three mutations.

    ``best`` is uniform among the ``n_pbest`` (strategy 0) or ``n_qbest``
    (strategy 2) best live individuals, ``x`` itself included; ``r1``
    among live individuals other than ``i``; ``r2`` among live
    individuals other than ``i`` and ``r1``, plus the valid archive rows
    for strategies 0 and 1 (the authors' ``gnR1R2``).

    Parameters
    ----------
    key : PRNGKeyArray
        Key for the draws.
    pop : DEPopulation
        Population and archive.
    strategy : Int[Array, " n_max"]
        Mutation strategy per slot: 0 current-to-pbest/1 + archive,
        1 current-to-rand/1 + archive, 2 weighted-rand-to-qbest/1.
    n_pbest, n_qbest : ArrayLike
        Sizes of the p-best and q-best pools.
    within_strategy : bool, optional
        Draw ``r1`` and the population part of ``r2`` only among slots
        with the same strategy (MetaBox quirk 5). Defaults to False.

    Returns
    -------
    tuple[Int[Array, " n_max"], Int[Array, " n_max"], Int[Array, " n_max"]]
        ``best`` and ``r1`` index the population; ``r2`` indexes
        ``concat(x, archive)`` (always the population for strategy 2).
        A slot without a valid candidate gets its own index.
    """
    n = pop.x.shape[0]
    n_archive = pop.archive.shape[0]
    idx = jnp.arange(n)
    rank = _rank(pop)
    n_top = jnp.where(strategy == 2, n_qbest, n_pbest)
    valid_best = pop.live[None, :] & (rank[None, :] < n_top[:, None])
    k_best, k_r1, k_r2 = jr.split(key, 3)
    best = dc.sample_index(k_best, valid_best, fallback=idx)
    not_self = idx[None, :] != idx[:, None]
    peers = pop.live[None, :] & not_self
    if within_strategy:
        peers = peers & (strategy[None, :] == strategy[:, None])
    r1 = dc.sample_index(k_r1, peers, fallback=idx)
    archive_valid = jnp.arange(n_archive) < pop.archive_count
    uses_archive = (strategy != 2)[:, None] & archive_valid[None, :]
    valid_r2 = jnp.concatenate(
        [peers & (idx[None, :] != r1[:, None]), uses_archive], axis=1
    )
    r2 = dc.sample_index(k_r2, valid_r2, fallback=idx)
    return best, r1, r2


def madde_mutants(
    pop: dc.DEPopulation,
    f: Float[Array, " n_max"],
    fa: ArrayLike,
    strategy: Int[Array, " n_max"],
    best: Int[Array, " n_max"],
    r1: Int[Array, " n_max"],
    r2: Int[Array, " n_max"],
) -> Float[Array, "n_max d"]:
    """The three MadDE mutations (eq. 4-6), picked per slot by ``strategy``.

    Parameters
    ----------
    pop : DEPopulation
        Population and archive.
    f : Float[Array, " n_max"]
        Scale factor per slot.
    fa : ArrayLike
        Attraction factor ``Fa`` (eq. 8).
    strategy : Int[Array, " n_max"]
        0, 1 or 2, as in :func:`sample_donors`.
    best, r1, r2 : Int[Array, " n_max"]
        Donors, as :func:`sample_donors` returns them.

    Returns
    -------
    Float[Array, "n_max d"]
        Mutants; ``NaN`` rows at dead slots.
    """
    union = jnp.concatenate([pop.x, pop.archive])
    f = f[:, None]
    x = pop.x
    v_pbest = x + f * (x[best] - x) + f * (x[r1] - union[r2])
    v_rand = x + f * (x[r1] - union[r2])
    v_qbest = f * x[r1] + f * fa * (
        x[best] - x[jnp.clip(r2, 0, x.shape[0] - 1)]
    )
    s = strategy[:, None]
    v = jnp.where(s == 0, v_pbest, jnp.where(s == 1, v_rand, v_qbest))
    return jnp.where(pop.live[:, None], v, jnp.nan)


def qbest_pool(pop: dc.DEPopulation, n_pool: ArrayLike) -> Bool[Array, " u"]:
    """The qBX pool: the leading ``n_pool`` rows of ``[P sorted; A]``.

    Both references take the first rows of the population sorted by
    cost followed by the archive in storage order (the archive's costs
    are not kept), so the pool is the best individuals and, only when
    ``n_pool`` exceeds the population size, archive rows.

    Parameters
    ----------
    pop : DEPopulation
        Population and archive.
    n_pool : ArrayLike
        Pool size.

    Returns
    -------
    Bool[Array, " u"]
        Mask over ``concat(x, archive)``.
    """
    rank = _rank(pop)
    in_pop = pop.live & (rank < n_pool)
    a = jnp.arange(pop.archive.shape[0])
    in_archive = (a < pop.archive_count) & (pop.size + a < n_pool)
    return jnp.concatenate([in_pop, in_archive])


def qbx_crossover(
    x: Float[Array, "n d"],
    v: Float[Array, "n d"],
    cr: Float[Array, " n"],
    u: Float[Array, "n d"],
    jrand: Int[Array, " n"],
    use_qbest: Bool[Array, " n"],
    qbest_rows: Float[Array, "n d"],
) -> Float[Array, "n d"]:
    """Binomial crossover against the parent or, where drawn, a q-best row.

    Eq. 10-11: with probability ``p_qBX`` (the caller's ``use_qbest``) the
    target vector is a row of the qBX pool (:func:`qbest_pool`) instead
    of the parent.

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
    use_qbest : Bool[Array, " n"]
        Rows crossed against ``qbest_rows``.
    qbest_rows : Float[Array, "n d"]
        The drawn q-best row per slot.

    Returns
    -------
    Float[Array, "n d"]
        Trial vectors.
    """
    base = jnp.where(use_qbest[:, None], qbest_rows, x)
    return dc.binomial_crossover(base, v, cr, u, jrand)


def update_strategy_probabilities(
    cost: Float[Array, " n"],
    trial_cost: Float[Array, " n"],
    strategy: Int[Array, " n"],
    live: Bool[Array, " n"],
    *,
    metabox: bool = False,
) -> Float[Array, " 3"]:
    """The mutation-strategy probabilities after a generation (eq. 20-21).

    Each strategy scores the mean relative improvement
    ``max(0, f(x) - f(u)) / |f(x)|`` of the live slots that used it;
    ``p_m`` is the score over the sum, truncated to ``[0.1, 0.9]`` and
    renormalized, or uniform if nothing improved. The new value does not
    depend on the old one.

    Parameters
    ----------
    cost : Float[Array, " n"]
        Parents' costs.
    trial_cost : Float[Array, " n"]
        Trials' costs (``inf`` where not evaluated).
    strategy : Int[Array, " n"]
        Strategy per slot.
    live : Bool[Array, " n"]
        Live slots.
    metabox : bool, optional
        MetaBox's rule (quirk 8): divide by the signed cost, and reset to
        uniform when a strategy went unused. Defaults to the paper's
        (authors' code; deviations 3, 4 and 6).

    Returns
    -------
    Float[Array, " 3"]
        New ``p_m``.
    """
    gain = jnp.maximum(0.0, cost - trial_cost)
    if metabox:
        ratio = gain / cost
    else:
        scale = jnp.abs(cost)
        ratio = jnp.where(
            scale > 0, gain / jnp.where(scale > 0, scale, 1.0), 0.0
        )
        ratio = jnp.where(jnp.isfinite(ratio), ratio, 0.0)
    member = live[None, :] & (strategy[None, :] == jnp.arange(3)[:, None])
    n = jnp.sum(member, axis=1)
    total = jnp.sum(jnp.where(member, ratio[None, :], 0.0), axis=1)
    if metabox:
        mean = total / n  # NaN for an unused strategy, as np.mean([])
    else:
        mean = jnp.where(n > 0, total / jnp.maximum(n, 1), 0.0)
    norm = jnp.sum(mean)
    new = jnp.clip(mean / jnp.where(norm > 0, norm, 1.0), 0.1, 0.9)
    new = new / jnp.sum(new)
    adapt = (norm > 0) & jnp.all(jnp.isfinite(new))
    return jnp.where(adapt, new, jnp.full(3, 1.0 / 3.0))


#: Terminal value of ``M_Cr`` (paper eq. 16): ``Cr`` is 0 from then on.
TERMINAL_CR = -1.0


def _round_half_up(x: float) -> int:
    """MATLAB's ``round`` for non-negative ``x``."""
    return int(math.floor(x + 0.5))


class MadDEState(dc.DEPrivateState):
    """MadDE's own adaptive state.

    Attributes
    ----------
    pm : Float[Array, " 3"]
        Probabilities of the three mutation strategies.
    """

    pm: Float[Array, " 3"]


class MadDEAux(eqx.Module):
    """What ``ask`` hands to ``tell``: the trials' parameters and draws."""

    strategy: Int[Array, " n_max"]
    f: Float[Array, " n_max"]
    cr: Float[Array, " n_max"]
    archive_u: Float[Array, " n_max"]
    prune_u: Float[Array, " archive_max"]


class MadDE(dc.AbstractDEVariant):
    """MadDE on the shared DE state; build it with :func:`madde_variant`
    (the paper) or :func:`madde_metabox_variant` (MetaBox v2).

    Attributes
    ----------
    d : int
        Dimensionality.
    n_max, n_min : int
        Starting and final population size.
    memory_size : int
        Memory entries ``H``.
    archive_rate : float
        Archive capacity per individual.
    archive_max : int
        Starting (largest) archive capacity.
    p_best : float
        ``p``; ``q`` runs from ``2p`` to ``p`` (eq. 7).
    p_qbx : float
        Probability of the q-best crossover.
    memory_f, memory_cr : float
        Initial memory entries ``F0``, ``Cr0``.
    f_init, cr_init : float
        Initial per-individual ``F`` / ``Cr`` (unused by MadDE itself,
        which draws fresh ones each generation; kept in the shared state).
    low, high : float or None
        Bound-repair box.
    metabox : bool
        Reproduce MetaBox's quirks (see the module docstring).
    """

    d: int = eqx.field(static=True)
    n_max: int = eqx.field(static=True)
    n_min: int = eqx.field(static=True)
    memory_size: int = eqx.field(static=True)
    archive_rate: float = eqx.field(static=True)
    archive_max: int = eqx.field(static=True)
    p_best: float = eqx.field(static=True)
    p_qbx: float = eqx.field(static=True)
    memory_f: float = eqx.field(static=True)
    memory_cr: float = eqx.field(static=True)
    f_init: float = eqx.field(static=True, default=0.5)
    cr_init: float = eqx.field(static=True, default=0.9)
    low: float | None = eqx.field(static=True, default=None)
    high: float | None = eqx.field(static=True, default=None)
    metabox: bool = eqx.field(static=True, default=False)

    def top_count(self, q, n):
        """``max(round(q n), 2)`` (paper code) or ``max(int(q n), 2)``."""
        qn = jnp.asarray(q) * jnp.asarray(n)
        count = jnp.floor(qn) if self.metabox else jnp.floor(qn + 0.5)
        return jnp.maximum(count, 2).astype(jnp.int32)

    def archive_size(self, n):
        """Archive capacity for population size ``n``.

        Tabulated in float64 at trace time, so ``int(2.1 * 10) = 21`` as
        in numpy, not 20 as in float32.
        """
        sizes = range(self.n_max + 1)
        if self.metabox:
            table = [
                int(max(k * self.archive_rate, self.n_min)) for k in sizes
            ]
        else:
            table = [_round_half_up(k * self.archive_rate) for k in sizes]
        table = jnp.asarray(
            np.minimum(table, self.archive_max), dtype=jnp.int32
        )
        return table[jnp.clip(n, 0, self.n_max)]

    def population_size(self, fevals, max_fevals):
        """Linear ``n_max -> n_min`` (paper) or non-linear (MetaBox)."""
        schedule = (
            dc.nonlinear_population_size
            if self.metabox
            else dc.linear_population_size
        )
        return schedule(fevals, max_fevals, n_max=self.n_max, n_min=self.n_min)

    def init(self, key, x):
        """Population from the starting points ``x``; uniform ``p_m``."""
        pop = dc.init_population(
            key,
            (0.0, 1.0),
            n_max=self.n_max,
            d=self.d,
            memory_size=self.memory_size,
            archive_max=self.archive_max,
            memory_f=self.memory_f,
            memory_cr=self.memory_cr,
            f_init=self.f_init,
            cr_init=self.cr_init,
            x=x,
        )
        return pop, MadDEState(pm=jnp.full(3, 1.0 / 3.0))

    def ask(self, key, pop, private, progress):
        """Trials of every live slot and the draws ``tell`` needs."""
        n, d = pop.x.shape
        keys = jr.split(key, 11)
        ratio = progress.ratio
        q = 2.0 * self.p_best - self.p_best * ratio
        fa = 0.5 + 0.5 * ratio

        slot = jr.randint(keys[0], (n,), 0, self.memory_size)
        f, cr = dc.f_cr_from_draws(
            pop,
            slot,
            jr.normal(keys[1], (n,)),
            jr.uniform(keys[2], (n,)),
            negative_f="reflect" if self.metabox else "truncate",
        )
        if not self.metabox:
            cr = jnp.where(pop.memory_cr[slot] == TERMINAL_CR, 0.0, cr)

        strategy = jr.categorical(keys[3], jnp.log(private.pm), shape=(n,))
        size = pop.size
        best, r1, r2 = sample_donors(
            keys[4],
            pop,
            strategy,
            self.top_count(self.p_best, size),
            self.top_count(q, size),
            within_strategy=self.metabox,
        )
        v = madde_mutants(pop, f, fa, strategy, best, r1, r2)
        v = repair_bounds(v, pop.x, self.low, self.high)

        use_qbest = jr.uniform(keys[5], (n,)) <= self.p_qbx
        pool = qbest_pool(pop, self.top_count(q, size + pop.archive_count))
        pick = dc.sample_index(
            keys[6], jnp.broadcast_to(pool[None, :], (n, pool.shape[0]))
        )
        union = jnp.concatenate([pop.x, pop.archive])
        trials = qbx_crossover(
            pop.x,
            v,
            cr,
            jr.uniform(keys[7], (n, d)),
            jr.randint(keys[8], (n,), 0, d),
            use_qbest,
            union[pick],
        )
        trials = jnp.where(pop.live[:, None], trials, jnp.nan)
        aux = MadDEAux(
            strategy=strategy,
            f=f,
            cr=cr,
            archive_u=jr.uniform(keys[9], (n,)),
            prune_u=jr.uniform(keys[10], (self.archive_max,)),
        )
        return trials, aux

    def tell(self, pop, private, trials, trial_cost, aux, progress):
        """Selection, archive, memories, ``p_m``, then the reduction."""
        live = pop.live
        improved = live & (trial_cost < pop.cost)
        if self.metabox:
            replace = improved
        else:
            replace = (
                live & jnp.isfinite(trial_cost) & (trial_cost <= pop.cost)
            )

        # Defeated parents go to the archive before the population changes.
        pop = dc.insert_archive(pop, pop.x, improved, aux.archive_u)

        gain = pop.cost - trial_cost
        gain = jnp.where(jnp.isfinite(gain), gain, 0.0)
        pop = dc.record_successes(pop, aux.f, aux.cr, gain, improved)
        k = pop.memory_index
        has = pop.success_count > 0
        valid = jnp.arange(pop.success_cr.shape[0]) < pop.success_count
        max_cr = jnp.max(jnp.where(valid, pop.success_cr, -jnp.inf))
        terminal = has & ((max_cr == 0.0) | (pop.memory_cr[k] == TERMINAL_CR))
        if self.metabox:
            pop = dc.update_memory(
                pop,
                on_empty=0.5,
                small_sum_threshold=1e-6,
                small_sum_value=0.5,
            )
        else:
            pop = dc.update_memory(pop, on_empty=0.5)
            memory_cr = pop.memory_cr.at[k].set(
                jnp.where(terminal, TERMINAL_CR, pop.memory_cr[k])
            )
            pop = eqx.tree_at(lambda p: p.memory_cr, pop, memory_cr)

        pm = update_strategy_probabilities(
            pop.cost, trial_cost, aux.strategy, live, metabox=self.metabox
        )
        pop = eqx.tree_at(
            lambda p: (p.x, p.cost),
            pop,
            (
                jnp.where(replace[:, None], trials, pop.x),
                jnp.where(replace, trial_cost, pop.cost),
            ),
        )

        n_next = self.population_size(progress.fevals, progress.max_fevals)
        old_size = pop.size
        pop = dc.shrink(pop, n_next)
        if self.metabox:
            # NLPSR: the capacity changes only when the archive is over it.
            limit = self.archive_size(n_next)
            capacity = jnp.where(
                pop.archive_count > limit, limit, pop.archive_capacity
            )
        else:
            capacity = jnp.where(
                n_next < old_size,
                self.archive_size(n_next),
                pop.archive_capacity,
            )
            # Pruning keeps a random subset (randperm in the authors' code).
            n_archive = pop.archive.shape[0]
            in_archive = jnp.arange(n_archive) < pop.archive_count
            order = jnp.argsort(jnp.where(in_archive, aux.prune_u, jnp.inf))
            archive = jnp.where(
                pop.archive_count > capacity, pop.archive[order], pop.archive
            )
            pop = eqx.tree_at(lambda p: p.archive, pop, archive)
        pop = dc.resize_archive(pop, capacity)
        return pop, MadDEState(pm=pm)


def madde_variant(
    d: int,
    *,
    popsize: int | None = None,
    popsize_multiplier: float = 2.0,
    n_min: int = 4,
    memory_multiplier: float = 10.0,
    archive_rate: float = 2.3,
    p_best: float = 0.18,
    p_qbx: float = 0.01,
    f0: float = 0.2,
    cr0: float = 0.2,
    bounds: tuple[float | None, float | None] = (None, None),
) -> MadDE:
    """MadDE as published (registry name ``madde``).

    Parameters
    ----------
    d : int
        Dimensionality.
    popsize : int or None, optional
        Starting population size; ``None`` (default) is the paper's
        ``round(popsize_multiplier * d ** 2)``. At least ``n_min``.
    popsize_multiplier : float, optional
        ``NP_m``, 2.
    n_min : int, optional
        Final population size, 4.
    memory_multiplier : float, optional
        ``H_m``: ``H = round(H_m d)``, 10.
    archive_rate : float, optional
        ``A_rate``, 2.3.
    p_best : float, optional
        ``p``, 0.18.
    p_qbx : float, optional
        ``p_qBX``, 0.01.
    f0, cr0 : float, optional
        Initial memory entries, 0.2 and 0.2.
    bounds : tuple[float | None, float | None], optional
        Box for the bound repair; ``(None, None)`` repairs nothing.

    Returns
    -------
    MadDE
        The variant.
    """
    if popsize is None:
        popsize = _round_half_up(popsize_multiplier * d * d)
    n_max = max(int(popsize), n_min)
    low, high = bounds
    return MadDE(
        d=d,
        n_max=n_max,
        n_min=n_min,
        memory_size=max(_round_half_up(memory_multiplier * d), 1),
        archive_rate=archive_rate,
        archive_max=_round_half_up(archive_rate * n_max),
        p_best=p_best,
        p_qbx=p_qbx,
        memory_f=f0,
        memory_cr=cr0,
        low=low,
        high=high,
    )


def madde_metabox_variant(
    d: int,
    *,
    n_max: int = 170,
    n_min: int = 30,
    memory_multiplier: int = 20,
    archive_rate: float = 2.1,
    p_best: float = 0.18,
    p_qbx: float = 0.01,
    memory_init: float = 0.2,
    f_init: float = 0.5,
    cr_init: float = 0.9,
    repair_bound: float = 100.0,
) -> MadDE:
    """MadDE as MetaBox v2 runs it in RL-DAS (``madde_metabox``).

    Parameters
    ----------
    d : int
        Dimensionality.
    n_max, n_min : int, optional
        ``Population.Nmax`` / ``Nmin``: 170 and 30.
    memory_multiplier : int, optional
        Memory entries per dimension, 20.
    archive_rate : float, optional
        2.1 (``int(2.1 NP)``).
    p_best : float, optional
        ``MadDE.p``, 0.18.
    p_qbx : float, optional
        ``MadDE.PqBX``, 0.01.
    memory_init : float, optional
        ``M_F`` and ``M_Cr`` start, 0.2.
    f_init, cr_init : float, optional
        ``Population.F`` / ``Cr`` start, 0.5 and 0.9.
    repair_bound : float, optional
        The hard-coded repair box ``+-100``.

    Returns
    -------
    MadDE
        The variant.
    """
    return MadDE(
        d=d,
        n_max=n_max,
        n_min=n_min,
        memory_size=memory_multiplier * d,
        archive_rate=archive_rate,
        archive_max=int(archive_rate * n_max),
        p_best=p_best,
        p_qbx=p_qbx,
        memory_f=memory_init,
        memory_cr=memory_init,
        f_init=f_init,
        cr_init=cr_init,
        low=-repair_bound,
        high=repair_bound,
        metabox=True,
    )


#                                                             Registry entries
# =============================================================================


def _dimensionality(model: PyTree) -> int:
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    return int(ravel_pytree(params)[0].size)


def _variant_update(
    variant: MadDE,
    name: str,
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None,
    stop_fn: StopFunction | None,
    max_fevals: float | None,
) -> UpdateClass:
    """:func:`de_variant_update`, refusing to start a run without a budget."""
    update = dc.de_variant_update(
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        variant=variant,
        # Any positive budget gives the starting size; the run is refused
        # below before this placeholder plans anything.
        max_fevals=1.0 if max_fevals is None else max_fevals,
        bounded=bounded,
        stop_fn=stop_fn,
    )
    if max_fevals is not None:
        return update

    def init_fn(*args, **kwargs):
        raise ValueError(
            f"{name!r} plans its population-size reduction against the "
            "run's evaluation budget: pass the hyperparameter max_fevals "
            "(the budget B) to run it."
        )

    return dataclasses.replace(update, init_fn=init_fn)


def madde_update(
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
    """Registry factory for ``madde`` (the published algorithm).

    Parameters
    ----------
    model : PyTree
        Model whose inexact-array leaves are optimized; their total size
        is the dimensionality ``D``.
    loss_fn : LossFunction
        Scalar loss.
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword.
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    bounded : tuple[float | None, float | None] or None, optional
        The run's box: the bound repair's box, and trials are clipped to
        it after ``ask``.
    stop_fn : StopFunction or None, optional
        Stopping function.
    max_fevals : float or None, optional
        Budget ``B`` the reduction is planned against. Required to run;
        ``None`` builds an entry whose ``init_fn`` raises ``ValueError``
        (enough to read ``popsize``).
    **hyperparameters
        Keywords of :func:`madde_variant` other than ``d`` and
        ``bounds``.

    Returns
    -------
    UpdateClass
        Population-family update class (``popsize = NP_max``).
    """
    bounds = (None, None) if bounded is None else tuple(bounded)
    variant = madde_variant(
        _dimensionality(model), bounds=bounds, **hyperparameters
    )
    return _variant_update(
        variant,
        "madde",
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
        max_fevals=max_fevals,
    )


def madde_metabox_update(
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
    """Registry factory for ``madde_metabox`` (MetaBox v2's MadDE).

    Parameters are those of :func:`madde_update`; ``**hyperparameters``
    go to :func:`madde_metabox_variant`. ``bounded`` only clips the
    trials (the bound repair uses ``repair_bound``).

    Returns
    -------
    UpdateClass
        Population-family update class (``popsize = 170``).
    """
    variant = madde_metabox_variant(_dimensionality(model), **hyperparameters)
    return _variant_update(
        variant,
        "madde_metabox",
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
        max_fevals=max_fevals,
    )


madde_mapping = {
    "madde": Partial(madde_update),
    "madde_metabox": Partial(madde_metabox_update),
}
