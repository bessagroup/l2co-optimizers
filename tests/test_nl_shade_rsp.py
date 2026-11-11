"""Tests for ``nl_shade_rsp`` and ``nl_shade_rsp_metabox``.

Operators are fed fixed random draws and checked against values worked
out by hand from the paper (Stanovov, Akhmedova and Semenkin, CEC 2021)
or from MetaBox v2's ``NL_SHADE_RSP`` (vendored in
``tests/reference/metabox_rldas``). The registry tests drive both
names through ``optimizer_mapping``; the slow tests compare whole runs
with the vendored MetaBox code through a stored fixture.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import json
from pathlib import Path

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.stats import mannwhitneyu

# Local
from l2co_optimizers import optimizer_mapping
from l2co_optimizers._src.de_common import DEProgress, init_population
from l2co_optimizers._src.nl_shade_rsp import (
    NLSHADERSP,
    NLSHADERSPMetaBox,
    archive_probability,
    binomial_crossover_rate,
    crossover,
    draw_donors,
    metabox_archive_probability,
    metabox_binomial_crossover_rate,
    metabox_crossover,
    rank_selection_weights,
    resample_out_of_bounds,
    sort_by_rank,
    uses_archive,
)

from .toy_problems import (
    realization_best_losses,
    rosenbrock_problem,
    sphere_problem,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def _population(cost, live=None, d=2, **kwargs):
    """A population with the given costs; dead slots NaN-padded."""
    cost = jnp.asarray(cost, dtype=float)
    n = cost.shape[0]
    live = jnp.ones(n, bool) if live is None else jnp.asarray(live)
    x = jnp.arange(n * d, dtype=float).reshape(n, d)
    pop = init_population(
        None,
        (0.0, 1.0),
        n_max=n,
        d=d,
        memory_size=kwargs.pop("memory_size", 4),
        archive_max=kwargs.pop("archive_max", 2 * n),
        x=x,
        **kwargs,
    )
    return eqx.tree_at(
        lambda p: (p.x, p.cost, p.live),
        pop,
        (
            jnp.where(live[:, None], x, jnp.nan),
            jnp.where(live, cost, jnp.inf),
            live,
        ),
    )


# =============================================================================
#                                                         Rank-based selection


def test_rank_selection_weights_follow_exp_of_minus_rank_over_np():
    # Paper eq. (11), R_i = exp(-i / NP) with i = 1 for the best; MetaBox
    # pr = exp(-(i + 1) / NP), i from 0. Three live, one dead slot.
    pop = _population([3.0, 1.0, 0.0, 2.0], live=[True, True, False, True])
    weights = rank_selection_weights(pop)
    expected = jnp.array([0.36787944, 0.71653131, 0.0, 0.51341712])
    assert jnp.allclose(weights, expected, atol=1e-7)


# =============================================================================
#                                                  Archive usage probability


@pytest.mark.parametrize(
    ("gain_archive", "gain_population", "n_archive", "expected"),
    [
        # eq. (14): (1 / 4) / (1 / 4 + 3 / 6) = 1 / 3
        (1.0, 3.0, 4, 1 / 3),
        # (3 / 2) / (3 / 2 + 1 / 8) = 0.923 -> clipped to 0.9
        (3.0, 1.0, 2, 0.9),
        # archive offspring improved nothing: 0 -> clipped to 0.1
        (0.0, 2.0, 5, 0.1),
        # every offspring used the archive: the population term is 0
        (2.0, 0.0, 10, 0.9),
        # nothing used the archive, or nothing improved: back to 0.5
        (0.0, 2.0, 0, 0.5),
        (0.0, 0.0, 5, 0.5),
    ],
)
def test_archive_probability_follows_eq_14_clipped(
    gain_archive, gain_population, n_archive, expected
):
    p = archive_probability(
        jnp.asarray(gain_archive),
        jnp.asarray(gain_population),
        jnp.asarray(n_archive),
        jnp.asarray(10),
    )
    assert jnp.allclose(p, expected)


@pytest.mark.parametrize(
    ("gain_archive", "gain_population", "n_archive", "expected"),
    [
        # MetaBox passes the population gains as fa and the archive gains
        # as fp: fa=3, fp=1, na=2 -> (3/2) / (3/2 + 1/8) = 0.923 -> 0.9
        (1.0, 3.0, 2, 0.9),
        # fa=1, fp=3, na=2 -> (1/2) / (1/2 + 3/8) = 0.5714
        (3.0, 1.0, 2, 0.5 / 0.875),
        # fa=0.1, fp=8, na=5 -> 0.02 / (0.02 + 1.6) = 0.0123 -> 0.1
        (8.0, 0.1, 5, 0.1),
        # na == 0 or fa == 0: back to 0.5
        (1.0, 3.0, 0, 0.5),
        (2.0, 0.0, 3, 0.5),
    ],
)
def test_metabox_archive_probability_swaps_fa_and_fp(
    gain_archive, gain_population, n_archive, expected
):
    p = metabox_archive_probability(
        jnp.asarray(gain_archive),
        jnp.asarray(gain_population),
        jnp.asarray(n_archive),
        jnp.asarray(10),
    )
    assert jnp.allclose(p, expected, rtol=1e-6)


def test_uses_archive_needs_a_draw_below_p_and_enough_archive_rows():
    u = jnp.array([0.1, 0.49, 0.5, 0.9])
    # Paper: with probability p_A, unless the archive is empty.
    assert uses_archive(u, 0.5, 1, min_archive=1).tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert not uses_archive(u, 0.5, 0, min_archive=1).any()
    # MetaBox: only once the archive holds 25 rows.
    assert not uses_archive(u, 0.5, 24, min_archive=25).any()
    assert uses_archive(u, 0.5, 25, min_archive=25).tolist() == [
        True,
        True,
        False,
        False,
    ]


# =============================================================================
#                                                          Crossover rates


@pytest.mark.parametrize(
    ("fevals", "expected"), [(0, 0.0), (5000, 0.0), (7500, 0.5), (10000, 1.0)]
)
def test_binomial_crossover_rate_is_zero_then_rises_to_one(fevals, expected):
    # Paper eq. (15): 0 in the first half, then 2 (NFE / NFE_max - 0.5).
    assert jnp.allclose(binomial_crossover_rate(fevals, 10000), expected)


def test_metabox_binomial_crossover_rate_is_negative_then_zero():
    # MetaBox: Crb[i] = 2 ((FEs + i) / MaxFEs - 0.5) while FEs + i is in
    # the first half, else 0 -- the paper's condition turned around.
    position = jnp.arange(4)
    crb = metabox_binomial_crossover_rate(4998, 10000, position)
    assert jnp.allclose(
        crb, jnp.array([-0.0004, -0.0002, 0.0, 0.0]), atol=1e-6
    )
    assert jnp.allclose(
        metabox_binomial_crossover_rate(0, 10000, position[:1]), -1.0
    )


def test_sort_by_rank_gives_smaller_values_to_better_individuals():
    # Paper Alg. 1 l.17 and MetaBox ``Cr = np.sort(Cr)`` on a sorted
    # population: the best gets the smallest Cr. Dead slot gets NaN.
    pop = _population([3.0, 1.0, 0.0, 2.0], live=[True, True, False, True])
    values = jnp.array([0.2, 0.9, 0.7, 0.5])
    got = sort_by_rank(values, pop)
    assert jnp.allclose(
        got, jnp.array([0.9, 0.2, jnp.nan, 0.5]), equal_nan=True
    )


# =============================================================================
#                                                                  Crossover


def test_crossover_picks_binomial_or_exponential_per_individual():
    # Paper Alg. 1 l.32-36: per individual, binomial with Cr_b on a
    # draw below 0.5, else exponential with Cr_i.
    x, v = jnp.zeros((2, 4)), jnp.ones((2, 4))
    trials = crossover(
        x,
        v,
        cr=jnp.array([0.5, 0.5]),
        crb=jnp.array(0.0),
        branch_u=jnp.array([0.2, 0.7]),
        u=jnp.array([[0.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.9, 0.1]]),
        jrand=jnp.array([1, 0]),
        start=jnp.array([0, 2]),
    )
    # Row 0, binomial with Cr_b = 0: only jrand = 1. Row 1, exponential
    # from 2: component 2, then 3 (u[1] = 0.1 < 0.5), stop (u[2] = 0.9).
    assert trials.tolist() == [[0, 1, 0, 0], [0, 0, 1, 1]]


def test_metabox_crossover_swaps_the_branches_for_the_whole_population():
    x, v = jnp.zeros((2, 4)), jnp.ones((2, 4))
    kwargs = dict(
        cr=jnp.array([0.5, 0.5]),
        crb=jnp.array([-0.2, -0.1]),
        u=jnp.array([[0.9, 0.1, 0.9, 0.6], [0.1, 0.9, 0.2, 0.8]]),
        jrand=jnp.array([3, 0]),
        start=jnp.array([1, 0]),
    )
    # One draw for everyone. Below 0.5 ("CrossExponential") MetaBox runs
    # binomial with Crb <= 0: only jrand crosses.
    low = metabox_crossover(x, v, branch_u=jnp.array(0.3), **kwargs)
    assert low.tolist() == [[0, 0, 0, 1], [1, 0, 0, 0]]
    # Otherwise its "exponential": every j >= start whose draw is above
    # Cr, not a contiguous run.
    high = metabox_crossover(x, v, branch_u=jnp.array(0.7), **kwargs)
    assert high.tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]


def test_resample_out_of_bounds_draws_uniformly_in_the_box():
    trials = jnp.array([[-150.0, 50.0, 120.0, 100.0]])
    u = jnp.array([[0.25, 0.5, 0.75, 0.0]])
    got = resample_out_of_bounds(trials, -100.0, 100.0, u)
    assert got.tolist() == [[-50.0, 50.0, 50.0, 100.0]]
    # Unbounded: nothing to repair.
    same = resample_out_of_bounds(trials, -jnp.inf, jnp.inf, u)
    assert same.tolist() == trials.tolist()


@pytest.mark.parametrize(
    "low, high, want",
    [
        (0.0, jnp.inf, [[0.0, 50.0, 120.0, 100.0]]),
        (-jnp.inf, 60.0, [[-150.0, 50.0, 60.0, 60.0]]),
    ],
)
def test_resample_out_of_bounds_clips_to_a_half_open_box(low, high, want):
    # A redraw needs both sides finite; with one side open, a component
    # past the finite side is clipped onto it, never sent to infinity.
    trials = jnp.array([[-150.0, 50.0, 120.0, 100.0]])
    u = jnp.array([[0.25, 0.5, 0.75, 0.0]])
    got = resample_out_of_bounds(trials, low, high, u)
    assert got.tolist() == want


# =============================================================================
#                                                                  Reduction


def _generation(variant, pop, private, trial_cost, fevals, max_fevals, key=0):
    """One ask / tell with the given trial costs; returns the tell."""
    progress = DEProgress(jnp.asarray(fevals), jnp.asarray(max_fevals))
    trials, aux = variant.ask(jr.key(key), pop, private, progress)
    return variant.tell(pop, private, trials, trial_cost, aux, progress)


def _evaluated(variant, cost, archive_rows=0):
    """``variant.init`` with the given costs and a partly filled archive."""
    n = variant.n_max
    x = jnp.arange(n * variant.d, dtype=float).reshape(n, variant.d)
    pop, private = variant.init(jr.key(0), x)
    rows = 1000.0 + jnp.arange(archive_rows * variant.d, dtype=float)
    archive = pop.archive.at[:archive_rows].set(
        rows.reshape(archive_rows, variant.d)
    )
    pop = eqx.tree_at(
        lambda p: (p.cost, p.archive, p.archive_count),
        pop,
        (jnp.asarray(cost, dtype=float), archive, jnp.array(archive_rows)),
    )
    return pop, private


def test_population_size_follows_each_variants_nlpsr():
    # round(NP_max + (NP_min - NP_max) * r ** (1 - r)); paper 30 D -> 4,
    # MetaBox 170 -> 30. At r = 0.5: 300 - 296 * 0.7071 = 90.7 and
    # 170 - 140 * 0.7071 = 71.0.
    paper, metabox = NLSHADERSP(d=10), NLSHADERSPMetaBox(d=10)
    budget = 10000
    sizes = [int(paper.population_size(f, budget)) for f in (0, 5000, budget)]
    assert sizes == [300, 91, 4]
    sizes = [
        int(metabox.population_size(f, budget)) for f in (0, 5000, budget)
    ]
    assert sizes == [170, 71, 30]


def test_paper_reduction_keeps_the_best_and_a_random_archive_subset():
    variant = NLSHADERSP(d=2, n_max=10)
    cost = jnp.arange(10.0)[::-1]  # slot 9 is the best
    pop, private = _evaluated(variant, cost, archive_rows=21)
    no_success = jnp.full(10, jnp.inf)
    pop, _ = _generation(variant, pop, private, no_success, 1000, 1000)
    # NP_min = 4 survive: the four best, sorted.
    assert int(pop.size) == 4
    assert pop.cost[:4].tolist() == [0.0, 1.0, 2.0, 3.0]
    # N_A = round(2.1 * 4) = 8 archive rows, a subset of the old ones.
    assert int(pop.archive_capacity) == 8
    assert int(pop.archive_count) == 8
    kept = {tuple(r) for r in pop.archive[:8].tolist()}
    old = {(1000.0 + 2 * i, 1001.0 + 2 * i) for i in range(21)}
    assert kept <= old
    assert jnp.isnan(pop.archive[8:]).all()


def test_metabox_reduction_cuts_the_archive_to_its_leading_rows():
    variant = NLSHADERSPMetaBox(d=2, n_max=10, n_min=4)
    pop, private = _evaluated(variant, jnp.arange(10.0), archive_rows=15)
    no_success = jnp.full(10, jnp.inf)
    pop, _ = _generation(variant, pop, private, no_success, 1000, 1000)
    assert int(pop.size) == 4
    # A = int(max(2.1 * 4, N_min)) = 8 < 15: archive[:8], NA = 8.
    assert int(pop.archive_capacity) == 8
    assert pop.archive[:8, 0].tolist() == [1000.0 + 2 * i for i in range(8)]


def test_metabox_reduction_leaves_a_smaller_archive_and_its_capacity():
    variant = NLSHADERSPMetaBox(d=2, n_max=10, n_min=4)
    pop, private = _evaluated(variant, jnp.arange(10.0), archive_rows=5)
    capacity = int(pop.archive_capacity)
    no_success = jnp.full(10, jnp.inf)
    pop, _ = _generation(variant, pop, private, no_success, 1000, 1000)
    # A = 8 is not below the 5 rows held: NA keeps its old value.
    assert int(pop.archive_capacity) == capacity == 21
    assert int(pop.archive_count) == 5


# =============================================================================
#                                                       Selection and archive


def _rows(a):
    return {tuple(r) for r in jnp.asarray(a).tolist()}


def _told(variant, pop, private, trial_cost):
    """ask, then tell ``trial_cost``, early in a long run (no shrink)."""
    progress = DEProgress(jnp.asarray(0), jnp.asarray(10**9))
    trials, aux = variant.ask(jr.key(3), pop, private, progress)
    pop, private = variant.tell(
        pop, private, trials, jnp.asarray(trial_cost), aux, progress
    )
    return trials, pop, private


def test_paper_selection_archives_defeated_parents_and_replaces_on_ties():
    variant = NLSHADERSP(d=2, n_max=6)
    pop0, private = _evaluated(variant, [5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    inf = jnp.inf
    trials, pop, _ = _told(
        variant, pop0, private, [3.0, 4.0, inf, inf, inf, inf]
    )
    # Eq. (5): replace when f(u) <= f(x); Alg. 1 l.38-39: only a strict
    # improvement sends the parent to the archive.
    assert _rows(trials[:2]) <= _rows(pop.x)
    assert not _rows(pop0.x[:2]) & _rows(pop.x)
    assert int(pop.archive_count) == 1
    assert pop.archive[0].tolist() == pop0.x[0].tolist()
    assert sorted(pop.cost.tolist()) == [0.0, 1.0, 2.0, 3.0, 3.0, 4.0]


def test_metabox_selection_archives_the_leading_rows_and_keeps_ties():
    variant = NLSHADERSPMetaBox(d=2, n_max=6, n_min=4)
    pop0, private = _evaluated(variant, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    inf = jnp.inf
    trials, pop, _ = _told(
        variant, pop0, private, [inf, inf, inf, 3.0, 1.5, 2.5]
    )
    # Strict: the tie at slot 3 keeps its parent.
    assert pop0.x[3].tolist() in pop.x.tolist()
    assert _rows(trials[4:]) <= _rows(pop.x)
    # ``update_archive(i) for i in range(len(optim))``: two successes
    # archive the rows at positions 0 and 1 of the sorted population --
    # the two best parents, which were not replaced.
    assert int(pop.archive_count) == 2
    assert _rows(pop.archive[:2]) == _rows(pop0.x[:2])


# =============================================================================
#                                                      Success-history memory


def _tell_with(variant, pop, private, trial_cost, f, cr, **extra):
    """tell with fixed per-slot F, Cr and draws, early in a long run."""
    n = pop.x.shape[0]
    aux = {
        "f": jnp.asarray(f, dtype=float),
        "cr": jnp.asarray(cr, dtype=float),
        "u_insert": jnp.zeros(n),
        "from_archive": jnp.zeros(n, bool),
        **extra,
    }
    progress = DEProgress(jnp.asarray(0), jnp.asarray(10**9))
    return variant.tell(
        pop, private, pop.x + 1.0, jnp.asarray(trial_cost), aux, progress
    )


def _paper_extra(variant, pop, from_archive=None):
    n = pop.x.shape[0]
    return {
        "from_archive": (
            jnp.zeros(n, bool)
            if from_archive is None
            else jnp.asarray(from_archive)
        ),
        "u_resize": jnp.zeros(pop.archive.shape[0]),
    }


_F = [0.6, 0.8, 0.5, 0.5, 0.5, 0.5]
_CR = [0.3, 0.5, 0.5, 0.5, 0.5, 0.5]


def test_paper_memory_blends_the_weighted_lehmer_mean_half_and_half():
    variant = NLSHADERSP(d=2, n_max=6, memory_size=3)
    pop, private = _evaluated(variant, [5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    inf = jnp.inf
    pop, _ = _tell_with(
        variant,
        pop,
        private,
        [3.0, 3.0, inf, inf, inf, inf],
        _F,
        _CR,
        **_paper_extra(variant, pop),
    )
    # Improvements 2 and 1 -> w = (2/3, 1/3). Lehmer means: F 0.68,
    # Cr 0.390909; eq. (8)-(9) with c = 0.5 against the old 0.2.
    assert jnp.allclose(pop.memory_f, jnp.array([0.44, 0.2, 0.2]))
    assert jnp.allclose(pop.memory_cr, jnp.array([0.2954545, 0.2, 0.2]))
    assert int(pop.memory_index) == 1


def test_paper_memory_index_advances_without_success():
    variant = NLSHADERSP(d=2, n_max=6, memory_size=3)
    pop, private = _evaluated(variant, [5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    pop, _ = _tell_with(
        variant,
        pop,
        private,
        jnp.full(6, jnp.inf),
        _F,
        _CR,
        **_paper_extra(variant, pop),
    )
    # Alg. 1 l.51: k = mod(k, H) + 1 every generation.
    assert jnp.allclose(pop.memory_f, 0.2)
    assert int(pop.memory_index) == 1


def test_metabox_memory_uses_relative_weights_and_its_fallbacks():
    variant = NLSHADERSPMetaBox(d=2, n_max=6, n_min=4, memory_size=3)
    pop, private = _evaluated(variant, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    inf = jnp.inf
    crb = [-0.1] * 6  # a binomial generation records Crb, not Cr
    told, _ = _tell_with(
        variant,
        pop,
        private,
        [inf, inf, inf, inf, 2.0, 1.0],
        [0.5, 0.5, 0.5, 0.5, 0.6, 0.8],
        crb,
    )
    # df = (old - new) / (old + 1e-9): 0.5 and 0.8. Lehmer F = 0.736170;
    # sum(w Cr) <= 1e-6, so mean_wL gives 0.5. No blending.
    assert jnp.allclose(told.memory_f, jnp.array([0.736170, 0.2, 0.2]))
    assert jnp.allclose(told.memory_cr, jnp.array([0.5, 0.2, 0.2]))
    assert int(told.memory_index) == 1
    # No success: 0.5 written at k, and k stays.
    told, _ = _tell_with(variant, pop, private, jnp.full(6, inf), _F, _CR)
    assert jnp.allclose(told.memory_f, jnp.array([0.5, 0.2, 0.2]))
    assert jnp.allclose(told.memory_cr, jnp.array([0.5, 0.2, 0.2]))
    assert int(told.memory_index) == 0


# =============================================================================
#                                                               Private state


def test_paper_private_state_carries_the_adapted_archive_probability():
    variant = NLSHADERSP(d=2, n_max=6)
    pop, private = _evaluated(variant, [5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    assert float(private.p_archive) == 0.5
    inf = jnp.inf
    _, private = _tell_with(
        variant,
        pop,
        private,
        [3.0, 3.0, inf, inf, inf, inf],
        _F,
        _CR,
        **_paper_extra(variant, pop, [True, False, True] + [False] * 3),
    )
    # n_A = 2 offspring built with the archive (one improved by 2), four
    # without (one improved by 1): (2 / 2) / (2 / 2 + 1 / 4) = 0.8.
    assert jnp.allclose(private.p_archive, 0.8)


def test_metabox_private_state_carries_pa_within_a_turn():
    # MetaBox resets pa to 0.5 at each step() call (one RL-DAS turn) and
    # adapts it after every generation of that turn with update_Pa.
    variant = NLSHADERSPMetaBox(d=2, n_max=6, n_min=4)
    pop, private = _evaluated(variant, [5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    assert float(private.p_archive) == 0.5
    inf = jnp.inf
    _, private = _tell_with(
        variant,
        pop,
        private,
        [3.0, 3.0, inf, inf, inf, inf],
        _F,
        _CR,
        from_archive=jnp.array([True, False, True] + [False] * 3),
    )
    # Relative gains (f(x) - f(u)) / (f(x) + 1e-9): slot 0 0.4 (archive),
    # slot 1 0.25 (population); slot 2 used the archive but failed, so
    # na = 1 counts successful archive users only. MetaBox's swap: fa =
    # 0.25, fp = 0.4, NP = 6 -> 0.25 / (0.25 + 0.4 / 5) = 0.7576.
    assert jnp.allclose(private.p_archive, 0.25 / 0.33, rtol=1e-5)
    # A fresh turn starts from the variant's initial state again.
    _, fresh = variant.init(jr.key(1), pop.x)
    assert float(fresh.p_archive) == 0.5


# =============================================================================
#                                                                      Donors


def test_draw_donors_excludes_duplicates_and_favours_good_r2():
    # Six slots, slot 5 dead; slot 0 the best. Two valid archive rows.
    pop = _population(
        [0.0, 1.0, 2.0, 3.0, 4.0, 0.0],
        live=[True] * 5 + [False],
        archive_max=4,
    )
    pop = eqx.tree_at(lambda p: p.archive_count, pop, jnp.array(2))
    from_archive = jnp.array([False, False, False, False, True, False])
    keys = jr.split(jr.key(7), 4000)
    pbest, r1, r2 = jax.vmap(lambda k: draw_donors(k, pop, 2, from_archive))(
        keys
    )
    i = jnp.arange(6)[None, :]
    live_rows = slice(0, 5)
    # pbest among the two best, never i; all three distinct from i.
    assert jnp.isin(pbest[:, live_rows], jnp.array([0, 1])).all()
    for a, b in [(pbest, i), (r1, i), (r1, pbest), (r2, i), (r2, pbest)]:
        b = jnp.broadcast_to(b, a.shape)
        assert (a[:, live_rows] != b[:, live_rows]).all()
    assert (r2[:, :4] != r1[:, :4]).all()
    # Population r2 never a dead slot; archive r2 is a valid archive row.
    assert (r2[:, :4] < 5).all()
    assert jnp.isin(r2[:, 4], jnp.array([6, 7])).all()
    # Rank pressure. For slot 3, slots 2 and 4 are never pbest and are
    # r1 equally often, so only their weights set how often each is r2:
    # exp(-3 / 5) / exp(-5 / 5) = 1.49 (uniform would give 1).
    counts = jnp.bincount(r2[:, 3], length=6)
    assert counts[2] > 1.25 * counts[4]


# =============================================================================
#                                                                    Registry

_VARIANTS = {
    "nl_shade_rsp": NLSHADERSP,
    "nl_shade_rsp_metabox": NLSHADERSPMetaBox,
}


def _build(name, problem, bounded=(None, None), **hyperparameters):
    return optimizer_mapping(name)(
        **problem,
        opt_hash=0,
        bounded=bounded,
        stop_fn=None,
        **hyperparameters,
    )


@pytest.mark.parametrize("name", sorted(_VARIANTS))
def test_registry_builds_without_a_budget_and_refuses_to_run(name):
    problem = sphere_problem(2)
    update_class = _build(name, problem)
    assert update_class.popsize == _VARIANTS[name](d=2).n_max
    params = jnp.zeros((update_class.popsize, 2))
    with pytest.raises(ValueError, match="max_fevals"):
        update_class.init_fn(params, jr.key(0))


@pytest.mark.parametrize("name", sorted(_VARIANTS))
def test_registry_steps_bill_only_evaluated_rows_within_budget(name):
    problem = sphere_problem(2)
    budget = 3000
    # A half-open box: out-of-box components must stay finite (clipped
    # onto the finite side), never be redrawn towards infinity.
    update_class = _build(
        name, problem, bounded=(0.0, None), max_fevals=budget
    )
    n_steps = _VARIANTS[name](d=2).generations_for_budget(budget)
    params = jr.uniform(jr.key(1), (update_class.popsize, 2))
    state = update_class.init_fn(params, jr.key(2))
    carry = (params, state, jr.key(3))
    billed, padded = [], []
    for _ in range(n_steps):
        carry, history = update_class.step_fn(carry, {})
        rows = ~jnp.isnan(history.params).all(axis=1)
        assert int(history.fevals) == int(rows.sum())
        assert jnp.isnan(history.loss[~rows]).all()
        x, loss = np.asarray(history.params), np.asarray(history.loss)
        live = ~np.isnan(x).all(axis=1)
        assert np.isfinite(x[live]).all() and (x[live] >= 0.0).all()
        assert np.isfinite(loss[live]).all()
        billed.append(int(history.fevals))
        padded.append(int((~rows).sum()))
    assert sum(billed) <= budget
    # The population shrank: later steps NaN-pad the freed slots.
    assert billed[0] == update_class.popsize and padded[-1] > 0
    # And the run made progress on the sphere.
    assert float(jnp.nanmin(history.loss)) < 0.01


@pytest.mark.parametrize("name", sorted(_VARIANTS))
def test_realization_result_does_not_depend_on_batch_size(name):
    # l2co-optimizers#24, for the new entries.
    problem = rosenbrock_problem(2)
    kwargs = dict(max_fevals=5000)
    alone = realization_best_losses(name, problem, 1, 6, **kwargs)
    batched = realization_best_losses(name, problem, 3, 6, **kwargs)
    assert jnp.array_equal(alone[0], batched[0])


# =============================================================================
#                                              Parity with MetaBox (slow)

_FIXTURE = Path(__file__).parent / "data" / "nl_shade_rsp_metabox_fixture.json"

#: MetaBox floors errors below 1e-8 to 0 (``NL_SHADE_RSP.error``) and stops
#: there; both sides are compared above that floor.
_FLOOR = 1e-8


def _sphere(x):
    return jnp.sum(x**2)


def _rastrigin(x):
    return 10.0 * x.size + jnp.sum(x**2 - 10.0 * jnp.cos(2 * jnp.pi * x))


def _rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


_FUNCTIONS = {
    "sphere": _sphere,
    "rastrigin": _rastrigin,
    "rosenbrock": _rosenbrock,
}


def _best_costs(name, fn, seeds, fixture, bounded=(None, None)):
    """Best loss of a whole run per seed, as the fixture's runs."""
    d, budget = fixture["dim"], fixture["max_fevals"]
    low, high = fixture["bounds"]
    with jax.enable_x64(True):
        update_class = optimizer_mapping(name)(
            model=jnp.zeros(d),
            loss_fn=fn,
            pass_rng=False,
            opt_hash=0,
            bounded=bounded,
            stop_fn=None,
            max_fevals=budget,
        )
        n_steps = _VARIANTS[name](d=d).generations_for_budget(budget)

        def run(seed):
            key = jr.key(seed)
            x0 = jr.uniform(
                key, (update_class.popsize, d), minval=low, maxval=high
            )
            state = update_class.init_fn(x0, key)

            def step(carry, _):
                carry, history = update_class.step_fn(carry, {})
                return carry, jnp.nanmin(history.loss)

            _, best = jax.lax.scan(step, (x0, state, key), None, n_steps)
            return jnp.min(best)

        best = jax.jit(jax.vmap(run))(jnp.asarray(seeds))
    return np.maximum(np.asarray(best), _FLOOR)


@pytest.mark.slow
@pytest.mark.parametrize("function", sorted(_FUNCTIONS))
def test_metabox_variant_matches_vendored_metabox_statistically(function):
    fixture = json.loads(_FIXTURE.read_text())
    reference = np.maximum(fixture["functions"][function]["best"], _FLOOR)
    ours = _best_costs(
        "nl_shade_rsp_metabox",
        _FUNCTIONS[function],
        fixture["seeds"],
        fixture,
    )
    # Same distribution of best costs over 20 seeds: a two-sided
    # Mann-Whitney U test must not reject at the 1% level.
    if np.all(ours == _FLOOR) and np.all(reference == _FLOOR):
        return  # both solve every run
    assert mannwhitneyu(ours, reference).pvalue > 0.01, (
        np.median(ours),
        np.median(reference),
    )


#: The paper variant's median best cost must be at most this many times
#: MetaBox's (above the floor). A sanity bound, not a parity claim: the
#: paper's 30 D starting population spends a 20000 budget differently.
_PAPER_FACTOR = 10.0


@pytest.mark.slow
@pytest.mark.parametrize("function", sorted(_FUNCTIONS))
def test_paper_variant_lands_near_metabox(function):
    fixture = json.loads(_FIXTURE.read_text())
    reference = max(fixture["functions"][function]["median"], _FLOOR)
    ours = _best_costs(
        "nl_shade_rsp",
        _FUNCTIONS[function],
        list(range(5)),
        fixture,
        bounded=tuple(fixture["bounds"]),
    )
    assert np.median(ours) <= _PAPER_FACTOR * reference, (
        np.median(ours),
        reference,
    )
