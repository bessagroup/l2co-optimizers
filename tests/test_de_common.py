"""Tests for the shared differential-evolution foundation.

Three seams, each checked against hand-computed values: the
population-size schedules and ``generations_for_budget``; the shared
operators, fed fixed random draws; and the registry adapter, driven with
a test-only toy variant.
"""

#                                                                       Modules
# =============================================================================

# Standard
from functools import partial

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# Local
from l2co_optimizers._src import de_common as dc

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


# =============================================================================
#                                                     Population-size schedules


@pytest.mark.parametrize(
    ("fevals", "expected"),
    [(0, 170), (5000, 121), (10000, 71), (20000, 30), (25000, 30)],
)
def test_nonlinear_schedule_matches_metabox_formula(fevals, expected):
    # round(170 + (30 - 170) * p ** (1 - p)), p = fevals / 20000:
    # p = 0.25 -> 170 - 140 * 0.35355 = 120.50 -> 121;
    # p = 0.5  -> 170 - 140 * 0.70711 = 71.005 -> 71.
    n = dc.nonlinear_population_size(fevals, 20000, n_max=170, n_min=30)
    assert int(n) == expected


@pytest.mark.parametrize(
    ("fevals", "expected"),
    [(0, 100), (333, 68), (500, 52), (1000, 4), (1200, 4)],
)
def test_linear_schedule_matches_lshade_formula(fevals, expected):
    # round((4 - 100) / 1000 * fevals + 100): 333 -> 68.03 -> 68.
    n = dc.linear_population_size(fevals, 1000, n_max=100, n_min=4)
    assert int(n) == expected


@pytest.mark.parametrize(
    ("fevals", "expected"),
    [
        (0, 410),
        (2499, 410),
        (2500, 210),
        (5000, 110),
        (7500, 60),
        (9999, 60),
        (10000, 60),
    ],
)
def test_j21_schedule_halves_big_population_at_quarters(fevals, expected):
    # j21 paper's example: big population 400 -> 200, 100, 50 at
    # MaxFEs/4, 2/4, 3/4; the small population (10) is never reduced.
    n = dc.j21_population_size(fevals, 10000, n_big=400, n_small=10)
    assert int(n) == expected


# =============================================================================
#                                                         Budget -> generations


@pytest.mark.parametrize(("budget", "expected"), [(50, 7), (48, 6)])
def test_generations_for_budget_never_exceeds_budget(budget, expected):
    # Linear 10 -> 4 over MaxFEs = budget. The starting population is
    # billed twice (its evaluation, then the first generation; the
    # reduction runs at the end of a generation). Bills worked by hand,
    # budget 50: 10, 10, round(7.6) = 8, round(6.64) = 7, round(5.8) = 6,
    # round(5.08) = 5, round(4.48) = 4 (cumulative 50); the next bill (4)
    # would reach 54. Budget 48: 10, 10, round(7.5) = 8 (half to even, as
    # numpy), round(6.5) = 6, round(5.75) = 6, 5 (cumulative 45); the
    # next, round(4.375) = 4, would reach 49.
    schedule = partial(dc.linear_population_size, n_max=10, n_min=4)
    steps = dc.generations_for_budget(
        dc.bill_live_population(schedule), budget
    )
    assert steps == expected


# =============================================================================
#                                                  Parameter adaptation (SHADE)


def _population(n_max=4, d=2, memory_size=3, archive_max=5, **kwargs):
    return dc.init_population(
        jr.key(0),
        (-5.0, 5.0),
        n_max=n_max,
        d=d,
        memory_size=memory_size,
        archive_max=archive_max,
        **kwargs,
    )


def _with_successes(pop, f, cr, weight):
    n = len(f)
    mask = jnp.arange(pop.x.shape[0]) < n
    pad = pop.x.shape[0] - n
    return dc.record_successes(
        pop,
        jnp.asarray(list(f) + [0.0] * pad),
        jnp.asarray(list(cr) + [0.0] * pad),
        jnp.asarray(list(weight) + [0.0] * pad),
        mask,
    )


def test_record_successes_appends_only_masked_entries():
    pop = _population(n_max=4)
    pop = dc.record_successes(
        pop,
        jnp.array([0.1, 0.2, 0.3, 0.4]),
        jnp.array([0.5, 0.6, 0.7, 0.8]),
        jnp.array([1.0, 2.0, 3.0, 4.0]),
        jnp.array([False, True, False, True]),
    )
    pop = dc.record_successes(
        pop,
        jnp.array([0.9, 0.0, 0.0, 0.0]),
        jnp.array([0.3, 0.0, 0.0, 0.0]),
        jnp.array([5.0, 0.0, 0.0, 0.0]),
        jnp.array([True, False, False, False]),
    )
    assert int(pop.success_count) == 3
    np.testing.assert_allclose(pop.success_f[:3], [0.2, 0.4, 0.9])
    np.testing.assert_allclose(pop.success_cr[:3], [0.6, 0.8, 0.3])
    np.testing.assert_allclose(pop.success_weight[:3], [2.0, 4.0, 5.0])
    assert jnp.all(jnp.isnan(pop.success_f[3:]))


def test_memory_update_writes_weighted_lehmer_means():
    # S_F = [0.2, 0.6], S_Cr = [0.5, 0.9], improvements [1, 3] -> w =
    # [0.25, 0.75]. Lehmer F: (0.01 + 0.27) / (0.05 + 0.45) = 0.56.
    # Lehmer Cr: (0.0625 + 0.6075) / (0.125 + 0.675) = 0.8375.
    pop = _with_successes(
        _population(memory_f=0.2, memory_cr=0.2),
        [0.2, 0.6],
        [0.5, 0.9],
        [1.0, 3.0],
    )
    new = dc.update_memory(pop)
    np.testing.assert_allclose(new.memory_f, [0.56, 0.2, 0.2], rtol=1e-6)
    np.testing.assert_allclose(new.memory_cr, [0.8375, 0.2, 0.2], rtol=1e-6)
    assert int(new.memory_index) == 1
    assert int(new.success_count) == 0
    assert jnp.all(jnp.isnan(new.success_f))


def test_memory_update_arithmetic_cr_and_blend_with_old_entry():
    # Arithmetic Cr: 0.25 * 0.5 + 0.75 * 0.9 = 0.8. Blending half the
    # old entry (jSO style): F (0.56 + 0.2) / 2 = 0.38, Cr
    # (0.8 + 0.2) / 2 = 0.5.
    pop = _with_successes(
        _population(memory_f=0.2, memory_cr=0.2),
        [0.2, 0.6],
        [0.5, 0.9],
        [1.0, 3.0],
    )
    new = dc.update_memory(pop, cr_mean="arithmetic", old_weight=0.5)
    np.testing.assert_allclose(new.memory_f[0], 0.38, rtol=1e-6)
    np.testing.assert_allclose(new.memory_cr[0], 0.5, rtol=1e-6)


def test_memory_update_write_index_wraps():
    pop = _with_successes(_population(), [0.5], [0.5], [1.0])
    pop = eqx.tree_at(lambda p: p.memory_index, pop, jnp.array(2))
    assert int(dc.update_memory(pop).memory_index) == 0


def test_memory_update_without_successes():
    pop = _population(memory_f=0.2, memory_cr=0.3)
    pop = eqx.tree_at(lambda p: p.memory_index, pop, jnp.array(1))
    # Paper SHADE: no success, no update.
    kept = dc.update_memory(pop)
    np.testing.assert_allclose(kept.memory_f, [0.2, 0.2, 0.2])
    assert int(kept.memory_index) == 1
    # MetaBox: writes 0.5 at k and does not advance k.
    reset = dc.update_memory(pop, on_empty=0.5)
    np.testing.assert_allclose(reset.memory_f, [0.2, 0.5, 0.2])
    np.testing.assert_allclose(reset.memory_cr, [0.3, 0.5, 0.3])
    assert int(reset.memory_index) == 1


def test_memory_update_degenerate_mean_fallback():
    # Every successful Cr is 0: sum(w * s) = 0. By default the entry is
    # kept; MetaBox's mean_wL returns 0.5 when sum(w * s) <= 1e-6.
    pop = _with_successes(
        _population(memory_f=0.2, memory_cr=0.3), [0.4], [0.0], [1.0]
    )
    kept = dc.update_memory(pop)
    np.testing.assert_allclose(kept.memory_f[0], 0.4, rtol=1e-6)
    np.testing.assert_allclose(kept.memory_cr[0], 0.3)
    metabox = dc.update_memory(
        pop, small_sum_threshold=1e-6, small_sum_value=0.5
    )
    np.testing.assert_allclose(metabox.memory_cr[0], 0.5)


def test_f_cr_from_draws_reflects_negative_scale_factors():
    pop = _population(memory_size=2)
    pop = eqx.tree_at(
        lambda p: (p.memory_f, p.memory_cr),
        pop,
        (jnp.array([0.5, 0.3]), jnp.array([0.6, 0.9])),
    )
    # u = 0.75 -> tan(pi / 4) = 1; u = 0.25 -> -1; u = 0.5 + atan(-5) / pi
    # -> -5, so F = 0.3 - 0.5 = -0.2, reflected to 2 * 0.3 + 0.2 = 0.8.
    # u = 0.999 -> F far above 1, clipped to 1.
    slot = jnp.array([0, 1, 1, 0])
    u = jnp.array([0.75, 0.25, 0.06283295818900114, 0.999])
    z = jnp.array([1.0, 2.0, -20.0, 0.0])
    f, cr = dc.f_cr_from_draws(pop, slot, z, u, negative_f="reflect")
    np.testing.assert_allclose(f, [0.6, 0.2, 0.8, 1.0], rtol=1e-5)
    np.testing.assert_allclose(cr, [0.7, 1.0, 0.0, 0.6], rtol=1e-6)


def test_f_cr_from_draws_truncates_cauchy_at_zero():
    # The median (u = 0.5) of Cauchy(0.3, 0.1) truncated to (0, inf) is
    # sqrt(0.3 ** 2 + 0.1 ** 2) = 0.316228.
    pop = _population(memory_size=1, memory_f=0.3)
    f, _ = dc.f_cr_from_draws(
        pop, jnp.array([0]), jnp.array([0.0]), jnp.array([0.5])
    )
    np.testing.assert_allclose(f, [0.31622777], rtol=1e-5)


# =============================================================================
#                                                       Mutation and crossover


def test_binomial_crossover_takes_mutant_below_cr_and_at_jrand():
    x = jnp.zeros((2, 3))
    v = jnp.ones((2, 3))
    u = jnp.array([[0.1, 0.6, 0.7], [0.1, 0.1, 0.1]])
    trial = dc.binomial_crossover(
        x, v, jnp.array([0.5, 0.0]), u, jnp.array([2, 2])
    )
    np.testing.assert_array_equal(trial, [[1, 0, 1], [0, 0, 1]])


def test_exponential_crossover_copies_a_cyclic_run_from_start():
    # u[:, k] is the draw that decides whether the run continues to the
    # k-th position after ``start`` (k = 0 is always copied).
    # Row 0: start 2, Cr 0.5: positions 2 (always), 3 (0.3 < 0.5), then
    # 0.7 stops the run -> [0, 0, 1, 1]. Row 1: Cr 1 copies all four,
    # wrapping round from 3. Row 2: Cr 0 copies only the start.
    x = jnp.zeros((3, 4))
    v = jnp.ones((3, 4))
    u = jnp.array(
        [[0.9, 0.3, 0.7, 0.1], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    trial = dc.exponential_crossover(
        x, v, jnp.array([0.5, 1.0, 0.0]), jnp.array([2, 3, 0]), u
    )
    np.testing.assert_array_equal(
        trial, [[0, 0, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0]]
    )


def test_current_to_pbest_1_with_archive_donor():
    # v_i = x_i + F_i (x_pbest - x_i) + F_i (x_r1 - z_r2), z = pop U archive.
    # i=0: (0,0) + 0.5 (1,0) + 0.5 ((0,2) - (4,4)) = (-1.5, -1).
    # i=1: (1,0) + ((0,0) - (1,0)) + ((0,2) - (0,0)) = (0, 2).
    # i=2: (0,2) + 0.5 ((0,0) - (0,2)) + 0.5 ((1,0) - (4,4)) = (-1.5, -1).
    pop = _population(n_max=3, d=2, archive_max=2)
    pop = eqx.tree_at(
        lambda p: (p.x, p.archive, p.archive_count),
        pop,
        (
            jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]),
            jnp.array([[4.0, 4.0], [jnp.nan, jnp.nan]]),
            jnp.array(1),
        ),
    )
    v = dc.current_to_pbest_1(
        pop,
        jnp.array([0.5, 1.0, 0.5]),
        pbest=jnp.array([1, 0, 0]),
        r1=jnp.array([2, 2, 1]),
        r2=jnp.array([3, 0, 3]),
    )
    np.testing.assert_allclose(v, [[-1.5, -1.0], [0.0, 2.0], [-1.5, -1.0]])


def test_sample_index_picks_only_valid_weighted_columns():
    valid = jnp.array([[True, True, False], [False, False, False]])
    weights = jnp.array([0.0, 1.0, 5.0])
    picks = jax.vmap(
        lambda k: dc.sample_index(
            k, valid, weights=weights, fallback=jnp.array([0, 2])
        )
    )(jr.split(jr.key(3), 50))
    assert jnp.all(picks[:, 0] == 1)
    assert jnp.all(picks[:, 1] == 2)


def _with_live(pop, cost, live):
    return eqx.tree_at(
        lambda p: (p.cost, p.live),
        pop,
        (jnp.asarray(cost, dtype=float), jnp.asarray(live)),
    )


def test_sample_current_to_pbest_live_population_only():
    # Three live slots of five, no archive: r1 and r2 must be the two
    # other live individuals, so {i, r1, r2} = {0, 1, 2}. With
    # n_pbest = 1 pbest is the best live slot (slot 1; dead slot 4 has
    # a lower stale cost but is not live).
    pop = _with_live(
        _population(n_max=5),
        [3.0, 1.0, 2.0, jnp.inf, -9.0],
        [True, True, True, False, False],
    )
    pbest, r1, r2 = jax.vmap(
        lambda k: dc.sample_current_to_pbest(k, pop, n_pbest=1)
    )(jr.split(jr.key(4), 30))
    assert jnp.all(pbest[:, :3] == 1)
    for i in range(3):
        trio = jnp.stack([jnp.full(30, i), r1[:, i], r2[:, i]], axis=1)
        assert jnp.all(jnp.sort(trio, axis=1) == jnp.array([0, 1, 2]))


def test_sample_current_to_pbest_draws_r2_from_valid_archive_rows():
    # Two live slots of four, one valid archive row of three: r1 is the
    # other live slot and r2 can only be archive row 0 (union index 4).
    pop = _with_live(
        _population(n_max=4, archive_max=3),
        [1.0, 2.0, jnp.inf, jnp.inf],
        [True, True, False, False],
    )
    pop = eqx.tree_at(
        lambda p: (p.archive, p.archive_count),
        pop,
        (pop.archive.at[0].set(7.0), jnp.array(1)),
    )
    _, r1, r2 = jax.vmap(
        lambda k: dc.sample_current_to_pbest(k, pop, n_pbest=2)
    )(jr.split(jr.key(5), 30))
    assert jnp.all(r1[:, 0] == 1) and jnp.all(r1[:, 1] == 0)
    assert jnp.all(r2[:, :2] == 4)


def test_sample_current_to_pbest_can_exclude_pbest():
    # MetaBox also keeps pbest out of r1 and r2 and away from i. Four
    # live slots, n_pbest = 2 (slots 0, 1 best): {i, pbest, r1, r2} is
    # all four slots.
    pop = _with_live(_population(n_max=4), [0.0, 1.0, 2.0, 3.0], [True] * 4)
    pbest, r1, r2 = jax.vmap(
        lambda k: dc.sample_current_to_pbest(
            k, pop, n_pbest=2, exclude_pbest=True, use_archive=False
        )
    )(jr.split(jr.key(6), 30))
    for i in range(4):
        group = jnp.stack(
            [jnp.full(30, i), pbest[:, i], r1[:, i], r2[:, i]], axis=1
        )
        assert jnp.all(jnp.sort(group, axis=1) == jnp.arange(4))
        assert jnp.all((pbest[:, i] == 0) | (pbest[:, i] == 1))


# =============================================================================
#                                              Archive, ordering and shrinking


def test_insert_archive_appends_then_replaces_at_drawn_rows():
    # Capacity 2 holding [A]. R0 fills row 1; the archive is then full,
    # so R1 replaces row floor(0.6 * 2) = 1 and R2 row floor(0.2 * 2) = 0.
    # The unmasked fourth row is never inserted.
    pop = _population(n_max=4, d=2, archive_max=3, archive_capacity=2)
    pop = eqx.tree_at(
        lambda p: (p.archive, p.archive_count),
        pop,
        (pop.archive.at[0].set(-1.0), jnp.array(1)),
    )
    rows = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [9.0, 9.0]])
    new = dc.insert_archive(
        pop,
        rows,
        jnp.array([True, True, True, False]),
        jnp.array([0.9, 0.6, 0.2, 0.0]),
    )
    np.testing.assert_array_equal(
        new.archive, [[3.0, 3.0], [2.0, 2.0], [jnp.nan, jnp.nan]]
    )
    assert int(new.archive_count) == 2


def test_resize_archive_keeps_leading_rows_or_a_random_subset():
    pop = _population(n_max=4, d=1, archive_max=3)
    pop = eqx.tree_at(
        lambda p: (p.archive, p.archive_count),
        pop,
        (jnp.array([[1.0], [2.0], [3.0]]), jnp.array(3)),
    )
    first = dc.resize_archive(pop, 2)
    np.testing.assert_array_equal(first.archive[:, 0], [1.0, 2.0, jnp.nan])
    assert int(first.archive_count) == 2
    assert int(first.archive_capacity) == 2
    grown = dc.resize_archive(first, 3)
    np.testing.assert_array_equal(grown.archive[:, 0], [1.0, 2.0, jnp.nan])
    assert int(grown.archive_count) == 2
    assert int(grown.archive_capacity) == 3
    sampled = dc.resize_archive(pop, 2, key=jr.key(7))
    kept = np.asarray(sampled.archive[:2, 0])
    assert len(set(kept)) == 2 and set(kept) <= {1.0, 2.0, 3.0}
    assert np.isnan(sampled.archive[2, 0])


def _ranked_population():
    # Live slots 0-3 with costs [3, 1, 4, 2]; slot 4 dead. Row value =
    # slot index, F = slot / 10, so a move is visible in every field.
    pop = _population(n_max=5, d=1)
    return eqx.tree_at(
        lambda p: (p.x, p.cost, p.live, p.f, p.cr),
        pop,
        (
            jnp.array([[0.0], [1.0], [2.0], [3.0], [jnp.nan]]),
            jnp.array([3.0, 1.0, 4.0, 2.0, jnp.inf]),
            jnp.array([True, True, True, True, False]),
            jnp.array([0.0, 0.1, 0.2, 0.3, jnp.nan]),
            jnp.array([0.5, 0.6, 0.7, 0.8, jnp.nan]),
        ),
    )


def test_sort_by_cost_puts_live_individuals_first_in_cost_order():
    new = dc.sort_by_cost(_ranked_population())
    np.testing.assert_array_equal(new.x[:, 0], [1.0, 3.0, 0.0, 2.0, jnp.nan])
    np.testing.assert_array_equal(new.cost, [1.0, 2.0, 3.0, 4.0, jnp.inf])
    np.testing.assert_allclose(new.cr[:4], [0.6, 0.8, 0.5, 0.7])
    np.testing.assert_array_equal(new.live, [True] * 4 + [False])


def test_shrink_keeps_the_best_and_nan_pads_the_rest():
    new = dc.shrink(_ranked_population(), 2)
    np.testing.assert_array_equal(new.x[:, 0], [1.0, 3.0] + [jnp.nan] * 3)
    np.testing.assert_array_equal(new.cost, [1.0, 2.0] + [jnp.inf] * 3)
    np.testing.assert_allclose(new.f, [0.1, 0.3] + [jnp.nan] * 3)
    np.testing.assert_array_equal(new.live, [True, True, False, False, False])


def test_shrink_tail_keeps_the_last_live_slots():
    # MetaBox's JDE21 keeps group[-NP:], whatever the costs.
    pop = _ranked_population()
    new = dc.shrink(pop, 2, mode="tail")
    np.testing.assert_array_equal(new.x[:, 0], [2.0, 3.0] + [jnp.nan] * 3)
    np.testing.assert_array_equal(new.cost, [4.0, 2.0] + [jnp.inf] * 3)
    unchanged = dc.shrink(pop, 9, mode="tail")
    np.testing.assert_array_equal(unchanged.x, pop.x)


def test_halve_pairwise_keeps_the_winner_of_each_pair():
    # j21 halving (Brest and Maucec 2008) of the big population in
    # slots 0-3: pairs (0, 2) and (1, 3); costs 3 vs 4 -> slot 0,
    # 1 vs 2 -> slot 1. The rest (slot 4, the small population) moves
    # up behind the winners.
    pop = _ranked_population()
    pop = eqx.tree_at(
        lambda p: (p.x, p.cost, p.live),
        pop,
        (
            pop.x.at[4].set(4.0),
            pop.cost.at[4].set(0.5),
            pop.live.at[4].set(True),
        ),
    )
    new = dc.halve_pairwise(pop, 4)
    np.testing.assert_array_equal(
        new.x[:, 0], [0.0, 1.0, 4.0, jnp.nan, jnp.nan]
    )
    np.testing.assert_array_equal(new.cost, [3.0, 1.0, 0.5, jnp.inf, jnp.inf])
    np.testing.assert_array_equal(new.live, [True, True, True, False, False])


# =============================================================================
#                                                             Registry adapter


class _ToyDE(dc.AbstractDEVariant):
    """DE/current-to-pbest/1/bin, fixed F and Cr, linear 8 -> 4 reduction.

    Test-only: exercises the protocol, not any published variant.
    """

    n_max: int = 8
    n_min: int = 4

    def init(self, key, x):
        pop = dc.init_population(
            key,
            (-5.0, 5.0),
            n_max=self.n_max,
            d=x.shape[1],
            memory_size=1,
            archive_max=self.n_max,
            x=x,
        )
        return pop, dc.DEPrivateState()

    def population_size(self, fevals, max_fevals):
        return dc.linear_population_size(
            fevals, max_fevals, n_max=self.n_max, n_min=self.n_min
        )

    def ask(self, key, pop, private, progress):
        k_idx, k_u, k_j = jr.split(key, 3)
        pbest, r1, r2 = dc.sample_current_to_pbest(k_idx, pop, n_pbest=2)
        f = jnp.full(pop.f.shape, 0.5)
        v = dc.current_to_pbest_1(pop, f, pbest, r1, r2)
        u = jr.uniform(k_u, pop.x.shape)
        jrand = jr.randint(k_j, pop.f.shape, 0, pop.x.shape[1])
        trials = dc.binomial_crossover(
            pop.x, v, jnp.full(pop.f.shape, 0.9), u, jrand
        )
        return trials, jnp.array(0)

    def tell(self, pop, private, trials, trial_cost, aux, progress):
        better = pop.live & (trial_cost <= pop.cost)
        pop = eqx.tree_at(
            lambda p: (p.x, p.cost),
            pop,
            (
                jnp.where(better[:, None], trials, pop.x),
                jnp.where(better, trial_cost, pop.cost),
            ),
        )
        n_next = self.population_size(progress.fevals, progress.max_fevals)
        return dc.shrink(pop, n_next), private


def test_adapter_bills_only_live_individuals_and_nan_pads():
    # Linear 8 -> 4 over max_fevals = 40. Bills by hand: 8 (the starting
    # population), 8 (the first generation), round(6.4) = 6,
    # round(5.8) = 6, round(5.2) = 5, round(4.7) = 5; cumulative 38, and
    # the next bill, round(4.2) = 4, would exceed 40.
    from .toy_problems import sphere_problem

    variant = _ToyDE()
    assert variant.generations_for_budget(40) == 6

    problem = sphere_problem(2)
    update = dc.de_variant_update(
        **problem, opt_hash=7, variant=variant, max_fevals=40
    )
    assert update.popsize == 8
    params = jr.uniform(jr.key(8), (8, 2), minval=-1.0, maxval=1.0)
    state = update.init_fn(params, jr.key(9))
    carry = (params, state, jr.key(10))
    step = eqx.filter_jit(update.step_fn)
    bills = []
    for _ in range(6):
        carry, history = step(carry, {})
        n = int(history.fevals)
        bills.append(n)
        # The billed individuals are the leading slots; the rest of the
        # history row is NaN padding.
        assert jnp.all(jnp.isfinite(history.loss[:n]))
        assert jnp.all(jnp.isnan(history.loss[n:]))
        assert int(history.update_step) == 7
    assert bills == [8, 8, 6, 6, 5, 5]
    params, state, _ = carry
    # Reduced to round(4.2) = 4 after the last generation: four live
    # trials to evaluate next, NaN rows behind them.
    assert int(state.pop.size) == 4
    assert jnp.all(jnp.isfinite(params[:4])) and jnp.all(jnp.isnan(params[4:]))
    assert int(state.fevals) == 38
