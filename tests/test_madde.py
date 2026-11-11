"""MadDE: the paper variant (``madde``) and MetaBox's (``madde_metabox``).

Operator tests feed fixed draws and compare with values worked out by
hand from the paper's equations (Biswas et al., CEC 2021) and from the
vendored MetaBox code (``tests/reference/metabox_rldas``). The slow tests
compare whole runs with a stored fixture written by
``tests/reference/metabox_rldas/make_fixture_madde.py``.
"""

#                                                                       Modules
# =============================================================================

# Standard
import json
from pathlib import Path

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# Local
from l2co_optimizers import optimizer_mapping
from l2co_optimizers._src import de_common as dc
from l2co_optimizers._src import madde

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


#                                                                  Bound repair
# =============================================================================


def test_repair_bounds_moves_halfway_from_parent_to_violated_bound():
    # Paper eq. (9) with the run's box [-5, 5].
    x = jnp.array([[1.0, -2.0, 0.0]])
    v = jnp.array([[7.0, -9.0, 3.0]])
    got = madde.repair_bounds(v, x, -5.0, 5.0)
    np.testing.assert_allclose(got, [[3.0, -3.5, 3.0]])


def test_metabox_repair_uses_the_hard_coded_hundred_box():
    # MetaBox: v[v < -100] = (x - 100) / 2, v[v > 100] = (x + 100) / 2,
    # whatever the problem's bounds; inside +-100 nothing changes.
    variant = madde.madde_metabox_variant(d=3)
    x = jnp.array([[4.0, -4.0, 4.0]])
    v = jnp.array([[150.0, -120.0, 50.0]])
    got = madde.repair_bounds(v, x, variant.low, variant.high)
    np.testing.assert_allclose(got, [[52.0, -52.0, 50.0]])


#                                                                     Mutations
# =============================================================================


def _population(x, archive_rows=(), live=None, cost=None, archive_max=2):
    x = jnp.asarray(x, dtype=float)
    n, d = x.shape
    pop = dc.init_population(
        jr.key(0),
        (0.0, 1.0),
        n_max=n,
        d=d,
        memory_size=2,
        archive_max=archive_max,
        x=x,
    )
    archive = pop.archive
    for i, row in enumerate(archive_rows):
        archive = archive.at[i].set(jnp.asarray(row, dtype=float))
    live = jnp.ones(n, bool) if live is None else jnp.asarray(live)
    cost = jnp.zeros(n) if cost is None else jnp.asarray(cost, dtype=float)
    return eqx.tree_at(
        lambda p: (p.archive, p.archive_count, p.live, p.cost),
        pop,
        (
            archive,
            jnp.array(len(archive_rows), dtype=jnp.int32),
            live,
            jnp.where(live, cost, jnp.inf),
        ),
    )


def test_the_three_mutation_strategies_with_fixed_donors():
    # x0..x3 = (0,0), (1,0), (0,1), (2,2); archive row a0 = (10,10) sits
    # at union index 4. F = 0.5, Fa = 0.75. Slot 3 is dead.
    pop = _population(
        [[0, 0], [1, 0], [0, 1], [2, 2]],
        archive_rows=[[10, 10]],
        live=[True, True, True, False],
    )
    strategy = jnp.array([0, 1, 2, 0])
    best = jnp.array([3, 0, 3, 0])
    r1 = jnp.array([1, 2, 1, 0])
    r2 = jnp.array([4, 3, 0, 0])
    v = madde.madde_mutants(
        pop, jnp.full(4, 0.5), 0.75, strategy, best, r1, r2
    )
    # current-to-pbest/1 + archive (eq. 4):
    #   x0 + F (x3 - x0) + F (x1 - a0) = (1, 1) + (-4.5, -5)
    # current-to-rand/1 + archive (eq. 5):
    #   x1 + F (x2 - x3) = (1, 0) + (-1, -0.5)
    # weighted-rand-to-qbest/1 (eq. 6):
    #   F x1 + F Fa (x3 - x0) = (0.5, 0) + 0.375 (2, 2)
    np.testing.assert_allclose(
        v[:3], [[-3.5, -4.0], [0.0, -0.5], [1.25, 0.75]]
    )
    assert jnp.all(jnp.isnan(v[3]))


def _donor_draws(within_strategy, n_keys=200):
    # Ten slots, the last two dead; cost order 5, 2, 7, 0, 4, 1, 6, 3.
    # Two valid archive rows (union indices 10, 11) of three.
    cost = [4.0, 5.0, 1.0, 7.0, 4.5, 0.0, 6.0, 2.0, 0.0, 0.0]
    live = [True] * 8 + [False] * 2
    pop = _population(
        np.arange(20.0).reshape(10, 2),
        archive_rows=[[0, 0], [1, 1]],
        live=live,
        cost=cost,
        archive_max=3,
    )
    strategy = jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    def draw(key):
        return madde.sample_donors(
            key,
            pop,
            strategy,
            n_pbest=2,
            n_qbest=3,
            within_strategy=within_strategy,
        )

    best, r1, r2 = jax.vmap(draw)(jr.split(jr.key(3), n_keys))
    return (
        np.asarray(strategy),
        np.asarray(best),
        np.asarray(r1),
        np.asarray(r2),
    )


def _assert_best_from_the_top(strategy, best):
    # pbest among the 2 best (slots 5, 2), qbest among the 3 best.
    for i in range(8):
        if strategy[i] == 0:
            assert set(best[:, i]) == {5, 2}
        elif strategy[i] == 2:
            assert set(best[:, i]) == {5, 2, 7}


def test_paper_donors_come_from_the_whole_population():
    # gnR1R2 in the authors' code: r1 != i from P; r2 != i, r1 from P U A
    # (eq. 4-5); the weighted strategy's second donor from P (eq. 6).
    strategy, best, r1, r2 = _donor_draws(False)
    _assert_best_from_the_top(strategy, best)
    for i in range(8):
        assert set(r1[:, i]) <= set(range(8)) - {i}
        assert np.all(r2[:, i] != r1[:, i]) and np.all(r2[:, i] != i)
        allowed = set(range(8)) | ({10, 11} if strategy[i] != 2 else set())
        assert set(r2[:, i]) <= allowed
    # Not restricted to the slot's own strategy.
    assert {int(strategy[j]) for j in r1[:, 0]} == {0, 1, 2}


def test_metabox_donors_come_from_the_slots_own_strategy_group():
    # MetaBox splits the population by strategy and runs each mutation on
    # its own group (p1 = group[mu == 0], ...): x_r1 and the population
    # part of x_r2 belong to the same group as x_i; pbest and qbest come
    # from the whole population.
    strategy, best, r1, r2 = _donor_draws(True)
    _assert_best_from_the_top(strategy, best)
    for i in range(8):
        group = {j for j in range(8) if strategy[j] == strategy[i]} - {i}
        assert set(r1[:, i]) <= group
        archive = {10, 11} if strategy[i] != 2 else set()
        for a, b in zip(r1[:, i], r2[:, i], strict=False):
            # No candidate left (slot 2's group is {2, 5}): MetaBox keeps
            # a duplicate after its re-draws; the port falls back to i.
            assert b in ((group - {a}) | archive or {i})


#                                                    q-best binomial crossover
# =============================================================================


def test_qbx_crossover_swaps_the_parent_for_a_qbest_row_when_drawn():
    # Eq. (10)-(11): with probability p_qBX the target vector of the
    # binomial crossover is a q-best row instead of the parent.
    x = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    v = jnp.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]])
    qbest_rows = jnp.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    u = jnp.array([[0.1, 0.9, 0.9], [0.9, 0.1, 0.9]])
    got = madde.qbx_crossover(
        x,
        v,
        cr=jnp.array([0.5, 0.5]),
        u=u,
        jrand=jnp.array([2, 0]),
        use_qbest=jnp.array([True, False]),
        qbest_rows=qbest_rows,
    )
    np.testing.assert_allclose(got, [[9.0, 5.0, 9.0], [8.0, 8.0, 1.0]])


def test_qbest_pool_is_the_leading_rows_of_sorted_population_then_archive():
    # Both references take the first rows of [P sorted by cost; A].
    pop = _population(
        np.arange(10.0).reshape(5, 2),
        archive_rows=[[0, 0], [1, 1]],
        live=[True, True, True, True, False],
        cost=[3.0, 1.0, 2.0, 0.0, 0.0],
        archive_max=3,
    )
    union = np.asarray(madde.qbest_pool(pop, 5))
    assert union.tolist() == [True] * 4 + [False] + [True, False, False]
    best_two = np.asarray(madde.qbest_pool(pop, 2))
    assert np.flatnonzero(best_two).tolist() == [1, 3]


def test_top_counts_round_in_the_paper_and_truncate_in_metabox():
    # Paper code: max(round(q * NP), 2); MetaBox: max(int(q * NP), 2).
    paper = madde.madde_variant(d=2)
    metabox = madde.madde_metabox_variant(d=2)
    assert paper.top_count(0.25, 34) == 9
    assert metabox.top_count(0.25, 34) == 8
    assert paper.top_count(0.01, 30) == metabox.top_count(0.01, 30) == 2


#                                                   Strategy probabilities p_m
# =============================================================================


def _pm(metabox, cost, trial, strategy, live=None):
    live = jnp.ones(len(cost), bool) if live is None else jnp.asarray(live)
    return np.asarray(
        madde.update_strategy_probabilities(
            jnp.asarray(cost, dtype=float),
            jnp.asarray(trial, dtype=float),
            jnp.asarray(strategy),
            live,
            metabox=metabox,
        )
    )


def test_paper_strategy_probabilities_follow_mean_relative_improvement():
    # Eq. (20)-(21), authors' code: ratio max(0, f(x) - f(u)) / |f(x)|,
    # averaged per strategy, then p_m = ratio / sum, truncated to
    # [0.1, 0.9] (and renormalized). Means: s0 (0.02 + 0) / 2 = 0.01
    # (slot 6 is dead), s1 0, s2 (2 / |-2| + 0) / 2 = 0.5 (slot 5 was
    # not evaluated). 0.01 / 0.51 and 0 truncate to 0.1, 0.5 / 0.51 to 0.9.
    got = _pm(
        False,
        cost=[10, 10, 4, 5, -2, 3, 1],
        trial=[9.8, 12, 4, 5, -4, np.inf, 0],
        strategy=[0, 0, 1, 1, 2, 2, 0],
        live=[True] * 6 + [False],
    )
    np.testing.assert_allclose(got, [1 / 11, 1 / 11, 9 / 11], rtol=1e-6)
    # No improvement anywhere: back to uniform.
    got = _pm(False, [1, 1, 1], [1, 2, 1], [0, 1, 2])
    np.testing.assert_allclose(got, [1 / 3] * 3)
    # A strategy nobody used counts as no improvement (MATLAB's
    # max(0, mean([])) = 0); the others still adapt.
    got = _pm(False, [1, 1, 1, 1], [0.5, 1, 0.5, 1], [0, 0, 2, 2])
    np.testing.assert_allclose(got, [0.5 / 1.1, 0.1 / 1.1, 0.5 / 1.1])


def test_metabox_strategy_probabilities_keep_its_signed_and_empty_quirks():
    # MetaBox: count_S[i] = mean(df[mu == i] / cost[mu == i]) with
    # df = max(0, cost - ncost); p = clip(count / sum, 0.1, 0.9),
    # renormalized, if sum(count) > 0, else uniform.
    got = _pm(
        True,
        cost=[10, 10, 4, 5, 2, 3],
        trial=[9.8, 12, 4, 5, 1, np.inf],
        strategy=[0, 0, 1, 1, 2, 2],
    )
    np.testing.assert_allclose(got, [1 / 11, 1 / 11, 9 / 11], rtol=1e-6)
    # Dividing by a negative cost flips the sign: the sum is negative and
    # the probabilities reset to uniform (means 0.01, 0.25, -0.5).
    got = _pm(True, [10, 10, 4, -2, 3], [9.8, 12, 3, -4, 3], [0, 0, 1, 2, 2])
    np.testing.assert_allclose(got, [1 / 3] * 3)
    # An unused strategy's mean is NaN, so the sum is NaN: uniform.
    got = _pm(True, [1, 1, 1, 1], [0.5, 1, 0.5, 1], [0, 0, 2, 2])
    np.testing.assert_allclose(got, [1 / 3] * 3)


#                                                     Population-size reduction
# =============================================================================


def _one_generation(variant, fevals, max_fevals, trial_better):
    """Init, evaluate, ask and tell once at ``fevals``; the new population."""
    n = int(variant.population_size(0, max_fevals))
    x = jr.uniform(jr.key(1), (n, variant.d), minval=-5.0, maxval=5.0)
    pop, private = variant.init(jr.key(2), x)
    pop = eqx.tree_at(lambda p: p.cost, pop, jnp.arange(n, dtype=float) + 1.0)
    progress = dc.DEProgress(jnp.array(fevals), jnp.array(max_fevals))
    trials, aux = variant.ask(jr.key(3), pop, private, progress)
    trial_cost = pop.cost - 0.5 if trial_better else pop.cost + 0.5
    pop, private = variant.tell(
        pop, private, trials, trial_cost, aux, progress
    )
    return n, pop


def test_paper_reduction_is_linear_from_2_d_squared_to_4_with_archive_rate():
    # Eq. (19) with NP_max = 2 D^2 = 200 (D = 10), NP_min = 4; the archive
    # holds round(2.3 NP): 460 at the start, 235 for NP = 102.
    variant = madde.madde_variant(d=10)
    assert int(variant.population_size(0, 10_000)) == 200
    assert int(variant.population_size(10_000, 10_000)) == 4
    n, pop = _one_generation(variant, 5_000, 10_000, trial_better=True)
    assert n == 200 and pop.archive.shape[0] == 460
    # round(200 - 196 * 0.5) = 102 survivors, the best ones, sorted.
    assert int(pop.size) == 102
    assert jnp.all(jnp.diff(pop.cost[:102]) >= 0)
    assert jnp.all(jnp.isnan(pop.x[102:]))
    # All 200 replaced parents were archived, 200 <= 235 fit.
    assert int(pop.archive_capacity) == 235
    assert int(pop.archive_count) == 200


def test_metabox_reduction_is_nonlinear_170_to_30_and_trims_archive_late():
    # Population.NLPSR: N = round(170 - 140 p^(1 - p)) = 71 at p = 0.5;
    # A = int(max(2.1 N, 30)) = 149, applied only once the archive holds
    # more than A rows (else the capacity stays int(2.1 * 170) = 357).
    variant = madde.madde_metabox_variant(d=10)
    assert int(variant.population_size(10_000, 10_000)) == 30
    n, pop = _one_generation(variant, 5_000, 10_000, trial_better=True)
    assert n == 170 and pop.archive.shape[0] == 357
    assert int(pop.size) == 71
    assert int(pop.archive_count) == 149
    assert int(pop.archive_capacity) == 149
    _, pop = _one_generation(variant, 5_000, 10_000, trial_better=False)
    assert int(pop.size) == 71
    assert int(pop.archive_count) == 0
    assert int(pop.archive_capacity) == 357


#                                                            Registry entries
# =============================================================================

_REGISTRY_CASES = [
    # name, its variant at d = 2, budget
    ("madde", madde.madde_variant(d=2), 120),
    ("madde_metabox", madde.madde_metabox_variant(d=2), 2_000),
]


@pytest.mark.parametrize(("name", "variant", "budget"), _REGISTRY_CASES)
def test_registry_entry_steps_bills_live_rows_and_stays_in_budget(
    name, variant, budget
):
    from .toy_problems import sphere_problem

    update = optimizer_mapping(name)(
        **sphere_problem(2),
        opt_hash=11,
        bounded=(-5.0, 5.0),
        stop_fn=None,
        max_fevals=budget,
    )
    n_max = int(variant.population_size(0, budget))
    assert update.popsize == n_max
    steps = variant.generations_for_budget(budget)
    params = jr.uniform(jr.key(4), (n_max, 2), minval=-5.0, maxval=5.0)
    carry = (params, update.init_fn(params, jr.key(5)), jr.key(6))
    step = eqx.filter_jit(update.step_fn)
    total = 0
    for _ in range(steps):
        rows = carry[0]
        carry, history = step(carry, {})
        n = int(history.fevals)
        evaluated = ~jnp.all(jnp.isnan(rows), axis=1)
        # Billed = the rows evaluated; the rest of the history is NaN.
        assert n == int(jnp.sum(evaluated))
        assert jnp.all(jnp.isfinite(history.loss[evaluated]))
        assert jnp.all(jnp.isnan(history.loss[~evaluated]))
        total += n
    state = carry[1]
    assert total == int(state.fevals) <= budget
    # One more generation would overshoot the budget.
    assert total + int(state.pop.size) > budget
    # The population shrank as planned and NaN-pads the freed slots.
    assert int(state.pop.size) == int(variant.population_size(total, budget))
    assert int(state.pop.size) < n_max
    assert jnp.all(jnp.isnan(carry[0][int(state.pop.size) :]))


@pytest.mark.parametrize("name", ["madde", "madde_metabox"])
def test_registry_entry_builds_without_budget_but_refuses_to_run(name):
    # resolve_popsize and the config checks build every name without a
    # budget; only starting a run needs max_fevals.
    from .toy_problems import sphere_problem

    update = optimizer_mapping(name)(
        **sphere_problem(2), opt_hash=1, bounded=(None, None), stop_fn=None
    )
    params = jnp.zeros((update.popsize, 2))
    with pytest.raises(ValueError, match="max_fevals"):
        update.init_fn(params, jr.key(0))


#                                             Whole runs against the reference
# =============================================================================

_FIXTURE = Path(__file__).parent / "data" / "madde_metabox_fixture.json"
#: MetaBox stops a run once its best cost is below 1e-8; compare above it.
_FLOOR = 1e-8


def _functions(shift):
    def sphere(x, **_):
        return jnp.sum((x - shift) ** 2)

    def rastrigin(x, **_):
        z = x - shift
        return 10.0 * z.size + jnp.sum(z**2 - 10.0 * jnp.cos(2 * jnp.pi * z))

    def rosenbrock(x, **_):
        return jnp.sum(
            100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2
        )

    return {"sphere": sphere, "rastrigin": rastrigin, "rosenbrock": rosenbrock}


def _final_bests(name, loss_fn, seeds, fixture, bounded, fevals=None):
    """Best cost of runs of registry entry ``name`` over the budget.

    The schedule is planned against the fixture's budget. ``fevals``, if
    given, is the exact number of evaluations to spend (MetaBox's own
    count, which overshoots the budget by its last generation); by
    default the run stops at the last whole step within the budget.
    """
    dim, budget = fixture["dim"], fixture["max_fevals"]
    low, high = fixture["bounds"]
    update = optimizer_mapping(name)(
        model=jnp.zeros(dim),
        loss_fn=loss_fn,
        pass_rng=False,
        opt_hash=0,
        bounded=bounded,
        stop_fn=None,
        max_fevals=budget,
    )
    variant = (
        madde.madde_metabox_variant(d=dim)
        if name == "madde_metabox"
        else madde.madde_variant(d=dim, bounds=(low, high))
    )
    steps = (
        variant.generations_for_budget(budget)
        if fevals is None
        else _steps_spending(variant, fevals, budget)
    )

    def run(key):
        k_x, k_init, k_run = jr.split(key, 3)
        params = jr.uniform(
            k_x, (update.popsize, dim), minval=low, maxval=high
        )
        carry = (params, update.init_fn(params, k_init), k_run)
        carry = jax.lax.fori_loop(
            0, steps, lambda _, c: update.step_fn(c, {})[0], carry
        )
        return jnp.min(carry[1].pop.cost)

    keys = jnp.stack([jr.key(seed) for seed in seeds])
    return np.asarray(eqx.filter_jit(jax.vmap(run))(keys)).tolist()


def _steps_spending(variant, fevals, max_fevals):
    """Steps whose cumulative bill is exactly ``fevals``.

    The run's schedule is planned against ``max_fevals``. Raises when no
    whole number of steps bills exactly ``fevals``.
    """
    bill, spent, step = variant.bill(), 0, 0
    while spent < fevals:
        spent += int(bill(step + 1, spent, max_fevals))
        step += 1
    if spent != fevals:
        raise ValueError(f"no step count bills exactly {fevals}: {spent}")
    return step


def test_parity_harness_spends_what_metabox_spent():
    # MetaBox's loop (``while FEs < MaxFEs: ... FEs += NP``) lets its last
    # generation run past MaxFEs, so its runs evaluate slightly more than
    # the budget. The parity test gives the port the same number of
    # evaluations (one generation more than generations_for_budget), so
    # neither side gets extra search.
    fixture = json.loads(_FIXTURE.read_text())
    budget = fixture["max_fevals"]
    variant = madde.madde_metabox_variant(d=fixture["dim"])
    for function, entry in fixture["functions"].items():
        used = set(entry["fevals_used"])
        assert len(used) == 1, (function, used)
        (fevals,) = used
        assert fevals > budget
        steps = _steps_spending(variant, fevals, budget)
        assert steps == variant.generations_for_budget(budget) + 1


def _floored_log(values):
    return np.log10(np.maximum(np.asarray(values), _FLOOR))


@pytest.mark.slow
@pytest.mark.parametrize("function", ["sphere", "rastrigin", "rosenbrock"])
def test_madde_metabox_matches_metabox_madde_statistically(function):
    # 20 seeds against the 20 MetaBox runs in the fixture: a two-sided
    # Mann-Whitney U test on log10 of the final best cost (floored at
    # MetaBox's 1e-8 stop) must not reject at the 1% level, and the
    # medians must agree within a factor of 10. Both sides spend the same
    # evaluations: MetaBox's last generation overshoots the budget (see
    # test_parity_harness_spends_what_metabox_spent), and the port runs
    # that generation too, on the same schedule.
    from scipy.stats import mannwhitneyu

    fixture = json.loads(_FIXTURE.read_text())
    loss_fn = _functions(fixture["shift"])[function]
    entry = fixture["functions"][function]
    reference = entry["final_best"]
    ours = _final_bests(
        "madde_metabox",
        loss_fn,
        fixture["seeds"],
        fixture,
        (None, None),
        fevals=entry["fevals_used"][0],
    )
    a, b = _floored_log(ours), _floored_log(reference)
    assert mannwhitneyu(a, b, alternative="two-sided").pvalue > 0.01, (
        np.median(ours),
        np.median(reference),
    )
    assert abs(np.median(a) - np.median(b)) <= 1.0


@pytest.mark.slow
@pytest.mark.parametrize("function", ["sphere", "rastrigin", "rosenbrock"])
def test_madde_paper_variant_is_within_a_factor_of_the_metabox_median(
    function,
):
    # The paper variant (2 D^2 = 200 individuals, linear reduction to 4,
    # bound repair towards the run's box) over 5 seeds: its median final
    # best cost is at most 10 times MetaBox's median (both floored).
    fixture = json.loads(_FIXTURE.read_text())
    loss_fn = _functions(fixture["shift"])[function]
    ours = _final_bests(
        "madde", loss_fn, range(5), fixture, tuple(fixture["bounds"])
    )
    reference = fixture["functions"][function]["median"]
    assert (
        np.median(_floored_log(ours)) <= _floored_log([reference])[0] + 1.0
    ), (
        np.median(ours),
        reference,
    )
