"""Tests for ``jde21`` (the paper's j21) and ``jde21_metabox`` (MetaBox v2).

Operator tests feed each jde21-specific operator fixed random draws and
check the result against values worked out by hand from Brest, Maučec
and Bošković (CEC 2021, Table II and Algorithms 1-2) or from MetaBox v2
``JDE21`` (``tests/reference/metabox_rldas/rldas_optimizer.py``). The
registry test drives both names through ``optimizer_mapping``. The slow
tests compare ``jde21_metabox`` with the vendored MetaBox ``JDE21``
through a stored fixture (``tests/data/jde21_metabox_fixture.json``,
written by ``tests/reference/metabox_rldas/make_fixture_jde21.py``) and
check that the paper variant is not far behind it.
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
from l2co_optimizers._src import jde21 as j

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

VARIANTS = {"jde21": j.JDE21, "jde21_metabox": j.JDE21Metabox}


# =============================================================================
#                                                  Self-adaptation of F and CR


@pytest.mark.parametrize(
    ("name", "small_cr"),
    [
        # Paper, Table II: CR_l,s + rand * CR_u,s = 0.1 + 0.5 * 0.8.
        ("jde21", 0.5),
        # MetaBox draws the small population's CR with CRu_b:
        # 0.1 + 0.5 * 1.1.
        ("jde21_metabox", 0.65),
    ],
)
def test_self_adaptation_redraws_with_probability_tau(name, small_cr):
    variant = VARIANTS[name]()
    f = jnp.array([0.3, 0.3])
    cr = jnp.array([0.4, 0.4])
    rand = jnp.array([0.5, 0.5])
    # tau1 = tau2 = 0.1: row 0 redraws (0.05 < 0.1), row 1 keeps (0.5).
    tau = jnp.array([0.05, 0.5])

    f_big, cr_big = variant.adapt(f, cr, rand, tau, rand, tau, big=True)
    # Big: F_l,b + rand * F_u = 0.1 + 0.5 * 1.1; CR_l,b + rand * CR_u,b
    # = 0.0 + 0.5 * 1.1.
    np.testing.assert_allclose(f_big, [0.65, 0.3])
    np.testing.assert_allclose(cr_big, [0.55, 0.4])

    f_small, cr_small = variant.adapt(f, cr, rand, tau, rand, tau, big=False)
    # Small: F_l,s + rand * F_u = 0.17 + 0.5 * 1.1.
    np.testing.assert_allclose(f_small, [0.72, 0.3])
    np.testing.assert_allclose(cr_small, [small_cr, 0.4])


# =============================================================================
#                                                  Crowding and jDE selection


def _population(variant, x, cost):
    """A population of ``len(x)`` one-dimensional individuals, evaluated."""
    x = jnp.asarray(x, dtype=float)[:, None]
    pop, private = variant.init(jr.key(0), x)
    return eqx.tree_at(
        lambda p: p.cost, pop, jnp.asarray(cost, float)
    ), private


@pytest.mark.parametrize("name", ["jde21", "jde21_metabox"])
def test_crowding_trial_competes_with_nearest_big_individual(name):
    variant = VARIANTS[name](n_big=4, n_small=2, low=-20.0, high=20.0)
    # Big population at 0, 1, 2, 3 (cost 5 each); small at 7, 8.
    pop, private = _population(
        variant, [0.0, 1.0, 2.0, 3.0, 7.0, 8.0], [5, 5, 5, 5, 1, 2]
    )
    nan = jnp.nan
    # Nearest big individuals: 1, 1, 3, 3. The small rows are not
    # evaluated in a big step.
    trials = jnp.array([0.9, 1.1, 2.9, 10.0, nan, nan])[:, None]
    cost = jnp.array([4.0, 3.0, 6.0, 5.0, jnp.inf, jnp.inf])
    f = jnp.array([0.11, 0.12, 0.13, 0.14, nan, nan])
    cr = jnp.array([0.21, 0.22, 0.23, 0.24, nan, nan])
    u_archive = jnp.zeros(6)

    new, _ = variant.select(
        pop, private, trials, cost, f, cr, u_archive, big=True
    )

    # Slot 1 is beaten by trial 0 (4 < 5), then by trial 1 (3 < 4); it
    # keeps trial 1 and the F, CR that built it. Slot 3 is a tie with
    # trial 3 (5 vs 5): jDE replaces on <= (Alg. 1, step 17), MetaBox
    # only on <.
    tie = name == "jde21"
    np.testing.assert_allclose(
        new.x[:, 0], [0.0, 1.1, 2.0, 10.0 if tie else 3.0, 7.0, 8.0]
    )
    np.testing.assert_allclose(new.cost, [5, 3, 5, 5, 1, 2])
    np.testing.assert_allclose(
        new.f[:4], [0.5, 0.12, 0.5, 0.14 if tie else 0.5]
    )
    np.testing.assert_allclose(
        new.cr[:4], [0.9, 0.22, 0.9, 0.24 if tie else 0.9]
    )


def test_metabox_selection_records_archive_and_index_i_improvement():
    variant = j.JDE21Metabox(n_big=4, n_small=2, low=-20.0, high=20.0)
    pop, private = _population(
        variant, [0.0, 1.0, 2.0, 3.0, 7.0, 8.0], [5, 5, 5, 5, 1, 2]
    )
    nan = jnp.nan
    trials = jnp.array([0.9, 1.1, 2.9, 10.0, nan, nan])[:, None]
    cost = jnp.array([4.0, 3.0, 6.0, 5.0, jnp.inf, jnp.inf])
    f = jnp.array([0.11, 0.12, 0.13, 0.14, nan, nan])
    cr = jnp.array([0.21, 0.22, 0.23, 0.24, nan, nan])

    new, _ = variant.select(
        pop, private, trials, cost, f, cr, jnp.zeros(6), big=True
    )

    # Each success archives the individual it displaces: the parent at
    # 1.0, then trial 0 (0.9), which trial 1 displaces in turn.
    assert int(new.archive_count) == 2
    np.testing.assert_allclose(new.archive[:2, 0], [1.0, 0.9])
    # Successes in loop order. MetaBox's improvement is
    # (cost[i] - c_i) / (cost[i] + 1e-9) with the loop index i, not the
    # replaced slot, read after the write: trial 0 reads slot 0
    # (5, untouched): 1 / 5; trial 1 reads slot 1, which it has just
    # set to 3: 0.
    assert int(new.success_count) == 2
    np.testing.assert_allclose(new.success_f[:2], [0.11, 0.12])
    np.testing.assert_allclose(new.success_cr[:2], [0.21, 0.22])
    np.testing.assert_allclose(new.success_weight[:2], [0.2, 0.0], atol=1e-9)


# =============================================================================
#                                                        Age and restarts


def _state(variant, private, **fields):
    """``private`` with some fields replaced (MetaBox: already started)."""
    if isinstance(private, j.JDE21MetaboxState):
        fields.setdefault("started", True)
    names = tuple(fields)
    return eqx.tree_at(
        lambda s: tuple(getattr(s, k) for k in names),
        private,
        tuple(jnp.asarray(v) for v in fields.values()),
    )


def _fresh(n):
    """Restart rows: 10, 11, ... so a test can tell which slot took which."""
    return (10.0 + jnp.arange(n, dtype=float))[:, None]


@pytest.mark.parametrize("name", ["jde21", "jde21_metabox"])
@pytest.mark.parametrize(
    ("cost", "age", "restarts"),
    [
        # ageLmt = MaxFEs / 10 = 10: the big population restarts once
        # its best has not improved for more than 10 evaluations.
        ([5, 6, 7, 8, 1, 2], 10, False),
        ([5, 6, 7, 8, 1, 2], 11, True),
        # Stagnation: more than 2 and more than myEqs = 25 % of the big
        # population lie within eps of the best.
        ([3, 3, 3, 8, 1, 2], 0, True),
        ([3, 3, 7, 8, 1, 2], 0, False),
    ],
)
def test_big_population_restarts_on_age_or_stagnation(
    name, cost, age, restarts
):
    variant = VARIANTS[name](n_big=4, n_small=2, low=-20.0, high=20.0)
    pop, private = _population(variant, [0.0, 1.0, 2.0, 3.0, 7.0, 8.0], cost)
    if name == "jde21_metabox":
        private = _state(variant, private, age=age, best_slot=4)
        # MetaBox compares with the global best; the paper with the big
        # population's best. Give MetaBox the same stagnation signal.
        if cost[0] == 3:
            pop = eqx.tree_at(lambda p: p.cost, pop, pop.cost.at[4:].set(3.0))
    else:
        private = _state(variant, private, age=age)
    progress = j.dc.DEProgress(jnp.array(50), jnp.array(100.0))

    new, state = variant.prepare(pop, private, progress, _fresh(6))

    unevaluated = 1e15 if name == "jde21_metabox" else jnp.inf
    if restarts:
        np.testing.assert_allclose(new.x[:, 0], [10, 11, 12, 13, 7, 8])
        np.testing.assert_allclose(new.cost[:4], unevaluated)
        np.testing.assert_allclose(new.f[:4], 0.5)
        np.testing.assert_allclose(new.cr[:4], 0.9)
        assert int(state.age) == 0
    else:
        np.testing.assert_allclose(new.x[:, 0], [0, 1, 2, 3, 7, 8])
        np.testing.assert_allclose(new.cost, pop.cost)
        assert int(state.age) == age


@pytest.mark.parametrize("name", ["jde21", "jde21_metabox"])
def test_small_population_restart_keeps_its_best(name):
    variant = VARIANTS[name](n_big=4, n_small=4, low=-20.0, high=20.0)
    pop, private = _population(
        variant,
        [0.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 10.0],
        [5, 6, 7, 8, 1, 1, 1, 2],
    )
    # Three of four small-population costs within eps of its best.
    # The paper checks before the big step (Alg. 2, step 6); MetaBox
    # after it, when its tracked best is in the small population.
    if name == "jde21_metabox":
        private = _state(variant, private, phase=1, best_slot=4)
    progress = j.dc.DEProgress(jnp.array(50), jnp.array(100.0))

    new, _ = variant.prepare(pop, private, progress, _fresh(8))

    unevaluated = 1e15 if name == "jde21_metabox" else jnp.inf
    np.testing.assert_allclose(new.x[:, 0], [0, 1, 2, 3, 7, 15, 16, 17])
    np.testing.assert_allclose(new.cost[:5], [5, 6, 7, 8, 1])
    np.testing.assert_allclose(new.cost[5:], unevaluated)
    np.testing.assert_allclose(new.f[5:], 0.5)
    np.testing.assert_allclose(new.cr[5:], 0.9)


@pytest.mark.parametrize("name", ["jde21", "jde21_metabox"])
def test_best_of_big_population_migrates_into_small(name):
    variant = VARIANTS[name](n_big=4, n_small=2, low=-20.0, high=20.0)
    pop, private = _population(
        variant, [0.0, 1.0, 2.0, 3.0, 7.0, 8.0], [0.5, 6, 7, 8, 1, 2]
    )
    # After the big step, the best (slot 0) is copied over the small
    # population's first slot (Alg. 2, steps 16-18; MetaBox ``cCopy``).
    if name == "jde21_metabox":
        private = _state(variant, private, phase=1, best_slot=0)
    else:
        private = _state(variant, private, phase=1)
    progress = j.dc.DEProgress(jnp.array(50), jnp.array(100.0))

    new, state = variant.prepare(pop, private, progress, _fresh(6))

    np.testing.assert_allclose(new.x[:, 0], [0, 1, 2, 3, 0, 8])
    np.testing.assert_allclose(new.cost, [0.5, 6, 7, 8, 0.5, 2])
    if name == "jde21_metabox":
        # MetaBox's tracked best now points at the copy.
        assert int(state.best_slot) == 4


def test_metabox_step_opens_with_a_descending_sort():
    variant = j.JDE21Metabox(n_big=4, n_small=2, low=-20.0, high=20.0)
    pop, private = _population(
        variant, [0.0, 1.0, 2.0, 3.0, 7.0, 8.0], [5, 6, 7, 8, 1, 2]
    )
    progress = j.dc.DEProgress(jnp.array(6), jnp.array(100.0))

    new, state = variant.prepare(pop, private, progress, _fresh(6))

    # ``population.sort(NP, True)``: worst first, so the two best
    # individuals form the small population at the tail.
    np.testing.assert_allclose(new.cost, [8, 7, 6, 5, 2, 1])
    np.testing.assert_allclose(new.x[:, 0], [3, 2, 1, 0, 8, 7])
    assert int(state.best_slot) == 5
    assert bool(state.started)


# =============================================================================
#                                                  End of a generation


@pytest.mark.parametrize(
    ("fevals", "expected_x"), [(10, None), (30, [2, 1, 7, 8])]
)
def test_paper_halves_big_population_pairwise_at_a_quarter(fevals, expected_x):
    variant = j.JDE21(n_big=4, n_small=2, low=-20.0, high=20.0)
    pop, private = _population(
        variant, [0.0, 1.0, 2.0, 3.0, 7.0, 8.0], [5, 1, 2, 6, 0, 0]
    )
    progress = j.dc.DEProgress(jnp.array(fevals), jnp.array(100.0))

    new, _ = variant.end_generation(pop, private, progress)

    if expected_x is None:  # before MaxFEs / 4: nothing happens
        np.testing.assert_allclose(new.x[:, 0], [0, 1, 2, 3, 7, 8])
        return
    # Slot j competes with slot j + bNP / 2 (Brest and Maučec 2008):
    # 5 vs 2 keeps 2, 1 vs 6 keeps 1; the small population follows.
    assert int(new.size) == 4
    np.testing.assert_allclose(new.x[:4, 0], expected_x)
    assert jnp.all(jnp.isnan(new.x[4:]))


def test_metabox_generation_end_keeps_tail_and_writes_signed_lehmer_means():
    variant = j.JDE21Metabox(n_big=8, n_small=2, n_min=4, low=-20.0, high=20.0)
    x = jnp.arange(10, dtype=float)
    pop, private = _population(variant, x, 10.0 - x)
    pop = j.dc.record_successes(
        pop,
        jnp.array([0.5, 0.7]),
        jnp.array([0.2, 0.4]),
        jnp.array([1.0, -3.0]),
        jnp.array([True, True]),
    )
    progress = j.dc.DEProgress(jnp.array(50), jnp.array(100.0))

    new, state = variant.end_generation(pop, private, progress)

    # NLPSR: round(10 + (4 - 10) * 0.5 ** 0.5) = 6; MetaBox keeps
    # ``group[-NP:]``, the last six slots, whatever their costs.
    assert int(new.size) == 6
    np.testing.assert_allclose(new.x[:6, 0], [4, 5, 6, 7, 8, 9])
    assert int(state.best_slot) == 5
    # ``mean_wL`` with w = df / sum(df) = [-0.5, 1.5], signs kept:
    # F: (-0.5 * 0.25 + 1.5 * 0.49) / (-0.5 * 0.5 + 1.5 * 0.7) = 0.7625;
    # Cr: (-0.5 * 0.04 + 1.5 * 0.16) / (-0.5 * 0.2 + 1.5 * 0.4) = 0.44.
    np.testing.assert_allclose(new.memory_f[0], 0.7625, rtol=1e-5)
    np.testing.assert_allclose(new.memory_cr[0], 0.44, rtol=1e-5)
    assert int(new.memory_index) == 1
    assert int(new.success_count) == 0


# =============================================================================
#                                                        Registry entries


@pytest.mark.parametrize("name", ["jde21", "jde21_metabox"])
def test_registry_entry_steps_and_bills_only_evaluated_rows(name):
    from .toy_problems import sphere_problem

    sizes = dict(n_big=40, n_small=10)
    if name == "jde21_metabox":
        sizes["n_min"] = 30
    budget = 1500
    update = optimizer_mapping(name)(
        **sphere_problem(3),
        opt_hash=5,
        bounded=(-5.0, 5.0),
        stop_fn=None,
        max_fevals=budget,
        **sizes,
    )
    assert update.popsize == 50
    variant = VARIANTS[name](**sizes, low=-5.0, high=5.0)
    n_steps = variant.generations_for_budget(budget)

    params = jr.uniform(jr.key(1), (50, 3), minval=-5.0, maxval=5.0)
    carry = (params, update.init_fn(params, jr.key(2)), jr.key(3))
    step = eqx.filter_jit(update.step_fn)
    bills, best = [], []
    for _ in range(n_steps):
        rows = ~jnp.all(jnp.isnan(carry[0]), axis=1)
        carry, history = step(carry, {})
        n = int(history.fevals)
        bills.append(n)
        # Billed rows are exactly the rows handed in for evaluation;
        # every other history entry is NaN padding.
        assert n == int(jnp.sum(rows))
        assert jnp.all(jnp.isfinite(history.loss[rows]))
        assert jnp.all(jnp.isnan(history.loss[~rows]))
        best.append(float(jnp.nanmin(history.loss)))

    # The starting population, then a big step (bNP = 40 rows) and
    # bNP / sNP = 4 small rounds of 10.
    assert bills[:6] == [50, 40, 10, 10, 10, 10]
    total = sum(bills)
    assert total <= budget
    assert total + int(variant.bill()(n_steps, total, budget)) > budget
    assert int(carry[1].fevals) == total
    # It optimizes: the best evaluated loss went down.
    assert min(best[-5:]) < best[0]


@pytest.mark.parametrize("name", ["jde21", "jde21_metabox"])
def test_registry_entry_without_budget_builds_but_does_not_run(name):
    from .toy_problems import sphere_problem

    # Config checks and popsize resolution build every name with no
    # budget; only a run needs it.
    update = optimizer_mapping(name)(
        **sphere_problem(3), opt_hash=5, bounded=(-5.0, 5.0), stop_fn=None
    )
    assert update.popsize == 170
    params = jnp.zeros((170, 3))
    with pytest.raises(ValueError, match="max_fevals"):
        update.init_fn(params, jr.key(0))


# =============================================================================
#                                       Statistical parity with MetaBox (slow)

_FIXTURE = Path(__file__).parent / "data" / "jde21_metabox_fixture.json"
#: MetaBox's JDE21 stops a run once a cost falls below 1e-8 and reports
#: 0; below this floor all results count as solved.
_FLOOR = 1e-8


def _sphere(x, **_):
    return jnp.sum((x - 0.5) ** 2)


def _rastrigin(x, **_):
    z = x - 0.5
    return 10.0 * z.size + jnp.sum(z**2 - 10.0 * jnp.cos(2 * jnp.pi * z))


def _rosenbrock(x, **_):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


#: Same definitions as ``make_fixture_jde21.py``.
_FUNCTIONS = {
    "sphere": _sphere,
    "rastrigin": _rastrigin,
    "rosenbrock": _rosenbrock,
}


def _final_best(name, loss_fn, *, dim, budget, low, high, seeds):
    """Best evaluated cost of ``seeds`` standalone runs, floored."""
    update = optimizer_mapping(name)(
        model=jnp.zeros(dim),
        loss_fn=loss_fn,
        pass_rng=False,
        opt_hash=0,
        bounded=(low, high),
        stop_fn=None,
        max_fevals=budget,
    )
    n_steps = VARIANTS[name](low=low, high=high).generations_for_budget(budget)

    def one(key):
        k_x, k_init, k_run = jr.split(key, 3)
        x = jr.uniform(k_x, (update.popsize, dim), minval=low, maxval=high)
        carry = (x, update.init_fn(x, k_init), k_run)

        def body(carry, _):
            carry, history = update.step_fn(carry, {})
            return carry, jnp.nanmin(history.loss)

        _, best = jax.lax.scan(body, carry, None, length=n_steps)
        return jnp.min(best)

    finals = jax.jit(jax.vmap(one))(jr.split(jr.key(0), seeds))
    return np.maximum(np.asarray(finals, dtype=float), _FLOOR)


@pytest.mark.slow
@pytest.mark.parametrize("function", list(_FUNCTIONS))
def test_metabox_variant_matches_vendored_metabox_statistically(function):
    fixture = json.loads(_FIXTURE.read_text())
    reference = np.maximum(fixture["results"][function]["finals"], _FLOOR)
    ours = _final_best(
        "jde21_metabox",
        _FUNCTIONS[function],
        dim=fixture["dim"],
        budget=fixture["max_fevals"],
        low=fixture["bounds"][0],
        high=fixture["bounds"][1],
        seeds=fixture["seeds"],
    )
    # Medians within a factor of 2 of each other, and a two-sided
    # Mann-Whitney U test that does not tell the samples apart at 1 %.
    ratio = np.median(ours) / np.median(reference)
    assert 0.5 <= ratio <= 2.0, (np.median(ours), np.median(reference))
    p = mannwhitneyu(ours, reference, alternative="two-sided").pvalue
    assert p > 0.01, p


@pytest.mark.slow
@pytest.mark.parametrize("function", list(_FUNCTIONS))
def test_paper_variant_is_within_a_factor_of_metabox_variant(function):
    fixture = json.loads(_FIXTURE.read_text())
    kwargs = dict(
        dim=fixture["dim"],
        budget=fixture["max_fevals"],
        low=fixture["bounds"][0],
        high=fixture["bounds"][1],
        seeds=5,
    )
    paper = _final_best("jde21", _FUNCTIONS[function], **kwargs)
    metabox = _final_best("jde21_metabox", _FUNCTIONS[function], **kwargs)
    # Sanity bound, not a benchmark: the paper's median final cost is at
    # most 10 times the MetaBox variant's (or better).
    assert np.median(paper) <= 10.0 * np.median(metabox)


# =============================================================================
#                                                The caller owns the population


@pytest.mark.parametrize("name", ["jde21", "jde21_metabox"])
def test_tell_keeps_edits_the_caller_made_between_ask_and_tell(name):
    # Under RL-DAS the wrapper owns the population and may write to it
    # between ``ask`` and ``tell`` (context memory, a switch's edits).
    # ``tell`` must fold the trials into the population it is handed,
    # not into a copy ``ask`` made.
    sizes = dict(n_big=8, n_small=4)
    if name == "jde21_metabox":
        sizes["n_min"] = 6
    variant = VARIANTS[name](**sizes, low=-5.0, high=5.0)
    x = jr.uniform(jr.key(0), (12, 2), minval=-5.0, maxval=5.0)
    pop, private = variant.init(jr.key(1), x)
    pop = eqx.tree_at(lambda p: p.cost, pop, jnp.sum(x**2, axis=1))
    private = _state(variant, private)
    progress = j.dc.DEProgress(jnp.array(12), jnp.array(1000.0))
    trials, aux = variant.ask(jr.key(2), pop, private, progress)
    big = ~jnp.all(jnp.isnan(trials), axis=1)
    assert jnp.all(big[:8]) and not jnp.any(big[8:])

    edited = eqx.tree_at(
        lambda p: (p.memory_f, p.x),
        pop,
        (jnp.full_like(pop.memory_f, 0.123), pop.x.at[9].set(4.5)),
    )
    cost = jnp.where(
        big, jnp.sum(jnp.nan_to_num(trials) ** 2, axis=1), jnp.inf
    )
    after = j.dc.DEProgress(jnp.array(20), jnp.array(1000.0))
    told, _ = variant.tell(edited, private, trials, cost, aux, after)

    np.testing.assert_allclose(told.memory_f, 0.123)
    np.testing.assert_allclose(told.x[9], 4.5)
    # The same step without the edits leaves those fields as they were.
    plain, _ = variant.tell(pop, private, trials, cost, aux, after)
    np.testing.assert_allclose(plain.x[9], pop.x[9])
