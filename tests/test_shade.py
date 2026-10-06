"""Unit tests for the SHADE optimizer internals.

Covers the success-history memory update math (hand-computed weighted
arithmetic / Lehmer means), archive bookkeeping, donor-index
distinctness, the truncated-Cauchy scale-factor sampler, the
turning-ring geometry, NaN robustness, and the factory's
constructor-vs-Params hyperparameter split.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from l2co_optimizers._src.popsize import shade_popsize, variable_popsize
from l2co_optimizers._src.shade import (
    SHADE,
    _donor_indices,
    _ring_radii,
    _truncated_cauchy,
    shade_update,
)

from .toy_problems import sphere_problem

# =============================================================================


def _make_algo(popsize=4, num_dims=3, **kwargs):
    return SHADE(
        population_size=popsize,
        solution=jnp.zeros(num_dims),
        **kwargs,
    )


def _init_state(algo, population, fitness, params=None, key=None):
    params = algo.default_params if params is None else params
    key = jr.key(0) if key is None else key
    return algo.init(
        key=key, population=population, fitness=fitness, params=params
    )


# =============================================================================
#                                                                       Helpers


def test_truncated_cauchy_strictly_positive():
    samples = _truncated_cauchy(jr.key(0), jnp.full(20_000, 0.05))
    assert jnp.all(samples > 0.0)


def test_truncated_cauchy_median_matches_inverse_cdf():
    location = 0.5
    scale = 0.1
    samples = _truncated_cauchy(jr.key(1), jnp.full(50_000, location))
    # Median of the truncated distribution: u = (cdf(0) + 1) / 2.
    cdf_at_zero = jnp.arctan(-location / scale) / jnp.pi + 0.5
    u_median = (cdf_at_zero + 1.0) / 2.0
    expected = location + scale * jnp.tan(jnp.pi * (u_median - 0.5))
    assert jnp.abs(jnp.median(samples) - expected) < 0.005


def test_donor_indices_distinctness():
    popsize = 6
    for archive_count in (0, 3):
        keys = jr.split(jr.key(2), 2000)
        r1, r2 = jax.vmap(
            lambda k, n=archive_count: _donor_indices(
                *jr.split(k), popsize, jnp.array(n)
            )
        )(keys)
        member_ids = jnp.arange(popsize)
        assert jnp.all(r1 != member_ids)
        assert jnp.all(r2 != member_ids)
        assert jnp.all(r2 != r1)
        assert jnp.all((r1 >= 0) & (r1 < popsize))
        assert jnp.all((r2 >= 0) & (r2 < popsize + archive_count))
        # Every admissible union index is actually reachable.
        assert jnp.unique(r2).size == popsize + archive_count


def test_ring_radii_schedule():
    span = jnp.asarray(10.0)
    num_dims = 4  # sqrt(D) = 2
    max_fevals = jnp.asarray(1000.0)
    inner0, outer0 = _ring_radii(span, num_dims, 0.0, max_fevals)
    assert outer0 == pytest.approx(10.0)  # sqrt(D) * span / 2
    assert inner0 == pytest.approx(0.5)  # sqrt(D) * span / 40
    _, outer_mid = _ring_radii(span, num_dims, 500.0, max_fevals)
    assert outer_mid == pytest.approx((10.0 + 0.5) / 2)
    inner_end, outer_end = _ring_radii(span, num_dims, 1000.0, max_fevals)
    assert outer_end == pytest.approx(float(inner_end))
    _, outer_past = _ring_radii(span, num_dims, 2000.0, max_fevals)
    assert outer_past == pytest.approx(float(inner_end))


# =============================================================================
#                                                    Selection, memory, archive


def test_tell_memory_update_weighted_means():
    algo = _make_algo()
    population = jnp.arange(12.0).reshape(4, 3)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = _init_state(algo, population, fitness)
    state = state.replace(
        scale_factors=jnp.array([0.5, 0.6, 0.7, 0.8]),
        crossover_rates=jnp.array([0.1, 0.2, 0.3, 0.4]),
    )

    trials = population + 100.0
    # successes: member 0 (df=0.5) and member 3 (df=1.0); member 1 ties
    # (replaced but not a success); member 2 worsens.
    trial_fitness = jnp.array([0.5, 2.0, 5.0, 3.0])
    new_state, _ = algo.tell(
        key=jr.key(3),
        population=trials,
        fitness=trial_fitness,
        state=state,
        params=algo.default_params,
    )

    # Selection: replace on <=, keep on >.
    assert jnp.allclose(new_state.population[0], trials[0])
    assert jnp.allclose(new_state.population[1], trials[1])  # tie
    assert jnp.allclose(new_state.population[2], population[2])
    assert jnp.allclose(new_state.population[3], trials[3])
    assert jnp.allclose(new_state.fitness, jnp.array([0.5, 2.0, 3.0, 3.0]))

    # Weights: df / sum(df) = [1/3, 0, 0, 2/3].
    w0, w3 = 1.0 / 3.0, 2.0 / 3.0
    expected_cr = w0 * 0.1 + w3 * 0.4
    expected_f = (w0 * 0.5**2 + w3 * 0.8**2) / (w0 * 0.5 + w3 * 0.8)
    assert new_state.memory_cr[0] == pytest.approx(expected_cr)
    assert new_state.memory_f[0] == pytest.approx(expected_f)
    assert int(new_state.memory_index) == 1
    # Untouched cells keep the 0.5 init value.
    assert jnp.allclose(new_state.memory_cr[1:], 0.5)
    assert jnp.allclose(new_state.memory_f[1:], 0.5)

    # Archive holds exactly the two defeated parents.
    assert int(new_state.archive_count) == 2
    valid_rows = new_state.archive[:2]
    for parent_idx in (0, 3):
        assert jnp.any(jnp.all(valid_rows == population[parent_idx], axis=1))


def test_tell_no_success_leaves_memory_and_archive():
    algo = _make_algo()
    population = jnp.arange(12.0).reshape(4, 3)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = _init_state(algo, population, fitness)

    new_state, _ = algo.tell(
        key=jr.key(4),
        population=population + 1.0,
        fitness=fitness + 1.0,  # strictly worse everywhere
        state=state,
        params=algo.default_params,
    )
    assert jnp.allclose(new_state.population, population)
    assert jnp.allclose(new_state.memory_f, 0.5)
    assert jnp.allclose(new_state.memory_cr, 0.5)
    assert int(new_state.memory_index) == 0
    assert int(new_state.archive_count) == 0


def test_memory_index_wraps_at_memory_size():
    algo = _make_algo(memory_size=2)
    population = jnp.zeros((4, 3))
    fitness = jnp.array([4.0, 4.0, 4.0, 4.0])
    state = _init_state(algo, population, fitness)

    for step in range(3):
        state, _ = algo.tell(
            key=jr.key(step),
            population=population,
            fitness=state.fitness - 1.0,  # always a strict improvement
            state=state,
            params=algo.default_params,
        )
    assert state.memory_f.shape == (2,)
    assert int(state.memory_index) == 1  # 0 -> 1 -> 0 -> 1


def test_archive_capacity_capped():
    algo = _make_algo(archive_size=2)
    population = jnp.arange(12.0).reshape(4, 3)
    fitness = jnp.array([1.0, 2.0, 3.0, 4.0])
    state = _init_state(algo, population, fitness)

    for step in range(3):
        state, _ = algo.tell(
            key=jr.key(step),
            population=state.population + 1.0,
            fitness=state.fitness - 1.0,
            state=state,
            params=algo.default_params,
        )
    assert int(state.archive_count) == 2
    assert jnp.all(jnp.isfinite(state.archive))


def test_tell_nan_robustness():
    algo = _make_algo()
    population = jnp.arange(12.0).reshape(4, 3)
    fitness = jnp.array([jnp.nan, 1.0, 2.0, 3.0])
    state = _init_state(algo, population, fitness)
    state = state.replace(
        scale_factors=jnp.array([0.4, 0.5, 0.6, 0.7]),
        crossover_rates=jnp.array([0.2, 0.4, 0.6, 0.8]),
    )

    trials = population + 100.0
    # member 0: finite trial beats NaN parent (infinite improvement);
    # member 1: NaN trial loses to finite parent; member 2: finite
    # improvement; member 3: worse.
    trial_fitness = jnp.array([1.0, jnp.nan, 1.0, 4.0])
    new_state, _ = algo.tell(
        key=jr.key(5),
        population=trials,
        fitness=trial_fitness,
        state=state,
        params=algo.default_params,
    )

    assert jnp.allclose(new_state.population[0], trials[0])
    assert jnp.allclose(new_state.population[1], population[1])
    assert jnp.allclose(new_state.fitness, jnp.array([1.0, 1.0, 1.0, 3.0]))
    # Infinite total improvement falls back to uniform weights over the
    # successes (members 0 and 2).
    expected_cr = (0.2 + 0.6) / 2
    expected_f = (0.5 * 0.4**2 + 0.5 * 0.6**2) / (0.5 * 0.4 + 0.5 * 0.6)
    assert new_state.memory_cr[0] == pytest.approx(expected_cr)
    assert new_state.memory_f[0] == pytest.approx(expected_f)
    assert bool(jnp.all(jnp.isfinite(new_state.memory_f)))
    assert bool(jnp.all(jnp.isfinite(new_state.memory_cr)))


# =============================================================================
#                                                                           Ask


def test_ask_respects_bounds_via_midpoint_repair():
    algo = _make_algo(popsize=6, num_dims=3)
    params = algo.default_params.replace(x_min=0.0, x_max=1.0)
    population = jr.uniform(jr.key(6), (6, 3))
    fitness = jnp.arange(6.0)
    state = _init_state(algo, population, fitness, params=params)

    for i in range(50):
        trials, _ = algo.ask(key=jr.key(i), state=state, params=params)
        assert jnp.all(trials >= 0.0) and jnp.all(trials <= 1.0)


def test_ask_unbounded_defaults_are_finite():
    algo = _make_algo(popsize=5, num_dims=2)
    population = jr.normal(jr.key(7), (5, 2))
    fitness = jnp.arange(5.0)
    state = _init_state(algo, population, fitness)

    trials, new_state = algo.ask(
        key=jr.key(8), state=state, params=algo.default_params
    )
    assert trials.shape == (5, 2)
    assert jnp.all(jnp.isfinite(trials))
    # The per-individual F / CR stash was refreshed for the next tell.
    assert jnp.all(new_state.scale_factors > 0.0)
    assert jnp.all(new_state.scale_factors <= 1.0)
    assert jnp.all(
        (new_state.crossover_rates >= 0.0) & (new_state.crossover_rates <= 1.0)
    )


def test_turning_only_rewrites_individuals_inside_ring():
    # D = 2, bounds [0, 1]: OR_init = sqrt(2)/2 ~ 0.707, IR ~ 0.0354
    # (max_fevals is large, so the outer radius stays at OR_init). The
    # two best members sit together at the origin so every possible
    # pbest gives the same ring membership: member 2 at distance 0.5 is
    # inside the ring, member 3 at ~0.014 is inside IR, member 4 at
    # sqrt(2) is outside OR.
    algo = _make_algo(popsize=5, num_dims=2)
    population = jnp.array(
        [
            [0.0, 0.0],
            [1e-4, 0.0],
            [0.5, 0.0],
            [0.01, 0.01],
            [1.0, 1.0],
        ]
    )
    fitness = jnp.array([0.0, 0.1, 1.0, 2.0, 3.0])
    base = algo.default_params.replace(x_min=0.0, x_max=1.0)
    turning = base.replace(use_turning=True, max_fevals=1e12)

    state = _init_state(algo, population, fitness, params=base)
    in_ring = jnp.array([False, False, True, False, False])

    n_diff = jnp.zeros(5, dtype=int)
    for i in range(30):
        trials_off, _ = algo.ask(key=jr.key(i), state=state, params=base)
        trials_on, _ = algo.ask(key=jr.key(i), state=state, params=turning)
        row_differs = jnp.any(trials_off != trials_on, axis=1)
        assert not jnp.any(row_differs & ~in_ring)
        n_diff = n_diff + row_differs
    # The in-ring individual must actually be rewritten (crossover can
    # mask single draws, but not all 30).
    assert int(n_diff[2]) > 0


def test_small_population_p_range_degrades_gracefully():
    # NP < 10 makes 2 / NP exceed p_max; the range collapses to a fixed
    # p instead of erroring.
    algo = _make_algo(popsize=4, num_dims=2)
    population = jr.uniform(jr.key(9), (4, 2))
    state = _init_state(algo, population, jnp.arange(4.0))
    trials, _ = algo.ask(
        key=jr.key(10), state=state, params=algo.default_params
    )
    assert trials.shape == (4, 2)
    assert jnp.all(jnp.isfinite(trials))


def test_population_size_minimum_enforced():
    with pytest.raises(AssertionError):
        _make_algo(popsize=3)


# =============================================================================
#                                                                       Factory


def test_shade_popsize_floor():
    assert shade_popsize(2) == 10
    assert shade_popsize(100_000) == variable_popsize(100_000) > 10


def test_factory_constructor_vs_params_split():
    problem = sphere_problem(2)
    update_class = shade_update(
        **problem,
        opt_hash=1,
        bounded=(0.0, 1.0),
        stop_fn=None,
        popsize=12,
        memory_size=5,
        archive_size=7,
        p_max=0.15,
    )
    assert update_class.popsize == 12

    params, _ = eqx.partition(problem["model"], eqx.is_inexact_array)
    population = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (12, *x.shape)), params
    )
    state = update_class.init_fn(population, jr.key(11))
    assert state.memory_f.shape == (5,)
    assert state.archive.shape[0] == 7


def test_factory_turning_requires_bounds_and_budget():
    problem = sphere_problem(2)
    with pytest.raises(ValueError, match="finite box bounds"):
        shade_update(
            **problem,
            opt_hash=1,
            bounded=(None, None),
            stop_fn=None,
            use_turning=True,
            max_fevals=1000.0,
        )
    with pytest.raises(ValueError, match="max_fevals"):
        shade_update(
            **problem,
            opt_hash=1,
            bounded=(0.0, 1.0),
            stop_fn=None,
            use_turning=True,
        )
    # Fully specified turning configuration constructs fine.
    shade_update(
        **problem,
        opt_hash=1,
        bounded=(0.0, 1.0),
        stop_fn=None,
        use_turning=True,
        max_fevals=1000.0,
    )


def test_first_generation_tie_does_not_pollute_memory():
    # The adapter's first step re-evaluates the initial population and
    # tells it against itself; ties must not count as successes.
    problem = sphere_problem(2)
    update_class = shade_update(
        **problem, opt_hash=1, bounded=(0.0, 1.0), stop_fn=None
    )
    params, _ = eqx.partition(problem["model"], eqx.is_inexact_array)
    population = jax.tree.map(
        lambda x: (
            jnp.broadcast_to(x, (update_class.popsize, *x.shape))
            + 0.1 * jr.normal(jr.key(12), (update_class.popsize, *x.shape))
        ),
        params,
    )
    key = jr.key(13)
    state = update_class.init_fn(population, key)
    (_, new_state, _), _ = update_class.step_fn((population, state, key), {})
    assert jnp.allclose(new_state.memory_f, 0.5)
    assert jnp.allclose(new_state.memory_cr, 0.5)
    assert int(new_state.memory_index) == 0
    assert int(new_state.archive_count) == 0
