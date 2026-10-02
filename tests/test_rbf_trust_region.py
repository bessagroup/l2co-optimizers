"""Invariants of the RBF trust-region searcher.

The algorithm is only useful if three things hold, and each is easy to
break silently:

it optimizes
    A surrogate that is fitted wrong still returns candidates, and a
    run still completes -- it just stops beating random sampling. That
    is the failure this catches.

state stays fixed-shape
    The rollout harness runs optimizers under ``eqx.filter_vmap`` with a
    fixed-length scan, so any state whose shape depends on the iteration
    count breaks batching rather than raising. The archive is a ring
    buffer for exactly this reason.

the trust region actually adapts
    Radius growth on repeated success and collapse-then-restart on
    repeated failure are what separate this from a fixed-width random
    search around the incumbent.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from l2co_optimizers._src.rbf_trust_region import (
    Params,
    RBFTrustRegion,
    _mean_sq_dist,
)

#                                                                      Fixtures
# =============================================================================

_DIM = 6
_POP = 4


def _algo(**kw) -> RBFTrustRegion:
    return RBFTrustRegion(
        population_size=_POP,
        solution=jnp.zeros((_DIM,)),
        archive_size=kw.pop("archive_size", 12),
        n_candidates=kw.pop("n_candidates", 32),
    )


def _sphere(x):
    return jnp.sum(x**2, axis=-1)


def _seed_state(algo, key, params, dim=_DIM):
    """Initial state from one seed population, as evosax expects."""
    pop = jax.random.normal(key, (algo.population_size, dim)) * 0.1
    return algo.init(key, pop, _sphere(pop), params)


def _drive(algo, steps, key, params=None):
    """Run an ask/tell loop on the sphere, returning the final state."""
    params = params or algo.default_params
    state = _seed_state(algo, key, params)
    for i in range(steps):
        k = jax.random.fold_in(key, i)
        pop, state = algo.ask(k, state, params)
        state, _ = algo.tell(k, pop, _sphere(pop), state, params)
    return state


#                                                                         Tests
# =============================================================================


def test_it_beats_random_sampling_on_a_sphere() -> None:
    """The point of a surrogate is to beat sampling at equal budget."""
    key = jax.random.key(0)
    algo = _algo()
    state = _drive(algo, 40, key)

    budget = 40 * _POP
    draws = jax.random.normal(jax.random.key(1), (budget, _DIM)) * 0.2
    assert float(state.best_fitness) < float(_sphere(draws).min()), (
        "surrogate search did not beat random sampling at equal budget"
    )


def test_state_shape_is_independent_of_iteration_count() -> None:
    """Fixed-shape state is what lets the harness vmap the rollout."""
    key = jax.random.key(0)
    algo = _algo()
    short = eqx.filter_eval_shape(lambda: _drive(algo, 3, key))
    long = eqx.filter_eval_shape(lambda: _drive(algo, 30, key))
    assert jax.tree.structure(short) == jax.tree.structure(long)
    shapes_s = [x.shape for x in jax.tree.leaves(short)]
    shapes_l = [x.shape for x in jax.tree.leaves(long)]
    assert shapes_s == shapes_l


def test_archive_never_exceeds_its_capacity() -> None:
    """The ring buffer must overwrite, not grow."""
    key = jax.random.key(0)
    algo = _algo(archive_size=8)
    state = _drive(algo, 25, key)  # 25 * 4 evals >> capacity 8
    assert state.archive.shape == (8, _DIM)
    assert state.archive_fitness.shape == (8,)
    assert jnp.all(jnp.isfinite(state.archive_fitness)), (
        "a full ring buffer should hold no unwritten slots"
    )


def test_radius_grows_on_a_run_of_successes() -> None:
    """Consecutive improvement must widen the trust region."""
    key = jax.random.key(0)
    algo = _algo()
    params = algo.default_params
    state = _seed_state(algo, key, params)
    start = float(state.radius)
    # Feed strictly improving fitness so every tell is a success.
    for i in range(params.success_tol + 1):
        pop = jnp.zeros((_POP, _DIM))
        state, _ = algo.tell(key, pop, jnp.full(_POP, 10.0 - i), state, params)
    assert float(state.radius) > start


def test_radius_collapse_triggers_a_restart_not_a_stall() -> None:
    """At the floor the radius resets, so the search does not freeze."""
    key = jax.random.key(0)
    algo = _algo()
    params = Params(radius_init=0.2, radius_min=0.05, failure_tol=1)
    state = _seed_state(algo, key, params)
    state = state.replace(center_fitness=jnp.array(-1.0))
    for _ in range(6):  # every tell fails
        pop = jnp.zeros((_POP, _DIM))
        state, _ = algo.tell(key, pop, jnp.full(_POP, 5.0), state, params)
    assert float(state.radius) >= params.radius_min


def test_an_empty_archive_still_produces_distinct_candidates() -> None:
    """Step one has no model; it must still explore, not return zeros."""
    key = jax.random.key(0)
    algo = _algo()
    params = algo.default_params
    state = _seed_state(algo, key, params)
    pop, _ = algo.ask(key, state, params)
    assert pop.shape == (_POP, _DIM)
    assert jnp.all(jnp.isfinite(pop))
    assert float(jnp.abs(pop).max()) > 0.0, "all candidates sit on the centre"


def test_bounds_are_respected() -> None:
    """Candidates are clipped into the box the run declares."""
    key = jax.random.key(0)
    algo = _algo()
    params = Params(x_min=-0.5, x_max=0.5, radius_init=5.0)
    state = _seed_state(algo, key, params)
    pop, _ = algo.ask(key, state, params)
    assert float(pop.min()) >= -0.5 - 1e-6
    assert float(pop.max()) <= 0.5 + 1e-6


@pytest.mark.parametrize("dim", [2, 64])
def test_distance_scale_is_dimension_robust(dim: int) -> None:
    """Per-coordinate averaging keeps one lengthscale usable at any d.

    Summed squared distance grows with d, which would make a single
    kernel lengthscale wrong everywhere except the dimension it was
    tuned at.
    """
    key = jax.random.key(0)
    a = jax.random.normal(key, (5, dim))
    b = jax.random.normal(jax.random.key(1), (7, dim))
    d2 = _mean_sq_dist(a, b)
    assert d2.shape == (5, 7)
    assert float(d2.min()) >= 0.0
    # Two independent standard normals differ by ~2 per coordinate.
    assert 1.0 < float(d2.mean()) < 3.0


def test_the_solve_is_independent_of_dimensionality() -> None:
    """No polynomial tail: the kernel system is archive-sized, not d-sized."""
    for dim in (4, 256):
        algo = RBFTrustRegion(
            population_size=2,
            solution=jnp.zeros((dim,)),
            archive_size=9,
            n_candidates=16,
        )
        params = algo.default_params
        state = _seed_state(algo, jax.random.key(0), params, dim=dim)
        assert state.archive.shape == (9, dim)
        assert state.archive_fitness.shape == (9,)


def test_the_registry_resolves_every_spelling_of_the_name() -> None:
    """Both lookup paths must agree on how a name is keyed.

    ``RunState.init`` resolves the raw name a config wrote;
    ``sub_optimizer.resolve_popsize`` resolves one already put through
    ``normalize_key``, which strips underscores. Storing the key raw made
    this optimizer usable directly but not as a meta-optimizer sub-step,
    and that failed at run time -- 768 cells into a campaign -- rather
    than at registration.
    """
    from l2co_optimizers._src.mapping import optimizer_mapping
    from l2co_optimizers._src.utils import normalize_key

    for spelling in ("rbf_trust_region", "rbftrustregion", "RBF_Trust_Region"):
        assert optimizer_mapping(spelling) is optimizer_mapping(
            normalize_key(spelling)
        )


def test_every_registry_key_is_already_normalized() -> None:
    """A raw key would be unreachable from the normalizing call path."""
    from l2co_optimizers._src.mapping import optimizers
    from l2co_optimizers._src.utils import normalize_key

    unreachable = [k for k in optimizers if normalize_key(k) != k]
    assert not unreachable, (
        f"these keys cannot be resolved after normalize_key: {unreachable}"
    )


#                                            Candidate count vs dimensionality
# =============================================================================


def test_candidate_count_holds_a_flop_budget() -> None:
    """Scoring cost stays flat as the dimension grows.

    Scoring is the only part of a step that scales with dimensionality
    (``n_candidates * archive_size * num_dims``), and a flat count is
    what made a 1024-dimensional cell take an hour.
    """
    from l2co_optimizers._src.rbf_trust_region import (
        CANDIDATE_FLOPS,
        N_CANDIDATES_MIN,
        default_n_candidates,
    )

    archive = 32
    for dim in (256, 512, 1024):
        n = default_n_candidates(dim, archive)
        if n > N_CANDIDATES_MIN:  # not yet clamped by the floor
            assert n * archive * dim <= CANDIDATE_FLOPS


def test_low_dimensions_are_unchanged() -> None:
    """The cap still binds up to 256 dimensions.

    Results already measured at 2-256 dimensions must not move when this
    default changes, so the resolved count there has to stay 128.
    """
    from l2co_optimizers._src.rbf_trust_region import (
        N_CANDIDATES_MAX,
        default_n_candidates,
    )

    for dim in (2, 10, 40, 64, 256):
        assert default_n_candidates(dim, 32) == N_CANDIDATES_MAX


def test_high_dimensions_shrink_but_do_not_vanish() -> None:
    """A floor keeps the surrogate minimisation meaningful."""
    from l2co_optimizers._src.rbf_trust_region import (
        N_CANDIDATES_MAX,
        N_CANDIDATES_MIN,
        default_n_candidates,
    )

    assert default_n_candidates(1024, 32) < N_CANDIDATES_MAX
    for dim in (1024, 4096, 100_000):
        assert default_n_candidates(dim, 32) >= N_CANDIDATES_MIN


def test_the_algorithm_resolves_its_own_candidate_count() -> None:
    """Constructing without the argument picks the count from the task."""
    small = RBFTrustRegion(
        population_size=2, solution=jnp.zeros((10,)), archive_size=32
    )
    large = RBFTrustRegion(
        population_size=2, solution=jnp.zeros((1024,)), archive_size=32
    )
    assert small.n_candidates > large.n_candidates
    assert small.n_candidates == 128


def test_an_explicit_candidate_count_still_wins() -> None:
    """The scaling is a default, not a policy."""
    algo = RBFTrustRegion(
        population_size=2,
        solution=jnp.zeros((1024,)),
        archive_size=32,
        n_candidates=256,
    )
    assert algo.n_candidates == 256


def test_a_resolved_count_below_the_population_is_rejected() -> None:
    """The floor must not silently drop below what a step evaluates."""
    # archive_size must clear population_size first, or that guard fires
    # instead and the test passes for the wrong reason.
    with pytest.raises(ValueError, match="n_candidates"):
        RBFTrustRegion(
            population_size=64,
            solution=jnp.zeros((100_000,)),
            archive_size=128,
        )
