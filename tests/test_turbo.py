"""Invariants of the trust-region Bayesian optimizer.

Beyond the fixed-shape-state requirements shared with
:mod:`l2co_optimizers._src.rbf_trust_region`, two properties are specific
to this one and are what make it a different optimizer rather than a
second surrogate searcher:

the posterior conditions on what was observed
    A Gaussian process that is fitted wrong still returns candidates and
    a run still completes. What it stops doing is preferring the region
    the data favours -- so the test drives selection with two archives
    that disagree and checks the choice follows.

selection is stochastic
    Thompson sampling draws from the posterior rather than minimising a
    point estimate. Identical state under different keys must give
    different picks, or the exploration this buys is absent.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

# Local
from l2co_optimizers._src.rbf_trust_region import default_n_candidates
from l2co_optimizers._src.turbo import Params, TuRBO, _matern52

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#                                                                      Fixtures
# =============================================================================

_DIM = 6
_POP = 4


def _algo(dim: int = _DIM, **kw) -> TuRBO:
    return TuRBO(
        population_size=kw.pop("population_size", _POP),
        solution=jnp.zeros((dim,)),
        archive_size=kw.pop("archive_size", 16),
        n_candidates=kw.pop("n_candidates", 32),
    )


def _sphere(x):
    return jnp.sum(x**2, axis=-1)


def _seed(algo, key, params, dim: int = _DIM):
    pop = jax.random.normal(key, (algo.population_size, dim)) * 0.1
    return algo.init(key, pop, _sphere(pop), params)


def _drive(algo, steps, key, params=None):
    params = params or algo.default_params
    state = _seed(algo, key, params)
    for i in range(steps):
        k = jax.random.fold_in(key, i)
        pop, state = algo.ask(k, state, params)
        state, _ = algo.tell(k, pop, _sphere(pop), state, params)
    return state


#                                                              The model itself
# =============================================================================


def test_kernel_is_one_at_zero_and_decays() -> None:
    """Matern-5/2 basics, so a sign error surfaces here not downstream."""
    ell = jnp.array(1.0)
    assert float(_matern52(jnp.array(0.0), ell)) == pytest.approx(1.0)
    vals = [float(_matern52(jnp.array(r**2), ell)) for r in (0.5, 1.0, 4.0)]
    assert vals[0] > vals[1] > vals[2] > 0.0


def test_selection_follows_the_observations() -> None:
    """The posterior must steer the choice toward the better region.

    Two archives that disagree about which half of the space is good,
    everything else held equal including the key. If the GP is
    conditioning, the selected points move with the data.
    """
    algo = _algo(n_candidates=256)
    params = algo.default_params
    key = jax.random.key(0)
    base = _seed(algo, key, params)

    pts = jax.random.normal(jax.random.key(7), (algo.archive_size, _DIM))
    for sign in (+1.0, -1.0):
        # Reward points whose first coordinate matches `sign`.
        fit = -sign * pts[:, 0]
        st = base.replace(
            archive=pts,
            archive_fitness=fit,
            center=jnp.zeros((_DIM,)),
            length=jnp.array(1.0),
        )
        pop, _ = algo.ask(jax.random.key(3), st, params)
        mean_x0 = float(jnp.mean(pop[:, 0]))
        if sign > 0:
            hi = mean_x0
        else:
            lo = mean_x0
    assert hi > lo, (
        "selection did not follow the archive: the posterior is not "
        f"conditioning (got {hi:.3f} vs {lo:.3f})"
    )


def test_thompson_sampling_is_stochastic() -> None:
    """Different keys, identical state, different picks."""
    algo = _algo(n_candidates=128)
    params = algo.default_params
    state = _drive(algo, 6, jax.random.key(0))
    a, _ = algo.ask(jax.random.key(1), state, params)
    b, _ = algo.ask(jax.random.key(2), state, params)
    assert not jnp.allclose(a, b), "selection is deterministic"


def test_it_beats_random_sampling_on_a_sphere() -> None:
    """A model that is not helping is worse than sampling."""
    key = jax.random.key(0)
    algo = _algo()
    state = _drive(algo, 40, key)
    draws = jax.random.normal(jax.random.key(1), (40 * _POP, _DIM)) * 0.2
    assert float(state.best_fitness) < float(_sphere(draws).min())


#                                                        Harness compatibility
# =============================================================================


def test_state_shape_is_independent_of_iteration_count() -> None:
    """Fixed-shape state is what lets the rollout vmap realizations."""
    key = jax.random.key(0)
    algo = _algo()
    short = eqx.filter_eval_shape(lambda: _drive(algo, 3, key))
    long = eqx.filter_eval_shape(lambda: _drive(algo, 30, key))
    assert jax.tree.structure(short) == jax.tree.structure(long)
    assert [x.shape for x in jax.tree.leaves(short)] == [
        x.shape for x in jax.tree.leaves(long)
    ]


def test_archive_never_exceeds_its_capacity() -> None:
    """The observation set is a ring buffer, not a growing list."""
    algo = _algo(archive_size=8, population_size=2)
    state = _drive(algo, 25, jax.random.key(0))
    assert state.archive.shape == (8, _DIM)
    assert jnp.all(jnp.isfinite(state.archive_fitness))


def test_candidate_count_scales_with_dimension() -> None:
    """The Cholesky is fixed-cost; scoring is not, so it is capped."""
    small = TuRBO(
        population_size=2, solution=jnp.zeros((10,)), archive_size=64
    )
    large = TuRBO(
        population_size=2, solution=jnp.zeros((1024,)), archive_size=64
    )
    assert small.n_candidates == default_n_candidates(10, 64)
    assert large.n_candidates < small.n_candidates


#                                                                 Trust region
# =============================================================================


def test_length_grows_on_a_run_of_successes() -> None:
    algo = _algo()
    params = algo.default_params
    state = _seed(algo, jax.random.key(0), params)
    start = float(state.length)
    for i in range(params.success_tol + 1):
        state, _ = algo.tell(
            jax.random.key(0),
            jnp.zeros((_POP, _DIM)),
            jnp.full(_POP, 10.0 - i),
            state,
            params,
        )
    assert float(state.length) > start


def test_length_collapse_restarts_rather_than_stalls() -> None:
    algo = _algo()
    params = Params(length_init=0.8, length_min=0.2, failure_tol=1)
    state = _seed(algo, jax.random.key(0), params)
    state = state.replace(center_fitness=jnp.array(-1.0))
    for _ in range(8):
        state, _ = algo.tell(
            jax.random.key(0),
            jnp.zeros((_POP, _DIM)),
            jnp.full(_POP, 5.0),
            state,
            params,
        )
    assert float(state.length) >= params.length_min


def test_bounds_are_respected() -> None:
    algo = _algo()
    params = Params(x_min=-0.5, x_max=0.5, length_init=5.0)
    state = _seed(algo, jax.random.key(0), params)
    pop, _ = algo.ask(jax.random.key(0), state, params)
    assert float(pop.min()) >= -0.5 - 1e-6
    assert float(pop.max()) <= 0.5 + 1e-6


#                                                                     Registry
# =============================================================================


def test_registered_on_both_lookup_paths() -> None:
    """Direct use and meta-optimizer use resolve the same optimizer.

    Registering only the name map is what broke the RBF searcher: it ran
    standalone and failed as a sub-step, 768 cells into a campaign.
    """
    from l2co_optimizers._src.mapping import optimizer_mapping
    from l2co_optimizers._src.sub_optimizer import (
        CONSTRUCTOR_HYPERPARAMETERS,
        l2co_native_evosax,
    )

    assert optimizer_mapping("turbo") is not None
    assert l2co_native_evosax["turbo"] is TuRBO
    assert CONSTRUCTOR_HYPERPARAMETERS["turbo"] == (
        "archive_size",
        "n_candidates",
    )
