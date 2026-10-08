"""``relative_normal_sampling`` starts runs from the model (ADR 0004).

Member 0 is the parameters themselves; every other member is spread
around them by ``std * max(|x|, 1)`` per coordinate. Checked on its own
and through ``reset``, which draws every run's starting population.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# Local
from l2co_optimizers import (
    BatchState,
    OptimizationStep,
    RunState,
    get_sampler,
    optimizer_mapping,
    relative_normal_sampling,
    reset,
)

from .toy_problems import _sphere, quadratic_problem

# =============================================================================

#: Coordinates across scales, the way CUTEst's ``x0`` mixes them: large,
#: negative, zero and tiny.
_X = jnp.array([1.0e4, -3.0, 0.0, 1.0e-3])
_N = 4000


def test_member_zero_is_the_parameters_exactly():
    params = {"w": _X, "b": jnp.array(2.5)}
    samples = relative_normal_sampling(jr.key(0), params, 5)
    assert np.array_equal(samples["w"][0], params["w"])
    assert np.array_equal(samples["b"][0], params["b"])


def test_samples_have_a_leading_axis_and_keep_dtype():
    params = {"w": _X, "m": jnp.ones((2, 3), dtype=jnp.float32)}
    samples = relative_normal_sampling(jr.key(0), params, 7)
    assert samples["w"].shape == (7, 4)
    assert samples["m"].shape == (7, 2, 3)
    assert samples["m"].dtype == jnp.float32


def test_spread_is_relative_to_each_coordinate():
    """std * max(|x|, 1): 1e4, 3, 1 and 1 for the coordinates of _X."""
    std = 0.5
    rest = relative_normal_sampling(jr.key(1), _X, _N, std=std)[1:]
    expected = std * np.maximum(np.abs(np.asarray(_X)), 1.0)
    np.testing.assert_allclose(np.std(rest, axis=0), expected, rtol=0.06)
    np.testing.assert_allclose(
        np.mean(rest, axis=0), _X, atol=float(4 * expected.max() / _N**0.5)
    )


def test_zero_std_puts_every_member_at_the_parameters():
    samples = relative_normal_sampling(jr.key(2), _X, 6, std=0.0)
    assert np.array_equal(samples, jnp.tile(_X, (6, 1)))


def test_deterministic_in_the_key():
    first = relative_normal_sampling(jr.key(3), _X, 5)
    assert np.array_equal(first, relative_normal_sampling(jr.key(3), _X, 5))
    assert not np.array_equal(
        first, relative_normal_sampling(jr.key(4), _X, 5)
    )


def test_one_sample_is_the_parameters():
    assert np.array_equal(relative_normal_sampling(jr.key(5), _X, 1), _X[None])


def test_reachable_by_name():
    assert get_sampler("relative_normal") is relative_normal_sampling


@pytest.mark.parametrize(
    ("step", "popsize"),
    [
        (OptimizationStep("adam", hyperparameters={"learning_rate": 0.1}), 1),
        (
            OptimizationStep(
                "differentialevolution", hyperparameters={"popsize": 6}
            ),
            6,
        ),
    ],
    ids=["adam", "differentialevolution"],
)
def test_reset_starts_the_population_at_the_model(step, popsize):
    """Through ``reset``, which draws every run's starting population."""
    problem = quadratic_problem(_X, _sphere)
    update_class = optimizer_mapping(step.optimizer)(
        **step.hyperparameters,
        **problem,
        opt_hash=step.hash,
        bounded=(None, None),
        stop_fn=step.stopping_fn,
    )
    run_state = RunState.init(
        update_class,
        model=problem["model"],
        dataset={},
        batch_size=None,
        key=jr.key(0),
    )
    static = eqx.filter(problem["model"], eqx.is_inexact_array, inverse=True)
    reset_state = reset(
        run_state,
        BatchState.init(dataset={}, batch_size=None, key=jr.key(0)),
        static,
        {},
        problem["loss_fn"],
        problem["pass_rng"],
        relative_normal_sampling,
        jr.key(6),
    )
    population = jax.tree.leaves(reset_state.params)[0]
    assert population.shape == (popsize, _X.size)
    assert np.array_equal(population[0], _X)
    if popsize > 1:
        # The other members are spread, so DE has differences to use.
        assert float(jnp.min(jnp.std(population, axis=0))) > 0.0
