"""The noise-free series in the run history (spec decision 12).

A noisy task is scored on its noise-free value, so the run records, per
step, the lowest noise-free loss over the evaluated population next to
the observed one. ``UpdateClass.true_loss_fn`` supplies that loss for one
member's parameters; ``None`` (the default) means the task is noise-free
and the observed value already is the noise-free one.

The tests use an observed loss of ``sphere + 1`` with a noise-free loss of
``sphere``. Both rank the population the same way, so the noise-free
series must equal the observed series minus one, step for step.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import l2co_optimizers._src.random_search as rs_mod
from l2co_optimizers import (
    BatchState,
    HistoryState,
    UpdateClass,
    optimizer_mapping,
)

from .toy_problems import _sphere, quadratic_problem

DIM = 4
N_ITERATIONS = 6
N_REALIZATIONS = 3


def _observed(x, **_):
    return _sphere(x) + 1.0


def _noise_free(params):
    return _sphere(params)


_STEPS = [
    ("adam", {"learning_rate": 0.05}),
    ("cmaes", {"popsize": 4}),
    ("randomsearch", {"sampler": "normal", "popsize": 3}),
]
_IDS = [name for name, _ in _STEPS]


def _build(name, hyperparameters, true_loss_fn=_noise_free) -> UpdateClass:
    update_class = optimizer_mapping(name)(
        **hyperparameters,
        **quadratic_problem(jnp.zeros(DIM), _observed),
        opt_hash=7,
        bounded=(None, None),
        stop_fn=None,
    )
    return update_class.with_true_loss_fn(true_loss_fn)


def _batch_inputs(update_class: UpdateClass, key):
    init_key, params_key = jr.split(key)
    keys = jr.split(init_key, N_REALIZATIONS)
    batch_state = BatchState.init(dataset={}, batch_size=None, key=key)
    params = jr.normal(params_key, (N_REALIZATIONS, update_class.popsize, DIM))
    opt_state = eqx.filter_vmap(
        update_class.init_state, in_axes=(0, None, None, 0)
    )(params, batch_state, {}, keys)
    return dict(
        opt_state=opt_state,
        params=params,
        batch_state=batch_state,
        dataset={},
        key=keys,
        n_iterations=N_ITERATIONS,
        verbose=False,
    )


def _single_inputs(update_class: UpdateClass, key):
    inputs = _batch_inputs(update_class, key)
    return {
        **inputs,
        "opt_state": jax.tree.map(lambda x: x[0], inputs["opt_state"]),
        "params": inputs["params"][0],
        "key": inputs["key"][0],
    }


def _history(result):
    return result[-1]


#                                                              HistoryState
# =============================================================================


def test_history_state_has_no_noise_free_series_by_default():
    assert HistoryState.init(max_iterations=5).output_min_true is None
    reduced = HistoryState.from_reduced(
        jnp.ones(3),
        jnp.ones(3),
        jnp.zeros(3),
        jnp.ones(3, int),
        jnp.ones(3, int),
        jnp.ones(3, int),
    )
    assert reduced.output_min_true is None


def test_from_reduced_keeps_the_noise_free_series():
    state = HistoryState.from_reduced(
        jnp.array([3.0, 2.0]),
        jnp.ones(2),
        jnp.zeros(2),
        jnp.ones(2, int),
        jnp.ones(2, int),
        jnp.ones(2, int),
        output_min_true=jnp.array([2.5, 1.5]),
    )
    np.testing.assert_array_equal(state.output_min_true, [2.5, 1.5])


def _delta(observed, noise_free=None):
    n = len(observed)
    return HistoryState.from_reduced(
        jnp.asarray(observed),
        jnp.ones(n),
        jnp.zeros(n),
        jnp.ones(n, int),
        jnp.ones(n, int),
        jnp.ones(n, int),
        output_min_true=(
            None if noise_free is None else jnp.asarray(noise_free)
        ),
    )


def test_add_without_noise_free_values_stays_none():
    state = HistoryState.init(max_iterations=4).add(_delta([3.0, 2.0]))
    assert state.output_min_true is None


def test_add_mixes_in_the_observed_value_where_none_is_given():
    """A step without a noise-free value uses its observed value.

    rl2co fills one history from several runs, and its episode's first
    entry has no noise-free value. Where noise-free values exist they
    are written; elsewhere the observed value stands in.
    """
    state = HistoryState.init(max_iterations=4)
    state = state.add(_delta([5.0]))  # no noise-free value
    state = state.add(_delta([3.0, 2.0], noise_free=[2.5, 1.5]))
    np.testing.assert_array_equal(state.output_min_true[:3], [5.0, 2.5, 1.5])
    assert np.isnan(np.asarray(state.output_min_true[3]))
    state = state.add(_delta([1.0]))  # no noise-free value again
    np.testing.assert_array_equal(state.output_min_true, [5.0, 2.5, 1.5, 1.0])


#                                                                       run
# =============================================================================


@pytest.mark.parametrize(("name", "hp"), _STEPS, ids=_IDS)
def test_run_records_the_noise_free_minimum_per_step(name, hp):
    update_class = _build(name, hp)
    history = _history(
        update_class.run(**_single_inputs(update_class, jr.key(3)))
    )
    assert history.output_min_true.shape == (N_ITERATIONS,)
    np.testing.assert_allclose(
        history.output_min_true, history.output_min - 1.0, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(("name", "hp"), _STEPS, ids=_IDS)
def test_batch_run_records_the_noise_free_minimum_per_step(name, hp):
    update_class = _build(name, hp)
    history = _history(
        update_class.batch_run(**_batch_inputs(update_class, jr.key(3)))
    )
    assert history.output_min_true.shape == (N_REALIZATIONS, N_ITERATIONS)
    np.testing.assert_allclose(
        history.output_min_true, history.output_min - 1.0, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(("name", "hp"), _STEPS, ids=_IDS)
def test_noise_free_tasks_record_no_separate_series(name, hp):
    update_class = _build(name, hp, true_loss_fn=None)
    history = _history(
        update_class.run(**_single_inputs(update_class, jr.key(3)))
    )
    assert history.output_min_true is None


def test_random_search_noise_free_series_survives_chunking(monkeypatch):
    """Padded chunks: the noise-free series covers exactly n_iterations."""
    monkeypatch.setattr(rs_mod, "_RS_CHUNK_TARGET_CANDIDATES", 8)
    update_class = _build("randomsearch", {"sampler": "normal", "popsize": 3})
    inputs = {**_single_inputs(update_class, jr.key(5)), "n_iterations": 7}
    history = _history(update_class.run(**inputs))
    assert history.output_min_true.shape == (7,)
    np.testing.assert_allclose(
        history.output_min_true, history.output_min - 1.0, rtol=1e-5, atol=1e-5
    )


def test_with_true_loss_fn_leaves_the_original_unchanged():
    original = _build("cmaes", {"popsize": 4}, true_loss_fn=None)
    copied = original.with_true_loss_fn(_noise_free)
    assert original.true_loss_fn is None
    assert copied.true_loss_fn is _noise_free
    assert copied.popsize == original.popsize


def test_whole_run_scipy_entries_refuse_a_noise_free_loss():
    """The whole-run driver cannot record a noise-free series."""
    update_class = _build("cobyla", {})
    with pytest.raises(NotImplementedError, match="noise-free"):
        update_class.run(**_single_inputs(update_class, jr.key(3)))
