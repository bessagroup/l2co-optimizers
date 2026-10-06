"""Unit tests for the front-fill semantics of ``HistoryState.add``."""

#                                                                       Modules
# =============================================================================

# Third-party
import jax.numpy as jnp

# Local
from l2co_optimizers import HistoryState, OptHistory

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def _value_state(values, n: int) -> HistoryState:
    """Build a HistoryState of length ``n`` whose first ``len(values)``
    slots match ``values`` and whose remainder is the empty padding."""
    K = len(values)
    arr = jnp.array(values, dtype=jnp.float32)
    nan_pad = jnp.full((n - K,), jnp.nan, dtype=jnp.float32)
    int_arr = jnp.arange(1, K + 1, dtype=jnp.int32)
    int_pad = jnp.zeros((n - K,), dtype=jnp.int32)

    return HistoryState(
        output_min=jnp.concatenate([arr, nan_pad]),
        output_mean=jnp.concatenate([arr, nan_pad]),
        output_std=jnp.concatenate([arr, nan_pad]),
        update_step=jnp.concatenate([int_arr, int_pad]),
        fevals=jnp.concatenate([int_arr, int_pad]),
        iterations=jnp.concatenate([int_arr, int_pad]),
        cursor=jnp.array(K, dtype=jnp.int32),
    )


def test_init_starts_with_cursor_zero_and_nan_floats():
    state = HistoryState.init(max_iterations=5)

    assert int(state.cursor) == 0
    assert state.output_min.shape == (5,)
    assert jnp.all(jnp.isnan(state.output_min))
    assert jnp.all(jnp.isnan(state.output_mean))
    assert jnp.all(jnp.isnan(state.output_std))
    assert jnp.all(state.update_step == 0)
    assert jnp.all(state.fevals == 0)
    assert jnp.all(state.iterations == 0)


def test_reset_returns_cursor_to_zero():
    state = _value_state([1.0, 2.0, 3.0], n=5)
    reset_state = state.reset()

    assert int(reset_state.cursor) == 0
    assert jnp.all(jnp.isnan(reset_state.output_min))


def test_add_into_empty_buffer_writes_at_front():
    empty = HistoryState.init(max_iterations=5)
    new = _value_state([3.0, 5.0], n=2)

    result = empty.add(new)

    expected = jnp.array([3.0, 5.0, jnp.nan, jnp.nan, jnp.nan])
    # First two slots filled, remainder still NaN.
    assert jnp.allclose(result.output_min[:2], expected[:2])
    assert jnp.all(jnp.isnan(result.output_min[2:]))
    assert int(result.cursor) == 2


def test_add_into_partially_filled_buffer_continues_front_to_back():
    # Buffer with first 3 slots filled with [2., 2., 1.], cursor at 3.
    partial = _value_state([2.0, 2.0, 1.0], n=5)
    new = _value_state([3.0, 5.0], n=2)

    result = partial.add(new)

    # Expected: [2, 2, 1, 3, 5] — exactly the example from the spec.
    assert jnp.allclose(
        result.output_min, jnp.array([2.0, 2.0, 1.0, 3.0, 5.0])
    )
    assert int(result.cursor) == 5


def test_add_preserves_int_fields():
    partial = _value_state([2.0, 2.0, 1.0], n=5)
    new = _value_state([3.0, 5.0], n=2)

    result = partial.add(new)

    # Int fields used [1, 2, 3] then [1, 2] — front-fill should give
    # [1, 2, 3, 1, 2] with cursor advancing to 5.
    assert jnp.array_equal(
        result.update_step, jnp.array([1, 2, 3, 1, 2], dtype=jnp.int32)
    )
    assert jnp.array_equal(
        result.fevals, jnp.array([1, 2, 3, 1, 2], dtype=jnp.int32)
    )


def test_from_history_sets_cursor_to_full_length():
    # Build a tiny OptHistory of shape (n_iter=4, popsize=2).
    n_iter, popsize = 4, 2
    loss = jnp.arange(n_iter * popsize, dtype=jnp.float32).reshape(
        n_iter, popsize
    )
    update_step = jnp.arange(n_iter, dtype=jnp.int32)
    fevals = jnp.full((n_iter,), popsize, dtype=jnp.int32)
    iterations = jnp.ones((n_iter,), dtype=jnp.int32)

    history = OptHistory(
        loss=loss,
        params=jnp.zeros((n_iter, popsize, 1), dtype=jnp.float32),
        grads=jnp.zeros((n_iter, popsize, 1), dtype=jnp.float32),
        update_step=update_step,
        fevals=fevals,
        iterations=iterations,
    )

    state = HistoryState.from_history(history)

    assert int(state.cursor) == n_iter
    assert state.output_min.shape == (n_iter,)
    # output_min reduces over the population (axis=1).
    assert jnp.allclose(state.output_min, jnp.min(loss, axis=1))


def test_chained_adds_preserve_insertion_order():
    state = HistoryState.init(max_iterations=6)
    state = state.add(_value_state([10.0], n=1))
    state = state.add(_value_state([20.0, 30.0], n=2))
    state = state.add(_value_state([40.0], n=1))

    assert int(state.cursor) == 4
    assert jnp.allclose(
        state.output_min[:4], jnp.array([10.0, 20.0, 30.0, 40.0])
    )
    assert jnp.all(jnp.isnan(state.output_min[4:]))
