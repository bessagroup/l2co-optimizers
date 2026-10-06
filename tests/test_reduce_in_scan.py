"""Guards for the reduce-in-scan rollout path.

The scan drivers (:meth:`UpdateClass.run` /
:meth:`~UpdateClass.batch_run_fused`) no longer stack the full
``(n_iter, popsize, dim)`` population/gradient trajectory as scan output.
Instead each generation is reduced in place to the six per-iteration
:class:`HistoryState` fields, and the best-so-far is threaded through the
scan carry. These tests pin the two identities that equivalence rests on:

1. ``HistoryState.from_reduced`` on per-iteration ``nanmin/nanmean/nanstd``
   equals ``HistoryState.from_history`` on the full stacked trajectory.
2. Folding ``_reduce_entry`` / ``_update_best`` across the steps yields the
   same ``(best_params, best_loss)`` as ``OptHistory.flatten().retrieve_best``
   over the whole trajectory -- including NaN-padded members and a fully
   skipped (all-NaN) generation.
"""

import jax
import jax.numpy as jnp
import numpy as np

from l2co_optimizers import HistoryState, OptHistory
from l2co_optimizers._src.update_class import _reduce_entry, _update_best

N_ITER = 6
POPSIZE = 5
DIM = 4


def _make_history() -> OptHistory:
    """A stacked ``OptHistory`` with realistic NaN padding.

    Row 2 is a fully skipped generation (all-NaN loss + params, as
    ``OptHistory.init`` emits from ``skip_step``); one member of row 1 is
    NaN-padded (as the rl2co policy wrapper emits for ``[sub.popsize:]``).
    """
    rng = np.random.default_rng(0)
    loss = rng.standard_normal((N_ITER, POPSIZE))
    params = rng.standard_normal((N_ITER, POPSIZE, DIM))
    grads = rng.standard_normal((N_ITER, POPSIZE, DIM))

    loss[2, :] = np.nan
    params[2, :, :] = np.nan
    loss[1, POPSIZE - 1] = np.nan
    params[1, POPSIZE - 1, :] = np.nan

    return OptHistory(
        loss=jnp.asarray(loss),
        params=jnp.asarray(params),
        grads=jnp.asarray(grads),
        update_step=jnp.arange(N_ITER, dtype=int),
        fevals=jnp.full((N_ITER,), POPSIZE, dtype=int),
        iterations=jnp.ones((N_ITER,), dtype=int),
    )


def _eq_nan(a, b) -> bool:
    return np.allclose(np.asarray(a), np.asarray(b), equal_nan=True)


def test_from_reduced_matches_from_history():
    h = _make_history()

    reference = HistoryState.from_history(h)
    reduced = HistoryState.from_reduced(
        jnp.nanmin(h.loss, axis=1),
        jnp.nanmean(h.loss, axis=1),
        jnp.nanstd(h.loss, axis=1),
        h.update_step,
        h.fevals,
        h.iterations,
    )

    for field in (
        "output_min",
        "output_mean",
        "output_std",
        "update_step",
        "fevals",
        "iterations",
    ):
        assert _eq_nan(getattr(reference, field), getattr(reduced, field))
    assert int(reference.cursor) == int(reduced.cursor) == N_ITER


def test_running_best_matches_retrieve_best():
    h = _make_history()

    ref_params, ref_loss = h.flatten().retrieve_best()

    # Fold the per-step reduce + carry-best exactly as ``UpdateClass.run``
    # does, seeding the best with a correctly-shaped member and +inf loss.
    best_params = jax.tree.map(lambda p: p[0, 0], h.params)
    best_loss = jnp.array(jnp.inf, dtype=h.loss.dtype)
    for i in range(N_ITER):
        entry = OptHistory(
            loss=h.loss[i],
            params=h.params[i],
            grads=h.grads[i],
            update_step=h.update_step[i],
            fevals=h.fevals[i],
            iterations=h.iterations[i],
        )
        _, cand_params, cand_loss = _reduce_entry(entry)
        best_params, best_loss = _update_best(
            best_params, best_loss, cand_params, cand_loss
        )

    assert np.allclose(np.asarray(best_loss), np.asarray(ref_loss))
    assert np.allclose(np.asarray(best_params), np.asarray(ref_params))


def test_update_best_ignores_nan_and_keeps_earliest_on_tie():
    best_params = jnp.array([9.0, 9.0])
    best_loss = jnp.array(jnp.inf)

    # A NaN candidate must never displace the running best.
    best_params, best_loss = _update_best(
        best_params, best_loss, jnp.array([1.0, 1.0]), jnp.array(jnp.nan)
    )
    assert bool(jnp.isinf(best_loss))

    # First real candidate wins; an equal-loss later candidate does not
    # replace it (strict ``<`` -> earliest generation kept).
    best_params, best_loss = _update_best(
        best_params, best_loss, jnp.array([2.0, 3.0]), jnp.array(0.5)
    )
    best_params, best_loss = _update_best(
        best_params, best_loss, jnp.array([4.0, 5.0]), jnp.array(0.5)
    )
    assert float(best_loss) == 0.5
    assert np.allclose(np.asarray(best_params), [2.0, 3.0])
