"""The run loop on ``UpdateClass`` (l2co ADR 0017).

``init_state`` / ``step`` / ``run`` / ``batch_run`` used to be free
functions in l2co. These tests drive them on the toy tasks, so they need
neither l2co nor ``l2co_tasks``:

1. ``run`` and both multi-realization drivers honour the output contract
   (shapes, a realization axis, a running best that is the trajectory's
   minimum), and the sequential driver is a drop-in for the fused one on
   a plain optimizer.
2. ``batch_run`` routes on ``sequential_realizations``, and
   ``RandomSearchUpdateClass`` always takes the sequential driver, mapped
   over its own ``run``.
3. Random search's chunked ``run`` stays self-consistent across chunk
   splits, and ``stop_fn`` turns generations into skipped (NaN) entries.

The meta-optimizer side of the marker (``L2COUpdateClass`` /
``RL2COUpdateClass``) is asserted in l2co's and rl2co's suites.
"""

from __future__ import annotations

import dataclasses
import inspect

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
    RandomSearchUpdateClass,
    UpdateClass,
    optimizer_mapping,
)

from .toy_tasks import sphere_task

N_ITERATIONS = 6
N_REALIZATIONS = 3
DIM = 4

# float32 reduction-order tolerance between the fused and sequential
# lowerings; a plumbing bug (mis-sliced key, dropped realization axis)
# moves values by O(1), not by 1e-5.
_RTOL = 1e-4
_ATOL = 1e-5

_PLAIN = [
    ("adam", {"learning_rate": 0.05}),
    ("cmaes", {"popsize": 4}),
]
_PLAIN_IDS = [name for name, _ in _PLAIN]


def _build(name, hyperparameters, stop_fn=None, opt_hash=7):
    return optimizer_mapping(name)(
        **hyperparameters,
        task=sphere_task(DIM),
        opt_hash=opt_hash,
        bounded=(None, None),
        stop_fn=stop_fn,
    )


def _batch_inputs(update_class: UpdateClass, key):
    """Realization-axis inputs, as :func:`batch_evaluate` builds them."""
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


#                                                          output contract
# =============================================================================


@pytest.mark.parametrize(("name", "hyper"), _PLAIN, ids=_PLAIN_IDS)
def test_run_returns_the_documented_contract(name, hyper):
    update_class = _build(name, hyper)
    params, best_params, best_loss, _, batch_state, history = update_class.run(
        **_single_inputs(update_class, jr.key(0))
    )

    assert params.shape == (update_class.popsize, DIM)
    assert best_params.shape == (DIM,)
    assert isinstance(batch_state, BatchState)
    assert isinstance(history, HistoryState)
    assert history.output_min.shape == (N_ITERATIONS,)
    assert int(history.cursor) == N_ITERATIONS
    assert bool(jnp.all(history.update_step == 7))
    # The running best is the best member ever evaluated, which is at
    # most the best per-generation minimum the history records (the
    # leading sentinel generation is dropped from the history but can
    # still hold the best).
    assert float(best_loss) <= float(jnp.nanmin(history.output_min)) + 1e-6
    assert float(best_loss) == pytest.approx(
        float(sphere_task(DIM).loss_fn(best_params)), rel=1e-5
    )


@pytest.mark.parametrize(("name", "hyper"), _PLAIN, ids=_PLAIN_IDS)
def test_sequential_driver_matches_fused_driver(name, hyper):
    update_class = _build(name, hyper)
    inputs = _batch_inputs(update_class, jr.key(1))

    fused = update_class.batch_run_fused(**inputs)
    sequential = update_class.batch_run_sequential(**inputs)

    assert jax.tree.structure(fused) == jax.tree.structure(sequential)
    for a, b in zip(
        jax.tree.leaves(fused), jax.tree.leaves(sequential), strict=True
    ):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape
        assert a.shape[0] == N_REALIZATIONS
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.allclose(a, b, rtol=_RTOL, atol=_ATOL, equal_nan=True)


@pytest.mark.parametrize(("name", "hyper"), _PLAIN, ids=_PLAIN_IDS)
def test_sequential_realization_equals_its_own_run(name, hyper):
    """Realization ``r`` of the sequential map is just ``run`` on slice r."""
    update_class = _build(name, hyper)
    inputs = _batch_inputs(update_class, jr.key(2))
    batched = update_class.batch_run_sequential(**inputs)

    for r in range(N_REALIZATIONS):
        single = update_class.run(
            **{
                **inputs,
                "opt_state": jax.tree.map(
                    lambda x, r=r: x[r], inputs["opt_state"]
                ),
                "params": inputs["params"][r],
                "key": inputs["key"][r],
            }
        )
        for a, b in zip(
            jax.tree.leaves(batched), jax.tree.leaves(single), strict=True
        ):
            assert np.allclose(
                np.asarray(a)[r],
                np.asarray(b),
                rtol=_RTOL,
                atol=_ATOL,
                equal_nan=True,
            )


def test_stop_fn_skips_generations():
    """Once ``stop_fn`` fires, every generation is a NaN skip entry."""
    update_class = _build(
        "adam",
        {"learning_rate": 0.05},
        stop_fn=lambda recent_history, opt_state: jnp.array(True),
    )
    inputs = _single_inputs(update_class, jr.key(3))
    params, _, best_loss, _, _, history = update_class.run(**inputs)

    assert bool(jnp.all(jnp.isnan(history.output_min)))
    assert bool(jnp.all(jnp.isnan(params)))
    assert float(best_loss) == float("inf")


#                                                                  routing
# =============================================================================


def test_marker_is_off_for_plain_optimizers():
    assert UpdateClass.sequential_realizations is False


def test_marker_is_not_a_dataclass_field():
    """A ``ClassVar`` marker must not widen any constructor signature.

    ``dataclasses.fields`` is the check that matters:
    ``__dataclass_fields__`` also carries ``ClassVar`` pseudo-entries, so
    it would not distinguish a marker from a real field.
    """
    names = {f.name for f in dataclasses.fields(UpdateClass)}
    assert "sequential_realizations" not in names
    assert (
        "sequential_realizations"
        not in inspect.signature(UpdateClass.__init__).parameters
    )


def _probe(base: type, marker: bool):
    """An instance of ``base`` with ``sequential_realizations = marker``."""
    if base is RandomSearchUpdateClass:

        class _Probe(RandomSearchUpdateClass):
            sequential_realizations = marker

        return _Probe(
            sampling_fn=lambda *a, **k: None,
            static_model=None,
            loss_fn=lambda *a, **k: None,
            pass_rng=False,
            bounded=(None, None),
            popsize=1,
            hash=0,
        )

    class _Probe(UpdateClass):
        sequential_realizations = marker

    return _Probe(
        init_fn=lambda *a, **k: None,
        step_fn=lambda *a, **k: None,
        popsize=1,
        hash=0,
    )


@pytest.mark.parametrize(
    ("base", "marker", "expected"),
    [
        (UpdateClass, False, "batch_run_fused"),
        (UpdateClass, True, "batch_run_sequential"),
        # Random search is sequential for a different reason (peak
        # memory), so it ignores the marker either way.
        (RandomSearchUpdateClass, False, "batch_run_sequential"),
        (RandomSearchUpdateClass, True, "batch_run_sequential"),
    ],
    ids=["plain", "meta", "randomsearch", "randomsearch-and-marker"],
)
def test_batch_run_routes_on_the_marker(monkeypatch, base, marker, expected):
    called = []

    def _spy(name):
        def _driver(self, **kwargs):
            called.append(name)
            return name

        return _driver

    for name in ("batch_run_fused", "batch_run_sequential"):
        monkeypatch.setattr(UpdateClass, name, _spy(name))

    result = _probe(base, marker).batch_run(
        opt_state=None,
        params=None,
        batch_state=None,
        dataset={},
        key=None,
        n_iterations=1,
        verbose=False,
    )
    assert called == [expected]
    assert result == expected


def test_random_search_batch_run_maps_its_own_run(monkeypatch):
    """The sequential driver maps ``self.run``, so the override is used."""
    update_class = _build("randomsearch", {"sampler": "normal", "popsize": 3})
    calls = []
    original = RandomSearchUpdateClass.run

    def _counting_run(self, **kwargs):
        calls.append(1)
        return original(self, **kwargs)

    monkeypatch.setattr(RandomSearchUpdateClass, "run", _counting_run)
    update_class.batch_run(**_batch_inputs(update_class, jr.key(4)))
    # ``lax.map`` traces the body once, however many realizations it maps.
    assert calls == [1]


#                                                    random-search chunking
# =============================================================================


def test_random_search_chunked_semantics(monkeypatch):
    """Chunked random search stays self-consistent across chunk splits.

    Forcing a tiny chunk target splits a small budget into several padded
    chunks, exercising the scan, the padding mask, and the cross-chunk
    best/final reductions. ``n_iterations=7`` is deliberately not a
    multiple of the chunk size, so the last chunk carries a padded
    iteration that must be masked out.
    """
    # 8 candidates/chunk with popsize 3 -> chunk_iters=2, n_chunks=4 for
    # n_iterations=7 (padded to 8; last iteration is padding).
    monkeypatch.setattr(rs_mod, "_RS_CHUNK_TARGET_CANDIDATES", 8)
    assert rs_mod._rs_chunking(7, 3) == (4, 2)

    popsize = 3
    n_iterations = 7
    update_class = _build(
        "randomsearch", {"sampler": "normal", "popsize": popsize}
    )
    inputs = {
        **_batch_inputs(update_class, jr.key(5)),
        "n_iterations": n_iterations,
    }
    params, _, best_loss, _, _, history = update_class.batch_run(**inputs)

    assert params.shape == (N_REALIZATIONS, popsize, DIM)
    assert history.output_min.shape == (N_REALIZATIONS, n_iterations)
    assert int(history.fevals.sum()) == (
        N_REALIZATIONS * n_iterations * popsize
    )
    assert bool(jnp.all(jnp.isfinite(best_loss)))
    assert bool(jnp.all(jnp.isfinite(history.output_min)))
    for r in range(N_REALIZATIONS):
        assert float(best_loss[r]) == pytest.approx(
            float(history.output_min[r].min()), abs=1e-5
        )
