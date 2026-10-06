"""Unit tests for the menu-dispatched population evaluator.

Covers the four properties ``docs/adr/0011`` commits to: branch
deduplication and per-optimizer evaluation widths, the NaN contract,
numerical equivalence with the undispatched full-width evaluation, and
``vmap`` lane semantics (the batched driver's path).

The tests use synthetic ``SubOpt`` adapters and a trivial ``loss_fn``
rather than real tasks: only ``is_pop`` and ``popsize`` are read from
the menu, so nothing here needs an optimizer to actually run.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

# Local
from l2co_optimizers import (
    GRAD_EVALUATION_WIDTH,
    SubOpt,
    menu_loss_and_grad,
    vmapped_loss,
    vmapped_loss_and_grad,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

POPSIZE = 27
DIM = 6


def _sub(is_pop: bool, popsize: int) -> SubOpt:
    """A menu entry carrying only the fields the evaluator reads."""
    return SubOpt(
        is_pop=is_pop,
        popsize=popsize,
        name_hash=0,
        init_fn=lambda params, key: None,
        step_fn=lambda *a, **k: None,
        tell_fn=None,
    )


# The canonical conf/optimizers/l2co.yaml menu: sepcmaes, adam, lbfgs,
# pso -- two population entries at the same popsize, two gradient ones.
CANONICAL_MENU = (
    _sub(True, POPSIZE),
    _sub(False, 1),
    _sub(False, 1),
    _sub(True, POPSIZE),
)


class Model(eqx.Module):
    """Minimal task model: one differentiable vector of parameters."""

    w: jax.Array


def _loss(model: Model) -> jax.Array:
    """Smooth, per-candidate-independent scalar loss."""
    return jnp.sum(model.w**2) + jnp.sum(jnp.sin(model.w))


def _population(key, n=POPSIZE):
    """A population of ``n`` distinct finite candidates."""
    return Model(w=jr.normal(key, (n, DIM)))


def _split(model):
    """Partition a model into (population, static) as the bridge does."""
    return eqx.partition(model, eqx.is_inexact_array)


def _build(menu=CANONICAL_MENU, population_size=POPSIZE, pass_rng=False):
    """Build an evaluator over ``menu`` for the test model."""
    _, static = _split(_population(jr.key(0), population_size))
    return menu_loss_and_grad(
        sub_optimizers=menu,
        population_size=population_size,
        static=static,
        loss_fn=_loss,
        pass_rng=pass_rng,
    )


#                                                     Dedup and branch widths
# =============================================================================


def test_canonical_menu_collapses_to_two_branches():
    """Four menu entries, two distinct (kind, width) pairs."""
    evaluate = _build()
    assert evaluate.branches == (("pop", POPSIZE), ("grad", 1))
    # sepcmaes and pso share the pop branch; adam and lbfgs the grad one.
    assert evaluate.menu_to_branch == (0, 1, 1, 0)


def test_distinct_popsizes_get_distinct_branches():
    """Deduplication merges only genuinely identical widths."""
    menu = (_sub(True, 27), _sub(True, 15), _sub(False, 1), _sub(True, 27))
    evaluate = _build(menu=menu)
    assert evaluate.branches == (
        ("pop", 27),
        ("pop", 15),
        ("grad", 1),
    )
    assert evaluate.menu_to_branch == (0, 1, 2, 0)


def test_grad_entries_use_the_grad_evaluation_width():
    """Gradient branches evaluate ``GRAD_EVALUATION_WIDTH`` rows."""
    evaluate = _build(menu=(_sub(False, 1), _sub(True, POPSIZE)))
    kinds = dict(evaluate.branches)
    assert kinds["grad"] == GRAD_EVALUATION_WIDTH


def test_single_branch_menu_skips_the_switch():
    """An all-gradient menu compiles exactly one branch."""
    evaluate = _build(menu=(_sub(False, 1), _sub(False, 1)), population_size=1)
    assert evaluate.branches == (("grad", 1),)
    assert evaluate.menu_to_branch == (0, 0)


def test_population_size_below_widest_is_rejected():
    """The generation width must cover the widest menu entry."""
    with pytest.raises(ValueError, match="below the widest"):
        _build(menu=(_sub(True, POPSIZE),), population_size=4)


def test_empty_menu_is_rejected():
    """An empty menu has no branch to dispatch to."""
    with pytest.raises(ValueError, match="non-empty"):
        _build(menu=())


#                                                                 NaN contract
# =============================================================================


def test_padding_rows_come_back_nan_on_a_grad_generation():
    """A gradient generation reports only the iterate."""
    evaluate = _build()
    pop, _ = _split(_population(jr.key(1)))
    keys = jr.split(jr.key(2), POPSIZE)

    loss, grads = evaluate(jnp.array(1), pop, {}, keys)

    assert jnp.isfinite(loss[0])
    assert bool(jnp.all(jnp.isnan(loss[1:])))
    assert bool(jnp.all(jnp.isfinite(grads.w[0])))
    assert bool(jnp.all(jnp.isnan(grads.w[1:])))


def test_pop_generation_reports_all_rows_and_nan_grads():
    """A population generation evaluates its full popsize, no grads."""
    evaluate = _build()
    pop, _ = _split(_population(jr.key(1)))
    keys = jr.split(jr.key(2), POPSIZE)

    loss, grads = evaluate(jnp.array(0), pop, {}, keys)

    assert bool(jnp.all(jnp.isfinite(loss)))
    # Population optimizers consume no gradients; matching what l2co's
    # native population factories record in OptHistory.
    assert bool(jnp.all(jnp.isnan(grads.w)))


def test_diverged_candidate_is_masked_even_when_loss_is_finite():
    """A row with NaN params yields NaN loss regardless of loss_fn.

    The mask exists because a ``loss_fn`` that clips or masks can
    return a finite value from NaN inputs, and best-tracking must never
    adopt such a candidate.
    """
    _, static = _split(_population(jr.key(0)))

    def clipping_loss(model: Model) -> jax.Array:
        """Returns a finite value even for NaN parameters."""
        return jnp.sum(jnp.nan_to_num(model.w, nan=1.0) ** 2)

    evaluate = menu_loss_and_grad(
        sub_optimizers=CANONICAL_MENU,
        population_size=POPSIZE,
        static=static,
        loss_fn=clipping_loss,
        pass_rng=False,
    )

    pop, _ = _split(_population(jr.key(1)))
    pop = eqx.tree_at(lambda p: p.w, pop, pop.w.at[3].set(jnp.nan))
    keys = jr.split(jr.key(2), POPSIZE)

    loss, _grads = evaluate(jnp.array(0), pop, {}, keys)

    # Row 3 would have been finite without the mask.
    assert bool(jnp.isnan(loss[3]))
    assert bool(jnp.isfinite(loss[0]))
    assert bool(jnp.all(jnp.isfinite(jnp.delete(loss, 3, axis=0))))


#                                                                  Equivalence
# =============================================================================


def test_pop_generation_matches_a_full_width_evaluation():
    """A population generation evaluates every row of its popsize.

    Compared to float32 tolerance rather than exactly, and deliberately
    so. Dispatching through ``lax.switch`` emits a different HLO graph
    than a bare full-width call — there is a slice (a no-op on the
    values, *not* on the graph) and a conditional — and XLA may fuse the
    two differently. On x86_64 Linux the results are bit-identical; on
    macOS/arm64 they differ by up to one float32 ulp. Both full-width
    references agree with *each other* there, so the difference comes
    from the dispatch restructuring the graph, not from dropping the
    backward pass.

    ``rtol=1e-6`` is ~8 float32 ulp: loose enough for fusion noise,
    tight enough to catch evaluating the wrong rows, the wrong number of
    rows, or under the wrong keys.

    Consequence recorded in ``docs/adr/0011``: **no** menu is guaranteed
    to bit-reproduce runs predating the dispatch.
    """
    evaluate = _build()
    pop, static = _split(_population(jr.key(1)))
    keys = jr.split(jr.key(2), POPSIZE)

    loss, _ = evaluate(jnp.array(0), pop, {}, keys)
    value_only = vmapped_loss(eqx.combine(pop, static), _loss, {})
    old_path, _ = vmapped_loss_and_grad(eqx.combine(pop, static), _loss, {})

    # Every row is really evaluated -- no silent truncation to slot 0.
    assert bool(jnp.all(jnp.isfinite(loss)))
    assert loss.shape == (POPSIZE,)
    assert bool(jnp.allclose(loss, value_only, rtol=1e-6, atol=0.0))
    assert bool(jnp.allclose(loss, old_path, rtol=1e-6, atol=0.0))


def test_grad_iterate_matches_full_width_within_tolerance():
    """Row-0 gradient at width 1 vs the full-width backward pass.

    This test *pins* the ``GRAD_EVALUATION_WIDTH == 1`` decision from
    ``docs/adr/0011``. Width 1 takes XLA's degenerate-vmap path and
    differs by ~4e-6 relative; every width >= 2 is bit-exact. If this
    assertion is tightened to exact equality, the width decision has
    been changed and the ADR needs revisiting.
    """
    evaluate = _build()
    pop, static = _split(_population(jr.key(1)))
    keys = jr.split(jr.key(2), POPSIZE)

    _, grads = evaluate(jnp.array(1), pop, {}, keys)
    _, reference = vmapped_loss_and_grad(eqx.combine(pop, static), _loss, {})

    scale = jnp.max(jnp.abs(reference.w[0]))
    rel = jnp.max(jnp.abs(grads.w[0] - reference.w[0])) / scale
    assert float(rel) < 1e-5


def test_pass_rng_routes_each_row_its_own_key():
    """Row j is evaluated under eval_keys[j], as full width would be."""
    _, static = _split(_population(jr.key(0)))

    def noisy_loss(model: Model, key) -> jax.Array:
        """Loss whose value depends on the per-row key."""
        return jnp.sum(model.w**2) + jr.normal(key)

    evaluate = menu_loss_and_grad(
        sub_optimizers=CANONICAL_MENU,
        population_size=POPSIZE,
        static=static,
        loss_fn=noisy_loss,
        pass_rng=True,
    )

    pop, _ = _split(_population(jr.key(1)))
    keys = jr.split(jr.key(2), POPSIZE)

    loss, _ = evaluate(jnp.array(0), pop, {}, keys)
    per_row = jnp.array(
        [
            noisy_loss(
                eqx.combine(jax.tree.map(lambda p, j=i: p[j], pop), static),
                keys[i],
            )
            for i in range(POPSIZE)
        ]
    )
    assert bool(jnp.allclose(loss, per_row))


#                                                               vmap semantics
# =============================================================================


def test_vmap_over_active_index_matches_per_lane_results():
    """Each realization's lane equals its unvmapped result.

    ``batch_run_`` vmaps ``step`` over ``n_realizations``, so the
    dispatch index is per-realization and ``lax.switch`` lowers to a
    ``select`` over every branch. Correctness must survive that; the
    cost of it is why branches are deduplicated.

    The NaN *pattern* must match exactly -- that is what says each lane
    got its own optimizer's evaluation width. Finite values are only
    compared to float32 tolerance: vmapping widens the batch, which
    retiles reductions and moves the last bits. Bit-exactness across a
    ``vmap`` boundary is not a property this evaluator claims.
    """
    evaluate = _build()
    n_real = 4
    pops = [_split(_population(jr.key(10 + i)))[0] for i in range(n_real)]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *pops)
    indices = jnp.array([0, 1, 2, 3])  # sepcmaes, adam, lbfgs, pso
    keys = jnp.stack(
        [jr.split(jr.key(20 + i), POPSIZE) for i in range(n_real)]
    )

    batched = jax.vmap(evaluate, in_axes=(0, 0, None, 0))
    loss_b, grads_b = batched(indices, stacked, {}, keys)

    for i in range(n_real):
        loss_i, grads_i = evaluate(indices[i], pops[i], {}, keys[i])
        # Which rows are masked is exact; the values are float32-close.
        assert bool(jnp.array_equal(jnp.isnan(loss_b[i]), jnp.isnan(loss_i)))
        assert bool(
            jnp.array_equal(jnp.isnan(grads_b.w[i]), jnp.isnan(grads_i.w))
        )
        assert bool(jnp.allclose(loss_b[i], loss_i, rtol=1e-5, equal_nan=True))
        assert bool(
            jnp.allclose(grads_b.w[i], grads_i.w, rtol=1e-5, equal_nan=True)
        )
