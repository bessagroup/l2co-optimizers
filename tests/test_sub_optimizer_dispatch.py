"""Unit tests for ``optstep_to_subopt``'s name dispatch.

``optstep_to_subopt`` resolves an optimizer name against five
registries that are collectively *narrower* than l2co's optimizer
registry: ``normalized_optax_normal``,
``normalized_optax_from_state``, ``l2co_native_optax``,
``normalized_evosax`` and ``l2co_native_evosax``. Nothing forces the
two sides to agree, so an optimizer can be perfectly runnable through
``optimizer_mapping`` and still be unresolvable to a ``SubOpt`` --
which is exactly what happened when ``"lbfgs"`` moved out of
``opt_names_from_state`` into its own module and left
``normalized_optax_from_state`` empty. The coverage test below is the
guard against that class of drift.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import l2co_optimizers
from l2co_optimizers import OptimizationStep
from l2co_optimizers._src.evosax_implementations import normalized_evosax
from l2co_optimizers._src.lbfgs import stock_lbfgs
from l2co_optimizers._src.mapping import optimizers as OPTIMIZER_REGISTRY
from l2co_optimizers._src.optax_implementations import (
    normalized_optax_from_state,
    normalized_optax_normal,
)
from l2co_optimizers._src.sub_optimizer import (
    l2co_native_evosax,
    l2co_native_optax,
    optstep_to_subopt,
)

from .toy_tasks import sphere_task

# =============================================================================

# Meta-optimizers select *among* sub-optimizers, so they are not
# themselves resolvable to a ``SubOpt``. ``"l2co"`` self-registers on
# ``import l2co``; ``"rl2co"`` does the same when rl2co is imported, so
# exclude it too even though it is absent from this suite's registry.
META_OPTIMIZERS = frozenset({"l2co", "rl2co"})

# Random search is a structural baseline, not a menu entry: its
# ``RandomSearchUpdateClass`` bypasses the step loop entirely. It used to
# pass this check only because evosax's ``RandomSearch`` sat under the
# same name in ``normalized_evosax`` -- and building that SubOpt raised
# (no ``sampling_fn``).
NON_SUBOPT_OPTIMIZERS = META_OPTIMIZERS | {"randomsearch"}


def _subopt_registries() -> set[str]:
    """Union of every registry ``optstep_to_subopt`` dispatches on."""
    return (
        set(normalized_optax_normal)
        | set(normalized_optax_from_state)
        | set(l2co_native_optax)
        | set(normalized_evosax)
        | set(l2co_native_evosax)
    )


@pytest.fixture(scope="module")
def task():
    return sphere_task(4)


class TestRegistryCoverage:
    """Every runnable optimizer must resolve to a ``SubOpt``."""

    def test_registered_optimizers_are_subopt_resolvable(self):
        missing = sorted(
            set(l2co_optimizers.optimizers)
            - _subopt_registries()
            - NON_SUBOPT_OPTIMIZERS
        )
        assert not missing, (
            f"registered but not resolvable to a SubOpt: {missing}. "
            f"Add each to the matching registry in "
            f"l2co_optimizers._src.sub_optimizer."
        )

    def test_lbfgs_is_registered_as_native_optax(self):
        # The specific regression: lbfgs is assembled by l2co rather
        # than named on the optax module, so it lives in
        # l2co_native_optax and not normalized_optax_*.
        assert l2co_native_optax["lbfgs"] is stock_lbfgs
        assert "lbfgs" not in normalized_optax_normal
        assert "lbfgs" not in normalized_optax_from_state


class TestLBFGSDispatch:
    """``"lbfgs"`` resolves to a working gradient-based ``SubOpt``."""

    def test_resolves_to_grad_subopt(self, task):
        sub = optstep_to_subopt(OptimizationStep(optimizer="lbfgs"), task)
        assert sub.is_pop is False
        assert sub.popsize == 1
        assert sub.tell_fn is None

    def test_hyperparameters_reach_the_transform(self, task):
        # memory_size is a stock_lbfgs kwarg; a bogus one must surface
        # as a TypeError rather than being silently dropped.
        sub = optstep_to_subopt(
            OptimizationStep(
                optimizer="lbfgs", hyperparameters={"memory_size": 3}
            ),
            task,
        )
        assert sub.is_pop is False
        with pytest.raises(TypeError):
            optstep_to_subopt(
                OptimizationStep(
                    optimizer="lbfgs",
                    hyperparameters={"not_an_lbfgs_kwarg": 1},
                ),
                task,
            )

    def test_step_takes_a_line_searched_step(self, task):
        # The line search needs the value/grad/value_fn extra args; a
        # plain step_fn call without them would TypeError. A successful
        # step on a quadratic must decrease the loss.
        sub = optstep_to_subopt(OptimizationStep(optimizer="lbfgs"), task)

        def value_fn(p):
            return jnp.sum(p**2)

        params = jnp.full((4,), 2.0)
        sub_state = sub.init_fn(params, jr.key(0))
        grads = jax.grad(value_fn)(params)
        update, _ = sub.step_fn(
            grads,
            params,
            sub_state,
            jr.key(1),
            value=value_fn(params),
            grad=grads,
            value_fn=value_fn,
        )
        new_params = params + update
        assert float(value_fn(new_params)) < float(value_fn(params))


class TestUnresolvableName:
    """The error names the registries, not just ``optax``/``evosax``."""

    def test_raises_with_actionable_message(self, task):
        # Reproduce the drift the coverage test guards: a name present
        # in the optimizer registry but in none of the SubOpt
        # registries. An *unregistered* name cannot reach this error --
        # resolve_popsize's optimizer_mapping lookup rejects it first.
        # No underscores: optimizer_mapping normalises the lookup key,
        # so the registered name must survive normalize_key unchanged.
        name = "dummyregisterednotsubopt"
        l2co_optimizers.register_optimizer(
            name, l2co_optimizers.optimizer_mapping("adam")
        )
        try:
            with pytest.raises(ValueError, match="l2co_native_optax"):
                optstep_to_subopt(
                    OptimizationStep(
                        optimizer=name,
                        hyperparameters={"learning_rate": 0.01},
                    ),
                    task,
                )
        finally:
            OPTIMIZER_REGISTRY.pop(name, None)

    def test_unregistered_name_raises_earlier(self, task):
        with pytest.raises(ValueError, match="not recognized"):
            optstep_to_subopt(
                OptimizationStep(optimizer="no_such_optimizer"), task
            )
