"""Unit tests for ``optimizer_parts``.

``optimizer_parts`` resolves an optimizer name against five registries
that are collectively *narrower* than the optimizer registry:
``normalized_optax_normal``, ``normalized_optax_from_state``,
``l2co_native_optax``, ``normalized_evosax`` and ``l2co_native_evosax``.
Nothing forces the two sides to agree, so an optimizer can be perfectly
runnable through ``optimizer_mapping`` and still be impossible to unpack
-- which is exactly what happened when ``"lbfgs"`` moved out of
``opt_names_from_state`` into its own module and left
``normalized_optax_from_state`` empty. The coverage test below is the
guard against that class of drift.

The parts are the second construction path for an optimizer (next to
the ``UpdateClass`` factory), so the per-optimizer properties they carry
-- popsize, family, ``own_ask``, the feval bound -- are checked against
the registry path here.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

# Local
import l2co_optimizers
from l2co_optimizers import (
    FAMILY_DISTRIBUTION,
    FAMILY_GRADIENT,
    FAMILY_POPULATION,
    GradientParts,
    OptimizationStep,
    PopulationParts,
    optimizer_parts,
)
from l2co_optimizers._src.core.state_transfer import transfer_spec_for
from l2co_optimizers._src.evosax_implementations import normalized_evosax
from l2co_optimizers._src.lbfgs import stock_lbfgs
from l2co_optimizers._src.mapping import optimizers as OPTIMIZER_REGISTRY
from l2co_optimizers._src.optax_implementations import (
    normalized_optax_from_state,
    normalized_optax_normal,
)
from l2co_optimizers._src.optimistix_implementations import (
    OPTIMISTIX_OPTIMIZERS,
)
from l2co_optimizers._src.optimizer_parts import (
    l2co_native_evosax,
    l2co_native_optax,
)
from l2co_optimizers._src.scipy_implementations import SCIPY_OPTIMIZERS

from .toy_problems import sphere_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# Meta-optimizers select *among* bare optimizers, so they cannot be
# unpacked into parts. ``"l2co"`` self-registers on ``import l2co``;
# ``"rl2co"`` does the same when rl2co is imported, so exclude both even
# though they are absent from this suite's registry.
META_OPTIMIZERS = frozenset({"l2co", "rl2co"})

# Random search is a structural baseline, not a menu entry: its
# ``RandomSearchUpdateClass`` bypasses the step loop entirely. It used to
# pass this check only because evosax's ``RandomSearch`` sat under the
# same name in ``normalized_evosax`` -- and unpacking it raised (no
# ``sampling_fn``).
#
# The optimistix minimisers evaluate the objective themselves, which is
# neither an optax transform nor an ask/tell population, so they are
# plain-run entries only (ADR 0001). The scipy minimisers run their whole
# budget inside one host callback, so they have no step to unpack at all
# (ADR 0002).
NON_PARTS_OPTIMIZERS = (
    META_OPTIMIZERS
    | {"randomsearch"}
    | OPTIMISTIX_OPTIMIZERS
    | SCIPY_OPTIMIZERS
)


def _parts_registries() -> set[str]:
    """Union of every registry ``optimizer_parts`` dispatches on."""
    return (
        set(normalized_optax_normal)
        | set(normalized_optax_from_state)
        | set(l2co_native_optax)
        | set(normalized_evosax)
        | set(l2co_native_evosax)
    )


UNPACKABLE = sorted(
    (set(l2co_optimizers.optimizers) & _parts_registries())
    - NON_PARTS_OPTIMIZERS
)


_OPTAX_NAMED = set(normalized_optax_normal) | set(normalized_optax_from_state)


@pytest.fixture(scope="module")
def problem():
    return sphere_problem(4)


class TestRegistryCoverage:
    """Every runnable optimizer must unpack into parts."""

    def test_registered_optimizers_are_unpackable(self):
        missing = sorted(
            set(l2co_optimizers.optimizers)
            - _parts_registries()
            - NON_PARTS_OPTIMIZERS
        )
        assert not missing, (
            f"registered but not unpackable into parts: {missing}. "
            f"Add each to the matching registry in "
            f"l2co_optimizers._src.optimizer_parts."
        )

    def test_lbfgs_is_registered_as_native_optax(self):
        # The specific regression: lbfgs is assembled here rather than
        # named on the optax module, so it lives in l2co_native_optax
        # and not normalized_optax_*.
        assert l2co_native_optax["lbfgs"] is stock_lbfgs
        assert "lbfgs" not in normalized_optax_normal
        assert "lbfgs" not in normalized_optax_from_state


class TestAgreesWithRegistryPath:
    """Parts carry the same per-optimizer properties as the factory."""

    @pytest.mark.parametrize("name", UNPACKABLE)
    def test_properties_match_update_class(self, name, problem):
        # Most optax transforms have no default learning rate.
        hyperparameters = (
            {"learning_rate": 0.01} if name in _OPTAX_NAMED else {}
        )
        opt_step = OptimizationStep(
            optimizer=name, hyperparameters=hyperparameters
        )
        parts = optimizer_parts(opt_step, **problem)
        update_class = l2co_optimizers.optimizer_mapping(name)(
            opt_hash=opt_step.hash,
            bounded=(None, None),
            stop_fn=None,
            **problem,
            **hyperparameters,
        )
        assert parts.popsize == int(update_class.popsize)
        assert parts.family == update_class.family
        assert parts.name_hash == int(opt_step.hash)
        if parts.family == FAMILY_GRADIENT:
            assert isinstance(parts, GradientParts)
        else:
            assert isinstance(parts, PopulationParts)
            assert (
                parts.own_ask == transfer_spec_for(name, parts.family).own_ask
            )

    def test_families_cover_both_kinds(self, problem):
        families = {
            optimizer_parts(
                OptimizationStep(optimizer=n, hyperparameters=hp), **problem
            ).family
            for n, hp in (
                ("adam", {"learning_rate": 0.01}),
                ("cmaes", {}),
                ("differentialevolution", {}),
            )
        }
        assert families == {
            FAMILY_GRADIENT,
            FAMILY_DISTRIBUTION,
            FAMILY_POPULATION,
        }

    def test_lbfgs_carries_its_linesearch_bound(self, problem):
        parts = optimizer_parts(
            OptimizationStep(
                optimizer="lbfgs",
                hyperparameters={"max_linesearch_steps": 7},
            ),
            **problem,
        )
        assert parts.max_fevals_per_step == 8

    def test_plain_transform_has_no_bound(self, problem):
        parts = optimizer_parts(
            OptimizationStep(
                optimizer="adam", hyperparameters={"learning_rate": 0.01}
            ),
            **problem,
        )
        assert parts.max_fevals_per_step is None

    def test_explicit_popsize_wins(self, problem):
        parts = optimizer_parts(
            OptimizationStep(
                optimizer="cmaes", hyperparameters={"popsize": 12}
            ),
            **problem,
        )
        assert parts.popsize == 12


class TestLBFGSDispatch:
    """``"lbfgs"`` unpacks into working gradient parts."""

    def test_resolves_to_gradient_parts(self, problem):
        parts = optimizer_parts(OptimizationStep(optimizer="lbfgs"), **problem)
        assert isinstance(parts, GradientParts)
        assert parts.popsize == 1

    def test_hyperparameters_reach_the_transform(self, problem):
        # memory_size is a stock_lbfgs kwarg; a bogus one must surface
        # as a TypeError rather than being silently dropped.
        parts = optimizer_parts(
            OptimizationStep(
                optimizer="lbfgs", hyperparameters={"memory_size": 3}
            ),
            **problem,
        )
        assert isinstance(parts, GradientParts)
        with pytest.raises(TypeError):
            optimizer_parts(
                OptimizationStep(
                    optimizer="lbfgs",
                    hyperparameters={"not_an_lbfgs_kwarg": 1},
                ),
                **problem,
            )

    def test_transform_takes_a_line_searched_step(self, problem):
        # The line search needs the value/grad/value_fn extra args. A
        # successful step on a quadratic must decrease the loss.
        parts = optimizer_parts(OptimizationStep(optimizer="lbfgs"), **problem)

        def value_fn(p):
            return jnp.sum(p**2)

        params = jnp.full((4,), 2.0)
        state = parts.transform.init(params)
        grads = jax.grad(value_fn)(params)
        update, _ = parts.transform.update(
            grads,
            state,
            params,
            value=value_fn(params),
            grad=grads,
            value_fn=value_fn,
        )
        new_params = params + update
        assert float(value_fn(new_params)) < float(value_fn(params))


class TestPopulationParts:
    """The ``(init, ask, tell)`` triple round-trips one generation."""

    @pytest.mark.parametrize(
        "name", ["cmaes", "differentialevolution", "shade"]
    )
    def test_ask_tell(self, name, problem):
        parts = optimizer_parts(OptimizationStep(optimizer=name), **problem)
        params = problem["model"]
        state = parts.init_fn(params, jr.key(0))
        population, state = parts.ask_fn(params, state, jr.key(1))
        assert population.shape == (parts.popsize, *params.shape)
        fitness = jax.vmap(problem["loss_fn"])(population)
        state = parts.tell_fn(state, population, fitness, jr.key(2))
        sigma, _ = parts.transfer_read_fn(state)
        assert jnp.isfinite(sigma)


class TestUnresolvableName:
    """The error names the registries, not just ``optax``/``evosax``."""

    def test_raises_with_actionable_message(self, problem):
        # Reproduce the drift the coverage test guards: a name present
        # in the optimizer registry but in none of the parts
        # registries. An *unregistered* name cannot reach this error --
        # resolve_popsize's optimizer_mapping lookup rejects it first.
        # No underscores: optimizer_mapping normalises the lookup key,
        # so the registered name must survive normalize_key unchanged.
        name = "dummyregisterednotparts"
        l2co_optimizers.register_optimizer(
            name, l2co_optimizers.optimizer_mapping("adam")
        )
        try:
            with pytest.raises(ValueError, match="l2co_native_optax"):
                optimizer_parts(
                    OptimizationStep(
                        optimizer=name,
                        hyperparameters={"learning_rate": 0.01},
                    ),
                    **problem,
                )
        finally:
            OPTIMIZER_REGISTRY.pop(name, None)

    def test_unregistered_name_raises_earlier(self, problem):
        with pytest.raises(ValueError, match="not recognized"):
            optimizer_parts(
                OptimizationStep(optimizer="no_such_optimizer"), **problem
            )
