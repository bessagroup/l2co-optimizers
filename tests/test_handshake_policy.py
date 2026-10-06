"""The shared handshake policy table (``docs/adr/0009``).

The table says which handshake variants an optimizer needs when control
passes to it. It lives in l2co so the rl2co environment and both
deployment wrappers resolve from one source instead of three; these
tests pin the table's contents, the fallback, and the propagation into
``SubOpt.opt_state_handshake`` that lets the wrappers act on it.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import pytest

# Local
from l2co_optimizers import (
    DEFAULT_HANDSHAKE_POLICY,
    HANDSHAKE_POLICY,
    OPT_STATE_VARIANTS,
    POPULATION_VARIANTS,
    HandshakePolicy,
    OptimizationStep,
    handshake_policy_for,
    optstep_to_subopt,
)

from .toy_problems import rastrigin_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


@pytest.fixture(scope="module")
def problem():
    return rastrigin_problem(6)


class TestTable:
    def test_every_entry_names_a_known_variant(self):
        for name, policy in HANDSHAKE_POLICY.items():
            assert policy.population in POPULATION_VARIANTS, name
            assert policy.opt_state in OPT_STATE_VARIANTS, name

    def test_lbfgs_is_the_one_reset_entry(self):
        # The whole reason the table exists: L-BFGS curvature and
        # line-search state describe the neighbourhood of an iterate the
        # handshake jumps away from.
        resets = [
            name
            for name, policy in HANDSHAKE_POLICY.items()
            if policy.opt_state == "reset"
        ]
        assert resets == ["lbfgs"]
        assert HANDSHAKE_POLICY["lbfgs"].population == "best"

    def test_gradient_optimizers_take_best_and_continue(self):
        for name in ("adam", "sgd"):
            assert HANDSHAKE_POLICY[name] == HandshakePolicy(
                "best", "continue"
            )

    def test_population_optimizers_take_best_one_and_continue(self):
        for name in ("sepcmaes", "cmaes", "pso", "shade"):
            assert HANDSHAKE_POLICY[name] == HandshakePolicy(
                "best_one", "continue"
            )


class TestResolution:
    def test_resolves_by_name_or_optimization_step(self):
        assert handshake_policy_for("lbfgs") is HANDSHAKE_POLICY["lbfgs"]
        assert (
            handshake_policy_for(OptimizationStep(optimizer="lbfgs"))
            is HANDSHAKE_POLICY["lbfgs"]
        )

    def test_hyperparameters_do_not_affect_resolution(self):
        # The policy is a property of the algorithm, not its tuning.
        assert (
            handshake_policy_for(
                OptimizationStep(
                    optimizer="adam", hyperparameters={"learning_rate": 10.0}
                )
            )
            is HANDSHAKE_POLICY["adam"]
        )

    @pytest.mark.parametrize("name", ["adamw", "snes"])
    def test_untabulated_optimizer_falls_back(self, name):
        # Both are in the registry but not the table; the registry is
        # extensible, so absence must not raise.
        assert name not in HANDSHAKE_POLICY
        assert handshake_policy_for(name) is DEFAULT_HANDSHAKE_POLICY

    def test_default_is_the_conservative_pair(self):
        assert DEFAULT_HANDSHAKE_POLICY == HandshakePolicy(
            "best_one", "continue"
        )


class TestSubOptPropagation:
    """The wrappers read the policy off ``SubOpt``, not off the name."""

    def test_optstep_to_subopt_carries_the_policy(self, problem):
        lbfgs = optstep_to_subopt(
            OptimizationStep(optimizer="lbfgs"), **problem
        )
        assert lbfgs.opt_state_handshake == "reset"

        cmaes = optstep_to_subopt(
            OptimizationStep(optimizer="sepcmaes"), **problem
        )
        assert cmaes.opt_state_handshake == "continue"

    def test_untabulated_subopt_defaults_to_continue(self, problem):
        snes = optstep_to_subopt(OptimizationStep(optimizer="snes"), **problem)
        assert snes.opt_state_handshake == "continue"

    def test_field_is_static(self, problem):
        # Must contribute no array leaves: the wrappers branch on it at
        # Python trace time.
        import jax

        sub = optstep_to_subopt(OptimizationStep(optimizer="lbfgs"), **problem)
        assert jax.tree.leaves(sub) == []
