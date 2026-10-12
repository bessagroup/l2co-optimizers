"""Every registry entry declares how far it can enforce constraints (ADR
0008).

The built-ins' levels themselves are pinned by the "Constraints" column
of the README's optimizer table (``test_readme_optimizer_table.py``).
These tests cover the lookup and what ``register_optimizer`` records.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import pytest

# Local
import l2co_optimizers
from l2co_optimizers import (
    CONSTRAINT_HANDLING_LEVELS,
    CONSTRAINTS_BOX,
    CONSTRAINTS_GENERAL,
    CONSTRAINTS_NONE,
    constraint_handling,
    optimizers,
    register_optimizer,
)
from l2co_optimizers._src import mapping

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# No underscores: the key must survive normalize_key unchanged.
NAME = "dummyconstrainthandling"


@pytest.fixture
def unregister():
    """Remove ``NAME`` from the registry after the test."""
    yield
    optimizers.pop(NAME, None)
    mapping._constraint_handling.pop(NAME, None)


def test_levels_are_ordered_weakest_first():
    assert CONSTRAINT_HANDLING_LEVELS == (
        CONSTRAINTS_NONE,
        CONSTRAINTS_BOX,
        CONSTRAINTS_GENERAL,
    )


def test_every_built_in_has_a_level():
    for name in optimizers:
        assert constraint_handling(name) in CONSTRAINT_HANDLING_LEVELS


def test_lookup_normalizes_the_name():
    assert constraint_handling("rbf_trust_region") == CONSTRAINTS_BOX
    assert constraint_handling("RBFTrustRegion") == CONSTRAINTS_BOX


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="not recognized"):
        constraint_handling("notanoptimizer")


def test_registering_without_a_level_is_none(unregister):
    register_optimizer(NAME, l2co_optimizers.optimizer_mapping("adam"))
    assert constraint_handling(NAME) == CONSTRAINTS_NONE


def test_registering_records_the_declared_level(unregister):
    register_optimizer(
        NAME,
        l2co_optimizers.optimizer_mapping("shade"),
        constraint_handling=CONSTRAINTS_BOX,
    )
    assert constraint_handling(NAME) == CONSTRAINTS_BOX


def test_re_registering_replaces_the_level(unregister):
    factory = l2co_optimizers.optimizer_mapping("slsqp")
    register_optimizer(NAME, factory, constraint_handling=CONSTRAINTS_GENERAL)
    register_optimizer(NAME, factory)
    assert constraint_handling(NAME) == CONSTRAINTS_NONE


def test_registering_an_unknown_level_raises_and_registers_nothing(
    unregister,
):
    with pytest.raises(ValueError, match="constraint_handling"):
        register_optimizer(
            NAME,
            l2co_optimizers.optimizer_mapping("adam"),
            constraint_handling="bounds",
        )
    assert NAME not in optimizers


def test_a_name_added_without_register_optimizer_is_none(unregister):
    optimizers[NAME] = l2co_optimizers.optimizer_mapping("adam")
    assert constraint_handling(NAME) == CONSTRAINTS_NONE
