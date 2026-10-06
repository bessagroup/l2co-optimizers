"""Regression tests for the even-population-size default.

Several EvoSax distribution-based algorithms require an even population
size: the antithetic / mirrored-sampling strategies assert
``population_size % 2 == 0`` at construction, and ESMC silently emits
``popsize - 1`` candidates for an odd population size. These default to
:func:`variable_popsize_even` instead of :func:`variable_popsize` so the
default population size is never odd (``variable_popsize`` is odd whenever
``int(4 + 3 * log(d))`` is odd, e.g. ``d == 3`` gives ``7``).
"""

#                                                                       Modules
# =============================================================================

# Third-party
import pytest

# Local
from l2co_optimizers._src.evosax_implementations import (
    _EVEN_POPSIZE_REQUIRED,
    evosax_mapping,
)
from l2co_optimizers._src.popsize import (
    variable_popsize,
    variable_popsize_even,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# d == 3 is the smallest dimensionality whose ``variable_popsize`` is odd.
_ODD_DIMS = [3, 7, 11]
_EVEN_DIMS = [2, 4, 5, 10, 20]


@pytest.mark.parametrize("dim", _ODD_DIMS + _EVEN_DIMS)
def test_variable_popsize_even_is_even(dim: int):
    assert variable_popsize_even(dim) % 2 == 0


@pytest.mark.parametrize("dim", _ODD_DIMS + _EVEN_DIMS)
def test_variable_popsize_even_rounds_up_by_at_most_one(dim: int):
    base = variable_popsize(dim)
    even = variable_popsize_even(dim)
    assert even >= base
    assert even - base == (base % 2)  # +1 iff base is odd, else unchanged


def test_odd_dim_makes_plain_default_odd():
    # Guards the premise: without rounding, d == 3 yields an odd popsize.
    assert variable_popsize(3) == 7
    assert variable_popsize_even(3) == 8


@pytest.mark.parametrize("name", sorted(_EVEN_POPSIZE_REQUIRED))
def test_even_required_optimizers_default_to_even_popsize(name: str):
    bound = evosax_mapping[name].keywords.get("popsize")
    assert bound is variable_popsize_even
    assert bound(3) % 2 == 0


@pytest.mark.parametrize("name", ["cmaes", "snes", "xnes", "pso", "simplega"])
def test_other_optimizers_keep_classmethod_default(name: str):
    # Unconstrained algorithms do not bind ``popsize`` in the partial; they
    # fall back to the classmethod's ``variable_popsize`` default.
    assert "popsize" not in evosax_mapping[name].keywords
