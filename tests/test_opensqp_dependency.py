"""``"opensqp"`` without modOpt (ADR 0010).

modOpt is not a dependency of this package, and the ``modopt`` on PyPI
is an unrelated library. Building the entry without LSDOlab's modOpt
must fail at once, saying how to install it: inside a run, the driver
would take the error for a failed run and idle at the start.
"""

#                                                                       Modules
# =============================================================================

# Standard
import sys
import types

# Third-party
import jax.numpy as jnp
import pytest

# Local
from l2co_optimizers._src.mapping import optimizer_mapping
from l2co_optimizers._src.opensqp import _modopt

from .toy_problems import quadratic_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


@pytest.fixture
def fresh_import():
    """Forget the cached modOpt import before and after the test."""
    _modopt.cache_clear()
    yield
    _modopt.cache_clear()


def _build():
    return optimizer_mapping("opensqp")(
        **quadratic_problem(jnp.zeros(3), lambda x, **_: jnp.sum(x**2)),
        opt_hash=7,
        bounded=(None, None),
        stop_fn=None,
    )


def test_missing_modopt_fails_at_construction(monkeypatch, fresh_import):
    monkeypatch.setitem(sys.modules, "modopt", None)
    with pytest.raises(ImportError, match="uv sync --group opensqp"):
        _build()


def test_the_unrelated_pypi_modopt_fails_at_construction(
    monkeypatch, fresh_import
):
    """PyPI's ``modopt`` (CEA's image-processing ModOpt) has no OpenSQP."""
    monkeypatch.setitem(sys.modules, "modopt", types.ModuleType("modopt"))
    with pytest.raises(ImportError, match="unrelated library"):
        _build()


def test_a_missing_qp_solver_fails_at_construction(monkeypatch, fresh_import):
    pytest.importorskip("quadprog")
    monkeypatch.setitem(sys.modules, "quadprog", None)
    with pytest.raises(ImportError, match="import of quadprog halted"):
        _build()
