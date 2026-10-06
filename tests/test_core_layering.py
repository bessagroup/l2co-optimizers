"""``_src/core`` is optimizer-agnostic: it never imports from outside itself.

The implementations (optax, evosax, L-BFGS, SHADE, TuRBO, RBF trust
region, random search) and the modules that must see all of them
(``mapping``, ``optimizer_parts``) depend on the core, never
the other way round. Checked statically, including ``TYPE_CHECKING`` and
function-local imports, so a cycle cannot hide behind a lazy import.
"""

#                                                                       Modules
# =============================================================================

# Standard
import ast
from pathlib import Path

# Third-party
import pytest

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

SRC = Path(__file__).parents[1] / "src" / "l2co_optimizers" / "_src"
CORE_MODULES = sorted((SRC / "core").glob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


@pytest.mark.parametrize("path", CORE_MODULES, ids=lambda p: p.stem)
def test_core_imports_only_core(path: Path) -> None:
    outside = sorted(
        module
        for module in _imported_modules(path)
        if module.startswith("l2co_optimizers")
        and not module.startswith("l2co_optimizers._src.core")
    )
    assert outside == [], f"core/{path.name} imports {outside}"


def test_core_is_not_empty() -> None:
    assert len(CORE_MODULES) > 10
