"""The "Available optimizers" table in the README and docs lists exactly
the built-in registry, with a check or a cross for each kind of problem
an entry's constraint-handling level covers.

The table is hand-written, so nothing else stops it drifting when an
optimizer is added to or removed from the registry, or when an entry's
level changes (ADR 0008). The check runs in a
fresh interpreter: the registry is a module-level dict that importing a
meta-optimizer package (or a test calling ``register_optimizer``) mutates,
and only a clean import shows the built-ins alone.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(r"^\| `([^`]+)` \| (.*) \|$", re.MULTILINE)
# Among the cells after the name: Unconstrained, Constrained, Box
# constraints.
CONSTRAINTS_COLUMNS = slice(2, 5)
#: The marks each level puts in those three columns.
MARKS = {
    "none": ("✓", "✗", "✗"),
    "box": ("✓", "✗", "✓"),
    "general": ("✓", "✓", "✓"),
}


def _built_in_levels() -> dict[str, str]:
    code = (
        "import json; from l2co_optimizers._src.mapping import "
        "constraint_handling, optimizers; "
        "print(json.dumps({n: constraint_handling(n) for n in optimizers}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _table_rows(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """``(name, marks)`` for every row of the optimizer tables."""
    text = path.read_text()
    section = text.split("## Available optimizers", 1)[1]
    section = section.split("\n## ", 1)[0]
    return [
        (name, tuple(rest.split(" | ")[CONSTRAINTS_COLUMNS]))
        for name, rest in ROW.findall(section)
    ]


@pytest.mark.parametrize("doc", ["README.md", "docs/index.md"])
def test_optimizer_table_matches_registry(doc: str) -> None:
    from l2co_optimizers._src.core.utils import normalize_key

    rows = _table_rows(ROOT / doc)
    names = [name for name, _ in rows]
    assert len(names) == len(set(names)), f"duplicate rows in {doc}"
    assert {normalize_key(n) for n in names} == set(_built_in_levels())


@pytest.mark.parametrize("doc", ["README.md", "docs/index.md"])
def test_optimizer_table_constraints_match_registry(doc: str) -> None:
    from l2co_optimizers._src.core.utils import normalize_key

    expected = {
        name: MARKS[level] for name, level in _built_in_levels().items()
    }
    table = {
        normalize_key(name): marks for name, marks in _table_rows(ROOT / doc)
    }
    assert table == expected
