"""The "Available optimizers" table in the README and docs lists exactly
the built-in registry.

The table is hand-written, so nothing else stops it drifting when an
optimizer is added to or removed from the registry. The check runs in a
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
ROW = re.compile(r"^\| `([^`]+)` \|", re.MULTILINE)


def _built_in_names() -> set[str]:
    code = (
        "import json; from l2co_optimizers._src.mapping import optimizers; "
        "print(json.dumps(sorted(optimizers)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(json.loads(out.stdout))


def _table_names(path: Path) -> list[str]:
    text = path.read_text()
    section = text.split("## Available optimizers", 1)[1]
    section = section.split("\n## ", 1)[0]
    return ROW.findall(section)


@pytest.mark.parametrize("doc", ["README.md", "docs/index.md"])
def test_optimizer_table_matches_registry(doc: str) -> None:
    from l2co_optimizers._src.core.utils import normalize_key

    names = _table_names(ROOT / doc)
    assert len(names) == len(set(names)), f"duplicate rows in {doc}"
    assert {normalize_key(n) for n in names} == _built_in_names()
