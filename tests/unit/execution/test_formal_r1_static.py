"""Static review guards for local-provenance documentation.

These are AST-only source checks, not backend runtime qualification. They
must remain independent of optional Balsam imports and facility access.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[3]
DOC_FILES = (
    Path("tpen/execution/balsam.py"),
    Path("tools/e5b_polaris_app.py"),
    Path("tools/e5b_polaris_pilot.py"),
)
FORBIDDEN_LOCAL_PROVENANCE = ("/Users/yizhonghu", "/home/rhu")


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


@pytest.mark.parametrize("relative_path", DOC_FILES)
def test_review_docstrings_contain_no_local_provenance(relative_path: Path) -> None:
    constants = _string_constants(REPO_ROOT / relative_path)
    for value in constants:
        assert not any(marker in value for marker in FORBIDDEN_LOCAL_PROVENANCE), (
            f"{relative_path} contains local provenance in an AST string literal"
        )
