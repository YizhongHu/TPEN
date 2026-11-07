"""Enforce the H-v2 experiment boundary: config/``run_from_config``/``tpen`` only.

Operator ruling (root note ``operator-decisions-20261106-imports-mirflow-readonly``,
item 1): experiment code and tests reach core only through configuration,
``tpen.run.run_from_config``, or the installed ``tpen`` command. An
integration test that needs core internals lives in the core ``tests/`` tree
instead (item 2). This test parses every ``.py`` file in this directory with
``ast`` and fails on any ``tpen`` import other than
``tpen.run.run_from_config``, and on any import of ``hydra.utils.instantiate``.
"""

from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).parent

#: The only tpen import allowed under this directory.
ALLOWED_TPEN_IMPORT = ("tpen.run", "run_from_config")


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tpen" or alias.name.startswith("tpen."):
                    found.append(f"{path.name}: import {alias.name}")
                if alias.name == "hydra.utils":
                    found.append(f"{path.name}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "hydra.utils" and any(alias.name == "instantiate" for alias in node.names):
                found.append(f"{path.name}: from {module} import instantiate")
            if module == "tpen" or module.startswith("tpen."):
                if module == ALLOWED_TPEN_IMPORT[0] and all(
                    alias.name == ALLOWED_TPEN_IMPORT[1] for alias in node.names
                ):
                    continue
                names = ", ".join(alias.name for alias in node.names)
                found.append(f"{path.name}: from {module} import {names}")
    return found


def test_experiment_files_only_reach_core_through_run_from_config() -> None:
    violations: list[str] = []
    for path in sorted(HERE.glob("*.py")):
        violations.extend(_violations(path))
    assert not violations, "experiment boundary violated:\n" + "\n".join(violations)
