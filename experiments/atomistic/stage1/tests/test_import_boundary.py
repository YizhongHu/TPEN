"""Enforce the experiments-to-core import boundary for this study slice."""

from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).parents[1]


def test_only_train_and_eval_use_the_public_tpen_gateway() -> None:
    violations: list[str] = []
    for path in sorted(HERE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("tpen") and alias.name != "tpen.run":
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("tpen") and module != "tpen.run":
                    violations.append(f"{path}: from {module}")
                if module == "tpen.run" and any(alias.name not in {"run_from_config", "load_config"}
                                                for alias in node.names):
                    violations.append(f"{path}: non-gateway import")
    assert not violations, "import boundary violated:\n" + "\n".join(violations)
