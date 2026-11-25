"""Enforce the experiments-to-core import boundary for this study slice."""

from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).parents[1]


_PUBLIC_IMPORTS = {
    "tpen.inspection": {"RunEvidence", "inspect_run"},
    "tpen.run": {"run_from_config", "load_config"},
    "tpen.callback": {"PlannedYield"},
}


def _violations(path: Path, source: str) -> list[str]:
    """Return public-boundary violations for one study source file."""

    tree = ast.parse(source, filename=str(path))
    allowed = _PUBLIC_IMPORTS
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tpen") and alias.name not in allowed:
                    violations.append(f"{path}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not module.startswith("tpen"):
                continue
            names = allowed.get(module)
            if names is None:
                violations.append(f"{path}: from {module}")
                continue
            for alias in node.names:
                if alias.name not in names:
                    violations.append(f"{path}: from {module} import {alias.name}")
    return violations


def test_only_train_and_eval_use_the_public_tpen_gateway() -> None:
    """Check production modules; tests deliberately probe core internals."""

    violations: list[str] = []
    tests_root = HERE / "tests"
    for path in sorted(HERE.rglob("*.py")):
        # Stage-1 test probes may exercise core internals; production modules
        # under the study root remain subject to this boundary.
        if tests_root in path.parents:
            continue
        violations.extend(_violations(path, path.read_text(encoding="utf-8")))
    assert not violations, "import boundary violated:\n" + "\n".join(violations)


def test_restart_deep_submodule_import_is_rejected() -> None:
    restart = HERE / "restart.py"
    source = "from tpen.inspection._private import inspect_run\n"
    violations = _violations(restart, source)
    assert violations == [f"{restart}: from tpen.inspection._private"]
