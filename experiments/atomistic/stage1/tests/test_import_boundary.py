"""Enforce the experiments-to-core import boundary for this study slice."""

from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).parents[1]


_PUBLIC_IMPORTS = {
    "tpen.run": {"run_from_config", "load_config"},
}
_RESTART_IMPORTS = {
    "tpen.execution": {
        "create_json_receipt", "discover_published_checkpoint", "ReceiptExistsError",
        "OperationalRestoreRequest", "ScienceIntent", "CheckpointEvidence",
        "select_recovery", "validate_native_recovery_profile",
    },
    "tpen.checkpoint": {
        "list_complete_checkpoints", "CheckpointRef", "configured_load_mode",
    },
    "tpen.run": {"run_from_config", "load_config"},
}


def _violations(path: Path, source: str) -> list[str]:
    """Return public-boundary violations for one study source file."""

    tree = ast.parse(source, filename=str(path))
    allowed = _RESTART_IMPORTS if path.name == "restart.py" else _PUBLIC_IMPORTS
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
    violations: list[str] = []
    for path in sorted(HERE.rglob("*.py")):
        # Test probes may exercise core internals; the boundary applies to
        # Stage-1 production modules, not the tests that inspect them.
        if path.name.startswith("test_"):
            continue
        violations.extend(_violations(path, path.read_text(encoding="utf-8")))
    assert not violations, "import boundary violated:\n" + "\n".join(violations)


def test_restart_deep_submodule_import_is_rejected() -> None:
    restart = HERE / "restart.py"
    source = "from tpen.execution.invocation import create_json_receipt\n"
    violations = _violations(restart, source)
    assert violations == [f"{restart}: from tpen.execution.invocation"]
