"""Keep the scientific core independent from the executor project."""

import ast
from pathlib import Path


CORE_ROOT = Path(__file__).parents[2] / "tpen"


def test_core_does_not_import_executor() -> None:
    """No module under ``tpen`` may import ``tpen_executor``."""
    violations: list[str] = []
    for path in sorted(CORE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "tpen_executor" or alias.name.startswith("tpen_executor.") for alias in node.names):
                    violations.append(f"{path}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if node.module == "tpen_executor" or (node.module or "").startswith("tpen_executor."):
                    violations.append(f"{path}:{node.lineno}")
    assert not violations, "core imports executor: " + ", ".join(violations)
