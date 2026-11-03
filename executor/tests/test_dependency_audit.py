"""Keep the executor package independent from TPEN and optional backends."""

import ast
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "tpen_executor"
FORBIDDEN_ROOTS = {"tpen", "experiments", "torch", "parsl", "balsam", "hydra", "omegaconf"}


def _import_root(node: ast.Import | ast.ImportFrom) -> str:
    """Return the top-level module named by an import node."""
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".", maxsplit=1)[0]
    return (node.module or "").split(".", maxsplit=1)[0]


def test_source_imports_only_stdlib_or_executor() -> None:
    """The initial project boundary must not depend on core or backends."""
    violations: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                root = _import_root(node)
                if root in FORBIDDEN_ROOTS:
                    violations.append(f"{path}: {root}")
                elif root != "tpen_executor" and root not in sys.stdlib_module_names:
                    violations.append(f"{path}: non-stdlib {root}")
    assert not violations, "forbidden executor imports: " + ", ".join(violations)
