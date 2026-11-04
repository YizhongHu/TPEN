"""Keep the scientific core independent from the executor project."""

import ast
from pathlib import Path


CORE_ROOT = Path(__file__).parents[2] / "tpen"


def _dynamic_import_root(node: ast.Call) -> str | None:
    """Return the root of a literal dynamic import, when one is auditable."""
    callee = node.func
    is_importlib_call = (
        isinstance(callee, ast.Attribute)
        and callee.attr == "import_module"
        and isinstance(callee.value, ast.Name)
        and callee.value.id == "importlib"
    )
    is_bare_call = isinstance(callee, ast.Name) and callee.id in {
        "import_module",
        "__import__",
    }
    if not (is_importlib_call or is_bare_call):
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        return None
    return node.args[0].value.split(".", maxsplit=1)[0]


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


def test_core_does_not_dynamically_import_executor() -> None:
    """Literal dynamic imports must not cross into the executor project.

    Subprocess/CLI calls and non-literal module names are intentionally outside
    this small AST guard; only string literals are checked.
    """
    violations: list[str] = []
    for path in sorted(CORE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dynamic_import_root(node) == "tpen_executor":
                violations.append(f"{path}:{node.lineno}")
    assert not violations, "core dynamically imports executor: " + ", ".join(violations)
