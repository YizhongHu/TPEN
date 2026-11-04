"""Keep the executor package independent from TPEN and optional backends."""

import ast
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "tpen_executor"
FORBIDDEN_ROOTS = {"tpen", "experiments", "torch", "parsl", "balsam"}
CONFIGURATION_ROOTS = {"hydra", "omegaconf"}


def _import_roots(node: ast.Import | ast.ImportFrom) -> list[str]:
    """Return every top-level module named by an import node."""
    if isinstance(node, ast.Import):
        return [alias.name.split(".", maxsplit=1)[0] for alias in node.names]
    return [(node.module or "").split(".", maxsplit=1)[0]]


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


def test_source_imports_only_stdlib_or_executor() -> None:
    """The initial project boundary must not depend on core or backends."""
    violations: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.level > 0:
                    continue
                for root in _import_roots(node):
                    if root in FORBIDDEN_ROOTS or (root in CONFIGURATION_ROOTS and path.name not in {"configuration.py", "programs.py"}):
                        violations.append(f"{path}: {root}")
                    elif root not in {"tpen_executor", *CONFIGURATION_ROOTS} and root not in sys.stdlib_module_names:
                        violations.append(f"{path}: non-stdlib {root}")
    assert not violations, "forbidden executor imports: " + ", ".join(violations)


def test_source_has_no_dynamic_forbidden_imports() -> None:
    """Literal dynamic imports must obey the same boundary.

    This deliberately does not inspect subprocess/CLI invocations or dynamic
    non-literal module names; the audit is limited to string literals whose
    first dotted segment can be checked statically.
    """
    violations: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            root = _dynamic_import_root(node)
            if root in FORBIDDEN_ROOTS or (root in CONFIGURATION_ROOTS and path.name not in {"configuration.py", "programs.py"}):
                violations.append(f"{path}:{node.lineno}: {root}")
            elif root is not None and root not in {"tpen_executor", *CONFIGURATION_ROOTS} and root not in sys.stdlib_module_names:
                violations.append(f"{path}:{node.lineno}: non-stdlib {root}")
    assert not violations, "forbidden dynamic executor imports: " + ", ".join(violations)
