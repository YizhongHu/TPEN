"""Keep the executor package independent from TPEN and optional backends."""

import ast
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "tpen_executor"
FORBIDDEN_ROOTS = {"tpen", "experiments", "torch", "parsl", "balsam"}
CONFIGURATION_ROOTS = {"hydra", "omegaconf"}
OPTIONAL_IMPORT_PATH = Path("dispatch/parsl.py")
BALSAM_IMPORT_PATH = Path("lifecycle/balsam.py")


def _allowed_balsam_import(path: Path, node: ast.AST, tree: ast.Module) -> bool:
    """Allow exactly one static SDK import in the named top-level factory."""
    imports = [candidate for candidate in ast.walk(tree)
               if isinstance(candidate, (ast.Import, ast.ImportFrom))
               and "balsam" in _import_roots(candidate)]
    factories = [candidate for candidate in tree.body
                 if isinstance(candidate, ast.FunctionDef) and candidate.name == "make_group_application"]
    return (
        path == BALSAM_IMPORT_PATH and len(imports) == len(factories) == 1
        and node is imports[0] and node in list(ast.walk(factories[0]))
        and isinstance(node, ast.ImportFrom) and node.level == 0
        and node.module == "balsam" and len(node.names) == 1
        and node.names[0].name == "api" and node.names[0].asname == "loaded_api"
    )


def _constructor_lines(tree: ast.AST) -> set[int]:
    """Return source lines inside ``ParslDispatcher.__post_init__``."""

    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__post_init__":
            lines.update(child.lineno for child in ast.walk(node) if hasattr(child, "lineno"))
    return lines


def _allowed_parsl_import(path: Path, node: ast.AST, constructor_lines: set[int]) -> bool:
    """Allow only constructor-local Parsl imports in the optional adapter."""

    return path == OPTIONAL_IMPORT_PATH and getattr(node, "lineno", -1) in constructor_lines


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
        constructor_lines = _constructor_lines(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.level > 0:
                    continue
                for root in _import_roots(node):
                    relative = path.relative_to(SOURCE_ROOT)
                    if root == "parsl" and _allowed_parsl_import(relative, node, constructor_lines):
                        continue
                    if root == "balsam" and _allowed_balsam_import(relative, node, tree):
                        continue
                    if root in FORBIDDEN_ROOTS or (root in CONFIGURATION_ROOTS and relative not in {Path("configuration.py"), Path("programs.py")}):
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
        constructor_lines = _constructor_lines(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            root = _dynamic_import_root(node)
            relative = path.relative_to(SOURCE_ROOT)
            if root == "parsl" and _allowed_parsl_import(relative, node, constructor_lines):
                continue
            if root in FORBIDDEN_ROOTS or (root in CONFIGURATION_ROOTS and relative not in {Path("configuration.py"), Path("programs.py")}):
                violations.append(f"{path}:{node.lineno}: {root}")
            elif root is not None and root not in {"tpen_executor", *CONFIGURATION_ROOTS} and root not in sys.stdlib_module_names:
                violations.append(f"{path}:{node.lineno}: non-stdlib {root}")
    assert not violations, "forbidden dynamic executor imports: " + ", ".join(violations)


def test_balsam_allowlist_is_exact() -> None:
    tree = ast.parse((SOURCE_ROOT / BALSAM_IMPORT_PATH).read_text())
    imports = [node for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom)) and "balsam" in _import_roots(node)]
    assert len(imports) == 1
    assert _allowed_balsam_import(BALSAM_IMPORT_PATH, imports[0], tree)
    assert not _allowed_balsam_import(Path("other.py"), imports[0], tree)
    for source in (
        "from balsam import api as loaded_api",
        "def other():\n    from balsam import api as loaded_api",
        "def make_group_application():\n    import balsam",
        "def make_group_application():\n    from balsam import api as loaded_api\n    from balsam import api as loaded_api",
    ):
        invalid = ast.parse(source)
        assert not any(_allowed_balsam_import(BALSAM_IMPORT_PATH, node, invalid) for node in ast.walk(invalid))
