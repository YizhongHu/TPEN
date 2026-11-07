"""Enforce the H-v2 experiment boundary: config/``run_from_config``/``tpen`` only.

Operator ruling (root note ``operator-decisions-20261106-imports-mirflow-readonly``,
item 1): experiment code and tests reach core only through configuration,
``tpen.run.run_from_config``, or the installed ``tpen`` command. An
integration test that needs core internals lives in the core ``tests/`` tree
instead (item 2). This test parses every ``.py`` file under this directory
(recursively) with ``ast`` and fails on any ``tpen`` import other than
``tpen.run.run_from_config``, on any import of ``hydra`` (direct or via
``hydra.utils``), and on reaching either module through a literal-string
dynamic import (``importlib.import_module``, ``__import__``,
``pytest.importorskip``, ``importlib.util.spec_from_file_location``).

Amendment pre-review round 1, finding 1: the prior non-recursive, args[0]-only
scanner missed a nested file and `spec_from_file_location`'s path argument
(which is not the call's first argument). This version sweeps recursively and
inspects every string-constant argument of a candidate dynamic-import call,
and self-checks that the swept corpus is real so a mis-resolved ``HERE`` (or
an accidentally empty directory) fails loudly instead of certifying nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).parent

#: The only tpen import allowed under this directory.
ALLOWED_TPEN_IMPORT = ("tpen.run", "run_from_config")


def _chain(node: ast.AST) -> str | None:
    """Resolve a dotted AST name without evaluating source code."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _chain(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _literal(node: ast.AST) -> str | None:
    """Return a literal string, if present."""

    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    importlib_names = {"importlib"}
    import_module_names: set[str] = set()
    spec_from_file_location_names: set[str] = set()
    pytest_names = {"pytest"}
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if alias.name == "importlib":
                    importlib_names.add(bound)
                if alias.name == "pytest":
                    pytest_names.add(bound)
                if alias.name == "hydra" or alias.name.startswith("hydra."):
                    found.append(f"{path.name}: import {alias.name}")
                if alias.name == "tpen" or alias.name.startswith("tpen."):
                    found.append(f"{path.name}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                bound = alias.asname or alias.name
                if module == "importlib" and alias.name == "import_module":
                    import_module_names.add(bound)
                if module == "importlib" and alias.name == "util":
                    importlib_names.add(bound)
                if module == "importlib.util" and alias.name == "spec_from_file_location":
                    spec_from_file_location_names.add(bound)
                if module == "hydra" or module.startswith("hydra."):
                    found.append(f"{path.name}: from {module} import {alias.name}")
                if module == "tpen" or module.startswith("tpen."):
                    if module == ALLOWED_TPEN_IMPORT[0] and alias.name == ALLOWED_TPEN_IMPORT[1]:
                        continue
                    found.append(f"{path.name}: from {module} import {alias.name}")
        elif isinstance(node, ast.Call) and node.args:
            function = _chain(node.func)
            dynamic = (
                function in import_module_names
                or (function and function.endswith(".import_module") and function.split(".")[0] in importlib_names)
                or (function == "__import__")
                or (function and function.endswith(".__import__"))
                or (function and function.endswith(".importorskip") and function.split(".")[0] in pytest_names)
                or (function and function.endswith(".spec_from_file_location") and function.split(".")[0] in importlib_names)
                or (function in spec_from_file_location_names)
            )
            if dynamic:
                for arg in node.args:
                    name = _literal(arg)
                    if name is not None and (name.startswith(("tpen", "hydra")) or "/tpen/" in name):
                        found.append(f"{path.name}: dynamic {function}({name!r})")
                        break
    return found


def _swept() -> tuple[Path, ...]:
    """Return the recursive, non-cache Python corpus under this directory."""

    return tuple(sorted(path for path in HERE.rglob("*.py") if "__pycache__" not in path.parts))


def test_experiment_files_only_reach_core_through_run_from_config() -> None:
    paths = _swept()
    names = {path.name for path in paths}
    required = {"protocol.py", "collect.py", "test_pipeline.py", "test_collect.py", "conftest.py"}
    assert required <= names, f"swept corpus looks wrong, found only: {sorted(names)}"
    violations = [hit for path in paths for hit in _violations(path)]
    assert not violations, "experiment boundary violated:\n" + "\n".join(violations)
