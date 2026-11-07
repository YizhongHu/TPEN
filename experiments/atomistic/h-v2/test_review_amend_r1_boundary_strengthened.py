"""Strengthened recursive H-v2 experiment-boundary candidate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

HERE = Path(__file__).parent


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
    """Find forbidden direct, Hydra, and selected dynamic imports."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    importlib_names = {"importlib"}
    import_module_names: set[str] = set()
    pytest_names = {"pytest"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if alias.name == "importlib":
                    importlib_names.add(bound)
                if alias.name == "pytest":
                    pytest_names.add(bound)
                if alias.name == "hydra" or alias.name.startswith("hydra."):
                    violations.append(f"{path.name}: import {alias.name}")
                if alias.name == "tpen" or alias.name.startswith("tpen."):
                    violations.append(f"{path.name}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                bound = alias.asname or alias.name
                if module == "importlib" and alias.name == "import_module":
                    import_module_names.add(bound)
                if module == "hydra" or module.startswith("hydra."):
                    violations.append(f"{path.name}: from {module} import {alias.name}")
                if module == "tpen" or module.startswith("tpen."):
                    if not (module == "tpen.run" and alias.name == "run_from_config"):
                        violations.append(f"{path.name}: from {module} import {alias.name}")
        elif isinstance(node, ast.Call) and node.args:
            function = _chain(node.func)
            dynamic = (
                function in import_module_names
                or (function and function.endswith(".import_module") and function.split(".")[0] in importlib_names)
                or (function == "__import__")
                or (function and function.endswith(".__import__"))
                or (function and function.endswith(".importorskip") and function.split(".")[0] in pytest_names)
                or (function and function.endswith(".spec_from_file_location") and function.split(".")[0] in importlib_names)
            )
            name = _literal(node.args[0])
            if dynamic and name is not None and (name.startswith(("tpen", "hydra")) or "/tpen/" in name):
                violations.append(f"{path.name}: dynamic {function}({name!r})")
    return violations


def _swept() -> tuple[Path, ...]:
    """Return the recursive, non-cache Python corpus."""

    return tuple(sorted(path for path in HERE.rglob("*.py") if "__pycache__" not in path.parts))


def test_strengthened_boundary_sweeps_real_corpus() -> None:
    """Enforce the recursive boundary and prove the corpus is real."""

    paths = _swept()
    names = {path.name for path in paths}
    required = {"protocol.py", "collect.py", "test_pipeline.py", "test_collect.py", "conftest.py"}
    assert required <= names
    findings = [hit for path in paths for hit in _violations(path)]
    assert any("run_from_config" not in hit and "tpen.run" in hit for hit in findings) is False
    assert any("tpen.run" in path.read_text(encoding="utf-8") and "run_from_config" in path.read_text(encoding="utf-8") for path in paths)
    assert not findings, "experiment boundary violated:\n" + "\n".join(findings)


@pytest.mark.parametrize(
    "source",
    [
        'import importlib\nm = importlib.import_module("tpen.physics.kinetic")',
        'k = __import__("tpen.physics.kinetic")',
        'import pytest\nk = pytest.importorskip("tpen.physics.kinetic")',
        'import hydra\nobj = hydra.utils.instantiate({})',
        'from hydra import utils\nobj = utils.instantiate({})',
        'from hydra.utils import call',
        'from hydra.utils import get_class',
        'import importlib.util\nspec = importlib.util.spec_from_file_location("k", "../../../tpen/physics/kinetic.py")',
    ],
)
def test_strengthened_scanner_detects_dynamic_forms(source: str, tmp_path: Path) -> None:
    """All dynamic forms in the review matrix are rejected."""

    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    assert _violations(path)


def test_strengthened_controls() -> None:
    """The allowed launcher import survives while core imports do not."""

    assert _violations(Path(__file__)) == []
    assert _violations(HERE / "test_import_boundary.py") == []
