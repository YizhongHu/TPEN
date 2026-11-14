"""Import and statistics boundary tests for the Stage-1 collector."""

import ast
from pathlib import Path


def test_collector_imports_only_toolkit_reader_and_registry() -> None:
    tree = ast.parse(Path(__file__).parents[1].joinpath("collect.py").read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    modules = {node.module for node in imports if isinstance(node, ast.ImportFrom)}
    assert all(not module.startswith("tpen") for module in modules)
    assert "experiments.toolkit.trajectory_records" in modules
