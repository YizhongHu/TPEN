"""Import and statistics boundary tests for the Stage-1 collector."""

import ast
from pathlib import Path


def test_collector_imports_only_toolkit_reader_and_registry() -> None:
    tree = ast.parse(Path(__file__).parents[1].joinpath("collect.py").read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name for node in imports for alias in node.names}
    assert "tpen" not in names
    assert "experiments.toolkit.trajectory_records" in names
