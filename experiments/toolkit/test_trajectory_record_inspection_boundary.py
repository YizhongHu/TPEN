"""Experiment-boundary contract for the historical trajectory-record reader.

Per ``experiments/README.md``, the experiments/toolkit shared results reader
may import core's public read-only inspection interface (``tpen.inspection``)
and nothing else from ``tpen``. This test exercises the S1-F historical
reader exactly as a study collector would reach it: through
``tpen.inspection`` only, never through ``tpen.evaluation`` directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tpen.inspection import TrajectoryRecordInspection, inspect_trajectory_record

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_METADATA = (
    _ROOT
    / "tests"
    / "integration"
    / "artifacts"
    / "trajectory_records"
    / "h_like_frozen_main"
    / "sampled_eval_table.metadata.json"
)


def test_this_module_imports_only_tpen_inspection() -> None:
    """Static self-check: this file names no ``tpen`` import but ``tpen.inspection``."""

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    tpen_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "tpen" or node.module.startswith("tpen.")
        ):
            tpen_imports.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tpen" or alias.name.startswith("tpen."):
                    tpen_imports.add(alias.name)
    assert tpen_imports == {"tpen.inspection"}


def test_inspect_trajectory_record_reachable_through_public_inspection() -> None:
    if not _FIXTURE_METADATA.is_file():
        pytest.skip(
            "h_like_frozen_main fixture is not generated in this checkout; "
            "see tests/integration/artifacts/trajectory_records/generate_v1_fixtures.py"
        )
    result = inspect_trajectory_record(_FIXTURE_METADATA)
    assert isinstance(result, TrajectoryRecordInspection)
    assert result.physical_context.available is False
