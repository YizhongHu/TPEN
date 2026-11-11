"""Round-2 residual probes for the H-v2 experiment import boundary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "h_v2_r2_boundary_under_test", HERE / "test_import_boundary.py"
)
assert spec is not None and spec.loader is not None
boundary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = boundary
spec.loader.exec_module(boundary)


@pytest.mark.parametrize(
    ("number", "source"),
    [
        (
            "R1",
            'from importlib import util\nspec = util.spec_from_file_location("k", "../../../tpen/physics/kinetic.py")',
        ),
        (
            "R2",
            'from importlib.util import spec_from_file_location\nspec = spec_from_file_location("k", "../../../tpen/physics/kinetic.py")',
        ),
        (
            "R3",
            'from importlib import import_module as im\nm = im("tpen.physics.kinetic")',
        ),
        (
            "R4",
            'import importlib as il\nm = il.import_module("tpen.physics.kinetic")',
        ),
        (
            "R5",
            'import builtins\nm = builtins.__import__("tpen.physics.kinetic")',
        ),
    ],
)
def test_residual_dynamic_import_forms(number: str, source: str, tmp_path: Path) -> None:
    path = tmp_path / f"{number}.py"
    path.write_text(source, encoding="utf-8")
    assert boundary._violations(path), f"residual probe {number} was missed"
