"""PR #572 AB-4a renderer pre-review round 2 tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "he_importance_ab_universe_review_r2", HERE / "ab_universe.py"
)
assert spec is not None and spec.loader is not None
ab_universe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ab_universe
spec.loader.exec_module(ab_universe)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_a_stream_key_is_per_row_and_round_trips() -> None:
    universe = ab_universe.render_ab_universe()
    a_packages, b_packages = universe["A"], universe["B"]
    assert len(a_packages) == 254
    for package in a_packages:
        key = package["model_initialization_stream_key"]
        assert set(key) == {"stage", "literal_configuration"}
        assert key["stage"] == "A"
        loaded = json.loads(key["literal_configuration"])
        assert loaded == package["literal_configuration"]
        assert "package" not in loaded
    assert len({_canonical(p["model_initialization_stream_key"]) for p in a_packages}) == 254

    # Round-1 F6 shape: producer/update falsely pairs distinct A rows.
    mutant_keys = [
        {"stage": "A", "producer_path": p["literal_configuration"]["producer_path"],
         "update": p["literal_configuration"]["update"]}
        for p in a_packages
    ]
    with pytest.raises(AssertionError):
        assert len({_canonical(key) for key in mutant_keys}) == 254

    assert len({_canonical(p["model_initialization_stream_key"]) for p in b_packages}) == 15
    assert all(
        set(p["model_initialization_stream_key"]) == {"stage", "producer_path", "update"}
        for p in b_packages
    )
