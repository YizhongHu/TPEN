"""HI-S5 mutable-state inventory and continuation-refusal controls."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


inventory = importlib.import_module(
    "experiments.atomistic.he-importance.mutable_state_inventory"
)
launch = importlib.import_module("experiments.atomistic.he-importance.launch")


def test_real_hi_train_config_is_fully_classified() -> None:
    entries = inventory.validate_mutable_state_inventory()
    assert entries
    assert {entry.classification for entry in entries} <= {
        "restored",
        "reconstructible",
        "allocation-local",
        "typed-refused",
    }
    assert {"sampler", "trainer"} <= {entry.path for entry in entries}
    refused = inventory.refused_inventory_entries()
    assert {entry.target for entry in refused} >= {
        "tpen.callback.RuntimeEquivariance",
        "tpen.callback.ResourceUsage",
    }
    assert all(":line " in entry.source for entry in refused)


def test_native_profile_refuses_hi_train_config_at_expected_entries() -> None:
    from tpen.execution.native_profile import (
        UnsupportedMutableStateError,
        validate_native_recovery_profile,
    )
    train_config = importlib.import_module(
        "experiments.atomistic.he-importance.train_config"
    )

    with pytest.raises(UnsupportedMutableStateError, match=r"callbacks\[8\]"):
        validate_native_recovery_profile(train_config._compose_base_config())


def test_continuation_refuses_before_runner_or_leaf_write(tmp_path: Path) -> None:
    predecessor = tmp_path / "predecessor-leaf"
    predecessor.mkdir()
    marker = predecessor / "bytes.bin"
    marker.write_bytes(b"fixture predecessor leaf; no real HI run")
    before = marker.read_bytes()

    with pytest.raises(launch.HIContinuationRefusal, match="hi-continuation-refused"):
        launch.refuse_hi_continuation(object())

    assert marker.read_bytes() == before
    assert list(predecessor.iterdir()) == [marker]


def test_two_process_successor_refusal_preserves_fixture_leaf(tmp_path: Path) -> None:
    predecessor = tmp_path / "predecessor-leaf"
    predecessor.mkdir()
    marker = predecessor / "state"
    marker.write_bytes(b"fixture predecessor leaf")
    before = marker.read_bytes()
    code = (
        "import importlib; "
        "m=importlib.import_module('experiments.atomistic.he-importance.launch'); "
        "m.refuse_hi_continuation(object())"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=env["PYTHONPATH"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "hi-continuation-refused" in result.stderr
    assert marker.read_bytes() == before


def test_contract_mutants_are_mapped_to_refusal() -> None:
    assert set(inventory.MUTANT_REFUSAL_MAP) == {
        "M1-omitted-sampler-rng",
        "M2-reset-cadence-counter",
        "M3-callback-private-rng-drift",
        "M4-event-before-restore",
        "M5-missing-state-classification",
    }
    assert all("refused" in result for result in inventory.MUTANT_REFUSAL_MAP.values())
