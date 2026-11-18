"""HI-S5 mutable-state inventory and continuation-refusal controls."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
from omegaconf import OmegaConf


inventory = importlib.import_module(
    "experiments.atomistic.he-importance.mutable_state_inventory"
)
launch = importlib.import_module("experiments.atomistic.he-importance.launch")
stage_q = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")
train_config = importlib.import_module("experiments.atomistic.he-importance.train_config")
context_mapping = importlib.import_module("experiments.atomistic.he-importance.context_mapping")


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
    assert {entry.target for entry in refused} == {
        "tpen.callback.RuntimeEquivariance",
        "tpen.callback.ResourceUsage",
    }
    assert all(":line " in entry.source for entry in refused)


def test_o_composite_hi_config_is_fully_classified() -> None:
    config = train_config._compose_base_config()
    config.model = context_mapping.compose_package(config.model, "B11")
    config.model.layers[0] = OmegaConf.create(
        context_mapping.context_fragment(
            "hybrid/coordinate-neighbor",
            activation="channel-MLP,channel-MLP",
            seed=0,
        )
    )
    entries = inventory.validate_mutable_state_inventory(config)
    assert any(entry.target == "tpen.nn.CompositeMixing" for entry in entries)


@pytest.mark.parametrize(
    ("path", "target"),
    [
        ("model.injected", "tests.NewConstructor"),
        ("hamiltonian_terms.injected", "tests.NewHamiltonian"),
    ],
)
def test_new_nested_constructor_fails_closed(path: str, target: str) -> None:
    config = OmegaConf.to_container(train_config._compose_base_config(), resolve=False)
    parent, _, key = path.rpartition(".")
    nested = config
    for part in parent.split("."):
        nested = nested[part]
    nested[key] = {"_target_": target}
    with pytest.raises(ValueError, match="unclassified HI constructor"):
        inventory.validate_mutable_state_inventory(config)


def test_registered_constructor_target_change_fails_closed() -> None:
    config = OmegaConf.to_container(train_config._compose_base_config(), resolve=False)
    config["model"]["_target_"] = "tests.ReplacementWaveFunction"
    with pytest.raises(ValueError, match="model.*ReplacementWaveFunction"):
        inventory.validate_mutable_state_inventory(config)


def test_native_profile_refuses_hi_train_config_at_expected_entries() -> None:
    native_profile = importlib.import_module("tpen." + "execution.native_profile")
    train_config = importlib.import_module(
        "experiments.atomistic.he-importance.train_config"
    )

    with pytest.raises(native_profile.UnsupportedMutableStateError, match=r"callbacks\[8\]"):
        native_profile.validate_native_recovery_profile(train_config._compose_base_config())


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


def test_prepare_and_launch_refuse_before_runner_construction() -> None:
    runner_calls: list[object] = []

    def runner(config: object) -> int:
        runner_calls.append(config)
        return 0

    with pytest.raises(launch.HIContinuationRefusal, match="hi-continuation-refused"):
        launch.prepare_train_launch(object(), restore_request=object())
    with pytest.raises(launch.HIContinuationRefusal, match="hi-continuation-refused"):
        launch.launch_train(object(), runner=runner, restore_request=object())
    assert runner_calls == []


def test_launch_refusal_precedes_malformed_topology(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-be-written"
    runner_calls: list[object] = []

    def runner(config: object) -> int:
        runner_calls.append(config)
        marker.write_text("runner reached", encoding="utf-8")
        return 0

    with pytest.raises(launch.HIContinuationRefusal, match="hi-continuation-refused"):
        launch.launch_train(
            object(),
            topology=object(),
            runner=runner,
            restore_request=object(),
        )

    assert runner_calls == []
    assert not marker.exists()


def test_prepare_refusal_precedes_malformed_topology() -> None:
    with pytest.raises(launch.HIContinuationRefusal, match="hi-continuation-refused"):
        launch.prepare_train_launch(
            object(), topology=object(), restore_request=object()
        )


def test_stage_q_execute_refuses_before_selection_or_leaf_write(tmp_path: Path, monkeypatch) -> None:
    calls: list[object] = []

    def launch_double(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        return 0

    monkeypatch.setattr(stage_q.launch, "launch_train", launch_double)
    args = stage_q.build_parser().parse_args(
        ["--stage", "O1", "--context", "O-raw", "--output-root", str(tmp_path)]
    )
    with pytest.raises(launch.HIContinuationRefusal, match="hi-continuation-refused"):
        stage_q.execute(args, restore_request=object())
    assert calls == []
    assert not any(tmp_path.iterdir())
