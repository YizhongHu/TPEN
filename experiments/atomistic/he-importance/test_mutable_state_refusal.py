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
architecture_stage_family = importlib.import_module(
    "experiments.atomistic.he-importance.architecture_stage_family"
)
stage_coordinate = importlib.import_module(
    "experiments.atomistic.he-importance.stage_coordinate"
)
optimizer_grid = importlib.import_module(
    "experiments.atomistic.he-importance.optimizer_grid"
)


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


def test_restored_entries_name_real_train_resume_components() -> None:
    """Every restored disposition maps to a native TrainResume component."""

    entries = inventory.validate_mutable_state_inventory()
    restored = [entry for entry in entries if entry.classification == "restored"]
    assert restored
    assert {entry.target for entry in restored} == {
        "TrainResume.model",
        "TrainResume.optimizer",
        "TrainResume.trainer",
        "TrainResume.sampler",
        "TrainResume.rng",
    }


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


def test_real_resolved_rows_validate_aliases_and_refusal_owners(tmp_path: Path) -> None:
    """Validate resolver-produced O/A/B rows, including their runner aliases."""

    q_cells = stage_coordinate.materialize_intended_configurations(
        tmp_path / "q",
        availability=optimizer_grid.roster_availability(),
        stages=("Q",),
    )
    resolved_rows = {
        f"{context}-resolved": train_config.resolve_train_config(
            next(
                cell
                for cell in q_cells
                if cell.manifest["scientific_identity"]["context"]["name"] == context
                and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
            )
        )
        for context in ("O-raw", "O-composite")
    }
    for stage in ("A", "B"):
        cells = architecture_stage_family.materialize_family_stage(stage, tmp_path / stage)
        cell = next(
            item
            for item in cells
            if item.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        )
        resolved_rows[f"{stage}-resolved"] = train_config.resolve_train_config(cell)

    alias_pairs = ("model", "sampler", "trainer", "optimizer")
    expected_refused = {
        "tpen.callback.RuntimeEquivariance",
        "tpen.callback.ResourceUsage",
    }
    for label, config in resolved_rows.items():
        entries = inventory.validate_mutable_state_inventory(config)
        assert entries, label
        by_path = {entry.path: entry for entry in entries}
        for path in alias_pairs:
            assert by_path[f"runner.{path}"].classification == by_path[path].classification, label
        assert {
            entry.target
            for entry in entries
            if entry.classification == "typed-refused"
        } == expected_refused, label

    with pytest.raises(launch.HIContinuationRefusal) as refusal:
        launch.refuse_hi_continuation(object())
    message = str(refusal.value)
    assert "typed-refused mutable owners" in message
    assert all(target in message for target in expected_refused)
    assert "tpen.sampling.metropolis.MetropolisSampler" not in message
    assert "tpen.training.trainer.VMCTrainer" not in message


@pytest.mark.parametrize(
    ("stage", "producer_path", "activation"),
    [
        ("A", "linear/orbit-complete", "tanh,tanh"),
        ("A", "hybrid/orbit-complete", "exact GELU,exact GELU"),
        ("A", "hybrid/orbit-complete", "Softplus beta2,Softplus beta2"),
    ],
)
def test_real_resolved_rows_cover_admitted_constructor_shapes(
    tmp_path: Path, stage: str, producer_path: str, activation: str
) -> None:
    """Every representative orbit/activation constructor shape is resolved and classified."""

    cells = architecture_stage_family.materialize_family_stage(stage, tmp_path / stage)
    cell = next(
        item
        for item in cells
        if item.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and item.manifest["scientific_identity"]["context"]["literal_configuration"]["producer_path"] == producer_path
        and item.manifest["scientific_identity"]["context"]["literal_configuration"]["activation"] == activation
    )
    config = train_config.resolve_train_config(cell)
    entries = inventory.validate_mutable_state_inventory(config)
    constructors = {(entry.path, entry.target) for entry in entries}
    assert ("model.layers.0.layout.linear.explicit.7", "tpen.data.paths.SupportPath") in constructors
    activation_target = {
        "tanh,tanh": "torch.nn.Tanh",
        "exact GELU,exact GELU": "torch.nn.GELU",
        "Softplus beta2,Softplus beta2": "tpen.nn.SoftplusBeta",
    }[activation]
    assert ("model.layers.0.mixing.activation", activation_target) in constructors


@pytest.mark.parametrize("config", [{}, {"scalar": 1}])
def test_empty_or_scalar_only_config_refuses(config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="constructs no inventory objects"):
        inventory.validate_mutable_state_inventory(config)


def test_nonempty_config_still_validates_after_census_guard() -> None:
    assert inventory.validate_mutable_state_inventory()


@pytest.mark.parametrize("update", ["replacement", "Gaussian residual"])
def test_admitted_architecture_update_variants_are_classified(update: str) -> None:
    """Admitted update variants are explicit inventory constructors."""

    config = train_config._compose_base_config()
    config.model = context_mapping.compose_architecture_package(
        config.model,
        producer_path="hybrid/coordinate-neighbor",
        activation="SiLU,SiLU",
        update=update,
        package="B11",
        seed=0,
    )
    entries = inventory.validate_mutable_state_inventory(config)
    assert entries


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


@pytest.mark.parametrize("restore_payload", [{}, False, 0, ""])
def test_falsey_non_none_restore_requests_use_typed_refusal_before_validation(
    restore_payload: object,
) -> None:
    """Falsey continuation payloads cannot bypass the stable refusal code."""

    with pytest.raises(launch.HIContinuationRefusal, match="hi-continuation-refused"):
        launch.prepare_train_launch(
            object(), topology=object(), restore_request=restore_payload
        )


def test_subprocess_execute_refusal_preserves_predecessor_path(tmp_path: Path) -> None:
    """The real execute route leaves the complete predecessor tree untouched."""

    predecessor = tmp_path / "predecessor"
    predecessor.mkdir()
    marker = predecessor / "state"
    marker.write_bytes(b"predecessor bytes")
    before = {path.relative_to(predecessor): path.read_bytes()
              for path in predecessor.rglob("*") if path.is_file()}
    output_root = tmp_path / "successor"
    code = (
        "import importlib, sys; "
        "m=importlib.import_module('experiments.atomistic.he-importance.run_stage_q'); "
        "args=m.build_parser().parse_args(['--stage','O1','--context','O-raw',"
        "'--output-root',sys.argv[1]]); "
        "m.execute(args, restore_request={'predecessor': sys.argv[2]})"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    result = subprocess.run(
        [sys.executable, "-c", code, str(output_root), str(predecessor)],
        cwd=env["PYTHONPATH"], env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "hi-continuation-refused" in result.stderr
    assert {path.relative_to(predecessor): path.read_bytes()
            for path in predecessor.rglob("*") if path.is_file()} == before
    assert not output_root.exists()


def test_inventory_drift_is_typed_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inventory errors on restore are translated to the stable refusal."""

    original = train_config._compose_base_config

    def drifted_config():
        config = OmegaConf.to_container(original(), resolve=False)
        config["model"]["injected"] = {"_target_": "tests.DriftedConstructor"}
        return OmegaConf.create(config)

    monkeypatch.setattr(train_config, "_compose_base_config", drifted_config)
    with pytest.raises(launch.HIContinuationRefusal, match="hi-continuation-refused"):
        launch.prepare_train_launch(object(), restore_request=object())


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
