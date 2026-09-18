"""Tests for the production HI train-config resolver."""

from __future__ import annotations

import builtins
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf


_STAGE_SPEC = importlib.util.spec_from_file_location(
    "he_importance_stage_coordinate_for_train_config",
    Path(__file__).with_name("stage_coordinate.py"),
)
assert _STAGE_SPEC is not None and _STAGE_SPEC.loader is not None
stage_coordinate = importlib.util.module_from_spec(_STAGE_SPEC)
sys.modules[_STAGE_SPEC.name] = stage_coordinate
_STAGE_SPEC.loader.exec_module(stage_coordinate)

_RESOLVER_SPEC = importlib.util.spec_from_file_location(
    "he_importance_train_config",
    Path(__file__).with_name("train_config.py"),
)
assert _RESOLVER_SPEC is not None and _RESOLVER_SPEC.loader is not None
train_config = importlib.util.module_from_spec(_RESOLVER_SPEC)
sys.modules[_RESOLVER_SPEC.name] = train_config
_RESOLVER_SPEC.loader.exec_module(train_config)


def _cell(tmp_path: Path, *, method: str = "adam", status: str = "available") -> object:
    reason = None if status == "available" else "qualification pending"
    return stage_coordinate.materialize_stage(
        "O1",
        [
            {
                "scientific_identity": {"architecture": "control", "optimizer": method},
                "payload": {"updates": 50_000},
                "topology": {},
            }
        ],
        stage_coordinate.OptimizerCell(method, status, reason),
        tmp_path,
    )[0]


def _callback_checkers(cfg: object) -> list[object]:
    return [
        checker
        for callback in cfg.callbacks
        for checker in callback.get("checkers", [])
    ]


def test_resolve_maps_stage_horizon_to_trainer_max_steps(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    resolved = train_config.resolve_train_config(cell)

    assert resolved.trainer.max_steps == stage_coordinate.stage_definition("O1").updates


def test_resolve_binds_content_hash_to_run_identity(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    resolved = train_config.resolve_train_config(cell)

    assert resolved.run.run_id == cell.content_hash


def test_resolve_binds_output_path_to_run_root_and_dir(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    resolved = train_config.resolve_train_config(cell)

    assert resolved.run.root == str(cell.output_path.parent.parent)
    assert resolved.run.layout == "flat"
    assert resolved.run.dir == str(cell.output_path)


def test_resolve_binds_seed_streams_to_runtime_and_named_initializers(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    resolved = train_config.resolve_train_config(cell)
    streams = cell.seed_streams

    assert resolved.runtime.seed == streams["model_initialization"]
    assert resolved.sampler.seed == streams["training_sampler"]
    assert resolved.model.embedding.initializer.seed == streams["model_initialization"]
    assert resolved.model.embedding.initializer.stream == "embedding"
    assert resolved.model.layers[0].path_aggregation.initializer.seed == streams["model_initialization"]
    assert resolved.model.layers[0].path_aggregation.initializer.stream == "layer0/path_aggregation"
    assert all(checker.seed == streams["diagnostic"] for checker in _callback_checkers(resolved))


def test_resolve_accepts_training_packet_wrapping_a_cell(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    packet = stage_coordinate.TrainingPacket(
        cell=cell,
        checkpoint_cadence=stage_coordinate.CheckpointCadence(1_000, (1_000,)),
        ddp_provenance={"launcher": "test"},
    )

    resolved = train_config.resolve_train_config(packet)

    assert resolved.run.run_id == cell.content_hash


def test_resolve_refuses_a_source_without_a_materialized_cell() -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="source"):
        train_config.resolve_train_config(object())


def test_resolve_refuses_a_hash_that_does_not_bind_the_manifest(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    broken = replace(cell, content_hash="0" * 64)

    with pytest.raises(train_config.TrainConfigResolutionError, match="content hash"):
        train_config.resolve_train_config(broken)


def test_resolve_refuses_a_non_absolute_output_path(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    broken = replace(cell, output_path=Path("relative-output"))

    with pytest.raises(train_config.TrainConfigResolutionError, match="absolute"):
        train_config.resolve_train_config(broken)


def test_resolve_refuses_missing_model_initialization_seed(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    streams = dict(cell.seed_streams)
    del streams["model_initialization"]
    broken = replace(cell, seed_streams=streams)

    with pytest.raises(train_config.TrainConfigResolutionError, match="model_initialization"):
        train_config.resolve_train_config(broken)


def test_resolve_refuses_an_unnamed_initializer_from_the_production_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    original_compose = train_config._compose_base_config

    def malformed_config() -> object:
        config = original_compose()
        config.model.embedding.initializer.stream = None
        return config

    monkeypatch.setattr(train_config, "_compose_base_config", malformed_config)
    with pytest.raises(train_config.TrainConfigResolutionError, match="named stream"):
        train_config.resolve_train_config(cell)


def test_resolve_binds_admitted_optimizer_cell_through_live_roster(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    resolved = train_config.resolve_train_config(cell)
    from tpen.hi_schema import HI_METHOD_ROSTER

    method = cell.manifest["scientific_identity"]["optimizer_cell"]["method"]
    entry = {candidate.method: candidate for candidate in HI_METHOD_ROSTER}[method]
    assert entry.admitted
    assert resolved.optimizer._target_ == entry.target


def test_resolve_refuses_unavailable_optimizer_cell_without_adam_substitution(tmp_path: Path) -> None:
    cell = _cell(tmp_path, method="linear_method", status="unavailable")

    with pytest.raises(train_config.UnavailableOptimizerError, match="unavailable"):
        train_config.resolve_train_config(cell)

    assert cell.manifest["scientific_identity"]["optimizer_cell"]["status"] == "unavailable"


def test_resolve_refuses_stage_without_fixed_training_horizon(tmp_path: Path) -> None:
    cell = stage_coordinate.materialize_stage(
        "F",
        [
            {
                "scientific_identity": {"architecture": "control", "optimizer": "adam"},
                "payload": {"updates": 50_000},
                "topology": {},
            }
        ],
        stage_coordinate.OptimizerCell("adam", "available"),
        tmp_path,
    )[0]

    with pytest.raises(train_config.TrainConfigResolutionError, match="no fixed training horizon"):
        train_config.resolve_train_config(cell)


def test_resolved_config_passes_hi_schema_and_contains_no_reference_energy(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    resolved = train_config.resolve_train_config(cell)
    serialized = json.dumps(OmegaConf.to_container(resolved, resolve=True), sort_keys=True)

    assert "-2.903724377034119598" not in serialized
    assert "reference_energy" not in serialized
    assert "evaluation.yaml" not in serialized


def test_resolver_routes_schema_validation_through_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    schema = __import__("tpen.hi_schema", fromlist=["validate_hi_train_config"])
    original = schema.validate_hi_train_config
    called: list[object] = []

    def recording_validator(config: object) -> None:
        called.append(config)
        original(config)

    monkeypatch.setattr(schema, "validate_hi_train_config", recording_validator)
    train_config.resolve_train_config(cell)

    assert called and called[0] is not None


def test_resolver_does_not_read_evaluation_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    original_open = builtins.open
    original_path_open = Path.open

    def guarded_open(file: object, *args: object, **kwargs: object) -> object:
        if str(file).endswith("manifests/evaluation.yaml"):
            raise AssertionError("resolver read the evaluation manifest")
        return original_open(file, *args, **kwargs)

    def guarded_path_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.as_posix().endswith("manifests/evaluation.yaml"):
            raise AssertionError("resolver read the evaluation manifest")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    train_config.resolve_train_config(cell)


def test_resolver_requires_repo_root_on_sys_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cell = _cell(tmp_path)
    repo_root = Path(train_config.__file__).resolve().parents[3]
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if Path(entry or Path.cwd()).resolve() != repo_root],
    )

    with pytest.raises(train_config.CheckoutPreconditionError, match="checkout root"):
        train_config.resolve_train_config(cell)
