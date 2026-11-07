"""PR #595 review probes for pre-resolution HI operational overrides."""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

import pytest
from omegaconf import OmegaConf


def _load(name: str, filename: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage_coordinate = _load("he_importance_stage_coordinate_for_pr595_r1", "stage_coordinate.py")
train_config = _load("he_importance_train_config_for_pr595_r1", "train_config.py")
ab_universe = _load("he_importance_ab_universe_for_pr595_r1", "ab_universe.py")


def _cell(tmp_path: Path) -> Any:
    levels = {
        "lr": 0.005, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
        "weight_decay": 0.0, "parameter_groups": 1, "schedule": "constant",
    }
    return stage_coordinate.materialize_stage(
        "O1",
        [{
            "scientific_identity": {
                "context": {"name": "O-raw", "overrides": {"input_features": "B00", "feature_update": "residual"}},
                "optimizer": {"method": "adam", "levels": levels},
            },
            "payload": {"updates": stage_coordinate.stage_definition("O1").updates},
            "topology": {},
        }],
        stage_coordinate.OptimizerCell("adam", "available"),
        tmp_path,
    )[0]


def _ab_cell(tmp_path: Path, package: Mapping[str, Any], axis: str) -> Any:
    adam = next(slot for slot in package["optimizer_slots"] if slot["method"] == "adam")
    return stage_coordinate.materialize_stage(
        axis,
        [{
            "scientific_identity": {
                "context": {"axis": axis, "literal_configuration": dict(package["literal_configuration"])},
                "optimizer": {"method": adam["method"], "levels": dict(adam["levels"])},
            },
            "payload": {"updates": stage_coordinate.stage_definition(axis).updates},
            "topology": {},
        }],
        stage_coordinate.OptimizerCell(adam["method"], adam["status"]),
        tmp_path,
    )[0]


def test_t1_optimizer_override_is_not_silently_clobbered(tmp_path: Path) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="override"):
        resolved = train_config.resolve_train_config(_cell(tmp_path), overrides=["optimizer.lr=0.1"])
        assert resolved.optimizer.lr == 0.1
        assert resolved.runner.optimizer.lr == 0.1


def test_t1b_ab_model_override_is_preserved_or_refused(tmp_path: Path) -> None:
    package = ab_universe.render_a_packages()[0]
    with pytest.raises(train_config.TrainConfigResolutionError, match="override"):
        resolved = train_config.resolve_train_config(
            _ab_cell(tmp_path, package, "A"), overrides=["model.embedding.hidden_channels=999"]
        )
        assert resolved.model.embedding.hidden_channels == 999


@pytest.mark.parametrize("path", ["run.root", "run.dir"])
def test_t2_run_path_override_moves_all_sinks_or_is_refused(tmp_path: Path, path: str) -> None:
    new_root = tmp_path / "new-root"
    with pytest.raises(train_config.TrainConfigResolutionError, match=path):
        train_config.resolve_train_config(_cell(tmp_path), overrides=[f"{path}={new_root}"])


def test_t3_runtime_device_override_reaches_shared_consumer(tmp_path: Path) -> None:
    baseline = train_config.resolve_train_config(_cell(tmp_path / "baseline"))
    assert baseline.runtime.device == "cuda"
    assert baseline.callbacks[10].allocator_probe.device == "cuda"
    resolved = train_config.resolve_train_config(_cell(tmp_path / "override"), overrides=["runtime.device=cpu"])
    assert resolved.runtime.device == "cpu"
    assert resolved.callbacks[10].allocator_probe.device == "cpu"


def test_t4_mapping_overrides_propagate_to_runner_and_sinks(tmp_path: Path) -> None:
    cell = _cell(tmp_path / "cell")
    resolved = train_config.resolve_train_config(
        cell,
        overrides={"trainer": {"max_steps": 9}, "runtime": {"device": "cpu"}},
    )
    assert resolved.runner.trainer.max_steps == 9
    assert resolved.trainer.max_steps == 9
    assert resolved.callbacks[9].output_dir == str(cell.output_path / "checkpoints")
    assert resolved.loggers[1].path == str(cell.output_path / "metrics.jsonl")


def test_t5_checker_seed_override_is_refused(tmp_path: Path) -> None:
    base = train_config._compose_base_config()
    raw = OmegaConf.to_container(base, resolve=False)
    path = next(
        f"callbacks.{i}.checkers.{j}.seed"
        for i, callback in enumerate(raw["callbacks"])
        for j, checker in enumerate(callback.get("checkers", []))
        if isinstance(checker, Mapping) and "seed" in checker
    )
    with pytest.raises(train_config.TrainConfigResolutionError, match="resolver-owned seed"):
        train_config.resolve_train_config(_cell(tmp_path), overrides={path: 5})


@pytest.mark.parametrize("override", ["runner.trainer.max_steps=7", "runner.sampler.seed=5"])
def test_t6_runner_alias_override_is_explicitly_refused(tmp_path: Path, override: str) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match=r"top-level path: runner"):
        train_config.resolve_train_config(_cell(tmp_path), overrides=[override])
