"""PR #595 round-2 tests for pre-resolution train-config overlays."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

import pytest


def _load(name: str, filename: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage_coordinate = _load("he_importance_stage_coordinate_for_pr595_r2", "stage_coordinate.py")
train_config = _load("he_importance_train_config_for_pr595_r2", "train_config.py")


def _cell(tmp_path: Path) -> Any:
    levels = {
        "lr": 0.005,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "parameter_groups": 1,
        "schedule": "constant",
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


def test_t9a_interpolation_override_is_preserved(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    resolved = train_config.resolve_train_config(
        cell, overrides=["loggers.0.path=${run.dir}/custom.csv"]
    )
    assert resolved.loggers[0].path == str(cell.output_path / "custom.csv")


def test_t9b_unresolvable_interpolation_is_a_resolution_error(tmp_path: Path) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError):
        train_config.resolve_train_config(
            _cell(tmp_path), overrides=["trainer.max_steps=${no.such.key}"]
        )


def test_t10_run_id_override_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="run.run_id"):
        resolved = train_config.resolve_train_config(
            _cell(tmp_path), overrides=["run.run_id=forged"]
        )
        assert resolved.run.run_id == "forged"


def test_t11_walker_count_override_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="sampler.n_walkers"):
        train_config.resolve_train_config(_cell(tmp_path), overrides=["sampler.n_walkers=8"])


def test_t12_layout_override_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="run.layout"):
        resolved = train_config.resolve_train_config(
            _cell(tmp_path), overrides=["run.layout=nested"]
        )
        assert resolved.run.layout == "nested"
