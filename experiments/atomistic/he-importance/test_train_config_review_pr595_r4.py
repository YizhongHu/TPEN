"""PR #595 round-4 probes for bracketed override path refusal."""

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


stage_coordinate = _load("he_importance_stage_coordinate_for_pr595_r4", "stage_coordinate.py")
train_config = _load("he_importance_train_config_for_pr595_r4", "train_config.py")


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


def test_t16_mapping_bracket_refusal_is_enforced(tmp_path: Path) -> None:
    with pytest.raises(
        train_config.TrainConfigResolutionError,
        match=r"run\[root\].*dotted segments",
    ):
        train_config.resolve_train_config(
            _cell(tmp_path), overrides={"run[root]": str(tmp_path / "elsewhere")}
        )


def test_t16b_mapping_bracket_path_is_refused() -> None:
    with pytest.raises(
        train_config.TrainConfigResolutionError,
        match=r"run\[root\].*dotted segments",
    ):
        train_config._override_items({"run[root]": "x"})


def test_t17_dotlist_bracket_path_is_refused() -> None:
    with pytest.raises(
        train_config.TrainConfigResolutionError,
        match=r"loggers\[0\]\.path.*dotted segments",
    ):
        train_config._override_items(["loggers[0].path=custom.csv"])


def test_dotted_logger_path_control_is_applied(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    resolved = train_config.resolve_train_config(
        cell, overrides=["loggers.0.path=${run.dir}/custom.csv"]
    )
    assert resolved.loggers[0].path == str(cell.output_path / "custom.csv")


def test_t19_dotlist_sequence_index_walk(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    resolved = train_config.resolve_train_config(
        cell, overrides=["callbacks.9.output_dir=${run.dir}/ckpt"]
    )
    assert resolved.callbacks[9].output_dir == str(cell.output_path / "ckpt")
