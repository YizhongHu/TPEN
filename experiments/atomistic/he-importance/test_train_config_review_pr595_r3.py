"""PR #595 round-3 probes for list-valued override readback."""

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


stage_coordinate = _load("he_importance_stage_coordinate_for_pr595_r3", "stage_coordinate.py")
train_config = _load("he_importance_train_config_for_pr595_r3", "train_config.py")


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


def test_t14_list_valued_override_equal_to_committed_value_is_preserved(tmp_path: Path) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError) as captured:
        resolved = train_config.resolve_train_config(
            _cell(tmp_path), overrides=["system.nuclei.charges=[2.0]"]
        )
        assert list(resolved.system.nuclei.charges) == [2.0]
        assert list(resolved.atoms.charges.data) == [2.0]

    assert type(captured.value).__name__ == "TrainConfigResolutionError"
    assert str(captured.value) == (
        "override path 'system.nuclei.charges' was not preserved before resolution"
    )


def test_t15_scalar_override_is_preserved_and_read_back(tmp_path: Path) -> None:
    resolved = train_config.resolve_train_config(
        _cell(tmp_path), overrides=["trainer.max_steps=7"]
    )
    assert resolved.runner.trainer.max_steps == 7
