"""Stage-Q integration checks for the closed attempt-delta seam."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest


runner = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")
delta_module = importlib.import_module("experiments.atomistic.he-importance.attempt_delta")


def _delta(mode: str = "fresh") -> SimpleNamespace:
    return SimpleNamespace(
        attempt_core_dir=Path("/store/groups/g/task/attempt/core"),
        attempt_id="attempt",
        group_digest="g",
        load_mode=mode,
        load_path=None if mode == "fresh" else Path("/store/groups/g/task/previous/core/checkpoint"),
        task_id="task",
    )


def test_delta_load_is_one_closed_runner_step() -> None:
    config = OmegaConf.create({"runner": {}})
    runner._apply_attempt_delta_load(config, _delta("train_resume"))
    assert OmegaConf.to_container(config.runner.load, resolve=True) == {
        "mode": "train_resume",
        "path": "/store/groups/g/task/previous/core/checkpoint",
    }


def test_fresh_delta_does_not_add_a_runner_load() -> None:
    config = OmegaConf.create({"runner": {}})
    runner._apply_attempt_delta_load(config, _delta("fresh"))
    assert OmegaConf.select(config, "runner.load", default=None) is None


def test_c_comparison_keeps_science_and_removes_only_operational_paths() -> None:
    report = {
        "selection": {"selected": {"context": "O-raw"}},
        "walkers": 4096,
        "roster_vs_design": {"status": "PASS"},
        "train_packet": {"content_hash": "row"},
        "resolved_config": {
            "trainer": {"max_steps": 2000},
            "run": {"dir": "/attempt/core", "root": "/attempt", "run_id": "x", "layout": "flat"},
        },
    }
    c = runner._cell_science_report(report)
    assert c["resolved_config"]["trainer"] == {"max_steps": 2000}
    assert c["resolved_config"]["run"] == {"layout": "flat"}


def test_attempt_delta_parser_rejects_before_any_stage_work() -> None:
    raw = '{"attempt_core_dir":"/store/groups/g/task/attempt/core","attempt_id":"attempt","group_digest":"g","load":{"mode":"fresh","path":null},"schema":"wrong","task_id":"task"}'
    with pytest.raises(delta_module.AttemptDeltaError):
        delta_module.parse_attempt_delta(raw)
