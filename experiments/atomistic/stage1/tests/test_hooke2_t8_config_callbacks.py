"""Acceptance tests for the Hooke-2e T8 arm callback contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import _binding_token
from experiments.atomistic.stage1.protocol import resolve_config_path
from experiments.toolkit.run_paths import resolve_run_dir


CONFIG_ROOT = Path(__file__).parents[1] / "configs" / "hooke2"
ARMS = ("positive_singlet_tail", "positive_singlet_comparator")


def _config(name: str):
    return OmegaConf.load(CONFIG_ROOT / f"{name}.yaml")


def test_t8_arms_match_train_checkpoint_contract() -> None:
    """Each arm retains train.yaml's callback types and checkpoint wiring."""

    standard = _config("train")
    expected_callbacks = OmegaConf.to_container(standard.callbacks, resolve=False)
    for arm in ARMS:
        cfg = _config(arm)
        assert list(cfg.checkpoint_steps) == list(standard.checkpoint_steps)
        assert OmegaConf.to_container(cfg.callbacks, resolve=False) == expected_callbacks

        checkpoint = next(
            callback for callback in cfg.callbacks
            if callback._target_ == "tpen.callback.Checkpoint"
        )
        assert checkpoint.schedule._target_ == "tpen.checkpoint.ExplicitUpdates"
        assert list(checkpoint.schedule.updates) == list(standard.checkpoint_steps)
        assert checkpoint.payload._target_ == "tpen.checkpoint.TrainResume"


def test_every_t8_arm_declares_a_train_resume_checkpoint() -> None:
    """The parity guard names every arm, so a new arm cannot omit the scaffold."""

    train = _config("train")
    for arm in ARMS:
        cfg = _config(arm)
        assert list(cfg.checkpoint_steps) == list(train.checkpoint_steps)
        checkpoints = [
            callback for callback in cfg.callbacks
            if callback._target_ == "tpen.callback.Checkpoint"
        ]
        assert len(checkpoints) == 1
        assert checkpoints[0].payload._target_ == "tpen.checkpoint.TrainResume"
        assert list(checkpoints[0].schedule.updates) == list(cfg.checkpoint_steps)


@pytest.mark.parametrize("arm", ARMS)
def test_native_train_planned_yield_publishes_train_resume_checkpoint(
    arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native train gateway must yield only after a full checkpoint is published."""

    row = {
        "kind": "train",
        "system": "hooke2",
        "seed": 17,
        "config": f"configs/hooke2/{arm}.yaml",
        "results_root": str(tmp_path / "results"),
        "experiment_name": "stage1_hooke2",
        "sector": "atomistic",
        "run_id": f"native-{arm}",
        "walkers": 2,
        "updates": 3,
        "checkpoint_steps": [0, 1, 2, 3],
        "device": "cpu",
    }
    assert resolve_config_path(row["config"]).is_file()

    monkeypatch.setattr(
        stage1_train.sys,
        "argv",
        ["stage1-train", "--binding", _binding_token(row)],
    )
    assert stage1_train.main() == 2

    run_dir = resolve_run_dir(row)
    outcome = json.loads((run_dir / "run-outcome.json").read_text(encoding="utf-8"))
    assert outcome["status"] == "planned_yield"
    assert outcome["next_iteration"] == 1
    assert outcome["completed_updates"] == 1
    assert outcome["checkpoint_ref"]["profile"] == "train_resume"
    checkpoint = run_dir / "checkpoints" / "step_000001"
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["payload"]["profile"] == "train_resume"
    assert (checkpoint / "COMPLETE").is_file()
