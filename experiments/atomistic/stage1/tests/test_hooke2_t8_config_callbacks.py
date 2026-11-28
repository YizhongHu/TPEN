"""Acceptance tests for the Hooke-2e T8 arm callback contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import _binding_token
from experiments.atomistic.stage1.protocol import (
    Protocol,
    build_manifest,
    derive_smoke_request,
    resolve_config_path,
)
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.checkpoint import deserialize_checkpoint_ref
from tpen.run import PLANNED_YIELD_EXIT_CODE


CONFIG_ROOT = Path(__file__).parents[1] / "configs" / "hooke2"
ARMS = ("positive_singlet_tail", "positive_singlet_comparator")


def _config(name: str):
    return OmegaConf.load(CONFIG_ROOT / f"{name}.yaml")


def _planned_train_row(tmp_path: Path, arm: str) -> dict[str, object]:
    """Build one production-shaped train row and derive its bounded smoke."""

    planned = build_manifest(
        protocol=Protocol(
            smoke_updates=1,
            smoke_walkers=2,
            lineage_updates=3,
            lineage_walkers=2,
            lineages=3,
            eval_streams=4,
            eval_walkers=2,
            eval_draws=2,
        ),
        source_sha="a" * 40,
        # This callback contract test intentionally uses the legacy digest
        # shape: the opt-in arm is not the canonical Stage-1 train.yaml.
        config_sha256="b" * 64,
        checkpoints=[0, 1, 2, 3],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path / "planned"),
        systems=("hooke2",),
        phase=1,
        include_smoke=False,
    )
    source = next(row for row in planned if row["kind"] == "train")
    source["config"] = f"configs/hooke2/{arm}.yaml"
    return derive_smoke_request(
        source,
        results_root=str(tmp_path / f"smoke-{arm}"),
        stop_at=1,
    )


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

    row = _planned_train_row(tmp_path, arm)
    assert resolve_config_path(row["config"]).is_file()

    monkeypatch.setattr(
        stage1_train.sys,
        "argv",
        ["stage1-train", "--binding", _binding_token(row)],
    )
    exit_code = stage1_train.main()
    run_dir = resolve_run_dir(row)
    outcome = json.loads((run_dir / "run-outcome.json").read_text(encoding="utf-8"))
    assert PLANNED_YIELD_EXIT_CODE == 2  # tpen.run._status_to_exit_code convention
    # At the unfixed head, the same correctly bounded row reaches PlannedYield
    # without a publisher and must expose this exact refusal to the control.
    if exit_code != PLANNED_YIELD_EXIT_CODE:
        assert outcome["error"] == (
            "planned yield boundary lacks a published full-state checkpoint"
        )
    assert exit_code == PLANNED_YIELD_EXIT_CODE, outcome
    assert outcome["status"] == "planned_yield"
    stop_at = int(row["execution"]["stop_at"])
    assert outcome["next_iteration"] == stop_at
    assert outcome["completed_updates"] == stop_at
    checkpoint_ref = deserialize_checkpoint_ref(outcome["checkpoint_ref"])
    checkpoint = run_dir / "checkpoints" / f"step_{stop_at:06d}"
    assert checkpoint_ref.next_iteration == stop_at
    assert checkpoint_ref.completed_updates == stop_at
    assert checkpoint_ref.checkpoint_dir == checkpoint
    assert checkpoint.is_dir()
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["payload"]["profile"] == "train_resume"
    assert (checkpoint / "COMPLETE").is_file()
