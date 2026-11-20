"""Unit contracts for the Stage-1 restartable-start adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1.restart import (
    RestartApproval,
    RestartRefusal,
    StartDecision,
    configure_attempt_local_diagnostics,
    select_restart,
)


def _cfg() -> object:
    return OmegaConf.create(
        {
            "trainer": {"max_steps": 100000},
            "callbacks": [
                {"_target_": "tpen.callback.InitialCheckpoint", "output_dir": "old"},
                {"_target_": "tpen.callback.RawTrainingBatchRecorder", "path": "old-history"},
                {"_target_": "tpen.callback.ParameterActivity", "output_path": "old-activity"},
            ],
            "loggers": [{"_target_": "tpen.logging.JSONL", "path": "old-metrics.jsonl"}],
        }
    )


def _row(tmp_path: Path) -> dict[str, object]:
    return {
        "row_id": "he-lineage-0",
        "system": "he",
        "system_id": "he_atom",
        "lineage": 0,
        "seed": 11,
        "results_root": str(tmp_path),
        "experiment_name": "stage1_he",
        "sector": "atomistic",
        "run_id": "run-0",
        "updates": 100000,
        "config_sha256": "a" * 64,
    }


def test_fresh_selection_is_read_only_and_has_initial_marker(tmp_path: Path, monkeypatch) -> None:
    from experiments.atomistic.stage1 import restart

    monkeypatch.setattr(restart, "discover_published_checkpoint", lambda _: None)
    selected = select_restart(cfg=_cfg(), row=_row(tmp_path), run_dir=tmp_path / "run")
    assert selected.decision is StartDecision.FRESH
    assert selected.content_id == "initial"
    assert not (tmp_path / "run").exists()


def test_attempt_diagnostics_are_never_shared_with_previous_attempt(tmp_path: Path) -> None:
    cfg = _cfg()
    configure_attempt_local_diagnostics(cfg, tmp_path / "attempt-2")
    assert cfg.callbacks[1].path == str(tmp_path / "attempt-2" / "training_history")
    assert cfg.callbacks[2].output_path == str(tmp_path / "attempt-2" / "parameter_activity.json")
    assert cfg.loggers[0].path == str(tmp_path / "attempt-2" / "metrics.jsonl")


def test_takeover_requires_terminal_quiescent_budget_and_identity() -> None:
    approval = RestartApproval(
        predecessor_content_id="content",
        terminal_pbs_proof="pbs-terminal",
        quiescent=True,
        adopted_budget="budget-1",
        lineage_id="lineage",
        config_identity="config",
        overlay_identity="overlay",
        target_iterations=100000,
    )
    approval.validate(
        expected={
            "predecessor_content_id": "content",
            "lineage_id": "lineage",
            "config_identity": "config",
            "overlay_identity": "overlay",
            "target_iterations": 100000,
        }
    )
    with pytest.raises(RestartRefusal, match="quiescence"):
        RestartApproval(
            predecessor_content_id="content", terminal_pbs_proof="proof",
            quiescent=False, adopted_budget="budget", lineage_id="lineage",
            config_identity="config", overlay_identity="overlay", target_iterations=100000,
        ).validate(expected={"predecessor_content_id": "content"})
