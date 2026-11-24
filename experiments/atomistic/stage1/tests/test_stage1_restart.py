"""Contracts for the configured Stage-1 train-resume gateway."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import restart


def _cfg(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "experiment": {"name": "stage1_he", "sector": "atomistic"},
            "run": {"root": str(tmp_path), "run_id": "run-0", "layout": "nested"},
            "trainer": {"max_steps": 100000},
            "runtime": {"seed": 11},
            "overlay_marker": "preserve-me",
            "runner": {"_target_": "tpen.runner.Train"},
            "callbacks": [
                {"_target_": "tpen.callback.InitialCheckpoint", "output_dir": "old"},
                {"_target_": "tpen.callback.ParameterActivity", "output_path": "old-activity"},
            ],
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


def _evidence(run_dir: Path, checkpoint_dir: Path, *, step: int = 0) -> object:
    return SimpleNamespace(
        verdict="incomplete",
        refusal=None,
        checkpoint={
            "checkpoint_dir": str(checkpoint_dir),
            "next_iteration": step,
            "completed_updates": step,
        },
        checkpoint_ref=None,
        run_dir=str(run_dir),
    )


def test_missing_resume_refuses_before_runner_call(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    calls: list[object] = []
    refusal = SimpleNamespace(
        verdict="refused",
        refusal={"code": "missing-checkpoint", "reason": "checkpoint is absent"},
        checkpoint=None,
        checkpoint_ref=None,
    )
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    with pytest.raises(restart.RestartRefusal, match="checkpoint is absent"):
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: refusal,
            run=lambda *_a, **_k: calls.append(True) or 0,
        )
    assert calls == []


def test_exact_runner_load_and_overlay_survive_configured_restore(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    run_dir = Path(tmp_path) / "stage1_he" / "atomistic" / "run-0"
    checkpoint = run_dir / "checkpoints" / "step_000000"
    checkpoint.mkdir(parents=True)
    observed: dict[str, object] = {}
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    def fake_run(run_cfg, **kwargs):
        observed["load"] = OmegaConf.to_container(run_cfg.runner.load, resolve=True)
        observed["overlay_marker"] = run_cfg.overlay_marker
        observed["callback_targets"] = [item["_target_"] for item in run_cfg.callbacks]
        observed["kwargs"] = kwargs
        return 0

    result = restart.execute_restart(
        cfg=cfg,
        row=row,
        config_path="train.yaml",
        inspector=lambda path: _evidence(run_dir, checkpoint),
        run=fake_run,
    )

    assert observed["load"] == {"mode": "train_resume", "path": str(checkpoint.resolve())}
    assert observed["overlay_marker"] == "preserve-me"
    assert observed["callback_targets"] == ["tpen.callback.ParameterActivity"]
    assert observed["kwargs"] == {"config_path": "train.yaml", "command": "stage1 restart"}
    assert result.runner_load == observed["load"]
    assert result.exit_code == 0


def test_row_and_config_run_identity_are_preserved(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    cfg.run.run_id = "different-run"
    calls: list[object] = []
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    with pytest.raises(restart.RestartRefusal, match="run directory"):
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: pytest.fail("inspection must not run"),
            run=lambda *_a, **_k: calls.append(True) or 0,
        )
    assert calls == []


def test_requested_resume_never_falls_back_to_fresh(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    refusal = SimpleNamespace(
        verdict="refused",
        refusal={"code": "missing-run-dir", "reason": "run directory not found"},
        checkpoint=None,
        checkpoint_ref=None,
    )
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    with pytest.raises(restart.RestartRefusal) as error:
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: refusal,
            run=lambda *_a, **_k: pytest.fail("fresh fallback"),
        )
    assert error.value.cause is restart.RestartCause.MISSING_RESUME


def test_incompatible_checkpoint_counters_refuse_before_runner(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    run_dir = Path(tmp_path) / "stage1_he" / "atomistic" / "run-0"
    checkpoint = run_dir / "checkpoints" / "step_100001"
    checkpoint.mkdir(parents=True)
    evidence = _evidence(run_dir, checkpoint, step=100001)
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    with pytest.raises(restart.RestartRefusal, match="incompatible"):
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: evidence,
            run=lambda *_a, **_k: pytest.fail("incompatible resume launched"),
        )
