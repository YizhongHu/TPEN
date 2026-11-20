"""Unit contracts for the Stage-1 restartable-start adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1.restart import (
    RestartCause,
    RestartApproval,
    RestartRefusal,
    StartDecision,
    configure_attempt_local_diagnostics,
    select_restart,
)
from experiments.toolkit.run_paths import resolve_run_dir


def _cfg(tmp_path: Path | None = None) -> object:
    root = str(tmp_path or Path.cwd())
    return OmegaConf.create(
        {
            "experiment": {"name": "stage1_he", "sector": "atomistic"},
            "run": {"root": root, "run_id": "run-0", "layout": "nested"},
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
    selected = select_restart(cfg=_cfg(tmp_path), row=_row(tmp_path))
    assert selected.decision is StartDecision.FRESH
    assert selected.content_id == "initial"
    assert not resolve_run_dir(_row(tmp_path)).exists()


def test_config_run_dir_mismatch_refuses_before_lease_or_driver(
    tmp_path: Path, monkeypatch,
) -> None:
    from experiments.atomistic.stage1 import restart

    row = _row(tmp_path)
    cfg = _cfg(tmp_path)
    cfg.run.root = str(tmp_path / "wrong-root")
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    monkeypatch.setattr(restart, "validate_native_recovery_profile", lambda _cfg: None)
    with pytest.raises(RestartRefusal, match="config-derived run directory differs") as refused:
        restart.execute_restart(
            cfg=cfg, row=row, config_path="train.yaml",
            run=lambda *_a, **_k: pytest.fail("driver"),
        )
    assert refused.value.cause is RestartCause.RUN_DIRECTORY_MISMATCH
    assert not (resolve_run_dir(row) / "attempts").exists()


def test_empty_post_run_publication_has_typed_nonempty_park_cause(
    tmp_path: Path, monkeypatch,
) -> None:
    from experiments.atomistic.stage1 import restart

    row = _row(tmp_path)
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    monkeypatch.setattr(restart, "validate_native_recovery_profile", lambda _cfg: None)
    monkeypatch.setattr(restart, "discover_published_checkpoint", lambda _path: None)
    result = restart.execute_restart(
        cfg=cfg, row=row, config_path="train.yaml", run=lambda *_a, **_k: 2,
    )
    assert result.outcome == "parked"
    assert result.receipt["cause"] == RestartCause.NO_PUBLICATION.value
    assert result.receipt["reason"]


def test_attempt_diagnostics_are_never_shared_with_previous_attempt(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
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


def test_execute_calls_profile_gate_before_driver_and_uses_durable_outcome(
    tmp_path: Path, monkeypatch,
) -> None:
    from experiments.atomistic.stage1 import restart

    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    run_dir = resolve_run_dir(row)
    calls: list[object] = []
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        restart, "validate_native_recovery_profile", lambda _cfg: calls.append("profile")
    )
    monkeypatch.setattr(
        restart,
        "select_restart",
        lambda **_kwargs: restart.RestartSelection(
            StartDecision.FRESH, run_dir, None, None, None
        ),
    )

    def fake_run(run_cfg, **_kwargs):
        calls.append("driver")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "status.json").write_text('{"status":"completed"}', encoding="utf-8")
        return 99

    terminal = SimpleNamespace(next_iteration=100000, content_id="terminal-content")
    discoveries = iter([terminal])
    monkeypatch.setattr(restart, "discover_published_checkpoint", lambda _path: next(discoveries))
    result = restart.execute_restart(
        cfg=cfg, row=row, config_path="train.yaml", run=fake_run,
        run_dir=run_dir, profile_validator=restart.validate_native_recovery_profile,
    )
    assert calls == ["profile", "driver"]
    assert result.outcome == "completed"
    assert result.exit_code == 99
    assert result.receipt["published_content_id"] == "terminal-content"


def test_occupied_lease_refuses_without_overwrite_or_driver(tmp_path: Path, monkeypatch) -> None:
    from experiments.atomistic.stage1 import restart

    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    run_dir = resolve_run_dir(row)
    lease_dir = run_dir / "attempts"
    lease_dir.mkdir(parents=True)
    lease = lease_dir / "initial.json"
    lease.write_text('{"owner":"first"}\n', encoding="utf-8")
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(restart, "validate_native_recovery_profile", lambda _cfg: None)
    monkeypatch.setattr(
        restart,
        "select_restart",
        lambda **_kwargs: restart.RestartSelection(
            StartDecision.FRESH, run_dir, None, None, None
        ),
    )
    with pytest.raises(RestartRefusal, match="occupied"):
        restart.execute_restart(
            cfg=cfg, row=row, config_path="train.yaml", run=lambda *_a, **_k: pytest.fail("driver"),
            run_dir=run_dir,
        )
    assert lease.read_text(encoding="utf-8") == '{"owner":"first"}\n'


def test_restore_passes_typed_request_and_drops_initial_checkpoint(tmp_path: Path, monkeypatch) -> None:
    from experiments.atomistic.stage1 import restart

    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    run_dir = resolve_run_dir(row)
    request = object()
    predecessor = SimpleNamespace(content_id="published-content")
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(restart, "validate_native_recovery_profile", lambda _cfg: None)
    monkeypatch.setattr(
        restart,
        "select_restart",
        lambda **_kwargs: restart.RestartSelection(
            StartDecision.RESTORE, run_dir, predecessor, None, request
        ),
    )
    observed: dict[str, object] = {}

    def fake_run(run_cfg, **kwargs):
        observed["request"] = kwargs["restore_request"]
        observed["initial_callbacks"] = [
            item.get("_target_") for item in run_cfg.callbacks
        ]
        return 0

    monkeypatch.setattr(restart, "discover_published_checkpoint", lambda _path: None)
    restart.execute_restart(
        cfg=cfg, row=row, config_path="train.yaml", run=fake_run,
        run_dir=run_dir,
    )
    assert observed == {"request": request, "initial_callbacks": [
        "tpen.callback.RawTrainingBatchRecorder", "tpen.callback.ParameterActivity",
    ]}


def test_newer_complete_checkpoint_is_parked_before_start(tmp_path: Path, monkeypatch) -> None:
    from experiments.atomistic.stage1 import restart

    row = _row(tmp_path)
    run_dir = resolve_run_dir(row)
    (run_dir / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(restart, "discover_published_checkpoint", lambda _path: None)
    monkeypatch.setattr(
        restart,
        "_complete_refs",
        lambda _root: [SimpleNamespace(next_iteration=1000, checkpoint_dir=run_dir / "checkpoints" / "step_001000")],
    )
    selected = select_restart(cfg=_cfg(tmp_path), row=row, run_dir=run_dir)
    assert selected.decision is StartDecision.PARK
    assert "complete checkpoint is newer" in selected.reason


def test_terminal_publication_cannot_hide_a_failed_durable_status(
    tmp_path: Path, monkeypatch,
) -> None:
    from experiments.atomistic.stage1 import restart

    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    run_dir = resolve_run_dir(row)
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    monkeypatch.setattr(restart, "validate_native_recovery_profile", lambda _cfg: None)
    monkeypatch.setattr(
        restart,
        "select_restart",
        lambda **_kwargs: restart.RestartSelection(
            StartDecision.FRESH, run_dir, None, None, None
        ),
    )

    def failed_run(run_cfg, **_kwargs):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "status.json").write_text('{"status":"failed"}', encoding="utf-8")
        return 0

    terminal = SimpleNamespace(next_iteration=100000, content_id="terminal-content")
    monkeypatch.setattr(restart, "discover_published_checkpoint", lambda _path: terminal)
    result = restart.execute_restart(
        cfg=cfg, row=row, config_path="train.yaml", run=failed_run, run_dir=run_dir,
    )
    assert result.outcome == "failed"
    assert result.receipt["exit_code"] == 0
