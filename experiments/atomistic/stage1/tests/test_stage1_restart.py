"""Contracts for the configured Stage-1 train-resume gateway."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import restart
from experiments.atomistic.stage1.launch import row_overrides
from experiments.atomistic.stage1.protocol import (
    Protocol,
    build_manifest,
    config_identities,
    overlay_sha256,
)
from experiments.atomistic.stage1.train import build_train_config
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.run import run_from_config


CONFIG_ROOT = Path(__file__).parents[1] / "configs"
OVERLAY = CONFIG_ROOT / "overlays" / "diagnostics-v1.yaml"
REPO_ROOT = Path(__file__).resolve().parents[4]


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
    run_dir = Path(tmp_path) / "stage1_he" / "atomistic" / "run-0"
    checkpoint = run_dir / "checkpoints" / "step_000000"
    checkpoint.mkdir(parents=True)
    refusal = SimpleNamespace(
        verdict="refused",
        refusal={"code": "missing-checkpoint", "reason": "checkpoint is absent"},
        checkpoint=None,
        checkpoint_ref=None,
    )
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    monkeypatch.setattr(restart, "inspect_run", lambda _path: _evidence(run_dir, checkpoint))
    monkeypatch.setattr(restart, "run_from_config", lambda *_a, **_k: calls.append(True) or 0)

    with pytest.raises(restart.RestartRefusal, match="checkpoint is absent"):
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: refusal,
        )
    assert calls == []


def test_row_update_horizon_mismatch_refuses_before_runner_call(
    tmp_path: Path, monkeypatch,
) -> None:
    """A row/config H disagreement is rejected before inspection or launch."""

    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    row["updates"] = 99_999
    calls: list[object] = []
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    monkeypatch.setattr(restart, "run_from_config", lambda *_a, **_k: calls.append(True) or 0)

    with pytest.raises(restart.RestartRefusal, match="horizon") as error:
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: pytest.fail("inspection must not run"),
        )
    assert error.value.cause is restart.RestartCause.HORIZON_MISMATCH
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


def test_full_horizon_checkpoint_is_incompatible_resume(tmp_path: Path, monkeypatch) -> None:
    """R2 keeps a generic incompatible-input refusal, not terminal lineage classification."""

    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    run_dir = Path(tmp_path) / "stage1_he" / "atomistic" / "run-0"
    checkpoint = run_dir / "checkpoints" / "step_100000"
    checkpoint.mkdir(parents=True)
    evidence = _evidence(run_dir, checkpoint, step=100000)
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    monkeypatch.setattr(restart, "run_from_config", lambda *_a, **_k: pytest.fail("full horizon launched"))

    with pytest.raises(restart.RestartRefusal, match="full Stage-1 horizon") as error:
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: evidence,
        )
    assert error.value.cause is restart.RestartCause.INVALID_RESUME


def _real_restart_row(tmp_path: Path, *, run_id: str) -> dict[str, object]:
    """Build a two-update He row retaining the committed diagnostics overlay."""

    protocol = Protocol(
        smoke_updates=1,
        smoke_walkers=2,
        lineage_updates=2,
        lineage_walkers=2,
        lineages=3,
        eval_streams=4,
        eval_walkers=2,
        eval_draws=1,
        retention_cadence=1,
    )
    rows = build_manifest(
        protocol=protocol,
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=CONFIG_ROOT, systems=("he",)),
        checkpoints=[0, 1, 2],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path),
        systems=("he",),
        phase=1,
        device="cpu",
        overlay=OVERLAY,
        overlay_sha256=overlay_sha256(OVERLAY),
    )
    row = next(item for item in rows if item["kind"] == "train")
    row["run_id"] = run_id
    return row


def _real_restart_child() -> None:
    """Run one fresh, resumed, or uninterrupted phase in a new interpreter."""

    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    row = json.loads(Path(payload["row_path"]).read_text(encoding="utf-8"))
    config_path = Path(str(payload["config_path"]))
    cfg = build_train_config(
        row,
        config_path=config_path,
        overrides=row_overrides(row),
        overlay=str(OVERLAY),
        overlay_sha256=overlay_sha256(OVERLAY),
    )
    phase = str(payload["phase"])
    if phase == "fresh":
        from tpen.callback.planned_yield import PlannedYield

        code = run_from_config(
            cfg,
            config_path=str(config_path),
            command="stage1 real restart fresh",
            runtime_callbacks=(PlannedYield(1, int(row["updates"])),),
            raise_exceptions=True,
        )
        result: dict[str, object] = {"outcome": "planned_yield", "exit_code": code}
    elif phase == "resumed":
        resumed = restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path=config_path,
            command="stage1 real restart resumed",
        )
        result = {"outcome": resumed.outcome, "exit_code": resumed.exit_code}
    elif phase == "reference":
        code = run_from_config(
            cfg,
            config_path=str(config_path),
            command="stage1 real restart reference",
            raise_exceptions=True,
        )
        result = {"outcome": "completed", "exit_code": code}
    else:
        raise ValueError(f"unknown phase: {phase}")
    print(json.dumps(result, sort_keys=True))


def _run_real_restart_child(payload: dict[str, object]) -> dict[str, object]:
    """Launch one real phase and return its final JSON result."""

    row_path = Path(str(payload["row_path"]))
    phase = str(payload["phase"])
    argument_path = row_path.with_name(f"{row_path.stem}.{phase}.args.json")
    stdout_path = row_path.with_name(f"{row_path.stem}.{phase}.stdout")
    stderr_path = row_path.with_name(f"{row_path.stem}.{phase}.stderr")
    argument_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from experiments.atomistic.stage1.tests.test_stage1_restart "
                "import _real_restart_child; _real_restart_child()"
            ),
            str(argument_path),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    assert completed.returncode == 0, (
        f"{phase} child exited {completed.returncode}; "
        f"stdout={stdout_path}\n{completed.stdout}\n"
        f"stderr={stderr_path}\n{completed.stderr}"
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.integration
def test_real_fresh_process_restore_parity_with_checkpoint_zero_and_overlay(
    tmp_path: Path,
) -> None:
    """A fresh-process configured restore matches an uninterrupted two-step run.

    This deliberately exercises the real TPEN He model on CPU for only two
    updates; the bounded integration cost is justified by the contract's
    checkpoint-zero, overlay, and process-boundary parity requirement.
    """

    config_path = CONFIG_ROOT / "he" / "train.yaml"
    resumed_row = _real_restart_row(tmp_path / "resumed", run_id="train-lineage-0")
    resumed_row_path = tmp_path / "resumed-row.json"
    resumed_row_path.parent.mkdir(parents=True, exist_ok=True)
    resumed_row_path.write_text(json.dumps(resumed_row, sort_keys=True) + "\n", encoding="utf-8")
    fresh_payload = {
        "phase": "fresh", "row_path": str(resumed_row_path), "config_path": str(config_path),
    }
    fresh_result = _run_real_restart_child(fresh_payload)
    assert fresh_result == {"outcome": "planned_yield", "exit_code": 2}

    resumed_result = _run_real_restart_child({**fresh_payload, "phase": "resumed"})
    if resumed_result != {"outcome": "completed", "exit_code": 0}:
        resumed_stderr = resumed_row_path.with_name("resumed-row.resumed.stderr")
        resumed_stdout = resumed_row_path.with_name("resumed-row.resumed.stdout")
        pytest.fail(
            "resumed child did not complete:\n"
            f"result={resumed_result!r}\n"
            f"stdout_path={resumed_stdout}\n{resumed_stdout.read_text(encoding='utf-8')}\n"
            f"stderr_path={resumed_stderr}\n{resumed_stderr.read_text(encoding='utf-8')}"
        )

    reference_row = _real_restart_row(tmp_path / "reference", run_id="train-lineage-reference")
    reference_row_path = tmp_path / "reference-row.json"
    reference_row_path.write_text(json.dumps(reference_row, sort_keys=True) + "\n", encoding="utf-8")
    reference_result = _run_real_restart_child({
        "phase": "reference", "row_path": str(reference_row_path), "config_path": str(config_path),
    })
    assert reference_result == {"outcome": "completed", "exit_code": 0}

    resumed_dir = resolve_run_dir(resumed_row)
    reference_dir = resolve_run_dir(reference_row)
    for run_dir in (resumed_dir, reference_dir):
        assert (run_dir / "checkpoints" / "step_000000" / "COMPLETE").is_file()
        resolved = OmegaConf.load(run_dir / "resolved_config.yaml")
        assert resolved.stage1_overlay.path == "configs/overlays/diagnostics-v1.yaml"
        assert resolved.stage1_overlay.sha256 == overlay_sha256(OVERLAY)
        assert any(
            callback.get("_target_") == "tpen.callback.DataIntegrity"
            for callback in resolved.callbacks
        )

    resumed_checkpoint = resumed_dir / "checkpoints" / "step_000002"
    reference_checkpoint = reference_dir / "checkpoints" / "step_000002"
    for name in ("model.pt", "optimizer.pt", "trainer.json", "sampler.pt", "rng.pt"):
        assert (resumed_checkpoint / name).read_bytes() == (reference_checkpoint / name).read_bytes()
