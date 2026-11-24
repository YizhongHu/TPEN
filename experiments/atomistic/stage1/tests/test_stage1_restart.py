"""Contracts for the configured Stage-1 train-resume gateway."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import shutil
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


def test_recorder_configured_refuses_before_runner_call(tmp_path: Path, monkeypatch) -> None:
    """Deferred recorder resume is loud and never reaches inspection or launch."""

    cfg = _cfg(tmp_path)
    cfg.stage1_overlay = {"path": "configs/overlays/diagnostics-v1.yaml"}
    cfg.callbacks.append({
        "_target_": "tpen.callback.RawTrainingBatchRecorder",
        "path": "old-training-history",
    })
    row = _row(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    monkeypatch.setattr(restart, "run_from_config", lambda *_a, **_k: calls.append(True) or 0)

    with pytest.raises(restart.RestartRefusal, match="7eaf9e0c.*diagnostics-v1"):
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: pytest.fail("recorder refusal must precede inspection"),
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


def _assert_checkpoint_payload_equal(left: Path, right: Path) -> None:
    """Compare serialized tensor/state payloads, not archive container bytes."""

    import numpy as np
    import torch

    left_payload = torch.load(left, map_location="cpu", weights_only=False)
    right_payload = torch.load(right, map_location="cpu", weights_only=False)

    def compare(actual: object, expected: object, path: str) -> None:
        if torch.is_tensor(actual) or torch.is_tensor(expected):
            assert torch.is_tensor(actual) and torch.is_tensor(expected), path
            assert torch.equal(actual, expected), path
        elif isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
            assert isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray), path
            assert np.array_equal(actual, expected), path
        elif dataclasses.is_dataclass(actual) or dataclasses.is_dataclass(expected):
            assert dataclasses.is_dataclass(actual) and dataclasses.is_dataclass(expected), path
            assert type(actual) is type(expected), path
            for field in dataclasses.fields(actual):
                compare(
                    getattr(actual, field.name),
                    getattr(expected, field.name),
                    f"{path}.{field.name}",
                )
        elif isinstance(actual, dict) or isinstance(expected, dict):
            assert isinstance(actual, dict) and isinstance(expected, dict), path
            assert actual.keys() == expected.keys(), path
            for key in actual:
                compare(actual[key], expected[key], f"{path}[{key!r}]")
        elif isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
            assert type(actual) is type(expected), path
            assert len(actual) == len(expected), path
            for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                compare(actual_item, expected_item, f"{path}[{index}]")
        else:
            assert actual == expected, path

    compare(left_payload, right_payload, left.name)


def _real_restart_child() -> None:
    """Run one fresh, resumed, or uninterrupted phase in a new interpreter."""

    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    row = json.loads(Path(payload["row_path"]).read_text(encoding="utf-8"))
    recorder_free_row = {**row, "overlay": None, "overlay_sha256": None}
    config_path = Path(str(payload["config_path"]))
    cfg = build_train_config(
        recorder_free_row,
        config_path=config_path,
        overrides=row_overrides(recorder_free_row),
        overlay=str(OVERLAY),
        overlay_sha256=overlay_sha256(OVERLAY),
    )
    # The diagnostics overlay remains bound and visible, but this parity arm
    # deliberately exercises the recorder-free Stage-1 configuration while
    # core item 7eaf9e0c owns recorder resume semantics.
    cfg.callbacks = [
        callback for callback in cfg.callbacks
        if callback.get("_target_") != "tpen.callback.RawTrainingBatchRecorder"
    ]
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
            row=recorder_free_row,
            config_path=config_path,
            command="stage1 real restart resumed",
        )
        result = {"outcome": resumed.outcome, "exit_code": resumed.exit_code}
        Path(str(payload["diagnostics_path"])).write_text(
            json.dumps(
                {
                    "result": result,
                    "run_dir": str(resumed.selection.run_dir),
                    "checkpoint_dir": str(resumed.selection.checkpoint_dir),
                    "runner_load": dict(resumed.runner_load),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
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
    diagnostics_path = row_path.with_name(f"{row_path.stem}.{phase}.diagnostics.json")
    child_payload = {**payload, "diagnostics_path": str(diagnostics_path)}
    argument_path.write_text(json.dumps(child_payload, sort_keys=True) + "\n", encoding="utf-8")
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
    if phase == "resumed" and diagnostics_path.is_file():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        run_dir = Path(str(diagnostics["run_dir"]))
        evidence_root = os.environ.get("VERIFY_EVIDENCE_DIR")
        if evidence_root is None and os.environ.get("RUN_ROOT"):
            evidence_root = str(Path(os.environ["RUN_ROOT"]) / "evidence")
        if evidence_root:
            preserved = Path(evidence_root) / "stage1-restart" / phase
            preserved.mkdir(parents=True, exist_ok=True)
            for name in ("run-outcome.json", "status.json", "resolved_config.yaml", "error.json"):
                source = run_dir / name
                if source.is_file():
                    shutil.copyfile(source, preserved / name)
            shutil.copyfile(diagnostics_path, preserved / "diagnostics.json")
            shutil.copyfile(stdout_path, preserved / "child.stdout")
            shutil.copyfile(stderr_path, preserved / "child.stderr")
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
        resumed_diagnostics = resumed_row_path.with_name("resumed-row.resumed.diagnostics.json")
        diagnostics = json.loads(resumed_diagnostics.read_text(encoding="utf-8"))
        run_dir = Path(str(diagnostics["run_dir"]))
        def read_artifact(name: str) -> str:
            path = run_dir / name
            return path.read_text(encoding="utf-8") if path.is_file() else "<missing>"

        pytest.fail(
            "resumed child did not complete:\n"
            f"result={resumed_result!r}\n"
            f"runner_load={diagnostics['runner_load']!r}\n"
            f"checkpoint_dir={diagnostics['checkpoint_dir']}\n"
            f"run-outcome.json={read_artifact('run-outcome.json')}\n"
            f"status.json={read_artifact('status.json')}\n"
            f"stdout_path={resumed_stdout}\n{resumed_stdout.read_text(encoding='utf-8')}\n"
            f"stderr_path={resumed_stderr}\n"
            f"{chr(10).join(resumed_stderr.read_text(encoding='utf-8').splitlines()[-80:])}"
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
        assert not any(
            callback.get("_target_") == "tpen.callback.RawTrainingBatchRecorder"
            for callback in resolved.callbacks
        )
        assert any(
            callback.get("_target_") == "tpen.callback.DataIntegrity"
            for callback in resolved.callbacks
        )

    resumed_checkpoint = resumed_dir / "checkpoints" / "step_000002"
    reference_checkpoint = reference_dir / "checkpoints" / "step_000002"
    for name in ("model.pt", "optimizer.pt", "sampler.pt", "rng.pt"):
        _assert_checkpoint_payload_equal(resumed_checkpoint / name, reference_checkpoint / name)
    assert json.loads((resumed_checkpoint / "trainer.json").read_text(encoding="utf-8")) == json.loads(
        (reference_checkpoint / "trainer.json").read_text(encoding="utf-8")
    )
