"""R9-3: transient model-only manifest I/O must remain reconcilable."""

from __future__ import annotations

from pathlib import Path

import json

from tpen.checkpoint import CheckpointRef
from tpen.checkpoint.payload import TrainResume
from tpen.execution import (
    CheckpointEvidence,
    InvocationOutcome,
    RecoveryOutcome,
    RecoveryPhase,
    RefusalCode,
    classify_core_outcome,
    select_recovery,
)
import tpen.execution.recovery as recovery


def _decision(tmp_path: Path):
    intent = _intent()
    # The builder always writes a valid full-state TrainResume manifest.
    checkpoint = _checkpoint(tmp_path)
    evidence = CheckpointEvidence.from_ref(
        CheckpointRef.from_directory(checkpoint),
        science_input=intent.science_input,
        target_iterations=intent.target_iterations,
    )
    return intent, evidence


def _intent():
    from tpen.execution import capture_science_intent
    return capture_science_intent(
        {"system": "hooke", "electrons": 2}, target_iterations=10,
        source_config={"model": {"_target_": "tests.helpers.NeverConstructed"}},
        overrides=("trainer.max_steps=10",),
    )


def _checkpoint(root: Path) -> Path:
    path = root / "step_000005"
    path.mkdir()
    files = {"model": "model.pt", "optimizer": "optimizer.pt", "trainer": "trainer.json",
             "sampler": "sampler.pt", "rng": "rng.pt"}
    for name in files.values():
        (path / name).write_bytes(b"payload")
    (path / "trainer.json").write_text(json.dumps({"next_iteration": 5, "completed_updates": 3}))
    (path / "manifest.json").write_text(json.dumps({
        "schema_version": 2, "kind": "tpen.checkpoint", "next_iteration": 5,
        "completed_updates": 3, "created_at_unix": 1.0, "files": files, "hashes": {},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": "e1", "git_sha": "test"},
        "payload": TrainResume().to_manifest(),
    }))
    (path / "COMPLETE").write_text("complete\n")
    return path


def test_r9_transient_oserror_on_model_only_manifest_read_parks(tmp_path: Path, monkeypatch) -> None:
    # GREEN: the core outcome classifier parks an incomplete recovery decision.
    intent, evidence = _decision(tmp_path)
    original = recovery.read_manifest

    def injected(path, *, mode):
        if mode == "model_only":
            raise OSError("injected EIO")
        return original(path, mode=mode)

    monkeypatch.setattr(recovery, "read_manifest", injected)
    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
    outcome, _reason = classify_core_outcome(None, decision=decision)
    assert outcome is InvocationOutcome.PARKED


def test_r9_transient_oserror_on_model_only_manifest_read_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    # GREEN after the E1 recovery fix.
    intent, evidence = _decision(tmp_path)
    original = recovery.read_manifest

    def injected(path, *, mode):
        if mode == "model_only":
            raise OSError("injected EIO")
        return original(path, mode=mode)

    monkeypatch.setattr(recovery, "read_manifest", injected)
    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPLETE


def test_r9_valid_checkpoint_selects_train_resume_without_injection(tmp_path: Path) -> None:
    # GREEN control.
    intent, evidence = _decision(tmp_path)
    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
    assert decision.refusal is None
    assert decision.outcome is RecoveryOutcome.RECOVERY
    assert decision.load is not None and decision.load["mode"] == "train_resume"


def test_r9_transient_oserror_on_train_resume_manifest_read_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    # GREEN control.
    intent, evidence = _decision(tmp_path)
    original = recovery.read_manifest

    def injected(path, *, mode):
        if mode == "train_resume":
            raise OSError("injected EIO")
        return original(path, mode=mode)

    monkeypatch.setattr(recovery, "read_manifest", injected)
    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPLETE
