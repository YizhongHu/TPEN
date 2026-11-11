"""Source tests for the E1 checkpoint-recovery contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen.checkpoint import CheckpointRef, restore_checkpoint
from tpen.checkpoint.payload import ModelOnly, TrainResume
from tpen.execution import (
    CheckpointEvidence,
    RecoveryOutcome,
    RecoveryPhase,
    RefusalCode,
    capture_science_intent,
    select_recovery,
    train_resume_load_overlay,
)


def _manifest(
    step: int,
    *,
    completed_updates: int = 3,
    payload: dict[str, object] | None = None,
    files: dict[str, str] | None = None,
    schema_version: int = 2,
    kind: str = "tpen.checkpoint",
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "kind": kind,
        "next_iteration": step,
        "completed_updates": completed_updates,
        "created_at_unix": 1.0,
        "files": files or {"model": "model.pt"},
        "hashes": {},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": "e1", "git_sha": "test"},
        **({} if payload is None else {"payload": payload}),
    }


def _checkpoint(
    root: Path,
    *,
    step: int = 5,
    completed_updates: int = 3,
    payload: dict[str, object] | None = None,
    full_state: bool = True,
    complete: bool = True,
) -> Path:
    path = root / f"step_{step:06d}"
    path.mkdir(parents=True)
    files = {"model": "model.pt"}
    if full_state:
        files.update(
            {
                "optimizer": "optimizer.pt",
                "trainer": "trainer.json",
                "sampler": "sampler.pt",
                "rng": "rng.pt",
            }
        )
    for name in files.values():
        (path / name).write_bytes(b"payload")
    (path / "trainer.json").write_text(
        json.dumps({"next_iteration": step, "completed_updates": completed_updates}),
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            _manifest(
                step,
                completed_updates=completed_updates,
                payload=payload,
                files=files,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if complete:
        (path / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return path


def _intent(*, target_iterations: int = 10, science_input: dict[str, object] | None = None):
    return capture_science_intent(
        science_input or {"system": "hooke", "electrons": 2},
        target_iterations=target_iterations,
        source_config={"model": {"_target_": "tests.helpers.NeverConstructed"}},
        overrides=("trainer.max_steps=10",),
    )


def test_initial_selection_without_checkpoint_starts_from_immutable_intent() -> None:
    intent = _intent()

    decision = select_recovery(intent, phase=RecoveryPhase.INITIAL)

    assert decision.outcome is RecoveryOutcome.INITIAL
    assert decision.load is None
    assert decision.next_iteration == 0
    assert decision.completed_updates == 0
    with pytest.raises(TypeError):
        intent.science_input["system"] = "changed"  # type: ignore[index]


def test_valid_full_state_evidence_emits_exact_existing_train_resume_overlay(tmp_path: Path) -> None:
    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, step=5, completed_updates=3, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    evidence = CheckpointEvidence.from_ref(
        ref,
        science_input=intent.science_input,
        target_iterations=intent.target_iterations,
    )

    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)

    assert decision.outcome is RecoveryOutcome.RECOVERY
    assert decision.load == {"mode": "train_resume", "path": str(checkpoint_dir)}
    assert decision.load == train_resume_load_overlay(ref)
    assert decision.next_iteration == 5
    assert decision.completed_updates == 3
    assert decision.next_iteration != decision.completed_updates


@pytest.mark.parametrize(
    ("payload", "full_state", "expected"),
    [
        (ModelOnly().to_manifest(), False, RefusalCode.MODEL_ONLY),
        (None, False, RefusalCode.INCOMPLETE),
    ],
)
def test_recovery_refuses_model_only_and_incomplete_evidence(
    tmp_path: Path,
    payload: dict[str, object] | None,
    full_state: bool,
    expected: RefusalCode,
) -> None:
    intent = _intent()
    checkpoint_dir = _checkpoint(
        tmp_path,
        payload=payload,
        full_state=full_state,
        complete=expected is not RefusalCode.INCOMPLETE,
    )

    decision = select_recovery(
        intent,
        phase=RecoveryPhase.RECOVERY,
        evidence=CheckpointEvidence.from_path(
            checkpoint_dir,
            science_input=intent.science_input,
            target_iterations=intent.target_iterations,
        ),
    )

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is expected
    assert decision.load is None


def test_recovery_refuses_unsupported_schema(tmp_path: Path) -> None:
    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest["kind"] = "spenn.checkpoint"
    manifest.pop("completed_updates")
    manifest["step"] = 5
    (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    decision = select_recovery(
        intent,
        phase=RecoveryPhase.RECOVERY,
        evidence=CheckpointEvidence(
            ref=ref,
            science_input_digest=intent.science_input_digest,
            target_iterations=intent.target_iterations,
        ),
    )

    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.UNSUPPORTED


def test_recovery_refuses_incompatible_binding_and_does_not_restart(tmp_path: Path) -> None:
    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    evidence = CheckpointEvidence.from_ref(
        ref,
        science_input={"system": "different"},
        target_iterations=intent.target_iterations,
    )

    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPATIBLE
    assert decision.load is None


def test_recovery_acceptance_mutation_pair_same_binding_is_required(tmp_path: Path) -> None:
    """Control half of the pair: removing the binding refusal makes this red."""

    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    evidence = CheckpointEvidence.from_ref(
        ref,
        science_input=intent.science_input,
        target_iterations=intent.target_iterations,
    )

    accepted = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
    refused = select_recovery(
        _intent(science_input={"system": "mutated"}),
        phase=RecoveryPhase.RECOVERY,
        evidence=evidence,
    )

    assert accepted.outcome is RecoveryOutcome.RECOVERY
    assert refused.refusal is not None
    assert refused.refusal.code is RefusalCode.INCOMPATIBLE


def test_no_checkpoint_is_distinct_from_initial_restart() -> None:
    decision = select_recovery(_intent(), phase=RecoveryPhase.RECOVERY)

    assert decision.outcome is RecoveryOutcome.NO_CHECKPOINT
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.NO_CHECKPOINT
    assert decision.load is None


def test_source_capture_keeps_target_opaque_and_never_instantiates_science() -> None:
    intent = capture_science_intent(
        {"model": {"_target_": "tests.helpers.NeverConstructed"}},
        target_iterations=4,
        source_config={"sampler": {"_target_": "tests.helpers.AlsoNeverConstructed"}},
        overrides=("model.width=8",),
    )

    assert intent.source_config["sampler"]["_target_"] == "tests.helpers.AlsoNeverConstructed"
    assert intent.to_dict()["overrides"] == ["model.width=8"]


def test_existing_direct_load_none_path_remains_unchanged() -> None:
    report = restore_checkpoint(
        load={"mode": "none", "path": "this-path-is-not-read"},
        model=object(),
        context=object(),
    )

    assert report.to_dict() == {
        "mode": "none",
        "checkpoint_dir": None,
        "schema_version": None,
        "next_iteration": None,
        "completed_updates": None,
        "loaded_model": False,
        "loaded_optimizer": False,
        "loaded_trainer": False,
        "loaded_sampler": False,
        "loaded_rng": False,
    }
