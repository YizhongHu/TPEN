"""Ported R4 controls for E1's initial checkpoint evidence boundary."""

from __future__ import annotations

from pathlib import Path

from tpen.checkpoint import CheckpointRef
from tpen.checkpoint.payload import TrainResume
from tpen.execution import CheckpointEvidence, RecoveryOutcome, RecoveryPhase, RefusalCode, select_recovery
from tests.unit.execution.test_recovery import _checkpoint, _intent


def test_initial_checkpoint_ref_form_is_not_auto_bound(tmp_path: Path) -> None:
    intent = _intent()
    checkpoint = _checkpoint(tmp_path, payload=TrainResume().to_manifest())
    decision = select_recovery(
        intent,
        phase=RecoveryPhase.INITIAL,
        evidence=CheckpointRef.from_directory(checkpoint),
    )
    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPATIBLE


def test_initial_checkpoint_path_form_is_not_auto_bound(tmp_path: Path) -> None:
    intent = _intent()
    checkpoint = _checkpoint(tmp_path, payload=TrainResume().to_manifest())
    decision = select_recovery(intent, phase=RecoveryPhase.INITIAL, evidence=str(checkpoint))
    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPATIBLE


def test_initial_durable_mapping_remains_the_green_bound_form(tmp_path: Path) -> None:
    intent = _intent()
    checkpoint = _checkpoint(tmp_path, payload=TrainResume().to_manifest())
    evidence = CheckpointEvidence.from_ref(
        CheckpointRef.from_directory(checkpoint),
        science_input_digest=intent.science_input_digest,
        target_iterations=intent.target_iterations,
    ).to_dict()
    decision = select_recovery(intent, phase=RecoveryPhase.INITIAL, evidence=evidence)
    assert decision.outcome is RecoveryOutcome.INITIAL_CHECKPOINT
