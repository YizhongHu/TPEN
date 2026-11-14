"""CX-04a core continuation admission gates.

The ``fork`` refusal is executor-owned by CX-04b; this core slice exposes the
stable code but does not detect executor admission-ledger conflicts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from tpen.checkpoint import CheckpointRef, TrainResume, save_checkpoint
from tpen.checkpoint.hashing import resolved_config_identity
from tpen.execution.recovery import (
    AdmissionRefusalCode,
    SUCCESSOR_ADMISSION_SCHEMA,
    ScienceIntent,
    SuccessorAdmissionPredecessor,
    SuccessorAdmissionRequest,
    SuccessorAdmissionResult,
    SuccessorAdmissionSuccessor,
    decode_successor_admission_request,
    select_successor_admission,
)
from tpen.inspection import RUN_EVIDENCE_SCHEMA_V3, RunEvidence


_TARGET = 8


def _checkpoint(tmp_path: Path) -> CheckpointRef:
    """Create one complete published train-resume checkpoint."""

    import torch

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    context = SimpleNamespace(
        cfg=OmegaConf.create({"study": {"name": "cx04"}, "trainer": {"max_steps": _TARGET}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id="cx04", extra={}),
        run_dir=tmp_path,
    )
    trainer = SimpleNamespace(
        state_dict=lambda: {"next_iteration": 3, "completed_updates": 3}
    )
    sampler = SimpleNamespace(mcmc_state_dict=lambda: {"walker": 3})
    directory = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=3,
        completed_updates=3,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
        payload=TrainResume(),
        context=context,
    )
    return CheckpointRef.from_directory(directory)


def _evidence(ref: CheckpointRef) -> RunEvidence:
    """Build v3 planned-yield evidence from the real published ref."""

    identity = resolved_config_identity({"study": "cx04"})
    return RunEvidence(
        schema=RUN_EVIDENCE_SCHEMA_V3,
        run_dir=str(ref.checkpoint_dir.parent.parent),
        verdict="incomplete",
        run_status="planned_yield",
        checkpoint={
            "checkpoint_dir": str(ref.checkpoint_dir),
            "schema_version": ref.schema_version,
            "next_iteration": ref.next_iteration,
            "completed_updates": ref.completed_updates,
        },
        refusal=None,
        resolved_config_identity=identity,
        outcome="planned_yield",
        checkpoint_ref={**ref.to_dict(), "profile": "train_resume"},
    )


def _request(ref: CheckpointRef, *, evidence: RunEvidence | None = None) -> SuccessorAdmissionRequest:
    """Build a request carrying the supplied evidence and matching intent."""

    evidence = _evidence(ref) if evidence is None else evidence
    return SuccessorAdmissionRequest(
        predecessor=SuccessorAdmissionPredecessor(
            task_id="task-1",
            attempt_id="attempt-1",
            core_outcome_digest=hashlib.sha256(b"core-outcome").hexdigest(),
            resolved_config_identity=evidence.resolved_config_identity or {},
        ),
        successor=SuccessorAdmissionSuccessor(task_id="task-1", attempt_id="attempt-2"),
        science_input_digest=ScienceIntent({"input": "cx04"}, target_iterations=_TARGET).science_input_digest,
        target_iterations=_TARGET,
        core_evidence=evidence,
    )


def _intent() -> ScienceIntent:
    return ScienceIntent({"input": "cx04"}, target_iterations=_TARGET)


def test_decode_and_admit_real_published_checkpoint(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    request = _request(ref)
    decoded = decode_successor_admission_request(request.to_dict())

    result = select_successor_admission(_intent(), decoded)

    assert isinstance(result, SuccessorAdmissionResult)
    assert result.accepted
    assert result.restore_request is not None
    assert result.restore_request.ref == ref
    assert result.restore_request.ref.content_id == ref.content_id
    assert result.restore_request.load["mode"] == "train_resume"
    assert result.restore_request.target_iterations == _TARGET
    assert result.restore_request.next_iteration == ref.next_iteration
    assert result.restore_request.completed_updates == ref.completed_updates
    assert result.restore_request.load == {"mode": "train_resume", "path": str(ref.checkpoint_dir)}
    assert decoded.to_dict() == request.to_dict()
    assert decoded.to_dict()["schema"] == SUCCESSOR_ADMISSION_SCHEMA


@pytest.mark.parametrize("field", ["schema", "predecessor", "successor", "core_evidence"])
def test_decoder_rejects_missing_top_level_fields(tmp_path: Path, field: str) -> None:
    data = _request(_checkpoint(tmp_path)).to_dict()
    del data[field]
    with pytest.raises(ValueError):
        decode_successor_admission_request(data)


def test_decoder_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    data = _request(_checkpoint(tmp_path)).to_dict()
    data["unexpected"] = True
    with pytest.raises(ValueError):
        decode_successor_admission_request(data)


@pytest.mark.parametrize("outcome", ["completed", "failed", "unknown"])
def test_non_eligible_outcomes_are_refused(tmp_path: Path, outcome: str) -> None:
    ref = _checkpoint(tmp_path)
    evidence = replace(
        _evidence(ref),
        verdict="completed" if outcome == "completed" else "incomplete",
        run_status=outcome,
        outcome=outcome,
    )
    result = select_successor_admission(_intent(), _request(ref, evidence=evidence))
    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.NON_ELIGIBLE_OUTCOME


def test_false_eligibility_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref = _checkpoint(tmp_path)
    evidence = _evidence(ref)
    monkeypatch.setattr(RunEvidence, "continuation_eligible", property(lambda _evidence: False))
    result = select_successor_admission(_intent(), _request(ref, evidence=evidence))
    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.NON_ELIGIBLE_OUTCOME


def test_wire_model_only_profile_is_refused_by_admission(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    wire = _request(ref).to_dict()
    wire["core_evidence"]["checkpoint_ref"]["profile"] = "model_only"

    decoded = decode_successor_admission_request(json.dumps(wire).encode())
    result = select_successor_admission(_intent(), decoded)

    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.MODEL_ONLY


def test_wire_planned_yield_without_checkpoint_summary_is_ambiguous(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    wire = _request(ref).to_dict()
    wire["core_evidence"]["checkpoint"] = None

    decoded = decode_successor_admission_request(wire)
    result = select_successor_admission(_intent(), decoded)

    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.AMBIGUOUS_YIELD


def test_model_only_profile_is_refused_before_restore(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    evidence = _evidence(ref)
    evidence.checkpoint_ref["profile"] = "model_only"
    result = select_successor_admission(_intent(), _request(ref, evidence=evidence))
    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.MODEL_ONLY


def test_missing_catalog_or_receipt_is_missing_publication(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    (ref.checkpoint_dir.parent / "publications.jsonl").unlink()
    result = select_successor_admission(_intent(), _request(ref))
    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.MISSING_PUBLICATION


def test_missing_receipt_is_missing_publication(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    (ref.checkpoint_dir.parent / "publication_receipts.jsonl").unlink()
    result = select_successor_admission(_intent(), _request(ref))
    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.MISSING_PUBLICATION


def test_latest_pointer_without_catalog_is_not_publication(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    (ref.checkpoint_dir.parent / "publications.jsonl").unlink()
    assert (ref.checkpoint_dir.parent / "latest.json").is_file()
    result = select_successor_admission(_intent(), _request(ref))
    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.MISSING_PUBLICATION


def test_content_id_mutation_is_corrupt_ref(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    evidence = _evidence(ref)
    evidence.checkpoint_ref["content_id"] = "0" * 64
    result = select_successor_admission(_intent(), _request(ref, evidence=evidence))
    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.CORRUPT_REF


@pytest.mark.parametrize("field", ["manifest_sha256", "model_sha256", "provenance"])
def test_ref_manifest_provenance_hash_corruption_is_corrupt_ref(tmp_path: Path, field: str) -> None:
    ref = _checkpoint(tmp_path)
    evidence = _evidence(ref)
    evidence.checkpoint_ref[field] = {"tampered": True} if field == "provenance" else "0" * 64

    result = select_successor_admission(_intent(), _request(ref, evidence=evidence))

    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.CORRUPT_REF


def test_live_model_mutation_is_stale_ref(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    model_path = ref.checkpoint_dir / "model.pt"
    mutated = bytearray(model_path.read_bytes())
    mutated[0] ^= 0xFF
    model_path.write_bytes(mutated)
    result = select_successor_admission(_intent(), _request(ref))
    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.STALE_REF


def test_planned_yield_without_checkpoint_binding_is_ambiguous(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    evidence = _evidence(ref)
    object.__setattr__(evidence, "checkpoint_ref", None)
    result = select_successor_admission(_intent(), _request(ref, evidence=evidence))
    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.AMBIGUOUS_YIELD


def test_identity_mismatch_is_refused(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    request = _request(ref)
    wrong_intent = ScienceIntent({"input": "other"}, target_iterations=_TARGET)
    result = select_successor_admission(wrong_intent, request)
    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.IDENTITY_MISMATCH


def test_config_identity_mismatch_is_refused(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    request = _request(ref)
    predecessor = replace(
        request.predecessor,
        resolved_config_identity={
            "schema": request.predecessor.resolved_config_identity["schema"],
            "sha256": "0" * 64,
        },
    )

    result = select_successor_admission(
        _intent(), replace(request, predecessor=predecessor)
    )

    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.IDENTITY_MISMATCH


def test_predecessor_successor_task_id_mismatch_is_refused(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    request = _request(ref)
    successor = replace(request.successor, task_id="task-2")

    result = select_successor_admission(_intent(), replace(request, successor=successor))

    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.IDENTITY_MISMATCH


def test_checkpoint_counter_overflow_is_identity_mismatch(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    request = replace(_request(ref), target_iterations=2)
    intent = ScienceIntent({"input": "cx04"}, target_iterations=2)

    result = select_successor_admission(intent, request)

    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.IDENTITY_MISMATCH


def test_checkpoint_counter_mismatch_with_summary_is_identity_mismatch(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    evidence = _evidence(ref)
    evidence.checkpoint["completed_updates"] += 1

    result = select_successor_admission(_intent(), _request(ref, evidence=evidence))

    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.IDENTITY_MISMATCH


def test_status_ref_disagreement_is_ambiguous_yield(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    wire = _request(ref).to_dict()
    wire["core_evidence"]["run_status"] = "failed"

    decoded = decode_successor_admission_request(wire)
    result = select_successor_admission(_intent(), decoded)

    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.AMBIGUOUS_YIELD
