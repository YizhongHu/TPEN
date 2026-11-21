"""Shared real CPU checkpoint fixtures for core-admission tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from tpen.checkpoint import CheckpointRef, TrainResume, save_checkpoint
from tpen.checkpoint.hashing import resolved_config_identity
from tpen.execution.core_admission import CoreAdmissionContextV1
from tpen.execution.recovery import (
    ScienceIntent,
    SuccessorAdmissionPredecessor,
    SuccessorAdmissionRequest,
    SuccessorAdmissionSuccessor,
)
from tpen.inspection import RUN_EVIDENCE_SCHEMA_V3, RunEvidence


TARGET_ITERATIONS = 8


def checkpoint(tmp_path: Path) -> CheckpointRef:
    """Create one complete, published CPU train-resume checkpoint."""

    import torch

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    context = SimpleNamespace(
        cfg=OmegaConf.create({"study": {"name": "cx04"}, "trainer": {"max_steps": TARGET_ITERATIONS}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id="cx04", extra={}),
        run_dir=tmp_path,
    )
    trainer = SimpleNamespace(state_dict=lambda: {"next_iteration": 3, "completed_updates": 3})
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


def evidence(ref: CheckpointRef) -> RunEvidence:
    """Build v3 planned-yield evidence from a real published reference."""

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


_EVIDENCE_FACTORY = evidence


def request(
    ref: CheckpointRef,
    *,
    evidence: RunEvidence | None = None,
    task_id: str = "task-1",
    predecessor_attempt_id: str = "attempt-1",
    successor_attempt_id: str = "attempt-2",
    science_payload: object | None = None,
    target_iterations: int = TARGET_ITERATIONS,
) -> SuccessorAdmissionRequest:
    """Build a request whose identities and science digest match its evidence."""

    evidence_record = _EVIDENCE_FACTORY(ref) if evidence is None else evidence
    science_payload = {"input": "cx04"} if science_payload is None else science_payload
    science_input_digest = ScienceIntent(
        science_payload, target_iterations=target_iterations
    ).science_input_digest
    return SuccessorAdmissionRequest(
        predecessor=SuccessorAdmissionPredecessor(
            task_id=task_id,
            attempt_id=predecessor_attempt_id,
            core_outcome_digest=hashlib.sha256(b"core-outcome").hexdigest(),
            resolved_config_identity=evidence_record.resolved_config_identity or {},
        ),
        successor=SuccessorAdmissionSuccessor(
            task_id=task_id, attempt_id=successor_attempt_id
        ),
        science_input_digest=science_input_digest,
        target_iterations=target_iterations,
        core_evidence=evidence_record,
    )


def write_run_start(run_dir: Path) -> None:
    """Write the resolved identity required by typed v3 inspection."""

    run_dir.joinpath("run_start.json").write_text(
        json.dumps({"resolved_config_identity": resolved_config_identity({"study": "cx04"})})
        + "\n",
        encoding="utf-8",
    )


def write_outcome(run_dir: Path, status: str, ref: CheckpointRef | None = None) -> None:
    """Write an explicit typed outcome artifact for one test run."""

    ref_mapping = None if ref is None else ref.to_dict()
    data = {
        "schema": "tpen.run-outcome/v1",
        "status": status,
        "run_dir": str(run_dir),
        "error": None,
        "checkpoint_ref": ref_mapping,
        "next_iteration": None if ref is None else ref.next_iteration,
        "completed_updates": None if ref is None else ref.completed_updates,
        "target_iterations": TARGET_ITERATIONS,
        "exception_type": None,
        "publication_stage": None,
    }
    run_dir.joinpath("run-outcome.json").write_text(
        json.dumps(data) + "\n", encoding="utf-8"
    )


def planned_yield_run(tmp_path: Path) -> CheckpointRef:
    """Create a run directory with a typed planned yield and published ref."""

    ref = checkpoint(tmp_path)
    write_run_start(tmp_path)
    (tmp_path / "status.json").write_text('{"status":"planned_yield"}\n', encoding="utf-8")
    write_outcome(tmp_path, "planned_yield", ref)
    return ref


def admission_context(
    run_dir: Path | None,
    successor_request: SuccessorAdmissionRequest | None = None,
    *,
    config: dict[str, object] | None = None,
    task_id: str = "task-1",
) -> CoreAdmissionContextV1:
    """Build a context bound to the shared real CPU fixture conventions."""

    identity = resolved_config_identity({"study": "cx04"})
    return CoreAdmissionContextV1(
        task_id=task_id,
        lineage_id="lineage-1",
        core_run_dir=run_dir,
        science_input_digest=(
            successor_request.science_input_digest
            if successor_request is not None
            else identity["sha256"]
        ),
        successor_request=successor_request,
        unresolved_config={"study": "cx04"} if config is None else config,
        profile_name="tpen.native-recovery/v1",
    )


__all__ = [
    "TARGET_ITERATIONS",
    "admission_context",
    "checkpoint",
    "evidence",
    "planned_yield_run",
    "request",
    "write_outcome",
    "write_run_start",
]
