"""Worker-side E5b runner used by the real Balsam ApplicationDefinition.

The first Balsam allocation runs a bounded prefix, writes durable checkpoint
evidence, and remains alive until PBS walltime kills it.  A later start of the
same Balsam Job observes the evidence marker and routes recovery through E4
and E3.  No worker operation contacts the Balsam API.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from tpen.checkpoint import CheckpointRef
from tpen.execution import (
    BalsamAdapter,
    ReadyBatch,
    RecoveryPhase,
    ScienceIntent,
    WorkerStartGateway,
    decode_ready_batch,
    validate_native_recovery_profile,
)
from tpen.execution.invocation import (
    _committed_checkpoint,
    _core_outcome_from_artifacts,
    _materialize_configured_worker_config,
    _run_status_artifact,
    run_configured_once,
)
from tpen.execution.recovery import CheckpointEvidence
from tools.e5b_records import write_replace_json


class NoLiveBackendClient:
    """Client-shaped proof that a Balsam worker needs no live API."""

    def register_job(self, **_kwargs: Any) -> None:  # pragma: no cover - safety tripwire
        raise AssertionError("worker attempted Balsam registration")

    def observe_job(self, job_id: str | int) -> None:  # pragma: no cover - safety tripwire
        del job_id
        raise AssertionError("worker attempted Balsam observation")


class PrefixCheckpointReached(RuntimeError):
    """Stop bounded prefix science after its full checkpoint is published."""


class StopAfterCheckpoint:
    """Operational stop observer, supplied outside all hashed config sections."""

    def __init__(self, next_iteration: int) -> None:
        self.next_iteration = next_iteration

    def handle_occurrence(self, occurrence: Any, context: Any) -> None:
        from tpen.training.events import TrainingIterationCompleted

        if isinstance(occurrence.event, TrainingIterationCompleted):
            # Runtime observers run after the configured checkpoint callback,
            # so admission waits for the same published ref E3 resolves.
            checkpoint = _committed_checkpoint(context.artifact_manager.run_dir)
            if checkpoint is not None and checkpoint.next_iteration >= self.next_iteration:
                raise PrefixCheckpointReached("bounded prefix checkpoint is published")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write one durable JSON marker before the next allocation can start."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_replace_json(path, data)


def _load_inputs(job_data: Mapping[str, Any]) -> tuple[ReadyBatch, ScienceIntent, Any, Path]:
    """Load only serialized batch/intent/config data on the worker."""

    root = Path(str(job_data["pilot_root"]))
    batch = decode_ready_batch((root / "batch.json").read_bytes())
    intent_data = json.loads((root / "intent.json").read_text(encoding="utf-8"))
    # The serialized digest is derived evidence, not a ScienceIntent
    # constructor field; reconstruct the immutable intent from its owners.
    intent = ScienceIntent(
        science_input=intent_data["science_input"],
        target_iterations=intent_data["target_iterations"],
        source_config=intent_data.get("source_config", {}),
        overrides=intent_data.get("overrides", ()),
    )
    config = OmegaConf.load(str(job_data["config_path"]))
    validate_native_recovery_profile(config)
    return batch, intent, config, root


def _run_once_factory(
    config: Any,
    *,
    config_path: str,
    phase: RecoveryPhase,
    prefix_steps: int | None = None,
):
    """Return a configured-run callback for one E3 invocation."""

    def run_once(decision: Any, output_path: Path, *, intent: ScienceIntent) -> Any:
        # Use the same E3 materialization as start_configured before the
        # observable core-call seam; source binding must precede construction.
        worker_config = _materialize_configured_worker_config(
            config, intent=intent, decision=decision, output_path=output_path,
            declared_overlays=None,
        )
        observers = ()
        if phase is RecoveryPhase.INITIAL and prefix_steps is not None:
            observers = (StopAfterCheckpoint(prefix_steps),)
        run_configured_once(
            worker_config,
            decision=decision,
            output_path=output_path,
            config_path=config_path,
            command="e5b-polaris-worker",
            raise_exceptions=True,
            runtime_callbacks=observers,
        )
        return _core_outcome_from_artifacts(output_path, decision=decision)

    return run_once


def _record_application(root: Path, name: str, data: Mapping[str, Any]) -> None:
    """Publish a per-allocation application receipt."""

    _write_json(root / name, {"schema": "tpen.execution/e5b-application-v1", **dict(data)})


def run_attempt(*, job_id: int | str, job_data: Mapping[str, Any]) -> int:
    """Execute initial-prefix or successor-recovery work for one Balsam start."""

    batch, intent, config, root = _load_inputs(job_data)
    evidence_path = root / "checkpoint-evidence.json"
    receipts = root / "receipts"
    outputs = root / "outputs"
    adapter = BalsamAdapter(
        batch.execution,
        NoLiveBackendClient(),
        WorkerStartGateway(outputs),
        receipts,
    )
    if not evidence_path.is_file():
        if tuple(outputs.rglob("start.json")):
            reason = "a prior worker start exists but no admissible committed checkpoint was recorded"
            _write_json(
                root / "no-checkpoint-park.json",
                {
                    "schema": "tpen.execution/e5b-no-checkpoint-park-v1",
                    "job_id": str(job_id),
                    "reason": reason,
                },
            )
            _record_application(
                root,
                "prefix-application.json",
                {"status": "parked", "job_id": str(job_id), "reason": reason},
            )
            return 2
        prefix_steps = int(job_data.get("prefix_steps", max(1, batch.target_iterations // 2)))
        callback = _run_once_factory(
            config, config_path=str(job_data["config_path"]),
            phase=RecoveryPhase.INITIAL, prefix_steps=prefix_steps,
        )
        result = adapter.start(
            batch=batch,
            intent=intent,
            phase=RecoveryPhase.INITIAL,
            run_once=lambda decision, output_path: callback(decision, output_path, intent=intent),
            job_id=job_id,
            attempt_id=f"polaris-prefix-{os.environ.get('PBS_JOBID', 'unknown')}",
        )
        if result.invocation is None:
            _record_application(root, "prefix-application.json", {"status": result.status.value, "reason": result.reason})
            return 2
        run_dir = Path(result.invocation.start.output_path)
        status_result = _run_status_artifact(run_dir)
        checkpoint = None if status_result is None else _committed_checkpoint(status_result[0])
        if checkpoint is None or checkpoint.next_iteration >= batch.target_iterations:
            _record_application(root, "prefix-application.json", {"status": "no-prefix-checkpoint", "job_id": str(job_id)})
            return 3
        evidence = CheckpointEvidence.from_ref(
            checkpoint,
            science_input_digest=intent.science_input_digest,
            target_iterations=batch.target_iterations,
        )
        _write_json(root / "checkpoint-evidence.json", evidence.to_dict())
        _write_json(
            root / "prefix-ready.json",
            {
                "schema": "tpen.execution/e5b-prefix-ready-v1",
                "job_id": str(job_id),
                "pbs_job_id": os.environ.get("PBS_JOBID"),
                "checkpoint_dir": str(checkpoint.checkpoint_dir),
                "next_iteration": checkpoint.next_iteration,
                "completed_updates": checkpoint.completed_updates,
                "note": "process intentionally remains alive for PBS walltime allocation loss",
            },
        )
        _record_application(root, "prefix-application.json", {"status": "prefix-ready", "job_id": str(job_id)})
        # The scheduler, not this worker, must deliver the allocation loss.
        while True:
            time.sleep(60)

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    recovery_callback = _run_once_factory(
        config, config_path=str(job_data["config_path"]), phase=RecoveryPhase.RECOVERY,
    )
    successor = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.RECOVERY,
        run_once=lambda decision, output_path: recovery_callback(decision, output_path, intent=intent),
        job_id=job_id,
        evidence=evidence,
        attempt_id=f"polaris-successor-{os.environ.get('PBS_JOBID', 'unknown')}",
    )
    if successor.invocation is None:
        _record_application(root, "successor-application.json", {"status": successor.status.value, "reason": successor.reason})
        return 2

    # The uninterrupted reference is another fresh E3 invocation in the same
    # successor process, kept separate from the recovered science output.
    reference_callback = _run_once_factory(
        config, config_path=str(job_data["config_path"]), phase=RecoveryPhase.INITIAL,
    )
    reference = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda decision, output_path: reference_callback(decision, output_path, intent=intent),
        job_id=job_id,
        attempt_id=f"polaris-reference-{os.environ.get('PBS_JOBID', 'unknown')}",
    )
    _record_application(
        root,
        "successor-application.json",
        {
            "status": successor.status.value,
            "job_id": str(job_id),
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "successor_outcome": successor.invocation.outcome.outcome.value,
            "reference_outcome": None if reference.invocation is None else reference.invocation.outcome.outcome.value,
            "successor_output": successor.invocation.start.output_path,
            "reference_output": None if reference.invocation is None else reference.invocation.start.output_path,
            "exit_code": 0,
        },
    )
    return 0 if successor.invocation.outcome.outcome.value == "completed" else 2


__all__ = ["NoLiveBackendClient", "run_attempt"]
