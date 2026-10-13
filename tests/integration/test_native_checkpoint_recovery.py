"""E5a native CPU qualification through the E1-E4 operational path.

This test deliberately launches actual worker subprocesses.  The parent is the
submitter/reconciler: it registers one fake persistent Job, observes an injected
allocation loss, and never supplies a live backend object to a worker.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tpen.checkpoint import CheckpointRef
from tpen.execution import (
    BalsamAdapter,
    BalsamJobObservation,
    BalsamJobState,
    ExecutionConfig,
    ReadyBatch,
    ReadyItem,
    WorkerStartGateway,
    capture_science_intent,
    payload_digest,
    source_config_digest,
)
from tpen.execution.recovery import CheckpointEvidence, RecoveryPhase
from tpen.execution.invocation import _committed_checkpoint
from tests.helpers.trainable_wavefunctions import TrainableHookeSingletAnsatz


FIXTURE = Path(__file__).parent / "artifacts" / "execution" / "native_recovery.yaml"
WORKER = Path(__file__).parents[1] / "helpers" / "native_recovery_worker.py"
# This threshold is fixed before the qualification measurement.  It is large
# enough to exclude serialization noise and small enough to catch a real Adam
# update in the float64 fixture.
PARAMETER_CHANGE_THRESHOLD = 1.0e-10


class QualificationClient:
    """Fake submitter-side client with explicit allocation-loss observation."""

    def __init__(self) -> None:
        self.register_calls = 0
        self.observe_calls = 0
        self.state = BalsamJobState.ACTIVE

    def register_job(self, *, lineage_id, batch, backend_options):
        del lineage_id, batch, backend_options
        self.register_calls += 1
        return {"id": 730013}

    def observe_job(self, job_id):
        self.observe_calls += 1
        return BalsamJobObservation(job_id=job_id, state=self.state)


def _batch() -> tuple[ReadyBatch, object, dict]:
    config = OmegaConf.load(FIXTURE)
    source = OmegaConf.to_container(config, resolve=False)
    science_input = {
        "fixture": "native-recovery-v1",
        "system": source["system"],
        "target_iterations": int(config.trainer.max_steps),
    }
    intent = capture_science_intent(
        science_input,
        target_iterations=int(config.trainer.max_steps),
        source_config=source,
    )
    payload = {
        "fixture": "native-recovery-v1",
        "science_input": science_input,
        "target_iterations": int(config.trainer.max_steps),
    }
    payload_hash = payload_digest(payload)
    execution = ExecutionConfig(
        lineage_id="e5-native-lineage",
        backend="balsam",
        recovery="checkpoint",
        source_config_digest=source_config_digest(source),
        payload_digest=payload_hash,
        backend_options={"site": "fake-cannon"},
    )
    batch = ReadyBatch(
        lineage_id="e5-native-lineage",
        batch_id="e5-native-batch",
        source_config_digest=execution.source_config_digest,
        payload_digest=payload_hash,
        science_input_digest=intent.science_input_digest,
        target_iterations=int(config.trainer.max_steps),
        execution=execution,
        items=(
            ReadyItem(
                invocation_id="e5-native-logical-invocation",
                logical_task_id="e5-native-task",
                fresh_output_id="e5-native-planned-output",
                argv=("tpen-native-worker", "native_recovery.yaml"),
                result_dir="native-recovery",
                payload={"seed": int(config.runtime.seed)},
            ),
        ),
        payload=payload,
    )
    return batch, intent, source


def _write_inputs(root: Path, batch: ReadyBatch, intent) -> tuple[Path, Path]:
    batch_path = root / "batch.json"
    intent_path = root / "intent.json"
    batch_path.write_bytes(batch.encode())
    intent_path.write_text(json.dumps(intent.to_dict(), sort_keys=True), encoding="utf-8")
    return batch_path, intent_path


def _launch(
    *,
    root: Path,
    batch_path: Path,
    intent_path: Path,
    config_path: Path,
    receipts: Path,
    outputs: Path,
    phase: str,
    label: str | None = None,
    evidence: Path | None = None,
    partial_restore: bool = False,
) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        str(WORKER),
        "--batch",
        str(batch_path),
        "--config",
        str(config_path),
        "--intent",
        str(intent_path),
        "--receipts",
        str(receipts),
        "--outputs",
        str(outputs),
        "--job-id",
        "730013",
        "--phase",
        phase,
        "--label",
        label or phase,
    ]
    if evidence is not None:
        command.extend(("--evidence", str(evidence)))
    if partial_restore:
        command.append("--partial-restore")
    log = (root / f"worker-{label or phase}-{time.monotonic_ns()}.log").open("w+")
    environment = os.environ.copy()
    # Executing a helper by absolute path makes Python place tests/helpers,
    # rather than the checkout root, first on sys.path.  Keep the fixture's
    # trainable test model importable without changing the installed package.
    checkout_root = str(Path(__file__).parents[2])
    environment["PYTHONPATH"] = checkout_root + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, text=True, env=environment
    )


def _wait_for_checkpoint(
    outputs: Path, *, target: int, process: subprocess.Popen[str] | None = None
) -> Path:
    """Wait for one committed prefix checkpoint, returning its run directory."""

    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        candidates = sorted(outputs.rglob("checkpoints/step_*/COMPLETE"))
        prefixes = [path.parent for path in candidates if int(path.parent.name.split("_")[-1]) < target]
        if prefixes:
            return prefixes[-1].parent.parent
        if process is not None and process.poll() is not None:
            logs = sorted(outputs.parent.glob("worker-*.log"))
            detail = "\n".join(path.read_text() for path in logs)
            raise AssertionError(f"worker exited before a prefix checkpoint (rc={process.returncode})\n{detail}")
        time.sleep(0.1)
    raise AssertionError("worker wrote no committed periodic checkpoint before the target")


def _result_from_log(root: Path, label: str) -> dict:
    logs = sorted(root.glob(f"worker-{label}-*.log"))
    assert logs
    lines = [line for line in logs[-1].read_text().splitlines() if line.strip()]
    return json.loads(lines[-1])


def _run_dir(output_path: Path) -> Path:
    statuses = list(output_path.rglob("status.json"))
    assert len(statuses) == 1, statuses
    return statuses[0].parent


def _final_checkpoint(output_path: Path) -> Path:
    run_dir = _run_dir(output_path)
    checkpoints = sorted(run_dir.glob("checkpoints/step_*/COMPLETE"))
    assert checkpoints
    return checkpoints[-1].parent


def _train_lines(output_path: Path) -> list[tuple[int, str]]:
    lines = []
    for line in (_run_dir(output_path) / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("namespace") == "train":
            lines.append((int(record["step"]), line))
    return lines


def _kill_prefix(process: subprocess.Popen[str], outputs: Path, target: int) -> CheckpointRef:
    """Kill after an observed prefix, then select the latest committed ref."""

    run_dir = _wait_for_checkpoint(outputs, target=target, process=process)
    # Stop the worker before issuing SIGKILL so a fast CPU run cannot commit the
    # target between the observation and the simulated allocation loss.
    os.kill(process.pid, signal.SIGSTOP)
    os.kill(process.pid, signal.SIGKILL)
    process.wait(timeout=30)
    checkpoint = _committed_checkpoint(run_dir)
    assert checkpoint is not None
    assert checkpoint.next_iteration < target
    return checkpoint


def test_kill_before_checkpoint_parks_without_restart(tmp_path: Path) -> None:
    """A lost prefix with no committed evidence never silently restarts."""

    batch, intent, _ = _batch()
    called: list[bool] = []
    result = WorkerStartGateway(tmp_path).start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.RECOVERY,
        run_once=lambda *_: called.append(True),
    )
    assert called == []
    assert result.outcome.outcome.value == "parked"
    assert "no committed checkpoint" in result.outcome.reason


@pytest.mark.integration
def test_native_e1_to_e4_recovery_parity_divergence_and_receipt(tmp_path: Path) -> None:
    """Qualify recovery and a single-variable partial-restore control."""

    batch, intent, _source = _batch()
    initial_state = {
        name: value.detach().clone()
        for name, value in TrainableHookeSingletAnsatz().state_dict().items()
    }
    batch_path, intent_path = _write_inputs(tmp_path, batch, intent)
    receipts = tmp_path / "receipts"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    client = QualificationClient()
    adapter = BalsamAdapter(
        batch.execution, client, WorkerStartGateway(outputs), receipts
    )
    registration = adapter.register_lineage(batch)
    assert registration is not None and registration.job_id == "730013"
    assert client.register_calls == 1

    prefix = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="initial",
        label="prefix",
    )
    prefix_checkpoint = _kill_prefix(prefix, outputs, batch.target_iterations)
    assert prefix.returncode == -signal.SIGKILL
    assert prefix_checkpoint.checkpoint_dir.exists()

    client.state = BalsamJobState.ALLOCATION_LOST
    observation = adapter.observe_lineage(batch)
    assert observation is not None and observation.state is BalsamJobState.ALLOCATION_LOST
    assert client.observe_calls == 1

    evidence = CheckpointEvidence.from_ref(
        prefix_checkpoint,
        science_input_digest=intent.science_input_digest,
        target_iterations=batch.target_iterations,
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence.to_dict(), sort_keys=True), encoding="utf-8")
    successor = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="recovery",
        evidence=evidence_path,
    )
    successor.wait(timeout=180)
    assert successor.returncode == 0
    successor_result = _result_from_log(tmp_path, "recovery")
    assert successor_result["status"] == "started"
    assert successor_result["invocation"]["outcome"] == "completed"
    assert successor_result["invocation"]["next_iteration"] == batch.target_iterations
    assert successor_result["invocation"]["completed_updates"] <= batch.target_iterations

    # An uninterrupted worker uses the same E4 registration and configured
    # route.  It is the scientific reference, not a scheduler-success proxy.
    reference = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="initial",
        label="reference",
    )
    reference.wait(timeout=180)
    assert reference.returncode == 0
    reference_result = _result_from_log(tmp_path, "reference")
    assert reference_result["invocation"]["outcome"] == "completed"

    resumed_output = Path(successor_result["invocation"]["output_path"])
    reference_output = Path(reference_result["invocation"]["output_path"])
    assert resumed_output != reference_output
    resumed_lines = _train_lines(resumed_output)
    reference_lines = _train_lines(reference_output)
    resume_step = prefix_checkpoint.next_iteration
    selected_trainer = json.loads(
        prefix_checkpoint.checkpoint_dir.joinpath("trainer.json").read_text()
    )
    assert selected_trainer["next_iteration"] == resume_step
    parity_steps = [step for step, _ in resumed_lines] == list(range(resume_step, batch.target_iterations))
    expected_tail = [line for step, line in reference_lines if step >= resume_step]
    parity_metrics = [line for _, line in resumed_lines] == expected_tail

    resumed_state = torch.load(_final_checkpoint(resumed_output) / "model.pt", map_location="cpu", weights_only=False)
    reference_state = torch.load(_final_checkpoint(reference_output) / "model.pt", map_location="cpu", weights_only=False)
    assert resumed_state.keys() == reference_state.keys()
    restored_parameters_bitwise = all(
        torch.equal(resumed_state[name], reference_state[name]) for name in resumed_state
    )
    max_restore_parameter_abs_diff = max(
        float(torch.max(torch.abs(resumed_state[name] - reference_state[name])).item())
        for name in resumed_state
    )
    max_parameter_delta = max(
        float(torch.max(torch.abs(reference_state[name] - initial_state[name])).item())
        for name in reference_state
    )
    parameters_changed = max_parameter_delta > PARAMETER_CHANGE_THRESHOLD

    # The negative control keeps the same fixture seed, prefix, recovery phase,
    # selected cursor, and Job.  Only the restored state differs: this
    # test-only overlay uses model_only to restore parameters, retains the
    # cursor, and leaves optimizer/sampler/process-RNG state fresh.  It
    # intentionally bypasses Train's production model_only refusal.
    partial = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="recovery",
        label="partial",
        evidence=evidence_path,
        partial_restore=True,
    )
    partial.wait(timeout=180)
    assert partial.returncode == 0
    partial_result = _result_from_log(tmp_path, "partial")
    assert partial_result["invocation"]["outcome"] == "completed"
    partial_output = Path(partial_result["invocation"]["output_path"])
    assert partial_output not in {resumed_output, reference_output}
    partial_lines = _train_lines(partial_output)
    partial_steps = [step for step, _ in partial_lines] == list(range(resume_step, batch.target_iterations))
    partial_tail = [line for _, line in partial_lines]
    restore_disabled_diverged = bool(partial_steps and partial_tail != expected_tail)
    partial_state = torch.load(_final_checkpoint(partial_output) / "model.pt", map_location="cpu", weights_only=False)
    max_partial_parameter_abs_diff = max(
        float(torch.max(torch.abs(partial_state[name] - reference_state[name])).item())
        for name in reference_state
    )
    tail_bitwise = bool(parity_steps and parity_metrics and restored_parameters_bitwise)
    print(f"TAIL_BITWISE={tail_bitwise}")
    print(f"RESTORE_DISABLED_DIVERGENCE={restore_disabled_diverged}")
    print(f"MAX_PARAMETER_DELTA={max_parameter_delta:.17g}")
    print(f"MAX_RESTORE_PARAMETER_ABS_DIFF={max_restore_parameter_abs_diff:.17g}")
    print(f"MAX_PARTIAL_PARAMETER_ABS_DIFF={max_partial_parameter_abs_diff:.17g}")
    print(f"PARAMETERS_CHANGED={parameters_changed}")
    assert parity_steps, "restored successor did not start at the selected latest checkpoint cursor"
    assert parity_metrics, "restored successor tail metrics differ from uninterrupted reference"
    assert restored_parameters_bitwise, "restored successor final parameters differ from reference"
    assert partial_steps, "partial restore control did not start at the selected recovery cursor"
    assert restore_disabled_diverged, "partial restore control did not diverge from reference tail"
    assert parameters_changed, (
        "reference final trainable parameters did not change from initial state "
        f"by threshold {PARAMETER_CHANGE_THRESHOLD}"
    )

    receipt = {
        "schema": "tpen.execution/qualification-native-recovery-v1",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scheduler": {"kind": "slurm", "partition": os.environ.get("SLURM_JOB_PARTITION", "unknown")},
        "backend": {"kind": "fake-balsam-client", "job_id": registration.job_id, "allocation_state": observation.state.value},
        "wrapper": {"registration_calls": client.register_calls, "observation_calls": client.observe_calls, "prefix_signal": -signal.SIGKILL},
        "application": {"prefix_exit": prefix.returncode, "successor": successor.returncode, "reference": reference.returncode, "partial": partial.returncode},
        "scientific": {
            "successor_outcome": successor_result["invocation"]["outcome"],
            "target_iterations": batch.target_iterations,
            "resume_checkpoint_next_iteration": resume_step,
            "next_iteration": successor_result["invocation"]["next_iteration"],
            "completed_updates": successor_result["invocation"]["completed_updates"],
            "tail_bitwise": tail_bitwise,
            "restore_disabled_mode": "test-only model_only overlay; E1 model_only refusal bypassed",
            "restore_disabled_diverged": restore_disabled_diverged,
            "parameters_changed": parameters_changed,
            "parameter_change_threshold": PARAMETER_CHANGE_THRESHOLD,
            "max_parameter_delta": max_parameter_delta,
            "max_restore_parameter_abs_diff": max_restore_parameter_abs_diff,
            "max_partial_parameter_abs_diff": max_partial_parameter_abs_diff,
        },
    }
    receipt_path = tmp_path / "qualification-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    measured = json.loads(receipt_path.read_text())["scientific"]
    assert measured["tail_bitwise"] is tail_bitwise
    assert measured["restore_disabled_diverged"] is restore_disabled_diverged
    assert measured["parameters_changed"] is parameters_changed
    assert measured["max_parameter_delta"] == max_parameter_delta
