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
from tpen.execution.chain_resume import (
    ChannelParityError,
    ProcessEvidence,
    UnmeasuredArmError,
    assert_positive_control,
    compare_channel_parity,
    first_subsequent_event,
)
from tpen.execution.chain_resume import E5_CHANNEL_MAP
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
    trace: Path | None = None,
    resume_step: int = 0,
    disable_limb: str | None = None,
    disable_limbs: tuple[str, ...] = (),
    suppress_channel: str | None = None,
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
    if trace is not None:
        command.extend(("--trace", str(trace), "--resume-step", str(resume_step)))
    if disable_limb is not None:
        command.extend(("--disable-limb", disable_limb))
    for limb in disable_limbs:
        command.extend(("--disable-limb", limb))
    if suppress_channel is not None:
        command.extend(("--suppress-channel", suppress_channel))
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


def _trace(path: Path) -> ProcessEvidence:
    """Read a real worker's test-side trace without importing the worker."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        json.dumps({"record": "channel_event", **event}, sort_keys=True)
        for event in payload["events"]
    ]
    return ProcessEvidence.from_json_output(
        phase=payload["phase"],
        pid=int(payload["pid"]),
        seed=int(payload["seed"]),
        seed_source=payload["seed_source"],
        command=("native_recovery_worker", payload["phase"]),
        returncode=0,
        stdout="\n".join(lines),
    )


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
        trace=tmp_path / "trace-prefix.json",
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
        trace=tmp_path / "trace-recovery.json",
        resume_step=prefix_checkpoint.next_iteration,
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
        trace=tmp_path / "trace-reference.json",
    )
    reference.wait(timeout=180)
    assert reference.returncode == 0
    reference_result = _result_from_log(tmp_path, "reference")
    assert reference_result["invocation"]["outcome"] == "completed"

    reference_trace = _trace(tmp_path / "trace-reference.json")
    resumed_trace = _trace(tmp_path / "trace-recovery.json")
    assert reference_trace.pid != resumed_trace.pid
    assert reference_trace.seed == resumed_trace.seed == int(batch.items[0].payload["seed"])
    assert reference_trace.seed_source != resumed_trace.seed_source
    # ``next_iteration`` is the first step attempted by the successor, so the
    # comparison cursor is the preceding completed step.
    real_channel_results = compare_channel_parity(
        reference_trace,
        resumed_trace,
        resume_step=prefix_checkpoint.next_iteration - 1,
    )
    assert all(real_channel_results.values())

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


@pytest.mark.integration
def test_real_restore_omission_channels_are_observed(tmp_path: Path) -> None:
    """Exercise each E5 restore owner against real E3/native executions.

    The disabled-owner runs use a test-only hook in the checkpoint restore
    module.  Production validation still runs; only one component loader is
    skipped.  Model and sampler omissions can causally alter downstream
    channels, so this test asserts only the channels that remain distinguishable
    for those arms instead of claiming false one-limb isolation.
    """

    batch, intent, _source = _batch()
    batch_path, intent_path = _write_inputs(tmp_path, batch, intent)
    receipts = tmp_path / "receipts"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    client = QualificationClient()
    adapter = BalsamAdapter(
        batch.execution, client, WorkerStartGateway(outputs), receipts
    )
    registration = adapter.register_lineage(batch)
    assert registration is not None

    prefix = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="initial",
        label="channel-prefix",
    )
    prefix_checkpoint = _kill_prefix(prefix, outputs, batch.target_iterations)
    evidence = CheckpointEvidence.from_ref(
        prefix_checkpoint,
        science_input_digest=intent.science_input_digest,
        target_iterations=batch.target_iterations,
    )
    evidence_path = tmp_path / "channel-evidence.json"
    evidence_path.write_text(json.dumps(evidence.to_dict(), sort_keys=True), encoding="utf-8")

    reference = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="initial",
        label="channel-reference",
        trace=tmp_path / "channel-reference.trace.json",
        resume_step=0,
    )
    reference.wait(timeout=180)
    assert reference.returncode == 0
    reference_trace = _trace(tmp_path / "channel-reference.trace.json")

    resume_cursor = prefix_checkpoint.next_iteration
    results: dict[str, dict[str, bool]] = {}
    disabled_traces: dict[str, ProcessEvidence] = {}
    for limb in sorted({spec.limb for spec in E5_CHANNEL_MAP}):
        label = f"channel-disabled-{limb}"
        trace_path = tmp_path / f"{label}.trace.json"
        worker = _launch(
            root=tmp_path,
            batch_path=batch_path,
            intent_path=intent_path,
            config_path=FIXTURE,
            receipts=receipts,
            outputs=outputs,
            phase="recovery",
            label=label,
            evidence=evidence_path,
            trace=trace_path,
            resume_step=resume_cursor,
            disable_limb=limb,
        )
        worker.wait(timeout=180)
        assert worker.returncode == 0, label
        disabled_trace = _trace(trace_path)
        disabled_traces[limb] = disabled_trace
        assert disabled_trace.pid != reference_trace.pid
        assert disabled_trace.seed == reference_trace.seed
        # The pure core comparator intentionally raises when one arm changes
        # a downstream observable.  This test needs the complete measured
        # vector first so it can distinguish a named omission from a causal
        # downstream change rather than misreporting the arm as unmeasured.
        result = {
            spec.channel: (
                first_subsequent_event(
                    reference_trace, spec, resume_step=resume_cursor - 1
                ).value
                == first_subsequent_event(
                    disabled_trace, spec, resume_step=resume_cursor - 1
                ).value
            )
            for spec in E5_CHANNEL_MAP
        }
        results[limb] = result
        channel = next(spec.channel for spec in E5_CHANNEL_MAP if spec.limb == limb)
        if limb == "trainer_cursor":
            # E3 supplies the selected cursor to the configured runner before
            # the checkpoint mutator runs.  Therefore this hook cannot isolate
            # a cursor-only omission from that already-bound decision.  Record
            # the actual first step and assert the distinguishable fact: the
            # hook did not cause a restart at the selected resume cursor.
            first_cursor = next(
                event.step
                for event in disabled_trace.events
                if event.channel == "trainer_cursor"
            )
            assert first_cursor != resume_cursor
            result[channel] = first_cursor == next(
                event.step
                for event in reference_trace.events
                if event.channel == "trainer_cursor" and event.step > resume_cursor - 1
            )
        else:
            assert result[channel] is False, limb

        if limb == "optimizer":
            # The real configured topology does not isolate the global-RNG
            # trace from an optimizer-loader omission, so downstream values
            # are evidence, not claimed independent channels.  The resumed
            # cursor remains independently observable.
            assert result["trainer_cursor"] is True, (limb, "trainer_cursor")
        elif limb == "process_rng":
            for unaffected in ("model_logabs", "optimizer_update", "sampler_proposal", "trainer_cursor"):
                assert result[unaffected] is True, (limb, unaffected)
        elif limb == "model":
            # A model omission changes the model evaluations used by the
            # sampler and optimizer, and the measured process-RNG channel is
            # likewise not isolatable through this topology.  Only the cursor
            # remains independently distinguishable here.
            for unaffected in ("trainer_cursor",):
                assert result[unaffected] is True, (limb, unaffected)
        elif limb == "sampler_mcmc":
            assert result["trainer_cursor"] is True, (limb, "trainer_cursor")

    assert set(results) == {spec.limb for spec in E5_CHANNEL_MAP}

    # Positive controls now consume the vectors emitted by the real native
    # workers above.  The synthetic probe has a separate adapter-unit label
    # and is not used as evidence for this assertion.
    for limb, disabled_trace in disabled_traces.items():
        if limb == "trainer_cursor":
            # Cursor-only omission is explicitly non-isolatable in this
            # topology; its measured hook result was recorded above, but it is
            # not presented as a positive-control mismatch.
            continue
        spec = next(spec for spec in E5_CHANNEL_MAP if spec.limb == limb)
        assert_positive_control(
            reference_trace,
            disabled_trace,
            spec,
            resume_step=resume_cursor - 1,
        )

    # Observability mutant: the restored limb remains live in a real E3 run,
    # but its first event is suppressed in the test-side trace.  Missing data
    # is unmeasured, never a parity result.
    suppressed_trace_path = tmp_path / "channel-suppressed.trace.json"
    suppressed = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="recovery",
        label="channel-suppressed",
        evidence=evidence_path,
        trace=suppressed_trace_path,
        resume_step=resume_cursor,
        suppress_channel="sampler_proposal",
    )
    suppressed.wait(timeout=180)
    assert suppressed.returncode == 0
    suppressed_trace = _trace(suppressed_trace_path)
    with pytest.raises(UnmeasuredArmError, match="sampler_proposal"):
        first_subsequent_event(
            suppressed_trace,
            next(spec for spec in E5_CHANNEL_MAP if spec.channel == "sampler_proposal"),
            resume_step=resume_cursor - 1,
        )

    # Wrong-channel attribution mutant: the real process-RNG omission is
    # attributed to model_logabs.  The first channel assertion must reject
    # that attribution before a later channel can accidentally mask it.
    with pytest.raises(ChannelParityError, match="model_logabs"):
        compare_channel_parity(
            reference_trace,
            disabled_traces["process_rng"],
            resume_step=resume_cursor - 1,
            disabled_limbs=("model",),
        )

    # Mutual-masking mutant: use one real single-limb omission while claiming
    # every limb was disabled, then use a real multi-omission run while
    # claiming only one.  Both assertions must fail on the observed vector.
    with pytest.raises(ChannelParityError, match="model_logabs"):
        compare_channel_parity(
            reference_trace,
            disabled_traces["process_rng"],
            resume_step=resume_cursor - 1,
            disabled_limbs=tuple(spec.limb for spec in E5_CHANNEL_MAP),
        )
    all_disabled_path = tmp_path / "channel-all-disabled.trace.json"
    all_disabled = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="recovery",
        label="channel-all-disabled",
        evidence=evidence_path,
        trace=all_disabled_path,
        resume_step=resume_cursor,
        disable_limbs=tuple(spec.limb for spec in E5_CHANNEL_MAP),
    )
    all_disabled.wait(timeout=180)
    assert all_disabled.returncode == 0
    with pytest.raises(ChannelParityError, match="channel"):
        compare_channel_parity(
            reference_trace,
            _trace(all_disabled_path),
            resume_step=resume_cursor - 1,
            disabled_limbs=("model",),
        )


@pytest.mark.integration
def test_real_checkpoint_boundary_faults_are_typed(tmp_path: Path) -> None:
    """Exercise no-checkpoint, before-commit, after-commit, and corruption.

    Every arm enters through the E4 worker seam and E3 configured-run gateway.
    The scheduler/process signal is retained separately from the typed
    checkpoint/application result; a killed prefix is never treated as a
    successful science outcome.
    """

    batch, intent, _source = _batch()
    batch_path, intent_path = _write_inputs(tmp_path, batch, intent)
    receipts = tmp_path / "receipts"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    client = QualificationClient()
    adapter = BalsamAdapter(
        batch.execution, client, WorkerStartGateway(outputs), receipts
    )
    assert adapter.register_lineage(batch) is not None

    # No evidence is an explicit E1 park through the real worker process.
    no_checkpoint = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="recovery",
        label="fault-no-checkpoint",
    )
    no_checkpoint.wait(timeout=60)
    assert no_checkpoint.returncode == 0
    no_checkpoint_result = _result_from_log(tmp_path, "fault-no-checkpoint")
    assert no_checkpoint_result["status"] == "started"
    assert no_checkpoint_result["invocation"]["outcome"] == "parked"
    assert "no committed checkpoint" in no_checkpoint_result["invocation"]["reason"]

    # Kill a fresh worker before it can publish a committed checkpoint.  This
    # is the application/scheduler fact; the successor still parks through E1
    # rather than silently restarting.
    before_root = tmp_path / "before-commit-outputs"
    before_root.mkdir()
    before = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=before_root,
        phase="initial",
        label="fault-before-commit",
    )
    os.kill(before.pid, signal.SIGSTOP)
    os.kill(before.pid, signal.SIGKILL)
    before.wait(timeout=30)
    assert before.returncode == -signal.SIGKILL
    assert not list(before_root.rglob("checkpoints/step_*/COMPLETE"))

    before_successor = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=before_root,
        phase="recovery",
        label="fault-before-commit-successor",
    )
    before_successor.wait(timeout=60)
    assert before_successor.returncode == 0
    before_result = _result_from_log(tmp_path, "fault-before-commit-successor")
    assert before_result["status"] == "started"
    assert before_result["invocation"]["outcome"] == "parked"
    assert "no committed checkpoint" in before_result["invocation"]["reason"]

    # A committed prefix is the after-commit arm.  Select through the same
    # catalog/latest resolver used by E3, then corrupt only its manifest before
    # the successor enters restore.  The immutable checkpoint binding must
    # park this as application evidence, not as a scheduler loss or a silent
    # restart.  A changed committed artifact is not a deterministic config
    # refusal that the worker may retry.
    prefix = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="initial",
        label="fault-after-commit",
    )
    checkpoint = _kill_prefix(prefix, outputs, batch.target_iterations)
    evidence = CheckpointEvidence.from_ref(
        checkpoint,
        science_input_digest=intent.science_input_digest,
        target_iterations=batch.target_iterations,
    )
    evidence_path = tmp_path / "fault-corrupt-evidence.json"
    evidence_path.write_text(json.dumps(evidence.to_dict(), sort_keys=True), encoding="utf-8")
    manifest = checkpoint.checkpoint_dir / "manifest.json"
    original_manifest = manifest.read_text(encoding="utf-8")
    corrupt_manifest = json.loads(original_manifest)
    corrupt_manifest["schema_version"] = 999
    manifest.write_text(json.dumps(corrupt_manifest, sort_keys=True) + "\n", encoding="utf-8")
    corrupt = _launch(
        root=tmp_path,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="recovery",
        label="fault-corrupt-committed",
        evidence=evidence_path,
    )
    corrupt.wait(timeout=60)
    assert corrupt.returncode == 0
    corrupt_result = _result_from_log(tmp_path, "fault-corrupt-committed")
    assert corrupt_result["status"] == "started"
    assert corrupt_result["invocation"]["outcome"] == "parked"
    assert corrupt_result["invocation"]["status"] == "parked"
    assert corrupt_result["invocation"]["reason"]
    manifest.write_text(original_manifest, encoding="utf-8")
