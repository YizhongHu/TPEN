"""Formal review round 3: the native trainer_cursor arm is measured."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest

from tpen.checkpoint import CheckpointRef, list_complete_checkpoints
from tpen.execution import BalsamAdapter, WorkerStartGateway
from tpen.execution.chain_resume import ChannelParityError, compare_channel_parity
from tpen.execution.recovery import CheckpointEvidence
from tests.integration.test_native_checkpoint_recovery import (
    FIXTURE,
    QualificationClient,
    _batch,
    _launch,
    _trace,
    _write_inputs,
)


def _wait_for_two_prefix_checkpoints(outputs: Path, *, target: int, process) -> list[Path]:
    """Wait until at least two committed prefix checkpoints exist."""

    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        complete = [
            path
            for root in sorted(outputs.rglob("checkpoints"))
            for path in list_complete_checkpoints(root)
        ]
        prefixes = [
            path
            for path in complete
            if int(path.name.removeprefix("step_")) < target
        ]
        if len(prefixes) >= 2:
            return prefixes
        if process.poll() is not None:
            raise AssertionError(
                f"prefix worker exited (rc={process.returncode}) before two "
                "committed prefix checkpoints"
            )
        time.sleep(0.05)
    raise AssertionError("no two committed prefix checkpoints before deadline")


@pytest.fixture(scope="module")
def cursor_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Create a killed prefix and four fresh traced native workers."""

    root = tmp_path_factory.mktemp("r3-native-cursor")
    batch, intent, _source = _batch()
    batch_path, intent_path = _write_inputs(root, batch, intent)
    receipts = root / "receipts"
    outputs = root / "outputs"
    outputs.mkdir()
    adapter = BalsamAdapter(
        batch.execution, QualificationClient(), WorkerStartGateway(outputs), receipts
    )
    assert adapter.register_lineage(batch) is not None

    prefix = _launch(
        root=root,
        batch_path=batch_path,
        intent_path=intent_path,
        config_path=FIXTURE,
        receipts=receipts,
        outputs=outputs,
        phase="initial",
        label="r3-cursor-prefix",
    )
    prefix_dirs = _wait_for_two_prefix_checkpoints(
        outputs, target=batch.target_iterations, process=prefix
    )
    os.kill(prefix.pid, signal.SIGSTOP)
    os.kill(prefix.pid, signal.SIGKILL)
    prefix.wait(timeout=30)

    refs = sorted(
        (CheckpointRef.from_directory(path) for path in prefix_dirs),
        key=lambda ref: ref.next_iteration,
    )
    low, high = refs[0], refs[-1]
    assert low.next_iteration < high.next_iteration < batch.target_iterations

    def _evidence_file(ref: CheckpointRef, name: str) -> Path:
        evidence = CheckpointEvidence.from_ref(
            ref,
            science_input_digest=intent.science_input_digest,
            target_iterations=batch.target_iterations,
        )
        path = root / name
        path.write_text(json.dumps(evidence.to_dict(), sort_keys=True), encoding="utf-8")
        return path

    low_evidence = _evidence_file(low, "r3-evidence-low.json")
    high_evidence = _evidence_file(high, "r3-evidence-high.json")
    launches = {
        "reference": dict(phase="initial", evidence=None, resume_step=0, disable=None),
        "resume-high": dict(
            phase="recovery", evidence=high_evidence,
            resume_step=high.next_iteration, disable=None,
        ),
        "resume-low": dict(
            phase="recovery", evidence=low_evidence,
            resume_step=low.next_iteration, disable=None,
        ),
        "cursor-disabled": dict(
            phase="recovery", evidence=high_evidence,
            resume_step=high.next_iteration, disable="trainer_cursor",
        ),
    }
    traces = {}
    for label, spec in launches.items():
        trace_path = root / f"r3-{label}.trace.json"
        worker = _launch(
            root=root,
            batch_path=batch_path,
            intent_path=intent_path,
            config_path=FIXTURE,
            receipts=receipts,
            outputs=outputs,
            phase=spec["phase"],
            label=f"r3-{label}",
            evidence=spec["evidence"],
            trace=trace_path,
            resume_step=spec["resume_step"],
            disable_limb=spec["disable"],
        )
        worker.wait(timeout=240)
        assert worker.returncode == 0, label
        traces[label] = _trace(trace_path)

    return {
        "traces": traces,
        "low_cursor": low.next_iteration,
        "high_cursor": high.next_iteration,
        "seed": int(batch.items[0].payload["seed"]),
    }


def _first_cursor_step(trace) -> int:
    """Return the first trainer_cursor event of one real worker."""

    return next(
        event.step for event in trace.events if event.channel == "trainer_cursor"
    )


@pytest.mark.integration
def test_trainer_cursor_omission_is_measurable(cursor_evidence: dict) -> None:
    """The restore-disabled cursor arm has a measurable own-channel signature."""

    traces = cursor_evidence["traces"]
    high_cursor = cursor_evidence["high_cursor"]
    enabled_first = _first_cursor_step(traces["resume-high"])
    disabled_first = _first_cursor_step(traces["cursor-disabled"])
    assert enabled_first == high_cursor
    assert disabled_first != high_cursor
    assert disabled_first == 0
    assert traces["resume-high"].pid != traces["cursor-disabled"].pid


@pytest.mark.integration
def test_trainer_cursor_positive_control_moves_own_channel(cursor_evidence: dict) -> None:
    """Different committed cursors move the first-resumed-step observable."""

    traces = cursor_evidence["traces"]
    low_first = _first_cursor_step(traces["resume-low"])
    high_first = _first_cursor_step(traces["resume-high"])
    assert low_first == cursor_evidence["low_cursor"]
    assert high_first == cursor_evidence["high_cursor"]
    assert low_first != high_first
    assert traces["resume-low"].seed == traces["resume-high"].seed == cursor_evidence["seed"]
    assert traces["resume-low"].pid != traces["resume-high"].pid


@pytest.mark.integration
def test_trainer_cursor_named_mismatch_through_public_adapter(cursor_evidence: dict) -> None:
    """The public comparator names the cursor restart."""

    traces = cursor_evidence["traces"]
    high_cursor = cursor_evidence["high_cursor"]
    with pytest.raises(ChannelParityError, match="trainer_cursor"):
        compare_channel_parity(
            traces["reference"],
            traces["cursor-disabled"],
            resume_step=high_cursor - 1,
            disabled_limbs=("trainer_cursor",),
        )
