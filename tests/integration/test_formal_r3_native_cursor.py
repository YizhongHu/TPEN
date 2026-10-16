"""Formal review round 3: the trainer_cursor arms declared UNMEASURED, measured.

Target: CR @ 596033ecff7ca37a5c8ff5e5cda635c9b9a3ab60.

The r2 completion declared ``restore-disabled:trainer_cursor`` and
``positive-control:trainer_cursor`` UNMEASURABLE because "E3 binds the recovery
cursor before the test-only trainer loader hook".  Production ``Train.run``
never rebinds the trainer cursor from the restore report and E3 only records
the selected cursor in the start receipt; the sole cursor binding is
``tpen.checkpoint.restore._load_trainer``, which the test hook does skip.  The
cursor limb is therefore measurable through its own channel when the
observable is the FIRST trainer_cursor event of the resumed process rather
than the value at a step-aligned event (which is tautologically equal to the
step).  These arms measure exactly that, with real worker processes through
the E4 seam and E3 gateway.

Polarity at CR (preregistered):
- test_trainer_cursor_omission_is_measurable: EXPECTED GREEN.  Demonstrates
  the restore-disabled cursor arm: the disabled worker restarts at step 0, the
  enabled worker starts at the selected cursor.
- test_trainer_cursor_positive_control_moves_own_channel: EXPECTED GREEN.
  Demonstrates the own-channel positive control: resuming from two different
  committed cursors moves the first-resumed-step observable, with identical
  seed, config, and lineage.
- test_trainer_cursor_named_mismatch_through_public_adapter: EXPECTED RED.
  The public comparator detects the restart only through the generic
  before-cursor guard on whichever channel sorts first, so it cannot name
  ``trainer_cursor`` as the mismatched limb.
"""

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
    """Wait until at least two committed prefix checkpoints exist, then return them."""

    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        # The checkpoint writer stages ``step_NNNNNN.tmp/`` (COMPLETE included)
        # before its commit rename, so discovery must use the core predicate
        # rather than a ``step_*/COMPLETE`` glob that matches the staging dir.
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
    """One prefix with >=2 committed cursors, then four traced workers."""

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
    """Return the first trainer_cursor event step of one real worker process."""

    return next(
        event.step for event in trace.events if event.channel == "trainer_cursor"
    )


@pytest.mark.integration
def test_trainer_cursor_omission_is_measurable(cursor_evidence: dict) -> None:
    """GREEN: the restore-disabled cursor arm has a measurable own-channel signature."""

    traces = cursor_evidence["traces"]
    high_cursor = cursor_evidence["high_cursor"]
    enabled_first = _first_cursor_step(traces["resume-high"])
    disabled_first = _first_cursor_step(traces["cursor-disabled"])
    assert enabled_first == high_cursor, (
        "enabled restore did not resume at the selected cursor; the cursor "
        f"channel observable is broken (first step {enabled_first})"
    )
    assert disabled_first != high_cursor, (
        "the trainer loader hook did not change the resumed start; the "
        "declared-unmeasured reason would be substantiated"
    )
    assert disabled_first == 0, (
        f"cursor-disabled run started at {disabled_first}, expected a restart at 0"
    )
    assert traces["resume-high"].pid != traces["cursor-disabled"].pid


@pytest.mark.integration
def test_trainer_cursor_positive_control_moves_own_channel(cursor_evidence: dict) -> None:
    """GREEN: a cursor-only perturbation moves the cursor channel observable."""

    traces = cursor_evidence["traces"]
    low_first = _first_cursor_step(traces["resume-low"])
    high_first = _first_cursor_step(traces["resume-high"])
    assert low_first == cursor_evidence["low_cursor"]
    assert high_first == cursor_evidence["high_cursor"]
    assert low_first != high_first, (
        "resuming from two different committed cursors did not move the "
        "trainer_cursor first-resumed-step observable"
    )
    # Same seed and config; the only perturbation is the selected cursor.
    assert traces["resume-low"].seed == traces["resume-high"].seed == cursor_evidence["seed"]
    assert traces["resume-low"].pid != traces["resume-high"].pid


@pytest.mark.integration
def test_trainer_cursor_named_mismatch_through_public_adapter(cursor_evidence: dict) -> None:
    """RED at CR: the public comparator cannot attribute the restart to the cursor."""

    traces = cursor_evidence["traces"]
    high_cursor = cursor_evidence["high_cursor"]
    with pytest.raises(ChannelParityError, match="trainer_cursor"):
        compare_channel_parity(
            traces["reference"],
            traces["cursor-disabled"],
            resume_step=high_cursor - 1,
            disabled_limbs=("trainer_cursor",),
        )
