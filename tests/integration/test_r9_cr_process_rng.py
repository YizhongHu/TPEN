"""R9-1: traced native process-RNG coupling and controls."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest

from tests.integration import test_native_checkpoint_recovery as native
from tpen.execution import BalsamAdapter, WorkerStartGateway
from tpen.execution.chain_resume import (
    E5_CHANNEL_MAP,
    E5_NATIVE_EXPECTED_MISMATCHES,
    compare_channel_parity,
    first_subsequent_event,
)
from tpen.execution.invocation import _committed_checkpoint
from tpen.execution.recovery import CheckpointEvidence

LIMBS = tuple(sorted({spec.limb for spec in E5_CHANNEL_MAP}))


class _Lane:
    """One isolated fake-Balsam lineage."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.batch, self.intent, _source = native._batch()
        self.batch_path, self.intent_path = native._write_inputs(root, self.batch, self.intent)
        self.receipts = root / "receipts"
        self.outputs = root / "outputs"
        self.outputs.mkdir()
        adapter = BalsamAdapter(
            self.batch.execution, native.QualificationClient(), WorkerStartGateway(self.outputs), self.receipts
        )
        assert adapter.register_lineage(self.batch) is not None

    def launch(self, **kwargs):
        return native._launch(
            root=self.root, batch_path=self.batch_path, intent_path=self.intent_path,
            config_path=native.FIXTURE, receipts=self.receipts, outputs=self.outputs, **kwargs,
        )

    def evidence(self, ref) -> Path:
        path = self.root / "evidence-resume.json"
        path.write_text(json.dumps(CheckpointEvidence.from_ref(
            ref, science_input_digest=self.intent.science_input_digest,
            target_iterations=self.batch.target_iterations,
        ).to_dict(), sort_keys=True), encoding="utf-8")
        return path


def _kill(process) -> int:
    os.kill(process.pid, signal.SIGSTOP)
    os.kill(process.pid, signal.SIGKILL)
    return process.wait(timeout=30)


def _build_lane(tmp_path_factory):
    lane = _Lane(tmp_path_factory.mktemp("r9-traced"))
    prefix_trace = lane.root / "prefix.trace.json"
    prefix = lane.launch(phase="initial", label="r9-prefix", trace=prefix_trace)
    run_dir, _observed = native._wait_for_prefix_checkpoints(
        lane.outputs, target=lane.batch.target_iterations, minimum=2, process=prefix
    )
    _kill(prefix)
    selected = _committed_checkpoint(run_dir)
    assert selected is not None
    rc = selected.next_iteration
    assert rc < lane.batch.target_iterations
    evidence = lane.evidence(selected)
    jobs = {"reference": dict(phase="initial", resume_step=0)}
    jobs["full"] = dict(phase="recovery", evidence=evidence, resume_step=rc)
    jobs.update({f"disabled-{limb}": dict(phase="recovery", evidence=evidence, resume_step=rc, disable_limb=limb) for limb in LIMBS})
    traces = {}
    for label, options in jobs.items():
        trace = lane.root / f"{label}.trace.json"
        worker = lane.launch(label=f"r9-{label}", trace=trace, **options)
        rc_worker = worker.wait(timeout=180)
        assert rc_worker == 0, f"native worker {label} returned {rc_worker!r}"
        traces[label] = native._trace(trace)
    print(f"R9_LANE traced rc={rc}")
    return lane, rc, traces


@pytest.fixture(scope="module")
def traced_lane(tmp_path_factory):
    return _build_lane(tmp_path_factory)


@pytest.mark.integration
# PREDICTED: GREEN
def test_r9_traced_full_restore_satisfies_uncoupled_parity(traced_lane):
    _lane, rc, traces = traced_lane
    compare_channel_parity(traces["reference"], traces["full"], resume_step=rc - 1)


@pytest.mark.integration
# PREDICTED: GREEN
def test_r9_traced_process_rng_omission_is_distinguishable_from_full_restore(traced_lane):
    _lane, rc, traces = traced_lane
    spec = next(s for s in E5_CHANNEL_MAP if s.channel == "process_random")
    full = first_subsequent_event(traces["full"], spec, resume_step=rc - 1)
    omitted = first_subsequent_event(traces["disabled-process_rng"], spec, resume_step=rc - 1)
    assert omitted.value != full.value


def _signatures(rc, traces):
    result = {}
    for limb in LIMBS:
        disabled = traces[f"disabled-{limb}"]
        mismatches = set()
        for spec in E5_CHANNEL_MAP:
            left = first_subsequent_event(traces["reference"], spec, resume_step=rc - 1).value
            right = first_subsequent_event(disabled, spec, resume_step=-1 if limb == "trainer_cursor" else rc - 1).value
            if left != right:
                mismatches.add(spec.channel)
        result[limb] = sorted(mismatches)
    return result


@pytest.mark.integration
# PREDICTED: GREEN
def test_r9_traced_single_omission_signatures(traced_lane):
    lane, rc, traces = traced_lane
    observed = _signatures(rc, traces)
    expected = {"model": ["model_logabs", "optimizer_update"], "optimizer": ["optimizer_update"],
                "process_rng": ["process_random"], "sampler_mcmc": ["model_logabs", "optimizer_update", "sampler_proposal"],
                "trainer_cursor": ["trainer_cursor"]}
    payload = {k: set(v) for k, v in observed.items()}
    (lane.root / "r9-signatures.json").write_text(json.dumps(observed, sort_keys=True), encoding="utf-8")
    print("R9_SIGNATURES=" + json.dumps(observed, sort_keys=True))
    assert payload == {k: set(v) for k, v in expected.items()}
    assert payload == {k: set(v) for k, v in E5_NATIVE_EXPECTED_MISMATCHES.items()}
