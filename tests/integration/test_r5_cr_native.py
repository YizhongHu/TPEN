"""Formal review round 5 reviewer tests (CR native): issues 2, 3, 4, 5 and 11.

Every worker here is a real fresh OS process launched through the CR native
helper ``tests.integration.test_native_checkpoint_recovery._launch``.  The
parent registers one fake persistent Job; workers never call a backend.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.integration import test_native_checkpoint_recovery as native
from tpen.checkpoint import list_complete_checkpoints
from tpen.execution import BalsamAdapter, InvocationOutcome, WorkerStartGateway
from tpen.execution.chain_resume import E5_NATIVE_EXPECTED_MISMATCHES, E5_CHANNEL_MAP, assert_fresh_processes, compare_channel_parity
from tpen.execution.invocation import _committed_checkpoint
from tpen.execution.recovery import CheckpointEvidence

LIMBS = tuple(sorted({spec.limb for spec in E5_CHANNEL_MAP}))


class _Lane:
    """One registered fake-Balsam lineage rooted at ``root``."""

    def __init__(self, root: Path, *, config_path: Path | None = None) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.batch, self.intent, _source = native._batch()
        self.batch_path, self.intent_path = native._write_inputs(root, self.batch, self.intent)
        self.receipts = root / "receipts"
        self.outputs = root / "outputs"
        self.outputs.mkdir(exist_ok=True)
        self.config_path = native.FIXTURE if config_path is None else config_path
        adapter = BalsamAdapter(
            self.batch.execution, native.QualificationClient(), WorkerStartGateway(self.outputs), self.receipts
        )
        assert adapter.register_lineage(self.batch) is not None, "fake registration was not acknowledged"

    def launch(self, **kwargs):
        return native._launch(
            root=self.root,
            batch_path=self.batch_path,
            intent_path=self.intent_path,
            config_path=self.config_path,
            receipts=self.receipts,
            outputs=self.outputs,
            **kwargs,
        )

    def evidence(self, ref, name: str) -> Path:
        path = self.root / name
        data = CheckpointEvidence.from_ref(
            ref,
            science_input_digest=self.intent.science_input_digest,
            target_iterations=self.batch.target_iterations,
        ).to_dict()
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        return path


def _first_cursor(trace) -> int:
    return next(event.step for event in trace.events if event.channel == "trainer_cursor")


def _kill(process) -> int:
    os.kill(process.pid, signal.SIGSTOP)
    os.kill(process.pid, signal.SIGKILL)
    return process.wait(timeout=30)


@pytest.fixture(scope="module")
def r5_traces(tmp_path_factory):
    """One killed prefix, then ten traced fresh workers (reference, full, omissions, mutant)."""

    lane = _Lane(tmp_path_factory.mktemp("r5-native"))
    target = lane.batch.target_iterations
    prefix = lane.launch(phase="initial", label="r5-prefix")
    run_dir, observed_refs = native._wait_for_prefix_checkpoints(
        lane.outputs, target=target, minimum=2, process=prefix
    )
    _kill(prefix)
    prefix_checkpoint = _committed_checkpoint(run_dir)
    assert prefix_checkpoint is not None, "no committed prefix checkpoint after the kill"
    rc = prefix_checkpoint.next_iteration
    assert rc < target, f"prefix reached the target before the kill: cursor {rc}"
    lower = [ref for ref in observed_refs if ref.next_iteration < rc]
    assert lower, f"no observed checkpoint below the selected cursor {rc}"
    resume_evidence = lane.evidence(prefix_checkpoint, "evidence-resume.json")
    lower_evidence = lane.evidence(lower[0], "evidence-lower.json")

    jobs = [
        ("reference", dict(phase="initial", resume_step=0)),
        ("full", dict(phase="recovery", evidence=resume_evidence, resume_step=rc)),
    ]
    jobs += [
        (f"disabled-{limb}", dict(phase="recovery", evidence=resume_evidence, resume_step=rc, disable_limb=limb))
        for limb in LIMBS
    ]
    jobs += [
        ("all-disabled", dict(phase="recovery", evidence=resume_evidence, resume_step=rc, disable_limbs=LIMBS)),
        # Mutant: restored from the LOWER checkpoint while the trace claims cursor rc.
        ("wrong-cursor", dict(phase="recovery", evidence=lower_evidence, resume_step=rc)),
    ]
    traces = {}
    for label, options in jobs:
        trace_path = lane.root / f"{label}.trace.json"
        process = lane.launch(label=f"r5-{label}", trace=trace_path, **options)
        returncode = process.wait(timeout=180)
        assert returncode == 0, f"native worker {label} returned {returncode!r}"
        traces[label] = native._trace(trace_path)
    print(f"R5_TRACES rc={rc} lower={lower[0].next_iteration} pids={ {k: v.pid for k, v in traces.items()} }")
    return {"rc": rc, "lower_cursor": lower[0].next_iteration, "traces": traces}


@pytest.mark.integration
# PREDICTED: GREEN at CR (issue 2 control)
def test_r5_reference_first_cursor_is_zero(r5_traces):
    observed = _first_cursor(r5_traces["traces"]["reference"])
    assert observed == 0, f"observed reference first cursor={observed!r}"


@pytest.mark.integration
# PREDICTED: GREEN at CR (issue 2 control)
def test_r5_full_restore_first_cursor_is_resume_cursor(r5_traces):
    observed = _first_cursor(r5_traces["traces"]["full"])
    assert observed == r5_traces["rc"], f"observed full-restore first cursor={observed!r}, selected={r5_traces['rc']!r}"


@pytest.mark.integration
# PREDICTED: GREEN at CR (issue 2 measured fact: the CR predicate is already False with NO omission)
def test_r5_cr_cursor_predicate_is_false_without_any_omission(r5_traces):
    traces = r5_traces["traces"]
    # Exact predicate of tests/integration/test_native_checkpoint_recovery.py:776-800 at CR.
    cr_value = _first_cursor(traces["full"]) == _first_cursor(traces["reference"])
    assert cr_value is False, f"CR predicate on the full-restore trace observed {cr_value!r}"


@pytest.mark.integration
@pytest.mark.parametrize("limb", ["optimizer", "model", "sampler_mcmc", "process_rng"])
# PREDICTED: GREEN at CR (issue 2: a non-cursor omission leaves the restored cursor unchanged)
def test_r5_non_cursor_omissions_leave_cursor_restored(r5_traces, limb):
    observed = _first_cursor(r5_traces["traces"][f"disabled-{limb}"])
    assert observed == r5_traces["rc"], f"{limb} omission first cursor={observed!r}, selected={r5_traces['rc']!r}"


@pytest.mark.integration
# PREDICTED: RED at CR (issue 2)
def test_r5_cr_cursor_predicate_detects_wrong_cursor_mutant(r5_traces):
    traces = r5_traces["traces"]
    # R5 issue 2: consume the real assignment AND assertion rather than
    # permanently replaying the defective predicate in the reviewer probe.
    tree = ast.parse(Path(native.__file__).read_text(encoding="utf-8"))
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "test_real_restore_omission_channels_are_observed")
    branches = [node for node in ast.walk(func) if isinstance(node, ast.If)
                and ast.dump(node.test) == ast.dump(ast.parse('limb == "optimizer"', mode="eval").body)]
    assert len(branches) == 1, "cursor assertion branch missing or ambiguous"
    assignments = [node for node in branches[0].body if isinstance(node, ast.Assign)]
    assertions = [node for node in branches[0].body if isinstance(node, ast.Assert)]
    assert len(assignments) == len(assertions) == 1
    assignment, assertion = assignments[0], assertions[0]
    assert ast.unparse(assignment.targets[0]) == "result['trainer_cursor']"
    assert isinstance(assignment.value, ast.Compare) and len(assignment.value.ops) == 1
    assert isinstance(assignment.value.ops[0], ast.Eq)
    assert ast.unparse(assignment.value.left) == "disabled_first_cursor"
    assert ast.unparse(assignment.value.comparators[0]) in {"resume_cursor", "reference_first_cursor"}
    assert isinstance(assertion.test, ast.Compare) and len(assertion.test.ops) == 1
    assert isinstance(assertion.test.ops[0], ast.Is)
    assert ast.unparse(assertion.test.left) == "result['trainer_cursor']"
    assert isinstance(assertion.test.comparators[0], ast.Constant)
    assert isinstance(assertion.test.comparators[0].value, bool)
    namespace = dict(result={}, disabled_first_cursor=_first_cursor(traces["wrong-cursor"]),
                     resume_cursor=r5_traces["rc"], reference_first_cursor=_first_cursor(traces["reference"]))
    namespace["result"]["trainer_cursor"] = eval(compile(ast.Expression(assignment.value), "cursor-assignment", "eval"), namespace)
    cr_assertion_holds = eval(compile(ast.Expression(assertion.test), "cursor-assertion", "eval"), namespace)
    assert not cr_assertion_holds, (
        f"CR cursor predicate accepts a run restored from cursor {r5_traces['lower_cursor']} "
        f"while the selected cursor is {r5_traces['rc']}"
    )


@pytest.mark.integration
# PREDICTED: GREEN at CR (issue 2 control: the corrected predicate rejects the mutant)
def test_r5_corrected_cursor_predicate_detects_wrong_cursor_mutant(r5_traces):
    observed = _first_cursor(r5_traces["traces"]["wrong-cursor"])
    assert observed == r5_traces["lower_cursor"], f"mutant first cursor={observed!r}, lower={r5_traces['lower_cursor']!r}"
    assert observed != r5_traces["rc"], f"mutant first cursor equals the selected cursor {r5_traces['rc']!r}"


@pytest.mark.integration
@pytest.mark.parametrize("limb", LIMBS)
# PREDICTED: GREEN for optimizer and process_rng; RED for model, sampler_mcmc and trainer_cursor at CR (issue 3)
def test_r5_correct_single_limb_attribution_passes(r5_traces, limb):
    traces = r5_traces["traces"]
    try:
        compare_channel_parity(
            traces["reference"],
            traces[f"disabled-{limb}"],
            resume_step=r5_traces["rc"] - 1,
            disabled_limbs=(limb,),
            expected_mismatches=E5_NATIVE_EXPECTED_MISMATCHES,
        )
    except Exception as exc:
        raise AssertionError(f"correct single-limb attribution for {limb} raised {type(exc).__name__}: {exc}") from exc


@pytest.mark.integration
# PREDICTED: RED at CR (issue 3)
def test_r5_all_disabled_correct_attribution_passes(r5_traces):
    traces = r5_traces["traces"]
    try:
        compare_channel_parity(
            traces["reference"],
            traces["all-disabled"],
            resume_step=r5_traces["rc"] - 1,
            disabled_limbs=LIMBS,
            expected_mismatches=E5_NATIVE_EXPECTED_MISMATCHES,
        )
    except Exception as exc:
        raise AssertionError(f"correct all-limb attribution raised {type(exc).__name__}: {exc}") from exc


@pytest.mark.integration
# PREDICTED: RED at CR (issue 11)
def test_r5_resumed_seed_provenance_names_checkpoint_rng(r5_traces):
    reference, full = r5_traces["traces"]["reference"], r5_traces["traces"]["full"]
    print(f"SEED_SOURCE reference={reference.seed_source!r} full={full.seed_source!r}")
    assert "checkpoint" in full.seed_source or "train_resume" in full.seed_source, (
        f"resumed seed_source={full.seed_source!r}; reference seed_source={reference.seed_source!r}"
    )


@pytest.mark.integration
# PREDICTED: GREEN at CR (issue 11 control)
def test_r5_fresh_process_guard_accepts_real_evidence(r5_traces):
    reference, full = r5_traces["traces"]["reference"], r5_traces["traces"]["full"]
    assert_fresh_processes(reference, full)
    assert reference.pid != full.pid, f"observed PIDs reference={reference.pid!r}, full={full.pid!r}"


@pytest.mark.integration
# PREDICTED: RED at CR (issue 4)
def test_r5_before_commit_replica_reaches_e3_before_kill(tmp_path):
    lane = _Lane(tmp_path / "before")
    # R5 issue 4: exercise the shared corrected boundary rather than a
    # disconnected replica of the old Popen/immediate-kill timing.
    process = lane.launch(phase="initial", label="r5-before-commit-replica", pause_before_science=True)
    native._kill_before_first_commit(process, lane.outputs)
    returncode = process.returncode
    assert returncode == -signal.SIGKILL, f"replica returncode={returncode!r}"
    starts = list(lane.outputs.rglob("start.json"))
    log_bytes = sum(path.stat().st_size for path in lane.root.glob("worker-r5-before-commit-replica-*.log"))
    assert starts, f"observed {len(starts)} start.json files; worker log bytes={log_bytes}"


@pytest.mark.integration
# PREDICTED: GREEN at CR (issue 4 control: a real kill between start receipt and first commit)
def test_r5_real_before_commit_kill_reconciles_parked(tmp_path):
    lane = _Lane(tmp_path / "real")
    process = lane.launch(phase="initial", label="r5-real-before-commit")
    deadline = time.monotonic() + 180
    starts: list[Path] = []
    while time.monotonic() < deadline:
        starts = list(lane.outputs.rglob("start.json"))
        if starts:
            break
        if process.poll() is not None:
            pytest.fail(f"worker exited (rc={process.returncode}) before publishing a start receipt")
        time.sleep(0.05)
    assert starts, "start receipt was not observed before the deadline"
    _kill(process)
    committed = [path for root in lane.outputs.rglob("checkpoints") for path in list_complete_checkpoints(root)]
    assert not committed, f"kill landed after the first commit: {committed!r}"
    parked = WorkerStartGateway.reconcile(starts[0].parent)
    assert parked is not None and parked.outcome is InvocationOutcome.PARKED, f"observed reconcile={parked!r}"
    assert "no durable core outcome" in parked.reason, f"observed reason={parked.reason!r}"


@pytest.fixture(scope="module")
def r5_qualification_receipt(tmp_path_factory):
    root = tmp_path_factory.mktemp("r5-qualification")
    native.test_native_e1_to_e4_recovery_parity_divergence_and_receipt(root)
    return json.loads((root / "qualification-receipt.json").read_text(encoding="utf-8"))


@pytest.mark.integration
# PREDICTED: RED at CR (issue 4)
def test_r5_qualification_receipt_records_checkpoint_faults(r5_qualification_receipt):
    faults = r5_qualification_receipt["checkpoint_faults"]
    kinds = {entry.get("kind") for entry in faults}
    assert {"no_checkpoint", "before_commit", "after_commit", "corrupt_committed"} <= kinds, (
        f"observed checkpoint_faults={faults!r}"
    )


@pytest.mark.integration
# PREDICTED: GREEN at CR (issue 4 control)
def test_r5_qualification_receipt_records_scheduler_and_backend_layers(r5_qualification_receipt):
    layers = {entry.get("layer") for entry in r5_qualification_receipt["raw_fault_evidence"]}
    assert {"scheduler", "backend"} <= layers, f"observed raw_fault_evidence layers={layers!r}"


@pytest.mark.integration
# PREDICTED: RED at CR (issue 5)
def test_r5_native_worker_refuses_config_not_bound_to_intent(tmp_path):
    changed = tmp_path / "changed.yaml"
    cfg = OmegaConf.load(native.FIXTURE)
    cfg.optimizer.lr = 0.02
    OmegaConf.save(cfg, changed)
    # Batch and intent are built from the ORIGINAL fixture (optimizer.lr 0.01).
    lane = _Lane(tmp_path / "unbound", config_path=changed)
    process = lane.launch(phase="initial", label="r5-unbound-config")
    returncode = process.wait(timeout=180)
    result = native._result_from_log(lane.root, "r5-unbound-config")
    print(f"UNBOUND_CONFIG returncode={returncode!r} result={result!r}")
    invocation = result.get("invocation")
    assert invocation is None or invocation.get("outcome") != "completed", (
        f"worker completed science with a config not bound to the intent: {result!r}"
    )
