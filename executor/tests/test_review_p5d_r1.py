"""Independent P5d probes. Run only in the authorized Cannon allocation."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.dispatch import LocalDispatcher
from tpen_executor.dispatch.contract import _evidence_payload
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.programs import AllocationProgram, WorkerProgram
from tpen_executor.runners import CoreRunEvidence, EvidenceUnavailable
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.workers import ProcessOutcome


def group(*ids):
    ref = ProgramSpecRef({}, digest({}))
    return ReadyGroup("review", tuple(TaskSpec(i, i, ref, digest(i), 1, "steps") for i in ids), digest({"review": 1}))


def runs(tmp_path, *ids):
    return {i: ConfiguredRun(i, str(tmp_path / (i + ".yaml"))) for i in ids}


def allocation():
    return ConfiguredAllocation(AllocationId("review-allocation"), BackendJobId("review-job"), {})


class RecordingLifecycle(AttachedAllocation):
    def __init__(self, store):
        super().__init__(allocation(), store)
        object.__setattr__(self, "begun", [])
        object.__setattr__(self, "finished", [])

    def begin(self, group, task_id, started_at_utc=""):
        fence = super().begin(group, task_id, started_at_utc)
        self.begun.append(fence)
        return fence

    def finish(self, fence, status, finished_at_utc, result_digest=None):
        self.finished.append((fence.task_id, status, result_digest))
        return super().finish(fence, status, finished_at_utc, result_digest)


def payload(run_dir, verdict="completed", *, identity=True, create=True):
    if create:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
    value = {
        "schema": "tpen.run-evidence/v2" if identity else "tpen.run-evidence/v1", "run_dir": str(run_dir),
        "verdict": verdict, "run_status": "failed" if verdict == "refused" else verdict,
        "checkpoint": None if verdict == "refused" else {
            "checkpoint_dir": str(Path(run_dir) / "checkpoint"), "schema_version": 2,
            "next_iteration": 3, "completed_updates": 2,
        },
        "refusal": {"code": "missing-checkpoint", "reason": "review"} if verdict == "refused" else None,
    }
    if identity:
        value["resolved_config_identity"] = {"schema": "tpen-config-identity-v1", "sha256": "a" * 64}
    return value


def test_r1_t8_legacy_evidence_digest_omits_absent_identity(tmp_path: Path) -> None:
    """Legacy v1 evidence keeps its pre-identity canonical bytes and digest."""
    legacy = payload("/runs/legacy/core", identity=False, create=False)
    evidence = CoreRunEvidence.from_mapping(legacy)
    assert _evidence_payload(evidence) == legacy
    assert digest(_evidence_payload(evidence)) == "366743525e63dace38629a282b6836bd49538e8fa3e669daa5af631b0c2c8c19"


class ReviewAbort(BaseException):
    pass


@pytest.mark.parametrize("phase", ["materialize", "process", "evidence"])
@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit, ReviewAbort], ids=["keyboard", "system-exit", "base-exception"])
def test_r1_t1_begun_fence_is_finished_before_baseexception_can_escape(tmp_path, phase, error):
    """RED: cancellation may propagate, but cannot skip exactly-one finish."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))

    class Process:
        def run(self, environment, args, **kw):
            if phase == "process":
                raise error("review abort")
            return ProcessOutcome(tuple(args), 0, kw["stdout_path"], kw["stderr_path"], False)

    class Evidence:
        def read(self, *args, **kw):
            if phase == "evidence":
                raise error("review abort")
            return CoreRunEvidence.from_mapping(payload(args[2]))

    def materialize(env):
        if phase == "materialize":
            raise error("review abort")
        return {"environment": object(), "process": Process(), "runner": object(), "evidence": Evidence()}

    propagated = None
    try:
        LocalDispatcher(SimpleNamespace(materialize=materialize)).run(group("a"), runs(tmp_path, "a"), lifecycle)
    except BaseException as exc:
        propagated = type(exc).__name__
    manifest = lifecycle.store.manifest(group("a"))
    assert len(lifecycle.begun) == 1
    assert len(lifecycle.finished) == 1, ("begun fence skipped finish", phase, propagated, lifecycle.finished)
    assert manifest.attempts[0].outcome_receipt is not None


@pytest.mark.parametrize("phase", ["materialize", "process", "evidence"])
def test_r1_t2_ordinary_member_exception_finishes_once_and_later_members_run(tmp_path, phase):
    """GREEN: one shared fault counter; new worker objects for every member."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    materializations = []
    launches = []

    def materialize(env):
        index = len(materializations)
        materializations.append(index)
        if phase == "materialize" and index == 1:
            raise RuntimeError("one materialization failure")

        class Process:
            def run(self, environment, args, **kw):
                launches.append(index)
                if phase == "process" and index == 1:
                    raise RuntimeError("one process failure")
                return ProcessOutcome(tuple(args), 0, kw["stdout_path"], kw["stderr_path"], False)

        class Evidence:
            def read(self, *args, **kw):
                if phase == "evidence" and index == 1:
                    raise RuntimeError("one evidence failure")
                return CoreRunEvidence.from_mapping(payload(args[2]))

        return {"environment": object(), "process": Process(), "runner": object(), "evidence": Evidence()}

    manifest = LocalDispatcher(SimpleNamespace(materialize=materialize)).run(group("a", "b", "c"), runs(tmp_path, "a", "b", "c"), lifecycle)
    assert materializations == [0, 1, 2]
    assert launches == ([0, 2] if phase == "materialize" else [0, 1, 2])
    assert [x[0] for x in lifecycle.finished] == ["a", "b", "c"]
    assert [x.outcome_receipt.status for x in manifest.attempts] == ["completed", "launch_failed", "completed"]


def test_r1_t3_begin_failure_does_not_abandon_later_independent_members(tmp_path):
    """RED: a per-member begin refusal must not prevent later valid members."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    checked = []
    original = lifecycle.begin

    def begin(g, task_id, started_at_utc):
        checked.append(task_id)
        if task_id == "b":
            raise OSError("injected member-local begin failure")
        return original(g, task_id, started_at_utc)

    worker = normal_worker()
    observed_error = None
    try:
        LocalDispatcher(worker).run(group("a", "b", "c"), runs(tmp_path, "a", "b", "c"), SimpleNamespace(begin=begin, finish=lifecycle.finish, store=lifecycle.store))
    except OSError as exc:
        observed_error = str(exc)
    assert checked == ["a", "b", "c"], (checked, observed_error)
    assert [x[0] for x in lifecycle.finished] == ["a", "c"]


def normal_worker(verdict="completed", code=0, timed_out=False, unavailable=False):
    class Process:
        def run(self, env, args, **kw):
            return ProcessOutcome(tuple(args), code, kw["stdout_path"], kw["stderr_path"], timed_out)

    class Evidence:
        def read(self, *args, **kw):
            if unavailable:
                raise EvidenceUnavailable("review unavailable")
            return CoreRunEvidence.from_mapping(payload(args[2], verdict, identity=verdict != "refused"))

    return SimpleNamespace(materialize=lambda env: {"environment": object(), "process": Process(), "runner": object(), "evidence": Evidence()})


@pytest.mark.parametrize("code,timed_out,verdict,unavailable,status", [
    (0, False, "refused", False, "completed"),
    (3, False, "completed", False, "failed"),
    (-9, True, "completed", False, "timed_out"),
    (0, False, "completed", True, "completed"),
    (3, False, "completed", True, "failed"),
    (-9, True, "completed", True, "timed_out"),
], ids=["rc0-refused", "rc3-completed", "timeout-completed", "rc0-unavailable", "rc3-unavailable", "timeout-unavailable"])
def test_r1_t4_operational_status_is_independent_and_digest_is_exact(tmp_path, code, timed_out, verdict, unavailable, status):
    """GREEN: assert the canonical digest, not merely that it is non-null."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    manifest = LocalDispatcher(normal_worker(verdict, code, timed_out, unavailable)).run(group("a"), runs(tmp_path, "a"), lifecycle)
    record = manifest.attempts[0]
    assert record.outcome_receipt.status == status
    expected = None if unavailable else digest(payload(Path(record.start_receipt.output_dir) / "core", verdict, identity=verdict != "refused"))
    assert record.outcome_receipt.result_digest == expected
    assert len(lifecycle.finished) == 1


def test_r1_t5_launch_oserror_finishes_once_without_reading_evidence(tmp_path):
    """GREEN: a failed launch is distinct from a reaped child failure."""
    evidence_calls = []

    class Process:
        def run(self, *args, **kw):
            raise FileNotFoundError("missing prepared interpreter")

    class Evidence:
        def read(self, *args, **kw):
            evidence_calls.append(args)
            raise AssertionError("not launched")

    worker = SimpleNamespace(materialize=lambda env: {"environment": object(), "process": Process(), "runner": object(), "evidence": Evidence()})
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    manifest = LocalDispatcher(worker).run(group("a"), runs(tmp_path, "a"), lifecycle)
    assert evidence_calls == []
    assert lifecycle.finished == [("a", "launch_failed", None)]
    assert manifest.attempts[0].outcome_receipt.status == "launch_failed"


def test_r1_t6_finish_exception_does_not_repeat_finish_or_stop_later_members(tmp_path):
    """GREEN: preserve first finish failure after processing other members."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    original = lifecycle.finish

    def finish(*args):
        result = original(*args)
        if args[0].task_id == "a":
            raise RuntimeError("post-publication finish failure")
        return result

    with pytest.raises(RuntimeError, match="post-publication"):
        LocalDispatcher(normal_worker()).run(group("a", "b"), runs(tmp_path, "a", "b"), SimpleNamespace(begin=lifecycle.begin, finish=finish, store=lifecycle.store))
    assert [x[0] for x in lifecycle.finished] == ["a", "b"]
    assert all(x.outcome_receipt is not None for x in lifecycle.store.manifest(group("a", "b")).attempts)


def science_worker(extra_env=None):
    python = os.environ["P5B_REVIEW_SCIENCE_PYTHON"]
    raw = {
        "_target_": "tpen_executor.programs.WorkerProgram", "_recursive_": False,
        "environment": {"_target_": "tpen_executor.environments.PreparedPythonEnvironment", "_recursive_": False, "python": python, "env": extra_env or {}},
        "process": {"_target_": "tpen_executor.workers.SubprocessWorker", "_recursive_": False},
        "runner": {"_target_": "tpen_executor.runners.TPENCommand", "_recursive_": False, "config": "/unused/runner.yaml"},
        "evidence": {"_target_": "tpen_executor.runners.TPENEvidence", "_recursive_": False},
    }
    return WorkerProgram(ProgramSpecRef(raw, digest(raw)))


def real_core_config(tmp_path):
    config = tmp_path / "core.yaml"
    # The base Runner deliberately raises NotImplementedError after creating
    # its context. This is a path/transport probe, with expected failed rc,
    # and makes no scientific-completion or training claim.
    config.write_text("run: {root: /unused, layout: nested, run_id: configured}\nexperiment: {name: review}\nrunner: {_target_: tpen.runner.Runner}\n", encoding="utf-8")
    return config


def test_r1_t7_real_core_pinned_overrides_win_and_evidence_names_actual_run(tmp_path):
    """GREEN: real core dotlist merge, ArtifactManager and inspection route."""
    config = real_core_config(tmp_path)
    unwanted = tmp_path / "unwanted"
    configured = ConfiguredRun("a", str(config), (f"run.root={unwanted}", "run.layout=nested", "run.run_id=unwanted"))
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    manifest = LocalDispatcher(science_worker()).run(group("a"), {"a": configured}, lifecycle, timeout=30)
    output = Path(lifecycle.begun[0].output_dir)
    evidence = json.loads((output / "evidence.stdout.log").read_text())
    assert (output / "core").is_dir(), (output / "stderr.log").read_text()
    assert not unwanted.exists()
    assert evidence["run_dir"] == str(output / "core")
    assert evidence["refusal"]["code"] != "missing-run-dir" if evidence["refusal"] else True
    assert manifest.attempts[0].outcome_receipt.status == "failed"
    assert manifest.attempts[0].outcome_receipt.result_digest == digest(evidence)


def test_r1_t8_store_path_interpolation_cannot_redirect_core_outside_fence(tmp_path):
    """RED: normal POSIX path syntax must not become a core interpolation."""
    config = real_core_config(tmp_path)
    root = tmp_path / "${oc.env:P5D_REVIEW_PATH}"
    lifecycle = RecordingLifecycle(DirectoryStore(root))
    error = None
    try:
        LocalDispatcher(science_worker({"P5D_REVIEW_PATH": "redirected"})).run(group("a"), {"a": ConfiguredRun("a", str(config))}, lifecycle, timeout=30)
    except (TypeError, ValueError) as exc:
        error = str(exc)
    redirected = tmp_path / "redirected"
    assert not redirected.exists(), ("core wrote outside literal store fence", str(root), str(redirected), sorted(str(p.relative_to(redirected)) for p in redirected.rglob("*")))
    if error is not None:
        assert not root.exists(), "path syntax refusal must happen before begin"
    else:
        output = Path(lifecycle.begun[0].output_dir)
        assert (output / "core").is_dir(), (output / "stderr.log").read_text()


@pytest.mark.parametrize("value", ["scalar", 7, None])
def test_r1_t9_scalar_admission_uses_contract_valueerror_before_any_begin(tmp_path, value):
    """RED: contract B names ValueError for scalar run inputs."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    with pytest.raises(ValueError):
        LocalDispatcher(normal_worker()).run(group("a"), value, lifecycle)
    assert lifecycle.begun == []
    assert not (tmp_path / "store").exists()


def test_r1_t10_nan_timeout_is_refused_before_artifacts(tmp_path):
    """RED: NaN does not satisfy a positive timeout contract."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    with pytest.raises(ValueError):
        LocalDispatcher(normal_worker()).run(group("a"), runs(tmp_path, "a"), lifecycle, timeout=float("nan"))
    assert not (tmp_path / "store").exists()


def test_r1_t11_parent_write_trace_includes_log_opens_and_successful_members(tmp_path, monkeypatch):
    """GREEN: no swallowed trap failure can stand in for successful dispatch."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    config = real_core_config(tmp_path)
    writes = []
    original = io.open

    def traced(path, mode="r", *args, **kw):
        if isinstance(mode, str) and any(c in mode for c in "wax+") and not isinstance(path, int):
            writes.append(os.path.abspath(os.fspath(path)))
        return original(path, mode, *args, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(io, "open", traced)
        manifest = LocalDispatcher(science_worker()).run(group("a", "b"), {i: ConfiguredRun(i, str(config)) for i in ("a", "b")}, lifecycle, timeout=30)
    assert [x[0] for x in lifecycle.finished] == ["a", "b"]
    assert [x.outcome_receipt.status for x in manifest.attempts] == ["failed", "failed"]
    assert all(x.outcome_receipt.result_digest is not None for x in manifest.attempts)
    assert {Path(p).name for p in writes} >= {"stdout.log", "stderr.log", "evidence.stdout.log", "evidence.stderr.log"}
    assert all(Path(p).is_relative_to(Path(lifecycle.store.root)) for p in writes)
