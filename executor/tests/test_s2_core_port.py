"""S2-L1 process-boundary tests (run on Cannon by the executor harness)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId, ExecutionPlanV2, GroupId, ProgramSpecRef, ReadyGroupV2, TaskSpec
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle.core_port import CoreAdmissionProcessRequest, ScienceProcessCoreAdmissionPort
from tpen_executor.recovery import CoreAdmissionUnavailable, MemberRecoveryInputV2
from tpen_executor.results import OutcomeReceipt, StartReceipt
from tpen_executor.workers import ProcessOutcome


MUTANTS = [
    {"id": "cross-task", "target": "executor/src/tpen_executor/lifecycle/core_port.py", "old": 'if request.task_id != member.task_id:', "new": 'if False:', "killer": "test_crossed_task_refused_before_spawn"},
    {"id": "cross-lineage", "target": "executor/src/tpen_executor/lifecycle/core_port.py", "old": 'if request.lineage_id != member.lineage_id:', "new": 'if False:', "killer": "test_crossed_lineage_refused_before_spawn"},
    {"id": "cross-digest", "target": "executor/src/tpen_executor/lifecycle/core_port.py", "old": 'if request.successor_request.get("science_input_digest") != request.science_input_digest:', "new": 'if False:', "killer": "test_crossed_digest_refused_before_spawn"},
    {"id": "timeout", "target": "executor/src/tpen_executor/lifecycle/core_port.py", "old": 'if outcome.timed_out:', "new": 'if False:', "killer": "test_timeout_is_unavailable"},
    {"id": "nonzero", "target": "executor/src/tpen_executor/lifecycle/core_port.py", "old": 'if outcome.returncode != 0:', "new": 'if False:', "killer": "test_nonzero_exit_is_unavailable"},
    {"id": "launch-failure", "target": "executor/src/tpen_executor/lifecycle/core_port.py", "old": 'except (OSError, TypeError, ValueError, RuntimeError) as exc:', "new": 'except TypeError as exc:', "killer": "test_launch_failure_is_unavailable"},
    {"id": "malformed-json", "target": "executor/src/tpen_executor/lifecycle/core_port.py", "old": 'decoded = json.loads(payload)', "new": 'decoded = None', "killer": "test_malformed_json_is_unavailable"},
    {"id": "strict-decoder", "target": "executor/src/tpen_executor/lifecycle/core_port.py", "old": 'return decoder.admit(member)', "new": 'return decoded', "killer": "test_extra_field_is_unavailable"},
]


class _Worker:
    def __init__(self, payload: object, *, returncode: int = 0, timed_out: bool = False) -> None:
        self.payload = payload
        self.returncode = returncode
        self.timed_out = timed_out
        self.calls = 0

    def run(self, _environment, _args, *, stdout_path, stderr_path, **_kwargs):
        self.calls += 1
        Path(stdout_path).write_text(json.dumps(self.payload), encoding="utf-8")
        Path(stderr_path).write_text("", encoding="utf-8")
        return ProcessOutcome((sys.executable, "-c", ""), self.returncode, str(stdout_path), str(stderr_path), self.timed_out)


class _LaunchFailure:
    def run(self, *_args, **_kwargs):
        raise OSError("launch failed")


def _member() -> MemberRecoveryInputV2:
    plan_ref = {"members": ["task"]}
    group = ReadyGroupV2(
        GroupId("s2-group"),
        (TaskSpec("task", "lineage", ProgramSpecRef({"task": "task"}, digest({"task": "task"})), "a" * 64, 1, "steps"),),
        ExecutionPlanV2(plan_ref, digest(plan_ref), ("task",), "profile", "protocol"),
    )
    attempt = AttemptId("attempt")
    start = StartReceipt("task", attempt, AllocationId("allocation"), BackendJobId("job"), group.identity, "2026-11-20T00:00:00Z", "/tmp/s2-core")
    outcome = OutcomeReceipt("task", attempt, group.identity, "2026-11-20T00:01:00Z", "failed", start.output_dir)
    return MemberRecoveryInputV2("task", "lineage", attempt, start, outcome)


def _request(member: MemberRecoveryInputV2, *, successor_request=None, digest_value="a" * 64) -> CoreAdmissionProcessRequest:
    return CoreAdmissionProcessRequest(member.task_id, member.lineage_id, None, digest_value, successor_request, {"study": "s2"}, "profile", PreparedPythonEnvironment(sys.executable))


def _port(tmp_path: Path, worker: _Worker):
    return ScienceProcessCoreAdmissionPort(worker), tmp_path / "out.json", tmp_path / "err.log"


def test_missing_leaf_returns_none(tmp_path: Path) -> None:
    member = _member()
    worker = _Worker(None)
    port, out, err = _port(tmp_path, worker)
    assert port.admit(_request(member), member, stdout_path=out, stderr_path=err) is None


@pytest.mark.parametrize("payload", [{"schema": "tpen-core-admission-v1"}, {"schema": "tpen-core-admission-v1", "extra": 1}, {"schema": "wrong"}])
def test_malformed_or_wrong_wire_is_unavailable(tmp_path: Path, payload: dict[str, object]) -> None:
    member = _member()
    worker = _Worker(payload)
    port, out, err = _port(tmp_path, worker)
    with pytest.raises(CoreAdmissionUnavailable):
        port.admit(_request(member), member, stdout_path=out, stderr_path=err)


def test_extra_field_is_unavailable(tmp_path: Path) -> None:
    """The closed decoder, rather than a duplicate classifier, rejects extras."""

    member = _member()
    payload = {
        "schema": "tpen-core-admission-v1",
        "outcome": "completed",
        "admission": "unknown",
        "checkpoint": "absent",
        "quiescence": "unknown",
        "profile": None,
        "reason": "completed",
        "extra": "reject-me",
    }
    port, out, err = _port(tmp_path, _Worker(payload))
    with pytest.raises(CoreAdmissionUnavailable):
        port.admit(_request(member), member, stdout_path=out, stderr_path=err)


def test_nonzero_exit_is_unavailable(tmp_path: Path) -> None:
    member = _member()
    worker = _Worker(None, returncode=3)
    port, out, err = _port(tmp_path, worker)
    with pytest.raises(CoreAdmissionUnavailable):
        port.admit(_request(member), member, stdout_path=out, stderr_path=err)


def test_timeout_is_unavailable(tmp_path: Path) -> None:
    member = _member()
    worker = _Worker(None, timed_out=True)
    port, out, err = _port(tmp_path, worker)
    with pytest.raises(CoreAdmissionUnavailable):
        port.admit(_request(member), member, stdout_path=out, stderr_path=err)


def test_crossed_task_refused_before_spawn(tmp_path: Path) -> None:
    member = _member()
    successor = {"predecessor": {"task_id": "other"}, "successor": {"task_id": "task"}, "science_input_digest": "a" * 64}
    worker = _Worker(None)
    port, out, err = _port(tmp_path, worker)
    with pytest.raises(CoreAdmissionUnavailable):
        port.admit(_request(member, successor_request=successor), member, stdout_path=out, stderr_path=err)
    assert worker.calls == 0


def test_crossed_digest_refused_before_spawn(tmp_path: Path) -> None:
    member = _member()
    successor = {"predecessor": {"task_id": "task"}, "successor": {"task_id": "task"}, "science_input_digest": "b" * 64}
    worker = _Worker(None)
    port, out, err = _port(tmp_path, worker)
    with pytest.raises(CoreAdmissionUnavailable):
        port.admit(_request(member, successor_request=successor), member, stdout_path=out, stderr_path=err)
    assert worker.calls == 0


def test_crossed_lineage_refused_before_spawn(tmp_path: Path) -> None:
    member = _member()
    worker = _Worker(None)
    port, out, err = _port(tmp_path, worker)
    with pytest.raises(CoreAdmissionUnavailable):
        port.admit(_request(member), member.__class__(member.task_id, "other-lineage", member.attempt_id, member.start_receipt, member.outcome_receipt), stdout_path=out, stderr_path=err)
    assert worker.calls == 0


def test_launch_failure_is_unavailable(tmp_path: Path) -> None:
    member = _member()
    port, out, err = _port(tmp_path, _LaunchFailure())  # type: ignore[arg-type]
    with pytest.raises(CoreAdmissionUnavailable):
        port.admit(_request(member), member, stdout_path=out, stderr_path=err)


def test_malformed_json_is_unavailable(tmp_path: Path) -> None:
    class _Malformed(_Worker):
        def run(self, _environment, _args, *, stdout_path, stderr_path, **_kwargs):
            Path(stdout_path).write_text("not-json", encoding="utf-8")
            Path(stderr_path).write_text("", encoding="utf-8")
            return ProcessOutcome((sys.executable,), 0, str(stdout_path), str(stderr_path), False)

    member = _member()
    port, out, err = _port(tmp_path, _Malformed(None))
    with pytest.raises(CoreAdmissionUnavailable):
        port.admit(_request(member), member, stdout_path=out, stderr_path=err)


def test_control_interpreter_has_no_tpen_import(tmp_path: Path) -> None:
    member = _member()
    port = ScienceProcessCoreAdmissionPort()
    assert port.admit(
        _request(member), member, stdout_path=tmp_path / "out.json", stderr_path=tmp_path / "err.log"
    ) is None
    assert not any(name == "tpen" or name.startswith("tpen.") for name in sys.modules)


def test_mutant_rows_have_unique_source_anchors_and_killers() -> None:
    root = Path(__file__).parents[2]
    source = (root / "executor/src/tpen_executor/lifecycle/core_port.py").read_text()
    test_source = Path(__file__).read_text()
    for row in MUTANTS:
        assert source.count(row["old"]) == 1, row["id"]
        compile(source.replace(row["old"], row["new"], 1), str(root), "exec")
        assert f"def {row['killer']}(" in test_source
