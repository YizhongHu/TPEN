"""S2b-1 child-generation and CPU-provable placement contract tests.

Each test names the smallest mutant that would make its assertion disappear.
The packet is deliberately executor-only: no TPEN core or experiment import is
needed to inspect the child-side contract.

| test | mutant | why |
| --- | --- | --- |
| T1 | M1: compare PID only | final-`)` parsing requires the generation start token |
| T2 | M2: omit child `sched_setaffinity` | ordering proves affinity is applied before core import |
| T3 | M3: accept another attempt/task identity | every fence and payload identity is cross-checked |
| T5-CPU | M5: empty-success or fabricated GPU | CPU path stays explicitly unavailable with provenance |
| T6 | M6: drop start token/comparison or old read | strict receipt and legacy manifest decoding both remain required |
| F1 | M1a: reuse placement.json for the child sidecar | sidecar and controller receipt remain distinct and readable |
| F2 | M2a: enforce stale placement fields without a placement request | legacy no-placement children retain their behavior |
| F3 | M3a: omit placement overlay in the Parsl pool | every packed child emits the same sidecar/receipt contract |
| F4 | M4: trust child-reported PID/start ticks | controller binding requires the spawned generation token |
| F7 | M7: read thread settings only from the legacy observation allowlist | all declared thread settings are recorded truthfully |
| F8 | M8: hide child placement refusal | an affinity refusal is visible before core import |
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

from tpen_executor._codec import digest
from tpen_executor.contracts import AllocationId, BackendJobId, AttemptId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.dispatch.contract import _placement_receipt
from tpen_executor.placement import ChildPlacementObservation, PlacementRefusal, parse_proc_start_ticks, read_child_placement
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.results import ManifestAttempt, OutcomeReceipt, SciencePlacementReceipt, StartReceipt
from tpen_executor.runners import RUN_SHIM
from tpen_executor.storage import DirectoryStore
from tpen_executor.workers import ProcessOutcome, SubprocessWorker


GROUP_DIGEST = digest({"group": "s2b-1"})


def _group() -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    return ReadyGroup("s2b-1", (TaskSpec("task-a", "task-a", program, digest("task-a"), 1, "steps"),), GROUP_DIGEST)


def _configured(tmp_path: Path):
    from tpen_executor.tasks import ConfiguredRun

    config = tmp_path / "config.yaml"
    config.write_text("mode: test\n", encoding="utf-8")
    return ConfiguredRun("task-a", str(config), ("payload.seed=1",))


def _observation(tmp_path: Path, **changes: object) -> ChildPlacementObservation:
    payload = digest({"config": str(tmp_path / "config.yaml"), "overrides": ["payload.seed=1"]})
    values: dict[str, object] = {
        "task_id": "task-a",
        "attempt_id": "attempt-a",
        "group_digest": GROUP_DIGEST,
        "allocation_id": "allocation-a",
        "backend_job_id": "job-a",
        "slot_id": "slot-a",
        "payload_digest": payload,
        "output_dir": str(tmp_path / "output"),
        "host": "node-a",
        "pid": 1234,
        "proc_start_ticks": 55,
        "affinity": (3, 5),
        "thread_settings": {"OMP_NUM_THREADS": "1"},
        "cuda_visible_devices": None,
        "observed_at_utc": "2026-11-08T00:00:00+00:00",
        "refusal": None,
    }
    values.update(changes)
    return ChildPlacementObservation(**values)


def _receipt_inputs(tmp_path: Path):
    group = _group()
    configured = _configured(tmp_path)
    allocation = ConfiguredAllocation(AllocationId("allocation-a"), BackendJobId("job-a"), {})
    fence = SimpleNamespace(
        attempt_id=AttemptId("attempt-a"),
        output_dir=str(tmp_path / "output"),
    )
    environment = PreparedPythonEnvironment("/usr/bin/python3", None, {
        "TPEN_SLOT_ID": "slot-a",
        "TPEN_SLOT_NODE": "node-a",
        "TPEN_SLOT_CPUS": "3,5",
        "TPEN_PLACEMENT_DECLARED_DEVICES": "",
    })
    return group, configured, allocation, fence, environment


def test_t1_proc_parser_requires_start_ticks_not_pid_only() -> None:
    """T1 kills M1: comparing PID only accepts a reused process identity."""

    stat = "123 (worker has ) punctuation) S " + " ".join(str(index) for index in range(4, 24))
    assert parse_proc_start_ticks(123, stat) == 22
    with pytest.raises(PlacementRefusal, match="start ticks"):
        parse_proc_start_ticks(123, "123 (worker) S 1 2")


def test_t2_shim_affinity_is_before_core_import() -> None:
    """T2 kills M2: omitting child-only sched_setaffinity or sampling the controller."""

    set_position = RUN_SHIM.index("os.sched_setaffinity")
    import_position = RUN_SHIM.index("from tpen.run import main")
    assert set_position < import_position
    assert "os.sched_getaffinity(0)" in RUN_SHIM


def test_t3_receipt_rejects_cross_attempt_identity(tmp_path: Path) -> None:
    """T3 kills M3: accepting task-index, worker-PID, or another attempt."""

    group, configured, allocation, fence, environment = _receipt_inputs(tmp_path)
    observation = _observation(tmp_path, attempt_id="other-attempt")
    receipt = _placement_receipt(observation, allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((sys.executable,), 0, "", "", False, 1234, 55))
    assert receipt.refusal is not None
    assert receipt.refusal["code"] == "placement-identity-mismatch"


def test_t5_cpu_path_is_explicitly_unavailable_for_gpu(tmp_path: Path) -> None:
    """T5-CPU kills M5: empty-success or fabricated GPU status is forbidden."""

    group, configured, allocation, fence, environment = _receipt_inputs(tmp_path)
    receipt = _placement_receipt(_observation(tmp_path, group_digest=group.identity), allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((sys.executable,), 0, "", "", False, 1234, 55))
    assert receipt.gpu["status"] == "unavailable"
    assert receipt.gpu["reason"]
    assert receipt.gpu["method"]
    assert receipt.gpu["started_at_utc"]
    assert receipt.gpu["finished_at_utc"]
    assert receipt.gpu["samples"] == ()


def test_t6_receipt_and_manifest_reference_round_trip_backward_readable(tmp_path: Path) -> None:
    """T6 kills M6: dropping start token/comparison or breaking old manifests."""

    group, configured, allocation, fence, environment = _receipt_inputs(tmp_path)
    placement = _placement_receipt(_observation(tmp_path, group_digest=group.identity), allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((sys.executable,), 0, "", "", False, 1234, 55))
    start = StartReceipt("task-a", fence.attempt_id, allocation.allocation_id, allocation.backend_job_id, group.identity, "start", fence.output_dir)
    outcome = OutcomeReceipt("task-a", fence.attempt_id, group.identity, "finish", "completed", fence.output_dir)
    manifest_attempt = ManifestAttempt("task-a", fence.attempt_id, start, outcome, fence.output_dir, placement)
    decoded = ManifestAttempt.from_mapping(manifest_attempt.to_dict())
    assert decoded.placement_receipt == placement
    assert decoded.placement_receipt.child["proc_start_ticks"] == 55
    old = ManifestAttempt.from_mapping({
        "schema": "tpen-executor/manifest-attempt-v1",
        "task_id": "task-a",
        "attempt_id": fence.attempt_id.value,
        "start_receipt": start.to_dict(),
        "outcome_receipt": outcome.to_dict(),
        "output_dir": fence.output_dir,
    })
    assert old.placement_receipt is None


def test_f1_child_sidecar_and_controller_receipt_are_distinct(tmp_path: Path) -> None:
    """F1 kills M1a: the child sidecar cannot occupy the controller receipt path."""

    group = _group()
    allocation = ConfiguredAllocation(AllocationId("allocation-a"), BackendJobId("job-a"), {})
    store = DirectoryStore(tmp_path / "store")
    fence = store.begin_attempt(group, "task-a", allocation, "start")
    configured = _configured(tmp_path)
    environment = PreparedPythonEnvironment(sys.executable, None, {
        "TPEN_SLOT_ID": "slot-a",
        "TPEN_SLOT_NODE": "node-a",
        "TPEN_SLOT_CPUS": "3,5",
        "TPEN_PLACEMENT_DECLARED_DEVICES": "",
    })
    sidecar = _observation(tmp_path, group_digest=group.identity, output_dir=fence.output_dir)
    sidecar_path = Path(fence.output_dir, "child-placement.json")
    sidecar_path.write_text(json.dumps(sidecar.to_dict()) + "\n", encoding="utf-8")
    store.finish_attempt(fence, "completed", "finish")
    receipt = _placement_receipt(
        read_child_placement(sidecar_path),
        allocation,
        fence,
        group,
        group.members[0],
        configured,
        environment,
        ProcessOutcome((sys.executable,), 0, "", "", False, 1234, 55),
    )
    store.publish_placement(fence, receipt)
    manifest = store.manifest(group)
    assert manifest.attempts[0].placement_receipt == receipt
    assert Path(fence.output_dir, "placement.json").is_file()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux child placement behavior")
def test_f2_no_placement_request_preserves_legacy_child_behavior(tmp_path: Path) -> None:
    """F2 kills M2a: stale placement fields do not enforce without a request."""

    package = tmp_path / "pkg" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text("def main(argv):\n    return 0\n", encoding="utf-8")
    environment = PreparedPythonEnvironment(sys.executable, None, {
        "PYTHONPATH": str(package.parent),
        "TPEN_PLACEMENT_DECLARED_HOST": "not-" + socket.gethostname(),
    })
    outcome = SubprocessWorker().run(
        environment,
        ("-c", RUN_SHIM, f"run.root={tmp_path}"),
        cwd=tmp_path,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
    )
    assert outcome.returncode == 0
    assert (tmp_path / "err.log").read_text(encoding="utf-8") == ""


def test_f3_parsl_pool_applies_child_placement_overlay() -> None:
    """F3 kills M3a: packed Parsl children use the placement path."""

    source = Path(__import__("tpen_executor.dispatch.parsl", fromlist=["x"]).__file__).read_text(encoding="utf-8")
    launch = source[source.index("def launch(member"):]
    assert "_with_child_placement" in launch[: launch.index("_submit_process")]


def test_f4_unbound_generation_token_is_refused(tmp_path: Path) -> None:
    """F4 kills M4: sidecar PID/start ticks must come from the spawned child."""

    group, configured, allocation, fence, environment = _receipt_inputs(tmp_path)
    forged = _observation(tmp_path, group_digest=group.identity, pid=1, proc_start_ticks=1)
    receipt = _placement_receipt(forged, allocation, fence, group, group.members[0], configured, environment)
    assert receipt.refusal is not None
    assert receipt.refusal["code"] == "placement-generation-mismatch"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux child placement behavior")
def test_f7_shim_records_openblas_from_child_environment(tmp_path: Path) -> None:
    """F7 kills M7: thread settings are sampled from the actual child env."""

    package = tmp_path / "pkg" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text("def main(argv):\n    return 0\n", encoding="utf-8")
    sidecar = tmp_path / "child-placement.json"
    cpus = ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
    environment = PreparedPythonEnvironment(sys.executable, None, {
        "PYTHONPATH": str(package.parent),
        "OPENBLAS_NUM_THREADS": "3",
        "TPEN_PLACEMENT_OUTPUT": str(sidecar),
        "TPEN_PLACEMENT_DECLARED_CPUS": cpus,
        "TPEN_PLACEMENT_DECLARED_HOST": socket.gethostname(),
    })
    outcome = SubprocessWorker().run(
        environment,
        ("-c", RUN_SHIM, f"run.root={tmp_path}"),
        cwd=tmp_path,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
    )
    assert outcome.returncode == 0
    observed = json.loads(sidecar.read_text(encoding="utf-8"))
    assert observed["refusal"] is None
    assert observed["thread_settings"]["OPENBLAS_NUM_THREADS"] == "3"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux child placement behavior")
def test_f8_shim_exposes_affinity_refusal(tmp_path: Path) -> None:
    """F8 kills M8: an unavailable affinity is a visible typed refusal."""

    package = tmp_path / "pkg" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text("def main(argv):\n    return 0\n", encoding="utf-8")
    sidecar = tmp_path / "child-placement.json"
    unavailable_cpu = max(os.sched_getaffinity(0)) + 1
    environment = PreparedPythonEnvironment(sys.executable, None, {
        "PYTHONPATH": str(package.parent),
        "TPEN_PLACEMENT_OUTPUT": str(sidecar),
        "TPEN_PLACEMENT_DECLARED_CPUS": str(unavailable_cpu),
        "TPEN_PLACEMENT_DECLARED_HOST": socket.gethostname(),
    })
    outcome = SubprocessWorker().run(
        environment,
        ("-c", RUN_SHIM, f"run.root={tmp_path}"),
        cwd=tmp_path,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
    )
    assert outcome.returncode == 2
    observed = json.loads(sidecar.read_text(encoding="utf-8"))
    assert observed["refusal"]["code"] == "child-placement-refused"
    assert b"child-placement-refused" in (tmp_path / "err.log").read_bytes()
