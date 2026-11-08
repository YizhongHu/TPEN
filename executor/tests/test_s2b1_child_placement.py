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
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tpen_executor._codec import digest
from tpen_executor.contracts import AllocationId, BackendJobId, AttemptId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.dispatch.contract import _placement_receipt
from tpen_executor.placement import ChildPlacementObservation, PlacementRefusal, parse_proc_start_ticks
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.results import ManifestAttempt, OutcomeReceipt, SciencePlacementReceipt, StartReceipt
from tpen_executor.runners import RUN_SHIM


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
    receipt = _placement_receipt(observation, allocation, fence, group, group.members[0], configured, environment)
    assert receipt.refusal is not None
    assert receipt.refusal["code"] == "placement-identity-mismatch"


def test_t5_cpu_path_is_explicitly_unavailable_for_gpu(tmp_path: Path) -> None:
    """T5-CPU kills M5: empty-success or fabricated GPU status is forbidden."""

    group, configured, allocation, fence, environment = _receipt_inputs(tmp_path)
    receipt = _placement_receipt(_observation(tmp_path), allocation, fence, group, group.members[0], configured, environment)
    assert receipt.gpu["status"] == "unavailable"
    assert receipt.gpu["reason"]
    assert receipt.gpu["method"]
    assert receipt.gpu["started_at_utc"]
    assert receipt.gpu["finished_at_utc"]
    assert receipt.gpu["samples"] == ()


def test_t6_receipt_and_manifest_reference_round_trip_backward_readable(tmp_path: Path) -> None:
    """T6 kills M6: dropping start token/comparison or breaking old manifests."""

    group, configured, allocation, fence, environment = _receipt_inputs(tmp_path)
    placement = _placement_receipt(_observation(tmp_path), allocation, fence, group, group.members[0], configured, environment)
    start = StartReceipt("task-a", fence.attempt_id, allocation.allocation_id, allocation.backend_job_id, GROUP_DIGEST, "start", fence.output_dir)
    outcome = OutcomeReceipt("task-a", fence.attempt_id, GROUP_DIGEST, "finish", "completed", fence.output_dir)
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
