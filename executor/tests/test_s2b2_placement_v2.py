"""S2b-2 L1 executable mutant killers.

The suite is intentionally executor-only and covers the trimmed L1a contract. GPU sampler production and sampler mutants belong to L1b.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tpen_executor._codec import digest
from tpen_executor.contracts import AllocationId, BackendJobId, AttemptId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.cli import _operational_exit
from tpen_executor.dispatch.contract import _placement_receipt_v2, _with_child_placement
from tpen_executor.dispatch.local import LocalDispatcher
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.placement import (
    AllocationSlot,
    ChildPlacementObservation,
    PlacementRefusal,
    expand_slurm_nodelist,
    physical_slots_from_allocation,
    scheduler_host_map,
)
from tpen_executor.results import AcceptedSegment, ManifestAttempt, OutcomeReceipt, RunManifest, SciencePlacementReceiptV2, StartReceipt
from tpen_executor.runners import CoreRunEvidence
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.workers import ProcessOutcome

GPU_A = "GPU-12345678-1234-1234-1234-123456789ABC"
GPU_B = "GPU-12345678-1234-1234-1234-123456789ABD"
GPU_C = "GPU-12345678-1234-1234-1234-123456789ABE"
PBS_NODEFILE_TEXT = "physical-01\n"


L1A_MUTANT_TABLE = (
    {
        "id": "M1", "target": "executor/src/tpen_executor/results.py",
        "old": '        if not isinstance(self.verdict, Mapping) or set(self.verdict) != expected_verdict:',
        "new": "        if not isinstance(self.verdict, Mapping):",
        "killer": "test_l1a_v2_wire_is_closed_and_v1_remains_readable",
        "suite": "executor", "assertion": "unexpected verdict keys are rejected",
    },
    {
        "id": "M2", "target": "executor/src/tpen_executor/allocations.py",
        "old": "                    hosts = ()",
        "new": "                    hosts = scheduler_host_map(scheduler, environ={})",
        "killer": "test_d6_unreadable_pbs_nodefile_is_typed_launch_failure",
        "suite": "executor", "assertion": "unreadable PBS nodefile becomes launch_failed rc=3",
    },
    {
        "id": "M3", "target": "executor/src/tpen_executor/results.py",
        "old": '            if receipt.verdict["reason_class"] == "not-sampled":',
        "new": "            if False:",
        "killer": "test_c2_not_sampled_is_loud_but_not_science_blocking",
        "suite": "executor", "assertion": "manifest retains the not-sampled flag",
    },
    {
        "id": "M4", "target": "executor/src/tpen_executor/runners.py",
        "old": '    "raise SystemExit(main(sys.argv[1:]))"',
        "new": '    "raise SystemExit(2)"',
        "killer": "test_d1_run_shim_without_placement_runs_real_core_to_completion",
        "suite": "executor", "assertion": "an unplaced child reaches core and completes",
    },
    {
        "id": "M5", "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": '            status == "completed"',
        "new": '            status == "failed"',
        "killer": "test_m5_real_local_dispatch_preserves_segment_and_rc_zero",
        "suite": "executor", "assertion": "advisory mismatch still accepts the completed segment",
    },
    {
        "id": "N1", "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": "    if type(reason) is not str or not reason:", "new": "    if False:",
        "killer": "test_malformed_child_refusal_reason_finishes_typed_launch_failure",
        "suite": "executor", "assertion": "empty refusal reason cannot strand the begun attempt",
    },
    {
        "id": "N2", "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": '        and base.refusal.get("code") in {"placement-affinity-mismatch", "child-placement-mismatch", "placement-generation-mismatch"}',
        "new": '        and base.refusal.get("code") in {"placement-affinity-mismatch", "child-placement-mismatch"}',
        "killer": "test_generation_mismatch_preserves_physical_evidence",
        "suite": "executor", "assertion": "generation mismatch remains a typed mismatch with matched host evidence",
    },
    {
        "id": "F3", "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": '        comparison_affinity = "matched" if affinity_match else ("unavailable" if not observation.affinity else "mismatch")',
        "new": '        comparison_affinity = "unavailable"',
        "killer": "test_late_proc_refusal_preserves_observed_affinity_and_cause",
        "suite": "executor", "assertion": "late /proc refusal retains matched affinity and child cause",
    },
)


def _group() -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    member = TaskSpec("task-a", "task-a", program, digest("task-a"), 1, "steps")
    return ReadyGroup("s2b2", (member,), digest("s2b2"))


def _allocation(*, scheduler: object | None = None, device: str | tuple[str, ...] = GPU_A) -> ConfiguredAllocation:
    allowed = sorted(getattr(os, "sched_getaffinity", lambda _pid: (0,))(0))
    devices = (device,) if isinstance(device, str) else device
    slot = AllocationSlot("logical-slot", "logical-label", devices, (allowed[0],), (), ()).to_dict()
    facts: dict[str, object] = {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot]}}
    if scheduler is not None:
        facts["scheduler"] = scheduler
    return ConfiguredAllocation(AllocationId("allocation-a"), BackendJobId("job-a"), facts)


def test_m1_scheduler_map_replaces_logical_label_with_physical_node() -> None:
    """M1: accepting ``slot_id`` as host is killed by the PBS map."""

    slots = physical_slots_from_allocation(_allocation(scheduler={"PBS_NODEFILE": "physical-01\n"}))
    assert slots[0].slot_id == "logical-slot"
    assert slots[0].node == "physical-01"
    assert physical_slots_from_allocation(_allocation(scheduler={"SLURM_NODELIST": "nid[001-001]"}))[0].node == "nid001"


def test_r8_scheduler_path_is_captured_before_allocation_freezes(tmp_path: Path) -> None:
    """R8: changing a nodefile cannot change an existing allocation map."""

    nodefile = tmp_path / "PBS_NODEFILE"
    nodefile.write_text("physical-01\n", encoding="utf-8")
    allocation = _allocation(scheduler={"pbs_nodefile_path": str(nodefile)})
    encoded = allocation.encode()
    nodefile.write_text("physical-02\n", encoding="utf-8")
    assert physical_slots_from_allocation(allocation)[0].node == "physical-01"
    assert b"physical-01" in encoded and b"physical-02" not in encoded
    assert scheduler_host_map({"pbs_nodefile_path": str(nodefile)}, environ={}) == ("physical-02",)


def test_r12_slurm_expansion_bounds_each_bracket_token() -> None:
    """R12: cumulative range budget is checked before list expansion."""

    with pytest.raises(PlacementRefusal, match="bounded host limit"):
        expand_slurm_nodelist("nid[0001-1024,0001-1024]")


def _v2_inputs(tmp_path: Path, declared_devices: tuple[str, ...] = (GPU_A,)):
    group = _group()
    configured = ConfiguredRun("task-a", str(tmp_path / "config.yaml"), ("payload.seed=1",))
    (tmp_path / "config.yaml").write_text("ok", encoding="utf-8")
    allocation = _allocation(scheduler={"PBS_NODEFILE": PBS_NODEFILE_TEXT}, device=declared_devices)
    fence = SimpleNamespace(attempt_id=AttemptId("attempt-a"), output_dir=str(tmp_path / "out"))
    allowed = sorted(getattr(os, "sched_getaffinity", lambda _pid: (0,))(0))
    environment = PreparedPythonEnvironment(sys.executable, None, {
        "TPEN_SLOT_ID": "logical-slot",
        "TPEN_SLOT_NODE": "physical-01",
        "TPEN_SLOT_CPUS": str(allowed[0]),
        "TPEN_PLACEMENT_DECLARED_DEVICES": ",".join(declared_devices),
        "TPEN_PLACEMENT_RECEIPT_VERSION": "2",
        "TPEN_PLACEMENT_HOST_STATUS": "mapped",
    })
    observation = ChildPlacementObservation(
        "task-a", "attempt-a", group.identity, "allocation-a", "job-a", "logical-slot",
        digest({"config": configured.config, "overrides": list(configured.overrides)}), fence.output_dir,
        "physical-01", 1234, 55, (allowed[0],), {}, None, "2026-11-10T00:00:00Z", None,
    )
    return group, configured, allocation, fence, environment, observation


def _receipt(tmp_path: Path, declared_devices: tuple[str, ...] = (GPU_A,)):
    group, configured, allocation, fence, environment, observation = _v2_inputs(tmp_path, declared_devices)
    return _placement_receipt_v2(
        observation, allocation, fence, group, group.members[0], configured, environment,
        ProcessOutcome((), 0, "", "", False, 1234, 55),
    ), group, configured, allocation, fence, environment


def test_l1a_v2_wire_is_closed_and_v1_remains_readable(tmp_path: Path) -> None:
    """The v2 envelope is closed while its nested v1 record remains immutable."""

    receipt, *_ = _receipt(tmp_path)
    raw = receipt.to_dict()
    assert set(raw) == {
        "schema", "task_id", "attempt_id", "group_digest", "allocation_id", "backend_job_id",
        "slot_id", "payload_digest", "output_dir", "child", "declared", "gpu", "comparison",
        "refusal", "host_mapping", "verdict",
    }
    assert receipt.base.schema == "tpen-executor/science-placement-receipt-v1"
    raw["verdict"]["unexpected"] = "tamper"
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV2.from_mapping(raw)
    raw = receipt.to_dict()
    raw["unexpected"] = "tamper"
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV2.from_mapping(raw)


def test_c1_receipt_reason_distinguishes_not_sampled_mismatch_and_refused(tmp_path: Path) -> None:
    """L1a keeps not-sampled distinct from mismatch and prelaunch refusal."""

    receipt, group, configured, allocation, fence, environment = _receipt(tmp_path)
    assert receipt.verdict["reason_class"] == "not-sampled"
    assert receipt.gpu["reason"] == "not-sampled"
    wrong = ChildPlacementObservation(
        "task-a", receipt.attempt_id.value, group.identity, allocation.allocation_id.value,
        allocation.backend_job_id.value, receipt.slot_id, receipt.payload_digest, fence.output_dir,
        "other-node", 1234, 55, receipt.child["affinity"], {}, None,
        "2026-11-10T00:00:00Z", None,
    )
    mismatch = _placement_receipt_v2(
        wrong, allocation, fence, group, group.members[0], configured, environment,
        ProcessOutcome((), 0, "", "", False, 1234, 55),
    )
    assert mismatch.verdict["reason_class"] == "mismatch"
    assert mismatch.verdict["science_acceptance"] == "not-accepted"
    assert mismatch.verdict["reason_class"] != "not-sampled"


def test_c2_not_sampled_is_loud_but_not_science_blocking(tmp_path: Path) -> None:
    """The L1a device reason is loud in both receipt and manifest, not blocking."""

    receipt, group, _configured, allocation, fence, _environment = _receipt(tmp_path)
    assert receipt.verdict["device"] == "unavailable"
    assert receipt.verdict["qualification"] == "unqualified"
    assert receipt.verdict["science_acceptance"] == "accepted"
    assert "NOT-SAMPLED" in receipt.verdict["warning"]
    start = StartReceipt("task-a", fence.attempt_id, allocation.allocation_id, allocation.backend_job_id, group.identity, "2026-11-10T00:00:00Z", fence.output_dir)
    outcome = OutcomeReceipt("task-a", fence.attempt_id, group.identity, "2026-11-10T00:00:01Z", "completed", fence.output_dir)
    attempt = ManifestAttempt("task-a", fence.attempt_id, start, outcome, fence.output_dir, receipt)
    segment = AcceptedSegment("task-a", fence.attempt_id, None, str(tmp_path / "checkpoint"), 1, 1)
    manifest = RunManifest(group.group_id, group.identity, (attempt,), (segment,))
    assert manifest.segments == (segment,)
    assert manifest.placement_verdict["status"] == "not-sampled"
    assert manifest.placement_verdict["science_acceptance"] == "accepted"
    assert "NOT-SAMPLED" in manifest.placement_verdict["message"]
    assert RunManifest.decode(manifest.encode()).placement_verdict["status"] == "not-sampled"


def test_c3_logical_slot_name_does_not_drive_physical_verdict(tmp_path: Path) -> None:
    """C3: a mapped physical host matches even when slot_id is logical."""

    receipt, *_ = _receipt(tmp_path)
    assert receipt.slot_id == "logical-slot"
    assert receipt.declared["host"] == "physical-01"
    assert receipt.child["host"] == "physical-01"
    assert receipt.verdict["host"] == "matched"
    assert receipt.verdict["device"] == "unavailable"
    assert receipt.verdict["reason_class"] == "not-sampled"


def test_decoder_closes_host_and_affinity_contradictions(tmp_path: Path) -> None:
    """Persisted v2 records cannot contradict physical host/affinity evidence."""

    receipt, *_ = _receipt(tmp_path)
    raw = receipt.to_dict()
    raw["host_mapping"]["host"] = "other-physical"
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV2.from_mapping(raw)
    raw = receipt.to_dict()
    raw["child"]["affinity"] = [999999]
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV2.from_mapping(raw)


def test_decoder_derives_verdict_from_evidence(tmp_path: Path) -> None:
    """F1: valid-key labels cannot relabel contradictory L1a evidence."""

    receipt, *_ = _receipt(tmp_path)

    raw = receipt.to_dict()
    raw["host_mapping"].update({"status": "unavailable", "host": "unavailable", "source": "declared-slot"})
    raw["comparison"]["host"] = "unavailable"
    raw["verdict"]["host"] = "unavailable"
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV2.from_mapping(raw)

    raw = receipt.to_dict()
    raw["child"]["affinity"] = [999999]
    raw["comparison"]["affinity"] = "unavailable"
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV2.from_mapping(raw)

    raw = receipt.to_dict()
    raw["gpu"].update({"method": "nvidia-smi", "samples": [{"active": True}]})
    raw["comparison"]["device"] = "matched"
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV2.from_mapping(raw)


def test_physical_remap_rewrites_derived_node_overlay() -> None:
    """F2: physical remapping updates a redundant TPEN_SLOT_NODE entry."""

    allowed = sorted(getattr(os, "sched_getaffinity", lambda _pid: (0,))(0))
    slot = AllocationSlot(
        "logical-slot", "logical-label", (GPU_A,), (allowed[0],), (),
        (("TPEN_SLOT_NODE", "logical-label"),),
    )
    allocation = ConfiguredAllocation(
        AllocationId("allocation-a"), BackendJobId("job-a"),
        {
            "scheduler": {"PBS_NODEFILE": PBS_NODEFILE_TEXT},
            "slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot.to_dict()]},
        },
    )
    physical = physical_slots_from_allocation(allocation)[0]
    assert physical.node == "physical-01"
    assert physical.env["TPEN_SLOT_NODE"] == "physical-01"


def test_child_precore_refusal_stays_launch_failed(tmp_path: Path) -> None:
    """F3: a RUN_SHIM affinity refusal is classified before lifecycle.finish."""

    group = _group()
    configured = ConfiguredRun("task-a", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("ok", encoding="utf-8")
    allocation = _allocation(scheduler={"PBS_NODEFILE": PBS_NODEFILE_TEXT})

    class Process:
        def run(self, environment, *args, **kwargs):
            observation = ChildPlacementObservation(
                environment.env["TPEN_PLACEMENT_TASK_ID"], environment.env["TPEN_PLACEMENT_ATTEMPT_ID"],
                environment.env["TPEN_PLACEMENT_GROUP_DIGEST"], environment.env["TPEN_PLACEMENT_ALLOCATION_ID"],
                environment.env["TPEN_PLACEMENT_BACKEND_JOB_ID"], environment.env["TPEN_PLACEMENT_SLOT_ID"],
                digest({"config": configured.config, "overrides": []}), environment.env["TPEN_PLACEMENT_OUTPUT_DIR"],
                "physical-01", 123, 0, (), {}, None, "2026-11-10T00:00:00Z",
                {"code": "child-placement-refused", "reason": "sched_setaffinity denied"},
            )
            payload = observation.to_dict()
            payload["refusal"] = dict(observation.refusal)
            Path(environment.env["TPEN_PLACEMENT_OUTPUT"]).write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
            return ProcessOutcome((sys.executable,), 2, "", "", False, 123, 55)

    class Evidence:
        def read(self, *args, **kwargs):
            raise AssertionError("pre-core refusal must not inspect core evidence")

    worker = SimpleNamespace(materialize=lambda _env=None: {
        "environment": PreparedPythonEnvironment(sys.executable, None, {}),
        "process": Process(), "runner": object(), "evidence": Evidence(),
    })
    manifest = LocalDispatcher(worker).run(
        group, {"task-a": configured},
        AttachedAllocation(allocation, DirectoryStore(tmp_path / "store")),
    )
    attempt = manifest.attempts[0]
    assert attempt.outcome_receipt.status == "launch_failed"
    assert _operational_exit(manifest) == 3
    assert attempt.placement_receipt.refusal["code"] == "child-placement-refused"
    assert attempt.placement_receipt.verdict["reason_class"] == "refused"
    assert not manifest.segments


@pytest.mark.parametrize("reason", ["", 123], ids=["empty", "non-string"])
def test_malformed_child_refusal_reason_finishes_typed_launch_failure(tmp_path: Path, reason: object) -> None:
    """N1: malformed child refusal still gets one terminal lifecycle receipt."""

    group = _group()
    configured = ConfiguredRun("task-a", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("ok", encoding="utf-8")
    allocation = _allocation(scheduler={"PBS_NODEFILE": PBS_NODEFILE_TEXT})

    class Process:
        def run(self, environment, *args, **kwargs):
            observation = ChildPlacementObservation(
                environment.env["TPEN_PLACEMENT_TASK_ID"], environment.env["TPEN_PLACEMENT_ATTEMPT_ID"],
                environment.env["TPEN_PLACEMENT_GROUP_DIGEST"], environment.env["TPEN_PLACEMENT_ALLOCATION_ID"],
                environment.env["TPEN_PLACEMENT_BACKEND_JOB_ID"], environment.env["TPEN_PLACEMENT_SLOT_ID"],
                digest({"config": configured.config, "overrides": []}), environment.env["TPEN_PLACEMENT_OUTPUT_DIR"],
                "physical-01", 123, 0, (), {}, None, "2026-11-10T00:00:00Z",
                {"code": "child-placement-refused", "reason": reason},
            )
            payload = observation.to_dict()
            payload["refusal"] = dict(observation.refusal)
            Path(environment.env["TPEN_PLACEMENT_OUTPUT"]).write_text(json.dumps(payload) + "\n", encoding="utf-8")
            return ProcessOutcome((sys.executable,), 2, "", "", False, 123, 55)

    class Evidence:
        def read(self, *args, **kwargs):
            raise AssertionError("malformed pre-core refusal must not inspect core evidence")

    worker = SimpleNamespace(materialize=lambda _env=None: {
        "environment": PreparedPythonEnvironment(sys.executable, None, {}),
        "process": Process(), "runner": object(), "evidence": Evidence(),
    })
    inner = AttachedAllocation(allocation, DirectoryStore(tmp_path / "store"))

    class CountingLifecycle:
        def __init__(self, wrapped):
            self.allocation = wrapped.allocation
            self.store = wrapped.store
            self.wrapped = wrapped
            self.finished = []

        def begin(self, *args, **kwargs):
            return self.wrapped.begin(*args, **kwargs)

        def finish(self, *args, **kwargs):
            self.finished.append(args[0])
            return self.wrapped.finish(*args, **kwargs)

    lifecycle = CountingLifecycle(inner)
    manifest = LocalDispatcher(worker).run(group, {"task-a": configured}, lifecycle)
    attempt = manifest.attempts[0]
    assert len(lifecycle.finished) == 1
    assert attempt.outcome_receipt is not None
    assert attempt.outcome_receipt.status == "launch_failed"
    assert attempt.placement_receipt.refusal["code"] == "placement-receipt-invalid"
    assert "non-empty string" in attempt.placement_receipt.refusal["reason"]


def test_generation_mismatch_preserves_physical_evidence(tmp_path: Path) -> None:
    """N2: a child generation mismatch is not rewritten as refused host evidence."""

    group, configured, allocation, fence, environment, observation = _v2_inputs(tmp_path)
    mismatched = ChildPlacementObservation(
        observation.task_id, observation.attempt_id, observation.group_digest,
        observation.allocation_id, observation.backend_job_id, observation.slot_id,
        observation.payload_digest, observation.output_dir, observation.host,
        observation.pid, observation.proc_start_ticks, observation.affinity,
        observation.thread_settings, observation.cuda_visible_devices,
        observation.observed_at_utc, None,
    )
    receipt = _placement_receipt_v2(
        mismatched, allocation, fence, group, group.members[0], configured, environment,
        ProcessOutcome((), 0, "", "", False, 1234, 999),
    )
    assert receipt.refusal["code"] == "placement-generation-mismatch"
    assert receipt.host_mapping["status"] == "matched"
    assert receipt.host_mapping["host"] == "physical-01"
    assert receipt.comparison["host"] == "matched"
    assert receipt.comparison["affinity"] == "matched"
    assert receipt.verdict["reason_class"] == "mismatch"


def test_late_proc_refusal_preserves_observed_affinity_and_cause(tmp_path: Path) -> None:
    """F3: /proc failure after affinity observation keeps the child cause."""

    group = _group()
    configured = ConfiguredRun("task-a", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("ok", encoding="utf-8")
    allocation = _allocation(scheduler={"PBS_NODEFILE": PBS_NODEFILE_TEXT})
    allowed = sorted(getattr(os, "sched_getaffinity", lambda _pid: (0,))(0))

    class Process:
        def run(self, environment, *args, **kwargs):
            observation = ChildPlacementObservation(
                environment.env["TPEN_PLACEMENT_TASK_ID"], environment.env["TPEN_PLACEMENT_ATTEMPT_ID"],
                environment.env["TPEN_PLACEMENT_GROUP_DIGEST"], environment.env["TPEN_PLACEMENT_ALLOCATION_ID"],
                environment.env["TPEN_PLACEMENT_BACKEND_JOB_ID"], environment.env["TPEN_PLACEMENT_SLOT_ID"],
                digest({"config": configured.config, "overrides": []}), environment.env["TPEN_PLACEMENT_OUTPUT_DIR"],
                "physical-01", 123, 0, (allowed[0],), {}, None, "2026-11-10T00:00:00Z",
                {"code": "child-placement-refused", "reason": "/proc stat unavailable"},
            )
            payload = observation.to_dict()
            payload["refusal"] = dict(observation.refusal)
            Path(environment.env["TPEN_PLACEMENT_OUTPUT"]).write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
            return ProcessOutcome((sys.executable,), 2, "", "", False, 123, 55)

    class Evidence:
        def read(self, *args, **kwargs):
            raise AssertionError("late pre-core refusal must not inspect core evidence")

    worker = SimpleNamespace(materialize=lambda _env=None: {
        "environment": PreparedPythonEnvironment(sys.executable, None, {}),
        "process": Process(), "runner": object(), "evidence": Evidence(),
    })
    manifest = LocalDispatcher(worker).run(
        group, {"task-a": configured}, AttachedAllocation(allocation, DirectoryStore(tmp_path / "store")),
    )
    attempt = manifest.attempts[0]
    assert attempt.outcome_receipt.status == "launch_failed"
    assert _operational_exit(manifest) == 3
    assert attempt.placement_receipt.refusal["code"] == "child-placement-refused"
    assert attempt.placement_receipt.comparison["affinity"] == "matched"
    assert "/proc stat unavailable" in attempt.placement_receipt.refusal["reason"]
    assert not manifest.segments


def test_slurm_range_width_and_bracket_products() -> None:
    """F4: Slurm preserves start-token width and expands bracket products."""

    assert expand_slurm_nodelist("nid[9-10]") == ("nid9", "nid10")
    assert expand_slurm_nodelist("nid[09-10]") == ("nid09", "nid10")
    assert expand_slurm_nodelist("rack[0-1]_blade[0-1]") == (
        "rack0_blade0", "rack0_blade1", "rack1_blade0", "rack1_blade1",
    )


def test_scheduler_host_validation_precedes_suffix_truncation() -> None:
    """F5: malformed complete scheduler hostnames cannot become short names."""

    with pytest.raises(PlacementRefusal) as captured:
        scheduler_host_map({"pbs_nodefile_text": "physical-01./evil\n"})
    assert captured.value.code == "host-map-ambiguous"


def test_l1a_mutant_table_has_exact_fields_and_unique_old_anchors() -> None:
    """F7: every mutation packet row is executable and has a unique anchor."""

    root = Path(__file__).parents[2]
    assert {row["id"] for row in L1A_MUTANT_TABLE} == {"M1", "M2", "M3", "M4", "M5", "N1", "N2", "F3"}
    for row in L1A_MUTANT_TABLE:
        target = root / row["target"]
        result = subprocess.run(
            ["grep", "-F", "-c", row["old"], str(target)],
            check=True, capture_output=True, text=True,
        )
        assert result.stdout.strip() == "1", row
        assert row["target"].startswith("executor/src/"), row
        assert row["suite"] == "executor"
        assert row["killer"] and row["assertion"] and row["new"]


def test_m4_prelaunch_host_refusal_is_launch_failed_without_segment(tmp_path: Path) -> None:
    """M4: a missing physical map fails closed before child launch."""

    called = False

    class Process:
        def run(self, *args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("prelaunch refusal must not launch the child")

    class Evidence:
        def read(self, *args, **kwargs):
            raise AssertionError("prelaunch refusal must not inspect a child")

    worker = SimpleNamespace(materialize=lambda _env=None: {
        "environment": PreparedPythonEnvironment(sys.executable, None, {}),
        "process": Process(), "runner": object(), "evidence": Evidence(),
    })
    manifest = LocalDispatcher(worker).run(
        _group(), {"task-a": ConfiguredRun("task-a", str(tmp_path / "config.yaml"))},
        AttachedAllocation(_allocation(scheduler=None), DirectoryStore(tmp_path / "store")),
    )
    assert not called
    assert manifest.attempts[0].outcome_receipt.status == "launch_failed"
    assert _operational_exit(manifest) == 3
    assert manifest.attempts[0].placement_receipt.schema == "tpen-executor/science-placement-receipt-v2"
    assert manifest.attempts[0].placement_receipt.host_mapping["status"] == "unavailable"
    assert manifest.attempts[0].placement_receipt.verdict["reason_class"] == "refused"
    assert not manifest.segments


def test_d6_unreadable_pbs_nodefile_is_typed_launch_failure(tmp_path: Path) -> None:
    """D6: an unreadable PBS path is not admission rc=2."""

    class Process:
        def run(self, *args, **kwargs):
            raise AssertionError("unreadable PBS_NODEFILE must not launch the child")

    class Evidence:
        def read(self, *args, **kwargs):
            raise AssertionError("unreadable PBS_NODEFILE must not inspect a child")

    allocation = _allocation(scheduler={"pbs_nodefile_path": str(tmp_path / "missing-nodefile")})
    worker = SimpleNamespace(materialize=lambda _env=None: {
        "environment": PreparedPythonEnvironment(sys.executable, None, {}),
        "process": Process(), "runner": object(), "evidence": Evidence(),
    })
    manifest = LocalDispatcher(worker).run(
        _group(), {"task-a": ConfiguredRun("task-a", str(tmp_path / "config.yaml"))},
        AttachedAllocation(allocation, DirectoryStore(tmp_path / "store")),
    )
    receipt = manifest.attempts[0].placement_receipt
    assert allocation.facts["scheduler"]["pbs_nodefile_error"]["code"] == "host-map-unavailable"
    assert manifest.attempts[0].outcome_receipt.status == "launch_failed"
    assert _operational_exit(manifest) == 3
    assert receipt.verdict["reason_class"] == "refused"
    assert receipt.host_mapping["host"] == "unavailable"
    assert not manifest.segments


def test_r7_refusal_does_not_attribute_logical_host_to_scheduler_map(tmp_path: Path) -> None:
    """R7: refusal keeps slot identity but never reports a logical host as mapped."""

    class Process:
        def run(self, *args, **kwargs):
            raise AssertionError("prelaunch refusal must not launch the child")

    class Evidence:
        def read(self, *args, **kwargs):
            raise AssertionError("prelaunch refusal must not inspect a child")

    worker = SimpleNamespace(materialize=lambda _env=None: {
        "environment": PreparedPythonEnvironment(sys.executable, None, {}),
        "process": Process(), "runner": object(), "evidence": Evidence(),
    })
    manifest = LocalDispatcher(worker).run(
        _group(), {"task-a": ConfiguredRun("task-a", str(tmp_path / "config.yaml"))},
        AttachedAllocation(_allocation(scheduler=None), DirectoryStore(tmp_path / "store")),
    )
    receipt = manifest.attempts[0].placement_receipt
    assert receipt.slot_id == "logical-slot"
    assert receipt.declared["host"] == "unavailable"
    assert receipt.host_mapping["source"] == "declared-slot"
    assert receipt.declared["devices"] == (GPU_A,)
    assert receipt.verdict["warning"]


def test_m5_advisory_postlaunch_mismatch_keeps_segment_and_rc_zero(tmp_path: Path) -> None:
    """M5: post-launch mismatch is advisory and leaves operational acceptance unchanged."""

    group, configured, allocation, fence, environment, observation = _v2_inputs(tmp_path)
    wrong = ChildPlacementObservation(
        observation.task_id, observation.attempt_id, observation.group_digest, observation.allocation_id,
        observation.backend_job_id, observation.slot_id, observation.payload_digest, observation.output_dir,
        "other-node", observation.pid, observation.proc_start_ticks, observation.affinity,
        observation.thread_settings, observation.cuda_visible_devices, observation.observed_at_utc, None,
    )
    receipt = _placement_receipt_v2(
        wrong, allocation, fence, group, group.members[0], configured, environment,
        ProcessOutcome((), 0, "", "", False, 1234, 55),
    )
    assert receipt.verdict["reason_class"] == "mismatch"
    assert receipt.verdict["qualification"] == "unqualified"


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="requires Linux controller affinity contract")
def test_m5_real_local_dispatch_preserves_segment_and_rc_zero(tmp_path: Path) -> None:
    """M5: serialized advisory evidence still yields an accepted segment."""

    group = _group()
    configured = ConfiguredRun("task-a", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("ok", encoding="utf-8")
    allocation = _allocation(scheduler={"PBS_NODEFILE": PBS_NODEFILE_TEXT})

    class Process:
        def run(self, environment, *args, **kwargs):
            base = ChildPlacementObservation(
                environment.env["TPEN_PLACEMENT_TASK_ID"], environment.env["TPEN_PLACEMENT_ATTEMPT_ID"],
                environment.env["TPEN_PLACEMENT_GROUP_DIGEST"], environment.env["TPEN_PLACEMENT_ALLOCATION_ID"],
                environment.env["TPEN_PLACEMENT_BACKEND_JOB_ID"], environment.env["TPEN_PLACEMENT_SLOT_ID"],
                digest({"config": configured.config, "overrides": []}), environment.env["TPEN_PLACEMENT_OUTPUT_DIR"],
                "other-node", 123, 55, (int(environment.env["TPEN_PLACEMENT_DECLARED_CPUS"]),), {}, None,
                "2026-11-10T00:00:00Z", None,
            )
            Path(environment.env["TPEN_PLACEMENT_OUTPUT"]).write_text(
                json.dumps(base.to_dict()) + "\n", encoding="utf-8"
            )
            Path(environment.env["TPEN_PLACEMENT_OUTPUT_DIR"], "core").mkdir(parents=True, exist_ok=True)
            Path(tmp_path / "checkpoint").mkdir(parents=True, exist_ok=True)
            return ProcessOutcome((sys.executable,), 0, "", "", False, 123, 55)

    class Evidence:
        def read(self, _process, _environment, run_dir, **_kwargs):
            # New accepted segments require core's resolved-config identity.
            return CoreRunEvidence(
                "tpen.run-evidence/v2", str(run_dir), "completed", "completed",
                {"checkpoint_dir": str(tmp_path / "checkpoint"), "schema_version": 1,
                 "next_iteration": 1, "completed_updates": 1}, None,
                {"schema": "tpen-config-identity-v1", "sha256": "a" * 64},
            )

    worker = SimpleNamespace(materialize=lambda _env=None: {
        "environment": PreparedPythonEnvironment(sys.executable, None, {}),
        "process": Process(), "runner": object(), "evidence": Evidence(),
    })
    manifest = LocalDispatcher(worker).run(group, {"task-a": configured},
                                           AttachedAllocation(allocation, DirectoryStore(tmp_path / "store")))
    attempt = manifest.attempts[0]
    assert attempt.outcome_receipt.status == "completed"
    assert _operational_exit(manifest) == 0
    assert attempt.placement_receipt.verdict["reason_class"] == "mismatch"
    assert manifest.segments
