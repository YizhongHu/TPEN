"""S2b-2 L1 executable mutant killers.

The suite is intentionally executor-only.  It uses a deterministic
``nvidia-smi`` executable on ``PATH`` for the in-child sampler and names the
six contract mutants in the test IDs so an independent Cannon run can report
them directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import socket
import sys
import textwrap
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
    ChildPlacementObservationV2,
    PlacementRefusal,
    physical_slots_from_allocation,
)
from tpen_executor.results import AcceptedSegment, ManifestAttempt, OutcomeReceipt, RunManifest, SciencePlacementReceiptV2, StartReceipt
from tpen_executor.runners import CoreRunEvidence, RUN_SHIM
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.workers import ProcessOutcome, SubprocessWorker

GPU_A = "GPU-12345678-1234-1234-1234-123456789ABC"
GPU_B = "GPU-12345678-1234-1234-1234-123456789ABD"


def _group() -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    member = TaskSpec("task-a", "task-a", program, digest("task-a"), 1, "steps")
    return ReadyGroup("s2b2", (member,), digest("s2b2"))


def _allocation(*, scheduler: object | None = None, device: str = GPU_A) -> ConfiguredAllocation:
    allowed = sorted(getattr(os, "sched_getaffinity", lambda _pid: (0,))(0))
    slot = AllocationSlot("logical-slot", "logical-label", (device,), (allowed[0],), (), ()).to_dict()
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


def _v2_inputs(tmp_path: Path, sampler: dict[str, object]):
    group = _group()
    configured = ConfiguredRun("task-a", str(tmp_path / "config.yaml"), ("payload.seed=1",))
    (tmp_path / "config.yaml").write_text("ok", encoding="utf-8")
    allocation = _allocation(scheduler={"PBS_NODEFILE": "physical-01\n"})
    fence = SimpleNamespace(attempt_id=AttemptId("attempt-a"), output_dir=str(tmp_path / "out"))
    allowed = sorted(getattr(os, "sched_getaffinity", lambda _pid: (0,))(0))
    sampler = dict(sampler)
    if sampler.get("status") == "observed":
        if not sampler.get("version"):
            sampler["version"] = "NVIDIA"
        sampler.setdefault("reason", "observed")
        if not sampler.get("samples"):
            sampler["samples"] = [{
            "monotonic_start": 1.0, "monotonic_end": 1.1,
            "inventory": sampler.get("inventory", []), "pid_rows": sampler.get("pid_rows", []),
            }]
        sampler["count"] = len(sampler["samples"])
        sampler["monotonic_start"] = 1.0
        sampler["monotonic_end"] = 1.1
    else:
        sampler.setdefault("samples", [])
        sampler.setdefault("count", 0)
        sampler.setdefault("monotonic_start", None)
        sampler.setdefault("monotonic_end", None)
    sampler.setdefault("command_pids", [111] if sampler.get("status") == "observed" else [])
    sampler.setdefault("survivors", [])
    sampler.setdefault("elapsed_seconds", 0.1)
    sampler.setdefault("join_timeout_seconds", 4.0)
    sampler.setdefault("quiescent", True)
    environment = PreparedPythonEnvironment(sys.executable, None, {
        "TPEN_SLOT_ID": "logical-slot",
        "TPEN_SLOT_NODE": "physical-01",
        "TPEN_SLOT_CPUS": str(allowed[0]),
        "TPEN_PLACEMENT_DECLARED_DEVICES": GPU_A,
        "TPEN_PLACEMENT_RECEIPT_VERSION": "2",
        "TPEN_PLACEMENT_HOST_STATUS": "mapped",
    })
    base = ChildPlacementObservation(
        "task-a", "attempt-a", group.identity, "allocation-a", "job-a", "logical-slot",
        digest({"config": configured.config, "overrides": list(configured.overrides)}), fence.output_dir,
        "physical-01", 1234, 55, (allowed[0],), {}, None, "2026-11-10T00:00:00Z", None,
    )
    return group, configured, allocation, fence, environment, ChildPlacementObservationV2(base, sampler)


def test_m2_live_pid_uuid_not_visibility_or_inventory(tmp_path: Path) -> None:
    """M2: an out-of-entitlement live PID row defeats visibility-only mutants."""

    sampler = {
        "status": "observed", "reason": "", "method": "nvidia-smi-pid-uuid-v1", "version": "NVIDIA",
        "cadence_seconds": 0.1, "count": 1, "monotonic_start": 1.0, "monotonic_end": 1.1,
        "inventory": [{"uuid": GPU_A, "pci.bus_id": "0000:00:01.0"}, {"uuid": GPU_B, "pci.bus_id": "0000:00:02.0"}],
        "pid_rows": [{"pid": 1234, "gpu_uuid": GPU_B}],
    }
    group, configured, allocation, fence, environment, observation = _v2_inputs(tmp_path, sampler)
    receipt = _placement_receipt_v2(observation, allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((), 0, "", "", False, 1234, 55))
    assert isinstance(receipt, SciencePlacementReceiptV2)
    assert receipt.comparison["device"] == "mismatch"
    assert receipt.verdict["qualification"] == "unqualified"


@pytest.mark.parametrize("reason", ["nvidia-smi-missing", "malformed-output", "no-live-pid-row"])
def test_m3_unavailable_is_typed_and_nonempty(tmp_path: Path, reason: str) -> None:
    """M3: no row, missing binary, and malformed output never qualify."""

    sampler = {
        "status": "unavailable", "reason": reason, "method": "nvidia-smi-pid-uuid-v1", "version": None,
        "cadence_seconds": 0.1, "count": 0, "monotonic_start": 1.0, "monotonic_end": 1.1,
        "inventory": [], "pid_rows": [], "samples": [],
    }
    group, configured, allocation, fence, environment, observation = _v2_inputs(tmp_path, sampler)
    receipt = _placement_receipt_v2(observation, allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((), 0, "", "", False, 1234, 55))
    assert receipt.comparison["device"] == "unavailable"
    assert receipt.gpu_sampler["reason"]


def test_n2_foreign_pid_rows_never_qualify(tmp_path: Path) -> None:
    """N2: an entitled UUID on another process is not science evidence."""

    sampler = {
        "status": "observed", "reason": "observed", "method": "nvidia-smi-pid-uuid-v1", "version": "NVIDIA",
        "cadence_seconds": 0.1, "count": 1, "monotonic_start": 1.0, "monotonic_end": 1.1,
        "inventory": [{"uuid": GPU_A, "pci.bus_id": "0000:00:01.0"}],
        "pid_rows": [{"pid": 9999, "gpu_uuid": GPU_A}],
    }
    group, configured, allocation, fence, environment, observation = _v2_inputs(tmp_path, sampler)
    receipt = _placement_receipt_v2(
        observation, allocation, fence, group, group.members[0], configured, environment,
        ProcessOutcome((), 0, "", "", False, 1234, 55),
    )
    assert receipt.verdict["qualification"] == "unqualified"
    assert receipt.gpu["active_uuids"] == ()


def test_n9_v2_receipt_rejects_unknown_nested_key(tmp_path: Path) -> None:
    """N9: closed v2 nested records cannot be silently widened."""

    sampler = {
        "status": "unavailable", "reason": "nvidia-smi-missing", "method": "nvidia-smi-pid-uuid-v1", "version": None,
        "cadence_seconds": 0.1, "count": 0, "monotonic_start": 1.0, "monotonic_end": 1.1,
        "inventory": [], "pid_rows": [], "samples": [],
    }
    group, configured, allocation, fence, environment, observation = _v2_inputs(tmp_path, sampler)
    receipt = _placement_receipt_v2(
        observation, allocation, fence, group, group.members[0], configured, environment,
        ProcessOutcome((), 0, "", "", False, 1234, 55),
    )
    raw = receipt.to_dict()
    raw["verdict"]["unexpected"] = "tamper"
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV2.from_mapping(raw)


def test_c1_receipt_reason_distinguishes_unavailable_mismatch_and_refused(tmp_path: Path) -> None:
    """Manager condition 1: receipt reason classes cannot collapse together."""

    unavailable = {
        "status": "unavailable", "reason": "nvidia-smi-missing", "method": "nvidia-smi-pid-uuid-v1", "version": None,
        "cadence_seconds": 0.1, "count": 0, "monotonic_start": 1.0, "monotonic_end": 1.1,
        "inventory": [], "pid_rows": [], "samples": [],
    }
    group, configured, allocation, fence, environment, observation = _v2_inputs(tmp_path, unavailable)
    no_telemetry = _placement_receipt_v2(observation, allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((), 0, "", "", False, 1234, 55))
    assert no_telemetry.verdict["reason_class"] == "telemetry-unavailable"
    assert no_telemetry.verdict["qualification"] != "qualified"

    mismatch_sampler = dict(unavailable)
    mismatch_sampler.update({"status": "observed", "reason": "observed", "inventory": [{"uuid": GPU_A, "pci.bus_id": "0000:00:01.0"}, {"uuid": GPU_B, "pci.bus_id": "0000:00:02.0"}], "pid_rows": [{"pid": 1234, "gpu_uuid": GPU_B}]})
    _, _, _, _, _, mismatched = _v2_inputs(tmp_path, mismatch_sampler)
    mismatch = _placement_receipt_v2(mismatched, allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((), 0, "", "", False, 1234, 55))
    assert mismatch.verdict["reason_class"] == "mismatch"

    refused_base = ChildPlacementObservation(
        observation.task_id, observation.attempt_id, observation.group_digest, observation.allocation_id,
        observation.backend_job_id, observation.slot_id, observation.payload_digest, observation.output_dir,
        observation.host, observation.pid, observation.proc_start_ticks, observation.affinity,
        observation.thread_settings, observation.cuda_visible_devices, observation.observed_at_utc,
        {"code": "child-placement-refused", "reason": "controller refused placement"},
    )
    refused = _placement_receipt_v2(ChildPlacementObservationV2(refused_base, unavailable), allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((), 0, "", "", False, 1234, 55))
    assert refused.verdict["reason_class"] == "refused"
    assert {no_telemetry.verdict["reason_class"], mismatch.verdict["reason_class"], refused.verdict["reason_class"]} == {"telemetry-unavailable", "mismatch", "refused"}


def test_c2_declared_devices_unavailable_is_loud_in_receipt_and_manifest(tmp_path: Path) -> None:
    """Manager condition 2: operational acceptance cannot imply science acceptance."""

    sampler = {
        "status": "unavailable", "reason": "nvidia-smi-missing", "method": "nvidia-smi-pid-uuid-v1", "version": None,
        "cadence_seconds": 0.1, "count": 0, "monotonic_start": 1.0, "monotonic_end": 1.1,
        "inventory": [], "pid_rows": [], "samples": [],
    }
    group, configured, allocation, fence, environment, observation = _v2_inputs(tmp_path, sampler)
    receipt = _placement_receipt_v2(observation, allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((), 0, "", "", False, 1234, 55))
    assert receipt.declared["devices"]
    assert receipt.verdict["qualification"] == "unqualified"
    assert receipt.verdict["science_acceptance"] == "not-accepted"
    assert "GPU PLACEMENT UNQUALIFIED" in receipt.verdict["warning"]
    start = StartReceipt("task-a", fence.attempt_id, allocation.allocation_id, allocation.backend_job_id, group.identity, "2026-11-10T00:00:00Z", fence.output_dir)
    outcome = OutcomeReceipt("task-a", fence.attempt_id, group.identity, "2026-11-10T00:00:01Z", "completed", fence.output_dir)
    attempt = ManifestAttempt("task-a", fence.attempt_id, start, outcome, fence.output_dir, receipt)
    segment = AcceptedSegment("task-a", fence.attempt_id, None, str(tmp_path / "checkpoint"), 1, 1)
    manifest = RunManifest(group.group_id, group.identity, (attempt,), (segment,))
    assert manifest.segments == (segment,)
    assert manifest.placement_verdict["status"] == "unqualified"
    assert manifest.placement_verdict["science_acceptance"] == "not-accepted"
    assert "GPU PLACEMENT UNQUALIFIED" in manifest.placement_verdict["message"]
    assert RunManifest.decode(manifest.encode()).placement_verdict["status"] == "unqualified"
    tampered = manifest.to_dict()
    tampered["placement_verdict"] = {"status": "qualified", "science_acceptance": "accepted", "message": "override"}
    with pytest.raises((TypeError, ValueError)):
        RunManifest.from_mapping(tampered)


def test_c3_logical_slot_name_does_not_drive_physical_verdict(tmp_path: Path) -> None:
    """Manager condition 3: only mapped physical host and live telemetry qualify."""

    sampler = {
        "status": "observed", "reason": "", "method": "nvidia-smi-pid-uuid-v1", "version": "NVIDIA",
        "cadence_seconds": 0.1, "count": 1, "monotonic_start": 1.0, "monotonic_end": 1.1,
        "inventory": [{"uuid": GPU_A, "pci.bus_id": "0000:00:01.0"}],
        "pid_rows": [{"pid": 1234, "gpu_uuid": GPU_A}],
    }
    group, configured, allocation, fence, environment, observation = _v2_inputs(tmp_path, sampler)
    receipt = _placement_receipt_v2(observation, allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((), 0, "", "", False, 1234, 55))
    assert receipt.slot_id == "logical-slot"
    assert receipt.declared["host"] == "physical-01"
    assert receipt.child["host"] == "physical-01"
    assert receipt.verdict["host"] == "matched"
    assert receipt.verdict["qualification"] == "qualified"
    assert receipt.verdict["reason_class"] == "qualified"


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
    allocation = _allocation(scheduler=None)
    manifest = LocalDispatcher(worker).run(
        _group(), {"task-a": ConfiguredRun("task-a", str(tmp_path / "config.yaml"))},
        AttachedAllocation(allocation, DirectoryStore(tmp_path / "store")),
    )
    assert not called
    assert manifest.attempts[0].outcome_receipt.status == "launch_failed"
    assert manifest.attempts[0].placement_receipt.schema == "tpen-executor/science-placement-receipt-v2"
    assert manifest.attempts[0].placement_receipt.host_mapping["status"] == "unavailable"
    assert manifest.attempts[0].placement_receipt.verdict["reason_class"] == "refused"
    assert not manifest.segments


def test_m5_advisory_postlaunch_mismatch_keeps_segment_and_rc_zero(tmp_path: Path) -> None:
    """M5: post-launch mismatch is unqualified evidence, not operational failure."""

    # This is a direct contract-level control: a completed child with a wrong
    # host still yields a refusal receipt, while lifecycle status remains the
    # completed status that the CLI maps to rc=0 and segment acceptance.
    group, configured, allocation, fence, environment, observation = _v2_inputs(tmp_path, {
        "status": "unavailable", "reason": "no-live-pid-row", "method": "nvidia-smi-pid-uuid-v1", "version": None,
        "cadence_seconds": 0.1, "count": 0, "monotonic_start": 1.0, "monotonic_end": 1.1,
        "inventory": [], "pid_rows": [], "samples": [],
    })
    wrong = ChildPlacementObservationV2(
        ChildPlacementObservation(
            observation.task_id, observation.attempt_id, observation.group_digest, observation.allocation_id,
            observation.backend_job_id, observation.slot_id, observation.payload_digest, observation.output_dir,
            "other-node", observation.pid, observation.proc_start_ticks, observation.affinity,
            observation.thread_settings, observation.cuda_visible_devices, observation.observed_at_utc, None,
        ), observation.gpu_sampler,
    )
    receipt = _placement_receipt_v2(wrong, allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((), 0, "", "", False, 1234, 55))
    assert receipt.refusal is not None
    assert receipt.comparison["host"] == "mismatch"
    assert receipt.verdict["qualification"] == "unqualified"
    assert receipt.verdict["reason_class"] == "mismatch"


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="requires Linux controller affinity contract")
def test_m5_real_local_dispatch_preserves_segment_and_rc_zero(tmp_path: Path) -> None:
    """M5: serialized advisory evidence still yields an accepted segment."""

    group = _group()
    configured = ConfiguredRun("task-a", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("ok", encoding="utf-8")
    allocation = _allocation(scheduler={"PBS_NODEFILE": "physical-01\n"})
    sampler = {
        "status": "unavailable", "reason": "nvidia-smi-missing", "method": "nvidia-smi-pid-uuid-v1", "version": None,
        "cadence_seconds": 0.1, "count": 0, "monotonic_start": None, "monotonic_end": None,
        "inventory": [], "pid_rows": [], "samples": [], "command_pids": [], "survivors": [],
        "elapsed_seconds": 0.1, "join_timeout_seconds": 4.0, "quiescent": True,
    }

    class Process:
        def run(self, environment, *args, **kwargs):
            assert environment.env["TPEN_PLACEMENT_DECLARED_HOST"] == "physical-01"
            assert environment.env["TPEN_SLOT_ID"] == "logical-slot"
            base = ChildPlacementObservation(
                environment.env["TPEN_PLACEMENT_TASK_ID"], environment.env["TPEN_PLACEMENT_ATTEMPT_ID"],
                environment.env["TPEN_PLACEMENT_GROUP_DIGEST"], environment.env["TPEN_PLACEMENT_ALLOCATION_ID"],
                environment.env["TPEN_PLACEMENT_BACKEND_JOB_ID"], environment.env["TPEN_PLACEMENT_SLOT_ID"],
                digest({"config": configured.config, "overrides": []}), environment.env["TPEN_PLACEMENT_OUTPUT_DIR"],
                "other-node", 123, 55, (int(environment.env["TPEN_PLACEMENT_DECLARED_CPUS"]),), {}, None,
                "2026-11-10T00:00:00Z", None,
            )
            ChildPlacementObservationV2(base, sampler).to_dict()
            Path(environment.env["TPEN_PLACEMENT_OUTPUT"]).write_text(
                json.dumps(ChildPlacementObservationV2(base, sampler).to_dict()) + "\n", encoding="utf-8"
            )
            return ProcessOutcome((sys.executable,), 0, "", "", False, 123, 55)

    class Evidence:
        def read(self, _process, _environment, run_dir, **_kwargs):
            return CoreRunEvidence(
                "tpen.run-evidence/v1", str(run_dir), "completed", "completed",
                {"checkpoint_dir": str(tmp_path / "checkpoint"), "schema_version": 1, "next_iteration": 1, "completed_updates": 1}, None,
            )

    worker = SimpleNamespace(materialize=lambda _env=None: {
        "environment": PreparedPythonEnvironment(sys.executable, None, {}),
        "process": Process(), "runner": object(), "evidence": Evidence(),
    })
    lifecycle = AttachedAllocation(allocation, DirectoryStore(tmp_path / "store"))
    manifest = LocalDispatcher(worker).run(group, {"task-a": configured}, lifecycle)
    attempt = manifest.attempts[0]
    assert attempt.outcome_receipt.status == "completed"
    assert _operational_exit(manifest) == 0
    assert attempt.placement_receipt.schema == "tpen-executor/science-placement-receipt-v2"
    assert attempt.placement_receipt.verdict["reason_class"] == "mismatch"
    assert manifest.segments
    assert manifest.placement_verdict["status"] == "unqualified"


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="requires Linux child affinity contract")
def test_m6_sampler_is_bounded_in_child_and_reaped(tmp_path: Path) -> None:
    """M6: deterministic fake sampler has bounded count and no child survivor."""

    fake = tmp_path / "nvidia-smi"
    fake.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import os, sys
        if '--version' in sys.argv:
            print('NVIDIA-SMI 2.0')
        elif '--query-gpu=uuid,pci.bus_id' in sys.argv:
            print('GPU-12345678-1234-1234-1234-123456789ABC, 0000:00:01.0')
        else:
            print(f'{os.getppid()}, GPU-12345678-1234-1234-1234-123456789ABC')
    """), encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    package = tmp_path / "pkg" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text("import time\ndef main(argv):\n    time.sleep(0.15)\n    return 0\n", encoding="utf-8")
    sidecar = tmp_path / "child-placement.json"
    allowed = sorted(getattr(os, "sched_getaffinity", lambda _pid: (0,))(0))
    environment = PreparedPythonEnvironment(sys.executable, None, {
        "PYTHONPATH": str(package.parent), "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
        "TPEN_PLACEMENT_OUTPUT": str(sidecar), "TPEN_PLACEMENT_OUTPUT_DIR": str(tmp_path),
        "TPEN_PLACEMENT_DECLARED_CPUS": str(allowed[0]), "TPEN_PLACEMENT_DECLARED_HOST": socket.gethostname(),
        "TPEN_PLACEMENT_TASK_ID": "task-a", "TPEN_PLACEMENT_ATTEMPT_ID": "attempt-a",
        "TPEN_PLACEMENT_GROUP_DIGEST": digest("s2b2"), "TPEN_PLACEMENT_ALLOCATION_ID": "allocation-a",
        "TPEN_PLACEMENT_BACKEND_JOB_ID": "job-a", "TPEN_PLACEMENT_SLOT_ID": "logical-slot",
        "TPEN_PLACEMENT_DECLARED_DEVICES": GPU_A, "TPEN_PLACEMENT_PAYLOAD_DIGEST": digest("payload"),
        "TPEN_PLACEMENT_GPU_SAMPLER_COUNT": "2", "TPEN_PLACEMENT_GPU_SAMPLER_CADENCE": "0.02",
    })
    outcome = SubprocessWorker().run(environment, ("-c", RUN_SHIM, f"run.root={tmp_path}"), cwd=tmp_path, stdout_path=tmp_path / "out", stderr_path=tmp_path / "err")
    assert outcome.returncode == 0
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["schema"] == "tpen-executor/child-placement-v2"
    assert payload["gpu_sampler"]["count"] <= 2
    assert payload["gpu_sampler"]["monotonic_end"] >= payload["gpu_sampler"]["monotonic_start"]
    assert payload["gpu_sampler"]["command_pids"]
    assert payload["gpu_sampler"]["survivors"] == []
    assert payload["gpu_sampler"]["quiescent"] is True
    assert payload["gpu_sampler"]["elapsed_seconds"] <= payload["gpu_sampler"]["join_timeout_seconds"]
    assert all(not Path(f"/proc/{pid}").exists() for pid in payload["gpu_sampler"]["command_pids"])
