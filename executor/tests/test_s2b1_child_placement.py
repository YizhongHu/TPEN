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
| F9 | M9: accept arbitrary comparison labels | placement comparisons use a closed vocabulary |
| C1 | drop refusal receipt construction | missing sidecars remain typed and readable |
| C2 | treat empty devices as absent | CPU-only children retain explicit placement input |
| M1 | compare raw host strings only | short/FQDN host forms still match canonically |
| N1 | drop the CUDA visibility fallback | Local and Parsl preset device declarations agree |
| H1 | delete the controller affinity subset check | dispatcher refuses an out-of-affinity slot before launch |
| H2 | delete invalid-receipt exception handling | dispatcher stores a typed construction refusal |
| F12 | omit NumpyDoc contract details | placement helpers document inputs, outputs, and refusals |
| L1 | emit PID zero for a refused receipt | refused receipts retain the positive compatibility sentinel |
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tpen_executor._codec import digest
from tpen_executor.contracts import AllocationId, BackendJobId, AttemptId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.dispatch import contract as contract_module
from tpen_executor.dispatch.contract import _placement_receipt, _placement_refusal_receipt, _with_child_placement
from tpen_executor.dispatch.local import LocalDispatcher
from tpen_executor.placement import ChildPlacementObservation, PlacementRefusal, parse_proc_start_ticks, read_child_placement
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.results import ManifestAttempt, OutcomeReceipt, SciencePlacementReceipt, StartReceipt
from tpen_executor.runners import EvidenceUnavailable, RUN_SHIM, TPENEvidence
from tpen_executor.storage import DirectoryStore
from tpen_executor.placement import AllocationSlot
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


@pytest.mark.parametrize(("field", "value"), [
    ("task_id", "other-task"),
    ("attempt_id", "other-attempt"),
    ("group_digest", digest({"other": True})),
    ("allocation_id", "other-allocation"),
    ("backend_job_id", "other-job"),
    ("slot_id", "other-slot"),
    ("payload_digest", digest({"other": True})),
    ("output_dir", "/elsewhere"),
    ("host", "other-node"),
    ("affinity", (3,)),
])
def test_t3_receipt_rejects_every_cross_binding_identity(tmp_path: Path, field: str, value: object) -> None:
    """T3 kills M3: accepting task-index, worker-PID, or another attempt."""

    group, configured, allocation, fence, environment = _receipt_inputs(tmp_path)
    observation = _observation(tmp_path, **{field: value})
    receipt = _placement_receipt(observation, allocation, fence, group, group.members[0], configured, environment, ProcessOutcome((sys.executable,), 0, "", "", False, 1234, 55))
    assert receipt.refusal is not None
    assert receipt.refusal["code"] in {"placement-identity-mismatch", "placement-affinity-mismatch"}


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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux /proc and sched affinity")
def test_f8_real_shim_applies_a_strict_permitted_cpu_subset(tmp_path: Path) -> None:
    """F8/M8: the science child, not the controller, applies a permitted subset."""

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("needs at least two permitted CPUs")
    package = tmp_path / "pkg" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text("def main(argv):\n    return 0\n", encoding="utf-8")
    sidecar = tmp_path / "child-placement.json"
    environment = PreparedPythonEnvironment(sys.executable, None, {
        "PYTHONPATH": str(package.parent),
        "TPEN_PLACEMENT_OUTPUT": str(sidecar),
        "TPEN_PLACEMENT_DECLARED_CPUS": str(allowed[0]),
        "TPEN_PLACEMENT_DECLARED_HOST": socket.gethostname(),
    })
    outcome = SubprocessWorker().run(
        environment,
        ("-c", RUN_SHIM, f"run.root={tmp_path}", "run.run_id=core"),
        cwd=tmp_path,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
    )
    assert outcome.returncode == 0
    observed = json.loads(sidecar.read_text(encoding="utf-8"))
    assert observed["refusal"] is None
    assert observed["affinity"] == [allowed[0]]
    assert observed["pid"] == outcome.pid
    assert observed["proc_start_ticks"] == outcome.proc_start_ticks


def test_f9_receipt_rejects_unknown_comparison_state() -> None:
    """F9 kills M9: comparison fields cannot carry an unreviewed state."""

    with pytest.raises(ValueError, match="unknown state"):
        SciencePlacementReceipt(
            "task-a", AttemptId("attempt-a"), GROUP_DIGEST, "allocation-a", "job-a", "slot-a",
            digest("payload"), "/tmp/output",
            {"host": "node-a", "pid": 1, "proc_start_ticks": 2, "affinity": [0], "thread_settings": {}, "cuda_visible_devices": None, "observed_at_utc": "now"},
            {"host": "node-a", "cpus": [0], "devices": []},
            {"status": "unavailable", "reason": "cpu", "method": "test", "started_at_utc": "now", "finished_at_utc": "now", "samples": [], "active_uuids": [], "active_pci": []},
            {"host": "matched", "affinity": "matched", "device": "not-a-state"},
        )


def _stub_package(tmp_path: Path) -> Path:
    """Create a minimal child-only ``tpen`` package for RUN_SHIM tests."""

    package = tmp_path / "stub" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        textwrap.dedent(
            """
            import json
            from pathlib import Path

            def main(argv):
                values = dict(item.split('=', 1) for item in argv if '=' in item)
                core = Path(values['run.root']) / values['run.run_id']
                core.mkdir(parents=True, exist_ok=True)
                (core / 'evidence.json').write_text(json.dumps({
                    'schema': 'tpen.run-evidence/v1',
                    'run_dir': str(core),
                    'verdict': 'completed',
                    'run_status': 'completed',
                    'checkpoint': None,
                    'refusal': None,
                }), encoding='utf-8')
                return 0
            """
        ),
        encoding="utf-8",
    )
    return package.parent


def _slot_allocation(slot: AllocationSlot) -> ConfiguredAllocation:
    return ConfiguredAllocation(
        AllocationId("allocation-a"),
        BackendJobId("job-a"),
        {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot.to_dict()]}},
    )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux sched affinity")
def test_n1_local_and_parsl_preset_device_declarations_agree(tmp_path: Path) -> None:
    """N1 kills a dropped CUDA visibility fallback for preset device slots."""

    cpu = min(os.sched_getaffinity(0))
    group = _group()
    configured = _configured(tmp_path)
    fence = SimpleNamespace(attempt_id=AttemptId("attempt-a"), output_dir=str(tmp_path / "output"))
    slot = AllocationSlot("slot-a", socket.gethostname(), ("0",), (cpu,), (), ())
    local = _with_child_placement(
        PreparedPythonEnvironment(sys.executable, None, {}),
        _slot_allocation(slot), fence, group, group.members[0], configured,
    )
    preset = _with_child_placement(
        PreparedPythonEnvironment(sys.executable, None, dict(slot.env)),
        ConfiguredAllocation(AllocationId("allocation-a"), BackendJobId("job-a"), {}),
        fence, group, group.members[0], configured,
    )
    assert local.env["TPEN_PLACEMENT_DECLARED_DEVICES"] == "0"
    assert preset.env["TPEN_PLACEMENT_DECLARED_DEVICES"] == local.env["TPEN_PLACEMENT_DECLARED_DEVICES"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux sched affinity")
@pytest.mark.parametrize("devices", [("0",), ()], ids=["device-control", "cpu-only"])
def test_prepared_visibility_does_not_change_slot_entitlement(
    tmp_path: Path, devices: tuple[str, ...],
) -> None:
    """D1 kills deriving declared devices from merged CUDA visibility."""

    slot = AllocationSlot(
        "slot-a", socket.gethostname(), devices,
        (min(os.sched_getaffinity(0)),), (), (),
    )
    allocation = _slot_allocation(slot)
    group = _group()
    member = group.members[0]
    configured = _configured(tmp_path)
    fence = SimpleNamespace(attempt_id=AttemptId("attempt-a"), output_dir=str(tmp_path / "output"))
    prepared = PreparedPythonEnvironment(sys.executable, env={"CUDA_VISIBLE_DEVICES": "7"})

    local = _with_child_placement(prepared, allocation, fence, group, member, configured)
    preset = ParslDispatcher._worker_environment(
        SimpleNamespace(), {"environment": prepared}, binding=slot.binding(),
    )
    parsl = _with_child_placement(preset, allocation, fence, group, member, configured)

    assert local.env["TPEN_PLACEMENT_DECLARED_DEVICES"] == ",".join(devices)
    assert parsl.env["TPEN_PLACEMENT_DECLARED_DEVICES"] == local.env["TPEN_PLACEMENT_DECLARED_DEVICES"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux sched affinity")
def test_h1_dispatcher_refuses_cpu_outside_controller_affinity(tmp_path: Path) -> None:
    """H1 kills deletion of the controller CPU-affinity subset check."""

    outside = max(os.sched_getaffinity(0)) + 1
    slot = AllocationSlot("slot-a", socket.gethostname(), (), (outside,), (), ())
    launched: list[object] = []

    class Process:
        def run(self, environment, args, **kwargs):
            launched.append(args)
            return ProcessOutcome((sys.executable,), 0, kwargs["stdout_path"], kwargs["stderr_path"], False)

    worker = SimpleNamespace(
        materialize=lambda env: {
            "environment": PreparedPythonEnvironment(sys.executable),
            "process": Process(),
            "runner": object(),
            "evidence": TPENEvidence(),
        }
    )
    lifecycle = AttachedAllocation(_slot_allocation(slot), DirectoryStore(tmp_path / "store"))
    manifest = LocalDispatcher(worker).run(_group(), {"task-a": _configured(tmp_path)}, lifecycle)
    assert launched == []
    attempt = manifest.attempts[0]
    assert attempt.outcome_receipt.status == "launch_failed"
    assert attempt.placement_receipt.refusal["code"] == "placement-cpu-outside-controller-affinity"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux process execution")
def test_h2_invalid_receipt_construction_becomes_typed_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H2 kills removal of the invalid receipt-construction handler."""

    cpu = min(os.sched_getaffinity(0))
    package = _stub_package(tmp_path)
    environment = PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})
    worker = SimpleNamespace(
        materialize=lambda env: {
            "environment": environment,
            "process": SubprocessWorker(),
            "runner": object(),
            "evidence": TPENEvidence(),
        }
    )

    def broken_receipt(*args, **kwargs):
        raise TypeError("declared host must be non-empty text")

    monkeypatch.setattr(contract_module, "_placement_receipt", broken_receipt)
    lifecycle = AttachedAllocation(
        _slot_allocation(AllocationSlot("slot-a", socket.gethostname(), (), (cpu,), (), ())),
        DirectoryStore(tmp_path / "store"),
    )
    manifest = LocalDispatcher(worker).run(_group(), {"task-a": _configured(tmp_path)}, lifecycle)
    attempt = manifest.attempts[0]
    assert attempt.outcome_receipt.status == "completed"
    assert attempt.placement_receipt.refusal["code"] == "placement-receipt-invalid"
    assert "declared host" in attempt.placement_receipt.refusal["reason"]


def test_c1a_refusal_receipt_is_constructible(tmp_path: Path) -> None:
    """C1/M1a: a missing sidecar is a valid refused placement receipt."""

    group, configured, allocation, fence, environment = _receipt_inputs(tmp_path)
    receipt = _placement_refusal_receipt(
        PlacementRefusal("placement-missing", "child placement sidecar is unavailable"),
        allocation,
        fence,
        group,
        group.members[0],
        configured,
        environment,
        ProcessOutcome((sys.executable,), 1, "", "", False, None, None),
    )
    assert receipt.refusal == {"code": "placement-missing", "reason": "child placement sidecar is unavailable"}
    assert receipt.child["pid"] == 1
    assert dict(receipt.comparison) == {"host": "refused", "affinity": "refused", "device": "unavailable"}


def test_c1b_local_missing_sidecar_records_refusal_and_continues(tmp_path: Path) -> None:
    """C1/M1b: one missing sidecar does not abort later members."""

    cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    slot = AllocationSlot("slot-a", socket.gethostname(), (), (cpu,), (), ())

    class Process:
        def run(self, environment, args, **kwargs):
            del environment, args
            return ProcessOutcome((sys.executable,), 0, kwargs["stdout_path"], kwargs["stderr_path"], False)

    evidence = SimpleNamespace(read=lambda *args, **kwargs: (_ for _ in ()).throw(EvidenceUnavailable("none")))
    worker = SimpleNamespace(
        materialize=lambda env: {
            "environment": PreparedPythonEnvironment(sys.executable),
            "process": Process(),
            "runner": object(),
            "evidence": evidence,
        }
    )
    group = ReadyGroup(
        "s2b1-two",
        tuple(TaskSpec(task, task, ProgramSpecRef({}, digest({})), digest(task), 1, "steps") for task in ("a", "b")),
        digest({"group": "s2b1-two"}),
    )
    runs = {}
    for task in ("a", "b"):
        config = tmp_path / f"{task}.yaml"
        config.write_text("ok\n", encoding="utf-8")
        from tpen_executor.tasks import ConfiguredRun
        runs[task] = ConfiguredRun(task, str(config), (f"payload.task={task}",))
    lifecycle = AttachedAllocation(_slot_allocation(slot), DirectoryStore(tmp_path / "store"))
    manifest = LocalDispatcher(worker).run(group, runs, lifecycle)
    assert [attempt.outcome_receipt.status for attempt in manifest.attempts] == ["completed", "completed"]
    assert all(attempt.placement_receipt is not None for attempt in manifest.attempts)
    assert all(attempt.placement_receipt.refusal["code"] == "placement-missing" for attempt in manifest.attempts)


def test_c1c_cancellation_is_preserved_with_refusal_receipt(tmp_path: Path) -> None:
    """C1/M1c: receipt construction cannot replace owner cancellation."""

    cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    slot = AllocationSlot("slot-a", socket.gethostname(), (), (cpu,), (), ())

    class Process:
        def run(self, environment, args, **kwargs):
            del environment, args, kwargs
            raise KeyboardInterrupt("cancel before child spawn")

    worker = SimpleNamespace(
        materialize=lambda env: {
            "environment": PreparedPythonEnvironment(sys.executable),
            "process": Process(),
            "runner": object(),
            "evidence": TPENEvidence(),
        }
    )
    group = _group()
    lifecycle = AttachedAllocation(_slot_allocation(slot), DirectoryStore(tmp_path / "store"))
    with pytest.raises(KeyboardInterrupt, match="cancel before child spawn"):
        LocalDispatcher(worker).run(group, {"task-a": _configured(tmp_path)}, lifecycle)
    manifest = lifecycle.store.manifest(group)
    assert manifest.attempts[0].outcome_receipt.status == "failed"
    assert manifest.attempts[0].placement_receipt.refusal["code"] == "placement-missing"


def test_c2a_cpu_only_slot_gets_a_valid_overlay(tmp_path: Path) -> None:
    """C2/M2a: an empty device entitlement is valid CPU placement input."""

    group = _group()
    configured = _configured(tmp_path)
    fence = SimpleNamespace(attempt_id=AttemptId("attempt-a"), output_dir=str(tmp_path / "output"))
    cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    allocation = _slot_allocation(AllocationSlot("slot-a", socket.gethostname(), (), (cpu,), (), ()))
    environment = _with_child_placement(
        PreparedPythonEnvironment(sys.executable, None, {}), allocation, fence, group, group.members[0], configured
    )
    assert environment.env["TPEN_PLACEMENT_DECLARED_DEVICES"] == ""
    assert environment.env["TPEN_PLACEMENT_DECLARED_CPUS"] == str(cpu)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux /proc and sched affinity")
def test_c2b_cpu_only_real_child_publishes_matched_receipt(tmp_path: Path) -> None:
    """C2/M2b: the real CPU-only child reaches core and publishes placement."""

    cpu = min(os.sched_getaffinity(0))
    package = _stub_package(tmp_path)
    environment = PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})
    worker = SimpleNamespace(
        materialize=lambda env: {
            "environment": environment,
            "process": SubprocessWorker(),
            "runner": object(),
            "evidence": TPENEvidence(),
        }
    )
    group = _group()
    lifecycle = AttachedAllocation(
        _slot_allocation(AllocationSlot("slot-a", socket.gethostname(), (), (cpu,), (), ())),
        DirectoryStore(tmp_path / "store"),
    )
    manifest = LocalDispatcher(worker).run(group, {"task-a": _configured(tmp_path)}, lifecycle)
    attempt = manifest.attempts[0]
    assert attempt.outcome_receipt.status == "completed"
    assert attempt.placement_receipt is not None
    assert attempt.placement_receipt.refusal is None
    assert dict(attempt.placement_receipt.comparison) == {"host": "matched", "affinity": "matched", "device": "unavailable"}


def test_m1_short_and_fqdn_hosts_compare_canonically(tmp_path: Path) -> None:
    """M1: raw host labels are retained while short/FQDN comparison matches."""

    group, configured, allocation, fence, environment = _receipt_inputs(tmp_path)
    receipt = _placement_receipt(
        _observation(tmp_path, group_digest=group.identity, host="node-a.example.org"),
        allocation,
        fence,
        group,
        group.members[0],
        configured,
        environment,
        ProcessOutcome((sys.executable,), 0, "", "", False, 1234, 55),
    )
    assert receipt.refusal is None
    assert receipt.child["host"] == "node-a.example.org"
    assert receipt.declared["host"] == "node-a"
