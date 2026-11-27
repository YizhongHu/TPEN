"""S2-L3 generic group-runtime composition tests."""

import os
from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation, ControllerGenerationV1
from tpen_executor.contracts import AllocationId, BackendJobId, ExecutionPlanV2, GroupId, ProgramSpecRef, ReadyGroupV2, TaskSpec
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle.group_runtime import GroupRuntimeDispatcher, PackedGroupRequest, PreparedGroupInput, PreparedRuntimeIdentity
from tpen_executor.recovery import MemberRecoveryInputV2


class _Materialized:
    def __init__(self, slots):
        self.slots = slots
        for key, value in slots.items():
            setattr(self, key, value)

    def __contains__(self, key):
        return key in self.slots


class _Program:
    def __init__(self, materialized):
        self.materialized = materialized
        self.envs = []
        self.spec = ProgramSpecRef({"fixture": True}, digest({"fixture": True}))

    def materialize(self, *, env):
        self.envs.append(dict(env))
        return self.materialized


def _prepared(tmp_path: Path, *, allocation_program=None, worker_program=None):
    config = {"program": "fixture"}
    task = TaskSpec("task", "lineage", ProgramSpecRef(config, digest(config)), "a" * 64, 1, "steps")
    plan = ExecutionPlanV2(config, digest(config), ("task",), "profile", "protocol")
    group = ReadyGroupV2(GroupId("group"), (task,), plan)
    allocation = ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})
    controller = ControllerGenerationV1(group.group_id, 1, allocation.allocation_id, allocation.backend_job_id, "b" * 64, "nonce", None)
    env = {"paths": {"control_python": "/prepared/control", "science_python": "/prepared/science"}}
    allocation_program = allocation_program or _Program(_Materialized({"dispatcher": _Materialized({"run": lambda: None})}))
    worker_program = worker_program or _Program(_Materialized({"environment": PreparedPythonEnvironment("/prepared/science"), "process": object(), "runner": object(), "evidence": object()}))
    deployment = ProgramSpecRef({"deployment": "fixture"}, digest({"deployment": "fixture"}))
    identity = PreparedRuntimeIdentity(group.identity, allocation_program.spec, worker_program.spec, deployment)
    return PreparedGroupInput(
        group, allocation, controller, (MemberRecoveryInputV2("task", "lineage"),),
        allocation_program, worker_program, env, str(tmp_path / "store"), identity.bundle_digest,
        deployment.content_digest, "b" * 64, "nonce", "2026-11-26T00:00:00Z", identity=identity,
    )


def test_materialize_forwards_explicit_env_to_both_owner_processes(tmp_path):
    allocation = _Program(_Materialized({"dispatcher": _Materialized({"run": lambda: None})}))
    worker = _Program(_Materialized({"environment": PreparedPythonEnvironment("/prepared/science"), "process": object(), "runner": object(), "evidence": object()}))
    prepared = _prepared(tmp_path, allocation_program=allocation, worker_program=worker)
    prepared.materialize()
    assert allocation.envs == [prepared.env]
    assert worker.envs == [prepared.env]


def test_packed_request_is_immutable_and_rejects_partial_members(tmp_path):
    prepared = _prepared(tmp_path)
    request = PackedGroupRequest(prepared)
    with pytest.raises((AttributeError, TypeError)):
        request.reentry = True
    with pytest.raises(ValueError):
        PreparedGroupInput(
            prepared.group, prepared.allocation, prepared.controller, (), prepared.allocation_program,
            prepared.worker_program, prepared.env, prepared.store_root, prepared.execution_bundle_digest,
            prepared.deployment_lineage, prepared.registration_receipt_digest, prepared.claimant_nonce,
            prepared.observed_at_utc,
        )


def test_dispatcher_has_finite_deadlines():
    dispatcher = GroupRuntimeDispatcher(timeout=3.0, core_timeout=2.0)
    assert dispatcher.timeout == 3.0
    assert dispatcher.core_timeout == 2.0
    with pytest.raises(ValueError):
        GroupRuntimeDispatcher(timeout=0)


@pytest.mark.parametrize("field", ["timeout", "core_timeout"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0, -1, "2", None, True])
def test_nonfinite_deadlines_rejected_before_side_effects(tmp_path, field, value):
    from dataclasses import replace

    prepared = _prepared(tmp_path)
    with pytest.raises(ValueError):
        replace(prepared, **{field: value})
    with pytest.raises(ValueError):
        GroupRuntimeDispatcher(**{field: value})
    assert not Path(prepared.store_root).exists()
    assert prepared.allocation_program.envs == []


@pytest.mark.parametrize("field,value", [
    ("registration_receipt_digest", "e" * 64),
    ("claimant_nonce", "other"),
    ("predecessor_record_digest", "f" * 64),
])
def test_crossed_authority_rejected_before_store(tmp_path, field, value):
    from dataclasses import replace

    prepared = _prepared(tmp_path)
    with pytest.raises(ValueError, match="authority facts"):
        replace(prepared, **{field: value})
    assert not Path(prepared.store_root).exists()


def test_crossed_allocation_rejected_before_store(tmp_path):
    from dataclasses import replace

    prepared = _prepared(tmp_path)
    with pytest.raises(ValueError, match="allocation identity"):
        replace(prepared, allocation=ConfiguredAllocation(AllocationId("other"), BackendJobId("job"), {}))
    assert not Path(prepared.store_root).exists()


def test_dispatcher_uses_admitted_controller_and_configured_deadlines(tmp_path):
    from dataclasses import replace
    from tpen_executor.storage import DirectoryStore
    from tpen_executor.tasks import ConfiguredRun

    prepared = _prepared(tmp_path)
    prepared = replace(prepared, runs={"task": ConfiguredRun("task", str(tmp_path / "science.yaml"))})
    observed = {}

    class Executor:
        def __init__(self, *args, **kwargs):
            observed.update(kwargs)
        def close(self):
            observed["closed"] = True

    def controller(binding, job, facts, store, port, executor):
        assert binding.intent.group_id == prepared.group.group_id
        assert binding.intent.group_digest == prepared.group.identity
        assert binding.backend_job_id == job == prepared.controller.backend_job_id
        rebuilt = ControllerGenerationV1(facts.group.group_id, facts.controller_generation,
            facts.allocation.allocation_id, job, facts.registration_receipt_digest,
            facts.claimant_nonce, facts.predecessor_record_digest)
        assert rebuilt.encode() == prepared.controller.encode()
        assert store.acquire_controller_generation(facts.group, rebuilt) == rebuilt
        assert facts.poll_timeout == 7
        assert port._core_timeout == 11
        return "called"

    dispatcher = GroupRuntimeDispatcher(controller_runner=controller, local_executor_factory=Executor,
                                        timeout=7, core_timeout=11)
    assert dispatcher.run(PackedGroupRequest(prepared)) == "called"
    assert observed == {"capacity": 1, "timeout": 7, "closed": True}


@pytest.mark.parametrize("root_kind", ["regular-file", "symlink"])
@pytest.mark.parametrize("reentry", [False, True])
def test_invalid_preflight_store_root_is_typed_refusal_without_launch(
    tmp_path, root_kind, reentry,
):
    """Invalid existing roots refuse before controller or leaf admission."""
    from dataclasses import replace
    from tpen_executor.lifecycle.group_store import GroupStoreRefused
    from tpen_executor.tasks import ConfiguredRun

    prepared = replace(
        _prepared(tmp_path),
        runs={"task": ConfiguredRun("task", str(tmp_path / "science.yaml"))},
    )
    root = Path(prepared.store_root)
    if root_kind == "regular-file":
        root.write_bytes(b"store-root-file")
        before = root.read_bytes()
    else:
        target = tmp_path / "store-target"
        target.mkdir()
        (target / "marker").write_bytes(b"store-target")
        root.symlink_to(target, target_is_directory=True)
        before = os.readlink(root)

    def no_controller_or_leaf(*_args, **_kwargs):
        pytest.fail("invalid store root reached controller or leaf")

    with pytest.raises(GroupStoreRefused):
        GroupRuntimeDispatcher(
            controller_runner=no_controller_or_leaf,
            local_executor_factory=no_controller_or_leaf,
        ).run(PackedGroupRequest(prepared, reentry=reentry))

    if root_kind == "regular-file":
        assert root.read_bytes() == before
    else:
        assert os.readlink(root) == before
        assert (root / "marker").read_bytes() == b"store-target"


@pytest.mark.parametrize("field", ["execution_bundle_digest", "deployment_lineage"])
def test_crossed_bundle_or_deployment_refused_without_store(tmp_path, field):
    from dataclasses import replace
    from tpen_executor.lifecycle.group_store import GroupStoreRefused

    prepared = replace(_prepared(tmp_path), **{field: "e" * 64})
    with pytest.raises(GroupStoreRefused, match="identity crosses"):
        GroupRuntimeDispatcher().run(PackedGroupRequest(prepared))
    assert not Path(prepared.store_root).exists()
    assert prepared.allocation_program.envs == []


def test_bound_port_preserves_config_and_each_call_artifacts(tmp_path):
    import json
    from tpen_executor.lifecycle.core_port import ScienceProcessCoreAdmissionPort
    from tpen_executor.lifecycle.group_runtime import _BoundCorePort
    from test_s2_core_port import _Worker

    prepared = _prepared(tmp_path)
    worker = _Worker(None)
    port = _BoundCorePort(prepared, ScienceProcessCoreAdmissionPort(worker), tmp_path / "calls",
                         prepared.worker_program.materialized.environment, 4)
    member = prepared.member_inputs[0]
    assert port.admit(member) is None
    first = {p: p.read_bytes() for p in (tmp_path / "calls").rglob("*") if p.is_file()}
    assert port.admit(member) is None
    assert worker.calls == 2
    assert all(p.read_bytes() == content for p, content in first.items())
    assert len(list((tmp_path / "calls").rglob("stdout.log"))) == 2
    assert len(list((tmp_path / "calls").rglob("stderr.log"))) == 2


def test_real_dispatcher_controller_local_executor_refills_two_slots(tmp_path, monkeypatch):
    import os
    # Hardware entitlement is an injected boundary; no real child is launched.
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1}, raising=False)
    from dataclasses import replace
    from threading import Event
    from tpen_executor.dispatch.transport import LeafRequest
    from tpen_executor.lifecycle.core_port import ScienceProcessCoreAdmissionPort
    from tpen_executor.lifecycle.local_executor import LocalLeafExecutor
    from tpen_executor.placement import AllocationSlot
    from tpen_executor.tasks import ConfiguredRun
    from test_b1e1_local_executor import _failure_result
    from test_s2_core_port import _Worker

    prepared = _prepared(tmp_path)
    names = ("one", "two", "three")
    tasks = tuple(replace(prepared.group.members[0], task_id=name, lineage_id="lineage-" + name) for name in names)
    group = replace(prepared.group, members=tasks,
                    execution_plan=replace(prepared.group.execution_plan, member_task_ids=names))
    slots = [AllocationSlot("slot-" + str(i), "localhost", (), (i,), (), ()).to_dict() for i in range(2)]
    allocation = replace(prepared.allocation, facts={"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": slots}})
    identity = replace(prepared.identity, group_digest=group.identity)
    prepared = replace(prepared, group=group, allocation=allocation,
                       member_inputs=tuple(MemberRecoveryInputV2(t.task_id, t.lineage_id) for t in tasks),
                       identity=identity, execution_bundle_digest=identity.bundle_digest,
                       runs={name: ConfiguredRun(name, str(tmp_path / (name + ".yaml"))) for name in names})
    submitted, collected, requests = [], [], []
    two_started = Event()

    def leaf(payload):
        request = LeafRequest.decode(payload)
        requests.append(request)
        if len(requests) == 2:
            two_started.set()
        assert two_started.wait(5)
        return _failure_result(request).encode()

    class ObservedExecutor(LocalLeafExecutor):
        def submit(self, fence, invocation):
            if len(submitted) == 2:
                assert collected, "third submission must follow collection"
            result = super().submit(fence, invocation)
            submitted.append(invocation.task_id)
            assert self._active <= 2
            return result
        def collect(self, handle, timeout=None):
            assert timeout == 7
            result = super().collect(handle, timeout)
            collected.append(handle)
            return result

    def factory(*args, **kwargs):
        assert kwargs == {"capacity": 2, "timeout": 7}
        return ObservedExecutor(*args, **kwargs, leaf_runner=leaf)

    worker = _Worker(None)
    dispatcher = GroupRuntimeDispatcher(core_port=ScienceProcessCoreAdmissionPort(worker),
        local_executor_factory=factory, timeout=7, core_timeout=11)
    result = dispatcher.run(PackedGroupRequest(prepared))
    assert submitted == list(names), result
    assert len(collected) == 3
    assert len(result.reconciliation.members) == 3
    assert requests[0].slot_id != requests[1].slot_id
    assert requests[2].slot_id in {requests[0].slot_id, requests[1].slot_id}
    assert worker.calls == 6  # initial and final admission for every member


def test_successor_wire_and_publication_facts_share_store_owned_request(tmp_path, monkeypatch):
    from dataclasses import replace
    from tpen_executor.lifecycle import group_runtime as runtime
    from tpen_executor.lifecycle.group_store import CanonicalGroupBinding, GroupStoreResult
    from tpen_executor.tasks import ConfiguredRun
    from test_b1a_b_publication import _fixture
    from test_s2_core_port import _Worker
    from tpen_executor.lifecycle.core_port import ScienceProcessCoreAdmissionPort
    import json

    store, group, lease, members, bindings, successors, fences = _fixture(tmp_path)
    prepared = _prepared(tmp_path)
    allocation = ConfiguredAllocation(lease.allocation_id, lease.backend_job_id, {})
    controller = ControllerGenerationV1(group.group_id, 1, allocation.allocation_id,
        allocation.backend_job_id, "b" * 64, "nonce", None)
    identity = replace(prepared.identity, group_digest=group.identity)
    wire, acceptance, invocation = successors["successor"]
    wire = store.build_successor_admission_request(fences["successor"], invocation.attempt_id,
        group.members[0].science_digest, group.members[0].target_horizon)
    prepared = replace(prepared, group=group, allocation=allocation, controller=controller,
        member_inputs=members, identity=identity, execution_bundle_digest=identity.bundle_digest,
        runs={"successor": ConfiguredRun("successor", str(tmp_path / "science.yaml"))},
        successors=(runtime.PreparedSuccessor(fences["successor"], invocation, acceptance, digest(wire)),))
    # This fixture already has predecessor evidence. Isolate reopen admission;
    # request construction and science-port wire decoding remain real owners.
    binding = CanonicalGroupBinding(group.identity, identity.bundle_digest,
                                    identity.deployment.content_digest, "b" * 64)
    monkeypatch.setattr(runtime, "reentry", lambda *_args: GroupStoreResult(binding, str(tmp_path / "invocation")))
    captured = []

    class Worker(_Worker):
        def run(self, environment, args, **kwargs):
            captured.append(json.loads(args[2]))
            assert kwargs["timeout"] == 11
            return super().run(environment, args, **kwargs)

    def controller_run(binding, job, facts, store, port, executor):
        actual, accepted, proposed = facts.successor_requests["successor"]
        assert actual == wire
        assert accepted == acceptance
        assert proposed == invocation
        assert actual["predecessor"]["attempt_id"] != actual["successor"]["attempt_id"]
        assert set(actual) == {"schema", "predecessor", "successor", "science_input_digest", "target_iterations", "core_evidence"}
        port.admit(members[0])
        assert captured[0]["successor_request"] == actual
        assert captured[0]["core_run_dir"] == str(Path(members[0].start_receipt.output_dir) / "core")
        assert captured[0]["unresolved_config"] == dict(group.members[0].program.config)
        assert captured[0]["profile_name"] == group.execution_plan.continuation_protocol
        # A collected successor must not be evaluated against its predecessor request.
        port.admit(replace(members[0], attempt_id=invocation.attempt_id, start_receipt=None, outcome_receipt=None))
        assert captured[1]["successor_request"] is None
        return "admitted"

    dispatcher = GroupRuntimeDispatcher(core_port=ScienceProcessCoreAdmissionPort(Worker(None)),
        controller_runner=controller_run, core_timeout=11)
    assert dispatcher.run(PackedGroupRequest(prepared, reentry=True)) == "admitted"


@pytest.mark.parametrize("case", ["completed", "planned", "forbidden-callback"])
def test_bound_adapter_real_science_producer_and_decoder(tmp_path, case):
    """Use the harness's independent science interpreter; never import core here."""
    import json
    import os
    import subprocess
    from dataclasses import replace
    from tpen_executor.contracts import AttemptId
    from tpen_executor.lifecycle.core_port import ScienceProcessCoreAdmissionPort
    from tpen_executor.lifecycle.group_runtime import _BoundCorePort
    from tpen_executor.results import StartReceipt

    python = os.environ.get("P5B_REVIEW_SCIENCE_PYTHON")
    if python is None:
        pytest.skip("independent root --extra cpu environment required")
    root = Path(__file__).resolve().parents[2]
    core = tmp_path / "attempt" / "core"
    core.mkdir(parents=True)
    produce = r'''
import json, sys
from pathlib import Path
from tests.helpers.core_admission import planned_yield_run, request, write_run_start, write_outcome
p = Path(sys.argv[1])
if sys.argv[2] == "completed":
    ref = planned_yield_run(p)
    (p / "status.json").write_text('{"status":"completed"}\n')
    write_outcome(p, "completed", ref)
    print(json.dumps(None))
else:
    ref = planned_yield_run(p)
    print(json.dumps(request(ref, task_id="task", predecessor_attempt_id="attempt").to_dict()))
'''
    output = subprocess.run([python, "-c", produce, str(core), case], cwd=root,
                            capture_output=True, text=True, timeout=60, check=True)
    wire = json.loads(output.stdout)
    prepared = _prepared(tmp_path)
    config = {"study": "cx04"}
    if case == "forbidden-callback":
        config["callbacks"] = [{"_target_": "not.allowed"}]
    task = replace(prepared.group.members[0], program=ProgramSpecRef(config, digest(config)),
                   science_digest=wire["science_input_digest"] if wire else "a" * 64)
    group = replace(prepared.group, members=(task,), execution_plan=replace(
        prepared.group.execution_plan, continuation_protocol="tpen.native-recovery/v1"))
    prepared = replace(prepared, group=group)
    start = StartReceipt("task", AttemptId("attempt"), prepared.allocation.allocation_id,
        prepared.allocation.backend_job_id, group.identity, "utc", str(core.parent))
    member = MemberRecoveryInputV2("task", "lineage", AttemptId("attempt"), start)
    port = _BoundCorePort(prepared, ScienceProcessCoreAdmissionPort(), tmp_path / "admission",
        PreparedPythonEnvironment(python, env={"PYTHONPATH": str(root)}), 30,
        {} if wire is None else {"task": (wire, None, None)})
    verdict = port.admit(member)
    if case == "completed":
        assert verdict.outcome == "completed"
    elif case == "planned":
        assert verdict.outcome == "successor" and verdict.admission == "admitted"
        assert verdict.profile.verdict == "supported"
    else:
        assert verdict.admission == "refused" and verdict.reason == "unsupported-callback"


def test_successor_publishes_launches_once_and_replay_does_not_relaunch(tmp_path, monkeypatch):
    """Real controller/store/local executor with a controlled science-port result."""
    import os
    from dataclasses import replace
    from tpen_executor.lifecycle import group_runtime as runtime
    from tpen_executor.lifecycle.group_store import CanonicalGroupBinding, GroupStoreResult
    from tpen_executor.lifecycle.core_port import ScienceProcessCoreAdmissionPort
    from tpen_executor.lifecycle.local_executor import LocalLeafExecutor
    from tpen_executor.placement import AllocationSlot
    from tpen_executor.tasks import ConfiguredRun
    from tpen_executor.dispatch.transport import LeafRequest
    from test_b1a_b_publication import _fixture
    from test_b1a_reconciliation import _core, _profile
    from test_b1e1_local_executor import _failure_result
    from test_c22_output_binding import _program
    from test_s2_core_port import _Worker

    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0}, raising=False)
    store, group, lease, members, bindings, successors, fences = _fixture(tmp_path)
    prepared = _prepared(tmp_path)
    slot = AllocationSlot("slot", "localhost", (), (0,), (), ())
    allocation = ConfiguredAllocation(lease.allocation_id, lease.backend_job_id,
        {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot.to_dict()]}})
    controller = ControllerGenerationV1(group.group_id, 1, allocation.allocation_id,
        allocation.backend_job_id, "b" * 64, "nonce", None)
    identity = replace(prepared.identity, group_digest=group.identity)
    _, acceptance, invocation = successors["successor"]
    wire = store.build_successor_admission_request(fences["successor"], invocation.attempt_id,
        group.members[0].science_digest, group.members[0].target_horizon)
    prepared = replace(prepared, group=group, allocation=allocation, controller=controller,
        member_inputs=members, identity=identity, execution_bundle_digest=identity.bundle_digest,
        runs={"successor": ConfiguredRun("successor", "", program=_program(tmp_path))},
        successors=(runtime.PreparedSuccessor(fences["successor"], invocation, acceptance, digest(wire)),))
    binding = CanonicalGroupBinding(group.identity, identity.bundle_digest, identity.deployment.content_digest, "b" * 64)
    monkeypatch.setattr(runtime, "reentry", lambda *_args: GroupStoreResult(binding, str(tmp_path / "invocation")))
    launched = []
    def leaf(payload):
        request = LeafRequest.decode(payload)
        launched.append(request)
        return _failure_result(request).encode()
    def factory(*args, **kwargs):
        return LocalLeafExecutor(*args, **kwargs, leaf_runner=leaf)
    verdict = _core("successor", profile=_profile(), admission="admitted", checkpoint="valid", quiescence="unknown")
    dispatcher = GroupRuntimeDispatcher(core_port=ScienceProcessCoreAdmissionPort(_Worker({**verdict.to_dict(), "schema": "tpen-core-admission-v1"})),
        local_executor_factory=factory)
    result = dispatcher.run(PackedGroupRequest(prepared, reentry=True))
    assert len(launched) == 1, result
    assert launched[0].attempt_id == invocation.attempt_id.value
    assert list(Path(store.root).rglob("successor-admission.json"))
    with pytest.raises(runtime.GroupStoreRefused, match="stale"):
        dispatcher.run(PackedGroupRequest(prepared, reentry=True))
    assert len(launched) == 1


def _real_reentry_successor_fixture(tmp_path):
    """Build generation-one predecessor evidence for real group-store reopen tests."""
    from dataclasses import replace
    from tpen_executor.allocations import AllocationAttemptV2, MemberInvocationV2
    from tpen_executor.contracts import AllocationAttemptId, AttemptId, InvocationId
    from tpen_executor.lifecycle.group_runtime import PreparedSuccessor
    from tpen_executor.lifecycle.group_store import InitialGroupStoreRequest, initial
    from tpen_executor.results import OutcomeReceipt, StartReceipt
    from tpen_executor.storage import DirectoryStore
    from tpen_executor.tasks import ConfiguredRun
    from test_b1a_reconciliation import _group
    from test_core_evidence_v3 import _planned_evidence

    group = _group(("successor",))
    prepared = _prepared(tmp_path)
    identity = replace(prepared.identity, group_digest=group.identity)
    allocation_one = ConfiguredAllocation(AllocationId("allocation-1"), BackendJobId("job-1"), {})
    controller_one = ControllerGenerationV1(
        group.group_id, 1, allocation_one.allocation_id, allocation_one.backend_job_id,
        "b" * 64, "nonce-1", None,
    )
    root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, identity.bundle_digest, identity.deployment.content_digest, controller_one), root)
    store = DirectoryStore(root)
    lease_one = AllocationAttemptV2(
        group.group_id, AllocationAttemptId("lease-1"), allocation_one.allocation_id,
        allocation_one.backend_job_id, "controller-1", "utc-1",
    )
    fence = store.begin_attempt(
        group, "successor", allocation_one, "utc-1", "local",
        allocation_attempt=lease_one, invocation_id=InvocationId("invocation-successor"),
    )
    store.finish_attempt(fence, "failed", "utc-2")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    store.record_core_outcome(fence, _planned_evidence(core_dir))

    allocation_two = ConfiguredAllocation(AllocationId("allocation-2"), BackendJobId("job-2"), {})
    controller_two = ControllerGenerationV1(
        group.group_id, 2, allocation_two.allocation_id, allocation_two.backend_job_id,
        "b" * 64, "nonce-2", controller_one.record_digest,
    )
    invocation = MemberInvocationV2(
        AllocationAttemptV2(
            group.group_id, AllocationAttemptId("lease-2"), allocation_two.allocation_id,
            allocation_two.backend_job_id, "controller-2", "utc-3",
        ), group.identity, "successor", group.members[0].lineage_id,
        AttemptId("successor-1"), InvocationId("invocation-successor-1"),
    )
    wire = store.build_successor_admission_request(
        fence, invocation.attempt_id, group.members[0].science_digest, group.members[0].target_horizon,
    )
    start = StartReceipt.decode(Path(fence.output_dir, "start.json").read_bytes())
    outcome = OutcomeReceipt.decode(Path(fence.output_dir, "outcome.json").read_bytes())
    prepared = replace(
        prepared, group=group, allocation=allocation_two, controller=controller_two,
        member_inputs=(MemberRecoveryInputV2("successor", group.members[0].lineage_id, fence.attempt_id,
                                             start, outcome, None),),
        registration_receipt_digest="b" * 64, claimant_nonce="nonce-2",
        predecessor_record_digest=controller_one.record_digest,
        identity=identity, execution_bundle_digest=identity.bundle_digest,
        runs={"successor": ConfiguredRun("successor", str(tmp_path / "successor.yaml"))},
        successors=(PreparedSuccessor(fence, invocation, "d" * 64, digest(wire)),),
    )
    return prepared, store, fence, wire


@pytest.mark.parametrize("damage", ["mismatched-digest", "missing-predecessor"])
def test_successor_preflight_refuses_before_generation_two_publication(tmp_path, damage):
    """Invalid successor evidence cannot publish authority or an invocation."""
    from dataclasses import replace
    from tpen_executor.lifecycle.group_store import GroupStoreRefused

    prepared, store, fence, wire = _real_reentry_successor_fixture(tmp_path)
    proposal = prepared.successors[0]
    if damage == "mismatched-digest":
        proposal = replace(proposal, request_digest=digest({**wire, "target_iterations": 11}))
    else:
        Path(fence.output_dir, "core-outcome.json").unlink()
    prepared = replace(prepared, successors=(proposal,))
    with pytest.raises(GroupStoreRefused):
        GroupRuntimeDispatcher(controller_runner=lambda *_args: pytest.fail("child launch")).run(
            PackedGroupRequest(prepared, reentry=True)
        )
    ledger = next(Path(store.root).rglob("controller.lock")).parent
    assert not (ledger / "generation-2.json").exists()
    assert not (Path(store.root) / "invocations").exists()


def test_successor_real_reentry_publishes_once_and_replay_is_fenced(tmp_path):
    """A valid successor reopens generation two; replay adds no authority."""
    prepared, store, _fence, _wire = _real_reentry_successor_fixture(tmp_path)
    launches = []

    def controller(*_args):
        launches.append(True)
        return "admitted"

    dispatcher = GroupRuntimeDispatcher(controller_runner=controller)
    assert dispatcher.run(PackedGroupRequest(prepared, reentry=True)) == "admitted"
    assert len(launches) == 1
    assert next(Path(store.root).rglob("generation-2.json")).is_file()
    assert dispatcher.run(PackedGroupRequest(prepared, reentry=True)) == "admitted"
    assert len(launches) == 2
    assert len(list(Path(store.root).rglob("generation-2.json"))) == 1


def test_successor_point_of_use_failure_reports_published_authority(tmp_path, monkeypatch):
    """A predecessor race after acquisition is reconciliation-required."""
    from tpen_executor.lifecycle.group_store import GroupStoreAuthorityPublished
    from tpen_executor.storage import DirectoryStore

    prepared, store, fence, _wire = _real_reentry_successor_fixture(tmp_path)
    original = DirectoryStore.build_successor_admission_request
    calls = 0

    def remove_predecessor_on_point_of_use(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            Path(fence.output_dir, "core-outcome.json").unlink()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DirectoryStore, "build_successor_admission_request", remove_predecessor_on_point_of_use)
    with pytest.raises(GroupStoreAuthorityPublished):
        GroupRuntimeDispatcher(controller_runner=lambda *_args: pytest.fail("child launch")).run(
            PackedGroupRequest(prepared, reentry=True)
        )
    assert (next(Path(store.root).rglob("generation-2.json"))).is_file()
