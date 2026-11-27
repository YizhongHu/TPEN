"""Typed composition of the group-store, core-port, and local leaf seams.

This module is deliberately a small adapter.  It owns the outer binding for a
group and turns the science-process port into the legacy, member-shaped port
accepted by :func:`run_group_controller`; controller fencing and receipt
publication remain in their existing owners.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from uuid import uuid4
from typing import TYPE_CHECKING, Any, Callable

from .._records import frozen_record
from .._codec import digest, sha256_text
from ..allocations import ConfiguredAllocation, ControllerGenerationV1, MemberInvocationV2
from ..contracts import BackendJobId, GroupId, ProgramSpecRef, ReadyGroupV2
from ..placement import physical_slots_from_allocation
from ..environments import PreparedPythonEnvironment
from ..storage import AttemptFence, DirectoryStore, FenceRejected
from ..tasks import ConfiguredRun
from ..recovery import CoreAdmissionPort, CoreAdmissionUnavailable, MemberRecoveryInputV2
from .controller import GroupControllerFactsV1, GroupControllerOutcome, run_group_controller
from .core_port import CoreAdmissionProcessRequest, ScienceProcessCoreAdmissionPort
from .group_store import (
    GroupStoreAuthorityPublished,
    GroupStoreRefused,
    InitialGroupStoreRequest,
    ReentryGroupStoreRequest,
    initial,
    reentry,
)

if TYPE_CHECKING:
    from ..programs import MaterializedProgram


@frozen_record
class PreparedRuntimeIdentity:
    """Immutable bundle and deployment owners captured before dispatch.

    The bundle is bound to the group and both unresolved program specs. The
    deployment reference holds the deployment configuration, not a free label.
    """

    group_digest: str
    allocation_program: ProgramSpecRef
    worker_program: ProgramSpecRef
    deployment: ProgramSpecRef

    def __post_init__(self) -> None:
        sha256_text(self.group_digest, "group_digest")
        if any(not isinstance(ref, ProgramSpecRef) for ref in
               (self.allocation_program, self.worker_program, self.deployment)):
            raise TypeError("runtime identity requires immutable program references")

    @property
    def bundle_digest(self) -> str:
        return digest({"group_digest": self.group_digest,
                       "allocation_program": self.allocation_program.to_dict(),
                       "worker_program": self.worker_program.to_dict()})


@frozen_record
class PreparedSuccessor:
    """Owner-produced successor proposal and core restore acceptance.

    ``acceptance_digest`` is the science owner's accepted OperationalRestoreRequest
    digest, not a digest of the six-field admission verdict.
    """

    predecessor: AttemptFence
    invocation: MemberInvocationV2
    acceptance_digest: str
    request_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.predecessor, AttemptFence) or not isinstance(self.invocation, MemberInvocationV2):
            raise TypeError("successor requires typed fence and invocation")
        sha256_text(self.acceptance_digest, "acceptance_digest")
        sha256_text(self.request_digest, "request_digest")
        if self.predecessor.attempt_id == self.invocation.attempt_id:
            raise ValueError("successor must be a distinct attempt")


@frozen_record
class PreparedGroupInput:
    """Immutable owner-process inputs for one admitted group.

    ``env`` is intentionally explicit: it is passed unchanged to both program
    materializers and is never read from process-global state.
    ``timeout`` and ``core_timeout`` are validated legacy inputs; dispatcher
    constructor deadlines are authoritative for all calls.
    """

    group: ReadyGroupV2
    allocation: ConfiguredAllocation
    controller: ControllerGenerationV1
    member_inputs: tuple[MemberRecoveryInputV2, ...]
    allocation_program: Any
    worker_program: Any
    env: Mapping[str, Any]
    store_root: str
    execution_bundle_digest: str
    deployment_lineage: str
    registration_receipt_digest: str
    claimant_nonce: str
    observed_at_utc: str
    runs: Mapping[str, ConfiguredRun] | None = None
    predecessor_record_digest: str | None = None
    timeout: float = 30.0
    core_timeout: float = 30.0
    identity: PreparedRuntimeIdentity | None = None
    successors: tuple[PreparedSuccessor, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.group, ReadyGroupV2) or not isinstance(self.allocation, ConfiguredAllocation):
            raise TypeError("group and allocation must be typed records")
        if not isinstance(self.controller, ControllerGenerationV1):
            raise TypeError("controller must be ControllerGenerationV1")
        if (self.controller.group_id != self.group.group_id
                or self.controller.allocation_id != self.allocation.allocation_id
                or self.controller.backend_job_id != self.allocation.backend_job_id):
            raise ValueError("controller crosses group or allocation identity")
        if (self.registration_receipt_digest != self.controller.registration_receipt_digest
                or self.claimant_nonce != self.controller.claimant_nonce
                or self.predecessor_record_digest != self.controller.predecessor_record_digest):
            raise ValueError("controller authority facts differ")
        if any(not isinstance(proposal, PreparedSuccessor) for proposal in self.successors):
            raise TypeError("successors must contain PreparedSuccessor records")
        object.__setattr__(self, "successors", tuple(self.successors))
        members = tuple(self.member_inputs)
        if len(members) != len(self.group.members):
            raise ValueError("member_inputs must cover every group member")
        if {item.task_id for item in members} != {item.task_id for item in self.group.members}:
            raise ValueError("member_inputs do not match group members")
        by_task = {task.task_id: task for task in self.group.members}
        if any(member.lineage_id != by_task[member.task_id].lineage_id for member in members):
            raise ValueError("member_inputs cross group lineage")
        if any(not isinstance(task.program.config, Mapping) for task in self.group.members):
            raise ValueError("member program must own an unresolved science mapping")
        proposed_tasks = set()
        for proposal in self.successors:
            fence, invocation = proposal.predecessor, proposal.invocation
            member = next((item for item in members if item.task_id == fence.task_id), None)
            task = by_task.get(fence.task_id)
            if (member is None or task is None or fence.root != self.store_root
                    or fence.group_digest != self.group.identity
                    or member.attempt_id != fence.attempt_id
                    or invocation.task_id != task.task_id or invocation.lineage_id != task.lineage_id
                    or invocation.group_digest != self.group.identity
                    or task.task_id in proposed_tasks):
                raise ValueError("successor proposal crosses immutable member")
            proposed_tasks.add(task.task_id)
        if not isinstance(self.env, Mapping):
            raise TypeError("env must be a mapping")
        if self.runs is not None:
            if not isinstance(self.runs, Mapping) or set(self.runs) != {item.task_id for item in self.group.members}:
                raise ValueError("runs must cover every group member")
            if any(not isinstance(value, ConfiguredRun) for value in self.runs.values()):
                raise TypeError("runs must contain ConfiguredRun records")
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if type(self.core_timeout) not in (int, float) or not math.isfinite(self.core_timeout) or self.core_timeout <= 0:
            raise ValueError("core_timeout must be positive")

    def materialize(self) -> tuple[MaterializedProgram, MaterializedProgram]:
        """Materialize allocation and worker programs with the same explicit env."""
        if not callable(getattr(self.allocation_program, "materialize", None)):
            raise GroupStoreRefused("allocation program is not materializable")
        if not callable(getattr(self.worker_program, "materialize", None)):
            raise GroupStoreRefused("worker program is not materializable")
        allocation = self.allocation_program.materialize(env=self.env)
        worker = self.worker_program.materialize(env=self.env)
        return allocation, worker


@frozen_record
class PackedGroupRequest:
    """The closed request consumed by :class:`GroupRuntimeDispatcher`."""

    prepared: PreparedGroupInput
    reentry: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedGroupInput):
            raise TypeError("prepared must be PreparedGroupInput")


@frozen_record
class _ControllerIntent:
    """Controller-facing identity derived from the admitted group."""

    group_id: GroupId
    group_digest: str


@frozen_record
class _ControllerBinding:
    """Backend-neutral projection of the admitted registration owners."""

    intent: _ControllerIntent
    backend_job_id: BackendJobId


class _BoundCorePort(CoreAdmissionPort):
    """Bind immutable task/group facts before entering the science process."""

    def __init__(self, prepared: PreparedGroupInput, port: ScienceProcessCoreAdmissionPort, output_root: Path,
                 worker_environment: PreparedPythonEnvironment, core_timeout: float,
                 successor_requests: Mapping[str, Any] | None = None) -> None:
        self._prepared, self._port, self._output_root = prepared, port, output_root
        self._worker_environment = worker_environment
        self._core_timeout = core_timeout
        self._successor_requests = {} if successor_requests is None else successor_requests

    def admit(self, member: MemberRecoveryInputV2) -> Any:
        task = next((item for item in self._prepared.group.members if item.task_id == member.task_id), None)
        if task is None or task.lineage_id != member.lineage_id:
            raise CoreAdmissionUnavailable("core request crosses immutable group member")
        proposal = self._successor_requests.get(member.task_id)
        successor_request = None
        if proposal is not None and proposal[0]["predecessor"]["attempt_id"] == (
                None if member.attempt_id is None else member.attempt_id.value):
            successor_request = proposal[0]
        request = CoreAdmissionProcessRequest(
            task.task_id, task.lineage_id,
            None if member.start_receipt is None else str(Path(member.start_receipt.output_dir) / "core"),
            task.science_digest, successor_request, task.program.config,
            self._prepared.group.execution_plan.continuation_protocol,
            self._worker_environment,
        )
        base = self._output_root / member.task_id / uuid4().hex
        base.mkdir(parents=True, exist_ok=False)
        return self._port.admit(
            request, member,
            stdout_path=base / "stdout.log",
            stderr_path=base / "stderr.log",
            cwd=base,
            timeout=self._core_timeout,
        )


class GroupRuntimeDispatcher:
    """Compose one immutable group through store, controller, and local leaves."""

    def __init__(self, *, core_port: ScienceProcessCoreAdmissionPort | None = None,
                 controller_runner: Callable[..., Any] = run_group_controller,
                 local_executor_factory: Callable[..., Any] | None = None,
                 timeout: float = 30.0, core_timeout: float = 30.0) -> None:
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
                or type(core_timeout) not in (int, float) or not math.isfinite(core_timeout) or core_timeout <= 0):
            raise ValueError("timeouts must be positive")
        if local_executor_factory is None:
            from .local_executor import LocalLeafExecutor

            local_executor_factory = LocalLeafExecutor
        self.core_port = ScienceProcessCoreAdmissionPort() if core_port is None else core_port
        self.controller_runner = controller_runner
        self.local_executor_factory = local_executor_factory
        self.timeout = timeout
        self.core_timeout = core_timeout

    def run(self, request: PackedGroupRequest) -> GroupControllerOutcome | Any:
        prepared = request.prepared
        # Validate immutable admission inputs before publishing store authority.
        prepared.__post_init__()
        identity = prepared.identity
        if not isinstance(identity, PreparedRuntimeIdentity):
            raise GroupStoreRefused("group runtime requires immutable bundle/deployment owners")
        if (identity.group_digest != prepared.group.identity
                or identity.allocation_program != getattr(prepared.allocation_program, "spec", None)
                or identity.worker_program != getattr(prepared.worker_program, "spec", None)
                or identity.bundle_digest != prepared.execution_bundle_digest
                or identity.deployment.content_digest != prepared.deployment_lineage):
            raise GroupStoreRefused("runtime identity crosses bundle or deployment owners")
        slots = tuple(physical_slots_from_allocation(prepared.allocation))
        runs = prepared.runs
        if runs is None:
            raise GroupStoreRefused("group runtime requires owner-materialized configured runs")
        allocation_program, worker_program = prepared.materialize()
        if "dispatcher" not in allocation_program:
            raise GroupStoreRefused("allocation materialization lacks dispatcher")
        if not callable(getattr(allocation_program.dispatcher, "run", None)):
            raise GroupStoreRefused("allocation materialization dispatcher is not runnable")
        store_result = reentry(
            ReentryGroupStoreRequest(prepared.group, prepared.execution_bundle_digest,
                                     prepared.deployment_lineage, prepared.controller),
            prepared.store_root,
        ) if request.reentry else initial(
            InitialGroupStoreRequest(prepared.group, prepared.execution_bundle_digest,
                                     prepared.deployment_lineage, prepared.controller),
            prepared.store_root,
        )
        store = DirectoryStore(prepared.store_root)
        successor_requests = {}
        for proposal in prepared.successors:
            fence, invocation = proposal.predecessor, proposal.invocation
            task = next(item for item in prepared.group.members if item.task_id == fence.task_id)
            try:
                wire = store.build_successor_admission_request(
                    fence, invocation.attempt_id, task.science_digest, task.target_horizon)
            except (FenceRejected, OSError, TypeError, ValueError) as error:
                raise GroupStoreRefused("successor predecessor evidence is unavailable or stale") from error
            if digest(wire) != proposal.request_digest:
                raise GroupStoreRefused("successor acceptance binds a different request")
            successor_requests[task.task_id] = (wire, proposal.acceptance_digest, invocation)
        output_root = Path(store_result.invocation_result_path or prepared.store_root) / "core-admission"
        facts = GroupControllerFactsV1(
            prepared.group, prepared.allocation, prepared.controller.registration_receipt_digest,
            prepared.controller.claimant_nonce, prepared.observed_at_utc, prepared.member_inputs,
            successor_requests=successor_requests,
            controller_generation=prepared.controller.generation,
            predecessor_record_digest=prepared.controller.predecessor_record_digest,
            poll_timeout=self.timeout,
        )
        executor = self.local_executor_factory(
            prepared.group, prepared.allocation, runs, worker_program.environment,
            capacity=len(slots) or 1, timeout=self.timeout,
        )
        try:
            return self.controller_runner(
                _ControllerBinding(_ControllerIntent(prepared.group.group_id, store_result.binding.group_digest),
                                   prepared.controller.backend_job_id), prepared.allocation.backend_job_id, facts,
                store,
                _BoundCorePort(prepared, self.core_port, output_root, worker_program.environment, self.core_timeout, successor_requests), executor,
            )
        finally:
            close = getattr(executor, "close", None)
            if callable(close):
                close()


__all__ = ["GroupRuntimeDispatcher", "PackedGroupRequest", "PreparedGroupInput",
           "PreparedRuntimeIdentity", "PreparedSuccessor"]
