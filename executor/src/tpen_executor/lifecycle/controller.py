"""Backend-neutral allocation-group controller loop.

The controller owns ordering and fencing only.  It does not import an optional
backend SDK, infer completion from backend observations, or repeat an uncertain
launch.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from .._records import frozen_record
from ..allocations import AllocationAttemptV2, ConfiguredAllocation, ControllerGenerationV1, MemberInvocationV2
from ..contracts import AllocationAttemptId, BackendJobId, GroupId, InvocationId, ReadyGroupV2
from ..recovery import CoreAdmissionPort, CoreAdmissionVerdictV2, GroupReconciliationV2, MemberRecoveryInputV2, reconcile_group
from ..results import SuccessorAdmissionRecord
from ..storage import AttemptFence, ControllerGenerationRejected, DirectoryStore, FenceRejected, LaunchIntentAbsent, LaunchIntentRejected


@frozen_record
class GroupControllerFactsV1:
    """Immutable facts supplied by the registration/controller caller."""

    group: ReadyGroupV2
    allocation: ConfiguredAllocation
    registration_receipt_digest: str
    claimant_nonce: str
    observed_at_utc: str
    member_inputs: tuple[MemberRecoveryInputV2, ...]
    successor_requests: Mapping[str, tuple[Mapping[str, Any], str, MemberInvocationV2]] = ()
    controller_generation: int = 1
    predecessor_record_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group, ReadyGroupV2):
            raise TypeError("group must be ReadyGroupV2")
        if not isinstance(self.allocation, ConfiguredAllocation):
            raise TypeError("allocation must be ConfiguredAllocation")
        if not isinstance(self.member_inputs, Sequence):
            raise TypeError("member_inputs must be a sequence")
        object.__setattr__(self, "member_inputs", tuple(self.member_inputs))
        object.__setattr__(self, "successor_requests", dict(self.successor_requests))


@frozen_record
class MemberLaunchResult:
    """Typed operational acknowledgement returned by an injected executor."""

    acknowledged: bool
    status: str = "launched"
    detail: str = ""

    def __post_init__(self) -> None:
        if type(self.acknowledged) is not bool:
            raise TypeError("acknowledged must be bool")


class AllocationLocalExecutor(Protocol):
    """Fake-friendly seam for one already-fenced local launch."""

    def launch(self, fence: AttemptFence, invocation: MemberInvocationV2) -> MemberLaunchResult:
        """Launch exactly once for the supplied durable fence and invocation."""


@frozen_record
class GroupControllerOutcome:
    """Controller result, including operational member outcomes and exit code."""

    reconciliation: GroupReconciliationV2
    member_outcomes: tuple[tuple[str, str], ...]
    exit_code: int


def _pre_start(store: DirectoryStore, reason: str, *, group: ReadyGroupV2 | GroupId, backend_job_id: BackendJobId, observed_job_id: BackendJobId) -> Any:
    """Persist a refusal when possible; storage uncertainty remains PRE_START."""
    try:
        store.record_pre_start_receipt(group, backend_job_id, observed_job_id, reason)
    except (OSError, FenceRejected):
        pass
    from .balsam import GroupFailure

    return GroupFailure.PRE_START


def _park(group: ReadyGroupV2, reason: str) -> GroupControllerOutcome:
    from ..recovery import MemberReconciliationV2

    result = GroupReconciliationV2(group.group_id, group.identity, tuple(
        MemberReconciliationV2(member.task_id, "park", reason, None) for member in group.members
    ))
    return GroupControllerOutcome(result, tuple((member.task_id, "parked") for member in result.members), 2)


def _attempt(group: ReadyGroupV2, allocation: Any, controller: ControllerGenerationV1, stamp: str) -> AllocationAttemptV2:
    return AllocationAttemptV2(
        group.group_id, AllocationAttemptId(uuid4().hex), allocation.allocation_id,
        allocation.backend_job_id, str(controller.generation), stamp,
    )


def _require_current_controller(store: DirectoryStore, group: ReadyGroupV2, controller: ControllerGenerationV1) -> None:
    """Apply one durable authority check at a controller boundary."""
    store.require_current_controller(group, generation=controller.generation, nonce=controller.claimant_nonce)


def run_group_controller(
    binding: Any,
    observed_job_id: BackendJobId,
    allocation_facts: GroupControllerFactsV1,
    store: DirectoryStore,
    admission_port: CoreAdmissionPort,
    executor: AllocationLocalExecutor,
) -> GroupControllerOutcome | GroupFailure:
    """Run the ordered, fail-closed group controller state machine.

    A recovered launch intent is permanent evidence of an uncertain external
    call: it parks the member and never invokes ``executor`` again.  The final
    authority check is intentionally immediately before the injected external
    call; that call is not atomic with the filesystem lock and the residual
    window is therefore documented rather than hidden.
    """
    facts = allocation_facts
    group = facts.group
    allocation = facts.allocation
    if not isinstance(observed_job_id, BackendJobId):
        return _pre_start(store, "malformed-registration", group=group, backend_job_id=allocation.backend_job_id, observed_job_id=observed_job_id)
    try:
        bound_job_id = binding.backend_job_id
        bound_group_id = binding.intent.group_id
        bound_group_digest = binding.intent.group_digest
    except AttributeError:
        return _pre_start(store, "malformed-registration", group=group, backend_job_id=allocation.backend_job_id, observed_job_id=observed_job_id)
    if observed_job_id != bound_job_id or allocation.backend_job_id != observed_job_id:
        return _pre_start(store, "backend-job-mismatch", group=group, backend_job_id=bound_job_id, observed_job_id=observed_job_id)
    if bound_group_id != group.group_id or bound_group_digest != group.identity:
        return _pre_start(store, "registration-group-mismatch", group=group, backend_job_id=bound_job_id, observed_job_id=observed_job_id)
    try:
        controller_request = ControllerGenerationV1(
            group.group_id, facts.controller_generation, allocation.allocation_id, observed_job_id,
            facts.registration_receipt_digest, facts.claimant_nonce, facts.predecessor_record_digest,
        )
        controller = store.acquire_controller_generation(group, controller_request)
        attempt = _attempt(group, allocation, controller, facts.observed_at_utc)
    except (ControllerGenerationRejected, TypeError, ValueError, AttributeError, OSError):
        return _park(group, "controller-authority-refused")
    try:
        frozen_inputs = []
        admission_failures = set()
        for item in facts.member_inputs:
            try:
                verdict = admission_port.admit(item)
                if verdict is not None and not isinstance(verdict, CoreAdmissionVerdictV2):
                    admission_failures.add(item.task_id)
                    verdict = CoreAdmissionVerdictV2("unknown", "unknown", "unknown", "unknown", None, "malformed-core-admission")
            except Exception:
                admission_failures.add(item.task_id)
                verdict = CoreAdmissionVerdictV2("unknown", "unknown", "unknown", "unknown", None, "core-admission-unavailable")
            frozen_inputs.append(replace(item, core_verdict=verdict))
        checked = reconcile_group(group, tuple(frozen_inputs))
        if admission_failures:
            checked = replace(checked, members=tuple(
                replace(member, action="park", reason_code="core-admission-unavailable")
                if member.task_id in admission_failures else member for member in checked.members
            ))
    except Exception:
        return _park(group, "admission-port-failed")
    try:
        _require_current_controller(store, group, controller)
        bindings = {}
        for item in frozen_inputs:
            if item.start_receipt is not None:
                invocation_path = Path(item.start_receipt.output_dir) / "invocation.json"
                bindings[item.task_id] = MemberInvocationV2.decode(invocation_path.read_bytes())
        successor_requests = {}
        for task_id, (request, acceptance, proposed) in facts.successor_requests.items():
            successor_requests[task_id] = (request, acceptance, MemberInvocationV2(
                attempt, group.identity, task_id,
                next(task.lineage_id for task in group.members if task.task_id == task_id),
                proposed.attempt_id, InvocationId(uuid4().hex),
            ))
        published = store.publish_reconciliation(
            group, attempt, tuple(frozen_inputs), bindings=bindings,
            successors=successor_requests,
        )
        if published.encode() != checked.encode():
            return _park(group, "reconciliation-publication-conflict")
    except Exception:
        return _park(group, "publication-failed")
    outcomes = []
    for member in published.members:
        if member.action == "preserve_completed" or member.action == "preserve_terminal":
            outcomes.append((member.task_id, member.action))
            continue
        try:
            if member.action == "start_fresh":
                invocation_id = InvocationId(uuid4().hex)
                fence = store.begin_attempt_current_controller(
                    group, member.task_id, allocation, facts.observed_at_utc, "local",
                    allocation_attempt=attempt, invocation_id=invocation_id, controller_record=controller,
                )
                invocation = MemberInvocationV2(
                    attempt, group.identity, member.task_id,
                    next(task.lineage_id for task in group.members if task.task_id == member.task_id),
                    fence.attempt_id, invocation_id,
                )
            elif member.action == "admit_successor":
                publication = store.read_reconciliation(group, attempt)
                if publication is None:
                    raise LaunchIntentRejected("successor admission unavailable")
                record = next((item.admission for item in publication.members if item.evidence.task_id == member.task_id), None)
                if not isinstance(record, SuccessorAdmissionRecord) or record.successor_binding is None:
                    raise LaunchIntentRejected("successor admission unavailable")
                fence = store.begin_admitted_successor_current_controller(
                    group, member.task_id, allocation, facts.observed_at_utc, "local",
                    allocation_attempt=record.successor_binding.allocation_attempt, admission=record,
                    controller_record=controller,
                )
                invocation = record.successor_binding
            else:
                outcomes.append((member.task_id, "parked"))
                continue
            try:
                existing = store.read_launch_intent(fence)
                outcomes.append((member.task_id, "parked"))
                continue
            except LaunchIntentAbsent:
                pass
            except Exception:
                outcomes.append((member.task_id, "parked"))
                continue
            store.create_launch_intent(fence, controller, invocation)
            store.require_current_controller(group, controller.generation, controller.claimant_nonce)
            try:
                launched = executor.launch(fence, invocation)
                if not isinstance(launched, MemberLaunchResult) or not launched.acknowledged:
                    store.record_controller_outcome(fence, controller, invocation, "unacknowledged", "invalid-launch-ack")
                    outcomes.append((member.task_id, "parked"))
                else:
                    store.record_controller_outcome(fence, controller, invocation, "launched", launched.detail)
                    outcomes.append((member.task_id, "launched"))
            except Exception:
                try:
                    store.record_controller_outcome(fence, controller, invocation, "unacknowledged", "executor-failed")
                except (OSError, FenceRejected):
                    pass
                outcomes.append((member.task_id, "parked"))
        except Exception:
            outcomes.append((member.task_id, "parked"))
    parked = {task_id for task_id, status in outcomes if status == "parked"}
    final = published if not parked else replace(
        published,
        members=tuple(
            replace(member, action="park", reason_code="controller-uncertainty") if member.task_id in parked else member
            for member in published.members
        ),
    )
    return GroupControllerOutcome(final, tuple(outcomes), 0 if all(member.action == "preserve_completed" for member in final.members) else 2)


__all__ = ["AllocationLocalExecutor", "GroupControllerFactsV1", "GroupControllerOutcome", "MemberLaunchResult", "run_group_controller"]
