"""Backend-neutral allocation-group controller loop.

The controller owns ordering and fencing only.  It does not import an optional
backend SDK, infer completion from backend observations, or retry an uncertain
launch.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from .._records import frozen_record
from ..allocations import AllocationAttemptV2, ControllerGenerationV1, MemberInvocationV2
from ..contracts import AllocationAttemptId, BackendJobId, InvocationId, ReadyGroupV2
from ..recovery import CoreAdmissionPort, GroupReconciliationV2, MemberRecoveryInputV2, reconcile_group
from ..results import SuccessorAdmissionRecord
from ..storage import AttemptFence, ControllerGenerationRejected, DirectoryStore, LaunchIntentRejected


@frozen_record
class GroupControllerFactsV1:
    """Immutable facts supplied by the registration/controller caller."""

    group: ReadyGroupV2
    allocation: Any
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


def _pre_start(store: DirectoryStore, reason: str, *, group: Any = None, backend_job_id: Any = None, observed_job_id: Any = None) -> Any:
    """Best-effort durable pre-start receipt; ambiguity remains non-success."""
    writer = getattr(store, "record_pre_start_receipt", None)
    if writer is not None:
        try:
            try:
                writer(group, backend_job_id, observed_job_id, reason)
            except TypeError:
                writer(reason)
        except Exception:
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
    if not isinstance(observed_job_id, BackendJobId) or not hasattr(binding, "backend_job_id"):
        return _pre_start(store, "malformed-registration", group=getattr(binding, "intent", None), backend_job_id=getattr(binding, "backend_job_id", None), observed_job_id=observed_job_id)
    if observed_job_id != binding.backend_job_id or allocation.backend_job_id != observed_job_id:
        return _pre_start(store, "backend-job-mismatch", group=group, backend_job_id=binding.backend_job_id, observed_job_id=observed_job_id)
    if binding.intent.group_id != group.group_id or binding.intent.group_digest != group.identity:
        return _pre_start(store, "registration-group-mismatch", group=group, backend_job_id=binding.backend_job_id, observed_job_id=observed_job_id)
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
        for item in facts.member_inputs:
            verdict = admission_port.admit(item)
            frozen_inputs.append(replace(item, core_verdict=verdict))
        checked = reconcile_group(group, tuple(frozen_inputs))
    except Exception:
        return _park(group, "admission-port-failed")
    try:
        _require_current_controller(store, group, controller)
        bindings = {}
        for item in frozen_inputs:
            if item.start_receipt is not None:
                bindings[item.task_id] = MemberInvocationV2(
                    attempt, group.identity, item.task_id,
                    next(member.lineage_id for member in group.members if member.task_id == item.task_id),
                    item.attempt_id, InvocationId(uuid4().hex),
                )
        published = store.publish_reconciliation(
            group, attempt, tuple(frozen_inputs), bindings=bindings,
            successors=facts.successor_requests,
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
                fence = store.begin_attempt(
                    group, member.task_id, allocation, facts.observed_at_utc, "local",
                    allocation_attempt=attempt, invocation_id=invocation_id,
                )
                invocation = MemberInvocationV2(
                    attempt, group.identity, member.task_id,
                    next(task.lineage_id for task in group.members if task.task_id == member.task_id),
                    fence.attempt_id, invocation_id,
                )
            elif member.action == "admit_successor":
                request = facts.successor_requests[member.task_id]
                admission = request[2]
                if not isinstance(admission, MemberInvocationV2):
                    raise TypeError("successor request lacks binding")
                # The durable publication contains the admission; callers may
                # pass the typed record in the request tuple for this seam.
                record = getattr(published, "admissions", {}).get(member.task_id)
                if record is None:
                    record = getattr(store, "read_successor_admission", lambda *_: None)(member.task_id)
                if record is None:
                    publication = getattr(store, "read_reconciliation", lambda *_: None)(group, attempt)
                    if publication is not None:
                        record = next((item.admission for item in publication.members if item.evidence.task_id == member.task_id), None)
                if not isinstance(record, SuccessorAdmissionRecord):
                    record = request[0] if isinstance(request[0], SuccessorAdmissionRecord) else None
                if record is None:
                    raise LaunchIntentRejected("successor admission unavailable")
                fence = store.begin_admitted_successor(
                    group, member.task_id, allocation, facts.observed_at_utc, "local",
                    allocation_attempt=attempt, admission=record,
                )
                invocation = admission
            else:
                outcomes.append((member.task_id, "parked"))
                continue
            try:
                existing = store.read_launch_intent(fence)
                outcomes.append((member.task_id, "parked"))
                continue
            except Exception:
                pass
            store.create_launch_intent(fence, controller, invocation)
            store.require_current_controller(group, controller.generation, controller.claimant_nonce)
            launched = executor.launch(fence, invocation)
            acknowledged = launched.acknowledged if isinstance(launched, MemberLaunchResult) else bool(launched)
            outcomes.append((member.task_id, "launched" if acknowledged else "parked"))
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
