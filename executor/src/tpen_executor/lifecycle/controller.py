"""Backend-neutral allocation-group controller loop.

The controller owns ordering and fencing only.  It does not import an optional
backend SDK, infer completion from backend observations, or repeat an uncertain
launch.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import math
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
    poll_timeout: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.group, ReadyGroupV2):
            raise TypeError("group must be ReadyGroupV2")
        if not isinstance(self.allocation, ConfiguredAllocation):
            raise TypeError("allocation must be ConfiguredAllocation")
        if not isinstance(self.member_inputs, Sequence):
            raise TypeError("member_inputs must be a sequence")
        object.__setattr__(self, "member_inputs", tuple(self.member_inputs))
        object.__setattr__(self, "successor_requests", dict(self.successor_requests))
        if type(self.poll_timeout) not in (int, float) or not math.isfinite(self.poll_timeout) or self.poll_timeout <= 0:
            raise ValueError("poll_timeout must be a positive finite number")


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

    def submit(self, fence: AttemptFence, invocation: MemberInvocationV2) -> tuple[MemberLaunchResult, Any]:
        """Submit exactly once and return an acknowledgement plus opaque handle."""

    def collect(self, handle: Any, timeout: float) -> Any:
        """Collect one submitted handle with a finite poll timeout."""


@frozen_record
class GroupControllerOutcome:
    """Controller result, including operational member outcomes and exit code."""

    reconciliation: GroupReconciliationV2
    member_outcomes: tuple[tuple[str, str], ...]
    exit_code: int
    member_reasons: tuple[tuple[str, str], ...] = ()


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


def _reason(phase: str, error: BaseException | None = None) -> str:
    """Return a compact typed diagnostic for one member transition."""
    if error is None:
        return phase
    detail = str(error).strip() or "no-message"
    return f"{phase}:{type(error).__name__}:{detail}"


def _merge_final_reconciliation(
    published: GroupReconciliationV2,
    receipt_reconciliation: GroupReconciliationV2 | None,
    parked_this_run: Mapping[str, str],
    collected_members: set[str],
    fallback_reason: str = "final-reconciliation-unavailable",
) -> GroupReconciliationV2:
    """Merge receipt-derived actions with this run's controller decisions.

    Only members that were submitted and collected may be replaced by the
    receipt-derived reconciliation.  A park recorded during this invocation
    remains authoritative even when its absent receipts would otherwise make
    ``reconcile_group`` derive a fresh launch or successor admission.
    """

    receipt_members = {} if receipt_reconciliation is None else {
        member.task_id: member for member in receipt_reconciliation.members
    }
    members = []
    for member in published.members:
        if member.task_id in parked_this_run:
            members.append(replace(
                member, action="park", reason_code=parked_this_run[member.task_id],
            ))
        elif member.task_id in collected_members:
            members.append(receipt_members.get(
                member.task_id,
                replace(member, action="park", reason_code=fallback_reason),
            ))
        else:
            members.append(member)
    return replace(published, members=tuple(members))


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
        launch_attempt = AllocationAttemptV2(
            group.group_id, AllocationAttemptId(uuid4().hex), allocation.allocation_id,
            allocation.backend_job_id, str(controller.generation), facts.observed_at_utc,
        )
    except (ControllerGenerationRejected, TypeError, ValueError, AttributeError, OSError):
        return _park(group, "controller-authority-refused")
    try:
        frozen_inputs = []
        for item in facts.member_inputs:
            try:
                verdict = admission_port.admit(item)
                if verdict is not None and not isinstance(verdict, CoreAdmissionVerdictV2):
                    verdict = CoreAdmissionVerdictV2("unknown", "unknown", "unknown", "unknown", None, "malformed-core-admission")
            except Exception:
                verdict = CoreAdmissionVerdictV2("unknown", "unknown", "unknown", "unknown", None, "core-admission-unavailable")
            frozen_inputs.append(replace(item, core_verdict=verdict))
        checked = reconcile_group(group, tuple(frozen_inputs))
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
            if next(member.action for member in checked.members if member.task_id == task_id) != "admit_successor":
                continue
            successor_requests[task_id] = (request, acceptance, MemberInvocationV2(
                launch_attempt, group.identity, task_id,
                next(task.lineage_id for task in group.members if task.task_id == task_id),
                proposed.attempt_id, InvocationId(uuid4().hex),
            ))
        published = store.publish_reconciliation(
            group, attempt, tuple(frozen_inputs), bindings=bindings,
            successors=successor_requests,
        )
        if published.encode() != checked.encode():
            return _park(group, "reconciliation-publication-conflict")
        durable_publication = store.read_reconciliation(group, attempt)
        if durable_publication is None or durable_publication.reconciliation.encode() != checked.encode():
            return _park(group, "reconciliation-publication-unavailable")
    except Exception:
        return _park(group, "publication-failed")
    # Importing these adapters here keeps the controller protocol backend-neutral
    # and avoids an import cycle through the local executor's handle type.
    from .attached import AttachedAllocation
    from ..dispatch.completion import complete_leaf_result
    from ..dispatch.transport import LeafResultRejected, validate_leaf_result
    from .local_executor import CollectionWaitTimeout, LeafCompletionContext

    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    outcomes: list[tuple[str, str]] = []
    reasons: dict[str, str] = {}
    parked_this_run: dict[str, str] = {}
    collected_members: set[str] = set()
    active: list[tuple[Any, Any, Any, Any, Any, bool]] = []

    def _park_member(task_id: str, phase: str, error: BaseException | None = None) -> None:
        outcomes.append((task_id, "parked"))
        reason = _reason(phase, error)
        reasons[task_id] = reason
        parked_this_run[task_id] = reason

    for member in published.members:
        if member.action in {"preserve_completed", "preserve_terminal"}:
            outcomes.append((member.task_id, member.action))
            continue
        if member.action not in {"start_fresh", "admit_successor"}:
            _park_member(member.task_id, "reconcile-action")
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
            else:
                record = next((item.admission for item in durable_publication.members if item.evidence.task_id == member.task_id), None)
                if not isinstance(record, SuccessorAdmissionRecord) or record.successor_binding is None:
                    raise LaunchIntentRejected("successor admission unavailable")
                fence = store.begin_admitted_successor_current_controller(
                    group, member.task_id, allocation, facts.observed_at_utc, "local",
                    allocation_attempt=record.successor_binding.allocation_attempt, admission=record,
                    controller_record=controller,
                )
                invocation = record.successor_binding
            try:
                existing = store.read_launch_intent(fence)
            except LaunchIntentAbsent:
                existing = None
            except Exception as error:
                _park_member(member.task_id, "intent-read", error)
                continue
            if existing is not None:
                _park_member(member.task_id, "intent-recovered")
                continue
            store.create_launch_intent(fence, controller, invocation)
            store.require_current_controller(group, controller.generation, controller.claimant_nonce)
            try:
                launched, handle = executor.submit(fence, invocation)
            except Exception as error:
                try:
                    store.record_controller_outcome(fence, controller, invocation, "unacknowledged", "submit-failed")
                except Exception:
                    pass
                _park_member(member.task_id, "executor-launch", error)
                continue
            if not isinstance(launched, MemberLaunchResult) or not launched.acknowledged:
                try:
                    store.record_controller_outcome(fence, controller, invocation, "unacknowledged", "invalid-submit-ack")
                except Exception:
                    pass
                _park_member(member.task_id, "launch-ack")
                continue
            # The handle is retained even if this acknowledgement write fails.
            try:
                store.record_controller_outcome(fence, controller, invocation, "launched", launched.detail or "acknowledged")
            except Exception as error:
                reasons[member.task_id] = _reason("outcome-receipt", error)
            active.append((member, fence, invocation, handle, controller, member.action == "admit_successor"))
        except Exception as error:
            _park_member(member.task_id, "member-transition", error)
        except BaseException:
            try:
                AttachedAllocation(allocation, store, invocation.allocation_attempt).finish(
                    fence, "failed", _now(), None,
                )
            except Exception:
                pass
            raise

    while active:
        member, fence, invocation, handle, owner, successor = active.pop(0)
        try:
            result = executor.collect(handle, facts.poll_timeout)
        except CollectionWaitTimeout:
            # Polling is the only repeatable operation; keep this same handle (never resubmit).
            active.append((member, fence, invocation, handle, owner, successor))
            continue
        except BaseException as error:
            try:
                AttachedAllocation(allocation, store, invocation.allocation_attempt).finish(
                    fence, "remote-quiescence-unconfirmed", _now(), None,
                )
            except Exception:
                pass
            _park_member(member.task_id, "collection", error)
            if not isinstance(error, Exception):
                for _, pending_fence, pending_invocation, _, _, _ in active:
                    try:
                        AttachedAllocation(allocation, store, pending_invocation.allocation_attempt).finish(
                            pending_fence, "remote-quiescence-unconfirmed", _now(), None,
                        )
                    except Exception:
                        pass
                raise
            continue
        if not isinstance(result, LeafCompletionContext):
            error = LeafResultRejected("collector returned a non-LeafResult")
            try:
                AttachedAllocation(allocation, store, invocation.allocation_attempt).finish(
                    fence, "remote-quiescence-unconfirmed", _now(), None,
                )
            except Exception:
                pass
            _park_member(member.task_id, "validation", error)
            continue
        try:
            task = next(task for task in group.members if task.task_id == member.task_id)
            try:
                validated = validate_leaf_result(result.request, result.result)
            except (TypeError, LeafResultRejected) as error:
                try:
                    AttachedAllocation(allocation, store, invocation.allocation_attempt).finish(
                        fence, "remote-quiescence-unconfirmed", _now(), None,
                    )
                except Exception:
                    pass
                _park_member(member.task_id, "validation", error)
                continue
            error = complete_leaf_result(
                AttachedAllocation(allocation, store, invocation.allocation_attempt),
                fence, result.request, validated, group=group, member=task,
                configured=result.configured, environment=result.environment,
                accept_segment=not successor,
            )
            if error is not None:
                raise error
            outcomes.append((member.task_id, "collected"))
            collected_members.add(member.task_id)
        except BaseException as error:
            _park_member(member.task_id, "completion", error)
            if not isinstance(error, Exception):
                try:
                    AttachedAllocation(allocation, store, invocation.allocation_attempt).finish(
                        fence, "failed", _now(), None,
                    )
                except Exception:
                    pass
                for _, pending_fence, pending_invocation, _, _, _ in active:
                    try:
                        AttachedAllocation(allocation, store, pending_invocation.allocation_attempt).finish(
                            pending_fence, "remote-quiescence-unconfirmed", _now(), None,
                        )
                    except Exception:
                        pass
                raise

    # Rebuild the exact member inputs from the durable manifest and ask the
    # admission port again before deriving the authoritative final action.
    try:
        manifest = store.manifest(group)
        latest: dict[str, Any] = {}
        for record in manifest.attempts:
            latest[record.task_id] = record
        final_inputs = []
        for item in facts.member_inputs:
            if item.task_id not in collected_members:
                continue
            record = latest.get(item.task_id)
            rebuilt = item
            if record is not None:
                rebuilt = replace(item, attempt_id=record.attempt_id, start_receipt=record.start_receipt,
                                  outcome_receipt=record.outcome_receipt)
            try:
                verdict = admission_port.admit(rebuilt)
                if verdict is not None and not isinstance(verdict, CoreAdmissionVerdictV2):
                    verdict = CoreAdmissionVerdictV2("unknown", "unknown", "unknown", "unknown", None, "malformed-core-admission")
            except Exception:
                verdict = CoreAdmissionVerdictV2("unknown", "unknown", "unknown", "unknown", None, "core-admission-unavailable")
            final_inputs.append(replace(rebuilt, core_verdict=verdict))
        final = _merge_final_reconciliation(
            published, reconcile_group(group, tuple(final_inputs)),
            parked_this_run, collected_members,
        )
    except Exception as error:
        final = _merge_final_reconciliation(
            published, None, parked_this_run, collected_members,
            _reason("final-reconciliation", error),
        )
        reasons.setdefault("<group>", _reason("final-reconciliation", error))
    reason_records = tuple((task_id, reasons.get(task_id, "collected" if status == "collected" else "controller-uncertainty"))
                           for task_id, status in outcomes)
    return GroupControllerOutcome(final, tuple(outcomes), 0 if all(member.action == "preserve_completed" for member in final.members) else 2, reason_records)


__all__ = ["AllocationLocalExecutor", "GroupControllerFactsV1", "GroupControllerOutcome", "MemberLaunchResult", "run_group_controller"]
