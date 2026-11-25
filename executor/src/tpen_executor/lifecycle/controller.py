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
from ..recovery import CoreAdmissionPort, CoreAdmissionVerdictV2, GroupReconciliationV2, MemberRecoveryInputV2, admit_recovery, reconcile_group
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

    capacity: int

    def submit(self, fence: AttemptFence, invocation: MemberInvocationV2) -> tuple[MemberLaunchResult, Any]:
        """Submit exactly once and return an acknowledgement plus opaque handle."""

    def submit_admitted_successor(
        self,
        fence: AttemptFence,
        invocation: MemberInvocationV2,
        checkpoint_path: str,
    ) -> tuple[MemberLaunchResult, Any]:
        """Submit an admitted successor with its exact C2.2 load path."""

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


def _collected_attempt_record(
    attempts: Sequence[Any], task_id: str, attempt_id: Any,
) -> Any | None:
    """Select the manifest record for one exact collected fence."""
    return next((record for record in attempts
                 if record.task_id == task_id and record.attempt_id == attempt_id), None)


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


def _apply_recovery_admission(item: MemberRecoveryInputV2, store: DirectoryStore) -> MemberRecoveryInputV2:
    """Evaluate real R2 facts against the live store root before reconcile."""

    request = item.recovery_input
    if request is None:
        return item
    admission = admit_recovery(
        request.evidence,
        request.successor_attempt_id,
        now_utc=request.now_utc,
        trusted_store_root=store.root,
        existing_successors=request.existing_successors,
    )
    # A parked R2 result has no successor identity.  Clear the raw input before
    # attaching it so MemberRecoveryInputV2 does not reject the typed park as a
    # crossed recovery input; the controller still uses the original input to
    # select the R2 path and never calls the legacy admission port.
    return replace(
        item,
        recovery_input=None,
        recovery_admission=admission,
        recovery_evidence=request.evidence,
    )


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
                admitted_item = _apply_recovery_admission(item, store)
                # R2 facts are the complete recovery decision.  The legacy
                # core selector remains available only to legacy callers.
                verdict = None if item.recovery_input is not None else admission_port.admit(admitted_item)
                if verdict is not None and not isinstance(verdict, CoreAdmissionVerdictV2):
                    verdict = CoreAdmissionVerdictV2("unknown", "unknown", "unknown", "unknown", None, "malformed-core-admission")
            except Exception:
                admitted_item = replace(item, recovery_admission=None)
                verdict = CoreAdmissionVerdictV2("unknown", "unknown", "unknown", "unknown", None, "core-admission-unavailable")
            frozen_inputs.append(replace(admitted_item, core_verdict=verdict))
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
        recovery_admissions = {}
        recovery_evidence = {}
        for task_id, (request, acceptance, proposed) in facts.successor_requests.items():
            if next(member.action for member in checked.members if member.task_id == task_id) != "admit_successor":
                continue
            successor_requests[task_id] = (request, acceptance, MemberInvocationV2(
                launch_attempt, group.identity, task_id,
                next(task.lineage_id for task in group.members if task.task_id == task_id),
                proposed.attempt_id, InvocationId(uuid4().hex),
            ))
            member_input = next(item for item in frozen_inputs if item.task_id == task_id)
            if member_input.recovery_admission is not None:
                recovery_admissions[task_id] = member_input.recovery_admission
                if member_input.recovery_evidence is not None:
                    recovery_evidence[task_id] = member_input.recovery_evidence
        publication_kwargs = {"bindings": bindings, "successors": successor_requests}
        if recovery_admissions:
            publication_kwargs["recovery_admissions"] = recovery_admissions
        if recovery_evidence:
            publication_kwargs["recovery_evidence"] = recovery_evidence
        published = store.publish_reconciliation(group, attempt, tuple(frozen_inputs), **publication_kwargs)
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
    from .local_executor import CollectionWaitTimeout, LeafCompletionContext, LeafSubmissionError

    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    outcomes: list[tuple[str, str]] = []
    reasons: dict[str, str] = {}
    parked_this_run: dict[str, str] = {}
    collected_members: set[str] = set()
    collected_fences: dict[str, Any] = {}
    active: list[tuple[Any, Any, Any, Any, Any, bool]] = []
    pending: list[Any] = []
    current_owned: tuple[Any, Any, Any, Any, Any, bool] | None = None

    def _park_member(task_id: str, phase: str, error: BaseException | None = None) -> None:
        outcomes.append((task_id, "parked"))
        reason = _reason(phase, error)
        reasons[task_id] = reason
        parked_this_run[task_id] = reason

    def _finish_unconfirmed(entry: tuple[Any, Any, Any, Any, Any, bool]) -> None:
        _, fence, invocation, _, _, _ = entry
        try:
            AttachedAllocation(allocation, store, invocation.allocation_attempt).finish(
                fence, "remote-quiescence-unconfirmed", _now(), None,
            )
        except Exception:
            pass

    def _cleanup_unconfirmed(extra: tuple[Any, Any, Any, Any, Any, bool] | None = None) -> None:
        entries = ([] if extra is None else [extra]) + active
        seen = set()
        for entry in entries:
            key = (entry[1].task_id, entry[1].attempt_id)
            if key in seen:
                continue
            seen.add(key)
            _finish_unconfirmed(entry)

    for member in published.members:
        if member.action in {"preserve_completed", "preserve_terminal"}:
            outcomes.append((member.task_id, member.action))
        elif member.action in {"start_fresh", "admit_successor"}:
            pending.append(member)
        elif member.action == "park":
            _park_member(member.task_id, member.reason_code)
        else:
            _park_member(member.task_id, "reconcile-action")

    capacity = getattr(executor, "capacity", 1)
    if type(capacity) is not int or capacity <= 0:
        capacity = 1

    def _fill_capacity() -> None:
        nonlocal current_owned
        while pending and len(active) < capacity:
            member = pending.pop(0)
            current_owned = None
            try:
                admission_record = None
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
                    admission_record = next((item.admission for item in durable_publication.members if item.evidence.task_id == member.task_id), None)
                    if not isinstance(admission_record, SuccessorAdmissionRecord) or admission_record.successor_binding is None:
                        raise LaunchIntentRejected("successor admission unavailable")
                    current_fence = None
                    if admission_record.recovery_admission is not None:
                        member_input = next(item for item in frozen_inputs if item.task_id == member.task_id)
                        if member_input.attempt_id is None or member_input.start_receipt is None:
                            raise LaunchIntentRejected("recovery current fence unavailable")
                        current_fence = AttemptFence(
                            store.root,
                            group.identity,
                            member.task_id,
                            member_input.start_receipt.allocation_id.value,
                            member_input.attempt_id,
                            member_input.start_receipt.output_dir,
                        )
                    fence = store.begin_admitted_successor_current_controller(
                        group, member.task_id, allocation, facts.observed_at_utc, "local",
                        allocation_attempt=admission_record.successor_binding.allocation_attempt, admission=admission_record,
                        current_fence=current_fence, controller_record=controller,
                    )
                    invocation = admission_record.successor_binding
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
                    if admission_record is not None and admission_record.recovery_admission is not None:
                        submit_admitted = getattr(executor, "submit_admitted_successor", None)
                        if not callable(submit_admitted):
                            raise LeafSubmissionError("executor cannot honor admitted successor checkpoint")
                        launched, handle = submit_admitted(
                            fence, invocation, admission_record.checkpoint_path,
                        )
                    elif admission_record is not None and hasattr(executor, "submit_admitted_successor"):
                        launched, handle = executor.submit_admitted_successor(
                            fence, invocation, admission_record.checkpoint_path,
                        )
                    else:
                        launched, handle = executor.submit(fence, invocation)
                except Exception as error:
                    try:
                        store.record_controller_outcome(fence, controller, invocation, "unacknowledged", "executor-failed")
                    except Exception:
                        pass
                    _park_member(member.task_id, "executor-launch", error)
                    continue
                if not isinstance(launched, MemberLaunchResult) or not launched.acknowledged:
                    try:
                        store.record_controller_outcome(fence, controller, invocation, "unacknowledged", "invalid-launch-ack")
                    except Exception:
                        pass
                    _park_member(member.task_id, "launch-ack")
                    continue
                entry = (member, fence, invocation, handle, controller, member.action == "admit_successor")
                current_owned = entry
                try:
                    store.record_controller_outcome(fence, controller, invocation, "launched", launched.detail or "acknowledged")
                except Exception as error:
                    _park_member(member.task_id, "outcome-receipt", error)
                active.append(entry)
                current_owned = None
            except Exception as error:
                _park_member(member.task_id, "member-transition", error)

    try:
        while pending or active:
            _fill_capacity()
            if not active:
                continue
            entry = active.pop(0)
            current_owned = entry
            member, fence, invocation, handle, owner, successor = entry
            try:
                result = executor.collect(handle, facts.poll_timeout)
            except CollectionWaitTimeout:
                # Polling is the only repeatable operation; keep this same handle (never resubmit).
                active.append(entry)
                current_owned = None
                continue
            except BaseException as error:
                _finish_unconfirmed(entry)
                _park_member(member.task_id, "collection", error)
                current_owned = None
                if not isinstance(error, Exception):
                    _cleanup_unconfirmed()
                    raise
                continue
            current_owned = None
            if not isinstance(result, LeafCompletionContext):
                error = LeafResultRejected("collector returned a non-LeafResult")
                _finish_unconfirmed(entry)
                _park_member(member.task_id, "validation", error)
                continue
            try:
                task = next(task for task in group.members if task.task_id == member.task_id)
                try:
                    validated = validate_leaf_result(result.request, result.result)
                except (TypeError, LeafResultRejected) as error:
                    _finish_unconfirmed(entry)
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
                collected_members.add(member.task_id)
                collected_fences[member.task_id] = fence.attempt_id
                if member.task_id not in parked_this_run:
                    outcomes.append((member.task_id, "collected"))
            except BaseException as error:
                _finish_unconfirmed(entry)
                _park_member(member.task_id, "completion", error)
                if not isinstance(error, Exception):
                    _cleanup_unconfirmed()
                    raise
    except BaseException:
        _cleanup_unconfirmed(current_owned)
        raise

    # Rebuild the exact member inputs from the durable manifest and ask the
    # admission port again before deriving the authoritative final action.
    try:
        manifest = store.manifest(group)
        final_inputs = []
        for item in facts.member_inputs:
            if item.task_id not in collected_members:
                continue
            attempt_id = collected_fences[item.task_id]
            record = _collected_attempt_record(manifest.attempts, item.task_id, attempt_id)
            if record is None:
                raise ValueError("collected attempt is absent from durable manifest")
            rebuilt = _apply_recovery_admission(item, store)
            rebuilt = replace(rebuilt, attempt_id=record.attempt_id, start_receipt=record.start_receipt,
                              outcome_receipt=record.outcome_receipt)
            if item.recovery_input is not None:
                rebuilt = replace(rebuilt, recovery_admission=None)
                outcome = (
                    "completed" if record.outcome_receipt is not None and record.outcome_receipt.status == "completed"
                    else "unknown" if record.core_outcome is None else record.core_outcome.outcome
                )
                if outcome not in {"completed", "terminal", "unknown", "successor"}:
                    outcome = "unknown"
                verdict = CoreAdmissionVerdictV2(
                    outcome, "unknown", "unknown", "unknown", None, "collected-core-outcome",
                )
            else:
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
