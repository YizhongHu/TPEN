"""Executor client for the core-declared checkpoint-recovery profile."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Protocol

from ._codec import canonical_bytes, decode_json, fields, sequence, sha256_text, text
from .contracts import AttemptId, GroupId, ReadyGroupV2
from .environments import PreparedPythonEnvironment
from ._records import frozen_record
from .results import OutcomeReceipt, StartReceipt
from .workers import SubprocessWorker

PROFILE_CHECK_SCHEMA = "tpen.profile-check/v1"
REFUSAL_CODES = frozenset(
    {
        "unknown-profile",
        "unsupported-callback",
        "unsupported-callback-cadence",
        "unsupported-logger",
        "malformed-config",
    }
)
PROFILE_REFUSAL_CODES = REFUSAL_CODES


class ProfileUnavailable(RuntimeError):
    """Raised when core cannot provide a trustworthy profile capability."""


@frozen_record
class ProfileCheck:
    """Strict immutable executor mirror of core's profile-check record."""

    schema: ClassVar[str] = PROFILE_CHECK_SCHEMA
    profile: str
    verdict: str
    state_owners: frozenset[str]
    refusal: Mapping[str, str] | None

    def __post_init__(self) -> None:
        if type(self.profile) is not str or not self.profile:
            raise TypeError("profile must be a non-empty string")
        if self.verdict not in {"supported", "refused"}:
            raise ValueError("verdict must be supported or refused")
        if any(type(owner) is not str or not owner for owner in self.state_owners):
            raise TypeError("state_owners must contain non-empty strings")
        object.__setattr__(self, "state_owners", frozenset(self.state_owners))
        if self.refusal is None:
            if self.verdict == "refused":
                raise ValueError("refused profile check requires refusal")
            return
        if self.verdict != "refused":
            raise ValueError("supported profile check cannot contain refusal")
        if not isinstance(self.refusal, Mapping) or set(self.refusal) != {"code", "reason"}:
            raise TypeError("refusal must contain code and reason")
        if type(self.refusal["code"]) is not str or self.refusal["code"] not in REFUSAL_CODES:
            raise ValueError("unknown refusal code")
        if type(self.refusal["reason"]) is not str:
            raise TypeError("refusal reason must be a string")
        object.__setattr__(self, "refusal", MappingProxyType(dict(self.refusal)))

    @classmethod
    def from_mapping(cls, value: Any) -> "ProfileCheck":
        """Decode one strict v1 core profile-check mapping."""

        if type(value) is not dict:
            raise ProfileUnavailable("profile check is not a JSON object")
        try:
            fields(value, {"schema", "profile", "verdict", "state_owners", "refusal"}, cls.schema, "ProfileCheck")
            if type(value["profile"]) is not str or not value["profile"]:
                raise TypeError("profile must be a non-empty string")
            if type(value["verdict"]) is not str or value["verdict"] not in {"supported", "refused"}:
                raise ValueError("unknown profile verdict")
            owners = sequence(value["state_owners"], "state_owners")
            if any(type(owner) is not str or not owner for owner in owners):
                raise TypeError("state_owners must contain non-empty strings")
            if len(set(owners)) != len(owners):
                raise ValueError("state_owners contains duplicates")
            refusal = value["refusal"]
            if value["verdict"] == "supported" and refusal is not None:
                raise ValueError("supported profile check cannot contain refusal")
            if value["verdict"] == "refused" and refusal is None:
                raise ValueError("refused profile check requires refusal")
            return cls(value["profile"], value["verdict"], frozenset(owners), refusal)
        except (TypeError, ValueError, KeyError) as exc:
            raise ProfileUnavailable("profile check failed strict v1 decode") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-shaped record."""

        return {
            "schema": self.schema,
            "profile": self.profile,
            "verdict": self.verdict,
            "state_owners": sorted(self.state_owners),
            "refusal": None if self.refusal is None else dict(self.refusal),
        }

    def __hash__(self) -> int:
        """Hash canonical bytes so equality and hashing agree."""

        return hash(canonical_bytes(self.to_dict()))


@frozen_record
class CoreProfileReference:
    """Reference to a core config and declared recovery profile.

    Parameters
    ----------
    config : str
        Absolute path to the unresolved core configuration.
    profile : str
        Declared profile identifier.
    """

    config: str
    profile: str

    def __post_init__(self) -> None:
        if type(self.config) is not str or not os.path.isabs(self.config):
            raise ValueError("config must be an absolute path")
        if type(self.profile) is not str or not self.profile:
            raise ValueError("profile must be a non-empty string")

    def read(
        self,
        worker: SubprocessWorker,
        environment: PreparedPythonEnvironment,
        *,
        cwd: str | os.PathLike[str] | None = None,
        stdout_path: str | os.PathLike[str],
        stderr_path: str | os.PathLike[str],
        timeout: float | None = None,
    ) -> ProfileCheck:
        """Launch core's profile CLI and decode one bound profile record.

        Returns
        -------
        ProfileCheck
            The decoded supported or refused result for ``self.profile``. A
            refused result is a valid, typed capability response and is
            returned to the caller.

        Raises
        ------
        ProfileUnavailable
            If the process, transport, JSON payload, strict v1 record, or
            requested-profile binding cannot be trusted. A response naming a
            different profile is unavailable rather than a refusal for this
            reference.
        """

        try:
            outcome = worker.run(
                environment,
                ("-m", "tpen.checkpoint.profiles", self.config, "--profile", self.profile),
                cwd=cwd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout=timeout,
            )
        except OSError as exc:
            raise ProfileUnavailable("profile process could not be launched") from exc
        if outcome.timed_out or outcome.returncode != 0:
            raise ProfileUnavailable("profile process did not complete successfully")
        try:
            payload = Path(outcome.stdout_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ProfileUnavailable("profile stdout could not be read") from exc
        if not payload.strip():
            raise ProfileUnavailable("profile stdout is empty")
        try:
            decoded = decode_json(payload)
        except (TypeError, ValueError) as exc:
            raise ProfileUnavailable("profile stdout is not one JSON object") from exc
        result = ProfileCheck.from_mapping(decoded)
        if result.profile != self.profile:
            raise ProfileUnavailable("profile response does not match requested profile")
        return result


CORE_RECOVERY_OUTCOMES = frozenset({"completed", "successor", "terminal", "unknown"})
CORE_ADMISSION_VERDICTS = frozenset({"admitted", "refused", "unknown"})
CORE_CHECKPOINT_VERDICTS = frozenset({"valid", "invalid", "absent", "unknown"})
CORE_QUIESCENCE_VERDICTS = frozenset({"valid", "invalid", "unknown"})
RECONCILIATION_ACTIONS = frozenset(
    {"preserve_completed", "admit_successor", "start_fresh", "preserve_terminal", "park"}
)
# This outcome records that remote teardown was not confirmed. It is not a
# settled operational result and must never reach successor admission.
NON_SETTLED_OPERATIONAL_OUTCOMES = frozenset({"remote-quiescence-unconfirmed"})


@frozen_record
class CoreAdmissionVerdictV2:
    """Typed core-owned recovery/admission facts consumed by B1a.

    The executor only validates this closed reference.  It does not inspect
    checkpoints, reconstruct core state, or import the core package.
    """

    schema: ClassVar[str] = "tpen-executor/core-admission-verdict-v2"
    outcome: str
    admission: str
    checkpoint: str
    quiescence: str
    profile: ProfileCheck | None
    reason: str

    def __post_init__(self) -> None:
        if self.outcome not in CORE_RECOVERY_OUTCOMES:
            raise ValueError("unknown core recovery outcome")
        if self.admission not in CORE_ADMISSION_VERDICTS:
            raise ValueError("unknown core admission verdict")
        if self.checkpoint not in CORE_CHECKPOINT_VERDICTS:
            raise ValueError("unknown core checkpoint verdict")
        if self.quiescence not in CORE_QUIESCENCE_VERDICTS:
            raise ValueError("unknown core quiescence verdict")
        if self.profile is not None and not isinstance(self.profile, ProfileCheck):
            raise TypeError("profile must be ProfileCheck or None")
        object.__setattr__(self, "reason", text(self.reason, "reason"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "outcome": self.outcome,
            "admission": self.admission,
            "checkpoint": self.checkpoint,
            "quiescence": self.quiescence,
            "profile": None if self.profile is None else self.profile.to_dict(),
            "reason": self.reason,
        }

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "CoreAdmissionVerdictV2":
        fields(
            data,
            {"schema", "outcome", "admission", "checkpoint", "quiescence", "profile", "reason"},
            cls.schema,
            cls.__name__,
        )
        if any(type(data[key]) is not str for key in ("outcome", "admission", "checkpoint", "quiescence", "reason")):
            raise TypeError("CoreAdmissionVerdictV2 scalar fields have invalid types")
        profile = None if data["profile"] is None else ProfileCheck.from_mapping(data["profile"])
        return cls(data["outcome"], data["admission"], data["checkpoint"], data["quiescence"], profile, data["reason"])

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "CoreAdmissionVerdictV2":
        return cls.from_mapping(decode_json(data))


@frozen_record
class MemberRecoveryInputV2:
    """Closed references used to reconcile one v2 group member."""

    schema: ClassVar[str] = "tpen-executor/member-recovery-input-v2"
    task_id: str
    lineage_id: str
    attempt_id: AttemptId | None = None
    start_receipt: StartReceipt | None = None
    outcome_receipt: OutcomeReceipt | None = None
    core_verdict: CoreAdmissionVerdictV2 | None = None
    receipt_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", text(self.task_id, "task_id"))
        object.__setattr__(self, "lineage_id", text(self.lineage_id, "lineage_id"))
        if self.attempt_id is not None and not isinstance(self.attempt_id, AttemptId):
            raise TypeError("attempt_id must be AttemptId or None")
        if self.start_receipt is not None and not isinstance(self.start_receipt, StartReceipt):
            raise TypeError("start_receipt must be StartReceipt or None")
        if self.outcome_receipt is not None and not isinstance(self.outcome_receipt, OutcomeReceipt):
            raise TypeError("outcome_receipt must be OutcomeReceipt or None")
        if self.core_verdict is not None and not isinstance(self.core_verdict, CoreAdmissionVerdictV2):
            raise TypeError("core_verdict must be CoreAdmissionVerdictV2 or None")
        identities = [
            value for value in (
                None if self.attempt_id is None else self.attempt_id,
                None if self.start_receipt is None else self.start_receipt.attempt_id,
                None if self.outcome_receipt is None else self.outcome_receipt.attempt_id,
            )
            if value is not None
        ]
        if len(set(identities)) > 1:
            raise ValueError("member recovery receipts cross attempt identity")
        if self.start_receipt is not None and self.start_receipt.task_id != self.task_id:
            raise ValueError("start receipt crosses member task identity")
        if self.outcome_receipt is not None and self.outcome_receipt.task_id != self.task_id:
            raise ValueError("outcome receipt crosses member task identity")
        if self.start_receipt is not None and self.outcome_receipt is not None:
            if self.start_receipt.group_digest != self.outcome_receipt.group_digest or self.start_receipt.output_dir != self.outcome_receipt.output_dir:
                raise ValueError("member recovery receipts cross group or output identity")
        if self.receipt_digest is not None:
            object.__setattr__(self, "receipt_digest", sha256_text(self.receipt_digest, "receipt_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "lineage_id": self.lineage_id,
            "attempt_id": None if self.attempt_id is None else self.attempt_id.to_dict(),
            "start_receipt": None if self.start_receipt is None else self.start_receipt.to_dict(),
            "outcome_receipt": None if self.outcome_receipt is None else self.outcome_receipt.to_dict(),
            "core_verdict": None if self.core_verdict is None else self.core_verdict.to_dict(),
            "receipt_digest": self.receipt_digest,
        }

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "MemberRecoveryInputV2":
        fields(
            data,
            {"schema", "task_id", "lineage_id", "attempt_id", "start_receipt", "outcome_receipt", "core_verdict", "receipt_digest"},
            cls.schema,
            cls.__name__,
        )
        if type(data["task_id"]) is not str or type(data["lineage_id"]) is not str:
            raise TypeError("MemberRecoveryInputV2 identity fields must be strings")
        return cls(
            data["task_id"],
            data["lineage_id"],
            None if data["attempt_id"] is None else AttemptId.from_mapping(data["attempt_id"]),
            None if data["start_receipt"] is None else StartReceipt.from_mapping(data["start_receipt"]),
            None if data["outcome_receipt"] is None else OutcomeReceipt.from_mapping(data["outcome_receipt"]),
            None if data["core_verdict"] is None else CoreAdmissionVerdictV2.from_mapping(data["core_verdict"]),
            data["receipt_digest"],
        )

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "MemberRecoveryInputV2":
        return cls.from_mapping(decode_json(data))


@frozen_record
class MemberReconciliationV2:
    """One derived action and stable reason for a group member."""

    schema: ClassVar[str] = "tpen-executor/member-reconciliation-v2"
    task_id: str
    action: str
    reason_code: str
    attempt_id: AttemptId | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", text(self.task_id, "task_id"))
        if self.action not in RECONCILIATION_ACTIONS:
            raise ValueError("unknown member reconciliation action")
        object.__setattr__(self, "reason_code", text(self.reason_code, "reason_code"))
        if self.attempt_id is not None and not isinstance(self.attempt_id, AttemptId):
            raise TypeError("attempt_id must be AttemptId or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "action": self.action,
            "reason_code": self.reason_code,
            "attempt_id": None if self.attempt_id is None else self.attempt_id.to_dict(),
        }

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "MemberReconciliationV2":
        fields(data, {"schema", "task_id", "action", "reason_code", "attempt_id"}, cls.schema, cls.__name__)
        if any(type(data[key]) is not str for key in ("task_id", "action", "reason_code")):
            raise TypeError("MemberReconciliationV2 scalar fields have invalid types")
        return cls(data["task_id"], data["action"], data["reason_code"], None if data["attempt_id"] is None else AttemptId.from_mapping(data["attempt_id"]))

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "MemberReconciliationV2":
        return cls.from_mapping(decode_json(data))


@frozen_record
class GroupReconciliationV2:
    """Derived aggregate of every member outcome in one v2 group."""

    schema: ClassVar[str] = "tpen-executor/group-reconciliation-v2"
    group_id: GroupId
    group_digest: str
    members: tuple[MemberReconciliationV2, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, GroupId):
            raise TypeError("group_id must be GroupId")
        object.__setattr__(self, "group_digest", sha256_text(self.group_digest, "group_digest"))
        members = sequence(self.members, "members")
        if not members or any(not isinstance(member, MemberReconciliationV2) for member in members):
            raise TypeError("members must be a non-empty sequence of MemberReconciliationV2")
        if len({member.task_id for member in members}) != len(members):
            raise ValueError("group reconciliation members must be unique")
        object.__setattr__(self, "members", members)

    @property
    def aggregate(self) -> str:
        if any(member.action == "park" for member in self.members):
            return "parked"
        if any(member.action in {"admit_successor", "start_fresh"} for member in self.members):
            return "replacement_required"
        return "settled"

    @property
    def completed_count(self) -> int:
        return sum(member.action == "preserve_completed" for member in self.members)

    @property
    def terminal_count(self) -> int:
        return sum(member.action == "preserve_terminal" for member in self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "group_id": self.group_id.to_dict(),
            "group_digest": self.group_digest,
            "members": [member.to_dict() for member in self.members],
            "aggregate": self.aggregate,
            "completed_count": self.completed_count,
            "terminal_count": self.terminal_count,
        }

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "GroupReconciliationV2":
        fields(
            data,
            {"schema", "group_id", "group_digest", "members", "aggregate", "completed_count", "terminal_count"},
            cls.schema,
            cls.__name__,
        )
        if type(data["group_digest"]) is not str or type(data["aggregate"]) is not str or type(data["completed_count"]) is not int or type(data["terminal_count"]) is not int:
            raise TypeError("GroupReconciliationV2 derived fields have invalid types")
        record = cls(
            GroupId.from_mapping(data["group_id"]),
            data["group_digest"],
            tuple(MemberReconciliationV2.from_mapping(item) for item in sequence(data["members"], "members")),
        )
        if (record.aggregate, record.completed_count, record.terminal_count) != (data["aggregate"], data["completed_count"], data["terminal_count"]):
            raise ValueError("GroupReconciliationV2 aggregate is not derived from members")
        return record

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "GroupReconciliationV2":
        return cls.from_mapping(decode_json(data))


class CoreAdmissionPort(Protocol):
    """Fake-friendly seam for typed core admission/profile verdicts."""

    def admit(self, member: MemberRecoveryInputV2) -> CoreAdmissionVerdictV2 | None:
        """Return a typed verdict, or ``None`` when core evidence is absent."""


CORE_ADMISSION_WIRE_SCHEMA = "tpen-core-admission-v1"
_CORE_ADMISSION_WIRE_FIELDS = frozenset(
    {"schema", "outcome", "admission", "checkpoint", "quiescence", "profile", "reason"}
)


class MappingCoreAdmissionPort:
    """Decode core's strict admission mapping at the executor boundary.

    Parameters
    ----------
    evaluate : callable
        In-process callable receiving one :class:`MemberRecoveryInputV2` and
        returning a core-owned mapping or ``None``.  The callable is invoked
        exactly once per ``admit`` call.

    Notes
    -----
    The wire schema is ``tpen-core-admission-v1`` and has a closed field set.
    Scalar fields are exact strings, while ``profile`` is either ``None`` or
    decoded by :meth:`ProfileCheck.from_mapping`.  Core-owned quiescence is
    deliberately ignored: the executor fills it as ``valid`` only when the
    member has a settled outcome receipt, and as ``unknown`` otherwise.  Any
    callable exception, malformed mapping, or absent evidence returns
    ``None``.  The callable is in-process; no timeout machinery is provided.
    """

    def __init__(self, evaluate: Callable[[MemberRecoveryInputV2], Mapping[str, Any] | None]) -> None:
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        self._evaluate = evaluate

    def admit(self, member: MemberRecoveryInputV2) -> CoreAdmissionVerdictV2 | None:
        """Evaluate and strictly decode one member's core admission mapping."""

        try:
            payload = self._evaluate(member)
            if payload is None:
                return None
            if not isinstance(payload, Mapping):
                raise TypeError("payload must be a mapping")
            if set(payload) != _CORE_ADMISSION_WIRE_FIELDS:
                raise ValueError("core admission mapping has an invalid closed field set")
            if type(payload["schema"]) is not str or payload["schema"] != CORE_ADMISSION_WIRE_SCHEMA:
                raise ValueError("core admission schema is not tpen-core-admission-v1")
            if any(
                type(payload[key]) is not str
                for key in ("outcome", "admission", "checkpoint", "quiescence", "reason")
            ):
                raise TypeError("core admission scalar fields have invalid types")
            quiescence = (
                "valid"
                if member.outcome_receipt is not None
                and member.outcome_receipt.status not in NON_SETTLED_OPERATIONAL_OUTCOMES
                else "unknown"
            )
            if payload["quiescence"] != "unknown":
                raise ValueError("core admission quiescence must be unknown")
            profile = None if payload["profile"] is None else ProfileCheck.from_mapping(payload["profile"])
            return CoreAdmissionVerdictV2(
                payload["outcome"],
                payload["admission"],
                payload["checkpoint"],
                quiescence,
                profile,
                payload["reason"],
            )
        except Exception:
            # Any evaluator or decoder failure is unavailable evidence.
            return None  # unavailable


def _park(member: MemberRecoveryInputV2, reason_code: str) -> MemberReconciliationV2:
    return MemberReconciliationV2(member.task_id, "park", reason_code, member.attempt_id)


def _member_reconciliation(
    member: MemberRecoveryInputV2,
    *,
    group_digest: str,
    admission_port: CoreAdmissionPort | None,
    restore_callback: Callable[[MemberRecoveryInputV2], Any] | None,
) -> MemberReconciliationV2:
    """Classify one member without creating an allocation or process."""

    if member.start_receipt is None:
        if member.outcome_receipt is None and member.core_verdict is None and member.receipt_digest is None:
            return MemberReconciliationV2(member.task_id, "start_fresh", "no-durable-start", None)
        return _park(member, "crossed-or-incomplete-receipt")
    if member.attempt_id is None:
        return _park(member, "missing-attempt-identity")
    if member.start_receipt.group_digest != group_digest or (
        member.outcome_receipt is not None and member.outcome_receipt.group_digest != group_digest
    ):
        return _park(member, "crossed-group-receipt")
    if member.outcome_receipt is None:
        return _park(member, "missing-settled-outcome")
    if member.outcome_receipt.status in NON_SETTLED_OPERATIONAL_OUTCOMES:
        return _park(member, member.outcome_receipt.status)
    verdict = member.core_verdict
    if admission_port is not None:
        try:
            verdict = admission_port.admit(member)
        except (TypeError, ValueError, RuntimeError):
            verdict = None
        if verdict is None:
            return _park(member, "core-admission-unavailable")
        if not isinstance(verdict, CoreAdmissionVerdictV2):
            return _park(member, "malformed-core-admission")
    if verdict is None:
        return _park(member, "missing-core-verdict")
    if verdict.outcome == "completed":
        if member.outcome_receipt.status != "completed":
            return _park(member, "contradictory-completion")
        return MemberReconciliationV2(member.task_id, "preserve_completed", "completed-and-settled", member.attempt_id)
    if verdict.outcome == "terminal":
        return MemberReconciliationV2(member.task_id, "preserve_terminal", "core-terminal", member.attempt_id)
    if verdict.outcome == "unknown":
        return _park(member, "unknown-core-result")
    if verdict.profile is None or verdict.profile.verdict != "supported":
        return _park(member, "unsupported-profile")
    if verdict.admission != "admitted":
        return _park(member, "core-admission-refused")
    if verdict.checkpoint != "valid" or verdict.quiescence != "valid":
        return _park(member, "invalid-successor-evidence")
    if restore_callback is not None:
        try:
            if restore_callback(member) is False:
                return _park(member, "restore-refused")
        except Exception:
            return _park(member, "restore-failed")
    return MemberReconciliationV2(member.task_id, "admit_successor", "valid-successor-admission", member.attempt_id)


def reconcile_group(
    group: ReadyGroupV2,
    inputs: Sequence[MemberRecoveryInputV2],
    *,
    admission_port: CoreAdmissionPort | None = None,
    restore_callback: Callable[[MemberRecoveryInputV2], Any] | None = None,
) -> GroupReconciliationV2:
    """Purely reconcile one v2 group from typed fake/durable references.

    The function only derives actions.  It never submits, replaces, signals,
    scans checkpoints, writes a ledger, or maps an outcome to a scheduler
    status.  Missing or crossed evidence becomes a member-level ``park``.
    """

    if not isinstance(group, ReadyGroupV2):
        raise TypeError("group must be ReadyGroupV2")
    if isinstance(inputs, (str, bytes, bytearray)) or not isinstance(inputs, Sequence):
        raise TypeError("inputs must be a sequence")
    by_task: dict[str, MemberRecoveryInputV2] = {}
    for item in inputs:
        if not isinstance(item, MemberRecoveryInputV2):
            raise TypeError("inputs must contain MemberRecoveryInputV2 records")
        if item.task_id in by_task:
            raise ValueError("duplicate member recovery input")
        by_task[item.task_id] = item
    member_by_task = {member.task_id: member for member in group.members}
    foreign = set(by_task) - set(member_by_task)
    if foreign:
        # Membership is a group-wide precondition. Do not invoke any port or
        # restore callback while foreign evidence is present, and emit the
        # fail-closed result in immutable ReadyGroup member order.
        outcomes = tuple(
            _park(by_task[task.task_id], "foreign-member-evidence")
            if task.task_id in by_task
            else MemberReconciliationV2(task.task_id, "park", "foreign-member-evidence", None)
            for task in group.members
        )
        return GroupReconciliationV2(group.group_id, group.identity, outcomes)
    outcomes: list[MemberReconciliationV2] = []
    for task in group.members:
        item = by_task.get(task.task_id)
        if item is None:
            outcomes.append(MemberReconciliationV2(task.task_id, "park", "missing-member-evidence", None))
            continue
        if item.lineage_id != task.lineage_id:
            outcomes.append(_park(item, "crossed-lineage"))
            continue
        outcomes.append(
            _member_reconciliation(
                item,
                group_digest=group.identity,
                admission_port=admission_port,
                restore_callback=restore_callback,
            )
        )
    return GroupReconciliationV2(group.group_id, group.identity, tuple(outcomes))


reconcile_allocation_group = reconcile_group
reconcile = reconcile_group


__all__ = [
    "CORE_ADMISSION_VERDICTS", "CORE_CHECKPOINT_VERDICTS", "CORE_QUIESCENCE_VERDICTS", "CORE_RECOVERY_OUTCOMES",
    "CoreProfileReference",
    "CoreAdmissionPort", "CoreAdmissionVerdictV2", "CORE_ADMISSION_WIRE_SCHEMA", "GroupReconciliationV2", "MappingCoreAdmissionPort", "MemberReconciliationV2", "MemberRecoveryInputV2",
    "PROFILE_REFUSAL_CODES",
    "ProfileCheck",
    "ProfileUnavailable",
    "RECONCILIATION_ACTIONS", "reconcile", "reconcile_allocation_group", "reconcile_group",
    "REFUSAL_CODES",
]
