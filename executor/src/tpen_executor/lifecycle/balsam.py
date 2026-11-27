"""Optional group registration and application boundary for Balsam.

This module does not implement the allocation-group controller. Backend state
is opaque observation data, never evidence of scientific or group completion.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
import os
from pathlib import Path
from typing import Any, ClassVar, Protocol

from .._codec import canonical_bytes, decode_json, digest, fields, sha256_text
from .._records import frozen_record
from ..contracts import BackendJobId, GroupId
from ..recovery import GroupReconciliationV2
from ..storage import DirectoryStore, ensure_durable_directory, require_regular_file, publish_json_exclusive


class BalsamGroupRefusal(RuntimeError):
    """A registration binding or optional application dependency was refused."""


class BalsamGroupParked(RuntimeError):
    """Registration acknowledgement is uncertain; never submit again."""


@frozen_record
class BalsamGroupRegistrationIntent:
    """Immutable group, execution plan and backend request binding.

    Parameters
    ----------
    group_id : GroupId
        Persistent logical group, independent of allocation identities.
    group_digest, backend_options_digest : str
        Canonical ready-group and complete backend-request digests. The latter
        must include the application and all deployment options.
    """

    schema: ClassVar[str] = "tpen-executor/balsam-group-registration-intent-v1"
    group_id: GroupId
    group_digest: str
    backend_options_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, GroupId):
            raise TypeError("group_id must be GroupId")
        for name in ("group_digest", "backend_options_digest"):
            sha256_text(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned wire record."""
        return {"schema": self.schema, "group_id": self.group_id.to_dict(),
                "group_digest": self.group_digest, "backend_options_digest": self.backend_options_digest}

    def encode(self) -> bytes:
        """Return canonical durable bytes."""
        return canonical_bytes(self.to_dict())

    @classmethod
    def decode(cls, payload: bytes | str | Mapping[str, Any]) -> BalsamGroupRegistrationIntent:
        """Decode a strict versioned intent."""
        data = decode_json(payload)
        fields(data, {"schema", "group_id", "group_digest", "backend_options_digest"}, cls.schema, cls.__name__)
        return cls(GroupId.from_mapping(data["group_id"]), data["group_digest"], data["backend_options_digest"])


@frozen_record
class BalsamGroupRegistrationReceipt:
    """Acknowledged persistent Job bound to the full registration intent."""

    schema: ClassVar[str] = "tpen-executor/balsam-group-registration-receipt-v1"
    intent: BalsamGroupRegistrationIntent
    backend_job_id: BackendJobId

    def __post_init__(self) -> None:
        if not isinstance(self.intent, BalsamGroupRegistrationIntent):
            raise TypeError("intent must be BalsamGroupRegistrationIntent")
        if not isinstance(self.backend_job_id, BackendJobId):
            raise TypeError("backend_job_id must be BackendJobId")

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned wire record."""
        return {"schema": self.schema, "intent": self.intent.to_dict(),
                "backend_job_id": self.backend_job_id.to_dict()}

    def encode(self) -> bytes:
        """Return canonical durable bytes."""
        return canonical_bytes(self.to_dict())

    @classmethod
    def decode(cls, payload: bytes | str | Mapping[str, Any]) -> BalsamGroupRegistrationReceipt:
        """Decode a strict versioned acknowledgement."""
        data = decode_json(payload)
        fields(data, {"schema", "intent", "backend_job_id"}, cls.schema, cls.__name__)
        return cls(BalsamGroupRegistrationIntent.decode(data["intent"]), BackendJobId.from_mapping(data["backend_job_id"]))


class GroupBalsamClient(Protocol):
    """Group-keyed registration port; observation has no completion semantics."""

    def register_group(self, intent: BalsamGroupRegistrationIntent) -> BackendJobId:
        """Register once and return an explicit Job acknowledgement."""
        ...

    def observe_job(self, job_id: BackendJobId) -> object:
        """Return backend-private facts, never a group completion verdict."""
        ...


def _read_record(path: Path, record_type: Any) -> Any:
    require_regular_file(str(path), "Balsam group record")
    payload = path.read_bytes()
    record = record_type.decode(payload)
    if payload != record.encode():
        raise BalsamGroupRefusal("noncanonical Balsam group record")
    return record


def register_group(
    intent: BalsamGroupRegistrationIntent,
    store: DirectoryStore,
    client: GroupBalsamClient,
) -> BalsamGroupRegistrationReceipt:
    """Durably register at most one Job per GroupId in this store.

    An exclusive, fsynced intent is also the permanent ownership marker. Only
    its creator may call the backend. A crash before the call is deliberately
    indistinguishable from a lost acknowledgement. Reopening either case
    parks; a subsequent caller never guesses whether registration occurred.

    Raises
    ------
    BalsamGroupRefusal
        The intent changed, durable bytes are invalid, or the client supplied
        a typed permanent refusal.
    BalsamGroupParked
        Acknowledgement is missing or ambiguous. No automatic retry is safe.
    """
    if not isinstance(intent, BalsamGroupRegistrationIntent) or not isinstance(store, DirectoryStore):
        raise TypeError("registration requires a typed intent and DirectoryStore")
    directory = Path(store.root, "balsam-groups", digest(intent.group_id.to_dict()))
    try:
        ensure_durable_directory(store.root)
        ensure_durable_directory(str(directory.parent))
        ensure_durable_directory(str(directory))
        intent_path, receipt_path = directory / "intent.json", directory / "receipt.json"
        refusal_path = directory / "refusal.json"
        owns_intent = False
        if not os.path.lexists(intent_path):
            if os.path.lexists(receipt_path) or os.path.lexists(refusal_path):
                raise BalsamGroupRefusal("acknowledgement/refusal without intent")
            try:
                publish_json_exclusive(str(intent_path), intent.encode())
                owns_intent = True
            except FileExistsError:
                pass  # A competing controller owns the one possible call.
        stored = _read_record(intent_path, BalsamGroupRegistrationIntent)
        if stored != intent:
            raise BalsamGroupRefusal("GroupId already bound to a different intent")
        if os.path.lexists(refusal_path):
            raise BalsamGroupRefusal("registration was permanently refused")
        if os.path.lexists(receipt_path):
            receipt = _read_record(receipt_path, BalsamGroupRegistrationReceipt)
            if receipt.intent != intent:
                raise BalsamGroupRefusal("registration receipt crosses intent")
            return receipt
        if not owns_intent:
            raise BalsamGroupParked("acknowledgement unknown; no second registration")
        try:
            job_id = client.register_group(intent)
        except BalsamGroupRefusal as error:
            publish_json_exclusive(str(refusal_path), canonical_bytes({
                "schema": "tpen-executor/balsam-group-refusal-v1",
                "intent": intent.to_dict(), "reason": str(error),
            }))
            raise
        except Exception as error:
            raise BalsamGroupParked("acknowledgement lost; no second registration") from error
        if not isinstance(job_id, BackendJobId):
            raise BalsamGroupParked("missing typed Job acknowledgement")
        receipt = BalsamGroupRegistrationReceipt(intent, job_id)
        publish_json_exclusive(str(receipt_path), receipt.encode())
        return receipt
    except (BalsamGroupRefusal, BalsamGroupParked):
        raise
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        # This includes acknowledgement publication failure. The intent stays
        # in place, preventing another call even after process replacement.
        raise BalsamGroupParked("registration storage is uncertain") from error


class GroupFailure(str, Enum):
    """Non-success outcomes that never request backend retry."""

    PARKED = "parked"
    PRE_START = "pre-start-failure"
    UNCAUGHT = "uncaught-exception"
    REPLACEMENT_REQUIRED = "replacement_required"


@frozen_record
class GroupExitContract:
    """Versioned application exit policy, pending deployment qualification.

    Parameters
    ----------
    backend_retry_codes : tuple of int
        Explicit deployed backend retry set; historical values are not defaults.
    no_retry_code : int, optional
        Version one fixes this to 2. It must be outside the backend retry set.
    """

    schema: ClassVar[str] = "tpen-executor/balsam-group-exit-contract-v1"
    backend_retry_codes: tuple[int, ...]
    no_retry_code: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.backend_retry_codes, (tuple, list)) or any(type(code) is not int for code in self.backend_retry_codes):
            raise TypeError("backend_retry_codes must be a sequence of integers")
        object.__setattr__(self, "backend_retry_codes", tuple(self.backend_retry_codes))
        if type(self.no_retry_code) is not int or self.no_retry_code != 2:
            raise BalsamGroupRefusal("v1 no_retry_code must be 2")
        if self.no_retry_code in self.backend_retry_codes or 0 in self.backend_retry_codes:
            raise BalsamGroupRefusal("application exit code is in backend retry set")

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned exit policy."""
        return {"schema": self.schema, "backend_retry_codes": list(self.backend_retry_codes), "no_retry_code": self.no_retry_code}

    @classmethod
    def decode(cls, payload: bytes | str | Mapping[str, Any]) -> GroupExitContract:
        """Decode a strict deployment exit policy."""
        data = decode_json(payload)
        fields(data, {"schema", "backend_retry_codes", "no_retry_code"}, cls.schema, cls.__name__)
        return cls(data["backend_retry_codes"], data["no_retry_code"])


def group_exit_code(outcome: GroupReconciliationV2 | Any, contract: GroupExitContract) -> int:
    """Map authoritative member reconciliation or a failure to process exit.

    Success requires *every* member to be preserved completed. Terminal-only
    groups are settled in B1a but still exit 2. Backend states and integer
    return values are deliberately not accepted as authoritative outcomes.
    """
    if not isinstance(contract, GroupExitContract):
        raise TypeError("contract must be GroupExitContract")
    from .controller import GroupControllerOutcome
    if isinstance(outcome, GroupControllerOutcome):
        outcome = outcome.reconciliation
    if isinstance(outcome, GroupReconciliationV2):
        if all(member.action == "preserve_completed" for member in outcome.members):
            return 0
        return contract.no_retry_code
    if isinstance(outcome, GroupFailure):
        return contract.no_retry_code
    raise BalsamGroupRefusal("controller returned no typed group outcome")


def make_group_application(
    api: Any = None,
    *,
    controller: Callable[[Any], GroupReconciliationV2 | GroupFailure],
    contract: GroupExitContract,
    record_failure: Callable[[Any, BaseException], None],
    timeout_handler: Callable[[Any], None] | None = None,
    error_handler: Callable[[Any, BaseException | None], None] | None = None,
) -> type:
    """Build an optional ApplicationDefinition around an injected controller.

    Parameters
    ----------
    api : object or None
        Injected Balsam API (a fake in tests); None loads the optional SDK here.
    controller : callable
        Receives the launched Job. Owns all durable start/pre-start and member
        receipts and returns authoritative reconciliation, never backend state.
    contract : GroupExitContract
        Explicit deployment policy. Its values need independent qualification.
    record_failure : callable
        Writes a durable pre-start/controller failure receipt using the Job
        and exception. A recorder failure still exits 2 and is attached to the
        original exception; durable storage failure cannot promise a receipt.
    timeout_handler, error_handler : callable, optional
        Explicit non-retry observation hooks.  The generated application calls
        these after marking the Job non-runnable; neither hook may rearm it.

    Notes
    -----
    The shell preamble is empty and no environment override is installed.
    Environment preparation and its durable readback belong to the injected
    controller. This factory neither discovers allocations nor registers Jobs.
    """
    if (not isinstance(contract, GroupExitContract) or not callable(controller)
            or not callable(record_failure)
            or (timeout_handler is not None and not callable(timeout_handler))
            or (error_handler is not None and not callable(error_handler))):
        raise BalsamGroupRefusal("application requires controller, failure recorder and exit contract")
    if api is None:
        try:
            from balsam import api as loaded_api
        except ImportError as error:
            raise BalsamGroupRefusal("optional Balsam API is unavailable") from error
        api = loaded_api
    base = getattr(api, "ApplicationDefinition", None)
    if not isinstance(base, type):
        raise BalsamGroupRefusal("API has no ApplicationDefinition class")

    class GroupApplication(base):
        """Application boundary; controller receipt is the success authority."""

        environment_variables = {}

        def shell_preamble(self) -> str:
            """Leave preparation to Python, where failures can be recorded."""
            return ""

        def run(self) -> int:
            """Delegate once and use process status, not a serialized integer."""
            job = None
            try:
                job = self.job
                code = group_exit_code(controller(job), contract)
            except BaseException as error:
                try:
                    record_failure(job, error)
                except BaseException as recording_error:
                    error.add_note(f"failure receipt unavailable: {recording_error}")
                raise SystemExit(group_exit_code(GroupFailure.UNCAUGHT, contract)) from error
            if code != 0:
                # Balsam's Python runner can serialize returned integers and
                # exit zero. Non-success must be an actual process exit.
                raise SystemExit(code)
            return 0

        def handle_timeout(self) -> None:
            """Mark timeout non-runnable and wait for explicit reentry."""
            job = getattr(self, "job", None)
            if job is not None and hasattr(job, "state"):
                job.state = "failed"
            if timeout_handler is not None:
                timeout_handler(job)

        def handle_error(self, error: BaseException | None = None) -> None:
            """Mark an application error non-runnable without inherited retry."""
            job = getattr(self, "job", None)
            if job is not None and hasattr(job, "state"):
                job.state = "failed"
            if error_handler is not None:
                error_handler(job, error)

        def request_rearm(self, *args: Any, **kwargs: Any) -> None:
            """Refuse implicit rearm; S8 supplies the explicit typed path."""
            del args, kwargs
            raise BalsamGroupRefusal("rearm requires an explicit grant-bound transaction")

    return GroupApplication
