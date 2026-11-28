"""S7 Balsam carrier transactions and the production handoff boundary.

The executor owns the finite carrier intent and its journal.  The isolated
control process owns the optional SDK and performs exactly one ``BatchJob``
creation after receiving the persisted output paths.  Neither backend state nor
rendered deployment text can authorize a fresh mutation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
from typing import Any, ClassVar, Protocol

from .._codec import canonical_bytes, decode_json, digest, fields, sha256_text, text
from .._records import frozen_record
from ..contracts import BackendJobId, BatchJobId, GroupId, ReadyGroupV2
from ..storage import DirectoryStore, ensure_durable_directory, publish_json_exclusive
from .allocation_capture import FullNodeAssignment
from .balsam import (
    BalsamGroupParked,
    BalsamGroupRefusal,
    BalsamGroupRegistrationReceipt,
)


class BalsamBatchClient(Protocol):
    """Finite client seam for one fresh carrier mutation."""

    def submit_batch(
        self,
        group_id: GroupId,
        backend_job_id: BackendJobId,
        batch_digest: str,
        *,
        stdout_path: str,
        stderr_path: str,
    ) -> BatchJobId:
        """Create one BatchJob and return its typed acknowledgement."""


@frozen_record
class BalsamBatchIntent:
    """Immutable carrier request persisted before the sole BatchJob mutation.

    Parameters
    ----------
    group_id, backend_job_id : GroupId, BackendJobId
        The already registered immutable group and its sole Balsam Job.
    batch_digest : str
        Digest of the complete finite carrier request.
    stdout_path, stderr_path : str
        Absolute output destinations generated from the persisted carrier
        intent.  No other SDK options are represented by this record.
    """

    schema: ClassVar[str] = "tpen-executor/balsam-batch-intent-v1"
    group_id: GroupId
    backend_job_id: BackendJobId
    batch_digest: str
    stdout_path: str
    stderr_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, GroupId):
            raise TypeError("group_id must be GroupId")
        if not isinstance(self.backend_job_id, BackendJobId):
            raise TypeError("backend_job_id must be BackendJobId")
        sha256_text(self.batch_digest, "batch_digest")
        for name in ("stdout_path", "stderr_path"):
            value = text(getattr(self, name), name)
            if not Path(value).is_absolute():
                raise BalsamGroupRefusal(f"{name} must be absolute")

    def to_dict(self) -> dict[str, Any]:
        """Return the strict finite wire mapping."""

        return {
            "schema": self.schema,
            "group_id": self.group_id.to_dict(),
            "backend_job_id": self.backend_job_id.to_dict(),
            "batch_digest": self.batch_digest,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
        }

    def encode(self) -> bytes:
        """Return canonical durable bytes."""

        return canonical_bytes(self.to_dict())

    @classmethod
    def decode(cls, payload: bytes | str | Mapping[str, Any]) -> "BalsamBatchIntent":
        """Decode one canonical persisted carrier intent."""

        data = decode_json(payload)
        fields(data, {"schema", "group_id", "backend_job_id", "batch_digest", "stdout_path", "stderr_path"},
               cls.schema, cls.__name__)
        return cls(
            GroupId.from_mapping(data["group_id"]),
            BackendJobId.from_mapping(data["backend_job_id"]),
            data["batch_digest"], data["stdout_path"], data["stderr_path"],
        )


@frozen_record
class BalsamBatchReceipt:
    """Acknowledgement bound to the exact carrier intent."""

    schema: ClassVar[str] = "tpen-executor/balsam-batch-receipt-v1"
    intent: BalsamBatchIntent
    batch_job_id: BatchJobId

    def __post_init__(self) -> None:
        if not isinstance(self.intent, BalsamBatchIntent):
            raise TypeError("intent must be BalsamBatchIntent")
        if not isinstance(self.batch_job_id, BatchJobId):
            raise TypeError("batch_job_id must be BatchJobId")

    def to_dict(self) -> dict[str, Any]:
        """Return the strict acknowledgement mapping."""

        return {"schema": self.schema, "intent": self.intent.to_dict(),
                "batch_job_id": self.batch_job_id.to_dict()}

    def encode(self) -> bytes:
        """Return canonical durable bytes."""

        return canonical_bytes(self.to_dict())

    @classmethod
    def decode(cls, payload: bytes | str | Mapping[str, Any]) -> "BalsamBatchReceipt":
        """Decode one canonical acknowledgement."""

        data = decode_json(payload)
        fields(data, {"schema", "intent", "batch_job_id"}, cls.schema, cls.__name__)
        return cls(BalsamBatchIntent.decode(data["intent"]), BatchJobId.from_mapping(data["batch_job_id"]))


def derive_batch_output_paths(intent: BalsamBatchIntent) -> tuple[str, str]:
    """Derive the only two output paths from a persisted finite intent.

    A caller cannot supply a second mapping of SDK options: decoding the
    canonical intent is the source of truth for both destinations.
    """

    if not isinstance(intent, BalsamBatchIntent):
        raise TypeError("intent must be BalsamBatchIntent")
    return intent.stdout_path, intent.stderr_path


def _record_paths(store: DirectoryStore, intent: BalsamBatchIntent) -> tuple[Path, Path, Path]:
    """Return the one-group carrier journal paths."""

    root = Path(store.root) / "balsam-batches" / digest(intent.group_id.to_dict())
    ensure_durable_directory(store.root)
    ensure_durable_directory(str(root.parent.parent))
    ensure_durable_directory(str(root.parent))
    ensure_durable_directory(str(root))
    return root / "intent.json", root / "receipt.json", root / "refusal.json"


def _read_intent(path: Path) -> BalsamBatchIntent:
    """Read and canonicalize one existing carrier intent."""

    if path.is_symlink() or not path.is_file():
        raise BalsamGroupRefusal("Balsam BatchJob intent is not a regular file")
    payload = path.read_bytes()
    record = BalsamBatchIntent.decode(payload)
    if payload != record.encode():
        raise BalsamGroupRefusal("Balsam BatchJob intent is noncanonical")
    return record


def submit_fresh_batch(
    intent: BalsamBatchIntent,
    registration: BalsamGroupRegistrationReceipt,
    store: DirectoryStore,
    client: BalsamBatchClient,
    *,
    authority: Callable[[], None],
) -> BalsamBatchReceipt:
    """Journal and submit one fresh BatchJob, with replay and park semantics.

    The registration receipt proves that this group has exactly one Balsam Job.
    The authority callback is evaluated before intent publication and again
    immediately before the external mutation.  If an acknowledgement is lost,
    the durable intent remains and every replay parks without another call.
    """

    if not isinstance(intent, BalsamBatchIntent) or not isinstance(registration, BalsamGroupRegistrationReceipt):
        raise TypeError("intent and registration must be typed records")
    if not isinstance(store, DirectoryStore) or not callable(authority):
        raise TypeError("store and authority callback are required")
    if registration.intent.group_id != intent.group_id or registration.backend_job_id != intent.backend_job_id:
        raise BalsamGroupRefusal("BatchJob intent crosses the registered group or Job")
    intent_path, receipt_path, refusal_path = _record_paths(store, intent)
    if os.path.lexists(intent_path):
        authority = lambda: None
    authority()
    intent_path, receipt_path, refusal_path = _record_paths(store, intent)
    if any(path.is_symlink() for path in (intent_path, receipt_path, refusal_path)):
        raise BalsamGroupRefusal("Balsam BatchJob journal contains a symlink")
    if os.path.lexists(intent_path):
        stored = _read_intent(intent_path)
        if stored != intent:
            raise BalsamGroupRefusal("BatchJob intent conflicts with the immutable group")
        if os.path.lexists(receipt_path):
            if not receipt_path.is_file():
                raise BalsamGroupRefusal("Balsam BatchJob receipt is not a regular file")
            payload = receipt_path.read_bytes()
            receipt = BalsamBatchReceipt.decode(payload)
            if payload != receipt.encode() or receipt.intent != intent:
                raise BalsamGroupRefusal("BatchJob receipt crosses the persisted intent")
            return receipt
        if os.path.lexists(refusal_path):
            if not refusal_path.is_file():
                raise BalsamGroupRefusal("Balsam BatchJob refusal is not a regular file")
            raise BalsamGroupRefusal("BatchJob submission was refused")
        raise BalsamGroupParked("BatchJob acknowledgement is unknown; no second submission")
    if os.path.lexists(receipt_path) or os.path.lexists(refusal_path):
        raise BalsamGroupRefusal("BatchJob acknowledgement exists without its intent")
    try:
        publish_json_exclusive(str(intent_path), intent.encode())
    except FileExistsError as error:
        raise BalsamGroupParked("BatchJob intent is already owned; reconcile before retry") from error
    try:
        authority()
        stdout_path, stderr_path = derive_batch_output_paths(intent)
        batch_job_id = client.submit_batch(
            intent.group_id, intent.backend_job_id, intent.batch_digest,
            stdout_path=stdout_path, stderr_path=stderr_path,
        )
        if not isinstance(batch_job_id, BatchJobId):
            raise BalsamGroupParked("BatchJob acknowledgement has no typed identity")
        receipt = BalsamBatchReceipt(intent, batch_job_id)
        publish_json_exclusive(str(receipt_path), receipt.encode())
        return receipt
    except BalsamGroupRefusal as error:
        try:
            publish_json_exclusive(str(refusal_path), canonical_bytes({
                "schema": "tpen-executor/balsam-batch-refusal-v1",
                "intent": intent.to_dict(), "reason": str(error),
            }))
        except OSError:
            pass
        raise
    except BalsamGroupParked:
        raise
    except Exception as error:
        raise BalsamGroupParked("BatchJob acknowledgement is unknown; no second submission") from error


def associate_full_node(
    assignment: FullNodeAssignment,
    registration: BalsamGroupRegistrationReceipt,
    *,
    observed_job_id: BackendJobId | None = None,
):
    """Validate S6 scope and return the S2 allocation handed to the controller."""

    if not isinstance(assignment, FullNodeAssignment) or not isinstance(registration, BalsamGroupRegistrationReceipt):
        raise TypeError("assignment and registration must be typed records")
    if assignment.job_id != registration.backend_job_id:
        raise BalsamGroupRefusal("S6 assignment is not associated with the registered Job")
    if observed_job_id is not None and observed_job_id != assignment.job_id:
        raise BalsamGroupRefusal("observed Job is not associated with the S6 assignment")
    # S6 has already checked the full-node evidence; S2 receives only its
    # effective one-host allocation, never the parent's broader nodefile.
    return assignment.to_configured_allocation()


def run_validated_association(
    binding: Any,
    observed_job_id: BackendJobId,
    registration: BalsamGroupRegistrationReceipt,
    assignment: FullNodeAssignment,
    group: ReadyGroupV2,
    *,
    registration_receipt_digest: str,
    claimant_nonce: str,
    observed_at_utc: str,
    member_inputs: Sequence[Any],
    store: DirectoryStore,
    admission_port: Any,
    executor: Any,
):
    """Invoke S2's controller only after validating the S6 association."""

    from .controller import GroupControllerFactsV1, run_group_controller

    if registration.intent.group_id != group.group_id:
        raise BalsamGroupRefusal("registration is not associated with the immutable group")
    allocation = associate_full_node(assignment, registration, observed_job_id=observed_job_id)
    facts = GroupControllerFactsV1(
        group, allocation, registration_receipt_digest, text(claimant_nonce, "claimant_nonce"),
        text(observed_at_utc, "observed_at_utc"), tuple(member_inputs),
    )
    return run_group_controller(binding, observed_job_id, facts, store, admission_port, executor)


submit_batch_once = submit_fresh_batch
submit_carrier_batch = submit_fresh_batch


__all__ = [
    "BalsamBatchClient", "BalsamBatchIntent", "BalsamBatchReceipt",
    "associate_full_node", "derive_batch_output_paths", "run_validated_association",
    "submit_batch_once", "submit_carrier_batch", "submit_fresh_batch",
]
