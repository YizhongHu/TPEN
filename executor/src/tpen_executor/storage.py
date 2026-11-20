"""Durable, fenced attempt receipts for an executor allocation."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from collections.abc import Mapping
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
from uuid import uuid4

from ._codec import canonical_bytes, decode_json, digest, fields, sha256_text, text, thaw
from .allocations import AllocationAttemptV2, ConfiguredAllocation, ControllerGenerationV1, LaunchIntentV1, MemberInvocationV2
from .contracts import AllocationId, AttemptId, BackendJobId, GroupId, InvocationId, ReadyGroup, ReadyGroupV2
from .results import (
    AcceptedSegment, CoreOutcomeRecord, ManifestAttempt, OutcomeReceipt, PLACEMENT_RECEIPT_TYPES, RemoteTeardownReceipt, RunManifest,
    SciencePlacementReceipt, SciencePlacementReceiptV2, SciencePlacementReceiptV3, StartReceipt,
    SuccessorAdmissionRecord, MemberPublicationV2, ReconciliationPublicationV2, decode_placement_receipt,
)
from .runners import CoreRunEvidence
from ._records import frozen_record


class FenceRejected(RuntimeError):
    """Raised when an attempt fence is stale, crossed, or already finished."""


class AdmissionFork(FenceRejected):
    """Raised when one predecessor is presented with a different successor."""

    code = "fork"


class ControllerGenerationRejected(FenceRejected):
    """Raised when a controller generation is stale, crossed, or forked."""


class LaunchIntentRejected(FenceRejected):
    """Raised when a member launch intent cannot be created or read safely."""


@frozen_record
class AttemptFence:
    """Opaque identity binding for one store-owned attempt.

    Parameters
    ----------
    root : str
        Lexically normalized store root that owns the attempt.
    group_digest : str
        Digest of the :class:`~tpen_executor.contracts.ReadyGroup` that issued
        the attempt.
    task_id : str
        Safe, reversible task namespace component.
    allocation_id : str
        Explicit allocation identity bound into the start receipt.
    attempt_id : AttemptId
        Fresh attempt identity.
    output_dir : str
        Fenced attempt directory below ``root``.

    Notes
    -----
    A fence is not transferable between stores or attached allocations.  The
    store rechecks all fields against the durable start receipt before writing
    an outcome.
    """

    root: str
    group_digest: str
    task_id: str
    allocation_id: str
    attempt_id: AttemptId
    output_dir: str


def _lexical(path: str | os.PathLike[str]) -> str:
    """Return an absolute normalized path without filesystem resolution."""
    return os.path.abspath(os.path.normpath(os.fspath(path)))


def _inside(root: str, path: str) -> bool:
    """Check lexical containment, including the root itself."""
    return path == root or path.startswith(root + os.sep)


_CORE_EVIDENCE_FENCE_REASON = "core evidence run_dir crosses the attempt fence"


def _strict_real_directory(path: str | os.PathLike[str]) -> str:
    """Return an existing directory's canonical path or reject the fence.

    Parameters
    ----------
    path : str or os.PathLike
        Directory path whose target must exist before resolution.

    Returns
    -------
    str
        The canonical path of the existing directory.

    Raises
    ------
    FenceRejected
        If ``path`` is missing, cannot be resolved, or is not a directory.
    """
    try:
        resolved = os.path.realpath(os.fspath(path), strict=True)
    except (OSError, ValueError) as error:
        raise FenceRejected(_CORE_EVIDENCE_FENCE_REASON) from error
    if not os.path.isdir(resolved):
        raise FenceRejected(_CORE_EVIDENCE_FENCE_REASON)
    return resolved


_SAFE_TASK_ID = re.compile(r"[A-Za-z0-9._:-]+\Z")
_TASK_LOCK = ".task.lock"
_SUCCESSOR_ADMISSIONS = "successor-admissions"
_SUCCESSOR_ADMISSION_FILENAME = "successor-admission.json"
_SUCCESSOR_ADMISSION_TEMP_PREFIX = ".tmp-admission-"
_SUCCESSOR_ADMISSION_SCHEMA = "tpen.execution/successor-admission-request-v1"
_REMOTE_TEARDOWN_FILENAME = "remote-teardown.json"
_TEARDOWN_UNCONFIRMED_REASON = "teardown-unconfirmed"
_LEAF_TEARDOWN_SCHEMA = "tpen-executor/leaf-teardown-v1"
_CONTROLLER_LEDGER_DIR = "controller-generations"
_CONTROLLER_LOCK = "controller.lock"
_LAUNCH_INTENT_FILENAME = "launch-intent.json"


def _fsync_directory(path: str) -> None:
    """Persist directory entries for a directory that is already durable."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _existing(path: str, name: str, error_type: type[Exception] = FenceRejected) -> os.stat_result:
    """Return an existing non-symlink entry or raise a typed error."""
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise error_type(f"{name} is missing") from error
    if stat.S_ISLNK(info.st_mode):
        raise error_type(f"{name} must not be a symlink")
    return info


def _ensure_directory(path: str) -> None:
    """Create one directory and persist its parent link, refusing symlinks."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        try:
            os.mkdir(path)
        except FileExistsError:
            try:
                info = os.lstat(path)
            except FileNotFoundError as error:
                raise FenceRejected(f"store component is missing: {path}") from error
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise FenceRejected(f"store component is not a directory: {path}")
            _fsync_directory(os.path.dirname(path))
            return
        _fsync_directory(os.path.dirname(path))
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise FenceRejected(f"store component is not a directory: {path}")


def ensure_durable_directory(path: str) -> None:
    """Create or validate one directory and durably publish its parent link.

    Parameters
    ----------
    path : str
        Directory component to publish. Its parent must already be durable;
        callers publish an ancestor chain in parent-to-child order.

    Notes
    -----
    Existing entries are validated as directories, never symlinks, and their
    parent is synced too: another creator may have stopped after mkdir. This
    POSIX publication contract assumes no concurrent removal or replacement of
    validated directories. Sync failures propagate before publication succeeds.
    Existing private storage callers retain their original contract.
    """
    path = _lexical(path)
    try:
        os.mkdir(path)
    except FileExistsError:
        _require_directory(path, "publication directory")
        # Visibility does not prove that another creator synced this link.
        _fsync_directory(os.path.dirname(path))  # durable existing publication
        return
    _require_directory(path, "publication directory")
    _fsync_directory(os.path.dirname(path))


def require_regular_file(path: str, name: str) -> None:
    """Require a non-symlink regular file, raising ``FenceRejected`` otherwise.

    Parameters
    ----------
    path : str
        File to validate before reading a published record.
    name : str
        Human-readable record name used in validation errors.
    """
    _require_regular(path, name)


def publish_json_exclusive(path: str, payload: bytes) -> None:
    """Publish encoded JSON bytes durably without replacing an existing entry.

    Parameters
    ----------
    path : str
        Destination under an already durable directory chain.
    payload : bytes
        Complete JSON record encoded and validated by its owning codec.

    Notes
    -----
    Syncs the payload before exclusive linking, then syncs the destination
    directory. Raises ``FileExistsError`` on collision; other storage failures
    propagate. A failed call can leave a published record, so callers must
    treat failure as uncertain rather than permission to repeat a side effect.
    """
    _write_new(path, payload)


def _mkdir_new(path: str) -> None:
    """Create one fresh directory and persist its parent link."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        try:
            os.mkdir(path)
        except FileExistsError as error:
            raise FenceRejected("attempt directory already exists") from error
        _fsync_directory(os.path.dirname(path))
        return
    raise FenceRejected("attempt directory already exists")


def _require_directory(path: str, name: str, error_type: type[Exception] = FenceRejected) -> None:
    """Require a real directory at ``path``."""
    info = _existing(path, name, error_type)
    if not stat.S_ISDIR(info.st_mode):
        raise error_type(f"{name} is not a directory")


def _require_regular(path: str, name: str, error_type: type[Exception] = FenceRejected) -> None:
    """Require a non-symlink regular file at ``path``."""
    info = _existing(path, name, error_type)
    if not stat.S_ISREG(info.st_mode):
        raise error_type(f"{name} is not a regular file")


@contextmanager
def _task_lock(path: str):
    """Hold one POSIX advisory lock descriptor for a task critical section."""
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise FenceRejected("task lock is not a regular file")
        fcntl.lockf(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.lockf(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _write_new(path: str, payload: bytes, *, temporary_dir: str | None = None) -> None:
    """Write a new file durably, refusing replacement of an existing file."""
    parent = os.path.dirname(path)
    _require_directory(parent, "receipt parent")
    staging_dir = parent if temporary_dir is None else _lexical(temporary_dir)
    _require_directory(staging_dir, "receipt staging directory")
    fd, temporary = tempfile.mkstemp(
        prefix=_SUCCESSOR_ADMISSION_TEMP_PREFIX if temporary_dir is not None else ".tmp-",
        dir=staging_dir,
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(path):
            _existing(path, "receipt destination")
            raise FileExistsError(path)
        # The complete, fsynced payload becomes visible in one no-replace link.
        os.link(temporary, path)
        _fsync_directory(parent)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


@frozen_record
class DirectoryStore:
    """The sole durable receipt writer beneath one lexical root.

    Parameters
    ----------
    root : str or os.PathLike
        Directory used for all receipt and fenced-output writes.
    """

    root: str | os.PathLike[str]

    def __post_init__(self) -> None:
        root = _lexical(self.root)
        if os.path.lexists(root):
            _require_directory(root, "store root")
        object.__setattr__(self, "root", root)

    def record_pre_start_receipt(self, group, backend_job_id, observed_job_id, reason: str) -> None:
        """Durably record one refusal keyed by group and backend observations."""
        ensure_durable_directory(self.root)
        group_key = getattr(group, "identity", digest(getattr(group, "to_dict", lambda: {"group": str(group)})()))
        key = digest({"group": group_key, "backend_job_id": str(backend_job_id), "observed_job_id": str(observed_job_id)})
        directory = os.path.join(self.root, "groups", group_key)
        ensure_durable_directory(os.path.join(self.root, "groups"))
        ensure_durable_directory(directory)
        path = os.path.join(directory, f"pre-start-{key}.json")
        payload = canonical_bytes({"schema": "tpen-executor/pre-start-v1", "group": group_key, "backend_job_id": str(backend_job_id), "observed_job_id": str(observed_job_id), "reason": str(reason)})
        if os.path.lexists(path):
            if Path(path).read_bytes() != payload:
                raise FenceRejected("pre-start receipt conflicts")
            return
        _write_new(path, payload)

    def _group_dir(self, group: ReadyGroup | ReadyGroupV2) -> str:
        path = _lexical(os.path.join(self.root, "groups", group.identity))
        if not _inside(self.root, path):
            raise FenceRejected("group path escapes store root")
        return path

    def _controller_dir(self, group: ReadyGroupV2 | GroupId) -> str:
        """Return the durable controller ledger directory for a GroupId."""
        if not isinstance(group, (ReadyGroupV2, GroupId)):
            raise TypeError("controller authority requires a ReadyGroupV2")
        # Hashing the opaque identity keeps arbitrary GroupId text out of paths.
        group_id = group.group_id if isinstance(group, ReadyGroupV2) else group
        group_key = digest(group_id.to_dict())
        path = _lexical(os.path.join(self.root, "groups", group_key, _CONTROLLER_LEDGER_DIR))
        if not _inside(self.root, path):
            raise ControllerGenerationRejected("controller ledger escapes store root")
        return path

    def _controller_records(self, directory: str, group_id: GroupId) -> list[ControllerGenerationV1]:
        """Load and validate the complete canonical generation chain."""
        records: list[tuple[int, ControllerGenerationV1]] = []
        for name in os.listdir(directory):
            if name == _CONTROLLER_LOCK or name.startswith(".tmp-"):
                continue
            match = re.fullmatch(r"generation-(\d+)\.json", name)
            if match is None:
                raise ControllerGenerationRejected("unexpected controller ledger entry")
            path = os.path.join(directory, name)
            _require_regular(path, "controller generation", ControllerGenerationRejected)
            encoded = Path(path).read_bytes()
            try:
                record = ControllerGenerationV1.decode(encoded)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                raise ControllerGenerationRejected("controller generation is malformed") from error
            if encoded != record.encode() or int(match.group(1)) != record.generation or record.group_id != group_id:
                raise ControllerGenerationRejected("controller generation identity is crossed")
            records.append((record.generation, record))
        records.sort(key=lambda item: item[0])
        if records and records[0][0] != 1 or any(number != index for index, (number, _) in enumerate(records, 1)):
            raise ControllerGenerationRejected("controller generation chain is not contiguous")
        previous = None
        for _, record in records:
            expected = None if previous is None else previous.record_digest
            if record.predecessor_record_digest != expected:
                raise ControllerGenerationRejected("controller generation predecessor is crossed")
            previous = record
        return [record for _, record in records]

    def _prepare_controller_dir(self, group: ReadyGroupV2) -> str:
        """Create and durably publish the controller ledger ancestry."""
        directory = self._controller_dir(group)
        group_dir = os.path.dirname(directory)
        ensure_durable_directory(self.root)
        ensure_durable_directory(os.path.join(self.root, "groups"))
        ensure_durable_directory(group_dir)
        ensure_durable_directory(directory)
        return directory

    def _current_controller_locked(self, directory: str, group_id: GroupId,
                                   expected: ControllerGenerationV1 | None = None) -> ControllerGenerationV1:
        """Load, sync, and optionally compare authority with the lock held."""
        records = self._controller_records(directory, group_id)
        _fsync_directory(directory)  # authority gate barrier
        if not records:
            raise ControllerGenerationRejected("controller ledger is empty")
        current = records[-1]
        if expected is not None and current.encode() != expected.encode():
            raise ControllerGenerationRejected("controller record is not current")
        return current

    def acquire_controller_generation(self, group: ReadyGroupV2, record: ControllerGenerationV1) -> ControllerGenerationV1:
        """Durably acquire the next controller generation for ``group``.

        Identical retries return the existing canonical record.  A skipped,
        stale, or forked generation is refused before any replacement write.
        """
        if not isinstance(record, ControllerGenerationV1) or record.group_id != group.group_id:
            raise ControllerGenerationRejected("controller generation is crossed")
        directory = self._prepare_controller_dir(group)
        with _task_lock(os.path.join(directory, _CONTROLLER_LOCK)):
            records = self._controller_records(directory, group.group_id)
            _fsync_directory(directory)  # replay barrier
            current = records[-1] if records else None
            if current is not None and record.generation == current.generation and record.encode() == current.encode():
                return current
            expected = 1 if current is None else current.generation + 1
            predecessor = None if current is None else current.record_digest
            if record.generation != expected or record.predecessor_record_digest != predecessor:
                raise ControllerGenerationRejected("controller generation is stale, skipped, or forked")
            path = os.path.join(directory, f"generation-{record.generation}.json")
            try:
                _write_new(path, record.encode())
            except FileExistsError as error:
                raise ControllerGenerationRejected("controller generation write is ambiguous") from error
            return record

    def require_current_controller(self, group: ReadyGroupV2 | GroupId, generation: int, nonce: str) -> ControllerGenerationV1:
        """Re-read and require the highest generation and exact claimant nonce."""
        if type(generation) is not int or generation < 1 or type(nonce) is not str:
            raise ControllerGenerationRejected("invalid controller authority")
        directory = self._controller_dir(group)
        try:
            with _task_lock(os.path.join(directory, _CONTROLLER_LOCK)):
                current = self._current_controller_locked(directory, group.group_id if isinstance(group, ReadyGroupV2) else group)
        except (FileNotFoundError, OSError, FenceRejected, ValueError, TypeError) as error:
            raise ControllerGenerationRejected("controller ledger is unavailable") from error
        if current.generation != generation or current.claimant_nonce != nonce:
            raise ControllerGenerationRejected("controller is not current")
        return current

    def create_launch_intent(self, fence: AttemptFence, controller_record: ControllerGenerationV1,
                             invocation: MemberInvocationV2) -> LaunchIntentV1:
        """Exclusively create one launch intent after a current-controller check."""
        if not isinstance(controller_record, ControllerGenerationV1) or not isinstance(invocation, MemberInvocationV2):
            raise TypeError("launch intent requires typed controller and invocation records")
        directory = self._controller_dir(controller_record.group_id)
        with _task_lock(os.path.join(directory, _CONTROLLER_LOCK)):
            current = self._current_controller_locked(directory, controller_record.group_id, controller_record)
            task_dir = self._task_dir_for_fence(fence)
            with _task_lock(os.path.join(task_dir, _TASK_LOCK)):
                start = self._validate_fence(fence, task_dir)
                self._validate_invocation(start, invocation)  # durable intent gate
                if (invocation.attempt_id != fence.attempt_id
                        or invocation.group_digest != fence.group_digest
                        or invocation.allocation_attempt.group_id != controller_record.group_id
                        or invocation.allocation_attempt.allocation_id != controller_record.allocation_id
                        or invocation.allocation_attempt.backend_job_id != controller_record.backend_job_id):
                    raise LaunchIntentRejected("launch intent binding crosses its fence")
                intent = LaunchIntentV1(current.group_id, fence.task_id, fence.attempt_id, invocation.invocation_id,
                                        current.generation, current.claimant_nonce,
                                        current.allocation_id, current.backend_job_id)
                intent_path = os.path.join(fence.output_dir, _LAUNCH_INTENT_FILENAME)
                if os.path.lexists(intent_path):
                    raise LaunchIntentRejected("launch intent already exists")
                try:
                    _write_new(intent_path, intent.encode())
                except FileExistsError as error:
                    raise LaunchIntentRejected("launch intent already exists") from error
                return intent

    def read_launch_intent(self, fence: AttemptFence) -> LaunchIntentV1:
        """Read a canonical historical launch intent for recovery.

        The attempt start and invocation binding are revalidated, but current
        controller authority is deliberately not required: an old intent is
        evidence for permanent parking, never permission to relaunch.
        """
        task_dir = self._task_dir_for_fence(fence)
        path = os.path.join(fence.output_dir, _LAUNCH_INTENT_FILENAME)
        with _task_lock(os.path.join(task_dir, _TASK_LOCK)):
            try:
                start = self._validate_fence_immutable(fence, task_dir)
                invocation_path = os.path.join(fence.output_dir, "invocation.json")
                _require_regular(invocation_path, "invocation binding")
                invocation_bytes = Path(invocation_path).read_bytes()
                invocation = MemberInvocationV2.decode(invocation_bytes)
                self._validate_invocation(start, invocation)
                _require_regular(path, "launch intent", LaunchIntentRejected)
                encoded = Path(path).read_bytes()
                intent = LaunchIntentV1.decode(encoded)
                if encoded != intent.encode():
                    raise LaunchIntentRejected("launch intent is not canonical")
            except (OSError, ValueError, TypeError) as error:
                raise LaunchIntentRejected("launch intent is unreadable") from error
        if (intent.attempt_id != fence.attempt_id or intent.task_id != fence.task_id
                or intent.group_id != invocation.allocation_attempt.group_id
                or intent.invocation_id != invocation.invocation_id
                or intent.allocation_id != invocation.allocation_attempt.allocation_id
                or intent.backend_job_id != invocation.allocation_attempt.backend_job_id):
            raise LaunchIntentRejected("launch intent is crossed")
        return intent

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        """Validate the reversible task directory namespace."""
        text(task_id, "task_id")
        if task_id in {".", ".."} or _SAFE_TASK_ID.fullmatch(task_id) is None:
            raise ValueError("task_id must match [A-Za-z0-9._:-]+ and must not be a traversal component")

    def _prepare_task_dir(self, group: ReadyGroup | ReadyGroupV2, task_id: str) -> str:
        """Create and durably link the store ancestry for one task."""
        _ensure_directory(self.root)
        groups_dir = _lexical(os.path.join(self.root, "groups"))
        _ensure_directory(groups_dir)
        group_dir = self._group_dir(group)
        _ensure_directory(group_dir)
        task_dir = _lexical(os.path.join(group_dir, task_id))
        if not _inside(group_dir, task_dir):
            raise FenceRejected("task path escapes store root")
        _ensure_directory(task_dir)
        return task_dir

    def _task_dir_for_fence(self, fence: AttemptFence) -> str:
        """Return the validated task directory named by a fence."""
        if not isinstance(fence, AttemptFence) or fence.root != self.root:
            raise FenceRejected("fence belongs to another store")
        try:
            self._validate_task_id(fence.task_id)
            group_dir = _lexical(os.path.join(self.root, "groups", fence.group_digest))
            if not _inside(self.root, group_dir):
                raise FenceRejected("fence path escapes store root")
        except (TypeError, ValueError, FenceRejected) as error:
            raise FenceRejected("fence has an invalid group or task identity") from error
        task_dir = _lexical(os.path.join(group_dir, fence.task_id))
        if not _inside(group_dir, task_dir):
            raise FenceRejected("fence path escapes store root")
        _require_directory(self.root, "store root")
        _require_directory(_lexical(os.path.join(self.root, "groups")), "groups directory")
        _require_directory(group_dir, "group directory")
        _require_directory(task_dir, "task directory")
        return task_dir

    def begin_attempt(
        self,
        group: ReadyGroup | ReadyGroupV2,
        task_id: str,
        allocation: ConfiguredAllocation,
        started_at_utc: str = "",
        transport_kind: str | None = "local",
        *,
        allocation_attempt: AllocationAttemptV2 | None = None,
        invocation_id: InvocationId | None = None,
    ) -> AttemptFence:
        """Create a fresh fenced output directory and durable start receipt.

        Parameters
        ----------
        group : ReadyGroup
            Group containing ``task_id``.
        task_id : str
            Member task to start.
        allocation : ConfiguredAllocation
            Explicit allocation identity and backend job.
        started_at_utc : str, optional
            UTC timestamp; an empty value is accepted for caller compatibility.
        transport_kind : {"local", "remote"} or None, optional
            Durable transport identity for continuation admission. ``None`` is
            retained as unknown and therefore fails closed at the gate.
        allocation_attempt : AllocationAttemptV2 or None, optional
            Required for v2 groups; forbidden for legacy v1 groups.
        invocation_id : InvocationId or None, optional
            Controller-issued fresh process identity, required for v2 starts.

        Returns
        -------
        AttemptFence
            Fence required by :meth:`finish_attempt`.

        Raises
        ------
        TypeError
            If an identity, timestamp, or other argument has the wrong type.
        ValueError
            If ``task_id`` is not a group member or is not a safe path
            component.
        FenceRejected
            If a store-owned path is crossed by a symlink or an attempt ID is
            already present.
        """
        if not isinstance(group, (ReadyGroup, ReadyGroupV2)) or not isinstance(allocation, ConfiguredAllocation):
            raise TypeError("group and allocation have invalid types")
        text(task_id, "task_id")
        self._validate_task_id(task_id)
        if task_id not in {member.task_id for member in group.members}:
            raise ValueError("task_id is not a member of group")
        if type(started_at_utc) is not str:
            raise TypeError("started_at_utc must be a string")
        if transport_kind not in {None, "local", "remote"}:
            raise ValueError("transport_kind must be local, remote, or None")
        if isinstance(group, ReadyGroupV2):
            if not isinstance(allocation_attempt, AllocationAttemptV2):
                raise TypeError("v2 start requires an allocation-attempt binding")
            if (allocation_attempt.group_id != group.group_id
                    or allocation_attempt.allocation_id != allocation.allocation_id
                    or allocation_attempt.backend_job_id != allocation.backend_job_id):
                raise FenceRejected("crossed allocation-attempt binding")
            if not isinstance(invocation_id, InvocationId):
                raise TypeError("v2 start requires a controller-issued InvocationId")
        elif allocation_attempt is not None or invocation_id is not None:
            raise ValueError("v1 start cannot carry v2 identities")
        stamp = started_at_utc or "unknown"
        task_dir = self._prepare_task_dir(group, task_id)
        lock_path = os.path.join(task_dir, _TASK_LOCK)
        _existing(lock_path, "task lock") if os.path.lexists(lock_path) else None
        with _task_lock(lock_path):
            attempt = AttemptId(uuid4().hex)
            output_dir = _lexical(os.path.join(task_dir, attempt.value))
            _mkdir_new(output_dir)
            receipt = StartReceipt(task_id, attempt, allocation.allocation_id, allocation.backend_job_id, group.identity, stamp, output_dir, transport_kind)
            _write_new(os.path.join(output_dir, "start.json"), receipt.encode())
            if allocation_attempt is not None:
                task = next(member for member in group.members if member.task_id == task_id)
                binding = MemberInvocationV2(allocation_attempt, group.identity, task_id, task.lineage_id, attempt, invocation_id)
                _write_new(os.path.join(output_dir, "invocation.json"), binding.encode())
            pointer = os.path.join(task_dir, "current-attempt.json")
            _write_pointer(pointer, attempt.value)
            return AttemptFence(self.root, group.identity, task_id, allocation.allocation_id.value, attempt, output_dir)

    def begin_admitted_successor(
        self,
        group: ReadyGroupV2,
        task_id: str,
        allocation: ConfiguredAllocation,
        started_at_utc: str,
        transport_kind: str | None,
        *,
        allocation_attempt: AllocationAttemptV2,
        admission: SuccessorAdmissionRecord,
    ) -> AttemptFence:
        """Start the exact successor recorded by the durable CX-04 admission.

        The predecessor task lock serializes admission revalidation, exact
        successor identity consumption, and the current-attempt pointer.  A
        replay of the same canonical start returns its original fence; any
        fork, crossed binding, legacy group, or foreign path is refused.
        """
        if not isinstance(group, ReadyGroupV2):
            raise TypeError("admitted successor requires a v2 group")
        if not isinstance(allocation, ConfiguredAllocation) or not isinstance(allocation_attempt, AllocationAttemptV2):
            raise TypeError("admitted successor requires typed allocation records")
        if not isinstance(admission, SuccessorAdmissionRecord) or admission.successor_binding is None:
            raise TypeError("admitted successor requires a bound admission record")
        self._validate_task_id(task_id)
        if task_id not in {member.task_id for member in group.members}:
            raise ValueError("task_id is not a member of group")
        if type(started_at_utc) is not str or transport_kind not in {None, "local", "remote"}:
            raise TypeError("invalid successor start metadata")
        successor = admission.successor_binding
        if (admission.predecessor_task_id != task_id
                or admission.successor_task_id != task_id):
            raise FenceRejected("crossed-successor-admission")
        if (allocation_attempt.group_id != group.group_id
                or allocation_attempt.allocation_id != allocation.allocation_id
                or allocation_attempt.backend_job_id != allocation.backend_job_id
                or successor.group_digest != group.identity
                or successor.task_id != task_id
                or successor.lineage_id != next(member.lineage_id for member in group.members if member.task_id == task_id)
                or successor.allocation_attempt != allocation_attempt):
            raise FenceRejected("crossed-successor-binding")
        task_dir = self._prepare_task_dir(group, task_id)
        lock_path = os.path.join(task_dir, _TASK_LOCK)
        with _task_lock(lock_path):
            records = self._admission_records_for_task(task_dir, task_id)
            matches = [record for record in records if record.predecessor_attempt_id == admission.predecessor_attempt_id]
            if len(matches) != 1 or matches[0].encode() != admission.encode():
                raise AdmissionFork("admission record is missing or differs")
            predecessor_dir = _lexical(os.path.join(task_dir, admission.predecessor_attempt_id.value))
            predecessor_start_path = os.path.join(predecessor_dir, "start.json")
            _require_regular(predecessor_start_path, "predecessor start receipt")
            predecessor_start = StartReceipt.decode(Path(predecessor_start_path).read_bytes())
            predecessor_fence = AttemptFence(
                self.root, group.identity, task_id,
                predecessor_start.allocation_id.value,
                admission.predecessor_attempt_id, predecessor_dir,
            )
            pointer = os.path.join(task_dir, "current-attempt.json")
            current = None
            if os.path.lexists(pointer):
                _require_regular(pointer, "current attempt pointer")
                current = decode_json(Path(pointer).read_bytes()).get("attempt_id")
            output_dir = _lexical(os.path.join(task_dir, successor.attempt_id.value))
            start = StartReceipt(task_id, successor.attempt_id, allocation.allocation_id,
                                 allocation.backend_job_id, group.identity,
                                 started_at_utc or "unknown", output_dir, transport_kind)
            binding = successor
            start_bytes = start.encode()
            invocation_bytes = binding.encode()
            if os.path.lexists(output_dir):
                _require_directory(output_dir, "admitted successor attempt")
                if current not in {successor.attempt_id.value, admission.predecessor_attempt_id.value}:
                    raise FenceRejected("admitted successor attempt is not current")
                if current == admission.predecessor_attempt_id.value:
                    self._validate_fence(predecessor_fence, task_dir)
                for name, expected in (("start.json", start_bytes), ("invocation.json", invocation_bytes)):
                    path = os.path.join(output_dir, name)
                    if os.path.lexists(path):
                        _require_regular(path, name)
                        if Path(path).read_bytes() != expected:
                            raise AdmissionFork("admitted successor start differs")
                    elif current == admission.predecessor_attempt_id.value:
                        _write_new(path, expected)
                    else:
                        raise FenceRejected("admitted successor receipt is missing")
                if current == admission.predecessor_attempt_id.value:
                    _fsync_directory(output_dir)
                    _write_pointer(pointer, successor.attempt_id.value)
                return AttemptFence(self.root, group.identity, task_id, allocation.allocation_id.value,
                                    successor.attempt_id, output_dir)
            if current != admission.predecessor_attempt_id.value:
                raise FenceRejected("admission predecessor is not current")
            # Validate the predecessor fence while its task lock is held.
            self._validate_fence(predecessor_fence, task_dir)
            _mkdir_new(output_dir)
            _write_new(os.path.join(output_dir, "start.json"), start_bytes)
            _write_new(os.path.join(output_dir, "invocation.json"), invocation_bytes)
            _write_pointer(pointer, successor.attempt_id.value)
            return AttemptFence(self.root, group.identity, task_id, allocation.allocation_id.value,
                                successor.attempt_id, output_dir)

    @staticmethod
    def _validate_invocation(start: StartReceipt, binding: MemberInvocationV2) -> None:
        """Recheck the leaf echo against the controller's immutable binding."""
        path = os.path.join(start.output_dir, "invocation.json")
        _require_regular(path, "invocation binding")
        encoded = Path(path).read_bytes()
        stored = MemberInvocationV2.decode(encoded)
        if encoded != stored.encode() or encoded != binding.encode():
            raise FenceRejected("crossed-invocation")
        if (binding.task_id != start.task_id or binding.attempt_id != start.attempt_id
                or binding.group_digest != start.group_digest
                or binding.allocation_attempt.allocation_id != start.allocation_id
                or binding.allocation_attempt.backend_job_id != start.backend_job_id):
            raise FenceRejected("crossed-invocation")

    def _reconciliation_directory(self) -> str:
        # One allocation attempt binds exactly one group/plan. Key records by
        # allocation-attempt identity across the group store so changing even
        # both caller-supplied group bindings cannot fork an existing record.
        return os.path.join(self.root, "groups", ".reconciliation-publications")

    def read_reconciliation(
        self, group: ReadyGroupV2, allocation_attempt: AllocationAttemptV2,
    ) -> ReconciliationPublicationV2 | None:
        """Decode a canonical publication and recheck its exact plan binding."""
        directory = self._reconciliation_directory()
        path = os.path.join(directory, digest(allocation_attempt.allocation_attempt_id.to_dict()) + ".json")
        if not os.path.lexists(path):
            return None
        _require_directory(self.root, "store root")
        _require_directory(os.path.join(self.root, "groups"), "groups directory")
        _require_directory(directory, "reconciliation directory")
        _require_regular(path, "reconciliation publication")
        encoded = Path(path).read_bytes()
        record = ReconciliationPublicationV2.decode(encoded)
        if (encoded != record.encode() or record.group.encode() != group.encode()
                or record.allocation_attempt.encode() != allocation_attempt.encode()):
            raise FenceRejected("crossed-reconciliation-publication")
        return record

    def publish_reconciliation(
        self,
        group: ReadyGroupV2,
        allocation_attempt: AllocationAttemptV2,
        inputs,
        *,
        bindings: Mapping[str, MemberInvocationV2],
        successors: Mapping[str, tuple[Mapping[str, Any], str, MemberInvocationV2]] | None = None,
    ):
        """Publish one fake-core reconciliation, never launch or restore work.

        ``bindings`` are the leaf echoes of controller-issued invocations.
        ``successors`` contains the accepted CX-04 request, opaque core
        acceptance digest, and proposed fresh binding for each successor.
        Real-core compatibility remains unproven. Invalid/crossed evidence
        returns a parked result without publishing or merging records.
        Successful member admissions are independently durable; a crash before
        the group record is recoverable by identical ledger replay.
        """
        from .recovery import GroupReconciliationV2, MemberReconciliationV2, reconcile_group

        if not isinstance(group, ReadyGroupV2) or not isinstance(allocation_attempt, AllocationAttemptV2):
            raise TypeError("publication requires v2 group and allocation attempt")
        inputs = tuple(inputs)
        result = reconcile_group(group, inputs)

        def park(reason):
            return GroupReconciliationV2(group.group_id, group.identity, tuple(
                MemberReconciliationV2(member.task_id, "park", reason, member.attempt_id)
                for member in result.members
            ))

        successors = {} if successors is None else dict(successors)
        by_task = {item.task_id: item for item in inputs}
        expected = {member.task_id for member in group.members}
        if (allocation_attempt.group_id != group.group_id or set(by_task) != expected
                or set(bindings) != {item.task_id for item in inputs if item.start_receipt is not None}
                or set(successors) != {member.task_id for member in result.members if member.action == "admit_successor"}):
            return park("crossed-reconciliation-input")
        # Missing or untrusted core evidence cannot create even an admission
        # intent. Unknown but well-formed core outcomes may be durably parked.
        invalid_reasons = {
            "missing-core-verdict", "crossed-lineage", "crossed-group-receipt",
            "crossed-or-incomplete-receipt", "missing-attempt-identity",
            "missing-settled-outcome", "unsupported-profile", "core-admission-refused",
            "invalid-successor-evidence", "remote-quiescence-unconfirmed",
        }
        if any(member.reason_code in invalid_reasons for member in result.members):
            return result
        fences = {}
        try:
            # Validate all proposed identity records before any ledger write.
            proposed = []
            for member in result.members:
                admission = None
                if member.action == "admit_successor":
                    request, acceptance, successor = successors[member.task_id]
                    request = self._successor_request_shape(request)
                    admission = SuccessorAdmissionRecord(
                        member.task_id, member.attempt_id,
                        request["core_evidence"]["checkpoint_ref"]["content_id"],
                        request["successor"]["task_id"], AttemptId(request["successor"]["attempt_id"]),
                        digest(request), acceptance, successor,
                    )
                proposed.append(MemberPublicationV2(by_task[member.task_id], member, bindings.get(member.task_id), admission))
            candidate = ReconciliationPublicationV2(group, allocation_attempt, tuple(proposed))
            _ensure_directory(self.root)
            _ensure_directory(os.path.join(self.root, "groups"))
            directory = self._reconciliation_directory()
            _ensure_directory(directory)
            with _task_lock(os.path.join(directory, ".publication.lock")), ExitStack() as fresh_locks:
                existing = self.read_reconciliation(group, allocation_attempt)
                if existing is not None:
                    if existing.encode() != candidate.encode():
                        return park("reconciliation-publication-conflict")
                    return existing.reconciliation
                # Mutable state gates apply only to first publication.
                for task in group.members:
                    self._validate_task_id(task.task_id)
                    item = by_task[task.task_id]
                    if item.start_receipt is None:
                        task_dir = os.path.join(self._group_dir(group), task.task_id)
                        if os.path.lexists(task_dir):
                            _require_directory(task_dir, "fresh task directory")
                            if set(os.listdir(task_dir)) - {_TASK_LOCK}:
                                raise FenceRejected("fresh-member-has-durable-state")
                        continue
                    binding = bindings[task.task_id]
                    if not isinstance(binding, MemberInvocationV2):
                        raise TypeError("leaf echo must be MemberInvocationV2")
                    if (binding.allocation_attempt.group_id != group.group_id
                            or binding.lineage_id != task.lineage_id
                            or binding.group_digest != group.identity):
                        raise FenceRejected("crossed-allocation-evidence")
                    start = item.start_receipt
                    if item.receipt_digest is not None and item.receipt_digest != digest({
                        "start_receipt": start.to_dict(), "outcome_receipt": item.outcome_receipt.to_dict(),
                    }):
                        raise FenceRejected("crossed-receipt-digest")
                    if (start.allocation_id != binding.allocation_attempt.allocation_id
                            or start.backend_job_id != binding.allocation_attempt.backend_job_id):
                        raise FenceRejected("crossed-allocation-evidence")
                    fence = AttemptFence(self.root, group.identity, task.task_id,
                                         binding.allocation_attempt.allocation_id.value,
                                         item.attempt_id, start.output_dir)
                    task_dir = self._task_dir_for_fence(fence)
                    with _task_lock(os.path.join(task_dir, _TASK_LOCK)):
                        durable = self._validate_fence(fence, task_dir)
                        self._validate_invocation(durable, binding)
                        outcome_path = os.path.join(start.output_dir, "outcome.json")
                        _require_regular(outcome_path, "outcome receipt")
                        if (Path(os.path.join(start.output_dir, "start.json")).read_bytes() != start.encode()
                                or Path(outcome_path).read_bytes() != item.outcome_receipt.encode()):
                            raise FenceRejected("crossed-durable-receipt")
                    fences[task.task_id] = fence
                # A concurrent first start must not race a fresh_pending
                # publication. Existing-member successor gates own their
                # separate task-lock recheck below.
                for member in candidate.members:
                    if member.evidence.start_receipt is None:
                        task_dir = self._prepare_task_dir(group, member.evidence.task_id)
                        fresh_locks.enter_context(_task_lock(os.path.join(task_dir, _TASK_LOCK)))
                        if set(os.listdir(task_dir)) - {_TASK_LOCK}:
                            raise FenceRejected("fresh-member-has-durable-state")
                for member in candidate.members:
                    if member.admission is None:
                        continue
                    request, acceptance, successor = successors[member.evidence.task_id]
                    actual = self.record_successor_admission(
                        fences[member.evidence.task_id], request, acceptance,
                        predecessor_binding=member.binding, successor_binding=successor,
                    )
                    if actual.encode() != member.admission.encode():
                        raise FenceRejected("crossed-successor-admission")
                path = os.path.join(directory, digest(allocation_attempt.allocation_attempt_id.to_dict()) + ".json")
                _write_new(path, candidate.encode())
                return candidate.reconciliation
        except AdmissionFork:
            return park("successor-fork")
        except (FenceRejected, OSError, TypeError, ValueError, KeyError):
            return park("invalid-durable-evidence")

    def finish_attempt(self, fence: AttemptFence, status: str, finished_at_utc: str, result_digest: str | None = None) -> OutcomeReceipt:
        """Write exactly one durable outcome after validating a complete fence.

        Parameters
        ----------
        fence : AttemptFence
            Store-owned attempt identity returned by :meth:`begin_attempt`.
        status : str
            Operational status from :data:`tpen_executor.results.OUTCOME_STATUSES`.
        finished_at_utc : str
            UTC completion timestamp text.
        result_digest : str or None, optional
            Optional digest of the produced result.

        Returns
        -------
        OutcomeReceipt
            The newly published terminal receipt.

        Raises
        ------
        FenceRejected
            If the fence is stale, crossed, symlinked, or already finished.
        ValueError, TypeError
            If receipt fields are malformed or ``status`` is outside the
            closed operational vocabulary.

        Notes
        -----
        POSIX advisory locking serializes pointer validation and outcome
        publication per task.  Publication links a complete fsynced temporary
        payload into place without exposing an empty final receipt.
        """
        task_dir = self._task_dir_for_fence(fence)
        lock_path = os.path.join(task_dir, _TASK_LOCK)
        _existing(lock_path, "task lock") if os.path.lexists(lock_path) else None
        with _task_lock(lock_path):
            start = self._validate_fence(fence, task_dir)
            output_dir = _lexical(fence.output_dir)
            outcome_path = os.path.join(output_dir, "outcome.json")
            if os.path.lexists(outcome_path):
                _require_regular(outcome_path, "outcome receipt")
                raise FenceRejected("attempt is already finished")
            try:
                outcome = OutcomeReceipt(fence.task_id, fence.attempt_id, fence.group_digest, finished_at_utc, status, start.output_dir, result_digest)
                _write_new(outcome_path, outcome.encode())
            except (OSError, ValueError, TypeError) as error:
                if isinstance(error, (ValueError, TypeError, FenceRejected)):
                    raise
                raise FenceRejected("outcome could not be written") from error
            return outcome

    def finish_remote_attempt(
        self,
        fence: AttemptFence,
        teardown: Mapping[str, Any],
        leaf_result_digest: str,
        status: str,
        finished_at_utc: str,
        result_digest: str | None = None,
    ) -> OutcomeReceipt:
        """Persist a validated remote teardown before its terminal outcome.

        Parameters
        ----------
        fence : AttemptFence
            Store-owned fence returned by :meth:`begin_attempt`.
        teardown : mapping
            Primitive leaf teardown facts that were already validated against
            the transport request by the controller.
        leaf_result_digest : str
            Digest of the canonical validated leaf result.
        status, finished_at_utc : str
            Operational terminal status and UTC completion timestamp.
        result_digest : str or None, optional
            Core evidence digest for the matching operational outcome.

        Returns
        -------
        OutcomeReceipt
            The terminal outcome written after ``remote-teardown.json``.

        Raises
        ------
        FenceRejected
            If the fence, teardown identities, or write-once ordering is not
            valid.  The method refuses before creating an outcome or ledger.

        Notes
        -----
        This is deliberately separate from :meth:`finish_attempt`: local and
        ThreadPool paths cannot manufacture a remote teardown proof.
        """
        if not isinstance(teardown, Mapping):
            raise TypeError("remote teardown must be a mapping")
        try:
            if set(teardown) != {
                "schema", "group_digest", "task_id", "attempt_id", "allocation_id",
                "backend_job_id", "child_group_reaped", "inspection",
            } or teardown.get("schema") != _LEAF_TEARDOWN_SCHEMA:
                raise ValueError("remote teardown has an invalid shape")
            if any(
                type(teardown[key]) is not str
                for key in ("group_digest", "task_id", "attempt_id", "allocation_id", "backend_job_id", "inspection")
            ) or type(teardown["child_group_reaped"]) is not bool:
                raise TypeError("remote teardown has an invalid field type")
            receipt = RemoteTeardownReceipt(
                teardown["task_id"],
                AttemptId(teardown["attempt_id"]),
                AllocationId(teardown["allocation_id"]),
                BackendJobId(teardown["backend_job_id"]),
                teardown["group_digest"],
                teardown["child_group_reaped"],
                teardown["inspection"],
                leaf_result_digest,
            )
        except (TypeError, ValueError) as error:
            raise FenceRejected(_TEARDOWN_UNCONFIRMED_REASON) from error
        if not receipt.child_group_reaped:
            raise FenceRejected(_TEARDOWN_UNCONFIRMED_REASON)
        task_dir = self._task_dir_for_fence(fence)
        lock_path = os.path.join(task_dir, _TASK_LOCK)
        _existing(lock_path, "task lock") if os.path.lexists(lock_path) else None
        with _task_lock(lock_path):
            start = self._validate_fence(fence, task_dir)
            output_dir = _lexical(fence.output_dir)
            receipt_path = os.path.join(output_dir, _REMOTE_TEARDOWN_FILENAME)
            outcome_path = os.path.join(output_dir, "outcome.json")
            if os.path.lexists(receipt_path):
                _require_regular(receipt_path, "remote teardown receipt")
                raise FenceRejected("remote teardown receipt already exists")
            if os.path.lexists(outcome_path):
                _require_regular(outcome_path, "outcome receipt")
                raise FenceRejected("attempt is already finished")
            if (
                receipt.task_id != start.task_id
                or receipt.attempt_id != start.attempt_id
                or receipt.group_digest != start.group_digest
                or receipt.allocation_id != start.allocation_id
                or receipt.backend_job_id != start.backend_job_id
            ):
                raise FenceRejected(_TEARDOWN_UNCONFIRMED_REASON)
            try:
                outcome = OutcomeReceipt(
                    fence.task_id,
                    fence.attempt_id,
                    fence.group_digest,
                    finished_at_utc,
                    status,
                    start.output_dir,
                    result_digest,
                )
                _write_new(receipt_path, receipt.encode())
                _write_new(outcome_path, outcome.encode())
            except (OSError, ValueError, TypeError) as error:
                if isinstance(error, (ValueError, TypeError, FenceRejected)):
                    raise
                raise FenceRejected("remote teardown outcome could not be written") from error
            return outcome

    def accept_segment(
        self,
        fence: AttemptFence,
        evidence: CoreRunEvidence,
        predecessor_checkpoint: str | None = None,
    ) -> AcceptedSegment:
        """Durably accept one checkpoint from a completed attempt.

        Parameters
        ----------
        fence : AttemptFence
            Store-owned fence for the attempt being accepted.
        evidence : CoreRunEvidence
            Public core evidence for the fenced run.  Only completed or
            incomplete evidence carrying a checkpoint may be accepted.
        predecessor_checkpoint : str or None, optional
            Checkpoint from which this segment continued.  P8 only publishes
            fresh-start segments, so callers normally leave this as ``None``.

        Returns
        -------
        AcceptedSegment
            The immutable receipt published beneath the attempt directory.

        Raises
        ------
        FenceRejected
            If the fence is foreign/stale, the attempt is not completed, the
            evidence lacks a checkpoint, the evidence run directory is not
            exactly the canonical ``<attempt>/core`` directory, or this
            attempt was already accepted.
        ValueError, TypeError
            If the predecessor or evidence shape is malformed.

        Notes
        -----
        The store-side attempt directory is resolved strictly and its
        ``core`` child must be an existing non-symlink directory.  The
        evidence run directory is also resolved strictly, then compared by
        exact canonical identity with that store-side core directory.
        """

        if not isinstance(evidence, CoreRunEvidence):
            raise TypeError("evidence must be CoreRunEvidence")
        if evidence.schema == "tpen.run-evidence/v3" and (
            evidence.outcome != "completed" or evidence.verdict != "completed"
        ):
            raise FenceRejected("v3 non-completed evidence cannot be accepted as a completed segment")
        if evidence.verdict not in {"completed", "incomplete"} or evidence.checkpoint is None:
            raise FenceRejected("accepted segment requires core evidence with a checkpoint")
        task_dir = self._task_dir_for_fence(fence)
        lock_path = os.path.join(task_dir, _TASK_LOCK)
        _existing(lock_path, "task lock") if os.path.lexists(lock_path) else None
        with _task_lock(lock_path):
            start = self._validate_fence(fence, task_dir)
            output_dir = _lexical(fence.output_dir)
            attempt_dir = _strict_real_directory(output_dir)
            core_path = os.path.join(attempt_dir, "core")
            try:
                _require_directory(core_path, "core evidence directory")
            except FenceRejected as error:
                raise FenceRejected(_CORE_EVIDENCE_FENCE_REASON) from error
            expected_core_dir = _strict_real_directory(core_path)
            evidence_core_dir = _strict_real_directory(evidence.run_dir)
            if evidence_core_dir != expected_core_dir:
                raise FenceRejected(_CORE_EVIDENCE_FENCE_REASON)
            # Integrity failures must win over the newer identity requirement:
            # never let malformed or foreign evidence bypass the attempt fence.
            if evidence.resolved_config_identity is None:
                raise FenceRejected("new accepted segment requires resolved-config identity")
            outcome_path = os.path.join(output_dir, "outcome.json")
            _require_regular(outcome_path, "outcome receipt")
            try:
                outcome = OutcomeReceipt.decode(Path(outcome_path).read_bytes())
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                raise FenceRejected("outcome receipt is unavailable") from error
            if (
                outcome.task_id != start.task_id
                or outcome.attempt_id != start.attempt_id
                or outcome.group_digest != start.group_digest
                or outcome.output_dir != start.output_dir
            ):
                raise FenceRejected("outcome receipt crosses the attempt fence")
            if outcome.status != "completed":
                raise FenceRejected("accepted segment requires a completed attempt")
            segment_path = os.path.join(output_dir, "segment.json")
            if os.path.lexists(segment_path):
                _require_regular(segment_path, "accepted segment")
                raise FenceRejected("attempt already has an accepted segment")
            checkpoint = evidence.checkpoint
            segment = AcceptedSegment(
                fence.task_id,
                fence.attempt_id,
                predecessor_checkpoint,
                str(checkpoint["checkpoint_dir"]),
                checkpoint["next_iteration"],
                checkpoint["completed_updates"],
                evidence.resolved_config_identity,
            )
            prior_segments = self._segments_for_task(task_dir)
            if prior_segments:
                previous = prior_segments[-1]
                if segment.predecessor_checkpoint != previous.checkpoint_dir:
                    raise FenceRejected("accepted segment predecessor disagrees with prior checkpoint")
                previous_coordinate = (previous.next_iteration, previous.completed_updates)
                coordinate = (segment.next_iteration, segment.completed_updates)
                if coordinate[0] < previous_coordinate[0] or coordinate[1] < previous_coordinate[1] or coordinate == previous_coordinate:
                    raise FenceRejected("accepted segment progress coordinate is not strictly increasing")
            elif segment.predecessor_checkpoint is not None:
                raise FenceRejected("first accepted segment must have no predecessor")
            _write_new(segment_path, segment.encode())
            return segment

    def record_core_outcome(self, fence: AttemptFence, evidence: CoreRunEvidence) -> CoreOutcomeRecord:
        """Persist typed v3 evidence without turning it into a segment.

        A planned yield is deliberately recorded in ``core-outcome.json`` and
        remains operationally non-completed.  Successor admission is outside
        this method and belongs to the continuation slice.
        """
        if not isinstance(evidence, CoreRunEvidence):
            raise TypeError("evidence must be CoreRunEvidence")
        if evidence.schema != "tpen.run-evidence/v3":
            raise FenceRejected("typed core outcome requires v3 evidence")
        task_dir = self._task_dir_for_fence(fence)
        lock_path = os.path.join(task_dir, _TASK_LOCK)
        _existing(lock_path, "task lock") if os.path.lexists(lock_path) else None
        with _task_lock(lock_path):
            start = self._validate_fence(fence, task_dir)
            output_dir = _lexical(fence.output_dir)
            attempt_dir = _strict_real_directory(output_dir)
            core_path = os.path.join(attempt_dir, "core")
            try:
                _require_directory(core_path, "core evidence directory")
            except FenceRejected as error:
                raise FenceRejected(_CORE_EVIDENCE_FENCE_REASON) from error
            expected_core_dir = _strict_real_directory(core_path)
            evidence_core_dir = _strict_real_directory(evidence.run_dir)
            if evidence_core_dir != expected_core_dir:
                raise FenceRejected(_CORE_EVIDENCE_FENCE_REASON)
            outcome_path = os.path.join(output_dir, "outcome.json")
            _require_regular(outcome_path, "outcome receipt")
            try:
                outcome = OutcomeReceipt.decode(Path(outcome_path).read_bytes())
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                raise FenceRejected("outcome receipt is unavailable") from error
            if (
                outcome.task_id != start.task_id
                or outcome.attempt_id != start.attempt_id
                or outcome.group_digest != start.group_digest
                or outcome.output_dir != start.output_dir
            ):
                raise FenceRejected("outcome receipt crosses the attempt fence")
            if evidence.outcome == "planned_yield" and outcome.status == "completed":
                raise FenceRejected("planned_yield cannot be recorded as a completed attempt")
            record = CoreOutcomeRecord.from_evidence(fence.task_id, fence.attempt_id, evidence)
            record_path = os.path.join(output_dir, "core-outcome.json")
            if os.path.lexists(record_path):
                _require_regular(record_path, "core outcome record")
                raise FenceRejected("attempt already has a core outcome record")
            _write_new(record_path, record.encode())
            return record

    @staticmethod
    def _successor_request_shape(request: Any) -> dict[str, Any]:
        """Validate the executor-facing shape without importing core."""

        if not isinstance(request, dict):
            if not isinstance(request, Mapping):
                raise TypeError("successor admission request must be a mapping")
            request = dict(request)
        expected = {
            "schema", "predecessor", "successor", "science_input_digest",
            "target_iterations", "core_evidence",
        }
        if set(request) != expected or request.get("schema") != _SUCCESSOR_ADMISSION_SCHEMA:
            raise ValueError("successor admission request has an invalid closed shape")
        predecessor = request["predecessor"]
        successor = request["successor"]
        if not isinstance(predecessor, Mapping) or set(predecessor) != {
            "task_id", "attempt_id", "core_outcome_digest", "resolved_config_identity",
        }:
            raise ValueError("successor admission predecessor has an invalid closed shape")
        if not isinstance(successor, Mapping) or set(successor) != {"task_id", "attempt_id"}:
            raise ValueError("successor admission successor has an invalid closed shape")
        for value, name in (
            (predecessor["task_id"], "predecessor.task_id"),
            (predecessor["attempt_id"], "predecessor.attempt_id"),
            (predecessor["core_outcome_digest"], "predecessor.core_outcome_digest"),
            (successor["task_id"], "successor.task_id"),
            (successor["attempt_id"], "successor.attempt_id"),
            (request["science_input_digest"], "science_input_digest"),
        ):
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be non-empty text")
        sha256_text(predecessor["core_outcome_digest"], "predecessor.core_outcome_digest")
        sha256_text(request["science_input_digest"], "science_input_digest")
        if type(request["target_iterations"]) is not int or request["target_iterations"] < 0:
            raise TypeError("target_iterations must be a nonnegative integer")
        if not isinstance(predecessor["resolved_config_identity"], Mapping) or set(predecessor["resolved_config_identity"]) != {"schema", "sha256"}:
            raise ValueError("predecessor.resolved_config_identity has an invalid shape")
        if predecessor["resolved_config_identity"].get("schema") != "tpen-config-identity-v1":
            raise ValueError("predecessor.resolved_config_identity has an invalid schema")
        sha256_text(predecessor["resolved_config_identity"].get("sha256"), "predecessor.resolved_config_identity.sha256")
        if not isinstance(request["core_evidence"], Mapping):
            raise TypeError("core_evidence must be a mapping")
        return request

    def _continuation_predecessor_locked(
        self, fence: AttemptFence, task_dir: str,
    ) -> tuple[StartReceipt, CoreOutcomeRecord]:
        """Read the complete continuation predicate while the task is locked."""
        start = self._validate_fence(fence, task_dir)
        output_dir = _lexical(fence.output_dir)
        core_path = _lexical(os.path.join(output_dir, "core"))
        _require_directory(core_path, "core evidence directory")
        outcome_path = os.path.join(output_dir, "outcome.json")
        _require_regular(outcome_path, "outcome receipt")
        try:
            outcome = OutcomeReceipt.decode(Path(outcome_path).read_bytes())
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise FenceRejected("outcome receipt is unavailable") from error
        if (
            outcome.task_id != start.task_id
            or outcome.attempt_id != start.attempt_id
            or outcome.group_digest != start.group_digest
            or outcome.output_dir != start.output_dir
        ):
            raise FenceRejected("outcome receipt crosses the attempt fence")
        if outcome.status != "failed":
            raise FenceRejected(_TEARDOWN_UNCONFIRMED_REASON)

        # The positive remote proof is a separate canonical, write-once fact;
        # the operational ``failed`` status alone is never sufficient for a
        # remote (or legacy/unknown) attempt.  Local attempts retain the
        # CX-04b predicate and do not manufacture a remote teardown receipt.
        # Migration note: legacy v1 local attempts decode with an unknown
        # transport kind and therefore require positive remote proof here;
        # they are not admissible for continuation after upgrade (fail closed).
        if start.transport_kind != "local":
            teardown_path = os.path.join(output_dir, _REMOTE_TEARDOWN_FILENAME)
            try:
                _require_regular(teardown_path, "remote teardown receipt")
                encoded_teardown = Path(teardown_path).read_bytes()
                teardown = RemoteTeardownReceipt.decode(encoded_teardown)
            except (OSError, TypeError, ValueError, FenceRejected, json.JSONDecodeError) as error:
                raise FenceRejected(_TEARDOWN_UNCONFIRMED_REASON) from error
            if encoded_teardown != teardown.encode():
                raise FenceRejected(_TEARDOWN_UNCONFIRMED_REASON)
            if (
                teardown.task_id != start.task_id
                or teardown.attempt_id != start.attempt_id
                or teardown.group_digest != start.group_digest
                or teardown.allocation_id != start.allocation_id
                or teardown.backend_job_id != start.backend_job_id
                or not teardown.child_group_reaped
                or teardown.inspection != "complete"
            ):
                raise FenceRejected(_TEARDOWN_UNCONFIRMED_REASON)

        record_path = os.path.join(output_dir, "core-outcome.json")
        _require_regular(record_path, "core outcome record")
        try:
            encoded = Path(record_path).read_bytes()
            record = CoreOutcomeRecord.decode(encoded)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise FenceRejected("core outcome record is unavailable") from error
        if encoded != record.encode():
            raise FenceRejected("core outcome record is not canonical")
        try:
            record_core_dir = _strict_real_directory(record.evidence.get("run_dir"))
            expected_core_dir = _strict_real_directory(core_path)
        except (OSError, TypeError, ValueError) as error:
            raise FenceRejected(_CORE_EVIDENCE_FENCE_REASON) from error
        if (
            record.task_id != start.task_id
            or record.attempt_id != start.attempt_id
            or record_core_dir != expected_core_dir
        ):
            raise FenceRejected("core outcome record crosses the attempt fence")
        if not record.continuation_eligible or record.outcome != "planned_yield":
            raise FenceRejected("core outcome is not a continuation-eligible planned yield")
        checkpoint_ref = record.evidence.get("checkpoint_ref")
        config_identity = record.evidence.get("resolved_config_identity")
        if not isinstance(checkpoint_ref, Mapping) or type(checkpoint_ref.get("content_id")) is not str:
            raise FenceRejected("planned-yield core outcome lacks a checkpoint content identity")
        if not isinstance(config_identity, Mapping):
            raise FenceRejected("core outcome lacks resolved config identity")
        return start, record

    @staticmethod
    def _successor_request_from_record(
        record: CoreOutcomeRecord,
        successor_attempt_id: AttemptId,
        science_input_digest: str,
        target_iterations: int,
    ) -> dict[str, Any]:
        """Build a successor request from an already validated core record."""
        config_identity = record.evidence["resolved_config_identity"]
        return {
            "schema": _SUCCESSOR_ADMISSION_SCHEMA,
            "predecessor": {
                "task_id": record.task_id,
                "attempt_id": record.attempt_id.value,
                "core_outcome_digest": digest(record.to_dict()),
                "resolved_config_identity": thaw(config_identity),
            },
            "successor": {"task_id": record.task_id, "attempt_id": successor_attempt_id.value},
            "science_input_digest": science_input_digest,
            "target_iterations": target_iterations,
            "core_evidence": thaw(record.evidence),
        }

    def build_successor_admission_request(
        self,
        fence: AttemptFence,
        successor_attempt_id: AttemptId,
        science_input_digest: str,
        target_iterations: int,
    ) -> dict[str, Any]:
        """Build core's closed admission mapping from a fenced outcome.

        This method only produces a JSON-shaped mapping.  The caller must pass
        it to ``tpen.execution.recovery.select_successor_admission`` outside
        this package and report the resulting acceptance before the ledger can
        be written.
        """

        if not isinstance(successor_attempt_id, AttemptId):
            raise TypeError("successor_attempt_id must be AttemptId")
        if isinstance(fence, AttemptFence) and successor_attempt_id == fence.attempt_id:
            raise ValueError("successor_attempt_id must differ from predecessor attempt")
        sha256_text(science_input_digest, "science_input_digest")
        if type(target_iterations) is not int or target_iterations < 0:
            raise TypeError("target_iterations must be a nonnegative integer")
        task_dir = self._task_dir_for_fence(fence)
        lock_path = os.path.join(task_dir, _TASK_LOCK)
        _existing(lock_path, "task lock") if os.path.lexists(lock_path) else None
        with _task_lock(lock_path):
            _start, record = self._continuation_predecessor_locked(fence, task_dir)
            return self._successor_request_from_record(
                record, successor_attempt_id, science_input_digest, target_iterations,
            )

    def record_successor_admission(
        self,
        fence: AttemptFence,
        request: Mapping[str, Any],
        core_acceptance_digest: str,
        *,
        predecessor_binding: MemberInvocationV2 | None = None,
        successor_binding: MemberInvocationV2 | None = None,
    ) -> SuccessorAdmissionRecord:
        """Record a core-accepted successor request exactly once.

        ``core_acceptance_digest`` is the caller-supplied SHA-256 digest of
        core's accepted ``OperationalRestoreRequest``.  It is deliberately
        opaque here: the dispatcher or integration owner is responsible for
        obtaining and authenticating that acceptance, while the executor
        cannot verify it without importing ``tpen``.  The digest is persisted
        and participates in replay/fork comparison alongside ``request_digest``.
        """

        if type(core_acceptance_digest) is not str:
            raise TypeError("core_acceptance_digest must be a SHA-256 digest")
        sha256_text(core_acceptance_digest, "core_acceptance_digest")
        request = self._successor_request_shape(request)
        request_digest = digest(request)
        built = self.build_successor_admission_request(
            fence,
            AttemptId(request["successor"]["attempt_id"]),
            request["science_input_digest"],
            request["target_iterations"],
        )
        task_dir = self._task_dir_for_fence(fence)
        lock_path = os.path.join(task_dir, _TASK_LOCK)
        _existing(lock_path, "task lock") if os.path.lexists(lock_path) else None
        with _task_lock(lock_path):
            start, record = self._continuation_predecessor_locked(fence, task_dir)
            rebuilt = self._successor_request_from_record(
                record,
                AttemptId(request["successor"]["attempt_id"]),
                request["science_input_digest"],
                request["target_iterations"],
            )
            if digest(rebuilt) != request_digest or rebuilt != request or digest(built) != request_digest:
                raise FenceRejected("successor admission request does not match fenced core outcome")
            if predecessor_binding is not None or successor_binding is not None:
                if not isinstance(predecessor_binding, MemberInvocationV2) or not isinstance(successor_binding, MemberInvocationV2):
                    raise FenceRejected("both invocation bindings are required")
                self._validate_invocation(start, predecessor_binding)
                if (successor_binding.task_id != start.task_id
                        or successor_binding.attempt_id.value != request["successor"]["attempt_id"]
                        or successor_binding.lineage_id != predecessor_binding.lineage_id
                        or successor_binding.group_digest != predecessor_binding.group_digest
                        or successor_binding.allocation_attempt.group_id != predecessor_binding.allocation_attempt.group_id
                        or successor_binding.allocation_attempt.allocation_attempt_id == predecessor_binding.allocation_attempt.allocation_attempt_id
                        or successor_binding.invocation_id == predecessor_binding.invocation_id):
                    raise FenceRejected("successor binding is crossed or not fresh")
            elif os.path.lexists(os.path.join(start.output_dir, "invocation.json")):
                raise FenceRejected("v2 predecessor requires successor invocation binding")
            segment_path = os.path.join(_lexical(fence.output_dir), "segment.json")
            if os.path.lexists(segment_path):
                raise FenceRejected("planned-yield admission cannot coexist with an accepted segment")
            successor = request["successor"]
            record = SuccessorAdmissionRecord(
                start.task_id,
                start.attempt_id,
                request["core_evidence"]["checkpoint_ref"]["content_id"],
                successor["task_id"],
                AttemptId(successor["attempt_id"]),
                request_digest,
                core_acceptance_digest,
                successor_binding,
            )
            ledger_dir = _lexical(os.path.join(task_dir, _SUCCESSOR_ADMISSIONS))
            _ensure_directory(ledger_dir)
            existing_records = self._admission_records_for_task(task_dir, start.task_id)
            for existing in existing_records:
                if existing.predecessor_attempt_id == start.attempt_id:
                    if existing.to_dict() == record.to_dict():
                        return existing
                    raise AdmissionFork("fork: a different successor is already admitted for this predecessor")
            predecessor_dir = _lexical(os.path.join(ledger_dir, start.attempt_id.value))
            _ensure_directory(predecessor_dir)
            try:
                _write_new(
                    os.path.join(predecessor_dir, _SUCCESSOR_ADMISSION_FILENAME),
                    record.encode(),
                    temporary_dir=ledger_dir,
                )
            except FileExistsError as error:
                raise AdmissionFork("fork: successor admission ledger already exists") from error
            return record

    @staticmethod
    def _admission_records_for_task(task_dir: str, task_id: str) -> tuple[SuccessorAdmissionRecord, ...]:
        """Read every admission record and reject duplicate predecessors."""

        root = _lexical(os.path.join(task_dir, _SUCCESSOR_ADMISSIONS))
        if not os.path.lexists(root):
            return ()
        _require_directory(root, "successor admissions", ValueError)
        records: list[SuccessorAdmissionRecord] = []
        seen: set[AttemptId] = set()
        for predecessor_attempt_id in sorted(os.listdir(root)):
            if predecessor_attempt_id.startswith(_SUCCESSOR_ADMISSION_TEMP_PREFIX):
                continue
            predecessor_dir = _lexical(os.path.join(root, predecessor_attempt_id))
            _require_directory(predecessor_dir, "successor admission predecessor directory", ValueError)
            if predecessor_attempt_id in {".", ".."}:
                raise ValueError("successor admission predecessor directory is malformed")
            entries = os.listdir(predecessor_dir)
            if entries != [_SUCCESSOR_ADMISSION_FILENAME]:
                if sorted(entries) != [_SUCCESSOR_ADMISSION_FILENAME]:
                    raise ValueError("successor admission predecessor directory has unexpected entries")
            path = os.path.join(predecessor_dir, _SUCCESSOR_ADMISSION_FILENAME)
            _require_regular(path, "successor admission record", ValueError)
            encoded = Path(path).read_bytes()
            record = SuccessorAdmissionRecord.decode(encoded)
            if encoded != record.encode():
                raise ValueError("successor admission record is not canonical")
            if record.predecessor_task_id != task_id or record.predecessor_attempt_id.value != predecessor_attempt_id:
                raise ValueError("foreign successor admission record in task store")
            if record.predecessor_attempt_id in seen:
                raise ValueError("successor admission ledger contains a predecessor fork")
            seen.add(record.predecessor_attempt_id)
            records.append(record)
        return tuple(records)

    def publish_placement(
        self,
        fence: AttemptFence,
        receipt: SciencePlacementReceipt | SciencePlacementReceiptV2 | SciencePlacementReceiptV3,
    ) -> SciencePlacementReceipt | SciencePlacementReceiptV2 | SciencePlacementReceiptV3:
        """Publish one validated placement receipt after child quiescence.

        The receipt is written beneath the same attempt fence as the start and
        outcome records.  Requiring the terminal outcome first prevents a
        controller from presenting an in-flight child as placement evidence.

        Parameters
        ----------
        fence : AttemptFence
            Store-owned attempt fence returned by :meth:`begin_attempt`.
        receipt : SciencePlacementReceipt or SciencePlacementReceiptV2 or SciencePlacementReceiptV3
            Controller-validated child-generation record.

        Returns
        -------
        SciencePlacementReceipt or SciencePlacementReceiptV2 or SciencePlacementReceiptV3
            The immutable receipt that was durably published.

        Raises
        ------
        FenceRejected
            If the attempt is foreign, unfinished, already has a placement
            receipt, or the receipt crosses any fence identity.
        """

        if not isinstance(receipt, PLACEMENT_RECEIPT_TYPES):
            raise TypeError("receipt must be a v1, v2, or v3 science placement receipt")
        task_dir = self._task_dir_for_fence(fence)
        lock_path = os.path.join(task_dir, _TASK_LOCK)
        _existing(lock_path, "task lock") if os.path.lexists(lock_path) else None
        with _task_lock(lock_path):
            start = self._validate_fence(fence, task_dir)
            output_dir = _lexical(fence.output_dir)
            outcome_path = os.path.join(output_dir, "outcome.json")
            _require_regular(outcome_path, "outcome receipt")
            try:
                outcome = OutcomeReceipt.decode(Path(outcome_path).read_bytes())
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise FenceRejected("outcome receipt is unavailable") from error
            if (
                outcome.task_id != start.task_id
                or outcome.attempt_id != start.attempt_id
                or outcome.group_digest != start.group_digest
                or outcome.output_dir != output_dir
            ):
                raise FenceRejected("outcome receipt crosses the attempt fence")
            if (
                receipt.task_id != start.task_id
                or receipt.attempt_id != start.attempt_id
                or receipt.group_digest != start.group_digest
                or receipt.allocation_id != start.allocation_id.value
                or receipt.backend_job_id != start.backend_job_id.value
                or receipt.output_dir != output_dir
            ):
                raise FenceRejected("placement receipt crosses the attempt fence")
            placement_path = os.path.join(output_dir, "placement.json")
            if os.path.lexists(placement_path):
                _require_regular(placement_path, "placement receipt")
                raise FenceRejected("attempt already has a placement receipt")
            _write_new(placement_path, receipt.encode())
            return receipt

    @staticmethod
    def _segments_for_task(task_dir: str) -> tuple[AcceptedSegment, ...]:
        """Read accepted segments in predecessor-linked chain order.

        Raises
        ------
        FenceRejected
            If durable segment receipts form a fork, cycle, or disconnected
            chain instead of one linear history.
        """

        segments: list[AcceptedSegment] = []
        for attempt_id in sorted(os.listdir(task_dir)):
            if attempt_id in {_TASK_LOCK, "current-attempt.json"}:
                continue
            output_dir = _lexical(os.path.join(task_dir, attempt_id))
            if not os.path.isdir(output_dir):
                continue
            segment_path = os.path.join(output_dir, "segment.json")
            if not os.path.lexists(segment_path):
                continue
            _require_regular(segment_path, "accepted segment", ValueError)
            segments.append(AcceptedSegment.decode(Path(segment_path).read_bytes()))
        if not segments:
            return ()
        successors: dict[str, list[AcceptedSegment]] = {}
        roots = []
        for segment in segments:
            if segment.predecessor_checkpoint is None:
                roots.append(segment)
            else:
                successors.setdefault(segment.predecessor_checkpoint, []).append(segment)
        if len(roots) != 1:
            raise FenceRejected("accepted segments do not have one chain root")
        ordered: list[AcceptedSegment] = []
        current = roots[0]
        seen_attempts: set[object] = set()
        while True:
            if current.attempt_id in seen_attempts:
                raise FenceRejected("accepted segments contain a cycle")
            seen_attempts.add(current.attempt_id)
            ordered.append(current)
            next_segments = successors.get(current.checkpoint_dir, [])
            if len(next_segments) > 1:
                raise FenceRejected("accepted segments contain a fork")
            if not next_segments:
                break
            current = next_segments[0]
        if len(ordered) != len(segments):
            raise FenceRejected("accepted segments contain a disconnected chain")
        for previous, segment in zip(ordered, ordered[1:]):
            previous_coordinate = (previous.next_iteration, previous.completed_updates)
            coordinate = (segment.next_iteration, segment.completed_updates)
            if coordinate[0] < previous_coordinate[0] or coordinate[1] < previous_coordinate[1] or coordinate == previous_coordinate:
                raise FenceRejected("accepted segment progress coordinate is not strictly increasing")
        return tuple(ordered)

    def _validate_fence_immutable(self, fence: AttemptFence, task_dir: str) -> StartReceipt:
        """Validate the immutable attempt path and start receipt binding."""
        expected = _lexical(os.path.join(task_dir, fence.attempt_id.value))
        if expected != _lexical(fence.output_dir):
            raise FenceRejected("crossed attempt fence")
        _require_directory(expected, "attempt directory")
        start_path = os.path.join(expected, "start.json")
        _require_regular(start_path, "start receipt")
        try:
            start = StartReceipt.decode(Path(start_path).read_bytes())
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise FenceRejected("start receipt is unavailable") from error
        if (
            start.task_id != fence.task_id
            or start.attempt_id != fence.attempt_id
            or start.group_digest != fence.group_digest
            or start.allocation_id.value != fence.allocation_id
            or start.output_dir != expected
        ):
            raise FenceRejected("crossed attempt fence")
        return start

    def _validate_fence(self, fence: AttemptFence, task_dir: str) -> StartReceipt:
        """Validate current pointer plus immutable attempt fence binding."""
        pointer = os.path.join(task_dir, "current-attempt.json")
        _require_regular(pointer, "current attempt pointer")
        try:
            current = decode_json(Path(pointer).read_bytes())
            fields(current, {"schema", "attempt_id"}, "tpen-executor/current-attempt-v1", "current attempt")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise FenceRejected("current attempt pointer is unavailable") from error
        if current["attempt_id"] != fence.attempt_id.value:
            raise FenceRejected("stale attempt fence")
        return self._validate_fence_immutable(fence, task_dir)

    def manifest(self, group: ReadyGroup) -> RunManifest:
        """Rebuild a strict manifest only from durable receipts on disk.

        Parameters
        ----------
        group : ReadyGroup
            Group whose expected digest and member task namespace are read.

        Returns
        -------
        RunManifest
            Receipt-derived manifest.  A fresh store instance reconstructs the
            same value; the current pointer is used only by finishing and is
            not treated as a receipt.

        Raises
        ------
        TypeError
            If ``group`` is not a :class:`ReadyGroup`.
        ValueError, OSError
            If any store component, receipt, or directory entry is malformed,
            foreign, truncated, or a symlink.

        Notes
        -----
        The store requires POSIX directories and advisory locking for its
        writer protocol.  Symlink swapping during a call is outside this
        contract; pre-existing symlinks are always refused.
        """
        if not isinstance(group, ReadyGroup):
            raise TypeError("group must be ReadyGroup")
        group_dir = self._group_dir(group)
        attempts: list[ManifestAttempt] = []
        segments: list[AcceptedSegment] = []
        admissions: dict[tuple[str, AttemptId], SuccessorAdmissionRecord] = {}
        members = {member.task_id for member in group.members}
        if not os.path.lexists(self.root):
            return RunManifest(group.group_id, group.identity, ())
        _require_directory(self.root, "store root", ValueError)
        groups_dir = _lexical(os.path.join(self.root, "groups"))
        if not os.path.lexists(groups_dir):
            return RunManifest(group.group_id, group.identity, ())
        _require_directory(groups_dir, "groups directory", ValueError)
        if not os.path.lexists(group_dir):
            return RunManifest(group.group_id, group.identity, ())
        _require_directory(group_dir, "group directory", ValueError)
        for task_id in sorted(os.listdir(group_dir)):
            task_dir = _lexical(os.path.join(group_dir, task_id))
            if task_id not in members or _SAFE_TASK_ID.fullmatch(task_id) is None:
                raise ValueError("foreign or malformed task directory in group store")
            _require_directory(task_dir, "task directory", ValueError)
            for attempt_id in sorted(os.listdir(task_dir)):
                if attempt_id == "current-attempt.json":
                    _require_regular(os.path.join(task_dir, attempt_id), "current attempt pointer", ValueError)
                    continue
                if attempt_id == _TASK_LOCK:
                    _require_regular(os.path.join(task_dir, attempt_id), "task lock", ValueError)
                    continue
                if attempt_id == _SUCCESSOR_ADMISSIONS:
                    continue
                output_dir = _lexical(os.path.join(task_dir, attempt_id))
                _require_directory(output_dir, "attempt directory", ValueError)
                start_path = os.path.join(output_dir, "start.json")
                _require_regular(start_path, "start receipt", ValueError)
                start = StartReceipt.decode(Path(start_path).read_bytes())
                if (
                    start.group_digest != group.identity
                    or start.task_id != task_id
                    or start.attempt_id.value != attempt_id
                    or start.output_dir != output_dir
                ):
                    raise ValueError("foreign receipt in group store")
                outcome_file = os.path.join(output_dir, "outcome.json")
                if os.path.lexists(outcome_file):
                    _require_regular(outcome_file, "outcome receipt", ValueError)
                    outcome = OutcomeReceipt.decode(Path(outcome_file).read_bytes())
                    if (
                        outcome.task_id != task_id
                        or outcome.attempt_id.value != attempt_id
                        or outcome.group_digest != group.identity
                        or outcome.output_dir != output_dir
                    ):
                        raise ValueError("foreign outcome receipt in group store")
                else:
                    outcome = None
                placement_path = os.path.join(output_dir, "placement.json")
                if os.path.lexists(placement_path):
                    _require_regular(placement_path, "placement receipt", ValueError)
                    raw_placement = decode_json(Path(placement_path).read_bytes())
                    placement = decode_placement_receipt(raw_placement)
                    if (
                        placement.task_id != task_id
                        or placement.attempt_id.value != attempt_id
                        or placement.group_digest != group.identity
                        or placement.output_dir != output_dir
                        or placement.allocation_id != start.allocation_id.value
                        or placement.backend_job_id != start.backend_job_id.value
                    ):
                        raise ValueError("foreign placement receipt in group store")
                else:
                    placement = None
                core_outcome_path = os.path.join(output_dir, "core-outcome.json")
                if os.path.lexists(core_outcome_path):
                    _require_regular(core_outcome_path, "core outcome record", ValueError)
                    core_outcome = CoreOutcomeRecord.decode(Path(core_outcome_path).read_bytes())
                    if core_outcome.task_id != task_id or core_outcome.attempt_id.value != attempt_id:
                        raise ValueError("foreign core outcome record in group store")
                else:
                    core_outcome = None
                segment_path = os.path.join(output_dir, "segment.json")
                if os.path.lexists(segment_path):
                    _require_regular(segment_path, "accepted segment", ValueError)
                    segment = AcceptedSegment.decode(Path(segment_path).read_bytes())
                    if segment.task_id != task_id or segment.attempt_id.value != attempt_id:
                        raise ValueError("foreign accepted segment in group store")
                    segments.append(segment)
                attempts.append(ManifestAttempt(task_id, start.attempt_id, start, outcome, start.output_dir, placement, core_outcome))
            for admission in self._admission_records_for_task(task_dir, task_id):
                key = (task_id, admission.predecessor_attempt_id)
                if key in admissions:
                    raise ValueError("successor admission ledger contains a predecessor fork")
                matching = [attempt for attempt in attempts if (attempt.task_id, attempt.attempt_id) == key]
                if len(matching) != 1:
                    raise ValueError("successor admission references an unknown predecessor attempt")
                predecessor = matching[0]
                if predecessor.core_outcome is None or predecessor.core_outcome.outcome != "planned_yield":
                    raise ValueError("successor admission requires a planned-yield core outcome")
                checkpoint_ref = predecessor.core_outcome.evidence.get("checkpoint_ref")
                if not isinstance(checkpoint_ref, Mapping) or admission.predecessor_content_id != checkpoint_ref.get("content_id"):
                    raise ValueError("successor admission content identity disagrees with core outcome")
                if any(segment.attempt_id == admission.predecessor_attempt_id for segment in segments):
                    raise ValueError("successor admission cannot coexist with an accepted segment")
                admissions[key] = admission
        projected_attempts = [
            ManifestAttempt(
                attempt.task_id,
                attempt.attempt_id,
                attempt.start_receipt,
                attempt.outcome_receipt,
                attempt.output_dir,
                attempt.placement_receipt,
                attempt.core_outcome,
                admissions.get((attempt.task_id, attempt.attempt_id)),
            )
            for attempt in attempts
        ]
        segments.sort(key=lambda item: (item.task_id, item.next_iteration, item.completed_updates))
        return RunManifest(group.group_id, group.identity, tuple(projected_attempts), tuple(segments))


def _write_pointer(path: str, attempt_id: str) -> None:
    """Atomically advance a task pointer; pointer replacement is intentional."""
    payload = canonical_bytes({"schema": "tpen-executor/current-attempt-v1", "attempt_id": attempt_id})
    parent = os.path.dirname(path)
    _require_directory(parent, "pointer parent")
    if os.path.lexists(path):
        _existing(path, "current attempt pointer")
    fd, temporary = tempfile.mkstemp(prefix=".tmp-pointer-", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = ""
        _fsync_directory(parent)
    finally:
        if temporary:
            try: os.unlink(temporary)
            except FileNotFoundError: pass


__all__ = ["AdmissionFork", "AttemptFence", "ControllerGenerationRejected", "DirectoryStore", "FenceRejected", "LaunchIntentRejected"]
