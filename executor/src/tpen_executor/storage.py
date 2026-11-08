"""Durable, fenced attempt receipts for an executor allocation."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
from uuid import uuid4

from ._codec import canonical_bytes, decode_json, fields, text
from .allocations import ConfiguredAllocation
from .contracts import AttemptId, ReadyGroup
from .results import AcceptedSegment, ManifestAttempt, OutcomeReceipt, RunManifest, SciencePlacementReceipt, StartReceipt
from .runners import CoreRunEvidence
from ._records import frozen_record


class FenceRejected(RuntimeError):
    """Raised when an attempt fence is stale, crossed, or already finished."""


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


def _write_new(path: str, payload: bytes) -> None:
    """Write a new file durably, refusing replacement of an existing file."""
    parent = os.path.dirname(path)
    _require_directory(parent, "receipt parent")
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=parent)
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

    def _group_dir(self, group: ReadyGroup) -> str:
        path = _lexical(os.path.join(self.root, "groups", group.identity))
        if not _inside(self.root, path):
            raise FenceRejected("group path escapes store root")
        return path

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        """Validate the reversible task directory namespace."""
        text(task_id, "task_id")
        if task_id in {".", ".."} or _SAFE_TASK_ID.fullmatch(task_id) is None:
            raise ValueError("task_id must match [A-Za-z0-9._:-]+ and must not be a traversal component")

    def _prepare_task_dir(self, group: ReadyGroup, task_id: str) -> str:
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

    def begin_attempt(self, group: ReadyGroup, task_id: str, allocation: ConfiguredAllocation, started_at_utc: str = "") -> AttemptFence:
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
        if not isinstance(group, ReadyGroup) or not isinstance(allocation, ConfiguredAllocation):
            raise TypeError("group and allocation have invalid types")
        text(task_id, "task_id")
        self._validate_task_id(task_id)
        if task_id not in {member.task_id for member in group.members}:
            raise ValueError("task_id is not a member of group")
        if type(started_at_utc) is not str:
            raise TypeError("started_at_utc must be a string")
        stamp = started_at_utc or "unknown"
        task_dir = self._prepare_task_dir(group, task_id)
        lock_path = os.path.join(task_dir, _TASK_LOCK)
        _existing(lock_path, "task lock") if os.path.lexists(lock_path) else None
        with _task_lock(lock_path):
            attempt = AttemptId(uuid4().hex)
            output_dir = _lexical(os.path.join(task_dir, attempt.value))
            _mkdir_new(output_dir)
            receipt = StartReceipt(task_id, attempt, allocation.allocation_id, allocation.backend_job_id, group.identity, stamp, output_dir)
            _write_new(os.path.join(output_dir, "start.json"), receipt.encode())
            pointer = os.path.join(task_dir, "current-attempt.json")
            _write_pointer(pointer, attempt.value)
            return AttemptFence(self.root, group.identity, task_id, allocation.allocation_id.value, attempt, output_dir)

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

    def publish_placement(self, fence: AttemptFence, receipt: SciencePlacementReceipt) -> SciencePlacementReceipt:
        """Publish one validated placement receipt after child quiescence.

        The receipt is written beneath the same attempt fence as the start and
        outcome records.  Requiring the terminal outcome first prevents a
        controller from presenting an in-flight child as placement evidence.

        Parameters
        ----------
        fence : AttemptFence
            Store-owned attempt fence returned by :meth:`begin_attempt`.
        receipt : SciencePlacementReceipt
            Controller-validated child-generation record.

        Returns
        -------
        SciencePlacementReceipt
            The immutable receipt that was durably published.

        Raises
        ------
        FenceRejected
            If the attempt is foreign, unfinished, already has a placement
            receipt, or the receipt crosses any fence identity.
        """

        if not isinstance(receipt, SciencePlacementReceipt):
            raise TypeError("receipt must be SciencePlacementReceipt")
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

    def _validate_fence(self, fence: AttemptFence, task_dir: str) -> StartReceipt:
        """Validate pointer, path, and every start-receipt fence binding."""
        pointer = os.path.join(task_dir, "current-attempt.json")
        _require_regular(pointer, "current attempt pointer")
        try:
            current = decode_json(Path(pointer).read_bytes())
            fields(current, {"schema", "attempt_id"}, "tpen-executor/current-attempt-v1", "current attempt")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise FenceRejected("current attempt pointer is unavailable") from error
        if current["attempt_id"] != fence.attempt_id.value:
            raise FenceRejected("stale attempt fence")
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
                    placement = SciencePlacementReceipt.decode(Path(placement_path).read_bytes())
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
                segment_path = os.path.join(output_dir, "segment.json")
                if os.path.lexists(segment_path):
                    _require_regular(segment_path, "accepted segment", ValueError)
                    segment = AcceptedSegment.decode(Path(segment_path).read_bytes())
                    if segment.task_id != task_id or segment.attempt_id.value != attempt_id:
                        raise ValueError("foreign accepted segment in group store")
                    segments.append(segment)
                attempts.append(ManifestAttempt(task_id, start.attempt_id, start, outcome, start.output_dir, placement))
        segments.sort(key=lambda item: (item.task_id, item.next_iteration, item.completed_updates))
        return RunManifest(group.group_id, group.identity, tuple(attempts), tuple(segments))


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


__all__ = ["AttemptFence", "DirectoryStore", "FenceRejected"]
