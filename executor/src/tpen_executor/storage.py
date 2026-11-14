"""Durable, fenced attempt receipts for an executor allocation."""

from __future__ import annotations

from contextlib import contextmanager
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
from .allocations import ConfiguredAllocation
from .contracts import AttemptId, ReadyGroup
from .results import (
    AcceptedSegment, CoreOutcomeRecord, ManifestAttempt, OutcomeReceipt, PLACEMENT_RECEIPT_TYPES, RunManifest,
    SciencePlacementReceipt, SciencePlacementReceiptV2, SciencePlacementReceiptV3, StartReceipt,
    SuccessorAdmissionRecord, decode_placement_receipt,
)
from .runners import CoreRunEvidence
from ._records import frozen_record


class FenceRejected(RuntimeError):
    """Raised when an attempt fence is stale, crossed, or already finished."""


class AdmissionFork(FenceRejected):
    """Raised when one predecessor is presented with a different successor."""

    code = "fork"


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
_SUCCESSOR_ADMISSION_SCHEMA = "tpen.execution/successor-admission-request-v1"


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

    def record_successor_admission(
        self,
        fence: AttemptFence,
        request: Mapping[str, Any],
        core_acceptance_digest: str,
    ) -> SuccessorAdmissionRecord:
        """Record a core-accepted successor request exactly once.

        ``core_acceptance_digest`` is an opaque digest supplied by the caller
        after core's admission decision (for example, a digest of an
        ``OperationalRestoreRequest``).  It is deliberately not interpreted
        or persisted as a second wire field; ``request_digest`` is the
        canonical digest of the closed admission request itself.
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
            start = self._validate_fence(fence, task_dir)
            if digest(built) != request_digest or built != request:
                raise FenceRejected("successor admission request does not match fenced core outcome")
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
                _write_new(os.path.join(predecessor_dir, _SUCCESSOR_ADMISSION_FILENAME), record.encode())
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
            record = SuccessorAdmissionRecord.decode(Path(path).read_bytes())
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


__all__ = ["AdmissionFork", "AttemptFence", "DirectoryStore", "FenceRejected"]
