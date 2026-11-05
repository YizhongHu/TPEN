"""Durable, fenced attempt receipts for an executor allocation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

from ._codec import canonical_bytes, decode_json, fields, text
from .allocations import ConfiguredAllocation
from .contracts import AttemptId, ReadyGroup
from .results import ManifestAttempt, OutcomeReceipt, RunManifest, StartReceipt


class FenceRejected(RuntimeError):
    """Raised when an attempt fence is stale, crossed, or already finished."""


@dataclass(frozen=True, slots=True)
class AttemptFence:
    """Opaque identity binding for one store-owned attempt."""

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


def _write_new(path: str, payload: bytes) -> None:
    """Write a new file durably, refusing replacement of an existing file."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Claim the destination before replace: a receipt is never overwritten.
        claim = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(claim)
        try:
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            raise
        temporary = ""
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


@dataclass(frozen=True, slots=True)
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
        os.makedirs(root, exist_ok=True)
        object.__setattr__(self, "root", root)

    def _group_dir(self, group: ReadyGroup) -> str:
        path = _lexical(os.path.join(self.root, "groups", group.identity))
        if not _inside(self.root, path):
            raise FenceRejected("group path escapes store root")
        return path

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
        """
        if not isinstance(group, ReadyGroup) or not isinstance(allocation, ConfiguredAllocation):
            raise TypeError("group and allocation have invalid types")
        text(task_id, "task_id")
        if task_id not in {member.task_id for member in group.members}:
            raise ValueError("task_id is not a member of group")
        group_dir = self._group_dir(group)
        task_dir = _lexical(os.path.join(group_dir, task_id))
        if not _inside(group_dir, task_dir):
            raise FenceRejected("task path escapes store root")
        os.makedirs(task_dir, exist_ok=True)
        attempt = AttemptId(uuid4().hex)
        output_dir = _lexical(os.path.join(task_dir, attempt.value))
        try:
            os.mkdir(output_dir)
        except FileExistsError as error:
            raise FenceRejected("attempt directory already exists") from error
        stamp = started_at_utc or "unknown"
        receipt = StartReceipt(task_id, attempt, allocation.allocation_id, allocation.backend_job_id, group.identity, stamp, output_dir)
        _write_new(os.path.join(output_dir, "start.json"), receipt.encode())
        pointer = os.path.join(task_dir, "current-attempt.json")
        _write_pointer(pointer, attempt.value)
        return AttemptFence(self.root, group.identity, task_id, allocation.allocation_id.value, attempt, output_dir)

    def finish_attempt(self, fence: AttemptFence, status: str, finished_at_utc: str, result_digest: str | None = None) -> OutcomeReceipt:
        """Write exactly one outcome after validating the complete fence.

        Raises
        ------
        FenceRejected
            If the fence is stale, crossed, or already finished.
        """
        self._validate_fence(fence)
        output_dir = _lexical(fence.output_dir)
        outcome_path = os.path.join(output_dir, "outcome.json")
        if os.path.exists(outcome_path):
            raise FenceRejected("attempt is already finished")
        start = StartReceipt.decode(Path(os.path.join(output_dir, "start.json")).read_bytes())
        try:
            outcome = OutcomeReceipt(fence.task_id, fence.attempt_id, fence.group_digest, finished_at_utc, status, start.output_dir, result_digest)
            _write_new(outcome_path, outcome.encode())
        except (OSError, ValueError, TypeError) as error:
            if isinstance(error, (ValueError, TypeError)):
                raise
            raise FenceRejected("outcome could not be written") from error
        return outcome

    def _validate_fence(self, fence: AttemptFence) -> None:
        if not isinstance(fence, AttemptFence) or fence.root != self.root:
            raise FenceRejected("fence belongs to another store")
        pointer = _lexical(os.path.join(self.root, "groups", fence.group_digest, fence.task_id, "current-attempt.json"))
        if not _inside(self.root, pointer):
            raise FenceRejected("fence path escapes store root")
        try:
            current = decode_json(Path(pointer).read_bytes())
            fields(current, {"schema", "attempt_id"}, "tpen-executor/current-attempt-v1", "current attempt")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise FenceRejected("current attempt pointer is unavailable") from error
        if current["attempt_id"] != fence.attempt_id.value:
            raise FenceRejected("stale attempt fence")
        expected = _lexical(os.path.join(self.root, "groups", fence.group_digest, fence.task_id, fence.attempt_id.value))
        if expected != _lexical(fence.output_dir) or fence.allocation_id != self._read_allocation(expected):
            raise FenceRejected("crossed attempt fence")

    @staticmethod
    def _read_allocation(output_dir: str) -> str:
        start = StartReceipt.decode(Path(os.path.join(output_dir, "start.json")).read_bytes())
        return start.allocation_id.value

    def manifest(self, group: ReadyGroup) -> RunManifest:
        """Rebuild a strict manifest only from durable receipts on disk."""
        if not isinstance(group, ReadyGroup):
            raise TypeError("group must be ReadyGroup")
        group_dir = self._group_dir(group)
        attempts: list[ManifestAttempt] = []
        members = {member.task_id for member in group.members}
        if not os.path.isdir(group_dir):
            return RunManifest(group.group_id, group.identity, ())
        for task_id in sorted(os.listdir(group_dir)):
            task_dir = _lexical(os.path.join(group_dir, task_id))
            if not os.path.isdir(task_dir) or task_id not in members:
                raise ValueError("foreign or malformed task directory in group store")
            for attempt_id in sorted(os.listdir(task_dir)):
                if attempt_id == "current-attempt.json":
                    continue
                output_dir = _lexical(os.path.join(task_dir, attempt_id))
                if not os.path.isdir(output_dir):
                    raise ValueError("foreign entry in task store")
                start = StartReceipt.decode(Path(os.path.join(output_dir, "start.json")).read_bytes())
                if start.group_digest != group.identity or start.task_id != task_id or start.attempt_id.value != attempt_id:
                    raise ValueError("foreign receipt in group store")
                outcome_file = os.path.join(output_dir, "outcome.json")
                outcome = OutcomeReceipt.decode(Path(outcome_file).read_bytes()) if os.path.exists(outcome_file) else None
                attempts.append(ManifestAttempt(task_id, start.attempt_id, start, outcome, start.output_dir))
        return RunManifest(group.group_id, group.identity, tuple(attempts))


def _write_pointer(path: str, attempt_id: str) -> None:
    """Atomically advance a task pointer; pointer replacement is intentional."""
    payload = canonical_bytes({"schema": "tpen-executor/current-attempt-v1", "attempt_id": attempt_id})
    parent = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-pointer-", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            try: os.unlink(temporary)
            except FileNotFoundError: pass


__all__ = ["AttemptFence", "DirectoryStore", "FenceRejected"]
