"""Read the executor's derived full-horizon completion record.

The reader is intentionally standard-library-only.  HI derives science
completion from this task-level record, never from process exit status or a
checkpoint marker in an attempt leaf.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any


SCHEMA = "tpen-executor/full-horizon-completion-v1"
_REQUIRED = frozenset(
    {
        "attempt_chain", "completed_at", "completed_updates", "core_acceptance_digest",
        "final_attempt_id", "final_checkpoint", "group_digest", "schema", "science_digest",
        "target_iterations", "task_id",
    }
)
_OPTIONAL = frozenset({"program_digest"})


class FullHorizonCompletionError(ValueError):
    """Raised when a completion record is malformed or crossed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hex64(value: object, field: str) -> None:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise FullHorizonCompletionError(f"{field} must be lowercase 64-hex")


def read_full_horizon_completion(
    attempt_core_dir: Path,
    *,
    group_digest: str,
    task_id: str,
    science_digest: str,
    target_iterations: int,
) -> bool:
    """Return whether the task-level record proves full-H completion.

    Parameters
    ----------
    attempt_core_dir
        Final attempt's ``<store>/groups/<group>/<task>/<attempt>/core`` path.
    group_digest, task_id
        Identities that must match the record and derived task directory.
    science_digest
        HI-issued row identity, compared by equality with the record.
    target_iterations
        Resolved trainer horizon; completion requires exact equality.
    """

    core = Path(attempt_core_dir)
    if not core.is_absolute() or core.name != "core":
        raise FullHorizonCompletionError("attempt_core_dir must be an absolute core path")
    task_dir = core.parent.parent
    record_path = task_dir / "full-horizon-completion.json"
    try:
        raw = record_path.read_bytes()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise FullHorizonCompletionError(f"completion record is unreadable: {record_path}") from error
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, FullHorizonCompletionError) as error:
        raise FullHorizonCompletionError("completion record is invalid JSON") from error
    if type(value) is not dict or not _REQUIRED <= set(value) <= _REQUIRED | _OPTIONAL:
        raise FullHorizonCompletionError("completion record has missing or extra fields")
    if _canonical(value) != raw:
        raise FullHorizonCompletionError("completion record is not canonical JSON")
    if value["schema"] != SCHEMA or value["group_digest"] != group_digest or value["task_id"] != task_id:
        raise FullHorizonCompletionError("completion record identity does not match the requested task")
    _hex64(value["group_digest"], "group_digest")
    _hex64(value["science_digest"], "science_digest")
    if value["science_digest"] != science_digest:
        return False
    if type(value["task_id"]) is not str or not value["task_id"] or Path(value["task_id"]).name != value["task_id"]:
        raise FullHorizonCompletionError("task_id must be a non-empty path component")
    if type(value["completed_updates"]) is not int or type(value["target_iterations"]) is not int:
        raise FullHorizonCompletionError("completion counters must be integers")
    if value["target_iterations"] != target_iterations or value["completed_updates"] < target_iterations:
        return False
    if type(value["final_attempt_id"]) is not str or not value["final_attempt_id"]:
        raise FullHorizonCompletionError("final_attempt_id must be non-empty")
    checkpoint = value["final_checkpoint"]
    if type(checkpoint) is not dict or set(checkpoint) != {"path", "content_id"} or any(type(checkpoint[key]) is not str or not checkpoint[key] for key in checkpoint):
        raise FullHorizonCompletionError("final_checkpoint must contain two non-empty strings")
    _hex64(value["core_acceptance_digest"], "core_acceptance_digest")
    if "program_digest" in value:
        _hex64(value["program_digest"], "program_digest")
    if type(value["completed_at"]) is not str or not value["completed_at"].endswith("Z"):
        raise FullHorizonCompletionError("completed_at must be canonical UTC text")
    try:
        timestamp = datetime.fromisoformat(value["completed_at"].removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise FullHorizonCompletionError("completed_at is not an ISO UTC timestamp") from error
    if timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value["completed_at"]:
        raise FullHorizonCompletionError("completed_at is not canonical UTC text")
    chain = value["attempt_chain"]
    if type(chain) is not list:
        raise FullHorizonCompletionError("attempt_chain must be a list")
    for link in chain:
        if type(link) is not dict or set(link) != {"predecessor_attempt_id", "successor_attempt_id", "request_digest"}:
            raise FullHorizonCompletionError("attempt_chain contains an invalid link")
        if any(type(link[key]) is not str or not link[key] for key in link):
            raise FullHorizonCompletionError("attempt_chain contains an empty identity")
        _hex64(link["request_digest"], "request_digest")
    if chain:
        if chain[-1]["successor_attempt_id"] != value["final_attempt_id"]:
            raise FullHorizonCompletionError("attempt_chain does not terminate at final_attempt_id")
        visited = {chain[0]["predecessor_attempt_id"]}
        for link in chain:
            if link["successor_attempt_id"] in visited:
                raise FullHorizonCompletionError("attempt_chain contains a cycle")
            visited.add(link["successor_attempt_id"])
        for previous, current in zip(chain, chain[1:]):
            if previous["successor_attempt_id"] != current["predecessor_attempt_id"]:
                raise FullHorizonCompletionError("attempt_chain is disconnected")
    return True


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FullHorizonCompletionError(f"duplicate completion field: {key}")
        result[key] = value
    return result


__all__ = ["FullHorizonCompletionError", "SCHEMA", "read_full_horizon_completion"]
