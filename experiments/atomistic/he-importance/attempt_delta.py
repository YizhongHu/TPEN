"""Validate the executor's closed attempt-delta wire value.

This module deliberately uses only the Python standard library.  The HI
launcher consumes the delta as an opaque operational hand-off and must not
import executor implementation modules to inspect it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import posixpath
from pathlib import Path
from typing import Any


SCHEMA = "tpen-executor/attempt-delta-v1"
_FIELDS = frozenset({"attempt_core_dir", "attempt_id", "group_digest", "load", "schema", "task_id"})
_LOAD_FIELDS = frozenset({"mode", "path"})


class AttemptDeltaError(ValueError):
    """Raised when a supplied attempt delta is malformed or crossed."""


@dataclass(frozen=True)
class AttemptDelta:
    """The validated operational allowance for one physical attempt."""

    attempt_core_dir: Path
    attempt_id: str
    group_digest: str
    load_mode: str
    load_path: Path | None
    task_id: str
    digest: str


def _canonical(value: Any) -> bytes:
    """Return the contract's compact, recursively sorted UTF-8 JSON bytes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _single_component(value: object, field: str) -> str:
    if type(value) is not str or not value or Path(value).name != value or value in {".", ".."}:
        raise AttemptDeltaError(f"{field} must be one non-empty path component")
    return value


def _absolute_normalized(value: object, field: str) -> Path:
    if type(value) is not str or not value:
        raise AttemptDeltaError(f"{field} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or path != Path(_os_path_norm(value)):
        raise AttemptDeltaError(f"{field} must be absolute and normalized")
    return path


def _os_path_norm(value: str) -> str:
    """Normalize a path without resolving symlinks or touching the filesystem."""

    return posixpath.normpath(value)


def parse_attempt_delta(raw: str | bytes) -> AttemptDelta:
    """Parse and validate one canonical ``--attempt-delta`` JSON argument.

    Parameters
    ----------
    raw
        Canonical JSON text supplied by the configured executor program.
    """

    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise AttemptDeltaError("attempt delta is not valid JSON") from error
    if type(value) is not dict or set(value) != _FIELDS:
        raise AttemptDeltaError("attempt delta must contain exactly the six contract fields")
    if value["schema"] != SCHEMA:
        raise AttemptDeltaError(f"unsupported attempt delta schema: {value['schema']!r}")
    attempt_id = _single_component(value["attempt_id"], "attempt_id")
    group_digest = _single_component(value["group_digest"], "group_digest")
    task_id = _single_component(value["task_id"], "task_id")
    core = _absolute_normalized(value["attempt_core_dir"], "attempt_core_dir")
    if (
        len(core.parts) < 4
        or core.name != "core"
        or core.parent.name != attempt_id
        or core.parent.parent.name != task_id
        or core.parent.parent.parent.name != group_digest
    ):
        raise AttemptDeltaError("attempt_core_dir does not end in <task>/<attempt>/core")
    load = value["load"]
    if type(load) is not dict or set(load) != _LOAD_FIELDS:
        raise AttemptDeltaError("load must contain exactly mode and path")
    mode = load["mode"]
    if mode not in {"fresh", "train_resume"}:
        raise AttemptDeltaError("load.mode must be fresh or train_resume")
    load_path = load["path"]
    if mode == "fresh":
        if load_path is not None:
            raise AttemptDeltaError("fresh load.path must be null")
    else:
        load_path = _absolute_normalized(load_path, "load.path")
        if load_path == core or core in load_path.parents:
            raise AttemptDeltaError("train_resume load.path must be outside the current attempt leaf")
    canonical = _canonical(value)
    raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else raw
    if raw_bytes != canonical:
        raise AttemptDeltaError("attempt delta is not canonical JSON")
    return AttemptDelta(core, attempt_id, group_digest, mode, load_path, task_id, hashlib.sha256(canonical).hexdigest())


__all__ = ["AttemptDelta", "AttemptDeltaError", "SCHEMA", "parse_attempt_delta"]
