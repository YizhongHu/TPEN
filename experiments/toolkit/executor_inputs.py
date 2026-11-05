"""Build dependency-free executor input records from study plans.

The root experiment project deliberately emits mappings rather than importing
``tpen_executor``.  The mappings use the executor wire schema and the same
canonical JSON rule as the executor project, so they can cross the project
boundary through bytes or a JSON file without resolving study configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROGRAM_SCHEMA = "tpen-executor/program-spec-ref-v1"
TASK_SCHEMA = "tpen-executor/task-spec-v1"
GROUP_SCHEMA = "tpen-executor/ready-group-v1"


def _json_safe(value: Any, path: str = "value") -> Any:
    """Validate JSON-shaped values and return a detached JSON value."""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{path} keys must be strings")
        return {key: _json_safe(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item, f"{path}[]") for item in value]
    raise TypeError(f"{path} is not JSON-shaped")


def canonical_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes for one wire payload.

    Parameters
    ----------
    value : object
        JSON-shaped mapping or sequence to encode.
    """

    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_digest(value: Any) -> str:
    """Return the lowercase SHA-256 digest of canonical payload bytes.

    Parameters
    ----------
    value : object
        JSON-shaped value to digest.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


def program_spec_ref(config_ref: Mapping[str, Any]) -> dict[str, Any]:
    """Build an unresolved ``ProgramSpecRef`` wire mapping.

    Parameters
    ----------
    config_ref : mapping
        Unresolved study configuration or a minimal reference to it.  Values
        such as ``${oc.env:...}`` are retained literally.
    """

    if not isinstance(config_ref, Mapping):
        raise TypeError("config_ref must be a mapping")
    payload = _json_safe(dict(config_ref), "config_ref")
    return {
        "schema": PROGRAM_SCHEMA,
        "config_ref": payload,
        "content_digest": payload_digest(payload),
    }


def _task_config(task: Any) -> dict[str, Any]:
    """Choose a faithful unresolved reference without resolving study code."""

    params = dict(getattr(task, "params", {}))
    metadata = dict(getattr(task, "metadata", {}))
    if "config" in params:
        return {"stage": task.stage, "config": params["config"]}
    if "matrix" in metadata:
        reference = {"matrix": metadata["matrix"]}
        if "runtime" in metadata:
            reference["runtime"] = metadata["runtime"]
        return reference
    # Rung plans have no study config object: retain the logical program and
    # science parameters, not an environment-specific executable path.
    return {"runtime": metadata.get("runtime", "unknown"), "params": params}


def _target(task: Any) -> tuple[int, str]:
    params = dict(getattr(task, "params", {}))
    for key, units in (("max_steps", "steps"), ("steps", "steps"), ("record_capacity", "records")):
        if key in params:
            value = params[key]
            if type(value) is not int or value < 0:
                raise TypeError(f"{key} must be a nonnegative integer")
            return value, units
    # A generic baseline pipeline may receive an intentionally minimal test or
    # adapter task with no declared scientific horizon.  Preserve the task
    # identity without inventing a step count; concrete study planners above
    # always provide ``steps``, ``max_steps``, or ``record_capacity``.
    return 0, "unspecified"


def _plan_payload(plan: Any) -> dict[str, Any]:
    """Project a plan through its public manifest and task-row surfaces."""

    if not hasattr(plan, "to_manifest"):
        raise TypeError("plan must expose the StagePlanV2 to_manifest method")
    return {
        "manifest": plan.to_manifest(),
        "tasks": [task.to_dict() for task in _sequence(plan.tasks, "tasks")],
    }


def task_spec(task: Any) -> dict[str, Any]:
    """Build one ``TaskSpec`` wire mapping from a logical study task.

    Parameters
    ----------
    task : object
        Stage-plan task exposing logical identity, run identity, stage, and
        JSON-shaped parameters.

    Returns
    -------
    dict
        Plain executor ``TaskSpec`` wire mapping.
    """

    task_id = _text(task.logical_task_id, "task_id")
    lineage_id = _text(task.run_id, "lineage_id")
    horizon, units = _target(task)
    return {
        "schema": TASK_SCHEMA,
        "task_id": task_id,
        "lineage_id": lineage_id,
        "program": program_spec_ref(_task_config(task)),
        "science_digest": payload_digest(
            {"stage": task.stage, "run_id": task.run_id, "params": dict(task.params)}
        ),
        "target_horizon": horizon,
        "target_horizon_units": units,
    }


def ready_group(plan: Any, *, group_id: str | None = None) -> dict[str, Any]:
    """Build one ordered ``ReadyGroup`` mapping from a stage plan.

    Parameters
    ----------
    plan : StagePlanV2-like object
        A validated plan with an ordered ``tasks`` sequence.
    group_id : str, optional
        Stable group identity.  By default the study, stage, and plan id are
        namespaced together.
    """

    tasks = _sequence(plan.tasks, "tasks")
    if not tasks:
        raise ValueError("tasks must be a non-empty sequence")
    members = [task_spec(task) for task in tasks]
    resolved_group_id = group_id or f"{plan.study}:{plan.stage}:{plan.plan_id}"
    _text(resolved_group_id, "group_id")
    return {
        "schema": GROUP_SCHEMA,
        "group_id": resolved_group_id,
        "members": members,
        "execution_plan_digest": payload_digest(_plan_payload(plan)),
    }


def write_ready_group(path: str | Path, plan: Any, *, group_id: str | None = None) -> Path:
    """Write one ready-group mapping as canonical bytes and return its path.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file.
    plan : object
        Stage-plan object accepted by :func:`ready_group`.
    group_id : str, optional
        Optional stable group identity override.

    Returns
    -------
    pathlib.Path
        The destination path.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_bytes(ready_group(plan, group_id=group_id)))
    return target


__all__ = [
    "GROUP_SCHEMA",
    "PROGRAM_SCHEMA",
    "TASK_SCHEMA",
    "canonical_bytes",
    "payload_digest",
    "program_spec_ref",
    "ready_group",
    "task_spec",
    "write_ready_group",
]
