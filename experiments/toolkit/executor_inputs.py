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
import re
from typing import Any


PROGRAM_SCHEMA = "tpen-executor/program-spec-ref-v1"
TASK_SCHEMA = "tpen-executor/task-spec-v1"
GROUP_SCHEMA = "tpen-executor/ready-group-v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _omega_container(value: Any) -> Any:
    """Copy an OmegaConf container without resolving interpolations."""

    try:
        from omegaconf import OmegaConf
    except ImportError:
        return value
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False)
    return value


def _portable_text(value: str, path_roots: tuple[Path, ...]) -> str:
    """Return a location-independent POSIX representation of an absolute path."""

    if not value.startswith("/"):
        return value
    candidate = Path(value)
    for root in (*path_roots, _REPO_ROOT):
        try:
            return candidate.resolve().relative_to(root.resolve()).as_posix() or "."
        except ValueError:
            continue
    # An external absolute path is still not allowed into a digest.  The
    # marker keeps its spelling visible without making it an absolute path.
    return f"<absolute>/{candidate.as_posix().lstrip('/')}"


def _json_safe(
    value: Any,
    path: str = "value",
    *,
    path_roots: tuple[Path, ...] = (),
) -> Any:
    """Validate JSON-shaped values and return a detached JSON value."""

    value = _omega_container(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return _portable_text(value, path_roots) if isinstance(value, str) else value
    if isinstance(value, Mapping):
        if type(value) is not dict:
            raise TypeError(f"{path} must be a plain mapping or OmegaConf container")
        if any(type(key) is not str for key in value):
            raise TypeError(f"{path} keys must be strings")
        return {
            key: _json_safe(item, f"{path}.{key}", path_roots=path_roots)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if type(value) not in (list, tuple):
            raise TypeError(f"{path} must be a plain sequence or OmegaConf container")
        return [_json_safe(item, f"{path}[]", path_roots=path_roots) for item in value]
    raise TypeError(f"{path} is not JSON-shaped")


def canonical_bytes(value: Any, *, path_roots: tuple[str | Path, ...] = ()) -> bytes:
    """Return canonical UTF-8 JSON bytes for one wire payload.

    Parameters
    ----------
    value : object
        JSON-shaped mapping or sequence to encode.
    path_roots : sequence of str or pathlib.Path, optional
        Absolute roots whose paths are emitted relative to the root.  The
        repository root is always included; this keeps digests portable across
        checkouts while preserving result paths relative to a plan root.
    """

    return json.dumps(
        _json_safe(value, path_roots=tuple(Path(root) for root in path_roots)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_digest(value: Any, *, path_roots: tuple[str | Path, ...] = ()) -> str:
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

    return hashlib.sha256(canonical_bytes(value, path_roots=path_roots)).hexdigest()


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


def program_spec_ref(
    config_ref: Mapping[str, Any], *, path_roots: tuple[str | Path, ...] = ()
) -> dict[str, Any]:
    """Build an unresolved ``ProgramSpecRef`` wire mapping.

    Parameters
    ----------
    config_ref : mapping
        Unresolved study configuration or a minimal reference to it.  Values
        such as ``${oc.env:...}`` are retained literally.

    Returns
    -------
    dict
        Plain executor ``ProgramSpecRef`` wire mapping.
    """

    if not isinstance(config_ref, Mapping):
        raise TypeError("config_ref must be a mapping")
    roots = tuple(Path(root) for root in path_roots)
    payload = _json_safe(config_ref, "config_ref", path_roots=roots)
    return {
        "schema": PROGRAM_SCHEMA,
        "config_ref": payload,
        "content_digest": payload_digest(payload, path_roots=path_roots),
    }


def _output_overrides(params: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the row-owned output overrides without resolving study config."""

    result_dir = params.get("result_dir")
    output: dict[str, Any] = {"run.layout": "flat"}
    if result_dir is not None:
        output["run.root"] = str(Path(str(result_dir)).parent)
    if "row_id" in params:
        output["run.run_id"] = params["row_id"]
    return output


def _task_config(task: Any) -> dict[str, Any]:
    """Choose a faithful unresolved reference without resolving study code."""

    params = dict(getattr(task, "params", {}))
    metadata = dict(getattr(task, "metadata", {}))
    if "config" in params:
        if params.get("kind") == "train":
            overrides = {
                "seed": params["seed"],
                "n_walkers": params["n_walkers"],
                "max_steps": params["max_steps"],
                "output": _output_overrides(params),
            }
        elif params.get("kind") == "eval":
            coordinate_keys = (
                "n_walkers",
                "n_draws",
                "burn_in",
                "discard_draws",
                "stride",
                "chunk_size",
                "task_names",
            )
            overrides = {
                "seed": params["seed"],
                "checkpoint": params["checkpoint_dir"],
                "coordinates": {key: params[key] for key in coordinate_keys},
                "output": _output_overrides(params),
            }
        else:
            overrides = {"params": params}
        return {"stage": task.stage, "config": params["config"], "overrides": overrides}
    if "matrix" in metadata:
        reference = {"matrix": metadata["matrix"]}
        if "runtime" in metadata:
            reference["runtime"] = metadata["runtime"]
        return reference
    # Rung plans have no study config object: retain the logical program and
    # science parameters, not an environment-specific executable path.
    return {"runtime": metadata.get("runtime", "unknown"), "params": params}


def _target(task: Any) -> tuple[int, str, dict[str, Any] | None]:
    params = dict(getattr(task, "params", {}))
    for key, units in (("max_steps", "steps"), ("steps", "steps"), ("record_capacity", "records")):
        if key in params:
            value = params[key]
            if type(value) is int and value >= 0:
                return value, units, None
            if isinstance(value, str) and re.fullmatch(r"\+?\d+", value.strip()):
                return int(value.strip()), units, None
            return (
                0,
                "unrepresentable",
                {
                    "field": key,
                    "value": repr(value),
                    "reason": "wire target_horizon requires a nonnegative integer",
                },
            )
    # A generic baseline pipeline may receive an intentionally minimal test or
    # adapter task with no declared scientific horizon.  Preserve the task
    # identity without inventing a step count; concrete study planners above
    # always provide ``steps``, ``max_steps``, or ``record_capacity``.
    return 0, "unspecified", None


def _plan_payload(plan: Any) -> dict[str, Any]:
    """Project a plan through its public manifest and task-row surfaces."""

    if not hasattr(plan, "to_manifest"):
        raise TypeError("plan must expose the StagePlanV2 to_manifest method")
    return {
        "manifest": plan.to_manifest(),
        "tasks": [task.to_dict() for task in _sequence(plan.tasks, "tasks")],
    }


def task_spec(
    task: Any, *, path_roots: tuple[str | Path, ...] = ()
) -> dict[str, Any]:
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
    horizon, units, _ = _target(task)
    return {
        "schema": TASK_SCHEMA,
        "task_id": task_id,
        "lineage_id": lineage_id,
        "program": program_spec_ref(_task_config(task), path_roots=path_roots),
        "science_digest": payload_digest(
            {"stage": task.stage, "run_id": task.run_id, "params": dict(task.params)},
            path_roots=path_roots,
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

    Returns
    -------
    dict
        Plain executor ``ReadyGroup`` wire mapping.  If a horizon cannot be
        represented by the wire schema, an ``unrepresentable`` entry records
        the task, value, and reason without blocking legacy dispatch.
    """

    tasks = _sequence(plan.tasks, "tasks")
    if not tasks:
        raise ValueError("tasks must be a non-empty sequence")
    path_roots = (Path(plan.results_root),)
    encoded = [task_spec(task, path_roots=path_roots) for task in tasks]
    members = encoded
    unrepresentable = [
        {"task_id": task.logical_task_id, **detail}
        for task in tasks
        for _, _, detail in [_target(task)]
        if detail is not None
    ]
    resolved_group_id = group_id or f"{plan.study}:{plan.stage}:{plan.plan_id}"
    _text(resolved_group_id, "group_id")
    group = {
        "schema": GROUP_SCHEMA,
        "group_id": resolved_group_id,
        "members": members,
        "execution_plan_digest": payload_digest(
            _plan_payload(plan), path_roots=path_roots
        ),
    }
    if unrepresentable:
        group["unrepresentable"] = unrepresentable
    return group


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
