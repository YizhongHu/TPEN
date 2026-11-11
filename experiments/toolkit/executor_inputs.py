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
import os
from pathlib import Path
import re
import tempfile
from typing import Any


PROGRAM_SCHEMA = "tpen-executor/program-spec-ref-v1"
TASK_SCHEMA = "tpen-executor/task-spec-v1"
GROUP_SCHEMA = "tpen-executor/ready-group-v1"
DIAGNOSTICS_SCHEMA = "tpen-experiments/executor-input-diagnostics-v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATH_FIELDS = frozenset(
    {
        "config",
        "config_file",
        "config_path",
        "checkpoint",
        "checkpoint_dir",
        "checkpoint_path",
        "inputs",
        "log",
        "log_path",
        "logs",
        "output",
        "output_dir",
        "outputs",
        "output_paths",
        "path",
        "paths",
        "result_dir",
        "results_dir",
        "results_root",
        "run.root",
        "script",
        "script_path",
        "status_path",
    }
)
_REPO_RELATIVE_PATH_FIELDS = frozenset(
    {"config", "config_file", "config_path", "script", "script_path"}
)


class UnrepresentableInput(ValueError):
    """A producer value that cannot be carried by an executor wire record.

    The additive producer route catches only this exception, writes a
    toolkit-owned diagnostic record, and leaves the legacy dispatch route
    untouched.  Other exceptions are programming or validation errors and
    continue to propagate.
    """

    def __init__(
        self,
        field: str,
        value: Any,
        reason: str,
        *,
        diagnostics: list[dict[str, Any]] | None = None,
    ) -> None:
        self.field = field
        self.value_repr = repr(value)
        self.reason = reason
        self.diagnostics = diagnostics
        super().__init__(f"{field}: {reason}; value={self.value_repr}")

    def detail(self, *, task_id: str | None = None) -> dict[str, Any]:
        """Return the stable diagnostic shape for this refusal."""

        detail = {
            "field": self.field,
            "value": self.value_repr,
            "reason": self.reason,
        }
        if task_id is not None:
            detail["task_id"] = task_id
        return detail


def _path_projection(value: Any, field: str, path_roots: tuple[Path, ...]) -> Any:
    """Tag one declared path/reference with its owning root namespace."""

    if isinstance(value, Path):
        text = value.as_posix()
    elif type(value) is str:
        text = value
    elif isinstance(value, Mapping):
        return _project_declared_paths(value, path_roots)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_path_projection(item, field, path_roots) for item in value]
    else:
        return value

    candidate = Path(text)
    if not candidate.is_absolute():
        root = "repo" if field in _REPO_RELATIVE_PATH_FIELDS else "results"
        return {"root": root, "path": candidate.as_posix()}
    for root in path_roots:
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        return {"root": "results", "path": relative.as_posix() or "."}
    try:
        relative = candidate.relative_to(_REPO_ROOT)
    except ValueError:
        return {"root": "external", "path": candidate.as_posix()}
    return {"root": "repo", "path": relative.as_posix() or "."}


def _command_projection(
    value: Any, path_roots: tuple[Path, ...], script_slot: int | None
) -> Any:
    """Project only the producer-declared script token in an argv field."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return value
    if script_slot is None:
        return value
    projected = []
    for index, token in enumerate(value):
        if index == script_slot:
            projected.append(_path_projection(token, "script", path_roots))
        else:
            projected.append(token)
    return projected


def _project_declared_paths(
    value: Any,
    path_roots: tuple[Path, ...] = (),
    *,
    script_slot: int | None = None,
) -> Any:
    """Project only declared path/reference fields before canonical encoding."""

    value = _omega_container(value)
    if isinstance(value, Mapping):
        if type(value) is not dict:
            return value
        return {
            key: _command_projection(item, path_roots, script_slot)
            if key == "command"
            else _path_projection(item, key, path_roots)
            if key in _PATH_FIELDS
            else _project_declared_paths(item, path_roots, script_slot=script_slot)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if type(value) not in (list, tuple):
            return value
        return [_project_declared_paths(item, path_roots, script_slot=script_slot) for item in value]
    return value


def _omega_container(value: Any) -> Any:
    """Copy an OmegaConf container without resolving interpolations."""

    try:
        from omegaconf import OmegaConf
    except ImportError:
        return value
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False)
    return value


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
            raise UnrepresentableInput(path, value, "wire JSON cannot carry a non-finite number")
        return value
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
    raise UnrepresentableInput(path, value, "wire JSON cannot carry this value")


def canonical_bytes(value: Any, *, path_roots: tuple[str | Path, ...] = ()) -> bytes:
    """Return canonical UTF-8 JSON bytes for one wire payload.

    Parameters
    ----------
    value : object
        JSON-shaped mapping or sequence to encode.
    path_roots : sequence of str or pathlib.Path, optional
        Compatibility argument retained for callers of the r2 API.  It does
        not alter canonical encoding; declared path/reference fields must be
        projected explicitly before calling this function.
    """

    # ``path_roots`` remains accepted for callers that used the r2 API, but
    # canonical encoding itself never rewrites values.  Projection is an
    # explicit producer-boundary operation performed before this function.
    return json.dumps(
        _json_safe(value),
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

    Raises
    ------
    UnrepresentableInput
        If a config value cannot be carried by the JSON wire schema.  Decimal
        integer strings are not coerced here; the coercion rule applies to
        declared integer horizon fields in :func:`task_spec`.
    """

    if not isinstance(config_ref, Mapping):
        raise TypeError("config_ref must be a mapping")
    roots = tuple(Path(root) for root in path_roots)
    payload = _project_declared_paths(config_ref, roots)
    payload = _json_safe(payload, "config_ref")
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
            raise UnrepresentableInput(
                key,
                value,
                "wire target_horizon requires a nonnegative integer",
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

    Notes
    -----
    Decimal integer strings such as ``"100"`` are normalized to integer
    horizons.  Values that are not representable by the wire schema raise
    :class:`UnrepresentableInput`; the additive producer boundary records that
    refusal separately and does not block legacy dispatch.
    """

    task_id = _text(task.logical_task_id, "task_id")
    lineage_id = _text(task.run_id, "lineage_id")
    horizon, units, _ = _target(task)
    params = _project_declared_paths(dict(task.params), tuple(Path(root) for root in path_roots))
    return {
        "schema": TASK_SCHEMA,
        "task_id": task_id,
        "lineage_id": lineage_id,
        "program": program_spec_ref(_task_config(task), path_roots=path_roots),
        "science_digest": payload_digest(
            {"stage": task.stage, "run_id": task.run_id, "params": params},
        ),
        "target_horizon": horizon,
        "target_horizon_units": units,
    }


def ready_group(
    plan: Any, *, group_id: str | None = None, script_slot: int | None = None
) -> dict[str, Any]:
    """Build one ordered ``ReadyGroup`` mapping from a stage plan.

    Parameters
    ----------
    plan : StagePlanV2-like object
        A validated plan with an ordered ``tasks`` sequence.
    group_id : str, optional
        Stable group identity.  By default the study, stage, and plan id are
        namespaced together.
    script_slot : int, optional
        Index of the producer-owned script token in each task command.  Only
        this explicitly declared token is projected across checkout roots;
        arbitrary argv values are retained byte-for-byte.  ``None`` disables
        command projection.

    Returns
    -------
    dict
        Plain executor ``ReadyGroup`` wire mapping.  Every member must be
        decoder-valid; a group containing any unrepresentable member raises
        :class:`UnrepresentableInput` so no partial group can be emitted.

    Raises
    ------
    UnrepresentableInput
        If any member or plan field cannot be represented.  The additive
        producer boundary writes ``executor_input_diagnostics.json`` and keeps
        the legacy dispatch route running.
    """

    tasks = _sequence(plan.tasks, "tasks")
    if not tasks:
        raise ValueError("tasks must be a non-empty sequence")
    path_roots = (Path(plan.results_root),)
    encoded = []
    diagnostics = []
    for task in tasks:
        try:
            encoded.append(task_spec(task, path_roots=path_roots))
        except UnrepresentableInput as exc:
            diagnostics.extend(
                exc.diagnostics
                or [exc.detail(task_id=str(task.logical_task_id))]
            )
    if diagnostics:
        raise UnrepresentableInput(
            "group",
            diagnostics,
            "at least one group member is not representable",
            diagnostics=diagnostics,
        )
    members = encoded
    resolved_group_id = group_id or f"{plan.study}:{plan.stage}:{plan.plan_id}"
    _text(resolved_group_id, "group_id")
    group = {
        "schema": GROUP_SCHEMA,
        "group_id": resolved_group_id,
        "members": members,
        "execution_plan_digest": payload_digest(
            _project_declared_paths(
                _plan_payload(plan), path_roots, script_slot=script_slot
            )
        ),
    }
    return group


def _diagnostic_payload(
    plan: Any, exc: UnrepresentableInput, *, group_id: str | None = None
) -> dict[str, Any]:
    """Build the separate, non-executor diagnostic envelope."""

    resolved_group_id = group_id or f"{plan.study}:{plan.stage}:{plan.plan_id}"
    details = exc.diagnostics or [exc.detail()]
    normalized = []
    for detail in details:
        normalized.append({"group_id": resolved_group_id, **detail})
    return {
        "schema": DIAGNOSTICS_SCHEMA,
        "group_id": resolved_group_id,
        "unrepresentable": normalized,
    }


def write_ready_group(path: str | Path, plan: Any, *, group_id: str | None = None) -> Path:
    """Write canonical group bytes, or a diagnostic envelope on refusal.

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

    Notes
    -----
    Producer pipelines should use :func:`write_additive_ready_group` so a
    refusal is written to the sibling ``executor_input_diagnostics.json`` and
    the requested ``executor_input.json`` remains decoder-valid-only.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = ready_group(plan, group_id=group_id)
    except UnrepresentableInput as exc:
        # This low-level helper is retained for callers that explicitly name a
        # diagnostic artifact.  Producer pipelines use the additive helper
        # below so executor_input.json is never a diagnostic envelope.
        payload = _diagnostic_payload(plan, exc, group_id=group_id)
    _atomic_write(target, canonical_bytes(payload))
    return target


def _history_digest(plan: Any, *, script_slot: int | None) -> str:
    """Return a stable generation token, including for refused plans."""

    roots = (Path(plan.results_root),)
    projected = _project_declared_paths(
        _plan_payload(plan), roots, script_slot=script_slot
    )
    try:
        return payload_digest(projected)
    except UnrepresentableInput:
        # A refused plan can contain a value such as Path("100.5") in its
        # horizon.  It still needs a non-colliding history name even though
        # that value cannot enter the executor wire document.
        return hashlib.sha256(
            json.dumps(
                projected,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=repr,
            ).encode("utf-8")
        ).hexdigest()


def _retire_current(path: Path, *, digest: str) -> None:
    """Move a current artifact to a never-overwritten history name."""

    if not path.exists():
        return
    candidate = path.with_name(f"{path.stem}.superseded-{digest}{path.suffix}")
    serial = 2
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}.superseded-{digest}-{serial}{path.suffix}"
        )
        serial += 1
    os.replace(path, candidate)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes through a same-directory temporary file and rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_additive_ready_group(
    path: str | Path,
    plan: Any,
    *,
    group_id: str | None = None,
    script_slot: int | None = None,
) -> Path | None:
    """Write a valid group or a sibling diagnostic without blocking dispatch.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination for the decoder-valid ``executor_input.json`` document.
    plan : object
        Stage-plan object accepted by :func:`ready_group`.
    group_id : str, optional
        Optional stable group identity override.
    script_slot : int, optional
        Index of the producer-declared script token, or ``None`` when no
        command token is a producer-owned script reference.

    Returns
    -------
    pathlib.Path or None
        The executor-input path on success, or ``None`` when a diagnostic was
        recorded because a group member was unrepresentable.

    Raises
    ------
    UnrepresentableInput
        Never escapes this additive boundary; it is recorded in the sibling
        ``executor_input_diagnostics.json`` file.
    Exception
        Any exception other than :class:`UnrepresentableInput` propagates.
    """

    target = Path(path)
    diagnostic = target.with_name("executor_input_diagnostics.json")
    digest = _history_digest(plan, script_slot=script_slot)
    try:
        payload = ready_group(plan, group_id=group_id, script_slot=script_slot)
    except UnrepresentableInput as exc:
        _retire_current(target, digest=digest)
        _atomic_write(
            diagnostic, canonical_bytes(_diagnostic_payload(plan, exc, group_id=group_id))
        )
        return None
    _retire_current(diagnostic, digest=payload["execution_plan_digest"])
    _atomic_write(target, canonical_bytes(payload))
    return target


__all__ = [
    "GROUP_SCHEMA",
    "DIAGNOSTICS_SCHEMA",
    "PROGRAM_SCHEMA",
    "TASK_SCHEMA",
    "UnrepresentableInput",
    "canonical_bytes",
    "payload_digest",
    "program_spec_ref",
    "ready_group",
    "task_spec",
    "write_additive_ready_group",
    "write_ready_group",
]
