"""Stable config and checkpoint-content hashing for checkpoint manifests."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

RESOLVED_CONFIG_HASH_EXCLUSIONS = (
    ("run", "run_id"),
    ("run", "dir"),
)

RESOLVED_CONFIG_IDENTITY_SCHEMA = "tpen-config-identity-v1"
RESOLVED_CONFIG_IDENTITY_EXCLUSIONS = (
    ("run", "root"),
    ("run", "run_id"),
    ("run", "dir"),
    ("run", "layout"),
    ("run", "timezone"),
)

# These fields are populated or consumed by the core run boundary in
# ``tpen/run.py:147-192``.  They are deliberately exact: ``run.timestamp`` and
# ``run.start_time_unix`` are artifact JSON fields, not config producers.
_IDENTITY_RUN_DEFAULTS = {
    "root": "outputs",
    "layout": "nested",
    "timezone": "UTC",
}
_IDENTITY_RUN_FIELDS = ("root", "run_id", "dir", "layout", "timezone")
_RUN_INTERPOLATION = re.compile(r"\$\{run(?:\.[^{}]+)?\}")
_RUN_SENTINEL_MARKER = "__tpen_identity_run_sentinel_7b1e4c93f0a64b9d__"

# Operational slots are schema-bound to the built-in producers.  A user
# callback/logger with an arbitrary ``output_path`` or ``path`` is not silently
# removed from a science identity.
_CALLBACK_OUTPUT_FIELDS = {
    "tpen.callback.ConfigSnapshot": {"output_path"},
    "tpen.callback.ResolvedConfigSnapshot": {"output_path"},
    "tpen.callback.Metadata": {"output_path"},
    "tpen.callback.Status": {"output_path"},
    "tpen.callback.Checkpoint": {"output_dir"},
    "tpen.callback.InitialCheckpoint": {"output_dir"},
    # ArtifactIndex.path is an optional diagnostics index destination, not a
    # value consumed by the evaluation model.
    "tpen.callback.ArtifactIndex": {"path"},
    # FailureLog.path is an optional diagnostics failure-log destination, not
    # a value consumed by the evaluation model.
    "tpen.callback.FailureLog": {"path"},
    # ParameterActivity.output_path is the run's first-update inventory file,
    # an observational artifact rather than a science input.
    "tpen.callback.ParameterActivity": {"output_path"},
    # RawTrainingBatchRecorder.path is the destination for its immutable
    # training-history partitions, not a model or sampler input.
    "tpen.callback.RawTrainingBatchRecorder": {"path"},
    # The package exports this compatibility alias for the same recorder.
    "tpen.callback.TrainingBatchRecorder": {"path"},
    # Producer: ``tpen.callback.RuntimeEquivariance`` writes its diagnostics
    # below this directory; consumer: ``RuntimeEquivariance.__init__`` and its
    # artifact writer.  This is an operational destination, not model input.
    "tpen.callback.RuntimeEquivariance": {"artifact_dir"},
    # Defining-module targets are also valid Hydra targets for these same
    # callback implementations; keep their operational slots closed too.
    "tpen.callback.checkpoint.Checkpoint": {"output_dir"},
    "tpen.callback.equivariance.RuntimeEquivariance": {"artifact_dir"},
    "tpen.callback.evaluation.ArtifactIndex": {"path"},
    "tpen.callback.evaluation.FailureLog": {"path"},
    "tpen.callback.initial_checkpoint.InitialCheckpoint": {"output_dir"},
    "tpen.callback.metadata.Metadata": {"output_path"},
    "tpen.callback.parameter_activity.ParameterActivity": {"output_path"},
    "tpen.callback.raw_training_batch_recorder.RawTrainingBatchRecorder": {"path"},
    "tpen.callback.raw_training_batch_recorder.TrainingBatchRecorder": {"path"},
    "tpen.callback.snapshot.ConfigSnapshot": {"output_path"},
    "tpen.callback.snapshot.ResolvedConfigSnapshot": {"output_path"},
    "tpen.callback.status.Status": {"output_path"},
    # The ``spenn`` study configs use the same callback producers under their
    # compatibility import prefix.  Keep these exact targets in the closed
    # table rather than treating arbitrary callback output-like fields as
    # operational.
    "spenn.callback.ConfigSnapshot": {"output_path"},
    "spenn.callback.ResolvedConfigSnapshot": {"output_path"},
    "spenn.callback.Metadata": {"output_path"},
    "spenn.callback.Status": {"output_path"},
    "spenn.callback.Checkpoint": {"output_dir"},
    "spenn.callback.InitialCheckpoint": {"output_dir"},
    # Compatibility aliases retain the same operational destinations for the
    # evaluation diagnostics, parameter inventory, and training history.
    "spenn.callback.ArtifactIndex": {"path"},
    "spenn.callback.FailureLog": {"path"},
    "spenn.callback.ParameterActivity": {"output_path"},
    "spenn.callback.RawTrainingBatchRecorder": {"path"},
    "spenn.callback.TrainingBatchRecorder": {"path"},
    "spenn.callback.RuntimeEquivariance": {"artifact_dir"},
}
_LOGGER_PATH_FIELDS = {
    "tpen.logging.CSV": {"path"},
    "tpen.logging.JSONL": {"path"},
    # WandB.dir controls the local dashboard-artifact destination; it is not
    # part of the science config represented by the logger's metrics.
    "tpen.logging.WandB": {"dir"},
    "tpen.logging.csv.CSV": {"path"},
    "tpen.logging.jsonl.JSONL": {"path"},
    "tpen.logging.wandb.WandB": {"dir"},
    "spenn.logging.CSV": {"path"},
    "spenn.logging.JSONL": {"path"},
    "spenn.logging.WandB": {"dir"},
}
# No callback/logger constructor path-like fields are science inputs on this
# head. Keep the empty, reason-bearing registry closed so a future exception
# must be named explicitly instead of being inferred from a parameter name.
_NON_OPERATIONAL_PATH_FIELDS: dict[str, dict[str, str]] = {}
_EVALUATOR_TARGET = "tpen.evaluation.Evaluator"


_FILE_HASH_CHUNK_BYTES = 1 << 20


class ResolvedConfigIdentityError(ValueError):
    """Raised when a run-derived value would enter the public config identity."""


def file_sha256(path: Path | str) -> str:
    """Return the sha256 of a file's contents.

    Used to identify a checkpoint by what it *is* rather than by where it sits.
    A path-keyed identity breaks the moment a run tree is moved, re-collected,
    or archived, and two different checkpoints can occupy the same path over
    the life of a run.

    Parameters
    ----------
    path : pathlib.Path or str
        File to hash.

    Returns
    -------
    str
        Lowercase 64-character hex digest.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    IsADirectoryError
        If `path` is a directory. Checkpoint identity is per file; hashing a
        directory would depend on traversal order.
    """

    resolved = Path(path)
    if resolved.is_dir():
        raise IsADirectoryError(f"checkpoint hash needs a file, got directory: {resolved}")
    digest = hashlib.sha256()
    # Chunked so a multi-gigabyte checkpoint is never held in memory at once.
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_FILE_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_config_hash(config: Any) -> str:
    """Return a sha256 hash of a JSON-safe, resolved config container.

    The hash uses canonical JSON with sorted keys. Unsupported objects fail
    loudly instead of being coerced with ``default=str``.
    """

    safe = _json_safe(to_plain_container(config))
    encoded = json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolved_config_hash(config: Any) -> str:
    """Hash the resolved run config after documented run-specific exclusions."""

    plain = to_plain_container(config)
    if isinstance(plain, Mapping):
        plain = copy.deepcopy(plain)
        for path in RESOLVED_CONFIG_HASH_EXCLUSIONS:
            _delete_path(plain, path)
    return stable_config_hash(plain)


def canonical_resolved_config_bytes(
    config: Any, *, run_values: Mapping[str, Any] | None = None
) -> bytes:
    """Return canonical bytes for the public resolved-config identity.

    The identity is deliberately separate from checkpoint manifest hashes:
    checkpoint compatibility retains its historical exclusions, while this
    public contract has a closed, documented run-artifact exclusion list.
    """

    normalized = _identity_resolved_container(config, run_values=run_values)
    if not isinstance(normalized, Mapping):
        raise TypeError("resolved configuration identity requires a mapping")
    for path in RESOLVED_CONFIG_IDENTITY_EXCLUSIONS:
        _delete_pattern(normalized, path)
    _delete_known_operational_slots(normalized)
    safe = _json_safe(normalized)
    return json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def resolved_config_identity(
    config: Any, *, run_values: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """Return the versioned public identity for one intended configuration.

    ``config`` remains unresolved until run-derived fields have been replaced
    by field-specific sentinels.  The core launch path supplies actual run
    values separately while the study side normally omits them, so both sides
    use the same unresolved composition and refusal boundary.
    """

    return {
        "schema": RESOLVED_CONFIG_IDENTITY_SCHEMA,
        "sha256": hashlib.sha256(
            canonical_resolved_config_bytes(config, run_values=run_values)
        ).hexdigest(),
    }


def component_config_hash(config: Any, section: str) -> str | None:
    """Hash one top-level config section, returning ``None`` when absent."""

    plain = to_plain_container(config)
    if not isinstance(plain, Mapping) or section not in plain:
        return None
    return stable_config_hash(plain[section])


def checkpoint_hashes(config: Any) -> dict[str, str | None]:
    """Return the standard checkpoint manifest hashes for `config`."""

    return {
        "resolved_config": resolved_config_hash(config),
        "model_config": component_config_hash(config, "model"),
        "optimizer_config": component_config_hash(config, "optimizer"),
        "trainer_config": component_config_hash(config, "trainer"),
        "sampler_config": component_config_hash(config, "sampler"),
        "hamiltonian_config": component_config_hash(config, "hamiltonian_terms"),
    }


def to_plain_container(config: Any) -> Any:
    """Resolve OmegaConf containers and return plain Python containers."""

    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return config


def populate_resolved_config_defaults(config: Any, *, timezone: str | None = None) -> dict[str, Any]:
    """Apply the core run-boundary defaults to an unresolved config copy.

    This is the one shared preparation step used by the core launch path and
    the study-side identity helper.  It intentionally does not manufacture a
    run id or run directory; those are populated by the launch and replaced by
    the identity sentinel before interpolation resolution.
    """

    plain = to_unresolved_container(config)
    if not isinstance(plain, Mapping):
        raise TypeError("resolved configuration identity requires a mapping")
    if any(type(key) is not str for key in plain):
        raise TypeError("resolved configuration mapping keys must be strings")
    prepared = copy.deepcopy(dict(plain))
    experiment = prepared.setdefault("experiment", {})
    run = prepared.setdefault("run", {})
    if not isinstance(experiment, dict) or not isinstance(run, dict):
        raise TypeError("experiment and run config sections must be mappings")
    name = experiment.get("name")
    run_name = _normalized_text(experiment.get("run_name"))
    if run_name is None:
        run_name = _normalized_text(name) or "tpen_run"
    experiment["run_name"] = run_name
    for key, default in _IDENTITY_RUN_DEFAULTS.items():
        if key == "timezone" and run.get(key) is None:
            run[key] = timezone if timezone is not None else default
        else:
            run.setdefault(key, default)
    return prepared


def to_unresolved_container(config: Any) -> Any:
    """Return plain containers without resolving OmegaConf interpolations."""

    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=False)
    return config


def _identity_resolved_container(
    config: Any, *, run_values: Mapping[str, Any] | None = None
) -> Any:
    prepared = populate_resolved_config_defaults(config)
    prepared.setdefault("run", {})
    run = prepared["run"]
    if not isinstance(run, dict):
        raise TypeError("run config section must be a mapping")
    if run_values is not None:
        if not isinstance(run_values, Mapping):
            raise TypeError("run_values must be a mapping")
        for key, value in run_values.items():
            if key not in _IDENTITY_RUN_FIELDS:
                raise TypeError(f"unsupported run identity value {key!r}")
            run[key] = copy.deepcopy(value)
    # The launch populates these at ``tpen/run.py:180-192``.  Add the same
    # slots on the study side, then replace every value before OmegaConf can
    # resolve an interpolation that copied it into another subtree.
    run.setdefault("run_id", None)
    run.setdefault("dir", None)
    _reject_science_run_interpolations(prepared)
    for field in _IDENTITY_RUN_FIELDS:
        run[field] = _run_sentinel(field)
    # Resolve the same config semantics as the run path.  The public identity
    # is over the resolved config, so resolver values are part of science
    # identity and must not be replaced by expression text.
    resolved = OmegaConf.to_container(OmegaConf.create(prepared), resolve=True)
    normalized = copy.deepcopy(resolved)
    if not isinstance(normalized, Mapping):
        raise TypeError("resolved configuration identity requires a mapping")
    # Remove explicit exclusions before scanning. Operational consumers may
    # receive run paths; retained science leaves may not, including strings
    # nested in lists or reached through indirect aliases.
    scan_target = copy.deepcopy(normalized)
    for path in RESOLVED_CONFIG_IDENTITY_EXCLUSIONS:
        _delete_pattern(scan_target, path)
    _delete_known_operational_slots(scan_target)
    _reject_run_sentinels(scan_target)
    return normalized


def _normalized_text(value: Any) -> str | None:
    """Match the core run-boundary normalization for durable text fields."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _run_sentinel(field: str) -> str:
    return f"{_RUN_SENTINEL_MARKER}:{field}__"


def _reject_run_sentinels(container: Any, path: tuple[object, ...] = ()) -> None:
    """Refuse any retained leaf contaminated by a run-field sentinel."""

    if isinstance(container, Mapping):
        for key, value in container.items():
            _reject_run_sentinels(value, path + (key,))
    elif isinstance(container, (list, tuple)):
        for index, value in enumerate(container):
            _reject_run_sentinels(value, path + (index,))
    elif isinstance(container, str) and _RUN_SENTINEL_MARKER in container:
        dotted = ".".join(str(part) for part in path)
        raise ResolvedConfigIdentityError(
            f"run-derived sentinel retained at {dotted}; run values cannot feed science"
        )


def _reject_science_run_interpolations(container: Any, path: tuple[object, ...] = (), ancestors: tuple[str, ...] = ()) -> None:
    if isinstance(container, Mapping):
        target = container.get("_target_")
        next_ancestors = ancestors + ((target,) if isinstance(target, str) else ())
        for key, value in container.items():
            child_path = path + (key,)
            if isinstance(value, str) and _RUN_INTERPOLATION.search(value) and not _is_operational_slot(child_path, container, key, next_ancestors):
                dotted = ".".join(str(part) for part in child_path)
                raise ResolvedConfigIdentityError(
                    f"run-populated interpolation at {dotted} would feed science; "
                    "bind it to an operational slot"
                )
            _reject_science_run_interpolations(value, child_path, next_ancestors)
    elif isinstance(container, (list, tuple)):
        for index, value in enumerate(container):
            _reject_science_run_interpolations(value, path + (index,), ancestors)


def _is_operational_slot(path: tuple[object, ...], owner: Mapping[str, Any], key: object, ancestors: tuple[str, ...]) -> bool:
    if path and path[0] == "run":
        return True
    target = owner.get("_target_")
    if isinstance(target, str) and key in _CALLBACK_OUTPUT_FIELDS.get(target, set()):
        return True
    if isinstance(target, str) and key in _LOGGER_PATH_FIELDS.get(target, set()):
        return True
    # Producer: evaluation config fragments are composed into
    # ``evaluator.tasks`` by Hydra/OmegaConf; consumer: ``Evaluator`` resolves
    # each task's output directory before running the generator.  The fragment
    # exists at this path before composition, so it needs its own closed slot.
    if (
        key == "output_dir"
        and len(path) == 3
        and path[0] == "evaluation_tasks"
        and isinstance(path[1], str)
    ):
        return True
    return (
        key == "output_dir"
        and len(path) == 4
        and path[:2] == ("evaluator", "tasks")
        and isinstance(path[2], int)
        and _EVALUATOR_TARGET in ancestors
    )


def _delete_known_operational_slots(container: Any) -> None:
    if isinstance(container, Mapping):
        target = container.get("_target_")
        if isinstance(container, dict):
            if isinstance(target, str):
                for key in _CALLBACK_OUTPUT_FIELDS.get(target, set()) | _LOGGER_PATH_FIELDS.get(target, set()):
                    container.pop(key, None)
                if target == _EVALUATOR_TARGET:
                    tasks = container.get("tasks")
                    if isinstance(tasks, (list, tuple)):
                        for task in tasks:
                            if isinstance(task, dict):
                                task.pop("output_dir", None)
            evaluation_tasks = container.get("evaluation_tasks")
            if isinstance(evaluation_tasks, dict):
                for task in evaluation_tasks.values():
                    if isinstance(task, dict):
                        task.pop("output_dir", None)
            for value in tuple(container.values()):
                _delete_known_operational_slots(value)
    elif isinstance(container, (list, tuple)):
        for item in container:
            _delete_known_operational_slots(item)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"non-finite float is not JSON-safe for hashing: {value!r}")
        return value
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"config hash mapping keys must be strings, got {type(key).__name__}")
            safe[key] = _json_safe(item)
        return safe
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"config value is not JSON/YAML-safe for hashing: {type(value).__name__}")


def _delete_path(container: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = container
    for key in path[:-1]:
        if not isinstance(current, dict) or key not in current:
            return
        current = current[key]
    if isinstance(current, dict):
        current.pop(path[-1], None)


def _delete_pattern(container: dict[Any, Any] | list[Any] | tuple[Any, ...], path: tuple[str, ...]) -> None:
    """Delete one explicitly declared path, allowing list-index wildcards."""

    if not path:
        return
    key = path[0]
    if key == "*":
        if isinstance(container, (list, tuple)):
            for item in container:
                if isinstance(item, (dict, list, tuple)):
                    _delete_pattern(item, path[1:])
        return
    if not isinstance(container, dict) or key not in container:
        return
    if len(path) == 1:
        container.pop(key, None)
        return
    child = container[key]
    if isinstance(child, (dict, list, tuple)):
        _delete_pattern(child, path[1:])
