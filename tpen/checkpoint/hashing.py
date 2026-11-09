"""Stable config and checkpoint-content hashing for checkpoint manifests."""

from __future__ import annotations

import copy
import hashlib
import json
import math
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
    ("run", "timestamp"),
    ("run", "start_time_unix"),
    ("callbacks", "*", "output_path"),
    ("callbacks", "*", "output_dir"),
    ("loggers", "*", "path"),
    ("loggers", "*", "output_path"),
    ("loggers", "*", "output_dir"),
    ("evaluator", "tasks", "*", "output_dir"),
)


_FILE_HASH_CHUNK_BYTES = 1 << 20


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


def canonical_resolved_config_bytes(config: Any) -> bytes:
    """Return canonical bytes for the public resolved-config identity.

    The identity is deliberately separate from checkpoint manifest hashes:
    checkpoint compatibility retains its historical exclusions, while this
    public contract has a closed, documented run-artifact exclusion list.
    """

    plain = to_plain_container(config)
    if not isinstance(plain, Mapping):
        raise TypeError("resolved configuration identity requires a mapping")
    normalized = copy.deepcopy(plain)
    for path in RESOLVED_CONFIG_IDENTITY_EXCLUSIONS:
        _delete_pattern(normalized, path)
    safe = _json_safe(normalized)
    return json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def resolved_config_identity(config: Any) -> dict[str, str]:
    """Return the versioned public identity for one resolved configuration."""

    return {
        "schema": RESOLVED_CONFIG_IDENTITY_SCHEMA,
        "sha256": hashlib.sha256(canonical_resolved_config_bytes(config)).hexdigest(),
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


def _delete_pattern(container: dict[str, Any], path: tuple[str, ...]) -> None:
    """Delete one explicitly declared path, allowing list-index wildcards."""

    if not path:
        return
    key = path[0]
    if key == "*":
        if isinstance(container, list):
            for item in container:
                if isinstance(item, (dict, list)):
                    _delete_pattern(item, path[1:])
        return
    if not isinstance(container, dict) or key not in container:
        return
    if len(path) == 1:
        container.pop(key, None)
        return
    child = container[key]
    if isinstance(child, (dict, list)):
        _delete_pattern(child, path[1:])
