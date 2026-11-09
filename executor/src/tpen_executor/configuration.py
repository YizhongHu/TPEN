"""Hydra composition with an explicit unresolved future-process boundary.

Program markers are captured with ``resolve=False`` before any target is
instantiated.  An absolute key interpolation inside a program subtree is
allowed when its key is absent from the submitter's composed configuration
(for example, ``${remote_only.value}``) and is therefore left literal for the
owner process.  It is also allowed when the referenced key is inside the same
program subtree.  If the key exists on the submitter but is outside that
subtree, composition fails: re-rooting the self-contained program reference
would otherwise silently sever the reference.  Resolver interpolations such
as ``${oc.env:NAME}`` are not key references and remain literal.

The shared marker path is the dot-and-index path from the composed root to the
marker (for example, ``program.payload.workers[0]``).  It is carried by the
process-local handle, used as the materialization mount path, and defines the
subtree boundary for this compose-time audit.  Relative interpolations are
legal only when they remain inside that path.  Key references nested inside
resolver arguments are a known limitation of this audit; they are deferred to
materialization, where an invalid reference fails loudly, and are followed up
by P5a.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from hydra import compose, initialize_config_dir, initialize_config_module
from omegaconf import DictConfig, OmegaConf
from omegaconf.grammar_parser import parse as parse_interpolation

from ._codec import digest
from .contracts import ProgramSpecRef
from ._records import frozen_record
from .programs import AllocationProgram, WorkerProgram, payload_entries, validate_marker_shape


_PROGRAM_TYPES = {
    "tpen_executor.programs.AllocationProgram": AllocationProgram,
    "tpen_executor.programs.WorkerProgram": WorkerProgram,
}


def _raw(value: Any) -> Any:
    """Copy a config without resolving interpolations or constructing targets."""

    return OmegaConf.to_container(value, resolve=False, enum_to_str=True)


def _target_name(value: Mapping[str, Any]) -> str | None:
    """Return the fully qualified target name for a raw Hydra mapping."""

    target = value.get("_target_")
    return target if isinstance(target, str) else None


def _path_parts(path: str) -> list[str | int]:
    """Split a shared marker path into mapping keys and sequence indices."""

    parts: list[str | int] = []
    for segment in path.split(".") if path else ():
        for token in re.findall(r"[^\[\]]+|\[\d+\]", segment):
            parts.append(int(token[1:-1]) if token.startswith("[") else token)
    return parts


def _interpolation_refs(value: str) -> list[str]:
    """Return direct interpolation references using OmegaConf's grammar."""

    references: list[str] = []
    tree = parse_interpolation(value)
    text = tree.text()
    for interpolation in text.interpolation():
        node = interpolation.interpolationNode()
        if node is None:
            # Resolver interpolations (for example ``oc.env``) are not key
            # references and are intentionally left for the owner process.
            continue
        references.append(node.getText()[2:-1])
    return references


def _key_exists(raw_config: Any, key: str) -> bool:
    """Check raw submitter structure without resolving any intermediate node."""

    current = raw_config
    for part in _path_parts(key):
        if isinstance(part, int):
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes, bytearray)):
                return False
            if part >= len(current):
                return False
            current = current[part]
        else:
            if not isinstance(current, Mapping) or part not in current:
                return False
            current = current[part]
    return True


def _is_inside(reference: str, subtree: str) -> bool:
    """Return whether an absolute OmegaConf key is within a marker subtree."""

    reference_parts = _path_parts(reference)
    subtree_parts = _path_parts(subtree)
    return reference_parts[: len(subtree_parts)] == subtree_parts


def _relative_target(path: str, reference: str) -> list[str | int]:
    """Resolve a relative reference's path lexically without resolving values."""

    dots = len(reference) - len(reference.lstrip("."))
    current_parts = _path_parts(path)
    if dots > len(current_parts):
        return []
    return current_parts[:-dots] + _path_parts(reference[dots:])


def _validate_marker(value: Any, path: str, raw_config: Any, subtree: str | None = None) -> None:
    """Reject submitter-known absolute references that would be severed."""

    if subtree is None:
        subtree = path
    if isinstance(value, Mapping):
        target = _target_name(value)
        if target in _PROGRAM_TYPES and path != "" and path != subtree:
            # A nested marker has its own self-contained reference boundary;
            # it is validated when the recursive capture reaches that marker.
            return
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _validate_marker(child, child_path, raw_config, subtree)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_marker(child, f"{path}[{index}]", raw_config, subtree)
        return
    if not isinstance(value, str) or "${" not in value:
        return
    for reference in _interpolation_refs(value):
        if reference.startswith("."):
            relative_target = _relative_target(path, reference)
            subtree_parts = _path_parts(subtree)
            if relative_target[: len(subtree_parts)] == subtree_parts:
                continue
            raise ValueError(
                f"program marker {path!r} has relative interpolation {value!r} "
                f"that climbs outside its subtree {subtree!r}"
            )
        if _is_inside(reference, subtree) or not _key_exists(raw_config, reference):
            continue
        raise ValueError(
            f"program marker {path!r} has absolute interpolation {value!r} "
            f"to submitter key {reference!r} outside its subtree; inline the "
            "value or provide it through the materialize environment"
        )


def _capture(value: Any, path: str, found: dict[str, Any], raw_config: Any) -> None:
    """Capture marked program subtrees recursively before Hydra sees them."""

    if isinstance(value, Mapping):
        target = _target_name(value)
        marked = value.get("_recursive_") is False
        if target in _PROGRAM_TYPES and not marked:
            raise ValueError(f"program target {target!r} at {path!r} must set _recursive_: false")
        if marked and target in _PROGRAM_TYPES:
            if not path:
                raise ValueError("a program marker cannot be the configuration root")
            validate_marker_shape(value, path)
            payload_entries(value, path)
            _validate_marker(value, path, raw_config)
            ref = ProgramSpecRef(value, digest(value))
            found[path] = _PROGRAM_TYPES[target](ref, path)
            # Walk the opaque payload too: nested WorkerProgram descriptions
            # must receive their own handles without being instantiated.
            for key, child in value.items():
                if key not in {"_target_", "_recursive_"}:
                    child_path = f"{path}.{key}" if path else str(key)
                    _capture(child, child_path, found, raw_config)
            return
        for key, child in value.items():
            _capture(child, f"{path}.{key}" if path else str(key), found, raw_config)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _capture(child, f"{path}[{index}]", found, raw_config)


def _stage(config: DictConfig) -> StagedConfig:
    """Capture one already-composed config through the shared capture path."""

    if not isinstance(config, DictConfig):
        raise TypeError("StagedHydra.stage expects an OmegaConf DictConfig")
    raw = _raw(config)
    found: dict[str, Any] = {}
    _capture(raw, "", found, raw)
    return StagedConfig(config=config, programs=found)


@frozen_record
class StagedConfig:
    """Composed configuration plus opaque future-process program handles."""

    config: DictConfig
    programs: Mapping[str, AllocationProgram | WorkerProgram]

    @property
    def program(self) -> AllocationProgram | WorkerProgram | None:
        """Return the root ``program`` handle when one is present."""

        return self.programs.get("program")


class StagedHydra:
    """Compose Hydra configuration while preserving marked program subtrees."""

    def __init__(self, config_module: str = "tpen_executor.conf", config_dir: str | None = None) -> None:
        if config_module and config_dir is not None:
            raise ValueError("config_module and config_dir are mutually exclusive")
        if not config_module and config_dir is None:
            raise ValueError("StagedHydra requires either config_module or config_dir")
        self.config_module = config_module
        self.config_dir = config_dir

    def compose_config(self, config_name: str = "executor", overrides: Sequence[str] = ()) -> StagedConfig:
        """Compose one config and capture future-process programs before resolution."""

        if self.config_dir is not None:
            with initialize_config_dir(version_base=None, config_dir=self.config_dir):
                config = compose(config_name=config_name, overrides=list(overrides))
        else:
            with initialize_config_module(version_base=None, config_module=self.config_module):
                config = compose(config_name=config_name, overrides=list(overrides))
        return _stage(config)

    @staticmethod
    def stage(config: DictConfig) -> StagedConfig:
        """Capture a config already composed by the caller's Hydra app."""

        return _stage(config)

    @staticmethod
    def stage_source(source: str) -> StagedConfig:
        """Parse and stage a YAML source without resolving program markers.

        Parameters
        ----------
        source : str
            YAML document containing a mapping or one top-level program marker.

        Returns
        -------
        StagedConfig
            Parsed configuration and unresolved program handles.

        Raises
        ------
        TypeError, ValueError
            If the source is not a mapping or cannot be staged.
        """

        loaded = OmegaConf.create(source)
        if not isinstance(loaded, DictConfig):
            raise TypeError("configuration source must be a YAML mapping")
        raw = _raw(loaded)
        if isinstance(raw, Mapping) and isinstance(raw.get("_target_"), str):
            loaded = OmegaConf.create({"worker": raw})
        return _stage(loaded)

    @classmethod
    def compose(cls, config_name: str = "executor", overrides: Sequence[str] = (), *, config_module: str = "tpen_executor.conf", config_dir: str | None = None) -> StagedConfig:
        """Convenience class API for composing a staged configuration."""

        return cls(config_module=config_module, config_dir=config_dir).compose_config(config_name, overrides)


__all__ = ["StagedConfig", "StagedHydra"]
