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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hydra import compose, initialize_config_dir, initialize_config_module
from omegaconf import DictConfig, OmegaConf
from omegaconf._impl import select_node
from omegaconf.grammar_parser import parse as parse_interpolation

from ._codec import digest
from .contracts import ProgramSpecRef
from .programs import AllocationProgram, WorkerProgram


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


def _interpolation_keys(value: str) -> list[str]:
    """Return absolute key references using OmegaConf's interpolation grammar."""

    keys: list[str] = []
    tree = parse_interpolation(value)
    text = tree.text()
    for interpolation in text.interpolation():
        node = interpolation.interpolationNode()
        if node is None:
            # Resolver interpolations (for example ``oc.env``) are not key
            # references and are intentionally left for the owner process.
            continue
        raw = node.getText()[2:-1]
        if raw.startswith("."):
            # Relative references are evaluated by OmegaConf in the owning
            # subtree and are not submitter-root absolute references.
            continue
        keys.append(raw)
    return keys


def _key_exists(config: DictConfig, key: str) -> bool:
    """Check submitter-root key existence without resolving its value."""

    return select_node(
        config,
        key,
        absolute_key=True,
        throw_on_resolution_failure=False,
        throw_on_missing=False,
    ) is not None


def _is_inside(reference: str, subtree: str) -> bool:
    """Return whether an absolute OmegaConf key is within a marker subtree."""

    return reference == subtree or reference.startswith(f"{subtree}.") or reference.startswith(f"{subtree}[")


def _validate_marker(value: Any, path: str, config: DictConfig, subtree: str | None = None) -> None:
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
            _validate_marker(child, child_path, config, subtree)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _validate_marker(child, path, config, subtree)
        return
    if not isinstance(value, str) or "${" not in value:
        return
    for reference in _interpolation_keys(value):
        if _is_inside(reference, subtree) or not _key_exists(config, reference):
            continue
        raise ValueError(
            f"program marker {path!r} has absolute interpolation {value!r} "
            f"to submitter key {reference!r} outside its subtree; inline the "
            "value or provide it through the materialize environment"
        )


def _capture(value: Any, path: str, found: dict[str, Any], config: DictConfig) -> None:
    """Capture marked program subtrees recursively before Hydra sees them."""

    if isinstance(value, Mapping):
        target = _target_name(value)
        marked = value.get("_recursive_") is False
        if target in _PROGRAM_TYPES and not marked:
            raise ValueError(f"program target {target!r} at {path!r} must set _recursive_: false")
        if marked and target in _PROGRAM_TYPES:
            if not path:
                raise ValueError("a program marker cannot be the configuration root")
            _validate_marker(value, path, config)
            ref = ProgramSpecRef(value, digest(value))
            found[path] = _PROGRAM_TYPES[target](ref)
            # Walk the opaque payload too: nested WorkerProgram descriptions
            # must receive their own handles without being instantiated.
            for key, child in value.items():
                if key not in {"_target_", "_recursive_"}:
                    child_path = f"{path}.{key}" if path else str(key)
                    _capture(child, child_path, found, config)
            return
        for key, child in value.items():
            _capture(child, f"{path}.{key}" if path else str(key), found, config)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _capture(child, f"{path}[{index}]", found, config)


def _stage(config: DictConfig) -> StagedConfig:
    """Capture one already-composed config through the shared capture path."""

    if not isinstance(config, DictConfig):
        raise TypeError("StagedHydra.stage expects an OmegaConf DictConfig")
    raw = _raw(config)
    found: dict[str, Any] = {}
    _capture(raw, "", found, config)
    return StagedConfig(config=config, programs=found)


@dataclass(frozen=True, slots=True)
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

    @classmethod
    def compose(cls, config_name: str = "executor", overrides: Sequence[str] = (), *, config_module: str = "tpen_executor.conf", config_dir: str | None = None) -> StagedConfig:
        """Convenience class API for composing a staged configuration."""

        return cls(config_module=config_module, config_dir=config_dir).compose_config(config_name, overrides)


__all__ = ["StagedConfig", "StagedHydra"]
