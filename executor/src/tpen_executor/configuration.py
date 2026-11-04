"""Hydra composition with an explicit unresolved future-process boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hydra import compose, initialize_config_dir, initialize_config_module
from omegaconf import DictConfig, ListConfig, OmegaConf

from ._codec import digest
from .contracts import ProgramSpecRef
from .programs import AllocationProgram, WorkerProgram


_PROGRAM_TYPES = {"AllocationProgram": AllocationProgram, "WorkerProgram": WorkerProgram}


def _raw(value: Any) -> Any:
    """Copy a config without resolving interpolations or constructing targets."""

    return OmegaConf.to_container(value, resolve=False, enum_to_str=True)


def _target_name(value: Mapping[str, Any]) -> str | None:
    """Return the short target class name for a raw Hydra mapping."""

    target = value.get("_target_")
    return target.rsplit(".", maxsplit=1)[-1] if isinstance(target, str) else None


def _capture(value: Any, path: str, found: dict[str, Any]) -> Any:
    """Capture marked program subtrees recursively before Hydra sees them."""

    if isinstance(value, Mapping):
        kind = _target_name(value)
        if kind in _PROGRAM_TYPES:
            ref = ProgramSpecRef(value, digest(value))
            found[path] = _PROGRAM_TYPES[kind](ref)
            # Walk the opaque payload too: nested WorkerProgram descriptions
            # must receive their own handles without being instantiated.
            for key, child in value.items():
                if key not in {"_target_", "_recursive_"}:
                    child_path = f"{path}.{key}" if path else str(key)
                    _capture(child, child_path, found)
            return value
        return {key: _capture(child, f"{path}.{key}" if path else str(key), found) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_capture(child, f"{path}[{index}]", found) for index, child in enumerate(value)]
    return value


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
        raw = _raw(config)
        found: dict[str, Any] = {}
        _capture(raw, "", found)
        return StagedConfig(config=config, programs=found)

    @classmethod
    def compose(cls, config_name: str = "executor", overrides: Sequence[str] = (), *, config_module: str = "tpen_executor.conf", config_dir: str | None = None) -> StagedConfig:
        """Convenience class API for composing a staged configuration."""

        return cls(config_module=config_module, config_dir=config_dir).compose_config(config_name, overrides)


__all__ = ["StagedConfig", "StagedHydra"]
