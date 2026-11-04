"""Process-local materialization of staged executor programs.

Each program marker currently has exactly one concrete payload mapping; any
additional marker-level fields are retained in the immutable digest but are
not constructor arguments.  Nested program markers are substituted with
opaque ``WorkerProgram``/``AllocationProgram`` handles before Hydra
instantiation.  Multi-payload program layouts are introduced by P5a.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.nodes import AnyNode

from ._codec import thaw
from .contracts import ProgramSpecRef


_PROGRAM_TARGETS = {
    "tpen_executor.programs.AllocationProgram",
    "tpen_executor.programs.WorkerProgram",
}
_HYDRA_RESERVED_KEYS = {"_target_", "_recursive_", "_convert_", "_partial_", "_args_"}
_PROGRAM_MOUNT_KEY = "__program__"


def _payload(raw: Mapping[str, Any]) -> Any:
    """Select the concrete target nested inside a program marker."""

    candidates = [
        value
        for key, value in raw.items()
        if key not in {"_target_", "_recursive_"}
        and isinstance(value, Mapping)
        and value.get("_target_") not in _PROGRAM_TARGETS
    ]
    if len(candidates) != 1:
        raise ValueError("a program marker must contain exactly one concrete payload")
    return candidates[0]


def _program_handle(raw: Mapping[str, Any]) -> AllocationProgram | WorkerProgram:
    """Build the handle represented by one nested program marker."""

    target = raw.get("_target_")
    handle_type = AllocationProgram if target == "tpen_executor.programs.AllocationProgram" else WorkerProgram
    return handle_type(ProgramSpecRef(raw, digest(raw)))


def _install_handles(raw: Any, node: DictConfig | ListConfig) -> None:
    """Replace nested program marker nodes with identity-preserving handles."""

    if isinstance(raw, Mapping) and isinstance(node, DictConfig):
        for key, child in raw.items():
            if not isinstance(child, Mapping):
                continue
            if child.get("_target_") in _PROGRAM_TARGETS:
                handle = _program_handle(child)
                node._content[key] = AnyNode(
                    handle,
                    key=key,
                    parent=node,
                    flags={"allow_objects": True},
                )
                continue
            child_node = node._content[key]
            if isinstance(child_node, (DictConfig, ListConfig)):
                _install_handles(child, child_node)
        return
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)) and isinstance(node, ListConfig):
        for index, child in enumerate(raw):
            if isinstance(child, Mapping) and child.get("_target_") in _PROGRAM_TARGETS:
                node._content[index] = AnyNode(
                    _program_handle(child),
                    key=index,
                    parent=node,
                    flags={"allow_objects": True},
                )
            elif isinstance(child, (Mapping, Sequence)) and isinstance(node._content[index], (DictConfig, ListConfig)):
                _install_handles(child, node._content[index])


def _materialize(ref: ProgramSpecRef, env: Mapping[str, Any] | None) -> Any:
    """Resolve and instantiate one captured specification in the owner process."""

    raw = thaw(ref.config_ref)
    marker = raw
    if raw.get("_target_") in _PROGRAM_TARGETS:
        raw = _payload(raw)
    environment = dict(env or {})
    collisions = _HYDRA_RESERVED_KEYS.intersection(environment) | {_PROGRAM_MOUNT_KEY}.intersection(environment)
    if collisions:
        raise ValueError(f"materialize environment contains reserved keys: {sorted(collisions)!r}")
    # Keep the process-local environment beside, rather than inside, the
    # target payload.  This preserves the interpolation parent while ensuring
    # env values can never become constructor kwargs or replace _target_.
    # ``program`` is the stable root name used by absolute references inside
    # the packaged marker.  The reserved mount remains the object actually
    # passed to Hydra, while this alias keeps marker-internal references
    # resolvable after the concrete payload is selected.
    root_data = {**environment, _PROGRAM_MOUNT_KEY: raw}
    if "program" not in environment:
        root_data["program"] = marker
    root = OmegaConf.create(root_data, flags={"allow_objects": True})
    _install_handles(raw, root[_PROGRAM_MOUNT_KEY])
    return instantiate(root[_PROGRAM_MOUNT_KEY])


@dataclass(frozen=True, slots=True)
class WorkerProgram:
    """Thin holder for an unresolved worker-process program specification."""

    spec: ProgramSpecRef

    def materialize(self, env: Mapping[str, Any] | None = None) -> Any:
        """Resolve and construct the worker program in its owning process."""

        return _materialize(self.spec, env)


@dataclass(frozen=True, slots=True)
class AllocationProgram:
    """Thin holder for an unresolved allocation-process program specification."""

    spec: ProgramSpecRef

    def materialize(self, env: Mapping[str, Any] | None = None) -> Any:
        """Resolve and construct the allocation program in its owning process."""

        return _materialize(self.spec, env)


class DummyAllocation:
    """Minimal packaged target used by the configuration composition example."""

    def __init__(self, worker: Any = None, **kwargs: Any) -> None:
        self.worker = worker
        self.options = kwargs


class DummyWorker:
    """Minimal packaged worker target used by the configuration example."""

    def __init__(self, command: str = "worker", **kwargs: Any) -> None:
        self.command = command
        self.options = kwargs


__all__ = ["AllocationProgram", "DummyAllocation", "DummyWorker", "WorkerProgram"]
