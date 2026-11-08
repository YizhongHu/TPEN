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
import re
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.nodes import AnyNode

from ._codec import digest, thaw
from .contracts import ProgramSpecRef


_PROGRAM_TARGETS = {
    "tpen_executor.programs.AllocationProgram",
    "tpen_executor.programs.WorkerProgram",
}
_HYDRA_RESERVED_KEYS = {"_target_", "_recursive_", "_convert_", "_partial_", "_args_"}
_PROGRAM_MOUNT_KEY = "__program__"


def _payload_entry(raw: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    """Select the concrete target nested inside a program marker."""

    candidates = [
        (key, value)
        for key, value in raw.items()
        if key not in {"_target_", "_recursive_"}
        and isinstance(value, Mapping)
        and value.get("_target_") not in _PROGRAM_TARGETS
    ]
    if len(candidates) != 1:
        raise ValueError("a program marker must contain exactly one concrete payload")
    return candidates[0]


def _path_parts(path: str) -> list[str | int]:
    """Split a marker path into mapping keys and sequence indices."""

    parts: list[str | int] = []
    for segment in path.split(".") if path else ():
        for token in re.findall(r"[^\[\]]+|\[\d+\]", segment):
            parts.append(int(token[1:-1]) if token.startswith("[") else token)
    return parts


def _mount_marker(root: dict[str, Any], path: str, marker: Mapping[str, Any]) -> None:
    """Mount a marker at its original shared path beside the materialize env."""

    parts = _path_parts(path)
    if not parts or not isinstance(parts[0], str):
        raise ValueError(f"program marker path must begin with a mapping key: {path!r}")
    current: Any = root
    for index, part in enumerate(parts[:-1]):
        next_part = parts[index + 1]
        if isinstance(part, int):
            if not isinstance(current, list):
                raise ValueError(f"program marker path has a non-list parent: {path!r}")
            while len(current) <= part:
                current.append([] if isinstance(next_part, int) else {})
            current = current[part]
            continue
        if not isinstance(current, dict):
            raise ValueError(f"program marker path has a non-mapping parent: {path!r}")
        if part not in current:
            current[part] = [] if isinstance(next_part, int) else {}
        current = current[part]
    final = parts[-1]
    if isinstance(final, int):
        if not isinstance(current, list):
            raise ValueError(f"program marker path has a non-list parent: {path!r}")
        while len(current) <= final:
            current.append(None)
        current[final] = marker
    else:
        if not isinstance(current, dict):
            raise ValueError(f"program marker path has a non-mapping parent: {path!r}")
        current[final] = marker


def _at_path(root: Any, path: str) -> Any:
    """Return the OmegaConf node mounted at a shared marker path."""

    current = root
    for part in _path_parts(path):
        current = current[part]
    return current


def _child_path(path: str, key: str | int) -> str:
    """Extend a shared marker path with one mapping key or sequence index."""

    return f"{path}[{key}]" if isinstance(key, int) else f"{path}.{key}"


def _program_handle(raw: Mapping[str, Any], path: str) -> AllocationProgram | WorkerProgram:
    """Build the handle represented by one nested program marker."""

    target = raw.get("_target_")
    handle_type = AllocationProgram if target == "tpen_executor.programs.AllocationProgram" else WorkerProgram
    return handle_type(ProgramSpecRef(raw, digest(raw)), path)


def _install_handles(raw: Any, node: DictConfig | ListConfig, path: str) -> None:
    """Replace nested program marker nodes with identity-preserving handles."""

    if isinstance(raw, Mapping) and isinstance(node, DictConfig):
        for key, child in raw.items():
            child_node = node._content[key]
            child_path = _child_path(path, key)
            if isinstance(child, Mapping) and child.get("_target_") in _PROGRAM_TARGETS:
                handle = _program_handle(child, child_path)
                # AnyNode is pinned to OmegaConf 2.3.0; it preserves the
                # opaque handle object without asking Hydra to instantiate it.
                node._content[key] = AnyNode(
                    handle,
                    key=key,
                    parent=node,
                    flags={"allow_objects": True},
                )
                continue
            if isinstance(child, (Mapping, Sequence)) and not isinstance(child, (str, bytes, bytearray)) and isinstance(child_node, (DictConfig, ListConfig)):
                _install_handles(child, child_node, child_path)
        return
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)) and isinstance(node, ListConfig):
        for index, child in enumerate(raw):
            child_path = _child_path(path, index)
            child_node = node._content[index]
            if isinstance(child, Mapping) and child.get("_target_") in _PROGRAM_TARGETS:
                node._content[index] = AnyNode(
                    _program_handle(child, child_path),
                    key=index,
                    parent=node,
                    flags={"allow_objects": True},
                )
            elif isinstance(child, (Mapping, Sequence)) and not isinstance(child, (str, bytes, bytearray)) and isinstance(child_node, (DictConfig, ListConfig)):
                _install_handles(child, child_node, child_path)


def _materialize(ref: ProgramSpecRef, path: str, env: Mapping[str, Any] | None) -> Any:
    """Resolve and instantiate one captured specification in the owner process."""

    raw = thaw(ref.config_ref)
    if not isinstance(raw, Mapping) or raw.get("_target_") not in _PROGRAM_TARGETS:
        raise ValueError("program materialization requires a recognized program marker")
    payload_key, _ = _payload_entry(raw)
    environment = dict(env or {})
    path_parts = _path_parts(path)
    first_segment = path_parts[0] if path_parts else "program"
    collisions = _HYDRA_RESERVED_KEYS.intersection(environment) | {_PROGRAM_MOUNT_KEY}.intersection(environment)
    if isinstance(first_segment, str) and first_segment in environment:
        collisions.add(first_segment)
    if collisions:
        raise ValueError(f"materialize environment contains reserved keys: {sorted(collisions)!r}")
    # Keep env beside the marker tree and instantiate the concrete payload in
    # place.  The marker's original path supplies the same interpolation root
    # that the submitter captured and prevents env values from becoming target
    # kwargs or replacing Hydra controls.
    root_data = dict(environment)
    _mount_marker(root_data, path or "program", raw)
    root = OmegaConf.create(root_data, flags={"allow_objects": True})
    marker_node = _at_path(root, path or "program")
    _install_handles(raw, marker_node, path or "program")
    return instantiate(marker_node[payload_key])


@dataclass(frozen=True, slots=True)
class WorkerProgram:
    """Thin holder for an unresolved worker-process program specification."""

    spec: ProgramSpecRef
    path: str = "program"

    def materialize(self, env: Mapping[str, Any] | None = None) -> Any:
        """Resolve and construct the worker program in its owning process."""

        return _materialize(self.spec, self.path, env)


@dataclass(frozen=True, slots=True)
class AllocationProgram:
    """Thin holder for an unresolved allocation-process program specification."""

    spec: ProgramSpecRef
    path: str = "program"

    def materialize(self, env: Mapping[str, Any] | None = None) -> Any:
        """Resolve and construct the allocation program in its owning process."""

        return _materialize(self.spec, self.path, env)


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
