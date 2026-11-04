"""Immutable descriptions of already-prepared owner-process environments.

This module deliberately describes environments rather than provisioning them.
The submitter may validate the shape of a path, but it must not inspect or
modify the path: preparation belongs to the process that owns the environment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from ._codec import canonical_bytes, decode_json, fields, thaw


def _absolute_path(value: Any, name: str) -> str:
    """Validate an absolute path without checking the filesystem."""

    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")
    if not os.path.isabs(value):
        raise ValueError(f"{name} must be an absolute path")
    return value


def _string_mapping(value: Any, name: str) -> MappingProxyType:
    """Copy a string-to-string mapping into an immutable mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen: dict[str, str] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise TypeError(f"{name} keys must be non-empty strings")
        if type(item) is not str:
            raise TypeError(f"{name} values must be strings")
        frozen[key] = item
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class SourceScript:
    """Description of a setup script owned by the target process.

    Parameters
    ----------
    path : str
        Absolute path to the script.  The path is not opened or otherwise
        checked by this descriptor.
    """

    schema: ClassVar[str] = "tpen-executor/source-script-v1"
    path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _absolute_path(self.path, "path"))

    def to_dict(self) -> dict[str, str]:
        """Return the strict wire representation."""

        return {"schema": self.schema, "path": self.path}

    def encode(self) -> bytes:
        """Encode this descriptor as canonical JSON bytes."""

        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "SourceScript":
        """Decode one strict mapping."""

        fields(data, {"schema", "path"}, cls.schema, "SourceScript")
        return cls(data["path"])

    @classmethod
    def decode(cls, data: bytes | str | Mapping[str, Any]) -> "SourceScript":
        """Decode canonical JSON or an already-decoded mapping."""

        return cls.from_mapping(decode_json(data))


@dataclass(frozen=True, slots=True)
class PreparedPythonEnvironment:
    """Description of a prepared Python environment.

    Parameters
    ----------
    python : str
        Absolute path to the owner-process Python interpreter.
    setup : SourceScript or None
        Optional setup script to be consumed by the owner process.
    env : mapping[str, str]
        Environment additions.  The mapping is copied and frozen.
    """

    schema: ClassVar[str] = "tpen-executor/prepared-python-environment-v1"
    python: str
    setup: SourceScript | None = None
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "python", _absolute_path(self.python, "python"))
        if self.setup is not None and not isinstance(self.setup, SourceScript):
            raise TypeError("setup must be SourceScript or None")
        object.__setattr__(self, "env", _string_mapping(self.env, "env"))

    def to_dict(self) -> dict[str, Any]:
        """Return the strict wire representation."""

        return {
            "schema": self.schema,
            "python": self.python,
            "setup": None if self.setup is None else self.setup.to_dict(),
            "env": thaw(self.env),
        }

    def encode(self) -> bytes:
        """Encode this descriptor as canonical JSON bytes."""

        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "PreparedPythonEnvironment":
        """Decode one strict mapping."""

        fields(data, {"schema", "python", "setup", "env"}, cls.schema, "PreparedPythonEnvironment")
        setup = None if data["setup"] is None else SourceScript.from_mapping(data["setup"])
        return cls(data["python"], setup, data["env"])

    @classmethod
    def decode(cls, data: bytes | str | Mapping[str, Any]) -> "PreparedPythonEnvironment":
        """Decode canonical JSON or an already-decoded mapping."""

        return cls.from_mapping(decode_json(data))


__all__ = ["PreparedPythonEnvironment", "SourceScript"]
