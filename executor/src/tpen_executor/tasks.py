"""Configured logical executor tasks."""

from __future__ import annotations

import os
from typing import Any, ClassVar, Sequence

from ._codec import decode_json, fields, sha256_text
from .runners import TPENCommand
from ._records import frozen_record


@frozen_record
class ConfiguredRun:
    """Hydra-constructible command configuration for one logical task.

    Parameters
    ----------
    id : str
        Stable logical task identifier.
    config : str
        Absolute Hydra configuration path.
    overrides : sequence of str, optional
        Dot-list overrides passed unchanged to the child process.
    """

    schema: ClassVar[str] = "tpen-executor/configured-run-v1"
    id: str
    config: str
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id.strip():
            raise TypeError("id must be a non-empty string")
        if type(self.config) is not str or not os.path.isabs(self.config):
            raise ValueError("config must be an absolute path")
        if isinstance(self.overrides, (str, bytes, bytearray)) or not isinstance(self.overrides, Sequence) or any(type(item) is not str for item in self.overrides):
            raise TypeError("overrides must be a sequence of strings")
        object.__setattr__(self, "overrides", tuple(self.overrides))

    def command(self) -> TPENCommand:
        """Return the executor command represented by this task."""
        return TPENCommand(self.config, self.overrides)


@frozen_record
class ConfiguredRunTable:
    """Strict sidecar table binding configured runs to one ready group.

    Parameters
    ----------
    group_digest : str
        Order-sensitive :class:`~tpen_executor.contracts.ReadyGroup` identity.
    runs : sequence of ConfiguredRun
        Runs in the same order as the producer group's members.
    """

    schema: ClassVar[str] = "tpen-executor/configured-runs-v1"
    group_digest: str
    runs: tuple[ConfiguredRun, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_digest", sha256_text(self.group_digest, "group_digest"))
        if isinstance(self.runs, (str, bytes, bytearray)) or not isinstance(self.runs, Sequence):
            raise TypeError("runs must be a sequence")
        runs = tuple(self.runs)
        if any(not isinstance(run, ConfiguredRun) for run in runs):
            raise TypeError("runs must contain ConfiguredRun records")
        if len({run.id for run in runs}) != len(runs):
            raise ValueError("runs must contain unique ids")
        if not runs:
            raise ValueError("runs must not be empty")
        object.__setattr__(self, "runs", runs)

    def to_dict(self) -> dict[str, Any]:
        """Return the exact sidecar wire representation."""

        return {
            "schema": self.schema,
            "group_digest": self.group_digest,
            "runs": [
                {"id": run.id, "config": run.config, "overrides": list(run.overrides)}
                for run in self.runs
            ],
        }

    def encode(self) -> bytes:
        """Encode the sidecar using canonical JSON."""

        from ._codec import canonical_bytes

        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "ConfiguredRunTable":
        """Decode a strict configured-run sidecar mapping."""

        fields(data, {"schema", "group_digest", "runs"}, cls.schema, "ConfiguredRunTable")
        if type(data["group_digest"]) is not str:
            raise TypeError("group_digest must be a string")
        if isinstance(data["runs"], (str, bytes, bytearray)) or not isinstance(data["runs"], Sequence):
            raise TypeError("runs must be a sequence")
        decoded: list[ConfiguredRun] = []
        for item in data["runs"]:
            if not isinstance(item, dict):
                raise TypeError("ConfiguredRun must be a mapping")
            expected = {"id", "config", "overrides"}
            if set(item) != expected:
                missing, extra = expected - set(item), set(item) - expected
                raise ValueError(f"ConfiguredRun keys differ (missing={sorted(missing)!r}, extra={sorted(extra)!r})")
            if type(item["id"]) is not str or type(item["config"]) is not str:
                raise TypeError("ConfiguredRun id and config must be strings")
            overrides = item["overrides"]
            if isinstance(overrides, (str, bytes, bytearray)) or not isinstance(overrides, Sequence):
                raise TypeError("ConfiguredRun overrides must be a sequence")
            if any(type(value) is not str for value in overrides):
                raise TypeError("ConfiguredRun overrides must contain strings")
            decoded.append(ConfiguredRun(item["id"], item["config"], tuple(overrides)))
        return cls(data["group_digest"], tuple(decoded))

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "ConfiguredRunTable":
        """Decode canonical JSON bytes or a decoded mapping."""

        return cls.from_mapping(decode_json(data))


RunTable = ConfiguredRunTable
ConfiguredRuns = ConfiguredRunTable


def encode_run_table(table: ConfiguredRunTable) -> bytes:
    """Encode one configured-run sidecar."""

    if not isinstance(table, ConfiguredRunTable):
        raise TypeError("table must be ConfiguredRunTable")
    return table.encode()


def decode_run_table(data: bytes | str | dict[str, Any]) -> ConfiguredRunTable:
    """Decode one configured-run sidecar."""

    return ConfiguredRunTable.decode(data)


__all__ = [
    "ConfiguredRun",
    "ConfiguredRunTable",
    "ConfiguredRuns",
    "RunTable",
    "decode_run_table",
    "encode_run_table",
]
