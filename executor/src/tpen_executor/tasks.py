"""Configured logical executor tasks."""

from __future__ import annotations

import os
from typing import Any, ClassVar, Sequence

from ._codec import decode_json, digest, sha256_text
from .configured_program import AttemptDelta, ConfiguredProgram
from .runners import ConfiguredProgramCommand, TPENCommand
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

    schema: ClassVar[str] = "tpen-executor/configured-run-v2"
    id: str
    config: str
    overrides: tuple[str, ...] = ()
    program: ConfiguredProgram | None = None

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id.strip():
            raise TypeError("id must be a non-empty string")
        if type(self.config) is not str:
            raise TypeError("config must be a string")
        if self.program is None and not os.path.isabs(self.config):
            raise ValueError("config must be an absolute path")
        if isinstance(self.overrides, (str, bytes, bytearray)) or not isinstance(self.overrides, Sequence) or any(type(item) is not str for item in self.overrides):
            raise TypeError("overrides must be a sequence of strings")
        object.__setattr__(self, "overrides", tuple(self.overrides))
        if self.program is not None and type(self.program) is not ConfiguredProgram:
            raise TypeError("program must be ConfiguredProgram or None")
        if self.program is not None and (self.config or self.overrides):
            raise ValueError("configured program cannot carry an unbound config or overrides until C2.2")

    def command(self) -> TPENCommand | ConfiguredProgramCommand:
        """Return the executor command represented by this task."""
        if self.program is not None:
            return ConfiguredProgramCommand(self.program)
        return TPENCommand(self.config, self.overrides)

    @property
    def binding_digest(self) -> str:
        """Return the digest of the configured payload before placement."""

        if self.program is not None:
            return self.program.program_digest
        return digest({"config": self.config, "overrides": list(self.overrides)})

    def attempt_delta(
        self,
        output_dir: str,
        *,
        root: str | None = None,
        group_digest: str,
        task_id: str,
        attempt_id: str,
    ) -> AttemptDelta:
        """Build one placement-bound delta for a configured program."""

        if self.program is None:
            raise ValueError("legacy TPEN runs do not carry an attempt delta")
        if self.id != task_id:
            raise ValueError("configured run does not belong to the attempt task")
        if type(root) is not str or not os.path.isabs(root) or os.path.normpath(root) != root:
            raise ValueError("configured attempt binding requires a normalized absolute root")
        expected_output = os.path.join(root, "groups", group_digest, task_id, attempt_id)
        if output_dir != os.path.normpath(output_dir) or output_dir != expected_output:
            raise ValueError("configured attempt output does not bind the placement fence")
        return AttemptDelta(
            os.path.join(output_dir, "core"), self.program.load,
            group_digest, task_id, attempt_id,
        )

    def attempt_binding_digest(
        self,
        output_dir: str,
        *,
        root: str | None = None,
        group_digest: str,
        task_id: str,
        attempt_id: str,
    ) -> str:
        """Return the run/request digest including a placement-bound delta."""

        if self.program is None:
            return self.binding_digest
        delta = self.attempt_delta(
            output_dir, root=root, group_digest=group_digest, task_id=task_id, attempt_id=attempt_id,
        )
        return digest({"program_digest": self.program.program_digest, "attempt_delta_digest": delta.digest})

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical configured-run payload used for binding."""

        return {
            "schema": self.schema,
            "id": self.id,
            "config": self.config,
            "overrides": list(self.overrides),
            "program": self.program.to_dict() if self.program is not None else None,
        }

    @property
    def program_digest(self) -> str:
        """Return the digest binding config, typed inputs, and output contract."""

        return digest(self.to_dict())

    def command_for_attempt(
        self,
        output_dir: str,
        *,
        root: str | None = None,
        group_digest: str | None = None,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> TPENCommand | ConfiguredProgramCommand:
        """Return the fixed command after validating an attempt output path.

        Legacy TPEN output binding appends the existing three run.* overrides.
        Configured programs receive one canonical attempt delta instead.
        Configured calls require the placement group, task and attempt
        identities because those values are not known when the program is
        prepared.
        """

        if type(output_dir) is not str or not os.path.isabs(output_dir):
            raise ValueError("output_dir must be an absolute path")
        if self.program is None:
            return TPENCommand(
                self.config,
                (*self.overrides, f"run.root={output_dir}", "run.layout=flat", "run.run_id=core"),
            )
        if group_digest is None or task_id is None or attempt_id is None:
            raise ValueError("configured attempt binding requires group, task and attempt identities")
        delta = self.attempt_delta(
            output_dir, root=root, group_digest=group_digest, task_id=task_id, attempt_id=attempt_id,
        )
        return ConfiguredProgramCommand(self.program, delta)


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
                {
                    "id": run.id,
                    "config": run.config,
                    "overrides": list(run.overrides),
                    **({"program": run.program.to_dict()} if run.program is not None else {}),
                }
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

        if not isinstance(data, dict) and not hasattr(data, "keys"):
            raise TypeError("ConfiguredRunTable must be a mapping")
        if set(data) != {"schema", "group_digest", "runs"} or data.get("schema") not in {"tpen-executor/configured-runs-v1", cls.schema}:
            raise ValueError("ConfiguredRunTable keys or schema differ")
        if type(data["group_digest"]) is not str:
            raise TypeError("group_digest must be a string")
        if isinstance(data["runs"], (str, bytes, bytearray)) or not isinstance(data["runs"], Sequence):
            raise TypeError("runs must be a sequence")
        decoded: list[ConfiguredRun] = []
        for item in data["runs"]:
            if not isinstance(item, dict):
                raise TypeError("ConfiguredRun must be a mapping")
            expected = {"id", "config", "overrides"}
            allowed = expected | {"program"}
            if set(item) - allowed or expected - set(item):
                missing, extra = expected - set(item), set(item) - expected
                raise ValueError(f"ConfiguredRun keys differ (missing={sorted(missing)!r}, extra={sorted(extra)!r})")
            if type(item["id"]) is not str or type(item["config"]) is not str:
                raise TypeError("ConfiguredRun id and config must be strings")
            overrides = item["overrides"]
            if isinstance(overrides, (str, bytes, bytearray)) or not isinstance(overrides, Sequence):
                raise TypeError("ConfiguredRun overrides must be a sequence")
            if any(type(value) is not str for value in overrides):
                raise TypeError("ConfiguredRun overrides must contain strings")
            program = ConfiguredProgram.from_mapping(item["program"]) if "program" in item else None
            decoded.append(ConfiguredRun(item["id"], item["config"], tuple(overrides), program))
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
