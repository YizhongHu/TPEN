"""Hydra-free configured leaf-program records."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, ClassVar

from ._codec import digest, fields
from ._records import frozen_record


# Data, not an experiment-specific branch in the runner.  Every entry names a
# tracked script and a typed argument shape.
CONFIGURED_PROGRAM_ALLOWLIST = {
    "stage-q": {
        "script": "experiments/atomistic/he-importance/run_stage_q.py",
        "arguments": ("manifest", "row", "output_root"),
        "output_contract": "attempt-core",
    },
}


def _absolute_path(value: Any, name: str) -> str:
    """Validate one absolute path without silently rebasing user input."""

    if type(value) is not str or not os.path.isabs(value):
        raise ValueError(f"{name} must be an absolute path")
    return os.path.normpath(value)


@frozen_record
class ConfiguredProgram:
    """Closed, typed description of one non-legacy leaf program."""

    schema: ClassVar[str] = "tpen-executor/configured-program-v1"
    entrypoint: str
    script_path: str
    manifest: str
    row: str
    output_root: str
    output_contract: str = "attempt-core"

    def __post_init__(self) -> None:
        if type(self.entrypoint) is not str or self.entrypoint not in CONFIGURED_PROGRAM_ALLOWLIST:
            raise ValueError(f"configured program entrypoint is not allowlisted: {self.entrypoint!r}")
        if "." in self.entrypoint or not re.fullmatch(r"[a-z][a-z0-9-]*", self.entrypoint):
            raise ValueError("configured program entrypoint must be a closed simple name")
        script_path = _absolute_path(self.script_path, "script_path")
        expected = CONFIGURED_PROGRAM_ALLOWLIST[self.entrypoint]["script"]
        if tuple(Path(script_path).parts[-len(Path(expected).parts):]) != tuple(Path(expected).parts):
            raise ValueError("script_path does not match the configured-program allowlist")
        if not Path(script_path).is_file():
            raise ValueError("configured program script_path must name a tracked file")
        manifest = _absolute_path(self.manifest, "manifest")
        if Path(manifest).name != "o1_scan_manifest_v1.json":
            raise ValueError("manifest is not the Stage-Q scan manifest")
        if type(self.row) is not str or re.fullmatch(r"[0-9a-f]{64}", self.row) is None:
            raise ValueError("row must be a lowercase SHA-256 row identifier")
        output_root = _absolute_path(self.output_root, "output_root")
        expected_contract = CONFIGURED_PROGRAM_ALLOWLIST[self.entrypoint]["output_contract"]
        if self.output_contract != expected_contract:
            raise ValueError("configured program output contract is not allowlisted")
        object.__setattr__(self, "script_path", script_path)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "output_root", output_root)

    @property
    def program_digest(self) -> str:
        """Return the canonical digest of the executable and typed inputs."""

        return digest(self.to_dict(include_schema=False))

    def argv(self) -> tuple[str, ...]:
        """Return generated script arguments, never caller-provided argv."""

        return (
            "--manifest", self.manifest,
            "--row", self.row,
            "--output-root", self.output_root,
        )

    def to_dict(self, *, include_schema: bool = True) -> dict[str, Any]:
        """Return the canonical configured-program mapping."""

        value = {
            "entrypoint": self.entrypoint,
            "script_path": self.script_path,
            "manifest": self.manifest,
            "row": self.row,
            "output_root": self.output_root,
            "output_contract": self.output_contract,
        }
        return {"schema": self.schema, **value} if include_schema else value

    @classmethod
    def from_mapping(cls, data: Any) -> "ConfiguredProgram":
        """Decode a strict configured-program mapping."""

        fields(data, {"schema", "entrypoint", "script_path", "manifest", "row", "output_root", "output_contract"}, cls.schema, "ConfiguredProgram")
        return cls(
            data["entrypoint"], data["script_path"], data["manifest"],
            data["row"], data["output_root"], data["output_contract"],
        )


# Design notes call the typed configured entry a LeafProgram; retain that
# vocabulary as an alias without creating a second model or digest path.
LeafProgram = ConfiguredProgram


__all__ = ["CONFIGURED_PROGRAM_ALLOWLIST", "ConfiguredProgram", "LeafProgram"]
