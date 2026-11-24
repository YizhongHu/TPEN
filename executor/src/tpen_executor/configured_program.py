"""Hydra-free configured leaf-program records.

C2.1 configured programs are supported only from a TPEN source checkout or
editable checkout.  Installed-package launches are intentionally fail-closed
until a separate trusted-checkout binding is available.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
import re
import stat
from collections.abc import Mapping
from typing import Any, ClassVar

from ._codec import canonical_bytes, decode_json, digest, fields
from ._records import frozen_record


# Data, not an experiment-specific branch in the runner.  Every entry names a
# tracked script and a typed argument shape.
CONFIGURED_PROGRAM_ALLOWLIST = {
    "stage-q": {
        "trusted_root": os.path.realpath(Path(__file__).resolve().parents[3]),
        "relative_path": "experiments/atomistic/he-importance/run_stage_q.py",
        "arguments": ("manifest", "row", "output_root"),
        "output_contract": "attempt-core",
    },
}


@frozen_record
class ConfiguredLoad:
    """Explicit load decision carried by a prepared configured program."""

    mode: str
    path: str | None

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in {"fresh", "train_resume"}:
            raise ValueError("configured load mode must be 'fresh' or 'train_resume'")
        if self.mode == "fresh":
            if self.path is not None:
                raise ValueError("fresh configured load must have a null path")
            return
        if type(self.path) is not str or not os.path.isabs(self.path):
            raise ValueError("train_resume configured load path must be absolute")
        normalized = os.path.normpath(self.path)
        if normalized != self.path:
            raise ValueError("train_resume configured load path must be normalized")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "path": self.path}

    @classmethod
    def from_mapping(cls, data: Any) -> "ConfiguredLoad":
        if not isinstance(data, Mapping) or set(data) != {"mode", "path"}:
            raise ValueError("ConfiguredLoad keys differ")
        if type(data["mode"]) is not str or type(data["path"]) not in (str, type(None)):
            raise TypeError("configured load fields must be primitive")
        return cls(data["mode"], data["path"])


ATTEMPT_DELTA_SCHEMA = "tpen-executor/attempt-delta-v1"


@frozen_record
class AttemptDelta:
    """Placement-bound load and physical attempt/core destination."""

    schema: ClassVar[str] = ATTEMPT_DELTA_SCHEMA
    attempt_core_dir: str
    load: ConfiguredLoad
    group_digest: str
    task_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        if type(self.attempt_core_dir) is not str or not os.path.isabs(self.attempt_core_dir):
            raise ValueError("attempt_core_dir must be an absolute path")
        normalized = os.path.normpath(self.attempt_core_dir)
        if normalized != self.attempt_core_dir or os.path.basename(normalized) != "core":
            raise ValueError("attempt_core_dir must be a normalized .../<attempt>/core path")
        if type(self.load) is not ConfiguredLoad:
            raise TypeError("attempt delta load must be ConfiguredLoad")
        for name in ("group_digest", "task_id", "attempt_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or os.sep in value or (os.altsep and os.altsep in value):
                raise ValueError(f"attempt delta {name} must be one path component")
        if self.load.path is not None:
            attempt_leaf = os.path.dirname(self.attempt_core_dir)
            if os.path.commonpath((attempt_leaf, self.load.path)) == attempt_leaf:
                raise ValueError("train_resume load path must not lie in its own attempt leaf")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def digest(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "attempt_core_dir": self.attempt_core_dir,
            "load": self.load.to_dict(),
            "group_digest": self.group_digest,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> "AttemptDelta":
        fields(data, {"schema", "attempt_core_dir", "load", "group_digest", "task_id", "attempt_id"}, cls.schema, "AttemptDelta")
        return cls(
            data["attempt_core_dir"], ConfiguredLoad.from_mapping(data["load"]),
            data["group_digest"], data["task_id"], data["attempt_id"],
        )

    @classmethod
    def decode_canonical(cls, value: str) -> "AttemptDelta":
        if type(value) is not str:
            raise TypeError("attempt delta argument must be a string")
        delta = cls.from_mapping(decode_json(value.encode("utf-8")))
        if delta.canonical_bytes.decode("utf-8") != value:
            raise ValueError("attempt delta is not canonical JSON")
        return delta


def _absolute_path(value: Any, name: str) -> str:
    """Validate one absolute path without silently rebasing user input."""

    if type(value) is not str or not os.path.isabs(value):
        raise ValueError(f"{name} must be an absolute path")
    return os.path.normpath(value)


@frozen_record
class ConfiguredProgram:
    """Closed, typed description of one non-legacy leaf program.

    ``load`` is explicit even for a fresh run; an omitted load is invalid.
    Placement-dependent output and attempt identities are carried separately
    by :class:`AttemptDelta`.
    """

    schema: ClassVar[str] = "tpen-executor/configured-program-v1"
    entrypoint: str
    script_path: str
    manifest: str
    row: str
    output_root: str
    output_contract: str = "attempt-core"
    manifest_sha256: str = ""
    load: ConfiguredLoad | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.entrypoint) is not str or self.entrypoint not in CONFIGURED_PROGRAM_ALLOWLIST:
            raise ValueError(f"configured program entrypoint is not allowlisted: {self.entrypoint!r}")
        if "." in self.entrypoint or not re.fullmatch(r"[a-z][a-z0-9-]*", self.entrypoint):
            raise ValueError("configured program entrypoint must be a closed simple name")
        script_path = _absolute_path(self.script_path, "script_path")
        entry = CONFIGURED_PROGRAM_ALLOWLIST[self.entrypoint]
        trusted_root = os.path.realpath(entry["trusted_root"])
        expected = os.path.realpath(os.path.join(trusted_root, entry["relative_path"]))
        resolved = os.path.realpath(script_path)
        if os.path.commonpath((trusted_root, resolved)) != trusted_root or resolved != expected or resolved != script_path:
            raise ValueError("script_path does not match the trusted configured-program allowlist")
        if not stat.S_ISREG(os.stat(script_path, follow_symlinks=False).st_mode):
            raise ValueError("configured program script_path must be a regular file")
        manifest = _absolute_path(self.manifest, "manifest")
        try:
            manifest_stat = os.stat(manifest, follow_symlinks=False)
        except OSError as error:
            raise ValueError("manifest must exist before configured launch") from error
        if not stat.S_ISREG(manifest_stat.st_mode):
            raise ValueError("manifest must be a regular file")
        try:
            computed_manifest_sha256 = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError("manifest must be readable before configured launch") from error
        if self.manifest_sha256 and self.manifest_sha256 != computed_manifest_sha256:
            raise ValueError("manifest_sha256 does not match manifest bytes")
        if type(self.row) is not str or re.fullmatch(r"[0-9a-f]{64}", self.row) is None:
            raise ValueError("row must be a lowercase SHA-256 row identifier")
        output_root = _absolute_path(self.output_root, "output_root")
        expected_contract = CONFIGURED_PROGRAM_ALLOWLIST[self.entrypoint]["output_contract"]
        if self.output_contract != expected_contract:
            raise ValueError("configured program output contract is not allowlisted")
        if self.load is None:
            raise ValueError("configured program requires an explicit load object")
        load = self.load if type(self.load) is ConfiguredLoad else ConfiguredLoad.from_mapping(self.load)
        object.__setattr__(self, "load", load)
        object.__setattr__(self, "script_path", script_path)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "manifest_sha256", computed_manifest_sha256)

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
            "manifest_sha256": self.manifest_sha256,
            "load": self.load.to_dict(),
        }
        return {"schema": self.schema, **value} if include_schema else value

    @classmethod
    def from_mapping(cls, data: Any) -> "ConfiguredProgram":
        """Decode a strict configured-program mapping."""

        expected = {"schema", "entrypoint", "script_path", "manifest", "row", "output_root", "output_contract", "load"}
        keys = set(data)
        if (keys != expected and keys != expected | {"manifest_sha256"}) or data.get("schema") != cls.schema:
            raise ValueError("ConfiguredProgram keys or schema differ")
        return cls(
            data["entrypoint"], data["script_path"], data["manifest"],
            data["row"], data["output_root"], data["output_contract"], data.get("manifest_sha256", ""), data["load"],
        )


# Design notes call the typed configured entry a LeafProgram; retain that
# vocabulary as an alias without creating a second model or digest path.
LeafProgram = ConfiguredProgram


__all__ = ["ATTEMPT_DELTA_SCHEMA", "AttemptDelta", "CONFIGURED_PROGRAM_ALLOWLIST", "ConfiguredLoad", "ConfiguredProgram", "LeafProgram"]
