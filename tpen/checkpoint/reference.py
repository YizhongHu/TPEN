"""Immutable identity for one published checkpoint artifact.

Checkpoint directories are the atomic storage unit, while ``latest.json`` is
only a mutable convenience pointer.  ``CheckpointRef`` deliberately hashes
the manifest and model contents and leaves the directory path out of its
content identity, so moving or collecting a run does not change which
checkpoint a publication names.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .artifact import require_complete_checkpoint_dir
from .hashing import file_sha256
from .manifest import CHECKPOINT_SCHEMA_VERSION
from .schema import read_manifest

CHECKPOINT_REF_SCHEMA = "tpen.checkpoint-ref/v1"
_STEP_NAME = re.compile(r"step_(?P<step>[0-9]{6})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """Validated, immutable identity for a complete checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path or str
        Concrete ``step_*`` directory containing ``COMPLETE`` and
        ``manifest.json``.  The path is a location, not part of ``content_id``.
    schema_version : int
        Checkpoint manifest schema version.
    kind : str
        Checkpoint manifest kind.
    next_iteration : int
        Resume cursor and step-directory identity recorded by the manifest.
    completed_updates : int
        Number of applied optimizer updates recorded by the manifest.
    manifest_sha256 : str
        SHA-256 digest of the manifest bytes.
    model_sha256 : str
        SHA-256 digest of the model payload named by the manifest.
    provenance : Mapping[str, Any]
        Immutable copy of the manifest's run provenance.

    Notes
    -----
    Use :meth:`from_directory` at an artifact boundary.  The public
    constructor validates value shape and immutability but cannot prove that a
    path currently contains the bytes named by the other fields until
    :meth:`validate` is called.
    """

    checkpoint_dir: Path
    schema_version: int
    kind: str
    next_iteration: int
    completed_updates: int
    manifest_sha256: str
    model_sha256: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        object.__setattr__(self, "schema_version", _nonnegative_int(self.schema_version, "schema_version"))
        object.__setattr__(self, "kind", _nonempty_text(self.kind, "kind"))
        object.__setattr__(self, "next_iteration", _nonnegative_int(self.next_iteration, "next_iteration"))
        object.__setattr__(
            self,
            "completed_updates",
            _nonnegative_int(self.completed_updates, "completed_updates"),
        )
        object.__setattr__(self, "manifest_sha256", _require_sha256(self.manifest_sha256, "manifest_sha256"))
        object.__setattr__(self, "model_sha256", _require_sha256(self.model_sha256, "model_sha256"))
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    @classmethod
    def from_directory(cls, path: str | Path) -> "CheckpointRef":
        """Validate and identify one concrete complete checkpoint directory.

        Raises
        ------
        ValueError
            If the directory is temporary, not canonically named, contains an
            unsupported manifest, or disagrees with its manifest.
        FileNotFoundError
            If the directory or one of its required files is absent.
        """

        checkpoint_dir = require_complete_checkpoint_dir(path)
        directory_match = _STEP_NAME.fullmatch(checkpoint_dir.name)
        if directory_match is None:
            raise ValueError(
                "CheckpointRef requires a canonical step_* directory, got "
                f"{checkpoint_dir.name!r}"
            )

        manifest_path = checkpoint_dir / "manifest.json"
        manifest = read_manifest(manifest_path, mode="model_only")
        if manifest.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "CheckpointRef requires a current v2 manifest with completed_updates; "
                f"got schema_version {manifest.schema_version}"
            )
        expected_name = f"step_{manifest.next_iteration:06d}"
        if checkpoint_dir.name != expected_name:
            raise ValueError(
                f"checkpoint path {checkpoint_dir.name!r} disagrees with manifest "
                f"next_iteration {manifest.next_iteration} (expected {expected_name!r})"
            )
        if manifest.completed_updates is None:
            raise ValueError("CheckpointRef manifest lacks completed_updates")
        if manifest.files.get("model") != "model.pt":
            raise ValueError(
                "CheckpointRef requires the manifest model entry to bind model.pt; "
                f"got {manifest.files.get('model')!r}"
            )
        model_path = checkpoint_dir / "model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"checkpoint model payload not found: {model_path}")

        return cls(
            checkpoint_dir=checkpoint_dir,
            schema_version=manifest.schema_version,
            kind=manifest.kind,
            next_iteration=manifest.next_iteration,
            completed_updates=manifest.completed_updates,
            manifest_sha256=file_sha256(manifest_path),
            model_sha256=file_sha256(model_path),
            provenance=manifest.provenance,
        )

    # These aliases keep the artifact boundary readable at call sites that
    # speak in terms of a checkpoint rather than a generic directory.
    from_checkpoint_dir = from_directory
    from_path = from_directory

    @property
    def manifest_digest(self) -> str:
        """Alias for the manifest content digest."""

        return self.manifest_sha256

    @property
    def model_digest(self) -> str:
        """Alias for the model content digest."""

        return self.model_sha256

    @property
    def run_provenance(self) -> Mapping[str, Any]:
        """Alias for the immutable manifest provenance mapping."""

        return self.provenance

    @property
    def identity(self) -> dict[str, Any]:
        """Return the path-independent fields that define this checkpoint."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
            "manifest_sha256": self.manifest_sha256,
            "model_sha256": self.model_sha256,
            "provenance": _thaw(self.provenance),
        }

    @property
    def content_id(self) -> str:
        """Return the stable path-independent content identity."""

        encoded = json.dumps(
            self.identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def content_sha256(self) -> str:
        """Alias for :attr:`content_id`."""

        return self.content_id

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe publication representation."""

        return {
            "schema": CHECKPOINT_REF_SCHEMA,
            "checkpoint_dir": str(self.checkpoint_dir),
            **self.identity,
            "content_id": self.content_id,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CheckpointRef":
        """Construct a ref from serialized fields and verify its content ID.

        Deserialization verifies the signed-by-content relationship without
        requiring the source directory to remain present.  Call ``validate``
        when a live artifact must be checked again.
        """

        if not isinstance(data, Mapping):
            raise TypeError("serialized CheckpointRef must be a mapping")
        if data.get("schema") != CHECKPOINT_REF_SCHEMA:
            raise ValueError(
                f"unsupported CheckpointRef schema {data.get('schema')!r}; "
                f"expected {CHECKPOINT_REF_SCHEMA!r}"
            )
        required = (
            "checkpoint_dir",
            "schema_version",
            "kind",
            "next_iteration",
            "completed_updates",
            "manifest_sha256",
            "model_sha256",
            "provenance",
            "content_id",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"serialized CheckpointRef missing fields: {missing}")
        ref = cls(
            checkpoint_dir=Path(str(data["checkpoint_dir"])),
            schema_version=data["schema_version"],
            kind=data["kind"],
            next_iteration=data["next_iteration"],
            completed_updates=data["completed_updates"],
            manifest_sha256=data["manifest_sha256"],
            model_sha256=data["model_sha256"],
            provenance=data["provenance"],
        )
        if data["content_id"] != ref.content_id:
            raise ValueError(
                "CheckpointRef content_id does not match its immutable identity"
            )
        return ref

    from_dict = from_mapping

    def validate(self) -> "CheckpointRef":
        """Revalidate this ref against the current bytes at its path."""

        current = type(self).from_directory(self.checkpoint_dir)
        if current != self:
            differences = {
                field: (getattr(self, field), getattr(current, field))
                for field in (
                    "schema_version",
                    "kind",
                    "next_iteration",
                    "completed_updates",
                    "manifest_sha256",
                    "model_sha256",
                    "provenance",
                )
                if getattr(self, field) != getattr(current, field)
            }
            raise ValueError(
                f"checkpoint bytes no longer match CheckpointRef at {self.checkpoint_dir}: "
                f"{differences}"
            )
        return self


def checkpoint_ref(path: str | Path) -> CheckpointRef:
    """Build a validated :class:`CheckpointRef` from a step directory."""

    return CheckpointRef.from_directory(path)


def serialize_checkpoint_ref(ref: CheckpointRef) -> dict[str, Any]:
    """Serialize one checkpoint ref for a publication record."""

    if not isinstance(ref, CheckpointRef):
        raise TypeError(f"expected CheckpointRef, got {type(ref).__name__}")
    return ref.to_dict()


def deserialize_checkpoint_ref(data: Mapping[str, Any]) -> CheckpointRef:
    """Deserialize one checkpoint ref and verify its stable content ID."""

    return CheckpointRef.from_mapping(data)


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value}")
    return int(value)


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _freeze(value: Any) -> Any:
    """Deep-freeze JSON-compatible provenance without coercing its values."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("provenance mapping keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        "provenance must contain only JSON-compatible values, got "
        f"{type(value).__name__}"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


__all__ = [
    "CHECKPOINT_REF_SCHEMA",
    "CheckpointRef",
    "checkpoint_ref",
    "deserialize_checkpoint_ref",
    "serialize_checkpoint_ref",
]
