"""Checkpoint manifest schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tpen.artifacts import write_json

CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_KIND = "tpen.checkpoint"

# Schema v1 recorded a single ambiguous ``step`` -- which was in fact the
# durable resume cursor -- under the pre-rename kind, and never recorded the
# applied-update count. Nothing writes v1 any more, but `from_mapping` still
# reads it so an archived artifact stays usable for ``model_only`` restores.
# There is deliberately no v1 -> v2 upgrade path: v1 is read, never rewritten.
LEGACY_CHECKPOINT_SCHEMA_VERSION = 1
LEGACY_CHECKPOINT_KIND = "spenn.checkpoint"


@dataclass(frozen=True)
class CheckpointManifest:
    """Readable metadata for one checkpoint directory.

    Parameters
    ----------
    schema_version : int
        Manifest schema version. `CHECKPOINT_SCHEMA_VERSION` for anything
        written by this package; `LEGACY_CHECKPOINT_SCHEMA_VERSION` for an
        archived artifact.
    kind : str
        Artifact kind. Pinned per schema version -- v2 uses `CHECKPOINT_KIND`,
        v1 keeps `LEGACY_CHECKPOINT_KIND`.
    next_iteration : int
        Trainer resume cursor at write time: the iteration a ``train_resume``
        run continues from. This is the checkpoint's directory identity.
    completed_updates : int or None
        Optimizer updates that had actually been applied at write time. The
        two counters diverge whenever a completed iteration skipped its
        update. ``None`` for a v1 manifest, which never recorded it.
    created_at_unix : float
        Wall-clock write time.
    files : dict
        Component name to checkpoint-relative file name.
    hashes : dict
        Component config hashes used to gate restores.
    runtime : dict
        Device/dtype/torch metadata for the writing process.
    provenance : dict
        Run, git, host, and package-version metadata.
    payload : dict or None, optional
        Canonical :class:`CheckpointPayload` manifest describing the files and
        restore intents this artifact supports.  ``None`` is retained when
        reading older manifests that predate payload profiles.
    """

    schema_version: int
    kind: str
    next_iteration: int
    completed_updates: int | None
    created_at_unix: float
    files: dict[str, str]
    hashes: dict[str, str | None]
    runtime: dict[str, Any]
    provenance: dict[str, Any]
    payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest mapping.

        Returns
        -------
        dict
            The manifest's own fields, mirrored verbatim. Both progress
            counters are named explicitly; neither is re-derived from the
            other.
        """

        data = {
            "schema_version": int(self.schema_version),
            "kind": self.kind,
            "next_iteration": int(self.next_iteration),
            "completed_updates": (
                None if self.completed_updates is None else int(self.completed_updates)
            ),
            "created_at_unix": float(self.created_at_unix),
            "files": dict(self.files),
            "hashes": dict(self.hashes),
            "runtime": dict(self.runtime),
            "provenance": dict(self.provenance),
        }
        if self.payload is not None:
            data["payload"] = dict(self.payload)
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CheckpointManifest":
        """Build a manifest from a loaded JSON mapping.

        Reading is version-aware so an archived v1 artifact stays constructible
        and can proceed to a ``model_only`` restore. Acceptance is *not*
        decided here -- `tpen.checkpoint.schema.validate_manifest_schema` owns
        which versions each restore mode admits.

        Parameters
        ----------
        data : Mapping
            Parsed ``manifest.json`` contents.

        Returns
        -------
        CheckpointManifest
            The manifest. ``completed_updates`` is ``None`` for v1.

        Raises
        ------
        KeyError
            If a required field for the manifest's own schema version is
            missing.
        """

        schema_version = int(data["schema_version"])
        if schema_version == LEGACY_CHECKPOINT_SCHEMA_VERSION:
            # v1's lone `step` was the resume cursor, so it maps onto
            # `next_iteration` exactly; the update count was never written.
            next_iteration = int(data["step"])
            completed_updates: int | None = None
        else:
            # Both counters are required from v2 onward. Any unrecognized
            # version is read with the v2 field names, which is a best effort
            # rather than a guarantee: a future version that keeps both names
            # constructs here and is then rejected by the schema gate with its
            # version reported, but one that renames or drops either field
            # raises KeyError from this line instead, before the gate can run.
            next_iteration = int(data["next_iteration"])
            completed_updates = int(data["completed_updates"])
        return cls(
            schema_version=schema_version,
            kind=str(data["kind"]),
            next_iteration=next_iteration,
            completed_updates=completed_updates,
            created_at_unix=float(data["created_at_unix"]),
            files={str(key): str(value) for key, value in dict(data["files"]).items()},
            hashes={str(key): value for key, value in dict(data.get("hashes", {})).items()},
            runtime=dict(data.get("runtime", {})),
            provenance=dict(data.get("provenance", {})),
            payload=(None if data.get("payload") is None else dict(data["payload"])),
        )

    def write(self, path: str | Path) -> None:
        """Write the manifest JSON."""

        write_json(Path(path), self.to_dict())
