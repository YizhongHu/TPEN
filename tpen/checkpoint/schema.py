"""Checkpoint schema dispatch."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .manifest import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    LEGACY_CHECKPOINT_KIND,
    LEGACY_CHECKPOINT_SCHEMA_VERSION,
    CheckpointManifest,
)
from .payload import CheckpointPayload, TRAIN_RESUME_PAYLOAD

# Acceptance is mode-dependent, not global. ``model_only`` reads `model.pt` and
# the config hashes, both of which a v1 manifest carries, so an archived
# artifact still restores weights.
#
# ``train_resume`` refuses every v1 manifest because a v1 manifest cannot prove
# which `trainer.json` key set it carries. B1 renamed the trainer keys from
# `global_step`/`completed_steps` to `next_iteration`/`completed_updates`
# *without* bumping the manifest schema, so schema_version 1 covers both: an
# artifact written before B1 is genuinely unresumable, while one written after
# B1 would resume fine. The version cannot tell them apart, so all v1 is
# refused rather than guessed. This is a rejection, not a migration -- the two
# populations are not distinguishable well enough to upgrade either.
#
# Note the reason is *not* that the restore path reads
# `manifest.completed_updates`: it does not. That path takes trainer state from
# `trainer.json` via `_load_trainer`, and the manifest counters only reach
# `RestoreReport`.
SCHEMA_VERSIONS_BY_RESTORE_MODE: dict[str, frozenset[int]] = {
    "model_only": frozenset({LEGACY_CHECKPOINT_SCHEMA_VERSION, CHECKPOINT_SCHEMA_VERSION}),
    "train_resume": frozenset({CHECKPOINT_SCHEMA_VERSION}),
}

# Each schema version pins its own artifact kind: v1 artifacts keep the
# pre-rename spelling and must stay readable without being rewritten.
KIND_BY_SCHEMA_VERSION: dict[int, str] = {
    LEGACY_CHECKPOINT_SCHEMA_VERSION: LEGACY_CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION: CHECKPOINT_KIND,
}


def read_manifest(path: str | Path, *, mode: str) -> CheckpointManifest:
    """Read and validate a checkpoint manifest for one restore mode.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the checkpoint's ``manifest.json``.
    mode : str
        Restore mode the manifest is being read for. Acceptance is
        mode-dependent; see `SCHEMA_VERSIONS_BY_RESTORE_MODE`.

    Returns
    -------
    CheckpointManifest
        The validated manifest.
    """

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    manifest = CheckpointManifest.from_mapping(data)
    validate_manifest_schema(manifest, mode=mode, path=manifest_path)
    return manifest


def validate_manifest_schema(
    manifest: CheckpointManifest, *, mode: str, path: Path | None = None
) -> None:
    """Fail if `manifest` is not a TPEN checkpoint schema `mode` can restore.

    Parameters
    ----------
    manifest : CheckpointManifest
        Manifest to check.
    mode : str
        Restore mode requesting the manifest. Passed explicitly rather than
        held in module state, so concurrent restores in different modes cannot
        observe each other's acceptance set.
    path : pathlib.Path or None, optional
        Manifest path, used only to label error messages.

    Raises
    ------
    ValueError
        If `mode` is not a restore mode with a schema policy, if the manifest's
        schema version is not accepted for `mode`, or if the manifest's kind
        does not match the kind pinned for its schema version.
    """

    label = "" if path is None else f"{path}: "
    accepted = SCHEMA_VERSIONS_BY_RESTORE_MODE.get(mode)
    if accepted is None:
        raise ValueError(
            f"{label}no checkpoint schema policy for restore mode {mode!r}; "
            f"modes with a policy: {sorted(SCHEMA_VERSIONS_BY_RESTORE_MODE)}"
        )
    # Version-vs-mode is checked before kind so a v1 artifact under
    # `train_resume` is refused for the reason that actually applies.
    if manifest.schema_version not in accepted:
        raise ValueError(
            f"{label}checkpoint schema_version {manifest.schema_version} is not supported "
            f"for restore mode {mode!r}; supported versions: {sorted(accepted)}"
        )
    expected_kind = KIND_BY_SCHEMA_VERSION[manifest.schema_version]
    if manifest.kind != expected_kind:
        raise ValueError(
            f"{label}unsupported checkpoint kind {manifest.kind!r} for "
            f"schema_version {manifest.schema_version}; expected {expected_kind!r}"
        )
    if manifest.payload is not None:
        payload = CheckpointPayload.from_manifest(manifest.payload)
        payload.validate_restore_intent(mode)
        payload.validate_files(manifest.files)


def validate_manifest_state(
    manifest: CheckpointManifest, state: Mapping[str, Any], *, mode: str
) -> None:
    """Validate restore state against the manifest's mode-specific payload.

    State validation is separate from manifest-file validation because the
    trainer state lives in ``trainer.json`` rather than in ``manifest.json``.
    Restore callers must invoke this after pre-reading that file and before
    loading any mutable component.
    """

    if mode != "train_resume":
        return
    payload = (
        TRAIN_RESUME_PAYLOAD
        if manifest.payload is None
        else CheckpointPayload.from_manifest(manifest.payload)
    )
    payload.validate_state(state)
