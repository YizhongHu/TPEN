"""The immutable scientific coordinate and closed manifest boundary for HI v2.

The committed JSON authority is the executable transcription of section 2.8 of
the consolidated HI design. This module owns only coordinate and schema
closedness; content-addressed identities and seed-stream derivation belong to
L2b.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


TRAIN_MANIFEST_SCHEMA = "he-importance/train/v1"
EVALUATION_MANIFEST_SCHEMA = "he-importance/evaluation/v1"
_AUTHORITY_PATH = Path(__file__).with_name("stage_coordinate_authority.json")


class ManifestSchemaError(ValueError):
    """A manifest does not satisfy its closed stage-specific schema."""


@dataclass(frozen=True)
class StageDefinition:
    """One stage from the committed consolidated-coordinate authority."""

    code: str
    purpose: str
    seeds_per_point: int
    maximum_lineages: int | None
    updates: int | None


def _load_stage_definitions() -> tuple[StageDefinition, ...]:
    """Load the committed authority, refusing a malformed transcription."""

    raw = json.loads(_AUTHORITY_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ManifestSchemaError("stage authority must be a list")
    expected = frozenset({"code", "purpose", "seeds_per_point", "maximum_lineages", "updates"})
    definitions = []
    for entry in raw:
        if not isinstance(entry, Mapping) or frozenset(entry) != expected:
            raise ManifestSchemaError("stage authority entry has an invalid schema")
        definitions.append(StageDefinition(**entry))
    if len({definition.code for definition in definitions}) != len(definitions):
        raise ManifestSchemaError("stage authority repeats a code")
    return tuple(definitions)


STAGE_DEFINITIONS = _load_stage_definitions()
STAGES = {definition.code: definition for definition in STAGE_DEFINITIONS}

_COMMON_KEYS = frozenset({"schema", "stage", "scientific_identity", "seed_identity", "payload"})
_TRAIN_KEYS = _COMMON_KEYS
_EVALUATION_KEYS = _COMMON_KEYS | {"reference", "accuracy"}
_SCIENTIFIC_IDENTITY_KEYS = frozenset({"architecture", "optimizer"})
_SEED_IDENTITY_KEYS = frozenset({"stage_seed", "lineage"})
_PAYLOAD_KEYS = frozenset({"updates"})
_REFERENCE_KEYS = frozenset({"energy"})
_ACCURACY_KEYS = frozenset({"conventional_band"})


def stage_definition(code: str) -> StageDefinition:
    """Return the declared stage definition for ``code``."""

    try:
        return STAGES[code]
    except KeyError as error:
        raise ManifestSchemaError(f"unknown HI stage {code!r}") from error


def _require_exact_keys(value: Any, expected: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping) or frozenset(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ManifestSchemaError(f"{label} keys mismatch: expected={sorted(expected)}, actual={actual}")


def _require_scalar(value: Any, expected_type: type[object], label: str) -> None:
    """Reject container values so no unvalidated nested shape can enter."""

    if type(value) is not expected_type:
        raise ManifestSchemaError(f"{label} must be {expected_type.__name__}")


def _validate_common(manifest: Mapping[str, Any], schema: str, expected_keys: frozenset[str]) -> None:
    _require_exact_keys(manifest, expected_keys, "manifest")
    if manifest["schema"] != schema:
        raise ManifestSchemaError(f"expected schema {schema!r}, got {manifest['schema']!r}")
    if not isinstance(manifest["stage"], str):
        raise ManifestSchemaError("stage must be a string")
    stage_definition(manifest["stage"])
    _require_exact_keys(manifest["scientific_identity"], _SCIENTIFIC_IDENTITY_KEYS, "scientific_identity")
    _require_exact_keys(manifest["seed_identity"], _SEED_IDENTITY_KEYS, "seed_identity")
    _require_exact_keys(manifest["payload"], _PAYLOAD_KEYS, "payload")
    _require_scalar(manifest["scientific_identity"]["architecture"], str, "architecture")
    _require_scalar(manifest["scientific_identity"]["optimizer"], str, "optimizer")
    _require_scalar(manifest["seed_identity"]["stage_seed"], int, "stage_seed")
    _require_scalar(manifest["seed_identity"]["lineage"], int, "lineage")
    _require_scalar(manifest["payload"]["updates"], int, "updates")


def validate_train_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the closed, reference-free training manifest schema."""

    _validate_common(manifest, TRAIN_MANIFEST_SCHEMA, _TRAIN_KEYS)


def validate_evaluation_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the distinct evaluation schema, which owns reference inputs."""

    _validate_common(manifest, EVALUATION_MANIFEST_SCHEMA, _EVALUATION_KEYS)
    _require_exact_keys(manifest["reference"], _REFERENCE_KEYS, "reference")
    _require_exact_keys(manifest["accuracy"], _ACCURACY_KEYS, "accuracy")
    _require_scalar(manifest["reference"]["energy"], float, "reference energy")
    _require_scalar(manifest["accuracy"]["conventional_band"], float, "conventional band")
