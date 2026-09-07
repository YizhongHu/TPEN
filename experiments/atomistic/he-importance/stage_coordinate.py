"""The immutable scientific coordinate and closed manifest boundary for HI v2.

The committed JSON authority is the executable transcription of section 2.8 of
the consolidated HI design. This module owns only coordinate and schema
closedness; content-addressed identities and seed-stream derivation belong to
L2b.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
_PAYLOAD_KEYS = frozenset({"updates", "configuration"})
_REFERENCE_KEYS = frozenset({"energy"})
_ACCURACY_KEYS = frozenset({"conventional_band"})
_REFERENCE_ENERGY = -2.903724377034119598
_REFERENCE_ENERGY_TEXT = "-2.903724377034119598"
_FORBIDDEN_TRAIN_CONTENT_KEYS = frozenset(
    {
        "reference",
        "referenceEnergy",
        "reference_energy",
        "accuracy",
        "accuracyBand",
        "accuracy_band",
    }
)


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


def _validate_delegated_subtree(value: Any, label: str) -> None:
    """Apply L2a's structural and train-content rules without owning keys.

    Identity vocabulary and configuration semantics are delegated to later
    layers, but mapping shape, sequence traversal, and the train-reference
    firewall remain load-bearing here.
    """

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ManifestSchemaError(f"{label} keys must be strings")
            if key in _FORBIDDEN_TRAIN_CONTENT_KEYS:
                raise ManifestSchemaError(f"{label} contains forbidden train content {key!r}")
            _validate_delegated_subtree(nested, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _validate_delegated_subtree(nested, f"{label}[{index}]")
        return
    if type(value) not in {str, int, float, bool, type(None)}:
        raise ManifestSchemaError(f"{label} has unsupported value type {type(value).__name__}")
    if (type(value) is float and value == _REFERENCE_ENERGY) or (
        type(value) is str and value == _REFERENCE_ENERGY_TEXT
    ):
        raise ManifestSchemaError(f"{label} contains the evaluation reference energy")


def _validate_common(manifest: Mapping[str, Any], schema: str, expected_keys: frozenset[str]) -> None:
    _require_exact_keys(manifest, expected_keys, "manifest")
    if manifest["schema"] != schema:
        raise ManifestSchemaError(f"expected schema {schema!r}, got {manifest['schema']!r}")
    if not isinstance(manifest["stage"], str):
        raise ManifestSchemaError("stage must be a string")
    stage_definition(manifest["stage"])
    if not isinstance(manifest["scientific_identity"], Mapping):
        raise ManifestSchemaError("scientific_identity must be a mapping")
    if not isinstance(manifest["seed_identity"], Mapping):
        raise ManifestSchemaError("seed_identity must be a mapping")
    _require_exact_keys(manifest["payload"], _PAYLOAD_KEYS, "payload")
    _require_scalar(manifest["payload"]["updates"], int, "updates")
    if not isinstance(manifest["payload"]["configuration"], Mapping):
        raise ManifestSchemaError("payload.configuration must be a mapping")
    _validate_delegated_subtree(manifest["scientific_identity"], "scientific_identity")
    _validate_delegated_subtree(manifest["seed_identity"], "seed_identity")
    _validate_delegated_subtree(manifest["payload"]["configuration"], "payload.configuration")


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
