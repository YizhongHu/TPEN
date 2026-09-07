"""The immutable scientific coordinate and manifest boundary for HI v2.

This module deliberately describes scientific inputs only.  Materializing rows,
deriving content hashes, allocating seed streams, and generating job packets
belong to later planner slices.  In particular, a rank, device, or other
physical-topology fact is not a scientific identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


TRAIN_MANIFEST_SCHEMA = "he-importance/train/v1"
EVALUATION_MANIFEST_SCHEMA = "he-importance/evaluation/v1"


class ManifestSchemaError(ValueError):
    """A manifest does not satisfy its closed stage-specific schema."""


@dataclass(frozen=True)
class StageDefinition:
    """One scientific stage from the consolidated HI scan authority.

    Parameters
    ----------
    code
        Immutable stage coordinate.
    purpose
        ``"mechanics"``, ``"scientific"``, or ``"confirmation"``.
    seeds_per_point
        Fresh training lineages assigned to an admitted scientific point.
    maximum_lineages
        Maximum training lineages in the stated design, or ``None`` where a
        later frozen selection determines the number of protocols.
    updates
        Fixed training horizon.  ``None`` denotes the F horizon, which must be
        frozen before confirmation rather than guessed by a manifest writer.
    """

    code: str
    purpose: str
    seeds_per_point: int
    maximum_lineages: int | None
    updates: int | None


# The values below are the exact stage table in the consolidated design's
# section 2.8.  This is the single in-repository executable coordinate; later
# materializers consume it rather than maintaining a second table.
STAGE_DEFINITIONS = (
    StageDefinition("Q", "mechanics", 2, 480, 2_000),
    StageDefinition("O1", "scientific", 8, 1_920, 50_000),
    StageDefinition("O2", "scientific", 8, 320, 50_000),
    StageDefinition("A", "scientific", 8, 4_064, 50_000),
    StageDefinition("B", "scientific", 8, 960, 50_000),
    StageDefinition("R", "scientific", 12, 768, 50_000),
    StageDefinition("F", "confirmation", 48, 96, None),
)
STAGES = {definition.code: definition for definition in STAGE_DEFINITIONS}

_COMMON_KEYS = frozenset({"schema", "stage", "scientific_identity", "seed_identity", "payload"})
_TRAIN_KEYS = _COMMON_KEYS
_EVALUATION_KEYS = _COMMON_KEYS | {"reference", "accuracy"}
_TOPOLOGY_TOKENS = frozenset({"rank", "device", "gpu", "world", "node", "worker", "host"})
_TRAIN_FORBIDDEN_TOKENS = frozenset({"reference", "accuracy"})


def stage_definition(code: str) -> StageDefinition:
    """Return the declared stage definition for ``code``.

    Raises
    ------
    ManifestSchemaError
        If ``code`` is not a coordinate in the consolidated authority.
    """

    try:
        return STAGES[code]
    except KeyError as error:
        raise ManifestSchemaError(f"unknown HI stage {code!r}") from error


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ManifestSchemaError(
            f"{label} keys mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
        )


def _tokens(value: str) -> frozenset[str]:
    """Split a schema key into stable lowercase identity tokens."""

    normalized = "".join(character if character.isalnum() else " " for character in value)
    return frozenset(token.lower() for token in normalized.split())


def _reject_tokens(value: Any, forbidden: frozenset[str], label: str) -> None:
    """Reject forbidden field names at every nested mapping level."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ManifestSchemaError(f"{label} keys must be strings")
            matched = _tokens(key) & forbidden
            if matched:
                raise ManifestSchemaError(f"{label} contains forbidden field {key!r}")
            _reject_tokens(nested, forbidden, label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_tokens(nested, forbidden, label)


def _validate_common(manifest: Mapping[str, Any], schema: str, expected_keys: frozenset[str]) -> None:
    _require_exact_keys(manifest, expected_keys, "manifest")
    if manifest["schema"] != schema:
        raise ManifestSchemaError(f"expected schema {schema!r}, got {manifest['schema']!r}")
    if not isinstance(manifest["stage"], str):
        raise ManifestSchemaError("stage must be a string")
    stage_definition(manifest["stage"])
    for key in ("scientific_identity", "seed_identity", "payload"):
        if not isinstance(manifest[key], Mapping):
            raise ManifestSchemaError(f"{key} must be a mapping")
    _reject_tokens(manifest["scientific_identity"], _TOPOLOGY_TOKENS, "scientific_identity")
    _reject_tokens(manifest["seed_identity"], _TOPOLOGY_TOKENS, "seed_identity")


def validate_train_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the closed, reference-free training manifest schema."""

    _validate_common(manifest, TRAIN_MANIFEST_SCHEMA, _TRAIN_KEYS)
    _reject_tokens(manifest, _TRAIN_FORBIDDEN_TOKENS, "train manifest")


def validate_evaluation_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the distinct evaluation schema, which owns reference inputs."""

    _validate_common(manifest, EVALUATION_MANIFEST_SCHEMA, _EVALUATION_KEYS)
    if not isinstance(manifest["reference"], Mapping):
        raise ManifestSchemaError("evaluation reference must be a mapping")
    if not isinstance(manifest["accuracy"], Mapping):
        raise ManifestSchemaError("evaluation accuracy must be a mapping")

