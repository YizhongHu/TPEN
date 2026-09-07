"""The immutable scientific coordinate and closed manifest boundary for HI v2.

The committed JSON authority is the executable transcription of section 2.8 of
the consolidated HI design. This module owns only coordinate and schema
closedness; content-addressed identities and seed-stream derivation belong to
L2b. ``topology`` is the sole caller-open execution subtree: L2b excludes it
wholesale from canonical identity, while facts outside it are science by
caller declaration and therefore identity-significant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable
from types import MappingProxyType


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

TOPOLOGY_KEY = "topology"
_COMMON_KEYS = frozenset({"schema", "stage", "scientific_identity", "seed_identity", "payload", TOPOLOGY_KEY})
_TRAIN_KEYS = _COMMON_KEYS
_EVALUATION_KEYS = _COMMON_KEYS | {"reference", "accuracy"}
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
_L2B_SEED_IDENTITY_KEYS = frozenset({"stage", "label", "namespace"})
_TOPOLOGY_KEYS = frozenset(
    {
        "rank",
        "ranks",
        "device",
        "deviceid",
        "devices",
        "gpu",
        "gpus",
        "world",
        "worldsize",
        "node",
        "nodecount",
        "nodes",
        "worker",
        "workerindex",
        "workers",
        "host",
        "hostname",
        "topology",
    }
)
_INTENDED_CONFIGURATIONS_PATH = Path(__file__).with_name("intended_configurations.json")


# These labels are scientific namespaces, rather than an inventory of rows.
# In particular, no stage materializer may use the historical fixed breadth
# count as a substitute for expanding its literal factor union.
SEED_NAMESPACE_STARTS = {
    "Q": 810_001,
    "O1": 820_001,
    "O2": 821_001,
    "A": 830_001,
    "B": 840_001,
    "R": 850_001,
    "F": 860_001,
}
SEED_STREAMS = (
    "model_initialization",
    "training_sampler",
    "method_randomness",
    "diagnostic",
    "evaluation_calibration",
    "evaluation_inference",
)


class MaterializationError(ValueError):
    """A requested HI configuration cannot become an immutable row."""


@dataclass(frozen=True)
class OptimizerCell:
    """A literal optimizer cell, including an explicit unavailable state.

    Parameters
    ----------
    method
        Declared method name.  This is preserved even when unavailable.
    status
        Either ``"available"`` or ``"unavailable"``.
    reason
        Required only for unavailable cells.  It makes an implementation
        qualification failure visible instead of silently selecting Adam.
    """

    method: str
    status: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "unavailable"}:
            raise MaterializationError(f"unknown optimizer status {self.status!r}")
        if self.status == "unavailable" and not self.reason:
            raise MaterializationError("unavailable optimizer cells need a reason")
        if self.status == "available" and self.reason is not None:
            raise MaterializationError("available optimizer cells cannot carry an unavailable reason")


@dataclass(frozen=True)
class MaterializedCell:
    """One content-addressed train row without an attempt identity."""

    manifest: Mapping[str, Any]
    content_hash: str
    output_path: Path
    seed_streams: Mapping[str, int]


def canonical_json(value: Any) -> bytes:
    """Return canonical identity bytes after topology projection.

    Identity-bearing callers may supply open scientific/configuration mappings.
    Physical execution topology is not scientific content and is therefore
    removed before both hashing and path derivation.
    """

    try:
        return json.dumps(
            _project_identity(value),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise MaterializationError("identity values must be finite JSON data") from error


def content_hash(value: Any) -> str:
    """Return the SHA-256 identity of canonical literal content."""

    return sha256(canonical_json(value)).hexdigest()


def _freeze(value: Any) -> Any:
    """Recursively freeze a JSON-shaped manifest after it has been hashed."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(nested) for key, nested in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(nested) for nested in value)
    return value


def _normalized_key(key: str) -> str:
    """Normalize a mapping key without treating spelling as scientific content."""

    return "".join(character.lower() for character in key if character.isalnum())


def _project_identity(value: Any) -> Any:
    """Convert frozen containers to JSON and exclude physical topology facts."""

    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise MaterializationError("identity mapping keys must be strings")
            if _normalized_key(key) not in _TOPOLOGY_KEYS:
                projected[key] = _project_identity(nested)
        return projected
    if isinstance(value, (list, tuple)):
        return [_project_identity(nested) for nested in value]
    return value


def seed_labels(stage: str) -> tuple[int, ...]:
    """Return the fixed fresh-seed namespace for one stage."""

    definition = stage_definition(stage)
    start = SEED_NAMESPACE_STARTS[stage]
    return tuple(range(start, start + definition.seeds_per_point))


def seed_namespace(stage: str, label: int, *, cohort: str = "he-importance/v2") -> dict[str, int]:
    """Derive collision-resistant named streams for one scientific seed label.

    A digest, not Python's process-randomized ``hash()``, maps each semantic
    stream to a positive signed-63-bit integer accepted by common RNG APIs.
    """

    if label not in seed_labels(stage):
        raise MaterializationError(f"seed label {label} is outside the {stage} namespace")
    if not cohort:
        raise MaterializationError("seed cohorts must be explicit")
    streams = {
        stream: int.from_bytes(
            sha256(canonical_json({"cohort": cohort, "stage": stage, "label": label, "stream": stream})).digest()[:8],
            "big",
        )
        & ((1 << 63) - 1)
        for stream in SEED_STREAMS
    }
    if 0 in streams.values() or len(set(streams.values())) != len(streams):
        raise MaterializationError("seed namespace collision")
    return streams


def seed_namespaces(stage: str, *, cohort: str = "he-importance/v2") -> dict[int, dict[str, int]]:
    """Return the complete, pairwise-distinct stream namespace for a stage."""

    namespaces = {label: seed_namespace(stage, label, cohort=cohort) for label in seed_labels(stage)}
    signatures = {canonical_json(streams) for streams in namespaces.values()}
    if len(signatures) != len(namespaces):
        raise MaterializationError(f"seed labels in {stage} do not have distinct RNG streams")
    return namespaces


def _absolute_unique_path(output_root: Path, stage: str, digest: str, seen: set[Path]) -> Path:
    root = output_root.resolve()
    if not root.is_absolute():  # pragma: no cover - Path.resolve is absolute by contract.
        raise MaterializationError("output root must resolve to an absolute path")
    path = root / stage / digest
    if path in seen:
        raise MaterializationError(f"output path reused: {path}")
    seen.add(path)
    return path


def materialize_stage(
    stage: str,
    configurations: Iterable[Mapping[str, Any]],
    optimizer: OptimizerCell,
    output_root: Path,
) -> tuple[MaterializedCell, ...]:
    """Materialize one exact resolved union as immutable, content-addressed rows.

    ``configurations`` are literal resolved configurations: each must provide a
    non-empty ``scientific_identity`` and a complete ``payload`` mapping.  A
    duplicate is removed only when its resolved scientific identity is exactly
    equal.  Conflicting definitions of that identity fail rather than choosing
    an arbitrary display-name representative.
    """

    stage_definition(stage)
    namespaces = seed_namespaces(stage)
    identities: dict[str, Mapping[str, Any]] = {}
    seen_paths: set[Path] = set()
    cells: list[MaterializedCell] = []
    for configuration in configurations:
        if frozenset(configuration) != frozenset({"scientific_identity", "payload"}):
            raise MaterializationError("resolved configurations require exactly scientific_identity and payload")
        identity = configuration["scientific_identity"]
        payload = configuration["payload"]
        if not isinstance(identity, Mapping) or not identity:
            raise MaterializationError("scientific_identity must be a non-empty mapping")
        if not isinstance(payload, Mapping):
            raise MaterializationError("payload must be a literal mapping")
        identity_hash = content_hash(identity)
        previous = identities.get(identity_hash)
        if previous is not None:
            if canonical_json(previous) != canonical_json(configuration):
                raise MaterializationError("one scientific identity has conflicting resolved content")
            continue
        identities[identity_hash] = configuration
        optimizer_payload = {"method": optimizer.method, "status": optimizer.status}
        if optimizer.reason is not None:
            optimizer_payload["unavailable_reason"] = optimizer.reason
        for label in seed_labels(stage):
            manifest = {
                "schema": TRAIN_MANIFEST_SCHEMA,
                "stage": stage,
                "scientific_identity": dict(identity),
                "seed_identity": {"stage": stage, "label": label, "namespace": "fresh-training"},
                "payload": {"configuration": dict(payload), "optimizer": optimizer_payload},
            }
            _validate_l2b_materialized_manifest(manifest)
            digest = content_hash(manifest)
            cells.append(
                MaterializedCell(
                    manifest=_freeze(manifest),
                    content_hash=digest,
                    output_path=_absolute_unique_path(output_root, stage, digest, seen_paths),
                    seed_streams=_freeze(namespaces[label]),
                )
            )
    return tuple(cells)


def _validate_l2b_materialized_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate L2b-owned structure without closing caller-owned subtrees.

    L2a closes its own fixture schema.  The scientific identity and payload
    configuration mappings are deliberately delegated because their legal keys
    are caller-owned; L2b owns and closes the seed-identity vocabulary.
    """

    _require_exact_keys(manifest, _TRAIN_KEYS, "materialized manifest")
    if manifest["schema"] != TRAIN_MANIFEST_SCHEMA or not isinstance(manifest["stage"], str):
        raise MaterializationError("materialized manifest has an invalid common coordinate")
    stage_definition(manifest["stage"])
    if not isinstance(manifest["scientific_identity"], Mapping):
        raise MaterializationError("scientific_identity must be a mapping")
    _require_exact_keys(manifest["seed_identity"], _L2B_SEED_IDENTITY_KEYS, "seed_identity")
    if manifest["seed_identity"]["stage"] != manifest["stage"]:
        raise MaterializationError("seed identity stage does not match manifest stage")
    if manifest["seed_identity"]["label"] not in seed_labels(manifest["stage"]):
        raise MaterializationError("seed identity label is outside the stage namespace")
    if manifest["seed_identity"]["namespace"] != "fresh-training":
        raise MaterializationError("seed identity namespace is not declared")
    if not isinstance(manifest["payload"], Mapping):
        raise MaterializationError("payload must be a mapping")
    _require_exact_keys(manifest["payload"], frozenset({"configuration", "optimizer"}), "payload")
    if not isinstance(manifest["payload"]["configuration"], Mapping):
        raise MaterializationError("payload configuration must be a mapping")


def intended_configurations() -> tuple[Mapping[str, Any], ...]:
    """Load the committed, literal L2b configuration enumeration."""

    raw = json.loads(_INTENDED_CONFIGURATIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise MaterializationError("intended configuration inventory must be a list")
    for entry in raw:
        if not isinstance(entry, Mapping) or frozenset(entry) != frozenset(
            {"stage", "optimizer", "configurations"}
        ):
            raise MaterializationError("intended configuration entry has an invalid schema")
    return tuple(raw)


def materialize_intended_configurations(output_root: Path) -> tuple[MaterializedCell, ...]:
    """Materialize every configuration in the committed L2b inventory."""

    cells: list[MaterializedCell] = []
    for entry in intended_configurations():
        optimizer_data = entry["optimizer"]
        if not isinstance(optimizer_data, Mapping):
            raise MaterializationError("intended optimizer must be a mapping")
        optimizer = OptimizerCell(
            method=optimizer_data["method"],
            status=optimizer_data["status"],
            reason=optimizer_data.get("reason"),
        )
        cells.extend(materialize_stage(entry["stage"], entry["configurations"], optimizer, output_root))
    return tuple(cells)


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
    if not isinstance(manifest[TOPOLOGY_KEY], Mapping):
        raise ManifestSchemaError("topology must be a mapping")
    _require_exact_keys(manifest["payload"], _PAYLOAD_KEYS, "payload")
    _require_scalar(manifest["payload"]["updates"], int, "updates")
    if not isinstance(manifest["payload"]["configuration"], Mapping):
        raise ManifestSchemaError("payload.configuration must be a mapping")
    _validate_delegated_subtree(manifest["scientific_identity"], "scientific_identity")
    _validate_delegated_subtree(manifest["seed_identity"], "seed_identity")
    _validate_delegated_subtree(manifest["payload"]["configuration"], "payload.configuration")
    _validate_delegated_subtree(manifest[TOPOLOGY_KEY], TOPOLOGY_KEY)


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
