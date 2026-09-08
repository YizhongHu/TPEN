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
from decimal import Decimal, InvalidOperation
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable
from types import MappingProxyType
import warnings


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
_REFERENCE_ENERGY_DIGITS = _REFERENCE_ENERGY_TEXT.removeprefix("-").replace(".", "")
_RANKING_INPUT_SCHEMA = {"statistic": None}
_INDEPENDENT_SAMPLER_INPUT_SCHEMA = {
    "sampler": {"walkers": None},
    "walkers": None,
    "burn_in_sweeps": None,
}
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


@dataclass(frozen=True)
class CheckpointCadence:
    """Science-selected dense checkpoint observations.

    Parameters
    ----------
    every_n_updates
        Dense persistence cadence. This is scientific study input, not a
        scheduler setting.
    selected_updates
        Immutable checkpoint observations to materialize into evaluation
        packets. Their selection is fixed before training and cannot be
        replaced by an in-training callback.
    """

    every_n_updates: int
    selected_updates: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.every_n_updates <= 0:
            raise MaterializationError("checkpoint cadence must be positive")
        if not self.selected_updates:
            raise MaterializationError("at least one checkpoint observation is required")
        if tuple(sorted(set(self.selected_updates))) != self.selected_updates:
            raise MaterializationError("selected checkpoints must be unique and increasing")
        if any(update <= 0 or update % self.every_n_updates for update in self.selected_updates):
            raise MaterializationError("selected checkpoints must lie on the dense cadence")

    def science_parameters(self) -> Mapping[str, Any]:
        """Return the immutable scientific checkpoint parameter block."""

        return _freeze(
            {
                "every_n_updates": self.every_n_updates,
                "selected_updates": list(self.selected_updates),
            }
        )


@dataclass(frozen=True)
class TrainingPacket:
    """One train packet; construction accepts no callback operation."""

    cell: MaterializedCell
    checkpoint_cadence: CheckpointCadence
    ddp_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class RankingPacket:
    """Cheap ranking packet with construction-time-only metric restrictions.

    This describes packet construction, not the runtime behavior of an
    evaluator. Runtime emission enforcement belongs to the evaluator layer.
    """

    checkpoint_path: Path
    source_content_hash: str
    ranking_inputs: Mapping[str, Any]
    emitted_metrics: tuple["RankingStatistic", ...]
    ddp_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class IndependentSamplerTestPacket:
    """Expensive test packet fed only immutable independent-sampler inputs."""

    checkpoint_path: Path
    source_content_hash: str
    independent_sampler_inputs: Mapping[str, Any]
    ddp_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class PacketMaterialization:
    """Separate train, ranking, and independent-sampler packet collections."""

    train: tuple[TrainingPacket, ...]
    ranking: tuple[RankingPacket, ...]
    independent_sampler_test: tuple[IndependentSamplerTestPacket, ...]


class RankingStatistic(Enum):
    """The closed, non-energy statistics available to a ranking packet."""

    LOGABS_VARIANCE = "logabs_variance"
    ACCEPTANCE_RATE = "acceptance_rate"


def canonical_json(value: Any) -> bytes:
    """Return canonical identity bytes for a value or an HI train manifest.

    The L2a-designated topology member is structurally excluded only from a
    manifest root.  A caller-owned mapping passed alone may legitimately use
    the same word as a scientific fact, so it remains literal content.
    """

    return _canonical_json(value, exclude_root_topology=_is_manifest_root(value))


def _canonical_json(value: Any, *, exclude_root_topology: bool) -> bytes:
    """Serialize an identity with an explicit structural-boundary decision."""

    try:
        return json.dumps(
            _project_identity(value, exclude_root_topology=exclude_root_topology),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except MaterializationError:
        raise
    except (TypeError, ValueError) as error:
        raise MaterializationError("identity values must be finite JSON data") from error


def content_hash(value: Any) -> str:
    """Return the SHA-256 identity of canonical literal content."""

    return sha256(canonical_json(value)).hexdigest()


def _is_manifest_root(value: Any) -> bool:
    """Recognize the sole L2a-owned position where topology is execution data."""

    return (
        isinstance(value, Mapping)
        and value.get("schema") in {TRAIN_MANIFEST_SCHEMA, EVALUATION_MANIFEST_SCHEMA}
        and TOPOLOGY_KEY in value
    )


def _freeze(value: Any) -> Any:
    """Recursively freeze a JSON-shaped manifest after it has been hashed."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(nested) for key, nested in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(nested) for nested in value)
    return value


def _project_identity(value: Any, *, exclude_root_topology: bool, is_root: bool = True) -> Any:
    """Convert frozen containers to JSON, omitting only root execution topology.

    L2a declares ``topology`` as the one top-level execution subtree.  Its
    contents are deliberately unenumerable: this structural boundary removes
    the whole subtree, while an identically named nested caller fact remains
    scientific content and therefore contributes to the identity.
    """

    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise MaterializationError("identity mapping keys must be strings")
            if is_root and exclude_root_topology and key == TOPOLOGY_KEY:
                continue
            projected[key] = _project_identity(nested, exclude_root_topology=False, is_root=False)
        return projected
    if isinstance(value, (list, tuple)):
        return [_project_identity(nested, exclude_root_topology=False, is_root=False) for nested in value]
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
    non-empty ``scientific_identity``, a complete ``payload`` mapping, and the
    caller-supplied execution ``topology`` mapping.  A
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
        if frozenset(configuration) != frozenset({"scientific_identity", "payload", TOPOLOGY_KEY}):
            raise MaterializationError(
                "resolved configurations require exactly scientific_identity, payload, and topology"
            )
        identity = configuration["scientific_identity"]
        payload = configuration["payload"]
        topology = configuration[TOPOLOGY_KEY]
        if not isinstance(identity, Mapping) or not identity:
            raise MaterializationError("scientific_identity must be a non-empty mapping")
        if not isinstance(payload, Mapping):
            raise MaterializationError("payload must be a literal mapping")
        if not isinstance(topology, Mapping):
            raise MaterializationError("topology must be a literal mapping")
        identity_hash = content_hash(identity)
        previous = identities.get(identity_hash)
        if previous is not None:
            if _canonical_json(previous, exclude_root_topology=True) != _canonical_json(configuration, exclude_root_topology=True):
                raise MaterializationError("one scientific identity has conflicting resolved content")
            if _canonical_json(previous, exclude_root_topology=False) != _canonical_json(configuration, exclude_root_topology=False):
                warnings.warn(
                    f"execution-topology collision for scientific identity {identity_hash}: "
                    f"{previous!r} and {configuration!r}",
                    stacklevel=2,
                )
            continue
        identities[identity_hash] = configuration
        optimizer_identity = {"method": optimizer.method, "status": optimizer.status}
        if optimizer.reason is not None:
            optimizer_identity["unavailable_reason"] = optimizer.reason
        for label in seed_labels(stage):
            if not isinstance(payload.get("updates"), int):
                raise MaterializationError("payload updates must be an integer")
            manifest = {
                "schema": TRAIN_MANIFEST_SCHEMA,
                "stage": stage,
                "scientific_identity": {**identity, "optimizer_cell": optimizer_identity},
                "seed_identity": {"stage": stage, "label": label, "namespace": "fresh-training"},
                "payload": {"updates": payload["updates"], "configuration": dict(payload)},
                TOPOLOGY_KEY: topology,
            }
            validate_materialized_manifest(manifest)
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


def _immutable_packet_inputs(
    inputs: Mapping[str, Any],
    label: str,
    *,
    schema: Mapping[str, Any] | None = None,
    require_nonempty: bool = True,
) -> Mapping[str, Any]:
    """Screen then freeze a packet input block before it is handed to a job."""

    if not isinstance(inputs, Mapping) or (require_nonempty and not inputs):
        requirement = "a non-empty mapping" if require_nonempty else "a mapping"
        raise MaterializationError(f"{label} must be {requirement}")
    # Schema refusal precedes content screening: clause-2 reach probes must use
    # accepted keys, or an unknown reference-bearing key reports only the schema error.
    if schema is not None:
        _validate_packet_input_schema(inputs, schema, label)
    _refuse_packet_content(inputs, label)
    try:
        canonical_json(inputs)
    except MaterializationError as error:
        raise MaterializationError(f"{label} must contain finite JSON data") from error
    return _freeze(dict(inputs))


def _checkpoint_path(cell: MaterializedCell, update: int) -> Path:
    """Name one immutable observation beneath its content-addressed train row."""

    return cell.output_path / "checkpoints" / f"update-{update:08d}"


def _refuse_packet_content(value: Any, label: str) -> None:
    """Refuse callable and reference-bearing packet values recursively.

    Exact reference values and textual decimal truncations with at least seven
    significant digits are refused through frozen mappings and sequences.
    Numerically perturbed values are not inferable from their representation.
    The primary input-key allowlists limit which packet-input channels exist;
    this backstop makes no false claim to recognize perturbed values there.
    """

    if callable(value):
        raise MaterializationError(f"{label} cannot contain a callable")
    if _is_reference_energy_representation(value):
        raise MaterializationError(f"{label} contains the evaluation reference energy")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise MaterializationError(f"{label} keys must be strings")
            _refuse_packet_content(nested, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _refuse_packet_content(nested, f"{label}[{index}]")


def _is_reference_energy_representation(value: Any) -> bool:
    """Recognize decimal agreement with the reference at seven-plus figures."""

    if type(value) is float:
        if value == _REFERENCE_ENERGY:
            return True
        text = repr(value)
    elif type(value) is str:
        text = value
    else:
        return False
    try:
        decimal = Decimal(text)
    except (InvalidOperation, ValueError):
        return False
    if decimal >= 0 or not decimal.is_finite():
        return False
    digits = decimal.as_tuple().digits
    first = next((index for index, digit in enumerate(digits) if digit), len(digits))
    significant = len(digits) - first
    if significant < 7:
        return False
    reference = Decimal(_REFERENCE_ENERGY_TEXT)
    return decimal.adjusted() == reference.adjusted() and digits[first : first + 7] == reference.as_tuple().digits[:7]


def _validate_packet_input_schema(value: Any, schema: Mapping[str, Any], label: str) -> None:
    """Refuse undeclared mappings; ``None`` is a declared caller-open value."""

    if not isinstance(value, Mapping):
        raise MaterializationError(f"{label} must be a mapping")
    for key, nested in value.items():
        if not isinstance(key, str):
            raise MaterializationError(f"{label} keys must be strings")
        if key not in schema:
            raise MaterializationError(f"{label} contains unknown input key {key!r}")
        child_schema = schema[key]
        if child_schema is not None and isinstance(nested, Mapping):
            _validate_packet_input_schema(nested, child_schema, f"{label}.{key}")


def _validate_packet_cell(cell: MaterializedCell) -> StageDefinition:
    """Validate the train artifact embedded wholesale in a train packet."""

    _refuse_packet_content(cell.manifest, "cell manifest")
    validate_materialized_manifest(cell.manifest)
    if content_hash(cell.manifest) != cell.content_hash:
        raise MaterializationError("cell content hash does not bind its manifest")
    return stage_definition(cell.manifest["stage"])


def materialize_job_packets(
    cells: Iterable[MaterializedCell],
    checkpoint_cadence: CheckpointCadence,
    ranking_inputs: Mapping[str, Any],
    ranking_metrics: Iterable[RankingStatistic],
    independent_sampler_inputs: Mapping[str, Any],
    *,
    ddp_provenance: Mapping[str, Any],
) -> PacketMaterialization:
    """Materialize the three disjoint HI job-packet classes.

    The returned packets deliberately contain no scheduler, facility, device,
    or launch defaults. Those are operator-supplied execution concerns. DDP
    information is retained as provenance only and is not used to derive a
    scientific identity or packet multiplicity. This construction boundary
    does not establish a cheap evaluator's runtime behavior.
    """

    frozen_ranking_inputs = _immutable_packet_inputs(
        ranking_inputs, "ranking inputs", schema=_RANKING_INPUT_SCHEMA
    )
    frozen_independent_inputs = _immutable_packet_inputs(
        independent_sampler_inputs,
        "independent sampler inputs",
        schema=_INDEPENDENT_SAMPLER_INPUT_SCHEMA,
    )
    frozen_ddp_provenance = _immutable_packet_inputs(
        ddp_provenance, "DDP provenance", require_nonempty=False
    )
    metrics = tuple(ranking_metrics)
    if not metrics or any(type(metric) is not RankingStatistic for metric in metrics):
        raise MaterializationError("ranking metrics must be RankingStatistic values")

    train: list[TrainingPacket] = []
    ranking: list[RankingPacket] = []
    independent_sampler_test: list[IndependentSamplerTestPacket] = []
    seen_cells: set[str] = set()
    for cell in cells:
        stage = _validate_packet_cell(cell)
        if stage.updates is None:
            raise MaterializationError("packet stages require a fixed training horizon")
        if any(update > stage.updates for update in checkpoint_cadence.selected_updates):
            raise MaterializationError("selected checkpoints exceed the stage horizon")
        if cell.content_hash in seen_cells:
            raise MaterializationError("train cells must be unique by content identity")
        seen_cells.add(cell.content_hash)
        train.append(
            TrainingPacket(
                cell=cell,
                checkpoint_cadence=checkpoint_cadence,
                ddp_provenance=frozen_ddp_provenance,
            )
        )
        for update in checkpoint_cadence.selected_updates:
            checkpoint_path = _checkpoint_path(cell, update)
            ranking.append(
                RankingPacket(
                    checkpoint_path=checkpoint_path,
                    source_content_hash=cell.content_hash,
                    ranking_inputs=frozen_ranking_inputs,
                    emitted_metrics=metrics,
                    ddp_provenance=frozen_ddp_provenance,
                )
            )
            independent_sampler_test.append(
                IndependentSamplerTestPacket(
                    checkpoint_path=checkpoint_path,
                    source_content_hash=cell.content_hash,
                    independent_sampler_inputs=frozen_independent_inputs,
                    ddp_provenance=frozen_ddp_provenance,
                )
            )
    return PacketMaterialization(
        train=tuple(train),
        ranking=tuple(ranking),
        independent_sampler_test=tuple(independent_sampler_test),
    )


def validate_materialized_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the L2b materializer interface on top of L2a's firewall.

    L2a's train validator owns structural sanity and reference-content screening
    at every depth. Scientific identity and payload configuration keys remain
    caller-owned; L2b owns and closes only seed identity.
    """

    validate_train_manifest(manifest)
    _require_exact_keys(manifest["seed_identity"], _L2B_SEED_IDENTITY_KEYS, "seed_identity")
    if manifest["seed_identity"]["stage"] != manifest["stage"]:
        raise MaterializationError("seed identity stage does not match manifest stage")
    if manifest["seed_identity"]["label"] not in seed_labels(manifest["stage"]):
        raise MaterializationError("seed identity label is outside the stage namespace")
    if manifest["seed_identity"]["namespace"] != "fresh-training":
        raise MaterializationError("seed identity namespace is not declared")


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
