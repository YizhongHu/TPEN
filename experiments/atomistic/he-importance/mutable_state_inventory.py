"""Closed mutable-state inventory for the HI training configuration.

The inventory is deliberately derived from the committed Hydra tree rather
than maintained as a second list. A newly constructed target therefore fails
closed until its recovery ownership is classified here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_CONFIG_PATH = Path(__file__).with_name("configs") / "train.yaml"
_RESTORED = "restored", "native train_resume owns the corresponding state"
_RECONSTRUCTIBLE = (
    "reconstructible",
    "immutable construction policy is rebuilt from the committed HI configuration; mutable trained state is owned by TrainResume",
)
_TYPED_REFUSED = (
    "typed-refused",
    "HI study policy keeps this mutable owner typed-refused pending S5b qualification; core admits it after CX-07, but contract-repin-20261118 and S5b ddb2794a require HI continuation refusal before construction",
)
_ALLOCATION_LOCAL = (
    "allocation-local",
    "the measurement is local to one process allocation",
)


def _registry_entries(
    paths: Sequence[tuple[str, str]], disposition: tuple[str, str]
) -> dict[tuple[str, str], tuple[str, str]]:
    """Expand an explicit constructor list into the sealed registry."""

    return {key: disposition for key in paths}


# This is intentionally a path-and-target registry, rather than a target or
# path-prefix allowlist.  The raw and O-composite trees are both listed so a
# new nested Hydra constructor cannot acquire an accidental disposition.
_REGISTRY: dict[tuple[str, str], tuple[str, str]] = {
    **_registry_entries(
        [
            ("sampler", "tpen.sampling.metropolis.MetropolisSampler"),
            ("trainer", "tpen.training.trainer.VMCTrainer"),
        ],
        _RECONSTRUCTIBLE,
    ),
    **_registry_entries(
        [
            ("callbacks.8", "tpen.callback.RuntimeEquivariance"),
            ("callbacks.10", "tpen.callback.ResourceUsage"),
        ],
        _TYPED_REFUSED,
    ),
    (
        "callbacks.9.payload",
        "tpen.checkpoint.TrainResume",
    ): _RECONSTRUCTIBLE,
    ("callbacks.10.allocator_probe", "tpen.accelerator.TorchAllocatorPeakProbe"): _ALLOCATION_LOCAL,
}

_RAW_RECONSTRUCTIBLE = [
    ("atoms", "tpen.data.atomic_configuration.AtomicConfiguration"),
    ("atoms.positions", "torch.tensor"), ("atoms.charges", "torch.tensor"),
    ("runner", "tpen.runner.Train"), ("model", "tpen.nn.TPENWaveFunction"),
    ("model.embedding", "tpen.nn.Embedding"),
    ("model.embedding.initializer", "tpen.nn.initialization.TorchInitializer"),
    ("model.layers.0", "tpen.nn.TPENLayer"),
    ("model.layers.0.mixing", "tpen.nn.EquivariantMixing"),
    ("model.layers.0.mixing.activation", "torch.nn.SiLU"),
    ("model.layers.0.path_aggregation", "tpen.nn.PathAggregation"),
    ("model.layers.0.path_aggregation.activation", "torch.nn.SiLU"),
    ("model.layers.0.path_aggregation.initializer", "tpen.nn.initialization.TorchInitializer"),
    ("model.layers.0.update", "tpen.nn.ResidualUpdater"),
    ("model.factors.0", "tpen.nn.ElectronElectronCusp"),
    ("model.factors.1", "tpen.nn.ElectronNucleusCusp"),
    ("model.factors.1.law", "tpen.nn.TailSafeElectronNucleusCuspLaw"),
    ("model.readout", "tpen.nn.readout.PfaffianReadout"),
    ("hamiltonian_terms.kinetic", "tpen.physics.kinetic.KineticEnergy"),
    ("hamiltonian_terms.electron_nucleus", "tpen.physics.potential.ElectronNucleusPotential"),
    ("hamiltonian_terms.electron_electron", "tpen.physics.potential.ElectronElectronInteraction"),
    ("hamiltonian_terms.nucleus_nucleus", "tpen.physics.potential.NucleusNucleusPotential"),
    ("sampler.nuclear_positions", "torch.tensor"), ("sampler.nuclear_charges", "torch.tensor"),
    ("optimizer", "torch.optim.Adam"),
    ("callbacks.0", "tpen.callback.ConfigSnapshot"), ("callbacks.1", "tpen.callback.ResolvedConfigSnapshot"),
    ("callbacks.2", "tpen.callback.Metadata"), ("callbacks.3", "tpen.callback.Status"),
    ("callbacks.4", "tpen.callback.DataIntegrity"), ("callbacks.5", "tpen.callback.GradientStats"),
    ("callbacks.6", "tpen.callback.SamplerHealth"), ("callbacks.7", "tpen.callback.FactorScalars"),
    ("callbacks.8.checkers.0", "tpen.equivariance.checks.FullModelEquivarianceChecker"),
    ("callbacks.8.checkers.1", "tpen.equivariance.checks.TraceEquivarianceChecker"),
    ("callbacks.9", "tpen.callback.Checkpoint"), ("callbacks.9.schedule", "tpen.checkpoint.EveryNUpdates"),
    ("loggers.0", "tpen.logging.CSV"), ("loggers.1", "tpen.logging.JSONL"),
]
_REGISTRY.update(_registry_entries(_RAW_RECONSTRUCTIBLE, _RECONSTRUCTIBLE))
_REGISTRY.update(
    _registry_entries(
        [
            ("model.embedding", "tpen.nn.embedding.matched_start_embedding"),
            ("model.embedding.raw", "tpen.nn.Embedding"),
            ("model.embedding.raw.initializer", "tpen.nn.initialization.TorchInitializer"),
            ("model.embedding.augmented", "tpen.nn.Embedding"),
            ("model.embedding.augmented.initializer", "tpen.nn.initialization.TorchInitializer"),
            ("model.layers.0.layout", "tpen.data.paths.compose_path_layout"),
            ("model.layers.0.layout.linear", "tpen.data.paths.LinearPathMetadata.generate"),
            ("model.layers.0.layout.linear.explicit.0", "tpen.data.paths.SupportPath"),
            ("model.layers.0.layout.linear.explicit.1", "tpen.data.paths.SupportPath"),
            ("model.layers.0.layout.linear.explicit.2", "tpen.data.paths.SupportPath"),
            *( (f"model.layers.0.layout.linear.explicit.{index}", "tpen.data.paths.SupportPath") for index in range(3, 8) ),
            ("model.layers.0.layout.tensor_product", "tpen.data.paths.PathMetadata.generate"),
            ("model.layers.0.layout.input_orders", "tpen.data.paths.NormalizedOrders"),
            ("model.layers.0.layout.output_orders", "tpen.data.paths.NormalizedOrders"),
            ("model.layers.0.layout.input_channels", "tpen.data.paths.NormalizedChannels"),
            ("model.layers.0.layout.output_channels", "tpen.data.paths.NormalizedChannels"),
            ("model.layers.0.mixing", "tpen.nn.CompositeMixing"),
            ("model.layers.0.mixing.producers.0", "tpen.nn.LinearEquivariantMixing"),
            ("model.layers.0.mixing.producers.1", "tpen.nn.EquivariantMixing"),
            ("model.layers.0.mixing.activation", "tpen.nn.ChannelPreservingMLPActivation"),
            ("model.layers.0.mixing.activation", "torch.nn.Tanh"),
            ("model.layers.0.mixing.activation", "torch.nn.GELU"),
            ("model.layers.0.mixing.activation", "tpen.nn.SoftplusBeta"),
            ("model.layers.0.mixing.activation.layout", "tpen.nn.OrderMLPLayout"),
            ("model.layers.0.mixing.activation.layout.axes", "tpen.nn.ChannelActivationAxes"),
            ("model.layers.0.mixing.activation.layout.specs.0", "tpen.nn.OrderMLPSpec"),
            ("model.layers.0.mixing.activation.layout.specs.0.activation", "torch.nn.SiLU"),
            ("model.layers.0.mixing.activation.layout.specs.1", "tpen.nn.OrderMLPSpec"),
            ("model.layers.0.mixing.activation.layout.specs.1.activation", "torch.nn.SiLU"),
            ("model.layers.0.mixing.activation.initializer", "tpen.nn.TorchInitializer"),
            ("model.layers.0.path_aggregation.activation", "tpen.nn.ChannelPreservingMLPActivation"),
            ("model.layers.0.path_aggregation.activation", "torch.nn.Tanh"),
            ("model.layers.0.path_aggregation.activation", "torch.nn.GELU"),
            ("model.layers.0.path_aggregation.activation", "tpen.nn.SoftplusBeta"),
            ("model.layers.0.path_aggregation.activation.layout", "tpen.nn.OrderMLPLayout"),
            ("model.layers.0.path_aggregation.activation.layout.axes", "tpen.nn.ChannelActivationAxes"),
            ("model.layers.0.path_aggregation.activation.layout.specs.0", "tpen.nn.OrderMLPSpec"),
            ("model.layers.0.path_aggregation.activation.layout.specs.0.activation", "torch.nn.SiLU"),
            ("model.layers.0.path_aggregation.activation.layout.specs.1", "tpen.nn.OrderMLPSpec"),
            ("model.layers.0.path_aggregation.activation.layout.specs.1.activation", "torch.nn.SiLU"),
            ("model.layers.0.path_aggregation.activation.initializer", "tpen.nn.TorchInitializer"),
            ("model.layers.0.path_aggregation.initializer", "tpen.nn.TorchInitializer"),
            ("model.factors.2", "tpen.nn.jastrow.BoundedTwoCoefficientJastrow"),
            ("model.basis", "tpen.nn.basis.BoundedDistanceBasis"), ("model.basis.inner", "tpen.nn.RawCoordinateBasis"),
        ],
        _RECONSTRUCTIBLE,
    )
)

# Resolved family rows retain the same objects under ``runner.*`` aliases.
_REGISTRY.update(
    {
        (f"runner.{path}", target): disposition
        for (path, target), disposition in tuple(_REGISTRY.items())
        if path in {"model", "sampler", "trainer", "optimizer"}
    }
)
_REGISTRY.update(
    _registry_entries(
        [
            ("model.layers.0.mixing.layout", "tpen.data.paths.compose_path_layout"),
            ("model.layers.0.mixing.layout.linear", "tpen.data.paths.LinearPathMetadata.generate"),
            ("model.layers.0.mixing.layout.linear.explicit.0", "tpen.data.paths.SupportPath"),
            ("model.layers.0.mixing.layout.linear.explicit.1", "tpen.data.paths.SupportPath"),
            ("model.layers.0.mixing.layout.linear.explicit.2", "tpen.data.paths.SupportPath"),
            *( (f"model.layers.0.mixing.layout.linear.explicit.{index}", "tpen.data.paths.SupportPath") for index in range(3, 8) ),
            ("model.layers.0.mixing.layout.tensor_product", "tpen.data.paths.PathMetadata.generate"),
            ("model.layers.0.mixing.layout.input_orders", "tpen.data.paths.NormalizedOrders"),
            ("model.layers.0.mixing.layout.output_orders", "tpen.data.paths.NormalizedOrders"),
            ("model.layers.0.mixing.layout.input_channels", "tpen.data.paths.NormalizedChannels"),
            ("model.layers.0.mixing.layout.output_channels", "tpen.data.paths.NormalizedChannels"),
            ("model.layers.0.mixing.producers.0", "tpen.nn.EquivariantMixing"),
            ("model.layers.0.mixing.producers.0.metadata", "tpen.data.paths.LinearPathMetadata.generate"),
            ("model.layers.0.mixing.producers.0.metadata.explicit.0", "tpen.data.paths.SupportPath"),
            ("model.layers.0.mixing.producers.0.metadata.explicit.1", "tpen.data.paths.SupportPath"),
            ("model.layers.0.mixing.producers.0.metadata.explicit.2", "tpen.data.paths.SupportPath"),
            *( (f"model.layers.0.mixing.producers.0.metadata.explicit.{index}", "tpen.data.paths.SupportPath") for index in range(3, 8) ),
            ("model.layers.0.mixing.producers.0.paths", "tpen.data.paths.PathMetadata.generate"),
            ("model.layers.0.mixing.producers.1.paths", "tpen.data.paths.PathMetadata.generate"),
            ("model.layers.0.path_aggregation.layout", "tpen.data.paths.compose_path_layout"),
            ("model.layers.0.path_aggregation.layout.linear", "tpen.data.paths.LinearPathMetadata.generate"),
            ("model.layers.0.path_aggregation.layout.linear.explicit.0", "tpen.data.paths.SupportPath"),
            ("model.layers.0.path_aggregation.layout.linear.explicit.1", "tpen.data.paths.SupportPath"),
            ("model.layers.0.path_aggregation.layout.linear.explicit.2", "tpen.data.paths.SupportPath"),
            *( (f"model.layers.0.path_aggregation.layout.linear.explicit.{index}", "tpen.data.paths.SupportPath") for index in range(3, 8) ),
            ("model.layers.0.path_aggregation.layout.tensor_product", "tpen.data.paths.PathMetadata.generate"),
            ("model.layers.0.path_aggregation.layout.input_orders", "tpen.data.paths.NormalizedOrders"),
            ("model.layers.0.path_aggregation.layout.output_orders", "tpen.data.paths.NormalizedOrders"),
            ("model.layers.0.path_aggregation.layout.input_channels", "tpen.data.paths.NormalizedChannels"),
            ("model.layers.0.path_aggregation.layout.output_channels", "tpen.data.paths.NormalizedChannels"),
            ("model.factors.1.atoms", "tpen.data.atomic_configuration.AtomicConfiguration"),
            ("model.factors.1.atoms.positions", "torch.tensor"),
            ("model.factors.1.atoms.charges", "torch.tensor"),
            ("hamiltonian_terms.electron_nucleus.atoms", "tpen.data.atomic_configuration.AtomicConfiguration"),
            ("hamiltonian_terms.electron_nucleus.atoms.positions", "torch.tensor"),
            ("hamiltonian_terms.electron_nucleus.atoms.charges", "torch.tensor"),
            ("hamiltonian_terms.nucleus_nucleus.atoms", "tpen.data.atomic_configuration.AtomicConfiguration"),
            ("hamiltonian_terms.nucleus_nucleus.atoms.positions", "torch.tensor"),
            ("hamiltonian_terms.nucleus_nucleus.atoms.charges", "torch.tensor"),
            ("model.layers.0.update", "tpen.nn.ReplaceUpdater"),
            ("model.layers.0.update_envelope", "tpen.nn.coordinate_envelopes.GaussianCoordinateEnvelope"),
        ],
        _RECONSTRUCTIBLE,
    )
)

# These are the mutable recovery owners named by the native checkpoint
# contract.  They are inventory records, not Hydra constructors.
_TRAIN_RESUME_STATE = (
    ("state.model", "TrainResume.model", "model"),
    ("state.optimizer", "TrainResume.optimizer", "optimizer"),
    ("state.trainer", "TrainResume.trainer", "trainer"),
    ("state.sampler", "TrainResume.sampler", "sampler"),
    ("state.process_rng", "TrainResume.rng", "rng"),
)
_ATTEMPT_LOCAL_STATE = (
    ("state.loggers", "attempt-local.loggers", "loggers"),
    ("state.checkpoint_publication", "attempt-local.checkpoint_publication", "checkpoint publication fields"),
)


@dataclass(frozen=True, slots=True)
class MutableStateEntry:
    """One constructed HI object and its explicit continuation disposition."""

    path: str
    target: str
    classification: str
    reason: str
    source: str


class HIContinuationAdmissionRefusal(ValueError):
    """A sealed HI continuation row does not satisfy the study contract."""

    code = "hi-continuation-admission-refused"

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


class HIContinuationRowIdentityRefusal(HIContinuationAdmissionRefusal):
    """A supplied config is not the producer-resolved row being admitted."""


@dataclass(frozen=True, slots=True)
class SealedHIContinuationIdentity:
    """Stable identity for one inventory- and context-qualified HI row."""

    schema: str
    stage: str
    context: str
    content_hash: str
    constructor_hash: str

    def as_dict(self) -> dict[str, str]:
        """Return the identity in a JSON-shaped, deterministic form."""

        return {
            "schema": self.schema,
            "stage": self.stage,
            "context": self.context,
            "content_hash": self.content_hash,
            "constructor_hash": self.constructor_hash,
        }


@dataclass(frozen=True, slots=True)
class HIContinuationAdmission:
    """Validated study-side admission facts for a future continuation."""

    identity: SealedHIContinuationIdentity
    inventory: tuple[MutableStateEntry, ...]


_ADMISSION_SCHEMA = "he-importance/continuation-admission/v1"
_O_CONTEXTS = {
    "O-raw": {"name": "O-raw", "overrides": {"input_features": "B00", "feature_update": "residual"}},
    "O-composite": {
        "name": "O-composite",
        "overrides": {
            "producer_policy": "hybrid/coordinate-neighbor",
            "aggregation_activation": "channel-MLP at both activation sites",
            "input_features": "B11",
            "feature_update": "residual",
        },
    },
}

# S5b-A admits no operational overlays.  S5b-B may extend this only under its
# own contract; restore_request and artifact_destination are outside this tree.
_SCIENTIFIC_CONTEXT_OVERLAY_ALLOWLIST: frozenset[str] = frozenset()


def _source_ref(path: str, target: str) -> str:
    """Return a stable train.yaml line reference for a config path."""

    lines = _CONFIG_PATH.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if "_target_:" in line and target in line:
            return f"{_CONFIG_PATH}:line {number}"
    return f"{_CONFIG_PATH}:path {path}"


def _classification(path: str, target: str) -> tuple[str, str]:
    """Look up one target in the sealed HI constructor registry."""

    try:
        return _REGISTRY[(path, target)]
    except KeyError as error:
        raise ValueError(
            f"unclassified HI constructor: path={path!r}, target={target!r}"
        ) from error


def _walk_targets(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Enumerate every Hydra constructor target in a config tree."""

    found: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        target = value.get("_target_")
        if isinstance(target, str) and target:
            found.append((path, target))
        for key, nested in value.items():
            child = f"{path}.{key}" if path else str(key)
            found.extend(_walk_targets(nested, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            child = f"{path}.{index}" if path else str(index)
            found.extend(_walk_targets(nested, child))
    return found


def _config_tree(config: Any | None) -> Mapping[str, Any]:
    """Convert an OmegaConf or mapping input to a constructor tree.

    Parameters
    ----------
    config : object, optional
        HI configuration; ``None`` composes the committed base tree.

    Returns
    -------
    Mapping
        Plain mapping containing the constructor tree.

    Raises
    ------
    TypeError
        If ``config`` is not mapping-like.
    """
    if config is None:
        from . import train_config

        config = train_config._compose_base_config()
    if isinstance(config, DictConfig) or OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=False)
    if not isinstance(config, Mapping):
        raise TypeError("HI mutable-state inventory requires a mapping config")
    return config


def build_mutable_state_inventory(config: Any | None = None) -> tuple[MutableStateEntry, ...]:
    """Enumerate and classify every constructor in the HI train config.

    Parameters
    ----------
    config : object, optional
        HI configuration to inspect.

    Returns
    -------
    tuple of MutableStateEntry
        Explicit constructor and native-state ownership records.

    Raises
    ------
    ValueError
        If a constructor is not in the sealed registry or the tree is empty.
    """

    tree = _config_tree(config)
    entries: list[MutableStateEntry] = []
    for path, target in _walk_targets(tree):
        try:
            classification, reason = _classification(path, target)
        except ValueError:
            # ``runner.*`` is a producer alias emitted by resolve_train_config;
            # accept only an exact registered target at its canonical path,
            # including nested model/sampler constructors.
            if path.startswith("runner."):
                classification, reason = _classification(path[7:], target)
            else:
                raise
        entries.append(
            MutableStateEntry(
                path=path,
                target=target,
                classification=classification,
                reason=reason,
                source=_source_ref(path, target),
            )
        )
    if not entries:
        raise ValueError("HI train config constructs no inventory objects")
    entries.extend(
        MutableStateEntry(
            path=path,
            target=target,
            classification="restored",
            reason=f"native TrainResume component {component!r} from tpen/checkpoint/save.py and restore.py",
            source="tpen/checkpoint/save.py:171-196; tpen/checkpoint/restore.py:203-208",
        )
        for path, target, component in _TRAIN_RESUME_STATE
    )
    entries.extend(
        MutableStateEntry(
            path=path,
            target=target,
            classification="allocation-local",
            reason=f"{component} is regenerated for each training attempt",
            source="tpen/callback/checkpoint.py:228-283,410-417; tpen/logging/csv.py:33-52",
        )
        for path, target, component in _ATTEMPT_LOCAL_STATE
    )
    return tuple(entries)


def validate_mutable_state_inventory(config: Any | None = None) -> tuple[MutableStateEntry, ...]:
    """Return the inventory, refusing any unclassified object.

    Parameters
    ----------
    config : object, optional
        HI configuration to validate.

    Returns
    -------
    tuple of MutableStateEntry
        Validated inventory records.

    Raises
    ------
    ValueError
        If a constructor or disposition is not explicitly admitted.
    """

    entries = build_mutable_state_inventory(config)
    allowed = {"restored", "reconstructible", "allocation-local", "typed-refused"}
    invalid = [entry for entry in entries if entry.classification not in allowed]
    if invalid:
        raise ValueError(f"unclassified HI mutable-state object(s): {invalid!r}")
    return entries


def refused_inventory_entries(config: Any | None = None) -> tuple[MutableStateEntry, ...]:
    """Return typed-refused entries used in the stable refusal reason.

    Parameters
    ----------
    config : object, optional
        HI configuration to inspect.

    Returns
    -------
    tuple of MutableStateEntry
        Entries that prevent continuation qualification.

    Raises
    ------
    ValueError
        If the inventory is not closed.
    """

    return tuple(
        entry
        for entry in validate_mutable_state_inventory(config)
        if entry.classification == "typed-refused"
    )


def _canonical_inventory(entries: Sequence[MutableStateEntry]) -> tuple[dict[str, str], ...]:
    """Return the path, target, and disposition fields used for sealing."""

    return tuple(
        {
            "path": entry.path,
            "target": entry.target,
            "classification": entry.classification,
        }
        for entry in sorted(entries, key=lambda item: (item.path, item.target, item.classification))
    )


def _canonical_scientific_context_tree(config: Any) -> bytes:
    """Serialize the complete resolved scientific/context config tree.

    The overlay allowlist is intentionally empty for S5b-A, so every value in
    the producer-resolved tree is identity-significant.  JSON encoding keeps
    booleans, integers, and other JSON scalar types distinct during equality.
    """

    if isinstance(config, DictConfig) or OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise TypeError("HI continuation config must be a mapping")
    tree = dict(config)
    for path in _SCIENTIFIC_CONTEXT_OVERLAY_ALLOWLIST:
        cursor: Any = tree
        segments = path.split(".")
        for segment in segments[:-1]:
            if not isinstance(cursor, dict) or segment not in cursor:
                break
            cursor = cursor[segment]
        else:
            if isinstance(cursor, dict):
                cursor.pop(segments[-1], None)
    return json.dumps(
        tree,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _admission_context(source: Any) -> tuple[str, str]:
    """Read and validate the O1 context identity from a materialized row."""

    manifest = getattr(source, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise HIContinuationAdmissionRefusal("source is not a materialized HI row")
    if manifest.get("stage") != "O1":
        raise HIContinuationAdmissionRefusal(
            f"continuation admission requires stage 'O1', got {manifest.get('stage')!r}"
        )
    identity = manifest.get("scientific_identity")
    if not isinstance(identity, Mapping):
        raise HIContinuationAdmissionRefusal("resolved row is missing scientific_identity")
    optimizer = identity.get("optimizer")
    if not isinstance(optimizer, Mapping) or optimizer.get("method") != "adam":
        raise HIContinuationAdmissionRefusal("continuation admission requires an O1 Adam row")
    context = identity.get("context")
    if not isinstance(context, Mapping):
        raise HIContinuationAdmissionRefusal("resolved row is missing context identity")
    content_hash = getattr(source, "content_hash", None)
    if type(content_hash) is not str or not content_hash:
        raise HIContinuationAdmissionRefusal("materialized row has no exact content hash")
    for name, expected in _O_CONTEXTS.items():
        if dict(context) == expected:
            return name, content_hash
    raise HIContinuationAdmissionRefusal(
        "resolved row context is not the sealed O-raw or O-composite literal"
    )


def validate_hi_continuation_admission(
    source: Any,
    config: Any,
) -> HIContinuationAdmission:
    """Validate a sealed O1 row before a future HI continuation is admitted.

    The helper is intentionally study-local and does not open the continuation
    route.  It reconstructs the expected resolved row from the materialized
    source and requires the supplied config's complete canonical tree to equal
    that producer-resolved tree before sealing its constructor census.  The
    empty S5b-A overlay allowlist therefore catches row drift, raw
    composite/basis injections, and noncanonical ``runner.*`` aliases without
    maintaining a second context allowlist here.

    Parameters
    ----------
    source : object
        Materialized O1 Adam row whose context identity is sealed by its
        manifest and content hash.
    config : mapping
        The resolved Hydra configuration that would be handed to the runner.

    Returns
    -------
    HIContinuationAdmission
        Deterministic identity and the closed inventory for the qualified row.

    Raises
    ------
    HIContinuationAdmissionRefusal
        If the row, config tree, inventory, or context identity is not exactly
        qualified.
    """

    context, content_hash = _admission_context(source)

    try:
        from . import train_config

        expected_config = train_config.resolve_train_config(source)
    except Exception as error:
        raise HIContinuationAdmissionRefusal(
            f"sealed O1 row cannot be resolved: {error}"
        ) from error

    try:
        expected_tree = _canonical_scientific_context_tree(expected_config)
        observed_tree = _canonical_scientific_context_tree(config)
    except Exception as error:
        raise HIContinuationAdmissionRefusal(
            f"resolved-row config tree cannot be canonicalized: {error}"
        ) from error
    if observed_tree != expected_tree:
        raise HIContinuationRowIdentityRefusal(
            "resolved-row context identity does not match the sealed O1 row"
        )

    try:
        expected_entries = validate_mutable_state_inventory(expected_config)
        observed_entries = validate_mutable_state_inventory(config)
    except Exception as error:
        raise HIContinuationAdmissionRefusal(
            f"sealed constructor inventory refused: {error}"
        ) from error

    expected_signature = _canonical_inventory(expected_entries)
    observed_signature = _canonical_inventory(observed_entries)
    if observed_signature != expected_signature:
        raise HIContinuationAdmissionRefusal(
            "resolved-row context identity does not match the sealed O1 row"
        )

    identity = SealedHIContinuationIdentity(
        schema=_ADMISSION_SCHEMA,
        stage="O1",
        context=context,
        content_hash=content_hash,
        constructor_hash=hashlib.sha256(
            json.dumps(expected_signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )
    return HIContinuationAdmission(
        identity=identity,
        inventory=observed_entries,
    )


__all__ = [
    "HIContinuationAdmission",
    "HIContinuationAdmissionRefusal",
    "HIContinuationRowIdentityRefusal",
    "MutableStateEntry",
    "SealedHIContinuationIdentity",
    "build_mutable_state_inventory",
    "refused_inventory_entries",
    "validate_hi_continuation_admission",
    "validate_mutable_state_inventory",
]
