"""Closed mutable-state inventory for the HI training configuration.

The inventory is deliberately derived from the committed Hydra tree rather
than maintained as a second list. A newly constructed target therefore fails
closed until its recovery ownership is classified here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_CONFIG_PATH = Path(__file__).with_name("configs") / "train.yaml"
_RESTORED = "restored", "native train_resume owns the corresponding state"
_RECONSTRUCTIBLE = (
    "reconstructible",
    "the object is rebuilt from the committed HI configuration",
)
_TYPED_REFUSED = (
    "typed-refused",
    "native recovery profile does not admit this mutable owner; HI continuation refuses before construction",
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
        _RESTORED,
    ),
    **_registry_entries(
        [
            ("callbacks.8", "tpen.callback.RuntimeEquivariance"),
            ("callbacks.9.payload", "tpen.checkpoint.TrainResume"),
            ("callbacks.10", "tpen.callback.ResourceUsage"),
        ],
        _TYPED_REFUSED,
    ),
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
            ("model.layers.0.layout.tensor_product", "tpen.data.paths.PathMetadata.generate"),
            ("model.layers.0.layout.input_orders", "tpen.data.paths.NormalizedOrders"),
            ("model.layers.0.layout.output_orders", "tpen.data.paths.NormalizedOrders"),
            ("model.layers.0.layout.input_channels", "tpen.data.paths.NormalizedChannels"),
            ("model.layers.0.layout.output_channels", "tpen.data.paths.NormalizedChannels"),
            ("model.layers.0.mixing", "tpen.nn.CompositeMixing"),
            ("model.layers.0.mixing.producers.0", "tpen.nn.LinearEquivariantMixing"),
            ("model.layers.0.mixing.producers.1", "tpen.nn.EquivariantMixing"),
            ("model.layers.0.mixing.activation", "tpen.nn.ChannelPreservingMLPActivation"),
            ("model.layers.0.mixing.activation.layout", "tpen.nn.OrderMLPLayout"),
            ("model.layers.0.mixing.activation.layout.axes", "tpen.nn.ChannelActivationAxes"),
            ("model.layers.0.mixing.activation.layout.specs.0", "tpen.nn.OrderMLPSpec"),
            ("model.layers.0.mixing.activation.layout.specs.0.activation", "torch.nn.SiLU"),
            ("model.layers.0.mixing.activation.layout.specs.1", "tpen.nn.OrderMLPSpec"),
            ("model.layers.0.mixing.activation.layout.specs.1.activation", "torch.nn.SiLU"),
            ("model.layers.0.mixing.activation.initializer", "tpen.nn.TorchInitializer"),
            ("model.layers.0.path_aggregation.activation", "tpen.nn.ChannelPreservingMLPActivation"),
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


@dataclass(frozen=True, slots=True)
class MutableStateEntry:
    """One constructed HI object and its explicit continuation disposition."""

    path: str
    target: str
    classification: str
    reason: str
    source: str


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
    if config is None:
        from . import train_config

        config = train_config._compose_base_config()
    if isinstance(config, DictConfig) or OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=False)
    if not isinstance(config, Mapping):
        raise TypeError("HI mutable-state inventory requires a mapping config")
    return config


def build_mutable_state_inventory(config: Any | None = None) -> tuple[MutableStateEntry, ...]:
    """Enumerate and classify every constructor in the HI train config."""

    tree = _config_tree(config)
    entries: list[MutableStateEntry] = []
    for path, target in _walk_targets(tree):
        classification, reason = _classification(path, target)
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
    return tuple(entries)


def validate_mutable_state_inventory(config: Any | None = None) -> tuple[MutableStateEntry, ...]:
    """Return the inventory, refusing any unclassified constructed object."""

    entries = build_mutable_state_inventory(config)
    allowed = {"restored", "reconstructible", "allocation-local", "typed-refused"}
    invalid = [entry for entry in entries if entry.classification not in allowed]
    if invalid:
        raise ValueError(f"unclassified HI mutable-state object(s): {invalid!r}")
    return entries


def refused_inventory_entries(config: Any | None = None) -> tuple[MutableStateEntry, ...]:
    """Return the typed-refused entries used in the stable refusal reason."""

    return tuple(
        entry
        for entry in validate_mutable_state_inventory(config)
        if entry.classification == "typed-refused"
    )


__all__ = [
    "MutableStateEntry",
    "build_mutable_state_inventory",
    "refused_inventory_entries",
    "validate_mutable_state_inventory",
]
