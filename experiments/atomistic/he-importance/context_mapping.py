"""Literal Hydra fragments for the helium-importance context axes.

The mapping in this module is deliberately an experiment-boundary adapter.
It names existing TPEN constructors without importing the package, and keeps
the scientific labels separate from the resolver that consumes them later.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf


class ContextMappingError(ValueError):
    """Base error for malformed or unsupported context labels."""


class UnknownContextLabel(ContextMappingError):
    """Raised when a requested A1/A2 or A6 label is not declared."""


class UnsupportedActivationLabel(ContextMappingError):
    """Raised when a declared activation has no admissible implementation."""


_INPUT_FEATURE_LABELS = {
    "raw",
    "raw plus bounded distances",
}


_JASTROW_AXIS_LABELS = {
    "absent",
    "two-coefficient",
}


_PACKAGE_LABELS = {
    "B00": ("raw", "absent"),
    "B10": ("raw plus bounded distances", "absent"),
    "B01": ("raw", "two-coefficient"),
    "B11": ("raw plus bounded distances", "two-coefficient"),
}


_PATH_LABELS = {
    "tensor": ("tensor", None),
    "linear/coordinate-neighbor": ("linear", "coordinate_neighbor"),
    "linear/orbit-complete": ("linear", "orbit_complete"),
    "hybrid/coordinate-neighbor": ("hybrid", "coordinate_neighbor"),
    "hybrid/orbit-complete": ("hybrid", "orbit_complete"),
}


# These labels are deliberately strings rather than generated numeric ranges.
# The experiment vocabulary is closed: a near miss must not silently select a
# nearby architecture.
_CHANNEL_LEVELS = {
    "(16,16)": (16, 16),
    "(24,24)": (24, 24),
    "(32,32)": (32, 32),
    "(48,48)": (48, 48),
    "(40,24)": (40, 24),
    "(24,40)": (24, 40),
}
_EMBEDDING_LEVELS = {
    "(64,1)": (64, 1),
    "(128,1)": (128, 1),
    "(128,2)": (128, 2),
    "(256,2)": (256, 2),
}
_UPDATE_LEVELS = {"residual", "replacement", "Gaussian residual"}
_PRODUCER_INITIALIZATION_LEVELS = {".25": 0.25, ".50": 0.50, "1.0": 1.0}
_EE_RANGE_LEVELS = {".25": 0.25, "1": 1.0, "4": 4.0}
_EN_INITIALIZATION_LEVELS = {
    "(.0025,.25)": (0.0025, 0.25),
    "(.01,1)": (0.01, 1.0),
    "(.04,4)": (0.04, 4.0),
}
_READOUT_INITIALIZATION_LEVELS = {
    "equal": "equal",
    "signed L1": "signed_l1",
    "random-direction L2": "random_direction_l2",
}

_ACTIVATION_LABELS = {
    "SiLU,SiLU": ("silu", "silu"),
    "tanh,tanh": ("tanh", "tanh"),
    "exact GELU,exact GELU": ("gelu", "gelu"),
    "Softplus beta2,Softplus beta2": ("softplus_beta2", "softplus_beta2"),
    "channel-MLP,SiLU": ("channel_mlp", "silu"),
    "SiLU,channel-MLP": ("silu", "channel_mlp"),
    "channel-MLP,channel-MLP": ("channel_mlp", "channel_mlp"),
}


def context_fragment(
    label: str,
    *,
    activation: str = "SiLU,SiLU",
    channels: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Return one complete ``TPENLayer`` Hydra fragment.

    Parameters
    ----------
    label
        One of the five A1/A2 producer and linear-path-policy labels.
    activation
        One of the seven A6 ``producer,aggregation`` labels.
    channels
        Persistent channel width for both body orders.
    seed
        Base seed used by named initializer streams.

    Raises
    ------
    UnknownContextLabel
        If either scientific label is absent from the declared grid.
    ContextMappingError
        If ``channels`` or ``seed`` is invalid.
    """

    if label not in _PATH_LABELS:
        raise UnknownContextLabel(f"unknown producer/path context label: {label!r}")
    if activation not in _ACTIVATION_LABELS:
        raise UnknownContextLabel(f"unknown activation context label: {activation!r}")
    if type(channels) is not int or channels <= 0:
        raise ContextMappingError("channels must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ContextMappingError("seed must be a non-negative integer")

    family, policy = _PATH_LABELS[label]
    producer_activation, aggregation_activation = _ACTIVATION_LABELS[activation]
    fragment = _layer_fragment(
        family=family,
        policy=policy,
        producer_activation=producer_activation,
        aggregation_activation=aggregation_activation,
        channels=channels,
        seed=seed,
    )
    # Return a fresh tree: callers may add Hydra interpolation or scan fields
    # without mutating a later cell produced from the same label.
    return deepcopy(fragment)


def build_context_fragment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`context_fragment` used by materialization callers."""

    return context_fragment(*args, **kwargs)


def channel_fragment(label: str) -> dict[str, Any]:
    """Return the A4/A5 persistent-channel Hydra fragment.

    Parameters
    ----------
    label
        One of the six declared ``(C1,C2)`` levels.

    Raises
    ------
    UnknownContextLabel
        If ``label`` is not one of the declared levels.
    """

    try:
        c1, c2 = _CHANNEL_LEVELS[label]
    except KeyError as exc:
        raise UnknownContextLabel(f"unknown channel context label: {label!r}") from exc
    return {
        "embedding": {"out_channels": _channels_by_order(c1, c2)},
        "readout": {"pair_channels": c2},
    }


def embedding_fragment(label: str) -> dict[str, Any]:
    """Return the A7 embedding hidden-width/depth Hydra fragment."""

    try:
        hidden_channels, num_hidden_layers = _EMBEDDING_LEVELS[label]
    except KeyError as exc:
        raise UnknownContextLabel(f"unknown embedding context label: {label!r}") from exc
    return {
        "embedding": {
            "hidden_channels": hidden_channels,
            "num_hidden_layers": num_hidden_layers,
        }
    }


def update_fragment(label: str) -> dict[str, Any]:
    """Return the A8 feature-update Hydra fragment."""

    if label not in _UPDATE_LEVELS:
        raise UnknownContextLabel(f"unknown feature-update context label: {label!r}")
    if label == "replacement":
        return {"update": {"_target_": "tpen.nn.ReplaceUpdater"}}
    fragment = {"update": {"_target_": "tpen.nn.ResidualUpdater"}}
    if label == "Gaussian residual":
        fragment["update_envelope"] = {
            "_target_": "tpen.nn.coordinate_envelopes.GaussianCoordinateEnvelope",
            "sigma": 1.0,
        }
    return fragment


def producer_initialization_fragment(label: str) -> dict[str, Any]:
    """Return the A9 constant producer-weight initialization fragment."""

    try:
        initial_weight = _PRODUCER_INITIALIZATION_LEVELS[label]
    except KeyError as exc:
        raise UnknownContextLabel(
            f"unknown producer-initialization context label: {label!r}"
        ) from exc
    return {"initial_weight": initial_weight}


def electron_electron_fragment(label: str) -> dict[str, Any]:
    """Return the A10 trainable same/opposite-spin range fragment."""

    try:
        opposite_range = _EE_RANGE_LEVELS[label]
    except KeyError as exc:
        raise UnknownContextLabel(f"unknown e-e range context label: {label!r}") from exc
    return {
        "_target_": "tpen.nn.ElectronElectronCusp",
        "trainable_range": True,
        "same_range_parameter": 1.0,
        "opposite_range_parameter": opposite_range,
    }


def electron_nucleus_fragment(label: str) -> dict[str, Any]:
    """Return the A11 ``(c,d)``-to-``(d,kappa)`` Hydra fragment at ``Z=2``."""

    try:
        curvature_coefficient, curvature_range = _EN_INITIALIZATION_LEVELS[label]
    except KeyError as exc:
        raise UnknownContextLabel(f"unknown e-n initialization label: {label!r}") from exc
    return {
        "law": {
            "_target_": "tpen.nn.TailSafeElectronNucleusCuspLaw",
            "curvature_range": curvature_range,
            "tail_slope": 2.0 - curvature_coefficient / curvature_range,
        },
        "trainable": True,
    }


def readout_initialization_fragment(label: str, *, seed: int = 0) -> dict[str, Any]:
    """Return the A12 readout initialization and named-stream fragment."""

    try:
        initialization = _READOUT_INITIALIZATION_LEVELS[label]
    except KeyError as exc:
        raise UnknownContextLabel(f"unknown readout-initialization label: {label!r}") from exc
    if type(seed) is not int or seed < 0:
        raise ContextMappingError("seed must be a non-negative integer")
    fragment: dict[str, Any] = {"initialization": initialization}
    if initialization != "equal":
        fragment["initializer"] = {
            "_target_": "tpen.nn.initialization.TorchInitializer",
            "seed": seed,
            "stream": "readout/a12",
        }
    return fragment


def architecture_fragment(
    *,
    producer_path: str = "tensor",
    activation: str = "SiLU,SiLU",
    channels: str = "(32,32)",
    embedding: str = "(128,1)",
    update: str = "residual",
    producer_initialization: str = ".50",
    electron_electron: str = "1",
    electron_nucleus: str = "(.01,1)",
    readout_initialization: str = "equal",
    seed: int = 0,
) -> dict[str, Any]:
    """Return one complete A-axis model fragment.

    The returned mapping is merge-shaped for a literal ``model`` config.  Its
    factor list is intentionally omitted: :func:`compose_architecture_package`
    applies factor overrides by position so existing control factors cannot be
    replaced by a one-entry list.
    """

    channel = channel_fragment(channels)
    embedding_override = embedding_fragment(embedding)["embedding"]
    layer = _layer_fragment(
        family=_require_path_label(producer_path)[0],
        policy=_require_path_label(producer_path)[1],
        producer_activation=_require_activation_label(activation)[0],
        aggregation_activation=_require_activation_label(activation)[1],
        channels=_CHANNEL_LEVELS[channels],
        seed=_validate_seed(seed),
        update=update,
        initial_weight=_PRODUCER_INITIALIZATION_LEVELS[_require_level(
            producer_initialization, _PRODUCER_INITIALIZATION_LEVELS, "producer initialization"
        )],
    )
    # Validate the factor labels here even though factor-list mutation belongs
    # to the composition entry point below.
    electron_electron_fragment(electron_electron)
    electron_nucleus_fragment(electron_nucleus)
    return {
        "embedding": {**embedding_override, **channel["embedding"]},
        "layers": [layer],
        "readout": {
            **channel["readout"],
            **readout_initialization_fragment(readout_initialization, seed=seed),
        },
    }


def compose_architecture_package(
    model_cfg: Mapping[str, Any],
    *,
    package: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply a complete A package to a literal model config.

    Nested mappings are merged and interpolations remain unresolved.  The
    existing ``factors`` list is copied and its two declared control factors
    are updated in place; it is never replaced.  If ``package`` is supplied,
    the ordinary B composition is applied after the A package.
    """

    fragment = architecture_fragment(**kwargs)
    factor_overrides = {
        "electron_electron": electron_electron_fragment(kwargs.get("electron_electron", "1")),
        "electron_nucleus": electron_nucleus_fragment(kwargs.get("electron_nucleus", "(.01,1)")),
    }
    composed = _merge_mappings(
        OmegaConf.to_container(model_cfg, resolve=False)
        if OmegaConf.is_config(model_cfg)
        else deepcopy(model_cfg),
        fragment,
    )
    factors = deepcopy(composed.get("factors", []))
    if len(factors) < 2:
        raise ContextMappingError("literal control model must declare e-e and e-n factors")
    factors[0] = _merge_mappings(factors[0], factor_overrides["electron_electron"])
    factors[1] = _merge_mappings(factors[1], factor_overrides["electron_nucleus"])
    composed["factors"] = factors
    if package is not None:
        composed = compose_package(composed, package)
    return composed


def _validate_seed(seed: int) -> int:
    if type(seed) is not int or seed < 0:
        raise ContextMappingError("seed must be a non-negative integer")
    return seed


def _require_level(label: str, levels: Mapping[str, Any], name: str) -> str:
    if label not in levels:
        raise UnknownContextLabel(f"unknown {name} context label: {label!r}")
    return label


def _require_path_label(label: str) -> tuple[str, str | None]:
    try:
        return _PATH_LABELS[label]
    except KeyError as exc:
        raise UnknownContextLabel(f"unknown producer/path context label: {label!r}") from exc


def _require_activation_label(label: str) -> tuple[str, str]:
    try:
        return _ACTIVATION_LABELS[label]
    except KeyError as exc:
        raise UnknownContextLabel(f"unknown activation context label: {label!r}") from exc


def input_feature_fragment(
    label: str,
    *,
    spatial_dim: int = 3,
    include_spin: bool = True,
    length: float = 1.0,
) -> dict[str, Any]:
    """Return the model fragment for one B input-feature package.

    ``"raw"`` is an empty override so composing it with the literal control
    leaves that model exactly unchanged.  The bounded-distance package keeps
    raw Cartesian and spin columns, appends one one-body distance column, and
    consumes the typed ordered-pair channel ``(q_i, q_j, q_ij)``.
    """

    if label not in _INPUT_FEATURE_LABELS:
        raise UnknownContextLabel(f"unknown input-feature context label: {label!r}")
    if type(spatial_dim) is not int or spatial_dim <= 0:
        raise ContextMappingError("spatial_dim must be a positive integer")
    if type(include_spin) is not bool:
        raise ContextMappingError("include_spin must be a boolean")
    if type(length) not in {int, float} or isinstance(length, bool) or length != 1.0:
        raise ContextMappingError("length is fixed at 1.0 bohr for this study")

    if label == "raw":
        return {}

    raw_width = spatial_dim + (1 if include_spin else 0)
    return {
        "basis": {
            "_target_": "tpen.nn.basis.BoundedDistanceBasis",
            "inner": {
                "_target_": "tpen.nn.RawCoordinateBasis",
                "spatial_dim": spatial_dim,
                "include_spin": include_spin,
            },
            # The study fixes the bounded-distance length to one bohr.
            "length": 1.0,
        },
        "embedding": {
            "in_features": raw_width + 1,
            "pair_input_channels": 3,
        },
    }


def build_input_feature_fragment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`input_feature_fragment` used by materializers."""

    return input_feature_fragment(*args, **kwargs)


def jastrow_fragment(label: str, *, length: float = 1.0) -> dict[str, Any]:
    """Return the optional additive log-Jastrow factor fragment.

    The absent axis is an empty fragment.  The two-coefficient axis writes all
    constructor values explicitly so a package cannot inherit an identity
    factor accidentally.  The caller appends this mapping to the existing
    ``model.factors`` list; the control factor entries are never replaced here.
    """

    if label not in _JASTROW_AXIS_LABELS:
        raise UnknownContextLabel(f"unknown Jastrow axis label: {label!r}")
    if type(length) not in {int, float} or isinstance(length, bool) or length != 1.0:
        raise ContextMappingError("Jastrow length is fixed at 1.0 bohr for this study")
    if label == "absent":
        return {}
    return {
        "_target_": "tpen.nn.jastrow.BoundedTwoCoefficientJastrow",
        "trainable": True,
        "length": 1.0,
        "mean_coefficient": 0.0,
        "difference_coefficient": 0.0,
    }


def build_jastrow_fragment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for :func:`jastrow_fragment` used by materializers."""

    return jastrow_fragment(*args, **kwargs)


def package_fragment(label: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return the input override and optional factor for one B package.

    The factor is deliberately returned separately from the merge-shaped input
    mapping.  This prevents a caller from passing a one-entry ``factors`` list
    to ``OmegaConf.merge`` and silently replacing the control cusps.
    """

    try:
        input_label, jastrow_label = _PACKAGE_LABELS[label]
    except KeyError as exc:
        raise UnknownContextLabel(f"unknown B package label: {label!r}") from exc

    return input_feature_fragment(input_label), jastrow_fragment(jastrow_label) or None


def compose_package(model_cfg: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Compose one package onto a model config without replacing its factors.

    The returned mapping is fresh.  Nested input-feature overrides are merged,
    while an optional Jastrow is appended after the existing factor entries.
    Assign it back at the full configuration's model key (for example,
    ``cfg.model = compose_package(cfg.model, label)``) before resolving it.
    Interpolations are intentionally preserved unresolved and resolve against
    that full configuration root; resolving the standalone mapping is invalid.
    """

    fragment, factor = package_fragment(label)
    if OmegaConf.is_config(model_cfg):
        # Detach from the source config before merging.  Otherwise deepcopy
        # preserves DictConfig parent links and interpolation access can
        # silently resolve against a different root.
        base = OmegaConf.to_container(model_cfg, resolve=False)
    else:
        base = deepcopy(model_cfg)
    composed = _merge_mappings(base, fragment)
    if fragment.get("embedding"):
        # Construct the raw twin from the verbatim control embedding and the
        # augmented target from the same config with only the package's input
        # overrides. Hydra constructs both before the factory transplants the
        # shared parameters, preserving the authoritative seed stream.
        control_embedding = deepcopy(base["embedding"])
        augmented_embedding = deepcopy(composed["embedding"])
        composed["embedding"] = {
            "_target_": "tpen.nn.embedding.matched_start_embedding",
            "raw": control_embedding,
            "augmented": augmented_embedding,
        }
    if factor is not None:
        factors = deepcopy(composed.get("factors", []))
        factors.append(deepcopy(factor))
        composed["factors"] = factors
    return composed


def _merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge mappings while preserving list-valued fields verbatim."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_package_fragment(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Alias for :func:`package_fragment` used by materializers."""

    return package_fragment(*args, **kwargs)


def _layer_fragment(
    *,
    family: str,
    policy: str | None,
    producer_activation: str,
    aggregation_activation: str,
    channels: int | tuple[int, int],
    seed: int,
    update: str = "residual",
    initial_weight: float = 0.5,
) -> dict[str, Any]:
    """Build the literal layer tree after labels have been validated."""

    channel_map = _channels_by_order(*channels) if isinstance(channels, tuple) else _channels(channels)

    layout = {
        "_target_": "tpen.data.paths.compose_path_layout",
        "linear": _linear_metadata(policy) if policy is not None else None,
        "tensor_product": _tensor_product_metadata() if family in {"tensor", "hybrid"} else None,
        "input_orders": _orders(),
        "output_orders": _orders(),
        "input_channels": channel_map,
        "output_channels": deepcopy(channel_map),
        "max_virtual_order": 2,
    }
    producers = []
    if family in {"linear", "hybrid"}:
        producers.append({
            "_target_": "tpen.nn.LinearEquivariantMixing",
            "max_order": 2,
            "channels": channel_map,
            "metadata": "${...layout.linear}",
            "aggregation": "sum",
            "implementation": "vectorized",
            "initial_weight": initial_weight,
        })
    if family in {"tensor", "hybrid"}:
        producers.append({
            "_target_": "tpen.nn.EquivariantMixing",
            "max_order": 2,
            "max_virtual_order": 2,
            "channels": channel_map,
            "paths": "${...layout.tensor_product}",
            "implementation": "vectorized",
            "activation": None,
            "initial_weight": initial_weight,
        })
    fragment = {
        "_target_": "tpen.nn.TPENLayer",
        "layout": layout,
        "mixing": {
            "_target_": "tpen.nn.CompositeMixing",
            "layout": "${..layout}",
            "producers": producers,
            "activation": _activation(producer_activation, seed=seed, site="producer", channels=channels),
        },
        "path_aggregation": {
            "_target_": "tpen.nn.PathAggregation",
            "max_order": 2,
            "channels": channels,
            "layout": "${..layout}",
            "activation": _activation(aggregation_activation, seed=seed, site="aggregation", channels=channels),
            "initializer": {
                "_target_": "tpen.nn.TorchInitializer",
                "seed": seed,
                "stream": "layer0/path_aggregation",
            },
        },
        "update": update_fragment(update),
    }
    return fragment


def _orders() -> dict[str, Any]:
    return {"_target_": "tpen.data.paths.NormalizedOrders", "values": [1, 2]}


def _channels(channels: int) -> dict[str, Any]:
    return _channels_by_order(channels, channels)


def _channels_by_order(order_one: int, order_two: int) -> dict[str, Any]:
    return {
        "_target_": "tpen.data.paths.NormalizedChannels",
        "values": [[1, order_one], [2, order_two]],
    }


def _linear_metadata(policy: str) -> dict[str, Any]:
    explicit = _COORDINATE_PATHS if policy == "coordinate_neighbor" else _ORBIT_COMPLETE_PATHS
    return {
        "_target_": "tpen.data.paths.LinearPathMetadata.generate",
        "_convert_": "object",
        "max_order": 2,
        "policy": "explicit",
        "normalization": "sum",
        "explicit": deepcopy(explicit),
    }


def _support_path(output_order: int, input_order: int, tau_out: list[int], tau_in: list[int]) -> dict[str, Any]:
    return {
        "_target_": "tpen.data.paths.SupportPath",
        "output_order": output_order,
        "input_order": input_order,
        "tau_out": tau_out,
        "tau_in": tau_in,
        "normalization": "sum",
    }


# These are literal, hand-written helium-admissible records.  In particular,
# no generator output is filtered here: support-three paths must stay absent.
_COORDINATE_PATHS = [
    _support_path(1, 1, [0], [0]),
    _support_path(1, 1, [0], [1]),
    _support_path(2, 2, [0, 1], [0, 1]),
]

_ORBIT_COMPLETE_PATHS = [
    _support_path(1, 1, [0], [0]),
    _support_path(1, 1, [0], [1]),
    _support_path(1, 2, [0], [0, 1]),
    _support_path(1, 2, [0], [1, 0]),
    _support_path(2, 1, [0, 1], [0]),
    _support_path(2, 1, [0, 1], [1]),
    _support_path(2, 2, [0, 1], [0, 1]),
    _support_path(2, 2, [0, 1], [1, 0]),
]


def _tensor_product_metadata() -> dict[str, Any]:
    return {
        "_target_": "tpen.data.paths.PathMetadata.generate",
        "max_order": 2,
        "max_virtual_order": 2,
        "output_embedding": "canonical",
    }


def _channel_widths(channels: int | Mapping[int, int] | tuple[int, int]) -> dict[int, int]:
    """Normalize persistent channel widths for activation construction."""

    if isinstance(channels, Mapping):
        normalized = {int(order): int(width) for order, width in channels.items()}
    elif isinstance(channels, tuple):
        normalized = {1: int(channels[0]), 2: int(channels[1])}
    else:
        normalized = {1: int(channels), 2: int(channels)}
    if set(normalized) != {1, 2} or any(width <= 0 for width in normalized.values()):
        raise ContextMappingError(f"channels must declare positive order-1/order-2 widths: {channels!r}")
    return normalized


def _activation(name: str, *, seed: int, site: str, channels: int) -> Any:
    if name == "silu":
        return {"_target_": "torch.nn.SiLU"}
    if name == "tanh":
        return {"_target_": "torch.nn.Tanh"}
    if name == "gelu":
        return {"_target_": "torch.nn.GELU", "approximate": "none"}
    if name == "softplus_beta2":
        # No dev module implements the required all-range logaddexp formula.
        raise UnsupportedActivationLabel(
            "Softplus beta2 requires stable logaddexp(0, 2*z)/2; no admissible dev module"
        )
    if name == "channel_mlp":
        tuple_axes_start = 3 if site == "producer" else 2
        channel_widths = _channel_widths(channels)
        return {
            "_target_": "tpen.nn.ChannelPreservingMLPActivation",
            "layout": {
                "_target_": "tpen.nn.OrderMLPLayout",
                "_recursive_": True,
                "_convert_": "object",
                "axes": {
                    "_target_": "tpen.nn.ChannelActivationAxes",
                    "channel_axis": 1,
                    "tuple_axes_start": tuple_axes_start,
                },
                "specs": [
                    {
                        "_target_": "tpen.nn.OrderMLPSpec",
                        "order": order,
                        "channels": channel_widths[order],
                        "hidden_channels": 4 * channel_widths[order],
                        "num_hidden_layers": 1,
                        "activation": {"_target_": "torch.nn.SiLU"},
                        "bias": True,
                    }
                    for order in (1, 2)
                ],
            },
            "initializer": {
                "_target_": "tpen.nn.TorchInitializer",
                "seed": seed,
                "stream": f"layer0/{site}_activation",
            },
        }
    raise AssertionError(f"unmapped internal activation {name!r}")


__all__ = [
    "architecture_fragment",
    "channel_fragment",
    "ContextMappingError",
    "compose_architecture_package",
    "electron_electron_fragment",
    "electron_nucleus_fragment",
    "embedding_fragment",
    "UnknownContextLabel",
    "UnsupportedActivationLabel",
    "build_input_feature_fragment",
    "build_jastrow_fragment",
    "build_package_fragment",
    "build_context_fragment",
    "compose_package",
    "context_fragment",
    "producer_initialization_fragment",
    "readout_initialization_fragment",
    "update_fragment",
    "input_feature_fragment",
    "jastrow_fragment",
    "package_fragment",
]
