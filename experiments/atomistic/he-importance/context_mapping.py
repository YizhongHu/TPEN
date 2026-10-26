"""Literal Hydra fragments for the helium-importance context axes.

The mapping in this module is deliberately an experiment-boundary adapter.
It names existing TPEN constructors without importing the package, and keeps
the scientific labels separate from the resolver that consumes them later.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


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


_PATH_LABELS = {
    "tensor": ("tensor", None),
    "linear/coordinate-neighbor": ("linear", "coordinate_neighbor"),
    "linear/orbit-complete": ("linear", "orbit_complete"),
    "hybrid/coordinate-neighbor": ("hybrid", "coordinate_neighbor"),
    "hybrid/orbit-complete": ("hybrid", "orbit_complete"),
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


def _layer_fragment(
    *,
    family: str,
    policy: str | None,
    producer_activation: str,
    aggregation_activation: str,
    channels: int,
    seed: int,
) -> dict[str, Any]:
    """Build the literal layer tree after labels have been validated."""

    layout = {
        "_target_": "tpen.data.paths.compose_path_layout",
        "linear": _linear_metadata(policy) if policy is not None else None,
        "tensor_product": _tensor_product_metadata() if family in {"tensor", "hybrid"} else None,
        "input_orders": _orders(),
        "output_orders": _orders(),
        "input_channels": _channels(channels),
        "output_channels": _channels(channels),
        "max_virtual_order": 2,
    }
    producers = []
    if family in {"linear", "hybrid"}:
        producers.append({
            "_target_": "tpen.nn.LinearEquivariantMixing",
            "max_order": 2,
            "channels": channels,
            "metadata": "${...layout.linear}",
            "aggregation": "sum",
            "implementation": "vectorized",
            "initial_weight": 0.5,
        })
    if family in {"tensor", "hybrid"}:
        producers.append({
            "_target_": "tpen.nn.EquivariantMixing",
            "max_order": 2,
            "max_virtual_order": 2,
            "channels": channels,
            "paths": "${...layout.tensor_product}",
            "implementation": "vectorized",
            "activation": None,
            "initial_weight": 0.5,
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
        "update": {"_target_": "tpen.nn.ResidualUpdater"},
    }
    return fragment


def _orders() -> dict[str, Any]:
    return {"_target_": "tpen.data.paths.NormalizedOrders", "values": [1, 2]}


def _channels(channels: int) -> dict[str, Any]:
    return {
        "_target_": "tpen.data.paths.NormalizedChannels",
        "values": [[1, channels], [2, channels]],
    }


def _linear_metadata(policy: str) -> dict[str, Any]:
    explicit = _COORDINATE_PATHS if policy == "coordinate_neighbor" else _ORBIT_COMPLETE_PATHS
    return {
        "_target_": "tpen.data.paths.LinearPathMetadata.generate",
        "_convert_": "object",
        "max_order": 2,
        "policy": "explicit",
        "normalization": "sum",
        "explicit": {
            "_target_": "builtins.tuple",
            "_args_": [deepcopy(explicit)],
        },
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
                        "channels": channels,
                        "hidden_channels": 4 * channels,
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
    "ContextMappingError",
    "UnknownContextLabel",
    "UnsupportedActivationLabel",
    "build_input_feature_fragment",
    "build_context_fragment",
    "context_fragment",
    "input_feature_fragment",
]
