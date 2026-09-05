"""Closed scientific factor dictionary for the He-importance S1 surface.

This module is intentionally an experiment-boundary adapter.  It does not
import :mod:`tpen`; it emits one fully resolved, Hydra-compatible mapping whose
``_target_`` values name the already-landed TPEN modules.  Stage S2 owns rows,
manifests, seed namespaces, and paths; this module only validates one complete
assignment and resolves its runnable configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml


DICTIONARY_PATH = Path(__file__).with_name("factor_dictionary.yaml")
SCHEMA = "tpen-he-importance-factor-dictionary-v1"
VERSION = 1

MIXING_PACKAGES = ("tensor", "k-GNN-like-linear", "hybrid-additive")
ACTIVATION_PACKAGES = ("pointwise", "MLP-mixing", "MLP-aggregation", "MLP-both")
OPTIMIZER_FAMILIES = ("Adam", "RAdam", "RMSprop", "SGD-Nesterov")
LR_MULTIPLIERS = (0.5, 1.0, 2.0)
BREADTH_COLUMNS = tuple("ABCDEFGHIJKL")
INDEPENDENT_BREADTH_COLUMNS = tuple("ABCDEFGH")
ANCHORS = ("Q0", "Q1", "Q2", "Q3")
BODY_MAX_ORDER = 2

_EXPECTED_TOP_LEVEL_KEYS = (
    "schema",
    "version",
    "experiment_id",
    "vocabularies",
    "stage_0",
    "core",
    "breadth",
    "anchors",
    "shared_choice_surface",
    "control_reference",
    "invariants",
)
_REQUIRED_ASSIGNMENT_KEYS = (
    "mixing",
    "activation",
    "optimizer",
    "lr_multiplier",
    *BREADTH_COLUMNS,
)
_DERIVED_COLUMNS = {"I": "ABCD", "J": "ABEF", "K": "ACEG", "L": "BDFH"}
_DEFAULT_CPMLP_SEEDS = {"mixing": 701, "aggregation": 702}


def _canonical_json(value: object) -> str:
    """Serialize a JSON-compatible value with stable bytes."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_yaml(value: Mapping[str, Any]) -> str:
    """Return the canonical YAML representation used by the dictionary.

    Parameters
    ----------
    value : mapping
        A validated dictionary payload.
    """

    return yaml.safe_dump(
        dict(value),
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )


def load_dictionary(path: str | Path = DICTIONARY_PATH) -> dict[str, Any]:
    """Load and validate the versioned canonical factor dictionary."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("factor dictionary must contain a mapping")
    validate_dictionary(payload)
    return payload


def validate_dictionary(payload: Mapping[str, Any]) -> None:
    """Validate exact dictionary structure, order, and scientific values.

    The closed-set checks deliberately compare ordered tuples.  Consequently
    an unknown, missing, extra, duplicate, or reordered vocabulary item fails
    at the boundary instead of silently changing the experiment surface.
    """

    _require_exact_keys(payload, _EXPECTED_TOP_LEVEL_KEYS, "dictionary")
    if payload["schema"] != SCHEMA or payload["version"] != VERSION:
        raise ValueError("unsupported factor dictionary schema or version")
    if payload["experiment_id"] != "he-importance-v1":
        raise ValueError("unexpected experiment_id")

    vocabularies = _mapping(payload, "vocabularies")
    _require_exact_keys(
        vocabularies,
        ("mixing_packages", "activation_packages", "optimizer_families", "lr_multipliers", "binary_factors", "anchors"),
        "vocabularies",
    )
    _require_ordered(vocabularies["mixing_packages"], MIXING_PACKAGES, "mixing packages")
    _require_ordered(vocabularies["activation_packages"], ACTIVATION_PACKAGES, "activation packages")
    _require_ordered(vocabularies["optimizer_families"], OPTIMIZER_FAMILIES, "optimizer families")
    _require_ordered(vocabularies["lr_multipliers"], LR_MULTIPLIERS, "LR multipliers")
    _require_ordered(vocabularies["binary_factors"], BREADTH_COLUMNS, "binary factors")
    _require_ordered(vocabularies["anchors"], ANCHORS, "anchors")

    stage_0 = _mapping(payload, "stage_0")
    _require_exact_keys(stage_0, ("optimizer_families", "families", "center_policy"), "stage_0")
    _require_exact_keys(stage_0["families"], OPTIMIZER_FAMILIES, "Stage 0 families")
    _require_ordered(stage_0["optimizer_families"], OPTIMIZER_FAMILIES, "Stage 0 optimizer families")
    for family in OPTIMIZER_FAMILIES:
        entry = _mapping(stage_0["families"], family)
        expected = _STAGE0[family]
        _require_exact_keys(entry, ("target", "fixed", "lr_candidates"), f"Stage 0 {family}")
        if entry["target"] != expected["target"]:
            raise ValueError(f"Stage 0 target mismatch for {family}")
        if entry["fixed"] != expected["fixed"]:
            raise ValueError(f"Stage 0 fixed settings mismatch for {family}")
        _require_ordered(entry["lr_candidates"], expected["lr_candidates"], f"{family} LR candidates")

    core = _mapping(payload, "core")
    _require_exact_keys(core, ("body_max_order", "parameter_count", "causal_warning", "absent_factor_label", "mixing_packages", "activation_packages", "cpmlp"), "core")
    if core["body_max_order"] != BODY_MAX_ORDER:
        raise ValueError("body_max_order must be fixed at 2")
    _require_ordered(core["mixing_packages"], MIXING_PACKAGES, "core mixing packages")
    _require_ordered(core["activation_packages"], ACTIVATION_PACKAGES, "core activation packages")
    _require_exact_keys(core["mixing_packages"], MIXING_PACKAGES, "core mixing package IDs")
    _require_exact_keys(core["activation_packages"], ACTIVATION_PACKAGES, "core activation package IDs")

    breadth = _mapping(payload, "breadth")
    _require_ordered(breadth["independent_columns"], INDEPENDENT_BREADTH_COLUMNS, "independent breadth columns")
    if breadth["derived_columns"] != _DERIVED_COLUMNS:
        raise ValueError("derived breadth columns do not match the resolution-V definition")
    _require_ordered(breadth["columns"].keys(), BREADTH_COLUMNS, "breadth column IDs")
    for column in BREADTH_COLUMNS:
        entry = _mapping(breadth["columns"], column)
        if set(entry) - {"name", "minus", "plus", "body_max_order", "amendment"}:
            raise ValueError(f"unknown keys in breadth column {column}")
        if column == "C":
            if entry.get("body_max_order") != BODY_MAX_ORDER:
                raise ValueError("column C must retain body max_order=2")
            if entry.get("amendment") != "user decision 2026-09-04 (Q1)":
                raise ValueError("column C amendment citation is missing")

    anchors = _mapping(payload, "anchors")
    _require_ordered(anchors.keys(), ANCHORS, "anchor IDs")
    for anchor in ANCHORS:
        entry = _mapping(anchors, anchor)
        for key in ("mixing", "activation", "optimizer", "lr_multiplier"):
            if key not in entry:
                raise ValueError(f"anchor {anchor} is missing {key}")
        _require_member(entry["mixing"], MIXING_PACKAGES, f"anchor {anchor} mixing")
        _require_member(entry["activation"], ACTIVATION_PACKAGES, f"anchor {anchor} activation")
        _require_member(entry["optimizer"], OPTIMIZER_FAMILIES, f"anchor {anchor} optimizer")
        _require_member(entry["lr_multiplier"], LR_MULTIPLIERS, f"anchor {anchor} multiplier")

    control = _mapping(payload, "control_reference")
    if control.get("coordinates") != "CONTROL_COORDINATES=PENDING-USER-DETAILED-REVIEW":
        raise ValueError("control coordinates must remain pending")


def derive_breadth_columns(independent: Mapping[str, int]) -> dict[str, int]:
    """Derive all A--L signs from independent A--H signs.

    Parameters
    ----------
    independent : mapping
        Exactly the eight independent columns A--H, each with sign ``-1`` or
        ``+1``.  The helper does not allocate experiment rows or manifests.
    """

    _require_exact_keys(independent, INDEPENDENT_BREADTH_COLUMNS, "independent breadth signs")
    result = {column: _sign(independent[column], column) for column in INDEPENDENT_BREADTH_COLUMNS}
    for column, word in _DERIVED_COLUMNS.items():
        value = 1
        for source in word:
            value *= result[source]
        result[column] = value
    return result


def breadth_levels(signs: Mapping[str, int]) -> dict[str, Any]:
    """Translate any complete A--L sign assignment to exact factor levels.

    Generator membership is a property of the 256-row design fraction, not a
    validity condition on an assignment.  In particular, the named control is
    deliberately outside that fraction and must still resolve here.
    """

    _require_key_set(signs, BREADTH_COLUMNS, "breadth signs")
    normalized = {column: _sign(signs[column], column) for column in BREADTH_COLUMNS}
    columns = load_dictionary()["breadth"]["columns"]
    return {
        column: columns[column]["plus" if normalized[column] == 1 else "minus"]
        for column in BREADTH_COLUMNS
    }


def is_fraction_row(signs: Mapping[str, int]) -> bool:
    """Return whether complete A--L signs belong to the 256-row fraction.

    This is a diagnostic predicate for callers that explicitly care about the
    design subset.  It does not constrain :func:`breadth_levels` or the
    complete-assignment resolver: the named control is deliberately outside
    the fraction.
    """

    _require_key_set(signs, BREADTH_COLUMNS, "breadth signs")
    normalized = {column: _sign(signs[column], column) for column in BREADTH_COLUMNS}
    return derive_breadth_columns(
        {column: normalized[column] for column in INDEPENDENT_BREADTH_COLUMNS}
    ) == normalized


def iter_breadth_signs() -> tuple[dict[str, int], ...]:
    """Return the 256 resolution-V sign records in lexicographic order.

    This is a dictionary utility only: it emits no scientific IDs, paths,
    manifests, or production planner state.
    """

    records = []
    for values in itertools.product((-1, 1), repeat=len(INDEPENDENT_BREADTH_COLUMNS)):
        records.append(
            derive_breadth_columns(dict(zip(INDEPENDENT_BREADTH_COLUMNS, values)))
        )
    return tuple(records)


def resolve_center_times_multiplier(
    optimizer: str,
    multiplier: float,
    *,
    signed_centers: Mapping[str, float] | None = None,
) -> float:
    """Resolve a family centre times multiplier, failing closed when unsigned.

    Adam's centre is fixed at ``0.005``.  A non-Adam family has no selected
    centre in S1a, so it requires a later signed centre supplied by the caller.
    """

    _require_member(optimizer, OPTIMIZER_FAMILIES, "optimizer")
    _require_member(multiplier, LR_MULTIPLIERS, "LR multiplier")
    if optimizer == "Adam":
        centre = 0.005
    else:
        if signed_centers is None or optimizer not in signed_centers:
            raise ValueError(f"no signed Stage 0 centre supplied for non-Adam family {optimizer!r}")
        centre = _positive_float(signed_centers[optimizer], f"signed center for {optimizer}")
    return centre * float(multiplier)


def resolve_optimizer(optimizer: str, absolute_lr: float) -> dict[str, Any]:
    """Return the exact runnable optimizer target and settings."""

    _require_member(optimizer, OPTIMIZER_FAMILIES, "optimizer")
    lr = _positive_float(absolute_lr, "absolute_lr")
    entry = load_dictionary()["stage_0"]["families"][optimizer]
    return {
        "_target_": entry["target"],
        "lr": lr,
        **deepcopy(entry["fixed"]),
    }


def cpmlp_initializer_streams(
    runtime_seed: int | None,
    *,
    layer_index: int = 0,
) -> dict[str, dict[str, Any]]:
    """Derive named, non-colliding CPMLP initializer streams.

    A host runtime seed is the one source of entropy for both slots.  The
    stream names carry the layer and slot identity, so the resulting generators
    are distinct without inventing a second user-facing seed.  ``None`` keeps
    the standalone Hooke preset defaults ``701`` and ``702``.
    """

    if runtime_seed is not None:
        if not isinstance(runtime_seed, int) or isinstance(runtime_seed, bool):
            raise TypeError("runtime_seed must be an integer or None")
        seeds = {slot: int(runtime_seed) for slot in _DEFAULT_CPMLP_SEEDS}
    else:
        seeds = dict(_DEFAULT_CPMLP_SEEDS)
    if not isinstance(layer_index, int) or isinstance(layer_index, bool) or layer_index < 0:
        raise ValueError("layer_index must be a nonnegative integer")
    return {
        slot: {
            "seed": seeds[slot],
            "stream": f"he-importance/cpmlp/layer_{layer_index}/{slot}",
        }
        for slot in ("mixing", "aggregation")
    }


def build_shared_hooke_choice_surface(
    *,
    channels: int = 2,
    max_order: int = BODY_MAX_ORDER,
    max_virtual_order: int = BODY_MAX_ORDER,
    path_family: str = "canonical",
    aggregation: str = "completion_mean",
    initial_weight: float = 0.5,
    mixing_package: str = "tensor",
    activation_package: str = "pointwise",
    runtime_seed: int | None = None,
    layer_index: int = 0,
) -> dict[str, Any]:
    """Build a self-contained Hydra fragment for one Hooke choice.

    The returned mapping has no OmegaConf interpolation and can therefore be
    merged into a host config or passed directly to ``hydra.utils.instantiate``
    by a caller.  Every path, order, width, aggregation, weight, activation,
    and initializer choice is explicit.  The function does not instantiate
    TPEN modules itself.
    """

    channels = _positive_int(channels, "channels")
    max_order = _positive_int(max_order, "max_order")
    max_virtual_order = _positive_int(max_virtual_order, "max_virtual_order")
    if path_family not in ("canonical", "full"):
        raise ValueError(f"unsupported path family {path_family!r}")
    if aggregation == "completion mean":
        aggregation = "completion_mean"
    if aggregation not in ("sum", "completion_mean"):
        raise ValueError(f"unsupported aggregation {aggregation!r}")
    if mixing_package not in MIXING_PACKAGES:
        raise ValueError(f"unsupported mixing package {mixing_package!r}")
    if activation_package not in ACTIVATION_PACKAGES:
        raise ValueError(f"unsupported activation package {activation_package!r}")
    if not isinstance(initial_weight, (int, float)) or isinstance(initial_weight, bool):
        raise TypeError("initial_weight must be numeric")
    streams = cpmlp_initializer_streams(runtime_seed, layer_index=layer_index)
    activations = _activation_configs(
        channels=channels,
        max_order=max_order,
        activation_package=activation_package,
        streams=streams,
    )
    interaction = _interaction_config(
        channels=channels,
        max_order=max_order,
        max_virtual_order=max_virtual_order,
        path_family=path_family,
        aggregation=aggregation,
        initial_weight=float(initial_weight),
        mixing_package=mixing_package,
        activations=activations,
    )
    return {
        "choices": {
            "shared": {
                "channels": channels,
                "max_order": max_order,
                "max_virtual_order": max_virtual_order,
                "path_family": path_family,
                "aggregation": aggregation,
                "initial_weight": float(initial_weight),
                "cpmlp_initializer_streams": streams,
            },
            "activation": activations,
            "interaction": {mixing_package: interaction},
        }
    }


def resolve_assignment(
    assignment: Mapping[str, Any],
    *,
    absolute_lr: float,
    runtime_seed: int,
    science_assignment_id: str = "unassigned",
    layer_index: int = 0,
) -> "ResolvedConfiguration":
    """Resolve one complete factor assignment to a runnable config.

    ``science_assignment_id`` is carried separately from the structural
    signature.  This deliberately retains distinct scientific assignments when
    their runnable structures collide.
    """

    _require_key_set(assignment, _REQUIRED_ASSIGNMENT_KEYS, "complete assignment")
    _require_member(assignment["mixing"], MIXING_PACKAGES, "mixing")
    _require_member(assignment["activation"], ACTIVATION_PACKAGES, "activation")
    _require_member(assignment["optimizer"], OPTIMIZER_FAMILIES, "optimizer")
    _require_member(assignment["lr_multiplier"], LR_MULTIPLIERS, "lr_multiplier")
    if not isinstance(science_assignment_id, str) or not science_assignment_id:
        raise ValueError("science_assignment_id must be a non-empty string")
    if not isinstance(runtime_seed, int) or isinstance(runtime_seed, bool):
        raise TypeError("runtime_seed must be an integer")
    signs = {column: _sign(assignment[column], column) for column in BREADTH_COLUMNS}
    # This checks C's amended meaning while retaining the complete A--L record.
    levels = breadth_levels(signs)
    choice = build_shared_hooke_choice_surface(
        channels=levels["A"],
        max_order=BODY_MAX_ORDER,
        max_virtual_order=levels["C"],
        path_family=levels["D"],
        aggregation=_runtime_aggregation(levels["E"]),
        initial_weight=levels["F"],
        mixing_package=assignment["mixing"],
        activation_package=assignment["activation"],
        runtime_seed=runtime_seed,
        layer_index=layer_index,
    )
    optimizer = resolve_optimizer(assignment["optimizer"], absolute_lr)
    factors = {
        column: {
            "sign": signs[column],
            "level": levels[column],
        }
        for column in BREADTH_COLUMNS
    }
    # The executable config contains all levels, including endpoints that are
    # not consumed by the generic Hooke fragment yet.  This keeps absent factors
    # visible as dictionary levels rather than silently calling them irrelevant.
    config = {
        "schema": SCHEMA,
        "version": VERSION,
        "experiment_id": "he-importance-v1",
        "science_assignment_id": science_assignment_id,
        "runtime": {"seed": runtime_seed},
        "control_reference": "CONTROL_COORDINATES=PENDING-USER-DETAILED-REVIEW",
        "factors": {
            "mixing": assignment["mixing"],
            "activation": assignment["activation"],
            "optimizer": assignment["optimizer"],
            "lr_multiplier": float(assignment["lr_multiplier"]),
            "breadth": factors,
        },
        "optimizer": optimizer,
        "model": {
            "body_max_order": BODY_MAX_ORDER,
            "layers": levels["B"],
            "channels": levels["A"],
            "max_virtual_order": levels["C"],
            "path_family": levels["D"],
            "aggregation": levels["E"],
            "mixing_initial_weight": levels["F"],
            "pfaffian_readout": levels["G"],
            "electron_electron_cusp_range": levels["H"],
            "electron_nucleus_cusp_law": levels["I"],
            "gradient_clip": levels["J"],
            "proposal_scale": levels["K"],
            "mcmc_steps_per_update": levels["L"],
            "choice_surface": choice["choices"],
        },
        "parameter_count": {
            "kind": "endpoint",
            "value": None,
            "note": "Record resolved trainable parameter count; core package effects are not parameter-count-matched causal effects.",
        },
        "evaluation": {
            "reference_energy": None,
            "reference_error": None,
            "chemical_accuracy_decision": None,
            "mode": "train plus reference-free raw/mechanics evaluation",
        },
    }
    structural = _structural_view(config)
    signature = hashlib.sha256(_canonical_json(structural).encode("utf-8")).hexdigest()
    return ResolvedConfiguration(
        science_assignment_id=science_assignment_id,
        structural_signature=signature,
        config=config,
    )


@dataclass(frozen=True)
class ResolvedConfiguration:
    """Resolved one-file config plus separate science and structure identity."""

    science_assignment_id: str
    structural_signature: str
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a copy-safe serializable representation."""

        return {
            "science_assignment_id": self.science_assignment_id,
            "structural_signature": self.structural_signature,
            "config": deepcopy(self.config),
        }

    def to_yaml(self) -> str:
        """Serialize the resolved one-file config with stable YAML bytes."""

        return canonical_yaml(self.as_dict())


def _structural_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Select normalized structure fields while excluding science identity/seed."""

    model = deepcopy(config["model"])
    # Runtime seed affects CPMLP parameters, but not structural identity.  Keep
    # stream names (layer/slot ownership) while excluding seed values.
    shared = model["choice_surface"]["shared"]
    streams = shared["cpmlp_initializer_streams"]
    shared["cpmlp_initializer_streams"] = {
        slot: {"stream": details["stream"]} for slot, details in streams.items()
    }
    for slot in model["choice_surface"]["activation"].values():
        if isinstance(slot, dict) and "initializer" in slot:
            slot["initializer"].pop("seed", None)
    return {
        "model": model,
        "optimizer": deepcopy(config["optimizer"]),
    }


def _activation_configs(
    *,
    channels: int,
    max_order: int,
    activation_package: str,
    streams: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the two explicit activation consumer slots."""

    mixing = _pointwise_activation()
    aggregation = _pointwise_activation()
    if activation_package in ("MLP-mixing", "MLP-both"):
        mixing = _cpmlp_activation(
            channels, max_order, tuple_axes_start=3, initializer=streams["mixing"]
        )
    if activation_package in ("MLP-aggregation", "MLP-both"):
        aggregation = _cpmlp_activation(
            channels, max_order, tuple_axes_start=2, initializer=streams["aggregation"]
        )
    return {"mixing": mixing, "aggregation": aggregation}


def _cpmlp_activation(
    channels: int,
    max_order: int,
    *,
    tuple_axes_start: int,
    initializer: Mapping[str, Any],
) -> dict[str, Any]:
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
                    "hidden_channels": 2 * channels,
                    "num_hidden_layers": 1,
                    "activation": {"_target_": "torch.nn.Tanh"},
                    "bias": True,
                }
                for order in range(1, max_order + 1)
            ],
        },
        "initializer": {
            "_target_": "tpen.nn.TorchInitializer",
            **dict(initializer),
        },
    }


def _pointwise_activation() -> dict[str, str]:
    return {"_target_": "torch.nn.SiLU"}


def _runtime_aggregation(value: str) -> str:
    """Translate the dictionary's display label to the TPEN boundary token."""

    if value == "completion mean":
        return "completion_mean"
    if value in ("sum", "completion_mean"):
        return value
    raise ValueError(f"unsupported completion aggregation level {value!r}")


def _interaction_config(
    *,
    channels: int,
    max_order: int,
    max_virtual_order: int,
    path_family: str,
    aggregation: str,
    initial_weight: float,
    mixing_package: str,
    activations: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a concrete producer list and shared layout for one mode."""

    linear_metadata = {
        "_target_": "tpen.data.paths.LinearPathMetadata.generate",
        "max_order": max_order,
    }
    tensor_metadata = {
        "_target_": "tpen.data.paths.PathMetadata.generate",
        "max_order": max_order,
        "max_virtual_order": max_virtual_order,
        "output_embedding": path_family,
    }
    if mixing_package == "tensor":
        families = ("tensor_product",)
        producer_configs = [
            {
                "_target_": "tpen.nn.EquivariantMixing",
                "max_order": max_order,
                "max_virtual_order": max_virtual_order,
                "channels": channels,
                "paths": deepcopy(tensor_metadata),
                "aggregation": aggregation,
                "initial_weight": initial_weight,
                "implementation": "vectorized",
                "activation": None,
            }
        ]
        layout_linear = None
        layout_tensor = tensor_metadata
        mode = "tensor_product"
    elif mixing_package == "k-GNN-like-linear":
        families = ("linear",)
        producer_configs = [
            {
                "_target_": "tpen.nn.LinearEquivariantMixing",
                "max_order": max_order,
                "channels": channels,
                "metadata": deepcopy(linear_metadata),
                "aggregation": aggregation,
                "initial_weight": initial_weight,
                "implementation": "vectorized",
            }
        ]
        layout_linear = linear_metadata
        layout_tensor = None
        mode = "linear"
    else:
        families = ("linear", "tensor_product")
        producer_configs = [
            {
                "_target_": "tpen.nn.LinearEquivariantMixing",
                "max_order": max_order,
                "channels": channels,
                "metadata": deepcopy(linear_metadata),
                "aggregation": aggregation,
                "initial_weight": initial_weight,
                "implementation": "vectorized",
            },
            {
                "_target_": "tpen.nn.EquivariantMixing",
                "max_order": max_order,
                "max_virtual_order": max_virtual_order,
                "channels": channels,
                "paths": deepcopy(tensor_metadata),
                "aggregation": aggregation,
                "initial_weight": initial_weight,
                "implementation": "vectorized",
                "activation": None,
            },
        ]
        layout_linear = linear_metadata
        layout_tensor = tensor_metadata
        mode = "hybrid"

    layout = _layout_config(
        max_order=max_order,
        channels=channels,
        linear=layout_linear,
        tensor_product=layout_tensor,
    )
    mixing = {
        "_target_": "tpen.nn.CompositeMixing",
        "layout": deepcopy(layout),
        "producers": producer_configs,
        "activation": deepcopy(activations["mixing"]),
    }
    path_aggregation = {
        "_target_": "tpen.nn.PathAggregation",
        "max_order": max_order,
        "channels": channels,
        "layout": deepcopy(layout),
        "activation": deepcopy(activations["aggregation"]),
    }
    return {
        "mode": mode,
        "producer_families": list(families),
        "layout": layout,
        "mixing": mixing,
        "path_aggregation": path_aggregation,
        "layer": {
            "_target_": "tpen.nn.TPENLayer",
            "mixing": mixing,
            "path_aggregation": path_aggregation,
            "update": {"_target_": "tpen.nn.ResidualUpdater"},
            "layout": deepcopy(layout),
        },
    }


def _layout_config(
    *,
    max_order: int,
    channels: int,
    linear: Mapping[str, Any] | None,
    tensor_product: Mapping[str, Any] | None,
) -> dict[str, Any]:
    orders = list(range(1, max_order + 1))
    channel_values = [[order, channels] for order in orders]
    return {
        "_target_": "tpen.data.paths.compose_path_layout",
        "linear": deepcopy(linear),
        "tensor_product": deepcopy(tensor_product),
        "input_orders": {"_target_": "tpen.data.paths.NormalizedOrders", "values": orders},
        "output_orders": {"_target_": "tpen.data.paths.NormalizedOrders", "values": orders},
        "input_channels": {"_target_": "tpen.data.paths.NormalizedChannels", "values": channel_values},
        "output_channels": {"_target_": "tpen.data.paths.NormalizedChannels", "values": channel_values},
    }


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return result


def _require_exact_keys(value: Mapping[str, Any], expected: Sequence[str], name: str) -> None:
    actual = tuple(value.keys())
    expected_tuple = tuple(expected)
    if actual != expected_tuple:
        missing = [key for key in expected_tuple if key not in value]
        extra = [key for key in actual if key not in expected_tuple]
        raise ValueError(f"{name} keys must be ordered {expected_tuple}; missing={missing}, extra={extra}")


def _require_key_set(value: Mapping[str, Any], expected: Sequence[str], name: str) -> None:
    """Require a mapping's keys without imposing semantic insertion order."""

    actual = set(value.keys())
    expected_set = set(expected)
    if actual != expected_set:
        missing = [key for key in expected if key not in value]
        extra = [key for key in value if key not in expected_set]
        raise ValueError(f"{name} keys differ; missing={missing}, extra={extra}")


def _require_ordered(actual: Sequence[Any], expected: Sequence[Any], name: str) -> None:
    if tuple(actual) != tuple(expected):
        raise ValueError(f"{name} must be exactly {tuple(expected)} in that order")


def _require_member(value: Any, allowed: Sequence[Any], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"unsupported {name} {value!r}; allowed={tuple(allowed)}")


def _sign(value: Any, name: str) -> int:
    if isinstance(value, bool) or value not in (-1, 1):
        raise ValueError(f"{name} must be a sign -1 or +1")
    return int(value)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a positive finite number") from error
    if result <= 0 or result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a positive finite number")
    return result


_STAGE0 = {
    "Adam": {
        "target": "torch.optim.Adam",
        "fixed": {"betas": [0.9, 0.999], "eps": 1e-8},
        "lr_candidates": [0.00125, 0.0025, 0.005, 0.010],
    },
    "RAdam": {
        "target": "torch.optim.RAdam",
        "fixed": {"betas": [0.9, 0.999], "eps": 1e-8},
        "lr_candidates": [0.00125, 0.0025, 0.005, 0.010],
    },
    "RMSprop": {
        "target": "torch.optim.RMSprop",
        "fixed": {"alpha": 0.99, "eps": 1e-8, "momentum": 0.0},
        "lr_candidates": [0.0001, 0.0003, 0.001, 0.003],
    },
    "SGD-Nesterov": {
        "target": "torch.optim.SGD",
        "fixed": {"momentum": 0.9, "nesterov": True},
        "lr_candidates": [0.001, 0.003, 0.010, 0.030],
    },
}


__all__ = [
    "ACTIVATION_PACKAGES",
    "ANCHORS",
    "BREADTH_COLUMNS",
    "BODY_MAX_ORDER",
    "DICTIONARY_PATH",
    "INDEPENDENT_BREADTH_COLUMNS",
    "LR_MULTIPLIERS",
    "MIXING_PACKAGES",
    "OPTIMIZER_FAMILIES",
    "ResolvedConfiguration",
    "breadth_levels",
    "build_shared_hooke_choice_surface",
    "canonical_yaml",
    "cpmlp_initializer_streams",
    "derive_breadth_columns",
    "iter_breadth_signs",
    "is_fraction_row",
    "load_dictionary",
    "resolve_assignment",
    "resolve_center_times_multiplier",
    "resolve_optimizer",
    "validate_dictionary",
]
