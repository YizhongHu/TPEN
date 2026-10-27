"""Select, resolve, and optionally run exactly one Stage-Q mechanics cell.

The command deliberately owns selection rather than relying on inventory order.
It materializes the committed inventory, filters the live admitted method roster,
and requires exactly one ``(stage, context, optimizer, seed_label)`` match after
checking that the declared optimizer cell is truthful about the resolver output.
The default Q/O-raw/Adam filters therefore require an explicit seed label while
the authority contains both Q seed labels.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

from omegaconf import OmegaConf

# Direct ``python path/to/run_stage_q.py`` execution starts with the script
# directory on ``sys.path`` rather than the checkout root. Add the root before
# loading the hyphenated experiment namespace and before L1 checks its boundary.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

stage_coordinate = importlib.import_module("experiments.atomistic.he-importance.stage_coordinate")
launch = importlib.import_module("experiments.atomistic.he-importance.launch")
optimizer_grid = importlib.import_module("experiments.atomistic.he-importance.optimizer_grid")
train_config = importlib.import_module("experiments.atomistic.he-importance.train_config")

DEFAULT_STAGE = "Q"
DEFAULT_CONTEXT = "O-raw"
DEFAULT_OPTIMIZER = "adam"
Q_CHECKPOINT_EVERY_N_UPDATES = 1_000


class StageQSelectionError(ValueError):
    """The inventory does not identify exactly one admitted Stage-Q cell."""


@dataclass(frozen=True)
class CellDescriptor:
    """The auditable selection coordinates of one materialized cell."""

    stage: str
    context: str
    optimizer: Mapping[str, object]
    seed_label: int
    status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "context": self.context,
            "optimizer": self.optimizer,
            "seed_label": self.seed_label,
            "status": self.status,
        }


@dataclass(frozen=True)
class CellSelection:
    """One selected cell plus the complete discard record."""

    cell: Any
    selected: CellDescriptor
    discarded: tuple[CellDescriptor, ...]

    def receipt(self) -> dict[str, object]:
        return {
            "rule": (
                "exactly one truthful admitted cell matching stage + context + optimizer levels + seed_label; "
                "refuse zero or multiple matches"
            ),
            "selected": self.selected.as_dict(),
            "discarded": [descriptor.as_dict() for descriptor in self.discarded],
        }


def _descriptor(cell: Any) -> CellDescriptor:
    manifest = cell.manifest
    identity = manifest["scientific_identity"]
    context = identity["context"]
    optimizer = identity["optimizer"]
    optimizer_cell = identity["optimizer_cell"]
    return CellDescriptor(
        stage=str(manifest["stage"]),
        context=str(context["name"]),
        optimizer={
            "method": str(optimizer["method"]),
            "levels": dict(optimizer["levels"]),
        },
        seed_label=int(manifest["seed_identity"]["label"]),
        status=str(optimizer_cell["status"]),
    )


def _roster() -> tuple[Any, ...]:
    from tpen.hi_schema import HI_METHOD_ROSTER

    return tuple(HI_METHOD_ROSTER)


def _require_q_stage(stage: str) -> None:
    """Refuse non-Q stages before this mechanics-only route does any work.

    Parameters
    ----------
    stage : str
        Requested stage code.  This runner admits only the committed Stage-Q
        mechanics cell and must reject every other stage before materializing
        selection input or resolving a training packet.

    Raises
    ------
    StageQSelectionError
        If *stage* is not ``"Q"``.
    """

    if stage != DEFAULT_STAGE:
        raise StageQSelectionError(
            f"Stage-Q runner accepts only stage {DEFAULT_STAGE!r}; got {stage!r}"
        )


def roster_design_receipt(stage: str) -> dict[str, object]:
    """Report live roster admission against the authority's intended count."""

    roster = _roster()
    admitted = tuple(entry.method for entry in roster if entry.admitted and entry.target)
    unavailable = {
        entry.method: entry.requires for entry in roster if not entry.admitted or not entry.target
    }
    intended_count = stage_coordinate.stage_definition(stage).maximum_lineages
    if intended_count is None:
        raise StageQSelectionError(f"stage {stage!r} has no intended lineage count")
    return {
        "admitted_methods": admitted,
        "unavailable_methods": unavailable,
        "intended_count_from_authority": intended_count,
        "difference": {
            "intended_count_minus_admitted_method_count": intended_count - len(admitted),
            "dimensions": (
                "authority design runlets minus admitted method arms; "
                "reported without reconciliation"
            ),
        },
    }


def select_cell(
    cells: Iterable[Any],
    *,
    stage: str = DEFAULT_STAGE,
    context: str = DEFAULT_CONTEXT,
    optimizer: str = DEFAULT_OPTIMIZER,
    lr: float = 0.005,
    beta2: float = 0.999,
    seed_label: int | None = None,
) -> CellSelection:
    """Select one truthful admitted cell and refuse first-match fallback.

    The resolver's consumed block is read before the exact-one decision. Each
    candidate is then checked against that block, so a declared-but-unconsumed
    optimizer level cannot win merely by appearing first in the inventory.
    """

    _require_q_stage(stage)
    if context not in {DEFAULT_CONTEXT, "O-composite"}:
        raise StageQSelectionError(
            "truthfulness guard refused field 'context.name': "
            f"declared value {context!r}; only O-raw and O-composite are admitted "
            "(item dd964863)"
        )
    admitted_methods = {
        entry.method for entry in _roster() if entry.admitted and entry.target
    }
    materialized = tuple(cells)
    descriptors = tuple(_descriptor(cell) for cell in materialized)
    requested = [
        (cell, descriptor)
        for cell, descriptor in zip(materialized, descriptors)
        if descriptor.stage == stage
        and descriptor.context == context
        and descriptor.optimizer["method"] == optimizer
        and descriptor.optimizer["levels"].get("lr") == lr
        and descriptor.optimizer["levels"].get("beta2") == beta2
        and descriptor.optimizer["method"] in admitted_methods
        and descriptor.status == "available"
        and (seed_label is None or descriptor.seed_label == seed_label)
    ]
    if not requested:
        raise StageQSelectionError(
            "selection matched zero admitted cells for "
            f"stage={stage!r}, context={context!r}, optimizer={optimizer!r}, "
            f"lr={lr!r}, beta2={beta2!r}, seed_label={seed_label!r}"
        )
    truthful: list[tuple[Any, CellDescriptor]] = []
    refusals: list[str] = []
    for cell, descriptor in requested:
        try:
            # Resolve each candidate's own packet. The resolver may eventually
            # consume declared optimizer levels, so resolving requested[0]
            # once would make that first candidate win by construction.
            resolved = train_config.resolve_train_config(_q_training_packet(cell))
            _require_truthful_cell(cell, resolved)
        except train_config.TrainConfigResolutionError as error:
            # Resolver refusals remain resolver-owned, but selection reports
            # them through its typed refusal surface and never first-matches.
            refusals.append(str(error))
        except StageQSelectionError as error:
            refusals.append(str(error))
        else:
            truthful.append((cell, descriptor))

    if len(truthful) != 1:
        if not truthful:
            detail = refusals[0] if refusals else "no candidate survived the truthfulness guard"
            raise StageQSelectionError(
                "truthfulness guard rejected every selected candidate: " + detail
            )
        candidates = [descriptor.as_dict() for _, descriptor in truthful]
        raise StageQSelectionError(
            "selection is ambiguous; pass --seed-label explicitly: "
            + json.dumps(candidates, sort_keys=True)
        )
    selected_cell, selected_descriptor = truthful[0]
    discarded = tuple(
        descriptor for descriptor in descriptors if descriptor != selected_descriptor
    )
    return CellSelection(selected_cell, selected_descriptor, discarded)


def _adam_signature_defaults() -> dict[str, object]:
    """Read Adam's constructor defaults from the runtime torch installation."""

    try:
        import torch
    except ModuleNotFoundError as error:
        raise StageQSelectionError(
            "truthfulness guard requires the runtime torch.optim.Adam signature"
        ) from error

    signature = inspect.signature(torch.optim.Adam)
    defaults: dict[str, object] = {}
    for name in ("betas", "eps", "weight_decay"):
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            raise StageQSelectionError(
                f"truthfulness guard cannot read torch.optim.Adam default for {name!r}"
            )
        defaults[name] = parameter.default
    return defaults


def _scheduler_configured(resolved: Any) -> bool:
    """Return whether the consumed config declares a scheduler block."""

    config = (
        resolved
        if isinstance(resolved, Mapping)
        else OmegaConf.to_container(resolved, resolve=True)
    )

    def contains_scheduler(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                key in {"scheduler", "lr_scheduler"} and nested is not None
                or contains_scheduler(nested)
                for key, nested in value.items()
            )
        if isinstance(value, (tuple, list)):
            return any(contains_scheduler(nested) for nested in value)
        return False

    return contains_scheduler(config)


def _authority_o_raw_context() -> Mapping[str, Any]:
    """Return the exact O-raw coordinate mapping from the loaded authority."""

    return _authority_context(DEFAULT_CONTEXT)


def _authority_context(name: str) -> Mapping[str, Any]:
    """Return one named context mapping from the committed authority."""

    authority = optimizer_grid.load_authority()
    for context in authority["contexts"]:
        if context["name"] == name:
            return {
                "name": context["name"],
                "overrides": dict(context["overrides"]),
            }
    raise StageQSelectionError(f"optimizer-grid authority has no {name!r} context")


# The authority's ``feature_update`` values are labels, so keep their consumer
# mapping explicit rather than deriving an oracle from the config being checked.
# ``residual`` is verified against configs/train.yaml:model.layers[].update and
# tpen/nn/update.py:ResidualUpdater (exported from tpen.nn).
_FEATURE_UPDATE_TARGETS = {
    "residual": "tpen.nn.ResidualUpdater",
}


_ROOT_RUNNER_TWINS = ("model", "optimizer", "sampler", "trainer", "hamiltonian_terms")
_MISSING = object()

# B00 source mapping: RawCoordinateBasis is the raw-coordinate pass-through
# at tpen/nn/basis.py:189-220, with spatial_dim=3 and include_spin=True from
# the control's configs/train.yaml:model at :85-97.  Its raw one-body width is
# 3 coordinates plus one spin channel (basis.py:130-145).  Embedding's
# tpen/nn/embedding.py:91-125 defaults are in_features=None (derived raw
# width) and pair_input_channels=0.  The control leaves basis, in_features,
# and pair_input_channels absent, so the mapping admits those defaults or the
# equivalent explicit raw values, never bounded-distance features.
_B00_MODEL_REQUIREMENTS = {
    "target": "tpen.nn.TPENWaveFunction",
    "basis_targets": ("tpen.nn.RawCoordinateBasis", "tpen.nn.basis.RawCoordinateBasis"),
    "raw_basis": {"spatial_dim": 3, "include_spin": True},
    "embedding": {
        "target": "tpen.nn.Embedding",
        "spatial_dim": 3,
        "in_features": 4,
        "pair_input_channels": 0,
    },
}

_COMPOSITE_MODEL_REQUIREMENTS = {
    "target": "tpen.nn.TPENWaveFunction",
    "basis": "tpen.nn.basis.BoundedDistanceBasis",
    "raw_basis": "tpen.nn.RawCoordinateBasis",
    "embedding": "tpen.nn.embedding.matched_start_embedding",
    "embedding_target": "tpen.nn.Embedding",
    "raw_in_features": 4,
    "augmented_in_features": 5,
    "pair_input_channels": 3,
    "length": 1.0,
    "mixing": "tpen.nn.CompositeMixing",
    "linear_producer": "tpen.nn.LinearEquivariantMixing",
    "tensor_producer": "tpen.nn.EquivariantMixing",
    "activation": "tpen.nn.ChannelPreservingMLPActivation",
}

_COMPOSITE_FACTOR_TARGETS = (
    "tpen.nn.ElectronElectronCusp",
    "tpen.nn.ElectronNucleusCusp",
    "tpen.nn.jastrow.BoundedTwoCoefficientJastrow",
)

# Recompute this literal from the committed control config with:
#
#   cfg = train_config._compose_base_config(); OmegaConf.resolve(cfg)
#   model = OmegaConf.to_container(cfg.runner.model, resolve=True)
#   recursively replace only each mapping's initializer.seed value with
#   "<initializer-seed>"; json.dumps(model, sort_keys=True, separators=(",", ":"));
#   sha256 the resulting UTF-8 bytes.
#
# Any legitimate control-model edit must update this digest in review.  Only
# initializer seed values are normalized because resolver-injected cell seeds
# are expected to differ while every other control-model field is authoritative.
_CONTROL_MODEL_FINGERPRINT = "66bfb3e38b28b50bbbb23caf3bcf8947183ed864b65058eb7511278ecb0f5682"
_INITIALIZER_SEED_PLACEHOLDER = "<initializer-seed>"


def _normalize_initializer_seeds(value: Any) -> Any:
    """Normalize only values named ``seed`` inside ``initializer`` mappings."""

    if isinstance(value, Mapping):
        normalized = {
            key: _normalize_initializer_seeds(nested) for key, nested in value.items()
        }
        initializer = normalized.get("initializer")
        if isinstance(initializer, Mapping) and "seed" in initializer:
            initializer = dict(initializer)
            initializer["seed"] = _INITIALIZER_SEED_PLACEHOLDER
            normalized["initializer"] = initializer
        return normalized
    if isinstance(value, list):
        return [_normalize_initializer_seeds(nested) for nested in value]
    return value


def _model_fingerprint(model: Any) -> str:
    """Return the canonical seed-normalized fingerprint of one model mapping."""

    normalized = _normalize_initializer_seeds(model)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _seed_refusal(field: str, declared: Any, consumed: Any) -> StageQSelectionError:
    declared_text = "<missing>" if declared is _MISSING else repr(declared)
    consumed_text = "<missing>" if consumed is _MISSING else repr(consumed)
    return StageQSelectionError(
        f"truthfulness guard refused field '{field}': declared value {declared_text}, "
        f"consumed value {consumed_text}; item dd964863"
    )


def _initializer_seed_items(
    value: Any, path: str = "runner.model"
) -> tuple[tuple[str, Any], ...]:
    """Return every consumed initializer seed and its structural path."""

    found: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        initializer = value.get("initializer", _MISSING)
        if isinstance(initializer, Mapping):
            found.append((f"{path}.initializer.seed", initializer.get("seed", _MISSING)))
        for key, nested in value.items():
            found.extend(_initializer_seed_items(nested, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            found.extend(_initializer_seed_items(nested, f"{path}[{index}]"))
    return tuple(found)


def _require_seed_bindings(cell: Any, model: Any, sampler: Any) -> None:
    """Bind carrier and consumed seeds to the cell's declared seed identity."""

    seed_identity = cell.manifest["seed_identity"]
    stage = seed_identity["stage"]
    label = seed_identity["label"]
    try:
        expected_streams = stage_coordinate.seed_namespace(stage, label)
    except Exception as error:
        raise _seed_refusal(
            "cell.manifest.seed_identity", seed_identity, "valid stage/label namespace"
        ) from error

    carrier = cell.seed_streams
    if not isinstance(carrier, Mapping):
        raise _seed_refusal("cell.seed_streams", carrier, expected_streams)
    if set(carrier) != set(expected_streams):
        raise _seed_refusal("cell.seed_streams", dict(carrier), expected_streams)
    for stream, expected in expected_streams.items():
        actual = carrier[stream]
        if type(actual) is not int or actual != expected:
            raise _seed_refusal(
                f"cell.seed_streams[{stream!r}]", actual, expected
            )

    expected_model_seed = expected_streams["model_initialization"]
    initializer_items = _initializer_seed_items(model)
    if not initializer_items:
        raise _seed_refusal(
            "runner.model.initializer.seed", _MISSING, expected_model_seed
        )
    for field, actual in initializer_items:
        if type(actual) is not int or actual != expected_model_seed:
            raise _seed_refusal(field, actual, expected_model_seed)

    # train_config._set_optional_execution_seeds binds sampler.seed from the
    # training_sampler stream when that field exists (train_config.py:153-157),
    # but the consumed MetropolisSampler must always receive that binding.
    expected_sampler_seed = expected_streams["training_sampler"]
    if not isinstance(sampler, Mapping) or "seed" not in sampler:
        raise _seed_refusal("runner.sampler.seed", _MISSING, expected_sampler_seed)
    actual_sampler_seed = sampler["seed"]
    if type(actual_sampler_seed) is not int or actual_sampler_seed != expected_sampler_seed:
        raise _seed_refusal(
            "runner.sampler.seed", actual_sampler_seed, expected_sampler_seed
        )


def _first_config_difference(left: Any, right: Any, path: str) -> tuple[str, Any, Any] | None:
    """Return the first structural difference between two resolved values."""

    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, Mapping):
        for key in sorted(set(left) | set(right), key=repr):
            difference = _first_config_difference(
                left.get(key, _MISSING),
                right.get(key, _MISSING),
                f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.length", len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_config_difference(
                left_item,
                right_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if left != right:
        return path, left, right
    return None


def _agreed_consumed_views(resolved: Any) -> dict[str, Any]:
    """Require root and runner component views to be structurally identical."""

    agreed: dict[str, Any] = {}
    for name in _ROOT_RUNNER_TWINS:
        root_selected = OmegaConf.select(resolved, name, default=_MISSING)
        consumed_selected = OmegaConf.select(
            resolved, f"runner.{name}", default=_MISSING
        )
        root_view = (
            _MISSING
            if root_selected is _MISSING
            else OmegaConf.to_container(root_selected, resolve=True)
        )
        consumed_view = (
            _MISSING
            if consumed_selected is _MISSING
            else OmegaConf.to_container(consumed_selected, resolve=True)
        )
        difference = _first_config_difference(
            root_view, consumed_view, f"runner.{name}"
        )
        if difference is not None:
            path, root_value, consumed_value = difference
            raise StageQSelectionError(
                "truthfulness guard refused root/consumed divergence at "
                f"'{path}': root value {root_value!r}, consumed value {consumed_value!r}"
            )
        agreed[name] = consumed_view
    return agreed


def _input_feature_refusal(field: str, declared: Any, consumed: Any) -> StageQSelectionError:
    return StageQSelectionError(
        f"truthfulness guard refused field '{field}': declared value {declared!r}, "
        f"consumed value {consumed!r}; item dd964863"
    )


def _require_b00_model(model: Mapping[str, Any], declared: Any) -> None:
    """Require the consumed model to implement the authority's B00 mapping."""

    expected = _B00_MODEL_REQUIREMENTS
    if not isinstance(model, Mapping):
        raise _input_feature_refusal(
            "context.overrides.input_features.model",
            declared,
            model,
        )
    if model.get("_target_") != expected["target"]:
        raise _input_feature_refusal(
            "context.overrides.input_features.model._target_",
            declared,
            model.get("_target_"),
        )

    basis = model.get("basis", _MISSING)
    if basis is not _MISSING and basis is not None:
        if not isinstance(basis, Mapping):
            raise _input_feature_refusal(
                "context.overrides.input_features.model.basis",
                declared,
                basis,
            )
        target = basis.get("_target_")
        if target not in expected["basis_targets"]:
            raise _input_feature_refusal(
                "context.overrides.input_features.model.basis._target_",
                declared,
                target,
            )
        raw_basis = expected["raw_basis"]
        if basis.get("spatial_dim") != raw_basis["spatial_dim"]:
            raise _input_feature_refusal(
                "context.overrides.input_features.model.basis.spatial_dim",
                declared,
                basis.get("spatial_dim"),
            )
        if basis.get("include_spin", True) != raw_basis["include_spin"]:
            raise _input_feature_refusal(
                "context.overrides.input_features.model.basis.include_spin",
                declared,
                basis.get("include_spin", True),
            )

    embedding = model.get("embedding")
    if not isinstance(embedding, Mapping):
        raise _input_feature_refusal(
            "context.overrides.input_features.model.embedding",
            declared,
            embedding,
        )
    raw_embedding = expected["embedding"]
    if embedding.get("_target_") != raw_embedding["target"]:
        raise _input_feature_refusal(
            "context.overrides.input_features.model.embedding._target_",
            declared,
            embedding.get("_target_"),
        )
    if embedding.get("spatial_dim") != raw_embedding["spatial_dim"]:
        raise _input_feature_refusal(
            "context.overrides.input_features.model.embedding.spatial_dim",
            declared,
            embedding.get("spatial_dim"),
        )
    include_spins = embedding.get("include_spins", True)
    if include_spins is not True:
        raise _input_feature_refusal(
            "context.overrides.input_features.model.embedding.include_spins",
            declared,
            include_spins,
        )
    aux_feature_channels = embedding.get("aux_feature_channels", _MISSING)
    if (
        aux_feature_channels is not _MISSING
        and aux_feature_channels is not None
        and aux_feature_channels != {}
    ):
        raise _input_feature_refusal(
            "context.overrides.input_features.model.embedding.aux_feature_channels",
            declared,
            aux_feature_channels,
        )
    in_features = embedding.get("in_features", _MISSING)
    if (
        in_features is not _MISSING
        and in_features is not None
        and in_features != raw_embedding["in_features"]
    ):
        raise _input_feature_refusal(
            "context.overrides.input_features.model.embedding.in_features",
            declared,
            in_features,
        )
    pair_input_channels = embedding.get("pair_input_channels", _MISSING)
    if (
        pair_input_channels is not _MISSING
        and pair_input_channels != raw_embedding["pair_input_channels"]
    ):
        raise _input_feature_refusal(
            "context.overrides.input_features.model.embedding.pair_input_channels",
            declared,
            pair_input_channels,
        )
    auxiliary_width = (
        0
        if aux_feature_channels is _MISSING or aux_feature_channels is None
        else sum(aux_feature_channels.values())
    )
    derived_width = embedding["spatial_dim"] + (1 if include_spins else 0) + auxiliary_width
    effective_width = (
        derived_width
        if in_features is _MISSING or in_features is None
        else in_features
    )
    if effective_width != 4:
        raise _input_feature_refusal(
            "context.overrides.input_features.model.embedding.effective_input_width",
            declared,
            effective_width,
        )


def _require_o_composite_model(model: Mapping[str, Any], declared: Any) -> None:
    """Require the design-derived B11 model structure consumed by O-composite."""

    expected = _COMPOSITE_MODEL_REQUIREMENTS
    if not isinstance(model, Mapping):
        raise _input_feature_refusal("context.overrides.input_features.model", declared, model)
    if model.get("_target_") != expected["target"]:
        raise _input_feature_refusal("context.overrides.input_features.model._target_", declared, model.get("_target_"))

    basis = model.get("basis")
    if not isinstance(basis, Mapping) or basis.get("_target_") != expected["basis"]:
        raise _input_feature_refusal("context.overrides.input_features.model.basis._target_", declared, basis)
    inner = basis.get("inner")
    if not isinstance(inner, Mapping) or inner.get("_target_") not in {
        expected["raw_basis"],
        "tpen.nn.basis.RawCoordinateBasis",
    }:
        raise _input_feature_refusal("context.overrides.input_features.model.basis.inner._target_", declared, inner)
    for field, value in (("spatial_dim", 3), ("include_spin", True), ("length", expected["length"])):
        actual = inner.get(field) if field != "length" else basis.get(field)
        if actual != value:
            raise _input_feature_refusal(
                f"context.overrides.input_features.model.basis.{field}", declared, actual
            )

    embedding = model.get("embedding")
    if not isinstance(embedding, Mapping) or embedding.get("_target_") != expected["embedding"]:
        raise _input_feature_refusal("context.overrides.input_features.model.embedding._target_", declared, embedding)
    for member, width, pair_channels in (
        ("raw", expected["raw_in_features"], 0),
        ("augmented", expected["augmented_in_features"], expected["pair_input_channels"]),
    ):
        nested = embedding.get(member)
        if not isinstance(nested, Mapping) or nested.get("_target_") != expected["embedding_target"]:
            raise _input_feature_refusal(
                f"context.overrides.input_features.model.embedding.{member}", declared, nested
            )
        for field, value in (("spatial_dim", 3), ("in_features", width), ("pair_input_channels", pair_channels)):
            if nested.get(field, None if field != "pair_input_channels" else 0) != value:
                raise _input_feature_refusal(
                    f"context.overrides.input_features.model.embedding.{member}.{field}",
                    declared,
                    nested.get(field),
                )

    factors = model.get("factors")
    if not isinstance(factors, Sequence) or isinstance(factors, (str, bytes, bytearray)):
        raise _input_feature_refusal("context.overrides.input_features.model.factors", declared, factors)
    targets = tuple(factor.get("_target_") for factor in factors if isinstance(factor, Mapping))
    if targets != _COMPOSITE_FACTOR_TARGETS:
        raise _input_feature_refusal(
            "context.overrides.input_features.model.factors", declared, targets
        )
    control_ee = factors[0]
    if not isinstance(control_ee, Mapping) or control_ee.get("trainable_range") is not True:
        raise _input_feature_refusal(
            "context.overrides.input_features.model.factors[0]", declared, control_ee
        )
    control_en = factors[1]
    expected_law = {
        "_target_": "tpen.nn.TailSafeElectronNucleusCuspLaw",
        "curvature_range": 1.0,
        "tail_slope": 1.99,
    }
    if (
        not isinstance(control_en, Mapping)
        or control_en.get("trainable") is not True
        or control_en.get("law") != expected_law
    ):
        raise _input_feature_refusal(
            "context.overrides.input_features.model.factors[1]", declared, control_en
        )
    jastrow = factors[-1]
    expected_jastrow = {
        "_target_": _COMPOSITE_FACTOR_TARGETS[-1],
        "trainable": True,
        "length": 1.0,
        "mean_coefficient": 0.0,
        "difference_coefficient": 0.0,
    }
    if dict(jastrow) != expected_jastrow:
        raise _input_feature_refusal(
            "context.overrides.input_features.model.factors[-1]", declared, jastrow
        )

    layers = model.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes, bytearray)) or not layers:
        raise _input_feature_refusal("context.overrides.input_features.model.layers", declared, layers)
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise _input_feature_refusal(f"context.overrides.input_features.model.layers[{index}]", declared, layer)
        mixing = layer.get("mixing")
        if not isinstance(mixing, Mapping) or mixing.get("_target_") != expected["mixing"]:
            raise _input_feature_refusal(f"context.overrides.input_features.model.layers[{index}].mixing", declared, mixing)
        producers = mixing.get("producers")
        producer_targets = tuple(
            producer.get("_target_") for producer in producers
            if isinstance(producer, Mapping)
        ) if isinstance(producers, Sequence) else ()
        if producer_targets != (expected["linear_producer"], expected["tensor_producer"]):
            raise _input_feature_refusal(
                f"context.overrides.input_features.model.layers[{index}].mixing.producers",
                declared,
                producer_targets,
            )
        for site in ("mixing", "path_aggregation"):
            activation = layer.get(site, {}).get("activation") if isinstance(layer.get(site), Mapping) else None
            if not isinstance(activation, Mapping) or activation.get("_target_") != expected["activation"]:
                raise _input_feature_refusal(
                    f"context.overrides.input_features.model.layers[{index}].{site}.activation",
                    declared,
                    activation,
                )


def _context_refusal(field: str, declared: Any, consumed: Any) -> StageQSelectionError:
    return StageQSelectionError(
        f"truthfulness guard refused field '{field}': declared value {declared!r}, "
        f"consumed value {consumed!r}; context/levels are not consumed by the resolver "
        "(item dd964863)"
    )


def _configured_optimizer_value(
    optimizer: Mapping[str, Any], field: str, default: Any
) -> Any:
    """Read one Adam option, distinguishing absence from explicit null."""

    if field not in optimizer:
        return default
    value = optimizer[field]
    numeric = lambda item: isinstance(item, (int, float)) and not isinstance(item, bool)
    if field == "betas":
        valid = (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and len(value) == 2
            and all(numeric(item) for item in value)
        )
        expected = "a length-2 numeric sequence"
    else:
        valid = numeric(value)
        expected = "a number"
    if not valid:
        raise StageQSelectionError(
            f"truthfulness guard refused field 'optimizer.{field}': "
            f"declared value {value!r}, consumed value must be {expected}; "
            "explicit null does not select the runtime default"
        )
    return value


def _require_truthful_cell(cell: Any, resolved: Any) -> None:
    """Refuse a cell whose declared context or Adam levels are not consumed."""

    identity = cell.manifest["scientific_identity"]
    agreed = _agreed_consumed_views(resolved)
    declared_context = identity["context"]
    context_name = declared_context.get("name")
    if context_name not in {DEFAULT_CONTEXT, "O-composite"}:
        raise _context_refusal("context.name", context_name, "O-raw or O-composite")
    authority_context = _authority_context(context_name)
    declared_context_keys = set(declared_context)
    authority_context_keys = set(authority_context)
    for field in sorted(declared_context_keys | authority_context_keys):
        if field == "overrides":
            continue
        declared_value = declared_context.get(field)
        consumed_value = authority_context.get(field)
        if declared_value != consumed_value:
            raise _context_refusal(f"context.{field}", declared_value, consumed_value)
    if declared_context_keys != authority_context_keys:
        raise _context_refusal(
            "context", dict(declared_context), dict(authority_context)
        )
    declared_overrides = declared_context["overrides"]
    authority_overrides = authority_context["overrides"]
    declared_feature_update = declared_overrides.get("feature_update")
    residual_target = _FEATURE_UPDATE_TARGETS.get(declared_feature_update)
    if residual_target is None:
        raise _context_refusal(
            "context.overrides.feature_update",
            declared_feature_update,
            authority_overrides.get("feature_update"),
        )
    for field in sorted(set(declared_overrides) | set(authority_overrides)):
        declared_value = declared_overrides.get(field)
        consumed_value = authority_overrides.get(field)
        if declared_value != consumed_value:
            raise _context_refusal(
                f"context.overrides.{field}", declared_value, consumed_value
            )

    model = agreed["model"]
    _require_seed_bindings(cell, model, agreed["sampler"])
    if context_name == DEFAULT_CONTEXT:
        _require_b00_model(model, authority_overrides["input_features"])
    else:
        _require_o_composite_model(model, authority_overrides["input_features"])
    layers = model.get("layers", ()) if isinstance(model, Mapping) else ()
    if not layers:
        raise _context_refusal(
            "context.overrides.feature_update", "residual", None
        )
    for index, layer in enumerate(layers):
        consumed_target = layer.get("update", {}).get("_target_")
        if consumed_target != residual_target:
            raise _context_refusal(
                f"context.overrides.feature_update.model.layers[{index}].update._target_",
                "residual",
                consumed_target,
            )
    actual_fingerprint = _model_fingerprint(model)
    if context_name == DEFAULT_CONTEXT and actual_fingerprint != _CONTROL_MODEL_FINGERPRINT:
        raise StageQSelectionError(
            "truthfulness guard refused field 'runner.model.fingerprint': "
            f"expected digest {_CONTROL_MODEL_FINGERPRINT!r}, "
            f"actual digest {actual_fingerprint!r}; item dd964863"
        )

    declared_optimizer = identity["optimizer"]
    declared_method = declared_optimizer["method"]
    if declared_method != "adam":
        raise StageQSelectionError(
            "truthfulness guard refused field 'optimizer.method': "
            f"declared value {declared_method!r}, consumed value 'adam'"
        )

    declared = declared_optimizer["levels"]
    optimizer = agreed["optimizer"]
    if not isinstance(optimizer, Mapping):
        raise StageQSelectionError(
            "truthfulness guard refused field 'optimizer': declared root "
            f"value {optimizer!r}, consumed runner value {optimizer!r}"
        )
    consumed_target = optimizer.get("_target_")
    if consumed_target != "torch.optim.Adam":
        raise StageQSelectionError(
            "truthfulness guard refused field 'optimizer._target_': "
            "declared value 'torch.optim.Adam' for method 'adam', "
            f"consumed value {consumed_target!r}"
        )
    defaults = _adam_signature_defaults()
    betas = _configured_optimizer_value(optimizer, "betas", defaults["betas"])
    epsilon = _configured_optimizer_value(optimizer, "eps", defaults["eps"])
    weight_decay = _configured_optimizer_value(
        optimizer, "weight_decay", defaults["weight_decay"]
    )

    consumed = {
        "lr": optimizer.get("lr"),
        "beta1": betas[0],
        "beta2": betas[1],
        "epsilon": epsilon,
        "weight_decay": weight_decay,
        "parameter_groups": 1,
        "schedule": "constant" if not _scheduler_configured(optimizer) else "scheduler",
    }
    declared_keys = set(declared)
    consumed_keys = set(consumed)
    if declared_keys != consumed_keys:
        missing = sorted(consumed_keys - declared_keys)
        extra = sorted(declared_keys - consumed_keys)
        details = []
        if missing:
            details.append(f"missing declared key(s) {missing!r}")
        if extra:
            details.append(f"extra declared key(s) {extra!r}")
        raise StageQSelectionError(
            "truthfulness guard refused optimizer level keys: " + "; ".join(details)
        )
    for field, consumed_value in consumed.items():
        declared_value = declared.get(field)
        if declared_value != consumed_value:
            raise StageQSelectionError(
                "truthfulness guard refused optimizer level "
                f"'{field}': declared value {declared_value!r}, "
                f"consumed value {consumed_value!r}"
            )


def _q_training_packet(cell: Any) -> Any:
    definition = stage_coordinate.stage_definition(cell.manifest["stage"])
    if definition.updates is None:
        raise StageQSelectionError("selected stage has no fixed training horizon")
    cadence = stage_coordinate.CheckpointCadence(
        Q_CHECKPOINT_EVERY_N_UPDATES,
        tuple(
            range(
                Q_CHECKPOINT_EVERY_N_UPDATES,
                definition.updates + 1,
                Q_CHECKPOINT_EVERY_N_UPDATES,
            )
        ),
    )
    packets = stage_coordinate.materialize_job_packets(
        (cell,),
        cadence,
        {"statistic": "logabs_variance"},
        (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
        {"walkers": 1, "burn_in_sweeps": 0},
        ddp_provenance={"launcher": "single-process-cli"},
    )
    if len(packets.train) != 1:
        raise StageQSelectionError("Stage-Q mechanics route did not produce exactly one train packet")
    # The shared materializer has one API for all HI stages and consequently
    # constructs analysis packets in memory. Stage Q is mechanics-only: retain
    # only ``train`` here; ranking and independent-sampler packets are discarded
    # in memory and are never written, scheduled, or run by this entry point.
    return packets.train[0]


def _resolve_selected_config(selection: CellSelection) -> tuple[Any, Any]:
    """Resolve and truthfulness-check one selected packet.

    Parameters
    ----------
    selection : CellSelection
        The single cell selected by the Stage-Q admission and truthfulness
        route.

    Returns
    -------
    tuple[Any, Any]
        The materialized training packet and its resolved configuration. This
        is the single selection-side resolve-and-check route used by execute
        and by the launch-path equality test.

    Raises
    ------
    StageQSelectionError
        If packet materialization, resolution, or the consumed-model
        truthfulness checks refuse the selected cell.
    """

    packet = _q_training_packet(selection.cell)
    resolved = train_config.resolve_train_config(packet)
    _require_truthful_cell(selection.cell, resolved)
    return packet, resolved


def _execution_facts() -> dict[str, object]:
    """Collect one-process launcher facts without importing TPEN topology types."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Stage-Q launch requires one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    return {
        "global_rank": 0,
        "global_size": 1,
        "local_rank": 0,
        "local_size": 1,
        "node_rank": 0,
        "node_size": 1,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "device": "cuda:0",
        "job_id": os.environ.get("PBS_JOBID") or os.environ.get("PBS_JOB_ID"),
        "device_identity": {
            "kind": "cuda",
            "index": 0,
            "uuid": str(properties.uuid),
        },
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refuse_occupied_output_path(path: Path) -> None:
    """Atomically claim an unoccupied cell leaf before writing evidence.

    Parameters
    ----------
    path : pathlib.Path
        Deterministic cell artifact leaf for the current execution attempt.
        Parents may be created non-exclusively, but the leaf is claimed with
        ``mkdir(exist_ok=False)``.  A path is occupied when it names an
        existing file, directory, or symlink, including a dangling symlink.
        A claimed path is the writer's exclusive ownership marker; an
        occupied path is retained as prior evidence.

    Raises
    ------
    StageQSelectionError
        If *path* cannot be claimed because it is occupied.  The function
        never removes or overwrites an occupant.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise StageQSelectionError(
            f"selected cell artifact path is already occupied: {path}"
        ) from error


def _jsonable(value: Any) -> Any:
    """Convert frozen packet mappings into ordinary strict-JSON containers."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(nested) for nested in value]
    if isinstance(value, Path):
        return str(value)
    return value


def execute(args: argparse.Namespace) -> int:
    _require_q_stage(args.stage)
    output_root = Path(args.output_root).expanduser().resolve()
    cells = stage_coordinate.materialize_intended_configurations(
        output_root, availability=optimizer_grid.roster_availability()
    )
    selection = select_cell(
        cells,
        stage=args.stage,
        context=getattr(args, "context", DEFAULT_CONTEXT),
        optimizer=args.optimizer,
        lr=getattr(args, "lr", 0.005),
        beta2=getattr(args, "beta2", 0.999),
        seed_label=args.seed_label,
    )
    if args.walkers == stage_coordinate.COMMITTED_WALKERS:
        cell = selection.cell
    elif args.walkers == stage_coordinate.Q_MECHANICS_WALKERS:
        cell = stage_coordinate.with_q_mechanics_walkers(selection.cell)
    else:
        raise StageQSelectionError(f"unsupported Stage-Q walker count: {args.walkers}")
    selection = replace(selection, cell=cell)
    packet, resolved = _resolve_selected_config(selection)
    recorded_resolved = OmegaConf.to_container(resolved, resolve=True)
    report = {
        "selection": selection.receipt(),
        "walkers": stage_coordinate.walker_count(selection.cell.manifest),
        "roster_vs_design": roster_design_receipt(args.stage),
        "train_packet": {
            "content_hash": packet.cell.content_hash,
            "checkpoint_cadence": _jsonable(packet.checkpoint_cadence.science_parameters()),
            "ranking_packets": 0,
            "independent_sampler_packets": 0,
        },
        "resolved_config": recorded_resolved,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.dry_run:
        return 0

    output_path = Path(packet.cell.output_path)
    _refuse_occupied_output_path(output_path)
    _write_json(output_path / "selection_receipt.json", report)

    def _guarded_runner(config: Any, *, topology: object) -> int:
        """Validate launch.py's late resolution before delegating to TPEN."""

        _require_truthful_cell(selection.cell, config)
        launched_resolved = OmegaConf.to_container(config, resolve=True)
        if launched_resolved != recorded_resolved:
            raise StageQSelectionError(
                "launch truthfulness guard refused field 'resolved_config': "
                f"recorded value {recorded_resolved!r}, launched value {launched_resolved!r}"
            )
        return launch.run_from_config(config, topology=topology)

    exit_code = launch.launch_train(
        packet, _execution_facts(), runner=_guarded_runner
    )
    if exit_code == 0:
        resolved_snapshot = output_path / "resolved_config.yaml"
        if not resolved_snapshot.is_file():
            raise RuntimeError(f"runner returned success without {resolved_snapshot}")
        (output_path / "COMPLETE").write_text("completed\n", encoding="utf-8")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--context", default=DEFAULT_CONTEXT)
    parser.add_argument("--optimizer", default=DEFAULT_OPTIMIZER)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--seed-label", type=int)
    parser.add_argument("--walkers", type=int, default=stage_coordinate.COMMITTED_WALKERS)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("outputs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return execute(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI ramp
    raise SystemExit(main())


__all__ = [
    "CellDescriptor",
    "CellSelection",
    "StageQSelectionError",
    "build_parser",
    "main",
    "roster_design_receipt",
    "select_cell",
]
