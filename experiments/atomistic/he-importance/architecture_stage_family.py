"""Declared, non-grid architecture-stage families for the HI runner.

The registry is deliberately small and typed.  It owns admission metadata and
selection identity, while ``ab_universe`` remains the A/B renderer,
``context_mapping`` remains the composition owner, and ``stage_coordinate``
remains the materialization and seed owner.  A later stage can register a
descriptor with the same contract without adding another branch to the runner.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


stage_coordinate = importlib.import_module(
    "experiments.atomistic.he-importance.stage_coordinate"
)
ab_universe = importlib.import_module(
    "experiments.atomistic.he-importance.ab_universe"
)


class ArchitectureStageFamilyError(ValueError):
    """A family registry or descriptor is not contract-conformant."""


@dataclass(frozen=True)
class OptimizerSlot:
    """One complete optimizer policy for every rendered family row."""

    method: str
    status: str
    levels: Mapping[str, Any] | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.method or self.status not in {"available", "unavailable"}:
            raise ArchitectureStageFamilyError("optimizer slots need a method and valid status")
        if self.status == "unavailable" and not self.unavailable_reason:
            raise ArchitectureStageFamilyError("unavailable optimizer slots need a reason")
        if self.status == "available" and self.unavailable_reason is not None:
            raise ArchitectureStageFamilyError(
                "available optimizer slots cannot carry an unavailable reason"
            )
        if self.status == "available" and not isinstance(self.levels, Mapping):
            raise ArchitectureStageFamilyError("available optimizer slots need complete levels")

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "method": self.method,
            "status": self.status,
            "levels": None if self.levels is None else dict(self.levels),
        }
        if self.unavailable_reason is not None:
            result["unavailable_reason"] = self.unavailable_reason
        return result


@dataclass(frozen=True)
class FamilyCellDescriptor:
    """Generic identity-bearing coordinates returned by a family selector."""

    stage: str
    context: Any
    optimizer: OptimizerSlot
    seed_label: int
    coordinates: Any = None

    @property
    def axis(self) -> str:
        """Return the A/B axis for compatibility with the A/B selector."""

        source = self.coordinates if self.coordinates is not None else self.context
        if isinstance(source, Mapping) and "axis" in source:
            return str(source["axis"])
        return str(self.context)

    @property
    def literal_configuration(self) -> Mapping[str, str] | None:
        """Return the A/B literal coordinate when the selector provides it."""

        source = self.coordinates if self.coordinates is not None else self.context
        if isinstance(source, Mapping) and "literal_configuration" in source:
            literal = source["literal_configuration"]
            return dict(literal) if isinstance(literal, Mapping) else None
        return None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "stage": self.stage,
            "context": self.context,
            "optimizer": self.optimizer.as_dict(),
            "seed_label": self.seed_label,
            "status": self.optimizer.status,
        }
        literal = self.literal_configuration
        if literal is not None:
            result["literal_configuration"] = literal
        return result


@dataclass(frozen=True)
class FamilySelection:
    """One exact family selection and its discarded identity records."""

    cell: Any
    selected: FamilyCellDescriptor
    discarded: tuple[FamilyCellDescriptor, ...]

    def receipt(self) -> dict[str, Any]:
        return {
            "rule": (
                "exactly one truthful family cell matching stage + selector-owned "
                "context + optimizer slot + seed_label; refuse zero or multiple matches"
            ),
            "selected": self.selected.as_dict(),
            "discarded": [item.as_dict() for item in self.discarded],
        }


Resolver = Callable[[Any], Any]
ContextValidator = Callable[[Mapping[str, Any]], None]
ResolvedValidator = Callable[[Any, Any], None]
Renderer = Callable[[str], Sequence[Mapping[str, Any]]]
SlotPolicy = Callable[[Mapping[str, Any]], Sequence[OptimizerSlot]]
Selector = Callable[..., FamilySelection]


@dataclass(frozen=True)
class ArchitectureStageDescriptor:
    """Static admission contract for one member of an architecture family."""

    stage: str
    family: str
    renderer: Renderer
    context_validator: ContextValidator
    resolver: Resolver
    resolved_validator: ResolvedValidator
    slot_policy: SlotPolicy
    seed_namespace: Callable[[str], tuple[int, ...]]
    selector: Selector
    packet_policy: Callable[[str], Mapping[str, Any]]
    launch_adapter: Callable[..., Any]


STAGE_AUTHORITY: Mapping[str, Any] = stage_coordinate.STAGES
ARCHITECTURE_STAGE_REGISTRY: dict[str, ArchitectureStageDescriptor] = {}

_AB_RENDER_CACHE: tuple[object, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None
_AB_LITERAL_KEYS: dict[str, frozenset[str]] = {}
_AB_LITERAL_VALUES: dict[str, frozenset[str]] = {}


def _canonical(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, OptimizerSlot):
            return item.as_dict()
        if isinstance(item, Mapping):
            return {str(key): normalize(nested) for key, nested in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(nested) for nested in item]
        return item

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"))


def _resolver(source: Any) -> Any:
    module = importlib.import_module("experiments.atomistic.he-importance.train_config")
    return module.resolve_train_config(source)


def _ab_universe_snapshot() -> Mapping[str, Sequence[Mapping[str, Any]]]:
    """Memoize the renderer output while preserving test-time replacement."""

    global _AB_RENDER_CACHE, _AB_LITERAL_KEYS, _AB_LITERAL_VALUES
    renderer = ab_universe.render_ab_universe
    if _AB_RENDER_CACHE is None or _AB_RENDER_CACHE[0] is not renderer:
        rendered = renderer()
        if not isinstance(rendered, Mapping):
            raise ArchitectureStageFamilyError("A/B renderer must return a stage mapping")
        _AB_RENDER_CACHE = (renderer, rendered)
        _AB_LITERAL_KEYS = {
            stage: frozenset(package["literal_configuration"])
            for stage, packages in rendered.items()
            for package in packages[:1]
        }
        _AB_LITERAL_VALUES = {
            stage: frozenset(_canonical(package["literal_configuration"]) for package in packages)
            for stage, packages in rendered.items()
        }
    return _AB_RENDER_CACHE[1]


def _validate_ab_context(context: Mapping[str, Any]) -> None:
    if frozenset(context) != frozenset({"axis", "literal_configuration"}):
        raise ArchitectureStageFamilyError(
            "A/B context must contain exactly axis and literal_configuration"
        )
    axis = context["axis"]
    literal = context["literal_configuration"]
    if axis not in {"A", "B"} or not isinstance(literal, Mapping):
        raise ArchitectureStageFamilyError("A/B context has an invalid axis or literal configuration")
    if any(type(key) is not str or type(value) is not str for key, value in literal.items()):
        raise ArchitectureStageFamilyError("A/B literal configuration must map exact strings to strings")
    _ab_universe_snapshot()
    expected = _AB_LITERAL_KEYS[axis]
    if frozenset(literal) != frozenset(expected):
        raise ArchitectureStageFamilyError(
            f"A/B literal configuration for {axis} has a closed-vocabulary mismatch"
        )
    if _canonical(dict(literal)) not in _AB_LITERAL_VALUES[axis]:
        raise ArchitectureStageFamilyError(
            f"A/B literal configuration for {axis} is not in the rendered universe"
        )


def _validate_ab_resolved(cell: Any, resolved: Any) -> None:
    manifest = cell.manifest
    context = manifest["scientific_identity"]["context"]
    _validate_ab_context(context)
    optimizer = manifest["scientific_identity"]["optimizer"]
    if optimizer["method"] != "adam":
        raise ArchitectureStageFamilyError(
            f"family stage {manifest['stage']!r} cannot resolve optimizer {optimizer['method']!r}"
        )
    target = getattr(getattr(resolved, "optimizer", None), "_target_", None)
    if target != "torch.optim.Adam":
        raise ArchitectureStageFamilyError(
            f"family stage {manifest['stage']!r} consumed optimizer target {target!r}, expected Adam"
        )
    train_config = importlib.import_module("experiments.atomistic.he-importance.train_config")
    context_mapping = importlib.import_module(
        "experiments.atomistic.he-importance.context_mapping"
    )
    axis = str(context["axis"])
    labels = train_config._ab_context_labels(manifest, axis)
    package = labels.pop("package", None)
    model_seed = int(cell.seed_streams["model_initialization"])
    expected_config = train_config._compose_base_config()
    expected_config.model = context_mapping.compose_architecture_package(
        expected_config.model,
        package=package,
        seed=model_seed,
        **labels,
    )
    train_config._set_named_initializer_seeds(expected_config, model_seed)
    OmegaConf.resolve(expected_config)
    expected_model = OmegaConf.to_container(expected_config.model, resolve=True)
    actual_model = OmegaConf.to_container(resolved.model, resolve=True)
    if _canonical(actual_model) != _canonical(expected_model):
        raise ArchitectureStageFamilyError(
            f"family stage {manifest['stage']!r} resolved model does not match "
            "context_mapping.compose_architecture_package for its literal configuration"
        )


def _ab_renderer(stage: str) -> Sequence[Mapping[str, Any]]:
    if stage not in {"A", "B"}:
        raise ArchitectureStageFamilyError(f"A/B renderer cannot render stage {stage!r}")
    rows: list[Mapping[str, Any]] = []
    for package in _ab_universe_snapshot()[stage]:
        context = {
            "axis": stage,
            "literal_configuration": dict(package["literal_configuration"]),
        }
        for slot in _ab_slot_policy(package):
            rows.append(
                {
                    "stage": stage,
                    "package": package,
                    "optimizer": slot.as_dict(),
                    "configurations": (
                        {
                            "scientific_identity": {
                                "context": context,
                                "optimizer": {
                                    "method": slot.method,
                                    "levels": None if slot.levels is None else dict(slot.levels),
                                },
                            },
                            "payload": {"updates": stage_coordinate.stage_definition(stage).updates},
                            "topology": {},
                        },
                    ),
                }
            )
    return rows


def _ab_slot_policy(package: Mapping[str, Any]) -> tuple[OptimizerSlot, ...]:
    slots = package.get("optimizer_slots")
    if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
        raise ArchitectureStageFamilyError("A/B package optimizer_slots must be a sequence")
    result = tuple(
        OptimizerSlot(
            method=str(slot["method"]),
            status=str(slot["status"]),
            levels=slot.get("levels"),
            unavailable_reason=slot.get("unavailable_reason"),
        )
        for slot in slots
        if isinstance(slot, Mapping)
    )
    if len(result) != len(slots) or {slot.method for slot in result} != {"adam", "O*"}:
        raise ArchitectureStageFamilyError("each A/B package must declare Adam and O* exactly once")
    return result


def _ab_packet_policy(stage: str) -> Mapping[str, Any]:
    definition = stage_coordinate.stage_definition(stage)
    if definition.updates is None:
        raise ArchitectureStageFamilyError(f"family stage {stage!r} has no fixed horizon")
    return {"checkpoint_every_n_updates": 1_000, "updates": definition.updates}


def _ab_launch_adapter(*args: Any, **kwargs: Any) -> Any:
    launch = importlib.import_module("experiments.atomistic.he-importance.launch")
    return launch.launch_train(*args, **kwargs)


def _ab_seed_namespace(stage: str) -> tuple[int, ...]:
    return stage_coordinate.seed_labels(stage)


def validate_descriptor(
    descriptor: ArchitectureStageDescriptor,
    *,
    authority: Mapping[str, Any] | None = None,
) -> None:
    """Validate one descriptor against the supplied stage authority."""

    authority = STAGE_AUTHORITY if authority is None else authority
    if not descriptor.stage or not descriptor.family:
        raise ArchitectureStageFamilyError("family descriptors need stage and family names")
    if descriptor.stage not in authority:
        raise ArchitectureStageFamilyError(
            f"family stage {descriptor.stage!r} is absent from the stage authority"
        )
    for name in (
        "renderer",
        "context_validator",
        "resolver",
        "resolved_validator",
        "slot_policy",
        "seed_namespace",
        "selector",
        "packet_policy",
        "launch_adapter",
    ):
        if not callable(getattr(descriptor, name)):
            raise ArchitectureStageFamilyError(f"descriptor member {name!r} must be callable")
    first = tuple(descriptor.renderer(descriptor.stage))
    second = tuple(descriptor.renderer(descriptor.stage))
    if _canonical(first) != _canonical(second):
        raise ArchitectureStageFamilyError("family renderer must be deterministic")
    if not first:
        raise ArchitectureStageFamilyError("family renderer must produce at least one row")
    labels = tuple(descriptor.seed_namespace(descriptor.stage))
    if not labels or len(set(labels)) != len(labels):
        raise ArchitectureStageFamilyError("family seed namespace must be non-empty and unique")
    definition = authority[descriptor.stage]
    updates = getattr(definition, "updates", definition.get("updates") if isinstance(definition, Mapping) else None)
    if updates is None:
        raise ArchitectureStageFamilyError("family descriptor requires a fixed authority horizon")
    if labels != tuple(stage_coordinate.seed_labels(descriptor.stage)):
        raise ArchitectureStageFamilyError(
            f"family seed namespace for {descriptor.stage!r} does not match the stage authority"
        )
    packet_policy = descriptor.packet_policy(descriptor.stage)
    if not isinstance(packet_policy, Mapping):
        raise ArchitectureStageFamilyError("family packet policy must return a mapping")
    if packet_policy.get("updates") != updates:
        raise ArchitectureStageFamilyError("family packet policy does not bind authority horizon")
    cadence = packet_policy.get("checkpoint_every_n_updates")
    if type(cadence) is not int or cadence <= 0 or cadence > updates:
        raise ArchitectureStageFamilyError(
            "family packet policy checkpoint cadence must be a positive integer "
            f"no greater than the horizon ({updates})"
        )
    for row in first:
        if row.get("stage") != descriptor.stage:
            raise ArchitectureStageFamilyError("family renderer emitted a row for another stage")
        configurations = row.get("configurations")
        if not isinstance(configurations, Sequence) or len(configurations) != 1:
            raise ArchitectureStageFamilyError("family rows must contain one literal configuration")
        payload = configurations[0].get("payload")
        if not isinstance(payload, Mapping) or payload.get("updates") != updates:
            raise ArchitectureStageFamilyError(
                "family row payload updates must match the packet policy horizon"
            )
        identity = configurations[0].get("scientific_identity")
        if not isinstance(identity, Mapping):
            raise ArchitectureStageFamilyError("family rows need a scientific identity")
        descriptor.context_validator(identity.get("context", {}))
        source = row.get("package", row)
        if not isinstance(source, Mapping):
            raise ArchitectureStageFamilyError("family rows need a mapping for slot policy")
        slots = tuple(descriptor.slot_policy(source))
        if not slots or len({slot.method for slot in slots}) != len(slots):
            raise ArchitectureStageFamilyError("family rows need one complete policy per optimizer slot")
        if row["optimizer"]["method"] not in {slot.method for slot in slots}:
            raise ArchitectureStageFamilyError("row optimizer is absent from its registry-owned slot policy")


def register_stage_descriptor(
    descriptor: ArchitectureStageDescriptor,
    *,
    registry: dict[str, ArchitectureStageDescriptor] | None = None,
    authority: Mapping[str, Any] | None = None,
) -> ArchitectureStageDescriptor:
    """Validate and register a descriptor, primarily for test-only extensions."""

    registry = ARCHITECTURE_STAGE_REGISTRY if registry is None else registry
    validate_descriptor(descriptor, authority=authority)
    if descriptor.stage in registry:
        raise ArchitectureStageFamilyError(f"family stage {descriptor.stage!r} is already registered")
    registry[descriptor.stage] = descriptor
    return descriptor


def family_stage_codes(
    registry: Mapping[str, ArchitectureStageDescriptor] | None = None,
) -> tuple[str, ...]:
    registry = ARCHITECTURE_STAGE_REGISTRY if registry is None else registry
    codes = tuple(registry)
    if len(codes) != len(set(codes)):
        raise ArchitectureStageFamilyError("family registry repeats a stage code")
    for code, descriptor in registry.items():
        if code != descriptor.stage:
            raise ArchitectureStageFamilyError("family registry key does not match descriptor stage")
    return codes


def is_family_stage(stage: str) -> bool:
    return stage in family_stage_codes()


def descriptor_for(stage: str) -> ArchitectureStageDescriptor:
    try:
        return ARCHITECTURE_STAGE_REGISTRY[stage]
    except KeyError as error:
        raise ArchitectureStageFamilyError(f"stage {stage!r} is not a declared family member") from error


def render_family_inventory(stage: str) -> tuple[Mapping[str, Any], ...]:
    descriptor = descriptor_for(stage)
    validate_descriptor(descriptor)
    return tuple(descriptor.renderer(stage))


def materialize_family_stage(stage: str, output_root: Path) -> tuple[Any, ...]:
    """Materialize every family row through the shared stage-coordinate API."""

    descriptor = descriptor_for(stage)
    rows = render_family_inventory(stage)
    cells: list[Any] = []
    for row in rows:
        slot = next(
            item
            for item in descriptor.slot_policy(row.get("package", row))
            if item.method == row["optimizer"]["method"]
        )
        optimizer = stage_coordinate.OptimizerCell(
            slot.method,
            slot.status,
            slot.unavailable_reason,
        )
        cells.extend(
            stage_coordinate.materialize_stage(
                stage,
                row["configurations"],
                optimizer,
                output_root,
            )
        )
    return tuple(cells)


def _cell_descriptor(cell: Any) -> FamilyCellDescriptor:
    identity = cell.manifest["scientific_identity"]
    optimizer = identity["optimizer"]
    slot_identity = identity["optimizer_cell"]
    slot = OptimizerSlot(
        method=str(optimizer["method"]),
        status=str(slot_identity["status"]),
        levels=stage_coordinate._thaw(optimizer.get("levels")),
        unavailable_reason=slot_identity.get("unavailable_reason"),
    )
    context = identity["context"]
    return FamilyCellDescriptor(
        stage=str(cell.manifest["stage"]),
        context=stage_coordinate._thaw(context),
        optimizer=slot,
        seed_label=int(cell.manifest["seed_identity"]["label"]),
    )


def select_exact_one(
    cells: Iterable[Any],
    *,
    stage: str,
    context: str | Mapping[str, Any],
    optimizer: str,
    seed_label: int | None,
    literal_configuration: Mapping[str, str] | None = None,
    lr: float | None = None,
    beta2: float | None = None,
) -> FamilySelection:
    """Select one A/B row by its literal identity and optimizer slot."""

    descriptor = descriptor_for(stage)
    if isinstance(context, Mapping):
        descriptor.context_validator(context)
        axis = str(context["axis"])
        requested_literal = dict(context["literal_configuration"])
    else:
        axis = context
        requested_literal = None if literal_configuration is None else dict(literal_configuration)
    if axis != stage:
        raise ArchitectureStageFamilyError(
            f"family selection context {axis!r} does not match stage {stage!r}"
        )
    if requested_literal is not None:
        descriptor.context_validator({"axis": axis, "literal_configuration": requested_literal})
    materialized = tuple(cells)
    all_descriptors = tuple(_cell_descriptor(cell) for cell in materialized)
    matched: list[tuple[Any, FamilyCellDescriptor]] = []
    for cell, item in zip(materialized, all_descriptors):
        if item.stage != stage or item.axis != axis:
            continue
        if requested_literal is not None and dict(item.literal_configuration) != requested_literal:
            continue
        if item.optimizer.method != optimizer:
            continue
        if optimizer == "adam" and lr is not None:
            if not isinstance(item.optimizer.levels, Mapping):
                continue
            if item.optimizer.levels.get("lr") != lr or item.optimizer.levels.get("beta2") != beta2:
                continue
        if seed_label is not None and item.seed_label != seed_label:
            continue
        matched.append((cell, item))
    if not matched:
        raise ArchitectureStageFamilyError(
            "selection matched zero family cells for "
            f"stage={stage!r}, context={axis!r}, literal_configuration={requested_literal!r}, "
            f"optimizer={optimizer!r}, seed_label={seed_label!r}"
        )
    if len(matched) != 1:
        raise ArchitectureStageFamilyError(
            "selection is ambiguous for "
            f"stage={stage!r}, context={axis!r}, literal_configuration={requested_literal!r}, "
            f"optimizer={optimizer!r}, seed_label={seed_label!r}: {len(matched)} matches"
        )
    cell, selected = matched[0]
    if selected.optimizer.status != "available":
        raise ArchitectureStageFamilyError(
            f"optimizer slot {selected.optimizer.method!r} is unavailable: "
            f"{selected.optimizer.unavailable_reason}"
        )
    resolved = descriptor.resolver(cell)
    descriptor.resolved_validator(cell, resolved)
    selected_public = FamilyCellDescriptor(
        stage=selected.stage,
        context=axis,
        optimizer=selected.optimizer,
        seed_label=selected.seed_label,
        coordinates=selected.context,
    )
    discarded = tuple(
        FamilyCellDescriptor(
            stage=item.stage,
            context=item.axis,
            optimizer=item.optimizer,
            seed_label=item.seed_label,
            coordinates=item.context,
        )
        for item in all_descriptors
        if item != selected
    )
    return FamilySelection(cell, selected_public, discarded)


def validate_partition(
    declared: Sequence[str],
    family_codes: Sequence[str],
    undeclared: Mapping[str, str],
    authority_codes: Iterable[str],
) -> frozenset[str]:
    """Check a duplicate-sensitive three-way authority partition."""

    declared_tuple = tuple(declared)
    family_tuple = tuple(family_codes)
    if len(declared_tuple) != len(set(declared_tuple)):
        raise ArchitectureStageFamilyError("optimizer-grid stages repeat a stage code")
    if len(family_tuple) != len(set(family_tuple)):
        raise ArchitectureStageFamilyError("family registry repeats a stage code")
    declared_set, family_set, undeclared_set = set(declared_tuple), set(family_tuple), set(undeclared)
    if (
        declared_set & family_set
        or declared_set & undeclared_set
        or family_set & undeclared_set
        or declared_set | family_set | undeclared_set != set(authority_codes)
    ):
        raise ArchitectureStageFamilyError(
            "stage authority must be partitioned into optimizer-grid, family, and undeclared stages: "
            f"grid={sorted(declared_set)}, family={sorted(family_set)}, "
            f"undeclared={sorted(undeclared_set)}, authority={sorted(set(authority_codes))}"
        )
    return frozenset(authority_codes)


_A_DESCRIPTOR = ArchitectureStageDescriptor(
    stage="A",
    family="architecture",
    renderer=_ab_renderer,
    context_validator=_validate_ab_context,
    resolver=_resolver,
    resolved_validator=_validate_ab_resolved,
    slot_policy=_ab_slot_policy,
    seed_namespace=_ab_seed_namespace,
    selector=select_exact_one,
    packet_policy=_ab_packet_policy,
    launch_adapter=_ab_launch_adapter,
)
_B_DESCRIPTOR = ArchitectureStageDescriptor(
    stage="B",
    family="architecture",
    renderer=_ab_renderer,
    context_validator=_validate_ab_context,
    resolver=_resolver,
    resolved_validator=_validate_ab_resolved,
    slot_policy=_ab_slot_policy,
    seed_namespace=_ab_seed_namespace,
    selector=select_exact_one,
    packet_policy=_ab_packet_policy,
    launch_adapter=_ab_launch_adapter,
)
register_stage_descriptor(_A_DESCRIPTOR)
register_stage_descriptor(_B_DESCRIPTOR)


__all__ = [
    "ARCHITECTURE_STAGE_REGISTRY",
    "ArchitectureStageDescriptor",
    "ArchitectureStageFamilyError",
    "FamilyCellDescriptor",
    "FamilySelection",
    "OptimizerSlot",
    "STAGE_AUTHORITY",
    "descriptor_for",
    "family_stage_codes",
    "is_family_stage",
    "materialize_family_stage",
    "register_stage_descriptor",
    "render_family_inventory",
    "select_exact_one",
    "validate_descriptor",
    "validate_partition",
]
