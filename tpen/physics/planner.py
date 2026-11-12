"""Explicit lowering registry and immutable local-energy plans.

This module owns composition, not the numerical implementation of any
operator.  A lowering is supplied to a planner explicitly and is selected by
the exact operator type.  The planner therefore has no vocabulary of physics
terms to maintain and no reason to inspect an operator's implementation.

The small amount of extra metadata on :class:`OperatorKernel` is deliberate:
``claimed_operator_ids`` is the executable proof that a kernel consumes a
particular semantic operator.  It permits a future fused kernel to claim
several operators while keeping the execution list unique, so accumulation
cannot silently omit an operator or add one twice.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from tpen.physics.context import (
    ContextKey,
    ContextRequirements,
    ReadOnlyContext,
)
from tpen.physics.operators import (
    OperatorId,
    is_registered_operator,
)
from tpen.physics.receipt import ExecutionReceipt, JsonScalar


type EnergyContribution = object


class OperatorKernel(Protocol):
    """Executable kernel for one or more semantic operators.

    ``claimed_operator_ids`` is a frozen set because the planner uses it as
    the manifest's ownership declaration.  A kernel may claim multiple ids;
    it still appears only once in a plan's ``kernels`` tuple.
    """

    claimed_operator_ids: frozenset[OperatorId]
    requirements: ContextRequirements

    def evaluate(self, context: ReadOnlyContext) -> EnergyContribution:
        """Evaluate this kernel against the planned context."""
        ...


class OperatorLowering[OperatorT](Protocol):
    """Lower one concrete operator type to an executable kernel."""

    operator_type: type[OperatorT]

    def lower(self, operator: OperatorT, capabilities: object) -> OperatorKernel:
        """Create the kernel that owns this operator's computation."""
        ...


@dataclass(frozen=True)
class _ValueProvider:
    """Adapt a static capability mapping into the provider contract."""

    key: ContextKey[object]
    value: object
    dependencies: ContextRequirements = frozenset()

    def build(self, context: ReadOnlyContext) -> object:
        del context
        return self.value


class OperatorLoweringRegistry:
    """Immutable exact-type mapping from operators to lowerings.

    Parameters
    ----------
    lowerings : iterable of OperatorLowering
        Explicit lowering plugins.  The registry is local to the planner and
        does not mutate any module-global state.
    """

    def __init__(
        self,
        lowerings: Iterable[object] | Mapping[type[object], object] = (),
    ) -> None:
        entries: dict[type[object], object] = {}
        if isinstance(lowerings, Mapping):
            candidates = tuple(lowerings.items())
        else:
            candidates = tuple((None, lowering) for lowering in lowerings)  # type: ignore[arg-type]
        for mapped_type, lowering in candidates:
            operator_type = lowering.operator_type
            if not isinstance(operator_type, type):
                raise TypeError("operator lowering operator_type must be a type")
            if mapped_type is not None and mapped_type is not operator_type:
                raise ValueError(
                    "lowering mapping key must match its operator_type "
                    f"({operator_type.__name__})"
                )
            if operator_type in entries:
                raise ValueError(
                    f"duplicate lowering for operator type {operator_type.__name__}"
                )
            entries[operator_type] = lowering
        self._entries = MappingProxyType(entries)

    @property
    def lowerings(self) -> Mapping[type[object], object]:
        """Return the immutable exact-type lowering table."""

        return self._entries

    def __contains__(self, operator_type: object) -> bool:
        return operator_type in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def lowering_for(self, operator_type: type[object]) -> object:
        """Return the lowering for exactly ``operator_type``."""

        if operator_type not in self._entries:
            raise KeyError(operator_type)
        return self._entries[operator_type]

    __getitem__ = lowering_for


@dataclass(frozen=True)
class LocalEnergyPlan:
    """Frozen executable plan produced by :class:`LocalEnergyPlanner`."""

    kernels: tuple[object, ...]
    providers: tuple[object, ...]
    manifest: Mapping[OperatorId, object]
    identity: str

    @property
    def provider_graph(self) -> Mapping[ContextKey[object], object]:
        """Return the frozen key-to-provider view of the planned DAG."""

        return MappingProxyType({provider.key: provider for provider in self.providers})

    @property
    def consumption_manifest(self) -> Mapping[OperatorId, object]:
        """Alias naming the manifest's exactly-once ownership role."""

        return self.manifest

    def evaluate(self, context: ReadOnlyContext) -> EnergyContribution:
        """Build planned artifacts and accumulate every kernel once.

        A new read-only context is made for each provider step.  This keeps
        provider execution local to one batch and avoids adding mutation to
        ``ReadOnlyContext`` merely for planner convenience.
        """

        entries = dict(context._entries)
        for provider in self.providers:
            provider_context = ReadOnlyContext(entries)
            entries[provider.key] = provider.build(provider_context)
        planned_context = ReadOnlyContext(entries)

        total: object | None = None
        for kernel in self.kernels:
            contribution = kernel.evaluate(planned_context)
            total = contribution if total is None else total + contribution  # type: ignore[operator]
        if total is None:
            raise ValueError("local-energy plan has no executable kernels")
        return total

    execute = evaluate


class LocalEnergyPlanner:
    """Compile explicit operators and capabilities into a validated plan."""

    def __init__(self, registry: OperatorLoweringRegistry | Iterable[object]) -> None:
        self.registry = (
            registry if isinstance(registry, OperatorLoweringRegistry)
            else OperatorLoweringRegistry(registry)
        )

    def plan(
        self,
        operators: Iterable[object],
        capabilities: object = (),
    ) -> LocalEnergyPlan:
        """Return a deterministic plan for ``operators``.

        ``capabilities`` is normally a sequence of typed providers.  A
        mapping from ``ContextKey`` to already-available values is also
        accepted as a convenience for source artifacts; those values are
        represented by explicit zero-dependency providers during planning.
        """

        operator_list = sorted(tuple(operators), key=_operator_sort_key)
        self._validate_operator_set(operator_list)

        lowered: list[object] = []
        lowerings_by_operator: dict[OperatorId, object] = {}
        owners: dict[OperatorId, object] = {}
        manifest: dict[OperatorId, object] = {}
        for operator in operator_list:
            operator_type = type(operator)
            lowering = self.registry.lowering_for(operator_type)
            kernel = lowering.lower(operator, capabilities)
            claims = kernel.claimed_operator_ids
            if not isinstance(claims, frozenset):
                raise TypeError(
                    f"lowering for {operator_type.__name__} must return a kernel "
                    "with frozenset claimed_operator_ids"
                )
            expected_id = operator_type.operator_id
            lowerings_by_operator[expected_id] = lowering
            if expected_id not in claims:
                raise ValueError(
                    f"operator {operator_type.__name__} is not claimed by its lowering"
                )
            requested_ids = {item_type.operator_id for item_type in map(type, operator_list)}
            unexpected = claims - requested_ids
            if unexpected:
                names = ", ".join(sorted(map(str, unexpected)))
                raise ValueError(
                    f"kernel for operator {operator_type.__name__} claims unplanned "
                    f"operator(s): {names}"
                )
            for operator_id in claims:
                existing = owners.get(operator_id)
                if existing is not None and existing is not kernel:
                    raise ValueError(
                        f"operator {operator_id} is claimed by more than one kernel"
                    )
                owners[operator_id] = kernel
                manifest[operator_id] = kernel
            if not any(existing is kernel for existing in lowered):
                lowered.append(kernel)

        requested_ids = {
            operator_type.operator_id for operator_type in map(type, operator_list)
        }
        missing = requested_ids - owners.keys()
        if missing:
            names = ", ".join(sorted(map(str, missing)))
            raise ValueError(f"unclaimed operator(s): {names}")
        if len(manifest) != len(requested_ids):
            raise ValueError("operator manifest is not complete and disjoint")

        requirements_by_operator: dict[OperatorId, ContextRequirements] = {}
        all_requirements: set[ContextKey[object]] = set()
        for kernel in lowered:
            requirements = kernel.requirements
            if not isinstance(requirements, frozenset):
                raise TypeError("operator kernel requirements must be a frozenset")
            all_requirements.update(requirements)
            for operator_id, owner in manifest.items():
                if owner is kernel:
                    requirements_by_operator[operator_id] = requirements

        providers = _normalise_capabilities(capabilities)
        ordered_providers = self._close_provider_graph(
            providers,
            frozenset(all_requirements),
            requirements_by_operator,
        )
        frozen_manifest = MappingProxyType(
            {operator_id: manifest[operator_id] for operator_id in sorted(manifest, key=str)}
        )
        identity = _plan_identity(
            operator_list,
            lowered,
            ordered_providers,
            frozen_manifest,
            lowerings_by_operator,
        )
        return LocalEnergyPlan(
            kernels=tuple(lowered),
            providers=tuple(ordered_providers),
            manifest=frozen_manifest,
            identity=identity,
        )

    compile = plan

    @staticmethod
    def _validate_operator_set(operators: Sequence[object]) -> None:
        seen_ids: set[OperatorId] = set()
        for operator in operators:
            operator_type = type(operator)
            if not is_registered_operator(operator_type):
                raise ValueError(
                    f"unregistered operator class {operator_type.__name__} cannot be planned"
                )
            operator_id = operator_type.operator_id
            if operator_id in seen_ids:
                raise ValueError(f"operator {operator_id} occurs more than once")
            seen_ids.add(operator_id)

    @staticmethod
    def _close_provider_graph(
        providers: Sequence[object],
        required: ContextRequirements,
        requirements_by_operator: Mapping[OperatorId, ContextRequirements],
    ) -> tuple[object, ...]:
        by_key: dict[ContextKey[object], object] = {}
        for provider in providers:
            key = provider.key
            if not isinstance(key, ContextKey):
                raise TypeError("context providers must declare a ContextKey key")
            if key in by_key:
                raise ValueError(f"multiple providers declare capability {key.name!r}")
            by_key[key] = provider

        needed: set[ContextKey[object]] = set(required)
        provenance: dict[ContextKey[object], set[OperatorId]] = {
            key: {
                operator_id
                for operator_id, operator_requirements in requirements_by_operator.items()
                if key in operator_requirements
            }
            for key in required
        }
        pending = sorted(needed, key=_context_key_sort_key)
        while pending:
            key = pending.pop(0)
            provider = by_key.get(key)
            if provider is None:
                operator_text = _operator_for_requirement(
                    provenance.get(key, set()), requirements_by_operator
                )
                raise ValueError(
                    f"operator {operator_text} requires unavailable capability {key.name!r}"
                )
            dependencies = provider.dependencies
            if not isinstance(dependencies, frozenset):
                raise TypeError("context provider dependencies must be a frozenset")
            for dependency in sorted(dependencies, key=_context_key_sort_key):
                if dependency not in needed:
                    needed.add(dependency)
                    provenance[dependency] = set(provenance[key])
                    pending.append(dependency)
                    pending.sort(key=_context_key_sort_key)

        # Kahn's algorithm with a canonical ready queue makes the resulting
        # provider order independent of the input mapping/list order.
        selected = {key: by_key[key] for key in needed}
        indegree = {
            key: sum(dependency in selected for dependency in provider.dependencies)
            for key, provider in selected.items()
        }
        dependents: dict[ContextKey[object], list[ContextKey[object]]] = {
            key: [] for key in selected
        }
        for key, provider in selected.items():
            for dependency in provider.dependencies:
                if dependency in selected:
                    dependents[dependency].append(key)
        ready = sorted(
            (key for key, degree in indegree.items() if degree == 0),
            key=_context_key_sort_key,
        )
        ordered: list[object] = []
        while ready:
            key = ready.pop(0)
            ordered.append(selected[key])
            for dependent in sorted(dependents[key], key=_context_key_sort_key):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
                    ready.sort(key=_context_key_sort_key)
        if len(ordered) != len(selected):
            cyclic = sorted(
                (key.name for key, degree in indegree.items() if degree),
            )
            operator_text = _operator_for_requirement(
                set(requirements_by_operator), requirements_by_operator
            ) if cyclic else "unknown"
            raise ValueError(
                f"operator {operator_text} has cyclic context dependency involving "
                f"{', '.join(cyclic)}"
            )
        return tuple(ordered)


def _normalise_capabilities(capabilities: object) -> tuple[object, ...]:
    if isinstance(capabilities, Mapping):
        values: list[object] = []
        for key, value in capabilities.items():
            if not isinstance(key, ContextKey):
                raise TypeError("capability mappings must be keyed by ContextKey")
            values.append(_ValueProvider(key, value))
        return tuple(sorted(values, key=lambda provider: _context_key_sort_key(provider.key)))
    return tuple(capabilities)  # type: ignore[arg-type]


def _operator_for_requirement(
    candidates: set[OperatorId],
    requirements_by_operator: Mapping[OperatorId, ContextRequirements],
) -> str:
    for operator_id in sorted(candidates, key=str):
        return str(operator_id)
    for operator_id in sorted(requirements_by_operator, key=str):
        return str(operator_id)
    return "unknown"


def _type_sort_key(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _operator_sort_key(operator: object) -> str:
    # Sort before validation so an unregistered term can receive the intended
    # diagnostic without being probed for an identity it never declared.
    return _type_sort_key(operator)


def _context_key_sort_key(key: ContextKey[object]) -> tuple[str, str]:
    return (key.name, _type_sort_key(key.value_type))


def _plan_identity(
    operators: Sequence[object],
    kernels: Sequence[object],
    providers: Sequence[object],
    manifest: Mapping[OperatorId, object],
    lowerings: Mapping[OperatorId, object],
) -> str:
    """Hash stable plan and lowering descriptors, never object addresses.

    Lowering class identity is the backend identity available in the S3
    protocol.  Parameterised instances of one class intentionally share an
    identity until a future protocol gives lowerings an explicit, stable
    semantic configuration token; hashing instance state here would require
    forbidden structural probing and would not define a cross-process format.
    """

    def kernel_position(kernel: object) -> int | None:
        for index, candidate in enumerate(kernels):
            if candidate is kernel:
                return index
        return None

    payload = {
        "operators": [
            [
                _type_sort_key(operator),
                str(type(operator).operator_id),
            ]
            for operator in operators
        ],
        "kernels": [
            [
                _type_sort_key(kernel),
                sorted(map(str, kernel.claimed_operator_ids)),
                sorted(_context_key_descriptor(key) for key in kernel.requirements),
            ]
            for kernel in kernels
        ],
        "providers": [
            [
                _type_sort_key(provider),
                _context_key_descriptor(provider.key),
                sorted(_context_key_descriptor(key) for key in provider.dependencies),
            ]
            for provider in providers
        ],
        "manifest": [
            [str(operator_id), kernel_position(kernel)]
            for operator_id, kernel in sorted(manifest.items(), key=lambda item: str(item[0]))
        ],
        "lowerings": [
            [str(operator_id), _type_sort_key(lowering)]
            for operator_id, lowering in sorted(lowerings.items(), key=lambda item: str(item[0]))
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _context_key_descriptor(key: ContextKey[object]) -> list[str]:
    return [key.name, _type_sort_key(key.value_type)]


#: Sentinel distinguishing a kernel that declares no `method_tokens`
#: attribute at all from one that declares an explicitly empty mapping.
#: `emit_receipt` fails closed on the former and accepts the latter.
_MISSING_METHOD_TOKENS = object()


def emit_receipt(plan: LocalEnergyPlan, system_id: str) -> ExecutionReceipt:
    """Return the `ExecutionReceipt` declared by a planned, executable plan.

    Narrow adapter from `LocalEnergyPlan` to `ExecutionReceipt` (S1-E
    review-r1 finding 5): `plan.manifest` maps `OperatorId` to the planned
    *kernel object* itself, not to a contribution-name tuple, and fused names
    need not match any `normalize_hamiltonian_terms` input name -- so the
    manifest cannot be copied in directly. `plan.identity` is retained
    unchanged in `numerical_policy`.

    review-r2 finding 1: `plan.identity` (`_plan_identity`) is a *structural*
    execution identity -- operator/lowering/kernel types, requirements, and
    provider/consumption topology -- and deliberately shares identity across
    parameterized instances of one lowering class (see `_plan_identity`'s
    docstring); it is not a complete method declaration. Two plans whose
    kernels are configured differently (for example a derivative step size)
    but share a class therefore share `plan.identity`. `emit_receipt` cannot
    recover that gap by introspecting `__dict__`, `repr`, or any other
    structural scan -- the design explicitly forbids that -- so it instead
    requires each kernel to *declare* its own immutable semantic method
    tokens through a `method_tokens: Mapping[str, JsonScalar]` attribute.
    Every declared token is folded into `numerical_policy`, scoped by the
    kernel's own stable column name so two different kernels' tokens with
    the same name never collide. A kernel that does not declare
    `method_tokens` at all -- not even an empty mapping -- fails closed:
    this rollout supports only kernels that state a complete method
    declaration (fixed naive-style kernels with no parameterized settings
    declare an empty mapping), never a silent, incomplete receipt.

    Every kernel gets one stable, non-stringified result-column name --
    its position in `plan.kernels` -- shared by every `OperatorId` it claims,
    so a fused kernel claiming several operators reports one column under
    all of them rather than inventing a column per operator.

    Parameters
    ----------
    plan : LocalEnergyPlan
        A plan returned by `LocalEnergyPlanner.plan`.
    system_id : str
        The physical system identity this plan was executed against, as
        returned by `tpen.physics.system.SystemDescriptor.system_id`.

    Returns
    -------
    ExecutionReceipt
        Declares `plan.identity` plus every kernel's declared `method_tokens`
        as the method-affecting settings, and one stable per-kernel column
        name per claimed `OperatorId`.

    Raises
    ------
    ValueError
        If any kernel in `plan.kernels` does not declare a `method_tokens`
        attribute.
    TypeError
        If a declared `method_tokens` is not a `Mapping`.
    """

    manifest: dict[OperatorId, list[str]] = {}
    numerical_policy: dict[str, JsonScalar] = {"plan_identity": plan.identity}
    for index, kernel in enumerate(plan.kernels):
        column = f"kernel_{index}"
        tokens = getattr(kernel, "method_tokens", _MISSING_METHOD_TOKENS)
        if tokens is _MISSING_METHOD_TOKENS:
            raise ValueError(
                f"emit_receipt: {column} ({_type_sort_key(kernel)}) does not declare "
                "'method_tokens' -- an explicit, immutable Mapping[str, JsonScalar] of any "
                "method-affecting setting not already captured by plan.identity (operator, "
                "lowering/kernel class, context requirements, and provider graph; see "
                "_plan_identity). A kernel with no parameterized settings declares an empty "
                "mapping; emit_receipt never introspects __dict__/repr to recover an omitted "
                "declaration, so an undeclared kernel is rejected rather than silently sharing "
                "a method id with a differently-configured instance of the same class"
            )
        if not isinstance(tokens, Mapping):
            raise TypeError(f"{column}.method_tokens must be a Mapping[str, JsonScalar]")
        for operator_id in kernel.claimed_operator_ids:
            manifest.setdefault(operator_id, []).append(column)
        for token_key, token_value in tokens.items():
            if not isinstance(token_key, str) or not token_key.strip():
                raise ValueError(f"{column}.method_tokens keys must be non-empty strings")
            numerical_policy[f"{column}.{token_key}"] = token_value
    return ExecutionReceipt(
        system_id=system_id,
        implementation=f"{__name__}.{LocalEnergyPlan.__qualname__}",
        version=2,
        numerical_policy=numerical_policy,
        manifest={operator_id: tuple(names) for operator_id, names in manifest.items()},
    )


__all__ = [
    "EnergyContribution",
    "LocalEnergyPlan",
    "LocalEnergyPlanner",
    "OperatorKernel",
    "OperatorLowering",
    "OperatorLoweringRegistry",
    "emit_receipt",
]
