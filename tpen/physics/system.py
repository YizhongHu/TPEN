"""Physical system identity: versioned semantic declarations.

This module owns the *physical-system identity* repair described in
``system-context-records-design-r{1,2,3}.md``: an immutable, versioned
description of the actual configured physics, built from the actual
Hamiltonian terms rather than from a scanner, molecule name, or separately
maintained config mirror.

Three values compose the contract:

``HamiltonianTermDescriptor``
    One operator's explicit, versioned physical declaration: its
    :class:`~tpen.physics.operators.OperatorId`, fixed numeric parameters, and
    an optional typed resource identity (for example a nuclear geometry
    content id).  A term declares this through
    :class:`DescribesPhysicalOperator`; nothing infers it by scanning
    attributes, class names, or molecule labels.
``ParticleSpec``
    Explicit electron count, spin sector, and spatial dimension. Never
    inferred from nuclear charge or by probing a container.
``SystemDescriptor``
    The immutable aggregate of a ``ParticleSpec`` and the ordered term
    declarations actually constructed for one Hamiltonian. Its
    :meth:`SystemDescriptor.system_id` is the canonical content identity
    published through :data:`SYSTEM_DESCRIPTOR_KEY`. Its
    :attr:`SystemDescriptor.resources` carries one immutable, deduplicated
    `AtomicConfiguration` snapshot per distinct resource reference declared by
    its terms (for example a shared nuclear geometry referenced by both an
    electron-nucleus and a nucleus-nucleus term).

External context is compositional: a system may declare zero, one, or several
external-potential terms (a harmonic trap, nuclei, or both at once); absence
of a term is simply its absence from :attr:`SystemDescriptor.terms`, never a
null placeholder standing in for "no nucleus".

Out of scope here (see the S1-D acceptance contract): numerical lowering
choice, a planner, trajectory records, and the execution receipt. Those are
separate slices that consume ``system_id`` as an opaque value.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.physics.context import ContextKey, ContextRequirements, ReadOnlyContext, requirements
from tpen.physics.hamiltonian import normalize_hamiltonian_terms
from tpen.physics.operators import OperatorId, is_registered_operator

#: JSON-safe scalar accepted as a term parameter value. Deliberately narrow:
#: a canonical content id must encode every accepted value deterministically,
#: and ``bool``/``int``/``float``/``str`` cover every physical parameter
#: shipped today (trap frequency, distance floor, nuclear-geometry id).
JsonScalar = bool | int | float | str

#: v2 (r1 fix): every variable-length field (operator namespace/name, string
#: parameter values, parameter count, resource id) is now framed with an
#: explicit big-endian length/count prefix; see `_encode_length_prefixed`.
#: Under v1, two fields concatenated without a boundary could produce
#: identical bytes for different content -- e.g. parameters
#: ``{"a": "bs", "b": ""}`` and ``{"a": "", "b": "bs"}`` both encoded to
#: ``b"asbsbs"`` and therefore hashed to the same `system_id`.
_DESCRIPTOR_SCHEMA_VERSION = 2


def _encode_length_prefixed(data: bytes) -> bytes:
    """Return `data` framed with an explicit 8-byte big-endian length prefix.

    Concatenating unframed variable-length byte strings is ambiguous: the
    boundary between two fields can move without changing the overall bytes
    (see the v1->v2 note on `_DESCRIPTOR_SCHEMA_VERSION`). A length prefix
    fixes the boundary so canonical encoding is injective in each field.
    """

    return struct.pack(">Q", len(data)) + data


def _encode_scalar(value: JsonScalar) -> bytes:
    """Return a type-tagged, deterministic, length-framed byte encoding.

    The type tag keeps ``1`` (int) and ``1.0`` (float) from colliding, and
    keeps ``True`` (bool) from colliding with ``1`` (int): Python's ``bool``
    is an ``int`` subclass, so the ``bool`` check must run first. The ``str``
    case is length-framed (see `_encode_length_prefixed`); the other cases
    are fixed-width and need no framing.
    """

    if isinstance(value, bool):
        return b"b" + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        return b"i" + struct.pack(">q", value)
    if isinstance(value, float):
        return b"f" + struct.pack(">d", value)
    if isinstance(value, str):
        return b"s" + _encode_length_prefixed(value.encode("utf-8"))
    raise TypeError(f"term parameter values must be bool, int, float, or str; got {type(value).__name__}")


def _canonical_parameters(parameters: Mapping[str, JsonScalar]) -> MappingProxyType[str, JsonScalar]:
    """Validate and freeze a parameter mapping, failing loudly on bad input.

    Rejects non-finite floats (NaN, +inf, -inf): a canonical content id must
    be a deterministic function of the declared physics, and a non-finite
    value is both unphysical for every parameter shipped today and not
    representable by strict JSON. Canonicalizes ``-0.0`` to ``0.0`` so that
    physically-equivalent signed zero does not change `system_id`.
    """

    validated: dict[str, JsonScalar] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("term parameter names must be non-empty strings")
        if not isinstance(value, (bool, int, float, str)):
            raise TypeError(
                f"term parameter {key!r} must be bool, int, float, or str; got {type(value).__name__}"
            )
        if isinstance(value, float) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError(f"term parameter {key!r} must be finite, got {value!r}")
            if value == 0.0:
                value = 0.0
        validated[key] = value
    return MappingProxyType(validated)


@dataclass(frozen=True)
class ParticleSpec:
    """Explicit particle count, spin sector, and spatial dimension.

    Parameters
    ----------
    n_electrons : int
        Total electron count.
    n_spin_up : int
        Electrons assigned to the up sector.
    n_spin_down : int
        Electrons assigned to the down sector. Must sum with ``n_spin_up`` to
        ``n_electrons``; there is no spinless convention that omits this
        split, so a spin-agnostic caller still states an explicit sector.
    spatial_dim : int
        Spatial dimension of each electron coordinate.

    Notes
    -----
    Every field is stated explicitly. None is derived from nuclear charge,
    a batch shape, or any other container; a caller that does not know its
    own particle count or spin split has nothing to build a system from.
    """

    n_electrons: int
    n_spin_up: int
    n_spin_down: int
    spatial_dim: int

    def __post_init__(self) -> None:
        if self.n_electrons < 0:
            raise ValueError("ParticleSpec.n_electrons must be non-negative")
        if self.n_spin_up < 0 or self.n_spin_down < 0:
            raise ValueError("ParticleSpec spin counts must be non-negative")
        if self.n_spin_up + self.n_spin_down != self.n_electrons:
            raise ValueError(
                "ParticleSpec.n_spin_up + n_spin_down must equal n_electrons, got "
                f"{self.n_spin_up} + {self.n_spin_down} != {self.n_electrons}"
            )
        if self.spatial_dim < 1:
            raise ValueError("ParticleSpec.spatial_dim must be at least 1")

    def canonical_bytes(self) -> bytes:
        """Return the deterministic byte encoding used by `SystemDescriptor.system_id`."""

        return struct.pack(">qqqq", self.n_electrons, self.n_spin_up, self.n_spin_down, self.spatial_dim)


@dataclass(frozen=True)
class HamiltonianTermDescriptor:
    """One operator's explicit, versioned physical declaration.

    Parameters
    ----------
    operator_id : OperatorId
        The declared physical operator (see `tpen.physics.operators`).
    slot : str
        The configured term name used for the energy decomposition (the key
        `normalize_hamiltonian_terms` resolved for this term). Carried for
        execution/record binding; excluded from `system_id` so that renaming
        a configured slot cannot change physical identity.
    parameters : Mapping[str, JsonScalar], optional
        Fixed, effective numeric parameters of this operator (for example a
        harmonic trap's ``omega``). Values are validated and frozen.
    resource_id : str or None, optional
        Content identity of a typed fixed resource this operator references
        (for example `AtomicConfiguration.content_id`). ``None`` when the
        operator references no resource.
    resource : AtomicConfiguration or None, optional
        The actual typed resource payload identified by `resource_id` (for
        example the construction-time nuclear geometry). Must be provided
        together with `resource_id`, and its `AtomicConfiguration.content_id`
        must equal `resource_id`. Snapshotted into an independently-owned
        copy at construction (never the caller's own object), and excluded
        from `canonical_bytes` (`resource_id` already determines identity).
        Reading the `resource` attribute returns a fresh clone of that
        snapshot on every access (copy-on-read): mutating a previously
        returned payload in place can never change what this descriptor, or
        `SystemDescriptor.resources`, publish.
    version : int, optional
        Declaration version for this operator's description. Bump when the
        meaning of `parameters` changes for a given `operator_id`.

    Notes
    -----
    Two terms sharing one `operator_id` within one `SystemDescriptor` are
    rejected: the planner's exactly-once manifest already forbids repeated
    operator identities, and a system description must not silently disagree
    with that limitation (see r2 of the system-context design).
    """

    operator_id: OperatorId
    slot: str
    parameters: Mapping[str, JsonScalar] = field(default_factory=dict)
    resource_id: str | None = None
    version: int = 1
    #: Constructor-only: snapshotted into `_resource` by `__post_init__`, then
    #: never read again. `resource` is declared as an `InitVar` (not a plain
    #: field) specifically so the `resource` *attribute* can be a copy-on-read
    #: property (defined right after this class) instead of a stored
    #: reference to whatever object the caller passed in.
    resource: InitVar[AtomicConfiguration | None] = None
    #: Private, independently-owned snapshot backing the `resource` property.
    _resource: AtomicConfiguration | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self, resource: AtomicConfiguration | None) -> None:
        if not isinstance(self.operator_id, OperatorId):
            raise TypeError("HamiltonianTermDescriptor.operator_id must be an OperatorId")
        if not self.slot or not self.slot.strip():
            raise ValueError("HamiltonianTermDescriptor.slot must be a non-empty string")
        if self.resource_id is not None and not isinstance(self.resource_id, str):
            raise TypeError("HamiltonianTermDescriptor.resource_id must be a string or None")
        if self.resource_id is not None and not self.resource_id.strip():
            raise ValueError("HamiltonianTermDescriptor.resource_id must be non-empty when provided")
        if (self.resource_id is None) != (resource is None):
            raise ValueError(
                "HamiltonianTermDescriptor.resource_id and resource must be provided together"
            )
        if resource is not None:
            if not isinstance(resource, AtomicConfiguration):
                raise TypeError("HamiltonianTermDescriptor.resource must be an AtomicConfiguration")
            # Snapshot now, before validating: `AtomicConfiguration.__post_init__`
            # clones and detaches `positions`/`charges`, so `snapshot` is already
            # independent of `resource` (and of whatever the caller does to its
            # own object afterward). Validating the snapshot's own content id --
            # not the caller's live object's -- is what the owned payload will
            # actually be identified by from here on.
            snapshot = AtomicConfiguration(resource.positions, resource.charges)
            if snapshot.content_id() != self.resource_id:
                raise ValueError(
                    "HamiltonianTermDescriptor.resource_id must equal resource.content_id()"
                )
            object.__setattr__(self, "_resource", snapshot)
        object.__setattr__(self, "parameters", _canonical_parameters(self.parameters))

    def canonical_bytes(self) -> bytes:
        """Return the deterministic byte encoding used by `SystemDescriptor.system_id`.

        Physical identity is canonicalized by `operator_id`; it deliberately
        excludes `slot` (an execution/record-binding label, not physics) and
        `resource` (the actual payload; `resource_id` already determines
        identity and is itself a content hash of that payload).
        """

        digest = hashlib.sha256()
        digest.update(_encode_length_prefixed(self.operator_id.namespace.encode("utf-8")))
        digest.update(_encode_length_prefixed(self.operator_id.name.encode("utf-8")))
        digest.update(struct.pack(">q", self.version))
        items = sorted(self.parameters.items())
        digest.update(struct.pack(">Q", len(items)))
        for key, value in items:
            digest.update(_encode_length_prefixed(key.encode("utf-8")))
            digest.update(_encode_scalar(value))
        if self.resource_id is None:
            digest.update(b"\x00")
        else:
            digest.update(b"\x01")
            digest.update(_encode_length_prefixed(self.resource_id.encode("utf-8")))
        return digest.digest()


def _hamiltonian_term_descriptor_resource(self: HamiltonianTermDescriptor) -> AtomicConfiguration | None:
    """Return a fresh, independently-owned clone of this term's resource, or ``None``."""

    if self._resource is None:
        return None
    return AtomicConfiguration(self._resource.positions, self._resource.charges)


# Attached after `@dataclass` has already built `HamiltonianTermDescriptor`
# (not as a class-body `@property`): a class-body property literally named
# `resource` would shadow the `resource: InitVar[...] = None` default that
# `@dataclass` reads at decoration time, breaking the constructor's default
# value. Assigning the property here, once the class already exists, avoids
# that collision while still making `.resource` a normal attribute to callers.
HamiltonianTermDescriptor.resource = property(_hamiltonian_term_descriptor_resource)


@runtime_checkable
class DescribesPhysicalOperator(Protocol):
    """Protocol for a Hamiltonian term that declares its physical description.

    Notes
    -----
    Deliberately not inferred: a term either implements
    ``describe_physical_operator`` or it does not participate in a
    `SystemDescriptor`. This mirrors
    `tpen.physics.operators.ElectronNucleusGeometryDeclaration` -- an explicit
    typed capability a term supplies on its own class, not a central registry
    a reviewer could forget to update.
    """

    def describe_physical_operator(self) -> HamiltonianTermDescriptor:
        """Return this term's immutable, versioned physical declaration."""
        ...


@dataclass(frozen=True)
class SystemDescriptor:
    """The immutable physical-system identity of one configured Hamiltonian.

    Parameters
    ----------
    particle : ParticleSpec
        Explicit particle/spin/dimension metadata.
    units : str
        Physical unit/convention label (for example ``"atomic_units"``).
    terms : Sequence[HamiltonianTermDescriptor]
        Ordered term declarations, in the order the Hamiltonian was
        configured. Order is preserved here for execution/record binding, but
        `system_id` canonicalizes by `OperatorId` so that construction order
        cannot change physical identity.

    Notes
    -----
    Construction rejects two terms sharing one `OperatorId`: that composition
    is not representable by today's planner's exactly-once manifest, and a
    system description must not advertise an identity the rest of the design
    cannot execute.

    `resources` is derived, not supplied by the caller: for every distinct
    `HamiltonianTermDescriptor.resource_id` declared by `terms`, it holds one
    independently-owned `AtomicConfiguration` snapshot (so later mutation of
    a term's own resource cannot change the published payload). Two terms
    sharing one `resource_id` (e.g. an electron-nucleus and a nucleus-nucleus
    term over the same geometry) resolve to the same single entry. Reading
    `resources` returns a fresh `Mapping` of fresh clones on every access
    (copy-on-read): mutating a tensor obtained from one access can never
    change what a later access, or another term's `resource`, observe.
    """

    particle: ParticleSpec
    units: str
    terms: tuple[HamiltonianTermDescriptor, ...]
    #: Private, independently-owned snapshots backing the `resources` property.
    _resources: Mapping[str, AtomicConfiguration] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.particle, ParticleSpec):
            raise TypeError("SystemDescriptor.particle must be a ParticleSpec")
        if not self.units or not self.units.strip():
            raise ValueError("SystemDescriptor.units must be a non-empty string")
        terms = tuple(self.terms)
        seen: set[OperatorId] = set()
        for term in terms:
            if not isinstance(term, HamiltonianTermDescriptor):
                raise TypeError("SystemDescriptor.terms must contain HamiltonianTermDescriptor values")
            if term.operator_id in seen:
                raise ValueError(
                    f"SystemDescriptor has more than one term declaring operator {term.operator_id}"
                )
            seen.add(term.operator_id)
        object.__setattr__(self, "terms", terms)

        resources: dict[str, AtomicConfiguration] = {}
        for term in terms:
            if term.resource_id is None:
                continue
            # `term.resource` is a copy-on-read property: this is already a
            # fresh, independently-owned clone, decoupled from the term's own
            # `_resource` snapshot and from whatever object the original
            # caller passed in.
            resource = term.resource
            if resource is None:
                raise ValueError(
                    f"term declaring operator {term.operator_id} has a resource_id but no resource payload"
                )
            # Defense in depth: `HamiltonianTermDescriptor.__post_init__`
            # already validated this at its own construction time, but a
            # cached/reused term declaration's resource must still match the
            # id it is being aggregated under here, not just at some earlier
            # moment.
            if resource.content_id() != term.resource_id:
                raise ValueError(
                    f"term declaring operator {term.operator_id} resource payload does not match "
                    f"its declared resource_id {term.resource_id!r} "
                    f"(resource content id {resource.content_id()!r})"
                )
            if resource.spatial_dim != self.particle.spatial_dim:
                raise ValueError(
                    f"resource for operator {term.operator_id} has spatial_dim "
                    f"{resource.spatial_dim}, which disagrees with particle.spatial_dim "
                    f"{self.particle.spatial_dim}"
                )
            existing = resources.get(term.resource_id)
            if existing is None:
                resources[term.resource_id] = resource
            elif existing.content_id() != resource.content_id():
                raise ValueError(
                    f"resource_id {term.resource_id!r} resolves to two different resource payloads"
                )
        object.__setattr__(self, "_resources", resources)

    @property
    def resources(self) -> Mapping[str, AtomicConfiguration]:
        """Return a fresh `Mapping` of fresh resource clones (copy-on-read).

        Returns
        -------
        Mapping[str, AtomicConfiguration]
            One independently-owned `AtomicConfiguration` clone per distinct
            resource id declared by `terms`, freshly cloned on every access
            so a caller mutating a previously returned payload cannot change
            what a later access observes.
        """

        return MappingProxyType(
            {
                resource_id: AtomicConfiguration(resource.positions, resource.charges)
                for resource_id, resource in self._resources.items()
            }
        )

    def system_id(self) -> str:
        """Return the canonical, deterministic content identity of this system.

        Returns
        -------
        str
            Hex SHA-256 digest. Deterministic across processes, independent
            of the order terms were constructed in (terms are canonicalized
            by `OperatorId` before hashing), and depends only on
            `particle`, `units`, and each term's `HamiltonianTermDescriptor`
            (`resources` is not separately hashed: every resource's content
            is already captured by its `resource_id`, which is part of the
            owning term's `canonical_bytes`).
        """

        digest = hashlib.sha256()
        digest.update(f"tpen.physics.system.SystemDescriptor.v{_DESCRIPTOR_SCHEMA_VERSION}".encode("utf-8"))
        digest.update(self.particle.canonical_bytes())
        digest.update(_encode_length_prefixed(self.units.encode("utf-8")))
        canonical_terms = sorted(self.terms, key=lambda term: (term.operator_id.namespace, term.operator_id.name))
        digest.update(struct.pack(">Q", len(canonical_terms)))
        for term in canonical_terms:
            digest.update(term.canonical_bytes())
        return digest.hexdigest()


def build_system_descriptor(
    particle: ParticleSpec,
    terms: Mapping[Any, Any] | Sequence[Any],
    *,
    units: str = "atomic_units",
) -> SystemDescriptor:
    """Build a `SystemDescriptor` from the actual constructed Hamiltonian terms.

    Parameters
    ----------
    particle : ParticleSpec
        Explicit particle/spin/dimension metadata for the system.
    terms : Mapping or Sequence of HamiltonianTerm
        The same term specification accepted by
        `tpen.physics.hamiltonian.normalize_hamiltonian_terms` -- the actual
        terms the generator evaluates, not a separately maintained mirror.
    units : str, optional
        Physical unit/convention label.

    Returns
    -------
    SystemDescriptor
        The canonical aggregate description.

    Raises
    ------
    ValueError
        If any term's class is not a registered operator
        (`tpen.physics.operators.register_operator`), or does not implement
        `DescribesPhysicalOperator`, or declares an `operator_id` mismatched
        with its class registration. The error names the offending term
        before any `SystemDescriptor`, content id, or record is produced.

    Notes
    -----
    Each resulting `HamiltonianTermDescriptor.slot` is the name
    `normalize_hamiltonian_terms` resolved for that term here -- the same
    key execution/record binding uses -- not whatever `describe_physical_operator`
    happened to set. This keeps slot one authoritative value instead of two
    that a reviewer could let drift apart (e.g. a term named ``"trap_energy"``
    in the configured dict must not silently report slot ``"harmonic_trap"``).
    """

    normalized = normalize_hamiltonian_terms(terms)
    descriptors: list[HamiltonianTermDescriptor] = []
    for name, term in normalized.items():
        term_type = type(term)
        if not is_registered_operator(term_type):
            raise ValueError(
                f"hamiltonian term {name!r} ({term_type.__name__}) is not a registered operator; "
                "register it with tpen.physics.operators.register_operator before it can "
                "participate in a SystemDescriptor"
            )
        if not isinstance(term, DescribesPhysicalOperator):
            raise ValueError(
                f"hamiltonian term {name!r} ({term_type.__name__}) does not implement "
                "describe_physical_operator; a SystemDescriptor requires an explicit physical "
                "declaration from every term it describes"
            )
        descriptor = term.describe_physical_operator()
        if not isinstance(descriptor, HamiltonianTermDescriptor):
            raise TypeError(
                f"hamiltonian term {name!r} ({term_type.__name__}) describe_physical_operator "
                "must return a HamiltonianTermDescriptor"
            )
        if descriptor.operator_id != term_type.operator_id:
            raise ValueError(
                f"hamiltonian term {name!r} ({term_type.__name__}) describe_physical_operator "
                f"declared {descriptor.operator_id}, which disagrees with its registered "
                f"operator {term_type.operator_id}"
            )
        if descriptor.slot != name:
            # `dataclasses.replace` cannot be used here: `resource` is an
            # `InitVar` (see `HamiltonianTermDescriptor`), so `replace` would
            # silently fall back to its default (`None`) instead of carrying
            # the original resource forward. Reconstruct explicitly, passing
            # `descriptor.resource` (an already-independent, copy-on-read
            # clone) through.
            descriptor = HamiltonianTermDescriptor(
                operator_id=descriptor.operator_id,
                slot=name,
                parameters=descriptor.parameters,
                resource_id=descriptor.resource_id,
                version=descriptor.version,
                resource=descriptor.resource,
            )
        descriptors.append(descriptor)
    return SystemDescriptor(particle=particle, units=units, terms=tuple(descriptors))


#: Canonical context key for the one physics-owned `SystemDescriptor` artifact.
#: Keyed by object identity, per `tpen.physics.context` conventions: the name
#: below is diagnostic only.
SYSTEM_DESCRIPTOR_KEY: ContextKey[SystemDescriptor] = ContextKey("system_descriptor", SystemDescriptor)


@dataclass(frozen=True)
class SystemDescriptorProvider:
    """Pure provider publishing one fixed, already-built `SystemDescriptor`.

    The descriptor is built once from the actual configured terms (see
    `build_system_descriptor`) and republished unchanged; the provider itself
    performs no computation and declares no dependencies.
    """

    descriptor: SystemDescriptor
    key: ContextKey[SystemDescriptor] = SYSTEM_DESCRIPTOR_KEY
    dependencies: ContextRequirements = field(default_factory=requirements)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, SystemDescriptor):
            raise TypeError("SystemDescriptorProvider.descriptor must be a SystemDescriptor")
        if self.key is not SYSTEM_DESCRIPTOR_KEY:
            raise ValueError("SystemDescriptorProvider must publish SYSTEM_DESCRIPTOR_KEY")

    def build(self, context: ReadOnlyContext) -> SystemDescriptor:
        """Return the fixed descriptor, ignoring the supplied context."""

        del context
        return self.descriptor


__all__ = [
    "DescribesPhysicalOperator",
    "HamiltonianTermDescriptor",
    "JsonScalar",
    "ParticleSpec",
    "SYSTEM_DESCRIPTOR_KEY",
    "SystemDescriptor",
    "SystemDescriptorProvider",
    "build_system_descriptor",
]
