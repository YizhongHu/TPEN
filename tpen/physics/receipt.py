"""Shared local-energy execution receipt.

This module owns the *execution receipt* half of the system/method identity
split described in ``system-context-records-design-r{1,2,3}.md``:
:class:`~tpen.physics.system.SystemDescriptor` declares what physics was
requested; :class:`ExecutionReceipt` declares which implementation actually
produced the returned energy, under what numerical policy, and which
configured contributions own which physical operators. A producing evaluator
returns both together through
:class:`~tpen.physics.hamiltonian.LocalEnergyResult`.

Out of scope here: the recording adapter that attaches a receipt to an
evaluator's result (see ``tpen.physics.hamiltonian``), the planner-backed
execution identity (``LocalEnergyPlan.identity``), and any trajectory-record
serialization (S1-G).
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from tpen.physics.operators import OperatorId

#: JSON-safe scalar accepted as a numerical-policy value. Matches
#: `tpen.physics.system.JsonScalar`'s narrowness for the same reason: a
#: canonical content id must encode every accepted value deterministically.
JsonScalar = bool | int | float | str

_RECEIPT_SCHEMA_VERSION = 1


def _encode_scalar(value: JsonScalar) -> bytes:
    """Return a type-tagged, deterministic byte encoding of one scalar.

    The type tag keeps ``1`` (int) and ``1.0`` (float) from colliding, and
    keeps ``True`` (bool) from colliding with ``1`` (int); ``bool`` is an
    ``int`` subclass in Python, so that check must run first.
    """

    if isinstance(value, bool):
        return b"b" + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        return b"i" + struct.pack(">q", value)
    if isinstance(value, float):
        return b"f" + struct.pack(">d", value)
    if isinstance(value, str):
        return b"s" + value.encode("utf-8")
    raise TypeError(f"numerical policy values must be bool, int, float, or str; got {type(value).__name__}")


def _canonical_policy(policy: Mapping[str, JsonScalar]) -> MappingProxyType[str, JsonScalar]:
    """Validate and freeze a numerical-policy mapping, failing loudly on bad input."""

    validated: dict[str, JsonScalar] = {}
    for key, value in policy.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("numerical policy keys must be non-empty strings")
        if not isinstance(value, (bool, int, float, str)):
            raise TypeError(
                f"numerical policy {key!r} must be bool, int, float, or str; got {type(value).__name__}"
            )
        validated[key] = value
    return MappingProxyType(validated)


def _canonical_manifest(
    manifest: Mapping[OperatorId, tuple[str, ...] | list[str]],
) -> MappingProxyType[OperatorId, tuple[str, ...]]:
    """Validate and freeze a consumption manifest, failing loudly on bad input."""

    if not manifest:
        raise ValueError("ExecutionReceipt.manifest must declare at least one contribution")
    validated: dict[OperatorId, tuple[str, ...]] = {}
    for operator_id, names in manifest.items():
        if not isinstance(operator_id, OperatorId):
            raise TypeError("ExecutionReceipt.manifest keys must be OperatorId")
        names = tuple(names)
        if not names:
            raise ValueError(f"ExecutionReceipt.manifest[{operator_id}] must name at least one contribution")
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"ExecutionReceipt.manifest[{operator_id}] contribution names must be non-empty strings"
                )
        validated[operator_id] = names
    return MappingProxyType(validated)


@dataclass(frozen=True)
class DiagnosticDeclaration:
    """One optional physical diagnostic's availability and derivation.

    A diagnostic is never added to the local-energy total; it exists so a
    receipt can honestly state whether, say, a per-term kinetic value was
    actually (re)computed versus reconstructed from a fused total, without
    inventing a per-operator value a fused kernel never produced.

    Parameters
    ----------
    name : str
        Diagnostic identity, unique within one receipt's ``diagnostics``.
    available : bool
        Whether this diagnostic was actually produced for this evaluation.
    derivation : str
        Short, non-empty statement of how the diagnostic was (or would be)
        obtained, e.g. ``"directly evaluated"`` or ``"unavailable: fused
        kernel does not separate this contribution"``.
    """

    name: str
    available: bool
    derivation: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("DiagnosticDeclaration.name must be a non-empty string")
        if not isinstance(self.available, bool):
            raise TypeError("DiagnosticDeclaration.available must be a bool")
        if not self.derivation or not self.derivation.strip():
            raise ValueError("DiagnosticDeclaration.derivation must be a non-empty string")


def _canonical_diagnostics(
    diagnostics: Mapping[str, DiagnosticDeclaration],
) -> MappingProxyType[str, DiagnosticDeclaration]:
    """Validate and freeze a diagnostics mapping, failing loudly on bad input."""

    validated: dict[str, DiagnosticDeclaration] = {}
    for key, declaration in diagnostics.items():
        if not isinstance(declaration, DiagnosticDeclaration):
            raise TypeError("ExecutionReceipt.diagnostics values must be DiagnosticDeclaration")
        if key != declaration.name:
            raise ValueError(
                f"ExecutionReceipt.diagnostics key {key!r} must equal its declaration name "
                f"{declaration.name!r}"
            )
        validated[key] = declaration
    return MappingProxyType(validated)


@dataclass(frozen=True)
class ExecutionReceipt:
    """The declared identity of the evaluator that actually produced an energy.

    Parameters
    ----------
    system_id : str
        Content identity of the physical system actually evaluated, as
        returned by `tpen.physics.system.SystemDescriptor.system_id`. Opaque
        to this module; compare with `require_matching_system`.
    implementation : str
        Stable, versioned implementation token (for example the evaluator's
        fully-qualified class name). Never a `__dict__`/`repr`/address.
    version : int
        Declaration version for this implementation's receipt shape. Bump
        when the meaning of `numerical_policy` or `manifest` changes for a
        given `implementation`.
    numerical_policy : Mapping[str, JsonScalar]
        Immutable, dtype/device-independent declarations of method-affecting
        numerical behavior (for example term accumulation order). Never the
        runtime dtype/device of any tensor — those vary per call and are not
        part of declared method identity.
    manifest : Mapping[OperatorId, tuple[str, ...]]
        Consumption/contribution manifest: each physical operator this
        evaluation actually covered, mapped to the configured contribution
        name(s) that cover it. A fused contribution's name appears for every
        `OperatorId` it claims.
    diagnostics : Mapping[str, DiagnosticDeclaration], optional
        Optional physical diagnostics, keyed by their own `name`. Never
        folded into the local-energy total.

    Notes
    -----
    Two receipts built from identical declared settings compare equal and
    produce identical `method_id`; any change to `implementation`,
    `version`, `numerical_policy`, or `manifest` changes both.
    """

    system_id: str
    implementation: str
    version: int
    numerical_policy: Mapping[str, JsonScalar]
    manifest: Mapping[OperatorId, tuple[str, ...]]
    diagnostics: Mapping[str, DiagnosticDeclaration] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.system_id or not self.system_id.strip():
            raise ValueError("ExecutionReceipt.system_id must be a non-empty string")
        if not self.implementation or not self.implementation.strip():
            raise ValueError("ExecutionReceipt.implementation must be a non-empty string")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("ExecutionReceipt.version must be a positive integer")
        object.__setattr__(self, "numerical_policy", _canonical_policy(self.numerical_policy))
        object.__setattr__(self, "manifest", _canonical_manifest(self.manifest))
        object.__setattr__(self, "diagnostics", _canonical_diagnostics(self.diagnostics))

    def method_id(self) -> str:
        """Return the canonical, deterministic content identity of the declared method.

        Returns
        -------
        str
            Hex SHA-256 digest. Deterministic across processes; depends only
            on `implementation`, `version`, `numerical_policy`, and
            `manifest` (never `system_id`, which identifies the physics, not
            the method, and never `diagnostics`, which are optional and
            non-method-affecting).
        """

        digest = hashlib.sha256()
        digest.update(f"tpen.physics.receipt.ExecutionReceipt.v{_RECEIPT_SCHEMA_VERSION}".encode("utf-8"))
        digest.update(self.implementation.encode("utf-8"))
        digest.update(struct.pack(">q", self.version))
        for key in sorted(self.numerical_policy):
            digest.update(key.encode("utf-8"))
            digest.update(_encode_scalar(self.numerical_policy[key]))
        for operator_id in sorted(self.manifest, key=str):
            digest.update(str(operator_id).encode("utf-8"))
            for name in self.manifest[operator_id]:
                digest.update(b"\x00")
                digest.update(name.encode("utf-8"))
            digest.update(b"\xff")
        return digest.hexdigest()


def require_matching_system(receipt: ExecutionReceipt, system_id: str) -> ExecutionReceipt:
    """Return ``receipt`` unchanged if it was bound to ``system_id``, else raise.

    Parameters
    ----------
    receipt : ExecutionReceipt
        A receipt produced by some evaluator.
    system_id : str
        The expected physical system identity, typically
        `tpen.physics.system.SystemDescriptor.system_id`'s return value.

    Raises
    ------
    ValueError
        If ``receipt.system_id`` disagrees with ``system_id``.
    """

    if receipt.system_id != system_id:
        raise ValueError(
            f"execution receipt system_id {receipt.system_id!r} does not match the supplied "
            f"system descriptor id {system_id!r}"
        )
    return receipt


__all__ = [
    "DiagnosticDeclaration",
    "ExecutionReceipt",
    "JsonScalar",
    "require_matching_system",
]
