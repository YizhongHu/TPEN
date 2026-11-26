"""Fail-closed creation and reopening of one canonical execution group store."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from .._codec import canonical_bytes, decode_json, digest, fields, sha256_text, text
from .._records import frozen_record
from ..allocations import ControllerGenerationV1
from ..contracts import ReadyGroupV2
from ..storage import ControllerGenerationRejected, DirectoryStore, FenceRejected


class GroupStoreRefused(ValueError):
    """The requested create or reopen cannot be admitted without mutation."""


@frozen_record
class CanonicalGroupBinding:
    """The single immutable identity binding written at store creation."""

    schema: ClassVar[str] = "tpen-executor/canonical-group-binding-v1"
    group_identity: str
    execution_bundle_digest: str
    allocation_lineage: str
    deployment_lineage: str
    controller_lineage: str
    controller_generation: int
    controller_nonce: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_identity", sha256_text(self.group_identity, "group_identity"))
        object.__setattr__(self, "execution_bundle_digest", sha256_text(self.execution_bundle_digest, "execution_bundle_digest"))
        object.__setattr__(self, "controller_lineage", sha256_text(self.controller_lineage, "controller_lineage"))
        for name in ("allocation_lineage", "deployment_lineage", "controller_nonce"):
            object.__setattr__(self, name, text(getattr(self, name), name))
        if type(self.controller_generation) is not int or self.controller_generation < 1:
            raise TypeError("controller_generation must be a positive integer")

    @property
    def group_digest(self) -> str:
        """Compatibility name for the canonical ReadyGroupV2 identity."""
        return self.group_identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "group_identity": self.group_identity,
            "execution_bundle_digest": self.execution_bundle_digest,
            "allocation_lineage": self.allocation_lineage,
            "deployment_lineage": self.deployment_lineage,
            "controller_lineage": self.controller_lineage,
            "controller_generation": self.controller_generation,
            "controller_nonce": self.controller_nonce,
        }

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "CanonicalGroupBinding":
        fields(data, {"schema", "group_identity", "execution_bundle_digest", "allocation_lineage",
                      "deployment_lineage", "controller_lineage", "controller_generation", "controller_nonce"},
               cls.schema, cls.__name__)
        return cls(data["group_identity"], data["execution_bundle_digest"], data["allocation_lineage"],
                   data["deployment_lineage"], data["controller_lineage"], data["controller_generation"], data["controller_nonce"])

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "CanonicalGroupBinding":
        return cls.from_mapping(decode_json(data))


@frozen_record
class InitialGroupStoreRequest:
    """Typed inputs for exclusive group-store creation."""

    group: ReadyGroupV2
    execution_bundle_digest: str
    allocation_lineage: str
    deployment_lineage: str
    controller: ControllerGenerationV1


@frozen_record
class ReentryGroupStoreRequest:
    """Typed inputs for a fail-closed store reopen."""

    group: ReadyGroupV2
    execution_bundle_digest: str
    allocation_lineage: str
    deployment_lineage: str
    controller: ControllerGenerationV1


@frozen_record
class GroupStoreResult:
    """Admitted binding and the fresh invocation-result location."""

    binding: CanonicalGroupBinding
    invocation_result_path: str | None


_BINDING_PATH = "group-binding.json"


def _binding(request: InitialGroupStoreRequest | ReentryGroupStoreRequest) -> CanonicalGroupBinding:
    controller = request.controller
    if not isinstance(request.group, ReadyGroupV2) or not isinstance(controller, ControllerGenerationV1):
        raise TypeError("group and controller must be typed records")
    if controller.group_id != request.group.group_id:
        raise GroupStoreRefused("controller group mismatch")
    return CanonicalGroupBinding(
        request.group.identity, request.execution_bundle_digest, request.allocation_lineage,
        request.deployment_lineage, digest(controller.to_dict()), controller.generation, controller.claimant_nonce,
    )


def initial(request: InitialGroupStoreRequest, root: str | os.PathLike[str]) -> GroupStoreResult:
    """Create one store and publish its binding exactly once."""
    if not isinstance(request, InitialGroupStoreRequest):
        raise TypeError("initial requires InitialGroupStoreRequest")
    binding = _binding(request)
    root_path = os.path.abspath(os.path.normpath(os.fspath(root)))
    try:
        os.mkdir(root_path)
    except FileExistsError as error:
        raise GroupStoreRefused("store root already exists") from error
    except OSError as error:
        raise GroupStoreRefused("store root could not be created") from error
    store = DirectoryStore(root_path)
    try:
        store.acquire_controller_generation(request.group, request.controller)
        store.require_current_controller(request.group, binding.controller_generation, binding.controller_nonce)
        store.write_once(_BINDING_PATH, binding.encode())
    except (ControllerGenerationRejected, FenceRejected, OSError, ValueError, TypeError) as error:
        raise GroupStoreRefused(str(error)) from error
    return GroupStoreResult(binding, None)


def reentry(request: ReentryGroupStoreRequest, root: str | os.PathLike[str]) -> GroupStoreResult:
    """Strictly reconstruct and reopen a store without replacing any record."""
    if not isinstance(request, ReentryGroupStoreRequest):
        raise TypeError("reentry requires ReentryGroupStoreRequest")
    expected = _binding(request)
    try:
        store = DirectoryStore(root)
        # The manifest is receipt-derived and rejects malformed, foreign,
        # truncated, or symlinked components before admission proceeds.
        store.manifest(request.group)
        stored = store.read_canonical(_BINDING_PATH, CanonicalGroupBinding.decode)
        if stored != expected or stored.encode() != Path(os.path.join(store.root, _BINDING_PATH)).read_bytes():
            raise GroupStoreRefused("group binding mismatch or non-canonical bytes")
        store.require_current_controller(request.group, expected.controller_generation, expected.controller_nonce)
        invocation_dir = os.path.join(store.root, "invocations", uuid4().hex)
        os.makedirs(os.path.dirname(invocation_dir), exist_ok=True)
        os.mkdir(invocation_dir)
    except GroupStoreRefused:
        raise
    except (ControllerGenerationRejected, FenceRejected, OSError, ValueError, TypeError) as error:
        raise GroupStoreRefused(str(error)) from error
    return GroupStoreResult(stored, invocation_dir)


__all__ = [
    "CanonicalGroupBinding", "GroupStoreRefused", "GroupStoreResult",
    "InitialGroupStoreRequest", "ReentryGroupStoreRequest", "initial", "reentry",
]
