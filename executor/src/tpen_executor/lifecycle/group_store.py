"""Fail-closed creation and reopening of one canonical execution group store.

Allocation/controller incarnation is admitted only through the controller-generation fence.
"""

from __future__ import annotations

import os
import stat
from typing import Any, ClassVar
from uuid import uuid4

from .._codec import canonical_bytes, decode_json, fields, sha256_text, text
from .._records import frozen_record
from ..allocations import ControllerGenerationV1
from ..contracts import ReadyGroupV2
from ..storage import (
    ControllerGenerationPublicationUncertain,
    ControllerGenerationRejected,
    DirectoryStore,
    FenceRejected,
)


class GroupStoreError(ValueError):
    """Base class for typed group-store admission outcomes."""


class GroupStoreRefused(GroupStoreError):
    """The requested create or reopen cannot be admitted without mutation."""


class GroupStoreAuthorityPublished(GroupStoreError):
    """Controller authority may have published; outcome must be reconciled."""


@frozen_record
class CanonicalGroupBinding:
    """The single immutable identity binding written at store creation."""

    schema: ClassVar[str] = "tpen-executor/canonical-group-binding-v1"
    group_identity: str
    execution_bundle_digest: str
    deployment_lineage: str
    controller_lineage: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_identity", sha256_text(self.group_identity, "group_identity"))
        object.__setattr__(self, "execution_bundle_digest", sha256_text(self.execution_bundle_digest, "execution_bundle_digest"))
        object.__setattr__(self, "controller_lineage", sha256_text(self.controller_lineage, "controller_lineage"))
        for name in ("deployment_lineage",):
            object.__setattr__(self, name, text(getattr(self, name), name))

    @property
    def group_digest(self) -> str:
        """Compatibility name for the canonical ReadyGroupV2 identity."""
        return self.group_identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "group_identity": self.group_identity,
            "execution_bundle_digest": self.execution_bundle_digest,
            "deployment_lineage": self.deployment_lineage,
            "controller_lineage": self.controller_lineage,
        }

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "CanonicalGroupBinding":
        fields(data, {"schema", "group_identity", "execution_bundle_digest", "deployment_lineage", "controller_lineage"},
               cls.schema, cls.__name__)
        return cls(data["group_identity"], data["execution_bundle_digest"], data["deployment_lineage"],
                   data["controller_lineage"])

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "CanonicalGroupBinding":
        return cls.from_mapping(decode_json(data))


@frozen_record
class InitialGroupStoreRequest:
    """Typed inputs for exclusive group-store creation."""

    group: ReadyGroupV2
    execution_bundle_digest: str
    deployment_lineage: str
    controller: ControllerGenerationV1
    controller_lineage_anchor: str | None = None


@frozen_record
class ReentryGroupStoreRequest:
    """Typed inputs for a fail-closed store reopen."""

    group: ReadyGroupV2
    execution_bundle_digest: str
    deployment_lineage: str
    controller: ControllerGenerationV1
    controller_lineage_anchor: str | None = None


@frozen_record
class GroupStoreResult:
    """Admitted binding and the fresh invocation-result location."""

    binding: CanonicalGroupBinding
    invocation_result_path: str | None


_BINDING_PATH = "group-binding.json"


def _validate_invocation_parent(path: str) -> None:
    """Validate an existing invocation parent without following symlinks."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(info.st_mode):
        raise GroupStoreRefused("invocation parent is not a real directory")


def _binding(request: InitialGroupStoreRequest | ReentryGroupStoreRequest, *, anchor: str | None = None) -> CanonicalGroupBinding:
    controller = request.controller
    if not isinstance(request.group, ReadyGroupV2) or not isinstance(controller, ControllerGenerationV1):
        raise TypeError("group and controller must be typed records")
    if controller.group_id != request.group.group_id:
        raise GroupStoreRefused("controller group mismatch")
    if anchor is None:
        anchor = request.controller_lineage_anchor or controller.record_digest
    return CanonicalGroupBinding(request.group.identity, request.execution_bundle_digest,
                                 request.deployment_lineage, anchor)


def _acquire_for_reentry(store: DirectoryStore, request: ReentryGroupStoreRequest) -> ControllerGenerationV1:
    """Acquire authority while preserving clean refusals and publication uncertainty."""
    try:
        acquired = store.acquire_controller_generation(request.group, request.controller)
        return acquired
    except ControllerGenerationPublicationUncertain as error:
        raise GroupStoreAuthorityPublished(str(error)) from error
    except ControllerGenerationRejected as error:
        raise GroupStoreRefused(str(error)) from error
    except OSError as error:
        raise GroupStoreAuthorityPublished(str(error)) from error


def initial(request: InitialGroupStoreRequest, root: str | os.PathLike[str]) -> GroupStoreResult:
    """Create one store and publish its binding exactly once."""
    if not isinstance(request, InitialGroupStoreRequest):
        raise TypeError("initial requires InitialGroupStoreRequest")
    if (request.controller_lineage_anchor is not None
            and request.controller_lineage_anchor != request.controller.record_digest):
        raise GroupStoreRefused("initial controller lineage anchor is foreign")
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
        if request.controller.generation != 1:
            raise GroupStoreRefused("initial controller must be generation one")
        store.require_current_controller(request.group, request.controller.generation, request.controller.claimant_nonce)
        store.write_once(_BINDING_PATH, binding.encode())
    except (ControllerGenerationRejected, FenceRejected, OSError, ValueError, TypeError) as error:
        raise GroupStoreRefused(str(error)) from error
    return GroupStoreResult(binding, None)


def reentry(request: ReentryGroupStoreRequest, root: str | os.PathLike[str]) -> GroupStoreResult:
    """Strictly reconstruct and reopen a store without replacing any record.

    Acquisition may have published controller authority before reporting an
    I/O failure; such outcomes are reported as requiring reconciliation.
    """
    if not isinstance(request, ReentryGroupStoreRequest):
        raise TypeError("reentry requires ReentryGroupStoreRequest")
    try:
        store = DirectoryStore(root)
        # The manifest is receipt-derived and rejects malformed, foreign,
        # truncated, or symlinked components before admission proceeds.
        store.manifest(request.group)
        stored = store.read_canonical(_BINDING_PATH, CanonicalGroupBinding.decode)
        generations = store.controller_generations(request.group)
        if not generations:
            raise GroupStoreRefused("group binding crosses controller generation-one ledger")
        if (request.controller_lineage_anchor is not None
                and request.controller_lineage_anchor != generations[0].record_digest):
            raise GroupStoreRefused("controller lineage anchor is foreign")
        expected = _binding(request, anchor=generations[0].record_digest)
        if stored != expected:
            raise GroupStoreRefused("group binding mismatch or non-canonical bytes")
        invocation_parent = os.path.join(store.root, "invocations")
        _validate_invocation_parent(invocation_parent)
        # The immutable binding carries only the stable lineage anchor.  The
        # successor's allocation/controller incarnation is admitted by the
        # existing monotonic generation fence and is never compared bytewise
        # with the creation-time controller.
        acquired = _acquire_for_reentry(store, request)
        try:
            store.require_current_controller(request.group, acquired.generation, acquired.claimant_nonce)
            try:
                os.mkdir(invocation_parent)
            except FileExistsError:
                _validate_invocation_parent(invocation_parent)
            invocation_dir = os.path.join(invocation_parent, uuid4().hex)
            os.mkdir(invocation_dir)
        except (ControllerGenerationRejected, FenceRejected, OSError, ValueError, TypeError) as error:
            raise GroupStoreAuthorityPublished(str(error)) from error
    except GroupStoreAuthorityPublished:
        raise
    except GroupStoreRefused:
        raise
    except (ControllerGenerationRejected, FenceRejected, OSError, ValueError, TypeError) as error:
        raise GroupStoreRefused(str(error)) from error
    return GroupStoreResult(stored, invocation_dir)


__all__ = [
    "CanonicalGroupBinding", "GroupStoreAuthorityPublished", "GroupStoreError", "GroupStoreRefused", "GroupStoreResult",
    "InitialGroupStoreRequest", "ReentryGroupStoreRequest", "initial", "reentry",
]
