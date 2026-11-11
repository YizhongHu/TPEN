"""Typed Hydra boundary values for composable interaction producers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tpen.data.paths import PathLayout


class ProducerFamily(str, Enum):
    """Closed producer vocabulary accepted at the Hydra boundary."""

    LINEAR = "linear"
    TENSOR_PRODUCT = "tensor_product"


class InteractionMode(str, Enum):
    """Named Hydra presets for the closed producer sequences."""

    LINEAR = "linear"
    HYBRID = "hybrid"
    TENSOR_PRODUCT = "tensor_product"

    @property
    def producer_order(self) -> tuple[ProducerFamily, ...]:
        """Return the immutable concrete producer order for this preset."""

        if self is InteractionMode.LINEAR:
            return (ProducerFamily.LINEAR,)
        if self is InteractionMode.HYBRID:
            return (ProducerFamily.LINEAR, ProducerFamily.TENSOR_PRODUCT)
        return (ProducerFamily.TENSOR_PRODUCT,)


def normalize_interaction_mode(value: InteractionMode | str) -> InteractionMode:
    """Normalize a closed Hydra mode at the configuration boundary."""

    if isinstance(value, InteractionMode):
        return value
    try:
        return InteractionMode(value)
    except ValueError as error:
        raise ValueError(f"Unsupported interaction mode {value!r}") from error


def normalize_producer_order(
    producers: tuple[ProducerFamily | str, ...] | list[ProducerFamily | str],
) -> tuple[ProducerFamily, ...]:
    """Normalize a configured producer sequence to immutable typed values.

    Parameters
    ----------
    producers : sequence of ProducerFamily or str
        Closed Hydra values. Strings are consumed only at this boundary.

    Returns
    -------
    tuple of ProducerFamily
        The validated producer order.
    """

    normalized = tuple(
        value if isinstance(value, ProducerFamily) else ProducerFamily(value)
        for value in producers
    )
    if not normalized:
        raise ValueError("at least one interaction producer is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("interaction producer families may occur only once")
    expected = tuple(sorted(normalized, key=lambda value: value.value))
    if normalized != expected:
        raise ValueError("producer order must be linear then tensor_product")
    return normalized


@dataclass(frozen=True)
class ResolvedInteractionConfig:
    """Resolved, serializable identity of one interaction layout.

    This value is intended for resolved Hydra snapshots and diagnostics. The
    executable modules receive the immutable :class:`PathLayout` directly.
    """

    basis: Literal["canonical", "full"]
    normalization: Literal["sum", "completion_mean"]
    metadata_identity: str
    fingerprint: str
    producer_order: tuple[ProducerFamily, ...]

    def __post_init__(self) -> None:
        if self.basis not in ("canonical", "full"):
            raise ValueError(f"unsupported basis {self.basis!r}")
        if self.normalization not in ("sum", "completion_mean"):
            raise ValueError(f"unsupported normalization {self.normalization!r}")
        if not self.metadata_identity:
            raise ValueError("metadata_identity must be non-empty")
        if len(self.fingerprint) != 64:
            raise ValueError("fingerprint must be a SHA-256 hex digest")
        object.__setattr__(self, "producer_order", normalize_producer_order(self.producer_order))

    @classmethod
    def from_layout(
        cls,
        layout: PathLayout,
        *,
        basis: Literal["canonical", "full"] = "canonical",
        normalization: Literal["sum", "completion_mean"] = "completion_mean",
        metadata_identity: str = "checked-in-path-metadata-v1",
    ) -> "ResolvedInteractionConfig":
        """Resolve diagnostic identity directly from a static layout."""

        order = tuple(ProducerFamily(slice_.family) for slice_ in layout.family_slices)
        return cls(basis, normalization, metadata_identity, layout.fingerprint, order)

    def as_tuple(self) -> tuple[object, ...]:
        """Return a deterministic value suitable for a resolved snapshot."""

        return (
            self.basis,
            self.normalization,
            self.metadata_identity,
            self.fingerprint,
            tuple(value.value for value in self.producer_order),
        )

    def as_dict(self) -> dict[str, object]:
        """Return the boundary representation for a resolved Hydra snapshot."""

        return {
            "basis": self.basis,
            "normalization": self.normalization,
            "metadata_identity": self.metadata_identity,
            "fingerprint": self.fingerprint,
            "producer_order": tuple(value.value for value in self.producer_order),
        }


__all__ = [
    "InteractionMode",
    "ProducerFamily",
    "ResolvedInteractionConfig",
    "normalize_interaction_mode",
    "normalize_producer_order",
]
