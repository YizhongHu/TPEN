"""Immutable polymorphic wavefunction forward requests.

Requests use direct double dispatch: a model delegates to
``request.evaluate(model, batch)``, and the exact request calls one typed
provider method.  There is no registry, string lookup, or capability probing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from tpen.data.batch import (
    CoordinateForwardPacket,
    ElectronBatch,
    ParameterScoreForwardPacket,
    WavefunctionPacket,
)
from tpen.dependencies import require_torch
from tpen.nn.structured_curvature import (
    DenseFactorBlock,
    KroneckerFactorBlock,
    StructuredCurvatureForwardPacket,
)

torch = require_torch(feature="wavefunction forward requests")


PacketT = TypeVar("PacketT", bound=WavefunctionPacket)


@dataclass(frozen=True, kw_only=True)
class WavefunctionForwardRequest(Generic[PacketT], ABC):
    """Nominal strategy that evaluates one exact wavefunction packet type."""

    @abstractmethod
    def evaluate(
        self,
        provider: "WavefunctionRequestProvider",
        batch: ElectronBatch,
    ) -> PacketT:
        """Evaluate this request through direct typed provider dispatch."""


@dataclass(frozen=True, kw_only=True)
class CoordinateGradientRequest(WavefunctionForwardRequest[CoordinateForwardPacket]):
    """Request values plus the exact real-logabs coordinate gradient."""

    def evaluate(
        self,
        provider: "CoordinateGradientProvider",
        batch: ElectronBatch,
    ) -> CoordinateForwardPacket:
        """Delegate to the provider's coordinate-gradient implementation."""

        return provider.evaluate_coordinate_gradient_request(request=self, batch=batch)


@dataclass(frozen=True, kw_only=True)
class ParameterScoreRequest(WavefunctionForwardRequest[ParameterScoreForwardPacket], ABC):
    """Nominal base for exact real-logabs parameter-score strategies."""


@dataclass(frozen=True, kw_only=True)
class MaterializedParameterScoreRequest(ParameterScoreRequest):
    """Request raw parameter-shaped score blocks.

    Parameters
    ----------
    chunk_size : int or None, optional
        Positive sample chunk size for a batched implementation. ``None``
        leaves the execution choice to the provider.
    """

    chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.chunk_size is not None and (
            not isinstance(self.chunk_size, int)
            or isinstance(self.chunk_size, bool)
            or self.chunk_size < 1
        ):
            raise ValueError("MaterializedParameterScoreRequest.chunk_size must be positive or None")

    def evaluate(
        self,
        provider: "ParameterScoreProvider",
        batch: ElectronBatch,
    ) -> ParameterScoreForwardPacket:
        """Delegate to the provider's materialized-score implementation."""

        return provider.evaluate_materialized_parameter_score_request(request=self, batch=batch)


@dataclass(frozen=True, kw_only=True)
class StructuredCurvatureForwardRequest(
    WavefunctionForwardRequest[StructuredCurvatureForwardPacket]
):
    """Request physics-free structured KFAC curvature factors and half-gradient.

    Parameters
    ----------
    sample_weights : torch.Tensor
        Detached 1D per-sample weights of shape ``(batch.batch_size,)`` used to
        compute the exact weighted energy half-gradient in one retained-graph
        backward pass.
    solve_dtype : torch.dtype, optional
        Real floating dtype (``torch.float32`` or ``torch.float64``) for
        captured curvature factors and energy half-gradient blocks.
    max_dense_block_numel : int, optional
        Maximum allowed scalar element count for any :class:`DenseFactorBlock`.
    """

    sample_weights: torch.Tensor
    solve_dtype: torch.dtype = torch.float64
    max_dense_block_numel: int = 2048

    def __post_init__(self) -> None:
        if not isinstance(self.sample_weights, torch.Tensor):
            raise TypeError(
                "StructuredCurvatureForwardRequest.sample_weights must be a torch.Tensor"
            )
        if self.sample_weights.requires_grad:
            raise ValueError(
                "StructuredCurvatureForwardRequest.sample_weights must be detached"
            )
        if not self.sample_weights.is_floating_point() or self.sample_weights.is_complex():
            raise TypeError(
                "StructuredCurvatureForwardRequest.sample_weights must have a real floating dtype"
            )
        if self.sample_weights.ndim != 1:
            raise ValueError(
                "StructuredCurvatureForwardRequest.sample_weights must be 1D, "
                f"got shape {tuple(self.sample_weights.shape)}"
            )
        if not isinstance(self.solve_dtype, torch.dtype) or self.solve_dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError(
                "StructuredCurvatureForwardRequest.solve_dtype must be torch.float32 or torch.float64"
            )
        if (
            isinstance(self.max_dense_block_numel, bool)
            or not isinstance(self.max_dense_block_numel, int)
            or self.max_dense_block_numel < 1
        ):
            raise ValueError(
                "StructuredCurvatureForwardRequest.max_dense_block_numel must be a positive integer"
            )

    def evaluate(
        self,
        provider: "StructuredCurvatureProvider",
        batch: ElectronBatch,
    ) -> StructuredCurvatureForwardPacket:
        """Delegate to the provider's structured-curvature implementation."""

        return provider.evaluate_structured_curvature_request(request=self, batch=batch)


@runtime_checkable
class StructuredCurvatureProvider(Protocol):
    """Typed provider targeted by :class:`StructuredCurvatureForwardRequest`."""

    def evaluate_structured_curvature_request(
        self,
        *,
        request: StructuredCurvatureForwardRequest,
        batch: ElectronBatch,
    ) -> StructuredCurvatureForwardPacket:
        """Evaluate a structured-curvature forward request."""

        ...


@runtime_checkable
class CoordinateGradientProvider(Protocol):
    """Typed provider targeted by :class:`CoordinateGradientRequest`."""

    def evaluate_coordinate_gradient_request(
        self,
        *,
        request: CoordinateGradientRequest,
        batch: ElectronBatch,
    ) -> CoordinateForwardPacket:
        """Evaluate a coordinate-gradient request."""

        ...


@runtime_checkable
class ParameterScoreProvider(Protocol):
    """Typed provider targeted by materialized parameter-score requests."""

    def evaluate_materialized_parameter_score_request(
        self,
        *,
        request: MaterializedParameterScoreRequest,
        batch: ElectronBatch,
    ) -> ParameterScoreForwardPacket:
        """Evaluate a materialized parameter-score request."""

        ...


@runtime_checkable
class WavefunctionRequestProvider(CoordinateGradientProvider, ParameterScoreProvider, Protocol):
    """Rich provider combining the independent F2 and F3 request seams."""


__all__ = [
    "CoordinateGradientRequest",
    "CoordinateGradientProvider",
    "DenseFactorBlock",
    "KroneckerFactorBlock",
    "MaterializedParameterScoreRequest",
    "ParameterScoreRequest",
    "ParameterScoreProvider",
    "StructuredCurvatureForwardPacket",
    "StructuredCurvatureForwardRequest",
    "StructuredCurvatureProvider",
    "WavefunctionForwardRequest",
    "WavefunctionRequestProvider",
]
