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
    "MaterializedParameterScoreRequest",
    "ParameterScoreRequest",
    "ParameterScoreProvider",
    "WavefunctionForwardRequest",
    "WavefunctionRequestProvider",
]
