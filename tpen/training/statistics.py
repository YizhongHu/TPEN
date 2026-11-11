"""Typed rank-local statistics reduction contracts for VMC updates.

The reducer consumes already-aggregated local sums.  It deliberately does not
receive materialized per-sample scores: those remain rank-local and
uncentered until the score consumer explicitly chooses to center them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tpen.dependencies import require_torch

torch = require_torch(feature="VMC statistics reducers")


@dataclass(frozen=True, kw_only=True)
class StatisticsSums:
    """Local count and tensor sums passed across the reducer seam.

    Parameters
    ----------
    count : int
        Number of local samples represented by the sums.
    sums : tuple[torch.Tensor, ...]
        Ordered floating-point sums.  A consumer gives each position a
        meaning, such as a score sum, an energy-gradient sum, or a QGT
        product sum.  Individual tensors may have different shapes, but all
        must be real floating-point values.

    Notes
    -----
    This record contains no raw score rows and no centered values.  Keeping
    those concepts out of the reducer input makes ownership explicit: the
    forward path owns raw rank-local scores, the consumer owns centering, and
    the reducer owns combining counts and sums.
    """

    count: int
    sums: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sums", tuple(self.sums))
        self.validate()

    def validate(self) -> "StatisticsSums":
        """Validate the local aggregate contract."""

        if type(self.count) is not int or self.count < 0:
            raise ValueError("StatisticsSums.count must be a non-negative integer")
        if any(not isinstance(value, torch.Tensor) for value in self.sums):
            raise TypeError("StatisticsSums.sums must contain only torch.Tensor values")
        if any(not value.is_floating_point() for value in self.sums):
            raise TypeError("StatisticsSums.sums must contain real floating-point tensors")
        return self


class StatisticsReducer(ABC):
    """Typed contract for reducing local counts and aggregate tensor sums.

    A future distributed implementation can replace the two primitive
    operations while leaving score construction and centering downstream.  A
    reducer never centers raw scores itself.
    """

    def reduce(self, local: StatisticsSums) -> StatisticsSums:
        """Reduce one local aggregate into the current statistics domain."""

        if not isinstance(local, StatisticsSums):
            raise TypeError("StatisticsReducer.reduce requires StatisticsSums")
        local.validate()
        reduced = StatisticsSums(
            count=self.reduce_count(local.count),
            sums=tuple(self.reduce_sum(value) for value in local.sums),
        )
        return reduced

    @abstractmethod
    def reduce_count(self, count: int) -> int:
        """Reduce a local sample count."""

    @abstractmethod
    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        """Reduce one already-aggregated local tensor sum."""


class IdentityStatisticsReducer(StatisticsReducer):
    """Single-process reducer preserving counts and sums exactly."""

    def reduce_count(self, count: int) -> int:
        """Return the local count unchanged."""

        if type(count) is not int or count < 0:
            raise ValueError("IdentityStatisticsReducer count must be a non-negative integer")
        return count

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        """Return the local tensor sum unchanged, without centering it."""

        if not isinstance(value, torch.Tensor):
            raise TypeError("IdentityStatisticsReducer sums must be torch.Tensor values")
        if not value.is_floating_point():
            raise TypeError("IdentityStatisticsReducer sums must be real floating-point tensors")
        return value


def center_statistics(
    values: torch.Tensor,
    *,
    count: int,
    total: torch.Tensor,
) -> torch.Tensor:
    """Center values using an explicitly reduced total and count.

    Centering belongs to the score/QGT consumer rather than to a reducer.  In
    particular, passing raw score rows here leaves the rows rank-local and
    uncentered until this explicit consumer-owned operation is requested.
    """

    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise TypeError("center_statistics.values must be a real floating-point tensor")
    if type(count) is not int or count <= 0:
        raise ValueError("center_statistics.count must be a positive integer")
    if not isinstance(total, torch.Tensor) or not total.is_floating_point():
        raise TypeError("center_statistics.total must be a real floating-point tensor")
    if values.device != total.device:
        raise ValueError("center_statistics values and total must share one device")
    if values.dtype != total.dtype:
        raise ValueError("center_statistics values and total must share one dtype")
    return values - total / count


__all__ = [
    "IdentityStatisticsReducer",
    "StatisticsReducer",
    "StatisticsSums",
    "center_statistics",
]
