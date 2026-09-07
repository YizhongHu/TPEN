"""Mergeable finite-energy statistics for the Accelerate spike.

MERGE ALGEBRA TAKEN VERBATIM from ``tests/spikes/native_ddp/statistics.py`` at
DS-N0's tip ``36d5900e10521ee3561dc3153714a0c70a25ed4f``; only the runtime import
differs, so the collectives underneath are Accelerate's while the arithmetic on
top is byte-identical.

That split is deliberate. DG0 is choosing a RUNTIME, so the reduction arithmetic
must be held fixed across the two spikes -- otherwise a difference in results
could be a difference in Chan/Welford implementations rather than in the runtime,
and the comparison would establish nothing. Packets are folded in stable rank
order and rank-local means are never averaged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tests.spikes.accelerate_ddp.runtime import DistributedRuntime


@dataclass(frozen=True)
class FiniteStatistics:
    """Count/mean/M2 packet merged with Chan's parallel formula."""

    total_count: int
    finite_count: int
    mean: float
    m2: float

    @property
    def variance(self) -> float:
        """Return population variance over finite entries."""

        return self.m2 / self.finite_count if self.finite_count else 0.0

    @classmethod
    def from_values(cls, values: torch.Tensor) -> "FiniteStatistics":
        """Summarize one rank's raw tensor without dropping raw-count evidence."""

        finite = values.detach()[torch.isfinite(values.detach())]
        count = int(finite.numel())
        if count == 0:
            return cls(total_count=int(values.numel()), finite_count=0, mean=0.0, m2=0.0)
        mean = float(finite.mean().item())
        m2 = float(((finite - mean) ** 2).sum().item())
        return cls(total_count=int(values.numel()), finite_count=count, mean=mean, m2=m2)

    def merge(self, other: "FiniteStatistics") -> "FiniteStatistics":
        """Merge two packets without averaging rank-local means."""

        if self.finite_count == 0:
            return FiniteStatistics(
                total_count=self.total_count + other.total_count,
                finite_count=other.finite_count,
                mean=other.mean,
                m2=other.m2,
            )
        if other.finite_count == 0:
            return FiniteStatistics(
                total_count=self.total_count + other.total_count,
                finite_count=self.finite_count,
                mean=self.mean,
                m2=self.m2,
            )
        count = self.finite_count + other.finite_count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.finite_count / count
        m2 = self.m2 + other.m2 + delta * delta * self.finite_count * other.finite_count / count
        return FiniteStatistics(
            total_count=self.total_count + other.total_count,
            finite_count=count,
            mean=mean,
            m2=m2,
        )

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-safe packet for cross-rank diagnostics."""

        return {
            "total_count": self.total_count,
            "finite_count": self.finite_count,
            "mean": self.mean,
            "m2": self.m2,
        }


def reduce_statistics(runtime: DistributedRuntime, values: torch.Tensor) -> FiniteStatistics:
    """Gather packets and fold them in stable rank order."""

    packets = runtime.all_gather_objects(FiniteStatistics.from_values(values).as_dict())
    merged = FiniteStatistics(total_count=0, finite_count=0, mean=0.0, m2=0.0)
    for packet in packets:
        merged = merged.merge(
            FiniteStatistics(
                total_count=int(packet["total_count"]),
                finite_count=int(packet["finite_count"]),
                mean=float(packet["mean"]),
                m2=float(packet["m2"]),
            )
        )
    return merged


def centered_terms(
    logabs: torch.Tensor,
    energy: torch.Tensor,
    stats: FiniteStatistics,
) -> torch.Tensor:
    """Return finite-only score terms while retaining an empty-shard graph."""

    mask = torch.isfinite(energy)
    centered = torch.where(mask, energy.detach() - stats.mean, torch.zeros_like(energy))
    return centered * torch.where(mask, logabs, torch.zeros_like(logabs))


def local_centered_objective(
    logabs_shards: Sequence[torch.Tensor], energy_shards: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Intentionally wrong per-shard-centering control for N-G1b."""

    result = torch.zeros((), dtype=torch.float64)
    for logabs, energy in zip(logabs_shards, energy_shards, strict=True):
        mask = torch.isfinite(energy)
        finite_energy = energy[mask]
        if finite_energy.numel() == 0:
            result = result + (logabs * 0.0).sum()
            continue
        local_mean = finite_energy.detach().mean()
        result = result + ((finite_energy.detach() - local_mean) * logabs[mask]).sum()
    return result


__all__ = [
    "FiniteStatistics",
    "centered_terms",
    "local_centered_objective",
    "reduce_statistics",
]
