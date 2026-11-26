"""Typed operations owned by the sampling domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from tpen.events import Operation


@dataclass(frozen=True)
class SamplerBurnIn(Operation):
    """One actual, one-time sampler burn-in transition."""

    metric_fragment: ClassVar[str] = "sampler_burn_in"
    n_steps: int


@dataclass(frozen=True)
class SamplerProduction(Operation):
    """One sampler production transition."""

    metric_fragment: ClassVar[str] = "sampler_production"
    n_steps: int


__all__ = ["SamplerBurnIn", "SamplerProduction"]
