"""Structural interfaces for sampling callers."""

from __future__ import annotations

from typing import Protocol

from tpen.data.batch.walkers import Walkers
from tpen.events import EventEmitter
from tpen.sampling.stats import SamplerStats


class SamplingCollector(Protocol):
    """Sampler interface used by training and evaluation generators."""

    def collect_samples(
        self,
        model,
        *,
        reset: bool = False,
        device=None,
        emitter: EventEmitter | None,
    ) -> tuple[Walkers, SamplerStats]:
        """Collect persistent-chain samples with explicit event delivery."""


__all__ = ["SamplingCollector"]
