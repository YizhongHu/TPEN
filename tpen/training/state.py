"""Mutable training-loop state shared with callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from tpen.data.batch import WavefunctionOutput
from tpen.events import TrainingTiming, TrainingTimingState
from tpen.sampling.stats import SamplerStats
from tpen.training.update import VMCUpdateState


@dataclass
class TrainerState(TrainingTimingState):
    """Snapshot of the VMC training loop at one step.

    This is the training domain's `DomainState`: the trainer passes it beside
    every typed occurrence it emits, and a `tpen.callback.StatefulCallback`
    declaring ``state_type = TrainerState`` receives it as a typed handler
    argument.

    The state is updated in place each step and handed to callbacks (notably
    `tpen.callback.Checkpoint`) through ``Event.state``. Fields beyond
    ``step``/``metrics`` carry the most recent loop artifacts for inspection.

    Parameters
    ----------
    step : int, optional
        Index of the most recently completed step (0-based; ``-1`` before the
        first step).
    metrics : dict, optional
        Scalar metrics logged for the most recent step.
    optimizer_step : bool, optional
        Whether the most recently completed iteration applied an optimizer
        update. ``False`` before the first step, and on any iteration that
        deliberately skipped its update (the zero-electron vacuum). This is the
        in-process typed carrier of the same fact the ``train/optimizer_step``
        metric publishes durably, and `tpen.metrics_naming` documents it as the
        authoritative discriminator between "no update happened" and "an update
        happened". A health callback that observes update by-products needs it
        to tell an empty observation apart from a legitimately empty iteration.
    model : Any, optional
        Wavefunction model being optimized.
    optimizer : Any, optional
        Compatibility reference to the optimizer in ``update_state``.
    update_state : VMCUpdateState or None, optional
        The single typed update-state authority used by the trainer.  The
        compatibility ``optimizer`` field is always the same object when this
        field is populated.
    trainer : Any, optional
        Trainer object owning train-loop progress state.
    sampler : Any, optional
        Sampler producing walker configurations.
    samples : Any, optional
        Most recent walker state, when retained.
    batch : Any, optional
        Most recent electron batch.
    local_energy : Any, optional
        Most recent per-sample local energy (detached).
    local_energy_terms : mapping of str to Any or None, optional
        Per-term local energies from the same evaluation as ``local_energy``.
        ``None`` means the trainer was not configured to return terms.
    loss : torch.Tensor or None, optional
        Most recent surrogate loss (detached).
    wavefunction_output : WavefunctionOutput or None, optional
        Most recent wavefunction output (signed-log form) for the batch.
    sampler_stats : SamplerStats or None, optional
        Most recent typed sampler diagnostics, or ``None`` before the first
        collection.
    timing : TrainingTiming or None, optional
        Most recent whole-iteration timing, or ``None`` before the first
        completed iteration.
    """

    step: int = -1
    metrics: dict[str, Any] = field(default_factory=dict)
    # A typed field rather than a `metrics["optimizer_step"]` lookup: reading a
    # published metric back by string key to decide behaviour would make a
    # durable metric name the mechanism one part of the program uses to find
    # another, which ADR-E006 bars. The metric stays the durable spelling; this
    # is the in-process one, and they carry the same name so neither can drift
    # into meaning something else.
    optimizer_step: bool = False
    model: Any = None
    optimizer: Any = None
    update_state: VMCUpdateState | None = None
    trainer: Any = None
    sampler: Any = None
    samples: Any = None
    batch: Any = None
    local_energy: Any = None
    local_energy_terms: dict[str, Any] | None = None
    loss: torch.Tensor | None = None
    wavefunction_output: WavefunctionOutput | None = None
    sampler_stats: SamplerStats | None = None
    timing: TrainingTiming | None = None


__all__ = ["TrainerState", "TrainingTiming"]
