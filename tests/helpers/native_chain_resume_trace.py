"""Test-side spies and selective restore hooks for the native E5 fixture.

This module is deliberately outside :mod:`tpen.execution`.  It wraps the
existing model, sampler, optimizer, and configured-run checkpoint boundary;
it does not add production callbacks or change TPEN core code.  The one
synthetic process-RNG draw is an observation-only probe: the native fixture
does not otherwise consume the global CPU stream during its training loop.
"""

from __future__ import annotations

import json
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from tpen.execution.chain_resume import ChannelEvent


class NativeTrace:
    """Collect first random-dependent events from one real worker process."""

    def __init__(self, *, phase: str, seed: int, resume_step: int) -> None:
        self.phase = phase
        self.pid = os.getpid()
        self.seed = seed
        self.seed_source = f"{phase}:runtime.seed={seed}"
        self.resume_step = resume_step
        self.current_step = resume_step
        self._step_count = 0
        self.trainer: Any | None = None
        self._seen: set[tuple[str, int]] = set()
        self.events: list[ChannelEvent] = []

    def begin_step(self) -> None:
        """Record the first process-RNG draw and cursor event for a step."""

        actual_step = None if self.trainer is None else getattr(self.trainer, "next_iteration", None)
        self.current_step = (
            int(actual_step)
            if actual_step is not None
            else self.resume_step + self._step_count
        )
        self._step_count += 1
        self._record(
            "trainer_cursor",
            "first_resumed_step",
            self.current_step,
            random_dependent=True,
        )
        # The fixture's sampler owns proposal randomness.  This observation
        # consumes only the global CPU stream, which is checkpointed by the
        # existing train_resume payload and is not used by the science model.
        process_draw = float(torch.rand((), device="cpu").item())
        self._record("process_random", "first_process_random", process_draw)

    def _record(
        self,
        channel: str,
        event_kind: str,
        value: Any,
        *,
        random_dependent: bool = True,
    ) -> None:
        key = (channel, self.current_step)
        if key in self._seen:
            return
        self._seen.add(key)
        if hasattr(value, "item"):
            value = value.item()
        self.events.append(
            ChannelEvent(
                channel=channel,
                step=int(self.current_step),
                event_kind=event_kind,
                value=float(value) if isinstance(value, (int, float)) else value,
                random_dependent=random_dependent,
            )
        )

    def write(self, path: Path) -> None:
        """Write process, seed, and event provenance as JSON."""

        payload = {
            "phase": self.phase,
            "pid": self.pid,
            "seed": self.seed,
            "seed_source": self.seed_source,
            "events": [
                {
                    "record": "channel_event",
                    "channel": event.channel,
                    "step": event.step,
                    "event_kind": event.event_kind,
                    "value": event.value,
                    "random_dependent": event.random_dependent,
                }
                for event in self.events
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def install_native_trace(trace: NativeTrace) -> ExitStack:
    """Install test-only spies on the real native fixture classes.

    The returned stack must be kept alive for the complete configured run.
    Instrumentation points correspond to the public CR declaration:
    ``MetropolisSampler.collect_samples`` (proposal), the fixture model's
    ``forward`` (model evaluation), ``torch.optim.Adam.step`` (optimizer), the
    observation-only global ``torch.rand`` draw above (process RNG), and the
    sampler-call boundary (trainer cursor).
    """

    from tests.helpers.trainable_wavefunctions import TrainableHookeSingletAnsatz
    from tpen.sampling.metropolis import MetropolisSampler
    from tpen.training.trainer import VMCTrainer

    stack = ExitStack()
    original_collect = MetropolisSampler.collect_samples
    original_forward = TrainableHookeSingletAnsatz.forward
    original_step = torch.optim.Adam.step
    original_fit = VMCTrainer.fit

    def collect_samples(self, model, *args, **kwargs):
        trace.begin_step()
        result = original_collect(self, model, *args, **kwargs)
        walkers, _stats = result
        trace._record(
            "sampler_proposal",
            "first_sampler_proposal",
            float(walkers.positions.detach().mean().item()),
        )
        return result

    def forward(self, batch):
        output = original_forward(self, batch)
        trace._record(
            "model_logabs",
            "first_model_evaluation",
            float(output.logabs.detach().mean().item()),
        )
        return output

    def optimizer_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        norm = 0.0
        parameters = [parameter.detach() for group in self.param_groups for parameter in group["params"]]
        if parameters:
            norm = float(torch.cat([parameter.reshape(-1) for parameter in parameters]).norm().item())
        trace._record("optimizer_update", "first_optimizer_update", norm)
        return result

    def fit(self, *args, **kwargs):
        trace.trainer = self
        return original_fit(self, *args, **kwargs)

    MetropolisSampler.collect_samples = collect_samples
    TrainableHookeSingletAnsatz.forward = forward
    torch.optim.Adam.step = optimizer_step
    VMCTrainer.fit = fit

    def undo() -> None:
        MetropolisSampler.collect_samples = original_collect
        TrainableHookeSingletAnsatz.forward = original_forward
        torch.optim.Adam.step = original_step
        VMCTrainer.fit = original_fit

    stack.callback(undo)
    return stack


@contextmanager
def skip_restore_owner(owner: str | None) -> Iterator[None]:
    """Skip exactly one component loader in a test-only resume process.

    The production ``train_resume`` gate and validation still run.  Only the
    selected mutating loader is replaced, allowing the CR to observe a real
    restore omission without weakening the production restore contract.
    """

    if owner is None:
        yield
        return
    allowed = {"model", "optimizer", "trainer_cursor", "sampler_mcmc", "process_rng"}
    if owner not in allowed:
        raise ValueError(f"unknown native restore owner {owner!r}")
    import tpen.checkpoint.restore as restore

    target = {
        "model": "_load_model",
        "optimizer": "_load_optimizer",
        "trainer_cursor": "_load_trainer",
        "sampler_mcmc": "_load_sampler",
    }.get(owner)
    original = getattr(restore, target) if target else restore.apply_rng_state

    def skipped(*_args, **_kwargs):
        return None

    if target:
        setattr(restore, target, skipped)
    else:
        restore.apply_rng_state = skipped
    try:
        yield
    finally:
        if target:
            setattr(restore, target, original)
        else:
            restore.apply_rng_state = original


__all__ = ["NativeTrace", "install_native_trace", "skip_restore_owner"]
