"""Acceptance tests for typed sampler scopes and timing attribution."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from tpen.callback.timing.sampling_timing import SamplingScopeTiming
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.evaluation.events import EvaluationTaskRun
from tpen.events import Ended, Occurrence, Started
from tpen.sampling import MetropolisSampler
from tpen.sampling.events import SamplerBurnIn, SamplerProduction
from tpen.sampling.trajectory import collect_observable_trajectory
from tpen.training.events import TrainingIteration, TrainingIterationCompleted


class _LinearModel(torch.nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = batch.positions.sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _FailingModel(_LinearModel):
    def __init__(self) -> None:
        super().__init__()
        self.fail = False

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        if self.fail:
            raise RuntimeError("sampler model failure")
        return super().forward(batch)


class _RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[type[object], int, bool | None]] = []
        self._counts: dict[type[object], int] = {}

    @contextmanager
    def scope(self, operation, *, state=None):
        del state
        operation_type = type(operation)
        count = self._counts.get(operation_type, 0) + 1
        self._counts[operation_type] = count
        self.events.append((operation_type, count, None))
        try:
            yield None
        except Exception:
            self.events.append((operation_type, count, False))
            raise
        else:
            self.events.append((operation_type, count, True))


def _sampler(*, burn_in: int = 2, n_steps: int = 2, seed: int = 17) -> MetropolisSampler:
    return MetropolisSampler(
        n_walkers=4,
        burn_in=burn_in,
        n_steps=n_steps,
        proposal_scale=0.2,
        seed=seed,
        n_electrons=1,
        spatial_dim=1,
        dtype=torch.float64,
    )


def _event_kinds(emitter: _RecordingEmitter) -> list[type[object]]:
    return [event[0] for event in emitter.events]


def test_sampler_scope_pairing_order_and_burn_in_conditions() -> None:
    model = _LinearModel()
    emitter = _RecordingEmitter()
    sampler = _sampler()

    sampler.collect_samples(model, emitter=emitter)
    sampler.collect_samples(model, emitter=emitter)

    assert _event_kinds(emitter) == [
        SamplerBurnIn, SamplerBurnIn, SamplerProduction, SamplerProduction,
        SamplerProduction, SamplerProduction,
    ]
    assert [(count, succeeded) for _, count, succeeded in emitter.events] == [
        (1, None), (1, True), (1, None), (1, True), (2, None), (2, True)
    ]

    no_burn = _RecordingEmitter()
    _sampler(burn_in=0).collect_samples(model, emitter=no_burn)
    assert _event_kinds(no_burn) == [SamplerProduction, SamplerProduction]


def test_sampler_failures_pair_and_preserve_exception_and_burn_state() -> None:
    model = _FailingModel()
    emitter = _RecordingEmitter()
    sampler = _sampler()
    model.fail = True
    with pytest.raises(RuntimeError, match="sampler model failure") as burn_error:
        sampler.collect_samples(model, emitter=emitter)
    assert burn_error.value.args == ("sampler model failure",)
    assert sampler.has_burned_in is False
    assert emitter.events[-1] == (SamplerBurnIn, 1, False)

    model.fail = False
    sampler.collect_samples(model, emitter=emitter)
    model.fail = True
    with pytest.raises(RuntimeError, match="sampler model failure") as production_error:
        sampler.collect_samples(model, emitter=emitter)
    assert production_error.value.args == ("sampler model failure",)
    assert sampler.has_burned_in is True
    assert emitter.events[-1] == (SamplerProduction, 2, False)


def test_scopes_off_is_bit_exact_and_generator_state_is_unchanged() -> None:
    model_off = _LinearModel()
    model_on = _LinearModel()
    off = _sampler()
    on = _sampler()
    emitter = _RecordingEmitter()
    for _ in range(2):
        walkers_off, stats_off = off.collect_samples(model_off, emitter=None)
        walkers_on, stats_on = on.collect_samples(model_on, emitter=emitter)
        assert torch.equal(walkers_off.positions, walkers_on.positions)
        assert torch.equal(walkers_off.logabs, walkers_on.logabs)
        assert torch.equal(walkers_off.sign, walkers_on.sign)
        assert torch.equal(walkers_off.aux["accepted"], walkers_on.aux["accepted"])
        assert stats_off == stats_on
        assert torch.equal(
            off.mcmc_state_dict()["generator_state"], on.mcmc_state_dict()["generator_state"]
        )


def test_restore_is_production_only_and_reset_reburns() -> None:
    model = _LinearModel()
    source = _sampler()
    source.collect_samples(model, emitter=None)
    state = source.mcmc_state_dict()
    restored = _sampler()
    restored.load_mcmc_state_dict(state)
    emitter = _RecordingEmitter()
    restored.collect_samples(model, emitter=emitter)
    assert _event_kinds(emitter) == [SamplerProduction, SamplerProduction]

    emitter.events.clear()
    restored.collect_samples(model, reset=True, emitter=emitter)
    assert _event_kinds(emitter) == [SamplerBurnIn, SamplerBurnIn, SamplerProduction, SamplerProduction]


class _MetricContext:
    def __init__(self) -> None:
        self.records: list[tuple[dict[str, float], int, str]] = []

    def log(self, metrics, *, step: int, namespace: str) -> None:
        self.records.append((dict(metrics), step, namespace))


def _deliver(callback, event, count: int, context: _MetricContext) -> None:
    callback.handle_occurrence_impl(Occurrence(event, count, monotonic_time=float(count)), context)


def test_sampling_scope_timing_correlates_training_and_evaluation_and_cleans_up() -> None:
    callback = SamplingScopeTiming(clock=lambda: 99.0)
    context = _MetricContext()
    iteration = TrainingIteration(step=3)
    _deliver(callback, Started(iteration), 1, context)
    _deliver(callback, Started(SamplerBurnIn(n_steps=2)), 1, context)
    _deliver(callback, Ended(SamplerBurnIn(n_steps=2)), 1, context)
    _deliver(callback, Started(SamplerProduction(n_steps=2)), 1, context)
    _deliver(callback, Ended(SamplerProduction(n_steps=2)), 1, context)
    _deliver(callback, TrainingIterationCompleted(iteration), 1, context)
    assert context.records[-1][2] == "train/perf"
    assert set(context.records[-1][0]) == {
        "sampler_burn_in_time_sec", "sampler_production_time_sec"
    }

    task = EvaluationTaskRun("sample", "sample", Path("/tmp/sample"))
    _deliver(callback, Started(task), 1, context)
    for count in (1, 2):
        op = SamplerProduction(n_steps=1)
        _deliver(callback, Started(op), count + 1, context)
        _deliver(callback, Ended(op), count + 1, context)
    _deliver(callback, Ended(task), 1, context)
    assert context.records[-1][2] == "eval/perf/sample"
    assert set(context.records[-1][0]) == {"sampler_production_time_sec"}

    _deliver(callback, Started(task), 2, context)
    _deliver(callback, Started(SamplerProduction(n_steps=1)), 4, context)
    _deliver(callback, Ended(task, succeeded=False), 2, context)
    _deliver(callback, Ended(SamplerProduction(n_steps=1), succeeded=True), 4, context)
    assert len(context.records) == 2

    callback._reset_typed_state()
    with pytest.raises(RuntimeError, match="active training or evaluation parent"):
        _deliver(callback, Started(SamplerProduction(n_steps=1)), 5, context)


def test_emitter_keyword_is_required_on_sampler_and_trajectory_protocol() -> None:
    sampler = _sampler()
    with pytest.raises(TypeError):
        sampler.collect_samples(_LinearModel())  # type: ignore[call-arg]

