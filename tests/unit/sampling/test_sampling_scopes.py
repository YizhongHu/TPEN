"""Acceptance tests for typed sampler scopes and timing attribution."""

from __future__ import annotations

from contextlib import contextmanager
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest
import torch

from tpen.callback.timing.sampling_timing import SamplingScopeTiming
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.evaluation.events import EvaluationTaskRun
from tpen.events import Ended, Occurrence, Started
from tpen.sampling import MetropolisSampler
from tpen.sampling.events import SamplerBurnIn, SamplerProduction
from tpen.sampling.trajectory import (
    collect_observable_trajectory,
    collect_observable_trajectory_with_diagnostics,
)
from tpen.training.events import TrainingIteration, TrainingIterationCompleted
from tests.unit.callback.support import RecordingContext


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

    def emit(self, event, *, state=None):
        del event, state
        return None

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


def test_sampler_failures_pair_and_preserve_exception_and_burn_state(monkeypatch) -> None:
    model = _FailingModel()
    emitter = _RecordingEmitter()
    sampler = _sampler(burn_in=3, n_steps=2)
    model.fail = True
    with pytest.raises(RuntimeError, match="sampler model failure") as burn_error:
        sampler.collect_samples(model, emitter=emitter)
    assert burn_error.value.args == ("sampler model failure",)
    assert sampler.has_burned_in is False
    assert emitter.events[-1] == (SamplerBurnIn, 1, False)

    model.fail = False
    sampler.collect_samples(model, emitter=emitter)
    def fail_production(model_arg, walkers, n_steps):
        if n_steps == sampler.n_steps:
            raise RuntimeError("sampler model failure")
        return original_sample(model_arg, walkers, n_steps)

    original_sample = sampler.sample
    monkeypatch.setattr(sampler, "sample", fail_production)
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


def _collect_samples_c65_reference(
    sampler: MetropolisSampler, model, *, reset: bool = False
) -> tuple[object, object]:
    """Run the frozen c65e35bd collection sequence without scope plumbing.

    This deliberately mirrors the pre-PR public method, so a common-path RNG
    draw can be distinguished from the current off/on comparison (where both
    new paths would share the same accidental draw).
    """

    if reset or sampler._walkers is None:
        sampler.reset()
    else:
        sampler._require_device(None)
        with torch.no_grad():
            sampler._walkers = sampler._walkers.update_cache(model)
    if not sampler._has_burned_in and sampler.burn_in:
        sampler._walkers = sampler.sample(model, sampler._walkers, sampler.burn_in)
        sampler._has_burned_in = True
    sampler._walkers = sampler.sample(model, sampler._walkers, sampler.n_steps)
    return sampler._walkers, sampler.acceptance_rate


def _assert_same_sampler_state(left: MetropolisSampler, right: MetropolisSampler) -> None:
    left_walkers = left._walkers
    right_walkers = right._walkers
    assert left_walkers is not None and right_walkers is not None
    assert torch.equal(left_walkers.positions, right_walkers.positions)
    assert torch.equal(left_walkers.logabs, right_walkers.logabs)
    assert torch.equal(left_walkers.sign, right_walkers.sign)
    assert torch.equal(left_walkers.aux["accepted"], right_walkers.aux["accepted"])
    if "log_accept_ratio" in left_walkers.aux or "log_accept_ratio" in right_walkers.aux:
        assert torch.equal(
            left_walkers.aux["log_accept_ratio"], right_walkers.aux["log_accept_ratio"]
        )
    assert left.acceptance_rate == right.acceptance_rate
    assert torch.equal(
        left.mcmc_state_dict()["generator_state"], right.mcmc_state_dict()["generator_state"]
    )


@pytest.mark.parametrize("emitter_mode", ["off", "recording"])
def test_scopes_match_frozen_uninstrumented_reference_across_reset_and_restore(
    emitter_mode: str,
) -> None:
    model = _LinearModel()
    reference = _sampler(seed=41)
    instrumented = _sampler(seed=41)
    emitter = None if emitter_mode == "off" else _RecordingEmitter()

    for reset in (False, False):
        _collect_samples_c65_reference(reference, model)
        instrumented.collect_samples(model, reset=reset, emitter=emitter)
        _assert_same_sampler_state(reference, instrumented)

    state = reference.mcmc_state_dict()
    restored_reference = _sampler(seed=999)
    restored_reference.load_mcmc_state_dict(state)
    restored_instrumented = _sampler(seed=999)
    restored_instrumented.load_mcmc_state_dict(state)
    _collect_samples_c65_reference(restored_reference, model)
    restored_instrumented.collect_samples(model, emitter=emitter)
    _assert_same_sampler_state(restored_reference, restored_instrumented)

    _collect_samples_c65_reference(reference, model, reset=True)
    instrumented.collect_samples(model, reset=True, emitter=emitter)
    _collect_samples_c65_reference(restored_reference, model, reset=True)
    restored_instrumented.collect_samples(model, reset=True, emitter=emitter)
    _assert_same_sampler_state(reference, instrumented)
    _assert_same_sampler_state(restored_reference, restored_instrumented)


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
    from typeguard import suppress_type_checks

    with suppress_type_checks():
        callback.handle_occurrence_impl(
            Occurrence(event, count, monotonic_time=float(count)), context
        )


def _deliver_public(callback, event, count: int, context: RecordingContext) -> None:
    callback.handle_occurrence(
        Occurrence(event, count, monotonic_time=float(count)), context
    )


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


def test_sampling_scope_timing_uses_occurrence_durations_not_zero_clock() -> None:
    callback = SamplingScopeTiming(clock=lambda: -1.0)
    context = RecordingContext()
    iteration = TrainingIteration(step=7)

    def _deliver_at(event, count, timestamp):
        callback.handle_occurrence(
            Occurrence(event, count, monotonic_time=timestamp), context
        )

    _deliver_at(Started(iteration), 1, 10.0)
    _deliver_at(Started(SamplerBurnIn(n_steps=2)), 1, 11.0)
    _deliver_at(Ended(SamplerBurnIn(n_steps=2)), 1, 14.0)
    _deliver_at(Started(SamplerProduction(n_steps=2)), 1, 15.0)
    _deliver_at(Ended(SamplerProduction(n_steps=2)), 1, 21.0)
    _deliver_at(TrainingIterationCompleted(iteration), 1, 22.0)

    assert context.records[-1]["metrics"] == {
        "sampler_burn_in_time_sec": 3.0,
        "sampler_production_time_sec": 6.0,
    }


def test_sampling_scope_timing_excludes_failed_scope_duration() -> None:
    callback = SamplingScopeTiming(clock=lambda: 0.0)
    context = RecordingContext()
    task = EvaluationTaskRun("failed-sibling", "failed-sibling", Path("/tmp/failed-sibling"))

    def deliver(event, count, timestamp):
        callback.handle_occurrence(
            Occurrence(event, count, monotonic_time=timestamp), context
        )

    deliver(Started(task), 1, 0.0)
    operation = SamplerProduction(n_steps=1)
    deliver(Started(operation), 1, 1.0)
    deliver(Ended(operation, succeeded=False), 1, 9.0)
    deliver(Started(operation), 2, 10.0)
    deliver(Ended(operation, succeeded=True), 2, 13.0)
    deliver(Ended(task), 1, 14.0)

    assert context.records[-1]["metrics"] == {"sampler_production_time_sec": 3.0}


def test_sampling_scope_timing_eval_aggregation_preserves_parent_totals() -> None:
    callback = SamplingScopeTiming(clock=lambda: 0.0)
    context = RecordingContext()
    task = EvaluationTaskRun("multi-draw", "multi-draw", Path("/tmp/multi-draw"))

    def deliver(event, count, timestamp):
        callback.handle_occurrence(
            Occurrence(event, count, monotonic_time=timestamp), context
        )

    deliver(Started(task), 1, 0.0)
    for count, start, end in ((1, 1.0, 2.5), (2, 4.0, 6.0)):
        operation = SamplerProduction(n_steps=2)
        deliver(Started(operation), count, start)
        deliver(Ended(operation), count, end)
    deliver(Ended(task), 1, 7.0)

    assert context.records[-1]["metrics"] == {"sampler_production_time_sec": 3.5}
    # The nested attribution is not a rewrite of the established parent
    # generator/sampling totals; this callback emits only sampler_* keys.
    assert "generator_time_sec" not in context.records[-1]["metrics"]
    assert "sampling_time_sec" not in context.records[-1]["metrics"]


def test_sampling_scope_timing_restores_training_parent_after_nested_evaluation() -> None:
    callback = SamplingScopeTiming(clock=lambda: 0.0)
    context = RecordingContext()
    iteration = TrainingIteration(step=8)
    task_a = EvaluationTaskRun("a", "a", Path("/tmp/a"))

    _deliver_public(callback, Started(iteration), 1, context)
    _deliver_public(callback, Started(task_a), 1, context)
    _deliver_public(callback, Started(SamplerProduction(n_steps=1)), 2, context)
    _deliver_public(callback, Ended(SamplerProduction(n_steps=1)), 2, context)
    _deliver_public(callback, Ended(task_a), 1, context)

    _deliver_public(callback, Started(SamplerProduction(n_steps=1)), 3, context)
    _deliver_public(callback, Ended(SamplerProduction(n_steps=1)), 3, context)
    _deliver_public(callback, TrainingIterationCompleted(iteration), 1, context)

    assert [record["namespace"] for record in context.records] == [
        "eval/perf/a", "train/perf"
    ]


def test_sampling_scope_timing_restores_outer_evaluation_parent() -> None:
    callback = SamplingScopeTiming(clock=lambda: 0.0)
    context = RecordingContext()
    task_a = EvaluationTaskRun("a", "a", Path("/tmp/a"))
    task_b = EvaluationTaskRun("b", "b", Path("/tmp/b"))

    _deliver_public(callback, Started(task_a), 1, context)
    _deliver_public(callback, Started(task_b), 2, context)
    _deliver_public(callback, Started(SamplerProduction(n_steps=1)), 3, context)
    _deliver_public(callback, Ended(SamplerProduction(n_steps=1)), 3, context)
    _deliver_public(callback, Ended(task_b), 2, context)
    _deliver_public(callback, Started(SamplerProduction(n_steps=1)), 4, context)
    _deliver_public(callback, Ended(SamplerProduction(n_steps=1)), 4, context)
    _deliver_public(callback, Ended(task_a), 1, context)

    assert [record["namespace"] for record in context.records] == [
        "eval/perf/b", "eval/perf/a"
    ]


def test_emitter_keyword_is_required_on_sampler_and_trajectory_protocol() -> None:
    sampler = _sampler()
    with pytest.raises(TypeError):
        sampler.collect_samples(_LinearModel())  # type: ignore[call-arg]

    for collector in (
        collect_observable_trajectory,
        collect_observable_trajectory_with_diagnostics,
    ):
        emitter = inspect.signature(collector).parameters["emitter"]
        assert emitter.default is inspect.Parameter.empty
        assert get_type_hints(collector)["sampler"] is SamplingCollector
        with pytest.raises(TypeError):
            collector(  # type: ignore[call-arg]
                sampler,
                _LinearModel(),
                lambda walkers: walkers.positions[:, 0, 0],
                observable_name="position",
                n_draws=1,
            )

