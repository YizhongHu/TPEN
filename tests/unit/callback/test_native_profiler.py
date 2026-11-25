"""Contract tests for the opt-in native profiler callback."""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys

import pytest

from tests.helpers.run_context import make_run_context
from tpen.callback import (
    ChromeTraceExport,
    NativeProfiler,
    TorchOperatorSummary,
    TrainingIterationWindow,
    claims_conflict,
)
from tpen.events import Ended, Occurrence, Started
from tpen.sampling.events import SamplerBurnIn, SamplerProduction
from tpen.training.events import CollectSamples, TrainingIteration


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")

    def step(self) -> None:
        self.calls.append("step")

    def stop(self) -> None:
        self.calls.append("stop")

    def key_averages(self) -> list[object]:
        return [type("Row", (), {"key": f"op-{i}"})() for i in range(20)]

    def record_function(self, label: str):
        self.calls.append(f"start:{label}")

        class _Range:
            def __enter__(_self):
                return _self

            def __exit__(_self, *_args):
                self.calls.append(f"end:{label}")
                return False

        return _Range()


def test_window_uses_durable_numeric_selection() -> None:
    window = TrainingIterationWindow(start_step=4, warmup_steps=2, active_steps=3)
    assert [window.phase(i) for i in range(8)] == [
        "before", "before", "before", "before", "warmup", "warmup", "active", "active"
    ]
    assert window.end_step == 8


def test_window_rejects_selector_shaped_configuration() -> None:
    with pytest.raises(TypeError):
        NativeProfiler(window={"start_step": 0, "warmup_steps": 0, "active_steps": 1, "selector": "x"})


def test_disabled_path_has_no_claim_and_no_backend(tmp_path) -> None:
    called = False

    def backend() -> _FakeBackend:
        nonlocal called
        called = True
        return _FakeBackend()

    callback = NativeProfiler(enabled=False, backend_factory=backend)
    assert callback.ownership_claims() == []
    with callback.run_scope(make_run_context(tmp_path)):
        pass
    assert called is False


def test_fake_backend_is_started_stepped_and_stopped(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        backend_factory=lambda: backend,
    )
    session = callback.run_scope(make_run_context(tmp_path))
    with session:
        session._start()  # type: ignore[attr-defined]
        session.checkpoint()  # type: ignore[attr-defined]
    assert backend.calls == ["start", "step", "stop"]


def test_summary_and_trace_are_bounded_and_contained() -> None:
    backend = _FakeBackend()
    assert len(TorchOperatorSummary(max_rows=3).rows(backend)) == 3
    with pytest.raises(ValueError):
        ChromeTraceExport("../escape.json")


def test_live_callback_and_session_refuse_serialization() -> None:
    callback = NativeProfiler(enabled=True, backend_factory=_FakeBackend)
    for operation in (pickle.dumps, copy.deepcopy):
        with pytest.raises(TypeError):
            operation(callback)


def test_typed_ranges_nest_in_fixed_order_and_skip_absent_sampler(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(enabled=True, backend_factory=lambda: backend)
    context = make_run_context(tmp_path)
    session = callback.run_scope(context)
    with session:
        for count, event in enumerate(
            (
                Started(TrainingIteration(0)),
                Started(CollectSamples(0)),
                Started(SamplerBurnIn(2)),
                Ended(SamplerBurnIn(2)),
                Started(SamplerProduction(3)),
                Ended(SamplerProduction(3)),
                Ended(CollectSamples(0)),
                Ended(TrainingIteration(0)),
            ),
            1,
        ):
            callback.handle_occurrence(Occurrence(event, count), context)
    assert backend.calls[1:9] == [
        "start:TrainingIteration", "start:CollectSamples", "start:SamplerBurnIn",
        "end:SamplerBurnIn", "start:SamplerProduction", "end:SamplerProduction",
        "end:CollectSamples", "end:TrainingIteration",
    ]


def test_two_enabled_callbacks_conflict_before_scope_mutation() -> None:
    first = NativeProfiler(enabled=True)
    second = NativeProfiler(enabled=True)
    assert claims_conflict(first.ownership_claims()[0], second.ownership_claims()[0])


def test_disabled_fresh_process_does_not_load_torch_profiler() -> None:
    code = (
        "import sys; from tpen.callback import NativeProfiler; "
        "p=NativeProfiler(enabled=False); assert p.ownership_claims()==[]; "
        "assert 'torch.profiler' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert result.stderr == ""


def test_missed_window_does_not_start_backend(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(5, 1, 1),
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    with callback.run_scope(context):
        callback.handle_occurrence(Occurrence(Started(TrainingIteration(9)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(TrainingIteration(9)), 1), context)
    assert backend.calls == []

