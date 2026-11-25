"""Contract tests for the opt-in native profiler callback."""

from __future__ import annotations

import copy
import pickle

import pytest

from tpen.callback import (
    ChromeTraceExport,
    NativeProfiler,
    TorchOperatorSummary,
    TrainingIterationWindow,
)


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


def test_window_uses_durable_numeric_selection() -> None:
    window = TrainingIterationWindow(start_step=4, warmup_steps=2, active_steps=3)
    assert [window.phase(i) for i in range(8)] == [
        "before", "before", "before", "before", "warmup", "warmup", "active", "active"
    ]
    assert window.end_step == 8


def test_window_rejects_selector_shaped_configuration() -> None:
    with pytest.raises(TypeError):
        NativeProfiler(window={"start_step": 0, "warmup_steps": 0, "active_steps": 1, "selector": "x"})


def test_disabled_path_has_no_claim_and_no_backend() -> None:
    called = False

    def backend() -> _FakeBackend:
        nonlocal called
        called = True
        return _FakeBackend()

    callback = NativeProfiler(enabled=False, backend_factory=backend)
    assert callback.ownership_claims() == []
    with callback.run_scope(object()):
        pass
    assert called is False


def test_fake_backend_is_started_stepped_and_stopped() -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        backend_factory=lambda: backend,
    )
    session = callback.run_scope(object())
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

