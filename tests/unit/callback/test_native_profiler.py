"""Contract tests for the opt-in native profiler callback."""

from __future__ import annotations

import copy
import os
import json
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
from tpen.observability_artifacts import (
    ObservabilityBundleWriter,
    ObservabilityManifest,
    RunArtifactSink,
    inspect_run_observability,
)
from tpen.observability_reconciliation import reconcile_run_observability
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


class _FailingBackend(_FakeBackend):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure

    def start(self) -> None:
        if self.failure == "start":
            raise RuntimeError("start failed")
        super().start()

    def step(self) -> None:
        if self.failure == "step":
            raise RuntimeError("step failed")
        super().step()

    def stop(self) -> None:
        if self.failure == "stop":
            raise RuntimeError("stop failed")
        super().stop()

    def export_chrome_trace(self, _path: str) -> None:
        if self.failure == "export":
            raise RuntimeError("export failed")


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
        events = (
            (1, Started(TrainingIteration(0))), (1, Started(CollectSamples(0))),
            (1, Started(SamplerBurnIn(2))), (1, Ended(SamplerBurnIn(2))),
            (1, Started(SamplerProduction(3))), (1, Ended(SamplerProduction(3))),
            (1, Ended(CollectSamples(0))), (1, Ended(TrainingIteration(0))),
        )
        for count, event in events:
            callback.handle_occurrence(Occurrence(event, count), context)
    range_calls = [call for call in backend.calls if call.startswith(("start:", "end:"))]
    assert range_calls == [
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
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=os.getcwd(), env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    assert result.returncode == 0, result.stderr
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


def test_window_counts_and_overshoot_are_recorded(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(enabled=True, window=TrainingIterationWindow(2, 1, 1), backend_factory=lambda: backend)
    context = make_run_context(tmp_path)
    with callback.run_scope(context):
        for step in (2, 3, 4):
            callback.handle_occurrence(Occurrence(Started(TrainingIteration(step)), 1), context)
            callback.handle_occurrence(Occurrence(Ended(TrainingIteration(step)), 1), context)
    status = next(context.run_dir.rglob("profile-status.json")).read_text()
    payload = json.loads(status)
    assert payload["durable_steps"] == [2, 3, 4]
    assert "overshoot_steps=1" in payload["activity_diagnostics"]


def test_open_partial_native_bundle_is_read_only_reconcilable(tmp_path) -> None:
    context = make_run_context(tmp_path)
    sink = RunArtifactSink(relative_dir="native-profiling")
    writer = ObservabilityBundleWriter(context.artifact_manager, context.invocation_id, sink=sink)
    manifest = ObservabilityManifest(
        invocation_id=context.invocation_id, lifecycle_state="open", data_completeness="unknown",
        run_id=context.metadata.run_id, producer={"name": "test"},
        configuration_digest="0" * 64, inventory_id="inventory-test",
        descriptor_set_id="descriptors-test", observers=({"observer_id": "native-profiler"},), limits={},
    )
    writer.write_open_manifest(manifest)
    inspected = inspect_run_observability(context.run_dir, relative_dir="native-profiling")
    reconciled = reconcile_run_observability(context.run_dir, relative_dir="native-profiling")
    assert inspected.status == "present"
    assert reconciled.invocations[0].reconciliation_state == "open_interrupted"


@pytest.mark.parametrize("failure", ["start", "step", "stop", "export"])
def test_backend_failures_clear_session_and_preserve_primary(failure: str, tmp_path) -> None:
    backend = _FailingBackend(failure)
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        chrome_trace=ChromeTraceExport() if failure == "export" else None,
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    primary = RuntimeError("application failed")
    expected = "application failed" if failure in {"stop", "export"} else f"{failure} failed"
    with pytest.raises(RuntimeError, match=expected):
        with callback.run_scope(context) as session:
            session._start()  # type: ignore[attr-defined]
            if failure == "step":
                session.checkpoint()  # type: ignore[attr-defined]
            elif failure == "start":
                raise primary
            else:
                raise primary
    assert callback._active_session is None  # type: ignore[attr-defined]

