"""Contract tests for the opt-in native profiler callback."""

from __future__ import annotations

import copy
import os
import json
import pickle
import subprocess
import sys
import warnings

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

    def export_chrome_trace(self, path: str) -> None:
        self.calls.append("export_chrome_trace")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"traceEvents": []}, handle)


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


class _SummaryFailingBackend(_FakeBackend):
    def key_averages(self) -> list[object]:
        raise RuntimeError("summary failed")


class _RangeExitFailureBackend(_FakeBackend):
    def record_function(self, label: str):
        self.calls.append(f"start:{label}")
        parent = self

        class _Range:
            def __enter__(_self):
                return _self

            def __exit__(_self, *_args):
                parent.calls.append(f"end:{label}")
                raise RuntimeError("range exit failed")

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
    assert backend.calls == ["start", "stop"]


def test_summary_is_bounded() -> None:
    backend = _FakeBackend()
    assert len(TorchOperatorSummary(max_rows=3).rows(backend)) == 3


def test_chrome_trace_export_is_fixed_and_published_to_both_refs(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        trace={},
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    with callback.run_scope(context):
        callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(TrainingIteration(0)), 1), context)

    status_path = next(context.run_dir.rglob("profile-status.json"))
    status = json.loads(status_path.read_text())
    trace_path = next(
        context.run_dir.rglob("chrome-trace.json"),
        None,
    )
    assert trace_path is not None, status
    assert trace_path.read_text(encoding="utf-8") == '{"traceEvents": []}'
    manifest = json.loads(next(context.run_dir.rglob("manifest.json")).read_text())
    status_ref = next(ref for ref in status["artifacts"] if ref["kind"] == "trace")
    manifest_ref = next(ref for ref in manifest["artifacts"] if ref["kind"] == "trace")
    assert status_ref == manifest_ref
    assert status_ref["path"] == (
        "profiles/native-profiler/step-0-0/chrome-trace.json"
    )
    assert status["export_status"] == "complete"


def test_chrome_trace_missing_backend_capability_is_partial_and_preserves_primary(tmp_path) -> None:
    class _NoTraceBackend(_FakeBackend):
        export_chrome_trace = None  # type: ignore[assignment]

    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        trace=ChromeTraceExport(),
        backend_factory=_NoTraceBackend,
    )
    context = make_run_context(tmp_path)
    with callback.run_scope(context):
        callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(TrainingIteration(0)), 1), context)
    status = json.loads(next(context.run_dir.rglob("profile-status.json")).read_text())
    assert status["export_status"] == "failed"
    assert status["data_completeness"] == "partial"
    assert status.get("artifacts") is None
    assert any(item.startswith("trace_export_failed") for item in status["activity_diagnostics"])


def test_live_callback_and_session_refuse_serialization() -> None:
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        backend_factory=_FakeBackend,
    )
    for operation in (pickle.dumps, copy.deepcopy):
        with pytest.raises(TypeError):
            operation(callback)


def test_typed_ranges_nest_in_fixed_order_and_skip_absent_sampler(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        backend_factory=lambda: backend,
    )
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
    window = TrainingIterationWindow(0, 0, 1)
    first = NativeProfiler(enabled=True, window=window)
    second = NativeProfiler(enabled=True, window=window)
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


def test_first_observed_warmup_step_after_window_start_is_not_started(tmp_path) -> None:
    """Resume after start_step must not shift the requested capture window."""
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(start_step=5, warmup_steps=2, active_steps=2),
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    with callback.run_scope(context):
        callback.handle_occurrence(Occurrence(Started(TrainingIteration(6)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(TrainingIteration(6)), 1), context)

    assert backend.calls == []
    payload = json.loads(next(context.run_dir.rglob("profile-status.json")).read_text())
    assert payload["lifecycle_state"] == "not_started"
    assert "window_start_not_observed" in payload["activity_diagnostics"]
    inspected = inspect_run_observability(context.run_dir, relative_dir="native-profiling")
    assert inspected.invocations[0].profile_statuses[0].lifecycle_state == "not_started"
    reconciled = reconcile_run_observability(context.run_dir, relative_dir="native-profiling")
    assert reconciled.invocations[0].summary.profile_windows[0].lifecycle_state == "not_started"


def test_request_stop_before_window_prevents_activation(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 1, 1),
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    with callback.run_scope(context) as session:
        session.request_stop()
        callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(TrainingIteration(0)), 1), context)
    assert backend.calls == []


def test_final_active_iteration_stops_backend_before_next_callback_event(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    with callback.run_scope(context):
        callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(TrainingIteration(0)), 1), context)
        assert backend.calls.count("stop") == 1
        ranges_before = [call for call in backend.calls if call.startswith(("start:", "end:"))]
        callback.handle_occurrence(Occurrence(Started(CollectSamples(0)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(CollectSamples(0)), 1), context)
        ranges_after = [call for call in backend.calls if call.startswith(("start:", "end:"))]
        assert ranges_after == ranges_before
    assert backend.calls.count("stop") == 1


def test_out_of_order_ended_occurrence_does_not_pop_unrelated_range(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    with callback.run_scope(context):
        callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), context)
        callback.handle_occurrence(Occurrence(Started(CollectSamples(0)), 1), context)
        callback.handle_occurrence(Occurrence(Started(SamplerBurnIn(2)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(CollectSamples(0)), 99), context)
        assert "end:SamplerBurnIn" not in backend.calls
        callback.handle_occurrence(Occurrence(Ended(SamplerBurnIn(2)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(CollectSamples(0)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(TrainingIteration(0)), 1), context)
    range_calls = [call for call in backend.calls if call.startswith(("start:", "end:"))]
    assert range_calls[-3:] == ["end:SamplerBurnIn", "end:CollectSamples", "end:TrainingIteration"]


def test_open_manifest_is_visible_while_capture_is_active(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    with callback.run_scope(context):
        callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), context)
        assert list((context.run_dir / "native-profiling").rglob("manifest.json"))


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
    assert payload["durable_steps"] == [2, 3]
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


@pytest.mark.parametrize("failure", ["start", "step", "stop"])
def test_backend_failures_clear_session_and_preserve_primary(failure: str, tmp_path) -> None:
    backend = _FailingBackend(failure)
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 2 if failure == "step" else 1),
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    primary = RuntimeError("application failed")
    expected = "application failed"
    with pytest.raises(RuntimeError, match=expected):
        with callback.run_scope(context) as session:
            session._start()  # type: ignore[attr-defined]
            if failure == "step":
                callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), context)
                callback.handle_occurrence(Occurrence(Ended(TrainingIteration(0)), 1), context)
                raise primary
            elif failure == "start":
                raise primary
            else:
                raise primary
    assert callback._active_session is None  # type: ignore[attr-defined]


@pytest.mark.parametrize("warmup_steps", [0, 1, 2])
def test_real_torch_profiler_adapter_emits_phases_and_fixed_ranges(
    warmup_steps: int, tmp_path
) -> None:
    """The CPU adapter must use module-level record_function on Cannon's torch."""

    import torch
    from torch.profiler import ProfilerAction

    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, warmup_steps, 2),
        summary=TorchOperatorSummary(max_rows=256),
    )
    context = make_run_context(tmp_path)
    observed_actions = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with callback.run_scope(context) as session:
            for step in range(warmup_steps + 2):
                callback.handle_occurrence(Occurrence(Started(TrainingIteration(step)), 1), context)
                assert session.backend is not None  # activation precedes the range
                observed_actions.append(session.backend._raw.current_action)  # type: ignore[attr-defined]
                with torch.profiler.record_function(
                    "warmup_only" if step < warmup_steps else f"active_only_{step - warmup_steps}"
                ):
                    _ = torch.ones(4) + step
                callback.handle_occurrence(Occurrence(Ended(TrainingIteration(step)), 1), context)
            callback.handle_occurrence(Occurrence(Started(TrainingIteration(warmup_steps + 2)), 1), context)
            with torch.profiler.record_function("post_window_only"):
                _ = torch.ones(4) + 3
            callback.handle_occurrence(Occurrence(Ended(TrainingIteration(warmup_steps + 2)), 1), context)
            callback.handle_occurrence(Occurrence(Started(CollectSamples(0)), 1), context)
            callback.handle_occurrence(Occurrence(Ended(CollectSamples(0)), 1), context)

    expected_actions = (
        [ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE]
        if warmup_steps == 0
        else [ProfilerAction.WARMUP] * warmup_steps
        + [ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE]
    )
    assert observed_actions == expected_actions
    if warmup_steps >= 1:
        assert all("Profiler won't be using warmup" not in str(w.message) for w in caught)
    payload = json.loads(next(context.run_dir.rglob("summary.json")).read_text())
    names = {row["name"] for row in payload["rows"]}
    assert {f"active_only_{i}" for i in range(2)} <= names
    if warmup_steps:
        assert "warmup_only" not in names
    assert "post_window_only" not in names
    assert payload["truncated"] is False


def test_summary_failure_is_failed_partial_through_inspection_and_reconciliation(tmp_path) -> None:
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        backend_factory=_SummaryFailingBackend,
    )
    context = make_run_context(tmp_path)
    with callback.run_scope(context):
        callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(TrainingIteration(0)), 1), context)
    status = json.loads(next(context.run_dir.rglob("profile-status.json")).read_text())
    manifest = json.loads(next(context.run_dir.rglob("manifest.json")).read_text())
    assert status["export_status"] == "failed"
    assert status["data_completeness"] == "partial"
    assert manifest["data_completeness"] == "partial"
    inspected = inspect_run_observability(context.run_dir, relative_dir="native-profiling")
    assert inspected.invocations[0].profile_statuses[0].export_status == "failed"
    reconciled = reconcile_run_observability(context.run_dir, relative_dir="native-profiling")
    bundle = reconciled.invocations[0]
    assert bundle.reconciliation_state == "failed"
    assert bundle.data_completeness == "partial"
    assert bundle.summary.profile_windows[0].export_status == "failed"


def test_failed_end_dispatch_preserves_application_primary(tmp_path) -> None:
    backend = _RangeExitFailureBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    session = callback.run_scope(context)
    with pytest.raises(RuntimeError, match="primary"):
        with session as live:
            callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), context)
            callback.handle_occurrence(Occurrence(Ended(TrainingIteration(0)), 1), context)
            raise RuntimeError("primary")
    assert "range_exit_failed:RuntimeError:range exit failed" in live._diagnostics  # type: ignore[attr-defined]


def test_matched_out_of_order_range_is_unwound_once_on_exit(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    session = callback.run_scope(context)
    with session:
        callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), context)
        callback.handle_occurrence(Occurrence(Started(CollectSamples(0)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(TrainingIteration(0)), 1), context)
    assert "range_lifo_mismatch:TrainingIteration:1" in session._diagnostics  # type: ignore[attr-defined]
    assert backend.calls.count("end:CollectSamples") == 1
    assert backend.calls.count("end:TrainingIteration") == 1


def test_summary_truncation_and_failed_window_finalization_are_honest(tmp_path) -> None:
    backend = _FakeBackend()
    callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        summary=TorchOperatorSummary(max_rows=3),
        backend_factory=lambda: backend,
    )
    context = make_run_context(tmp_path)
    with callback.run_scope(context):
        callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), context)
        callback.handle_occurrence(Occurrence(Ended(TrainingIteration(0)), 1), context)
    summary = json.loads(next(context.run_dir.rglob("summary.json")).read_text())
    status = json.loads(next(context.run_dir.rglob("profile-status.json")).read_text())
    manifest = json.loads(next(context.run_dir.rglob("manifest.json")).read_text())
    assert summary["total_rows"] == 20
    assert summary["truncated"] is True
    assert status["lifecycle_state"] == "closed"
    assert status["data_completeness"] == "complete"
    assert manifest["lifecycle_state"] == "closed"

    failed = _FailingBackend("stop")
    failed_callback = NativeProfiler(
        enabled=True,
        window=TrainingIterationWindow(0, 0, 1),
        backend_factory=lambda: failed,
    )
    failed_context = make_run_context(tmp_path / "failed")
    with failed_callback.run_scope(failed_context):
        failed_callback.handle_occurrence(Occurrence(Started(TrainingIteration(0)), 1), failed_context)
        failed_callback.handle_occurrence(Occurrence(Ended(TrainingIteration(0)), 1), failed_context)
    failed_status = json.loads(next(failed_context.run_dir.rglob("profile-status.json")).read_text())
    failed_manifest = json.loads(next(failed_context.run_dir.rglob("manifest.json")).read_text())
    assert failed_status["lifecycle_state"] == "unsafe"
    assert failed_status["data_completeness"] == "partial"
    assert "stop_overshoot_bounds_seconds" not in failed_status
    assert failed_manifest["lifecycle_state"] == "unsafe"


def test_enabled_profiler_requires_explicit_window_and_exact_sink() -> None:
    with pytest.raises(ValueError, match="explicit window"):
        NativeProfiler(enabled=True)

    class _SinkSubclass(RunArtifactSink):
        pass

    with pytest.raises(TypeError, match="exactly RunArtifactSink"):
        NativeProfiler(
            enabled=True,
            window=TrainingIterationWindow(0, 0, 1),
            sink=_SinkSubclass(),
        )

