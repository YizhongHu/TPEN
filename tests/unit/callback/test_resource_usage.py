"""Tests for the run-level resource-usage callback.

Migrated off the ``run_start``/``run_end``/``exception``/``run_failed`` strings
onto `tpen.run_events` (item ``39eacd99``). These call ``handle_occurrence``
directly, which is this callback's own contract. Delivery of those occurrences
by the real dispatcher, and their emission by `tpen.run`, are covered in
``test_typed_run_lifecycle.py``.
"""

from __future__ import annotations

import json
import math
import pytest
from types import SimpleNamespace

from tpen.callback import ResourceUsage
from tpen.callback import resource_usage as resource_usage_module
from tpen.events import Occurrence
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tpen.process_resources import (
    ProcessRUsageProbe,
    ProcessResourceBaseline,
    ProcessResourceResult,
    ResourceScope,
    ResourceUnavailable,
)
from tpen.accelerator import (
    AcceleratorIdentity,
    AcceleratorKind,
    AllocatorUnavailable,
    AllocatorUsage,
)
from tpen.distributed import ExecutionTopology, RankLocalJSONLWriter
from tpen.logging import JSONL
from tpen.logging.base import LogRecord
from tpen import process_resources as process_resources_module
from tests.unit.callback.support import RecordingContext


def _deliver(callback: ResourceUsage, context: RecordingContext, event: object) -> None:
    """Hand one run-lifecycle occurrence to the callback."""

    callback.handle_occurrence(Occurrence(event=event, count=1), context)


class _FakeAllocatorProbe:
    """Configured-device probe stand-in for callback projection tests."""

    def __init__(self, usage: AllocatorUsage) -> None:
        self.usage = usage
        self.reset_calls = 0

    def reset(self) -> AcceleratorIdentity:
        self.reset_calls += 1
        return self.usage.identity

    def read(self) -> AllocatorUsage:
        return self.usage


def test_resource_usage_logs_peak_rss_at_run_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    context = RecordingContext()
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 512.0)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    assert context.by_namespace("runtime") == [
        {"metrics": {"peak_memory_mb": 512.0}, "step": 0, "namespace": "runtime"}
    ]


def test_resource_usage_resets_and_logs_cuda_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    context = RecordingContext()
    allocator = _FakeAllocatorProbe(
        AllocatorUsage(
            identity=AcceleratorIdentity(AcceleratorKind.CUDA, 1, "GPU-1"),
            allocated_mb=3.0,
            reserved_mb=8.0,
            device_count=2,
        )
    )
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 512.0, allocator_probe=allocator)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    assert allocator.reset_calls == 1
    assert context.latest("runtime") == {
        "peak_memory_mb": 512.0,
        "cuda_max_memory_allocated_mb": 3.0,
        "cuda_max_memory_reserved_mb": 8.0,
        "cuda_device_count": 2,
        "accelerator_max_memory_allocated_mb": 3.0,
        "accelerator_max_memory_reserved_mb": 8.0,
        "accelerator_device_count": 2,
    }


def test_resource_usage_writes_process_and_device_profiles_through_real_writer(tmp_path) -> None:
    topology = ExecutionTopology(
        global_rank=0,
        global_size=1,
        local_rank=0,
        local_size=1,
        node_rank=0,
        node_size=1,
        host="node-a",
        pid=42,
        device="cuda:1",
        device_identity=AcceleratorIdentity(AcceleratorKind.CUDA, 1, "GPU-1"),
    )
    context = RecordingContext()
    context.topology = topology
    context.profile_writer = RankLocalJSONLWriter(tmp_path, topology)
    allocator = _FakeAllocatorProbe(
        AllocatorUsage(
            identity=topology.device_identity,
            allocated_mb=3.0,
            reserved_mb=8.0,
            device_count=2,
        )
    )
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 512.0, allocator_probe=allocator)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    lines = [json.loads(line) for line in context.profile_writer.path.read_text().splitlines()]
    assert [line["scope"] for line in lines] == ["device", "process"]
    assert lines[0]["metrics"]["allocated_mb"] == 3.0
    assert "peak_rss_mb" in lines[1]["metrics"]


@pytest.mark.parametrize("invalid_allocated_mb", [math.nan, math.inf])
def test_resource_usage_isolates_nonfinite_allocator_metric_at_terminal_boundary(
    tmp_path, invalid_allocated_mb: float
) -> None:
    """One bad allocator counter cannot erase finite terminal resource evidence."""

    class JSONLRecordingContext(RecordingContext):
        """Capture callback records while routing them through strict JSONL."""

        def __init__(self) -> None:
            super().__init__()
            self.terminal_logger = JSONL(tmp_path / "metrics.jsonl")

        def log(self, metrics, *, step=None, namespace="run") -> None:
            super().log(metrics, step=step, namespace=namespace)
            self.terminal_logger.log(
                LogRecord(step=step, namespace=namespace, metrics=dict(metrics))
            )

    topology = ExecutionTopology(
        global_rank=0,
        global_size=1,
        local_rank=0,
        local_size=1,
        node_rank=0,
        node_size=1,
        host="node-a",
        pid=42,
        device="cuda:1",
        device_identity=AcceleratorIdentity(AcceleratorKind.CUDA, 1, "GPU-1"),
    )
    context = JSONLRecordingContext()
    context.topology = topology
    context.profile_writer = RankLocalJSONLWriter(tmp_path, topology)
    allocator = _FakeAllocatorProbe(
        AllocatorUsage(
            identity=topology.device_identity,
            allocated_mb=invalid_allocated_mb,
            reserved_mb=8.0,
            device_count=2,
        )
    )
    callback = ResourceUsage(
        process_probe=_FixedProbe(_fixed_result()),
        peak_rss_mb_reader=lambda: 512.0,
        allocator_probe=allocator,
    )

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    terminal_lines = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [line["namespace"] for line in terminal_lines] == ["runtime", "process"]
    runtime = terminal_lines[0]["metrics"]
    assert runtime["accelerator_max_memory_allocated_unavailable"] is True
    assert runtime["accelerator_max_memory_reserved_mb"] == 8.0
    assert runtime["accelerator_device_count"] == 2
    assert "cuda_max_memory_allocated_mb" not in runtime

    profile_lines = [
        json.loads(line)
        for line in context.profile_writer.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["scope"] for line in profile_lines] == ["device", "process"]
    assert "metrics" in profile_lines[0], "finite allocator siblings must survive in the device profile metrics"
    assert profile_lines[0]["metrics"] == {
        "allocated_mb_unavailable": True,
        "device_count": 2,
        "reserved_mb": 8.0,
    }
    assert "peak_rss_mb" in profile_lines[1]["metrics"]


def test_resource_usage_builds_probe_from_configured_context_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RecordingContext()
    context.metadata.device = "cuda:3"
    allocator = _FakeAllocatorProbe(
        AllocatorUsage(
            identity=AcceleratorIdentity(AcceleratorKind.CUDA, 3, "GPU-3"),
            allocated_mb=2.0,
            reserved_mb=4.0,
            device_count=4,
        )
    )
    devices: list[str] = []

    def build_probe(device: str) -> _FakeAllocatorProbe:
        devices.append(device)
        return allocator

    monkeypatch.setattr(resource_usage_module, "TorchAllocatorPeakProbe", build_probe)
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 1.0)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    assert devices == ["cuda:3"]
    assert allocator.reset_calls == 1


def test_resource_usage_recreates_auto_probe_for_each_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = (RecordingContext(), RecordingContext())
    contexts[0].metadata.device = "cuda:0"
    contexts[1].metadata.device = "xpu:1"
    devices: list[str] = []
    probes: list[_FakeAllocatorProbe] = []

    def build_probe(device: str) -> _FakeAllocatorProbe:
        devices.append(device)
        kind = AcceleratorKind.CUDA if device == "cuda:0" else AcceleratorKind.OTHER
        probe = _FakeAllocatorProbe(
            usage=AllocatorUsage(
                identity=AcceleratorIdentity(kind=kind, index=0, uuid=device),
                allocated_mb=1.0 if device == "cuda:0" else 2.0,
                reserved_mb=3.0 if device == "cuda:0" else 4.0,
                device_count=1 if device == "cuda:0" else 2,
            )
        )
        probes.append(probe)
        return probe

    monkeypatch.setattr(resource_usage_module, "TorchAllocatorPeakProbe", build_probe)
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 1.0)

    for context in contexts:
        _deliver(callback, context, RunStarted())
        _deliver(callback, context, RunCompleted())

    assert devices == ["cuda:0", "xpu:1"]
    assert [probe.reset_calls for probe in probes] == [1, 1]
    assert contexts[0].latest("runtime")["accelerator_max_memory_allocated_mb"] == 1.0
    assert contexts[1].latest("runtime")["accelerator_max_memory_allocated_mb"] == 2.0


def test_resource_usage_does_not_emit_allocator_metrics_for_xpu() -> None:
    context = RecordingContext()
    allocator = _FakeAllocatorProbe(
        AllocatorUsage(
            identity=AcceleratorIdentity(AcceleratorKind.OTHER, 2, "XPU-2"),
            allocated_mb=3.0,
            reserved_mb=5.0,
            device_count=4,
        )
    )
    callback = ResourceUsage(allocator_probe=allocator)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    runtime = context.latest("runtime")
    assert runtime["accelerator_max_memory_allocated_mb"] == 3.0
    assert runtime["accelerator_max_memory_reserved_mb"] == 5.0
    assert runtime["accelerator_device_count"] == 4
    assert not any(key.startswith("cuda_") for key in runtime)


def test_allocator_counter_failures_preserve_independent_readings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    import tpen.accelerator as accelerator

    class Backend:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(index: int):
            return SimpleNamespace(uuid="GPU-1")

        @staticmethod
        def max_memory_allocated(device: object) -> int:
            return 3 * 1024 * 1024

        @staticmethod
        def max_memory_reserved(device: object) -> int:
            raise AttributeError("reserved counter unavailable")

        @staticmethod
        def device_count() -> int:
            return 4

    class FakeTorch:
        device = staticmethod(torch.device)
        version = SimpleNamespace(hip=None)

    monkeypatch.setattr(accelerator, "_torch", lambda feature: FakeTorch)
    monkeypatch.setattr(accelerator, "device_module", lambda *args, **kwargs: Backend)

    usage = accelerator.TorchAllocatorPeakProbe("cuda:1").read()

    assert usage == AllocatorUsage(
        identity=AcceleratorIdentity(
            kind=AcceleratorKind.CUDA,
            index=1,
            uuid="GPU-1",
        ),
        allocated_mb=3.0,
        reserved_mb=AllocatorUnavailable(reason="AttributeError: reserved counter unavailable"),
        device_count=4,
    )


def test_resource_usage_keeps_cuda_aliases_for_rocm() -> None:
    context = RecordingContext()
    allocator = _FakeAllocatorProbe(
        AllocatorUsage(
            identity=AcceleratorIdentity(AcceleratorKind.ROCM, 1, "GPU-ROCM-1"),
            allocated_mb=3.0,
            reserved_mb=8.0,
            device_count=2,
        )
    )
    callback = ResourceUsage(allocator_probe=allocator)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    runtime = context.latest("runtime")
    assert runtime["accelerator_max_memory_allocated_mb"] == 3.0
    assert runtime["accelerator_max_memory_reserved_mb"] == 8.0
    assert runtime["accelerator_device_count"] == 2
    assert runtime["cuda_max_memory_allocated_mb"] == 3.0
    assert runtime["cuda_max_memory_reserved_mb"] == 8.0
    assert runtime["cuda_device_count"] == 2


def test_resource_usage_logs_at_the_failure_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """One typed `RunFailed` replaces the ``run_failed``/``exception`` pair.

    Both strings carried the same payload and this callback answered both, so a
    failed run logged ``runtime`` twice. The single record here is the observable
    difference.
    """

    context = RecordingContext()
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 100.5)

    _deliver(callback, context, RunFailed(exception_type="RuntimeError", exception_message="boom"))

    assert context.by_namespace("runtime") == [
        {"metrics": {"peak_memory_mb": 100.5}, "step": 0, "namespace": "runtime"}
    ]


def test_resource_usage_omits_metrics_from_failing_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_reader() -> float:
        raise OSError("getrusage unavailable")

    context = RecordingContext()
    callback = ResourceUsage(peak_rss_mb_reader=broken_reader)

    _deliver(callback, context, RunCompleted())

    assert context.records == []


def test_resource_usage_default_reader_returns_positive_mib() -> None:
    assert resource_usage_module._default_peak_rss_mb() > 0.0


def test_process_probe_reports_counter_deltas_and_linux_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    readings = iter(
        (
            SimpleNamespace(
                ru_utime=1.25,
                ru_stime=0.5,
                ru_inblock=4,
                ru_oublock=6,
                ru_nvcsw=8,
                ru_nivcsw=10,
                ru_maxrss=1024,
            ),
            SimpleNamespace(
                ru_utime=2.0,
                ru_stime=0.75,
                ru_inblock=9,
                ru_oublock=7,
                ru_nvcsw=11,
                ru_nivcsw=14,
                ru_maxrss=2048,
            ),
        )
    )
    monkeypatch.setattr(process_resources_module.resource, "getrusage", lambda scope: next(readings))
    monkeypatch.setattr(process_resources_module.sys, "platform", "linux")

    probe = ProcessRUsageProbe(ResourceScope.PROCESS)
    baseline = probe.read()
    result = probe.result(baseline)

    assert result.user_cpu_seconds == pytest.approx(0.75)
    assert result.system_cpu_seconds == pytest.approx(0.25)
    assert result.read_block_operations == 5
    assert result.write_block_operations == 1
    assert result.voluntary_context_switches == 3
    assert result.involuntary_context_switches == 4
    assert result.peak_rss_mb == pytest.approx(2.0)


def test_process_probe_preserves_unavailable_counter_evidence_when_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(scope):
        raise OSError("getrusage unavailable")

    monkeypatch.setattr(process_resources_module.resource, "getrusage", fail)

    result = ProcessRUsageProbe().read()

    assert all(
        isinstance(value, ResourceUnavailable)
        for value in (
            result.user_cpu_seconds,
            result.system_cpu_seconds,
            result.read_block_operations,
            result.write_block_operations,
            result.voluntary_context_switches,
            result.involuntary_context_switches,
            result.peak_rss_mb,
        )
    )


def test_process_probe_normalizes_macos_peak_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        process_resources_module.resource,
        "getrusage",
        lambda scope: SimpleNamespace(
            ru_utime=0.0,
            ru_stime=0.0,
            ru_inblock=0,
            ru_oublock=0,
            ru_nvcsw=0,
            ru_nivcsw=0,
            ru_maxrss=2 * 1024 * 1024,
        ),
    )
    monkeypatch.setattr(process_resources_module.sys, "platform", "darwin")

    assert ProcessRUsageProbe().read().peak_rss_mb == pytest.approx(2.0)


class _FixedProbe(ProcessRUsageProbe):
    """Probe stand-in that makes callback boundary assertions deterministic."""

    def __init__(self, result: ProcessResourceResult) -> None:
        super().__init__()
        self.result_value = result

    def read(self) -> ProcessResourceBaseline:
        return ProcessResourceBaseline(
            user_cpu_seconds=0,
            system_cpu_seconds=0,
            read_block_operations=0,
            write_block_operations=0,
            voluntary_context_switches=0,
            involuntary_context_switches=0,
            peak_rss_mb=0,
        )

    def result(self, baseline: ProcessResourceBaseline) -> ProcessResourceResult:
        return self.result_value


def _fixed_result(*, unavailable: bool = False) -> ProcessResourceResult:
    value = ResourceUnavailable("probe failed") if unavailable else 1
    return ProcessResourceResult(
        user_cpu_seconds=value,
        system_cpu_seconds=value,
        read_block_operations=value,
        write_block_operations=value,
        voluntary_context_switches=value,
        involuntary_context_switches=value,
        peak_rss_mb=value,
    )


def test_resource_usage_logs_process_metrics_at_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    context = RecordingContext()
    callback = ResourceUsage(process_probe=_FixedProbe(_fixed_result()), peak_rss_mb_reader=lambda: 4.0)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    assert context.latest("process") == {
        "process_user_cpu_seconds": 1.0,
        "process_system_cpu_seconds": 1.0,
        "process_read_block_operations": 1.0,
        "process_write_block_operations": 1.0,
        "process_voluntary_context_switches": 1.0,
        "process_involuntary_context_switches": 1.0,
    }
    assert context.latest("runtime")["peak_memory_mb"] == 4.0


def test_resource_usage_keeps_terminal_logging_when_profile_write_fails() -> None:
    context = RecordingContext()
    context.topology = ExecutionTopology(
        global_rank=0,
        global_size=1,
        local_rank=0,
        local_size=1,
        node_rank=0,
        node_size=1,
        host="node-a",
        pid=42,
        device="cpu",
    )
    context.profile_writer = object()

    def fail_profile_write(record: object) -> None:
        raise OSError("profile filesystem unavailable")

    context.write_profile = fail_profile_write
    callback = ResourceUsage(
        process_probe=_FixedProbe(_fixed_result()),
        peak_rss_mb_reader=lambda: 4.0,
    )

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    assert context.latest("runtime")["peak_memory_mb"] == 4.0
    assert context.latest("process")["process_user_cpu_seconds"] == 1.0


def test_resource_usage_prefers_process_peak_over_default_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = ProcessResourceResult(
        user_cpu_seconds=1,
        system_cpu_seconds=1,
        read_block_operations=1,
        write_block_operations=1,
        voluntary_context_switches=1,
        involuntary_context_switches=1,
        peak_rss_mb=7.0,
    )
    monkeypatch.setattr(resource_usage_module, "_default_peak_rss_mb", lambda: 99.0)
    context = RecordingContext()
    callback = ResourceUsage(process_probe=_FixedProbe(result))

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    assert context.by_namespace("runtime") == [
        {"metrics": {"peak_memory_mb": 7.0}, "step": 0, "namespace": "runtime"}
    ]


def test_resource_usage_does_not_fallback_when_process_peak_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = ProcessResourceResult(
        user_cpu_seconds=1,
        system_cpu_seconds=1,
        read_block_operations=1,
        write_block_operations=1,
        voluntary_context_switches=1,
        involuntary_context_switches=1,
        peak_rss_mb=ResourceUnavailable("probe failed"),
    )
    monkeypatch.setattr(resource_usage_module, "_default_peak_rss_mb", lambda: 99.0)
    context = RecordingContext()
    callback = ResourceUsage(process_probe=_FixedProbe(result))

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    assert context.by_namespace("runtime") == []
    assert context.latest("process")["process_peak_rss_unavailable"] is True


def test_resource_usage_logs_process_receipt_at_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    context = RecordingContext()
    callback = ResourceUsage(process_probe=_FixedProbe(_fixed_result()), peak_rss_mb_reader=lambda: 4.0)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunFailed(exception_type="RuntimeError", exception_message="boom"))

    assert context.latest("process") == {
        "process_user_cpu_seconds": 1.0,
        "process_system_cpu_seconds": 1.0,
        "process_read_block_operations": 1.0,
        "process_write_block_operations": 1.0,
        "process_voluntary_context_switches": 1.0,
        "process_involuntary_context_switches": 1.0,
    }
    assert context.latest("runtime")["peak_memory_mb"] == 4.0


def test_resource_usage_projects_unavailable_process_readings_as_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RecordingContext()
    callback = ResourceUsage(process_probe=_FixedProbe(_fixed_result(unavailable=True)))

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    process = context.latest("process")
    assert process["process_user_cpu_seconds_unavailable"] is True
    assert process["process_peak_rss_unavailable"] is True


def _strict_json_terminal_context(tmp_path) -> RecordingContext:
    """Build a recording context whose ``log`` also crosses a strict-JSON boundary.

    The callback's own assertions read `RecordingContext.records`, but a metric
    that is merely *recorded* has not yet proven it can be persisted: the real
    terminal boundary is `tpen.logging.jsonl.JSONL`, which serialises with
    ``allow_nan=False``. Routing every record through a real JSONL logger makes
    a non-finite scalar fail here exactly as it would in a run.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-test temporary directory owning ``metrics.jsonl``.

    Returns
    -------
    RecordingContext
        Context recording in memory and appending to ``metrics.jsonl``.
    """

    class JSONLRecordingContext(RecordingContext):
        """Capture callback records while routing them through strict JSONL."""

        def __init__(self) -> None:
            super().__init__()
            self.terminal_logger = JSONL(tmp_path / "metrics.jsonl")

        def log(self, metrics, *, step=None, namespace="run") -> None:
            super().log(metrics, step=step, namespace=namespace)
            self.terminal_logger.log(
                LogRecord(step=step, namespace=namespace, metrics=dict(metrics))
            )

    return JSONLRecordingContext()


def _terminal_records(tmp_path) -> list[dict]:
    """Read back every record that survived the strict-JSON terminal boundary."""

    return [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.parametrize("invalid_peak_rss_mb", [math.nan, math.inf])
def test_resource_usage_isolates_nonfinite_peak_rss_reader_at_terminal_boundary(
    tmp_path, invalid_peak_rss_mb: float
) -> None:
    """A non-finite injected peak RSS must not abort terminal logging.

    Sibling of ``test_resource_usage_isolates_nonfinite_allocator_metric_at_
    terminal_boundary``: that test pinned the invariant for allocator scalars,
    which `_is_unavailable_scalar` guards. The same invariant -- no single bad
    scalar erases the rest of the terminal evidence -- is asserted here for the
    ``peak_memory_mb`` egress path, whose reachable source today is the public
    ``peak_rss_mb_reader`` seam.
    """

    context = _strict_json_terminal_context(tmp_path)
    callback = ResourceUsage(
        process_probe=_FixedProbe(_fixed_result()),
        peak_rss_mb_reader=lambda: invalid_peak_rss_mb,
    )

    _deliver(callback, context, RunStarted())
    # No exception may escape the callback at the terminal boundary.
    _deliver(callback, context, RunCompleted())

    terminal_lines = _terminal_records(tmp_path)
    runtime_lines = [line for line in terminal_lines if line["namespace"] == "runtime"]
    process_lines = [line for line in terminal_lines if line["namespace"] == "process"]

    # The bad scalar is either dropped outright or degraded to a typed flag;
    # what it may never do is carry a non-finite value across the boundary.
    for line in runtime_lines:
        assert "peak_memory_mb" not in line["metrics"]

    # The finite process evidence must survive the bad runtime scalar intact.
    assert len(process_lines) == 1
    process = process_lines[0]["metrics"]
    assert process["process_user_cpu_seconds"] == 1.0
    assert process["process_system_cpu_seconds"] == 1.0
    assert "process_peak_rss_unavailable" not in process


@pytest.mark.parametrize("invalid_user_cpu_seconds", [math.nan, math.inf])
def test_resource_usage_isolates_nonfinite_process_counter_at_terminal_boundary(
    tmp_path, invalid_user_cpu_seconds: float
) -> None:
    """One non-finite process counter must not erase its finite siblings.

    `_process_metrics` calls ``float(value)`` on every counter that is not a
    typed `tpen.process_resources.ResourceUnavailable`, so a non-finite reading
    reaches `tpen.logging.jsonl.JSONL` and aborts the whole process record --
    the same failure mode the allocator projection already guards against.
    """

    result = ProcessResourceResult(
        user_cpu_seconds=invalid_user_cpu_seconds,
        system_cpu_seconds=2.0,
        read_block_operations=3.0,
        write_block_operations=4.0,
        voluntary_context_switches=5.0,
        involuntary_context_switches=6.0,
        peak_rss_mb=7.0,
    )
    context = _strict_json_terminal_context(tmp_path)
    callback = ResourceUsage(process_probe=_FixedProbe(result))

    _deliver(callback, context, RunStarted())
    # No exception may escape the callback at the terminal boundary.
    _deliver(callback, context, RunCompleted())

    terminal_lines = _terminal_records(tmp_path)
    process_lines = [line for line in terminal_lines if line["namespace"] == "process"]
    assert len(process_lines) == 1
    process = process_lines[0]["metrics"]

    # The offending counter degrades to the typed flag this module already
    # uses for unavailable readings.
    assert process["process_user_cpu_seconds_unavailable"] is True
    assert "process_user_cpu_seconds" not in process

    # Every finite sibling is still reported, unchanged.
    assert process["process_system_cpu_seconds"] == 2.0
    assert process["process_read_block_operations"] == 3.0
    assert process["process_write_block_operations"] == 4.0
    assert process["process_voluntary_context_switches"] == 5.0
    assert process["process_involuntary_context_switches"] == 6.0

    # The finite runtime peak, sourced from the same probe result, survives.
    runtime_lines = [line for line in terminal_lines if line["namespace"] == "runtime"]
    assert len(runtime_lines) == 1
    assert runtime_lines[0]["metrics"]["peak_memory_mb"] == 7.0


def test_resource_usage_isolates_nonfinite_default_peak_rss_at_terminal_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Writer-authored: the third `peak_memory_mb` arm, `_default_peak_rss_mb`.

    Neither adopted test above reaches this arm: it fires only when there is
    no `peak_rss_mb_reader` override AND no process-probe baseline yet (`_log_
    peaks` called without a preceding `RunStarted` for this callback
    instance). Demonstrated red-before-fix since this test carries no
    independent red arm of its own.
    """

    monkeypatch.setattr(resource_usage_module, "_default_peak_rss_mb", lambda: math.nan)
    context = _strict_json_terminal_context(tmp_path)
    allocator = _FakeAllocatorProbe(
        AllocatorUsage(
            identity=AcceleratorIdentity(AcceleratorKind.CUDA, 1, "GPU-1"),
            allocated_mb=3.0,
            reserved_mb=8.0,
            device_count=2,
        )
    )
    # Passed directly rather than context-derived, so it is set regardless of
    # RunStarted -- this is what forces a non-empty `runtime` record even
    # though the `peak_memory_mb` arm below contributes nothing.
    callback = ResourceUsage(allocator_probe=allocator)

    # No RunStarted: `_process_baseline` stays None, forcing the
    # `_default_peak_rss_mb` fallback arm inside `_log_peaks`.
    _deliver(callback, context, RunCompleted())

    terminal_lines = _terminal_records(tmp_path)
    runtime_lines = [line for line in terminal_lines if line["namespace"] == "runtime"]
    assert len(runtime_lines) == 1
    assert "peak_memory_mb" not in runtime_lines[0]["metrics"]
    assert runtime_lines[0]["metrics"]["accelerator_max_memory_allocated_mb"] == 3.0
