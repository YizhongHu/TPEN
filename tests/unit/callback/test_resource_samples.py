"""Unit tests for the O3 ``ResourceSamples`` callback, RUSAGE, and allocator ownership."""

from __future__ import annotations

import ast
from collections.abc import Iterator
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import types
from typing import Any

from omegaconf import OmegaConf
import pytest

from tests.helpers.run_context import make_run_context
import tpen.accelerator as accelerator_mod
from tpen.accelerator import (
    AcceleratorIdentity,
    AcceleratorKind,
    AllocatorCurrentUsage,
    AllocatorUnavailable,
    AllocatorUsage,
)
from tpen.artifacts import (
    ArtifactDestination,
    ArtifactManager,
    ObservabilityBundleWriter,
    RunArtifactSink,
    RunContext,
    RunResult,
)
from tpen.callback import (
    AllocatorSource,
    LinuxMemorySource,
    ManagedLifecycleError,
    RUsageSource,
    ResourceOwnershipConflictError,
    ResourceSamples,
    ResourceUsage,
    _KNOWN_NON_MANAGED_CALLBACK_TARGETS,
    claims_conflict,
    is_managed_callback,
)
from tpen.callback.resource_samples import (
    ALLOCATOR_CURRENT_QUANTITY_IDS,
    ALLOCATOR_PEAK_QUANTITY_IDS,
    ALLOCATOR_SOURCE_ID,
    LINUX_MEMORY_QUANTITY_IDS,
    LINUX_MEMORY_SOURCE_ID,
    RUSAGE_QUANTITY_IDS,
    RUSAGE_SOURCE_ID,
)
from tpen.checkpoint import TrainResume, save_checkpoint
from tpen.checkpoint.hashing import resolved_config_identity
from tpen.checkpoint.operational_slots import OPERATIONAL_SLOT_MANIFEST
from tpen.checkpoint.profiles import NATIVE_RECOVERY_PROFILE, check_profile
from tpen.distributed import ExecutionTopology
from tpen.inspection import inspect_observability_bundle, inspect_run_observability
from tpen.observability_artifacts import ObservabilityPathError
from tpen.process_resources import (
    LinuxMemoryUnavailable,
    LinuxProcessGeneration,
    LinuxProcessMemoryProbe,
    LinuxProcessMemoryReading,
    ProcessResourceBaseline,
    ResourceUnavailable,
)
import tpen.run as run_mod
from tpen.run import _ManagedLifecycleRegistry, open_run_context, run_from_config
from tpen.run_events import RunCompleted, RunStarted
from tpen.runner import Runner


_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _isolated_managed_registry(monkeypatch: pytest.MonkeyPatch) -> _ManagedLifecycleRegistry:
    """Isolate process-wide managed leases between unit tests."""

    fresh = _ManagedLifecycleRegistry()
    monkeypatch.setattr(run_mod, "_MANAGED_LIFECYCLE_REGISTRY", fresh)
    return fresh


def _base_cfg(
    tmp_path: Path,
    *,
    device: str = "cpu",
    callbacks: list[Any] | None = None,
    runner: dict[str, Any] | None = None,
) -> Any:
    return OmegaConf.create(
        {
            "experiment": {"name": "o3_test", "sector": "unit"},
            "run": {
                "root": str(tmp_path / "outputs"),
                "layout": "flat",
                "run_id": "o3-run",
                "timezone": "UTC",
            },
            "runtime": {"device": device, "dtype": "float64"},
            "callbacks": [] if callbacks is None else callbacks,
            "loggers": [],
            "runner": (
                {"_target_": f"{__name__}._NoopRunner"}
                if runner is None
                else runner
            ),
        }
    )


class _NoopRunner(Runner):
    def run(self, context: RunContext) -> RunResult:
        del context
        return RunResult(status="completed")


class _CheckpointRunner(Runner):
    """Runner emitting several lightweight checkpoints to drive cooperative sampling."""

    def __init__(self, steps: int = 3, sleep_seconds: float = 0.0) -> None:
        self.steps = steps
        self.sleep_seconds = sleep_seconds

    def run(self, context: RunContext) -> RunResult:
        for _ in range(self.steps):
            if self.sleep_seconds > 0.0:
                time.sleep(self.sleep_seconds)
            if context._checkpoint_hook is not None:
                context._checkpoint_hook()
        return RunResult(status="completed")


class _RaisingRunner(Runner):
    def run(self, context: RunContext) -> RunResult:
        del context
        raise ValueError("runner failed deliberately")


class _KeyboardInterruptRunner(Runner):
    def run(self, context: RunContext) -> RunResult:
        del context
        raise KeyboardInterrupt("operator interrupt")


class _FakeRUsageProbe:
    """Deterministic sequence-backed RUSAGE probe for unit tests."""

    def __init__(self, readings: list[ProcessResourceBaseline | Exception]) -> None:
        self._readings = list(readings)
        self._index = 0
        self.thread_ids: list[int] = []
        self.calls = 0

    def read(self) -> ProcessResourceBaseline:
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        idx = min(self._index, len(self._readings) - 1)
        self._index += 1
        item = self._readings[idx]
        if isinstance(item, Exception):
            raise item
        return item


class _BlockingRUsageProbe:
    """RUSAGE probe that blocks on the background thread to exercise close timeouts."""

    def __init__(self) -> None:
        self.first_read_done = threading.Event()
        self.blocked_in_second = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def read(self) -> ProcessResourceBaseline:
        self.calls += 1
        if self.calls == 1:
            self.first_read_done.set()
            return ProcessResourceBaseline(
                user_cpu_seconds=0.5,
                system_cpu_seconds=0.1,
                read_block_operations=0,
                write_block_operations=0,
                voluntary_context_switches=1,
                involuntary_context_switches=0,
                peak_rss_mb=32.0,
            )
        self.blocked_in_second.set()
        self.release.wait(timeout=5.0)
        return ProcessResourceBaseline(
            user_cpu_seconds=1.0,
            system_cpu_seconds=0.2,
            read_block_operations=0,
            write_block_operations=0,
            voluntary_context_switches=2,
            involuntary_context_switches=0,
            peak_rss_mb=32.0,
        )


class _FakeCurrentAllocatorProbe:
    """Cooperative current-allocator test double."""

    def __init__(
        self,
        readings: list[AllocatorCurrentUsage],
        *,
        identity: AcceleratorIdentity | None = None,
    ) -> None:
        self._readings = list(readings)
        self._index = 0
        self._id = readings[0].identity if identity is None else identity
        self.read_calls = 0
        self.thread_ids: list[int] = []

    def identity(self) -> AcceleratorIdentity:
        return self._id

    def read(self) -> AllocatorCurrentUsage:
        self.read_calls += 1
        self.thread_ids.append(threading.get_ident())
        idx = min(self._index, len(self._readings) - 1)
        self._index += 1
        return self._readings[idx]


class _FakePeakAllocatorProbe:
    """Cooperative peak-allocator test double with reset tracking."""

    def __init__(
        self,
        readings: list[AllocatorUsage],
        *,
        reset_result: AcceleratorIdentity | AllocatorUnavailable | None = None,
    ) -> None:
        self._readings = list(readings)
        self._index = 0
        self._id = readings[0].identity
        self._reset_result = self._id if reset_result is None else reset_result
        self.reset_calls = 0
        self.read_calls = 0
        self.thread_ids: list[int] = []

    def identity(self) -> AcceleratorIdentity:
        return self._id

    def reset(self) -> AcceleratorIdentity | AllocatorUnavailable:
        self.reset_calls += 1
        self.thread_ids.append(threading.get_ident())
        return self._reset_result

    def read(self) -> AllocatorUsage:
        self.read_calls += 1
        self.thread_ids.append(threading.get_ident())
        idx = min(self._index, len(self._readings) - 1)
        self._index += 1
        return self._readings[idx]


class _SteppingClock:
    """Deterministic nanosecond monotonic clock advancing by a fixed step per call."""

    def __init__(self, start_ns: int = 1_000_000_000_000, step_ns: int = 1_000_000) -> None:
        self._now = start_ns
        self._step = step_ns
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            val = self._now
            self._now += self._step
            return val

    def advance(self, delta_ns: int) -> None:
        with self._lock:
            self._now += delta_ns


# ---------------------------------------------------------------------------
# Import hygiene, registration, and constructor validation
# ---------------------------------------------------------------------------


def test_resource_samples_module_is_ast_clean_and_torch_free_in_subprocess() -> None:
    """Import and construction must remain strictly Torch-, experiments-, and executor-free."""

    module_path = _REPO_ROOT / "tpen" / "callback" / "resource_samples.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    forbidden_roots = {"torch", "experiments", "tpen_executor"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden_roots
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".")[0] not in forbidden_roots

    code = (
        "import sys, tpen.callback, tpen.callback.resource_samples; "
        "cb = tpen.callback.ResourceSamples("
        "rusage=tpen.callback.RUsageSource(), "
        "allocator=tpen.callback.AllocatorSource(enabled=True, device='cuda:0')); "
        "assert 'torch' not in sys.modules; "
        "assert 'experiments' not in sys.modules; "
        "assert 'tpen_executor' not in sys.modules; "
        "print('TORCH_FREE_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "TORCH_FREE_OK" in proc.stdout


def test_resource_samples_registration_in_managed_and_operational_registries() -> None:
    """ResourceSamples is managed, sink-free admitted, and excluded from non-managed targets."""

    cb = ResourceSamples()
    assert is_managed_callback(cb)
    assert "tpen.callback.ResourceSamples" not in _KNOWN_NON_MANAGED_CALLBACK_TARGETS
    assert (
        "tpen.callback.resource_samples.ResourceSamples"
        not in _KNOWN_NON_MANAGED_CALLBACK_TARGETS
    )
    slot = OPERATIONAL_SLOT_MANIFEST["tpen.callback.resource_samples.ResourceSamples"]
    assert slot.routed is True
    assert slot.sink_free is True


@pytest.mark.parametrize(
    "bad_mode",
    ["helper", "subprocess", "helper_subprocess", "external_command", ""],
)
def test_rusage_source_rejects_helper_or_subprocess_execution_mode(bad_mode: str) -> None:
    """RUSAGE_SELF must never run in a helper subprocess and pretend to be the app."""

    with pytest.raises(ValueError, match="RUSAGE_SELF"):
        RUsageSource(execution_mode=bad_mode)


def test_rusage_source_rejects_foreign_pid() -> None:
    """RUSAGE_SELF rejects targeting any PID other than the current process."""

    foreign_pid = os.getpid() + 1
    with pytest.raises(ValueError, match="foreign PID"):
        RUsageSource(pid=foreign_pid)

    # Current PID is accepted explicitly.
    same = RUsageSource(pid=os.getpid())
    assert same.pid == os.getpid()


@pytest.mark.parametrize(
    "bad_mode",
    ["in_process_thread", "helper", "subprocess", "helper_subprocess"],
)
def test_allocator_source_rejects_non_cooperative_execution_mode(bad_mode: str) -> None:
    """Torch allocator probes must run strictly cooperatively on the owning thread."""

    with pytest.raises(ValueError, match="cooperatively"):
        AllocatorSource(execution_mode=bad_mode)


def test_allocator_source_and_resource_samples_validate_numeric_and_semantic_bounds() -> None:
    """Malformed bounds, booleans, and inconsistent peak/reset flags fail eagerly."""

    with pytest.raises(ValueError, match="include_peak=True"):
        AllocatorSource(include_current=True, include_peak=False, reset_peaks=True)
    with pytest.raises(ValueError, match="at least one"):
        AllocatorSource(enabled=True, include_current=False, include_peak=False, reset_peaks=False)

    for bad_interval in (0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            ResourceSamples(interval_seconds=bad_interval)
        with pytest.raises(ValueError):
            ResourceSamples(close_timeout_seconds=bad_interval)
    with pytest.raises(TypeError):
        ResourceSamples(interval_seconds=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ResourceSamples(max_pending_envelopes=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ResourceSamples(max_pending_envelopes=0)
    with pytest.raises(ValueError, match="at least one enabled source"):
        ResourceSamples(
            enabled=True,
            rusage=RUsageSource(enabled=False),
            allocator=AllocatorSource(enabled=False, reset_peaks=False),
        )
    with pytest.raises(ValueError, match="distinct source_id"):
        ResourceSamples(
            rusage=RUsageSource(enabled=True, source_id="dup"),
            allocator=AllocatorSource(enabled=True, source_id="dup"),
        )
    with pytest.raises(Exception, match="status"):
        ResourceSamples(observer_id="status")


@pytest.mark.parametrize(
    "selector_key",
    [
        "triggers",
        "typed_groups",
        "selectors",
        "subscriptions",
        "operation",
        "operation_selector",
        "event_selector",
        "event_type",
        "operation_type",
    ],
)
def test_free_form_selector_fields_refused_in_python_and_yaml(
    tmp_path: Path, selector_key: str
) -> None:
    """Free-form event/type selector fields are forbidden both in Python and YAML."""

    with pytest.raises(ValueError, match="forbidden"):
        ResourceSamples(**{selector_key: ["step"]})

    cfg = _base_cfg(
        tmp_path / selector_key,
        callbacks=[
            {
                "_target_": "tpen.callback.ResourceSamples",
                selector_key: ["TrainingIteration"],
            }
        ],
    )
    with pytest.raises(ValueError, match="forbidden"):
        run_from_config(cfg, raise_exceptions=True)
    assert not (tmp_path / selector_key / "outputs").exists()


# ---------------------------------------------------------------------------
# Disabled configuration emits nothing
# ---------------------------------------------------------------------------


def test_disabled_resource_samples_creates_no_bundle_threads_or_probes(
    tmp_path: Path,
) -> None:
    """enabled=False yields None, claims nothing, constructs no probes, and emits zero files."""

    factory_calls: list[str] = []

    def fail_factory(device: str) -> object:
        factory_calls.append(device)
        raise AssertionError("probe factory must not run when disabled")

    cb = ResourceSamples(
        enabled=False,
        rusage=RUsageSource(enabled=True),
        allocator=AllocatorSource(
            enabled=True,
            device="cuda:0",
            reset_peaks=True,
            current_probe_factory=fail_factory,
            peak_probe_factory=fail_factory,
        ),
    )
    assert cb.ownership_claims(cfg=_base_cfg(tmp_path, device="cuda:0"), run_dir=tmp_path) == []

    # Coexists freely with legacy ResourceUsage when disabled.
    cfg = _base_cfg(
        tmp_path,
        device="cuda:0",
        callbacks=[{"_target_": "tpen.callback.ResourceUsage"}],
    )
    exit_code = run_from_config(cfg, raise_exceptions=True, runtime_callbacks=[cb])
    assert exit_code == 0
    assert factory_calls == []
    run_dir = tmp_path / "outputs" / "o3-run"
    assert not (run_dir / "observability").exists()
    inspection = inspect_run_observability(run_dir)
    assert inspection.status == "absent"
    assert inspection.invocations == ()


# ---------------------------------------------------------------------------
# RUSAGE semantics: zero vs unavailable vs error, counter vs high-water
# ---------------------------------------------------------------------------


def test_rusage_distinguishes_zero_unavailable_error_counters_and_high_water(
    tmp_path: Path,
) -> None:
    """RUSAGE records preserve zero, unavailable, non-finite errors, and counter/high-water semantics."""

    probe = _FakeRUsageProbe(
        [
            # Reading 0: legitimate zeros + finite lifetime high-water RSS (512 MiB)
            ProcessResourceBaseline(
                user_cpu_seconds=0.0,
                system_cpu_seconds=0.0,
                read_block_operations=0,
                write_block_operations=0,
                voluntary_context_switches=0,
                involuntary_context_switches=0,
                peak_rss_mb=512.0,
            ),
            # Reading 1: mixed ok, unavailable, and non-finite error
            ProcessResourceBaseline(
                user_cpu_seconds=10.0,
                system_cpu_seconds=2.5,
                read_block_operations=ResourceUnavailable("OSError: blocked"),
                write_block_operations=4,
                voluntary_context_switches=12,
                involuntary_context_switches=math.nan,
                peak_rss_mb=512.0,
            ),
        ]
    )
    clock = _SteppingClock(start_ns=1_000_000_000_000, step_ns=1_000_000_000)
    cb = ResourceSamples(
        interval_seconds=60.0,
        rusage=RUsageSource(enabled=True, probe=probe),
        monotonic_ns=clock,
    )

    with open_run_context(_base_cfg(tmp_path), runtime_callbacks=[cb]) as ctx:
        run_dir = ctx.run_dir

    run_obs = inspect_run_observability(run_dir)
    assert run_obs.status == "present"
    assert len(run_obs.invocations) == 1
    bundle = run_obs.invocations[0]
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_finalization == "confirmed"
    # Partial because reading 1 had unavailable + error fields.
    assert bundle.derived_data_completeness == "partial"

    descriptors = bundle.descriptors
    assert descriptors is not None
    assert set(descriptors.quantities.keys()) == set(RUSAGE_QUANTITY_IDS)
    assert descriptors.quantities["process.rusage.user_cpu_seconds"].statistic == "counter"
    assert descriptors.quantities["process.rusage.user_cpu_seconds"].window == "process_lifetime"
    rss_desc = descriptors.quantities["process.rusage.max_rss_bytes"]
    assert rss_desc.statistic == "high_water"
    assert rss_desc.window == "process_lifetime"
    assert rss_desc.unit == "B"
    assert "process_lifetime_high_water_not_current_rss" in rss_desc.quality

    samples = bundle.samples["resource-samples"]
    assert len(samples) == 2
    first, second = samples[0], samples[1]
    assert first.sequence == 0
    assert second.sequence == 1

    # Zero is a legitimate ok reading, never unavailable.
    assert first.readings["process.rusage.user_cpu_seconds"].status == "ok"
    assert first.readings["process.rusage.user_cpu_seconds"].value == 0.0
    assert first.readings["process.rusage.read_block_operations"].status == "ok"
    assert first.readings["process.rusage.read_block_operations"].value == 0
    assert first.readings["process.rusage.max_rss_bytes"].value == 512 * 1024 * 1024

    # Unavailable has no numeric value; NaN becomes typed error.
    unavail = second.readings["process.rusage.read_block_operations"]
    assert unavail.status == "unavailable"
    assert unavail.value is None
    assert unavail.reason_code == "rusage_unavailable"
    assert "value" not in unavail.to_dict()

    err = second.readings["process.rusage.involuntary_context_switches"]
    assert err.status == "error"
    assert err.value is None
    assert err.reason_code == "non_finite_reading"

    # Counter reduction computes valid delta/rate; high-water reports observed_max.
    user_summary = bundle.summary.quantity(RUSAGE_SOURCE_ID, "process.rusage.user_cpu_seconds")
    assert user_summary is not None
    assert user_summary.valid_interval_count == 1
    assert user_summary.valid_delta_total == pytest.approx(10.0)
    assert user_summary.valid_rate_per_second is not None and user_summary.valid_rate_per_second > 0

    rss_summary = bundle.summary.quantity(RUSAGE_SOURCE_ID, "process.rusage.max_rss_bytes")
    assert rss_summary is not None
    assert rss_summary.observed_max == 512 * 1024 * 1024
    assert rss_summary.sampled_max is None  # High-water is not reported as gauge sampled_max.

    status = bundle.statuses["resource-samples"]
    assert status.application_outcome == "not_supplied"
    assert status.invocation_outcome == "completed"
    assert status.counts == {"attempted": 2, "written": 2, "dropped": 0, "failed": 0}
    assert status.source_status[RUSAGE_SOURCE_ID] == "partial"


def test_rusage_periodic_background_thread_captures_multiple_samples(tmp_path: Path) -> None:
    """Periodic RUSAGE sampling runs on a background thread independently of events."""

    main_tid = threading.get_ident()
    probe = _FakeRUsageProbe(
        [
            ProcessResourceBaseline(0.1 * i, 0.05 * i, i, i, i, i, 64.0)
            for i in range(1, 20)
        ]
    )
    cb = ResourceSamples(
        interval_seconds=0.02,
        max_pending_envelopes=64,
        rusage=RUsageSource(enabled=True, probe=probe),
    )
    cfg = _base_cfg(
        tmp_path,
        runner={
            "_target_": f"{__name__}._CheckpointRunner",
            "steps": 1,
            "sleep_seconds": 0.15,
        },
    )
    exit_code = run_from_config(cfg, raise_exceptions=True, runtime_callbacks=[cb])
    assert exit_code == 0

    bundle = inspect_run_observability(tmp_path / "outputs" / "o3-run").invocations[0]
    status = bundle.statuses["resource-samples"]
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "complete", (
        f"expected complete bundle, got {bundle.derived_data_completeness!r} "
        f"(reason={bundle.derived_reason!r}, counts={status.counts!r}, "
        f"truncation_reasons={status.truncation_reasons!r}, "
        f"diagnostics={bundle.diagnostics!r})"
    )
    samples = bundle.samples["resource-samples"]
    assert len(samples) >= 3
    # At least one sample came from the background RUSAGE thread, not the main thread.
    assert any(tid != main_tid for tid in probe.thread_ids)
    assert status.application_outcome == "completed"
    assert status.invocation_outcome == "completed"
    assert status.helper_cleanup["state"] == "confirmed_stopped"


# ---------------------------------------------------------------------------
# Allocator semantics, deferred activation, reset authority & ownership conflicts
# ---------------------------------------------------------------------------


def test_allocator_current_vs_peak_and_deferred_cooperative_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allocator probes activate lazily inside run_scope, reset once, and run cooperatively."""

    main_tid = threading.get_ident()
    identity = AcceleratorIdentity(AcceleratorKind.CUDA, 0, "GPU-AAA")
    created_devices: list[str] = []

    current_probe = _FakeCurrentAllocatorProbe(
        [
            AllocatorCurrentUsage(identity, 1024, 4096, 1),
            AllocatorCurrentUsage(identity, 8192, 16384, 1),
            AllocatorCurrentUsage(identity, 2048, 16384, 1),
        ]
    )
    peak_probe = _FakePeakAllocatorProbe(
        [
            AllocatorUsage(identity, 1.0, 2.0, 1),
            AllocatorUsage(identity, 4.0, 8.0, 1),
            AllocatorUsage(identity, 4.0, 8.0, 1),
        ]
    )

    def make_current(dev: str) -> _FakeCurrentAllocatorProbe:
        created_devices.append(f"current:{dev}")
        return current_probe

    def make_peak(dev: str) -> _FakePeakAllocatorProbe:
        created_devices.append(f"peak:{dev}")
        return peak_probe

    # 0.1s per monotonic_ns() call with 0.5s interval exercises cooperative
    # throttling cleanly (multiple checkpoints skipped between due samples)
    # while staying well below max_pending_envelopes=32 even on 1-CPU nodes.
    clock = _SteppingClock(start_ns=1_000_000_000_000, step_ns=100_000_000)
    cb = ResourceSamples(
        interval_seconds=0.5,
        max_pending_envelopes=32,
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(
            enabled=True,
            include_current=True,
            include_peak=True,
            reset_peaks=True,
            current_probe_factory=make_current,
            peak_probe_factory=make_peak,
        ),
        monotonic_ns=clock,
    )
    # Construction must NOT invoke probe factories.
    assert created_devices == []

    checkpoint_calls = 0
    original_run_scope = cb.run_scope

    @contextlib.contextmanager
    def counting_run_scope(ctx: RunContext) -> Iterator[Any]:
        with original_run_scope(ctx) as handle:
            assert handle is not None

            class _CountingHandle:
                def checkpoint(self) -> None:
                    nonlocal checkpoint_calls
                    checkpoint_calls += 1
                    handle.checkpoint()

                def request_stop(self) -> None:
                    handle.request_stop()

            yield _CountingHandle()

    monkeypatch.setattr(cb, "run_scope", counting_run_scope)

    cfg = _base_cfg(
        tmp_path,
        device="cuda:0",
        runner={"_target_": f"{__name__}._CheckpointRunner", "steps": 6},
    )
    topology = ExecutionTopology.single_process(device="cuda:0", device_identity=identity)
    exit_code = run_from_config(
        cfg,
        raise_exceptions=True,
        topology=topology,
        runtime_callbacks=[cb],
    )
    assert exit_code == 0

    assert created_devices == ["current:cuda:0", "peak:cuda:0"]
    assert peak_probe.reset_calls == 1
    assert checkpoint_calls >= 10
    # Cooperative throttling must fire multiple samples while skipping non-due checkpoints.
    assert 3 <= current_probe.read_calls < checkpoint_calls
    assert peak_probe.read_calls == current_probe.read_calls
    # Every allocator call happened strictly on the owning main thread.
    assert set(current_probe.thread_ids) == {main_tid}
    assert set(peak_probe.thread_ids) == {main_tid}

    bundle = inspect_run_observability(tmp_path / "outputs" / "o3-run").invocations[0]
    status = bundle.statuses["resource-samples"]
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "complete", (
        f"expected complete bundle, got {bundle.derived_data_completeness!r} "
        f"(reason={bundle.derived_reason!r}, counts={status.counts!r}, "
        f"truncation_reasons={status.truncation_reasons!r}, "
        f"diagnostics={bundle.diagnostics!r})"
    )

    descriptors = bundle.descriptors
    assert descriptors is not None
    assert set(descriptors.quantities.keys()) == set(
        ALLOCATOR_CURRENT_QUANTITY_IDS + ALLOCATOR_PEAK_QUANTITY_IDS
    )
    cur_desc = descriptors.quantities["allocator.current.allocated_bytes"]
    peak_desc = descriptors.quantities["allocator.peak.allocated_bytes"]
    assert cur_desc.statistic == "gauge"
    assert cur_desc.window == "read_interval"
    assert cur_desc.subject_kind == "process_allocator"
    assert peak_desc.statistic == "high_water"
    assert peak_desc.window == "allocator_reset_window"
    assert peak_desc.subject_kind == "process_allocator"

    cur_summary = bundle.summary.quantity(
        ALLOCATOR_SOURCE_ID, "allocator.current.allocated_bytes"
    )
    peak_summary = bundle.summary.quantity(
        ALLOCATOR_SOURCE_ID, "allocator.peak.allocated_bytes"
    )
    assert cur_summary is not None
    assert cur_summary.sampled_min == 1024
    assert cur_summary.sampled_max == 8192
    assert cur_summary.sampled_max_is_lower_bound is True

    assert peak_summary is not None
    assert peak_summary.observed_max == 4 * 1024 * 1024
    assert peak_summary.sampled_max is None

    assert status.helper_cleanup["state"] == "not_started"
    assert status.counts["attempted"] == status.counts["written"]


@pytest.mark.parametrize(
    ("run_device", "samples_device"),
    [("cuda:0", None), ("cuda", "cuda:0"), ("cuda:0", "cuda")],
)
def test_allocator_reset_ownership_conflicts_with_legacy_resource_usage(
    tmp_path: Path, run_device: str, samples_device: str | None
) -> None:
    """Preflight refuses coexistence of resetting ResourceSamples and legacy ResourceUsage."""

    cb = ResourceSamples(
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(
            enabled=True,
            device=samples_device,
            reset_peaks=True,
        ),
    )
    cfg = _base_cfg(
        tmp_path,
        device=run_device,
        callbacks=[{"_target_": "tpen.callback.ResourceUsage"}],
    )
    with pytest.raises(ResourceOwnershipConflictError):
        run_from_config(cfg, raise_exceptions=True, runtime_callbacks=[cb])
    assert not (tmp_path / "outputs").exists()


def test_allocator_without_reset_coexists_with_resource_usage_and_never_resets(
    tmp_path: Path,
) -> None:
    """reset_peaks=False emits no allocator_reset claim, never resets peaks, and labels window unknown."""

    identity = AcceleratorIdentity(AcceleratorKind.CUDA, 0, "GPU-RO")
    current_probe = _FakeCurrentAllocatorProbe(
        [AllocatorCurrentUsage(identity, 4096, 8192, 1)]
    )
    peak_probe = _FakePeakAllocatorProbe([AllocatorUsage(identity, 2.0, 4.0, 1)])

    cb = ResourceSamples(
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(
            enabled=True,
            device="cuda:0",
            include_current=True,
            include_peak=True,
            reset_peaks=False,
            current_probe_factory=lambda _d: current_probe,
            peak_probe_factory=lambda _d: peak_probe,
        ),
    )
    claims = cb.ownership_claims(cfg=_base_cfg(tmp_path, device="cuda:0"), run_dir=tmp_path)
    assert all(c.kind != "allocator_reset" for c in claims)

    legacy = ResourceUsage(
        allocator_probe=_FakePeakAllocatorProbe([AllocatorUsage(identity, 2.0, 4.0, 1)])
    )
    # Supply explicit topology matching the fake device identity so CPU-only
    # nodes without CUDA do not mismatch uuid=None against uuid='GPU-RO'.
    topology = ExecutionTopology.single_process(device="cuda:0", device_identity=identity)
    exit_code = run_from_config(
        _base_cfg(tmp_path, device="cuda:0"),
        raise_exceptions=True,
        topology=topology,
        runtime_callbacks=[cb, legacy],
    )
    assert exit_code == 0
    assert peak_probe.reset_calls == 0

    bundle = inspect_run_observability(tmp_path / "outputs" / "o3-run").invocations[0]
    status = bundle.statuses["resource-samples"]
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "complete", (
        f"expected complete bundle, got {bundle.derived_data_completeness!r} "
        f"(reason={bundle.derived_reason!r}, counts={status.counts!r}, "
        f"truncation_reasons={status.truncation_reasons!r}, "
        f"diagnostics={bundle.diagnostics!r})"
    )
    assert bundle.descriptors is not None
    peak_desc = bundle.descriptors.quantities["allocator.peak.allocated_bytes"]
    assert peak_desc.window == "unknown"
    assert "unreset_peak_window_unknown" in peak_desc.quality
    sample = bundle.samples["resource-samples"][0]
    assert sample.subject["reset_epoch_owner"] == "none"


def test_allocator_reset_failure_blanks_only_peaks_while_current_stays_ok(
    tmp_path: Path,
) -> None:
    """Failed peak reset emits reset_failed for peaks while current readings remain ok."""

    identity = AcceleratorIdentity(AcceleratorKind.CUDA, 0, "GPU-RF")
    current_probe = _FakeCurrentAllocatorProbe(
        [AllocatorCurrentUsage(identity, 65536, 131072, 1)]
    )
    peak_probe = _FakePeakAllocatorProbe(
        [AllocatorUsage(identity, 999.0, 999.0, 1)],
        reset_result=AllocatorUnavailable("driver refused reset_peak_memory_stats"),
    )

    cb = ResourceSamples(
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(
            enabled=True,
            device="cuda:0",
            include_current=True,
            include_peak=True,
            reset_peaks=True,
            current_probe_factory=lambda _d: current_probe,
            peak_probe_factory=lambda _d: peak_probe,
        ),
    )

    exit_code = run_from_config(
        _base_cfg(tmp_path, device="cuda:0"),
        raise_exceptions=True,
        runtime_callbacks=[cb],
    )
    assert exit_code == 0
    assert peak_probe.reset_calls == 1
    assert peak_probe.read_calls == 0  # Stale pre-reset peaks must never be published.

    bundle = inspect_run_observability(tmp_path / "outputs" / "o3-run").invocations[0]
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "partial"
    sample = bundle.samples["resource-samples"][0]
    assert sample.readings["allocator.current.allocated_bytes"].status == "ok"
    assert sample.readings["allocator.current.allocated_bytes"].value == 65536
    assert sample.readings["allocator.current.reserved_bytes"].value == 131072
    peak_alloc = sample.readings["allocator.peak.allocated_bytes"]
    assert peak_alloc.status == "unavailable"
    assert peak_alloc.reason_code == "reset_failed"
    assert peak_alloc.value is None
    assert bundle.statuses["resource-samples"].source_status[ALLOCATOR_SOURCE_ID] == "partial"


# ---------------------------------------------------------------------------
# Admitted O2 routed writing & destination conflict protection
# ---------------------------------------------------------------------------


def test_configured_yaml_resource_samples_writes_only_under_routed_artifact_destination(
    tmp_path: Path,
) -> None:
    """Routed run_from_config admits ResourceSamples and routes all bundle bytes physically."""

    physical_root = tmp_path / "physical_leaf"
    cfg = _base_cfg(
        tmp_path,
        device="cpu",
        callbacks=[
            {
                "_target_": "tpen.callback.ResourceSamples",
                "observer_id": "routed-obs",
                "interval_seconds": 60.0,
                "sink": {
                    "_target_": "tpen.artifacts.RunArtifactSink",
                    "relative_dir": "observability",
                },
            }
        ],
    )
    exit_code = run_from_config(
        cfg,
        raise_exceptions=True,
        artifact_destination=ArtifactDestination(physical_root),
    )
    assert exit_code == 0
    # Logical run root must remain untouched.
    assert not (tmp_path / "outputs").exists()

    run_obs = inspect_run_observability(physical_root)
    assert run_obs.status == "present"
    assert len(run_obs.invocations) == 1
    bundle = run_obs.invocations[0]
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_finalization == "confirmed"
    assert bundle.manifest.run_id == "o3-run"
    assert "routed-obs" in bundle.samples
    assert len(bundle.samples["routed-obs"]) >= 2
    assert bundle.statuses["routed-obs"].application_outcome == "completed"


def test_duplicate_sink_bundle_refused_while_distinct_relative_dirs_coexist(
    tmp_path: Path,
) -> None:
    """Two ResourceSamples targeting the same bundle conflict; distinct relative_dirs succeed."""

    cb_a = ResourceSamples(observer_id="obs-a", sink=RunArtifactSink(relative_dir="observability"))
    cb_b = ResourceSamples(observer_id="obs-b", sink=RunArtifactSink(relative_dir="observability"))
    with pytest.raises(ResourceOwnershipConflictError):
        run_from_config(
            _base_cfg(tmp_path / "conflict"),
            raise_exceptions=True,
            runtime_callbacks=[cb_a, cb_b],
        )
    assert not (tmp_path / "conflict" / "outputs").exists()

    cb_1 = ResourceSamples(
        observer_id="obs-1",
        interval_seconds=60.0,
        sink=RunArtifactSink(relative_dir="observability_a"),
    )
    cb_2 = ResourceSamples(
        observer_id="obs-2",
        interval_seconds=60.0,
        sink=RunArtifactSink(relative_dir="observability_b"),
    )
    exit_code = run_from_config(
        _base_cfg(tmp_path / "distinct"),
        raise_exceptions=True,
        runtime_callbacks=[cb_1, cb_2],
    )
    assert exit_code == 0
    run_dir = tmp_path / "distinct" / "outputs" / "o3-run"
    assert inspect_run_observability(run_dir, relative_dir="observability_a").status == "present"
    assert inspect_run_observability(run_dir, relative_dir="observability_b").status == "present"


# ---------------------------------------------------------------------------
# Queue saturation, byte cap, writer errors, unsafe timeout, exception safety
# ---------------------------------------------------------------------------


def test_queue_saturation_drops_nonblockingly_and_reconciles_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the writer stalls, bounded queue drops new samples nonblockingly and records gaps."""

    writer_entered = threading.Event()
    release_writer = threading.Event()
    original_append = ObservabilityBundleWriter.append_sample
    call_count = 0

    def slow_first_append(self: ObservabilityBundleWriter, sample: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            writer_entered.set()
            release_writer.wait(timeout=2.0)
        return original_append(self, sample)

    monkeypatch.setattr(ObservabilityBundleWriter, "append_sample", slow_first_append)

    identity = AcceleratorIdentity(AcceleratorKind.CUDA, 0, "GPU-Q")
    current_probe = _FakeCurrentAllocatorProbe(
        [AllocatorCurrentUsage(identity, 1024 * (i + 1), 4096 * (i + 1), 1) for i in range(10)]
    )
    clock = _SteppingClock(start_ns=1_000_000_000_000, step_ns=1_000_000_000)
    cb = ResourceSamples(
        interval_seconds=0.5,
        max_pending_envelopes=1,
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(
            enabled=True,
            include_current=True,
            include_peak=False,
            reset_peaks=False,
            current_probe_factory=lambda _d: current_probe,
        ),
        monotonic_ns=clock,
    )

    with open_run_context(_base_cfg(tmp_path), runtime_callbacks=[cb]) as ctx:
        assert writer_entered.wait(timeout=2.0)
        # Produce several rapid checkpoints while the drain thread is blocked on sample 0.
        for _ in range(4):
            assert ctx._checkpoint_hook is not None
            ctx._checkpoint_hook()
        release_writer.set()
        run_dir = ctx.run_dir

    bundle = inspect_run_observability(run_dir).invocations[0]
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "partial"
    status = bundle.statuses["resource-samples"]
    assert status.counts["dropped"] >= 1
    assert (
        status.counts["attempted"]
        == status.counts["written"] + status.counts["dropped"] + status.counts["failed"]
    )
    assert "queue_full" in status.truncation_reasons


def test_sample_byte_cap_and_writer_oserror_do_not_crash_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw sample byte cap and writer OSError are contained cleanly without failing the run."""

    # Case 1: tight sample byte cap preserves valid complete lines and records truncation.
    probe = _FakeRUsageProbe(
        [ProcessResourceBaseline(0.1 * i, 0.0, 0, 0, 0, 0, 10.0) for i in range(1, 5)]
    )
    cb_cap = ResourceSamples(
        interval_seconds=60.0,
        rusage=RUsageSource(
            enabled=True,
            quantities=("process.rusage.user_cpu_seconds",),
            probe=probe,
        ),
        sink=RunArtifactSink(max_sample_bytes=700),
    )
    assert run_from_config(_base_cfg(tmp_path / "cap"), raise_exceptions=True, runtime_callbacks=[cb_cap]) == 0
    bundle_cap = inspect_run_observability(tmp_path / "cap" / "outputs" / "o3-run").invocations[0]
    assert bundle_cap.derived_lifecycle_state == "closed"
    assert len(bundle_cap.samples["resource-samples"]) == 1
    assert "sample_bytes_cap_reached" in bundle_cap.statuses["resource-samples"].truncation_reasons
    assert bundle_cap.statuses["resource-samples"].counts["written"] == 1
    assert bundle_cap.statuses["resource-samples"].counts["dropped"] >= 1

    # Case 2: OSError during append_sample (simulated ENOSPC) is contained cleanly.
    def fail_append(self: ObservabilityBundleWriter, sample: Any) -> Any:
        del self, sample
        raise OSError("No space left on device")

    monkeypatch.setattr(ObservabilityBundleWriter, "append_sample", fail_append)
    cb_err = ResourceSamples(interval_seconds=60.0)
    assert run_from_config(_base_cfg(tmp_path / "enospc"), raise_exceptions=True, runtime_callbacks=[cb_err]) == 0
    bundle_err = inspect_run_observability(tmp_path / "enospc" / "outputs" / "o3-run").invocations[0]
    status_err = bundle_err.statuses["resource-samples"]
    assert "writer_error" in status_err.truncation_reasons
    assert status_err.counts["failed"] >= 1


def test_unquiescent_worker_marks_bundle_unsafe_and_refuses_reuse(tmp_path: Path) -> None:
    """A hung RUSAGE worker triggers bounded timeout, unsafe status, and reuse refusal."""

    probe = _BlockingRUsageProbe()
    cb = ResourceSamples(
        interval_seconds=0.01,
        close_timeout_seconds=0.08,
        rusage=RUsageSource(enabled=True, probe=probe),
    )
    try:
        with pytest.raises(ManagedLifecycleError, match="quiescence"):
            with open_run_context(_base_cfg(tmp_path / "first"), runtime_callbacks=[cb]) as ctx:
                assert probe.blocked_in_second.wait(timeout=2.0)
                first_run_dir = ctx.run_dir

        bundle = inspect_run_observability(first_run_dir).invocations[0]
        assert bundle.derived_lifecycle_state == "unsafe"
        assert bundle.derived_finalization == "unsafe"
        status = bundle.statuses["resource-samples"]
        assert status.lifecycle_state == "unsafe"
        assert status.helper_cleanup["state"] == "unconfirmed_timeout"
        assert "close_timeout" in status.truncation_reasons
        assert cb.is_managed_scope_active() is True

        # Re-entering an unsafe callback must be refused.
        with pytest.raises(ManagedLifecycleError):
            with open_run_context(_base_cfg(tmp_path / "second"), runtime_callbacks=[cb]):
                pass
    finally:
        probe.release.set()


def test_body_exception_and_keyboard_interrupt_preserved_while_bundle_closes(
    tmp_path: Path,
) -> None:
    """Primary Exception and KeyboardInterrupt propagate while observability still finalizes."""

    cb1 = ResourceSamples(interval_seconds=60.0)
    with pytest.raises(ValueError, match="runner failed deliberately"):
        run_from_config(
            _base_cfg(tmp_path / "err", runner={"_target_": f"{__name__}._RaisingRunner"}),
            raise_exceptions=True,
            runtime_callbacks=[cb1],
        )
    bundle1 = inspect_run_observability(tmp_path / "err" / "outputs" / "o3-run").invocations[0]
    assert bundle1.derived_lifecycle_state == "closed"
    assert bundle1.statuses["resource-samples"].application_outcome == "failed"
    assert bundle1.statuses["resource-samples"].invocation_outcome == "failed"

    cb2 = ResourceSamples(interval_seconds=60.0)
    with pytest.raises(KeyboardInterrupt, match="operator interrupt"):
        run_from_config(
            _base_cfg(
                tmp_path / "kb",
                runner={"_target_": f"{__name__}._KeyboardInterruptRunner"},
            ),
            raise_exceptions=True,
            runtime_callbacks=[cb2],
        )
    bundle2 = inspect_run_observability(tmp_path / "kb" / "outputs" / "o3-run").invocations[0]
    assert bundle2.derived_lifecycle_state == "closed"
    assert bundle2.statuses["resource-samples"].invocation_outcome == "interrupted"


def test_active_reentry_refused_while_clean_sequential_reuse_gets_fresh_invocation(
    tmp_path: Path,
) -> None:
    """Active re-entry raises ManagedLifecycleError; clean exit allows fresh sequential reuse."""

    cb = ResourceSamples(interval_seconds=60.0)
    with open_run_context(_base_cfg(tmp_path / "seq1"), runtime_callbacks=[cb]) as ctx1:
        with pytest.raises(ManagedLifecycleError, match="already active"):
            with cb.run_scope(ctx1):
                pass
        first_inv = ctx1.invocation_id

    assert cb.is_managed_scope_active() is False

    with open_run_context(_base_cfg(tmp_path / "seq2"), runtime_callbacks=[cb]) as ctx2:
        second_inv = ctx2.invocation_id

    assert first_inv != second_inv
    b1 = inspect_run_observability(tmp_path / "seq1" / "outputs" / "o3-run").invocations[0]
    b2 = inspect_run_observability(tmp_path / "seq2" / "outputs" / "o3-run").invocations[0]
    assert b1.invocation_id == first_inv
    assert b2.invocation_id == second_inv
    assert b1.samples["resource-samples"][0].sequence == 0
    assert b2.samples["resource-samples"][0].sequence == 0


# ---------------------------------------------------------------------------
# Scientific & sampler RNG invariance
# ---------------------------------------------------------------------------


def _run_bounded_training_attempt(
    root: Path,
    *,
    with_resource_samples: bool,
) -> dict[str, Any]:
    """Run a real bounded VMCTrainer + MetropolisSampler attempt with or without ResourceSamples."""

    torch = pytest.importorskip("torch")
    import numpy as np

    from tests.helpers import runtime_equivariance_recovery as recovery
    from tpen.training import trainer as trainer_module

    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)

    config = recovery._config(root.parent / "logical-run", "o3-science")
    context = recovery._context(root, "o3-science", config)
    context.monotonic_clock = lambda: 42.0
    model = recovery._ProbeModel(non_equivariant=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sampler = recovery._Sampler()
    trainer = recovery.VMCTrainer(max_steps=4, log_every_n_steps=1)

    identity = AcceleratorIdentity(AcceleratorKind.CUDA, 0, "GPU-INV")
    cb = (
        ResourceSamples(
            interval_seconds=0.05,
            rusage=RUsageSource(enabled=True),
            allocator=AllocatorSource(
                enabled=True,
                device="cuda:0",
                reset_peaks=True,
                current_probe_factory=lambda _d: _FakeCurrentAllocatorProbe(
                    [AllocatorCurrentUsage(identity, 2048, 4096, 1)]
                ),
                peak_probe_factory=lambda _d: _FakePeakAllocatorProbe(
                    [AllocatorUsage(identity, 2.0, 4.0, 1)]
                ),
            ),
        )
        if with_resource_samples
        else None
    )

    original_local_energy = trainer_module.local_energy
    trainer_module.local_energy = recovery._fake_local_energy
    try:
        if cb is not None:
            context.callbacks.append(cb)
            with cb.run_scope(context) as handle:
                assert handle is not None
                context._checkpoint_hook = handle.checkpoint
                context.emit(RunStarted())
                trainer.fit(
                    model=model,
                    sampler=sampler,
                    hamiltonian_terms=[],
                    optimizer=optimizer,
                    context=context,
                    emit=lambda *_args, **_kwargs: None,
                )
                context.emit(RunCompleted())
                context._checkpoint_hook = None
                handle.request_stop()
        else:
            context.emit(RunStarted())
            trainer.fit(
                model=model,
                sampler=sampler,
                hamiltonian_terms=[],
                optimizer=optimizer,
                context=context,
                emit=lambda *_args, **_kwargs: None,
            )
            context.emit(RunCompleted())

        checkpoint = save_checkpoint(
            output_dir=root / "checkpoint",
            next_iteration=trainer.next_iteration,
            completed_updates=trainer.completed_updates,
            model=model,
            optimizer=optimizer,
            trainer=trainer,
            sampler=sampler,
            context=context,
            payload=TrainResume(),
        )
    finally:
        trainer_module.local_energy = original_local_energy

    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "sampler": copy.deepcopy(sampler.mcmc_state_dict()),
        "trainer": copy.deepcopy(trainer.state_dict()),
        "rng": torch.get_rng_state().clone(),
        "records": list(context.records),
        "hashes": manifest["hashes"],
        "run_dir": context.run_dir,
    }


def test_resource_samples_preserves_exact_scientific_and_sampler_rng_state(
    tmp_path: Path,
) -> None:
    """Enabling ResourceSamples leaves model, optimizer, sampler, RNG, and checkpoint hashes identical."""

    torch = pytest.importorskip("torch")

    baseline = _run_bounded_training_attempt(tmp_path / "off", with_resource_samples=False)
    observed = _run_bounded_training_attempt(tmp_path / "on", with_resource_samples=True)

    for key in baseline["model"]:
        assert torch.equal(baseline["model"][key], observed["model"][key])
    assert baseline["sampler"] == observed["sampler"]
    assert baseline["trainer"] == observed["trainer"]
    assert torch.equal(baseline["rng"], observed["rng"])
    assert baseline["records"] == observed["records"]
    assert baseline["hashes"] == observed["hashes"]

    # Confirm ResourceSamples actually persisted a valid closed bundle during the observed run.
    obs = inspect_run_observability(observed["run_dir"])
    assert obs.status == "present"
    assert obs.invocations[0].derived_lifecycle_state == "closed"
    assert len(obs.invocations[0].samples["resource-samples"]) >= 2


# ---------------------------------------------------------------------------
# Review-test suite (T1-T9): defect-discriminating & coverage-gap tests
# ---------------------------------------------------------------------------


def test_close_drops_slow_writer_backlog_at_deadline_while_remaining_quiescent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T1 (F1): Close stops drain thread promptly and drops slow backlog at deadline cleanly."""

    writer_entered = threading.Event()
    release_writer = threading.Event()
    original_append = ObservabilityBundleWriter.append_sample
    call_count = 0

    def slow_append(self: ObservabilityBundleWriter, sample: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            writer_entered.set()
            release_writer.wait(timeout=2.0)
        time.sleep(0.04)
        return original_append(self, sample)

    monkeypatch.setattr(ObservabilityBundleWriter, "append_sample", slow_append)

    identity = AcceleratorIdentity(AcceleratorKind.CUDA, 0, "GPU-T1")
    current_probe = _FakeCurrentAllocatorProbe(
        [AllocatorCurrentUsage(identity, 1024 * (i + 1), 4096 * (i + 1), 1) for i in range(16)]
    )
    clock = _SteppingClock(start_ns=1_000_000_000_000, step_ns=1_000_000_000)
    cb = ResourceSamples(
        interval_seconds=0.5,
        max_pending_envelopes=16,
        close_timeout_seconds=0.16,
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(
            enabled=True,
            include_current=True,
            include_peak=False,
            reset_peaks=False,
            current_probe_factory=lambda _d: current_probe,
        ),
        monotonic_ns=clock,
    )

    run_dir: Path | None = None
    try:
        with open_run_context(_base_cfg(tmp_path), runtime_callbacks=[cb]) as ctx:
            run_dir = ctx.run_dir
            assert writer_entered.wait(timeout=2.0)
            for _ in range(8):
                assert ctx._checkpoint_hook is not None
                ctx._checkpoint_hook()
            release_writer.set()
    finally:
        release_writer.set()

    assert run_dir is not None
    assert cb.is_managed_scope_active() is False
    bundle = inspect_run_observability(run_dir).invocations[0]
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "partial"
    status = bundle.statuses["resource-samples"]
    assert "close_timeout" in status.truncation_reasons
    assert status.counts["written"] >= 1
    assert status.counts["dropped"] >= 1
    assert (
        status.counts["attempted"]
        == status.counts["written"] + status.counts["dropped"] + status.counts["failed"]
    )


def test_aborted_session_close_fails_closed_and_refuses_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2 (F2): An unexpected abort inside session.close() fails closed and blocks reuse."""

    probe = _BlockingRUsageProbe()
    cb = ResourceSamples(
        interval_seconds=0.01,
        close_timeout_seconds=0.5,
        rusage=RUsageSource(enabled=True, probe=probe),
    )
    ctx = make_run_context(tmp_path, run_id="t2-run")
    session: Any = None
    rusage_thread: threading.Thread | None = None
    drain_thread: threading.Thread | None = None

    def _abort_join(timeout: float | None = None) -> None:
        del timeout
        raise RuntimeError("injected close abort")

    try:
        with pytest.raises(RuntimeError, match="injected close abort"):
            with cb.run_scope(ctx):
                assert probe.blocked_in_second.wait(timeout=2.0)
                session = cb._active_session
                assert session is not None
                rusage_thread = session._rusage_thread
                drain_thread = session._drain_thread
                assert rusage_thread is not None
                monkeypatch.setattr(rusage_thread, "join", _abort_join)

        assert cb.is_managed_scope_active() is True
        with pytest.raises(ManagedLifecycleError):
            with cb.run_scope(ctx):
                pass
    finally:
        probe.release.set()
        if session is not None:
            session._stop_event.set()
            session._drain_stop_event.set()
            session._drain_wake_event.set()
        if rusage_thread is not None and rusage_thread.is_alive():
            threading.Thread.join(rusage_thread, timeout=2.0)
        if drain_thread is not None and drain_thread.is_alive():
            threading.Thread.join(drain_thread, timeout=2.0)


def test_ownership_claims_detect_same_bundle_conflict_across_all_call_forms(
    tmp_path: Path,
) -> None:
    """T3 (F3): Same-bundle artifact_writer conflict is detected in all three claim call forms."""

    cb_a = ResourceSamples(
        observer_id="obs-a", sink=RunArtifactSink(relative_dir="observability")
    )
    cb_b = ResourceSamples(
        observer_id="obs-b", sink=RunArtifactSink(relative_dir="observability")
    )
    cb_other = ResourceSamples(
        observer_id="obs-c", sink=RunArtifactSink(relative_dir="observability_other")
    )
    am = ArtifactManager(tmp_path / "outputs", "o3_test", "unit", "o3-run", layout="flat")

    def _any_conflict(first: list[Any], second: list[Any]) -> bool:
        return any(claims_conflict(a, b) for a in first for b in second)

    forms = {
        "artifact_manager": (
            cb_a.ownership_claims(artifact_manager=am, invocation_id="inv-1"),
            cb_b.ownership_claims(artifact_manager=am, invocation_id="inv-1"),
            cb_other.ownership_claims(artifact_manager=am, invocation_id="inv-1"),
        ),
        "run_dir": (
            cb_a.ownership_claims(run_dir=am.run_dir),
            cb_b.ownership_claims(run_dir=am.run_dir),
            cb_other.ownership_claims(run_dir=am.run_dir),
        ),
        "no_args": (
            cb_a.ownership_claims(),
            cb_b.ownership_claims(),
            cb_other.ownership_claims(),
        ),
    }

    other_bundle = {name: _any_conflict(a, c) for name, (a, _, c) in forms.items()}
    same_bundle = {name: _any_conflict(a, b) for name, (a, b, _) in forms.items()}
    assert other_bundle == {
        "artifact_manager": False,
        "run_dir": False,
        "no_args": False,
    }
    assert same_bundle == {
        "artifact_manager": True,
        "run_dir": True,
        "no_args": True,
    }


def test_resource_samples_rejects_subclasses_of_sink_and_source_configs() -> None:
    """T4 (F4): Exact-type coercion refuses subclasses of RunArtifactSink, RUsageSource, AllocatorSource."""

    class _SubSink(RunArtifactSink):
        pass

    class _SubRUsage(RUsageSource):
        pass

    class _SubAllocator(AllocatorSource):
        pass

    accepted: list[str] = []
    for label, factory in (
        ("sink", lambda: ResourceSamples(sink=_SubSink())),
        ("rusage", lambda: ResourceSamples(rusage=_SubRUsage())),
        (
            "allocator",
            lambda: ResourceSamples(
                rusage=RUsageSource(enabled=False),
                allocator=_SubAllocator(enabled=True, reset_peaks=False),
            ),
        ),
    ):
        try:
            factory()
        except TypeError:
            continue
        accepted.append(label)

    assert accepted == [], (
        f"ResourceSamples accepted subclass(es) instead of raising TypeError: {accepted}"
    )


def test_containment_refusal_in_start_writes_no_post_refusal_status_or_closed_manifest(
    tmp_path: Path,
) -> None:
    """T5 (F5): Containment refusal during start() must not write status or a closed manifest."""

    # Sub-case A: symlinked manifest.json
    ctx_a = make_run_context(tmp_path / "case_a", run_id="case-a")
    ctx_a.invocation_id = "inv-symlink-a"
    bundle_a = ctx_a.run_dir / "observability" / "inv-symlink-a"
    bundle_a.mkdir(parents=True, exist_ok=True)
    (bundle_a / "manifest.json").symlink_to(tmp_path / "outside_manifest.json")

    cb_a = ResourceSamples(interval_seconds=60.0)
    with pytest.raises(ObservabilityPathError):
        with cb_a.run_scope(ctx_a):
            pass
    assert cb_a.is_managed_scope_active() is False
    a_status_exists = (bundle_a / "status" / "resource-samples.json").exists()

    # Sub-case B: symlinked inventory.json
    ctx_b = make_run_context(tmp_path / "case_b", run_id="case-b")
    ctx_b.invocation_id = "inv-symlink-b"
    bundle_b = ctx_b.run_dir / "observability" / "inv-symlink-b"
    bundle_b.mkdir(parents=True, exist_ok=True)
    (bundle_b / "inventory.json").symlink_to(tmp_path / "outside_inventory.json")

    cb_b = ResourceSamples(interval_seconds=60.0)
    with pytest.raises(ObservabilityPathError):
        with cb_b.run_scope(ctx_b):
            pass
    assert cb_b.is_managed_scope_active() is False
    b_status_exists = (bundle_b / "status" / "resource-samples.json").exists()
    b_manifest_state: str | None = None
    manifest_b = bundle_b / "manifest.json"
    if manifest_b.exists():
        b_manifest_state = json.loads(manifest_b.read_text(encoding="utf-8"))["lifecycle_state"]

    failures: list[str] = []
    if a_status_exists:
        failures.append("sub-case A wrote status/resource-samples.json after manifest symlink refusal")
    if b_status_exists:
        failures.append("sub-case B wrote status/resource-samples.json after inventory symlink refusal")
    if b_manifest_state == "closed":
        failures.append("sub-case B overwrote manifest.json to lifecycle_state='closed' after inventory symlink refusal")
    assert failures == [], "; ".join(failures)


def test_allocator_provenance_honesty_on_unavailable_backend_reset_failure_and_dual_init_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T6 (F6): Inventory capability, reset_epoch_owner, and dual init error details stay honest."""

    failures: list[str] = []

    # Sub-case A: real TorchAllocatorCurrentProbe + TorchAllocatorPeakProbe on unavailable CUDA
    class _FakeTorchDevice:
        def __init__(self, spec: Any, index: int | None = None) -> None:
            if isinstance(spec, _FakeTorchDevice):
                self.type = spec.type
                self.index = spec.index if index is None else index
            else:
                text = str(spec)
                if ":" in text:
                    dev_type, idx_str = text.split(":", 1)
                    self.type = dev_type
                    self.index = int(idx_str)
                else:
                    self.type = text
                    self.index = index

    class _UnavailableBackend:
        def is_available(self) -> bool:
            return False

    fake_torch = types.SimpleNamespace(
        device=_FakeTorchDevice,
        version=types.SimpleNamespace(hip=None),
        get_device_module=lambda _dev_type: _UnavailableBackend(),
    )
    with monkeypatch.context() as mp:
        mp.setattr(accelerator_mod, "_torch", lambda _feature: fake_torch)
        mp.setattr(
            accelerator_mod,
            "device_module",
            lambda _device=None, *, feature="": _UnavailableBackend(),
        )
        cb_a = ResourceSamples(
            interval_seconds=60.0,
            rusage=RUsageSource(enabled=False),
            allocator=AllocatorSource(enabled=True, device="cuda:0", reset_peaks=True),
        )
        with open_run_context(
            _base_cfg(tmp_path / "unavail", device="cuda:0"), runtime_callbacks=[cb_a]
        ) as ctx_a:
            run_dir_a = ctx_a.run_dir

    bundle_a = inspect_run_observability(run_dir_a).invocations[0]
    assert bundle_a.inventory is not None
    cap_a = bundle_a.inventory.sources[ALLOCATOR_SOURCE_ID]["capability"]
    owner_a = bundle_a.samples["resource-samples"][0].subject["reset_epoch_owner"]
    if cap_a != "unavailable":
        failures.append(f"sub-case A capability={cap_a!r} (expected 'unavailable')")
    if owner_a == "self":
        failures.append(f"sub-case A reset_epoch_owner={owner_a!r} (expected != 'self')")

    # Sub-case B: reset failure while current probe stays ok
    identity_b = AcceleratorIdentity(AcceleratorKind.CUDA, 0, "GPU-F6B")
    current_b = _FakeCurrentAllocatorProbe([AllocatorCurrentUsage(identity_b, 4096, 8192, 1)])
    peak_b = _FakePeakAllocatorProbe(
        [AllocatorUsage(identity_b, 2.0, 4.0, 1)],
        reset_result=AllocatorUnavailable("driver refused reset"),
    )
    cb_b = ResourceSamples(
        interval_seconds=60.0,
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(
            enabled=True,
            device="cuda:0",
            include_current=True,
            include_peak=True,
            reset_peaks=True,
            current_probe_factory=lambda _d: current_b,
            peak_probe_factory=lambda _d: peak_b,
        ),
    )
    with open_run_context(
        _base_cfg(tmp_path / "reset_fail", device="cuda:0"), runtime_callbacks=[cb_b]
    ) as ctx_b:
        run_dir_b = ctx_b.run_dir

    bundle_b = inspect_run_observability(run_dir_b).invocations[0]
    assert bundle_b.inventory is not None
    cap_b = bundle_b.inventory.sources[ALLOCATOR_SOURCE_ID]["capability"]
    owner_b = bundle_b.samples["resource-samples"][0].subject["reset_epoch_owner"]
    if cap_b != "partial":
        failures.append(f"sub-case B capability={cap_b!r} (expected 'partial')")
    if owner_b == "self":
        failures.append(f"sub-case B reset_epoch_owner={owner_b!r} (expected != 'self')")

    # Sub-case C: both factories fail with distinct exceptions
    def _fail_current(_dev: Any) -> Any:
        raise RuntimeError("current-init-boom")

    def _fail_peak(_dev: Any) -> Any:
        raise RuntimeError("peak-init-boom")

    cb_c = ResourceSamples(
        interval_seconds=60.0,
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(
            enabled=True,
            device="cuda:0",
            include_current=True,
            include_peak=True,
            reset_peaks=True,
            current_probe_factory=_fail_current,
            peak_probe_factory=_fail_peak,
        ),
    )
    with open_run_context(
        _base_cfg(tmp_path / "dual_init", device="cuda:0"), runtime_callbacks=[cb_c]
    ) as ctx_c:
        run_dir_c = ctx_c.run_dir

    bundle_c = inspect_run_observability(run_dir_c).invocations[0]
    assert bundle_c.inventory is not None
    sample_c = bundle_c.samples["resource-samples"][0]
    cur_detail = sample_c.readings["allocator.current.allocated_bytes"].detail or ""
    peak_detail = sample_c.readings["allocator.peak.allocated_bytes"].detail or ""
    cap_c = bundle_c.inventory.sources[ALLOCATOR_SOURCE_ID]["capability"]
    owner_c = sample_c.subject["reset_epoch_owner"]
    if "current-init-boom" not in cur_detail:
        failures.append(f"sub-case C current detail={cur_detail!r} missing 'current-init-boom'")
    if "peak-init-boom" not in peak_detail:
        failures.append(f"sub-case C peak detail={peak_detail!r} missing 'peak-init-boom'")
    if cap_c != "unavailable":
        failures.append(f"sub-case C capability={cap_c!r} (expected 'unavailable')")
    if owner_c == "self":
        failures.append(f"sub-case C reset_epoch_owner={owner_c!r} (expected != 'self')")

    assert failures == [], "; ".join(failures)


def test_yaml_hydra_nested_targets_plain_dicts_config_identity_and_profile_refusal(
    tmp_path: Path,
) -> None:
    """T7: YAML nested targets, plain dicts, config identity sensitivity, and profile refusal."""

    # Test A: Full YAML with nested _target_ entries and interpolation
    yaml_a = f"""
experiment:
  name: o3_test
  sector: unit
run:
  root: {json.dumps(str(tmp_path / "yaml_nested" / "outputs"))}
  layout: flat
  run_id: o3-run
  timezone: UTC
runtime:
  device: cpu
  dtype: float64
callbacks:
  - _target_: tpen.callback.ResourceSamples
    interval_seconds: 60.0
    rusage:
      _target_: tpen.callback.RUsageSource
      enabled: true
      quantities:
        - process.rusage.user_cpu_seconds
        - process.rusage.max_rss_bytes
    allocator:
      _target_: tpen.callback.AllocatorSource
      enabled: true
      device: "${{runtime.device}}"
      reset_peaks: false
    sink:
      _target_: tpen.artifacts.RunArtifactSink
      relative_dir: observability
loggers: []
runner:
  _target_: {__name__}._NoopRunner
"""
    cfg_a = OmegaConf.create(yaml_a)
    assert run_from_config(cfg_a, raise_exceptions=True) == 0

    bundle_a = inspect_run_observability(
        tmp_path / "yaml_nested" / "outputs" / "o3-run"
    ).invocations[0]
    assert bundle_a.derived_lifecycle_state == "closed"
    assert bundle_a.descriptors is not None
    expected_rusage = {"process.rusage.user_cpu_seconds", "process.rusage.max_rss_bytes"}
    expected_alloc = set(ALLOCATOR_CURRENT_QUANTITY_IDS + ALLOCATOR_PEAK_QUANTITY_IDS)
    assert set(bundle_a.descriptors.quantities.keys()) == expected_rusage | expected_alloc

    samples_a = bundle_a.samples["resource-samples"]
    rusage_samples = [s for s in samples_a if s.source_id == RUSAGE_SOURCE_ID]
    alloc_samples = [s for s in samples_a if s.source_id == ALLOCATOR_SOURCE_ID]
    assert len(rusage_samples) >= 1
    assert len(alloc_samples) >= 1
    assert set(rusage_samples[0].readings.keys()) == expected_rusage
    assert set(alloc_samples[0].readings.keys()) == expected_alloc

    # Test B: Plain YAML mapping config (coerced dicts without nested _target_)
    yaml_plain = f"""
experiment:
  name: o3_test
  sector: unit
run:
  root: {json.dumps(str(tmp_path / "yaml_plain" / "outputs"))}
  layout: flat
  run_id: o3-run
  timezone: UTC
runtime:
  device: cpu
  dtype: float64
callbacks:
  - _target_: tpen.callback.ResourceSamples
    interval_seconds: 60.0
    rusage:
      enabled: true
      quantities:
        - process.rusage.user_cpu_seconds
    sink:
      relative_dir: obs_plain
loggers: []
runner:
  _target_: {__name__}._NoopRunner
"""
    cfg_plain = OmegaConf.create(yaml_plain)
    assert run_from_config(cfg_plain, raise_exceptions=True) == 0
    obs_plain = inspect_run_observability(
        tmp_path / "yaml_plain" / "outputs" / "o3-run",
        relative_dir="obs_plain",
    )
    assert obs_plain.status == "present"
    assert obs_plain.invocations[0].derived_lifecycle_state == "closed"

    # Test C: Observation config changes resolved_config_identity normally without error
    cfg_b = OmegaConf.create(yaml_a.replace("interval_seconds: 60.0", "interval_seconds: 30.0"))
    assert resolved_config_identity(cfg_a) != resolved_config_identity(cfg_b)

    # Test D: Fail-closed native recovery profile admission refuses ResourceSamples
    profile_check = check_profile(cfg_a, NATIVE_RECOVERY_PROFILE)
    assert profile_check.verdict == "refused"
    assert profile_check.refusal is not None
    assert profile_check.refusal["code"] == "unsupported-callback"


def test_pending_bytes_cap_reached_drops_nonblockingly_and_releases_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T8: Pending byte cap drops nonblockingly and releases all reservations on exit."""

    # Sub-case A: max_pending_bytes smaller than a single sample drops all samples cleanly
    cb_tiny = ResourceSamples(
        interval_seconds=60.0,
        sink=RunArtifactSink(max_pending_bytes=64),
    )
    with open_run_context(_base_cfg(tmp_path / "tiny"), runtime_callbacks=[cb_tiny]) as ctx_tiny:
        run_dir_tiny = ctx_tiny.run_dir

    bundle_tiny = inspect_run_observability(run_dir_tiny).invocations[0]
    assert bundle_tiny.derived_lifecycle_state == "closed"
    assert bundle_tiny.derived_data_completeness == "none"
    status_tiny = bundle_tiny.statuses["resource-samples"]
    assert "pending_bytes_cap_reached" in status_tiny.truncation_reasons
    assert status_tiny.counts["written"] == 0
    assert status_tiny.counts["dropped"] >= 1
    assert status_tiny.counts["attempted"] == status_tiny.counts["dropped"]

    # Sub-case B: Stalled writer fills max_pending_bytes=1200, drops overflow, releases to 0
    writer_entered = threading.Event()
    release_writer = threading.Event()
    original_append = ObservabilityBundleWriter.append_sample
    call_count = 0

    def stalled_first_append(self: ObservabilityBundleWriter, sample: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            writer_entered.set()
            release_writer.wait(timeout=2.0)
        return original_append(self, sample)

    monkeypatch.setattr(ObservabilityBundleWriter, "append_sample", stalled_first_append)

    identity = AcceleratorIdentity(AcceleratorKind.CUDA, 0, "GPU-T8")
    current_probe = _FakeCurrentAllocatorProbe(
        [AllocatorCurrentUsage(identity, 1024 * (i + 1), 4096 * (i + 1), 1) for i in range(12)]
    )
    clock = _SteppingClock(start_ns=1_000_000_000_000, step_ns=1_000_000_000)
    cb_stalled = ResourceSamples(
        interval_seconds=0.5,
        max_pending_envelopes=16,
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(
            enabled=True,
            include_current=True,
            include_peak=False,
            reset_peaks=False,
            current_probe_factory=lambda _d: current_probe,
        ),
        sink=RunArtifactSink(max_pending_bytes=1200),
        monotonic_ns=clock,
    )

    writer: ObservabilityBundleWriter | None = None
    try:
        with open_run_context(
            _base_cfg(tmp_path / "stalled"), runtime_callbacks=[cb_stalled]
        ) as ctx_stalled:
            run_dir_stalled = ctx_stalled.run_dir
            assert writer_entered.wait(timeout=2.0)
            assert cb_stalled._active_session is not None
            writer = cb_stalled._active_session._writer
            for _ in range(5):
                assert ctx_stalled._checkpoint_hook is not None
                ctx_stalled._checkpoint_hook()
            release_writer.set()
    finally:
        release_writer.set()

    assert writer is not None
    assert writer.pending_bytes("resource-samples") == 0
    bundle_stalled = inspect_run_observability(run_dir_stalled).invocations[0]
    assert bundle_stalled.derived_lifecycle_state == "closed"
    status_stalled = bundle_stalled.statuses["resource-samples"]
    assert "pending_bytes_cap_reached" in status_stalled.truncation_reasons
    assert status_stalled.counts["written"] >= 1
    assert status_stalled.counts["dropped"] >= 1
    assert (
        status_stalled.counts["attempted"]
        == status_stalled.counts["written"]
        + status_stalled.counts["dropped"]
        + status_stalled.counts["failed"]
    )


def test_allocator_reset_ownership_conflicts_across_rocm_hip_injected_and_duplicate_resetters(
    tmp_path: Path,
) -> None:
    """T9: Allocator reset conflicts cover cuda/cuda:0, hip:0/rocm, unknown probes, and distinct indices."""

    # Conflict 1: Two ResourceSamples with distinct relative_dirs, cuda vs cuda:0
    cb_c1a = ResourceSamples(
        observer_id="obs-a",
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(enabled=True, device="cuda", reset_peaks=True),
        sink=RunArtifactSink(relative_dir="obs_a"),
    )
    cb_c1b = ResourceSamples(
        observer_id="obs-b",
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(enabled=True, device="cuda:0", reset_peaks=True),
        sink=RunArtifactSink(relative_dir="obs_b"),
    )
    with pytest.raises(ResourceOwnershipConflictError):
        run_from_config(
            _base_cfg(tmp_path / "conflict_cuda"),
            raise_exceptions=True,
            runtime_callbacks=[cb_c1a, cb_c1b],
        )
    assert not (tmp_path / "conflict_cuda" / "outputs").exists()

    # Conflict 2: Two ResourceSamples with distinct relative_dirs, hip:0 vs rocm
    cb_c2a = ResourceSamples(
        observer_id="obs-a",
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(enabled=True, device="hip:0", reset_peaks=True),
        sink=RunArtifactSink(relative_dir="obs_a"),
    )
    cb_c2b = ResourceSamples(
        observer_id="obs-b",
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(enabled=True, device="rocm", reset_peaks=True),
        sink=RunArtifactSink(relative_dir="obs_b"),
    )
    with pytest.raises(ResourceOwnershipConflictError):
        run_from_config(
            _base_cfg(tmp_path / "conflict_rocm"),
            raise_exceptions=True,
            runtime_callbacks=[cb_c2a, cb_c2b],
        )
    assert not (tmp_path / "conflict_rocm" / "outputs").exists()

    # Conflict 3: Resetting ResourceSamples vs ResourceUsage with unknown-target probe
    class _UnknownResetProbe:
        def reset(self) -> AcceleratorIdentity | AllocatorUnavailable:
            return AcceleratorIdentity(AcceleratorKind.CUDA, 0, "GPU-UNK")

        def read(self) -> AllocatorUsage:
            identity = AcceleratorIdentity(AcceleratorKind.CUDA, 0, "GPU-UNK")
            return AllocatorUsage(identity, 1.0, 2.0, 1)

    cb_c3 = ResourceSamples(
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(enabled=True, device="cuda:0", reset_peaks=True),
    )
    ru_unknown = ResourceUsage(allocator_probe=_UnknownResetProbe())
    with pytest.raises(ResourceOwnershipConflictError):
        run_from_config(
            _base_cfg(tmp_path / "conflict_unknown"),
            raise_exceptions=True,
            runtime_callbacks=[cb_c3, ru_unknown],
        )
    assert not (tmp_path / "conflict_unknown" / "outputs").exists()

    # Coexistence: Two ResourceSamples with distinct relative_dirs and distinct indexed devices
    id_0 = AcceleratorIdentity(AcceleratorKind.CUDA, 0, "GPU-0")
    id_1 = AcceleratorIdentity(AcceleratorKind.CUDA, 1, "GPU-1")
    cur_0 = _FakeCurrentAllocatorProbe([AllocatorCurrentUsage(id_0, 1024, 2048, 2)])
    peak_0 = _FakePeakAllocatorProbe([AllocatorUsage(id_0, 1.0, 2.0, 2)])
    cur_1 = _FakeCurrentAllocatorProbe([AllocatorCurrentUsage(id_1, 4096, 8192, 2)])
    peak_1 = _FakePeakAllocatorProbe([AllocatorUsage(id_1, 3.0, 4.0, 2)])

    cb_0 = ResourceSamples(
        observer_id="obs-0",
        interval_seconds=60.0,
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(
            enabled=True,
            device="cuda:0",
            include_current=True,
            include_peak=True,
            reset_peaks=True,
            current_probe_factory=lambda _d: cur_0,
            peak_probe_factory=lambda _d: peak_0,
        ),
        sink=RunArtifactSink(relative_dir="obs_0"),
    )
    cb_1 = ResourceSamples(
        observer_id="obs-1",
        interval_seconds=60.0,
        rusage=RUsageSource(enabled=False),
        allocator=AllocatorSource(
            enabled=True,
            device="cuda:1",
            include_current=True,
            include_peak=True,
            reset_peaks=True,
            current_probe_factory=lambda _d: cur_1,
            peak_probe_factory=lambda _d: peak_1,
        ),
        sink=RunArtifactSink(relative_dir="obs_1"),
    )
    exit_code = run_from_config(
        _base_cfg(tmp_path / "coexist", device="cpu"),
        raise_exceptions=True,
        runtime_callbacks=[cb_0, cb_1],
    )
    assert exit_code == 0
    assert peak_0.reset_calls == 1
    assert peak_1.reset_calls == 1
    run_dir = tmp_path / "coexist" / "outputs" / "o3-run"
    obs_0 = inspect_run_observability(run_dir, relative_dir="obs_0")
    obs_1 = inspect_run_observability(run_dir, relative_dir="obs_1")
    assert obs_0.status == "present"
    assert obs_1.status == "present"
    assert obs_0.invocations[0].derived_lifecycle_state == "closed"
    assert obs_1.invocations[0].derived_lifecycle_state == "closed"


# ---------------------------------------------------------------------------
# O4: LinuxMemorySource configuration, O2 wire round-trip, and failure modes
# ---------------------------------------------------------------------------


class _FakeLinuxMemoryProbe:
    """Deterministic sequence-backed Linux memory probe for callback unit tests."""

    def __init__(
        self,
        readings: list[LinuxProcessMemoryReading | Exception],
        *,
        generation: LinuxProcessGeneration | None = None,
        bind_error: LinuxMemoryUnavailable | None = None,
        target_reads: int | None = None,
    ) -> None:
        self._readings = list(readings)
        self._index = 0
        self._generation = generation
        self._bind_error = bind_error
        self._target_reads = target_reads
        self.reached_target = threading.Event()
        self.bind_calls = 0
        self.read_calls = 0
        self.thread_ids: list[int] = []

    @property
    def target_generation(self) -> LinuxProcessGeneration | None:
        return self._generation

    @property
    def bind_error(self) -> LinuxMemoryUnavailable | None:
        return self._bind_error

    def bind_target(self) -> LinuxProcessGeneration | None:
        self.bind_calls += 1
        return self._generation

    def read(self) -> LinuxProcessMemoryReading:
        self.read_calls += 1
        self.thread_ids.append(threading.get_ident())
        idx = min(self._index, len(self._readings) - 1)
        self._index += 1
        if self._target_reads is not None and self.read_calls >= self._target_reads:
            self.reached_target.set()
        item = self._readings[idx]
        if isinstance(item, Exception):
            raise item
        return item


class _BlockingLinuxMemoryProbe:
    """Linux memory probe that blocks on its second read to verify close timeouts."""

    def __init__(self, generation: LinuxProcessGeneration) -> None:
        self._generation = generation
        self.blocked_in_second = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def bind_target(self) -> LinuxProcessGeneration:
        return self._generation

    def read(self) -> LinuxProcessMemoryReading:
        self.calls += 1
        if self.calls == 1:
            return LinuxProcessMemoryReading(
                status="ok",
                rss_bytes=4 * 1024 * 1024,
                virtual_bytes=16 * 1024 * 1024,
                thread_count=2,
                read_start_ns=100,
                read_end_ns=110,
                target_generation=self._generation,
                observed_generation_before=self._generation,
                observed_generation_after=self._generation,
            )
        self.blocked_in_second.set()
        self.release.wait(timeout=5.0)
        return LinuxProcessMemoryReading(
            status="ok",
            rss_bytes=8 * 1024 * 1024,
            virtual_bytes=16 * 1024 * 1024,
            thread_count=2,
            read_start_ns=200,
            read_end_ns=210,
            target_generation=self._generation,
            observed_generation_before=self._generation,
            observed_generation_after=self._generation,
        )


def _write_fake_proc_tree(
    proc_root: Path,
    *,
    pid: int,
    start_ticks: int = 9000,
    boot_id: str = "00000000-0000-4000-8000-000000000002",
    pid_ns: str = "pid:[4026531836]",
    vm_rss_kib: int = 1024,
    vm_size_kib: int = 4096,
    threads: int = 3,
) -> Path:
    proc_root.mkdir(parents=True, exist_ok=True)
    boot_file = proc_root / "sys" / "kernel" / "random" / "boot_id"
    boot_file.parent.mkdir(parents=True, exist_ok=True)
    boot_file.write_text(f"{boot_id}\n", encoding="utf-8")
    pid_dir = proc_root / str(pid)
    pid_dir.mkdir(parents=True, exist_ok=True)
    stat_fields = "S 1 1 1 0 -1 4194304 10 0 0 0 1 1 0 0 20 0 3 0"
    (pid_dir / "stat").write_text(
        f"{pid} (python3) {stat_fields} {start_ticks} 1000 50\n",
        encoding="utf-8",
    )
    (pid_dir / "status").write_text(
        f"Name:\tpython3\nPid:\t{pid}\nVmSize:\t{vm_size_kib} kB\n"
        f"VmRSS:\t{vm_rss_kib} kB\nThreads:\t{threads}\n",
        encoding="utf-8",
    )
    ns_dir = pid_dir / "ns"
    ns_dir.mkdir(parents=True, exist_ok=True)
    ns_link = ns_dir / "pid"
    if ns_link.is_symlink() or ns_link.exists():
        ns_link.unlink()
    os.symlink(pid_ns, ns_link)
    return proc_root


def test_linux_memory_source_torch_free_export_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lazy export is Torch-free, pure at construction, and enforces strict validation."""

    assert "tpen.callback.LinuxMemorySource" not in _KNOWN_NON_MANAGED_CALLBACK_TARGETS
    assert (
        "tpen.callback.resource_samples.LinuxMemorySource"
        not in _KNOWN_NON_MANAGED_CALLBACK_TARGETS
    )

    # Fresh subprocess Torch-free check covering process_resources + LinuxMemorySource.
    code = (
        "import sys, tpen.process_resources, tpen.callback.resource_samples; "
        "from tpen.callback import LinuxMemorySource, ResourceSamples; "
        "cb = ResourceSamples(linux_memory=LinuxMemorySource(enabled=True)); "
        "assert cb.linux_memory.enabled is True; "
        "assert cb.rusage.enabled is False; "
        "assert cb.allocator.enabled is False; "
        "assert 'torch' not in sys.modules; "
        "assert 'experiments' not in sys.modules; "
        "assert 'tpen_executor' not in sys.modules; "
        "print('TORCH_FREE_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "TORCH_FREE_OK" in proc.stdout

    # Default ResourceSamples preserves O3 defaults (linux_memory disabled).
    default_cb = ResourceSamples()
    assert default_cb.rusage.enabled is True
    assert default_cb.allocator.enabled is False
    assert default_cb.linux_memory.enabled is False

    # Subclass refusal (exact-type coercion).
    class _SubLinuxMemory(LinuxMemorySource):
        pass

    with pytest.raises(TypeError, match="subclass"):
        ResourceSamples(linux_memory=_SubLinuxMemory())

    # Forbidden selector keys refused in mapping coercion.
    with pytest.raises(ValueError, match="forbidden"):
        ResourceSamples(linux_memory={"enabled": True, "triggers": ["step"]})

    # Duplicate source_id across enabled sources refused.
    with pytest.raises(ValueError, match="distinct source_id"):
        ResourceSamples(
            rusage=RUsageSource(enabled=True, source_id="shared"),
            linux_memory=LinuxMemorySource(enabled=True, source_id="shared"),
        )
    with pytest.raises(ValueError, match="distinct source_id"):
        ResourceSamples(
            allocator=AllocatorSource(enabled=True, reset_peaks=False, source_id="shared"),
            linux_memory=LinuxMemorySource(enabled=True, source_id="shared"),
        )

    # Foreign PID refused; current PID accepted; pid=None never calls os.getpid().
    foreign_pid = os.getpid() + 1
    with pytest.raises(ValueError, match="foreign PID"):
        LinuxMemorySource(pid=foreign_pid)
    assert LinuxMemorySource(pid=os.getpid()).pid == os.getpid()

    with monkeypatch.context() as mp:
        mp.setattr(
            os,
            "getpid",
            lambda: (_ for _ in ()).throw(AssertionError("must not freeze pid at init")),
        )
        pure_src = LinuxMemorySource(enabled=True)
        assert pure_src.pid is None

    # Invalid execution_mode, bounds, and quantities refused.
    for bad_mode in ("helper", "subprocess", "in_process_cooperative", ""):
        with pytest.raises(ValueError, match="execution_mode"):
            LinuxMemorySource(execution_mode=bad_mode)
    with pytest.raises(TypeError):
        LinuxMemorySource(max_read_bytes=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LinuxMemorySource(max_read_bytes=0)
    with pytest.raises(ValueError):
        LinuxMemorySource(max_read_seconds=0.0)
    with pytest.raises(ValueError):
        LinuxMemorySource(max_read_seconds=math.nan)
    with pytest.raises(ValueError):
        LinuxMemorySource(quantities=())
    with pytest.raises(ValueError):
        LinuxMemorySource(quantities=("process.rusage.max_rss_bytes",))
    with pytest.raises(ValueError):
        LinuxMemorySource(
            quantities=("process.memory.rss_bytes", "process.memory.rss_bytes")
        )
    with pytest.raises(ValueError):
        LinuxMemorySource(proc_root="   ")
    with pytest.raises(TypeError):
        LinuxMemorySource(proc_root=123)  # type: ignore[arg-type]


def test_linux_memory_alongside_rusage_distinguishes_gauge_from_lifetime_high_water(
    tmp_path: Path,
) -> None:
    """Enabled LinuxMemorySource + RUsageSource keep gauge current RSS distinct from RUSAGE peak."""

    gen = LinuxProcessGeneration(
        pid=os.getpid(),
        start_ticks=91000,
        boot_id="11111111-1111-4111-8111-111111111111",
        pid_ns_inode="pid:[4026531836]",
        host_id="host-o4",
    )
    lm_probe = _FakeLinuxMemoryProbe(
        [
            LinuxProcessMemoryReading(
                status="ok",
                rss_bytes=1 * 1024 * 1024 * 1024,
                virtual_bytes=4 * 1024 * 1024 * 1024,
                thread_count=4,
                read_start_ns=10,
                read_end_ns=20,
                target_generation=gen,
                observed_generation_before=gen,
                observed_generation_after=gen,
            ),
            LinuxProcessMemoryReading(
                status="ok",
                rss_bytes=2 * 1024 * 1024 * 1024,
                virtual_bytes=5 * 1024 * 1024 * 1024,
                thread_count=6,
                read_start_ns=30,
                read_end_ns=40,
                target_generation=gen,
                observed_generation_before=gen,
                observed_generation_after=gen,
            ),
        ],
        generation=gen,
    )
    rusage_probe = _FakeRUsageProbe(
        [
            ProcessResourceBaseline(1.0, 0.2, 0, 0, 1, 0, 5120.0),
            ProcessResourceBaseline(2.0, 0.4, 0, 0, 2, 0, 5120.0),
        ]
    )
    clock = _SteppingClock(start_ns=1_000_000_000_000, step_ns=1_000_000_000)
    cb = ResourceSamples(
        interval_seconds=60.0,
        max_pending_envelopes=16,
        rusage=RUsageSource(enabled=True, probe=rusage_probe),
        linux_memory=LinuxMemorySource(enabled=True, probe=lm_probe),
        monotonic_ns=clock,
    )
    assert lm_probe.bind_calls == 0
    assert lm_probe.read_calls == 0

    with open_run_context(_base_cfg(tmp_path), runtime_callbacks=[cb]) as ctx:
        run_dir = ctx.run_dir

    assert lm_probe.bind_calls == 1
    assert lm_probe.read_calls == 2

    bundle_dir = next((run_dir / "observability").iterdir())
    bundle = inspect_observability_bundle(bundle_dir)
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "complete"

    assert bundle.descriptors is not None
    lm_rss_desc = bundle.descriptors.quantities["process.memory.rss_bytes"]
    lm_virt_desc = bundle.descriptors.quantities["process.memory.virtual_bytes"]
    lm_thr_desc = bundle.descriptors.quantities["process.memory.thread_count"]
    ru_rss_desc = bundle.descriptors.quantities["process.rusage.max_rss_bytes"]

    assert lm_rss_desc.source_id == LINUX_MEMORY_SOURCE_ID
    assert lm_rss_desc.unit == "B"
    assert lm_rss_desc.statistic == "gauge"
    assert lm_rss_desc.subject_kind == "process"
    assert lm_rss_desc.window == "read_interval"
    assert lm_rss_desc.native_name == "VmRSS"
    assert lm_rss_desc.native_unit == "KiB"
    assert lm_rss_desc.conversion == "multiply_by_1024"
    assert lm_rss_desc.quality == ("approximate_rss",)

    assert lm_virt_desc.statistic == "gauge"
    assert lm_virt_desc.native_name == "VmSize"
    assert lm_virt_desc.native_unit == "KiB"
    assert lm_virt_desc.conversion == "multiply_by_1024"

    assert lm_thr_desc.unit == "1"
    assert lm_thr_desc.statistic == "gauge"
    assert lm_thr_desc.native_name == "Threads"
    assert lm_thr_desc.conversion == "identity"

    assert ru_rss_desc.statistic == "high_water"
    assert ru_rss_desc.window == "process_lifetime"

    # Inventory & sample subject record exact generation provenance.
    assert bundle.inventory is not None
    assert bundle.inventory.sources[LINUX_MEMORY_SOURCE_ID]["capability"] == "available"
    assert bundle.inventory.sources[LINUX_MEMORY_SOURCE_ID]["target_generation"] == gen.to_dict()
    assert bundle.inventory.process["start_ticks"] == 91000
    assert bundle.inventory.process["boot_id"] == gen.boot_id
    assert bundle.inventory.process["pid_ns_inode"] == gen.pid_ns_inode

    lm_samples = [
        s for s in bundle.samples["resource-samples"] if s.source_id == LINUX_MEMORY_SOURCE_ID
    ]
    assert len(lm_samples) == 2
    assert [s.sequence for s in lm_samples] == [0, 1]
    assert lm_samples[0].subject["kind"] == "process"
    assert lm_samples[0].subject["start_ticks"] == 91000
    assert lm_samples[0].subject["boot_id"] == gen.boot_id
    assert lm_samples[0].subject["pid_ns_inode"] == gen.pid_ns_inode

    # O2 summary keeps gauge (sampled_min/sampled_max lower bound) separate from high-water.
    lm_rss_summary = bundle.summary.quantity(
        LINUX_MEMORY_SOURCE_ID, "process.memory.rss_bytes"
    )
    ru_rss_summary = bundle.summary.quantity(
        RUSAGE_SOURCE_ID, "process.rusage.max_rss_bytes"
    )
    assert lm_rss_summary is not None
    assert lm_rss_summary.sampled_min == 1 * 1024 * 1024 * 1024
    assert lm_rss_summary.sampled_max == 2 * 1024 * 1024 * 1024
    assert lm_rss_summary.sampled_max_is_lower_bound is True
    assert lm_rss_summary.observed_max is None

    assert ru_rss_summary is not None
    assert ru_rss_summary.observed_max == 5120 * 1024 * 1024
    assert ru_rss_summary.sampled_max is None

    status = bundle.statuses["resource-samples"]
    assert status.source_status[LINUX_MEMORY_SOURCE_ID] == "available"
    assert status.source_status[RUSAGE_SOURCE_ID] == "available"
    assert status.counts == {"attempted": 4, "written": 4, "dropped": 0, "failed": 0}


def test_linux_memory_only_and_disabled_configurations(tmp_path: Path) -> None:
    """LinuxMemorySource-only omits RUSAGE; disabled configs emit zero linux-memory records."""

    proc_root = _write_fake_proc_tree(
        tmp_path / "proc",
        pid=os.getpid(),
        start_ticks=12345,
        vm_rss_kib=2048,
        vm_size_kib=8192,
        threads=3,
    )

    # 1. LinuxMemorySource-only (via YAML/mapping coercion)
    cb_only = ResourceSamples(
        interval_seconds=60.0,
        linux_memory={"enabled": True, "proc_root": str(proc_root)},
    )
    with open_run_context(_base_cfg(tmp_path / "only"), runtime_callbacks=[cb_only]) as ctx:
        only_dir = ctx.run_dir

    bundle_only = inspect_run_observability(only_dir).invocations[0]
    assert bundle_only.derived_lifecycle_state == "closed"
    assert bundle_only.derived_data_completeness == "complete"
    assert bundle_only.descriptors is not None
    assert set(bundle_only.descriptors.quantities.keys()) == set(LINUX_MEMORY_QUANTITY_IDS)
    assert RUSAGE_SOURCE_ID not in bundle_only.statuses["resource-samples"].source_status
    assert bundle_only.statuses["resource-samples"].source_status[LINUX_MEMORY_SOURCE_ID] == "available"
    assert bundle_only.statuses["resource-samples"].helper_cleanup["state"] == "confirmed_stopped"

    # 2. linux_memory.enabled=False alongside enabled rusage emits zero linux-memory records
    untouched_probe = _FakeLinuxMemoryProbe([])
    cb_lm_off = ResourceSamples(
        interval_seconds=60.0,
        rusage=RUsageSource(enabled=True),
        linux_memory=LinuxMemorySource(enabled=False, probe=untouched_probe),
    )
    with open_run_context(_base_cfg(tmp_path / "lm_off"), runtime_callbacks=[cb_lm_off]) as ctx:
        off_dir = ctx.run_dir
    assert untouched_probe.bind_calls == 0
    assert untouched_probe.read_calls == 0
    bundle_off = inspect_run_observability(off_dir).invocations[0]
    assert bundle_off.descriptors is not None
    assert not any(
        d.source_id == LINUX_MEMORY_SOURCE_ID
        for d in bundle_off.descriptors.quantities.values()
    )
    assert bundle_off.inventory is not None
    assert LINUX_MEMORY_SOURCE_ID not in bundle_off.inventory.sources
    assert all(
        s.source_id != LINUX_MEMORY_SOURCE_ID
        for s in bundle_off.samples["resource-samples"]
    )

    # 3. ResourceSamples(enabled=False, linux_memory=enabled) emits zero files or probe calls
    cb_all_off = ResourceSamples(
        enabled=False,
        linux_memory=LinuxMemorySource(enabled=True, probe=untouched_probe),
    )
    assert (
        run_from_config(
            _base_cfg(tmp_path / "all_off"),
            raise_exceptions=True,
            runtime_callbacks=[cb_all_off],
        )
        == 0
    )
    assert untouched_probe.bind_calls == 0
    assert untouched_probe.read_calls == 0
    assert inspect_run_observability(tmp_path / "all_off" / "outputs" / "o3-run").status == "absent"


def test_linux_memory_partial_unavailable_changed_generation_oversize_and_timeout_roundtrip(
    tmp_path: Path,
) -> None:
    """Partial, generation_changed, oversize, timeout, and non-Linux outcomes reconcile cleanly."""

    gen = LinuxProcessGeneration(
        pid=os.getpid(),
        start_ticks=55555,
        boot_id="22222222-2222-4222-8222-222222222222",
        pid_ns_inode="pid:[4026531836]",
        host_id="host-err",
    )
    changed_gen = LinuxProcessGeneration(
        pid=os.getpid(),
        start_ticks=66666,
        boot_id="22222222-2222-4222-8222-222222222222",
        pid_ns_inode="pid:[4026531836]",
        host_id="host-err",
    )
    field_missing = LinuxMemoryUnavailable(
        reason="VmSize missing", reason_code="field_missing", status="unavailable"
    )
    gen_changed = LinuxMemoryUnavailable(
        reason="start_ticks changed", reason_code="generation_changed", status="unavailable"
    )
    oversize = LinuxMemoryUnavailable(
        reason="status exceeded max_read_bytes",
        reason_code="input_limit_exceeded",
        status="error",
    )
    timeout = LinuxMemoryUnavailable(
        reason="status exceeded max_read_seconds",
        reason_code="read_timeout",
        status="error",
    )

    # Sequence covering partial -> generation_changed -> oversize -> timeout
    sequence_probe = _FakeLinuxMemoryProbe(
        [
            LinuxProcessMemoryReading(
                status="partial",
                rss_bytes=0,  # legitimate zero!
                virtual_bytes=field_missing,
                thread_count=2,
                read_start_ns=10,
                read_end_ns=15,
                reason_code="field_missing",
                detail="VmSize missing",
                target_generation=gen,
                observed_generation_before=gen,
                observed_generation_after=gen,
            ),
            LinuxProcessMemoryReading(
                status="unavailable",
                rss_bytes=gen_changed,
                virtual_bytes=gen_changed,
                thread_count=gen_changed,
                read_start_ns=20,
                read_end_ns=25,
                reason_code="generation_changed",
                detail="start_ticks changed",
                target_generation=gen,
                observed_generation_before=changed_gen,
                observed_generation_after=None,
            ),
            LinuxProcessMemoryReading(
                status="error",
                rss_bytes=oversize,
                virtual_bytes=oversize,
                thread_count=oversize,
                read_start_ns=30,
                read_end_ns=35,
                reason_code="input_limit_exceeded",
                detail="status exceeded max_read_bytes",
                target_generation=gen,
                observed_generation_before=gen,
                observed_generation_after=None,
            ),
            LinuxProcessMemoryReading(
                status="error",
                rss_bytes=timeout,
                virtual_bytes=timeout,
                thread_count=timeout,
                read_start_ns=40,
                read_end_ns=45,
                reason_code="read_timeout",
                detail="status exceeded max_read_seconds",
                target_generation=gen,
                observed_generation_before=gen,
                observed_generation_after=None,
            ),
        ],
        generation=gen,
        target_reads=4,
    )

    cb = ResourceSamples(
        interval_seconds=0.015,
        max_pending_envelopes=32,
        linux_memory=LinuxMemorySource(enabled=True, probe=sequence_probe),
    )
    with open_run_context(_base_cfg(tmp_path / "outcomes"), runtime_callbacks=[cb]) as ctx:
        assert sequence_probe.reached_target.wait(timeout=2.0)
        outcomes_run_dir = ctx.run_dir

    bundle_dir = next((outcomes_run_dir / "observability").iterdir())
    bundle = inspect_observability_bundle(bundle_dir)
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "partial"

    samples = bundle.samples["resource-samples"]
    assert len(samples) >= 4
    # Sample 0: partial (legitimate 0 ok + field_missing unavailable)
    assert samples[0].readings["process.memory.rss_bytes"].status == "ok"
    assert samples[0].readings["process.memory.rss_bytes"].value == 0
    assert samples[0].readings["process.memory.virtual_bytes"].status == "unavailable"
    assert samples[0].readings["process.memory.virtual_bytes"].reason_code == "field_missing"
    assert samples[0].readings["process.memory.virtual_bytes"].value is None
    # Sample 1: generation_changed
    assert samples[1].readings["process.memory.rss_bytes"].status == "unavailable"
    assert samples[1].readings["process.memory.rss_bytes"].reason_code == "generation_changed"
    assert samples[1].readings["process.memory.rss_bytes"].value is None
    # Sample 2: input_limit_exceeded
    assert samples[2].readings["process.memory.rss_bytes"].status == "error"
    assert samples[2].readings["process.memory.rss_bytes"].reason_code == "input_limit_exceeded"
    # Sample 3: read_timeout
    assert samples[3].readings["process.memory.rss_bytes"].status == "error"
    assert samples[3].readings["process.memory.rss_bytes"].reason_code == "read_timeout"

    status = bundle.statuses["resource-samples"]
    assert status.source_status[LINUX_MEMORY_SOURCE_ID] == "partial"
    assert status.counts["failed"] == 0
    assert status.counts["attempted"] == (
        status.counts["written"] + status.counts["dropped"] + status.counts["failed"]
    )
    assert status.counts["written"] == len(samples)

    # Non-Linux platform probe: inventory capability='unavailable', samples typed unavailable.
    darwin_probe = LinuxProcessMemoryProbe(platform_name="darwin")
    cb_darwin = ResourceSamples(
        interval_seconds=60.0,
        linux_memory=LinuxMemorySource(enabled=True, probe=darwin_probe),
    )
    with open_run_context(_base_cfg(tmp_path / "darwin"), runtime_callbacks=[cb_darwin]) as ctx:
        darwin_dir = ctx.run_dir

    darwin_bundle = inspect_run_observability(darwin_dir).invocations[0]
    assert darwin_bundle.derived_lifecycle_state == "closed"
    assert darwin_bundle.inventory is not None
    assert darwin_bundle.inventory.sources[LINUX_MEMORY_SOURCE_ID]["capability"] == "unavailable"
    assert (
        darwin_bundle.inventory.sources[LINUX_MEMORY_SOURCE_ID]["unavailable_reason_code"]
        == "unsupported_platform"
    )
    darwin_status = darwin_bundle.statuses["resource-samples"]
    assert darwin_status.source_status[LINUX_MEMORY_SOURCE_ID] == "unavailable"
    assert darwin_status.counts == {"attempted": 2, "written": 2, "dropped": 0, "failed": 0}
    first_darwin = darwin_bundle.samples["resource-samples"][0]
    for qty_id in LINUX_MEMORY_QUANTITY_IDS:
        assert first_darwin.readings[qty_id].status == "unavailable"
        assert first_darwin.readings[qty_id].reason_code == "unsupported_platform"
        assert first_darwin.readings[qty_id].value is None


def test_blocking_linux_memory_worker_marks_bundle_unsafe_and_refuses_reuse(
    tmp_path: Path,
) -> None:
    """A hung linux-memory thread respects close_timeout_seconds, marks unsafe, and blocks reuse."""

    gen = LinuxProcessGeneration(pid=os.getpid(), start_ticks=12345)
    probe = _BlockingLinuxMemoryProbe(gen)
    cb = ResourceSamples(
        interval_seconds=0.01,
        close_timeout_seconds=0.08,
        linux_memory=LinuxMemorySource(enabled=True, probe=probe),
    )
    try:
        with pytest.raises(ManagedLifecycleError, match="quiescence"):
            with open_run_context(_base_cfg(tmp_path / "hung"), runtime_callbacks=[cb]) as ctx:
                assert probe.blocked_in_second.wait(timeout=2.0)
                hung_dir = ctx.run_dir

        bundle = inspect_run_observability(hung_dir).invocations[0]
        assert bundle.derived_lifecycle_state == "unsafe"
        assert bundle.derived_finalization == "unsafe"
        status = bundle.statuses["resource-samples"]
        assert status.helper_cleanup["state"] == "unconfirmed_timeout"
        assert "close_timeout" in status.truncation_reasons
        assert cb.is_managed_scope_active() is True
    finally:
        probe.release.set()


@pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux /proc filesystem")
def test_linux_memory_live_procfs_roundtrip_with_rusage_on_linux(tmp_path: Path) -> None:
    """On Linux (Cannon), live LinuxMemorySource + RUsageSource round-trip through O2 cleanly."""

    cb = ResourceSamples(
        interval_seconds=60.0,
        max_pending_envelopes=16,
        rusage=RUsageSource(enabled=True),
        linux_memory=LinuxMemorySource(enabled=True),
    )
    assert (
        run_from_config(_base_cfg(tmp_path), raise_exceptions=True, runtime_callbacks=[cb])
        == 0
    )
    bundle = inspect_run_observability(tmp_path / "outputs" / "o3-run").invocations[0]
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "complete"
    assert bundle.inventory is not None
    assert bundle.inventory.sources[LINUX_MEMORY_SOURCE_ID]["capability"] == "available"
    assert bundle.inventory.process["pid"] == os.getpid()
    assert bundle.inventory.process["start_ticks"] > 0

    lm_summary = bundle.summary.quantity(
        LINUX_MEMORY_SOURCE_ID, "process.memory.rss_bytes"
    )
    assert lm_summary is not None
    assert lm_summary.ok_count >= 2
    assert lm_summary.sampled_min is not None and lm_summary.sampled_min > 0
    assert lm_summary.sampled_max is not None and lm_summary.sampled_max >= lm_summary.sampled_min
    assert lm_summary.sampled_max_is_lower_bound is True

