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
from tpen.process_resources import ProcessResourceBaseline, ResourceUnavailable
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
