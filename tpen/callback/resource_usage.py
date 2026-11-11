"""Best-effort process and configured-accelerator resource callback."""

from __future__ import annotations

import math
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.accelerator import (
    AcceleratorKind,
    AllocatorPeakProbe,
    AllocatorUnavailable,
    AllocatorUsage,
    TorchAllocatorPeakProbe,
)
from tpen.dependencies import OptionalDependencyError
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tpen.process_resources import (
    ProcessRUsageProbe,
    ProcessResourceResult,
    ResourceUnavailable,
)
from tpen.distributed import ProfileRecord, ProfileScope

from .base import Callback
from .cadence import SubscriptionGroup


def _default_peak_rss_mb() -> float:
    """Return process peak RSS in MiB for compatibility callers."""

    value = ProcessRUsageProbe().read().peak_rss_mb
    if isinstance(value, ResourceUnavailable):
        raise OSError(value.reason)
    return float(value)


class ResourceUsage(Callback):
    """Log run-level peak process and configured-device memory.

    Allocator peak counters are reset when the run starts so the logged peaks
    cover exactly this run. Readings are best-effort runtime metadata: a failing
    reader emits typed-unavailable flags instead of failing the run.

    Data-free, so a plain `tpen.callback.Callback`: it reads process and
    configured-device counters, never any domain state.

    Notes
    -----
    On a failed run this used to log ``runtime`` TWICE. Its shipped default
    ``triggers`` answered both ``run_failed`` and ``exception``, which
    `tpen.run` emits on consecutive lines with the same payload, and
    ``on_run_failed`` was an alias calling the same method as ``on_exception``;
    ``pair_stability.yaml`` configured both names explicitly. One typed
    `tpen.run_events.RunFailed` makes the duplicate structurally impossible. No
    metric name changes and no series disappears -- one record per failed run
    replaces two identical ones.

    These metrics are logger-owned; they are not an inter-callback contract.

    Parameters
    ----------
    peak_rss_mb_reader : callable, optional
        Override returning the process peak RSS in MiB, for tests.
    **kwargs
        Forwarded to `tpen.callback.Callback`.
    """

    def __init__(
        self,
        *,
        peak_rss_mb_reader: Callable[[], float] | None = None,
        process_probe: ProcessRUsageProbe | None = None,
        allocator_probe: AllocatorPeakProbe | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(RunStarted),
                        Subscription.of(RunCompleted),
                        Subscription.of(RunFailed),
                    )
                ),
            ),
            **kwargs,
        )
        self.process_probe = ProcessRUsageProbe() if process_probe is None else process_probe
        self.peak_rss_mb_reader = peak_rss_mb_reader
        self.allocator_probe = allocator_probe
        self._injected_allocator_probe = allocator_probe is not None
        self._allocator_reset_failure: AllocatorUnavailable | None = None
        self._process_baseline = None
        self._reported = False

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Reset allocator peaks at the start and report them once at either end."""

        event = occurrence.event
        if isinstance(event, RunStarted):
            self._process_baseline = self.process_probe.read()
            self._reported = False
            if not self._injected_allocator_probe:
                # A context-derived probe is scoped to one run: recreate it on
                # every RunStarted rather than reusing the first run's probe,
                # which would stay bound to that run's context and device.
                configured_device = context.metadata.device
                try:
                    self.allocator_probe = TorchAllocatorPeakProbe(configured_device)
                except OptionalDependencyError:
                    self.allocator_probe = None
            if self.allocator_probe is not None:
                reset_result = self.allocator_probe.reset()
                self._allocator_reset_failure = (
                    reset_result if isinstance(reset_result, AllocatorUnavailable) else None
                )
            return
        # Both terminal boundaries report the same peaks; a failed run has no
        # different memory story to tell, which is why they share one path.
        # A run reaches at most one of these boundaries -- except when a later
        # callback raises while handling `RunCompleted`, which the harness
        # then reports as a `RunFailed` on the same context. `_reported` makes
        # that one logical run answer once regardless of how many terminal
        # events it produces.
        if isinstance(event, (RunCompleted, RunFailed)) and not self._reported:
            self._reported = True
            self._log_peaks(context)

    def _log_peaks(self, context: RunContext) -> None:
        metrics: dict[str, Any] = {}
        process_metrics: dict[str, Any] = {}
        result = None
        if self._process_baseline is not None:
            result = self.process_probe.result(self._process_baseline)
            process_metrics.update(_process_metrics(result))
            if isinstance(result.peak_rss_mb, ResourceUnavailable):
                process_metrics["process_peak_rss_unavailable"] = True
        if self.peak_rss_mb_reader is not None:
            try:
                _set_peak_memory_mb(metrics, float(self.peak_rss_mb_reader()))
            except OSError:
                pass
        elif result is not None:
            if not isinstance(result.peak_rss_mb, ResourceUnavailable):
                _set_peak_memory_mb(metrics, float(result.peak_rss_mb))
        else:
            try:
                _set_peak_memory_mb(metrics, float(_default_peak_rss_mb()))
            except OSError:
                pass
        if self.allocator_probe is not None:
            allocator = self.allocator_probe.read()
            if self._allocator_reset_failure is not None:
                allocator = AllocatorUsage(
                    identity=allocator.identity,
                    allocated_mb=self._allocator_reset_failure,
                    reserved_mb=self._allocator_reset_failure,
                    device_count=allocator.device_count,
                )
            if allocator.identity.kind is not AcceleratorKind.CPU:
                metrics.update(_allocator_metrics(allocator))
            if context.profile_writer is not None and context.topology is not None:
                try:
                    context.write_profile(
                        ProfileRecord(
                            scope=ProfileScope.DEVICE,
                            monotonic_time=context.monotonic_clock(),
                            topology=context.topology,
                            device=allocator,
                        )
                    )
                except OSError:
                    # Persistence is best-effort: a failed durable write must
                    # not suppress the terminal logs below or fail an
                    # otherwise-successful run.
                    pass
        if result is not None and context.profile_writer is not None and context.topology is not None:
            try:
                context.write_profile(
                    ProfileRecord(
                        scope=ProfileScope.PROCESS,
                        monotonic_time=context.monotonic_clock(),
                        topology=context.topology,
                        process=result,
                    )
                )
            except OSError:
                pass
        if metrics:
            context.log(metrics, step=0, namespace="runtime")
        if process_metrics:
            context.log(process_metrics, step=0, namespace="process")


def _is_unavailable_scalar(value: object) -> bool:
    """Return True for typed-unavailable evidence or a non-finite reading.

    A non-finite float (e.g. a driver glitch) cannot cross the strict JSON
    boundary; degrading it here keeps the other, finite readings intact.
    """

    if value is None or isinstance(value, (AllocatorUnavailable, ResourceUnavailable)):
        return True
    return isinstance(value, float) and not math.isfinite(value)


def _set_peak_memory_mb(metrics: dict[str, Any], value: float) -> None:
    """Set the peak-memory terminal metric unless the reading is non-finite.

    Mirrors the OSError arms this is always called alongside: a bad reading
    is dropped silently rather than reaching the strict JSON boundary and
    aborting the rest of the terminal record.
    """

    if not _is_unavailable_scalar(value):
        metrics["peak_memory_mb"] = value


def _allocator_metrics(usage: AllocatorUsage) -> dict[str, Any]:
    """Project one allocator record to backend-neutral and legacy keys."""

    metrics: dict[str, Any] = {}
    if _is_unavailable_scalar(usage.allocated_mb):
        metrics["accelerator_max_memory_allocated_unavailable"] = True
    else:
        metrics["accelerator_max_memory_allocated_mb"] = usage.allocated_mb
        if usage.identity.kind in (AcceleratorKind.CUDA, AcceleratorKind.ROCM):
            # ROCm exposes the torch.cuda-compatible allocator API, so these
            # legacy aliases remain valid for ROCm while OTHER backends do not.
            metrics["cuda_max_memory_allocated_mb"] = usage.allocated_mb
    if _is_unavailable_scalar(usage.reserved_mb):
        metrics["accelerator_max_memory_reserved_unavailable"] = True
    else:
        metrics["accelerator_max_memory_reserved_mb"] = usage.reserved_mb
        if usage.identity.kind in (AcceleratorKind.CUDA, AcceleratorKind.ROCM):
            metrics["cuda_max_memory_reserved_mb"] = usage.reserved_mb
    if _is_unavailable_scalar(usage.device_count):
        metrics["accelerator_device_count_unavailable"] = True
    else:
        metrics["accelerator_device_count"] = usage.device_count
        if usage.identity.kind in (AcceleratorKind.CUDA, AcceleratorKind.ROCM):
            metrics["cuda_device_count"] = usage.device_count
    return metrics


def _process_metrics(result: ProcessResourceResult) -> dict[str, Any]:
    """Project typed process results onto the logger's mapping boundary."""

    names = (
        ("process_user_cpu_seconds", result.user_cpu_seconds),
        ("process_system_cpu_seconds", result.system_cpu_seconds),
        ("process_read_block_operations", result.read_block_operations),
        ("process_write_block_operations", result.write_block_operations),
        ("process_voluntary_context_switches", result.voluntary_context_switches),
        ("process_involuntary_context_switches", result.involuntary_context_switches),
    )
    metrics: dict[str, Any] = {}
    for name, value in names:
        if _is_unavailable_scalar(value):
            metrics[f"{name}_unavailable"] = True
        else:
            metrics[name] = float(value)
    return metrics


__all__ = ["ResourceUsage"]
