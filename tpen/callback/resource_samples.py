"""Opt-in managed callback for single-process RUSAGE, Linux memory, and Torch allocator samples.

This module implements TPEN's ``tpen.run-observability/v1`` single-process
resource observability sources:

* :class:`RUsageSource` — self-process ``RUSAGE_SELF`` readings adapted from
  :class:`tpen.process_resources.ProcessRUsageProbe`, running strictly
  in-process on a bounded TPEN-owned sampling thread (never relocated to a
  helper subprocess or targeted at a foreign PID).
* :class:`LinuxMemorySource` — self-process Linux ``/proc/<pid>/status``
  current-memory gauges (``VmRSS``, ``VmSize``, ``Threads``) adapted from
  :class:`tpen.process_resources.LinuxProcessMemoryProbe` with exact target
  generation / PID-reuse defense and bounded input/time limits, running
  in-process on a bounded TPEN-owned sampling thread.
* :class:`AllocatorSource` — configured-device PyTorch allocator current and
  peak readings (:class:`tpen.accelerator.TorchAllocatorCurrentProbe` and
  :class:`tpen.accelerator.TorchAllocatorPeakProbe`), deferred strictly until
  scope activation, running strictly cooperatively on the owning thread via
  :class:`tpen.callback.base.RunResource.checkpoint` with explicit peak-reset
  epoch ownership.
* :class:`ResourceSamples` — opt-in :class:`tpen.callback.base.Callback`
  implementing the core-owned ``run_scope(context)`` protocol, bounded
  nonblocking queueing, single-deadline teardown, and pure routed persistence
  through :class:`tpen.observability_artifacts.ObservabilityBundleWriter`.

This module is strictly Torch-free at import time and at construction time, and
imports nothing from ``experiments`` or ``tpen_executor``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, ContextManager

from tpen.accelerator import (
    AcceleratorIdentity,
    AcceleratorKind,
    DEFAULT_NVIDIA_SMI_EXECUTABLE,
    DEFAULT_NVIDIA_SMI_MAX_OUTPUT_BYTES,
    DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS,
    MAX_NVIDIA_SMI_TIMEOUT_SECONDS,
    NvidiaDeviceReading,
    NvidiaDeviceUnavailable,
    NvidiaSmiDeviceProbe,
    canonicalize_nvidia_device_uuid,
)
from tpen.artifacts import ArtifactRoutingError
from tpen.callback.base import (
    Callback,
    CallbackContext,
    ManagedLifecycleError,
    ResourceOwnershipClaim,
    RunResource,
    canonicalize_allocator_reset_scope,
)
from tpen.callback.cadence import SubscriptionGroup
from tpen.checkpoint.hashing import resolved_config_hash
from tpen.dependencies import OptionalDependencyError
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.observability_artifacts import (
    ObservabilityArtifactError,
    ObservabilityBundleWriter,
    ObservabilityDescriptorSet,
    ObservabilityInventory,
    ObservabilityManifest,
    ObservabilityPathError,
    ObservabilityReading,
    ObservabilitySample,
    ObservabilitySchemaError,
    ObservabilityStatus,
    QuantityDescriptor,
    RunArtifactSink,
    validate_identifier,
)
from tpen.process_resources import (
    DEFAULT_LINUX_CGROUP_MAX_ANCESTOR_DEPTH,
    DEFAULT_LINUX_CGROUP_MAX_MOUNTINFO_BYTES,
    DEFAULT_LINUX_CGROUP_MAX_READ_BYTES,
    DEFAULT_LINUX_CGROUP_MAX_READ_SECONDS,
    DEFAULT_LINUX_MEMORY_MAX_READ_BYTES,
    DEFAULT_LINUX_MEMORY_MAX_READ_SECONDS,
    DEFAULT_LINUX_TREE_MAX_ENUMERATED_CANDIDATES,
    DEFAULT_LINUX_TREE_MAX_OBSERVED_MEMBERS,
    DEFAULT_LINUX_TREE_MAX_READ_BYTES,
    DEFAULT_LINUX_TREE_MAX_SCAN_SECONDS,
    LinuxCgroupV2Identity,
    LinuxCgroupV2Probe,
    LinuxCgroupV2Reading,
    LinuxMemoryUnavailable,
    LinuxProcessGeneration,
    LinuxProcessMemoryProbe,
    LinuxProcessMemoryReading,
    LinuxProcessTreeProbe,
    LinuxProcessTreeReading,
    ProcessRUsageProbe,
    ProcessResourceBaseline,
    ResourceUnavailable,
)
from tpen.run_events import RunCompleted, RunFailed

if TYPE_CHECKING:
    from tpen.artifacts import ArtifactManager, RunContext


_BYTES_PER_MIB = 1024 * 1024
_NS_PER_SECOND = 1_000_000_000
_DEFAULT_DESCENDANT_SAMPLE_RECORD_BUDGET_BYTES = 262_144

RUSAGE_SOURCE_ID = "self-rusage"
ALLOCATOR_SOURCE_ID = "torch-allocator"
LINUX_MEMORY_SOURCE_ID = "linux-memory"
LINUX_PROCESS_TREE_SOURCE_ID = "linux-process-tree"
LINUX_CGROUP_V2_SOURCE_ID = "linux-cgroup-v2"
NVIDIA_DEVICE_SOURCE_ID = "nvidia-smi-device"

RUSAGE_EXECUTION_MODE = "in_process_thread"
ALLOCATOR_EXECUTION_MODE = "in_process_cooperative"
LINUX_MEMORY_EXECUTION_MODE = "in_process_thread"
LINUX_PROCESS_TREE_EXECUTION_MODE = "in_process_thread"
LINUX_CGROUP_V2_EXECUTION_MODE = "in_process_thread"
NVIDIA_DEVICE_EXECUTION_MODE = "supervised_subprocess"

RUSAGE_QUANTITY_IDS: tuple[str, ...] = (
    "process.rusage.user_cpu_seconds",
    "process.rusage.system_cpu_seconds",
    "process.rusage.read_block_operations",
    "process.rusage.write_block_operations",
    "process.rusage.voluntary_context_switches",
    "process.rusage.involuntary_context_switches",
    "process.rusage.max_rss_bytes",
)
DEFAULT_RUSAGE_QUANTITIES: tuple[str, ...] = RUSAGE_QUANTITY_IDS

LINUX_MEMORY_QUANTITY_IDS: tuple[str, ...] = (
    "process.memory.rss_bytes",
    "process.memory.virtual_bytes",
    "process.memory.thread_count",
)
DEFAULT_LINUX_MEMORY_QUANTITIES: tuple[str, ...] = LINUX_MEMORY_QUANTITY_IDS

LINUX_PROCESS_TREE_QUANTITY_IDS: tuple[str, ...] = (
    "descendants.live_member_count",
    "descendants.sum_rss_bytes",
    "descendants.matched_cpu_user_delta_seconds",
    "descendants.matched_cpu_system_delta_seconds",
)
DEFAULT_LINUX_PROCESS_TREE_QUANTITIES: tuple[str, ...] = LINUX_PROCESS_TREE_QUANTITY_IDS

LINUX_CGROUP_V2_QUANTITY_IDS: tuple[str, ...] = (
    "cgroup.memory.current_bytes",
    "cgroup.memory.peak_bytes",
    "cgroup.memory.max_bytes",
    "cgroup.cpu.usage_seconds",
    "cgroup.cpu.user_seconds",
    "cgroup.cpu.system_seconds",
    "cgroup.cpu.throttled_seconds",
    "cgroup.cpu.nr_throttled",
    "cgroup.cpu.max_cores",
    "cgroup.cpuset.effective_cpus",
)
DEFAULT_LINUX_CGROUP_V2_QUANTITIES: tuple[str, ...] = LINUX_CGROUP_V2_QUANTITY_IDS

ALLOCATOR_CURRENT_QUANTITY_IDS: tuple[str, ...] = (
    "allocator.current.allocated_bytes",
    "allocator.current.reserved_bytes",
)
ALLOCATOR_PEAK_QUANTITY_IDS: tuple[str, ...] = (
    "allocator.peak.allocated_bytes",
    "allocator.peak.reserved_bytes",
)

NVIDIA_DEVICE_QUANTITY_IDS: tuple[str, ...] = (
    "device.framebuffer.used_bytes",
    "device.framebuffer.total_bytes",
    "device.busy_fraction",
)
DEFAULT_NVIDIA_DEVICE_QUANTITIES: tuple[str, ...] = NVIDIA_DEVICE_QUANTITY_IDS

_FORBIDDEN_SELECTOR_KWARGS: frozenset[str] = frozenset(
    {
        "typed_groups",
        "selectors",
        "subscriptions",
        "triggers",
        "event_selector",
        "event_selectors",
        "event_type",
        "event_types",
        "operation",
        "operation_selector",
        "operation_selectors",
        "operation_type",
        "operation_types",
    }
)

_CLOCK_DOMAIN_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------------------
# Strict validation helpers
# ---------------------------------------------------------------------------


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return number


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a positive integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _reject_forbidden_selector_keys(mapping: Mapping[str, Any], *, owner: str) -> None:
    """Refuse free-form event/type selector keys in Python or dict configuration."""

    for raw_key, value in mapping.items():
        key = str(raw_key)
        if key in _FORBIDDEN_SELECTOR_KWARGS:
            raise ValueError(
                f"{owner}.{key} is forbidden: {owner} owns its typed subscriptions "
                "in code and must not accept free-form event/type selectors"
            )
        if isinstance(value, Mapping):
            _reject_forbidden_selector_keys(value, owner=f"{owner}.{key}")


def _sanitize_clock_domain_id(host_id: str) -> str:
    cleaned = _CLOCK_DOMAIN_SANITIZE_RE.sub("-", host_id).strip(".-_")
    prefix = cleaned[:96] if cleaned else "host"
    return f"{prefix}-monotonic-ns"


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _is_allocator_unavailable(value: object) -> bool:
    """Detect ``tpen.accelerator.AllocatorUnavailable`` without importing torch."""

    return (
        type(value).__name__ == "AllocatorUnavailable"
        and isinstance(getattr(value, "reason", None), str)
    )


def _identity_kind_text(identity: object) -> str:
    kind = getattr(identity, "kind", None)
    raw = getattr(kind, "value", kind)
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    return "unknown"


def _probe_is_available(probe: Any, fallback_identity: Any) -> bool:
    """Return whether an allocator probe reports an available non-CPU backend."""

    if probe is None:
        return False
    for attr in ("available", "_available"):
        fn = getattr(probe, attr, None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
    probe_identity = fallback_identity
    id_fn = getattr(probe, "identity", None)
    if callable(id_fn):
        try:
            probe_identity = id_fn()
        except Exception:
            probe_identity = fallback_identity
    return _identity_kind_text(probe_identity) != "cpu"


# ---------------------------------------------------------------------------
# Typed source configuration dataclasses (pure & torch-free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RUsageSource:
    """Configuration for self-process ``RUSAGE_SELF`` periodic sampling.

    ``RUSAGE_SELF`` measures the current process (including its threads and
    observer work, excluding child processes) and reports process-lifetime
    cumulative counters plus the process-lifetime RSS high-water mark (distinct
    from current RSS). It must execute in-process on a TPEN-owned thread and
    never in a helper subprocess or against a foreign PID.

    Parameters
    ----------
    enabled : bool, optional
        Whether this source is active (default ``True``).
    source_id : str, optional
        Stable source identifier written into descriptors and samples.
    execution_mode : str, optional
        Must be ``"in_process_thread"``. Helper/subprocess execution modes are
        explicitly rejected because ``RUSAGE_SELF`` in a helper would measure
        the helper rather than the TPEN process.
    interval_seconds : float or None, optional
        Optional per-source sampling interval in seconds. ``None`` inherits
        :attr:`ResourceSamples.interval_seconds`.
    quantities : sequence of str, optional
        Subset of :data:`RUSAGE_QUANTITY_IDS` to collect.
    pid : int or None, optional
        Optional explicit PID assertion. Only ``None`` or the current process
        PID (``os.getpid()``) is accepted; any foreign PID is refused.
    probe : ProcessRUsageProbe or object or None, optional
        Optional injected probe exposing callable ``read()`` for tests.
    """

    enabled: bool = True
    source_id: str = RUSAGE_SOURCE_ID
    execution_mode: str = RUSAGE_EXECUTION_MODE
    interval_seconds: float | None = None
    quantities: tuple[str, ...] = DEFAULT_RUSAGE_QUANTITIES
    pid: int | None = None
    probe: Any = None

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "RUsageSource.enabled")
        validate_identifier(self.source_id, "RUsageSource.source_id")
        if type(self.execution_mode) is not str or self.execution_mode != RUSAGE_EXECUTION_MODE:
            raise ValueError(
                f"RUsageSource.execution_mode must be {RUSAGE_EXECUTION_MODE!r}, "
                f"got {self.execution_mode!r}: RUSAGE_SELF must execute in-process "
                "and never in a helper subprocess"
            )
        if self.interval_seconds is not None:
            object.__setattr__(
                self,
                "interval_seconds",
                _require_positive_finite(
                    self.interval_seconds, "RUsageSource.interval_seconds"
                ),
            )
        if isinstance(self.quantities, (str, bytes)) or not isinstance(
            self.quantities, Sequence
        ):
            raise TypeError("RUsageSource.quantities must be a sequence of quantity IDs")
        selected = tuple(self.quantities)
        if not selected:
            raise ValueError("RUsageSource.quantities must be nonempty")
        seen: set[str] = set()
        allowed = frozenset(RUSAGE_QUANTITY_IDS)
        for qty_id in selected:
            if type(qty_id) is not str or qty_id not in allowed:
                raise ValueError(
                    f"unsupported RUsageSource quantity {qty_id!r}; "
                    f"expected subset of {RUSAGE_QUANTITY_IDS}"
                )
            if qty_id in seen:
                raise ValueError(f"duplicate RUsageSource quantity {qty_id!r}")
            seen.add(qty_id)
        object.__setattr__(self, "quantities", selected)

        if self.pid is not None:
            pid = _require_positive_int(self.pid, "RUsageSource.pid")
            current_pid = os.getpid()
            if pid != current_pid:
                raise ValueError(
                    f"RUsageSource uses RUSAGE_SELF and cannot target foreign PID "
                    f"{pid} (current process PID is {current_pid})"
                )
        if self.probe is not None and not callable(getattr(self.probe, "read", None)):
            raise TypeError("RUsageSource.probe must expose callable read()")

    def descriptors(self) -> dict[str, QuantityDescriptor]:
        """Return immutable :class:`QuantityDescriptor` map for selected quantities."""

        catalog: dict[str, QuantityDescriptor] = {
            "process.rusage.user_cpu_seconds": QuantityDescriptor(
                source_id=self.source_id,
                unit="s",
                statistic="counter",
                subject_kind="process",
                window="process_lifetime",
                native_name="ru_utime",
                native_unit="s",
                conversion="identity",
            ),
            "process.rusage.system_cpu_seconds": QuantityDescriptor(
                source_id=self.source_id,
                unit="s",
                statistic="counter",
                subject_kind="process",
                window="process_lifetime",
                native_name="ru_stime",
                native_unit="s",
                conversion="identity",
            ),
            "process.rusage.read_block_operations": QuantityDescriptor(
                source_id=self.source_id,
                unit="1",
                statistic="counter",
                subject_kind="process",
                window="process_lifetime",
                native_name="ru_inblock",
                native_unit="1",
                conversion="identity",
            ),
            "process.rusage.write_block_operations": QuantityDescriptor(
                source_id=self.source_id,
                unit="1",
                statistic="counter",
                subject_kind="process",
                window="process_lifetime",
                native_name="ru_oublock",
                native_unit="1",
                conversion="identity",
            ),
            "process.rusage.voluntary_context_switches": QuantityDescriptor(
                source_id=self.source_id,
                unit="1",
                statistic="counter",
                subject_kind="process",
                window="process_lifetime",
                native_name="ru_nvcsw",
                native_unit="1",
                conversion="identity",
            ),
            "process.rusage.involuntary_context_switches": QuantityDescriptor(
                source_id=self.source_id,
                unit="1",
                statistic="counter",
                subject_kind="process",
                window="process_lifetime",
                native_name="ru_nivcsw",
                native_unit="1",
                conversion="identity",
            ),
            "process.rusage.max_rss_bytes": QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="high_water",
                subject_kind="process",
                window="process_lifetime",
                native_name="peak_rss_mb",
                native_unit="MiB",
                conversion="multiply_by_1048576",
                quality=("process_lifetime_high_water_not_current_rss",),
            ),
        }
        return {qty_id: catalog[qty_id] for qty_id in self.quantities}


@dataclass(frozen=True)
class LinuxMemorySource:
    """Configuration for Linux self-process current-memory procfs sampling.

    Reads ``VmRSS``, ``VmSize``, and ``Threads`` from ``/proc/<pid>/status``
    while bracketing every read with exact target-generation verification
    (``pid``, ``start_ticks``, ``boot_id``, ``pid_ns_inode``) to defend against
    PID reuse. Unlike ``RUsageSource`` (which reports process-lifetime RSS
    high-water marks), ``LinuxMemorySource`` reports current-memory gauges
    (``statistic="gauge"``, ``window="read_interval"``) with explicit
    ``"approximate_rss"`` quality provenance.

    Construction is strictly pure and Torch-free: it performs zero filesystem
    I/O and never freezes ``os.getpid()`` when ``pid=None``.

    Parameters
    ----------
    enabled : bool, optional
        Whether this source is active (default ``True``).
    source_id : str, optional
        Stable source identifier written into descriptors and samples.
    execution_mode : str, optional
        Must be ``"in_process_thread"``.
    interval_seconds : float or None, optional
        Optional per-source sampling interval in seconds. ``None`` inherits
        :attr:`ResourceSamples.interval_seconds`.
    quantities : sequence of str, optional
        Subset of :data:`LINUX_MEMORY_QUANTITY_IDS` to collect.
    max_read_bytes : int, optional
        Maximum bytes allowed per procfs file read (default ``65536``).
    max_read_seconds : float, optional
        Maximum monotonic seconds allowed per procfs read (default ``1.0``).
    pid : int or None, optional
        Optional explicit PID assertion. Only ``None`` or the current process
        PID (``os.getpid()``) is accepted; any foreign PID is refused.
    probe : LinuxProcessMemoryProbe or object or None, optional
        Optional injected probe exposing callable ``read()`` for tests.
    proc_root : str or pathlib.Path or None, optional
        Optional procfs root directory override for deterministic tests.
    """

    enabled: bool = True
    source_id: str = LINUX_MEMORY_SOURCE_ID
    execution_mode: str = LINUX_MEMORY_EXECUTION_MODE
    interval_seconds: float | None = None
    quantities: tuple[str, ...] = DEFAULT_LINUX_MEMORY_QUANTITIES
    max_read_bytes: int = DEFAULT_LINUX_MEMORY_MAX_READ_BYTES
    max_read_seconds: float = DEFAULT_LINUX_MEMORY_MAX_READ_SECONDS
    pid: int | None = None
    probe: Any = None
    proc_root: str | Path | None = None

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "LinuxMemorySource.enabled")
        validate_identifier(self.source_id, "LinuxMemorySource.source_id")
        if (
            type(self.execution_mode) is not str
            or self.execution_mode != LINUX_MEMORY_EXECUTION_MODE
        ):
            raise ValueError(
                f"LinuxMemorySource.execution_mode must be "
                f"{LINUX_MEMORY_EXECUTION_MODE!r}, got {self.execution_mode!r}"
            )
        if self.interval_seconds is not None:
            object.__setattr__(
                self,
                "interval_seconds",
                _require_positive_finite(
                    self.interval_seconds, "LinuxMemorySource.interval_seconds"
                ),
            )
        if isinstance(self.quantities, (str, bytes)) or not isinstance(
            self.quantities, Sequence
        ):
            raise TypeError(
                "LinuxMemorySource.quantities must be a sequence of quantity IDs"
            )
        selected = tuple(self.quantities)
        if not selected:
            raise ValueError("LinuxMemorySource.quantities must be nonempty")
        seen: set[str] = set()
        allowed = frozenset(LINUX_MEMORY_QUANTITY_IDS)
        for qty_id in selected:
            if type(qty_id) is not str or qty_id not in allowed:
                raise ValueError(
                    f"unsupported LinuxMemorySource quantity {qty_id!r}; "
                    f"expected subset of {LINUX_MEMORY_QUANTITY_IDS}"
                )
            if qty_id in seen:
                raise ValueError(f"duplicate LinuxMemorySource quantity {qty_id!r}")
            seen.add(qty_id)
        object.__setattr__(self, "quantities", selected)

        _require_positive_int(self.max_read_bytes, "LinuxMemorySource.max_read_bytes")
        object.__setattr__(
            self,
            "max_read_seconds",
            _require_positive_finite(
                self.max_read_seconds, "LinuxMemorySource.max_read_seconds"
            ),
        )

        if self.pid is not None:
            pid = _require_positive_int(self.pid, "LinuxMemorySource.pid")
            current_pid = os.getpid()
            if pid != current_pid:
                raise ValueError(
                    f"LinuxMemorySource targets only the current self process and "
                    f"cannot target foreign PID {pid} (current process PID is {current_pid})"
                )
        if self.probe is not None and not callable(getattr(self.probe, "read", None)):
            raise TypeError("LinuxMemorySource.probe must expose callable read()")
        if self.proc_root is not None:
            if isinstance(self.proc_root, Path):
                pass
            elif type(self.proc_root) is str:
                if not self.proc_root.strip():
                    raise ValueError(
                        "LinuxMemorySource.proc_root must be a nonempty path or None"
                    )
                object.__setattr__(self, "proc_root", Path(self.proc_root))
            else:
                raise TypeError(
                    "LinuxMemorySource.proc_root must be str, Path, or None, "
                    f"got {type(self.proc_root).__name__}"
                )

    def descriptors(self) -> dict[str, QuantityDescriptor]:
        """Return immutable :class:`QuantityDescriptor` map for selected quantities."""

        catalog: dict[str, QuantityDescriptor] = {
            "process.memory.rss_bytes": QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="gauge",
                subject_kind="process",
                window="read_interval",
                native_name="VmRSS",
                native_unit="KiB",
                conversion="multiply_by_1024",
                quality=("approximate_rss",),
            ),
            "process.memory.virtual_bytes": QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="gauge",
                subject_kind="process",
                window="read_interval",
                native_name="VmSize",
                native_unit="KiB",
                conversion="multiply_by_1024",
            ),
            "process.memory.thread_count": QuantityDescriptor(
                source_id=self.source_id,
                unit="1",
                statistic="gauge",
                subject_kind="process",
                window="read_interval",
                native_name="Threads",
                native_unit="1",
                conversion="identity",
            ),
        }
        return {qty_id: catalog[qty_id] for qty_id in self.quantities}


@dataclass(frozen=True)
class LinuxProcessTreeSource:
    """Configuration for bounded Linux procfs descendant-tree sampling.

    Runs strictly in-process on a bounded TPEN-owned sampling thread, targeting
    the live self process generation bound at scope activation (never freezing
    ``os.getpid()`` at construction and never targeting a foreign PID).
    """

    enabled: bool = True
    source_id: str = LINUX_PROCESS_TREE_SOURCE_ID
    execution_mode: str = LINUX_PROCESS_TREE_EXECUTION_MODE
    interval_seconds: float | None = None
    quantities: tuple[str, ...] = DEFAULT_LINUX_PROCESS_TREE_QUANTITIES
    max_scan_seconds: float = DEFAULT_LINUX_TREE_MAX_SCAN_SECONDS
    max_observed_members: int = DEFAULT_LINUX_TREE_MAX_OBSERVED_MEMBERS
    max_enumerated_candidates: int = DEFAULT_LINUX_TREE_MAX_ENUMERATED_CANDIDATES
    max_read_bytes: int = DEFAULT_LINUX_TREE_MAX_READ_BYTES
    observer_pids: tuple[int, ...] = ()
    pid: int | None = None
    probe: Any = None
    proc_root: str | Path | None = None

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "LinuxProcessTreeSource.enabled")
        validate_identifier(self.source_id, "LinuxProcessTreeSource.source_id")
        if (
            type(self.execution_mode) is not str
            or self.execution_mode != LINUX_PROCESS_TREE_EXECUTION_MODE
        ):
            raise ValueError(
                f"LinuxProcessTreeSource.execution_mode must be "
                f"{LINUX_PROCESS_TREE_EXECUTION_MODE!r}, got {self.execution_mode!r}"
            )
        if self.interval_seconds is not None:
            object.__setattr__(
                self,
                "interval_seconds",
                _require_positive_finite(
                    self.interval_seconds, "LinuxProcessTreeSource.interval_seconds"
                ),
            )
        if isinstance(self.quantities, (str, bytes)) or not isinstance(
            self.quantities, Sequence
        ):
            raise TypeError(
                "LinuxProcessTreeSource.quantities must be a sequence of quantity IDs"
            )
        selected = tuple(self.quantities)
        if not selected:
            raise ValueError("LinuxProcessTreeSource.quantities must be nonempty")
        seen: set[str] = set()
        allowed = frozenset(LINUX_PROCESS_TREE_QUANTITY_IDS)
        for qty_id in selected:
            if type(qty_id) is not str or qty_id not in allowed:
                raise ValueError(
                    f"unsupported LinuxProcessTreeSource quantity {qty_id!r}; "
                    f"expected subset of {LINUX_PROCESS_TREE_QUANTITY_IDS}"
                )
            if qty_id in seen:
                raise ValueError(f"duplicate LinuxProcessTreeSource quantity {qty_id!r}")
            seen.add(qty_id)
        object.__setattr__(self, "quantities", selected)

        object.__setattr__(
            self,
            "max_scan_seconds",
            _require_positive_finite(
                self.max_scan_seconds, "LinuxProcessTreeSource.max_scan_seconds"
            ),
        )
        _require_positive_int(
            self.max_observed_members, "LinuxProcessTreeSource.max_observed_members"
        )
        _require_positive_int(
            self.max_enumerated_candidates,
            "LinuxProcessTreeSource.max_enumerated_candidates",
        )
        _require_positive_int(self.max_read_bytes, "LinuxProcessTreeSource.max_read_bytes")

        if isinstance(self.observer_pids, (str, bytes)) or not isinstance(
            self.observer_pids, Sequence
        ):
            raise TypeError(
                "LinuxProcessTreeSource.observer_pids must be a sequence of positive ints"
            )
        obs_seen: set[int] = set()
        obs_tuple: list[int] = []
        for raw_pid in self.observer_pids:
            valid_pid = _require_positive_int(
                raw_pid, "LinuxProcessTreeSource.observer_pids[]"
            )
            if valid_pid not in obs_seen:
                obs_seen.add(valid_pid)
                obs_tuple.append(valid_pid)
        object.__setattr__(self, "observer_pids", tuple(obs_tuple))

        if self.pid is not None:
            pid = _require_positive_int(self.pid, "LinuxProcessTreeSource.pid")
            current_pid = os.getpid()
            if pid != current_pid:
                raise ValueError(
                    f"LinuxProcessTreeSource targets only the current self process and "
                    f"cannot target foreign PID {pid} (current process PID is {current_pid})"
                )
        if self.probe is not None and not callable(getattr(self.probe, "read", None)):
            raise TypeError("LinuxProcessTreeSource.probe must expose callable read()")
        if self.proc_root is not None:
            if isinstance(self.proc_root, Path):
                pass
            elif type(self.proc_root) is str:
                if not self.proc_root.strip():
                    raise ValueError(
                        "LinuxProcessTreeSource.proc_root must be a nonempty path or None"
                    )
                object.__setattr__(self, "proc_root", Path(self.proc_root))
            else:
                raise TypeError(
                    "LinuxProcessTreeSource.proc_root must be str, Path, or None, "
                    f"got {type(self.proc_root).__name__}"
                )

    def descriptors(self) -> dict[str, QuantityDescriptor]:
        """Return immutable :class:`QuantityDescriptor` map for selected quantities."""

        catalog: dict[str, QuantityDescriptor] = {
            "descendants.live_member_count": QuantityDescriptor(
                source_id=self.source_id,
                unit="1",
                statistic="gauge",
                subject_kind="descendant_set",
                window="scan_interval",
                native_name="descendant_live_member_count",
                native_unit="1",
                conversion="identity",
                quality=("bounded_procfs_scan_not_atomic",),
            ),
            "descendants.sum_rss_bytes": QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="gauge",
                subject_kind="descendant_set",
                window="scan_interval",
                native_name="sum_VmRSS",
                native_unit="KiB",
                conversion="multiply_by_1024",
                quality=(
                    "approximate_rss",
                    "sum_rss_double_counts_shared_pages",
                    "bounded_procfs_scan_not_atomic",
                ),
            ),
            "descendants.matched_cpu_user_delta_seconds": QuantityDescriptor(
                source_id=self.source_id,
                unit="s",
                statistic="gauge",
                subject_kind="descendant_set",
                window="scan_interval",
                native_name="matched_delta_utime",
                native_unit="clock_ticks",
                conversion="divide_by_clock_ticks_per_second",
                quality=(
                    "matched_generation_scan_delta_only",
                    "excludes_reaped_children",
                ),
            ),
            "descendants.matched_cpu_system_delta_seconds": QuantityDescriptor(
                source_id=self.source_id,
                unit="s",
                statistic="gauge",
                subject_kind="descendant_set",
                window="scan_interval",
                native_name="matched_delta_stime",
                native_unit="clock_ticks",
                conversion="divide_by_clock_ticks_per_second",
                quality=(
                    "matched_generation_scan_delta_only",
                    "excludes_reaped_children",
                ),
            ),
        }
        return {qty_id: catalog[qty_id] for qty_id in self.quantities}


@dataclass(frozen=True)
class LinuxCgroupV2Source:
    """Configuration for read-only Linux cgroup v2 membership and limit sampling.

    Strictly read-only and Torch-free: never guesses cgroup paths from
    scheduler job IDs, never writes or resets ``memory.peak``, refuses cgroup
    v1 pretending to be v2, and records explicit shared-scope provenance.
    """

    enabled: bool = True
    source_id: str = LINUX_CGROUP_V2_SOURCE_ID
    execution_mode: str = LINUX_CGROUP_V2_EXECUTION_MODE
    interval_seconds: float | None = None
    quantities: tuple[str, ...] = DEFAULT_LINUX_CGROUP_V2_QUANTITIES
    max_read_bytes: int = DEFAULT_LINUX_CGROUP_MAX_READ_BYTES
    max_mountinfo_bytes: int = DEFAULT_LINUX_CGROUP_MAX_MOUNTINFO_BYTES
    max_read_seconds: float = DEFAULT_LINUX_CGROUP_MAX_READ_SECONDS
    max_ancestor_depth: int = DEFAULT_LINUX_CGROUP_MAX_ANCESTOR_DEPTH
    pid: int | None = None
    probe: Any = None
    proc_root: str | Path | None = None
    cgroup_root: str | Path | None = None

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "LinuxCgroupV2Source.enabled")
        validate_identifier(self.source_id, "LinuxCgroupV2Source.source_id")
        if (
            type(self.execution_mode) is not str
            or self.execution_mode != LINUX_CGROUP_V2_EXECUTION_MODE
        ):
            raise ValueError(
                f"LinuxCgroupV2Source.execution_mode must be "
                f"{LINUX_CGROUP_V2_EXECUTION_MODE!r}, got {self.execution_mode!r}"
            )
        if self.interval_seconds is not None:
            object.__setattr__(
                self,
                "interval_seconds",
                _require_positive_finite(
                    self.interval_seconds, "LinuxCgroupV2Source.interval_seconds"
                ),
            )
        if isinstance(self.quantities, (str, bytes)) or not isinstance(
            self.quantities, Sequence
        ):
            raise TypeError(
                "LinuxCgroupV2Source.quantities must be a sequence of quantity IDs"
            )
        selected = tuple(self.quantities)
        if not selected:
            raise ValueError("LinuxCgroupV2Source.quantities must be nonempty")
        seen: set[str] = set()
        allowed = frozenset(LINUX_CGROUP_V2_QUANTITY_IDS)
        for qty_id in selected:
            if type(qty_id) is not str or qty_id not in allowed:
                raise ValueError(
                    f"unsupported LinuxCgroupV2Source quantity {qty_id!r}; "
                    f"expected subset of {LINUX_CGROUP_V2_QUANTITY_IDS}"
                )
            if qty_id in seen:
                raise ValueError(f"duplicate LinuxCgroupV2Source quantity {qty_id!r}")
            seen.add(qty_id)
        object.__setattr__(self, "quantities", selected)

        _require_positive_int(self.max_read_bytes, "LinuxCgroupV2Source.max_read_bytes")
        _require_positive_int(
            self.max_mountinfo_bytes, "LinuxCgroupV2Source.max_mountinfo_bytes"
        )
        object.__setattr__(
            self,
            "max_read_seconds",
            _require_positive_finite(
                self.max_read_seconds, "LinuxCgroupV2Source.max_read_seconds"
            ),
        )
        _require_positive_int(
            self.max_ancestor_depth, "LinuxCgroupV2Source.max_ancestor_depth"
        )

        if self.pid is not None:
            pid = _require_positive_int(self.pid, "LinuxCgroupV2Source.pid")
            current_pid = os.getpid()
            if pid != current_pid:
                raise ValueError(
                    f"LinuxCgroupV2Source targets only the current self process and "
                    f"cannot target foreign PID {pid} (current process PID is {current_pid})"
                )
        if self.probe is not None and not callable(getattr(self.probe, "read", None)):
            raise TypeError("LinuxCgroupV2Source.probe must expose callable read()")
        for attr_name in ("proc_root", "cgroup_root"):
            raw_path = getattr(self, attr_name)
            if raw_path is None or isinstance(raw_path, Path):
                continue
            if type(raw_path) is str:
                if not raw_path.strip():
                    raise ValueError(
                        f"LinuxCgroupV2Source.{attr_name} must be a nonempty path or None"
                    )
                object.__setattr__(self, attr_name, Path(raw_path))
            else:
                raise TypeError(
                    f"LinuxCgroupV2Source.{attr_name} must be str, Path, or None, "
                    f"got {type(raw_path).__name__}"
                )

    def descriptors(self) -> dict[str, QuantityDescriptor]:
        """Return immutable :class:`QuantityDescriptor` map for selected quantities."""

        shared_q = "shared_cgroup_scope_not_exclusive_entitlement"
        catalog: dict[str, QuantityDescriptor] = {
            "cgroup.memory.current_bytes": QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="gauge",
                subject_kind="cgroup",
                window="cgroup_read_interval",
                native_name="memory.current",
                native_unit="B",
                conversion="identity",
                quality=("cgroup_scope_not_sum_rss", shared_q),
            ),
            "cgroup.memory.peak_bytes": QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="high_water",
                subject_kind="cgroup",
                window="cgroup_lifetime",
                native_name="memory.peak",
                native_unit="B",
                conversion="identity",
                quality=("cgroup_lifetime_high_water_not_reset_per_run", shared_q),
            ),
            "cgroup.memory.max_bytes": QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="capacity",
                subject_kind="cgroup",
                window="cgroup_read_interval",
                native_name="memory.max",
                native_unit="B",
                conversion="identity",
                quality=("hierarchical_effective_limit", shared_q),
            ),
            "cgroup.cpu.usage_seconds": QuantityDescriptor(
                source_id=self.source_id,
                unit="s",
                statistic="counter",
                subject_kind="cgroup",
                window="cgroup_lifetime",
                native_name="cpu.stat.usage_usec",
                native_unit="us",
                conversion="divide_by_1000000",
                quality=(shared_q,),
            ),
            "cgroup.cpu.user_seconds": QuantityDescriptor(
                source_id=self.source_id,
                unit="s",
                statistic="counter",
                subject_kind="cgroup",
                window="cgroup_lifetime",
                native_name="cpu.stat.user_usec",
                native_unit="us",
                conversion="divide_by_1000000",
                quality=(shared_q,),
            ),
            "cgroup.cpu.system_seconds": QuantityDescriptor(
                source_id=self.source_id,
                unit="s",
                statistic="counter",
                subject_kind="cgroup",
                window="cgroup_lifetime",
                native_name="cpu.stat.system_usec",
                native_unit="us",
                conversion="divide_by_1000000",
                quality=(shared_q,),
            ),
            "cgroup.cpu.throttled_seconds": QuantityDescriptor(
                source_id=self.source_id,
                unit="s",
                statistic="counter",
                subject_kind="cgroup",
                window="cgroup_lifetime",
                native_name="cpu.stat.throttled_usec",
                native_unit="us",
                conversion="divide_by_1000000",
                quality=(shared_q,),
            ),
            "cgroup.cpu.nr_throttled": QuantityDescriptor(
                source_id=self.source_id,
                unit="1",
                statistic="counter",
                subject_kind="cgroup",
                window="cgroup_lifetime",
                native_name="cpu.stat.nr_throttled",
                native_unit="1",
                conversion="identity",
                quality=(shared_q,),
            ),
            "cgroup.cpu.max_cores": QuantityDescriptor(
                source_id=self.source_id,
                unit="1",
                statistic="capacity",
                subject_kind="cgroup",
                window="cgroup_read_interval",
                native_name="cpu.max",
                native_unit="quota_over_period",
                conversion="quota_divided_by_period",
                quality=("hierarchical_effective_limit", shared_q),
            ),
            "cgroup.cpuset.effective_cpus": QuantityDescriptor(
                source_id=self.source_id,
                unit="1",
                statistic="capacity",
                subject_kind="cgroup",
                window="cgroup_read_interval",
                native_name="cpuset.cpus.effective",
                native_unit="cpu_list",
                conversion="count_cpu_list_members",
                quality=(shared_q,),
            ),
        }
        return {qty_id: catalog[qty_id] for qty_id in self.quantities}


@dataclass(frozen=True)
class AllocatorSource:
    """Configuration for configured-device PyTorch allocator observations.

    Construction is strictly Torch-free: concrete Torch probes are constructed
    only during ``ResourceSamples.run_scope`` activation on the owning thread.
    Current allocator usage (``statistic="gauge"``) and peak allocator usage
    (``statistic="high_water"``) are kept semantically separate, and peak-reset
    authority is explicit via ``reset_peaks`` and
    :class:`tpen.callback.base.ResourceOwnershipClaim`.

    Parameters
    ----------
    enabled : bool, optional
        Whether this source is active (default ``True``).
    source_id : str, optional
        Stable source identifier written into descriptors and samples.
    execution_mode : str, optional
        Must be ``"in_process_cooperative"``. Allocator probes run strictly on
        the owning thread at cooperative core checkpoints, never on a
        background thread or in a helper subprocess.
    device : str or None, optional
        Optional explicit device string; ``None`` resolves from the run's
        configured ``runtime.device`` / ``context.metadata.device``.
    interval_seconds : float or None, optional
        Optional cooperative throttling cadence in seconds. ``None`` inherits
        :attr:`ResourceSamples.interval_seconds`.
    include_current : bool, optional
        Collect current ``allocated_bytes`` / ``reserved_bytes`` gauges.
    include_peak : bool, optional
        Collect peak ``allocated_bytes`` / ``reserved_bytes`` high-water marks.
    reset_peaks : bool, optional
        When ``True``, this observer claims exclusive ``allocator_reset``
        ownership for the configured device and resets peaks once on activation.
        When ``False``, peaks are never silently reset and no
        ``allocator_reset`` claim is emitted.
    current_probe_factory : callable or None, optional
        Optional deferred factory ``(device) -> AllocatorCurrentProbe``.
    peak_probe_factory : callable or None, optional
        Optional deferred factory ``(device) -> AllocatorPeakProbe``.
    """

    enabled: bool = True
    source_id: str = ALLOCATOR_SOURCE_ID
    execution_mode: str = ALLOCATOR_EXECUTION_MODE
    device: str | None = None
    interval_seconds: float | None = None
    include_current: bool = True
    include_peak: bool = True
    reset_peaks: bool = True
    current_probe_factory: Callable[[Any], Any] | None = None
    peak_probe_factory: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "AllocatorSource.enabled")
        validate_identifier(self.source_id, "AllocatorSource.source_id")
        if (
            type(self.execution_mode) is not str
            or self.execution_mode != ALLOCATOR_EXECUTION_MODE
        ):
            raise ValueError(
                f"AllocatorSource.execution_mode must be {ALLOCATOR_EXECUTION_MODE!r}, "
                f"got {self.execution_mode!r}: Torch allocator probes must run "
                "cooperatively on the owning thread"
            )
        if self.device is not None:
            if type(self.device) is not str or not self.device.strip():
                raise ValueError("AllocatorSource.device must be a nonempty string or None")
            object.__setattr__(self, "device", self.device.strip())
        if self.interval_seconds is not None:
            object.__setattr__(
                self,
                "interval_seconds",
                _require_positive_finite(
                    self.interval_seconds, "AllocatorSource.interval_seconds"
                ),
            )
        _require_bool(self.include_current, "AllocatorSource.include_current")
        _require_bool(self.include_peak, "AllocatorSource.include_peak")
        _require_bool(self.reset_peaks, "AllocatorSource.reset_peaks")
        if self.reset_peaks and not self.include_peak:
            raise ValueError(
                "AllocatorSource.reset_peaks=True requires include_peak=True"
            )
        if self.enabled and not (self.include_current or self.include_peak):
            raise ValueError(
                "AllocatorSource requires at least one of include_current or "
                "include_peak when enabled=True"
            )
        if self.current_probe_factory is not None and not callable(
            self.current_probe_factory
        ):
            raise TypeError("AllocatorSource.current_probe_factory must be callable or None")
        if self.peak_probe_factory is not None and not callable(self.peak_probe_factory):
            raise TypeError("AllocatorSource.peak_probe_factory must be callable or None")

    @property
    def selected_quantities(self) -> tuple[str, ...]:
        """Return ordered quantity IDs selected by this allocator source."""

        items: list[str] = []
        if self.include_current:
            items.extend(ALLOCATOR_CURRENT_QUANTITY_IDS)
        if self.include_peak:
            items.extend(ALLOCATOR_PEAK_QUANTITY_IDS)
        return tuple(items)

    def descriptors(self) -> dict[str, QuantityDescriptor]:
        """Return immutable :class:`QuantityDescriptor` map for selected quantities."""

        out: dict[str, QuantityDescriptor] = {}
        if self.include_current:
            out["allocator.current.allocated_bytes"] = QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="gauge",
                subject_kind="process_allocator",
                window="read_interval",
                native_name="memory_allocated",
                native_unit="B",
                conversion="identity",
            )
            out["allocator.current.reserved_bytes"] = QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="gauge",
                subject_kind="process_allocator",
                window="read_interval",
                native_name="memory_reserved",
                native_unit="B",
                conversion="identity",
            )
        if self.include_peak:
            peak_window = (
                "allocator_reset_window" if self.reset_peaks else "unknown"
            )
            peak_quality: tuple[str, ...] = (
                () if self.reset_peaks else ("unreset_peak_window_unknown",)
            )
            out["allocator.peak.allocated_bytes"] = QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="high_water",
                subject_kind="process_allocator",
                window=peak_window,
                native_name="max_memory_allocated",
                native_unit="MiB",
                conversion="multiply_by_1048576",
                quality=peak_quality,
            )
            out["allocator.peak.reserved_bytes"] = QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="high_water",
                subject_kind="process_allocator",
                window=peak_window,
                native_name="max_memory_reserved",
                native_unit="MiB",
                conversion="multiply_by_1048576",
                quality=peak_quality,
            )
        return out


def _coerce_accelerator_identity(
    raw: AcceleratorIdentity | Mapping[str, Any] | None,
) -> AcceleratorIdentity | None:
    """Strictly coerce an explicit ``AcceleratorIdentity`` or plain mapping."""

    if raw is None:
        return None
    if type(raw) is AcceleratorIdentity:
        if not isinstance(raw.kind, AcceleratorKind):
            raise TypeError(
                f"NvidiaDeviceSource.identity.kind must be AcceleratorKind, "
                f"got {type(raw.kind).__name__}"
            )
        if raw.index is not None and (type(raw.index) is not int or raw.index < 0):
            raise ValueError(
                f"NvidiaDeviceSource.identity.index must be a non-negative int or None, "
                f"got {raw.index!r}"
            )
        if raw.uuid is not None and type(raw.uuid) is not str:
            raise TypeError(
                f"NvidiaDeviceSource.identity.uuid must be str or None, "
                f"got {type(raw.uuid).__name__}"
            )
        return raw
    if isinstance(raw, AcceleratorIdentity):
        raise TypeError(
            "NvidiaDeviceSource.identity must be exact AcceleratorIdentity, "
            f"mapping, or None; got subclass {type(raw).__name__}"
        )
    if not isinstance(raw, Mapping):
        raise TypeError(
            "NvidiaDeviceSource.identity must be AcceleratorIdentity, mapping, or None; "
            f"got {type(raw).__name__}"
        )
    _reject_forbidden_selector_keys(raw, owner="NvidiaDeviceSource.identity")
    allowed_keys = frozenset({"kind", "index", "uuid"})
    raw_keys = set(raw.keys())
    for key in raw_keys:
        if type(key) is not str:
            raise TypeError(
                f"NvidiaDeviceSource.identity keys must be strings, got {type(key).__name__}"
            )
    extra = sorted(raw_keys - allowed_keys)
    if extra:
        raise ValueError(
            f"NvidiaDeviceSource.identity has unsupported key(s): {extra}"
        )
    if "kind" not in raw:
        raise ValueError("NvidiaDeviceSource.identity requires 'kind'")

    raw_kind = raw["kind"]
    if isinstance(raw_kind, AcceleratorKind):
        kind = raw_kind
    elif type(raw_kind) is str and raw_kind.strip():
        try:
            kind = AcceleratorKind(raw_kind.strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"unsupported NvidiaDeviceSource.identity.kind {raw_kind!r}; "
                f"expected one of {[k.value for k in AcceleratorKind]}"
            ) from exc
    else:
        raise TypeError(
            "NvidiaDeviceSource.identity.kind must be AcceleratorKind or nonempty str, "
            f"got {type(raw_kind).__name__}"
        )

    raw_index = raw.get("index")
    if raw_index is not None:
        if type(raw_index) is not int:
            raise TypeError(
                "NvidiaDeviceSource.identity.index must be int or None, "
                f"got {type(raw_index).__name__}"
            )
        if raw_index < 0:
            raise ValueError(
                f"NvidiaDeviceSource.identity.index must be >= 0, got {raw_index}"
            )
    raw_uuid = raw.get("uuid")
    if raw_uuid is not None:
        if type(raw_uuid) is not str:
            raise TypeError(
                "NvidiaDeviceSource.identity.uuid must be str or None, "
                f"got {type(raw_uuid).__name__}"
            )
        if not raw_uuid.strip():
            raise ValueError(
                "NvidiaDeviceSource.identity.uuid must be a nonempty string or None"
            )
    return AcceleratorIdentity(kind=kind, index=raw_index, uuid=raw_uuid)


@dataclass(frozen=True)
class NvidiaDeviceSource:
    """Configuration for explicit, bounded NVIDIA physical-GPU ``nvidia-smi`` sampling.

    Targets strictly the already-selected physical :class:`AcceleratorIdentity`
    via a supervised bounded subprocess (``<= 10.0`` s maximum), recording
    whole-device framebuffer usage/total and recent vendor-window busy fraction
    with explicit shared-device caveats and MIG/subdevice ambiguity refusal.
    Never claims ``allocator_reset`` authority and never resets allocator peaks.
    """

    enabled: bool = True
    source_id: str = NVIDIA_DEVICE_SOURCE_ID
    execution_mode: str = NVIDIA_DEVICE_EXECUTION_MODE
    interval_seconds: float | None = None
    quantities: tuple[str, ...] = DEFAULT_NVIDIA_DEVICE_QUANTITIES
    device: str | None = None
    identity: AcceleratorIdentity | Mapping[str, Any] | None = None
    executable: str | Path = DEFAULT_NVIDIA_SMI_EXECUTABLE
    timeout_seconds: float = DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS
    max_output_bytes: int = DEFAULT_NVIDIA_SMI_MAX_OUTPUT_BYTES
    probe: Any = None
    identity_resolver: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "NvidiaDeviceSource.enabled")
        validate_identifier(self.source_id, "NvidiaDeviceSource.source_id")
        if (
            type(self.execution_mode) is not str
            or self.execution_mode != NVIDIA_DEVICE_EXECUTION_MODE
        ):
            raise ValueError(
                f"NvidiaDeviceSource.execution_mode must be "
                f"{NVIDIA_DEVICE_EXECUTION_MODE!r}, got {self.execution_mode!r}"
            )
        if self.interval_seconds is not None:
            object.__setattr__(
                self,
                "interval_seconds",
                _require_positive_finite(
                    self.interval_seconds, "NvidiaDeviceSource.interval_seconds"
                ),
            )
        if isinstance(self.quantities, (str, bytes)) or not isinstance(
            self.quantities, Sequence
        ):
            raise TypeError(
                "NvidiaDeviceSource.quantities must be a sequence of quantity IDs"
            )
        selected = tuple(self.quantities)
        if not selected:
            raise ValueError("NvidiaDeviceSource.quantities must be nonempty")
        seen: set[str] = set()
        allowed = frozenset(NVIDIA_DEVICE_QUANTITY_IDS)
        for qty_id in selected:
            if type(qty_id) is not str or qty_id not in allowed:
                raise ValueError(
                    f"unsupported NvidiaDeviceSource quantity {qty_id!r}; "
                    f"expected subset of {NVIDIA_DEVICE_QUANTITY_IDS}"
                )
            if qty_id in seen:
                raise ValueError(f"duplicate NvidiaDeviceSource quantity {qty_id!r}")
            seen.add(qty_id)
        object.__setattr__(self, "quantities", selected)

        if self.device is not None:
            if type(self.device) is not str or not self.device.strip():
                raise ValueError(
                    "NvidiaDeviceSource.device must be a nonempty string or None"
                )
            object.__setattr__(self, "device", self.device.strip())

        coerced_identity = _coerce_accelerator_identity(self.identity)
        object.__setattr__(self, "identity", coerced_identity)

        if isinstance(self.executable, Path):
            exe_text = str(self.executable)
        elif type(self.executable) is str:
            exe_text = self.executable
        else:
            raise TypeError(
                "NvidiaDeviceSource.executable must be str or Path, "
                f"got {type(self.executable).__name__}"
            )
        if not exe_text.strip() or any(ord(ch) < 32 or ord(ch) == 127 for ch in exe_text):
            raise ValueError(
                "NvidiaDeviceSource.executable must be a nonempty executable name or path "
                "without control characters"
            )

        timeout_val = _require_positive_finite(
            self.timeout_seconds, "NvidiaDeviceSource.timeout_seconds"
        )
        if timeout_val > MAX_NVIDIA_SMI_TIMEOUT_SECONDS:
            raise ValueError(
                f"NvidiaDeviceSource.timeout_seconds must be <= "
                f"{MAX_NVIDIA_SMI_TIMEOUT_SECONDS} s, got {self.timeout_seconds!r}"
            )
        object.__setattr__(self, "timeout_seconds", timeout_val)

        _require_positive_int(
            self.max_output_bytes, "NvidiaDeviceSource.max_output_bytes"
        )
        if self.probe is not None and not callable(getattr(self.probe, "read", None)):
            raise TypeError("NvidiaDeviceSource.probe must expose callable read()")
        if self.identity_resolver is not None and not callable(self.identity_resolver):
            raise TypeError(
                "NvidiaDeviceSource.identity_resolver must be callable or None"
            )

    def descriptors(self) -> dict[str, QuantityDescriptor]:
        """Return immutable :class:`QuantityDescriptor` map for selected quantities."""

        shared_q = "shared_physical_device_not_per_process_attribution"
        catalog: dict[str, QuantityDescriptor] = {
            "device.framebuffer.used_bytes": QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="gauge",
                subject_kind="physical_gpu",
                window="read_interval",
                native_name="memory.used",
                native_unit="MiB",
                conversion="multiply_by_1048576",
                quality=(shared_q, "physical_gpu_scope_not_process_allocator"),
            ),
            "device.framebuffer.total_bytes": QuantityDescriptor(
                source_id=self.source_id,
                unit="B",
                statistic="capacity",
                subject_kind="physical_gpu",
                window="inventory_or_read_interval",
                native_name="memory.total",
                native_unit="MiB",
                conversion="multiply_by_1048576",
                quality=(shared_q, "physical_gpu_scope_not_process_allocator"),
            ),
            "device.busy_fraction": QuantityDescriptor(
                source_id=self.source_id,
                unit="1",
                statistic="vendor_window_mean",
                subject_kind="physical_gpu",
                window="vendor_recent_window_unknown_bounds",
                native_name="utilization.gpu",
                native_unit="percent",
                conversion="divide_by_100",
                quality=(shared_q, "vendor_recent_window_not_extrapolated_utilization"),
            ),
        }
        return {qty_id: catalog[qty_id] for qty_id in self.quantities}


# ---------------------------------------------------------------------------
# Reading conversion helpers
# ---------------------------------------------------------------------------


def _rusage_baseline_to_readings(
    baseline: ProcessResourceBaseline,
    quantities: Sequence[str],
) -> dict[str, ObservabilityReading]:
    """Convert one :class:`ProcessResourceBaseline` into tagged v1 readings."""

    field_specs: dict[str, tuple[str, str]] = {
        "process.rusage.user_cpu_seconds": ("user_cpu_seconds", "float"),
        "process.rusage.system_cpu_seconds": ("system_cpu_seconds", "float"),
        "process.rusage.read_block_operations": ("read_block_operations", "int"),
        "process.rusage.write_block_operations": ("write_block_operations", "int"),
        "process.rusage.voluntary_context_switches": ("voluntary_context_switches", "int"),
        "process.rusage.involuntary_context_switches": (
            "involuntary_context_switches",
            "int",
        ),
        "process.rusage.max_rss_bytes": ("peak_rss_mb", "mib_to_bytes"),
    }
    readings: dict[str, ObservabilityReading] = {}
    for qty_id in quantities:
        attr_name, mode = field_specs[qty_id]
        raw = getattr(baseline, attr_name)
        if isinstance(raw, ResourceUnavailable):
            readings[qty_id] = ObservabilityReading(
                status="unavailable",
                reason_code="rusage_unavailable",
                detail=raw.reason or "getrusage unavailable",
            )
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            readings[qty_id] = ObservabilityReading(
                status="error",
                reason_code="invalid_reading",
                detail=f"unexpected {attr_name} type {type(raw).__name__}",
            )
            continue
        if isinstance(raw, float) and not math.isfinite(raw):
            readings[qty_id] = ObservabilityReading(
                status="error",
                reason_code="non_finite_reading",
                detail=f"{attr_name} was non-finite ({raw!r})",
            )
            continue
        if raw < 0:
            readings[qty_id] = ObservabilityReading(
                status="error",
                reason_code="negative_reading",
                detail=f"{attr_name} was negative ({raw!r})",
            )
            continue
        if mode == "float":
            readings[qty_id] = ObservabilityReading(status="ok", value=float(raw))
        elif mode == "int":
            readings[qty_id] = ObservabilityReading(status="ok", value=int(raw))
        else:
            # Exact integer conversion from MiB back to bytes; zero stays 0.
            readings[qty_id] = ObservabilityReading(
                status="ok", value=int(round(float(raw) * _BYTES_PER_MIB))
            )
    return readings


def _linux_memory_reading_to_readings(
    reading: LinuxProcessMemoryReading,
    quantities: Sequence[str],
) -> dict[str, ObservabilityReading]:
    """Convert one :class:`LinuxProcessMemoryReading` into tagged v1 readings."""

    field_map: dict[str, str] = {
        "process.memory.rss_bytes": "rss_bytes",
        "process.memory.virtual_bytes": "virtual_bytes",
        "process.memory.thread_count": "thread_count",
    }
    fallback_status = "error" if getattr(reading, "status", None) == "error" else "unavailable"
    fallback_code = getattr(reading, "reason_code", None) or "linux_memory_unavailable"
    fallback_detail = getattr(reading, "detail", None) or "linux memory unavailable"

    out: dict[str, ObservabilityReading] = {}
    for qty_id in quantities:
        attr_name = field_map[qty_id]
        raw = getattr(reading, attr_name)
        if isinstance(raw, LinuxMemoryUnavailable):
            out[qty_id] = ObservabilityReading(
                status=raw.status,
                reason_code=raw.reason_code,
                detail=raw.reason,
            )
            continue
        if isinstance(raw, ResourceUnavailable):
            out[qty_id] = ObservabilityReading(
                status=fallback_status,
                reason_code=fallback_code,
                detail=raw.reason or fallback_detail,
            )
            continue
        if type(raw) is not int:
            out[qty_id] = ObservabilityReading(
                status="error",
                reason_code="invalid_reading",
                detail=f"unexpected {attr_name} type {type(raw).__name__}",
            )
            continue
        if raw < 0:
            out[qty_id] = ObservabilityReading(
                status="error",
                reason_code="negative_reading",
                detail=f"{attr_name} was negative ({raw!r})",
            )
            continue
        # Legitimate 0 bytes or 0 count is preserved strictly as status='ok'.
        out[qty_id] = ObservabilityReading(status="ok", value=raw)
    return out


def _linux_tree_reading_to_readings(
    reading: LinuxProcessTreeReading,
    quantities: Sequence[str],
) -> dict[str, ObservabilityReading]:
    """Convert one :class:`LinuxProcessTreeReading` into tagged v1 readings."""

    field_specs: dict[str, tuple[str, str]] = {
        "descendants.live_member_count": ("live_member_count", "int"),
        "descendants.sum_rss_bytes": ("sum_rss_bytes", "int"),
        "descendants.matched_cpu_user_delta_seconds": (
            "matched_cpu_user_delta_seconds",
            "float",
        ),
        "descendants.matched_cpu_system_delta_seconds": (
            "matched_cpu_system_delta_seconds",
            "float",
        ),
    }
    fallback_status = "error" if getattr(reading, "status", None) == "error" else "unavailable"
    fallback_code = getattr(reading, "reason_code", None) or "linux_tree_unavailable"
    fallback_detail = getattr(reading, "detail", None) or "linux process tree unavailable"

    out: dict[str, ObservabilityReading] = {}
    for qty_id in quantities:
        attr_name, mode = field_specs[qty_id]
        raw = getattr(reading, attr_name)
        if isinstance(raw, LinuxMemoryUnavailable):
            out[qty_id] = ObservabilityReading(
                status=raw.status,
                reason_code=raw.reason_code,
                detail=raw.reason,
            )
            continue
        if isinstance(raw, ResourceUnavailable):
            out[qty_id] = ObservabilityReading(
                status=fallback_status,
                reason_code=fallback_code,
                detail=raw.reason or fallback_detail,
            )
            continue
        if mode == "int":
            if type(raw) is not int:
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="invalid_reading",
                    detail=f"unexpected {attr_name} type {type(raw).__name__}",
                )
                continue
            if raw < 0:
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="negative_reading",
                    detail=f"{attr_name} was negative ({raw!r})",
                )
                continue
            out[qty_id] = ObservabilityReading(status="ok", value=raw)
        else:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="invalid_reading",
                    detail=f"unexpected {attr_name} type {type(raw).__name__}",
                )
                continue
            fval = float(raw)
            if not math.isfinite(fval):
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="non_finite_reading",
                    detail=f"{attr_name} was non-finite ({raw!r})",
                )
                continue
            if fval < 0.0:
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="negative_reading",
                    detail=f"{attr_name} was negative ({raw!r})",
                )
                continue
            out[qty_id] = ObservabilityReading(status="ok", value=fval)
    return out


def _linux_cgroup_reading_to_readings(
    reading: LinuxCgroupV2Reading,
    quantities: Sequence[str],
) -> dict[str, ObservabilityReading]:
    """Convert one :class:`LinuxCgroupV2Reading` into tagged v1 readings."""

    field_specs: dict[str, tuple[str, str]] = {
        "cgroup.memory.current_bytes": ("memory_current_bytes", "int"),
        "cgroup.memory.peak_bytes": ("memory_peak_bytes", "int"),
        "cgroup.memory.max_bytes": ("memory_max_bytes", "int"),
        "cgroup.cpu.usage_seconds": ("cpu_usage_seconds", "float"),
        "cgroup.cpu.user_seconds": ("cpu_user_seconds", "float"),
        "cgroup.cpu.system_seconds": ("cpu_system_seconds", "float"),
        "cgroup.cpu.throttled_seconds": ("cpu_throttled_seconds", "float"),
        "cgroup.cpu.nr_throttled": ("cpu_nr_throttled", "int"),
        "cgroup.cpu.max_cores": ("cpu_max_cores", "float"),
        "cgroup.cpuset.effective_cpus": ("cpuset_effective_cpus", "int"),
    }
    fallback_status = "error" if getattr(reading, "status", None) == "error" else "unavailable"
    fallback_code = getattr(reading, "reason_code", None) or "linux_cgroup_unavailable"
    fallback_detail = getattr(reading, "detail", None) or "linux cgroup v2 unavailable"

    out: dict[str, ObservabilityReading] = {}
    for qty_id in quantities:
        attr_name, mode = field_specs[qty_id]
        raw = getattr(reading, attr_name)
        if isinstance(raw, LinuxMemoryUnavailable):
            out[qty_id] = ObservabilityReading(
                status=raw.status,
                reason_code=raw.reason_code,
                detail=raw.reason,
            )
            continue
        if isinstance(raw, ResourceUnavailable):
            out[qty_id] = ObservabilityReading(
                status=fallback_status,
                reason_code=fallback_code,
                detail=raw.reason or fallback_detail,
            )
            continue
        if mode == "int":
            if type(raw) is not int:
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="invalid_reading",
                    detail=f"unexpected {attr_name} type {type(raw).__name__}",
                )
                continue
            if raw < 0:
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="negative_reading",
                    detail=f"{attr_name} was negative ({raw!r})",
                )
                continue
            out[qty_id] = ObservabilityReading(status="ok", value=raw)
        else:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="invalid_reading",
                    detail=f"unexpected {attr_name} type {type(raw).__name__}",
                )
                continue
            fval = float(raw)
            if not math.isfinite(fval):
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="non_finite_reading",
                    detail=f"{attr_name} was non-finite ({raw!r})",
                )
                continue
            if fval < 0.0:
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="negative_reading",
                    detail=f"{attr_name} was negative ({raw!r})",
                )
                continue
            out[qty_id] = ObservabilityReading(status="ok", value=fval)
    return out


def _allocator_bytes_reading(
    value: object,
    *,
    field_name: str,
    from_mib: bool = False,
) -> ObservabilityReading:
    """Convert one allocator reading (bytes or MiB) into a tagged v1 reading."""

    if _is_allocator_unavailable(value):
        reason = str(getattr(value, "reason", "")).strip() or "allocator unavailable"
        return ObservabilityReading(
            status="unavailable",
            reason_code="allocator_unavailable",
            detail=reason,
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ObservabilityReading(
            status="error",
            reason_code="invalid_reading",
            detail=f"unexpected {field_name} type {type(value).__name__}",
        )
    if isinstance(value, float) and not math.isfinite(value):
        return ObservabilityReading(
            status="error",
            reason_code="non_finite_reading",
            detail=f"{field_name} was non-finite ({value!r})",
        )
    if value < 0:
        return ObservabilityReading(
            status="error",
            reason_code="negative_reading",
            detail=f"{field_name} was negative ({value!r})",
        )
    if from_mib:
        return ObservabilityReading(
            status="ok", value=int(round(float(value) * _BYTES_PER_MIB))
        )
    return ObservabilityReading(status="ok", value=int(value))


def _nvidia_device_reading_to_readings(
    reading: NvidiaDeviceReading,
    quantities: Sequence[str],
) -> dict[str, ObservabilityReading]:
    """Convert one :class:`NvidiaDeviceReading` into tagged v1 readings."""

    field_specs: dict[str, tuple[str, str]] = {
        "device.framebuffer.used_bytes": ("framebuffer_used_bytes", "int"),
        "device.framebuffer.total_bytes": ("framebuffer_total_bytes", "int"),
        "device.busy_fraction": ("busy_fraction", "fraction"),
    }
    fallback_status = "error" if getattr(reading, "status", None) == "error" else "unavailable"
    fallback_code = getattr(reading, "reason_code", None) or "nvidia_smi_unavailable"
    fallback_detail = getattr(reading, "detail", None) or "nvidia-smi device unavailable"

    out: dict[str, ObservabilityReading] = {}
    for qty_id in quantities:
        attr_name, mode = field_specs[qty_id]
        raw = getattr(reading, attr_name)
        if isinstance(raw, NvidiaDeviceUnavailable):
            out[qty_id] = ObservabilityReading(
                status=raw.status,
                reason_code=raw.reason_code,
                detail=raw.reason,
            )
            continue
        if isinstance(raw, ResourceUnavailable):
            out[qty_id] = ObservabilityReading(
                status=fallback_status,
                reason_code=fallback_code,
                detail=raw.reason or fallback_detail,
            )
            continue
        if mode == "int":
            if type(raw) is not int:
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="invalid_reading",
                    detail=f"unexpected {attr_name} type {type(raw).__name__}",
                )
                continue
            if raw < 0:
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="negative_reading",
                    detail=f"{attr_name} was negative ({raw!r})",
                )
                continue
            out[qty_id] = ObservabilityReading(status="ok", value=raw)
        else:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="invalid_reading",
                    detail=f"unexpected {attr_name} type {type(raw).__name__}",
                )
                continue
            fval = float(raw)
            if not math.isfinite(fval):
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="non_finite_reading",
                    detail=f"{attr_name} was non-finite ({raw!r})",
                )
                continue
            if fval < 0.0 or fval > 1.0:
                out[qty_id] = ObservabilityReading(
                    status="error",
                    reason_code="out_of_range_reading",
                    detail=f"{attr_name} was outside [0.0, 1.0] ({raw!r})",
                )
                continue
            out[qty_id] = ObservabilityReading(status="ok", value=fval)
    return out


def _encode_sample_line_bytes(sample: ObservabilitySample) -> int:
    """Return exact ASCII JSONL byte length (including trailing newline)."""

    encoded = json.dumps(
        sample.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    )
    return len(encoded.encode("ascii")) + 1


def _compute_configuration_digest(context: RunContext, observer_id: str) -> str:
    """Compute a 64-char SHA-256 digest for the run configuration."""

    try:
        return resolved_config_hash(context.cfg)
    except Exception:
        fallback = json.dumps(
            {
                "invocation_id": context.invocation_id,
                "run_id": context.metadata.run_id,
                "observer_id": observer_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(fallback.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Active invocation session & cooperative RunResource handle
# ---------------------------------------------------------------------------


class _ResourceSamplesHandle:
    """Cooperative :class:`RunResource` handle for one active invocation."""

    def __init__(self, session: _ResourceSamplesSession) -> None:
        self._session = session

    def checkpoint(self) -> None:
        """Advance due cooperative allocator sampling on the owning thread."""

        self._session.checkpoint()

    def request_stop(self) -> None:
        """Signal nonblocking stop without waiting, joining, or flushing."""

        self._session.request_stop()


class _ResourceSamplesSession:
    """Runtime owner for one active ``ResourceSamples.run_scope(context)``."""

    def __init__(
        self,
        callback: ResourceSamples,
        context: RunContext,
    ) -> None:
        self._callback = callback
        self._context = context
        self._monotonic_ns = callback._monotonic_ns
        self._observer_id = callback.observer_id
        self._invocation_id = context.invocation_id
        self._host_id = socket.gethostname() or "unknown-host"
        self._pid = os.getpid()
        self._clock_domain_id = _sanitize_clock_domain_id(self._host_id)
        self._inventory_id = f"inventory-{self._invocation_id}"
        self._descriptor_set_id = f"descriptors-{self._invocation_id}"

        self._lock = threading.Lock()
        self._writer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._drain_stop_event = threading.Event()
        self._drain_wake_event = threading.Event()

        self._pending: deque[tuple[ObservabilitySample, int]] = deque()
        self._next_seq: dict[str, int] = {}
        self._attempted = 0
        self._written = 0
        self._dropped = 0
        self._failed = 0
        self._missed_deadlines = 0
        self._source_windows_ns: dict[str, list[int]] = {}
        self._source_ok_counts: dict[str, int] = {}
        self._source_non_ok_counts: dict[str, int] = {}
        self._source_partial_flags: set[str] = set()
        self._source_unavailable_fields: set[str] = set()
        self._truncation_reasons: list[str] = []
        self._diagnostics: list[str] = []
        self._raw_retention_stopped = False
        self._accepting_writes = True
        self._writer_had_error = False
        self._closed = False
        self._unsafe = False
        self._start_refused = False

        self._writer: ObservabilityBundleWriter | None = None
        self._manifest_base: dict[str, Any] = {}
        self._activation_ns = 0
        self._rusage_probe: Any = None
        self._rusage_subject: dict[str, Any] = {}
        self._rusage_interval_ns = 0
        self._rusage_thread: threading.Thread | None = None
        self._rusage_thread_started = False

        self._linux_memory_probe: Any = None
        self._linux_memory_init_error: LinuxMemoryUnavailable | None = None
        self._linux_memory_target_generation: LinuxProcessGeneration | None = None
        self._linux_memory_subject: dict[str, Any] = {}
        self._linux_memory_interval_ns = 0
        self._linux_memory_thread: threading.Thread | None = None
        self._linux_memory_thread_started = False

        self._linux_tree_probe: Any = None
        self._linux_tree_init_error: LinuxMemoryUnavailable | None = None
        self._linux_tree_target_generation: LinuxProcessGeneration | None = None
        self._linux_tree_subject: dict[str, Any] = {}
        self._linux_tree_interval_ns = 0
        self._linux_tree_thread: threading.Thread | None = None
        self._linux_tree_thread_started = False

        self._linux_cgroup_probe: Any = None
        self._linux_cgroup_init_error: LinuxMemoryUnavailable | None = None
        self._linux_cgroup_target_generation: LinuxProcessGeneration | None = None
        self._linux_cgroup_subject: dict[str, Any] = {}
        self._linux_cgroup_interval_ns = 0
        self._linux_cgroup_thread: threading.Thread | None = None
        self._linux_cgroup_thread_started = False

        self._drain_thread: threading.Thread | None = None
        self._drain_thread_started = False

        self._allocator_current_probe: Any = None
        self._allocator_peak_probe: Any = None
        self._allocator_subject: dict[str, Any] = {}
        self._allocator_interval_ns = 0
        self._allocator_next_due_ns = 0
        self._allocator_current_init_unavailable_reason: tuple[str, str] | None = None
        self._allocator_peak_init_unavailable_reason: tuple[str, str] | None = None
        self._allocator_peak_reset_failure: str | None = None
        self._allocator_reset_epoch_status = "not_requested"
        self._allocator_capability = "unavailable"
        self._allocator_configured_device = context.metadata.device
        self._allocator_resolved_identity: AcceleratorIdentity | None = None
        self._allocator_owner_thread_id = threading.get_ident()

        self._nvidia_device_probe: Any = None
        self._nvidia_device_init_error: NvidiaDeviceUnavailable | None = None
        self._nvidia_device_identity: AcceleratorIdentity | None = None
        self._nvidia_device_configured_device = context.metadata.device
        self._nvidia_device_subject_base: dict[str, Any] = {}
        self._nvidia_device_interval_ns = 0
        self._nvidia_device_thread: threading.Thread | None = None
        self._nvidia_device_thread_started = False
        self._nvidia_device_read_supports_cancel = False
        self._nvidia_device_read_supports_deadline = False

    def start(self) -> RunResource:
        """Activate writer, publish immutable metadata, take initial samples, start threads."""

        try:
            writer = ObservabilityBundleWriter(
                self._context.artifact_manager,
                self._invocation_id,
                sink=self._callback.sink,
            )
        except (ObservabilityPathError, ObservabilitySchemaError):
            self._start_refused = True
            self._writer = None
            raise

        # Capture correlated monotonic + UTC clock anchor.
        bracket_start = int(self._monotonic_ns())
        anchor_utc = datetime.now(UTC).isoformat()
        bracket_end = max(int(self._monotonic_ns()), bracket_start)
        anchor_mid = bracket_start + (bracket_end - bracket_start) // 2
        self._activation_ns = bracket_end

        # Activate configured sources (deferred Torch allocator activation happens here).
        sources_inventory: dict[str, dict[str, Any]] = {}
        selected_sources: dict[str, list[str]] = {}
        descriptors_map: dict[str, QuantityDescriptor] = {}

        if self._callback.rusage.enabled:
            rusage_cfg = self._callback.rusage
            self._rusage_probe = (
                ProcessRUsageProbe() if rusage_cfg.probe is None else rusage_cfg.probe
            )
            rusage_interval = (
                self._callback.interval_seconds
                if rusage_cfg.interval_seconds is None
                else rusage_cfg.interval_seconds
            )
            self._rusage_interval_ns = max(1, int(round(rusage_interval * _NS_PER_SECOND)))
            self._rusage_subject = {
                "kind": "process",
                "host_id": self._host_id,
                "pid": self._pid,
                "invocation_token": self._invocation_id,
            }
            descriptors_map.update(rusage_cfg.descriptors())
            selected_sources[rusage_cfg.source_id] = list(rusage_cfg.quantities)
            sources_inventory[rusage_cfg.source_id] = {
                "producer": "tpen.process_resources.ProcessRUsageProbe",
                "source_version": "1",
                "execution": rusage_cfg.execution_mode,
                "selected_quantities": list(rusage_cfg.quantities),
                "interval_seconds": rusage_interval,
                "capability": "available",
            }

        if self._callback.linux_memory.enabled:
            lm_cfg = self._callback.linux_memory
            self._linux_memory_probe = (
                LinuxProcessMemoryProbe(
                    pid=self._pid,
                    proc_root=lm_cfg.proc_root,
                    max_read_bytes=lm_cfg.max_read_bytes,
                    max_read_seconds=lm_cfg.max_read_seconds,
                    host_id=self._host_id,
                )
                if lm_cfg.probe is None
                else lm_cfg.probe
            )
            lm_interval = (
                self._callback.interval_seconds
                if lm_cfg.interval_seconds is None
                else lm_cfg.interval_seconds
            )
            self._linux_memory_interval_ns = max(
                1, int(round(lm_interval * _NS_PER_SECOND))
            )

            # Bind live target generation at scope activation (never at construction).
            bound_generation: LinuxProcessGeneration | None = None
            bind_error: ResourceUnavailable | None = None
            bind_fn = getattr(self._linux_memory_probe, "bind_target", None)
            if callable(bind_fn):
                try:
                    bound_generation = bind_fn()
                except Exception as exc:
                    bind_error = LinuxMemoryUnavailable(
                        reason=f"{type(exc).__name__}: {exc}",
                        reason_code="probe_init_failed",
                        status="error",
                    )
                    self._linux_memory_init_error = bind_error
                    self._linux_memory_probe = None
                    self._diagnostics.append(
                        f"{lm_cfg.source_id} bind_target failed: {type(exc).__name__}: {exc}"
                    )
                else:
                    if bound_generation is None:
                        raw_bind_error = getattr(
                            self._linux_memory_probe, "bind_error", None
                        )
                        if isinstance(raw_bind_error, ResourceUnavailable):
                            bind_error = raw_bind_error
                        else:
                            bind_error = LinuxMemoryUnavailable(
                                reason=f"{lm_cfg.source_id} bind_target returned None",
                                reason_code="target_bind_failed",
                                status="unavailable",
                            )
            else:
                candidate_gen = getattr(
                    self._linux_memory_probe, "target_generation", None
                )
                if isinstance(candidate_gen, LinuxProcessGeneration):
                    bound_generation = candidate_gen

            self._linux_memory_target_generation = bound_generation
            if bound_generation is not None:
                self._linux_memory_subject = bound_generation.to_subject_dict(
                    host_id=self._host_id,
                    invocation_token=self._invocation_id,
                )
            else:
                self._linux_memory_subject = {
                    "kind": "process",
                    "host_id": self._host_id,
                    "pid": self._pid,
                    "invocation_token": self._invocation_id,
                }

            lm_capability = "unavailable" if bind_error is not None else "available"
            lm_inventory_entry: dict[str, Any] = {
                "producer": "tpen.process_resources.LinuxProcessMemoryProbe",
                "source_version": "1",
                "execution": lm_cfg.execution_mode,
                "selected_quantities": list(lm_cfg.quantities),
                "interval_seconds": lm_interval,
                "max_read_bytes": lm_cfg.max_read_bytes,
                "max_read_seconds": lm_cfg.max_read_seconds,
                "capability": lm_capability,
            }
            if bound_generation is not None:
                lm_inventory_entry["target_generation"] = bound_generation.to_dict()
            if bind_error is not None:
                lm_inventory_entry["unavailable_reason_code"] = getattr(
                    bind_error, "reason_code", "unavailable"
                )

            descriptors_map.update(lm_cfg.descriptors())
            selected_sources[lm_cfg.source_id] = list(lm_cfg.quantities)
            sources_inventory[lm_cfg.source_id] = lm_inventory_entry

        if self._callback.linux_tree.enabled:
            lt_cfg = self._callback.linux_tree
            self._linux_tree_probe = (
                LinuxProcessTreeProbe(
                    pid=self._pid,
                    observer_pids=lt_cfg.observer_pids,
                    proc_root=lt_cfg.proc_root,
                    max_scan_seconds=lt_cfg.max_scan_seconds,
                    max_observed_members=lt_cfg.max_observed_members,
                    max_enumerated_candidates=lt_cfg.max_enumerated_candidates,
                    max_read_bytes=lt_cfg.max_read_bytes,
                    host_id=self._host_id,
                )
                if lt_cfg.probe is None
                else lt_cfg.probe
            )
            lt_interval = (
                self._callback.interval_seconds
                if lt_cfg.interval_seconds is None
                else lt_cfg.interval_seconds
            )
            self._linux_tree_interval_ns = max(
                1, int(round(lt_interval * _NS_PER_SECOND))
            )

            lt_bound_gen: LinuxProcessGeneration | None = None
            lt_bind_error: LinuxMemoryUnavailable | None = None
            lt_bind_fn = getattr(self._linux_tree_probe, "bind_target", None)
            if callable(lt_bind_fn):
                try:
                    lt_bound_gen = lt_bind_fn()
                except Exception as exc:
                    lt_bind_error = LinuxMemoryUnavailable(
                        reason=f"{type(exc).__name__}: {exc}",
                        reason_code="probe_init_failed",
                        status="error",
                    )
                    self._linux_tree_init_error = lt_bind_error
                    self._linux_tree_probe = None
                    self._diagnostics.append(
                        f"{lt_cfg.source_id} bind_target failed: {type(exc).__name__}: {exc}"
                    )
                else:
                    if lt_bound_gen is None:
                        raw_err = getattr(self._linux_tree_probe, "bind_error", None)
                        if isinstance(raw_err, LinuxMemoryUnavailable):
                            lt_bind_error = raw_err
                        elif isinstance(raw_err, ResourceUnavailable):
                            lt_bind_error = LinuxMemoryUnavailable(
                                reason=raw_err.reason or f"{lt_cfg.source_id} bind_target failed",
                                reason_code="target_bind_failed",
                                status="unavailable",
                            )
                        else:
                            lt_bind_error = LinuxMemoryUnavailable(
                                reason=f"{lt_cfg.source_id} bind_target returned None",
                                reason_code="target_bind_failed",
                                status="unavailable",
                            )
                        self._linux_tree_init_error = lt_bind_error
                        self._linux_tree_probe = None
            else:
                candidate_gen = getattr(
                    self._linux_tree_probe, "target_generation", None
                )
                if isinstance(candidate_gen, LinuxProcessGeneration):
                    lt_bound_gen = candidate_gen

            self._linux_tree_target_generation = lt_bound_gen
            root_sub = (
                lt_bound_gen.to_subject_dict(
                    host_id=self._host_id,
                    invocation_token=self._invocation_id,
                )
                if lt_bound_gen is not None
                else {
                    "kind": "process",
                    "host_id": self._host_id,
                    "pid": self._pid,
                    "invocation_token": self._invocation_id,
                }
            )
            self._linux_tree_subject = {
                "kind": "descendant_set",
                "root": root_sub,
                "members": [],
                "membership_complete": False,
                "partial_reasons": (
                    [lt_bind_error.reason_code] if lt_bind_error is not None else []
                ),
                "enumerated_candidates": 0,
                "observed_member_count": 0,
                "matched_delta_member_count": 0,
                "permission_denied_count": 0,
                "excluded_count": 0,
            }

            lt_capability = "unavailable" if lt_bind_error is not None else "available"
            lt_inventory_entry: dict[str, Any] = {
                "producer": "tpen.process_resources.LinuxProcessTreeProbe",
                "source_version": "1",
                "execution": lt_cfg.execution_mode,
                "selected_quantities": list(lt_cfg.quantities),
                "interval_seconds": lt_interval,
                "max_scan_seconds": lt_cfg.max_scan_seconds,
                "max_observed_members": lt_cfg.max_observed_members,
                "max_enumerated_candidates": lt_cfg.max_enumerated_candidates,
                "max_read_bytes": lt_cfg.max_read_bytes,
                "observer_pids": list(lt_cfg.observer_pids),
                "capability": lt_capability,
            }
            if lt_bound_gen is not None:
                lt_inventory_entry["target_generation"] = lt_bound_gen.to_dict()
            if lt_bind_error is not None:
                lt_inventory_entry["unavailable_reason_code"] = lt_bind_error.reason_code

            descriptors_map.update(lt_cfg.descriptors())
            selected_sources[lt_cfg.source_id] = list(lt_cfg.quantities)
            sources_inventory[lt_cfg.source_id] = lt_inventory_entry

        if self._callback.linux_cgroup.enabled:
            lc_cfg = self._callback.linux_cgroup
            self._linux_cgroup_probe = (
                LinuxCgroupV2Probe(
                    pid=self._pid,
                    proc_root=lc_cfg.proc_root,
                    cgroup_root=lc_cfg.cgroup_root,
                    max_read_bytes=lc_cfg.max_read_bytes,
                    max_mountinfo_bytes=lc_cfg.max_mountinfo_bytes,
                    max_read_seconds=lc_cfg.max_read_seconds,
                    max_ancestor_depth=lc_cfg.max_ancestor_depth,
                    host_id=self._host_id,
                )
                if lc_cfg.probe is None
                else lc_cfg.probe
            )
            lc_interval = (
                self._callback.interval_seconds
                if lc_cfg.interval_seconds is None
                else lc_cfg.interval_seconds
            )
            self._linux_cgroup_interval_ns = max(
                1, int(round(lc_interval * _NS_PER_SECOND))
            )

            lc_bound_gen: LinuxProcessGeneration | None = None
            lc_bind_error: LinuxMemoryUnavailable | None = None
            lc_bind_fn = getattr(self._linux_cgroup_probe, "bind_target", None)
            if callable(lc_bind_fn):
                try:
                    lc_bound_gen = lc_bind_fn()
                except Exception as exc:
                    lc_bind_error = LinuxMemoryUnavailable(
                        reason=f"{type(exc).__name__}: {exc}",
                        reason_code="probe_init_failed",
                        status="error",
                    )
                    self._linux_cgroup_init_error = lc_bind_error
                    self._linux_cgroup_probe = None
                    self._diagnostics.append(
                        f"{lc_cfg.source_id} bind_target failed: {type(exc).__name__}: {exc}"
                    )
                else:
                    if lc_bound_gen is None:
                        raw_err = getattr(self._linux_cgroup_probe, "bind_error", None)
                        if isinstance(raw_err, LinuxMemoryUnavailable):
                            lc_bind_error = raw_err
                        elif isinstance(raw_err, ResourceUnavailable):
                            lc_bind_error = LinuxMemoryUnavailable(
                                reason=raw_err.reason or f"{lc_cfg.source_id} bind_target failed",
                                reason_code="target_bind_failed",
                                status="unavailable",
                            )
                        else:
                            lc_bind_error = LinuxMemoryUnavailable(
                                reason=f"{lc_cfg.source_id} bind_target returned None",
                                reason_code="target_bind_failed",
                                status="unavailable",
                            )
                        self._linux_cgroup_init_error = lc_bind_error
                        self._linux_cgroup_probe = None
            else:
                candidate_gen = getattr(
                    self._linux_cgroup_probe, "target_generation", None
                )
                if isinstance(candidate_gen, LinuxProcessGeneration):
                    lc_bound_gen = candidate_gen

            self._linux_cgroup_target_generation = lc_bound_gen
            lc_identity: LinuxCgroupV2Identity | None = None
            lc_cgroup_issue: LinuxMemoryUnavailable | None = None
            if lc_bind_error is None and self._linux_cgroup_probe is not None:
                resolve_fn = getattr(
                    self._linux_cgroup_probe, "resolve_cgroup_identity", None
                )
                if callable(resolve_fn):
                    try:
                        resolved = resolve_fn()
                    except Exception as exc:
                        lc_cgroup_issue = LinuxMemoryUnavailable(
                            reason=f"{type(exc).__name__}: {exc}",
                            reason_code="cgroup_resolve_failed",
                            status="error",
                        )
                        self._linux_cgroup_init_error = lc_cgroup_issue
                        self._linux_cgroup_probe = None
                        self._diagnostics.append(
                            f"{lc_cfg.source_id} resolve_cgroup_identity failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        if isinstance(resolved, LinuxCgroupV2Identity):
                            lc_identity = resolved
                        elif isinstance(resolved, LinuxMemoryUnavailable):
                            lc_cgroup_issue = resolved
                        else:
                            raw_cg_err = getattr(
                                self._linux_cgroup_probe, "cgroup_error", None
                            )
                            if isinstance(raw_cg_err, LinuxMemoryUnavailable):
                                lc_cgroup_issue = raw_cg_err
                            elif isinstance(resolved, ResourceUnavailable):
                                lc_cgroup_issue = LinuxMemoryUnavailable(
                                    reason=(
                                        resolved.reason
                                        or f"{lc_cfg.source_id} resolve_cgroup_identity unavailable"
                                    ),
                                    reason_code="cgroup_resolve_failed",
                                    status="unavailable",
                                )
                            elif isinstance(raw_cg_err, ResourceUnavailable):
                                lc_cgroup_issue = LinuxMemoryUnavailable(
                                    reason=(
                                        raw_cg_err.reason
                                        or f"{lc_cfg.source_id} resolve_cgroup_identity failed"
                                    ),
                                    reason_code="cgroup_resolve_failed",
                                    status="unavailable",
                                )
                            else:
                                lc_cgroup_issue = LinuxMemoryUnavailable(
                                    reason=(
                                        f"{lc_cfg.source_id} resolve_cgroup_identity returned None"
                                        if resolved is None
                                        else f"{lc_cfg.source_id} resolve_cgroup_identity returned {type(resolved).__name__}"
                                    ),
                                    reason_code="cgroup_resolve_failed",
                                    status="unavailable",
                                )

            if lc_identity is not None:
                self._linux_cgroup_subject = lc_identity.to_subject_dict(
                    target_generation=lc_bound_gen
                )
            else:
                fallback_cg_sub: dict[str, Any] = {
                    "kind": "cgroup",
                    "cgroup_version": 2,
                    "mount_root": None,
                    "cgroup_path": None,
                    "effective_memory_limit_path": None,
                    "effective_cpu_limit_path": None,
                    "observed_member_count": None,
                    "includes_other_tasks": None,
                    "shared_scope_not_exclusive": True,
                }
                if lc_bound_gen is not None:
                    fallback_cg_sub["target_generation"] = lc_bound_gen.to_dict()
                self._linux_cgroup_subject = fallback_cg_sub

            lc_issue = lc_bind_error if lc_bind_error is not None else lc_cgroup_issue
            lc_capability = "unavailable" if lc_issue is not None else "available"
            lc_inventory_entry: dict[str, Any] = {
                "producer": "tpen.process_resources.LinuxCgroupV2Probe",
                "source_version": "1",
                "execution": lc_cfg.execution_mode,
                "selected_quantities": list(lc_cfg.quantities),
                "interval_seconds": lc_interval,
                "max_read_bytes": lc_cfg.max_read_bytes,
                "max_mountinfo_bytes": lc_cfg.max_mountinfo_bytes,
                "max_read_seconds": lc_cfg.max_read_seconds,
                "max_ancestor_depth": lc_cfg.max_ancestor_depth,
                "cgroup_version": 2,
                "shared_scope_not_exclusive": True,
                "capability": lc_capability,
            }
            if lc_bound_gen is not None:
                lc_inventory_entry["target_generation"] = lc_bound_gen.to_dict()
            if lc_identity is not None:
                lc_inventory_entry["mount_root"] = lc_identity.mount_root
                lc_inventory_entry["cgroup_path"] = lc_identity.cgroup_path
            if lc_issue is not None:
                lc_inventory_entry["unavailable_reason_code"] = lc_issue.reason_code

            descriptors_map.update(lc_cfg.descriptors())
            selected_sources[lc_cfg.source_id] = list(lc_cfg.quantities)
            sources_inventory[lc_cfg.source_id] = lc_inventory_entry

        if self._callback.allocator.enabled:
            alloc_cfg = self._callback.allocator
            self._activate_allocator_source(alloc_cfg)
            alloc_interval = (
                self._callback.interval_seconds
                if alloc_cfg.interval_seconds is None
                else alloc_cfg.interval_seconds
            )
            self._allocator_interval_ns = max(
                1, int(round(alloc_interval * _NS_PER_SECOND))
            )
            descriptors_map.update(alloc_cfg.descriptors())
            selected_sources[alloc_cfg.source_id] = list(alloc_cfg.selected_quantities)
            sources_inventory[alloc_cfg.source_id] = {
                "producer": "tpen.accelerator.TorchAllocatorCurrentProbe+TorchAllocatorPeakProbe",
                "source_version": "1",
                "execution": alloc_cfg.execution_mode,
                "selected_quantities": list(alloc_cfg.selected_quantities),
                "configured_device": self._allocator_configured_device,
                "reset_peaks": alloc_cfg.reset_peaks,
                "reset_epoch_status": self._allocator_reset_epoch_status,
                "interval_seconds": alloc_interval,
                "capability": self._allocator_capability,
            }

        if self._callback.nvidia_device.enabled:
            nd_cfg = self._callback.nvidia_device
            nd_inventory_entry = self._activate_nvidia_device_source(nd_cfg)
            descriptors_map.update(nd_cfg.descriptors())
            selected_sources[nd_cfg.source_id] = list(nd_cfg.quantities)
            sources_inventory[nd_cfg.source_id] = nd_inventory_entry

        observer_entry: dict[str, Any] = {
            "observer_id": self._observer_id,
            "callback_type": f"{type(self._callback).__module__}.{type(self._callback).__qualname__}",
            "selected_sources": selected_sources,
        }
        producer_info: dict[str, Any] = {
            "name": f"{type(self._callback).__module__}.{type(self._callback).__qualname__}",
            "version": "1",
            "source_commit": (
                {"status": "ok", "value": self._context.metadata.git_commit}
                if self._context.metadata.git_commit
                else {"status": "unavailable", "reason_code": "not_supplied"}
            ),
        }
        limits_info: dict[str, Any] = {
            "interval_seconds": self._callback.interval_seconds,
            "close_timeout_seconds": self._callback.close_timeout_seconds,
            "max_pending_envelopes": self._callback.max_pending_envelopes,
            "max_pending_bytes": self._callback.sink.max_pending_bytes,
            "raw_sample_bytes": self._callback.sink.max_sample_bytes,
            "status_bytes": self._callback.sink.max_status_bytes,
        }
        config_digest = _compute_configuration_digest(self._context, self._observer_id)
        self._manifest_base = {
            "invocation_id": self._invocation_id,
            "run_id": self._context.metadata.run_id,
            "producer": producer_info,
            "configuration_digest": config_digest,
            "inventory_id": self._inventory_id,
            "descriptor_set_id": self._descriptor_set_id,
            "observers": (observer_entry,),
            "limits": limits_info,
        }

        open_manifest = ObservabilityManifest(
            lifecycle_state="open",
            data_completeness="unknown",
            **self._manifest_base,
        )
        host_inventory: dict[str, Any] = {
            "host_id": self._host_id,
            "platform": sys.platform,
        }
        process_inventory: dict[str, Any] = {
            "kind": "process",
            "host_id": self._host_id,
            "pid": self._pid,
            "invocation_token": self._invocation_id,
        }
        bound_linux_gen = (
            self._linux_memory_target_generation
            or self._linux_tree_target_generation
            or self._linux_cgroup_target_generation
        )
        if bound_linux_gen is not None:
            gen = bound_linux_gen
            if gen.boot_id is not None:
                host_inventory["boot_id"] = gen.boot_id
            if gen.pid == self._pid:
                process_inventory["start_ticks"] = gen.start_ticks
                if gen.boot_id is not None:
                    process_inventory["boot_id"] = gen.boot_id
                if gen.pid_ns_inode is not None:
                    process_inventory["pid_ns_inode"] = gen.pid_ns_inode

        inventory = ObservabilityInventory(
            invocation_id=self._invocation_id,
            inventory_id=self._inventory_id,
            host=host_inventory,
            process=process_inventory,
            clock_domains=(
                {
                    "clock_domain_id": self._clock_domain_id,
                    "kind": "host_monotonic_ns",
                    "host_id": self._host_id,
                    "anchor_monotonic_ns": anchor_mid,
                    "anchor_read_bracket_ns": [bracket_start, bracket_end],
                    "anchor_utc": anchor_utc,
                    "run_timezone": self._context.metadata.timezone,
                },
            ),
            requested={
                "device": self._context.metadata.device,
                "dtype": self._context.metadata.dtype,
            },
            observed={
                "software": {
                    "python": platform.python_version(),
                },
            },
            sources=sources_inventory,
        )
        descriptor_set = ObservabilityDescriptorSet(
            invocation_id=self._invocation_id,
            descriptor_set_id=self._descriptor_set_id,
            quantities=descriptors_map,
        )

        # Write initial metadata before starting producers so readers always see
        # open manifest, inventory, and descriptors before any sample rows.
        try:
            with self._writer_lock:
                writer.write_open_manifest(open_manifest)
                writer.write_inventory(inventory)
                writer.write_descriptors(descriptor_set)
        except (ObservabilityPathError, ObservabilitySchemaError):
            # Refuse containment/schema violations cleanly without emitting
            # post-refusal status artifacts or closing manifest during rollback.
            self._start_refused = True
            self._writer = None
            raise
        except (OSError, ObservabilityArtifactError) as exc:
            # Advisory observability I/O failure is contained so science continues.
            self._record_writer_error(f"bundle initialization failed: {type(exc).__name__}: {exc}")
        self._writer = writer

        # Start bounded background drain thread so sample persistence never
        # blocks the scientific thread.
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            name=f"tpen-obs-drain-{self._observer_id}",
            daemon=True,
        )
        self._drain_thread.start()
        self._drain_thread_started = True

        # Initial reachable capture on activation.
        if self._callback.rusage.enabled:
            self._sample_rusage()
            self._rusage_thread = threading.Thread(
                target=self._rusage_loop,
                name=f"tpen-obs-rusage-{self._observer_id}",
                daemon=True,
            )
            self._rusage_thread.start()
            self._rusage_thread_started = True

        if self._callback.linux_memory.enabled:
            self._sample_linux_memory()
            if self._linux_memory_init_error is None:
                self._linux_memory_thread = threading.Thread(
                    target=self._linux_memory_loop,
                    name=f"tpen-obs-linux-memory-{self._observer_id}",
                    daemon=True,
                )
                self._linux_memory_thread.start()
                self._linux_memory_thread_started = True

        if self._callback.linux_tree.enabled:
            self._sample_linux_tree()
            if self._linux_tree_init_error is None:
                self._linux_tree_thread = threading.Thread(
                    target=self._linux_tree_loop,
                    name=f"tpen-obs-linux-tree-{self._observer_id}",
                    daemon=True,
                )
                self._linux_tree_thread.start()
                self._linux_tree_thread_started = True

        if self._callback.linux_cgroup.enabled:
            self._sample_linux_cgroup()
            if self._linux_cgroup_init_error is None:
                self._linux_cgroup_thread = threading.Thread(
                    target=self._linux_cgroup_loop,
                    name=f"tpen-obs-linux-cgroup-{self._observer_id}",
                    daemon=True,
                )
                self._linux_cgroup_thread.start()
                self._linux_cgroup_thread_started = True

        if self._callback.allocator.enabled:
            read_end_ns = self._sample_allocator()
            self._allocator_next_due_ns = read_end_ns + self._allocator_interval_ns

        if self._callback.nvidia_device.enabled:
            self._sample_nvidia_device(cancel_event=self._stop_event)
            if self._nvidia_device_init_error is None:
                self._nvidia_device_thread = threading.Thread(
                    target=self._nvidia_device_loop,
                    name=f"tpen-obs-nvidia-device-{self._observer_id}",
                    daemon=True,
                )
                self._nvidia_device_thread.start()
                self._nvidia_device_thread_started = True

        return _ResourceSamplesHandle(self)

    def _activate_allocator_source(self, alloc_cfg: AllocatorSource) -> None:
        """Deferred activation of Torch allocator probes inside ``run_scope``."""

        configured_device = alloc_cfg.device or self._context.metadata.device
        self._allocator_configured_device = configured_device

        current_factory = alloc_cfg.current_probe_factory
        peak_factory = alloc_cfg.peak_probe_factory
        if (alloc_cfg.include_current and current_factory is None) or (
            alloc_cfg.include_peak and peak_factory is None
        ):
            try:
                from tpen.accelerator import (
                    TorchAllocatorCurrentProbe,
                    TorchAllocatorPeakProbe,
                )
            except OptionalDependencyError as exc:
                reason = (
                    "torch_unavailable",
                    str(exc) or "torch is not installed",
                )
                if alloc_cfg.include_current:
                    self._allocator_current_init_unavailable_reason = reason
                if alloc_cfg.include_peak:
                    self._allocator_peak_init_unavailable_reason = reason
                self._allocator_reset_epoch_status = (
                    "unavailable" if alloc_cfg.reset_peaks else "not_requested"
                )
                self._allocator_capability = "unavailable"
                self._allocator_subject = self._build_allocator_subject(
                    configured_device=configured_device,
                    identity=None,
                )
                return
            if alloc_cfg.include_current and current_factory is None:
                current_factory = TorchAllocatorCurrentProbe
            if alloc_cfg.include_peak and peak_factory is None:
                peak_factory = TorchAllocatorPeakProbe

        identity: Any = None
        if alloc_cfg.include_current and current_factory is not None:
            try:
                self._allocator_current_probe = current_factory(configured_device)
            except OptionalDependencyError as exc:
                self._allocator_current_init_unavailable_reason = (
                    "torch_unavailable",
                    str(exc) or "torch is not installed",
                )
            except Exception as exc:
                self._allocator_current_init_unavailable_reason = (
                    "probe_init_failed",
                    f"{type(exc).__name__}: {exc}",
                )
            else:
                id_fn = getattr(self._allocator_current_probe, "identity", None)
                if callable(id_fn):
                    try:
                        identity = id_fn()
                    except Exception:
                        identity = None

        if alloc_cfg.include_peak and peak_factory is not None:
            try:
                self._allocator_peak_probe = peak_factory(configured_device)
            except OptionalDependencyError as exc:
                self._allocator_peak_init_unavailable_reason = (
                    "torch_unavailable",
                    str(exc) or "torch is not installed",
                )
                if alloc_cfg.reset_peaks:
                    self._allocator_reset_epoch_status = "unavailable"
            except Exception as exc:
                self._allocator_peak_init_unavailable_reason = (
                    "probe_init_failed",
                    f"{type(exc).__name__}: {exc}",
                )
                if alloc_cfg.reset_peaks:
                    self._allocator_reset_epoch_status = "unavailable"
            else:
                if alloc_cfg.reset_peaks:
                    try:
                        reset_result = self._allocator_peak_probe.reset()
                    except Exception as exc:
                        self._allocator_peak_reset_failure = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._allocator_reset_epoch_status = "reset_failed"
                    else:
                        if _is_allocator_unavailable(reset_result):
                            self._allocator_peak_reset_failure = (
                                str(getattr(reset_result, "reason", "")).strip()
                                or "allocator reset unavailable"
                            )
                            self._allocator_reset_epoch_status = "reset_failed"
                        else:
                            identity = reset_result if identity is None else identity
                            if _identity_kind_text(reset_result) == "cpu":
                                self._allocator_reset_epoch_status = "cpu_no_reset"
                            else:
                                self._allocator_reset_epoch_status = "owned_reset"
                else:
                    self._allocator_reset_epoch_status = "not_requested"
                    if identity is None:
                        id_fn = getattr(self._allocator_peak_probe, "identity", None)
                        if callable(id_fn):
                            try:
                                identity = id_fn()
                            except Exception:
                                identity = None

        if identity is None and self._allocator_peak_probe is not None:
            id_fn = getattr(self._allocator_peak_probe, "identity", None)
            if callable(id_fn):
                try:
                    identity = id_fn()
                except Exception:
                    identity = None

        current_available = alloc_cfg.include_current and _probe_is_available(
            self._allocator_current_probe, identity
        )
        peak_available = (
            alloc_cfg.include_peak
            and self._allocator_peak_reset_failure is None
            and _probe_is_available(self._allocator_peak_probe, identity)
        )
        requested_parts = int(alloc_cfg.include_current) + int(alloc_cfg.include_peak)
        available_parts = int(current_available) + int(peak_available)
        if available_parts == 0:
            self._allocator_capability = "unavailable"
        elif available_parts == requested_parts:
            self._allocator_capability = "available"
        else:
            self._allocator_capability = "partial"

        if isinstance(identity, AcceleratorIdentity):
            self._allocator_resolved_identity = identity
        self._allocator_subject = self._build_allocator_subject(
            configured_device=configured_device,
            identity=identity,
        )

    def _activate_nvidia_device_source(
        self, nd_cfg: NvidiaDeviceSource
    ) -> dict[str, Any]:
        """Lazily resolve selected physical GPU identity and build NVIDIA probe."""

        configured_device = nd_cfg.device or self._context.metadata.device
        self._nvidia_device_configured_device = configured_device
        nd_interval = (
            self._callback.interval_seconds
            if nd_cfg.interval_seconds is None
            else nd_cfg.interval_seconds
        )
        self._nvidia_device_interval_ns = max(
            1, int(round(nd_interval * _NS_PER_SECOND))
        )

        selected_identity: AcceleratorIdentity | None = None
        init_error: NvidiaDeviceUnavailable | None = None

        if nd_cfg.identity is not None:
            selected_identity = nd_cfg.identity
        elif nd_cfg.identity_resolver is not None:
            try:
                raw_identity = nd_cfg.identity_resolver(configured_device)
                selected_identity = _coerce_accelerator_identity(raw_identity)
            except OptionalDependencyError as exc:
                init_error = NvidiaDeviceUnavailable(
                    reason=str(exc) or "torch is not installed",
                    reason_code="torch_unavailable",
                    status="unavailable",
                )
            except Exception as exc:
                init_error = NvidiaDeviceUnavailable(
                    reason=f"{type(exc).__name__}: {exc}",
                    reason_code="identity_resolution_failed",
                    status="error",
                )
        elif nd_cfg.probe is not None:
            candidate_identity = getattr(nd_cfg.probe, "identity", None)
            if isinstance(candidate_identity, AcceleratorIdentity):
                selected_identity = candidate_identity
        elif (
            self._callback.allocator.enabled
            and self._allocator_resolved_identity is not None
            and self._allocator_configured_device == configured_device
        ):
            selected_identity = self._allocator_resolved_identity
        else:
            try:
                from tpen.accelerator import resolve_selected_accelerator_identity

                selected_identity = resolve_selected_accelerator_identity(
                    configured_device
                )
            except OptionalDependencyError as exc:
                init_error = NvidiaDeviceUnavailable(
                    reason=str(exc) or "torch is not installed",
                    reason_code="torch_unavailable",
                    status="unavailable",
                )
            except Exception as exc:
                init_error = NvidiaDeviceUnavailable(
                    reason=f"{type(exc).__name__}: {exc}",
                    reason_code="identity_resolution_failed",
                    status="error",
                )

        self._nvidia_device_identity = selected_identity
        self._nvidia_device_init_error = init_error
        self._nvidia_device_subject_base = self._build_nvidia_device_subject(
            configured_device=configured_device,
            identity=selected_identity,
        )

        preflight_issue: NvidiaDeviceUnavailable | None = init_error
        if init_error is None:
            if nd_cfg.probe is not None:
                self._nvidia_device_probe = nd_cfg.probe
            else:
                # Always enforce subprocess wall-clock timeouts and close()
                # deadlines on real monotonic time, even when sample timestamps
                # use an injected test clock.
                self._nvidia_device_probe = NvidiaSmiDeviceProbe(
                    identity=selected_identity,
                    executable=nd_cfg.executable,
                    timeout_seconds=nd_cfg.timeout_seconds,
                    max_output_bytes=nd_cfg.max_output_bytes,
                )
            read_fn = getattr(self._nvidia_device_probe, "read", None)
            if callable(read_fn):
                try:
                    sig = inspect.signature(read_fn)
                except (TypeError, ValueError):
                    self._nvidia_device_read_supports_cancel = False
                    self._nvidia_device_read_supports_deadline = False
                else:
                    params = sig.parameters
                    has_var_kw = any(
                        p.kind is inspect.Parameter.VAR_KEYWORD
                        for p in params.values()
                    )
                    self._nvidia_device_read_supports_cancel = (
                        has_var_kw or "cancel_event" in params
                    )
                    self._nvidia_device_read_supports_deadline = (
                        has_var_kw or "deadline_ns" in params
                    )
            preflight_fn = getattr(self._nvidia_device_probe, "preflight_issue", None)
            if callable(preflight_fn):
                try:
                    preflight_issue = preflight_fn()
                except Exception:
                    preflight_issue = None
        else:
            self._nvidia_device_probe = None
            self._diagnostics.append(
                f"{nd_cfg.source_id} identity resolution failed: {init_error.reason}"
            )

        capability = "unavailable" if preflight_issue is not None else "available"
        entry: dict[str, Any] = {
            "producer": "tpen.accelerator.NvidiaSmiDeviceProbe",
            "source_version": "1",
            "execution": nd_cfg.execution_mode,
            "selected_quantities": list(nd_cfg.quantities),
            "interval_seconds": nd_interval,
            "configured_device": configured_device,
            "timeout_seconds": nd_cfg.timeout_seconds,
            "max_output_bytes": nd_cfg.max_output_bytes,
            "shared_device_not_exclusive": True,
            "per_process_attribution": False,
            "allocator_reset_authority": "none",
            "selected_identity": (
                {
                    "kind": _identity_kind_text(selected_identity),
                    "index": selected_identity.index,
                    "uuid": selected_identity.uuid,
                }
                if selected_identity is not None
                else None
            ),
            "capability": capability,
        }
        if preflight_issue is not None:
            entry["unavailable_reason_code"] = preflight_issue.reason_code
        return entry

    def _build_nvidia_device_subject(
        self,
        *,
        configured_device: str,
        identity: AcceleratorIdentity | None,
    ) -> dict[str, Any]:
        backend = _identity_kind_text(identity) if identity is not None else "unknown"
        device_index: int | None = None
        device_uuid: str | None = None
        if identity is not None:
            if type(identity.index) is int:
                device_index = identity.index
            if isinstance(identity.uuid, str) and identity.uuid:
                canonical = canonicalize_nvidia_device_uuid(identity.uuid)
                device_uuid = (
                    canonical if isinstance(canonical, str) else identity.uuid
                )
        return {
            "kind": "physical_gpu",
            "host_id": self._host_id,
            "backend": backend,
            "configured_device": configured_device,
            "device_index": device_index,
            "device_uuid": device_uuid,
            "shared_device_not_exclusive": True,
            "per_process_attribution": False,
        }

    def _build_allocator_subject(
        self,
        *,
        configured_device: str,
        identity: Any,
    ) -> dict[str, Any]:
        backend = _identity_kind_text(identity) if identity is not None else "unknown"
        subject: dict[str, Any] = {
            "kind": "process_allocator",
            "host_id": self._host_id,
            "pid": self._pid,
            "invocation_token": self._invocation_id,
            "configured_device": configured_device,
            "backend": backend,
            "reset_epoch_owner": (
                "self" if self._allocator_reset_epoch_status == "owned_reset" else "none"
            ),
        }
        if identity is not None:
            index = getattr(identity, "index", None)
            if type(index) is int:
                subject["device_index"] = index
            uuid = getattr(identity, "uuid", None)
            if isinstance(uuid, str) and uuid:
                subject["device_uuid"] = uuid
        return subject

    def _rusage_loop(self) -> None:
        """Monotonic-deadline sampling loop for in-process ``RUSAGE_SELF``."""

        next_due_ns = int(self._monotonic_ns()) + self._rusage_interval_ns
        while not self._stop_event.is_set():
            now_ns = int(self._monotonic_ns())
            wait_ns = next_due_ns - now_ns
            if wait_ns > 0:
                if self._stop_event.wait(timeout=wait_ns / _NS_PER_SECOND):
                    break
                now_ns = int(self._monotonic_ns())
                if now_ns < next_due_ns:
                    continue
            if self._stop_event.is_set():
                break
            # Advance deadline without a catch-up storm if scheduling fell behind.
            lag_ns = now_ns - next_due_ns
            if lag_ns >= self._rusage_interval_ns:
                missed = lag_ns // self._rusage_interval_ns
                with self._lock:
                    self._missed_deadlines += int(missed)
                next_due_ns = now_ns + self._rusage_interval_ns
            else:
                next_due_ns += self._rusage_interval_ns
            self._sample_rusage()

    def _sample_rusage(self) -> None:
        """Capture one self-process ``RUSAGE`` sample and enqueue nonblockingly."""

        source_id = self._callback.rusage.source_id
        read_start_ns = int(self._monotonic_ns())
        try:
            baseline = self._rusage_probe.read()
        except Exception as exc:
            with self._lock:
                self._attempted += 1
                self._failed += 1
                self._diagnostics.append(
                    f"{source_id} read failed: {type(exc).__name__}: {exc}"
                )
            return
        read_end_ns = max(int(self._monotonic_ns()), read_start_ns)
        try:
            readings = _rusage_baseline_to_readings(
                baseline, self._callback.rusage.quantities
            )
        except Exception as exc:
            with self._lock:
                self._attempted += 1
                self._failed += 1
                self._diagnostics.append(
                    f"{source_id} reading conversion error: {type(exc).__name__}: {exc}"
                )
            return
        self._enqueue_reading_packet(
            source_id=source_id,
            subject=self._rusage_subject,
            read_start_ns=read_start_ns,
            read_end_ns=read_end_ns,
            readings=readings,
        )

    def _linux_memory_loop(self) -> None:
        """Monotonic-deadline sampling loop for bounded Linux procfs current memory."""

        next_due_ns = int(self._monotonic_ns()) + self._linux_memory_interval_ns
        while not self._stop_event.is_set():
            now_ns = int(self._monotonic_ns())
            wait_ns = next_due_ns - now_ns
            if wait_ns > 0:
                if self._stop_event.wait(timeout=wait_ns / _NS_PER_SECOND):
                    break
                now_ns = int(self._monotonic_ns())
                if now_ns < next_due_ns:
                    continue
            if self._stop_event.is_set():
                break
            # Advance deadline without a catch-up storm if scheduling fell behind.
            lag_ns = now_ns - next_due_ns
            if lag_ns >= self._linux_memory_interval_ns:
                missed = lag_ns // self._linux_memory_interval_ns
                with self._lock:
                    self._missed_deadlines += int(missed)
                next_due_ns = now_ns + self._linux_memory_interval_ns
            else:
                next_due_ns += self._linux_memory_interval_ns
            self._sample_linux_memory()

    def _sample_linux_memory(self) -> None:
        """Capture one Linux procfs current-memory sample and enqueue nonblockingly."""

        lm_cfg = self._callback.linux_memory
        source_id = lm_cfg.source_id
        # Bracket probe.read() on the session's clock domain so sample timestamps
        # stay correlated with inventory.clock_domains and coverage windows.
        read_start_ns = int(self._monotonic_ns())
        init_error = self._linux_memory_init_error
        if init_error is not None or self._linux_memory_probe is None:
            issue = (
                init_error
                if init_error is not None
                else LinuxMemoryUnavailable(
                    reason=f"{source_id} probe unavailable",
                    reason_code="probe_init_failed",
                    status="error",
                )
            )
            read_end_ns = max(int(self._monotonic_ns()), read_start_ns)
            readings = {
                qty_id: ObservabilityReading(
                    status=issue.status,
                    reason_code=issue.reason_code,
                    detail=issue.reason,
                )
                for qty_id in lm_cfg.quantities
            }
            self._enqueue_reading_packet(
                source_id=source_id,
                subject=self._linux_memory_subject,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                readings=readings,
            )
            return

        try:
            reading = self._linux_memory_probe.read()
        except Exception as exc:
            with self._lock:
                self._attempted += 1
                self._failed += 1
                self._diagnostics.append(
                    f"{source_id} read failed: {type(exc).__name__}: {exc}"
                )
            return
        read_end_ns = max(int(self._monotonic_ns()), read_start_ns)

        target_gen = getattr(reading, "target_generation", None)
        if isinstance(target_gen, LinuxProcessGeneration):
            subject = target_gen.to_subject_dict(
                host_id=self._host_id,
                invocation_token=self._invocation_id,
            )
            if self._linux_memory_target_generation is None:
                self._linux_memory_target_generation = target_gen
                self._linux_memory_subject = subject
        else:
            subject = self._linux_memory_subject

        try:
            readings = _linux_memory_reading_to_readings(reading, lm_cfg.quantities)
        except Exception as exc:
            with self._lock:
                self._attempted += 1
                self._failed += 1
                self._diagnostics.append(
                    f"{source_id} reading conversion error: {type(exc).__name__}: {exc}"
                )
            return

        self._enqueue_reading_packet(
            source_id=source_id,
            subject=subject,
            read_start_ns=read_start_ns,
            read_end_ns=read_end_ns,
            readings=readings,
        )

    def _descendant_record_budget_bytes(self) -> int:
        sink = self._callback.sink
        candidates = [
            sink.max_sample_bytes,
            sink.max_pending_bytes,
            _DEFAULT_DESCENDANT_SAMPLE_RECORD_BUDGET_BYTES,
        ]
        raw_record_cap = getattr(sink, "max_record_bytes", None)
        if type(raw_record_cap) is int and raw_record_cap >= 1:
            candidates.append(raw_record_cap)
        return max(1, min(candidates))

    def _build_bounded_descendant_subject(
        self,
        reading: LinuxProcessTreeReading,
        *,
        source_id: str,
        read_start_ns: int,
        read_end_ns: int,
        readings: Mapping[str, ObservabilityReading],
    ) -> dict[str, Any]:
        """Build O2 ``descendant_set`` subject bounded cleanly within the record budget."""

        full_subject = reading.to_subject_dict(
            host_id=self._host_id,
            invocation_token=self._invocation_id,
            fallback_pid=self._pid,
        )
        member_count = len(reading.members)
        if member_count == 0:
            return full_subject

        budget = self._descendant_record_budget_bytes()
        probe_sample = ObservabilitySample(
            invocation_id=self._invocation_id,
            observer_id=self._observer_id,
            source_id=source_id,
            descriptor_set_id=self._descriptor_set_id,
            sequence=0,
            clock_domain_id=self._clock_domain_id,
            subject=full_subject,
            read_start_ns=read_start_ns,
            read_end_ns=read_end_ns,
            readings=readings,
        )
        # Reserve a small margin for larger sequence integers.
        margin = 32
        if _encode_sample_line_bytes(probe_sample) + margin <= budget:
            return full_subject

        # Binary search largest prefix of members that fits cleanly within budget.
        lo, hi = 0, member_count - 1
        best_k = 0
        best_subject = reading.to_subject_dict(
            host_id=self._host_id,
            invocation_token=self._invocation_id,
            fallback_pid=self._pid,
            max_members=0,
        )
        while lo <= hi:
            mid = (lo + hi) // 2
            cand_subject = reading.to_subject_dict(
                host_id=self._host_id,
                invocation_token=self._invocation_id,
                fallback_pid=self._pid,
                max_members=mid,
            )
            cand_sample = ObservabilitySample(
                invocation_id=self._invocation_id,
                observer_id=self._observer_id,
                source_id=source_id,
                descriptor_set_id=self._descriptor_set_id,
                sequence=0,
                clock_domain_id=self._clock_domain_id,
                subject=cand_subject,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                readings=readings,
            )
            if _encode_sample_line_bytes(cand_sample) + margin <= budget:
                best_k = mid
                best_subject = cand_subject
                lo = mid + 1
            else:
                hi = mid - 1

        with self._lock:
            _append_unique(self._truncation_reasons, "descendant_members_truncated")
        if best_k < member_count:
            best_subject["members_truncated_for_record_budget"] = True
            best_subject["serialized_member_count"] = best_k
        return best_subject

    def _linux_tree_loop(self) -> None:
        """Monotonic-deadline sampling loop for bounded Linux descendant scans."""

        next_due_ns = int(self._monotonic_ns()) + self._linux_tree_interval_ns
        while not self._stop_event.is_set():
            now_ns = int(self._monotonic_ns())
            wait_ns = next_due_ns - now_ns
            if wait_ns > 0:
                if self._stop_event.wait(timeout=wait_ns / _NS_PER_SECOND):
                    break
                now_ns = int(self._monotonic_ns())
                if now_ns < next_due_ns:
                    continue
            if self._stop_event.is_set():
                break
            lag_ns = now_ns - next_due_ns
            if lag_ns >= self._linux_tree_interval_ns:
                missed = lag_ns // self._linux_tree_interval_ns
                with self._lock:
                    self._missed_deadlines += int(missed)
                next_due_ns = now_ns + self._linux_tree_interval_ns
            else:
                next_due_ns += self._linux_tree_interval_ns
            self._sample_linux_tree()

    def _sample_linux_tree(self) -> None:
        """Capture one bounded Linux descendant-tree sample and enqueue nonblockingly."""

        lt_cfg = self._callback.linux_tree
        source_id = lt_cfg.source_id
        read_start_ns = int(self._monotonic_ns())
        init_error = self._linux_tree_init_error
        if init_error is not None or self._linux_tree_probe is None:
            issue = (
                init_error
                if init_error is not None
                else LinuxMemoryUnavailable(
                    reason=f"{source_id} probe unavailable",
                    reason_code="probe_init_failed",
                    status="error",
                )
            )
            read_end_ns = max(int(self._monotonic_ns()), read_start_ns)
            readings = {
                qty_id: ObservabilityReading(
                    status=issue.status,
                    reason_code=issue.reason_code,
                    detail=issue.reason,
                )
                for qty_id in lt_cfg.quantities
            }
            self._enqueue_reading_packet(
                source_id=source_id,
                subject=self._linux_tree_subject,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                readings=readings,
            )
            return

        try:
            reading = self._linux_tree_probe.read()
        except Exception as exc:
            with self._lock:
                self._attempted += 1
                self._failed += 1
                self._diagnostics.append(
                    f"{source_id} read failed: {type(exc).__name__}: {exc}"
                )
            return
        read_end_ns = max(int(self._monotonic_ns()), read_start_ns)

        try:
            readings = _linux_tree_reading_to_readings(reading, lt_cfg.quantities)
        except Exception as exc:
            with self._lock:
                self._attempted += 1
                self._failed += 1
                self._diagnostics.append(
                    f"{source_id} reading conversion error: {type(exc).__name__}: {exc}"
                )
            return

        if isinstance(reading, LinuxProcessTreeReading):
            if not reading.membership_complete or bool(reading.partial_reasons):
                with self._lock:
                    self._source_partial_flags.add(source_id)
            target_gen = reading.target_generation
            if (
                isinstance(target_gen, LinuxProcessGeneration)
                and self._linux_tree_target_generation is None
            ):
                self._linux_tree_target_generation = target_gen
            try:
                subject = self._build_bounded_descendant_subject(
                    reading,
                    source_id=source_id,
                    read_start_ns=read_start_ns,
                    read_end_ns=read_end_ns,
                    readings=readings,
                )
            except Exception as exc:
                with self._lock:
                    self._attempted += 1
                    self._failed += 1
                    self._diagnostics.append(
                        f"{source_id} subject encoding error: {type(exc).__name__}: {exc}"
                    )
                return
        else:
            subject = self._linux_tree_subject

        self._enqueue_reading_packet(
            source_id=source_id,
            subject=subject,
            read_start_ns=read_start_ns,
            read_end_ns=read_end_ns,
            readings=readings,
        )

    def _linux_cgroup_loop(self) -> None:
        """Monotonic-deadline sampling loop for read-only Linux cgroup v2."""

        next_due_ns = int(self._monotonic_ns()) + self._linux_cgroup_interval_ns
        while not self._stop_event.is_set():
            now_ns = int(self._monotonic_ns())
            wait_ns = next_due_ns - now_ns
            if wait_ns > 0:
                if self._stop_event.wait(timeout=wait_ns / _NS_PER_SECOND):
                    break
                now_ns = int(self._monotonic_ns())
                if now_ns < next_due_ns:
                    continue
            if self._stop_event.is_set():
                break
            lag_ns = now_ns - next_due_ns
            if lag_ns >= self._linux_cgroup_interval_ns:
                missed = lag_ns // self._linux_cgroup_interval_ns
                with self._lock:
                    self._missed_deadlines += int(missed)
                next_due_ns = now_ns + self._linux_cgroup_interval_ns
            else:
                next_due_ns += self._linux_cgroup_interval_ns
            self._sample_linux_cgroup()

    def _sample_linux_cgroup(self) -> None:
        """Capture one read-only Linux cgroup v2 sample and enqueue nonblockingly."""

        lc_cfg = self._callback.linux_cgroup
        source_id = lc_cfg.source_id
        read_start_ns = int(self._monotonic_ns())
        init_error = self._linux_cgroup_init_error
        if init_error is not None or self._linux_cgroup_probe is None:
            issue = (
                init_error
                if init_error is not None
                else LinuxMemoryUnavailable(
                    reason=f"{source_id} probe unavailable",
                    reason_code="probe_init_failed",
                    status="error",
                )
            )
            read_end_ns = max(int(self._monotonic_ns()), read_start_ns)
            readings = {
                qty_id: ObservabilityReading(
                    status=issue.status,
                    reason_code=issue.reason_code,
                    detail=issue.reason,
                )
                for qty_id in lc_cfg.quantities
            }
            self._enqueue_reading_packet(
                source_id=source_id,
                subject=self._linux_cgroup_subject,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                readings=readings,
            )
            return

        try:
            reading = self._linux_cgroup_probe.read()
        except Exception as exc:
            with self._lock:
                self._attempted += 1
                self._failed += 1
                self._diagnostics.append(
                    f"{source_id} read failed: {type(exc).__name__}: {exc}"
                )
            return
        read_end_ns = max(int(self._monotonic_ns()), read_start_ns)

        target_gen = getattr(reading, "target_generation", None)
        cg_ident = getattr(reading, "cgroup_identity", None)
        if isinstance(cg_ident, LinuxCgroupV2Identity):
            subject = cg_ident.to_subject_dict(target_generation=target_gen)
            self._linux_cgroup_subject = subject
        else:
            subject = dict(self._linux_cgroup_subject)
            if isinstance(target_gen, LinuxProcessGeneration):
                subject["target_generation"] = target_gen.to_dict()
        if (
            isinstance(target_gen, LinuxProcessGeneration)
            and self._linux_cgroup_target_generation is None
        ):
            self._linux_cgroup_target_generation = target_gen

        try:
            readings = _linux_cgroup_reading_to_readings(reading, lc_cfg.quantities)
        except Exception as exc:
            with self._lock:
                self._attempted += 1
                self._failed += 1
                self._diagnostics.append(
                    f"{source_id} reading conversion error: {type(exc).__name__}: {exc}"
                )
            return

        self._enqueue_reading_packet(
            source_id=source_id,
            subject=subject,
            read_start_ns=read_start_ns,
            read_end_ns=read_end_ns,
            readings=readings,
        )

    def _sample_allocator(self) -> int:
        """Capture one cooperative allocator sample on the owning thread."""

        alloc_cfg = self._callback.allocator
        source_id = alloc_cfg.source_id
        read_start_ns = int(self._monotonic_ns())
        readings: dict[str, ObservabilityReading] = {}
        observed_identity: Any = None

        if alloc_cfg.include_current:
            if self._allocator_current_probe is None:
                reason_code, detail = (
                    self._allocator_current_init_unavailable_reason
                    if self._allocator_current_init_unavailable_reason is not None
                    else ("allocator_unavailable", "current allocator probe unavailable")
                )
                for qty_id in ALLOCATOR_CURRENT_QUANTITY_IDS:
                    readings[qty_id] = ObservabilityReading(
                        status="unavailable",
                        reason_code=reason_code,
                        detail=detail,
                    )
            else:
                try:
                    current_usage = self._allocator_current_probe.read()
                except Exception as exc:
                    err_detail = f"{type(exc).__name__}: {exc}"
                    for qty_id in ALLOCATOR_CURRENT_QUANTITY_IDS:
                        readings[qty_id] = ObservabilityReading(
                            status="error",
                            reason_code="read_failed",
                            detail=err_detail,
                        )
                else:
                    observed_identity = getattr(current_usage, "identity", None)
                    readings["allocator.current.allocated_bytes"] = _allocator_bytes_reading(
                        getattr(current_usage, "allocated_bytes", None),
                        field_name="allocated_bytes",
                        from_mib=False,
                    )
                    readings["allocator.current.reserved_bytes"] = _allocator_bytes_reading(
                        getattr(current_usage, "reserved_bytes", None),
                        field_name="reserved_bytes",
                        from_mib=False,
                    )

        if alloc_cfg.include_peak:
            if self._allocator_peak_reset_failure is not None:
                for qty_id in ALLOCATOR_PEAK_QUANTITY_IDS:
                    readings[qty_id] = ObservabilityReading(
                        status="unavailable",
                        reason_code="reset_failed",
                        detail=self._allocator_peak_reset_failure,
                    )
            elif self._allocator_peak_probe is None:
                reason_code, detail = (
                    self._allocator_peak_init_unavailable_reason
                    if self._allocator_peak_init_unavailable_reason is not None
                    else ("allocator_unavailable", "peak allocator probe unavailable")
                )
                for qty_id in ALLOCATOR_PEAK_QUANTITY_IDS:
                    readings[qty_id] = ObservabilityReading(
                        status="unavailable",
                        reason_code=reason_code,
                        detail=detail,
                    )
            else:
                try:
                    peak_usage = self._allocator_peak_probe.read()
                except Exception as exc:
                    err_detail = f"{type(exc).__name__}: {exc}"
                    for qty_id in ALLOCATOR_PEAK_QUANTITY_IDS:
                        readings[qty_id] = ObservabilityReading(
                            status="error",
                            reason_code="read_failed",
                            detail=err_detail,
                        )
                else:
                    if observed_identity is None:
                        observed_identity = getattr(peak_usage, "identity", None)
                    readings["allocator.peak.allocated_bytes"] = _allocator_bytes_reading(
                        getattr(peak_usage, "allocated_mb", None),
                        field_name="allocated_mb",
                        from_mib=True,
                    )
                    readings["allocator.peak.reserved_bytes"] = _allocator_bytes_reading(
                        getattr(peak_usage, "reserved_mb", None),
                        field_name="reserved_mb",
                        from_mib=True,
                    )

        read_end_ns = max(int(self._monotonic_ns()), read_start_ns)
        if observed_identity is not None and self._allocator_subject.get("backend") == "unknown":
            self._allocator_subject = self._build_allocator_subject(
                configured_device=self._allocator_configured_device,
                identity=observed_identity,
            )

        self._enqueue_reading_packet(
            source_id=source_id,
            subject=self._allocator_subject,
            read_start_ns=read_start_ns,
            read_end_ns=read_end_ns,
            readings=readings,
        )
        return read_end_ns

    def _nvidia_device_loop(self) -> None:
        """Monotonic-deadline sampling loop for bounded ``nvidia-smi`` reads."""

        next_due_ns = int(self._monotonic_ns()) + self._nvidia_device_interval_ns
        while not self._stop_event.is_set():
            now_ns = int(self._monotonic_ns())
            wait_ns = next_due_ns - now_ns
            if wait_ns > 0:
                if self._stop_event.wait(timeout=wait_ns / _NS_PER_SECOND):
                    break
                now_ns = int(self._monotonic_ns())
                if now_ns < next_due_ns:
                    continue
            if self._stop_event.is_set():
                break
            lag_ns = now_ns - next_due_ns
            if lag_ns >= self._nvidia_device_interval_ns:
                missed = lag_ns // self._nvidia_device_interval_ns
                with self._lock:
                    self._missed_deadlines += int(missed)
                next_due_ns = now_ns + self._nvidia_device_interval_ns
            else:
                next_due_ns += self._nvidia_device_interval_ns
            self._sample_nvidia_device(cancel_event=self._stop_event)

    def _sample_nvidia_device(
        self,
        *,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> None:
        """Capture one bounded physical-GPU ``nvidia-smi`` sample and enqueue."""

        nd_cfg = self._callback.nvidia_device
        source_id = nd_cfg.source_id
        read_start_ns = int(self._monotonic_ns())
        init_error = self._nvidia_device_init_error
        if init_error is not None or self._nvidia_device_probe is None:
            issue = (
                init_error
                if init_error is not None
                else NvidiaDeviceUnavailable(
                    reason=f"{source_id} probe unavailable",
                    reason_code="probe_init_failed",
                    status="error",
                )
            )
            read_end_ns = max(int(self._monotonic_ns()), read_start_ns)
            readings = {
                qty_id: ObservabilityReading(
                    status=issue.status,
                    reason_code=issue.reason_code,
                    detail=issue.reason,
                )
                for qty_id in nd_cfg.quantities
            }
            subject = dict(self._nvidia_device_subject_base)
            subject["evidence"] = {
                "vendor_timestamp": None,
                "mig_mode": "unknown",
                "mig_ambiguous": False,
            }
            self._enqueue_reading_packet(
                source_id=source_id,
                subject=subject,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                readings=readings,
            )
            return

        call_kwargs: dict[str, Any] = {}
        if self._nvidia_device_read_supports_cancel:
            call_kwargs["cancel_event"] = cancel_event
        if self._nvidia_device_read_supports_deadline:
            call_kwargs["deadline_ns"] = deadline_ns
        try:
            reading = self._nvidia_device_probe.read(**call_kwargs)
        except Exception as exc:
            with self._lock:
                self._attempted += 1
                self._failed += 1
                self._source_partial_flags.add(source_id)
                self._diagnostics.append(
                    f"{source_id} read failed: {type(exc).__name__}: {exc}"
                )
            return
        read_end_ns = max(int(self._monotonic_ns()), read_start_ns)

        # Normal scope teardown cancels any in-flight periodic poll via
        # ``self._stop_event`` and immediately follows with a bounded final
        # capture (``cancel_event=None, deadline_ns=deadline_ns``). Discard a
        # teardown-cancelled periodic poll cleanly so healthy runs stay
        # available/complete without spurious unavailable packets.
        if (
            cancel_event is self._stop_event
            and cancel_event.is_set()
            and isinstance(reading, NvidiaDeviceReading)
            and reading.status == "unavailable"
            and reading.reason_code == "read_cancelled"
        ):
            return

        try:
            readings = _nvidia_device_reading_to_readings(reading, nd_cfg.quantities)
        except Exception as exc:
            with self._lock:
                self._attempted += 1
                self._failed += 1
                self._source_partial_flags.add(source_id)
                self._diagnostics.append(
                    f"{source_id} reading conversion error: {type(exc).__name__}: {exc}"
                )
            return

        # Only mark the source partial when a selected quantity is non-ok
        # (never merely because an unselected sibling metric failed).
        if any(item.status != "ok" for item in readings.values()):
            with self._lock:
                self._source_partial_flags.add(source_id)

        subject = dict(self._nvidia_device_subject_base)
        if isinstance(reading, NvidiaDeviceReading):
            if subject.get("device_uuid") is None and reading.observed_uuid is not None:
                subject["device_uuid"] = reading.observed_uuid
            subject["evidence"] = {
                "vendor_timestamp": reading.vendor_timestamp,
                "mig_mode": reading.mig_mode,
                "mig_ambiguous": reading.mig_ambiguous,
            }
        else:
            subject["evidence"] = {
                "vendor_timestamp": getattr(reading, "vendor_timestamp", None),
                "mig_mode": getattr(reading, "mig_mode", "unknown"),
                "mig_ambiguous": bool(getattr(reading, "mig_ambiguous", False)),
            }

        self._enqueue_reading_packet(
            source_id=source_id,
            subject=subject,
            read_start_ns=read_start_ns,
            read_end_ns=read_end_ns,
            readings=readings,
        )

    def _enqueue_reading_packet(
        self,
        *,
        source_id: str,
        subject: Mapping[str, Any],
        read_start_ns: int,
        read_end_ns: int,
        readings: Mapping[str, ObservabilityReading],
    ) -> None:
        """Assign monotonic sequence and enqueue without ever blocking callers."""

        writer = self._writer
        with self._lock:
            self._attempted += 1
            seq = self._next_seq.get(source_id, 0)
            self._next_seq[source_id] = seq + 1

            if not self._accepting_writes or self._raw_retention_stopped or writer is None:
                self._dropped += 1
                _append_unique(
                    self._truncation_reasons,
                    "writer_error" if self._writer_had_error else "retention_stopped",
                )
                return

            try:
                sample = ObservabilitySample(
                    invocation_id=self._invocation_id,
                    observer_id=self._observer_id,
                    source_id=source_id,
                    descriptor_set_id=self._descriptor_set_id,
                    sequence=seq,
                    clock_domain_id=self._clock_domain_id,
                    subject=subject,
                    read_start_ns=read_start_ns,
                    read_end_ns=read_end_ns,
                    readings=readings,
                )
                encoded_bytes = _encode_sample_line_bytes(sample)
            except Exception as exc:
                self._failed += 1
                self._diagnostics.append(
                    f"{source_id} packet error: {type(exc).__name__}: {exc}"
                )
                return

            if len(self._pending) >= self._callback.max_pending_envelopes:
                self._dropped += 1
                _append_unique(self._truncation_reasons, "queue_full")
                return

            if not writer.try_reserve_pending(self._observer_id, encoded_bytes):
                self._dropped += 1
                _append_unique(self._truncation_reasons, "pending_bytes_cap_reached")
                return

            self._pending.append((sample, encoded_bytes))
            self._drain_wake_event.set()

    def _record_writer_error(self, message: str) -> None:
        with self._lock:
            self._writer_had_error = True
            self._raw_retention_stopped = True
            _append_unique(self._truncation_reasons, "writer_error")
            self._diagnostics.append(message)

    def _write_one_sample(self, sample: ObservabilitySample, reserved_bytes: int) -> None:
        """Persist one dequeued sample through the single bounded bundle writer."""

        writer = self._writer
        try:
            if writer is None or self._raw_retention_stopped or not self._accepting_writes:
                with self._lock:
                    self._dropped += 1
                    _append_unique(
                        self._truncation_reasons,
                        "writer_error" if self._writer_had_error else "retention_stopped",
                    )
                return

            with self._writer_lock:
                if not self._accepting_writes or writer.finalized:
                    with self._lock:
                        self._dropped += 1
                        _append_unique(self._truncation_reasons, "close_timeout")
                    return
                result = writer.append_sample(sample)
        except Exception as exc:
            with self._lock:
                self._failed += 1
                self._writer_had_error = True
                self._raw_retention_stopped = True
                _append_unique(self._truncation_reasons, "writer_error")
                self._diagnostics.append(
                    f"sample write failed: {type(exc).__name__}: {exc}"
                )
        else:
            with self._lock:
                if result.written:
                    self._written += 1
                    win = self._source_windows_ns.get(sample.source_id)
                    if win is None:
                        self._source_windows_ns[sample.source_id] = [
                            sample.read_end_ns,
                            sample.read_end_ns,
                        ]
                    else:
                        win[0] = min(win[0], sample.read_end_ns)
                        win[1] = max(win[1], sample.read_end_ns)
                    for qty_id, reading in sample.readings.items():
                        if reading.status == "ok":
                            self._source_ok_counts[sample.source_id] = (
                                self._source_ok_counts.get(sample.source_id, 0) + 1
                            )
                        else:
                            self._source_non_ok_counts[sample.source_id] = (
                                self._source_non_ok_counts.get(sample.source_id, 0) + 1
                            )
                            self._source_unavailable_fields.add(
                                f"{sample.source_id}/{qty_id}"
                            )
                else:
                    self._dropped += 1
                    if result.reason:
                        _append_unique(self._truncation_reasons, result.reason)
        finally:
            if writer is not None:
                with self._lock:
                    writer.release_pending(self._observer_id, reserved_bytes)

    def _drain_loop(self) -> None:
        """Background drain thread writing queued samples to routed storage."""

        while True:
            item: tuple[ObservabilitySample, int] | None = None
            with self._lock:
                # Honor stop signal before popping so close() hands any remaining
                # backlog to _drain_pending_until(deadline_ns) after at most one
                # in-flight write.
                if self._drain_stop_event.is_set():
                    return
                if self._pending:
                    item = self._pending.popleft()
                else:
                    self._drain_wake_event.clear()
            if item is not None:
                self._write_one_sample(item[0], item[1])
                continue
            self._drain_wake_event.wait(timeout=0.05)

    def checkpoint(self) -> None:
        """Run due cooperative allocator sampling on the owning thread."""

        if self._stop_event.is_set() or self._closed:
            return
        if not self._callback.allocator.enabled:
            return
        if threading.get_ident() != self._allocator_owner_thread_id:
            return
        now_ns = int(self._monotonic_ns())
        if now_ns < self._allocator_next_due_ns:
            return
        lag_ns = now_ns - self._allocator_next_due_ns
        if lag_ns >= self._allocator_interval_ns:
            missed = lag_ns // self._allocator_interval_ns
            with self._lock:
                self._missed_deadlines += int(missed)
            self._allocator_next_due_ns = now_ns + self._allocator_interval_ns
        else:
            self._allocator_next_due_ns += self._allocator_interval_ns
        read_end_ns = self._sample_allocator()
        if read_end_ns >= self._allocator_next_due_ns:
            # Avoid a back-to-back catch-up storm when a single probe read
            # spans past the next scheduled deadline.
            self._allocator_next_due_ns = read_end_ns + self._allocator_interval_ns

    def request_stop(self) -> None:
        """Signal nonblocking stop immediately without waiting or joining."""

        self._stop_event.set()
        self._drain_wake_event.set()

    def _drain_pending_until(self, deadline_ns: int) -> None:
        """Drain queued samples on the closing thread up to ``deadline_ns``."""

        while True:
            item: tuple[ObservabilitySample, int] | None = None
            with self._lock:
                if not self._pending:
                    return
                if time.monotonic_ns() >= deadline_ns:
                    # Drop unpersisted pending packets cleanly on close timeout.
                    while self._pending:
                        _, reserved = self._pending.popleft()
                        self._dropped += 1
                        if self._writer is not None:
                            self._writer.release_pending(self._observer_id, reserved)
                    _append_unique(self._truncation_reasons, "close_timeout")
                    return
                item = self._pending.popleft()
            if item is not None:
                self._write_one_sample(item[0], item[1])

    def close(self, primary_exc: BaseException | None) -> bool:
        """Stop threads, drain within one total close deadline, and finalize bundle.

        Returns
        -------
        bool
            ``True`` when all owned worker threads are confirmed stopped, or
            ``False`` when shutdown timed out and the session must be treated
            as unsafe.
        """

        if self._closed:
            return not self._unsafe
        self._closed = True

        # Enforce one shared real monotonic close deadline for thread joins and
        # queue draining even when sample timestamps use an injected test clock.
        total_timeout_s = self._callback.close_timeout_seconds
        start_close_ns = time.monotonic_ns()
        deadline_ns = start_close_ns + int(round(total_timeout_s * _NS_PER_SECOND))
        producer_deadline_ns = start_close_ns + int(
            round(total_timeout_s * 0.75 * _NS_PER_SECOND)
        )
        self._stop_event.set()
        self._drain_wake_event.set()

        # 1. Join all periodic producer threads within the shared close deadline,
        # reserving a small fraction of the same total budget so drain/status
        # finalization can still complete if any producer hangs.
        rusage_stopped = True
        if self._rusage_thread is not None and self._rusage_thread_started:
            remaining = max(
                0.0, (producer_deadline_ns - time.monotonic_ns()) / _NS_PER_SECOND
            )
            self._rusage_thread.join(timeout=remaining)
            rusage_stopped = not self._rusage_thread.is_alive()

        linux_memory_stopped = True
        if self._linux_memory_thread is not None and self._linux_memory_thread_started:
            remaining = max(
                0.0, (producer_deadline_ns - time.monotonic_ns()) / _NS_PER_SECOND
            )
            self._linux_memory_thread.join(timeout=remaining)
            linux_memory_stopped = not self._linux_memory_thread.is_alive()

        linux_tree_stopped = True
        if self._linux_tree_thread is not None and self._linux_tree_thread_started:
            remaining = max(
                0.0, (producer_deadline_ns - time.monotonic_ns()) / _NS_PER_SECOND
            )
            self._linux_tree_thread.join(timeout=remaining)
            linux_tree_stopped = not self._linux_tree_thread.is_alive()

        linux_cgroup_stopped = True
        if self._linux_cgroup_thread is not None and self._linux_cgroup_thread_started:
            remaining = max(
                0.0, (producer_deadline_ns - time.monotonic_ns()) / _NS_PER_SECOND
            )
            self._linux_cgroup_thread.join(timeout=remaining)
            linux_cgroup_stopped = not self._linux_cgroup_thread.is_alive()

        nvidia_device_stopped = True
        if self._nvidia_device_thread is not None and self._nvidia_device_thread_started:
            remaining = max(
                0.0, (producer_deadline_ns - time.monotonic_ns()) / _NS_PER_SECOND
            )
            self._nvidia_device_thread.join(timeout=remaining)
            nvidia_device_stopped = not self._nvidia_device_thread.is_alive()

        producer_stopped = (
            rusage_stopped
            and linux_memory_stopped
            and linux_tree_stopped
            and linux_cgroup_stopped
            and nvidia_device_stopped
        )

        # 2. Stop and join the drain thread within the SAME remaining deadline
        # before taking final captures so earlier queued samples never compete
        # with final captures under small max_pending_envelopes bounds.
        self._drain_stop_event.set()
        self._drain_wake_event.set()
        drain_stopped = True
        if self._drain_thread is not None and self._drain_thread_started:
            remaining = max(0.0, (deadline_ns - time.monotonic_ns()) / _NS_PER_SECOND)
            self._drain_thread.join(timeout=remaining)
            drain_stopped = not self._drain_thread.is_alive()

        # 3. If drain thread stopped, flush any remaining queued samples, then
        # perform final reachable captures on the owning thread and drain each
        # immediately within the same single close deadline.
        if drain_stopped:
            self._drain_pending_until(deadline_ns)
            if rusage_stopped and time.monotonic_ns() < deadline_ns:
                if (
                    self._callback.rusage.enabled
                    and self._rusage_probe is not None
                    and self._writer is not None
                ):
                    self._sample_rusage()
                    self._drain_pending_until(deadline_ns)
            if linux_memory_stopped and time.monotonic_ns() < deadline_ns:
                if (
                    self._callback.linux_memory.enabled
                    and (
                        self._linux_memory_probe is not None
                        or self._linux_memory_init_error is not None
                    )
                    and self._writer is not None
                ):
                    self._sample_linux_memory()
                    self._drain_pending_until(deadline_ns)
            if linux_tree_stopped and time.monotonic_ns() < deadline_ns:
                if (
                    self._callback.linux_tree.enabled
                    and (
                        self._linux_tree_probe is not None
                        or self._linux_tree_init_error is not None
                    )
                    and self._writer is not None
                ):
                    self._sample_linux_tree()
                    self._drain_pending_until(deadline_ns)
            if linux_cgroup_stopped and time.monotonic_ns() < deadline_ns:
                if (
                    self._callback.linux_cgroup.enabled
                    and (
                        self._linux_cgroup_probe is not None
                        or self._linux_cgroup_init_error is not None
                    )
                    and self._writer is not None
                ):
                    self._sample_linux_cgroup()
                    self._drain_pending_until(deadline_ns)
            if time.monotonic_ns() < deadline_ns:
                if (
                    self._callback.allocator.enabled
                    and self._writer is not None
                    and threading.get_ident() == self._allocator_owner_thread_id
                ):
                    self._sample_allocator()
                    self._drain_pending_until(deadline_ns)
            if nvidia_device_stopped and time.monotonic_ns() < deadline_ns:
                if (
                    self._callback.nvidia_device.enabled
                    and (
                        self._nvidia_device_probe is not None
                        or self._nvidia_device_init_error is not None
                    )
                    and self._writer is not None
                ):
                    self._sample_nvidia_device(
                        cancel_event=None, deadline_ns=deadline_ns
                    )
                    self._drain_pending_until(deadline_ns)

        quiescent = producer_stopped and drain_stopped
        if not quiescent:
            self._unsafe = True
            with self._lock:
                self._accepting_writes = False
                _append_unique(self._truncation_reasons, "close_timeout")
                if not rusage_stopped:
                    self._diagnostics.append(
                        "rusage sampling thread failed to stop within close_timeout_seconds"
                    )
                if not linux_memory_stopped:
                    self._diagnostics.append(
                        "linux-memory sampling thread failed to stop within close_timeout_seconds"
                    )
                if not linux_tree_stopped:
                    self._diagnostics.append(
                        "linux-process-tree sampling thread failed to stop within close_timeout_seconds"
                    )
                if not linux_cgroup_stopped:
                    self._diagnostics.append(
                        "linux-cgroup-v2 sampling thread failed to stop within close_timeout_seconds"
                    )
                if not nvidia_device_stopped:
                    self._diagnostics.append(
                        "nvidia-smi-device sampling thread failed to stop within close_timeout_seconds"
                    )
                if not drain_stopped:
                    self._diagnostics.append(
                        "observability drain thread failed to stop within close_timeout_seconds"
                    )
        else:
            with self._lock:
                self._accepting_writes = False

        # 5. Write final status and finalize manifest atomically if writer lock is available.
        self._finalize_bundle(
            quiescent=quiescent,
            producer_stopped=producer_stopped,
            drain_stopped=drain_stopped,
            primary_exc=primary_exc,
        )
        return quiescent

    def _finalize_bundle(
        self,
        *,
        quiescent: bool,
        producer_stopped: bool,
        drain_stopped: bool,
        primary_exc: BaseException | None,
    ) -> None:
        writer = self._writer
        if writer is None or self._start_refused:
            return
        # Never deadlock waiting on a hung drain thread holding _writer_lock.
        acquired = self._writer_lock.acquire(blocking=drain_stopped)
        if not acquired:
            return
        try:
            if writer.finalized:
                return
            lifecycle_state = "closed" if quiescent else "unsafe"
            acquisition_end_ns = max(int(self._monotonic_ns()), self._activation_ns)

            with self._lock:
                attempted = self._attempted
                written = self._written
                dropped = self._dropped
                failed = self._failed
                pending = len(self._pending)
                missed_deadlines = self._missed_deadlines
                source_windows = {
                    src: [pair[0], pair[1]]
                    for src, pair in sorted(self._source_windows_ns.items())
                }
                reasons = list(self._truncation_reasons)
                for writer_reason in writer.observer_truncation_reasons(self._observer_id):
                    _append_unique(reasons, writer_reason)
                unavailable_fields = tuple(sorted(self._source_unavailable_fields))
                diagnostics = tuple(self._diagnostics)
                source_ok_counts = dict(self._source_ok_counts)
                source_non_ok_counts = dict(self._source_non_ok_counts)
                source_partial_flags = set(self._source_partial_flags)
                writer_had_error = self._writer_had_error

            source_status: dict[str, str] = {}
            for enabled_flag, sid in (
                (self._callback.rusage.enabled, self._callback.rusage.source_id),
                (self._callback.linux_memory.enabled, self._callback.linux_memory.source_id),
                (self._callback.linux_tree.enabled, self._callback.linux_tree.source_id),
                (self._callback.linux_cgroup.enabled, self._callback.linux_cgroup.source_id),
                (self._callback.allocator.enabled, self._callback.allocator.source_id),
                (
                    self._callback.nvidia_device.enabled,
                    self._callback.nvidia_device.source_id,
                ),
            ):
                if not enabled_flag:
                    continue
                ok_c = source_ok_counts.get(sid, 0)
                non_ok_c = source_non_ok_counts.get(sid, 0)
                is_partial = sid in source_partial_flags
                if ok_c > 0 and non_ok_c == 0 and not is_partial:
                    source_status[sid] = "available"
                elif ok_c > 0 and (non_ok_c > 0 or is_partial):
                    source_status[sid] = "partial"
                else:
                    source_status[sid] = "unavailable"

            any_ok_readings = sum(source_ok_counts.values()) > 0
            any_non_ok_readings = sum(source_non_ok_counts.values()) > 0
            if not quiescent:
                data_completeness = "partial" if written > 0 else "unknown"
            elif written == 0 and attempted == 0:
                data_completeness = "none"
            elif written == 0:
                data_completeness = "none"
            elif (
                dropped > 0
                or failed > 0
                or bool(reasons)
                or any_non_ok_readings
                or any(st != "available" for st in source_status.values())
            ):
                data_completeness = "partial" if any_ok_readings or written > 0 else "none"
            else:
                data_completeness = "complete"

            counts: dict[str, int] = {
                "attempted": attempted,
                "written": written,
                "dropped": dropped,
                "failed": failed,
            }
            if lifecycle_state != "closed":
                # Keep attempted == written + dropped + failed + pending exact.
                counts["pending"] = max(0, attempted - (written + dropped + failed))
            else:
                # Reconcile any residual discrepancy defensively as failed so closed
                # counts always satisfy attempted == written + dropped + failed.
                residual = attempted - (written + dropped + failed)
                if residual > 0:
                    counts["failed"] += residual

            app_outcome = self._callback._observed_application_outcome
            if primary_exc is not None:
                invocation_outcome = (
                    "interrupted"
                    if isinstance(primary_exc, KeyboardInterrupt)
                    else "failed"
                )
            elif app_outcome != "not_supplied":
                invocation_outcome = app_outcome
            else:
                invocation_outcome = "completed"

            any_producer_started = (
                self._rusage_thread_started
                or self._linux_memory_thread_started
                or self._linux_tree_thread_started
                or self._linux_cgroup_thread_started
                or self._nvidia_device_thread_started
            )
            if not any_producer_started:
                helper_state = "not_started"
            elif producer_stopped:
                helper_state = "confirmed_stopped"
            else:
                helper_state = "unconfirmed_timeout"

            writer_state = (
                "unsafe"
                if not quiescent
                else ("error" if writer_had_error else "closed")
            )

            status = ObservabilityStatus(
                invocation_id=self._invocation_id,
                observer_id=self._observer_id,
                lifecycle_state=lifecycle_state,
                data_completeness=data_completeness,
                source_status=source_status,
                coverage={
                    "activation_ns": self._activation_ns,
                    "acquisition_end_ns": acquisition_end_ns,
                    "clock_domain_id": self._clock_domain_id,
                    "bootstrap": "unmeasured",
                    "requested_interval_seconds": self._callback.interval_seconds,
                    "missed_deadlines": missed_deadlines,
                    "source_windows_ns": source_windows,
                },
                counts=counts,
                truncation_reasons=tuple(reasons),
                writer={
                    "state": writer_state,
                    "flush_confirmed": not writer_had_error and quiescent,
                    "fsync_confirmed": False,
                },
                helper_cleanup={"state": helper_state},
                application_outcome=app_outcome,
                invocation_outcome=invocation_outcome,
                export_status="not_requested",
                unavailable_fields=unavailable_fields or None,
                diagnostics=diagnostics or None,
            )
            writer.write_status(status)
            final_manifest = ObservabilityManifest(
                lifecycle_state=lifecycle_state,
                data_completeness=data_completeness,
                **self._manifest_base,
            )
            writer.finalize_manifest(final_manifest)
        except Exception as exc:
            # Advisory finalization errors must not mask the run's primary result.
            self._record_writer_error(
                f"bundle finalization failed: {type(exc).__name__}: {exc}"
            )
        finally:
            self._writer_lock.release()


class _DisabledScope(AbstractContextManager[RunResource | None]):
    """No-op managed scope returned when ``ResourceSamples(enabled=False)``."""

    def __enter__(self) -> RunResource | None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        del exc_type, exc_val, exc_tb
        return False


class _ResourceSamplesScope(AbstractContextManager[RunResource | None]):
    """Context manager acquiring and releasing one :class:`_ResourceSamplesSession`."""

    def __init__(self, callback: ResourceSamples, context: RunContext) -> None:
        self._callback = callback
        self._context = context
        self._session: _ResourceSamplesSession | None = None
        self._exited = False

    def __enter__(self) -> RunResource | None:
        session = self._callback._begin_session(self._context)
        try:
            handle = session.start()
        except BaseException as exc:
            quiescent = False
            try:
                quiescent = session.close(exc)
            except Exception:
                quiescent = False
            finally:
                self._callback._end_session(session, quiescent=quiescent)
            raise
        self._session = session
        return handle

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        del exc_type, exc_tb
        if self._exited or self._session is None:
            return False
        self._exited = True
        session = self._session
        quiescent = False
        try:
            quiescent = session.close(exc_val)
        finally:
            self._callback._end_session(session, quiescent=quiescent)
        # Never suppress a primary body exception or BaseException.
        return False


# ---------------------------------------------------------------------------
# Public managed callback
# ---------------------------------------------------------------------------


def _coerce_rusage_source(raw: RUsageSource | Mapping[str, Any] | None) -> RUsageSource | None:
    if raw is None:
        return None
    if type(raw) is RUsageSource:
        return raw
    if isinstance(raw, RUsageSource):
        raise TypeError(
            f"rusage must be exact RUsageSource, mapping, or None; got subclass {type(raw).__name__}"
        )
    if isinstance(raw, Mapping):
        _reject_forbidden_selector_keys(raw, owner="RUsageSource")
        return RUsageSource(**dict(raw))
    raise TypeError(f"rusage must be RUsageSource, mapping, or None; got {type(raw).__name__}")


def _coerce_allocator_source(
    raw: AllocatorSource | Mapping[str, Any] | None,
) -> AllocatorSource | None:
    if raw is None:
        return None
    if type(raw) is AllocatorSource:
        return raw
    if isinstance(raw, AllocatorSource):
        raise TypeError(
            f"allocator must be exact AllocatorSource, mapping, or None; got subclass {type(raw).__name__}"
        )
    if isinstance(raw, Mapping):
        _reject_forbidden_selector_keys(raw, owner="AllocatorSource")
        return AllocatorSource(**dict(raw))
    raise TypeError(
        f"allocator must be AllocatorSource, mapping, or None; got {type(raw).__name__}"
    )


def _coerce_linux_memory_source(
    raw: LinuxMemorySource | Mapping[str, Any] | None,
) -> LinuxMemorySource | None:
    if raw is None:
        return None
    if type(raw) is LinuxMemorySource:
        return raw
    if isinstance(raw, LinuxMemorySource):
        raise TypeError(
            f"linux_memory must be exact LinuxMemorySource, mapping, or None; "
            f"got subclass {type(raw).__name__}"
        )
    if isinstance(raw, Mapping):
        _reject_forbidden_selector_keys(raw, owner="LinuxMemorySource")
        return LinuxMemorySource(**dict(raw))
    raise TypeError(
        f"linux_memory must be LinuxMemorySource, mapping, or None; "
        f"got {type(raw).__name__}"
    )


def _coerce_linux_tree_source(
    raw: LinuxProcessTreeSource | Mapping[str, Any] | None,
) -> LinuxProcessTreeSource | None:
    if raw is None:
        return None
    if type(raw) is LinuxProcessTreeSource:
        return raw
    if isinstance(raw, LinuxProcessTreeSource):
        raise TypeError(
            f"linux_tree must be exact LinuxProcessTreeSource, mapping, or None; "
            f"got subclass {type(raw).__name__}"
        )
    if isinstance(raw, Mapping):
        _reject_forbidden_selector_keys(raw, owner="LinuxProcessTreeSource")
        return LinuxProcessTreeSource(**dict(raw))
    raise TypeError(
        f"linux_tree must be LinuxProcessTreeSource, mapping, or None; "
        f"got {type(raw).__name__}"
    )


def _coerce_linux_cgroup_source(
    raw: LinuxCgroupV2Source | Mapping[str, Any] | None,
) -> LinuxCgroupV2Source | None:
    if raw is None:
        return None
    if type(raw) is LinuxCgroupV2Source:
        return raw
    if isinstance(raw, LinuxCgroupV2Source):
        raise TypeError(
            f"linux_cgroup must be exact LinuxCgroupV2Source, mapping, or None; "
            f"got subclass {type(raw).__name__}"
        )
    if isinstance(raw, Mapping):
        _reject_forbidden_selector_keys(raw, owner="LinuxCgroupV2Source")
        return LinuxCgroupV2Source(**dict(raw))
    raise TypeError(
        f"linux_cgroup must be LinuxCgroupV2Source, mapping, or None; "
        f"got {type(raw).__name__}"
    )


def _coerce_nvidia_device_source(
    raw: NvidiaDeviceSource | Mapping[str, Any] | None,
) -> NvidiaDeviceSource | None:
    if raw is None:
        return None
    if type(raw) is NvidiaDeviceSource:
        return raw
    if isinstance(raw, NvidiaDeviceSource):
        raise TypeError(
            f"nvidia_device must be exact NvidiaDeviceSource, mapping, or None; "
            f"got subclass {type(raw).__name__}"
        )
    if isinstance(raw, Mapping):
        _reject_forbidden_selector_keys(raw, owner="NvidiaDeviceSource")
        return NvidiaDeviceSource(**dict(raw))
    raise TypeError(
        f"nvidia_device must be NvidiaDeviceSource, mapping, or None; "
        f"got {type(raw).__name__}"
    )


def _coerce_sink(raw: RunArtifactSink | Mapping[str, Any] | None) -> RunArtifactSink:
    if raw is None:
        return RunArtifactSink()
    if type(raw) is RunArtifactSink:
        return raw
    if isinstance(raw, RunArtifactSink):
        raise TypeError(
            f"sink must be exact RunArtifactSink, mapping, or None; got subclass {type(raw).__name__}"
        )
    if isinstance(raw, Mapping):
        _reject_forbidden_selector_keys(raw, owner="RunArtifactSink")
        return RunArtifactSink(**dict(raw))
    raise TypeError(
        f"sink must be RunArtifactSink, mapping, or None; got {type(raw).__name__}"
    )


class ResourceSamples(Callback):
    """Opt-in managed callback for single-process RUSAGE and allocator samples.

    Construction is strictly Torch-free and performs zero filesystem I/O,
    thread creation, or accelerator queries. All resources are acquired only
    when core enters :meth:`run_scope` and released within a single total
    ``close_timeout_seconds`` deadline on scope exit.

    Parameters
    ----------
    enabled : bool, optional
        When ``False``, :meth:`ownership_claims` returns ``[]`` and
        :meth:`run_scope` yields ``None`` without creating files, threads, or
        probes. All configuration arguments are still validated eagerly.
    observer_id : str, optional
        Safe identifier scoping ``resources/<observer_id>.jsonl`` and
        ``status/<observer_id>.json`` inside the observability bundle.
    interval_seconds : float, optional
        Default sampling interval in seconds (strictly positive, finite).
    close_timeout_seconds : float, optional
        Single total close/drain allowance in seconds on scope exit.
    max_pending_envelopes : int, optional
        Maximum queued unpersisted sample packets before nonblocking drops.
    rusage : RUsageSource or mapping or None, optional
        Self-process ``RUSAGE_SELF`` source configuration. Defaults to enabled
        when ``rusage``, ``allocator``, and ``linux_memory`` are all omitted.
    allocator : AllocatorSource or mapping or None, optional
        Configured-device Torch allocator source configuration. Defaults to
        disabled when omitted so default construction never claims
        ``allocator_reset`` or imports Torch.
    linux_memory : LinuxMemorySource or mapping or None, optional
        Linux self-process current-memory procfs source configuration. Defaults
        to disabled when omitted.
    linux_tree : LinuxProcessTreeSource or mapping or None, optional
        Bounded Linux procfs descendant-tree source configuration. Defaults to
        disabled when omitted.
    linux_cgroup : LinuxCgroupV2Source or mapping or None, optional
        Read-only Linux cgroup v2 source configuration. Defaults to disabled
        when omitted.
    nvidia_device : NvidiaDeviceSource or mapping or None, optional
        Explicit, bounded NVIDIA physical-GPU ``nvidia-smi`` source
        configuration. Defaults to disabled when omitted; never claims
        ``allocator_reset`` authority or resets allocator peaks.
    sink : RunArtifactSink or mapping or None, optional
        Pure destination and byte/text budget configuration routed through
        :class:`tpen.artifacts.ArtifactManager`.
    monotonic_ns : callable or None, optional
        Optional nanosecond monotonic clock override for deterministic tests.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        observer_id: str = "resource-samples",
        interval_seconds: float = 10.0,
        close_timeout_seconds: float = 10.0,
        max_pending_envelopes: int = 2,
        rusage: RUsageSource | Mapping[str, Any] | None = None,
        allocator: AllocatorSource | Mapping[str, Any] | None = None,
        linux_memory: LinuxMemorySource | Mapping[str, Any] | None = None,
        linux_tree: LinuxProcessTreeSource | Mapping[str, Any] | None = None,
        linux_cgroup: LinuxCgroupV2Source | Mapping[str, Any] | None = None,
        nvidia_device: NvidiaDeviceSource | Mapping[str, Any] | None = None,
        sink: RunArtifactSink | Mapping[str, Any] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            _reject_forbidden_selector_keys(kwargs, owner="ResourceSamples")
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"ResourceSamples.__init__() got unexpected keyword argument(s): {unexpected}"
            )

        self.enabled = _require_bool(enabled, "ResourceSamples.enabled")
        self.observer_id = validate_identifier(
            observer_id, "ResourceSamples.observer_id", forbid_status_collision=True
        )
        self.interval_seconds = _require_positive_finite(
            interval_seconds, "ResourceSamples.interval_seconds"
        )
        self.close_timeout_seconds = _require_positive_finite(
            close_timeout_seconds, "ResourceSamples.close_timeout_seconds"
        )
        self.max_pending_envelopes = _require_positive_int(
            max_pending_envelopes, "ResourceSamples.max_pending_envelopes"
        )

        coerced_rusage = _coerce_rusage_source(rusage)
        coerced_allocator = _coerce_allocator_source(allocator)
        coerced_linux_memory = _coerce_linux_memory_source(linux_memory)
        coerced_linux_tree = _coerce_linux_tree_source(linux_tree)
        coerced_linux_cgroup = _coerce_linux_cgroup_source(linux_cgroup)
        coerced_nvidia_device = _coerce_nvidia_device_source(nvidia_device)
        if (
            coerced_rusage is None
            and coerced_allocator is None
            and coerced_linux_memory is None
            and coerced_linux_tree is None
            and coerced_linux_cgroup is None
            and coerced_nvidia_device is None
        ):
            coerced_rusage = RUsageSource(enabled=True)
            coerced_allocator = AllocatorSource(enabled=False, reset_peaks=False)
            coerced_linux_memory = LinuxMemorySource(enabled=False)
            coerced_linux_tree = LinuxProcessTreeSource(enabled=False)
            coerced_linux_cgroup = LinuxCgroupV2Source(enabled=False)
            coerced_nvidia_device = NvidiaDeviceSource(enabled=False)
        else:
            if coerced_rusage is None:
                coerced_rusage = RUsageSource(enabled=False)
            if coerced_allocator is None:
                coerced_allocator = AllocatorSource(enabled=False, reset_peaks=False)
            if coerced_linux_memory is None:
                coerced_linux_memory = LinuxMemorySource(enabled=False)
            if coerced_linux_tree is None:
                coerced_linux_tree = LinuxProcessTreeSource(enabled=False)
            if coerced_linux_cgroup is None:
                coerced_linux_cgroup = LinuxCgroupV2Source(enabled=False)
            if coerced_nvidia_device is None:
                coerced_nvidia_device = NvidiaDeviceSource(enabled=False)

        self.rusage = coerced_rusage
        self.allocator = coerced_allocator
        self.linux_memory = coerced_linux_memory
        self.linux_tree = coerced_linux_tree
        self.linux_cgroup = coerced_linux_cgroup
        self.nvidia_device = coerced_nvidia_device
        self.sink = _coerce_sink(sink)

        enabled_sources: list[tuple[str, str]] = []
        if self.rusage.enabled:
            enabled_sources.append(("rusage", self.rusage.source_id))
        if self.allocator.enabled:
            enabled_sources.append(("allocator", self.allocator.source_id))
        if self.linux_memory.enabled:
            enabled_sources.append(("linux_memory", self.linux_memory.source_id))
        if self.linux_tree.enabled:
            enabled_sources.append(("linux_tree", self.linux_tree.source_id))
        if self.linux_cgroup.enabled:
            enabled_sources.append(("linux_cgroup", self.linux_cgroup.source_id))
        if self.nvidia_device.enabled:
            enabled_sources.append(("nvidia_device", self.nvidia_device.source_id))

        if self.enabled and not enabled_sources:
            raise ValueError(
                "ResourceSamples requires at least one enabled source "
                "(rusage, allocator, linux_memory, linux_tree, linux_cgroup, "
                "or nvidia_device) when enabled=True"
            )
        seen_source_ids: dict[str, str] = {}
        for source_name, source_id in enabled_sources:
            if source_id in seen_source_ids:
                raise ValueError(
                    f"{seen_source_ids[source_id]} and {source_name} sources must use "
                    f"distinct source_id values, got {source_id!r}"
                )
            seen_source_ids[source_id] = source_name

        if monotonic_ns is not None and not callable(monotonic_ns):
            raise TypeError("ResourceSamples.monotonic_ns must be callable or None")
        self._monotonic_ns: Callable[[], int] = (
            time.monotonic_ns if monotonic_ns is None else monotonic_ns
        )

        # Class-owned typed subscriptions strictly track terminal run outcomes
        # without exposing any YAML event/type selector bridge.
        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(RunCompleted),
                        Subscription.of(RunFailed),
                    )
                ),
            )
        )

        self._state_lock = threading.Lock()
        self._active = False
        self._unsafe = False
        self._active_session: _ResourceSamplesSession | None = None
        self._observed_application_outcome: str = "not_supplied"

    def _reset_typed_state(self) -> None:
        # Only reset between invocations; an active scope keeps its live state.
        with self._state_lock:
            if not self._active:
                self._observed_application_outcome = "not_supplied"

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: CallbackContext
    ) -> None:
        """Record core-emitted terminal application outcome for status reporting."""

        del context
        event = occurrence.event
        if isinstance(event, RunCompleted):
            self._observed_application_outcome = event.status
        elif isinstance(event, RunFailed):
            self._observed_application_outcome = "failed"

    def is_managed_scope_active(self) -> bool:
        """Return whether this callback currently holds an active or unsafe scope."""

        with self._state_lock:
            return self._active or self._unsafe

    def _resolve_claim_device(
        self, *, cfg: Any = None, metadata: Any = None
    ) -> str | None:
        if self.allocator.device is not None:
            return self.allocator.device
        meta_device = getattr(metadata, "device", None)
        if isinstance(meta_device, str) and meta_device.strip():
            return meta_device.strip()
        if cfg is not None:
            if isinstance(cfg, Mapping):
                runtime = cfg.get("runtime")
                if isinstance(runtime, Mapping) and runtime.get("device") is not None:
                    return str(runtime["device"]).strip()
            else:
                # OmegaConf DictConfig or attribute-style container.
                try:
                    from omegaconf import OmegaConf

                    selected = OmegaConf.select(cfg, "runtime.device", default=None)
                except Exception:
                    selected = None
                if selected is not None:
                    return str(selected).strip()
        return None

    def ownership_claims(
        self,
        cfg: Any = None,
        run_dir: Any = None,
        metadata: Any = None,
        *,
        artifact_manager: ArtifactManager | None = None,
        invocation_id: str | None = None,
    ) -> list[ResourceOwnershipClaim]:
        """Declare exclusive artifact-writer and allocator-reset ownership claims.

        Returns an empty list when ``enabled=False``. When ``enabled=True``,
        returns the routed ``artifact_writer`` claim(s) for this observer plus
        an ``allocator_reset`` claim whenever ``allocator.enabled`` and
        ``allocator.reset_peaks`` target a non-CPU (or unresolved) device.
        """

        if not self.enabled:
            return []

        claims: list[ResourceOwnershipClaim] = []
        if artifact_manager is not None and invocation_id is not None:
            claims.append(
                self.sink.ownership_claim(
                    artifact_manager, invocation_id, self.observer_id
                )
            )
            bundle_dir = self.sink.resolve_bundle_dir(artifact_manager, invocation_id)
            valid_invocation = validate_identifier(invocation_id, "invocation_id")
            manifest_rel = (
                *self.sink.relative_dir.split("/"),
                valid_invocation,
                "manifest.json",
            )
            try:
                manifest_path = artifact_manager.path(*manifest_rel)
            except ArtifactRoutingError as exc:
                raise ObservabilityPathError(str(exc)) from exc
            logical_manifest = artifact_manager.run_dir.joinpath(*manifest_rel)
            if manifest_path.is_symlink() or logical_manifest.is_symlink():
                symlink_path = (
                    manifest_path if manifest_path.is_symlink() else logical_manifest
                )
                raise ObservabilityPathError(
                    f"refusing symlinked observability path {symlink_path!s}"
                )
            resolved_bundle = bundle_dir.resolve(strict=False)
            resolved_manifest = manifest_path.resolve(strict=False)
            try:
                resolved_manifest.relative_to(resolved_bundle)
            except ValueError as exc:
                raise ObservabilityPathError(
                    f"observability path {manifest_path!s} resolves to {resolved_manifest!s}, "
                    f"outside contained root {resolved_bundle!s}"
                ) from exc
            claims.append(
                ResourceOwnershipClaim(
                    kind="artifact_writer",
                    scope=str(resolved_manifest),
                )
            )
        elif run_dir is not None:
            from pathlib import Path

            base = Path(run_dir) / self.sink.relative_dir
            stream_scope = str(
                (base / "resources" / f"{self.observer_id}.jsonl").resolve(strict=False)
            )
            manifest_scope = str((base / "manifest.json").resolve(strict=False))
            claims.append(
                ResourceOwnershipClaim(kind="artifact_writer", scope=stream_scope)
            )
            claims.append(
                ResourceOwnershipClaim(kind="artifact_writer", scope=manifest_scope)
            )
        else:
            claims.append(
                ResourceOwnershipClaim(
                    kind="artifact_writer",
                    scope=f"{self.sink.relative_dir}/resources/{self.observer_id}.jsonl",
                )
            )
            claims.append(
                ResourceOwnershipClaim(
                    kind="artifact_writer",
                    scope=f"{self.sink.relative_dir}/manifest.json",
                )
            )

        if self.allocator.enabled and self.allocator.reset_peaks:
            configured_device = self._resolve_claim_device(cfg=cfg, metadata=metadata)
            scope = (
                "unknown"
                if configured_device is None
                else canonicalize_allocator_reset_scope(configured_device)
            )
            if scope is not None:
                claims.append(
                    ResourceOwnershipClaim(kind="allocator_reset", scope=scope)
                )

        return claims

    def _begin_session(self, context: RunContext) -> _ResourceSamplesSession:
        with self._state_lock:
            if self._unsafe:
                raise ManagedLifecycleError(
                    "ResourceSamples cannot be re-entered after unconfirmed prior cleanup"
                )
            if self._active:
                raise ManagedLifecycleError(
                    "ResourceSamples scope is already active; concurrent re-entry is forbidden"
                )
            self._active = True
            self._observed_application_outcome = "not_supplied"
            session = _ResourceSamplesSession(self, context)
            self._active_session = session
        # Bind outside _state_lock (and after _active=True so _reset_typed_state
        # does not clear active session state or re-enter _state_lock).
        self.bind_typed_context(context)
        return session

    def _end_session(
        self, session: _ResourceSamplesSession, *, quiescent: bool
    ) -> None:
        with self._state_lock:
            if self._active_session is session:
                self._active_session = None
            self._active = False
            if not quiescent:
                self._unsafe = True

    def run_scope(self, context: RunContext) -> ContextManager[RunResource | None]:
        """Return a context manager owning sampling resources for one invocation."""

        if not self.enabled:
            return _DisabledScope()
        return _ResourceSamplesScope(self, context)


__all__ = [
    "ALLOCATOR_CURRENT_QUANTITY_IDS",
    "ALLOCATOR_EXECUTION_MODE",
    "ALLOCATOR_PEAK_QUANTITY_IDS",
    "ALLOCATOR_SOURCE_ID",
    "DEFAULT_LINUX_CGROUP_V2_QUANTITIES",
    "DEFAULT_LINUX_MEMORY_QUANTITIES",
    "DEFAULT_LINUX_PROCESS_TREE_QUANTITIES",
    "DEFAULT_NVIDIA_DEVICE_QUANTITIES",
    "DEFAULT_RUSAGE_QUANTITIES",
    "LINUX_CGROUP_V2_EXECUTION_MODE",
    "LINUX_CGROUP_V2_QUANTITY_IDS",
    "LINUX_CGROUP_V2_SOURCE_ID",
    "LINUX_MEMORY_EXECUTION_MODE",
    "LINUX_MEMORY_QUANTITY_IDS",
    "LINUX_MEMORY_SOURCE_ID",
    "LINUX_PROCESS_TREE_EXECUTION_MODE",
    "LINUX_PROCESS_TREE_QUANTITY_IDS",
    "LINUX_PROCESS_TREE_SOURCE_ID",
    "NVIDIA_DEVICE_EXECUTION_MODE",
    "NVIDIA_DEVICE_QUANTITY_IDS",
    "NVIDIA_DEVICE_SOURCE_ID",
    "RUSAGE_EXECUTION_MODE",
    "RUSAGE_QUANTITY_IDS",
    "RUSAGE_SOURCE_ID",
    "AllocatorSource",
    "LinuxCgroupV2Source",
    "LinuxMemorySource",
    "LinuxProcessTreeSource",
    "NvidiaDeviceSource",
    "RUsageSource",
    "ResourceSamples",
]
