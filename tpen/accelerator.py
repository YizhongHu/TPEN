"""Accelerator backend resolution for a configured runtime device.

This module owns the mapping from a device (``runtime.device``) to the torch
module implementing that device's accelerator API, so no other call site names a
concrete backend such as ``torch.cuda``. Naming a backend directly is what makes
code CUDA-only: on an XPU build ``torch.cuda`` still exists and
``torch.cuda.is_available()`` returns ``False`` without raising, so a hardcoded
CUDA path silently degrades to CPU instead of failing.

``torch.get_device_module`` exposes the same surface for every backend
(``manual_seed_all``, ``get_rng_state_all``, ``synchronize``, memory statistics),
so no per-backend branching is required here.

See ADR-013: generator state is device-bound and never portable across device
types. Nothing in this module reinterprets RNG state; `canonical_device` only
makes devices comparable so callers can detect a mismatch and fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from tpen.dependencies import require_torch

_MIN_TORCH_HINT = "torch>=2.5 provides torch.get_device_module"
_BYTES_PER_MIB = 1024 * 1024


class AcceleratorKind(Enum):
    """Accelerator identity kinds understood by resource profiling."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    OTHER = "other"


@dataclass(frozen=True)
class AcceleratorIdentity:
    """The configured backend and physical device identity, when available.

    R3's resource artifacts consume this provenance record when persisting
    run-level hardware identity.

    Attributes
    ----------
    kind : AcceleratorKind
        Backend classification for the configured device.
    index : int or None
        Index of the configured device, when the backend reports one.
    uuid : str or None
        Physical device UUID, when the backend reports one. On MIG-partitioned
        CUDA devices this identifies the physical GPU rather than the MIG
        instance, so separate MIG slices of one GPU can share this value.
    """

    kind: AcceleratorKind
    index: int | None
    uuid: str | None


@dataclass(frozen=True)
class AllocatorUnavailable:
    """Typed evidence that an allocator counter could not be read."""

    reason: str


AllocatorReading = float | AllocatorUnavailable


@dataclass(frozen=True)
class AllocatorUsage:
    """Peak allocator readings for one configured device."""

    identity: AcceleratorIdentity
    allocated_mb: AllocatorReading
    reserved_mb: AllocatorReading
    device_count: int | AllocatorUnavailable | None = None


@runtime_checkable
class AllocatorPeakProbe(Protocol):
    """Callback seam for a configured-device allocator probe."""

    def reset(self) -> AcceleratorIdentity | AllocatorUnavailable:
        """Reset peaks for the configured device."""

    def read(self) -> AllocatorUsage:
        """Read peaks for the configured device."""


def _accelerator_kind(torch: Any, device_type: str) -> AcceleratorKind:
    """Classify a torch device without importing a vendor module."""

    if device_type == "cpu":
        return AcceleratorKind.CPU
    if device_type == "cuda":
        try:
            hip = torch.version.hip
        except AttributeError:
            hip = None
        return AcceleratorKind.ROCM if hip is not None else AcceleratorKind.CUDA
    return AcceleratorKind.OTHER


class TorchAllocatorPeakProbe:
    """Read peak allocator counters for exactly one configured torch device."""

    def __init__(self, device: Any) -> None:
        torch = _torch("allocator peak metrics")
        self.device = torch.device(device)
        self.module = device_module(self.device, feature="allocator peak metrics")
        self.kind = _accelerator_kind(torch, self.device.type)
        if self.device.index is None and self.kind is not AcceleratorKind.CPU and self._available():
            # An indexless device (e.g. "cuda") means "whatever torch calls
            # current" at the moment of each call, which can differ between
            # reset() and read(). Resolve it once here so every subsequent
            # operation targets the same physical device for this probe's
            # whole lifetime.
            try:
                index = self.module.current_device()
            except (AttributeError, RuntimeError):
                index = None
            if index is not None:
                self.device = torch.device(self.device.type, index)

    def _identity(self) -> AcceleratorIdentity:
        index = self.device.index
        uuid = None
        try:
            available = self.kind is not AcceleratorKind.CPU and self.module.is_available()
        except (AttributeError, RuntimeError):
            return AcceleratorIdentity(kind=self.kind, index=index, uuid=None)
        if available and index is None:
            try:
                index = self.module.current_device()
            except (AttributeError, RuntimeError):
                return AcceleratorIdentity(kind=self.kind, index=None, uuid=None)
        if available and index is not None:
            try:
                raw_uuid = self.module.get_device_properties(index).uuid
                uuid = str(raw_uuid) if raw_uuid is not None else None
            except (AttributeError, RuntimeError):
                uuid = None
        return AcceleratorIdentity(kind=self.kind, index=index, uuid=uuid)

    def identity(self) -> AcceleratorIdentity:
        """Return the exact identity for the configured device."""

        return self._identity()

    def _available(self) -> bool:
        """Return backend availability without allowing telemetry to fail a run."""

        try:
            return self.kind is not AcceleratorKind.CPU and self.module.is_available()
        except (AttributeError, RuntimeError):
            return False

    def reset(self) -> AcceleratorIdentity | AllocatorUnavailable:
        """Reset this device's peaks, returning typed unavailable evidence."""

        identity = self._identity()
        if self.kind is AcceleratorKind.CPU:
            return identity
        if not self._available():
            return AllocatorUnavailable("configured accelerator is unavailable")
        try:
            self.module.reset_peak_memory_stats(self.device)
        except (RuntimeError, AttributeError) as exc:
            return AllocatorUnavailable(f"{type(exc).__name__}: {exc}")
        return identity

    def read(self) -> AllocatorUsage:
        """Read peak allocated and reserved memory in MiB, and device count.

        Each counter is read independently so one failing counter does not
        blank the others -- a real backend can answer some queries and not
        others.
        """

        identity = self._identity()
        if self.kind is AcceleratorKind.CPU or not self._available():
            unavailable = AllocatorUnavailable("configured accelerator is unavailable")
            return AllocatorUsage(identity, unavailable, unavailable, unavailable)
        allocated = self._read_mib(self.module.max_memory_allocated)
        reserved = self._read_mib(self.module.max_memory_reserved)
        device_count = self._read_device_count()
        return AllocatorUsage(identity, allocated, reserved, device_count)

    def _read_mib(self, counter: Any) -> AllocatorReading:
        try:
            return float(counter(self.device)) / _BYTES_PER_MIB
        except (RuntimeError, AttributeError) as exc:
            return AllocatorUnavailable(f"{type(exc).__name__}: {exc}")

    def _read_device_count(self) -> int | AllocatorUnavailable:
        try:
            return int(self.module.device_count())
        except (RuntimeError, AttributeError) as exc:
            return AllocatorUnavailable(f"{type(exc).__name__}: {exc}")


def _torch(feature: str) -> Any:
    torch = require_torch(feature=feature)
    if not hasattr(torch, "get_device_module"):  # pragma: no cover - ancient torch
        raise RuntimeError(f"{feature} requires a torch exposing get_device_module ({_MIN_TORCH_HINT}); got {torch.__version__}")
    return torch


def _optional_device_module(torch: Any, device_type: str) -> Any | None:
    """Return the device module for ``device_type``, or ``None`` when none exists.

    Device types such as ``meta`` are valid to construct but have no registered
    accelerator module, and therefore no current device to resolve.
    """

    try:
        return torch.get_device_module(device_type)
    except (AttributeError, RuntimeError):
        return None


def current_accelerator_type(*, feature: str = "accelerator detection") -> str:
    """Return the active accelerator device type, or ``"cpu"`` when there is none.

    Parameters
    ----------
    feature : str, optional
        Feature name used in the torch-missing error message.

    Returns
    -------
    str
        Device type such as ``"cuda"``, ``"xpu"``, or ``"cpu"``.
    """

    torch = _torch(feature)
    accelerator = getattr(torch, "accelerator", None)
    current = getattr(accelerator, "current_accelerator", None)
    if callable(current):
        device = current()
        if device is not None:
            return str(torch.device(device).type)
    return "cpu"


def device_module(device: Any = None, *, feature: str = "accelerator support") -> Any:
    """Return the torch module implementing ``device``'s accelerator API.

    Parameters
    ----------
    device : torch.device, str, or None, optional
        Target device. ``None`` resolves the active accelerator, falling back to
        CPU when no accelerator is present.
    feature : str, optional
        Feature name used in the torch-missing error message.

    Returns
    -------
    module
        ``torch.cpu``, ``torch.cuda``, ``torch.xpu``, or another backend module.
    """

    torch = _torch(feature)
    device_type = current_accelerator_type(feature=feature) if device is None else str(torch.device(device).type)
    return torch.get_device_module(device_type)


def canonical_device(device: Any, *, feature: str = "device comparison") -> Any:
    """Return a fully indexed device so ``cuda`` and ``cuda:0`` compare equal.

    Tensors report an indexed accelerator device (``cuda:0``, ``xpu:0``) while
    configs and callers usually pass the index-less form, and ``torch.device``
    treats those as unequal. CPU devices are reported index-less by tensors and
    pass through unchanged, as do devices that already carry an index, device
    types with no registered accelerator module (such as ``meta``), and
    accelerators that are not currently available.

    Parameters
    ----------
    device : torch.device or str
        Device to canonicalize.
    feature : str, optional
        Feature name used in the torch-missing error message.

    Returns
    -------
    torch.device
        Device with an explicit index when one can be resolved.
    """

    torch = _torch(feature)
    resolved = torch.device(device)
    if resolved.type == "cpu" or resolved.index is not None:
        return resolved
    module = _optional_device_module(torch, resolved.type)
    if module is None:
        return resolved
    is_available = getattr(module, "is_available", None)
    if not callable(is_available) or not is_available():
        return resolved
    current_device = getattr(module, "current_device", None)
    if not callable(current_device):
        return resolved
    return torch.device(resolved.type, current_device())


def synchronize(device: Any = None, *, feature: str = "device synchronization") -> None:
    """Synchronize the accelerator so host-side timing measures device work.

    Does nothing when the accelerator is unavailable or exposes no
    ``synchronize``, which keeps CPU runs free of accelerator calls.

    Parameters
    ----------
    device : torch.device, str, or None, optional
        Device to synchronize. ``None`` uses the active accelerator.
    feature : str, optional
        Feature name used in the torch-missing error message.
    """

    module = device_module(device, feature=feature)
    is_available = getattr(module, "is_available", None)
    if callable(is_available) and not is_available():
        return
    synchronize_fn = getattr(module, "synchronize", None)
    if callable(synchronize_fn):
        synchronize_fn()


@dataclass
class DeviceEventTimer:
    """Measure elapsed device time with one accelerator event pair.

    Instances are deliberately single-use between `start` and `stop`: one
    operation scope owns one timer, so accepting nested starts would silently
    associate the wrong device interval with a metric.
    """

    _module: Any
    _event_factory: Any
    _start_event: Any | None = None

    def start(self) -> None:
        """Record the beginning of one device interval."""

        if self._start_event is not None:
            raise RuntimeError("device event timer is already running")
        start_event = self._event_factory(enable_timing=True)
        start_event.record()
        self._start_event = start_event

    def stop(self) -> float:
        """Synchronize and return elapsed device seconds for one interval."""

        start_event = self._start_event
        if start_event is None:
            raise RuntimeError("device event timer was not started")
        try:
            end_event = self._event_factory(enable_timing=True)
            end_event.record()
            self._module.synchronize()
            return float(start_event.elapsed_time(end_event)) / 1_000.0
        finally:
            self._start_event = None


def device_event_timer(
    device: Any = None, *, feature: str = "device event timing"
) -> DeviceEventTimer:
    """Return a fail-loud event timer for an available accelerator.

    Unlike `synchronize`, this never degrades to a host timer: callers opt in
    precisely because they need device elapsed time.
    """

    module = device_module(device, feature=feature)
    available = getattr(module, "is_available", None)
    if not callable(available) or not available():
        raise RuntimeError(f"{feature} requires an available accelerator")
    event_factory = getattr(module, "Event", None)
    synchronize_fn = getattr(module, "synchronize", None)
    if not callable(event_factory) or not callable(synchronize_fn):
        raise RuntimeError(f"{feature} requires timing-event support from the accelerator backend")
    return DeviceEventTimer(_module=module, _event_factory=event_factory)


def seed_all(seed: int, *, feature: str = "seeded run") -> None:
    """Seed the accelerator's global RNG for every visible device.

    This is run-level global seeding only. Per ADR-013, stochastic components own
    their own generators and must not be seeded on their behalf; this function
    exists so the one explicit run-level seed reaches the accelerator whatever
    the backend.

    Parameters
    ----------
    seed : int
        Seed value.
    feature : str, optional
        Feature name used in the torch-missing error message.
    """

    module = device_module(feature=feature)
    seed_fn = getattr(module, "manual_seed_all", None)
    if callable(seed_fn):
        seed_fn(int(seed))


__all__ = [
    "AcceleratorIdentity",
    "AcceleratorKind",
    "AllocatorPeakProbe",
    "AllocatorUnavailable",
    "AllocatorUsage",
    "DeviceEventTimer",
    "canonical_device",
    "current_accelerator_type",
    "device_event_timer",
    "device_module",
    "seed_all",
    "synchronize",
    "TorchAllocatorPeakProbe",
]
