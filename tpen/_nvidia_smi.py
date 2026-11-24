"""Explicit, bounded NVIDIA ``nvidia-smi`` physical-device probe (strictly stdlib).

This internal implementation module is publicly re-exported by
:mod:`tpen.accelerator`. It provides a strictly standard-library, Torch-free,
read-only probe for a single already-selected physical CUDA GPU:

* :class:`NvidiaSmiDeviceProbe` — explicitly configured probe that queries
  only the already-selected physical :class:`tpen.accelerator.AcceleratorIdentity`
  by canonical device UUID via a supervised, bounded ``nvidia-smi`` subprocess
  (hard timeout cap of 10.0 seconds, live output-byte cap, prompt cancellation,
  and process-group cleanup on timeout/cancellation/overflow).
* :func:`canonicalize_nvidia_device_uuid` — strict UUID canonicalizer that
  normalizes bare or ``GPU-`` prefixed UUIDs to canonical ``GPU-<lowercase-uuid>``,
  maps ``MIG-...`` identifiers to typed ``mig_subdevice_ambiguous``
  unavailability, and rejects flag-like or malformed tokens before CLI use.
* :func:`parse_nvidia_smi_query_output` — strict single-row CSV parser that
  cross-checks the returned UUID against the selected physical device, refuses
  MIG-enabled or permission-ambiguous MIG devices fail-closed, preserves
  vendor-window timestamps, and converts ``memory.used``/``memory.total``
  (MiB -> bytes) and ``utilization.gpu`` (percent -> ``[0.0, 1.0]`` busy
  fraction) independently without guessing ``0`` for unavailable fields.

This module never installs or downloads executables, never falls back to
querying GPU 0 or unrelated GPUs, never attributes physical GPU readings to
individual processes, and never resets or mutates Torch allocator state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tpen.accelerator import AcceleratorIdentity, AcceleratorKind


_BYTES_PER_MIB = 1024 * 1024
_NS_PER_SECOND = 1_000_000_000
_POLL_SLICE_SECONDS = 0.05
_READ_CHUNK_BYTES = 4096
_TERMINATE_GRACE_SECONDS = 0.2
_KILL_WAIT_SECONDS = 1.0
_MAX_DETAIL_CHARS = 256

DEFAULT_NVIDIA_SMI_EXECUTABLE = "nvidia-smi"
DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
MAX_NVIDIA_SMI_TIMEOUT_SECONDS = 10.0
DEFAULT_NVIDIA_SMI_MAX_OUTPUT_BYTES = 65536

NVIDIA_DEVICE_READING_STATUSES: frozenset[str] = frozenset(
    {"ok", "partial", "unavailable", "error"}
)
NVIDIA_DEVICE_MIG_MODES: frozenset[str] = frozenset(
    {"disabled", "enabled", "not_supported", "insufficient_permissions", "unknown"}
)
NVIDIA_SMI_QUERY_FIELDS: tuple[str, ...] = (
    "uuid",
    "timestamp",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "mig.mode.current",
)
NVIDIA_SMI_QUERY_FORMAT = "csv,noheader,nounits"

_GPU_UUID_RE = re.compile(
    r"^(?:[Gg][Pp][Uu]-)?([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})\Z"
)
_STRICT_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z"
)

_NA_TOKENS: frozenset[str] = frozenset({"[n/a]", "n/a"})
_NOT_SUPPORTED_TOKENS: frozenset[str] = frozenset({"[not supported]"})
_INSUFFICIENT_PERMISSIONS_TOKENS: frozenset[str] = frozenset(
    {"[insufficient permissions]"}
)


def _accel() -> Any:
    """Return :mod:`tpen.accelerator` lazily without circular top-level import."""

    import tpen.accelerator as accelerator

    return accelerator


def _has_control_chars(text: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in text)


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a positive integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _require_timeout_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    if number > MAX_NVIDIA_SMI_TIMEOUT_SECONDS:
        raise ValueError(
            f"{name} must be <= {MAX_NVIDIA_SMI_TIMEOUT_SECONDS} s, got {value!r}"
        )
    return number


def _validate_executable(value: object, name: str) -> str:
    if isinstance(value, Path):
        text = str(value)
    elif type(value) is str:
        text = value
    else:
        raise TypeError(f"{name} must be str or Path, got {type(value).__name__}")
    if not text.strip() or _has_control_chars(text):
        raise ValueError(
            f"{name} must be a nonempty executable name or path without control characters"
        )
    return text


def _sanitize_detail(text: str) -> str:
    cleaned = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if len(cleaned) > _MAX_DETAIL_CHARS:
        return cleaned[: _MAX_DETAIL_CHARS - 3] + "..."
    return cleaned


@dataclass(frozen=True)
class NvidiaDeviceUnavailable:
    """Typed evidence that an NVIDIA physical-device reading is unavailable or errored.

    Attributes
    ----------
    reason : str
        Human-readable diagnostic explaining why the reading is absent.
    reason_code : str, optional
        Machine-readable classification code (e.g. ``"executable_unavailable"``,
        ``"mig_subdevice_ambiguous"``, ``"read_timeout"``).
    status : {"unavailable", "error"}, optional
        Distinguishes expected/environmental unavailability from hard command
        or parser errors.
    """

    reason: str
    reason_code: str = "nvidia_smi_unavailable"
    status: str = "unavailable"

    def __post_init__(self) -> None:
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("NvidiaDeviceUnavailable.reason must be a nonempty string")
        if type(self.reason_code) is not str or not self.reason_code.strip():
            raise ValueError(
                "NvidiaDeviceUnavailable.reason_code must be a nonempty string"
            )
        if self.status not in {"unavailable", "error"}:
            raise ValueError(
                "NvidiaDeviceUnavailable.status must be 'unavailable' or 'error', "
                f"got {self.status!r}"
            )


@dataclass(frozen=True)
class NvidiaDeviceCommandResult:
    """Bounded result returned by the supervised ``nvidia-smi`` command runner.

    Attributes
    ----------
    returncode : int or None
        Process exit status, or ``None`` when the command did not complete
        normally (timeout, cancellation, output-limit breach, or launch error).
    stdout : str
        Captured UTF-8 decoded standard output within ``max_output_bytes``.
    stderr : str, optional
        Captured UTF-8 decoded standard error within ``max_output_bytes``.
    timed_out : bool, optional
        Whether the command exceeded its bounded execution deadline.
    cancelled : bool, optional
        Whether execution was terminated early via ``cancel_event``.
    output_limit_exceeded : bool, optional
        Whether stdout or stderr exceeded ``max_output_bytes``.
    error_reason_code : str or None, optional
        Optional machine-readable error code when ``returncode is None``.
    error_detail : str or None, optional
        Optional concise diagnostic message when ``returncode is None``.
    """

    returncode: int | None
    stdout: str
    stderr: str = ""
    timed_out: bool = False
    cancelled: bool = False
    output_limit_exceeded: bool = False
    error_reason_code: str | None = None
    error_detail: str | None = None

    def __post_init__(self) -> None:
        if self.returncode is not None and type(self.returncode) is not int:
            raise TypeError(
                "NvidiaDeviceCommandResult.returncode must be int or None, "
                f"got {type(self.returncode).__name__}"
            )
        if type(self.stdout) is not str:
            raise TypeError(
                f"NvidiaDeviceCommandResult.stdout must be str, got {type(self.stdout).__name__}"
            )
        if type(self.stderr) is not str:
            raise TypeError(
                f"NvidiaDeviceCommandResult.stderr must be str, got {type(self.stderr).__name__}"
            )
        _require_bool(self.timed_out, "NvidiaDeviceCommandResult.timed_out")
        _require_bool(self.cancelled, "NvidiaDeviceCommandResult.cancelled")
        _require_bool(
            self.output_limit_exceeded,
            "NvidiaDeviceCommandResult.output_limit_exceeded",
        )
        if self.error_reason_code is not None and (
            type(self.error_reason_code) is not str
            or not self.error_reason_code.strip()
        ):
            raise ValueError(
                "NvidiaDeviceCommandResult.error_reason_code must be a nonempty string or None"
            )
        if self.error_detail is not None and type(self.error_detail) is not str:
            raise TypeError(
                "NvidiaDeviceCommandResult.error_detail must be str or None, "
                f"got {type(self.error_detail).__name__}"
            )


@dataclass(frozen=True)
class NvidiaDeviceReading:
    """One bounded physical-GPU reading captured from ``nvidia-smi``.

    All numeric metrics reflect whole-physical-GPU state for the already
    selected :class:`tpen.accelerator.AcceleratorIdentity` (never per-process
    attribution or extrapolated utilization).

    Attributes
    ----------
    status : {"ok", "partial", "unavailable", "error"}
        Overall reading classification. ``"ok"`` holds iff all three numeric
        fields are valid and MIG ambiguity is absent; ``"partial"`` holds iff
        1--2 numeric fields are valid; ``"unavailable"`` and ``"error"`` hold
        iff 0 numeric fields are valid.
    framebuffer_used_bytes : int or NvidiaDeviceUnavailable
        Current physical framebuffer memory used in bytes (converted from MiB).
    framebuffer_total_bytes : int or NvidiaDeviceUnavailable
        Total physical framebuffer memory capacity in bytes (converted from MiB).
    busy_fraction : float or NvidiaDeviceUnavailable
        Recent vendor-window GPU busy fraction in ``[0.0, 1.0]`` (converted
        from percent; not extrapolated utilization).
    vendor_timestamp : str or None, optional
        Vendor-reported timestamp string from ``nvidia-smi``, when available.
    selected_identity : AcceleratorIdentity or None, optional
        Selected physical accelerator identity targeted by this probe.
    observed_uuid : str or None, optional
        Canonical physical GPU UUID returned by ``nvidia-smi``, when observed.
    mig_mode : str, optional
        Normalized MIG mode state (``"disabled"``, ``"enabled"``,
        ``"not_supported"``, ``"insufficient_permissions"``, or ``"unknown"``).
    mig_ambiguous : bool, optional
        ``True`` when MIG/subdevice ambiguity prevented trusting whole-GPU
        readings.
    shared_device_not_exclusive : bool, optional
        Always ``True`` for physical GPU readings to record that whole-device
        metrics are shared and not per-process attribution.
    reason_code : str or None, optional
        Machine-readable summary reason code when ``status != "ok"``.
    detail : str or None, optional
        Optional bounded diagnostic detail when ``status != "ok"``.
    """

    status: str
    framebuffer_used_bytes: int | NvidiaDeviceUnavailable
    framebuffer_total_bytes: int | NvidiaDeviceUnavailable
    busy_fraction: float | NvidiaDeviceUnavailable
    vendor_timestamp: str | None = None
    selected_identity: AcceleratorIdentity | None = None
    observed_uuid: str | None = None
    mig_mode: str = "unknown"
    mig_ambiguous: bool = False
    shared_device_not_exclusive: bool = True
    reason_code: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status not in NVIDIA_DEVICE_READING_STATUSES:
            raise ValueError(
                f"NvidiaDeviceReading.status must be one of "
                f"{sorted(NVIDIA_DEVICE_READING_STATUSES)}, got {self.status!r}"
            )
        _require_bool(self.mig_ambiguous, "NvidiaDeviceReading.mig_ambiguous")
        _require_bool(
            self.shared_device_not_exclusive,
            "NvidiaDeviceReading.shared_device_not_exclusive",
        )
        if type(self.mig_mode) is not str or self.mig_mode not in NVIDIA_DEVICE_MIG_MODES:
            raise ValueError(
                f"NvidiaDeviceReading.mig_mode must be one of "
                f"{sorted(NVIDIA_DEVICE_MIG_MODES)}, got {self.mig_mode!r}"
            )
        if self.vendor_timestamp is not None and (
            type(self.vendor_timestamp) is not str or not self.vendor_timestamp.strip()
        ):
            raise ValueError(
                "NvidiaDeviceReading.vendor_timestamp must be a nonempty string or None"
            )
        if self.observed_uuid is not None and (
            type(self.observed_uuid) is not str or not self.observed_uuid.strip()
        ):
            raise ValueError(
                "NvidiaDeviceReading.observed_uuid must be a nonempty string or None"
            )
        if self.reason_code is not None and (
            type(self.reason_code) is not str or not self.reason_code.strip()
        ):
            raise ValueError(
                "NvidiaDeviceReading.reason_code must be a nonempty string or None"
            )
        if self.detail is not None and type(self.detail) is not str:
            raise TypeError(
                f"NvidiaDeviceReading.detail must be str or None, got {type(self.detail).__name__}"
            )

        valid_count = 0
        for field_name in ("framebuffer_used_bytes", "framebuffer_total_bytes"):
            val = getattr(self, field_name)
            if isinstance(val, NvidiaDeviceUnavailable):
                continue
            if type(val) is not int or val < 0:
                raise ValueError(
                    f"NvidiaDeviceReading.{field_name} must be a non-negative int "
                    f"or NvidiaDeviceUnavailable, got {val!r}"
                )
            valid_count += 1

        busy = self.busy_fraction
        if not isinstance(busy, NvidiaDeviceUnavailable):
            if isinstance(busy, bool) or not isinstance(busy, (int, float)):
                raise ValueError(
                    "NvidiaDeviceReading.busy_fraction must be a float in [0.0, 1.0] "
                    f"or NvidiaDeviceUnavailable, got {busy!r}"
                )
            busy_float = float(busy)
            if not math.isfinite(busy_float) or busy_float < 0.0 or busy_float > 1.0:
                raise ValueError(
                    "NvidiaDeviceReading.busy_fraction must be finite and in [0.0, 1.0], "
                    f"got {busy!r}"
                )
            object.__setattr__(self, "busy_fraction", busy_float)
            valid_count += 1

        if self.status == "ok":
            if valid_count != 3:
                raise ValueError(
                    f"NvidiaDeviceReading status='ok' requires all 3 valid numeric fields, got {valid_count}"
                )
            if self.mig_ambiguous:
                raise ValueError(
                    "NvidiaDeviceReading status='ok' is incompatible with mig_ambiguous=True"
                )
        elif self.status == "partial":
            if valid_count not in {1, 2}:
                raise ValueError(
                    f"NvidiaDeviceReading status='partial' requires 1 or 2 valid numeric fields, got {valid_count}"
                )
            if self.reason_code is None:
                raise ValueError(
                    "NvidiaDeviceReading status='partial' requires a nonempty reason_code"
                )
        else:
            if valid_count != 0:
                raise ValueError(
                    f"NvidiaDeviceReading status={self.status!r} requires 0 valid numeric fields, got {valid_count}"
                )
            if self.reason_code is None:
                raise ValueError(
                    f"NvidiaDeviceReading status={self.status!r} requires a nonempty reason_code"
                )


def _uniform_reading(
    *,
    issue: NvidiaDeviceUnavailable,
    selected_identity: AcceleratorIdentity | None = None,
    observed_uuid: str | None = None,
    vendor_timestamp: str | None = None,
    mig_mode: str = "unknown",
    mig_ambiguous: bool = False,
) -> NvidiaDeviceReading:
    """Build a fail-closed reading where all three numeric fields carry ``issue``."""

    return NvidiaDeviceReading(
        status=issue.status,
        framebuffer_used_bytes=issue,
        framebuffer_total_bytes=issue,
        busy_fraction=issue,
        vendor_timestamp=vendor_timestamp,
        selected_identity=selected_identity,
        observed_uuid=observed_uuid,
        mig_mode=mig_mode,
        mig_ambiguous=mig_ambiguous,
        shared_device_not_exclusive=True,
        reason_code=issue.reason_code,
        detail=issue.reason,
    )


def canonicalize_nvidia_device_uuid(raw_uuid: object) -> str | NvidiaDeviceUnavailable:
    """Canonicalize an NVIDIA device UUID and block MIG or CLI-injection tokens.

    Parameters
    ----------
    raw_uuid : object
        Candidate UUID string (bare ``8-4-4-4-12`` hex as reported by PyTorch
        ``str(get_device_properties(idx).uuid)``, or ``GPU-<8-4-4-4-12 hex>``).

    Returns
    -------
    str or NvidiaDeviceUnavailable
        Canonical ``"GPU-<lowercase-8-4-4-4-12>"`` string for a valid physical
        GPU UUID, typed ``unavailable`` (``reason_code="mig_subdevice_ambiguous"``)
        for ``MIG-...`` identifiers, or typed ``error``
        (``reason_code="invalid_device_uuid"``) for malformed or unsafe tokens.
    """

    if type(raw_uuid) is not str:
        return NvidiaDeviceUnavailable(
            reason=f"device UUID must be a string, got {type(raw_uuid).__name__}",
            reason_code="invalid_device_uuid",
            status="error",
        )
    if (
        not raw_uuid
        or _has_control_chars(raw_uuid)
        or any(ch.isspace() for ch in raw_uuid)
        or "," in raw_uuid
        or raw_uuid.startswith("-")
    ):
        return NvidiaDeviceUnavailable(
            reason=f"invalid NVIDIA device UUID {raw_uuid!r}",
            reason_code="invalid_device_uuid",
            status="error",
        )
    if raw_uuid.upper().startswith("MIG-"):
        if len(raw_uuid) <= 4:
            return NvidiaDeviceUnavailable(
                reason=f"invalid NVIDIA device UUID {raw_uuid!r}",
                reason_code="invalid_device_uuid",
                status="error",
            )
        return NvidiaDeviceUnavailable(
            reason=(
                f"selected device UUID {raw_uuid!r} identifies a MIG subdevice; "
                "whole-GPU nvidia-smi metrics are ambiguous"
            ),
            reason_code="mig_subdevice_ambiguous",
            status="unavailable",
        )
    match = _GPU_UUID_RE.match(raw_uuid)
    if match is None:
        return NvidiaDeviceUnavailable(
            reason=f"malformed NVIDIA physical GPU UUID {raw_uuid!r}",
            reason_code="invalid_device_uuid",
            status="error",
        )
    return f"GPU-{match.group(1).lower()}"


def _parse_mib_field(raw: str, field_name: str) -> int | NvidiaDeviceUnavailable:
    norm = raw.strip().lower()
    if norm in _NA_TOKENS:
        return NvidiaDeviceUnavailable(
            reason=f"{field_name} reported N/A",
            reason_code="metric_not_available",
            status="unavailable",
        )
    if norm in _NOT_SUPPORTED_TOKENS:
        return NvidiaDeviceUnavailable(
            reason=f"{field_name} reported [Not Supported]",
            reason_code="metric_not_supported",
            status="unavailable",
        )
    if norm in _INSUFFICIENT_PERMISSIONS_TOKENS:
        return NvidiaDeviceUnavailable(
            reason=f"{field_name} reported [Insufficient Permissions]",
            reason_code="permission_denied",
            status="unavailable",
        )
    token = raw.strip()
    if not _STRICT_NUMBER_RE.match(token):
        return NvidiaDeviceUnavailable(
            reason=f"malformed {field_name} value {raw!r}",
            reason_code="malformed_output",
            status="error",
        )
    value = float(token)
    if not math.isfinite(value) or value < 0.0:
        return NvidiaDeviceUnavailable(
            reason=f"invalid {field_name} value {raw!r}",
            reason_code="malformed_output",
            status="error",
        )
    scaled = value * _BYTES_PER_MIB
    if not math.isfinite(scaled):
        return NvidiaDeviceUnavailable(
            reason=f"overflowing {field_name} value {raw!r}",
            reason_code="malformed_output",
            status="error",
        )
    return int(round(scaled))


def _parse_busy_percent_field(raw: str, field_name: str) -> float | NvidiaDeviceUnavailable:
    norm = raw.strip().lower()
    if norm in _NA_TOKENS:
        return NvidiaDeviceUnavailable(
            reason=f"{field_name} reported N/A",
            reason_code="metric_not_available",
            status="unavailable",
        )
    if norm in _NOT_SUPPORTED_TOKENS:
        return NvidiaDeviceUnavailable(
            reason=f"{field_name} reported [Not Supported]",
            reason_code="metric_not_supported",
            status="unavailable",
        )
    if norm in _INSUFFICIENT_PERMISSIONS_TOKENS:
        return NvidiaDeviceUnavailable(
            reason=f"{field_name} reported [Insufficient Permissions]",
            reason_code="permission_denied",
            status="unavailable",
        )
    token = raw.strip()
    if not _STRICT_NUMBER_RE.match(token):
        return NvidiaDeviceUnavailable(
            reason=f"malformed {field_name} value {raw!r}",
            reason_code="malformed_output",
            status="error",
        )
    value = float(token)
    if not math.isfinite(value) or value < 0.0 or value > 100.0:
        return NvidiaDeviceUnavailable(
            reason=f"out-of-range {field_name} percent value {raw!r}",
            reason_code="malformed_output",
            status="error",
        )
    return value / 100.0


def parse_nvidia_smi_query_output(
    stdout: str,
    *,
    expected_uuid: str,
    selected_identity: AcceleratorIdentity | None = None,
) -> NvidiaDeviceReading:
    """Parse one single-row ``nvidia-smi`` CSV response with strict identity binding.

    Expects exact columns ``uuid,timestamp,memory.used,memory.total,
    utilization.gpu,mig.mode.current`` formatted with ``csv,noheader,nounits``.

    Parameters
    ----------
    stdout : str
        Captured standard output from ``nvidia-smi``.
    expected_uuid : str
        Canonical or bare UUID of the already-selected physical GPU.
    selected_identity : AcceleratorIdentity or None, optional
        Optional selected accelerator identity attached to the reading.

    Returns
    -------
    NvidiaDeviceReading
        Parsed reading with explicit status, vendor timestamp, MIG state,
        shared-device caveat, and per-field typed unavailability/errors.
    """

    canonical_expected = canonicalize_nvidia_device_uuid(expected_uuid)
    if isinstance(canonical_expected, NvidiaDeviceUnavailable):
        return _uniform_reading(
            issue=canonical_expected,
            selected_identity=selected_identity,
            mig_ambiguous=canonical_expected.reason_code == "mig_subdevice_ambiguous",
        )

    if type(stdout) is not str:
        return _uniform_reading(
            issue=NvidiaDeviceUnavailable(
                reason=f"nvidia-smi output must be str, got {type(stdout).__name__}",
                reason_code="malformed_output",
                status="error",
            ),
            selected_identity=selected_identity,
        )

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return _uniform_reading(
            issue=NvidiaDeviceUnavailable(
                reason=f"expected exactly 1 CSV row from nvidia-smi, got {len(lines)}",
                reason_code="malformed_output",
                status="error",
            ),
            selected_identity=selected_identity,
        )

    columns = [col.strip() for col in lines[0].split(",")]
    if len(columns) != len(NVIDIA_SMI_QUERY_FIELDS):
        return _uniform_reading(
            issue=NvidiaDeviceUnavailable(
                reason=(
                    f"expected {len(NVIDIA_SMI_QUERY_FIELDS)} CSV columns from "
                    f"nvidia-smi, got {len(columns)}"
                ),
                reason_code="malformed_output",
                status="error",
            ),
            selected_identity=selected_identity,
        )

    raw_uuid, raw_ts, raw_used, raw_total, raw_busy, raw_mig = columns

    canonical_observed = canonicalize_nvidia_device_uuid(raw_uuid)
    if isinstance(canonical_observed, NvidiaDeviceUnavailable):
        if canonical_observed.reason_code == "mig_subdevice_ambiguous":
            return _uniform_reading(
                issue=canonical_observed,
                selected_identity=selected_identity,
                observed_uuid=raw_uuid or None,
                mig_mode="enabled",
                mig_ambiguous=True,
            )
        return _uniform_reading(
            issue=NvidiaDeviceUnavailable(
                reason=f"malformed GPU UUID column in nvidia-smi output: {raw_uuid!r}",
                reason_code="malformed_output",
                status="error",
            ),
            selected_identity=selected_identity,
        )

    if canonical_observed != canonical_expected:
        return _uniform_reading(
            issue=NvidiaDeviceUnavailable(
                reason=(
                    f"nvidia-smi returned UUID {canonical_observed!r}, "
                    f"expected selected physical GPU {canonical_expected!r}"
                ),
                reason_code="identity_mismatch",
                status="error",
            ),
            selected_identity=selected_identity,
            observed_uuid=canonical_observed,
        )

    # Preserve vendor timestamp cleanly when present; reject control chars.
    ts_norm = raw_ts.lower()
    if (
        not raw_ts
        or ts_norm in _NA_TOKENS
        or ts_norm in _NOT_SUPPORTED_TOKENS
        or ts_norm in _INSUFFICIENT_PERMISSIONS_TOKENS
    ):
        vendor_timestamp: str | None = None
    elif _has_control_chars(raw_ts):
        return _uniform_reading(
            issue=NvidiaDeviceUnavailable(
                reason="nvidia-smi timestamp column contains control characters",
                reason_code="malformed_output",
                status="error",
            ),
            selected_identity=selected_identity,
            observed_uuid=canonical_observed,
        )
    else:
        vendor_timestamp = raw_ts

    # Evaluate MIG mode before trusting whole-GPU numeric readings.
    mig_norm = raw_mig.lower()
    if mig_norm == "enabled":
        return _uniform_reading(
            issue=NvidiaDeviceUnavailable(
                reason=(
                    "MIG mode is enabled on the selected physical GPU; whole-GPU "
                    "nvidia-smi metrics are ambiguous for MIG subdevices"
                ),
                reason_code="mig_subdevice_ambiguous",
                status="unavailable",
            ),
            selected_identity=selected_identity,
            observed_uuid=canonical_observed,
            vendor_timestamp=vendor_timestamp,
            mig_mode="enabled",
            mig_ambiguous=True,
        )
    if mig_norm in _INSUFFICIENT_PERMISSIONS_TOKENS:
        return _uniform_reading(
            issue=NvidiaDeviceUnavailable(
                reason=(
                    "mig.mode.current reported [Insufficient Permissions]; "
                    "refusing whole-GPU metrics because MIG state cannot be verified"
                ),
                reason_code="mig_status_unavailable",
                status="unavailable",
            ),
            selected_identity=selected_identity,
            observed_uuid=canonical_observed,
            vendor_timestamp=vendor_timestamp,
            mig_mode="insufficient_permissions",
            mig_ambiguous=True,
        )
    if mig_norm == "disabled":
        mig_mode = "disabled"
    elif mig_norm in _NA_TOKENS or mig_norm in _NOT_SUPPORTED_TOKENS:
        mig_mode = "not_supported"
    else:
        return _uniform_reading(
            issue=NvidiaDeviceUnavailable(
                reason=f"unrecognized mig.mode.current value {raw_mig!r}",
                reason_code="malformed_output",
                status="error",
            ),
            selected_identity=selected_identity,
            observed_uuid=canonical_observed,
            vendor_timestamp=vendor_timestamp,
            mig_mode="unknown",
            mig_ambiguous=False,
        )

    used_bytes = _parse_mib_field(raw_used, "memory.used")
    total_bytes = _parse_mib_field(raw_total, "memory.total")
    busy_fraction = _parse_busy_percent_field(raw_busy, "utilization.gpu")

    if (
        type(used_bytes) is int
        and type(total_bytes) is int
        and used_bytes > total_bytes
    ):
        inconsistent = NvidiaDeviceUnavailable(
            reason=(
                f"inconsistent framebuffer memory: used ({used_bytes} B) "
                f"exceeds total ({total_bytes} B)"
            ),
            reason_code="malformed_output",
            status="error",
        )
        used_bytes = inconsistent
        total_bytes = inconsistent

    failures = [
        item
        for item in (used_bytes, total_bytes, busy_fraction)
        if isinstance(item, NvidiaDeviceUnavailable)
    ]
    if not failures:
        return NvidiaDeviceReading(
            status="ok",
            framebuffer_used_bytes=used_bytes,
            framebuffer_total_bytes=total_bytes,
            busy_fraction=busy_fraction,
            vendor_timestamp=vendor_timestamp,
            selected_identity=selected_identity,
            observed_uuid=canonical_observed,
            mig_mode=mig_mode,
            mig_ambiguous=False,
            shared_device_not_exclusive=True,
            reason_code=None,
            detail=None,
        )

    error_failures = [item for item in failures if item.status == "error"]
    primary = error_failures[0] if error_failures else failures[0]
    if len(failures) < 3:
        overall_status = "partial"
    else:
        overall_status = "error" if error_failures else "unavailable"

    return NvidiaDeviceReading(
        status=overall_status,
        framebuffer_used_bytes=used_bytes,
        framebuffer_total_bytes=total_bytes,
        busy_fraction=busy_fraction,
        vendor_timestamp=vendor_timestamp,
        selected_identity=selected_identity,
        observed_uuid=canonical_observed,
        mig_mode=mig_mode,
        mig_ambiguous=False,
        shared_device_not_exclusive=True,
        reason_code=primary.reason_code,
        detail=primary.reason,
    )


def _posix_process_group_alive(pgid: int) -> bool:
    """Return True if any member of POSIX process group ``pgid`` still exists."""

    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Terminate a child process (and its POSIX process group) without zombies."""

    if os.name == "posix":
        pgid = proc.pid
        # Always signal the POSIX process group before reaping ``proc``:
        # an unreaped zombie ``proc`` pins ``pgid == proc.pid``, and surviving
        # descendants (e.g. non-exec shell children holding pipes) remain in
        # that group even if the leader has already exited.
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
        except (PermissionError, OSError):
            try:
                proc.terminate()
            except OSError:
                pass

        grace_deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
        if proc.poll() is None:
            remaining = max(0.0, grace_deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

        # Once ``proc`` is reaped, poll the process group up to the remaining
        # grace window so surviving descendants are detected without zombie
        # false positives on ``proc`` itself.
        if proc.poll() is not None:
            while _posix_process_group_alive(pgid):
                remaining = grace_deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(0.01, remaining))

        if proc.poll() is None or _posix_process_group_alive(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except OSError:
                        pass
            except (PermissionError, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass

            kill_deadline = time.monotonic() + _KILL_WAIT_SECONDS
            if proc.poll() is None:
                remaining = max(0.0, kill_deadline - time.monotonic())
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass

            while _posix_process_group_alive(pgid):
                remaining = kill_deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(0.01, remaining))
        return

    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass

    try:
        proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        proc.kill()
    except OSError:
        pass

    try:
        proc.wait(timeout=_KILL_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _run_bounded_command(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    max_output_bytes: int,
    cancel_event: threading.Event | None = None,
    deadline_ns: int | None = None,
    env: Mapping[str, str] | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> NvidiaDeviceCommandResult:
    """Run one bounded, non-shell subprocess with live byte/time/cancel limits."""

    now_ns = int(monotonic_ns())
    probe_deadline_ns = now_ns + int(round(timeout_seconds * _NS_PER_SECOND))
    effective_deadline_ns = (
        min(probe_deadline_ns, int(deadline_ns))
        if deadline_ns is not None
        else probe_deadline_ns
    )

    if cancel_event is not None and cancel_event.is_set():
        return NvidiaDeviceCommandResult(
            returncode=None,
            stdout="",
            stderr="",
            cancelled=True,
            error_reason_code="read_cancelled",
            error_detail="nvidia-smi read cancelled before subprocess launch",
        )
    if now_ns >= effective_deadline_ns:
        return NvidiaDeviceCommandResult(
            returncode=None,
            stdout="",
            stderr="",
            timed_out=True,
            error_reason_code="read_timeout",
            error_detail="nvidia-smi read deadline expired before subprocess launch",
        )

    popen_kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": False,
        "close_fds": True,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    if env is not None:
        popen_kwargs["env"] = dict(env)

    try:
        proc: subprocess.Popen[bytes] = subprocess.Popen(list(argv), **popen_kwargs)
    except FileNotFoundError as exc:
        return NvidiaDeviceCommandResult(
            returncode=None,
            stdout="",
            stderr="",
            error_reason_code="executable_unavailable",
            error_detail=_sanitize_detail(str(exc) or "executable not found"),
        )
    except PermissionError as exc:
        return NvidiaDeviceCommandResult(
            returncode=None,
            stdout="",
            stderr="",
            error_reason_code="permission_denied",
            error_detail=_sanitize_detail(str(exc) or "permission denied"),
        )
    except OSError as exc:
        return NvidiaDeviceCommandResult(
            returncode=None,
            stdout="",
            stderr="",
            error_reason_code="command_launch_failed",
            error_detail=_sanitize_detail(f"{type(exc).__name__}: {exc}"),
        )

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    stdout_len = 0
    stderr_len = 0
    timed_out = False
    cancelled = False
    output_limit_exceeded = False
    terminated = False
    completed_normally = False

    selector = selectors.DefaultSelector()
    try:
        assert proc.stdout is not None
        assert proc.stderr is not None
        if os.name == "posix":
            os.set_blocking(proc.stdout.fileno(), False)
            os.set_blocking(proc.stderr.fileno(), False)
        selector.register(proc.stdout, selectors.EVENT_READ, data="stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, data="stderr")

        while selector.get_map():
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                _terminate_process_group(proc)
                terminated = True
                break
            remaining_ns = effective_deadline_ns - int(monotonic_ns())
            if remaining_ns <= 0:
                timed_out = True
                _terminate_process_group(proc)
                terminated = True
                break

            wait_s = min(_POLL_SLICE_SECONDS, remaining_ns / _NS_PER_SECOND)
            events = selector.select(timeout=wait_s)
            for key, _ in events:
                stream = key.fileobj
                tag = key.data
                try:
                    chunk = os.read(stream.fileno(), _READ_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""

                if not chunk:
                    selector.unregister(stream)
                    try:
                        stream.close()
                    except OSError:
                        pass
                    continue

                if tag == "stdout":
                    stdout_len += len(chunk)
                    if stdout_len > max_output_bytes:
                        output_limit_exceeded = True
                        _terminate_process_group(proc)
                        terminated = True
                        break
                    stdout_chunks.append(chunk)
                else:
                    stderr_len += len(chunk)
                    if stderr_len > max_output_bytes:
                        output_limit_exceeded = True
                        _terminate_process_group(proc)
                        terminated = True
                        break
                    stderr_chunks.append(chunk)

            if output_limit_exceeded:
                break

        if not (cancelled or timed_out or output_limit_exceeded):
            while proc.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    _terminate_process_group(proc)
                    terminated = True
                    break
                remaining_ns = effective_deadline_ns - int(monotonic_ns())
                if remaining_ns <= 0:
                    timed_out = True
                    _terminate_process_group(proc)
                    terminated = True
                    break
                try:
                    proc.wait(
                        timeout=min(_POLL_SLICE_SECONDS, remaining_ns / _NS_PER_SECOND)
                    )
                except subprocess.TimeoutExpired:
                    continue
            else:
                completed_normally = True
    finally:
        try:
            selector.close()
        except Exception:
            pass
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if not completed_normally and not terminated:
            _terminate_process_group(proc)

    stdout_text = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

    if cancelled:
        return NvidiaDeviceCommandResult(
            returncode=None,
            stdout=stdout_text,
            stderr=stderr_text,
            cancelled=True,
            error_reason_code="read_cancelled",
            error_detail="nvidia-smi read cancelled during execution",
        )
    if timed_out:
        return NvidiaDeviceCommandResult(
            returncode=None,
            stdout=stdout_text,
            stderr=stderr_text,
            timed_out=True,
            error_reason_code="read_timeout",
            error_detail=f"nvidia-smi timed out after bounded limit ({timeout_seconds} s)",
        )
    if output_limit_exceeded:
        return NvidiaDeviceCommandResult(
            returncode=None,
            stdout=stdout_text,
            stderr=stderr_text,
            output_limit_exceeded=True,
            error_reason_code="input_limit_exceeded",
            error_detail=f"nvidia-smi output exceeded max_output_bytes ({max_output_bytes})",
        )

    return NvidiaDeviceCommandResult(
        returncode=proc.returncode,
        stdout=stdout_text,
        stderr=stderr_text,
    )


def _classify_nonzero_exit(result: NvidiaDeviceCommandResult) -> tuple[str, str]:
    combined = f"{result.stderr}\n{result.stdout}".lower()
    snippet = _sanitize_detail(result.stderr or result.stdout)
    detail = (
        f"nvidia-smi exited with code {result.returncode}: {snippet}"
        if snippet
        else f"nvidia-smi exited with code {result.returncode}"
    )
    if any(
        needle in combined
        for needle in (
            "insufficient permissions",
            "permission denied",
            "not permitted",
            "access denied",
        )
    ):
        return "permission_denied", detail
    if any(
        needle in combined
        for needle in (
            "no devices were found",
            "not found",
            "unable to determine the device handle",
            "no matching",
            "invalid gpu",
        )
    ):
        return "device_not_found", detail
    return "command_failed", detail


class NvidiaSmiDeviceProbe:
    """Explicit, bounded ``nvidia-smi`` probe for one selected physical CUDA GPU.

    Construction is strictly pure and Torch-free: it performs zero subprocess
    launches, zero Torch imports, and zero filesystem lookups. All executable
    resolution and bounded subprocess invocation happen strictly inside
    :meth:`read`.

    Parameters
    ----------
    identity : AcceleratorIdentity or None, optional
        Already-selected physical accelerator identity. ``None``, non-CUDA
        identities, missing UUIDs, MIG UUIDs, and malformed UUIDs fail closed
        without falling back to GPU 0 or unrelated GPUs.
    executable : str or pathlib.Path, optional
        Explicit ``nvidia-smi`` command name or executable path
        (default ``"nvidia-smi"``). Never installs or downloads executables.
    timeout_seconds : float, optional
        Maximum execution time per read in seconds (default ``5.0``; hard
        maximum ``10.0``).
    max_output_bytes : int, optional
        Maximum stdout or stderr bytes buffered per read (default ``65536``).
    command_runner : callable or None, optional
        Optional injectable runner seam for deterministic tests. Called as
        ``command_runner(argv, timeout_seconds=..., max_output_bytes=...,
        cancel_event=..., deadline_ns=...)`` and expected to return
        :class:`NvidiaDeviceCommandResult`.
    env : Mapping[str, str] or None, optional
        Optional environment mapping override for deterministic
        ``CUDA_VISIBLE_DEVICES`` / ``PATH`` tests (defaults to ``os.environ``).
    monotonic_ns : callable or None, optional
        Optional monotonic nanosecond clock override (defaults to
        ``time.monotonic_ns``).
    """

    def __init__(
        self,
        identity: AcceleratorIdentity | None = None,
        *,
        executable: str | Path = DEFAULT_NVIDIA_SMI_EXECUTABLE,
        timeout_seconds: float = DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_NVIDIA_SMI_MAX_OUTPUT_BYTES,
        command_runner: Callable[..., NvidiaDeviceCommandResult] | None = None,
        env: Mapping[str, str] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        if identity is not None:
            accel = _accel()
            if type(identity) is not accel.AcceleratorIdentity:
                raise TypeError(
                    "NvidiaSmiDeviceProbe.identity must be AcceleratorIdentity or None, "
                    f"got {type(identity).__name__}"
                )
        self._identity = identity
        self._executable = _validate_executable(
            executable, "NvidiaSmiDeviceProbe.executable"
        )
        self._timeout_seconds = _require_timeout_seconds(
            timeout_seconds, "NvidiaSmiDeviceProbe.timeout_seconds"
        )
        self._max_output_bytes = _require_positive_int(
            max_output_bytes, "NvidiaSmiDeviceProbe.max_output_bytes"
        )
        if command_runner is not None and not callable(command_runner):
            raise TypeError(
                "NvidiaSmiDeviceProbe.command_runner must be callable or None"
            )
        self._command_runner = command_runner

        if env is not None:
            if not isinstance(env, Mapping):
                raise TypeError(
                    f"NvidiaSmiDeviceProbe.env must be a Mapping or None, got {type(env).__name__}"
                )
            copied_env: dict[str, str] = {}
            for key, val in env.items():
                if type(key) is not str or type(val) is not str:
                    raise TypeError(
                        "NvidiaSmiDeviceProbe.env keys and values must be strings"
                    )
                copied_env[key] = val
            self._env: dict[str, str] | None = copied_env
        else:
            self._env = None

        if monotonic_ns is not None and not callable(monotonic_ns):
            raise TypeError(
                "NvidiaSmiDeviceProbe.monotonic_ns must be callable or None"
            )
        self._monotonic_ns: Callable[[], int] = (
            time.monotonic_ns if monotonic_ns is None else monotonic_ns
        )

    @property
    def identity(self) -> AcceleratorIdentity | None:
        """Selected physical accelerator identity bound to this probe."""

        return self._identity

    @property
    def executable(self) -> str:
        """Configured ``nvidia-smi`` command name or executable path."""

        return self._executable

    @property
    def timeout_seconds(self) -> float:
        """Per-read bounded execution timeout in seconds (``<= 10.0``)."""

        return self._timeout_seconds

    @property
    def max_output_bytes(self) -> int:
        """Maximum buffered stdout/stderr bytes per read."""

        return self._max_output_bytes

    def _effective_env(self) -> Mapping[str, str]:
        return os.environ if self._env is None else self._env

    def _check_cuda_visible_devices_mig(self) -> NvidiaDeviceUnavailable | None:
        identity = self._identity
        if identity is None:
            return None
        raw_cvd = self._effective_env().get("CUDA_VISIBLE_DEVICES")
        if not isinstance(raw_cvd, str) or not raw_cvd.strip():
            return None
        tokens = [tok.strip() for tok in raw_cvd.split(",")]
        if identity.index is not None:
            if 0 <= identity.index < len(tokens):
                selected_token = tokens[identity.index]
                if selected_token.upper().startswith("MIG-"):
                    return NvidiaDeviceUnavailable(
                        reason=(
                            f"CUDA_VISIBLE_DEVICES entry {identity.index} "
                            f"({selected_token!r}) selects a MIG subdevice; "
                            "whole-GPU nvidia-smi metrics are ambiguous"
                        ),
                        reason_code="mig_subdevice_ambiguous",
                        status="unavailable",
                    )
        else:
            for token in tokens:
                if token and token.upper().startswith("MIG-"):
                    return NvidiaDeviceUnavailable(
                        reason=(
                            f"CUDA_VISIBLE_DEVICES contains MIG subdevice {token!r} "
                            "and selected device index is unknown; whole-GPU "
                            "nvidia-smi metrics are ambiguous"
                        ),
                        reason_code="mig_subdevice_ambiguous",
                        status="unavailable",
                    )
        return None

    def _resolve_executable(self) -> str | NvidiaDeviceUnavailable:
        exe = self._executable
        has_sep = os.sep in exe or (os.altsep is not None and os.altsep in exe)
        if has_sep:
            if os.path.isfile(exe) and os.access(exe, os.X_OK):
                return exe
            return NvidiaDeviceUnavailable(
                reason=f"configured nvidia-smi executable {exe!r} is unavailable or not executable",
                reason_code="executable_unavailable",
                status="unavailable",
            )
        search_path = self._env.get("PATH", "") if self._env is not None else None
        resolved = shutil.which(exe, path=search_path)
        if resolved is not None and os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return resolved
        return NvidiaDeviceUnavailable(
            reason=f"nvidia-smi executable {exe!r} was not found on PATH",
            reason_code="executable_unavailable",
            status="unavailable",
        )

    def _validate_target_and_canonical_uuid(
        self,
    ) -> tuple[str | None, NvidiaDeviceUnavailable | None, bool]:
        """Validate selected identity, UUID, and MIG env without spawning."""

        identity = self._identity
        if identity is None:
            return (
                None,
                NvidiaDeviceUnavailable(
                    reason="no selected physical accelerator identity is configured",
                    reason_code="no_selected_identity",
                    status="unavailable",
                ),
                False,
            )

        accel = _accel()
        if identity.kind is not accel.AcceleratorKind.CUDA:
            return (
                None,
                NvidiaDeviceUnavailable(
                    reason=(
                        f"nvidia-smi only supports CUDA accelerators, got "
                        f"{identity.kind.value!r}"
                    ),
                    reason_code="non_cuda_accelerator",
                    status="unavailable",
                ),
                False,
            )

        if identity.uuid is None or (
            isinstance(identity.uuid, str) and not identity.uuid
        ):
            return (
                None,
                NvidiaDeviceUnavailable(
                    reason=(
                        "selected CUDA accelerator has no physical device UUID; "
                        "refusing fallback to unrelated GPU indices"
                    ),
                    reason_code="missing_device_uuid",
                    status="unavailable",
                ),
                False,
            )

        canonical = canonicalize_nvidia_device_uuid(identity.uuid)
        if isinstance(canonical, NvidiaDeviceUnavailable):
            return (
                None,
                canonical,
                canonical.reason_code == "mig_subdevice_ambiguous",
            )

        mig_env_issue = self._check_cuda_visible_devices_mig()
        if mig_env_issue is not None:
            return None, mig_env_issue, True

        return canonical, None, False

    def preflight_issue(self) -> NvidiaDeviceUnavailable | None:
        """Return any static binding/executable issue without invoking ``nvidia-smi``."""

        _, issue, _ = self._validate_target_and_canonical_uuid()
        if issue is not None:
            return issue
        if self._command_runner is None:
            resolved = self._resolve_executable()
            if isinstance(resolved, NvidiaDeviceUnavailable):
                return resolved
        return None

    def read(
        self,
        *,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> NvidiaDeviceReading:
        """Capture one bounded physical-GPU reading for the selected CUDA device.

        Parameters
        ----------
        cancel_event : threading.Event or None, optional
            Optional cancellation event checked before launch and during
            command execution so scope close terminates in-flight reads promptly.
        deadline_ns : int or None, optional
            Optional absolute monotonic nanosecond deadline capping this read
            together with ``timeout_seconds``.

        Returns
        -------
        NvidiaDeviceReading
            Parsed reading or fail-closed typed unavailability/error reading.
        """

        canonical_uuid, binding_issue, mig_ambiguous = (
            self._validate_target_and_canonical_uuid()
        )
        if binding_issue is not None or canonical_uuid is None:
            assert binding_issue is not None
            return _uniform_reading(
                issue=binding_issue,
                selected_identity=self._identity,
                mig_mode="enabled" if mig_ambiguous else "unknown",
                mig_ambiguous=mig_ambiguous,
            )

        if self._command_runner is None:
            resolved_exe = self._resolve_executable()
            if isinstance(resolved_exe, NvidiaDeviceUnavailable):
                return _uniform_reading(
                    issue=resolved_exe,
                    selected_identity=self._identity,
                )
            exe_token = resolved_exe
        else:
            exe_token = self._executable

        argv = [
            exe_token,
            "-i",
            canonical_uuid,
            f"--query-gpu={','.join(NVIDIA_SMI_QUERY_FIELDS)}",
            f"--format={NVIDIA_SMI_QUERY_FORMAT}",
        ]

        if self._command_runner is None:
            cmd_result = _run_bounded_command(
                argv,
                timeout_seconds=self._timeout_seconds,
                max_output_bytes=self._max_output_bytes,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
                env=self._env,
                monotonic_ns=self._monotonic_ns,
            )
        else:
            try:
                cmd_result = self._command_runner(
                    argv,
                    timeout_seconds=self._timeout_seconds,
                    max_output_bytes=self._max_output_bytes,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
            except FileNotFoundError as exc:
                return _uniform_reading(
                    issue=NvidiaDeviceUnavailable(
                        reason=_sanitize_detail(str(exc) or "executable not found"),
                        reason_code="executable_unavailable",
                        status="unavailable",
                    ),
                    selected_identity=self._identity,
                )
            except PermissionError as exc:
                return _uniform_reading(
                    issue=NvidiaDeviceUnavailable(
                        reason=_sanitize_detail(str(exc) or "permission denied"),
                        reason_code="permission_denied",
                        status="error",
                    ),
                    selected_identity=self._identity,
                )
            except Exception as exc:
                return _uniform_reading(
                    issue=NvidiaDeviceUnavailable(
                        reason=_sanitize_detail(f"{type(exc).__name__}: {exc}"),
                        reason_code="command_failed",
                        status="error",
                    ),
                    selected_identity=self._identity,
                )

        if not isinstance(cmd_result, NvidiaDeviceCommandResult):
            return _uniform_reading(
                issue=NvidiaDeviceUnavailable(
                    reason=(
                        "command_runner must return NvidiaDeviceCommandResult, "
                        f"got {type(cmd_result).__name__}"
                    ),
                    reason_code="command_failed",
                    status="error",
                ),
                selected_identity=self._identity,
            )

        if cmd_result.cancelled:
            return _uniform_reading(
                issue=NvidiaDeviceUnavailable(
                    reason=cmd_result.error_detail or "nvidia-smi read cancelled",
                    reason_code=cmd_result.error_reason_code or "read_cancelled",
                    status="unavailable",
                ),
                selected_identity=self._identity,
            )
        if cmd_result.timed_out:
            return _uniform_reading(
                issue=NvidiaDeviceUnavailable(
                    reason=(
                        cmd_result.error_detail
                        or f"nvidia-smi timed out after {self._timeout_seconds} s"
                    ),
                    reason_code=cmd_result.error_reason_code or "read_timeout",
                    status="error",
                ),
                selected_identity=self._identity,
            )
        if cmd_result.output_limit_exceeded:
            return _uniform_reading(
                issue=NvidiaDeviceUnavailable(
                    reason=(
                        cmd_result.error_detail
                        or f"nvidia-smi output exceeded {self._max_output_bytes} bytes"
                    ),
                    reason_code=cmd_result.error_reason_code or "input_limit_exceeded",
                    status="error",
                ),
                selected_identity=self._identity,
            )
        if cmd_result.returncode is None:
            reason_code = cmd_result.error_reason_code or "command_failed"
            status = (
                "unavailable"
                if reason_code in {"executable_unavailable", "read_cancelled"}
                else "error"
            )
            return _uniform_reading(
                issue=NvidiaDeviceUnavailable(
                    reason=cmd_result.error_detail or "nvidia-smi command did not complete",
                    reason_code=reason_code,
                    status=status,
                ),
                selected_identity=self._identity,
            )
        if cmd_result.returncode != 0:
            reason_code, detail = _classify_nonzero_exit(cmd_result)
            return _uniform_reading(
                issue=NvidiaDeviceUnavailable(
                    reason=detail,
                    reason_code=reason_code,
                    status="error",
                ),
                selected_identity=self._identity,
            )

        return parse_nvidia_smi_query_output(
            cmd_result.stdout,
            expected_uuid=canonical_uuid,
            selected_identity=self._identity,
        )


__all__ = [
    "DEFAULT_NVIDIA_SMI_EXECUTABLE",
    "DEFAULT_NVIDIA_SMI_MAX_OUTPUT_BYTES",
    "DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS",
    "MAX_NVIDIA_SMI_TIMEOUT_SECONDS",
    "NVIDIA_DEVICE_MIG_MODES",
    "NVIDIA_DEVICE_READING_STATUSES",
    "NVIDIA_SMI_QUERY_FIELDS",
    "NVIDIA_SMI_QUERY_FORMAT",
    "NvidiaDeviceCommandResult",
    "NvidiaDeviceReading",
    "NvidiaDeviceUnavailable",
    "NvidiaSmiDeviceProbe",
    "canonicalize_nvidia_device_uuid",
    "parse_nvidia_smi_query_output",
]
