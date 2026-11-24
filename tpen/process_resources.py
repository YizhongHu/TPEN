"""Typed process resource readings, stdlib ``resource`` probe, and Linux procfs memory probe.

Linux and macOS expose the seven RUSAGE fields used here through ``getrusage``
(``getrusage(2)`` on `man7.org <https://man7.org/linux/man-pages/man2/getrusage.2.html>`_
and the Apple `getrusage(2) <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/getrusage.2.html>`_).
Their values may legitimately be zero; ``ProcessRUsageProbe`` therefore treats
only a failed process-level read as unavailable evidence.

On Linux, :class:`LinuxProcessMemoryProbe` reads current process memory gauges
(``VmRSS``, ``VmSize``, ``Threads``) from ``/proc/<pid>/status`` while
bracketing every read with exact target-generation verification from
``/proc/<pid>/stat``, ``/proc/sys/kernel/random/boot_id``, and
``/proc/<pid>/ns/pid`` to defend against PID reuse. This module is strictly
standard-library and Torch-free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import math
import os
from pathlib import Path
import resource
import socket
import stat
import sys
import time
from typing import Any, Union


_BYTES_PER_KIB = 1024
_BYTES_PER_MIB = 1024 * 1024
_NS_PER_SECOND = 1_000_000_000
_READ_CHUNK_BYTES = 16384

DEFAULT_PROC_ROOT = Path("/proc")
DEFAULT_LINUX_MEMORY_MAX_READ_BYTES = 65536
DEFAULT_LINUX_MEMORY_MAX_READ_SECONDS = 1.0
LINUX_MEMORY_READING_STATUSES: frozenset[str] = frozenset(
    {"ok", "partial", "unavailable", "error"}
)
_FIELD_ISSUE_STATUSES: frozenset[str] = frozenset({"unavailable", "error"})


class ResourceScope(Enum):
    """Resource scope supported by this layer."""

    PROCESS = "process"


@dataclass(frozen=True)
class ResourceUnavailable:
    """Evidence that one process resource counter could not be read."""

    reason: str


ResourceReading = Union[float, int, ResourceUnavailable]


@dataclass(frozen=True)
class ProcessResourceBaseline:
    """Process resource counters captured at run start."""

    user_cpu_seconds: ResourceReading
    system_cpu_seconds: ResourceReading
    read_block_operations: ResourceReading
    write_block_operations: ResourceReading
    voluntary_context_switches: ResourceReading
    involuntary_context_switches: ResourceReading
    peak_rss_mb: ResourceReading


@dataclass(frozen=True)
class ProcessResourceResult:
    """Process resource counters consumed for one run."""

    user_cpu_seconds: ResourceReading
    system_cpu_seconds: ResourceReading
    read_block_operations: ResourceReading
    write_block_operations: ResourceReading
    voluntary_context_switches: ResourceReading
    involuntary_context_switches: ResourceReading
    peak_rss_mb: ResourceReading


def _peak_rss_mb(value: ResourceReading) -> ResourceReading:
    """Normalize ``ru_maxrss`` to MiB on both supported Unix conventions."""

    if isinstance(value, ResourceUnavailable):
        return value
    divisor = _BYTES_PER_MIB if sys.platform == "darwin" else 1024
    return float(value) / divisor


def _subtract(current: ResourceReading, baseline: ResourceReading) -> ResourceReading:
    """Subtract counters while retaining an unavailable operand as evidence."""

    if isinstance(current, ResourceUnavailable):
        return current
    if isinstance(baseline, ResourceUnavailable):
        return baseline
    return current - baseline


class ProcessRUsageProbe:
    """Capture process counters from :func:`resource.getrusage`.

    Parameters
    ----------
    scope : ResourceScope, optional
        Only ``ResourceScope.PROCESS`` is supported in this layer.
    """

    def __init__(self, scope: ResourceScope = ResourceScope.PROCESS) -> None:
        if scope is not ResourceScope.PROCESS:
            raise ValueError("ProcessRUsageProbe supports only ResourceScope.PROCESS")
        self.scope = scope

    def read(self) -> ProcessResourceBaseline:
        """Read the current process usage as a baseline-shaped record."""

        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
        except OSError as exc:
            unavailable = ResourceUnavailable(f"{type(exc).__name__}: {exc}")
            return ProcessResourceBaseline(
                user_cpu_seconds=unavailable,
                system_cpu_seconds=unavailable,
                read_block_operations=unavailable,
                write_block_operations=unavailable,
                voluntary_context_switches=unavailable,
                involuntary_context_switches=unavailable,
                peak_rss_mb=unavailable,
            )
        return ProcessResourceBaseline(
            user_cpu_seconds=usage.ru_utime,
            system_cpu_seconds=usage.ru_stime,
            read_block_operations=usage.ru_inblock,
            write_block_operations=usage.ru_oublock,
            voluntary_context_switches=usage.ru_nvcsw,
            involuntary_context_switches=usage.ru_nivcsw,
            peak_rss_mb=_peak_rss_mb(usage.ru_maxrss),
        )

    def result(self, baseline: ProcessResourceBaseline) -> ProcessResourceResult:
        """Read terminal counters and calculate run deltas."""

        current = self.read()
        return ProcessResourceResult(
            user_cpu_seconds=_subtract(current.user_cpu_seconds, baseline.user_cpu_seconds),
            system_cpu_seconds=_subtract(current.system_cpu_seconds, baseline.system_cpu_seconds),
            read_block_operations=_subtract(current.read_block_operations, baseline.read_block_operations),
            write_block_operations=_subtract(current.write_block_operations, baseline.write_block_operations),
            voluntary_context_switches=_subtract(
                current.voluntary_context_switches, baseline.voluntary_context_switches
            ),
            involuntary_context_switches=_subtract(
                current.involuntary_context_switches, baseline.involuntary_context_switches
            ),
            peak_rss_mb=current.peak_rss_mb,
        )


# ---------------------------------------------------------------------------
# Linux exact-generation current-memory probe (procfs, strictly stdlib)
# ---------------------------------------------------------------------------


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _require_non_negative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _require_positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a finite positive number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return number


def _require_optional_nonempty_str(value: object, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{name} must be a string or None, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must be a nonempty string or None")
    return stripped


def _is_strict_non_negative_ascii_int(token: str) -> bool:
    """Return True iff ``token`` is a strict ASCII base-10 non-negative integer."""

    return bool(token) and token.isascii() and token.isdigit()


@dataclass(frozen=True)
class LinuxMemoryUnavailable(ResourceUnavailable):
    """Typed unavailability or error for one Linux procfs memory field.

    Subclasses :class:`ResourceUnavailable` so callers checking
    ``isinstance(value, ResourceUnavailable)`` handle it uniformly, while
    preserving structured ``status`` (``"unavailable"`` or ``"error"``) and
    machine-readable ``reason_code`` for ``tpen.run-observability/v1`` records.

    Parameters
    ----------
    reason : str
        Bounded human-readable diagnostic detail.
    reason_code : str, optional
        Machine-readable unavailability/error code (for example
        ``"field_missing"``, ``"generation_changed"``, ``"malformed_procfs"``,
        ``"input_limit_exceeded"``, ``"read_timeout"``, ``"permission_denied"``,
        ``"process_exited"``, ``"procfs_missing"``, or ``"unsupported_platform"``).
    status : {"unavailable", "error"}, optional
        Tagged reading status for this field issue.
    """

    reason_code: str = "unavailable"
    status: str = "unavailable"

    def __post_init__(self) -> None:
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("LinuxMemoryUnavailable.reason must be a nonempty string")
        if type(self.reason_code) is not str or not self.reason_code.strip():
            raise ValueError(
                "LinuxMemoryUnavailable.reason_code must be a nonempty string"
            )
        if type(self.status) is not str or self.status not in _FIELD_ISSUE_STATUSES:
            raise ValueError(
                f"LinuxMemoryUnavailable.status must be one of "
                f"{sorted(_FIELD_ISSUE_STATUSES)}, got {self.status!r}"
            )


LinuxMemoryFieldValue = Union[int, ResourceUnavailable]


@dataclass(frozen=True)
class LinuxProcessGeneration:
    """Exact Linux process generation identity defending against PID reuse.

    Parameters
    ----------
    pid : int
        Positive Linux process identifier.
    start_ticks : int
        Process start time since boot in clock ticks (field 22 of
        ``/proc/<pid>/stat``).
    boot_id : str or None, optional
        Kernel boot UUID from ``/proc/sys/kernel/random/boot_id`` when readable.
    pid_ns_inode : str or None, optional
        PID namespace symlink target from ``/proc/<pid>/ns/pid`` when readable.
    host_id : str or None, optional
        Optional host identifier associated with this process generation.
    """

    pid: int
    start_ticks: int
    boot_id: str | None = None
    pid_ns_inode: str | None = None
    host_id: str | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.pid, "LinuxProcessGeneration.pid")
        _require_non_negative_int(
            self.start_ticks, "LinuxProcessGeneration.start_ticks"
        )
        object.__setattr__(
            self,
            "boot_id",
            _require_optional_nonempty_str(
                self.boot_id, "LinuxProcessGeneration.boot_id"
            ),
        )
        object.__setattr__(
            self,
            "pid_ns_inode",
            _require_optional_nonempty_str(
                self.pid_ns_inode, "LinuxProcessGeneration.pid_ns_inode"
            ),
        )
        object.__setattr__(
            self,
            "host_id",
            _require_optional_nonempty_str(
                self.host_id, "LinuxProcessGeneration.host_id"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary describing this process generation."""

        data: dict[str, Any] = {
            "pid": self.pid,
            "start_ticks": self.start_ticks,
        }
        if self.boot_id is not None:
            data["boot_id"] = self.boot_id
        if self.pid_ns_inode is not None:
            data["pid_ns_inode"] = self.pid_ns_inode
        if self.host_id is not None:
            data["host_id"] = self.host_id
        return data

    def to_subject_dict(
        self,
        *,
        host_id: str | None = None,
        invocation_token: str | None = None,
    ) -> dict[str, Any]:
        """Return an O2 ``tpen.run-observability/v1`` ``process`` subject mapping."""

        subject: dict[str, Any] = {
            "kind": "process",
            "pid": self.pid,
            "start_ticks": self.start_ticks,
        }
        effective_host = self.host_id if self.host_id is not None else host_id
        if effective_host is not None:
            subject["host_id"] = effective_host
        if self.boot_id is not None:
            subject["boot_id"] = self.boot_id
        if self.pid_ns_inode is not None:
            subject["pid_ns_inode"] = self.pid_ns_inode
        if invocation_token is not None:
            subject["invocation_token"] = invocation_token
        return subject


@dataclass(frozen=True)
class LinuxProcessMemoryReading:
    """Typed Linux self-process current-memory reading with generation proof.

    Explicitly separates valid observations (including legitimate ``0`` values)
    from partial, unavailable, and error outcomes so callers never confuse
    missing/unverified procfs fields with zero memory usage.

    Parameters
    ----------
    status : {"ok", "partial", "unavailable", "error"}
        Overall reading status.
    rss_bytes : int or ResourceUnavailable
        Current resident set size in bytes (``VmRSS`` KiB * 1024) or typed
        unavailability/error.
    virtual_bytes : int or ResourceUnavailable
        Current virtual memory size in bytes (``VmSize`` KiB * 1024) or typed
        unavailability/error.
    thread_count : int or ResourceUnavailable
        Current process thread count (``Threads``) or typed unavailability/error.
    read_start_ns : int
        Monotonic nanosecond timestamp captured before the pre-read check.
    read_end_ns : int
        Monotonic nanosecond timestamp captured after the post-read check.
    reason_code : str or None, optional
        Machine-readable summary reason code when ``status != "ok"``.
    detail : str or None, optional
        Bounded diagnostic detail when ``status != "ok"``.
    target_generation : LinuxProcessGeneration or None, optional
        Bound target generation expected for this read.
    observed_generation_before : LinuxProcessGeneration or None, optional
        Generation observed immediately prior to reading ``/proc/<pid>/status``.
    observed_generation_after : LinuxProcessGeneration or None, optional
        Generation observed immediately after reading ``/proc/<pid>/status``.
    """

    status: str
    rss_bytes: LinuxMemoryFieldValue
    virtual_bytes: LinuxMemoryFieldValue
    thread_count: LinuxMemoryFieldValue
    read_start_ns: int
    read_end_ns: int
    reason_code: str | None = None
    detail: str | None = None
    target_generation: LinuxProcessGeneration | None = None
    observed_generation_before: LinuxProcessGeneration | None = None
    observed_generation_after: LinuxProcessGeneration | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in LINUX_MEMORY_READING_STATUSES:
            raise ValueError(
                f"LinuxProcessMemoryReading.status must be one of "
                f"{sorted(LINUX_MEMORY_READING_STATUSES)}, got {self.status!r}"
            )
        start = _require_non_negative_int(
            self.read_start_ns, "LinuxProcessMemoryReading.read_start_ns"
        )
        end = _require_non_negative_int(
            self.read_end_ns, "LinuxProcessMemoryReading.read_end_ns"
        )
        if end < start:
            raise ValueError(
                f"LinuxProcessMemoryReading.read_end_ns ({end}) must be >= "
                f"read_start_ns ({start})"
            )
        fields = (
            ("rss_bytes", self.rss_bytes),
            ("virtual_bytes", self.virtual_bytes),
            ("thread_count", self.thread_count),
        )
        ok_count = 0
        for field_name, val in fields:
            if isinstance(val, ResourceUnavailable):
                continue
            if type(val) is not int or val < 0:
                raise ValueError(
                    f"LinuxProcessMemoryReading.{field_name} must be a non-negative "
                    f"int or ResourceUnavailable, got {val!r}"
                )
            ok_count += 1

        if self.status == "ok":
            if ok_count != 3:
                raise ValueError(
                    "LinuxProcessMemoryReading with status='ok' requires all fields "
                    "to be valid non-negative integers"
                )
        elif self.status == "partial":
            if ok_count == 0 or ok_count == 3:
                raise ValueError(
                    "LinuxProcessMemoryReading with status='partial' requires a mix "
                    "of valid integer and ResourceUnavailable fields"
                )
            if type(self.reason_code) is not str or not self.reason_code.strip():
                raise ValueError(
                    "LinuxProcessMemoryReading with status='partial' requires reason_code"
                )
        else:
            if ok_count != 0:
                raise ValueError(
                    f"LinuxProcessMemoryReading with status={self.status!r} must not "
                    "carry numeric field values"
                )
            if type(self.reason_code) is not str or not self.reason_code.strip():
                raise ValueError(
                    f"LinuxProcessMemoryReading with status={self.status!r} requires reason_code"
                )

        for gen_name in (
            "target_generation",
            "observed_generation_before",
            "observed_generation_after",
        ):
            gen = getattr(self, gen_name)
            if gen is not None and not isinstance(gen, LinuxProcessGeneration):
                raise TypeError(
                    f"LinuxProcessMemoryReading.{gen_name} must be "
                    f"LinuxProcessGeneration or None, got {type(gen).__name__}"
                )


def _generation_mismatch_detail(
    expected: LinuxProcessGeneration,
    observed: LinuxProcessGeneration,
    *,
    exact_optional: bool = True,
) -> str | None:
    """Return a concise mismatch diagnostic when two generations do not match."""

    if expected.pid != observed.pid:
        return f"pid changed (expected {expected.pid}, observed {observed.pid})"
    if expected.start_ticks != observed.start_ticks:
        return (
            f"start_ticks changed (expected {expected.start_ticks}, "
            f"observed {observed.start_ticks})"
        )
    if exact_optional:
        if expected.boot_id != observed.boot_id:
            return (
                f"boot_id changed (expected {expected.boot_id!r}, "
                f"observed {observed.boot_id!r})"
            )
        if expected.pid_ns_inode != observed.pid_ns_inode:
            return (
                f"pid_ns_inode changed (expected {expected.pid_ns_inode!r}, "
                f"observed {observed.pid_ns_inode!r})"
            )
    else:
        if expected.boot_id is not None and observed.boot_id != expected.boot_id:
            return (
                f"boot_id changed (expected {expected.boot_id!r}, "
                f"observed {observed.boot_id!r})"
            )
        if (
            expected.pid_ns_inode is not None
            and observed.pid_ns_inode != expected.pid_ns_inode
        ):
            return (
                f"pid_ns_inode changed (expected {expected.pid_ns_inode!r}, "
                f"observed {observed.pid_ns_inode!r})"
            )
    if expected.host_id is not None and observed.host_id != expected.host_id:
        return (
            f"host_id changed (expected {expected.host_id!r}, "
            f"observed {observed.host_id!r})"
        )
    return None


def parse_proc_stat_start_ticks(
    raw: str | bytes,
    *,
    expected_pid: int | None = None,
) -> tuple[int, int] | LinuxMemoryUnavailable:
    """Parse ``(pid, start_ticks)`` from Linux ``/proc/<pid>/stat`` content.

    Locates the final ``')'`` via ``rfind`` so process names (``comm``) that
    contain spaces, newlines, parentheses, or arbitrary bytes do not shift
    subsequent fields. Field 22 (``starttime``) is index 19 after the final
    closing parenthesis (index 0 is field 3, ``state``).

    Parameters
    ----------
    raw : str or bytes
        Raw contents read from ``/proc/<pid>/stat``.
    expected_pid : int or None, optional
        When supplied, reject content whose parsed PID differs.

    Returns
    -------
    tuple of (int, int) or LinuxMemoryUnavailable
        ``(pid, start_ticks)`` on valid input, or a typed
        ``malformed_procfs`` error.
    """

    if isinstance(raw, bytes):
        first_lparen = raw.find(b"(")
        last_rparen = raw.rfind(b")")
        if first_lparen <= 0 or last_rparen <= first_lparen:
            return LinuxMemoryUnavailable(
                reason="malformed /proc/<pid>/stat: missing comm parentheses",
                reason_code="malformed_procfs",
                status="error",
            )
        try:
            pid_text = raw[:first_lparen].decode("ascii").strip()
            after_text = raw[last_rparen + 1 :].decode("ascii")
        except UnicodeDecodeError:
            return LinuxMemoryUnavailable(
                reason="malformed /proc/<pid>/stat: non-ASCII field outside comm",
                reason_code="malformed_procfs",
                status="error",
            )
    elif isinstance(raw, str):
        first_lparen = raw.find("(")
        last_rparen = raw.rfind(")")
        if first_lparen <= 0 or last_rparen <= first_lparen:
            return LinuxMemoryUnavailable(
                reason="malformed /proc/<pid>/stat: missing comm parentheses",
                reason_code="malformed_procfs",
                status="error",
            )
        pid_text = raw[:first_lparen].strip()
        after_text = raw[last_rparen + 1 :]
    else:
        return LinuxMemoryUnavailable(
            reason=f"malformed /proc/<pid>/stat: unexpected type {type(raw).__name__}",
            reason_code="malformed_procfs",
            status="error",
        )

    if not _is_strict_non_negative_ascii_int(pid_text):
        return LinuxMemoryUnavailable(
            reason=f"malformed /proc/<pid>/stat: invalid pid token {pid_text!r}",
            reason_code="malformed_procfs",
            status="error",
        )
    parsed_pid = int(pid_text)
    if parsed_pid < 1:
        return LinuxMemoryUnavailable(
            reason=f"malformed /proc/<pid>/stat: non-positive pid {parsed_pid}",
            reason_code="malformed_procfs",
            status="error",
        )
    if expected_pid is not None and parsed_pid != expected_pid:
        return LinuxMemoryUnavailable(
            reason=(
                f"malformed /proc/{expected_pid}/stat: pid field {parsed_pid} "
                f"does not match target {expected_pid}"
            ),
            reason_code="malformed_procfs",
            status="error",
        )

    after_fields = after_text.split()
    # Field 3 (state) is index 0; field 22 (starttime) is index 19.
    if len(after_fields) < 20:
        return LinuxMemoryUnavailable(
            reason=(
                f"malformed /proc/<pid>/stat: expected at least 20 fields after "
                f"comm, got {len(after_fields)}"
            ),
            reason_code="malformed_procfs",
            status="error",
        )
    state_token = after_fields[0]
    if len(state_token) != 1 or not state_token.isascii() or not state_token.isalpha():
        return LinuxMemoryUnavailable(
            reason=f"malformed /proc/<pid>/stat: invalid state token {state_token!r}",
            reason_code="malformed_procfs",
            status="error",
        )
    start_token = after_fields[19]
    if not _is_strict_non_negative_ascii_int(start_token):
        return LinuxMemoryUnavailable(
            reason=f"malformed /proc/<pid>/stat: invalid starttime token {start_token!r}",
            reason_code="malformed_procfs",
            status="error",
        )
    return (parsed_pid, int(start_token))


def _parse_proc_status_kib_field(key: str, raw_value: str) -> LinuxMemoryFieldValue:
    """Parse one ``<integer> kB`` field from ``/proc/<pid>/status`` into bytes."""

    parts = raw_value.strip().split()
    if len(parts) != 2:
        return LinuxMemoryUnavailable(
            reason=f"malformed {key} value in proc status: {raw_value.strip()!r}",
            reason_code="malformed_procfs",
            status="error",
        )
    amount_token, unit_token = parts
    if unit_token != "kB":
        return LinuxMemoryUnavailable(
            reason=f"unsupported {key} unit {unit_token!r} in proc status (expected 'kB')",
            reason_code="malformed_procfs",
            status="error",
        )
    if not _is_strict_non_negative_ascii_int(amount_token):
        return LinuxMemoryUnavailable(
            reason=f"invalid {key} integer value {amount_token!r} in proc status",
            reason_code="malformed_procfs",
            status="error",
        )
    return int(amount_token) * _BYTES_PER_KIB


def _parse_proc_status_threads_field(raw_value: str) -> LinuxMemoryFieldValue:
    """Parse ``Threads`` integer count from ``/proc/<pid>/status``."""

    parts = raw_value.strip().split()
    if len(parts) != 1 or not _is_strict_non_negative_ascii_int(parts[0]):
        return LinuxMemoryUnavailable(
            reason=f"malformed Threads value in proc status: {raw_value.strip()!r}",
            reason_code="malformed_procfs",
            status="error",
        )
    return int(parts[0])


def parse_proc_status_memory(
    raw: str | bytes,
) -> tuple[LinuxMemoryFieldValue, LinuxMemoryFieldValue, LinuxMemoryFieldValue] | LinuxMemoryUnavailable:
    """Parse ``(rss_bytes, virtual_bytes, thread_count)`` from ``/proc/<pid>/status``.

    Explicitly preserves ``0 kB`` / ``0`` as ``0`` and reports missing keys as
    per-field ``field_missing`` unavailability (never guessed zero). Malformed
    lines, duplicate target keys, negative/non-integer values, or non-``kB``
    units produce typed ``malformed_procfs`` errors.

    Parameters
    ----------
    raw : str or bytes
        Raw contents read from ``/proc/<pid>/status``.

    Returns
    -------
    tuple of 3 LinuxMemoryFieldValue or LinuxMemoryUnavailable
        ``(rss_bytes, virtual_bytes, thread_count)`` or a whole-file
        ``malformed_procfs`` error when the status document is empty or
        structurally corrupt.
    """

    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return LinuxMemoryUnavailable(
                reason="malformed /proc/<pid>/status: invalid UTF-8 bytes",
                reason_code="malformed_procfs",
                status="error",
            )
    elif isinstance(raw, str):
        text = raw
    else:
        return LinuxMemoryUnavailable(
            reason=f"malformed /proc/<pid>/status: unexpected type {type(raw).__name__}",
            reason_code="malformed_procfs",
            status="error",
        )

    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return LinuxMemoryUnavailable(
            reason="malformed /proc/<pid>/status: empty content",
            reason_code="malformed_procfs",
            status="error",
        )

    parsed_values: dict[str, LinuxMemoryFieldValue] = {}
    target_keys = frozenset({"VmRSS", "VmSize", "Threads"})

    for line in nonempty_lines:
        key, sep, rest = line.partition(":")
        key_clean = key.strip()
        if not sep or not key_clean:
            return LinuxMemoryUnavailable(
                reason="malformed /proc/<pid>/status: line missing 'Key: Value' format",
                reason_code="malformed_procfs",
                status="error",
            )
        if key_clean not in target_keys:
            continue
        if key_clean in parsed_values:
            parsed_values[key_clean] = LinuxMemoryUnavailable(
                reason=f"duplicate {key_clean} entry in proc status",
                reason_code="malformed_procfs",
                status="error",
            )
            continue
        if key_clean in ("VmRSS", "VmSize"):
            parsed_values[key_clean] = _parse_proc_status_kib_field(key_clean, rest)
        else:
            parsed_values[key_clean] = _parse_proc_status_threads_field(rest)

    rss_bytes = parsed_values.get(
        "VmRSS",
        LinuxMemoryUnavailable(
            reason="VmRSS key missing from /proc/<pid>/status",
            reason_code="field_missing",
            status="unavailable",
        ),
    )
    virtual_bytes = parsed_values.get(
        "VmSize",
        LinuxMemoryUnavailable(
            reason="VmSize key missing from /proc/<pid>/status",
            reason_code="field_missing",
            status="unavailable",
        ),
    )
    thread_count = parsed_values.get(
        "Threads",
        LinuxMemoryUnavailable(
            reason="Threads key missing from /proc/<pid>/status",
            reason_code="field_missing",
            status="unavailable",
        ),
    )
    return (rss_bytes, virtual_bytes, thread_count)


class LinuxProcessMemoryProbe:
    """Bounded Linux procfs current-memory probe with exact-generation defense.

    Construction is strictly pure and side-effect-free: it performs no
    filesystem I/O and never freezes ``os.getpid()`` when ``pid=None``. Call
    :meth:`bind_target` at activation (or let the first :meth:`read` bind the
    live process) to capture the target :class:`LinuxProcessGeneration`. Every
    subsequent :meth:`read` brackets ``/proc/<pid>/status`` with both pre-read
    and post-read generation checks so PID reuse or mid-read exit never returns
    stale or cross-process memory values.

    Parameters
    ----------
    pid : int or None, optional
        Optional target PID. When ``None``, resolved lazily at activation or
        first read via :func:`os.getpid`.
    expected_generation : LinuxProcessGeneration or None, optional
        Optional pre-bound target generation to verify against on every read.
    proc_root : str or pathlib.Path or None, optional
        Optional procfs root directory override for deterministic tests. When
        ``None``, defaults to ``/proc`` and enforces Linux platform checks.
    max_read_bytes : int, optional
        Maximum bytes allowed per procfs file read (default ``65536``). Files
        exceeding this bound return typed ``input_limit_exceeded`` without
        unbounded memory allocation.
    max_read_seconds : float, optional
        Maximum monotonic seconds allowed for one complete read (default
        ``1.0``). Exceeding this bound returns typed ``read_timeout``.
    monotonic_ns : callable or None, optional
        Optional nanosecond monotonic clock override for deterministic tests.
    platform_name : str or None, optional
        Optional platform override (defaults to :data:`sys.platform`).
    host_id : str or None, optional
        Optional host identifier override attached to observed generations.
    """

    def __init__(
        self,
        *,
        pid: int | None = None,
        expected_generation: LinuxProcessGeneration | None = None,
        proc_root: str | Path | None = None,
        max_read_bytes: int = DEFAULT_LINUX_MEMORY_MAX_READ_BYTES,
        max_read_seconds: float = DEFAULT_LINUX_MEMORY_MAX_READ_SECONDS,
        monotonic_ns: Callable[[], int] | None = None,
        platform_name: str | None = None,
        host_id: str | None = None,
    ) -> None:
        if pid is not None:
            pid = _require_positive_int(pid, "LinuxProcessMemoryProbe.pid")
        if expected_generation is not None:
            if not isinstance(expected_generation, LinuxProcessGeneration):
                raise TypeError(
                    "LinuxProcessMemoryProbe.expected_generation must be "
                    f"LinuxProcessGeneration or None, got {type(expected_generation).__name__}"
                )
            if pid is not None and pid != expected_generation.pid:
                raise ValueError(
                    f"LinuxProcessMemoryProbe.pid ({pid}) disagrees with "
                    f"expected_generation.pid ({expected_generation.pid})"
                )
        self._pid: int | None = pid
        self._bound_generation: LinuxProcessGeneration | None = expected_generation
        self._bind_error: LinuxMemoryUnavailable | None = None

        if proc_root is None:
            self._proc_root = DEFAULT_PROC_ROOT
            self._uses_custom_proc_root = False
        elif isinstance(proc_root, Path):
            self._proc_root = proc_root
            self._uses_custom_proc_root = True
        elif type(proc_root) is str:
            if not proc_root.strip():
                raise ValueError(
                    "LinuxProcessMemoryProbe.proc_root must be a nonempty path or None"
                )
            self._proc_root = Path(proc_root)
            self._uses_custom_proc_root = True
        else:
            raise TypeError(
                "LinuxProcessMemoryProbe.proc_root must be str, Path, or None, "
                f"got {type(proc_root).__name__}"
            )

        self._max_read_bytes = _require_positive_int(
            max_read_bytes, "LinuxProcessMemoryProbe.max_read_bytes"
        )
        self._max_read_seconds = _require_positive_finite_float(
            max_read_seconds, "LinuxProcessMemoryProbe.max_read_seconds"
        )
        self._max_read_ns = max(
            1, int(round(self._max_read_seconds * _NS_PER_SECOND))
        )
        if monotonic_ns is not None and not callable(monotonic_ns):
            raise TypeError("LinuxProcessMemoryProbe.monotonic_ns must be callable or None")
        self._monotonic_ns: Callable[[], int] = (
            time.monotonic_ns if monotonic_ns is None else monotonic_ns
        )
        self._platform_name = _require_optional_nonempty_str(
            platform_name, "LinuxProcessMemoryProbe.platform_name"
        )
        self._host_id = _require_optional_nonempty_str(
            host_id, "LinuxProcessMemoryProbe.host_id"
        )

    @property
    def target_generation(self) -> LinuxProcessGeneration | None:
        """Return the currently bound target generation, or ``None`` if unbound."""

        return self._bound_generation

    @property
    def bind_error(self) -> LinuxMemoryUnavailable | None:
        """Return the last :meth:`bind_target` unavailability/error, if any."""

        return self._bind_error

    @property
    def max_read_bytes(self) -> int:
        """Maximum bytes permitted per procfs file read."""

        return self._max_read_bytes

    @property
    def max_read_seconds(self) -> float:
        """Maximum monotonic seconds permitted per :meth:`read` call."""

        return self._max_read_seconds

    @property
    def proc_root(self) -> Path:
        """Configured procfs root directory."""

        return self._proc_root

    def _effective_platform(self) -> str:
        return sys.platform if self._platform_name is None else self._platform_name

    def _resolve_host_id(self) -> str:
        if self._host_id is not None:
            return self._host_id
        return socket.gethostname() or "unknown-host"

    def _check_platform(self) -> LinuxMemoryUnavailable | None:
        if (
            self._platform_name is None
            and self._uses_custom_proc_root
            and self._proc_root != DEFAULT_PROC_ROOT
        ):
            return None
        plat = self._effective_platform()
        if not plat.startswith("linux"):
            return LinuxMemoryUnavailable(
                reason=(
                    f"LinuxProcessMemoryProbe requires Linux procfs "
                    f"(current platform is {plat!r}; no non-Linux fallback)"
                ),
                reason_code="unsupported_platform",
                status="unavailable",
            )
        return None

    def _read_bounded_bytes(
        self,
        path: Path,
        *,
        deadline_ns: int,
        missing_is_process_exit: bool,
    ) -> bytes | LinuxMemoryUnavailable:
        """Read at most ``max_read_bytes`` from ``path`` within ``deadline_ns``."""

        try:
            # Refuse symlinks and non-regular files (FIFOs, sockets, devices)
            # before opening so special files can never bypass max_read_seconds
            # by blocking in open(2) or redirect reads outside procfs.
            st = os.lstat(path)
            if stat.S_ISLNK(st.st_mode):
                return LinuxMemoryUnavailable(
                    reason=f"refusing symlinked procfs file {path}",
                    reason_code="unsafe_procfs_file",
                    status="error",
                )
            if not stat.S_ISREG(st.st_mode):
                return LinuxMemoryUnavailable(
                    reason=f"refusing non-regular procfs file {path}",
                    reason_code="unsafe_procfs_file",
                    status="error",
                )
            with path.open("rb") as handle:
                fileno_fn = getattr(handle, "fileno", None)
                if callable(fileno_fn):
                    try:
                        fd = fileno_fn()
                    except (OSError, ValueError):
                        fd = None
                    if isinstance(fd, int):
                        fst = os.fstat(fd)
                        if not stat.S_ISREG(fst.st_mode):
                            return LinuxMemoryUnavailable(
                                reason=f"refusing non-regular opened procfs file {path}",
                                reason_code="unsafe_procfs_file",
                                status="error",
                            )
                chunks: list[bytes] = []
                total = 0
                limit = self._max_read_bytes + 1
                while total < limit:
                    if total >= _READ_CHUNK_BYTES and int(self._monotonic_ns()) > deadline_ns:
                        return LinuxMemoryUnavailable(
                            reason=(
                                f"reading {path} exceeded max_read_seconds "
                                f"({self._max_read_seconds}s)"
                            ),
                            reason_code="read_timeout",
                            status="error",
                        )
                    chunk = handle.read(min(limit - total, _READ_CHUNK_BYTES))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
        except PermissionError as exc:
            return LinuxMemoryUnavailable(
                reason=f"permission denied reading {path}: {exc}",
                reason_code="permission_denied",
                status="unavailable",
            )
        except (FileNotFoundError, ProcessLookupError) as exc:
            if not self._proc_root.exists():
                return LinuxMemoryUnavailable(
                    reason=f"procfs root {self._proc_root} is missing: {exc}",
                    reason_code="procfs_missing",
                    status="unavailable",
                )
            reason_code = "process_exited" if missing_is_process_exit else "procfs_missing"
            return LinuxMemoryUnavailable(
                reason=f"procfs path {path} not found: {exc}",
                reason_code=reason_code,
                status="unavailable",
            )
        except OSError as exc:
            return LinuxMemoryUnavailable(
                reason=f"OS error reading {path}: {type(exc).__name__}: {exc}",
                reason_code="procfs_read_error",
                status="error",
            )

        if total > self._max_read_bytes:
            return LinuxMemoryUnavailable(
                reason=(
                    f"procfs file {path} exceeded max_read_bytes "
                    f"({self._max_read_bytes})"
                ),
                reason_code="input_limit_exceeded",
                status="error",
            )
        return b"".join(chunks)

    def _read_generation(
        self,
        pid: int,
        *,
        deadline_ns: int,
    ) -> LinuxProcessGeneration | LinuxMemoryUnavailable:
        """Read live ``(pid, start_ticks, boot_id, pid_ns_inode)`` for ``pid``."""

        stat_path = self._proc_root / str(pid) / "stat"
        stat_bytes = self._read_bounded_bytes(
            stat_path, deadline_ns=deadline_ns, missing_is_process_exit=True
        )
        if isinstance(stat_bytes, LinuxMemoryUnavailable):
            return stat_bytes

        parsed = parse_proc_stat_start_ticks(stat_bytes, expected_pid=pid)
        if isinstance(parsed, LinuxMemoryUnavailable):
            return parsed
        parsed_pid, start_ticks = parsed

        # Best-effort boot_id from /proc/sys/kernel/random/boot_id; still enforce
        # input, file-type, and time bounds if the file is present.
        boot_id: str | None = None
        boot_path = self._proc_root / "sys" / "kernel" / "random" / "boot_id"
        boot_raw = self._read_bounded_bytes(
            boot_path, deadline_ns=deadline_ns, missing_is_process_exit=False
        )
        if isinstance(boot_raw, LinuxMemoryUnavailable):
            if boot_raw.reason_code in (
                "input_limit_exceeded",
                "read_timeout",
                "unsafe_procfs_file",
            ):
                return boot_raw
        else:
            try:
                boot_text = boot_raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                boot_text = ""
            if boot_text:
                boot_id = boot_text

        # Best-effort PID namespace symlink target via os.readlink.
        pid_ns_inode: str | None = None
        ns_path = self._proc_root / str(pid) / "ns" / "pid"
        try:
            ns_target = os.readlink(ns_path).strip()
        except OSError:
            ns_target = ""
        if ns_target:
            pid_ns_inode = ns_target

        return LinuxProcessGeneration(
            pid=parsed_pid,
            start_ticks=start_ticks,
            boot_id=boot_id,
            pid_ns_inode=pid_ns_inode,
            host_id=self._resolve_host_id(),
        )

    def _unavailable_reading(
        self,
        issue: LinuxMemoryUnavailable,
        *,
        read_start_ns: int,
        read_end_ns: int,
        target_generation: LinuxProcessGeneration | None,
        observed_before: LinuxProcessGeneration | None,
        observed_after: LinuxProcessGeneration | None,
    ) -> LinuxProcessMemoryReading:
        return LinuxProcessMemoryReading(
            status=issue.status,
            rss_bytes=issue,
            virtual_bytes=issue,
            thread_count=issue,
            read_start_ns=read_start_ns,
            read_end_ns=max(read_end_ns, read_start_ns),
            reason_code=issue.reason_code,
            detail=issue.reason,
            target_generation=target_generation,
            observed_generation_before=observed_before,
            observed_generation_after=observed_after,
        )

    def bind_target(self, *, pid: int | None = None) -> LinuxProcessGeneration | None:
        """Bind the exact live target generation at activation time.

        If ``expected_generation`` was already supplied at construction, it is
        validated against ``pid`` (if given) and returned unchanged so later
        reads still detect drift from that injected expectation.

        Parameters
        ----------
        pid : int or None, optional
            Optional explicit target PID to bind.

        Returns
        -------
        LinuxProcessGeneration or None
            Bound generation on success, or ``None`` (with :attr:`bind_error`
            populated) if the platform is unsupported or procfs is unreadable.
        """

        if pid is not None:
            pid = _require_positive_int(pid, "LinuxProcessMemoryProbe.bind_target.pid")
            if self._pid is not None and pid != self._pid:
                raise ValueError(
                    f"bind_target pid ({pid}) disagrees with configured pid ({self._pid})"
                )
            if self._bound_generation is not None and pid != self._bound_generation.pid:
                raise ValueError(
                    f"bind_target pid ({pid}) disagrees with bound generation pid "
                    f"({self._bound_generation.pid})"
                )

        platform_issue = self._check_platform()
        if platform_issue is not None:
            if pid is not None:
                self._pid = pid
            self._bind_error = platform_issue
            return None

        if self._bound_generation is not None:
            if pid is not None:
                self._pid = pid
            self._bind_error = None
            return self._bound_generation

        target_pid = (
            pid
            if pid is not None
            else (self._pid if self._pid is not None else os.getpid())
        )
        # Lock validated target_pid before reading procfs so a failed bind
        # cannot cause later read() calls to silently retarget to os.getpid().
        self._pid = target_pid
        start_ns = int(self._monotonic_ns())
        deadline_ns = start_ns + self._max_read_ns
        gen = self._read_generation(target_pid, deadline_ns=deadline_ns)
        if isinstance(gen, LinuxMemoryUnavailable):
            self._bind_error = gen
            return None
        if int(self._monotonic_ns()) > deadline_ns:
            self._bind_error = LinuxMemoryUnavailable(
                reason=(
                    f"binding target generation for pid {target_pid} exceeded "
                    f"max_read_seconds ({self._max_read_seconds}s)"
                ),
                reason_code="read_timeout",
                status="error",
            )
            return None
        self._bound_generation = gen
        self._bind_error = None
        return gen

    def read(self) -> LinuxProcessMemoryReading:
        """Read current Linux process memory with pre- and post-read generation verification."""

        read_start_ns = int(self._monotonic_ns())
        deadline_ns = read_start_ns + self._max_read_ns

        platform_issue = self._check_platform()
        if platform_issue is not None:
            return self._unavailable_reading(
                platform_issue,
                read_start_ns=read_start_ns,
                read_end_ns=int(self._monotonic_ns()),
                target_generation=self._bound_generation,
                observed_before=None,
                observed_after=None,
            )

        expected = self._bound_generation
        target_pid = (
            expected.pid
            if expected is not None
            else (self._pid if self._pid is not None else os.getpid())
        )

        # 1. Pre-read generation read and verification.
        gen_before = self._read_generation(target_pid, deadline_ns=deadline_ns)
        after_pre_ns = int(self._monotonic_ns())
        if isinstance(gen_before, LinuxMemoryUnavailable):
            return self._unavailable_reading(
                gen_before,
                read_start_ns=read_start_ns,
                read_end_ns=after_pre_ns,
                target_generation=expected,
                observed_before=None,
                observed_after=None,
            )
        if after_pre_ns > deadline_ns:
            timeout_issue = LinuxMemoryUnavailable(
                reason=(
                    f"pre-read generation check exceeded max_read_seconds "
                    f"({self._max_read_seconds}s)"
                ),
                reason_code="read_timeout",
                status="error",
            )
            return self._unavailable_reading(
                timeout_issue,
                read_start_ns=read_start_ns,
                read_end_ns=after_pre_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
            )

        if expected is None:
            expected = gen_before
            self._bound_generation = gen_before
            self._bind_error = None

        pre_mismatch = _generation_mismatch_detail(
            expected, gen_before, exact_optional=True
        )
        if pre_mismatch is not None:
            changed = LinuxMemoryUnavailable(
                reason=f"target process generation changed before read: {pre_mismatch}",
                reason_code="generation_changed",
                status="unavailable",
            )
            return self._unavailable_reading(
                changed,
                read_start_ns=read_start_ns,
                read_end_ns=after_pre_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
            )

        # 2. Bounded /proc/<pid>/status read.
        status_path = self._proc_root / str(target_pid) / "status"
        status_bytes = self._read_bounded_bytes(
            status_path, deadline_ns=deadline_ns, missing_is_process_exit=True
        )
        after_status_ns = int(self._monotonic_ns())
        if isinstance(status_bytes, LinuxMemoryUnavailable):
            return self._unavailable_reading(
                status_bytes,
                read_start_ns=read_start_ns,
                read_end_ns=after_status_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
            )
        if after_status_ns > deadline_ns:
            timeout_issue = LinuxMemoryUnavailable(
                reason=(
                    f"reading {status_path} exceeded max_read_seconds "
                    f"({self._max_read_seconds}s)"
                ),
                reason_code="read_timeout",
                status="error",
            )
            return self._unavailable_reading(
                timeout_issue,
                read_start_ns=read_start_ns,
                read_end_ns=after_status_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
            )

        # 3. Post-read generation verification before accepting any memory value.
        gen_after = self._read_generation(target_pid, deadline_ns=deadline_ns)
        read_end_ns = max(int(self._monotonic_ns()), read_start_ns)
        if isinstance(gen_after, LinuxMemoryUnavailable):
            return self._unavailable_reading(
                gen_after,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
            )
        if read_end_ns > deadline_ns:
            timeout_issue = LinuxMemoryUnavailable(
                reason=(
                    f"post-read generation verification exceeded max_read_seconds "
                    f"({self._max_read_seconds}s)"
                ),
                reason_code="read_timeout",
                status="error",
            )
            return self._unavailable_reading(
                timeout_issue,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=gen_after,
            )

        post_mismatch = _generation_mismatch_detail(
            expected, gen_after, exact_optional=True
        ) or _generation_mismatch_detail(gen_before, gen_after, exact_optional=True)
        if post_mismatch is not None:
            changed = LinuxMemoryUnavailable(
                reason=f"target process generation changed during read: {post_mismatch}",
                reason_code="generation_changed",
                status="unavailable",
            )
            return self._unavailable_reading(
                changed,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=gen_after,
            )

        # 4. Parse /proc/<pid>/status only after generation stability is proven.
        parsed_status = parse_proc_status_memory(status_bytes)
        if isinstance(parsed_status, LinuxMemoryUnavailable):
            return self._unavailable_reading(
                parsed_status,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=gen_after,
            )

        rss_bytes, virtual_bytes, thread_count = parsed_status
        issues: list[LinuxMemoryUnavailable] = [
            val
            for val in (rss_bytes, virtual_bytes, thread_count)
            if isinstance(val, LinuxMemoryUnavailable)
        ]
        if not issues:
            return LinuxProcessMemoryReading(
                status="ok",
                rss_bytes=rss_bytes,
                virtual_bytes=virtual_bytes,
                thread_count=thread_count,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                target_generation=expected,
                observed_generation_before=gen_before,
                observed_generation_after=gen_after,
            )

        unique_codes = {issue.reason_code for issue in issues}
        detail = "; ".join(issue.reason for issue in issues)
        if len(issues) < 3:
            summary_code = (
                next(iter(unique_codes)) if len(unique_codes) == 1 else "partial_reading"
            )
            return LinuxProcessMemoryReading(
                status="partial",
                rss_bytes=rss_bytes,
                virtual_bytes=virtual_bytes,
                thread_count=thread_count,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                reason_code=summary_code,
                detail=detail,
                target_generation=expected,
                observed_generation_before=gen_before,
                observed_generation_after=gen_after,
            )

        overall_status = (
            "error" if any(issue.status == "error" for issue in issues) else "unavailable"
        )
        summary_code = (
            next(iter(unique_codes)) if len(unique_codes) == 1 else issues[0].reason_code
        )
        return LinuxProcessMemoryReading(
            status=overall_status,
            rss_bytes=rss_bytes,
            virtual_bytes=virtual_bytes,
            thread_count=thread_count,
            read_start_ns=read_start_ns,
            read_end_ns=read_end_ns,
            reason_code=summary_code,
            detail=detail,
            target_generation=expected,
            observed_generation_before=gen_before,
            observed_generation_after=gen_after,
        )


__all__ = [
    "DEFAULT_LINUX_MEMORY_MAX_READ_BYTES",
    "DEFAULT_LINUX_MEMORY_MAX_READ_SECONDS",
    "DEFAULT_PROC_ROOT",
    "LINUX_MEMORY_READING_STATUSES",
    "LinuxMemoryFieldValue",
    "LinuxMemoryUnavailable",
    "LinuxProcessGeneration",
    "LinuxProcessMemoryProbe",
    "LinuxProcessMemoryReading",
    "ProcessRUsageProbe",
    "ProcessResourceBaseline",
    "ProcessResourceResult",
    "ResourceScope",
    "ResourceUnavailable",
    "parse_proc_stat_start_ticks",
    "parse_proc_status_memory",
]
