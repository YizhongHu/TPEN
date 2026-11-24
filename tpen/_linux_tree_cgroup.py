"""Bounded Linux descendant-tree and read-only cgroup v2 probes (strictly stdlib).

This internal implementation module is publicly re-exported by
:mod:`tpen.process_resources`. It provides two strictly standard-library,
Torch-free, read-only Linux observability probes:

* :class:`LinuxProcessTreeProbe` — bounded procfs scan for verified live
  descendants of a bound target process generation. Enforces scan time,
  enumerated-candidate, observed-member, and per-file byte limits; excludes
  zombies, foreign/unrelated processes, PID-reused candidates, and processes
  that exit, reparent, or change generation between ``stat`` and ``status``
  reads; tags known observer helper PIDs honestly with ``role="observer"``
  without invisible subtraction; and tracks per-member exact-generation CPU
  baselines across scans.
* :class:`LinuxCgroupV2Probe` — read-only cgroup v2 probe resolving unified
  ``0::<path>`` membership from ``/proc/<pid>/cgroup`` and ``cgroup2`` mounts
  from ``/proc/<pid>/mountinfo`` with strict path-containment and no-symlink
  enforcement, refusing cgroup v1 pretending to be v2, resolving hierarchical
  effective ``memory.max`` and ``cpu.max`` limits up the ancestor chain, and
  recording explicit ``shared_scope_not_exclusive=True`` and co-member
  provenance from ``cgroup.procs``.

All public symbols are re-exported from :mod:`tpen.process_resources`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import socket
import stat
import sys
import time
from typing import Any, Union


_NS_PER_SECOND = 1_000_000_000
_USEC_PER_SECOND = 1_000_000.0
_READ_CHUNK_BYTES = 16384
_MAX_CPUSET_SPAN = 1_048_576
_DEAD_OR_ZOMBIE_STATES: frozenset[str] = frozenset({"Z", "X", "x"})

DEFAULT_PROC_ROOT = Path("/proc")
DEFAULT_LINUX_TREE_MAX_SCAN_SECONDS = 1.0
DEFAULT_LINUX_TREE_MAX_OBSERVED_MEMBERS = 1024
DEFAULT_LINUX_TREE_MAX_ENUMERATED_CANDIDATES = 16384
DEFAULT_LINUX_TREE_MAX_READ_BYTES = 65536

DEFAULT_LINUX_CGROUP_MAX_READ_BYTES = 65536
DEFAULT_LINUX_CGROUP_MAX_MOUNTINFO_BYTES = 2 * 1024 * 1024
DEFAULT_LINUX_CGROUP_MAX_READ_SECONDS = 1.0
DEFAULT_LINUX_CGROUP_MAX_ANCESTOR_DEPTH = 64
DEFAULT_LINUX_CGROUP_MAX_PROCS = 4096

LINUX_TREE_MEMBER_ROLES: frozenset[str] = frozenset({"descendant", "observer"})
LINUX_TREE_READING_STATUSES: frozenset[str] = frozenset(
    {"ok", "partial", "unavailable", "error"}
)
LINUX_CGROUP_READING_STATUSES: frozenset[str] = frozenset(
    {"ok", "partial", "unavailable", "error"}
)

_MOUNTINFO_OCTAL_RE = re.compile(r"\\([0-7]{3})")


def _pr() -> Any:
    """Return :mod:`tpen.process_resources` without circular top-level import."""

    import tpen.process_resources as pr

    return pr


# ---------------------------------------------------------------------------
# Local strict validation helpers (pure stdlib)
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


def _require_non_negative_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a finite non-negative number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
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
    return bool(token) and token.isascii() and token.isdigit()


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _generation_exact_mismatch(
    expected: Any,
    observed: Any,
) -> str | None:
    """Return a concise mismatch diagnostic enforcing exact optional matching.

    Enforces exact equality across ``pid``, ``start_ticks``, ``boot_id``, and
    ``pid_ns_inode`` (``exact_optional=True``), and compares ``host_id`` when
    ``expected.host_id`` is explicitly specified (matching O4
    ``LinuxProcessMemoryProbe`` parity).
    """

    if expected.pid != observed.pid:
        return f"pid changed (expected {expected.pid}, observed {observed.pid})"
    if expected.start_ticks != observed.start_ticks:
        return (
            f"start_ticks changed (expected {expected.start_ticks}, "
            f"observed {observed.start_ticks})"
        )
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
    if expected.host_id is not None and expected.host_id != observed.host_id:
        return (
            f"host_id changed (expected {expected.host_id!r}, "
            f"observed {observed.host_id!r})"
        )
    return None


def _read_bounded_regular_bytes(
    path: Path,
    *,
    root_path: Path,
    max_read_bytes: int,
    max_read_seconds: float,
    deadline_ns: int,
    monotonic_ns: Callable[[], int],
    missing_reason_code: str,
    root_missing_reason_code: str,
    unsafe_reason_code: str = "unsafe_procfs_file",
    read_error_reason_code: str = "procfs_read_error",
    enforce_containment_root: Path | None = None,
) -> bytes | Any:
    """Safely read up to ``max_read_bytes`` from a regular, non-symlink file.

    Refuses symlinks (``S_ISLNK``) and non-regular files (FIFOs, sockets,
    devices) via ``os.lstat`` before ``open`` and ``os.fstat`` after ``open``,
    enforces optional strict path containment, and checks monotonic deadlines.
    """

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    try:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            return LinuxMemoryUnavailable(
                reason=f"refusing symlinked file {path}",
                reason_code=unsafe_reason_code,
                status="error",
            )
        if not stat.S_ISREG(st.st_mode):
            return LinuxMemoryUnavailable(
                reason=f"refusing non-regular file {path}",
                reason_code=unsafe_reason_code,
                status="error",
            )
        if enforce_containment_root is not None:
            resolved_root = enforce_containment_root.resolve(strict=False)
            resolved_path = path.resolve(strict=False)
            try:
                resolved_path.relative_to(resolved_root)
            except ValueError:
                return LinuxMemoryUnavailable(
                    reason=(
                        f"path {path} resolves outside contained root "
                        f"{enforce_containment_root}"
                    ),
                    reason_code=unsafe_reason_code,
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
                            reason=f"refusing non-regular opened file {path}",
                            reason_code=unsafe_reason_code,
                            status="error",
                        )
            chunks: list[bytes] = []
            total = 0
            limit = max_read_bytes + 1
            while total < limit:
                if total >= _READ_CHUNK_BYTES and int(monotonic_ns()) > deadline_ns:
                    return LinuxMemoryUnavailable(
                        reason=(
                            f"reading {path} exceeded time bound "
                            f"({max_read_seconds}s)"
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
        if not root_path.exists():
            return LinuxMemoryUnavailable(
                reason=f"root directory {root_path} is missing: {exc}",
                reason_code=root_missing_reason_code,
                status="unavailable",
            )
        return LinuxMemoryUnavailable(
            reason=f"path {path} not found: {exc}",
            reason_code=missing_reason_code,
            status="unavailable",
        )
    except OSError as exc:
        return LinuxMemoryUnavailable(
            reason=f"OS error reading {path}: {type(exc).__name__}: {exc}",
            reason_code=read_error_reason_code,
            status="error",
        )

    if total > max_read_bytes:
        return LinuxMemoryUnavailable(
            reason=f"file {path} exceeded max_read_bytes ({max_read_bytes})",
            reason_code="input_limit_exceeded",
            status="error",
        )
    return b"".join(chunks)


def _check_safe_proc_pid_dir(
    proc_root: Path,
    pid: int,
) -> Path | Any:
    """Verify ``proc_root`` and ``proc_root / str(pid)`` are real directories, not symlinks."""

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    try:
        root_st = os.lstat(proc_root)
    except PermissionError as exc:
        return LinuxMemoryUnavailable(
            reason=f"permission denied accessing procfs root {proc_root}: {exc}",
            reason_code="permission_denied",
            status="unavailable",
        )
    except FileNotFoundError as exc:
        return LinuxMemoryUnavailable(
            reason=f"procfs root {proc_root} is missing: {exc}",
            reason_code="procfs_missing",
            status="unavailable",
        )
    except OSError as exc:
        return LinuxMemoryUnavailable(
            reason=f"OS error accessing procfs root {proc_root}: {exc}",
            reason_code="procfs_read_error",
            status="error",
        )
    if stat.S_ISLNK(root_st.st_mode) or not stat.S_ISDIR(root_st.st_mode):
        return LinuxMemoryUnavailable(
            reason=f"refusing symlinked or non-directory procfs root {proc_root}",
            reason_code="unsafe_procfs_file",
            status="error",
        )

    pid_dir = proc_root / str(pid)
    try:
        pid_st = os.lstat(pid_dir)
    except PermissionError as exc:
        return LinuxMemoryUnavailable(
            reason=f"permission denied accessing {pid_dir}: {exc}",
            reason_code="permission_denied",
            status="unavailable",
        )
    except (FileNotFoundError, ProcessLookupError) as exc:
        return LinuxMemoryUnavailable(
            reason=f"process directory {pid_dir} not found: {exc}",
            reason_code="process_exited",
            status="unavailable",
        )
    except OSError as exc:
        return LinuxMemoryUnavailable(
            reason=f"OS error accessing {pid_dir}: {exc}",
            reason_code="procfs_read_error",
            status="error",
        )
    if stat.S_ISLNK(pid_st.st_mode) or not stat.S_ISDIR(pid_st.st_mode):
        return LinuxMemoryUnavailable(
            reason=f"refusing symlinked or non-directory procfs pid dir {pid_dir}",
            reason_code="unsafe_procfs_file",
            status="error",
        )
    return pid_dir


def _read_pid_ns_inode(pid_dir: Path) -> tuple[str | None, Any | None]:
    """Read ``<pid_dir>/ns/pid`` symlink target safely.

    Returns ``(ns_inode_or_None, error_or_None)``. Absence or benign ``OSError``
    returns ``(None, None)``; ``PermissionError`` or symlinked ``ns/`` parent
    directory returns a structured :class:`LinuxMemoryUnavailable`.
    """

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable
    ns_dir = pid_dir / "ns"
    try:
        ns_dir_st = os.lstat(ns_dir)
    except PermissionError as exc:
        return (
            None,
            LinuxMemoryUnavailable(
                reason=f"permission denied accessing {ns_dir}: {exc}",
                reason_code="permission_denied",
                status="unavailable",
            ),
        )
    except (FileNotFoundError, ProcessLookupError, OSError):
        return (None, None)
    if stat.S_ISLNK(ns_dir_st.st_mode) or not stat.S_ISDIR(ns_dir_st.st_mode):
        return (
            None,
            LinuxMemoryUnavailable(
                reason=f"refusing symlinked or non-directory {ns_dir}",
                reason_code="unsafe_procfs_file",
                status="error",
            ),
        )

    ns_path = ns_dir / "pid"
    try:
        ns_target = os.readlink(ns_path).strip()
    except PermissionError as exc:
        return (
            None,
            LinuxMemoryUnavailable(
                reason=f"permission denied reading {ns_path}: {exc}",
                reason_code="permission_denied",
                status="unavailable",
            ),
        )
    except OSError:
        return (None, None)
    return (ns_target if ns_target else None, None)


def _read_boot_id(
    proc_root: Path,
    *,
    max_read_bytes: int,
    max_read_seconds: float,
    deadline_ns: int,
    monotonic_ns: Callable[[], int],
) -> tuple[str | None, Any | None]:
    """Read optional ``/proc/sys/kernel/random/boot_id`` with hard safety bounds."""

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable
    boot_path = proc_root / "sys" / "kernel" / "random" / "boot_id"
    boot_raw = _read_bounded_regular_bytes(
        boot_path,
        root_path=proc_root,
        max_read_bytes=max_read_bytes,
        max_read_seconds=max_read_seconds,
        deadline_ns=deadline_ns,
        monotonic_ns=monotonic_ns,
        missing_reason_code="procfs_missing",
        root_missing_reason_code="procfs_missing",
    )
    if isinstance(boot_raw, LinuxMemoryUnavailable):
        if boot_raw.reason_code in (
            "input_limit_exceeded",
            "read_timeout",
            "unsafe_procfs_file",
        ):
            return (None, boot_raw)
        return (None, None)
    try:
        boot_text = boot_raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return (None, None)
    return (boot_text if boot_text else None, None)


# ---------------------------------------------------------------------------
# Pure /proc/<pid>/stat full-field parser (for process tree scans)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcStatSnapshot:
    """Parsed identity, linkage, state, and CPU fields from ``/proc/<pid>/stat``.

    Parameters
    ----------
    pid : int
        Process ID (field 1).
    state : str
        Single-character process state (field 3, e.g. ``'R'``, ``'S'``, ``'Z'``).
    ppid : int
        Parent process ID (field 4, ``>= 0``).
    utime_ticks : int
        User-mode CPU time in clock ticks (field 14, ``>= 0``).
    stime_ticks : int
        Kernel-mode CPU time in clock ticks (field 15, ``>= 0``).
    start_ticks : int
        Process start time since boot in clock ticks (field 22, ``>= 0``).
    """

    pid: int
    state: str
    ppid: int
    utime_ticks: int
    stime_ticks: int
    start_ticks: int

    def __post_init__(self) -> None:
        _require_positive_int(self.pid, "ProcStatSnapshot.pid")
        if (
            type(self.state) is not str
            or len(self.state) != 1
            or not self.state.isascii()
            or not self.state.isalpha()
        ):
            raise ValueError(
                f"ProcStatSnapshot.state must be a single ASCII letter, got {self.state!r}"
            )
        _require_non_negative_int(self.ppid, "ProcStatSnapshot.ppid")
        _require_non_negative_int(self.utime_ticks, "ProcStatSnapshot.utime_ticks")
        _require_non_negative_int(self.stime_ticks, "ProcStatSnapshot.stime_ticks")
        _require_non_negative_int(self.start_ticks, "ProcStatSnapshot.start_ticks")


def parse_proc_stat_fields(
    raw: str | bytes,
    *,
    expected_pid: int | None = None,
) -> ProcStatSnapshot | Any:
    """Parse ``ProcStatSnapshot`` from Linux ``/proc/<pid>/stat`` content.

    Locates the final ``')'`` via ``rfind`` so arbitrary process names
    (``comm``) containing spaces, newlines, parentheses, or non-UTF-8 bytes
    never shift subsequent fields. After the final ``')'``:

    * ``after[0]``: ``state`` (field 3)
    * ``after[1]``: ``ppid`` (field 4)
    * ``after[11]``: ``utime`` (field 14)
    * ``after[12]``: ``stime`` (field 15)
    * ``after[19]``: ``starttime`` (field 22)
    """

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

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

    for label, idx in (
        ("ppid", 1),
        ("utime", 11),
        ("stime", 12),
        ("starttime", 19),
    ):
        tok = after_fields[idx]
        if not _is_strict_non_negative_ascii_int(tok):
            return LinuxMemoryUnavailable(
                reason=f"malformed /proc/<pid>/stat: invalid {label} token {tok!r}",
                reason_code="malformed_procfs",
                status="error",
            )

    return ProcStatSnapshot(
        pid=parsed_pid,
        state=state_token,
        ppid=int(after_fields[1]),
        utime_ticks=int(after_fields[11]),
        stime_ticks=int(after_fields[12]),
        start_ticks=int(after_fields[19]),
    )


# ---------------------------------------------------------------------------
# Bounded Linux descendant-tree probe and typed reading models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinuxProcessTreeMember:
    """One verified live descendant observed during a bounded procfs tree scan."""

    generation: Any  # LinuxProcessGeneration
    ppid: int
    role: str
    state: str
    rss_bytes: Any  # int | ResourceUnavailable
    utime_ticks: int
    stime_ticks: int
    user_cpu_seconds: float
    system_cpu_seconds: float
    delta_matched: bool = False
    cpu_user_delta_seconds: float | None = None
    cpu_system_delta_seconds: float | None = None

    def __post_init__(self) -> None:
        pr = _pr()
        if not isinstance(self.generation, pr.LinuxProcessGeneration):
            raise TypeError(
                "LinuxProcessTreeMember.generation must be LinuxProcessGeneration, "
                f"got {type(self.generation).__name__}"
            )
        _require_non_negative_int(self.ppid, "LinuxProcessTreeMember.ppid")
        if type(self.role) is not str or self.role not in LINUX_TREE_MEMBER_ROLES:
            raise ValueError(
                f"LinuxProcessTreeMember.role must be one of "
                f"{sorted(LINUX_TREE_MEMBER_ROLES)}, got {self.role!r}"
            )
        if (
            type(self.state) is not str
            or len(self.state) != 1
            or not self.state.isascii()
            or not self.state.isalpha()
        ):
            raise ValueError(
                f"LinuxProcessTreeMember.state must be a single ASCII letter, got {self.state!r}"
            )
        if not isinstance(self.rss_bytes, pr.ResourceUnavailable):
            _require_non_negative_int(self.rss_bytes, "LinuxProcessTreeMember.rss_bytes")
        _require_non_negative_int(self.utime_ticks, "LinuxProcessTreeMember.utime_ticks")
        _require_non_negative_int(self.stime_ticks, "LinuxProcessTreeMember.stime_ticks")
        object.__setattr__(
            self,
            "user_cpu_seconds",
            _require_non_negative_finite_float(
                self.user_cpu_seconds, "LinuxProcessTreeMember.user_cpu_seconds"
            ),
        )
        object.__setattr__(
            self,
            "system_cpu_seconds",
            _require_non_negative_finite_float(
                self.system_cpu_seconds, "LinuxProcessTreeMember.system_cpu_seconds"
            ),
        )
        if type(self.delta_matched) is not bool:
            raise TypeError(
                "LinuxProcessTreeMember.delta_matched must be bool, "
                f"got {type(self.delta_matched).__name__}"
            )
        if self.cpu_user_delta_seconds is not None:
            object.__setattr__(
                self,
                "cpu_user_delta_seconds",
                _require_non_negative_finite_float(
                    self.cpu_user_delta_seconds,
                    "LinuxProcessTreeMember.cpu_user_delta_seconds",
                ),
            )
        if self.cpu_system_delta_seconds is not None:
            object.__setattr__(
                self,
                "cpu_system_delta_seconds",
                _require_non_negative_finite_float(
                    self.cpu_system_delta_seconds,
                    "LinuxProcessTreeMember.cpu_system_delta_seconds",
                ),
            )

    def to_subject_member_dict(self) -> dict[str, Any]:
        """Return a JSON-safe member provenance dictionary for ``subject['members']``."""

        pr = _pr()
        data = self.generation.to_dict()
        data["ppid"] = self.ppid
        data["role"] = self.role
        data["state"] = self.state
        data["utime_ticks"] = self.utime_ticks
        data["stime_ticks"] = self.stime_ticks
        data["user_cpu_seconds"] = self.user_cpu_seconds
        data["system_cpu_seconds"] = self.system_cpu_seconds
        data["delta_matched"] = self.delta_matched
        if isinstance(self.rss_bytes, pr.ResourceUnavailable):
            data["rss_status"] = getattr(self.rss_bytes, "status", "unavailable")
            data["rss_reason_code"] = getattr(
                self.rss_bytes, "reason_code", "unavailable"
            )
        else:
            data["rss_bytes"] = self.rss_bytes
        return data


@dataclass(frozen=True)
class LinuxProcessTreeReading:
    """Typed bounded Linux descendant-tree reading with full membership provenance."""

    status: str
    live_member_count: Any  # int | ResourceUnavailable
    sum_rss_bytes: Any  # int | ResourceUnavailable
    matched_cpu_user_delta_seconds: Any  # float | ResourceUnavailable
    matched_cpu_system_delta_seconds: Any  # float | ResourceUnavailable
    read_start_ns: int
    read_end_ns: int
    members: tuple[LinuxProcessTreeMember, ...] = ()
    membership_complete: bool = False
    partial_reasons: tuple[str, ...] = ()
    enumerated_candidates: int = 0
    observed_member_count: int = 0
    matched_delta_member_count: int = 0
    permission_denied_count: int = 0
    excluded_count: int = 0
    clock_ticks_per_second: int = 100
    reason_code: str | None = None
    detail: str | None = None
    target_generation: Any = None  # LinuxProcessGeneration | None
    observed_generation_before: Any = None
    observed_generation_after: Any = None

    def __post_init__(self) -> None:
        pr = _pr()
        if type(self.status) is not str or self.status not in LINUX_TREE_READING_STATUSES:
            raise ValueError(
                f"LinuxProcessTreeReading.status must be one of "
                f"{sorted(LINUX_TREE_READING_STATUSES)}, got {self.status!r}"
            )
        start = _require_non_negative_int(
            self.read_start_ns, "LinuxProcessTreeReading.read_start_ns"
        )
        end = _require_non_negative_int(
            self.read_end_ns, "LinuxProcessTreeReading.read_end_ns"
        )
        if end < start:
            raise ValueError(
                f"LinuxProcessTreeReading.read_end_ns ({end}) must be >= "
                f"read_start_ns ({start})"
            )
        if type(self.membership_complete) is not bool:
            raise TypeError(
                "LinuxProcessTreeReading.membership_complete must be bool, "
                f"got {type(self.membership_complete).__name__}"
            )
        _require_non_negative_int(
            self.enumerated_candidates, "LinuxProcessTreeReading.enumerated_candidates"
        )
        _require_non_negative_int(
            self.observed_member_count, "LinuxProcessTreeReading.observed_member_count"
        )
        _require_non_negative_int(
            self.matched_delta_member_count,
            "LinuxProcessTreeReading.matched_delta_member_count",
        )
        _require_non_negative_int(
            self.permission_denied_count,
            "LinuxProcessTreeReading.permission_denied_count",
        )
        _require_non_negative_int(
            self.excluded_count, "LinuxProcessTreeReading.excluded_count"
        )
        _require_positive_int(
            self.clock_ticks_per_second,
            "LinuxProcessTreeReading.clock_ticks_per_second",
        )
        if not isinstance(self.members, tuple) or not all(
            isinstance(item, LinuxProcessTreeMember) for item in self.members
        ):
            raise TypeError(
                "LinuxProcessTreeReading.members must be a tuple of LinuxProcessTreeMember"
            )
        if not isinstance(self.partial_reasons, tuple) or not all(
            type(item) is str and bool(item.strip()) for item in self.partial_reasons
        ):
            raise TypeError(
                "LinuxProcessTreeReading.partial_reasons must be a tuple of nonempty strings"
            )

        ok_count = 0
        for int_name in ("live_member_count", "sum_rss_bytes"):
            val = getattr(self, int_name)
            if isinstance(val, pr.ResourceUnavailable):
                continue
            if type(val) is not int or val < 0:
                raise ValueError(
                    f"LinuxProcessTreeReading.{int_name} must be a non-negative int "
                    f"or ResourceUnavailable, got {val!r}"
                )
            ok_count += 1
        for flt_name in (
            "matched_cpu_user_delta_seconds",
            "matched_cpu_system_delta_seconds",
        ):
            val = getattr(self, flt_name)
            if isinstance(val, pr.ResourceUnavailable):
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"LinuxProcessTreeReading.{flt_name} must be a finite non-negative "
                    f"number or ResourceUnavailable, got {val!r}"
                )
            number = float(val)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(
                    f"LinuxProcessTreeReading.{flt_name} must be finite and >= 0, got {val!r}"
                )
            object.__setattr__(self, flt_name, number)
            ok_count += 1

        if self.status == "ok":
            if ok_count != 4 or not self.membership_complete:
                raise ValueError(
                    "LinuxProcessTreeReading with status='ok' requires all 4 fields "
                    "to be valid numbers and membership_complete=True"
                )
        elif self.status == "partial":
            if ok_count == 0:
                raise ValueError(
                    "LinuxProcessTreeReading with status='partial' requires at least "
                    "one valid numeric field"
                )
            if ok_count == 4 and self.membership_complete and not self.partial_reasons:
                raise ValueError(
                    "LinuxProcessTreeReading with status='partial' requires an unavailable "
                    "field or incomplete membership/partial_reasons"
                )
            if type(self.reason_code) is not str or not self.reason_code.strip():
                raise ValueError(
                    "LinuxProcessTreeReading with status='partial' requires reason_code"
                )
        else:
            if ok_count != 0:
                raise ValueError(
                    f"LinuxProcessTreeReading with status={self.status!r} must not "
                    "carry numeric field values"
                )
            if type(self.reason_code) is not str or not self.reason_code.strip():
                raise ValueError(
                    f"LinuxProcessTreeReading with status={self.status!r} requires reason_code"
                )

        for gen_name in (
            "target_generation",
            "observed_generation_before",
            "observed_generation_after",
        ):
            gen = getattr(self, gen_name)
            if gen is not None and not isinstance(gen, pr.LinuxProcessGeneration):
                raise TypeError(
                    f"LinuxProcessTreeReading.{gen_name} must be "
                    f"LinuxProcessGeneration or None, got {type(gen).__name__}"
                )

    def to_subject_dict(
        self,
        *,
        host_id: str | None = None,
        invocation_token: str | None = None,
        fallback_pid: int | None = None,
        max_members: int | None = None,
    ) -> dict[str, Any]:
        """Return an O2 ``tpen.run-observability/v1`` ``descendant_set`` subject mapping."""

        if self.target_generation is not None:
            root_dict = self.target_generation.to_subject_dict(
                host_id=host_id,
                invocation_token=invocation_token,
            )
        else:
            root_dict = {"kind": "process"}
            if host_id is not None:
                root_dict["host_id"] = host_id
            if fallback_pid is not None:
                root_dict["pid"] = fallback_pid
            if invocation_token is not None:
                root_dict["invocation_token"] = invocation_token

        truncated_for_budget = False
        if max_members is not None and len(self.members) > max_members:
            member_slice = self.members[:max_members]
            truncated_for_budget = True
        else:
            member_slice = self.members

        subject: dict[str, Any] = {
            "kind": "descendant_set",
            "root": root_dict,
            "members": [member.to_subject_member_dict() for member in member_slice],
            "membership_complete": self.membership_complete,
            "partial_reasons": list(self.partial_reasons),
            "enumerated_candidates": self.enumerated_candidates,
            "observed_member_count": self.observed_member_count,
            "matched_delta_member_count": self.matched_delta_member_count,
            "permission_denied_count": self.permission_denied_count,
            "excluded_count": self.excluded_count,
            "clock_ticks_per_second": self.clock_ticks_per_second,
        }
        if truncated_for_budget:
            subject["members_truncated_for_record_budget"] = True
            subject["serialized_member_count"] = len(member_slice)
        return subject


class LinuxProcessTreeProbe:
    """Bounded Linux procfs descendant probe with exact-generation tracking.

    Construction is strictly pure and side-effect-free: it performs zero
    filesystem I/O and never calls ``os.getpid()`` when ``pid=None``.
    """

    def __init__(
        self,
        *,
        pid: int | None = None,
        expected_generation: Any = None,
        observer_pids: Collection[int] = (),
        proc_root: str | Path | None = None,
        max_scan_seconds: float = DEFAULT_LINUX_TREE_MAX_SCAN_SECONDS,
        max_observed_members: int = DEFAULT_LINUX_TREE_MAX_OBSERVED_MEMBERS,
        max_enumerated_candidates: int = DEFAULT_LINUX_TREE_MAX_ENUMERATED_CANDIDATES,
        max_read_bytes: int = DEFAULT_LINUX_TREE_MAX_READ_BYTES,
        clock_ticks_per_second: int | None = None,
        monotonic_ns: Callable[[], int] | None = None,
        platform_name: str | None = None,
        host_id: str | None = None,
    ) -> None:
        pr = _pr()
        if pid is not None:
            pid = _require_positive_int(pid, "LinuxProcessTreeProbe.pid")
        if expected_generation is not None:
            if not isinstance(expected_generation, pr.LinuxProcessGeneration):
                raise TypeError(
                    "LinuxProcessTreeProbe.expected_generation must be "
                    f"LinuxProcessGeneration or None, got {type(expected_generation).__name__}"
                )
            if pid is not None and pid != expected_generation.pid:
                raise ValueError(
                    f"LinuxProcessTreeProbe.pid ({pid}) disagrees with "
                    f"expected_generation.pid ({expected_generation.pid})"
                )

        if isinstance(observer_pids, (str, bytes)) or not isinstance(
            observer_pids, Collection
        ):
            raise TypeError(
                "LinuxProcessTreeProbe.observer_pids must be a collection of positive ints"
            )
        validated_observers: set[int] = set()
        for raw_obs_pid in observer_pids:
            validated_observers.add(
                _require_positive_int(raw_obs_pid, "LinuxProcessTreeProbe.observer_pids[]")
            )
        self._observer_pids: frozenset[int] = frozenset(validated_observers)

        self._pid: int | None = pid
        self._bound_generation: Any = expected_generation
        self._bind_error: Any = None

        if proc_root is None:
            self._proc_root = DEFAULT_PROC_ROOT
            self._uses_custom_proc_root = False
        elif isinstance(proc_root, Path):
            self._proc_root = proc_root
            self._uses_custom_proc_root = True
        elif type(proc_root) is str:
            if not proc_root.strip():
                raise ValueError(
                    "LinuxProcessTreeProbe.proc_root must be a nonempty path or None"
                )
            self._proc_root = Path(proc_root)
            self._uses_custom_proc_root = True
        else:
            raise TypeError(
                "LinuxProcessTreeProbe.proc_root must be str, Path, or None, "
                f"got {type(proc_root).__name__}"
            )

        self._max_scan_seconds = _require_positive_finite_float(
            max_scan_seconds, "LinuxProcessTreeProbe.max_scan_seconds"
        )
        self._max_scan_ns = max(1, int(round(self._max_scan_seconds * _NS_PER_SECOND)))
        self._max_observed_members = _require_positive_int(
            max_observed_members, "LinuxProcessTreeProbe.max_observed_members"
        )
        self._max_enumerated_candidates = _require_positive_int(
            max_enumerated_candidates, "LinuxProcessTreeProbe.max_enumerated_candidates"
        )
        self._max_read_bytes = _require_positive_int(
            max_read_bytes, "LinuxProcessTreeProbe.max_read_bytes"
        )
        if clock_ticks_per_second is not None:
            clock_ticks_per_second = _require_positive_int(
                clock_ticks_per_second,
                "LinuxProcessTreeProbe.clock_ticks_per_second",
            )
        self._clock_ticks_per_second = clock_ticks_per_second
        if monotonic_ns is not None and not callable(monotonic_ns):
            raise TypeError("LinuxProcessTreeProbe.monotonic_ns must be callable or None")
        self._monotonic_ns: Callable[[], int] = (
            time.monotonic_ns if monotonic_ns is None else monotonic_ns
        )
        self._platform_name = _require_optional_nonempty_str(
            platform_name, "LinuxProcessTreeProbe.platform_name"
        )
        self._host_id = _require_optional_nonempty_str(
            host_id, "LinuxProcessTreeProbe.host_id"
        )

        # Per-member exact-generation baseline:
        # (pid, start_ticks, boot_id, pid_ns_inode, host_id) -> (utime_ticks, stime_ticks)
        self._member_baselines: dict[
            tuple[int, int, str | None, str | None, str | None], tuple[int, int]
        ] = {}
        self._has_baseline_scan = False

    @property
    def target_generation(self) -> Any:
        """Return currently bound root :class:`LinuxProcessGeneration`, or ``None``."""

        return self._bound_generation

    @property
    def bind_error(self) -> Any:
        """Return the last :meth:`bind_target` unavailability/error, if any."""

        return self._bind_error

    @property
    def max_scan_seconds(self) -> float:
        return self._max_scan_seconds

    @property
    def max_observed_members(self) -> int:
        return self._max_observed_members

    @property
    def max_enumerated_candidates(self) -> int:
        return self._max_enumerated_candidates

    @property
    def max_read_bytes(self) -> int:
        return self._max_read_bytes

    @property
    def observer_pids(self) -> frozenset[int]:
        return self._observer_pids

    @property
    def proc_root(self) -> Path:
        return self._proc_root

    def _effective_platform(self) -> str:
        return sys.platform if self._platform_name is None else self._platform_name

    def _resolve_host_id(self) -> str:
        if self._host_id is not None:
            return self._host_id
        return socket.gethostname() or "unknown-host"

    def _resolve_clock_ticks_per_second(self) -> int:
        if self._clock_ticks_per_second is not None:
            return self._clock_ticks_per_second
        try:
            ticks = int(os.sysconf("SC_CLK_TCK"))
            if ticks >= 1:
                return ticks
        except (AttributeError, OSError, ValueError):
            pass
        return 100

    def _check_platform(self) -> Any:
        pr = _pr()
        if (
            self._platform_name is None
            and self._uses_custom_proc_root
            and self._proc_root != DEFAULT_PROC_ROOT
        ):
            return None
        plat = self._effective_platform()
        if not plat.startswith("linux"):
            return pr.LinuxMemoryUnavailable(
                reason=(
                    f"LinuxProcessTreeProbe requires Linux procfs "
                    f"(current platform is {plat!r}; no non-Linux fallback)"
                ),
                reason_code="unsupported_platform",
                status="unavailable",
            )
        return None

    def _read_root_generation_and_stat(
        self,
        pid: int,
        *,
        deadline_ns: int,
    ) -> tuple[Any, ProcStatSnapshot] | Any:
        pr = _pr()
        LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

        pid_dir = _check_safe_proc_pid_dir(self._proc_root, pid)
        if isinstance(pid_dir, LinuxMemoryUnavailable):
            return pid_dir

        stat_bytes = _read_bounded_regular_bytes(
            pid_dir / "stat",
            root_path=self._proc_root,
            max_read_bytes=self._max_read_bytes,
            max_read_seconds=self._max_scan_seconds,
            deadline_ns=deadline_ns,
            monotonic_ns=self._monotonic_ns,
            missing_reason_code="process_exited",
            root_missing_reason_code="procfs_missing",
        )
        if isinstance(stat_bytes, LinuxMemoryUnavailable):
            return stat_bytes

        snap = parse_proc_stat_fields(stat_bytes, expected_pid=pid)
        if isinstance(snap, LinuxMemoryUnavailable):
            return snap

        boot_id, boot_err = _read_boot_id(
            self._proc_root,
            max_read_bytes=self._max_read_bytes,
            max_read_seconds=self._max_scan_seconds,
            deadline_ns=deadline_ns,
            monotonic_ns=self._monotonic_ns,
        )
        if boot_err is not None:
            return boot_err

        pid_ns_inode, ns_err = _read_pid_ns_inode(pid_dir)
        if ns_err is not None and ns_err.reason_code == "unsafe_procfs_file":
            return ns_err

        gen = pr.LinuxProcessGeneration(
            pid=snap.pid,
            start_ticks=snap.start_ticks,
            boot_id=boot_id,
            pid_ns_inode=pid_ns_inode,
            host_id=self._resolve_host_id(),
        )
        return (gen, snap)

    def _unavailable_reading(
        self,
        issue: Any,
        *,
        read_start_ns: int,
        read_end_ns: int,
        target_generation: Any,
        observed_before: Any,
        observed_after: Any,
        clock_ticks_per_second: int,
        partial_reasons: tuple[str, ...] = (),
        enumerated_candidates: int = 0,
        permission_denied_count: int = 0,
        excluded_count: int = 0,
    ) -> LinuxProcessTreeReading:
        return LinuxProcessTreeReading(
            status=issue.status,
            live_member_count=issue,
            sum_rss_bytes=issue,
            matched_cpu_user_delta_seconds=issue,
            matched_cpu_system_delta_seconds=issue,
            read_start_ns=read_start_ns,
            read_end_ns=max(read_end_ns, read_start_ns),
            members=(),
            membership_complete=False,
            partial_reasons=partial_reasons,
            enumerated_candidates=enumerated_candidates,
            observed_member_count=0,
            matched_delta_member_count=0,
            permission_denied_count=permission_denied_count,
            excluded_count=excluded_count,
            clock_ticks_per_second=clock_ticks_per_second,
            reason_code=issue.reason_code,
            detail=issue.reason,
            target_generation=target_generation,
            observed_generation_before=observed_before,
            observed_generation_after=observed_after,
        )

    def bind_target(self, *, pid: int | None = None) -> Any:
        """Bind the exact live root target generation at activation time.

        Verifies platform BEFORE returning any pre-bound ``expected_generation``
        and persists validated ``self._pid`` before reading procfs.
        """

        pr = _pr()
        LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

        if pid is not None:
            pid = _require_positive_int(pid, "LinuxProcessTreeProbe.bind_target.pid")
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
        # Persist validated target_pid before procfs I/O so failed bind never
        # silently retargets later reads to os.getpid().
        self._pid = target_pid
        start_ns = int(self._monotonic_ns())
        deadline_ns = start_ns + self._max_scan_ns
        res = self._read_root_generation_and_stat(target_pid, deadline_ns=deadline_ns)
        if isinstance(res, LinuxMemoryUnavailable):
            self._bind_error = res
            return None
        gen, snap = res
        if snap.state in _DEAD_OR_ZOMBIE_STATES:
            self._bind_error = LinuxMemoryUnavailable(
                reason=f"target pid {target_pid} is zombie/dead (state={snap.state!r})",
                reason_code="process_exited",
                status="unavailable",
            )
            return None
        if int(self._monotonic_ns()) > deadline_ns:
            self._bind_error = LinuxMemoryUnavailable(
                reason=(
                    f"binding target generation for pid {target_pid} exceeded "
                    f"max_scan_seconds ({self._max_scan_seconds}s)"
                ),
                reason_code="read_timeout",
                status="error",
            )
            return None
        self._bound_generation = gen
        self._bind_error = None
        return gen

    def read(self) -> LinuxProcessTreeReading:
        """Execute one bounded descendant-tree scan with pre/post root verification."""

        pr = _pr()
        LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable
        LinuxProcessGeneration = pr.LinuxProcessGeneration
        parse_proc_status_memory = pr.parse_proc_status_memory

        read_start_ns = int(self._monotonic_ns())
        deadline_ns = read_start_ns + self._max_scan_ns
        clk_tck = self._resolve_clock_ticks_per_second()

        platform_issue = self._check_platform()
        if platform_issue is not None:
            return self._unavailable_reading(
                platform_issue,
                read_start_ns=read_start_ns,
                read_end_ns=int(self._monotonic_ns()),
                target_generation=self._bound_generation,
                observed_before=None,
                observed_after=None,
                clock_ticks_per_second=clk_tck,
            )

        expected = self._bound_generation
        target_pid = (
            expected.pid
            if expected is not None
            else (self._pid if self._pid is not None else os.getpid())
        )

        # 1. Pre-scan root generation verification.
        pre_res = self._read_root_generation_and_stat(target_pid, deadline_ns=deadline_ns)
        after_pre_ns = int(self._monotonic_ns())
        if isinstance(pre_res, LinuxMemoryUnavailable):
            return self._unavailable_reading(
                pre_res,
                read_start_ns=read_start_ns,
                read_end_ns=after_pre_ns,
                target_generation=expected,
                observed_before=None,
                observed_after=None,
                clock_ticks_per_second=clk_tck,
            )
        gen_before, root_snap = pre_res
        if root_snap.state in _DEAD_OR_ZOMBIE_STATES:
            exited = LinuxMemoryUnavailable(
                reason=f"target pid {target_pid} is zombie/dead (state={root_snap.state!r})",
                reason_code="process_exited",
                status="unavailable",
            )
            return self._unavailable_reading(
                exited,
                read_start_ns=read_start_ns,
                read_end_ns=after_pre_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
                clock_ticks_per_second=clk_tck,
            )
        if after_pre_ns > deadline_ns:
            timeout_issue = LinuxMemoryUnavailable(
                reason=(
                    f"pre-scan root generation check exceeded max_scan_seconds "
                    f"({self._max_scan_seconds}s)"
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
                clock_ticks_per_second=clk_tck,
            )

        if expected is None:
            expected = gen_before
            self._bound_generation = gen_before
            self._bind_error = None

        pre_mismatch = _generation_exact_mismatch(expected, gen_before)
        if pre_mismatch is not None:
            changed = LinuxMemoryUnavailable(
                reason=f"root target generation changed before scan: {pre_mismatch}",
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
                clock_ticks_per_second=clk_tck,
            )

        # 2. Bounded enumeration of numeric PID entries under proc_root.
        try:
            raw_entries = os.listdir(self._proc_root)
        except PermissionError as exc:
            return self._unavailable_reading(
                LinuxMemoryUnavailable(
                    reason=f"permission denied listing {self._proc_root}: {exc}",
                    reason_code="permission_denied",
                    status="unavailable",
                ),
                read_start_ns=read_start_ns,
                read_end_ns=int(self._monotonic_ns()),
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
                clock_ticks_per_second=clk_tck,
            )
        except FileNotFoundError as exc:
            return self._unavailable_reading(
                LinuxMemoryUnavailable(
                    reason=f"procfs root {self._proc_root} missing: {exc}",
                    reason_code="procfs_missing",
                    status="unavailable",
                ),
                read_start_ns=read_start_ns,
                read_end_ns=int(self._monotonic_ns()),
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
                clock_ticks_per_second=clk_tck,
            )
        except OSError as exc:
            return self._unavailable_reading(
                LinuxMemoryUnavailable(
                    reason=f"OS error listing {self._proc_root}: {exc}",
                    reason_code="procfs_read_error",
                    status="error",
                ),
                read_start_ns=read_start_ns,
                read_end_ns=int(self._monotonic_ns()),
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
                clock_ticks_per_second=clk_tck,
            )

        candidate_pids = sorted(
            int(name)
            for name in raw_entries
            if _is_strict_non_negative_ascii_int(name)
            and int(name) >= 1
            and int(name) != target_pid
        )

        partial_reasons: list[str] = []
        membership_complete = True
        enumerated_candidates = 0
        permission_denied_count = 0
        excluded_count = 0
        snapshots: dict[int, ProcStatSnapshot] = {}

        for cand_pid in candidate_pids:
            if enumerated_candidates >= self._max_enumerated_candidates:
                membership_complete = False
                _append_unique(partial_reasons, "max_enumerated_candidates_exceeded")
                break
            if int(self._monotonic_ns()) > deadline_ns:
                membership_complete = False
                _append_unique(partial_reasons, "max_scan_seconds_exceeded")
                break

            enumerated_candidates += 1
            cand_dir = self._proc_root / str(cand_pid)
            try:
                cand_st = os.lstat(cand_dir)
            except PermissionError:
                permission_denied_count += 1
                membership_complete = False
                _append_unique(partial_reasons, "permission_denied")
                continue
            except (FileNotFoundError, ProcessLookupError):
                excluded_count += 1
                continue
            except OSError:
                excluded_count += 1
                membership_complete = False
                _append_unique(partial_reasons, "procfs_read_error")
                continue

            if stat.S_ISLNK(cand_st.st_mode) or not stat.S_ISDIR(cand_st.st_mode):
                excluded_count += 1
                membership_complete = False
                _append_unique(partial_reasons, "unsafe_procfs_file")
                continue

            stat_bytes = _read_bounded_regular_bytes(
                cand_dir / "stat",
                root_path=self._proc_root,
                max_read_bytes=self._max_read_bytes,
                max_read_seconds=self._max_scan_seconds,
                deadline_ns=deadline_ns,
                monotonic_ns=self._monotonic_ns,
                missing_reason_code="process_exited",
                root_missing_reason_code="procfs_missing",
            )
            if isinstance(stat_bytes, LinuxMemoryUnavailable):
                if stat_bytes.reason_code == "read_timeout":
                    membership_complete = False
                    _append_unique(partial_reasons, "max_scan_seconds_exceeded")
                    break
                if stat_bytes.reason_code == "permission_denied":
                    permission_denied_count += 1
                    membership_complete = False
                    _append_unique(partial_reasons, "permission_denied")
                elif stat_bytes.reason_code == "process_exited":
                    excluded_count += 1
                else:
                    excluded_count += 1
                    membership_complete = False
                    _append_unique(partial_reasons, stat_bytes.reason_code)
                continue

            snap = parse_proc_stat_fields(stat_bytes, expected_pid=cand_pid)
            if isinstance(snap, LinuxMemoryUnavailable):
                excluded_count += 1
                membership_complete = False
                _append_unique(partial_reasons, "malformed_procfs")
                continue

            if snap.state in _DEAD_OR_ZOMBIE_STATES:
                excluded_count += 1
                continue

            snapshots[cand_pid] = snap

        # 3. Traverse verified descendant tree via BFS from target_pid.
        children_by_ppid: dict[int, list[int]] = {}
        for cpid, csnap in snapshots.items():
            children_by_ppid.setdefault(csnap.ppid, []).append(cpid)
        for child_list in children_by_ppid.values():
            child_list.sort()

        visited: set[int] = {target_pid}
        parent_start_ticks: dict[int, int] = {target_pid: root_snap.start_ticks}
        bfs: deque[int] = deque(children_by_ppid.get(target_pid, ()))
        verified_members_raw: list[
            tuple[Any, ProcStatSnapshot, Any]
        ] = []  # (LinuxProcessGeneration, verified_stat, rss_bytes)
        accepted_pids: set[int] = set()

        while bfs:
            if int(self._monotonic_ns()) > deadline_ns:
                membership_complete = False
                _append_unique(partial_reasons, "max_scan_seconds_exceeded")
                break

            child_pid = bfs.popleft()
            if child_pid in visited:
                continue
            visited.add(child_pid)

            child_snap = snapshots[child_pid]
            # Causality / PID-reuse check: a descendant cannot predate its parent.
            p_ticks = parent_start_ticks.get(child_snap.ppid)
            if p_ticks is None or child_snap.start_ticks < p_ticks:
                excluded_count += 1
                continue

            if len(verified_members_raw) >= self._max_observed_members:
                membership_complete = False
                _append_unique(partial_reasons, "max_observed_members_exceeded")
                break

            child_dir = self._proc_root / str(child_pid)
            # Namespace check when root pid_ns_inode is bound.
            child_ns_before, ns_err = _read_pid_ns_inode(child_dir)
            if ns_err is not None:
                if ns_err.reason_code == "permission_denied":
                    permission_denied_count += 1
                    membership_complete = False
                    _append_unique(partial_reasons, "permission_denied")
                    excluded_count += 1
                else:
                    excluded_count += 1
                    membership_complete = False
                    _append_unique(partial_reasons, ns_err.reason_code)
                continue
            if (
                gen_before.pid_ns_inode is not None
                and child_ns_before != gen_before.pid_ns_inode
            ):
                # Foreign PID namespace -> exclude descendant.
                excluded_count += 1
                continue

            # Bounded read of /proc/<child_pid>/status for VmRSS.
            status_bytes = _read_bounded_regular_bytes(
                child_dir / "status",
                root_path=self._proc_root,
                max_read_bytes=self._max_read_bytes,
                max_read_seconds=self._max_scan_seconds,
                deadline_ns=deadline_ns,
                monotonic_ns=self._monotonic_ns,
                missing_reason_code="process_exited",
                root_missing_reason_code="procfs_missing",
            )
            if isinstance(status_bytes, LinuxMemoryUnavailable):
                if status_bytes.reason_code == "read_timeout":
                    membership_complete = False
                    _append_unique(partial_reasons, "max_scan_seconds_exceeded")
                    break
                if status_bytes.reason_code == "process_exited":
                    excluded_count += 1
                    continue
                if status_bytes.reason_code == "permission_denied":
                    permission_denied_count += 1
                    membership_complete = False
                    _append_unique(partial_reasons, "permission_denied")
                    rss_val: Any = status_bytes
                else:
                    excluded_count += 1
                    membership_complete = False
                    _append_unique(partial_reasons, status_bytes.reason_code)
                    continue
            else:
                parsed_status = parse_proc_status_memory(status_bytes)
                if isinstance(parsed_status, LinuxMemoryUnavailable):
                    rss_val = parsed_status
                else:
                    rss_val = parsed_status[0]

            # Post-status re-verification of /proc/<child_pid>/stat and ns/pid to
            # exclude processes that exited, became zombie, reparented, or reused
            # PID between stat and status reads.
            stat_after_bytes = _read_bounded_regular_bytes(
                child_dir / "stat",
                root_path=self._proc_root,
                max_read_bytes=self._max_read_bytes,
                max_read_seconds=self._max_scan_seconds,
                deadline_ns=deadline_ns,
                monotonic_ns=self._monotonic_ns,
                missing_reason_code="process_exited",
                root_missing_reason_code="procfs_missing",
            )
            if isinstance(stat_after_bytes, LinuxMemoryUnavailable):
                if stat_after_bytes.reason_code == "read_timeout":
                    membership_complete = False
                    _append_unique(partial_reasons, "max_scan_seconds_exceeded")
                    break
                if stat_after_bytes.reason_code == "permission_denied":
                    permission_denied_count += 1
                    membership_complete = False
                    _append_unique(partial_reasons, "permission_denied")
                excluded_count += 1
                continue

            snap_after = parse_proc_stat_fields(stat_after_bytes, expected_pid=child_pid)
            if isinstance(snap_after, LinuxMemoryUnavailable):
                excluded_count += 1
                membership_complete = False
                _append_unique(partial_reasons, "malformed_procfs")
                continue

            if (
                snap_after.state in _DEAD_OR_ZOMBIE_STATES
                or snap_after.start_ticks != child_snap.start_ticks
                or snap_after.ppid != child_snap.ppid
            ):
                # Exited, zombified, reparented, or PID-reused mid-read.
                excluded_count += 1
                continue

            child_ns_after, ns_after_err = _read_pid_ns_inode(child_dir)
            if ns_after_err is not None or child_ns_after != child_ns_before:
                excluded_count += 1
                continue

            member_gen = LinuxProcessGeneration(
                pid=child_pid,
                start_ticks=snap_after.start_ticks,
                boot_id=gen_before.boot_id,
                pid_ns_inode=child_ns_after,
                host_id=gen_before.host_id,
            )
            verified_members_raw.append((member_gen, snap_after, rss_val))
            accepted_pids.add(child_pid)
            parent_start_ticks[child_pid] = snap_after.start_ticks
            for grandchild_pid in children_by_ppid.get(child_pid, ()):
                if grandchild_pid not in visited:
                    bfs.append(grandchild_pid)

        # Count unrelated/foreign processes that were parsed in snapshots but never
        # attached to the root's descendant tree.
        foreign_count = sum(1 for cpid in snapshots if cpid not in visited)
        excluded_count += foreign_count

        # 4. Post-scan root generation verification before accepting any values
        # or mutating per-member CPU baselines.
        # Use a small verification deadline if scan cap stopped the loop cleanly
        # so root verification still validates generation without clobbering
        # partial cap results with read_timeout.
        post_deadline_ns = (
            max(deadline_ns, int(self._monotonic_ns()) + self._max_scan_ns)
            if "max_scan_seconds_exceeded" in partial_reasons
            else deadline_ns
        )
        post_res = self._read_root_generation_and_stat(
            target_pid, deadline_ns=post_deadline_ns
        )
        read_end_ns = max(int(self._monotonic_ns()), read_start_ns)
        if isinstance(post_res, LinuxMemoryUnavailable):
            return self._unavailable_reading(
                post_res,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
                clock_ticks_per_second=clk_tck,
                partial_reasons=tuple(partial_reasons),
                enumerated_candidates=enumerated_candidates,
                permission_denied_count=permission_denied_count,
                excluded_count=excluded_count,
            )
        gen_after, root_snap_after = post_res
        if root_snap_after.state in _DEAD_OR_ZOMBIE_STATES:
            exited = LinuxMemoryUnavailable(
                reason=(
                    f"target pid {target_pid} became zombie/dead during scan "
                    f"(state={root_snap_after.state!r})"
                ),
                reason_code="process_exited",
                status="unavailable",
            )
            return self._unavailable_reading(
                exited,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=gen_after,
                clock_ticks_per_second=clk_tck,
                partial_reasons=tuple(partial_reasons),
                enumerated_candidates=enumerated_candidates,
                permission_denied_count=permission_denied_count,
                excluded_count=excluded_count,
            )
        post_mismatch = _generation_exact_mismatch(
            expected, gen_after
        ) or _generation_exact_mismatch(gen_before, gen_after)
        if post_mismatch is not None:
            changed = LinuxMemoryUnavailable(
                reason=f"root target generation changed during scan: {post_mismatch}",
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
                clock_ticks_per_second=clk_tck,
                partial_reasons=tuple(partial_reasons),
                enumerated_candidates=enumerated_candidates,
                permission_denied_count=permission_denied_count,
                excluded_count=excluded_count,
            )

        # 5. Compute per-member CPU deltas from matched-generation baselines.
        prev_baselines = self._member_baselines
        had_baseline = self._has_baseline_scan
        new_baselines: dict[
            tuple[int, int, str | None, str | None, str | None], tuple[int, int]
        ] = {}
        members_out: list[LinuxProcessTreeMember] = []
        matched_delta_count = 0
        regression_count = 0
        sum_user_delta = 0.0
        sum_sys_delta = 0.0
        rss_total = 0
        rss_issues: list[Any] = []

        for member_gen, snap_after, rss_val in verified_members_raw:
            gen_key = (
                member_gen.pid,
                member_gen.start_ticks,
                member_gen.boot_id,
                member_gen.pid_ns_inode,
                member_gen.host_id,
            )
            new_baselines[gen_key] = (snap_after.utime_ticks, snap_after.stime_ticks)
            user_s = snap_after.utime_ticks / float(clk_tck)
            sys_s = snap_after.stime_ticks / float(clk_tck)

            delta_matched = False
            u_delta: float | None = None
            s_delta: float | None = None
            if had_baseline and gen_key in prev_baselines:
                prev_u, prev_s = prev_baselines[gen_key]
                if snap_after.utime_ticks < prev_u or snap_after.stime_ticks < prev_s:
                    # Counter regression breaks this member's delta interval.
                    regression_count += 1
                    _append_unique(partial_reasons, "counter_regression")
                else:
                    u_delta = (snap_after.utime_ticks - prev_u) / float(clk_tck)
                    s_delta = (snap_after.stime_ticks - prev_s) / float(clk_tck)
                    delta_matched = True
                    matched_delta_count += 1
                    sum_user_delta += u_delta
                    sum_sys_delta += s_delta

            if isinstance(rss_val, LinuxMemoryUnavailable):
                rss_issues.append(rss_val)
            else:
                rss_total += int(rss_val)

            role = "observer" if member_gen.pid in self._observer_pids else "descendant"
            members_out.append(
                LinuxProcessTreeMember(
                    generation=member_gen,
                    ppid=snap_after.ppid,
                    role=role,
                    state=snap_after.state,
                    rss_bytes=rss_val,
                    utime_ticks=snap_after.utime_ticks,
                    stime_ticks=snap_after.stime_ticks,
                    user_cpu_seconds=user_s,
                    system_cpu_seconds=sys_s,
                    delta_matched=delta_matched,
                    cpu_user_delta_seconds=u_delta,
                    cpu_system_delta_seconds=s_delta,
                )
            )

        observed_count = len(members_out)
        reasons_tuple = tuple(partial_reasons)

        # If zero members were observed AND membership is incomplete (e.g.
        # permission_denied or scan timeout before finding any member), do NOT
        # report 0 members or 0 RSS as verified zero, and reset baselines so a
        # subsequent healthy scan cannot fabricate a 0.0 CPU delta or stitch
        # across an unavailable gap.
        if observed_count == 0 and not membership_complete:
            self._member_baselines = {}
            self._has_baseline_scan = False
            primary_code = reasons_tuple[0] if reasons_tuple else "partial_scan"
            issue = LinuxMemoryUnavailable(
                reason=(
                    f"descendant scan incomplete with zero observed members "
                    f"({', '.join(reasons_tuple) or primary_code})"
                ),
                reason_code=primary_code,
                status="unavailable",
            )
            return self._unavailable_reading(
                issue,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=gen_after,
                clock_ticks_per_second=clk_tck,
                partial_reasons=reasons_tuple,
                enumerated_candidates=enumerated_candidates,
                permission_denied_count=permission_denied_count,
                excluded_count=excluded_count,
            )

        # Commit baselines only after post-scan root verification succeeded and
        # the scan produced a usable (non-unavailable) observation.
        self._member_baselines = new_baselines
        self._has_baseline_scan = True

        live_member_val: Any = observed_count
        if rss_issues:
            first_rss_issue = rss_issues[0]
            sum_rss_val: Any = LinuxMemoryUnavailable(
                reason=(
                    f"sum_rss_bytes incomplete: {len(rss_issues)} observed member(s) "
                    f"had unreadable VmRSS ({first_rss_issue.reason})"
                ),
                reason_code=first_rss_issue.reason_code,
                status=first_rss_issue.status,
            )
        else:
            sum_rss_val = rss_total

        if not had_baseline:
            pending_issue = LinuxMemoryUnavailable(
                reason="initial scan establishes per-member descendant CPU baseline",
                reason_code="baseline_pending",
                status="unavailable",
            )
            cpu_user_val: Any = pending_issue
            cpu_sys_val: Any = pending_issue
        elif observed_count == 0:
            # membership_complete is True here -> legitimately zero descendants.
            cpu_user_val = 0.0
            cpu_sys_val = 0.0
        elif matched_delta_count == 0:
            if regression_count > 0:
                reg_issue = LinuxMemoryUnavailable(
                    reason="descendant CPU counter regression broke matched delta interval",
                    reason_code="counter_regression",
                    status="unavailable",
                )
                cpu_user_val = reg_issue
                cpu_sys_val = reg_issue
            else:
                unmatched_issue = LinuxMemoryUnavailable(
                    reason=(
                        "no observed descendants matched prior-scan generations "
                        "for CPU delta computation"
                    ),
                    reason_code="baseline_pending",
                    status="unavailable",
                )
                cpu_user_val = unmatched_issue
                cpu_sys_val = unmatched_issue
        else:
            cpu_user_val = sum_user_delta
            cpu_sys_val = sum_sys_delta

        field_issues = [
            v
            for v in (live_member_val, sum_rss_val, cpu_user_val, cpu_sys_val)
            if isinstance(v, LinuxMemoryUnavailable)
        ]

        if not field_issues and membership_complete and not reasons_tuple:
            return LinuxProcessTreeReading(
                status="ok",
                live_member_count=live_member_val,
                sum_rss_bytes=sum_rss_val,
                matched_cpu_user_delta_seconds=cpu_user_val,
                matched_cpu_system_delta_seconds=cpu_sys_val,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                members=tuple(members_out),
                membership_complete=True,
                partial_reasons=(),
                enumerated_candidates=enumerated_candidates,
                observed_member_count=observed_count,
                matched_delta_member_count=matched_delta_count,
                permission_denied_count=permission_denied_count,
                excluded_count=excluded_count,
                clock_ticks_per_second=clk_tck,
                target_generation=expected,
                observed_generation_before=gen_before,
                observed_generation_after=gen_after,
            )

        issue_codes: list[str] = list(reasons_tuple)
        detail_parts: list[str] = []
        if reasons_tuple:
            detail_parts.append(f"partial_reasons={', '.join(reasons_tuple)}")
        for issue in field_issues:
            _append_unique(issue_codes, issue.reason_code)
            _append_unique(detail_parts, issue.reason)
        summary_code = issue_codes[0] if len(issue_codes) == 1 else "partial_scan"
        return LinuxProcessTreeReading(
            status="partial",
            live_member_count=live_member_val,
            sum_rss_bytes=sum_rss_val,
            matched_cpu_user_delta_seconds=cpu_user_val,
            matched_cpu_system_delta_seconds=cpu_sys_val,
            read_start_ns=read_start_ns,
            read_end_ns=read_end_ns,
            members=tuple(members_out),
            membership_complete=membership_complete,
            partial_reasons=reasons_tuple,
            enumerated_candidates=enumerated_candidates,
            observed_member_count=observed_count,
            matched_delta_member_count=matched_delta_count,
            permission_denied_count=permission_denied_count,
            excluded_count=excluded_count,
            clock_ticks_per_second=clk_tck,
            reason_code=summary_code,
            detail="; ".join(detail_parts) or summary_code,
            target_generation=expected,
            observed_generation_before=gen_before,
            observed_generation_after=gen_after,
        )


# ---------------------------------------------------------------------------
# Pure Linux cgroup v2 parsers
# ---------------------------------------------------------------------------


def _decode_utf8_or_error(
    raw: str | bytes,
    *,
    context: str,
    reason_code: str,
) -> str | Any:
    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return LinuxMemoryUnavailable(
                reason=f"malformed {context}: invalid UTF-8 bytes",
                reason_code=reason_code,
                status="error",
            )
    if isinstance(raw, str):
        return raw
    return LinuxMemoryUnavailable(
        reason=f"malformed {context}: unexpected type {type(raw).__name__}",
        reason_code=reason_code,
        status="error",
    )


def _unescape_mountinfo_field(text: str) -> str:
    return _MOUNTINFO_OCTAL_RE.sub(lambda m: chr(int(m.group(1), 8)), text)


def _validate_cgroup_rel_path(raw_path: str) -> tuple[str, tuple[str, ...]] | Any:
    """Validate cgroup v2 relative path strictly against traversal or unsafe chars."""

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    if "\x00" in raw_path or "\\" in raw_path:
        return LinuxMemoryUnavailable(
            reason=f"unsafe cgroup path {raw_path!r}: contains NUL or backslash",
            reason_code="unsafe_cgroup_path",
            status="error",
        )
    if not raw_path.startswith("/"):
        return LinuxMemoryUnavailable(
            reason=f"unsafe cgroup path {raw_path!r}: must start with '/'",
            reason_code="unsafe_cgroup_path",
            status="error",
        )
    if raw_path == "/":
        return ("/", ())
    trimmed = raw_path[1:]
    segments = trimmed.split("/")
    for seg in segments:
        if seg in ("", ".", ".."):
            return LinuxMemoryUnavailable(
                reason=(
                    f"unsafe cgroup path {raw_path!r}: contains traversal or empty segment"
                ),
                reason_code="unsafe_cgroup_path",
                status="error",
            )
    normalized = "/" + "/".join(segments)
    return (normalized, tuple(segments))


def parse_proc_cgroup_v2_path(raw: str | bytes) -> str | Any:
    """Parse the unified cgroup v2 path ``0::<rel_path>`` from ``/proc/<pid>/cgroup``.

    Refuses cgroup v1 entries pretending to be v2 (returns
    ``cgroup_v2_not_mounted_or_v1_only``) and rejects path traversal segments.
    """

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    text = _decode_utf8_or_error(
        raw, context="/proc/<pid>/cgroup", reason_code="malformed_procfs"
    )
    if isinstance(text, LinuxMemoryUnavailable):
        return text

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return LinuxMemoryUnavailable(
            reason="/proc/<pid>/cgroup is empty; cgroup v2 not present",
            reason_code="cgroup_v2_not_mounted_or_v1_only",
            status="unavailable",
        )

    v2_paths: list[str] = []
    saw_v1 = False
    for line in nonempty_lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            return LinuxMemoryUnavailable(
                reason=f"malformed /proc/<pid>/cgroup line {line!r}",
                reason_code="malformed_procfs",
                status="error",
            )
        hierarchy_id, controllers, cg_path = parts
        if not _is_strict_non_negative_ascii_int(hierarchy_id):
            return LinuxMemoryUnavailable(
                reason=f"malformed /proc/<pid>/cgroup hierarchy id {hierarchy_id!r}",
                reason_code="malformed_procfs",
                status="error",
            )
        if hierarchy_id == "0" and controllers == "":
            v2_paths.append(cg_path)
        else:
            saw_v1 = True

    if not v2_paths:
        detail = (
            "only cgroup v1 hierarchy entries found; cgroup v2 unified hierarchy absent"
            if saw_v1
            else "no cgroup v2 (0::) hierarchy entry found"
        )
        return LinuxMemoryUnavailable(
            reason=detail,
            reason_code="cgroup_v2_not_mounted_or_v1_only",
            status="unavailable",
        )
    if len(v2_paths) > 1:
        return LinuxMemoryUnavailable(
            reason="malformed /proc/<pid>/cgroup: duplicate 0:: cgroup v2 entries",
            reason_code="malformed_procfs",
            status="error",
        )

    validated = _validate_cgroup_rel_path(v2_paths[0])
    if isinstance(validated, LinuxMemoryUnavailable):
        return validated
    return validated[0]


def _consume_mountinfo_line_bytes(
    line_bytes: bytes,
    root_mounts: list[str],
    other_mounts: list[str],
) -> Any:
    """Parse one raw ``/proc/<pid>/mountinfo`` line at byte level.

    Skips blank lines and non-``cgroup2`` mounts immediately without UTF-8
    decoding unrelated mount paths so raw non-UTF-8 bytes in unrelated
    NFS/autofs mounts never break ``cgroup2`` discovery. Validates ``cgroup2``
    mount points strictly and appends ``mount_fs_root == "/"`` entries to
    ``root_mounts`` and other entries to ``other_mounts``.
    """

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    stripped = line_bytes.strip()
    if not stripped:
        return None

    left, sep, right = stripped.partition(b" - ")
    if not sep:
        return LinuxMemoryUnavailable(
            reason="malformed /proc/<pid>/mountinfo: missing ' - ' separator",
            reason_code="malformed_procfs",
            status="error",
        )
    left_fields = left.split()
    right_fields = right.split()
    if len(left_fields) < 5 or len(right_fields) < 2:
        return LinuxMemoryUnavailable(
            reason="malformed /proc/<pid>/mountinfo: insufficient fields",
            reason_code="malformed_procfs",
            status="error",
        )
    if right_fields[0] != b"cgroup2":
        return None

    try:
        raw_fs_root = left_fields[3].decode("utf-8")
        raw_mount_point = left_fields[4].decode("utf-8")
    except UnicodeDecodeError:
        return LinuxMemoryUnavailable(
            reason="malformed /proc/<pid>/mountinfo: invalid UTF-8 bytes in cgroup2 entry",
            reason_code="malformed_procfs",
            status="error",
        )

    mount_fs_root = _unescape_mountinfo_field(raw_fs_root)
    mount_point = _unescape_mountinfo_field(raw_mount_point)
    if "\x00" in mount_point or not mount_point.startswith("/"):
        return LinuxMemoryUnavailable(
            reason=f"unsafe cgroup2 mount point {mount_point!r} in mountinfo",
            reason_code="unsafe_cgroup_path",
            status="error",
        )
    mp_parts = [seg for seg in mount_point.split("/") if seg]
    if any(seg in (".", "..") for seg in mp_parts):
        return LinuxMemoryUnavailable(
            reason=f"unsafe cgroup2 mount point traversal {mount_point!r} in mountinfo",
            reason_code="unsafe_cgroup_path",
            status="error",
        )
    if mount_fs_root == "/":
        _append_unique(root_mounts, mount_point)
    else:
        _append_unique(other_mounts, mount_point)
    return None


def parse_proc_mountinfo_cgroup2(raw: str | bytes) -> tuple[str, ...] | Any:
    """Parse ``cgroup2`` mount points from ``/proc/<pid>/mountinfo``.

    Inspects entries strictly past the ``' - '`` separator for
    ``fs_type == "cgroup2"`` (never accepting cgroup v1 ``fs_type == "cgroup"``).
    Unrelated mount entries are skipped at the byte level without UTF-8
    decoding so raw non-UTF-8 bytes in unrelated NFS/autofs mounts never break
    ``cgroup2`` discovery. Entries rooted at ``'/'`` inside the cgroup2
    filesystem are ordered first.
    """

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    if isinstance(raw, bytes):
        raw_bytes = raw
    elif isinstance(raw, str):
        try:
            raw_bytes = raw.encode("utf-8")
        except UnicodeEncodeError:
            return LinuxMemoryUnavailable(
                reason="malformed /proc/<pid>/mountinfo: invalid UTF-8 characters",
                reason_code="malformed_procfs",
                status="error",
            )
    else:
        return LinuxMemoryUnavailable(
            reason=(
                f"malformed /proc/<pid>/mountinfo: unexpected type "
                f"{type(raw).__name__}"
            ),
            reason_code="malformed_procfs",
            status="error",
        )

    root_mounts: list[str] = []
    other_mounts: list[str] = []
    saw_nonempty = False
    for line_bytes in raw_bytes.split(b"\n"):
        if line_bytes.strip():
            saw_nonempty = True
        err = _consume_mountinfo_line_bytes(line_bytes, root_mounts, other_mounts)
        if err is not None:
            return err

    if not saw_nonempty:
        return LinuxMemoryUnavailable(
            reason="/proc/<pid>/mountinfo is empty; no cgroup2 mount found",
            reason_code="cgroup_v2_not_mounted_or_v1_only",
            status="unavailable",
        )
    ordered = tuple(root_mounts + [m for m in other_mounts if m not in root_mounts])
    if not ordered:
        return LinuxMemoryUnavailable(
            reason="no cgroup2 filesystem mount found in mountinfo",
            reason_code="cgroup_v2_not_mounted_or_v1_only",
            status="unavailable",
        )
    return ordered


def _read_bounded_mountinfo_cgroup2(
    path: Path,
    *,
    root_path: Path,
    max_mountinfo_bytes: int,
    max_read_seconds: float,
    deadline_ns: int,
    monotonic_ns: Callable[[], int],
    missing_reason_code: str,
    root_missing_reason_code: str,
    unsafe_reason_code: str = "unsafe_procfs_file",
    read_error_reason_code: str = "procfs_read_error",
) -> tuple[str, ...] | Any:
    """Stream ``/proc/<pid>/mountinfo`` in bounded chunks to extract ``cgroup2`` mounts.

    Refuses symlinks (``S_ISLNK``) and non-regular files (FIFOs, sockets,
    devices) via ``os.lstat`` before ``open`` and ``os.fstat`` after ``open``.
    Streams in :data:`_READ_CHUNK_BYTES` (16 KiB) chunks up to
    ``max_mountinfo_bytes + 1``, keeping peak memory ``O(one line + matched
    cgroup2 mounts)`` rather than buffering large NFS/autofs mount tables.
    """

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    root_mounts: list[str] = []
    other_mounts: list[str] = []
    saw_nonempty = False

    try:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            return LinuxMemoryUnavailable(
                reason=f"refusing symlinked file {path}",
                reason_code=unsafe_reason_code,
                status="error",
            )
        if not stat.S_ISREG(st.st_mode):
            return LinuxMemoryUnavailable(
                reason=f"refusing non-regular file {path}",
                reason_code=unsafe_reason_code,
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
                            reason=f"refusing non-regular opened file {path}",
                            reason_code=unsafe_reason_code,
                            status="error",
                        )
            pending = b""
            total = 0
            limit = max_mountinfo_bytes + 1
            while total < limit:
                if int(monotonic_ns()) > deadline_ns:
                    return LinuxMemoryUnavailable(
                        reason=(
                            f"reading {path} exceeded time bound "
                            f"({max_read_seconds}s)"
                        ),
                        reason_code="read_timeout",
                        status="error",
                    )
                chunk = handle.read(min(limit - total, _READ_CHUNK_BYTES))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_mountinfo_bytes:
                    return LinuxMemoryUnavailable(
                        reason=(
                            f"file {path} exceeded max_mountinfo_bytes "
                            f"({max_mountinfo_bytes})"
                        ),
                        reason_code="input_limit_exceeded",
                        status="error",
                    )
                data = pending + chunk
                lines = data.split(b"\n")
                pending = lines.pop()
                for line_bytes in lines:
                    if line_bytes.strip():
                        saw_nonempty = True
                    err = _consume_mountinfo_line_bytes(
                        line_bytes, root_mounts, other_mounts
                    )
                    if err is not None:
                        return err
            if pending:
                if pending.strip():
                    saw_nonempty = True
                err = _consume_mountinfo_line_bytes(
                    pending, root_mounts, other_mounts
                )
                if err is not None:
                    return err
    except PermissionError as exc:
        return LinuxMemoryUnavailable(
            reason=f"permission denied reading {path}: {exc}",
            reason_code="permission_denied",
            status="unavailable",
        )
    except (FileNotFoundError, ProcessLookupError) as exc:
        if not root_path.exists():
            return LinuxMemoryUnavailable(
                reason=f"root directory {root_path} is missing: {exc}",
                reason_code=root_missing_reason_code,
                status="unavailable",
            )
        return LinuxMemoryUnavailable(
            reason=f"path {path} not found: {exc}",
            reason_code=missing_reason_code,
            status="unavailable",
        )
    except OSError as exc:
        return LinuxMemoryUnavailable(
            reason=f"OS error reading {path}: {type(exc).__name__}: {exc}",
            reason_code=read_error_reason_code,
            status="error",
        )

    if not saw_nonempty:
        return LinuxMemoryUnavailable(
            reason="/proc/<pid>/mountinfo is empty; no cgroup2 mount found",
            reason_code="cgroup_v2_not_mounted_or_v1_only",
            status="unavailable",
        )
    ordered = tuple(root_mounts + [m for m in other_mounts if m not in root_mounts])
    if not ordered:
        return LinuxMemoryUnavailable(
            reason="no cgroup2 filesystem mount found in mountinfo",
            reason_code="cgroup_v2_not_mounted_or_v1_only",
            status="unavailable",
        )
    return ordered


def parse_cgroup_uint(raw: str | bytes, *, field_name: str) -> int | Any:
    """Parse a strict non-negative integer cgroup v2 scalar (preserving ``0``)."""

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    text = _decode_utf8_or_error(
        raw, context=field_name, reason_code="malformed_cgroup"
    )
    if isinstance(text, LinuxMemoryUnavailable):
        return text
    token = text.strip()
    if not _is_strict_non_negative_ascii_int(token):
        return LinuxMemoryUnavailable(
            reason=f"malformed {field_name} value {token!r}",
            reason_code="malformed_cgroup",
            status="error",
        )
    return int(token)


def parse_cgroup_memory_max(raw: str | bytes) -> int | Any:
    """Parse cgroup v2 ``memory.max`` (``"max"`` -> ``unavailable('unlimited')``)."""

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    text = _decode_utf8_or_error(
        raw, context="memory.max", reason_code="malformed_cgroup"
    )
    if isinstance(text, LinuxMemoryUnavailable):
        return text
    token = text.strip()
    if token == "max":
        return LinuxMemoryUnavailable(
            reason="cgroup memory.max is unlimited ('max')",
            reason_code="unlimited",
            status="unavailable",
        )
    if not _is_strict_non_negative_ascii_int(token):
        return LinuxMemoryUnavailable(
            reason=f"malformed memory.max value {token!r}",
            reason_code="malformed_cgroup",
            status="error",
        )
    return int(token)


def parse_cgroup_cpu_stat(raw: str | bytes) -> dict[str, Any] | Any:
    """Parse ``cpu.stat`` microsecond counters and ``nr_throttled``."""

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    text = _decode_utf8_or_error(raw, context="cpu.stat", reason_code="malformed_cgroup")
    if isinstance(text, LinuxMemoryUnavailable):
        return text

    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty:
        return LinuxMemoryUnavailable(
            reason="malformed cpu.stat: empty content",
            reason_code="malformed_cgroup",
            status="error",
        )

    target_keys = frozenset(
        {"usage_usec", "user_usec", "system_usec", "throttled_usec", "nr_throttled"}
    )
    parsed: dict[str, Any] = {}
    for line in nonempty:
        parts = line.split()
        if len(parts) != 2:
            return LinuxMemoryUnavailable(
                reason=f"malformed cpu.stat line {line!r}",
                reason_code="malformed_cgroup",
                status="error",
            )
        key, val_tok = parts
        if key not in target_keys:
            continue
        if key in parsed:
            parsed[key] = LinuxMemoryUnavailable(
                reason=f"duplicate {key} entry in cpu.stat",
                reason_code="malformed_cgroup",
                status="error",
            )
            continue
        if not _is_strict_non_negative_ascii_int(val_tok):
            parsed[key] = LinuxMemoryUnavailable(
                reason=f"invalid {key} integer value {val_tok!r} in cpu.stat",
                reason_code="malformed_cgroup",
                status="error",
            )
            continue
        ival = int(val_tok)
        if key == "nr_throttled":
            parsed[key] = ival
        else:
            parsed[key] = ival / _USEC_PER_SECOND

    def _get(k: str) -> Any:
        if k in parsed:
            return parsed[k]
        return LinuxMemoryUnavailable(
            reason=f"{k} missing from cpu.stat",
            reason_code="field_missing",
            status="unavailable",
        )

    return {
        "cpu_usage_seconds": _get("usage_usec"),
        "cpu_user_seconds": _get("user_usec"),
        "cpu_system_seconds": _get("system_usec"),
        "cpu_throttled_seconds": _get("throttled_usec"),
        "cpu_nr_throttled": _get("nr_throttled"),
    }


def parse_cgroup_cpu_max(raw: str | bytes) -> float | Any:
    """Parse cgroup v2 ``cpu.max`` (``"$MAX $PERIOD"``) into core-equivalents.

    ``"max <period>"`` yields ``unavailable(reason_code="unlimited")``, never ``0``.
    """

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    text = _decode_utf8_or_error(raw, context="cpu.max", reason_code="malformed_cgroup")
    if isinstance(text, LinuxMemoryUnavailable):
        return text
    parts = text.strip().split()
    if len(parts) != 2:
        return LinuxMemoryUnavailable(
            reason=f"malformed cpu.max value {text.strip()!r} (expected '<max> <period>')",
            reason_code="malformed_cgroup",
            status="error",
        )
    quota_tok, period_tok = parts
    if not _is_strict_non_negative_ascii_int(period_tok) or int(period_tok) <= 0:
        return LinuxMemoryUnavailable(
            reason=f"malformed cpu.max period {period_tok!r} (expected positive integer)",
            reason_code="malformed_cgroup",
            status="error",
        )
    period = int(period_tok)
    if quota_tok == "max":
        return LinuxMemoryUnavailable(
            reason="cgroup cpu.max is unlimited ('max')",
            reason_code="unlimited",
            status="unavailable",
        )
    if not _is_strict_non_negative_ascii_int(quota_tok):
        return LinuxMemoryUnavailable(
            reason=f"malformed cpu.max quota {quota_tok!r}",
            reason_code="malformed_cgroup",
            status="error",
        )
    return float(int(quota_tok) / period)


def parse_cpuset_effective_cpus(raw: str | bytes) -> int | Any:
    """Parse a Linux CPU list (e.g. ``'0-3,6'`` -> ``5``) into an integer CPU count."""

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    text = _decode_utf8_or_error(
        raw, context="cpuset.cpus.effective", reason_code="malformed_cgroup"
    )
    if isinstance(text, LinuxMemoryUnavailable):
        return text
    stripped = text.strip()
    if not stripped:
        return LinuxMemoryUnavailable(
            reason="cpuset.cpus.effective is empty",
            reason_code="field_missing",
            status="unavailable",
        )

    intervals: list[tuple[int, int]] = []
    for part in stripped.split(","):
        tok = part.strip()
        if not tok:
            return LinuxMemoryUnavailable(
                reason=f"malformed cpuset list {stripped!r}: empty item",
                reason_code="malformed_cgroup",
                status="error",
            )
        if "-" in tok:
            bounds = tok.split("-")
            if len(bounds) != 2:
                return LinuxMemoryUnavailable(
                    reason=f"malformed cpuset range {tok!r}",
                    reason_code="malformed_cgroup",
                    status="error",
                )
            lo_s, hi_s = bounds[0].strip(), bounds[1].strip()
            if not (
                _is_strict_non_negative_ascii_int(lo_s)
                and _is_strict_non_negative_ascii_int(hi_s)
            ):
                return LinuxMemoryUnavailable(
                    reason=f"malformed cpuset range bounds {tok!r}",
                    reason_code="malformed_cgroup",
                    status="error",
                )
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo or (hi - lo) > _MAX_CPUSET_SPAN:
                return LinuxMemoryUnavailable(
                    reason=f"invalid cpuset range {tok!r}",
                    reason_code="malformed_cgroup",
                    status="error",
                )
            intervals.append((lo, hi))
        else:
            if not _is_strict_non_negative_ascii_int(tok):
                return LinuxMemoryUnavailable(
                    reason=f"malformed cpuset cpu id {tok!r}",
                    reason_code="malformed_cgroup",
                    status="error",
                )
            cpu_id = int(tok)
            intervals.append((cpu_id, cpu_id))

    # Merge overlapping/adjacent inclusive intervals so duplicates never double-count.
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1] + 1:
            merged.append((lo, hi))
        else:
            prev_lo, prev_hi = merged[-1]
            merged[-1] = (prev_lo, max(prev_hi, hi))
    return sum((hi - lo + 1) for lo, hi in merged)


def parse_cgroup_procs(
    raw: str | bytes,
    *,
    max_procs: int = DEFAULT_LINUX_CGROUP_MAX_PROCS,
) -> tuple[tuple[int, ...], bool] | Any:
    """Parse bounded PID list from ``cgroup.procs`` into ``(pids, truncated)``."""

    pr = _pr()
    LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

    text = _decode_utf8_or_error(
        raw, context="cgroup.procs", reason_code="malformed_cgroup"
    )
    if isinstance(text, LinuxMemoryUnavailable):
        return text

    pids: list[int] = []
    seen: set[int] = set()
    truncated = False
    for raw_line in text.splitlines():
        tok = raw_line.strip()
        if not tok:
            continue
        if not _is_strict_non_negative_ascii_int(tok) or int(tok) < 1:
            return LinuxMemoryUnavailable(
                reason=f"malformed pid {tok!r} in cgroup.procs",
                reason_code="malformed_cgroup",
                status="error",
            )
        pid_val = int(tok)
        if pid_val in seen:
            continue
        if len(pids) >= max_procs:
            truncated = True
            break
        seen.add(pid_val)
        pids.append(pid_val)
    return (tuple(pids), truncated)


# ---------------------------------------------------------------------------
# Read-only Linux cgroup v2 identity, reading, and probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinuxCgroupV2Identity:
    """Resolved cgroup v2 identity, effective limit paths, and co-member scope."""

    mount_root: str
    cgroup_path: str
    cgroup_version: int = 2
    effective_memory_limit_path: str | None = None
    effective_cpu_limit_path: str | None = None
    observed_member_count: int | None = None
    includes_other_tasks: bool | None = None
    shared_scope_not_exclusive: bool = True
    procs_truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.cgroup_version) is not int or self.cgroup_version != 2:
            raise ValueError(
                f"LinuxCgroupV2Identity.cgroup_version must be 2, got {self.cgroup_version!r}"
            )
        if type(self.mount_root) is not str or not self.mount_root.strip():
            raise ValueError("LinuxCgroupV2Identity.mount_root must be a nonempty string")
        if type(self.cgroup_path) is not str or not self.cgroup_path.startswith("/"):
            raise ValueError(
                "LinuxCgroupV2Identity.cgroup_path must be a string starting with '/'"
            )
        if self.effective_memory_limit_path is not None and (
            type(self.effective_memory_limit_path) is not str
            or not self.effective_memory_limit_path.startswith("/")
        ):
            raise ValueError(
                "LinuxCgroupV2Identity.effective_memory_limit_path must start with '/' or be None"
            )
        if self.effective_cpu_limit_path is not None and (
            type(self.effective_cpu_limit_path) is not str
            or not self.effective_cpu_limit_path.startswith("/")
        ):
            raise ValueError(
                "LinuxCgroupV2Identity.effective_cpu_limit_path must start with '/' or be None"
            )
        if self.observed_member_count is not None:
            _require_non_negative_int(
                self.observed_member_count,
                "LinuxCgroupV2Identity.observed_member_count",
            )
        if self.includes_other_tasks is not None and type(self.includes_other_tasks) is not bool:
            raise TypeError(
                "LinuxCgroupV2Identity.includes_other_tasks must be bool or None"
            )
        if self.shared_scope_not_exclusive is not True:
            raise ValueError(
                "LinuxCgroupV2Identity.shared_scope_not_exclusive must always be True"
            )
        if type(self.procs_truncated) is not bool:
            raise TypeError("LinuxCgroupV2Identity.procs_truncated must be bool")

    def to_subject_dict(
        self,
        *,
        target_generation: Any = None,
    ) -> dict[str, Any]:
        """Return an O2 ``tpen.run-observability/v1`` ``cgroup`` subject mapping."""

        subject: dict[str, Any] = {
            "kind": "cgroup",
            "cgroup_version": self.cgroup_version,
            "mount_root": self.mount_root,
            "cgroup_path": self.cgroup_path,
            "effective_memory_limit_path": self.effective_memory_limit_path,
            "effective_cpu_limit_path": self.effective_cpu_limit_path,
            "observed_member_count": self.observed_member_count,
            "includes_other_tasks": self.includes_other_tasks,
            "shared_scope_not_exclusive": True,
        }
        if self.procs_truncated:
            subject["procs_truncated"] = True
        if target_generation is not None:
            subject["target_generation"] = target_generation.to_dict()
        return subject


@dataclass(frozen=True)
class LinuxCgroupV2Reading:
    """Typed read-only Linux cgroup v2 reading with hierarchical limit provenance."""

    status: str
    memory_current_bytes: Any  # int | ResourceUnavailable
    memory_peak_bytes: Any  # int | ResourceUnavailable
    memory_max_bytes: Any  # int | ResourceUnavailable
    cpu_usage_seconds: Any  # float | ResourceUnavailable
    cpu_user_seconds: Any  # float | ResourceUnavailable
    cpu_system_seconds: Any  # float | ResourceUnavailable
    cpu_throttled_seconds: Any  # float | ResourceUnavailable
    cpu_nr_throttled: Any  # int | ResourceUnavailable
    cpu_max_cores: Any  # float | ResourceUnavailable
    cpuset_effective_cpus: Any  # int | ResourceUnavailable
    read_start_ns: int
    read_end_ns: int
    cgroup_identity: LinuxCgroupV2Identity | None = None
    reason_code: str | None = None
    detail: str | None = None
    target_generation: Any = None
    observed_generation_before: Any = None
    observed_generation_after: Any = None

    def __post_init__(self) -> None:
        pr = _pr()
        if type(self.status) is not str or self.status not in LINUX_CGROUP_READING_STATUSES:
            raise ValueError(
                f"LinuxCgroupV2Reading.status must be one of "
                f"{sorted(LINUX_CGROUP_READING_STATUSES)}, got {self.status!r}"
            )
        start = _require_non_negative_int(
            self.read_start_ns, "LinuxCgroupV2Reading.read_start_ns"
        )
        end = _require_non_negative_int(
            self.read_end_ns, "LinuxCgroupV2Reading.read_end_ns"
        )
        if end < start:
            raise ValueError(
                f"LinuxCgroupV2Reading.read_end_ns ({end}) must be >= "
                f"read_start_ns ({start})"
            )
        if self.cgroup_identity is not None and not isinstance(
            self.cgroup_identity, LinuxCgroupV2Identity
        ):
            raise TypeError(
                "LinuxCgroupV2Reading.cgroup_identity must be LinuxCgroupV2Identity or None, "
                f"got {type(self.cgroup_identity).__name__}"
            )

        int_fields = (
            "memory_current_bytes",
            "memory_peak_bytes",
            "memory_max_bytes",
            "cpu_nr_throttled",
            "cpuset_effective_cpus",
        )
        float_fields = (
            "cpu_usage_seconds",
            "cpu_user_seconds",
            "cpu_system_seconds",
            "cpu_throttled_seconds",
            "cpu_max_cores",
        )
        ok_count = 0
        for fname in int_fields:
            val = getattr(self, fname)
            if isinstance(val, pr.ResourceUnavailable):
                continue
            if type(val) is not int or val < 0:
                raise ValueError(
                    f"LinuxCgroupV2Reading.{fname} must be a non-negative int or "
                    f"ResourceUnavailable, got {val!r}"
                )
            ok_count += 1
        for fname in float_fields:
            val = getattr(self, fname)
            if isinstance(val, pr.ResourceUnavailable):
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"LinuxCgroupV2Reading.{fname} must be a finite non-negative number "
                    f"or ResourceUnavailable, got {val!r}"
                )
            num = float(val)
            if not math.isfinite(num) or num < 0.0:
                raise ValueError(
                    f"LinuxCgroupV2Reading.{fname} must be finite and >= 0, got {val!r}"
                )
            object.__setattr__(self, fname, num)
            ok_count += 1

        total_fields = len(int_fields) + len(float_fields)
        if self.status == "ok":
            if ok_count != total_fields:
                raise ValueError(
                    "LinuxCgroupV2Reading with status='ok' requires all 10 fields "
                    "to be valid numbers"
                )
        elif self.status == "partial":
            if ok_count == 0 or ok_count == total_fields:
                raise ValueError(
                    "LinuxCgroupV2Reading with status='partial' requires a mix of "
                    "valid numeric and ResourceUnavailable fields"
                )
            if type(self.reason_code) is not str or not self.reason_code.strip():
                raise ValueError(
                    "LinuxCgroupV2Reading with status='partial' requires reason_code"
                )
        else:
            if ok_count != 0:
                raise ValueError(
                    f"LinuxCgroupV2Reading with status={self.status!r} must not "
                    "carry numeric field values"
                )
            if type(self.reason_code) is not str or not self.reason_code.strip():
                raise ValueError(
                    f"LinuxCgroupV2Reading with status={self.status!r} requires reason_code"
                )

        for gen_name in (
            "target_generation",
            "observed_generation_before",
            "observed_generation_after",
        ):
            gen = getattr(self, gen_name)
            if gen is not None and not isinstance(gen, pr.LinuxProcessGeneration):
                raise TypeError(
                    f"LinuxCgroupV2Reading.{gen_name} must be "
                    f"LinuxProcessGeneration or None, got {type(gen).__name__}"
                )


class LinuxCgroupV2Probe:
    """Strictly read-only Linux cgroup v2 membership, usage, and limit probe.

    Construction is strictly pure and side-effect-free: it performs zero
    filesystem I/O and never freezes ``os.getpid()`` when ``pid=None``.
    Never guesses cgroup paths from scheduler job IDs, never writes or resets
    ``memory.peak``, and refuses cgroup v1 pretending to be v2.
    """

    def __init__(
        self,
        *,
        pid: int | None = None,
        expected_generation: Any = None,
        proc_root: str | Path | None = None,
        cgroup_root: str | Path | None = None,
        max_read_bytes: int = DEFAULT_LINUX_CGROUP_MAX_READ_BYTES,
        max_mountinfo_bytes: int = DEFAULT_LINUX_CGROUP_MAX_MOUNTINFO_BYTES,
        max_read_seconds: float = DEFAULT_LINUX_CGROUP_MAX_READ_SECONDS,
        max_ancestor_depth: int = DEFAULT_LINUX_CGROUP_MAX_ANCESTOR_DEPTH,
        max_procs: int = DEFAULT_LINUX_CGROUP_MAX_PROCS,
        monotonic_ns: Callable[[], int] | None = None,
        platform_name: str | None = None,
        host_id: str | None = None,
    ) -> None:
        pr = _pr()
        if pid is not None:
            pid = _require_positive_int(pid, "LinuxCgroupV2Probe.pid")
        if expected_generation is not None:
            if not isinstance(expected_generation, pr.LinuxProcessGeneration):
                raise TypeError(
                    "LinuxCgroupV2Probe.expected_generation must be "
                    f"LinuxProcessGeneration or None, got {type(expected_generation).__name__}"
                )
            if pid is not None and pid != expected_generation.pid:
                raise ValueError(
                    f"LinuxCgroupV2Probe.pid ({pid}) disagrees with "
                    f"expected_generation.pid ({expected_generation.pid})"
                )

        self._pid: int | None = pid
        self._bound_generation: Any = expected_generation
        self._bind_error: Any = None
        self._last_cgroup_identity: LinuxCgroupV2Identity | None = None
        self._last_cgroup_error: Any = None

        if proc_root is None:
            self._proc_root = DEFAULT_PROC_ROOT
            self._uses_custom_proc_root = False
        elif isinstance(proc_root, Path):
            self._proc_root = proc_root
            self._uses_custom_proc_root = True
        elif type(proc_root) is str:
            if not proc_root.strip():
                raise ValueError(
                    "LinuxCgroupV2Probe.proc_root must be a nonempty path or None"
                )
            self._proc_root = Path(proc_root)
            self._uses_custom_proc_root = True
        else:
            raise TypeError(
                "LinuxCgroupV2Probe.proc_root must be str, Path, or None, "
                f"got {type(proc_root).__name__}"
            )

        if cgroup_root is None:
            self._cgroup_root: Path | None = None
        elif isinstance(cgroup_root, Path):
            self._cgroup_root = cgroup_root
        elif type(cgroup_root) is str:
            if not cgroup_root.strip():
                raise ValueError(
                    "LinuxCgroupV2Probe.cgroup_root must be a nonempty path or None"
                )
            self._cgroup_root = Path(cgroup_root)
        else:
            raise TypeError(
                "LinuxCgroupV2Probe.cgroup_root must be str, Path, or None, "
                f"got {type(cgroup_root).__name__}"
            )

        self._max_read_bytes = _require_positive_int(
            max_read_bytes, "LinuxCgroupV2Probe.max_read_bytes"
        )
        self._max_mountinfo_bytes = _require_positive_int(
            max_mountinfo_bytes, "LinuxCgroupV2Probe.max_mountinfo_bytes"
        )
        self._max_read_seconds = _require_positive_finite_float(
            max_read_seconds, "LinuxCgroupV2Probe.max_read_seconds"
        )
        self._max_read_ns = max(1, int(round(self._max_read_seconds * _NS_PER_SECOND)))
        self._max_ancestor_depth = _require_positive_int(
            max_ancestor_depth, "LinuxCgroupV2Probe.max_ancestor_depth"
        )
        self._max_procs = _require_positive_int(
            max_procs, "LinuxCgroupV2Probe.max_procs"
        )
        if monotonic_ns is not None and not callable(monotonic_ns):
            raise TypeError("LinuxCgroupV2Probe.monotonic_ns must be callable or None")
        self._monotonic_ns: Callable[[], int] = (
            time.monotonic_ns if monotonic_ns is None else monotonic_ns
        )
        self._platform_name = _require_optional_nonempty_str(
            platform_name, "LinuxCgroupV2Probe.platform_name"
        )
        self._host_id = _require_optional_nonempty_str(
            host_id, "LinuxCgroupV2Probe.host_id"
        )

    @property
    def target_generation(self) -> Any:
        return self._bound_generation

    @property
    def bind_error(self) -> Any:
        return self._bind_error

    @property
    def cgroup_identity(self) -> LinuxCgroupV2Identity | None:
        return self._last_cgroup_identity

    @property
    def cgroup_error(self) -> Any:
        return self._last_cgroup_error

    @property
    def max_read_bytes(self) -> int:
        return self._max_read_bytes

    @property
    def max_mountinfo_bytes(self) -> int:
        return self._max_mountinfo_bytes

    @property
    def max_read_seconds(self) -> float:
        return self._max_read_seconds

    @property
    def max_ancestor_depth(self) -> int:
        return self._max_ancestor_depth

    @property
    def proc_root(self) -> Path:
        return self._proc_root

    @property
    def cgroup_root(self) -> Path | None:
        return self._cgroup_root

    def _effective_platform(self) -> str:
        return sys.platform if self._platform_name is None else self._platform_name

    def _resolve_host_id(self) -> str:
        if self._host_id is not None:
            return self._host_id
        return socket.gethostname() or "unknown-host"

    def _check_platform(self) -> Any:
        pr = _pr()
        if (
            self._platform_name is None
            and self._uses_custom_proc_root
            and self._proc_root != DEFAULT_PROC_ROOT
        ):
            return None
        plat = self._effective_platform()
        if not plat.startswith("linux"):
            return pr.LinuxMemoryUnavailable(
                reason=(
                    f"LinuxCgroupV2Probe requires Linux procfs/cgroup2 "
                    f"(current platform is {plat!r}; no non-Linux fallback)"
                ),
                reason_code="unsupported_platform",
                status="unavailable",
            )
        return None

    def _read_generation(
        self,
        pid: int,
        *,
        deadline_ns: int,
    ) -> Any:
        pr = _pr()
        LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

        pid_dir = _check_safe_proc_pid_dir(self._proc_root, pid)
        if isinstance(pid_dir, LinuxMemoryUnavailable):
            return pid_dir

        stat_bytes = _read_bounded_regular_bytes(
            pid_dir / "stat",
            root_path=self._proc_root,
            max_read_bytes=self._max_read_bytes,
            max_read_seconds=self._max_read_seconds,
            deadline_ns=deadline_ns,
            monotonic_ns=self._monotonic_ns,
            missing_reason_code="process_exited",
            root_missing_reason_code="procfs_missing",
        )
        if isinstance(stat_bytes, LinuxMemoryUnavailable):
            return stat_bytes

        snap = parse_proc_stat_fields(stat_bytes, expected_pid=pid)
        if isinstance(snap, LinuxMemoryUnavailable):
            return snap

        boot_id, boot_err = _read_boot_id(
            self._proc_root,
            max_read_bytes=self._max_read_bytes,
            max_read_seconds=self._max_read_seconds,
            deadline_ns=deadline_ns,
            monotonic_ns=self._monotonic_ns,
        )
        if boot_err is not None:
            return boot_err

        pid_ns_inode, ns_err = _read_pid_ns_inode(pid_dir)
        if ns_err is not None and ns_err.reason_code == "unsafe_procfs_file":
            return ns_err

        return pr.LinuxProcessGeneration(
            pid=snap.pid,
            start_ticks=snap.start_ticks,
            boot_id=boot_id,
            pid_ns_inode=pid_ns_inode,
            host_id=self._resolve_host_id(),
        )

    def _read_cgroup_rel_path(self, pid: int, *, deadline_ns: int) -> str | Any:
        pr = _pr()
        LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

        pid_dir = _check_safe_proc_pid_dir(self._proc_root, pid)
        if isinstance(pid_dir, LinuxMemoryUnavailable):
            return pid_dir

        cg_bytes = _read_bounded_regular_bytes(
            pid_dir / "cgroup",
            root_path=self._proc_root,
            max_read_bytes=self._max_read_bytes,
            max_read_seconds=self._max_read_seconds,
            deadline_ns=deadline_ns,
            monotonic_ns=self._monotonic_ns,
            missing_reason_code="process_exited",
            root_missing_reason_code="procfs_missing",
        )
        if isinstance(cg_bytes, LinuxMemoryUnavailable):
            return cg_bytes
        return parse_proc_cgroup_v2_path(cg_bytes)

    def _read_mountinfo_cgroup2_root(
        self,
        pid: int,
        *,
        deadline_ns: int,
    ) -> Path | Any:
        pr = _pr()
        LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

        pid_dir = _check_safe_proc_pid_dir(self._proc_root, pid)
        if isinstance(pid_dir, LinuxMemoryUnavailable):
            return pid_dir

        mount_points = _read_bounded_mountinfo_cgroup2(
            pid_dir / "mountinfo",
            root_path=self._proc_root,
            max_mountinfo_bytes=self._max_mountinfo_bytes,
            max_read_seconds=self._max_read_seconds,
            deadline_ns=deadline_ns,
            monotonic_ns=self._monotonic_ns,
            missing_reason_code="mountinfo_missing",
            root_missing_reason_code="procfs_missing",
        )
        if (
            isinstance(mount_points, LinuxMemoryUnavailable)
            and mount_points.reason_code == "mountinfo_missing"
            and (pid == os.getpid() or self._uses_custom_proc_root)
        ):
            # Safe fallback to /proc/self/mountinfo when appropriate.
            self_mi = self._proc_root / "self" / "mountinfo"
            mount_points = _read_bounded_mountinfo_cgroup2(
                self_mi,
                root_path=self._proc_root,
                max_mountinfo_bytes=self._max_mountinfo_bytes,
                max_read_seconds=self._max_read_seconds,
                deadline_ns=deadline_ns,
                monotonic_ns=self._monotonic_ns,
                missing_reason_code="cgroup_v2_not_mounted_or_v1_only",
                root_missing_reason_code="procfs_missing",
            )
        if isinstance(mount_points, LinuxMemoryUnavailable):
            return mount_points

        if self._cgroup_root is not None:
            return self._cgroup_root
        return Path(mount_points[0])

    def _resolve_contained_cgroup_chain(
        self,
        mount_root: Path,
        rel_path: str,
    ) -> list[tuple[str, Path]] | Any:
        """Validate ``mount_root`` and walk ``rel_path`` with strict no-symlink containment.

        Returns ancestor chain ordered leaf-first:
        ``[(rel_path, leaf_dir), ..., ("/", mount_root)]``.
        """

        pr = _pr()
        LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

        validated = _validate_cgroup_rel_path(rel_path)
        if isinstance(validated, LinuxMemoryUnavailable):
            return validated
        norm_rel, segments = validated

        if len(segments) > self._max_ancestor_depth:
            return LinuxMemoryUnavailable(
                reason=(
                    f"cgroup path {norm_rel!r} depth ({len(segments)}) exceeds "
                    f"max_ancestor_depth ({self._max_ancestor_depth})"
                ),
                reason_code="ancestor_depth_exceeded",
                status="error",
            )

        try:
            root_st = os.lstat(mount_root)
        except PermissionError as exc:
            return LinuxMemoryUnavailable(
                reason=f"permission denied accessing cgroup2 mount {mount_root}: {exc}",
                reason_code="permission_denied",
                status="unavailable",
            )
        except FileNotFoundError as exc:
            return LinuxMemoryUnavailable(
                reason=f"cgroup2 mount root {mount_root} not found: {exc}",
                reason_code="cgroup_v2_not_mounted_or_v1_only",
                status="unavailable",
            )
        except OSError as exc:
            return LinuxMemoryUnavailable(
                reason=f"OS error accessing cgroup2 mount {mount_root}: {exc}",
                reason_code="cgroup_read_error",
                status="error",
            )

        if stat.S_ISLNK(root_st.st_mode) or not stat.S_ISDIR(root_st.st_mode):
            return LinuxMemoryUnavailable(
                reason=f"refusing symlinked or non-directory cgroup2 mount {mount_root}",
                reason_code="unsafe_cgroup_path",
                status="error",
            )

        resolved_mount = mount_root.resolve(strict=False)
        chain_top_down: list[tuple[str, Path]] = [("/", mount_root)]
        cur_dir = mount_root
        acc_segments: list[str] = []
        for seg in segments:
            cur_dir = cur_dir / seg
            acc_segments.append(seg)
            try:
                seg_st = os.lstat(cur_dir)
            except PermissionError as exc:
                return LinuxMemoryUnavailable(
                    reason=f"permission denied accessing cgroup dir {cur_dir}: {exc}",
                    reason_code="permission_denied",
                    status="unavailable",
                )
            except FileNotFoundError as exc:
                return LinuxMemoryUnavailable(
                    reason=f"cgroup directory {cur_dir} not found: {exc}",
                    reason_code="cgroup_path_missing",
                    status="unavailable",
                )
            except OSError as exc:
                return LinuxMemoryUnavailable(
                    reason=f"OS error accessing cgroup dir {cur_dir}: {exc}",
                    reason_code="cgroup_read_error",
                    status="error",
                )
            if stat.S_ISLNK(seg_st.st_mode) or not stat.S_ISDIR(seg_st.st_mode):
                return LinuxMemoryUnavailable(
                    reason=f"refusing symlinked or non-directory cgroup path {cur_dir}",
                    reason_code="unsafe_cgroup_path",
                    status="error",
                )
            resolved_cur = cur_dir.resolve(strict=False)
            try:
                resolved_cur.relative_to(resolved_mount)
            except ValueError:
                return LinuxMemoryUnavailable(
                    reason=(
                        f"cgroup directory {cur_dir} resolves outside mount root "
                        f"{mount_root}"
                    ),
                    reason_code="unsafe_cgroup_path",
                    status="error",
                )
            chain_top_down.append(("/" + "/".join(acc_segments), cur_dir))

        chain_top_down.reverse()  # leaf-first up to mount_root
        return chain_top_down

    def _read_cgroup_file(
        self,
        path: Path,
        *,
        mount_root: Path,
        deadline_ns: int,
    ) -> bytes | Any:
        return _read_bounded_regular_bytes(
            path,
            root_path=mount_root,
            max_read_bytes=self._max_read_bytes,
            max_read_seconds=self._max_read_seconds,
            deadline_ns=deadline_ns,
            monotonic_ns=self._monotonic_ns,
            missing_reason_code="field_missing",
            root_missing_reason_code="cgroup_v2_not_mounted_or_v1_only",
            unsafe_reason_code="unsafe_cgroup_path",
            read_error_reason_code="cgroup_read_error",
            enforce_containment_root=mount_root,
        )

    def _resolve_hierarchical_limit(
        self,
        chain_leaf_first: Sequence[tuple[str, Path]],
        *,
        mount_root: Path,
        filename: str,
        parser: Callable[[bytes], Any],
        deadline_ns: int,
    ) -> tuple[Any, str | None]:
        """Resolve effective minimum hierarchical limit up ``chain_leaf_first``."""

        pr = _pr()
        LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

        best_value: int | float | None = None
        best_path: str | None = None
        saw_unlimited = False

        for rel_path, directory in chain_leaf_first:
            if int(self._monotonic_ns()) > deadline_ns:
                return (
                    LinuxMemoryUnavailable(
                        reason=(
                            f"resolving hierarchical {filename} exceeded "
                            f"max_read_seconds ({self._max_read_seconds}s)"
                        ),
                        reason_code="read_timeout",
                        status="error",
                    ),
                    None,
                )
            raw = self._read_cgroup_file(
                directory / filename,
                mount_root=mount_root,
                deadline_ns=deadline_ns,
            )
            if isinstance(raw, LinuxMemoryUnavailable):
                if raw.reason_code == "field_missing":
                    continue
                # Fail closed on ancestor permission_denied / unsafe / malformed / timeout
                # because we cannot prove a tighter ancestor limit does not exist.
                return (raw, None)

            parsed = parser(raw)
            if isinstance(parsed, LinuxMemoryUnavailable):
                if parsed.reason_code == "unlimited":
                    saw_unlimited = True
                    continue
                return (parsed, None)

            if best_value is None or parsed < best_value:
                best_value = parsed
                best_path = rel_path

        if best_value is not None:
            return (best_value, best_path)
        if saw_unlimited:
            return (
                LinuxMemoryUnavailable(
                    reason=f"effective hierarchical {filename} is unlimited ('max')",
                    reason_code="unlimited",
                    status="unavailable",
                ),
                None,
            )
        return (
            LinuxMemoryUnavailable(
                reason=f"{filename} absent along cgroup hierarchy",
                reason_code="field_missing",
                status="unavailable",
            ),
            None,
        )

    def _unavailable_reading(
        self,
        issue: Any,
        *,
        read_start_ns: int,
        read_end_ns: int,
        target_generation: Any,
        observed_before: Any,
        observed_after: Any,
        cgroup_identity: LinuxCgroupV2Identity | None = None,
    ) -> LinuxCgroupV2Reading:
        return LinuxCgroupV2Reading(
            status=issue.status,
            memory_current_bytes=issue,
            memory_peak_bytes=issue,
            memory_max_bytes=issue,
            cpu_usage_seconds=issue,
            cpu_user_seconds=issue,
            cpu_system_seconds=issue,
            cpu_throttled_seconds=issue,
            cpu_nr_throttled=issue,
            cpu_max_cores=issue,
            cpuset_effective_cpus=issue,
            read_start_ns=read_start_ns,
            read_end_ns=max(read_end_ns, read_start_ns),
            cgroup_identity=cgroup_identity,
            reason_code=issue.reason_code,
            detail=issue.reason,
            target_generation=target_generation,
            observed_generation_before=observed_before,
            observed_generation_after=observed_after,
        )

    def bind_target(self, *, pid: int | None = None) -> Any:
        """Bind exact live target process generation at activation time.

        Verifies platform BEFORE returning any pre-bound ``expected_generation``
        and persists validated ``self._pid`` before reading procfs.
        """

        pr = _pr()
        LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

        if pid is not None:
            pid = _require_positive_int(pid, "LinuxCgroupV2Probe.bind_target.pid")
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

    def resolve_cgroup_identity(self) -> LinuxCgroupV2Identity | Any:
        """Resolve read-only cgroup v2 mount and path identity for inventory checks."""

        pr = _pr()
        LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

        platform_issue = self._check_platform()
        if platform_issue is not None:
            self._last_cgroup_error = platform_issue
            return platform_issue

        expected = self._bound_generation
        target_pid = (
            expected.pid
            if expected is not None
            else (self._pid if self._pid is not None else os.getpid())
        )
        deadline_ns = int(self._monotonic_ns()) + self._max_read_ns

        # 1. Pre-read target generation verification.
        gen_before = self._read_generation(target_pid, deadline_ns=deadline_ns)
        if isinstance(gen_before, LinuxMemoryUnavailable):
            self._last_cgroup_error = gen_before
            return gen_before
        if int(self._monotonic_ns()) > deadline_ns:
            timeout_issue = LinuxMemoryUnavailable(
                reason=(
                    f"resolving cgroup v2 identity for pid {target_pid} exceeded "
                    f"max_read_seconds ({self._max_read_seconds}s)"
                ),
                reason_code="read_timeout",
                status="error",
            )
            self._last_cgroup_error = timeout_issue
            return timeout_issue
        if expected is not None:
            pre_mismatch = _generation_exact_mismatch(expected, gen_before)
            if pre_mismatch is not None:
                changed = LinuxMemoryUnavailable(
                    reason=(
                        f"target process generation changed before cgroup identity "
                        f"resolution: {pre_mismatch}"
                    ),
                    reason_code="generation_changed",
                    status="unavailable",
                )
                self._last_cgroup_error = changed
                return changed

        # 2. Resolve unified cgroup v2 membership, mount root, and contained chain.
        rel_path = self._read_cgroup_rel_path(target_pid, deadline_ns=deadline_ns)
        if isinstance(rel_path, LinuxMemoryUnavailable):
            self._last_cgroup_error = rel_path
            return rel_path
        mount_root = self._read_mountinfo_cgroup2_root(
            target_pid, deadline_ns=deadline_ns
        )
        if isinstance(mount_root, LinuxMemoryUnavailable):
            self._last_cgroup_error = mount_root
            return mount_root
        if int(self._monotonic_ns()) > deadline_ns:
            timeout_issue = LinuxMemoryUnavailable(
                reason=(
                    f"resolving cgroup v2 identity for pid {target_pid} exceeded "
                    f"max_read_seconds ({self._max_read_seconds}s)"
                ),
                reason_code="read_timeout",
                status="error",
            )
            self._last_cgroup_error = timeout_issue
            return timeout_issue
        chain = self._resolve_contained_cgroup_chain(mount_root, rel_path)
        if isinstance(chain, LinuxMemoryUnavailable):
            self._last_cgroup_error = chain
            return chain
        cg_rel_path = chain[0][0]

        # 3. Post-read cgroup-path stability, target generation, and deadline check.
        rel_path_after = self._read_cgroup_rel_path(
            target_pid, deadline_ns=deadline_ns
        )
        gen_after = self._read_generation(target_pid, deadline_ns=deadline_ns)
        if isinstance(gen_after, LinuxMemoryUnavailable):
            self._last_cgroup_error = gen_after
            return gen_after
        if int(self._monotonic_ns()) > deadline_ns:
            timeout_issue = LinuxMemoryUnavailable(
                reason=(
                    f"resolving cgroup v2 identity for pid {target_pid} exceeded "
                    f"max_read_seconds ({self._max_read_seconds}s)"
                ),
                reason_code="read_timeout",
                status="error",
            )
            self._last_cgroup_error = timeout_issue
            return timeout_issue

        post_mismatch = (
            _generation_exact_mismatch(expected, gen_after)
            if expected is not None
            else None
        ) or _generation_exact_mismatch(gen_before, gen_after)
        if post_mismatch is not None:
            changed = LinuxMemoryUnavailable(
                reason=(
                    f"target process generation changed during cgroup identity "
                    f"resolution: {post_mismatch}"
                ),
                reason_code="generation_changed",
                status="unavailable",
            )
            self._last_cgroup_error = changed
            return changed

        if isinstance(rel_path_after, LinuxMemoryUnavailable):
            self._last_cgroup_error = rel_path_after
            return rel_path_after
        if rel_path_after != cg_rel_path:
            moved = LinuxMemoryUnavailable(
                reason=(
                    f"target cgroup path changed during identity resolution "
                    f"({cg_rel_path!r} -> {rel_path_after!r})"
                ),
                reason_code="cgroup_changed",
                status="unavailable",
            )
            self._last_cgroup_error = moved
            return moved

        ident = LinuxCgroupV2Identity(
            mount_root=str(mount_root),
            cgroup_path=cg_rel_path,
            shared_scope_not_exclusive=True,
        )
        self._last_cgroup_identity = ident
        self._last_cgroup_error = None
        return ident

    def read(self) -> LinuxCgroupV2Reading:
        """Read cgroup v2 counters and hierarchical limits with pre/post target verification."""

        pr = _pr()
        LinuxMemoryUnavailable = pr.LinuxMemoryUnavailable

        read_start_ns = int(self._monotonic_ns())
        deadline_ns = read_start_ns + self._max_read_ns

        platform_issue = self._check_platform()
        if platform_issue is not None:
            self._last_cgroup_error = platform_issue
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

        # 1. Pre-read target generation verification.
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

        pre_mismatch = _generation_exact_mismatch(expected, gen_before)
        if pre_mismatch is not None:
            changed = LinuxMemoryUnavailable(
                reason=f"target process generation changed before cgroup read: {pre_mismatch}",
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

        # 2. Resolve unified cgroup v2 membership & mount root.
        rel_path_before = self._read_cgroup_rel_path(target_pid, deadline_ns=deadline_ns)
        if isinstance(rel_path_before, LinuxMemoryUnavailable):
            self._last_cgroup_error = rel_path_before
            return self._unavailable_reading(
                rel_path_before,
                read_start_ns=read_start_ns,
                read_end_ns=int(self._monotonic_ns()),
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
            )

        mount_root = self._read_mountinfo_cgroup2_root(
            target_pid, deadline_ns=deadline_ns
        )
        if isinstance(mount_root, LinuxMemoryUnavailable):
            self._last_cgroup_error = mount_root
            return self._unavailable_reading(
                mount_root,
                read_start_ns=read_start_ns,
                read_end_ns=int(self._monotonic_ns()),
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
            )

        chain = self._resolve_contained_cgroup_chain(mount_root, rel_path_before)
        if isinstance(chain, LinuxMemoryUnavailable):
            self._last_cgroup_error = chain
            return self._unavailable_reading(
                chain,
                read_start_ns=read_start_ns,
                read_end_ns=int(self._monotonic_ns()),
                target_generation=expected,
                observed_before=gen_before,
                observed_after=None,
            )

        cg_rel_path, cg_dir = chain[0]

        # 3. Read cgroup v2 files (strictly read-only).
        mem_cur_raw = self._read_cgroup_file(
            cg_dir / "memory.current", mount_root=mount_root, deadline_ns=deadline_ns
        )
        memory_current = (
            mem_cur_raw
            if isinstance(mem_cur_raw, LinuxMemoryUnavailable)
            else parse_cgroup_uint(mem_cur_raw, field_name="memory.current")
        )

        mem_peak_raw = self._read_cgroup_file(
            cg_dir / "memory.peak", mount_root=mount_root, deadline_ns=deadline_ns
        )
        memory_peak = (
            mem_peak_raw
            if isinstance(mem_peak_raw, LinuxMemoryUnavailable)
            else parse_cgroup_uint(mem_peak_raw, field_name="memory.peak")
        )

        memory_max, eff_mem_path = self._resolve_hierarchical_limit(
            chain,
            mount_root=mount_root,
            filename="memory.max",
            parser=parse_cgroup_memory_max,
            deadline_ns=deadline_ns,
        )

        cpu_stat_raw = self._read_cgroup_file(
            cg_dir / "cpu.stat", mount_root=mount_root, deadline_ns=deadline_ns
        )
        if isinstance(cpu_stat_raw, LinuxMemoryUnavailable):
            cpu_usage = cpu_stat_raw
            cpu_user = cpu_stat_raw
            cpu_system = cpu_stat_raw
            cpu_throttled = cpu_stat_raw
            cpu_nr_throttled = cpu_stat_raw
        else:
            parsed_cpu_stat = parse_cgroup_cpu_stat(cpu_stat_raw)
            if isinstance(parsed_cpu_stat, LinuxMemoryUnavailable):
                cpu_usage = parsed_cpu_stat
                cpu_user = parsed_cpu_stat
                cpu_system = parsed_cpu_stat
                cpu_throttled = parsed_cpu_stat
                cpu_nr_throttled = parsed_cpu_stat
            else:
                cpu_usage = parsed_cpu_stat["cpu_usage_seconds"]
                cpu_user = parsed_cpu_stat["cpu_user_seconds"]
                cpu_system = parsed_cpu_stat["cpu_system_seconds"]
                cpu_throttled = parsed_cpu_stat["cpu_throttled_seconds"]
                cpu_nr_throttled = parsed_cpu_stat["cpu_nr_throttled"]

        cpu_max, eff_cpu_path = self._resolve_hierarchical_limit(
            chain,
            mount_root=mount_root,
            filename="cpu.max",
            parser=parse_cgroup_cpu_max,
            deadline_ns=deadline_ns,
        )

        cpuset_raw = self._read_cgroup_file(
            cg_dir / "cpuset.cpus.effective",
            mount_root=mount_root,
            deadline_ns=deadline_ns,
        )
        if (
            isinstance(cpuset_raw, LinuxMemoryUnavailable)
            and cpuset_raw.reason_code == "field_missing"
        ):
            cpuset_raw = self._read_cgroup_file(
                cg_dir / "cpuset.cpus",
                mount_root=mount_root,
                deadline_ns=deadline_ns,
            )
        cpuset_effective = (
            cpuset_raw
            if isinstance(cpuset_raw, LinuxMemoryUnavailable)
            else parse_cpuset_effective_cpus(cpuset_raw)
        )

        # Read bounded cgroup.procs for co-member provenance.
        observed_member_count: int | None = None
        includes_other_tasks: bool | None = None
        procs_truncated = False
        procs_raw = self._read_cgroup_file(
            cg_dir / "cgroup.procs", mount_root=mount_root, deadline_ns=deadline_ns
        )
        if isinstance(procs_raw, LinuxMemoryUnavailable):
            if procs_raw.reason_code in (
                "unsafe_cgroup_path",
                "read_timeout",
            ):
                return self._unavailable_reading(
                    procs_raw,
                    read_start_ns=read_start_ns,
                    read_end_ns=int(self._monotonic_ns()),
                    target_generation=expected,
                    observed_before=gen_before,
                    observed_after=None,
                )
            if procs_raw.reason_code == "input_limit_exceeded":
                # On a large shared cgroup, cgroup.procs may exceed max_read_bytes;
                # record bounded truncation rather than discarding valid metrics.
                procs_truncated = True
                includes_other_tasks = True
                observed_member_count = None
        else:
            parsed_procs = parse_cgroup_procs(procs_raw, max_procs=self._max_procs)
            if not isinstance(parsed_procs, LinuxMemoryUnavailable):
                pids_tuple, procs_truncated = parsed_procs
                observed_member_count = len(pids_tuple)
                includes_other_tasks = procs_truncated or any(
                    p != target_pid for p in pids_tuple
                )

        # 4. Post-read target generation & cgroup-path stability verification.
        rel_path_after = self._read_cgroup_rel_path(target_pid, deadline_ns=deadline_ns)
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
                    f"cgroup v2 read exceeded max_read_seconds "
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

        post_mismatch = _generation_exact_mismatch(
            expected, gen_after
        ) or _generation_exact_mismatch(gen_before, gen_after)
        if post_mismatch is not None:
            changed = LinuxMemoryUnavailable(
                reason=f"target process generation changed during cgroup read: {post_mismatch}",
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

        if isinstance(rel_path_after, LinuxMemoryUnavailable):
            return self._unavailable_reading(
                rel_path_after,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=gen_after,
            )
        if rel_path_after != cg_rel_path:
            moved = LinuxMemoryUnavailable(
                reason=(
                    f"target cgroup path changed during read "
                    f"({cg_rel_path!r} -> {rel_path_after!r})"
                ),
                reason_code="cgroup_changed",
                status="unavailable",
            )
            return self._unavailable_reading(
                moved,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                target_generation=expected,
                observed_before=gen_before,
                observed_after=gen_after,
            )

        identity = LinuxCgroupV2Identity(
            mount_root=str(mount_root),
            cgroup_path=cg_rel_path,
            effective_memory_limit_path=eff_mem_path,
            effective_cpu_limit_path=eff_cpu_path,
            observed_member_count=observed_member_count,
            includes_other_tasks=includes_other_tasks,
            shared_scope_not_exclusive=True,
            procs_truncated=procs_truncated,
        )
        self._last_cgroup_identity = identity
        self._last_cgroup_error = None

        all_fields = (
            memory_current,
            memory_peak,
            memory_max,
            cpu_usage,
            cpu_user,
            cpu_system,
            cpu_throttled,
            cpu_nr_throttled,
            cpu_max,
            cpuset_effective,
        )
        issues = [v for v in all_fields if isinstance(v, LinuxMemoryUnavailable)]
        if not issues:
            return LinuxCgroupV2Reading(
                status="ok",
                memory_current_bytes=memory_current,
                memory_peak_bytes=memory_peak,
                memory_max_bytes=memory_max,
                cpu_usage_seconds=cpu_usage,
                cpu_user_seconds=cpu_user,
                cpu_system_seconds=cpu_system,
                cpu_throttled_seconds=cpu_throttled,
                cpu_nr_throttled=cpu_nr_throttled,
                cpu_max_cores=cpu_max,
                cpuset_effective_cpus=cpuset_effective,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                cgroup_identity=identity,
                target_generation=expected,
                observed_generation_before=gen_before,
                observed_generation_after=gen_after,
            )

        unique_codes = {issue.reason_code for issue in issues}
        detail = "; ".join(dict.fromkeys(issue.reason for issue in issues))
        if len(issues) < len(all_fields):
            summary_code = (
                next(iter(unique_codes))
                if len(unique_codes) == 1
                else "partial_reading"
            )
            return LinuxCgroupV2Reading(
                status="partial",
                memory_current_bytes=memory_current,
                memory_peak_bytes=memory_peak,
                memory_max_bytes=memory_max,
                cpu_usage_seconds=cpu_usage,
                cpu_user_seconds=cpu_user,
                cpu_system_seconds=cpu_system,
                cpu_throttled_seconds=cpu_throttled,
                cpu_nr_throttled=cpu_nr_throttled,
                cpu_max_cores=cpu_max,
                cpuset_effective_cpus=cpuset_effective,
                read_start_ns=read_start_ns,
                read_end_ns=read_end_ns,
                cgroup_identity=identity,
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
        return LinuxCgroupV2Reading(
            status=overall_status,
            memory_current_bytes=memory_current,
            memory_peak_bytes=memory_peak,
            memory_max_bytes=memory_max,
            cpu_usage_seconds=cpu_usage,
            cpu_user_seconds=cpu_user,
            cpu_system_seconds=cpu_system,
            cpu_throttled_seconds=cpu_throttled,
            cpu_nr_throttled=cpu_nr_throttled,
            cpu_max_cores=cpu_max,
            cpuset_effective_cpus=cpuset_effective,
            read_start_ns=read_start_ns,
            read_end_ns=read_end_ns,
            cgroup_identity=identity,
            reason_code=summary_code,
            detail=detail,
            target_generation=expected,
            observed_generation_before=gen_before,
            observed_generation_after=gen_after,
        )


__all__ = [
    "DEFAULT_LINUX_CGROUP_MAX_ANCESTOR_DEPTH",
    "DEFAULT_LINUX_CGROUP_MAX_MOUNTINFO_BYTES",
    "DEFAULT_LINUX_CGROUP_MAX_PROCS",
    "DEFAULT_LINUX_CGROUP_MAX_READ_BYTES",
    "DEFAULT_LINUX_CGROUP_MAX_READ_SECONDS",
    "DEFAULT_LINUX_TREE_MAX_ENUMERATED_CANDIDATES",
    "DEFAULT_LINUX_TREE_MAX_OBSERVED_MEMBERS",
    "DEFAULT_LINUX_TREE_MAX_READ_BYTES",
    "DEFAULT_LINUX_TREE_MAX_SCAN_SECONDS",
    "LINUX_CGROUP_READING_STATUSES",
    "LINUX_TREE_MEMBER_ROLES",
    "LINUX_TREE_READING_STATUSES",
    "LinuxCgroupV2Identity",
    "LinuxCgroupV2Probe",
    "LinuxCgroupV2Reading",
    "LinuxProcessTreeMember",
    "LinuxProcessTreeProbe",
    "LinuxProcessTreeReading",
    "ProcStatSnapshot",
    "parse_cgroup_cpu_max",
    "parse_cgroup_cpu_stat",
    "parse_cgroup_memory_max",
    "parse_cgroup_procs",
    "parse_cgroup_uint",
    "parse_cpuset_effective_cpus",
    "parse_proc_cgroup_v2_path",
    "parse_proc_mountinfo_cgroup2",
    "parse_proc_stat_fields",
]
