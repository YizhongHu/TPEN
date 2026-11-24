"""Unit tests for ``tpen.process_resources`` including O4 Linux procfs memory probing."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from tpen.process_resources import (
    DEFAULT_LINUX_MEMORY_MAX_READ_BYTES,
    DEFAULT_LINUX_MEMORY_MAX_READ_SECONDS,
    LinuxMemoryUnavailable,
    LinuxProcessGeneration,
    LinuxProcessMemoryProbe,
    LinuxProcessMemoryReading,
    ProcessRUsageProbe,
    ResourceScope,
    ResourceUnavailable,
    parse_proc_stat_start_ticks,
    parse_proc_status_memory,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_stat_line(
    pid: int,
    comm: str = "python3",
    *,
    state: str = "S",
    start_ticks: int = 9000,
) -> str:
    """Build a valid Linux ``/proc/<pid>/stat`` line with field 22 = ``start_ticks``."""

    # Fields 3..21 (19 tokens before field 22 starttime), plus trailing vsize/rss.
    fields_3_to_21 = [
        state,  # 3: state
        "1",  # 4: ppid
        str(pid),  # 5: pgrp
        str(pid),  # 6: session
        "0",  # 7: tty_nr
        "-1",  # 8: tpgid
        "4194304",  # 9: flags
        "100",  # 10: minflt
        "0",  # 11: cminflt
        "0",  # 12: majflt
        "0",  # 13: cmajflt
        "10",  # 14: utime
        "5",  # 15: stime
        "0",  # 16: cutime
        "0",  # 17: cstime
        "20",  # 18: priority
        "0",  # 19: nice
        "4",  # 20: num_threads
        "0",  # 21: itrealvalue
    ]
    return f"{pid} ({comm}) " + " ".join(fields_3_to_21) + f" {start_ticks} 123456 50\n"


def _make_status_text(
    *,
    name: str = "python3",
    pid: int = 1234,
    vm_rss_kib: int | str | None = 1024,
    vm_size_kib: int | str | None = 4096,
    threads: int | str | None = 4,
    rss_unit: str = "kB",
    size_unit: str = "kB",
) -> str:
    """Build a synthetic Linux ``/proc/<pid>/status`` file."""

    lines = [
        f"Name:\t{name}",
        "State:\tS (sleeping)",
        f"Pid:\t{pid}",
    ]
    if vm_size_kib is not None:
        lines.append(f"VmSize:\t{vm_size_kib} {size_unit}".rstrip())
    if vm_rss_kib is not None:
        lines.append(f"VmRSS:\t{vm_rss_kib} {rss_unit}".rstrip())
    if threads is not None:
        lines.append(f"Threads:\t{threads}")
    return "\n".join(lines) + "\n"


def _populate_proc_fixture(
    proc_root: Path,
    *,
    pid: int = 1234,
    comm: str = "python3",
    start_ticks: int = 9000,
    boot_id: str | None = "00000000-0000-4000-8000-000000000002",
    pid_ns_target: str | None = "pid:[4026531836]",
    status_text: str | None = None,
    write_stat: bool = True,
    write_status: bool = True,
) -> Path:
    """Create a deterministic fake procfs tree under ``proc_root``."""

    proc_root.mkdir(parents=True, exist_ok=True)
    if boot_id is not None:
        boot_file = proc_root / "sys" / "kernel" / "random" / "boot_id"
        boot_file.parent.mkdir(parents=True, exist_ok=True)
        boot_file.write_text(f"{boot_id}\n", encoding="utf-8")

    pid_dir = proc_root / str(pid)
    pid_dir.mkdir(parents=True, exist_ok=True)
    if write_stat:
        (pid_dir / "stat").write_text(
            _make_stat_line(pid, comm=comm, start_ticks=start_ticks),
            encoding="utf-8",
        )
    if write_status:
        text = (
            _make_status_text(pid=pid)
            if status_text is None
            else status_text
        )
        (pid_dir / "status").write_text(text, encoding="utf-8")
    if pid_ns_target is not None:
        ns_dir = pid_dir / "ns"
        ns_dir.mkdir(parents=True, exist_ok=True)
        ns_link = ns_dir / "pid"
        if ns_link.is_symlink() or ns_link.exists():
            ns_link.unlink()
        os.symlink(pid_ns_target, ns_link)
    return proc_root


class _FixedStepClock:
    """Deterministic nanosecond monotonic clock for probe unit tests."""

    def __init__(self, start_ns: int = 1_000_000_000, step_ns: int = 1_000) -> None:
        self._now = start_ns
        self._step = step_ns

    def __call__(self) -> int:
        val = self._now
        self._now += self._step
        return val


# ---------------------------------------------------------------------------
# Import hygiene & existing RUSAGE preservation
# ---------------------------------------------------------------------------


def test_process_resources_is_stdlib_only_and_torch_free_in_subprocess() -> None:
    """``tpen.process_resources`` imports only stdlib and remains Torch- and psutil-free."""

    module_path = _REPO_ROOT / "tpen" / "process_resources.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    forbidden_roots = {"torch", "experiments", "tpen_executor", "psutil", "omegaconf"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden_roots
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".")[0] not in forbidden_roots

    code = (
        "import sys, tpen.process_resources as pr; "
        "probe = pr.LinuxProcessMemoryProbe(); "
        "assert probe.target_generation is None; "
        "assert 'torch' not in sys.modules; "
        "assert 'psutil' not in sys.modules; "
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


def test_existing_rusage_probe_behavior_is_preserved() -> None:
    """Existing ProcessRUsageProbe API and read/result semantics stay intact."""

    probe = ProcessRUsageProbe(ResourceScope.PROCESS)
    baseline = probe.read()
    result = probe.result(baseline)
    assert not isinstance(result.user_cpu_seconds, ResourceUnavailable)
    assert not isinstance(result.peak_rss_mb, ResourceUnavailable)


def test_probe_construction_is_pure_and_does_not_freeze_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing ``LinuxProcessMemoryProbe(pid=None)`` must not call ``os.getpid()``."""

    def _forbid_getpid() -> int:
        raise AssertionError("os.getpid() must not be called during probe construction")

    monkeypatch.setattr(os, "getpid", _forbid_getpid)
    probe = LinuxProcessMemoryProbe()
    assert probe.target_generation is None
    assert probe.bind_error is None
    assert probe.max_read_bytes == DEFAULT_LINUX_MEMORY_MAX_READ_BYTES
    assert probe.max_read_seconds == DEFAULT_LINUX_MEMORY_MAX_READ_SECONDS


# ---------------------------------------------------------------------------
# Pure /proc/<pid>/stat and /proc/<pid>/status parser tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "comm",
    [
        "python3",
        "my python worker",
        "worker (nested) ) )",
        "line1\nline2) (extra)",
        "",
    ],
)
def test_parse_proc_stat_handles_spaces_newlines_and_multiple_parentheses(comm: str) -> None:
    """Parser uses final ')' via rfind so tricky ``comm`` names parse starttime cleanly."""

    raw = _make_stat_line(4321, comm=comm, start_ticks=987654321)
    parsed = parse_proc_stat_start_ticks(raw, expected_pid=4321)
    assert parsed == (4321, 987654321)


def test_parse_proc_stat_handles_raw_non_utf8_bytes_in_comm() -> None:
    """Non-UTF-8 bytes inside ``comm`` do not prevent ASCII starttime extraction."""

    raw = b"77 (\xff\xfe) ) S 1 77 77 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 424242 0 0\n"
    parsed = parse_proc_stat_start_ticks(raw, expected_pid=77)
    assert parsed == (77, 424242)


@pytest.mark.parametrize(
    "bad_stat",
    [
        "",
        "1234 no_parens S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
        "1234 (truncated) S 1 2 3",
        "1234 (neg_start) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 -99",
        "1234 (float_start) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 12.5",
        "1234 (alpha_start) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 abc",
        "1234 (bad_state) 99 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 500",
        "0 (zero_pid) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 500",
        "9999 (wrong_pid) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 500",
    ],
)
def test_parse_proc_stat_rejects_malformed_inputs(bad_stat: str) -> None:
    """Malformed, truncated, negative, non-integer, or wrong-PID stat lines return typed error."""

    result = parse_proc_stat_start_ticks(bad_stat, expected_pid=1234)
    assert isinstance(result, LinuxMemoryUnavailable)
    assert isinstance(result, ResourceUnavailable)
    assert result.status == "error"
    assert result.reason_code == "malformed_procfs"


def test_parse_proc_status_normal_and_legitimate_zero() -> None:
    """Valid KiB fields convert by *1024 and explicit ``0 kB`` / ``0`` stay integer 0."""

    normal = parse_proc_status_memory(
        _make_status_text(vm_rss_kib=2048, vm_size_kib=8192, threads=6)
    )
    assert normal == (2048 * 1024, 8192 * 1024, 6)

    zeros = parse_proc_status_memory(
        _make_status_text(vm_rss_kib=0, vm_size_kib=0, threads=0)
    )
    assert zeros == (0, 0, 0)
    assert all(type(val) is int for val in zeros)


def test_parse_proc_status_missing_individual_keys_yields_per_field_unavailable() -> None:
    """Missing keys return typed ``field_missing`` unavailability, never guessed 0."""

    partial = parse_proc_status_memory(
        _make_status_text(vm_rss_kib=512, vm_size_kib=None, threads=3)
    )
    assert isinstance(partial, tuple)
    rss, vmsize, threads = partial
    assert rss == 512 * 1024
    assert threads == 3
    assert isinstance(vmsize, LinuxMemoryUnavailable)
    assert vmsize.status == "unavailable"
    assert vmsize.reason_code == "field_missing"


@pytest.mark.parametrize(
    "bad_status",
    [
        "",
        "   \n\t\n",
        "not a key value line\n",
        ": missing key\n",
        b"\xff\xfe\xfd",
    ],
)
def test_parse_proc_status_rejects_empty_corrupt_or_invalid_utf8(
    bad_status: str | bytes,
) -> None:
    """Empty, structurally corrupt, or invalid-UTF-8 status returns whole-file malformed error."""

    result = parse_proc_status_memory(bad_status)
    assert isinstance(result, LinuxMemoryUnavailable)
    assert result.status == "error"
    assert result.reason_code == "malformed_procfs"


@pytest.mark.parametrize(
    ("status_text", "bad_index"),
    [
        (_make_status_text(vm_rss_kib=-10), 0),
        (_make_status_text(vm_rss_kib="12.5"), 0),
        (_make_status_text(vm_rss_kib="abc"), 0),
        (_make_status_text(vm_rss_kib=1024, rss_unit="MB"), 0),
        (_make_status_text(vm_rss_kib=1024, rss_unit="KiB"), 0),
        (_make_status_text(vm_rss_kib=1024, rss_unit=""), 0),
        (_make_status_text(vm_size_kib=-1), 1),
        (_make_status_text(vm_size_kib=1024, size_unit="B"), 1),
        (_make_status_text(threads=-2), 2),
        (_make_status_text(threads="2.0"), 2),
        (_make_status_text(threads="4 extra"), 2),
        ("Name:\tpy\nVmRSS:\t10 kB\nVmRSS:\t20 kB\nVmSize:\t30 kB\nThreads:\t1\n", 0),
    ],
)
def test_parse_proc_status_rejects_bad_field_values_units_and_duplicates(
    status_text: str, bad_index: int
) -> None:
    """Negative, non-integer, non-kB-unit, or duplicate keys produce typed malformed errors."""

    result = parse_proc_status_memory(status_text)
    assert isinstance(result, tuple)
    bad = result[bad_index]
    assert isinstance(bad, LinuxMemoryUnavailable)
    assert bad.status == "error"
    assert bad.reason_code == "malformed_procfs"


# ---------------------------------------------------------------------------
# LinuxProcessMemoryProbe fixture, generation defense, bounds, and platform tests
# ---------------------------------------------------------------------------


def test_probe_normal_fixture_read_and_exact_generation_provenance(tmp_path: Path) -> None:
    """Normal fixture read binds generation, brackets timestamps, and returns exact bytes."""

    proc_root = _populate_proc_fixture(
        tmp_path / "proc",
        pid=2222,
        comm="tpen (worker) 1",
        start_ticks=77777,
        boot_id="11111111-2222-4333-8444-555555555555",
        pid_ns_target="pid:[4026539999]",
        status_text=_make_status_text(pid=2222, vm_rss_kib=4096, vm_size_kib=16384, threads=8),
    )
    clock = _FixedStepClock(start_ns=5_000_000, step_ns=500)
    probe = LinuxProcessMemoryProbe(
        pid=2222,
        proc_root=proc_root,
        monotonic_ns=clock,
        host_id="node-test",
    )
    bound = probe.bind_target()
    assert bound == LinuxProcessGeneration(
        pid=2222,
        start_ticks=77777,
        boot_id="11111111-2222-4333-8444-555555555555",
        pid_ns_inode="pid:[4026539999]",
        host_id="node-test",
    )

    reading = probe.read()
    assert isinstance(reading, LinuxProcessMemoryReading)
    assert reading.status == "ok"
    assert reading.reason_code is None
    assert reading.rss_bytes == 4096 * 1024
    assert reading.virtual_bytes == 16384 * 1024
    assert reading.thread_count == 8
    assert reading.target_generation == bound
    assert reading.observed_generation_before == bound
    assert reading.observed_generation_after == bound
    assert reading.read_end_ns >= reading.read_start_ns


def test_probe_partial_and_missing_procfs_outcomes(tmp_path: Path) -> None:
    """Missing individual keys yield ``partial``; missing stat/status/procfs yield unavailable."""

    # Partial status (VmSize omitted)
    proc_partial = _populate_proc_fixture(
        tmp_path / "proc_partial",
        pid=3001,
        status_text=_make_status_text(pid=3001, vm_rss_kib=0, vm_size_kib=None, threads=2),
    )
    reading = LinuxProcessMemoryProbe(pid=3001, proc_root=proc_partial).read()
    assert reading.status == "partial"
    assert reading.reason_code == "field_missing"
    assert reading.rss_bytes == 0
    assert reading.thread_count == 2
    assert isinstance(reading.virtual_bytes, LinuxMemoryUnavailable)
    assert reading.virtual_bytes.reason_code == "field_missing"

    # Missing /proc/<pid>/stat -> process_exited
    proc_no_stat = _populate_proc_fixture(
        tmp_path / "proc_no_stat", pid=3002, write_stat=False, write_status=True
    )
    r_no_stat = LinuxProcessMemoryProbe(pid=3002, proc_root=proc_no_stat).read()
    assert r_no_stat.status == "unavailable"
    assert r_no_stat.reason_code == "process_exited"
    assert isinstance(r_no_stat.rss_bytes, ResourceUnavailable)

    # Missing /proc/<pid>/status -> process_exited
    proc_no_status = _populate_proc_fixture(
        tmp_path / "proc_no_status", pid=3003, write_stat=True, write_status=False
    )
    r_no_status = LinuxProcessMemoryProbe(pid=3003, proc_root=proc_no_status).read()
    assert r_no_status.status == "unavailable"
    assert r_no_status.reason_code == "process_exited"
    assert isinstance(r_no_status.rss_bytes, ResourceUnavailable)

    # Missing proc_root -> procfs_missing
    r_no_proc = LinuxProcessMemoryProbe(
        pid=3004, proc_root=tmp_path / "nonexistent_proc"
    ).read()
    assert r_no_proc.status == "unavailable"
    assert r_no_proc.reason_code == "procfs_missing"


def test_probe_permission_error_yields_permission_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PermissionError on stat or status yields typed ``permission_denied`` without raising."""

    proc_root = _populate_proc_fixture(tmp_path / "proc_perm", pid=4001)
    status_path = proc_root / "4001" / "status"
    real_open = Path.open

    def _deny_status(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == status_path:
            raise PermissionError("EACCES simulated on status")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _deny_status)
    reading = LinuxProcessMemoryProbe(pid=4001, proc_root=proc_root).read()
    assert reading.status == "unavailable"
    assert reading.reason_code == "permission_denied"
    assert isinstance(reading.rss_bytes, LinuxMemoryUnavailable)
    assert reading.rss_bytes.reason_code == "permission_denied"


@pytest.mark.parametrize(
    ("mutation_kind",),
    [("start_ticks",), ("boot_id",), ("pid_ns_inode",)],
)
def test_probe_detects_generation_change_before_read(
    tmp_path: Path, mutation_kind: str
) -> None:
    """Pre-read generation drift (start_ticks, boot_id, or pid_ns_inode) rejects memory values."""

    proc_root = _populate_proc_fixture(
        tmp_path / f"proc_pre_{mutation_kind}",
        pid=5001,
        start_ticks=1000,
        boot_id="boot-aaa",
        pid_ns_target="pid:[100]",
    )
    probe = LinuxProcessMemoryProbe(pid=5001, proc_root=proc_root, host_id="h1")
    expected = probe.bind_target()
    assert expected is not None

    if mutation_kind == "start_ticks":
        (proc_root / "5001" / "stat").write_text(
            _make_stat_line(5001, start_ticks=2000), encoding="utf-8"
        )
    elif mutation_kind == "boot_id":
        (proc_root / "sys" / "kernel" / "random" / "boot_id").write_text(
            "boot-bbb\n", encoding="utf-8"
        )
    else:
        ns_link = proc_root / "5001" / "ns" / "pid"
        ns_link.unlink()
        os.symlink("pid:[200]", ns_link)

    reading = probe.read()
    assert reading.status == "unavailable"
    assert reading.reason_code == "generation_changed"
    assert reading.target_generation == expected
    assert reading.observed_generation_before is not None
    assert reading.observed_generation_before != expected
    assert reading.observed_generation_after is None
    assert isinstance(reading.rss_bytes, LinuxMemoryUnavailable)
    assert reading.rss_bytes.reason_code == "generation_changed"
    assert isinstance(reading.virtual_bytes, LinuxMemoryUnavailable)
    assert isinstance(reading.thread_count, LinuxMemoryUnavailable)


@pytest.mark.parametrize("mid_event", ["start_ticks", "boot_id", "pid_ns_inode", "exit"])
def test_probe_detects_generation_change_or_exit_mid_read(
    tmp_path: Path, mid_event: str
) -> None:
    """Post-read verification discards status memory if PID generation changes mid-read."""

    proc_root = _populate_proc_fixture(
        tmp_path / f"proc_mid_{mid_event}",
        pid=5002,
        start_ticks=1000,
        boot_id="boot-orig",
        pid_ns_target="pid:[111]",
        status_text=_make_status_text(pid=5002, vm_rss_kib=999999),
    )
    expected = LinuxProcessGeneration(
        pid=5002,
        start_ticks=1000,
        boot_id="boot-orig",
        pid_ns_inode="pid:[111]",
        host_id="h1",
    )

    clock_calls = 0

    def _mutating_clock() -> int:
        nonlocal clock_calls
        clock_calls += 1
        # Call 1: read_start_ns; Call 2: after pre-read gen; Call 3: after status read
        # (immediately before post-read generation verification).
        if clock_calls == 3:
            if mid_event == "start_ticks":
                (proc_root / "5002" / "stat").write_text(
                    _make_stat_line(5002, start_ticks=5555), encoding="utf-8"
                )
            elif mid_event == "boot_id":
                (proc_root / "sys" / "kernel" / "random" / "boot_id").write_text(
                    "boot-rebooted\n", encoding="utf-8"
                )
            elif mid_event == "pid_ns_inode":
                ns_link = proc_root / "5002" / "ns" / "pid"
                ns_link.unlink()
                os.symlink("pid:[999]", ns_link)
            else:
                (proc_root / "5002" / "stat").unlink()
        return 1_000_000_000 + clock_calls * 1000

    probe = LinuxProcessMemoryProbe(
        pid=5002,
        expected_generation=expected,
        proc_root=proc_root,
        monotonic_ns=_mutating_clock,
        host_id="h1",
    )
    reading = probe.read()
    assert reading.observed_generation_before == expected
    assert isinstance(reading.rss_bytes, LinuxMemoryUnavailable)
    assert isinstance(reading.virtual_bytes, LinuxMemoryUnavailable)
    assert isinstance(reading.thread_count, LinuxMemoryUnavailable)

    if mid_event == "exit":
        assert reading.status == "unavailable"
        assert reading.reason_code == "process_exited"
        assert reading.observed_generation_after is None
    else:
        assert reading.status == "unavailable"
        assert reading.reason_code == "generation_changed"
        assert reading.observed_generation_after is not None
        assert reading.observed_generation_after != expected


def test_probe_enforces_max_read_bytes_without_guessing_zero(tmp_path: Path) -> None:
    """Oversized procfs status returns typed ``input_limit_exceeded`` while exact bound passes."""

    status_text = _make_status_text(pid=6001, vm_rss_kib=256, vm_size_kib=512, threads=2)
    status_bytes_len = len(status_text.encode("utf-8"))
    proc_root = _populate_proc_fixture(
        tmp_path / "proc_bounds", pid=6001, status_text=status_text
    )

    # Exact byte boundary succeeds (stat is shorter than status here, check max of both).
    stat_len = len((proc_root / "6001" / "stat").read_bytes())
    exact_limit = max(status_bytes_len, stat_len)
    ok_probe = LinuxProcessMemoryProbe(
        pid=6001, proc_root=proc_root, max_read_bytes=exact_limit
    )
    ok_reading = ok_probe.read()
    assert ok_reading.status == "ok"
    assert ok_reading.rss_bytes == 256 * 1024

    # Adding padding beyond max_read_bytes triggers input_limit_exceeded.
    padded = status_text + ("# padding line: " + "x" * 200 + "\n")
    (proc_root / "6001" / "status").write_text(padded, encoding="utf-8")
    over_probe = LinuxProcessMemoryProbe(
        pid=6001, proc_root=proc_root, max_read_bytes=exact_limit
    )
    over_reading = over_probe.read()
    assert over_reading.status == "error"
    assert over_reading.reason_code == "input_limit_exceeded"
    assert isinstance(over_reading.rss_bytes, LinuxMemoryUnavailable)
    assert over_reading.rss_bytes.reason_code == "input_limit_exceeded"
    assert isinstance(over_reading.virtual_bytes, LinuxMemoryUnavailable)
    assert isinstance(over_reading.thread_count, LinuxMemoryUnavailable)


@pytest.mark.parametrize("timeout_at_call", [2, 3, 4])
def test_probe_enforces_max_read_seconds_via_deterministic_clock(
    tmp_path: Path, timeout_at_call: int
) -> None:
    """Exceeding ``max_read_seconds`` at any stage returns typed ``read_timeout``."""

    proc_root = _populate_proc_fixture(tmp_path / f"proc_to_{timeout_at_call}", pid=6002)
    expected = LinuxProcessGeneration(
        pid=6002,
        start_ticks=9000,
        boot_id="00000000-0000-4000-8000-000000000002",
        pid_ns_inode="pid:[4026531836]",
        host_id="h1",
    )
    calls = 0

    def _timeout_clock() -> int:
        nonlocal calls
        calls += 1
        if calls >= timeout_at_call:
            return 1_000_000_000 + 2_000_000_000  # +2.0s > 0.5s limit
        return 1_000_000_000 + calls * 1_000

    probe = LinuxProcessMemoryProbe(
        pid=6002,
        expected_generation=expected,
        proc_root=proc_root,
        max_read_seconds=0.5,
        monotonic_ns=_timeout_clock,
        host_id="h1",
    )
    reading = probe.read()
    assert reading.status == "error"
    assert reading.reason_code == "read_timeout"
    assert isinstance(reading.rss_bytes, LinuxMemoryUnavailable)
    assert reading.rss_bytes.reason_code == "read_timeout"
    assert reading.read_end_ns > reading.read_start_ns


@pytest.mark.parametrize("plat", ["darwin", "win32", "freebsd14"])
def test_probe_refuses_non_linux_platform_without_custom_proc_root(plat: str) -> None:
    """Non-Linux platforms return ``unsupported_platform`` without fallback or psutil."""

    probe = LinuxProcessMemoryProbe(platform_name=plat)
    assert probe.bind_target() is None
    assert probe.bind_error is not None
    assert probe.bind_error.reason_code == "unsupported_platform"

    reading = probe.read()
    assert reading.status == "unavailable"
    assert reading.reason_code == "unsupported_platform"
    assert isinstance(reading.rss_bytes, LinuxMemoryUnavailable)
    assert reading.rss_bytes.reason_code == "unsupported_platform"


@pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux /proc filesystem")
def test_probe_reads_live_linux_self_process_from_real_procfs() -> None:
    """On Linux (Cannon), live self-process read returns real VmRSS, Threads, and start_ticks."""

    probe = LinuxProcessMemoryProbe()
    gen = probe.bind_target()
    assert gen is not None
    assert gen.pid == os.getpid()
    assert gen.start_ticks > 0

    reading = probe.read()
    assert reading.status == "ok"
    assert reading.reason_code is None
    assert type(reading.rss_bytes) is int and reading.rss_bytes > 0
    assert type(reading.virtual_bytes) is int and reading.virtual_bytes > 0
    assert type(reading.thread_count) is int and reading.thread_count >= 1
    assert reading.target_generation == gen
    assert reading.observed_generation_before == gen
    assert reading.observed_generation_after == gen
    assert reading.read_end_ns >= reading.read_start_ns


# ---------------------------------------------------------------------------
# O4 T2 review tests
# ---------------------------------------------------------------------------


def test_probe_rejects_boot_id_and_pid_ns_appearance_or_drift_after_none_optional_bind(
    tmp_path: Path,
) -> None:
    """T1 (F1): Optional boot_id/pid_ns_inode appearance or drift after None bind must fail closed."""

    failures: list[str] = []

    # Sub-case A: Bind when boot_id and pid_ns_inode are absent (None), then both appear.
    proc_root_a = _populate_proc_fixture(
        tmp_path / "proc_a",
        pid=5101,
        start_ticks=1000,
        boot_id=None,
        pid_ns_target=None,
    )
    probe_a = LinuxProcessMemoryProbe(pid=5101, proc_root=proc_root_a, host_id="h1")
    bound_a = probe_a.bind_target()
    assert bound_a is not None
    assert bound_a.boot_id is None
    assert bound_a.pid_ns_inode is None

    boot_file_a = proc_root_a / "sys" / "kernel" / "random" / "boot_id"
    boot_file_a.parent.mkdir(parents=True, exist_ok=True)
    boot_file_a.write_text("boot-A\n", encoding="utf-8")
    ns_dir_a = proc_root_a / "5101" / "ns"
    ns_dir_a.mkdir(parents=True, exist_ok=True)
    os.symlink("pid:[100]", ns_dir_a / "pid")

    r1 = probe_a.read()
    if (
        r1.status != "unavailable"
        or r1.reason_code != "generation_changed"
        or not isinstance(r1.rss_bytes, LinuxMemoryUnavailable)
    ):
        failures.append(
            "sub-case A accepted boot_id/pid_ns appearance after None bind: "
            f"status={r1.status!r}, reason_code={r1.reason_code!r}, rss_bytes={r1.rss_bytes!r}"
        )

    # Sub-case B: Pre-bound expected_generation with None optional fields must not pass across drift.
    proc_root_b = _populate_proc_fixture(
        tmp_path / "proc_b",
        pid=5102,
        start_ticks=1000,
        boot_id="boot-A",
        pid_ns_target="pid:[100]",
    )
    probe_b = LinuxProcessMemoryProbe(
        pid=5102,
        expected_generation=LinuxProcessGeneration(
            pid=5102, start_ticks=1000, host_id="h1"
        ),
        proc_root=proc_root_b,
        host_id="h1",
    )
    r_a = probe_b.read()

    (proc_root_b / "sys" / "kernel" / "random" / "boot_id").write_text(
        "boot-B\n", encoding="utf-8"
    )
    ns_link_b = proc_root_b / "5102" / "ns" / "pid"
    ns_link_b.unlink()
    os.symlink("pid:[200]", ns_link_b)

    r_b = probe_b.read()
    if (
        r_b.status != "unavailable"
        or r_b.reason_code != "generation_changed"
        or not isinstance(r_b.rss_bytes, LinuxMemoryUnavailable)
        or (r_a.status == "ok" and r_b.status == "ok")
    ):
        failures.append(
            "sub-case B failed to reject boot_id/pid_ns drift with None optional expected_generation: "
            f"r_a=({r_a.status!r}, {r_a.reason_code!r}), "
            f"r_b=({r_b.status!r}, {r_b.reason_code!r}, rss_bytes={r_b.rss_bytes!r})"
        )

    assert failures == [], "; ".join(failures)


def test_probe_refuses_symlinked_procfs_files_and_does_not_block_on_fifo(
    tmp_path: Path,
) -> None:
    """T2 (F2): Procfs reader must refuse symlinked stat/status files and never block on FIFOs."""

    failures: list[str] = []

    # Sub-case A1: symlinked /proc/<pid>/status
    outside_status = tmp_path / "outside_status.txt"
    outside_status.write_text(
        _make_status_text(pid=6101, vm_rss_kib=999999, vm_size_kib=999999, threads=99),
        encoding="utf-8",
    )
    proc_root_a = _populate_proc_fixture(
        tmp_path / "proc_sym_status", pid=6101, write_status=False
    )
    os.symlink(outside_status, proc_root_a / "6101" / "status")
    r_sym_status = LinuxProcessMemoryProbe(pid=6101, proc_root=proc_root_a).read()
    if r_sym_status.status not in ("error", "unavailable") or not isinstance(
        r_sym_status.rss_bytes, LinuxMemoryUnavailable
    ):
        failures.append(
            "sub-case A1 followed symlinked status: "
            f"status={r_sym_status.status!r}, rss_bytes={r_sym_status.rss_bytes!r}"
        )

    # Sub-case A2: symlinked /proc/<pid>/stat
    outside_stat = tmp_path / "outside_stat.txt"
    outside_stat.write_text(_make_stat_line(6101, start_ticks=9000), encoding="utf-8")
    proc_root_stat = _populate_proc_fixture(
        tmp_path / "proc_sym_stat", pid=6101, write_stat=False
    )
    os.symlink(outside_stat, proc_root_stat / "6101" / "stat")
    r_sym_stat = LinuxProcessMemoryProbe(pid=6101, proc_root=proc_root_stat).read()
    if r_sym_stat.status not in ("error", "unavailable") or not isinstance(
        r_sym_stat.rss_bytes, LinuxMemoryUnavailable
    ):
        failures.append(
            "sub-case A2 followed symlinked stat: "
            f"status={r_sym_stat.status!r}, rss_bytes={r_sym_stat.rss_bytes!r}"
        )

    # Sub-case B: FIFO at /proc/<pid>/status must not block open(2)
    proc_root_b = _populate_proc_fixture(
        tmp_path / "proc_fifo", pid=6102, write_status=False
    )
    fifo = proc_root_b / "6102" / "status"
    os.mkfifo(fifo)
    probe_fifo = LinuxProcessMemoryProbe(
        pid=6102, proc_root=proc_root_b, max_read_seconds=0.05
    )

    result_box: list[LinuxProcessMemoryReading] = []
    exc_box: list[BaseException] = []

    def _run_read() -> None:
        try:
            result_box.append(probe_fifo.read())
        except BaseException as exc:  # pragma: no cover - captured for assertion
            exc_box.append(exc)

    thread = threading.Thread(target=_run_read, daemon=True)
    thread.start()
    try:
        thread.join(timeout=0.35)
        blocked = thread.is_alive()
    finally:
        if thread.is_alive():
            for _ in range(5):
                try:
                    fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                except OSError:
                    time.sleep(0.02)
                else:
                    os.close(fd)
                    break
            thread.join(timeout=2.0)

    if blocked:
        failures.append(
            "sub-case B blocked in open(2) on FIFO /proc/<pid>/status beyond 0.35s"
        )
    elif exc_box:
        failures.append(f"sub-case B raised unexpected exception: {exc_box[0]!r}")
    elif not result_box:
        failures.append("sub-case B produced no reading")
    else:
        r_fifo = result_box[0]
        if r_fifo.status not in ("error", "unavailable") or not isinstance(
            r_fifo.rss_bytes, LinuxMemoryUnavailable
        ):
            failures.append(
                "sub-case B accepted FIFO status read: "
                f"status={r_fifo.status!r}, rss_bytes={r_fifo.rss_bytes!r}"
            )

    assert failures == [], "; ".join(failures)


def test_failed_bind_target_with_explicit_pid_does_not_retarget_read_to_self(
    tmp_path: Path,
) -> None:
    """T4 (F4): Failed bind_target(pid=missing) must not cause read() to silently retarget to self."""

    self_pid = os.getpid()
    missing_pid = self_pid + 100000
    proc_root = _populate_proc_fixture(
        tmp_path / "proc_retarget",
        pid=self_pid,
        status_text=_make_status_text(
            pid=self_pid, vm_rss_kib=2048, vm_size_kib=8192, threads=4
        ),
    )
    probe = LinuxProcessMemoryProbe(proc_root=proc_root)

    assert probe.bind_target(pid=missing_pid) is None
    assert probe.bind_error is not None
    assert probe.bind_error.reason_code == "process_exited"

    reading = probe.read()
    assert (
        reading.status == "unavailable"
        and reading.reason_code == "process_exited"
        and probe.target_generation is None
        and isinstance(reading.rss_bytes, LinuxMemoryUnavailable)
    ), (
        "probe.read() retargeted to self after failed bind_target(pid=missing_pid): "
        f"status={reading.status!r}, reason_code={reading.reason_code!r}, "
        f"target_generation={probe.target_generation!r}, rss_bytes={reading.rss_bytes!r}"
    )
