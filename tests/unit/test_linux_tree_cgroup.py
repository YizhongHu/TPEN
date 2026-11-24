"""Unit tests for O5 bounded Linux descendant-tree and read-only cgroup v2 probes."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from tpen.process_resources import (
    DEFAULT_LINUX_CGROUP_MAX_ANCESTOR_DEPTH,
    DEFAULT_LINUX_CGROUP_MAX_MOUNTINFO_BYTES,
    DEFAULT_LINUX_CGROUP_MAX_READ_BYTES,
    DEFAULT_LINUX_CGROUP_MAX_READ_SECONDS,
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
    LinuxProcessTreeMember,
    LinuxProcessTreeProbe,
    LinuxProcessTreeReading,
    ProcStatSnapshot,
    ResourceUnavailable,
    parse_cgroup_cpu_max,
    parse_cgroup_cpu_stat,
    parse_cgroup_memory_max,
    parse_cgroup_procs,
    parse_cgroup_uint,
    parse_cpuset_effective_cpus,
    parse_proc_cgroup_v2_path,
    parse_proc_mountinfo_cgroup2,
    parse_proc_stat_fields,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_stat_line(
    pid: int,
    comm: str = "python3",
    *,
    state: str = "S",
    ppid: int = 1,
    utime: int = 10,
    stime: int = 5,
    start_ticks: int = 1000,
) -> str:
    """Build a valid Linux ``/proc/<pid>/stat`` line with exact field positions."""

    fields_3_to_21 = [
        state,  # 3: state
        str(ppid),  # 4: ppid
        str(pid),  # 5: pgrp
        str(pid),  # 6: session
        "0",  # 7: tty_nr
        "-1",  # 8: tpgid
        "4194304",  # 9: flags
        "100",  # 10: minflt
        "0",  # 11: cminflt
        "0",  # 12: majflt
        "0",  # 13: cmajflt
        str(utime),  # 14: utime
        str(stime),  # 15: stime
        "0",  # 16: cutime
        "0",  # 17: cstime
        "20",  # 18: priority
        "0",  # 19: nice
        "1",  # 20: num_threads
        "0",  # 21: itrealvalue
    ]
    return f"{pid} ({comm}) " + " ".join(fields_3_to_21) + f" {start_ticks} 123456 50\n"


def _make_status_text(
    *,
    pid: int,
    vm_rss_kib: int | None = 1024,
    vm_size_kib: int | None = 4096,
    threads: int | None = 2,
) -> str:
    lines = [f"Name:\tworker-{pid}", "State:\tS (sleeping)", f"Pid:\t{pid}"]
    if vm_size_kib is not None:
        lines.append(f"VmSize:\t{vm_size_kib} kB")
    if vm_rss_kib is not None:
        lines.append(f"VmRSS:\t{vm_rss_kib} kB")
    if threads is not None:
        lines.append(f"Threads:\t{threads}")
    return "\n".join(lines) + "\n"


def _write_proc_entry(
    proc_root: Path,
    *,
    pid: int,
    ppid: int = 1,
    comm: str = "python3",
    state: str = "S",
    utime: int = 10,
    stime: int = 5,
    start_ticks: int = 1000,
    vm_rss_kib: int | None = 1024,
    pid_ns_target: str | None = "pid:[4026531836]",
    write_status: bool = True,
) -> Path:
    pid_dir = proc_root / str(pid)
    pid_dir.mkdir(parents=True, exist_ok=True)
    (pid_dir / "stat").write_text(
        _make_stat_line(
            pid,
            comm=comm,
            state=state,
            ppid=ppid,
            utime=utime,
            stime=stime,
            start_ticks=start_ticks,
        ),
        encoding="utf-8",
    )
    if write_status:
        (pid_dir / "status").write_text(
            _make_status_text(pid=pid, vm_rss_kib=vm_rss_kib),
            encoding="utf-8",
        )
    if pid_ns_target is not None:
        ns_dir = pid_dir / "ns"
        ns_dir.mkdir(parents=True, exist_ok=True)
        ns_link = ns_dir / "pid"
        if ns_link.is_symlink() or ns_link.exists():
            ns_link.unlink()
        os.symlink(pid_ns_target, ns_link)
    return pid_dir


def _init_proc_root(
    proc_root: Path,
    *,
    boot_id: str | None = "00000000-0000-4000-8000-000000000002",
) -> Path:
    proc_root.mkdir(parents=True, exist_ok=True)
    if boot_id is not None:
        boot_file = proc_root / "sys" / "kernel" / "random" / "boot_id"
        boot_file.parent.mkdir(parents=True, exist_ok=True)
        boot_file.write_text(f"{boot_id}\n", encoding="utf-8")
    return proc_root


def _make_mountinfo_line(
    mount_point: str | Path,
    *,
    fs_type: str = "cgroup2",
    fs_root: str = "/",
    mount_id: int = 36,
) -> str:
    return (
        f"{mount_id} 28 0:30 {fs_root} {mount_point} rw,nosuid,nodev,noexec "
        f"shared:1 - {fs_type} {fs_type} rw\n"
    )


def _setup_cgroup_v2_fixture(
    base_dir: Path,
    *,
    pid: int = 2000,
    start_ticks: int = 5000,
    cgroup_rel_path: str = "/slurm/uid_1000/job_42",
    memory_current: str | None = "104857600",
    memory_peak: str | None = "209715200",
    memory_max_by_rel: dict[str, str] | None = None,
    cpu_stat: str | None = (
        "usage_usec 2500000\n"
        "user_usec 2000000\n"
        "system_usec 500000\n"
        "nr_periods 100\n"
        "nr_throttled 3\n"
        "throttled_usec 150000\n"
    ),
    cpu_max_by_rel: dict[str, str] | None = None,
    cpuset_effective: str | None = "0-3,6",
    cpuset_cpus: str | None = None,
    cgroup_procs: str | None = "2000\n2001\n",
) -> tuple[Path, Path]:
    proc_root = _init_proc_root(base_dir / "proc")
    pid_dir = _write_proc_entry(
        proc_root, pid=pid, ppid=1, start_ticks=start_ticks, vm_rss_kib=4096
    )
    cg_mount = base_dir / "sys_fs_cgroup"
    cg_mount.mkdir(parents=True, exist_ok=True)

    (pid_dir / "cgroup").write_text(f"0::{cgroup_rel_path}\n", encoding="utf-8")
    (pid_dir / "mountinfo").write_text(
        _make_mountinfo_line("/proc", fs_type="proc", mount_id=20)
        + _make_mountinfo_line(cg_mount, fs_type="cgroup2", mount_id=36),
        encoding="utf-8",
    )

    rel_parts = [p for p in cgroup_rel_path.split("/") if p]
    leaf_dir = cg_mount.joinpath(*rel_parts)
    leaf_dir.mkdir(parents=True, exist_ok=True)

    if memory_current is not None:
        (leaf_dir / "memory.current").write_text(f"{memory_current}\n", encoding="utf-8")
    if memory_peak is not None:
        (leaf_dir / "memory.peak").write_text(f"{memory_peak}\n", encoding="utf-8")
    if memory_max_by_rel is None:
        (leaf_dir / "memory.max").write_text("4294967296\n", encoding="utf-8")
    else:
        for rel, content in memory_max_by_rel.items():
            d = cg_mount.joinpath(*[p for p in rel.split("/") if p])
            d.mkdir(parents=True, exist_ok=True)
            (d / "memory.max").write_text(f"{content}\n", encoding="utf-8")

    if cpu_stat is not None:
        (leaf_dir / "cpu.stat").write_text(cpu_stat, encoding="utf-8")
    if cpu_max_by_rel is None:
        (leaf_dir / "cpu.max").write_text("400000 100000\n", encoding="utf-8")
    else:
        for rel, content in cpu_max_by_rel.items():
            d = cg_mount.joinpath(*[p for p in rel.split("/") if p])
            d.mkdir(parents=True, exist_ok=True)
            (d / "cpu.max").write_text(f"{content}\n", encoding="utf-8")

    if cpuset_effective is not None:
        (leaf_dir / "cpuset.cpus.effective").write_text(
            f"{cpuset_effective}\n", encoding="utf-8"
        )
    if cpuset_cpus is not None:
        (leaf_dir / "cpuset.cpus").write_text(f"{cpuset_cpus}\n", encoding="utf-8")
    if cgroup_procs is not None:
        (leaf_dir / "cgroup.procs").write_text(cgroup_procs, encoding="utf-8")

    return proc_root, cg_mount


# ---------------------------------------------------------------------------
# Import hygiene, purity, and zero-circular-import checks
# ---------------------------------------------------------------------------


def test_linux_tree_cgroup_modules_are_stdlib_only_and_torch_free() -> None:
    """Both internal and public modules stay strictly stdlib, Torch-free, and psutil-free."""

    forbidden_roots = {"torch", "experiments", "tpen_executor", "psutil", "omegaconf"}
    for rel in ("tpen/_linux_tree_cgroup.py", "tpen/process_resources.py"):
        module_path = _REPO_ROOT / rel
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_roots
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert node.module.split(".")[0] not in forbidden_roots

    # Test both import orders in fresh subprocesses to prove zero circular import.
    for import_stmt in (
        "import tpen.process_resources as pr; import tpen._linux_tree_cgroup as ltc",
        "import tpen._linux_tree_cgroup as ltc; import tpen.process_resources as pr",
    ):
        code = (
            f"import sys; {import_stmt}; "
            "t = pr.LinuxProcessTreeProbe(); "
            "c = pr.LinuxCgroupV2Probe(); "
            "assert t.target_generation is None and c.target_generation is None; "
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


def test_probe_constructors_are_pure_and_do_not_freeze_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing LinuxProcessTreeProbe / LinuxCgroupV2Probe must never call os.getpid()."""

    def _forbid_getpid() -> int:
        raise AssertionError("os.getpid() must not be called at probe construction")

    monkeypatch.setattr(os, "getpid", _forbid_getpid)
    tree_probe = LinuxProcessTreeProbe()
    cg_probe = LinuxCgroupV2Probe()
    assert tree_probe.target_generation is None
    assert tree_probe.max_scan_seconds == DEFAULT_LINUX_TREE_MAX_SCAN_SECONDS
    assert tree_probe.max_observed_members == DEFAULT_LINUX_TREE_MAX_OBSERVED_MEMBERS
    assert tree_probe.max_enumerated_candidates == DEFAULT_LINUX_TREE_MAX_ENUMERATED_CANDIDATES
    assert tree_probe.max_read_bytes == DEFAULT_LINUX_TREE_MAX_READ_BYTES
    assert cg_probe.target_generation is None
    assert cg_probe.max_read_bytes == DEFAULT_LINUX_CGROUP_MAX_READ_BYTES
    assert cg_probe.max_mountinfo_bytes == DEFAULT_LINUX_CGROUP_MAX_MOUNTINFO_BYTES
    assert cg_probe.max_read_seconds == DEFAULT_LINUX_CGROUP_MAX_READ_SECONDS
    assert cg_probe.max_ancestor_depth == DEFAULT_LINUX_CGROUP_MAX_ANCESTOR_DEPTH


# ---------------------------------------------------------------------------
# Pure /proc/<pid>/stat full-field parser tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "comm",
    ["python3", "worker (1) ) )", "line1\nline2) extra", ""],
)
def test_parse_proc_stat_fields_handles_tricky_comm(comm: str) -> None:
    """Parser uses rfind(')') so tricky comm names do not shift ppid/utime/stime/starttime."""

    raw = _make_stat_line(
        42, comm=comm, state="R", ppid=10, utime=123, stime=45, start_ticks=99999
    )
    snap = parse_proc_stat_fields(raw, expected_pid=42)
    assert snap == ProcStatSnapshot(
        pid=42, state="R", ppid=10, utime_ticks=123, stime_ticks=45, start_ticks=99999
    )


def test_parse_proc_stat_fields_handles_non_utf8_comm_bytes_and_rejects_bad_inputs() -> None:
    raw = b"77 (\xff\xfe) )) S 10 77 77 0 -1 0 0 0 0 0 33 12 0 0 20 0 1 0 4242 0 0\n"
    snap = parse_proc_stat_fields(raw, expected_pid=77)
    assert snap == ProcStatSnapshot(
        pid=77, state="S", ppid=10, utime_ticks=33, stime_ticks=12, start_ticks=4242
    )

    bad = parse_proc_stat_fields("77 (bad) S -1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20")
    assert isinstance(bad, LinuxMemoryUnavailable)
    assert bad.status == "error"
    assert bad.reason_code == "malformed_procfs"


# ---------------------------------------------------------------------------
# LinuxProcessTreeProbe: filtering, observer roles, multi-scan CPU deltas, caps
# ---------------------------------------------------------------------------


def test_tree_probe_excludes_zombies_foreign_reused_and_mismatched_ns_and_tags_observers(
    tmp_path: Path,
) -> None:
    """Tree scan excludes zombies, dead, foreign, PID-reused, and wrong-NS; tags observer PIDs."""

    proc_root = _init_proc_root(tmp_path / "proc")
    # Root PID 100, start_ticks=1000
    _write_proc_entry(proc_root, pid=100, ppid=1, start_ticks=1000, vm_rss_kib=10000)
    # Valid child PID 101 (descendant), RSS=2048 KiB
    _write_proc_entry(
        proc_root, pid=101, ppid=100, start_ticks=1100, utime=50, stime=10, vm_rss_kib=2048
    )
    # Valid grandchild PID 102 (descendant of 101), RSS=1024 KiB
    _write_proc_entry(
        proc_root, pid=102, ppid=101, start_ticks=1200, utime=20, stime=5, vm_rss_kib=1024
    )
    # Valid observer helper PID 103 (child of root, tagged role="observer"), RSS=512 KiB
    _write_proc_entry(
        proc_root, pid=103, ppid=100, start_ticks=1150, utime=10, stime=2, vm_rss_kib=512
    )
    # Zombie child PID 104 -> excluded
    _write_proc_entry(
        proc_root, pid=104, ppid=100, state="Z", start_ticks=1160, vm_rss_kib=9999
    )
    # Dead child PID 105 -> excluded
    _write_proc_entry(
        proc_root, pid=105, ppid=100, state="X", start_ticks=1170, vm_rss_kib=9999
    )
    # PID-reused candidate PID 106 (start_ticks=500 < parent root start_ticks=1000) -> excluded
    _write_proc_entry(
        proc_root, pid=106, ppid=100, start_ticks=500, vm_rss_kib=9999
    )
    # Child of PID-reused candidate PID 107 -> also excluded because parent 106 excluded
    _write_proc_entry(
        proc_root, pid=107, ppid=106, start_ticks=1300, vm_rss_kib=9999
    )
    # Foreign/unrelated process PID 200 (ppid=1) -> excluded
    _write_proc_entry(
        proc_root, pid=200, ppid=1, start_ticks=1500, vm_rss_kib=9999
    )
    # Mismatched PID namespace child PID 108 -> excluded
    _write_proc_entry(
        proc_root,
        pid=108,
        ppid=100,
        start_ticks=1180,
        vm_rss_kib=9999,
        pid_ns_target="pid:[9999999999]",
    )

    probe = LinuxProcessTreeProbe(
        pid=100,
        observer_pids=(103,),
        proc_root=proc_root,
        clock_ticks_per_second=100,
        host_id="node-a",
    )
    bound = probe.bind_target()
    assert bound is not None and bound.pid == 100

    # Scan 1: establishes baseline -> CPU deltas MUST be baseline_pending (NOT 0).
    r1 = probe.read()
    assert r1.status == "partial"
    assert r1.reason_code == "baseline_pending"
    assert r1.membership_complete is True
    assert r1.live_member_count == 3
    assert r1.sum_rss_bytes == (2048 + 1024 + 512) * 1024
    assert isinstance(r1.matched_cpu_user_delta_seconds, LinuxMemoryUnavailable)
    assert r1.matched_cpu_user_delta_seconds.reason_code == "baseline_pending"
    assert isinstance(r1.matched_cpu_system_delta_seconds, LinuxMemoryUnavailable)
    assert r1.matched_cpu_system_delta_seconds.reason_code == "baseline_pending"
    assert {m.generation.pid for m in r1.members} == {101, 102, 103}
    roles = {m.generation.pid: m.role for m in r1.members}
    assert roles == {101: "descendant", 102: "descendant", 103: "observer"}
    assert r1.excluded_count >= 5

    # Advance counters for 101 (+100u/+20s), keep 102 idle (+0/+0),
    # PID-reuse 103 (new start_ticks=2000 -> breaks 103's baseline!),
    # and spawn brand-new child 109 (no prior baseline).
    _write_proc_entry(
        proc_root, pid=101, ppid=100, start_ticks=1100, utime=150, stime=30, vm_rss_kib=2048
    )
    _write_proc_entry(
        proc_root, pid=102, ppid=101, start_ticks=1200, utime=20, stime=5, vm_rss_kib=1024
    )
    _write_proc_entry(
        proc_root, pid=103, ppid=100, start_ticks=2000, utime=500, stime=500, vm_rss_kib=512
    )
    _write_proc_entry(
        proc_root, pid=109, ppid=100, start_ticks=2100, utime=80, stime=40, vm_rss_kib=256
    )

    r2 = probe.read()
    assert r2.status == "ok"
    assert r2.membership_complete is True
    assert r2.live_member_count == 4
    assert r2.matched_delta_member_count == 2  # only 101 and 102 matched exact generation
    assert r2.matched_cpu_user_delta_seconds == pytest.approx(1.0)
    assert r2.matched_cpu_system_delta_seconds == pytest.approx(0.2)
    assert r2.sum_rss_bytes == (2048 + 1024 + 512 + 256) * 1024

    # Scan 3: counter regression on 101 breaks 101's interval; 102/103/109 advance cleanly.
    _write_proc_entry(
        proc_root, pid=101, ppid=100, start_ticks=1100, utime=10, stime=30, vm_rss_kib=2048
    )
    _write_proc_entry(
        proc_root, pid=102, ppid=101, start_ticks=1200, utime=30, stime=15, vm_rss_kib=1024
    )
    _write_proc_entry(
        proc_root, pid=103, ppid=100, start_ticks=2000, utime=510, stime=505, vm_rss_kib=512
    )
    _write_proc_entry(
        proc_root, pid=109, ppid=100, start_ticks=2100, utime=100, stime=40, vm_rss_kib=256
    )

    r3 = probe.read()
    assert r3.status == "partial"
    assert "counter_regression" in r3.partial_reasons
    assert r3.matched_delta_member_count == 3  # 102, 103, 109 matched; 101 regressed
    # Deltas: 102 (+0.10u, +0.10s) + 103 (+0.10u, +0.05s) + 109 (+0.20u, +0.00s)
    assert r3.matched_cpu_user_delta_seconds == pytest.approx(0.40)
    assert r3.matched_cpu_system_delta_seconds == pytest.approx(0.15)


def test_tree_probe_counter_regression_on_sole_member_yields_unavailable_not_zero(
    tmp_path: Path,
) -> None:
    """When the only descendant regresses CPU ticks, CPU delta is unavailable, not 0."""

    proc_root = _init_proc_root(tmp_path / "proc_reg")
    _write_proc_entry(proc_root, pid=100, ppid=1, start_ticks=1000)
    _write_proc_entry(proc_root, pid=101, ppid=100, start_ticks=1100, utime=100, stime=50)

    probe = LinuxProcessTreeProbe(pid=100, proc_root=proc_root, clock_ticks_per_second=100)
    assert probe.read().reason_code == "baseline_pending"

    _write_proc_entry(proc_root, pid=101, ppid=100, start_ticks=1100, utime=20, stime=50)
    r2 = probe.read()
    assert r2.status == "partial"
    assert "counter_regression" in r2.partial_reasons
    assert isinstance(r2.matched_cpu_user_delta_seconds, LinuxMemoryUnavailable)
    assert r2.matched_cpu_user_delta_seconds.reason_code == "counter_regression"
    assert isinstance(r2.matched_cpu_system_delta_seconds, LinuxMemoryUnavailable)
    assert r2.matched_cpu_system_delta_seconds.reason_code == "counter_regression"


def test_tree_probe_excludes_candidates_exiting_reparenting_or_reusing_between_stat_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mid-read exit, reparenting, or start_ticks drift between stat and status excludes candidate."""

    for mode in ("exit", "reparent", "reuse_start_ticks", "zombie"):
        proc_root = _init_proc_root(tmp_path / f"proc_{mode}")
        _write_proc_entry(proc_root, pid=100, ppid=1, start_ticks=1000)
        _write_proc_entry(proc_root, pid=101, ppid=100, start_ticks=1100, vm_rss_kib=5000)
        status_101 = proc_root / "101" / "status"
        stat_101 = proc_root / "101" / "stat"
        real_open = Path.open

        def _mutate_on_status(self: Path, *args: Any, **kwargs: Any) -> Any:
            handle = real_open(self, *args, **kwargs)
            if self == status_101:
                if mode == "exit":
                    stat_101.unlink()
                elif mode == "reparent":
                    stat_101.write_text(
                        _make_stat_line(101, ppid=1, start_ticks=1100), encoding="utf-8"
                    )
                elif mode == "reuse_start_ticks":
                    stat_101.write_text(
                        _make_stat_line(101, ppid=100, start_ticks=9999), encoding="utf-8"
                    )
                elif mode == "zombie":
                    stat_101.write_text(
                        _make_stat_line(101, state="Z", ppid=100, start_ticks=1100),
                        encoding="utf-8",
                    )
            return handle

        monkeypatch.setattr(Path, "open", _mutate_on_status)
        probe = LinuxProcessTreeProbe(pid=100, proc_root=proc_root)
        reading = probe.read()
        monkeypatch.setattr(Path, "open", real_open)

        assert reading.live_member_count == 0, f"mode={mode} should exclude 101"
        assert reading.sum_rss_bytes == 0
        assert reading.members == ()


def test_tree_probe_enforces_scan_caps_and_permission_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Member cap, candidate cap, scan timeout, and permission_denied report partial/unavailable."""

    # 1. max_observed_members cap
    proc_mem = _init_proc_root(tmp_path / "proc_mem_cap")
    _write_proc_entry(proc_mem, pid=100, ppid=1, start_ticks=1000)
    for cpid in range(101, 106):
        _write_proc_entry(proc_mem, pid=cpid, ppid=100, start_ticks=1100 + cpid)
    probe_mem = LinuxProcessTreeProbe(pid=100, proc_root=proc_mem, max_observed_members=2)
    r_mem = probe_mem.read()
    assert r_mem.status == "partial"
    assert r_mem.membership_complete is False
    assert "max_observed_members_exceeded" in r_mem.partial_reasons
    assert r_mem.live_member_count == 2

    # 2. max_enumerated_candidates cap
    probe_cand = LinuxProcessTreeProbe(
        pid=100, proc_root=proc_mem, max_enumerated_candidates=2
    )
    r_cand = probe_cand.read()
    assert r_cand.status == "partial"
    assert r_cand.membership_complete is False
    assert "max_enumerated_candidates_exceeded" in r_cand.partial_reasons
    assert r_cand.enumerated_candidates == 2

    # 3. max_scan_seconds cap via deterministic clock
    calls = 0

    def _timeout_after_one_child() -> int:
        nonlocal calls
        calls += 1
        # Jump past 0.1s scan limit after a few steps in the scan loop.
        if calls >= 5:
            return 1_000_000_000 + 500_000_000
        return 1_000_000_000 + calls * 1_000

    probe_time = LinuxProcessTreeProbe(
        pid=100,
        proc_root=proc_mem,
        max_scan_seconds=0.1,
        monotonic_ns=_timeout_after_one_child,
    )
    r_time = probe_time.read()
    assert r_time.membership_complete is False
    assert "max_scan_seconds_exceeded" in r_time.partial_reasons
    assert r_time.status in ("partial", "unavailable")

    # 4. permission_denied on all candidates -> unavailable (NEVER false zero!)
    proc_perm = _init_proc_root(tmp_path / "proc_perm")
    _write_proc_entry(proc_perm, pid=100, ppid=1, start_ticks=1000)
    _write_proc_entry(proc_perm, pid=101, ppid=100, start_ticks=1100)
    denied_stat = proc_perm / "101" / "stat"
    real_open = Path.open

    def _deny_101(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == denied_stat:
            raise PermissionError("EACCES")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _deny_101)
    r_perm = LinuxProcessTreeProbe(pid=100, proc_root=proc_perm).read()
    monkeypatch.setattr(Path, "open", real_open)
    assert r_perm.status == "unavailable"
    assert r_perm.membership_complete is False
    assert r_perm.permission_denied_count == 1
    assert "permission_denied" in r_perm.partial_reasons
    assert isinstance(r_perm.live_member_count, LinuxMemoryUnavailable)
    assert isinstance(r_perm.sum_rss_bytes, LinuxMemoryUnavailable)


def test_tree_probe_refuses_symlinks_fifos_optional_drift_and_non_linux(
    tmp_path: Path,
) -> None:
    """Applies O4 T2 F1-F4 hardening: symlinks, FIFOs, exact optional matching, non-Linux."""

    # Symlinked /proc/<pid> directory and FIFO status never block or follow symlinks.
    proc_root = _init_proc_root(tmp_path / "proc_sec")
    _write_proc_entry(proc_root, pid=100, ppid=1, start_ticks=1000)
    outside = tmp_path / "outside_pid"
    outside.mkdir()
    os.symlink(outside, proc_root / "101")

    _write_proc_entry(proc_root, pid=102, ppid=100, start_ticks=1100, write_status=False)
    fifo_path = proc_root / "102" / "status"
    os.mkfifo(fifo_path)

    result_holder: list[LinuxProcessTreeReading] = []

    def _run_scan() -> None:
        result_holder.append(LinuxProcessTreeProbe(pid=100, proc_root=proc_root).read())

    t = threading.Thread(target=_run_scan)
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "FIFO status blocked tree scan!"
    reading = result_holder[0]
    assert reading.membership_complete is False
    assert "unsafe_procfs_file" in reading.partial_reasons
    assert isinstance(reading.live_member_count, LinuxMemoryUnavailable)

    # Exact optional-field matching (boot_id / pid_ns_inode / host_id).
    proc_opt = _init_proc_root(tmp_path / "proc_opt", boot_id=None)
    _write_proc_entry(proc_opt, pid=100, ppid=1, start_ticks=1000, pid_ns_target=None)
    probe_opt = LinuxProcessTreeProbe(pid=100, proc_root=proc_opt, host_id="h1")
    assert probe_opt.bind_target() is not None
    # Now boot_id appears -> must fail closed with generation_changed.
    boot_file = proc_opt / "sys" / "kernel" / "random" / "boot_id"
    boot_file.parent.mkdir(parents=True, exist_ok=True)
    boot_file.write_text("boot-appeared\n", encoding="utf-8")
    r_drift = probe_opt.read()
    assert r_drift.status == "unavailable"
    assert r_drift.reason_code == "generation_changed"

    # Non-Linux platform refused BEFORE returning expected_generation.
    gen = LinuxProcessGeneration(pid=100, start_ticks=1000, host_id="h1")
    darwin_probe = LinuxProcessTreeProbe(
        expected_generation=gen, proc_root=proc_opt, platform_name="darwin"
    )
    assert darwin_probe.bind_target() is None
    assert darwin_probe.bind_error is not None
    assert darwin_probe.bind_error.reason_code == "unsupported_platform"
    assert darwin_probe.read().reason_code == "unsupported_platform"

    # bind_target(pid=...) persists self._pid even when bind fails so read() does not retarget.
    missing_probe = LinuxProcessTreeProbe(proc_root=proc_opt)
    assert missing_probe.bind_target(pid=999999) is None
    assert missing_probe.read().reason_code == "process_exited"


# ---------------------------------------------------------------------------
# Pure Linux cgroup v2 parser tests
# ---------------------------------------------------------------------------


def test_cgroup_v2_pure_parsers_normal_unlimited_v1_and_malformed() -> None:
    """Verify exact cgroup v2 path, mountinfo, memory, cpu.stat, cpu.max, cpuset, procs parsers."""

    assert parse_proc_cgroup_v2_path("0::/\n") == "/"
    assert parse_proc_cgroup_v2_path("0::/slurm/uid_10/job_2\n") == "/slurm/uid_10/job_2"

    v1_only = parse_proc_cgroup_v2_path("12:memory:/slurm/job\n1:name=systemd:/user\n")
    assert isinstance(v1_only, LinuxMemoryUnavailable)
    assert v1_only.status == "unavailable"
    assert v1_only.reason_code == "cgroup_v2_not_mounted_or_v1_only"

    for bad_path in ("0::/a/../b\n", "0::/a/./b\n", "0::/a//b\n", "0::relative/path\n", "0::/a\\b\n"):
        res = parse_proc_cgroup_v2_path(bad_path)
        assert isinstance(res, LinuxMemoryUnavailable)
        assert res.reason_code == "unsafe_cgroup_path"

    mi = (
        _make_mountinfo_line("/sys/fs/cgroup/memory", fs_type="cgroup", mount_id=21)
        + _make_mountinfo_line("/sys/fs/cgroup", fs_type="cgroup2", mount_id=22)
    )
    assert parse_proc_mountinfo_cgroup2(mi) == ("/sys/fs/cgroup",)

    mi_v1 = _make_mountinfo_line("/sys/fs/cgroup/memory", fs_type="cgroup", mount_id=21)
    assert parse_proc_mountinfo_cgroup2(mi_v1).reason_code == "cgroup_v2_not_mounted_or_v1_only"

    assert parse_cgroup_uint("0\n", field_name="memory.current") == 0
    assert parse_cgroup_memory_max("0\n") == 0
    assert parse_cgroup_memory_max("4096\n") == 4096
    unlim_mem = parse_cgroup_memory_max("max\n")
    assert isinstance(unlim_mem, LinuxMemoryUnavailable)
    assert unlim_mem.status == "unavailable"
    assert unlim_mem.reason_code == "unlimited"

    cpu_stat = parse_cgroup_cpu_stat(
        "usage_usec 1500000\nuser_usec 1000000\nsystem_usec 500000\nnr_throttled 0\n"
    )
    assert isinstance(cpu_stat, dict)
    assert cpu_stat["cpu_usage_seconds"] == pytest.approx(1.5)
    assert cpu_stat["cpu_user_seconds"] == pytest.approx(1.0)
    assert cpu_stat["cpu_system_seconds"] == pytest.approx(0.5)
    assert cpu_stat["cpu_nr_throttled"] == 0
    assert isinstance(cpu_stat["cpu_throttled_seconds"], LinuxMemoryUnavailable)
    assert cpu_stat["cpu_throttled_seconds"].reason_code == "field_missing"

    assert parse_cgroup_cpu_max("250000 100000\n") == pytest.approx(2.5)
    assert parse_cgroup_cpu_max("0 100000\n") == pytest.approx(0.0)
    unlim_cpu = parse_cgroup_cpu_max("max 100000\n")
    assert isinstance(unlim_cpu, LinuxMemoryUnavailable)
    assert unlim_cpu.reason_code == "unlimited"
    assert parse_cgroup_cpu_max("100000 0\n").reason_code == "malformed_cgroup"

    assert parse_cpuset_effective_cpus("0-3,6\n") == 5
    assert parse_cpuset_effective_cpus("0\n") == 1
    assert parse_cpuset_effective_cpus("0-3,2-5\n") == 6
    assert parse_cpuset_effective_cpus("  \n").reason_code == "field_missing"
    assert parse_cpuset_effective_cpus("5-2\n").reason_code == "malformed_cgroup"

    assert parse_cgroup_procs("100\n101\n100\n", max_procs=2) == ((100, 101), False)
    assert parse_cgroup_procs("100\n101\n102\n", max_procs=2) == ((100, 101), True)


# ---------------------------------------------------------------------------
# LinuxCgroupV2Probe fixture tests
# ---------------------------------------------------------------------------


def test_cgroup_v2_probe_normal_hierarchical_limits_and_comember_provenance(
    tmp_path: Path,
) -> None:
    """Resolves hierarchical min(memory.max) and min(cpu.max) up ancestors + co-member scope."""

    proc_root, cg_mount = _setup_cgroup_v2_fixture(
        tmp_path / "cg_hier",
        pid=2000,
        cgroup_rel_path="/slurm/uid_1000/job_42",
        memory_current="104857600",
        memory_peak="209715200",
        memory_max_by_rel={
            "/slurm": "8589934592",  # 8 GiB
            "/slurm/uid_1000": "2147483648",  # 2 GiB (tightest!)
            "/slurm/uid_1000/job_42": "max",
        },
        cpu_max_by_rel={
            "/slurm": "800000 100000",  # 8 cores
            "/slurm/uid_1000": "200000 100000",  # 2 cores (tightest!)
            "/slurm/uid_1000/job_42": "400000 100000",  # 4 cores
        },
        cpuset_effective="0-3,6",
        cgroup_procs="2000\n2001\n2002\n",
    )

    probe = LinuxCgroupV2Probe(pid=2000, proc_root=proc_root, host_id="node-cg")
    bound = probe.bind_target()
    assert bound is not None and bound.pid == 2000

    reading = probe.read()
    assert reading.status == "ok"
    assert reading.memory_current_bytes == 104_857_600
    assert reading.memory_peak_bytes == 209_715_200
    assert reading.memory_max_bytes == 2_147_483_648
    assert reading.cpu_usage_seconds == pytest.approx(2.5)
    assert reading.cpu_user_seconds == pytest.approx(2.0)
    assert reading.cpu_system_seconds == pytest.approx(0.5)
    assert reading.cpu_throttled_seconds == pytest.approx(0.15)
    assert reading.cpu_nr_throttled == 3
    assert reading.cpu_max_cores == pytest.approx(2.0)
    assert reading.cpuset_effective_cpus == 5

    ident = reading.cgroup_identity
    assert isinstance(ident, LinuxCgroupV2Identity)
    assert ident.cgroup_version == 2
    assert ident.mount_root == str(cg_mount)
    assert ident.cgroup_path == "/slurm/uid_1000/job_42"
    assert ident.effective_memory_limit_path == "/slurm/uid_1000"
    assert ident.effective_cpu_limit_path == "/slurm/uid_1000"
    assert ident.observed_member_count == 3
    assert ident.includes_other_tasks is True
    assert ident.shared_scope_not_exclusive is True

    subject = ident.to_subject_dict(target_generation=bound)
    assert subject["kind"] == "cgroup"
    assert subject["shared_scope_not_exclusive"] is True
    assert subject["includes_other_tasks"] is True
    assert subject["effective_memory_limit_path"] == "/slurm/uid_1000"
    assert subject["effective_cpu_limit_path"] == "/slurm/uid_1000"


def test_cgroup_v2_probe_unlimited_missing_peak_cpuset_fallback_and_v1_refusal(
    tmp_path: Path,
) -> None:
    """'max' -> unlimited (not 0), absent memory.peak -> field_missing (not 0), v1 refused."""

    proc_root, cg_mount = _setup_cgroup_v2_fixture(
        tmp_path / "cg_partial",
        pid=2100,
        cgroup_rel_path="/job_unlim",
        memory_peak=None,  # absent!
        memory_max_by_rel={"/job_unlim": "max"},
        cpu_max_by_rel={"/job_unlim": "max 100000"},
        cpuset_effective=None,
        cpuset_cpus="0-1",  # fallback
        cgroup_procs="2100\n",
    )
    reading = LinuxCgroupV2Probe(pid=2100, proc_root=proc_root).read()
    assert reading.status == "partial"
    assert reading.memory_current_bytes == 104_857_600
    assert isinstance(reading.memory_peak_bytes, LinuxMemoryUnavailable)
    assert reading.memory_peak_bytes.reason_code == "field_missing"
    assert not (cg_mount / "job_unlim" / "memory.peak").exists()  # strictly read-only!
    assert isinstance(reading.memory_max_bytes, LinuxMemoryUnavailable)
    assert reading.memory_max_bytes.reason_code == "unlimited"
    assert isinstance(reading.cpu_max_cores, LinuxMemoryUnavailable)
    assert reading.cpu_max_cores.reason_code == "unlimited"
    assert reading.cpuset_effective_cpus == 2
    assert reading.cgroup_identity is not None
    assert reading.cgroup_identity.includes_other_tasks is False
    assert reading.cgroup_identity.shared_scope_not_exclusive is True

    # Cgroup v1 only refusal.
    (proc_root / "2100" / "cgroup").write_text(
        "12:memory:/slurm/job_2100\n", encoding="utf-8"
    )
    v1_reading = LinuxCgroupV2Probe(pid=2100, proc_root=proc_root).read()
    assert v1_reading.status == "unavailable"
    assert v1_reading.reason_code == "cgroup_v2_not_mounted_or_v1_only"
    assert isinstance(v1_reading.memory_current_bytes, LinuxMemoryUnavailable)


def test_cgroup_v2_probe_refuses_traversal_symlink_escape_fifo_and_non_linux(
    tmp_path: Path,
) -> None:
    """Refuses path traversal, symlink escapes outside mount root, FIFOs, and non-Linux."""

    # 1. Path traversal in /proc/<pid>/cgroup
    proc_trav, _ = _setup_cgroup_v2_fixture(tmp_path / "cg_trav", pid=2200)
    (proc_trav / "2200" / "cgroup").write_text("0::/slurm/../../etc\n", encoding="utf-8")
    r_trav = LinuxCgroupV2Probe(pid=2200, proc_root=proc_trav).read()
    assert r_trav.status == "error"
    assert r_trav.reason_code == "unsafe_cgroup_path"

    # 2. Symlinked cgroup directory escaping mount root
    proc_sym, cg_mount = _setup_cgroup_v2_fixture(
        tmp_path / "cg_sym", pid=2201, cgroup_rel_path="/escape"
    )
    outside = tmp_path / "outside_cg"
    outside.mkdir()
    (outside / "memory.current").write_text("12345\n", encoding="utf-8")
    # Replace /escape dir with symlink to outside
    for child in (cg_mount / "escape").iterdir():
        child.unlink()
    (cg_mount / "escape").rmdir()
    os.symlink(outside, cg_mount / "escape")
    r_sym = LinuxCgroupV2Probe(pid=2201, proc_root=proc_sym).read()
    assert r_sym.status == "error"
    assert r_sym.reason_code == "unsafe_cgroup_path"

    # 3. FIFO memory.current never blocks
    proc_fifo, cg_fifo = _setup_cgroup_v2_fixture(
        tmp_path / "cg_fifo", pid=2202, cgroup_rel_path="/fifo_cg", memory_current=None
    )
    os.mkfifo(cg_fifo / "fifo_cg" / "memory.current")
    out: list[LinuxCgroupV2Reading] = []
    t = threading.Thread(
        target=lambda: out.append(LinuxCgroupV2Probe(pid=2202, proc_root=proc_fifo).read())
    )
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "FIFO memory.current blocked cgroup probe!"
    assert isinstance(out[0].memory_current_bytes, LinuxMemoryUnavailable)
    assert out[0].memory_current_bytes.reason_code == "unsafe_cgroup_path"

    # 4. Non-Linux platform refusal without fallback
    gen = LinuxProcessGeneration(pid=2200, start_ticks=5000, host_id="h1")
    darwin_probe = LinuxCgroupV2Probe(
        expected_generation=gen, proc_root=proc_trav, platform_name="darwin"
    )
    assert darwin_probe.bind_target() is None
    assert darwin_probe.bind_error is not None
    assert darwin_probe.bind_error.reason_code == "unsupported_platform"
    assert darwin_probe.read().reason_code == "unsupported_platform"


def test_cgroup_v2_streaming_mountinfo_large_non_utf8_limit_and_unsafe_files(
    tmp_path: Path,
) -> None:
    """Streaming mountinfo handles >64 KiB + non-UTF-8 unrelated mounts, limits, symlinks, FIFOs."""

    # 1. Synthetic ~200 KiB /proc/<pid>/mountinfo (~900 unrelated NFS/autofs mounts,
    # including raw non-UTF-8 bytes in an unrelated NFS mount path + cgroup2 at the end
    # without a trailing newline) succeeds under default LinuxCgroupV2Probe() (max_read_bytes=65536).
    proc_big, cg_big = _setup_cgroup_v2_fixture(
        tmp_path / "cg_big_mi",
        pid=2300,
        cgroup_rel_path="/system.slice/slurmstepd.scope/job_42",
    )
    mount_lines: list[bytes] = []
    for idx in range(900):
        if idx == 450:
            mount_lines.append(
                b"550 28 0:99 / /mnt/nfs/raw_\xff\xfe_export/"
                + (b"x" * 160)
                + b" rw,relatime shared:4 - nfs4 server:/raw_export rw\n"
            )
        else:
            mount_lines.append(
                (
                    f"{100 + idx} 28 0:{100 + idx} / /mnt/autofs/export_{idx:04d}/"
                    + ("p" * 160)
                    + " rw,relatime shared:4 - nfs4 nfs.rc.example:/export rw\n"
                ).encode("ascii")
            )
    # Append cgroup2 mount at the very end WITHOUT a trailing newline to exercise EOF flush.
    mount_lines.append(
        f"29 22 0:25 / {cg_big} rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw".encode(
            "utf-8"
        )
    )
    big_mi_bytes = b"".join(mount_lines)
    assert len(big_mi_bytes) > 180_000
    (proc_big / "2300" / "mountinfo").write_bytes(big_mi_bytes)

    # Also prove parse_proc_mountinfo_cgroup2 directly skips raw non-UTF-8 unrelated mounts.
    assert parse_proc_mountinfo_cgroup2(big_mi_bytes) == (str(cg_big),)

    default_probe = LinuxCgroupV2Probe(pid=2300, proc_root=proc_big)
    assert default_probe.max_read_bytes == DEFAULT_LINUX_CGROUP_MAX_READ_BYTES
    assert default_probe.max_mountinfo_bytes == DEFAULT_LINUX_CGROUP_MAX_MOUNTINFO_BYTES
    r_big = default_probe.read()
    assert r_big.status == "ok"
    assert r_big.memory_current_bytes == 104_857_600
    assert r_big.cgroup_identity is not None
    assert r_big.cgroup_identity.mount_root == str(cg_big)
    assert r_big.cgroup_identity.cgroup_path == "/system.slice/slurmstepd.scope/job_42"

    # Also verify /proc/self/mountinfo fallback routes through streaming bounded reader.
    (proc_big / "2300" / "mountinfo").unlink()
    self_dir = proc_big / "self"
    self_dir.mkdir(exist_ok=True)
    (self_dir / "mountinfo").write_bytes(big_mi_bytes)
    r_self_fallback = LinuxCgroupV2Probe(pid=2300, proc_root=proc_big).read()
    assert r_self_fallback.status == "ok"
    assert r_self_fallback.cgroup_identity is not None
    assert r_self_fallback.cgroup_identity.mount_root == str(cg_big)
    # Restore /proc/2300/mountinfo for the limit test below.
    (proc_big / "2300" / "mountinfo").write_bytes(big_mi_bytes)

    # 2. Synthetic mountinfo exceeding max_mountinfo_bytes returns input_limit_exceeded error.
    capped_probe = LinuxCgroupV2Probe(
        pid=2300, proc_root=proc_big, max_mountinfo_bytes=65536
    )
    r_cap = capped_probe.read()
    assert r_cap.status == "error"
    assert r_cap.reason_code == "input_limit_exceeded"
    assert isinstance(r_cap.memory_current_bytes, LinuxMemoryUnavailable)
    assert r_cap.memory_current_bytes.status == "error"
    assert r_cap.memory_current_bytes.reason_code == "input_limit_exceeded"

    # 3. Symlinked /proc/<pid>/mountinfo refused with unsafe_procfs_file.
    proc_sym, _ = _setup_cgroup_v2_fixture(tmp_path / "cg_mi_sym", pid=2301)
    real_mi = proc_sym / "2301" / "mountinfo"
    outside_mi = tmp_path / "outside_mountinfo"
    outside_mi.write_bytes(real_mi.read_bytes())
    real_mi.unlink()
    os.symlink(outside_mi, real_mi)
    r_sym = LinuxCgroupV2Probe(pid=2301, proc_root=proc_sym).read()
    assert r_sym.status == "error"
    assert r_sym.reason_code == "unsafe_procfs_file"

    # 4. FIFO /proc/<pid>/mountinfo refused non-blockingly with unsafe_procfs_file.
    proc_fifo, _ = _setup_cgroup_v2_fixture(tmp_path / "cg_mi_fifo", pid=2302)
    fifo_mi = proc_fifo / "2302" / "mountinfo"
    fifo_mi.unlink()
    os.mkfifo(fifo_mi)
    fifo_out: list[LinuxCgroupV2Reading] = []
    t = threading.Thread(
        target=lambda: fifo_out.append(
            LinuxCgroupV2Probe(pid=2302, proc_root=proc_fifo).read()
        )
    )
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "FIFO /proc/<pid>/mountinfo blocked cgroup probe!"
    assert fifo_out[0].status == "error"
    assert fifo_out[0].reason_code == "unsafe_procfs_file"


def test_tree_probe_scan1_detail_deduplicates_baseline_pending(tmp_path: Path) -> None:
    """Scan-1 partial detail records the baseline_pending clause once, not duplicated."""

    proc_root = _init_proc_root(tmp_path / "proc_dedup")
    _write_proc_entry(proc_root, pid=300, ppid=1, start_ticks=1000)
    _write_proc_entry(proc_root, pid=301, ppid=300, start_ticks=1100)

    r1 = LinuxProcessTreeProbe(pid=300, proc_root=proc_root).read()
    assert r1.status == "partial"
    assert r1.reason_code == "baseline_pending"
    assert r1.detail is not None
    clause = "initial scan establishes per-member descendant CPU baseline"
    assert r1.detail.count(clause) == 1
    parts = [p.strip() for p in r1.detail.split(";")]
    assert len(parts) == len(set(parts))


def test_cgroup_v2_large_cgroup_procs_truncates_without_failing_metrics(
    tmp_path: Path,
) -> None:
    """Oversized cgroup.procs records bounded truncation without failing valid metrics."""

    large_procs = "".join(f"{10000 + i}\n" for i in range(15000))
    assert len(large_procs.encode("ascii")) > DEFAULT_LINUX_CGROUP_MAX_READ_BYTES
    proc_root, _ = _setup_cgroup_v2_fixture(
        tmp_path / "cg_big_procs",
        pid=2400,
        cgroup_rel_path="/shared_slice/job_large",
        cgroup_procs=large_procs,
    )
    reading = LinuxCgroupV2Probe(pid=2400, proc_root=proc_root).read()
    assert reading.status == "ok"
    assert reading.memory_current_bytes == 104_857_600
    assert reading.cpu_usage_seconds == pytest.approx(2.5)
    assert reading.cgroup_identity is not None
    assert reading.cgroup_identity.procs_truncated is True
    assert reading.cgroup_identity.includes_other_tasks is True
    assert reading.cgroup_identity.observed_member_count is None
    assert reading.cgroup_identity.shared_scope_not_exclusive is True

    # Symlinked cgroup.procs still fails closed with unsafe_cgroup_path.
    proc_sym, cg_sym = _setup_cgroup_v2_fixture(
        tmp_path / "cg_procs_sym",
        pid=2401,
        cgroup_rel_path="/shared_slice/job_sym",
    )
    procs_file = cg_sym / "shared_slice" / "job_sym" / "cgroup.procs"
    outside_procs = tmp_path / "outside_procs"
    outside_procs.write_text("2401\n", encoding="utf-8")
    procs_file.unlink()
    os.symlink(outside_procs, procs_file)
    r_sym = LinuxCgroupV2Probe(pid=2401, proc_root=proc_sym).read()
    assert r_sym.status == "error"
    assert r_sym.reason_code == "unsafe_cgroup_path"


# ---------------------------------------------------------------------------
# Live Linux /proc smoke tests (Cannon / Linux only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux /proc filesystem")
def test_live_linux_tree_and_cgroup_v2_probes_on_real_procfs() -> None:
    """On Linux (Cannon), tree probe observes a live child and cgroup probe handles v2 or v1 honestly."""

    tree_probe = LinuxProcessTreeProbe()
    bound = tree_probe.bind_target()
    assert bound is not None and bound.pid == os.getpid()

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        r1 = tree_probe.read()
        assert r1.target_generation == bound
        assert r1.status in ("partial", "ok")
        if r1.detail is not None:
            assert (
                r1.detail.count(
                    "initial scan establishes per-member descendant CPU baseline"
                )
                <= 1
            )
        assert isinstance(r1.live_member_count, int) and r1.live_member_count >= 1
        assert any(m.generation.pid == child.pid for m in r1.members)
        assert isinstance(r1.matched_cpu_user_delta_seconds, LinuxMemoryUnavailable)
        assert r1.matched_cpu_user_delta_seconds.reason_code == "baseline_pending"

        r2 = tree_probe.read()
        assert isinstance(r2.live_member_count, int) and r2.live_member_count >= 1
        assert isinstance(r2.matched_cpu_user_delta_seconds, float)
        assert r2.matched_cpu_user_delta_seconds >= 0.0
    finally:
        child.terminate()
        child.wait(timeout=5)

    cg_probe = LinuxCgroupV2Probe()
    cg_bound = cg_probe.bind_target()
    assert cg_bound is not None and cg_bound.pid == os.getpid()
    cg_reading = cg_probe.read()
    assert cg_reading.target_generation == cg_bound
    assert cg_reading.status != "error"
    if cg_reading.status == "unavailable":
        # Cannon compute nodes run Linux 4.18 with unified cgroup v2 mounted at /sys/fs/cgroup
        # (not cgroup v1); non-v2 Linux hosts still return honest unavailability, never guessed 0.
        assert cg_reading.reason_code in (
            "cgroup_v2_not_mounted_or_v1_only",
            "permission_denied",
            "cgroup_path_missing",
        )
        assert isinstance(cg_reading.memory_current_bytes, LinuxMemoryUnavailable)
    else:
        assert cg_reading.status in ("ok", "partial")
        assert cg_reading.cgroup_identity is not None
        assert cg_reading.cgroup_identity.cgroup_version == 2
        assert cg_reading.cgroup_identity.shared_scope_not_exclusive is True
        assert cg_reading.cgroup_identity.cgroup_path.startswith("/")
        for field_val in (
            cg_reading.memory_current_bytes,
            cg_reading.memory_peak_bytes,
            cg_reading.memory_max_bytes,
            cg_reading.cpu_usage_seconds,
            cg_reading.cpu_user_seconds,
            cg_reading.cpu_system_seconds,
            cg_reading.cpu_throttled_seconds,
            cg_reading.cpu_nr_throttled,
            cg_reading.cpu_max_cores,
            cg_reading.cpuset_effective_cpus,
        ):
            assert not (
                isinstance(field_val, LinuxMemoryUnavailable)
                and field_val.status == "error"
            )


# ---------------------------------------------------------------------------
# O5 T2 Round-1 review tests (F2a, F3, F5 defect checks + T6, T7a coverage)
# ---------------------------------------------------------------------------


def test_o5_review_f2a_resolve_cgroup_identity_enforces_target_generation_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2a: resolve_cgroup_identity() must enforce bound generation and max_read_seconds."""

    failures: list[str] = []

    # Sub-case A: PID reuse before resolve_cgroup_identity() must fail closed with generation_changed.
    proc_root, cg_mount = _setup_cgroup_v2_fixture(
        tmp_path / "f2a_reuse",
        pid=2500,
        start_ticks=5000,
        cgroup_rel_path="/slurm/uid_1000/job_old",
    )
    probe = LinuxCgroupV2Probe(pid=2500, proc_root=proc_root, host_id="node-cg")
    bound = probe.bind_target()
    assert bound is not None and bound.start_ticks == 5000

    # Simulate PID reuse: new start_ticks and different cgroup path.
    (proc_root / "2500" / "stat").write_text(
        _make_stat_line(2500, ppid=1, start_ticks=9999), encoding="utf-8"
    )
    (proc_root / "2500" / "cgroup").write_text(
        "0::/slurm/uid_1000/job_reused\n", encoding="utf-8"
    )
    (cg_mount / "slurm" / "uid_1000" / "job_reused").mkdir(parents=True, exist_ok=True)

    res = probe.resolve_cgroup_identity()
    if not (
        isinstance(res, LinuxMemoryUnavailable)
        and res.status == "unavailable"
        and res.reason_code == "generation_changed"
    ):
        failures.append(
            "Sub-case A (PID reuse before resolve_cgroup_identity): expected "
            f"LinuxMemoryUnavailable(status='unavailable', reason_code='generation_changed'), got {res!r}"
        )

    # Sub-case B: max_read_seconds timeout enforcement on small (<16 KiB) files in resolve_cgroup_identity().
    proc_t, _ = _setup_cgroup_v2_fixture(
        tmp_path / "f2a_timeout",
        pid=2501,
        start_ticks=5001,
        cgroup_rel_path="/slurm/uid_1000/job_t",
    )
    mountinfo_path = proc_t / "2501" / "mountinfo"
    now_ns = [1_000_000_000]
    real_open = Path.open

    class _ClockAdvancingHandle:
        """Proxy file handle that advances the test clock past deadline after mountinfo is read."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __enter__(self) -> _ClockAdvancingHandle:
            self._inner.__enter__()
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
            try:
                return self._inner.__exit__(exc_type, exc, tb)
            finally:
                # Advance clock 5s (> max_read_seconds=0.1s) after mountinfo finishes reading.
                now_ns[0] = 6_000_000_000

        def read(self, *args: Any, **kwargs: Any) -> Any:
            return self._inner.read(*args, **kwargs)

        def fileno(self) -> int:
            return self._inner.fileno()

    def _open_with_timeout_hook(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        if self == mountinfo_path:
            return _ClockAdvancingHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", _open_with_timeout_hook)
    try:
        probe_t = LinuxCgroupV2Probe(
            pid=2501,
            proc_root=proc_t,
            host_id="node-cg",
            max_read_seconds=0.1,
            monotonic_ns=lambda: now_ns[0],
        )
        res_t = probe_t.resolve_cgroup_identity()
    finally:
        monkeypatch.setattr(Path, "open", real_open)

    if not (
        isinstance(res_t, LinuxMemoryUnavailable)
        and res_t.status == "error"
        and res_t.reason_code == "read_timeout"
    ):
        failures.append(
            "Sub-case B (max_read_seconds timeout in resolve_cgroup_identity): expected "
            f"LinuxMemoryUnavailable(status='error', reason_code='read_timeout'), got {res_t!r}"
        )

    assert failures == [], "; ".join(failures)


def test_o5_review_f3_expected_generation_with_default_host_id_none_matches_like_o4(
    tmp_path: Path,
) -> None:
    """F3: expected_generation with default host_id=None matches in O5 probes just like O4."""

    proc_root, _ = _setup_cgroup_v2_fixture(
        tmp_path / "f3_host_id", pid=2600, start_ticks=5000
    )
    _write_proc_entry(
        proc_root, pid=2601, ppid=2600, start_ticks=5100, vm_rss_kib=1024
    )
    gen_default_host = LinuxProcessGeneration(
        pid=2600,
        start_ticks=5000,
        boot_id="00000000-0000-4000-8000-000000000002",
        pid_ns_inode="pid:[4026531836]",
    )
    assert gen_default_host.host_id is None

    # Confirm O4 parity: LinuxProcessMemoryProbe accepts default host_id=None.
    o4_reading = LinuxProcessMemoryProbe(
        expected_generation=gen_default_host, proc_root=proc_root
    ).read()
    assert o4_reading.status == "ok"

    failures: list[str] = []

    # Tree probe with default host_id=None must match rather than fail with generation_changed.
    tp = LinuxProcessTreeProbe(
        expected_generation=gen_default_host, proc_root=proc_root
    )
    tr1 = tp.read()
    if not (
        tr1.status == "partial"
        and tr1.reason_code == "baseline_pending"
        and tr1.live_member_count == 1
    ):
        failures.append(
            "Tree probe scan 1 with host_id=None: expected status='partial', "
            f"reason_code='baseline_pending', live_member_count=1; got "
            f"status={tr1.status!r}, reason_code={tr1.reason_code!r}, "
            f"live_member_count={tr1.live_member_count!r}"
        )
    tr2 = tp.read()
    if not (tr2.status == "ok" and tr2.live_member_count == 1):
        failures.append(
            "Tree probe scan 2 with host_id=None: expected status='ok', "
            f"live_member_count=1; got status={tr2.status!r}, "
            f"reason_code={tr2.reason_code!r}, live_member_count={tr2.live_member_count!r}"
        )

    # Cgroup probe with default host_id=None must match rather than fail with generation_changed.
    cp = LinuxCgroupV2Probe(expected_generation=gen_default_host, proc_root=proc_root)
    cr = cp.read()
    if not (cr.status == "ok" and cr.memory_current_bytes == 104_857_600):
        failures.append(
            "Cgroup probe with host_id=None: expected status='ok', "
            f"memory_current_bytes=104857600; got status={cr.status!r}, "
            f"reason_code={cr.reason_code!r}, memory_current_bytes={cr.memory_current_bytes!r}"
        )

    # Positive control: explicit mismatched host_id must still fail with generation_changed.
    gen_wrong_host = LinuxProcessGeneration(
        pid=2600,
        start_ticks=5000,
        boot_id="00000000-0000-4000-8000-000000000002",
        pid_ns_inode="pid:[4026531836]",
        host_id="different-host-xyz",
    )
    tr_wrong = LinuxProcessTreeProbe(
        expected_generation=gen_wrong_host, proc_root=proc_root
    ).read()
    if not (
        tr_wrong.status == "unavailable"
        and tr_wrong.reason_code == "generation_changed"
    ):
        failures.append(
            "Positive control tree probe with wrong host_id: expected "
            f"unavailable/generation_changed, got {tr_wrong.status!r}/{tr_wrong.reason_code!r}"
        )
    cr_wrong = LinuxCgroupV2Probe(
        expected_generation=gen_wrong_host, proc_root=proc_root
    ).read()
    if not (
        cr_wrong.status == "unavailable"
        and cr_wrong.reason_code == "generation_changed"
    ):
        failures.append(
            "Positive control cgroup probe with wrong host_id: expected "
            f"unavailable/generation_changed, got {cr_wrong.status!r}/{cr_wrong.reason_code!r}"
        )

    assert failures == [], "; ".join(failures)


def test_o5_review_f5_tree_probe_unavailable_incomplete_scan_does_not_fabricate_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5: an unavailable incomplete scan must not commit _has_baseline_scan=True."""

    proc_root = _init_proc_root(tmp_path / "f5_baseline")
    _write_proc_entry(proc_root, pid=400, ppid=1, start_ticks=1000)
    _write_proc_entry(proc_root, pid=401, ppid=400, start_ticks=1100)

    probe = LinuxProcessTreeProbe(pid=400, proc_root=proc_root)
    denied_stat = proc_root / "401" / "stat"
    real_open = Path.open

    def _deny_401(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == denied_stat:
            raise PermissionError("EACCES")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _deny_401)
    try:
        r1 = probe.read()
    finally:
        monkeypatch.setattr(Path, "open", real_open)

    assert r1.status == "unavailable"
    assert r1.reason_code == "permission_denied"
    assert r1.membership_complete is False
    assert r1.observed_member_count == 0

    # Remove PID 401 so Scan 2 has 0 live descendants and membership_complete=True.
    shutil.rmtree(proc_root / "401")

    r2 = probe.read()
    assert r2.status == "partial"
    assert r2.reason_code == "baseline_pending"
    assert r2.live_member_count == 0
    assert isinstance(r2.matched_cpu_user_delta_seconds, LinuxMemoryUnavailable)
    assert r2.matched_cpu_user_delta_seconds.reason_code == "baseline_pending"
    assert isinstance(r2.matched_cpu_system_delta_seconds, LinuxMemoryUnavailable)
    assert r2.matched_cpu_system_delta_seconds.reason_code == "baseline_pending"

    r3 = probe.read()
    assert r3.status == "ok"
    assert r3.live_member_count == 0
    assert r3.matched_cpu_user_delta_seconds == 0.0
    assert r3.matched_cpu_system_delta_seconds == 0.0


def test_o5_review_t6_cgroup_ancestor_fail_closed_depth_cap_and_mountinfo_escaping_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T6: ancestor limit fail-closed, depth cap, octal-escaped mountinfo, root mount precedence."""

    # Sub-case A: ancestor memory.max raises PermissionError -> fails closed (not leaf's looser 4 GiB).
    proc_a, cg_a = _setup_cgroup_v2_fixture(
        tmp_path / "t6_perm",
        pid=2700,
        cgroup_rel_path="/slurm/uid_1000/job_42",
        memory_max_by_rel={
            "/slurm/uid_1000": "2147483648",
            "/slurm/uid_1000/job_42": "4294967296",
        },
    )
    denied_anc_mem = cg_a / "slurm" / "uid_1000" / "memory.max"
    real_open = Path.open

    def _deny_anc_mem(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == denied_anc_mem:
            raise PermissionError("EACCES ancestor memory.max")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _deny_anc_mem)
    try:
        r_a = LinuxCgroupV2Probe(pid=2700, proc_root=proc_a).read()
    finally:
        monkeypatch.setattr(Path, "open", real_open)

    assert r_a.status == "partial"
    assert isinstance(r_a.memory_max_bytes, LinuxMemoryUnavailable)
    assert r_a.memory_max_bytes.status == "unavailable"
    assert r_a.memory_max_bytes.reason_code == "permission_denied"
    assert r_a.cgroup_identity is not None
    assert r_a.cgroup_identity.effective_memory_limit_path is None
    assert r_a.memory_current_bytes == 104_857_600
    assert r_a.cpu_max_cores == pytest.approx(4.0)

    # Sub-case B: malformed ancestor cpu.max -> fails closed (not leaf's looser 4.0).
    proc_b, _ = _setup_cgroup_v2_fixture(
        tmp_path / "t6_malformed_cpu",
        pid=2701,
        cgroup_rel_path="/slurm/uid_1000/job_42",
        cpu_max_by_rel={
            "/slurm/uid_1000": "not-a-quota 100000",
            "/slurm/uid_1000/job_42": "400000 100000",
        },
    )
    r_b = LinuxCgroupV2Probe(pid=2701, proc_root=proc_b).read()
    assert r_b.status == "partial"
    assert isinstance(r_b.cpu_max_cores, LinuxMemoryUnavailable)
    assert r_b.cpu_max_cores.status == "error"
    assert r_b.cpu_max_cores.reason_code == "malformed_cgroup"
    assert r_b.cgroup_identity is not None
    assert r_b.cgroup_identity.effective_cpu_limit_path is None
    assert r_b.memory_current_bytes == 104_857_600

    # Sub-case C: symlinked ancestor memory.max -> fails closed with unsafe_cgroup_path.
    proc_c, cg_c = _setup_cgroup_v2_fixture(
        tmp_path / "t6_symlink_mem",
        pid=2702,
        cgroup_rel_path="/slurm/uid_1000/job_42",
        memory_max_by_rel={"/slurm/uid_1000/job_42": "4294967296"},
    )
    outside_mem = tmp_path / "t6_outside_memory.max"
    outside_mem.write_text("1073741824\n", encoding="utf-8")
    os.symlink(outside_mem, cg_c / "slurm" / "uid_1000" / "memory.max")
    r_c = LinuxCgroupV2Probe(pid=2702, proc_root=proc_c).read()
    assert r_c.status == "partial"
    assert isinstance(r_c.memory_max_bytes, LinuxMemoryUnavailable)
    assert r_c.memory_max_bytes.status == "error"
    assert r_c.memory_max_bytes.reason_code == "unsafe_cgroup_path"
    assert r_c.cgroup_identity is not None
    assert r_c.cgroup_identity.effective_memory_limit_path is None
    assert r_c.memory_current_bytes == 104_857_600

    # Sub-case D: max_ancestor_depth=2 with 3-segment cgroup path -> ancestor_depth_exceeded.
    proc_d, _ = _setup_cgroup_v2_fixture(
        tmp_path / "t6_depth",
        pid=2703,
        cgroup_rel_path="/slurm/uid_1000/job_42",
    )
    r_d = LinuxCgroupV2Probe(pid=2703, proc_root=proc_d, max_ancestor_depth=2).read()
    assert r_d.status == "error"
    assert r_d.reason_code == "ancestor_depth_exceeded"

    # Sub-case E: octal-escaped space (\040) in mountinfo + root mount ('/') precedes non-root ('/sub').
    mi_esc = _make_mountinfo_line(
        "/mnt/sub\\040cg", fs_type="cgroup2", fs_root="/sub", mount_id=35
    ) + _make_mountinfo_line(
        "/mnt/cg\\040mount", fs_type="cgroup2", fs_root="/", mount_id=36
    )
    assert parse_proc_mountinfo_cgroup2(mi_esc) == ("/mnt/cg mount", "/mnt/sub cg")

    base_e = tmp_path / "t6_esc"
    proc_e = _init_proc_root(base_e / "proc")
    pid_dir_e = _write_proc_entry(proc_e, pid=2704, ppid=1, start_ticks=5000)
    sub_mount = base_e / "sub mount"
    sub_mount.mkdir(parents=True, exist_ok=True)
    root_mount = base_e / "cg mount"
    leaf_e = root_mount / "job_esc"
    leaf_e.mkdir(parents=True, exist_ok=True)
    (leaf_e / "memory.current").write_text("104857600\n", encoding="utf-8")
    (leaf_e / "memory.peak").write_text("209715200\n", encoding="utf-8")
    (leaf_e / "memory.max").write_text("4294967296\n", encoding="utf-8")
    (leaf_e / "cpu.stat").write_text(
        "usage_usec 2500000\nuser_usec 2000000\nsystem_usec 500000\n"
        "nr_throttled 3\nthrottled_usec 150000\n",
        encoding="utf-8",
    )
    (leaf_e / "cpu.max").write_text("400000 100000\n", encoding="utf-8")
    (leaf_e / "cpuset.cpus.effective").write_text("0-3,6\n", encoding="utf-8")
    (leaf_e / "cgroup.procs").write_text("2704\n", encoding="utf-8")
    (pid_dir_e / "cgroup").write_text("0::/job_esc\n", encoding="utf-8")
    (pid_dir_e / "mountinfo").write_text(
        _make_mountinfo_line(
            str(sub_mount).replace(" ", "\\040"),
            fs_type="cgroup2",
            fs_root="/sub",
            mount_id=35,
        )
        + _make_mountinfo_line(
            str(root_mount).replace(" ", "\\040"),
            fs_type="cgroup2",
            fs_root="/",
            mount_id=36,
        ),
        encoding="utf-8",
    )
    r_e = LinuxCgroupV2Probe(pid=2704, proc_root=proc_e).read()
    assert r_e.status == "ok"
    assert r_e.cgroup_identity is not None
    assert r_e.cgroup_identity.mount_root == str(root_mount)
    assert r_e.cgroup_identity.cgroup_path == "/job_esc"
    assert r_e.memory_current_bytes == 104_857_600


def test_o5_review_t7a_mid_read_cgroup_changed_and_post_read_generation_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7a: mid-read cgroup migration and post-read generation drift fail closed without committing state."""

    real_open = Path.open

    # Sub-case A: mid-read cgroup migration in LinuxCgroupV2Probe.read() -> cgroup_changed.
    proc_a, cg_a = _setup_cgroup_v2_fixture(
        tmp_path / "t7a_cg_mig",
        pid=2800,
        start_ticks=5000,
        cgroup_rel_path="/slurm/uid_1000/job_orig",
    )
    probe_a = LinuxCgroupV2Probe(pid=2800, proc_root=proc_a, host_id="node-cg")
    r_a0 = probe_a.read()
    assert r_a0.status == "ok"
    assert probe_a.cgroup_identity is not None
    assert probe_a.cgroup_identity.cgroup_path == "/slurm/uid_1000/job_orig"

    mem_cur_a = cg_a / "slurm" / "uid_1000" / "job_orig" / "memory.current"

    def _mutate_cgroup_on_mem_current(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        if self == mem_cur_a:
            (proc_a / "2800" / "cgroup").write_text(
                "0::/slurm/uid_1000/job_other\n", encoding="utf-8"
            )
        return handle

    monkeypatch.setattr(Path, "open", _mutate_cgroup_on_mem_current)
    try:
        r_a = probe_a.read()
    finally:
        monkeypatch.setattr(Path, "open", real_open)

    assert r_a.status == "unavailable"
    assert r_a.reason_code == "cgroup_changed"
    for cg_field in (
        r_a.memory_current_bytes,
        r_a.memory_peak_bytes,
        r_a.memory_max_bytes,
        r_a.cpu_usage_seconds,
        r_a.cpu_user_seconds,
        r_a.cpu_system_seconds,
        r_a.cpu_throttled_seconds,
        r_a.cpu_nr_throttled,
        r_a.cpu_max_cores,
        r_a.cpuset_effective_cpus,
    ):
        assert isinstance(cg_field, LinuxMemoryUnavailable)
    assert probe_a.cgroup_identity is not None
    assert probe_a.cgroup_identity.cgroup_path == "/slurm/uid_1000/job_orig"

    # Sub-case B: post-read target generation drift in LinuxCgroupV2Probe.read() -> generation_changed.
    proc_b, cg_b = _setup_cgroup_v2_fixture(
        tmp_path / "t7a_cg_gen",
        pid=2801,
        start_ticks=5000,
        cgroup_rel_path="/slurm/uid_1000/job_gen",
    )
    probe_b = LinuxCgroupV2Probe(pid=2801, proc_root=proc_b, host_id="node-cg")
    mem_cur_b = cg_b / "slurm" / "uid_1000" / "job_gen" / "memory.current"

    def _mutate_stat_on_mem_current(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        if self == mem_cur_b:
            (proc_b / "2801" / "stat").write_text(
                _make_stat_line(2801, ppid=1, start_ticks=9999), encoding="utf-8"
            )
        return handle

    monkeypatch.setattr(Path, "open", _mutate_stat_on_mem_current)
    try:
        r_b = probe_b.read()
    finally:
        monkeypatch.setattr(Path, "open", real_open)

    assert r_b.status == "unavailable"
    assert r_b.reason_code == "generation_changed"
    for cg_field in (
        r_b.memory_current_bytes,
        r_b.memory_peak_bytes,
        r_b.memory_max_bytes,
        r_b.cpu_usage_seconds,
        r_b.cpu_user_seconds,
        r_b.cpu_system_seconds,
        r_b.cpu_throttled_seconds,
        r_b.cpu_nr_throttled,
        r_b.cpu_max_cores,
        r_b.cpuset_effective_cpus,
    ):
        assert isinstance(cg_field, LinuxMemoryUnavailable)

    # Sub-case C: post-scan root generation drift in LinuxProcessTreeProbe.read() -> generation_changed
    # without committing baseline state.
    proc_c = _init_proc_root(tmp_path / "t7a_tree_gen")
    _write_proc_entry(proc_c, pid=2810, ppid=1, start_ticks=5000)
    _write_proc_entry(proc_c, pid=2811, ppid=2810, start_ticks=5100)
    probe_c = LinuxProcessTreeProbe(pid=2810, proc_root=proc_c, host_id="node-cg")
    child_status = proc_c / "2811" / "status"

    def _mutate_root_stat_on_child_status(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        if self == child_status:
            (proc_c / "2810" / "stat").write_text(
                _make_stat_line(2810, ppid=1, start_ticks=9999), encoding="utf-8"
            )
        return handle

    monkeypatch.setattr(Path, "open", _mutate_root_stat_on_child_status)
    try:
        r_c = probe_c.read()
    finally:
        monkeypatch.setattr(Path, "open", real_open)

    assert r_c.status == "unavailable"
    assert r_c.reason_code == "generation_changed"
    for tree_field in (
        r_c.live_member_count,
        r_c.sum_rss_bytes,
        r_c.matched_cpu_user_delta_seconds,
        r_c.matched_cpu_system_delta_seconds,
    ):
        assert isinstance(tree_field, LinuxMemoryUnavailable)
    assert probe_c._has_baseline_scan is False
    assert probe_c._member_baselines == {}

