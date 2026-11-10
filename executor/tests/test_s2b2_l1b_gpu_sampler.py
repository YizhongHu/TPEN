"""S2b-2 L1b: in-child GPU sampler, v3 placement receipt, and contract mutants.

The suite is executor-only.  Process-lifecycle tests drive the real
:mod:`tpen_executor._child_gpu_sampler` code (the exact source RUN_SHIM
embeds) against deterministic fake ``nvidia-smi`` scripts; receipt tests go
through the real builder and decoder.  The four end-to-end producer outcomes
crossing into the P8 reader live in
``experiments/toolkit/test_review_p8_toolkit_r3.py``.

Historical mutant ids (review-l1 r1/delta/r3): N6 (late-failure sample
preservation), N8 (qualified-unavailable decoding), and r3 N11 (cumulative
Slurm expansion) are NOT closed by renaming.  N6/N8 objectives are covered by
named behaviour tests here (``test_sampler_keeps_samples_taken_before_a_late_failure``,
``test_out_of_slot_use_in_any_sample_dominates`` and the
``qualified-over-failed-sampler`` decoder case); N11 is L1a host-map scope
and is not touched by this layer.  The L1b mutants below use new ids K1-K5.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from tpen_executor import _child_gpu_sampler as child
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.cli import _operational_exit
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.dispatch.contract import _placement_receipt_v3
from tpen_executor.dispatch.local import LocalDispatcher
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.gpu_telemetry import not_sampled_sampler, validate_sampler_wire
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.placement import AllocationSlot, ChildPlacementObservation, ChildPlacementObservationV2
from tpen_executor.results import (
    PLACEMENT_NOT_SAMPLED_MESSAGE,
    PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE,
    PLACEMENT_UNQUALIFIED_MESSAGE,
    ManifestAttempt,
    OutcomeReceipt,
    RunManifest,
    SciencePlacementReceipt,
    SciencePlacementReceiptV2,
    SciencePlacementReceiptV3,
    StartReceipt,
    decode_placement_receipt,
)
from tpen_executor.runners import CoreRunEvidence, RUN_SHIM
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.workers import ProcessOutcome, SubprocessWorker

GPU_A = "GPU-12345678-1234-1234-1234-123456789ABC"
GPU_B = "GPU-12345678-1234-1234-1234-123456789ABD"
GPU_C = "GPU-12345678-1234-1234-1234-123456789ABE"
FIXTURES = Path(__file__).parent / "fixtures"
POSIX_GROUPS = pytest.mark.skipif(not hasattr(os, "killpg"), reason="requires POSIX process groups")
LINUX_CHILD = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not hasattr(os, "sched_setaffinity"),
    reason="requires the Linux child affinity contract",
)

L1B_MUTANT_TABLE = (
    {
        "id": "K1", "target": "executor/src/tpen_executor/_child_gpu_sampler.py",
        "old": "            if not self._attempted:",
        "new": "            if not self._samples:",
        "killer": "test_sampler_launch_failure_is_attempted_telemetry_failed",
        "suite": "executor", "assertion": "a launch failure is attempted telemetry-failed, never not-sampled",
    },
    {
        "id": "K2", "target": "executor/src/tpen_executor/_child_gpu_sampler.py",
        "old": "        gpu_kill_group(pgid)",
        "new": "        pass",
        "killer": "test_sampler_success_path_kills_lingering_descendant",
        "suite": "executor", "assertion": "the success path kills the retained group before claiming quiescence",
    },
    {
        "id": "K3", "target": "executor/src/tpen_executor/_child_gpu_sampler.py",
        "old": "            gpu_kill_group(current[1])",
        "new": "            pass",
        "killer": "test_sampler_join_expiry_cancels_and_drains_inflight_command",
        "suite": "executor", "assertion": "join expiry kills and drains the in-flight command group",
    },
    {
        "id": "K4", "target": "executor/src/tpen_executor/gpu_telemetry.py",
        "old": '    for sample in sampler["samples"]:',
        "new": '    for sample in sampler["samples"][-1:]:',
        "killer": "test_out_of_slot_use_in_any_sample_dominates",
        "suite": "executor", "assertion": "out-of-entitlement use in an earlier sample is not masked by a later one",
    },
    {
        "id": "K5", "target": "executor/src/tpen_executor/gpu_telemetry.py",
        "old": '        if not sampler["quiescent"] or version is None or start is None or count < 1 or not command_pids:',
        "new": '        if version is None or start is None or count < 1 or not command_pids:',
        "killer": "test_decoder_rejects_contradictory_sampler_evidence[quiescent-false]",
        "suite": "executor", "assertion": "a qualified receipt requires confirmed quiescence",
    },
)


def test_l1b_mutant_table_targets_unique_executable_anchors() -> None:
    """Every mutant names a unique executable anchor and an existing killer."""

    root = Path(__file__).parents[2]
    module = Path(__file__).read_text(encoding="utf-8")
    assert [row["id"] for row in L1B_MUTANT_TABLE] == ["K1", "K2", "K3", "K4", "K5"]
    for row in L1B_MUTANT_TABLE:
        source = (root / row["target"]).read_text(encoding="utf-8")
        assert source.count(row["old"]) == 1, row
        assert row["old"] != row["new"] and row["suite"] == "executor" and row["assertion"]
        # Anchors are code, not docstrings or comments.
        assert not row["old"].strip().startswith(("#", '"""', "'"))
        assert f"def {row['killer'].split('[')[0]}(" in module, row


# ---------------------------------------------------------------- sampler


def _fake(tmp_path: Path, name: str, body: str) -> str:
    """Write one executable POSIX shell fake and return its path."""

    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


_OK_BODY = """\
case "$1" in
  --version) echo "NVIDIA-SMI fake" ;;
  --query-gpu=*) echo "GPU-12345678-1234-1234-1234-123456789ABC, 0000:00:01.0" ;;
  *) echo "$PPID, GPU-12345678-1234-1234-1234-123456789ABC" ;;
esac
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pids: list[int], bound: float = 2.0) -> list[bool]:
    """Return liveness after allowing init to reap killed orphans."""

    deadline = time.monotonic() + bound
    while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.02)
    return [_alive(pid) for pid in pids]


def _finished(sampler: child.GpuSampler) -> dict[str, object]:
    sampler.start()
    return sampler.finish()


@POSIX_GROUPS
def test_sampler_launch_failure_is_attempted_telemetry_failed(tmp_path: Path) -> None:
    """F1/K1: a missing executable is attempted telemetry failure with provenance."""

    record = _finished(child.GpuSampler(cadence=0.01, limit=2, executable=str(tmp_path / "missing-nvidia-smi")))
    assert record["attempted"] is True
    assert record["reason"] == "telemetry-failed"
    assert record["status"] == "unavailable"
    assert record["failure"]["code"] == "launch-failed"
    assert "FileNotFoundError" in record["failure"]["detail"]
    assert record["command_pids"] == []
    assert validate_sampler_wire(record)["reason"] == "telemetry-failed"


@POSIX_GROUPS
def test_sampler_one_session_mechanism_launches_commands(tmp_path: Path) -> None:
    """F1: launching with start_new_session alone yields observed, quiescent rows."""

    record = _finished(child.GpuSampler(cadence=0.01, limit=2, executable=_fake(tmp_path, "nvidia-smi", _OK_BODY)))
    assert record["reason"] == "observed", record["failure"]
    assert record["quiescent"] is True and record["survivors"] == []
    assert len(record["command_pids"]) == 3 * record["count"]
    assert record["samples"][0]["pid_rows"] == [{"pid": os.getpid(), "gpu_uuid": GPU_A}]
    assert validate_sampler_wire(record) == record


def test_unfinalized_sampler_snapshot_is_telemetry_failed() -> None:
    """F1: attempted is recorded before launch; an unfinished record is never not-sampled."""

    sampler = child.GpuSampler()
    assert sampler.record() == child.gpu_not_sampled_record() == not_sampled_sampler()
    sampler.begin()
    record = sampler.record()
    assert record["attempted"] is True
    assert record["reason"] == "telemetry-failed"
    assert record["failure"]["code"] == "sampler-unfinalized"
    assert record["quiescent"] is False
    assert validate_sampler_wire(record)["failure"]["code"] == "sampler-unfinalized"


@POSIX_GROUPS
def test_sampler_success_path_kills_lingering_descendant(tmp_path: Path) -> None:
    """F2/K2: a successful command's lingering descendant is killed and confirmed gone."""

    pids = tmp_path / "descendants"
    executable = _fake(tmp_path, "nvidia-smi", f"""\
        sleep 30 </dev/null >/dev/null 2>&1 &
        echo $! >> {pids}
        """ + _OK_BODY)
    record = _finished(child.GpuSampler(cadence=0.01, limit=1, executable=executable))
    descendants = [int(line) for line in pids.read_text(encoding="utf-8").split()]
    assert len(descendants) == 3
    assert _wait_dead(descendants) == [False, False, False]
    assert record["reason"] == "observed", record["failure"]
    assert record["quiescent"] is True and record["survivors"] == []


@POSIX_GROUPS
def test_sampler_timeout_kills_and_confirms_retained_group(tmp_path: Path) -> None:
    """F2: a timed-out command group is killed by its retained id and reaped."""

    executable = _fake(tmp_path, "nvidia-smi", "sleep 30\n")
    started = time.monotonic()
    record = _finished(child.GpuSampler(executable=executable, command_timeout=0.3, cleanup_timeout=0.5))
    assert time.monotonic() - started < 3.0
    assert record["failure"]["code"] == "command-timeout"
    assert record["quiescent"] is True and record["survivors"] == []
    assert _wait_dead(record["command_pids"]) == [False]


@POSIX_GROUPS
def test_sampler_read_error_path_cleans_retained_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F2: an OSError while reading still kills, reaps, and confirms the group."""

    executable = _fake(tmp_path, "nvidia-smi", "sleep 30\n")
    original = subprocess.Popen.communicate
    calls = []

    def failing_once(self, *args, **kwargs):
        calls.append(kwargs.get("timeout"))
        if len(calls) == 1:
            raise OSError("injected read failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(child.subprocess.Popen, "communicate", failing_once)
    record = _finished(child.GpuSampler(executable=executable, command_timeout=5.0, cleanup_timeout=1.0))
    assert record["failure"]["code"] == "read-failed"
    assert record["quiescent"] is True and record["survivors"] == []
    assert _wait_dead(record["command_pids"]) == [False]


@POSIX_GROUPS
def test_sampler_join_expiry_cancels_and_drains_inflight_command(tmp_path: Path) -> None:
    """F2/K3: join expiry kills the in-flight group through the cancellation handle."""

    executable = _fake(tmp_path, "nvidia-smi", "sleep 30\n")
    sampler = child.GpuSampler(executable=executable, command_timeout=3.0, cleanup_timeout=0.3, join_timeout=0.3)
    started = time.monotonic()
    record = _finished(sampler)
    assert time.monotonic() - started < 2.5
    assert record["failure"]["code"] == "sampler-cancelled"
    assert record["quiescent"] is True and record["survivors"] == []
    assert _wait_dead(record["command_pids"]) == [False]


@POSIX_GROUPS
def test_sampler_unconfirmed_group_is_survivor_and_not_quiescent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F2: quiescence requires ESRCH; an unconfirmed group is preserved as a survivor."""

    monkeypatch.setattr(child, "gpu_group_empty", lambda _pgid: False)
    record = _finished(child.GpuSampler(executable=_fake(tmp_path, "nvidia-smi", _OK_BODY), cleanup_timeout=0.1))
    assert record["failure"]["code"] == "cleanup-unconfirmed"
    assert record["quiescent"] is False
    assert record["survivors"] == record["command_pids"][:1]
    assert validate_sampler_wire(record)["quiescent"] is False


@POSIX_GROUPS
def test_sampler_keeps_samples_taken_before_a_late_failure(tmp_path: Path) -> None:
    """Historical N6 objective: a late failure preserves every earlier sample."""

    counter = tmp_path / "count"
    executable = _fake(tmp_path, "nvidia-smi", f"""\
        n=$(cat {counter} 2>/dev/null || echo 0); n=$((n + 1)); echo $n > {counter}
        if [ "$n" -gt 3 ]; then exit 1; fi
        """ + _OK_BODY)
    sampler = child.GpuSampler(cadence=0.01, limit=3, executable=executable)
    sampler.start()
    deadline = time.monotonic() + 10.0
    while sampler.record()["failure"]["code"] == "sampler-unfinalized" and time.monotonic() < deadline:
        time.sleep(0.02)
    record = sampler.finish()
    assert record["failure"]["code"] == "command-failed"
    assert record["count"] == 1 and len(record["samples"]) == 1
    assert validate_sampler_wire(record)["samples"] == record["samples"]


# ------------------------------------------------------- real RUN_SHIM child


def _shim_environment(tmp_path: Path, path_dir: Path, *, devices: str) -> PreparedPythonEnvironment:
    package = tmp_path / "pkg" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text("import time\ndef main(argv):\n    time.sleep(0.1)\n    return 0\n", encoding="utf-8")
    return PreparedPythonEnvironment(sys.executable, None, {
        "PYTHONPATH": str(package.parent), "PATH": str(path_dir),
        "TPEN_PLACEMENT_OUTPUT": str(tmp_path / "child-placement.json"), "TPEN_PLACEMENT_OUTPUT_DIR": str(tmp_path),
        "TPEN_PLACEMENT_DECLARED_CPUS": str(sorted(os.sched_getaffinity(0))[0]),
        "TPEN_PLACEMENT_DECLARED_HOST": socket.gethostname(), "TPEN_PLACEMENT_DECLARED_DEVICES": devices,
        "TPEN_PLACEMENT_RECEIPT_VERSION": "2",
        "TPEN_PLACEMENT_TASK_ID": "task-a", "TPEN_PLACEMENT_ATTEMPT_ID": "attempt-a",
        "TPEN_PLACEMENT_GROUP_DIGEST": digest("s2b2"), "TPEN_PLACEMENT_ALLOCATION_ID": "allocation-a",
        "TPEN_PLACEMENT_BACKEND_JOB_ID": "job-a", "TPEN_PLACEMENT_SLOT_ID": "logical-slot",
        "TPEN_PLACEMENT_PAYLOAD_DIGEST": digest("payload"),
        "TPEN_PLACEMENT_GPU_SAMPLER_COUNT": "2", "TPEN_PLACEMENT_GPU_SAMPLER_CADENCE": "0.02",
    })


def _run_shim(tmp_path: Path, path_dir: Path, *, devices: str = GPU_A) -> tuple[ProcessOutcome, dict[str, object]]:
    environment = _shim_environment(tmp_path, path_dir, devices=devices)
    outcome = SubprocessWorker().run(
        environment, ("-c", RUN_SHIM, f"run.root={tmp_path}"), cwd=tmp_path,
        stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
    )
    return outcome, json.loads((tmp_path / "child-placement.json").read_text(encoding="utf-8"))


@LINUX_CHILD
def test_real_shim_matched_sidecar_is_observed_and_quiescent(tmp_path: Path) -> None:
    """F1/F2: the real child samples its own PID and finalizes quiescent evidence."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "nvidia-smi", _OK_BODY)
    outcome, payload = _run_shim(tmp_path, bin_dir)
    sampler = payload["gpu_sampler"]
    assert outcome.returncode == 0
    assert payload["schema"] == "tpen-executor/child-placement-v2"
    assert sampler["reason"] == "observed", sampler["failure"]
    assert sampler["quiescent"] is True and sampler["survivors"] == []
    assert {row["pid"] for sample in sampler["samples"] for row in sample["pid_rows"]} == {outcome.pid}
    assert validate_sampler_wire(sampler) == sampler


@LINUX_CHILD
def test_real_shim_missing_nvidia_smi_is_telemetry_failed(tmp_path: Path) -> None:
    """F1: a declared-GPU child without nvidia-smi records attempted telemetry failure."""

    empty = tmp_path / "empty-bin"
    empty.mkdir()
    outcome, payload = _run_shim(tmp_path, empty)
    sampler = payload["gpu_sampler"]
    assert outcome.returncode == 0
    assert sampler["attempted"] is True
    assert sampler["reason"] == "telemetry-failed"
    assert sampler["failure"]["code"] == "launch-failed"


@LINUX_CHILD
def test_real_shim_cpu_only_slot_is_not_sampled(tmp_path: Path) -> None:
    """A CPU-only slot intentionally never samples and stays not-sampled."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "nvidia-smi", "exit 1\n")
    outcome, payload = _run_shim(tmp_path, bin_dir, devices="")
    assert outcome.returncode == 0
    assert payload["gpu_sampler"] == not_sampled_sampler()


# ------------------------------------------------------------ v3 receipts


def _group() -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    return ReadyGroup("s2b2", (TaskSpec("task-a", "task-a", program, digest("task-a"), 1, "steps"),), digest("s2b2"))


def _allocation(devices: tuple[str, ...]) -> ConfiguredAllocation:
    allowed = sorted(getattr(os, "sched_getaffinity", lambda _pid: (0,))(0))
    slot = AllocationSlot("logical-slot", "logical-label", devices, (allowed[0],), (), ()).to_dict()
    return ConfiguredAllocation(
        AllocationId("allocation-a"), BackendJobId("job-a"),
        {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot]}, "scheduler": {"PBS_NODEFILE": "physical-01\n"}},
    )


def _receipt(tmp_path: Path, sampler: object, devices: tuple[str, ...] = (GPU_A,)) -> SciencePlacementReceiptV3:
    group = _group()
    configured = ConfiguredRun("task-a", str(tmp_path / "config.yaml"), ("payload.seed=1",))
    allowed = sorted(getattr(os, "sched_getaffinity", lambda _pid: (0,))(0))
    fence = SimpleNamespace(attempt_id=AttemptId("attempt-a"), output_dir=str(tmp_path / "out"))
    environment = PreparedPythonEnvironment(sys.executable, None, {
        "TPEN_SLOT_ID": "logical-slot", "TPEN_SLOT_NODE": "physical-01", "TPEN_SLOT_CPUS": str(allowed[0]),
        "TPEN_PLACEMENT_DECLARED_DEVICES": ",".join(devices), "TPEN_PLACEMENT_RECEIPT_VERSION": "2",
        "TPEN_PLACEMENT_HOST_STATUS": "mapped",
    })
    base = ChildPlacementObservation(
        "task-a", "attempt-a", group.identity, "allocation-a", "job-a", "logical-slot",
        digest({"config": configured.config, "overrides": list(configured.overrides)}), fence.output_dir,
        "physical-01", 1234, 55, (allowed[0],), {}, None, "2026-11-10T00:00:00Z", None,
    )
    return _placement_receipt_v3(
        ChildPlacementObservationV2(base, sampler), _allocation(devices), fence, group, group.members[0],
        configured, environment, ProcessOutcome((), 0, "", "", False, 1234, 55),
    )


def _sampler(samples: list[list[tuple[int, str]]], *, inventory: tuple[str, ...] = (GPU_A, GPU_B), failure: str | None = None) -> dict[str, object]:
    """Build a valid child sampler record with one sample per row list."""

    rows_inventory = [{"uuid": uuid, "pci.bus_id": f"0000:00:0{index}.0"} for index, uuid in enumerate(inventory, 1)]
    records = [
        {"monotonic_start": 1.0 + 0.2 * index, "monotonic_end": 1.1 + 0.2 * index, "inventory": rows_inventory,
         "pid_rows": [{"pid": pid, "gpu_uuid": uuid} for pid, uuid in rows]}
        for index, rows in enumerate(samples)
    ]
    end = records[-1]["monotonic_end"] + 0.05 if records else 1.05
    return {
        "status": "unavailable" if failure else "observed", "reason": "telemetry-failed" if failure else "observed",
        "method": "nvidia-smi-pid-uuid-v1", "version": "NVIDIA-SMI fake" if records else None, "attempted": True,
        "failure": {"code": failure, "detail": "fake failure"} if failure else None, "cadence_seconds": 0.1,
        "count": len(records), "monotonic_start": 1.0, "monotonic_end": end, "samples": records,
        "command_pids": [101, 102, 103], "survivors": [], "elapsed_seconds": end - 1.0,
        "join_timeout_seconds": 4.0, "quiescent": True,
    }


def test_four_outcomes_are_typed_and_distinct(tmp_path: Path) -> None:
    """matched, mismatch, telemetry-failed, and not-sampled stay distinct and typed."""

    cases = {
        "qualified": (_sampler([[(1234, GPU_A)]]), (GPU_A,), "matched", "accepted"),
        "mismatch": (_sampler([[(1234, GPU_B)]]), (GPU_A,), "out-of-entitlement", "not-accepted"),
        "telemetry-failed": (_sampler([], failure="launch-failed"), (GPU_A,), "sampler-failed", "not-accepted"),
        "not-sampled": (not_sampled_sampler(), (), "cpu-only", "accepted"),
    }
    for reason_class, (sampler, devices, code, acceptance) in cases.items():
        case_dir = tmp_path / reason_class
        case_dir.mkdir()
        receipt = _receipt(case_dir, sampler, devices)
        assert receipt.schema == "tpen-executor/science-placement-receipt-v3"
        assert receipt.verdict["reason_class"] == reason_class
        assert receipt.gpu["reason"] == code
        assert receipt.verdict["science_acceptance"] == acceptance
        assert receipt.verdict["qualification"] == ("qualified" if reason_class == "qualified" else "unqualified")
        assert SciencePlacementReceiptV3.from_mapping(json.loads(json.dumps(receipt.to_dict()))).to_dict() == receipt.to_dict()


def test_out_of_slot_use_in_any_sample_dominates(tmp_path: Path) -> None:
    """F4/K4: the union of exact-PID rows over all samples decides; outside use wins."""

    cases = {
        "earlier-B-later-A": (_sampler([[(1234, GPU_B)], [(1234, GPU_A)]]), (GPU_A,)),
        "B-then-late-failure": (_sampler([[(1234, GPU_B)]], failure="command-failed"), (GPU_A,)),
        "B-with-unresolved-C": (_sampler([[(1234, GPU_B)]]), (GPU_A, GPU_C)),
    }
    for name, (sampler, devices) in cases.items():
        case_dir = tmp_path / name
        case_dir.mkdir()
        receipt = _receipt(case_dir, sampler, devices)
        assert receipt.verdict["reason_class"] == "mismatch", name
        assert receipt.verdict["device"] == "mismatch", name
        assert GPU_B in receipt.gpu["active_uuids"], name
        assert len(receipt.gpu_sampler["samples"]) == len(sampler["samples"]), name
        raw = receipt.to_dict()
        assert SciencePlacementReceiptV3.from_mapping(raw).verdict["reason_class"] == "mismatch", name
        # Relabelling the earlier out-of-slot evidence as qualified is rejected.
        raw["verdict"]["reason_class"] = "qualified"
        with pytest.raises((TypeError, ValueError)):
            SciencePlacementReceiptV3.from_mapping(raw)


def _set_all_rows(raw: dict, key: str, value: object) -> None:
    for sample in raw["gpu_sampler"]["samples"]:
        for row in sample["pid_rows"]:
            row[key] = value


CONTRADICTIONS = {
    "quiescent-false": lambda raw: raw["gpu_sampler"].__setitem__("quiescent", False),
    "survivors": lambda raw: raw["gpu_sampler"].update(survivors=[101], quiescent=False),
    "foreign-pid-rows": lambda raw: _set_all_rows(raw, "pid", 9999),
    "outside-uuid": lambda raw: _set_all_rows(raw, "gpu_uuid", GPU_B),
    "row-absent-from-inventory": lambda raw: raw["gpu_sampler"]["samples"][0]["pid_rows"].append({"pid": 1234, "gpu_uuid": GPU_C}),
    "unresolved-entitlement": lambda raw: raw["declared"].__setitem__("devices", [GPU_A, GPU_C]),
    "sample-outside-outer": lambda raw: raw["gpu_sampler"]["samples"][0].update(monotonic_start=100.0, monotonic_end=101.0),
    "reversed-sample": lambda raw: raw["gpu_sampler"]["samples"][0].update(monotonic_start=1.1, monotonic_end=1.0),
    "null-outer-interval": lambda raw: raw["gpu_sampler"].update(monotonic_start=None, monotonic_end=None),
    "null-version": lambda raw: raw["gpu_sampler"].__setitem__("version", None),
    "elapsed-disagrees": lambda raw: raw["gpu_sampler"].__setitem__("elapsed_seconds", 9.0),
    "qualified-over-failed-sampler": lambda raw: raw["gpu_sampler"].update(status="unavailable", reason="telemetry-failed", failure={"code": "command-failed", "detail": "x"}),
    "attempted-not-sampled": lambda raw: raw["gpu_sampler"].update(status="unavailable", reason="not-sampled"),
    "unknown-failure-code": lambda raw: raw["gpu_sampler"].update(status="unavailable", reason="telemetry-failed", failure={"code": "invented", "detail": "x"}),
    "extra-sampler-key": lambda raw: raw["gpu_sampler"].__setitem__("extra", 1),
    "relabelled-mismatch": lambda raw: raw["verdict"].__setitem__("reason_class", "mismatch"),
    "active-uuid-tamper": lambda raw: raw["gpu"].__setitem__("active_uuids", [GPU_B]),
    "warning-tamper": lambda raw: raw["verdict"].__setitem__("warning", "quiet"),
}


@pytest.mark.parametrize("name", sorted(CONTRADICTIONS))
def test_decoder_rejects_contradictory_sampler_evidence(tmp_path: Path, name: str) -> None:
    """F4/K5: every tampered qualified receipt is rejected by the shared derivation."""

    raw = _receipt(tmp_path, _sampler([[(1234, GPU_A)]])).to_dict()
    assert SciencePlacementReceiptV3.from_mapping(copy.deepcopy(raw)).verdict["reason_class"] == "qualified"
    CONTRADICTIONS[name](raw)
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV3.from_mapping(raw)


def test_observed_but_unusable_telemetry_is_typed_failure(tmp_path: Path) -> None:
    """F6: usable-telemetry failures are typed receipts that keep the raw samples."""

    cases = {
        "no-exact-pid-row": (_sampler([[]]), (GPU_A,)),
        "foreign-only": (_sampler([[(9999, GPU_A)]]), (GPU_A,)),
        "entitlement-unresolved": (_sampler([[(1234, GPU_A)]]), (GPU_A, GPU_C)),
    }
    for name, (sampler, devices) in cases.items():
        case_dir = tmp_path / name
        case_dir.mkdir()
        receipt = _receipt(case_dir, sampler, devices)
        code = "no-exact-pid-row" if name == "foreign-only" else name
        assert receipt.verdict["reason_class"] == "telemetry-failed", name
        assert receipt.verdict["science_acceptance"] == "not-accepted", name
        assert receipt.gpu["reason"] == code, name
        assert receipt.refusal is None, name
        assert receipt.to_dict()["gpu_sampler"] == sampler, name


def test_invalid_child_sampler_becomes_typed_failure(tmp_path: Path) -> None:
    """Structurally invalid child evidence is typed telemetry failure, not not-sampled."""

    receipt = _receipt(tmp_path, {"status": "observed"})
    assert receipt.verdict["reason_class"] == "telemetry-failed"
    assert receipt.gpu_sampler["failure"]["code"] == "sampler-evidence-invalid"
    assert receipt.gpu_sampler["attempted"] is True


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="requires Linux controller affinity contract")
def test_dispatch_records_typed_telemetry_failure_not_refusal(tmp_path: Path) -> None:
    """F6: a no-row sidecar crosses dispatch as telemetry-failed, rc 0, accepted segment."""

    group = _group()
    configured = ConfiguredRun("task-a", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("ok", encoding="utf-8")
    sampler = _sampler([[(9999, GPU_A)]])

    class Process:
        def run(self, environment, *args, **kwargs):
            values = environment.env
            base = ChildPlacementObservation(
                values["TPEN_PLACEMENT_TASK_ID"], values["TPEN_PLACEMENT_ATTEMPT_ID"], values["TPEN_PLACEMENT_GROUP_DIGEST"],
                values["TPEN_PLACEMENT_ALLOCATION_ID"], values["TPEN_PLACEMENT_BACKEND_JOB_ID"], values["TPEN_PLACEMENT_SLOT_ID"],
                digest({"config": configured.config, "overrides": []}), values["TPEN_PLACEMENT_OUTPUT_DIR"],
                "physical-01", 123, 55, (int(values["TPEN_PLACEMENT_DECLARED_CPUS"]),), {}, None, "2026-11-10T00:00:00Z", None,
            )
            Path(values["TPEN_PLACEMENT_OUTPUT"]).write_text(
                json.dumps(ChildPlacementObservationV2(base, sampler).to_dict()) + "\n", encoding="utf-8"
            )
            Path(values["TPEN_PLACEMENT_OUTPUT_DIR"], "core").mkdir(parents=True, exist_ok=True)
            Path(tmp_path / "checkpoint").mkdir(parents=True, exist_ok=True)
            return ProcessOutcome((sys.executable,), 0, "", "", False, 123, 55)

    class Evidence:
        def read(self, _process, _environment, run_dir, **_kwargs):
            return CoreRunEvidence(
                "tpen.run-evidence/v1", str(run_dir), "completed", "completed",
                {"checkpoint_dir": str(tmp_path / "checkpoint"), "schema_version": 1, "next_iteration": 1, "completed_updates": 1}, None,
            )

    worker = SimpleNamespace(materialize=lambda _env=None: {
        "environment": PreparedPythonEnvironment(sys.executable, None, {}),
        "process": Process(), "runner": object(), "evidence": Evidence(),
    })
    store = DirectoryStore(tmp_path / "store")
    manifest = LocalDispatcher(worker).run(group, {"task-a": configured}, AttachedAllocation(_allocation((GPU_A,)), store))
    attempt = manifest.attempts[0]
    assert attempt.outcome_receipt.status == "completed"
    assert _operational_exit(manifest) == 0
    assert manifest.segments
    assert isinstance(attempt.placement_receipt, SciencePlacementReceiptV3)
    assert attempt.placement_receipt.verdict["reason_class"] == "telemetry-failed"
    assert attempt.placement_receipt.refusal is None
    assert manifest.placement_verdict["message"] == PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE
    assert store.manifest(group).attempts[0].placement_receipt.to_dict() == attempt.placement_receipt.to_dict()


def test_untouched_l1a_v2_bytes_decode_and_v2_stays_strict() -> None:
    """F3: L1a v2 bytes (produced at f384f44d) still decode; v2 and v3 stay closed."""

    fixtures = json.loads((FIXTURES / "science-placement-receipts-l1a-v2.json").read_text(encoding="utf-8"))
    assert set(fixtures) == {"not-sampled", "mismatch"}
    for name, raw in fixtures.items():
        receipt = decode_placement_receipt(copy.deepcopy(raw))
        assert isinstance(receipt, SciencePlacementReceiptV2), name
        assert receipt.to_dict() == raw, name
        assert receipt.verdict["reason_class"] == name
        assert receipt.base.schema == SciencePlacementReceipt.schema
        with_sampler = dict(raw, gpu_sampler=not_sampled_sampler())
        with pytest.raises((TypeError, ValueError)):
            SciencePlacementReceiptV2.from_mapping(with_sampler)
        with pytest.raises((TypeError, ValueError)):
            SciencePlacementReceiptV3.from_mapping(dict(raw, schema=SciencePlacementReceiptV3.schema))
        with pytest.raises((TypeError, ValueError)):
            decode_placement_receipt(dict(raw, schema="tpen-executor/science-placement-receipt-v9"))


def _manifest_with(receipt: SciencePlacementReceiptV3) -> RunManifest:
    start = StartReceipt("task-a", receipt.attempt_id, AllocationId("allocation-a"), BackendJobId("job-a"), receipt.group_digest, "2026-11-10T00:00:00Z", receipt.output_dir)
    outcome = OutcomeReceipt("task-a", receipt.attempt_id, receipt.group_digest, "2026-11-10T00:00:01Z", "completed", receipt.output_dir)
    return RunManifest("s2b2", receipt.group_digest, (ManifestAttempt("task-a", receipt.attempt_id, start, outcome, receipt.output_dir, receipt),))


def test_manifest_message_is_derived_from_one_source(tmp_path: Path) -> None:
    """F5: every blocking reason serializes the shared message constant."""

    # The L1a mismatch/refusal message is pinned: L1a manifests must stay readable.
    assert PLACEMENT_UNQUALIFIED_MESSAGE == "GPU PLACEMENT UNQUALIFIED: placement mismatch/refusal requires science review."
    assert PLACEMENT_NOT_SAMPLED_MESSAGE.startswith("GPU PLACEMENT NOT-SAMPLED")
    cases = {
        "matched": (_sampler([[(1234, GPU_A)]]), (GPU_A,), None),
        "mismatch": (_sampler([[(1234, GPU_B)]]), (GPU_A,), PLACEMENT_UNQUALIFIED_MESSAGE),
        "sampler-failed": (_sampler([], failure="command-failed"), (GPU_A,), PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
        "no-exact-pid-row": (_sampler([[]]), (GPU_A,), PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
        "entitlement-unresolved": (_sampler([[(1234, GPU_A)]]), (GPU_A, GPU_C), PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
    }
    for name, (sampler, devices, message) in cases.items():
        case_dir = tmp_path / name
        case_dir.mkdir()
        manifest = _manifest_with(_receipt(case_dir, sampler, devices))
        decoded = RunManifest.decode(manifest.encode())
        if message is None:
            assert decoded.placement_verdict == {}, name
            assert json.loads(manifest.encode())["schema"] == "tpen-executor/run-manifest-v2"
        else:
            assert decoded.placement_verdict["message"] == message, name
            assert decoded.placement_verdict["status"] == "unqualified", name


def test_checked_in_v3_reader_fixture_is_executor_valid() -> None:
    """The P8 v3 fixture is exact executor producer output and still decodes."""

    path = Path(__file__).parents[2] / "experiments" / "toolkit" / "fixtures" / "run-manifests-v3-sampler.json"
    if not path.is_file():
        pytest.skip("named skip: repository experiments fixtures are not present")
    manifests = json.loads(path.read_text(encoding="utf-8"))
    assert set(manifests) == {"matched", "out-of-entitlement", "sampler-failed", "no-exact-pid-row", "entitlement-unresolved", "not-sampled"}
    for name, manifest in manifests.items():
        decoded = RunManifest.decode(json.dumps(manifest))
        assert decoded.to_dict() == manifest, name
        assert isinstance(decoded.attempts[0].placement_receipt, SciencePlacementReceiptV3), name
