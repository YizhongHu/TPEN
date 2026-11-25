"""Cannon-runnable tests for the external Polaris PBS wrapper."""

from __future__ import annotations

import ast
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA,
    HE_H2_EVAL_CHECKPOINTS,
    manifest_hash,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
WRAPPER = HERE.parent / "polaris" / "pbs_stage1.sh"
FIXTURES = HERE / "fixtures" / "polaris_p1_r2"
EAGLE_ROOT = "/eagle/HetRxnEnergy/rhu/stage1-p2-test"
SOURCE_SHA = "f" * 40


def _lscpu_fixture(*, physical_cores: int = 32, smt: bool = True) -> str:
    """Return lscpu's comma-separated CPU/core/node format for a fake node."""

    logical_cpus = physical_cores * 2 if smt else physical_cores
    lines = ["# comment lines from lscpu are ignored"]
    for cpu in range(logical_cpus):
        core = cpu % physical_cores if smt else cpu
        node = 0 if core < physical_cores // 2 else 1
        lines.append(f"{cpu},{core},{node}")
    return "\n".join(lines) + "\n"


def _stub_bin(tmp_path: Path, *, gpu_lines: str | None = None,
              mig_lines: str | None = None, device_count: str = "1",
              lscpu_text: str | None = None) -> dict[str, str]:
    """Create command seams replaying the captured P1 R2 observations."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    nvidia = tmp_path / "nvidia-smi"
    query = gpu_lines if gpu_lines is not None else (FIXTURES / "nvidia-smi-query.txt").read_text()
    mig = mig_lines if mig_lines is not None else (FIXTURES / "nvidia-smi-mig.txt").read_text()
    inventory = (FIXTURES / "nvidia-smi-L.txt").read_text()
    query_path = tmp_path / "query.txt"
    mig_path = tmp_path / "mig.txt"
    inventory_path = tmp_path / "inventory.txt"
    query_path.write_text(query, encoding="utf-8")
    mig_path.write_text(mig, encoding="utf-8")
    inventory_path.write_text(inventory, encoding="utf-8")
    nvidia.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        f"  *query-gpu=name,memory.total*) cat {shlex.quote(str(query_path))} ;;\n"
        f"  *query-gpu=mig.mode.current*) cat {shlex.quote(str(mig_path))} ;;\n"
        f"  *-L*) cat {shlex.quote(str(inventory_path))} ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    nvidia.chmod(0o750)
    python = tmp_path / "guard-python"
    binding_path = tmp_path / "torch-binding.txt"
    binding_text = (FIXTURES / "torch-binding.txt").read_text()
    if device_count != "1":
        binding_text = binding_text.replace("DEVICE_COUNT 1", f"DEVICE_COUNT {device_count}")
    binding_path.write_text(binding_text, encoding="utf-8")
    python.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        f"  *device_count*) awk -v gpu=\"${{CUDA_VISIBLE_DEVICES:-0}}\" "
        "'NR == gpu + 1 { for (i = 1; i <= NF; i++) "
        "if ($i == \"DEVICE_COUNT\") print $(i + 1) }' "
        f"{shlex.quote(str(binding_path))} ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    python.chmod(0o750)
    lscpu = tmp_path / "lscpu"
    lscpu_path = tmp_path / "lscpu.txt"
    lscpu_path.write_text(lscpu_text or _lscpu_fixture(), encoding="utf-8")
    lscpu.write_text(
        "#!/bin/sh\n"
        f"cat {shlex.quote(str(lscpu_path))}\n",
        encoding="utf-8",
    )
    lscpu.chmod(0o750)
    taskset = tmp_path / "taskset"
    taskset_record = tmp_path / "taskset-record.txt"
    taskset.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  -pc) printf \"pid %s's current affinity list: %s\\n\" \"$2\" \"${TASKSET_ALLOWED:-0-63}\" ;;\n"
        "  --cpu-list) printf '%s\\n' \"$2\" >> \"${TASKSET_RECORD:?}\"; shift 2; exec \"$@\" ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    taskset.chmod(0o750)
    return {
        "NVIDIA_SMI_BIN": str(nvidia), "GUARD_PYTHON_BIN": str(python),
        "POLARIS_LSCPU_BIN": str(lscpu), "TASKSET_BIN": str(taskset),
        "TASKSET_RECORD": str(taskset_record),
    }


def _env(tmp_path: Path, *, stubs: dict[str, str] | None = None,
         **overrides: str) -> dict[str, str]:
    """Build a dry, non-writing PBS context for one guard invocation."""

    env = os.environ.copy()
    env.update({
        "RESULTS_ROOT": EAGLE_ROOT,
        "SOURCE_SHA": SOURCE_SHA,
        "PBS_JOBID": "dry-7860235",
        "PBS_QUEUE": "debug",
        "EXPECTED_QUEUE": "debug",
        "APPROVAL_ID": "test-approval",
        "DRY_EXEC": "1",
        "PYBIN": sys.executable,
        "CHECKOUT_ROOT": str(REPO_ROOT),
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "BOOT_PARENT": str(tmp_path / "boot-parent"),
        "TASKSET_ALLOWED": "0-63",
        "TASKSET_RECORD": str(tmp_path / "taskset-record.txt"),
    })
    Path(env["UV_CACHE_DIR"]).mkdir()
    env.update(stubs if stubs is not None else _stub_bin(tmp_path))
    env.update(overrides)
    # The consumer now receives one frozen row object.  Keep older fixture
    # callers readable by materializing their first row into a test-owned file.
    manifest = env.pop("PLAN_MANIFEST", None)
    if manifest is not None:
        document = json.loads(Path(manifest).read_text(encoding="utf-8"))
        rows = document.get("rows", [])
        row = rows[0] if rows else {
            "kind": "train", "lineage": 0, "row_id": "test-row-0",
            "source_sha": env["SOURCE_SHA"], "results_root": env["RESULTS_ROOT"],
            "device": "cuda",
        }
        selected = Path(manifest).with_name("selected-row.json")
        selected.write_text(json.dumps(row) + "\n", encoding="utf-8")
        env["SELECTED_ROW"] = str(selected)
    env.pop("MODE", None)
    return env


def _run(mode: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run one wrapper mode and retain stdout/stderr for guard assertions."""

    return subprocess.run(
        ["bash", str(WRAPPER), mode],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_guard_replays_p1_r2_capture(tmp_path: Path) -> None:
    result = _run("guard", _env(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "POLARIS_GUARD PBS_JOBID=dry-7860235 PBS_QUEUE=debug" in result.stdout
    assert "NVIDIA A100-SXM4-40GB, 40960" in result.stdout
    assert "POLARIS_GUARD nvidia-smi-L=GPU 0: NVIDIA A100-SXM4-40GB" in result.stdout
    assert "POLARIS_GUARD mig=Disabled" in result.stdout
    assert "POLARIS_GUARD binding=0 device_count=1" in result.stdout
    assert "POLARIS_GUARD binding=3 device_count=1" in result.stdout


def test_guard_replays_real_polaris_lscpu_capture(tmp_path: Path) -> None:
    capture = (FIXTURES / "lscpu-p-CPU-CORE-NODE.txt").read_text(encoding="utf-8")
    result = _run("guard", _env(tmp_path, stubs=_stub_bin(tmp_path, lscpu_text=capture)))
    assert result.returncode == 0, result.stderr
    layout = [line for line in result.stdout.splitlines() if line.startswith("POLARIS_LAYOUT ")]
    assert len(layout) == 3
    expected = [set(range(0, 10)), set(range(10, 20)), set(range(20, 30))]
    for index, line in enumerate(layout):
        fields = dict(field.split("=", 1) for field in line.split()[1:])
        assert fields["lineage"] == str(index)
        assert fields["cores"] == "10"
        assert set(int(cpu) for cpu in fields["cpus"].split(",")) == expected[index]


def test_guard_derives_smt_deduplicated_disjoint_layout(tmp_path: Path) -> None:
    assert "assert not (core_sets[left] & core_sets[right])" in WRAPPER.read_text(encoding="utf-8")
    result = _run("guard", _env(tmp_path))
    assert result.returncode == 0, result.stderr
    layout = [line for line in result.stdout.splitlines() if line.startswith("POLARIS_LAYOUT ")]
    assert len(layout) == 3
    cpu_sets = []
    for index, line in enumerate(layout):
        fields = dict(field.split("=", 1) for field in line.split()[1:])
        assert fields["lineage"] == str(index)
        assert fields["cores"] == "10"
        cpus = tuple(int(cpu) for cpu in fields["cpus"].split(","))
        assert len(cpus) == len(set(cpus)) == 10
        assert all(cpu < 32 for cpu in cpus)
        cpu_sets.append(set(cpus))
    assert not (cpu_sets[0] & cpu_sets[1] or cpu_sets[0] & cpu_sets[2] or cpu_sets[1] & cpu_sets[2])


def test_guard_intersects_allowed_cpus_before_smt_deduplication(tmp_path: Path) -> None:
    stubs = _stub_bin(tmp_path)
    result = _run(
        "guard",
        _env(tmp_path, stubs=stubs, TASKSET_ALLOWED="0-7,32-39,48-55"),
    )
    assert result.returncode == 0, result.stderr
    layout = [line for line in result.stdout.splitlines() if line.startswith("POLARIS_LAYOUT ")]
    assert [line.split("cores=", 1)[1].split()[0] for line in layout] == ["5"] * 3
    cpus = {
        int(cpu)
        for line in layout
        for cpu in line.rsplit("cpus=", 1)[1].split(",")
    }
    assert cpus == set(range(8)) | set(range(48, 55))

    non_smt_stubs = _stub_bin(tmp_path / "non-smt", lscpu_text=_lscpu_fixture(smt=False))
    non_smt = _run(
        "guard",
        _env(tmp_path / "non-smt", stubs=non_smt_stubs, TASKSET_ALLOWED="0-31"),
    )
    assert non_smt.returncode == 0, non_smt.stderr
    non_smt_layout = [
        line for line in non_smt.stdout.splitlines() if line.startswith("POLARIS_LAYOUT ")
    ]
    assert all("cores=10" in line for line in non_smt_layout)
    assert all(len(line.rsplit("cpus=", 1)[1].split(",")) == 10 for line in non_smt_layout)


def test_guard_rejects_too_few_physical_cores_before_driver(tmp_path: Path) -> None:
    rows = [
        {
            "kind": "train",
            "lineage": lineage,
            "row_id": f"he-train-{lineage}",
            "source_sha": SOURCE_SHA,
            "results_root": EAGLE_ROOT,
            "device": "cuda",
        }
        for lineage in range(3)
    ]
    manifest = tmp_path / "phase1.json"
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    result = _run(
        "dry-exec",
        _env(tmp_path, TASKSET_ALLOWED="0-7", PLAN_MANIFEST=str(manifest)),
    )
    assert result.returncode == 2
    assert "fewer than 3 physical cores per lineage" in result.stderr


def test_guard_rejects_wrong_gpu_count(tmp_path: Path) -> None:
    stubs = _stub_bin(
        tmp_path,
        gpu_lines="NVIDIA A100-SXM4-40GB, 40960\n"
        "NVIDIA A100-SXM4-40GB, 40960\n",
    )
    result = _run(
        "guard",
        _env(tmp_path, stubs=stubs),
    )
    assert result.returncode == 2
    assert "expected 4 GPUs" in result.stderr


def test_guard_rejects_cannon_80gb_identity(tmp_path: Path) -> None:
    stubs = _stub_bin(
        tmp_path,
        gpu_lines="\n".join(["NVIDIA A100-SXM4-80GB, 81920"] * 4) + "\n",
    )
    result = _run("guard", _env(tmp_path, stubs=stubs))
    assert result.returncode == 2
    assert "unexpected GPU identity" in result.stderr


def test_guard_rejects_wrong_queue(tmp_path: Path) -> None:
    result = _run("guard", _env(tmp_path, PBS_QUEUE="capacity"))
    assert result.returncode == 2
    assert "delivered queue=capacity expected=debug" in result.stderr


def test_guard_rejects_mig_enabled(tmp_path: Path) -> None:
    stubs = _stub_bin(tmp_path, mig_lines="Enabled\nEnabled\nEnabled\nEnabled\n")
    result = _run("guard", _env(tmp_path, stubs=stubs))
    assert result.returncode == 2
    assert "MIG mode is not Disabled" in result.stderr


def test_guard_rejects_multiple_visible_devices(tmp_path: Path) -> None:
    stubs = _stub_bin(tmp_path, device_count="0")
    result = _run("guard", _env(tmp_path, stubs=stubs))
    assert result.returncode == 2
    assert "expected one visible GPU" in result.stderr


def test_guard_rejects_non_eagle_results_root(tmp_path: Path) -> None:
    result = _run("guard", _env(tmp_path, RESULTS_ROOT=str(tmp_path / "outside")))
    assert result.returncode == 2
    assert "outside /eagle/HetRxnEnergy/rhu" in result.stderr
    assert not (tmp_path / "outside").exists()


def test_guard_rejects_symlinked_results_root_before_any_write(tmp_path: Path) -> None:
    prefix = tmp_path / "test-prefix"
    prefix.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"preserve-me\0")
    results_root = prefix / "results"
    results_root.symlink_to(outside, target_is_directory=True)
    before_outside = sorted(path.relative_to(outside) for path in outside.rglob("*"))

    result = _run(
        "guard",
        _env(
            tmp_path,
            RESULTS_ROOT=str(results_root),
            TPEN_POLARIS_TEST_RESULTS_PREFIX=str(prefix),
            PBS_ENVIRONMENT="",
            PBS_JOBID="",
        ),
    )

    assert result.returncode == 2
    assert "RESULTS_ROOT is outside" in result.stderr
    assert sentinel.read_bytes() == b"preserve-me\0"
    assert sorted(path.relative_to(outside) for path in outside.rglob("*")) == before_outside


def test_dry_exec_runs_one_selected_train_control_row(tmp_path: Path) -> None:
    rows = [{
        "kind": "train", "lineage": 1, "row_id": "he-train-1",
        "source_sha": SOURCE_SHA, "results_root": EAGLE_ROOT, "device": "cuda",
    }]
    manifest = tmp_path / "phase1.json"
    manifest.write_text(json.dumps({"rows": rows}) + "\n", encoding="utf-8")
    env = _env(
        tmp_path,
        PLAN_MANIFEST=str(manifest),
        PYTHON_BIN=sys.executable,
        CHECKOUT_ROOT=str(REPO_ROOT),
        BOOT_ROOT=str(tmp_path / "boot"),
    )
    result = _run("dry-exec", env)
    assert result.returncode == 0, result.stderr
    assert "DRY_EXEC bootstrap=SKIPPED" in result.stdout
    assert result.stdout.count("DRY_EXEC compute=SKIPPED row=he-train-1") == 1
    assert "DRY_EXEC complete=PASS" in result.stdout


def test_selected_row_rejects_manifest_before_driver(tmp_path: Path) -> None:
    row = {
        "kind": "train", "lineage": 0, "row_id": "he-train-0",
        "source_sha": SOURCE_SHA, "results_root": EAGLE_ROOT, "device": "cuda",
    }
    selected = tmp_path / "selected-row.json"
    selected.write_text(json.dumps({"rows": [row]}) + "\n", encoding="utf-8")
    result = _run("dry-exec", _env(tmp_path, SELECTED_ROW=str(selected)))
    assert result.returncode == 2
    assert "one frozen row object, not a manifest" in result.stderr


def test_missing_approval_refuses_before_driver_effect(tmp_path: Path) -> None:
    row = {
        "kind": "train", "lineage": 0, "row_id": "he-train-0",
        "source_sha": SOURCE_SHA, "results_root": EAGLE_ROOT, "device": "cuda",
    }
    selected = tmp_path / "selected-row.json"
    selected.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = _run(
        "dry-exec",
        _env(tmp_path, SELECTED_ROW=str(selected), APPROVAL_ID=""),
    )
    assert result.returncode == 2
    assert "APPROVAL_ID is required" in result.stderr
    assert "DRY_EXEC compute=SKIPPED" not in result.stdout


def test_dry_exec_rejects_existing_boot_root_without_touching_it(tmp_path: Path) -> None:
    boot_root = tmp_path / "existing-boot"
    boot_root.mkdir()
    sentinel = boot_root / "sentinel"
    sentinel.write_bytes(b"preserve-me\0")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": []}), encoding="utf-8")
    result = _run("dry-exec", _env(tmp_path, BOOT_ROOT=str(boot_root), PLAN_MANIFEST=str(manifest)))
    assert result.returncode == 2
    assert "BOOT_ROOT already exists" in result.stderr
    assert sentinel.read_bytes() == b"preserve-me\0"


def test_dry_exec_rejects_missing_external_cache(tmp_path: Path) -> None:
    missing = tmp_path / "missing-cache"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": []}), encoding="utf-8")
    env = _env(tmp_path, UV_CACHE_DIR=str(missing), PLAN_MANIFEST=str(manifest))
    result = _run("dry-exec", env)
    assert result.returncode == 2
    assert "UV_CACHE_DIR is missing" in result.stderr
    assert not missing.exists()


def test_dry_exec_rejects_cache_named_inside_caller_boot_root(tmp_path: Path) -> None:
    boot_root = tmp_path / "existing-boot"
    boot_root.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": []}), encoding="utf-8")
    env = _env(
        tmp_path, BOOT_ROOT=str(boot_root), UV_CACHE_DIR=str(boot_root / "cache"),
        PLAN_MANIFEST=str(manifest),
    )
    result = _run("dry-exec", env)
    assert result.returncode == 2
    assert "BOOT_ROOT already exists" in result.stderr
    assert not (boot_root / "cache").exists()


def test_guard_rejects_dot_dot_escape_before_any_write(tmp_path: Path) -> None:
    result = _run("guard", _env(tmp_path, RESULTS_ROOT=EAGLE_ROOT + "/../outside"))
    assert result.returncode == 2
    assert "RESULTS_ROOT is outside" in result.stderr


def test_test_prefix_override_refused_in_pbs_batch(tmp_path: Path) -> None:
    prefix = tmp_path / "test-prefix"
    prefix.mkdir()
    sentinel = prefix / "sentinel"
    sentinel.write_bytes(b"preserve-me\0")
    env = _env(
        tmp_path,
        RESULTS_ROOT=str(prefix / "results"),
        TPEN_POLARIS_TEST_RESULTS_PREFIX=str(prefix),
        PBS_ENVIRONMENT="PBS_BATCH",
        PBS_JOBID="",
    )
    result = _run("guard", env)
    assert result.returncode == 2
    assert "TPEN_POLARIS_TEST_RESULTS_PREFIX is forbidden" in result.stderr
    assert sentinel.read_bytes() == b"preserve-me\0"


def test_test_prefix_override_refused_in_pbs_interactive(tmp_path: Path) -> None:
    prefix = tmp_path / "test-prefix"
    prefix.mkdir()
    sentinel = prefix / "sentinel"
    sentinel.write_bytes(b"preserve-me\0")
    env = _env(
        tmp_path,
        RESULTS_ROOT=str(prefix / "results"),
        TPEN_POLARIS_TEST_RESULTS_PREFIX=str(prefix),
        PBS_ENVIRONMENT="PBS_INTERACTIVE",
        PBS_JOBID="",
    )
    result = _run("guard", env)
    assert result.returncode == 2
    assert "TPEN_POLARIS_TEST_RESULTS_PREFIX is forbidden" in result.stderr
    assert sentinel.read_bytes() == b"preserve-me\0"


def test_test_prefix_override_refused_with_pbs_jobid(tmp_path: Path) -> None:
    prefix = tmp_path / "test-prefix"
    prefix.mkdir()
    sentinel = prefix / "sentinel"
    sentinel.write_bytes(b"preserve-me\0")
    env = _env(
        tmp_path,
        RESULTS_ROOT=str(prefix / "results"),
        TPEN_POLARIS_TEST_RESULTS_PREFIX=str(prefix),
        PBS_ENVIRONMENT="",
        PBS_JOBID="pbs-123.server",
    )
    result = _run("guard", env)
    assert result.returncode == 2
    assert "TPEN_POLARIS_TEST_RESULTS_PREFIX is forbidden" in result.stderr
    assert sentinel.read_bytes() == b"preserve-me\0"


def test_plan_rejects_relative_output_before_any_checkout_write(tmp_path: Path) -> None:
    prefix = tmp_path / "test-prefix"
    prefix.mkdir()
    relative_output = "relative-plan-r2-2.json"
    env = _env(
        tmp_path,
        RESULTS_ROOT=str(prefix / "results"),
        TPEN_POLARIS_TEST_RESULTS_PREFIX=str(prefix),
        PBS_ENVIRONMENT="",
        PBS_JOBID="",
        PLAN_OUTPUT=relative_output,
        PLAN_PHASE="1",
        SYSTEM="he",
        CHECKPOINTS="50000",
        LINEAGE_SEEDS="1:2:3",
    )
    env["MODE"] = "plan"
    result = _run("plan", env)
    assert result.returncode == 2
    assert "PLAN_OUTPUT must be absolute" in result.stderr
    assert not (REPO_ROOT / relative_output).exists()


def test_job_receipt_rejects_symlinked_logs_before_write(tmp_path: Path) -> None:
    prefix = tmp_path / "test-prefix"
    results_root = prefix / "results"
    results_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"preserve-me\0")
    (results_root / "logs").symlink_to(outside, target_is_directory=True)
    plan_dir = results_root / "plan"
    plan_dir.mkdir()
    env = _env(
        tmp_path,
        RESULTS_ROOT=str(results_root),
        TPEN_POLARIS_TEST_RESULTS_PREFIX=str(prefix),
        PBS_ENVIRONMENT="",
        PBS_JOBID="",
        PLAN_OUTPUT=str(plan_dir / "phase1.json"),
        PLAN_PHASE="1",
        SYSTEM="he",
        CHECKPOINTS="50000",
        LINEAGE_SEEDS="1:2:3",
    )
    env["MODE"] = "plan"
    result = _run("plan", env)
    assert result.returncode == 2
    assert "RESULTS_LOGS is outside" in result.stderr
    assert sentinel.read_bytes() == b"preserve-me\0"


def test_collect_skips_gpu_layout_guard_with_four_cpu_mask(tmp_path: Path) -> None:
    prefix = tmp_path / "test-prefix"
    results_root = prefix / "results"
    manifest_path = results_root / "plan" / "phase2.json"
    output_path = results_root / "collection.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "schema": "stage1/v3", "rows": [], "manifest_hash": manifest_hash([]),
    }) + "\n", encoding="utf-8")
    env = _env(
        tmp_path,
        RESULTS_ROOT=str(results_root),
        TPEN_POLARIS_TEST_RESULTS_PREFIX=str(prefix),
        PBS_ENVIRONMENT="",
        PBS_JOBID="",
        COLLECT_MANIFEST=str(manifest_path),
        COLLECT_OUTPUT=str(output_path),
        TASKSET_ALLOWED="0-3",
    )
    env["MODE"] = "collect"
    result = _run("collect", env)
    assert result.returncode == 0, result.stderr
    assert output_path.is_file()
    assert "POLARIS_LAYOUT" not in result.stdout


def test_plan_reports_small_cpu_layout_without_gpu_guard(tmp_path: Path) -> None:
    prefix = tmp_path / "test-prefix"
    results_root = prefix / "results"
    plan_dir = results_root / "plan"
    plan_dir.mkdir(parents=True)
    output_path = plan_dir / "phase1.json"
    env = _env(
        tmp_path,
        RESULTS_ROOT=str(results_root),
        TPEN_POLARIS_TEST_RESULTS_PREFIX=str(prefix),
        PBS_ENVIRONMENT="",
        PBS_JOBID="",
        PLAN_OUTPUT=str(output_path),
        PLAN_PHASE="1",
        SYSTEM="he",
        LINEAGE_SEEDS="1:2:3",
        TASKSET_ALLOWED="0-3",
    )
    env["MODE"] = "plan"
    result = _run("plan", env)
    assert result.returncode == 0, result.stderr
    assert "POLARIS_LAYOUT report-only=1 cores=4 cpus=0,1,2,3" in result.stdout
    assert output_path.is_file()


def test_plan_rejects_noncanonical_phase1_checkpoints(tmp_path: Path) -> None:
    prefix = tmp_path / "test-prefix"
    results_root = prefix / "results"
    plan_dir = results_root / "plan"
    plan_dir.mkdir(parents=True)
    env = _env(
        tmp_path,
        RESULTS_ROOT=str(results_root),
        TPEN_POLARIS_TEST_RESULTS_PREFIX=str(prefix),
        PBS_ENVIRONMENT="",
        PBS_JOBID="",
        PLAN_OUTPUT=str(plan_dir / "phase1.json"),
        PLAN_PHASE="1",
        SYSTEM="he",
        CHECKPOINTS="50000",
        LINEAGE_SEEDS="1:2:3",
        TASKSET_ALLOWED="0-3",
    )
    env["MODE"] = "plan"
    result = _run("plan", env)
    assert result.returncode == 1
    assert "train checkpoints must equal the declared retention cadence" in result.stderr


def test_mutant_python_replacements_parse_and_killers_are_collected() -> None:
    mutants = json.loads((HERE / "polaris-mutants.json").read_text(encoding="utf-8"))
    python_mutants = {
        "polaris-resolved-root-check-dropped",
        "polaris-cuda-row-admission-dropped",
        "polaris-smt-dedup-dropped",
        "polaris-layout-disjoint-assert-dropped",
    }
    source_cache: dict[Path, set[str]] = {}
    for mutant in mutants:
        if mutant["id"] in python_mutants:
            ast.parse(textwrap.dedent(mutant["new"]))
        path_text, node_text = mutant["killer"].split("::", 1)
        path = REPO_ROOT / path_text
        assert path.is_file(), mutant["killer"]
        if path not in source_cache:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            source_cache[path] = {
                node.name for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
        node_name = node_text.split("[", 1)[0]
        assert node_name in source_cache[path], mutant["killer"]


def test_production_rejects_promisor_checkout_before_uv(tmp_path: Path) -> None:
    checkout, source_sha = _make_exact_fixture(tmp_path)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "remote.origin.promisor", "true"],
        check=True,
    )
    env = _production_env(tmp_path, checkout, source_sha)
    selected = tmp_path / "selected-row.json"
    selected.write_text(json.dumps({
        "kind": "train", "lineage": 0, "row_id": "test-row-0",
        "source_sha": source_sha, "results_root": env["RESULTS_ROOT"],
        "device": "cuda",
    }), encoding="utf-8")
    env.update({"MODE": "train", "SELECTED_ROW": str(selected)})
    result = _run("train", env)
    assert result.returncode == 2
    assert "promisor remote" in result.stderr
    assert not Path(env["UV_RECORD"]).exists()


def _make_exact_fixture(tmp_path: Path) -> tuple[Path, str]:
    """Create a complete local exact-source fixture with a real Git head."""

    checkout = tmp_path / "checkout"
    shutil.copytree(REPO_ROOT / "experiments", checkout / "experiments")
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        shutil.copy2(REPO_ROOT / name, checkout / name)
    driver = (
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "record = os.environ['DRIVER_RECORD']\n"
        "with open(record, 'a', encoding='utf-8') as handle:\n"
        "    handle.write(f\"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}\\n\")\n"
        "if os.environ.get('FAIL_DRIVER') == 'all' or (os.environ.get('FAIL_DRIVER') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES') == '1'):\n"
        "    raise SystemExit(9)\n"
    )
    for name in ("train.py", "eval.py"):
        path = checkout / "experiments" / "atomistic" / "stage1" / name
        path.write_text(driver, encoding="utf-8")
        path.chmod(0o750)
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-m", "fixture"], check=True,
                   capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    return checkout, sha


def _production_stubs(tmp_path: Path) -> dict[str, str]:
    """Stub only uv, nvidia-smi, the driver, and network Git subcommands."""

    real_python = sys.executable
    uv_record = tmp_path / "uv-record.txt"
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/bin/sh\n"
            "printf '%%s|%%s|%%s|%%s\\n' \"$UV_PYTHON\" \"$UV_NO_MANAGED_PYTHON\" "
        "\"$UV_PYTHON_DOWNLOADS\" \"$UV_LINK_MODE\" >> \"$UV_RECORD\"\n"
        "case \"$1\" in\n"
        "  lock) exit 0 ;;\n"
        "  --version) echo 'uv 0.8.23'; exit 0 ;;\n"
        "  sync) mkdir -p \"$UV_PROJECT_ENVIRONMENT/bin\"; "
        "printf '#!/bin/sh\\ncase \"$*\" in *device_count*) echo 1 ;; *) exec %s \"$@\" ;; esac\\n' "
        "\"$REAL_PYTHON\" > \"$UV_PROJECT_ENVIRONMENT/bin/python\"; "
        "chmod 750 \"$UV_PROJECT_ENVIRONMENT/bin/python\"; exit 0 ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n" % real_python,
        encoding="utf-8",
    )
    uv.chmod(0o750)
    driver_record = tmp_path / "driver-record.txt"
    driver = tmp_path / "driver.py"
    driver.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS\" >> \"$DRIVER_RECORD\"\n"
        "case \"${FAIL_DRIVER:-0}:$CUDA_VISIBLE_DEVICES\" in 1:1) exit 9 ;; esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    driver.chmod(0o750)
    git_shim = tmp_path / "git"
    git_shim.write_text(
        "#!/bin/sh\n"
        "case \" $* \" in *' clone '*|*' fetch '*|*' pull '*|*' ls-remote '*) "
        "echo 'network Git forbidden' >&2; exit 91 ;; esac\n"
        "exec /usr/bin/git \"$@\"\n",
        encoding="utf-8",
    )
    git_shim.chmod(0o750)
    pybin = tmp_path / "pybin"
    pybin.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -m ] && [ \"$2\" = uv ]; then shift 2; exec \"$UV_STUB\" \"$@\"; fi\n"
        "exec \"$REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    pybin.chmod(0o750)
    return {
        "PYBIN": str(pybin), "UV_STUB": str(uv), "UV_RECORD": str(uv_record),
        "REAL_PYTHON": real_python, "DRIVER_RECORD": str(driver_record),
        "DRIVER": str(driver), "GIT_SHIM": str(git_shim),
    }


def _production_env(tmp_path: Path, checkout: Path, source_sha: str) -> dict[str, str]:
    stubs = _production_stubs(tmp_path)
    results_prefix = tmp_path / "test-results-prefix"
    env = os.environ.copy()
    env.update({
        "SOURCE_SHA": source_sha, "CHECKOUT_ROOT": str(checkout),
        "RESULTS_ROOT": str(results_prefix / f"t1-{tmp_path.name}"),
        "TPEN_POLARIS_TEST_RESULTS_PREFIX": str(results_prefix),
        "PBS_JOBID": "", "PBS_ENVIRONMENT": "", "PBS_QUEUE": "debug",
        "EXPECTED_QUEUE": "debug", "APPROVAL_ID": "test-approval", "DRY_EXEC": "0",
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "BOOT_PARENT": str(tmp_path / "boot-parent"),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    })
    Path(env["UV_CACHE_DIR"]).mkdir()
    env.update(_stub_bin(tmp_path))
    env.update(stubs)
    env.pop("GUARD_PYTHON_BIN", None)
    # The shim is named `git` and `pybin` invokes the real interpreter, while
    # only the requested external boundaries are stubbed.
    env["PATH"] = f"{tmp_path}:{os.environ['PATH']}"
    return env


def test_production_plan_train_eval_use_real_rows_and_uv_controls(tmp_path: Path) -> None:
    checkout, source_sha = _make_exact_fixture(tmp_path)
    env = _production_env(tmp_path, checkout, source_sha)
    results_root = Path(env["RESULTS_ROOT"])
    results_root.mkdir(parents=True, exist_ok=True)
    plan_path = results_root / "plan" / "phase1.json"
    env.update({
        "MODE": "plan", "PLAN_PHASE": "1", "SYSTEM": "he",
        "PLAN_OUTPUT": str(plan_path), "LINEAGE_SEEDS": "1:2:3",
    })
    planned = _run("plan", env)
    assert planned.returncode == 0, planned.stderr
    assert f"POLARIS_TEST_PREFIX_OVERRIDE={env['TPEN_POLARIS_TEST_RESULTS_PREFIX']}" in planned.stderr
    receipt = next(results_root.glob("logs/polaris-plan-*.env"))
    assert f"POLARIS_TEST_PREFIX_OVERRIDE={env['TPEN_POLARIS_TEST_RESULTS_PREFIX']}" in receipt.read_text()
    phase1 = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [row["device"] for row in phase1["rows"] if row["kind"] == "train"] == ["cuda"] * 3

    selected_train = results_root / "plan" / "selected-train.json"
    train_row = next(row for row in phase1["rows"] if row["kind"] == "train")
    selected_train.write_text(json.dumps(train_row), encoding="utf-8")
    env.update({"MODE": "train", "SELECTED_ROW": str(selected_train), "DRIVER": env["DRIVER"]})
    trained = _run("train", env)
    assert trained.returncode == 0, trained.stderr
    uv_controls = Path(env["UV_RECORD"]).read_text(encoding="utf-8").splitlines()
    assert uv_controls
    assert all(line.endswith("|1|never|copy") for line in uv_controls)
    records = Path(env["DRIVER_RECORD"]).read_text(encoding="utf-8").splitlines()
    statuses = list(results_root.glob("logs/*.status"))
    expected_core_counts = {}
    for path in statuses:
        fields = dict(field.split("=", 1) for field in path.read_text().split())
        expected_core_counts[fields["gpu"]] = len(fields["cpu"].split(","))
    assert sorted(records) == sorted(
        f"CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS={core_count} "
        f"MKL_NUM_THREADS={core_count}"
        for gpu, core_count in expected_core_counts.items()
    )

    invalid_manifest = results_root / "plan" / "cpu-row.json"
    invalid_doc = dict(train_row)
    invalid_doc["device"] = "cpu"
    invalid_manifest.write_text(json.dumps(invalid_doc), encoding="utf-8")
    before_invalid = Path(env["DRIVER_RECORD"]).read_bytes()
    env.update({"SELECTED_ROW": str(invalid_manifest)})
    invalid = _run("train", env)
    assert invalid.returncode == 2
    assert "must have device=cuda" in invalid.stderr
    assert Path(env["DRIVER_RECORD"]).read_bytes() == before_invalid
    invalid_doc.pop("device")
    invalid_manifest.write_text(json.dumps(invalid_doc), encoding="utf-8")
    invalid_missing = _run("train", env)
    assert invalid_missing.returncode == 2
    assert "must have device=cuda" in invalid_missing.stderr
    assert Path(env["DRIVER_RECORD"]).read_bytes() == before_invalid
    env["SELECTED_ROW"] = str(selected_train)

    env["FAIL_DRIVER"] = "all"
    failing = _run("train", env)
    assert failing.returncode != 0
    env.pop("FAIL_DRIVER")

    # Use the authentic phase-1 rows and protocol binding to build the eval
    # manifest; no hand-written command rows are used.
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1["manifest_hash"],
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in phase1["rows"] if row["kind"] == "train"
        },
        "checkpoints": {
            f"{row['system']}:{row['lineage']}:{checkpoint}": "0" * 64
            for row in phase1["rows"] if row["kind"] == "train"
            for checkpoint in HE_H2_EVAL_CHECKPOINTS
        },
    }
    evidence_path = results_root / "plan" / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    phase2_path = results_root / "plan" / "phase2.json"
    subprocess.run([
        sys.executable, "-m", "experiments.atomistic.stage1.plan", "--output", str(phase2_path),
        "--source-sha", source_sha, "--results-root", str(results_root), "--phase", "2",
        "--phase1-manifest", str(plan_path), "--checkpoint-bindings", str(evidence_path),
        "--checkpoints", *[str(checkpoint) for checkpoint in HE_H2_EVAL_CHECKPOINTS],
        "--lineage-seeds", "1", "2", "3",
    ], cwd=checkout, check=True)
    selected_eval = results_root / "plan" / "selected-eval.json"
    eval_row = next(row for row in json.loads(phase2_path.read_text())["rows"] if row["kind"] == "eval")
    selected_eval.write_text(json.dumps(eval_row), encoding="utf-8")
    env.update({"MODE": "eval", "SELECTED_ROW": str(selected_eval)})
    evaluated = _run("eval", env)
    assert evaluated.returncode == 0, evaluated.stderr
    env["FAIL_DRIVER"] = "all"
    evaluated_fail = _run("eval", env)
    assert evaluated_fail.returncode != 0
    assert len(list(results_root.glob("logs/*.status"))) >= 3
    assert set(Path(env["TASKSET_RECORD"]).read_text().splitlines()) == {
        "0,1,2,3,4,5,6,7,8,9",
    }
