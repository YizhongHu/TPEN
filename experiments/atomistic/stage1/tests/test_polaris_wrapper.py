"""Cannon-runnable tests for the external Polaris PBS wrapper."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
WRAPPER = HERE.parent / "polaris" / "pbs_stage1.sh"
FIXTURES = HERE / "fixtures" / "polaris_p1_r2"
EAGLE_ROOT = "/eagle/HetRxnEnergy/rhu/stage1-p2-test"
SOURCE_SHA = "f" * 40


def _stub_bin(tmp_path: Path, *, gpu_lines: str | None = None,
              mig_lines: str | None = None, device_count: str = "1") -> dict[str, str]:
    """Create command seams replaying the captured P1 R2 observations."""

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
    return {"NVIDIA_SMI_BIN": str(nvidia), "GUARD_PYTHON_BIN": str(python)}


def _env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    """Build a dry, non-writing PBS context for one guard invocation."""

    env = os.environ.copy()
    env.update({
        "RESULTS_ROOT": EAGLE_ROOT,
        "SOURCE_SHA": SOURCE_SHA,
        "PBS_JOBID": "dry-7860235",
        "PBS_QUEUE": "debug",
        "EXPECTED_QUEUE": "debug",
        "DRY_EXEC": "1",
    })
    env.update(_stub_bin(tmp_path))
    env.update(overrides)
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


def test_guard_rejects_wrong_gpu_count(tmp_path: Path) -> None:
    result = _run(
        "guard",
        _env(tmp_path, NVIDIA_SMI_BIN=_stub_bin(tmp_path, gpu_lines="NVIDIA A100-SXM4-40GB, 40960\n"
                                                   "NVIDIA A100-SXM4-40GB, 40960\n")["NVIDIA_SMI_BIN"]),
    )
    assert result.returncode == 2
    assert "expected 4 GPUs" in result.stderr


def test_guard_rejects_mig_enabled(tmp_path: Path) -> None:
    stubs = _stub_bin(tmp_path, mig_lines="Enabled\nEnabled\nEnabled\nEnabled\n")
    result = _run("guard", _env(tmp_path, **stubs))
    assert result.returncode == 2
    assert "MIG mode is not Disabled" in result.stderr


def test_guard_rejects_multiple_visible_devices(tmp_path: Path) -> None:
    stubs = _stub_bin(tmp_path, device_count="0")
    result = _run("guard", _env(tmp_path, **stubs))
    assert result.returncode == 2
    assert "expected one visible GPU" in result.stderr


def test_guard_rejects_non_eagle_results_root(tmp_path: Path) -> None:
    result = _run("guard", _env(tmp_path, RESULTS_ROOT=str(tmp_path / "outside")))
    assert result.returncode == 2
    assert "outside Eagle prefix" in result.stderr


def test_dry_exec_runs_three_train_control_rows(tmp_path: Path) -> None:
    rows = [
        {
            "kind": "train",
            "lineage": lineage,
            "row_id": f"he-train-{lineage}",
            "source_sha": SOURCE_SHA,
            "results_root": EAGLE_ROOT,
        }
        for lineage in range(3)
    ]
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
    assert result.stdout.count("DRY_EXEC compute=SKIPPED row=he-train-") == 3
    assert "DRY_EXEC complete=PASS" in result.stdout
