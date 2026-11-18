"""Cannon-runnable tests for the external Polaris PBS wrapper."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from experiments.atomistic.stage1.protocol import CHECKPOINT_EVIDENCE_SCHEMA, manifest_hash


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
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
        "DRY_EXEC": "1",
        "PYBIN": sys.executable,
        "CHECKOUT_ROOT": str(REPO_ROOT),
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "BOOT_PARENT": str(tmp_path / "boot-parent"),
    })
    Path(env["UV_CACHE_DIR"]).mkdir()
    env.update(stubs if stubs is not None else _stub_bin(tmp_path))
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
    assert "outside Eagle prefix" in result.stderr


def test_dry_exec_runs_three_train_control_rows(tmp_path: Path) -> None:
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
    assert "contains '..'" in result.stderr


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
        PBS_JOBID="pbs-123.server",
    )
    result = _run("guard", env)
    assert result.returncode == 2
    assert "TPEN_POLARIS_TEST_RESULTS_PREFIX is forbidden" in result.stderr
    assert sentinel.read_bytes() == b"preserve-me\0"


def test_production_rejects_promisor_checkout_before_uv(tmp_path: Path) -> None:
    checkout, source_sha = _make_exact_fixture(tmp_path)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "remote.origin.promisor", "true"],
        check=True,
    )
    env = _production_env(tmp_path, checkout, source_sha)
    env.update({"MODE": "train", "PLAN_MANIFEST": str(tmp_path / "manifest.json")})
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
        "    handle.write(f\"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\\n\")\n"
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
        "printf '%s\\n' \"CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES\" >> \"$DRIVER_RECORD\"\n"
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
        "EXPECTED_QUEUE": "debug", "DRY_EXEC": "0", "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "BOOT_PARENT": str(tmp_path / "boot-parent"), "NVIDIA_SMI_BIN": _stub_bin(tmp_path)["NVIDIA_SMI_BIN"],
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    })
    Path(env["UV_CACHE_DIR"]).mkdir()
    env.update(stubs)
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
        "PLAN_OUTPUT": str(plan_path), "CHECKPOINTS": "50000", "LINEAGE_SEEDS": "1:2:3",
    })
    planned = _run("plan", env)
    assert planned.returncode == 0, planned.stderr
    assert f"POLARIS_TEST_PREFIX_OVERRIDE={env['TPEN_POLARIS_TEST_RESULTS_PREFIX']}" in planned.stderr
    receipt = next(results_root.glob("logs/polaris-plan-*.env"))
    assert f"POLARIS_TEST_PREFIX_OVERRIDE={env['TPEN_POLARIS_TEST_RESULTS_PREFIX']}" in receipt.read_text()
    phase1 = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [row["device"] for row in phase1["rows"] if row["kind"] == "train"] == ["cuda"] * 3

    env.update({"MODE": "train", "PLAN_MANIFEST": str(plan_path), "DRIVER": env["DRIVER"]})
    trained = _run("train", env)
    assert trained.returncode == 0, trained.stderr
    uv_controls = Path(env["UV_RECORD"]).read_text(encoding="utf-8").splitlines()
    assert uv_controls
    assert all(line.endswith("|1|never|copy") for line in uv_controls)
    records = Path(env["DRIVER_RECORD"]).read_text(encoding="utf-8").splitlines()
    assert sorted(records) == ["CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=1", "CUDA_VISIBLE_DEVICES=2"]
    statuses = list(results_root.glob("logs/*.status"))
    assert {"cpu=0-15", "cpu=16-31", "cpu=32-47"} <= {line.split()[2] for path in statuses for line in path.read_text().splitlines()}

    invalid_manifest = results_root / "plan" / "cpu-row.json"
    invalid_doc = json.loads(plan_path.read_text(encoding="utf-8"))
    invalid_doc["rows"][1]["device"] = "cpu"
    invalid_manifest.write_text(json.dumps(invalid_doc), encoding="utf-8")
    before_invalid = Path(env["DRIVER_RECORD"]).read_bytes()
    env.update({"PLAN_MANIFEST": str(invalid_manifest)})
    invalid = _run("train", env)
    assert invalid.returncode == 2
    assert "must have device=cuda" in invalid.stderr
    assert Path(env["DRIVER_RECORD"]).read_bytes() == before_invalid
    invalid_doc["rows"][1].pop("device")
    invalid_manifest.write_text(json.dumps(invalid_doc), encoding="utf-8")
    invalid_missing = _run("train", env)
    assert invalid_missing.returncode == 2
    assert "must have device=cuda" in invalid_missing.stderr
    assert Path(env["DRIVER_RECORD"]).read_bytes() == before_invalid
    env["PLAN_MANIFEST"] = str(plan_path)

    env["FAIL_DRIVER"] = "1"
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
            f"{row['system']}:{row['lineage']}:50000": "0" * 64
            for row in phase1["rows"] if row["kind"] == "train"
        },
    }
    evidence_path = results_root / "plan" / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    phase2_path = results_root / "plan" / "phase2.json"
    subprocess.run([
        sys.executable, "-m", "experiments.atomistic.stage1.plan", "--output", str(phase2_path),
        "--source-sha", source_sha, "--results-root", str(results_root), "--phase", "2",
        "--phase1-manifest", str(plan_path), "--checkpoint-bindings", str(evidence_path),
        "--checkpoints", "50000", "--lineage-seeds", "1", "2", "3",
    ], cwd=checkout, check=True)
    env.update({"MODE": "eval", "PLAN_MANIFEST": str(phase2_path)})
    evaluated = _run("eval", env)
    assert evaluated.returncode == 0, evaluated.stderr
    env["FAIL_DRIVER"] = "all"
    evaluated_fail = _run("eval", env)
    assert evaluated_fail.returncode != 0
    assert len(list(results_root.glob("logs/*.status"))) >= 12
