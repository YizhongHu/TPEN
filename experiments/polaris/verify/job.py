"""Allocation-local preflight and driver launcher for the stored PBS script."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import shlex
import tomllib


def expected_versions(lock: dict) -> dict[str, str]:
    """Select the CPU torch resolution and the unique test dependencies."""
    packages = lock.get("package", [])
    expected: dict[str, str] = {}
    for package in packages:
        name = package.get("name")
        if name in {"pytest", "typeguard"}:
            if name in expected:
                raise ValueError(f"ambiguous lock entry for {name}")
            expected[name] = package["version"]
    cpu_torch = [
        package for package in packages
        if package.get("name") == "torch"
        and package.get("source", {}).get("registry", "").rstrip("/").endswith("/cpu")
    ]
    if len(cpu_torch) != 2:
        raise ValueError("lock must contain torch's plain and +cpu CPU resolutions")
    cpu_torch.sort(key=lambda package: package["version"].endswith("+cpu"), reverse=True)
    expected["torch"] = cpu_torch[0]["version"]
    if set(expected) != {"torch", "pytest", "typeguard"}:
        raise ValueError("lock is missing a required verification package")
    return expected


def main() -> int:
    env_path = Path(os.environ["RUN_ENV_PATH"])
    manifest_path = Path(shlex.split(next(line for line in env_path.read_text().splitlines() if line.startswith("RUN_MANIFEST=")), posix=True)[0].split("=", 1)[1])
    manifest = json.loads(manifest_path.read_text())
    receipt_path = Path(manifest["staging_receipt"])
    if not receipt_path.is_file():
        return 60
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("target_sha") != manifest["target_sha"] or receipt.get("cache_dir") != manifest["cache_dir"]:
        return 61
    evidence = Path(manifest["evidence_dir"]) / os.environ["PBS_JOBID"]
    evidence.mkdir(parents=True, exist_ok=True)
    node = subprocess.check_output(["hostname", "-s"], text=True).strip()
    if node.startswith(manifest["profile"]["compute_host_guard"]["refuse_prefix"]):
        return 70
    nodefile = Path(os.environ["PBS_NODEFILE"])
    if not nodefile.is_file() or not any(line.strip().split(".", 1)[0] == node for line in nodefile.read_text().splitlines()):
        return 71
    repo = Path(manifest["target_repo"])
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != manifest["target_sha"]:
        return 72
    if list(repo.rglob("._*")):
        return 73
    lock = tomllib.loads((repo / "uv.lock").read_text())
    try:
        expected = expected_versions(lock)
    except (KeyError, TypeError, ValueError):
        return 78
    profile = manifest["profile"]
    env = os.environ.copy()
    env.update(dict(item.split("=", 1) for item in profile["uv_flags"]))
    env["UV_CACHE_DIR"] = manifest["cache_dir"]
    job_env = Path("/local/scratch") / os.environ["PBS_JOBID"] / "tpen-verify-env"
    env["UV_PROJECT_ENVIRONMENT"] = str(job_env)
    uv = [profile["facility_python"], "-m", "uv"]
    uv_version = subprocess.check_output(uv + ["--version"], text=True).strip()
    if profile["uv_version"] not in uv_version:
        return 77
    subprocess.run(uv + ["sync", "--offline", "--extra", "cpu", "--locked", "--project", str(repo)], env=env, check=True)
    python = job_env / "bin" / "python"
    if not python.is_file():
        return 74
    observed = json.loads(subprocess.check_output([str(python), "-c", "import importlib.metadata as m, json; print(json.dumps({n:m.version(n) for n in ('torch','pytest','typeguard')}))"], text=True))
    if any(observed.get(name) != expected.get(name) for name in ("torch", "pytest", "typeguard")):
        return 75
    if str(python) != str(job_env / "bin" / "python"):
        return 76
    return subprocess.run([str(python), "-B", manifest["driver"], "--manifest", str(manifest_path), "--evidence", str(evidence)], env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
