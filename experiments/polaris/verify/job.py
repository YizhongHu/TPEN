"""Allocation-local preflight and driver launcher for the stored PBS script."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib


def expected_versions(lock: dict) -> dict[str, str]:
    """Select the Linux resolution declared by the project's ``cpu`` extra."""
    packages = lock.get("package", [])
    project = next((package for package in packages if package.get("source", {}).get("editable") == "."), None)
    cpu_requirements = (project or {}).get("optional-dependencies", {}).get("cpu", [])
    if not any(requirement.get("name") == "torch" for requirement in cpu_requirements):
        raise ValueError("lock project does not declare torch in its cpu extra")
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
        and "sys_platform != 'darwin'" in package.get("marker", "")
    ]
    if len(cpu_torch) != 1:
        raise ValueError("lock must contain one Linux torch resolution for its cpu extra")
    expected["torch"] = cpu_torch[0]["version"]
    if set(expected) != {"torch", "pytest", "typeguard"}:
        raise ValueError("lock is missing a required verification package")
    return expected


def _read_env(path: Path) -> dict[str, str]:
    """Read the literal run environment without evaluating shell syntax."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        values[key] = value.strip("'")
    return values


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    ).returncode == 0


def host_matches(node: str, nodefile_entries: list[str]) -> bool:
    """Compare short hostnames after normalising PBS FQDN entries."""
    return node in [entry.strip().split(".", 1)[0] for entry in nodefile_entries]


def receipt_matches(manifest: dict, receipt: dict) -> bool:
    """Require the staging receipt to bind both target and cache identity."""
    return receipt.get("target_sha") == manifest["target_sha"] and receipt.get("cache_dir") == manifest["cache_dir"]


def receipt_code(manifest: dict, receipt_path: Path) -> int:
    """Return the distinct missing, mismatched, or matching receipt code."""
    if not receipt_path.is_file():
        return 60
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return 0 if receipt_matches(manifest, receipt) else 61


class _Tee:
    """Mirror job output into durable evidence while retaining PBS output."""

    def __init__(self, original, path: Path):
        self.original = original
        self.stream = path.open("a", encoding="utf-8")

    def write(self, text: str) -> int:
        self.original.write(text)
        self.stream.write(text)
        self.stream.flush()
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        self.stream.flush()


def _record(path: Path, checks: list[dict], name: str, expected, observed, ok: bool, exit_code: int | None = None) -> bool:
    checks.append({"name": name, "expected": expected, "observed": observed, "ok": ok, "exit_code": None if ok else exit_code})
    path.write_text(json.dumps({"checks": checks, "exit_code": None if ok else exit_code}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ok


def main() -> int:
    """Run preflight checks and invoke the manifest-selected driver."""
    env_path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["RUN_ENV_PATH"])
    values = _read_env(env_path)
    manifest_path = Path(values["RUN_MANIFEST"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job_id = os.environ["PBS_JOBID"]
    evidence = Path(manifest["evidence_dir"]) / job_id
    evidence.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.stdout, evidence / "job.stdout.log")
    sys.stderr = _Tee(sys.stderr, evidence / "job.stderr.log")
    preflight = evidence / "preflight.json"
    checks: list[dict] = []

    receipt_path = Path(manifest["staging_receipt"])
    receipt_status = receipt_code(manifest, receipt_path)
    if not _record(preflight, checks, "staging_receipt_exists", True, receipt_path.is_file(), receipt_status != 60, 60):
        return 60
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_match = receipt_matches(manifest, receipt)
    if not _record(preflight, checks, "staging_receipt_matches", {"target_sha": manifest["target_sha"], "cache_dir": manifest["cache_dir"]}, {"target_sha": receipt.get("target_sha"), "cache_dir": receipt.get("cache_dir")}, receipt_match, 61):
        return 61

    node = subprocess.check_output(["hostname", "-s"], text=True).strip()
    guard_prefix = manifest["profile"]["compute_host_guard"]["refuse_prefix"]
    if not _record(preflight, checks, "compute_host_guard", f"not prefix {guard_prefix}", node, not node.startswith(guard_prefix), 70):
        return 70
    nodefile = Path(os.environ["PBS_NODEFILE"])
    entries = nodefile.read_text(encoding="utf-8").splitlines() if nodefile.is_file() else []
    if not _record(preflight, checks, "pbs_nodefile_host", node, entries, host_matches(node, entries), 71):
        return 71

    repo = Path(manifest["target_repo"])
    for ancestor in manifest.get("required_ancestors", []):
        ok = _is_ancestor(repo, ancestor, manifest["target_sha"])
        if not _record(preflight, checks, f"required_ancestor:{ancestor}", True, ok, ok, 79):
            return 79
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if not _record(preflight, checks, "target_head", manifest["target_sha"], head, head == manifest["target_sha"], 72):
        return 72
    apple_doubles = [str(path) for path in repo.rglob("._*")]
    if not _record(preflight, checks, "appledouble", 0, len(apple_doubles), not apple_doubles, 73):
        return 73
    try:
        expected = expected_versions(tomllib.loads((repo / "uv.lock").read_text(encoding="utf-8")))
        lock_ok = True
    except (KeyError, TypeError, ValueError) as exc:
        expected, lock_ok = {"error": str(exc)}, False
    if not _record(preflight, checks, "lock_versions", "CPU extra resolution", expected, lock_ok, 78):
        return 78

    profile = manifest["profile"]
    env = os.environ.copy()
    env.update(dict(item.split("=", 1) for item in profile["uv_flags"]))
    env["UV_CACHE_DIR"] = manifest["cache_dir"]
    job_env = Path("/local/scratch") / job_id / "tpen-verify-env"
    env["UV_PROJECT_ENVIRONMENT"] = str(job_env)
    uv = [profile["facility_python"], "-m", "uv"]
    uv_version = subprocess.check_output(uv + ["--version"], text=True).strip()
    if not _record(preflight, checks, "uv_version", "uv " + profile["uv_version"], uv_version, uv_version == "uv " + profile["uv_version"], 77):
        return 77
    try:
        subprocess.run(uv + ["sync", "--offline", "--extra", "cpu", "--locked", "--project", str(repo)], env=env, check=True)
    except subprocess.CalledProcessError as exc:
        _record(preflight, checks, "offline_sync", 0, exc.returncode, False, 79)
        return 79
    python = job_env / "bin" / "python"
    if not _record(preflight, checks, "venv_python_exists", str(python), str(python) if python.is_file() else None, python.is_file(), 74):
        return 74
    identity = json.loads(subprocess.check_output([str(python), "-c", "import json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix}))"], text=True))
    interpreter_ok = Path(identity["executable"]).resolve() == python.resolve() and Path(identity["prefix"]).resolve() == job_env.resolve()
    if not _record(preflight, checks, "venv_interpreter", {"executable": str(python), "prefix": str(job_env)}, identity, interpreter_ok, 76):
        return 76
    observed = json.loads(subprocess.check_output([str(python), "-c", "import importlib.metadata as m, json; print(json.dumps({n:m.version(n) for n in ('torch','pytest','typeguard')}))"], text=True))
    if not _record(preflight, checks, "package_versions", expected, observed, observed == expected, 75):
        return 75
    driver_rc = subprocess.run([str(python), "-B", manifest["driver"], "--manifest", str(manifest_path), "--evidence", str(evidence), "--job-python", str(python)], env=env).returncode
    _record(preflight, checks, "driver", 0, driver_rc, driver_rc == 0, driver_rc or 1)
    return driver_rc


if __name__ == "__main__":
    raise SystemExit(main())
