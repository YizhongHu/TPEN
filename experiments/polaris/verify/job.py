"""Allocation-local preflight and driver launcher for the stored PBS script."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib

from common import build_env, copied_hashes, is_ancestor


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
    cpu_entries = [entry for entry in cpu_requirements if entry.get("name") == "torch"]
    linux_cpu = [entry for entry in cpu_entries if "sys_platform != 'darwin'" in entry.get("marker", "")]
    if len(linux_cpu) != 1 or "version" not in linux_cpu[0]:
        raise ValueError("lock must contain one Linux torch resolution in the cpu extra")
    expected["torch"] = linux_cpu[0]["version"]
    if set(expected) != {"torch", "pytest", "typeguard"}:
        raise ValueError("lock is missing a required verification package")
    return expected


def _read_env(path: Path) -> dict[str, str]:
    """Read the literal run environment without evaluating shell syntax."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        parsed = shlex.split(value, posix=True)
        if len(parsed) != 1:
            raise ValueError(f"run.env value is not one literal token: {key}")
        values[key] = parsed[0]
    return values


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


def preflight_decision(*, receipt_status: int | None = None, login_host: bool | None = None,
                       node_match: bool | None = None, head_ok: bool | None = None,
                       ancestor_ok: bool | None = None, apple_double_count: int | None = None,
                       lock_ok: bool | None = None, uv_ok: bool | None = None,
                       sync_ok: bool | None = None, python_exists: bool | None = None,
                       interpreter_ok: bool | None = None, versions_ok: bool | None = None,
                       copied_ok: bool | None = None) -> int:
    """Return the first failed check in the same order as ``main``."""
    if copied_ok is False:
        return 81
    if receipt_status:
        return receipt_status
    if login_host:
        return 70
    if node_match is False:
        return 71
    if ancestor_ok is False:
        return 79
    if head_ok is False:
        return 72
    if apple_double_count not in (None, 0):
        return 73
    if lock_ok is False:
        return 78
    if uv_ok is False:
        return 77
    if sync_ok is False:
        return 80
    if python_exists is False:
        return 74
    if interpreter_ok is False:
        return 76
    if versions_ok is False:
        return 75
    return 0


def _hostname() -> str:
    """Read the short hostname used by the compute-node guard."""
    return subprocess.check_output(["hostname", "-s"], text=True).strip()


def _nodefile_entries() -> list[str]:
    """Read PBS nodefile entries for host matching."""
    path = Path(os.environ["PBS_NODEFILE"])
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def _git_head(repo: Path) -> str:
    """Read the target checkout HEAD."""
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def head_code(repo: Path, target_sha: str) -> int:
    """Return zero for an exact checkout or the contract's stale-head code."""
    return 0 if _git_head(repo) == target_sha else 72


def _appledoubles(repo: Path) -> list[str]:
    """Find macOS AppleDouble files in the target checkout."""
    return [str(path) for path in repo.rglob("._*")]


def _lock_text(repo: Path) -> str:
    """Read the target lockfile for CPU package derivation."""
    return (repo / "uv.lock").read_text(encoding="utf-8")


def _uv_version(uv: list[str]) -> str:
    """Read the facility uv version through its pinned interpreter."""
    return subprocess.check_output(uv + ["--version"], text=True).strip()


def _offline_sync(uv: list[str], repo: Path, env: dict[str, str], log: Path) -> int:
    """Run the locked offline sync and return its process status."""
    try:
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(uv + ["sync", "--offline", "--extra", "cpu", "--locked", "--project", str(repo)], env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


def _venv_exists(python: Path) -> bool:
    """Return whether the offline sync produced the job interpreter."""
    return python.is_file()


def _venv_identity(python: Path) -> dict:
    """Read interpreter identity from inside the job venv."""
    command = "import json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'base_executable':sys._base_executable}))"
    return json.loads(subprocess.check_output([str(python), "-c", command], text=True))


def _package_versions(python: Path) -> dict:
    """Read verification package versions from inside the job venv."""
    command = "import importlib.metadata as m, json; print(json.dumps({n:m.version(n) for n in ('torch','pytest','typeguard')}))"
    return json.loads(subprocess.check_output([str(python), "-c", command], text=True))


def _run_driver(python: Path, manifest: dict, manifest_path: Path, evidence: Path, env: dict[str, str], job_env: Path) -> int:
    """Run the manifest-selected driver with captured stdout and stderr."""
    command = [str(python), "-B", manifest["driver"], "--manifest", str(manifest_path), "--evidence", str(evidence), "--job-python", str(python), "--job-env", str(job_env)]
    with (evidence / "driver.stdout.log").open("w", encoding="utf-8") as stdout, (evidence / "driver.stderr.log").open("w", encoding="utf-8") as stderr:
        return subprocess.run(command, env=env, stdout=stdout, stderr=stderr).returncode


def _record(path: Path, checks: list[dict], name: str, expected, observed, ok: bool, exit_code: int | None = None) -> bool:
    pending = {"name": name, "expected": expected, "observed": observed, "ok": None, "exit_code": None}
    checks.append(pending)
    path.write_text(json.dumps({"checks": checks, "exit_code": None}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.update(ok=ok, exit_code=None if ok else exit_code)
    path.write_text(json.dumps({"checks": checks, "exit_code": None if ok else exit_code}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CHECK {name} ok={ok} observed={observed}", flush=True)
    return ok


def main() -> int:
    """Run the single ordered preflight path and invoke the selected driver."""
    env_path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["RUN_ENV_PATH"])
    values = _read_env(env_path)
    manifest_path = Path(values["RUN_MANIFEST"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job_id = os.environ["PBS_JOBID"]
    evidence = Path(manifest["evidence_dir"]) / job_id
    evidence.mkdir(parents=True, exist_ok=True)
    preflight = evidence / "preflight.json"
    checks: list[dict] = []
    state = {name: None for name in ("copied_ok", "receipt_status", "login_host", "node_match", "head_ok", "ancestor_ok", "apple_double_count", "lock_ok", "uv_ok", "sync_ok", "python_exists", "interpreter_ok", "versions_ok")}

    def decide() -> int:
        """Route every early exit through the one ordered decision table."""
        return preflight_decision(**state)

    copied_expected = manifest["copied_file_sha256"]
    copied_observed = copied_hashes([Path(manifest["job_helper"]).parent / name for name in copied_expected])
    state["copied_ok"] = copied_observed == copied_expected
    _record(preflight, checks, "copied_harness_hashes", copied_expected, copied_observed, state["copied_ok"], 81)
    if decide():
        return decide()

    receipt_path = Path(manifest["staging_receipt"])
    state["receipt_status"] = receipt_code(manifest, receipt_path)
    _record(preflight, checks, "staging_receipt", 0, state["receipt_status"], state["receipt_status"] == 0, state["receipt_status"] or 1)
    if decide():
        return decide()

    node = _hostname()
    guard_prefix = manifest["profile"]["compute_host_guard"]["refuse_prefix"]
    state["login_host"] = node.startswith(guard_prefix)
    _record(preflight, checks, "compute_host_guard", f"not prefix {guard_prefix}", node, not state["login_host"], 70)
    if decide():
        return decide()
    entries = _nodefile_entries()
    state["node_match"] = host_matches(node, entries)
    _record(preflight, checks, "pbs_nodefile_host", node, entries, state["node_match"], 71)
    if decide():
        return decide()

    repo = Path(manifest["target_repo"])
    for ancestor in manifest.get("required_ancestors", []):
        state["ancestor_ok"] = is_ancestor(repo, ancestor, manifest["target_sha"])
        _record(preflight, checks, f"required_ancestor:{ancestor}", True, state["ancestor_ok"], state["ancestor_ok"], 79)
        if decide():
            return decide()
    head = _git_head(repo)
    state["head_ok"] = head == manifest["target_sha"]
    _record(preflight, checks, "target_head", manifest["target_sha"], head, state["head_ok"], 72)
    if decide():
        return decide()
    apple_doubles = _appledoubles(repo)
    state["apple_double_count"] = len(apple_doubles)
    _record(preflight, checks, "appledouble", 0, len(apple_doubles), not apple_doubles, 73)
    if decide():
        return decide()
    try:
        expected = expected_versions(tomllib.loads(_lock_text(repo)))
        state["lock_ok"] = True
    except (KeyError, TypeError, ValueError) as exc:
        expected, state["lock_ok"] = {"error": str(exc)}, False
    _record(preflight, checks, "lock_versions", "CPU extra resolution", expected, state["lock_ok"], 78)
    if decide():
        return decide()

    profile = manifest["profile"]
    job_env = Path(profile["job_scratch_root"]) / job_id / "tpen-verify-env"
    job_env.parent.mkdir(parents=True, exist_ok=True)
    env = build_env(profile, Path(manifest["cache_dir"]), job_env)
    uv = [profile["facility_python"], "-m", "uv"]
    uv_version = _uv_version(uv)
    state["uv_ok"] = uv_version.split()[:2] == ["uv", profile["uv_version"]]
    _record(preflight, checks, "uv_version", ["uv", profile["uv_version"]], uv_version, state["uv_ok"], 77)
    if decide():
        return decide()
    sync_rc = _offline_sync(uv, repo, env, evidence / "offline-sync.stdout.log")
    state["sync_ok"] = sync_rc == 0
    _record(preflight, checks, "offline_sync", 0, sync_rc, state["sync_ok"], 80)
    if decide():
        return decide()
    python = job_env / "bin" / "python"
    state["python_exists"] = _venv_exists(python)
    _record(preflight, checks, "venv_python_exists", str(python), str(python) if state["python_exists"] else None, state["python_exists"], 74)
    if decide():
        return decide()
    identity = _venv_identity(python)
    facility = Path(profile["facility_python"]).resolve()
    base_prefix = facility.parent.parent
    state["interpreter_ok"] = (Path(identity["prefix"]).resolve() == job_env.resolve() and Path(identity["base_executable"]).resolve() == facility and Path(identity["base_prefix"]).resolve() == base_prefix.resolve())
    _record(preflight, checks, "venv_interpreter", {"prefix": str(job_env), "base_executable": str(facility), "base_prefix": str(base_prefix)}, identity, state["interpreter_ok"], 76)
    if decide():
        return decide()
    observed = _package_versions(python)
    state["versions_ok"] = observed == expected
    _record(preflight, checks, "package_versions", expected, observed, state["versions_ok"], 75)
    if decide():
        return decide()
    driver_rc = _run_driver(python, manifest, manifest_path, evidence, env, job_env)
    _record(preflight, checks, "driver", 0, driver_rc, driver_rc == 0, driver_rc or 1)
    return driver_rc


if __name__ == "__main__":
    raise SystemExit(main())
