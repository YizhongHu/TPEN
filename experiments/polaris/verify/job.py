"""Allocation-local preflight and driver launcher for the stored PBS script."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import shlex
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


def preflight_decision(*, receipt_status: int = 0, login_host: bool, node_match: bool,
                       head_ok: bool = True, ancestor_ok: bool = True,
                       apple_double_count: int, python_exists: bool,
                       interpreter_ok: bool, versions_ok: bool, uv_ok: bool,
                       lock_ok: bool, sync_ok: bool, copied_ok: bool) -> int:
    """Map injected preflight observations to the stable exit-code contract."""
    if receipt_status:
        return receipt_status
    if login_host:
        return 70
    if not node_match:
        return 71
    if not head_ok:
        return 72
    if not ancestor_ok:
        return 79
    if apple_double_count:
        return 73
    if not python_exists:
        return 74
    if not interpreter_ok:
        return 76
    if not versions_ok:
        return 75
    if not uv_ok:
        return 77
    if not lock_ok:
        return 78
    if not sync_ok:
        return 80
    if not copied_ok:
        return 81
    return 0


def head_code(repo: Path, target_sha: str) -> int:
    """Return zero for an exact checkout or the contract's stale-head code."""
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return 0 if head == target_sha else 72


def _record(path: Path, checks: list[dict], name: str, expected, observed, ok: bool, exit_code: int | None = None) -> bool:
    pending = {"name": name, "expected": expected, "observed": observed, "ok": None, "exit_code": None}
    checks.append(pending)
    path.write_text(json.dumps({"checks": checks, "exit_code": None}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.update(ok=ok, exit_code=None if ok else exit_code)
    path.write_text(json.dumps({"checks": checks, "exit_code": None if ok else exit_code}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CHECK {name} ok={ok} observed={observed}", flush=True)
    return ok


def _injected_main(deps) -> int:
    """Run the decision path with injected observations for seam tests."""
    env_path = Path(os.environ["RUN_ENV_PATH"])
    values = _read_env(env_path)
    manifest_path = Path(values["RUN_MANIFEST"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job_id = os.environ["PBS_JOBID"]
    evidence = Path(manifest["evidence_dir"]) / job_id
    evidence.mkdir(parents=True, exist_ok=True)
    preflight = evidence / "preflight.json"
    checks: list[dict] = []
    observed = {name: (value() if callable(value) else value) for name, value in deps.items() if name != "driver_runner"}
    status = preflight_decision(**observed)
    observations = {
        "receipt": observed.get("receipt_status", 0), "login_host": observed["login_host"],
        "node_match": observed["node_match"], "head": observed.get("head_ok", True),
        "ancestor": observed.get("ancestor_ok", True),
        "apple_double": observed["apple_double_count"], "python": observed["python_exists"],
        "interpreter": observed["interpreter_ok"], "versions": observed["versions_ok"],
        "uv": observed["uv_ok"], "lock": observed["lock_ok"], "sync": observed["sync_ok"],
        "copied": observed["copied_ok"],
    }
    _record(preflight, checks, "injected_observations", "all preflight checks", observations, status == 0, status or 1)
    if status:
        return status
    driver_rc = int(deps["driver_runner"]())
    _record(preflight, checks, "driver", 0, driver_rc, driver_rc == 0, driver_rc or 1)
    return driver_rc


def _real_main() -> int:
    """Run preflight checks and invoke the manifest-selected driver."""
    env_path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["RUN_ENV_PATH"])
    values = _read_env(env_path)
    manifest_path = Path(values["RUN_MANIFEST"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job_id = os.environ["PBS_JOBID"]
    evidence = Path(manifest["evidence_dir"]) / job_id
    evidence.mkdir(parents=True, exist_ok=True)
    preflight = evidence / "preflight.json"
    checks: list[dict] = []

    # Verify copied code before invoking any other copied harness helper.
    copied_expected = manifest["copied_file_sha256"]
    copied_observed = copied_hashes([Path(manifest["job_helper"]).parent / name for name in copied_expected])
    if not _record(preflight, checks, "copied_harness_hashes", copied_expected, copied_observed, copied_observed == copied_expected, 81):
        return 81

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
        ok = is_ancestor(repo, ancestor, manifest["target_sha"])
        if not _record(preflight, checks, f"required_ancestor:{ancestor}", True, ok, ok, 79):
            return 79
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if not _record(preflight, checks, "target_head", manifest["target_sha"], head, head_code(repo, manifest["target_sha"]) == 0, 72):
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
    job_env = Path(profile["job_scratch_root"]) / job_id / "tpen-verify-env"
    job_env.parent.mkdir(parents=True, exist_ok=True)
    env = build_env(profile, Path(manifest["cache_dir"]), job_env)
    uv = [profile["facility_python"], "-m", "uv"]
    uv_version = subprocess.check_output(uv + ["--version"], text=True).strip()
    if not _record(preflight, checks, "uv_version", ["uv", profile["uv_version"]], uv_version, uv_version.split()[:2] == ["uv", profile["uv_version"]], 77):
        return 77
    try:
        with (evidence / "offline-sync.stdout.log").open("w", encoding="utf-8") as sync_log:
            subprocess.run(uv + ["sync", "--offline", "--extra", "cpu", "--locked", "--project", str(repo)], env=env, check=True, stdout=sync_log, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        _record(preflight, checks, "offline_sync", 0, exc.returncode, False, 80)
        return 80
    python = job_env / "bin" / "python"
    if not _record(preflight, checks, "venv_python_exists", str(python), str(python) if python.is_file() else None, python.is_file(), 74):
        return 74
    identity = json.loads(subprocess.check_output([str(python), "-c", "import json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'base_executable':sys._base_executable}))"], text=True))
    facility = Path(profile["facility_python"]).resolve()
    base_prefix = facility.parent.parent
    interpreter_ok = Path(identity["prefix"]).resolve() == job_env.resolve() and Path(identity["base_executable"]).resolve() == facility and Path(identity["base_prefix"]).resolve() == base_prefix.resolve()
    if not _record(preflight, checks, "venv_interpreter", {"prefix": str(job_env), "base_executable": str(facility), "base_prefix": str(base_prefix)}, identity, interpreter_ok, 76):
        return 76
    observed = json.loads(subprocess.check_output([str(python), "-c", "import importlib.metadata as m, json; print(json.dumps({n:m.version(n) for n in ('torch','pytest','typeguard')}))"], text=True))
    if not _record(preflight, checks, "package_versions", expected, observed, observed == expected, 75):
        return 75
    with (evidence / "driver.stdout.log").open("w", encoding="utf-8") as stdout, (evidence / "driver.stderr.log").open("w", encoding="utf-8") as stderr:
        driver_rc = subprocess.run([str(python), "-B", manifest["driver"], "--manifest", str(manifest_path), "--evidence", str(evidence), "--job-python", str(python), "--job-env", str(job_env)], env=env, stdout=stdout, stderr=stderr).returncode
    _record(preflight, checks, "driver", 0, driver_rc, driver_rc == 0, driver_rc or 1)
    return driver_rc


def main(deps=None) -> int:
    """Run preflight checks and invoke the manifest-selected driver.

    Parameters
    ----------
    deps : dict, optional
        Injected observations and a ``driver_runner`` callable for seam tests.
    """
    return _injected_main(deps) if deps is not None else _real_main()


if __name__ == "__main__":
    raise SystemExit(main())
