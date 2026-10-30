"""Allocation-local GPU preflight and real Stage-Q launcher."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib
from common import build_env
from lock import expected_torch_version

RUNTIME_KIND = "polaris-gpu-qualification"
CANONICAL_STAGE = "experiments/atomistic/he-importance/run_stage_q.py"


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        parsed = shlex.split(value, posix=True)
        if len(parsed) != 1:
            raise ValueError(f"run.env value is not one literal token: {key}")
        values[key] = parsed[0]
    return values


def _record(path: Path, checks: list[dict], name: str, expected: object, observed: object, ok: bool, code: int) -> bool:
    row = {"name": name, "expected": expected, "observed": observed, "status": "PASS" if ok else "FAIL", "exit_code": None if ok else code}
    checks.append(row)
    path.write_text(json.dumps({"checks": checks, "exit_code": None if ok else code}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CHECK {name} status={row['status']} observed={observed}", flush=True)
    return ok


def copied_hashes(manifest: dict) -> dict[str, str]:
    harness = Path(manifest["job_helper"]).parent
    return {name: hashlib.sha256((harness / name).read_bytes()).hexdigest() for name in manifest["copied_file_sha256"]}


def hostname() -> str:
    return socket.gethostname().split(".", 1)[0]


def nodefile_entries() -> list[str]:
    path = Path(os.environ["PBS_NODEFILE"])
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def appledoubles(repo: Path) -> list[str]:
    return [str(path) for path in repo.rglob("._*")]


def _short_hosts(entries: list[str]) -> set[str]:
    return {entry.strip().split(".", 1)[0] for entry in entries if entry.strip()}


def _uv_version(uv: list[str]) -> str:
    return subprocess.check_output(uv + ["--version"], text=True).strip()


def _venv_identity(python: Path) -> dict:
    code = "import json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix}))"
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def _package_versions(python: Path) -> dict:
    code = "import importlib.metadata as m,json; print(json.dumps({'torch':m.version('torch')}))"
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def torch_probe(python: str, env: dict[str, str]) -> dict:
    code = ("import json,os,torch; x=torch.ones(1,device='cuda')+1; p=torch.cuda.get_device_properties(0); "
            "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'file':torch.__file__,'available':torch.cuda.is_available(),"
            "'count':torch.cuda.device_count(),'name':torch.cuda.get_device_name(0),'uuid':str(p.uuid),'op':float(x.item()),"
            "'ld':os.environ.get('LD_LIBRARY_PATH',''),'visible':os.environ.get('CUDA_VISIBLE_DEVICES')}))")
    result = subprocess.run([python, "-c", code], env=env, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Torch probe failed: {result.stderr.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def run_route(argv: list[str], cwd: str, env: dict[str, str], stdout, stderr) -> int:
    return subprocess.run(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False).returncode


def run_qualification(python: str, qualification: str, manifest: Path, evidence: Path, cwd: str, env: dict[str, str], stdout, stderr) -> int:
    return subprocess.run([python, qualification, "--manifest", str(manifest), "--evidence", str(evidence)], cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False).returncode


def main() -> int:
    env_path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["RUN_ENV_PATH"])
    values = _load_env(env_path)
    manifest_path = Path(values["RUN_MANIFEST"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = Path(manifest["evidence_dir"]) / os.environ.get("PBS_JOBID", "local")
    evidence.mkdir(parents=True, exist_ok=True)
    preflight = evidence / "preflight.json"
    checks: list[dict] = []

    def check(name, expected, observed, ok, code):
        return None if _record(preflight, checks, name, expected, observed, ok, code) else code

    code = check("runtime_kind", RUNTIME_KIND, manifest.get("runtime_kind"), manifest.get("runtime_kind") == RUNTIME_KIND, 84)
    if code is not None: return code
    receipt = Path(manifest["staging_receipt"])
    receipt_data = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
    expected_receipt = {"runtime_kind": RUNTIME_KIND, "target_sha": manifest["target_sha"], "cache_dir": manifest["cache_dir"], "extra": manifest["extra"]}
    observed_receipt = {key: receipt_data.get(key) for key in expected_receipt}
    code = check("staging_receipt", expected_receipt, observed_receipt, receipt.is_file() and observed_receipt == expected_receipt, 60 if not receipt.is_file() else 61)
    if code is not None: return code
    observed_hashes = copied_hashes(manifest)
    code = check("copied_harness_hashes", manifest["copied_file_sha256"], observed_hashes, observed_hashes == manifest["copied_file_sha256"], 81)
    if code is not None: return code
    node = hostname()
    code = check("compute_host_guard", "not polaris-login-", node, not node.startswith("polaris-login-"), 70)
    if code is not None: return code
    entries = nodefile_entries()
    code = check("pbs_nodefile_host", node, entries, node in _short_hosts(entries), 71)
    if code is not None: return code
    code = check("topology_nodes", manifest["nodes"], len(_short_hosts(entries)), len(_short_hosts(entries)) == manifest["nodes"], 85)
    if code is not None: return code
    repo = Path(manifest["target_repo"])
    head = git_head(repo)
    code = check("target_head", manifest["target_sha"], head, head == manifest["target_sha"], 72)
    if code is not None: return code
    apple = appledoubles(repo)
    code = check("appledouble", 0, len(apple), not apple, 73)
    if code is not None: return code
    for ancestor in manifest.get("required_ancestors", []):
        ok = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, manifest["target_sha"]], check=False).returncode == 0
        code = check(f"required_ancestor:{ancestor}", True, ok, ok, 79)
        if code is not None: return code
    try:
        lock = tomllib.loads((repo / "uv.lock").read_text(encoding="utf-8"))
        expected = expected_torch_version(lock, manifest["extra"])
        lock_ok = expected == manifest["expected_torch"]
        lock_observed = expected
    except (OSError, TypeError, ValueError) as error:
        lock_ok = False
        lock_observed = str(error)
        expected = manifest.get("expected_torch")
    code = check("lock_torch_version", manifest.get("expected_torch"), lock_observed, lock_ok, 78)
    if code is not None: return code
    uv = [values["RUN_PYTHON"], "-m", "uv"]
    uv_text = _uv_version(uv)
    code = check("uv_version", manifest["verify_profile"]["uv_version"], uv_text, uv_text.split()[:2] == ["uv", manifest["verify_profile"]["uv_version"]], 77)
    if code is not None: return code
    job_env_value = os.environ.get("JOB_ENV")
    code = check("job_env_export", "JOB_ENV", job_env_value, bool(job_env_value), 76)
    if code is not None: return code
    job_env = Path(job_env_value).resolve()
    sync_env = build_env(manifest["verify_profile"], Path(values["RUN_CACHE_DIR"]), job_env, base=dict(os.environ))
    sync_log = evidence / "offline-sync.stdout.log"
    sync_error = evidence / "offline-sync.stderr.log"
    try:
        with sync_log.open("w", encoding="utf-8") as stdout, sync_error.open("w", encoding="utf-8") as stderr:
            subprocess.run([values["RUN_PYTHON"], "-m", "uv", "sync", "--offline", "--extra", manifest["extra"], "--locked", "--project", values["RUN_PROJECT"]], env=sync_env, stdout=stdout, stderr=stderr, check=True, text=True)
        sync_ok = True
        sync_observed = 0
    except (OSError, subprocess.CalledProcessError) as error:
        sync_ok = False
        sync_observed = getattr(error, "returncode", str(error))
    code = check("offline_sync", 0, sync_observed, sync_ok, 80)
    if code is not None: return code
    python = job_env / "bin" / "python"
    identity = _venv_identity(python)
    identity_ok = Path(identity["prefix"]).resolve() == job_env and identity["executable"].startswith(str(job_env) + os.sep)
    code = check("venv_identity", str(job_env), identity, identity_ok, 76)
    if code is not None: return code
    versions = _package_versions(python)
    code = check("package_versions", {"torch": expected}, versions, versions == {"torch": expected}, 75)
    if code is not None: return code
    try:
        probe = torch_probe(str(python), dict(os.environ))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        check("torch_probe", "successful child probe", str(error), False, 83)
        return 83
    torch_file = Path(probe.get("file", "")).resolve()
    probe_ok = probe.get("torch") == expected and probe.get("available") and probe.get("count") == 1 and "A100" in probe.get("name", "") and probe.get("op") == 2.0 and probe.get("visible") == "0" and (torch_file == job_env or job_env in torch_file.parents)
    code = check("torch_probe", {"torch": expected, "count": 1, "visible": "0"}, probe, probe_ok, 83)
    if code is not None: return code
    (evidence / "runtime.json").write_text(json.dumps({"cuda_visible_devices": probe.get("visible"), "hostname": node, "nodefile": entries, "torch_probe": probe}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage_path = repo / CANONICAL_STAGE
    blob = subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"{manifest['target_sha']}:{CANONICAL_STAGE}"], text=True).strip()
    code = check("real_route_entry", {"path": CANONICAL_STAGE, "blob": blob}, {"path": CANONICAL_STAGE, "blob": blob, "exists": stage_path.is_file()}, stage_path.is_file(), 64)
    if code is not None: return code
    cell = manifest["cell"]
    argv = [str(python), str(stage_path), "--stage", cell["stage"], "--context", cell["context"], "--optimizer", cell["optimizer"], "--lr", str(cell["lr"]), "--beta2", str(cell["beta2"]), "--seed-label", str(cell["seed_label"]), "--walkers", str(cell["walkers"]), "--output-root", str(manifest["output_root"])]
    (evidence / "selection-argv.json").write_text(json.dumps({"argv": argv, "blob": blob}, indent=2) + "\n", encoding="utf-8")
    run_env = dict(os.environ); run_env["PYTHONPATH"] = manifest["target_repo"]
    with (evidence / "stage.stdout.log").open("w", encoding="utf-8") as stdout, (evidence / "stage.stderr.log").open("w", encoding="utf-8") as stderr:
        route_rc = run_route(argv, manifest["target_repo"], run_env, stdout, stderr)
    (evidence / "stage-result.json").write_text(json.dumps({"argv": argv, "returncode": route_rc}, indent=2) + "\n", encoding="utf-8")
    if route_rc != 0: return 1
    with (evidence / "qualification.stdout.log").open("w", encoding="utf-8") as stdout, (evidence / "qualification.stderr.log").open("w", encoding="utf-8") as stderr:
        qualify_rc = run_qualification(str(python), str(Path(manifest["job_helper"]).parent / "qualification.py"), manifest_path, evidence, manifest["target_repo"], run_env, stdout, stderr)
    return qualify_rc


if __name__ == "__main__":
    raise SystemExit(main())
