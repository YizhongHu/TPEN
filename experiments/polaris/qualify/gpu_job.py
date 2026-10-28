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


RUNTIME_KIND = "polaris-gpu-qualification"
CANONICAL_STAGE = "experiments/atomistic/he-importance/run_stage_q.py"


def _load_env(path: Path) -> dict[str, str]:
    """Read one-token shell assignments without evaluating shell syntax."""
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
    path.write_text(json.dumps({"checks": checks, "exit_code": None}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CHECK {name} status={row['status']} observed={observed}", flush=True)
    if not ok:
        path.write_text(json.dumps({"checks": checks, "exit_code": code}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ok


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts):
        digest.update(str(path.relative_to(root)).encode()); digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def overlay_snapshot(profile: dict) -> dict[str, str]:
    """Capture the immutable overlay files and tree, excluding bytecode."""
    overlay = Path(profile["overlay_root"])
    pyvenv = overlay / "pyvenv.cfg"
    return {"pyvenv": hashlib.sha256(pyvenv.read_bytes()).hexdigest(), "tree": _tree_digest(overlay), "pyvenv_text": pyvenv.read_text(encoding="utf-8")}


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


def torch_probe(python: str, env: dict[str, str]) -> dict:
    """Import Torch in a child process after the shell export boundary."""
    code = (
        "import json, os, torch; "
        "x=torch.ones(1, device='cuda') + 1; "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda, "
        "'file':torch.__file__, 'available':torch.cuda.is_available(), "
        "'count':torch.cuda.device_count(), 'name':torch.cuda.get_device_name(0), "
        "'uuid':str(p.uuid), 'op':float(x.item()), 'ld':os.environ.get('LD_LIBRARY_PATH',''), "
        "'visible':os.environ.get('CUDA_VISIBLE_DEVICES')}))"
    )
    result = subprocess.run([python, "-c", code], env=env, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Torch probe failed: {result.stderr.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def run_route(argv: list[str], cwd: str, env: dict[str, str], stdout, stderr) -> int:
    """Run the real route with streaming evidence and no output capture."""
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
    def check(name: str, expected: object, observed: object, ok: bool, code: int) -> int | None:
        return None if _record(preflight, checks, name, expected, observed, ok, code) else code

    code = check("runtime_kind", RUNTIME_KIND, manifest.get("runtime_kind"), manifest.get("runtime_kind") == RUNTIME_KIND, 84)
    if code is not None: return code
    receipt = Path(manifest["staging_receipt"])
    receipt_data = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
    expected_overlay = {key: manifest["runtime_profile"][key] for key in ("overlay_python", "overlay_root", "facility_site_packages")}
    observed_overlay = receipt_data.get("overlay")
    receipt_code = 60 if not receipt.is_file() else 61
    code = check("staging_receipt", {"runtime_kind": RUNTIME_KIND, "target_sha": manifest["target_sha"], "overlay": expected_overlay}, {"runtime_kind": receipt_data.get("runtime_kind"), "target_sha": receipt_data.get("target_sha"), "overlay": observed_overlay}, receipt.is_file() and receipt_data.get("runtime_kind") == RUNTIME_KIND and receipt_data.get("target_sha") == manifest["target_sha"] and observed_overlay == expected_overlay, receipt_code)
    if code is not None: return code
    code = check("copied_harness_hashes", manifest["copied_file_sha256"], copied_hashes(manifest), copied_hashes(manifest) == manifest["copied_file_sha256"], 81)
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
    profile = manifest["runtime_profile"]
    try:
        before = overlay_snapshot(profile)
    except (OSError, ValueError) as error:
        code = check("overlay_snapshot", "readable", str(error), False, 82)
        return code or 82
    required_libs = [profile["libshim"], profile["cuda13_lib"], profile["cuda129_lib"]]
    code = check("library_profile_paths", required_libs, required_libs, all(Path(path).is_dir() for path in required_libs), 82)
    if code is not None: return code
    applied = os.environ.get("LD_LIBRARY_PATH", "")
    expected_prefix = ":".join(required_libs)
    code = check("library_profile_applied", expected_prefix, applied, applied.startswith(expected_prefix) and "libfabric" in applied, 82)
    if code is not None: return code
    try:
        probe = torch_probe(values["RUN_PYTHON"], dict(os.environ))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        code = check("torch_probe", "successful child probe", str(error), False, 83)
        return code or 83
    facility_site = Path(profile["facility_site_packages"]).resolve()
    torch_file = Path(probe.get("file", "")).resolve()
    code = check("torch_probe", {"torch": "2.8.0", "cuda": "12.9", "count": 1, "visible": "0"}, probe, probe.get("torch", "").split("+")[0] == "2.8.0" and probe.get("cuda") == "12.9" and probe.get("available") and probe.get("count") == 1 and "A100" in probe.get("name", "") and probe.get("op") == 2.0 and probe.get("visible") == "0" and facility_site in torch_file.parents and probe.get("ld", "").startswith(expected_prefix), 83)
    if code is not None: return code
    (evidence / "runtime.json").write_text(json.dumps({"inherited_ld_library_path": os.environ.get("LD_LIBRARY_PATH"), "applied_ld_library_path": probe.get("ld"), "cuda_visible_devices": probe.get("visible"), "hostname": node, "nodefile": entries, "overlay_before": before, "torch_probe": probe}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage_path = repo / CANONICAL_STAGE
    blob = subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"{manifest['target_sha']}:{CANONICAL_STAGE}"], text=True).strip()
    code = check("real_route_entry", {"path": CANONICAL_STAGE, "blob": blob}, {"path": CANONICAL_STAGE, "blob": blob, "exists": stage_path.is_file()}, stage_path.is_file(), 64)
    if code is not None: return code
    cell = manifest["cell"]
    argv = [values["RUN_PYTHON"], str(stage_path), "--stage", cell["stage"], "--context", cell["context"], "--optimizer", cell["optimizer"], "--lr", str(cell["lr"]), "--beta2", str(cell["beta2"]), "--seed-label", str(cell["seed_label"]), "--walkers", str(cell["walkers"]), "--output-root", str(manifest["output_root"])]
    (evidence / "selection-argv.json").write_text(json.dumps({"argv": argv, "blob": blob}, indent=2) + "\n", encoding="utf-8")
    run_env = dict(os.environ); run_env["PYTHONPATH"] = manifest["target_repo"]
    with (evidence / "stage.stdout.log").open("w", encoding="utf-8") as stdout, (evidence / "stage.stderr.log").open("w", encoding="utf-8") as stderr:
        route_rc = run_route(argv, manifest["target_repo"], run_env, stdout, stderr)
    (evidence / "stage-result.json").write_text(json.dumps({"argv": argv, "returncode": route_rc}, indent=2) + "\n", encoding="utf-8")
    if route_rc != 0: return 1
    after = overlay_snapshot(profile)
    code = check("overlay_unchanged_after_payload", before, after, before == after, 83)
    if code is not None: return code
    with (evidence / "qualification.stdout.log").open("w", encoding="utf-8") as stdout, (evidence / "qualification.stderr.log").open("w", encoding="utf-8") as stderr:
        qualify_rc = run_qualification(values["RUN_PYTHON"], str(Path(manifest["job_helper"]).parent / "qualification.py"), manifest_path, evidence, manifest["target_repo"], run_env, stdout, stderr)
    after_qualification = overlay_snapshot(profile)
    code = check("overlay_unchanged_after_qualification", before, after_qualification, before == after_qualification, 83)
    if code is not None: return code
    return qualify_rc


if __name__ == "__main__":
    raise SystemExit(main())
