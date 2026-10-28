"""Allocation-local GPU preflight and real Stage-Q launcher."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from qualification import qualify


def _load_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        values[key] = value.strip("'\"")
    return values


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(str(path.relative_to(root)).encode()); digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _copied_hashes(manifest: dict) -> dict[str, str]:
    harness = Path(manifest["job_helper"]).parent
    return {name: hashlib.sha256((harness / name).read_bytes()).hexdigest() for name in manifest["copied_file_sha256"]}


def _nodes() -> set[str]:
    nodefile = Path(os.environ.get("PBS_NODEFILE", ""))
    if not nodefile.is_file():
        return set()
    return {line.strip().split(".", 1)[0] for line in nodefile.read_text().splitlines() if line.strip()}


def _profile_env(profile: dict) -> dict[str, str]:
    additions = [profile["libshim"], profile["cuda13_lib"], profile["cuda129_lib"]]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    value = ":".join(additions + ([existing] if existing else []))
    env = dict(os.environ); env["LD_LIBRARY_PATH"] = value; env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def main() -> int:
    env_path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["RUN_ENV_PATH"])
    values = _load_env(env_path)
    manifest_path = Path(values["RUN_MANIFEST"]); manifest = json.loads(manifest_path.read_text())
    evidence = Path(manifest["evidence_dir"]) / os.environ.get("PBS_JOBID", "local")
    evidence.mkdir(parents=True, exist_ok=True)
    if manifest.get("runtime_kind") != "polaris-gpu-qualification":
        return 84
    receipt = Path(manifest["staging_receipt"])
    if not receipt.is_file() or json.loads(receipt.read_text()).get("runtime_kind") != manifest["runtime_kind"]:
        return 61
    if _copied_hashes(manifest) != manifest["copied_file_sha256"]:
        return 81
    if len(_nodes()) != manifest["nodes"]:
        return 84
    profile = manifest["runtime_profile"]
    overlay = Path(profile["overlay_root"]); pyvenv = overlay / "pyvenv.cfg"
    before = {"pyvenv": hashlib.sha256(pyvenv.read_bytes()).hexdigest(), "tree": _tree_digest(overlay)}
    env = _profile_env(profile)
    if not all(Path(path).is_dir() for path in (profile["libshim"], profile["cuda13_lib"], profile["cuda129_lib"])):
        return 82
    os.environ.update(env)
    import torch
    if torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda != "12.9":
        return 83
    facility_site = Path(profile["facility_site_packages"]).resolve()
    if facility_site not in Path(torch.__file__).resolve().parents:
        return 83
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
        return 83
    _ = torch.ones(1, device="cuda") + 1
    after = {"pyvenv": hashlib.sha256(pyvenv.read_bytes()).hexdigest(), "tree": _tree_digest(overlay)}
    (evidence / "runtime.json").write_text(json.dumps({"profile": profile, "before": before, "after": after, "torch": torch.__version__, "cuda": torch.version.cuda, "torch_file": torch.__file__}, indent=2) + "\n")
    if before != after:
        return 83
    cell = manifest["cell"]
    argv = [str(Path(values["RUN_PYTHON"])), str(Path(manifest["target_repo"]) / "experiments/atomistic/he-importance/run_stage_q.py"), "--stage", cell["stage"], "--context", cell["context"], "--optimizer", cell["optimizer"], "--lr", str(cell["lr"]), "--beta2", str(cell["beta2"]), "--seed-label", str(cell["seed_label"]), "--walkers", str(cell["walkers"]), "--output-root", str(manifest["output_root"])]
    (evidence / "selection-argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    run_env = dict(os.environ); run_env["PYTHONPATH"] = manifest["target_repo"]
    result = subprocess.run(argv, cwd=manifest["target_repo"], env=run_env, text=True, capture_output=True)
    (evidence / "stage.stdout.log").write_text(result.stdout); (evidence / "stage.stderr.log").write_text(result.stderr)
    (evidence / "stage-result.json").write_text(json.dumps({"argv": argv, "returncode": result.returncode}, indent=2) + "\n")
    if result.returncode != 0:
        return 1
    return qualify(manifest_path, evidence)


if __name__ == "__main__":
    raise SystemExit(main())
