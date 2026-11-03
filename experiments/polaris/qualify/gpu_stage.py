"""Stage one tracked-lock GPU qualification cell for Polaris."""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

VERIFY = Path(__file__).resolve().parents[1] / "verify"
QUALIFY = Path(__file__).resolve().parent
if str(QUALIFY) not in sys.path:
    sys.path.insert(0, str(QUALIFY))
if str(VERIFY) not in sys.path:
    sys.path.insert(0, str(VERIFY))
from common import build_env, copied_hashes, is_ancestor, layout, read_profile  # noqa: E402
from stage import _queue_gate  # noqa: E402
from lock import expected_torch_version  # noqa: E402

RUNTIME_KIND = "polaris-gpu-qualification"
ALLOWED_WALKERS = {2048, 4096}
STORED_FILES = ("common.py", "lock.py", "gpu_job.py", "qualification.py", "job.pbs", "facility_profile.json")
PROFILE_PATH = Path(__file__).resolve().parent / "facility_profile.json"


def _run(args: list[str], *, cwd: Path | None = None, capture: bool = False, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, check=True, text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def _git(repo: Path, *args: str) -> str:
    return _run(["git", "-C", str(repo), *args], capture=True)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_component(value: str, name: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value or any(char.isspace() for char in value):
        raise ValueError(f"{name} must be one safe path component")
    return value


def _profile(path: Path | None = None) -> dict:
    path = PROFILE_PATH if path is None else path
    profile = read_profile(path)
    if profile.get("runtime_kind") != RUNTIME_KIND or not isinstance(profile.get("extra"), str) or not profile["extra"]:
        raise ValueError("qualification profile must admit a runtime_kind and non-empty extra")
    return profile


def _harness_repo() -> Path:
    return Path(_git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel"))


def gpu_layout(run_root: Path) -> dict[str, Path]:
    paths = layout(run_root)
    paths.update({"runtime_profile": run_root / "runtime-profile.json", "selection_argv": run_root / "selection-argv.json", "job_helper": run_root / "harness" / "gpu_job.py"})
    return paths


def validate_evidence_root(value: str) -> str:
    if not value or any(char.isspace() or char in ",'\"" for char in value):
        raise ValueError("evidence-root contains unsafe whitespace, comma, or quote")
    return value


def validate_queue(queue: str) -> str:
    profile = read_profile(VERIFY / "facility_profile.json")
    allowed = profile.get("pbs", {}).get("allowed_queues")
    if not isinstance(allowed, list) or queue not in allowed:
        raise ValueError(f"queue is not allowed by the facility profile: {queue}")
    return queue


def _validate(args: argparse.Namespace) -> None:
    if len(args.target_sha) != 40 or any(char not in "0123456789abcdef" for char in args.target_sha):
        raise ValueError("target-sha must be a full lowercase 40-hex SHA")
    if args.walkers not in ALLOWED_WALKERS or args.updates != 2000 or args.checkpoint_every != 1000:
        raise ValueError("B0 admits 2048/4096 walkers, 2000 updates, and 1000-update checkpoints")
    if args.nodes != 1:
        raise ValueError("GPU qualification currently supports exactly one node")
    _safe_component(args.cell_id, "cell-id")


def build_manifest(args: argparse.Namespace, run_root: Path, target_repo: Path, profile: dict, files: list[Path]) -> dict:
    target_sha = _git(target_repo, "rev-parse", "HEAD")
    if target_sha != args.target_sha:
        raise RuntimeError("target checkout is not at the requested SHA")
    paths = gpu_layout(run_root)
    verify_profile = read_profile(VERIFY / "facility_profile.json")
    lock = tomllib.loads((target_repo / "uv.lock").read_text(encoding="utf-8"))
    expected = expected_torch_version(lock, profile["extra"])
    return {"schema": 1, "runtime_kind": RUNTIME_KIND, "created_utc": datetime.now(timezone.utc).isoformat(), "target_sha": target_sha,
            "harness_sha": _git(Path(__file__).resolve().parents[3], "rev-parse", "HEAD"), "required_ancestors": list(args.required_ancestor),
            "cell": {"stage": args.stage, "context": args.context, "optimizer": args.optimizer, "lr": args.lr, "beta2": args.beta2,
                     "seed_label": args.seed_label, "walkers": args.walkers, "updates": args.updates, "checkpoint_every": args.checkpoint_every},
            "nodes": args.nodes, "scheduler": {"account": args.account, "queue": args.queue, "walltime": args.walltime, "filesystems": "home:eagle"},
            "run_root": str(run_root), "target_repo": str(target_repo), "output_root": str(run_root / "output"), "evidence_dir": str(paths["evidence_dir"]),
            "pbs_dir": str(paths["pbs_dir"]), "job_script": str(paths["job_script"]), "job_helper": str(paths["job_helper"]),
            "profile_path": str(paths["runtime_profile"]), "runtime_profile": profile, "verify_profile": verify_profile,
            "extra": profile["extra"], "expected_torch": expected, "cache_dir": str(paths["cache_dir"]),
            "prime_env": str(paths["prime_env"]), "dry_env": str(paths["dry_env"]), "copied_file_sha256": copied_hashes(files),
            "staging_receipt": str(run_root / "staging-receipt.json")}


def prime_and_dry_sync(target_repo: Path, profile: dict, paths: dict[str, Path], uv: list[str]) -> None:
    verify_profile = read_profile(VERIFY / "facility_profile.json")
    env = build_env(verify_profile, paths["cache_dir"], paths["prime_env"])
    _run(uv + ["sync", "--extra", profile["extra"], "--locked", "--project", str(target_repo)], cwd=target_repo, env=env)
    dry_env = build_env(verify_profile, paths["cache_dir"], paths["dry_env"])
    with (paths["run_root"] / "offline-dry.log").open("w", encoding="utf-8") as stream:
        subprocess.run(uv + ["sync", "--offline", "--extra", profile["extra"], "--locked", "--project", str(target_repo)], env=dry_env, check=True, text=True, stdout=stream, stderr=subprocess.STDOUT)


def stage(args: argparse.Namespace) -> int:
    _validate(args); validate_evidence_root(args.evidence_root); validate_queue(args.queue)
    profile = _profile(); verify_profile = read_profile(VERIFY / "facility_profile.json")
    harness = Path(__file__).resolve().parent; harness_repo = _harness_repo()
    if _git(harness_repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("harness checkout has tracked changes")
    evidence_root = Path(args.evidence_root).expanduser().resolve(); run_root = evidence_root / args.cell_id
    if run_root.exists(): raise RuntimeError(f"run root already exists: {run_root}")
    if not _queue_gate(args.queue, getpass.getuser(), evidence_root, f"{args.cell_id}.pre-submit"):
        raise RuntimeError("refusing staging because the queue-aware gate refused admission")
    run_root.mkdir(parents=True); paths = gpu_layout(run_root); (run_root / "output").mkdir(parents=True); paths["evidence_dir"].mkdir(); paths["pbs_dir"].mkdir(); paths["harness_dir"].mkdir()
    copied = [paths["harness_dir"] / name for name in STORED_FILES]
    for name, destination in zip(STORED_FILES, copied): shutil.copy2(VERIFY / name if name == "common.py" else harness / name, destination)
    target_repo = run_root / "target"
    _run(["git", "clone", "--no-single-branch", args.remote, str(target_repo)]); _run(["git", "-C", str(target_repo), "fetch", "origin", "+refs/*:refs/verification-source/*"]); _run(["git", "-C", str(target_repo), "checkout", "--detach", args.target_sha])
    if _git(target_repo, "rev-parse", "--is-shallow-repository") != "false" or list(target_repo.rglob("._*")): raise RuntimeError("target clone failed identity preflight")
    for ancestor in args.required_ancestor:
        if not is_ancestor(target_repo, ancestor, args.target_sha): raise RuntimeError(f"required ancestor is not an ancestor: {ancestor}")
    manifest = build_manifest(args, run_root, target_repo, profile, copied); manifest_path = run_root / "manifest.json"; _atomic_json(manifest_path, manifest); _atomic_json(paths["runtime_profile"], profile)
    uv = [verify_profile["facility_python"], "-m", "uv"]
    if _run(uv + ["--version"], capture=True).split()[:2] != ["uv", verify_profile["uv_version"]]: raise RuntimeError("facility uv version mismatch")
    prime_and_dry_sync(target_repo, profile, paths, uv)
    env_path = run_root / "run.env"; env_path.write_text(
        "RUN_MANIFEST=" + shlex.quote(str(manifest_path)) + "\n"
        + "RUN_PYTHON=" + shlex.quote(verify_profile["facility_python"]) + "\n"
        + "RUN_JOB_PY=" + shlex.quote(str(paths["harness_dir"] / "gpu_job.py")) + "\n"
        + "RUN_PROJECT=" + shlex.quote(str(target_repo)) + "\n"
        + "RUN_CACHE_DIR=" + shlex.quote(str(paths["cache_dir"])) + "\n",
        encoding="utf-8")
    _atomic_json(Path(manifest["staging_receipt"]), {"runtime_kind": RUNTIME_KIND, "target_sha": args.target_sha, "cache_dir": manifest["cache_dir"], "extra": profile["extra"], "manifest": str(manifest_path)})
    if not _queue_gate(args.queue, getpass.getuser(), run_root, "pre-submit"): raise RuntimeError("refusing qsub because the queue-aware gate refused admission")
    scheduler = manifest["scheduler"]; qsub = ["qsub", "-V", "-A", scheduler["account"], "-q", scheduler["queue"], "-l", f"select={manifest['nodes']}:system=polaris", "-l", f"walltime={scheduler['walltime']}", "-l", f"filesystems={scheduler['filesystems']}", "-o", str(paths["pbs_dir"]) + "/", "-e", str(paths["pbs_dir"]) + "/", "-v", f"RUN_ENV_PATH={env_path}", str(paths["harness_dir"] / "job.pbs")]
    print("SUBMIT " + shlex.join(qsub), flush=True); job_id = _run(qsub, capture=True); (run_root / "submission.txt").write_text(job_id + "\n" + shlex.join(qsub) + "\n", encoding="utf-8"); return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target-sha", "cell-id", "context", "optimizer", "stage", "queue", "walltime", "account", "evidence-root"): parser.add_argument("--" + name, required=True)
    parser.add_argument("--lr", type=float, required=True); parser.add_argument("--beta2", type=float, required=True); parser.add_argument("--seed-label", type=int, required=True); parser.add_argument("--walkers", type=int, required=True); parser.add_argument("--updates", type=int, required=True); parser.add_argument("--checkpoint-every", type=int, required=True); parser.add_argument("--nodes", type=int, required=True); parser.add_argument("--required-ancestor", action="append", default=[]); parser.add_argument("--remote", default="https://github.com/YizhongHu/TPEN.git")
    return stage(parser.parse_args(argv))


if __name__ == "__main__": raise SystemExit(main())
