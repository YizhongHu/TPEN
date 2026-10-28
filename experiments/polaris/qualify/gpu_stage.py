"""Stage one real Stage-Q GPU qualification cell for Polaris.

The stage manifest is the only source for payload and scheduler values.  This
module performs login-node checks and captures the exact stored PBS submission;
the allocation-local job owns runtime and artifact verification.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

VERIFY = Path(__file__).resolve().parents[1] / "verify"
if str(VERIFY) not in sys.path:
    sys.path.insert(0, str(VERIFY))
from common import COPIED_FILES, copied_hashes, is_ancestor, layout, read_profile  # noqa: E402


RUNTIME_KIND = "polaris-gpu-qualification"
ALLOWED_WALKERS = {2048, 4096}
STORED_FILES = tuple(COPIED_FILES) + ("gpu_stage.py", "gpu_job.py", "qualification.py", "job.pbs")


def _run(args: list[str], *, cwd: Path | None = None, capture: bool = False, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
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


def _profile(path: Path) -> dict:
    profile = read_profile(path)
    required = ("overlay_python", "overlay_root", "facility_site_packages", "libshim", "cuda13_lib", "cuda129_lib", "provenance")
    missing = [key for key in required if not profile.get(key)]
    if missing:
        raise ValueError(f"runtime profile missing required fields: {missing}")
    if profile.get("runtime_kind") != RUNTIME_KIND:
        raise ValueError("runtime profile has the wrong runtime_kind")
    return profile


def _qstat() -> str:
    result = subprocess.run(["qstat", "-u", getpass.getuser()], check=True, text=True, capture_output=True)
    return (result.stdout + result.stderr).strip()


def _validate(args: argparse.Namespace) -> None:
    if len(args.target_sha) != 40 or any(char not in "0123456789abcdef" for char in args.target_sha):
        raise ValueError("target-sha must be a full lowercase 40-hex SHA")
    if args.walkers not in ALLOWED_WALKERS:
        raise ValueError("B0 admits only 2048 or 4096 walkers")
    if args.updates != 2000 or args.checkpoint_every != 1000:
        raise ValueError("B0 uses the current Stage-Q horizon 2000 and cadence 1000")
    if args.nodes != 1:
        raise ValueError("GPU qualification currently supports exactly one node")
    _safe_component(args.cell_id, "cell-id")


def build_manifest(args: argparse.Namespace, run_root: Path, target_repo: Path, profile: dict, files: list[Path]) -> dict:
    """Build the immutable runtime-kind-bound manifest."""
    target_sha = _git(target_repo, "rev-parse", "HEAD")
    if target_sha != args.target_sha:
        raise RuntimeError("target checkout is not at the requested full SHA")
    paths = layout(run_root)
    paths.update({
        "overlay_receipt": run_root / "overlay-receipt.json",
        "runtime_profile": run_root / "runtime-profile.json",
        "selection_argv": run_root / "selection-argv.json",
    })
    return {
        "schema": 1,
        "runtime_kind": RUNTIME_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_sha": target_sha,
        "harness_sha": _git(Path(__file__).resolve().parents[3], "rev-parse", "HEAD"),
        "required_ancestors": list(args.required_ancestor),
        "cell": {"stage": args.stage, "context": args.context, "optimizer": args.optimizer,
                 "lr": args.lr, "beta2": args.beta2, "seed_label": args.seed_label,
                 "walkers": args.walkers, "updates": args.updates,
                 "checkpoint_every": args.checkpoint_every},
        "nodes": args.nodes,
        "run_root": str(run_root),
        "target_repo": str(target_repo),
        "output_root": str(run_root / "output"),
        "evidence_dir": str(paths["evidence_dir"]),
        "pbs_dir": str(paths["pbs_dir"]),
        "job_script": str(paths["job_script"]),
        "job_helper": str(paths["job_helper"]),
        "profile_path": str(paths["runtime_profile"]),
        "runtime_profile": profile,
        "copied_file_sha256": copied_hashes(files),
        "staging_receipt": str(run_root / "staging-receipt.json"),
    }


def stage(args: argparse.Namespace) -> int:
    """Clone, stage, and submit one manifest-driven GPU qualification cell."""
    _validate(args)
    profile = _profile(Path(args.runtime_profile).expanduser().resolve())
    harness = Path(__file__).resolve().parent
    harness_repo = Path(_git(harness, "rev-parse", "--show-toplevel"))
    if _git(harness_repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("harness checkout has tracked changes")
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    run_root = evidence_root / args.cell_id
    if run_root.exists():
        raise RuntimeError(f"run root already exists: {run_root}")
    first_qstat = _qstat()
    if first_qstat:
        raise RuntimeError("refusing staging while qstat is non-empty")
    run_root.mkdir(parents=True)
    paths = layout(run_root)
    for key in ("evidence_dir", "pbs_dir", "harness_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    copied = [paths["harness_dir"] / name for name in STORED_FILES]
    for name, destination in zip(COPIED_FILES, copied):
        shutil.copy2(VERIFY / name, destination)
    for name in STORED_FILES[len(COPIED_FILES):]:
        shutil.copy2(harness / name, paths["harness_dir"] / name)
    target_repo = run_root / "target"
    _run(["git", "clone", "--no-single-branch", args.remote, str(target_repo)])
    _run(["git", "-C", str(target_repo), "fetch", "origin", "+refs/*:refs/verification-source/*"])
    _run(["git", "-C", str(target_repo), "checkout", "--detach", args.target_sha])
    if _git(target_repo, "rev-parse", "--is-shallow-repository") != "false":
        raise RuntimeError("target clone is shallow")
    for ancestor in args.required_ancestor:
        if not is_ancestor(target_repo, ancestor, args.target_sha):
            raise RuntimeError(f"required ancestor is not an ancestor: {ancestor}")
    manifest = build_manifest(args, run_root, target_repo, profile, copied)
    manifest_path = run_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    _atomic_json(paths["runtime_profile"], profile)
    env_path = run_root / "run.env"
    env_path.write_text("RUN_MANIFEST=" + shlex.quote(str(manifest_path)) + "\nRUN_PYTHON=" + shlex.quote(profile["overlay_python"]) + "\nRUN_JOB_PY=" + shlex.quote(str(paths["harness_dir"] / "gpu_job.py")) + "\n", encoding="utf-8")
    _atomic_json(Path(manifest["staging_receipt"]), {"runtime_kind": RUNTIME_KIND, "target_sha": args.target_sha, "overlay_python": profile["overlay_python"], "manifest": str(manifest_path)})
    for key in ("evidence_dir", "pbs_dir", "harness_dir"):
        if not paths[key].is_dir():
            raise RuntimeError(f"missing staged directory: {paths[key]}")
    second_qstat = _qstat()
    (run_root / "pre-submit-qstat.txt").write_text(second_qstat + ("\n" if second_qstat else ""), encoding="utf-8")
    if second_qstat:
        raise RuntimeError("refusing qsub because qstat became non-empty")
    qsub = ["qsub", "-V", "-A", args.account, "-q", args.queue,
            "-l", f"select={args.nodes}:system=polaris", "-l", f"walltime={args.walltime}",
            "-l", "filesystems=home:eagle", "-o", str(paths["pbs_dir"]) + "/",
            "-e", str(paths["pbs_dir"]) + "/", "-v", f"RUN_ENV_PATH={env_path}", str(paths["harness_dir"] / "job.pbs")]
    print("SUBMIT " + shlex.join(qsub), flush=True)
    job_id = _run(qsub, capture=True)
    (run_root / "submission.txt").write_text(job_id + "\n" + shlex.join(qsub) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-sha", required=True); parser.add_argument("--cell-id", required=True)
    parser.add_argument("--context", required=True); parser.add_argument("--optimizer", required=True)
    parser.add_argument("--lr", type=float, required=True); parser.add_argument("--beta2", type=float, required=True)
    parser.add_argument("--seed-label", type=int, required=True); parser.add_argument("--walkers", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True); parser.add_argument("--checkpoint-every", type=int, required=True)
    parser.add_argument("--stage", required=True); parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--queue", required=True); parser.add_argument("--walltime", required=True)
    parser.add_argument("--account", required=True); parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--evidence-root", required=True); parser.add_argument("--required-ancestor", action="append", default=[])
    parser.add_argument("--remote", default="https://github.com/YizhongHu/TPEN.git")
    return stage(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
