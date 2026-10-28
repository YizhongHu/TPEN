"""Stage and submit a reproducible, CPU-only Polaris verification cell."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from common import COPIED_FILES, build_env, copied_hashes, is_ancestor, layout, read_profile


HEX40 = re.compile(r"^[0-9a-f]{40}$")
CELL_ID = re.compile(r"^[A-Za-z0-9._-]+$")
VERIFY_DIR = Path(__file__).resolve().parent


def _run(args: list[str], cwd: Path | None = None, *, capture: bool = False, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def _git(repo: Path, *args: str) -> str:
    return _run(["git", "-C", str(repo), *args], capture=True)


def _safe_arg(value: str, name: str) -> str:
    if not value or "\n" in value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty single-line value")
    return value


def validate_evidence_root(value: str) -> str:
    """Reject PBS -v hazardous evidence-root characters before any work."""
    if not value or any(char.isspace() or char in ",'\"" for char in value):
        raise ValueError("evidence-root must contain no whitespace, comma, or quote")
    return value


def validate_arm_syntax(arm: str) -> str:
    """Validate an arm before cloning or creating a run root."""
    _safe_arg(arm, "arm")
    outside_params = re.sub(r"\[[^\]]*\]", "", arm)
    if arm.startswith("-") or "=" in outside_params:
        raise ValueError("arm must be a pytest path/node id, not an option or override")
    return arm


def validate_arm(arm: str, repo: Path) -> str:
    """Return an arm after refusing overrides and paths outside ``repo``.

    Parameters
    ----------
    arm : str
        Relative pytest path or node id.
    repo : pathlib.Path
        Target checkout that owns the selection.
    """
    _safe_arg(arm, "arm")
    validate_arm_syntax(arm)
    path_part = arm.split("::", 1)[0]
    candidate = (repo / path_part).resolve()
    if repo.resolve() not in candidate.parents and candidate != repo.resolve():
        raise ValueError("arm must resolve inside the target tree")
    if not candidate.exists():
        raise ValueError(f"arm does not exist in target tree: {arm}")
    return arm


def _harness_repo() -> Path:
    """Return the checkout root containing this harness."""
    return Path(_git(VERIFY_DIR, "rev-parse", "--show-toplevel"))


def _write_literal_env(path: Path, manifest: Path, facility_python: str, job_helper: Path | None = None) -> None:
    """Write literal paths for the manifest, Python, and stored job helper."""
    helper = job_helper or manifest.parent / "job.py"
    path.write_text("RUN_MANIFEST=" + shlex.quote(str(manifest)) + "\nRUN_PYTHON=" + shlex.quote(facility_python) + "\nRUN_JOB_PY=" + shlex.quote(str(helper)) + "\n", encoding="utf-8")


def validate_cell_id(cell_id: str) -> str:
    """Validate the cell directory name used below the evidence root."""
    if not CELL_ID.fullmatch(cell_id) or cell_id in {".", ".."}:
        raise ValueError("cell-id must be one safe path component")
    return cell_id


def build_manifest(args: argparse.Namespace, run_root: Path, target_repo: Path, harness_dir: Path) -> dict:
    """Build the single-source manifest after target checkout is verified."""
    target_sha = _git(target_repo, "rev-parse", "HEAD")
    if not HEX40.fullmatch(target_sha) or target_sha != args.target_sha:
        raise RuntimeError("target checkout is not at the requested full SHA")
    harness_repo = _harness_repo()
    harness_sha = _git(harness_repo, "rev-parse", "HEAD")
    paths = layout(run_root)
    profile = read_profile(VERIFY_DIR / "facility_profile.json")
    arms = [validate_arm(arm, target_repo) for arm in args.arm]
    manifest = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cell_id": _safe_arg(args.cell_id, "cell-id"),
        "target_sha": target_sha,
        "harness_sha": harness_sha,
        "required_ancestors": args.required_ancestor,
        "run_root": str(paths["run_root"]),
        "target_repo": str(target_repo),
        "cache_dir": str(paths["cache_dir"]),
        "prime_env": str(paths["prime_env"]),
        "dry_env": str(paths["dry_env"]),
        "evidence_dir": str(paths["evidence_dir"]),
        "pbs_dir": str(paths["pbs_dir"]),
        "arms": arms,
        "profile": profile,
        "profile_path": str(paths["profile"]),
        "job_script": str(paths["job_script"]),
        "job_helper": str(paths["job_helper"]),
        "driver": str(paths["driver"]),
        "staging_receipt": str(paths["staging_receipt"]),
        "copied_file_sha256": {},
    }
    return manifest


def prime_and_dry_sync(target_repo: Path, profile: dict, paths: dict[str, Path], uv: list[str]) -> Path:
    """Prime the cache, then sync a separate fresh dry environment."""
    env = build_env(profile, paths["cache_dir"], paths["prime_env"])
    _run(uv + ["sync", "--extra", "cpu", "--locked", "--project", str(target_repo)], cwd=target_repo, env=env)
    dry_log = paths["run_root"] / "offline-dry.log"
    dry_env = build_env(profile, paths["cache_dir"], paths["dry_env"])
    with dry_log.open("w", encoding="utf-8") as stream:
        subprocess.run(uv + ["sync", "--offline", "--extra", "cpu", "--locked", "--project", str(target_repo)], env=dry_env, check=True, text=True, stdout=stream, stderr=subprocess.STDOUT)
    return dry_log


def stage(args: argparse.Namespace) -> int:
    """Perform login-node staging and submit the stored PBS script.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed target, selection, evidence, and provenance arguments.
    """
    validate_cell_id(args.cell_id)
    validate_evidence_root(args.evidence_root)
    for arm in args.arm:
        validate_arm_syntax(arm)
    harness_repo = _harness_repo()
    if _git(harness_repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("harness checkout has tracked changes; refuse staging")
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    run_root = evidence_root / args.cell_id
    evidence_root.mkdir(parents=True, exist_ok=True)
    initial_qstat = subprocess.run(["qstat", "-u", getpass.getuser()], check=True, text=True, capture_output=True)
    initial_qstat_output = (initial_qstat.stdout + initial_qstat.stderr).strip()
    initial_qstat_receipt = evidence_root / f"{args.cell_id}.pre-submit-qstat.txt"
    initial_qstat_receipt.write_text(initial_qstat_output + ("\n" if initial_qstat_output else ""), encoding="utf-8")
    if initial_qstat_output:
        print(initial_qstat_output, file=sys.stderr)
        raise RuntimeError("refusing staging while the user has a queued or running job")
    if run_root.exists():
        raise RuntimeError(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    for name in ("evidence", "pbs", "harness"):
        (run_root / name).mkdir()
    harness_dir = paths["harness_dir"]
    paths = layout(run_root)
    for name in COPIED_FILES:
        shutil.copy2(VERIFY_DIR / name, harness_dir / name)
    target_repo = run_root / "target"
    _run(["git", "clone", "--no-single-branch", args.remote, str(target_repo)])
    _run(["git", "-C", str(target_repo), "fetch", "origin", "+refs/*:refs/verification-source/*"])
    _run(["git", "-C", str(target_repo), "checkout", "--detach", args.target_sha])
    if _git(target_repo, "rev-parse", "--is-shallow-repository") != "false":
        raise RuntimeError("target clone is shallow")
    if list(target_repo.rglob("._*")):
        raise RuntimeError("AppleDouble files found in target clone")
    arms = [validate_arm(arm, target_repo) for arm in args.arm]
    for ancestor in args.required_ancestor:
        if not is_ancestor(target_repo, ancestor, args.target_sha):
            raise RuntimeError(f"required ancestor is not an ancestor of target: {ancestor}")
    profile = read_profile(VERIFY_DIR / "facility_profile.json")
    uv = [profile["facility_python"], "-m", "uv"]
    uv_version = _run(uv + ["--version"], capture=True)
    if uv_version.split()[:2] != ["uv", profile["uv_version"]]:
        raise RuntimeError(f"facility uv version mismatch: expected {profile['uv_version']}, got {uv_version}")
    dry_log = prime_and_dry_sync(target_repo, profile, paths, uv)
    manifest = build_manifest(args, run_root, target_repo, harness_dir)
    manifest["harness_clean_tracked"] = True
    manifest["copied_file_sha256"] = copied_hashes(paths["copied_files"])
    manifest["arms"] = arms
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env_path = run_root / "run.env"
    _write_literal_env(env_path, manifest_path, profile["facility_python"], harness_dir / "job.py")
    receipt = {"target_sha": args.target_sha, "cache_dir": manifest["cache_dir"], "manifest": str(manifest_path), "offline_dry_log": str(dry_log), "dry_env": manifest["dry_env"], "facility_python": profile["facility_python"]}
    Path(manifest["staging_receipt"]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in (manifest["evidence_dir"], manifest["pbs_dir"], manifest["cache_dir"], manifest["prime_env"], manifest["dry_env"], manifest["staging_receipt"]):
        if not Path(path).exists():
            raise RuntimeError(f"staging path missing: {path}")
    final_qstat = subprocess.run(["qstat", "-u", getpass.getuser()], check=True, text=True, capture_output=True)
    final_qstat_output = (final_qstat.stdout + final_qstat.stderr).strip()
    (run_root / "pre-submit-qstat.txt").write_text(final_qstat_output + ("\n" if final_qstat_output else ""), encoding="utf-8")
    if final_qstat_output:
        print(final_qstat_output, file=sys.stderr)
        raise RuntimeError("refusing qsub because another job appeared during staging")
    qsub = ["qsub", "-A", profile["pbs"]["account"], "-q", profile["pbs"]["queue"], "-l", f"select={profile['pbs']['select']}", "-l", f"walltime={profile['pbs']['walltime']}", "-l", f"filesystems={profile['pbs']['filesystems']}", "-o", str(manifest["pbs_dir"]) + "/", "-e", str(manifest["pbs_dir"]) + "/", "-v", f"RUN_ENV_PATH={env_path}", str(paths["job_script"])]
    print("SUBMIT " + shlex.join(qsub), flush=True)
    job_id = _run(qsub, capture=True)
    (run_root / "submission.txt").write_text(job_id + "\n" + shlex.join(qsub) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--required-ancestor", action="append", default=[])
    parser.add_argument("--remote", default="https://github.com/YizhongHu/TPEN.git")
    args = parser.parse_args()
    if not HEX40.fullmatch(args.target_sha):
        parser.error("--target-sha must be a full 40-hex SHA")
    if any(not HEX40.fullmatch(value) for value in args.required_ancestor):
        parser.error("--required-ancestor values must be full 40-hex SHAs")
    return stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
