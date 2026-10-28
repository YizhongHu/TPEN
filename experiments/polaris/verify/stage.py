"""Stage and submit a reproducible, CPU-only Polaris verification cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone


HEX40 = re.compile(r"^[0-9a-f]{40}$")
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
    if "=" in arm or arm.startswith("-"):
        raise ValueError("pytest overrides and options are not valid arms")
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


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    ).returncode == 0


def _write_literal_env(path: Path, manifest: Path, facility_python: str) -> None:
    """Write only the manifest pointer; shared values remain in JSON."""
    path.write_text("RUN_MANIFEST=" + shlex.quote(str(manifest)) + "\nRUN_PYTHON=" + shlex.quote(facility_python) + "\n", encoding="utf-8")


def build_manifest(args: argparse.Namespace, run_root: Path, target_repo: Path, harness_dir: Path) -> dict:
    """Build the single-source manifest after target checkout is verified."""
    target_sha = _git(target_repo, "rev-parse", "HEAD")
    if not HEX40.fullmatch(target_sha) or target_sha != args.target_sha:
        raise RuntimeError("target checkout is not at the requested full SHA")
    harness_repo = _harness_repo()
    harness_sha = _git(harness_repo, "rev-parse", "HEAD")
    profile = json.loads((VERIFY_DIR / "facility_profile.json").read_text())
    arms = [validate_arm(arm, target_repo) for arm in args.arm]
    manifest = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cell_id": _safe_arg(args.cell_id, "cell-id"),
        "target_sha": target_sha,
        "harness_sha": harness_sha,
        "required_ancestors": args.required_ancestor,
        "run_root": str(run_root),
        "target_repo": str(target_repo),
        "cache_dir": str(run_root / "uv-cache"),
        "prime_env": str(run_root / "prime-env"),
        "evidence_dir": str(run_root / "evidence"),
        "pbs_dir": str(run_root / "pbs"),
        "arms": arms,
        "profile": profile,
        "job_script": str(harness_dir / "job.pbs"),
        "job_helper": str(harness_dir / "job.py"),
        "driver": str(harness_dir / "driver.py"),
        "staging_receipt": str(run_root / "staging-receipt.json"),
        "copied_file_sha256": {},
    }
    return manifest


def stage(args: argparse.Namespace) -> int:
    """Perform login-node staging and submit the stored PBS script.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed target, selection, evidence, and provenance arguments.
    """
    harness_repo = _harness_repo()
    if _git(harness_repo, "status", "--porcelain"):
        raise RuntimeError("harness checkout has tracked changes; refuse staging")
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    run_root = evidence_root / args.cell_id
    if run_root.exists():
        raise RuntimeError(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    for name in ("evidence", "pbs", "harness"):
        (run_root / name).mkdir()
    qstat_path = run_root / "pre-submit-qstat.txt"
    qstat = subprocess.run(["qstat", "-u", os.environ.get("USER", "")], check=True, text=True, capture_output=True)
    qstat_output = (qstat.stdout + qstat.stderr).strip()
    qstat_path.write_text(qstat_output + ("\n" if qstat_output else ""), encoding="utf-8")
    if qstat_output:
        print(qstat_output, file=sys.stderr)
        raise RuntimeError("refusing qsub while the user has a queued or running job")
    harness_dir = run_root / "harness"
    for name in ("facility_profile.json", "job.pbs", "job.py", "driver.py"):
        shutil.copy2(VERIFY_DIR / name, harness_dir / name)
    target_repo = run_root / "target"
    _run(["git", "clone", "--no-single-branch", args.remote, str(target_repo)])
    _run(["git", "-C", str(target_repo), "fetch", "origin", "+refs/*:refs/verification-source/*"])
    _run(["git", "-C", str(target_repo), "checkout", "--detach", args.target_sha])
    if _git(target_repo, "rev-parse", "--is-shallow-repository") != "false":
        raise RuntimeError("target clone is shallow")
    if list(target_repo.rglob("._*")):
        raise RuntimeError("AppleDouble files found in target clone")
    for ancestor in args.required_ancestor:
        if not _is_ancestor(target_repo, ancestor, args.target_sha):
            raise RuntimeError(f"required ancestor is not an ancestor of target: {ancestor}")
    profile = json.loads((VERIFY_DIR / "facility_profile.json").read_text())
    env = os.environ.copy()
    env.update({key: value for key, value in (item.split("=", 1) for item in profile["uv_flags"])})
    env["UV_CACHE_DIR"] = str(run_root / "uv-cache")
    env["UV_PROJECT_ENVIRONMENT"] = str(run_root / "prime-env")
    uv = [profile["facility_python"], "-m", "uv"]
    uv_version = _run(uv + ["--version"], capture=True)
    if uv_version != "uv " + profile["uv_version"]:
        raise RuntimeError(f"facility uv version mismatch: expected {profile['uv_version']}, got {uv_version}")
    _run(uv + ["sync", "--extra", "cpu", "--locked", "--project", str(target_repo)], cwd=target_repo, env=env)
    dry_log = run_root / "offline-dry.log"
    with dry_log.open("w", encoding="utf-8") as stream:
        subprocess.run(uv + ["sync", "--offline", "--extra", "cpu", "--locked", "--project", str(target_repo)], env=env, check=True, text=True, stdout=stream, stderr=subprocess.STDOUT)
    manifest = build_manifest(args, run_root, target_repo, harness_dir)
    manifest["harness_clean_tracked"] = True
    manifest["copied_file_sha256"] = {
        name: hashlib.sha256((harness_dir / name).read_bytes()).hexdigest()
        for name in ("facility_profile.json", "job.pbs", "job.py", "driver.py")
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env_path = run_root / "run.env"
    _write_literal_env(env_path, manifest_path, profile["facility_python"])
    receipt = {"target_sha": args.target_sha, "cache_dir": manifest["cache_dir"], "manifest": str(manifest_path), "offline_dry_log": str(dry_log), "facility_python": profile["facility_python"]}
    Path(manifest["staging_receipt"]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in (manifest["evidence_dir"], manifest["pbs_dir"], manifest["cache_dir"], manifest["prime_env"], manifest["staging_receipt"]):
        if not Path(path).exists():
            raise RuntimeError(f"staging path missing: {path}")
    qsub = ["qsub", "-A", profile["pbs"]["account"], "-q", profile["pbs"]["queue"], "-l", f"select={profile['pbs']['select']},walltime={profile['pbs']['walltime']},filesystems={profile['pbs']['filesystems']}", "-k", "doe", "-o", str(manifest["pbs_dir"]) + "/", "-e", str(manifest["pbs_dir"]) + "/", "-v", f"RUN_ENV_PATH={env_path}", str(harness_dir / "job.pbs")]
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
