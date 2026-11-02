"""Shared layout and environment construction for the Polaris harness."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


COPIED_FILES = ("facility_profile.json", "job.pbs", "job.py", "driver.py", "common.py")


def layout(run_root: Path) -> dict[str, Path]:
    """Return every derived run path from one run root."""
    harness = run_root / "harness"
    return {
        "run_root": run_root,
        "cache_dir": run_root / "uv-cache",
        "prime_env": run_root / "prime-env",
        "dry_env": run_root / "dry-env",
        "evidence_dir": run_root / "evidence",
        "pbs_dir": run_root / "pbs",
        "harness_dir": harness,
        "job_script": harness / "job.pbs",
        "job_helper": harness / "job.py",
        "driver": harness / "driver.py",
        "profile": harness / "facility_profile.json",
        "common": harness / "common.py",
        "copied_files": [harness / name for name in COPIED_FILES],
        "staging_receipt": run_root / "staging-receipt.json",
    }


def build_env(profile: dict, cache_dir: Path, project_env: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Build the facility-pinned uv and test environment."""
    env = dict(os.environ if base is None else base)
    env.update(dict(item.split("=", 1) for item in profile["uv_flags"]))
    env.update(profile.get("thread_limits", {}))
    env["TZ"] = profile["timezone"]
    env["UV_PYTHON"] = profile["facility_python"]
    env["UV_CACHE_DIR"] = str(cache_dir)
    env["UV_PROJECT_ENVIRONMENT"] = str(project_env)
    return env


def copied_hashes(paths: list[Path]) -> dict[str, str]:
    """Hash copied harness files for the manifest identity check."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def read_profile(path: Path) -> dict:
    """Read one facility profile JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    """Return whether ``ancestor`` reaches ``descendant`` in a git checkout."""
    return subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    ).returncode == 0
