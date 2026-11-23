"""Exercise the executor harness JOB_NAME precheck without Slurm."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
JOB_SCRIPT = REPO_ROOT / "scripts" / "cannon" / "executor-harness" / "job.sbatch"
HEAD_SHA = "0123456789abcdef0123456789abcdef01234567"


def _job_name_guards() -> str:
    lines = JOB_SCRIPT.read_text(encoding="utf-8").splitlines()
    guards = [
        line
        for line in lines
        if '[[ "$JOB_NAME" =~' in line or '[[ "$JOB_NAME" ==' in line
    ]
    assert len(guards) == 2
    assert "${HEAD_SHA:0:8}" in guards[1]
    return "\n".join(guards)


def _run_guard(job_name: str) -> subprocess.CompletedProcess[str]:
    script = f"set -Eeuo pipefail\nvalidate_job_name() {{\n{_job_name_guards()}\n}}\nvalidate_job_name\n"
    return subprocess.run(
        ["bash", "-c", script],
        env={"HEAD_SHA": HEAD_SHA, "JOB_NAME": job_name, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_job_name_guard_accepts_sha8_and_rejects_missing_sha_or_bad_characters() -> None:
    accepted = _run_guard(f"executor-verify-ex-{HEAD_SHA[:8]}")
    missing_sha = _run_guard("executor-verify-ex")
    bad_characters = _run_guard(f"executor verify ex {HEAD_SHA[:8]}")

    assert accepted.returncode == 0, accepted.stderr
    assert missing_sha.returncode != 0
    assert bad_characters.returncode != 0
