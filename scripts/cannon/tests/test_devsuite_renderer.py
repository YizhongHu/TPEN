"""Local, no-SSH contract tests for the devsuite renderer."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "scripts" / "cannon" / "devsuite" / "submit_devsuite.sh"
SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def render(*, scheduled: bool) -> subprocess.CompletedProcess[str]:
    args = [str(LAUNCHER), "--local-dry-run"]
    if scheduled:
        args.append("--scheduled")
    args.append(SHA)
    return subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def test_local_render_uses_exact_sha_harness() -> None:
    result = render(scheduled=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "harness_source=scripts/cannon/executor-harness" in result.stdout
    assert f"harness_source_sha={SHA}" in result.stdout


def test_scheduled_render_carries_marker() -> None:
    result = render(scheduled=True)
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r" stage=(\S+)", result.stdout)
    assert match, result.stdout
    assert "/dev-fullsuite-scheduled-" in match.group(1)


def test_manual_render_has_no_marker() -> None:
    result = render(scheduled=False)
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r" stage=(\S+)", result.stdout)
    assert match, result.stdout
    assert "/dev-fullsuite-scheduled-" not in match.group(1)
