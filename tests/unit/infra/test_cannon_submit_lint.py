"""Contract tests for ``scripts/cannon/submit-lint.sh``.

Each test drives the linter as a subprocess against a small fixture tree
built under ``tmp_path`` — no Slurm, no cluster access, bash/stdlib only
(mirrors ``scripts/cannon/submit-lint-selftest.sh``, which exercises the same
checks for a local, pytest-free regression run).
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

LINTER = Path(__file__).resolve().parents[3] / "scripts" / "cannon" / "submit-lint.sh"
SHA = "1111111111111111111111111111111111111111"
SHORT_SHA = SHA[:12]
JOB_NAME = f"tpen-verify-{SHORT_SHA}"


@pytest.fixture()
def fixture_root(tmp_path: Path) -> Path:
    base = tmp_path.resolve()
    (base / "stage").mkdir()
    (base / "other-root").mkdir()
    (base / "home").mkdir()
    (base / "shared-env" / "bin").mkdir(parents=True)
    python = base / "shared-env" / "bin" / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    return base


def write_manifest(path: Path, *, sha: str, job_name: str, task_root: Path, run_root: Path, dry_run: int, python: Path) -> None:
    path.write_text(
        f"HEAD_SHA={sha}\nJOB_NAME={job_name}\nTASK_ROOT={task_root}\n"
        f"RUN_ROOT={run_root}\nDRY_RUN={dry_run}\nPYTHON={python}\n"
    )


def write_sbatch(path: Path, *, sha: str, job_name: str, task_root: Path, run_root: Path, dry_run: int, python: Path) -> None:
    path.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        f"export HEAD_SHA={sha}\nexport JOB_NAME={job_name}\nexport TASK_ROOT={task_root}\n"
        f"export RUN_ROOT={run_root}\nexport DRY_RUN={dry_run}\nexport PYTHON={python}\n"
    )


def good_fixture(base: Path) -> tuple[Path, Path, Path, Path, Path]:
    task_root = base / "stage" / "task"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "good.manifest"
    sbatch = base / "stage" / "good.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    write_sbatch(sbatch, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    return sbatch, manifest, task_root, run_root, python


def run_lint(base: Path, sbatch: Path, manifest: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        "bash",
        str(LINTER),
        "--sbatch",
        str(sbatch),
        "--manifest",
        str(manifest),
        "--allowed-root",
        str(base / "stage"),
        "--shared-env-root",
        str(base / "shared-env"),
        "--home",
        str(base / "home"),
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_good_fixture_passes(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    result = run_lint(fixture_root, sbatch, manifest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LINT_PASS" in result.stdout


def test_non_hex_sha_rejected(fixture_root: Path) -> None:
    sbatch, manifest, task_root, run_root, python = good_fixture(fixture_root)
    bad_manifest = fixture_root / "stage" / "bad.manifest"
    write_manifest(bad_manifest, sha="not-a-sha", job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    result = run_lint(fixture_root, sbatch, bad_manifest)
    assert result.returncode != 0
    assert "40-hex" in result.stdout


def test_job_name_missing_short_sha_rejected(fixture_root: Path) -> None:
    """Killer for mutant omit-job-sha-check: disabling the JOB_NAME/SHA[:12] check lets this through."""
    base = fixture_root
    task_root = base / "stage" / "task"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    job_name = "tpen-verify-nosha"
    manifest = base / "stage" / "bad2.manifest"
    sbatch = base / "stage" / "bad2.sbatch"
    write_manifest(manifest, sha=SHA, job_name=job_name, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    write_sbatch(sbatch, sha=SHA, job_name=job_name, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "does not contain SHA" in result.stdout


def test_relative_sbatch_path_rejected(fixture_root: Path) -> None:
    """Killer for mutant accept-relative-sbatch: disabling the absolute-path check lets this through."""
    sbatch, manifest, *_ = good_fixture(fixture_root)
    cmd = [
        "bash",
        str(LINTER),
        "--sbatch",
        os.path.relpath(sbatch, fixture_root),
        "--manifest",
        str(manifest),
        "--allowed-root",
        str(fixture_root / "stage"),
        "--shared-env-root",
        str(fixture_root / "shared-env"),
        "--home",
        str(fixture_root / "home"),
    ]
    result = subprocess.run(cmd, cwd=fixture_root, capture_output=True, text=True, check=False)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "not absolute" in result.stdout


def test_sbatch_path_outside_allowed_root_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    outside = fixture_root / "outside"
    outside.mkdir()
    outside_sbatch = outside / "good.sbatch"
    outside_sbatch.write_text(sbatch.read_text())
    result = run_lint(fixture_root, outside_sbatch, manifest)
    assert result.returncode != 0
    assert "escapes every declared stage root" in result.stdout


def test_stale_critical_export_rejected(fixture_root: Path) -> None:
    base = fixture_root
    task_root = base / "stage" / "task"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "good.manifest"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    stale_sbatch = base / "stage" / "stale.sbatch"
    write_sbatch(stale_sbatch, sha=SHA, job_name=JOB_NAME, task_root=base / "stage" / "task-stale", run_root=run_root, dry_run=0, python=python)
    result = run_lint(base, stale_sbatch, manifest)
    assert result.returncode != 0
    assert "stale" in result.stdout


def test_missing_critical_export_rejected(fixture_root: Path) -> None:
    base = fixture_root
    _, manifest, task_root, run_root, python = good_fixture(base)
    sbatch = base / "stage" / "missing.sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f"export HEAD_SHA={SHA}\nexport JOB_NAME={JOB_NAME}\n"
        f"export RUN_ROOT={run_root}\nexport DRY_RUN=0\nexport PYTHON={python}\n"
    )
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0
    assert "missing from sbatch script" in result.stdout


def test_extra_critical_export_rejected(fixture_root: Path) -> None:
    base = fixture_root
    sbatch, _, task_root, run_root, python = good_fixture(base)
    manifest = base / "stage" / "extra.manifest"
    manifest.write_text(
        f"HEAD_SHA={SHA}\nJOB_NAME={JOB_NAME}\nRUN_ROOT={run_root}\nDRY_RUN=0\nPYTHON={python}\n"
    )
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0
    assert "extra" in result.stdout


def test_home_staging_literal_rejected(fixture_root: Path) -> None:
    """Killer for mutant allow-home-staging: disabling the $HOME check lets this through."""
    base = fixture_root
    home_task_root = base / "home" / "staged-task"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "home.manifest"
    sbatch = base / "stage" / "home.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=home_task_root, run_root=run_root, dry_run=0, python=python)
    write_sbatch(sbatch, sha=SHA, job_name=JOB_NAME, task_root=home_task_root, run_root=run_root, dry_run=0, python=python)
    cmd = [
        "bash",
        str(LINTER),
        "--sbatch",
        str(sbatch),
        "--manifest",
        str(manifest),
        "--allowed-root",
        str(base / "stage"),
        "--allowed-root",
        str(base / "home"),
        "--shared-env-root",
        str(base / "shared-env"),
        "--home",
        str(base / "home"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "HOME" in result.stdout


def test_home_staging_via_symlink_rejected(fixture_root: Path) -> None:
    base = fixture_root
    real_staged = base / "home" / "real-staged"
    real_staged.mkdir()
    symlink = base / "stage" / "home-symlink"
    symlink.symlink_to(real_staged)
    task_root = symlink / "sub"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "symlink.manifest"
    sbatch = base / "stage" / "symlink.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    write_sbatch(sbatch, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    cmd = [
        "bash",
        str(LINTER),
        "--sbatch",
        str(sbatch),
        "--manifest",
        str(manifest),
        "--allowed-root",
        str(base / "stage"),
        "--allowed-root",
        str(base / "home"),
        "--shared-env-root",
        str(base / "shared-env"),
        "--home",
        str(base / "home"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "resolves under" in result.stdout


def test_facility_root_escape_rejected(fixture_root: Path) -> None:
    base = fixture_root
    escaped_task_root = base / "other-root" / "task"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "escape.manifest"
    sbatch = base / "stage" / "escape.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=escaped_task_root, run_root=run_root, dry_run=0, python=python)
    write_sbatch(sbatch, sha=SHA, job_name=JOB_NAME, task_root=escaped_task_root, run_root=run_root, dry_run=0, python=python)
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0
    assert "escapes every declared allowed root" in result.stdout


def test_dry_run_python_outside_shared_env_root_rejected(fixture_root: Path) -> None:
    base = fixture_root
    task_root = base / "stage" / "task"
    run_root = base / "stage" / "run"
    manifest = base / "stage" / "badpy.manifest"
    sbatch = base / "stage" / "badpy.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=1, python=Path("/usr/bin/python3"))
    write_sbatch(sbatch, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=1, python=Path("/usr/bin/python3"))
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0
    assert "dry-run PYTHON" in result.stdout


def test_dry_run_python_under_shared_env_root_accepted(fixture_root: Path) -> None:
    base = fixture_root
    task_root = base / "stage" / "task"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "goodpy.manifest"
    sbatch = base / "stage" / "goodpy.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=1, python=python)
    write_sbatch(sbatch, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=1, python=python)
    result = run_lint(base, sbatch, manifest)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ssh_without_batchmode_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    helper = fixture_root / "stage" / "helper-bad.sh"
    helper.write_text('#!/bin/bash\nssh cannon.example "sbatch x"\n')
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode != 0
    assert "BatchMode" in result.stdout


def test_ssh_with_batchmode_accepted(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    helper = fixture_root / "stage" / "helper-good.sh"
    helper.write_text('#!/bin/bash\nssh -o BatchMode=yes cannon.example "sbatch x"\n')
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_vs_dry_delta_limited_to_documented_keys_accepted(fixture_root: Path) -> None:
    base = fixture_root
    sbatch, manifest, task_root, *_ = good_fixture(base)
    real_env = base / "stage" / "real.env"
    dry_env = base / "stage" / "dry.env"
    real_env.write_text(f"HEAD_SHA={SHA}\nDRY_RUN=0\nTASK_ROOT={task_root}\n")
    dry_env.write_text(f"HEAD_SHA={SHA}\nDRY_RUN=1\nTASK_ROOT={task_root}\n")
    result = run_lint(base, sbatch, manifest, "--real-env", str(real_env), "--dry-env", str(dry_env), "--dry-only-keys", "DRY_RUN")
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_vs_dry_delta_on_undocumented_key_rejected(fixture_root: Path) -> None:
    base = fixture_root
    sbatch, manifest, task_root, *_ = good_fixture(base)
    real_env = base / "stage" / "real.env"
    dry_env = base / "stage" / "dry.env"
    real_env.write_text(f"HEAD_SHA={SHA}\nDRY_RUN=0\nTASK_ROOT={task_root}\n")
    dry_env.write_text(f"HEAD_SHA={SHA}\nDRY_RUN=1\nTASK_ROOT={task_root}-different\n")
    result = run_lint(base, sbatch, manifest, "--real-env", str(real_env), "--dry-env", str(dry_env), "--dry-only-keys", "DRY_RUN")
    assert result.returncode != 0
    assert "real-vs-dry export delta" in result.stdout
