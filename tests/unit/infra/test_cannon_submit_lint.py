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


def test_optional_key_unset_sentinel_vs_absent_rejected(fixture_root: Path) -> None:
    """real='__unset__' vs dry=<absent> is a presence delta, not a value match."""
    base = fixture_root
    sbatch, manifest, *_ = good_fixture(base)
    real_env = base / "stage" / "real-sentinel.env"
    dry_env = base / "stage" / "dry-sentinel.env"
    real_env.write_text(f"HEAD_SHA={SHA}\nDRY_RUN=0\nOPTIONAL=__unset__\n")
    dry_env.write_text(f"HEAD_SHA={SHA}\nDRY_RUN=1\n")
    result = run_lint(base, sbatch, manifest, "--real-env", str(real_env), "--dry-env", str(dry_env), "--dry-only-keys", "DRY_RUN")
    assert result.returncode != 0, result.stdout + result.stderr
    assert "present on only one side" in result.stdout


def test_optional_key_empty_on_both_sides_accepted(fixture_root: Path) -> None:
    base = fixture_root
    sbatch, manifest, *_ = good_fixture(base)
    real_env = base / "stage" / "real-empty.env"
    dry_env = base / "stage" / "dry-empty.env"
    real_env.write_text(f"HEAD_SHA={SHA}\nDRY_RUN=0\nOPTIONAL=\n")
    dry_env.write_text(f"HEAD_SHA={SHA}\nDRY_RUN=1\nOPTIONAL=\n")
    result = run_lint(base, sbatch, manifest, "--real-env", str(real_env), "--dry-env", str(dry_env), "--dry-only-keys", "DRY_RUN")
    assert result.returncode == 0, result.stdout + result.stderr


def test_critical_fields_missing_from_both_rejected(fixture_root: Path) -> None:
    """Killer for mutant accept-fields-missing-from-both: a required field absent
    from both the manifest and the sbatch script passes check 4 (nothing to
    compare) and must be caught by the separate presence/nonempty check."""
    base = fixture_root
    manifest = base / "stage" / "min.manifest"
    sbatch = base / "stage" / "min.sbatch"
    manifest.write_text(f"HEAD_SHA={SHA}\nJOB_NAME={JOB_NAME}\n")
    sbatch.write_text(f"#!/bin/bash\nexport HEAD_SHA={SHA}\nexport JOB_NAME={JOB_NAME}\n")
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "required critical field 'TASK_ROOT'" in result.stdout


def test_python_missing_from_both_under_dry_run_rejected(fixture_root: Path) -> None:
    base = fixture_root
    task_root = base / "stage" / "task"
    run_root = base / "stage" / "run"
    manifest = base / "stage" / "min-dry.manifest"
    sbatch = base / "stage" / "min-dry.sbatch"
    manifest.write_text(
        f"HEAD_SHA={SHA}\nJOB_NAME={JOB_NAME}\nTASK_ROOT={task_root}\nRUN_ROOT={run_root}\nDRY_RUN=1\n"
    )
    sbatch.write_text(
        "#!/bin/bash\n"
        f"export HEAD_SHA={SHA}\nexport JOB_NAME={JOB_NAME}\nexport TASK_ROOT={task_root}\n"
        f"export RUN_ROOT={run_root}\nexport DRY_RUN=1\n"
    )
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "required critical field 'PYTHON'" in result.stdout


def test_ssh_batchmode_mentioned_only_in_trailing_comment_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    helper = fixture_root / "stage" / "helper-comment-trick.sh"
    helper.write_text('#!/bin/bash\nssh cannon.example true # -o BatchMode=yes\n')
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "BatchMode" in result.stdout


def test_ssh_second_command_without_batchmode_rejected(fixture_root: Path) -> None:
    """Killer for mutant accept-ssh-second-command-no-batchmode: a second ssh
    invocation on the same logical line, separated by ';', must be checked on
    its own rather than inheriting the first invocation's BatchMode option."""
    sbatch, manifest, *_ = good_fixture(fixture_root)
    helper = fixture_root / "stage" / "helper-second-cmd.sh"
    helper.write_text('#!/bin/bash\nssh -o BatchMode=yes a true; ssh b true\n')
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "BatchMode" in result.stdout


def test_ssh_comment_mentioning_ssh_not_flagged(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    helper = fixture_root / "stage" / "helper-comment-mention.sh"
    helper.write_text('#!/bin/bash\n# ssh invocations below always carry -o BatchMode=yes\ntrue\n')
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode == 0, result.stdout + result.stderr


def test_ssh_continuation_with_batchmode_accepted(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    helper = fixture_root / "stage" / "helper-continuation.sh"
    helper.write_text('#!/bin/bash\nssh \\\n  -o BatchMode=yes cannon.example "sbatch x"\n')
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "helper_body",
    [
        "timeout 30 ssh host true\n",
        "env X=1 ssh host\n",
        "command ssh host\n",
    ],
)
def test_ssh_wrapped_by_another_command_without_batchmode_rejected(fixture_root: Path, helper_body: str) -> None:
    """ssh/scp/rsync must be caught as a command word anywhere in the
    segment, not only when the segment starts with it."""
    sbatch, manifest, *_ = good_fixture(fixture_root)
    helper = fixture_root / "stage" / "helper-wrapped.sh"
    helper.write_text(f"#!/bin/bash\n{helper_body}")
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "BatchMode" in result.stdout


def test_rsync_via_ssh_without_batchmode_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    helper = fixture_root / "stage" / "helper-rsync.sh"
    helper.write_text("#!/bin/bash\nrsync -a -e ssh src host:dst\n")
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "BatchMode" in result.stdout


def test_ssh_inside_command_substitution_without_batchmode_rejected(fixture_root: Path) -> None:
    """Killer for mutant accept-ssh-command-word-anywhere: a bare prefix-anchored
    check misses ssh invoked inside `$( ... )`."""
    sbatch, manifest, *_ = good_fixture(fixture_root)
    helper = fixture_root / "stage" / "helper-subshell.sh"
    helper.write_text("#!/bin/bash\nx=$(ssh host true)\n")
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "BatchMode" in result.stdout


def test_scp_without_batchmode_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    helper = fixture_root / "stage" / "helper-scp.sh"
    helper.write_text("#!/bin/bash\nscp host:a b\n")
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "BatchMode" in result.stdout


def test_sshd_config_and_rsync_log_substrings_not_flagged(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    helper = fixture_root / "stage" / "helper-no-match.sh"
    helper.write_text("#!/bin/bash\ncat /etc/sshd_config\nrotate rsync_log\n")
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_env_without_dry_env_rejected(fixture_root: Path) -> None:
    base = fixture_root
    sbatch, manifest, task_root, *_ = good_fixture(base)
    real_env = base / "stage" / "real-only.env"
    real_env.write_text(f"HEAD_SHA={SHA}\nDRY_RUN=0\nTASK_ROOT={task_root}\n")
    result = run_lint(base, sbatch, manifest, "--real-env", str(real_env))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "must be given together" in result.stdout


def test_dry_env_without_real_env_rejected(fixture_root: Path) -> None:
    base = fixture_root
    sbatch, manifest, task_root, *_ = good_fixture(base)
    dry_env = base / "stage" / "dry-only.env"
    dry_env.write_text(f"HEAD_SHA={SHA}\nDRY_RUN=1\nTASK_ROOT={task_root}\n")
    result = run_lint(base, sbatch, manifest, "--dry-env", str(dry_env))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "must be given together" in result.stdout


def test_realpath_without_dash_m_support_fails_clearly(fixture_root: Path, tmp_path: Path) -> None:
    """A BSD/macOS-shaped realpath (no -m) must produce a clear, named LINT_FAIL
    instead of a cryptic downstream error."""
    sbatch, manifest, *_ = good_fixture(fixture_root)
    orig_realpath = subprocess.run(["bash", "-c", "command -v realpath"], capture_output=True, text=True, check=True).stdout.strip()
    shim_dir = tmp_path / "bsd-realpath-shim"
    shim_dir.mkdir()
    shim = shim_dir / "realpath"
    shim.write_text(
        "#!/bin/bash\n"
        "for a in \"$@\"; do\n"
        "  if [[ \"$a\" == \"-m\" ]]; then\n"
        "    echo 'realpath: illegal option -- m' >&2\n"
        "    exit 1\n"
        "  fi\n"
        "done\n"
        f"exec {orig_realpath} \"$@\"\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}:{env['PATH']}"
    result = subprocess.run(
        [
            "bash", str(LINTER),
            "--sbatch", str(sbatch), "--manifest", str(manifest),
            "--allowed-root", str(fixture_root / "stage"),
            "--shared-env-root", str(fixture_root / "shared-env"),
            "--home", str(fixture_root / "home"),
        ],
        capture_output=True, text=True, check=False, env=env,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "GNU coreutils realpath is required" in result.stderr


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
