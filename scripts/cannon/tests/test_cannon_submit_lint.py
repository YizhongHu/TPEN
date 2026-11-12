"""Contract tests for ``scripts/cannon/submit-lint.sh``.

Each test drives the linter as a subprocess against a small fixture tree
built under ``tmp_path`` — no Slurm, no cluster access, bash/stdlib only
(mirrors ``scripts/cannon/submit-lint-selftest.sh``, which exercises the same
checks for a local, pytest-free regression run).
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

import pytest

LINTER = Path(__file__).resolve().parents[3] / "scripts" / "cannon" / "submit-lint.sh"
PRECHECK = Path(__file__).resolve().parents[3] / "scripts" / "cannon" / "shared-env-precheck.sh"
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
        "--shared-env-prefix",
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
        "--shared-env-prefix",
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
        "--shared-env-prefix",
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
        "--shared-env-prefix",
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
    assert result.returncode == 2, result.stdout + result.stderr
    assert "must be given together" in result.stderr


def test_dry_env_without_real_env_rejected(fixture_root: Path) -> None:
    base = fixture_root
    sbatch, manifest, task_root, *_ = good_fixture(base)
    dry_env = base / "stage" / "dry-only.env"
    dry_env.write_text(f"HEAD_SHA={SHA}\nDRY_RUN=1\nTASK_ROOT={task_root}\n")
    result = run_lint(base, sbatch, manifest, "--dry-env", str(dry_env))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "must be given together" in result.stderr


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
            "--shared-env-prefix", str(fixture_root / "shared-env"),
            "--home", str(fixture_root / "home"),
        ],
        capture_output=True, text=True, check=False, env=env,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "GNU coreutils realpath is required" in result.stderr


def test_bare_reexport_after_sbatch_export_accepted(fixture_root: Path) -> None:
    """False block (ef480b65 linter-trial-log, job 59534855): the harness
    passes critical values via `#SBATCH --export=K=V,...` and only
    bare-re-exports them in the body (`export A B C`, no '='). The old
    single-pattern grep (`export KEY=`) never saw these as present at all."""
    base = fixture_root
    task_root = base / "stage" / "task"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "reexport.manifest"
    sbatch = base / "stage" / "reexport.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    sbatch.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name={JOB_NAME}\n"
        f"#SBATCH --export=HEAD_SHA={SHA},JOB_NAME={JOB_NAME},TASK_ROOT={task_root},"
        f"RUN_ROOT={run_root},DRY_RUN=0,PYTHON={python}\n"
        "export HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT DRY_RUN PYTHON\n"
    )
    result = run_lint(base, sbatch, manifest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LINT_PASS" in result.stdout


def test_bare_reexport_not_in_export_list_still_missing(fixture_root: Path) -> None:
    """A bare re-export name that --export never supplied must still be
    caught as missing, not silently accepted."""
    base = fixture_root
    task_root = base / "stage" / "task"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "reexport-partial.manifest"
    sbatch = base / "stage" / "reexport-partial.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    sbatch.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --export=HEAD_SHA={SHA},JOB_NAME={JOB_NAME}\n"
        "export HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT DRY_RUN PYTHON\n"
    )
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "missing from sbatch script" in result.stdout


def test_multi_assignment_export_line_parsed_per_key(fixture_root: Path) -> None:
    """False block / linter bug (ef480b65 linter-trial-log, job 59534855): a
    multi-assignment export line (`export DRY_RUN="${DRY_RUN:-0}"
    DRY_NODE_TMP="${DRY_NODE_TMP:-}"`) made the old single-key extractor
    swallow the rest of the line into DRY_RUN's value, reporting a false
    'stale' critical export. DRY_RUN's self-referential default expansion
    must resolve from the sbatch --export list, and DRY_NODE_TMP (not a
    critical key) must not corrupt that parse."""
    base = fixture_root
    task_root = base / "stage" / "task"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "multiassign.manifest"
    sbatch = base / "stage" / "multiassign.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    sbatch.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name={JOB_NAME}\n"
        f"#SBATCH --export=HEAD_SHA={SHA},JOB_NAME={JOB_NAME},TASK_ROOT={task_root},"
        f"RUN_ROOT={run_root},DRY_RUN=0,PYTHON={python}\n"
        'export DRY_RUN="${DRY_RUN:-0}" DRY_NODE_TMP="${DRY_NODE_TMP:-}"\n'
        "export HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT PYTHON\n"
    )
    result = run_lint(base, sbatch, manifest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LINT_PASS" in result.stdout


def test_multi_assignment_export_line_second_key_not_swallowed(fixture_root: Path) -> None:
    """The second assignment on a multi-assignment line must parse to its
    own literal value, not get absorbed into the first key's value."""
    base = fixture_root
    task_root = base / "stage" / "task"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "multiassign2.manifest"
    sbatch = base / "stage" / "multiassign2.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    stale_run_root = run_root.parent / "run-stale"
    sbatch.write_text(
        "#!/bin/bash\n"
        f"export HEAD_SHA={SHA} JOB_NAME={JOB_NAME}\n"
        f"export TASK_ROOT={task_root} RUN_ROOT={stale_run_root}\n"
        f"export DRY_RUN=0 PYTHON={python}\n"
    )
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'RUN_ROOT' is stale" in result.stdout


def test_staging_root_mapping_accepts_sbatch_linted_at_local_root(fixture_root: Path) -> None:
    """False block (ef480b65 linter-trial-log, job 59534855): check 3 fired
    on the local staging path before rsync, because the linter assumed the
    sbatch script was already linted in place at its final root. With
    `--staging-root LOCAL=FINAL`, a sbatch script physically staged under
    LOCAL is validated for containment as if it already sat under FINAL."""
    base = fixture_root
    local_root = base / "local-stage"
    final_root = base / "final-stage"
    local_root.mkdir()
    task_root = final_root / "task"
    run_root = final_root / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = local_root / "staged.manifest"
    sbatch = local_root / "staged.sbatch"
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
        str(final_root),
        "--shared-env-prefix",
        str(base / "shared-env"),
        "--home",
        str(base / "home"),
        "--staging-root",
        f"{local_root}={final_root}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LINT_PASS" in result.stdout


def test_staging_root_mapping_does_not_relax_true_escape(fixture_root: Path) -> None:
    """A `--staging-root` mapping must never widen containment: a path that
    escapes the final root even after mapping stays rejected."""
    base = fixture_root
    local_root = base / "local-stage2"
    final_root = base / "final-stage2"
    other_root = base / "other-root"
    local_root.mkdir()
    task_root = other_root / "task"
    run_root = final_root / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = local_root / "escape.manifest"
    sbatch = local_root / "escape.sbatch"
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
        str(final_root),
        "--shared-env-prefix",
        str(base / "shared-env"),
        "--home",
        str(base / "home"),
        "--staging-root",
        f"{local_root}={final_root}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "escapes every declared allowed root" in result.stdout


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


# --- Round 2 (T1 review of PR #653): effective environment, quoting,
# mapped-destination symlinks, and the submit helper's `sbatch --export=`.


def stage_paths(base: Path) -> tuple[Path, Path, Path]:
    return base / "stage" / "task", base / "stage" / "run", base / "shared-env" / "bin" / "python"


def export_spec(base: Path, *, dry_run: str = "0") -> str:
    task_root, run_root, python = stage_paths(base)
    return (
        f"HEAD_SHA={SHA},JOB_NAME={JOB_NAME},TASK_ROOT={task_root},"
        f"RUN_ROOT={run_root},DRY_RUN={dry_run},PYTHON={python}"
    )


def lint_body(base: Path, name: str, body: str, *extra: str, directive: bool = True) -> subprocess.CompletedProcess[str]:
    """Lints a job whose `#SBATCH --export=` directive supplies all six
    critical values (DRY_RUN=0) followed by BODY, against a manifest with
    DRY_RUN=0. Only BODY decides whether the effective values still match."""
    task_root, run_root, python = stage_paths(base)
    manifest = base / "stage" / f"{name}.manifest"
    sbatch = base / "stage" / f"{name}.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    header = "#!/bin/bash\n" + (f"#SBATCH --export={export_spec(base)}\n" if directive else "")
    sbatch.write_text(header + body)
    return run_lint(base, sbatch, manifest, *extra)


REEXPORT_REST = "export HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT PYTHON\n"


def test_bare_reexport_keeps_current_value_not_directive_value(fixture_root: Path) -> None:
    """Finding 1: `export DRY_RUN=1` then a bare `export DRY_RUN` leaves the
    job with DRY_RUN=1; resolving the bare name from the directive table
    (DRY_RUN=0) passed a stale value."""
    result = lint_body(fixture_root, "f1-bare", "export DRY_RUN=1\nexport DRY_RUN\n" + REEXPORT_REST)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'DRY_RUN' is stale: manifest='0' sbatch='1'" in result.stdout


def test_self_default_uses_current_value_not_directive_value(fixture_root: Path) -> None:
    result = lint_body(fixture_root, "f1-default", 'export DRY_RUN=1\nexport DRY_RUN="${DRY_RUN:-0}"\n' + REEXPORT_REST)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'DRY_RUN' is stale: manifest='0' sbatch='1'" in result.stdout


def test_plain_assignment_after_export_updates_effective_value(fixture_root: Path) -> None:
    """A plain `DRY_RUN=1` changes an already-exported variable in place."""
    result = lint_body(fixture_root, "f1-plain", "export DRY_RUN\nDRY_RUN=1\n" + REEXPORT_REST)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'DRY_RUN' is stale: manifest='0' sbatch='1'" in result.stdout


def test_command_prefix_assignment_does_not_change_job_environment(fixture_root: Path) -> None:
    result = lint_body(fixture_root, "f1-prefix", "DRY_RUN=1 python run.py | tee log\nexport DRY_RUN\n" + REEXPORT_REST)
    assert result.returncode == 0, result.stdout + result.stderr


def test_single_quoted_value_is_never_expanded(fixture_root: Path) -> None:
    """Finding 2 (reviewer negative): bash keeps the literal `${DRY_RUN:-0}`
    inside single quotes; the old tokenizer expanded it to 0."""
    result = lint_body(fixture_root, "f2-squote", "export DRY_RUN='${DRY_RUN:-0}' NOTE='x y=z'\n" + REEXPORT_REST)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'DRY_RUN' is stale: manifest='0' sbatch='${DRY_RUN:-0}'" in result.stdout


def test_unquoted_comment_ends_export_statement(fixture_root: Path) -> None:
    """Finding 2 (reviewer negative): `# DRY_RUN=0` is a comment, not an assignment."""
    result = lint_body(fixture_root, "f2-comment", "export DRY_RUN=1 # DRY_RUN=0\n" + REEXPORT_REST)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'DRY_RUN' is stale: manifest='0' sbatch='1'" in result.stdout


def test_literal_quoted_spaces_and_equals_accepted(fixture_root: Path) -> None:
    """Positive control: quoted spaces and '=' stay inside their own value."""
    result = lint_body(fixture_root, "f2-literal", "export NOTE='x y=z' OTHER=\"a b=c\" DRY_RUN=0\n" + REEXPORT_REST)
    assert result.returncode == 0, result.stdout + result.stderr


def test_command_substitution_in_critical_value_rejected(fixture_root: Path) -> None:
    result = lint_body(fixture_root, "f2-cmdsub", 'export DRY_RUN="$(echo 0)"\n' + REEXPORT_REST)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "will not guess its value" in result.stdout


def test_command_substitution_in_noncritical_value_tolerated(fixture_root: Path) -> None:
    result = lint_body(fixture_root, "f2-cmdsub-ok", 'export PATH="$(pwd)/bin:$PATH"\nexport DRY_RUN\n' + REEXPORT_REST)
    assert result.returncode == 0, result.stdout + result.stderr


def staging_symlink_fixture(base: Path) -> tuple[Path, Path]:
    """LOCAL/link is a real local directory; FINAL/link -> HOME, so every
    path under LOCAL/link maps to a FINAL path that resolves into HOME."""
    local_root = base / "local-stage"
    final_root = base / "final-stage"
    (local_root / "link").mkdir(parents=True)
    final_root.mkdir()
    (final_root / "link").symlink_to(base / "home")
    return local_root, final_root


def lint_staged(base: Path, sbatch: Path, task_root: Path, run_root: Path, local_root: Path, final_root: Path) -> subprocess.CompletedProcess[str]:
    python = base / "shared-env" / "bin" / "python"
    manifest = Path(f"{sbatch}.manifest")
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    write_sbatch(sbatch, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    cmd = [
        "bash", str(LINTER),
        "--sbatch", str(sbatch), "--manifest", str(manifest),
        "--allowed-root", str(final_root),
        "--shared-env-prefix", str(base / "shared-env"),
        "--home", str(base / "home"),
        "--staging-root", f"{local_root}={final_root}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("key", ["TASK_ROOT", "RUN_ROOT"])
def test_staging_mapped_root_symlink_escape_rejected(fixture_root: Path, key: str) -> None:
    """Finding 3: LOCAL/link/x maps to FINAL/link/x, which resolves to
    HOME/x; the raw mapped string looked contained."""
    local_root, final_root = staging_symlink_fixture(fixture_root)
    escaped = local_root / "link" / "x"
    task_root = escaped if key == "TASK_ROOT" else final_root / "task"
    run_root = escaped if key == "RUN_ROOT" else final_root / "run"
    result = lint_staged(fixture_root, local_root / "job.sbatch", task_root, run_root, local_root, final_root)
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"{key} resolves under $HOME" in result.stdout
    assert f"{key} escapes every declared allowed root" in result.stdout


def test_staging_sbatch_symlink_escape_rejected(fixture_root: Path) -> None:
    """Finding 3, sbatch path: LOCAL/link/job.sbatch maps to FINAL/link/job.sbatch -> HOME."""
    local_root, final_root = staging_symlink_fixture(fixture_root)
    result = lint_staged(fixture_root, local_root / "link" / "job.sbatch", final_root / "task", final_root / "run", local_root, final_root)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "sbatch path escapes every declared stage root" in result.stdout


def helper_fixture(base: Path, *, dry_run: str) -> tuple[Path, Path, Path]:
    """Real CLI shape: the job holds only bare re-exports; the submit helper's
    own `sbatch --export=... <job>` line supplies all six values."""
    task_root, run_root, python = stage_paths(base)
    manifest = base / "stage" / "cli.manifest"
    sbatch = base / "stage" / "cli-job.sbatch"
    helper = base / "stage" / "cli-submit.sh"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    sbatch.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name={JOB_NAME}\n"
        "export HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT DRY_RUN PYTHON\n"
    )
    helper.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f"jobid=$(sbatch --parsable --export={export_spec(base, dry_run=dry_run)} {sbatch})\n"
        'echo "$jobid"\n'
    )
    return sbatch, manifest, helper


def test_submit_helper_export_line_supplies_bare_reexports(fixture_root: Path) -> None:
    """Finding 4 (trial-log false block): values arrive on the helper's
    command line, not in a `#SBATCH` directive."""
    sbatch, manifest, helper = helper_fixture(fixture_root, dry_run="0")
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LINT_PASS" in result.stdout


def test_submit_helper_export_line_wrong_value_rejected(fixture_root: Path) -> None:
    sbatch, manifest, helper = helper_fixture(fixture_root, dry_run="1")
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'DRY_RUN' is stale: manifest='0' sbatch='1'" in result.stdout


def test_submit_helper_export_overrides_directive(fixture_root: Path) -> None:
    """A command-line --export replaces the script's directive, as in Slurm."""
    sbatch, manifest, helper = helper_fixture(fixture_root, dry_run="1")
    sbatch.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --export={export_spec(fixture_root, dry_run='0')}\n"
        "export HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT DRY_RUN PYTHON\n"
    )
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'DRY_RUN' is stale: manifest='0' sbatch='1'" in result.stdout


def test_submit_helper_export_with_shell_expansion_rejected(fixture_root: Path) -> None:
    """Values the linter cannot evaluate are rejected, never inferred (and
    never taken from the manifest)."""
    sbatch, manifest, helper = helper_fixture(fixture_root, dry_run="0")
    helper.write_text(f'#!/bin/bash\nsbatch --export="HEAD_SHA=$SHA,DRY_RUN=0" {sbatch}\n')
    result = run_lint(fixture_root, sbatch, manifest, "--submit-helper", str(helper))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "unsupported --export value" in result.stdout


# --- --shared-env-prefix: the prefix split (disjoint from every
# --allowed-root and every --staging-root FINAL). ---------------------------


def test_shared_env_prefix_nested_in_allowed_root_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    nested_shared = fixture_root / "stage" / "shared-env-nested"
    (nested_shared / "bin").mkdir(parents=True)
    result = subprocess.run(
        [
            "bash", str(LINTER),
            "--sbatch", str(sbatch), "--manifest", str(manifest),
            "--allowed-root", str(fixture_root / "stage"),
            "--shared-env-prefix", str(nested_shared),
            "--home", str(fixture_root / "home"),
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "disjoint" in result.stdout


def test_shared_env_prefix_containing_allowed_root_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    inner_allowed = fixture_root / "shared-env" / "bin"
    result = subprocess.run(
        [
            "bash", str(LINTER),
            "--sbatch", str(sbatch), "--manifest", str(manifest),
            "--allowed-root", str(inner_allowed),
            "--shared-env-prefix", str(fixture_root / "shared-env"),
            "--home", str(fixture_root / "home"),
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "disjoint" in result.stdout


def test_shared_env_prefix_nested_in_staging_root_final_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    local_stage = fixture_root / "local-stage"
    final_stage = fixture_root / "final-stage"
    local_stage.mkdir()
    final_stage.mkdir()
    result = subprocess.run(
        [
            "bash", str(LINTER),
            "--sbatch", str(sbatch), "--manifest", str(manifest),
            "--allowed-root", str(fixture_root / "stage"),
            "--shared-env-prefix", str(final_stage / "shared-sub"),
            "--home", str(fixture_root / "home"),
            "--staging-root", f"{local_stage}={final_stage}",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "disjoint" in result.stdout


# --- Check 10: manifest SHARED_ENV_PREFIX/SHARED_ENV_PATH keys must stay
# consistent with --shared-env-prefix. ---------------------------------------


def test_manifest_shared_env_prefix_matching_cli_accepted(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    manifest2 = fixture_root / "stage" / "senv-match.manifest"
    manifest2.write_text(manifest.read_text() + f"SHARED_ENV_PREFIX={fixture_root / 'shared-env'}\n")
    sbatch2 = fixture_root / "stage" / "senv-match.sbatch"
    sbatch2.write_text(sbatch.read_text() + f"export SHARED_ENV_PREFIX={fixture_root / 'shared-env'}\n")
    result = run_lint(fixture_root, sbatch2, manifest2)
    assert result.returncode == 0, result.stdout + result.stderr


def test_manifest_shared_env_prefix_mismatch_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    manifest2 = fixture_root / "stage" / "senv-mismatch.manifest"
    manifest2.write_text(manifest.read_text() + f"SHARED_ENV_PREFIX={fixture_root / 'other-root'}\n")
    sbatch2 = fixture_root / "stage" / "senv-mismatch.sbatch"
    sbatch2.write_text(sbatch.read_text() + f"export SHARED_ENV_PREFIX={fixture_root / 'other-root'}\n")
    result = run_lint(fixture_root, sbatch2, manifest2)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "does not match --shared-env-prefix" in result.stdout


# --- Check 4 (T1, review-round-1 F1): SHARED_ENV_PREFIX is a critical key,
# reconciled between the manifest and the job's own effective export. --------


def test_job_exported_shared_env_prefix_stale_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    manifest2 = fixture_root / "stage" / "senv-prefix-stale.manifest"
    manifest2.write_text(manifest.read_text() + f"SHARED_ENV_PREFIX={fixture_root / 'shared-env'}\n")
    sbatch2 = fixture_root / "stage" / "senv-prefix-stale.sbatch"
    sbatch2.write_text(sbatch.read_text() + f"export SHARED_ENV_PREFIX={fixture_root / 'other-root'}\n")
    result = run_lint(fixture_root, sbatch2, manifest2)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'SHARED_ENV_PREFIX' is stale" in result.stdout


def test_job_exported_shared_env_prefix_missing_from_manifest_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    sbatch2 = fixture_root / "stage" / "senv-prefix-extra.sbatch"
    sbatch2.write_text(sbatch.read_text() + f"export SHARED_ENV_PREFIX={fixture_root / 'shared-env'}\n")
    result = run_lint(fixture_root, sbatch2, manifest)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'SHARED_ENV_PREFIX' present in sbatch script but not in manifest" in result.stdout


def test_manifest_shared_env_prefix_missing_from_job_export_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    manifest2 = fixture_root / "stage" / "senv-prefix-missing.manifest"
    manifest2.write_text(manifest.read_text() + f"SHARED_ENV_PREFIX={fixture_root / 'shared-env'}\n")
    result = run_lint(fixture_root, sbatch, manifest2)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'SHARED_ENV_PREFIX' missing from sbatch script" in result.stdout


# --- Check 4 (DT1, T2 delta review DF1): a bare `export SHARED_ENV_PREFIX`
# with no prior assignment leaves it exported with an undetermined value
# (state "unset"), which `eff_get` otherwise reports the same as "never
# mentioned" -- Check 4 must not read that as "absent from both sides". -----


def test_job_bare_exported_shared_env_prefix_no_manifest_rejected(fixture_root: Path) -> None:
    """DF1 regression: no manifest key, no prior assignment, just a bare
    re-export -- this must fail, not silently pass as mutually absent."""
    result = lint_body(
        fixture_root, "df1-bare-export",
        "export SHARED_ENV_PREFIX\n" + REEXPORT_REST,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'SHARED_ENV_PREFIX' is exported with no determined value" in result.stdout


def test_job_bare_exported_shared_env_prefix_ignores_inherited_real_value(fixture_root: Path) -> None:
    """The linter is static/syntax-only; a bare `export NAME` records only
    that NAME is exported, never the real process environment. An inherited,
    mismatching real value in the *linter's own* environment must not change
    the DT1 verdict above."""
    task_root, run_root, python = stage_paths(fixture_root)
    manifest = fixture_root / "stage" / "df1-inherited.manifest"
    sbatch = fixture_root / "stage" / "df1-inherited.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    sbatch.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --export={export_spec(fixture_root)}\n"
        "export SHARED_ENV_PREFIX\n" + REEXPORT_REST
    )
    env = dict(os.environ)
    env["SHARED_ENV_PREFIX"] = str(fixture_root / "other-root")
    cmd = [
        "bash", str(LINTER),
        "--sbatch", str(sbatch), "--manifest", str(manifest),
        "--allowed-root", str(fixture_root / "stage"),
        "--shared-env-prefix", str(fixture_root / "shared-env"),
        "--home", str(fixture_root / "home"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "critical export 'SHARED_ENV_PREFIX' is exported with no determined value" in result.stdout


def test_both_files_omit_shared_env_prefix_accepted(fixture_root: Path) -> None:
    """Green control: SHARED_ENV_PREFIX never mentioned anywhere (no
    directive, no assignment, no manifest key) is genuinely absent on both
    sides and must still pass."""
    sbatch, manifest, *_ = good_fixture(fixture_root)
    result = run_lint(fixture_root, sbatch, manifest)
    assert result.returncode == 0, result.stdout + result.stderr


def test_literal_assignment_plus_bare_reexport_matching_manifest_accepted(fixture_root: Path) -> None:
    """Green control: a literal assignment followed by a bare re-export keeps
    the already-known value (state stays "set", not "unset"), so normal
    reconciliation against a matching manifest still passes."""
    sbatch, manifest, *_ = good_fixture(fixture_root)
    senv = fixture_root / "shared-env"
    manifest2 = fixture_root / "stage" / "senv-literal-bare.manifest"
    manifest2.write_text(manifest.read_text() + f"SHARED_ENV_PREFIX={senv}\n")
    sbatch2 = fixture_root / "stage" / "senv-literal-bare.sbatch"
    sbatch2.write_text(sbatch.read_text() + f"export SHARED_ENV_PREFIX={senv}\nexport SHARED_ENV_PREFIX\n")
    result = run_lint(fixture_root, sbatch2, manifest2)
    assert result.returncode == 0, result.stdout + result.stderr


def test_directive_bare_shared_env_prefix_unknown_rejected(fixture_root: Path) -> None:
    """Already-accepted T1 case, pinned under the SHARED_ENV_PREFIX name:
    `#SBATCH --export=...,SHARED_ENV_PREFIX` (bare in the directive, no
    literal value anywhere) is "unknown", not "unset", and already rejected
    by the pre-existing generic unknown-export loop -- confirm DF1's new
    check does not change or duplicate that outcome."""
    task_root, run_root, python = stage_paths(fixture_root)
    manifest = fixture_root / "stage" / "df1-directive-bare.manifest"
    sbatch = fixture_root / "stage" / "df1-directive-bare.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    sbatch.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --export={export_spec(fixture_root)},SHARED_ENV_PREFIX\n" + REEXPORT_REST
    )
    result = run_lint(fixture_root, sbatch, manifest)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "has a value the linter cannot determine" in result.stdout


# --- Check 10 (T2, review-round-1 F2): --shared-env-prefix and its mirrored
# manifest keys must already be spelled canonically. -------------------------


def test_cli_shared_env_prefix_trailing_slash_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    result = subprocess.run(
        [
            "bash", str(LINTER),
            "--sbatch", str(sbatch), "--manifest", str(manifest),
            "--allowed-root", str(fixture_root / "stage"),
            "--shared-env-prefix", str(fixture_root / "shared-env") + "/",
            "--home", str(fixture_root / "home"),
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--shared-env-prefix is not canonical" in result.stdout


def test_cli_shared_env_prefix_relative_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    result = subprocess.run(
        [
            "bash", str(LINTER),
            "--sbatch", str(sbatch), "--manifest", str(manifest),
            "--allowed-root", str(fixture_root / "stage"),
            "--shared-env-prefix", "shared-env",
            "--home", str(fixture_root / "home"),
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--shared-env-prefix is not absolute" in result.stdout


def test_cli_shared_env_prefix_symlink_alias_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    alias = fixture_root / "shared-env-alias"
    alias.symlink_to(fixture_root / "shared-env")
    result = subprocess.run(
        [
            "bash", str(LINTER),
            "--sbatch", str(sbatch), "--manifest", str(manifest),
            "--allowed-root", str(fixture_root / "stage"),
            "--shared-env-prefix", str(alias),
            "--home", str(fixture_root / "home"),
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--shared-env-prefix is not canonical" in result.stdout


def test_manifest_shared_env_prefix_trailing_slash_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    manifest2 = fixture_root / "stage" / "senv-prefix-slash.manifest"
    manifest2.write_text(manifest.read_text() + f"SHARED_ENV_PREFIX={fixture_root / 'shared-env'}/\n")
    sbatch2 = fixture_root / "stage" / "senv-prefix-slash.sbatch"
    sbatch2.write_text(sbatch.read_text() + f"export SHARED_ENV_PREFIX={fixture_root / 'shared-env'}/\n")
    result = run_lint(fixture_root, sbatch2, manifest2)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "manifest SHARED_ENV_PREFIX is not canonical" in result.stdout


def test_manifest_shared_env_path_direct_child_accepted(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    manifest2 = fixture_root / "stage" / "senv-path-good.manifest"
    manifest2.write_text(
        manifest.read_text() + f"SHARED_ENV_PATH={fixture_root / 'shared-env' / 'tpen-abc123-cpu-py3.12'}\n"
    )
    result = run_lint(fixture_root, sbatch, manifest2)
    assert result.returncode == 0, result.stdout + result.stderr


def test_manifest_shared_env_path_two_levels_deep_rejected(fixture_root: Path) -> None:
    sbatch, manifest, *_ = good_fixture(fixture_root)
    manifest2 = fixture_root / "stage" / "senv-path-bad.manifest"
    manifest2.write_text(
        manifest.read_text() + f"SHARED_ENV_PATH={fixture_root / 'shared-env' / 'sub' / 'tpen-abc123-cpu-py3.12'}\n"
    )
    result = run_lint(fixture_root, sbatch, manifest2)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "not a direct child" in result.stdout


def test_manifest_shared_env_path_from_real_precheck_resolve_accepted(fixture_root: Path) -> None:
    """DT2 (T2 delta review, completing accepted T4): feed the real stdout of
    `shared-env-precheck.sh resolve` into the lint manifest's SHARED_ENV_PATH,
    rather than a handcrafted string -- the actual producer/consumer pair."""
    shared_prefix = fixture_root / "shared-env"
    lock_sha256 = hashlib.sha256(b"dt2-lock-contents\n").hexdigest()
    env_name = f"tpen-{lock_sha256[:12]}-cpu-py3.12"
    target = shared_prefix / env_name
    (target / "bin").mkdir(parents=True)
    python = target / "bin" / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    (target / ".complete").write_text(
        "head=deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n"
        f"uv_lock_sha256={lock_sha256}\n"
        "uv_version=uv 9.9.9-stub\n"
        "python_version=3.12.7\n"
        "python_request=3.12\n"
        "built_utc=2026-01-01T00:00:00Z\n"
    )
    resolve = subprocess.run(
        [
            "bash", str(PRECHECK), "resolve",
            "--shared-env-prefix", str(shared_prefix),
            "--project", "tpen", "--extra", "cpu", "--python", "3.12",
            "--lock-sha256", lock_sha256,
        ],
        capture_output=True, text=True, check=False,
    )
    assert resolve.returncode == 0, resolve.stdout + resolve.stderr
    resolved_path = resolve.stdout.strip()
    assert resolved_path == str(target)

    sbatch, manifest, *_ = good_fixture(fixture_root)
    manifest2 = fixture_root / "stage" / "senv-path-real-resolve.manifest"
    manifest2.write_text(manifest.read_text() + f"SHARED_ENV_PATH={resolved_path}\n")
    result = run_lint(fixture_root, sbatch, manifest2)
    assert result.returncode == 0, result.stdout + result.stderr


# --- Check 11: TASK_ROOT/RUN_ROOT must not resolve under --shared-env-prefix.


def test_task_root_under_shared_env_prefix_rejected(fixture_root: Path) -> None:
    base = fixture_root
    task_root = base / "shared-env" / "task"
    run_root = base / "stage" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "senv11-task.manifest"
    sbatch = base / "stage" / "senv11-task.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    write_sbatch(sbatch, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "resolves under --shared-env-prefix" in result.stdout


def test_run_root_under_shared_env_prefix_rejected(fixture_root: Path) -> None:
    base = fixture_root
    task_root = base / "stage" / "task"
    run_root = base / "shared-env" / "run"
    python = base / "shared-env" / "bin" / "python"
    manifest = base / "stage" / "senv11-run.manifest"
    sbatch = base / "stage" / "senv11-run.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    write_sbatch(sbatch, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=0, python=python)
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "resolves under --shared-env-prefix" in result.stdout


def shared_env_staging_symlink_fixture(base: Path) -> tuple[Path, Path]:
    """LOCAL/link is a real local directory; FINAL/link -> shared-env, so
    every path under LOCAL/link maps to a FINAL path that resolves into the
    reuse-only shared-env prefix -- the Check-11 analogue of
    `staging_symlink_fixture`'s $HOME escape (DT2: shared-prefix, not HOME)."""
    local_root = base / "local-stage-senv"
    final_root = base / "final-stage-senv"
    (local_root / "link").mkdir(parents=True)
    final_root.mkdir()
    (final_root / "link").symlink_to(base / "shared-env")
    return local_root, final_root


@pytest.mark.parametrize("key", ["TASK_ROOT", "RUN_ROOT"])
def test_staging_mapped_root_shared_env_prefix_escape_rejected(fixture_root: Path, key: str) -> None:
    """DT2: Check 11's staging-root mapping must catch a symlink escape into
    --shared-env-prefix, not only the $HOME case Check 5 already covers."""
    local_root, final_root = shared_env_staging_symlink_fixture(fixture_root)
    escaped = local_root / "link" / "x"
    task_root = escaped if key == "TASK_ROOT" else final_root / "task"
    run_root = escaped if key == "RUN_ROOT" else final_root / "run"
    result = lint_staged(fixture_root, local_root / "job.sbatch", task_root, run_root, local_root, final_root)
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"{key} resolves under --shared-env-prefix" in result.stdout


def test_staging_mapped_root_disjoint_from_shared_env_prefix_accepted(fixture_root: Path) -> None:
    """Disjoint control for the escape test above: the same LOCAL=FINAL
    staging mapping, but the mapped destination stays inside the allowed
    root and disjoint from --shared-env-prefix, so it passes."""
    local_root, final_root = shared_env_staging_symlink_fixture(fixture_root)
    (local_root / "elsewhere").mkdir()
    (final_root / "elsewhere").mkdir()
    task_root = local_root / "elsewhere" / "task"
    run_root = final_root / "run"
    result = lint_staged(fixture_root, local_root / "job.sbatch", task_root, run_root, local_root, final_root)
    assert result.returncode == 0, result.stdout + result.stderr


# --- Check 7 (T3, review-round-1 F3): the allowed-root interpreter exception
# is removed; PYTHON under an allowed root but outside --shared-env-prefix
# (e.g. a previous job's overlay) must be rejected in a dry run. ------------


def test_dry_run_python_under_allowed_root_not_shared_env_rejected(fixture_root: Path) -> None:
    base = fixture_root
    task_root = base / "stage" / "task"
    run_root = base / "stage" / "run"
    overlay_python = base / "stage" / "run" / "overlay" / "bin" / "python"
    overlay_python.parent.mkdir(parents=True)
    overlay_python.write_text("#!/bin/sh\n")
    overlay_python.chmod(overlay_python.stat().st_mode | stat.S_IEXEC)
    manifest = base / "stage" / "goodpy2.manifest"
    sbatch = base / "stage" / "goodpy2.sbatch"
    write_manifest(manifest, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=1, python=overlay_python)
    write_sbatch(sbatch, sha=SHA, job_name=JOB_NAME, task_root=task_root, run_root=run_root, dry_run=1, python=overlay_python)
    result = run_lint(base, sbatch, manifest)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "dry-run PYTHON is not under --shared-env-prefix" in result.stdout


def test_no_errexit_unsafe_arithmetic_statements() -> None:
    """Cannon job 59624204: under `set -e`, bash >= 4 aborts on an `((...))`
    command that evaluates to 0 (e.g. `((i++))` with i=0); bash 3.2 does
    not, so local runs masked it. Increments/assignments must use
    `x=$((...))`; `for ((...))` headers and tested conditions are safe."""
    import re

    risky = re.compile(r"(?<![$\w])\(\((?P<expr>[^()]*)\)\)")
    offenders = []
    for script in sorted(LINTER.parent.glob("*.sh")):
        for lineno, line in enumerate(script.read_text().splitlines(), 1):
            code = line.split(" #", 1)[0]
            if re.match(r"\s*#", code) or re.search(r"\bfor\s*\(\(", code):
                continue
            for match in risky.finditer(code):
                expr = match.group("expr")
                if re.search(r"\+\+|--|(?<![<>=!])=(?!=)", expr):
                    offenders.append(f"{script.name}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
