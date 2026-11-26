"""Local, no-SSH contract tests for the devsuite renderer."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
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


def git_bytes(repo: Path, *args: str) -> bytes:
    """Return bytes from a git command without decoding harness content."""

    return subprocess.check_output(["git", *args], cwd=repo)


def git_text(repo: Path, *args: str) -> str:
    """Return a stripped text value from a git command."""

    return git_bytes(repo, *args).decode().strip()


def test_local_render_uses_exact_sha_harness() -> None:
    result = render(scheduled=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "harness_source=scripts/cannon/executor-harness" in result.stdout
    assert f"harness_source_sha={SHA}" in result.stdout


def test_local_render_uses_operator_job_names(tmp_path: Path) -> None:
    result = subprocess.run(
        [str(LAUNCHER), "--local-dry-run", SHA],
        cwd=REPO_ROOT,
        env={**os.environ, "DEVSUITE_LOCAL_STAGE_DIR": str(tmp_path / "staged")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    root = (tmp_path / "staged" / "root" / "root.sbatch").read_text(encoding="utf-8")
    executor = (tmp_path / "staged" / "executor" / "executor.sbatch").read_text(encoding="utf-8")
    assert f"#SBATCH --job-name=devsuite-root-s1-{SHA[:8]}" in root
    assert f"#SBATCH --job-name=devsuite-exec-s1-{SHA[:8]}" in executor


def test_executor_harness_dry_run_accepts_sha8_job_name(tmp_path: Path) -> None:
    """The staged executor harness enforces the launcher’s SHA8 job name."""

    harness = tmp_path / "harness"
    shutil.copytree(REPO_ROOT / "scripts/cannon/executor-harness", harness)
    (harness / "repo.bundle").write_bytes(b"test bundle")
    (harness / "mutants.json").write_text('{"mutants": []}\n', encoding="utf-8")
    node_tmp = tmp_path / "node"
    for interpreter in (node_tmp / "fake-root/bin/python", node_tmp / "fake-executor/bin/python"):
        interpreter.parent.mkdir(parents=True, exist_ok=True)
        interpreter.symlink_to(sys.executable)

    head_sha = "0123456789abcdef" * 2 + "01234567"
    run_root = node_tmp / "run"
    harness_env = {
        **os.environ,
        "DRY_RUN": "1",
        "DRY_NODE_TMP": str(node_tmp),
        "HARNESS_ROOT": str(harness),
        "HEAD_SHA": head_sha,
        "JOB_NAME": f"devsuite-exec-s1-{head_sha[:8]}",
        "LOCAL_FAKE": "1",
        "LOCAL_FAKE_ROOT": str(tmp_path),
        "MUTANT_SPEC": str(harness / "mutants.json"),
        "RUN_ROOT": str(run_root),
        "SLURM_JOB_ID": "99999999",
        "SLURM_JOB_PARTITION": "test",
        "TASK_ROOT": str(harness),
        "USER": os.environ.get("USER", "test-user"),
    }
    result = subprocess.run(
        [str(harness / "job.sbatch")],
        cwd=harness,
        env=harness_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY_RUN=1 PASS" in (run_root / "evidence/driver.log").read_text(encoding="utf-8")

    rejected = subprocess.run(
        [str(harness / "job.sbatch")],
        cwd=harness,
        env={**harness_env, "JOB_NAME": "devsuite-exec-s1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "PRECHECK_FAILED JOB_NAME must contain" in rejected.stderr


def test_local_render_stages_harness_from_sha_not_any_other_ref(tmp_path: Path) -> None:
    """Inspect staged bytes when the SHA is neither HEAD nor the worktree."""

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--no-hardlinks", str(REPO_ROOT), str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "-C", str(clone), "config", "user.name", "renderer-test"], check=True)
    subprocess.run(
        ["git", "-C", str(clone), "config", "user.email", "renderer-test@example.invalid"],
        check=True,
    )
    harness_rel = Path("scripts/cannon/executor-harness/helpers/validate_inputs.sh")
    harness_path = clone / harness_rel
    original = harness_path.read_bytes()
    base_head = git_text(clone, "rev-parse", "HEAD")

    harness_path.write_bytes(original + b"\n# target SHA marker\n")
    subprocess.run(["git", "-C", str(clone), "add", str(harness_rel)], check=True)
    subprocess.run(["git", "-C", str(clone), "commit", "-m", "test target harness bytes"], check=True, capture_output=True)
    target_sha = git_text(clone, "rev-parse", "HEAD")

    subprocess.run(["git", "-C", str(clone), "reset", "--hard", base_head], check=True, capture_output=True)
    harness_path.write_bytes(original + b"\n# origin dev marker\n")
    subprocess.run(["git", "-C", str(clone), "add", str(harness_rel)], check=True)
    subprocess.run(["git", "-C", str(clone), "commit", "-m", "test origin dev harness bytes"], check=True, capture_output=True)
    origin_dev_sha = git_text(clone, "rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(clone), "reset", "--hard", base_head], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(clone), "update-ref", "refs/remotes/origin/dev", origin_dev_sha],
        check=True,
    )
    # Mutation arms edit the source checkout after its committed HEAD is set.
    # Copy that current launcher after the final reset so the throwaway clone
    # really exercises the deliberate edit instead of a clean committed launcher.
    clone_launcher = clone / "scripts" / "cannon" / "devsuite" / "submit_devsuite.sh"
    clone_launcher.write_bytes(LAUNCHER.read_bytes())
    harness_path.write_bytes(original + b"\n# working tree marker\n")

    assert target_sha not in {base_head, origin_dev_sha}
    stage_export = tmp_path / "staged"
    assert clone_launcher.read_bytes() == LAUNCHER.read_bytes()
    result = subprocess.run(
        [str(clone / "scripts/cannon/devsuite/submit_devsuite.sh"), "--local-dry-run", target_sha],
        cwd=clone,
        env={**os.environ, "DEVSUITE_LOCAL_STAGE_DIR": str(stage_export)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    staged = (stage_export / "root" / harness_rel).read_bytes()
    assert staged == git_bytes(clone, "show", f"{target_sha}:{harness_rel}")
    assert staged != git_bytes(clone, "show", f"{base_head}:{harness_rel}")
    assert staged != git_bytes(clone, "show", f"{origin_dev_sha}:{harness_rel}")
    assert staged != harness_path.read_bytes()


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
