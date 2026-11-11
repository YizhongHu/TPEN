"""Contract tests for ``scripts/cannon/build-shared-env.sh``.

Drives the builder as a subprocess against a small fixture tree built under
``tmp_path`` with a stubbed ``uv`` executable -- no Slurm, no cluster access,
no real dependency resolution (mirrors ``test_cannon_submit_lint.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER = REPO_ROOT / "scripts" / "cannon" / "build-shared-env.sh"
REAL_POLICY_FILE = REPO_ROOT / "scripts" / "cannon" / "shared-env-policy.json"

PROJECT = "tpen"
EXTRA = "cpu"
PYTHON_VERSION = "3.12"


def make_checkout(
    tmp_path: Path,
    *,
    policy_body: str | None = '{"MAX_SHARED_ENV_DIRS": 20}\n',
    dirty: bool = False,
) -> Path:
    repo = tmp_path / "checkout"
    repo.mkdir()
    env = dict(os.environ)
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "test"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "test@example.com"
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    (repo / "uv.lock").write_text("locked-dependency-contents\n")
    (repo / "scripts" / "cannon").mkdir(parents=True)
    if policy_body is not None:
        (repo / "scripts" / "cannon" / "shared-env-policy.json").write_text(policy_body)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=env)
    if dirty:
        (repo / "uv.lock").write_text("locked-dependency-contents-changed\n")
    return repo


def env_name_for(checkout: Path, *, project: str = PROJECT, extra: str = EXTRA, python_version: str = PYTHON_VERSION) -> str:
    lock_sha256 = hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest()
    return f"{project}-{lock_sha256[:12]}-{extra}-py{python_version}"


def make_uv_stub(tmp_path: Path) -> tuple[Path, Path]:
    uv = tmp_path / "uv-stub"
    log = tmp_path / "uv-calls.log"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{log}"\n'
        'if [[ "$1" == "--version" ]]; then echo "uv 9.9.9-stub"; exit 0; fi\n'
        'if [[ "$1" == "sync" ]]; then mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; : > "$UV_PROJECT_ENVIRONMENT/bin/python"; exit 0; fi\n'
        "exit 1\n"
    )
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    return uv, log


def make_filler_dirs(root: Path, count: int, *, names: list[str] | None = None) -> None:
    names = names or []
    for i in range(count):
        name = names[i] if i < len(names) else f"filler-{i}"
        (root / name).mkdir(parents=True, exist_ok=True)


def run_builder(
    *,
    envs_root: Path,
    checkout: Path,
    uv_bin: Path,
    cache_dir: Path,
    project: str = PROJECT,
    extra: str = EXTRA,
    python_version: str = PYTHON_VERSION,
    extra_env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(cache_dir)
    if extra_env:
        env.update(extra_env)
    cmd = [
        "bash",
        str(BUILDER),
        "--envs-root", str(envs_root),
        "--checkout", str(checkout),
        "--project", project,
        "--extra", extra,
        "--python", python_version,
        "--uv-bin", str(uv_bin),
        *(extra_args or []),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)


@pytest.fixture()
def envs_root(tmp_path: Path) -> Path:
    root = tmp_path / "envs"
    root.mkdir()
    return root


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "uv-cache"
    d.mkdir()
    return d


def test_committed_policy_value_is_20() -> None:
    """Raising the cap requires changing this committed file -- and this
    test -- together, visibly, in a reviewed PR."""
    assert json.loads(REAL_POLICY_FILE.read_text()) == {"MAX_SHARED_ENV_DIRS": 20}


def test_refuses_at_cap_with_malformed_dot_partial_dirs_counted(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    make_filler_dirs(
        envs_root,
        20,
        names=[
            "normal-one", ".hidden-dotdir", "tpen-deadbeefcafe-cpu-py3.11",
            "partial-no-complete", "malformed name with spaces", "..weird..",
        ],
    )
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert "cap reached" in result.stderr
    assert "count=20" in result.stderr
    assert "cap=20" in result.stderr
    assert not (envs_root / target).exists()
    assert not uv_log.exists() or uv_log.read_text() == ""


def test_success_at_19(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    make_filler_dirs(envs_root, 19)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BUILT" in result.stdout
    assert (envs_root / target / ".complete").is_file()
    manifest = (envs_root / target / ".complete").read_text()
    assert "uv_lock_sha256=" in manifest and "python_version=3.12" in manifest
    assert "sync" in uv_log.read_text()


def test_env_var_override_attempt_does_not_change_outcome(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    make_filler_dirs(envs_root, 20)
    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir,
        extra_env={"MAX_SHARED_ENV_DIRS": "1000"},
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "cap reached" in result.stderr
    assert "cap=20" in result.stderr


def test_flag_override_attempt_is_rejected(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, _ = make_uv_stub(tmp_path)
    make_filler_dirs(envs_root, 19)
    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir,
        extra_args=["--max-shared-env-dirs", "1000"],
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "unknown argument" in result.stderr


def test_missing_policy_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path, policy_body=None)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert "policy" in result.stderr and "missing" in result.stderr
    assert not uv_log.exists()


@pytest.mark.parametrize(
    "policy_body",
    [
        '{"MAX_SHARED_ENV_DIRS": "twenty"}\n',
        "{}\n",
        "not json at all\n",
        '{"MAX_SHARED_ENV_DIRS": -1}\n',
        '{"MAX_SHARED_ENV_DIRS": 3.5}\n',
    ],
)
def test_malformed_policy_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path, policy_body: str) -> None:
    checkout = make_checkout(tmp_path, policy_body=policy_body)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert "non-negative integer" in result.stderr
    assert not uv_log.exists()


def test_dirty_checkout_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path, dirty=True)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert "dirty" in result.stderr
    assert not uv_log.exists()


def test_complete_env_reused_at_cap(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target_name = env_name_for(checkout)
    make_filler_dirs(envs_root, 19)
    target_dir = envs_root / target_name
    target_dir.mkdir()
    (target_dir / ".complete").write_text("head=deadbeef\n")
    # 19 fillers + the pre-existing complete target == cap (20).
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REUSE" in result.stdout
    assert "count=20" in result.stdout
    assert not uv_log.exists() or uv_log.read_text() == ""


def test_partial_env_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target_name = env_name_for(checkout)
    make_filler_dirs(envs_root, 3)
    (envs_root / target_name).mkdir()
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert "partial" in result.stderr and "needs disposition" in result.stderr
    assert not uv_log.exists()


def test_noncanonical_root_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    link = tmp_path / "envs-link"
    link.symlink_to(envs_root)
    result = run_builder(envs_root=link, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PRECHECK_FAILED" in result.stderr
    assert "canonical" in result.stderr
    assert not uv_log.exists()


def test_no_mkdir_and_no_uv_call_on_cap_refusal(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    before_dirs = {p.name for p in envs_root.iterdir() if p.is_dir()}
    make_filler_dirs(envs_root, 20)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0
    # The root-level flock file (not a directory, and not counted toward the
    # cap) is expected; only directory entries must be limited to the fillers.
    after_dirs = {p.name for p in envs_root.iterdir() if p.is_dir()}
    assert target not in after_dirs
    assert after_dirs - before_dirs == {f"filler-{i}" for i in range(20)}
    assert not uv_log.exists()
