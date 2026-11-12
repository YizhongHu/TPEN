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
    (repo / "pyproject.toml").write_text("[project]\nname = \"fixture\"\n")
    (repo / "uv.lock").write_text("locked-dependency-contents\n")
    (repo / "scripts" / "cannon").mkdir(parents=True)
    if policy_body is not None:
        (repo / "scripts" / "cannon" / "shared-env-policy.json").write_text(policy_body)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=env)
    if dirty:
        (repo / "uv.lock").write_text("locked-dependency-contents-changed\n")
    return repo


def env_name_for(
    checkout: Path,
    *,
    project: str = PROJECT,
    extra: str = EXTRA,
    python_version: str = PYTHON_VERSION,
    project_dir: str = ".",
) -> str:
    lock_sha256 = hashlib.sha256((checkout / project_dir / "uv.lock").read_bytes()).hexdigest()
    return f"{project}-{lock_sha256[:12]}-{extra}-py{python_version}"


PY_FULL_VERSION_STUB = "3.12.7"


def _write_python_stub(bin_dir: Path, *, version: str | None = PY_FULL_VERSION_STUB) -> None:
    """Write an executable python stub into ``bin_dir``.

    ``version=None`` writes a non-executable placeholder (simulates a sync
    that produced no usable interpreter, so ``--version`` fails).
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    python = bin_dir / "python"
    if version is None:
        python.write_text("")
        return
    python.write_text(f'#!/usr/bin/env bash\necho "Python {version}"\n')
    python.chmod(python.stat().st_mode | stat.S_IEXEC)


def make_uv_stub(tmp_path: Path, *, python_version: str | None = PY_FULL_VERSION_STUB) -> tuple[Path, Path]:
    uv = tmp_path / "uv-stub"
    log = tmp_path / "uv-calls.log"
    marker = "__PYVER__" if python_version is None else python_version
    uv.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "CWD=$PWD ARGS=$*" >> "{log}"\n'
        'if [[ "$1" == "--version" ]]; then echo "uv 9.9.9-stub"; exit 0; fi\n'
        'if [[ "$1" == "sync" ]]; then\n'
        '  mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"\n'
        f'  if [[ "{marker}" == "__PYVER__" ]]; then\n'
        '    : > "$UV_PROJECT_ENVIRONMENT/bin/python"\n'
        "  else\n"
        f'    printf \'#!/usr/bin/env bash\\necho "Python {marker}"\\n\' > "$UV_PROJECT_ENVIRONMENT/bin/python"\n'
        '    chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"\n'
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    return uv, log


def make_failing_sync_uv_stub(tmp_path: Path) -> tuple[Path, Path]:
    """uv stub whose ``sync`` creates a partial target then exits nonzero."""
    uv = tmp_path / "uv-stub-fail"
    log = tmp_path / "uv-calls-fail.log"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{log}"\n'
        'if [[ "$1" == "--version" ]]; then echo "uv 9.9.9-stub"; exit 0; fi\n'
        'if [[ "$1" == "sync" ]]; then mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; exit 7; fi\n'
        "exit 1\n"
    )
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    return uv, log


def make_valid_manifest(
    checkout: Path,
    *,
    head: str = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    lock_sha256: str | None = None,
    project_dir: str | None = ".",
) -> str:
    """A ``.complete`` body the builder's manifest_is_valid accepts.

    ``head`` intentionally defaults to a value that need not equal the
    checkout's actual HEAD: dependency identity (the lock hash), not
    history, is what reuse validates. ``project_dir=None`` omits the
    seventh field entirely, producing the six-field manifest shape written
    before --project-dir existed (still required to be trusted for reuse).
    """
    lock_dir = checkout if project_dir is None else checkout / project_dir
    lock_sha256 = lock_sha256 or hashlib.sha256((lock_dir / "uv.lock").read_bytes()).hexdigest()
    body = (
        f"head={head}\n"
        f"uv_lock_sha256={lock_sha256}\n"
        "uv_version=uv 9.9.9-stub\n"
        f"python_version={PY_FULL_VERSION_STUB}\n"
        "python_request=3.12\n"
        "built_utc=2026-01-01T00:00:00Z\n"
    )
    if project_dir is not None:
        body += f"project_dir={project_dir}\n"
    return body


def add_subproject(
    checkout: Path,
    rel: str,
    *,
    lock_contents: bytes = b"sub-locked-dependency-contents\n",
    include_pyproject: bool = True,
    include_lock: bool = True,
) -> Path:
    """Add and commit a subdirectory with its own pyproject.toml/uv.lock,
    for --project-dir tests. Committed (not left dirty) so the builder's
    clean-checkout precheck still passes."""
    sub = checkout / rel
    sub.mkdir(parents=True, exist_ok=True)
    if include_pyproject:
        (sub / "pyproject.toml").write_text('[project]\nname = "sub"\n')
    if include_lock:
        (sub / "uv.lock").write_bytes(lock_contents)
    env = dict(os.environ)
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "test"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "test@example.com"
    subprocess.run(["git", "add", "-A"], cwd=checkout, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "add subproject"], cwd=checkout, check=True, env=env)
    return sub


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
    project_dir: str | None = None,
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
        *(["--project-dir", project_dir] if project_dir is not None else []),
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
    assert "uv_lock_sha256=" in manifest
    assert f"python_version={PY_FULL_VERSION_STUB}" in manifest
    assert "python_request=3.12" in manifest
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
    # Dependency identity (the lock hash) is what reuse checks -- seed a
    # manifest whose recorded HEAD deliberately differs from this
    # checkout's actual HEAD to assert that is not required to match.
    (target_dir / ".complete").write_text(make_valid_manifest(checkout))
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


# --- T1 (F1: target escape via --project/--extra, and symlink reuse) ------


@pytest.mark.parametrize("bad_project", ["../escape", "a/b", "..", "/abs", "a/../b"])
def test_project_traversal_is_rejected(tmp_path: Path, envs_root: Path, cache_dir: Path, bad_project: str) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    outside_before = {p.name for p in tmp_path.iterdir()}
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, project=bad_project)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PRECHECK_FAILED" in result.stderr
    assert not uv_log.exists()
    assert {p.name for p in tmp_path.iterdir()} == outside_before
    assert list(envs_root.iterdir()) == []


@pytest.mark.parametrize("bad_extra", ["../escape", "a/b", ".."])
def test_extra_traversal_is_rejected(tmp_path: Path, envs_root: Path, cache_dir: Path, bad_extra: str) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, extra=bad_extra)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PRECHECK_FAILED" in result.stderr
    assert not uv_log.exists()


def test_ordinary_project_and_extra_still_build(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (envs_root / target / ".complete").is_file()
    assert "sync" in uv_log.read_text()


def test_target_symlink_to_external_complete_dir_is_refused(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target_name = env_name_for(checkout)
    external = tmp_path / "external-complete-env"
    external.mkdir()
    (external / ".complete").write_text(make_valid_manifest(checkout))
    make_filler_dirs(envs_root, 19)
    (envs_root / target_name).symlink_to(external, target_is_directory=True)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert "symlink" in result.stderr
    assert not uv_log.exists()
    assert not (external / "mutated-by-test").exists()


# --- T2 (F2: inherited SHELLOPTS=noglob must not defeat the census) --------


def test_inherited_shellopts_noglob_does_not_bypass_cap(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    make_filler_dirs(envs_root, 20)
    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir,
        extra_env={"SHELLOPTS": "noglob"},
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "cap reached" in result.stderr
    assert "count=20" in result.stderr
    assert "cap=20" in result.stderr
    assert not (envs_root / target).exists()
    assert not uv_log.exists()


def test_control_without_noglob_also_refuses_at_cap(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    make_filler_dirs(envs_root, 20)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "cap reached" in result.stderr
    assert not uv_log.exists()


def test_noglob_with_19_fillers_still_succeeds(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    make_filler_dirs(envs_root, 19)
    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir,
        extra_env={"SHELLOPTS": "noglob"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (envs_root / target / ".complete").is_file()
    assert "sync" in uv_log.read_text()


# --- T3 (F3: whole-document policy parsing) --------------------------------


@pytest.mark.parametrize(
    "policy_body",
    [
        '{"MAX_SHARED_ENV_DIRS": 20\n',  # missing closing brace
        '{"MAX_SHARED_ENV_DIRS": 20}trailing garbage\n',  # trailing garbage
        '{"MAX_SHARED_ENV_DIRS": 1000, "MAX_SHARED_ENV_DIRS": 20}\n',  # duplicate key
        '{"outer": {"MAX_SHARED_ENV_DIRS": 20}}\n',  # nested-only key
        '{"MAX_SHARED_ENV_DIRS": true}\n',  # boolean
        '{"MAX_SHARED_ENV_DIRS": 020}\n',  # leading zero
        '{"MAX_SHARED_ENV_DIRS": 10000}\n',  # out-of-range (>3 digits)
    ],
)
def test_structurally_malformed_policy_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path, policy_body: str) -> None:
    checkout = make_checkout(tmp_path, policy_body=policy_body)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert "non-negative integer" in result.stderr
    assert not uv_log.exists()


def test_whitespace_formatted_valid_policy_still_green(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path, policy_body='{\n  "MAX_SHARED_ENV_DIRS"   :   20\n}\n')
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (envs_root / target / ".complete").is_file()
    assert "sync" in uv_log.read_text()


# --- T4 (F4: atomic + validated manifest; dependency, not HEAD, identity) --


def test_empty_complete_manifest_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target_name = env_name_for(checkout)
    target_dir = envs_root / target_name
    target_dir.mkdir()
    (target_dir / ".complete").write_text("")
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert "needs disposition" in result.stderr
    assert not uv_log.exists()


def test_truncated_complete_manifest_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target_name = env_name_for(checkout)
    target_dir = envs_root / target_name
    target_dir.mkdir()
    (target_dir / ".complete").write_text("head=deadbeef\nuv_lock_sha256=onlythis\n")
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert "needs disposition" in result.stderr
    assert not uv_log.exists()


def test_mismatched_lock_hash_manifest_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target_name = env_name_for(checkout)
    target_dir = envs_root / target_name
    target_dir.mkdir()
    (target_dir / ".complete").write_text(make_valid_manifest(checkout, lock_sha256="0" * 64))
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert "needs disposition" in result.stderr
    assert not uv_log.exists()


def test_valid_manifest_with_different_head_is_reused(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    """Dependency identity (the lock hash) governs reuse, not build HEAD."""
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target_name = env_name_for(checkout)
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True, capture_output=True, text=True,
    ).stdout.strip()
    target_dir = envs_root / target_name
    target_dir.mkdir()
    assert make_valid_manifest(checkout).splitlines()[0] != f"head={actual_head}"
    (target_dir / ".complete").write_text(make_valid_manifest(checkout))
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REUSE" in result.stdout
    assert not uv_log.exists()


def test_manifest_write_is_atomic_no_tmp_file_left(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    target_dir = envs_root / target
    assert (target_dir / ".complete").is_file()
    assert list(target_dir.glob(".complete.tmp.*")) == []


# --- T5 (F5: recorded python_version is the built interpreter, not the selector) --


def test_recorded_python_version_is_actual_interpreter_not_selector(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path, python_version="3.12.9")
    target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, python_version="3.12")
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = (envs_root / target / ".complete").read_text()
    assert "python_version=3.12.9" in manifest
    assert "python_request=3.12" in manifest


def test_interpreter_query_failure_blocks_publication(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path, python_version=None)
    target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert not (envs_root / target / ".complete").exists()
    assert (envs_root / target).is_dir()


def test_malformed_version_metadata_exits_zero_but_blocks_publication(
    tmp_path: Path, envs_root: Path, cache_dir: Path
) -> None:
    """Killer for mutant m6 (drop the pre-publication manifest_is_valid call;
    A1 review c6ee8bcc, declined as fail-safe for that slice with a follow-up
    to add this test, carried to this item's carried-follow-ups note): unlike
    test_interpreter_query_failure_blocks_publication (the query itself
    fails), the metadata query here exits 0 with a non-empty but malformed
    version string, so only the strict manifest_is_valid check -- not the
    `-n` emptiness check -- can catch it before publication."""
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path, python_version="not-a-version")
    target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    assert not (envs_root / target / ".complete").exists()
    assert (envs_root / target).is_dir()


# --- T6 (failed builds remain counted; expected green) ---------------------


def test_failed_build_remains_counted_and_blocks_other_targets(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    failing_uv, failing_log = make_failing_sync_uv_stub(tmp_path)
    make_filler_dirs(envs_root, 19)

    failed_target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=failing_uv, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert (envs_root / failed_target).is_dir()
    assert not (envs_root / failed_target / ".complete").exists()
    assert "sync" in failing_log.read_text()

    # 19 fillers + the failed-but-counted target == cap (20): a different
    # target must be refused without another sync call.
    other_uv, other_log = make_uv_stub(tmp_path)
    result2 = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=other_uv, cache_dir=cache_dir, python_version="3.13",
    )
    assert result2.returncode != 0, result2.stdout + result2.stderr
    assert "cap reached" in result2.stderr
    assert "count=20" in result2.stderr
    assert not other_log.exists()

    # Retrying the same failed target refuses as partial, also no new sync.
    result3 = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=other_uv, cache_dir=cache_dir)
    assert result3.returncode != 0, result3.stdout + result3.stderr
    assert "needs disposition" in result3.stderr
    assert not other_log.exists()


# --- T8 (directory census: symlink to external dir is counted) ------------


def test_symlink_filler_counts_toward_cap(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    external_dir = tmp_path / "external-unrelated-dir"
    external_dir.mkdir()
    make_filler_dirs(envs_root, 19)
    (envs_root / "symlinked-filler").symlink_to(external_dir, target_is_directory=True)
    # 19 real fillers + 1 symlinked dir == cap (20).
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "cap reached" in result.stderr
    assert "count=20" in result.stderr
    assert not (envs_root / target).exists()
    assert not uv_log.exists()


# --- DT1 (D1: literal '*' in real directory names must count, never abort,
#         and inherited failglob must not break an empty-root first build) --


def test_literal_asterisk_directory_names_count_not_abort(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    make_filler_dirs(envs_root, 17)
    (envs_root / "old*partial").mkdir()
    (envs_root / "*").mkdir()
    # 17 ordinary fillers + 2 asterisk-containing real dirs == 19; a new build succeeds.
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BUILT" in result.stdout
    assert (envs_root / target / ".complete").is_file()
    dirs = {p.name for p in envs_root.iterdir() if p.is_dir()}
    assert dirs == {f"filler-{i}" for i in range(17)} | {"old*partial", "*", target}


def test_literal_asterisk_directory_names_at_cap_give_normal_diagnostic(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    make_filler_dirs(envs_root, 18)
    (envs_root / "old*partial").mkdir()
    (envs_root / "*").mkdir()
    # 18 ordinary fillers + 2 asterisk-containing real dirs == cap (20).
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "cap reached" in result.stderr
    assert "count=20" in result.stderr
    assert "cap=20" in result.stderr
    assert not uv_log.exists()


def test_valid_target_reused_despite_asterisk_sibling(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target_name = env_name_for(checkout)
    make_filler_dirs(envs_root, 18)
    (envs_root / "old*partial").mkdir()
    target_dir = envs_root / target_name
    target_dir.mkdir()
    (target_dir / ".complete").write_text(make_valid_manifest(checkout))
    # 18 fillers + asterisk dir + the valid target == cap (20); must still reuse.
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REUSE" in result.stdout
    assert not uv_log.exists() or uv_log.read_text() == ""


def test_empty_root_with_inherited_failglob_still_builds(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir,
        extra_env={"BASHOPTS": "failglob"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (envs_root / target / ".complete").is_file()
    assert "sync" in uv_log.read_text()


def test_empty_root_without_failglob_control_still_builds(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (envs_root / target / ".complete").is_file()


# --- DT2 (D2: a real producer/consumer manifest late-truncated even past
#         every field must be rejected; round-trip reuse control) ----------


def test_real_manifest_round_trips_unchanged_on_reuse(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest_path = envs_root / target / ".complete"
    original_bytes = manifest_path.read_bytes()
    calls_before = uv_log.read_text()

    result2 = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result2.returncode == 0, result2.stdout + result2.stderr
    assert "REUSE" in result2.stdout
    assert manifest_path.read_bytes() == original_bytes
    assert uv_log.read_text() == calls_before


@pytest.mark.parametrize("trailing_newline", [True, False])
def test_real_manifest_late_truncated_at_built_utc_refuses(
    tmp_path: Path, envs_root: Path, cache_dir: Path, trailing_newline: bool,
) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest_path = envs_root / target / ".complete"
    lines = manifest_path.read_text().splitlines()
    built_utc_index = next(i for i, line in enumerate(lines) if line.startswith("built_utc="))
    # Simulate a write that stopped mid-value for built_utc: everything after
    # it (now including project_dir) was never written.
    truncated = "\n".join(lines[:built_utc_index] + ["built_utc=2"])
    if trailing_newline:
        truncated += "\n"
    manifest_path.write_text(truncated)
    calls_before = uv_log.read_text()

    result2 = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result2.returncode != 0, result2.stdout + result2.stderr
    assert "REFUSE" in result2.stderr
    assert "needs disposition" in result2.stderr
    assert uv_log.read_text() == calls_before


# --- DT3 (m7: a writer that fails after a partial manifest write must not
#         publish -- the real temp+rename contract, not just final content) --


def test_manifest_write_failure_leaves_no_complete_and_blocks_reuse(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    """A ``cat`` that forwards the policy-file read unchanged but truncates
    the manifest heredoc write at ``built_utc=2`` and exits nonzero must
    leave no ``.complete`` under the current temp+rename implementation; a
    direct-write implementation would instead publish the truncated
    receipt and make it reusable."""
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)

    fake_bin = tmp_path / "fake-bin-cat"
    fake_bin.mkdir()
    fake_cat = fake_bin / "cat"
    fake_cat.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--" ]]; then exec /bin/cat "$@"; fi\n'
        'input="$(/bin/cat)"\n'
        'printf \'%s\\n\' "${input%%built_utc=*}built_utc=2"\n'
        "exit 1\n"
    )
    fake_cat.chmod(fake_cat.stat().st_mode | stat.S_IEXEC)

    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir,
        extra_env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )
    assert result.returncode != 0, result.stdout + result.stderr
    target_dir = envs_root / target
    assert target_dir.is_dir()
    assert not (target_dir / ".complete").exists()
    assert "sync" in uv_log.read_text()

    calls_before = uv_log.read_text()
    result2 = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result2.returncode != 0, result2.stdout + result2.stderr
    assert "needs disposition" in result2.stderr
    assert uv_log.read_text() == calls_before


# --- DT4 (D2: a failed metadata query (timestamp) must block publication,
#         not be masked by an otherwise-successful manifest write) --------


def test_timestamp_query_failure_blocks_publication(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)

    fake_bin = tmp_path / "fake-bin-date"
    fake_bin.mkdir()
    fake_date = fake_bin / "date"
    fake_date.write_text("#!/usr/bin/env bash\nexit 1\n")
    fake_date.chmod(fake_date.stat().st_mode | stat.S_IEXEC)

    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir,
        extra_env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "REFUSE" in result.stderr
    target_dir = envs_root / target
    assert target_dir.is_dir()
    assert not (target_dir / ".complete").exists()
    assert "sync" in uv_log.read_text()


def test_timestamp_query_success_control_still_builds(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (envs_root / target / ".complete").is_file()


# --- A3 (--project-dir and --extra none) -----------------------------------


def test_project_dir_hashes_subproject_lock_and_syncs_there(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    sub = add_subproject(checkout, "executor", lock_contents=b"executor-locked-contents\n")
    uv_bin, uv_log = make_uv_stub(tmp_path)
    root_target = env_name_for(checkout)
    sub_target = env_name_for(checkout, project_dir="executor")
    assert root_target != sub_target, "fixture sanity: root and sub lock hashes must differ"

    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, project_dir="executor",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BUILT" in result.stdout
    manifest = (envs_root / sub_target / ".complete").read_text()
    expected_lock_sha256 = hashlib.sha256((sub / "uv.lock").read_bytes()).hexdigest()
    assert f"uv_lock_sha256={expected_lock_sha256}" in manifest
    assert "project_dir=executor" in manifest
    assert not (envs_root / root_target).exists()
    log_text = uv_log.read_text()
    assert "sync" in log_text
    assert f"CWD={sub}" in log_text


def test_project_dir_default_is_checkout_root(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = (envs_root / target / ".complete").read_text()
    assert "project_dir=.\n" in manifest
    assert f"CWD={checkout}" in uv_log.read_text()


@pytest.mark.parametrize("bad_project_dir", ["../escape", "..", "/abs", "a/../b", "a//b", "a/"])
def test_project_dir_traversal_or_absolute_is_rejected(
    tmp_path: Path, envs_root: Path, cache_dir: Path, bad_project_dir: str,
) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    outside_before = {p.name for p in tmp_path.iterdir()}
    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, project_dir=bad_project_dir,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PRECHECK_FAILED" in result.stderr
    assert not uv_log.exists()
    assert {p.name for p in tmp_path.iterdir()} == outside_before
    assert list(envs_root.iterdir()) == []


def test_project_dir_symlink_escape_is_rejected(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    outside = tmp_path / "outside-of-checkout"
    outside.mkdir()
    (outside / "pyproject.toml").write_text('[project]\nname = "outside"\n')
    (outside / "uv.lock").write_bytes(b"outside-lock\n")
    link = checkout / "escape_link"
    link.symlink_to(outside, target_is_directory=True)
    env = dict(os.environ)
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "test"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "test@example.com"
    subprocess.run(["git", "add", "-A"], cwd=checkout, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "add escape symlink"], cwd=checkout, check=True, env=env)
    uv_bin, uv_log = make_uv_stub(tmp_path)

    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, project_dir="escape_link",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PRECHECK_FAILED" in result.stderr
    assert "escapes" in result.stderr
    assert not uv_log.exists()
    assert not (outside / "mutated-by-test").exists()


def test_project_dir_missing_pyproject_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    add_subproject(checkout, "no_pyproject", include_pyproject=False)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, project_dir="no_pyproject",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PRECHECK_FAILED" in result.stderr
    assert "pyproject.toml" in result.stderr
    assert not uv_log.exists()


def test_project_dir_missing_uv_lock_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    add_subproject(checkout, "no_lock", include_lock=False)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, project_dir="no_lock",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PRECHECK_FAILED" in result.stderr
    assert "uv.lock" in result.stderr
    assert not uv_log.exists()


def test_project_dir_nonexistent_refuses(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, project_dir="does-not-exist",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "PRECHECK_FAILED" in result.stderr
    assert not uv_log.exists()


def test_policy_still_read_from_checkout_root_with_project_dir(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    """A different (and invalid) policy file sitting inside the project dir
    must never be consulted -- the cap always comes from
    <checkout>/scripts/cannon/shared-env-policy.json."""
    checkout = make_checkout(tmp_path)
    sub = add_subproject(checkout, "executor")
    (sub / "scripts").mkdir(parents=True)
    (sub / "scripts" / "cannon").mkdir()
    (sub / "scripts" / "cannon" / "shared-env-policy.json").write_text('{"MAX_SHARED_ENV_DIRS": "not-a-number"}\n')
    env = dict(os.environ)
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "test"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "test@example.com"
    subprocess.run(["git", "add", "-A"], cwd=checkout, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "add decoy policy"], cwd=checkout, check=True, env=env)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    sub_target = env_name_for(checkout, project_dir="executor")

    result = run_builder(
        envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, project_dir="executor",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BUILT" in result.stdout
    assert (envs_root / sub_target / ".complete").is_file()


def test_extra_none_omits_extra_flag_from_uv_sync(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target = env_name_for(checkout, extra="none")
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, extra="none")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BUILT" in result.stdout
    assert f"{PROJECT}-" in target and target.endswith("-none-py3.12")
    assert (envs_root / target / ".complete").is_file()
    sync_lines = [line for line in uv_log.read_text().splitlines() if "ARGS=sync" in line]
    assert len(sync_lines) == 1, sync_lines
    assert "--extra" not in sync_lines[0]


def test_extra_named_none_is_distinct_from_ordinary_extra(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    none_target = env_name_for(checkout, extra="none")
    cpu_target = env_name_for(checkout, extra=EXTRA)
    assert none_target != cpu_target
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir, extra="none")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (envs_root / none_target / ".complete").is_file()
    assert not (envs_root / cpu_target).exists()


def test_old_six_field_manifest_still_reused(tmp_path: Path, envs_root: Path, cache_dir: Path) -> None:
    """The existing Holystore env's manifest (written before --project-dir
    existed, six fields, no project_dir line) must stay reusable."""
    checkout = make_checkout(tmp_path)
    uv_bin, uv_log = make_uv_stub(tmp_path)
    target_name = env_name_for(checkout)
    target_dir = envs_root / target_name
    target_dir.mkdir()
    six_field_manifest = make_valid_manifest(checkout, project_dir=None)
    assert len(six_field_manifest.splitlines()) == 6
    (target_dir / ".complete").write_text(six_field_manifest)
    result = run_builder(envs_root=envs_root, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REUSE" in result.stdout
    assert not uv_log.exists() or uv_log.read_text() == ""
    # Reuse never rewrites the pre-existing manifest.
    assert (target_dir / ".complete").read_text() == six_field_manifest
