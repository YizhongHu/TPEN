"""Contract tests for ``scripts/cannon/shared-env-precheck.sh``.

This is the reuse-only precheck ``verify.sh`` delegates to: it must never
build or write an environment, only validate caller-supplied paths/names and
resolve the expected env path under a shared, out-of-repo Holystore-style
root. Driven as a subprocess against small fixture trees under ``tmp_path`` --
no Slurm, no cluster access, no real ``uv`` (mirrors
``test_build_shared_env.py``).

``shared-env-precheck.sh`` and ``verify.sh``'s ``SHARED_ENV_PREFIX`` consumer
contract are both new in this slice: at the prior head (base
``61c1f071566a7bbe49016d185b2f00967e2170fd``) this script does not exist and
``verify.sh`` took a ``SHARED_ENV_ROOT`` it built inline, so every test here
is necessarily red (``FileNotFoundError``/collection error) against that
head.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PRECHECK = REPO_ROOT / "scripts" / "cannon" / "shared-env-precheck.sh"
VERIFY = REPO_ROOT / "scripts" / "cannon" / "verify.sh"
BUILDER = REPO_ROOT / "scripts" / "cannon" / "build-shared-env.sh"
MANIFEST_HELPER = REPO_ROOT / "scripts" / "cannon" / "shared-env-manifest.sh"

PROJECT = "tpen"
EXTRA = "cpu"
PYTHON_VERSION = "3.12"
LOCK_SHA256 = hashlib.sha256(b"locked-dependency-contents\n").hexdigest()


def run_precheck(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(PRECHECK), *args], capture_output=True, text=True, check=False
    )


def make_shared_env(
    shared_prefix: Path,
    *,
    project: str = PROJECT,
    extra: str = EXTRA,
    python_version: str = PYTHON_VERSION,
    lock_sha256: str = LOCK_SHA256,
    complete: bool = True,
    manifest_lock_sha256: str | None = None,
    symlink_target: Path | None = None,
    bin_python: bool = True,
) -> Path:
    env_name = f"{project}-{lock_sha256[:12]}-{extra}-py{python_version}"
    target = shared_prefix / env_name
    if symlink_target is not None:
        symlink_target.mkdir(parents=True, exist_ok=True)
        target.symlink_to(symlink_target)
        return target
    target.mkdir(parents=True)
    if bin_python:
        (target / "bin").mkdir()
        python = target / "bin" / "python"
        python.write_text("#!/usr/bin/env bash\necho stub\n")
        python.chmod(python.stat().st_mode | stat.S_IEXEC)
    if complete:
        recorded_lock = manifest_lock_sha256 or lock_sha256
        (target / ".complete").write_text(
            "head=deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n"
            f"uv_lock_sha256={recorded_lock}\n"
            "uv_version=uv 9.9.9-stub\n"
            "python_version=3.12.7\n"
            "python_request=3.12\n"
            "built_utc=2026-01-01T00:00:00Z\n"
        )
    return target


def snapshot(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


@pytest.fixture()
def shared_prefix(tmp_path: Path) -> Path:
    root = tmp_path / "shared-envs"
    root.mkdir()
    return root


@pytest.fixture()
def facility_prefix(tmp_path: Path) -> Path:
    root = tmp_path / "facility-run"
    root.mkdir()
    return root


@pytest.fixture()
def work_root(tmp_path: Path) -> Path:
    root = tmp_path / "scratch-job"
    root.mkdir()
    return root


def run_paths(
    *,
    shared_prefix: Path,
    facility_prefix: Path,
    run_root: Path,
    work_root: Path,
) -> subprocess.CompletedProcess[str]:
    return run_precheck(
        "paths",
        "--shared-env-prefix", str(shared_prefix),
        "--facility-run-prefix", str(facility_prefix),
        "--run-root", str(run_root),
        "--work-root", str(work_root),
    )


# --- "paths" mode -----------------------------------------------------------


def test_paths_refuses_when_prefixes_equal(shared_prefix: Path, work_root: Path) -> None:
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=shared_prefix,
        run_root=shared_prefix / "run",
        work_root=work_root,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "PRECHECK_FAILED" in result.stderr
    assert "disjoint" in result.stderr


def test_paths_refuses_when_shared_nested_in_facility(shared_prefix: Path, facility_prefix: Path, work_root: Path) -> None:
    nested_shared = facility_prefix / "shared-envs"
    nested_shared.mkdir()
    result = run_paths(
        shared_prefix=nested_shared,
        facility_prefix=facility_prefix,
        run_root=facility_prefix / "run",
        work_root=work_root,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "disjoint" in result.stderr


def test_paths_refuses_when_facility_nested_in_shared(shared_prefix: Path, work_root: Path) -> None:
    nested_facility = shared_prefix / "facility-run"
    nested_facility.mkdir()
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=nested_facility,
        run_root=nested_facility / "run",
        work_root=work_root,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "disjoint" in result.stderr


def test_paths_refuses_when_run_root_outside_facility_prefix(
    shared_prefix: Path, facility_prefix: Path, work_root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere" / "run"
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=facility_prefix,
        run_root=outside,
        work_root=work_root,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "outside facility prefix" in result.stderr or "stay under" in result.stderr


def test_paths_refuses_when_run_root_inside_shared_prefix(shared_prefix: Path, facility_prefix: Path, work_root: Path) -> None:
    # Pathologically misconfigured wrapper: RUN_ROOT points under the shared
    # prefix even though it is disjoint from FACILITY_RUN_PREFIX. With
    # disjoint prefixes this is necessarily ALSO "not under the facility
    # prefix", so either refusal message is a correct, fail-closed outcome;
    # which one fires first is an implementation detail of check ordering.
    run_root = shared_prefix / "run"
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=facility_prefix,
        run_root=run_root,
        work_root=work_root,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert (
        "must not be under --shared-env-prefix" in result.stderr
        or "must stay under --facility-run-prefix" in result.stderr
    )


def test_paths_accepts_disjoint_prefixes_and_contained_run_root(
    shared_prefix: Path, facility_prefix: Path, work_root: Path
) -> None:
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=facility_prefix,
        run_root=facility_prefix / "job-123",
        work_root=work_root,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- "paths" mode: F2, shared prefix vs WORK_ROOT overlap --------------------
#
# Reviewed at head 0a1d2e53 (review-round1-static-0a1d2e53 / T2): a shared,
# reuse-only env prefix that overlaps the job's own writable WORK_ROOT lets
# the run create entries inside it (uv cache, tmp, clone) before any env is
# resolved, and WORK_ROOT's EXIT-trap `rm -rf` can delete into it. These
# cases are necessarily red before the --work-root check existed (the flag
# itself was unrecognized), and red again if the check were removed.


def test_paths_refuses_when_work_root_equals_shared_prefix(shared_prefix: Path, facility_prefix: Path) -> None:
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=facility_prefix,
        run_root=facility_prefix / "job-123",
        work_root=shared_prefix,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--work-root" in result.stderr and "disjoint" in result.stderr


def test_paths_refuses_when_work_root_nested_in_shared_prefix(shared_prefix: Path, facility_prefix: Path) -> None:
    nested_work = shared_prefix / "scratch-job"
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=facility_prefix,
        run_root=facility_prefix / "job-123",
        work_root=nested_work,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--work-root" in result.stderr and "disjoint" in result.stderr


def test_paths_refuses_when_shared_prefix_nested_in_work_root(shared_prefix: Path, facility_prefix: Path, tmp_path: Path) -> None:
    # The reverse nesting: a misconfigured wrapper points WORK_ROOT at an
    # ancestor of SHARED_ENV_PREFIX itself, so the job's own cleanup trap
    # would delete the shared prefix's parent.
    outer_work_root = shared_prefix.parent
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=facility_prefix,
        run_root=facility_prefix / "job-123",
        work_root=outer_work_root,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--work-root" in result.stderr and "disjoint" in result.stderr


def test_paths_accepts_work_root_disjoint_from_shared_prefix(
    shared_prefix: Path, facility_prefix: Path, work_root: Path
) -> None:
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=facility_prefix,
        run_root=facility_prefix / "job-123",
        work_root=work_root,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_paths_refuses_when_work_root_is_a_symlink_alias_into_shared_prefix(
    shared_prefix: Path, facility_prefix: Path, tmp_path: Path
) -> None:
    # canon() (realpath -m) resolves symlinks of existing ancestors, so an
    # alias is just as reachable an overlap as a direct nested path.
    alias = tmp_path / "scratch-alias"
    alias.symlink_to(shared_prefix, target_is_directory=True)
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=facility_prefix,
        run_root=facility_prefix / "job-123",
        work_root=alias,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--work-root" in result.stderr and "disjoint" in result.stderr


def test_paths_refuses_when_work_root_is_filesystem_root(shared_prefix: Path, facility_prefix: Path) -> None:
    """DT1 (B1 delta review, carried to this item's carried-follow-ups note):
    WORK_ROOT='/' exercises path_contains' own '/' special case directly as
    its first argument (not merely as the thing being contained), which
    verify.sh's own call site can never reach since WORK_ROOT is always the
    exact job scratch, never '/'. Kills mutant m3 (drop that branch) without
    relying on check_shared_env_prefix's separate '/' refusal, which only
    ever sees SHARED_ENV_PREFIX/FACILITY_RUN_PREFIX, never WORK_ROOT. The
    normal disjoint case stays green via
    test_paths_accepts_work_root_disjoint_from_shared_prefix."""
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=facility_prefix,
        run_root=facility_prefix / "job-123",
        work_root=Path("/"),
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--work-root" in result.stderr and "disjoint" in result.stderr


def test_verify_sh_runs_shared_env_paths_precheck_before_any_mkdir_or_clone() -> None:
    # The paths precheck (shared prefix vs facility/run/work-root
    # disjointness) must run before any write -- safe_mkdir, the EXIT-trap
    # registration, or the git clone -- so a misconfigured overlap refuses
    # closed rather than creating or deleting anything first.
    lines = VERIFY.read_text().splitlines()
    precheck_line = next(i for i, line in enumerate(lines) if "shared-env-precheck.sh\" paths" in line)
    mkdir_line = next(i for i, line in enumerate(lines) if line.strip().startswith("safe_mkdir"))
    trap_line = next(i for i, line in enumerate(lines) if line.strip() == 'trap cleanup EXIT')
    clone_line = next(i for i, line in enumerate(lines) if "git clone" in line)
    assert precheck_line < mkdir_line < trap_line < clone_line, (precheck_line, mkdir_line, trap_line, clone_line)


# --- "paths"/"resolve" modes: F3, filesystem-root edge cases -----------------
#
# A raw "$prefix/*" glob never matches an ordinary absolute path when
# prefix=="/" (that pattern literally starts with "//"), so a canonical "/"
# previously defeated every disjointness/containment check it appeared in.


def test_paths_refuses_shared_prefix_is_filesystem_root(facility_prefix: Path, work_root: Path) -> None:
    result = run_paths(
        shared_prefix=Path("/"),
        facility_prefix=facility_prefix,
        run_root=facility_prefix / "job-123",
        work_root=work_root,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "filesystem root" in result.stderr


def test_paths_refuses_facility_prefix_is_filesystem_root(shared_prefix: Path, work_root: Path) -> None:
    result = run_paths(
        shared_prefix=shared_prefix,
        facility_prefix=Path("/"),
        run_root=Path("/tmp/does-not-matter"),
        work_root=work_root,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "filesystem root" in result.stderr


def test_resolve_refuses_shared_prefix_is_filesystem_root() -> None:
    result = run_precheck(
        *resolve_args(Path("/")),
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "filesystem root" in result.stderr


# --- "resolve" mode ----------------------------------------------------------


def resolve_args(shared_prefix: Path, **overrides: str) -> list[str]:
    args = {
        "--shared-env-prefix": str(shared_prefix),
        "--project": PROJECT,
        "--extra": EXTRA,
        "--python": PYTHON_VERSION,
        "--lock-sha256": LOCK_SHA256,
    }
    for key, value in overrides.items():
        args[f"--{key.replace('_', '-')}"] = value
    out: list[str] = ["resolve"]
    for k, v in args.items():
        out += [k, v]
    return out


def test_resolve_refuses_when_env_missing(shared_prefix: Path) -> None:
    result = run_precheck(*resolve_args(shared_prefix))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "build it with build-shared-env.sh first" in result.stderr


def test_resolve_refuses_when_env_exists_without_complete(shared_prefix: Path) -> None:
    make_shared_env(shared_prefix, complete=False)
    result = run_precheck(*resolve_args(shared_prefix))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "no .complete manifest" in result.stderr


def test_resolve_refuses_on_lock_sha_mismatch(shared_prefix: Path) -> None:
    other_sha = hashlib.sha256(b"different\n").hexdigest()
    make_shared_env(shared_prefix, manifest_lock_sha256=other_sha)
    result = run_precheck(*resolve_args(shared_prefix))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "does not match the clone's" in result.stderr


def test_resolve_refuses_on_malformed_manifest_field(shared_prefix: Path) -> None:
    make_shared_env(shared_prefix)
    env_name = f"{PROJECT}-{LOCK_SHA256[:12]}-{EXTRA}-py{PYTHON_VERSION}"
    (shared_prefix / env_name / ".complete").write_text("not-a-manifest\n")
    result = run_precheck(*resolve_args(shared_prefix))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "no well-formed uv_lock_sha256 field" in result.stderr


def test_resolve_refuses_when_env_is_symlink(shared_prefix: Path, tmp_path: Path) -> None:
    real_target = tmp_path / "elsewhere-env"
    make_shared_env(shared_prefix, symlink_target=real_target)
    result = run_precheck(*resolve_args(shared_prefix))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "is a symlink" in result.stderr


def test_resolve_refuses_when_bin_python_missing(shared_prefix: Path) -> None:
    make_shared_env(shared_prefix, bin_python=False)
    result = run_precheck(*resolve_args(shared_prefix))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "no executable bin/python" in result.stderr


def test_resolve_refuses_on_traversal_project_name(shared_prefix: Path) -> None:
    make_shared_env(shared_prefix)
    result = run_precheck(*resolve_args(shared_prefix, project=".."))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "--project must be a single path component" in result.stderr


def test_resolve_accepts_valid_env_and_prints_path(shared_prefix: Path) -> None:
    target = make_shared_env(shared_prefix)
    result = run_precheck(*resolve_args(shared_prefix))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == str(target)


def test_resolve_valid_env_makes_no_write_under_shared_prefix(shared_prefix: Path) -> None:
    make_shared_env(shared_prefix)
    before = snapshot(shared_prefix)
    result = run_precheck(*resolve_args(shared_prefix))
    assert result.returncode == 0, result.stdout + result.stderr
    after = snapshot(shared_prefix)
    assert before == after


# --- static structural invariants on verify.sh itself ------------------------


def test_verify_sh_never_syncs_or_builds_shared_env() -> None:
    source = VERIFY.read_text()
    assert "uv sync" not in source, "verify.sh must never invoke uv sync for the shared env (B1 scope)"
    assert "SHARED_ENV_ROOT" not in source, "SHARED_ENV_ROOT is replaced by the reuse-only SHARED_ENV_PREFIX contract"
    assert "shared-env-precheck.sh" in source


def test_verify_sh_write_path_roots_exclude_shared_env_prefix() -> None:
    source = VERIFY.read_text()
    roots_line = next(line for line in source.splitlines() if line.strip().startswith('for root in "$CANON_WORK_ROOT"'))
    assert "SHARED_ENV" not in roots_line, roots_line


def test_verify_sh_discovers_all_test_python_vars_and_excludes_owned_names(tmp_path: Path) -> None:
    source = VERIFY.read_text()
    discovery_line = next(line for line in source.splitlines() if "grep -rhoE" in line and "_PYTHON" in line)

    pattern = re.search(r"'([^']+)'", discovery_line)
    assert pattern is not None
    discovery = re.compile(pattern.group(1))
    for name in ("H_V2_SCIENCE_PYTHON", "P9C_EXECUTOR_PYTHON"):
        assert discovery.fullmatch(name)
    deny_match = re.search(r'case "\$executor_var" in\n\s*([^)]*)\)', source)
    assert deny_match is not None
    assert set(deny_match.group(1).split("|")) == {"PYTHON", "TPEN_PYTHON", "ROOT_PYTHON"}

    loop_start = source.index("mapfile -t EXECUTOR_PYTHON_VARS")
    loop_end = source.index("\ndone", loop_start) + len("\ndone")
    loop = source[loop_start:loop_end]
    # macOS's system Bash lacks mapfile; retain the production grep and
    # binding loop while using an equivalent array fill for this probe.
    loop = loop.replace(
        "mapfile -t EXECUTOR_PYTHON_VARS < <(",
        "EXECUTOR_PYTHON_VARS=(); while IFS= read -r executor_var; do EXECUTOR_PYTHON_VARS+=(\"$executor_var\"); done < <(",
    )
    clone = tmp_path / "clone"
    (clone / "tests").mkdir(parents=True)
    (clone / "tests" / "vars.py").write_text(
        "H_V2_SCIENCE_PYTHON TPEN_PYTHON ROOT_PYTHON\n"
    )
    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    probe = (
        "set -Eeuo pipefail\n"
        f"CLONE=\"{clone}\"\n"
        "PYTHON=/tmp/overlay/bin/python\n"
        f"RUN_ROOT=\"{run_root}\"\n"
        "assert_write_path() { :; }\n"
        f"{loop}\n"
        "[[ \"${H_V2_SCIENCE_PYTHON:-}\" == \"$PYTHON\" ]]\n"
        "[[ -z \"${TPEN_PYTHON+x}\" ]]\n"
        "[[ -z \"${ROOT_PYTHON+x}\" ]]\n"
    )
    result = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run_root / "evidence" / "executor-python-bindings.txt").read_text().splitlines() == [
        "H_V2_SCIENCE_PYTHON=/tmp/overlay/bin/python"
    ]


def test_verify_sh_executor_resolution_precheck_rejects_outside_clone(tmp_path: Path) -> None:
    source = VERIFY.read_text()
    start = source.index("executor_resolution_precheck() {")
    end = source.index("\n}\nEXECUTOR_ROOT=", start) + 2
    function = source[start:end]
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$OUTSIDE/tpen_executor/__init__.py\"\n"
        "exit 1\n"
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)
    evidence = tmp_path / "executor-resolution.txt"
    clone_executor = tmp_path / "clone" / "executor"
    clone_executor.mkdir(parents=True)
    probe = (
        "set -Eeuo pipefail\n"
        "PYTHON=\"$FAKE_PYTHON\"\n"
        "assert_write_path() { :; }\n"
        "fail_precheck() { echo \"PRECHECK_FAILED $*\" >&2; exit 2; }\n"
        f"{function}\n"
        f"executor_resolution_precheck \"$CLONE_EXECUTOR\" \"$EVIDENCE\"\n"
    )
    result = subprocess.run(
        ["bash", "-c", probe],
        env={
            "FAKE_PYTHON": str(fake_python),
            "CLONE_EXECUTOR": str(clone_executor),
            "EVIDENCE": str(evidence),
            "OUTSIDE": str(tmp_path / "outside"),
            "PATH": os.environ["PATH"],
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "tpen_executor resolution failed" in result.stderr
    assert evidence.exists()


# --- T1 (reduced), F1: every isolated interpreter invocation carries -B ------
#
# Reviewed at head 0a1d2e53: `-I` ignores `PYTHONDONTWRITEBYTECODE`, so an
# isolated invocation that imports from the shared env needs `-B` too, or a
# writable/cold shared env gets bytecode writes `assert_write_path` cannot
# see. Full recursive real-runner immutability was declined for this slice
# (review-round-1-decisions); the independent verifier adds that against a
# disposable fixture prefix instead. This slice keeps (a) a static contract
# over every `-I` invocation in verify.sh, and (b) a behavioural check that
# an isolated, writable, cold-module site-packages gets no `.pyc`/
# `__pycache__` with `-I -B`, demonstrating the same bytecode behaviour
# `verify.sh`'s probe now relies on without needing the real shared env.


def test_verify_sh_every_isolated_invocation_passes_dash_b() -> None:
    source = VERIFY.read_text()
    isolated_invocations = [
        line for line in source.splitlines() if "-I" in line and '"$PYTHON"' in line
    ]
    assert isolated_invocations, "expected at least one isolated (-I) interpreter invocation in verify.sh"
    for line in isolated_invocations:
        assert "-B" in line, f"isolated invocation missing -B (bytecode writes into shared env possible): {line}"


ISOLATED_IMPORT_SNIPPET = "import sys; sys.path.insert(0, sys.argv[1]); import coldmod"


def test_isolated_cold_import_without_dash_b_writes_bytecode(tmp_path: Path) -> None:
    # Establishes the failure mode -B fixes: red without it. -I disables
    # PYTHONPATH too, so the site-packages dir is injected via sys.path in
    # the snippet itself rather than cwd/PYTHONPATH (both ignored under -I).
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (site_packages / "coldmod.py").write_text("VALUE = 1\n")
    result = subprocess.run(
        ["python3", "-I", "-c", ISOLATED_IMPORT_SNIPPET, str(site_packages)],
        env={"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (site_packages / "__pycache__").is_dir(), (
        "expected -I alone (ignoring PYTHONDONTWRITEBYTECODE) to still write bytecode; "
        "if this fails, the interpreter's -I semantics changed and the F1 fix may be unnecessary"
    )


def test_isolated_cold_import_with_dash_b_writes_no_bytecode(tmp_path: Path) -> None:
    # Green: the exact flag combination verify.sh's probe now uses.
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (site_packages / "coldmod.py").write_text("VALUE = 1\n")
    result = subprocess.run(
        ["python3", "-I", "-B", "-c", ISOLATED_IMPORT_SNIPPET, str(site_packages)],
        env={"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (site_packages / "__pycache__").exists(), "no bytecode should be written under -I -B"


# --- T4, F4: real producer output flows straight into the consumer ----------
#
# Runs the actual build-shared-env.sh (with a stubbed uv, mirroring
# test_build_shared_env.py) and feeds its real generated env name/manifest
# directly into shared-env-precheck.sh resolve, so naming/schema drift
# between the two scripts cannot hide behind hand-authored fixture manifests
# matching a copied format by coincidence.

BUILD_PY_FULL_VERSION = "3.12.7"


def make_git_checkout(tmp_path: Path, *, name: str = "checkout", lock_contents: bytes = b"locked-dependency-contents\n") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    env = dict(os.environ)
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "test"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "test@example.com"
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    (repo / "uv.lock").write_bytes(lock_contents)
    (repo / "scripts" / "cannon").mkdir(parents=True)
    (repo / "scripts" / "cannon" / "shared-env-policy.json").write_text('{"MAX_SHARED_ENV_DIRS": 20}\n')
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=env)
    return repo


def make_uv_stub(tmp_path: Path, *, name: str = "uv-stub", python_version: str = BUILD_PY_FULL_VERSION) -> Path:
    uv = tmp_path / name
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--version" ]]; then echo "uv 9.9.9-stub"; exit 0; fi\n'
        'if [[ "$1" == "sync" ]]; then\n'
        '  mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"\n'
        f'  printf \'#!/usr/bin/env bash\\necho "Python {python_version}"\\n\' > "$UV_PROJECT_ENVIRONMENT/bin/python"\n'
        '  chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"\n'
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    return uv


def run_real_builder(*, envs_root: Path, checkout: Path, uv_bin: Path, cache_dir: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(cache_dir)
    return subprocess.run(
        [
            "bash", str(BUILDER),
            "--envs-root", str(envs_root),
            "--checkout", str(checkout),
            "--project", PROJECT,
            "--extra", EXTRA,
            "--python", PYTHON_VERSION,
            "--uv-bin", str(uv_bin),
        ],
        capture_output=True, text=True, env=env, check=False,
    )


@pytest.fixture()
def real_built_env(tmp_path: Path, shared_prefix: Path) -> tuple[Path, Path, str]:
    """Build a real shared env with build-shared-env.sh; return (env dir, manifest path, lock sha256)."""
    checkout = make_git_checkout(tmp_path)
    uv_bin = make_uv_stub(tmp_path)
    cache_dir = tmp_path / "uv-cache"
    cache_dir.mkdir()
    result = run_real_builder(envs_root=shared_prefix, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    lock_sha256 = hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest()
    env_name = f"{PROJECT}-{lock_sha256[:12]}-{EXTRA}-py{PYTHON_VERSION}"
    env_dir = shared_prefix / env_name
    assert env_dir.is_dir(), result.stdout + result.stderr
    return env_dir, env_dir / ".complete", lock_sha256


def test_resolve_accepts_real_builder_output(shared_prefix: Path, real_built_env: tuple[Path, Path, str]) -> None:
    env_dir, _manifest, lock_sha256 = real_built_env
    result = run_precheck(*resolve_args(shared_prefix, lock_sha256=lock_sha256))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == str(env_dir)


def test_resolve_refuses_real_manifest_truncated_after_matching_hash(
    shared_prefix: Path, real_built_env: tuple[Path, Path, str]
) -> None:
    _env_dir, manifest, lock_sha256 = real_built_env
    lines = manifest.read_text().splitlines(keepends=True)
    manifest.write_text("".join(lines[:2]))  # keep only head= and uv_lock_sha256=
    result = run_precheck(*resolve_args(shared_prefix, lock_sha256=lock_sha256))
    assert result.returncode != 0, result.stdout + result.stderr


def test_resolve_refuses_real_manifest_missing_a_later_field(
    shared_prefix: Path, real_built_env: tuple[Path, Path, str]
) -> None:
    _env_dir, manifest, lock_sha256 = real_built_env
    lines = manifest.read_text().splitlines(keepends=True)
    del lines[4]  # drop python_request=
    manifest.write_text("".join(lines))
    result = run_precheck(*resolve_args(shared_prefix, lock_sha256=lock_sha256))
    assert result.returncode != 0, result.stdout + result.stderr


def test_resolve_refuses_real_manifest_with_duplicate_conflicting_hash(
    shared_prefix: Path, real_built_env: tuple[Path, Path, str]
) -> None:
    _env_dir, manifest, lock_sha256 = real_built_env
    other_sha = hashlib.sha256(b"different\n").hexdigest()
    manifest.write_text(manifest.read_text() + f"uv_lock_sha256={other_sha}\n")
    result = run_precheck(*resolve_args(shared_prefix, lock_sha256=lock_sha256))
    assert result.returncode != 0, result.stdout + result.stderr


def test_resolve_accepts_real_manifest_with_different_build_head(
    tmp_path: Path, shared_prefix: Path
) -> None:
    # The build HEAD may differ from the requesting HEAD: dependencies, not
    # history, define identity (build-shared-env.sh's own invariant).
    checkout = make_git_checkout(tmp_path)
    uv_bin = make_uv_stub(tmp_path)
    cache_dir = tmp_path / "uv-cache"
    cache_dir.mkdir()
    run_real_builder(envs_root=shared_prefix, checkout=checkout, uv_bin=uv_bin, cache_dir=cache_dir)
    lock_sha256 = hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest()
    env_name = f"{PROJECT}-{lock_sha256[:12]}-{EXTRA}-py{PYTHON_VERSION}"
    manifest = shared_prefix / env_name / ".complete"
    lines = manifest.read_text().splitlines()
    lines[0] = "head=1111111111111111111111111111111111111111"
    manifest.write_text("\n".join(lines) + "\n")
    result = run_precheck(*resolve_args(shared_prefix, lock_sha256=lock_sha256))
    assert result.returncode == 0, result.stdout + result.stderr


# --- T5: old-SHA compatibility, no builder/policy needed alongside ----------
#
# verify.sh sources shared-env-precheck.sh (and shared-env-precheck.sh in
# turn sources shared-env-manifest.sh) from its OWN SCRIPT_DIR, never from
# the target clone under test -- so a clone old enough to lack
# build-shared-env.sh/shared-env-policy.json entirely can still reuse an
# existing, matching shared env with no builder invocation.


def test_resolve_from_runner_owned_copy_without_builder_or_policy_nearby(
    tmp_path: Path, shared_prefix: Path
) -> None:
    runner_copy = tmp_path / "runner-owned-scripts"
    runner_copy.mkdir()
    shutil.copy2(PRECHECK, runner_copy / PRECHECK.name)
    shutil.copy2(MANIFEST_HELPER, runner_copy / MANIFEST_HELPER.name)
    assert not (runner_copy / "build-shared-env.sh").exists()
    assert not (runner_copy / "shared-env-policy.json").exists()
    target = make_shared_env(shared_prefix)
    result = subprocess.run(
        ["bash", str(runner_copy / PRECHECK.name), *resolve_args(shared_prefix)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == str(target)
