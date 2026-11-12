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
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PRECHECK = REPO_ROOT / "scripts" / "cannon" / "shared-env-precheck.sh"
VERIFY = REPO_ROOT / "scripts" / "cannon" / "verify.sh"

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


# --- "paths" mode -----------------------------------------------------------


def test_paths_refuses_when_prefixes_equal(shared_prefix: Path) -> None:
    result = run_precheck(
        "paths",
        "--shared-env-prefix", str(shared_prefix),
        "--facility-run-prefix", str(shared_prefix),
        "--run-root", str(shared_prefix / "run"),
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "PRECHECK_FAILED" in result.stderr
    assert "disjoint" in result.stderr


def test_paths_refuses_when_shared_nested_in_facility(shared_prefix: Path, facility_prefix: Path) -> None:
    nested_shared = facility_prefix / "shared-envs"
    nested_shared.mkdir()
    result = run_precheck(
        "paths",
        "--shared-env-prefix", str(nested_shared),
        "--facility-run-prefix", str(facility_prefix),
        "--run-root", str(facility_prefix / "run"),
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "disjoint" in result.stderr


def test_paths_refuses_when_facility_nested_in_shared(shared_prefix: Path) -> None:
    nested_facility = shared_prefix / "facility-run"
    nested_facility.mkdir()
    result = run_precheck(
        "paths",
        "--shared-env-prefix", str(shared_prefix),
        "--facility-run-prefix", str(nested_facility),
        "--run-root", str(nested_facility / "run"),
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "disjoint" in result.stderr


def test_paths_refuses_when_run_root_outside_facility_prefix(shared_prefix: Path, facility_prefix: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "run"
    result = run_precheck(
        "paths",
        "--shared-env-prefix", str(shared_prefix),
        "--facility-run-prefix", str(facility_prefix),
        "--run-root", str(outside),
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "outside facility prefix" in result.stderr or "stay under" in result.stderr


def test_paths_refuses_when_run_root_inside_shared_prefix(shared_prefix: Path, facility_prefix: Path) -> None:
    # Pathologically misconfigured wrapper: RUN_ROOT points under the shared
    # prefix even though it is disjoint from FACILITY_RUN_PREFIX.
    run_root = shared_prefix / "run"
    result = run_precheck(
        "paths",
        "--shared-env-prefix", str(shared_prefix),
        "--facility-run-prefix", str(facility_prefix),
        "--run-root", str(run_root),
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "must not be under --shared-env-prefix" in result.stderr


def test_paths_accepts_disjoint_prefixes_and_contained_run_root(shared_prefix: Path, facility_prefix: Path) -> None:
    result = run_precheck(
        "paths",
        "--shared-env-prefix", str(shared_prefix),
        "--facility-run-prefix", str(facility_prefix),
        "--run-root", str(facility_prefix / "job-123"),
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
