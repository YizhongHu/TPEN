"""Hermetic checks for full-suite artifact retention in ``verify.sh``."""

from __future__ import annotations

import hashlib
import ast
import os
import shlex
import subprocess
from pathlib import Path


VERIFY = Path(__file__).resolve().parents[3] / "scripts" / "cannon" / "verify.sh"


def _decode_inventory_path(value: str) -> str:
    """Decode the collector's one-line escaped path representation."""

    return ast.literal_eval(f"'{value}'")


def _collector_harness(source: str, run_root: Path, work_root: Path, clone: Path) -> str:
    """Return a shell fixture containing the collector without the Slurm runner."""

    start = source.index("artifact_is_excluded() {")
    end = source.index("\ncleanup() {")
    functions = source[start:end]
    return f"""\
set -Eeuo pipefail
RUN_ROOT={shlex.quote(str(run_root))}
WORK_ROOT={shlex.quote(str(work_root))}
TMPDIR={shlex.quote(str(work_root / 'tmp'))}
CLONE={shlex.quote(str(clone))}
assert_write_path() {{ :; }}
{functions}
collect_full_suite_artifacts
"""


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_clone(tmp_path: Path) -> Path:
    clone = tmp_path / "clone"
    clone.mkdir()
    _git(clone, "init", "--quiet")
    _git(clone, "config", "user.email", "test@example.invalid")
    _git(clone, "config", "user.name", "artifact-test")
    (clone / "tracked.txt").write_text("before\n")
    (clone / ".gitignore").write_text("outputs/\n")
    _git(clone, "add", "tracked.txt", ".gitignore")
    _git(clone, "commit", "--quiet", "-m", "fixture")
    (clone / "tracked.txt").write_text("after\n")
    (clone / "untracked.txt").write_text("untracked\n")
    (clone / "outputs").mkdir()
    (clone / "outputs" / "ignored.txt").write_text("ignored output\n")
    (clone / "outputs" / "newline\nname.txt").write_text("newline output\n")
    (clone / "__pycache__").mkdir()
    (clone / "__pycache__" / "skip.pyc").write_bytes(b"cache")
    return clone


def test_full_suite_collector_copies_allowlist_and_writes_inventory(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    work_root = tmp_path / "work"
    (work_root / "tmp" / "pytest-basetemp").mkdir(parents=True)
    (work_root / "tmp" / "pytest-basetemp" / "test.log").write_text("pytest\n")
    (work_root / "tmp" / "__pycache__").mkdir()
    (work_root / "tmp" / "__pycache__" / "ignored.pyc").write_bytes(b"cache")
    (work_root / "mutant-one.diff").write_text("diff\n")
    clone = _make_clone(tmp_path)

    result = subprocess.run(
        ["bash", "-c", _collector_harness(VERIFY.read_text(), run_root, work_root, clone)],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"]},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    artifacts = run_root / "artifacts"
    assert (artifacts / "tmpdir/pytest-basetemp/test.log").read_text() == "pytest\n"
    assert (artifacts / "checkout/tracked.txt").read_text() == "after\n"
    assert (artifacts / "checkout/untracked.txt").read_text() == "untracked\n"
    assert (artifacts / "checkout/outputs/ignored.txt").read_text() == "ignored output\n"
    assert (artifacts / "checkout/outputs/newline\nname.txt").read_text() == "newline output\n"
    assert (artifacts / "harness/mutant-one.diff").read_text() == "diff\n"
    assert not (artifacts / "tmpdir/__pycache__").exists()
    assert not (artifacts / "checkout/__pycache__").exists()
    assert not (artifacts / "checkout/.git").exists()

    inventory = (artifacts / "INVENTORY.sha256").read_text().splitlines()
    assert len(inventory) == 6
    for line in inventory[:5]:
        digest, relative = line.split("  ", 1)
        assert digest == hashlib.sha256((artifacts / _decode_inventory_path(relative)).read_bytes()).hexdigest()
    decoded_paths = [_decode_inventory_path(line.split("  ", 1)[1]) for line in inventory]
    assert decoded_paths.count("checkout/outputs/newline\nname.txt") == 1
    size_line = (run_root / "evidence/artifacts-size.txt").read_text().strip()
    expected_bytes = sum(
        len(content)
        for content in (b"pytest\n", b"after\n", b"untracked\n", b"ignored output\n", b"newline output\n", b"diff\n")
    )
    assert size_line == f"artifact_file_count=6 artifact_bytes={expected_bytes}"
    assert (run_root / "evidence/artifacts-status.txt").read_text() == "ARTIFACTS_OK\n"
    assert (run_root / "evidence/artifacts-timing.txt").read_text().startswith("artifact_collection_seconds=")


def test_collector_is_full_suite_only_and_verdict_isolation_is_fail_closed() -> None:
    source = VERIFY.read_text()
    cleanup = source[source.index("cleanup() {") : source.index("trap cleanup EXIT")]
    assert 'if [[ "$MODE" == measure && "$FULL_SUITE" == 1 ]]' in cleanup
    assert 'printf \'DRIVER_EXIT=%s\\n\' "$rc"' in cleanup
    assert "ARTIFACTS_FAILED reason=" in cleanup
    assert 'exit "$rc"' in cleanup
    assert '[[ "$WORK_ROOT" == "/scratch/$USER/$SLURM_JOB_ID" ]] && rm -rf' in cleanup

    # The test-only failure switch proves that a collection failure is possible
    # without making it a second product verdict or changing quick mode.
    assert 'ARTIFACTS_FORCE_FAILURE' in source
    assert 'FULL_SUITE="${FULL_SUITE:-0}"' in source
    assert "collect_full_suite_artifacts" in source


def test_full_suite_does_not_overwrite_an_existing_inventory(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    (run_root / "artifacts").mkdir()
    (run_root / "artifacts/INVENTORY.sha256").write_text("sentinel\n")
    work_root = tmp_path / "work"
    (work_root / "tmp").mkdir(parents=True)
    clone = _make_clone(tmp_path)

    script = _collector_harness(VERIFY.read_text(), run_root, work_root, clone)
    result = subprocess.run(
        ["bash", "-c", script], check=False, capture_output=True, text=True, env={"PATH": os.environ["PATH"]}
    )
    assert result.returncode != 0
    assert (run_root / "artifacts/INVENTORY.sha256").read_text() == "sentinel\n"


def test_collection_budget_writes_partial_without_touching_verdict_or_scratch(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    (run_root / "evidence/driver-exit.txt").write_text("DRIVER_EXIT=7\n")
    work_root = tmp_path / "work"
    (work_root / "tmp" / "pytest-basetemp").mkdir(parents=True)
    (work_root / "tmp" / "pytest-basetemp" / "test.log").write_text("pytest\n")
    clone = _make_clone(tmp_path)

    script = _collector_harness(VERIFY.read_text(), run_root, work_root, clone)
    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ARTIFACT_COLLECTION_BUDGET_SECONDS": "0.01",
            "ARTIFACT_TEST_ENUM_DELAY_SECONDS": "0.02",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run_root / "evidence/artifacts-status.txt").read_text().startswith("ARTIFACTS_PARTIAL reason=")
    assert "artifact_file_count=0 artifact_bytes=0" in (run_root / "evidence/artifacts-size.txt").read_text()
    assert (run_root / "evidence/artifacts-timing.txt").read_text().startswith("artifact_collection_seconds=")
    assert (run_root / "evidence/driver-exit.txt").read_text() == "DRIVER_EXIT=7\n"
    assert work_root.exists()
