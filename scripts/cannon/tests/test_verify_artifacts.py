"""Hermetic checks for full-suite artifact retention in ``verify.sh``."""

from __future__ import annotations

import hashlib
import ast
import io
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path


VERIFY = Path(__file__).resolve().parents[3] / "scripts" / "cannon" / "verify.sh"


def _decode_inventory_path(value: str) -> str:
    """Decode the collector's one-line escaped path representation."""

    return ast.literal_eval(f"'{value}'")


def _inventory_entry(line: str) -> tuple[str, str]:
    """Split one GNU checksum line, including its special-path prefix."""

    if line.startswith("\\"):
        line = line[1:]
    digest, relative = line.split("  ", 1)
    return digest, relative


def _manifest_entry(line: str) -> tuple[str, int, str]:
    """Split one escaped manifest row into path, byte count, and digest."""

    relative, size, digest = line.split("\t")
    return _decode_inventory_path(relative), int(size), digest


def _read_archive(path: Path) -> dict[str, bytes]:
    """Decompress the zstd stream and return regular tar members."""

    decompressed = subprocess.run(["zstd", "-q", "-d", "-c", str(path)], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(decompressed), mode="r:") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }


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


def _cleanup_harness(source: str, run_root: Path, work_root: Path, marker: Path, mode: str, driver_exit: int) -> str:
    """Return the real cleanup path with only scratch deletion observed."""

    start = source.index("artifact_is_excluded() {")
    cleanup_start = source.index("\ncleanup() {")
    trap_start = source.index("\ntrap cleanup EXIT", cleanup_start)
    functions = source[start:cleanup_start]
    cleanup = source[cleanup_start:trap_start]
    return f"""\
set -Eeuo pipefail
RUN_ROOT={shlex.quote(str(run_root))}
WORK_ROOT=/scratch/fixture-user/fixture-job
TMPDIR={shlex.quote(str(work_root / 'tmp'))}
CLONE={shlex.quote(str(work_root / 'clone'))}
USER=fixture-user
SLURM_JOB_ID=fixture-job
MODE={shlex.quote(mode)}
FULL_SUITE=1
PYTHON=python3
assert_write_path() {{ :; }}
rm() {{ touch {shlex.quote(str(marker))}; }}
{functions}
{cleanup}
trap cleanup EXIT
exit {driver_exit}
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


def test_full_suite_collector_archives_allowlist_and_writes_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    work_root = tmp_path / "work"
    (work_root / "tmp" / "pytest-basetemp").mkdir(parents=True)
    (work_root / "tmp" / "pytest-basetemp" / "test.log").write_text("pytest\n")
    (work_root / "tmp" / "pytest-basetemp" / "skipped-link").symlink_to(tmp_path / "outside")
    (tmp_path / "outside").write_text("outside\n")
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
    members = _read_archive(artifacts / "artifacts.tar.zst")
    assert members["tmpdir/pytest-basetemp/test.log"] == b"pytest\n"
    assert members["checkout/tracked.txt"] == b"after\n"
    assert members["checkout/untracked.txt"] == b"untracked\n"
    assert members["checkout/outputs/ignored.txt"] == b"ignored output\n"
    assert members["checkout/outputs/newline\nname.txt"] == b"newline output\n"
    assert members["harness/mutant-one.diff"] == b"diff\n"
    assert not any("__pycache__" in path or ".git" in path for path in members)

    manifest = [_manifest_entry(line) for line in (artifacts / "MANIFEST.tsv").read_text().splitlines()]
    assert len(manifest) == 6
    assert {path for path, _, _ in manifest} == set(members)
    for path, size, digest in manifest:
        assert size == len(members[path])
        assert digest == hashlib.sha256(members[path]).hexdigest()
    assert "checkout/outputs/newline\nname.txt" in members
    size_line = (run_root / "evidence/artifacts-size.txt").read_text().strip()
    expected_bytes = sum(
        len(content)
        for content in (b"pytest\n", b"after\n", b"untracked\n", b"ignored output\n", b"newline output\n", b"diff\n")
    )
    assert size_line.startswith(f"artifact_file_count=6 artifact_bytes={expected_bytes} artifact_tar_bytes=")
    status = (run_root / "evidence/artifacts-status.txt").read_text()
    assert status.startswith("ARTIFACTS_OK tar_bytes=")
    assert "member_count=6" in status
    assert "elapsed_seconds=" in status
    assert (run_root / "evidence/artifacts-timing.txt").read_text().startswith("artifact_collection_seconds=")


def test_collector_failure_preserves_verdict_and_cleans_scratch(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    (run_root / "evidence/verdict.txt").write_text("FAIL\n")
    work_root = tmp_path / "work"
    (work_root / "tmp").mkdir(parents=True)
    marker = tmp_path / "scratch-cleaned"
    script = _cleanup_harness(VERIFY.read_text(), run_root, work_root, marker, "measure", 7)
    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "ARTIFACTS_FORCE_FAILURE": "1"},
    )
    assert result.returncode == 7, result.stdout + result.stderr
    assert (run_root / "evidence/driver-exit.txt").read_text() == "DRIVER_EXIT=7\n"
    assert (run_root / "evidence/verdict.txt").read_text() == "FAIL\n"
    assert (run_root / "evidence/artifacts-status.txt").read_text().startswith("ARTIFACTS_FAILED reason=")
    assert marker.exists()


def test_quick_mode_cleanup_does_not_create_artifact_tree(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    work_root = tmp_path / "work"
    (work_root / "tmp").mkdir(parents=True)
    marker = tmp_path / "scratch-cleaned"
    script = _cleanup_harness(VERIFY.read_text(), run_root, work_root, marker, "quick", 0)
    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"]},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (run_root / "artifacts").exists()
    assert marker.exists()


def test_full_suite_does_not_overwrite_an_existing_inventory(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    (run_root / "artifacts").mkdir()
    (run_root / "artifacts/sentinel.txt").write_text("sentinel\n")
    work_root = tmp_path / "work"
    (work_root / "tmp").mkdir(parents=True)
    clone = _make_clone(tmp_path)

    script = _collector_harness(VERIFY.read_text(), run_root, work_root, clone)
    result = subprocess.run(
        ["bash", "-c", script], check=False, capture_output=True, text=True, env={"PATH": os.environ["PATH"]}
    )
    assert result.returncode != 0
    assert (run_root / "artifacts/sentinel.txt").read_text() == "sentinel\n"
    assert sorted(path.name for path in (run_root / "artifacts").iterdir()) == ["sentinel.txt"]


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


def test_changed_or_untracked_symlink_cannot_copy_outside_clone(tmp_path: Path) -> None:
    """A git-listed symlink must not make checkout retention escape CLONE."""

    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    work_root = tmp_path / "work"
    (work_root / "tmp").mkdir(parents=True)
    clone = _make_clone(tmp_path)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("outside secret\n")
    (clone / "escape.txt").symlink_to(outside)

    result = subprocess.run(
        ["bash", "-c", _collector_harness(VERIFY.read_text(), run_root, work_root, clone)],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"]},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (run_root / "artifacts/checkout/escape.txt").exists()


def test_collection_budget_covers_git_listing_time(tmp_path: Path) -> None:
    """A slow git enumeration must produce partial artifacts at the deadline."""

    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    work_root = tmp_path / "work"
    (work_root / "tmp").mkdir(parents=True)
    clone = _make_clone(tmp_path)
    shutil.rmtree(clone / "outputs")
    real_git = shutil.which("git")
    assert real_git
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in *' diff '*|*' ls-files '*) sleep 5; exit 0;; esac\n"
        f"exec {shlex.quote(real_git)} \"$@\"\n"
    )
    fake_git.chmod(0o755)

    result = subprocess.run(
        ["bash", "-c", _collector_harness(VERIFY.read_text(), run_root, work_root, clone)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "ARTIFACT_COLLECTION_BUDGET_SECONDS": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run_root / "evidence/artifacts-status.txt").read_text().startswith(
        "ARTIFACTS_PARTIAL reason=collection budget exceeded during git listing"
    )
    assert (run_root / "evidence/artifacts-size.txt").read_text().startswith("artifact_file_count=0 artifact_bytes=0")
    elapsed = float((run_root / "evidence/artifacts-timing.txt").read_text().split("=", 1)[1])
    assert elapsed < 3


def test_collection_budget_checks_each_scandir_entry(tmp_path: Path) -> None:
    """An entry that arrives after the budget is not recorded or copied."""

    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    work_root = tmp_path / "work"
    (work_root / "tmp").mkdir(parents=True)
    (work_root / "tmp/late-link").symlink_to(tmp_path / "outside")
    (tmp_path / "outside").write_text("outside\n")
    clone = _make_clone(tmp_path)

    result = subprocess.run(
        ["bash", "-c", _collector_harness(VERIFY.read_text(), run_root, work_root, clone)],
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
    assert "late-link" not in (run_root / "artifacts/artifact-inventory.tsv").read_text()


def test_final_walk_deadline_check_records_partial(tmp_path: Path) -> None:
    """Expiry after the final walk must not be reported as ARTIFACTS_OK."""

    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    work_root = tmp_path / "work"
    (work_root / "tmp").mkdir(parents=True)
    clone = _make_clone(tmp_path)

    result = subprocess.run(
        ["bash", "-c", _collector_harness(VERIFY.read_text(), run_root, work_root, clone)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ARTIFACT_COLLECTION_BUDGET_SECONDS": "1",
            "ARTIFACT_TEST_FINAL_WALK_DELAY_SECONDS": "1.2",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run_root / "evidence/artifacts-status.txt").read_text().startswith("ARTIFACTS_PARTIAL reason=")


def test_direct_work_root_symlink_is_recorded_as_skipped(tmp_path: Path) -> None:
    """A symlink directly under WORK_ROOT is retained in skip metadata."""

    run_root = tmp_path / "run"
    (run_root / "evidence").mkdir(parents=True)
    work_root = tmp_path / "work"
    (work_root / "tmp").mkdir(parents=True)
    (work_root / "direct-link").symlink_to(tmp_path / "outside")
    (tmp_path / "outside").write_text("outside\n")
    clone = _make_clone(tmp_path)

    result = subprocess.run(
        ["bash", "-c", _collector_harness(VERIFY.read_text(), run_root, work_root, clone)],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"]},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = (run_root / "artifacts/artifact-inventory.tsv").read_text()
    assert "skipped_symlink" in metadata
    assert "harness/direct-link" in metadata
