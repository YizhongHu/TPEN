"""Archive one completed Hooke V3 lineage through a bounded Slurm transfer.

``plan`` traces a final report back through ``00_grid``, writes a durable
``10_sync`` attempt, and performs a no-copy byte-accounting dry run. ``submit``
then launches ``execute`` on the requested Slurm partition; execution refuses
an over-limit or stale plan, excludes checkpoint payloads, and writes a
self-verifying archive through an atomic partial-directory rename.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

# Siblings are loaded study-scoped, not by bare import: experiments/ has several
# same-named modules and the first study loaded would otherwise own the bare name
# for every study after it. See experiments/toolkit/study_imports.py.
#
# The loader is reached BY PATH rather than by putting the repository root on
# sys.path. A study directory that mutates sys.path is the mechanism behind the
# very defect this import exists to fix, and he-cutover's gateway test forbids it
# outright -- so the fix must not reintroduce it in order to install itself.
import importlib.util as _tpen_importlib  # noqa: E402
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

if "_tpen_study_imports" not in _tpen_sys.modules:
    _tpen_spec = _tpen_importlib.spec_from_file_location(
        "_tpen_study_imports",
        _TpenPath(__file__).resolve().parents[3] / "experiments" / "toolkit" / "study_imports.py",
    )
    _tpen_module = _tpen_importlib.module_from_spec(_tpen_spec)
    _tpen_sys.modules["_tpen_study_imports"] = _tpen_module
    _tpen_spec.loader.exec_module(_tpen_module)
sibling = _tpen_sys.modules["_tpen_study_imports"].sibling

_tpen_utils_ancestry = sibling(__file__, 'utils.ancestry')
Ancestry = _tpen_utils_ancestry.Ancestry
trace_final_report_ancestry = _tpen_utils_ancestry.trace_final_report_ancestry
_tpen_utils_io = sibling(__file__, 'utils.io')
path_from_record = _tpen_utils_io.path_from_record
read_json_object = _tpen_utils_io.read_json_object
read_json_object_list = _tpen_utils_io.read_json_object_list
write_json = _tpen_utils_io.write_json
_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_COLLECT = _tpen_utils_layout.STAGE_COLLECT
STAGE_FINAL_COLLECT = _tpen_utils_layout.STAGE_FINAL_COLLECT
STAGE_FINAL_EVAL = _tpen_utils_layout.STAGE_FINAL_EVAL
STAGE_FINAL_GRID = _tpen_utils_layout.STAGE_FINAL_GRID
STAGE_FINAL_REPORT = _tpen_utils_layout.STAGE_FINAL_REPORT
STAGE_FINAL_TRAIN = _tpen_utils_layout.STAGE_FINAL_TRAIN
STAGE_GRID = _tpen_utils_layout.STAGE_GRID
STAGE_SELECT = _tpen_utils_layout.STAGE_SELECT
STAGE_TRAIN = _tpen_utils_layout.STAGE_TRAIN
STAGE_VALIDATION = _tpen_utils_layout.STAGE_VALIDATION
latest_attempt_id = _tpen_utils_layout.latest_attempt_id
stage_dir = _tpen_utils_layout.stage_dir
write_latest = _tpen_utils_layout.write_latest

STUDY_RELATIVE = Path("experiments/hooke/pair_stability_v3")
STAGE_SYNC = "10_sync"
DEFAULT_MAX_BYTES = 10_000_000_000
DEFAULT_TIMEZONE = "America/New_York"
CHECKPOINT_DIRNAME = "checkpoints"
PYCACHE_DIRNAME = "__pycache__"
MAX_ATTEMPT_ID_BYTES = 64
HASH_BUFFER_BYTES = 1024 * 1024
FINAL_MANIFEST_RESERVE_BYTES = 1024
SOURCE_PATHS = (
    "run.py",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "spenn",
    "experiments/toolkit",
    str(STUDY_RELATIVE),
)
RUN_STAGES = frozenset({STAGE_TRAIN, STAGE_VALIDATION, STAGE_FINAL_TRAIN, STAGE_FINAL_EVAL})
REQUIRED_STAGES = frozenset(
    {
        STAGE_GRID,
        STAGE_TRAIN,
        STAGE_VALIDATION,
        STAGE_COLLECT,
        STAGE_SELECT,
        STAGE_FINAL_GRID,
        STAGE_FINAL_TRAIN,
        STAGE_FINAL_EVAL,
        STAGE_FINAL_COLLECT,
        STAGE_FINAL_REPORT,
    }
)


@dataclass(frozen=True)
class PlannedFile:
    """One regular result payload file selected for archival."""

    relative_path: str
    size_bytes: int
    sha256: str = ""


@dataclass(frozen=True)
class ArchivePlan:
    """Immutable dry-run plan for a canonical completed V3 lineage."""

    source_root: str
    results_relative: str
    destination: str
    report_attempt_id: str
    source_revision: str
    max_bytes: int
    source_bytes: int
    result_bytes: int
    sync_bytes: int
    skipped_checkpoint_dirs: int
    stage_counts: dict[str, int]
    attempt_roots: tuple[str, ...]
    support_roots: tuple[str, ...]
    result_files: tuple[PlannedFile, ...]

    @property
    def planned_bytes(self) -> int:
        """Return source, result, and durable sync artifact bytes."""

        return self.source_bytes + self.result_bytes + self.sync_bytes

    @property
    def under_limit(self) -> bool:
        """Return whether this exact dry-run payload fits the hard cap."""

        return self.planned_bytes <= self.max_bytes

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-safe durable plan record."""

        record = asdict(self)
        record["planned_bytes"] = self.planned_bytes
        record["under_limit"] = self.under_limit
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ArchivePlan:
        """Restore a plan written by :meth:`to_record`."""

        files = tuple(PlannedFile(**entry) for entry in record["result_files"])
        return cls(
            source_root=str(record["source_root"]),
            results_relative=str(record["results_relative"]),
            destination=str(record["destination"]),
            report_attempt_id=str(record["report_attempt_id"]),
            source_revision=str(record["source_revision"]),
            max_bytes=int(record["max_bytes"]),
            source_bytes=int(record["source_bytes"]),
            result_bytes=int(record["result_bytes"]),
            sync_bytes=int(record.get("sync_bytes", 1_000_000)),
            skipped_checkpoint_dirs=int(record["skipped_checkpoint_dirs"]),
            stage_counts={str(key): int(value) for key, value in record["stage_counts"].items()},
            attempt_roots=tuple(str(value) for value in record["attempt_roots"]),
            support_roots=tuple(str(value) for value in record["support_roots"]),
            result_files=files,
        )


@dataclass(frozen=True)
class SyncAttempt:
    """Paths belonging to one ``10_sync`` attempt."""

    directory: Path
    plan_path: Path
    files_path: Path
    dry_run_path: Path


def build_archive_plan(
    *,
    source_root: str | Path,
    destination: str | Path,
    report_attempt_id: str | None = None,
    source_revision: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ArchivePlan:
    """Trace and size one completed V3 report lineage without copying data.

    Parameters
    ----------
    source_root
        Clean-or-historical Git checkout holding the completed V3 results.
    destination
        Exact final archive directory. It is not created by this function.
    report_attempt_id
        ``09_final_report`` attempt to archive. The valid latest pointer is
        used when omitted.
    source_revision
        Git revision used by the historical run. When omitted, this is inferred
        from the manifested run metadata and must be unique.
    max_bytes
        Hard logical-byte cap. The default is 10 GB, deliberately decimal and
        therefore stricter than 10 GiB.

    Returns
    -------
    ArchivePlan
        Exact source payload, ancestry roots, and byte accounting.
    """

    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    source_root = Path(source_root).resolve()
    results_relative = STUDY_RELATIVE / "results"
    results_root = source_root / results_relative
    if not results_root.is_dir():
        raise NotADirectoryError(f"V3 results root does not exist: {results_root}")
    report_attempt_id = report_attempt_id or _resolve_report_attempt(results_root)

    attempt_roots, support_roots = _collect_archive_roots(results_root, report_attempt_id)
    revision = source_revision or _infer_source_revision(attempt_roots)
    _require_git_revision(source_root, revision)
    source_bytes = _git_archive_bytes(source_root, revision)
    result_files, skipped_checkpoint_dirs = _collect_result_files(
        roots=(*attempt_roots, *support_roots),
        source_root=source_root,
    )
    stage_counts = _stage_counts(attempt_roots, results_root)
    missing_stages = sorted(REQUIRED_STAGES.difference(stage_counts))
    if missing_stages:
        raise ValueError(f"report ancestry is incomplete; missing stages: {', '.join(missing_stages)}")

    plan = ArchivePlan(
        source_root=str(source_root),
        results_relative=str(results_relative),
        destination=str(Path(destination).resolve()),
        report_attempt_id=report_attempt_id,
        source_revision=revision,
        max_bytes=int(max_bytes),
        source_bytes=source_bytes,
        result_bytes=sum(entry.size_bytes for entry in result_files),
        sync_bytes=0,
        skipped_checkpoint_dirs=skipped_checkpoint_dirs,
        stage_counts=stage_counts,
        attempt_roots=tuple(sorted(_relative_to_source(path, source_root) for path in attempt_roots)),
        support_roots=tuple(sorted(_relative_to_source(path, source_root) for path in support_roots)),
        result_files=result_files,
    )
    return _with_sync_budget(plan)


def write_dry_run(plan: ArchivePlan, *, results_root: str | Path, attempt_id: str | None = None) -> SyncAttempt:
    """Write durable ``10_sync`` dry-run artifacts for one bounded plan."""

    results_root = Path(results_root).resolve()
    attempt_id = attempt_id or _new_attempt_id()
    if len(attempt_id.encode("utf-8")) > MAX_ATTEMPT_ID_BYTES:
        raise ValueError(f"sync attempt id exceeds {MAX_ATTEMPT_ID_BYTES} UTF-8 bytes")
    plan = _with_sync_budget(plan)
    actual_sync_bytes = _sync_artifact_bytes(plan, attempt_id)
    if actual_sync_bytes > plan.sync_bytes:
        raise ValueError("sync artifact payload exceeds the plan's pre-transfer byte budget")

    sync_root = stage_dir(results_root, STAGE_SYNC)
    directory = sync_root / attempt_id
    if directory.exists():
        raise FileExistsError(f"sync attempt already exists: {directory}")
    directory.mkdir(parents=True)
    write_latest(sync_root, attempt_id)

    plan_path = directory / "archive_plan.json"
    files_path = directory / "result_files.txt"
    dry_run_path = directory / "dry_run.json"
    write_json(plan_path, plan.to_record())
    files_path.write_bytes(_file_list_bytes(plan.result_files))
    write_json(dry_run_path, _dry_run_record(plan, attempt_id))
    return SyncAttempt(directory=directory, plan_path=plan_path, files_path=files_path, dry_run_path=dry_run_path)


def submit_sync(
    *,
    sync_attempt_dir: str | Path,
    partition: str = "test",
    time_limit: str = "12:00:00",
    memory: str = "8G",
    cpus_per_task: int = 1,
) -> str:
    """Submit a bounded archive transfer to Slurm and record its job id.

    The transfer is never launched on the caller's node. The submission is
    rejected unless the preceding dry run is under its hard byte cap.
    """

    sync_attempt = _load_sync_attempt(sync_attempt_dir)
    plan = _load_plan(sync_attempt.plan_path)
    if not plan.under_limit:
        raise ValueError(
            f"dry run is {plan.planned_bytes} bytes, exceeding the {plan.max_bytes}-byte limit; refusing submission"
        )
    if cpus_per_task <= 0:
        raise ValueError(f"cpus_per_task must be positive, got {cpus_per_task}")

    script = sync_attempt.directory / "transfer.sbatch"
    script.write_text(
        _slurm_script(
            sync_attempt=sync_attempt.directory,
            partition=partition,
            time_limit=time_limit,
            memory=memory,
            cpus_per_task=cpus_per_task,
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(("sbatch", "--parsable", str(script)), check=True, text=True, capture_output=True)
    job_id = completed.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id:
        raise RuntimeError(f"sbatch did not return a job id: {completed.stdout!r}")
    write_json(
        sync_attempt.directory / "submission.json",
        {
            "job_id": job_id,
            "partition": partition,
            "time_limit": time_limit,
            "memory": memory,
            "cpus_per_task": cpus_per_task,
            "script": str(script),
            "dry_run": str(sync_attempt.dry_run_path),
        },
    )
    return job_id


def execute_sync(*, sync_attempt_dir: str | Path) -> dict[str, Any]:
    """Copy an approved plan into its final archive from a Slurm job."""

    sync_attempt = _load_sync_attempt(sync_attempt_dir)
    plan = _load_plan(sync_attempt.plan_path)
    _assert_plan_is_current(plan)
    if not plan.under_limit:
        raise ValueError(f"dry run exceeds the {plan.max_bytes}-byte cap")

    destination = Path(plan.destination)
    if destination.exists():
        raise FileExistsError(f"archive destination already exists: {destination}")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    partial = destination.parent / f".{destination.name}.{job_id}.partial"
    if partial.exists():
        raise FileExistsError(f"partial archive already exists: {partial}")
    partial.mkdir(parents=True)

    try:
        _extract_source_tree(plan, partial)
        sync_dir = partial / "_sync"
        sync_dir.mkdir()
        shutil.copy2(sync_attempt.plan_path, sync_dir / "archive_plan.json")
        shutil.copy2(sync_attempt.files_path, sync_dir / "result_files.txt")
        shutil.copy2(sync_attempt.dry_run_path, sync_dir / "dry_run.json")
        _copy_result_files(plan, sync_attempt.files_path, partial)
        verification = verify_archive(plan=plan, archive_root=partial)
        write_json(sync_dir / "archive_manifest.json", verification)
        verification = verify_archive(plan=plan, archive_root=partial)
        partial.rename(destination)
    except BaseException:
        write_json(
            sync_attempt.directory / "transfer_failure.json",
            {"partial_archive": str(partial), "job_id": job_id},
        )
        raise

    transfer = {
        "job_id": job_id,
        "archive": str(destination),
        "archive_bytes": verification["archive_bytes"],
        "checkpoint_payload_transferred": False,
        "symlinks_transferred": False,
        "verification": verification,
    }
    write_json(sync_attempt.directory / "transfer.json", transfer)
    return transfer


def verify_archive(*, plan: ArchivePlan, archive_root: str | Path) -> dict[str, Any]:
    """Verify exact planned files, ancestry stages, and archive exclusions."""

    archive_root = Path(archive_root).resolve()
    if not archive_root.is_dir():
        raise NotADirectoryError(f"archive does not exist: {archive_root}")
    missing: list[str] = []
    size_mismatches: list[str] = []
    content_mismatches: list[str] = []
    hash_buffer = bytearray(HASH_BUFFER_BYTES)
    for entry in plan.result_files:
        path = archive_root / entry.relative_path
        if not path.is_file() or path.is_symlink():
            missing.append(entry.relative_path)
        elif path.stat().st_size != entry.size_bytes:
            size_mismatches.append(entry.relative_path)
        elif entry.sha256 and _sha256(path, hash_buffer) != entry.sha256:
            content_mismatches.append(entry.relative_path)
    if missing:
        raise RuntimeError(f"archive is missing {len(missing)} planned files; first: {missing[:3]}")
    if size_mismatches:
        raise RuntimeError(f"archive has {len(size_mismatches)} size mismatches; first: {size_mismatches[:3]}")
    if content_mismatches:
        raise RuntimeError(f"archive has {len(content_mismatches)} content mismatches; first: {content_mismatches[:3]}")

    checkpoint_dirs: list[str] = []
    symlinks: list[str] = []
    for directory, dirnames, filenames in os.walk(archive_root, topdown=True, followlinks=False):
        current = Path(directory)
        for dirname in list(dirnames):
            child = current / dirname
            if child.is_symlink():
                symlinks.append(str(child.relative_to(archive_root)))
            if dirname == CHECKPOINT_DIRNAME:
                checkpoint_dirs.append(str(child.relative_to(archive_root)))
        for filename in filenames:
            child = current / filename
            if child.is_symlink():
                symlinks.append(str(child.relative_to(archive_root)))
    if checkpoint_dirs:
        raise RuntimeError(f"archive contains checkpoint directories: {checkpoint_dirs[:3]}")
    if symlinks:
        raise RuntimeError(f"archive contains symlinks: {symlinks[:3]}")

    results_root = archive_root / plan.results_relative
    missing_stages = sorted(stage for stage in REQUIRED_STAGES if not (results_root / stage).is_dir())
    if missing_stages:
        raise RuntimeError(f"archive is missing required stages: {', '.join(missing_stages)}")
    source_revision = archive_root / "SOURCE_REVISION"
    if source_revision.read_text(encoding="utf-8").strip() != plan.source_revision:
        raise RuntimeError("archive SOURCE_REVISION does not match the approved plan")

    archive_bytes = _tree_bytes(archive_root)
    if archive_bytes > plan.max_bytes:
        raise RuntimeError(f"archive is {archive_bytes} bytes, exceeding {plan.max_bytes} bytes")
    return {
        "archive_bytes": archive_bytes,
        "max_bytes": plan.max_bytes,
        "planned_bytes": plan.planned_bytes,
        "result_file_count": len(plan.result_files),
        "stage_counts": plan.stage_counts,
        "skipped_checkpoint_dirs": plan.skipped_checkpoint_dirs,
    }


def _collect_archive_roots(results_root: Path, report_attempt_id: str) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return complete report ancestry plus scan and stage-support roots."""

    ancestry: Ancestry = trace_final_report_ancestry(results_root, report_attempt_id)
    if ancestry.warnings:
        raise ValueError("; ".join(ancestry.warnings))
    attempts = set(ancestry.roots)
    collect_roots = [root for root in attempts if _stage_name(root, results_root) == STAGE_COLLECT]
    if len(collect_roots) != 1:
        raise ValueError(f"expected one collection root, found {len(collect_roots)}")
    collection_root = collect_roots[0]
    for record in read_json_object_list(collection_root / "source_validation_attempts.json"):
        validation_root = _record_path(record, "validation_attempt_dir", results_root)
        attempts.add(validation_root)
        train_record = read_json_object(validation_root / "source_train_attempt.json")
        attempts.add(_record_path(train_record, "train_attempt_dir", results_root))

    support: set[Path] = set()
    for root in attempts:
        stage = _stage_name(root, results_root)
        if stage not in RUN_STAGES:
            continue
        attempt_id = root.name
        for support_name in ("stage_plans", "chunk_status", "slurm_logs"):
            candidate = results_root / stage / support_name / attempt_id
            if candidate.is_dir():
                support.add(candidate)
    return tuple(sorted(attempts)), tuple(sorted(support))


def _record_path(record: dict[str, Any], key: str, results_root: Path) -> Path:
    """Return a manifested source path that belongs to this results tree."""

    path = path_from_record(record, key)
    if path is None:
        raise ValueError(f"missing provenance path {key}")
    path = path.resolve()
    try:
        path.relative_to(results_root)
    except ValueError as exc:
        raise ValueError(f"provenance path is outside results root: {path}") from exc
    if not path.is_dir():
        raise FileNotFoundError(f"manifested source attempt does not exist: {path}")
    return path


def _collect_result_files(*, roots: Iterable[Path], source_root: Path) -> tuple[tuple[PlannedFile, ...], int]:
    """Collect content-addressed regular files while excluding checkpoint payloads."""

    files: dict[str, PlannedFile] = {}
    skipped_checkpoint_dirs = 0
    hash_buffer = bytearray(HASH_BUFFER_BYTES)
    for root in roots:
        for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            current = Path(directory)
            kept_dirs: list[str] = []
            for dirname in sorted(dirnames):
                child = current / dirname
                if dirname in {CHECKPOINT_DIRNAME, PYCACHE_DIRNAME}:
                    if dirname == CHECKPOINT_DIRNAME:
                        skipped_checkpoint_dirs += 1
                    continue
                if child.is_symlink():
                    continue
                kept_dirs.append(dirname)
            dirnames[:] = kept_dirs
            for filename in sorted(filenames):
                path = current / filename
                if path.is_symlink() or path.suffix == ".pyc":
                    continue
                if not stat.S_ISREG(path.lstat().st_mode):
                    continue
                relative = _relative_to_source(path, source_root)
                if relative in files:
                    continue
                size_bytes = path.stat().st_size
                sha256 = _sha256(path, hash_buffer)
                if path.stat().st_size != size_bytes:
                    raise RuntimeError(f"source file changed while hashing: {path}")
                files[relative] = PlannedFile(relative_path=relative, size_bytes=size_bytes, sha256=sha256)
    return tuple(files[path] for path in sorted(files)), skipped_checkpoint_dirs


def _infer_source_revision(roots: Iterable[Path]) -> str:
    """Infer one clean source revision from manifested run metadata."""

    revisions: set[str] = set()
    for root in roots:
        metadata = read_json_object(root / "metadata.json", warnings=[])
        revision = str(metadata.get("git_commit") or "").strip()
        if revision:
            revisions.add(revision)
    if not revisions:
        raise ValueError("no git_commit found in lineage metadata; pass --source-revision explicitly")
    if len(revisions) != 1:
        raise ValueError(f"lineage has multiple source revisions: {', '.join(sorted(revisions))}")
    return next(iter(revisions))


def _require_git_revision(source_root: Path, revision: str) -> None:
    """Ensure the requested historical source tree is locally archiveable."""

    subprocess.run(
        ("git", "-C", str(source_root), "cat-file", "-e", f"{revision}^{{commit}}"),
        check=True,
        capture_output=True,
        text=True,
    )


def _git_archive_bytes(source_root: Path, revision: str) -> int:
    """Return exact regular-file bytes in the historical source archive."""

    process = subprocess.Popen(
        ("git", "-C", str(source_root), "archive", "--format=tar", revision, *SOURCE_PATHS),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    total = 0
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if member.isfile():
                    total += member.size
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode() if process.stderr is not None else ""
    if process.wait():
        raise RuntimeError(f"git archive failed: {stderr}")
    return total


def _extract_source_tree(plan: ArchivePlan, destination: Path) -> None:
    """Extract the exact historical source revision into an archive root."""

    source_root = Path(plan.source_root)
    process = subprocess.Popen(
        ("git", "-C", str(source_root), "archive", "--format=tar", plan.source_revision, *SOURCE_PATHS),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        extract = subprocess.run(("tar", "-x", "-C", str(destination)), stdin=process.stdout, check=False)
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode() if process.stderr is not None else ""
    if extract.returncode or process.wait():
        raise RuntimeError(f"source extraction failed: tar={extract.returncode}; git={stderr}")
    (destination / "SOURCE_REVISION").write_text(f"{plan.source_revision}\n", encoding="utf-8")


def _copy_result_files(plan: ArchivePlan, file_list: Path, destination: Path) -> None:
    """Copy only dry-run-approved regular result files with rsync."""

    subprocess.run(
        (
            "rsync",
            "-a",
            "--no-links",
            "--from0",
            "--files-from",
            str(file_list),
            f"{Path(plan.source_root)}/",
            f"{destination}/",
        ),
        check=True,
    )


def _assert_plan_is_current(plan: ArchivePlan) -> None:
    """Reject source drift before creating a partial archive directory."""

    if not plan.under_limit:
        raise ValueError(f"planned payload {plan.planned_bytes} exceeds {plan.max_bytes}")
    source_root = Path(plan.source_root)
    _require_git_revision(source_root, plan.source_revision)
    hash_buffer = bytearray(HASH_BUFFER_BYTES)
    for entry in plan.result_files:
        path = source_root / entry.relative_path
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"planned source file is unavailable: {path}")
        if path.stat().st_size != entry.size_bytes:
            raise RuntimeError(f"planned source file changed size: {path}")
        if entry.sha256 and _sha256(path, hash_buffer) != entry.sha256:
            raise RuntimeError(f"planned source file changed content: {path}")


def _load_sync_attempt(directory: str | Path) -> SyncAttempt:
    """Return the durable paths for an existing ``10_sync`` attempt."""

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"sync attempt does not exist: {directory}")
    attempt = SyncAttempt(
        directory=directory,
        plan_path=directory / "archive_plan.json",
        files_path=directory / "result_files.txt",
        dry_run_path=directory / "dry_run.json",
    )
    for path in (attempt.plan_path, attempt.files_path, attempt.dry_run_path):
        if not path.is_file():
            raise FileNotFoundError(f"sync attempt is incomplete: {path}")
    return attempt


def _load_plan(path: Path) -> ArchivePlan:
    """Load an immutable dry-run plan."""

    return ArchivePlan.from_record(read_json_object(path))


def _resolve_report_attempt(results_root: Path) -> str:
    """Resolve the valid latest report attempt under ``09_final_report``."""

    attempt_id = latest_attempt_id(stage_dir(results_root, STAGE_FINAL_REPORT))
    if attempt_id is None:
        raise FileNotFoundError(f"no final report attempts under {stage_dir(results_root, STAGE_FINAL_REPORT)}")
    return attempt_id


def _stage_name(path: Path, results_root: Path) -> str | None:
    try:
        return path.resolve().relative_to(results_root).parts[0]
    except ValueError:
        return None


def _stage_counts(roots: Iterable[Path], results_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for root in roots:
        stage = _stage_name(root, results_root)
        if stage is not None:
            counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _relative_to_source(path: Path, source_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(source_root))
    except ValueError as exc:
        raise ValueError(f"planned path is outside source root: {path}") from exc


def _tree_bytes(root: Path) -> int:
    """Return logical bytes without following symlinks."""

    total = 0
    for directory, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            path = Path(directory) / filename
            if not path.is_symlink() and path.is_file():
                total += path.stat().st_size
    return total

def _with_sync_budget(plan: ArchivePlan) -> ArchivePlan:
    """Return a plan with a stable upper bound for durable sync artifacts."""

    placeholder_attempt_id = "x" * MAX_ATTEMPT_ID_BYTES
    for _ in range(4):
        sync_bytes = _sync_artifact_bytes(plan, placeholder_attempt_id)
        if sync_bytes == plan.sync_bytes:
            return plan
        plan = replace(plan, sync_bytes=sync_bytes)
    raise RuntimeError("sync artifact byte budget did not converge")


def _sync_artifact_bytes(plan: ArchivePlan, attempt_id: str) -> int:
    """Return the durable plan, file-list, dry-run, and manifest byte budget."""

    return (
        len(_json_bytes(plan.to_record()))
        + len(_file_list_bytes(plan.result_files))
        + len(_json_bytes(_dry_run_record(plan, attempt_id)))
        + FINAL_MANIFEST_RESERVE_BYTES
    )


def _file_list_bytes(files: Iterable[PlannedFile]) -> bytes:
    """Encode rsync-safe relative paths with NUL delimiters."""

    payload = bytearray()
    for entry in files:
        payload.extend(entry.relative_path.encode("utf-8"))
        payload.append(0)
    return bytes(payload)


def _json_bytes(payload: Any) -> bytes:
    """Match the on-disk encoding emitted by :func:`write_json`."""

    return (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode("utf-8")


def _dry_run_record(plan: ArchivePlan, attempt_id: str) -> dict[str, Any]:
    """Return the durable no-copy accounting record for one sync attempt."""

    return {
        "stage": STAGE_SYNC,
        "attempt_id": attempt_id,
        "report_attempt_id": plan.report_attempt_id,
        "source_revision": plan.source_revision,
        "destination": plan.destination,
        "source_bytes": plan.source_bytes,
        "result_bytes": plan.result_bytes,
        "sync_bytes": plan.sync_bytes,
        "planned_bytes": plan.planned_bytes,
        "max_bytes": plan.max_bytes,
        "under_limit": plan.under_limit,
        "result_file_count": len(plan.result_files),
        "skipped_checkpoint_dirs": plan.skipped_checkpoint_dirs,
        "stage_counts": plan.stage_counts,
    }


def _sha256(path: Path, buffer: bytearray) -> str:
    """Return one regular file's SHA-256 using a caller-owned buffer."""

    digest = hashlib.sha256()
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as handle:
        while count := handle.readinto(view):
            digest.update(view[:count])
    return digest.hexdigest()


def _new_attempt_id() -> str:
    """Return a New York timestamp suitable for a durable sync stage."""

    return datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).strftime("%Y%m%dT%H%M%S%z")


def _tool_root() -> Path:
    """Return the repository that contains this durable sync implementation."""

    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not locate the repository containing sync.py")


def _slurm_script(
    *,
    sync_attempt: Path,
    partition: str,
    time_limit: str,
    memory: str,
    cpus_per_task: int,
) -> str:
    """Render the test-partition wrapper that performs the transfer."""

    tool_root = _tool_root()
    sync_path = Path(__file__).resolve()
    quoted = lambda value: shlex.quote(str(value))
    return "\n".join(
        (
            "#!/usr/bin/env bash",
            "#SBATCH --job-name=pair-v3-sync",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --mem={memory}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --output={quoted(sync_attempt / 'slurm-%j.out')}",
            f"#SBATCH --error={quoted(sync_attempt / 'slurm-%j.err')}",
            "set -euo pipefail",
            f"cd {quoted(tool_root)}",
            f"exec uv run --extra cpu python {quoted(sync_path)} execute --sync-attempt {quoted(sync_attempt)}",
            "",
        )
    )


def _print_record(record: dict[str, Any]) -> None:
    print(json.dumps(record, indent=2, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse archive plan, submission, execution, and verification commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    plan = subcommands.add_parser("plan", help="trace an archive lineage and write a no-copy dry run")
    plan.add_argument("--source-root", type=Path, default=_tool_root())
    plan.add_argument("--destination", type=Path, required=True)
    plan.add_argument("--report-attempt-id", default=None)
    plan.add_argument("--source-revision", default=None)
    plan.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    plan.add_argument("--attempt-id", default=None)

    submit = subcommands.add_parser("submit", help="submit an approved archive transfer to Slurm")
    submit.add_argument("--sync-attempt", type=Path, required=True)
    submit.add_argument("--partition", default="test")
    submit.add_argument("--time", dest="time_limit", default="12:00:00")
    submit.add_argument("--mem", dest="memory", default="8G")
    submit.add_argument("--cpus-per-task", type=int, default=1)

    execute = subcommands.add_parser("execute", help=argparse.SUPPRESS)
    execute.add_argument("--sync-attempt", type=Path, required=True)

    verify = subcommands.add_parser("verify", help="verify an existing archive against its dry run")
    verify.add_argument("--sync-attempt", type=Path, required=True)
    verify.add_argument("--archive", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected sync workflow command."""

    args = parse_args(argv)
    if args.command == "plan":
        plan = build_archive_plan(
            source_root=args.source_root,
            destination=args.destination,
            report_attempt_id=args.report_attempt_id,
            source_revision=args.source_revision,
            max_bytes=args.max_bytes,
        )
        attempt = write_dry_run(
            plan,
            results_root=Path(args.source_root).resolve() / STUDY_RELATIVE / "results",
            attempt_id=args.attempt_id,
        )
        _print_record({"sync_attempt": str(attempt.directory), **json.loads(attempt.dry_run_path.read_text())})
        return 0 if plan.under_limit else 2
    if args.command == "submit":
        _print_record({"job_id": submit_sync(
            sync_attempt_dir=args.sync_attempt,
            partition=args.partition,
            time_limit=args.time_limit,
            memory=args.memory,
            cpus_per_task=args.cpus_per_task,
        )})
        return 0
    if args.command == "execute":
        _print_record(execute_sync(sync_attempt_dir=args.sync_attempt))
        return 0
    if args.command == "verify":
        attempt = _load_sync_attempt(args.sync_attempt)
        plan = _load_plan(attempt.plan_path)
        archive = args.archive or Path(plan.destination)
        verification = verify_archive(plan=plan, archive_root=archive)
        write_json(attempt.directory / "verification.json", verification)
        _print_record(verification)
        return 0
    raise AssertionError(f"unknown sync command: {args.command}")


__all__ = [
    "ArchivePlan",
    "PlannedFile",
    "STAGE_SYNC",
    "SyncAttempt",
    "build_archive_plan",
    "execute_sync",
    "submit_sync",
    "verify_archive",
    "write_dry_run",
]


if __name__ == "__main__":
    raise SystemExit(main())
