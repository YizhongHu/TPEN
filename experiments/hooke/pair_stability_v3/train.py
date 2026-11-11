"""Launch train jobs from a planned ``00_grid`` attempt.

This script is intentionally a stage consumer: it reads a durable grid manifest
written by ``plan.py`` and emits training work into ``01_train``. It does not
expand grids, write ``00_grid`` attempts, or regenerate run commands.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

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

launch = sibling(__file__, 'launch')
_tpen_utils_io = sibling(__file__, 'utils.io')
write_json = _tpen_utils_io.write_json
_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_TRAIN = _tpen_utils_layout.STAGE_TRAIN
grid_attempt_dir = _tpen_utils_layout.grid_attempt_dir
stage_dir = _tpen_utils_layout.stage_dir
write_latest = _tpen_utils_layout.write_latest
_tpen_utils_naming = sibling(__file__, 'utils.naming')
log_prefix = _tpen_utils_naming.log_prefix
stage_job_name = _tpen_utils_naming.stage_job_name
study_name_from_manifest = _tpen_utils_naming.study_name_from_manifest

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.toolkit import (  # noqa: E402
    ExecutorOptions,
    LocalExecutor,
    StagePlan,
    SubmissionRequest,
    SubmititExecutor,
    write_execution_records,
)
from experiments.toolkit.resources import resource_from_profile  # noqa: E402
from experiments.toolkit.specs import tasks_from_commands  # noqa: E402

DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"

def _launch_jobs(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the jobs selected for this launch."""

    return [dict(job) for job in jobs]


def _execution_command(command: Sequence[str]) -> list[str]:
    """Return the run command after launch-mode overrides are applied."""

    return [str(part) for part in command]


def _train_attempt_dir(job: dict[str, Any], *, manifest: dict[str, Any], repo_root: Path) -> Path:
    if job.get("train_attempt_dir"):
        return launch.repo_path(str(job["train_attempt_dir"]), repo_root)
    if job.get("train_dir"):
        return launch.repo_path(str(job["train_dir"]), repo_root) / str(manifest["attempt_id"])
    raise ValueError(f"job {job.get('run_id', '<unknown>')!r} has no train attempt path")


def _stage_plan_dir(results_root: Path, attempt_id: str) -> Path:
    """Return the durable train stage-plan directory."""

    return stage_dir(results_root, STAGE_TRAIN) / "stage_plans" / attempt_id


def _executor(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    results_root: Path,
    study: str,
    log_attempt: str,
):
    """Return the toolkit executor for train submissions."""

    options = ExecutorOptions(
        backend=args.backend,
        args=args,
        repo_root=repo_root,
        log_dir=stage_dir(results_root, STAGE_TRAIN) / "slurm_logs" / log_attempt,
        job_name=stage_job_name(study, "train"),
        smoke=False,
        chunk_size=args.chunk_size,
        chunk_status_dir=stage_dir(results_root, STAGE_TRAIN) / "chunk_status" / log_attempt,
    )
    executor_cls = LocalExecutor if args.backend == "local" else SubmititExecutor
    return executor_cls(
        submit_command_sets=getattr(launch, "submit_command_sets"),
        options=options,
        claim_paths_for_statuses=launch.claim_paths_for_statuses,
    )


def _resource_spec(args: argparse.Namespace) -> Any:
    """Return a backend-neutral resource request for the selected device."""

    selector = launch.selected_device(args)
    profiles = launch.device_profiles(selector)
    resolved_profiles = {}
    for profile in profiles:
        uv_environment, uv_extras, _runtime_device = launch.resolve_uv_settings_for_profile(args, profile)
        slurm = launch.slurm_parameters(args, profile=profile)
        resolved_profiles[profile] = resource_from_profile(
            profile=profile,
            partition=slurm.get("slurm_partition"),
            timeout_min=slurm.get("timeout_min"),
            mem_gb=launch.slurm_resource_mem_gb(slurm),
            cpus=slurm.get("cpus_per_task"),
            gpus=slurm.get("gpus_per_node"),
            uv_environment=uv_environment,
            uv_extras=uv_extras,
        ).to_dict()
    if len(profiles) == 1:
        return resource_from_profile(
            profile=profiles[0],
            partition=resolved_profiles[profiles[0]].get("partition"),
            timeout_min=resolved_profiles[profiles[0]].get("timeout_min"),
            mem_gb=resolved_profiles[profiles[0]].get("mem_gb"),
            cpus=resolved_profiles[profiles[0]].get("threads"),
            gpus=resolved_profiles[profiles[0]].get("gpus"),
            uv_environment=resolved_profiles[profiles[0]].get("uv_environment"),
            uv_extras=resolved_profiles[profiles[0]].get("uv_extras", ()),
        )
    return resource_from_profile(
        profile=selector,
        partition=None,
        timeout_min=None,
        mem_gb=None,
        cpus=None,
        gpus=None,
        uv_environment=None,
        uv_extras=(),
        metadata={"profiles": resolved_profiles},
    )


def build_train_stage_plan(
    jobs: Sequence[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    results_root: Path,
    grid_attempt_id: str,
    repo_root: Path,
    commands: Sequence[Sequence[str]],
    row_status_paths: Sequence[Path],
    args: argparse.Namespace,
) -> StagePlan:
    """Build a reusable toolkit stage plan for train tasks."""

    attempt_id = grid_attempt_id
    result_dirs = [
        _train_attempt_dir(job, manifest=manifest, repo_root=repo_root)
        for job in jobs
    ]
    checkpoint_paths = [Path(result_dir) / "checkpoints" / "latest.json" for result_dir in result_dirs]
    tasks = tasks_from_commands(
        stage=STAGE_TRAIN,
        attempt_id=attempt_id,
        jobs=jobs,
        commands=commands,
        result_dirs=result_dirs,
        row_status_paths=row_status_paths,
        resources=_resource_spec(args),
        completion_policy="status_completed_with_checkpoint",
        checkpoint_paths=checkpoint_paths,
        source_attempts={"grid": grid_attempt_id},
    )
    return StagePlan(
        study=study_name_from_manifest(manifest),
        stage=STAGE_TRAIN,
        attempt_id=attempt_id,
        results_root=str(results_root),
        source_attempts={"grid": grid_attempt_id},
        timezone=manifest.get("timezone"),
        smoke=False,
        metadata={
            "backend": args.backend,
            "device": launch.selected_device(args),
            "chunk_size": args.chunk_size,
        },
        tasks=tasks,
    )


def write_train_submission_records(
    jobs: Sequence[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    results_root: Path,
    grid_attempt_id: str,
    repo_root: Path,
    backend: str,
    job_ids: Sequence[str],
    submitted_commands: Sequence[Sequence[str]],
) -> None:
    """Write train-stage provenance without mutating the ``00_grid`` manifest."""

    manifest_path = grid_attempt_dir(results_root, grid_attempt_id) / "manifest.json"
    grid_dir = grid_attempt_dir(results_root, grid_attempt_id)
    for index, (job, job_id) in enumerate(zip(jobs, job_ids, strict=True)):
        train_attempt = _train_attempt_dir(job, manifest=manifest, repo_root=repo_root)
        source = {
            "run_id": str(job["run_id"]),
            "grid_attempt_id": grid_attempt_id,
            "grid_attempt_dir": str(grid_dir),
            "manifest_path": str(manifest_path),
        }
        write_json(train_attempt / "source_grid_attempt.json", source)
        write_json(
            train_attempt / "submission.json",
            {
                "run_id": str(job["run_id"]),
                "grid_attempt_id": grid_attempt_id,
                "launcher": backend,
                "launcher_job_id": str(job_id),
                "command": job.get("command", ""),
                "submitted_command": shlex.join([str(part) for part in submitted_commands[index]]),
            },
        )


def write_train_launch_provenance(
    jobs: Sequence[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    results_root: Path,
    grid_attempt_id: str,
    repo_root: Path,
    submitted_commands: Sequence[Sequence[str]],
) -> list[Path]:
    """Create train attempt directories before scheduler execution starts."""

    manifest_path = grid_attempt_dir(results_root, grid_attempt_id) / "manifest.json"
    grid_dir = grid_attempt_dir(results_root, grid_attempt_id)
    row_status_paths: list[Path] = []
    for index, job in enumerate(jobs):
        train_attempt = _train_attempt_dir(job, manifest=manifest, repo_root=repo_root)
        source = {
            "run_id": str(job["run_id"]),
            "grid_attempt_id": grid_attempt_id,
            "grid_attempt_dir": str(grid_dir),
            "manifest_path": str(manifest_path),
        }
        write_json(train_attempt / "source_grid_attempt.json", source)
        (train_attempt / "command.txt").write_text(
            shlex.join([str(part) for part in submitted_commands[index]]) + "\n"
        )
        write_latest(train_attempt.parent, train_attempt.name)
        row_status_paths.append(train_attempt / "launcher_status.json")
    return row_status_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse train command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--grid-attempt-id", default=None, help="Grid attempt to launch (defaults to latest).")
    launch.add_launch_arguments(
        parser,
        smoke_help=(
            "Deprecated. Use configs/smoke.yaml with the normal stage stack."
        ),
    )
    args = parser.parse_args(argv)
    launch.reject_deprecated_smoke(parser, args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Launch train jobs from an existing ``00_grid`` attempt."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    repo_root = Path(args.repo_root) if args.repo_root else STUDY_DIR.parents[2]
    launch.ensure_submitit_launcher_environment(
        args,
        script_path=Path(__file__).resolve(),
        argv=raw_argv,
        repo_root=repo_root,
    )
    results_root = launch.repo_path(args.results_root, repo_root)
    grid_attempt_id = launch.resolve_grid_attempt_id(results_root, args.grid_attempt_id)
    manifest = launch.load_grid_manifest(results_root, grid_attempt_id)
    study = study_name_from_manifest(manifest)
    prefix = log_prefix(study)
    jobs = _launch_jobs(list(manifest.get("jobs", [])))
    commands = [
        _execution_command(launch.command_for_job(job))
        for job in jobs
    ]
    command_sets = launch.environment_command_sets(commands, args=args, repo_root=repo_root)
    submitted_commands = launch.summarize_command_sets(command_sets)

    if not jobs:
        print(f"{prefix} grid attempt {grid_attempt_id} has no jobs")
        return 0

    row_status_paths = write_train_launch_provenance(
        jobs,
        manifest=manifest,
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        repo_root=repo_root,
        submitted_commands=submitted_commands,
    )
    log_attempt = grid_attempt_id
    stage_plan = build_train_stage_plan(
        jobs,
        manifest=manifest,
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        repo_root=repo_root,
        commands=commands,
        row_status_paths=row_status_paths,
        args=args,
    )
    stage_plan_dir = stage_plan.write(_stage_plan_dir(results_root, log_attempt))
    execution_records = _executor(
        args=args,
        repo_root=repo_root,
        results_root=results_root,
        study=study,
        log_attempt=log_attempt,
    ).submit(
        stage_plan,
        stage_plan.tasks,
        SubmissionRequest(
            command_sets=command_sets,
            submitted_commands=submitted_commands,
        ),
    )
    job_ids = [record.launcher_job_id for record in execution_records]

    write_train_submission_records(
        jobs,
        manifest=manifest,
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        repo_root=repo_root,
        backend=args.backend,
        job_ids=job_ids,
        submitted_commands=submitted_commands,
    )
    write_execution_records(stage_plan_dir, execution_records)
    print(f"{prefix} launched {len(job_ids)} train jobs from 00_grid/{grid_attempt_id} via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
