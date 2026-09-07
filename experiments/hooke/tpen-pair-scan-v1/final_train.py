"""Launch final training from ``05_final_grid``.

Final training mirrors scan ``train.py`` but consumes final replicate rows with
explicit final model/sampler seeds. It writes ``06_final_train`` provenance for
each final run before launching the canonical ``run.py`` entrypoint.
"""

from __future__ import annotations

import argparse
import csv
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
read_json = _tpen_utils_io.read_json
write_json = _tpen_utils_io.write_json
_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_FINAL_GRID = _tpen_utils_layout.STAGE_FINAL_GRID
STAGE_FINAL_TRAIN = _tpen_utils_layout.STAGE_FINAL_TRAIN
final_grid_attempt_dir = _tpen_utils_layout.final_grid_attempt_dir
final_train_attempt_dir = _tpen_utils_layout.final_train_attempt_dir
final_train_run_dir = _tpen_utils_layout.final_train_run_dir
latest_attempt_id = _tpen_utils_layout.latest_attempt_id
stage_dir = _tpen_utils_layout.stage_dir
write_latest = _tpen_utils_layout.write_latest
_tpen_utils_naming = sibling(__file__, 'utils.naming')
experiment_run_name = _tpen_utils_naming.experiment_run_name
log_prefix = _tpen_utils_naming.log_prefix
stage_job_name = _tpen_utils_naming.stage_job_name
study_name_from_manifest = _tpen_utils_naming.study_name_from_manifest
_tpen_utils_overrides = sibling(__file__, 'utils.overrides')
AxisOverrideSpec = _tpen_utils_overrides.AxisOverrideSpec
axis_value_overrides = _tpen_utils_overrides.axis_value_overrides
format_override_value = _tpen_utils_overrides.format_override_value
normalize_axis_override_specs = _tpen_utils_overrides.normalize_axis_override_specs
_tpen_utils_seeds = sibling(__file__, 'utils.seeds')
seed_override_values = _tpen_utils_seeds.seed_override_values

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
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
from experiments.toolkit.task_state import (  # noqa: E402
    _checkpoint_step,
    _complete_checkpoint_dirs,
    _final_train_completed,
    _latest_complete_checkpoint,
    _resume_overrides,
)


def _exclude_completed_jobs(
    jobs: Sequence[dict[str, Any]],
    *,
    results_root: Path,
    attempt_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for job in jobs:
        attempt_dir = final_train_attempt_dir(results_root, str(job["final_run_id"]), attempt_id)
        if _final_train_completed(attempt_dir):
            completed.append(dict(job))
        else:
            eligible.append(dict(job))
    return eligible, completed


def _resolve_final_grid_attempt_id(results_root: Path, requested: str | None) -> str:
    if requested is not None:
        return requested
    final_grid_stage = stage_dir(results_root, STAGE_FINAL_GRID)
    attempt_id = latest_attempt_id(final_grid_stage)
    if attempt_id is None:
        raise FileNotFoundError(f"no final-grid attempts under {final_grid_stage}")
    return attempt_id


def load_final_grid_manifest(results_root: Path, final_grid_attempt_id: str) -> dict[str, Any]:
    """Read and validate the ``05_final_grid`` manifest."""

    manifest_path = final_grid_attempt_dir(results_root, final_grid_attempt_id) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"final-grid attempt has no manifest.json: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("stage") != STAGE_FINAL_GRID:
        raise ValueError(f"manifest {manifest_path} is not a {STAGE_FINAL_GRID} manifest")
    return manifest


def load_final_jobs(results_root: Path, final_grid_attempt_id: str) -> list[dict[str, Any]]:
    """Read final jobs in CSV order, enriching from per-job JSON records."""

    grid_dir = final_grid_attempt_dir(results_root, final_grid_attempt_id)
    final_jobs_path = grid_dir / "final_jobs.csv"
    if not final_jobs_path.is_file():
        raise FileNotFoundError(f"final-grid attempt has no final_jobs.csv: {final_jobs_path}")
    with final_jobs_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    jobs = []
    for row in rows:
        job_path = grid_dir / "jobs" / f"{row['final_run_id']}.json"
        if job_path.is_file():
            jobs.append(read_json(job_path))
        else:
            jobs.append(dict(row))
    return jobs


def _selected_jobs(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(job) for job in jobs]


def _attempt_id(args: argparse.Namespace, *, final_grid_attempt_id: str) -> str:
    if args.attempt_id:
        return args.attempt_id
    return final_grid_attempt_id


def final_scalar_axes(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return non-seed axes recorded in a final-grid manifest."""

    return tuple(str(axis) for axis in (*manifest.get("major_axes", []), *manifest.get("minor_axes", [])))


def final_axis_override_paths(manifest: dict[str, Any], axes: Sequence[str]) -> dict[str, AxisOverrideSpec]:
    """Return axis -> config override path from a final-grid manifest."""

    configured = manifest.get("axis_overrides")
    return normalize_axis_override_specs(configured, axes, context="final-grid manifest")


def _job_choices(job: dict[str, Any]) -> dict[str, Any]:
    """Return scalar final-job choices, falling back to legacy top-level fields."""

    choices = job.get("choices")
    if isinstance(choices, dict):
        return dict(choices)
    merged: dict[str, Any] = {}
    for block in (job.get("major_choices"), job.get("minor_choices")):
        if isinstance(block, dict):
            merged.update(block)
    if merged:
        return merged
    return dict(job)


def axis_value_overrides_for_job(
    job: dict[str, Any],
    *,
    scalar_axes: Sequence[str],
    override_paths: dict[str, AxisOverrideSpec],
    stage: str | None = None,
    static_stage_overrides: dict[str, Any] | None = None,
) -> list[str]:
    """Return config overrides for all scalar non-seed final-job choices."""

    choices = _job_choices(job)
    return axis_value_overrides(choices, axes=scalar_axes, override_specs=override_paths, stage=stage)


def final_train_overrides(
    job: dict[str, Any],
    *,
    study: str,
    final_run_id: str,
    attempt_id: str,
    results_root: str | Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, AxisOverrideSpec],
    static_stage_overrides: dict[str, Any] | None = None,
) -> list[str]:
    """Return OmegaConf overrides for one final training run."""

    stage_seed_overrides = job.get("stage_seed_overrides", {})
    seed_overrides = (
        stage_seed_overrides.get("final_train")
        if isinstance(stage_seed_overrides, dict)
        else None
    )
    if seed_overrides is None:
        seed_overrides = seed_override_values(None, "final_train", job)
    return [
        *axis_value_overrides_for_job(
            job,
            scalar_axes=scalar_axes,
            override_paths=override_paths,
            stage="final_train",
        ),
        *(f"{path}={value}" for path, value in seed_overrides.items()),
        *(f"{path}={format_override_value(value)}" for path, value in (static_stage_overrides or {}).items()),
        f"run.root={stage_dir(results_root, STAGE_FINAL_TRAIN)}",
        "run.layout=flat",
        f"run.run_id={final_run_id}/{attempt_id}",
        f"study.name={study}",
        "study.stage=06_final_train",
        f"study.attempt_id={attempt_id}",
        f"study.config_id={job['source_champion_id']}",
        f"experiment.name={study}",
        f"experiment.run_name={experiment_run_name(study, 'final_train')}",
    ]


def _command_for(config: str | Path, overrides: Sequence[str], *, python: str = "python") -> list[str]:
    return [python, "-u", "run.py", "--config", str(config), *overrides]


def _command_for_job(
    job: dict[str, Any],
    *,
    config: str | Path,
    study: str,
    attempt_id: str,
    results_root: Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, AxisOverrideSpec],
    static_stage_overrides: dict[str, Any] | None = None,
) -> list[str]:
    final_run_id = str(job["final_run_id"])
    attempt_dir = final_train_attempt_dir(results_root, final_run_id, attempt_id)
    command = _command_for(
        config,
        [
            *final_train_overrides(
                job,
                study=study,
                final_run_id=final_run_id,
                attempt_id=attempt_id,
                results_root=results_root,
                scalar_axes=scalar_axes,
                override_paths=override_paths,
                static_stage_overrides=static_stage_overrides,
            ),
            *_resume_overrides(attempt_dir),
        ],
    )
    return command


def _checkpoint_selection_record(attempt_dir: Path) -> dict[str, Any]:
    checkpoint_dir = attempt_dir / "checkpoints"
    return {
        "selection_policy": "latest_checkpoint_pointer",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_pointer": str(checkpoint_dir / "latest.json"),
        "resolved_checkpoint_dir": None,
    }


def _stage_plan_dir(results_root: Path, attempt_id: str) -> Path:
    """Return the durable final-train stage-plan directory."""

    return stage_dir(results_root, STAGE_FINAL_TRAIN) / "stage_plans" / attempt_id


def _executor(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    results_root: Path,
    study: str,
    log_attempt: str,
):
    """Return the toolkit executor for final-train submissions."""

    options = ExecutorOptions(
        backend=args.backend,
        args=args,
        repo_root=repo_root,
        log_dir=stage_dir(results_root, STAGE_FINAL_TRAIN) / "slurm_logs" / log_attempt,
        job_name=stage_job_name(study, "final-train"),
        smoke=False,
        chunk_size=args.chunk_size,
        claim_rows=True,
        chunk_status_dir=stage_dir(results_root, STAGE_FINAL_TRAIN) / "chunk_status" / log_attempt,
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


def build_final_train_stage_plan(
    jobs: Sequence[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    results_root: Path,
    final_grid_attempt_id: str,
    attempt_id: str,
    commands: Sequence[Sequence[str]],
    args: argparse.Namespace,
) -> StagePlan:
    """Build a reusable toolkit stage plan for final-train tasks."""

    result_dirs = [
        final_train_attempt_dir(results_root, str(job["final_run_id"]), attempt_id)
        for job in jobs
    ]
    row_status_paths = [result_dir / "launcher_status.json" for result_dir in result_dirs]
    checkpoint_paths = [result_dir / "checkpoints" / "latest.json" for result_dir in result_dirs]
    tasks = tasks_from_commands(
        stage=STAGE_FINAL_TRAIN,
        attempt_id=attempt_id,
        jobs=jobs,
        commands=commands,
        result_dirs=result_dirs,
        row_status_paths=row_status_paths,
        resources=_resource_spec(args),
        completion_policy="status_completed_with_checkpoint",
        checkpoint_paths=checkpoint_paths,
        source_attempts={"final_grid": final_grid_attempt_id},
    )
    return StagePlan(
        study=study_name_from_manifest(manifest),
        stage=STAGE_FINAL_TRAIN,
        attempt_id=attempt_id,
        results_root=str(results_root),
        source_attempts={"final_grid": final_grid_attempt_id},
        timezone=manifest.get("timezone"),
        smoke=False,
        metadata={
            "backend": args.backend,
            "device": launch.selected_device(args),
            "chunk_size": args.chunk_size,
        },
        tasks=tasks,
    )


def write_final_train_provenance(
    jobs: Sequence[dict[str, Any]],
    *,
    results_root: Path,
    final_grid_attempt_id: str,
    attempt_id: str,
    commands: Sequence[Sequence[str]],
) -> None:
    """Write per-final-run source pointers before launch."""

    grid_dir = final_grid_attempt_dir(results_root, final_grid_attempt_id)
    for job, command in zip(jobs, commands, strict=True):
        final_run_id = str(job["final_run_id"])
        attempt_dir = final_train_attempt_dir(results_root, final_run_id, attempt_id)
        write_json(
            attempt_dir / "source_final_grid_attempt.json",
            {
                "final_grid_attempt_id": final_grid_attempt_id,
                "final_grid_attempt_dir": str(grid_dir),
                "final_jobs_path": str(grid_dir / "final_jobs.csv"),
            },
        )
        write_json(attempt_dir / "source_final_job.json", job)
        write_json(attempt_dir / "source_champion.json", job.get("source_champion", {}))
        write_json(attempt_dir / "selected_checkpoint.json", _checkpoint_selection_record(attempt_dir))
        (attempt_dir / "command.txt").write_text(shlex.join([str(part) for part in command]) + "\n")
        write_latest(final_train_run_dir(results_root, final_run_id), attempt_id)


def write_final_train_submission_records(
    jobs: Sequence[dict[str, Any]],
    *,
    results_root: Path,
    final_grid_attempt_id: str,
    attempt_id: str,
    backend: str,
    job_ids: Sequence[str],
    submitted_commands: Sequence[Sequence[str]],
) -> None:
    """Write final-train submission records."""

    for index, (job, job_id) in enumerate(zip(jobs, job_ids, strict=True)):
        final_run_id = str(job["final_run_id"])
        attempt_dir = final_train_attempt_dir(results_root, final_run_id, attempt_id)
        write_json(
            attempt_dir / "submission.json",
            {
                "final_run_id": final_run_id,
                "final_grid_attempt_id": final_grid_attempt_id,
                "final_train_attempt_id": attempt_id,
                "launcher": backend,
                "launcher_job_id": str(job_id),
                "command": (attempt_dir / "command.txt").read_text().strip(),
                "submitted_command": shlex.join([str(part) for part in submitted_commands[index]]),
            },
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-train launch arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--final-grid-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--config", default=None, help="Train config path (defaults to final-grid manifest).")
    parser.add_argument(
        "--claim-existing-only",
        action="store_true",
        help=(
            "With --backend local, claim and run rows from the existing attempt "
            "without rewriting submission.json provenance."
        ),
    )
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
    """Launch final training jobs."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    if args.claim_existing_only and args.backend != "local":
        raise ValueError("--claim-existing-only is only valid with --backend local")
    repo_root = Path(args.repo_root) if args.repo_root else STUDY_DIR.parents[2]
    launch.ensure_submitit_launcher_environment(
        args,
        script_path=Path(__file__).resolve(),
        argv=raw_argv,
        repo_root=repo_root,
    )
    results_root = launch.repo_path(args.results_root, repo_root)
    final_grid_attempt_id = _resolve_final_grid_attempt_id(
        results_root,
        args.final_grid_attempt_id,
    )
    manifest = load_final_grid_manifest(results_root, final_grid_attempt_id)
    study = study_name_from_manifest(manifest)
    prefix = log_prefix(study)
    config = args.config or manifest.get("train_config")
    if not config:
        raise ValueError("final-grid manifest does not record train_config; pass --config")
    attempt_id = _attempt_id(args, final_grid_attempt_id=final_grid_attempt_id)
    jobs = _selected_jobs(load_final_jobs(results_root, final_grid_attempt_id))
    if not jobs:
        raise ValueError(f"final grid attempt {final_grid_attempt_id} has no jobs")
    jobs, completed_jobs = _exclude_completed_jobs(jobs, results_root=results_root, attempt_id=attempt_id)
    if not jobs:
        print(
            f"{prefix} no final-train rows to launch from 05_final_grid/{final_grid_attempt_id}; "
            f"excluded {len(completed_jobs)} completed rows"
        )
        return 0
    scalar_axes = final_scalar_axes(manifest)
    override_paths = final_axis_override_paths(manifest, scalar_axes)
    static_stage_overrides = {}
    configured_static_overrides = manifest.get("static_overrides")
    if isinstance(configured_static_overrides, dict):
        stage_overrides = configured_static_overrides.get("final_train") or {}
        if isinstance(stage_overrides, dict):
            static_stage_overrides = {str(path): value for path, value in stage_overrides.items()}
    commands = [
        launch.with_study_timezone(
            _command_for_job(
                job,
                config=config,
                study=study,
                attempt_id=attempt_id,
                results_root=results_root,
                scalar_axes=scalar_axes,
                override_paths=override_paths,
                static_stage_overrides=static_stage_overrides,
            )
        )
        for job in jobs
    ]
    write_final_train_provenance(
        jobs,
        results_root=results_root,
        final_grid_attempt_id=final_grid_attempt_id,
        attempt_id=attempt_id,
        commands=commands,
    )

    command_sets = launch.environment_command_sets(commands, args=args, repo_root=repo_root)
    submitted_commands = launch.summarize_command_sets(command_sets)

    log_attempt = attempt_id
    stage_plan = build_final_train_stage_plan(
        jobs,
        manifest=manifest,
        results_root=results_root,
        final_grid_attempt_id=final_grid_attempt_id,
        attempt_id=attempt_id,
        commands=commands,
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

    if args.claim_existing_only:
        print(
            f"{prefix} locally claimed {len(job_ids)} final-train rows from "
            f"05_final_grid/{final_grid_attempt_id}; submission records unchanged"
        )
        return 0

    write_final_train_submission_records(
        jobs,
        results_root=results_root,
        final_grid_attempt_id=final_grid_attempt_id,
        attempt_id=attempt_id,
        backend=args.backend,
        job_ids=job_ids,
        submitted_commands=submitted_commands,
    )
    write_execution_records(stage_plan_dir, execution_records)
    excluded = f"; excluded {len(completed_jobs)} completed rows" if completed_jobs else ""
    print(
        f"{prefix} launched {len(job_ids)} final-train jobs from "
        f"05_final_grid/{final_grid_attempt_id} via {args.backend}{excluded}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
