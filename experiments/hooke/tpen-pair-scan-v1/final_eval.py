"""Launch final evaluation from final train attempts.

Final evaluation consumes ``05_final_grid`` and completed ``06_final_train``
attempts. It records the exact final-train checkpoint directory evaluated and
launches the named ``final_eval`` suite from the final-grid eval config.
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
_tpen_final_train = sibling(__file__, 'final_train')
final_axis_override_paths = _tpen_final_train.final_axis_override_paths
final_scalar_axes = _tpen_final_train.final_scalar_axes
axis_value_overrides_for_job = _tpen_final_train.axis_value_overrides_for_job
load_final_grid_manifest = _tpen_final_train.load_final_grid_manifest
load_final_jobs = _tpen_final_train.load_final_jobs
_tpen_utils_io = sibling(__file__, 'utils.io')
write_json = _tpen_utils_io.write_json
_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_FINAL_EVAL = _tpen_utils_layout.STAGE_FINAL_EVAL
STAGE_FINAL_GRID = _tpen_utils_layout.STAGE_FINAL_GRID
attempt_ids = _tpen_utils_layout.attempt_ids
final_eval_attempt_dir = _tpen_utils_layout.final_eval_attempt_dir
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
from experiments.toolkit.task_state import _resolved_checkpoint  # noqa: E402

def _resolve_final_grid_attempt_id(results_root: Path, requested: str | None) -> str:
    if requested is not None:
        return requested
    final_grid_stage = stage_dir(results_root, STAGE_FINAL_GRID)
    attempt_id = latest_attempt_id(final_grid_stage)
    if attempt_id is None:
        raise FileNotFoundError(f"no final-grid attempts under {final_grid_stage}")
    return attempt_id


def _selected_jobs(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(job) for job in jobs]


def _attempt_id(args: argparse.Namespace, *, final_grid_attempt_id: str) -> str:
    if args.attempt_id:
        return args.attempt_id
    return final_grid_attempt_id


def latest_final_train_attempt_id(
    results_root: str | Path,
    final_run_id: str,
) -> str | None:
    """Return the latest eligible final-train attempt id for ``final_run_id``."""

    return latest_attempt_id(final_train_run_dir(results_root, final_run_id))


def _final_train_attempt_id_for_job(
    *,
    args: argparse.Namespace,
    results_root: Path,
    final_run_id: str,
) -> str | None:
    if args.final_train_attempt_id is not None:
        return args.final_train_attempt_id
    return _latest_ready_final_train_attempt_id(results_root, final_run_id)


def _latest_ready_final_train_attempt_id(
    results_root: str | Path,
    final_run_id: str,
) -> str | None:
    """Return the newest final-train attempt with a completed selected checkpoint."""

    run_dir = final_train_run_dir(results_root, final_run_id)
    preferred = latest_attempt_id(run_dir)
    ids = attempt_ids(run_dir)
    ordered = []
    if preferred is not None:
        ordered.append(preferred)
    ordered.extend(attempt_id for attempt_id in reversed(ids) if attempt_id != preferred)
    for attempt_id in ordered:
        train_attempt = final_train_attempt_dir(results_root, final_run_id, attempt_id)
        if _resolved_checkpoint(train_attempt) is not None:
            return attempt_id
    return None


def final_eval_overrides(
    job: dict[str, Any],
    *,
    study: str,
    final_run_id: str,
    attempt_id: str,
    results_root: str | Path,
    checkpoint_dir: str | Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, AxisOverrideSpec],
) -> list[str]:
    """Return OmegaConf overrides for one final evaluation run."""

    stage_seed_overrides = job.get("stage_seed_overrides", {})
    seed_overrides = (
        stage_seed_overrides.get("final_eval")
        if isinstance(stage_seed_overrides, dict)
        else None
    )
    if seed_overrides is None:
        seed_overrides = seed_override_values(None, "final_eval", job)
    return [
        *axis_value_overrides_for_job(
            job,
            scalar_axes=scalar_axes,
            override_paths=override_paths,
            stage="final_eval",
        ),
        *(f"{path}={value}" for path, value in seed_overrides.items()),
        "evaluation.suite=final_eval",
        f"load.path={checkpoint_dir}",
        f"run.root={stage_dir(results_root, STAGE_FINAL_EVAL)}",
        "run.layout=flat",
        f"run.run_id={final_run_id}/{attempt_id}",
        f"study.name={study}",
        "study.stage=07_final_eval",
        f"study.attempt_id={attempt_id}",
        f"study.config_id={job['source_champion_id']}",
        f"experiment.name={study}",
        f"experiment.run_name={experiment_run_name(study, 'final_eval')}",
    ]


def _command_for(config: str | Path, overrides: Sequence[str], *, python: str = "python") -> list[str]:
    return [python, "-u", "run.py", "--config", str(config), *overrides]


def plan_final_eval_jobs(
    jobs: Sequence[dict[str, Any]],
    *,
    args: argparse.Namespace,
    study: str,
    results_root: Path,
    final_grid_attempt_id: str,
    eval_config: str | Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, AxisOverrideSpec],
    static_stage_overrides: dict[str, object] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build final-eval launch records and write source provenance."""

    selected = _selected_jobs(jobs)
    final_eval_attempt_id = _attempt_id(args, final_grid_attempt_id=final_grid_attempt_id)
    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    grid_dir = final_grid_attempt_dir(results_root, final_grid_attempt_id)
    for job in selected:
        final_run_id = str(job["final_run_id"])
        train_attempt_id = _final_train_attempt_id_for_job(
            args=args,
            results_root=results_root,
            final_run_id=final_run_id,
        )
        if train_attempt_id is None:
            skipped.append({"final_run_id": final_run_id, "reason": "no eligible final-train attempt"})
            continue
        train_attempt = final_train_attempt_dir(results_root, final_run_id, train_attempt_id)
        checkpoint = _resolved_checkpoint(train_attempt)
        if checkpoint is None:
            skipped.append({"final_run_id": final_run_id, "reason": f"missing selected checkpoint in {train_attempt}"})
            continue

        final_eval_attempt = final_eval_attempt_dir(results_root, final_run_id, final_eval_attempt_id)
        write_json(
            final_eval_attempt / "source_final_grid_attempt.json",
            {
                "final_grid_attempt_id": final_grid_attempt_id,
                "final_grid_attempt_dir": str(grid_dir),
                "final_jobs_path": str(grid_dir / "final_jobs.csv"),
            },
        )
        write_json(
            final_eval_attempt / "source_final_train_attempt.json",
            {
                "final_run_id": final_run_id,
                "final_train_attempt_id": train_attempt_id,
                "final_train_attempt_dir": str(train_attempt),
                "checkpoint": checkpoint,
            },
        )
        write_json(final_eval_attempt / "source_final_job.json", job)
        write_json(final_eval_attempt / "source_champion.json", job.get("source_champion", {}))
        write_json(final_eval_attempt / "evaluated_checkpoint.json", checkpoint)

        command = _command_for(
            eval_config,
            final_eval_overrides(
                job,
                study=study,
                final_run_id=final_run_id,
                attempt_id=final_eval_attempt_id,
                results_root=results_root,
                checkpoint_dir=checkpoint["resolved_checkpoint_dir"],
                scalar_axes=scalar_axes,
                override_paths=override_paths,
            ),
        )
        command = launch.with_study_timezone(command)
        command = launch.with_overrides(command, static_stage_overrides or {})
        (final_eval_attempt / "command.txt").write_text(shlex.join(command) + "\n")
        write_latest(final_eval_attempt.parent, final_eval_attempt_id)
        planned.append(
            {
                "final_run_id": final_run_id,
                "final_grid_attempt_id": final_grid_attempt_id,
                "final_train_attempt_id": train_attempt_id,
                "final_eval_attempt_id": final_eval_attempt_id,
                "final_eval_attempt_dir": str(final_eval_attempt),
                "checkpoint": checkpoint,
                "command": shlex.join(command),
                "command_parts": command,
            }
        )
    return planned, skipped


def _stage_plan_dir(results_root: Path, attempt_id: str) -> Path:
    """Return the durable final-eval stage-plan directory."""

    return stage_dir(results_root, STAGE_FINAL_EVAL) / "stage_plans" / attempt_id


def _executor(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    results_root: Path,
    study: str,
    log_attempt: str,
):
    """Return the toolkit executor for final-eval submissions."""

    options = ExecutorOptions(
        backend=args.backend,
        args=args,
        repo_root=repo_root,
        log_dir=stage_dir(results_root, STAGE_FINAL_EVAL) / "slurm_logs" / log_attempt,
        job_name=stage_job_name(study, "final-eval"),
        smoke=False,
        chunk_size=args.chunk_size,
        allow_partial_failures=True,
        chunk_status_dir=stage_dir(results_root, STAGE_FINAL_EVAL) / "chunk_status" / log_attempt,
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


def build_final_eval_stage_plan(
    jobs: Sequence[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    results_root: Path,
    final_grid_attempt_id: str,
    attempt_id: str,
    args: argparse.Namespace,
) -> StagePlan:
    """Build a reusable toolkit stage plan for final-eval tasks."""

    result_dirs = [Path(str(job["final_eval_attempt_dir"])) for job in jobs]
    row_status_paths = [result_dir / "launcher_status.json" for result_dir in result_dirs]
    tasks = tasks_from_commands(
        stage=STAGE_FINAL_EVAL,
        attempt_id=attempt_id,
        jobs=jobs,
        commands=[job["command_parts"] for job in jobs],
        result_dirs=result_dirs,
        row_status_paths=row_status_paths,
        resources=_resource_spec(args),
        completion_policy="status_completed",
        source_attempts={"final_grid": final_grid_attempt_id},
    )
    return StagePlan(
        study=study_name_from_manifest(manifest),
        stage=STAGE_FINAL_EVAL,
        attempt_id=attempt_id,
        results_root=str(results_root),
        source_attempts={"final_grid": final_grid_attempt_id},
        timezone=manifest.get("timezone"),
        smoke=False,
        metadata={
            "backend": args.backend,
            "device": launch.selected_device(args),
            "chunk_size": args.chunk_size,
            "skips_allowed": True,
        },
        tasks=tasks,
    )


def write_final_eval_submission_records(
    jobs: Sequence[dict[str, Any]],
    *,
    backend: str,
    job_ids: Sequence[str],
    submitted_commands: Sequence[Sequence[str]],
) -> None:
    """Write final-eval submission provenance."""

    for index, (job, job_id) in enumerate(zip(jobs, job_ids, strict=True)):
        final_eval_attempt = Path(str(job["final_eval_attempt_dir"]))
        write_json(
            final_eval_attempt / "submission.json",
            {
                "final_run_id": str(job["final_run_id"]),
                "final_grid_attempt_id": str(job["final_grid_attempt_id"]),
                "final_train_attempt_id": str(job["final_train_attempt_id"]),
                "final_eval_attempt_id": str(job["final_eval_attempt_id"]),
                "launcher": backend,
                "launcher_job_id": str(job_id),
                "command": str(job["command"]),
                "submitted_command": shlex.join([str(part) for part in submitted_commands[index]]),
            },
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-eval launch arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--final-grid-attempt-id", default=None)
    parser.add_argument("--final-train-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--config", default=None, help="Eval config path (defaults to final-grid manifest).")
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
    """Launch final evaluation jobs."""

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
    final_grid_attempt_id = _resolve_final_grid_attempt_id(
        results_root,
        args.final_grid_attempt_id,
    )
    manifest = load_final_grid_manifest(results_root, final_grid_attempt_id)
    study = study_name_from_manifest(manifest)
    prefix = log_prefix(study)
    if args.wait_job:
        launch.submit_dependent_launcher(
            args.wait_job,
            script_path=Path(__file__).resolve(),
            argv=raw_argv,
            repo_root=repo_root,
            log_dir=stage_dir(results_root, STAGE_FINAL_EVAL) / "slurm_logs" / "dependent_launchers",
            job_name=stage_job_name(study, "final-eval-launcher"),
            partition=args.wait_launcher_partition,
            timeout_min=args.wait_launcher_timeout_min,
            study=study,
        )
        return 0
    eval_config = args.config or manifest.get("eval_config")
    if not eval_config:
        raise ValueError("final-grid manifest does not record eval_config; pass --config")
    scalar_axes = final_scalar_axes(manifest)
    override_paths = final_axis_override_paths(manifest, scalar_axes)
    static_stage_overrides = launch.static_overrides_for_stage("final_eval", manifest=manifest)
    jobs, skipped = plan_final_eval_jobs(
        load_final_jobs(results_root, final_grid_attempt_id),
        args=args,
        study=study,
        results_root=results_root,
        final_grid_attempt_id=final_grid_attempt_id,
        eval_config=eval_config,
        scalar_axes=scalar_axes,
        override_paths=override_paths,
        static_stage_overrides=static_stage_overrides,
    )
    command_sets = launch.environment_command_sets(
        [job["command_parts"] for job in jobs],
        args=args,
        repo_root=repo_root,
    )
    submitted_commands = launch.summarize_command_sets(command_sets)

    if skipped:
        print(f"{prefix} skipped {len(skipped)} final-eval jobs without ready final-train checkpoints")
    if not jobs:
        print(f"{prefix} no final-eval jobs ready for 05_final_grid/{final_grid_attempt_id}")
        return 1 if manifest.get("n_jobs") else 0

    attempt_id = str(jobs[0]["final_eval_attempt_id"])
    stage_plan = build_final_eval_stage_plan(
        jobs,
        manifest=manifest,
        results_root=results_root,
        final_grid_attempt_id=final_grid_attempt_id,
        attempt_id=attempt_id,
        args=args,
    )
    stage_plan_dir = stage_plan.write(_stage_plan_dir(results_root, attempt_id))
    execution_records = _executor(
        args=args,
        repo_root=repo_root,
        results_root=results_root,
        study=study,
        log_attempt=attempt_id,
    ).submit(
        stage_plan,
        stage_plan.tasks,
        SubmissionRequest(
            command_sets=command_sets,
            submitted_commands=submitted_commands,
        ),
    )
    job_ids = [record.launcher_job_id for record in execution_records]

    write_final_eval_submission_records(
        jobs,
        backend=args.backend,
        job_ids=job_ids,
        submitted_commands=submitted_commands,
    )
    write_execution_records(stage_plan_dir, execution_records)
    print(f"{prefix} launched {len(job_ids)} final-eval jobs from 05_final_grid/{final_grid_attempt_id} via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
