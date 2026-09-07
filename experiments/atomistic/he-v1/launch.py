"""Submit planned He-v1 rows to Slurm with explicit GPU stratum validation.

Production rows carry an explicit ``--constraint``. The reduced canary uses
Cannon's current ``gpu_test`` A100-MIG profile, which has no distinguishing
node feature, and still asserts the delivered MIG device inside the allocation.

Three further rules are enforced here rather than trusted:

no restart, no resume
    Rows carry ``--no-requeue`` and their wall time is checked against the
    partition's measured ceiling at plan time. A row is sized to finish or it
    fails.

no bare ``uv``
    ``uv`` is on PATH in no shell on Cannon; a job that calls it dies
    ``ExitCode 127:0`` at ``Elapsed 00:00:00`` before running anything. The uv
    binary is a required argument and is invoked by absolute path.

nothing is deleted
    Scripts, logs, and submission records are written per row per launch
    attempt and are never rewritten in place.

Submission is opt-in: without ``--submit`` this stage writes the scripts and
the submission plan and stops, which is what makes it reviewable before H-F3
actually spends GPU-days.

This module imports no ``tpen`` (``experiments/README.md``).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

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

canary = sibling(__file__, 'canary')
layout = sibling(__file__, 'layout')

plan_stage = sibling(__file__, 'plan')
strata = sibling(__file__, 'strata')


class LaunchError(RuntimeError):
    """A row cannot be submitted as planned."""


def row_result_dir(results_root: str | Path, row: Mapping[str, Any], attempt_id: str) -> Path:
    """Return the durable directory of one row's run artifacts."""

    return layout.row_dir(results_root, str(row["stage"]), str(row["row_id"]), attempt_id)


def run_dir_for_row(results_root: str | Path, row: Mapping[str, Any], attempt_id: str) -> Path:
    """Return the run directory the driver writes into.

    The run id is pinned to the row id and the layout to ``flat``, so a row's
    run directory is a pure function of its coordinates instead of a generated
    timestamp nobody can rediscover later.
    """

    return row_result_dir(results_root, row, attempt_id) / str(row["row_id"])


def checkpoint_dir_for_eval_row(
    results_root: str | Path,
    row: Mapping[str, Any],
    attempt_id: str,
    *,
    manifest: Mapping[str, Any],
    checkpoint_sources: Mapping[str, canary.CheckpointSource] | None = None,
) -> Path:
    """Return the checkpoint directory one evaluation row restores from."""

    if row["kind"] != "eval":
        raise LaunchError(f"row {row['row_id']!r} is not an evaluation row")
    if row.get("canary_protocol") == canary.CANARY_SCHEMA:
        if checkpoint_sources is None:
            raise LaunchError("canary row requires validated external checkpoint sources")
        try:
            return canary.source_for_row(row, checkpoint_sources).checkpoint_dir
        except canary.CanaryError as exc:
            raise LaunchError(str(exc)) from exc
    train_row = plan_stage.row_by_id(manifest, str(row["depends_on"][0]))
    train_run_dir = run_dir_for_row(results_root, train_row, attempt_id)
    return train_run_dir / "checkpoints" / str(row["checkpoint_dir_name"])


def driver_command(
    row: Mapping[str, Any],
    *,
    results_root: str | Path,
    manifest: Mapping[str, Any],
    attempt_id: str,
    launch_attempt_id: str,
    checkpoint_source_map: str | Path | None = None,
    checkpoint_sources: Mapping[str, canary.CheckpointSource] | None = None,
) -> list[str]:
    """Return the driver invocation for one row."""

    driver = "train.py" if row["kind"] == "train" else "eval.py"
    command = [
        "python",
        str(STUDY_DIR / driver),
        "--results-root",
        str(results_root),
        "--plan-attempt-id",
        str(attempt_id),
        "--launch-attempt-id",
        str(launch_attempt_id),
        "--row-id",
        str(row["row_id"]),
    ]
    if row["kind"] == "eval":
        if row.get("canary_protocol") == canary.CANARY_SCHEMA:
            if checkpoint_source_map is None:
                raise LaunchError("canary row requires --checkpoint-source-map")
            # Re-resolve the row now so command construction cannot bypass the
            # all-source validation performed before any allocation is made.
            checkpoint_dir_for_eval_row(
                results_root,
                row,
                attempt_id,
                manifest=manifest,
                checkpoint_sources=checkpoint_sources,
            )
            command += ["--checkpoint-source-map", str(checkpoint_source_map)]
        else:
            command += [
                "--checkpoint-dir",
                str(
                    checkpoint_dir_for_eval_row(
                        results_root, row, attempt_id, manifest=manifest
                    )
                ),
            ]
    return command


def sbatch_directives(
    row: Mapping[str, Any],
    *,
    job_name: str,
    log_dir: Path,
    account: str | None,
    dependency: str | None,
) -> list[str]:
    """Return the ``#SBATCH`` directives of one row.

    The stratum constraint is re-derived from :mod:`strata` rather than copied
    out of the manifest, so a hand-edited manifest cannot smuggle an unpinned
    or mismatched row past this stage.
    """

    resources = row["resources"]
    partition = str(resources["partition"])
    gpus = int(resources["gpus"])
    if gpus <= 0:
        raise LaunchError(
            f"row {row['row_id']!r} requests {gpus} GPUs; this study has no CPU-only rows, "
            "so a zero-GPU row is a planning error"
        )
    validator = canary.resource_validator(partition, str(resources["stratum"]))
    resolved = validator(
        partition=partition,
        stratum_name=str(resources["stratum"]),
        timeout_min=int(resources["timeout_min"]),
    )
    declared = str(resources.get("constraint") or "")
    if declared and declared != resolved.constraint:
        raise LaunchError(
            f"row {row['row_id']!r} declares constraint {declared!r} but stratum "
            f"{resolved.name!r} pins {resolved.constraint!r}"
        )
    directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres=gpu:{gpus}",
        f"#SBATCH --cpus-per-task={int(resources['cpus'])}",
        f"#SBATCH --mem={int(resources['mem_gb'])}G",
        f"#SBATCH --time={strata.slurm_time(int(resources['timeout_min']))}",
        # Rows may not resume, so a requeued row would silently restart from
        # scratch and burn the allocation it was sized against.
        "#SBATCH --no-requeue",
        f"#SBATCH --output={log_dir / 'slurm-%j.out'}",
        f"#SBATCH --error={log_dir / 'slurm-%j.err'}",
    ]
    if resolved.constraint:
        directives.insert(2, f"#SBATCH --constraint={resolved.constraint}")
    if account:
        directives.insert(1, f"#SBATCH --account={account}")
    if dependency:
        directives.append(f"#SBATCH --dependency={dependency}")
    return directives


def build_script(
    row: Mapping[str, Any],
    *,
    directives: Sequence[str],
    command: Sequence[str],
    repo_root: Path,
    uv_bin: str,
    uv_extras: Sequence[str],
    uv_project_environment: str,
    uv_cache_root: str,
    evaluation_git_sha: str | None = None,
) -> str:
    """Return the sbatch script text for one row."""

    if not str(uv_bin).strip():
        raise LaunchError("uv binary path is required; bare 'uv' is on PATH in no shell on Cannon")
    if Path(uv_bin).name == "uv" and not Path(uv_bin).is_absolute():
        raise LaunchError(
            f"uv binary {uv_bin!r} is not an absolute path; a bare 'uv' call dies "
            "ExitCode 127:0 at Elapsed 00:00:00 before the job runs anything"
        )
    if not uv_extras:
        raise LaunchError("at least one uv extra is required; the torch build must be explicit")

    resources = row["resources"]
    uv_command = [str(uv_bin), "run", "--locked"]
    for extra in uv_extras:
        uv_command += ["--extra", str(extra)]
    uv_command += list(command)

    lines = ["#!/bin/bash", *directives, "", "set -euo pipefail", ""]
    lines += [
        "# The cache is job-local: a shared NFS uv cache caused a .nfs rename race",
        "# that failed before pytest collected anything.",
        f"export UV_PROJECT_ENVIRONMENT={shlex.quote(str(uv_project_environment))}",
        f'export UV_CACHE_DIR={shlex.quote(str(uv_cache_root))}"/${{SLURM_JOB_ID}}"',
        'mkdir -p "${UV_CACHE_DIR}"',
        f"cd {shlex.quote(str(repo_root))}",
    ]
    if evaluation_git_sha is not None:
        lines += [
            "test -n \"${SLURM_JOB_ID:-}\"",
            f"test \"$(git rev-parse HEAD)\" = {shlex.quote(evaluation_git_sha)}",
            "test -z \"$(git status --porcelain --untracked-files=no)\"",
        ]
    lines += [
        "",
        "# The constraint is what was asked for; this banner is what was delivered.",
        f'echo "[he-v1] row={row["row_id"]} kind={row["kind"]} '
        f'requested_stratum={resources["stratum"]} '
        f'requested_constraint={resources.get("constraint", "")}"',
        'echo "[he-v1] job=${SLURM_JOB_ID:-absent} node=$(hostname) '
        'partition=${SLURM_JOB_PARTITION:-absent}"',
        "",
        shlex.join(uv_command),
        "",
    ]
    return "\n".join(lines)


def submit_script(script_path: Path, *, submit: bool) -> str | None:
    """Submit one script with ``sbatch --parsable``.

    Returns
    -------
    str or None
        The Slurm job id, or ``None`` when this launch is a dry run. ``None``
        is recorded as an explicit "not submitted", never as an empty job id.
    """

    if not submit:
        return None
    completed = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise LaunchError(
            f"sbatch failed for {script_path} with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    job_id = completed.stdout.strip().split(";")[0]
    if not job_id:
        raise LaunchError(f"sbatch returned no job id for {script_path}")
    return job_id


def select_rows(
    manifest: Mapping[str, Any],
    *,
    kinds: Sequence[str],
    row_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Return the manifest rows this launch covers, in manifest order."""

    rows = [dict(row) for row in manifest["rows"]]
    if kinds:
        rows = [row for row in rows if str(row["kind"]) in set(kinds)]
    if row_ids:
        wanted = list(dict.fromkeys(str(row_id) for row_id in row_ids))
        known = {str(row["row_id"]) for row in manifest["rows"]}
        unknown = [row_id for row_id in wanted if row_id not in known]
        if unknown:
            raise LaunchError(f"row ids are not in this plan: {unknown}")
        rows = [row for row in rows if str(row["row_id"]) in set(wanted)]
    return rows


def _submission_records(attempt_dir: Path) -> dict[str, dict[str, Any]]:
    """Read the per-row submission records for one launch attempt."""

    rows_dir = attempt_dir / "rows"
    if not rows_dir.is_dir():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for record_path in sorted(rows_dir.glob("*/submission.json")):
        record = layout.read_json(record_path)
        if not isinstance(record, dict) or "row_id" not in record:
            raise LaunchError(f"invalid submission record: {record_path}")
        row_id = str(record["row_id"])
        records[row_id] = record
    return records


def launch(
    *,
    manifest: Mapping[str, Any],
    results_root: Path,
    repo_root: Path,
    rows: Sequence[Mapping[str, Any]],
    launch_attempt_id: str,
    uv_bin: str,
    uv_extras: Sequence[str],
    uv_project_environment: str,
    uv_cache_root: str,
    account: str | None,
    submit: bool,
    checkpoint_source_map: str | Path | None = None,
) -> dict[str, Any]:
    """Write scripts and submission records for ``rows`` and optionally submit.

    Evaluation rows chain on their training row with ``afterok`` when that row
    is part of the same launch. When it is not, the checkpoint the row restores
    from must already exist: a row that would start against a missing
    checkpoint fails here rather than inside the allocation.
    """

    plan_attempt_id = str(manifest["attempt_id"])
    checkpoint_sources: Mapping[str, canary.CheckpointSource] | None = None
    evaluation_git_sha: str | None = None
    if manifest.get("canary_schema") == canary.CANARY_SCHEMA:
        if checkpoint_source_map is None:
            raise LaunchError("canary launch requires --checkpoint-source-map")
        try:
            checkpoint_sources = canary.reconcile_manifest_sources(
                manifest, checkpoint_source_map
            )
        except canary.CanaryError as exc:
            raise LaunchError(str(exc)) from exc
        evaluation_git_sha = str(manifest["evaluation_git_sha"])
        _require_repo_identity(repo_root, evaluation_git_sha)
    attempt_dir = layout.launch_attempt_dir(results_root, launch_attempt_id)
    submitted_job_ids: dict[str, str] = {}
    submission_records = _submission_records(attempt_dir)
    launched_row_ids = {str(row["row_id"]) for row in rows}
    records: list[dict[str, Any]] = []

    for row in rows:
        row_id = str(row["row_id"])
        prior_record = submission_records.get(row_id)
        if prior_record is not None:
            # Dry-run records are durable review artifacts; refusing to replace
            # one preserves the no-rewrite invariant and requires a new attempt.
            if (
                prior_record.get("submitted") is False
                and prior_record.get("job_id") is None
            ):
                prior = "prior record was a dry run (not submitted)"
            else:
                prior = f"prior job id {prior_record.get('job_id')!r}"
            raise LaunchError(
                f"row {row_id!r} already has a submission record in launch attempt "
                f"{launch_attempt_id!r}; {prior}"
            )
        row_launch_dir = attempt_dir / "rows" / row_id
        log_dir = row_launch_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        result_dir = row_result_dir(results_root, row, plan_attempt_id)
        result_dir.mkdir(parents=True, exist_ok=True)

        dependency = _dependency_for_row(
            row,
            manifest=manifest,
            results_root=results_root,
            plan_attempt_id=plan_attempt_id,
            submitted_job_ids=submitted_job_ids,
            launched_row_ids=launched_row_ids,
            submit=submit,
        )
        command = driver_command(
            row,
            results_root=results_root,
            manifest=manifest,
            attempt_id=plan_attempt_id,
            launch_attempt_id=launch_attempt_id,
            checkpoint_source_map=checkpoint_source_map,
            checkpoint_sources=checkpoint_sources,
        )
        directives = sbatch_directives(
            row,
            job_name=f"he-v1-{row_id}",
            log_dir=log_dir,
            account=account,
            dependency=dependency,
        )
        script = build_script(
            row,
            directives=directives,
            command=command,
            repo_root=repo_root,
            uv_bin=uv_bin,
            uv_extras=uv_extras,
            uv_project_environment=uv_project_environment,
            uv_cache_root=uv_cache_root,
            evaluation_git_sha=evaluation_git_sha,
        )
        script_path = row_launch_dir / "sbatch.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o755)

        job_id = submit_script(script_path, submit=submit)
        if job_id is not None:
            submitted_job_ids[row_id] = job_id

        record = {
            "row_id": row_id,
            "kind": str(row["kind"]),
            "plan_attempt_id": plan_attempt_id,
            "launch_attempt_id": launch_attempt_id,
            "partition": str(row["resources"]["partition"]),
            "requested_stratum": str(row["resources"]["stratum"]),
            "requested_constraint": strata.constraint_for(str(row["resources"]["stratum"])),
            "evaluation_git_sha": evaluation_git_sha,
            "checkpoint_source_map_sha256": manifest.get("source_map_sha256"),
            "timeout_min": int(row["resources"]["timeout_min"]),
            "gpus": int(row["resources"]["gpus"]),
            "account": account,
            "dependency": dependency,
            "script_path": str(script_path),
            "log_dir": str(log_dir),
            "result_dir": str(result_dir),
            "run_dir": str(run_dir_for_row(results_root, row, plan_attempt_id)),
            "command": command,
            "submitted": bool(job_id is not None),
            "job_id": job_id,
            "requeue": False,
            "resume": False,
        }
        layout.write_json(row_launch_dir / "submission.json", record)
        submission_records[row_id] = record
        records.append(record)

    summary = {
        "schema_version": plan_stage.SCHEMA_VERSION,
        "study": str(manifest["study"]),
        "plan_attempt_id": plan_attempt_id,
        "plan_hash": str(manifest["plan_hash"]),
        "launch_attempt_id": launch_attempt_id,
        "created_at": datetime.now(ZoneInfo(plan_stage.STUDY_TIMEZONE)).isoformat(),
        "timezone": plan_stage.STUDY_TIMEZONE,
        "submitted": bool(submit),
        "n_rows": len(records),
        "n_submitted": sum(1 for record in records if record["submitted"]),
        "uv_bin": str(uv_bin),
        "uv_extras": [str(extra) for extra in uv_extras],
        "resume_policy": "forbidden",
        "evaluation_git_sha": evaluation_git_sha,
        "checkpoint_source_map_sha256": manifest.get("source_map_sha256"),
        "rows": records,
    }
    layout.write_json(attempt_dir / "submissions.json", summary)
    layout.write_latest(layout.stage_dir(results_root, layout.STAGE_LAUNCH), launch_attempt_id)
    return summary


def _require_repo_identity(repo_root: Path, expected_sha: str) -> None:
    """Require the exact clean evaluation checkout before script allocation."""

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode != 0 or head.stdout.strip() != expected_sha:
        raise LaunchError(
            f"canary checkout must be exact SHA {expected_sha}; got {head.stdout.strip()!r}"
        )
    if status.returncode != 0 or status.stdout.strip():
        raise LaunchError("canary checkout must have a clean tracked tree before allocation")


def _dependency_for_row(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    results_root: Path,
    plan_attempt_id: str,
    submitted_job_ids: Mapping[str, str],
    launched_row_ids: set[str],
    submit: bool,
) -> str | None:
    """Return the Slurm dependency of one row, validating its inputs."""

    depends_on = [str(row_id) for row_id in row.get("depends_on", [])]
    if not depends_on:
        return None
    parent_id = depends_on[0]
    if parent_id in launched_row_ids:
        job_id = submitted_job_ids.get(parent_id)
        if job_id is not None:
            return f"afterok:{job_id}"
        if submit:
            raise LaunchError(
                f"row {row['row_id']!r} depends on {parent_id!r}, which is in this launch "
                "but has no job id yet; submit training rows before their evaluations"
            )
        return f"afterok:<{parent_id}>"
    checkpoint_dir = checkpoint_dir_for_eval_row(
        results_root, row, plan_attempt_id, manifest=manifest
    )
    if not checkpoint_dir.is_dir():
        raise LaunchError(
            f"row {row['row_id']!r} restores {checkpoint_dir}, which does not exist, and its "
            f"training row {parent_id!r} is not part of this launch"
        )
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse launch command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, help="Durable study results root.")
    parser.add_argument("--repo-root", required=True, help="Checkout the rows run from.")
    parser.add_argument("--plan-attempt-id", default=None, help="Plan attempt (defaults to latest).")
    parser.add_argument("--launch-attempt-id", default=None, help="Explicit launch attempt id.")
    parser.add_argument(
        "--kind",
        action="append",
        choices=["train", "eval"],
        default=[],
        help="Restrict to one row kind; repeatable.",
    )
    parser.add_argument("--row-id", action="append", default=[], help="Restrict to row ids.")
    parser.add_argument(
        "--checkpoint-source-map",
        default=None,
        help="External immutable source map required by a canary plan.",
    )
    parser.add_argument(
        "--uv-bin",
        required=True,
        help="Absolute path to the uv binary; bare 'uv' resolves in no shell on Cannon.",
    )
    parser.add_argument(
        "--uv-extra",
        action="append",
        required=True,
        help="uv extra selecting the torch build; repeatable, no default.",
    )
    parser.add_argument(
        "--uv-project-environment",
        required=True,
        help="UV_PROJECT_ENVIRONMENT for the rows (keep it off $HOME).",
    )
    parser.add_argument(
        "--uv-cache-root",
        required=True,
        help="Parent of the job-local UV_CACHE_DIR (keep it off $HOME and off a shared cache).",
    )
    parser.add_argument("--account", default=None, help="Slurm account, when one is required.")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually call sbatch. Without it this stage writes scripts and stops.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write per-row sbatch scripts for one plan attempt."""

    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    plan_attempt_id = layout.resolve_attempt_id(results_root, layout.STAGE_PLAN, args.plan_attempt_id)
    manifest = plan_stage.read_manifest(results_root, plan_attempt_id)
    rows = select_rows(manifest, kinds=args.kind, row_ids=args.row_id)
    if not rows:
        print(f"[he-v1] plan attempt {plan_attempt_id} selected no rows")
        return 0
    launch_attempt_id = args.launch_attempt_id or plan_stage.now_attempt_id()
    summary = launch(
        manifest=manifest,
        results_root=results_root,
        repo_root=repo_root,
        rows=rows,
        launch_attempt_id=launch_attempt_id,
        uv_bin=args.uv_bin,
        uv_extras=args.uv_extra,
        uv_project_environment=args.uv_project_environment,
        uv_cache_root=args.uv_cache_root,
        account=args.account,
        submit=args.submit,
        checkpoint_source_map=args.checkpoint_source_map,
    )
    mode = "submitted" if summary["submitted"] else "prepared (dry run)"
    print(
        f"[he-v1] {mode} {summary['n_rows']} rows for plan {plan_attempt_id} "
        f"as launch attempt {launch_attempt_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
