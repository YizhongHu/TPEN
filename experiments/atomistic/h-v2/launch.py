"""H-v2 study-side launch script (item 67f2eb57, design target section A4).

Per the manager's Q1 ruling (note ``manager-ruling-q1-submission-20261109``
on item 9d3aa0a3): the executor does not submit jobs, it only dispatches
inside an allocation it is already attached to. This script is the minimal
study-side submitter that closes that gap: per lineage, it runs ``sbatch``
for the training job, then ``sbatch --dependency=afterok:<train job id>``
for the evaluation job, and nothing else -- no retry, no polling, no
fallback. Each job runs ``run_job.py`` (the executor's attached route from
item 9d3aa0a3) inside its own allocation.

This script never cancels a scheduler job, and never invokes the ``tpen.run``
module directly as a script entrypoint (no ``__main__`` guard, operator
policy 2026-11-06); use the installed ``tpen`` console command instead. It
imports no ``tpen``, no ``tpen_executor``, and no ``hydra``
(``experiments/README.md``).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent


def _h_v2_module(name: str):
    """Load an h-v2 sibling under its own ``h_v2_<name>`` :mod:`sys.modules` key.

    ``experiments/`` has several same-named files across studies (notably
    ``launch.py`` itself, see ``he-v1``/``he-importance``), so a bare
    ``import`` would silently resolve to whichever study's copy loaded
    first. Mirrors ``attached.py``'s own private loader.
    """

    key = f"h_v2_{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launch_config = _h_v2_module("launch_config")


class LaunchError(RuntimeError):
    """A lineage's submission aborted before completing its two jobs."""


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), check=False, capture_output=True, text=True)


def _wrap_command(stage: str, lineage: str, *, checkout_root: Path) -> str:
    """Return the ``sh`` body for a self-contained, location-independent wrapper.

    F1: a ``sbatch`` submission only spools the SUBMITTED script's own bytes;
    it does not copy the checkout alongside it. Submitting ``run_job.py``
    directly therefore fails to find its siblings (``attached.py`` etc.) once
    Slurm runs the spooled copy. The fix is to submit this wrapper instead:
    it carries no sibling dependency of its own (every path is embedded as a
    literal, ``shlex``-quoted argument), ``cd``s into the explicit checkout,
    and ``exec``s the explicit science interpreter on the checkout's own
    ``run_job.py``. The interpreter comes from the ``$SCIENCE_PYTHON``
    environment variable rather than ``/usr/bin/env python3``, so the job
    runs in the prepared science environment, not whatever ``python3``
    happens to resolve to on the allocated node.
    """

    job_script = checkout_root / launch_config.JOB_SCRIPT_RELATIVE
    return (
        f"cd {shlex.quote(str(checkout_root))} && "
        f'exec "$SCIENCE_PYTHON" {shlex.quote(str(job_script))} '
        f"{shlex.quote(stage)} {shlex.quote(lineage)}"
    )


def build_sbatch_command(
    stage: str,
    lineage: str,
    *,
    run_root: Path,
    checkout_root: Path,
    time_value: str,
    dependency: str | None,
    commit: str | None = None,
    resources: launch_config.JobResources = launch_config.RESOURCES,
) -> list[str]:
    """Return the exact ``sbatch`` argv for one job (design target section A4).

    Parameters
    ----------
    stage : {"train", "eval"}
        Job stage.
    lineage : str
        Opaque lineage identity (for example ``"L1"``).
    run_root, checkout_root : pathlib.Path
        Durable run root (logs live under ``run_root/logs``) and the
        checkout the job script is read from.
    time_value : str
        Exact Slurm ``--time`` value (``T_WALL`` for train, ``E_WALL`` for
        eval); this module performs no wall-time arithmetic.
    dependency : str or None
        Exact ``--dependency`` value (for example ``"afterok:12345"``), or
        ``None`` for a job with no dependency.
    commit : str, optional
        Expected checkout commit (F5). When given, bound into the job's
        environment as ``--export=...,SHA=<commit>`` so the in-job identity
        check cannot be satisfied by a stale inherited ``SHA`` instead of the
        commit this script was actually told to launch.
    resources : JobResources
        Section C2 geometry; defaults to the single approved
        :data:`launch_config.RESOURCES`.

    Returns
    -------
    list of str
        The full ``sbatch`` argv: flags, then ``--wrap=<wrapper body>`` as
        the final argument (the wrapper runs the job script; see
        :func:`_wrap_command`).
    """

    if stage not in ("train", "eval"):
        raise LaunchError(f"stage must be 'train' or 'eval', got {stage!r}")
    run_root = Path(run_root)
    checkout_root = Path(checkout_root)
    job_name = f"h-v2-sci-{lineage}-{stage}"
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        f"--constraint={resources.constraint}",
        f"--gres=gpu:{resources.gpus}",
        f"--nodes={resources.nodes}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={time_value}",
    ]
    if resources.no_requeue:
        command.append("--no-requeue")
    # F5: RUN and (when known) SHA are bound explicitly here, from this
    # script's own parsed arguments, rather than left to whatever the
    # submitting shell happened to have exported. ALL still forwards the
    # rest of the submitting environment (SCIENCE_PYTHON, EXECUTOR_PY; sbatch
    # does not export by default -- facility note
    # sbatch-env-var-not-exported-by-default-20261104).
    export_parts = ["ALL", f"RUN={run_root}"]
    if commit is not None:
        export_parts.append(f"SHA={commit}")
    command.append("--export=" + ",".join(export_parts))
    command += [
        f"--output={run_root / 'logs' / '%x-%j.out'}",
        f"--error={run_root / 'logs' / '%x-%j.err'}",
    ]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(f"--wrap={_wrap_command(stage, lineage, checkout_root=checkout_root)}")
    return command


def _submit(command: Sequence[str], *, runner: Runner) -> str:
    """Run one ``sbatch`` command and return its parsed job id.

    Raises
    ------
    LaunchError
        If ``sbatch`` exits non-zero, or ``--parsable`` yields no job id.
        Either is a hard stop: the caller must not submit the dependent job.
    """

    result = runner(command)
    if result.returncode != 0:
        raise LaunchError(
            f"sbatch exited {result.returncode} for {shlex.join(command)}: {result.stderr.strip()}"
        )
    job_id = result.stdout.strip().split(";")[0]
    if not job_id:
        raise LaunchError(f"sbatch returned no job id for {shlex.join(command)}")
    return job_id


def _parse_wall_seconds(value: str) -> int:
    """Parse a Slurm ``--time`` value (``[D-]HH:MM:SS``/``HH:MM``/``MM``) into seconds.

    Raises
    ------
    LaunchError
        If ``value`` is malformed, or resolves to a non-positive duration.
    """

    remainder = value
    days = 0
    if "-" in value:
        days_part, remainder = value.split("-", 1)
        try:
            days = int(days_part)
        except ValueError:
            raise LaunchError(f"malformed wall-time value {value!r}") from None
    fields = remainder.split(":")
    try:
        parts = [int(field) for field in fields]
    except ValueError:
        raise LaunchError(f"malformed wall-time value {value!r}") from None
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = parts[0], parts[1], 0
    elif len(parts) == 1:
        hours, minutes, seconds = 0, 0, parts[0]
    else:
        raise LaunchError(f"malformed wall-time value {value!r}")
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise LaunchError(f"wall-time value {value!r} must be positive")
    return total


def _validate_wall_budget(t_wall: str, e_wall: str) -> None:
    """F4: refuse an over-budget T_WALL/E_WALL pair before any submission.

    Section C3's rule is FROZEN by D1: each of T_WALL and E_WALL is at most
    the ``seas_gpu`` ceiling (:data:`launch_config.MAX_JOB_WALL_SECONDS`),
    and their sum is at most the per-lineage cap
    (:data:`launch_config.MAX_LINEAGE_WALL_SECONDS`). Slurm itself then
    enforces the 32/96 GPU-hour budget once jobs are submitted, but nothing
    stops an over-budget pair from being accepted at submit time -- this is
    the check that refuses it before the first ``sbatch`` call.
    """

    t_seconds = _parse_wall_seconds(t_wall)
    e_seconds = _parse_wall_seconds(e_wall)
    if t_seconds > launch_config.MAX_JOB_WALL_SECONDS or e_seconds > launch_config.MAX_JOB_WALL_SECONDS:
        raise LaunchError(
            f"T_WALL and E_WALL must each be at most {launch_config.MAX_JOB_WALL_SECONDS}s "
            f"(2 days, the seas_gpu ceiling); got T_WALL={t_wall!r} ({t_seconds}s), "
            f"E_WALL={e_wall!r} ({e_seconds}s)"
        )
    total = t_seconds + e_seconds
    if total > launch_config.MAX_LINEAGE_WALL_SECONDS:
        raise LaunchError(
            f"T_WALL + E_WALL must be at most {launch_config.MAX_LINEAGE_WALL_SECONDS}s "
            f"(32 hours, design target section C3); got T_WALL={t_wall!r} + E_WALL={e_wall!r} "
            f"= {total}s"
        )


def submit_lineage(
    lineage: str,
    *,
    run_root: Path,
    checkout_root: Path,
    t_wall: str,
    e_wall: str,
    commit: str | None = None,
    runner: Runner = _default_runner,
    dry_run: bool = False,
) -> tuple[str | None, str | None]:
    """Submit one lineage's train job, then its eval job chained on it.

    Returns
    -------
    tuple of (str or None, str or None)
        The train and eval job ids, or ``(None, None)`` when ``dry_run`` is
        set: nothing is submitted, so no id exists yet, and this prints the
        exact commands (the eval command's ``--dependency`` carries a
        placeholder, since the real train job id is not known without
        submitting).

    Raises
    ------
    LaunchError
        If the train submission fails or yields no job id; the eval job is
        never submitted in that case.
    """

    train_command = build_sbatch_command(
        "train",
        lineage,
        run_root=run_root,
        checkout_root=checkout_root,
        time_value=t_wall,
        dependency=None,
        commit=commit,
    )
    if dry_run:
        print(shlex.join(train_command))
        placeholder_eval = build_sbatch_command(
            "eval",
            lineage,
            run_root=run_root,
            checkout_root=checkout_root,
            time_value=e_wall,
            dependency="afterok:<train job id>",
            commit=commit,
        )
        print(shlex.join(placeholder_eval))
        return None, None

    train_id = _submit(train_command, runner=runner)
    eval_command = build_sbatch_command(
        "eval",
        lineage,
        run_root=run_root,
        checkout_root=checkout_root,
        time_value=e_wall,
        dependency=f"afterok:{train_id}",
        commit=commit,
    )
    eval_id = _submit(eval_command, runner=runner)
    return train_id, eval_id


def submit_all(
    *,
    run_root: Path,
    checkout_root: Path,
    t_wall: str,
    e_wall: str,
    commit: str | None = None,
    lineages: Sequence[str] = launch_config.LINEAGES,
    runner: Runner = _default_runner,
    dry_run: bool = False,
) -> dict[str, tuple[str | None, str | None]]:
    """Submit every lineage's two jobs in order; abort on the first failure.

    Validates the T_WALL/E_WALL budget (F4) before any lineage is attempted.
    A lineage whose train submission fails raises before any later lineage
    is attempted: this function does no partial-progress bookkeeping beyond
    what each already-successful ``submit_lineage`` call returned to the
    caller before the failure.
    """

    _validate_wall_budget(t_wall, e_wall)
    results: dict[str, tuple[str | None, str | None]] = {}
    for lineage in lineages:
        results[lineage] = submit_lineage(
            lineage,
            run_root=run_root,
            checkout_root=checkout_root,
            t_wall=t_wall,
            e_wall=e_wall,
            commit=commit,
            runner=runner,
            dry_run=dry_run,
        )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse launch command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Durable run root (logs, checkout live under it).")
    parser.add_argument(
        "--commit",
        required=True,
        help=(
            "Expected checkout commit. Recorded, not enforced here: each job's own "
            "identity block (section B3) asserts it inside the allocation."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact sbatch commands; submit nothing.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Submit (or print, under ``--dry-run``) every lineage's two jobs."""

    args = parse_args(argv)
    run_root = Path(args.run_root).resolve()
    # F5: no custom-checkout option. run_job.py's in-job identity check always
    # asserts the checkout at RUN/checkout (section B3), so a launch-time
    # override here would submit a job whose wrapper and whose own identity
    # check disagree about where the checkout lives.
    checkout_root = run_root / "checkout"
    t_wall = os.environ.get("T_WALL")
    e_wall = os.environ.get("E_WALL")
    if not args.dry_run and (not t_wall or not e_wall):
        raise LaunchError("T_WALL and E_WALL environment variables are required to submit")
    t_wall = t_wall or "<T_WALL>"
    e_wall = e_wall or "<E_WALL>"
    results = submit_all(
        run_root=run_root,
        checkout_root=checkout_root,
        t_wall=t_wall,
        e_wall=e_wall,
        commit=args.commit,
        dry_run=args.dry_run,
    )
    for lineage, (train_id, eval_id) in results.items():
        print(f"[h-v2] {lineage}: train={train_id} eval={eval_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
