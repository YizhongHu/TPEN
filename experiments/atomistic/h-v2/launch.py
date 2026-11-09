"""H-v2 study-side launch script (item 67f2eb57, design target section A4).

Per the manager's Q1 ruling (note ``manager-ruling-q1-submission-20261109``
on item 9d3aa0a3): the executor does not submit jobs, it only dispatches
inside an allocation it is already attached to. This script is the minimal
study-side submitter that closes that gap: per lineage, it runs ``sbatch``
for the training job, then ``sbatch --dependency=afterok:<train job id>``
for the evaluation job, and nothing else -- no retry, no polling, no
fallback. Each job runs ``run_job.py`` (the executor's attached route from
item 9d3aa0a3) inside its own allocation.

Never ``scancel``. Never invoke the ``tpen.run`` module directly as a script
entrypoint (no ``__main__`` guard, operator policy 2026-11-06); use the
installed ``tpen`` console command instead. This module imports no ``tpen``,
no ``tpen_executor``, and no ``hydra`` (``experiments/README.md``).
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


def build_sbatch_command(
    stage: str,
    lineage: str,
    *,
    run_root: Path,
    checkout_root: Path,
    time_value: str,
    dependency: str | None,
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
    resources : JobResources
        Section C2 geometry; defaults to the single approved
        :data:`launch_config.RESOURCES`.

    Returns
    -------
    list of str
        The full ``sbatch`` argv, including the job program and its
        ``stage``/``lineage`` positional arguments.
    """

    if stage not in ("train", "eval"):
        raise LaunchError(f"stage must be 'train' or 'eval', got {stage!r}")
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
    command += [
        f"--output={run_root / 'logs' / '%x-%j.out'}",
        f"--error={run_root / 'logs' / '%x-%j.err'}",
    ]
    if dependency:
        command.append(f"--dependency={dependency}")
    command += [str(checkout_root / launch_config.JOB_SCRIPT_RELATIVE), stage, lineage]
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


def submit_lineage(
    lineage: str,
    *,
    run_root: Path,
    checkout_root: Path,
    t_wall: str,
    e_wall: str,
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
        "train", lineage, run_root=run_root, checkout_root=checkout_root, time_value=t_wall, dependency=None
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
    )
    eval_id = _submit(eval_command, runner=runner)
    return train_id, eval_id


def submit_all(
    *,
    run_root: Path,
    checkout_root: Path,
    t_wall: str,
    e_wall: str,
    lineages: Sequence[str] = launch_config.LINEAGES,
    runner: Runner = _default_runner,
    dry_run: bool = False,
) -> dict[str, tuple[str | None, str | None]]:
    """Submit every lineage's two jobs in order; abort on the first failure.

    A lineage whose train submission fails raises before any later lineage
    is attempted: this function does no partial-progress bookkeeping beyond
    what each already-successful ``submit_lineage`` call returned to the
    caller before the failure.
    """

    results: dict[str, tuple[str | None, str | None]] = {}
    for lineage in lineages:
        results[lineage] = submit_lineage(
            lineage,
            run_root=run_root,
            checkout_root=checkout_root,
            t_wall=t_wall,
            e_wall=e_wall,
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
        "--checkout-root", default=None, help="Defaults to <run-root>/checkout."
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
    checkout_root = Path(args.checkout_root).resolve() if args.checkout_root else run_root / "checkout"
    t_wall = os.environ.get("T_WALL")
    e_wall = os.environ.get("E_WALL")
    if not args.dry_run and (not t_wall or not e_wall):
        raise LaunchError("T_WALL and E_WALL environment variables are required to submit")
    t_wall = t_wall or "<T_WALL>"
    e_wall = e_wall or "<E_WALL>"
    results = submit_all(
        run_root=run_root, checkout_root=checkout_root, t_wall=t_wall, e_wall=e_wall, dry_run=args.dry_run
    )
    for lineage, (train_id, eval_id) in results.items():
        print(f"[h-v2] {lineage}: train={train_id} eval={eval_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
