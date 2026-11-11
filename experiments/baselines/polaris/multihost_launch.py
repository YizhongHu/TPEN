#!/usr/bin/env python3
"""Present a PBS/PALS allocation to DeepQMC as if it were Slurm, so multi-host DDP fires.

Run as the command under ``mpiexec``, one invocation per rank::

    mpiexec -n 2 --ppn 1 --cpu-bind none \\
        python -m experiments.baselines.polaris.multihost_launch \\
        "$VENV/bin/deepqmc" hydra.run.dir="$D" ansatz=deeperwin task.seed=0

Why a translation layer is needed
---------------------------------
DeepQMC detects peers **only** from Slurm: ``parallel.py maybe_init_multi_host()``
reads ``SLURM_NTASKS``/``SLURM_PROCID`` and calls ``jax.distributed.initialize()``
only when the count exceeds one, its docstring stating detection is "only
implemented for SLURM". Polaris runs PBS with the PALS launcher.

Without translation every rank initialises alone, and it does so **silently**: each
process runs as an independent job, all of them succeed, and the elapsed time looks
like a wide measurement while being nothing of the sort.

Why these five variables
------------------------
``jax`` 0.8.3's ``SlurmCluster.is_env_present()`` requires *all* of ``SLURM_JOB_ID``,
``SLURM_STEP_NODELIST``, ``SLURM_NTASKS``, ``SLURM_PROCID`` and ``SLURM_LOCALID``.
Miss one and no cluster is detected -- and because DeepQMC calls ``initialize()``
without a ``coordinator_address``, detection is the only thing that supplies it.
``jax`` then derives the coordinator host from the *first* entry of the node list and
the port as ``SLURM_JOB_ID % 4096 + 61440``, so the job id must be an integer.

The load-bearing detail: ``jax`` parses that node list in pure Python and never
shells out to ``scontrol``. That is the only reason a machine with no Slurm installed
can satisfy a Slurm-only code path.

Trap: PALS pins each rank to one core
-------------------------------------
``mpiexec`` under PALS delivers ``cores=0`` -- one core of 64 -- by default. Measured
on Polaris job 7588647 (He deeperwin, 4 GPUs, marginal s/step): direct 0.1125,
``mpiexec`` default 0.2975, ``mpiexec --cpu-bind none`` 0.1500. A 2.0-2.6x slowdown,
and **nothing fails** -- the job completes and simply takes far longer, because
DeepQMC is host-bound for sampling and dispatch.

Pass ``--cpu-bind none`` on the *mpiexec* command. This module runs inside mpiexec
and cannot set it. It reports the affinity it actually received so the trap appears
in the log rather than being inferred from a disappointing wall time.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping


def slurm_env_from_pbs(environ: Mapping[str, str], nodefile_text: str) -> dict[str, str]:
    """Translate a PBS/PALS environment into the Slurm variables ``jax`` requires.

    Pure: performs no I/O and does not read the real environment, so the mapping can
    be tested directly rather than through a subprocess.

    Parameters
    ----------
    environ : Mapping of str to str
        The PBS/PALS environment, normally ``os.environ``.
    nodefile_text : str
        Contents of ``PBS_NODEFILE``. PBS repeats a hostname once per slot and its
        order is not guaranteed.

    Returns
    -------
    dict of str to str
        The five ``SLURM_*`` variables plus ``CUDA_VISIBLE_DEVICES``.

    Raises
    ------
    KeyError
        If ``PBS_JOBID`` is absent, i.e. this is not a PBS allocation. Failing here
        is deliberate: outside PBS the mapping would be silently wrong, and a silent
        wrong mapping produces independent single-rank runs that all look successful.
    """
    # A PBS job id looks like "7587171.polaris-pbs-01.hsn..."; jax needs the integer
    # because it computes the coordination port as jobid % 4096 + 61440.
    job_id = environ["PBS_JOBID"].split(".", 1)[0]

    # Every rank must choose the SAME coordinator, and jax takes the first entry.
    # Sorting and de-duplicating identically on every rank is what makes them agree.
    hosts = sorted({line.strip() for line in nodefile_text.splitlines() if line.strip()})
    if not hosts:
        raise ValueError("PBS_NODEFILE contained no hostnames")

    # PMI_* is what PALS sets; PALS_* is the fallback spelling. Defaulting NTASKS to
    # 1 matters: DeepQMC starts distributed init only when it exceeds 1, so a wrong
    # value here makes a lone process wait on a coordinator handshake that no peer
    # will ever complete.
    ntasks = environ.get("PMI_SIZE") or environ.get("PALS_NRANKS") or "1"
    procid = environ.get("PMI_RANK") or environ.get("PALS_RANKID") or "0"
    localid = environ.get("PALS_LOCAL_RANKID") or "0"

    return {
        "SLURM_JOB_ID": job_id,
        "SLURM_STEP_NODELIST": ",".join(hosts),
        "SLURM_NTASKS": str(ntasks),
        "SLURM_PROCID": str(procid),
        "SLURM_LOCALID": str(localid),
        # initialize() asserts this is set and parses it as the local device id list.
        "CUDA_VISIBLE_DEVICES": environ.get("MH_VISIBLE", "0,1,2,3"),
    }


def _affinity() -> str:
    """CPU affinity of this process, or a marker if the platform cannot report it."""
    try:
        return ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - non-Linux
        return "unavailable"


def main(argv: list[str]) -> int:
    """Set the translated environment and ``exec`` the wrapped command."""
    if not argv:
        print("usage: multihost_launch.py <command> [args...]", file=sys.stderr)
        return 2
    try:
        nodefile = os.environ["PBS_NODEFILE"]
    except KeyError:
        print("PBS_NODEFILE unset -- this must run inside a PBS allocation", file=sys.stderr)
        return 2
    try:
        with open(nodefile, encoding="utf-8") as handle:
            mapped = slurm_env_from_pbs(os.environ, handle.read())
    except KeyError as exc:
        print(f"{exc.args[0]} unset -- this must run inside a PBS allocation", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env.update(mapped)

    affinity = _affinity()
    n_cores = len(affinity.split(",")) if affinity != "unavailable" else 0
    print(
        f"MHLAUNCH rank={mapped['SLURM_PROCID']}/{mapped['SLURM_NTASKS']} "
        f"host={os.uname().nodename} cores={affinity} ncores={n_cores} "
        f"vis={mapped['CUDA_VISIBLE_DEVICES']} jobid={mapped['SLURM_JOB_ID']}",
        file=sys.stderr,
        flush=True,
    )
    if n_cores == 1:
        # The 2.6x trap, called out at the moment it happens.
        print(
            "MHLAUNCH WARNING: pinned to a single core. PALS mpiexec does this by "
            "default and it costs ~2.6x on DeepQMC. Pass --cpu-bind none to mpiexec.",
            file=sys.stderr,
            flush=True,
        )
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
