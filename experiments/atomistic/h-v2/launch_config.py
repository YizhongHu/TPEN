"""Single source of H-v2 launch resource geometry (design target section C2, D1).

Every sbatch flag the launch script (``launch.py``, item 67f2eb57) emits
comes from here, not a literal in that module, so a flag drifting from the
operator-approved geometry is a one-file diff.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JobResources:
    """One job's Slurm resource request (design target section C2, D1)."""

    partition: str
    constraint: str
    gpus: int
    nodes: int
    ntasks: int
    cpus_per_task: int
    mem: str
    no_requeue: bool = True


#: Approved for both the TRAIN and EVAL job of every lineage (D1): one A100,
#: one node, 4 CPU cores, 32 GiB host memory.  Only wall time differs by
#: stage, and that is supplied per submission (T_WALL/E_WALL, section C3),
#: not carried here.
RESOURCES = JobResources(
    partition="seas_gpu,kozinsky_gpu",
    constraint="a100",
    gpus=1,
    nodes=1,
    ntasks=1,
    cpus_per_task=4,
    mem="32G",
)

#: Three independent lineages (section C2); L2 and L3 differ only in seeds,
#: which this layer does not assign (see ``run_job.py``'s module docstring).
LINEAGES: tuple[str, ...] = ("L1", "L2", "L3")

#: In-job entrypoint, relative to the checkout root every job runs from.
JOB_SCRIPT_RELATIVE = "experiments/atomistic/h-v2/run_job.py"

#: Per-lineage wall-time cap (D1): T_wall + E_wall <= this, each <= 2 days.
MAX_LINEAGE_WALL_SECONDS = 32 * 60 * 60

__all__ = [
    "JOB_SCRIPT_RELATIVE",
    "JobResources",
    "LINEAGES",
    "MAX_LINEAGE_WALL_SECONDS",
    "RESOURCES",
]
