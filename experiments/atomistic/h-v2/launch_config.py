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
    #: K2: optional typed GRES model name (for example
    #: ``"nvidia_a100_3g.20gb"``), rendered as ``--gres=gpu:<model>:<gpus>``
    #: instead of the generic ``--gres=gpu:<gpus>``. Empty for every resource
    #: profile except :data:`GPU_TEST_RESOURCES`: science/pipeline-check
    #: output must stay byte-identical.
    gpu_model: str = ""


def packed_eval_resources(base: JobResources, pack_k: int) -> JobResources:
    """Scale host-side concurrency resources for a packed EVAL job.

    The GPU request is intentionally unchanged: the measured pilot budget is
    about 2.3 GB per stream on one GPU.  CPU and host-memory requests scale
    with the number of independent evaluator subprocesses.
    """

    if not 1 <= pack_k <= 6:
        raise ValueError(f"pack_k must be between 1 and 6, got {pack_k}")
    if pack_k == 1:
        return base
    if not base.mem.endswith("G"):
        raise ValueError(f"cannot scale non-GiB memory request {base.mem!r}")
    return JobResources(
        partition=base.partition,
        constraint=base.constraint,
        gpus=base.gpus,
        nodes=base.nodes,
        ntasks=base.ntasks,
        cpus_per_task=base.cpus_per_task * pack_k,
        mem=f"{int(base.mem[:-1]) * pack_k}G",
        no_requeue=base.no_requeue,
        gpu_model=base.gpu_model,
    )


#: Approved for both the TRAIN and EVAL job of every lineage (D1): one A100,
#: one node, 4 CPU cores, 32 GiB host memory.  Only wall time differs by
#: stage, and that is supplied per submission (T_WALL/E_WALL, section C3),
#: not carried here.
RESOURCES = JobResources(
    partition="seas_gpu,kozinsky_gpu,gpu",
    constraint="a100",
    gpus=1,
    nodes=1,
    ntasks=1,
    cpus_per_task=4,
    mem="32G",
)

# Pilot payloads retain their historical scheduler request bytes.  The science
# resource profile above is the single operator-facing partition parameter;
# pilots intentionally use this separate profile so the science expansion does
# not alter pilot payloads.
PILOT_RESOURCES = JobResources(
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

#: Per-job wall-time ceiling (section C2/C3): the `seas_gpu` partition limit,
#: and the per-job cap independent of the combined lineage cap above.
MAX_JOB_WALL_SECONDS = 2 * 24 * 60 * 60

#: The one GPU stratum every TRAIN/EVAL job must verify it was delivered
#: (section B3/C2): ``--constraint=a100`` selects this card family, but a
#: requeue can silently change it (policy
#: ``gpu-hardware-strata-and-constraint-rule-2026-08-15``), so the delivered
#: device is checked inside every job, not merely requested at submission.
APPROVED_GPU_STRATUM = "NVIDIA A100-SXM4-80GB"

#: TO 63690220 (pilot and part (g) TRAIN payloads). ``gpu-route-g`` submits
#: to the ``gpu_test`` QOS/partition instead of :data:`RESOURCES`: no
#: ``a100`` constraint (``gpu_test`` serves MIG slices, a different hardware
#: stratum -- ``gpu-hardware-strata-and-constraint-rule-2026-08-15``), and a
#: dedicated GRES request for one MIG slice.
#: K2: the exact typed GRES model FASRC documents for the ``gpu_test`` MIG
#: slice (https://docs.rc.fas.harvard.edu/kb/running-jobs/); distinct from
#: :data:`APPROVED_GPU_TEST_STRATUM` (the in-job ``torch`` device-name
#: string), which is the delivered-device check, not the scheduler request.
GPU_TEST_GRES_MODEL = "nvidia_a100_3g.20gb"
GPU_TEST_RESOURCES = JobResources(
    partition="gpu_test",
    constraint="",
    gpus=1,
    nodes=1,
    ntasks=1,
    cpus_per_task=4,
    mem="32G",
    gpu_model=GPU_TEST_GRES_MODEL,
)
#: The exact MIG device name ``gpu-route-g`` must verify it was delivered
#: (orchestrator-decisions-open-points-20261110 item 6; measured on
#: ``gpu_test`` by layer-10 verifier probe job 59353097, 2026-11-10). A
#: mismatch exits before the payload, the same way :data:`APPROVED_GPU_STRATUM`
#: does for the science/pipeline-check route.
APPROVED_GPU_TEST_STRATUM = "NVIDIA A100-SXM4-40GB MIG 3g.20gb"

#: Per-payload pilot/part-(g) TRAIN resources, lineage label, GPU stratum,
#: and ``--time`` ceiling (acceptance-contract amendment r2 P1/P2: g capped
#: at one hour, pilot capped at eight). Neither payload submits an EVAL job
#: here (item E owns that route); :func:`JobResources` carries no stage
#: distinction because each of these payloads only ever runs TRAIN.
#: Acceptance-contract amendment 20261112-r1: a fresh 0->100,000-update
#: lineage, same resources/stratum/lineage label as ``capacity-pilot``, its
#: own 12-hour TRAIN wall cap, and its own run root
#: (``<prefix>/h-v2/capacity-pilot-100k/P``, via :data:`PILOT_PAYLOAD_LINEAGES`
#: and :func:`launch._validate_run_root_prefix`). Chain-resume from the
#: existing ``capacity-pilot`` checkpoint was dropped (core refuses an
#: extended ``trainer.max_steps`` under ``train_resume``, TO note
#: ``blocker-core-resume-horizon-20261112``); this is an independent fresh
#: lineage instead.
PILOT_PAYLOAD_RESOURCES: dict[str, JobResources] = {
    "gpu-route-g": GPU_TEST_RESOURCES,
    "capacity-pilot": PILOT_RESOURCES,
    "capacity-pilot-100k": PILOT_RESOURCES,
}
PILOT_PAYLOAD_LINEAGES: dict[str, str] = {
    "gpu-route-g": "G",
    "capacity-pilot": "P",
    "capacity-pilot-100k": "P",
}
PILOT_PAYLOAD_GPU_STRATA: dict[str, str] = {
    "gpu-route-g": APPROVED_GPU_TEST_STRATUM,
    "capacity-pilot": APPROVED_GPU_STRATUM,
    "capacity-pilot-100k": APPROVED_GPU_STRATUM,
}
PILOT_PAYLOAD_MAX_WALL_SECONDS: dict[str, int] = {
    "gpu-route-g": 1 * 60 * 60,
    "capacity-pilot": 8 * 60 * 60,
    "capacity-pilot-100k": 12 * 60 * 60,
}

#: Name of the environment variable the submitter must set to the durable,
#: lifecycle-governed run-root prefix (harness r3 addendum3) before any
#: pilot/part-(g) ``sbatch`` call. The exact facility path is never
#: hardcoded in this repository (``netscratch-never-referenced-from-repo``);
#: it is supplied at invocation time, the same way ``T_WALL``/``E_WALL``
#: already are.
PILOT_RUN_ROOT_PREFIX_ENV = "H_V2_PILOT_RUN_ROOT_PREFIX"

__all__ = [
    "APPROVED_GPU_STRATUM",
    "APPROVED_GPU_TEST_STRATUM",
    "GPU_TEST_GRES_MODEL",
    "GPU_TEST_RESOURCES",
    "JOB_SCRIPT_RELATIVE",
    "JobResources",
    "LINEAGES",
    "MAX_JOB_WALL_SECONDS",
    "MAX_LINEAGE_WALL_SECONDS",
    "PILOT_PAYLOAD_GPU_STRATA",
    "PILOT_PAYLOAD_LINEAGES",
    "PILOT_PAYLOAD_MAX_WALL_SECONDS",
    "PILOT_PAYLOAD_RESOURCES",
    "PILOT_RESOURCES",
    "PILOT_RUN_ROOT_PREFIX_ENV",
    "RESOURCES",
    "packed_eval_resources",
]
