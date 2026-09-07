"""Rendezvous probe worker: can Accelerate initialise over a FileStore?

This is a PROBE, not a gate. It answers one question that determines the shape
of everything else in the DS-A0 spike: can ``accelerate.Accelerator`` bring up a
CPU/Gloo process group over the *file* rendezvous that DF1's subprocess harness
provides, without a launcher binary?

The question is load-bearing because of how the DDP program frames the seam. The
selected runtime "may own launch/discovery"; Accelerate's usual answer to
launch/discovery is the ``accelerate launch`` CLI, which this spike's scope
statement explicitly defers to production because running a real launcher binary
puts its children outside DF1's reap discipline. So the spike needs Accelerate to
come up under a rendezvous *TPEN* arranged. Whether it can, and at what cost in
ceded ownership, is DG0 evidence rather than an implementation detail.

WHY THREE STRATEGIES RATHER THAN ONE. Writing the strategy this author expected
to work and reporting pass/fail would answer "does my guess work", not "what does
Accelerate require". The three below cede progressively more of launch/discovery
to TPEN, so the *first* one that works is the measurement: it locates the real
boundary between what Accelerate will own here and what TPEN must keep.

Selected by ``--strategy`` through the harness's ``worker_extra_args``, one
strategy per invocation. They cannot share a process: ``init_process_group``
installs global state, so a failed attempt cannot be cleanly retried in-process,
and a second attempt in the same interpreter would measure the wreckage of the
first rather than the strategy it names.

Invoked as ``python -m tests.spikes.accelerate_ddp.probe_worker`` by
:func:`tests.helpers.ddp_subprocess_harness.run_gloo_subprocess_group` via its
``worker_module`` keyword. It accepts DF1's existing worker CLI unchanged and
carries its richer evidence in the already-provided ``--state-path`` artifact, so
``RankReceipt`` is not modified.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

# Strategy names, ordered by how much of launch/discovery TPEN has to take back.
# The ordering is the point: the first that works marks the boundary.
STRATEGY_ACCELERATE_OWNS_INIT = "accelerate-owns-init"
STRATEGY_ACCELERATE_INIT_WITH_MASTER_ENV = "accelerate-init-with-master-env"
STRATEGY_TPEN_PREINITIALIZES_GROUP = "tpen-preinitializes-group"

STRATEGIES = (
    STRATEGY_ACCELERATE_OWNS_INIT,
    STRATEGY_ACCELERATE_INIT_WITH_MASTER_ENV,
    STRATEGY_TPEN_PREINITIALIZES_GROUP,
)


def _set_common_env(rank: int, world_size: int) -> None:
    """Publish the rank identity Accelerate reads from the environment.

    Accelerate derives rank and world size from the environment rather than from
    constructor arguments, because it normally sits downstream of a launcher that
    sets them. The harness passes them as CLI arguments instead, so the worker
    republishes them here. ``LOCAL_RANK`` equals ``RANK`` because this harness
    puts every rank on one node.
    """

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    # Force the CPU path explicitly. Left unset, Accelerate's device selection
    # would depend on what the machine happens to expose, which would make this
    # probe's result a property of the node rather than of Accelerate.
    os.environ["ACCELERATE_USE_CPU"] = "1"


def _init_accelerator(strategy: str, args: argparse.Namespace) -> tuple[object, dict]:
    """Bring up Accelerate under one strategy and report what it took.

    Returns the ``Accelerator`` and a dict of observations. Raises on failure --
    the caller records the traceback, because a failure here is a RESULT, not an
    error to be swallowed.
    """

    from accelerate import Accelerator, InitProcessGroupKwargs

    rendezvous = f"file://{args.rendezvous_file}"
    timeout = timedelta(seconds=args.pg_timeout)
    observations: dict = {"strategy": strategy, "rendezvous": rendezvous}

    if strategy == STRATEGY_ACCELERATE_OWNS_INIT:
        # Most ownership ceded to Accelerate: it calls init_process_group itself,
        # and the only thing TPEN supplies is the rendezvous address.
        kwargs = InitProcessGroupKwargs(backend="gloo", init_method=rendezvous, timeout=timeout)
        accelerator = Accelerator(cpu=True, kwargs_handlers=[kwargs])
        observations["init_process_group_caller"] = "accelerate"

    elif strategy == STRATEGY_ACCELERATE_INIT_WITH_MASTER_ENV:
        # Same, plus the MASTER_ADDR/MASTER_PORT pair Accelerate's env-based
        # discovery may insist on even when an explicit init_method makes them
        # redundant. If this succeeds where the first fails, the finding is that
        # Accelerate demands launcher-shaped environment it does not actually use.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        kwargs = InitProcessGroupKwargs(backend="gloo", init_method=rendezvous, timeout=timeout)
        accelerator = Accelerator(cpu=True, kwargs_handlers=[kwargs])
        observations["init_process_group_caller"] = "accelerate"
        observations["master_env_supplied"] = True

    elif strategy == STRATEGY_TPEN_PREINITIALIZES_GROUP:
        # Least ceded: TPEN owns launch/discovery outright and hands Accelerate a
        # live process group. This is the fallback that is expected to work; if it
        # is the ONLY one that works, then Accelerate cannot own launch/discovery
        # under a TPEN-arranged rendezvous, which is a direct answer to the seam
        # question rather than a workaround.
        dist.init_process_group(
            backend="gloo",
            init_method=rendezvous,
            rank=args.rank,
            world_size=args.world_size,
            timeout=timeout,
        )
        observations["init_process_group_caller"] = "tpen"
        observations["group_initialized_before_accelerator"] = dist.is_initialized()
        accelerator = Accelerator(cpu=True)

    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown strategy {strategy!r}")

    return accelerator, observations


def _collect_evidence(accelerator, args: argparse.Namespace, observations: dict) -> dict:
    """Exercise the affordances the later gates will actually stand on."""

    import accelerate

    observations["accelerate_version"] = accelerate.__version__
    observations["torch_version"] = torch.__version__
    observations["sys_executable"] = sys.executable

    # Identity as Accelerate reports it, checked against what the harness passed.
    # Agreement is not assumed: a runtime that silently renumbers ranks would
    # corrupt every rank-local sampler and RNG decision downstream.
    observations["accelerator_process_index"] = int(accelerator.process_index)
    observations["accelerator_num_processes"] = int(accelerator.num_processes)
    observations["accelerator_local_process_index"] = int(accelerator.local_process_index)
    observations["accelerator_device"] = str(accelerator.device)
    observations["accelerator_is_main_process"] = bool(accelerator.is_main_process)
    observations["identity_matches_harness"] = (
        int(accelerator.process_index) == args.rank
        and int(accelerator.num_processes) == args.world_size
    )

    # Is the group underneath the one TPEN named, or did Accelerate build its own?
    observations["dist_is_initialized"] = dist.is_initialized()
    observations["dist_backend"] = dist.get_backend() if dist.is_initialized() else None
    observations["dist_rank"] = dist.get_rank() if dist.is_initialized() else None
    observations["dist_world_size"] = dist.get_world_size() if dist.is_initialized() else None

    # The collectives A-G* will need. Object gather is the one the statistics
    # reducer and the checkpoint coordinator both depend on.
    from accelerate.utils import broadcast_object_list, gather_object

    gathered = gather_object([args.rank])
    observations["gather_object_result"] = gathered
    observations["gather_object_is_rank_ordered"] = gathered == list(range(args.world_size))

    payload = [f"from-rank-{args.rank}"]
    broadcast_object_list(payload, from_process=0)
    observations["broadcast_object_result"] = payload[0]
    observations["broadcast_came_from_rank_0"] = payload[0] == "from-rank-0"

    accelerator.wait_for_everyone()
    observations["wait_for_everyone_returned"] = True

    return observations


def run_worker(args: argparse.Namespace) -> int:
    """Run one strategy, write the DF1 receipt, and record what was observed."""

    rank, world_size = args.rank, args.world_size
    _set_common_env(rank, world_size)

    state: dict = {
        "rank": rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "strategy": args.strategy,
        "probe_ok": False,
    }

    accelerator = None
    try:
        accelerator, observations = _init_accelerator(args.strategy, args)
        state["observations"] = _collect_evidence(accelerator, args, observations)
        state["probe_ok"] = True
    except Exception as exc:  # noqa: BLE001 - a failure here is the measurement
        # Deliberately broad and deliberately not re-raised before the state
        # write. Which strategies FAIL is exactly what this probe is for, and a
        # failure that leaves no artifact would force a rerun to learn what it
        # already knew. The exception type and full traceback are preserved.
        state["probe_ok"] = False
        state["failure_type"] = type(exc).__name__
        state["failure_message"] = str(exc)
        state["failure_traceback"] = traceback.format_exc()

    Path(args.state_path).write_text(json.dumps(state, indent=2, default=str))

    # DF1's receipt contract, unchanged. Written on this rank's own success path
    # only, atomically, exactly as the reference worker does: a kill mid-write can
    # truncate the tmp path but never the receipt path.
    receipt = {
        "rank": rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "phase_sequence": [f"PROBE_{args.strategy.replace('-', '_').upper()}"],
        "collective_result": float(world_size) if state["probe_ok"] else None,
        "fault_kind": "NONE",
    }
    receipt_path = Path(args.receipt_path)
    tmp_receipt_path = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    tmp_receipt_path.write_text(json.dumps(receipt))
    os.replace(tmp_receipt_path, receipt_path)

    if rank == 0 and state["probe_ok"]:
        Path(args.complete_marker_path).write_text("COMPLETE")

    if dist.is_initialized():
        dist.destroy_process_group()

    # Non-zero on probe failure so the process exit code carries the result too.
    # The harness records exit codes per rank, and a scheduler state of COMPLETED
    # says nothing about what happened inside.
    return 0 if state["probe_ok"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # DF1's existing worker CLI, accepted unchanged.
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rendezvous-file", type=str, required=True)
    parser.add_argument("--receipt-path", type=str, required=True)
    parser.add_argument("--state-path", type=str, required=True)
    parser.add_argument("--complete-marker-path", type=str, required=True)
    parser.add_argument("--pg-timeout", type=float, required=True)
    parser.add_argument("--fault-plan-path", type=str, default=None)
    parser.add_argument("--spawn-decoy-grandchild", action="store_true")
    parser.add_argument("--grandchild-pid-path", type=str, default=None)
    # This worker's only addition, passed through worker_extra_args.
    parser.add_argument("--strategy", type=str, required=True, choices=STRATEGIES)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_worker(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
