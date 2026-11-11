"""Child-process worker for the CPU/Gloo subprocess harness (DF1).

Invoked as ``python -m tests.helpers.ddp_worker_entrypoint <args>``, always
as a genuinely separate OS process launched by
:mod:`tests.helpers.ddp_subprocess_harness` -- never in-process, and never
via ``torch.multiprocessing.spawn``, so the pytest process itself is never a
distributed worker.

Runs a fixed, minimal synthetic step sequence (one collective, one fake
optimizer-style update, one rank-local state write, one group-wide
publication barrier) through all 8 typed :class:`FaultPhase` hook points.
This body exists only to exercise the harness's safety properties -- it is
not a VMC training step and proves nothing about
``tpen.training.vmc.compute_vmc_objective`` (Stage T2, out of scope here).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from tests.helpers.ddp_fault_injection import (
    COLLECTIVE_FAULT_KINDS,
    FaultKind,
    FaultPhase,
    FaultPlan,
    read_fault_plan,
)


def _report_fault_applied(rank: int, phase_name: str, plan: FaultPlan) -> None:
    """Self-attribute this rank as the fault's true target, to stderr.

    This is the ONLY evidence
    :func:`tests.helpers.ddp_subprocess_harness.run_gloo_subprocess_group`
    uses to derive ``HarnessResult.culprit_rank`` after collection. Called
    the instant a rank's own code path matches the plan's target rank and
    phase -- before the fault's effect (raise, exit, or stall) can take
    hold -- so the report is durable (flushed to this rank's own captured
    log) even on a path that crashes before reaching its own receipt write.
    A peer that merely fails as a downstream consequence (e.g. blocked on a
    collective the culprit skipped, until its own process-group timeout
    fires) never emits this line about itself, so it is never mistaken for
    the culprit even when its own exit code and traceback are otherwise
    indistinguishable from a genuine fault.
    """

    print(
        f"ddp harness injected fault: rank {rank} phase {phase_name} kind {plan.kind.name}",
        file=sys.stderr,
        flush=True,
    )


def _maybe_apply_fault(phase: FaultPhase, rank: int, plan: FaultPlan | None) -> None:
    """Apply ``plan``'s fault if it targets this rank and this exact phase."""

    if plan is None or plan.kind == FaultKind.NONE:
        return
    if plan.target_rank != rank:
        return
    if phase != plan.phase:
        return
    _report_fault_applied(rank, phase.name, plan)
    if plan.kind == FaultKind.RAISE_BEFORE_BACKWARD:
        raise RuntimeError(f"ddp harness injected fault: rank {rank} phase {phase.name}")
    if plan.kind in (FaultKind.CRASH_AFTER_PUBLISH, FaultKind.CRASH_DURING_CHECKPOINT):
        os._exit(1)
    if plan.kind == FaultKind.STALL_BEFORE_COLLECTIVE:
        time.sleep(plan.delay_seconds)


def _enter_phase(
    phase: FaultPhase, rank: int, plan: FaultPlan | None, phase_sequence: list[str]
) -> None:
    phase_sequence.append(phase.name)
    _maybe_apply_fault(phase, rank, plan)


def _do_collective(rank: int, world_size: int, plan: FaultPlan | None) -> float | None:
    """Run the one synthetic collective, honoring collective-shape faults.

    Returns the collective's observed scalar result, or ``None`` if this
    rank's fault skipped the call entirely.
    """

    targets_this_rank = plan is not None and plan.target_rank == rank

    if targets_this_rank and plan.kind in COLLECTIVE_FAULT_KINDS:
        # plan.phase, never a hardcoded constant: validate_fault_plan (called
        # by the harness before any worker launches) has already rejected
        # any of these three kinds paired with a phase other than
        # BEFORE_COLLECTIVE, so this reports the phase that actually
        # matched rather than one a future collective kind might not.
        _report_fault_applied(rank, plan.phase.name, plan)

    if targets_this_rank and plan.kind == FaultKind.SKIP_COLLECTIVE:
        return None

    if targets_this_rank and plan.kind == FaultKind.MISMATCH_COLLECTIVE:
        tensor = torch.ones(1, dtype=torch.float64)
        gather_list = [torch.zeros(1, dtype=torch.float64) for _ in range(world_size)] if rank == 0 else None
        dist.gather(tensor, gather_list, dst=0)
        return None

    if targets_this_rank and plan.kind == FaultKind.MISMATCH_SHAPE:
        tensor = torch.ones(2, dtype=torch.float64)
    else:
        tensor = torch.ones(1, dtype=torch.float64)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0].item())


def run_worker(args: argparse.Namespace) -> int:
    plan = read_fault_plan(Path(args.fault_plan_path)) if args.fault_plan_path else None
    rank, world_size = args.rank, args.world_size

    if args.spawn_decoy_grandchild:
        # Deliberately NOT start_new_session: this grandchild inherits the
        # worker's own process group, which is exactly what a bare-PID reap
        # check (as opposed to a process-group reap check) would miss.
        grandchild = subprocess.Popen(["sleep", "30"])
        Path(args.grandchild_pid_path).write_text(str(grandchild.pid))

    # Pre-init sentinel: FaultPlan.phase is None only for this case, applied
    # before any process group exists so the inner process-group timeout
    # cannot bound it -- only the outer harness watchdog can.
    if (
        plan is not None
        and plan.target_rank == rank
        and plan.kind == FaultKind.STALL_BEFORE_COLLECTIVE
        and plan.phase is None
    ):
        _report_fault_applied(rank, "PRE_INIT", plan)
        time.sleep(plan.delay_seconds)

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{args.rendezvous_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.pg_timeout),
    )
    phase_sequence: list[str] = []
    _enter_phase(FaultPhase.BEFORE_COLLECTIVE, rank, plan, phase_sequence)
    collective_result = _do_collective(rank, world_size, plan)
    _enter_phase(FaultPhase.AFTER_COLLECTIVE, rank, plan, phase_sequence)

    _enter_phase(FaultPhase.BEFORE_OPTIMIZER_STEP, rank, plan, phase_sequence)
    param = torch.tensor([1.0], dtype=torch.float64)
    param -= 0.01  # fake optimizer-style in-place update; not a real optimizer
    # Marker distinguishing a fault that fired at BEFORE_OPTIMIZER_STEP (never
    # written) from one at AFTER_OPTIMIZER_STEP (written first, since the
    # update above already ran) -- the two hook points are otherwise
    # observationally identical from outside the process if a fault crashes
    # the rank at either one.
    Path(f"{args.state_path}.optimizer_done").write_text("1")
    _enter_phase(FaultPhase.AFTER_OPTIMIZER_STEP, rank, plan, phase_sequence)

    _enter_phase(FaultPhase.BEFORE_STATE_WRITE, rank, plan, phase_sequence)
    Path(args.state_path).write_text(json.dumps({"rank": rank, "note": "synthetic rank-local state"}))
    _enter_phase(FaultPhase.AFTER_STATE_WRITE, rank, plan, phase_sequence)

    _enter_phase(FaultPhase.BEFORE_PUBLICATION, rank, plan, phase_sequence)
    dist.barrier()
    _enter_phase(FaultPhase.AFTER_PUBLICATION, rank, plan, phase_sequence)

    if rank == 0:
        Path(args.complete_marker_path).write_text("COMPLETE")

    receipt = {
        "rank": rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "phase_sequence": phase_sequence,
        "collective_result": collective_result,
        "fault_kind": plan.kind.name if plan is not None else FaultKind.NONE.name,
    }
    # Atomic write: a tmp file in the same directory, then os.replace. A kill
    # landing mid-write can only ever leave the tmp path truncated -- never
    # the receipt path itself -- since os.replace is a single rename syscall
    # on the filesystems this harness runs on. Defence in depth alongside the
    # collector's own tolerance for a malformed receipt: this file cannot
    # assume the collector will always be the one reading it.
    receipt_path = Path(args.receipt_path)
    tmp_receipt_path = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    tmp_receipt_path.write_text(json.dumps(receipt))
    os.replace(tmp_receipt_path, receipt_path)
    # Deliberately NOT wrapped in a blanket try/finally: on any failure path
    # above (raise, or a collective that timed out) the process is exiting
    # anyway, and a poisoned communicator must never be reused, so there is
    # nothing left to clean up that is worth the risk of destroy_process_group
    # itself blocking on a group that a peer has already abandoned.
    dist.destroy_process_group()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_worker(args)


if __name__ == "__main__":
    sys.exit(main())
