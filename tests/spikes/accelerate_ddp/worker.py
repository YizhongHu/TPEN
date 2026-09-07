"""DF1-CLI worker running one DS-A0 scenario per invocation.

Accepts DF1's existing worker CLI unchanged and carries its richer per-rank
evidence in the already-provided ``--state-path`` artifact, so ``RankReceipt`` is
not modified. Selected by ``--scenario`` through the harness's
``worker_extra_args``.

Every scenario writes its state artifact on BOTH the success and the failure
path. A scenario that failed and left nothing behind would force a rerun to learn
what it already knew, and for the refusal scenarios the failure IS the result.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from pathlib import Path

import torch

from tests.spikes.accelerate_ddp.fixtures import scientific_fixture
from tests.spikes.accelerate_ddp.model_access import ModelAccess, SemanticWavefunction
from tests.spikes.accelerate_ddp.runtime import DistributedRuntime
from tests.spikes.accelerate_ddp.vmc_step import (
    install_gradient_counter,
    prepare_statistics,
    run_score_function_step,
)

SCENARIO_SCORE_STEP = "score-step"
SCENARIO_ALL_INVALID = "all-invalid"
SCENARIOS = (SCENARIO_SCORE_STEP, SCENARIO_ALL_INVALID)


def _tensor_to_list(tensor: torch.Tensor) -> list:
    """Serialize a float64 tensor losslessly enough for exact-ish comparison.

    ``tolist()`` on float64 round-trips through Python floats, which are also
    float64, so no precision is lost here. JSON's decimal rendering is the only
    lossy step, and ``repr``-style shortest round-trip output is what Python's
    json module emits, so the values reload bit-identically.
    """

    return tensor.detach().to(torch.float64).flatten().tolist()


def _run_score_step(runtime: DistributedRuntime, args: argparse.Namespace) -> dict:
    """One full score-function update, with every comparison input recorded."""

    kind = "m2" if args.world_size == 3 else "regular"
    features, energy = scientific_fixture(args.world_size, args.rank, kind=kind)

    raw_model = SemanticWavefunction()
    access = ModelAccess.create(raw_model, runtime.accelerator)
    counter = install_gradient_counter(access, runtime)
    optimizer = torch.optim.SGD(access.raw_model.parameters(), lr=0.1, momentum=0.9)

    # The statistics collective runs BEFORE any backward: mu and M are global, and
    # centering on a rank-local mean is the documented wrong answer (A-G1b).
    stats = prepare_statistics(runtime, energy)
    if stats.finite_count == 0:
        raise RuntimeError("score-step scenario requires a nonzero global finite count")

    observation = run_score_function_step(
        access, runtime, optimizer, features, energy, stats, counter
    )

    return {
        "fixture_kind": kind,
        "features": _tensor_to_list(features),
        "features_shape": list(features.shape),
        "energy": _tensor_to_list(energy),
        "stats": stats.as_dict(),
        "global_loss": observation.global_loss,
        "local_surrogate_loss": observation.local_surrogate_loss,
        "scale_factor": observation.scale_factor,
        "logabs": _tensor_to_list(observation.logabs),
        "coordinate_gradient": _tensor_to_list(observation.coordinate_gradient),
        "gradients": {
            name: _tensor_to_list(value) for name, value in observation.gradients.items()
        },
        "local_gradients": {
            name: _tensor_to_list(value) for name, value in observation.local_gradients.items()
        },
        "post_step_parameters": {
            name: _tensor_to_list(parameter)
            for name, parameter in access.raw_model.named_parameters()
        },
        "gradient_reductions": observation.gradient_reductions,
        "gradient_accumulation_steps": observation.gradient_accumulation_steps,
        "forward_counts": dict(access.forward_counts),
        "gradient_counts": dict(access.gradient_counts),
        "model_provenance": dict(access.provenance),
    }


def _run_all_invalid(runtime: DistributedRuntime, args: argparse.Namespace) -> dict:
    """Global M == 0: one common pre-backward refusal on every rank.

    Following DS-N's R2-F2 repair, the assertion consumes PARAMETER-GRADIENT HOOK
    ACTIVITY from the raw model and requires zero, rather than merely observing
    that state looks unchanged. Unchanged-looking state is compatible with a
    backward that happened and cancelled; a zero gradient-event count is not.
    """

    features, energy = scientific_fixture(args.world_size, args.rank, kind="all_invalid")

    raw_model = SemanticWavefunction()
    access = ModelAccess.create(raw_model, runtime.accelerator)
    counter = install_gradient_counter(access, runtime)
    optimizer = torch.optim.SGD(access.raw_model.parameters(), lr=0.1, momentum=0.9)

    parameters_before = {
        name: _tensor_to_list(parameter)
        for name, parameter in access.raw_model.named_parameters()
    }

    stats = prepare_statistics(runtime, energy)
    refused = stats.finite_count == 0

    parameters_after = {
        name: _tensor_to_list(parameter)
        for name, parameter in access.raw_model.named_parameters()
    }

    return {
        "fixture_kind": "all_invalid",
        "stats": stats.as_dict(),
        "refused_before_backward": refused,
        "parameters_before": parameters_before,
        "parameters_after": parameters_after,
        "parameters_unchanged": parameters_before == parameters_after,
        # The load-bearing evidence: zero gradient events and zero reductions.
        "parameter_gradient_events": access.gradient_counts["parameter"],
        "gradient_reductions": counter.count,
        "prepared_forwards": access.forward_counts["prepared"],
        "optimizer_state_empty": len(optimizer.state) == 0,
    }


def run_worker(args: argparse.Namespace) -> int:
    """Run one scenario, write the DF1 receipt, and record what was observed."""

    rank, world_size = args.rank, args.world_size
    state: dict = {
        "rank": rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "scenario": args.scenario,
        "sys_executable": sys.executable,
        "torch_version": torch.__version__,
        "ok": False,
    }

    runtime: DistributedRuntime | None = None
    try:
        import accelerate

        state["accelerate_version"] = accelerate.__version__
        runtime = DistributedRuntime.initialize(
            rank=rank,
            world_size=world_size,
            rendezvous_file=Path(args.rendezvous_file),
            process_group_timeout_seconds=args.pg_timeout,
        )
        if args.scenario == SCENARIO_SCORE_STEP:
            state["result"] = _run_score_step(runtime, args)
        elif args.scenario == SCENARIO_ALL_INVALID:
            state["result"] = _run_all_invalid(runtime, args)
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"unknown scenario {args.scenario!r}")
        state["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the failure is part of the evidence
        state["ok"] = False
        state["failure_type"] = type(exc).__name__
        state["failure_message"] = str(exc)
        state["failure_traceback"] = traceback.format_exc()

    Path(args.state_path).write_text(json.dumps(state, indent=2, default=str))

    receipt = {
        "rank": rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "phase_sequence": [f"SCENARIO_{args.scenario.replace('-', '_').upper()}"],
        "collective_result": float(world_size) if state["ok"] else None,
        "fault_kind": "NONE",
    }
    receipt_path = Path(args.receipt_path)
    tmp_receipt_path = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    tmp_receipt_path.write_text(json.dumps(receipt))
    os.replace(tmp_receipt_path, receipt_path)

    if rank == 0 and state["ok"]:
        Path(args.complete_marker_path).write_text("COMPLETE")

    if runtime is not None:
        runtime.close()

    return 0 if state["ok"] else 1


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
    parser.add_argument("--scenario", type=str, required=True, choices=SCENARIOS)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_worker(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
