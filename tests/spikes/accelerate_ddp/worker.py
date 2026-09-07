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

from tests.spikes.accelerate_ddp.checkpoint import (
    CheckpointPayloadStore,
    CheckpointTopologyMismatch,
)
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
SCENARIO_RESUME = "resume"
SCENARIO_PUBLICATION = "publication"
SCENARIO_COMM_COUNT = "comm-count"
SCENARIOS = (
    SCENARIO_SCORE_STEP,
    SCENARIO_ALL_INVALID,
    SCENARIO_RESUME,
    SCENARIO_PUBLICATION,
    SCENARIO_COMM_COUNT,
)


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



def _rank_local_sampler_state(rank: int, generator: torch.Generator) -> dict:
    """Rank-local walker and sampler-stream state TPEN owns outright.

    Deliberately rank-DISTINCT: identical state on every rank would make A-G3b's
    perturbation arm unable to tell a correct per-rank restore from a broadcast
    of rank 0's state, and the gate would pass for the wrong reason.
    """

    walkers = torch.arange(4, dtype=torch.float64) + float(rank) * 100.0
    return {
        "walkers": walkers,
        "proposal_count": 7 + rank,
        "generator_state": generator.get_state(),
    }


def _process_rng_state() -> dict:
    """Python, NumPy and Torch-CPU RNG state for this process."""

    import random

    import numpy as np

    return {
        "python": list(random.getstate()[1]),
        "numpy": [int(x) for x in np.random.get_state()[1]],
        "torch_cpu": torch.get_rng_state(),
    }


def _iterate(access: ModelAccess, runtime, optimizer, features, energy, counter, n: int) -> None:
    """Run ``n`` score updates, so continuous and split runs share one code path.

    Sharing the path matters: if the continuous arm and the resumed arm ran
    different code, a difference between them could be a difference in the
    drivers rather than in the checkpoint round trip.
    """

    for _ in range(n):
        stats = prepare_statistics(runtime, energy)
        run_score_function_step(access, runtime, optimizer, features, energy, stats, counter)


def _build(runtime, args) -> tuple:
    """Construct the identical model/optimizer/fixture triple every scenario uses."""

    kind = "m2" if args.world_size == 3 else "regular"
    features, energy = scientific_fixture(args.world_size, args.rank, kind=kind)
    raw_model = SemanticWavefunction()
    access = ModelAccess.create(raw_model, runtime.accelerator)
    counter = install_gradient_counter(access, runtime)
    optimizer = torch.optim.SGD(access.raw_model.parameters(), lr=0.05, momentum=0.9)
    # PREPARE THE OPTIMIZER, not just the model. `accelerator.save_state` persists
    # only the models and optimizers registered through `prepare`; an unprepared
    # optimizer is silently omitted, so its momentum would not survive the round
    # trip and A-G3 would compare a resumed run that had quietly lost state. That
    # is an ownership fact about Accelerate worth recording rather than a detail:
    # to get optimizer bytes transported, TPEN must hand the optimizer over.
    optimizer = runtime.accelerator.prepare(optimizer)
    return features, energy, access, counter, optimizer


def _optimizer_state(optimizer, model) -> dict:
    """Read momentum buffers through whatever wrapper `prepare` installed.

    Accelerate returns an `AcceleratedOptimizer`, so the state lives on the inner
    optimizer. Unwrapping explicitly beats relying on attribute proxying, which
    would fail silently by reporting an empty state dict rather than raising.
    """

    inner = getattr(optimizer, "optimizer", optimizer)
    return {
        name: _tensor_to_list(inner.state[p]["momentum_buffer"])
        for name, p in model.named_parameters()
        if p in inner.state and "momentum_buffer" in inner.state[p]
    }


def _run_resume(runtime, args: argparse.Namespace) -> dict:
    """A-G3/A-G3b: K+L continuous must equal K, checkpoint, teardown, restore, L.

    One invocation runs ONE phase; the driver runs the phases as separate harness
    invocations so that ALL processes are genuinely torn down between them. A
    resume test that never tore the processes down would be testing an in-process
    reload, which is a much weaker claim than the contract makes.
    """

    features, energy, access, counter, optimizer = _build(runtime, args)
    generator = torch.Generator()
    generator.manual_seed(1234 + args.rank)
    store = CheckpointPayloadStore(root=Path(args.checkpoint_root), runtime=runtime)
    observed: dict = {"phase": args.phase}

    if args.phase == "continuous":
        _iterate(access, runtime, optimizer, features, energy, counter, args.k + args.l)

    elif args.phase == "first-half":
        _iterate(access, runtime, optimizer, features, energy, counter, args.k)
        store.save(
            access.raw_model,
            optimizer,
            generation=1,
            sampler_state=_rank_local_sampler_state(args.rank, generator),
            rng_state=_process_rng_state(),
            completed_updates=args.k,
        )
        observed["saved_generation"] = 1

    elif args.phase == "second-half":
        restored = store.load(access.raw_model, optimizer, generation=1)
        observed["completed_updates_restored"] = restored["completed_updates"]
        observed["canonical_model_keys"] = restored["canonical_model_keys"]
        # No `module.` prefix in the canonical keys is asserted by the gate; it is
        # recorded here so the assertion reads recorded evidence.
        observed["sampler_walkers"] = restored["sampler_state"]["walkers"].tolist()
        observed["sampler_proposal_count"] = restored["sampler_state"]["proposal_count"]
        _iterate(access, runtime, optimizer, features, energy, counter, args.l)

    elif args.phase == "topology-refusal":
        # Refusal must precede ANY mutation, so the parameters are captured before
        # and after and required to be identical.
        before = {n: _tensor_to_list(p) for n, p in access.raw_model.named_parameters()}
        try:
            store.load(access.raw_model, optimizer, generation=1)
            observed["refused"] = False
        except CheckpointTopologyMismatch as exc:
            observed["refused"] = True
            observed["refusal_message"] = str(exc)
        after = {n: _tensor_to_list(p) for n, p in access.raw_model.named_parameters()}
        observed["parameters_unchanged"] = before == after
        return observed

    else:
        raise ValueError(f"unknown resume phase {args.phase!r}")

    observed["parameters"] = {
        name: _tensor_to_list(p) for name, p in access.raw_model.named_parameters()
    }
    observed["optimizer_momentum"] = _optimizer_state(optimizer, access.raw_model)
    observed["gradient_reductions"] = counter.count
    return observed


def _run_publication(runtime, args: argparse.Namespace) -> dict:
    """A-G5: a failed or delayed shard writer must leave NO selectable generation."""

    features, energy, access, counter, optimizer = _build(runtime, args)
    generator = torch.Generator()
    generator.manual_seed(4321 + args.rank)
    store = CheckpointPayloadStore(root=Path(args.checkpoint_root), runtime=runtime)

    _iterate(access, runtime, optimizer, features, energy, counter, 1)
    store.save(
        access.raw_model,
        optimizer,
        generation=1,
        sampler_state=_rank_local_sampler_state(args.rank, generator),
        rng_state=_process_rng_state(),
        completed_updates=1,
        failure_rank=args.failure_rank,
        delay_rank=args.delay_rank,
        delay_seconds=args.delay_seconds,
    )
    return {"selectable_generations": store.selectable_generations()}


def _run_comm_count(runtime, args: argparse.Namespace) -> dict:
    """A-G6: gradient reductions must not grow with sampling or kinetic work.

    MEASURED at two different proposal counts rather than asserted by inspection.
    The counter is also required to be NONZERO: a counter reading zero under both
    workloads would satisfy "does not grow" vacuously, which is the shape this
    gate exists to rule out.
    """

    features, energy, access, counter, optimizer = _build(runtime, args)
    stats = prepare_statistics(runtime, energy)

    # Sampling and kinetic-derivative work on the RAW module. If any of this
    # enlisted the prepared wrapper, the reduction count would scale with it.
    for _ in range(args.mcmc_steps):
        access.coordinate_forward(features)

    reductions_after_sampling = counter.count
    run_score_function_step(access, runtime, optimizer, features, energy, stats, counter)

    return {
        "mcmc_steps": args.mcmc_steps,
        "reductions_after_sampling_only": reductions_after_sampling,
        "reductions_after_one_update": counter.count,
        "raw_forwards": access.forward_counts["raw"],
        "prepared_forwards": access.forward_counts["prepared"],
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
        elif args.scenario == SCENARIO_RESUME:
            state["result"] = _run_resume(runtime, args)
        elif args.scenario == SCENARIO_PUBLICATION:
            state["result"] = _run_publication(runtime, args)
        elif args.scenario == SCENARIO_COMM_COUNT:
            state["result"] = _run_comm_count(runtime, args)
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
    parser.add_argument("--checkpoint-root", type=str, default=None)
    parser.add_argument("--phase", type=str, default="continuous")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--l", type=int, default=3)
    parser.add_argument("--failure-rank", type=int, default=None)
    parser.add_argument("--delay-rank", type=int, default=None)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--mcmc-steps", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_worker(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
