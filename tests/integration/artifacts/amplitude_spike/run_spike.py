"""Stage-2 bounded-spike driver for TO slice fa3c35a8 (amplitude-risk calibration).

Not a core entrypoint and not a checked-in study config. Run with the project
venv's python directly (never ``python -m tpen.run``, a documented no-op --
note 00-read-first-cannon / the TRAP in the spike dispatch message), from the
repository root, e.g.::

    .venv/bin/python tests/integration/artifacts/amplitude_spike/run_spike.py \\
        exact-hydrogen --alpha 1.0 --stream 0 --seed 920000 --out-root /path/to/out

See note amplitude-risk-design-20261106 on TO item 8a67671f for the full
specification (seeds, grid sizes, scope). Every subcommand prints a single
JSON receipt line to stdout on success (also written to ``<out-root>/receipts``).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
H_V2_DIR = REPO_ROOT / "experiments" / "atomistic" / "h-v2"

sys.path.insert(0, str(HERE))
import analysis  # noqa: E402
from hooke_exact_harness import run_hooke_exact_trajectory, save_hooke_raw  # noqa: E402
from training_capture import TrainingBatchLogabsCapture  # noqa: E402

from tpen.run import run_from_config  # noqa: E402


def _load_protocol():
    spec = importlib.util.spec_from_file_location("h_v2_protocol", H_V2_DIR / "protocol.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_config(config: dict, *, command: str, runtime_callbacks: tuple = ()) -> int:
    return run_from_config(
        OmegaConf.create(config), command=command, raise_exceptions=True, runtime_callbacks=runtime_callbacks
    )


def _read_sampled_eval_table(eval_dir: Path) -> dict[str, np.ndarray]:
    path = eval_dir / "sampled_eval_table.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty sampled_eval_table.csv at {path}")
    draw_index = np.array([int(r["draw_index"]) for r in rows], dtype=np.int64)
    walker_index = np.array([int(r["walker_index"]) for r in rows], dtype=np.int64)
    logabs = np.array([float(r["logabs"]) for r in rows], dtype=np.float64)
    sign = np.array([float(r["sign"]) for r in rows], dtype=np.float64)
    local_energy = np.array([float(r["local_energy"]) for r in rows], dtype=np.float64)
    finite = np.array([r["finite"] == "True" for r in rows], dtype=bool)
    return {
        "draw_index": draw_index,
        "walker_index": walker_index,
        "logabs": logabs,
        "sign": sign,
        "local_energy": local_energy,
        "finite": finite,
    }


def _reduce_table_by_draw(table: dict[str, np.ndarray]) -> dict[str, Any]:
    """Group a flat [draw,walker] raw table by draw and run the full analysis stack."""

    n_draws = int(table["draw_index"].max()) + 1
    draws: list[analysis.DrawReduction] = []
    walker_lists: list[np.ndarray] = []
    ell_lists: list[np.ndarray] = []
    groups_by_draw: list[list[dict]] = []
    for d in range(n_draws):
        mask = table["draw_index"] == d
        ell = table["logabs"][mask]
        sign = table["sign"][mask]
        walkers = table["walker_index"][mask]
        draws.append(analysis.reduce_draw(ell, sign))
        walker_lists.append(walkers)
        ell_lists.append(ell)
        groups = analysis.group_within_draw(ell, sign, walkers)
        groups_by_draw.append(
            [
                {"total_count": g.total_count, "median": g.median, "fractions": g.fractions}
                for g in groups
            ]
        )
    agg = analysis.aggregate_stream(draws, walker_lists, ell_lists)
    boot = analysis.block_bootstrap_fraction(draws, walker_lists, ell_lists)
    return {
        "n_draws": n_draws,
        "aggregate": agg,
        "bootstrap": {str(tau): v for tau, v in boot.items()},
        "per_draw_quantiles": [
            {"median": d.median, "quantiles": d.quantiles, "min_centered": d.min_centered}
            for d in draws
        ],
        "batch_group_effect": groups_by_draw,
    }


def _write_receipt(out_root: Path, name: str, payload: dict) -> None:
    receipts = out_root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    path = receipts / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"receipt": str(path), **payload}, default=str))


def cmd_exact_hydrogen(args: argparse.Namespace) -> None:
    protocol = _load_protocol()
    out_root = Path(args.out_root)
    config = protocol.load_yaml(H_V2_DIR / "configs" / "exact_canary_eval.yaml")
    config["model"]["alpha"] = args.alpha
    config["runtime"]["seed"] = args.seed
    config["run"]["root"] = str(out_root)
    run_id = f"exact-h-alpha{args.alpha}-stream{args.stream}"
    config["run"]["run_id"] = run_id
    sampler = config["sampler"]
    sampler.update(n_walkers=4096, burn_in=1000, n_steps=20, proposal_scale=0.5, seed=args.seed)
    config["evaluator"]["seed"] = args.seed
    config["evaluator"]["tasks"][0]["generator"]["n_draws"] = 64
    config["evaluator"]["tasks"][0]["generator"]["seed"] = args.seed
    config["evaluator"]["tasks"][0]["summaries"][0]["max_samples"] = 4096 * 64
    # Build a genuine model-only checkpoint exactly as
    # test_hydrogen_exact_canary.py::_build_model_only_checkpoint does, since
    # the real Evaluate runner requires a reconciled restored-checkpoint
    # identity even for this parameterless exact model.
    from datetime import UTC

    from hydra.utils import instantiate

    from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
    from tpen.checkpoint.save import save_checkpoint

    checkpoint_source = out_root / "checkpoint-source" / run_id
    cfg = OmegaConf.create(config)
    model = instantiate(cfg.model)
    artifact_manager = ArtifactManager(
        checkpoint_source,
        experiment="h_v2_amplitude_spike",
        sector="atomistic",
        run_id="checkpoint-source",
        layout="flat",
    )
    artifact_manager.make_dirs()
    metadata = RunMetadata(
        run_id="checkpoint-source",
        run_name="checkpoint-source",
        timestamp="2026-11-06T00:00:00+00:00",
        timezone="UTC",
        git_commit="checkpoint-source",
        git_branch="agent/h-v2-amplitude-spike",
        dirty_worktree=False,
        command="amplitude-spike",
        config_path=str(H_V2_DIR / "configs" / "exact_canary_eval.yaml"),
        resolved_config_path=str(artifact_manager.path("resolved_config.yaml")),
        run_dir=str(artifact_manager.run_dir),
        device="cpu",
        dtype="float64",
    )
    checkpoint_context = RunContext(
        cfg=cfg,
        source_cfg=cfg,
        artifact_manager=artifact_manager,
        metadata=metadata,
        clock=RunClock(timezone="UTC", tzinfo=UTC),
    )
    checkpoint_dir = save_checkpoint(
        output_dir=checkpoint_source / "checkpoints",
        next_iteration=0,
        completed_updates=0,
        model=model,
        context=checkpoint_context,
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )
    config["load"]["path"] = str(checkpoint_dir)

    t0 = time.monotonic()
    rc = _run_config(config, command=f"amplitude-spike exact-hydrogen alpha={args.alpha} stream={args.stream}")
    elapsed = time.monotonic() - t0
    if rc != 0:
        raise RuntimeError(f"exact-hydrogen run failed with rc={rc}")
    eval_dir = out_root / config["experiment"]["name"] / config["experiment"]["sector"] / run_id / "trajectory"
    table = _read_sampled_eval_table(eval_dir)
    reduction = _reduce_table_by_draw(table)
    _write_receipt(
        out_root,
        f"exact-hydrogen-alpha{args.alpha}-stream{args.stream}",
        {
            "arm": "exact_hydrogen",
            "alpha": args.alpha,
            "stream": args.stream,
            "seed": args.seed,
            "elapsed_sec": elapsed,
            "row_count": int(table["logabs"].size),
            "eval_dir": str(eval_dir),
            **reduction,
        },
    )


def cmd_exact_hooke(args: argparse.Namespace) -> None:
    import torch

    from tpen.physics.hooke import HookeSingletExact, HookeTripletExact
    from tpen.physics.kinetic import KineticEnergy
    from tpen.physics.potential import ElectronElectronInteraction, HarmonicTrap
    from tpen.sampling.metropolis import MetropolisSampler

    out_root = Path(args.out_root)
    if args.state == "singlet":
        model = HookeSingletExact()
        omega = 0.5
        initial_scale = 1.0
    elif args.state == "triplet":
        model = HookeTripletExact()
        omega = 0.25
        initial_scale = 2.0
    else:
        raise ValueError(f"unknown Hooke state {args.state!r}")

    hamiltonian_terms = {
        "kinetic": KineticEnergy(),
        "harmonic_trap": HarmonicTrap(omega=omega),
        "electron_electron": ElectronElectronInteraction(),
    }
    sampler = MetropolisSampler(
        n_walkers=4096,
        burn_in=1000,
        n_steps=20,
        proposal_scale=0.5,
        seed=args.seed,
        n_electrons=2,
        spatial_dim=3,
        initial_scale=initial_scale,
        dtype=torch.float64,
    )

    t0 = time.monotonic()
    arrays = run_hooke_exact_trajectory(
        model=model,
        sampler=sampler,
        hamiltonian_terms=hamiltonian_terms,
        n_draws=64,
        observable_name="local_energy",
        seed=args.seed,
    )
    elapsed = time.monotonic() - t0
    raw_path = out_root / "hooke" / f"exact-{args.state}-stream{args.stream}.npz"
    save_hooke_raw(raw_path, arrays)

    table = {
        "draw_index": arrays["draw_index"],
        "walker_index": arrays["walker_index"],
        "logabs": arrays["logabs"].reshape(-1),
        "sign": arrays["sign"].reshape(-1),
    }
    reduction = _reduce_table_by_draw(table)
    _write_receipt(
        out_root,
        f"exact-hooke-{args.state}-stream{args.stream}",
        {
            "arm": "exact_hooke",
            "state": args.state,
            "stream": args.stream,
            "seed": args.seed,
            "elapsed_sec": elapsed,
            "row_count": int(table["logabs"].size),
            "raw_path": str(raw_path),
            **reduction,
        },
    )


def cmd_learned_train(args: argparse.Namespace) -> None:
    protocol = _load_protocol()
    out_root = Path(args.out_root)
    train = protocol.training_config()
    if args.lineage == 1:
        train["runtime"]["seed"] = 910101
        train["sampler"]["seed"] = 910102
    train["run"]["root"] = str(out_root)

    # Initial (update-zero) checkpoint, same pattern as test_pipeline.py.
    initial = protocol.training_config() if args.lineage == 0 else dict(train)
    import copy

    initial = copy.deepcopy(train)
    initial["run"]["run_id"] = f"train-lineage{args.lineage}-initial"
    initial["trainer"]["max_steps"] = 0
    t0 = time.monotonic()
    rc = _run_config(initial, command=f"amplitude-spike learned-train lineage{args.lineage} initial")
    if rc != 0:
        raise RuntimeError(f"lineage {args.lineage} initial-checkpoint run failed rc={rc}")
    initial_checkpoint = out_root / "tpen_h_v2" / "atomistic" / initial["run"]["run_id"] / "checkpoints" / "step_000000"
    assert (initial_checkpoint / "COMPLETE").is_file()

    # Full 1000-update run, checkpointing every 100 updates (captures 100 and 1000),
    # with the spike-only batch-capture callback attached via runtime_callbacks
    # (not part of the checked-in science configuration).
    full = copy.deepcopy(train)
    full["run"]["run_id"] = f"train-lineage{args.lineage}-full"
    for callback in full["callbacks"]:
        if callback.get("_target_") == "tpen.callback.Checkpoint":
            callback["schedule"]["every_n"] = 100
    capture_path = out_root / "training_batches" / f"lineage{args.lineage}.npz"
    capture = TrainingBatchLogabsCapture(output_path=capture_path, every_n_steps=1)
    rc = _run_config(
        full,
        command=f"amplitude-spike learned-train lineage{args.lineage} full",
        runtime_callbacks=(capture,),
    )
    elapsed = time.monotonic() - t0
    if rc != 0:
        raise RuntimeError(f"lineage {args.lineage} full-training run failed rc={rc}")
    checkpoints_dir = out_root / "tpen_h_v2" / "atomistic" / full["run"]["run_id"] / "checkpoints"
    for step in (100, 1000):
        assert (checkpoints_dir / f"step_{step:06d}" / "COMPLETE").is_file(), step

    npz = np.load(capture_path)
    batch_reductions = []
    for i, step in enumerate(npz["step"]):
        d = analysis.reduce_draw(npz["logabs"][i], npz["sign"][i])
        batch_reductions.append(
            {
                "step": int(step),
                "optimizer_step": bool(npz["optimizer_step"][i]),
                "median": d.median,
                "fractions": d.fractions,
                "total_count": d.total_count,
            }
        )

    _write_receipt(
        out_root,
        f"learned-train-lineage{args.lineage}",
        {
            "arm": "learned_training_batches",
            "lineage": args.lineage,
            "elapsed_sec": elapsed,
            "initial_checkpoint": str(initial_checkpoint),
            "checkpoints_dir": str(checkpoints_dir),
            "capture_path": str(capture_path),
            "n_captured_steps": len(batch_reductions),
            "batch_reductions": batch_reductions,
        },
    )


def cmd_learned_eval(args: argparse.Namespace) -> None:
    protocol = _load_protocol()
    out_root = Path(args.out_root)
    if args.checkpoint == 0:
        checkpoint_dir = (
            out_root / "tpen_h_v2" / "atomistic" / f"train-lineage{args.lineage}-initial" / "checkpoints" / "step_000000"
        )
    else:
        checkpoint_dir = (
            out_root
            / "tpen_h_v2"
            / "atomistic"
            / f"train-lineage{args.lineage}-full"
            / "checkpoints"
            / f"step_{args.checkpoint:06d}"
        )
    assert (checkpoint_dir / "COMPLETE").is_file(), checkpoint_dir

    evaluation = protocol.load_yaml(H_V2_DIR / "configs" / "eval.yaml")
    evaluation["load"]["path"] = str(checkpoint_dir)
    evaluation["runtime"]["seed"] = args.seed
    evaluation["run"]["root"] = str(out_root)
    run_id = f"eval-lineage{args.lineage}-checkpoint{args.checkpoint}-stream{args.stream}"
    evaluation["run"]["run_id"] = run_id
    sampler = evaluation["sampler"]
    sampler.update(n_walkers=4096, burn_in=1000, n_steps=20, proposal_scale=0.5, seed=args.seed)
    evaluation["evaluator"]["seed"] = args.seed
    evaluation["evaluator"]["tasks"][0]["generator"]["n_draws"] = 64
    evaluation["evaluator"]["tasks"][0]["generator"]["seed"] = args.seed
    evaluation["evaluator"]["tasks"][0]["summaries"][0]["max_samples"] = 4096 * 64

    t0 = time.monotonic()
    rc = _run_config(
        evaluation,
        command=f"amplitude-spike learned-eval lineage{args.lineage} checkpoint{args.checkpoint} stream{args.stream}",
    )
    elapsed = time.monotonic() - t0
    if rc != 0:
        raise RuntimeError(f"learned-eval run failed rc={rc}")
    eval_dir = out_root / "tpen_h_v2" / "atomistic" / run_id / "trajectory"
    table = _read_sampled_eval_table(eval_dir)
    reduction = _reduce_table_by_draw(table)
    _write_receipt(
        out_root,
        f"learned-eval-lineage{args.lineage}-checkpoint{args.checkpoint}-stream{args.stream}",
        {
            "arm": "learned_evaluation",
            "lineage": args.lineage,
            "checkpoint": args.checkpoint,
            "stream": args.stream,
            "seed": args.seed,
            "elapsed_sec": elapsed,
            "row_count": int(table["logabs"].size),
            "eval_dir": str(eval_dir),
            **reduction,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("exact-hydrogen")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--stream", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-root", type=str, required=True)
    p.set_defaults(func=cmd_exact_hydrogen)

    p = sub.add_parser("exact-hooke")
    p.add_argument("--state", choices=["singlet", "triplet"], required=True)
    p.add_argument("--stream", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-root", type=str, required=True)
    p.set_defaults(func=cmd_exact_hooke)

    p = sub.add_parser("learned-train")
    p.add_argument("--lineage", type=int, choices=[0, 1], required=True)
    p.add_argument("--out-root", type=str, required=True)
    p.set_defaults(func=cmd_learned_train)

    p = sub.add_parser("learned-eval")
    p.add_argument("--lineage", type=int, choices=[0, 1], required=True)
    p.add_argument("--checkpoint", type=int, choices=[0, 100, 1000], required=True)
    p.add_argument("--stream", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-root", type=str, required=True)
    p.set_defaults(func=cmd_learned_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
