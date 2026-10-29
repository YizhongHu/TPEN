"""Frozen configuration and attempt-free planning for the one-electron study.

This module does not launch jobs or import the physics package. All executable
objects are declared through Hydra targets and run through the normal entrypoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from experiments.toolkit.dispatch import LogicalTaskSpec, StagePlanV2
from experiments.toolkit.resources import ResourceSpec
from experiments.toolkit.specs import CompletionSpec

STUDY = "one_electron_variance"
CHECKPOINTS = (2500, 5000, 10000, 12500, 15000, 25000, 40000, 45000, 50000)
WINDOWS = {"early": (2500, 5000), "middle": (10000, 12500, 15000),
           "late": (40000, 45000, 50000)}
EXACT_ENERGY = {"hooke": 0.75, "hydrogen": -0.5}
DESIGN_SHA256 = "9c7c3bf998e06502ea40814fbfdb3d1ed641e0002e581a609e797e2eaa5ddc51"
HE_ANCHOR = {"study": "He-v1", "lineage": "train-seed0000",
             "run": "tpen-f4-eval42-78152c40-20260826T194602--0400",
             "collection": "f4-collect-20260828T0100--0400",
             "estimator": "whole_trajectory_raw_variance", "historical_ratio": "approximately 80"}


def evaluation_seed(system: str, lineage: int, checkpoint: int, stream: int) -> int:
    """Return the preregistered fresh-stream seed, never a training seed."""
    _identity(system, lineage)
    if checkpoint not in CHECKPOINTS or stream not in (0, 1):
        raise ValueError("unregistered checkpoint or stream")
    return 1_000_000 + (100_000 if system == "hydrogen" else 0) + 1000 * lineage + 10 * CHECKPOINTS.index(checkpoint) + stream


def _identity(system: str, lineage: int) -> None:
    if system not in EXACT_ENERGY or type(lineage) is not int or lineage not in range(3):
        raise ValueError("expected hooke or hydrogen and lineage 0, 1, or 2")


def train_config(system: str, lineage: int, root: Path, *, device: str = "cuda") -> dict:
    """Construct one fresh 50,000-update lineage without changing generic defaults."""
    _identity(system, lineage)
    cfg = yaml.safe_load(Path(__file__).with_name("base.yaml").read_text())
    identity = f"{system}-lineage{lineage}"
    cfg["experiment"].update(sector=system, run_name=identity)
    cfg["run"].update(root=str(root), run_id=identity)
    cfg["runtime"].update(seed=73001 + lineage, device=device)
    cfg["sampler"]["seed"] = 83001 + lineage
    cfg["system"].update(id=system, reference_energy=EXACT_ENERGY[system])
    if system == "hooke":
        cfg["model"]["factors"] = [{"_target_": "tpen.nn.GaussianConfinement",
                                     "coefficient": 0.25, "trainable": True}]
        cfg["hamiltonian_terms"]["harmonic_trap"] = {
            "_target_": "tpen.physics.potential.HarmonicTrap", "omega": 0.5}
    else:
        cfg["atoms"] = {
            "_target_": "tpen.data.atomic_configuration.AtomicConfiguration",
            "positions": {"_target_": "torch.tensor", "data": [[0., 0., 0.]]},
            "charges": {"_target_": "torch.tensor", "data": [1.]},
        }
        cfg["model"]["factors"] = [{
            "_target_": "tpen.nn.ElectronNucleusCusp", "atoms": "${atoms}",
            "law": {"_target_": "tpen.nn.CurvatureElectronNucleusCuspLaw",
                    "curvature_coefficient": 0.01, "curvature_range": 1., "trainable": True},
        }]
        cfg["hamiltonian_terms"]["electron_nucleus"] = {
            "_target_": "tpen.physics.potential.ElectronNucleusPotential",
            "atoms": "${atoms}", "eps": 0.,
        }
        cfg["sampler"]["nuclear_positions"] = copy.deepcopy(cfg["atoms"]["positions"])
        cfg["sampler"]["nuclear_charges"] = copy.deepcopy(cfg["atoms"]["charges"])
    return cfg


def run_directory(cfg: dict) -> Path:
    """Return the normal entrypoint's nested output path."""
    return Path(cfg["run"]["root"]) / cfg["experiment"]["name"] / cfg["experiment"]["sector"] / cfg["run"]["run_id"]


def evaluation_config(system: str, lineage: int, checkpoint: int, stream: int,
                      root: Path, *, device: str = "cuda") -> dict:
    """Build a strict model restore with complete learned-target trajectory records."""
    cfg = train_config(system, lineage, root, device=device)
    checkpoint_path = run_directory(cfg) / "checkpoints" / f"step_{checkpoint:06d}"
    seed = evaluation_seed(system, lineage, checkpoint, stream)
    cfg["run"]["run_id"] += f"-update{checkpoint:06d}-stream{stream}"
    cfg["experiment"]["run_name"] = cfg["run"]["run_id"]
    cfg["runtime"]["seed"] = seed
    cfg["sampler"].update(seed=seed, burn_in=1000, n_steps=20)
    cfg["callbacks"] = cfg["callbacks"][:4] + [
        callback for callback in cfg["callbacks"]
        if callback["_target_"] == "tpen.callback.ResourceUsage"
    ]
    del cfg["optimizer"], cfg["trainer"]
    cfg["load"] = {"path": str(checkpoint_path), "mode": "model_only", "strict": True,
                   "allow_protocol_mismatch": False}
    cfg["runner"] = {"_target_": "tpen.runner.Evaluate", "model": "${model}",
                     "load": "${load}", "evaluator": "${evaluator}"}
    cfg["evaluator"] = {
        "_target_": "tpen.evaluation.Evaluator", "namespace": "eval", "seed": seed,
        "artifact_level": "records", "task_failure_policy": "fail_fast",
        "tasks": [{
            "name": "trajectory", "namespace": "trajectory", "output_dir": "trajectory",
            "generator": {
                "_target_": "tpen.evaluation.generators.readout_trajectory.ReadoutTrajectoryMCMCGenerator",
                "sampler": "${sampler}", "hamiltonian_terms": "${hamiltonian_terms}",
                "n_draws": 64, "discard_draws": 0, "seed": seed, "reset": True,
                "verify_model_unchanged": True, "readout_eps": 1e-12, "replay_chunk_size": 4096,
                "allow_nucleus_free_records": system == "hooke",
            },
            "calculators": [
                {"_target_": "tpen.evaluation.LocalEnergyCalculator",
                 "hamiltonian_terms": "${hamiltonian_terms}", "return_terms": True},
                {"_target_": "tpen.evaluation.WavefunctionCalculator"},
            ],
            "summaries": [
                {"_target_": "tpen.evaluation.SampledRecordWriter", "max_samples": 262144,
                 "include_term_energies": True},
                {"_target_": "tpen.evaluation.SamplerStatsSummary"},
            ],
        }],
    }
    return cfg


def prepare(root: Path, source_sha: str) -> tuple[StagePlanV2, StagePlanV2]:
    """Write configurations and logical plans; perform no admission or execution."""
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("source_sha must be a full Git commit identity")
    root = root.resolve()
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=False)
    tasks: dict[str, list] = {"train": [], "evaluate": []}
    manifest = {"design_sha256": DESIGN_SHA256, "source_sha": source_sha,
                "historical_anchor": HE_ANCHOR, "configs": [],
                "admission": "not_admitted", "retry_policy": "no_automatic_retry"}
    for system in EXACT_ENERGY:
        for lineage in range(3):
            train_id = f"train:{system}:{lineage}"
            rows = [("train", train_config(system, lineage, root), train_id, ())]
            rows.extend(("evaluate", evaluation_config(system, lineage, step, stream, root),
                         f"evaluate:{system}:{lineage}:{step}:{stream}", (train_id,))
                        for step in CHECKPOINTS for stream in (0, 1))
            for stage, cfg, logical_id, dependencies in rows:
                path = configs / f"{cfg['run']['run_id']}.yaml"
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                output = run_directory(cfg)
                tasks[stage].append(LogicalTaskSpec(
                    logical_task_id=logical_id, stage=stage, run_id=cfg["run"]["run_id"],
                    command=("{python}", "run.py", "--config", str(path)), result_dir=str(output),
                    inputs=(str(path),), outputs=(str(output),), dependencies=dependencies,
                    params={"system": system, "lineage": lineage},
                    resources=ResourceSpec(profile="cuda", device="cuda", gpus=1, threads=4, mem_gb=32),
                    completion=CompletionSpec(policy="status_completed", status_path=str(output / "status.json")),
                    metadata={"source_sha": source_sha, "science_receipt_required": True},
                ))
                manifest["configs"].append({"logical_id": logical_id, "path": str(path),
                                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plans = tuple(StagePlanV2(study=STUDY, stage=stage, plan_id=DESIGN_SHA256,
                             results_root=str(root), tasks=tuple(rows)).validate()
                  for stage, rows in tasks.items())
    for plan in plans:
        plan.write(root / f"{plan.stage}-plan")
    (root / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return plans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    prepare(args.root, args.source_sha)
