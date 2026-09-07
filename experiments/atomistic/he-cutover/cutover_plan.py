"""Build the attempt-free He-cutover train and evaluation plans."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.toolkit.dispatch import LogicalTaskSpec, StagePlanV2, logical_task_id_from_parts
from experiments.toolkit.resources import ResourceSpec
from experiments.toolkit.specs import CompletionSpec

# Siblings are loaded study-scoped, not by bare import: experiments/ has several
# same-named modules and the first study loaded would otherwise own the bare name
# for every study after it. See experiments/toolkit/study_imports.py.
#
# The loader is reached BY PATH rather than by putting the repository root on
# sys.path. A study directory that mutates sys.path is the mechanism behind the
# very defect this import exists to fix, and he-cutover's gateway test forbids it
# outright -- so the fix must not reintroduce it in order to install itself.
import importlib.util as _tpen_importlib  # noqa: E402
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

if "_tpen_study_imports" not in _tpen_sys.modules:
    _tpen_spec = _tpen_importlib.spec_from_file_location(
        "_tpen_study_imports",
        _TpenPath(__file__).resolve().parents[3] / "experiments" / "toolkit" / "study_imports.py",
    )
    _tpen_module = _tpen_importlib.module_from_spec(_tpen_spec)
    _tpen_sys.modules["_tpen_study_imports"] = _tpen_module
    _tpen_spec.loader.exec_module(_tpen_module)
sibling = _tpen_sys.modules["_tpen_study_imports"].sibling

cutover_strata = sibling(__file__, 'cutover_strata')

SMOKE_SCHEMA = "he-cutover-smoke-grid/v1"
PROOF_SCHEMA = "he-cutover-proof-grid/v1"
PRODUCTION_SCHEMA = "he-cutover-production-grid/v1"
STUDY = "he-cutover"
GRID_KEYS = {"schema", "study", "train_config", "eval_config", "train", "eval", "facilities"}
SMOKE_TRAIN_KEYS = {"seed", "max_steps", "n_walkers"}
PRODUCTION_TRAIN_KEYS = {"seeds", "max_steps", "n_walkers", "checkpoint_steps"}
SMOKE_EVAL_KEYS = {"chains", "n_walkers", "n_draws", "burn_in", "discard_draws", "stride", "chunk_size", "task_names"}
PRODUCTION_EVAL_KEYS = SMOKE_EVAL_KEYS | {"chain_seed_base"}
PRODUCTION_TASK_NAMES = (
    "mcmc_energy", "he_radial_profiles", "full_model_antisymmetry",
    "spatial_exchange_symmetry", "trace_equivariance", "he_en_numerical_atlas",
    "he_ee_ideal_vs_executed_numerical_atlas", "he_one_electron_tail_atlas",
    "he_center_of_mass_tail_atlas", "he_angular_shell_atlas",
)
FACILITY_KEYS = {"runtime", "partition", "stratum", "timeout_min", "cpus", "mem_gb", "gpus"}


class PlanError(ValueError):
    """The tracked grid is not exactly the declared smoke."""


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise PlanError(f"{label} keys mismatch: expected={sorted(keys)}, actual={sorted(value)}")


def load_grid(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate either the smoke or production grid."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PlanError("grid must be a mapping")
    _exact(payload, GRID_KEYS, "grid")
    schema = payload["schema"]
    if schema not in {SMOKE_SCHEMA, PROOF_SCHEMA, PRODUCTION_SCHEMA} or payload["study"] != STUDY:
        raise PlanError("grid schema/study identity changed")
    if payload["train_config"] != "experiments/atomistic/he-v1/configs/train.yaml" or payload["eval_config"] != "experiments/atomistic/he-v1/configs/eval.yaml":
        raise PlanError("He-v1 configs must be referenced verbatim")
    train, evaluation, facilities = payload["train"], payload["eval"], payload["facilities"]
    if not all(isinstance(item, Mapping) for item in (train, evaluation, facilities)):
        raise PlanError("train, eval, and facilities must be mappings")
    is_smoke = schema == SMOKE_SCHEMA
    is_proof = schema == PROOF_SCHEMA
    _exact(train, SMOKE_TRAIN_KEYS if (is_smoke or is_proof) else PRODUCTION_TRAIN_KEYS, "train")
    _exact(evaluation, SMOKE_EVAL_KEYS if (is_smoke or is_proof) else PRODUCTION_EVAL_KEYS, "eval")
    if is_smoke:
        if dict(train) != {"seed": 0, "max_steps": 25, "n_walkers": 16}:
            raise PlanError("train smoke must be seed 0, 25 steps, 16 walkers")
        expected_eval = {"chains": 2, "n_walkers": 16, "n_draws": 4, "burn_in": 4, "discard_draws": 0, "stride": 2, "chunk_size": 16, "task_names": ["mcmc_energy"]}
        if dict(evaluation) != expected_eval:
            raise PlanError("evaluation smoke coordinates changed")
        if set(facilities) != {"cannon", "polaris"}:
            raise PlanError("smoke facilities must be exactly cannon and polaris")
    elif is_proof:
        if dict(train) != {"seed": 0, "max_steps": 25, "n_walkers": 16}:
            raise PlanError("proof training must be seed 0, 25 steps, 16 walkers")
        expected_eval = {"chains": 40, "n_walkers": 16, "n_draws": 4, "burn_in": 4, "discard_draws": 0, "stride": 2, "chunk_size": 16, "task_names": ["mcmc_energy"]}
        if dict(evaluation) != expected_eval:
            raise PlanError("evaluation proof coordinates changed")
        if set(facilities) != {"polaris", "polaris_scaling"}:
            raise PlanError("proof facilities must be exactly polaris and polaris_scaling")
    else:
        expected_train = {"seeds": [0, 1, 2], "max_steps": 300000, "n_walkers": 4096, "checkpoint_steps": [100000, 200000, 300000]}
        if dict(train) != expected_train:
            raise PlanError("production training coordinates changed")
        if set(facilities) != {"polaris"}:
            raise PlanError("production facility must be exactly polaris")
        if evaluation["chains"] != 4 or evaluation["chain_seed_base"] != 1000:
            raise PlanError("production evaluation requires four chains from seed base 1000")
        if (evaluation["n_walkers"], evaluation["n_draws"], evaluation["burn_in"], evaluation["stride"], evaluation["chunk_size"]) != (4096, 256, 100, 20, 256):
            raise PlanError("production evaluation scale changed")
        if tuple(evaluation["task_names"]) != PRODUCTION_TASK_NAMES:
            raise PlanError("production evaluation must declare the complete ten-task suite")
    normalized = dict(payload)
    normalized["facilities"] = {}
    for name, raw in facilities.items():
        if not isinstance(raw, Mapping):
            raise PlanError(f"facility {name} must be a mapping")
        _exact(raw, FACILITY_KEYS, f"facility {name}")
        resolved = {key: (str(value) if key in {"runtime", "partition", "stratum"} else int(value)) for key, value in raw.items()}
        if resolved["gpus"] != 1:
            raise PlanError("each smoke row requires exactly one GPU")
        cutover_strata.validate_placement(facility=name, partition=resolved["partition"], stratum=resolved["stratum"], timeout_min=resolved["timeout_min"])
        normalized["facilities"][name] = resolved
    return normalized


def expand_rows(grid: Mapping[str, Any], *, facility: str, results_root: str | Path) -> tuple[dict[str, Any], ...]:
    """Expand the immutable smoke or full production coordinates."""

    placement = dict(grid["facilities"][facility])
    root = Path(results_root).resolve()
    common = {"facility": facility, "runtime": placement.pop("runtime"), "resources": placement}
    smoke = grid["schema"] in {SMOKE_SCHEMA, PROOF_SCHEMA}
    proof = grid["schema"] == PROOF_SCHEMA
    seeds = [grid["train"]["seed"]] if smoke else list(grid["train"]["seeds"])
    checkpoint_steps = [grid["train"]["max_steps"]] if smoke else list(grid["train"]["checkpoint_steps"])
    rows = []
    for seed in seeds:
        row_id = f"seed-{seed:03d}"
        train_dir = root / "02_train" / row_id
        terminal = train_dir / "checkpoints" / f"step_{grid['train']['max_steps']:06d}"
        rows.append({**common, "scale": "smoke" if smoke else "production", "kind": "train", "stage": "02_train", "row_id": row_id, "config": grid["train_config"], "seed": seed, "max_steps": grid["train"]["max_steps"], "n_walkers": grid["train"]["n_walkers"], "result_dir": str(train_dir), "checkpoint_dir": str(terminal)})
    if smoke:
        checkpoint = rows[0]["checkpoint_dir"]
        for chain in range(grid["eval"]["chains"]):
            row_id = f"seed-000-chain-{chain:02d}"
            evaluation = dict(grid["eval"])
            rows.append({**common, "scale": "smoke" if smoke else "production", "kind": "eval", "stage": "03_eval", "row_id": row_id, "config": grid["eval_config"], "seed": chain + 1, **evaluation, "record_capacity": grid["eval"]["n_walkers"] * grid["eval"]["n_draws"], "result_dir": str(root / "03_eval" / row_id), "checkpoint_dir": str(checkpoint)})
        return tuple(rows)
    for seed in seeds:
        for step in checkpoint_steps:
            checkpoint = root / "02_train" / f"seed-{seed:03d}" / "checkpoints" / f"step_{step:06d}"
            for chain in range(grid["eval"]["chains"]):
                row_id = f"seed-{seed:03d}-step-{step:06d}-chain-{chain:02d}"
                evaluation = dict(grid["eval"])
                evaluation.pop("chain_seed_base")
                rows.append({**common, "scale": "smoke" if smoke else "production", "kind": "eval", "stage": "03_eval", "row_id": row_id, "config": grid["eval_config"], "seed": grid["eval"]["chain_seed_base"] + seed * grid["eval"]["chains"] * len(checkpoint_steps) + checkpoint_steps.index(step) * grid["eval"]["chains"] + chain, **evaluation, "record_capacity": grid["eval"]["n_walkers"] * grid["eval"]["n_draws"], "result_dir": str(root / "03_eval" / row_id), "checkpoint_dir": str(checkpoint)})
    return tuple(rows)


def build_plans(grid: Mapping[str, Any], *, facility: str, results_root: str | Path, plan_id: str) -> tuple[StagePlanV2, StagePlanV2, dict[str, Any]]:
    """Build train/eval plans and the study manifest."""

    rows = expand_rows(grid, facility=facility, results_root=results_root)
    tasks: list[LogicalTaskSpec] = []
    for row in rows:
        result = Path(row["result_dir"])
        status = result / "status.json"
        checkpoint = row.get("checkpoint_dir")
        completion = CompletionSpec(policy="status_completed_with_checkpoint" if row["kind"] == "train" else "status_completed", status_path=str(status), checkpoint_path=str(Path(checkpoint) / "COMPLETE") if row["kind"] == "train" else None)
        script = "run_train_row.py" if row["kind"] == "train" else "run_eval_row.py"
        command = ("{python}", str(Path(__file__).resolve().parent / script), "--row-json", json.dumps(row, sort_keys=True), "--plan-attempt-id", plan_id)
        logical_id = logical_task_id_from_parts(stage=row["stage"], run_id=row["row_id"], plan_id=plan_id)
        train_run_id = row["row_id"] if row["kind"] == "train" else f"seed-{int(row['row_id'].split('-')[1]):03d}"
        dependencies = () if row["kind"] == "train" else (logical_task_id_from_parts(stage="02_train", run_id=train_run_id, plan_id=plan_id),)
        resources = row["resources"]
        tasks.append(LogicalTaskSpec(logical_task_id=logical_id, stage=row["stage"], run_id=row["row_id"], command=command, result_dir=str(result), logs=(str(status),), params=dict(row), resources=ResourceSpec(profile="cuda", device="cuda", partition=resources["partition"], threads=resources["cpus"], mem_gb=resources["mem_gb"], gpus=resources["gpus"], timeout_min=resources["timeout_min"], metadata={"stratum": resources["stratum"]}), dependencies=dependencies, completion=completion, metadata={"facility": facility, "runtime": row["runtime"]}))
    train_plan = StagePlanV2(STUDY, "02_train", plan_id, str(results_root), tuple(task for task in tasks if task.stage == "02_train")).validate()
    eval_plan = StagePlanV2(STUDY, "03_eval", plan_id, str(results_root), tuple(task for task in tasks if task.stage == "03_eval")).validate()
    manifest = {"schema": "he-cutover-plan/v1", "study": STUDY, "facility": facility, "plan_id": plan_id, "results_root": str(Path(results_root).resolve()), "rows": list(rows), "stages": {"02_train": "02_train", "03_eval": "03_eval"}}
    return train_plan, eval_plan, manifest


def write_plans(output_dir: str | Path, train_plan: StagePlanV2, eval_plan: StagePlanV2, manifest: Mapping[str, Any]) -> Path:
    output = Path(output_dir)
    train_plan.write(output / "02_train")
    eval_plan.write(output / "03_eval")
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("stage", "kind", "row_id", "facility", "runtime", "result_dir", "checkpoint_dir")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in manifest["rows"]:
            writer.writerow({field: row.get(field, "") for field in fields})
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default=str(Path(__file__).with_name("smoke_grid.yaml")))
    parser.add_argument("--facility", choices=("cannon", "polaris"), required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--plan-attempt-id", required=True)
    args = parser.parse_args(argv)
    plans = build_plans(load_grid(args.grid), facility=args.facility, results_root=args.results_root, plan_id=args.plan_attempt_id)
    write_plans(Path(args.results_root) / "00_plan" / args.plan_attempt_id, *plans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
