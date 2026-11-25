"""Compose immutable SOAP pilot configs and additive Stage-1 rows.

The composer materializes one fully resolved YAML document per invocation.  A
row points at those bytes through the frozen ``stage1-rendered-config/v1``
extension; no optimizer-specific branch is required in Stage-1.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml

from .rows import RenderedConfig, canonical_hash, row_id, sha256_file, validate_row


_ROOT = Path(__file__).resolve().parents[3]
_TRAIN_BASE = _ROOT / "experiments" / "atomistic" / "stage1" / "configs" / "hooke2" / "train.yaml"
_EVAL_BASE = _ROOT / "experiments" / "atomistic" / "stage1" / "configs" / "hooke2" / "eval.yaml"
_OVERLAY_BASE = _ROOT / "experiments" / "atomistic" / "stage1" / "configs" / "overlays" / "hooke-diagnostics-v2.yaml"
_INTERPOLATION = re.compile(r"\$\{([^{}]+)\}")


@dataclass(frozen=True)
class PilotProfile:
    """Resolved scale and callback choices for one pilot invocation."""

    name: str
    family: str
    updates: int
    walkers: int
    checkpoint_steps: tuple[int, ...]
    diagnostics_profile: str
    score_chunk_size: int | None = None
    evaluation: bool = False
    checkpoint: int | None = None


@dataclass(frozen=True)
class ComposedRow:
    """Rendered config plus its immutable row descriptor."""

    row: Mapping[str, Any]
    rendered: RenderedConfig


R2_PROFILES = (
    PilotProfile("r2-adam-100", "adam", 100, 512, (100,), "diagnostics-v2"),
    PilotProfile("r2-soap-100", "soap", 100, 512, (100,), "diagnostics-v2"),
    PilotProfile("r2-minsr64-100", "minsr64", 100, 512, (100,), "diagnostics-v2", 64),
    PilotProfile("r2-minsr64-20", "minsr64", 20, 512, (20,), "diagnostics-v2", 64),
)
TARGET_PROFILES = (
    PilotProfile("target-adam-5000", "adam", 5000, 512, (1000, 5000), "diagnostics-v2"),
    PilotProfile("target-soap-5000", "soap", 5000, 512, (1000, 5000), "diagnostics-v2"),
    PilotProfile("target-minsr64-5000", "minsr64", 5000, 512, (1000, 5000), "diagnostics-v2", 64),
)


def _lookup(document: Mapping[str, Any], path: str) -> Any:
    value: Any = document
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"unresolved config reference: {path}")
        value = value[part]
    return value


def _resolve(value: Any, document: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        return {key: _resolve(child, document) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve(child, document) for child in value]
    if not isinstance(value, str):
        return value
    match = _INTERPOLATION.fullmatch(value)
    if match:
        return deepcopy(_lookup(document, match.group(1)))
    return _INTERPOLATION.sub(lambda m: str(_lookup(document, m.group(1))), value)


def _set(mapping: dict[str, Any], path: str, value: Any) -> None:
    target = mapping
    parts = path.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _replace_targets(value: Any, *, old: str, new: str) -> Any:
    if isinstance(value, Mapping):
        return {key: _replace_targets(child, old=old, new=new) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_targets(child, old=old, new=new) for child in value]
    return new if value == old else value


def _callback(target: str, **values: Any) -> dict[str, Any]:
    result = {"_target_": target}
    result.update(values)
    return result


def _callbacks(base: list[Mapping[str, Any]], family: str, profile: str) -> list[dict[str, Any]]:
    callbacks = deepcopy(base)
    # The protocol requires OptimizerState for Adam/SOAP only.  This is a
    # policy derived from the P0 contract, not a Stage-1 optimizer branch.
    callbacks = [item for item in callbacks if item.get("_target_") != "tpen.callback.OptimizerState"]
    if family in {"adam", "soap"}:
        callbacks.append(_callback("tpen.callback.OptimizerState", every_n_steps=1))
    callbacks = [item for item in callbacks if item.get("_target_") != "tpen.callback.InitialCheckpoint"]
    callbacks.insert(0, _callback("tpen.callback.InitialCheckpoint", output_dir="${run.dir}/checkpoints"))
    for item in callbacks:
        if item.get("_target_") in {"tpen.callback.TrainPhaseTiming", "tpen.callback.TrainStepTiming"}:
            # R2/20, R1, and target keep synchronization disabled; the R2/100
            # callback rule is represented by RunTiming's synchronized clock.
            item["accelerator_synchronize"] = False
    callbacks.append(_callback("tpen.callback.RunTiming", accelerator_synchronize=True))
    return callbacks


def _optimizer_and_trainer(profile: PilotProfile) -> tuple[dict[str, Any], dict[str, Any]]:
    base_lr = 0.005
    trainer = {"_target_": "tpen.training.trainer.VMCTrainer", "max_steps": profile.updates, "log_every_n_steps": 1, "return_terms": True, "nonfinite_local_energy_policy": "fail"}
    if profile.family == "adam":
        return {"_target_": "torch.optim.Adam", "_partial_": True, "lr": base_lr}, trainer
    if profile.family == "soap":
        return {"_target_": "tpen.training.SOAP", "_partial_": True, "lr": base_lr, "betas": [0.95, 0.95], "precondition_frequency": 10, "max_precond_dim": 128, "precondition_1d": False, "correct_bias": True}, trainer
    if profile.family == "minsr64":
        trainer["update_method"] = {"_target_": "tpen.training.sr.StochasticReconfigurationUpdate", "_partial_": True, "policy": {"_target_": "tpen.training.sr.SRPolicy", "solve_space": "sample", "learning_rate": 0.01, "damping": {"_target_": "tpen.training.qgt.DampingPolicy", "relative": 0.001, "absolute": 0.0, "minimum": 1e-12}, "rank_cutoff": 0.0, "max_update_norm": None, "score_chunk_size": profile.score_chunk_size}, "conventions": {"_target_": "tpen.training.score_geometry.ScoreConventions", "solve_dtype": "float64"}}
        return {"_target_": "torch.optim.SGD", "_partial_": True, "lr": 0.01}, trainer
    raise ValueError(f"unknown pilot family {profile.family}")


def _load_base(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _materialize_train(profile: PilotProfile, *, results_root: str, run_id: str, source_sha: str, device: str) -> tuple[dict[str, Any], str]:
    config = _load_base(_TRAIN_BASE)
    optimizer, trainer = _optimizer_and_trainer(profile)
    _set(config, "run.root", results_root)
    _set(config, "run.run_id", run_id)
    _set(config, "run.dir", str(Path(results_root) / "stage1_hooke2" / "atomistic" / run_id))
    _set(config, "runtime.seed", 2435329684536531677)
    _set(config, "runtime.device", device)
    _set(config, "runtime.dtype", "float64")
    _set(config, "sampler.n_walkers", profile.walkers)
    _set(config, "sampler.seed", 7066630035130326805)
    _set(config, "trainer", trainer)
    _set(config, "optimizer", optimizer)
    _set(config, "checkpoint_steps", list(profile.checkpoint_steps))
    _set(config, "callbacks", _callbacks(config.get("callbacks", []), profile.family, profile.diagnostics_profile))
    _set(config, "stage1_overlay", {"path": str(_OVERLAY_BASE), "sha256": sha256_file(_OVERLAY_BASE), "profile": profile.diagnostics_profile})
    config = _resolve(config, config)
    if any("${" in repr(value) or value == "???" for value in config.values()):
        raise ValueError("rendered train config retains unresolved values")
    return config, "train"


def _materialize_eval(*, results_root: str, run_id: str, checkpoint_dir: str, checkpoint_file: str, source_sha: str, device: str) -> tuple[dict[str, Any], str]:
    config = _load_base(_EVAL_BASE)
    _set(config, "run.root", results_root)
    _set(config, "run.run_id", run_id)
    _set(config, "run.dir", str(Path(results_root) / "stage1_hooke2" / "atomistic" / run_id))
    _set(config, "runtime.seed", 6348702091350892190)
    _set(config, "runtime.device", device)
    _set(config, "runtime.dtype", "float64")
    _set(config, "load.path", checkpoint_dir)
    _set(config, "trajectory_identity.checkpoint_file", checkpoint_file)
    config = _resolve(config, config)
    return config, "eval"


def _write_yaml(path: Path, config: Mapping[str, Any]) -> RenderedConfig:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(config, sort_keys=False, default_flow_style=False).encode("utf-8")
    path.write_bytes(payload)
    return RenderedConfig(str(path.resolve()), hashlib.sha256(payload).hexdigest(), config, "diagnostics-v2")


def _common_row(*, rendered: RenderedConfig, profile: PilotProfile, results_root: str, run_id: str, source_sha: str, config_path: str, device: str) -> dict[str, Any]:
    row = {
        "system": "hooke_pair_singlet_omega0.5", "system_id": "hooke_pair_singlet_omega0.5", "kind": "train",
        "seed": 2435329684536531677, "source_sha": source_sha, "config_sha256": sha256_file(config_path),
        "config_identity_rule": "stage1-config-v1", "config": "configs/hooke2/train.yaml", "results_root": results_root,
        "experiment_name": "stage1_hooke2", "sector": "atomistic", "run_id": run_id, "updates": profile.updates,
        "walkers": profile.walkers, "device": device, "lineage": 0, "sampler_seed": 7066630035130326805,
        "checkpoint_steps": list(profile.checkpoint_steps), "overlay": "configs/overlays/hooke-diagnostics-v2.yaml",
        "overlay_sha256": sha256_file(_OVERLAY_BASE), "row_extension": {"version": "stage1-rendered-config/v1", "rendered_config": {"path": rendered.path, "sha256": rendered.sha256}, "diagnostics_profile": profile.diagnostics_profile},
        "overlay_profile": profile.diagnostics_profile,
    }
    row["row_id"] = row_id(row)
    validate_row(row, expected_kind="train")
    return row


def render_profile(profile: PilotProfile, *, output_dir: str | Path, results_root: str, source_sha: str, run_id: str | None = None, device: str = "cuda") -> ComposedRow:
    """Render one train profile and return its immutable row descriptor."""

    if run_id is None:
        run_id = profile.name.replace("-", "_")
    config, _ = _materialize_train(profile, results_root=results_root, run_id=run_id, source_sha=source_sha, device=device)
    path = Path(output_dir) / f"{profile.name}.yaml"
    rendered = _write_yaml(path, config)
    row = _common_row(rendered=rendered, profile=profile, results_root=results_root, run_id=run_id, source_sha=source_sha, config_path=str(_TRAIN_BASE), device=device)
    return ComposedRow(row, rendered)


def render_eval(*, output_dir: str | Path, results_root: str, source_sha: str, checkpoint_dir: str, checkpoint_sha256: str, run_id: str = "soap_eval_100", device: str = "cuda", parent_row: Mapping[str, Any] | None = None) -> ComposedRow:
    """Render the strict fresh SOAP checkpoint-100 evaluation row."""

    config, _ = _materialize_eval(results_root=results_root, run_id=run_id, checkpoint_dir=checkpoint_dir, checkpoint_file="model.pt", source_sha=source_sha, device=device)
    rendered = _write_yaml(Path(output_dir) / "r2-soap-eval-100.yaml", config)
    parent = parent_row or {"row_id": "parent", "system": "hooke_pair_singlet_omega0.5", "system_id": "hooke_pair_singlet_omega0.5", "seed": 2435329684536531677, "source_sha": source_sha, "config_sha256": sha256_file(_EVAL_BASE), "results_root": results_root, "experiment_name": "stage1_hooke2", "sector": "atomistic", "device": device, "lineage": 0}
    row = {
        "system": parent["system"], "system_id": parent["system_id"], "kind": "eval", "seed": 6348702091350892190,
        "source_sha": source_sha, "config_sha256": sha256_file(_EVAL_BASE), "config_identity_rule": "stage1-config-v1",
        "config": "configs/hooke2/eval.yaml", "results_root": results_root, "experiment_name": "stage1_hooke2", "sector": "atomistic",
        "run_id": run_id, "updates": 100, "walkers": 4096, "device": device, "row_extension": {"version": "stage1-rendered-config/v1", "rendered_config": {"path": rendered.path, "sha256": rendered.sha256}, "diagnostics_profile": "diagnostics-v2"},
        "lineage": parent.get("lineage", 0), "checkpoint": 100, "stream": 0, "draws": 256, "expected_physical_system_id": parent["system_id"],
        "checkpoint_dir": checkpoint_dir, "checkpoint_file": "model.pt", "checkpoint_sha256": checkpoint_sha256,
        "depends_on": parent["row_id"], "phase1_manifest_hash": canonical_hash([parent]), "phase1_train_row_hash": canonical_hash(parent),
    }
    row["row_id"] = row_id(row)
    validate_row(row, expected_kind="eval")
    return ComposedRow(row, rendered)


def render_profiles(*, output_dir: str | Path, results_root: str, source_sha: str, profiles: Iterable[PilotProfile] = R2_PROFILES, device: str = "cuda") -> dict[str, ComposedRow]:
    """Render a deterministic set of train profiles."""

    return {profile.name: render_profile(profile, output_dir=output_dir, results_root=results_root, source_sha=source_sha, device=device) for profile in profiles}


__all__ = ["ComposedRow", "PilotProfile", "R2_PROFILES", "TARGET_PROFILES", "render_eval", "render_profile", "render_profiles"]
