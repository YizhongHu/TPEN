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

from experiments.atomistic.stage1.protocol import SYSTEM_SPECS, config_identity, planned_physical_system_ids

from .rows import RenderedConfig, manifest_hash, row_id, sha256_file, validate_row


_ROOT = Path(__file__).resolve().parents[3]
_TRAIN_BASE = _ROOT / "experiments" / "atomistic" / "stage1" / "configs" / "hooke2" / "train.yaml"
_EVAL_BASE = _ROOT / "experiments" / "atomistic" / "stage1" / "configs" / "hooke2" / "eval.yaml"
_OVERLAY_BASE = _ROOT / "experiments" / "atomistic" / "stage1" / "configs" / "overlays" / "hooke-diagnostics-v2.yaml"
_STAGE1_CONFIG_ROOT = _ROOT / "experiments" / "atomistic" / "stage1" / "configs"
_HOOKE2 = next(spec for spec in SYSTEM_SPECS if spec.name == "hooke2")
_HOOKE2_CONFIG_SHA256 = config_identity(
    config_root=_STAGE1_CONFIG_ROOT, systems=("hooke2",)
)["sha256"]
_HOOKE2_PHYSICAL_SYSTEM_ID = planned_physical_system_ids()["hooke2"]
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
    train_timing_synchronize: bool = False


@dataclass(frozen=True)
class ComposedRow:
    """Rendered config plus its immutable row descriptor."""

    row: Mapping[str, Any]
    rendered: RenderedConfig


R2_PROFILES = (
    PilotProfile("r2-adam-100", "adam", 100, 512, (0, 100), "diagnostics-v2", train_timing_synchronize=True),
    PilotProfile("r2-soap-100", "soap", 100, 512, (0, 100), "diagnostics-v2", train_timing_synchronize=True),
    PilotProfile("r2-minsr64-100", "minsr64", 100, 512, (0, 100), "diagnostics-v2", 64, train_timing_synchronize=True),
    PilotProfile("r2-minsr64-20", "minsr64", 20, 512, (0, 20), "diagnostics-v2", 64),
)
TARGET_PROFILES = (
    PilotProfile("target-adam-5000", "adam", 5000, 512, (0, 1000, 5000), "diagnostics-v2"),
    PilotProfile("target-soap-5000", "soap", 5000, 512, (0, 1000, 5000), "diagnostics-v2"),
    PilotProfile("target-minsr64-5000", "minsr64", 5000, 512, (0, 1000, 5000), "diagnostics-v2", 64),
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


def _resolve_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve nested references to a fixed point without rereading YAML."""

    current: dict[str, Any] = deepcopy(dict(document))
    for _ in range(12):
        resolved = _resolve(current, current)
        if resolved == current:
            return resolved
        current = resolved
    raise ValueError("rendered config interpolation did not converge")


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


def _callbacks(base: list[Mapping[str, Any]], family: str, profile: str, *, synchronize: bool) -> list[dict[str, Any]]:
    dynamic_targets = {
        "tpen.callback.OptimizerState",
        "tpen.callback.InitialCheckpoint",
        "tpen.callback.ResourceUsage",
        "tpen.callback.TrainPhaseTiming",
        "tpen.callback.TrainStepTiming",
        "tpen.callback.RunTiming",
        "tpen.callback.RawTrainingBatchRecorder",
        "tpen.callback.ParameterActivity",
        "tpen.callback.Checkpoint",
    }
    optimizer_state = next(
        (deepcopy(item) for item in base if item.get("_target_") == "tpen.callback.OptimizerState"),
        None,
    )
    callbacks = [item for item in deepcopy(base) if item.get("_target_") not in dynamic_targets]
    # The protocol requires OptimizerState for Adam/SOAP only.  This is a
    # policy derived from the P0 contract, not a Stage-1 optimizer branch.
    if family in {"adam", "soap"}:
        if optimizer_state is None or "groups" not in optimizer_state:
            raise ValueError("OptimizerState callback requires groups in the base overlay")
        optimizer_state["every_n_steps"] = 1
        callbacks.append(optimizer_state)
    callbacks.append(_callback("tpen.callback.RawTrainingBatchRecorder", path="${run.dir}/training_history", every_n_steps=50, partition_size_steps=1000))
    callbacks.append(_callback("tpen.callback.Checkpoint", terminal=False, schedule={"_target_": "tpen.checkpoint.ExplicitUpdates", "updates": "${checkpoint_steps}"}, payload={"_target_": "tpen.checkpoint.TrainResume"}, output_dir="${run.dir}/checkpoints"))
    callbacks.append(_callback("tpen.callback.InitialCheckpoint", output_dir="${run.dir}/checkpoints"))
    callbacks.append(_callback("tpen.callback.ParameterActivity", output_path="${run.dir}/parameter_activity.json"))
    callbacks.append(_callback("tpen.callback.ResourceUsage", allocator_probe={"_target_": "tpen.accelerator.TorchAllocatorPeakProbe", "device": "${runtime.device}"}))
    callbacks.append(_callback("tpen.callback.TrainPhaseTiming", every_n_steps=1, accelerator_synchronize=synchronize))
    callbacks.append(_callback("tpen.callback.TrainStepTiming", every_n_steps=1, rolling_window=20, accelerator_synchronize=synchronize))
    for item in callbacks:
        if item.get("_target_") in {"tpen.callback.TrainPhaseTiming", "tpen.callback.TrainStepTiming"}:
            # R2/20, R1, and target keep synchronization disabled; the R2/100
            # callback rule is represented by RunTiming's synchronized clock.
            item["accelerator_synchronize"] = synchronize
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
    diagnostics = _load_base(_OVERLAY_BASE)
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
    # Stage-1 rows include the zero origin; materialized YAML uses positive steps.
    _set(config, "checkpoint_steps", list(profile.checkpoint_steps[1:]))
    # Compose the committed base callbacks with the committed diagnostic
    # profile.  The renderer only changes the explicitly declared pilot knobs;
    # it does not invent a second smoke callback set.
    _set(config, "callbacks", _callbacks(config.get("callbacks", []) + diagnostics.get("callbacks", []), profile.family, profile.diagnostics_profile, synchronize=profile.train_timing_synchronize))
    # Materialized admission compares this provenance object byte-for-byte
    # with the row-owned overlay basis.  Keep the rendered YAML self-contained
    # while preserving the v2 recorder-before-checkpoint composition contract.
    _set(config, "stage1_overlay", {
        "schema": "stage1-overlay/v2",
        "path": "configs/overlays/hooke-diagnostics-v2.yaml",
        "sha256": sha256_file(_OVERLAY_BASE),
        "composition_order": (
            "base-config -> recorder-before-checkpoint diagnostics-v2 overlay -> manifest overrides"
        ),
        "stream": "train",
        "profile": profile.diagnostics_profile,
    })
    config = _resolve_document(config)
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
    config = _resolve_document(config)
    load = config.get("load", {})
    if load.get("mode") != "model_only" or load.get("strict") is not True or load.get("allow_protocol_mismatch") is not False:
        raise ValueError("frozen evaluation must be strict model_only")
    return config, "eval"


def _write_yaml(path: Path, config: Mapping[str, Any]) -> RenderedConfig:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(config, sort_keys=False, default_flow_style=False).encode("utf-8")
    path.write_bytes(payload)
    return RenderedConfig(str(path.resolve()), hashlib.sha256(payload).hexdigest(), config, "diagnostics-v2")


def _common_row(*, rendered: RenderedConfig, profile: PilotProfile, results_root: str, run_id: str, source_sha: str, config_path: str, device: str) -> dict[str, Any]:
    row = {
        "system": _HOOKE2.name, "system_id": _HOOKE2.system_id, "kind": "train",
        "seed": 2435329684536531677, "source_sha": source_sha, "config_sha256": _HOOKE2_CONFIG_SHA256,
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
    """Reject the legacy concrete-eval composer path.

    A train row is not publication evidence: it may be a freshly rendered
    synthetic row and carries no proof that a SOAP100 checkpoint was
    published.  ``p3_render.finalize_eval_row`` is the sole supported path for
    producing a concrete eval row because it joins a real parent and named
    publication before writing output.
    """

    raise ValueError(
        "composer.render_eval cannot create a concrete eval row; use "
        "p3_render.render_eval_template and finalize_eval_row with a named "
        "complete SOAP100 publication"
    )


def render_profiles(*, output_dir: str | Path, results_root: str, source_sha: str, profiles: Iterable[PilotProfile] = R2_PROFILES, device: str = "cuda") -> dict[str, ComposedRow]:
    """Render a deterministic set of train profiles."""

    return {profile.name: render_profile(profile, output_dir=output_dir, results_root=results_root, source_sha=source_sha, device=device) for profile in profiles}


__all__ = ["ComposedRow", "PilotProfile", "R2_PROFILES", "TARGET_PROFILES", "render_eval", "render_profile", "render_profiles"]
