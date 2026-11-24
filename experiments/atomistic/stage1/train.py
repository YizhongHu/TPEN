"""Thin Stage-1 training gateway."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Direct script execution puts only this directory on sys.path.  Add the
# checkout root before importing the source-owned experiments package.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tpen.run import run_from_config
from experiments.atomistic.stage1.protocol import (
    compose_train_overlay,
    resolve_config_path,
    row_config_values,
    validate_resolved_train_config,
    validate_gateway_binding,
)


def _bind_output_identity(cfg: DictConfig, row: Mapping[str, object]) -> None:
    """Bind the row-owned output identity before any config snapshot."""

    for path, key in (
        ("run.root", "results_root"),
        ("experiment.name", "experiment_name"),
        ("experiment.sector", "sector"),
        ("run.run_id", "run_id"),
        ("run.layout", "layout"),
    ):
        value = row.get(key, "nested" if key == "layout" else None)
        if value is None:
            raise ValueError(f"manifest row is missing {key}")
        OmegaConf.update(cfg, path, value, merge=False)


def build_train_config(
    row: Mapping[str, object], *, config_path: str | Path | None = None,
    overrides: Sequence[str] | None = None, overlay: str | None = None,
    overlay_sha256: str | None = None,
) -> DictConfig:
    """Compose one train row and validate its final producer configuration.

    The output identity is bound before preserving the base snapshot.  This
    makes the config self-contained even when the YAML deliberately contains
    a mandatory output slot instead of an environment interpolation.
    """

    selected_config = resolve_config_path(config_path or str(row.get("config", "")))
    if not selected_config.is_file():
        raise FileNotFoundError(f"Stage-1 config path does not exist: {selected_config}")
    from tpen.run import load_config
    cfg = load_config(str(selected_config))
    _bind_output_identity(cfg, row)
    base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))

    selected_overlay = overlay if overlay is not None else row.get("overlay")
    selected_overlay_sha = (
        overlay_sha256 if overlay_sha256 is not None else row.get("overlay_sha256")
    )
    if selected_overlay is not None:
        if selected_overlay_sha is None:
            raise ValueError("train overlay is missing its sha256 binding")
        compose_train_overlay(
            cfg, overlay=str(selected_overlay), expected_sha256=str(selected_overlay_sha),
        )

    cfg = OmegaConf.merge(cfg, OmegaConf.create(row_config_values(row)))
    # The optional dotlist remains only for the pinned restart adapter.  The
    # real gateway has no positional override channel and never reaches this
    # branch.
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    # A caller-supplied compatibility override must not redirect artifacts.
    _bind_output_identity(cfg, row)
    if row.get("science_profile") == "hooke-hv2-v1":
        # Preserve the producer-boundary guard while keeping composition in
        # this shared builder for both fresh and restart gateways.
        if cfg.runtime.seed != row["seed"] or cfg.sampler.seed != row["sampler_seed"]:
            raise ValueError("Hooke science runtime and sampler seeds are not bound")
        if cfg.runtime.seed == cfg.sampler.seed:
            raise ValueError("Hooke science runtime and sampler seeds must be distinct")
    prepare_initial_checkpoint_config(cfg)
    validate_resolved_train_config(cfg, row, base_config=base_cfg)
    return cfg


def prepare_initial_checkpoint_config(cfg: DictConfig) -> None:
    """Route checkpoint zero to the existing initial-state callback.

    Stage-1 manifests retain zero as a checkpoint coordinate, but the core
    ``ExplicitUpdates`` schedule intentionally accepts positive updates only.
    Keep the manifest-facing value until gateway validation and composition
    have completed, then remove zero from the schedule and append the existing
    callback that writes the pre-sampling zero boundary.
    """

    raw_steps = list(cfg.get("checkpoint_steps", ()))
    if any(type(step) is not int or step < 0 for step in raw_steps):
        raise ValueError("checkpoint_steps must contain non-negative integers")
    if 0 not in raw_steps:
        return
    cfg.checkpoint_steps = [step for step in raw_steps if step > 0]
    cfg.callbacks.append({
        "_target_": "tpen.callback.InitialCheckpoint",
        "output_dir": "${run.dir}/checkpoints",
    })


def main() -> int:
    """Run one configured training job through TPEN's public gateway."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument(
        "--restart", action="store_true",
        help="run the admitted row through the Stage-1 restart protocol",
    )
    args = parser.parse_args()
    row = validate_gateway_binding(binding=args.binding)
    cfg = build_train_config(row)
    config_path = resolve_config_path(str(row["config"]))
    if args.restart:
        from experiments.atomistic.stage1.restart import execute_restart

        result = execute_restart(
            cfg=cfg, row=row, config_path=config_path, command="stage1 restart",
        )
        return 0 if result.outcome == "completed" else 1
    return run_from_config(cfg, config_path=str(config_path), command="stage1 train")


if __name__ == "__main__":
    raise SystemExit(main())
