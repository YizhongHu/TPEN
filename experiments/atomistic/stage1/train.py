"""Thin Stage-1 training gateway."""

from __future__ import annotations

import argparse
import sys
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
    validate_resolved_train_config,
    validate_gateway_binding,
)


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
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--kind", choices=("smoke", "train"), required=True)
    parser.add_argument("--overlay")
    parser.add_argument("--overlay-sha256")
    parser.add_argument(
        "--restart", action="store_true",
        help="run the admitted row through the Stage-1 restart protocol",
    )
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    row = validate_gateway_binding(
        binding=args.binding, config=str(config_path), overrides=args.overrides,
        kind=args.kind, overlay=args.overlay, overlay_sha256=args.overlay_sha256,
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-1 config path does not exist: {config_path}")
    # Config loading remains owned by the installed tpen entrypoint; this
    # wrapper is intentionally only a gateway and does not duplicate Runner.
    from tpen.run import load_config
    # Compose the declared train overlay before applying row overrides so the
    # manifest remains authoritative for seeds, sizes, and run identity.
    cfg = load_config(str(config_path))
    # Keep an immutable resolved snapshot for the final Hooke science boundary;
    # the same object is intentionally mutated by overlay composition below.
    base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if args.overlay is not None:
        compose_train_overlay(
            cfg, overlay=args.overlay, expected_sha256=args.overlay_sha256,
        )
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))
    if row.get("science_profile") == "hooke-hv2-v1":
        # Recheck the resolved producer boundary after OmegaConf composition;
        # checking only the textual override list would allow a later merge to
        # alias the initialization and sampler streams.
        if cfg.runtime.seed != row["seed"] or cfg.sampler.seed != row["sampler_seed"]:
            raise ValueError("Hooke science runtime and sampler seeds are not bound")
        if cfg.runtime.seed == cfg.sampler.seed:
            raise ValueError("Hooke science runtime and sampler seeds must be distinct")
    prepare_initial_checkpoint_config(cfg)
    validate_resolved_train_config(cfg, row, base_config=base_cfg)
    if args.restart:
        from experiments.atomistic.stage1.restart import execute_restart

        result = execute_restart(
            cfg=cfg, row=row, config_path=config_path, command="stage1 restart",
        )
        return 0 if result.outcome == "completed" else 1
    return run_from_config(cfg, config_path=str(config_path), command="stage1 train")


if __name__ == "__main__":
    raise SystemExit(main())
