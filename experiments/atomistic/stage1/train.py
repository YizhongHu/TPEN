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
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    validate_gateway_binding(
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
    if args.overlay is not None:
        compose_train_overlay(
            cfg, overlay=args.overlay, expected_sha256=args.overlay_sha256,
        )
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))
    prepare_initial_checkpoint_config(cfg)
    return run_from_config(cfg, config_path=str(config_path), command="stage1 train")


if __name__ == "__main__":
    raise SystemExit(main())
