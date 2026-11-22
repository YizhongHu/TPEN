"""Thin Stage-1 strict model-only evaluation gateway."""

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
from experiments.atomistic.stage1.protocol import resolve_config_path, validate_gateway_binding
from experiments.toolkit.run_paths import resolve_run_dir


class EvalRunRefusal(ValueError):
    """Raised when a Stage-1 eval would reuse an existing result directory."""


def _config_run_dir(cfg: DictConfig) -> Path:
    """Resolve the exact run directory the un-routed core gateway will write."""

    fields = {
        "results_root": OmegaConf.select(cfg, "run.root", default=None),
        "experiment_name": OmegaConf.select(cfg, "experiment.name", default=None),
        "sector": OmegaConf.select(cfg, "experiment.sector", default=None),
        "run_id": OmegaConf.select(cfg, "run.run_id", default=None),
    }
    if any(value is None or not str(value).strip() for value in fields.values()):
        raise ValueError("Stage-1 eval config lacks a complete run identity")
    return resolve_run_dir(fields)


def _refuse_existing_eval_run(cfg: DictConfig) -> None:
    """Refuse an existing row before the core run lifecycle can write."""

    run_dir = _config_run_dir(cfg)
    if run_dir.exists() or run_dir.is_symlink():
        raise EvalRunRefusal(
            "Stage-1 eval refuses existing result directory "
            f"{run_dir}: no run metadata, events, status, queue, log, or result "
            "files may be overwritten"
        )


def main() -> int:
    """Run evaluation through the public gateway with model-only restore."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--kind", choices=("eval",), required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    validate_gateway_binding(
        binding=args.binding, config=str(config_path), overrides=args.overrides, kind=args.kind,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-1 config path does not exist: {config_path}")
    from tpen.run import load_config
    cfg = load_config(str(config_path), args.overrides)
    _refuse_existing_eval_run(cfg)
    # The config/launcher supplies the strict model-only restore policy; this
    # driver does not import checkpoint internals or silently alter identity.
    return run_from_config(cfg, config_path=str(config_path), command="stage1 eval")


if __name__ == "__main__":
    raise SystemExit(main())
