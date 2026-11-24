"""Thin Stage-1 strict model-only evaluation gateway."""

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
    resolve_config_path,
    row_config_values,
    validate_gateway_binding,
)
from experiments.toolkit.run_paths import resolve_run_dir


class EvalRunRefusal(ValueError):
    """Raised when a Stage-1 eval would reuse an existing result directory."""


def _bind_output_identity(cfg: DictConfig, row: Mapping[str, object]) -> None:
    """Bind the row-owned eval output identity before config resolution."""

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


def build_eval_config(
    row: Mapping[str, object], *, config_path: str | Path | None = None,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Compose one eval row with its output slot filled from the row."""

    selected_config = resolve_config_path(config_path or str(row.get("config", "")))
    if not selected_config.is_file():
        raise FileNotFoundError(f"Stage-1 config path does not exist: {selected_config}")
    from tpen.run import load_config

    cfg = load_config(str(selected_config))
    _bind_output_identity(cfg, row)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(row_config_values(row)))
    # Kept solely for the pinned pre-LR-2 restart adapter; the real eval
    # gateway has no repeated override channel.
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    # Keep the row authoritative for artifact placement if a compatibility
    # caller supplied a contradictory run.root override.
    _bind_output_identity(cfg, row)
    return cfg


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
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    row = validate_gateway_binding(binding=args.binding)
    cfg = build_eval_config(row)
    config_path = resolve_config_path(str(row["config"]))
    _refuse_existing_eval_run(cfg)
    # The config/launcher supplies the strict model-only restore policy; this
    # driver does not import checkpoint internals or silently alter identity.
    return run_from_config(cfg, config_path=str(config_path), command="stage1 eval")


if __name__ == "__main__":
    raise SystemExit(main())
