"""Focused regression tests for the canonical Stage-Q chunk-size route."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


stage_coordinate = importlib.import_module("experiments.atomistic.he-importance.stage_coordinate")
optimizer_grid = importlib.import_module("experiments.atomistic.he-importance.optimizer_grid")
train_config = importlib.import_module("experiments.atomistic.he-importance.train_config")
run_stage_q = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")


def _cell(tmp_path: Path) -> object:
    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path, availability=optimizer_grid.roster_availability(), stages=("Q",)
    )
    return next(
        cell
        for cell in cells
        if cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["seed_identity"]["label"] == 810001
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
    )


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "3"])
def test_invalid_chunk_bounds_refuse_before_source_access(value: object) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="exact positive integer"):
        train_config.resolve_train_config(object(), local_energy_chunk_size=value)  # type: ignore[arg-type]


def test_resolver_binds_both_consumed_limbs_and_default_is_unbounded(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    bounded = train_config.resolve_train_config(cell, local_energy_chunk_size=3)
    assert bounded.trainer.local_energy_chunk_size == 3
    assert bounded.runner.trainer.local_energy_chunk_size == 3

    legacy = train_config.resolve_train_config(cell)
    assert legacy.trainer.local_energy_chunk_size is None
    assert legacy.runner.trainer.local_energy_chunk_size is None


def test_truthfulness_guard_rejects_consumed_limb_drift(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    config = train_config.resolve_train_config(cell, local_energy_chunk_size=3)
    config.runner.trainer.local_energy_chunk_size = None
    with pytest.raises(run_stage_q.StageQSelectionError, match="runner.trainer= None|runner.trainer=None"):
        run_stage_q._require_truthful_cell(cell, config, 3)

