"""Focused regression tests for the canonical Stage-Q chunk-size route."""

from __future__ import annotations

import importlib
from pathlib import Path
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


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_trainer_override_has_the_same_strict_validation(tmp_path: Path, value: object) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="exact positive integer"):
        train_config.resolve_train_config(
            _cell(tmp_path), overrides={"trainer.local_energy_chunk_size": value}
        )


def test_explicit_bound_cannot_conflict_with_trainer_override(tmp_path: Path) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="conflicts"):
        train_config.resolve_train_config(
            _cell(tmp_path),
            local_energy_chunk_size=3,
            overrides={"trainer.local_energy_chunk_size": 2},
        )


@pytest.mark.parametrize("argv", [["--local-energy-chunk-size", "0"], ["--local-energy-chunk-size", "-1"]])
def test_stage_q_invalid_cli_bounds_refuse_before_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    args = run_stage_q.build_parser().parse_args([*argv, "--output-root", str(tmp_path)])
    monkeypatch.setattr(run_stage_q, "select_cell", lambda *a, **k: pytest.fail("selection ran"))
    with pytest.raises(run_stage_q.StageQSelectionError, match="exact positive integer"):
        run_stage_q.execute(args)
    assert not tuple(tmp_path.iterdir())

