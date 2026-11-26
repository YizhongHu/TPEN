"""T2 probes for exact-type validation of Stage-Q chunk bounds."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from omegaconf import OmegaConf


stage_coordinate = importlib.import_module(
    "experiments.atomistic.he-importance.stage_coordinate"
)
optimizer_grid = importlib.import_module(
    "experiments.atomistic.he-importance.optimizer_grid"
)
train_config = importlib.import_module(
    "experiments.atomistic.he-importance.train_config"
)
run_stage_q = importlib.import_module(
    "experiments.atomistic.he-importance.run_stage_q"
)


def _cell(tmp_path: Path) -> object:
    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path,
        availability=optimizer_grid.roster_availability(),
        stages=("Q",),
    )
    return next(
        cell
        for cell in cells
        if cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["seed_identity"]["label"] == 810001
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
    )


@pytest.mark.parametrize(
    ("limb", "value"),
    [
        ("top", True),
        ("top", 1.0),
        ("consumed", True),
        ("consumed", 1.0),
    ],
)
def test_truthfulness_guard_rejects_bool_and_float_aliases_at_bound_one(
    tmp_path: Path, limb: str, value: object
) -> None:
    """True/1.0 must not satisfy an exact requested integer bound."""

    cell = _cell(tmp_path)
    config = train_config.resolve_train_config(cell, local_energy_chunk_size=1)
    if limb == "top":
        config.trainer.local_energy_chunk_size = value
    else:
        config.runner.trainer.local_energy_chunk_size = value

    with pytest.raises(run_stage_q.StageQSelectionError, match="local_energy_chunk_size"):
        run_stage_q._require_truthful_cell(cell, config, 1)


def test_truthfulness_guard_accepts_exact_int_one(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    config = train_config.resolve_train_config(cell, local_energy_chunk_size=1)
    run_stage_q._require_truthful_cell(cell, config, 1)


def _entrypoint_args(tmp_path: Path) -> object:
    return run_stage_q.build_parser().parse_args(
        [
            "--stage",
            "Q",
            "--context",
            "O-raw",
            "--optimizer",
            "adam",
            "--seed-label",
            "810001",
            "--walkers",
            "8",
            "--local-energy-chunk-size",
            "1",
            "--output-root",
            str(tmp_path),
        ]
    )


@pytest.mark.parametrize("limb", ["top", "consumed", "both"])
def test_late_resolution_alias_refuses_before_production_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limb: str
) -> None:
    """A late-resolved True limb must not reach the production runner."""

    original_overrides = stage_coordinate.sampler_override_values
    monkeypatch.setattr(
        stage_coordinate,
        "sampler_override_values",
        lambda stage, coordinate: (8,)
        if stage == "Q" and coordinate == "n_walkers"
        else original_overrides(stage, coordinate),
    )
    facts = {
        "global_rank": 0,
        "global_size": 1,
        "local_rank": 0,
        "local_size": 1,
        "node_rank": 0,
        "node_size": 1,
        "host": "cpu-test",
        "pid": 1,
        "device": "cpu",
        "job_id": "cpu-test",
        "device_identity": None,
    }
    runner_calls: list[object] = []
    monkeypatch.setattr(run_stage_q, "_execution_facts", lambda: facts)

    def late_resolve(
        packet: object,
        _facts: object,
        *,
        runner: object,
        **kwargs: object,
    ) -> int:
        del _facts, kwargs
        resolved = train_config.resolve_train_config(packet, local_energy_chunk_size=1)
        if limb in {"top", "both"}:
            resolved.trainer.local_energy_chunk_size = True
        if limb in {"consumed", "both"}:
            resolved.runner.trainer.local_energy_chunk_size = True
        return runner(resolved, topology=facts)  # type: ignore[operator]

    monkeypatch.setattr(run_stage_q.launch, "launch_train", late_resolve)
    monkeypatch.setattr(
        run_stage_q.launch,
        "run_from_config",
        lambda *args, **kwargs: runner_calls.append((args, kwargs)) or 0,
    )

    with pytest.raises(run_stage_q.StageQSelectionError, match="local_energy_chunk_size"):
        run_stage_q.execute(_entrypoint_args(tmp_path))
    assert not runner_calls


def test_legacy_none_remains_unbounded_and_not_an_alias(tmp_path: Path) -> None:
    config = train_config.resolve_train_config(_cell(tmp_path))
    assert OmegaConf.select(config, "trainer.local_energy_chunk_size") is None
    assert OmegaConf.select(config, "runner.trainer.local_energy_chunk_size") is None
    run_stage_q._require_truthful_cell(_cell(tmp_path / "check"), config, None)
