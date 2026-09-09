"""Integration coverage for Hooke smoke training across small electron counts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.run import run_from_config

CONFIG = Path(__file__).resolve().parents[1] / "artifacts" / "hooke" / "pair_train.yaml"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("n_electrons", "n_up", "n_down"),
    [
        (0, 0, 0),
        (1, 1, 0),
        (2, 1, 1),
        (3, 2, 1),
    ],
)
def test_hooke_models_initialize_and_train_for_small_electron_counts(
    tmp_path: Path,
    n_electrons: int,
    n_up: int,
    n_down: int,
) -> None:
    cfg = OmegaConf.load(CONFIG)
    cfg.run.root = str(tmp_path)
    cfg.experiment.name = f"hooke_{n_electrons}_electron_smoke"
    cfg.experiment.sector = f"n{n_electrons}"
    cfg.experiment.run_name = f"hooke_{n_electrons}_electron_train"
    cfg.system.n_particles = n_electrons
    cfg.system.spin.n_up = n_up
    cfg.system.spin.n_down = n_down
    cfg.sampler.n_walkers = 4
    cfg.sampler.burn_in = 1
    cfg.sampler.n_steps = 1
    cfg.trainer.max_steps = 1

    exit_code = run_from_config(cfg, config_path=str(CONFIG), command="pytest")

    assert exit_code == 0
    run_dirs = list((tmp_path / cfg.experiment.name / cfg.experiment.sector).iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    train_records = [record for record in records if record.get("namespace") == "train"]
    sampler_records = [record for record in records if record.get("namespace") == "train/sampler"]
    assert len(train_records) == 1
    assert len(sampler_records) == 1
    assert sampler_records[0]["metrics"]["n_walkers"] == 4

    train_metrics = train_records[0]["metrics"]
    assert train_metrics["local_energy_n_total"] == 4
    assert train_metrics["loss_has_grad"] is (n_electrons > 0)
    assert train_metrics["optimizer_step"] is (n_electrons > 0)

    occurrences = [
        json.loads(line)
        for line in (run_dir / "occurrences.jsonl").read_text().splitlines()
        if line.strip()
    ]
    typed_events = {record["event"] for record in occurrences}
    scoped_operations = {record["operation"] for record in occurrences if "operation" in record}
    if n_electrons == 0:
        # The vacuum skips its update: no OptimizerUpdate scope opens at all.
        assert "tpen.training.events.UpdateSkipped" in typed_events
        assert "tpen.training.events.UpdateCompleted" not in typed_events
        assert "tpen.training.events.OptimizerUpdate" not in scoped_operations
    else:
        assert "tpen.training.events.UpdateCompleted" in typed_events
        assert "tpen.training.events.UpdateSkipped" not in typed_events
        assert "tpen.training.events.OptimizerUpdate" in scoped_operations

    # Periodic checkpoint eligibility is decided by `UpdateCompleted`, so an
    # iteration that skipped its optimizer update writes no checkpoint at all.
    # This config wires a periodic-only Checkpoint (``terminal: false``), so the
    # vacuum run produces no checkpoint directory; its counters are pinned by
    # the trainer unit tests instead.
    checkpoint_dir = run_dir / "checkpoints" / "step_000001"
    if n_electrons == 0:
        assert not checkpoint_dir.exists()
    else:
        # The resume cursor advances for every attempted iteration; only
        # applied updates increment completed_updates.
        trainer_state = json.loads((checkpoint_dir / "trainer.json").read_text())
        assert set(trainer_state) == {
            "next_iteration",
            "completed_updates",
            "parameter_layout",
            "nonfinite_local_energy_policy",
        }
        assert trainer_state["next_iteration"] == 1
        assert trainer_state["completed_updates"] == 1
        # Pinned to the literal, not to the module default: see the note in
        # test_train_runner.py. This one line serves all three electron-count
        # parametrizations.
        assert trainer_state["nonfinite_local_energy_policy"] == "mask"
