"""Configured launch/restore/record wiring at a declared disposable scale."""

import csv
import copy
import json

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from tpen.run import run_from_config

from experiments.one_electron_variance.protocol import evaluation_config, run_directory, train_config
from experiments.one_electron_variance.launch import _run_config
from experiments.one_electron_variance.analysis import _verify_protocol_config, _verify_restore_receipt


@pytest.mark.parametrize("system", ["hooke", "hydrogen"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="accelerator qualification requires an allocated CUDA device"))])
def test_initial_checkpoint_training_and_complete_replay_records(tmp_path, system, device):
    cfg = train_config(system, 0, tmp_path, device=device)
    cfg["run"]["timezone"] = "UTC"
    cfg["sampler"].update(n_walkers=8, burn_in=2, n_steps=2)
    cfg["trainer"]["max_steps"] = 2
    # Keep cadence 2500: terminal publication must work between scheduled saves,
    # just as it must for the 1000-update excluded pilot.
    dispatch_dir = tmp_path / "dispatch"
    elapsed = _run_config(cfg, directory=dispatch_dir, logical_id=f"wiring:{system}",
                          admission_id="disposable-wiring", source_sha="test-source")
    assert elapsed > 0
    assert json.loads((dispatch_dir / "process_exit.json").read_text())["exit_code"] == 0
    assert (dispatch_dir / "dispatch_records.jsonl").is_file()
    with pytest.raises(FileExistsError, match="refusing a second attempt"):
        _run_config(cfg, directory=tmp_path / "forbidden-retry", logical_id=f"wiring:{system}",
                    admission_id="disposable-wiring", source_sha="test-source")
    assert not (tmp_path / "forbidden-retry").exists()
    train_dir = run_directory(cfg)
    consumed = yaml.safe_load((train_dir / "resolved_config.yaml").read_text())
    _verify_protocol_config(consumed, cfg, train_dir)
    consumed["sampler"]["n_steps"] += 1
    with pytest.raises(ValueError, match="consumed sampler"):
        _verify_protocol_config(consumed, cfg, train_dir)
    activity = json.loads((train_dir / "parameter_activity.json").read_text())
    assert activity["completed_updates"] == 1
    assert any(row["gradient"] == "nonzero" for row in activity["parameters"])
    metrics = [json.loads(line) for line in (train_dir / "metrics.jsonl").read_text().splitlines()]
    if system == "hooke":
        assert any("factors.0.coefficient" in record["metrics"] for record in metrics if record["namespace"] == "train/factors")
    initial = train_dir / "checkpoints" / "step_000000"
    terminal = train_dir / "checkpoints" / "step_000002"
    assert json.loads((initial / "trainer.json").read_text())["completed_updates"] == 0
    assert json.loads((terminal / "trainer.json").read_text())["completed_updates"] == 2
    before = torch.load(initial / "model.pt", map_location="cpu", weights_only=True)
    after = torch.load(terminal / "model.pt", map_location="cpu", weights_only=True)
    # The checkpoint owner's model.pt envelope is inspected, not inferred by file size.
    if "state_dict" in before:
        before, after = before["state_dict"], after["state_dict"]
    assert any(not torch.equal(before[key], after[key]) for key in before if isinstance(before[key], torch.Tensor))
    evaluation = evaluation_config(system, 0, 2500, 0, tmp_path, device=device)
    evaluation["run"]["timezone"] = "UTC"
    evaluation["load"]["path"] = str(terminal)
    evaluation["sampler"].update(n_walkers=8, burn_in=2, n_steps=2)
    generator = evaluation["evaluator"]["tasks"][0]["generator"]
    generator.update(n_draws=3, replay_chunk_size=5)
    assert run_from_config(OmegaConf.create(evaluation), command="pytest one-electron restore") == 0
    _verify_restore_receipt(run_directory(evaluation), terminal, 2)
    directory = run_directory(evaluation) / "trajectory"
    with (directory / "sampled_eval_table.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    with (directory / "raw_readout.csv").open() as handle:
        diagnostics = list(csv.DictReader(handle))
    assert len(rows) == len(diagnostics) == 24  # Three draws, not the final eight rows.
    assert [(int(r["draw_index"]), int(r["walker_index"])) for r in rows] == [(d, w) for d in range(3) for w in range(8)]
    assert [r["sample_index"] for r in rows] == [r["sample_index"] for r in diagnostics]
    receipt = json.loads((directory / "raw_readout.json").read_text())
    assert receipt["row_count"] == 24
    assert receipt["independent_energy_oracle"] is False
    assert (directory / "sampler_trajectory_diagnostics.json").is_file()
    incompatible = copy.deepcopy(evaluation)
    incompatible["run"]["run_id"] += "-incompatible-model"
    incompatible["model"]["readout"]["eps"] = 1e-8
    assert run_from_config(OmegaConf.create(incompatible), command="pytest strict-restore negative control") == 1
    assert not (run_directory(incompatible) / "trajectory" / "sampled_eval_table.csv").exists()
