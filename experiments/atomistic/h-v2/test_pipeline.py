"""Small configured-route controls for the real TPEN local-energy path."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.run import run_from_config


HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("h_v2_protocol", HERE / "protocol.py")
assert spec is not None and spec.loader is not None
protocol = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protocol
spec.loader.exec_module(protocol)


pytestmark = pytest.mark.integration


def _run_config(config: dict, *, command: str) -> int:
    return run_from_config(OmegaConf.create(config), command=command, raise_exceptions=True)


def test_configured_local_energy_sampler_and_model_only_restore(tmp_path: Path) -> None:
    train = protocol.training_config()
    train["run"]["root"] = str(tmp_path)
    train["run"]["run_id"] = "train"
    train["trainer"]["max_steps"] = 1
    train["sampler"].update(n_walkers=8, burn_in=2, n_steps=2)
    train["callbacks"][-1]["schedule"]["every_n"] = 1
    assert _run_config(train, command="h-v2 pipeline train") == 0
    train_dir = tmp_path / "tpen_h_v2" / "atomistic" / "train"
    checkpoint = train_dir / "checkpoints" / "step_000001"
    assert (checkpoint / "COMPLETE").is_file()

    evaluation = protocol.evaluation_config(checkpoint, stream=0, root=tmp_path)
    evaluation["run"]["run_id"] = "eval"
    evaluation["sampler"].update(n_walkers=8, burn_in=2, n_steps=2)
    evaluation["evaluator"]["tasks"][0]["generator"]["n_draws"] = 3
    evaluation["evaluator"]["tasks"][0]["generator"]["chunk_size"] = 5
    evaluation["evaluator"]["tasks"][0]["summaries"][0]["max_samples"] = 24
    result = _run_config(evaluation, command="h-v2 pipeline model-only evaluation")
    if result != 0:
        for path in sorted(tmp_path.rglob("occurrences.jsonl")):
            print(f"evaluation occurrences {path}:\\n{path.read_text(encoding='utf-8')}")
        for path in sorted(tmp_path.rglob("metrics.jsonl")):
            print(f"evaluation metrics {path}:\\n{path.read_text(encoding='utf-8')}")
    assert result == 0
    eval_dir = tmp_path / "tpen_h_v2" / "atomistic" / "eval" / "trajectory"
    with (eval_dir / "sampled_eval_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    assert all(row["finite"] == "True" for row in rows)
    occurrences = train_dir.parent.parent / "eval" / "occurrences.jsonl"
    reports = [json.loads(line) for line in occurrences.read_text().splitlines() if "CheckpointRestored" in line]
    assert len(reports) == 1
    report = reports[0]["fields"]["report"]
    assert report["mode"] == "model_only"
    assert report["loaded_model"] is True
    assert report["loaded_trainer"] is False


def test_training_state_restore_is_rejected_by_evaluation_config(tmp_path: Path) -> None:
    evaluation = protocol.evaluation_config(tmp_path / "checkpoint")
    evaluation["load"]["mode"] = "train_resume"
    with pytest.raises(ValueError, match="model-only"):
        protocol.assert_hydrogen_contract(evaluation, evaluation=True)
