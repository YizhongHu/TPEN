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
    assert _run_config(evaluation, command="h-v2 pipeline model-only evaluation") == 0
    eval_dir = tmp_path / "tpen_h_v2" / "atomistic" / "eval" / "trajectory"
    with (eval_dir / "sampled_eval_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    assert all(row["finite"] == "True" for row in rows)
    occurrences = train_dir.parent / "eval" / "occurrences.jsonl"
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


@pytest.mark.h2v2_pipeline
def test_part_e_contract_scale_training_and_evaluation_energy_criteria(tmp_path: Path) -> None:
    """Run the h-v2 short learned route at full contract scale (part e).

    Contract (``contract_inputs.json`` ``admission.pipeline_check`` part e):
    8 channels, 32 hidden, 1000 updates; training sampler 256 walkers,
    burn-in 100, 10 Metropolis steps/update, proposal scale 0.5 bohr; fresh
    evaluation 1024 walkers x 64 draws x 2 streams, burn-in 1000, stride 20.
    Computes the ``0.02`` Ha energy-error criterion and the ``0.25`` median-
    absolute-error improvement rule, with the little-dynamic-range branch
    when the untrained (update-zero) model is already within ``0.02`` Ha --
    in which case the contract requires only the imperfect learned-state/
    gradient control, which `test_protocol.test_exact_and_imperfect_analytic_controls`
    (``alpha=1.2``) already provides.

    Marked ``h2v2_pipeline`` for selection, not exclusion: this repo's pytest
    markers (see also ``integration``) are categorical only -- nothing here
    filters them out of the standard ``pytest -q experiments/atomistic/h-v2
    experiments/atomistic/he-v1`` invocation, and this test does run as part
    of it (measured on Cannon: ~5-6 extra minutes on top of the rest of the
    suite, well inside the contract's 2h/4-core CPU budget). Use the marker to
    run ONLY this test in isolation::

        uv run pytest -q -m h2v2_pipeline experiments/atomistic/h-v2/test_pipeline.py
    """

    def _median_abs_error(checkpoint: Path, run_id: str) -> float:
        errors: list[float] = []
        for stream in (0, 1):
            evaluation = protocol.evaluation_config(checkpoint, stream=stream, root=tmp_path)
            evaluation["run"]["run_id"] = f"{run_id}-stream{stream}"
            assert _run_config(evaluation, command=f"h-v2 pipeline part-e {run_id} stream {stream}") == 0
            eval_dir = tmp_path / "tpen_h_v2" / "atomistic" / f"{run_id}-stream{stream}" / "trajectory"
            with (eval_dir / "sampled_eval_table.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            assert rows
            assert all(row["finite"] == "True" for row in rows)
            errors.extend(abs(float(row["local_energy"]) + 0.5) for row in rows)
        errors.sort()
        midpoint = len(errors) // 2
        if len(errors) % 2:
            return errors[midpoint]
        return (errors[midpoint - 1] + errors[midpoint]) / 2.0

    # Control: the untrained model at update zero, via a terminal max_steps=0 run.
    initial_train = protocol.training_config()
    initial_train["run"]["root"] = str(tmp_path)
    initial_train["run"]["run_id"] = "train-initial"
    initial_train["trainer"]["max_steps"] = 0
    assert _run_config(initial_train, command="h-v2 pipeline part-e initial checkpoint") == 0
    initial_checkpoint = tmp_path / "tpen_h_v2" / "atomistic" / "train-initial" / "checkpoints" / "step_000000"
    assert (initial_checkpoint / "COMPLETE").is_file()

    # Treatment: the full 1000-update contract-scale training run.
    trained = protocol.training_config()
    trained["run"]["root"] = str(tmp_path)
    trained["run"]["run_id"] = "train-final"
    assert _run_config(trained, command="h-v2 pipeline part-e full training") == 0
    final_checkpoint = tmp_path / "tpen_h_v2" / "atomistic" / "train-final" / "checkpoints" / "step_001000"
    assert (final_checkpoint / "COMPLETE").is_file()

    initial_median_abs_error = _median_abs_error(initial_checkpoint, "eval-initial")
    final_median_abs_error = _median_abs_error(final_checkpoint, "eval-final")

    assert final_median_abs_error <= 0.02
    if initial_median_abs_error > 0.02:
        assert final_median_abs_error <= 0.75 * initial_median_abs_error
