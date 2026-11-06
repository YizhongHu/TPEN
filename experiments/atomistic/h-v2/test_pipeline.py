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
    protocol.assert_hydrogen_contract(train)
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
    protocol.assert_hydrogen_contract(evaluation, evaluation=True)
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


def test_training_state_restore_is_refused_by_the_real_evaluate_runner(tmp_path: Path) -> None:
    train = protocol.training_config()
    train["run"]["root"] = str(tmp_path)
    train["run"]["run_id"] = "runner-train"
    train["trainer"]["max_steps"] = 1
    train["sampler"].update(n_walkers=8, burn_in=2, n_steps=2)
    train["callbacks"][-1]["schedule"]["every_n"] = 1
    assert _run_config(train, command="h-v2 runner refusal train") == 0
    checkpoint = tmp_path / "tpen_h_v2" / "atomistic" / "runner-train" / "checkpoints" / "step_000001"
    evaluation = protocol.evaluation_config(checkpoint, root=tmp_path)
    evaluation["run"]["run_id"] = "runner-refusal"
    evaluation["sampler"].update(n_walkers=8, burn_in=2, n_steps=2)
    evaluation["load"]["mode"] = "train_resume"
    with pytest.raises(ValueError, match="train_resume"):
        _run_config(evaluation, command="h-v2 runner refusal evaluation")


@pytest.mark.h_v2_pipeline
def test_part_e_contract_scale_training_and_evaluation_energy_criteria(tmp_path: Path, record_property) -> None:
    """Run the h-v2 short learned route at full contract scale (part e).

    Contract (``contract_inputs.json`` ``admission.pipeline_check`` part e):
    8 channels, 32 hidden, 1000 updates; training sampler 256 walkers,
    burn-in 100, 10 Metropolis steps/update, proposal scale 0.5 bohr; fresh
    evaluation 1024 walkers x 64 draws x 2 streams, burn-in 1000, stride 20.
    Computes the ``0.02`` Ha energy-error criterion (the absolute value of the
    pooled mean local energy plus ``0.5`` Ha, not a median) and the ``0.25``
    median-absolute-error improvement rule as a separate criterion, with the
    little-dynamic-range branch reported explicitly (via the
    ``*_improvement_rule_evaluated`` junit property) when the untrained
    (update-zero) model is already within ``0.02`` Ha. In that branch the
    contract requires only the imperfect learned-state/gradient control.
    `test_stub_oracle_contract.test_stub_imperfect_state_is_not_reported_exact`
    (``alpha=1.2``) is an oracle self-consistency control, not a learned-state
    control: it proves the independent analytic instrument rejects an
    imperfect exact-form stub, not that this pipeline's trained model is
    imperfect. The imperfect learned-state/gradient control for this branch,
    when it applies, is this test's own measured numbers (recorded below), not
    that oracle check.

    Marked ``h_v2_pipeline`` for selection, not exclusion: this repo's pytest
    markers (see also ``integration``) are categorical only -- nothing here
    filters them out of the standard ``pytest -q experiments/atomistic/h-v2
    experiments/atomistic/he-v1`` invocation. This test is instead skipped by
    default via `conftest.py` unless ``--run-h-v2-pipeline`` is passed, because
    at contract scale it alone adds several minutes on top of the rest of the
    suite (well inside the contract's 2h/4-core CPU budget, but expensive to
    pay once per mutant during verification). Run it explicitly with::

        uv run pytest -q --run-h-v2-pipeline -m h_v2_pipeline experiments/atomistic/h-v2/test_pipeline.py
    """

    def _median_abs_error(checkpoint: Path, run_id: str) -> tuple[float, float]:
        errors: list[float] = []
        energies_all: list[float] = []
        for stream in (0, 1):
            evaluation = protocol.evaluation_config(checkpoint, stream=stream, root=tmp_path)
            evaluation["run"]["run_id"] = f"{run_id}-stream{stream}"
            assert _run_config(evaluation, command=f"h-v2 pipeline part-e {run_id} stream {stream}") == 0
            eval_dir = tmp_path / "tpen_h_v2" / "atomistic" / f"{run_id}-stream{stream}" / "trajectory"
            with (eval_dir / "sampled_eval_table.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            assert rows
            assert len(rows) == 1024 * 64
            assert {(int(row["draw_index"]), int(row["walker_index"])) for row in rows} == {
                (draw, walker) for draw in range(64) for walker in range(1024)
            }
            assert all(row["finite"] == "True" for row in rows)
            energies = [float(row["local_energy"]) for row in rows]
            stream_errors = [abs(value + 0.5) for value in energies]
            errors.extend(stream_errors)
            energies_all.extend(energies)
            stream_mean = sum(energies) / len(energies)
            sorted_stream_errors = sorted(stream_errors)
            stream_midpoint = len(sorted_stream_errors) // 2
            stream_median = (
                sorted_stream_errors[stream_midpoint]
                if len(sorted_stream_errors) % 2
                else (sorted_stream_errors[stream_midpoint - 1] + sorted_stream_errors[stream_midpoint]) / 2.0
            )
            record_property(f"{run_id}_stream{stream}_mean_local_energy", stream_mean)
            record_property(f"{run_id}_stream{stream}_median_abs_error", stream_median)
            print(f"{run_id} stream {stream}: mean={stream_mean} median_abs_error={stream_median}")
        errors.sort()
        midpoint = len(errors) // 2
        median = errors[midpoint] if len(errors) % 2 else (errors[midpoint - 1] + errors[midpoint]) / 2.0
        pooled_mean = sum(energies_all) / len(energies_all)
        energy_abs_error = abs(pooled_mean + 0.5)
        record_property(f"{run_id}_pooled_mean_local_energy", pooled_mean)
        record_property(f"{run_id}_pooled_median_abs_error", median)
        record_property(f"{run_id}_pooled_energy_abs_error", energy_abs_error)
        print(f"{run_id} pooled: mean={pooled_mean} median_abs_error={median} energy_abs_error={energy_abs_error}")
        return median, energy_abs_error

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

    initial_median_abs_error, initial_energy_abs_error = _median_abs_error(initial_checkpoint, "eval-initial")
    final_median_abs_error, energy_abs_error_final = _median_abs_error(final_checkpoint, "eval-final")
    improvement_rule_evaluated = initial_median_abs_error > 0.02
    improvement_ratio = final_median_abs_error / initial_median_abs_error
    for key, value in {
        "initial_median_abs_error": initial_median_abs_error,
        "final_median_abs_error": final_median_abs_error,
        "initial_energy_abs_error": initial_energy_abs_error,
        "energy_abs_error_final": energy_abs_error_final,
        "improvement_rule_evaluated": improvement_rule_evaluated,
        "improvement_ratio": improvement_ratio,
    }.items():
        record_property(key, value)
        print(f"{key}={value}")

    # Contract criterion (a): the final energy error, |mean(E_L) + 0.5|, not a
    # median (finding 1 of the round-1 review).
    assert energy_abs_error_final <= 0.02
    # Contract criterion (b), kept separate from (a): the median-absolute-error
    # improvement rule, evaluated only when the untrained control's error
    # exceeds the 0.02 Ha threshold (the little-dynamic-range branch above).
    if improvement_rule_evaluated:
        assert final_median_abs_error <= 0.75 * initial_median_abs_error
