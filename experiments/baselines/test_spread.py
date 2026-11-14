"""Tests for across-seed baseline summaries."""

import json

import pytest

from experiments.baselines.records import BaselineRecord
from experiments.baselines.spread import DuplicateSeedError, summarize


def record(seed, energy, *, estimator="inference", run_id=None):
    """Build a small valid hand-written record."""

    return BaselineRecord(
        system_id="toy", code="demo", ansatz="plain", estimator=estimator,
        seed=seed, energy_hartree=energy, energy_stderr_hartree=0.01,
        run_id=run_id or f"run-{seed}",
    )


def test_known_mean_sample_std_and_sem_use_seed_values_only():
    summary = summarize([record(1, 1.0), record(2, 2.0), record(3, 4.0), record(None, 100.0)])[0]
    assert summary.mean_hartree == pytest.approx(7 / 3)
    # A ddof=0 implementation returns sqrt(14/9), not the required sqrt(7/2).
    assert summary.sample_std_hartree == pytest.approx((7 / 3) ** 0.5)
    assert summary.sem_hartree == pytest.approx((7 / 9) ** 0.5)
    assert summary.n_records == 4
    assert summary.n_seedless == 1
    assert summary.seeds == (1, 2, 3)


def test_two_seeds_are_reported_but_flagged_insufficient():
    summary = summarize([record(10, -1.0), record(11, -2.0)])[0]
    assert summary.insufficient_seeds is True
    assert summary.sample_std_hartree == pytest.approx(2**0.5 / 2)


def test_estimator_is_in_group_key_instead_of_pooling():
    summaries = summarize([record(1, 1.0), record(2, 2.0, estimator="training_tail")])
    assert [item.key.estimator for item in summaries] == ["inference", "training_tail"]
    assert [item.mean_hartree for item in summaries] == [1.0, 2.0]


def test_duplicate_seed_is_loud():
    with pytest.raises(DuplicateSeedError, match="duplicate seed 1"):
        summarize([record(1, 1.0, run_id="a"), record(1, 1.1, run_id="b")])


def test_cli_round_trip(tmp_path):
    from experiments.baselines.spread import main

    input_path = tmp_path / "results.jsonl"
    input_path.write_text("\n".join(json.dumps(item.to_json_dict()) for item in [record(1, 1.0), record(2, 3.0)]) + "\n")
    output_path = tmp_path / "spread.json"
    markdown_path = tmp_path / "spread.md"
    assert main(["--results", str(input_path), "--output", str(output_path), "--markdown-output", str(markdown_path)]) == 0
    payload = json.loads(output_path.read_text())
    assert payload[0]["mean_hartree"] == 2.0
    assert "sample std" in markdown_path.read_text()
