"""Tests for across-seed baseline summaries."""

import json
from dataclasses import replace

import pytest

from experiments.baselines.records import BaselineRecord
from experiments.baselines.collect import collect, write_jsonl
from experiments.baselines.spread import DuplicateSeedError, MissingEnergyError, SpreadError, main, summarize


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
    input_path = tmp_path / "results.jsonl"
    input_path.write_text("\n".join(json.dumps(item.to_json_dict()) for item in [record(1, 1.0), record(2, 3.0)]) + "\n")
    output_path = tmp_path / "spread.json"
    markdown_path = tmp_path / "spread.md"
    assert main(["--results", str(input_path), "--output", str(output_path), "--markdown-output", str(markdown_path)]) == 0
    payload = json.loads(output_path.read_text())
    assert payload[0]["mean_hartree"] == 2.0
    assert "sample std" in markdown_path.read_text()


def test_nullable_ansatz_groups_remain_distinct_and_sort_deterministically():
    rows = [
        replace(record(1, energy), ansatz=ansatz)
        for ansatz, energy in [("plain", 3.0), (None, 1.0), ("", 2.0)]
    ]
    forward = summarize(rows)
    backward = summarize(reversed(rows))
    assert [s.to_json_dict() for s in forward] == [s.to_json_dict() for s in backward]
    assert {s.key.ansatz: s.mean_hartree for s in forward} == {
        None: 1.0, "": 2.0, "plain": 3.0,
    }


def test_seedless_run_ids_are_explicit_in_serialized_summary():
    payload = summarize([
        record(2, 3.0, run_id="B"),
        record(None, 100.0, run_id="unknown"),
        record(1, 1.0, run_id="A"),
    ])[0].to_json_dict()
    assert payload["seedless_run_ids"] == ["unknown"]
    assert payload["n_seedless"] == 1
    assert set(payload["run_ids"]) == {"A", "B", "unknown"}
    assert payload["seeds"] == [1, 2]
    assert payload["mean_hartree"] == 2.0


def test_collector_outputs_and_run_root_produce_the_same_spread(tmp_path):
    root = tmp_path / "runs"
    for name, seed, energy in [
        ("a", 1, 1.0), ("nested/b", 2, 2.0),
        ("nested/deeper/c", 3, 4.0), ("unknown", None, 100.0),
    ]:
        directory = root / name
        directory.mkdir(parents=True)
        row = replace(record(seed, energy, run_id=name), system_id="he_atom")
        (directory / "baseline_record.json").write_text(
            json.dumps(row.to_json_dict()), encoding="utf-8"
        )
    variant = replace(record(1, 999.0), system_id="he_atom")
    (root / "a" / "baseline_record.tailfrac0.10.job123.json").write_text(
        json.dumps(variant.to_json_dict()), encoding="utf-8"
    )
    collected = collect(root)
    assert collected.failures == []
    assert len(collected.records) == 4
    results = tmp_path / "results.jsonl"
    write_jsonl(collected.records, results)
    payloads = []
    for option, source, filename in [
        ("--results", results, "from-jsonl.json"),
        ("--run-root", root, "from-root.json"),
    ]:
        output = tmp_path / filename
        assert main([option, str(source), "--output", str(output)]) == 0
        payloads.append(json.loads(output.read_text()))
    assert payloads[0] == payloads[1]
    summary = payloads[0][0]
    assert summary["n_records"] == 4
    assert summary["n_distinct_seeds"] == 3
    assert summary["n_seedless"] == 1
    assert summary["mean_hartree"] == pytest.approx(7 / 3)
    assert [summary[k] for k in ("min_hartree", "max_hartree", "range_hartree")] == [1, 4, 3]
    assert summary["insufficient_seeds"] is False
    assert set(summary["run_ids"]) == {"a", "nested/b", "nested/deeper/c", "unknown"}
    assert summary["seedless_run_ids"] == ["unknown"]
    (root / "a" / "baseline_record.json").write_text("{bad", encoding="utf-8")
    failed_output = tmp_path / "must-not-exist.json"
    with pytest.raises(SpreadError, match="invalid records"):
        main(["--run-root", str(root), "--output", str(failed_output)])
    assert not failed_output.exists()


def test_all_seedless_group_has_null_statistics():
    summary = summarize([
        record(None, 1.0, run_id="a"), record(None, 9.0, run_id="b"),
    ])[0]
    assert (summary.n_records, summary.n_seedless, summary.n_distinct_seeds) == (2, 2, 0)
    assert summary.seeds == ()
    assert summary.seedless_run_ids == ("a", "b")
    assert summary.insufficient_seeds is True
    for field in ("mean_hartree", "min_hartree", "max_hartree", "range_hartree",
                  "sample_std_hartree", "sem_hartree"):
        assert getattr(summary, field) is None


@pytest.mark.parametrize("seed", [1, None])
def test_missing_energy_is_refused_even_if_seedless(seed):
    with pytest.raises(MissingEnergyError, match="missing energy"):
        summarize([record(seed, None)])
