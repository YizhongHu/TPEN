"""Collector tests use tiny synthetic tables and never invoke TPEN core."""

import csv
import hashlib
import json

import pytest

from experiments.atomistic.stage1.collect import CollectionError, collect_table, reference_energy


def test_missing_reference_key_is_an_explicit_error() -> None:
    with pytest.raises(CollectionError, match="no required key"):
        reference_energy("hooke1", {"systems": []})


def test_incomplete_grid_is_rejected(tmp_path) -> None:
    path = tmp_path / "sampled_eval_table.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["draw_index", "walker_index", "local_energy", "finite"])
        writer.writeheader()
        writer.writerow({"draw_index": 0, "walker_index": 0, "local_energy": 1.0, "finite": "True"})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "sampled_eval_table.metadata.json").write_text(json.dumps({
        "schema": "trajectory_records/v1",
        "row_semantics": "complete_draw_by_walker_grid",
        "walker_count": 1,
        "draw_count": 2,
        "row_count": 2,
        "nonfinite_count": 0,
        "csv_sha256": digest,
    }), encoding="utf-8")
    with pytest.raises(CollectionError, match="incomplete"):
        collect_table(path, system="he", registry={"systems": [{"id": "he_atom", "reference_energy_hartree": -2.9}]},
                     walkers=1, draws=2)


def test_real_table_reports_short_draw_floor(tmp_path) -> None:
    path = tmp_path / "sampled_eval_table.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["draw_index", "walker_index", "local_energy", "finite"])
        writer.writeheader()
        for draw in range(2):
            writer.writerow({"draw_index": draw, "walker_index": 0, "local_energy": 1.0 + draw, "finite": "True"})
    (tmp_path / "sampled_eval_table.metadata.json").write_text(json.dumps({
        "schema": "trajectory_records/v1", "row_semantics": "complete_draw_by_walker_grid",
        "walker_count": 1, "draw_count": 2, "row_count": 2, "nonfinite_count": 0,
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    result = collect_table(path, system="he", registry={"systems": [{"id": "he_atom", "reference_energy_hartree": -2.9}]},
                           walkers=1, draws=2)
    assert result["mcse"] is None
    assert result["mcse_status"] == "insufficient_draws_for_blocking_floor_32"
