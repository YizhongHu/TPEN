"""Collector tests use tiny synthetic tables and never invoke TPEN core."""

import csv

import pytest

from experiments.atomistic.stage1.collect import CollectionError, collect_table, reference_energy


def test_missing_reference_key_is_an_explicit_error() -> None:
    with pytest.raises(CollectionError, match="no required key"):
        reference_energy("hooke1", {"systems": []})


def test_incomplete_grid_is_rejected(tmp_path) -> None:
    path = tmp_path / "table.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["walker", "draw", "local_energy"])
        writer.writeheader()
        writer.writerow({"walker": 0, "draw": 0, "local_energy": 1.0})
    with pytest.raises(CollectionError, match="incomplete"):
        collect_table(path, system="he", registry={"systems": [{"id": "he_atom", "reference_energy_hartree": -2.9}]},
                     walkers=1, draws=2)
