"""Integrity controls for the complete H-v2 collector."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("h_v2_collect", HERE / "collect.py")
assert spec is not None and spec.loader is not None
collect = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = collect
spec.loader.exec_module(collect)


def _fixture(tmp_path: Path, *, rows: int = 6) -> tuple[Path, Path, collect.RecordSpec]:
    spec = collect.RecordSpec("source", "checkpoint", "config", walkers=2, draws=3)
    manifest = {
        "schema": spec.schema,
        "source_sha": spec.source_sha,
        "checkpoint_sha": spec.checkpoint_sha,
        "config_sha": spec.config_sha,
        "commit_marker": "committed",
        "row_count": spec.expected_rows,
        "walkers": spec.walkers,
        "draws": spec.draws,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    record_path = tmp_path / "records.csv"
    with record_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "source_sha", "checkpoint_sha", "config_sha", "commit_marker",
            "draw_index", "walker_index", "local_energy", "position_x", "position_y", "position_z",
        ])
        writer.writeheader()
        for index in range(rows):
            writer.writerow({
                "source_sha": spec.source_sha,
                "checkpoint_sha": spec.checkpoint_sha,
                "config_sha": spec.config_sha,
                "commit_marker": "committed",
                "draw_index": index // spec.walkers,
                "walker_index": index % spec.walkers,
                "local_energy": -0.5,
                "position_x": 1.0,
                "position_y": 0.0,
                "position_z": 0.0,
            })
    return record_path, manifest_path, spec


def test_complete_records_are_collected(tmp_path: Path) -> None:
    record, manifest, spec = _fixture(tmp_path)
    receipt = collect.collect_records(record, manifest, spec)
    assert receipt["status"] == "complete"
    assert receipt["row_count"] == 6


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows.pop(), "missing draw"),
    (lambda rows: rows.__setitem__(1, rows[0]), "duplicate"),
])
def test_missing_or_duplicate_rows_refuse(mutation, message, tmp_path: Path) -> None:
    record, manifest, spec = _fixture(tmp_path)
    with record.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mutation(rows)
    with pytest.raises(collect.RecordIntegrityError, match=message):
        collect.validate_rows(rows, spec, json.loads(manifest.read_text()))


@pytest.mark.parametrize("field, value, message", [
    ("source_sha", "other", "source_sha"),
    ("checkpoint_sha", "other", "checkpoint_sha"),
    ("config_sha", "other", "config_sha"),
    ("commit_marker", "pending", "commit marker"),
    ("local_energy", "nan", "nonfinite"),
])
def test_identity_nonfinite_and_commit_mutations_refuse(field, value, message, tmp_path: Path) -> None:
    record, manifest, spec = _fixture(tmp_path)
    with record.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = value
    with pytest.raises(collect.RecordIntegrityError, match=message):
        collect.validate_rows(rows, spec, json.loads(manifest.read_text()))


def test_training_state_restore_is_not_an_accepted_record_identity(tmp_path: Path) -> None:
    record, manifest, spec = _fixture(tmp_path)
    data = json.loads(manifest.read_text())
    data["restore_mode"] = "train_resume"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="model-only"):
        collect.validate_manifest(data, spec)
