"""Round-2 collector and conftest probes for the H-v2 review."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
test_collect_spec = importlib.util.spec_from_file_location(
    "h_v2_test_collect_review_r2", HERE / "test_collect.py"
)
assert test_collect_spec is not None and test_collect_spec.loader is not None
test_collect = importlib.util.module_from_spec(test_collect_spec)
sys.modules[test_collect_spec.name] = test_collect
test_collect_spec.loader.exec_module(test_collect)
collect = test_collect.collect


def _manifest(paths: dict[str, Path]) -> dict[str, object]:
    return json.loads(paths["metadata_path"].read_text(encoding="utf-8"))


def _append_well_formed_row(csv_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    assert fieldnames is not None
    row = {name: "" for name in fieldnames}
    row.update(
        {
            "sample_index": "6",
            "draw_index": "3",
            "walker_index": "0",
            "local_energy": "-0.5",
            "term/kinetic": "0.5",
            "term/electron_nucleus": "-1.0",
            "logabs": "-1.0",
            "sign": "1.0",
            "finite": "True",
            "position/electron_0/axis_0": "1.0",
            "position/electron_0/axis_1": "0.0",
            "position/electron_0/axis_2": "0.0",
        }
    )
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)


def _tamper_local_energy(csv_path: Path) -> None:
    text = csv_path.read_text(encoding="utf-8")
    assert text.count("-0.5") >= 1
    csv_path.write_text(text.replace("-0.5", "-0.6", 1), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("row_semantics", "complete draw-by-walker"),
        ("walker_count", "record shape"),
        ("draw_count", "record shape"),
        ("row_count", "not complete"),
        ("nonfinite_count", "nonfinite"),
        ("missing_nonfinite_count", "nonfinite"),
        ("missing_csv_sha256", "uncommitted tail"),
        ("append_csv_row", "uncommitted tail"),
        ("tamper_csv_byte", "uncommitted tail"),
    ],
)
def test_manifest_refusals_each_fire(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    paths, spec = test_collect._full_fixture(tmp_path)
    manifest = _manifest(paths)
    if mutation == "row_semantics":
        manifest[mutation] = "other"
    elif mutation in {"walker_count", "draw_count"}:
        manifest[mutation] = int(manifest[mutation]) + 1
    elif mutation == "row_count":
        manifest[mutation] = int(manifest[mutation]) - 1
    elif mutation == "nonfinite_count":
        manifest[mutation] = 1
    elif mutation == "missing_nonfinite_count":
        del manifest["nonfinite_count"]
    elif mutation == "missing_csv_sha256":
        del manifest["csv_sha256"]
    elif mutation == "append_csv_row":
        _append_well_formed_row(paths["csv_path"])
    elif mutation == "tamper_csv_byte":
        _tamper_local_energy(paths["csv_path"])
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")
    with pytest.raises(collect.RecordIntegrityError, match=expected):
        collect.validate_manifest(manifest, paths["csv_path"], spec)


def test_nonfinite_coordinate_is_refused(tmp_path: Path, record_property) -> None:
    paths, spec = test_collect._full_fixture(tmp_path)
    with paths["csv_path"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["position/electron_0/axis_0"] = "nan"
    try:
        collect.validate_rows(rows, spec)
    except collect.RecordIntegrityError as exc:
        record_property("outcome", f"raised: {exc}")
        assert "nonfinite" in str(exc)
    else:
        record_property("outcome", "NO ERROR: nonfinite coordinate accepted")
        pytest.fail("validate_rows accepted a nonfinite coordinate")
