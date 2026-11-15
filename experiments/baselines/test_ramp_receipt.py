"""Offline controls for the BL-R1 receipt, trace, and inventory mechanics."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.baselines.runbooks.ramp_receipt import (
    gate,
    inventory,
    record,
    validate_ferminet_trace,
)


def test_pbs_layout_records_and_gates_existing_row_directories(tmp_path):
    """Use the exact row/receipt arguments used by the PBS template."""

    root = tmp_path / "ramp"
    root.mkdir()
    rows = ["deepqmc-default-seed0", "deepqmc-ferminet-seed0"]
    for row in rows:
        (root / row).mkdir()
        record(root / row / "receipt.json", row=row, returncode=0)
    assert [entry["row"] for entry in gate(root, rows)["rows"]] == rows


def test_pbs_layout_rejects_missing_and_failed_receipts(tmp_path):
    """A pre-existing row alone, or a failed child, never opens the next wave."""

    root = tmp_path / "ramp"
    root.mkdir()
    (root / "missing").mkdir()
    (root / "failed").mkdir()
    record(root / "failed" / "receipt.json", row="failed", returncode=7)
    with pytest.raises(RuntimeError, match="missing"):
        gate(root, ["missing", "failed"])
    with pytest.raises(RuntimeError, match="returncode"):
        gate(root, ["failed"])


def test_record_refuses_an_existing_receipt_and_missing_row(tmp_path):
    """A prior attempt is preserved and PBS must establish each row directory."""

    path = tmp_path / "a" / "receipt.json"
    with pytest.raises(FileNotFoundError):
        record(path, row="a", returncode=0)
    path.parent.mkdir()
    record(path, row="a", returncode=0)
    with pytest.raises(FileExistsError):
        record(path, row="a", returncode=0)


def _write_trace(path, rows, fieldnames=("step", "energy", "ewmean", "ewvar", "pmove")):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_ferminet_trace_requires_pinned_schema_and_complete_values(tmp_path):
    """The producer schema permits only complete finite measurements."""

    path = tmp_path / "train_stats.csv"
    rows = [
        {"step": index, "energy": -0.5, "ewmean": -0.5, "ewvar": 0.1, "pmove": 0.5}
        for index in range(3)
    ]
    _write_trace(path, rows)
    assert validate_ferminet_trace(path, steps=3)["steps"] == 3
    _write_trace(path, rows, fieldnames=("step", "energy"))
    with pytest.raises(ValueError, match="schema"):
        validate_ferminet_trace(path, steps=3)
    _write_trace(path, [{**row, "energy": ""} if row["step"] == 1 else row for row in rows])
    with pytest.raises(ValueError, match="missing energy"):
        validate_ferminet_trace(path, steps=3)
    _write_trace(path, rows[:-1] + [{"step": 2, "energy": -0.5}])
    with pytest.raises(ValueError, match="missing ewmean"):
        validate_ferminet_trace(path, steps=3)


def test_inventory_preserves_partial_account_before_failure(tmp_path):
    """Missing required artifacts are recorded rather than silently omitted."""

    (tmp_path / "present.txt").write_text("evidence", encoding="utf-8")
    output = tmp_path / "inventory.jsonl"
    with pytest.raises(RuntimeError, match="missing.txt"):
        inventory(tmp_path, output, ["present.txt", "missing.txt"])
    assert '"status": "missing"' in output.read_text(encoding="utf-8")


def test_verify_submitter_has_complete_exported_control_surface():
    """Verification submission is a whole script, not a qsub splice fragment."""

    script = (Path(__file__).parent / "runbooks" / "h-atom-r1-verify-submit.sh").read_text(encoding="utf-8")
    assert 'success-mini) INJECT_FAIL=' in script
    assert 'fail-deepqmc-default) INJECT_FAIL=deepqmc-default-seed0' in script
    assert 'export BL_R1_MODE=verify BL_R1_VERIFY_STEPS=10' in script
    assert 'QSUB_VARIABLES=RUN_ROOT,SOURCE_CHECKOUT,RUNBOOK_COMMIT,SCIENCE,PRE_FREEZE_RECEIPT,PRE_FREEZE_RECEIPT_SHA256,BL_R1_MODE,BL_R1_VERIFY_STEPS' in script
    assert 'BL_R1_VERIFY_INJECT_FAIL' in script
    assert '--dry-run' in script
