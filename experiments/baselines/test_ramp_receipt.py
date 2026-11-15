"""Offline controls for the BL-R1 ramp receipt gate."""

from __future__ import annotations

import pytest

from experiments.baselines.runbooks.ramp_receipt import gate, record


def test_gate_requires_all_named_successes(tmp_path):
    """A failed or absent row cannot yield a successful ramp account."""

    record(tmp_path / "a" / "receipt.json", row="a", returncode=0)
    record(tmp_path / "b" / "receipt.json", row="b", returncode=7)
    with pytest.raises(RuntimeError, match="returncode"):
        gate(tmp_path, ["a", "b", "c"])


def test_gate_accounts_for_every_successful_row(tmp_path):
    """The sole success shape includes all and only the requested rows."""

    record(tmp_path / "a" / "receipt.json", row="a", returncode=0)
    record(tmp_path / "b" / "receipt.json", row="b", returncode=0)
    assert [row["row"] for row in gate(tmp_path, ["a", "b"])["rows"]] == ["a", "b"]


def test_record_refuses_an_existing_receipt(tmp_path):
    """A prior attempt is preserved instead of being overwritten."""

    path = tmp_path / "a" / "receipt.json"
    record(path, row="a", returncode=0)
    with pytest.raises(FileExistsError):
        record(path, row="a", returncode=0)
