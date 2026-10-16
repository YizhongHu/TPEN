"""Formal R4 source control for the E3 accepted polarity documentation."""

from __future__ import annotations

from pathlib import Path


def test_r3_13a_corrected_docstrings_are_present() -> None:
    root = Path(__file__).parents[3]
    source = (root / "tests/unit/execution/test_invocation_review_r1.py").read_text(
        encoding="utf-8"
    )
    assert "EXPECTED RED" not in source
