"""Verify the byte-identical executor harness import and its provenance table."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_ROOT = REPO_ROOT / "scripts" / "cannon" / "executor-harness"
PROVENANCE = HARNESS_ROOT / "PROVENANCE.md"
SHA_ROW = re.compile(r"^(?P<sha>[0-9a-f]{64})  (?P<path>.+)$")


def provenance_table() -> dict[str, str]:
    return {
        match.group("path"): match.group("sha")
        for line in PROVENANCE.read_text().splitlines()
        if (match := SHA_ROW.fullmatch(line))
    }


def test_imported_file_set_and_hashes_match_provenance() -> None:
    expected = provenance_table()
    actual = {
        path.relative_to(HARNESS_ROOT).as_posix()
        for path in HARNESS_ROOT.rglob("*")
        if path.is_file() and path.name != PROVENANCE.name
    }

    assert set(expected) == actual
    assert len(actual) == 12
    for relative_path, expected_sha in expected.items():
        path = HARNESS_ROOT / relative_path
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, relative_path
