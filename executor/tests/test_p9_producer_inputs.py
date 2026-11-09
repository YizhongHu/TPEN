"""Executor-side golden checks for P9 producer wire records."""

from __future__ import annotations

from pathlib import Path

from tpen_executor._codec import canonical_bytes
from tpen_executor.contracts import ProgramSpecRef, ReadyGroup, TaskSpec


FIXTURES = Path(__file__).parent / "fixtures" / "p9"


def test_p9_golden_groups_decode_and_reencode_byte_identically() -> None:
    for path in sorted(FIXTURES.glob("*.json")):
        raw = path.read_bytes().rstrip(b"\n")
        group = ReadyGroup.decode(raw)
        assert canonical_bytes(group.to_dict()) == raw
        for member in group.members:
            assert TaskSpec.decode(member.encode()) == member
            assert ProgramSpecRef.decode(member.program.encode()) == member.program
