"""Core producer/reader parity for the canonical executor-boundary fixture."""

from __future__ import annotations

import json
from pathlib import Path

from tpen.inspection import RunEvidence

from ._regen_fixtures import FIXTURE_PATH, regenerate


def test_real_core_producer_regenerates_canonical_v2_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert regenerate() == fixture


def test_core_reader_decodes_canonical_v2_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    evidence = RunEvidence.from_mapping(fixture)
    assert evidence.to_dict() == fixture
