"""Core producer/reader parity for the canonical executor-boundary fixture."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from tpen.inspection import RunEvidence, inspect_run

from ._regen_fixtures import (
    FIXTURE_PATH,
    REFUSED_FIXTURE_PATH,
    _make_run,
    normalize,
    regenerate,
    regenerate_refused,
)


def test_real_core_producer_regenerates_canonical_v2_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert regenerate() == fixture


def test_core_reader_decodes_canonical_v2_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    evidence = RunEvidence.from_mapping(fixture)
    assert evidence.to_dict() == fixture


def test_normalization_rejects_outside_and_preserves_wrong_step_paths() -> None:
    with tempfile.TemporaryDirectory(prefix="cx00-path-control-") as temporary:
        temporary_path = Path(temporary)
        evidence = inspect_run(_make_run(temporary_path))
        temporary_root = temporary_path / "attempt"
        outside = replace(
            evidence,
            checkpoint={
                **evidence.checkpoint,
                "checkpoint_dir": str(temporary_path / "outside" / "step_000007"),
            },
        )
        with pytest.raises(ValueError, match="outside temporary root"):
            normalize(outside, temporary_root)

        wrong_step = replace(
            evidence,
            checkpoint={
                **evidence.checkpoint,
                "checkpoint_dir": str(
                    temporary_path / "attempt" / "core" / "checkpoints" / "step_000008"
                ),
            },
        )
        assert normalize(wrong_step, temporary_root) != json.loads(
            FIXTURE_PATH.read_text(encoding="utf-8")
        )


def test_refused_v1_fixture_regenerates_and_decodes() -> None:
    fixture = json.loads(REFUSED_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert regenerate_refused() == fixture
    assert RunEvidence.from_mapping(fixture).to_dict() == fixture
