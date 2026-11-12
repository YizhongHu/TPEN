"""Compatibility contracts for the immutable v1 trajectory-record fixtures.

Covers the checked-in ``h_like_frozen_main`` and ``he_like_frozen_main``
fixtures under ``tests/integration/artifacts/trajectory_records/`` (written
once by ``generate_v1_fixtures.py`` with the frozen ``trajectory_records/v1``
writer, never by the reader under test). These prove the new historical
reader decodes genuine, provenance-pinned writer output -- not a format the
reader itself invented -- and that its bytes/values remain accepted unchanged
(S1-F acceptance contract items 1 and 4).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen.evaluation.trajectory_records import (
    TRAJECTORY_RECORD_SCHEMA,
    inspect_trajectory_record,
)

_FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "integration" / "artifacts" / "trajectory_records"

# Deterministic grid shared by both fixtures: row (draw, walker) carries local
# energy draw*10 + walker + 1, mirroring generate_v1_fixtures.py exactly.
_N_DRAWS = 2
_N_WALKERS = 3
_VALUES = [draw * 10 + walker + 1 for draw in range(_N_DRAWS) for walker in range(_N_WALKERS)]
_EXPECTED_ROW_COUNT = _N_DRAWS * _N_WALKERS
_EXPECTED_MEAN = sum(_VALUES) / _EXPECTED_ROW_COUNT
_EXPECTED_VARIANCE = sum(value * value for value in _VALUES) / _EXPECTED_ROW_COUNT - _EXPECTED_MEAN**2


def _fixture_dir(name: str) -> Path:
    directory = _FIXTURES_ROOT / name
    if not directory.is_dir():
        pytest.skip(
            f"fixture {name!r} is not generated in this checkout; run "
            "generate_v1_fixtures.py inside an authorized Cannon allocation "
            "and commit its output (see tests/integration/artifacts/"
            "trajectory_records/generate_v1_fixtures.py)"
        )
    return directory


def _provenance(directory: Path) -> dict:
    return json.loads((directory / "PROVENANCE.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["h_like_frozen_main", "he_like_frozen_main"])
def test_frozen_fixture_decodes_with_original_values(name: str) -> None:
    directory = _fixture_dir(name)
    metadata_path = directory / "sampled_eval_table.metadata.json"
    result = inspect_trajectory_record(metadata_path)

    assert result.record_schema == TRAJECTORY_RECORD_SCHEMA
    assert result.draw_count == _N_DRAWS
    assert result.walker_count == _N_WALKERS
    assert result.row_count == _EXPECTED_ROW_COUNT
    assert result.mean == pytest.approx(_EXPECTED_MEAN)
    assert result.variance == pytest.approx(_EXPECTED_VARIANCE)
    assert result.physical_context.available is False
    assert result.physical_context.system_id is None
    assert result.physical_context.reason


@pytest.mark.parametrize("name", ["h_like_frozen_main", "he_like_frozen_main"])
def test_frozen_fixture_provenance_matches_committed_bytes(name: str) -> None:
    directory = _fixture_dir(name)
    provenance = _provenance(directory)
    metadata_path = directory / "sampled_eval_table.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert provenance["generating_commit"], "fixture provenance must pin a generating commit"
    assert provenance["writer_module_blob_sha1"], "fixture provenance must pin the writer module blob"
    assert provenance["csv_sha256"] == metadata["csv_sha256"]
    assert provenance["byte_count"] == metadata["byte_count"]
    # Frozen-writer evidence: the pinned writer blob must either be proven
    # byte-identical to origin/main's writer blob at generation time, or the
    # comparison must be explicitly recorded as unavailable (never silently
    # dropped).
    assert "writer_blob_matches_origin_main" in provenance


@pytest.mark.parametrize(
    ("name", "expected_n_electrons", "expected_charge"),
    [("h_like_frozen_main", 1, 1.0), ("he_like_frozen_main", 2, 2.0)],
)
def test_frozen_fixture_matches_its_declared_system_shape(
    name: str, expected_n_electrons: int, expected_charge: float
) -> None:
    directory = _fixture_dir(name)
    provenance = _provenance(directory)
    metadata_path = directory / "sampled_eval_table.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert provenance["n_electrons"] == expected_n_electrons
    assert provenance["nuclear_charge"] == expected_charge
    assert metadata["n_electrons"] == expected_n_electrons
    assert metadata["atomic_configuration"]["charges"] == [expected_charge]
