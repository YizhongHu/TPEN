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

import hashlib
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

_LITERAL_ARTIFACT_SHA256 = {
    "h_like_frozen_main": {
        "sampled_eval_table.csv": "3c42647fa2e7802fea68e36d2d89124f308fc29c619f47991dcf78b5edb72988",
        "sampled_eval_table.metadata.json": "7b29dc52aa06b47f255fbf6a497d35e48bc0cade9631f5c6b5d89c36ef1590e9",
    },
    "he_like_frozen_main": {
        "sampled_eval_table.csv": "05f17a98e0e3d3f668484d1dc59da337d0fee589ede27f3a705f54b8290861ee",
        "sampled_eval_table.metadata.json": "edf80b463a229decec776bb7ab88036e28744654e8f3add9d3da3fe4a57befeb",
    },
}


def _fixture_dir(name: str) -> Path:
    directory = _FIXTURES_ROOT / name
    if not directory.is_dir():
        # These fixtures are immutable, committed compatibility evidence
        # (S1-F acceptance contract item 4): a missing fixture is a broken
        # checkout, not an optional extra, so it must fail the suite rather
        # than silently skip it.
        pytest.fail(
            f"fixture {name!r} is not generated in this checkout; run "
            "generate_v1_fixtures.py inside an authorized Cannon allocation "
            "and commit its output (see tests/integration/artifacts/"
            "trajectory_records/generate_v1_fixtures.py)"
        )
    return directory


def test_missing_required_fixture_fails_instead_of_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(__name__ + "._FIXTURES_ROOT", tmp_path)

    def unexpected_skip(*args: object, **kwargs: object) -> None:
        raise AssertionError("missing required fixture was skipped")

    # Turn the mutant's pytest.skip into an ordinary test failure. Pytest
    # treats SkipTest as a skip even when it occurs inside this assertion.
    monkeypatch.setattr(pytest, "skip", unexpected_skip)
    with pytest.raises(pytest.fail.Exception, match="not generated"):
        _fixture_dir("missing_required_fixture")


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


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_literal_csv_pin(name: str, csv_path: Path) -> None:
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == _LITERAL_ARTIFACT_SHA256[name][csv_path.name]


@pytest.mark.parametrize("name", ["h_like_frozen_main", "he_like_frozen_main"])
def test_frozen_fixture_provenance_matches_committed_bytes(name: str) -> None:
    directory = _fixture_dir(name)
    provenance = _provenance(directory)
    metadata_path = directory / "sampled_eval_table.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert provenance["generating_commit"], "fixture provenance must pin a generating commit"
    assert provenance["csv_sha256"] == metadata["csv_sha256"]
    assert provenance["byte_count"] == metadata["byte_count"]

    # Independent CSV + metadata digest pins (not just the csv_sha256 the
    # metadata sidecar already carries about itself): a joint refresh of
    # CSV/metadata/provenance cannot silently replace the intended
    # compatibility oracle without also moving this independently-recorded
    # metadata hash.
    assert _is_sha256_hex(provenance.get("metadata_sha256"))
    assert provenance["metadata_sha256"] == hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    csv_path = directory / "sampled_eval_table.csv"
    assert provenance["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    # Exercise the literal pin with an intentionally wrong expected digest as
    # well as the real fixture. This makes the test fail if the assertion is
    # weakened to ``assert True`` while leaving committed bytes untouched.
    literal_key = _LITERAL_ARTIFACT_SHA256[name][csv_path.name]
    _LITERAL_ARTIFACT_SHA256[name][csv_path.name] = "0" * 64
    try:
        with pytest.raises(AssertionError):
            _assert_literal_csv_pin(name, csv_path)
    finally:
        _LITERAL_ARTIFACT_SHA256[name][csv_path.name] = literal_key
    assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == _LITERAL_ARTIFACT_SHA256[name][metadata_path.name]

    # Reproducible generator/input pinning: source, exact command, and a
    # config/environment hash, so a replay has everything needed to
    # reproduce this fixture rather than trusting a bare commit string.
    assert _is_sha256_hex(provenance.get("generator_source_sha256"))
    assert provenance.get("generation_command")
    assert _is_sha256_hex(provenance.get("config_hash"))

    # Frozen-writer evidence: the writer classes actually executed by the
    # generator (hashed via `inspect.getsource` on the live objects, not an
    # inferred git blob) must match the same classes' source at the fixed,
    # named frozen-writer commit -- proving these fixtures came from the
    # frozen v1 writer and not from a modified copy.
    assert provenance["frozen_writer_commit"]
    assert _is_sha256_hex(provenance.get("executed_writer_source_sha256"))
    assert _is_sha256_hex(provenance.get("frozen_commit_writer_source_sha256"))
    assert provenance["writer_matches_frozen_commit"] is True
    assert provenance["executed_writer_source_sha256"] == provenance["frozen_commit_writer_source_sha256"]


def test_frozen_fixture_literal_hashes_are_independent_pins() -> None:
    for name, expected in _LITERAL_ARTIFACT_SHA256.items():
        directory = _fixture_dir(name)
        for filename, digest in expected.items():
            assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == digest


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
