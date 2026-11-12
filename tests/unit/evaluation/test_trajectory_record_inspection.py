"""Contracts for the version-aware historical trajectory-record reader.

Covers ``tpen.evaluation.trajectory_records.inspect_trajectory_record`` and
its re-export through the public ``tpen.inspection`` boundary (see
``experiments/README.md`` and
``system-context-records-design-r1.md`` section 5, "Historical readability").
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

import tpen.inspection as inspection
from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.evaluation import trajectory_records
from tpen.evaluation.trajectory_records import (
    TRAJECTORY_RECORD_INSPECTION_SCHEMA,
    TRAJECTORY_RECORD_METADATA_KEYS,
    TRAJECTORY_RECORD_SCHEMA,
    TamperedTrajectoryRecordError,
    TrajectoryPhysicalContext,
    TrajectoryRecordArtifact,
    TrajectoryRecordBatch,
    TrajectoryRecordStreamWriter,
    UnsupportedTrajectoryRecordSchemaError,
    inspect_trajectory_record,
)
from tpen.statistics.trajectory import ObservableTrajectory


def _write_v1_artifact(
    tmp_path: Path,
    values: torch.Tensor,
    *,
    n_electrons: int = 1,
    n_nuclei: int = 1,
    name: str = "trajectory.csv",
) -> TrajectoryRecordArtifact:
    values = values.to(torch.float64)
    n_draws, n_walkers = values.shape
    positions = torch.zeros(n_draws, n_walkers, n_electrons, 3, dtype=torch.float64)
    atoms = AtomicConfiguration(
        positions=torch.zeros(n_nuclei, 3, dtype=torch.float64),
        charges=torch.full((n_nuclei,), float(n_nuclei), dtype=torch.float64),
    )

    def record(draw: int) -> TrajectoryRecordBatch:
        row_values = values[draw]
        return TrajectoryRecordBatch(
            draw_index=torch.full((n_walkers,), draw, dtype=torch.int64),
            walker_index=torch.arange(n_walkers, dtype=torch.int64),
            positions=positions[draw],
            local_energy=row_values,
            term_energies={"kinetic": row_values},
            logabs=torch.full_like(row_values, -2.0),
            sign=torch.ones(n_walkers, dtype=torch.float64),
            finite_mask=torch.isfinite(row_values),
        )

    writer = TrajectoryRecordStreamWriter(
        tmp_path / name,
        observable="local_energy",
        n_draws=n_draws,
        n_walkers=n_walkers,
        term_names=("kinetic",),
        atomic_configuration=atoms,
        first_draw=record(0),
    )
    for draw in range(1, n_draws):
        writer.append(record(draw))
    return writer.finalize(
        ObservableTrajectory(
            observable="local_energy",
            values=values,
            draw_stride=1,
            burn_in_draws=0,
        )
    )


VALUES = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def _read_metadata(artifact: TrajectoryRecordArtifact) -> dict:
    return json.loads(artifact.metadata_path.read_text(encoding="utf-8"))


def _write_metadata(artifact: TrajectoryRecordArtifact, metadata: dict) -> None:
    artifact.metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Accepted v1 decoding


def test_v1_record_decodes_original_values(tmp_path: Path) -> None:
    artifact = _write_v1_artifact(tmp_path, VALUES)
    result = inspect_trajectory_record(artifact.metadata_path)
    assert result.schema == TRAJECTORY_RECORD_INSPECTION_SCHEMA
    assert result.record_schema == TRAJECTORY_RECORD_SCHEMA
    assert result.draw_count == artifact.n_draws
    assert result.walker_count == artifact.n_walkers
    assert result.row_count == artifact.row_count
    assert result.mean == pytest.approx(artifact.mean)
    assert result.variance == pytest.approx(artifact.variance)
    assert result.csv_sha256 == artifact.csv_sha256
    assert result.atomic_configuration_id == artifact.atomic_configuration.content_id()


def test_v1_reader_reachable_from_public_inspection_namespace(tmp_path: Path) -> None:
    artifact = _write_v1_artifact(tmp_path, VALUES)
    result = inspection.inspect_trajectory_record(artifact.metadata_path)
    assert isinstance(result, inspection.TrajectoryRecordInspection)


# ---------------------------------------------------------------------------
# Physical context: typed unavailable marker, never inferred from geometry


def test_v1_physical_context_is_explicitly_unavailable(tmp_path: Path) -> None:
    artifact = _write_v1_artifact(tmp_path, VALUES, n_nuclei=2)
    result = inspect_trajectory_record(artifact.metadata_path)
    assert result.physical_context.available is False
    assert result.physical_context.system_id is None
    assert result.physical_context.reason
    # atomic_configuration_id is a geometry resource ID; it must never be
    # repurposed as (or equal in meaning to) a physical-system identity.
    assert result.atomic_configuration_id == artifact.atomic_configuration.content_id()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"available": True, "system_id": None, "reason": None},
        {"available": True, "system_id": "abc", "reason": "x"},
        {"available": False, "system_id": "abc", "reason": None},
        {"available": False, "system_id": None, "reason": None},
    ],
)
def test_physical_context_rejects_inconsistent_construction(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TrajectoryPhysicalContext(**kwargs)


def test_physical_context_accepts_consistent_construction() -> None:
    TrajectoryPhysicalContext(available=True, system_id="abc", reason=None)
    TrajectoryPhysicalContext(available=False, system_id=None, reason="x")


# ---------------------------------------------------------------------------
# Rejections


def test_unsupported_schema_is_rejected(tmp_path: Path) -> None:
    artifact = _write_v1_artifact(tmp_path, VALUES)
    metadata = _read_metadata(artifact)
    metadata["schema"] = "trajectory_records/v99"
    _write_metadata(artifact, metadata)
    with pytest.raises(UnsupportedTrajectoryRecordSchemaError):
        inspect_trajectory_record(artifact.metadata_path)


def test_missing_schema_is_rejected(tmp_path: Path) -> None:
    artifact = _write_v1_artifact(tmp_path, VALUES)
    metadata = _read_metadata(artifact)
    del metadata["schema"]
    _write_metadata(artifact, metadata)
    with pytest.raises(UnsupportedTrajectoryRecordSchemaError):
        inspect_trajectory_record(artifact.metadata_path)


def test_tampered_csv_hash_is_rejected(tmp_path: Path) -> None:
    artifact = _write_v1_artifact(tmp_path, VALUES)
    with artifact.path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(TamperedTrajectoryRecordError, match="csv_sha256"):
        inspect_trajectory_record(artifact.metadata_path)


def test_tampered_sidecar_key_set_is_rejected(tmp_path: Path) -> None:
    artifact = _write_v1_artifact(tmp_path, VALUES)
    metadata = _read_metadata(artifact)
    del metadata["mean"]
    _write_metadata(artifact, metadata)
    with pytest.raises(TamperedTrajectoryRecordError, match="key set"):
        inspect_trajectory_record(artifact.metadata_path)


def test_sidecar_key_set_matches_declared_constant(tmp_path: Path) -> None:
    artifact = _write_v1_artifact(tmp_path, VALUES)
    assert set(_read_metadata(artifact)) == set(TRAJECTORY_RECORD_METADATA_KEYS)


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_duplicate_grid_row_is_rejected(tmp_path: Path) -> None:
    artifact = _write_v1_artifact(tmp_path, VALUES)
    with artifact.path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    rows.append(dict(rows[0]))
    _write_rows(artifact.path, fieldnames, rows)
    metadata = _read_metadata(artifact)
    metadata["csv_sha256"] = trajectory_records._file_sha256(artifact.path)
    metadata["byte_count"] = artifact.path.stat().st_size
    _write_metadata(artifact, metadata)
    with pytest.raises(TamperedTrajectoryRecordError, match="duplicate"):
        inspect_trajectory_record(artifact.metadata_path)


def test_incomplete_grid_row_is_rejected(tmp_path: Path) -> None:
    artifact = _write_v1_artifact(tmp_path, VALUES)
    with artifact.path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    del rows[-1]
    _write_rows(artifact.path, fieldnames, rows)
    metadata = _read_metadata(artifact)
    metadata["csv_sha256"] = trajectory_records._file_sha256(artifact.path)
    metadata["byte_count"] = artifact.path.stat().st_size
    _write_metadata(artifact, metadata)
    with pytest.raises(TamperedTrajectoryRecordError, match="incomplete"):
        inspect_trajectory_record(artifact.metadata_path)


# ---------------------------------------------------------------------------
# Immutable compatibility fixtures: frozen-main v1 writer output


FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "trajectory_records" / "v1"


@pytest.mark.parametrize(
    "fixture_name,n_electrons,nuclear_charge",
    [("h_like_frozen_main", 1, 1.0), ("he_like_frozen_main", 2, 2.0)],
)
def test_frozen_main_fixture_decodes_and_matches_provenance(
    fixture_name: str, n_electrons: int, nuclear_charge: float
) -> None:
    fixture_dir = FIXTURES_ROOT / fixture_name
    metadata_path = fixture_dir / "sampled_eval_table.metadata.json"
    provenance = json.loads((fixture_dir / "PROVENANCE.json").read_text(encoding="utf-8"))

    for filename, expected_sha256 in provenance["sha256"].items():
        actual_path = fixture_dir / filename
        digest = trajectory_records._file_sha256(actual_path)
        assert digest == expected_sha256, f"{filename} no longer matches its recorded PROVENANCE.json hash"

    result = inspect_trajectory_record(metadata_path)
    assert result.record_schema == TRAJECTORY_RECORD_SCHEMA
    assert result.draw_count == 2
    assert result.walker_count == 3
    assert result.row_count == 6
    assert result.mean == pytest.approx(7.0)
    assert result.variance == pytest.approx(25.66666666666667)
    assert result.physical_context.available is False
    assert result.physical_context.system_id is None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["n_electrons"] == n_electrons
    assert metadata["atomic_configuration"]["charges"] == [nuclear_charge]


def test_mean_variance_tamper_detected_even_with_consistent_grid(tmp_path: Path) -> None:
    """A sidecar whose mean/variance disagree with its own (otherwise intact) CSV is rejected."""

    artifact = _write_v1_artifact(tmp_path, VALUES)
    metadata = _read_metadata(artifact)
    metadata["mean"] = metadata["mean"] + 1.0
    _write_metadata(artifact, metadata)
    with pytest.raises(TamperedTrajectoryRecordError, match="mean/variance"):
        inspect_trajectory_record(artifact.metadata_path)
