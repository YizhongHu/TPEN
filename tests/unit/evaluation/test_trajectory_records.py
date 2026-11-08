"""Contracts for the trajectory-record metadata sidecar validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.evaluation.trajectory_records import (
    TrajectoryRecordArtifact,
    TrajectoryRecordBatch,
    TrajectoryRecordStreamWriter,
)
from tpen.statistics.trajectory import ObservableTrajectory

_EMITTED_METADATA_KEYS = (
    "schema",
    "row_semantics",
    "csv_filename",
    "observable",
    "draw_count",
    "walker_count",
    "draw_stride",
    "burn_in_draws",
    "n_electrons",
    "spatial_dim",
    "term_names",
    "row_count",
    "finite_count",
    "nonfinite_count",
    "mean",
    "variance",
    "observable_values_content_id",
    "csv_sha256",
    "byte_count",
    "atomic_configuration_id",
    "atomic_configuration",
)


def _artifact(tmp_path: Path, values: torch.Tensor, *, name: str = "trajectory.csv") -> TrajectoryRecordArtifact:
    values = values.to(torch.float64)
    n_draws, n_walkers = values.shape
    positions = torch.zeros(n_draws, n_walkers, 1, 3, dtype=torch.float64)
    atoms = AtomicConfiguration(
        positions=torch.zeros(1, 3, dtype=torch.float64),
        charges=torch.tensor([1.0], dtype=torch.float64),
    )

    def record(draw: int) -> TrajectoryRecordBatch:
        return TrajectoryRecordBatch(
            draw_index=torch.full((n_walkers,), draw, dtype=torch.int64),
            walker_index=torch.arange(n_walkers, dtype=torch.int64),
            positions=positions[draw],
            local_energy=values[draw],
            term_energies={"kinetic": values[draw]},
            logabs=torch.full_like(values[draw], -2.0),
            sign=torch.ones(n_walkers, dtype=torch.float64),
            finite_mask=torch.isfinite(values[draw]),
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


def _rewrite_metadata(artifact: TrajectoryRecordArtifact, metadata: dict) -> None:
    artifact.metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")


def test_freshly_written_artifact_validates(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    artifact.validate()


def test_nonfinite_mean_and_variance_round_trip(tmp_path: Path) -> None:
    values = torch.tensor([[float("inf"), float("-inf")], [1.0, 2.0]])
    artifact = _artifact(tmp_path, values, name="nonfinite.csv")
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["mean"] in ("nan", "inf", "-inf") or isinstance(metadata["mean"], float)
    artifact.validate()


@pytest.mark.parametrize("key", _EMITTED_METADATA_KEYS)
def test_altered_metadata_value_is_refused(tmp_path: Path, key: str) -> None:
    artifact = _artifact(tmp_path, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), name=f"altered-{key}.csv")
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    original = metadata[key]
    if isinstance(original, bool):
        metadata[key] = not original
    elif isinstance(original, (int, float)):
        metadata[key] = original + 1
    elif isinstance(original, str):
        metadata[key] = original + "_mutated"
    elif isinstance(original, list):
        metadata[key] = original + (original[:1] if original else [1])
    elif isinstance(original, dict):
        metadata[key] = {**original, "__mutated__": True}
    else:
        raise AssertionError(f"unexpected metadata type for {key!r}: {type(original)!r}")
    _rewrite_metadata(artifact, metadata)
    with pytest.raises(ValueError, match=key):
        artifact.validate()


@pytest.mark.parametrize("key", _EMITTED_METADATA_KEYS)
def test_missing_metadata_key_is_refused(tmp_path: Path, key: str) -> None:
    artifact = _artifact(tmp_path, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), name=f"missing-{key}.csv")
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    del metadata[key]
    _rewrite_metadata(artifact, metadata)
    with pytest.raises(ValueError, match=key):
        artifact.validate()
