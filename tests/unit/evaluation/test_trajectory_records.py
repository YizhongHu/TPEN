"""Contracts for the trajectory-record metadata sidecar validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.evaluation import trajectory_records
from tpen.evaluation.trajectory_records import (
    TRAJECTORY_RECORD_METADATA_KEYS,
    TrajectoryRecordArtifact,
    TrajectoryRecordBatch,
    TrajectoryRecordStreamWriter,
)
from tpen.statistics.trajectory import ObservableTrajectory


def _artifact(
    tmp_path: Path,
    values: torch.Tensor,
    *,
    name: str = "trajectory.csv",
    nucleus_free: bool = False,
    atomic_configuration: AtomicConfiguration | None = None,
) -> TrajectoryRecordArtifact:
    values = values.to(torch.float64)
    n_draws, n_walkers = values.shape
    positions = torch.zeros(n_draws, n_walkers, 1, 3, dtype=torch.float64)
    atoms = atomic_configuration
    if atoms is None and not nucleus_free:
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


def _rewrite_metadata(artifact: TrajectoryRecordArtifact, metadata: object) -> None:
    artifact.metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")


def _read_metadata(artifact: TrajectoryRecordArtifact) -> dict:
    return json.loads(artifact.metadata_path.read_text(encoding="utf-8"))


FINITE = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Literal, independent of TRAJECTORY_RECORD_METADATA_KEYS: a key dropped from the
# production constant must still appear here, so its altered-value/missing-key
# tests keep running and can catch the drift (PR 609 pre-review round 1, finding 1).
EXPECTED_METADATA_KEYS = (
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


def test_freshly_written_artifact_validates(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    artifact.validate()


def test_nucleus_free_artifact_round_trips_with_explicit_absence(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, FINITE, name="nucleus-free.csv", nucleus_free=True)

    assert artifact.atomic_configuration is None
    metadata = _read_metadata(artifact)
    assert metadata["atomic_configuration_id"] is None
    assert metadata["atomic_configuration"] is None
    artifact.validate()


def test_nucleus_record_bytes_remain_stable(tmp_path: Path) -> None:
    first = _artifact(tmp_path, FINITE, name="first.csv")
    second = _artifact(tmp_path, FINITE, name="second.csv")

    # The filenames are intentionally normalized before comparison: all other
    # CSV and sidecar bytes must remain identical for an existing atom record.
    first_csv = first.path.read_bytes().replace(b"first.csv", b"second.csv")
    assert first_csv == second.path.read_bytes()
    first_metadata = _read_metadata(first)
    second_metadata = _read_metadata(second)
    first_metadata["csv_filename"] = second_metadata["csv_filename"]
    assert first_metadata == second_metadata


def test_sidecar_key_set_equals_declared_constant(tmp_path: Path) -> None:
    """The writer emits exactly the declared key set, with no duplicates.

    Guards against drift between ``TRAJECTORY_RECORD_METADATA_KEYS`` and the
    sidecar the writer actually produces, independent of the literal key list
    the parametrized tests below use.
    """

    artifact = _artifact(tmp_path, FINITE)
    assert len(set(TRAJECTORY_RECORD_METADATA_KEYS)) == len(TRAJECTORY_RECORD_METADATA_KEYS)
    assert set(_read_metadata(artifact)) == set(TRAJECTORY_RECORD_METADATA_KEYS)
    assert set(TRAJECTORY_RECORD_METADATA_KEYS) == set(EXPECTED_METADATA_KEYS)


def test_constant_dict_divergence_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped/renamed key in the declared constant is a loud programming error.

    Survives `python -O`, unlike the `assert` it replaces (PR 609 pre-review
    round 2, finding 1).
    """

    artifact = _artifact(tmp_path, FINITE, name="divergent.csv")
    monkeypatch.setattr(
        trajectory_records,
        "TRAJECTORY_RECORD_METADATA_KEYS",
        tuple(key for key in TRAJECTORY_RECORD_METADATA_KEYS if key != "schema") + ("bogus",),
    )
    with pytest.raises(RuntimeError, match=r"missing=\['bogus'\], extra=\['schema'\]"):
        artifact.validate()


@pytest.mark.parametrize("key", EXPECTED_METADATA_KEYS)
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
    with pytest.raises(ValueError, match=f"{key!r} disagrees with typed manifest"):
        artifact.validate()


@pytest.mark.parametrize("key", EXPECTED_METADATA_KEYS)
def test_missing_metadata_key_is_refused(tmp_path: Path, key: str) -> None:
    artifact = _artifact(tmp_path, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), name=f"missing-{key}.csv")
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    del metadata[key]
    _rewrite_metadata(artifact, metadata)
    with pytest.raises(ValueError, match=f"missing required key {key!r}"):
        artifact.validate()


@pytest.mark.parametrize(
    ("values", "expected_mean", "expected_variance"),
    [
        ([[float("inf"), 1.0], [1.0, 2.0]], "inf", "nan"),
        ([[float("-inf"), 1.0], [1.0, 2.0]], "-inf", "nan"),
        ([[float("inf"), float("-inf")], [1.0, 2.0]], "nan", "nan"),
    ],
    ids=["mean-inf", "mean-neg-inf", "mean-nan"],
)
def test_nonfinite_sentinels_are_exact_and_validate(
    tmp_path: Path, values: list[list[float]], expected_mean: str, expected_variance: str
) -> None:
    """Each `_json_float` sentinel is asserted exactly, then `validate()` accepts it."""

    artifact = _artifact(tmp_path, torch.tensor(values), name="nonfinite.csv")
    metadata = _read_metadata(artifact)
    assert metadata["mean"] == expected_mean
    assert metadata["variance"] == expected_variance
    artifact.validate()


def test_tampered_nonfinite_sentinel_is_refused(tmp_path: Path) -> None:
    """Swapping one sentinel string for another is a value mismatch."""

    artifact = _artifact(tmp_path, torch.tensor([[float("inf"), float("-inf")], [1.0, 2.0]]), name="nonfinite.csv")
    metadata = _read_metadata(artifact)
    assert metadata["mean"] == "nan"
    metadata["mean"] = "inf"
    _rewrite_metadata(artifact, metadata)
    with pytest.raises(ValueError, match="'mean' disagrees with typed manifest"):
        artifact.validate()


@pytest.mark.parametrize("field", ["positions", "charges"])
def test_altered_atomic_configuration_entry_is_refused(tmp_path: Path, field: str) -> None:
    """A numeric change inside the nested payload, not only an added key, is refused."""

    artifact = _artifact(tmp_path, FINITE, name=f"atoms-{field}.csv")
    metadata = _read_metadata(artifact)
    if field == "positions":
        metadata["atomic_configuration"]["positions"][0][0] += 1e-12
    else:
        metadata["atomic_configuration"]["charges"][0] += 1.0
    _rewrite_metadata(artifact, metadata)
    with pytest.raises(ValueError, match="'atomic_configuration' disagrees with typed manifest"):
        artifact.validate()


def test_undeclared_extra_key_is_refused(tmp_path: Path) -> None:
    """Strict key-set equality (PR 609 pre-review round 1, finding 3, decided strict).

    The schema is versioned (``trajectory_records/v1``) and the writer emits
    exactly the declared keys, so an undeclared key signals a writer/schema
    mismatch and must be refused, naming the offending key.
    """

    artifact = _artifact(tmp_path, FINITE, name="extra.csv")
    metadata = _read_metadata(artifact)
    metadata["__undeclared__"] = 1
    _rewrite_metadata(artifact, metadata)
    with pytest.raises(ValueError, match=r"undeclared key\(s\): __undeclared__"):
        artifact.validate()


@pytest.mark.parametrize(
    "payload",
    ["null", "5", json.dumps(" ".join(TRAJECTORY_RECORD_METADATA_KEYS)), "[]"],
    ids=["null", "int", "str-containing-every-key", "list"],
)
def test_non_object_sidecar_is_refused_with_value_error(tmp_path: Path, payload: str) -> None:
    """A sidecar that is valid JSON but not an object is refused with `ValueError`."""

    artifact = _artifact(tmp_path, FINITE, name="malformed.csv")
    artifact.metadata_path.write_text(payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        artifact.validate()
