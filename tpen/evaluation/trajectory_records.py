"""Typed, streamed evaluation records for observable trajectories."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch
from tpen.statistics.trajectory import (
    ObservableTrajectory,
    ObservableTrajectoryReconciliation,
)

TRAJECTORY_RECORD_FILENAME = "sampled_eval_table.csv"
"""Canonical CSV name for a complete draw-by-walker trajectory artifact."""

TRAJECTORY_RECORD_SCHEMA = "trajectory_records/v1"
"""Versioned serialized trajectory-record schema."""

NUCLEUS_FREE_TRAJECTORY_RECORD_SCHEMA = "trajectory_records/v2"
"""Explicit null nuclear geometry; v1 nuclear records remain byte-compatible."""


@dataclass(frozen=True)
class TrajectoryRecordBatch:
    """One complete retained draw of row-aligned trajectory primitives.

    Every tensor is detached, cloned, and moved to CPU during construction.
    Energy-like fields are promoted to float64, while integer grid coordinates
    and the finite mask retain their semantic dtypes.

    Parameters
    ----------
    draw_index : torch.Tensor
        Repeated retained-draw index with shape ``[n_walkers]``.
    walker_index : torch.Tensor
        Complete walker grid ``0..n_walkers-1``.
    positions : torch.Tensor
        Raw electron coordinates with shape
        ``[n_walkers, n_electrons, spatial_dim]``.
    local_energy : torch.Tensor
        Total local energy for the same rows.
    term_energies : mapping of str to torch.Tensor
        Every configured Hamiltonian contribution for the same rows.
    logabs, sign : torch.Tensor
        Signed-log wavefunction values captured from the exact forward used by
        the kinetic local-energy evaluation.
    finite_mask : torch.Tensor
        Boolean mask equal to ``isfinite(local_energy)``.
    """

    draw_index: torch.Tensor
    walker_index: torch.Tensor
    positions: torch.Tensor
    local_energy: torch.Tensor
    term_energies: Mapping[str, torch.Tensor]
    logabs: torch.Tensor
    sign: torch.Tensor
    finite_mask: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "draw_index", _owned_cpu(self.draw_index, dtype=torch.int64))
        object.__setattr__(self, "walker_index", _owned_cpu(self.walker_index, dtype=torch.int64))
        object.__setattr__(self, "positions", _owned_cpu(self.positions, dtype=torch.float64))
        object.__setattr__(self, "local_energy", _owned_cpu(self.local_energy, dtype=torch.float64))
        object.__setattr__(self, "logabs", _owned_cpu(self.logabs, dtype=torch.float64))
        object.__setattr__(self, "sign", _owned_cpu(self.sign, dtype=torch.float64))
        object.__setattr__(self, "finite_mask", _owned_cpu(self.finite_mask, dtype=torch.bool))
        owned_terms = {
            str(name): _owned_cpu(value, dtype=torch.float64)
            for name, value in self.term_energies.items()
        }
        object.__setattr__(self, "term_energies", MappingProxyType(owned_terms))
        self._validate_fields()

    @property
    def row_count(self) -> int:
        """Return the number of walkers in this retained draw."""

        return int(self.local_energy.numel())

    @property
    def retained_draw_index(self) -> int:
        """Return the single retained-draw index represented by this batch."""

        return int(self.draw_index[0].item())

    @property
    def n_electrons(self) -> int:
        """Return the electron count carried by every row."""

        return int(self.positions.shape[1])

    @property
    def spatial_dim(self) -> int:
        """Return the coordinate dimension carried by every electron."""

        return int(self.positions.shape[2])

    def validate(self) -> "TrajectoryRecordBatch":
        """Validate the explicit row, grid, term, and finite-mask contract."""

        self._validate_fields()
        return self

    def content_id(self) -> str:
        """Return a device/dtype-independent hash over every row field."""

        digest = hashlib.sha256()
        digest.update(str((self.row_count, self.n_electrons, self.spatial_dim)).encode("utf-8"))
        for name, value in (
            ("draw_index", self.draw_index),
            ("walker_index", self.walker_index),
            ("positions", self.positions),
            ("local_energy", self.local_energy),
            ("logabs", self.logabs),
            ("sign", self.sign),
            ("finite_mask", self.finite_mask),
        ):
            _update_tensor_digest(digest, name, value)
        for name in sorted(self.term_energies):
            _update_tensor_digest(digest, f"term/{name}", self.term_energies[name])
        return digest.hexdigest()

    def _validate_fields(self) -> None:
        n_rows = self.row_count
        if n_rows < 1:
            raise ValueError("TrajectoryRecordBatch requires at least one walker row")
        if tuple(self.positions.shape[:1]) != (n_rows,) or self.positions.ndim != 3:
            raise ValueError(
                "TrajectoryRecordBatch.positions must have shape "
                "[n_walkers, n_electrons, spatial_dim]"
            )
        expected = (n_rows,)
        for name, value in (
            ("draw_index", self.draw_index),
            ("walker_index", self.walker_index),
            ("local_energy", self.local_energy),
            ("logabs", self.logabs),
            ("sign", self.sign),
            ("finite_mask", self.finite_mask),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"TrajectoryRecordBatch.{name} must have shape {expected}")
        if self.draw_index.dtype != torch.int64 or self.walker_index.dtype != torch.int64:
            raise ValueError("trajectory draw_index and walker_index must be int64")
        if self.finite_mask.dtype != torch.bool:
            raise ValueError("TrajectoryRecordBatch.finite_mask must be bool")
        if not torch.all(self.draw_index == self.draw_index[0]):
            raise ValueError("TrajectoryRecordBatch must contain exactly one retained draw")
        expected_walkers = torch.arange(n_rows, dtype=torch.int64)
        if not torch.equal(self.walker_index, expected_walkers):
            raise ValueError("TrajectoryRecordBatch walker_index must be the complete ordered grid")
        if not torch.equal(self.finite_mask, torch.isfinite(self.local_energy)):
            raise ValueError("TrajectoryRecordBatch.finite_mask must equal isfinite(local_energy)")
        if not self.term_energies:
            raise ValueError("TrajectoryRecordBatch requires every configured Hamiltonian term")
        for name, value in self.term_energies.items():
            if not name.strip():
                raise ValueError("trajectory Hamiltonian term names must be non-empty")
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"TrajectoryRecordBatch term {name!r} must have shape {expected}"
                )


@dataclass(frozen=True)
class TrajectoryRecordArtifact:
    """Typed manifest for one complete, streamed trajectory-record artifact."""

    path: Path
    metadata_path: Path
    observable: str
    n_draws: int
    n_walkers: int
    draw_stride: int
    burn_in_draws: int
    n_electrons: int
    spatial_dim: int
    term_names: tuple[str, ...]
    row_count: int
    finite_count: int
    mean: float
    variance: float
    observable_values_content_id: str
    csv_sha256: str
    byte_count: int
    atomic_configuration: AtomicConfiguration | None
    final_draw: TrajectoryRecordBatch

    @property
    def schema(self) -> str:
        """Return the schema that represents this artifact's nuclear geometry."""
        return (NUCLEUS_FREE_TRAJECTORY_RECORD_SCHEMA if self.atomic_configuration is None
                else TRAJECTORY_RECORD_SCHEMA)

    @property
    def atomic_configuration_id(self) -> str | None:
        """Return the nuclear geometry identity, or explicit absence of nuclei."""
        return None if self.atomic_configuration is None else self.atomic_configuration.content_id()

    @property
    def nonfinite_count(self) -> int:
        """Return the number of non-finite total-energy rows."""

        return self.row_count - self.finite_count

    def validate(self, *, check_files: bool = True) -> "TrajectoryRecordArtifact":
        """Validate manifest, final-draw, geometry, and optional file identity."""

        if not self.observable.strip():
            raise ValueError("trajectory record observable must be non-empty")
        if self.n_draws < 1 or self.n_walkers < 1:
            raise ValueError("trajectory record shape must have positive draw and walker axes")
        if self.row_count != self.n_draws * self.n_walkers:
            raise ValueError("trajectory record row_count must equal n_draws*n_walkers")
        if not 0 <= self.finite_count <= self.row_count:
            raise ValueError("trajectory record finite_count is outside the row count")
        if self.draw_stride < 1 or self.burn_in_draws < 0:
            raise ValueError("trajectory record stride/burn-in metadata is invalid")
        if len(set(self.term_names)) != len(self.term_names) or not self.term_names:
            raise ValueError("trajectory record term_names must be non-empty and unique")
        if any(not name.strip() for name in self.term_names):
            raise ValueError("trajectory record term_names must not contain blanks")
        if self.atomic_configuration is not None:
            self.atomic_configuration.validate()
            if self.atomic_configuration.spatial_dim != self.spatial_dim:
                raise ValueError("trajectory record geometry dimension disagrees with coordinates")
        final_draw = self.final_draw.validate()
        if final_draw.retained_draw_index != self.n_draws - 1:
            raise ValueError("trajectory record final_draw must be the last retained draw")
        if final_draw.row_count != self.n_walkers:
            raise ValueError("trajectory record final_draw must contain every walker")
        if final_draw.n_electrons != self.n_electrons or final_draw.spatial_dim != self.spatial_dim:
            raise ValueError("trajectory record final_draw coordinate shape disagrees with manifest")
        if tuple(final_draw.term_energies) != self.term_names:
            raise ValueError("trajectory record final_draw terms disagree with manifest")
        for name, value in (
            ("observable_values_content_id", self.observable_values_content_id),
            ("csv_sha256", self.csv_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"trajectory record {name} must be a lowercase sha256 digest")
        if self.byte_count < 1:
            raise ValueError("trajectory record byte_count must be positive")
        if check_files:
            if not self.path.is_file() or not self.metadata_path.is_file():
                raise ValueError("trajectory record CSV and metadata sidecar must both exist")
            if self.path.stat().st_size != self.byte_count:
                raise ValueError("trajectory record byte_count disagrees with the CSV")
            if _file_sha256(self.path) != self.csv_sha256:
                raise ValueError("trajectory record csv_sha256 disagrees with the CSV")
            self._validate_serialized_metadata()
        return self

    def reconcile(self, trajectory: ObservableTrajectory) -> None:
        """Fail loudly unless statistics input and record artifact agree exactly."""

        self.validate()
        expected = trajectory.reconciliation()
        mismatches: list[str] = []
        comparisons = {
            "observable": (self.observable, expected.observable),
            "draw_count": (self.n_draws, expected.draw_count),
            "walker_count": (self.n_walkers, expected.walker_count),
            "draw_stride": (self.draw_stride, expected.draw_stride),
            "burn_in_draws": (self.burn_in_draws, expected.burn_in_draws),
            "row_count": (self.row_count, expected.row_count),
            "finite_count": (self.finite_count, expected.finite_count),
            "nonfinite_count": (self.nonfinite_count, expected.nonfinite_count),
            "observable_values_content_id": (
                self.observable_values_content_id,
                expected.values_content_id,
            ),
        }
        for name, (actual, wanted) in comparisons.items():
            if actual != wanted:
                mismatches.append(f"{name}: record={actual!r} trajectory={wanted!r}")
        for name, actual, wanted in (
            ("mean", self.mean, expected.mean),
            ("variance", self.variance, expected.variance),
        ):
            if not _same_float(actual, wanted):
                mismatches.append(f"{name}: record={actual!r} trajectory={wanted!r}")
        if mismatches:
            raise ValueError("trajectory record/statistics reconciliation failed: " + "; ".join(mismatches))

    def validate_snapshot_batch(self, batch: ElectronBatch) -> int:
        """Validate that a generated batch is a prefix of the final retained draw."""

        flat = batch.flatten_samples()
        n_rows = flat.batch_size
        if n_rows > self.n_walkers:
            raise ValueError("final-draw snapshot cannot contain more rows than the trajectory")
        positions = flat.positions.detach().to(device="cpu", dtype=torch.float64)
        if not torch.equal(positions, self.final_draw.positions[:n_rows]):
            raise ValueError("generated snapshot is not the final retained trajectory draw")
        atoms = flat.atomic_configuration
        if (None if atoms is None else atoms.content_id()) != self.atomic_configuration_id:
            raise ValueError("generated snapshot atomic geometry disagrees with trajectory records")
        return n_rows

    def _validate_serialized_metadata(self) -> None:
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        expected_scalars = {
            "schema": self.schema,
            "row_semantics": "complete_draw_walker_grid",
            "csv_filename": self.path.name,
            "observable": self.observable,
            "draw_count": self.n_draws,
            "walker_count": self.n_walkers,
            "draw_stride": self.draw_stride,
            "burn_in_draws": self.burn_in_draws,
            "n_electrons": self.n_electrons,
            "spatial_dim": self.spatial_dim,
            "term_names": list(self.term_names),
            "row_count": self.row_count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "mean": _json_float(self.mean),
            "variance": _json_float(self.variance),
            "observable_values_content_id": self.observable_values_content_id,
            "csv_sha256": self.csv_sha256,
            "byte_count": self.byte_count,
            "atomic_configuration_id": self.atomic_configuration_id,
            "atomic_configuration": _atomic_geometry_payload(self.atomic_configuration),
        }
        for key, expected in expected_scalars.items():
            if key not in metadata or metadata[key] != expected:
                raise ValueError(f"trajectory record metadata {key!r} disagrees with typed manifest")


class TrajectoryRecordStreamWriter:
    """Incrementally write complete retained draws without accumulating rows."""

    def __init__(
        self,
        path: Path,
        *,
        observable: str,
        n_draws: int,
        n_walkers: int,
        term_names: tuple[str, ...],
        atomic_configuration: AtomicConfiguration | None,
        first_draw: TrajectoryRecordBatch,
    ) -> None:
        self.path = Path(path)
        self.observable = str(observable).strip()
        self.n_draws = int(n_draws)
        self.n_walkers = int(n_walkers)
        self.term_names = tuple(term_names)
        self.atomic_configuration = atomic_configuration
        first_draw.validate()
        if first_draw.row_count != self.n_walkers:
            raise ValueError("first trajectory record draw must contain every walker")
        if tuple(first_draw.term_energies) != self.term_names:
            raise ValueError("first trajectory record draw terms disagree with configured terms")
        if (self.atomic_configuration is not None
                and self.atomic_configuration.spatial_dim != first_draw.spatial_dim):
            raise ValueError("trajectory record coordinates disagree with AtomicConfiguration")

        self.n_electrons = first_draw.n_electrons
        self.spatial_dim = first_draw.spatial_dim
        self._next_draw = 0
        self._row_count = 0
        self._finite_count = 0
        self._draw_sums: list[torch.Tensor] = []
        self._draw_square_sums: list[torch.Tensor] = []
        self._final_draw: TrajectoryRecordBatch | None = None
        self._values_digest = hashlib.sha256()
        self._values_digest.update(str((self.n_draws, self.n_walkers)).encode("utf-8"))

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8", newline="")
        self._fieldnames = self._build_fieldnames()
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=self._fieldnames,
            lineterminator="\n",
        )
        self._writer.writeheader()
        self.append(first_draw)

    def append(self, draw: TrajectoryRecordBatch) -> None:
        """Write one complete retained draw and release it after this call."""

        draw.validate()
        if draw.retained_draw_index != self._next_draw:
            raise ValueError(
                f"trajectory record draws must be contiguous from zero; expected "
                f"{self._next_draw}, got {draw.retained_draw_index}"
            )
        if draw.row_count != self.n_walkers:
            raise ValueError("trajectory record draw must contain every walker")
        if draw.n_electrons != self.n_electrons or draw.spatial_dim != self.spatial_dim:
            raise ValueError("trajectory record coordinate shape changed between draws")
        if tuple(draw.term_energies) != self.term_names:
            raise ValueError("trajectory record Hamiltonian terms changed between draws")

        values = draw.local_energy.contiguous()
        self._values_digest.update(values.numpy().astype("<f8", copy=False).tobytes())
        self._draw_sums.append(values.sum())
        self._draw_square_sums.append(values.square().sum())
        self._finite_count += int(draw.finite_mask.sum().item())
        for walker_index in range(self.n_walkers):
            self._writer.writerow(self._row(draw, walker_index))
        self._row_count += self.n_walkers
        self._next_draw += 1
        self._final_draw = draw

    def finalize(
        self,
        trajectory: ObservableTrajectory,
    ) -> TrajectoryRecordArtifact:
        """Close the stream, reconcile it, and write one metadata sidecar."""

        if self._next_draw != self.n_draws:
            raise ValueError(
                f"trajectory record stream received {self._next_draw} draws, expected {self.n_draws}"
            )
        self.close()
        if self._final_draw is None:
            raise ValueError("trajectory record stream has no final draw")

        record_reconciliation = self._reconciliation(
            draw_stride=trajectory.draw_stride,
            burn_in_draws=trajectory.burn_in_draws,
        )
        expected = trajectory.reconciliation()
        if not _same_reconciliation(record_reconciliation, expected):
            raise ValueError(
                "trajectory record stream disagrees with ObservableTrajectory before artifact finalization"
            )

        csv_sha256 = _file_sha256(self.path)
        byte_count = self.path.stat().st_size
        metadata_path = self.path.with_suffix(".metadata.json")
        artifact = TrajectoryRecordArtifact(
            path=self.path,
            metadata_path=metadata_path,
            observable=self.observable,
            n_draws=self.n_draws,
            n_walkers=self.n_walkers,
            draw_stride=trajectory.draw_stride,
            burn_in_draws=trajectory.burn_in_draws,
            n_electrons=self.n_electrons,
            spatial_dim=self.spatial_dim,
            term_names=self.term_names,
            row_count=self._row_count,
            finite_count=self._finite_count,
            mean=record_reconciliation.mean,
            variance=record_reconciliation.variance,
            observable_values_content_id=record_reconciliation.values_content_id,
            csv_sha256=csv_sha256,
            byte_count=byte_count,
            atomic_configuration=self.atomic_configuration,
            final_draw=self._final_draw,
        )
        _write_metadata(artifact)
        artifact.validate()
        artifact.reconcile(trajectory)
        return artifact

    def close(self) -> None:
        """Close the output while preserving any partial file for diagnosis."""

        if not self._handle.closed:
            self._handle.close()

    def _reconciliation(
        self,
        *,
        draw_stride: int,
        burn_in_draws: int,
    ) -> ObservableTrajectoryReconciliation:
        draw_sums = torch.stack(self._draw_sums)
        draw_square_sums = torch.stack(self._draw_square_sums)
        total = draw_sums.sum()
        total_square = draw_square_sums.sum()
        mean_tensor = total / self._row_count
        mean = float(mean_tensor.item())
        variance = float((total_square / self._row_count - mean_tensor.square()).item())
        if math.isfinite(variance) and variance < 0.0:
            variance = 0.0
        return ObservableTrajectoryReconciliation(
            observable=self.observable,
            draw_count=self.n_draws,
            walker_count=self.n_walkers,
            draw_stride=draw_stride,
            burn_in_draws=burn_in_draws,
            row_count=self._row_count,
            finite_count=self._finite_count,
            mean=mean,
            variance=variance,
            values_content_id=self._values_digest.hexdigest(),
        )

    def _build_fieldnames(self) -> list[str]:
        coordinates = [
            f"position/electron_{electron}/axis_{axis}"
            for electron in range(self.n_electrons)
            for axis in range(self.spatial_dim)
        ]
        return [
            "sample_index",
            "draw_index",
            "walker_index",
            "local_energy",
            *(f"term/{name}" for name in self.term_names),
            "logabs",
            "sign",
            "finite",
            *coordinates,
        ]

    def _row(self, draw: TrajectoryRecordBatch, walker_index: int) -> dict[str, object]:
        row: dict[str, object] = {
            "sample_index": draw.retained_draw_index * self.n_walkers + walker_index,
            "draw_index": draw.retained_draw_index,
            "walker_index": walker_index,
            "local_energy": _float_or_text(draw.local_energy[walker_index]),
            "logabs": _float_or_text(draw.logabs[walker_index]),
            "sign": _float_or_text(draw.sign[walker_index]),
            "finite": bool(draw.finite_mask[walker_index].item()),
        }
        for name in self.term_names:
            row[f"term/{name}"] = _float_or_text(draw.term_energies[name][walker_index])
        for electron in range(self.n_electrons):
            for axis in range(self.spatial_dim):
                row[f"position/electron_{electron}/axis_{axis}"] = _float_or_text(
                    draw.positions[walker_index, electron, axis]
                )
        return row


def _write_metadata(artifact: TrajectoryRecordArtifact) -> None:
    atoms = artifact.atomic_configuration
    metadata = {
        "schema": artifact.schema,
        "row_semantics": "complete_draw_walker_grid",
        "csv_filename": artifact.path.name,
        "observable": artifact.observable,
        "draw_count": artifact.n_draws,
        "walker_count": artifact.n_walkers,
        "draw_stride": artifact.draw_stride,
        "burn_in_draws": artifact.burn_in_draws,
        "row_count": artifact.row_count,
        "finite_count": artifact.finite_count,
        "nonfinite_count": artifact.nonfinite_count,
        "mean": _json_float(artifact.mean),
        "variance": _json_float(artifact.variance),
        "term_names": list(artifact.term_names),
        "n_electrons": artifact.n_electrons,
        "spatial_dim": artifact.spatial_dim,
        "observable_values_content_id": artifact.observable_values_content_id,
        "csv_sha256": artifact.csv_sha256,
        "byte_count": artifact.byte_count,
        "atomic_configuration_id": artifact.atomic_configuration_id,
        "atomic_configuration": _atomic_geometry_payload(atoms),
    }
    with artifact.metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _atomic_geometry_payload(atoms: AtomicConfiguration | None) -> dict[str, Any] | None:
    """Serialize actual nuclei without inventing one for external confinement."""
    if atoms is None:
        return None
    return {"positions": atoms.positions.detach().to(torch.float64).cpu().tolist(),
            "charges": atoms.charges.detach().to(torch.float64).cpu().tolist()}


def _owned_cpu(value: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"trajectory record fields must be torch.Tensor, got {type(value).__name__}")
    return value.detach().to(device="cpu", dtype=dtype).clone().requires_grad_(False)


def _update_tensor_digest(digest: Any, name: str, value: torch.Tensor) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    if value.dtype == torch.bool:
        digest.update(value.to(torch.uint8).contiguous().numpy().tobytes())
    elif value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        digest.update(value.to(torch.int64).contiguous().numpy().astype("<i8", copy=False).tobytes())
    else:
        digest.update(value.to(torch.float64).contiguous().numpy().astype("<f8", copy=False).tobytes())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_or_text(value: torch.Tensor) -> float | str:
    number = float(value.item())
    if math.isfinite(number):
        return number
    return "inf" if number > 0 else "-inf" if number < 0 else "nan"


def _json_float(value: float) -> float | str:
    if math.isfinite(value):
        return value
    return "inf" if value > 0 else "-inf" if value < 0 else "nan"


def _same_float(left: float, right: float) -> bool:
    return left == right or (math.isnan(left) and math.isnan(right))


def _same_reconciliation(
    left: ObservableTrajectoryReconciliation,
    right: ObservableTrajectoryReconciliation,
) -> bool:
    scalar_fields = (
        "observable",
        "draw_count",
        "walker_count",
        "draw_stride",
        "burn_in_draws",
        "row_count",
        "finite_count",
        "values_content_id",
    )
    return all(getattr(left, field) == getattr(right, field) for field in scalar_fields) and _same_float(
        left.mean, right.mean
    ) and _same_float(left.variance, right.variance)


__all__ = [
    "TRAJECTORY_RECORD_FILENAME",
    "TRAJECTORY_RECORD_SCHEMA",
    "NUCLEUS_FREE_TRAJECTORY_RECORD_SCHEMA",
    "TrajectoryRecordArtifact",
    "TrajectoryRecordBatch",
    "TrajectoryRecordStreamWriter",
]
