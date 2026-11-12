"""Typed, streamed evaluation records for observable trajectories."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

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

TRAJECTORY_RECORD_METADATA_KEYS: tuple[str, ...] = (
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
"""Every key `_write_metadata` emits in the trajectory-record metadata sidecar."""

TRAJECTORY_RECORD_ROW_SEMANTICS = "complete_draw_walker_grid"
"""Row-ordering convention shared by the writer and the metadata validator."""


def _trajectory_record_fieldnames(
    *, n_electrons: int, spatial_dim: int, term_names: Sequence[str]
) -> list[str]:
    """Return the exact CSV column set for one trajectory-record schema shape.

    Shared by the writer (:meth:`TrajectoryRecordStreamWriter._build_fieldnames`)
    and the strict v1 reader, so a sidecar's declared ``n_electrons``,
    ``spatial_dim``, and ``term_names`` are bound to the CSV's actual columns
    rather than only to the metadata key set.
    """

    coordinates = [
        f"position/electron_{electron}/axis_{axis}"
        for electron in range(n_electrons)
        for axis in range(spatial_dim)
    ]
    return [
        "sample_index",
        "draw_index",
        "walker_index",
        "local_energy",
        *(f"term/{name}" for name in term_names),
        "logabs",
        "sign",
        "finite",
        *coordinates,
    ]


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
    atomic_configuration: AtomicConfiguration
    final_draw: TrajectoryRecordBatch

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
        if atoms is None or atoms.content_id() != self.atomic_configuration.content_id():
            raise ValueError("generated snapshot atomic geometry disagrees with trajectory records")
        return n_rows

    def _validate_serialized_metadata(self) -> None:
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("trajectory record metadata sidecar must be a JSON object")
        metadata: dict[str, Any] = raw
        atoms = self.atomic_configuration
        expected_scalars = {
            "schema": TRAJECTORY_RECORD_SCHEMA,
            "row_semantics": TRAJECTORY_RECORD_ROW_SEMANTICS,
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
            "atomic_configuration_id": atoms.content_id(),
            "atomic_configuration": _atomic_geometry_payload(atoms),
        }
        declared_keys = set(TRAJECTORY_RECORD_METADATA_KEYS)
        if expected_scalars.keys() != declared_keys:
            missing = sorted(declared_keys - expected_scalars.keys())
            extra = sorted(expected_scalars.keys() - declared_keys)
            raise RuntimeError(
                "trajectory_records.py: expected_scalars and TRAJECTORY_RECORD_METADATA_KEYS "
                f"have diverged; update both together (missing={missing}, extra={extra})"
            )
        extra_keys = set(metadata) - expected_scalars.keys()
        if extra_keys:
            raise ValueError(
                "trajectory record metadata has undeclared key(s): "
                + ", ".join(sorted(extra_keys))
            )
        for key in TRAJECTORY_RECORD_METADATA_KEYS:
            if key not in metadata:
                raise ValueError(f"trajectory record metadata is missing required key {key!r}")
        for key, expected in expected_scalars.items():
            if metadata[key] != expected:
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
        atomic_configuration: AtomicConfiguration,
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
        if self.atomic_configuration.spatial_dim != first_draw.spatial_dim:
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


TRAJECTORY_RECORD_INSPECTION_SCHEMA = "tpen.trajectory-record-inspection/v1"
"""Schema identifier for the version-aware historical-reader result below.

This is the inspection *result's* own schema, distinct from the
``schema`` field recorded inside a trajectory-record sidecar
(``TRAJECTORY_RECORD_SCHEMA`` and friends, e.g. ``trajectory_records/v1``).
A new inspection-result field would bump this identifier; a new sidecar
schema (e.g. a future ``trajectory_records/v2``) is handled by adding a
branch to :func:`inspect_trajectory_record` without changing this one.
"""


class UnsupportedTrajectoryRecordSchemaError(ValueError):
    """Raised when a sidecar declares a schema this reader does not decode."""


class TamperedTrajectoryRecordError(ValueError):
    """Raised when a sidecar or CSV fails an integrity check against itself."""


@dataclass(frozen=True, slots=True)
class TrajectoryPhysicalContext:
    """Physical-system identity and method provenance, or its documented absence.

    Historical ``trajectory_records/v1`` sidecars never recorded a physical
    descriptor or an evaluation-method declaration (see
    ``system-context-records-design-r1.md`` section 5). Decoding such a
    sidecar must state that fact explicitly rather than inventing a
    ``system_id`` from ``atomic_configuration_id`` or from geometry: the two
    identities are never aliases of one another.

    Parameters
    ----------
    available : bool
        Whether this record carries a verified physical-system identity.
    system_id : str or None
        The recorded system content ID, required when ``available`` is
        true and forbidden otherwise.
    reason : str or None
        Human-readable explanation, required when ``available`` is false
        and forbidden otherwise.
    """

    available: bool
    system_id: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.available:
            if self.system_id is None:
                raise ValueError("available physical context requires a system_id")
            if self.reason is not None:
                raise ValueError("available physical context must not carry a reason")
        else:
            if self.system_id is not None:
                raise ValueError("unavailable physical context must not carry a system_id")
            if self.reason is None:
                raise ValueError("unavailable physical context requires a reason")


_V1_PHYSICAL_CONTEXT_UNAVAILABLE = TrajectoryPhysicalContext(
    available=False,
    reason=(
        "trajectory_records/v1 sidecars record nuclear geometry only; they contain no "
        "physical-system descriptor and no evaluation-method declaration"
    ),
)


@dataclass(frozen=True, slots=True)
class TrajectoryRecordInspection:
    """Version-normalized, read-only view of one trajectory-record artifact.

    Returned by :func:`inspect_trajectory_record` for every supported
    sidecar schema. ``physical_context`` is the single place a caller must
    check before trusting a ``system_id``; a geometry-only legacy record
    reports it as unavailable rather than silently omitting the field.
    """

    record_schema: str
    path: Path
    metadata_path: Path
    observable: str
    draw_count: int
    walker_count: int
    row_count: int
    mean: float
    variance: float
    csv_sha256: str
    atomic_configuration_id: str | None
    physical_context: TrajectoryPhysicalContext
    schema: str = TRAJECTORY_RECORD_INSPECTION_SCHEMA


def inspect_trajectory_record(metadata_path: str | Path) -> TrajectoryRecordInspection:
    """Decode one trajectory-record sidecar under its own declared schema.

    Parameters
    ----------
    metadata_path : str or pathlib.Path
        Path to a ``*.metadata.json`` sidecar written by a trajectory-record
        writer (any supported schema).

    Returns
    -------
    TrajectoryRecordInspection
        A version-normalized view. For ``trajectory_records/v1`` sidecars,
        ``physical_context.available`` is always false.

    Raises
    ------
    UnsupportedTrajectoryRecordSchemaError
        The sidecar's ``schema`` field is missing or not a schema this
        reader decodes.
    TamperedTrajectoryRecordError
        The sidecar's key set, the CSV hash, or the draw/walker grid
        disagrees with what the declared schema requires.

    Notes
    -----
    This function never infers a physical-system identity from
    ``atomic_configuration_id`` or from recorded geometry: a v1 sidecar's
    nuclear positions describe a resource consumed by some term, not a
    verified complete Hamiltonian. Experiments reach this function only
    through the read-only ``tpen.inspection`` boundary
    (``experiments/README.md``), never by importing this module directly.
    """

    metadata_path = Path(metadata_path)
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TamperedTrajectoryRecordError(f"cannot read trajectory record sidecar: {exc}") from exc
    except ValueError as exc:
        raise TamperedTrajectoryRecordError(
            f"trajectory record sidecar is not valid JSON: {metadata_path}"
        ) from exc
    if not isinstance(raw, dict):
        raise TamperedTrajectoryRecordError("trajectory record sidecar must be a JSON object")

    record_schema = raw.get("schema")
    if record_schema == TRAJECTORY_RECORD_SCHEMA:
        return _inspect_v1(metadata_path, raw)
    raise UnsupportedTrajectoryRecordSchemaError(
        f"unsupported trajectory record schema {record_schema!r} in {metadata_path}"
    )


def _inspect_v1(metadata_path: Path, metadata: dict[str, Any]) -> TrajectoryRecordInspection:
    declared_keys = set(TRAJECTORY_RECORD_METADATA_KEYS)
    actual_keys = set(metadata)
    if actual_keys != declared_keys:
        missing = sorted(declared_keys - actual_keys)
        extra = sorted(actual_keys - declared_keys)
        raise TamperedTrajectoryRecordError(
            f"trajectory record sidecar key set disagrees with {TRAJECTORY_RECORD_SCHEMA!r}: "
            f"missing={missing}, extra={extra}"
        )
    if metadata.get("row_semantics") != TRAJECTORY_RECORD_ROW_SEMANTICS:
        raise TamperedTrajectoryRecordError(
            f"trajectory record row_semantics disagrees with {TRAJECTORY_RECORD_ROW_SEMANTICS!r}"
        )

    scalars = _validate_v1_scalar_types(metadata)

    csv_path = metadata_path.parent / str(metadata["csv_filename"])
    if not csv_path.is_file():
        raise TamperedTrajectoryRecordError(f"trajectory record CSV not found: {csv_path}")
    declared_sha256 = metadata["csv_sha256"]
    actual_sha256 = _file_sha256(csv_path)
    if actual_sha256 != declared_sha256:
        raise TamperedTrajectoryRecordError(
            f"trajectory record csv_sha256 disagrees with {csv_path}: "
            f"declared={declared_sha256!r}, actual={actual_sha256!r}"
        )
    actual_byte_count = csv_path.stat().st_size
    if actual_byte_count != scalars.byte_count:
        raise TamperedTrajectoryRecordError(
            f"trajectory record byte_count disagrees with {csv_path}: "
            f"declared={scalars.byte_count}, actual={actual_byte_count}"
        )

    atoms = _decode_v1_atomic_configuration(metadata)
    declared_atomic_configuration_id = metadata["atomic_configuration_id"]
    if atoms.content_id() != declared_atomic_configuration_id:
        raise TamperedTrajectoryRecordError(
            "trajectory record atomic_configuration_id disagrees with its atomic_configuration payload"
        )
    if atoms.spatial_dim != scalars.spatial_dim:
        raise TamperedTrajectoryRecordError(
            "trajectory record atomic_configuration dimension disagrees with spatial_dim"
        )

    scan = _scan_v1_csv(
        csv_path,
        draw_count=scalars.draw_count,
        walker_count=scalars.walker_count,
        n_electrons=scalars.n_electrons,
        spatial_dim=scalars.spatial_dim,
        term_names=scalars.term_names,
    )
    if scan.row_count != scalars.row_count:
        raise TamperedTrajectoryRecordError(
            f"trajectory record row_count disagrees with its CSV grid: "
            f"declared={scalars.row_count}, actual={scan.row_count}"
        )
    if scan.finite_count != scalars.finite_count:
        raise TamperedTrajectoryRecordError(
            f"trajectory record finite_count disagrees with its CSV: "
            f"declared={scalars.finite_count}, actual={scan.finite_count}"
        )
    declared_values_content_id = metadata["observable_values_content_id"]
    if scan.values_content_id != declared_values_content_id:
        raise TamperedTrajectoryRecordError(
            "trajectory record observable_values_content_id disagrees with its CSV"
        )
    declared_mean = _as_float(metadata["mean"])
    declared_variance = _as_float(metadata["variance"])
    if not _close_or_same_nonfinite(scan.mean, declared_mean) or not _close_or_same_nonfinite(
        scan.variance, declared_variance
    ):
        raise TamperedTrajectoryRecordError(
            "trajectory record mean/variance disagrees with its CSV: "
            f"declared=({declared_mean!r}, {declared_variance!r}), "
            f"recomputed=({scan.mean!r}, {scan.variance!r})"
        )

    return TrajectoryRecordInspection(
        record_schema=TRAJECTORY_RECORD_SCHEMA,
        path=csv_path,
        metadata_path=metadata_path,
        observable=str(metadata["observable"]),
        draw_count=scalars.draw_count,
        walker_count=scalars.walker_count,
        row_count=scalars.row_count,
        mean=declared_mean,
        variance=declared_variance,
        csv_sha256=declared_sha256,
        atomic_configuration_id=str(declared_atomic_configuration_id),
        physical_context=_V1_PHYSICAL_CONTEXT_UNAVAILABLE,
    )


class _V1ScalarMetadata(NamedTuple):
    """Type/domain-validated scalar fields of a ``trajectory_records/v1`` sidecar."""

    draw_count: int
    walker_count: int
    draw_stride: int
    burn_in_draws: int
    n_electrons: int
    spatial_dim: int
    row_count: int
    finite_count: int
    byte_count: int
    term_names: tuple[str, ...]


def _require_v1_int(metadata: dict[str, Any], key: str, *, minimum: int) -> int:
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TamperedTrajectoryRecordError(f"trajectory record {key!r} must be an integer")
    if value < minimum:
        raise TamperedTrajectoryRecordError(f"trajectory record {key!r} must be >= {minimum}")
    return value


def _validate_v1_scalar_types(metadata: dict[str, Any]) -> _V1ScalarMetadata:
    """Validate every v1 scalar field's type/domain and cross-field bindings.

    Catches the exact gaps the strict v1 decoder previously missed: a
    fractional ``draw_count`` silently truncated by ``int()``, out-of-range
    ``finite_count``/``nonfinite_count``, and an ``observable``/``term_names``
    that are not well-formed strings. Row/CSV-grid bindings (geometry, byte
    count, finite mask, value digest, coordinate/term columns) are checked
    separately once the CSV has been scanned.
    """

    draw_count = _require_v1_int(metadata, "draw_count", minimum=1)
    walker_count = _require_v1_int(metadata, "walker_count", minimum=1)
    draw_stride = _require_v1_int(metadata, "draw_stride", minimum=1)
    burn_in_draws = _require_v1_int(metadata, "burn_in_draws", minimum=0)
    n_electrons = _require_v1_int(metadata, "n_electrons", minimum=1)
    spatial_dim = _require_v1_int(metadata, "spatial_dim", minimum=1)
    row_count = _require_v1_int(metadata, "row_count", minimum=0)
    finite_count = _require_v1_int(metadata, "finite_count", minimum=0)
    nonfinite_count = _require_v1_int(metadata, "nonfinite_count", minimum=0)
    byte_count = _require_v1_int(metadata, "byte_count", minimum=1)

    if row_count != draw_count * walker_count:
        raise TamperedTrajectoryRecordError(
            "trajectory record row_count disagrees with draw_count*walker_count"
        )
    if finite_count > row_count:
        raise TamperedTrajectoryRecordError("trajectory record finite_count exceeds row_count")
    if nonfinite_count != row_count - finite_count:
        raise TamperedTrajectoryRecordError(
            "trajectory record nonfinite_count disagrees with row_count - finite_count"
        )

    observable = metadata["observable"]
    if not isinstance(observable, str) or not observable.strip():
        raise TamperedTrajectoryRecordError("trajectory record observable must be a non-empty string")

    term_names = metadata["term_names"]
    if (
        not isinstance(term_names, list)
        or not term_names
        or not all(isinstance(name, str) and name.strip() for name in term_names)
        or len(set(term_names)) != len(term_names)
    ):
        raise TamperedTrajectoryRecordError(
            "trajectory record term_names must be a non-empty list of unique, non-blank strings"
        )

    for key in ("observable_values_content_id", "atomic_configuration_id"):
        value = metadata[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise TamperedTrajectoryRecordError(
                f"trajectory record {key!r} must be a lowercase sha256 digest"
            )

    return _V1ScalarMetadata(
        draw_count=draw_count,
        walker_count=walker_count,
        draw_stride=draw_stride,
        burn_in_draws=burn_in_draws,
        n_electrons=n_electrons,
        spatial_dim=spatial_dim,
        row_count=row_count,
        finite_count=finite_count,
        byte_count=byte_count,
        term_names=tuple(term_names),
    )


def _decode_v1_atomic_configuration(metadata: dict[str, Any]) -> AtomicConfiguration:
    """Decode and validate the sidecar's nuclear-geometry payload.

    Never infers an identity from the payload beyond what
    :class:`AtomicConfiguration` itself validates; the caller is responsible
    for binding the result's ``content_id()`` back to the declared
    ``atomic_configuration_id``.
    """

    payload = metadata["atomic_configuration"]
    if not isinstance(payload, dict) or set(payload) != {"positions", "charges"}:
        raise TamperedTrajectoryRecordError(
            "trajectory record atomic_configuration must be a {'positions', 'charges'} payload"
        )
    try:
        positions = torch.tensor(payload["positions"], dtype=torch.float64)
        charges = torch.tensor(payload["charges"], dtype=torch.float64)
        return AtomicConfiguration(positions=positions, charges=charges)
    except (TypeError, ValueError) as exc:
        raise TamperedTrajectoryRecordError(
            f"trajectory record atomic_configuration is not a valid nuclear geometry: {exc}"
        ) from exc


class _V1CsvScan(NamedTuple):
    """Values independently recomputed from a v1 CSV's complete grid."""

    row_count: int
    finite_count: int
    mean: float
    variance: float
    values_content_id: str


def _scan_v1_csv(
    path: Path,
    *,
    draw_count: int,
    walker_count: int,
    n_electrons: int,
    spatial_dim: int,
    term_names: Sequence[str],
) -> _V1CsvScan:
    """Scan a v1 CSV for a complete, non-duplicated draw/walker grid.

    Validates that the header is exactly the column set implied by
    ``n_electrons``/``spatial_dim``/``term_names`` (binding the declared
    coordinate/term schema to the actual CSV columns, and rejecting
    duplicate headers or ragged rows), that every row's ``finite`` flag
    agrees with its own ``local_energy``, and that ``sample_index`` matches
    the original ``draw*walker_count + walker`` sample/grid semantics.
    Returns the row/finite counts plus the local-energy mean, variance, and
    values-content-id recomputed directly from the CSV using the writer's
    own per-draw-then-total reduction order, so a caller can cross-check
    them against the sidecar's declared values without trusting either file
    alone. Unlike an exact-summation baseline, this reduction order
    reproduces (rather than crashes on) the writer's own mixed-infinity
    semantics.
    """

    expected_fieldnames = _trajectory_record_fieldnames(
        n_electrons=n_electrons, spatial_dim=spatial_dim, term_names=term_names
    )
    expected_fieldname_set = set(expected_fieldnames)

    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if header is None:
        raise TamperedTrajectoryRecordError("trajectory record CSV has no header row")
    if len(set(header)) != len(header):
        raise TamperedTrajectoryRecordError("trajectory record CSV header has duplicate column names")
    if set(header) != expected_fieldname_set:
        missing = sorted(expected_fieldname_set - set(header))
        extra = sorted(set(header) - expected_fieldname_set)
        raise TamperedTrajectoryRecordError(
            "trajectory record CSV header disagrees with its declared coordinate/term schema: "
            f"missing={missing}, extra={extra}"
        )

    extra_sentinel = "__trajectory_record_extra_columns__"
    seen: set[tuple[int, int]] = set()
    row_count = 0
    finite_count = 0
    draw_sums: dict[int, float] = {}
    draw_square_sums: dict[int, float] = {}
    draw_values: dict[int, dict[int, float]] = {draw: {} for draw in range(draw_count)}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, restkey=extra_sentinel, restval=None)
        for row in reader:
            if row.get(extra_sentinel) is not None:
                raise TamperedTrajectoryRecordError("trajectory record CSV row has extra columns")
            if any(row.get(key) is None for key in expected_fieldnames):
                raise TamperedTrajectoryRecordError("trajectory record CSV row is missing column values")
            try:
                draw = int(row["draw_index"])
                walker = int(row["walker_index"])
                sample_index = int(row["sample_index"])
                energy = _parse_v1_csv_energy(row["local_energy"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TamperedTrajectoryRecordError(f"malformed trajectory record row: {row!r}") from exc
            if row["finite"] not in ("True", "False"):
                raise TamperedTrajectoryRecordError(
                    f"trajectory record row (draw={draw}, walker={walker}) has a malformed finite flag"
                )
            declared_finite = row["finite"] == "True"
            if declared_finite != math.isfinite(energy):
                raise TamperedTrajectoryRecordError(
                    f"trajectory record row (draw={draw}, walker={walker}) finite flag "
                    "disagrees with its local_energy"
                )
            if not 0 <= draw < draw_count or not 0 <= walker < walker_count:
                raise TamperedTrajectoryRecordError(
                    f"trajectory record row (draw={draw}, walker={walker}) is outside the declared grid"
                )
            key = (draw, walker)
            if key in seen:
                raise TamperedTrajectoryRecordError(f"duplicate trajectory record row (draw={draw}, walker={walker})")
            seen.add(key)
            if sample_index != draw * walker_count + walker:
                raise TamperedTrajectoryRecordError(
                    f"trajectory record row (draw={draw}, walker={walker}) sample_index "
                    "disagrees with the original sample/grid semantics"
                )

            # Per-draw-then-total reduction, matching the writer's own
            # draw_sums/draw_square_sums reconciliation: plain float addition
            # propagates +/-inf into nan rather than raising, unlike an
            # exact-summation baseline such as `math.fsum`.
            draw_sums[draw] = draw_sums.get(draw, 0.0) + energy
            draw_square_sums[draw] = draw_square_sums.get(draw, 0.0) + energy * energy
            if declared_finite:
                finite_count += 1
            draw_values[draw][walker] = energy
            row_count += 1

    expected_rows = draw_count * walker_count
    if row_count != expected_rows:
        raise TamperedTrajectoryRecordError(
            f"trajectory record grid is incomplete: expected {expected_rows} rows, found {row_count}"
        )

    total = 0.0
    total_square = 0.0
    for draw in range(draw_count):
        total += draw_sums.get(draw, 0.0)
        total_square += draw_square_sums.get(draw, 0.0)
    mean = total / row_count
    variance = total_square / row_count - mean * mean
    if math.isfinite(variance) and variance < 0.0:
        variance = 0.0

    values_digest = hashlib.sha256()
    values_digest.update(str((draw_count, walker_count)).encode("utf-8"))
    for draw in range(draw_count):
        ordered = torch.tensor(
            [draw_values[draw][walker] for walker in range(walker_count)], dtype=torch.float64
        )
        values_digest.update(ordered.contiguous().numpy().astype("<f8", copy=False).tobytes())

    return _V1CsvScan(
        row_count=row_count,
        finite_count=finite_count,
        mean=mean,
        variance=variance,
        values_content_id=values_digest.hexdigest(),
    )


def _as_float(value: object) -> float:
    if isinstance(value, str):
        if value == "inf":
            return math.inf
        if value == "-inf":
            return -math.inf
        if value == "nan":
            return math.nan
        raise TamperedTrajectoryRecordError(f"trajectory record non-finite field has unknown text {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    raise TamperedTrajectoryRecordError(f"trajectory record numeric field has unexpected type {type(value).__name__}")


def _parse_v1_csv_energy(text: str) -> float:
    """Parse one v1 CSV ``local_energy``/term cell, written by `_float_or_text`.

    Unlike `_as_float` (which only ever sees JSON-typed metadata scalars:
    a finite ``int``/``float`` or one of the three non-finite sentinel
    strings), every CSV cell is plain text, including finite values such as
    ``"1.0"``.
    """

    if text in ("inf", "-inf", "nan"):
        return _as_float(text)
    try:
        return float(text)
    except ValueError as exc:
        raise TamperedTrajectoryRecordError(f"trajectory record CSV field is not numeric: {text!r}") from exc


def _write_metadata(artifact: TrajectoryRecordArtifact) -> None:
    atoms = artifact.atomic_configuration
    metadata = {
        "schema": TRAJECTORY_RECORD_SCHEMA,
        "row_semantics": TRAJECTORY_RECORD_ROW_SEMANTICS,
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
        "atomic_configuration_id": atoms.content_id(),
        "atomic_configuration": _atomic_geometry_payload(atoms),
    }
    with artifact.metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _atomic_geometry_payload(atoms: AtomicConfiguration) -> dict[str, list]:
    """Return the JSON-serializable nuclear geometry payload for a sidecar."""

    return {
        "positions": atoms.positions.detach().to(torch.float64).cpu().tolist(),
        "charges": atoms.charges.detach().to(torch.float64).cpu().tolist(),
    }


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


def _close_or_same_nonfinite(left: float, right: float, *, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """Compare two floats tolerantly, treating matching non-finite values as equal.

    Used only to cross-check a reader's independent recomputation against a
    sidecar's declared value; two different summation orders (this module's
    plain accumulation versus the writer's torch reduction) can legitimately
    differ in the last few bits without either being tampered.
    """

    if math.isfinite(left) and math.isfinite(right):
        return math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol)
    return _same_float(left, right)


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
    "TRAJECTORY_RECORD_INSPECTION_SCHEMA",
    "TRAJECTORY_RECORD_METADATA_KEYS",
    "TRAJECTORY_RECORD_ROW_SEMANTICS",
    "TRAJECTORY_RECORD_SCHEMA",
    "TamperedTrajectoryRecordError",
    "TrajectoryPhysicalContext",
    "TrajectoryRecordArtifact",
    "TrajectoryRecordBatch",
    "TrajectoryRecordInspection",
    "TrajectoryRecordStreamWriter",
    "UnsupportedTrajectoryRecordSchemaError",
    "inspect_trajectory_record",
]
