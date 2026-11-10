"""Optional, integrity-checked raw training-history recording.

Training history describes a changing model and is therefore deliberately kept
separate from frozen evaluation records. This module owns the versioned
``training_history`` format: typed row batches, one-partition-at-a-time
writing, and manifest-selected reading/validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
from omegaconf import OmegaConf

from tpen.artifacts import RunContext
from tpen.checkpoint import CheckpointRestored
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.training.events import TrainingCompleted, TrainingIterationCompleted
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence

LEGACY_SCHEMA = "training_history/v1"
SCHEMA = "training_history/v2"
MANIFEST_FILENAME = "manifest.json"


def _array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Return an owned, one-dimensional CPU NumPy array of ``dtype``."""

    result = np.asarray(value, dtype=dtype)
    return np.array(result, dtype=dtype, copy=True).reshape(-1)


@dataclass(frozen=True)
class TrainingBatchRecord:
    """One pre-update training batch, detached from the Torch graph.

    Parameters
    ----------
    iteration_step : int
        Zero-based trainer iteration coordinate.
    completed_updates : int
        Number of optimizer updates completed after this iteration.
    update_applied : bool
        Whether this iteration applied an optimizer update.
    positions : ndarray
        Coordinates with shape ``[n_walkers, n_electrons, spatial_dim]``.
    local_energy, logabs, sign : ndarray
        One value per walker.
    term_energies : mapping of str to ndarray
        Per-term values from the trainer's existing local-energy result.
    """

    iteration_step: int
    completed_updates: int
    update_applied: bool
    walker_index: np.ndarray
    positions: np.ndarray
    local_energy: np.ndarray
    term_energies: Mapping[str, np.ndarray]
    logabs: np.ndarray
    sign: np.ndarray
    finite: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "iteration_step", int(self.iteration_step))
        object.__setattr__(self, "completed_updates", int(self.completed_updates))
        object.__setattr__(self, "update_applied", bool(self.update_applied))
        positions = np.array(self.positions, dtype=np.float64, copy=True)
        if positions.ndim != 3:
            raise ValueError("positions must have shape [walkers, electrons, dimension]")
        n = positions.shape[0]
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "walker_index", _array(self.walker_index, dtype=np.dtype("<i8")))
        object.__setattr__(self, "local_energy", _array(self.local_energy, dtype=np.dtype("<f8")))
        object.__setattr__(self, "logabs", _array(self.logabs, dtype=np.dtype("<f8")))
        object.__setattr__(self, "sign", _array(self.sign, dtype=np.dtype("<f8")))
        terms = {str(name): _array(value, dtype=np.dtype("<f8")) for name, value in self.term_energies.items()}
        object.__setattr__(self, "term_energies", terms)
        finite = np.asarray(self.finite, dtype=np.bool_).reshape(-1).copy()
        object.__setattr__(self, "finite", finite)
        lengths = {len(self.walker_index), len(self.local_energy), len(self.logabs), len(self.sign), len(finite), n}
        lengths.update(len(value) for value in terms.values())
        if len(lengths) != 1 or next(iter(lengths)) == 0:
            raise ValueError("all training batch fields must have one non-empty walker length")
        if not np.array_equal(self.walker_index, np.arange(n, dtype=np.int64)):
            raise ValueError("walker_index must be the complete ordered grid")
        expected_finite = (
            np.isfinite(self.positions).all(axis=(1, 2))
            & np.isfinite(self.local_energy)
            & np.isfinite(self.logabs)
            & np.isfinite(self.sign)
        )
        expected_finite &= np.isfinite(self.sign)
        for value in terms.values():
            expected_finite &= np.isfinite(value)
        if not np.array_equal(finite, expected_finite):
            raise ValueError("finite must equal the required-field finiteness mask")

    @property
    def n_walkers(self) -> int:
        """Return the number of walker rows."""

        return int(self.local_energy.size)

    @property
    def term_names(self) -> tuple[str, ...]:
        """Return term names in deterministic order."""

        return tuple(self.term_energies)


def _digest_arrays(record: TrainingBatchRecord) -> str:
    """Hash canonical, sorted columns independently of NumPy container format."""

    digest = hashlib.sha256()
    columns: dict[str, np.ndarray] = {
        "walker_index": record.walker_index,
        "positions": record.positions,
        "local_energy": record.local_energy,
        "logabs": record.logabs,
        "sign": record.sign,
        "finite": record.finite,
    }
    columns.update({f"term/{name}": value for name, value in record.term_energies.items()})
    for name in sorted(columns):
        value = np.ascontiguousarray(columns[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    """Hash a closed file in bounded-size chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrainingBatchPartitionWriter:
    """Collect and publish exactly one immutable NumPy partition."""

    def __init__(
        self,
        root: str | Path,
        *,
        partition_index: int,
        planned_steps: list[int],
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.partition_index = int(partition_index)
        self.planned_steps = [int(step) for step in planned_steps]
        self._records: list[TrainingBatchRecord] = []
        self._seen: set[int] = set()
        self._provenance = dict(provenance or _empty_provenance())
        self._sampler_stats: list[dict[str, Any]] = []
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def memory_bytes(self) -> int:
        """Return the bytes currently retained by this open partition."""

        return sum(int(value.nbytes) for record in self._records for value in _record_arrays(record))

    def append(self, record: TrainingBatchRecord, *, sampler_stats: Mapping[str, Any] | None = None) -> None:
        """Append one complete retained update to this partition."""

        if record.iteration_step in self._seen:
            raise ValueError(f"duplicate training update {record.iteration_step}")
        if record.iteration_step not in self.planned_steps:
            raise ValueError(f"training update {record.iteration_step} is outside planned partition")
        if self._records and record.n_walkers != self._records[0].n_walkers:
            raise ValueError("walker count changed within a training partition")
        if self._records and record.term_names != self._records[0].term_names:
            raise ValueError("Hamiltonian terms changed within a training partition")
        self._records.append(record)
        self._seen.add(record.iteration_step)
        if sampler_stats is not None:
            self._sampler_stats.append(dict(sampler_stats))

    def publish(self) -> dict[str, Any]:
        """Write one compressed bundle, its sidecar, and manifest entry."""

        if set(self._seen) != set(self.planned_steps):
            missing = sorted(set(self.planned_steps) - self._seen)
            raise ValueError(f"training partition is missing updates: {missing}")
        stem = f"partition-{self.partition_index:05d}"
        compressed_path = self.root / f"{stem}.npz"
        sidecar_path = self.root / f"{stem}.json"
        if compressed_path.exists() or sidecar_path.exists():
            raise FileExistsError(
                f"published training partition {self.partition_index} already exists; resume cannot rewrite it"
            )
        started = time.perf_counter()
        arrays = _concatenate_records(self._records)
        temp_path = self.root / f".{stem}.{os.getpid()}.tmp"
        with temp_path.open("wb") as handle:
            np.savez(handle, **arrays)
        uncompressed_bytes = temp_path.stat().st_size
        uncompressed_sha256 = _sha256(temp_path)
        compressed_temp = self.root / f".{stem}.{os.getpid()}.compressed.tmp"
        with compressed_temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        compressed_bytes = compressed_temp.stat().st_size
        os.replace(compressed_temp, compressed_path)
        # The uncompressed bundle is a measurement arm only.  The selected
        # compressed format is the sole published data object.
        temp_path.unlink()
        sidecar = {
            "schema": SCHEMA,
            "source": "tpen.callback.RawTrainingBatchRecorder",
            "capture_semantics": "pre_update_model_forward",
            "writer_version": SCHEMA,
            "partition_index": self.partition_index,
            "planned_updates": self.planned_steps,
            "observed_updates": sorted(self._seen),
            "rows_per_update": int(self._records[0].n_walkers),
            "walker_count": int(self._records[0].n_walkers),
            "term_names": list(self._records[0].term_names),
            "row_count": int(len(self._records) * self._records[0].n_walkers),
            "columns": sorted(arrays),
            "logical_sha256": _digest_arrays_concatenated(self._records),
            "compressed_filename": compressed_path.name,
            "uncompressed_filename": None,
            "compressed_byte_count": compressed_bytes,
            "uncompressed_byte_count": uncompressed_bytes,
            "compression_ratio": compressed_bytes / uncompressed_bytes,
            "compressed_sha256": _sha256(compressed_path),
            "uncompressed_sha256": uncompressed_sha256,
            "selected_filename": compressed_path.name,
            "selection_reason": "compressed bundle selected to reduce retained training-history bytes; the uncompressed arm is measured then removed",
            "write_time_sec": time.perf_counter() - started,
            "provenance": {
                **self._provenance,
                "acceptance_proposal_stats": self._sampler_stats,
            },
        }
        _write_json_exclusive(sidecar_path, sidecar)
        sidecar_sha256 = _sha256(sidecar_path)
        manifest_path = self.root / MANIFEST_FILENAME
        previous_hash = _sha256(manifest_path) if manifest_path.exists() else None
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema": SCHEMA, "partitions": []}
        manifest["schema"] = SCHEMA
        manifest["partitions"].append(
            {
                "sidecar": sidecar_path.name,
                "sidecar_sha256": sidecar_sha256,
                "parent_manifest_sha256": previous_hash,
            }
        )
        _atomic_json(manifest_path, manifest)
        self._records.clear()
        self._seen.clear()
        self._sampler_stats.clear()
        return sidecar


def _empty_provenance() -> dict[str, Any]:
    """Return the typed provenance envelope used by direct writer callers."""

    return {
        "run_id": None,
        "config_hash": None,
        "source_revision": None,
        "sampler_settings": {},
        "acceptance_proposal_stats": [],
    }


def _context_provenance(context: RunContext, sampler_stats: Any) -> dict[str, Any]:
    """Build small JSON-safe provenance from the run and sampler domains."""

    config = OmegaConf.to_container(context.cfg, resolve=True)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    settings: dict[str, Any] = {}
    if sampler_stats is not None:
        for name in ("n_walkers", "burn_in", "n_steps", "proposal_scale", "seed"):
            value = getattr(sampler_stats, name)
            settings[name] = value
        settings["geometry"] = dict(sampler_stats.geometry)
    return {
        "run_id": context.metadata.run_id,
        "config_hash": config_hash,
        "source_revision": context.metadata.git_commit,
        "sampler_settings": settings,
    }


def _sampler_provenance(sampler_stats: Any) -> dict[str, Any]:
    """Return the per-step acceptance/proposal subset of typed sampler stats."""

    if sampler_stats is None:
        return {}
    return {
        "iteration_acceptance_rate": float(sampler_stats.acceptance_rate),
        "proposal_scale": float(sampler_stats.proposal_scale),
    }


def _record_arrays(record: TrainingBatchRecord) -> list[np.ndarray]:
    """Return all NumPy storage owned by one record."""

    return [record.walker_index, record.positions, record.local_energy, record.logabs, record.sign, record.finite, *record.term_energies.values()]


def _concatenate_records(records: list[TrainingBatchRecord]) -> dict[str, np.ndarray]:
    """Build a canonical column bundle from one partition."""

    first = records[0]
    arrays: dict[str, np.ndarray] = {
        "iteration_step": np.concatenate([np.full(record.n_walkers, record.iteration_step, dtype="<i8") for record in records]),
        "completed_updates": np.concatenate([np.full(record.n_walkers, record.completed_updates, dtype="<i8") for record in records]),
        "update_applied": np.concatenate([np.full(record.n_walkers, record.update_applied, dtype=np.bool_) for record in records]),
        "walker_index": np.concatenate([record.walker_index for record in records]),
        "positions": np.concatenate([record.positions for record in records]),
        "local_energy": np.concatenate([record.local_energy for record in records]),
        "logabs": np.concatenate([record.logabs for record in records]),
        "sign": np.concatenate([record.sign for record in records]),
        "finite": np.concatenate([record.finite for record in records]),
    }
    arrays.update({f"term/{name}": np.concatenate([record.term_energies[name] for record in records]) for name in first.term_names})
    return arrays


def _digest_arrays_concatenated(records: list[TrainingBatchRecord]) -> str:
    """Hash the exact canonical arrays that will be stored in the bundle."""

    arrays = _concatenate_records(records)
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Write a sidecar exactly once, preserving a conflicting attempt."""

    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a manifest only after its complete temporary contents exist."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


class TrainingBatchReader:
    """Read only manifest-selected, integrity-validated partitions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.provenance: dict[int, Mapping[str, Any]] = {}

    def read(self, *, start_step: int | None = None, end_step: int | None = None) -> dict[str, np.ndarray]:
        """Return rows in the requested iteration range after full validation."""

        manifest_path = self.root / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") not in {LEGACY_SCHEMA, SCHEMA}:
            raise ValueError("training history manifest has the wrong schema")
        selected: list[dict[str, np.ndarray]] = []
        self.provenance = {}
        identities: set[tuple[int, int]] = set()
        for partition_number, entry in enumerate(manifest.get("partitions", [])):
            parent_hash = entry.get("parent_manifest_sha256")
            expected_parent = None
            if partition_number > 0:
                predecessor = {"schema": manifest["schema"], "partitions": manifest["partitions"][:partition_number]}
                expected_parent = _canonical_hash(predecessor)
            if parent_hash != expected_parent:
                raise ValueError("training manifest partition lineage is broken")
            sidecar_path = self.root / entry["sidecar"]
            expected_sidecar_hash = entry.get("sidecar_sha256")
            if not isinstance(expected_sidecar_hash, str):
                raise ValueError("training manifest entry is missing the sidecar digest")
            if not sidecar_path.exists() or _sha256(sidecar_path) != expected_sidecar_hash:
                raise ValueError(
                    "training partition sidecar digest does not match its manifest entry"
                    " (partition integrity failure)"
                )
            sidecar = json.loads(sidecar_path.read_text())
            _validate_sidecar(self.root, sidecar, entry=entry)
            if sidecar["schema"] == SCHEMA:
                self.provenance[partition_number] = dict(sidecar["provenance"])
            arrays = dict(np.load(self.root / sidecar["compressed_filename"], allow_pickle=False))
            _validate_bundle(arrays, sidecar)
            for step, walker in zip(arrays["iteration_step"], arrays["walker_index"]):
                identity = (int(step), int(walker))
                if identity in identities:
                    raise ValueError("duplicate (iteration, walker) identity across training partitions")
                identities.add(identity)
            mask = np.ones(len(arrays["iteration_step"]), dtype=bool)
            if start_step is not None:
                mask &= arrays["iteration_step"] >= int(start_step)
            if end_step is not None:
                mask &= arrays["iteration_step"] <= int(end_step)
            selected.append({name: value[mask] for name, value in arrays.items()})
        if not selected:
            return {}
        return {name: np.concatenate([part[name] for part in selected]) for name in selected[0]}

    def validate(self) -> None:
        """Validate every manifest-selected partition without returning rows."""

        self.read()


def _validate_sidecar(root: Path, sidecar: Mapping[str, Any], *, entry: Mapping[str, Any]) -> None:
    """Refuse missing, altered, or internally inconsistent partition envelopes."""

    required = {"schema", "planned_updates", "observed_updates", "columns", "logical_sha256", "compressed_filename", "compressed_sha256", "row_count", "walker_count"}
    missing = required - set(sidecar)
    if missing:
        raise ValueError(f"training sidecar is missing keys: {sorted(missing)}")
    if sidecar["schema"] not in {LEGACY_SCHEMA, SCHEMA} or sidecar["observed_updates"] != sorted(sidecar["observed_updates"]):
        raise ValueError("training sidecar schema or update ordering is invalid")
    if sidecar["schema"] == SCHEMA and not isinstance(sidecar.get("provenance"), Mapping):
        raise ValueError("training sidecar is missing typed provenance")
    if entry.get("sidecar") != Path(str(sidecar.get("compressed_filename", ""))).with_suffix(".json").name:
        raise ValueError("training manifest and sidecar disagree about the partition")
    if sidecar.get("selected_filename") != sidecar["compressed_filename"]:
        raise ValueError("training sidecar selected bundle is not the compressed bundle")
    if sidecar["planned_updates"] != sidecar["observed_updates"]:
        raise ValueError("training sidecar has missing or duplicate updates")
    path = root / str(sidecar["compressed_filename"])
    if not path.exists() or _sha256(path) != sidecar["compressed_sha256"]:
        raise ValueError("training partition physical hash does not match its sidecar")


def _validate_bundle(arrays: Mapping[str, np.ndarray], sidecar: Mapping[str, Any]) -> None:
    """Validate row identity, finiteness, and logical digest of a loaded bundle."""

    if sorted(arrays) != sorted(sidecar["columns"]):
        raise ValueError("training partition columns disagree with its sidecar")
    required = {"iteration_step", "completed_updates", "update_applied", "walker_index", "positions", "local_energy", "logabs", "sign", "finite"}
    if not required.issubset(arrays):
        raise ValueError("training partition is missing a required column")
    n = len(arrays["iteration_step"])
    if n != int(sidecar["row_count"]):
        raise ValueError("training partition row count disagrees with its sidecar")
    walkers = int(sidecar["walker_count"])
    for step in sidecar["observed_updates"]:
        mask = arrays["iteration_step"] == step
        if int(mask.sum()) != walkers or not np.array_equal(arrays["walker_index"][mask], np.arange(walkers, dtype=np.int64)):
            raise ValueError("training partition has a missing or duplicate (update, walker) pair")
    finite = np.isfinite(arrays["positions"]).all(axis=(1, 2)) & np.isfinite(arrays["local_energy"]) & np.isfinite(arrays["logabs"]) & np.isfinite(arrays["sign"])
    for name in arrays:
        if name.startswith("term/"):
            finite &= np.isfinite(arrays[name])
    if not np.array_equal(arrays["finite"], finite):
        raise ValueError("training partition finite flag is inconsistent")
    if not bool(finite.all()):
        raise ValueError("training partition contains a nonfinite required field")
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    if digest.hexdigest() != sidecar["logical_sha256"]:
        raise ValueError("training partition logical digest does not match its sidecar")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    """Hash a manifest prefix using the writer's canonical JSON encoding."""

    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    return hashlib.sha256(encoded + b"\n").hexdigest()


class RawTrainingBatchRecorder(StatefulCallback[TrainerState]):
    """Optionally capture pre-update batches as NumPy training-history partitions.

    The manifest commits each closed sidecar's SHA-256 in ``sidecar_sha256``;
    readers verify that digest before trusting sidecar provenance or bundle
    digests.  Because partition plans are deterministic, this recorder only
    accepts the full cadence (``probability=1`` and ``max_calls=None``).

    A restored ``train_resume`` lineage is refused before capture or file
    publication. Full recorder resume semantics are deferred to item
    ``7eaf9e0c``; accepting a resumed callback here would risk appending a
    history whose lineage cannot be proven.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, path: str | Path, *, partition_size_steps: int = 1000, **kwargs: Any) -> None:
        cadence = pop_step_cadence(kwargs)
        if cadence.probability != 1.0:
            raise ValueError("RawTrainingBatchRecorder requires probability=1.0")
        if cadence.max_calls is not None:
            raise ValueError("RawTrainingBatchRecorder requires max_calls=None")
        if partition_size_steps < 1:
            raise ValueError("partition_size_steps must be positive")
        super().__init__(typed_groups=(
            SubscriptionGroup(
                selectors=(Subscription.of(TrainingIterationCompleted), Subscription.of(TrainingCompleted))
            ),
            SubscriptionGroup(selectors=(Subscription.of(CheckpointRestored),), stateless=True),
        ), **kwargs)
        self.path = Path(path)
        self.partition_size_steps = int(partition_size_steps)
        self._steps = StepCadenceGate(cadence)
        self._partition: TrainingBatchPartitionWriter | None = None
        self.last_measurement: dict[str, Any] | None = None
        self._restored = False
        self._first_iteration_seen = False

    def handle_stateless_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Record that this callback's lineage was restored, not fresh."""

        del context
        if isinstance(occurrence.event, CheckpointRestored):
            self._restored = True

    def handle_occurrence_impl(self, occurrence: Occurrence[TypedEvent], context: RunContext, state: TrainerState) -> None:
        """Capture the state produced by the pre-update model at a scheduled step."""

        if self._restored:
            raise ValueError(
                "RawTrainingBatchRecorder refuses restored lineage before capture "
                "or publication [unsupported-callback]"
            )
        event = occurrence.event
        if isinstance(event, TrainingCompleted):
            self._publish(context, final=True)
            return
        if not isinstance(event, TrainingIterationCompleted):
            return
        if not self._first_iteration_seen:
            self._first_iteration_seen = True
            next_iteration = getattr(state.trainer, "next_iteration", None)
            nonzero_restored_cursor = (
                next_iteration is not None and int(next_iteration) != 0
            )
            if not nonzero_restored_cursor and next_iteration is None:
                nonzero_restored_cursor = (
                    event.iteration.step != 0
                    and int(getattr(state.trainer, "completed_updates", 0)) != 0
                )
            if nonzero_restored_cursor:
                raise ValueError(
                    "RawTrainingBatchRecorder refuses restored lineage before capture "
                    "or publication [unsupported-callback]"
                )
        if not self._steps.should_run(event.iteration.step):
            return
        if state.batch is None or state.local_energy is None or state.wavefunction_output is None:
            raise ValueError("raw training recorder requires batch, local energy, and wavefunction output")
        if state.local_energy_terms is None:
            raise ValueError("raw training recorder requires trainer return_terms=True")
        batch = state.batch
        positions = batch.positions.detach().cpu().numpy().reshape(-1, batch.n_electrons, batch.spatial_dim)
        local_energy = state.local_energy.detach().cpu().numpy().reshape(-1)
        output = state.wavefunction_output
        record = TrainingBatchRecord(
            iteration_step=event.iteration.step,
            completed_updates=int(getattr(state.trainer, "completed_updates", 0)),
            update_applied=state.optimizer_step,
            walker_index=np.arange(len(local_energy), dtype=np.int64),
            positions=positions,
            local_energy=local_energy,
            term_energies={name: value.detach().cpu().numpy().reshape(-1) for name, value in state.local_energy_terms.items()},
            logabs=output.logabs.detach().cpu().numpy().reshape(-1),
            sign=output.sign.detach().cpu().numpy().reshape(-1),
            finite=(
                np.isfinite(positions).all(axis=(1, 2))
                & np.isfinite(local_energy)
                & np.isfinite(output.logabs.detach().cpu().numpy().reshape(-1))
                & np.isfinite(output.sign.detach().cpu().numpy().reshape(-1))
                & np.logical_and.reduce(
                    [
                        np.isfinite(value.detach().cpu().numpy().reshape(-1))
                        for value in state.local_energy_terms.values()
                    ]
                )
            ),
        )
        index = record.iteration_step // self.partition_size_steps
        if self._partition is None or self._partition.partition_index != index:
            if self._partition is not None:
                self._publish(context)
            lower = index * self.partition_size_steps
            upper = lower + self.partition_size_steps
            cadence_start = self._steps.cadence.start
            every = self._steps.cadence.every_n
            first = cadence_start + max(0, (lower - cadence_start + every - 1) // every) * every
            planned = list(range(first, upper, every)) if first < upper else []
            self._partition = TrainingBatchPartitionWriter(
                self.path,
                partition_index=index,
                planned_steps=planned,
                provenance=_context_provenance(context, state.sampler_stats),
            )
        self._partition.append(record, sampler_stats=_sampler_provenance(state.sampler_stats))
        if self._partition.planned_steps and record.iteration_step == self._partition.planned_steps[-1]:
            self._publish(context)

    def _publish(self, context: RunContext, *, final: bool = False) -> None:
        """Publish the open partition and emit only measurement metadata."""

        if self._partition is None:
            return
        if final:
            self._partition.planned_steps = sorted(self._partition._seen)
        self.last_measurement = self._partition.publish()
        context.log({key: value for key, value in self.last_measurement.items() if isinstance(value, (int, float, bool))}, namespace="train/training_history")
        self._partition = None

    def close(self, context: RunContext) -> None:
        """Publish a complete open partition at an explicit run boundary."""

        self._publish(context, final=True)

    def _reset_typed_state(self) -> None:
        """Forget restore provenance when a new run context is delivered."""

        self._restored = False
        self._first_iteration_seen = False


TrainingBatchRecorder = RawTrainingBatchRecorder
TrainingHistoryReader = TrainingBatchReader
TrainingBatchValidator = TrainingBatchReader

__all__ = ["RawTrainingBatchRecorder", "TrainingBatchRecorder", "TrainingBatchPartitionWriter", "TrainingBatchReader", "TrainingHistoryReader", "TrainingBatchValidator", "TrainingBatchRecord", "SCHEMA"]
