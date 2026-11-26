"""Optional, integrity-checked raw training-history recording.

Training history describes a changing model and is therefore deliberately kept
separate from frozen evaluation records. This module owns the versioned
``training_history`` format: typed row batches, one-partition-at-a-time
writing, and manifest-selected reading/validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from typing import Any, ClassVar, Mapping

import numpy as np
from omegaconf import OmegaConf

from tpen.artifacts import RunContext
from tpen.checkpoint import CheckpointRestored, checkpoint_ref
from tpen.events import DomainState, Started, started
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.training.events import TrainingCompleted, TrainingIteration, TrainingIterationCompleted
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence

LEGACY_SCHEMA = "training_history/v1"
SCHEMA = "training_history/v2"
V3_SCHEMA = "training_history/v3"
MANIFEST_FILENAME = "manifest.json"
HISTORY_FILENAME = "history.json"
EVENTS_FILENAME = "events.jsonl"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_V3_COLUMNS = (
    "iteration_step",
    "completed_updates",
    "update_applied",
    "walker_index",
    "positions",
    "local_energy",
    "logabs",
    "sign",
    "finite",
)


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


def _strict_json_loads(data: bytes | str) -> Any:
    """Decode JSON without accepting duplicate keys, NaN, or Infinity."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"non-finite JSON number: {value}")
        return result

    return json.loads(
        data,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


def _pretty_json(value: Any) -> bytes:
    """Return the approved J(x) encoding used by immutable v3 JSON objects."""

    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def _compact_json(value: Any) -> bytes:
    """Return the approved L(x) encoding used by v3 event lines."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def _require_keys(value: Mapping[str, Any], required: set[str], *, optional: set[str] | None = None) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("v3 object must be a JSON object")
    allowed = required | (optional or set())
    if not required <= set(value) or not set(value) <= allowed:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - allowed)
        raise ValueError(f"v3 object keys invalid (missing={missing}, extra={extra})")


def _int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"v3 {name} must be an integer >= {minimum}")
    return value


def _hash(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"v3 {name} must be a lowercase SHA-256 digest")
    return value


def _safe_relative_path(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"v3 {name} must be a relative POSIX path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"v3 {name} escapes its owning history root")
    return value


def _reject_symlink_components(root: Path, relative: str, *, name: str) -> None:
    """Reject a path whose directory or file component is a symlink."""

    current = root
    try:
        root_stat = current.lstat()
    except OSError as exc:
        raise ValueError(f"v3 {name} owning history root is unavailable") from exc
    if os.path.islink(current) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"v3 {name} owning history root is not a real directory")
    for component in Path(relative).parts:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ValueError(f"v3 {name} path is missing") from exc
        if os.path.islink(current):
            raise ValueError(f"v3 {name} path contains a symlink")


def _finite_number(value: Any, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"v3 {name} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"v3 {name} must be a finite number >= {minimum}")
    return result


def _validate_v3_contract(contract: Any) -> None:
    if not isinstance(contract, Mapping):
        raise ValueError("v3 contract must be an object")
    required = {
        "coordinate", "start_step", "every_n_steps", "partition_size_steps",
        "walker_count", "n_electrons", "spatial_dim", "term_names", "capture_semantics",
    }
    _require_keys(contract, required)
    if contract["coordinate"] != "iteration_step" or contract["capture_semantics"] != "pre_update_model_forward":
        raise ValueError("v3 contract coordinate or capture semantics is incompatible")
    for key, minimum in (
        ("start_step", 0), ("every_n_steps", 1), ("partition_size_steps", 1),
        ("walker_count", 1), ("n_electrons", 0), ("spatial_dim", 1),
    ):
        _int(contract[key], name=key, minimum=minimum)
    terms = contract["term_names"]
    if (
        not isinstance(terms, list)
        or any(not isinstance(term, str) or not term for term in terms)
        or terms != sorted(set(terms))
    ):
        raise ValueError("v3 contract term_names must be sorted, unique, and non-empty strings")


def _validate_v3_header(raw: bytes, header: Any) -> dict[str, Any]:
    if not isinstance(header, Mapping):
        raise ValueError("v3 history header must be an object")
    _require_keys(header, {"schema", "stream_id", "run_id", "origin", "contract"})
    if raw != _pretty_json(header):
        raise ValueError("v3 history.json is not canonical immutable JSON")
    if header["schema"] != V3_SCHEMA or not isinstance(header["stream_id"], str) or not _UUID_RE.fullmatch(header["stream_id"]):
        raise ValueError("v3 history header has an invalid schema or stream_id")
    if not isinstance(header["run_id"], str) or not header["run_id"]:
        raise ValueError("v3 run_id must be a non-empty string")
    if header["origin"] not in {"fresh", "train_resume"}:
        raise ValueError("v3 history origin is invalid")
    _validate_v3_contract(header["contract"])
    return dict(header)


def _v3_event_digest(header_hash: str, line: bytes) -> str:
    return hashlib.sha256(bytes.fromhex(header_hash) + line).hexdigest()


@dataclass
class _V3Node:
    root: Path
    header: dict[str, Any]
    events: list[dict[str, Any]]
    digests: list[str]
    fragments: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]]
    seals: list[dict[str, Any]]
    effective_seals: list[dict[str, Any]]
    completion: dict[str, int] | None
    rows: dict[str, np.ndarray]
    active_steps: set[int]
    active_identities: set[tuple[int, int]]
    provenance: list[dict[str, Any]]


def _v3_event_shape(event: Mapping[str, Any], index: int) -> None:
    event_type = event.get("type")
    common = {"type", "previous_event_sha256"}
    shapes = {
        "genesis": common,
        "resume": common | {"parent", "checkpoint", "supersede_from_step"},
        "fragment": common | {"group_index", "start_step", "end_step", "sidecar", "sidecar_sha256"},
        "window_seal": common | {"group_index", "start_step", "end_step"},
        "complete": common | {"next_iteration", "completed_updates"},
    }
    if event_type not in shapes:
        raise ValueError("v3 event type is invalid")
    _require_keys(event, shapes[event_type])
    if index == 0 and event_type not in {"genesis", "resume"}:
        raise ValueError("v3 event[0] must be genesis or resume")
    if index > 0 and event_type in {"genesis", "resume"}:
        raise ValueError("v3 origin event may occur only at event[0]")
    if not (event["previous_event_sha256"] is None if index == 0 else isinstance(event["previous_event_sha256"], str)):
        raise ValueError("v3 event previous_event_sha256 has the wrong type")
    if index > 0:
        _hash(event["previous_event_sha256"], name="previous_event_sha256")


def _v3_array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_v3_sidecar(
    root: Path,
    header: Mapping[str, Any],
    contract: Mapping[str, Any],
    event: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sidecar_rel = _safe_relative_path(event["sidecar"], name="sidecar")
    sidecar_name = Path(sidecar_rel).name
    step_from_name = re.fullmatch(r"step-(\d{12,})\.json", sidecar_name)
    if Path(sidecar_rel).parent.as_posix() != "fragments" or step_from_name is None or int(step_from_name.group(1)) != int(event["start_step"]):
        raise ValueError("v3 fragment sidecar does not use the approved step naming")
    _reject_symlink_components(root, sidecar_rel, name="sidecar")
    sidecar_path = root / sidecar_rel
    if not sidecar_path.is_file() or _sha256(sidecar_path) != event["sidecar_sha256"]:
        raise ValueError("v3 sidecar is missing or its hash does not match the fragment event")
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar = _strict_json_loads(sidecar_bytes)
    if sidecar_bytes != _pretty_json(sidecar) or not isinstance(sidecar, Mapping):
        raise ValueError("v3 sidecar is not canonical JSON")
    required = {
        "schema", "source", "capture_semantics", "stream_id", "contract_sha256", "group_index",
        "start_step", "end_step", "walker_count", "row_count", "term_names", "columns",
        "compressed_filename", "compressed_sha256", "logical_sha256", "provenance",
    }
    _require_keys(sidecar, required, optional={"measurements"})
    if sidecar["schema"] != V3_SCHEMA or sidecar["source"] != "tpen.callback.RawTrainingBatchRecorder" or sidecar["capture_semantics"] != "pre_update_model_forward":
        raise ValueError("v3 sidecar source or schema is incompatible")
    if sidecar["stream_id"] != header["stream_id"]:
        raise ValueError("v3 sidecar stream_id does not match its owning header")
    if sidecar["contract_sha256"] != hashlib.sha256(_pretty_json(contract)).hexdigest():
        raise ValueError("v3 sidecar contract hash does not match its owning header")
    step = _int(event["start_step"], name="fragment start_step")
    if event["end_step"] != step + 1 or sidecar["start_step"] != step or sidecar["end_step"] != step + 1:
        raise ValueError("v3 fragment interval is invalid")
    _int(sidecar["group_index"], name="sidecar group_index")
    _int(sidecar["start_step"], name="sidecar start_step")
    _int(sidecar["end_step"], name="sidecar end_step", minimum=1)
    _int(sidecar["walker_count"], name="sidecar walker_count", minimum=1)
    _int(sidecar["row_count"], name="sidecar row_count", minimum=1)
    if sidecar["group_index"] != event["group_index"] or event["group_index"] != step // int(contract["partition_size_steps"]):
        raise ValueError("v3 fragment group_index is invalid")
    if sidecar["walker_count"] != contract["walker_count"] or sidecar["row_count"] != contract["walker_count"]:
        raise ValueError("v3 fragment row count disagrees with its contract")
    if sidecar["term_names"] != contract["term_names"]:
        raise ValueError("v3 fragment term_names disagree with its contract")
    columns = list(_V3_COLUMNS) + [f"term/{name}" for name in contract["term_names"]]
    if sidecar["columns"] != sorted(columns) or len(sidecar["columns"]) != len(set(sidecar["columns"])):
        raise ValueError("v3 fragment columns are malformed")
    compressed_rel = _safe_relative_path(sidecar["compressed_filename"], name="compressed_filename")
    if compressed_rel != str(Path(sidecar_rel).with_suffix(".npz")):
        raise ValueError("v3 sidecar and fragment event disagree about the bundle")
    _reject_symlink_components(root, compressed_rel, name="compressed bundle")
    bundle_path = root / compressed_rel
    if not bundle_path.is_file() or _sha256(bundle_path) != sidecar["compressed_sha256"]:
        raise ValueError("v3 bundle is missing or its hash does not match the sidecar")
    provenance = sidecar["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("v3 sidecar provenance must be an object")
    _require_keys(
        provenance,
        {"run_id", "config_hash", "source_revision", "sampler_settings", "acceptance_proposal_stats"},
        optional=set(provenance) - {"run_id", "config_hash", "source_revision", "sampler_settings", "acceptance_proposal_stats"},
    )
    if provenance["run_id"] != header["run_id"] or not isinstance(provenance["config_hash"], str) or not _HASH_RE.fullmatch(provenance["config_hash"]):
        raise ValueError("v3 sidecar provenance identity is invalid")
    if provenance["source_revision"] is not None and (
        not isinstance(provenance["source_revision"], str) or not provenance["source_revision"]
    ):
        raise ValueError("v3 sidecar source_revision is invalid")
    if not isinstance(provenance["sampler_settings"], Mapping) or not isinstance(provenance["acceptance_proposal_stats"], list):
        raise ValueError("v3 sidecar sampler provenance is invalid")
    if any(not isinstance(item, Mapping) for item in provenance["acceptance_proposal_stats"]):
        raise ValueError("v3 sidecar acceptance proposal stats are invalid")
    if "measurements" in sidecar:
        measurements = sidecar["measurements"]
        if not isinstance(measurements, Mapping):
            raise ValueError("v3 sidecar measurements must be an object")
        allowed_measurements = {
            "compressed_byte_count", "uncompressed_byte_count", "uncompressed_sha256",
            "compression_ratio", "write_time_sec",
        }
        _require_keys(measurements, set(), optional=allowed_measurements)
        for key in ("compressed_byte_count", "uncompressed_byte_count"):
            if key in measurements:
                _int(measurements[key], name=f"measurement {key}", minimum=1)
        if "uncompressed_sha256" in measurements:
            _hash(measurements["uncompressed_sha256"], name="measurement uncompressed_sha256")
        if "compression_ratio" in measurements:
            _finite_number(measurements["compression_ratio"], name="measurement compression_ratio")
            if measurements["compression_ratio"] <= 0:
                raise ValueError("v3 measurement compression_ratio must be positive")
        if "write_time_sec" in measurements:
            _finite_number(measurements["write_time_sec"], name="measurement write_time_sec")
    try:
        with np.load(bundle_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
    except Exception as exc:
        raise ValueError("v3 bundle cannot be read") from exc
    if sorted(arrays) != sorted(columns):
        raise ValueError("v3 bundle columns disagree with its sidecar")
    walkers = int(contract["walker_count"])
    expected_dtypes = {
        "iteration_step": "<i8", "completed_updates": "<i8", "update_applied": "|b1",
        "walker_index": "<i8", "positions": "<f8", "local_energy": "<f8",
        "logabs": "<f8", "sign": "<f8", "finite": "|b1",
    }
    for name in columns:
        expected = expected_dtypes.get(name, "<f8")
        if arrays[name].dtype.str != expected:
            raise ValueError(f"v3 bundle dtype is invalid for {name}")
    if arrays["iteration_step"].shape != (walkers,) or arrays["walker_index"].shape != (walkers,):
        raise ValueError("v3 bundle row shape is invalid")
    if arrays["positions"].shape != (walkers, int(contract["n_electrons"]), int(contract["spatial_dim"])):
        raise ValueError("v3 positions shape disagrees with its contract")
    for name in columns:
        if name not in {"positions"} and arrays[name].shape != (walkers,):
            raise ValueError("v3 scalar column shape is invalid")
    if not np.all(arrays["iteration_step"] == step) or not np.array_equal(arrays["walker_index"], np.arange(walkers, dtype=np.int64)):
        raise ValueError("v3 bundle has an invalid step or walker grid")
    if np.any(arrays["completed_updates"] > step + 1) or np.any(arrays["completed_updates"] < 0) or not np.all(arrays["completed_updates"] == arrays["completed_updates"][0]):
        raise ValueError("v3 completed_updates is invalid")
    if not np.all(arrays["update_applied"] == arrays["update_applied"][0]):
        raise ValueError("v3 update_applied is not constant within a fragment")
    finite = (
        np.isfinite(arrays["positions"]).all(axis=(1, 2))
        & np.isfinite(arrays["local_energy"])
        & np.isfinite(arrays["logabs"])
        & np.isfinite(arrays["sign"])
    )
    for name in columns:
        if name.startswith("term/"):
            finite &= np.isfinite(arrays[name])
    if not np.array_equal(arrays["finite"], finite) or not bool(finite.all()):
        raise ValueError("v3 finite flags are invalid")
    if sidecar["logical_sha256"] != _v3_array_digest(arrays):
        raise ValueError("v3 logical digest does not match the sidecar")
    return arrays, dict(sidecar)


class TrainingBatchReader:
    """Read v1/v2 manifests or a strict v3 lineage rooted at ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.provenance: dict[int, Mapping[str, Any]] = {}
        self._v3_completion: dict[str, int] | None = None

    def read(self, *, start_step: int | None = None, end_step: int | None = None) -> dict[str, np.ndarray]:
        """Validate the complete selected graph, then return an inclusive step range."""

        if (self.root / HISTORY_FILENAME).is_file():
            raw = (self.root / HISTORY_FILENAME).read_bytes()
            header = _strict_json_loads(raw)
            if isinstance(header, Mapping) and header.get("schema") == V3_SCHEMA:
                return self._read_v3(start_step=start_step, end_step=end_step)
        self._v3_completion = None
        return self._read_legacy(start_step=start_step, end_step=end_step)

    def validate(self, *, require_complete: bool = False) -> None:
        """Validate every committed object; optionally require a v3 completion event."""

        if (self.root / HISTORY_FILENAME).is_file():
            raw = (self.root / HISTORY_FILENAME).read_bytes()
            header = _strict_json_loads(raw)
            if isinstance(header, Mapping) and header.get("schema") == V3_SCHEMA:
                self._read_v3(require_complete=require_complete, retain_rows=False)
                return
        if require_complete:
            raise ValueError("require_complete is unavailable for legacy training histories")
        self._read_legacy()

    @property
    def completion(self) -> dict[str, int] | None:
        """Return v3 completion counters, or ``None`` for open/legacy histories."""

        if (self.root / HISTORY_FILENAME).is_file():
            self._read_v3(retain_rows=False)
            return dict(self._v3_completion) if self._v3_completion is not None else None
        return None

    def _read_legacy(self, *, start_step: int | None = None, end_step: int | None = None) -> dict[str, np.ndarray]:
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
                raise ValueError("training partition sidecar digest does not match its manifest entry (partition integrity failure)")
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

    def _read_v3(
        self,
        *,
        start_step: int | None = None,
        end_step: int | None = None,
        require_complete: bool = False,
        retain_rows: bool = True,
    ) -> dict[str, np.ndarray]:
        node = self._load_v3_node(
            self.root.resolve(),
            stack=(),
            seen_streams=set(),
            limit=None,
            select_start=start_step,
            select_end=end_step,
            retain_rows=retain_rows,
        )
        self._v3_completion = node.completion
        self.provenance = {index: value for index, value in enumerate(node.provenance)}
        if require_complete and node.completion is None:
            raise ValueError("v3 training history is open and is not complete")
        arrays = node.rows
        if not arrays:
            return {}
        order = np.lexsort((arrays["walker_index"], arrays["iteration_step"]))
        result = {name: value[order] for name, value in arrays.items()}
        mask = np.ones(len(result["iteration_step"]), dtype=bool)
        if start_step is not None:
            mask &= result["iteration_step"] >= int(start_step)
        if end_step is not None:
            mask &= result["iteration_step"] <= int(end_step)
        return {name: value[mask] for name, value in result.items()}

    def _load_v3_node(
        self,
        root: Path,
        *,
        stack: tuple[Path, ...],
        seen_streams: set[str],
        limit: int | None,
        expected_prefix_sha256: str | None = None,
        select_start: int | None = None,
        select_end: int | None = None,
        retain_rows: bool = True,
    ) -> _V3Node:
        if root in stack:
            raise ValueError("v3 lineage contains a cycle")
        history_path = root / HISTORY_FILENAME
        events_path = root / EVENTS_FILENAME
        if not history_path.is_file() or not events_path.is_file():
            raise ValueError("v3 history is missing history.json or events.jsonl")
        header_raw = history_path.read_bytes()
        header = _validate_v3_header(header_raw, _strict_json_loads(header_raw))
        stream_id = header["stream_id"]
        if stream_id in seen_streams and root not in stack:
            raise ValueError("v3 lineage contains a duplicate stream_id")
        local_seen = set(seen_streams)
        local_seen.add(stream_id)
        header_hash = hashlib.sha256(header_raw).hexdigest()
        raw_log = events_path.read_bytes()
        complete_bytes = raw_log if raw_log.endswith(b"\n") else raw_log[: raw_log.rfind(b"\n") + 1]
        events: list[dict[str, Any]] = []
        digests: list[str] = []
        for line in complete_bytes.splitlines(keepends=True):
            if not line.endswith(b"\n") or line in {b"\n", b"\r\n"}:
                raise ValueError("v3 events.jsonl contains an empty or malformed LF-terminated line")
            event = _strict_json_loads(line)
            if not isinstance(event, Mapping) or line != _compact_json(event):
                raise ValueError("v3 event line is not canonical JSON")
            event = dict(event)
            _v3_event_shape(event, len(events))
            if events and event["previous_event_sha256"] != digests[-1]:
                raise ValueError("v3 event chain predecessor digest is invalid")
            digest = _v3_event_digest(header_hash, line)
            events.append(event)
            digests.append(digest)
            if limit is not None and len(events) == limit:
                break
        if limit is not None and len(events) != limit:
            raise ValueError("v3 parent pin event_count exceeds its committed prefix")
        if not events:
            raise ValueError("v3 history has no committed origin event")
        if limit is not None and expected_prefix_sha256 is not None and digests[-1] != expected_prefix_sha256:
            raise ValueError("v3 parent pin prefix_sha256 does not match the pinned parent bytes")
        if limit is not None and limit < len(events):
            events = events[:limit]
            digests = digests[:limit]
        origin = events[0]
        if (header["origin"] == "fresh") != (origin["type"] == "genesis"):
            raise ValueError("v3 header origin disagrees with event[0]")
        if header["origin"] == "train_resume" and origin["type"] != "resume":
            raise ValueError("v3 resumed history must begin with resume")
        if header["origin"] == "fresh" and origin["type"] != "genesis":
            raise ValueError("v3 fresh history must begin with genesis")
        fragments: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]] = []
        seals: list[dict[str, Any]] = []
        completion: dict[str, int] | None = None
        fragment_steps: list[int] = []
        fragment_updates: list[int] = []
        active_steps: set[int] = set()
        active_identities: set[tuple[int, int]] = set()
        seal_positions: dict[int, int] = {}
        for index, event in enumerate(events):
            if event["type"] == "resume":
                self._validate_v3_resume_event(event, header)
            elif event["type"] == "fragment":
                _int(event["group_index"], name="fragment group_index")
                step = _int(event["start_step"], name="fragment start_step")
                _int(event["end_step"], name="fragment end_step", minimum=1)
                _hash(event["sidecar_sha256"], name="sidecar_sha256")
                cadence_start = int(header["contract"]["start_step"])
                cadence = int(header["contract"]["every_n_steps"])
                if event["end_step"] != step + 1 or step < cadence_start or (step - cadence_start) % cadence != 0 or (fragment_steps and step <= fragment_steps[-1]):
                    raise ValueError("v3 local fragment steps are not strictly increasing")
                if header["origin"] == "train_resume" and step < event0_cut(events[0]):
                    raise ValueError("v3 resumed fragment precedes its supersede cut")
                arrays, sidecar = _validate_v3_sidecar(root, header, header["contract"], event)
                for walker in arrays["walker_index"]:
                    identity = (step, int(walker))
                    if identity in active_identities:
                        raise ValueError("duplicate active (iteration_step, walker_index) in v3 lineage")
                    active_identities.add(identity)
                active_steps.add(step)
                selected = (
                    _select_v3_arrays(arrays, start=select_start, end=select_end)
                    if retain_rows
                    else {}
                )
                fragments.append((dict(event), selected, sidecar))
                fragment_steps.append(step)
                fragment_updates.append(int(arrays["completed_updates"][0]))
            elif event["type"] == "window_seal":
                group = _int(event["group_index"], name="window_seal group_index")
                start = _int(event["start_step"], name="window_seal start_step")
                end = _int(event["end_step"], name="window_seal end_step", minimum=1)
                partition = int(header["contract"]["partition_size_steps"])
                if start != group * partition or not start < end <= start + partition:
                    raise ValueError("v3 window seal interval is invalid")
                seal_positions[group] = index
                seals.append(dict(event))
            elif event["type"] == "complete":
                if completion is not None or index != len(events) - 1:
                    raise ValueError("v3 complete must be the sole final event")
                completion = {
                    "next_iteration": _int(event["next_iteration"], name="next_iteration"),
                    "completed_updates": _int(event["completed_updates"], name="completed_updates"),
                }
                if completion["completed_updates"] > completion["next_iteration"]:
                    raise ValueError("v3 completed_updates exceeds next_iteration")
        updates = fragment_updates
        if any(later < earlier for earlier, later in zip(updates, updates[1:])):
            raise ValueError("v3 completed_updates is not monotone across fragments")
        parent_node: _V3Node | None = None
        cut = 0
        if origin["type"] == "resume":
            cut = origin["supersede_from_step"]
            parent_pin = origin["parent"]
            checkpoint = origin["checkpoint"]
            if not isinstance(parent_pin, Mapping):
                if cut != 0:
                    raise ValueError("v3 nonzero resume requires a parent pin")
            else:
                _require_keys(parent_pin, {"root", "event_count", "prefix_sha256"})
                count = _int(parent_pin["event_count"], name="parent event_count", minimum=1)
                _hash(parent_pin["prefix_sha256"], name="parent prefix_sha256")
                if not isinstance(parent_pin["root"], str) or not Path(parent_pin["root"]).is_absolute():
                    raise ValueError("v3 parent root must be an absolute path")
                parent_root = Path(parent_pin["root"])
                parent_node = self._load_v3_node(
                    parent_root.resolve(),
                    stack=stack + (root,),
                    seen_streams=local_seen,
                    limit=count,
                    expected_prefix_sha256=parent_pin["prefix_sha256"],
                    select_start=select_start,
                    select_end=(
                        cut - 1
                        if select_end is None
                        else min(select_end, cut - 1)
                    ),
                    retain_rows=retain_rows,
                )
                if parent_node.header["contract"] != header["contract"]:
                    raise ValueError("v3 resumed contract is incompatible with its parent")
            self._validate_v3_checkpoint(checkpoint, header, parent_node, cut)
        elif origin["type"] == "genesis":
            if header["origin"] != "fresh":
                raise ValueError("v3 genesis header has the wrong origin")
        inherited_rows = parent_node.rows if parent_node is not None else {}
        local_arrays = [arrays for _, arrays, _ in fragments if arrays]
        rows = _merge_v3_rows(inherited_rows, local_arrays, cut)
        inherited_steps = parent_node.active_steps if parent_node is not None else set()
        inherited_ids = parent_node.active_identities if parent_node is not None else set()
        if parent_node is not None and not _v3_has_scheduled_prefix(inherited_steps, header["contract"], cut):
            raise ValueError("v3 parent pin does not cover the required scheduled prefix")
        partition = int(header["contract"]["partition_size_steps"])
        inherited_seals = [
            dict(seal)
            for seal in (parent_node.effective_seals if parent_node is not None else [])
            if int(seal["end_step"]) <= cut
            and int(seal["end_step"]) == (int(seal["group_index"]) + 1) * partition
        ]
        effective_seals = inherited_seals + seals
        if len({int(seal["group_index"]) for seal in effective_seals}) != len(effective_seals):
            raise ValueError("v3 lineage contains duplicate active window seals")
        combined_steps = set(inherited_steps)
        combined_steps = {step for step in combined_steps if step < cut} | active_steps
        combined_ids = {identity for identity in inherited_ids if identity[0] < cut} | active_identities
        self._validate_v3_seals(events, seals, fragments, parent_node, combined_steps, header["contract"], cut, completion)
        self._validate_v3_completion(completion, combined_steps, effective_seals, parent_node, header["contract"], cut)
        provenance = []
        if parent_node is not None:
            provenance.extend(dict(item) for item in parent_node.provenance if item.get("start_step", -1) < cut)
        for event, arrays, sidecar in fragments:
            if not arrays:
                continue
            provenance.append({
                **dict(sidecar["provenance"]),
                "stream_id": header["stream_id"],
                "history_root": str(root),
                "group_index": int(event["group_index"]),
                "start_step": int(event["start_step"]),
                "end_step": int(event["end_step"]),
            })
        compact_fragments = [(event, {}, sidecar) for event, _, sidecar in fragments]
        return _V3Node(
            root, header, events, digests, compact_fragments, seals, effective_seals, completion,
            rows, combined_steps, combined_ids, provenance,
        )

    @staticmethod
    def _validate_v3_checkpoint(
        checkpoint: Mapping[str, Any],
        header: Mapping[str, Any],
        parent: _V3Node | None,
        cut: int,
    ) -> None:
        _require_keys(checkpoint, {"path", "manifest_sha256", "model_sha256", "next_iteration", "completed_updates"})
        path_value = checkpoint["path"]
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ValueError("v3 checkpoint.path must be an absolute checkpoint domain path")
        checkpoint_path = Path(path_value)
        try:
            # §C pins one concrete step directory; root/latest.json locators
            # are mutable convenience pointers and are not wire data.
            ref = checkpoint_ref(checkpoint_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("v3 checkpoint path is not a valid complete checkpoint") from exc
        _hash(checkpoint["manifest_sha256"], name="manifest_sha256")
        _hash(checkpoint["model_sha256"], name="model_sha256")
        _int(checkpoint["next_iteration"], name="checkpoint next_iteration")
        _int(checkpoint["completed_updates"], name="checkpoint completed_updates")
        if checkpoint["manifest_sha256"] != ref.manifest_sha256 or checkpoint["model_sha256"] != ref.model_sha256:
            raise ValueError("v3 checkpoint hashes do not match the referenced checkpoint bytes")
        if checkpoint["next_iteration"] != ref.next_iteration or checkpoint["completed_updates"] != ref.completed_updates:
            raise ValueError("v3 checkpoint counters do not match its manifest")
        if checkpoint["completed_updates"] > checkpoint["next_iteration"] or checkpoint["next_iteration"] != cut:
            raise ValueError("v3 checkpoint counters do not bind supersede_from_step")
        expected_run_id = parent.header["run_id"] if parent is not None else header["run_id"]
        checkpoint_run_id = ref.provenance.get("run_id")
        if checkpoint_run_id != expected_run_id:
            raise ValueError("v3 checkpoint provenance does not bind the pinned history")
        if parent is not None:
            run_dir = ref.provenance.get("run_dir")
            if not isinstance(run_dir, str) or not run_dir:
                raise ValueError("v3 checkpoint source provenance does not bind the pinned history")
            try:
                checkpoint_run_dir = Path(run_dir).resolve()
            except OSError as exc:
                raise ValueError("v3 checkpoint source provenance does not bind the pinned history") from exc
            if checkpoint_run_dir != parent.root.resolve():
                raise ValueError("v3 checkpoint source provenance does not bind the pinned history")

    @staticmethod
    def _validate_v3_resume_event(event: Mapping[str, Any], header: Mapping[str, Any]) -> None:
        _int(event["supersede_from_step"], name="supersede_from_step")
        if header["origin"] != "train_resume":
            raise ValueError("v3 resume event requires train_resume origin")
        parent = event["parent"]
        if parent is not None and not isinstance(parent, Mapping):
            raise ValueError("v3 resume parent must be null or an object")

    @staticmethod
    def _validate_v3_seals(
        events: list[dict[str, Any]],
        seals: list[dict[str, Any]],
        fragments: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]],
        parent: _V3Node | None,
        active_steps: set[int],
        contract: Mapping[str, Any],
        cut: int,
        completion: Mapping[str, int] | None,
    ) -> None:
        partition = int(contract["partition_size_steps"])
        seal_indices: dict[int, int] = {}
        for index, event in enumerate(events):
            if event["type"] == "window_seal":
                group = int(event["group_index"])
                seal_indices[group] = index
                end = int(event["end_step"])
                if end < (group + 1) * partition and completion is None:
                    raise ValueError("v3 partial window seal requires completion")
                if end < (group + 1) * partition and (completion is None or index != len(events) - 2):
                    raise ValueError("v3 partial window seal must immediately precede complete")
        for event, _, _ in fragments:
            group = int(event["group_index"])
            if group in seal_indices:
                event_index = events.index(event)
                if event_index > seal_indices[group]:
                    raise ValueError("v3 fragment follows its window seal")
        for seal in seals:
            start, end = int(seal["start_step"]), int(seal["end_step"])
            expected_steps = {
                step for step in active_steps if start <= step < end
            }
            cadence_start = int(contract["start_step"])
            every = int(contract["every_n_steps"])
            first = cadence_start + max(0, (start - cadence_start + every - 1) // every) * every
            scheduled = set(range(first, end, every))
            if expected_steps != scheduled:
                raise ValueError("v3 window seal coverage is incomplete or contains an extra step")

    @staticmethod
    def _validate_v3_completion(
        completion: Mapping[str, int] | None,
        active_steps: set[int],
        seals: list[Mapping[str, Any]],
        parent: _V3Node | None,
        contract: Mapping[str, Any],
        cut: int,
    ) -> None:
        if completion is None:
            return
        next_iteration = int(completion["next_iteration"])
        steps = set(active_steps)
        start = int(contract["start_step"])
        every = int(contract["every_n_steps"])
        expected = set(range(start, next_iteration, every)) if next_iteration > start else set()
        if steps != expected or any(step >= next_iteration for step in steps):
            raise ValueError("v3 complete history has incorrect scheduled fragment coverage")
        partition = int(contract["partition_size_steps"])
        expected_seals = {
            group: (group * partition, min((group + 1) * partition, next_iteration))
            for group in range((next_iteration + partition - 1) // partition)
        }
        actual = {int(seal["group_index"]): (int(seal["start_step"]), int(seal["end_step"])) for seal in seals}
        if actual != expected_seals:
            raise ValueError("v3 complete history has incorrect window seals")


def event0_cut(event: Mapping[str, Any]) -> int:
    return int(event["supersede_from_step"]) if event.get("type") == "resume" else 0


def _select_v3_arrays(
    arrays: Mapping[str, np.ndarray], *, start: int | None, end: int | None
) -> dict[str, np.ndarray]:
    """Copy only rows in the requested inclusive range from one fragment."""

    if not arrays:
        return {}
    steps = arrays["iteration_step"]
    mask = np.ones(len(steps), dtype=bool)
    if start is not None:
        mask &= steps >= int(start)
    if end is not None:
        mask &= steps <= int(end)
    return {name: value[mask].copy() for name, value in arrays.items()} if bool(mask.any()) else {}


def _merge_v3_rows(parent: Mapping[str, np.ndarray], local: Iterable[Mapping[str, np.ndarray]], cut: int) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    if parent:
        mask = parent["iteration_step"] < cut
        parts.append({name: values[mask] for name, values in parent.items()})
    parts.extend(dict(values) for values in local)
    if not parts:
        return {}
    result = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    return result


def _v3_has_scheduled_prefix(active_steps: set[int], contract: Mapping[str, Any], cut: int) -> bool:
    start, every = int(contract["start_step"]), int(contract["every_n_steps"])
    if cut <= start:
        return True
    expected = set(range(start, cut, every))
    actual = {int(step) for step in active_steps if start <= step < cut}
    return expected == actual


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
    history whose lineage cannot be proven. Like ``InitialCheckpoint``, the
    defensive first-lineage check runs at ``started(TrainingIteration)``:
    this boundary is emitted before ``CollectSamples`` and therefore sees the
    pre-update trainer cursor, rather than the post-update counters available
    at ``TrainingIterationCompleted``.
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
                selectors=(
                    started(TrainingIteration),
                    Subscription.of(TrainingIterationCompleted),
                    Subscription.of(TrainingCompleted),
                )
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
        if isinstance(event, Started) and isinstance(event.operation, TrainingIteration):
            if not self._first_iteration_seen:
                self._first_iteration_seen = True
                # Match InitialCheckpoint: Started(TrainingIteration) is the
                # pre-sampling boundary, before post-update counters change.
                progress = state.trainer.state_dict()
                fresh_lineage = event.operation.step == 0 and all(
                    progress[key] == 0 for key in ("next_iteration", "completed_updates")
                )
                if not fresh_lineage:
                    raise ValueError(
                        "RawTrainingBatchRecorder refuses restored lineage before capture "
                        "or publication [unsupported-callback]"
                    )
            return
        if isinstance(event, TrainingCompleted):
            self._publish(context, final=True)
            return
        if not isinstance(event, TrainingIterationCompleted):
            return
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
