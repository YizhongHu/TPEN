"""Durable library writer for the ``training_history/v3`` format.

This module is intentionally independent of callback selection.  Callers hand
it one already-captured :class:`TrainingBatchRecord` at a time; the writer
never samples, replays, or retains earlier batches.
"""

from __future__ import annotations

import hashlib
import io
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .raw_training_batch_recorder import (
    EVENTS_FILENAME,
    HISTORY_FILENAME,
    TrainingBatchRecord,
    V3_SCHEMA,
    _compact_json,
    _pretty_json,
    _HASH_RE,
    _UUID_RE,
    _validate_v3_contract,
    _v3_array_digest,
    _v3_event_digest,
)


_ZERO_HASH = "0" * 64


def _write_all(fd: int, data: bytes) -> None:
    """Write all bytes, preserving short-write behavior for fault tests."""

    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short write returned no progress")
        view = view[count:]


def _fsync(fd: int) -> None:
    """Keep fsync in one seam so callers can inject ordered I/O faults."""

    os.fsync(fd)


def _fsync_directory(path: Path) -> None:
    """Durably persist directory entries on POSIX filesystems."""

    fd = os.open(path, os.O_RDONLY)
    try:
        _fsync(fd)
    finally:
        os.close(fd)


def _write_private(path: Path, data: bytes) -> None:
    """Create and durably write one temporary object without overwriting."""

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        _write_all(fd, data)
        _fsync(fd)
    finally:
        os.close(fd)


def _publish_exclusive(temp: Path, final: Path) -> None:
    """Publish a closed object under an exclusive final name."""

    os.link(temp, final)
    temp.unlink()


def _npz_bytes(record: TrainingBatchRecord) -> bytes:
    """Encode exactly one capture using NumPy's non-pickle compressed format."""

    arrays = {
        "iteration_step": np.full(record.n_walkers, record.iteration_step, dtype="<i8"),
        "completed_updates": np.full(record.n_walkers, record.completed_updates, dtype="<i8"),
        "update_applied": np.full(record.n_walkers, record.update_applied, dtype=np.bool_),
        "walker_index": np.asarray(record.walker_index, dtype="<i8"),
        "positions": np.asarray(record.positions, dtype="<f8"),
        "local_energy": np.asarray(record.local_energy, dtype="<f8"),
        "logabs": np.asarray(record.logabs, dtype="<f8"),
        "sign": np.asarray(record.sign, dtype="<f8"),
        "finite": np.asarray(record.finite, dtype=np.bool_),
    }
    arrays.update({f"term/{name}": np.asarray(value, dtype="<f8") for name, value in record.term_energies.items()})
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


class TrainingHistoryV3Writer:
    """Write one fresh, immutable v3 history root.

    Parameters
    ----------
    root : path-like
        New history root.  It must not already contain any entries.
    stream_id, run_id : str
        Immutable stream and run identifiers.
    contract : mapping
        The approved v3 contract.  It is validated by the R1 reader helpers
        when the resulting history is consumed.
    origin : {"fresh", "train_resume"}
        Origin marker for the sole event at log position zero.
    parent, checkpoint : mapping, optional
        Resume binding supplied by the caller; lineage discovery remains the
        reader's responsibility.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        stream_id: str | None = None,
        run_id: str,
        contract: Mapping[str, Any],
        origin: str = "fresh",
        parent: Mapping[str, Any] | None = None,
        checkpoint: Mapping[str, Any] | None = None,
        supersede_from_step: int = 0,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.fragments = self.root / "fragments"
        self._poisoned = False
        self._last_step: int | None = None
        self._last_digest: str | None = None
        self._event_count = 0
        self._captured_by_group: dict[int, int] = {}
        self._sealed: dict[int, tuple[int, int]] = {}
        self._completed = False
        self.contract = dict(contract)
        _validate_v3_contract(self.contract)
        if stream_id is not None and not _UUID_RE.fullmatch(stream_id):
            raise ValueError("v3 stream_id must be a canonical UUID")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("v3 run_id must be a non-empty string")
        self.provenance = dict(provenance or {})
        self.provenance.setdefault("run_id", run_id)
        self.provenance.setdefault("config_hash", _ZERO_HASH)
        self.provenance.setdefault("source_revision", None)
        self.provenance.setdefault("sampler_settings", {})
        self.provenance.setdefault("acceptance_proposal_stats", [])
        if self.root.exists():
            if not self.root.is_dir() or self.root.is_symlink() or any(self.root.iterdir()):
                raise FileExistsError("v3 history root must be a fresh, empty, non-symlink directory")
        else:
            self.root.mkdir(parents=True)
        self.fragments.mkdir()
        self._header = {
            "schema": V3_SCHEMA,
            "stream_id": stream_id or str(uuid.uuid4()),
            "run_id": run_id,
            "origin": origin,
            "contract": self.contract,
        }
        if origin not in {"fresh", "train_resume"}:
            raise ValueError("v3 origin must be fresh or train_resume")
        if origin == "fresh":
            event = {"type": "genesis", "previous_event_sha256": None}
        else:
            if checkpoint is None:
                raise ValueError("train_resume requires checkpoint binding")
            if parent is not None:
                required_parent = {"root", "event_count", "prefix_sha256"}
                if set(parent) != required_parent or not isinstance(parent["root"], str) or not Path(parent["root"]).is_absolute():
                    raise ValueError("v3 parent binding has the wrong shape")
                if not isinstance(parent["event_count"], int) or parent["event_count"] < 1 or not _HASH_RE.fullmatch(str(parent["prefix_sha256"])):
                    raise ValueError("v3 parent binding has invalid count or digest")
            required_checkpoint = {"path", "manifest_sha256", "model_sha256", "next_iteration", "completed_updates"}
            if set(checkpoint) != required_checkpoint or not isinstance(checkpoint.get("path"), str) or not Path(checkpoint["path"]).is_absolute():
                raise ValueError("v3 checkpoint binding has the wrong shape")
            if not _HASH_RE.fullmatch(str(checkpoint["manifest_sha256"])) or not _HASH_RE.fullmatch(str(checkpoint["model_sha256"])):
                raise ValueError("v3 checkpoint binding has invalid digest")
            event = {
                "type": "resume",
                "previous_event_sha256": None,
                "parent": parent,
                "checkpoint": dict(checkpoint),
                "supersede_from_step": int(supersede_from_step),
            }
        self._header_bytes = _pretty_json(self._header)
        self._header_hash = hashlib.sha256(self._header_bytes).hexdigest()
        self._create_origin(event)

    @property
    def poisoned(self) -> bool:
        """Whether a failed operation permanently stopped this writer."""

        return self._poisoned

    @property
    def event_count(self) -> int:
        """Return the number of acknowledged complete log events."""

        return self._event_count

    def _fail(self, exc: BaseException) -> None:
        self._poisoned = True
        raise exc

    def _check_open(self) -> None:
        if self._poisoned:
            raise RuntimeError("v3 history writer is poisoned")
        if self._completed:
            raise RuntimeError("v3 history writer is complete")

    def _create_origin(self, event: dict[str, Any]) -> None:
        try:
            _write_private(self.root / HISTORY_FILENAME, self._header_bytes)
            _write_private(self.root / EVENTS_FILENAME, b"")
            _fsync_directory(self.root)
            self._append_event(event)
        except BaseException as exc:
            self._poisoned = True
            raise exc

    def _append_event(self, event: dict[str, Any]) -> None:
        event["previous_event_sha256"] = self._last_digest
        line = _compact_json(event)
        digest = _v3_event_digest(self._header_hash, line)
        fd = os.open(self.root / EVENTS_FILENAME, os.O_WRONLY | os.O_APPEND)
        try:
            _write_all(fd, line)
            _fsync(fd)
        finally:
            os.close(fd)
        _fsync_directory(self.root)
        self._last_digest = digest
        self._event_count += 1

    def append_capture(self, record: TrainingBatchRecord, *, sampler_stats: Mapping[str, Any] | None = None) -> None:
        """Durably publish one scheduled capture and its fragment event."""

        try:
            self._check_open()
            self._validate_record(record)
            step = int(record.iteration_step)
            stem = f"step-{step:012d}"
            bundle_rel = f"fragments/{stem}.npz"
            sidecar_rel = f"fragments/{stem}.json"
            bundle = _npz_bytes(record)
            bundle_hash = hashlib.sha256(bundle).hexdigest()
            arrays = {
                "iteration_step": np.full(record.n_walkers, step, dtype="<i8"),
                "completed_updates": np.full(record.n_walkers, record.completed_updates, dtype="<i8"),
                "update_applied": np.full(record.n_walkers, record.update_applied, dtype=np.bool_),
                "walker_index": record.walker_index,
                "positions": record.positions,
                "local_energy": record.local_energy,
                "logabs": record.logabs,
                "sign": record.sign,
                "finite": record.finite,
            }
            arrays.update({f"term/{name}": value for name, value in record.term_energies.items()})
            group = step // int(self.contract["partition_size_steps"])
            if group in self._sealed:
                raise ValueError("capture follows its v3 window seal")
            sidecar = {
                "schema": V3_SCHEMA,
                "source": "tpen.callback.RawTrainingBatchRecorder",
                "capture_semantics": "pre_update_model_forward",
                "stream_id": self._header["stream_id"],
                "contract_sha256": hashlib.sha256(_pretty_json(self.contract)).hexdigest(),
                "group_index": group,
                "start_step": step,
                "end_step": step + 1,
                "walker_count": record.n_walkers,
                "row_count": record.n_walkers,
                "term_names": list(self.contract["term_names"]),
                "columns": sorted(arrays),
                "compressed_filename": bundle_rel,
                "compressed_sha256": bundle_hash,
                "logical_sha256": _v3_array_digest(arrays),
                "provenance": {
                    **self.provenance,
                    "acceptance_proposal_stats": [dict(sampler_stats)] if sampler_stats else self.provenance["acceptance_proposal_stats"],
                },
            }
            sidecar_bytes = _pretty_json(sidecar)
            bundle_tmp = self.fragments / f".{stem}.npz.tmp"
            sidecar_tmp = self.fragments / f".{stem}.json.tmp"
            _write_private(bundle_tmp, bundle)
            _write_private(sidecar_tmp, sidecar_bytes)
            _publish_exclusive(bundle_tmp, self.fragments / f"{stem}.npz")
            _publish_exclusive(sidecar_tmp, self.fragments / f"{stem}.json")
            _fsync_directory(self.fragments)
            self._append_event({
                "type": "fragment", "group_index": group, "start_step": step,
                "end_step": step + 1, "sidecar": sidecar_rel,
                "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            })
            self._last_step = step
            self._captured_by_group[group] = self._captured_by_group.get(group, 0) + 1
        except BaseException as exc:
            self._poisoned = True
            raise exc

    def seal_window(self, group_index: int, *, end_step: int | None = None) -> None:
        """Append a full window seal, or a final partial seal before complete."""

        try:
            self._check_open()
            group = int(group_index)
            partition = int(self.contract["partition_size_steps"])
            start = group * partition
            end = partition * (group + 1) if end_step is None else int(end_step)
            if group in self._sealed or end <= start or end > start + partition:
                raise ValueError("invalid or duplicate v3 window seal")
            if end < start + partition:
                raise ValueError("partial v3 window seals are allowed only immediately before complete")
            cadence_start = int(self.contract["start_step"])
            every = int(self.contract["every_n_steps"])
            first = cadence_start + max(0, (start - cadence_start + every - 1) // every) * every
            expected = len(range(first, end, every)) if first < end else 0
            if self._captured_by_group.get(group, 0) != expected:
                raise ValueError("v3 window seal coverage is incomplete")
            self._append_event({"type": "window_seal", "group_index": group, "start_step": start, "end_step": end})
            self._sealed[group] = (start, end)
        except BaseException as exc:
            self._poisoned = True
            raise exc

    # Short aliases keep the library surface convenient without creating a
    # second implementation or changing the callback's public names.
    seal = seal_window
    append = append_capture

    def complete(self, *, next_iteration: int, completed_updates: int) -> None:
        """Append the final seal when needed and the immutable completion event."""

        try:
            self._check_open()
            next_iteration = int(next_iteration)
            completed_updates = int(completed_updates)
            if next_iteration < 0 or completed_updates < 0 or completed_updates > next_iteration:
                raise ValueError("invalid v3 completion counters")
            start = int(self.contract["start_step"])
            every = int(self.contract["every_n_steps"])
            expected = set(range(start, next_iteration, every))
            if self._last_step is not None and self._last_step >= next_iteration:
                raise ValueError("capture is outside completion prefix")
            if next_iteration and expected and self._last_step != max(expected):
                raise ValueError("v3 completion has missing scheduled captures")
            groups = range((next_iteration + int(self.contract["partition_size_steps"]) - 1) // int(self.contract["partition_size_steps"]))
            for group in groups:
                partition = int(self.contract["partition_size_steps"])
                end = min((group + 1) * partition, next_iteration)
                if end < (group + 1) * partition:
                    if group in self._sealed:
                        raise ValueError("partial window already sealed")
                    cadence_start = int(self.contract["start_step"])
                    every = int(self.contract["every_n_steps"])
                    first = cadence_start + max(0, (group * partition - cadence_start + every - 1) // every) * every
                    expected = len(range(first, end, every)) if first < end else 0
                    if self._captured_by_group.get(group, 0) != expected:
                        raise ValueError("v3 partial window seal coverage is incomplete")
                    self._append_event({"type": "window_seal", "group_index": group, "start_step": group * partition, "end_step": end})
                    self._sealed[group] = (group * partition, end)
                elif group not in self._sealed:
                    raise ValueError("complete requires full window seals")
            self._append_event({"type": "complete", "next_iteration": next_iteration, "completed_updates": completed_updates})
            self._completed = True
        except BaseException as exc:
            self._poisoned = True
            raise exc

    finalize = complete

    def _validate_record(self, record: TrainingBatchRecord) -> None:
        """Validate cadence, monotonicity, shape, dtype, and finite contract."""

        if not isinstance(record, TrainingBatchRecord):
            raise TypeError("append_capture requires TrainingBatchRecord")
        step = int(record.iteration_step)
        start = int(self.contract["start_step"])
        every = int(self.contract["every_n_steps"])
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("capture steps must be strictly increasing")
        if step < start or (step - start) % every:
            raise ValueError("capture step is outside the v3 cadence")
        if record.n_walkers != int(self.contract["walker_count"]):
            raise ValueError("capture walker count disagrees with the v3 contract")
        if record.term_names != tuple(self.contract["term_names"]):
            raise ValueError("capture term names disagree with the v3 contract")
        if record.positions.shape[1:] != (int(self.contract["n_electrons"]), int(self.contract["spatial_dim"])):
            raise ValueError("capture positions disagree with the v3 geometry")


TrainingBatchV3Writer = TrainingHistoryV3Writer
TrainingHistoryWriter = TrainingHistoryV3Writer

__all__ = ["TrainingHistoryV3Writer", "TrainingBatchV3Writer", "TrainingHistoryWriter"]
