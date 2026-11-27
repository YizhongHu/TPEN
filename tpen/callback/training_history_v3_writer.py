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


def _reject_symlinked_root(path: Path) -> None:
    """Reject symlinks in or below the canonicalized root."""

    current = Path(path.anchor) if path.is_absolute() else Path()
    for component in path.parts[1:] if path.is_absolute() else path.parts:
        current = current / component
        if current.exists() and current.is_symlink():
            raise ValueError("v3 history root or root component is a symlink")
    if path.is_dir():
        for directory, directories, files in os.walk(path, followlinks=False):
            if any((Path(directory) / name).is_symlink() for name in (*directories, *files)):
                raise ValueError("v3 history root or root component is a symlink")


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


def _write_private(path: Path, data: bytes, *, fsync: bool = True) -> None:
    """Create and durably write one temporary object without overwriting."""

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        _write_all(fd, data)
        if fsync:
            _fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    """Fsync one already-created file by path."""

    fd = os.open(path, os.O_RDONLY)
    try:
        _fsync(fd)
    finally:
        os.close(fd)


def _fsync_log(fd: int) -> None:
    """Fsync the append-only event log through its own mutation seam."""

    _fsync(fd)


def _publish_exclusive(temp: Path, final: Path) -> None:
    """Publish a closed object under an exclusive final name."""

    if final.exists():
        raise FileExistsError(f"exclusive publication collision: {final}")
    os.link(temp, final)
    temp.unlink()


def _append_log_line(path: Path, line: bytes) -> None:
    """Append one complete event line and fsync the log file."""

    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        _write_all(fd, line)
        _fsync_log(fd)
    finally:
        os.close(fd)


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
        inherited_seal_frontier: int = 0,
        inherited_seals: Mapping[int, tuple[int, int]] | None = None,
    ) -> None:
        # Resolve caller-owned aliases once; descendants remain subject to the guard.
        self.root = Path(root).resolve()
        self.fragments = self.root / "fragments"
        self._poisoned = False
        self._last_step: int | None = None
        self._last_digest: str | None = None
        self._event_count = 0
        self._capture_group: int | None = None
        self._capture_count = 0
        self._sealed_frontier = 0
        self._partial_sealed_group: int | None = None
        self._completed = False
        self._cut = int(supersede_from_step) if origin == "train_resume" else 0
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
        _reject_symlinked_root(self.root)
        if self.root.exists():
            if not self.root.is_dir() or self.root.is_symlink() or any(self.root.iterdir()):
                raise FileExistsError("v3 history root must be a fresh, empty, non-symlink directory")
        else:
            self._mkdir_durable(self.root)
        self._mkdir_durable(self.fragments)
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
        partition = int(self.contract["partition_size_steps"])
        if inherited_seals is not None:
            if inherited_seal_frontier:
                raise ValueError("specify one inherited seal representation")
            if not inherited_seals or any(group not in inherited_seals for group in range(len(inherited_seals))):
                raise ValueError("inherited seals must be a contiguous mapping")
            inherited_seal_frontier = len(inherited_seals)
            for group, interval in inherited_seals.items():
                if not isinstance(group, int) or group < 0 or tuple(interval) != (group * partition, (group + 1) * partition):
                    raise ValueError("invalid inherited full seal")
        if inherited_seal_frontier < 0 or inherited_seal_frontier * partition > self._cut:
            raise ValueError("inherited seal frontier exceeds the resume cut")
        if origin != "train_resume" and inherited_seal_frontier:
            raise ValueError("inherited seals require train_resume origin")
        self._sealed_frontier = int(inherited_seal_frontier)
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

    @staticmethod
    def _mkdir_durable(path: Path) -> None:
        missing: list[Path] = []
        current = path
        while not current.exists():
            missing.append(current)
            if current.parent == current:
                break
            current = current.parent
        for directory in reversed(missing):
            directory.mkdir()
            _fsync_directory(directory.parent)

    def _create_origin(self, event: dict[str, Any]) -> None:
        try:
            _write_private(self.root / HISTORY_FILENAME, self._header_bytes)
            _write_private(self.root / EVENTS_FILENAME, b"")
            _fsync_directory(self.root)
            self._append_event(event)
        except BaseException as exc:
            self._fail(exc)

    def _append_event(self, event: dict[str, Any]) -> None:
        event["previous_event_sha256"] = self._last_digest
        line = _compact_json(event)
        digest = _v3_event_digest(self._header_hash, line)
        _append_log_line(self.root / EVENTS_FILENAME, line)
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
            if group < self._sealed_frontier or group == self._partial_sealed_group:
                raise ValueError("capture follows its v3 window seal")
            if self._capture_group is not None and group != self._capture_group:
                if group != self._sealed_frontier:
                    raise ValueError("capture follows an unsealed v3 window")
                self._capture_count = 0
            self._capture_group = group
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
            _write_private(bundle_tmp, bundle, fsync=False)
            _fsync_file(bundle_tmp)
            _write_private(sidecar_tmp, sidecar_bytes, fsync=False)
            _fsync_file(sidecar_tmp)
            event = {
                "type": "fragment", "group_index": group, "start_step": step,
                "end_step": step + 1, "sidecar": sidecar_rel,
                "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            }
            _publish_exclusive(bundle_tmp, self.fragments / f"{stem}.npz")
            _publish_exclusive(sidecar_tmp, self.fragments / f"{stem}.json")
            _fsync_directory(self.fragments)
            self._append_event(event)
            self._last_step = step
            self._capture_count += 1
        except BaseException as exc:
            self._fail(exc)

    def seal_window(self, group_index: int, *, end_step: int | None = None) -> None:
        """Append a full window seal, or a final partial seal before complete."""

        try:
            self._check_open()
            group = int(group_index)
            if group < 0:
                raise ValueError("invalid or duplicate v3 window seal")
            partition = int(self.contract["partition_size_steps"])
            start = group * partition
            end = partition * (group + 1) if end_step is None else int(end_step)
            if group != self._sealed_frontier or self._partial_sealed_group is not None or end <= start or end > start + partition:
                raise ValueError("invalid or duplicate v3 window seal")
            if end < start + partition:
                raise ValueError("partial v3 window seals are allowed only immediately before complete")
            expected = self._expected_capture_count(start, end)
            if self._capture_group == group and self._capture_count != expected:
                raise ValueError("v3 window seal coverage is incomplete")
            if self._capture_group != group and expected:
                raise ValueError("v3 window seal coverage is incomplete")
            self._append_event({"type": "window_seal", "group_index": group, "start_step": start, "end_step": end})
            if end == start + partition:
                self._sealed_frontier += 1
            else:
                self._partial_sealed_group = group
        except BaseException as exc:
            self._fail(exc)

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
            expected_count = self._expected_capture_count(max(start, self._cut), next_iteration)
            expected_last = self._expected_last_step(next_iteration)
            if self._last_step is not None and self._last_step >= next_iteration:
                raise ValueError("capture is outside completion prefix")
            if expected_count and self._last_step != expected_last:
                raise ValueError("v3 completion has missing scheduled captures")
            partition = int(self.contract["partition_size_steps"])
            required_groups = (next_iteration + partition - 1) // partition
            full_groups = next_iteration // partition
            if self._sealed_frontier < full_groups:
                raise ValueError("complete requires full window seals")
            if self._sealed_frontier > full_groups:
                raise ValueError("complete has seals outside its completion prefix")
            if required_groups > full_groups:
                group = full_groups
                end = next_iteration
                if self._partial_sealed_group is not None:
                    raise ValueError("partial window already sealed")
                expected = self._expected_capture_count(group * partition, end)
                observed = self._capture_count if self._capture_group == group else 0
                if observed != expected:
                    raise ValueError("v3 partial window seal coverage is incomplete")
                self._append_event({"type": "window_seal", "group_index": group, "start_step": group * partition, "end_step": end})
                self._partial_sealed_group = group
            elif self._partial_sealed_group is not None:
                raise ValueError("complete has seals outside its completion prefix")
            self._append_event({"type": "complete", "next_iteration": next_iteration, "completed_updates": completed_updates})
            self._completed = True
        except BaseException as exc:
            self._fail(exc)

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
        if step < self._cut:
            raise ValueError("resume capture precedes the supersede cut")
        if record.n_walkers != int(self.contract["walker_count"]):
            raise ValueError("capture walker count disagrees with the v3 contract")
        if record.walker_index.dtype != np.dtype("<i8") or not np.array_equal(
            record.walker_index, np.arange(record.n_walkers, dtype=record.walker_index.dtype)
        ):
            raise ValueError("capture walker_index must be 0..W-1")
        if record.term_names != tuple(self.contract["term_names"]):
            raise ValueError("capture term names disagree with the v3 contract")
        if record.positions.shape[1:] != (int(self.contract["n_electrons"]), int(self.contract["spatial_dim"])):
            raise ValueError("capture positions disagree with the v3 geometry")
        expected = {
            "walker_index": ("<i8", (record.n_walkers,)),
            "positions": ("<f8", (record.n_walkers, int(self.contract["n_electrons"]), int(self.contract["spatial_dim"]))),
            "local_energy": ("<f8", (record.n_walkers,)), "logabs": ("<f8", (record.n_walkers,)),
            "sign": ("<f8", (record.n_walkers,)), "finite": ("|b1", (record.n_walkers,)),
        }
        for name, (dtype, shape) in expected.items():
            value = getattr(record, name)
            if value.shape != shape or value.dtype.str != dtype:
                raise ValueError(f"capture {name} has invalid shape or dtype")
            if name != "finite" and not np.isfinite(value).all():
                raise ValueError("capture required arrays must be finite")
        for name in self.contract["term_names"]:
            value = record.term_energies[name]
            if value.shape != (record.n_walkers,) or value.dtype.str != "<f8" or not np.isfinite(value).all():
                raise ValueError("capture term arrays have invalid shape, dtype, or finiteness")
        if not bool(np.all(record.finite)):
            raise ValueError("capture finite flags must all be true")
        if record.completed_updates < 0 or record.completed_updates > step + 1:
            raise ValueError("capture completed_updates is out of range")

    def _expected_capture_count(self, start: int, end: int) -> int:
        """Count scheduled captures owned by this stream after its resume cut."""

        cadence_start = int(self.contract["start_step"])
        every = int(self.contract["every_n_steps"])
        first = cadence_start + max(0, (start - cadence_start + every - 1) // every) * every
        first = max(first, self._cut + ((cadence_start - self._cut) % every)) if self._cut > cadence_start else first
        return len(range(first, end, every)) if first < end else 0

    def _expected_last_step(self, end: int) -> int | None:
        cadence_start = int(self.contract["start_step"])
        every = int(self.contract["every_n_steps"])
        first = cadence_start
        if self._cut > first:
            first = self._cut + ((cadence_start - self._cut) % every)
        count = self._expected_capture_count(first, end)
        return first + (count - 1) * every if count else None


__all__ = ["TrainingHistoryV3Writer"]
