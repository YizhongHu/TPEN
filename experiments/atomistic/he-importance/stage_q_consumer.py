"""Consume one committed Stage-Q result as one logical HI sample.

The consumer is deliberately a thin boundary: manifest selection belongs to
the public toolkit reader, and completion belongs to the stdlib HI reader.
This module only binds those facts to the immutable cell receipt and emits a
source-digested, write-once evidence receipt.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping


class StageQConsumerError(ValueError):
    """The selected result cannot be admitted as one logical sample."""


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _record(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    raise StageQConsumerError("reader returned an invalid evidence record")


def _canonical_cell(path: Path, row_id: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise StageQConsumerError("selection receipt is invalid JSON") from error
    if not isinstance(value, dict):
        raise StageQConsumerError("selection receipt must be an object")
    canonical = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if raw != canonical:
        raise StageQConsumerError("selection receipt is not canonical")
    packet = value.get("train_packet")
    if not isinstance(packet, Mapping) or packet.get("content_hash") != row_id:
        raise StageQConsumerError("selection receipt train packet does not identify row")
    return value, hashlib.sha256(raw).hexdigest()


def _read_report(path: Path, row_id: str, segment: Any) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise StageQConsumerError("attempt report is invalid JSON") from error
    if not isinstance(value, dict) or value.get("row_id") != row_id:
        raise StageQConsumerError("attempt report row identity mismatch")
    delta = value.get("attempt_delta")
    if not isinstance(delta, Mapping):
        raise StageQConsumerError("attempt report has no attempt delta")
    expected = {
        "attempt_id": segment.attempt_id,
        "group_digest": Path(segment.checkpoint_dir).parent.parent.parent.name,
        "task_id": segment.task_id,
    }
    if any(delta.get(key) != item for key, item in expected.items()):
        raise StageQConsumerError("attempt report identity mismatch")
    core = Path(delta.get("attempt_core_dir", ""))
    if core != Path(segment.checkpoint_dir):
        raise StageQConsumerError("attempt report core does not match selected segment")
    return value, hashlib.sha256(raw).hexdigest()


def consume_stage_q(
    manifest_path: str | Path,
    campaign_root: str | Path,
    row_id: str,
    target_horizon: int,
    archive_path: str | Path,
    *,
    inspect_run: Callable[[str | Path], Any] | None = None,
) -> dict[str, Any]:
    """Admit one row and write a write-once derived consumer receipt."""

    manifest = Path(manifest_path)
    campaign = Path(campaign_root)
    c_path = campaign / "cells" / row_id / "selection_receipt.json"
    receipt = Path(archive_path)
    if receipt.exists() or receipt.is_symlink():
        raise StageQConsumerError("archive receipt already exists")
    c_value, c_hash = _canonical_cell(c_path, row_id)

    reader = importlib.import_module("experiments.toolkit.run_results")
    try:
        results = reader.read(manifest, inspect_run=inspect_run)
    except TypeError:
        # A test seam may provide a minimal reader with the public one-argument
        # contract; no fallback decoding or directory discovery is permitted.
        results = reader.read(manifest)
    chains = [chain for chain in results.chains if chain.segments]
    if len(chains) != 1:
        raise StageQConsumerError("requested row must have exactly one nonempty committed chain")
    chain = chains[0]
    reports: list[tuple[dict[str, Any], str]] = []
    source_paths = [manifest, c_path]
    for segment in chain.segments:
        report_path = Path(segment.checkpoint_dir) / "attempt_report.json"
        reports.append(_read_report(report_path, row_id, segment))
        source_paths.append(report_path)
    final = chain.segments[-1]
    completion = importlib.import_module("experiments.atomistic.he-importance.full_horizon_completion")
    group_digest = Path(final.checkpoint_dir).parent.parent.parent.name
    if not completion.read_full_horizon_completion(
        Path(final.checkpoint_dir), group_digest=group_digest, task_id=final.task_id,
        science_digest=row_id, target_iterations=target_horizon,
    ):
        raise StageQConsumerError("full-horizon completion is not proven")
    completion_path = Path(final.checkpoint_dir).parent.parent / "full-horizon-completion.json"
    source_paths.append(completion_path)
    before = {str(path): _digest(path) for path in source_paths}
    preserved = [_jsonable(_record(item)) for item in results.preserved_attempts]
    body: dict[str, Any] = {
        "schema": "tpen-hi/stage-q-consumer-archive-v1",
        "row_id": row_id,
        "sample_count": 1,
        "selected_chain": [_jsonable(_record(segment)) for segment in chain.segments],
        "preserved_attempts": preserved,
        "ranking_input": {
            "cell_directory": str(campaign / "cells" / row_id),
            "checkpoint_parent": str(Path(final.checkpoint_dir).parent),
            "selected_checkpoint": final.checkpoint_dir,
        },
        "source_sha256": {str(manifest): before[str(manifest)], str(c_path): c_hash,
                          **{str(path): digest for path, digest in zip(source_paths[2:], list(before.values())[2:])}},
        "attempt_report_sha256": [digest for _, digest in reports],
    }
    after = {str(path): _digest(path) for path in source_paths}
    if before != after:
        raise StageQConsumerError("source changed during export")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(body, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise StageQConsumerError("archive receipt already exists") from error
    return body


archive_stage_q = consume_stage_q


__all__ = ["StageQConsumerError", "archive_stage_q", "consume_stage_q"]
