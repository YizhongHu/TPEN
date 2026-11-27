"""Consume one committed Stage-Q result as one logical HI sample."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

class StageQConsumerError(ValueError):
    """The selected result cannot be admitted as one logical sample."""

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value

def _record(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    raise StageQConsumerError("reader returned an invalid evidence record")

def _canonical_cell(path: Path, row_id: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise StageQConsumerError("selection receipt is unreadable") from error
    try: value = json.loads(raw)
    except json.JSONDecodeError as error: raise StageQConsumerError("selection receipt is invalid JSON") from error
    if not isinstance(value, dict): raise StageQConsumerError("selection receipt must be an object")
    if raw != (json.dumps(value, indent=2, sort_keys=True) + "\n").encode():
        raise StageQConsumerError("selection receipt is not canonical")
    packet = value.get("train_packet")
    if not isinstance(packet, Mapping) or packet.get("content_hash") != row_id:
        raise StageQConsumerError("selection receipt train packet does not identify row")
    return value, hashlib.sha256(raw).hexdigest()

def _checkpoint_core(segment: Any) -> tuple[Path, str]:
    checkpoint = Path(segment.checkpoint_dir)
    if (
        str(checkpoint) != segment.checkpoint_dir
        or not checkpoint.is_absolute()
        or any(part in {".", ".."} for part in checkpoint.parts)
    ):
        raise StageQConsumerError("selected checkpoint path is not normalized and absolute")
    if checkpoint.parent.name != "checkpoints":
        raise StageQConsumerError("selected checkpoint is not under core/checkpoints")
    core = checkpoint.parent.parent
    resolved_checkpoint, resolved_core = checkpoint.resolve(), core.resolve()
    if (
        resolved_checkpoint == resolved_checkpoint.parent
        or resolved_checkpoint.parent.name != "checkpoints"
        or resolved_checkpoint.parent.parent != resolved_core
    ):
        raise StageQConsumerError("selected checkpoint is not directly beneath its core")
    if len(core.parts) < 4 or core.name != "core" or core.parent.name != segment.attempt_id or core.parent.parent.name != segment.task_id:
        raise StageQConsumerError("selected checkpoint core identity mismatch")
    group_digest = core.parent.parent.parent.name
    if _HEX64.fullmatch(group_digest) is None: raise StageQConsumerError("selected checkpoint group identity is invalid")
    return core, group_digest

def _read_report(raw: bytes, row_id: str, segment: Any, core: Path, group_digest: str) -> dict[str, Any]:
    try: value = json.loads(raw)
    except json.JSONDecodeError as error: raise StageQConsumerError("attempt report is invalid JSON") from error
    if not isinstance(value, dict) or value.get("row_id") != row_id: raise StageQConsumerError("attempt report row identity mismatch")
    delta = value.get("attempt_delta")
    if not isinstance(delta, Mapping): raise StageQConsumerError("attempt report has no attempt delta")
    try:
        parsed = importlib.import_module("experiments.atomistic.he-importance.attempt_delta").parse_attempt_delta(json.dumps(delta, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    except (ValueError, TypeError, json.JSONDecodeError) as error: raise StageQConsumerError("attempt delta is invalid") from error
    if parsed.attempt_core_dir != core or parsed.group_digest != group_digest or parsed.task_id != segment.task_id or parsed.attempt_id != segment.attempt_id:
        raise StageQConsumerError("attempt report does not match selected segment")
    if segment.predecessor_checkpoint is None:
        if parsed.load_mode != "fresh" or parsed.load_path is not None: raise StageQConsumerError("root attempt load does not match selected segment")
    elif parsed.load_mode != "train_resume" or parsed.load_path != Path(segment.predecessor_checkpoint):
        raise StageQConsumerError("successor attempt load does not match selected segment")
    return value

def consume_stage_q(manifest_path: str | Path, campaign_root: str | Path, row_id: str, target_horizon: int, archive_path: str | Path, *, inspect_run: Callable[[str | Path], Any] | None = None, task_id: str | None = None, before_recheck: Callable[[], None] | None = None) -> dict[str, Any]:
    """Admit one row and write a write-once derived receipt.
    preserved_attempts is filtered to the selected task because this is one single-row receipt.
    """
    if type(row_id) is not str or Path(row_id).name != row_id or row_id in {".", ".."} or _HEX64.fullmatch(row_id) is None:
        raise StageQConsumerError("row_id must be a lowercase 64-hex path component")
    if type(target_horizon) is not int or target_horizon <= 0: raise StageQConsumerError("target_horizon must be a positive integer")
    manifest, campaign, receipt = Path(manifest_path), Path(campaign_root), Path(archive_path)
    if receipt.exists() or receipt.is_symlink(): raise StageQConsumerError("archive receipt already exists")
    c_path = campaign / "cells" / row_id / "selection_receipt.json"
    c_value, c_hash = _canonical_cell(c_path, row_id)
    try:
        manifest_hash = _digest(manifest)
    except OSError as error:
        raise StageQConsumerError("manifest is unreadable") from error
    before = {str(c_path): c_hash, str(manifest): manifest_hash}
    reader = importlib.import_module("experiments.toolkit.run_results")
    try: results = reader.read(manifest, inspect_run=inspect_run)
    except TypeError: results = reader.read(manifest)
    chains = [chain for chain in results.chains if chain.segments and (task_id is None or chain.task_id == task_id)]
    if len(chains) != 1: raise StageQConsumerError("requested row must have exactly one nonempty committed chain")
    chain, reports, cores, group_digest = chains[0], [], {}, None
    for segment in chain.segments:
        core, segment_group = _checkpoint_core(segment)
        if group_digest is None: group_digest = segment_group
        elif group_digest != segment_group: raise StageQConsumerError("selected chain has multiple group identities")
        report_path = core / "attempt_report.json"
        try: report_raw = report_path.read_bytes()
        except OSError as error: raise StageQConsumerError("attempt report is unreadable") from error
        before[str(report_path)] = hashlib.sha256(report_raw).hexdigest()
        _read_report(report_raw, row_id, segment, core, segment_group)
        reports.append(before[str(report_path)])
        cores[(segment.task_id, segment.attempt_id)] = core
    assert group_digest is not None
    final, final_core = chain.segments[-1], cores[(chain.segments[-1].task_id, chain.segments[-1].attempt_id)]
    resolved = c_value.get("resolved_config")
    declared = resolved.get("trainer", {}).get("max_steps") if isinstance(resolved, Mapping) and isinstance(resolved.get("trainer"), Mapping) else c_value.get("target_horizon")
    if declared is not None and (type(declared) is not int or declared != target_horizon): raise StageQConsumerError("target horizon disagrees with selection receipt")
    completion = importlib.import_module("experiments.atomistic.he-importance.full_horizon_completion")
    completion_path = final_core.parent.parent / "full-horizon-completion.json"
    try: completion_raw = completion_path.read_bytes()
    except FileNotFoundError as error: raise StageQConsumerError("full-horizon completion is not proven") from error
    except OSError as error: raise StageQConsumerError("full-horizon completion is unreadable") from error
    before[str(completion_path)] = hashlib.sha256(completion_raw).hexdigest()
    if not completion.read_full_horizon_completion(final_core, group_digest=group_digest, task_id=final.task_id, science_digest=row_id, target_iterations=target_horizon):
        raise StageQConsumerError("full-horizon completion is not proven")
    try: completion_value = json.loads(completion_raw)
    except json.JSONDecodeError as error: raise StageQConsumerError("full-horizon completion is invalid JSON") from error
    selected = [s.attempt_id for s in chain.segments]
    expected, actual = list(zip(selected[:-1], selected[1:])), [(link["predecessor_attempt_id"], link["successor_attempt_id"]) for link in completion_value["attempt_chain"]]
    if completion_value.get("final_attempt_id") != final.attempt_id or completion_value.get("final_checkpoint", {}).get("path") != final.checkpoint_dir or completion_value.get("completed_updates") != final.completed_updates or final.completed_updates < target_horizon or actual != expected:
        raise StageQConsumerError("full-horizon completion does not match selected chain")
    body = {"schema": "tpen-hi/stage-q-consumer-archive-v1", "row_id": row_id, "sample_count": 1, "selected_chain": [_record(s) for s in chain.segments], "preserved_attempts": [_record(a) for a in results.preserved_attempts if a.task_id == chain.task_id], "ranking_input": {"cell_directory": str(final_core), "logical_cell_directory": str(campaign / "cells" / row_id), "checkpoint_parent": str(Path(final.checkpoint_dir).parent), "selected_checkpoint": final.checkpoint_dir}, "source_sha256": dict(before), "attempt_report_sha256": reports}
    if before_recheck is not None: before_recheck()
    try: after = {path: _digest(Path(path)) for path in before}
    except OSError as error: raise StageQConsumerError("source changed during export") from error
    if before != after: raise StageQConsumerError("source changed during export")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        with receipt.open("x", encoding="utf-8") as stream: json.dump(body, stream, indent=2, sort_keys=True); stream.write("\n")
    except FileExistsError as error: raise StageQConsumerError("archive receipt already exists") from error
    return body

archive_stage_q = consume_stage_q
__all__ = ["StageQConsumerError", "archive_stage_q", "consume_stage_q"]
