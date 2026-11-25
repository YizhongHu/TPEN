"""Contract tests for derived full-H completion."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


completion = importlib.import_module("experiments.atomistic.he-importance.full_horizon_completion")


def _record() -> dict[str, object]:
    return {
        "attempt_chain": [],
        "completed_at": "2026-11-25T14:00:00.000000Z",
        "completed_updates": 100,
        "core_acceptance_digest": "a" * 64,
        "final_attempt_id": "attempt-S",
        "final_checkpoint": {"content_id": "f" * 64, "path": "/store/final"},
        "group_digest": "b" * 64,
        "schema": completion.SCHEMA,
        "science_digest": "c" * 64,
        "target_iterations": 100,
        "task_id": "task-7",
    }


def _write(core: Path, record: dict[str, object]) -> None:
    core.parent.mkdir(parents=True)
    core.mkdir()
    task_dir = core.parent.parent
    task_dir.joinpath("full-horizon-completion.json").write_bytes(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def test_completion_reader_accepts_doc_shape(tmp_path: Path) -> None:
    core = tmp_path / "groups" / ("b" * 64) / "task-7" / "attempt-S" / "core"
    _write(core, _record())
    assert completion.read_full_horizon_completion(core, group_digest="b" * 64, task_id="task-7", science_digest="c" * 64, target_iterations=100)


@pytest.mark.parametrize("field", ["science_digest", "target_iterations", "completed_updates"])
def test_completion_reader_derives_false_for_nonmatching_progress(tmp_path: Path, field: str) -> None:
    core = tmp_path / "groups" / ("b" * 64) / "task-7" / "attempt-S" / "core"
    record = _record()
    record[field] = ("d" * 64) if field == "science_digest" else 99
    _write(core, record)
    assert not completion.read_full_horizon_completion(core, group_digest="b" * 64, task_id="task-7", science_digest="c" * 64, target_iterations=100)


def test_completion_reader_rejects_noncanonical_record(tmp_path: Path) -> None:
    core = tmp_path / "groups" / ("b" * 64) / "task-7" / "attempt-S" / "core"
    _write(core, _record())
    path = core.parent.parent / "full-horizon-completion.json"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    with pytest.raises(completion.FullHorizonCompletionError, match="canonical"):
        completion.read_full_horizon_completion(core, group_digest="b" * 64, task_id="task-7", science_digest="c" * 64, target_iterations=100)
