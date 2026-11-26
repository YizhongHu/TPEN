"""Bounded T2 probes for the HI S6B-R2 executor/consumer seam."""

from __future__ import annotations

import json
import importlib
from pathlib import Path
import sys

import pytest

from tpen_executor.configured_program import AttemptDelta, ConfiguredLoad
from tpen_executor._codec import canonical_bytes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_b1r2c_full_horizon import _publish, _single_fixture  # noqa: E402


completion = importlib.import_module(
    "experiments.atomistic.he-importance.full_horizon_completion"
)
delta_reader = importlib.import_module("experiments.atomistic.he-importance.attempt_delta")


def _delta(core: Path, mode: str) -> AttemptDelta:
    checkpoint = None if mode == "fresh" else str(core.parent.parent.parent / "previous" / "core" / "checkpoint")
    return AttemptDelta(
        str(core), ConfiguredLoad(mode, checkpoint), core.parent.parent.parent.name,
        core.parent.parent.name, core.parent.name,
    )


def _record(core: Path, *, final: str = "attempt-S", chain: list[dict[str, str]] | None = None) -> dict[str, object]:
    return {
        "attempt_chain": chain or [],
        "completed_at": "2026-11-25T14:00:00.000000Z",
        "completed_updates": 100,
        "core_acceptance_digest": "a" * 64,
        "final_attempt_id": final,
        "final_checkpoint": {"content_id": "f" * 64, "path": str(core / "checkpoints" / "final")},
        "group_digest": "b" * 64,
        "schema": completion.SCHEMA,
        "science_digest": "ab" * 32,
        "target_iterations": 100,
        "task_id": "task-7",
    }


def _write_completion(core: Path, value: dict[str, object]) -> None:
    core.mkdir(parents=True)
    (core.parent.parent / "full-horizon-completion.json").write_bytes(canonical_bytes(value))


def test_executor_begin_creates_attempt_but_not_core_source_contract() -> None:
    storage = Path(__file__).parents[1] / "src" / "tpen_executor" / "storage.py"
    transport = Path(__file__).parents[1] / "src" / "tpen_executor" / "dispatch" / "transport.py"
    storage_source = storage.read_text(encoding="utf-8")
    transport_source = transport.read_text(encoding="utf-8")
    assert "_mkdir_new(output_dir)" in storage_source
    assert "_mkdir_new(os.path.join(output_dir, \"core\"))" not in storage_source
    assert "existing core directory or log means the request is a replay" in transport_source


def test_real_executor_publication_reaches_hi_reader_and_science_mismatch_is_false(tmp_path: Path) -> None:
    store, group, fence, verdict, task = _single_fixture(tmp_path)
    record = _publish(store, group, fence, verdict, task)
    core = Path(fence.output_dir) / "core"
    assert completion.read_full_horizon_completion(
        core, group_digest=group.identity, task_id=task.task_id,
        science_digest=task.science_digest, target_iterations=1,
    ) is True
    path = Path(fence.output_dir).parent / "full-horizon-completion.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["science_digest"] = "c" * 64
    path.write_bytes(canonical_bytes(changed))
    assert record.science_digest == task.science_digest
    assert completion.read_full_horizon_completion(
        core, group_digest=group.identity, task_id=task.task_id,
        science_digest=task.science_digest, target_iterations=1,
    ) is False


@pytest.mark.parametrize("mode", ["fresh", "train_resume"])
def test_real_executor_attempt_delta_canonical_bytes_reach_hi_parser(tmp_path: Path, mode: str) -> None:
    core = tmp_path / "groups" / ("b" * 64) / "task-7" / "attempt-S" / "core"
    parsed = delta_reader.parse_attempt_delta(_delta(core, mode).canonical_bytes)
    assert parsed.attempt_core_dir == core
    assert parsed.load_mode == mode


@pytest.mark.parametrize(
    ("chain", "final", "match"),
    [
        ([{"predecessor_attempt_id": "A", "successor_attempt_id": "B", "request_digest": "a" * 64},
          {"predecessor_attempt_id": "B", "successor_attempt_id": "A", "request_digest": "b" * 64}], "A", "cycle"),
        ([{"predecessor_attempt_id": "A", "successor_attempt_id": "A", "request_digest": "a" * 64}], "A", "cycle"),
        ([{"predecessor_attempt_id": "A", "successor_attempt_id": "B", "request_digest": "a" * 64}], "other", "attempt_chain"),
        ([{"predecessor_attempt_id": "A", "successor_attempt_id": "B", "request_digest": "a" * 64},
          {"predecessor_attempt_id": "B", "successor_attempt_id": "C", "request_digest": "b" * 64}], "C", "accepted"),
    ],
)
def test_hi_reader_enforces_acyclic_chain_and_declared_terminal(
    tmp_path: Path, chain: list[dict[str, str]], final: str, match: str,
) -> None:
    core = tmp_path / "groups" / ("b" * 64) / "task-7" / "attempt-S" / "core"
    _write_completion(core, _record(core, final=final, chain=chain))
    if match == "accepted":
        assert completion.read_full_horizon_completion(
            core, group_digest="b" * 64, task_id="task-7",
            science_digest="ab" * 32, target_iterations=100,
        ) is True
    else:
        with pytest.raises(completion.FullHorizonCompletionError, match=match):
            completion.read_full_horizon_completion(
                core, group_digest="b" * 64, task_id="task-7",
                science_digest="ab" * 32, target_iterations=100,
            )


@pytest.mark.parametrize("raw_suffix", ["/a/./core", "/a//core", "/a/core/"])
def test_delta_reader_rejects_non_normalized_core_path(tmp_path: Path, raw_suffix: str) -> None:
    base = tmp_path / "groups" / "g" / "task"
    raw = {
        "attempt_core_dir": str(base) + raw_suffix,
        "attempt_id": "a", "group_digest": "g", "task_id": "task",
        "load": {"mode": "fresh", "path": None}, "schema": delta_reader.SCHEMA,
    }
    with pytest.raises(delta_reader.AttemptDeltaError, match="absolute and normalized|end in"):
        delta_reader.parse_attempt_delta(json.dumps(raw, sort_keys=True, separators=(",", ":")))


def test_delta_reader_rejects_current_attempt_sibling_but_accepts_predecessor(tmp_path: Path) -> None:
    core = tmp_path / "groups" / "g" / "task" / "a" / "core"
    base = json.loads(_delta(core, "train_resume").canonical_bytes)
    base["load"]["path"] = str(core.parent / "checkpoint")
    with pytest.raises(delta_reader.AttemptDeltaError, match="outside the current attempt leaf"):
        delta_reader.parse_attempt_delta(json.dumps(base, sort_keys=True, separators=(",", ":")))
    base["load"]["path"] = str(core.parent.parent / "previous" / "core" / "checkpoint")
    parsed = delta_reader.parse_attempt_delta(json.dumps(base, sort_keys=True, separators=(",", ":")))
    assert parsed.load_path == Path(base["load"]["path"])
