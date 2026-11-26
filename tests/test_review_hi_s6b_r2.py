"""Bounded T2 probes for the HI S6B-R2 executor/consumer seam."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest

from tpen_executor.configured_program import AttemptDelta, ConfiguredLoad
from tpen_executor._codec import canonical_bytes

sys.path.insert(0, str(Path(__file__).parents[1] / "executor" / "tests"))
from test_b1r2c_full_horizon import _publish, _single_fixture  # noqa: E402


completion = importlib.import_module(
    "experiments.atomistic.he-importance.full_horizon_completion"
)
delta_reader = importlib.import_module("experiments.atomistic.he-importance.attempt_delta")
qualification = importlib.import_module("experiments.polaris.qualify.qualification")


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


def _qualification_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    root = tmp_path / "root"
    store, group, fence, verdict, task = _single_fixture(root)
    _publish(store, group, fence, verdict, task)
    row_id = task.science_digest
    campaign = root / "campaign"
    (campaign / "cells" / row_id).mkdir(parents=True)
    (campaign / "cells" / row_id / "selection_receipt.json").write_bytes(canonical_bytes({"selection": {"selected": {"context": "O-raw"}}, "walkers": 1}))
    delta = _delta(Path(fence.output_dir) / "core", "fresh")
    report = {"row_id": row_id, "attempt_delta": json.loads(delta.canonical_bytes)}
    (Path(fence.output_dir) / "core" / "attempt_report.json").write_bytes(canonical_bytes(report))
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(canonical_bytes({"runtime_kind": "polaris-gpu-qualification", "output_root": str(campaign), "cell": {"context": "O-raw", "walkers": 1, "updates": 1, "checkpoint_every": 1}}))
    return campaign, manifest, Path(fence.output_dir) / "core", row_id


def test_qualification_cell_complete_consumes_matching_real_full_h_record(tmp_path: Path) -> None:
    campaign, manifest, _core, _row = _qualification_fixture(tmp_path)
    evidence = tmp_path / "evidence"
    qualification.qualify(manifest, evidence, artifact_root=campaign.parent)
    verdict = json.loads((evidence / "qualification.json").read_text(encoding="utf-8"))
    assert next(row for row in verdict["predicates"] if row["name"] == "cell_complete")["status"] == "PASS"


def test_qualification_does_not_short_circuit_later_invalid_same_row_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, manifest, core, row = _qualification_fixture(tmp_path)
    invalid_core = core.parents[2] / "task-invalid" / "attempt-invalid" / "core"
    invalid_core.mkdir(parents=True)
    (invalid_core.parent.parent / "full-horizon-completion.json").write_bytes(canonical_bytes({"schema": "malformed"}))
    invalid = invalid_core / "attempt_report.json"
    invalid.write_bytes(canonical_bytes({"row_id": row, "attempt_delta": json.loads(_delta(invalid_core, "fresh").canonical_bytes)}))
    real_find = qualification._find

    def ordered_find(root: Path, names: tuple[str, ...]) -> list[Path]:
        if names == ("attempt_report.json",):
            return [core / "attempt_report.json", invalid]
        return real_find(root, names)

    monkeypatch.setattr(qualification, "_find", ordered_find)
    with pytest.raises(completion.FullHorizonCompletionError):
        qualification.qualify(manifest, tmp_path / "evidence", artifact_root=campaign.parent)


def test_qualification_legacy_complete_survives_unrelated_delta_report(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    row = "row-a"
    (root / "cells" / row).mkdir(parents=True)
    (root / "cells" / row / "selection_receipt.json").write_bytes(canonical_bytes({"selection": {"selected": {"context": "O-raw"}}}))
    (root / "Q" / row / "COMPLETE").parent.mkdir(parents=True)
    (root / "Q" / row / "COMPLETE").write_text("completed\n", encoding="utf-8")
    (root / "unrelated" / "attempt_report.json").parent.mkdir(parents=True)
    (root / "unrelated" / "attempt_report.json").write_bytes(canonical_bytes({"row_id": "row-b"}))
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(canonical_bytes({"runtime_kind": "polaris-gpu-qualification", "output_root": str(root), "cell": {"context": "O-raw", "walkers": 1, "updates": 1, "checkpoint_every": 1}}))
    evidence = tmp_path / "evidence"
    qualification.qualify(manifest, evidence)
    verdict = json.loads((evidence / "qualification.json").read_text(encoding="utf-8"))
    assert next(row for row in verdict["predicates"] if row["name"] == "cell_complete")["status"] == "PASS"
