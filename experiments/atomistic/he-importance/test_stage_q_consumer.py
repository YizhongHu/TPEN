"""Real P8/full-H contract tests for the Stage-Q consumer."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
from typing import Any
import pytest

consumer = importlib.import_module("experiments.atomistic.he-importance.stage_q_consumer")
completion = importlib.import_module("experiments.atomistic.he-importance.full_horizon_completion")
run_results = importlib.import_module("experiments.toolkit.run_results")

def _compact(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()

def _pretty(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()

def _fixture(tmp_path: Path) -> tuple[Path, Path, str, dict[str, Path]]:
    row, group = "a" * 64, "b" * 64
    campaign = tmp_path / "campaign"
    c_path = campaign / "cells" / row / "selection_receipt.json"
    c_path.parent.mkdir(parents=True)
    c_path.write_bytes(_pretty({"resolved_config": {"trainer": {"max_steps": 2}}, "train_packet": {"content_hash": row}}))
    roots: dict[str, Path] = {}
    segments: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    previous: str | None = None
    for index in (1, 2):
        attempt = f"attempt-{index}"
        output = tmp_path / "groups" / group / "task-a" / attempt
        core = output / "core"
        checkpoint = core / "checkpoint" / f"step_00000{index}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "leaf.json").write_bytes(_compact({"attempt": attempt}))
        delta = {"attempt_core_dir": str(core), "attempt_id": attempt, "group_digest": group, "load": {"mode": "fresh" if index == 1 else "train_resume", "path": None if index == 1 else str(roots["core-1"] / "checkpoint" / "step_000001")}, "schema": "tpen-executor/attempt-delta-v1", "task_id": "task-a"}
        (core / "attempt_report.json").write_bytes(_pretty({"attempt_delta": delta, "row_id": row}))
        roots[f"core-{index}"] = core
        segments.append({"attempt_id": attempt, "checkpoint_dir": str(checkpoint), "completed_updates": index, "next_iteration": index, "predecessor_checkpoint": previous, "schema": "tpen-executor/accepted-segment-v1", "task_id": "task-a"})
        previous = str(checkpoint)
        attempts.append({"attempt_id": attempt, "output_dir": str(output), "schema": "tpen-executor/manifest-attempt-v1", "task_id": "task-a", "start_receipt": {"allocation_id": "allocation", "attempt_id": attempt, "backend_job_id": "job", "group_digest": group, "output_dir": str(output), "schema": "tpen-executor/start-receipt-v1", "started_at_utc": f"2026-11-04T00:0{index}:00Z", "task_id": "task-a"}, "outcome_receipt": {"attempt_id": attempt, "finished_at_utc": f"2026-11-04T00:1{index}:00Z", "group_digest": group, "output_dir": str(output), "result_digest": "c" * 64, "schema": "tpen-executor/outcome-receipt-v1", "status": "completed", "task_id": "task-a"}})
    failed = tmp_path / "groups" / group / "task-a" / "attempt-F"
    attempts.append({"attempt_id": "attempt-F", "output_dir": str(failed), "schema": "tpen-executor/manifest-attempt-v1", "task_id": "task-a", "start_receipt": {"allocation_id": "allocation", "attempt_id": "attempt-F", "backend_job_id": "job", "group_digest": group, "output_dir": str(failed), "schema": "tpen-executor/start-receipt-v1", "started_at_utc": "2026-11-04T00:03:00Z", "task_id": "task-a"}, "outcome_receipt": {"attempt_id": "attempt-F", "finished_at_utc": "2026-11-04T00:04:00Z", "group_digest": group, "output_dir": str(failed), "result_digest": "d" * 64, "schema": "tpen-executor/outcome-receipt-v1", "status": "failed", "task_id": "task-a"}})
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(_compact({"attempts": attempts, "group_digest": group, "group_id": "group", "schema": "tpen-executor/run-manifest-v2", "segments": segments}))
    task_dir = roots["core-2"].parent.parent
    record = {"attempt_chain": [{"predecessor_attempt_id": "attempt-1", "request_digest": "e" * 64, "successor_attempt_id": "attempt-2"}], "completed_at": "2026-11-04T00:20:00.000000Z", "completed_updates": 2, "core_acceptance_digest": "f" * 64, "final_attempt_id": "attempt-2", "final_checkpoint": {"content_id": "1" * 64, "path": str(roots["core-2"] / "checkpoint" / "step_000002")}, "group_digest": group, "schema": completion.SCHEMA, "science_digest": row, "target_iterations": 2, "task_id": "task-a"}
    (task_dir / "full-horizon-completion.json").write_bytes(_compact(record))
    return manifest, campaign, row, roots

def _inspect(core: str | Path) -> dict[str, Any]:
    checkpoint = sorted((Path(core) / "checkpoint").iterdir())[-1]
    step = int(checkpoint.name[-6:])
    return {"verdict": "completed", "checkpoint": {"checkpoint_dir": str(checkpoint), "schema_version": 1, "next_iteration": step, "completed_updates": step}}

def _sources(manifest: Path, campaign: Path, row: str, roots: dict[str, Path]) -> dict[str, bytes]:
    paths = [manifest, campaign / "cells" / row / "selection_receipt.json", roots["core-1"] / "attempt_report.json", roots["core-2"] / "attempt_report.json", roots["core-2"].parent.parent / "full-horizon-completion.json", roots["core-1"] / "checkpoint" / "step_000001" / "leaf.json", roots["core-2"] / "checkpoint" / "step_000002" / "leaf.json"]
    return {str(path): path.read_bytes() for path in paths}

def test_green_two_segment_chain_is_one_sample_and_preserves_failed_attempt(tmp_path: Path) -> None:
    manifest, campaign, row, roots = _fixture(tmp_path)
    before = _sources(manifest, campaign, row, roots)
    body = consumer.consume_stage_q(manifest, campaign, row, 2, tmp_path / "archive.json", inspect_run=_inspect, task_id="task-a")
    assert body["sample_count"] == 1
    assert [item["attempt_id"] for item in body["selected_chain"]] == ["attempt-1", "attempt-2"]
    assert [item["attempt_id"] for item in body["preserved_attempts"]] == ["attempt-F"]
    assert _sources(manifest, campaign, row, roots) == before

@pytest.mark.parametrize("mutation", ["row", "report", "checkpoint", "horizon", "missing-completion"])
def test_identity_and_completion_mutations_refuse_without_archive(tmp_path: Path, mutation: str) -> None:
    manifest, campaign, row, roots = _fixture(tmp_path)
    if mutation == "row":
        (campaign / "cells" / row / "selection_receipt.json").write_bytes(b"{}")
    elif mutation == "report":
        value = json.loads((roots["core-2"] / "attempt_report.json").read_bytes()); value["attempt_delta"]["attempt_id"] = "wrong"; (roots["core-2"] / "attempt_report.json").write_bytes(_pretty(value))
    elif mutation == "checkpoint":
        payload = json.loads(manifest.read_bytes()); payload["segments"][-1]["checkpoint_dir"] = str(tmp_path / "foreign"); manifest.write_bytes(_compact(payload))
    elif mutation == "horizon":
        path = roots["core-2"].parent.parent / "full-horizon-completion.json"; value = json.loads(path.read_bytes()); value["target_iterations"] = 3; path.write_bytes(_compact(value))
    else:
        (roots["core-2"].parent.parent / "full-horizon-completion.json").unlink()
    with pytest.raises((consumer.StageQConsumerError, run_results.ManifestRefusal)):
        consumer.consume_stage_q(manifest, campaign, row, 2, tmp_path / "archive.json", inspect_run=_inspect, task_id="task-a")
    assert not (tmp_path / "archive.json").exists()

def test_m4_reader_refuses_replayed_segment_and_m5_recheck_refuses_mutation(tmp_path: Path) -> None:
    manifest, campaign, row, roots = _fixture(tmp_path)
    payload = json.loads(manifest.read_bytes()); payload["segments"].append(dict(payload["segments"][-1])); manifest.write_bytes(_compact(payload))
    with pytest.raises(run_results.ManifestRefusal):
        consumer.consume_stage_q(manifest, campaign, row, 2, tmp_path / "m4.json", inspect_run=_inspect, task_id="task-a")
    manifest, campaign, row, roots = _fixture(tmp_path / "m5")
    def mutate() -> None:
        path = campaign / "cells" / row / "selection_receipt.json"; path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(consumer.StageQConsumerError, match="source changed"):
        consumer.consume_stage_q(manifest, campaign, row, 2, tmp_path / "m5.json", inspect_run=_inspect, task_id="task-a", before_recheck=mutate)

def test_write_once_refuses_existing_target(tmp_path: Path) -> None:
    manifest, campaign, row, roots = _fixture(tmp_path)
    target = tmp_path / "archive.json"; target.write_text("prior", encoding="utf-8")
    with pytest.raises(consumer.StageQConsumerError):
        consumer.consume_stage_q(manifest, campaign, row, 2, target, inspect_run=_inspect, task_id="task-a")
