"""Focused contract tests for the Stage-Q consumer boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


consumer = importlib.import_module("experiments.atomistic.he-importance.stage_q_consumer")
reader = importlib.import_module("experiments.toolkit.run_results")
completion = importlib.import_module("experiments.atomistic.he-importance.full_horizon_completion")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, object, object, dict[str, bytes]]:
    row = "ab" * 32
    campaign = tmp_path / "campaign"
    c_path = campaign / "cells" / row / "selection_receipt.json"
    c_path.parent.mkdir(parents=True)
    c_path.write_bytes(_canonical({"train_packet": {"content_hash": row, "payload": "fixed"}}))
    core = tmp_path / "groups" / ("b" * 64) / "task-7" / "attempt-S" / "core"
    core.mkdir(parents=True)
    delta = {"attempt_core_dir": str(core), "attempt_id": "attempt-S", "group_digest": "b" * 64,
             "load": {"mode": "fresh", "path": None}, "schema": "tpen-executor/attempt-delta-v1", "task_id": "task-7"}
    report = core / "attempt_report.json"
    report.write_bytes(_canonical({"row_id": row, "attempt_delta": delta}))
    completion_path = core.parent.parent / "full-horizon-completion.json"
    completion_path.write_bytes(b"completion")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"manifest")
    segment = SimpleNamespace(task_id="task-7", attempt_id="attempt-S", predecessor_checkpoint=None,
                              checkpoint_dir=str(core), next_iteration=1, completed_updates=1,
                              placement_receipt=None, resolved_config_identity=None)
    chain = SimpleNamespace(task_id="task-7", segments=(segment,))
    preserved = (SimpleNamespace(task_id="task-7", attempt_id="attempt-F", output_dir="failed",
                                 outcome="failed", reason="failed-or-incomplete", placement_receipt=None,
                                 core_outcome=None, successor_admission=None),)
    result = SimpleNamespace(chains=(chain,), preserved_attempts=preserved)
    before = {str(path): path.read_bytes() for path in (manifest, c_path, report, completion_path)}
    return manifest, campaign, tmp_path / "archive.json", result, core, before


def test_consumer_emits_one_sample_and_rechecks_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, campaign, archive, result, core, before = _fixture(tmp_path)
    monkeypatch.setattr(reader, "read", lambda path, inspect_run=None: result)
    monkeypatch.setattr(completion, "read_full_horizon_completion", lambda *args, **kwargs: True)
    body = consumer.consume_stage_q(manifest, campaign, "ab" * 32, 1, archive)
    assert body["sample_count"] == 1
    assert len(body["selected_chain"]) == 1
    assert len(body["preserved_attempts"]) == 1
    assert {path: Path(path).read_bytes() for path in before} == before
    assert archive.is_file()


@pytest.mark.parametrize("mutation", ["c-row", "report-identity", "not-complete", "duplicate-archive"])
def test_consumer_refuses_contract_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    manifest, campaign, archive, result, core, _ = _fixture(tmp_path)
    row = "ab" * 32
    if mutation == "c-row":
        (campaign / "cells" / row / "selection_receipt.json").write_bytes(_canonical({"train_packet": {"content_hash": "cd" * 32}}))
    elif mutation == "report-identity":
        report = core / "attempt_report.json"
        value = json.loads(report.read_bytes())
        value["attempt_delta"]["attempt_id"] = "wrong"
        report.write_bytes(_canonical(value))
    monkeypatch.setattr(reader, "read", lambda path, inspect_run=None: result)
    monkeypatch.setattr(completion, "read_full_horizon_completion", lambda *args, **kwargs: mutation != "not-complete")
    if mutation == "duplicate-archive":
        archive.write_text("prior", encoding="utf-8")
    with pytest.raises(consumer.StageQConsumerError):
        consumer.consume_stage_q(manifest, campaign, row, 1, archive)
    assert not archive.exists() or mutation == "duplicate-archive"
