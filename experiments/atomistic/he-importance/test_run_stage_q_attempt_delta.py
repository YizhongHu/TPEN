"""Stage-Q integration checks for the closed attempt-delta seam."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest


runner = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")
delta_module = importlib.import_module("experiments.atomistic.he-importance.attempt_delta")


def _delta(mode: str = "fresh") -> SimpleNamespace:
    return SimpleNamespace(
        attempt_core_dir=Path("/store/groups/g/task/attempt/core"),
        attempt_id="attempt",
        group_digest="g",
        load_mode=mode,
        load_path=None if mode == "fresh" else Path("/store/groups/g/task/previous/core/checkpoint"),
        task_id="task",
    )


def _delta_json(core: Path, *, mode: str = "fresh") -> str:
    value = {
        "attempt_core_dir": str(core),
        "attempt_id": core.parent.name,
        "group_digest": core.parent.parent.parent.name,
        "load": {"mode": mode, "path": None if mode == "fresh" else str(core.parent.parent.parent / "previous" / "core" / "checkpoint")},
        "schema": delta_module.SCHEMA,
        "task_id": core.parent.parent.name,
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _manifest_args(tmp_path: Path, row_id: str, raw_delta: str) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=tmp_path / "campaign",
        dry_run=False,
        manifest=runner.scan_manifest.MANIFEST_PATH,
        row=row_id,
        attempt_delta=raw_delta,
    )


def test_delta_load_is_one_closed_runner_step() -> None:
    config = OmegaConf.create({"runner": {}})
    runner._apply_attempt_delta_load(config, _delta("train_resume"))
    assert OmegaConf.to_container(config.runner.load, resolve=True) == {
        "mode": "train_resume",
        "path": "/store/groups/g/task/previous/core/checkpoint",
    }


def test_fresh_delta_does_not_add_a_runner_load() -> None:
    config = OmegaConf.create({"runner": {}})
    runner._apply_attempt_delta_load(config, _delta("fresh"))
    assert OmegaConf.select(config, "runner.load", default=None) is None


def test_c_comparison_keeps_science_and_removes_only_operational_paths() -> None:
    report = {
        "selection": {"selected": {"context": "O-raw"}},
        "walkers": 4096,
        "roster_vs_design": {"status": "PASS"},
        "train_packet": {"content_hash": "row"},
        "resolved_config": {
            "trainer": {"max_steps": 2000},
            "run": {"dir": "/attempt/core", "root": "/attempt", "run_id": "x", "layout": "flat"},
        },
    }
    c = runner._cell_science_report(report)
    assert c != report
    assert c["resolved_config"]["trainer"] == {"max_steps": 2000}
    assert c["resolved_config"]["run"] == {"layout": "flat"}


def test_delta_malformed_refuses_before_manifest_guard_or_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def unexpected_manifest(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("malformed delta reached manifest loading")

    monkeypatch.setattr(runner.scan_manifest, "load_manifest", unexpected_manifest)
    args = _manifest_args(tmp_path, "row", "{}")
    with pytest.raises(delta_module.AttemptDeltaError):
        runner.execute(args)
    assert not called
    assert not any(tmp_path.iterdir())


def test_attempt_delta_parser_rejects_before_any_stage_work() -> None:
    raw = '{"attempt_core_dir":"/store/groups/g/task/attempt/core","attempt_id":"attempt","group_digest":"g","load":{"mode":"fresh","path":null},"schema":"wrong","task_id":"task"}'
    with pytest.raises(delta_module.AttemptDeltaError):
        delta_module.parse_attempt_delta(raw)


def test_execute_delta_fresh_and_resume_use_only_physical_core(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = runner.scan_manifest.load_manifest()
    row_id = next(row["row_id"] for row in manifest["rows"] if row["context"] == "O-raw" and row["seed_label"] == 820001 and row["status"] == "available")
    logical_cells = runner.stage_coordinate.materialize_intended_configurations(
        tmp_path / "campaign",
        availability=runner.optimizer_grid.roster_availability(),
        stages=("O1",),
    )
    logical_path = next(cell.output_path for cell in logical_cells if cell.content_hash == row_id)
    facts = {"global_rank": 0, "global_size": 1, "local_rank": 0, "local_size": 1, "node_rank": 0, "node_size": 1, "host": "delta-test", "pid": 1, "device": "cuda:0", "job_id": "delta", "device_identity": None}
    monkeypatch.setattr(runner, "_execution_facts", lambda: dict(facts))
    observed: list[dict[str, object]] = []

    def fake_run(config: object, **_kwargs: object) -> int:
        observed.append({"run_dir": str(config.run.dir), "load": OmegaConf.to_container(config.runner.load, resolve=True) if OmegaConf.select(config, "runner.load", default=None) is not None else None})
        Path(config.run.dir).joinpath("resolved_config.yaml").write_text("resolved\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner.launch, "run_from_config", fake_run)

    first_core = tmp_path / "store" / "groups" / "group" / "task" / "attempt-1" / "core"
    assert runner.execute(_manifest_args(tmp_path, row_id, _delta_json(first_core))) == 0
    c_path = tmp_path / "campaign" / "cells" / row_id / "selection_receipt.json"
    assert c_path.is_file()
    assert (first_core / "attempt_report.json").is_file()
    assert not (first_core / "COMPLETE").exists()
    assert not logical_path.exists()

    second_core = tmp_path / "store" / "groups" / "group" / "task" / "attempt-2" / "core"
    assert runner.execute(_manifest_args(tmp_path, row_id, _delta_json(second_core, mode="train_resume"))) == 0
    assert observed[0]["run_dir"] == str(first_core)
    assert observed[0]["load"] is None
    assert observed[1] == {"run_dir": str(second_core), "load": {"mode": "train_resume", "path": str(second_core.parent.parent.parent / "previous" / "core" / "checkpoint")}}

    mutated = json.loads(c_path.read_text(encoding="utf-8"))
    mutated["walkers"] = 1
    c_path.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    third_core = tmp_path / "store" / "groups" / "group" / "task" / "attempt-3" / "core"
    with pytest.raises(runner.StageQSelectionError, match="canonical expected C"):
        runner.execute(_manifest_args(tmp_path, row_id, _delta_json(third_core, mode="train_resume")))
    assert not third_core.exists()
