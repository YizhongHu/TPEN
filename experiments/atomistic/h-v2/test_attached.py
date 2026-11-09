"""Unit controls for the H-v2 attached-executor activation route.

Exercises ``attached.py``'s plan builders, the study-side accepted-receipt
builder against synthetic ``run-manifest-v2`` fixtures shaped exactly like
the real schema (``experiments/toolkit/run_results.py``), and the
amendment's refusal matrix: admission refusal, core-run-incomplete,
failed/timed-out member, uncommitted tail, wrong source identity,
task-set mismatch, and the SMOKE #4 unknown-attempt-key regression. Only
``test_the_real_executor_cli_produces_a_manifest_this_reader_accepts``
drives the real installed executor CLI; every other test here uses
synthetic fixtures, as ``test_collect.py`` does for ``collect.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

HERE = Path(__file__).parent


def _module(name: str):
    key = f"h_v2_{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


attached = _module("attached")
protocol = _module("protocol")

from experiments.toolkit.run_results import ManifestRefusal  # noqa: E402


def _sha(data: bytes = b"x") -> str:
    return hashlib.sha256(data).hexdigest()


def _write_core_dir(
    core_dir: Path,
    *,
    git_commit: str = "a" * 40,
    resolved_config_text: str = "load:\n  mode: model_only\n",
    with_trajectory: bool = False,
    csv_bytes: bytes = b"draw_index,walker_index\n0,0\n",
    csv_sha256: str | None = None,
) -> None:
    core_dir.mkdir(parents=True, exist_ok=True)
    (core_dir / "resolved_config.yaml").write_text(resolved_config_text, encoding="utf-8")
    (core_dir / "metadata.json").write_text(json.dumps({"git_commit": git_commit}), encoding="utf-8")
    if with_trajectory:
        csv_path = core_dir / "sampled_eval_table.csv"
        csv_path.write_bytes(csv_bytes)
        actual_sha = csv_sha256 if csv_sha256 is not None else hashlib.sha256(csv_bytes).hexdigest()
        (core_dir / "sampled_eval_table.metadata.json").write_text(
            json.dumps({"csv_sha256": actual_sha}), encoding="utf-8"
        )


def _manifest_fixture(
    manifest_path: Path,
    *,
    task_id: str,
    attempt_id: str,
    output_dir: Path,
    status: str = "completed",
    checkpoint_dir: str = "/fake/checkpoint",
    next_iteration: int = 10,
    completed_updates: int = 10,
    predecessor_checkpoint: str | None = None,
    include_segment: bool = True,
    extra_attempt_key: str | None = None,
) -> None:
    group_digest = _sha(b"group")
    attempt: dict[str, Any] = {
        "schema": "tpen-executor/manifest-attempt-v1",
        "task_id": task_id,
        "attempt_id": attempt_id,
        "output_dir": str(output_dir),
        "start_receipt": {
            "schema": "tpen-executor/start-receipt-v1",
            "task_id": task_id,
            "attempt_id": attempt_id,
            "allocation_id": "alloc-1",
            "backend_job_id": "job-1",
            "group_digest": group_digest,
            "started_at_utc": "2026-01-01T00:00:00Z",
            "output_dir": str(output_dir),
        },
        "outcome_receipt": {
            "schema": "tpen-executor/outcome-receipt-v1",
            "task_id": task_id,
            "attempt_id": attempt_id,
            "group_digest": group_digest,
            "finished_at_utc": "2026-01-01T00:01:00Z",
            "status": status,
            "output_dir": str(output_dir),
            "result_digest": None,
        }
        if status is not None
        else None,
    }
    if extra_attempt_key is not None:
        attempt[extra_attempt_key] = "unexpected"
    segments = []
    if include_segment:
        segments.append(
            {
                "schema": "tpen-executor/accepted-segment-v1",
                "task_id": task_id,
                "attempt_id": attempt_id,
                "predecessor_checkpoint": predecessor_checkpoint,
                "checkpoint_dir": checkpoint_dir,
                "next_iteration": next_iteration,
                "completed_updates": completed_updates,
            }
        )
    payload = {
        "schema": "tpen-executor/run-manifest-v2",
        "group_id": "group-1",
        "group_digest": group_digest,
        "attempts": [attempt],
        "segments": segments,
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def _completed_evidence(checkpoint_dir: str, next_iteration: int, completed_updates: int) -> dict[str, Any]:
    return {
        "verdict": "completed",
        "run_status": "completed",
        "checkpoint": {
            "checkpoint_dir": checkpoint_dir,
            "schema_version": 1,
            "next_iteration": next_iteration,
            "completed_updates": completed_updates,
        },
    }


# --- plan builders -----------------------------------------------------


def test_build_h_v2_train_plan_shape(tmp_path: Path) -> None:
    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1"
    )
    assert plan.n_tasks == 1
    task = plan.tasks[0]
    assert task.logical_task_id == "h-v2-train-L1"
    assert task.params["kind"] == "train"
    assert Path(task.params["config"]).is_file()
    assert overrides[task.logical_task_id] == task.params["executor_overrides"]
    assert any(item.startswith("run.root=") for item in overrides[task.logical_task_id])
    assert any(item == "run.layout=flat" for item in overrides[task.logical_task_id])
    # The written config and the checked config must be the hydrogen contract.
    protocol.assert_hydrogen_contract(config, evaluation=False)


def test_build_h_v2_train_plan_requires_absolute_paths(tmp_path: Path) -> None:
    with pytest.raises(attached.HV2ReceiptRefusal):
        attached.build_h_v2_train_plan(output_dir="relative", results_root=tmp_path, lineage="L1")


def test_build_h_v2_train_plan_rejects_a_drifted_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_training_config = protocol.training_config

    def _bad_config() -> dict[str, Any]:
        config = original_training_config()
        config["system"]["n_particles"] = 2
        return config

    monkeypatch.setattr(protocol, "training_config", _bad_config)
    with pytest.raises(ValueError):
        attached.build_h_v2_train_plan(output_dir=tmp_path / "h", results_root=tmp_path / "r", lineage="L1")


def test_build_h_v2_eval_plan_shape(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    plan, overrides, config = attached.build_h_v2_eval_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1", checkpoint=checkpoint_dir
    )
    assert plan.n_tasks == 1
    task = plan.tasks[0]
    assert task.params["kind"] == "eval"
    assert task.params["checkpoint_dir"] == str(checkpoint_dir)
    assert any(item == f"load.path={checkpoint_dir}" for item in overrides[task.logical_task_id])
    protocol.assert_hydrogen_contract(config, evaluation=True)


# --- build_h_v2_accepted_receipt: success and refusal matrix ------------


def test_accepted_receipt_success_for_a_train_segment(tmp_path: Path) -> None:
    output_dir = tmp_path / "attempt"
    core_dir = output_dir / "core"
    _write_core_dir(core_dir, git_commit="c" * 40)
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir)

    payload = attached.build_h_v2_accepted_receipt(
        manifest_path,
        expected_task_ids=("t1",),
        expected_source_commit="c" * 40,
        inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
    )
    assert payload["schema"] == attached.H_V2_ACCEPTED_RECEIPT_SCHEMA
    assert payload["retry"] == "forbidden"
    (segment,) = payload["segments"]
    assert segment["task_id"] == "t1"
    assert segment["source_commit"] == "c" * 40
    assert segment["trajectory_csv_sha256"] is None
    assert "resolved_config" in segment["artifacts"]
    assert "metadata" in segment["artifacts"]


def test_accepted_receipt_success_for_an_eval_segment_records_trajectory(tmp_path: Path) -> None:
    output_dir = tmp_path / "attempt"
    core_dir = output_dir / "core"
    _write_core_dir(core_dir, git_commit="c" * 40, with_trajectory=True)
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir)

    payload = attached.build_h_v2_accepted_receipt(
        manifest_path,
        expected_task_ids=("t1",),
        expected_source_commit="c" * 40,
        inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
    )
    (segment,) = payload["segments"]
    assert segment["trajectory_csv_sha256"] is not None
    assert "trajectory_csv" in segment["artifacts"]
    assert "trajectory_metadata" in segment["artifacts"]


def test_accepted_receipt_refuses_core_run_incomplete(tmp_path: Path) -> None:
    output_dir = tmp_path / "attempt"
    _write_core_dir(output_dir / "core")
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir)

    def incomplete(path: Any) -> dict[str, Any]:
        evidence = _completed_evidence("/fake/checkpoint", 10, 10)
        evidence["verdict"] = "incomplete"
        return evidence

    with pytest.raises(ManifestRefusal) as excinfo:
        attached.build_h_v2_accepted_receipt(
            manifest_path, expected_task_ids=("t1",), expected_source_commit="a" * 40, inspect_run=incomplete
        )
    assert excinfo.value.code == "core-run-incomplete"


def test_accepted_receipt_refuses_a_failed_member(tmp_path: Path) -> None:
    output_dir = tmp_path / "attempt"
    _write_core_dir(output_dir / "core")
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir, status="failed")

    with pytest.raises(ManifestRefusal) as excinfo:
        attached.build_h_v2_accepted_receipt(
            manifest_path,
            expected_task_ids=("t1",),
            expected_source_commit="a" * 40,
            inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
        )
    assert excinfo.value.code == "segment-on-failed-attempt"


def test_accepted_receipt_refuses_an_uncommitted_attempt(tmp_path: Path) -> None:
    """No outcome receipt at all -> the segment references an unknown/absent outcome."""

    output_dir = tmp_path / "attempt"
    _write_core_dir(output_dir / "core")
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir, status=None)

    with pytest.raises(ManifestRefusal) as excinfo:
        attached.build_h_v2_accepted_receipt(
            manifest_path,
            expected_task_ids=("t1",),
            expected_source_commit="a" * 40,
            inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
        )
    assert excinfo.value.code == "segment-on-failed-attempt"


def test_accepted_receipt_refuses_an_uncommitted_trajectory_tail(tmp_path: Path) -> None:
    output_dir = tmp_path / "attempt"
    core_dir = output_dir / "core"
    _write_core_dir(core_dir, with_trajectory=True, csv_sha256="0" * 64)
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir)

    with pytest.raises(ManifestRefusal) as excinfo:
        attached.build_h_v2_accepted_receipt(
            manifest_path,
            expected_task_ids=("t1",),
            expected_source_commit="a" * 40,
            inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
        )
    assert excinfo.value.code == "uncommitted-tail"


def test_accepted_receipt_refuses_wrong_source_commit(tmp_path: Path) -> None:
    output_dir = tmp_path / "attempt"
    _write_core_dir(output_dir / "core", git_commit="b" * 40)
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir)

    with pytest.raises(attached.HV2ReceiptRefusal) as excinfo:
        attached.build_h_v2_accepted_receipt(
            manifest_path,
            expected_task_ids=("t1",),
            expected_source_commit="c" * 40,
            inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
        )
    assert excinfo.value.code == "wrong-source-commit"


def test_accepted_receipt_refuses_a_missing_declared_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "attempt"
    core_dir = output_dir / "core"
    _write_core_dir(core_dir, git_commit="a" * 40)
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir)

    with pytest.raises(attached.HV2ReceiptRefusal) as excinfo:
        attached.build_h_v2_accepted_receipt(
            manifest_path,
            expected_task_ids=("t1",),
            expected_source_commit="a" * 40,
            inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
            artifact_declarations={"checkpoint_payload": "checkpoints/model.pt"},
        )
    assert excinfo.value.code == "missing-artifact"


def test_accepted_receipt_refuses_task_set_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path / "attempt"
    _write_core_dir(output_dir / "core")
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir)

    with pytest.raises(ManifestRefusal) as excinfo:
        attached.build_h_v2_accepted_receipt(
            manifest_path,
            expected_task_ids=("other-task",),
            expected_source_commit="a" * 40,
            inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
        )
    assert excinfo.value.code == "task-set-mismatch"


def test_accepted_receipt_refuses_wrong_resolved_config_hash_across_a_chain(tmp_path: Path) -> None:
    """Two segments of the same task whose resolved_config.yaml bytes differ."""

    output_dir_1 = tmp_path / "attempt1"
    output_dir_2 = tmp_path / "attempt2"
    _write_core_dir(output_dir_1 / "core", resolved_config_text="load:\n  mode: model_only\n# v1\n")
    _write_core_dir(output_dir_2 / "core", resolved_config_text="load:\n  mode: model_only\n# v2\n")
    group_digest = _sha(b"group")
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "tpen-executor/run-manifest-v2",
                "group_id": "group-1",
                "group_digest": group_digest,
                "attempts": [
                    {
                        "schema": "tpen-executor/manifest-attempt-v1",
                        "task_id": "t1",
                        "attempt_id": attempt_id,
                        "output_dir": str(output_dir),
                        "start_receipt": {
                            "schema": "tpen-executor/start-receipt-v1",
                            "task_id": "t1",
                            "attempt_id": attempt_id,
                            "allocation_id": "alloc-1",
                            "backend_job_id": "job-1",
                            "group_digest": group_digest,
                            "started_at_utc": "2026-01-01T00:00:00Z",
                            "output_dir": str(output_dir),
                        },
                        "outcome_receipt": {
                            "schema": "tpen-executor/outcome-receipt-v1",
                            "task_id": "t1",
                            "attempt_id": attempt_id,
                            "group_digest": group_digest,
                            "finished_at_utc": "2026-01-01T00:01:00Z",
                            "status": "completed",
                            "output_dir": str(output_dir),
                            "result_digest": None,
                        },
                    }
                    for attempt_id, output_dir in (("a1", output_dir_1), ("a2", output_dir_2))
                ],
                "segments": [
                    {
                        "schema": "tpen-executor/accepted-segment-v1",
                        "task_id": "t1",
                        "attempt_id": "a1",
                        "predecessor_checkpoint": None,
                        "checkpoint_dir": "/fake/checkpoint-1",
                        "next_iteration": 5,
                        "completed_updates": 5,
                    },
                    {
                        "schema": "tpen-executor/accepted-segment-v1",
                        "task_id": "t1",
                        "attempt_id": "a2",
                        "predecessor_checkpoint": "/fake/checkpoint-1",
                        "checkpoint_dir": "/fake/checkpoint-2",
                        "next_iteration": 10,
                        "completed_updates": 10,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def inspector(path: Any) -> dict[str, Any]:
        path = Path(path)
        if path == output_dir_1 / "core":
            return _completed_evidence("/fake/checkpoint-1", 5, 5)
        return _completed_evidence("/fake/checkpoint-2", 10, 10)

    with pytest.raises(attached.HV2ReceiptRefusal) as excinfo:
        attached.build_h_v2_accepted_receipt(
            manifest_path, expected_task_ids=("t1",), expected_source_commit="a" * 40, inspect_run=inspector
        )
    assert excinfo.value.code == "wrong-resolved-config-hash"


def test_smoke_4_regression_unknown_attempt_key_is_refused(tmp_path: Path) -> None:
    """Pins the amendment's smoke #4 regression: read() refuses an unknown key."""

    output_dir = tmp_path / "attempt"
    _write_core_dir(output_dir / "core")
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(
        manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir, extra_attempt_key="placement_receipt"
    )

    with pytest.raises(ManifestRefusal) as excinfo:
        attached.build_h_v2_accepted_receipt(
            manifest_path,
            expected_task_ids=("t1",),
            expected_source_commit="a" * 40,
            inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
        )
    assert excinfo.value.code == "malformed-record"


# --- activate_h_v2_run: end-to-end with a faked executor CLI ------------


class _FakeDispatchResult:
    def __init__(self, returncode: int, manifest: Mapping[str, Any] | None, stderr: str = "") -> None:
        self.returncode = returncode
        self.manifest = manifest
        self.stdout = ""
        self.stderr = stderr


def _prepared_run(tmp_path: Path) -> "attached.PreparedHV2Run":
    plan, overrides, _ = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1"
    )
    return attached.prepare_h_v2_run(plan, overrides, output_dir=tmp_path / "handoff")


def test_activate_h_v2_run_propagates_admission_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executor CLI exit 2 (admission refusal): no manifest, non-zero exit."""

    prepared = _prepared_run(tmp_path)
    monkeypatch.setattr(
        attached, "check_profile", lambda *a, **k: attached.ProfileVerdict("p", "supported", (), None, 0)
    )
    monkeypatch.setattr(
        attached, "run_executor_cli", lambda *a, **k: _FakeDispatchResult(2, None, stderr="admission refused")
    )
    store = tmp_path / "store"
    store.mkdir()
    activation_receipt_path = tmp_path / "activation-receipt.json"
    with pytest.raises(attached.ActivationRefusal) as excinfo:
        attached.activate_h_v2_run(
            prepared,
            science_python=tmp_path / "python",
            executor_python=tmp_path / "python",
            allocation=tmp_path / "allocation.json",
            worker_config=tmp_path / "worker.yaml",
            store=store,
            activation_receipt_path=activation_receipt_path,
            h_v2_receipt_path=tmp_path / "h-v2-receipt.json",
            expected_source_commit="a" * 40,
        )
    assert excinfo.value.code == "executor-failed"
    failure_payload = json.loads(activation_receipt_path.read_text(encoding="utf-8"))
    assert failure_payload["status"] == "failed"
    assert failure_payload["refusal"]["code"] == "executor-exit-2"


def test_activate_h_v2_run_exits_non_zero_when_receipt_refuses_despite_cli_exit_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment: exit 0 from the executor CLI alone is not science complete."""

    prepared = _prepared_run(tmp_path)
    task_id = prepared.task_id
    output_dir = tmp_path / "attempt"
    _write_core_dir(output_dir / "core", git_commit="wrong-commit")
    store = tmp_path / "store"
    store.mkdir()
    manifest_path = store / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id=task_id, attempt_id="a1", output_dir=output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        attached, "check_profile", lambda *a, **k: attached.ProfileVerdict("p", "supported", (), None, 0)
    )
    monkeypatch.setattr(attached, "run_executor_cli", lambda *a, **k: _FakeDispatchResult(0, manifest))

    with pytest.raises(Exception):
        attached.activate_h_v2_run(
            prepared,
            science_python=tmp_path / "python",
            executor_python=tmp_path / "python",
            allocation=tmp_path / "allocation.json",
            worker_config=tmp_path / "worker.yaml",
            store=store,
            activation_receipt_path=tmp_path / "activation-receipt.json",
            h_v2_receipt_path=tmp_path / "h-v2-receipt.json",
            expected_source_commit="a" * 40,
            inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
        )
    failure_payload = json.loads((tmp_path / "h-v2-receipt.json").read_text(encoding="utf-8"))
    assert failure_payload["status"] == "failed"
