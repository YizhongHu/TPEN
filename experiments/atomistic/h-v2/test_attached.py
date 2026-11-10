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
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

HERE = Path(__file__).parent
REPO_ROOT = HERE.resolve().parents[2]


def _module(name: str):
    key = f"h_v2_{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


attached = _module("attached")
protocol = _module("protocol")

import experiments.toolkit.attached_activation as attached_activation  # noqa: E402
from experiments.toolkit.run_results import ManifestRefusal, read as read_manifest  # noqa: E402


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
        trajectory_dir = core_dir / "trajectory"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        csv_path = trajectory_dir / "sampled_eval_table.csv"
        csv_path.write_bytes(csv_bytes)
        actual_sha = csv_sha256 if csv_sha256 is not None else hashlib.sha256(csv_bytes).hexdigest()
        (trajectory_dir / "sampled_eval_table.metadata.json").write_text(
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


# --- prepare_h_v2_run: contract check on the exact bound composition ----


def test_prepare_h_v2_run_refuses_a_tampered_override(tmp_path: Path) -> None:
    """F2: an override appended after the builder ran must still be checked."""

    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1"
    )
    task_id = plan.tasks[0].logical_task_id
    tampered = {task_id: (*overrides[task_id], "system.n_particles=2")}
    with pytest.raises(ValueError):
        attached.prepare_h_v2_run(plan, tampered, config, output_dir=tmp_path / "handoff")


def test_prepare_h_v2_run_reaches_dispatch_normally_without_tampered_overrides(tmp_path: Path) -> None:
    """Untampered control: the real overrides compose and pass the contract."""

    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1"
    )
    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=tmp_path / "handoff")
    task_id = plan.tasks[0].logical_task_id
    assert prepared.overrides == tuple(overrides[task_id])
    assert Path(prepared.handoff.executor_input).is_file()


def test_prepare_h_v2_run_refuses_an_interpolated_seed_collision(tmp_path: Path) -> None:
    """D1/F2: an override whose VALUE is an unresolved interpolation that
    resolves to the same int as ``sampler.seed`` must still be refused as a
    forbidden training contract. Pins the delta-review bug where
    ``assert_hydrogen_contract`` compared the UNRESOLVED composition, so the
    left side was the literal string ``"${sampler.seed}"`` and never compared
    equal to any integer no matter what it resolved to.
    """

    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1"
    )
    task_id = plan.tasks[0].logical_task_id
    tampered = {task_id: (*overrides[task_id], "runtime.seed=${sampler.seed}")}
    with pytest.raises(ValueError, match="distinct"):
        attached.prepare_h_v2_run(plan, tampered, config, output_dir=tmp_path / "handoff")


def test_prepare_h_v2_run_accepts_distinct_literal_seeds_as_a_control(tmp_path: Path) -> None:
    """GREEN control: an override with a literal, still-distinct seed must
    compose and pass the contract, isolating the failure above to the
    interpolation hole rather than to overriding ``runtime.seed`` at all."""

    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1"
    )
    task_id = plan.tasks[0].logical_task_id
    literal_seed = 910011
    assert literal_seed != protocol.TRAINING_SEED
    assert literal_seed != protocol.SAMPLER_SEED
    tampered = {task_id: (*overrides[task_id], f"runtime.seed={literal_seed}")}
    prepared = attached.prepare_h_v2_run(plan, tampered, config, output_dir=tmp_path / "handoff")
    assert Path(prepared.handoff.executor_input).is_file()


def test_prepare_h_v2_run_accepts_a_valid_interpolated_n_particles(tmp_path: Path) -> None:
    """D1/F2, secondary instance: a VALID hydrogen composition expressed
    through an interpolation (``system.n_particles=${one_electron}`` with
    ``one_electron=1``) must be accepted once resolved. The unresolved-
    comparison bug false-refused it because the left side was the literal
    string ``"${one_electron}"``, never the integer ``1``.
    """

    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1"
    )
    task_id = plan.tasks[0].logical_task_id
    tampered = {
        task_id: (*overrides[task_id], "system.n_particles=${one_electron}", "one_electron=1")
    }
    attached.prepare_h_v2_run(plan, tampered, config, output_dir=tmp_path / "handoff")


# --- D3: artifact declarations derived from the bound composition -------


def test_prepare_h_v2_run_derives_initial_checkpoint_and_parameter_activity_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3: a configured InitialCheckpoint/ParameterActivity callback makes
    its own output mandatory, derived from the bound composition -- not
    merely declared by a caller restating it independently.
    """

    original_training_config = protocol.training_config

    def _with_extra_callbacks() -> dict[str, Any]:
        config = original_training_config()
        config["callbacks"].append(
            {"_target_": "tpen.callback.InitialCheckpoint", "output_dir": "${run.dir}/checkpoints"}
        )
        config["callbacks"].append(
            {"_target_": "tpen.callback.ParameterActivity", "output_path": "${run.dir}/parameter_activity.json"}
        )
        return config

    monkeypatch.setattr(protocol, "training_config", _with_extra_callbacks)
    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1"
    )
    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=tmp_path / "handoff")
    assert prepared.artifact_declarations["initial_checkpoint_manifest"] == "checkpoints/step_000000/manifest.json"
    assert prepared.artifact_declarations["initial_checkpoint_complete"] == "checkpoints/step_000000/COMPLETE"
    assert prepared.artifact_declarations["parameter_activity"] == "parameter_activity.json"


def test_prepare_h_v2_run_derives_the_trajectory_path_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3: EVAL trajectory artifacts are required at THEIR configured path,
    not a hardcoded default -- even when that differs from eval.yaml's own
    default ``${run.dir}/trajectory``.
    """

    original_evaluation_config = protocol.evaluation_config

    def _custom_trajectory_path(checkpoint: Path, *, stream: int = 0, root: Path | None = None) -> dict[str, Any]:
        config = original_evaluation_config(checkpoint, stream=stream, root=root)
        config["evaluator"]["tasks"][0]["output_dir"] = "${run.dir}/custom_traj"
        return config

    monkeypatch.setattr(protocol, "evaluation_config", _custom_trajectory_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    plan, overrides, config = attached.build_h_v2_eval_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1", checkpoint=checkpoint_dir
    )
    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=tmp_path / "handoff")
    assert prepared.artifact_declarations["trajectory_csv"] == "custom_traj/sampled_eval_table.csv"
    assert prepared.artifact_declarations["trajectory_metadata"] == "custom_traj/sampled_eval_table.metadata.json"


def test_activate_h_v2_run_refuses_a_mismatched_caller_artifact_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3: a caller-supplied declaration that disagrees with the bound
    composition's own derived requirement must refuse, not silently win."""

    original_training_config = protocol.training_config

    def _with_initial_checkpoint() -> dict[str, Any]:
        config = original_training_config()
        config["callbacks"].append(
            {"_target_": "tpen.callback.InitialCheckpoint", "output_dir": "${run.dir}/checkpoints"}
        )
        return config

    monkeypatch.setattr(protocol, "training_config", _with_initial_checkpoint)
    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1"
    )
    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=tmp_path / "handoff")
    assert prepared.artifact_declarations["initial_checkpoint_manifest"] == "checkpoints/step_000000/manifest.json"

    output_dir = tmp_path / "attempt"
    _write_core_dir(output_dir / "core", git_commit="a" * 40)
    store = tmp_path / "store"
    store.mkdir()
    manifest_path = store / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id=prepared.task_id, attempt_id="a1", output_dir=output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        attached, "check_profile", lambda *a, **k: attached.ProfileVerdict("p", "supported", (), None, 0)
    )
    monkeypatch.setattr(attached, "run_executor_cli", lambda *a, **k: _FakeDispatchResult(0, manifest))

    with pytest.raises(attached.HV2ReceiptRefusal) as excinfo:
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
            artifact_declarations={"initial_checkpoint_manifest": "wrong/path"},
        )
    assert excinfo.value.code == "artifact-declaration-mismatch"


def test_accepted_receipt_refuses_a_missing_derived_initial_checkpoint(tmp_path: Path) -> None:
    """D3: InitialCheckpoint's own configured output is mandatory once declared."""

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
            artifact_declarations={"initial_checkpoint_manifest": "checkpoints/step_000000/manifest.json"},
        )
    assert excinfo.value.code == "missing-artifact"


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
        stage=attached.EVAL_STAGE,
        inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
    )
    (segment,) = payload["segments"]
    assert segment["trajectory_csv_sha256"] is not None
    assert "trajectory_csv" in segment["artifacts"]
    assert "trajectory_metadata" in segment["artifacts"]
    # Positive assertion for the real eval.yaml:129 layout (core/trajectory/...),
    # not merely a characterization that the files were found.
    assert segment["artifacts"]["trajectory_csv"]["path"].endswith("trajectory/sampled_eval_table.csv")


def test_accepted_receipt_refuses_an_explicitly_declared_but_missing_trajectory_pair(tmp_path: Path) -> None:
    """An EVAL segment with no trajectory files at all must refuse, not omit."""

    output_dir = tmp_path / "attempt"
    core_dir = output_dir / "core"
    _write_core_dir(core_dir, git_commit="c" * 40, with_trajectory=False)
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir)

    with pytest.raises(attached.HV2ReceiptRefusal) as excinfo:
        attached.build_h_v2_accepted_receipt(
            manifest_path,
            expected_task_ids=("t1",),
            expected_source_commit="c" * 40,
            stage=attached.EVAL_STAGE,
            inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
        )
    assert excinfo.value.code == "missing-artifact"


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
            stage=attached.EVAL_STAGE,
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
    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1"
    )
    return attached.prepare_h_v2_run(plan, overrides, config, output_dir=tmp_path / "handoff")


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
    # F5: the H-v2 receipt must carry the ORIGINAL named refusal code, not a
    # generic placeholder pointing back at itself.
    assert failure_payload["refusal"]["code"] == "wrong-source-commit"


def test_activate_h_v2_run_operational_receipt_still_records_success_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D4: the generic activation receipt is unaffected by the H-v2 refusal.

    It is written (as "activated") before the H-v2 accepted-receipt step even
    runs, so it is a distinct, correctly-populated operational receipt.
    Imported from the round-1 reviewer tip 8a8cb2439a8bce4bcfb1eebbc72e8f9a8e521f02.
    """

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

    activation_receipt_path = tmp_path / "activation-receipt.json"
    with pytest.raises(attached.HV2ReceiptRefusal):
        attached.activate_h_v2_run(
            prepared,
            science_python=tmp_path / "python",
            executor_python=tmp_path / "python",
            allocation=tmp_path / "allocation.json",
            worker_config=tmp_path / "worker.yaml",
            store=store,
            activation_receipt_path=activation_receipt_path,
            h_v2_receipt_path=tmp_path / "h-v2-receipt.json",
            expected_source_commit="c" * 40,
            inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
        )
    operational_payload = json.loads(activation_receipt_path.read_text(encoding="utf-8"))
    assert operational_payload["status"] == "activated"


def test_activate_h_v2_run_admission_refusal_retires_a_stale_accepted_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5: an earlier accepted receipt at h_v2_receipt_path must not survive."""

    prepared = _prepared_run(tmp_path)
    h_v2_receipt_path = tmp_path / "h-v2-receipt.json"
    h_v2_receipt_path.write_text(
        json.dumps({"schema": attached.H_V2_ACCEPTED_RECEIPT_SCHEMA, "segments": [], "retry": "forbidden"}),
        encoding="utf-8",
    )

    def _refuse_profile(*args: Any, **kwargs: Any) -> attached.ProfileVerdict:
        raise attached.ActivationRefusal("profile-refused", "callback whitelist blocker")

    monkeypatch.setattr(attached, "check_profile", _refuse_profile)
    store = tmp_path / "store"
    store.mkdir()
    with pytest.raises(attached.ActivationRefusal) as excinfo:
        attached.activate_h_v2_run(
            prepared,
            science_python=tmp_path / "python",
            executor_python=tmp_path / "python",
            allocation=tmp_path / "allocation.json",
            worker_config=tmp_path / "worker.yaml",
            store=store,
            activation_receipt_path=tmp_path / "activation-receipt.json",
            h_v2_receipt_path=h_v2_receipt_path,
            expected_source_commit="a" * 40,
        )
    assert excinfo.value.code == "profile-refused"
    failure_payload = json.loads(h_v2_receipt_path.read_text(encoding="utf-8"))
    assert failure_payload["status"] == "failed"
    assert failure_payload["refusal"]["code"] == "profile-refused"
    assert "segments" not in failure_payload


def test_run_executor_cli_preflight_refusal_leaves_stale_receipts_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D2: a REAL ``run_executor_cli`` preflight refusal
    (``_validate_attached_sidecars`` raising ``handoff-group-mismatch``),
    which fires before any executor subprocess is spawned, must still cause
    ``activate_h_v2_run`` to retire both stale receipts with the real
    refusal.  ``check_profile`` is faked (profile checking is not under
    test here); ``run_executor_cli`` itself is the real function from
    ``attached_activation`` -- only ``subprocess.run`` is replaced with a
    spy that fails loudly if the preflight does not refuse first, so this
    demonstrates the actual helper's refusal rather than a monkeypatched
    stand-in for it.
    """

    prepared = _prepared_run(tmp_path)

    # Tamper with the on-disk executor_input.json: an unrelated extra key
    # changes payload_digest(group) without changing its validated shape, so
    # _validate_attached_sidecars's FIRST check (group digest) is the one
    # that fires: handoff.group_digest was computed and frozen before this
    # write, from the untampered content.
    executor_input_path = Path(prepared.handoff.executor_input)
    group_payload = json.loads(executor_input_path.read_text(encoding="utf-8"))
    group_payload["_review_delta_tamper"] = "forces a group-digest mismatch"
    executor_input_path.write_text(json.dumps(group_payload), encoding="utf-8")

    monkeypatch.setattr(
        attached, "check_profile", lambda *a, **k: attached.ProfileVerdict("p", "supported", (), None, 0)
    )

    def _must_not_spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "must not be reached: the sidecar preflight must refuse before any subprocess is spawned"
        )

    monkeypatch.setattr(attached_activation.subprocess, "run", _must_not_spawn)

    store = tmp_path / "store"
    store.mkdir()
    activation_receipt_path = tmp_path / "activation-receipt.json"
    h_v2_receipt_path = tmp_path / "h-v2-receipt.json"
    stale_activation = {
        "schema": attached_activation.ATTACHED_ACTIVATION_SCHEMA,
        "status": "activated",
        "marker": "stale-activation-receipt-from-an-earlier-generation",
    }
    stale_h_v2 = {
        "schema": attached.H_V2_ACCEPTED_RECEIPT_SCHEMA,
        "segments": [],
        "marker": "stale-h-v2-receipt-from-an-earlier-generation",
    }
    activation_receipt_path.write_text(json.dumps(stale_activation), encoding="utf-8")
    h_v2_receipt_path.write_text(json.dumps(stale_h_v2), encoding="utf-8")

    with pytest.raises(attached.ActivationRefusal) as excinfo:
        attached.activate_h_v2_run(
            prepared,
            science_python=tmp_path / "python",
            executor_python=tmp_path / "python",
            allocation=tmp_path / "allocation.json",
            worker_config=tmp_path / "worker.yaml",
            store=store,
            activation_receipt_path=activation_receipt_path,
            h_v2_receipt_path=h_v2_receipt_path,
            expected_source_commit="a" * 40,
        )
    assert excinfo.value.code == "handoff-group-mismatch"

    activation_payload = json.loads(activation_receipt_path.read_text(encoding="utf-8"))
    h_v2_payload = json.loads(h_v2_receipt_path.read_text(encoding="utf-8"))
    statuses = (activation_payload.get("status"), h_v2_payload.get("status"))
    codes = (
        activation_payload.get("refusal", {}).get("code"),
        h_v2_payload.get("refusal", {}).get("code"),
    )
    markers = (activation_payload.get("marker"), h_v2_payload.get("marker"))
    assert statuses == ("failed", "failed")
    assert codes == ("handoff-group-mismatch", "handoff-group-mismatch")
    assert markers == (None, None)


def test_h_v2_receipt_preserves_the_readers_named_code_on_a_malformed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D2: when ``write_activation_receipt`` raises because the manifest
    reader refuses a malformed record (here, an unknown attempt key), the
    operational P10 receipt correctly preserves the reader's own code
    (``malformed-record``) -- that happens inside ``write_activation_receipt``
    itself. ``activate_h_v2_run`` must preserve that SAME code on the
    study-side ``h_v2_receipt_path`` and on the raised exception, not
    substitute a generic ``reader-failed`` placeholder.
    """

    prepared = _prepared_run(tmp_path)
    task_id = prepared.task_id
    output_dir = tmp_path / "attempt"
    _write_core_dir(output_dir / "core")
    store = tmp_path / "store"
    store.mkdir()
    manifest_path = store / "run-manifest.json"
    _manifest_fixture(
        manifest_path, task_id=task_id, attempt_id="a1", output_dir=output_dir, extra_attempt_key="placement_receipt"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        attached, "check_profile", lambda *a, **k: attached.ProfileVerdict("p", "supported", (), None, 0)
    )
    monkeypatch.setattr(attached, "run_executor_cli", lambda *a, **k: _FakeDispatchResult(0, manifest))

    activation_receipt_path = tmp_path / "activation-receipt.json"
    h_v2_receipt_path = tmp_path / "h-v2-receipt.json"
    with pytest.raises(ManifestRefusal) as excinfo:
        attached.activate_h_v2_run(
            prepared,
            science_python=tmp_path / "python",
            executor_python=tmp_path / "python",
            allocation=tmp_path / "allocation.json",
            worker_config=tmp_path / "worker.yaml",
            store=store,
            activation_receipt_path=activation_receipt_path,
            h_v2_receipt_path=h_v2_receipt_path,
            expected_source_commit="a" * 40,
            inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 10, 10),
        )

    operational_payload = json.loads(activation_receipt_path.read_text(encoding="utf-8"))
    h_v2_payload = json.loads(h_v2_receipt_path.read_text(encoding="utf-8"))
    statuses = (operational_payload.get("status"), h_v2_payload.get("status"))
    codes = (
        operational_payload.get("refusal", {}).get("code"),
        h_v2_payload.get("refusal", {}).get("code"),
    )
    assert statuses == ("failed", "failed")
    assert codes == ("malformed-record", "malformed-record")
    assert excinfo.value.code == "malformed-record"


def test_activate_h_v2_run_profile_checks_the_composed_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F2: activate_h_v2_run must profile-check the SAME bound composition."""

    prepared = _prepared_run(tmp_path)
    captured: dict[str, Any] = {}

    def _record_check_profile(science_python: Any, config: Any, **kwargs: Any) -> attached.ProfileVerdict:
        captured["overrides"] = kwargs.get("overrides")
        return attached.ProfileVerdict("p", "supported", (), None, 0)

    monkeypatch.setattr(attached, "check_profile", _record_check_profile)
    monkeypatch.setattr(
        attached, "run_executor_cli", lambda *a, **k: _FakeDispatchResult(2, None, stderr="admission refused")
    )
    store = tmp_path / "store"
    store.mkdir()
    with pytest.raises(attached.ActivationRefusal):
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
        )
    assert captured["overrides"] == prepared.overrides


# --- activate_h_v2_run: F1, the EVAL route is explicitly unsupported ----


def test_activate_h_v2_run_refuses_the_eval_route_as_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F1: EVAL must refuse before any profile check or dispatch."""

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    plan, overrides, config = attached.build_h_v2_eval_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="L1", checkpoint=checkpoint_dir
    )
    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=tmp_path / "handoff")

    def _must_not_be_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not be reached: EVAL refuses before any profile check or dispatch")

    monkeypatch.setattr(attached, "check_profile", _must_not_be_called)
    monkeypatch.setattr(attached, "run_executor_cli", _must_not_be_called)
    store = tmp_path / "store"
    store.mkdir()
    activation_receipt_path = tmp_path / "activation-receipt.json"
    h_v2_receipt_path = tmp_path / "h-v2-receipt.json"
    with pytest.raises(attached.ActivationRefusal) as excinfo:
        attached.activate_h_v2_run(
            prepared,
            science_python=tmp_path / "python",
            executor_python=tmp_path / "python",
            allocation=tmp_path / "allocation.json",
            worker_config=tmp_path / "worker.yaml",
            store=store,
            activation_receipt_path=activation_receipt_path,
            h_v2_receipt_path=h_v2_receipt_path,
            expected_source_commit="a" * 40,
        )
    assert excinfo.value.code == attached.EVAL_ROUTE_UNSUPPORTED_CODE
    for path in (activation_receipt_path, h_v2_receipt_path):
        failure_payload = json.loads(path.read_text(encoding="utf-8"))
        assert failure_payload["status"] == "failed"
        assert failure_payload["refusal"]["code"] == attached.EVAL_ROUTE_UNSUPPORTED_CODE


@pytest.mark.skip(
    reason=(
        "blocked on an inter-lane core/executor contract decision for completed-EVAL "
        "checkpoint evidence (tpen/inspection.py's missing-checkpoint verdict and "
        "executor/src/tpen_executor/dispatch/contract.py's no-accepted-segment-without-"
        "checkpoint rule); see TO item 9d3aa0a3-0f59-43e6-b52b-dcf39a39785d, note "
        "blocker-recovery-profile-callback-whitelist-20261109. Unskip once that "
        "contract lands and activate_h_v2_run's EVAL_STAGE refusal is lifted."
    )
)
def test_eval_route_accepts_a_genuinely_completed_eval_run_once_the_checkpoint_evidence_contract_lands(
    tmp_path: Path,
) -> None:
    """Placeholder for the real EVAL-to-reader acceptance path (see skip reason)."""

    raise NotImplementedError


# --- real executor CLI: the only test here that is not a synthetic fixture


def _real_cli_allocation_and_worker(tmp_path: Path, *, science_python: str, placeholder_config: str) -> tuple[Path, Path]:
    """Minimal real allocation/worker-config fixtures for ``--dispatcher local``.

    Mirrors ``executor/tests/test_cli_cross_project.py``'s worker shape
    (``WorkerProgram`` + ``PreparedPythonEnvironment`` + ``SubprocessWorker``
    + the real ``TPENCommand``/``TPENEvidence`` runners, which really invoke
    ``tpen.run.main``/``python -m tpen.inspection``).  ``runner.config`` is a
    placeholder: the executor overrides it per task from the configured-runs
    sidecar this module's own ``write_attached_handoff`` already wrote.
    """

    allocation_path = tmp_path / "allocation.json"
    allocation_path.write_text(
        json.dumps(
            {
                "schema": "tpen-executor/configured-allocation-v1",
                "allocation_id": "h-v2-real-cli-alloc",
                "backend_job_id": "h-v2-real-cli-job",
                "facts": {},
            }
        ),
        encoding="utf-8",
    )
    worker_path = tmp_path / "worker.yaml"
    worker_path.write_text(
        "worker:\n"
        "  _target_: tpen_executor.programs.WorkerProgram\n"
        "  _recursive_: false\n"
        "  environment:\n"
        "    _target_: tpen_executor.environments.PreparedPythonEnvironment\n"
        "    _recursive_: false\n"
        f"    python: {json.dumps(science_python)}\n"
        "    env:\n"
        f"      PYTHONPATH: {json.dumps(str(REPO_ROOT))}\n"
        "  process:\n"
        "    _target_: tpen_executor.workers.SubprocessWorker\n"
        "    _recursive_: false\n"
        "  runner:\n"
        "    _target_: tpen_executor.runners.TPENCommand\n"
        "    _recursive_: false\n"
        f"    config: {json.dumps(placeholder_config)}\n"
        "  evidence:\n"
        "    _target_: tpen_executor.runners.TPENEvidence\n"
        "    _recursive_: false\n",
        encoding="utf-8",
    )
    return allocation_path, worker_path


def test_the_real_executor_cli_produces_a_manifest_this_reader_accepts(tmp_path: Path) -> None:
    """Item 4(b) / amendment smoke #4 follow-up: drive the REAL executor CLI.

    Every other test in this file uses a synthetic ``run-manifest-v2``
    fixture and a monkeypatched ``run_executor_cli``/``check_profile``, as
    ``test_collect.py`` does for ``collect.py``.  This test instead runs the
    real, installed ``python -m tpen_executor dispatch`` subprocess (not a
    fake dispatch) against a real, tiny H-v2 TRAIN task built through this
    module's own ``build_h_v2_train_plan``/``prepare_h_v2_run``, and confirms
    ``experiments.toolkit.run_results.read()`` genuinely accepts the wire
    manifest the real executor publishes.  ``check_profile`` is deliberately
    not exercised here (that is the known, separately-tracked recovery-profile
    callback-whitelist blocker, orthogonal to wire-format acceptance); this
    test only proves the executor's real bytes are readable.

    It then closes the amendment's SMOKE #4 regression
    (``item cb174fd4``, note ``smoke-result-7830278``: an unexpected attempt
    key reached a fake-dispatch test green while the real executor's reader
    refused it) with REAL executor-produced bytes rather than only the
    synthetic fixture in ``test_smoke_4_regression_unknown_attempt_key_is_refused``.
    """

    science_python = os.environ.get("H_V2_SCIENCE_PYTHON")
    executor_python = os.environ.get("H_V2_EXECUTOR_PYTHON")
    if not science_python:
        pytest.skip("named skip: H_V2_SCIENCE_PYTHON is unset")
    if not executor_python:
        pytest.skip("named skip: H_V2_EXECUTOR_PYTHON is unset")

    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff",
        results_root=tmp_path / "results",
        lineage="real-cli",
        n_walkers=4,
        max_steps=2,
    )
    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=tmp_path / "handoff")
    allocation_path, worker_path = _real_cli_allocation_and_worker(
        tmp_path, science_python=science_python, placeholder_config=prepared.effective_config
    )
    store = tmp_path / "store"

    dispatch = attached.run_executor_cli(
        executor_python,
        prepared.handoff,
        allocation=allocation_path,
        worker_config=worker_path,
        store=store,
        dispatcher="local",
        timeout=180.0,
    )
    assert dispatch.manifest is not None, (dispatch.returncode, dispatch.stdout, dispatch.stderr)
    assert dispatch.manifest["schema"] == "tpen-executor/run-manifest-v2"

    manifest_path = store / "run-manifest.json"
    result = read_manifest(manifest_path)
    assert len(result.chains) == 1
    assert result.chains[0].task_id == prepared.task_id
    assert len(result.chains[0].segments) == 1

    # SMOKE #4 regression, closed with REAL executor bytes: an unknown
    # attempt key in this REAL manifest must still be refused.
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["attempts"][0]["placement_receipt"] = {"unexpected": True}
    tampered_path = tmp_path / "tampered-manifest.json"
    tampered_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ManifestRefusal) as excinfo:
        read_manifest(tampered_path)
    assert excinfo.value.code == "malformed-record"
