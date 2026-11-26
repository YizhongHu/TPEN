"""Focused contract tests for He-v1 external-checkpoint evaluation plans."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Guard BEFORE importing torch.  Four sibling test modules already do this
# (test_production_grid.py, test_adapter_oneqmc.py, test_adapter_deepqmc.py,
# test_parsl_attach.py); this one did not, so on a machine without torch it
# failed at COLLECTION and aborted the entire experiments run rather than
# skipping itself.  One unguarded module-scope import is enough to stop all
# 1171 experiments tests from being run together.
pytest.importorskip("torch", reason="torch-side evaluation canary cannot import without torch")

import torch
import yaml

from tpen.data.batch import Walkers
from tpen.evaluation.generators import MCMCGenerator
from tpen.evaluation.protocols import EvaluationContext

STUDY_DIR = Path(__file__).resolve().parent
GRID_PATH = STUDY_DIR / "configs" / "eval_canary.yaml"
GRID_42_PATH = STUDY_DIR / "configs" / "eval_42.yaml"
EVALUATION_SHA = "a" * 40


def _load_collect() -> ModuleType:
    path = STUDY_DIR / "collect.py"
    spec = importlib.util.spec_from_file_location("he_v1_canary_collect", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collect = _load_collect()
canary = collect.canary
eval_stage = collect.eval_stage
launch = collect.launch_stage
plan = collect.plan_stage


def _checkpoint(root: Path, source_id: str, step: int) -> tuple[Path, dict[str, Any]]:
    checkpoint_dir = root / source_id
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(f"real model {step}\n".encode())
    (checkpoint_dir / "resolved_config.yaml").write_text(
        "experiment:\n  name: real-he-v1-training\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        # Deliberately independent of the checkpoint coordinate: skipped
        # updates are allowed, but this real count must remain immutable.
        "completed_updates": step - 3,
        "files": {"model": "model.pt", "resolved_config": "resolved_config.yaml"},
        "provenance": {"git_sha": "b" * 40, "tpen_version": "1.2.3"},
    }
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    source = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "next_iteration": step,
        "completed_updates": step - 3,
        "model_sha256": canary.file_sha256(checkpoint_dir / "model.pt"),
        "manifest_sha256": canary.file_sha256(checkpoint_dir / "manifest.json"),
        "complete_sha256": canary.file_sha256(checkpoint_dir / "COMPLETE"),
        "training_source_sha": "b" * 40,
    }
    return checkpoint_dir, source


def _case(tmp_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    source_root = tmp_path / "external-real-checkpoints"
    _, source_25 = _checkpoint(source_root, "actual-step-025000", 25_000)
    _, source_50 = _checkpoint(source_root, "actual-step-050000", 50_000)
    source_map = {
        "schema": canary.SOURCE_SCHEMA,
        "sources": {
            "actual-step-025000": source_25,
            "actual-step-050000": source_50,
        },
    }
    source_map_path = tmp_path / "checkpoint-sources.yaml"
    source_map_path.write_text(yaml.safe_dump(source_map, sort_keys=True), encoding="utf-8")
    grid = canary.load_grid(GRID_PATH)
    sources = canary.load_source_map(source_map_path)
    rows = canary.expand_rows(
        grid, canary.reconcile_grid_sources(grid, sources)
    )
    manifest = canary.build_manifest(
        grid=grid,
        rows=rows,
        attempt_id="20260825T120000",
        results_root=tmp_path / "results",
        grid_path=GRID_PATH,
        source_map_path=source_map_path,
        evaluation_git_sha=EVALUATION_SHA,
        created_at="2026-08-25T12:00:00-04:00",
    )
    return manifest, source_map_path, source_map


def test_canary_expands_exactly_two_separately_addressable_energy_rows(
    tmp_path: Path,
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)

    assert manifest["n_rows"] == manifest["n_eval_rows"] == 2
    assert [row["checkpoint_step"] for row in manifest["rows"]] == [25_000, 50_000]
    assert [row["row_id"] for row in manifest["rows"]] == [
        "eval-canary-step000025000",
        "eval-canary-step000050000",
    ]
    assert all(row["task_names"] == ["mcmc_energy"] for row in manifest["rows"])
    assert all(row["record_capacity"] == 64 for row in manifest["rows"])
    assert all(row["burn_in"] == 4 for row in manifest["rows"])
    assert all(row["discard_draws"] == 0 for row in manifest["rows"])
    assert all(row["depends_on"] == [] for row in manifest["rows"])
    assert all("checkpoint_dir" not in row for row in manifest["rows"])
    assert all(
        "checkpoint_dir" not in json.dumps(row["checkpoint_source"])
        for row in manifest["rows"]
    )
    assert canary.file_sha256(source_map_path) == manifest["source_map_sha256"]
    selected = launch.select_rows(
        manifest, kinds=[], row_ids=["eval-canary-step000050000"]
    )
    assert [row["checkpoint_step"] for row in selected] == [50_000]


def test_frozen_grid_declares_42_rows_and_all_required_arms() -> None:
    grid = canary.load_grid(GRID_42_PATH)
    rows = grid["rows"]
    assert len(rows) == 42
    assert {step: sum(row["checkpoint_step"] == step for row in rows) for step in (25_000, 50_000)} == {
        25_000: 21, 50_000: 21
    }
    for step in (25_000, 50_000):
        subset = [row for row in rows if row["checkpoint_step"] == step]
        assert [row["seed"] for row in subset[:4]] == [1000, 1001, 1002, 1003]
        assert [row["seed"] for row in subset[4:8]] == [2000, 2001, 2002, 2003]
        assert [row["seed"] for row in subset[8:12]] == [3000, 3001, 3100, 3101]
        assert subset[12]["seed"] == 5000
        assert [row["seed"] for row in subset[13:20]] == list(range(4000, 4007))
        assert subset[20]["seed"] == 6000
    assert all(row["resources"]["constraint"] == "a100" for row in rows)
    assert sum("factor_response" in row["task_names"] for row in rows) == 4
    assert all(
        row["task_names"] == ["mcmc_energy"]
        for row in rows
        if row["seed"] in {1000, 1001, 1002, 1003, 2000, 2001, 2002, 2003, 3000, 3001, 3100, 3101}
    )


def test_frozen_grid_schema_matches_canary_consumer_constant() -> None:
    payload = yaml.safe_load(GRID_42_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == canary.GRID_SCHEMA
    assert canary.load_grid(GRID_42_PATH)["schema"] == canary.GRID_SCHEMA
    assert canary.METRIC_NAMESPACE_SPEC_KEY == collect.METRIC_NAMESPACE_SPEC_KEY


def test_config_identity_hash_ignores_yaml_mapping_order(tmp_path: Path) -> None:
    """Formatting and mapping order must not change the config identity."""

    config = yaml.safe_load(
        (STUDY_DIR / "configs" / "eval.yaml").read_text(encoding="utf-8")
    )
    reordered = dict(reversed(list(config.items())))
    original_path = tmp_path / "eval-original.yaml"
    reordered_path = tmp_path / "eval-reordered.yaml"
    original_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    reordered_path.write_text(yaml.safe_dump(reordered, sort_keys=False), encoding="utf-8")

    overrides = ["run.root=results", "run.run_id=row-1"]
    assert yaml.safe_load(original_path.read_text(encoding="utf-8")) == yaml.safe_load(
        reordered_path.read_text(encoding="utf-8")
    )
    assert eval_stage.config_identity_hash(
        original_path, overrides
    ) == eval_stage.config_identity_hash(reordered_path, overrides)
    # Mutation kill: replacing the canonical JSON payload with read_bytes()
    # makes this assertion fail on the reordered YAML bytes.


def test_config_identity_hash_changes_for_semantic_config_edit(tmp_path: Path) -> None:
    """A meaningful scalar change must remain visible in the identity."""

    config = yaml.safe_load(
        (STUDY_DIR / "configs" / "eval.yaml").read_text(encoding="utf-8")
    )
    changed = copy.deepcopy(config)
    changed["evaluation_sampler"]["proposal_scale"] = 0.75
    original_path = tmp_path / "eval-original.yaml"
    changed_path = tmp_path / "eval-changed.yaml"
    original_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    changed_path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")

    overrides = ["run.root=results", "run.run_id=row-1"]
    assert eval_stage.config_identity_hash(
        original_path, overrides
    ) != eval_stage.config_identity_hash(changed_path, overrides)
    # Mutation kill: returning a constant digest (or omitting config_payload
    # from _config_identity_digest) makes this assertion fail.


def test_frozen_grid_bindings_are_carried_into_the_manifest(tmp_path: Path) -> None:
    grid = canary.load_grid(GRID_42_PATH)
    expected = {
        "local_energy_mean": "eval/mcmc_energy",
        "local_energy_stderr": "eval/mcmc_energy",
        "local_energy_variance": "eval/mcmc_energy",
    }

    assert grid["metric_namespaces"] == expected
    manifest = _completion_manifest_42(tmp_path)
    assert manifest["gate_spec"] == {collect.METRIC_NAMESPACE_SPEC_KEY: expected}
    # Bindings are provenance, not tolerances; no value thresholds are declared.
    assert manifest["gate_spec_declared"] is False


def test_frozen_grid_rejects_missing_metric_namespace_block(tmp_path: Path) -> None:
    payload = yaml.safe_load(GRID_42_PATH.read_text(encoding="utf-8"))
    del payload["metric_namespaces"]
    path = tmp_path / "eval_42_without_metric_namespaces.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(canary.CanaryError, match="missing=.*metric_namespaces"):
        canary.load_grid(path)


def test_declared_binding_resolves_collision_without_cli_flags(tmp_path: Path) -> None:
    manifest = _completion_manifest_42(tmp_path)
    thresholds, bindings = collect.split_gate_spec(manifest["gate_spec"])
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "namespace": "eval/mcmc_energy",
                    "metrics": {
                        "local_energy_mean": -1.0,
                        "local_energy_stderr": 0.01,
                        "local_energy_variance": 0.25,
                    },
                },
                {
                    "namespace": "eval/factor_response_re_equilibrated",
                    # Differing means are load-bearing: equal values would not
                    # be considered ambiguous by read_metrics_jsonl.
                    "metrics": {"local_energy_mean": -2.0},
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    namespaced, flat, ambiguous = collect.read_metrics_jsonl(metrics_path)
    namespaces = collect.logged_namespaces(namespaced)
    reasons: list[str] = []

    bound = collect.resolve_metric_bindings(
        bindings,
        namespaced=namespaced,
        flat=flat,
        ambiguous=ambiguous,
        namespaces=namespaces,
        reasons=reasons,
    )

    assert thresholds == {}
    assert bound["local_energy_mean"] == -1.0
    assert bound["local_energy_stderr"] == 0.01
    assert bound["local_energy_variance"] == 0.25
    assert reasons == []

    _, reason = collect.resolve_metric_request(
        "local_energy_mean",
        namespaced=namespaced,
        flat=flat,
        ambiguous=ambiguous,
        namespaces=namespaces,
    )
    assert reason is not None


def test_frozen_grid_rejects_a_real_gpu_constraint_drift() -> None:
    payload = yaml.safe_load(GRID_42_PATH.read_text(encoding="utf-8"))
    payload["rows"][0]["resources"]["constraint"] = None
    with pytest.raises(canary.CanaryError, match="constraint disagrees"):
        canary._load_grid_v2(payload)


def test_production_row_passes_plan_and_launch_with_stratum_constraint(
    tmp_path: Path,
) -> None:
    """Planner and launcher must share the production stratum validator."""

    grid = canary.load_grid(GRID_42_PATH)
    row = dict(grid["rows"][0])
    planned_resources = canary._resolve_resources(row["resources"], "production row")
    assert planned_resources["partition"] == "kozinsky_gpu,seas_gpu"
    assert planned_resources["stratum"] == "a100"
    assert planned_resources["constraint"] == "a100"

    directives = launch.sbatch_directives(
        row,
        job_name="he-v1-eval42-production",
        log_dir=tmp_path,
        account=None,
        dependency=None,
    )
    assert "#SBATCH --partition=kozinsky_gpu,seas_gpu" in directives
    assert "#SBATCH --constraint=a100" in directives


@pytest.mark.parametrize(
    "fault, match",
    [
        ("partial", "source map ids mismatch"),
        ("model", "content mismatch"),
        ("manifest", "content mismatch"),
        ("complete", "content mismatch"),
        ("updates", "completed_updates mismatch"),
        ("training_sha", "training source SHA mismatch"),
    ],
)
def test_immutable_source_validation_fails_closed(
    tmp_path: Path, fault: str, match: str
) -> None:
    manifest, source_map_path, source_map = _case(tmp_path)
    source_id = "actual-step-025000"
    checkpoint_dir = Path(source_map["sources"][source_id]["checkpoint_dir"])
    if fault == "partial":
        del source_map["sources"][source_id]
    elif fault in {"model", "manifest", "complete"}:
        filename = {"model": "model.pt", "manifest": "manifest.json", "complete": "COMPLETE"}[
            fault
        ]
        with (checkpoint_dir / filename).open("ab") as handle:
            handle.write(b"mutation")
    elif fault == "updates":
        source_map["sources"][source_id]["completed_updates"] -= 1
    else:
        source_map["sources"][source_id]["training_source_sha"] = "c" * 40
    source_map_path.write_text(yaml.safe_dump(source_map, sort_keys=True), encoding="utf-8")
    # This test targets source semantics rather than the map-file hash guard.
    manifest["source_map_sha256"] = canary.file_sha256(source_map_path)
    with pytest.raises(canary.CanaryError, match=match):
        canary.reconcile_manifest_sources(manifest, source_map_path)


def test_selected_row_launch_validates_both_sources_and_emits_exact_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source_map_path, source_map = _case(tmp_path)
    monkeypatch.setattr(launch, "_require_repo_identity", lambda *_: None)
    selected = launch.select_rows(
        manifest, kinds=[], row_ids=["eval-canary-step000050000"]
    )
    summary = launch.launch(
        manifest=manifest,
        results_root=tmp_path / "results",
        repo_root=tmp_path / "clean-checkout",
        rows=selected,
        launch_attempt_id="20260825T130000",
        uv_bin="/home/test/.local/bin/uv",
        uv_extras=["cu128"],
        uv_project_environment="/work/test/env",
        uv_cache_root="/work/test/cache",
        account="kozinsky_lab",
        submit=False,
        checkpoint_source_map=source_map_path,
    )
    assert [record["row_id"] for record in summary["rows"]] == [
        "eval-canary-step000050000"
    ]
    record = summary["rows"][0]
    script = Path(record["script_path"]).read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu_test" in script
    assert "#SBATCH --constraint" not in script
    assert f'test "$(git rev-parse HEAD)" = {EVALUATION_SHA}' in script
    assert 'git status --porcelain --untracked-files=no' in script
    assert record["command"][-2:] == ["--checkpoint-source-map", str(source_map_path)]

    # Selection cannot make an invalid unselected source disappear.
    Path(source_map["sources"]["actual-step-025000"]["checkpoint_dir"], "model.pt").write_bytes(
        b"changed"
    )
    with pytest.raises(launch.LaunchError, match="content mismatch"):
        launch.launch(
            manifest=manifest,
            results_root=tmp_path / "other-results",
            repo_root=tmp_path / "clean-checkout",
            rows=selected,
            launch_attempt_id="other",
            uv_bin="/home/test/.local/bin/uv",
            uv_extras=["cu128"],
            uv_project_environment="/work/test/env",
            uv_cache_root="/work/test/cache",
            account=None,
            submit=False,
            checkpoint_source_map=source_map_path,
        )


def test_runtime_transform_keeps_the_real_graph_and_only_reduces_scale(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _case(tmp_path)
    row = manifest["rows"][0]
    cfg = collect.eval_stage.driver.build_config(
        STUDY_DIR.parents[2] / row["config"], row["overrides"]
    )
    configured = eval_stage.configure_canary_evaluation(cfg, row)

    assert len(configured.evaluator.tasks) == 1
    assert configured.evaluator.tasks[0].name == "mcmc_energy"
    assert configured.evaluation_sampler.n_walkers == row["n_walkers"]
    assert configured.evaluation_sampler.burn_in == row["burn_in"]
    assert configured.evaluation_sampler.n_steps == row["stride"]
    assert configured.evaluator.tasks[0].generator.n_draws == row["n_draws"]
    assert configured.evaluator.tasks[0].generator.discard_draws == row["discard_draws"]
    targets = [callback._target_ for callback in configured.callbacks]
    assert targets[-2:] == ["tpen.callback.ArtifactIndex", "tpen.callback.FailureLog"]


def test_runtime_transform_selects_declared_multi_task_row_and_factor_task_is_inert(
    tmp_path: Path,
) -> None:
    """An unselected factor declaration cannot enter the evaluator graph."""

    manifest, _, _ = _case(tmp_path)
    row = dict(manifest["rows"][0])
    cfg = collect.eval_stage.driver.build_config(
        STUDY_DIR.parents[2] / row["config"], row["overrides"]
    )
    configured = eval_stage.configure_canary_evaluation(cfg, row)
    assert [task.name for task in configured.evaluator.tasks] == ["mcmc_energy"]
    assert all(task.name != "factor_response" for task in configured.evaluator.tasks)

    row["task_names"] = ["mcmc_energy", "factor_response"]
    cfg = collect.eval_stage.driver.build_config(
        STUDY_DIR.parents[2] / row["config"], row["overrides"]
    )
    configured = eval_stage.configure_canary_evaluation(cfg, row)
    assert [task.name for task in configured.evaluator.tasks] == [
        "mcmc_energy", "factor_response"
    ]


def test_factor_common_collector_contract_uses_real_generator_emission() -> None:
    """The common factor oracle must agree with the producer's real snapshot."""

    class FixedSampler:
        def collect_samples(self, model, *, reset=False, device=None, emitter=None):
            del model, reset, device, emitter
            walkers = Walkers(positions=torch.zeros(3, 1, 3, dtype=torch.float64))
            return walkers, object()

    row = {"n_walkers": 3, "n_draws": 8, "record_capacity": 24}
    generator = MCMCGenerator(sampler=FixedSampler(), max_samples=row["n_walkers"])
    generated = generator.generate(
        model=None,
        context=EvaluationContext(
            namespace="eval/factor_response",
            artifact_level="metrics_only",
            task_failure_policy="fail_fast",
            device=torch.device("cpu"),
            dtype=torch.float64,
            seed=None,
            run_dir=Path("/tmp"),
            task_output_dir=Path("/tmp"),
            metadata={},
        ),
    )
    emitted_configurations = generated.batch.batch_size
    assert emitted_configurations == row["n_walkers"]

    reasons: list[str] = []
    collect._reconcile_canary_task_content(
        "factor_response",
        {
            "factor_response_common_configuration": {
                "metadata": {
                    "comparison_kind": "common_configuration",
                    "rows": emitted_configurations * 7,
                    "arm_count": 7,
                    "configuration_count": emitted_configurations,
                    "model_state_restored": True,
                }
            }
        },
        row=row,
        reasons=reasons,
    )
    assert reasons == []

    truncated = MCMCGenerator(sampler=FixedSampler(), max_samples=2).generate(
        model=None,
        context=EvaluationContext(
            namespace="eval/factor_response",
            artifact_level="metrics_only",
            task_failure_policy="fail_fast",
            device=torch.device("cpu"),
            dtype=torch.float64,
            seed=None,
            run_dir=Path("/tmp"),
            task_output_dir=Path("/tmp"),
            metadata={},
        ),
    )
    assert truncated.batch.batch_size != row["n_walkers"]


def test_factor_task_caps_are_row_overrides_and_common_cap_covers_all_arms() -> None:
    config = yaml.safe_load((STUDY_DIR / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    common = config["evaluation_tasks"]["factor_response"]
    assert int(common["summaries"][0]["max_records"]) >= 4096 * 7
    reequilibrated = config["evaluation_tasks"]["factor_response_re_equilibrated"]
    assert "max_samples" not in reequilibrated["generator"]["generator"]
    assert "max_samples" not in reequilibrated["summaries"][1]


def test_re_equilibrated_factor_task_applies_row_draws_to_trajectory_generator(
    tmp_path: Path,
) -> None:
    """The configured re-equilibrated producer must receive the row draw count."""

    _, source_map_path, _ = _case(tmp_path)
    grid = canary.load_grid(GRID_42_PATH)
    rows = canary.expand_rows(grid, canary.reconcile_grid_sources(
        grid, canary.load_source_map(source_map_path)
    ))
    row = next(row for row in rows if "factor_response_re_equilibrated" in row["task_names"])
    cfg = collect.eval_stage.driver.build_config(
        STUDY_DIR.parents[2] / row["config"], row["overrides"]
    )
    configured = eval_stage.configure_canary_evaluation(cfg, row)
    task = next(
        task for task in configured.evaluator.tasks
        if task.name == "factor_response_re_equilibrated"
    )
    assert task.generator.generator._target_ == (
        "tpen.evaluation.generators.TrajectoryMCMCGenerator"
    )
    assert task.generator.generator.n_draws == row["n_draws"]
    assert task.generator.generator.discard_draws == row["discard_draws"]
    assert task.generator.generator.max_samples == row["record_capacity"]
    resolved_writer = next(
        summary for summary in task.summaries
        if summary._target_ == "tpen.evaluation.summaries.SampledRecordWriter"
    )
    assert resolved_writer.max_samples == row["record_capacity"]
    assert resolved_writer.include_term_energies is True



def test_planner_receipt_reports_the_actual_v2_plan_row_count(tmp_path: Path) -> None:
    """The planner's user-facing receipt must describe the plan it wrote."""

    _, source_map_path, _ = _case(tmp_path)
    results_root = tmp_path / "results"
    output = StringIO()
    with redirect_stdout(output):
        assert canary.main(
            [
                "--grid-config", str(GRID_42_PATH),
                "--checkpoint-source-map", str(source_map_path),
                "--results-root", str(results_root),
                "--attempt-id", "20260825T120000",
                "--evaluation-git-sha", EVALUATION_SHA,
            ]
        ) == 0

    manifest = plan.read_manifest(results_root, "20260825T120000")
    assert manifest["n_rows"] == 42
    assert f"wrote {manifest['n_rows']} rows" in output.getvalue()


def test_runtime_transform_rejects_unknown_or_duplicate_declared_tasks(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _case(tmp_path)
    row = dict(manifest["rows"][0])
    for task_names, match in [
        (["does_not_exist"], "unknown evaluation task"),
        (["mcmc_energy", "mcmc_energy"], "must be unique"),
    ]:
        row["task_names"] = task_names
        cfg = collect.eval_stage.driver.build_config(
            STUDY_DIR.parents[2] / row["config"], row["overrides"]
        )
        with pytest.raises(eval_stage.driver.DriverError, match=match):
            eval_stage.configure_canary_evaluation(cfg, row)


def _write_canary_outputs(
    tmp_path: Path,
    manifest: dict[str, Any],
    source_map_path: Path,
    *,
    row_index: int = 0,
) -> tuple[dict[str, Any], Any, Path, Path, dict[str, Any], dict[str, Any]]:
    row = manifest["rows"][row_index]
    sources = canary.reconcile_manifest_sources(manifest, source_map_path)
    source = canary.source_for_row(row, sources)
    result_dir = collect.layout.row_dir(
        tmp_path, row["stage"], row["row_id"], manifest["attempt_id"]
    )
    run_dir = result_dir / row["row_id"]
    task_dir = run_dir / "mcmc_energy"
    task_dir.mkdir(parents=True)
    (run_dir / "diagnostics").mkdir()

    collect.layout.write_json(result_dir / eval_stage.CHECKPOINT_BINDING_RECEIPT, source.receipt())
    row_record = {
        "row": row,
        "plan_attempt_id": manifest["attempt_id"],
        "launch_attempt_id": "launch-1",
        "job_id": "123",
    }
    metadata = {
        "run_id": row["row_id"],
        "git_commit": EVALUATION_SHA,
        "dirty_worktree": False,
    }
    collect.layout.write_json(result_dir / "row.json", row_record)
    collect.layout.write_json(run_dir / "metadata.json", metadata)

    replay = {
        "source_git_sha": source.training_source_sha,
        "source_tpen_version": row["checkpoint_source"]["source_tpen_version"],
        "checkpoint_schema_version": 2,
        "checkpoint_kind": "tpen.checkpoint",
        "checkpoint_model_sha256": source.model_sha256,
    }
    resolved_config = {
        "evaluator": {
            "tasks": [
                {
                    "name": "mcmc_energy",
                    "generator": {
                        "n_draws": row["n_draws"],
                        "discard_draws": row["discard_draws"],
                        "chunk_size": row["chunk_size"],
                    },
                    "summaries": [
                        {
                            "_target_": "tpen.evaluation.summaries.SampledRecordWriter",
                            "max_samples": row["record_capacity"],
                        }
                    ],
                }
            ]
        },
        "evaluation_sampler": {
            "n_walkers": row["n_walkers"],
            "burn_in": row["burn_in"],
            "n_steps": row["stride"],
        },
        "load": {"replay_semantics": replay},
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
    )

    csv_path = task_dir / "sampled_eval_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_index", "draw_index", "walker_index"]
        )
        writer.writeheader()
        for draw in range(row["n_draws"]):
            for walker in range(row["n_walkers"]):
                writer.writerow(
                    {
                        "sample_index": draw * row["n_walkers"] + walker,
                        "draw_index": draw,
                        "walker_index": walker,
                    }
                )
    csv_sha = plan.file_sha256(csv_path)
    record_metadata = {
        "schema": "trajectory_records/v1",
        "row_semantics": "complete_draw_walker_grid",
        "observable": "local_energy",
        "draw_count": row["n_draws"],
        "walker_count": row["n_walkers"],
        "draw_stride": row["stride"],
        "burn_in_draws": row["discard_draws"],
        "row_count": row["record_capacity"],
        "csv_sha256": csv_sha,
        "byte_count": csv_path.stat().st_size,
    }
    collect.layout.write_json(task_dir / "sampled_eval_table.metadata.json", record_metadata)

    config_sha = eval_stage.config_identity_hash(
        STUDY_DIR.parents[2] / row["config"],
        row["overrides"],
        identity_values={
            key: row.get(key)
            for key in (
                "canary_protocol",
                "checkpoint_source",
                "task_names",
                "n_walkers",
                "n_draws",
                "burn_in",
                "discard_draws",
                "stride",
                "chunk_size",
                "record_capacity",
            )
        },
    )
    identity = {
        "stage": row["stage"],
        "run_id": row["row_id"],
        "attempt_id": manifest["attempt_id"],
        "checkpoint_sha256": source.model_sha256,
        "config_sha256": config_sha,
        "observable": "local_energy",
        "evaluator_id": eval_stage.EVALUATOR_ID,
    }
    trajectory = {
        **identity,
        "status": "unresolved",
        "shape": {
            "walker_count": row["n_walkers"],
            "draw_count": row["n_draws"],
            "total_draws": row["record_capacity"],
            "draw_stride": row["stride"],
            "burn_in_draws": row["discard_draws"],
        },
    }
    trajectory_path = task_dir / "trajectory_statistics.jsonl"
    trajectory_path.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
    proposal_scale = 0.5
    discarded_draws = [
        _sampler_diagnostic_draw(
            row,
            collection_index=index,
            region_index=index,
            proposal_scale=proposal_scale,
        )
        for index in range(row["discard_draws"])
    ]
    retained_draws = [
        _sampler_diagnostic_draw(
            row,
            collection_index=row["discard_draws"] + index,
            region_index=index,
            proposal_scale=proposal_scale,
        )
        for index in range(row["n_draws"])
    ]
    sampler_diagnostics = {
        "schema": "sampler_trajectory_diagnostics/v1",
        "n_walkers": row["n_walkers"],
        "draw_stride": row["stride"],
        "sampler_burn_in": row["burn_in"],
        "proposal_scale": proposal_scale,
        "intermediate_sampler_steps_observed": False,
        "intermediate_sampler_steps_unobserved_reason": (
            "collector sees only stride-spaced states"
        ),
        "sampler_internal_burn_in_states_observed": False,
        "discarded_draw_acceptance_rate_series": [
            draw["acceptance_rate"] for draw in discarded_draws
        ],
        "retained_draw_acceptance_rate_series": [
            draw["acceptance_rate"] for draw in retained_draws
        ],
        "discarded_draws": discarded_draws,
        "retained_draws": retained_draws,
        "metrics": {
            "trajectory_retained_draw_count": row["n_draws"],
            "trajectory_discarded_draw_count": row["discard_draws"],
            "trajectory_n_walkers": row["n_walkers"],
            "trajectory_draw_stride": row["stride"],
            "trajectory_sampler_burn_in": row["burn_in"],
            "trajectory_proposal_scale": proposal_scale,
            "trajectory_retained_value_count": row["record_capacity"],
            "trajectory_discarded_value_count": row["discard_draws"]
            * row["n_walkers"],
            "trajectory_retained_transition_count": row["record_capacity"]
            * row["stride"],
            "trajectory_discarded_transition_count": row["discard_draws"]
            * row["n_walkers"]
            * row["stride"],
            "trajectory_retained_draw_acceptance_rate_mean": 0.5,
            "trajectory_intermediate_sampler_steps_observed": False,
        },
    }
    sampler_diagnostics_path = task_dir / "sampler_trajectory_diagnostics.json"
    collect.layout.write_json(sampler_diagnostics_path, sampler_diagnostics)
    index = {
        "tasks": [
            {
                "name": "mcmc_energy",
                "namespace": "eval/mcmc_energy",
                "output_dir": str(task_dir.resolve()),
                "status": "success",
                "artifacts": [
                    {
                        "name": "sampled_eval_table",
                        "kind": "csv",
                        "path": str(csv_path.resolve()),
                        "metadata": {
                            "rows": row["record_capacity"],
                            "n_total": row["record_capacity"],
                            "draw_count": row["n_draws"],
                            "walker_count": row["n_walkers"],
                            "truncated": False,
                            "selection": "complete_draw_walker_grid",
                            "content_id": csv_sha,
                            "bytes": csv_path.stat().st_size,
                        },
                    },
                    {
                        "name": "local_energy_trajectory_statistics",
                        "kind": "trajectory_statistics_sidecar",
                        "path": str(trajectory_path.resolve()),
                        "metadata": {**identity, "status": "unresolved"},
                    },
                    {
                        "name": "sampler_trajectory_diagnostics",
                        "kind": "sampler_trajectory_diagnostics",
                        "path": str(sampler_diagnostics_path.resolve()),
                        "metadata": {
                            "schema": "sampler_trajectory_diagnostics/v1",
                            "retained_draw_count": row["n_draws"],
                            "discarded_draw_count": row["discard_draws"],
                            "draw_stride": row["stride"],
                            "intermediate_sampler_steps_observed": False,
                        },
                    },
                ],
            }
        ]
    }
    collect.layout.write_json(run_dir / "diagnostics" / "index.json", index)
    return row, source, result_dir, run_dir, row_record, metadata


def _sampler_diagnostic_draw(
    row: dict[str, Any],
    *,
    collection_index: int,
    region_index: int,
    proposal_scale: float,
) -> dict[str, Any]:
    return {
        "collection_index": collection_index,
        "region_index": region_index,
        "acceptance_rate": 0.5,
        "n_walkers": row["n_walkers"],
        "burn_in": row["burn_in"],
        "draw_stride": row["stride"],
        "transition_count": row["n_walkers"] * row["stride"],
        "proposal_scale": proposal_scale,
        "seed": row["seed"],
        "minimum_electron_nucleus_radius": 0.25,
    }


def _reconcile_written_outputs(
    manifest: dict[str, Any],
    written: tuple[dict[str, Any], Any, Path, Path, dict[str, Any], dict[str, Any]],
) -> list[str]:
    row, source, result_dir, run_dir, row_record, metadata = written
    return collect.reconcile_canary_row(
        row,
        source=source,
        result_dir=result_dir,
        run_dir=run_dir,
        plan_attempt_id=manifest["attempt_id"],
        manifest=manifest,
        row_record=row_record,
        metadata=metadata,
    )


def test_collection_accepts_legacy_config_identity_and_records_match(
    tmp_path: Path,
) -> None:
    """Pre-canonical trajectory receipts remain re-collectable and explicit."""

    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, row_record, metadata = written
    identity_values = {
        key: row.get(key)
        for key in (
            "canary_protocol",
            "checkpoint_source",
            "task_names",
            "n_walkers",
            "n_draws",
            "burn_in",
            "discard_draws",
            "stride",
            "chunk_size",
            "record_capacity",
        )
    }
    canonical = eval_stage.config_identity_hash(
        STUDY_DIR.parents[2] / row["config"],
        row["overrides"],
        identity_values=identity_values,
    )
    legacy = eval_stage.legacy_config_identity_hash(
        STUDY_DIR.parents[2] / row["config"],
        row["overrides"],
        identity_values=identity_values,
    )
    assert canonical != legacy

    trajectory_path = run_dir / "mcmc_energy" / "trajectory_statistics.jsonl"
    receipt = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert receipt["config_sha256"] == canonical
    receipt["config_sha256"] = legacy
    trajectory_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    index = _artifact_index(run_dir)
    artifact = index["tasks"][0]["artifacts"][1]
    artifact["metadata"]["config_sha256"] = legacy
    _write_artifact_index(run_dir, index)

    diagnostics: dict[str, Any] = {}
    reasons = collect.reconcile_canary_row(
        row,
        source=written[1],
        result_dir=written[2],
        run_dir=run_dir,
        plan_attempt_id=manifest["attempt_id"],
        manifest=manifest,
        row_record=row_record,
        metadata=metadata,
        identity_diagnostics=diagnostics,
    )

    assert reasons == []
    assert diagnostics == {"config_identity_match": "legacy"}
    # Mutation kill: removing the legacy candidate from _match_config_identity
    # makes this receipt fail and leaves the recorded match unset.


def test_collection_rejects_unknown_config_identity(tmp_path: Path) -> None:
    """A hash matching neither accepted identity must fail closed."""

    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    trajectory_path = run_dir / "mcmc_energy" / "trajectory_statistics.jsonl"
    receipt = json.loads(trajectory_path.read_text(encoding="utf-8"))
    receipt["config_sha256"] = "0" * 64
    trajectory_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    index = _artifact_index(run_dir)
    index["tasks"][0]["artifacts"][1]["metadata"]["config_sha256"] = "0" * 64
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("matches neither canonical nor legacy hash" in reason for reason in reasons)


def test_collection_reports_missing_legacy_revision_per_row(tmp_path: Path) -> None:
    """An unavailable historical revision must not abort the whole collector."""

    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, source, result_dir, run_dir, row_record, metadata = written
    trajectory_path = run_dir / "mcmc_energy" / "trajectory_statistics.jsonl"
    receipt = json.loads(trajectory_path.read_text(encoding="utf-8"))
    receipt["config_sha256"] = "0" * 64
    trajectory_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    index = _artifact_index(run_dir)
    index["tasks"][0]["artifacts"][1]["metadata"]["config_sha256"] = "0" * 64
    _write_artifact_index(run_dir, index)

    reasons = collect.reconcile_canary_row(
        row,
        source=source,
        result_dir=result_dir,
        run_dir=run_dir,
        plan_attempt_id=manifest["attempt_id"],
        manifest=manifest,
        row_record=row_record,
        metadata=metadata,
        source_git_sha="f" * 40,
    )

    assert any("legacy config identity unavailable" in reason for reason in reasons)
    assert any("matches neither canonical nor legacy hash" in reason for reason in reasons)


def _completion_manifest_42(tmp_path: Path) -> dict[str, Any]:
    """Build the shipped 42-row plan with the fixture's immutable sources."""

    _, source_map_path, _ = _case(tmp_path)
    grid = canary.load_grid(GRID_42_PATH)
    sources = canary.load_source_map(source_map_path)
    rows = canary.expand_rows(grid, canary.reconcile_grid_sources(grid, sources))
    return canary.build_manifest(
        grid=grid,
        rows=rows,
        attempt_id="20260825T120000",
        results_root=tmp_path / "results",
        grid_path=GRID_42_PATH,
        source_map_path=source_map_path,
        evaluation_git_sha=EVALUATION_SHA,
        created_at="2026-08-25T12:00:00-04:00",
    )


def test_collection_completeness_uses_the_declared_42_row_shape(tmp_path: Path) -> None:
    manifest = _completion_manifest_42(tmp_path)
    rows = [
        {"identity": {"row_id": row["row_id"]}, "status": "pass"}
        for row in manifest["rows"]
    ]

    complete, reasons = collect._canary_completion(manifest, rows)

    assert manifest["n_rows"] == 42
    assert complete is True
    assert reasons == []


def test_collection_completeness_rejects_a_plan_short_by_one_row(tmp_path: Path) -> None:
    manifest = _completion_manifest_42(tmp_path)
    rows = [
        {"identity": {"row_id": row["row_id"]}, "status": "pass"}
        for row in manifest["rows"][:-1]
    ]

    complete, reasons = collect._canary_completion(manifest, rows)

    assert complete is False
    assert any("collected 41 rows but plan declares 42" in reason for reason in reasons)
    assert any(manifest["rows"][-1]["row_id"] in reason for reason in reasons)
    with pytest.raises(collect.CollectError, match=manifest["rows"][-1]["row_id"]):
        collect.require_complete_canary_collection(
            {
                "canary_complete": complete,
                "canary_completion_reasons": reasons,
                "rows": rows,
            }
        )


def _artifact_index(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "diagnostics" / "index.json").read_text(encoding="utf-8"))


def _write_artifact_index(run_dir: Path, index: dict[str, Any]) -> None:
    collect.layout.write_json(run_dir / "diagnostics" / "index.json", index)


def _index_reasons(
    tmp_path: Path, task_name: str, artifacts: list[dict[str, Any]]
) -> list[str]:
    task_dir = tmp_path / task_name
    reasons: list[str] = []
    collect._reconcile_canary_index(
        {
            "tasks": [
                {
                    "name": task_name,
                    "namespace": f"eval/{task_name}",
                    "output_dir": str(task_dir.resolve()),
                    "status": "success",
                    "artifacts": artifacts,
                }
            ]
        },
        run_dir=tmp_path,
        task_names=[task_name],
        row={},
        reasons=reasons,
    )
    return reasons


def test_metrics_only_tasks_accept_empty_artifact_lists(tmp_path: Path) -> None:
    metrics_only_tasks = (
        "full_model_antisymmetry",
        "spatial_exchange_symmetry",
        "trace_equivariance",
    )

    for task_name in metrics_only_tasks:
        assert _index_reasons(tmp_path, task_name, []) == []


def test_non_metrics_only_tasks_still_require_an_artifact(tmp_path: Path) -> None:
    reasons = _index_reasons(tmp_path, "mcmc_energy", [])

    assert any("unexpected empty artifacts" in reason for reason in reasons)


def test_artifact_index_still_rejects_duplicate_names(tmp_path: Path) -> None:
    reasons = _index_reasons(
        tmp_path,
        "trace_equivariance",
        [{"name": "same"}, {"name": "same"}],
    )

    assert any("duplicate" in reason for reason in reasons)


def test_artifact_index_still_rejects_missing_artifact_names(tmp_path: Path) -> None:
    reasons = _index_reasons(tmp_path, "trace_equivariance", [{}])

    assert any("invalid" in reason for reason in reasons)


def _artifact(index: dict[str, Any], name: str) -> dict[str, Any]:
    return next(
        artifact
        for artifact in index["tasks"][0]["artifacts"]
        if artifact["name"] == name
    )


def _refresh_record_identity(run_dir: Path) -> None:
    task_dir = run_dir / "mcmc_energy"
    csv_path = task_dir / "sampled_eval_table.csv"
    csv_sha = plan.file_sha256(csv_path)
    record_metadata_path = task_dir / "sampled_eval_table.metadata.json"
    record_metadata = json.loads(record_metadata_path.read_text(encoding="utf-8"))
    record_metadata["csv_sha256"] = csv_sha
    record_metadata["byte_count"] = csv_path.stat().st_size
    collect.layout.write_json(record_metadata_path, record_metadata)
    index = _artifact_index(run_dir)
    metadata = _artifact(index, "sampled_eval_table")["metadata"]
    metadata["content_id"] = csv_sha
    metadata["bytes"] = csv_path.stat().st_size
    _write_artifact_index(run_dir, index)


def _refresh_trajectory_identity(
    manifest: dict[str, Any], row: dict[str, Any], source: Any, run_dir: Path
) -> None:
    """Refresh fixture-owned trajectory identity after mutating the planned row."""

    config_sha = eval_stage.config_identity_hash(
        STUDY_DIR.parents[2] / row["config"],
        row["overrides"],
        identity_values={
            key: row.get(key)
            for key in (
                "canary_protocol", "checkpoint_source", "task_names", "n_walkers",
                "n_draws", "burn_in", "discard_draws", "stride", "chunk_size",
                "record_capacity",
            )
        },
    )
    trajectory_path = run_dir / "mcmc_energy" / "trajectory_statistics.jsonl"
    receipt = json.loads(trajectory_path.read_text(encoding="utf-8"))
    receipt.update({"stage": row["stage"], "run_id": row["row_id"],
                   "attempt_id": manifest["attempt_id"],
                   "checkpoint_sha256": source.model_sha256,
                   "config_sha256": config_sha,
                   "observable": "local_energy", "evaluator_id": eval_stage.EVALUATOR_ID})
    trajectory_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    index = _artifact_index(run_dir)
    metadata = _artifact(index, "local_energy_trajectory_statistics")["metadata"]
    metadata.update({"stage": row["stage"], "run_id": row["row_id"],
                     "attempt_id": manifest["attempt_id"],
                     "checkpoint_sha256": source.model_sha256,
                     "config_sha256": config_sha,
                     "observable": "local_energy", "evaluator_id": eval_stage.EVALUATOR_ID,
                     "status": receipt["status"]})
    _write_artifact_index(run_dir, index)


def test_collection_accepts_exact_three_artifact_contract(tmp_path: Path) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(
        tmp_path, manifest, source_map_path
    )
    run_dir = written[3]
    names = [artifact["name"] for artifact in _artifact_index(run_dir)["tasks"][0]["artifacts"]]

    assert set(names) == set(collect.CANARY_ARTIFACT_FILENAMES)
    assert len(names) == 3
    assert _reconcile_written_outputs(manifest, written) == []


def test_collection_reconciles_artifacts_for_each_declared_task(tmp_path: Path) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, source, _, run_dir, _, _ = written
    row["task_names"].append("factor_response")
    _refresh_trajectory_identity(manifest, row, source, run_dir)
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluator"]["tasks"].append({"name": "factor_response"})
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    factor_dir = run_dir / "factor_response"
    factor_dir.mkdir()
    factor_path = factor_dir / "factor_response.csv"
    factor_path.write_text("arm,energy\nbaseline,0\n", encoding="utf-8")
    index = _artifact_index(run_dir)
    index["tasks"].append(
        {
            "name": "factor_response",
            "namespace": "eval/factor_response",
            "output_dir": str(factor_dir.resolve()),
            "status": "success",
            "artifacts": [
                {
                    "name": "factor_response_common_configuration",
                    "kind": "csv",
                    "path": str(factor_path.resolve()),
                    "metadata": {
                        "comparison_kind": "common_configuration",
                        "rows": row["n_walkers"] * 7,
                        "arm_count": 7,
                        "configuration_count": row["n_walkers"],
                        "model_state_restored": True,
                    },
                }
            ],
        }
    )
    _write_artifact_index(run_dir, index)

    assert _reconcile_written_outputs(manifest, written) == []


def test_collection_rejects_declared_task_namespace_mismatch(tmp_path: Path) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    index = _artifact_index(run_dir)
    index["tasks"][0]["namespace"] = "eval/spatial_exchange_symmetry"
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("namespace disagrees" in reason for reason in reasons)


def test_collection_rejects_factor_content_shape_mismatch(tmp_path: Path) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    row["task_names"].append("factor_response")
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluator"]["tasks"].append({"name": "factor_response"})
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    factor_dir = run_dir / "factor_response"
    factor_dir.mkdir()
    factor_path = factor_dir / "factor_response.csv"
    factor_path.write_text("arm,energy\nbaseline,0\n", encoding="utf-8")
    index = _artifact_index(run_dir)
    index["tasks"].append({
        "name": "factor_response", "namespace": "eval/factor_response",
        "output_dir": str(factor_dir.resolve()), "status": "success",
        "artifacts": [{
            "name": "factor_response_common_configuration", "kind": "csv",
            "path": str(factor_path.resolve()),
            "metadata": {"comparison_kind": "common_configuration", "rows": 1,
                         "arm_count": 1, "configuration_count": row["n_walkers"],
                         "model_state_restored": True},
        }],
    })
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("factor_response artifact metadata disagrees" in reason for reason in reasons)


@pytest.mark.parametrize("field, base", [("rows", 56), ("arm_count", 7), ("configuration_count", 8)])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_factor_common_numeric_oracles_are_boundary_sensitive(
    field: str, base: int, delta: int
) -> None:
    row = {"n_walkers": 8, "record_capacity": 64}
    metadata = {
        "comparison_kind": "common_configuration",
        "rows": 56,
        "arm_count": 7,
        "configuration_count": 8,
        "model_state_restored": True,
    }
    metadata[field] = base + delta
    reasons: list[str] = []

    collect._reconcile_canary_task_content(
        "factor_response",
        {"factor_response_common_configuration": {"metadata": metadata}},
        row=row,
        reasons=reasons,
    )

    if delta == 0:
        assert reasons == []
    else:
        assert any("factor_response artifact metadata disagrees" in reason for reason in reasons)


def test_collection_rejects_reequilibrated_factor_row_count_mismatch(
    tmp_path: Path,
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, source, _, run_dir, _, _ = written
    row["task_names"].append("factor_response_re_equilibrated")
    _refresh_trajectory_identity(manifest, row, source, run_dir)
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluator"]["tasks"].append({"name": "factor_response_re_equilibrated"})
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    factor_dir = run_dir / "factor_response_re_equilibrated"
    factor_dir.mkdir()
    factor_path = factor_dir / "sampled_eval_table.csv"
    factor_path.write_text("sample_index\n0\n", encoding="utf-8")
    index = _artifact_index(run_dir)
    index["tasks"].append({
        "name": "factor_response_re_equilibrated",
        "namespace": "eval/factor_response_re_equilibrated",
        "output_dir": str(factor_dir.resolve()), "status": "success",
        "artifacts": [{
            "name": "sampled_eval_table", "kind": "csv",
            "path": str(factor_path.resolve()),
            "metadata": {"rows": 1},
        }],
    })
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any(
        "re-equilibrated factor sampled table metadata disagrees" in reason
        for reason in reasons
    )


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_reequilibrated_row_count_oracle_is_boundary_sensitive(delta: int) -> None:
    row = {"record_capacity": 64}
    reasons: list[str] = []
    collect._reconcile_canary_task_content(
        "factor_response_re_equilibrated",
        {"sampled_eval_table": {"metadata": {"rows": row["record_capacity"] + delta}}},
        row=row,
        reasons=reasons,
    )

    if delta == 0:
        assert reasons == []
    else:
        assert any(
            "re-equilibrated factor sampled table metadata disagrees" in reason
            for reason in reasons
        )


def test_collection_rejects_a_declared_task_missing_from_artifact_index(
    tmp_path: Path,
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    row["task_names"].append("factor_response")
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluator"]["tasks"].append({"name": "factor_response"})
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("artifact index task list disagrees" in reason for reason in reasons)


def test_collection_rejects_declared_task_artifact_outside_task_directory(
    tmp_path: Path,
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    row["task_names"].append("factor_response")
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluator"]["tasks"].append({"name": "factor_response"})
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    factor_dir = run_dir / "factor_response"
    factor_dir.mkdir()
    factor_path = factor_dir / "factor_response.csv"
    factor_path.write_text("arm,energy\nbaseline,0\n", encoding="utf-8")
    index = _artifact_index(run_dir)
    index["tasks"].append(
        {
            "name": "factor_response",
            "namespace": "eval/factor_response",
            "output_dir": str(factor_dir.resolve()),
            "status": "success",
            "artifacts": [{"name": "factor_response", "kind": "csv", "path": str((tmp_path / "escaped.csv").resolve())}],
        }
    )
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("outside its indexed task directory" in reason for reason in reasons)


@pytest.mark.parametrize("artifact_name", sorted(collect.CANARY_ARTIFACT_FILENAMES))
@pytest.mark.parametrize("mutation", ["missing", "renamed"])
def test_collection_rejects_each_missing_or_renamed_artifact(
    tmp_path: Path, artifact_name: str, mutation: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    index = _artifact_index(run_dir)
    artifacts = index["tasks"][0]["artifacts"]
    if mutation == "missing":
        index["tasks"][0]["artifacts"] = [
            artifact for artifact in artifacts if artifact["name"] != artifact_name
        ]
    else:
        _artifact(index, artifact_name)["name"] = f"renamed_{artifact_name}"
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("mcmc_energy artifacts mismatch" in reason for reason in reasons)


def test_collection_rejects_unknown_extra_artifact(tmp_path: Path) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    index = _artifact_index(run_dir)
    unknown = dict(index["tasks"][0]["artifacts"][0])
    unknown["name"] = "unknown_canary_artifact"
    index["tasks"][0]["artifacts"].append(unknown)
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("mcmc_energy artifacts mismatch" in reason for reason in reasons)


@pytest.mark.parametrize("artifact_name", sorted(collect.CANARY_ARTIFACT_FILENAMES))
def test_collection_rejects_artifact_provenance_mismatch(
    tmp_path: Path, artifact_name: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    index = _artifact_index(run_dir)
    _artifact(index, artifact_name)["kind"] = "wrong_kind"
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("kind disagrees" in reason for reason in reasons)


@pytest.mark.parametrize("channel", ["sampler_burn_in", "trajectory_discard"])
def test_collection_keeps_sampler_burn_in_and_trajectory_discard_distinct(
    tmp_path: Path, channel: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if channel == "sampler_burn_in":
        config["evaluation_sampler"]["burn_in"] = row["discard_draws"]
    else:
        config["evaluator"]["tasks"][0]["generator"]["discard_draws"] = row["burn_in"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("resolved" in reason and "scale disagrees" in reason for reason in reasons)


@pytest.mark.parametrize("field", ["draw_count", "walker_count", "row_count", "burn_in_draws"])
@pytest.mark.parametrize("delta", [-1, 1])
def test_collection_rejects_each_record_shape_mismatch(
    tmp_path: Path, field: str, delta: int
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    path = run_dir / "mcmc_energy" / "sampled_eval_table.metadata.json"
    record_metadata = json.loads(path.read_text(encoding="utf-8"))
    record_metadata[field] = record_metadata[field] + delta
    collect.layout.write_json(path, record_metadata)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("record metadata disagrees" in reason for reason in reasons)


@pytest.mark.parametrize("delta", [-1, 1])
def test_collection_rejects_boundary_mcmc_writer_capacity_mismatch(
    tmp_path: Path, delta: int
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluator"]["tasks"][0]["summaries"][0]["max_samples"] = (
        row["record_capacity"] + delta
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("writer capacity disagrees" in reason for reason in reasons)


@pytest.mark.parametrize("coordinate", ["sample_index", "draw_index", "walker_index"])
def test_collection_rejects_each_record_coordinate_mismatch(
    tmp_path: Path, coordinate: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    csv_path = run_dir / "mcmc_energy" / "sampled_eval_table.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    records[0][coordinate] = "99"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _refresh_record_identity(run_dir)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("complete ordered draw/walker grid" in reason for reason in reasons)


@pytest.mark.parametrize(
    "field",
    ["walker_count", "draw_count", "total_draws", "draw_stride", "burn_in_draws"],
)
@pytest.mark.parametrize("delta", [-1, 1])
def test_collection_rejects_each_trajectory_statistics_shape_mismatch(
    tmp_path: Path, field: str, delta: int
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    _, _, _, run_dir, _, _ = written
    path = run_dir / "mcmc_energy" / "trajectory_statistics.jsonl"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["shape"][field] = receipt["shape"][field] + delta
    path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("trajectory-statistics shape disagrees" in reason for reason in reasons)


@pytest.mark.parametrize(
    "mutation",
    ["retained_draw", "discarded_draw", "walker_count", "sampler_burn_in"],
)
def test_collection_rejects_sampler_diagnostics_shape_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    path = run_dir / "mcmc_energy" / "sampler_trajectory_diagnostics.json"
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "retained_draw":
        diagnostics["retained_draws"].pop()
    elif mutation == "discarded_draw":
        diagnostics["discarded_draws"].append(dict(diagnostics["retained_draws"][0]))
    elif mutation == "walker_count":
        diagnostics["n_walkers"] += 1
    else:
        diagnostics["sampler_burn_in"] = diagnostics["sampler_burn_in"] + 1
    collect.layout.write_json(path, diagnostics)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("diagnostics disagree" in reason for reason in reasons)


@pytest.mark.parametrize("delta", [-1, 1])
def test_collection_rejects_boundary_sampler_diagnostics_walker_count(
    tmp_path: Path, delta: int
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    path = run_dir / "mcmc_energy" / "sampler_trajectory_diagnostics.json"
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    diagnostics["n_walkers"] = row["n_walkers"] + delta
    collect.layout.write_json(path, diagnostics)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("diagnostics disagree" in reason for reason in reasons)


@pytest.mark.parametrize("mutation", ["trajectory_identity", "missing_record"])
def test_collection_reconciliation_fails_closed_on_partial_or_identity_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]

    trajectory_path = run_dir / "mcmc_energy" / "trajectory_statistics.jsonl"
    if mutation == "trajectory_identity":
        receipt = json.loads(trajectory_path.read_text(encoding="utf-8"))
        receipt["checkpoint_sha256"] = "f" * 64
        trajectory_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    else:
        csv_path = run_dir / "mcmc_energy" / "sampled_eval_table.csv"
        csv_path.rename(csv_path.with_suffix(".preserved-partial.csv"))

    reasons = _reconcile_written_outputs(manifest, written)

    assert reasons


def test_collection_refuses_partial_source_map_and_partial_row_set(tmp_path: Path) -> None:
    manifest, source_map_path, source_map = _case(tmp_path)
    del source_map["sources"]["actual-step-025000"]
    source_map_path.write_text(yaml.safe_dump(source_map, sort_keys=True), encoding="utf-8")
    manifest["source_map_sha256"] = canary.file_sha256(source_map_path)
    results_root = tmp_path / "results"
    plan.write_plan(manifest, results_root=results_root)
    with pytest.raises(collect.CollectError, match="incomplete"):
        collect.collect(
            results_root=results_root,
            plan_attempt_id=manifest["attempt_id"],
            collect_attempt_id="collect-1",
            gate_spec={},
            gate_spec_source="plan_manifest",
            checkpoint_source_map=source_map_path,
        )

    with pytest.raises(collect.CollectError, match="failing rows"):
        collect.require_complete_canary_collection(
            {
                "canary_complete": False,
                "rows": [
                    {"identity": {"row_id": "eval-canary-step000025000"}, "status": "pass"}
                ],
            }
        )
