"""Focused round-1 mutation tests for the H-v2 collector review.

The existing ``test_collect`` module is imported so its real producer fixture
and crafted artifact builders remain the single source of setup.

Each refusal test alters artifacts or the declared budget, not source code.
Its docstring names the source mutant it is intended to kill after a fix.
These are unexecuted review proposals; run them on Cannon from a clean checkout.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).parent
_TEST_COLLECT_SPEC = importlib.util.spec_from_file_location(
    "h_v2_test_collect_review_r1_base", HERE / "test_collect.py"
)
assert _TEST_COLLECT_SPEC is not None and _TEST_COLLECT_SPEC.loader is not None
base = importlib.util.module_from_spec(_TEST_COLLECT_SPEC)
sys.modules[_TEST_COLLECT_SPEC.name] = base
_TEST_COLLECT_SPEC.loader.exec_module(base)
collect = base.collect
real_run_artifacts = base.real_run_artifacts


@pytest.fixture(scope="module")
def real_two_update_training(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Produce a genuine two-update train-only run with one metrics row/update."""

    protocol_spec = importlib.util.spec_from_file_location(
        "h_v2_collect_review_r1_protocol", HERE / "protocol.py"
    )
    assert protocol_spec is not None and protocol_spec.loader is not None
    protocol = importlib.util.module_from_spec(protocol_spec)
    sys.modules[protocol_spec.name] = protocol
    protocol_spec.loader.exec_module(protocol)
    from omegaconf import OmegaConf
    from tpen.run import run_from_config

    root = tmp_path_factory.mktemp("real_two_update_training")
    train = protocol.training_config()
    train["run"]["root"] = str(root)
    train["run"]["run_id"] = "collector-two-update"
    train["model"]["readout"]["eps"] = 1.0e-12
    train["trainer"]["max_steps"] = 2
    train["trainer"]["log_every_n_steps"] = 1
    train["sampler"].update(n_walkers=8, burn_in=2, n_steps=1)
    train["callbacks"][-1]["schedule"]["every_n"] = 1
    assert run_from_config(OmegaConf.create(train), command="h-v2 collector two-step train", raise_exceptions=True) == 0
    return root / "tpen_h_v2" / "atomistic" / "collector-two-update"


def test_real_collector_clean_control(real_run_artifacts: dict) -> None:
    """Control: the unaltered real evaluation artifacts are admitted."""

    receipt = collect.collect_records(**base._real_collect_kwargs(real_run_artifacts))
    assert receipt["status"] == "complete"
    collect.validate_training_run(real_run_artifacts["training_dir"], expected_updates=1)


def _sampler_bound_fixture(tmp_path: Path) -> tuple[dict, collect.RecordSpec]:
    """Extend the existing crafted fixture with valid lifecycle and sampler facts."""

    paths, spec = base._full_fixture(tmp_path)
    resolved = yaml.safe_load(paths["resolved_config_path"].read_text(encoding="utf-8"))
    resolved["runtime"] = {"dtype": "float64", "seed": 910003}
    resolved["sampler"] = {
        "n_walkers": spec.walkers, "burn_in": 1000, "n_steps": 20,
        "proposal_scale": 0.5, "seed": 910003,
    }
    resolved["evaluator"] = {"tasks": [{"name": "trajectory", "generator": {
        "sampler": dict(resolved["sampler"]), "n_draws": spec.draws,
        "discard_draws": 0, "seed": 910003,
    }}]}
    paths["resolved_config_path"].write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    metadata = json.loads(paths["metadata_path"].read_text(encoding="utf-8"))
    # The trajectory records collector-discarded draws, not sampler burn-in.
    metadata.update({"burn_in_draws": 0, "draw_stride": 20})
    paths["metadata_path"].write_text(json.dumps(metadata), encoding="utf-8")
    run_metadata = json.loads(paths["run_metadata_path"].read_text(encoding="utf-8"))
    run_metadata["dirty_worktree"] = False
    paths["run_metadata_path"].write_text(json.dumps(run_metadata), encoding="utf-8")
    (tmp_path / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    spec = collect.RecordSpec(
        source_sha=spec.source_sha,
        checkpoint_model_sha256=spec.checkpoint_model_sha256,
        resolved_config_sha256=collect.expected_resolved_config_identity_from_path(
            paths["resolved_config_path"]
        )["sha256"],
        walkers=spec.walkers,
        draws=spec.draws,
    )
    return paths, spec


def test_crafted_sampler_binding_clean_control(tmp_path: Path) -> None:
    """Control: bound crafted sampler config, sidecar, and trajectory metadata admit."""

    paths, spec = _sampler_bound_fixture(tmp_path)
    collect.collect_records(**base._real_collect_kwargs({**paths, "spec": spec}))


def test_real_sampler_sidecar_burn_in_mutant_is_refused(real_run_artifacts: dict, tmp_path: Path) -> None:
    """Kill a range-only burn-in guard using a consistently altered real sidecar."""

    altered = tmp_path / "sampler_trajectory_diagnostics.json"
    sidecar = json.loads(real_run_artifacts["sampler_diagnostics_path"].read_text(encoding="utf-8"))
    sidecar["sampler_burn_in"] += 1
    for draw in sidecar["retained_draws"]:
        draw["burn_in"] += 1
    altered.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="sampler|burn-in"):
        collect.collect_records(**base._real_collect_kwargs(
            real_run_artifacts, sampler_diagnostics_path=altered,
        ))


def test_real_sampler_sidecar_stride_mutant_is_refused(real_run_artifacts: dict, tmp_path: Path) -> None:
    """Kill a range-only stride guard using changed stride and transition counts."""

    altered = tmp_path / "sampler_trajectory_diagnostics.json"
    sidecar = json.loads(real_run_artifacts["sampler_diagnostics_path"].read_text(encoding="utf-8"))
    sidecar["draw_stride"] += 1
    for draw in sidecar["retained_draws"]:
        draw["draw_stride"] += 1
        draw["transition_count"] = draw["n_walkers"] * draw["draw_stride"]
    altered.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="sampler|stride|transition"):
        collect.collect_records(**base._real_collect_kwargs(
            real_run_artifacts, sampler_diagnostics_path=altered,
        ))


def test_crafted_sampler_sidecar_mutant_is_refused(tmp_path: Path) -> None:
    """Kill an omitted config/sidecar burn-in binding with a crafted mismatch."""

    paths, spec = _sampler_bound_fixture(tmp_path)
    sidecar = json.loads(paths["sampler_diagnostics_path"].read_text(encoding="utf-8"))
    sidecar["sampler_burn_in"] += 1
    for draw in sidecar["retained_draws"]:
        draw["burn_in"] += 1
    paths["sampler_diagnostics_path"].write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="sampler|burn-in"):
        collect.collect_records(**base._real_collect_kwargs({**paths, "spec": spec}))


def test_real_sampler_duplicate_draw_index_is_refused(real_run_artifacts: dict, tmp_path: Path) -> None:
    """Kill a length-only draw guard by duplicating one real retained-draw index."""

    altered = tmp_path / "sampler_trajectory_diagnostics.json"
    sidecar = json.loads(real_run_artifacts["sampler_diagnostics_path"].read_text(encoding="utf-8"))
    sidecar["retained_draws"][1]["collection_index"] = sidecar["retained_draws"][0]["collection_index"]
    altered.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="sampler|draw|index"):
        collect.collect_records(**base._real_collect_kwargs(real_run_artifacts, sampler_diagnostics_path=altered))


def _alter_real_train_metrics(training_dir: Path, destination: Path, *, drop: bool) -> None:
    shutil.copytree(training_dir, destination)
    metrics_path = destination / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    train_rows = [record for record in records if record.get("namespace") == "train"]
    assert len(train_rows) == 1 and train_rows[0]["step"] == 0
    assert train_rows[0]["metrics"]["optimizer_step"] is True
    if drop:
        records.remove(train_rows[0])
    else:
        train_rows[0]["metrics"]["optimizer_step"] = False
    metrics_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_real_training_metrics_false_optimizer_step_is_refused(
    real_run_artifacts: dict, tmp_path: Path
) -> None:
    """Kill an omitted applied-update log check by flipping a real row's flag."""

    altered = tmp_path / "training"
    _alter_real_train_metrics(real_run_artifacts["training_dir"], altered, drop=False)
    with pytest.raises(collect.RecordIntegrityError, match="metrics|optimizer_step|update"):
        collect.validate_training_run(altered, expected_updates=1)


def test_real_training_metrics_dropped_train_row_is_refused(
    real_run_artifacts: dict, tmp_path: Path
) -> None:
    """Kill a checkpoint-only completeness check by deleting the real train row."""

    altered = tmp_path / "training"
    _alter_real_train_metrics(real_run_artifacts["training_dir"], altered, drop=True)
    with pytest.raises(collect.RecordIntegrityError, match="metrics|missing|update"):
        collect.validate_training_run(altered, expected_updates=1)


def test_two_update_training_has_exact_terminal_count(real_two_update_training: Path) -> None:
    """Control: genuine two-update training validates with expected_updates=2."""

    for step in (1, 2):
        manifest_path = real_two_update_training / "checkpoints" / f"step_{step:06d}" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["completed_updates"] == manifest["next_iteration"] == step
    collect.validate_training_run(real_two_update_training, expected_updates=2)


def test_two_update_training_overrun_is_refused(real_two_update_training: Path) -> None:
    """Kill target-exists admission using a genuine two-update run and budget one."""

    with pytest.raises(collect.RecordIntegrityError, match="declared|updates|terminal|metrics"):
        collect.validate_training_run(real_two_update_training, expected_updates=1)


@pytest.mark.parametrize("alteration", [None, "drop", "skip"], ids=["clean-control", "missing-row", "skipped-row"])
def test_crafted_training_log_evidence(tmp_path: Path, alteration: str | None) -> None:
    """Kill checkpoint-only log admission with missing/skipped crafted rows; include a control."""

    training_dir = base._training_run_fixture(tmp_path, expected_updates=1)
    (training_dir / "resolved_config.yaml").write_text(
        "trainer:\n  max_steps: 1\n  log_every_n_steps: 1\n", encoding="utf-8"
    )
    row = {"namespace": "train", "step": 0, "metrics": {"optimizer_step": alteration != "skip"}}
    (training_dir / "metrics.jsonl").write_text(
        "" if alteration == "drop" else json.dumps(row) + "\n", encoding="utf-8"
    )
    if alteration is None:
        collect.validate_training_run(training_dir, expected_updates=1)
    else:
        with pytest.raises(collect.RecordIntegrityError, match="metrics|log|update"):
            collect.validate_training_run(training_dir, expected_updates=1)


@pytest.mark.parametrize(
    ("metadata_mutation", "status_value", "message"),
    [(True, "completed", "dirty"), (False, "failed", "complete")],
    ids=["dirty-worktree", "failed-status"],
)
def test_crafted_collect_lifecycle_mutants_are_refused(
    tmp_path: Path, metadata_mutation: bool, status_value: str, message: str
) -> None:
    """Kill an omitted evaluation lifecycle call using crafted dirty/failed artifacts."""

    paths, spec = _sampler_bound_fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata = json.loads(paths["run_metadata_path"].read_text(encoding="utf-8"))
    if metadata_mutation:
        metadata["dirty_worktree"] = True
    altered_metadata = run_dir / "metadata.json"
    altered_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": status_value}), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match=message):
        collect.collect_records(**base._real_collect_kwargs({**paths, "spec": spec}, run_metadata_path=altered_metadata))


@pytest.mark.parametrize(
    ("dirty", "status", "reason"),
    [(True, "completed", "dirty"), (False, "failed", "complete")],
    ids=["dirty-worktree", "failed-status"],
)
def test_real_collect_lifecycle_is_enforced(
    real_run_artifacts: dict, tmp_path: Path, dirty: bool, status: str, reason: str
) -> None:
    """Kill an omitted evaluation lifecycle call by changing genuine run artifacts."""

    metadata = json.loads(real_run_artifacts["run_metadata_path"].read_text(encoding="utf-8"))
    status_record = json.loads((real_run_artifacts["eval_run_dir"] / "status.json").read_text(encoding="utf-8"))
    assert metadata["dirty_worktree"] is False and status_record["status"] == "completed"
    metadata["dirty_worktree"] = dirty
    status_record["status"] = status
    altered_metadata = tmp_path / "metadata.json"
    altered_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "status.json").write_text(json.dumps(status_record), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match=reason):
        collect.collect_records(**base._real_collect_kwargs(real_run_artifacts, run_metadata_path=altered_metadata))


def test_crafted_restore_missing_loaded_rng_reaches_collect_records(tmp_path: Path) -> None:
    """Kill a missing-means-false restore guard with a crafted absent loaded_rng."""

    paths, spec = _sampler_bound_fixture(tmp_path)
    record = json.loads(paths["occurrences_path"].read_text(encoding="utf-8"))
    del record["fields"]["report"]["loaded_rng"]
    paths["occurrences_path"].write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="loaded_rng|restore"):
        collect.collect_records(**base._real_collect_kwargs({**paths, "spec": spec}))


def test_real_restore_missing_loaded_rng_reaches_collect_records(
    real_run_artifacts: dict, tmp_path: Path
) -> None:
    """Kill a missing-means-false restore guard with a genuine absent loaded_rng."""

    altered = tmp_path / "occurrences.jsonl"
    rewritten = []
    for line in real_run_artifacts["occurrences_path"].read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") == collect.CHECKPOINT_RESTORED_EVENT:
            del record["fields"]["report"]["loaded_rng"]
        rewritten.append(json.dumps(record))
    altered.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="loaded_rng|restore"):
        collect.collect_records(**base._real_collect_kwargs(real_run_artifacts, occurrences_path=altered))
