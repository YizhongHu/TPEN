"""Integrity controls for the complete H-v2 collector, on the real schema.

``collect.py`` consumes TPEN's real trajectory CSV/metadata envelope, a run's
``resolved_config.yaml``/``metadata.json``, and a restored checkpoint
directory (see that module's docstring). Most refusals here are exercised on
synthetic fixtures shaped exactly like those real artifacts, which is enough
to pin each integrity check; ``test_collector_admits_the_real_smoke_evaluation_output``
is the one test that drives a real tiny train+eval through
``tpen.run.run_from_config`` and feeds the collector its actual output,
establishing that the shapes used elsewhere in this file are not a drifted
fiction (finding 3 of the round-1 review).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("h_v2_collect", HERE / "collect.py")
assert spec is not None and spec.loader is not None
collect = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = collect
spec.loader.exec_module(collect)


def _checkpoint_fixture(tmp_path: Path, *, model_bytes: bytes = b"model-bytes") -> tuple[Path, collect.RecordSpec]:
    """Write a minimal checkpoint directory and return it with a matching spec fragment."""

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").write_bytes(model_bytes)
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return checkpoint_dir, collect.file_sha256(checkpoint_dir / "model.pt")


def _run_metadata_fixture(tmp_path: Path, *, git_commit: str = "deadbeef") -> tuple[Path, Path, str]:
    """Write a minimal run metadata.json and resolved_config.yaml pair."""

    resolved_config_path = tmp_path / "resolved_config.yaml"
    resolved_config_path.write_text("load:\n  mode: model_only\n", encoding="utf-8")
    run_metadata_path = tmp_path / "metadata.json"
    run_metadata_path.write_text(json.dumps({"git_commit": git_commit, "status": "completed"}), encoding="utf-8")
    return run_metadata_path, resolved_config_path, collect.expected_resolved_config_identity_from_path(resolved_config_path)["sha256"]


#: One step's worth of restore identity shared by the fixtures below.
_CHECKPOINT_STEP = 1000

_TERM_NAMES = ["kinetic", "electron_nucleus"]


def _trajectory_fixture(tmp_path: Path, *, walkers: int = 2, draws: int = 3) -> tuple[Path, Path]:
    """Write a complete real-schema trajectory CSV and its metadata.json envelope."""

    csv_path = tmp_path / "sampled_eval_table.csv"
    fieldnames = [
        "sample_index", "draw_index", "walker_index", "local_energy",
        "term/kinetic", "term/electron_nucleus", "logabs", "sign", "finite",
        "position/electron_0/axis_0", "position/electron_0/axis_1", "position/electron_0/axis_2",
    ]
    rows = []
    for draw in range(draws):
        for walker in range(walkers):
            rows.append({
                "sample_index": draw * walkers + walker,
                "draw_index": draw,
                "walker_index": walker,
                "local_energy": -0.5,
                "term/kinetic": 0.5,
                "term/electron_nucleus": -1.0,
                "logabs": -1.0,
                "sign": 1.0,
                "finite": True,
                "position/electron_0/axis_0": 1.0,
                "position/electron_0/axis_1": 0.0,
                "position/electron_0/axis_2": 0.0,
            })
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    # Every row is identical, so the whole-trajectory variance is exactly 0.
    metadata_path = tmp_path / "sampled_eval_table.metadata.json"
    metadata_path.write_text(json.dumps({
        "schema": collect.TRAJECTORY_RECORD_SCHEMA,
        "row_semantics": collect.ROW_SEMANTICS,
        "walker_count": walkers,
        "draw_count": draws,
        "row_count": walkers * draws,
        "nonfinite_count": 0,
        "term_names": _TERM_NAMES,
        "variance": 0.0,
        "csv_sha256": collect.file_sha256(csv_path),
    }), encoding="utf-8")
    return csv_path, metadata_path


def _occurrences_fixture(tmp_path: Path, *, checkpoint_dir: Path, step: int = _CHECKPOINT_STEP) -> Path:
    """Write an occurrences.jsonl containing exactly one fresh model-only restore event."""

    occurrences_path = tmp_path / "occurrences.jsonl"
    record = {
        "count": 1,
        "event": collect.CHECKPOINT_RESTORED_EVENT,
        "run_id": "h-v2-eval-stream0",
        "time": "2026-11-10T00:00:00-05:00",
        "fields": {
            "report": {
                "mode": "model_only",
                "checkpoint_dir": str(checkpoint_dir),
                "schema_version": 2,
                "next_iteration": step,
                "completed_updates": step,
                "loaded_model": True,
                "loaded_optimizer": False,
                "loaded_trainer": False,
                "loaded_sampler": False,
                "loaded_rng": False,
            },
        },
    }
    occurrences_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return occurrences_path


def _sampler_diagnostics_fixture(tmp_path: Path, *, walkers: int = 2, draws: int = 3) -> Path:
    """Write a complete real-schema sampler draw receipt sidecar."""

    sidecar_path = tmp_path / "sampler_trajectory_diagnostics.json"
    draw = {
        "collection_index": 0,
        "region_index": 0,
        "acceptance_rate": 0.5,
        "n_walkers": walkers,
        "burn_in": 1000,
        "draw_stride": 20,
        "transition_count": walkers * 20,
        "proposal_scale": 0.5,
        "seed": 910003,
        "minimum_electron_nucleus_radius": None,
    }
    sidecar_path.write_text(json.dumps({
        "schema": collect.SAMPLER_TRAJECTORY_DIAGNOSTICS_SCHEMA,
        "n_walkers": walkers,
        "draw_stride": 20,
        "sampler_burn_in": 1000,
        "proposal_scale": 0.5,
        "discarded_draws": [],
        "retained_draws": [dict(draw, collection_index=index, region_index=index) for index in range(draws)],
    }), encoding="utf-8")
    return sidecar_path


def _full_fixture(tmp_path: Path) -> tuple[dict, collect.RecordSpec]:
    """Assemble one complete, self-consistent fixture: paths and a matching spec."""

    checkpoint_dir, checkpoint_sha = _checkpoint_fixture(tmp_path)
    run_metadata_path, resolved_config_path, config_sha = _run_metadata_fixture(tmp_path)
    csv_path, metadata_path = _trajectory_fixture(tmp_path)
    occurrences_path = _occurrences_fixture(tmp_path, checkpoint_dir=checkpoint_dir)
    sampler_diagnostics_path = _sampler_diagnostics_fixture(tmp_path)
    spec = collect.RecordSpec(
        source_sha="deadbeef",
        checkpoint_model_sha256=checkpoint_sha,
        resolved_config_sha256=config_sha,
        walkers=2,
        draws=3,
    )
    paths = {
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "resolved_config_path": resolved_config_path,
        "run_metadata_path": run_metadata_path,
        "checkpoint_dir": checkpoint_dir,
        "occurrences_path": occurrences_path,
        "sampler_diagnostics_path": sampler_diagnostics_path,
        "checkpoint_step": _CHECKPOINT_STEP,
    }
    return paths, spec


def test_complete_records_are_collected(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    receipt = collect.collect_records(spec=spec, **paths)
    assert receipt["status"] == "complete"
    assert receipt["row_count"] == 6


def test_study_recompute_matches_recorded_identity_and_tracks_science_override(tmp_path: Path) -> None:
    run_metadata_path, resolved_config_path, _ = _run_metadata_fixture(tmp_path)
    resolved_config_path.write_text(
        "run:\n  root: /tmp/one\n  run_id: first\n"
        "science:\n  charge: 2\nload:\n  mode: model_only\n",
        encoding="utf-8",
    )
    intended = collect.expected_resolved_config_identity_from_path(resolved_config_path)
    spec = collect.RecordSpec("deadbeef", "a" * 64, intended["sha256"])
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    collect.validate_run_identity(run_metadata, resolved_config_path, spec)

    resolved_config_path.write_text(
        "run:\n  root: /tmp/two\n  run_id: second\n"
        "science:\n  charge: 3\nload:\n  mode: model_only\n",
        encoding="utf-8",
    )
    assert collect.expected_resolved_config_identity_from_path(resolved_config_path)["sha256"] != intended["sha256"]
    with pytest.raises(collect.RecordIntegrityError, match="altered"):
        collect.validate_run_identity(run_metadata, resolved_config_path, spec)


def test_missing_draw_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    with paths["csv_path"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.pop()
    with pytest.raises(collect.RecordIntegrityError, match="missing draw"):
        collect.validate_rows(rows, spec)


def test_duplicate_walker_record_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    with paths["csv_path"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[1] = dict(rows[0])
    with pytest.raises(collect.RecordIntegrityError, match="duplicate"):
        collect.validate_rows(rows, spec)


def test_nonfinite_record_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    with paths["csv_path"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["finite"] = "False"
    with pytest.raises(collect.RecordIntegrityError, match="nonfinite"):
        collect.validate_rows(rows, spec)


@pytest.mark.parametrize("field", ["git_commit"])
def test_altered_source_identity_refuses(tmp_path: Path, field: str) -> None:
    paths, spec = _full_fixture(tmp_path)
    run_metadata = json.loads(paths["run_metadata_path"].read_text(encoding="utf-8"))
    run_metadata[field] = "other"
    with pytest.raises(collect.RecordIntegrityError, match="altered"):
        collect.validate_run_identity(run_metadata, paths["resolved_config_path"], spec)


def test_altered_resolved_config_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    paths["resolved_config_path"].write_text("load:\n  mode: model_only\n  extra: true\n", encoding="utf-8")
    run_metadata = json.loads(paths["run_metadata_path"].read_text(encoding="utf-8"))
    with pytest.raises(collect.RecordIntegrityError, match="altered"):
        collect.validate_run_identity(run_metadata, paths["resolved_config_path"], spec)


def test_altered_checkpoint_content_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    (paths["checkpoint_dir"] / "model.pt").write_bytes(b"different-bytes")
    with pytest.raises(collect.RecordIntegrityError, match="altered"):
        collect.validate_checkpoint_identity(paths["checkpoint_dir"], spec)


def test_missing_checkpoint_commit_marker_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    (paths["checkpoint_dir"] / "COMPLETE").unlink()
    with pytest.raises(collect.RecordIntegrityError, match="missing checkpoint commit marker"):
        collect.validate_checkpoint_identity(paths["checkpoint_dir"], spec)


def test_training_state_restore_is_refused(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    paths["resolved_config_path"].write_text("load:\n  mode: train_resume\n", encoding="utf-8")
    resolved_config = {"load": {"mode": "train_resume"}}
    with pytest.raises(collect.RecordIntegrityError, match="model-only"):
        collect.validate_restore_mode(resolved_config)


def test_absent_restore_mode_is_refused() -> None:
    """An absent ``restore_mode`` is a refusal, not a silent model-only default."""

    with pytest.raises(collect.RecordIntegrityError, match="explicit restore_mode"):
        collect.validate_restore_mode({})
    with pytest.raises(collect.RecordIntegrityError, match="explicit restore_mode"):
        collect.validate_restore_mode({"load": {}})


def test_uncommitted_tail_refuses_when_envelope_is_absent(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    paths["metadata_path"].unlink()
    with pytest.raises(collect.RecordIntegrityError, match="uncommitted tail"):
        collect.collect_records(spec=spec, **paths)


def test_wrong_schema_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    manifest = json.loads(paths["metadata_path"].read_text(encoding="utf-8"))
    manifest["schema"] = "other/v1"
    with pytest.raises(collect.RecordIntegrityError, match="wrong record schema"):
        collect.validate_manifest(manifest, paths["csv_path"], spec)


def test_term_sum_mismatch_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    with paths["csv_path"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["term/kinetic"] = "999"
    with pytest.raises(collect.RecordIntegrityError, match="sum of term energies"):
        collect.validate_term_sum_and_sign(rows, _TERM_NAMES)


def test_nonunit_sign_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    with paths["csv_path"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["sign"] = "0.5"
    with pytest.raises(collect.RecordIntegrityError, match="sign is not"):
        collect.validate_term_sum_and_sign(rows, _TERM_NAMES)


def test_missing_term_names_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    with paths["csv_path"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with pytest.raises(collect.RecordIntegrityError, match="missing term names"):
        collect.validate_term_sum_and_sign(rows, None)


def test_variance_reconciliation_mismatch_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    with paths["csv_path"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with pytest.raises(collect.RecordIntegrityError, match="variance does not reconcile"):
        collect.validate_variance_reconciliation(rows, {"variance": 1.0})


def test_missing_restore_event_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    paths["occurrences_path"].unlink()
    with pytest.raises(collect.RecordIntegrityError, match="missing checkpoint restore event log"):
        collect.validate_restore_event(
            paths["occurrences_path"], checkpoint_dir=paths["checkpoint_dir"], step=_CHECKPOINT_STEP,
        )


def test_duplicate_restore_event_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    line = paths["occurrences_path"].read_text(encoding="utf-8")
    paths["occurrences_path"].write_text(line + line, encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="exactly one checkpoint restore event"):
        collect.validate_restore_event(
            paths["occurrences_path"], checkpoint_dir=paths["checkpoint_dir"], step=_CHECKPOINT_STEP,
        )


def test_restore_event_with_loaded_optimizer_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    record = json.loads(paths["occurrences_path"].read_text(encoding="utf-8").splitlines()[0])
    record["fields"]["report"]["loaded_optimizer"] = True
    paths["occurrences_path"].write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="restore event differs"):
        collect.validate_restore_event(
            paths["occurrences_path"], checkpoint_dir=paths["checkpoint_dir"], step=_CHECKPOINT_STEP,
        )


def test_restore_event_wrong_step_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    with pytest.raises(collect.RecordIntegrityError, match="restore event differs"):
        collect.validate_restore_event(
            paths["occurrences_path"], checkpoint_dir=paths["checkpoint_dir"], step=_CHECKPOINT_STEP + 1,
        )


def test_missing_sampler_draw_receipt_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    paths["sampler_diagnostics_path"].unlink()
    with pytest.raises(collect.RecordIntegrityError, match="missing sampler draw receipt"):
        collect.validate_sampler_draw_receipt(paths["sampler_diagnostics_path"], spec)


def test_sampler_draw_receipt_wrong_draw_count_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    sidecar = json.loads(paths["sampler_diagnostics_path"].read_text(encoding="utf-8"))
    sidecar["retained_draws"].pop()
    paths["sampler_diagnostics_path"].write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="draw count differs"):
        collect.validate_sampler_draw_receipt(paths["sampler_diagnostics_path"], spec)


def test_sampler_draw_receipt_bad_acceptance_refuses(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    sidecar = json.loads(paths["sampler_diagnostics_path"].read_text(encoding="utf-8"))
    sidecar["retained_draws"][0]["acceptance_rate"] = 1.5
    paths["sampler_diagnostics_path"].write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="acceptance rate is invalid"):
        collect.validate_sampler_draw_receipt(paths["sampler_diagnostics_path"], spec)


def _training_run_fixture(tmp_path: Path, *, expected_updates: int = 1000) -> Path:
    """Write a minimal clean, complete training run with one checkpoint."""

    training_dir = tmp_path / "training"
    training_dir.mkdir()
    (training_dir / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (training_dir / "metadata.json").write_text(json.dumps({"dirty_worktree": False}), encoding="utf-8")
    checkpoints_dir = training_dir / "checkpoints"
    step_dir = checkpoints_dir / f"step_{expected_updates:06d}"
    step_dir.mkdir(parents=True)
    (step_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    model_bytes = b"model-bytes"
    (step_dir / "model.pt").write_bytes(model_bytes)
    manifest = {
        "completed_updates": expected_updates,
        "next_iteration": expected_updates,
        "files": {"model": "model.pt"},
        "hashes": {"model_sha256": collect.file_sha256(step_dir / "model.pt")},
    }
    (step_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return training_dir


def test_training_run_is_validated_clean(tmp_path: Path) -> None:
    training_dir = _training_run_fixture(tmp_path)
    collect.validate_training_run(training_dir, expected_updates=1000)


def test_dirty_worktree_refuses(tmp_path: Path) -> None:
    training_dir = _training_run_fixture(tmp_path)
    metadata_path = training_dir / "metadata.json"
    metadata_path.write_text(json.dumps({"dirty_worktree": True}), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="dirty"):
        collect.validate_training_run(training_dir, expected_updates=1000)


def test_run_not_completed_refuses(tmp_path: Path) -> None:
    training_dir = _training_run_fixture(tmp_path)
    status_path = training_dir / "status.json"
    status_path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="did not complete"):
        collect.validate_training_run(training_dir, expected_updates=1000)


def test_skipped_update_refuses(tmp_path: Path) -> None:
    training_dir = _training_run_fixture(tmp_path)
    manifest_path = training_dir / "checkpoints" / "step_001000" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_updates"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="skipped or mislabeled"):
        collect.validate_training_run(training_dir, expected_updates=1000)


def test_short_of_declared_updates_refuses(tmp_path: Path) -> None:
    training_dir = _training_run_fixture(tmp_path, expected_updates=500)
    with pytest.raises(collect.RecordIntegrityError, match="declared number of updates"):
        collect.validate_training_run(training_dir, expected_updates=1000)


def test_checkpoint_file_hash_mismatch_refuses(tmp_path: Path) -> None:
    training_dir = _training_run_fixture(tmp_path)
    step_dir = training_dir / "checkpoints" / "step_001000"
    (step_dir / "model.pt").write_bytes(b"corrupted-bytes")
    with pytest.raises(collect.RecordIntegrityError, match="does not match its manifest"):
        collect.validate_training_run(training_dir, expected_updates=1000)


@pytest.fixture(scope="module")
def real_run_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Drive one real tiny train+eval through ``tpen.run.run_from_config``.

    Module-scoped so the several real-producer alteration tests below reuse
    one train+eval run instead of each paying its own training cost.
    """

    import importlib.util as _importlib_util

    protocol_spec = _importlib_util.spec_from_file_location("h_v2_collect_test_protocol", HERE / "protocol.py")
    assert protocol_spec is not None and protocol_spec.loader is not None
    protocol = _importlib_util.module_from_spec(protocol_spec)
    sys.modules[protocol_spec.name] = protocol
    protocol_spec.loader.exec_module(protocol)

    from omegaconf import OmegaConf
    from tpen.run import run_from_config

    tmp_path = tmp_path_factory.mktemp("real_run_artifacts")

    train = protocol.training_config()
    train["run"]["root"] = str(tmp_path)
    train["run"]["run_id"] = "collector-train"
    train["model"]["readout"]["eps"] = 1.0e-12
    train["trainer"]["max_steps"] = 1
    train["sampler"].update(n_walkers=8, burn_in=2, n_steps=1)
    train["callbacks"][-1]["schedule"]["every_n"] = 1
    assert run_from_config(OmegaConf.create(train), command="h-v2 collector train", raise_exceptions=True) == 0
    checkpoint_dir = tmp_path / "tpen_h_v2" / "atomistic" / "collector-train" / "checkpoints" / "step_000001"

    evaluation = protocol.evaluation_config(checkpoint_dir, root=tmp_path)
    evaluation["run"]["run_id"] = "collector-eval"
    evaluation["sampler"].update(n_walkers=8, burn_in=2, n_steps=1)
    generator = evaluation["evaluator"]["tasks"][0]["generator"]
    generator["_target_"] = "tpen.evaluation.generators.ReadoutTrajectoryMCMCGenerator"
    generator["n_draws"] = 3
    generator["readout_eps"] = 1.0e-12
    evaluation["model"]["readout"]["eps"] = 1.0e-12
    evaluation["evaluator"]["tasks"][0]["generator"]["chunk_size"] = 5
    evaluation["evaluator"]["tasks"][0]["summaries"][0]["max_samples"] = 24
    assert run_from_config(OmegaConf.create(evaluation), command="h-v2 collector eval", raise_exceptions=True) == 0

    training_dir = tmp_path / "tpen_h_v2" / "atomistic" / "collector-train"
    eval_run_dir = tmp_path / "tpen_h_v2" / "atomistic" / "collector-eval"
    trajectory_dir = eval_run_dir / "trajectory"
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    spec = collect.RecordSpec(
        source_sha=source_sha,
        checkpoint_model_sha256=collect.file_sha256(checkpoint_dir / "model.pt"),
        resolved_config_sha256=collect.expected_resolved_config_identity_from_path(
            eval_run_dir / "resolved_config.yaml"
        )["sha256"],
        walkers=8,
        draws=3,
    )
    return {
        "tmp_path": tmp_path,
        "training_dir": training_dir,
        "eval_run_dir": eval_run_dir,
        "trajectory_dir": trajectory_dir,
        "checkpoint_dir": checkpoint_dir,
        "csv_path": trajectory_dir / "sampled_eval_table.csv",
        "metadata_path": trajectory_dir / "sampled_eval_table.metadata.json",
        "resolved_config_path": eval_run_dir / "resolved_config.yaml",
        "run_metadata_path": eval_run_dir / "metadata.json",
        "occurrences_path": eval_run_dir / "occurrences.jsonl",
        "sampler_diagnostics_path": trajectory_dir / "sampler_trajectory_diagnostics.json",
        "checkpoint_step": 1,
        "spec": spec,
    }


def _real_collect_kwargs(real_run_artifacts: dict, **overrides) -> dict:
    kwargs = {
        "csv_path": real_run_artifacts["csv_path"],
        "metadata_path": real_run_artifacts["metadata_path"],
        "resolved_config_path": real_run_artifacts["resolved_config_path"],
        "run_metadata_path": real_run_artifacts["run_metadata_path"],
        "checkpoint_dir": real_run_artifacts["checkpoint_dir"],
        "occurrences_path": real_run_artifacts["occurrences_path"],
        "sampler_diagnostics_path": real_run_artifacts["sampler_diagnostics_path"],
        "checkpoint_step": real_run_artifacts["checkpoint_step"],
        "spec": real_run_artifacts["spec"],
    }
    kwargs.update(overrides)
    return kwargs


def test_collector_admits_the_real_smoke_evaluation_output(real_run_artifacts: dict, record_property) -> None:
    """Feed the collector a real tiny train+eval run's own artifacts directly.

    Establishes finding 3 of the round-1 review: the collector's field names
    (``draw_index``/``walker_index``/``position/electron_0/axis_*`` columns,
    the ``trajectory_records/v1`` envelope) are the ones TPEN actually writes,
    not a schema nothing produces.
    """

    trajectory_dir = real_run_artifacts["trajectory_dir"]
    csv_path = real_run_artifacts["csv_path"]
    raw_readout_path = trajectory_dir / "raw_readout.csv"
    raw_readout_metadata_path = trajectory_dir / "raw_readout.json"
    assert raw_readout_path.is_file()
    assert raw_readout_metadata_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        header = csv.DictReader(handle).fieldnames
    real_metadata = json.loads(real_run_artifacts["metadata_path"].read_text(encoding="utf-8"))
    record_property("real_csv_header", json.dumps(header))
    record_property("real_metadata_keys", json.dumps(sorted(real_metadata)))
    print(f"real_csv_header={header}")
    print(f"real_metadata_keys={sorted(real_metadata)}")

    receipt = collect.collect_records(**_real_collect_kwargs(real_run_artifacts))
    assert receipt["status"] == "complete"
    assert receipt["row_count"] == 24
    assert receipt["csv_sha256"] == real_metadata["csv_sha256"]
    raw_fields = csv.DictReader(raw_readout_path.open(newline="")).fieldnames
    assert raw_fields is not None
    assert not set(raw_fields).intersection(
        {"local_energy", "term/kinetic", "logabs", "sign"}
    )


def test_real_producer_output_passes_training_run_validation(real_run_artifacts: dict) -> None:
    """The real training run's own checkpoints and lifecycle artifacts validate clean."""

    collect.validate_training_run(real_run_artifacts["training_dir"], expected_updates=1)


def test_real_producer_restore_event_alteration_refuses(real_run_artifacts: dict, tmp_path: Path) -> None:
    """A restore event altered at file level to carry optimizer state is refused."""

    altered = tmp_path / "occurrences.jsonl"
    lines = real_run_artifacts["occurrences_path"].read_text(encoding="utf-8").splitlines()
    rewritten = []
    for line in lines:
        record = json.loads(line)
        if record.get("event") == collect.CHECKPOINT_RESTORED_EVENT:
            record["fields"]["report"]["loaded_optimizer"] = True
        rewritten.append(json.dumps(record))
    altered.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="restore event differs"):
        collect.collect_records(**_real_collect_kwargs(real_run_artifacts, occurrences_path=altered))


def test_real_producer_dirty_worktree_refuses(real_run_artifacts: dict, tmp_path: Path) -> None:
    """A run metadata.json altered at file level to report a dirty worktree is refused."""

    altered_dir = tmp_path / "altered_eval_run"
    altered_dir.mkdir()
    altered_metadata_path = altered_dir / "metadata.json"
    metadata = json.loads(real_run_artifacts["run_metadata_path"].read_text(encoding="utf-8"))
    metadata["dirty_worktree"] = True
    altered_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    status_path = real_run_artifacts["eval_run_dir"] / "status.json"
    with pytest.raises(collect.RecordIntegrityError, match="dirty"):
        collect.validate_run_is_clean_and_complete(status_path, altered_metadata_path)


def test_real_producer_checkpoint_hash_alteration_refuses(real_run_artifacts: dict, tmp_path: Path) -> None:
    """A checkpoint file altered at file level after the manifest was written is refused."""

    import shutil

    altered_checkpoint_dir = tmp_path / "altered_checkpoint"
    shutil.copytree(real_run_artifacts["checkpoint_dir"], altered_checkpoint_dir)
    (altered_checkpoint_dir / "model.pt").write_bytes(b"corrupted")
    manifest = json.loads((altered_checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(collect.RecordIntegrityError, match="does not match its manifest"):
        collect.validate_checkpoint_manifest_hashes(altered_checkpoint_dir, manifest)


def test_real_producer_training_log_skip_refuses(real_run_artifacts: dict, tmp_path: Path) -> None:
    """A checkpoint manifest altered at file level to understate completed updates is refused."""

    import shutil

    altered_training_dir = tmp_path / "altered_training"
    shutil.copytree(real_run_artifacts["training_dir"], altered_training_dir)
    manifest_path = altered_training_dir / "checkpoints" / "step_000001" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_updates"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="skipped or mislabeled"):
        collect.validate_training_run(altered_training_dir, expected_updates=1)


def test_real_producer_variance_mismatch_refuses(real_run_artifacts: dict, tmp_path: Path) -> None:
    """A trajectory metadata altered at file level to carry a wrong variance is refused."""

    altered_metadata_path = tmp_path / "sampled_eval_table.metadata.json"
    metadata = json.loads(real_run_artifacts["metadata_path"].read_text(encoding="utf-8"))
    metadata["variance"] = metadata["variance"] + 1.0
    altered_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="variance does not reconcile"):
        collect.collect_records(**_real_collect_kwargs(real_run_artifacts, metadata_path=altered_metadata_path))


def test_real_producer_term_sum_alteration_refuses(real_run_artifacts: dict, tmp_path: Path) -> None:
    """A trajectory row altered at file level to break its term sum is refused.

    The CSV's own hash is re-pinned in a copied metadata envelope so the
    alteration is caught by the term-sum check itself, not by the earlier
    envelope-hash check that already covers an unrelated tamper.
    """

    altered_csv_path = tmp_path / "sampled_eval_table.csv"
    with real_run_artifacts["csv_path"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["term/kinetic"] = str(float(rows[0]["term/kinetic"]) + 1.0)
    with altered_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    altered_metadata_path = tmp_path / "sampled_eval_table.metadata.json"
    metadata = json.loads(real_run_artifacts["metadata_path"].read_text(encoding="utf-8"))
    metadata["csv_sha256"] = collect.file_sha256(altered_csv_path)
    altered_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="sum of term energies"):
        collect.collect_records(**_real_collect_kwargs(
            real_run_artifacts, csv_path=altered_csv_path, metadata_path=altered_metadata_path,
        ))


def test_real_producer_sampler_receipt_alteration_refuses(real_run_artifacts: dict, tmp_path: Path) -> None:
    """A sampler draw receipt altered at file level to drop a retained draw is refused."""

    altered_sidecar_path = tmp_path / "sampler_trajectory_diagnostics.json"
    sidecar = json.loads(real_run_artifacts["sampler_diagnostics_path"].read_text(encoding="utf-8"))
    sidecar["retained_draws"].pop()
    altered_sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(collect.RecordIntegrityError, match="draw count differs"):
        collect.collect_records(**_real_collect_kwargs(
            real_run_artifacts, sampler_diagnostics_path=altered_sidecar_path,
        ))
