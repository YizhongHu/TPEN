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
    return run_metadata_path, resolved_config_path, collect.file_sha256(resolved_config_path)


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
    metadata_path = tmp_path / "sampled_eval_table.metadata.json"
    metadata_path.write_text(json.dumps({
        "schema": collect.TRAJECTORY_RECORD_SCHEMA,
        "row_semantics": collect.ROW_SEMANTICS,
        "walker_count": walkers,
        "draw_count": draws,
        "row_count": walkers * draws,
        "nonfinite_count": 0,
        "csv_sha256": collect.file_sha256(csv_path),
    }), encoding="utf-8")
    return csv_path, metadata_path


def _full_fixture(tmp_path: Path) -> tuple[dict, collect.RecordSpec]:
    """Assemble one complete, self-consistent fixture: paths and a matching spec."""

    checkpoint_dir, checkpoint_sha = _checkpoint_fixture(tmp_path)
    run_metadata_path, resolved_config_path, config_sha = _run_metadata_fixture(tmp_path)
    csv_path, metadata_path = _trajectory_fixture(tmp_path)
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
    }
    return paths, spec


def test_complete_records_are_collected(tmp_path: Path) -> None:
    paths, spec = _full_fixture(tmp_path)
    receipt = collect.collect_records(spec=spec, **paths)
    assert receipt["status"] == "complete"
    assert receipt["row_count"] == 6


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


def test_collector_admits_the_real_smoke_evaluation_output(tmp_path: Path, record_property) -> None:
    """Feed the collector a real tiny train+eval run's own artifacts directly.

    Establishes finding 3 of the round-1 review: the collector's field names
    (``draw_index``/``walker_index``/``position/electron_0/axis_*`` columns,
    the ``trajectory_records/v1`` envelope) are the ones TPEN actually writes,
    not a schema nothing produces.
    """

    import importlib.util as _importlib_util

    protocol_spec = _importlib_util.spec_from_file_location("h_v2_collect_test_protocol", HERE / "protocol.py")
    assert protocol_spec is not None and protocol_spec.loader is not None
    protocol = _importlib_util.module_from_spec(protocol_spec)
    sys.modules[protocol_spec.name] = protocol
    protocol_spec.loader.exec_module(protocol)

    from omegaconf import OmegaConf
    from tpen.run import run_from_config

    train = protocol.training_config()
    train["run"]["root"] = str(tmp_path)
    train["run"]["run_id"] = "collector-train"
    train["trainer"]["max_steps"] = 1
    train["sampler"].update(n_walkers=8, burn_in=2, n_steps=1)
    train["callbacks"][-1]["schedule"]["every_n"] = 1
    assert run_from_config(OmegaConf.create(train), command="h-v2 collector train", raise_exceptions=True) == 0
    checkpoint_dir = tmp_path / "tpen_h_v2" / "atomistic" / "collector-train" / "checkpoints" / "step_000001"

    evaluation = protocol.evaluation_config(checkpoint_dir, root=tmp_path)
    evaluation["run"]["run_id"] = "collector-eval"
    evaluation["sampler"].update(n_walkers=8, burn_in=2, n_steps=1)
    evaluation["evaluator"]["tasks"][0]["generator"]["n_draws"] = 3
    evaluation["evaluator"]["tasks"][0]["generator"]["chunk_size"] = 5
    evaluation["evaluator"]["tasks"][0]["summaries"][0]["max_samples"] = 24
    assert run_from_config(OmegaConf.create(evaluation), command="h-v2 collector eval", raise_exceptions=True) == 0

    eval_run_dir = tmp_path / "tpen_h_v2" / "atomistic" / "collector-eval"
    trajectory_dir = eval_run_dir / "trajectory"
    csv_path = trajectory_dir / "sampled_eval_table.csv"
    metadata_path = trajectory_dir / "sampled_eval_table.metadata.json"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        header = csv.DictReader(handle).fieldnames
    real_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    record_property("real_csv_header", json.dumps(header))
    record_property("real_metadata_keys", json.dumps(sorted(real_metadata)))
    print(f"real_csv_header={header}")
    print(f"real_metadata_keys={sorted(real_metadata)}")

    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    resolved_config_path = eval_run_dir / "resolved_config.yaml"
    run_metadata_path = eval_run_dir / "metadata.json"
    spec = collect.RecordSpec(
        source_sha=source_sha,
        checkpoint_model_sha256=collect.file_sha256(checkpoint_dir / "model.pt"),
        resolved_config_sha256=collect.file_sha256(resolved_config_path),
        walkers=8,
        draws=3,
    )
    receipt = collect.collect_records(
        csv_path=csv_path,
        metadata_path=metadata_path,
        resolved_config_path=resolved_config_path,
        run_metadata_path=run_metadata_path,
        checkpoint_dir=checkpoint_dir,
        spec=spec,
    )
    assert receipt["status"] == "complete"
    assert receipt["row_count"] == 24
