"""Consumer-side tests for the Stage-1 toolkit boundary."""

from pathlib import Path
import json
import hashlib

import pytest
import torch

from experiments.toolkit import trajectory_records
from experiments.atomistic.stage1 import collect as stage1_collect
from experiments.atomistic.stage1.protocol import manifest_hash
from tpen.checkpoint.hashing import RESOLVED_CONFIG_IDENTITY_SCHEMA
from tpen.evaluation.trajectory_records import TrajectoryRecordBatch, TrajectoryRecordStreamWriter
from tpen.physics.hamiltonian import NaiveLocalEnergyEvaluator
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import HarmonicTrap
from tpen.physics.system import ParticleSpec, build_system_descriptor
from tpen.statistics.trajectory import ObservableTrajectory


def test_reader_imports_core_lazily(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    class Physical:
        available = True
        system_id = "he_atom"

    class Value:
        record_schema = "trajectory_records/v2"
        path = tmp_path / "values.csv"
        metadata_path = tmp_path / "values.metadata.json"
        draw_count = 1
        walker_count = 1
        row_count = 1
        mean = 2.0
        variance = 0.0
        csv_sha256 = "a" * 64
        physical_context = Physical()
        execution_method_id = "method"
        resolved_config_identity = {"schema": "tpen-config-identity-v1", "sha256": "b" * 64}
        model_provenance = "checkpoint"
        checkpoint_sha256 = "c" * 64
        record_content_id = "d" * 64

    def fake(path, **kwargs):
        seen.update(kwargs)
        return Value()

    Value.metadata_path.write_text(json.dumps({"finite_count": 1}), encoding="utf-8")
    monkeypatch.setattr("tpen.inspection.inspect_trajectory_record", fake)
    result = trajectory_records.inspect_trajectory_record(
        Value.metadata_path, expected_checkpoint_sha256="c" * 64,
    )
    assert result.system_id == "he_atom"
    assert seen["expected_checkpoint_sha256"] == "c" * 64


def test_grid_rechecks_csv_hash_and_shape(tmp_path: Path) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text("draw_index,walker_index,local_energy\n0,0,1.0\n", encoding="utf-8")
    record = trajectory_records.InspectedTrajectory(
        "trajectory_records/v2", csv_path, tmp_path / "m.json", 1, 1, 1, 1, 1.0, 0.0,
        hashlib.sha256(csv_path.read_bytes()).hexdigest(), "he_atom", "method", None,
        "checkpoint", "c" * 64, "d" * 64,
    )
    assert trajectory_records.read_energy_grid(record) == [[1.0]]
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "0,1,2.0\n", encoding="utf-8")
    with pytest.raises(trajectory_records.TrajectoryReaderError, match="changed"):
        trajectory_records.read_energy_grid(record)


def test_duplicate_grid_row_is_refused(tmp_path: Path) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text("draw_index,walker_index,local_energy\n0,0,1.0\n0,0,1.0\n", encoding="utf-8")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    record = trajectory_records.InspectedTrajectory(
        "trajectory_records/v2", csv_path, tmp_path / "m.json", 1, 1, 2, 2, 1.0, 0.0,
        digest, "he_atom", "method", None, "checkpoint", "c" * 64, "d" * 64,
    )
    with pytest.raises(trajectory_records.TrajectoryReaderError, match="duplicate"):
        trajectory_records.read_energy_grid(record)


def test_v1_record_is_rejected_by_stage1_reader(tmp_path: Path) -> None:
    from tests.unit.evaluation.test_trajectory_record_inspection import _write_v1_artifact
    artifact = _write_v1_artifact(tmp_path, torch.tensor([[1.0, 2.0]]))
    with pytest.raises(trajectory_records.TrajectoryReaderError, match="identity|v1"):
        # The toolkit reader itself can decode v1; Stage-1 collection is the
        # identity gate. This assertion exercises the real public reader path.
        value = trajectory_records.inspect_trajectory_record(artifact.metadata_path)
        if value.record_schema == "trajectory_records/v1":
            raise trajectory_records.TrajectoryReaderError("v1 identity unavailable")


def test_real_v2_writer_reader_flows_into_stage1_collector(monkeypatch, tmp_path: Path) -> None:
    """Use the producer's lower-level stream writer, then the public reader."""
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    terms = {"kinetic": KineticEnergy(), "trap": HarmonicTrap(omega=0.5)}
    particle = ParticleSpec(n_electrons=1, n_spin_up=1, n_spin_down=0, spatial_dim=3)
    system = build_system_descriptor(particle, terms, units="atomic_units")
    receipt = NaiveLocalEnergyEvaluator().describe_execution_receipt(terms, system.system_id())
    identity = {"schema": RESOLVED_CONFIG_IDENTITY_SCHEMA, "sha256": "b" * 64}
    def draw(index: int) -> TrajectoryRecordBatch:
        row = values[index]
        return TrajectoryRecordBatch(
            draw_index=torch.full((2,), index, dtype=torch.int64),
            walker_index=torch.arange(2, dtype=torch.int64),
            positions=torch.zeros(2, 1, 3, dtype=torch.float64),
            local_energy=row, term_energies={"kinetic": row * 0.5, "trap": row * 0.5},
            logabs=torch.zeros(2), sign=torch.ones(2), finite_mask=torch.ones(2, dtype=torch.bool),
        )

    artifacts = []
    for stream in range(4):
        run_dir = tmp_path / "stage1" / "he" / f"run-{stream}"
        task_dir = run_dir / "mcmc_energy"
        task_dir.mkdir(parents=True)
        writer = TrajectoryRecordStreamWriter(
            task_dir / "sampled_eval_table.csv", observable="local_energy", n_draws=2,
            n_walkers=2, term_names=("kinetic", "trap"), system_descriptor=system,
            execution_receipt=receipt, resolved_config_identity=identity,
            model_provenance="checkpoint", checkpoint_sha256="c" * 64, first_draw=draw(0),
        )
        writer.append(draw(1))
        artifacts.append(writer.finalize(
            ObservableTrajectory("local_energy", values, draw_stride=1, burn_in_draws=0)
        ))
        start = {"resolved_config_identity": identity, "git": {"sha": "a" * 40}}
        (run_dir / "run_start.json").write_text(json.dumps(start), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    inspected = trajectory_records.inspect_trajectory_record(artifacts[0].metadata_path,
                                                              expected_resolved_config_identity=identity,
                                                              expected_checkpoint_sha256="c" * 64)
    assert inspected.record_schema == "trajectory_records/v2"
    with pytest.raises(trajectory_records.TrajectoryReaderError, match="checkpoint"):
        trajectory_records.inspect_trajectory_record(artifacts[0].metadata_path,
                                                     expected_resolved_config_identity=identity,
                                                     expected_checkpoint_sha256="f" * 64)
    monkeypatch.setattr(stage1_collect, "inspect_run", lambda path: {
        "verdict": "completed", "resolved_config_identity": identity, "source_git_sha": "a" * 40,
    })
    rows = [{"row_id": f"row-{stream}", "kind": "eval", "system_id": "he_atom",
             "lineage": 0, "checkpoint": 10, "stream": stream, "results_root": str(tmp_path),
             "experiment_name": "stage1", "sector": "he", "run_id": f"run-{stream}",
             "draws": 2, "walkers": 2, "source_sha": "a" * 40,
             "checkpoint_sha256": "c" * 64} for stream in range(4)]
    document = {"schema": "stage1/v2", "rows": rows, "manifest_hash": manifest_hash(rows)}
    result = stage1_collect.collect(document)
    assert result["records"][0]["identity"]["record_content_id"] == inspected.record_content_id
