"""No-compute integration of planning, launch bindings, and collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from experiments.atomistic.stage1.collect import collect
from experiments.atomistic.stage1.launch import _binding_token, row_overrides
from experiments.atomistic.stage1.protocol import (
    Protocol, build_manifest, manifest_hash, validate_gateway_binding,
)
from experiments.toolkit.trajectory_records import inspect_run, inspect_trajectory_record
from tpen.artifacts import ArtifactManager
from tpen.checkpoint.artifact import resolve_checkpoint_dir
from tpen.evaluation.trajectory_records import TrajectoryRecordBatch, TrajectoryRecordStreamWriter
from tpen.physics.hamiltonian import NaiveLocalEnergyEvaluator
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import HarmonicTrap
from tpen.physics.system import ParticleSpec, build_system_descriptor
from tpen.statistics.trajectory import ObservableTrajectory


def _checkpoint(path: Path, payload: bytes) -> str:
    path.mkdir(parents=True)
    (path / "COMPLETE").write_text("", encoding="utf-8")
    (path / "manifest.json").write_text("{}\n", encoding="utf-8")
    model = path / "model.pt"
    model.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_planned_launch_paths_and_real_records_collect_without_compute(tmp_path: Path) -> None:
    """Reverting either checkpoint or run-path binding must break this test."""

    source_sha = "a" * 40
    model_payload = b"stage1-e2e-model"
    checkpoint_sha = hashlib.sha256(model_payload).hexdigest()
    protocol = Protocol(1, 2, 1, 2, 3, 4, 2, 2)
    bindings = {("hooke1", lineage, 7): checkpoint_sha for lineage in range(3)}
    planned = build_manifest(
        protocol=protocol, source_sha=source_sha, config_sha256="b" * 64,
        checkpoints=[7], lineage_seeds=[11, 12, 13], checkpoint_sha256=bindings,
        results_root=str(tmp_path),
    )
    rows = [row for row in planned if row["system"] == "hooke1" and row["lineage"] == 0
            and row["checkpoint"] == 7]
    document = {"schema": "stage1/v2", "manifest_hash": manifest_hash(rows), "rows": rows}

    train_row = next(row for row in planned if row["kind"] == "train" and row["system"] == "hooke1"
                     and row["lineage"] == 0)
    train_cfg = OmegaConf.from_dotlist(row_overrides(train_row))
    train_dir = ArtifactManager(
        train_cfg.run.root, train_cfg.experiment.name, train_cfg.experiment.sector,
        train_cfg.run.run_id,
    ).run_dir
    train_checkpoint = train_dir / "checkpoints" / "step_000007"
    assert _checkpoint(train_checkpoint, model_payload) == checkpoint_sha
    assert resolve_checkpoint_dir(train_checkpoint) == train_checkpoint

    terms = {"kinetic": KineticEnergy(), "trap": HarmonicTrap(omega=0.5)}
    particle = ParticleSpec(n_electrons=1, n_spin_up=1, n_spin_down=0, spatial_dim=3)
    system = build_system_descriptor(particle, terms, units="atomic_units")
    receipt = NaiveLocalEnergyEvaluator().describe_execution_receipt(terms, system.system_id())
    identity = {"schema": "tpen-config-identity-v1", "sha256": "c" * 64}
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)

    def draw(index: int) -> TrajectoryRecordBatch:
        row = values[index]
        return TrajectoryRecordBatch(
            draw_index=torch.full((2,), index, dtype=torch.int64),
            walker_index=torch.arange(2, dtype=torch.int64),
            positions=torch.zeros(2, 1, 3, dtype=torch.float64), local_energy=row,
            term_energies={"kinetic": row * 0.5, "trap": row * 0.5},
            logabs=torch.zeros(2), sign=torch.ones(2),
            finite_mask=torch.ones(2, dtype=torch.bool),
        )

    metadata_paths = []
    for row in rows:
        overrides = row_overrides(row)
        validate_gateway_binding(
            binding=_binding_token(row), config=row["config"], overrides=overrides,
            kind="eval", expected_checkpoint_sha256=checkpoint_sha,
        )
        assert f"load.path={row['checkpoint_dir']}" in overrides
        assert f"trajectory_identity.checkpoint_file={row['checkpoint_file']}" in overrides
        eval_cfg = OmegaConf.from_dotlist(overrides)
        eval_dir = ArtifactManager(
            eval_cfg.run.root, eval_cfg.experiment.name, eval_cfg.experiment.sector,
            eval_cfg.run.run_id,
        ).run_dir
        load_path = Path(eval_cfg.load.path)
        checkpoint_file = Path(eval_cfg.trajectory_identity.checkpoint_file)
        assert load_path == train_checkpoint
        assert checkpoint_file == load_path / "model.pt"
        assert hashlib.sha256(checkpoint_file.read_bytes()).hexdigest() == row["checkpoint_sha256"]
        assert resolve_checkpoint_dir(load_path) == load_path
        # A real eval run points at the planned train checkpoint for inspection;
        # it does not invent a second eval checkpoint.
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "checkpoints").symlink_to(train_dir / "checkpoints", target_is_directory=True)
        run_dir = eval_dir
        task_dir = run_dir / "mcmc_energy"
        task_dir.mkdir(parents=True)
        writer = TrajectoryRecordStreamWriter(
            task_dir / "sampled_eval_table.csv", observable="local_energy", n_draws=2,
            n_walkers=2, term_names=("kinetic", "trap"), system_descriptor=system,
            execution_receipt=receipt, resolved_config_identity=identity,
            model_provenance="checkpoint", checkpoint_sha256=checkpoint_sha,
            first_draw=draw(0),
        )
        writer.append(draw(1))
        artifact = writer.finalize(
            ObservableTrajectory("local_energy", values, draw_stride=1, burn_in_draws=0)
        )
        metadata_paths.append(artifact.metadata_path)
        (run_dir / "run_start.json").write_text(
            json.dumps({"resolved_config_identity": identity, "git": {"sha": source_sha}}),
            encoding="utf-8",
        )
        (run_dir / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        assert inspect_run(run_dir)["verdict"] == "completed"

    result = collect(document)
    assert len(result["records"]) == 4
    assert all(record["identity"]["checkpoint_sha256"] == checkpoint_sha
               for record in result["records"])
    for metadata_path in metadata_paths:
        inspected = inspect_trajectory_record(
            metadata_path, expected_resolved_config_identity=identity,
            expected_checkpoint_sha256=checkpoint_sha,
        )
        assert inspected.model_provenance == "checkpoint"
    assert all(record["identity"]["resolved_config_identity"] == identity
               for record in result["records"])
