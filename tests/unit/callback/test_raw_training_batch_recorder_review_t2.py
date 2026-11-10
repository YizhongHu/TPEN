"""T2 delta probes for the v2 training-history integrity envelope."""

from __future__ import annotations

import json
import random
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.callback.raw_training_batch_recorder import (
    RawTrainingBatchRecorder,
    TrainingBatchPartitionWriter,
    TrainingBatchReader,
    TrainingBatchRecord,
)
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.events import Occurrence
from tpen.training.events import TrainingCompleted, TrainingIteration, TrainingIterationCompleted
from tpen.training.state import TrainerState


def _context(tmp_path: Path) -> RunContext:
    manager = ArtifactManager(tmp_path / "artifacts", "review", "unit", "run", layout="flat")
    manager.make_dirs()
    return RunContext(
        cfg=OmegaConf.create({"trainer": {"max_steps": 4}}),
        source_cfg=OmegaConf.create({}),
        artifact_manager=manager,
        metadata=RunMetadata(
            run_id="run",
            run_name="run",
            timestamp="2026-01-01T00:00:00+00:00",
            timezone="UTC",
            git_commit="review-sha",
            git_branch="review",
            dirty_worktree=False,
            command="pytest",
            config_path="review.yaml",
            resolved_config_path=str(manager.path("resolved_config.yaml")),
            run_dir=str(manager.run_dir),
            device="cpu",
            dtype="float64",
        ),
        clock=RunClock(timezone="UTC", tzinfo=UTC),
    )


def _record(step: int) -> TrainingBatchRecord:
    return TrainingBatchRecord(
        iteration_step=step,
        completed_updates=step,
        update_applied=step > 0,
        walker_index=np.arange(2),
        positions=np.asarray([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]]),
        local_energy=np.asarray([step + 0.25, step + 0.5]),
        term_energies={"kinetic": np.asarray([1.0, 2.0]), "potential": np.asarray([-0.5, -0.25])},
        logabs=np.asarray([10.0 + step, 11.0 + step]),
        sign=np.ones(2),
        finite=np.ones(2, dtype=bool),
    )


def _state(step: int) -> TrainerState:
    return TrainerState(
        trainer=SimpleNamespace(completed_updates=step),
        batch=ElectronBatch(torch.tensor([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]])),
        local_energy=torch.tensor([step + 0.25, step + 0.5]),
        local_energy_terms={"kinetic": torch.tensor([1.0, 2.0]), "potential": torch.tensor([-0.5, -0.25])},
        wavefunction_output=WavefunctionOutput(torch.tensor([10.0 + step, 11.0 + step]), torch.ones(2)),
        optimizer_step=step > 0,
    )


def _emit_steps(callback: RawTrainingBatchRecorder, context: RunContext, steps: range) -> None:
    for step in steps:
        callback.handle_occurrence(
            Occurrence(TrainingIterationCompleted(TrainingIteration(step)), 1),
            context,
            _state(step),
        )
    callback.handle_occurrence(Occurrence(TrainingCompleted(), 1), context, _state(steps[-1]))


def test_reader_refuses_sidecar_provenance_mutation(tmp_path: Path) -> None:
    """A valid bundle must not remain valid after sidecar provenance changes."""

    provenance = {
        "run_id": "run-1",
        "source_revision": "source-sha",
        "config_hash": "config-sha",
        "sampler_settings": {"n_walkers": 2, "burn_in": 3},
    }
    for index in range(2):
        writer = TrainingBatchPartitionWriter(
            tmp_path, partition_index=index, planned_steps=[index], provenance=provenance
        )
        writer.append(_record(index))
        writer.publish()

    reader = TrainingBatchReader(tmp_path)
    reader.validate()
    assert reader.provenance[0]["run_id"] == "run-1"
    assert reader.provenance[0]["source_revision"] == "source-sha"
    assert reader.provenance[0]["config_hash"] == "config-sha"
    assert reader.provenance[0]["sampler_settings"] == {"n_walkers": 2, "burn_in": 3}
    with pytest.raises(AssertionError):
        assert reader.provenance[0]["run_id"] == "wrong"

    sidecar_path = tmp_path / "partition-00000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["provenance"]["run_id"] = "tampered-run"
    sidecar["provenance"]["source_revision"] = "tampered-source"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="sidecar|hash|provenance|envelope"):
        TrainingBatchReader(tmp_path).validate()


def test_reader_refuses_swapped_complete_partition_bundles(tmp_path: Path) -> None:
    """Manifest paths must bind to their original complete partition bundles."""

    for index in range(2):
        writer = TrainingBatchPartitionWriter(tmp_path, partition_index=index, planned_steps=[index])
        writer.append(_record(index))
        writer.publish()
    TrainingBatchReader(tmp_path).validate()
    manifest_before = (tmp_path / "manifest.json").read_bytes()

    sidecars = []
    bundles = []
    for index in range(2):
        sidecar_path = tmp_path / f"partition-{index:05d}.json"
        bundle_path = tmp_path / f"partition-{index:05d}.npz"
        sidecars.append(json.loads(sidecar_path.read_text()))
        bundles.append(bundle_path.read_bytes())
    for index in range(2):
        sidecar = sidecars[1 - index]
        filename = f"partition-{index:05d}.npz"
        sidecar["partition_index"] = index
        sidecar["compressed_filename"] = filename
        sidecar["selected_filename"] = filename
        (tmp_path / f"partition-{index:05d}.npz").write_bytes(bundles[1 - index])
        (tmp_path / f"partition-{index:05d}.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
        )

    assert (tmp_path / "manifest.json").read_bytes() == manifest_before
    with pytest.raises(ValueError, match="partition|lineage|identity"):
        TrainingBatchReader(tmp_path).validate()


def test_probabilistic_cadence_is_stable_across_partitions_or_rejected(tmp_path: Path) -> None:
    """The probability stream must not create a plan containing rejected steps."""

    try:
        callback = RawTrainingBatchRecorder(
            tmp_path / "probabilistic",
            partition_size_steps=2,
            every_n_steps=1,
            probability=0.5,
            seed=1,
        )
    except (TypeError, ValueError) as exc:
        assert "probability" in str(exc).lower()
        return

    oracle = random.Random(1)
    expected = [step for step in range(4) if oracle.random() < 0.5]
    assert expected == [0, 3]
    _emit_steps(callback, _context(tmp_path / "probabilistic"), range(4))
    rows = TrainingBatchReader(tmp_path / "probabilistic").read()
    np.testing.assert_array_equal(np.unique(rows["iteration_step"]), expected)


def test_probability_one_control_retains_every_step(tmp_path: Path) -> None:
    control = RawTrainingBatchRecorder(
        tmp_path / "deterministic",
        partition_size_steps=2,
        every_n_steps=1,
        probability=1.0,
        seed=1,
    )
    _emit_steps(control, _context(tmp_path / "deterministic"), range(4))
    control_rows = TrainingBatchReader(tmp_path / "deterministic").read()
    np.testing.assert_array_equal(np.unique(control_rows["iteration_step"]), [0, 1, 2, 3])
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(np.unique(control_rows["iteration_step"]), [0, 2])
