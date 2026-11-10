"""Contract tests for the optional raw training-history recorder."""

from __future__ import annotations

import hashlib
import json
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
from tpen.training.trainer import VMCTrainer
from tests.helpers.hooke_models import build_tiny_hamiltonian_terms, build_tiny_sampler, build_tiny_spenn


def _run_context(tmp_path: Path) -> RunContext:
    manager = ArtifactManager(tmp_path / "artifacts", "callback", "unit", "run", layout="flat")
    manager.make_dirs()
    return RunContext(
        cfg=OmegaConf.create({}),
        source_cfg=OmegaConf.create({}),
        artifact_manager=manager,
        metadata=RunMetadata(
            run_id="run",
            run_name="run",
            timestamp="2026-01-01T00:00:00+00:00",
            timezone="UTC",
            git_commit="test-sha",
            git_branch="test",
            dirty_worktree=False,
            command="pytest",
            config_path="test.yaml",
            resolved_config_path=str(manager.path("resolved_config.yaml")),
            run_dir=str(manager.run_dir),
            device="cpu",
            dtype="float64",
        ),
        clock=RunClock(timezone="UTC", tzinfo=UTC),
    )


def _record(step: int, *, applied: bool = True) -> TrainingBatchRecord:
    return TrainingBatchRecord(
        iteration_step=step,
        completed_updates=step + int(applied),
        update_applied=applied,
        walker_index=np.arange(2),
        positions=np.asarray([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]]),
        local_energy=np.asarray([step + 0.25, step + 0.5]),
        term_energies={"kinetic": np.asarray([1.0, 2.0]), "potential": np.asarray([-0.5, -0.25])},
        logabs=np.asarray([10.0 + step, 11.0 + step]),
        sign=np.ones(2),
        finite=np.ones(2, dtype=bool),
    )


def _publish_valid(root: Path) -> None:
    writer = TrainingBatchPartitionWriter(root, partition_index=0, planned_steps=[0])
    writer.append(_record(0))
    writer.publish()


def test_reader_consumes_real_training_loop_output(tmp_path: Path) -> None:
    """Validate partitions emitted by the real trainer, not hand-built rows."""

    history_root = tmp_path / "training-history"
    callback = RawTrainingBatchRecorder(history_root, partition_size_steps=1, every_n_steps=1)
    manager = ArtifactManager(tmp_path / "artifacts", "compat", "unit", "run", layout="flat")
    manager.make_dirs()
    context = RunContext(
        cfg=OmegaConf.create({"trainer": {"max_steps": 1}}),
        source_cfg=OmegaConf.create({}),
        artifact_manager=manager,
        metadata=RunMetadata(
            run_id="run",
            run_name="run",
            timestamp="2026-01-01T00:00:00+00:00",
            timezone="UTC",
            git_commit="test-sha",
            git_branch="test",
            dirty_worktree=False,
            command="pytest",
            config_path="test.yaml",
            resolved_config_path=str(manager.path("resolved_config.yaml")),
            run_dir=str(manager.run_dir),
            device="cpu",
            dtype="float64",
        ),
        clock=RunClock(timezone="UTC", tzinfo=UTC),
        callbacks=[callback],
    )
    torch.manual_seed(20261109)
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, return_terms=True)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=optimizer,
        context=context,
        emit=lambda *args, **kwargs: None,
    )

    manifest = json.loads((history_root / "manifest.json").read_text())
    assert len(manifest["partitions"]) == 1
    assert (history_root / manifest["partitions"][0]["sidecar"]).is_file()
    rows = TrainingBatchReader(history_root).read()
    assert rows["iteration_step"].size == rows["walker_index"].size
    assert set(rows["iteration_step"]) == {0}
    TrainingBatchReader(history_root).validate()


def _rewrite_bundle(root: Path, mutate) -> None:
    bundle = root / "partition-00000.npz"
    arrays = dict(np.load(bundle, allow_pickle=False))
    mutate(arrays)
    np.savez_compressed(bundle, **arrays)
    sidecar_path = root / "partition-00000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["compressed_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")


def test_round_trip_publishes_only_the_selected_compressed_bundle(tmp_path: Path) -> None:
    writer = TrainingBatchPartitionWriter(tmp_path, partition_index=0, planned_steps=[0, 1])
    writer.append(_record(0))
    writer.append(_record(1, applied=False))
    sidecar = writer.publish()

    assert sidecar["selected_filename"] == "partition-00000.npz"
    assert sidecar["uncompressed_filename"] is None
    assert sidecar["uncompressed_byte_count"] > sidecar["compressed_byte_count"]
    assert not list(tmp_path.glob("*.uncompressed.npz"))
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "manifest.json",
        "partition-00000.json",
        "partition-00000.npz",
    ]
    rows = TrainingBatchReader(tmp_path).read()
    np.testing.assert_array_equal(rows["iteration_step"], [0, 0, 1, 1])
    np.testing.assert_array_equal(rows["walker_index"], [0, 1, 0, 1])
    np.testing.assert_array_equal(rows["term/kinetic"], [1.0, 2.0, 1.0, 2.0])


def test_reader_refuses_physical_bundle_hash_mismatch(tmp_path: Path) -> None:
    _publish_valid(tmp_path)

    # Keep the archive readable and logically identical while changing its
    # physical representation, leaving the published sidecar hash stale.
    bundle = tmp_path / "partition-00000.npz"
    arrays = dict(np.load(bundle, allow_pickle=False))
    np.savez(bundle, **arrays)

    with pytest.raises(ValueError, match="physical hash"):
        TrainingBatchReader(tmp_path).validate()


def test_reader_refuses_bundle_byte_corruption_with_envelopes_unchanged(
    tmp_path: Path,
) -> None:
    """Reject altered bundle bytes before trusting the decoded arrays."""

    _publish_valid(tmp_path)
    sidecar_before = (tmp_path / "partition-00000.json").read_bytes()
    manifest_before = (tmp_path / "manifest.json").read_bytes()
    bundle = tmp_path / "partition-00000.npz"
    corrupted = bytearray(bundle.read_bytes())
    corrupted[-1] ^= 0x01
    bundle.write_bytes(corrupted)

    assert (tmp_path / "partition-00000.json").read_bytes() == sidecar_before
    assert (tmp_path / "manifest.json").read_bytes() == manifest_before
    with pytest.raises(ValueError, match="physical hash"):
        TrainingBatchReader(tmp_path).validate()


def test_reader_refuses_logical_digest_mismatch_after_rebinding_physical_hash(
    tmp_path: Path,
) -> None:
    """Reach the logical digest guard with both physical envelopes consistent."""

    _publish_valid(tmp_path)
    bundle = tmp_path / "partition-00000.npz"
    arrays = dict(np.load(bundle, allow_pickle=False))
    arrays["local_energy"] = arrays["local_energy"].copy()
    arrays["local_energy"][0] += 1.0
    np.savez_compressed(bundle, **arrays)

    sidecar_path = tmp_path / "partition-00000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["compressed_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["partitions"][0]["sidecar_sha256"] = hashlib.sha256(
        sidecar_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="logical digest"):
        TrainingBatchReader(tmp_path).validate()


def test_partition_provenance_is_written_and_exposed_by_reader(tmp_path: Path) -> None:
    """Keep run, source, sampler, and partition acceptance provenance typed."""

    provenance = {
        "run_id": "run-1",
        "config_hash": "config-sha",
        "source_revision": "source-sha",
        "sampler_settings": {"n_walkers": 2, "burn_in": 3, "n_steps": 4},
    }
    writer = TrainingBatchPartitionWriter(
        tmp_path, partition_index=0, planned_steps=[0], provenance=provenance
    )
    writer.append(
        _record(0),
        sampler_stats={"iteration_acceptance_rate": 0.75, "proposal_scale": 0.05},
    )
    writer.publish()

    sidecar = json.loads((tmp_path / "partition-00000.json").read_text())
    assert sidecar["provenance"]["run_id"] == "run-1"
    assert sidecar["provenance"]["acceptance_proposal_stats"] == [
        {"iteration_acceptance_rate": 0.75, "proposal_scale": 0.05}
    ]
    reader = TrainingBatchReader(tmp_path)
    reader.validate()
    assert reader.provenance[0]["source_revision"] == "source-sha"


def test_callback_captures_pre_update_forward_and_records_skipped_update(tmp_path: Path) -> None:
    callback = RawTrainingBatchRecorder(tmp_path, partition_size_steps=10, every_n_steps=1)
    context = _run_context(tmp_path)
    state = TrainerState(
        trainer=SimpleNamespace(completed_updates=0),
        batch=ElectronBatch(torch.tensor([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]])),
        local_energy=torch.tensor([2.0, 3.0]),
        local_energy_terms={"kinetic": torch.tensor([1.0, 1.5]), "potential": torch.tensor([1.0, 1.5])},
        wavefunction_output=WavefunctionOutput(torch.tensor([4.0, 5.0]), torch.ones(2)),
        optimizer_step=False,
    )
    callback.handle_occurrence(
        Occurrence(TrainingIterationCompleted(TrainingIteration(0)), 1), context, state
    )
    # Simulate later mutation of the in-process state after the callback's capture point.
    state.wavefunction_output = WavefunctionOutput(torch.tensor([40.0, 50.0]), torch.ones(2))
    state.local_energy = torch.tensor([20.0, 30.0])
    state.optimizer_step = True
    state.trainer.completed_updates = 1
    state.wavefunction_output = WavefunctionOutput(torch.tensor([6.0, 7.0]), torch.ones(2))
    state.local_energy = torch.tensor([6.0, 7.0])
    callback.handle_occurrence(
        Occurrence(TrainingIterationCompleted(TrainingIteration(1)), 1), context, state
    )
    # A post-update capture would observe this later state instead of the forward above.
    state.wavefunction_output = WavefunctionOutput(torch.tensor([60.0, 70.0]), torch.ones(2))
    state.local_energy = torch.tensor([60.0, 70.0])
    callback.handle_occurrence(Occurrence(TrainingCompleted(), 1), context, state)

    rows = TrainingBatchReader(tmp_path).read()
    np.testing.assert_array_equal(rows["logabs"], [4.0, 5.0, 6.0, 7.0])
    np.testing.assert_array_equal(rows["local_energy"], [2.0, 3.0, 6.0, 7.0])
    np.testing.assert_array_equal(rows["update_applied"], [False, False, True, True])
    np.testing.assert_array_equal(rows["completed_updates"], [0, 0, 1, 1])


@pytest.mark.parametrize("defect", ["missing", "duplicate", "grid", "nonfinite", "digest", "manifest"])
def test_reader_refuses_each_crafted_integrity_defect(tmp_path: Path, defect: str) -> None:
    _publish_valid(tmp_path)
    sidecar_path = tmp_path / "partition-00000.json"
    sidecar = json.loads(sidecar_path.read_text())
    if defect == "missing":
        sidecar["planned_updates"] = [0, 1]
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    elif defect == "duplicate":
        _rewrite_bundle(tmp_path, lambda arrays: arrays["walker_index"].__setitem__(1, 0))
    elif defect == "grid":
        _rewrite_bundle(tmp_path, lambda arrays: arrays["walker_index"].__setitem__(0, 1))
    elif defect == "nonfinite":
        _rewrite_bundle(tmp_path, lambda arrays: arrays["local_energy"].__setitem__(0, np.nan))
    elif defect == "digest":
        sidecar["logical_sha256"] = "0" * 64
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    else:
        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["partitions"][0]["sidecar"] = "partition-99999.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises((ValueError, FileNotFoundError)):
        TrainingBatchReader(tmp_path).validate()


def test_resume_never_rewrites_a_published_partition(tmp_path: Path) -> None:
    _publish_valid(tmp_path)
    bundle = tmp_path / "partition-00000.npz"
    before = (bundle.stat().st_mtime_ns, bundle.read_bytes())
    resumed = TrainingBatchPartitionWriter(tmp_path, partition_index=0, planned_steps=[0])
    resumed.append(_record(0))
    with pytest.raises(FileExistsError):
        resumed.publish()
    assert (bundle.stat().st_mtime_ns, bundle.read_bytes()) == before


def test_partition_memory_is_released_after_one_and_three_partitions(tmp_path: Path) -> None:
    retained: list[int] = []
    released: list[int] = []
    for partition_index in range(3):
        writer = TrainingBatchPartitionWriter(tmp_path / str(partition_index), partition_index=0, planned_steps=[0])
        writer.append(_record(0))
        retained.append(writer.memory_bytes)
        writer.publish()
        released.append(writer.memory_bytes)
    assert retained[0] > 0
    assert max(retained) - min(retained) <= 0.01 * retained[0]
    assert released == [0, 0, 0]

