"""Public-entry v3 recorder continuation, with the legacy default preserved."""

from pathlib import Path

import numpy as np
import pytest

from tpen.artifacts import ArtifactDestination
from tpen.callback import PlannedYield
from tpen.callback.raw_training_batch_recorder import TrainingHistoryReader
from tpen.run import run_from_config
from tests.integration.test_s4_planned_yield_public_entry import _cfg


def _recording_config(root: Path, run_id: str, every: int):
    cfg = _cfg(root, run_id, max_steps=5)
    cfg.trainer.return_terms = True
    cfg.callbacks.append({
        "_target_": "tpen.callback.RawTrainingBatchRecorder",
        "path": "${run.dir}/training_history",
        "format_version": 3,
        "partition_size_steps": 2,
        "every_n_steps": every,
    })
    return cfg


@pytest.mark.parametrize("cut,every", [(1, 1), (2, 1), (3, 2)])
def test_segmented_batches_equal_uninterrupted(tmp_path: Path, cut: int, every: int) -> None:
    """Use the real planned-yield checkpoint and restored sampler/model/RNG."""
    whole = tmp_path / "whole"
    first = tmp_path / "first"
    second = tmp_path / "second"
    logical = tmp_path / "logical"
    assert run_from_config(
        _recording_config(logical, "whole", every),
        artifact_destination=ArtifactDestination(whole), raise_exceptions=True,
    ) == 0
    assert run_from_config(
        _recording_config(logical, "first", every),
        artifact_destination=ArtifactDestination(first),
        runtime_callbacks=(PlannedYield(cut, 5),), raise_exceptions=True,
    ) == 2
    TrainingHistoryReader(first / "training_history").validate(require_complete=True)
    cfg = _recording_config(logical, "second", every)
    cfg.runner.load = {"mode": "train_resume", "path": str(first / "checkpoints" / f"step_{cut:06d}")}
    assert run_from_config(
        cfg, artifact_destination=ArtifactDestination(second), raise_exceptions=True,
    ) == 0
    reader = TrainingHistoryReader(second / "training_history")
    reader.validate(require_complete=True)
    actual = reader.read()
    expected = TrainingHistoryReader(whole / "training_history").read()
    assert actual.keys() == expected.keys()
    for column in expected:
        np.testing.assert_array_equal(actual[column], expected[column], err_msg=column)
    identities = list(zip(actual["iteration_step"], actual["walker_index"]))
    assert len(identities) == len(set(identities))
    assert set(actual["iteration_step"]) == set(range(0, 5, every))
