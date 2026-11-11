"""Static launch guards are tested without submitting a job."""

import pytest

from experiments.atomistic.stage1.launch import render
from experiments.atomistic.stage1.protocol import Protocol, build_manifest


ROW = build_manifest(Protocol(), checkpoints=[1000], seeds=[1, 2, 3],
                     results_root="/results/stage1")[0]


def test_template_is_partition_and_allocation_guarded() -> None:
    script = render(driver="train.py", row=ROW, python="/opt/python",
                   head_sha="a" * 40, results_root="/results/stage1")
    assert "#SBATCH --partition=test" in script
    assert "REQUESTED_PARTITION='test'" in script
    assert "SLURM_JOB_PARTITION" in script
    assert "SLURM_JOB_ID" in script
    assert "trap 'rm -rf -- \"$JOB_ROOT\"' EXIT" in script
    assert 'JOB_ROOT="/scratch/$USER/$SLURM_JOB_ID"' in script
    assert "HEAD_SHA" in script
    assert "--config configs/hooke1/smoke.yaml" in script
    assert "trainer.max_steps=1000" in script
    assert "run.root=/results/stage1/hooke1/smoke" in script
    assert "--job-root" not in script
    assert "exec \"$PYTHON\"" not in script


def test_relative_interpreter_is_rejected() -> None:
    with pytest.raises(ValueError, match="absolute"):
        render(driver="train.py", row=ROW, python="python", head_sha="a" * 40,
               results_root="/results/stage1")


def test_unapproved_partition_is_rejected() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        render(driver="train.py", row=ROW, python="/opt/python", head_sha="a" * 40,
               results_root="/results/stage1", partition="seas_compute")


def test_head_and_results_root_are_explicit() -> None:
    with pytest.raises(ValueError, match="full 40"):
        render(driver="train.py", row=ROW, python="/opt/python", head_sha="abc",
               results_root="/results/stage1")
    with pytest.raises(ValueError, match="outside"):
        render(driver="train.py", row=ROW, python="/opt/python", head_sha="a" * 40,
               results_root="experiments/atomistic/stage1/outputs")
