"""Static launch guards are tested without submitting a job."""

import pytest

from launch import render


def test_template_is_partition_and_allocation_guarded() -> None:
    script = render(driver="train.py", config="x.yaml", python="/opt/python", job_root="$TMPDIR/stage1")
    assert "#SBATCH --partition=test" in script
    assert "SLURM_JOB_PARTITION" in script
    assert "SLURM_JOB_ID" in script
    assert "trap 'rm -rf -- \"$JOB_ROOT\"' EXIT" in script


def test_relative_interpreter_is_rejected() -> None:
    with pytest.raises(ValueError, match="absolute"):
        render(driver="train.py", config="x.yaml", python="python", job_root="x")
