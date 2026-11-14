"""Executable launch-guard tests."""

from __future__ import annotations

import subprocess

import pytest

from experiments.atomistic.stage1.launch import render


ROW = {"kind": "train", "config": "configs/he/train.yaml",
       "source_sha": "c" * 40, "config_sha256": "b" * 64,
       "results_root": "/approved/netscratch/stage1/he/train"}


def test_render_contains_fail_closed_identity_and_cleanup_guards() -> None:
    script = render(row=ROW, driver="driver.py", python="/usr/bin/python3",
                    head_sha="c" * 40,
                    results_root="/approved/netscratch/stage1",
                    netscratch_root="/approved/netscratch")
    assert ": \"${SLURM_JOB_ID:?must run inside Slurm}\"" in script
    assert '"$SLURM_JOB_PARTITION" != "$REQUESTED_PARTITION"' in script
    assert 'JOB_ROOT="/scratch/$USER/$SLURM_JOB_ID"' in script
    assert "trap 'rm -rf -- \"$JOB_ROOT\"' EXIT" in script
    assert "exec " not in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_render_fails_for_unapproved_partition_or_results_root() -> None:
    with pytest.raises(ValueError, match="partition"):
        render(row=ROW, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
               results_root="/approved/netscratch/stage1", netscratch_root="/approved/netscratch",
               partition="gpu")
    with pytest.raises(ValueError, match="Netscratch"):
        render(row=ROW, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
               results_root="/tmp/stage1", netscratch_root="/approved/netscratch")


def test_partition_guard_is_exercised_by_fake_environment() -> None:
    script = render(row=ROW, driver="/bin/true", python="/usr/bin/true",
                    head_sha="c" * 40,
                    results_root="/approved/netscratch/stage1",
                    netscratch_root="/approved/netscratch")
    missing_job = subprocess.run(["bash"], input=script, text=True, capture_output=True)
    assert missing_job.returncode != 0
    mismatch = subprocess.run(["bash"], input=script, text=True, capture_output=True,
                              env={"USER": "tester", "SLURM_JOB_ID": "123",
                                   "SLURM_JOB_PARTITION": "gpu_test"})
    assert mismatch.returncode == 2
