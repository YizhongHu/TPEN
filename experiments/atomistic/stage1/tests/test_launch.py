"""Executable launch-guard tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from experiments.atomistic.stage1.launch import render


ROW = {"kind": "train", "config": "configs/he/train.yaml",
       "source_sha": "c" * 40, "config_sha256": "b" * 64,
       "results_root": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/he/train"}


def test_render_contains_fail_closed_identity_and_cleanup_guards() -> None:
    script = render(row=ROW, driver="driver.py", python="/usr/bin/python3",
                    head_sha="c" * 40,
                    results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1")
    assert "STAGE1_GUARD missing SLURM_JOB_ID" in script
    assert '"$SLURM_JOB_PARTITION" != "$REQUESTED_PARTITION"' in script
    assert 'JOB_ROOT="$SCRATCH_BASE/$USER/$SLURM_JOB_ID"' in script
    assert "STAGE1_GUARD results root outside" in script
    assert '[[ "$JOB_ROOT" == "$SCRATCH_BASE/$USER/$SLURM_JOB_ID" ]]' in script
    assert "trap 'rm -rf -- \"$JOB_ROOT\"' EXIT" in script
    assert "exec " not in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_render_fails_for_unapproved_partition_or_results_root() -> None:
    with pytest.raises(ValueError, match="partition"):
        render(row=ROW, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
               results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1",
               partition="gpu")
    with pytest.raises(ValueError, match="Netscratch"):
        render(row=ROW, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
               results_root="/tmp/stage1")


def test_partition_guard_is_exercised_by_fake_environment() -> None:
    script = render(row=ROW, driver="/bin/true", python="/usr/bin/true",
                    head_sha="c" * 40,
                    results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1")
    missing_job = subprocess.run(["bash"], input=script, text=True, capture_output=True)
    assert missing_job.returncode == 2
    assert "STAGE1_GUARD missing SLURM_JOB_ID" in missing_job.stderr
    mismatch = subprocess.run(["bash"], input=script, text=True, capture_output=True,
                              env={"USER": "tester", "SLURM_JOB_ID": "123",
                                   "SLURM_JOB_PARTITION": "gpu_test"})
    assert mismatch.returncode == 2
    assert "STAGE1_GUARD partition delivered=gpu_test requested=test" in mismatch.stderr


def _rendered_script() -> str:
    return render(row=ROW, driver="/bin/true", python="/usr/bin/true",
                  head_sha="c" * 40,
                  results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1")


def test_guard_fails_without_slurm_job_id() -> None:
    result = subprocess.run(["bash"], input=_rendered_script(), text=True,
                            capture_output=True)
    assert result.returncode == 2
    assert "STAGE1_GUARD missing SLURM_JOB_ID" in result.stderr


def test_rendered_prefix_guard_executes_under_fake_environment() -> None:
    script = _rendered_script().replace(
        "RESULTS_ROOT='/n/netscratch/kozinsky_lab/Lab/rhu/stage1'",
        "RESULTS_ROOT='/outside/approved/root'",
    )
    result = subprocess.run(["bash"], input=script, text=True, capture_output=True,
                            env={"USER": "tester", "SLURM_JOB_ID": "123",
                                 "SLURM_JOB_PARTITION": "test"})
    assert result.returncode == 2
    assert "STAGE1_GUARD results root outside /n/netscratch/kozinsky_lab/Lab/rhu/" in result.stderr


def test_rendered_exit_trap_removes_job_root(tmp_path: Path) -> None:
    marker = tmp_path / "tester" / "123"
    script = _rendered_script()
    result = subprocess.run(["bash"], input=script, text=True, capture_output=True,
                            env={"USER": "tester", "SLURM_JOB_ID": "123",
                                 "SLURM_JOB_PARTITION": "test",
                                 "STAGE1_SCRATCH_BASE": str(tmp_path)})
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
