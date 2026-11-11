"""Tests for the PBS/PALS -> SLURM translation used for DeepQMC multi-host DDP.

The mapping is a pure function, so most of this tests it directly. Two subprocess
tests cover the ``exec`` path, which a pure-function test cannot reach.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.baselines.polaris.multihost_launch import slurm_env_from_pbs

MODULE = "experiments.baselines.polaris.multihost_launch"
PBS = {"PBS_JOBID": "7587171.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"}


def test_job_id_is_the_integer_prefix() -> None:
    """jax computes its coordination port as SLURM_JOB_ID % 4096 + 61440.

    A PBS job id carries a hostname suffix; passing it through unchanged makes the
    int() conversion inside jax raise, so the whole handshake never starts.
    """

    assert slurm_env_from_pbs(PBS, "hostA\n")["SLURM_JOB_ID"] == "7587171"


def test_nodelist_is_sorted_and_deduplicated() -> None:
    """Every rank must derive the SAME coordinator, which jax takes as the first host.

    PBS_NODEFILE repeats a hostname once per slot and its order is not guaranteed,
    so identical sorting on every rank is what makes their choices agree. Ranks that
    disagree would each wait for a coordinator the others never contact.
    """

    got = slurm_env_from_pbs(PBS, "hostB\nhostA\nhostB\nhostA\n")
    assert got["SLURM_STEP_NODELIST"] == "hostA,hostB"


def test_multi_rank_values_come_from_pals() -> None:
    got = slurm_env_from_pbs(
        {**PBS, "PMI_SIZE": "2", "PMI_RANK": "1", "PALS_LOCAL_RANKID": "0"}, "a\nb\n"
    )
    assert (got["SLURM_NTASKS"], got["SLURM_PROCID"], got["SLURM_LOCALID"]) == ("2", "1", "0")


def test_pals_spelling_is_accepted_as_a_fallback() -> None:
    """PALS_* is the alternative spelling; relying on PMI_* alone would silently
    produce a single-rank mapping on a launcher that only sets PALS_*."""

    got = slurm_env_from_pbs({**PBS, "PALS_NRANKS": "4", "PALS_RANKID": "3"}, "a\n")
    assert (got["SLURM_NTASKS"], got["SLURM_PROCID"]) == ("4", "3")


def test_single_rank_must_not_claim_multiple_tasks() -> None:
    """The dangerous direction.

    DeepQMC calls jax.distributed.initialize() only when SLURM_NTASKS > 1. A lone
    process reporting 2 would attempt a coordinator handshake against a peer that
    never arrives, and hang until the initialization timeout instead of running.
    """

    got = slurm_env_from_pbs(PBS, "hostA\n")
    assert got["SLURM_NTASKS"] == "1"
    assert got["SLURM_PROCID"] == "0"


def test_visible_devices_default_and_override() -> None:
    """initialize() asserts CUDA_VISIBLE_DEVICES is set and parses it as device ids."""

    assert slurm_env_from_pbs(PBS, "a\n")["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert slurm_env_from_pbs({**PBS, "MH_VISIBLE": "0,1"}, "a\n")["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_outside_pbs_raises_rather_than_guessing() -> None:
    """Outside PBS the mapping would be silently wrong, and a silently wrong mapping
    yields independent single-rank runs that all look successful."""

    with pytest.raises(KeyError):
        slurm_env_from_pbs({}, "hostA\n")


def test_empty_nodefile_is_rejected() -> None:
    """An empty node list would make jax derive an empty coordinator host."""

    with pytest.raises(ValueError):
        slurm_env_from_pbs(PBS, "\n \n")


def test_exec_path_replaces_the_process_with_the_wrapped_command(tmp_path: Path) -> None:
    """Covers what the pure function cannot: that the command actually runs, in the
    translated environment."""

    nodefile = tmp_path / "nodes"
    nodefile.write_text("hostA\nhostB\n", encoding="utf-8")
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        "PBS_NODEFILE": str(nodefile),
        "PMI_SIZE": "2",
        "PMI_RANK": "1",
        **PBS,
    }
    proc = subprocess.run(
        [sys.executable, "-m", MODULE, "env"], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr
    got = dict(l.split("=", 1) for l in proc.stdout.splitlines() if "=" in l)
    assert got["SLURM_NTASKS"] == "2"
    assert got["SLURM_PROCID"] == "1"
    assert got["SLURM_STEP_NODELIST"] == "hostA,hostB"
    assert "MHLAUNCH rank=1/2" in proc.stderr


def test_exec_path_refuses_outside_an_allocation() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", MODULE, "env"],
        capture_output=True, text=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
             "PYTHONPATH": str(Path(__file__).resolve().parents[3])},
    )
    assert proc.returncode == 2
    assert "PBS_NODEFILE" in proc.stderr
