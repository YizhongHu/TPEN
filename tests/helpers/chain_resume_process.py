"""Test-only process launcher for the chain-resume adapter unit tests.

The launcher belongs under ``tests/helpers`` because process creation and the
synthetic probe are qualification machinery, not core execution contracts.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from tpen.execution.chain_resume import ProcessEvidence


def run_fresh_process(
    command: Sequence[str],
    *,
    phase: str,
    seed: int,
    seed_source: str,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ProcessEvidence:
    """Run one synthetic adapter fixture in a fresh process.

    Parameters
    ----------
    command : sequence of str
        Test helper command to execute.
    phase, seed, seed_source : str or int
        Evidence labels retained in the pure core ``ProcessEvidence`` value.
    cwd : pathlib.Path, optional
        Child working directory.
    env : mapping of str to str, optional
        Environment additions for the child.

    Returns
    -------
    ProcessEvidence
        Parsed event evidence, including the child PID.
    """

    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env["TPEN_CHAIN_RESUME_SEED"] = str(seed)
    child_env["TPEN_CHAIN_RESUME_PHASE"] = phase
    process = subprocess.Popen(
        tuple(command),
        cwd=None if cwd is None else str(cwd),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"chain-resume fixture process failed ({phase}, rc={process.returncode}): {stderr}"
        )
    return ProcessEvidence.from_json_output(
        phase=phase,
        pid=process.pid,
        seed=seed,
        seed_source=seed_source,
        command=tuple(command),
        returncode=process.returncode,
        stdout=stdout,
    )


__all__ = ["run_fresh_process"]
