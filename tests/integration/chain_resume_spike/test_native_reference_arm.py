"""ARM N: G0 reproduced against the REAL ``tpen.checkpoint`` save/restore path.

SKIPPED WHERE TORCH IS ABSENT, WHICH INCLUDES THE DEVELOPMENT WORKSTATION.
``torch`` is not installed in the project venv on this host, so every test in
this module skips locally and only executes inside a Cannon allocation. A skip
is UNMEASURED. It is never a pass, and a green local run says nothing whatever
about the native surfaces.

What this arm adds over ARM T: ARM T drives production's selection and validity
code but writes its payload without ``torch``. Here the payload writes, the
manifest, the rename, the catalog sequence AND the whole of
``restore_checkpoint`` are production's own. The two apply seams under test are
``restore.py:207`` (``_load_sampler``) and ``restore.py:208``
(``apply_rng_state``).

What this arm may NOT claim is in
``tests.helpers.chain_resume_spike.native_probe``'s module docstring, and it is
repeated in the recipe README: the domain objects are stand-ins, factor-rich
method and callback replay is HI L5a's (e2e512eb), and the mutation arms use a
MONKEYPATCH, which is not a production seam.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.chain_resume_spike.native_probe import (
    NATIVE_CHANNELS,
    NATIVE_SEAM_CHANNELS,
    first_native_divergence,
    torch_is_available,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not torch_is_available(),
        reason=(
            "ARM N requires torch, which is absent from this environment. This is "
            "UNMEASURED here, not passed; it executes only inside a Cannon allocation."
        ),
    ),
]

#: Steps the uninterrupted native arm runs.
TOTAL_STEPS = 4
#: Step at which it commits the generation the resumed arm restores.
CHECKPOINT_AT = 2

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_native_attempt(
    workspace: Path,
    *,
    steps: int,
    checkpoint_at: int | None = None,
    restore_from: Path | None = None,
    disabled_seams: tuple[str, ...] = (),
) -> dict:
    """Run one native attempt in a FRESH OS PROCESS and return its evidence.

    The native arm is held to the same rule as the torch-free one: no
    in-process resume. Reseeding globals inside a surviving interpreter cannot
    perturb module state or caches, so only a new process starts from nothing.
    ``sys.executable`` is absolute; a bare interpreter name is forbidden here by
    ``tests/unit/chain_resume_spike/test_spawn_uses_absolute_interpreter.py``.
    """

    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / "evidence.json"
    argv = [
        sys.executable,
        "-m",
        "tests.helpers.chain_resume_spike.native_probe",
        "--workspace",
        str(workspace),
        "--steps",
        str(steps),
        "--output",
        str(output),
    ]
    if checkpoint_at is not None:
        argv += ["--checkpoint-at", str(checkpoint_at)]
    if restore_from is not None:
        argv += ["--restore-from", str(restore_from)]
    for seam in disabled_seams:
        argv += ["--disable-seam", seam]

    completed = subprocess.run(
        argv, cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=600
    )
    assert completed.returncode == 0, (
        f"native attempt failed (rc={completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def uninterrupted_native(tmp_path_factory) -> tuple[dict, Path]:
    """Run the uninterrupted native arm once; return its evidence and gen K."""

    base = tmp_path_factory.mktemp("native-uninterrupted")
    evidence = _run_native_attempt(
        base / "arm-a", steps=TOTAL_STEPS, checkpoint_at=CHECKPOINT_AT
    )
    assert evidence["committed"], "the native arm committed no generation"
    return evidence, Path(evidence["committed"])


def test_the_job_reports_its_own_interpreter_and_torch_version(
    uninterrupted_native,
) -> None:
    """Provenance is read IN-JOB, not inferred from the submitting shell.

    On a facility host the two routinely disagree about which interpreter is
    selected, so a result that does not carry its own runtime cannot support a
    claim about which environment produced it.
    """

    evidence, _ = uninterrupted_native
    runtime = evidence["runtime"]
    assert Path(runtime["executable"]).is_absolute()
    assert runtime["torch_version"], "the job recorded no torch version"
    assert runtime["python_version"].startswith("3.")


def test_native_resume_reproduces_the_uninterrupted_subsequent_draws(
    uninterrupted_native, tmp_path
) -> None:
    """GREEN ARM, run first. Real save, real restore, compared on later draws."""

    evidence, generation = uninterrupted_native
    resumed = _run_native_attempt(
        tmp_path / "arm-b", steps=TOTAL_STEPS - CHECKPOINT_AT, restore_from=generation
    )

    assert resumed["pid"] != evidence["pid"], "the native resume was not a fresh process"
    assert resumed["process_seed"] != evidence["process_seed"], (
        "the resumed native process drew the same seed as its parent, so parity "
        "cannot distinguish a restore from a reinitialization"
    )
    assert resumed["restored_from"]["next_iteration"] == CHECKPOINT_AT

    divergence = first_native_divergence(evidence["trace"][CHECKPOINT_AT:], resumed["trace"])
    assert divergence is None, (
        f"native restore did not reproduce the uninterrupted stream at {divergence}"
    )


@pytest.mark.parametrize("seam", sorted(NATIVE_SEAM_CHANNELS), ids=lambda seam: seam)
def test_disabling_one_native_restore_seam_diverges_in_its_own_channel(
    uninterrupted_native, tmp_path, seam: str
) -> None:
    """One production apply seam skipped per arm, with channel attribution.

    Reproduces the DS-A0 defect deliberately: DS-A0 saved RNG state and never
    applied it, and its parity assertion could not see that. Here, skipping
    ``apply_rng_state`` must show up in the process-global channels and
    skipping ``_load_sampler`` in the sampler channel -- redness alone would
    not distinguish them.
    """

    evidence, generation = uninterrupted_native
    resumed = _run_native_attempt(
        tmp_path / f"arm-{seam}",
        steps=TOTAL_STEPS - CHECKPOINT_AT,
        restore_from=generation,
        disabled_seams=(seam,),
    )

    divergence = first_native_divergence(evidence["trace"][CHECKPOINT_AT:], resumed["trace"])
    assert divergence is not None, (
        f"skipping the real {seam} did not perturb the continuation, so the native "
        "parity gate is blind to that seam"
    )
    _, channel = divergence
    assert channel in NATIVE_SEAM_CHANNELS[seam], (
        f"skipping {seam} diverged in {channel!r}, which it does not feed "
        f"(it feeds {sorted(NATIVE_SEAM_CHANNELS[seam])})"
    )


def test_the_native_channel_map_is_disjoint_and_total() -> None:
    """Precondition for the attribution above, asserted rather than assumed."""

    seen: set[str] = set()
    for seam, channels in NATIVE_SEAM_CHANNELS.items():
        assert channels, f"{seam} feeds no channel, so its mutation arm is vacuous"
        assert not (seen & channels), f"{seam} shares a channel with another seam"
        seen |= channels
    assert seen == set(NATIVE_CHANNELS)
