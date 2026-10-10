"""ARM N: fresh-process native resume on the REAL VMCTrainer and MetropolisSampler.

SKIPPED WHERE TORCH IS ABSENT, WHICH INCLUDES THE DEVELOPMENT WORKSTATION. Every
node here skips locally and executes only inside a Cannon allocation. **A skip is
UNMEASURED. It is never a pass, and a green local run says nothing whatever
about the native surfaces.**

WHAT THIS ARM ADDS, AND WHAT IT DELIBERATELY DOES NOT RE-PROVE.
``tests/integration/training/test_train_runner.py`` already establishes native
bitwise resume equivalence on real components, compared on values and on
byte-identical ``train`` metric lines. This module does not repeat that. It adds
the three dimensions that test does not cover:

1. **every resume is a FRESH OS PROCESS** -- that test runs its arms inside one
   pytest process, where a live sampler and generator survive in memory;
2. **a per-limb disable at the APPLY SEAM with attribution** -- that test
   perturbs the saved sampler bytes, which is a different mechanism from
   skipping the restore call, and skipping the call is what DS-A0 actually did;
3. **which stream is load-bearing**, derived from source and then measured.

THE SOURCE-DERIVED EXPECTATION, stated before any run. Every sampling draw
passes the sampler's private generator (``metropolis.py`` 188, 255 and 298;
``moves.py`` 39, 93, 98, which owns no RNG). The STEP-level claim is
production's own, not an inference from the sampler: ``update.py`` 166-185
declares, AST-resolved, that no forward reached by ``model(batch)`` is
stochastic and that the local-energy path draws nothing.

So skipping ``_load_sampler`` must move the trajectory and skipping
``apply_rng_state`` should not. **If a run contradicts either, that is a finding
about production, not a licence to edit the expectation.**

PRECISION: the global stream IS consumed at CONSTRUCTION --
``path_aggregation.py:211`` calls ``nn.init.xavier_uniform_`` bare. The claim is
"no global draw DURING A STEP", not "nothing draws from the globals". Construction
precedes restore and its weights are overwritten, so trajectory is unaffected;
but **if any future code draws from the globals after construction, the
asymmetry flips** and the inert arm below becomes wrong.

Both mutation arms assert on a **witness that the no-op was actually invoked**,
so neither can pass because a patch silently failed to apply -- which would make
the ``inert`` arm green for entirely the wrong reason.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.chain_resume_spike.native_probe import (
    DISABLEABLE_SEAMS,
    SEAM_TRAJECTORY_EXPECTATION,
    VMC_MAX_STEPS,
    VMC_RESUME_STEP,
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

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_arm(
    workspace: Path,
    *,
    resume_from: Path | None = None,
    disabled_seams: tuple[str, ...] = (),
) -> dict:
    """Run one native VMC arm in a FRESH OS PROCESS and return its evidence.

    ``sys.executable`` is absolute; a bare interpreter name is forbidden across
    this package and enforced by
    ``tests/unit/chain_resume_spike/test_spawn_uses_absolute_interpreter.py``.
    """

    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / "evidence.json"
    argv = [
        sys.executable,
        "-m",
        "tests.helpers.chain_resume_spike.native_probe",
        "--run-root",
        str(workspace / "runs"),
        "--output",
        str(output),
    ]
    if resume_from is not None:
        argv += ["--resume-from", str(resume_from)]
    for seam in disabled_seams:
        argv += ["--disable-seam", seam]

    completed = subprocess.run(
        argv, cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=1800
    )
    assert completed.returncode == 0, (
        f"native arm failed (rc={completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _tail(evidence: dict) -> list[list]:
    """Return the metric lines from the resume step onward."""

    return [row for row in evidence["train_metric_lines"] if row[0] >= VMC_RESUME_STEP]


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory) -> tuple[dict, Path]:
    """ARM A: one uninterrupted native run; returns its evidence and gen K.

    Module-scoped so every resume arm restores from the identical baseline,
    which is what makes each mutation arm a single-variable change rather than
    two runs differing in an unknown number of ways.
    """

    workspace = tmp_path_factory.mktemp("native-uninterrupted")
    evidence = _run_arm(workspace)
    source = Path(evidence["run_dir"]) / "checkpoints" / f"step_{VMC_RESUME_STEP:06d}"
    assert (source / "COMPLETE").exists(), "arm A wrote no resume source"
    return evidence, source


def test_the_job_reports_its_own_interpreter_and_torch_version(uninterrupted) -> None:
    """Provenance is read IN-JOB, not inferred from the submitting shell.

    On a facility host the two routinely disagree about which interpreter is
    selected, so a result that does not carry its own runtime cannot support a
    claim about which environment produced it.
    """

    evidence, _ = uninterrupted
    runtime = evidence["runtime"]
    assert Path(runtime["executable"]).is_absolute()
    assert runtime["torch_version"], "the job recorded no torch version"
    assert runtime["python_version"].startswith("3.")


def test_the_seam_expectation_map_covers_every_disableable_seam() -> None:
    """Precondition for attribution: every seam has a stated expected direction.

    A seam with no stated expectation could not have a mutation arm that means
    anything -- whatever it did would be consistent with the map.
    """

    assert set(SEAM_TRAJECTORY_EXPECTATION) == set(DISABLEABLE_SEAMS)
    assert set(SEAM_TRAJECTORY_EXPECTATION.values()) == {"diverges", "inert"}, (
        "the map must contain both directions, or it asserts nothing by contrast"
    )


def test_a_fresh_process_native_resume_reproduces_the_uninterrupted_trajectory(
    uninterrupted, tmp_path
) -> None:
    """GREEN ARM, run first. Real save, real restore, fresh process, later draws.

    Compared on the ``train`` metric lines the resumed arm emits, which are a
    function of the batch sampled at each step and therefore an observation of
    SUBSEQUENT DRAWS -- not of a restored value read back. The final model
    digests and the durable counters are compared as well, so a run that
    reproduced the metrics while diverging in weights could not pass.
    """

    evidence, source = uninterrupted
    resumed = _run_arm(tmp_path / "resumed", resume_from=source)

    assert resumed["pid"] != evidence["pid"], "the native resume was not a fresh process"
    assert resumed["seams_invoked"] == [], "a seam was bypassed in the green arm"

    # The resumed arm runs only the steps it actually has left.
    assert [row[0] for row in resumed["train_metric_lines"]] == list(
        range(VMC_RESUME_STEP, VMC_MAX_STEPS)
    )
    assert resumed["train_metric_lines"] == _tail(evidence), (
        "the resumed native trajectory is not byte-identical to the uninterrupted one"
    )
    assert resumed["model_digests"] == evidence["model_digests"]
    assert resumed["trainer_state"] == evidence["trainer_state"]
    assert resumed["trainer_state"]["next_iteration"] == VMC_MAX_STEPS
    assert resumed["trainer_state"]["completed_updates"] == VMC_MAX_STEPS


def test_skipping_the_sampler_restore_moves_the_native_trajectory(
    uninterrupted, tmp_path
) -> None:
    """The operative stream. Skipping restore.py:207 must change what is drawn.

    This is the DS-A0 shape reproduced on real components: the state is present
    in the checkpoint and simply never applied. If this arm did not diverge, the
    green arm above would be passing without depending on the sampler state at
    all.
    """

    evidence, source = uninterrupted
    resumed = _run_arm(
        tmp_path / "no-sampler", resume_from=source, disabled_seams=("_load_sampler",)
    )

    assert resumed["seams_invoked"] == ["_load_sampler"], (
        "the sampler seam was never reached, so this arm skipped nothing and its "
        "result says nothing about that seam"
    )
    assert SEAM_TRAJECTORY_EXPECTATION["_load_sampler"] == "diverges"
    assert resumed["train_metric_lines"] != _tail(evidence), (
        "skipping the real _load_sampler did not perturb the trajectory; the "
        "native parity gate is blind to the sampler stream"
    )


def test_skipping_the_global_rng_restore_leaves_the_native_trajectory_unchanged(
    uninterrupted, tmp_path
) -> None:
    """The inert stream, and the finding worth carrying to R1/R2/R3.

    Derived from source before measuring: every sampling draw uses the sampler's
    private generator (metropolis.py 188/255/298), and ``update.py`` 166-185
    declares no stochastic forward and no local-energy draw, so no training step
    consumes the process globals ``apply_rng_state`` restores. Construction does
    (path_aggregation.py:211, a bare ``xavier_uniform_``), but that precedes
    restore and is overwritten by the restored ``state_dict``. This arm asserts that, and it is only
    meaningful because the witness proves the seam was genuinely reached and
    bypassed -- otherwise "unchanged" would be equally consistent with a patch
    that never applied.

    CONSEQUENCE, stated as a property rather than a reassurance: ``rng.pt`` is
    load-bearing for the restore-REFUSAL gate but not for TRAJECTORY in this
    configuration. A chain backend must preserve the sampler's own state; the
    process globals alone are not what has to survive a job boundary.

    If this arm ever goes red, something in the training path has begun drawing
    from the process globals AFTER construction, and the asymmetry has flipped.
    That is a finding about production and must be reported, not silenced by
    relaxing this assertion.
    """

    evidence, source = uninterrupted
    resumed = _run_arm(
        tmp_path / "no-global-rng", resume_from=source, disabled_seams=("apply_rng_state",)
    )

    assert resumed["seams_invoked"] == ["apply_rng_state"], (
        "the global-RNG seam was never reached, so 'unchanged' below would be "
        "vacuous -- it would be measuring a patch that never applied"
    )
    assert SEAM_TRAJECTORY_EXPECTATION["apply_rng_state"] == "inert"
    assert resumed["train_metric_lines"] == _tail(evidence), (
        "skipping apply_rng_state changed the native trajectory, so a training "
        "step now consumes the process-global RNG stream. This CONTRADICTS the "
        "source-derived expectation and is a finding about production."
    )
    assert resumed["model_digests"] == evidence["model_digests"]
