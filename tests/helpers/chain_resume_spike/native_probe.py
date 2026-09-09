"""ARM N: the native TPEN reference arm. Real ``VMCTrainer``, real ``MetropolisSampler``.

Runs only where ``torch`` is installed, which excludes the development
workstation. Every test that depends on it SKIPS elsewhere, and a skip is
UNMEASURED -- never a pass.

WHY THIS ARM EXISTS, given that native bitwise resume equivalence is ALREADY
established. ``tests/integration/training/test_train_runner.py``
(``test_resume_reproduces_the_uninterrupted_run_bitwise``) already proves
``train(2N) == train(N) + resume(N)`` on real components, compared on values and
on byte-identical ``train`` metric lines. This arm does NOT re-prove that. It
adds the three dimensions that test does not cover, each of which is a property
R1/R2/R3 depend on:

1. **Every resume runs in a FRESH OS PROCESS.** That test drives all of its arms
   inside one pytest process, where a live sampler and its generator survive in
   memory. A chain link is a new process on a new allocation, which is the case
   this lane must speak to.
2. **A per-limb disable AT THE APPLY SEAM, with channel attribution.** That test
   perturbs the SAVED sampler bytes; this one skips the RESTORE call itself,
   which is what a code omission does and what spike DS-A0 actually did.
3. **Which stream is load-bearing.** Derived from source below, then measured.

WHAT THIS ARM ESTABLISHES AND WHAT IT DOES NOT. It establishes that a
fresh-process native resume reproduces the uninterrupted trajectory, and which
restore seam that depends on. It does NOT establish anything about factor-rich
method or callback replay, which is HI L5a (``e2e512eb``).

WHICH STREAM IS LOAD-BEARING, derived from the source BEFORE any run rather than
fitted to a measurement. Every draw in the sampling path passes an explicit
private generator: ``metropolis.py:182`` (initial positions) and
``metropolis.py:294`` (acceptance uniforms) both pass
``generator=self._generator``, and ``moves.py`` (39, 93, 98) takes the generator
as a parameter, its docstring stating outright that "the move owns the proposal
shape/rules; it does not own an RNG ... all Markov-chain randomness belongs to
the sampler". Nothing in the training step draws from the PROCESS-GLOBAL torch
stream. Therefore:

* skipping ``_load_sampler`` (restore.py:207) MUST move the trajectory, and
* skipping ``apply_rng_state`` (restore.py:208) should NOT, because no step
  consumes the globals it restores.

That asymmetry is a finding rather than a nuisance: ``rng.pt`` is load-bearing
for the RESTORE-REFUSAL gate (``require_restorable_rng_state``, and
``test_train_runner``'s missing-RNG arm) but not for trajectory in this
configuration. A chain backend must preserve the sampler's own state; the
process globals alone would not be enough, and are not by themselves the thing
that has to survive a job boundary.

MONKEYPATCH DISCLOSURE. Production exposes no flag to skip one restore limb, so
the mutation arms rebind ``tpen.checkpoint.restore.apply_rng_state`` and
``tpen.checkpoint.restore._load_sampler`` at their call sites, from test code.
**A monkeypatch is not a production seam.** It reproduces what a code omission
would do; it does not show production has, or should have, a switch there. Each
no-op RECORDS ITS OWN INVOCATION, so an arm cannot pass because the patch
silently never applied -- which is the failure mode that would make the "inert"
arm above green for entirely the wrong reason.

No file under ``tpen/`` is modified by this lane.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The shipped VMC smoke configuration this arm drives.
VMC_FIXTURE = _REPO_ROOT / "tests" / "integration" / "artifacts" / "training" / "vmc_smoke.yaml"

#: Total optimizer updates in both arms. Both MUST use the same value: the
#: ``train_resume`` restore verifies the manifest's ``trainer_config`` hash,
#: which covers ``max_steps``, so a source run at a different ``max_steps``
#: is refused at the hash gate before any weight is read.
VMC_MAX_STEPS = 6

#: The mid-run checkpoint the resumed arm continues from. The fixture's periodic
#: cadence counts applied updates, so widening it to this value writes
#: ``step_000003`` and ``step_000006`` rather than 2/4/6.
VMC_RESUME_STEP = 3

#: Restore seams a mutation arm may skip, named as they are looked up in
#: ``tpen.checkpoint.restore``'s own namespace.
DISABLEABLE_SEAMS = ("apply_rng_state", "_load_sampler")

#: Source-derived expectation for each seam. Stated BEFORE measurement; if a run
#: contradicts one of these, that is a finding about production, not a licence
#: to edit the expectation.
SEAM_TRAJECTORY_EXPECTATION: dict[str, str] = {
    # The operative stream: every sampling draw uses the sampler's private
    # generator, and _load_sampler restores it along with the walkers.
    "_load_sampler": "diverges",
    # Nothing in a training step draws from the process globals this restores.
    "apply_rng_state": "inert",
}

#: Invocations recorded by the no-ops installed by :func:`disable_seams`.
_SEAM_INVOCATIONS: list[str] = []


def torch_is_available() -> bool:
    """Return whether ``torch`` can be imported in this interpreter.

    Checked via :mod:`importlib.util` rather than a ``try: import torch``, so
    asking the question does not pull in a heavyweight import on a host where
    the answer is no.
    """

    return importlib.util.find_spec("torch") is not None


def in_job_runtime_receipt() -> dict[str, Any]:
    """Return interpreter and torch provenance, read INSIDE the job.

    A claim about which environment produced a result is worth nothing unless
    the result carries it. On a facility host the submitting shell and the
    executing job routinely disagree about which interpreter is selected, so
    this is read in-process rather than inferred from the submission.
    """

    import torch

    return {
        "executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def disable_seams(seam_names: tuple[str, ...]) -> None:
    """Replace named restore seams with recording no-ops, in ``restore``'s namespace.

    MONKEYPATCH, NOT A PRODUCTION SEAM -- see the module docstring. Each no-op
    appends its own name to :data:`_SEAM_INVOCATIONS` when called, so the arm
    can prove the seam was genuinely reached and bypassed. Without that witness,
    a patch that never applied and a seam that genuinely does not matter are
    indistinguishable, and the ``inert`` expectation would be satisfied by the
    wrong thing.
    """

    from tpen.checkpoint import restore as restore_module

    for name in seam_names:
        if name not in DISABLEABLE_SEAMS:
            raise ValueError(f"unknown restore seam {name!r}; known: {DISABLEABLE_SEAMS}")

        def _noop(*args: Any, _seam: str = name, **kwargs: Any) -> None:
            del args, kwargs
            _SEAM_INVOCATIONS.append(_seam)

        setattr(restore_module, name, _noop)


def vmc_config(load: dict[str, str] | None = None) -> Any:
    """Return the shipped smoke config extended to a resumable run.

    Nothing about the fixture changes except ``max_steps`` and the periodic
    checkpoint cadence, so the resumed arm's restore satisfies every component
    hash the manifest carries.
    """

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(VMC_FIXTURE)
    cfg.trainer.max_steps = VMC_MAX_STEPS
    for callback in cfg.callbacks:
        if callback.get("_target_") != "tpen.callback.Checkpoint":
            continue
        if callback.get("periodic", True):
            callback.schedule.every_n = VMC_RESUME_STEP
    if load is not None:
        cfg.runner.load = load
    return cfg


def train_metric_lines(run_dir: Path) -> list[list[Any]]:
    """Return ``[step, raw JSONL line]`` for every ``train``-namespace record.

    The RAW line is kept rather than the parsed mapping, so equality is BYTE
    equality. ``tpen.logging.JSONL`` writes sorted keys and no timestamp, so two
    runs that computed the same numbers emit the same bytes. These lines are a
    function of the sampled batch at each step, which is what makes them an
    observation of SUBSEQUENT DRAWS rather than of a restored value.
    """

    lines: list[list[Any]] = []
    for line in (Path(run_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("namespace") == "train":
            lines.append([int(record["step"]), line])
    return lines


def model_digests(checkpoint_dir: Path) -> dict[str, str]:
    """Return a per-parameter content digest of a checkpoint's ``model.pt``.

    Digested rather than compared with ``torch.equal`` because these values
    cross a process boundary as JSON. Dtype and shape are folded in, so two
    tensors with identical bytes but different shapes cannot collide.
    """

    import torch

    state = torch.load(Path(checkpoint_dir) / "model.pt", map_location="cpu", weights_only=False)
    digests: dict[str, str] = {}
    for name, tensor in state.items():
        payload = (
            f"{tuple(tensor.shape)}|{tensor.dtype}|".encode("utf-8")
            + tensor.detach().cpu().contiguous().numpy().tobytes()
        )
        digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def _run_vmc_arm(
    run_root: Path,
    resume_from: Path | None,
    seams: tuple[str, ...],
    output: Path,
) -> int:
    """Run one native VMC arm in THIS process and write its evidence as JSON."""

    from tpen.run import run_from_config

    run_root.mkdir(parents=True, exist_ok=True)
    load = (
        None
        if resume_from is None
        else {"mode": "train_resume", "path": str(resume_from)}
    )
    if seams:
        # Installed BEFORE the run, so the restore inside it takes the patched
        # path. Patching afterwards would be a no-op that still looked applied.
        disable_seams(seams)

    cfg = vmc_config(load=load)
    cfg.run.root = str(run_root)
    exit_code = run_from_config(cfg, config_path=str(VMC_FIXTURE), command="chain-resume-spike")
    if exit_code != 0:
        raise RuntimeError(f"run_from_config returned {exit_code}")

    run_dirs = sorted(run_root.glob("vmc_smoke/*/*"))
    if len(run_dirs) != 1:
        raise RuntimeError(f"expected exactly one run dir under {run_root}, found {run_dirs}")
    run_dir = run_dirs[0]

    final_checkpoint = run_dir / "checkpoints" / f"step_{VMC_MAX_STEPS:06d}"
    payload = {
        "run_dir": str(run_dir),
        "pid": os.getpid(),
        "runtime": in_job_runtime_receipt(),
        "resumed_from": None if resume_from is None else str(resume_from),
        "seams_requested": sorted(seams),
        # OBSERVED, not echoed: each no-op records itself when actually called.
        "seams_invoked": sorted(_SEAM_INVOCATIONS),
        "train_metric_lines": train_metric_lines(run_dir),
        "model_digests": model_digests(final_checkpoint),
        "trainer_state": json.loads(
            (final_checkpoint / "trainer.json").read_text(encoding="utf-8")
        ),
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Run as ``python -m tests.helpers.chain_resume_spike.native_probe``.

    A fresh process per arm is not decoration: an in-process resume leaves the
    parent's live sampler and generator in memory, so a parity comparison can
    pass without anything having been restored.
    """

    import argparse

    parser = argparse.ArgumentParser(description="One native TPEN VMC chain-resume arm")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--disable-seam", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    return _run_vmc_arm(
        run_root=Path(args.run_root),
        resume_from=None if args.resume_from is None else Path(args.resume_from),
        seams=tuple(args.disable_seam),
        output=Path(args.output),
    )


if __name__ == "__main__":
    raise SystemExit(main())
