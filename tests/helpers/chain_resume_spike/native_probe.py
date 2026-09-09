"""ARM N: the native TPEN reference arm. Runs only where ``torch`` is installed.

ARM T drives production's *selection and validity* code (``artifact``,
``catalog``, ``reference``, ``manifest``, ``receipt``, ``hashing``), all of
which is torch-free at runtime. What it cannot drive is the part of the commit
sequence that calls ``torch.save`` and the whole of ``restore_checkpoint``. This
module drives those, against the real ``tpen.checkpoint.save.save_checkpoint``
and ``tpen.checkpoint.restore.restore_checkpoint``.

WHAT THIS ARM MAY CLAIM. That the real save/restore round trip preserves the
random streams a continued run subsequently draws from -- the exact property
spike DS-A0 asserted while its RNG was discarded. The two production apply
seams it exercises are ``restore.py:207`` (``_load_sampler``, the sampler's own
generator) and ``restore.py:208`` (``apply_rng_state``, the Python / NumPy /
Torch process globals).

WHAT THIS ARM MAY NOT CLAIM. Toy-level exact resume of a real TPEN run. The
model, optimizer, trainer and sampler here are minimal stand-ins that satisfy
the interfaces ``save_checkpoint`` and ``restore_checkpoint`` require; they are
not ``VMCTrainer``, ``MetropolisSampler`` or a real wavefunction. Factor-rich
method and callback replay belongs to HI L5a (e2e512eb) and is not covered here.

MONKEYPATCH DISCLOSURE. Production exposes no flag to skip one restore limb, so
the mutation arms patch ``tpen.checkpoint.restore.apply_rng_state`` and
``tpen.checkpoint.restore._load_sampler`` at their call sites, from test code.
**A monkeypatch is not a production seam.** It reproduces what a code omission
would do; it does not demonstrate that production has, or should have, a switch
there.
"""

from __future__ import annotations

import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Channels ARM N compares, in the order one step emits them. Deliberately a
#: different, smaller set than ARM T's: these are the streams the two real
#: production apply seams actually feed.
NATIVE_CHANNELS: tuple[str, ...] = (
    "sampler_draw",
    "torch_global_draw",
    "python_draw",
    "numpy_draw",
)

#: Which native channels each production apply seam feeds. Disjoint, for the
#: same attribution reason as ARM T's map.
NATIVE_SEAM_CHANNELS: dict[str, frozenset[str]] = {
    "_load_sampler": frozenset({"sampler_draw"}),
    "apply_rng_state": frozenset({"torch_global_draw", "python_draw", "numpy_draw"}),
}


def torch_is_available() -> bool:
    """Return whether ``torch`` can be imported in this interpreter.

    Checked via :mod:`importlib.util` rather than a ``try: import torch``, so
    asking the question does not itself pull in a heavyweight import on a host
    where the answer is no.
    """

    return importlib.util.find_spec("torch") is not None


def in_job_runtime_receipt() -> dict[str, Any]:
    """Return the interpreter and torch provenance, read INSIDE the job.

    A claim about which environment produced a result is worth nothing unless
    the result itself carries it. On a facility host the submitting shell and
    the executing job routinely disagree about which interpreter is selected,
    so this is read in-process, not inferred from the submission.
    """

    import sys

    import torch

    return {
        "executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }


class NativeSampler:
    """Minimal sampler satisfying TPEN's checkpoint sampler interface.

    Owns a real ``torch.Generator``, which is the stream
    ``restore.py:_load_sampler`` restores. Deliberately NOT
    ``tpen.sampling.metropolis.MetropolisSampler``: that class needs a
    configured atomic system and a wavefunction, none of which this arm is
    testing, and standing one up would make the arm's failures ambiguous
    between the checkpoint path and the physics setup.
    """

    def __init__(self, seed: int) -> None:
        import torch

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.walkers = torch.zeros(4, dtype=torch.float64)

    def mcmc_state_dict(self) -> dict[str, Any]:
        """Return checkpointable chain and generator state."""

        return {
            "walkers": self.walkers,
            "generator_state": self.generator.get_state(),
            "generator_device": "cpu",
        }

    def load_mcmc_state_dict(self, state: Any, *, device: Any = None) -> None:
        """Restore chain and generator state."""

        del device  # this stand-in is CPU-only; the parameter matches the real API
        self.walkers = state["walkers"]
        self.generator.set_state(state["generator_state"])

    def draw(self) -> float:
        """Take one sampler-stream draw."""

        import torch

        return float(torch.rand(1, generator=self.generator, dtype=torch.float64).item())


class NativeTrainer:
    """Minimal trainer carrying the two progress counters ``train_resume`` requires.

    ``tpen.checkpoint.payload._TRAINER_PROGRESS`` is exactly
    ``("next_iteration", "completed_updates")``, both required to be ``int``,
    so this is the whole contract for a payload-valid trainer state.
    """

    def __init__(self, next_iteration: int = 0, completed_updates: int = 0) -> None:
        self.next_iteration = int(next_iteration)
        self.completed_updates = int(completed_updates)

    def state_dict(self) -> dict[str, int]:
        """Return the durable progress counters."""

        return {
            "next_iteration": int(self.next_iteration),
            "completed_updates": int(self.completed_updates),
        }

    def load_state_dict(self, state: Any) -> None:
        """Restore the durable progress counters."""

        self.next_iteration = int(state["next_iteration"])
        self.completed_updates = int(state["completed_updates"])


@dataclass
class NativeSystem:
    """A model, optimizer, trainer and sampler wired for the native round trip."""

    model: Any
    optimizer: Any
    trainer: NativeTrainer
    sampler: NativeSampler

    @classmethod
    def fresh(cls, seed: int) -> "NativeSystem":
        """Build a system and seed every stream this arm observes.

        The model is built in ``float64`` because
        ``restore.py:_assert_model_runtime`` compares each parameter's dtype
        against ``context.metadata.dtype``, and the shared test run context
        declares ``float64``. A ``float32`` model would fail that check for a
        reason unrelated to anything this arm is testing.
        """

        import numpy as np
        import torch

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        model = torch.nn.Linear(3, 2, dtype=torch.float64)
        return cls(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            trainer=NativeTrainer(),
            sampler=NativeSampler(seed=seed + 1),
        )

    def step_once(self) -> dict[str, float]:
        """Advance one step, returning the channels it emits.

        Every one of the four channels is a genuine draw from a stream one of
        the two production apply seams is responsible for. A channel that did
        not consume entropy could not detect its seam being skipped.
        """

        import numpy as np
        import torch

        row = {
            "sampler_draw": self.sampler.draw(),
            "torch_global_draw": float(torch.rand(1, dtype=torch.float64).item()),
            "python_draw": random.random(),
            "numpy_draw": float(np.random.random()),
        }
        self.trainer.next_iteration += 1
        self.trainer.completed_updates += 1
        return row

    def run(self, steps: int) -> list[dict[str, float]]:
        """Advance ``steps`` steps, returning each step's channels."""

        return [self.step_once() for _ in range(steps)]


def first_native_divergence(
    left: list[dict[str, float]], right: list[dict[str, float]]
) -> tuple[int, str] | None:
    """Return the first ``(index, channel)`` at which two native traces differ.

    Compared in :data:`NATIVE_CHANNELS` emission order, for the same reason
    ARM T does: a mutation arm has to name the channel its own disabled seam
    feeds, and "first" only means something against emission order.
    """

    for index in range(min(len(left), len(right))):
        for channel in NATIVE_CHANNELS:
            if left[index][channel] != right[index][channel]:
                return index, channel
    if len(left) != len(right):
        return min(len(left), len(right)), "length"
    return None


def save_native_checkpoint(
    output_dir: Path, system: NativeSystem, context: Any, *, next_iteration: int
) -> Path:
    """Commit one generation through the REAL ``save_checkpoint``.

    Not a replica: this is ``tpen.checkpoint.save.save_checkpoint`` itself,
    including its ``torch.save`` payload writes, its manifest, its ``COMPLETE``
    marker, its rename, and the catalog / ``latest.json`` / receipt sequence
    after it.
    """

    from tpen.checkpoint.save import save_checkpoint

    return save_checkpoint(
        output_dir=output_dir,
        next_iteration=next_iteration,
        completed_updates=next_iteration,
        model=system.model,
        context=context,
        optimizer=system.optimizer,
        trainer=system.trainer,
        sampler=system.sampler,
        save_optimizer=True,
        save_trainer=True,
        save_sampler=True,
        save_rng=True,
    )


def restore_native_checkpoint(
    checkpoint_dir: Path, system: NativeSystem, context: Any
) -> Any:
    """Restore through the REAL ``restore_checkpoint`` in ``train_resume`` mode."""

    from tpen.checkpoint.restore import restore_checkpoint

    return restore_checkpoint(
        load={"mode": "train_resume", "path": str(checkpoint_dir)},
        model=system.model,
        context=context,
        optimizer=system.optimizer,
        trainer=system.trainer,
        sampler=system.sampler,
        mode="train_resume",
    )


__all__ = [
    "NATIVE_CHANNELS",
    "NATIVE_SEAM_CHANNELS",
    "NativeSampler",
    "NativeSystem",
    "NativeTrainer",
    "first_native_divergence",
    "in_job_runtime_receipt",
    "restore_native_checkpoint",
    "save_native_checkpoint",
    "torch_is_available",
]


# ----------------------------------------------------------------------
# Fresh-process entrypoint. ARM N obeys the same rule as ARM T.
# ----------------------------------------------------------------------
#
# Run as ``python -m tests.helpers.chain_resume_spike.native_probe``. The
# no-in-process-resume rule is not relaxed for the native arm: reseeding the
# globals in a surviving interpreter perturbs the four streams this arm
# compares, but it cannot perturb module-level state, caches, or any stream
# nobody thought to reseed. Only a new process starts from nothing.

#: Restore seams a mutation arm may skip, by the name they are looked up under
#: in ``tpen.checkpoint.restore``'s own namespace.
DISABLEABLE_SEAMS = ("apply_rng_state", "_load_sampler")


def _disable_seams(seam_names: tuple[str, ...]) -> None:
    """Replace named restore seams with no-ops, in ``restore``'s namespace.

    MONKEYPATCH, NOT A PRODUCTION SEAM. ``restore_checkpoint`` offers no flag
    to skip one limb, so a mutation arm reproduces what a code omission does by
    rebinding the name at the call site. This demonstrates that the parity gate
    is sensitive to each seam; it does NOT demonstrate that production has, or
    ought to have, a switch there.
    """

    from tpen.checkpoint import restore as restore_module

    for name in seam_names:
        if name not in DISABLEABLE_SEAMS:
            raise ValueError(f"unknown restore seam {name!r}; known: {DISABLEABLE_SEAMS}")
        # Bound with a default argument so each no-op reports which seam it
        # replaced, rather than every one of them looking alike in a traceback.
        def _noop(*args: Any, _seam: str = name, **kwargs: Any) -> None:
            del args, kwargs, _seam

        setattr(restore_module, name, _noop)


def main(argv: list[str] | None = None) -> int:
    """Run one native attempt in this process and write its JSON evidence."""

    import argparse
    import json
    import os
    import secrets
    import sys

    parser = argparse.ArgumentParser(description="One native TPEN chain-resume attempt")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--checkpoint-at", type=int, default=None)
    parser.add_argument("--restore-from", default=None)
    parser.add_argument("--disable-seam", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    from tests.helpers.run_context import make_run_context

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    context = make_run_context(workspace, run_id="native-chain-resume")

    process_seed = secrets.randbits(31)
    system = NativeSystem.fresh(process_seed)

    restored_from = None
    if args.restore_from is not None:
        _disable_seams(tuple(args.disable_seam))
        report = restore_native_checkpoint(Path(args.restore_from), system, context)
        restored_from = {
            "checkpoint_dir": report.checkpoint_dir,
            "next_iteration": report.next_iteration,
            "completed_updates": report.completed_updates,
        }

    trace: list[dict[str, float]] = []
    committed: str | None = None
    for index in range(args.steps):
        trace.append(system.step_once())
        if args.checkpoint_at is not None and index + 1 == args.checkpoint_at:
            committed = str(
                save_native_checkpoint(
                    workspace / "checkpoints",
                    system,
                    context,
                    next_iteration=system.trainer.next_iteration,
                )
            )

    payload = {
        "process_seed": process_seed,
        "pid": os.getpid(),
        "runtime": in_job_runtime_receipt(),
        "restored_from": restored_from,
        "disabled_seams": sorted(args.disable_seam),
        "committed": committed,
        "trace": trace,
    }
    output = Path(args.output)
    # Flushed and fsynced: a killed native arm must not lose its tail.
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    del sys
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
