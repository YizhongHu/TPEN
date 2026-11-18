"""Fresh-process RuntimeEquivariance train-resume fixture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from tpen.callback import RuntimeEquivariance
from tpen.checkpoint import TrainResume, restore_checkpoint, save_checkpoint
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.permutation import select_nonidentity_permutations
from tpen.events import Occurrence
from tpen.training.events import TrainingIteration, TrainingIterationCompleted
from tpen.training.state import TrainerState
from tests.unit.callback.support import RecordingContext


class _ProbeModel(nn.Module):
    """Tiny float64 model with an optional deliberate particle-order defect."""

    def __init__(self, *, non_equivariant: bool) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
        self.non_equivariant = non_equivariant

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        if self.non_equivariant:
            logabs = batch.positions[:, 0, :].sum(dim=-1) * self.scale
        else:
            logabs = batch.positions.square().sum(dim=(-1, -2)) * self.scale
        coordinates = batch.positions[:, :, 0]
        inversions = (coordinates.unsqueeze(-1) > coordinates.unsqueeze(-2)).triu(diagonal=1).sum(
            dim=(-1, -2)
        )
        sign = torch.where(
            inversions.remainder(2) == 0,
            torch.ones_like(logabs),
            -torch.ones_like(logabs),
        )
        return WavefunctionOutput(logabs=logabs, sign=sign)


class _Trainer:
    """Minimal trainer state owner accepted by the TrainResume payload."""

    def __init__(self, *, next_iteration: int = 2, completed_updates: int = 2) -> None:
        self.next_iteration = next_iteration
        self.completed_updates = completed_updates

    def state_dict(self) -> dict[str, int]:
        return {
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.next_iteration = int(state["next_iteration"])
        self.completed_updates = int(state["completed_updates"])


class _Sampler:
    """Minimal MCMC state owner accepted by the TrainResume payload."""

    def __init__(self) -> None:
        self.marker = 7

    def mcmc_state_dict(self) -> dict[str, int]:
        return {"marker": self.marker}

    def load_mcmc_state_dict(self, state: dict[str, int], *, device: str | None = None) -> None:
        del device
        self.marker = int(state["marker"])


def _config(root: Path) -> dict[str, object]:
    """Return a stable config whose hashes are identical across child PIDs."""

    return {
        "experiment": {"name": "runtime-equivariance-recovery"},
        "run": {"root": str(root), "run_id": "fixture", "dir": str(root)},
        "runtime": {"device": "cpu", "dtype": "float64", "seed": 19},
        "model": {"_target_": "tests.helpers.runtime_equivariance_recovery._ProbeModel"},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
        "trainer": {"_target_": "tests.helpers.runtime_equivariance_recovery._Trainer"},
        "sampler": {"_target_": "tests.helpers.runtime_equivariance_recovery._Sampler"},
        "hamiltonian_terms": {"fixture": "none"},
    }


def _context(root: Path) -> RecordingContext:
    context = RecordingContext()
    context.cfg = _config(root)
    context.run_dir = root
    context.metadata = SimpleNamespace(
        device="cpu",
        dtype="float64",
        run_id="fixture",
        extra={},
        git_commit=None,
        git_branch=None,
        dirty_worktree=None,
        command=None,
    )
    return context


def _batch() -> ElectronBatch:
    return ElectronBatch(
        positions=torch.tensor(
            [[[0.1, 0.2], [0.3, 0.5], [0.7, 1.1]]], dtype=torch.float64
        )
    )


def _make_callback(root: Path) -> RuntimeEquivariance:
    """Construct the exact shipped checker pair without Hydra construction."""

    from tpen.equivariance.checks import FullModelEquivarianceChecker, TraceEquivarianceChecker

    return RuntimeEquivariance(
        every_n_steps=5,
        probability=1.0,
        fail_fast=True,
        artifact_dir=root / "equivariance-artifacts",
        checkers=[
            FullModelEquivarianceChecker(
                permutation_fraction=0.5, max_permutations=1, seed=19, atol=1.0e-6, rtol=1.0e-6
            ),
            TraceEquivarianceChecker(
                permutation_fraction=0.5,
                max_permutations=1,
                seed=19,
                compare_output=False,
                atol=1.0e-6,
                rtol=1.0e-6,
            ),
        ],
    )


def _deliver(callback: RuntimeEquivariance, context: RecordingContext, state: TrainerState, step: int) -> None:
    callback.handle_occurrence(
        Occurrence(
            event=TrainingIterationCompleted(iteration=TrainingIteration(step=step)),
            count=1,
        ),
        context,
        state,
    )


def run(mode: str, root: Path, *, failure: bool) -> dict[str, object]:
    """Run one reference, prefix, or restored successor child."""

    root.mkdir(parents=True, exist_ok=True)
    context = _context(root)
    model = _ProbeModel(non_equivariant=failure)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = _Trainer()
    sampler = _Sampler()
    restored = False
    checkpoint_root = root / "checkpoint"
    if mode == "prefix":
        save_checkpoint(
            output_dir=checkpoint_root,
            next_iteration=2,
            completed_updates=2,
            model=model,
            optimizer=optimizer,
            trainer=trainer,
            sampler=sampler,
            context=context,
            payload=TrainResume(),
        )
    elif mode == "successor":
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(checkpoint_root)},
            model=model,
            optimizer=optimizer,
            trainer=trainer,
            sampler=sampler,
            context=context,
        )
        restored = True
    callback = _make_callback(root)
    batch = _batch()
    records: list[dict[str, object]] = []
    failure_message = None
    for step in (2, 7):
        state = TrainerState(step=-1, model=model, batch=batch)
        try:
            _deliver(callback, context, state, step)
        except RuntimeError as exc:
            failure_message = str(exc)
            break
    selected = [
        list(permutation.image)
        for step in (2, 7)
        for permutation in select_nonidentity_permutations(
            n_particles=3, fraction=0.5, max_count=1, seed=19, step=step
        )
    ]
    return {
        "pid": os.getpid(),
        "mode": mode,
        "restored": restored,
        "trainer_next_iteration": trainer.next_iteration,
        "records": context.records,
        "selected": selected,
        "failure_message": failure_message,
        "artifact_paths": sorted(
            str(path.relative_to(root))
            for path in (root / "equivariance-artifacts").rglob("failure.json")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefix", "reference", "successor"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--failure", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.root, failure=args.failure), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
