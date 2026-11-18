"""Fresh-process RuntimeEquivariance train-resume fixture."""

from __future__ import annotations

import argparse
from datetime import UTC
import json
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.callback import RuntimeEquivariance
from tpen.checkpoint import TrainResume, restore_checkpoint, save_checkpoint
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.equivariance import checks as checks_module
from tpen.physics.hamiltonian import LocalEnergyResult
from tpen.sampling import SamplerStats
from tpen.training.events import TrainingIterationCompleted
from tpen.training.trainer import VMCTrainer


_PERMUTATION_SEED = 19


class _ProbeModel(nn.Module):
    """Tiny float64 model with an optional deliberate particle-order defect."""

    def __init__(self, *, non_equivariant: bool) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
        self.non_equivariant = non_equivariant

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        from tpen.trace import ParticleTensor, trace_value

        # This real semantic trace entry makes TraceEquivarianceChecker compare
        # a particle-valued record instead of passing an empty trace.
        trace_value(
            ParticleTensor(batch.positions[:, :, 0], particle_axis=1),
            key="positions",
        )
        if self.non_equivariant:
            logabs = batch.positions[:, 0, :].sum(dim=-1) * self.scale
        else:
            logabs = batch.positions.square().sum(dim=(-1, -2)) * self.scale
        coordinates = batch.positions[:, :, 0]
        inversions = (
            coordinates.unsqueeze(-1) > coordinates.unsqueeze(-2)
        ).triu(diagonal=1).sum(dim=(-1, -2))
        sign = torch.where(
            inversions.remainder(2) == 0,
            torch.ones_like(logabs),
            -torch.ones_like(logabs),
        )
        return WavefunctionOutput(logabs=logabs, sign=sign)


class _Walkers:
    """Fixed three-particle walker batch used by the real trainer."""

    def make_batch(self) -> ElectronBatch:
        return _batch()


class _Sampler:
    """Sampler stand-in that still drives VMCTrainer's real cadence."""

    def __init__(self) -> None:
        self.marker = 7

    def collect_samples(self, model, *, device=None):
        del model, device
        return _Walkers(), SamplerStats(
            acceptance_rate=1.0,
            n_walkers=2,
            burn_in=0,
            n_steps=1,
            proposal_scale=0.0,
            geometry={"n_walkers": 2, "n_electrons": 3, "spatial_dim": 2},
            seed=19,
        )

    def mcmc_state_dict(self) -> dict[str, int]:
        return {"marker": self.marker}

    def load_mcmc_state_dict(self, state: dict[str, int], *, device=None) -> None:
        del device
        self.marker = int(state["marker"])


class _RecordingContext(RunContext):
    """Real RunContext with in-memory metric and event receipts."""

    def __init__(self, *, cfg: DictConfig, root: Path, mode: str) -> None:
        manager = ArtifactManager(
            root,
            experiment="runtime-equivariance-recovery",
            sector="fixture",
            run_id=mode,
            layout="flat",
        )
        metadata = RunMetadata(
            run_id=mode,
            run_name="runtime-equivariance-recovery",
            timestamp="2026-01-01T00:00:00+00:00",
            timezone="UTC",
            git_commit="fixture",
            git_branch="fixture",
            dirty_worktree=False,
            command="runtime-equivariance-recovery",
            config_path=None,
            resolved_config_path=str(root / "resolved_config.yaml"),
            run_dir=str(manager.run_dir),
            device="cpu",
            dtype="float64",
            extra={},
        )
        super().__init__(
            cfg=cfg,
            source_cfg=cfg,
            artifact_manager=manager,
            metadata=metadata,
            clock=RunClock("UTC", UTC),
        )
        self.records: list[dict[str, object]] = []
        self.completed_events: list[dict[str, int]] = []

    def log(self, metrics, *, step=None, namespace="run") -> None:
        self.records.append(
            {"metrics": dict(metrics), "step": step, "namespace": namespace}
        )

    def emit(self, event, *, state=None):
        if isinstance(event, TrainingIterationCompleted):
            self.completed_events.append(
                {"event_step": int(event.iteration.step), "state_step": int(state.step)}
            )
        return super().emit(event, state=state)


def _config(root: Path, mode: str) -> DictConfig:
    """Return the reconstructed callback config and stable checkpoint config."""

    return OmegaConf.create(
        {
            "experiment": {"name": "runtime-equivariance-recovery"},
            "run": {
                "root": str(root),
                "run_id": "fixture",
                "dir": str(root / "logical-run"),
            },
            "runtime": {"device": "cpu", "dtype": "float64", "seed": 19},
            "model": {
                "_target_": "tests.helpers.runtime_equivariance_recovery._ProbeModel"
            },
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
            "trainer": {
                "_target_": "tpen.training.trainer.VMCTrainer",
                "max_steps": 16,
            },
            "sampler": {
                "_target_": "tests.helpers.runtime_equivariance_recovery._Sampler"
            },
            "hamiltonian_terms": {"fixture": "none"},
            "callbacks": [
                {
                    "_target_": "tpen.callback.RuntimeEquivariance",
                    "every_n_steps": 5,
                    "probability": 1.0,
                    "fail_fast": True,
                    "artifact_dir": str(root / mode / "equivariance-artifacts"),
                    "checkers": [
                        {
                            "_target_": "tpen.equivariance.checks.FullModelEquivarianceChecker",
                            "permutation_fraction": 0.5,
                            "max_permutations": 1,
                            "seed": _PERMUTATION_SEED,
                            "atol": 1.0e-6,
                            "rtol": 1.0e-6,
                        },
                        {
                            "_target_": "tpen.equivariance.checks.TraceEquivarianceChecker",
                            "permutation_fraction": 0.5,
                            "max_permutations": 1,
                            "seed": _PERMUTATION_SEED,
                            "compare_output": False,
                            "atol": 1.0e-6,
                            "rtol": 1.0e-6,
                        },
                    ],
                }
            ],
        }
    )


def _context(root: Path, mode: str, config: DictConfig) -> _RecordingContext:
    return _RecordingContext(cfg=config, root=root, mode=mode)


def _batch() -> ElectronBatch:
    # Three particles and two fixed walkers make the selected subset
    # discriminating while keeping the CPU fixture tiny.
    return ElectronBatch(
        positions=torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.5], [0.7, 1.1]],
                [[-0.4, 0.8], [0.6, -0.2], [1.2, 0.9]],
            ],
            dtype=torch.float64,
        )
    )


def _fake_local_energy(terms, model, batch, *, return_terms=False):
    """Fixed-input evaluator used only to keep the real trainer inexpensive."""

    del terms, model, return_terms
    return LocalEnergyResult(
        total=torch.arange(1, batch.positions.shape[0] + 1, dtype=torch.float64),
    )


def _callback(config: DictConfig) -> RuntimeEquivariance:
    """Hydra-reconstruct the callback and both checker targets from config."""

    callback = instantiate(config.callbacks[0])
    if not isinstance(callback, RuntimeEquivariance):
        raise TypeError(f"unexpected callback type {type(callback).__name__}")
    return callback


def run(mode: str, root: Path, *, failure: bool) -> dict[str, object]:
    """Run one reference, prefix, or restored successor child."""

    root.mkdir(parents=True, exist_ok=True)
    mode_root = root / mode
    config = _config(root, mode)
    context = _context(root, mode, config)
    model = _ProbeModel(non_equivariant=failure)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sampler = _Sampler()
    trainer = VMCTrainer(max_steps=5 if mode == "prefix" else 16, log_every_n_steps=1)
    checkpoint_root = root / "checkpoint"
    restored = False
    failure_message = None
    selection_calls: list[dict[str, object]] = []
    original_selector = checks_module.select_nonidentity_permutations

    def record_selection(*args, **kwargs):
        selected = original_selector(*args, **kwargs)
        selection_calls.append(
            {
                "step": int(kwargs["step"]),
                "images": [list(permutation.image) for permutation in selected],
            }
        )
        return selected

    checks_module.select_nonidentity_permutations = record_selection
    from tpen.training import trainer as trainer_module

    original_local_energy = trainer_module.local_energy
    trainer_module.local_energy = _fake_local_energy
    try:
        if mode == "successor":
            # VMCTrainer owns the real update-state binding required to restore
            # its parameter layout before its fit loop resumes at cursor 5.
            trainer.resolve_update_state(model=model, optimizer=optimizer)
            restore_checkpoint(
                load={"mode": "train_resume", "path": str(checkpoint_root)},
                model=model,
                optimizer=optimizer,
                trainer=trainer,
                sampler=sampler,
                context=context,
            )
            restored = True

        context.callbacks.append(_callback(config))
        try:
            trainer.fit(
                model=model,
                sampler=sampler,
                hamiltonian_terms=[],
                optimizer=optimizer,
                context=context,
                emit=lambda *_args, **_kwargs: None,
            )
        except RuntimeError as exc:
            failure_message = str(exc)

        if mode == "prefix":
            save_checkpoint(
                output_dir=checkpoint_root,
                model=model,
                optimizer=optimizer,
                trainer=trainer,
                sampler=sampler,
                context=context,
                payload=TrainResume(),
            )
    finally:
        trainer_module.local_energy = original_local_energy
        checks_module.select_nonidentity_permutations = original_selector

    mode_root.mkdir(parents=True, exist_ok=True)
    artifact_records = []
    for path in sorted((mode_root / "equivariance-artifacts").rglob("failure.json")):
        artifact_records.append(
            {
                "path": str(path.relative_to(mode_root)),
                "step": int(json.loads(path.read_text(encoding="utf-8"))["step"]),
            }
        )
    check_records = [
        record
        for record in context.records
        if str(record["namespace"]).startswith("checks/equivariance/")
    ]
    return {
        "pid": os.getpid(),
        "mode": mode,
        "restored": restored,
        "checker_types": [
            type(checker).__name__ for checker in context.callbacks[0].checkers
        ],
        "trainer_next_iteration": trainer.next_iteration,
        "completed_events": context.completed_events,
        "records": check_records,
        "selection_calls": selection_calls,
        "failure_message": failure_message,
        "artifact_records": artifact_records,
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
