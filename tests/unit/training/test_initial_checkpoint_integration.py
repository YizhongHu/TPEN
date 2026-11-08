"""End-to-end `InitialCheckpoint` claims that need a real trainer loop.

Unlike `tests/unit/callback/test_initial_checkpoint.py`, which drives the
callback directly with hand-built occurrences, these tests run
`tpen.training.trainer.VMCTrainer.fit` over the real tiny Hooke-pair model and
sampler from `tests.helpers.hooke_models`, to pin:

- the checkpoint is written strictly before the first sample is drawn, and its
  model state matches a SEPARATELY constructed model at the same seed (not
  merely "whatever the running model happens to hold"), and
- resuming from the written checkpoint into fresh objects reproduces the first
  optimizer update bitwise, compared against an uninterrupted run.

Expected values are computed independently (a separate construction with the
same seed, never read back from the callback's own write), per the mutation
contract on this slice.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import torch
from omegaconf import OmegaConf

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.callback.initial_checkpoint import InitialCheckpoint
from tpen.checkpoint import checkpoint_step_dir_name
from tpen.checkpoint.rng import apply_rng_state, require_restorable_rng_state
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, HarmonicTrap
from tpen.training.trainer import VMCTrainer
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn

SEED = 20261108


def _terms():
    return [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]


def _no_emit(name: str, *, state=None, payload=None, step=None) -> None:
    del name, state, payload, step


def _context(tmp_path: Path, tag: str, *, callbacks=None) -> RunContext:
    artifact_manager = ArtifactManager(
        tmp_path / f"run-artifacts-{tag}",
        experiment="initial-checkpoint-integration",
        sector="unit",
        run_id=f"initial-checkpoint-integration-{tag}",
        layout="flat",
    )
    artifact_manager.make_dirs()
    cfg = OmegaConf.create(
        {
            "model": {"tag": "tiny-pair"},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.01},
            "trainer": {"max_steps": 1},
            "sampler": {"tag": "tiny-pair"},
            "hamiltonian_terms": {"kinetic": {}, "harmonic_trap": {}, "ee": {}},
        }
    )
    return RunContext(
        cfg=cfg,
        source_cfg=OmegaConf.create({}),
        artifact_manager=artifact_manager,
        metadata=RunMetadata(
            run_id=f"initial-checkpoint-integration-{tag}",
            run_name=f"initial-checkpoint-integration-{tag}",
            timestamp="2026-11-08T12:00:00+00:00",
            timezone="UTC",
            git_commit="test-sha",
            git_branch="test-branch",
            dirty_worktree=False,
            command="pytest",
            config_path="test.yaml",
            resolved_config_path=str(artifact_manager.path("resolved_config.yaml")),
            run_dir=str(artifact_manager.run_dir),
            device="cpu",
            dtype="float64",
        ),
        clock=RunClock(timezone="UTC", tzinfo=UTC),
        callbacks=list(callbacks or ()),
    )


def test_checkpoint_model_matches_a_separately_constructed_model_same_seed(
    tmp_path: Path,
) -> None:
    """The written model state equals a model built fresh with the same seed.

    Not merely "equals the running model's state after fit()": a separate
    process constructing a model with the same seed, never touched by this
    run, must see the identical bytes.
    """

    output_dir = tmp_path / "checkpoints"
    context = _context(tmp_path, "write", callbacks=[InitialCheckpoint(output_dir)])

    torch.manual_seed(SEED)
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=_terms(),
        optimizer=optimizer,
        context=context,
        emit=_no_emit,
    )

    torch.manual_seed(SEED)
    expected_model = build_tiny_spenn()

    final_dir = output_dir / checkpoint_step_dir_name(0)
    assert (final_dir / "COMPLETE").is_file()
    written_state = torch.load(final_dir / "model.pt", weights_only=False)
    expected_state = expected_model.state_dict()
    assert set(written_state) == set(expected_state)
    for name, expected_tensor in expected_state.items():
        assert torch.equal(written_state[name], expected_tensor), name


def test_resuming_from_it_reproduces_the_first_update_bitwise(tmp_path: Path) -> None:
    """A fresh process resumed from the checkpoint matches an uninterrupted run.

    Run A fits one step from scratch (uninterrupted). Run B constructs FRESH
    model/optimizer/sampler/trainer objects, loads every component the
    checkpoint wrote (model, optimizer, sampler, RNG) and the trainer's
    progress counters, then fits the same single step. Both must land on
    bitwise-identical model parameters.
    """

    output_dir = tmp_path / "checkpoints"
    context_a = _context(tmp_path, "a", callbacks=[InitialCheckpoint(output_dir)])

    torch.manual_seed(SEED)
    model_a = build_tiny_spenn()
    sampler_a = build_tiny_sampler()
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=0.01)
    trainer_a = VMCTrainer(max_steps=1, log_every_n_steps=1)
    trainer_a.fit(
        model=model_a,
        sampler=sampler_a,
        hamiltonian_terms=_terms(),
        optimizer=optimizer_a,
        context=context_a,
        emit=_no_emit,
    )
    expected_state = {name: tensor.clone() for name, tensor in model_a.state_dict().items()}

    final_dir = output_dir / checkpoint_step_dir_name(0)

    # Fresh objects: nothing here is the same Python object as run A's, which
    # is the point -- a resumed process never reuses in-memory state.
    model_b = build_tiny_spenn()
    model_b.load_state_dict(torch.load(final_dir / "model.pt", weights_only=False))
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=0.01)
    optimizer_b.load_state_dict(torch.load(final_dir / "optimizer.pt", weights_only=False))
    sampler_b = build_tiny_sampler()
    sampler_b.load_mcmc_state_dict(torch.load(final_dir / "sampler.pt", weights_only=False))
    trainer_b = VMCTrainer(max_steps=1, log_every_n_steps=1)
    trainer_b.load_state_dict(
        json.loads((final_dir / "trainer.json").read_text(encoding="utf-8"))
    )

    rng_state = torch.load(final_dir / "rng.pt", weights_only=False)
    require_restorable_rng_state(rng_state, "cpu", final_dir)
    apply_rng_state(rng_state, "cpu")

    context_b = _context(tmp_path, "b", callbacks=[])
    trainer_b.fit(
        model=model_b,
        sampler=sampler_b,
        hamiltonian_terms=_terms(),
        optimizer=optimizer_b,
        context=context_b,
        emit=_no_emit,
    )
    resumed_state = model_b.state_dict()

    assert set(resumed_state) == set(expected_state)
    for name, expected_tensor in expected_state.items():
        assert torch.equal(resumed_state[name], expected_tensor), name
