"""Trainer-level contracts for the exact block-NG update method.

These tests intentionally use the real score-bearing TPEN path.  A synthetic
score packet can prove the block solve, but cannot show that the trainer asks
for exactly that packet, records its telemetry, and restores the method state
that defines a resumed update.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from tpen.sampling.metropolis import MetropolisSampler
from tpen.training.block_ng import BlockDiagonalNaturalGradientUpdate, BlockNGPolicy
from tpen.training.trainer import VMCTrainer
from tpen.training.update import ModelParameterBinding
from tests.integration.training.test_train_runner import (
    EQUIVALENCE_MAX_STEPS,
    RESUME_STEP,
    _checkpoint_dir,
    _diverged_parameters,
    _equivalence_config,
    _run as _run_configured_training,
    _train_metric_lines,
)
from tests.unit.training.test_sr_trainer_integration import (
    _FixedSampler,
    _StubContext,
    build_connected_model,
    build_tiny_hamiltonian_terms,
)


LEARNING_RATE = 1.0e-3
DAMPING = 1.0e-2
BLOCK_NG_PRESET = Path(__file__).resolve().parents[3] / "experiments/configs/updater/block_ng.yaml"


def _method(model: Any, *, solve_dtype: torch.dtype = torch.float64) -> BlockDiagonalNaturalGradientUpdate:
    """Build block-NG with the exact SGD ownership required by the method."""

    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE)
    return BlockDiagonalNaturalGradientUpdate(
        optimizer,
        model_parameters=ModelParameterBinding(parameters=parameters),
        policy=BlockNGPolicy(
            damping=DAMPING,
            learning_rate=LEARNING_RATE,
            solve_dtype=solve_dtype,
        ),
    )


def _fit(
    *,
    model: Any,
    method: BlockDiagonalNaturalGradientUpdate,
    max_steps: int,
) -> tuple[VMCTrainer, _StubContext]:
    """Run a bounded real trainer invocation over one deterministic batch."""

    trainer = VMCTrainer(max_steps=max_steps, log_every_n_steps=1, update_method=method)
    context = _StubContext()
    trainer.fit(
        model=model,
        sampler=_FixedSampler(n_walkers=4, seed=17),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=context,
        emit=lambda **_: None,
    )
    return trainer, context


def _last_train_metrics(context: _StubContext) -> dict[str, Any]:
    """Return the last trainer record from the bounded run."""

    records = [metrics for namespace, metrics in context.records if namespace == "train"]
    assert records
    return records[-1]


def _clone_parameters(model: Any) -> tuple[torch.Tensor, ...]:
    """Copy parameters for exact post-run comparison."""

    return tuple(parameter.detach().clone() for parameter in model.parameters())


def _rng_sensitive_block_ng_config(*, load: dict[str, str] | None = None):
    """Use the persisted MCMC stream before each resumed Block-NG update.

    The trainer-only fixed sampler below is intentionally deterministic and is
    retained for method-state guards.  Resume parity instead uses the real
    checkpointed Metropolis sampler: its first post-resume draw consumes the
    restored generator, then supplies the score packet for the compared update.
    """

    config = _equivalence_config(load=load)
    # Compose the score-capable TPEN test model with the runner's existing
    # persisted MCMC/checkpoint path; no fixture invents score semantics.
    config.model._target_ = "tests.unit.training.test_sr_trainer_integration.build_connected_model"
    config.system.n_particles = 3
    config.sampler.n_electrons = 3
    config.sampler.n_up = 2
    config.sampler.n_down = 1
    preset = OmegaConf.load(BLOCK_NG_PRESET)
    config.optimizer = preset.optimizer
    config.trainer.update_method = preset.trainer.update_method
    return config


def test_block_ng_runs_through_the_trainer_and_logs_solve_dtype() -> None:
    """A bounded trainer smoke reaches a real block-NG update and its telemetry."""

    torch.manual_seed(71)
    model = build_connected_model()
    before = _clone_parameters(model)
    method = _method(model, solve_dtype=torch.float32)

    trainer, context = _fit(model=model, method=method, max_steps=1)

    assert trainer.completed_updates == 1
    assert method.last_telemetry is not None
    assert method.last_telemetry.solve_dtype == "torch.float32"
    metrics = _last_train_metrics(context)
    assert metrics["block_ng_applied"] is True
    assert metrics["block_ng_solve_dtype"] == "torch.float32"
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_block_ng_resume_after_sampler_draw_is_bitwise(tmp_path, monkeypatch) -> None:
    """A resumed MCMC draw must reproduce the uninterrupted Block-NG update.

    This is deliberately a runner/checkpoint fixture, not a fixed-batch
    trainer fixture: disabling sampler or global RNG restoration changes the
    first post-resume draw, hence its scores and the update this test compares.
    """

    uninterrupted = _run_configured_training(
        tmp_path / "uninterrupted", _rng_sensitive_block_ng_config()
    )
    source = _checkpoint_dir(uninterrupted, RESUME_STEP)
    assert (source / "COMPLETE").exists()
    # The checkpoint payload is the state handed to the checkpoint writer at
    # its sampler boundary.  Inspect it directly rather than assuming the
    # configured spin partition survived construction.
    source_sampler_state = torch.load(source / "sampler.pt", map_location="cpu", weights_only=False)
    saved_walkers = source_sampler_state["walkers"]
    assert saved_walkers is not None
    assert saved_walkers.spins is not None
    saved_spins = saved_walkers.spins.detach().clone()

    restored_spins: list[torch.Tensor] = []
    load_mcmc_state_dict = MetropolisSampler.load_mcmc_state_dict

    def record_restored_spins(self, state, *, device=None) -> None:
        load_mcmc_state_dict(self, state, device=device)
        assert self._walkers is not None
        assert self._walkers.spins is not None
        restored_spins.append(self._walkers.spins.detach().clone())

    monkeypatch.setattr(MetropolisSampler, "load_mcmc_state_dict", record_restored_spins)

    resumed = _run_configured_training(
        tmp_path / "resumed",
        _rng_sensitive_block_ng_config(load={"mode": "train_resume", "path": str(source)}),
    )
    # This is immediately after the restore call, before the resumed loop can
    # draw or otherwise repair a malformed ElectronBatch.
    assert len(restored_spins) == 1
    assert torch.equal(restored_spins[0], saved_spins)
    final_uninterrupted = _checkpoint_dir(uninterrupted, EQUIVALENCE_MAX_STEPS)
    final_resumed = _checkpoint_dir(resumed, EQUIVALENCE_MAX_STEPS)
    assert _diverged_parameters(final_uninterrupted, final_resumed) == []
    assert (final_uninterrupted / "trainer.json").read_bytes() == (
        final_resumed / "trainer.json"
    ).read_bytes()
    resumed_lines = _train_metric_lines(resumed)
    assert [step for step, _ in resumed_lines] == list(range(RESUME_STEP, EQUIVALENCE_MAX_STEPS))
    uninterrupted_tail = [
        line for step, line in _train_metric_lines(uninterrupted) if step >= RESUME_STEP
    ]
    assert [line for _, line in resumed_lines] == uninterrupted_tail


def test_block_ng_resume_is_bitwise_and_state_sensitive() -> None:
    """Resume restores parameters, method state, and float64 solve telemetry exactly.

    The negative arm is intentionally not a parameter-only comparison.  It
    mutates the saved method update count, proves the mutation landed, and
    shows that the resumed method's persistent result differs.  A separately
    mutated numerical-policy field is refused before any mixed-method update
    can run.  Thus this test would fail if method state were discarded even
    though this stateless-in-the-solver method can otherwise produce equal
    parameter tensors.
    """

    torch.manual_seed(72)
    template = build_connected_model()
    initial_model_state = deepcopy(template.state_dict())

    uninterrupted_model = build_connected_model()
    uninterrupted_model.load_state_dict(initial_model_state)
    uninterrupted_method = _method(uninterrupted_model)
    uninterrupted_trainer, uninterrupted_context = _fit(
        model=uninterrupted_model,
        method=uninterrupted_method,
        max_steps=2,
    )

    staged_model = build_connected_model()
    staged_model.load_state_dict(initial_model_state)
    staged_method = _method(staged_model)
    staged_trainer, _ = _fit(model=staged_model, method=staged_method, max_steps=1)
    checkpoint = deepcopy(staged_trainer.state_dict())
    saved_method_state = checkpoint["update_method"]
    assert saved_method_state == staged_method.method_state_dict()
    assert saved_method_state["completed_updates"] == 1
    assert saved_method_state["policy"]["solve_dtype"] == "torch.float64"

    resumed_model = build_connected_model()
    resumed_model.load_state_dict(staged_model.state_dict())
    resumed_method = _method(resumed_model)
    resumed_trainer = VMCTrainer(max_steps=2, log_every_n_steps=1, update_method=resumed_method)
    resumed_trainer.resolve_update_state(model=resumed_model, optimizer=resumed_method.optimizer)
    resumed_trainer.load_state_dict(checkpoint)
    # Direct, field-for-field evidence that state was restored.  Equal output
    # alone would be blind because the Fisher solve has no RNG consumption.
    assert resumed_method.method_state_dict() == saved_method_state
    resumed_context = _StubContext()
    resumed_trainer.fit(
        model=resumed_model,
        sampler=_FixedSampler(n_walkers=4, seed=17),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=resumed_method.optimizer,
        context=resumed_context,
        emit=lambda **_: None,
    )

    for expected, actual in zip(
        _clone_parameters(uninterrupted_model), _clone_parameters(resumed_model), strict=True
    ):
        assert torch.equal(expected, actual)
    assert resumed_trainer.state_dict() == uninterrupted_trainer.state_dict()
    assert resumed_method.last_telemetry is not None
    assert uninterrupted_method.last_telemetry is not None
    assert resumed_method.last_telemetry.solve_dtype == uninterrupted_method.last_telemetry.solve_dtype
    assert _last_train_metrics(resumed_context)["block_ng_solve_dtype"] == "torch.float64"
    assert _last_train_metrics(uninterrupted_context)["block_ng_solve_dtype"] == "torch.float64"

    wrong_counter = deepcopy(checkpoint)
    wrong_counter["update_method"]["completed_updates"] = 0
    assert wrong_counter["update_method"]["completed_updates"] != saved_method_state["completed_updates"]
    wrong_model = build_connected_model()
    wrong_model.load_state_dict(staged_model.state_dict())
    wrong_method = _method(wrong_model)
    wrong_trainer = VMCTrainer(max_steps=2, update_method=wrong_method)
    wrong_trainer.resolve_update_state(model=wrong_model, optimizer=wrong_method.optimizer)
    wrong_trainer.load_state_dict(wrong_counter)
    wrong_trainer.fit(
        model=wrong_model,
        sampler=_FixedSampler(n_walkers=4, seed=17),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=wrong_method.optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    assert wrong_method.method_state_dict()["completed_updates"] != uninterrupted_method.method_state_dict()[
        "completed_updates"
    ]

    wrong_policy = deepcopy(checkpoint)
    wrong_policy["update_method"]["policy"]["damping"] = DAMPING * 2.0
    assert wrong_policy["update_method"]["policy"]["damping"] != saved_method_state["policy"]["damping"]
    refused_model = build_connected_model()
    refused_model.load_state_dict(staged_model.state_dict())
    refused_method = _method(refused_model)
    refused_trainer = VMCTrainer(max_steps=2, update_method=refused_method)
    refused_trainer.resolve_update_state(model=refused_model, optimizer=refused_method.optimizer)
    with pytest.raises(ValueError, match="policy does not match"):
        refused_trainer.load_state_dict(wrong_policy)
