"""Trainer-level contracts for the exact block-NG update method.

These tests intentionally use the real score-bearing TPEN path.  A synthetic
score packet can prove the block solve, but cannot show that the trainer asks
for exactly that packet, records its telemetry, and restores the method state
that defines a resumed update.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch

from tpen.training.block_ng import BlockDiagonalNaturalGradientUpdate, BlockNGPolicy
from tpen.training.trainer import VMCTrainer
from tpen.training.update import ModelParameterBinding
from tests.unit.training.test_sr_trainer_integration import (
    _FixedSampler,
    _StubContext,
    build_connected_model,
    build_tiny_hamiltonian_terms,
)


LEARNING_RATE = 1.0e-3
DAMPING = 1.0e-2


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
