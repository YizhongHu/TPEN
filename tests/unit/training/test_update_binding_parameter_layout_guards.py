"""Pins for the two parameter-binding guards on the update-method boundary.

The acceptance contract for this slice requires the trainer to "validate live
parameter/optimizer identity".  The OPTIMIZER half of that clause is pinned in
several places.  The PARAMETER half was not pinned anywhere: the independent
verifier at head 1f4f1e26 disabled both guards at once and every arm reported
identical pass counts, so a regression, inversion or outright deletion of
either one would have passed the whole suite.

Each test below drives exactly one guard, through the trainer entry point that
reaches it, and asserts the guard's own message.  A method that returns a
binding over parameters the live model does not own is the discriminating
input in both cases, because ``ModelParameterBinding.compare`` requires
parameter-reference IDENTITY and not merely a matching layout: two models of
the same architecture compare unequal.

These probes assert what the code beneath them does.  Per-round measurements,
job identifiers and review history live in the record -- PR 524's thread and
the Task Orchestrator notes on item b6b5b4f5 -- and deliberately not here.
"""

from __future__ import annotations

import pytest
import torch

from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    ModelParameterBinding,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
)
from tests.helpers.hooke_models import build_tiny_spenn

LEARNING_RATE = 0.01


class _CarrierSwappingMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """A method that accepts the rebind correctly, then swaps its carrier.

    This is the shape the post-rebind ownership check exists for: everything
    about the parameter binding is honoured, so the parameter guard below it
    cannot fire, and only the optimizer-identity check can refuse. ``swap``
    is what makes one class serve as both the witness and its own control.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        binding: ModelParameterBinding,
        *,
        swap: bool,
    ) -> None:
        self._optimizer = optimizer
        self._binding = binding
        self._swap = swap
        self.rebind_calls = 0

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_test")

    def update_state(self) -> VMCUpdateState:
        return VMCUpdateState(optimizer=self._optimizer, model_parameters=self._binding)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self.rebind_calls += 1
        self._binding = model_parameters
        if self._swap:
            # A DIFFERENT Adam over the SAME live parameters: the run would
            # publish one carrier and mutate another.
            self._optimizer = torch.optim.Adam(
                [p for p in model_parameters.parameters], lr=LEARNING_RATE
            )


class _ForeignBindingMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """A stateful method whose reported binding is not the live model's.

    ``rebind_model_parameters`` is honoured or ignored according to
    ``honour_rebind``; that switch is what lets one class drive both guards.
    The optimizer is reported unchanged throughout, so the optimizer-ownership
    checks that precede both parameter checks pass and cannot be mistaken for
    the refusal under test.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        binding: ModelParameterBinding,
        *,
        honour_rebind: bool,
    ) -> None:
        self._optimizer = optimizer
        self._binding = binding
        self._honour_rebind = honour_rebind
        self.rebind_calls = 0

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_test")

    def update_state(self) -> VMCUpdateState:
        return VMCUpdateState(optimizer=self._optimizer, model_parameters=self._binding)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self.rebind_calls += 1
        if self._honour_rebind:
            self._binding = model_parameters


def test_resolve_refuses_a_method_bound_to_another_models_parameters() -> None:
    """`_resolve_method_state`'s parameter check refuses a foreign binding.

    The method reports the run's own optimizer, so the ownership check ahead of
    this one passes and the refusal under test is the only one that can fire.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    foreign_model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    foreign_binding = ModelParameterBinding.from_parameters(tuple(foreign_model.parameters()))

    # Control: the foreign binding is unequal for the reason claimed -- the
    # parameter REFERENCES differ -- and not because the layouts differ.
    live_binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))
    assert live_binding.layout.compare(foreign_binding.layout)[0], (
        "the two models must share a layout, or this test would pass for the wrong reason"
    )
    assert not live_binding.compare(foreign_binding)[0]

    method = _ForeignBindingMethod(optimizer, foreign_binding, honour_rebind=True)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    with pytest.raises(
        ValueError, match="update method parameter binding does not match live model"
    ):
        trainer.resolve_update_state(model=model, optimizer=optimizer)


def test_rebuild_refuses_a_method_that_ignores_the_rebuilt_binding() -> None:
    """`rebuild_update_state`'s parameter check refuses a dropped rebind.

    Restore rebinds the method onto the reloaded model's parameters. A method
    that accepts the call and keeps its old references would leave the resumed
    loop stepping parameters the run no longer owns; this guard is what refuses
    it, and the assertion below is on that guard's own message.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))

    method = _ForeignBindingMethod(optimizer, binding, honour_rebind=False)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    # Resolving succeeds: at this point the reported binding IS the live one.
    trainer.resolve_update_state(model=model, optimizer=optimizer)

    # A restore replaces the model objects; the method ignores the rebind.
    torch.manual_seed(123)
    reloaded_model = build_tiny_spenn()

    with pytest.raises(
        ValueError, match="update method did not retain the rebuilt model binding"
    ):
        trainer.rebuild_update_state(model=reloaded_model)

    assert method.rebind_calls == 1, (
        "the guard must fire because the rebind was DROPPED, not because it was never attempted"
    )


def test_rebuild_refuses_a_method_that_swaps_its_carrier_during_rebind() -> None:
    """`rebuild_update_state`'s ownership check refuses a post-rebind swap.

    The method honours the parameter rebind exactly, so the parameter guard
    beneath this one cannot be what refuses: the only disagreement left is the
    optimizer identity. A run that accepted this would publish one carrier and
    mutate another across a resume.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))

    method = _CarrierSwappingMethod(optimizer, binding, swap=True)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)
    trainer.resolve_update_state(model=model, optimizer=optimizer)

    torch.manual_seed(123)
    reloaded_model = build_tiny_spenn()

    with pytest.raises(ValueError, match="mismatched legacy optimizer ownership"):
        trainer.rebuild_update_state(model=reloaded_model)

    assert method.rebind_calls == 1, (
        "the guard must fire on a SWAP, not because the rebind was never attempted"
    )
    assert method._optimizer is not optimizer, (
        "the fixture did not actually swap the carrier, so the refusal proves nothing"
    )


def test_rebuild_accepts_a_method_that_retains_its_carrier_through_rebind() -> None:
    """Control for the test above: retaining the carrier is NOT refused.

    Without this, the swap test would be satisfied by a trainer that refuses
    every rebuild, which would pin nothing about carrier identity.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))

    method = _CarrierSwappingMethod(optimizer, binding, swap=False)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)
    trainer.resolve_update_state(model=model, optimizer=optimizer)

    torch.manual_seed(123)
    reloaded_model = build_tiny_spenn()

    rebuilt = trainer.rebuild_update_state(model=reloaded_model)

    assert rebuilt.optimizer is optimizer
    assert method.rebind_calls == 1
