"""Reject null-training carriers before sampling or mutable runner restore."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.physics.potential import HarmonicTrap
from tpen.runner.train import Train
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ModelParameterBinding,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
)
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext


class _Wavefunction(torch.nn.Module):
    """Two live parameters with a deterministic, nonconstant score."""

    def __init__(self) -> None:
        super().__init__()
        self.width = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.offset = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        x = batch.flatten_samples().positions[:, 0, 0]
        logabs = -self.width * x.square() + self.offset * x
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _Sampler:
    """Count actual sampling entry and return fixed, nondegenerate positions."""

    def __init__(self) -> None:
        self.calls = 0

    def collect_samples(self, model, *, device=None):
        del model, device
        self.calls += 1
        batch = ElectronBatch(
            positions=torch.tensor([[[0.2]], [[0.8]], [[1.4]]], dtype=torch.float64),
            spins=torch.ones(3, 1, dtype=torch.float64),
        )
        return SimpleNamespace(make_batch=lambda: batch, n_walkers=3), None


class _UncheckedStatefulUpdate(VMCUpdateMethod[AutogradUpdateInput]):
    """Own a real carrier and binding without constructor containment checks."""

    def __init__(self, optimizer, model_parameters) -> None:
        self.state = VMCUpdateState(optimizer=optimizer, model_parameters=model_parameters)

    def update_state(self) -> VMCUpdateState:
        return self.state

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        self.state.optimizer.zero_grad(set_to_none=True)
        update_input.objective.backward()
        self.state.optimizer.step()
        return VMCUpdateResult(applied=True, grad_norm=0.0)


def _entry(entry, model, optimizer, trainer, sampler, monkeypatch, restore_calls):
    """Enter the real fit or Train.run resume path with an armed restore spy."""

    context = _StubContext()
    if entry == "fit":
        trainer.fit(
            model=model, optimizer=optimizer, sampler=sampler,
            hamiltonian_terms=[HarmonicTrap(omega=0.5)], context=context, emit=lambda **_: None,
        )
    else:
        def restore(**kwargs):
            # A missed guard has a visible consequence before fit can refuse.
            restore_calls.append(True)
            with torch.no_grad():
                for parameter in kwargs["model"].parameters():
                    parameter.add_(0.125)
            for group in kwargs["optimizer"].param_groups:
                group["lr"] = 0.03125
            raise AssertionError("mutable restore reached before carrier refusal")

        monkeypatch.setattr("tpen.runner.train.restore_checkpoint_with_events", restore)
        Train(
            model=model, optimizer=optimizer, trainer=trainer, sampler=sampler,
            hamiltonian_terms=[], load={"mode": "train_resume"},
        ).run(context)


def _assert_refused(entry, model, optimizer, trainer, sampler, monkeypatch, message):
    """Check refusal and preserve both model tensors and learning rates."""

    before = [parameter.detach().clone() for parameter in model.parameters()]
    rates = [group["lr"] for group in optimizer.param_groups]
    restore_calls = []
    try:
        with pytest.raises(ValueError, match=message):
            _entry(entry, model, optimizer, trainer, sampler, monkeypatch, restore_calls)
    finally:
        # Assert mutation safety even when the expected refusal is absent.
        assert all(torch.equal(old, live) for old, live in zip(before, model.parameters()))
        assert [group["lr"] for group in optimizer.param_groups] == rates
        assert sampler.calls == 0
        assert restore_calls == []
    assert trainer.completed_updates == 0
    assert trainer.next_iteration == 0


@pytest.mark.parametrize("entry", ["fit", "resume"])
def test_fresh_default_adapter_refuses_foreign_carrier(entry, monkeypatch) -> None:
    """Behaviour witness defended by both constructor and trainer call sites.

    Either remaining site refuses if the other is deleted; these default-path
    tests deliberately do not claim to pin either site alone.
    """

    model, foreign = _Wavefunction(), _Wavefunction()
    assert {id(p) for p in model.parameters()}.isdisjoint(id(p) for p in foreign.parameters())
    optimizer = torch.optim.SGD(foreign.parameters(), lr=0.1)
    _assert_refused(entry, model, optimizer, VMCTrainer(max_steps=1), _Sampler(),
                    monkeypatch, "carrier contains foreign model parameters")


@pytest.mark.parametrize("entry", ["fit", "resume"])
def test_fresh_stateful_method_refuses_foreign_carrier(entry, monkeypatch) -> None:
    """A constructor without containment leaves refusal to the trainer site alone."""

    model, foreign = _Wavefunction(), _Wavefunction()
    optimizer = torch.optim.SGD(foreign.parameters(), lr=0.1)
    method = _UncheckedStatefulUpdate(
        optimizer, ModelParameterBinding(parameters=tuple(model.parameters()))
    )
    assert method.update_state().optimizer is optimizer
    assert {id(p) for p in model.parameters()}.isdisjoint(id(p) for p in foreign.parameters())
    _assert_refused(
        entry, model, optimizer, VMCTrainer(max_steps=1, update_method=method),
        _Sampler(), monkeypatch, "carrier contains foreign model parameters",
    )


@pytest.mark.parametrize("entry", ["fit", "resume"])
def test_resolved_adapter_refuses_carrier_drift(entry, monkeypatch) -> None:
    """Re-enter with the same validly resolved adapter after param_groups drift."""

    model, foreign = _Wavefunction(), _Wavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = VMCTrainer(max_steps=1)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    # The binding entry retains the default adapter after a valid resolve.
    # Re-entry must inspect current carrier contents, not cached admission.
    optimizer.param_groups[0]["params"] = list(foreign.parameters())
    _assert_refused(entry, model, optimizer, trainer, _Sampler(), monkeypatch,
                    "carrier contains foreign model parameters")


@pytest.mark.parametrize("entry", ["fit", "resume"])
@pytest.mark.parametrize("empty_shape", ["no_groups", "empty_group"])
def test_trainer_refuses_empty_carrier(entry, empty_shape, monkeypatch) -> None:
    """Empty groups and groups without parameters cannot pass containment vacuously."""

    model = _Wavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = VMCTrainer(max_steps=1)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    if empty_shape == "no_groups":
        optimizer.param_groups.clear()
    else:
        optimizer.param_groups[0]["params"].clear()
    _assert_refused(entry, model, optimizer, trainer, _Sampler(), monkeypatch,
                    "optimizer parameter carrier is empty")


@pytest.mark.parametrize("carrier", ["foreign", "empty"])
def test_direct_constructor_refuses_invalid_carrier(carrier) -> None:
    """Direct construction enforces the adapter's own model parameter domain."""

    model, foreign = _Wavefunction(), _Wavefunction()
    optimizer = torch.optim.SGD(foreign.parameters(), lr=0.1)
    if carrier == "empty":
        optimizer.param_groups[0]["params"].clear()
    before = [p.detach().clone() for p in model.parameters()]
    rates = [g["lr"] for g in optimizer.param_groups]
    message = "carrier is empty" if carrier == "empty" else "carrier contains foreign"
    with pytest.raises(ValueError, match=message):
        LegacyAutogradUpdate(
            optimizer,
            model_parameters=ModelParameterBinding(parameters=tuple(model.parameters())),
        )
    assert all(torch.equal(old, live) for old, live in zip(before, model.parameters()))
    assert [g["lr"] for g in optimizer.param_groups] == rates


def test_trainer_strict_subset_is_accepted_and_trains() -> None:
    """A lawful strict subset really steps the selected live tensor through fit."""

    model = _Wavefunction()
    optimizer = torch.optim.SGD([model.width], lr=0.1)
    trainer, sampler = VMCTrainer(max_steps=1), _Sampler()
    before = [p.detach().clone() for p in model.parameters()]
    trainer.fit(model=model, optimizer=optimizer, sampler=sampler,
                hamiltonian_terms=[HarmonicTrap(omega=0.5)],
                context=_StubContext(), emit=lambda **_: None)
    assert sampler.calls == 1
    assert trainer.completed_updates == 1
    assert trainer.next_iteration == 1
    assert not torch.equal(before[0], model.width)
    assert torch.equal(before[1], model.offset)


def test_lawful_runner_reaches_mutable_restore(monkeypatch) -> None:
    """The resume spy is armed on the real runner path and can mutate state."""

    model = _Wavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in model.parameters()]
    restore_calls = []
    with pytest.raises(AssertionError, match="mutable restore reached"):
        _entry("resume", model, optimizer, VMCTrainer(max_steps=1), _Sampler(),
               monkeypatch, restore_calls)
    assert restore_calls == [True]
    assert all(torch.equal(old + 0.125, live) for old, live in zip(before, model.parameters()))
    assert optimizer.param_groups[0]["lr"] == 0.03125


def test_containment_guard_can_refuse_equal_valued_foreign_parameters() -> None:
    """Non-vacuity: equal values do not imply parameter ownership."""

    from tpen.training.update import validate_optimizer_parameter_containment

    model, foreign = _Wavefunction(), _Wavefunction()
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), foreign.parameters()))
    optimizer = torch.optim.SGD(foreign.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="carrier contains foreign"):
        validate_optimizer_parameter_containment(
            optimizer, ModelParameterBinding(parameters=tuple(model.parameters()))
        )
