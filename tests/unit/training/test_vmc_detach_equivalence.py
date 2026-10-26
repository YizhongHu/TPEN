"""Bitwise contract tests for releasing the local-energy graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.physics.hamiltonian import LocalEnergyResult
from tpen.training import trainer as trainer_module
from tpen.training.trainer import VMCTrainer


@dataclass
class _FixedWalkers:
    batch: ElectronBatch

    def make_batch(self) -> ElectronBatch:
        return self.batch


class _FixedSampler:
    def __init__(self, batch: ElectronBatch) -> None:
        self.walkers = _FixedWalkers(batch)

    def collect_samples(self, model: Any, *, device: Any = None):
        del model, device
        return self.walkers, None


class _TinyWavefunction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = self.scale * batch.positions[..., 0].sum(dim=-1)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _RecordingAdam(torch.optim.Adam):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model.parameters(), lr=0.01)
        self.gradients: list[tuple[torch.Tensor | None, ...]] = []
        self.parameters_after_step: list[tuple[torch.Tensor, ...]] = []

    def step(self, closure=None):
        self.gradients.append(
            tuple(None if parameter.grad is None else parameter.grad.detach().clone() for parameter in self.param_groups[0]["params"])
        )
        result = super().step(closure)
        self.parameters_after_step.append(
            tuple(parameter.detach().clone() for parameter in self.param_groups[0]["params"])
        )
        return result


class _Records:
    def __init__(self) -> None:
        self.metadata = type("Metadata", (), {"device": "cpu", "dtype": "float64"})()
        self.records: list[tuple[str, dict[str, object]]] = []

    def log(self, metrics, *, step=None, namespace="run") -> None:
        del step
        self.records.append((namespace, dict(metrics)))

    def scope(self, *args, **kwargs):
        del args, kwargs
        from contextlib import nullcontext

        return nullcontext()

    def emit(self, *args, **kwargs) -> None:
        del args, kwargs


def _batch() -> ElectronBatch:
    positions = torch.tensor(
        [
            [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
            [[0.5, 0.0, 0.0], [0.6, 0.0, 0.0]],
            [[0.7, 0.0, 0.0], [0.8, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    spins = torch.tensor([[1.0, -1.0]] * 4, dtype=torch.float64)
    return ElectronBatch(positions=positions, spins=spins)


def _run(*, skip_detach: bool) -> dict[str, object]:
    model = _TinyWavefunction()
    optimizer = _RecordingAdam(model)
    sampler = _FixedSampler(_batch())
    records: list[LocalEnergyResult] = []

    def fake_local_energy(terms, wavefunction, batch, *, return_terms=False):
        del terms
        output = wavefunction(batch)
        base = output.logabs * 0 + torch.arange(4, dtype=batch.dtype)
        finite = torch.tensor([True, True, False, True])
        total = torch.where(finite, base, torch.full_like(base, float("nan")))
        result = LocalEnergyResult(
            total=total,
            terms={"alpha": total * 2.0, "beta": total * 3.0},
            wavefunction_output=WavefunctionOutput(
                logabs=output.logabs,
                sign=output.sign,
                aux={"captured": output.logabs * 7.0},
            ),
            per_electron_kinetic=total[:, None].expand(-1, 2),
        )
        records.append(result)
        return result if return_terms else result.total

    original_local_energy = trainer_module.local_energy
    original_detach = trainer_module._detach_local_energy_result
    trainer_module.local_energy = fake_local_energy
    if skip_detach:
        trainer_module._detach_local_energy_result = lambda result: result
    try:
        context = _Records()
        trainer = VMCTrainer(
            max_steps=3,
            return_terms=True,
            nonfinite_local_energy_policy="mask",
        )
        trainer.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=[],
            optimizer=optimizer,
            context=context,
            emit=lambda *args, **kwargs: None,
        )
    finally:
        trainer_module.local_energy = original_local_energy
        trainer_module._detach_local_energy_result = original_detach

    return {
        "losses": [metrics["loss"] for namespace, metrics in context.records if namespace == "train"],
        "gradients": optimizer.gradients,
        "parameters": optimizer.parameters_after_step,
        "results": records,
    }


def _assert_equal(left: dict[str, object], right: dict[str, object]) -> None:
    def equal_with_mask(left_tensor: torch.Tensor, right_tensor: torch.Tensor) -> None:
        assert torch.equal(torch.isfinite(left_tensor), torch.isfinite(right_tensor))
        assert torch.equal(torch.nan_to_num(left_tensor), torch.nan_to_num(right_tensor))

    assert left["losses"] == right["losses"]
    for left_gradients, right_gradients in zip(left["gradients"], right["gradients"]):
        for left_gradient, right_gradient in zip(left_gradients, right_gradients):
            assert left_gradient is not None and right_gradient is not None
            assert torch.equal(left_gradient, right_gradient)
    for left_parameters, right_parameters in zip(left["parameters"], right["parameters"]):
        for left_parameter, right_parameter in zip(left_parameters, right_parameters):
            assert torch.equal(left_parameter, right_parameter)
    for left_result, right_result in zip(left["results"], right["results"]):
        equal_with_mask(left_result.total, right_result.total)
        assert list(left_result.terms) == list(right_result.terms)
        for name in left_result.terms:
            equal_with_mask(left_result.terms[name], right_result.terms[name])
        assert left_result.wavefunction_output is not None
        assert right_result.wavefunction_output is not None
        assert torch.equal(left_result.wavefunction_output.logabs, right_result.wavefunction_output.logabs)
        assert torch.equal(left_result.wavefunction_output.sign, right_result.wavefunction_output.sign)
        equal_with_mask(left_result.per_electron_kinetic, right_result.per_electron_kinetic)


def test_current_and_detached_paths_are_bitwise_equivalent_over_three_updates() -> None:
    _assert_equal(_run(skip_detach=True), _run(skip_detach=False))


def test_red_omitting_a_term_fails_the_equivalence_oracle() -> None:
    current = _run(skip_detach=True)
    detached = _run(skip_detach=False)
    del detached["results"][0].terms["beta"]
    with pytest.raises(AssertionError):
        _assert_equal(current, detached)


def test_red_changing_the_loss_input_fails_the_equivalence_oracle() -> None:
    current = _run(skip_detach=True)
    detached = _run(skip_detach=False)
    detached["losses"][1] = detached["losses"][1] + 1.0
    with pytest.raises(AssertionError):
        _assert_equal(current, detached)


def test_detached_result_is_structurally_graph_free() -> None:
    source = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    result = LocalEnergyResult(
        total=source + 1.0,
        terms={"alpha": source * 2.0},
        wavefunction_output=WavefunctionOutput(
            logabs=source,
            sign=torch.ones_like(source),
            aux={"captured": source * 3.0},
        ),
        per_electron_kinetic=source[:, None],
    )
    detached = trainer_module._detach_local_energy_result(result)
    assert not detached.total.requires_grad
    assert not detached.terms["alpha"].requires_grad
    assert detached.wavefunction_output is not None
    assert not detached.wavefunction_output.logabs.requires_grad
    assert not detached.wavefunction_output.aux["captured"].requires_grad
    assert not detached.per_electron_kinetic.requires_grad


def test_red_identity_detach_mutant_fails_graph_release_assertion() -> None:
    source = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    result = LocalEnergyResult(
        total=source + 1.0,
        terms={"alpha": source},
        wavefunction_output=WavefunctionOutput(logabs=source, sign=torch.ones_like(source)),
    )
    with pytest.raises(AssertionError):
        current_path = result
        assert not current_path.total.requires_grad
