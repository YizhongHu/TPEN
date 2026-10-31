"""Bitwise contract tests for releasing the local-energy graph."""

from __future__ import annotations

import weakref
from collections import OrderedDict, UserDict, deque, namedtuple
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from typeguard import TypeCheckError

from tpen.artifacts import RunContext
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.physics.hamiltonian import LocalEnergyResult
from tpen.physics.kinetic import KineticEnergy
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
    def __init__(self, *, nonlinear: bool = False) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
        self.nonlinear = nonlinear

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        coordinate = batch.positions[..., 0]
        if self.nonlinear:
            coordinate = coordinate.square()
        logabs = self.scale * coordinate.sum(dim=-1)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def _synthetic_local_energy(wavefunction: torch.nn.Module, batch: ElectronBatch) -> LocalEnergyResult:
    output = wavefunction(batch)
    base = output.logabs * 0 + torch.arange(batch.batch_size, dtype=batch.dtype)
    finite = torch.ones_like(base, dtype=torch.bool)
    if base.numel() > 2:
        finite[2] = False
    total = torch.where(finite, base, torch.full_like(base, float("nan")))
    return LocalEnergyResult(
        total=total,
        terms={"alpha": total * 2.0, "beta": total * 3.0},
        wavefunction_output=WavefunctionOutput(
            logabs=output.logabs,
            sign=output.sign,
            aux={"captured": output.logabs * 7.0},
        ),
        per_electron_kinetic=total[:, None].expand(-1, batch.n_electrons),
    )


class _SyntheticChunkEvaluator:
    def validate_for_generator(self, terms, wavefunction, generator):
        del terms, wavefunction, generator

    def make_context(self, wavefunction, batch):
        return wavefunction, batch

    def evaluate(self, terms, context, *, return_terms=False):
        del terms
        result = _synthetic_local_energy(*context)
        return result if return_terms else result.total


class _RecordingAdam(torch.optim.Adam):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model.parameters(), lr=0.01)
        self.gradients: list[tuple[torch.Tensor | None, ...]] = []
        self.parameters_after_step: list[tuple[torch.Tensor, ...]] = []
        self.optimizer_states: list[tuple[dict[str, object], ...]] = []

    def step(self, closure=None):
        self.gradients.append(
            tuple(None if parameter.grad is None else parameter.grad.detach().clone() for parameter in self.param_groups[0]["params"])
        )
        result = super().step(closure)
        self.parameters_after_step.append(
            tuple(parameter.detach().clone() for parameter in self.param_groups[0]["params"])
        )
        self.optimizer_states.append(
            tuple(
                {
                    key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                    for key, value in self.state[parameter].items()
                }
                for parameter in self.param_groups[0]["params"]
            )
        )
        return result


class _Records(RunContext):
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


def _freeze_aux(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _freeze_aux(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_freeze_aux(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(_freeze_aux(nested) for nested in value)
    return value


def _freeze_result(result: LocalEnergyResult) -> LocalEnergyResult:
    output = result.wavefunction_output
    return LocalEnergyResult(
        total=result.total.detach().clone(),
        terms={name: value.detach().clone() for name, value in result.terms.items()},
        wavefunction_output=(
            None
            if output is None
            else WavefunctionOutput(
                logabs=output.logabs.detach().clone(),
                sign=output.sign.detach().clone(),
                phase=None if output.phase is None else output.phase.detach().clone(),
                aux={key: _freeze_aux(value) for key, value in output.aux.items()},
            )
        ),
        per_electron_kinetic=(
            None
            if result.per_electron_kinetic is None
            else result.per_electron_kinetic.detach().clone()
        ),
    )


def _run(
    *,
    skip_detach: bool,
    handoff_mutant: str | None = None,
    loss_mutant: bool = False,
    real_kinetic: bool = False,
    chunk_size: int | None = None,
) -> dict[str, object]:
    model = _TinyWavefunction(nonlinear=real_kinetic)
    optimizer = _RecordingAdam(model)
    sampler = _FixedSampler(_batch())
    snapshots: list[LocalEnergyResult] = []
    objective_inputs: list[torch.Tensor] = []
    original_local_energy = trainer_module.local_energy
    original_objective = trainer_module.compute_vmc_objective
    original_chunked = trainer_module.evaluate_local_energy_in_chunks
    original_value_detach = trainer_module._detach_local_energy_value
    original_wavefunction_detach = trainer_module._detach_wavefunction_output

    def fake_local_energy(terms, wavefunction, batch, *, return_terms=False):
        del terms
        if real_kinetic:
            return original_local_energy(
                [KineticEnergy()], wavefunction, batch, return_terms=return_terms
            )
        result = _synthetic_local_energy(wavefunction, batch)
        return result if return_terms else result.total

    trainer_module.local_energy = fake_local_energy
    bounded_calls: list[int] = []
    if chunk_size is not None:
        def bounded(*args, **kwargs):
            assert kwargs["chunk_size"] == chunk_size
            bounded_calls.append(kwargs["chunk_size"])
            if real_kinetic:
                return original_chunked([KineticEnergy(),], *args[1:], **kwargs)
            return original_chunked(
                *args,
                evaluator=_SyntheticChunkEvaluator(),
                **kwargs,
            )

        trainer_module.evaluate_local_energy_in_chunks = bounded

    def record_objective_input(logabs, local_energy, *, nonfinite_policy):
        if loss_mutant:
            local_energy = local_energy + 1.0
        objective_inputs.append(local_energy.detach().clone())
        return original_objective(
            logabs,
            local_energy,
            nonfinite_policy=nonfinite_policy,
        )

    trainer_module.compute_vmc_objective = record_objective_input

    def value_handoff(result: Any) -> Any:
        if isinstance(result, LocalEnergyResult):
            if handoff_mutant == "drop-term":
                result.terms.pop("beta")
            returned = result if skip_detach else original_value_detach(result)
            snapshots.append(_freeze_result(returned))
            return returned
        return result if skip_detach else original_value_detach(result)

    def production_handoff(result: LocalEnergyResult) -> LocalEnergyResult:
        if handoff_mutant == "flip-sign":
            detached = original_wavefunction_detach(result)
            detached.sign = -detached.sign
            return detached
        return original_wavefunction_detach(result)

    trainer_module._detach_local_energy_value = value_handoff
    trainer_module._detach_wavefunction_output = production_handoff
    try:
        context = _Records()
        trainer = VMCTrainer(
            max_steps=3,
            return_terms=True,
            nonfinite_local_energy_policy="mask",
            local_energy_chunk_size=chunk_size,
        )
        update_inputs: list[torch.Tensor] = []
        original_bind = trainer.bind_update_method
        wrapped_method = None

        def bind_and_record(*args, **kwargs):
            nonlocal wrapped_method
            method = original_bind(*args, **kwargs)
            if method is not wrapped_method:
                original_update = method.update

                def record_update(update_input):
                    update_inputs.append(update_input.local_energy.detach().clone())
                    return original_update(update_input)

                method.update = record_update
                wrapped_method = method
            return method

        trainer.bind_update_method = bind_and_record
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
        trainer_module.compute_vmc_objective = original_objective
        trainer_module.evaluate_local_energy_in_chunks = original_chunked
        trainer_module._detach_local_energy_value = original_value_detach
        trainer_module._detach_wavefunction_output = original_wavefunction_detach

    return {
        "losses": [metrics["loss"] for namespace, metrics in context.records if namespace == "train"],
        "metrics": [metrics for namespace, metrics in context.records if namespace == "train"],
        "objective_inputs": objective_inputs,
        "update_inputs": update_inputs,
        "gradients": optimizer.gradients,
        "parameters": optimizer.parameters_after_step,
        "optimizer_states": optimizer.optimizer_states,
        "bounded_calls": bounded_calls,
        "results": snapshots,
    }


def _assert_equal(left: dict[str, object], right: dict[str, object]) -> None:
    def equal_with_mask(left_tensor: torch.Tensor, right_tensor: torch.Tensor) -> None:
        assert torch.equal(torch.isfinite(left_tensor), torch.isfinite(right_tensor))
        assert torch.equal(torch.nan_to_num(left_tensor), torch.nan_to_num(right_tensor))

    assert left["losses"] == right["losses"]
    assert left["metrics"] == right["metrics"]
    assert len(left["objective_inputs"]) == len(right["objective_inputs"]) == 3
    for left_input, right_input in zip(left["objective_inputs"], right["objective_inputs"]):
        equal_with_mask(left_input, right_input)
    assert len(left["update_inputs"]) == len(right["update_inputs"]) == 3
    for left_input, right_input in zip(left["update_inputs"], right["update_inputs"]):
        equal_with_mask(left_input, right_input)
    assert len(left["gradients"]) == len(right["gradients"]) == 3
    assert len(left["parameters"]) == len(right["parameters"]) == 3
    assert len(left["results"]) == len(right["results"]) == 3
    for left_gradients, right_gradients in zip(left["gradients"], right["gradients"]):
        for left_gradient, right_gradient in zip(left_gradients, right_gradients):
            assert left_gradient is not None and right_gradient is not None
            assert torch.equal(left_gradient, right_gradient)
    for left_parameters, right_parameters in zip(left["parameters"], right["parameters"]):
        for left_parameter, right_parameter in zip(left_parameters, right_parameters):
            assert torch.equal(left_parameter, right_parameter)
    assert len(left["optimizer_states"]) == len(right["optimizer_states"]) == 3
    for left_states, right_states in zip(left["optimizer_states"], right["optimizer_states"]):
        for left_state, right_state in zip(left_states, right_states):
            assert left_state.keys() == right_state.keys()
            for key in left_state:
                left_value = left_state[key]
                right_value = right_state[key]
                if isinstance(left_value, torch.Tensor):
                    assert isinstance(right_value, torch.Tensor)
                    assert torch.equal(left_value, right_value)
                else:
                    assert left_value == right_value
    for left_result, right_result in zip(left["results"], right["results"]):
        equal_with_mask(left_result.total, right_result.total)
        assert list(left_result.terms) == list(right_result.terms)
        for name in left_result.terms:
            equal_with_mask(left_result.terms[name], right_result.terms[name])
        assert left_result.wavefunction_output is not None
        assert right_result.wavefunction_output is not None
        assert torch.equal(left_result.wavefunction_output.logabs, right_result.wavefunction_output.logabs)
        assert torch.equal(left_result.wavefunction_output.sign, right_result.wavefunction_output.sign)
        assert (left_result.wavefunction_output.phase is None) == (
            right_result.wavefunction_output.phase is None
        )
        if left_result.wavefunction_output.phase is not None:
            assert torch.equal(
                left_result.wavefunction_output.phase,
                right_result.wavefunction_output.phase,
            )
        assert set(left_result.wavefunction_output.aux) == set(right_result.wavefunction_output.aux)
        for name in left_result.wavefunction_output.aux:
            left_aux = left_result.wavefunction_output.aux[name]
            right_aux = right_result.wavefunction_output.aux[name]
            if isinstance(left_aux, torch.Tensor):
                assert isinstance(right_aux, torch.Tensor)
                equal_with_mask(left_aux, right_aux)
            else:
                assert left_aux == right_aux
        equal_with_mask(left_result.per_electron_kinetic, right_result.per_electron_kinetic)


def test_current_and_detached_paths_are_bitwise_equivalent_over_three_updates() -> None:
    _assert_equal(_run(skip_detach=True), _run(skip_detach=False))


def test_red_omitting_a_term_fails_the_equivalence_oracle() -> None:
    current = _run(skip_detach=True)
    detached = _run(skip_detach=False, handoff_mutant="drop-term")
    with pytest.raises(AssertionError):
        _assert_equal(current, detached)


def test_red_changing_the_loss_input_fails_the_equivalence_oracle() -> None:
    current = _run(skip_detach=True)
    detached = _run(skip_detach=False, loss_mutant=True)
    with pytest.raises(AssertionError):
        _assert_equal(current, detached)


def test_red_mutating_captured_sign_at_handoff_fails_the_equivalence_oracle() -> None:
    current = _run(skip_detach=True)
    detached = _run(skip_detach=False, handoff_mutant="flip-sign")
    with pytest.raises(AssertionError):
        _assert_equal(current, detached)


def test_real_kinetic_evaluator_result_is_detached() -> None:
    model = _TinyWavefunction(nonlinear=True)
    result = KineticEnergy().local_energy(model, _batch())
    detached = trainer_module._detach_local_energy_result(result)
    assert detached.wavefunction_output is not None
    assert not detached.total.requires_grad
    assert not detached.wavefunction_output.logabs.requires_grad
    assert not detached.per_electron_kinetic.requires_grad


def test_real_kinetic_current_and_detached_paths_are_bitwise_over_three_updates() -> None:
    _assert_equal(
        _run(skip_detach=True, real_kinetic=True),
        _run(skip_detach=False, real_kinetic=True),
    )


def test_enabled_chunk_route_matches_legacy_adam_steps_and_states() -> None:
    legacy = _run(skip_detach=False, chunk_size=None)
    chunked = _run(skip_detach=False, chunk_size=2)
    _assert_equal(legacy, chunked)
    assert legacy["bounded_calls"] == []
    assert chunked["bounded_calls"] == [2, 2, 2]
    assert any(
        not torch.equal(before, after)
        for before, after in zip(legacy["parameters"][0], legacy["parameters"][1])
    )


def test_real_kinetic_chunk_route_matches_legacy_adam_steps_and_states() -> None:
    legacy = _run(skip_detach=False, real_kinetic=True, chunk_size=None)
    chunked = _run(skip_detach=False, real_kinetic=True, chunk_size=2)
    _assert_equal(legacy, chunked)
    assert legacy["bounded_calls"] == []
    assert chunked["bounded_calls"] == [2, 2, 2]
    assert any(
        not torch.equal(before, after)
        for before, after in zip(legacy["parameters"][0], legacy["parameters"][1])
    )


def test_red_adam_state_mutation_fails_route_equivalence_oracle() -> None:
    legacy = _run(skip_detach=False, chunk_size=None)
    mutant = _run(skip_detach=False, chunk_size=2)
    mutant["optimizer_states"][1][0]["exp_avg"] += 1.0
    with pytest.raises(AssertionError):
        _assert_equal(legacy, mutant)


def _run_lifecycle(
    *, tensor_fallback: bool, skip_handoff: bool, trainer_return_terms: bool = True
) -> list[bool]:
    model = _TinyWavefunction()
    optimizer = _RecordingAdam(model)
    sampler = _FixedSampler(_batch())
    original_local_energy = trainer_module.local_energy
    original_value_detach = trainer_module._detach_local_energy_value
    previous_graphs: list[weakref.ReferenceType[torch.Tensor]] = []
    released_before_next_call: list[bool] = []

    def fake_local_energy(terms, wavefunction, batch, *, return_terms=False):
        del terms, return_terms
        if previous_graphs:
            released_before_next_call.extend(reference() is None for reference in previous_graphs)
            assert all(released_before_next_call[-len(previous_graphs) :])
        output = wavefunction(batch)
        total = output.logabs * 2.0
        if tensor_fallback:
            previous_graphs.append(weakref.ref(total))
            return total
        nested = output.logabs * 11.0
        result = LocalEnergyResult(
            total=total,
            terms={"kinetic": total},
            wavefunction_output=WavefunctionOutput(
                logabs=output.logabs,
                sign=output.sign,
                aux={"nested": [{"kinetic": nested}]},
            ),
        )
        previous_graphs.extend([weakref.ref(total), weakref.ref(nested)])
        return result

    trainer_module.local_energy = fake_local_energy
    if skip_handoff:
        trainer_module._detach_local_energy_value = lambda result: result
    try:
        trainer = VMCTrainer(max_steps=2, return_terms=trainer_return_terms)
        trainer.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=[],
            optimizer=optimizer,
            context=_Records(),
            emit=lambda *args, **kwargs: None,
        )
    finally:
        trainer_module.local_energy = original_local_energy
        trainer_module._detach_local_energy_value = original_value_detach
    return released_before_next_call


@pytest.mark.parametrize("tensor_fallback", [False, True])
def test_fit_releases_graphs_before_next_local_energy_call(tensor_fallback: bool) -> None:
    released = _run_lifecycle(tensor_fallback=tensor_fallback, skip_handoff=False)
    assert released and all(released)


def test_return_terms_false_uses_tensor_fallback_and_releases_graph() -> None:
    released = _run_lifecycle(
        tensor_fallback=True,
        skip_handoff=False,
        trainer_return_terms=False,
    )
    assert released and all(released)


def test_enabled_chunk_size_selects_bounded_local_energy_route(monkeypatch) -> None:
    model = _TinyWavefunction()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sampler = _FixedSampler(_batch())
    calls: list[int | None] = []
    original = trainer_module.evaluate_local_energy_in_chunks

    def bounded(*args, **kwargs):
        calls.append(kwargs["chunk_size"])
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "evaluate_local_energy_in_chunks", bounded)
    VMCTrainer(max_steps=1, local_energy_chunk_size=2).fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=[],
        optimizer=optimizer,
        context=_Records(),
        emit=lambda *args, **kwargs: None,
    )

    assert calls == [2]


def test_none_chunk_size_keeps_legacy_local_energy_route(monkeypatch) -> None:
    model = _TinyWavefunction()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sampler = _FixedSampler(_batch())
    legacy_calls = 0
    bounded_calls = 0
    original_legacy = trainer_module.local_energy

    def legacy(*args, **kwargs):
        nonlocal legacy_calls
        legacy_calls += 1
        return original_legacy(*args, **kwargs)

    def bounded(*args, **kwargs):
        nonlocal bounded_calls
        bounded_calls += 1
        return original_legacy(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "local_energy", legacy)
    monkeypatch.setattr(trainer_module, "evaluate_local_energy_in_chunks", bounded)
    VMCTrainer(max_steps=1).fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=[],
        optimizer=optimizer,
        context=_Records(),
        emit=lambda *args, **kwargs: None,
    )

    assert legacy_calls == 1
    assert bounded_calls == 0


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "2"])
def test_invalid_local_energy_chunk_size_rejects_before_sampling(invalid) -> None:
    with pytest.raises((ValueError, TypeCheckError), match="positive integer|did not match"):
        VMCTrainer(max_steps=1, local_energy_chunk_size=invalid)


def test_fit_graph_release_integration_is_red_without_handoff() -> None:
    with pytest.raises(AssertionError):
        _run_lifecycle(tensor_fallback=False, skip_handoff=True)


def test_tensor_fallback_detach_is_red_when_the_handoff_is_disabled() -> None:
    with pytest.raises(AssertionError):
        _run_lifecycle(tensor_fallback=True, skip_handoff=True)


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


def test_admitted_nested_aux_types_and_order_are_preserved() -> None:
    source = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    result = LocalEnergyResult(
        total=source,
        wavefunction_output=WavefunctionOutput(
            logabs=source,
            sign=torch.ones_like(source),
            aux={
                "nested": {"tensor": source, "list": [None, True, 3, 2.5, "x"]},
                "tuple": (source, "last"),
            },
        ),
    )
    detached = trainer_module._detach_local_energy_result(result)
    aux = detached.wavefunction_output.aux
    assert list(aux) == ["nested", "tuple"]
    assert type(aux["nested"]) is dict
    assert list(aux["nested"]) == ["tensor", "list"]
    assert type(aux["nested"]["list"]) is list
    assert type(aux["tuple"]) is tuple
    assert torch.equal(aux["nested"]["tensor"], torch.tensor([1.0], dtype=torch.float64))
    assert aux["nested"]["list"] == [None, True, 3, 2.5, "x"]
    assert torch.equal(aux["tuple"][0], torch.tensor([1.0], dtype=torch.float64))
    assert aux["tuple"][1] == "last"
    assert not aux["nested"]["tensor"].requires_grad
    assert not aux["tuple"][0].requires_grad


@pytest.mark.parametrize(
    "invalid_aux",
    [
        deque([torch.tensor([1.0])]),
        OrderedDict([("tensor", torch.tensor([1.0]))]),
        {1: torch.tensor([1.0])},
        namedtuple("Diagnostic", ["tensor"])(torch.tensor([1.0])),
    ],
)
def test_aux_contract_rejects_unadmitted_types_with_key_path(invalid_aux: Any) -> None:
    source = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    result = LocalEnergyResult(
        total=source,
        wavefunction_output=WavefunctionOutput(
            logabs=source,
            sign=torch.ones_like(source),
            aux={"diagnostics": invalid_aux},
        ),
    )
    with pytest.raises(TypeError, match=r"aux\['diagnostics'\]"):
        trainer_module._detach_local_energy_result(result)


@pytest.mark.parametrize("invalid_aux", [
    OrderedDict([("tensor", torch.tensor([1.0]))]),
    UserDict({"tensor": torch.tensor([1.0])}),
    type("AuxDict", (dict,), {})(tensor=torch.tensor([1.0])),
    [torch.tensor([1.0])],
    (torch.tensor([1.0]),),
    torch.tensor([1.0]),
])
def test_aux_contract_rejects_invalid_root_container(invalid_aux: Any) -> None:
    source = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    result = LocalEnergyResult(
        total=source,
        wavefunction_output=WavefunctionOutput(
            logabs=source,
            sign=torch.ones_like(source),
            aux=invalid_aux,
        ),
    )
    with pytest.raises(TypeError, match=r"at aux:"):
        trainer_module._detach_local_energy_result(result)
