"""Focused round-one review probes for the optimizer-state callback."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tpen.callback import OptimizerState
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.logging import CSV, JSONL
from tpen.physics.potential import HarmonicTrap
from tpen.training.state import TrainerState
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ModelParameterBinding,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
)
from tests.unit.callback.support import RecordingContext, deliver_completed_iteration
from tests.unit.callback.test_optimizer_state import LoggerContext
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext


class TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(2, 2, bias=False)
        self.right = nn.Linear(2, 2, bias=False)
        self.right.weight = self.left.weight
        self.extra = nn.Parameter(torch.tensor([0.25, -0.5]))
        with torch.no_grad():
            self.left.weight.copy_(torch.tensor([[0.1, -0.2], [0.3, -0.4]]))


def _bound_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> TrainerState:
    trainer = VMCTrainer(max_steps=1)
    method = trainer.bind_update_method(model=model, optimizer=optimizer)
    assert isinstance(method, LegacyAutogradUpdate)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))
    return TrainerState(
        model=model,
        optimizer=optimizer,
        update_state=VMCUpdateState(optimizer=optimizer, model_parameters=binding),
        trainer=trainer,
    )


def _optimizer_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()


def _expected(values: list[torch.Tensor]) -> tuple[float, float, float]:
    flat = torch.cat([value.detach().reshape(-1) for value in values])
    return (
        float(flat.norm().item()),
        float(flat.square().mean().sqrt().item()),
        float(flat.abs().max().item()),
    )


@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.AdamW])
def test_real_adam_carriers_recompute_all_moments_and_count_tied_identity_once(optimizer_type) -> None:
    """The six moment summaries and step are independently recomputable."""

    model = TiedModel().double()
    optimizer = optimizer_type(model.parameters(), lr=0.01)
    callback = OptimizerState({"tied": ["right"]})
    context = RecordingContext()
    state = _bound_state(model, optimizer)
    deliver_completed_iteration(callback, context, state, step=0)
    missing = context.latest("train/optimizer_state/tied")
    assert missing["n_parameters"] == 1
    assert missing["n_parameter_elements"] == model.left.weight.numel()
    assert missing["n_no_optimizer_state"] == 1
    for step in range(1, 4):
        _optimizer_step(model, optimizer)
        deliver_completed_iteration(callback, context, state, step=step)
    tied = context.latest("train/optimizer_state/tied")
    assert tied["n_parameters"] == 1
    assert tied["n_parameter_elements"] == model.left.weight.numel()
    assert tied["n_no_optimizer_state"] == 0
    assert tied["step"] == tied["step_min"] == tied["step_max"] == 3
    assert tied["step_mismatch"] is False
    adam_state = optimizer.state[model.left.weight]
    for name, values in (("exp_avg", [adam_state["exp_avg"]]), ("exp_avg_sq", [adam_state["exp_avg_sq"]])):
        l2, rms, max_abs = _expected(values)
        assert tied[f"{name}_l2"] == pytest.approx(l2)
        assert tied[f"{name}_rms"] == pytest.approx(rms)
        assert tied[f"{name}_max_abs"] == pytest.approx(max_abs)


def test_nonfinite_moments_remain_observable_in_emitted_metrics() -> None:
    """Invalid state must not become indistinguishable from an all-zero state."""

    model = TiedModel().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _optimizer_step(model, optimizer)
    adam_state = optimizer.state[model.left.weight]
    adam_state["exp_avg"].view(-1)[0] = float("nan")
    adam_state["exp_avg_sq"].fill_(float("inf"))
    context = RecordingContext()
    deliver_completed_iteration(OptimizerState({"tied": ["right"]}), context, _bound_state(model, optimizer), step=1)
    metrics = context.latest("train/optimizer_state/tied")
    for prefix in ("exp_avg", "exp_avg_sq"):
        assert metrics[f"{prefix}_nonfinite_count"] > 0


def test_nonfinite_moments_reach_real_csv_and_jsonl_sinks(tmp_path) -> None:
    """Invalidity counts and null reductions survive both scalar sinks."""

    model = TiedModel().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _optimizer_step(model, optimizer)
    state = optimizer.state[model.left.weight]
    state["exp_avg"].fill_(float("nan"))
    state["exp_avg_sq"].fill_(float("inf"))
    csv_path, jsonl_path = tmp_path / "metrics.csv", tmp_path / "metrics.jsonl"
    context = LoggerContext(CSV(csv_path), JSONL(jsonl_path))
    deliver_completed_iteration(OptimizerState({"tied": ["right"]}), context, _bound_state(model, optimizer), step=1)
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = {(row["namespace"], row["key"]): row["value"] for row in rows}
    assert values[("train/optimizer_state/tied", "exp_avg_nonfinite_count")] == "4"
    assert values[("train/optimizer_state/tied", "exp_avg_l2")] == ""
    with jsonl_path.open() as stream:
        record = json.loads(next(stream))
    assert record["metrics"]["exp_avg_nonfinite_count"] == 4
    assert record["metrics"]["exp_avg_l2"] is None


class _UnsupportedAdamCarrierUpdate(VMCUpdateMethod[AutogradUpdateInput]):
    """A real Adam-carrier method outside the callback's admitted authority."""

    def __init__(self, optimizer, model_parameters) -> None:
        self.optimizer, self.model_parameters, self.calls = optimizer, model_parameters, 0

    def update_state(self) -> VMCUpdateState:
        return VMCUpdateState(optimizer=self.optimizer, model_parameters=self.model_parameters)

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        self.calls += 1
        self.optimizer.zero_grad(set_to_none=True)
        update_input.objective.backward()
        self.optimizer.step()
        return VMCUpdateResult(applied=True, grad_norm=0.0)


class _OneElectronModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False).double()

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        x = batch.flatten_samples().positions[:, 0, 0, None]
        return WavefunctionOutput(logabs=self.linear(x).squeeze(-1), sign=torch.ones_like(x.squeeze(-1)))


class _OneElectronSampler:
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


class _DispatchingContext(_StubContext):
    def _dispatch_occurrence(self, occurrence, *, state=None) -> None:
        super()._dispatch_occurrence(occurrence, state=state)
        for callback in self.callbacks:
            callback.handle_occurrence(occurrence, self, state)


def _run_unsupported_with_callback(start_step: int):
    model = _OneElectronModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    method = _UnsupportedAdamCarrierUpdate(optimizer, ModelParameterBinding.from_parameters(tuple(model.parameters())))
    sampler = _OneElectronSampler()
    context = _DispatchingContext()
    context.callbacks = [OptimizerState({"all": ["linear"]}, start_step=start_step)]
    trainer = VMCTrainer(max_steps=1, update_method=method)
    return trainer, model, optimizer, sampler, context


@pytest.mark.parametrize("start_step", [0, 2])
def test_unsupported_method_is_refused_before_sampling_even_when_observation_is_delayed(start_step: int) -> None:
    """Support admission is independent of the observational metric window."""

    trainer, model, optimizer, sampler, context = _run_unsupported_with_callback(start_step)
    with pytest.raises(ValueError, match="LegacyAutogradUpdate"):
        trainer.fit(model=model, sampler=sampler, hamiltonian_terms=[HarmonicTrap(omega=0.5)], optimizer=optimizer, context=context, emit=lambda **_: None)
    assert sampler.calls == 0
    assert not optimizer.state
