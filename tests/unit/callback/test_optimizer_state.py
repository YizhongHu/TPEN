"""Tests for optional grouped Adam optimizer-state logging."""

from __future__ import annotations

import csv
import json

import pytest
import torch
from torch import nn

from tpen.callback import OptimizerState
from tpen.logging import CSV, JSONL, LogRecord
from tpen.training.state import TrainerState
from tpen.training.update import ModelParameterBinding, VMCUpdateState
from tests.unit.callback.support import RecordingContext, deliver_completed_iteration


class GroupedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Linear(2, 2, bias=False)
        self.readout = nn.Linear(2, 1, bias=False)


class LoggerContext(RecordingContext):
    """Forward callback records through the production logger interfaces."""

    def __init__(self, *loggers) -> None:
        super().__init__()
        self.loggers = loggers

    def log(self, metrics, *, step=None, namespace="run") -> None:
        record = LogRecord(step=step, namespace=namespace, metrics=dict(metrics))
        for logger in self.loggers:
            logger.log(record)
        super().log(metrics, step=step, namespace=namespace)


def _state(model: nn.Module, optimizer: torch.optim.Optimizer, trainer=None) -> TrainerState:
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))
    return TrainerState(
        model=model,
        optimizer=optimizer,
        update_state=VMCUpdateState(optimizer=optimizer, model_parameters=binding),
        trainer=trainer,
    )


def _step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()


def test_summarizes_adam_state_and_missing_state() -> None:
    model = GroupedModel().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    context = LoggerContext()
    callback = OptimizerState({"embed": ["embedding"]})
    deliver_completed_iteration(callback, context, _state(model, optimizer), step=0)
    assert context.latest("train/optimizer_state/embed")["n_no_optimizer_state"] == 1

    _step(model, optimizer)
    deliver_completed_iteration(callback, context, _state(model, optimizer), step=1)
    metrics = context.latest("train/optimizer_state/embed")
    parameter = model.embedding.weight
    state = optimizer.state[parameter]
    expected = state["exp_avg"].detach()
    expected_sq = state["exp_avg_sq"].detach()
    assert metrics["step"] == 1
    assert metrics["step_mismatch"] is False
    assert metrics["exp_avg_l2"] == pytest.approx(float(expected.norm().item()))
    assert metrics["exp_avg_sq_rms"] == pytest.approx(float(expected_sq.square().mean().sqrt().item()))


def test_foreign_state_and_unsupported_method_are_refused() -> None:
    model = GroupedModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    foreign = nn.Parameter(torch.ones(1))
    optimizer.state[foreign] = {
        "step": torch.tensor(1),
        "exp_avg": foreign,
        "exp_avg_sq": foreign,
    }
    with pytest.raises(ValueError, match="outside the live model"):
        deliver_completed_iteration(
            OptimizerState({"embed": ["embedding"]}),
            LoggerContext(),
            _state(model, optimizer),
            step=0,
        )

    class Unsupported:
        pass

    trainer = type("Trainer", (), {"_resolved_update_method": Unsupported()})()
    with pytest.raises(ValueError, match="LegacyAutogradUpdate"):
        deliver_completed_iteration(
            OptimizerState({"embed": ["embedding"]}),
            LoggerContext(),
            _state(model, optimizer, trainer),
            step=0,
        )


def test_real_csv_and_jsonl_consumers_receive_every_n_steps(tmp_path) -> None:
    model = GroupedModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    csv_path = tmp_path / "metrics.csv"
    jsonl_path = tmp_path / "metrics.jsonl"
    context = LoggerContext(CSV(csv_path), JSONL(jsonl_path))
    callback = OptimizerState({"embed": ["embedding"]}, every_n_steps=2)
    for step in range(4):
        _step(model, optimizer)
        deliver_completed_iteration(callback, context, _state(model, optimizer), step=step)

    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["step"]) for row in rows} == {0, 2}
    assert {row["namespace"] for row in rows} == {"train/optimizer_state/embed"}
    with jsonl_path.open() as stream:
        records = [json.loads(line) for line in stream]
    assert {record["step"] for record in records} == {0, 2}
    assert all(record["namespace"] == "train/optimizer_state/embed" for record in records)
