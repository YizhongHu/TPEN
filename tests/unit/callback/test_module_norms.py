"""Tests for optional module-grouped parameter and gradient norms."""

from __future__ import annotations

import csv
import json

import pytest
import torch
from torch import nn

from tpen.callback import GradientStats, ModuleNorms
from tpen.logging import CSV, JSONL, LogRecord
from tpen.parameter_groups import resolve_parameter_groups
from tests.unit.callback.support import RecordingContext, deliver_completed_iteration, training_state


class GroupedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Linear(2, 2, bias=False)
        self.readout = nn.Linear(2, 1, bias=False)
        self.extra = nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.readout.trace_name = "final_readout"


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


def test_resolver_uses_module_paths_and_explicit_trace_names() -> None:
    model = GroupedModel()
    groups = resolve_parameter_groups(
        model,
        {"embed": ["embedding"], "readout": ["final_readout"]},
    )

    assert [name for name, _ in groups["embed"]] == ["embedding.weight"]
    assert [name for name, _ in groups["readout"]] == ["readout.weight"]
    assert [name for name, _ in groups["ungrouped"]] == ["extra"]


@pytest.mark.parametrize(
    "groups, message",
    [
        ({"missing": ["nope"]}, "unknown module"),
        ({"a": ["embedding"], "b": ["embedding"]}, "overlapping"),
    ],
)
def test_resolver_refuses_invalid_group_specs(groups, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_parameter_groups(GroupedModel(), groups)


def test_callback_logs_group_norms_and_missing_gradients() -> None:
    model = GroupedModel().double()
    model.embedding.weight.grad = torch.ones_like(model.embedding.weight)
    model.readout.weight.grad = None
    model.extra.grad = torch.tensor([2.0], dtype=torch.float64)
    context = RecordingContext()

    deliver_completed_iteration(
        ModuleNorms({"embed": ["embedding"], "readout": ["final_readout"]}),
        context,
        training_state(model=model),
        step=4,
    )

    embed = context.latest("train/module_norms/embed")
    readout = context.latest("train/module_norms/readout")
    ungrouped = context.latest("train/module_norms/ungrouped")
    assert embed["n_parameters"] == 1
    assert embed["gradient_l2"] == pytest.approx(2.0)
    assert readout["n_no_gradient_parameters"] == 1
    assert readout["gradient_l2"] == 0.0
    assert ungrouped["gradient_l2"] == pytest.approx(2.0)


def test_callback_is_optional_and_cadenced() -> None:
    model = GroupedModel()
    context = RecordingContext()
    callback = ModuleNorms({"embed": ["embedding"]}, every_n_steps=2)
    state = training_state(model=model)
    deliver_completed_iteration(callback, context, state, step=1)
    deliver_completed_iteration(callback, context, state, step=2)
    assert len(context.by_namespace("train/module_norms/embed")) == 1


def test_group_gradient_norms_reconcile_with_gradient_stats() -> None:
    model = GroupedModel().double()
    model.embedding.weight.grad = torch.arange(4.0, dtype=torch.float64).reshape(2, 2)
    model.readout.weight.grad = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    model.extra.grad = torch.tensor([5.0], dtype=torch.float64)
    state = training_state(model=model)

    module_context = RecordingContext()
    deliver_completed_iteration(
        ModuleNorms({"embed": ["embedding"], "readout": ["final_readout"]}),
        module_context,
        state,
        step=4,
    )
    gradient_context = RecordingContext()
    deliver_completed_iteration(GradientStats(), gradient_context, state, step=4)

    squared_groups = sum(
        module_context.latest(namespace)["gradient_l2"] ** 2
        for namespace in (
            "train/module_norms/embed",
            "train/module_norms/readout",
            "train/module_norms/ungrouped",
        )
    )
    global_norm = gradient_context.latest("checks/gradient")["global_grad_norm"]
    assert squared_groups == pytest.approx(global_norm**2)


def test_logger_consumers_read_cadenced_module_norm_records(tmp_path) -> None:
    model = GroupedModel().double()
    model.embedding.weight.grad = torch.ones_like(model.embedding.weight)
    model.readout.weight.grad = torch.ones_like(model.readout.weight)
    model.extra.grad = torch.ones_like(model.extra)
    csv_path = tmp_path / "metrics.csv"
    jsonl_path = tmp_path / "metrics.jsonl"
    context = LoggerContext(CSV(csv_path), JSONL(jsonl_path))
    callback = ModuleNorms(
        {"embed": ["embedding"], "readout": ["final_readout"]},
        every_n_steps=2,
    )
    state = training_state(model=model)
    for step in (1, 2, 3, 4):
        deliver_completed_iteration(callback, context, state, step=step)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert {row["step"] for row in csv_rows} == {"2", "4"}
    assert {row["step"] for row in json_rows} == {2, 4}
    assert {row["namespace"] for row in csv_rows} == {
        "train/module_norms/embed",
        "train/module_norms/readout",
        "train/module_norms/ungrouped",
    }
    assert {row["namespace"] for row in json_rows} == {
        "train/module_norms/embed",
        "train/module_norms/readout",
        "train/module_norms/ungrouped",
    }
    assert {"step", "namespace", "key", "value"} == set(csv_rows[0])
    assert "gradient_l2" in json_rows[0]["metrics"]
