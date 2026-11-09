"""Tests for optional module-grouped parameter and gradient norms."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tpen.callback import ModuleNorms
from tpen.parameter_groups import resolve_parameter_groups
from tests.unit.callback.support import RecordingContext, deliver_completed_iteration, training_state


class GroupedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Linear(2, 2, bias=False)
        self.readout = nn.Linear(2, 1, bias=False)
        self.extra = nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.readout.trace_name = "final_readout"


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
