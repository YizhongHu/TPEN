"""Independent T1 review probes for the optional ModuleNorms callback."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from tpen.callback import GradientStats, ModuleNorms
from tpen.parameter_groups import resolve_parameter_groups
from tests.unit.callback.support import RecordingContext, deliver_completed_iteration, training_state


class TiedLinearPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(2, 2, bias=False)
        self.right = nn.Linear(2, 2, bias=False)
        self.right.weight = self.left.weight


class UntiedNested(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(2, 2, bias=False))
        self.readout = nn.Linear(2, 1, bias=False)
        self.readout.trace_name = "final_readout"
        self.extra = nn.Parameter(torch.ones(1))


class GradientFixture(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(values))


class NonfiniteGradientModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.finite = GradientFixture([3.0, 4.0])
        self.nan = GradientFixture([1.0, -2.0])
        self.inf = GradientFixture([2.0, -3.0])
        self.zero = GradientFixture([0.0, 0.0])
        self.none = GradientFixture([7.0])


def test_tied_distinct_children_refuse_overlapping_ownership() -> None:
    """A shared Parameter must not make left/right ownership silently disjoint."""

    with pytest.raises(ValueError, match="overlapping"):
        resolve_parameter_groups(
            TiedLinearPair(), {"left": ["left"], "right": ["right"]}
        )


def test_tied_parameter_selected_by_right_is_owned_once() -> None:
    model = TiedLinearPair()
    groups = resolve_parameter_groups(model, {"right": ["right"]})
    selected = groups["right"]
    assert len(selected) == 1
    assert selected[0][1] is model.left.weight
    assert "ungrouped" not in groups
    assert sum(parameter.numel() for values in groups.values() for _, parameter in values) == model.left.weight.numel()


def test_untied_nested_trace_and_unknown_overlap_controls_conserve_identities() -> None:
    model = UntiedNested()
    groups = resolve_parameter_groups(
        model,
        {"branch": ["branch"], "readout": ["final_readout"]},
    )
    owned = [parameter for values in groups.values() for _, parameter in values]
    assert len({id(parameter) for parameter in owned}) == len(owned)
    assert len(owned) == sum(1 for _ in model.parameters())
    assert sum(parameter.numel() for parameter in owned) == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert len(groups["ungrouped"]) == 1

    with pytest.raises(ValueError, match="overlapping"):
        resolve_parameter_groups(model, {"parent": ["branch"], "child": ["branch.0"]})
    with pytest.raises(ValueError, match="unknown"):
        resolve_parameter_groups(model, {"missing": ["no.such.module"]})


def test_nonfinite_gradient_metrics_are_finite_only_and_reconcile() -> None:
    model = NonfiniteGradientModel()
    model.finite.weight.grad = torch.tensor([3.0, 4.0])
    model.nan.weight.grad = torch.tensor([float("nan"), 1.0])
    model.inf.weight.grad = torch.tensor([float("inf"), float("-inf")])
    model.zero.weight.grad = torch.tensor([0.0, -0.0])
    model.none.weight.grad = None
    state = training_state(model=model)

    module_context = RecordingContext()
    deliver_completed_iteration(
        ModuleNorms({"finite": ["finite"], "nan": ["nan"], "inf": ["inf"], "zero": ["zero"]}),
        module_context,
        state,
        step=0,
    )
    gradient_context = RecordingContext()
    deliver_completed_iteration(GradientStats(), gradient_context, state, step=0)

    finite = module_context.latest("train/module_norms/finite")
    nan = module_context.latest("train/module_norms/nan")
    inf = module_context.latest("train/module_norms/inf")
    zero = module_context.latest("train/module_norms/zero")
    ungrouped = module_context.latest("train/module_norms/ungrouped")
    assert finite["gradient_l2"] == pytest.approx(5.0)
    assert finite["gradient_rms"] == pytest.approx(math.sqrt(25.0 / 2.0))
    assert finite["gradient_max_abs"] == pytest.approx(4.0)
    assert finite["parameter_l2"] == pytest.approx(5.0)
    assert finite["parameter_rms"] == pytest.approx(math.sqrt(25.0 / 2.0))
    assert finite["parameter_max_abs"] == pytest.approx(4.0)
    assert nan["gradient_l2"] == pytest.approx(1.0)
    assert inf["gradient_l2"] == 0.0
    assert nan["gradient_nonfinite_count"] == 1
    assert inf["gradient_nonfinite_count"] == 2
    assert nan["parameter_l2"] == pytest.approx(math.sqrt(5.0))
    assert nan["parameter_rms"] == pytest.approx(math.sqrt(2.5))
    assert nan["parameter_max_abs"] == pytest.approx(2.0)
    assert inf["parameter_l2"] == pytest.approx(math.sqrt(13.0))
    assert inf["parameter_rms"] == pytest.approx(math.sqrt(6.5))
    assert inf["parameter_max_abs"] == pytest.approx(3.0)
    assert zero["gradient_l2"] == zero["gradient_rms"] == zero["gradient_max_abs"] == 0.0
    assert zero["gradient_nonfinite_count"] == 0
    assert zero["parameter_l2"] == zero["parameter_rms"] == zero["parameter_max_abs"] == 0.0
    assert ungrouped["n_no_gradient_parameters"] == 1
    assert ungrouped["gradient_l2"] == 0.0
    assert finite["n_parameter_elements"] == 2
    assert sum(module_context.latest(namespace)["gradient_l2"] ** 2 for namespace in (
        "train/module_norms/finite",
        "train/module_norms/nan",
        "train/module_norms/inf",
        "train/module_norms/zero",
        "train/module_norms/ungrouped",
    )) == pytest.approx(gradient_context.latest("checks/gradient")["global_grad_norm"] ** 2)


def test_reconstruction_preserves_durable_step_cadence() -> None:
    model = UntiedNested()
    uninterrupted_context = RecordingContext()
    uninterrupted = ModuleNorms({"branch": ["branch"]}, every_n_steps=3, start_step=1)
    for step in range(11):
        deliver_completed_iteration(uninterrupted, uninterrupted_context, training_state(model=model), step=step)

    resumed_context = RecordingContext()
    first = ModuleNorms({"branch": ["branch"]}, every_n_steps=3, start_step=1)
    for step in range(6):
        deliver_completed_iteration(first, resumed_context, training_state(model=model), step=step)
    second = ModuleNorms({"branch": ["branch"]}, every_n_steps=3, start_step=1)
    for step in range(6, 11):
        deliver_completed_iteration(second, resumed_context, training_state(model=model), step=step)

    assert [record["step"] for record in uninterrupted_context.by_namespace("train/module_norms/branch")] == [1, 4, 7, 10]
    assert [record["step"] for record in resumed_context.by_namespace("train/module_norms/branch")] == [1, 4, 7, 10]
    assert [record["step"] for record in resumed_context.by_namespace("train/module_norms/branch")] == [
        record["step"] for record in uninterrupted_context.by_namespace("train/module_norms/branch")
    ]


def test_observational_cadence_restarts_on_resume() -> None:
    """Callback-local logging cadence restarts after reconstruction."""

    model = UntiedNested()
    context = RecordingContext()
    uninterrupted = ModuleNorms({"branch": ["branch"]}, every_n_steps=1, max_calls=2)
    for step in range(6):
        deliver_completed_iteration(uninterrupted, context, training_state(model=model), step=step)
    uninterrupted_steps = [
        record["step"] for record in context.by_namespace("train/module_norms/branch")
    ]
    assert uninterrupted_steps == [0, 1]

    context = RecordingContext()
    first = ModuleNorms({"branch": ["branch"]}, every_n_steps=1, max_calls=2)
    for step in range(3):
        deliver_completed_iteration(first, context, training_state(model=model), step=step)
    reconstructed = ModuleNorms({"branch": ["branch"]}, every_n_steps=1, max_calls=2)
    for step in range(3, 6):
        deliver_completed_iteration(reconstructed, context, training_state(model=model), step=step)

    assert [record["step"] for record in context.by_namespace("train/module_norms/branch")] == [
        0, 1, 3, 4
    ]
