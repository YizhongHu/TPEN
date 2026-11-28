"""Review probes for SOAP checker evidence wiring and rendered Hooke groups."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.atomistic.soap_pilot.checks import Verdict, run_rung1_checks
from experiments.atomistic.soap_pilot.composer import R2_PROFILES, render_profile
from experiments.atomistic.soap_pilot.evidence import MetricRecord
from tpen.parameter_groups import resolve_parameter_groups


def test_run_rung1_checks_exposes_metric_finite_gate_and_empty_is_unmeasured() -> None:
    """No metric records must be visible as an explicit UNMEASURED gate."""

    results = {result.name: result for result in run_rung1_checks({})}
    assert "metric-finite" in results
    assert results["metric-finite"].verdict is Verdict.UNMEASURED


def test_terminal_checkpoint_and_iteration_steps_are_distinct_controls() -> None:
    """A step-99 iteration must not satisfy a step-100 checkpoint assertion."""

    from experiments.atomistic.soap_pilot.checks import check_disconnected_optimizer_state

    records = [
        MetricRecord(99, "train/optimizer_state/embedding", {"n_no_optimizer_state": 8}),
        MetricRecord(99, "train/optimizer_state/readout", {"n_no_optimizer_state": 8}),
    ]
    result = check_disconnected_optimizer_state(
        records, groups=("embedding", "readout"), terminal_step=100
    )
    assert result.verdict is Verdict.UNMEASURED


@pytest.mark.parametrize("profile", [R2_PROFILES[0]], ids=lambda profile: profile.name)
def test_rendered_hooke_optimizer_group_inventory_is_explicit(
    tmp_path: Path, profile: object
) -> None:
    """Record actual model group inventory; parameter-free groups must not vanish silently."""

    rendered = render_profile(
        profile,
        output_dir=tmp_path / "inputs",
        results_root=str(tmp_path / "results"),
        source_sha="7b88f833babcb82148c8f975a08b53970838be6a",
        device="cpu",
    )
    model = instantiate(OmegaConf.create(rendered.rendered.values["model"]))
    callback = next(
        item
        for item in rendered.rendered.values["callbacks"]
        if item["_target_"] == "tpen.callback.OptimizerState"
    )
    declared = tuple(callback["groups"])
    resolved = resolve_parameter_groups(model, callback["groups"])
    inventory = {
        group: {
            "parameters": len(parameters),
            "elements": sum(int(parameter.numel()) for _, parameter in parameters),
        }
        for group, parameters in resolved.items()
    }
    # Keep this assertion deliberately about the actual rendered model.  It
    # fails if a declared parameter-free group (currently factors) is silently
    # dropped before the callback emits JSONL evidence.
    missing = [group for group in declared if group not in inventory]
    assert not missing, f"declared={declared!r}; actual_inventory={inventory!r}; missing={missing!r}"
