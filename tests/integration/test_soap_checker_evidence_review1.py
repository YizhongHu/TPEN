"""Review probes for SOAP checker evidence wiring and rendered Hooke groups."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.atomistic.soap_pilot.checks import Verdict, check_disconnected_optimizer_state, run_rung1_checks
from experiments.atomistic.soap_pilot.composer import R2_PROFILES, PilotProfile, render_profile
from experiments.atomistic.soap_pilot.evidence import load_metric_records
from tpen.callback import OptimizerState
from tpen.logging import JSONL, LogRecord
from tpen.parameter_groups import resolve_parameter_groups
from tpen.training.state import TrainerState
from tpen.training.update import ModelParameterBinding, VMCUpdateState
from tests.unit.callback.support import RecordingContext, deliver_completed_iteration


class _LoggerContext(RecordingContext):
    """Minimal callback context that writes the production JSONL envelopes."""

    def __init__(self, logger: JSONL) -> None:
        super().__init__()
        self.logger = logger

    def log(self, metrics, *, step=None, namespace="run") -> None:
        self.logger.log(LogRecord(step=step, namespace=namespace, metrics=dict(metrics)))


def _state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> TrainerState:
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))
    return TrainerState(
        model=model,
        optimizer=optimizer,
        update_state=VMCUpdateState(optimizer=optimizer, model_parameters=binding),
    )


def test_run_rung1_checks_exposes_metric_evidence_gate_and_empty_is_unmeasured() -> None:
    """No metric records must be visible as an explicit UNMEASURED gate."""

    results = {result.name: result for result in run_rung1_checks({})}
    metric = next(
        (results[name] for name in ("metric-jsonl-evidence", "metric-finite") if name in results),
        None,
    )
    assert metric is not None
    assert metric.verdict is Verdict.UNMEASURED


def _rendered_optimizer_fixture(tmp_path: Path, profile: PilotProfile):
    """Build the rendered model and report actual parameter inventory only."""

    rendered = render_profile(
        profile,
        output_dir=tmp_path / "inputs",
        results_root=str(tmp_path / "results"),
        source_sha="42687973d4319dbd6f533bb5b067857f12b73d83",
        device="cpu",
    )
    model = instantiate(OmegaConf.create(rendered.rendered.values["model"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    callback_config = next(
        item
        for item in rendered.rendered.values["callbacks"]
        if item["_target_"] == "tpen.callback.OptimizerState"
    )
    declared = tuple(callback_config["groups"])
    resolved = resolve_parameter_groups(model, callback_config["groups"])
    inventory = {
        group: {
            "parameters": len(parameters),
            "elements": sum(int(parameter.numel()) for _, parameter in parameters),
        }
        for group, parameters in resolved.items()
    }
    assert "factors" not in inventory
    assert set(inventory) == set(declared) - {"factors"}

    assert not optimizer.state, "no optimizer step is run by this synthetic wire fixture"
    return model, optimizer, declared, resolved, inventory, callback_config


def _emit_optimizer_records(tmp_path: Path, profile: PilotProfile):
    model, optimizer, declared, resolved, inventory, callback_config = _rendered_optimizer_fixture(
        tmp_path, profile
    )
    metrics_path = tmp_path / "metrics.jsonl"
    logger = JSONL(metrics_path)
    context = _LoggerContext(logger)
    callback = OptimizerState(callback_config["groups"])
    deliver_completed_iteration(callback, context, _state(model, optimizer), step=99)
    records = load_metric_records(tmp_path)
    assert records is not None
    assert all(record.step == 99 for record in records)
    assert {record.namespace.rsplit("/", 1)[-1] for record in records} == set(resolved)

    expected_tensors = sum(item["parameters"] for item in inventory.values())
    assert expected_tensors == sum(
        int(record.metrics["n_no_optimizer_state"]) for record in records
    )
    group_inventory = {**inventory, "factors": {"parameters": 0, "elements": 0}}
    return records, declared, resolved, expected_tensors, group_inventory


@pytest.mark.parametrize("profile", [R2_PROFILES[0]], ids=lambda profile: profile.name)
def test_real_optimizer_callback_group_detector_is_explicitly_red(
    tmp_path: Path, profile: PilotProfile
) -> None:
    """Declared parameter-free groups expose the producer/consumer mismatch."""

    records, declared, resolved, expected_tensors, inventory = _emit_optimizer_records(tmp_path, profile)
    assert check_disconnected_optimizer_state(
        records, groups=tuple(resolved), terminal_step=99, expected_tensors=expected_tensors
    ).passed

    # Compatibility detector: the declaration includes parameter-free factors,
    # but the real producer emits only resolved groups.
    assert check_disconnected_optimizer_state(
        records,
        groups=declared,
        terminal_step=99,
        expected_tensors=expected_tensors,
        group_inventory=inventory,
    ).passed


@pytest.mark.parametrize("profile", [R2_PROFILES[0]], ids=lambda profile: profile.name)
def test_real_optimizer_callback_checkpoint_coordinate_detector_is_explicitly_red(
    tmp_path: Path, profile: PilotProfile
) -> None:
    """Completed iteration 99 must not be treated as terminal checkpoint 100."""

    records, _, resolved, expected_tensors, _ = _emit_optimizer_records(tmp_path, profile)
    # Independent coordinate detector: completed iteration 99 vs checkpoint 100.
    assert check_disconnected_optimizer_state(
        records, groups=tuple(resolved), terminal_step=100, expected_tensors=expected_tensors
    ).passed
