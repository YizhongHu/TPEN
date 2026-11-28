"""Review probes for SOAP checker evidence wiring and rendered Hooke groups."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.atomistic.soap_pilot.checks import Verdict, check_disconnected_optimizer_state, run_rung1_checks
from experiments.atomistic.soap_pilot.composer import R2_PROFILES, PilotProfile, render_profile
from experiments.atomistic.soap_pilot import evidence as soap_evidence
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
        output_dir=tmp_path / "results" / "inputs",
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
    return rendered, model, optimizer, declared, resolved, inventory, callback_config


def _emit_optimizer_records(tmp_path: Path, profile: PilotProfile):
    rendered, model, optimizer, declared, resolved, inventory, callback_config = _rendered_optimizer_fixture(
        tmp_path, profile
    )
    metrics_path = tmp_path / "metrics.jsonl"
    logger = JSONL(metrics_path)
    context = _LoggerContext(logger)
    callback = OptimizerState(callback_config["groups"])
    deliver_completed_iteration(callback, context, _state(model, optimizer), step=99)
    records = soap_evidence.load_metric_records(tmp_path)
    assert records is not None
    assert all(record.step == 99 for record in records)
    assert {record.namespace.rsplit("/", 1)[-1] for record in records} == set(resolved)

    expected_tensors = sum(item["parameters"] for item in inventory.values())
    assert expected_tensors == sum(
        int(record.metrics["n_no_optimizer_state"]) for record in records
    )
    group_inventory = {**inventory, "factors": {"parameters": 0, "elements": 0}}
    return rendered, records, declared, resolved, expected_tensors, group_inventory


@pytest.mark.parametrize("profile", [R2_PROFILES[0]], ids=lambda profile: profile.name)
def test_real_optimizer_callback_group_detector_is_explicitly_red(
    tmp_path: Path, profile: PilotProfile
) -> None:
    """Declared parameter-free groups expose the producer/consumer mismatch."""

    _, records, declared, resolved, expected_tensors, inventory = _emit_optimizer_records(tmp_path, profile)
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

    _, records, _, resolved, expected_tensors, _ = _emit_optimizer_records(tmp_path, profile)
    # Independent coordinate detector: completed iteration 99 vs checkpoint 100.
    assert check_disconnected_optimizer_state(
        records, groups=tuple(resolved), terminal_step=100, expected_tensors=expected_tensors
    ).passed


@pytest.mark.parametrize("profile", [R2_PROFILES[0]], ids=lambda profile: profile.name)
def test_run_rung1_uses_real_rendered_inventory_and_reconciles_jsonl(
    tmp_path: Path, profile: PilotProfile
) -> None:
    """The production checker derives groups from rendered config, not a fixture inventory."""

    rendered, records, declared, resolved, expected_tensors, _ = _emit_optimizer_records(
        tmp_path, profile
    )
    groups, inventory = soap_evidence.build_optimizer_group_inventory(rendered.rendered.path)
    assert groups == declared
    assert set(inventory) == set(declared)
    assert inventory["factors"]["parameters"] == 0
    assert set(inventory) >= set(resolved)

    results = {
        result.name: result
        for result in run_rung1_checks(
            {
                "run_dir": tmp_path,
                "optimizer_state": {
                    "rendered_config_path": rendered.rendered.path,
                    "checkpoint_step": 100,
                },
            }
        )
    }
    assert results["disconnected-optimizer-state"].passed
    assert results["disconnected-optimizer-state"].evidence["empty_groups"] == ["factors"]
    assert sum(
        int(record.metrics["n_no_optimizer_state"])
        for record in records
    ) == expected_tensors


@pytest.mark.parametrize("profile", [R2_PROFILES[0]], ids=lambda profile: profile.name)
def test_run_rung1_missing_nonempty_group_is_unmeasured(
    tmp_path: Path, profile: PilotProfile
) -> None:
    """A real producer record removed for a parameter-bearing group cannot pass."""

    rendered, _, _, _, _, _ = _emit_optimizer_records(tmp_path, profile)
    metrics_path = tmp_path / "metrics.jsonl"
    records = soap_evidence.load_metric_records(tmp_path)
    assert records is not None
    metrics_path.write_text(
        "".join(
            json.dumps(record._asdict()) + "\n"
            for record in records
            if not record.namespace.endswith("/readout")
        ),
        encoding="utf-8",
    )
    results = {
        result.name: result
        for result in run_rung1_checks(
            {
                "run_dir": tmp_path,
                "optimizer_state": {
                    "rendered_config_path": rendered.rendered.path,
                    "checkpoint_step": 100,
                },
            }
        )
    }
    assert results["disconnected-optimizer-state"].verdict is Verdict.UNMEASURED


@pytest.mark.parametrize("profile", [R2_PROFILES[0]], ids=lambda profile: profile.name)
def test_run_rung1_rejects_inventory_contradicting_jsonl_parameter_count(
    tmp_path: Path, profile: PilotProfile
) -> None:
    """The selected producer n_parameters must agree with rendered ownership."""

    rendered, _, _, _, _, _ = _emit_optimizer_records(tmp_path, profile)
    records = soap_evidence.load_metric_records(tmp_path)
    assert records is not None
    payloads = []
    for record in records:
        payload = record._asdict()
        if record.namespace.endswith("/embedding"):
            payload["metrics"] = {**record.metrics, "n_parameters": record.metrics["n_parameters"] + 1}
        payloads.append(json.dumps(payload))
    (tmp_path / "metrics.jsonl").write_text("\n".join(payloads) + "\n", encoding="utf-8")
    results = {
        result.name: result
        for result in run_rung1_checks(
            {
                "run_dir": tmp_path,
                "optimizer_state": {
                    "rendered_config_path": rendered.rendered.path,
                    "checkpoint_step": 100,
                },
            }
        )
    }
    optimizer = results["disconnected-optimizer-state"]
    assert optimizer.verdict is Verdict.HARNESS_INVALID
    assert "contradicts" in optimizer.detail
