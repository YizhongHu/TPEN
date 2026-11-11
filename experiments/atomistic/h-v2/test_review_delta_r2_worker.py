"""PR651 T2 delta-r2 omission probes; test-only and Cannon-only.

These probes deliberately remain RED at the reviewed product head when the
corresponding omission is present.  The controls exercise the same builder,
prepare, and activation boundaries without requiring science execution.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def _module(name: str):
    key = f"h_v2_delta_r2_{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


protocol = _module("protocol")
attached = _module("attached")


def _assembled_science():
    config, identity = protocol.science_training_config("L1", 0)
    for target in protocol.candidate_modules_to_append():
        protocol.append_candidate_module(config, target)
    return config, identity


def test_science_contract_rejects_trainer_scalar_drift_after_builder_override(tmp_path):
    """A bound post-builder ``trainer.log_every_n_steps=100`` is forbidden."""

    config, identity = _assembled_science()
    plan, overrides, resolved = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff",
        results_root=tmp_path / "results",
        lineage="L1",
        config=config,
        science_config_identity=identity,
    )
    task_id = plan.tasks[0].logical_task_id
    overrides[task_id] = (*overrides[task_id], "trainer.log_every_n_steps=100")

    # This is the expected-refusal oracle.  The exact tip currently lets the
    # bad bound composition through, so this test is intentionally RED there.
    with pytest.raises(ValueError, match="trainer|scalar|log_every"):
        attached.prepare_h_v2_run(
            plan, overrides, resolved, output_dir=tmp_path / "prepared"
        )


def test_science_prepare_and_activate_reject_callback_replacement_before_dispatch(
    tmp_path, monkeypatch
):
    """Replacing callbacks after build must not reach executor dispatch.

    The override contains only the base science callbacks.  ``prepare`` must
    validate that bound composition, while activation must validate that same
    composition rather than the stale full-A1 ``prepared.config`` snapshot.
    """

    config, identity = _assembled_science()
    plan, overrides, resolved = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff",
        results_root=tmp_path / "results",
        lineage="L1",
        config=config,
        science_config_identity=identity,
    )
    task_id = plan.tasks[0].logical_task_id
    base_callbacks = protocol.science_training_config("L1", 0)[0]["callbacks"]
    overrides[task_id] = (
        *overrides[task_id],
        f"callbacks={base_callbacks!r}",
    )
    try:
        prepared = attached.prepare_h_v2_run(
            plan, overrides, resolved, output_dir=tmp_path / "prepared"
        )
    except ValueError as error:
        # A correct implementation may reject the replacement while composing
        # the bound config, before it creates a PreparedHV2Run.
        assert "callback" in str(error).lower() or "a1" in str(error).lower()
        return

    class DispatchReached(AssertionError):
        pass

    def fail_dispatch(*args, **kwargs):
        raise DispatchReached("executor dispatch reached stale-config boundary")

    monkeypatch.setattr(
        attached,
        "check_profile",
        lambda *args, **kwargs: SimpleNamespace(profile="valid-profile"),
    )
    monkeypatch.setattr(attached, "run_executor_cli", fail_dispatch)

    # The reviewed tip reaches run_executor_cli because activation consults
    # prepared.config (the stale full-A1 snapshot).  A fixed implementation
    # must raise ActivationRefusal before this sentinel is reached.
    with pytest.raises(attached.ActivationRefusal, match="incomplete-a1"):
        attached.activate_h_v2_run(
            prepared,
            science_python=sys.executable,
            executor_python=sys.executable,
            allocation=tmp_path / "allocation.json",
            worker_config=tmp_path / "worker.json",
            store=tmp_path / "store",
            activation_receipt_path=tmp_path / "activation.json",
            h_v2_receipt_path=tmp_path / "h-v2.json",
            expected_source_commit="a" * 40,
        )


def test_fully_assembled_science_control_is_accepted(tmp_path):
    """Pass-only control: the unmodified assembled science config is valid."""

    config, _ = _assembled_science()
    protocol.assert_hydrogen_contract(config)
    plan, overrides, resolved = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff",
        results_root=tmp_path / "results",
        lineage="L1",
        config=config,
        science_config_identity={"lineage": "L1"},
    )
    prepared = attached.prepare_h_v2_run(
        plan, overrides, resolved, output_dir=tmp_path / "prepared"
    )
    assert prepared.payload == "science"


def test_science_payload_cannot_opt_into_incomplete_a1(tmp_path):
    """A science identity cannot use the intermediate pipeline marker."""

    config, identity = protocol.science_training_config("L1", 0)
    with pytest.raises(ValueError, match="A1|callback"):
        attached.build_h_v2_train_plan(
            output_dir=tmp_path / "handoff",
            results_root=tmp_path / "results",
            lineage="L1",
            config=config,
            science_config_identity=identity,
            payload_label="pipeline-check",
        )
