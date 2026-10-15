"""CR audit of the native positive-control qualification actually executed.

The module-scoped fixture invokes the existing native omission-channel test one
time.  It records which real native positive-control assertions delegated to
the production comparator; it does not replace native execution with a probe.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tpen.execution.native_profile import SUPPORTED_STATE_OWNERS


@pytest.fixture(scope="module")
def native_positive_control_observation(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the existing native omission leaf once and retain its whole tree."""

    native = importlib.import_module("tests.integration.test_native_checkpoint_recovery")
    observed: list[str] = []
    invocations = 0
    real_assert_positive_control = native.assert_positive_control

    def recording_assert_positive_control(reference, perturbed, spec, *, resume_step):
        observed.append(spec.limb)
        return real_assert_positive_control(
            reference,
            perturbed,
            spec,
            resume_step=resume_step,
        )

    native_root = tmp_path_factory.mktemp("native-positive-controls")
    # This context is deliberately module-scoped: the expensive native test is
    # called once, and every generated trace/receipt remains below basetemp.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(native, "assert_positive_control", recording_assert_positive_control)
        native.test_real_restore_omission_channels_are_observed(native_root)
        invocations += 1

    record = {
        "schema": "formal-review-r2/native-positive-control-audit-v1",
        "original_test": "tests.integration.test_native_checkpoint_recovery::test_real_restore_omission_channels_are_observed",
        "original_invocations": invocations,
        "native_tmp_root": str(native_root),
        "observed_limbs": observed,
        "observed_limb_set": sorted(set(observed)),
        "supported_state_owners": sorted(SUPPORTED_STATE_OWNERS),
        "distinguishable_required": ["model", "optimizer", "process_rng", "sampler_mcmc"],
    }
    record_path = native_root / "native-positive-control-observation.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["record_path"] = str(record_path)
    return record


def test_native_positive_controls_cover_declared_state_owners(
    native_positive_control_observation: dict,
) -> None:
    """RED audit leaf: the executed native control set covers the declaration."""

    record = native_positive_control_observation
    assert record["original_invocations"] == 1
    assert set(record["observed_limbs"]) == set(record["supported_state_owners"]), (
        "native positive-control coverage mismatch; retained record="
        f"{record['record_path']} observed={record['observed_limb_set']} "
        f"missing={sorted(set(record['supported_state_owners']) - set(record['observed_limbs']))}"
    )


def test_native_positive_controls_are_declared_and_distinguishable(
    native_positive_control_observation: dict,
) -> None:
    """GREEN audit leaf: executed controls are valid without pinning the gap."""

    record = native_positive_control_observation
    observed = record["observed_limbs"]
    assert record["original_invocations"] == 1
    assert len(observed) == len(set(observed)), f"duplicate native controls; record={record['record_path']}"
    assert set(observed) <= set(record["supported_state_owners"])
    assert set(record["distinguishable_required"]) <= set(observed)
