"""R4-only native receipt probes over pristine E5 target helpers."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import pytest

from tests.integration import test_native_checkpoint_recovery as native

_native_target = native.test_native_e1_to_e4_recovery_parity_divergence_and_receipt
pytestmark = pytest.mark.skipif(
    os.environ.get("TPEN_REVIEW_LAYER", "CR") != "CR",
    reason="B4 receipt command/PID audit is CR-only",
)


@pytest.fixture(scope="module")
def native_qualification_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Run the pristine native target once and reuse its receipt for probes."""

    root = tmp_path_factory.mktemp("native-qualification-b4")
    launches: list[dict[str, object]] = []
    original_launch = native._launch

    def recording_launch(*launch_args, **launch_kwargs):
        """Capture only the Popen call made by the pristine native launcher."""

        original_popen = native.subprocess.Popen

        def recording_popen(args, *popen_args, **popen_kwargs):
            argv = tuple(str(value) for value in args)
            process = original_popen(args, *popen_args, **popen_kwargs)
            launches.append({"argv": list(argv), "pid": process.pid})
            return process

        launcher_patch = pytest.MonkeyPatch()
        launcher_patch.setattr(native.subprocess, "Popen", recording_popen)
        try:
            return original_launch(*launch_args, **launch_kwargs)
        finally:
            launcher_patch.undo()

    patch = pytest.MonkeyPatch()
    patch.setattr(native, "_launch", recording_launch)
    capture_path = root / "captured-launches.json"
    try:
        # This is the original CR target function, imported without edits.
        _native_target(root)
    finally:
        patch.undo()
        # Persist the raw capture even if the pristine target above raised,
        # so the actual argv/PID evidence survives independent of whether
        # the target's own assertions (or ours) pass.
        capture_path.write_text(json.dumps(launches, indent=2), encoding="utf-8")
    receipt_path = root / "qualification-receipt.json"
    return {
        "receipt": json.loads(receipt_path.read_text(encoding="utf-8")),
        "launches": launches,
        "capture_path": capture_path,
    }


def _by_label(launches: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        argv[argv.index("--label") + 1]: record
        for record in launches
        if isinstance(record, dict)
        and isinstance(record.get("argv"), list)
        and "--label" in record["argv"]
        for argv in (record["argv"],)
    }


@pytest.mark.integration
def test_b4_native_receipt_commands_match_actual_launch_argv(
    native_qualification_artifacts: dict[str, object],
) -> None:
    """Expected RED: receipt commands must equal untouched native argv."""

    receipt = native_qualification_artifacts["receipt"]
    launches = native_qualification_artifacts["launches"]
    assert isinstance(receipt, dict)
    assert isinstance(launches, list)
    by_label = _by_label(launches)
    assert {"reference", "recovery"} <= by_label.keys()
    assert receipt["commands"] == [by_label["reference"]["argv"], by_label["recovery"]["argv"]]


@pytest.mark.integration
def test_b4_native_receipt_process_evidence_pids_match_actual_launch_pids(
    native_qualification_artifacts: dict[str, object],
) -> None:
    """Expected GREEN, independent of the command-text assertion above."""

    receipt = native_qualification_artifacts["receipt"]
    launches = native_qualification_artifacts["launches"]
    assert isinstance(receipt, dict)
    assert isinstance(launches, list)
    by_label = _by_label(launches)
    assert {"reference", "recovery"} <= by_label.keys()
    evidence_by_phase = {entry["phase"]: entry for entry in receipt["process_evidence"]}
    assert evidence_by_phase["initial"]["pid"] == by_label["reference"]["pid"]
    assert evidence_by_phase["recovery"]["pid"] == by_label["recovery"]["pid"]


@pytest.mark.integration
def test_b4_native_receipt_hashes_and_unmeasured_fields_remain_green(
    native_qualification_artifacts: dict[str, object],
) -> None:
    """Expected GREEN: receipt hashes and unmeasured channels remain explicit."""

    receipt = native_qualification_artifacts["receipt"]
    assert isinstance(receipt, dict)
    expected_source_sha = os.environ.get(
        "TPEN_REVIEW_EXPECTED_SOURCE_SHA",
        "a84df60515cb988d0f825cb15f87c97c6c0aad02",
    )
    assert receipt["source_sha"] == expected_source_sha
    chain_resume = Path(native.__file__).parents[2] / "tpen" / "execution" / "chain_resume.py"
    assert receipt["adapter_sha"] == hashlib.sha256(chain_resume.read_bytes()).hexdigest()
    assert receipt["unmeasured_arms"] == receipt["scientific"]["unmeasured_arms"]
    assert set(receipt["scientific"]["executed_arms"]) == set(receipt["channel_results"])


@pytest.mark.integration
def test_r3_fix5_native_receipt_required_provenance_fields_are_present(
    native_qualification_artifacts: dict[str, object],
) -> None:
    """R3 issue 5, expected GREEN: required-field-presence is a separate check
    from the exact-command comparison above.

    This deliberately does not re-check the *content* of ``commands`` (that is
    ``test_b4_native_receipt_commands_match_actual_launch_argv``'s RED
    assertion, above). A receipt missing or renaming a required provenance
    key would pass a content comparison vacuously (e.g. ``KeyError`` is not
    the same failure as a wrong value) while still being a real provenance
    regression, so field presence is asserted independently and kept
    separate from the command-text comparison per the reviewer's
    instruction not to conflate the two results.
    """

    receipt = native_qualification_artifacts["receipt"]
    assert isinstance(receipt, dict)
    required_top_level_fields = {
        "schema",
        "source_sha",
        "adapter_sha",
        "profile_version",
        "channel_map_version",
        "commands",
        "process_evidence",
        "channel_results",
        "checkpoint_faults",
        "collection_output",
        "testcase_ids",
        "junit_identities",
        "raw_fault_evidence",
        "restore_disabled_control",
        "unmeasured_arms",
        "scheduler",
        "backend",
        "wrapper",
        "application",
        "scientific",
    }
    missing = required_top_level_fields - receipt.keys()
    assert not missing, f"receipt is missing required provenance fields: {sorted(missing)}"
    for process_record in receipt["process_evidence"]:
        assert {"phase", "pid", "seed", "seed_source", "command", "returncode"} <= process_record.keys()
    assert {"registration_calls", "observation_calls"} <= receipt["wrapper"].keys()


@pytest.mark.integration
def test_r3_issue13_zero_backend_calls_across_post_registration_launches(
    native_qualification_artifacts: dict[str, object],
) -> None:
    """R3 issue 13, expected GREEN: per-scenario zero-backend-call assertion.

    The pristine target (``test_native_e1_to_e4_recovery_parity_divergence_and_receipt``)
    asserts ``client.register_calls == 1`` and ``client.observe_calls == 1``
    immediately after ``register_lineage``/``observe_lineage``, before any of
    the three post-registration scenario launches (recovery/successor,
    reference, partial-restore control) run. Those three launches go through
    ``_launch`` (a direct subprocess invocation) and never touch the adapter
    or client again. Because both call counters are monotonically
    non-decreasing integers, the final counts recorded in the receipt's
    ``wrapper`` block equal 1 if and only if none of the three later
    scenarios made an additional backend call -- so asserting the final
    counts here is a valid zero-backend-call proof for every one of those
    three scenarios individually, without needing to re-run or re-instrument
    each one separately against the shared single native execution.
    """

    receipt = native_qualification_artifacts["receipt"]
    assert isinstance(receipt, dict)
    wrapper = receipt["wrapper"]
    assert wrapper["registration_calls"] == 1, (
        "a post-registration scenario (recovery, reference, or partial) made "
        "an unexpected additional backend registration call"
    )
    assert wrapper["observation_calls"] == 1, (
        "a post-registration scenario (recovery, reference, or partial) made "
        "an unexpected additional backend observation call"
    )
