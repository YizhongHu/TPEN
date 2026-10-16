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
    try:
        # This is the original CR target function, imported without edits.
        _native_target(root)
    finally:
        patch.undo()
    receipt_path = root / "qualification-receipt.json"
    return {
        "receipt": json.loads(receipt_path.read_text(encoding="utf-8")),
        "launches": launches,
    }


@pytest.mark.integration
def test_b4_native_receipt_commands_match_actual_launch_argv_and_pids(
    native_qualification_artifacts: dict[str, object],
) -> None:
    """Expected RED: receipt commands must equal untouched native argv."""

    receipt = native_qualification_artifacts["receipt"]
    launches = native_qualification_artifacts["launches"]
    assert isinstance(receipt, dict)
    assert isinstance(launches, list)
    by_label = {
        argv[argv.index("--label") + 1]: record
        for record in launches
        if isinstance(record, dict)
        and isinstance(record.get("argv"), list)
        and "--label" in record["argv"]
        for argv in (record["argv"],)
    }
    assert {"reference", "recovery"} <= by_label.keys()
    assert receipt["commands"] == [by_label["reference"]["argv"], by_label["recovery"]["argv"]]
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
