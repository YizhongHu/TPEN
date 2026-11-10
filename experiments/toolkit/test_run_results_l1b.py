"""P8 reader support for the S2b-2 L1b sampler-bearing placement receipt.

Fixtures (root suite, no executor import):

``run-manifest-l1a-v2-*.json``
    Bytes produced by the untouched L1a producer at ``f384f44d`` (the L2
    base): the backward-read control for ``science-placement-receipt-v2``.
``run-manifests-v3-sampler.json``
    Serialized output of the L1b producer (``SciencePlacementReceiptV3.from_l1a``
    over the L1a shadow, then ``RunManifest``) for every v3 outcome, including
    outside use seen only in a dropped sample (``evicted-out-of-entitlement``),
    one forgiven transient gap (``gaps-within-budget``), an exceeded gap
    budget (``gap-budget-exceeded``), and the distinct MIG failure
    (``mig-unsupported``); wall-clock fields pinned.  ``executor/tests/test_s2b2_l1b_gpu_sampler.py``
    re-decodes this file with the executor decoder byte-for-byte, and the live
    producer-to-reader gate lives in ``test_review_p8_toolkit_r3.py``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.toolkit.run_results import (
    PLACEMENT_NOT_SAMPLED_MESSAGE,
    PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE,
    PLACEMENT_UNQUALIFIED_MESSAGE,
    SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2,
    SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V3,
    ManifestRefusal,
    read,
)

FIXTURES = Path(__file__).parent / "fixtures"
GPU_A = "GPU-12345678-1234-1234-1234-123456789ABC"
GPU_B = "GPU-12345678-1234-1234-1234-123456789ABD"
GPU_C = "GPU-12345678-1234-1234-1234-123456789ABE"
V3_EXPECTED = {
    "matched": ("qualified", "matched", None),
    "out-of-entitlement": ("mismatch", "out-of-entitlement", PLACEMENT_UNQUALIFIED_MESSAGE),
    "evicted-out-of-entitlement": ("mismatch", "out-of-entitlement", PLACEMENT_UNQUALIFIED_MESSAGE),
    "sampler-failed": ("telemetry-failed", "sampler-failed", PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
    "no-exact-pid-row": ("telemetry-failed", "no-exact-pid-row", PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
    "entitlement-unresolved": ("telemetry-failed", "entitlement-unresolved", PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
    "not-sampled": ("not-sampled", "cpu-only", None),
    "gaps-within-budget": ("qualified", "matched", None),
    "gap-budget-exceeded": ("telemetry-failed", "sampler-failed", PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
    "mig-unsupported": ("telemetry-failed", "sampler-failed", PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
}


def _v3_manifests() -> dict[str, dict]:
    return json.loads((FIXTURES / "run-manifests-v3-sampler.json").read_text(encoding="utf-8"))


def _read(tmp_path: Path, name: str, manifest: dict):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return read(path, lambda _: pytest.fail("attempt-only manifests must not inspect core"))


def test_p8_reads_untouched_l1a_v2_manifests() -> None:
    """F3: L1a v2 producer bytes stay readable under the unchanged v2 decoder."""

    for name, message in (("not-sampled", PLACEMENT_NOT_SAMPLED_MESSAGE), ("mismatch", PLACEMENT_UNQUALIFIED_MESSAGE)):
        result = read(FIXTURES / f"run-manifest-l1a-v2-{name}.json", lambda _: pytest.fail("no segments"))
        receipt = result.preserved_attempts[0].placement_receipt
        assert receipt["schema"] == SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2
        assert "gpu_sampler" not in receipt
        assert receipt["verdict"]["reason_class"] == name
        assert result.placement_verdict["message"] == message


def test_p8_rejects_sampler_fields_on_a_v2_receipt(tmp_path: Path) -> None:
    """F3: v2 stays closed; sampler evidence is only legal under the v3 schema."""

    manifest = json.loads((FIXTURES / "run-manifest-l1a-v2-not-sampled.json").read_text(encoding="utf-8"))
    sampler = _v3_manifests()["not-sampled"]["attempts"][0]["placement_receipt"]["gpu_sampler"]
    manifest["attempts"][0]["placement_receipt"]["gpu_sampler"] = sampler
    with pytest.raises(ManifestRefusal, match="keys differ"):
        _read(tmp_path, "v2-with-sampler", manifest)


def test_p8_reads_every_real_v3_producer_outcome(tmp_path: Path) -> None:
    """F5/F7: real serialized v3 producer output crosses P8 with exact verdicts."""

    manifests = _v3_manifests()
    assert set(manifests) == set(V3_EXPECTED)
    for name, (reason_class, code, message) in V3_EXPECTED.items():
        result = _read(tmp_path, name, manifests[name])
        receipt = result.preserved_attempts[0].placement_receipt
        assert receipt["schema"] == SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V3, name
        assert receipt["verdict"]["reason_class"] == reason_class, name
        assert receipt["gpu"]["reason"] == code, name
        assert len(receipt["gpu_sampler"]["samples"]) == manifests[name]["attempts"][0]["placement_receipt"]["gpu_sampler"]["count"]
        if message is None:
            assert result.placement_verdict is None, name
            assert not result.placement_unqualified, name
        else:
            assert result.placement_verdict["message"] == message, name
            assert result.placement_unqualified, name


def _rows(receipt: dict, key: str, value: object) -> None:
    for sample in receipt["gpu_sampler"]["samples"]:
        for row in sample["pid_rows"]:
            row[key] = value


def _blank_active_pci(receipt: dict) -> None:
    """D3: the active UUID's PCI is whitespace everywhere it appears."""

    for sample in receipt["gpu_sampler"]["samples"]:
        for item in sample["inventory"]:
            if item["uuid"] == GPU_A:
                item["pci.bus_id"] = " "
    receipt["gpu_sampler"]["aggregate"]["exact_pid_gpus"][0]["pci.bus_id"] = " "
    receipt["gpu"]["active_pci"] = [" "]


def _add_gaps(receipt: dict, *events: tuple[str, float, float, int]) -> None:
    """Append gap events ``(code, start, end, consecutive)`` with consistent counters."""

    gaps = receipt["gpu_sampler"]["gaps"]
    for code, start, end, run in events:
        gaps["events"].append({"code": code, "monotonic_start": start, "monotonic_end": end, "consecutive": run})
    gaps["count"] = len(gaps["events"])
    gaps["max_consecutive"] = max(event["consecutive"] for event in gaps["events"])
    gaps["total_seconds"] = sum(event["monotonic_end"] - event["monotonic_start"] for event in gaps["events"])


def _hide_dropped_outside(receipt: dict) -> None:
    aggregate = receipt["gpu_sampler"]["aggregate"]
    aggregate["samples_taken"] = receipt["gpu_sampler"]["count"] + 1
    aggregate["exact_pid_gpus"] = aggregate["exact_pid_gpus"] + [{"uuid": GPU_B, "pci.bus_id": "0000:00:02.0"}]


# Every executor sampler invariant, mirrored independently by P8.
P8_CONTRADICTIONS = {
    "quiescent-false": lambda r: r["gpu_sampler"].__setitem__("quiescent", False),
    "survivors": lambda r: r["gpu_sampler"].update(survivors=[101], quiescent=False),
    "foreign-pid-rows": lambda r: _rows(r, "pid", 9999),
    "outside-uuid": lambda r: _rows(r, "gpu_uuid", GPU_B),
    "row-absent-from-inventory": lambda r: r["gpu_sampler"]["samples"][0]["pid_rows"].append({"pid": 1234, "gpu_uuid": GPU_C}),
    "unresolved-entitlement": lambda r: r["declared"].__setitem__("devices", [r["declared"]["devices"][0], GPU_C]),
    "sample-outside-outer": lambda r: r["gpu_sampler"]["samples"][0].update(monotonic_start=100.0, monotonic_end=101.0),
    "reversed-sample": lambda r: r["gpu_sampler"]["samples"][0].update(monotonic_start=1.125, monotonic_end=1.0),
    "overlapping-samples": lambda r: r["gpu_sampler"]["samples"][1].__setitem__("monotonic_start", 1.0),
    "null-outer-interval": lambda r: r["gpu_sampler"].update(monotonic_start=None, monotonic_end=None),
    "null-version": lambda r: r["gpu_sampler"].__setitem__("version", None),
    "elapsed-disagrees": lambda r: r["gpu_sampler"].__setitem__("elapsed_seconds", 9.0),
    "count-disagrees": lambda r: r["gpu_sampler"].__setitem__("count", 1),
    "qualified-over-failed-sampler": lambda r: r["gpu_sampler"].update(status="unavailable", reason="telemetry-failed", failure={"code": "command-failed", "detail": "x"}),
    "attempted-not-sampled": lambda r: r["gpu_sampler"].update(status="unavailable", reason="not-sampled"),
    "unknown-failure-code": lambda r: r["gpu_sampler"].update(status="unavailable", reason="telemetry-failed", failure={"code": "invented", "detail": "x"}),
    "extra-sampler-key": lambda r: r["gpu_sampler"].__setitem__("extra", 1),
    "relabelled-mismatch": lambda r: r["verdict"].__setitem__("reason_class", "mismatch"),
    "active-uuid-tamper": lambda r: r["gpu"].__setitem__("active_uuids", [GPU_B]),
    "warning-tamper": lambda r: r["verdict"].__setitem__("warning", "quiet"),
    "acceptance-tamper": lambda r: r["verdict"].__setitem__("science_acceptance", "not-accepted"),
    "device-comparison-tamper": lambda r: r["comparison"].__setitem__("device", "unavailable"),
    "inventory-pci-whitespace": _blank_active_pci,
    "target-pid-tamper": lambda r: r["gpu_sampler"].__setitem__("target_pid", 4321),
    "aggregate-outside-flag-tamper": lambda r: r["gpu_sampler"]["aggregate"].__setitem__("outside_use_seen", True),
    "aggregate-hides-dropped-outside": _hide_dropped_outside,
    "aggregate-omits-retained-gpu": lambda r: r["gpu_sampler"]["aggregate"].__setitem__("exact_pid_gpus", []),
    "aggregate-below-count": lambda r: r["gpu_sampler"]["aggregate"].__setitem__("samples_taken", 1),
    "aggregate-unresolved-tamper": lambda r: r["gpu_sampler"]["aggregate"].__setitem__("entitlement_unresolved_seen", True),
    # r4: resolved executable and the recorded gap budget (manager ruling Q1).
    "null-executable": lambda r: r["gpu_sampler"].__setitem__("executable", None),
    "gap-count-disagrees": lambda r: r["gpu_sampler"]["gaps"].__setitem__("count", 1),
    "gap-unforgivable-code": lambda r: _add_gaps(r, ("malformed-output", 1.13, 1.2, 1)),
    "gap-outside-outer": lambda r: _add_gaps(r, ("command-failed", 2.0, 2.1, 1)),
    "gap-budget-exceeded-but-observed": lambda r: _add_gaps(r, ("command-failed", 1.13, 1.15, 1), ("command-failed", 1.16, 1.2, 2)),
    "gap-total-disagrees": lambda r: (_add_gaps(r, ("command-timeout", 1.13, 1.2, 1)), r["gpu_sampler"]["gaps"].__setitem__("total_seconds", 5.0)),
    "gap-run-numbering": lambda r: _add_gaps(r, ("command-failed", 1.13, 1.2, 2)),
    "gap-budget-above-cap": lambda r: r["gpu_sampler"]["gaps"].__setitem__("max_total_allowed", 17),
}
# Contradictions applied to other baselines: (baseline fixture, mutation).
P8_OTHER_CONTRADICTIONS = {
    "failure-detail-whitespace": ("sampler-failed", lambda r: r["gpu_sampler"]["failure"].__setitem__("detail", " ")),
    "evicted-outside-flag-cleared": (
        "evicted-out-of-entitlement", lambda r: r["gpu_sampler"]["aggregate"].__setitem__("outside_use_seen", False),
    ),
    # An exceeded budget must be labelled gap-budget-exceeded, and only then.
    "gap-excess-relabelled": ("gap-budget-exceeded", lambda r: r["gpu_sampler"]["failure"].__setitem__("code", "command-failed")),
    "gap-excess-claimed-without-gaps": ("sampler-failed", lambda r: r["gpu_sampler"]["failure"].__setitem__("code", "gap-budget-exceeded")),
    "gap-event-dropped": ("gaps-within-budget", lambda r: r["gpu_sampler"]["gaps"].__setitem__("events", [])),
    "gap-forgiven-past-budget": (
        "gaps-within-budget", lambda r: r["gpu_sampler"]["gaps"].update(max_consecutive_allowed=0, max_total_allowed=0),
    ),
    "mig-failure-detail-whitespace": ("mig-unsupported", lambda r: r["gpu_sampler"]["failure"].__setitem__("detail", " ")),
}


@pytest.mark.parametrize("name", sorted(P8_CONTRADICTIONS))
def test_p8_rejects_contradictory_v3_sampler_evidence(tmp_path: Path, name: str) -> None:
    """F4/F5: P8 independently refuses every contradiction the executor refuses."""

    manifest = copy.deepcopy(_v3_manifests()["matched"])
    _read(tmp_path, "baseline", copy.deepcopy(manifest))
    P8_CONTRADICTIONS[name](manifest["attempts"][0]["placement_receipt"])
    with pytest.raises(ManifestRefusal):
        _read(tmp_path, name, manifest)


@pytest.mark.parametrize("name", sorted(P8_OTHER_CONTRADICTIONS))
def test_p8_rejects_contradictions_on_other_baselines(tmp_path: Path, name: str) -> None:
    """D3 and the lifetime aggregate: contradictions on non-matched baselines are refused."""

    baseline, mutate = P8_OTHER_CONTRADICTIONS[name]
    manifest = copy.deepcopy(_v3_manifests()[baseline])
    _read(tmp_path, "baseline", copy.deepcopy(manifest))
    mutate(manifest["attempts"][0]["placement_receipt"])
    with pytest.raises(ManifestRefusal):
        _read(tmp_path, name, manifest)


def test_p8_types_mig_and_gap_budget_failures_distinctly(tmp_path: Path) -> None:
    """Q1/Q5: MIG and an exceeded gap budget are distinct typed failures, not malformed output."""

    manifests = _v3_manifests()
    for name in ("mig-unsupported", "gap-budget-exceeded"):
        receipt = _read(tmp_path, name, manifests[name]).preserved_attempts[0].placement_receipt
        assert receipt["gpu_sampler"]["failure"]["code"] == name, name
        assert receipt["verdict"]["reason"] == f"telemetry-failed: attempted GPU sampler failed ({name})", name
    gaps = _read(tmp_path, "gaps", manifests["gaps-within-budget"]).preserved_attempts[0].placement_receipt["gpu_sampler"]["gaps"]
    # The budget in force is recorded with every gap.
    assert (gaps["max_consecutive_allowed"], gaps["max_total_allowed"], gaps["count"], gaps["max_consecutive"]) == (1, 3, 1, 1)


def test_p8_requires_the_telemetry_message_for_telemetry_failures(tmp_path: Path) -> None:
    """F5: a telemetry-failed manifest carrying the L1a-only message is refused."""

    manifest = copy.deepcopy(_v3_manifests()["sampler-failed"])
    manifest["placement_verdict"]["message"] = PLACEMENT_UNQUALIFIED_MESSAGE
    with pytest.raises(ManifestRefusal, match="loud"):
        _read(tmp_path, "quiet-telemetry", manifest)
    manifest = copy.deepcopy(_v3_manifests()["out-of-entitlement"])
    manifest["placement_verdict"]["message"] = PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE
    with pytest.raises(ManifestRefusal, match="loud"):
        _read(tmp_path, "wrong-mismatch-message", manifest)
