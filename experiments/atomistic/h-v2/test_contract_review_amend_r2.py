"""Round-2 verification gap tests for the 2026-11-06 amendment of the H-v2
contract.

Each test closes one of the three mutant-coverage gaps recorded in the
round-2 independent verification receipt (head 8ee32b0d): fields added by
accepted review-round-1 decisions 4 and 10 with no matching test, and the
amplitude-risk "no gate" invariant with no structural assertion against a
regrown gate key. Tests read only contract_inputs.json; stdlib + pytest
only, never import tpen.
"""

import json
from pathlib import Path

INPUTS = Path(__file__).with_name("contract_inputs.json")


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_storage_lifecycle_holystore_excludes_and_hpss_scope_are_exact() -> None:
    """Decision 4: the checkpoints are excluded from Holystore, kept in HPSS."""
    lifecycle = load_inputs()["logging"]["storage_lifecycle"]
    assert lifecycle["holystore_excludes"] == "the checkpoints themselves"
    assert lifecycle["hpss_scope"] == (
        "the whole run, checkpoints included, compressed"
    )


def test_superseded_stub_fields_match_the_stub_prose() -> None:
    """Decision 10: the stub's load-bearing facts live in fields, not just prose."""
    tail = load_inputs()["tail_characterization"]["sampler_tail_qualification"]
    fields = tail["superseded_stub_fields"]
    assert fields == {"max_radius_checked": 3.0, "band": 0.02}
    # The prose must agree with the field it was moved into.
    assert f"r={fields['max_radius_checked']:.0f}" in tail["superseded_stub"]


def test_amplitude_risk_has_no_gate_alarm_or_admission_key() -> None:
    """The amplitude-risk call is descriptive only; no structural gate key."""
    risk = load_inputs()["measurement"]["amplitude_risk"]
    assert set(risk.keys()) == {
        "status",
        "definition",
        "naming_rule",
        "interpretation",
        "decided_by",
        "thresholds",
        "evidence",
        "must_fix_before_science",
        "blocks",
        "source",
    }
    forbidden_substrings = ("gate", "alarm", "admission")
    for key in risk:
        lowered = key.lower()
        assert not any(bad in lowered for bad in forbidden_substrings), key
