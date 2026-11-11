"""Independent operator-review probes for the round-2 contract delta."""

import json
from pathlib import Path


INPUTS = Path(__file__).with_name("contract_inputs.json")


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_t3_confirmation_streams_are_independent() -> None:
    inputs = load_inputs()
    rules = inputs["decision_rules"]
    mode = rules["parameters"]["confirmation_mode"]
    recurrence = rules["recurrence"]

    assert "not pooled" in mode, f"confirmation mode sanity check failed: {mode!r}"
    assert "each confirmation stream independently" in recurrence, (
        "confirmation streams are not declared independently; "
        f"recurrence[:300]={recurrence[:300]!r}"
    )
    pooled_index = recurrence.find("streams 2/3 pooled")
    remeasure_index = recurrence.find("re-measure early, basin and late W")
    assert not (0 <= pooled_index < remeasure_index), (
        "W confirmation clause is preceded by pooled stream wording; "
        f"recurrence[:300]={recurrence[:300]!r}"
    )


def test_t4_bounded_null_pooling_family_and_upper_bound() -> None:
    inputs = load_inputs()
    parameters = inputs["decision_rules"]["parameters"]
    rules = inputs["decision_rules"]
    pooling = parameters["zero_event_bound_pooling"]
    text = rules["bounded_null"]

    n_bulk = sum(
        threshold < 0.5
        for threshold in inputs["measurement"]["fixed_exceedance_thresholds_hartree"]
    )
    n_tail = 2 + int("hydrogen-equivalent" in text)
    expected_k = (
        (n_bulk + n_tail)
        * len(parameters["bounded_null_reference_updates"])
        * len(parameters["late_updates"])
    )
    assert pooling["family_k_bounded_null"] == expected_k, (
        "bounded-null family_k mismatch: "
        f"recorded={pooling['family_k_bounded_null']}, expected={expected_k}"
    )

    upper_bound = 1 - (0.025 / expected_k) ** (1 / pooling["bounded_null_pooled_n"])
    assert upper_bound <= 0.001, f"bounded-null upper bound U={upper_bound:.12g}, threshold=0.001"


def test_t5_admission_requires_tail_freeze_and_pilot() -> None:
    inputs = load_inputs()
    science_lineages = inputs["admission"]["science_lineages"]
    blob = json.dumps(science_lineages)

    assert "hydrogen_equivalent_tail_threshold" in blob or "hydrogen-equivalent" in blob, (
        "science-lineage admission lacks the hydrogen-equivalent tail-threshold prerequisite; "
        f"admission keys={list(inputs['admission'])!r}"
    )
    assert "pilot" in inputs["admission"], (
        "admission lacks a pilot prerequisite; "
        f"admission keys={list(inputs['admission'])!r}"
    )
