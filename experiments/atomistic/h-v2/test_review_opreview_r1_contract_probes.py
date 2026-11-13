"""Independent operator-review probes for contract-input completeness."""

import json
from pathlib import Path


INPUTS = Path(__file__).with_name("contract_inputs.json")


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def _learned_model_proposal(inputs: dict) -> str:
    return next(
        item["proposal"]
        for item in inputs["lane_proposals"]
        if item["id"] == "learned_model_optimizer_and_training_sampler"
    )


def test_learned_model_proposal_names_nondefault_he_v1_fields() -> None:
    proposal = _learned_model_proposal(load_inputs())
    required = ("initial_weight=0.5", "implementation=vectorized", "include_spins=true", "float64")
    missing = [text for text in required if text not in proposal]
    assert not missing, f"missing non-default He-v1 fields: {missing}; proposal={proposal!r}"


def test_contract_does_not_forbid_and_retain_electron_electron_factor() -> None:
    inputs = load_inputs()
    description = inputs["admission"]["pipeline_check"]["parts"][0]["description"]
    proposal = _learned_model_proposal(inputs)
    assert not (
        "no electron-electron term/factor" in description
        and "retained as a declared factor" in proposal
    ), f"contradictory e-e wording: description={description!r}; proposal={proposal!r}"


def test_amplitude_risk_module_is_implemented_science_only_and_ungated() -> None:
    inputs = load_inputs()
    module = next(
        item for item in inputs["logging"]["modules"] if item["name"] == "amplitude_risk_fraction"
    )
    cadence = inputs["logging"]["cadence"]["amplitude_risk"]
    assert module["status"] == "implemented"
    assert module["kind"] == "training_callback"
    assert module["scope"] == "science_only"
    assert module["metrics_namespace"] == "train/amplitude_risk"
    assert cadence == {
        "scope": "science_only",
        "every_n_steps": 100,
        "start_step": 0,
        "includes_step_zero": True,
        "metrics_namespace": "train/amplitude_risk",
        "pilot": "omitted",
        "interpretation": "descriptive_only",
    }
    risk = inputs["measurement"]["amplitude_risk"]
    assert risk["interpretation"] == "descriptive only; introduces no alarm and no science gate"


def test_zero_event_bound_fits_declared_margins_and_documents_bonferroni_dependence() -> None:
    inputs = load_inputs()
    measurement = inputs["measurement"]["fresh_evaluation"]
    n = measurement["walkers"]
    alpha = (1.0 - measurement["uncertainty"]["simultaneous_band_level"]) / 2.0
    ub_4096 = 1.0 - alpha ** (1.0 / n)
    ub_16384 = 1.0 - alpha ** (1.0 / (4 * n))
    margins = inputs["decision_rules"]["parameters"]["exceedance_margins_hartree"]
    bonferroni_alpha = alpha / 27.0
    bonferroni_ub_4096 = 1.0 - bonferroni_alpha ** (1.0 / n)

    assert n == 4096
    assert ub_4096 <= margins["0.5"] / 2.0 and ub_4096 <= margins["1"] / 2.0, (
        f"ub_4096={ub_4096:.12g}, margin_0.5_half={margins['0.5'] / 2:.12g}, "
        f"margin_1_half={margins['1'] / 2:.12g}"
    )
    assert ub_16384 <= 0.001, f"ub_16384={ub_16384:.12g}, threshold=0.001"
    assert bonferroni_ub_4096 > 0.001, (
        f"bonferroni_ub_4096={bonferroni_ub_4096:.12g}, threshold=0.001; "
        f"ub_4096={ub_4096:.12g}, ub_16384={ub_16384:.12g}"
    )
