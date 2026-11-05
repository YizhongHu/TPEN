"""Small structural checks for the H-v2 contract inputs."""

import json
from pathlib import Path


INPUTS = Path(__file__).with_name("contract_inputs.json")


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_admission_is_prohibited_while_operator_decisions_are_pending() -> None:
    inputs = load_inputs()
    decisions = inputs["operator_decisions"]
    assert inputs["admission"]["current_status"] == "prohibited"
    assert all(item["status"] in {"absent", "pending"} for item in decisions.values())
    assert inputs["admission"]["allowed_only_if_all_operator_decisions_status"] == "decided"


def test_result_categories_are_closed_and_seed_identities_are_distinct() -> None:
    inputs = load_inputs()
    assert set(inputs["decision_rules"]["categories"]) == {
        "recurrence",
        "bounded_null",
        "unresolved",
    }
    assert "separate namespaces and must be distinct" in inputs["measurement"]["seed_identity_rule"]


def test_hydrogen_identity_and_comparison_sources_are_pinned() -> None:
    inputs = load_inputs()
    assert inputs["study"]["electron_count"] == 1
    assert inputs["study"]["nuclear_charge"] == 1
    assert inputs["study"]["exact_ground_energy_hartree"] == -0.5
    assert len(inputs["source_identity"]["comparison_paths"]) == 4
    assert len(inputs["source_identity"]["comparison_commit"]) == 40
