"""Round-4 verification gap closure for the 2026-11-06 amendment of the H-v2
contract.

The round-4 ``DECLARED_PATHS``/``contract_decisions_frozen.json`` mechanism
(curated whitelist of decision-bearing subtrees compared against a frozen
snapshot) is retired by the round-7 binding decisions: round 5's full-leaf
``contract_frozen_snapshot.json`` comparison in
``test_contract_review_amend_r5.py`` already covers every path round 4
declared, plus everything else, with one mechanism instead of a
hand-maintained list. Keeping both meant an intentional contract change under
a declared path had to touch three files with two different update
procedures and two different error messages; that duplication is removed
here along with the two now-redundant snapshot tests
(``test_frozen_snapshot_declares_exactly_the_documented_paths`` and
``test_every_declared_decision_path_matches_its_frozen_value``) and
``contract_decisions_frozen.json`` itself.

``test_admission_science_lineages_current_status_agrees_with_status`` is kept:
it asserts an invariant between two fields that both state the same
unauthorized-science gate, which is a semantic check the frozen-snapshot
comparison does not make (the snapshot only catches drift from a frozen
value, not internal agreement between two live fields).
"""

import json
from pathlib import Path

INPUTS = Path(__file__).with_name("contract_inputs.json")


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_admission_science_lineages_current_status_agrees_with_status() -> None:
    science_lineages = load_inputs()["admission"]["science_lineages"]
    assert science_lineages["current_status"] == "prohibited"
    assert science_lineages["status"] == "prohibited_until_operator_reviews_science_plan"
    # Both fields assert the same unauthorized-science gate; neither may drift
    # from the other.
    assert science_lineages["current_status"] in science_lineages["status"]
