"""Round-4 verification gap closure for the 2026-11-06 amendment of the H-v2
contract.

The round-3 independent verification (head 010a1cb1) found one surviving
mutant: ``admission.science_lineages.current_status`` flipped from
``"prohibited"`` to ``"admitted"`` with no test noticing, because it is a
second, parallel statement of the same gate as the already-tested ``status``
field. This is the third round in a row lost to an untested decision-bearing
field (round 2 lost three: ``holystore_excludes``, ``superseded_stub_fields``,
the amplitude-risk gate key), so this closes the class of bug rather than the
one field: a generative test asserts an explicitly declared, exhaustive list
of decision-bearing JSON subtrees against a frozen expected-value snapshot,
so a field "existing without a test" becomes a failing test instead of a
silent survivor.

``contract_decisions_frozen.json`` is the frozen snapshot: it was generated
once from the contract at this head and is never auto-regenerated. Any future
*intentional* contract change under one of the declared paths must update
both files by hand in the same commit, or this test fails.

DECLARED_PATHS was built by walking every key in ``contract_inputs.json``
(dicts and list items) for a key name containing one of: status, admission,
gate, threshold, count, cadence, format, retention, lifecycle, boundary. Each
match's nearest decision-bearing ancestor subtree is declared whole (not just
the matching leaf), so a mutation anywhere inside that subtree is caught, not
only at the exact keyword-matched leaf. Paths left out, and why:

- ``study.{name,system,nuclear_charge,exact_ground_energy_hartree,
  excluded_systems}``: physical constants/labels, no keyword match, not
  operator decisions.
- ``source_identity.*``: historical provenance pointers, no keyword match.
- ``measurement.{estimand,primary_statistic,primary_dispersion,
  secondary_dispersion,energy_error_profile,complete_record_rule,
  seed_roster,fresh_evaluation}``: no keyword match; the statistic/seed
  content is already pinned exactly by
  ``test_primary_statistic_drives_rules_and_mad_is_secondary``,
  ``test_all_seed_namespaces_and_evaluation_seeds_are_globally_distinct``,
  ``test_sampler_tail_qualification_seed_namespace_is_disjoint``, and
  ``test_training_and_fresh_evaluation_sampler_settings_are_exact``.
- ``controls``: a general control list, no keyword match.

``decision_rules.{recurrence,bounded_null,unresolved}`` have no keyword match
either, but the Cannon leaf sweep below found ``decision_rules.unresolved``
had zero test coverage (``recurrence``/``bounded_null`` only had the partial
substring checks in ``test_primary_statistic_drives_rules_and_mad_is_secondary``).
All three are declared here with exact-string equality rather than justified
away, closing that leftover instead of documenting it.
"""

import json
from pathlib import Path

INPUTS = Path(__file__).with_name("contract_inputs.json")
FROZEN = Path(__file__).with_name("contract_decisions_frozen.json")

# Every declared path is asserted as a whole subtree (dict/list/scalar), so a
# mutation anywhere inside it -- keyword-matched leaf or not -- is caught.
DECLARED_PATHS = [
    "study.electron_count",
    "measurement.fixed_exceedance_thresholds_hartree",
    "measurement.checkpoint_retention",
    "measurement.evaluated_checkpoints",
    "measurement.tail_rule",
    "measurement.normalizability_gate",
    "measurement.analytic_controls",
    "measurement.amplitude_risk",
    "decision_rules.categories",
    "decision_rules.parameters",
    "decision_rules.recurrence",
    "decision_rules.bounded_null",
    "decision_rules.unresolved",
    "lane_proposals",
    "admission",
    "logging",
    "tail_characterization",
    "pr_560_incorporation",
    "experiment_boundary",
]


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_frozen() -> dict:
    with FROZEN.open(encoding="utf-8") as stream:
        return json.load(stream)


def _get(obj: dict, dotted_path: str):
    cur = obj
    for part in dotted_path.split("."):
        cur = cur[part]
    return cur


def test_frozen_snapshot_declares_exactly_the_documented_paths() -> None:
    frozen = load_frozen()
    assert set(frozen) == set(DECLARED_PATHS)
    assert len(DECLARED_PATHS) == len(set(DECLARED_PATHS))


def test_every_declared_decision_path_matches_its_frozen_value() -> None:
    inputs = load_inputs()
    frozen = load_frozen()
    for path in DECLARED_PATHS:
        live = _get(inputs, path)
        assert live == frozen[path], f"decision path {path!r} drifted from its frozen value"


def test_admission_science_lineages_current_status_agrees_with_status() -> None:
    science_lineages = load_inputs()["admission"]["science_lineages"]
    assert science_lineages["current_status"] == "prohibited"
    assert science_lineages["status"] == "prohibited_until_operator_reviews_science_plan"
    # Both fields assert the same unauthorized-science gate; neither may drift
    # from the other.
    assert science_lineages["current_status"] in science_lineages["status"]
