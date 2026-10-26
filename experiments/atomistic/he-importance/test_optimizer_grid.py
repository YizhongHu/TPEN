"""Tests for the Stage-Q optimizer grid: declaration versus runnability.

These tests prove the two halves of the acceptance contract on Task
Orchestrator item 993de93d: the declared 120-cell universe reconciles to the
consolidated design (sections 2.7/2.8) and to the committed stage-coordinate
authority, and a build-time roster can change which cells are runnable
without ever changing which cells are declared.
"""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import pytest


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage_coordinate = _load("he_importance_stage_coordinate_for_grid_test", "stage_coordinate.py")
optimizer_grid = _load("he_importance_optimizer_grid_test", "optimizer_grid.py")

_EXPECTED_FAMILY_COUNTS = {
    "adam": 8,
    "sr": 32,
    "kfac": 32,
    "spring": 32,
    "linear_method": 16,
}


def _fake_roster(*, admitted: dict[str, bool], requires: dict[str, str] | None = None) -> list[object]:
    requires = requires or {}
    return [
        type(
            "MethodAvailability",
            (),
            {
                "method": method,
                "admitted": admitted[method],
                "target": "torch.optim.Adam" if admitted[method] else None,
                "requires": requires.get(method, f"{method} is not admitted"),
            },
        )()
        for method in _EXPECTED_FAMILY_COUNTS
    ]


def _all_available_roster() -> list[object]:
    return _fake_roster(admitted={method: True for method in _EXPECTED_FAMILY_COUNTS})


def _declared_signature(cell: "stage_coordinate.MaterializedCell") -> str:
    """The declared (availability-free) half of one materialized cell's identity."""

    identity = dict(cell.manifest["scientific_identity"])
    identity.pop("optimizer_cell", None)
    # ``canonical_json`` recurses through the frozen MappingProxyType/tuple
    # containers a materialized manifest carries; plain ``json.dumps`` cannot.
    return stage_coordinate.content_hash(identity)


# ---------------------------------------------------------------------------
# Per-family and total arithmetic
# ---------------------------------------------------------------------------


def test_per_family_counts_match_the_literal_section_2_7_table() -> None:
    authority = optimizer_grid.load_authority()
    declared = {entry["family"]: entry["declared_cells"] for entry in authority["families"]}
    assert declared == _EXPECTED_FAMILY_COUNTS
    assert sum(declared.values()) == 120


def test_declared_cells_total_and_per_family_match_an_independent_expansion() -> None:
    authority = optimizer_grid.load_authority()
    cells = optimizer_grid.declared_cells()
    assert len(cells) == 120

    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.family] = counts.get(cell.family, 0) + 1
    assert counts == _EXPECTED_FAMILY_COUNTS

    # Independent of declared_cells()'s own loop: recompute each family's
    # Cartesian product length directly from the authority's varied factors.
    for family_entry in authority["families"]:
        value_lists = [family_entry["varied"][name] for name in family_entry["varied"]]
        independent_count = len(list(itertools.product(*value_lists)))
        assert independent_count == family_entry["declared_cells"] == math.prod(len(v) for v in value_lists)


def test_malformed_authority_total_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority = json.loads(optimizer_grid._AUTHORITY_PATH.read_text(encoding="utf-8"))
    authority["families"][0]["declared_cells"] += 1  # desynchronize the arithmetic deliberately
    broken_path = tmp_path / "optimizer_grid_authority.json"
    broken_path.write_text(json.dumps(authority), encoding="utf-8")
    monkeypatch.setattr(optimizer_grid, "_AUTHORITY_PATH", broken_path)
    with pytest.raises(optimizer_grid.OptimizerGridError, match="expands to"):
        optimizer_grid.declared_cells()


# ---------------------------------------------------------------------------
# Cross-authority reconciliation
# ---------------------------------------------------------------------------


def test_cross_authority_reconciliation_to_480_q_runlets() -> None:
    cells = optimizer_grid.declared_cells()
    contexts = optimizer_grid.load_authority()["contexts"]
    stage = stage_coordinate.stage_definition("Q")

    total = len(cells) * len(contexts) * stage.seeds_per_point
    assert total == 480
    assert total == stage.maximum_lineages
    assert stage_coordinate.seed_labels("Q") == (810_001, 810_002)


# ---------------------------------------------------------------------------
# Deterministic ids
# ---------------------------------------------------------------------------


def test_declared_cells_is_stable_across_calls() -> None:
    first = optimizer_grid.declared_cells()
    second = optimizer_grid.declared_cells()
    assert [(c.family, dict(c.levels), c.cell_id, c.lexical_id) for c in first] == [
        (c.family, dict(c.levels), c.cell_id, c.lexical_id) for c in second
    ]


def test_all_cell_ids_and_lexical_ids_are_distinct() -> None:
    cells = optimizer_grid.declared_cells()
    assert len({cell.cell_id for cell in cells}) == 120
    assert len({cell.lexical_id for cell in cells}) == 120


# ---------------------------------------------------------------------------
# Contract clause 5: the inventory report is roster-derived, not hard-coded
# ---------------------------------------------------------------------------


def test_stage_q_report_is_derived_from_the_injected_roster() -> None:
    reason = "a passing fail-closed kfac-pytorch compatibility gate; a failing gate leaves KFAC unavailable"
    roster_adam_only = _fake_roster(
        admitted={"adam": True, "sr": False, "kfac": False, "spring": False, "linear_method": False},
        requires={"kfac": reason},
    )
    report = optimizer_grid.stage_q_report(roster=roster_adam_only)

    assert report["stage"] == "Q"
    assert report["declared"] == 120
    assert report["runnable"] == 8
    assert report["unavailable"] == 112
    assert report["families"]["adam"] == {"declared": 8, "runnable": 8, "unavailable": 0, "reason": None}
    assert report["families"]["kfac"] == {"declared": 32, "runnable": 0, "unavailable": 32, "reason": reason}
    assert report["families"]["sr"]["declared"] == 32
    assert report["families"]["sr"]["unavailable"] == 32
    assert report["families"]["sr"]["runnable"] == 0
    assert report["families"]["spring"]["declared"] == 32
    assert report["families"]["spring"]["unavailable"] == 32
    assert report["families"]["spring"]["runnable"] == 0
    assert report["families"]["linear_method"]["declared"] == 16
    assert report["families"]["linear_method"]["unavailable"] == 16
    assert report["families"]["linear_method"]["runnable"] == 0
    assert set(report["excluded_stages"]) == {"O1", "R", "F"}

    # A second roster, admitting one more family: clause 5 forbids a
    # hard-coded runnable count, so the totals must move with the roster.
    roster_adam_and_sr = _fake_roster(
        admitted={"adam": True, "sr": True, "kfac": False, "spring": False, "linear_method": False}
    )
    report_two = optimizer_grid.stage_q_report(roster=roster_adam_and_sr)
    assert report_two["declared"] == 120
    assert report_two["runnable"] == 40
    assert report_two["unavailable"] == 80
    assert report_two["families"]["sr"]["runnable"] == 32
    assert report_two["families"]["sr"]["unavailable"] == 0


# ---------------------------------------------------------------------------
# Zero unintended duplicate resolved scientific identities
# ---------------------------------------------------------------------------


def test_materialized_q_inventory_has_no_unintended_duplicate_identities(tmp_path: Path) -> None:
    availability = optimizer_grid.roster_availability(_all_available_roster())
    cells = stage_coordinate.materialize_intended_configurations(tmp_path, availability=availability)
    assert len(cells) == 480
    assert len({cell.content_hash for cell in cells}) == 480

    # The declared signature (scientific_identity minus the roster-derived
    # optimizer_cell status) is seed-free by design: 120 cells x 2 contexts =
    # 240 distinct signatures, each shared by exactly the 2 seeds (810001,
    # 810002) that make the full manifest content_hash unique above.
    signatures = [_declared_signature(cell) for cell in cells]
    assert len(set(signatures)) == 240
    counts: dict[str, int] = {}
    for signature in signatures:
        counts[signature] = counts.get(signature, 0) + 1
    assert set(counts.values()) == {2}


def test_duplicate_identity_detector_fires_on_a_cross_entry_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control for the exactly-twice invariant above.

    ``materialize_stage`` only dedups within one call, so it cannot catch two
    distinct inventory entries that happen to resolve to the same cell. This
    proves the signature-count check above actually fires on that failure
    mode rather than passing only because nothing in the committed inventory
    ever hits it.
    """

    entries = json.loads(Path(__file__).with_name("intended_configurations.json").read_text(encoding="utf-8"))
    assert len(entries) == 120
    entries[1] = copy.deepcopy(entries[0])
    broken_path = tmp_path / "intended_configurations.json"
    broken_path.write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(stage_coordinate, "_INTENDED_CONFIGURATIONS_PATH", broken_path)

    availability = optimizer_grid.roster_availability(_all_available_roster())
    cells = stage_coordinate.materialize_intended_configurations(tmp_path, availability=availability)
    assert len(cells) == 480

    signatures = [_declared_signature(cell) for cell in cells]
    assert len(set(signatures)) == 238

    counts: dict[str, int] = {}
    for signature in signatures:
        counts[signature] = counts.get(signature, 0) + 1
    assert sorted(set(counts.values())) == [2, 4]


# ---------------------------------------------------------------------------
# Contract clause 4: enumeration and runnability are separate
# ---------------------------------------------------------------------------


def test_contract_clause_4_roster_change_leaves_declared_universe_fixed(tmp_path: Path) -> None:
    roster_a = _fake_roster(admitted={"adam": True, "sr": False, "kfac": False, "spring": False, "linear_method": False})
    roster_b = _fake_roster(admitted={"adam": True, "sr": True, "kfac": False, "spring": False, "linear_method": False})

    availability_a = optimizer_grid.roster_availability(roster_a)
    availability_b = optimizer_grid.roster_availability(roster_b)
    assert availability_a["sr"].status == "unavailable"
    assert availability_b["sr"].status == "available"
    for method in ("adam", "kfac", "spring", "linear_method"):
        assert availability_a[method].status == availability_b[method].status

    declared_ids_before = {cell.cell_id for cell in optimizer_grid.declared_cells()}

    cells_a = stage_coordinate.materialize_intended_configurations(tmp_path / "a", availability=availability_a)
    cells_b = stage_coordinate.materialize_intended_configurations(tmp_path / "b", availability=availability_b)

    # declared_cells() takes no roster argument at all: its universe cannot
    # have moved between the two materializations.
    declared_ids_after = {cell.cell_id for cell in optimizer_grid.declared_cells()}
    assert declared_ids_after == declared_ids_before

    signatures_a = {_declared_signature(cell) for cell in cells_a}
    signatures_b = {_declared_signature(cell) for cell in cells_b}
    assert signatures_a == signatures_b
    # Seed-free declared signatures: 120 cells x 2 contexts = 240, each seed
    # (810001, 810002) sharing one; len(cells_*) stays 480.
    assert len(signatures_a) == 240
    assert len(cells_a) == len(cells_b) == 480

    def _sr_statuses(cells: tuple) -> set[str]:
        return {
            cell.manifest["scientific_identity"]["optimizer_cell"]["status"]
            for cell in cells
            if cell.manifest["scientific_identity"]["optimizer_cell"]["method"] == "sr"
        }

    assert _sr_statuses(cells_a) == {"unavailable"}
    assert _sr_statuses(cells_b) == {"available"}


# ---------------------------------------------------------------------------
# Fail-closed roster handling
# ---------------------------------------------------------------------------


def test_roster_availability_fails_closed_on_a_missing_family() -> None:
    roster = [entry for entry in _all_available_roster() if entry.method != "kfac"]
    with pytest.raises(optimizer_grid.OptimizerGridError, match="kfac"):
        optimizer_grid.roster_availability(roster)


def test_materialize_fails_closed_when_availability_omits_a_declared_method(tmp_path: Path) -> None:
    availability = optimizer_grid.roster_availability(_all_available_roster())
    del availability["kfac"]
    with pytest.raises(stage_coordinate.MaterializationError, match="kfac"):
        stage_coordinate.materialize_intended_configurations(tmp_path, availability=availability)


# ---------------------------------------------------------------------------
# Unavailable arms stay visible with the verbatim roster reason
# ---------------------------------------------------------------------------


def test_unavailable_arms_are_present_and_carry_the_roster_reason_verbatim(tmp_path: Path) -> None:
    reason = "a passing fail-closed kfac-pytorch compatibility gate; a failing gate leaves KFAC unavailable"
    roster = _fake_roster(
        admitted={"adam": True, "sr": False, "kfac": False, "spring": False, "linear_method": False},
        requires={"kfac": reason},
    )
    availability = optimizer_grid.roster_availability(roster)
    cells = stage_coordinate.materialize_intended_configurations(tmp_path, availability=availability)

    kfac_cells = [
        cell
        for cell in cells
        if cell.manifest["scientific_identity"]["optimizer_cell"]["method"] == "kfac"
    ]
    # 32 declared kfac cells x 2 contexts x 2 seeds.
    assert len(kfac_cells) == 32 * 2 * 2
    assert all(
        cell.manifest["scientific_identity"]["optimizer_cell"]
        == {"method": "kfac", "status": "unavailable", "unavailable_reason": reason}
        for cell in kfac_cells
    )


# ---------------------------------------------------------------------------
# Committed intended_configurations.json matches a fresh render
# ---------------------------------------------------------------------------


def test_committed_intended_configurations_is_byte_identical_to_a_fresh_render() -> None:
    committed = Path(__file__).with_name("intended_configurations.json").read_text(encoding="utf-8")
    assert committed == optimizer_grid.render_intended_configurations_text()


def test_committed_intended_configurations_has_no_status_or_reason_key() -> None:
    raw_text = Path(__file__).with_name("intended_configurations.json").read_text(encoding="utf-8")
    assert '"status"' not in raw_text
    assert '"reason"' not in raw_text

    entries = json.loads(raw_text)
    assert len(entries) == 120
    for entry in entries:
        assert frozenset(entry["optimizer"]) == frozenset({"method", "levels"})
        assert len(entry["configurations"]) == 2
        for configuration in entry["configurations"]:
            assert frozenset(configuration["scientific_identity"]["optimizer"]) == frozenset({"method", "levels"})


# ---------------------------------------------------------------------------
# Contexts carry literal coordinates in the identity, not just a label
# ---------------------------------------------------------------------------


def test_context_identity_carries_the_authoritys_literal_override_coordinates() -> None:
    contexts = optimizer_grid.load_authority()["contexts"]
    by_name = {context["name"]: dict(context["overrides"]) for context in contexts}

    for entry in optimizer_grid.render_intended_configurations():
        for configuration in entry["configurations"]:
            context_identity = configuration["scientific_identity"]["context"]
            assert frozenset(context_identity) == frozenset({"name", "overrides"})
            name = context_identity["name"]
            assert context_identity["overrides"] == by_name[name]


def test_o_raw_and_o_composite_identities_differ_by_more_than_the_name() -> None:
    entry = optimizer_grid.render_intended_configurations()[0]
    contexts_by_name = {
        configuration["scientific_identity"]["context"]["name"]: configuration["scientific_identity"]["context"]
        for configuration in entry["configurations"]
    }
    assert frozenset(contexts_by_name) == frozenset({"O-raw", "O-composite"})

    o_raw_without_name = {k: v for k, v in contexts_by_name["O-raw"].items() if k != "name"}
    o_composite_without_name = {k: v for k, v in contexts_by_name["O-composite"].items() if k != "name"}
    assert o_raw_without_name != o_composite_without_name
    assert o_raw_without_name["overrides"] != o_composite_without_name["overrides"]
