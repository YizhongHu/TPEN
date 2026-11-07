"""Round-5 frozen-snapshot closure for the 2026-11-06 amendment of the H-v2
contract.

Rounds 2, 3, and 4 each closed one more untested decision-bearing field
(``holystore_excludes``/``superseded_stub_fields``/the amplitude-risk gate
key in round 2, ``admission.science_lineages.current_status`` in round 4's
reused arms) and each time an independent verifier's exhaustive leaf sweep
found more. Round 4's ``DECLARED_PATHS`` mechanism in
``test_contract_review_amend_r4.py`` is itself a curated whitelist, so it has
the same shape of bug: round 4's own verifier found 30 further
decision-bearing leaves (mostly under ``measurement`` and ``controls``) that
matched none of its keywords.

This file closes the whole class with one mechanism instead of one field at a
time: ``contract_frozen_snapshot.json`` is a full, committed, byte-for-byte
copy of ``contract_inputs.json`` taken at this head. Every leaf of the live
contract is compared against the frozen copy. Nothing is implicitly safe;
a path is only skipped if it appears in ``EXCLUDED_PATHS`` below, each with a
one-line reason. Any leaf not listed there -- present or future -- is
compared exactly, so a newly added field starts out covered instead of
starting out silently uncovered.

To make an *intentional* contract change under a non-excluded path, update
``contract_inputs.json`` and then regenerate the frozen copy with the one
command::

    cp experiments/atomistic/h-v2/contract_inputs.json \\
        experiments/atomistic/h-v2/contract_frozen_snapshot.json

and review the resulting diff of ``contract_frozen_snapshot.json`` in the
same commit -- that diff is the visible record of the decision changing.

``EXCLUDED_PATHS`` was re-examined again by the round-5 independent verifier
(see TO note ``tip-verification-6a0a6a53``), who found that 3 of the 7
round-5 exclusions (``study.*``, ``source_identity.*``, and the three
``measurement.seed_roster`` seed-array leaves) were themselves
decision-bearing and should not have been excluded: physical constants that
scientific controls are defined in terms of (``nuclear_charge``,
``exact_ground_energy_hartree``), checksums and commit identifiers that are
the evidentiary record for the He-v1 comparison (``source_identity.*``), and
pre-registered seeds whose whole point is to be fixed *before* results are
seen, not just mutually distinct (the seed arrays). Per the binding
round-6 decision (``contract-test-coverage-decision-20261106`` amendment on
item ``fffc313a``), only one exclusion survives this round:

- ``schema``: a format-version tag, not a decision -- no study parameter,
  control, provenance identifier, or seed value is ever read from it.

Every other path -- all of ``study.*`` (including ``name``, ``system``,
``excluded_systems``, ``nuclear_charge``, ``exact_ground_energy_hartree``),
the whole ``source_identity`` subtree, and the seed-roster arrays
(``training_initialization_seeds``, ``training_sampler_seeds``,
``non_science_seed_namespaces``) -- is now frozen exactly by
``test_frozen_snapshot_matches_live_contract_except_excluded_paths`` below,
alongside everything round 5 already covered
(``measurement.{estimand,primary_dispersion,secondary_dispersion,
energy_error_profile,complete_record_rule,fresh_evaluation}`` and
``controls``). The seed-array distinctness/disjointness tests in
``test_contract.py`` (``test_all_seed_namespaces_and_evaluation_seeds_are_
globally_distinct``, ``test_sampler_tail_qualification_seed_namespace_is_
disjoint``) are unaffected and still run: freezing the exact integers and
checking their distinctness are complementary, not substitutes -- the
frozen-snapshot comparison catches a silent re-roll to a different (still
distinct) set, which the distinctness tests alone cannot.
"""

import json
from pathlib import Path

import pytest

INPUTS = Path(__file__).with_name("contract_inputs.json")
FROZEN_SNAPSHOT = Path(__file__).with_name("contract_frozen_snapshot.json")

_MISSING = object()

# Each entry is a path prefix (tuple of dict keys / list indices) and a
# one-line reason. A leaf is excluded if its path starts with one of these
# prefixes. See the module docstring for the full reasoning per entry.
EXCLUDED_PATHS: list[tuple[tuple, str]] = [
    (("schema",), "format-version tag, not a decision"),
]


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_frozen_snapshot() -> dict:
    with FROZEN_SNAPSHOT.open(encoding="utf-8") as stream:
        return json.load(stream)


def _path_str(path: tuple) -> str:
    parts: list[str] = []
    for part in path:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        elif parts:
            parts.append(f".{part}")
        else:
            parts.append(str(part))
    return "".join(parts)


def _is_excluded(path: tuple, excluded_prefixes: list[tuple]) -> bool:
    return any(path[: len(prefix)] == prefix for prefix in excluded_prefixes)


def _iter_diffs(live: object, frozen: object, path: tuple, excluded_prefixes: list[tuple]):
    """Yield (path, live_value, frozen_value) for every leaf that differs.

    Excluded subtrees are pruned before comparison, so neither a value
    change nor a key added/removed underneath an excluded path is reported.
    """
    if _is_excluded(path, excluded_prefixes):
        return
    if isinstance(live, dict) or isinstance(frozen, dict):
        if not (isinstance(live, dict) and isinstance(frozen, dict)):
            yield path, live, frozen
            return
        for key in sorted(set(live) | set(frozen), key=str):
            child_path = path + (key,)
            if _is_excluded(child_path, excluded_prefixes):
                continue
            yield from _iter_diffs(
                live.get(key, _MISSING), frozen.get(key, _MISSING), child_path, excluded_prefixes
            )
    elif isinstance(live, list) or isinstance(frozen, list):
        if not (isinstance(live, list) and isinstance(frozen, list)):
            yield path, live, frozen
            return
        for index in range(max(len(live), len(frozen))):
            child_path = path + (index,)
            if _is_excluded(child_path, excluded_prefixes):
                continue
            live_item = live[index] if index < len(live) else _MISSING
            frozen_item = frozen[index] if index < len(frozen) else _MISSING
            yield from _iter_diffs(live_item, frozen_item, child_path, excluded_prefixes)
    else:
        if live != frozen or type(live) is not type(frozen):
            yield path, live, frozen


def test_excluded_paths_resolve_in_the_live_contract() -> None:
    """Every exclusion must name a real path, so a rename can't hide a gap."""
    inputs = load_inputs()
    for prefix, _reason in EXCLUDED_PATHS:
        cur = inputs
        for part in prefix:
            assert isinstance(cur, (dict, list)), f"excluded path {_path_str(prefix)!r} does not resolve"
            if isinstance(cur, list):
                assert isinstance(part, int) and part < len(cur), (
                    f"excluded path {_path_str(prefix)!r} does not resolve"
                )
                cur = cur[part]
            else:
                assert part in cur, f"excluded path {_path_str(prefix)!r} does not resolve"
                cur = cur[part]


def test_frozen_comparison_is_type_strict() -> None:
    diffs = list(_iter_diffs({"flag": 1}, {"flag": True}, (), []))
    assert diffs == [(("flag",), 1, True)]


def test_frozen_snapshot_matches_live_contract_except_excluded_paths() -> None:
    inputs = load_inputs()
    frozen = load_frozen_snapshot()
    excluded_prefixes = [prefix for prefix, _reason in EXCLUDED_PATHS]
    diffs = list(_iter_diffs(inputs, frozen, (), excluded_prefixes))
    if diffs:
        lines = [
            f"  {_path_str(path)}: live={live!r} frozen={frozen_value!r}"
            for path, live, frozen_value in diffs
        ]
        pytest.fail(  # noqa: PT011 - custom multi-line message naming every changed path
            "contract_inputs.json has drifted from contract_frozen_snapshot.json "
            "at the following non-excluded path(s); if this is an intentional "
            "contract change, update the frozen copy with:\n"
            "  cp experiments/atomistic/h-v2/contract_inputs.json "
            "experiments/atomistic/h-v2/contract_frozen_snapshot.json\n"
            + "\n".join(lines)
        )
