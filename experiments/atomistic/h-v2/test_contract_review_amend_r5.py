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

``EXCLUDED_PATHS`` re-examines (not copies) the round-4 verifier's 55
"justified" survivors from its exhaustive 599-leaf sweep. Anything that could
change a scientific or admission decision was kept IN (covered by this
test); only true free prose/provenance/metadata, or fields already protected
by an exact-value test elsewhere, is excluded:

- ``schema``: a format-version tag, not a decision.
- ``study.{name,system,nuclear_charge,exact_ground_energy_hartree,
  excluded_systems}``: physical constants and labels fixed by the choice of
  system (H-v2 is atomic hydrogen, Z=1, exact ground energy -0.5 Ha by
  definition), not an operator decision. ``study.electron_count`` is left
  OUT of this exclusion (it stays covered) because the round-4 contract
  already treats it as decision-bearing.
- ``source_identity.*``: historical provenance pointers (commit hashes,
  comparison paths, prior-run identifiers). Nothing computational reads
  this subtree; confirmed by grep across every ``test_contract*.py`` file.
- ``measurement.seed_roster.training_initialization_seeds``,
  ``.training_sampler_seeds``, ``.non_science_seed_namespaces``: the actual
  contract for these is mutual distinctness and disjointness from every
  other seed namespace, enforced by
  ``test_all_seed_namespaces_and_evaluation_seeds_are_globally_distinct`` and
  ``test_sampler_tail_qualification_seed_namespace_is_disjoint`` in
  ``test_contract.py``. Any mutually-distinct set of integers is equally
  valid scientifically, so pinning the exact integers here would freeze an
  arbitrary choice, not a decision. ``measurement.seed_roster
  .evaluation_stream_seed_formula`` and ``.disjointness`` are deliberately
  left OUT of this exclusion (they stay covered by the frozen-snapshot
  comparison below): the formula's numeric pattern is checked by
  ``_evaluation_seeds`` via a regex, but only the matched prefix -- the
  round-4 verifier found its own append-only mutation to the trailing
  descriptive text survived, since that text isn't pinned anywhere. Covering
  the full string here closes that gap at zero cost.

Everything else that the round-4 verifier's sweep found uncovered --
``measurement.{estimand,primary_dispersion,secondary_dispersion,
energy_error_profile,complete_record_rule,fresh_evaluation}`` and
``controls`` -- is *not* excluded, and is therefore now caught by
``test_frozen_snapshot_matches_live_contract_except_excluded_paths`` below.
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
    (("study", "name"), "physical label of the fixed system, not a decision"),
    (("study", "system"), "physical label of the fixed system, not a decision"),
    (("study", "nuclear_charge"), "fixed by the choice of system (Z=1 for atomic hydrogen)"),
    (
        ("study", "exact_ground_energy_hartree"),
        "fixed physical constant for atomic hydrogen, not an operator decision",
    ),
    (("study", "excluded_systems"), "labels of systems explicitly out of scope, not a decision"),
    (
        ("source_identity",),
        "historical provenance pointers; nothing computational reads this subtree",
    ),
    (
        ("measurement", "seed_roster", "training_initialization_seeds"),
        "contract is mutual distinctness/disjointness, not the exact integers; "
        "enforced by test_all_seed_namespaces_and_evaluation_seeds_are_globally_distinct",
    ),
    (
        ("measurement", "seed_roster", "training_sampler_seeds"),
        "contract is mutual distinctness/disjointness, not the exact integers; "
        "enforced by test_all_seed_namespaces_and_evaluation_seeds_are_globally_distinct",
    ),
    (
        ("measurement", "seed_roster", "non_science_seed_namespaces"),
        "contract is mutual distinctness/disjointness, not the exact integers; "
        "enforced by test_all_seed_namespaces_and_evaluation_seeds_are_globally_distinct "
        "and test_sampler_tail_qualification_seed_namespace_is_disjoint",
    ),
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
        if live != frozen:
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
