"""G1 and G8: typed continuation outcomes, and credited work counted once.

Two failure modes are pinned here, and they pull in opposite directions.

A chain that treats a **yield as success** stops short of its total target
while reporting green. A chain that treats a **yield as a failure retry** spends
its bounded transient quota on its own normal operation and eliminates itself
over a long run. Both are avoided by making the four outcomes a closed enum and
deriving both policy questions from the kind.

A chain that **re-credits a replayed tail** reports the right total while having
done the wrong amount of work. ``CreditedLedger`` refuses a duplicate rather
than absorbing it, so the failure is loud at the moment it happens.
"""

from __future__ import annotations

import json

import pytest

from tests.helpers.chain_resume_spike.fixture import spawn_attempt
from tests.helpers.chain_resume_spike.identity import (
    ChainIdentity,
    CreditedLedger,
    DoubleCreditError,
    new_attempt,
)
from tests.helpers.chain_resume_spike.outcomes import (
    CONTINUABLE_KINDS,
    TERMINAL_KINDS,
    AttemptOutcome,
    OutcomeKind,
)
from tests.helpers.chain_resume_spike.receipt import read_receipt


def test_every_outcome_kind_is_classified_exactly_once() -> None:
    """The two classifications partition the enum.

    Asserted rather than assumed so that adding a fifth kind cannot leave it
    falling through both sets, silently continuable and terminal at once.
    """

    assert CONTINUABLE_KINDS | TERMINAL_KINDS == set(OutcomeKind)
    assert not (CONTINUABLE_KINDS & TERMINAL_KINDS)


def test_a_yield_is_neither_success_nor_a_retry() -> None:
    """The G1 distinction, in both directions."""

    outcome = AttemptOutcome(
        kind=OutcomeKind.YIELD,
        reason="step_budget_reached",
        credited_through=3,
        next_parent_generation=4,
    )
    assert outcome.is_continuable
    assert not outcome.is_scientific_success, "a yield reported as scientific success"
    assert not outcome.consumes_retry_budget, (
        "ordinary walltime continuation is spending the transient-failure quota"
    )


def test_only_a_transient_outcome_spends_the_retry_budget() -> None:
    """Exhaustively, across the closed enum."""

    spends = {
        kind
        for kind in OutcomeKind
        if AttemptOutcome(
            kind=kind, reason="r", credited_through=None, next_parent_generation=None
        ).consumes_retry_budget
    }
    assert spends == {OutcomeKind.TRANSIENT}


def test_only_a_final_outcome_is_scientific_success() -> None:
    """Exhaustively, across the closed enum."""

    succeeds = {
        kind
        for kind in OutcomeKind
        if AttemptOutcome(
            kind=kind, reason="r", credited_through=None, next_parent_generation=None
        ).is_scientific_success
    }
    assert succeeds == {OutcomeKind.FINAL}


def test_an_outcome_round_trips_through_its_serialized_form() -> None:
    """A yield must still be a yield after crossing a process boundary."""

    for kind in OutcomeKind:
        outcome = AttemptOutcome(
            kind=kind, reason="reason", credited_through=2, next_parent_generation=2
        )
        assert AttemptOutcome.from_mapping(json.loads(json.dumps(outcome.to_dict()))) == outcome


def test_the_ledger_refuses_to_credit_a_step_twice() -> None:
    """The double-count guard is loud, not absorbing."""

    ledger = CreditedLedger(total_target=4)
    ledger.credit_all((0, 1, 2))
    with pytest.raises(DoubleCreditError, match="already credited"):
        ledger.credit(2)
    assert ledger.credited_steps == (0, 1, 2)
    assert not ledger.is_complete
    assert ledger.missing_steps() == (3,)


def test_the_ledger_refuses_a_step_outside_the_fixed_target() -> None:
    """A fixed total target means a step beyond it is an accounting error."""

    ledger = CreditedLedger(total_target=3)
    with pytest.raises(ValueError, match="outside the fixed target range"):
        ledger.credit(3)
    with pytest.raises(ValueError, match="outside the fixed target range"):
        ledger.credit(-1)


def test_completeness_requires_the_whole_range_not_merely_the_count() -> None:
    """Three credited steps out of three is not the same as steps 0, 1 and 2.

    A count-based completeness check would call ``{0, 1, 1}`` complete if it
    ever got past the duplicate guard, and would call ``{0, 1, 3}`` complete
    outright. Both are gaps in the credited trace.
    """

    ledger = CreditedLedger(total_target=3)
    ledger.credit_all((0, 2))
    assert ledger.credited_count == 2
    assert not ledger.is_complete
    ledger.credit(1)
    assert ledger.is_complete


def test_identity_keeps_the_run_stable_while_the_attempt_changes() -> None:
    """One logical run, many attempts, an explicit parent each time."""

    first = new_attempt("run-a", 0, None)
    second = new_attempt("run-a", 1, 4)
    assert first.run_id == second.run_id
    assert first.attempt_id != second.attempt_id
    assert first.is_cold_start and not second.is_cold_start
    assert second.parent_generation == 4
    assert ChainIdentity.from_mapping(second.to_dict()) == second


def test_a_chain_interrupted_mid_link_credits_the_total_exactly_once(tmp_path) -> None:
    """G8 end to end: the interrupted tail is re-executed, not re-credited.

    Attempt 0 commits a generation at step 2 and then executes step 2 without
    committing it, so that credit dies with the process. Attempt 1 restores
    generation 2, re-executes step 2, and credits it for the first time. The
    union across the chain is exactly the fixed target, with no duplicate.
    """

    root = tmp_path / "root"
    total = 6

    first = spawn_attempt(
        tmp_path / "a0", root=root, run_id="ledger", attempt_index=0,
        steps=3, total_target=total, checkpoint_every=2,
    )
    assert first.exit_code == 0, first.log_path.read_text(encoding="utf-8")
    first_receipt = read_receipt(first.receipt_path)
    assert first_receipt.outcome.kind is OutcomeKind.YIELD

    second = spawn_attempt(
        tmp_path / "a1", root=root, run_id="ledger", attempt_index=1,
        steps=total, total_target=total, checkpoint_every=2,
    )
    assert second.exit_code == 0, second.log_path.read_text(encoding="utf-8")
    second_receipt = read_receipt(second.receipt_path)
    assert second_receipt.outcome.kind is OutcomeKind.FINAL

    # Durably credited work is what the committed generations carry, so the
    # chain's accounting is the union of what survived, not of what ran.
    durable = set(first_receipt.credited_steps[:2]) | set(second_receipt.credited_steps)
    assert sorted(durable) == list(range(total))
    overlap = set(first_receipt.credited_steps[:2]) & set(second_receipt.credited_steps)
    assert not overlap, f"steps credited by both attempts: {sorted(overlap)}"

    # Step 2 was EXECUTED by both attempts and CREDITED by exactly one.
    assert 2 in {row["step"] for row in first_receipt.trace}
    assert 2 in {row["step"] for row in second_receipt.trace}
    assert 2 in second_receipt.credited_steps


def test_a_step_already_credited_by_its_parent_is_replayed_not_recredited(
    tmp_path,
) -> None:
    """The replay branch, exercised rather than left as a field that never fills.

    Constructed by committing a generation whose recorded ``credited_steps``
    reach past its own step -- the shape a link leaves when it credited work in
    memory beyond its last commit. The resumed attempt re-executes those steps
    and must route them to ``replayed_steps`` instead of crediting them again.
    Without this arm the branch would never run, and a field that can never
    fill is not a detector.
    """

    root = tmp_path / "root"
    total = 6

    first = spawn_attempt(
        tmp_path / "a0", root=root, run_id="replay", attempt_index=0,
        steps=2, total_target=total, checkpoint_every=2,
    )
    assert first.exit_code == 0, first.log_path.read_text(encoding="utf-8")

    # Rewrite the committed generation's provenance so it claims credit for
    # steps 2 and 3 as well. The payload bytes are untouched.
    manifest_path = root / "step_000002" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["credited_steps"] = [0, 1, 2, 3]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    second = spawn_attempt(
        tmp_path / "a1", root=root, run_id="replay", attempt_index=1,
        steps=total, total_target=total, checkpoint_every=2,
        restore_from=root / "step_000002",
    )
    assert second.exit_code == 0, second.log_path.read_text(encoding="utf-8")
    receipt = read_receipt(second.receipt_path)

    assert receipt.replayed_steps == (2, 3), (
        "steps the parent already credited were not routed to the replay branch"
    )
    assert receipt.credited_steps == (4, 5)
    assert set(receipt.replayed_steps) & set(receipt.credited_steps) == set()
    assert receipt.outcome.kind is OutcomeKind.FINAL
