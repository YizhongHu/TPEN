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
from pathlib import Path

import pytest

from tests.helpers.chain_resume_spike.faults import (
    FaultAction,
    FaultPlan,
    FaultPoint,
)
from tests.helpers.chain_resume_spike.fixture import (
    pointer_target_name,
    read_generation_provenance,
    resume_generation,
    spawn_attempt,
)
from tests.helpers.chain_resume_spike.parity import first_divergence
from tpen.checkpoint.artifact import is_complete_checkpoint_dir
from tpen.checkpoint.catalog import reconcile_publication
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


# ----------------------------------------------------------------------
# G8 under OVERLAPPING committed generations. Fresh processes throughout.
# ----------------------------------------------------------------------

OVERLAP_TARGET = 6
OVERLAP_EVERY = 2
#: Cold-start seed shared by the reference run and the chain's first link, so
#: the two arms have the SAME initial condition. Only the cold start is
#: pinned; every resumed link still draws its own OS entropy, which is what
#: keeps the parity comparison non-vacuous.
OVERLAP_COLD_START_SEED = 20_260_909


def _drive_to_the_overlapping_state(tmp_path) -> tuple[Path, object, object]:
    """Build the two-overlapping-generations state, in fresh processes.

    Link 1 commits generation 1 cleanly. Link 2 is interrupted in the
    ``after_rename_before_catalog`` window, so generation 2 lands on disk
    complete, carrying credit for steps 0-3, while ``latest.json`` still names
    generation 1, which carries credit for 0-1 only.
    """

    root = tmp_path / "root"
    first = spawn_attempt(
        tmp_path / "a0", root=root, run_id="overlap", attempt_index=0,
        steps=OVERLAP_EVERY, total_target=OVERLAP_TARGET, checkpoint_every=OVERLAP_EVERY,
        seed=OVERLAP_COLD_START_SEED,
    )
    assert first.exit_code == 0, first.log_path.read_text(encoding="utf-8")
    first_receipt = read_receipt(first.receipt_path)
    assert first_receipt.outcome.kind is OutcomeKind.YIELD
    assert first_receipt.credited_steps == (0, 1)

    second = spawn_attempt(
        tmp_path / "a1", root=root, run_id="overlap", attempt_index=1,
        steps=OVERLAP_EVERY, total_target=OVERLAP_TARGET, checkpoint_every=OVERLAP_EVERY,
        fault=FaultPlan(FaultPoint.AFTER_RENAME_BEFORE_CATALOG, FaultAction.RAISE),
    )
    assert second.exit_code == 1, "the injected fault did not fail the attempt"
    second_receipt = read_receipt(second.receipt_path)
    assert second_receipt.outcome.kind is OutcomeKind.TRANSIENT

    assert is_complete_checkpoint_dir(root / "step_000002")
    assert is_complete_checkpoint_dir(root / "step_000004")
    assert pointer_target_name(root) == "step_000002", (
        "the pointer moved, so no overlap exists and these tests are not testing it"
    )
    assert read_generation_provenance(root / "step_000004")["credited_steps"] == [
        0, 1, 2, 3
    ], "generation 2 does not claim overlapping progress"
    return root, first_receipt, second_receipt


def test_an_orphaned_post_rename_generation_deadlocks_the_chain(tmp_path) -> None:
    """DISCOVERED HAZARD, PINNED AND NOT FIXED. The chain cannot continue.

    After an interruption between the rename and the catalog append, resume does
    the right thing at every individual step and still cannot make progress:

      * it follows ``latest.json`` to generation 1, correctly;
      * it replays steps 2-3, correctly, reproducing the same stream;
      * and it then tries to commit generation 2 -- whose directory ALREADY
        EXISTS, because the interrupted attempt renamed it into place.

    ``save_checkpoint`` refuses with ``FileExistsError: checkpoint already
    exists`` (save.py:142-143). The replica mirrors that guard line for line, so
    the refusal observed here is production's own rule, not the replica's.

    THE DIRECTION IS FAIL-CLOSED, which is safe -- loud, no corruption, nothing
    overwritten -- but a chain that cannot advance a single further generation
    is exactly the failure this R0-R4 program exists to prevent. The recovery is
    documented and is exercised by the next test; what is NOT present is
    anything that performs it automatically.

    THIS STATE DEFEATS ITS OWN OWNER'S ACCEPTANCE PREDICATES, which is why the
    disclosure is worth more than the deadlock alone. ``3b9b736a``'s layer-3
    acceptance experiment asks whether any ``step_`` directory lacks ``COMPLETE``
    or ``manifest.json``, and whether any ``.tmp`` residue survives. BOTH PASS
    HERE: ``save.py`` writes the manifest (209) and the marker (210) BEFORE the
    rename (211), so the orphan is a fully formed directory carrying both, and
    no ``.tmp`` exists because the tmp directory WAS what got renamed. The
    predicates look for a malformed directory and for residue; this is neither.
    Nor does that item's ``SIGTERM``-unwinding remedy narrow this class: the
    ``finally`` at save.py:245 removes ``tmp_dir``, and post-rename ``tmp_dir``
    does not exist -- save.py:252 says so itself.

    NOT FIXED HERE. ``tpen/checkpoint`` is outside this lane's write surface.
    Attributed to the ``tpen/checkpoint`` owner and to ``3b9b736a``
    (interruption safety), whose open scope is scheduler termination and real
    storage commit. Its layer-1 fix is landed and is not implicated.
    """

    root, _, _ = _drive_to_the_overlapping_state(tmp_path)

    third = spawn_attempt(
        tmp_path / "a2", root=root, run_id="overlap", attempt_index=2,
        steps=OVERLAP_TARGET, total_target=OVERLAP_TARGET, checkpoint_every=OVERLAP_EVERY,
    )

    assert third.exit_code == 1, (
        "the resumed attempt committed a generation whose directory already "
        "existed; the collision guard is gone"
    )
    log = third.log_path.read_text(encoding="utf-8")
    assert "FileExistsError" in log and "checkpoint already exists" in log, (
        f"the attempt failed for some other reason:\n{log}"
    )
    assert "step_000004" in log, "the collision was not on the orphaned generation"

    # Nothing was damaged: both generations survive and the pointer is unmoved.
    assert is_complete_checkpoint_dir(root / "step_000002")
    assert is_complete_checkpoint_dir(root / "step_000004")
    assert pointer_target_name(root) == "step_000002"


def test_reconciling_the_orphan_lets_the_chain_finish_and_credit_once(
    tmp_path,
) -> None:
    """The documented recovery, and the real G8 property once it is applied.

    ``reconcile_publication`` is production's repair for a committed-but-
    unacknowledged generation: it appends the missing catalog row, advances
    ``latest.json`` to that directory, and backfills the receipt
    (catalog.py:189-208). Applied to the orphan, it turns the deadlock above
    into a chain that finishes.

    Two things must then hold at once, and either is satisfiable while the other
    fails -- a chain can credit the right COUNT of the wrong steps, or replay
    correctly and count the tail twice:

      * the durable credited ledger totals the fixed target EXACTLY once, and
      * the credited trace equals the uninterrupted trace, step for step.

    The previous version of this test asserted neither: it sliced two entries
    off a receipt by hand and compared sets, which checks the arithmetic the
    test itself just performed.
    """

    reference_launch = spawn_attempt(
        tmp_path / "reference",
        root=tmp_path / "reference-root",
        run_id="overlap-reference",
        attempt_index=0,
        steps=OVERLAP_TARGET,
        total_target=OVERLAP_TARGET,
        checkpoint_every=OVERLAP_EVERY,
        seed=OVERLAP_COLD_START_SEED,
    )
    assert reference_launch.exit_code == 0, reference_launch.log_path.read_text(
        encoding="utf-8"
    )
    reference = read_receipt(reference_launch.receipt_path)
    assert reference.outcome.kind is OutcomeKind.FINAL

    root, first_receipt, second_receipt = _drive_to_the_overlapping_state(tmp_path)

    # THE RECOVERY. Reconciling the NEWEST complete directory is what
    # catalog.py's own torn-row message prescribes; reconciling an older one
    # would rewind the pointer, which the next test pins.
    reconcile_publication(root, root / "step_000004")
    assert pointer_target_name(root) == "step_000004"

    third = spawn_attempt(
        tmp_path / "a2", root=root, run_id="overlap", attempt_index=2,
        steps=OVERLAP_TARGET, total_target=OVERLAP_TARGET, checkpoint_every=OVERLAP_EVERY,
    )
    assert third.exit_code == 0, third.log_path.read_text(encoding="utf-8")
    third_receipt = read_receipt(third.receipt_path)
    assert third_receipt.outcome.kind is OutcomeKind.FINAL
    assert third_receipt.identity.parent_generation == 4

    # (1) EXACTLY ONCE, read from the durable record rather than reassembled here.
    durable_credit = read_generation_provenance(resume_generation(root))["credited_steps"]
    assert durable_credit == list(range(OVERLAP_TARGET)), (
        f"the chain's durable credited set is {durable_credit}, not the fixed target"
    )
    assert len(durable_credit) == len(set(durable_credit)), "a step was credited twice"

    # The tail was EXECUTED twice across the chain and CREDITED once.
    assert {row["step"] for row in second_receipt.trace} == {2, 3}
    assert set(third_receipt.credited_steps) == {4, 5}
    assert set(first_receipt.credited_steps).isdisjoint(second_receipt.credited_steps)
    assert set(second_receipt.credited_steps).isdisjoint(third_receipt.credited_steps)

    # (2) THE CREDITED TRACE EQUALS THE UNINTERRUPTED TRACE, step for step. The
    # tail is contributed by the attempt whose generation the chain adopted.
    credited_rows = (
        [row for row in first_receipt.trace if row["step"] in first_receipt.credited_steps]
        + [row for row in second_receipt.trace if row["step"] in second_receipt.credited_steps]
        + [row for row in third_receipt.trace if row["step"] in third_receipt.credited_steps]
    )
    assert [row["step"] for row in credited_rows] == list(range(OVERLAP_TARGET))
    divergence = first_divergence(reference.trace, credited_rows)
    assert divergence is None, (
        "the credited trace is not the uninterrupted trajectory: "
        + divergence.describe()
    )


def test_reconciling_an_older_generation_rewinds_the_resume_pointer(tmp_path) -> None:
    """REWIND HAZARD, PINNED FOR DIRECTION AND NOT FIXED.

    ``reconcile_publication`` builds its expected pointer from the directory it
    is HANDED (catalog.py:216-233), compares ``read_latest`` against it, and on
    any mismatch writes the pointer at THAT directory -- with no monotonicity
    guard on step. Handed an older generation while ``latest.json`` names a
    newer one, it moves the pointer BACKWARDS. Since resume follows the pointer,
    that means re-running committed science.

    THIS IS NOT A DEFECT IN CURRENT USAGE: today's only caller reaches it with
    the newest directory, so the hazard is held off by an accident of usage
    rather than by a guard. THE TRIGGER IS THE USAGE PATTERN THIS PROGRAM IS
    EVALUATING -- a chain controller that restarts and reconciles the generation
    it REMEMBERS handing off can hand it an older one and silently rewind past a
    newer committed generation.

    The subsystem documents the hazard about itself: the torn-row repair message
    in ``iter_publications`` warns that ``reconcile_publication`` rewrites
    ``latest.json`` unconditionally and would point it at an older checkpoint,
    and tells operators not to reconcile older directories "to be safe".

    Non-destructive: this asserts the DIRECTION and checks both payloads are
    untouched. No monotonicity guard is added to ``tpen/``.
    """

    root, _, _ = _drive_to_the_overlapping_state(tmp_path)
    reconcile_publication(root, root / "step_000004")
    assert pointer_target_name(root) == "step_000004"

    def _payloads(name: str) -> dict[str, bytes]:
        return {
            path.name: path.read_bytes()
            for path in sorted((root / name).iterdir())
            if path.is_file()
        }

    before = {"step_000002": _payloads("step_000002"), "step_000004": _payloads("step_000004")}

    # Hand it the OLDER generation while the pointer names the newer one.
    reconcile_publication(root, root / "step_000002")

    assert pointer_target_name(root) == "step_000002", (
        "reconcile_publication no longer rewinds; if a monotonicity guard was "
        "added upstream, retire this pin rather than weakening it"
    )
    assert resume_generation(root).name == "step_000002", (
        "resume follows the pointer, so the chain would now re-run committed steps"
    )
    after = {"step_000002": _payloads("step_000002"), "step_000004": _payloads("step_000004")}
    assert before == after, "the rewind altered committed checkpoint payloads"
