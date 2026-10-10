"""The evidence schema: durable, versioned, and honest about its exit status.

A receipt is the only thing that survives an attempt. Three properties are
pinned here because losing any one of them has cost this repository a result:

* it is written **durably** -- flushed, ``fsync``-ed, and committed by a rename
  -- so a reader never sees a half-written receipt and a killed process does not
  turn a recorded outcome into silence;
* it is **versioned**, and a reader refuses an unknown schema rather than
  misreading fields that happen to line up;
* it records whether its exit status was **EARNED or ENGINEERED**, because a
  harness that deliberately exits 0 when failures are the datum produces a green
  that means the opposite of what it looks like.
"""

from __future__ import annotations

import json

import pytest

from tests.helpers.chain_resume_spike.fixture import spawn_attempt
from tests.helpers.chain_resume_spike.identity import new_attempt
from tests.helpers.chain_resume_spike.outcomes import AttemptOutcome, OutcomeKind
from tests.helpers.chain_resume_spike.receipt import (
    ATTEMPT_RECEIPT_SCHEMA,
    AttemptReceipt,
    ExitKind,
    read_receipt,
    write_receipt,
)


def _receipt(**overrides) -> AttemptReceipt:
    """Build a minimal receipt, overridable field by field."""

    base = dict(
        identity=new_attempt("run", 1, 4),
        outcome=AttemptOutcome(
            kind=OutcomeKind.YIELD,
            reason="step_budget_reached",
            credited_through=5,
            next_parent_generation=4,
        ),
        process_seed=123456789,
        pid=4242,
        executable="/abs/path/to/python",
        trace=[{"step": 4, "proposal": [0.5]}],
        credited_steps=(4, 5),
        replayed_steps=(),
        disabled_limbs=(),
        limb_application=None,
        fault=None,
        exit_kind=ExitKind.EARNED,
    )
    base.update(overrides)
    return AttemptReceipt(**base)  # type: ignore[arg-type]


def test_a_receipt_round_trips_through_disk(tmp_path) -> None:
    """Every field survives the write and the read."""

    receipt = _receipt()
    path = tmp_path / "receipt.json"
    write_receipt(path, receipt)
    assert read_receipt(path) == receipt


def test_the_write_leaves_no_temporary_file_behind(tmp_path) -> None:
    """The rename is the commit; the ``.tmp`` must not survive it."""

    path = tmp_path / "receipt.json"
    write_receipt(path, _receipt())
    assert path.is_file()
    assert not (tmp_path / "receipt.json.tmp").exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["receipt.json"]


def test_the_committed_receipt_parses_as_one_whole_document(tmp_path) -> None:
    """The committed file is a complete document, read once after the write.

    The write goes to a sibling and renames, the same discipline
    ``tpen.checkpoint.save`` uses for a checkpoint directory, and the
    committed file parses as strict JSON: a truncated write would not.

    WHAT IT DOES NOT ESTABLISH, and the earlier name asserted otherwise: **A
    TORN READ IS NOT EXCLUDED.** This node constructs no partial state and
    runs no concurrent reader, so it never looks while a write is in flight.
    It samples one instant after the rename. The rename is what makes a torn
    read unreachable; this node observes the rename's RESULT and does not
    exercise the window the claim is about. Found by the round-5 claims sweep,
    in a file nobody had flagged.
    """

    path = tmp_path / "receipt.json"
    write_receipt(path, _receipt())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == ATTEMPT_RECEIPT_SCHEMA


def test_an_unknown_schema_is_refused_rather_than_misread(tmp_path) -> None:
    """Version the evidence, or a future reader silently reinterprets it."""

    path = tmp_path / "receipt.json"
    write_receipt(path, _receipt())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = "tpen.chain-resume-spike.attempt-receipt/v99"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported attempt receipt schema"):
        read_receipt(path)


def test_the_exit_kind_distinguishes_an_earned_status_from_an_arranged_one() -> None:
    """Both values must round-trip; neither may be the silent default."""

    for kind in ExitKind:
        receipt = _receipt(exit_kind=kind)
        assert AttemptReceipt.from_mapping(receipt.to_dict()).exit_kind is kind


def test_a_real_attempt_records_its_seed_pid_and_absolute_interpreter(tmp_path) -> None:
    """The provenance fields are populated on the real path, not just in a fixture.

    Each of the three answers a question a later reader cannot otherwise
    settle: which entropy this attempt started from, which OS process it was,
    and which interpreter produced the numbers.
    """

    launch = spawn_attempt(
        tmp_path / "attempt",
        root=tmp_path / "root",
        run_id="provenance",
        attempt_index=0,
        steps=2,
        total_target=2,
        checkpoint_every=2,
    )
    assert launch.exit_code == 0, launch.log_path.read_text(encoding="utf-8")

    receipt = read_receipt(launch.receipt_path)
    assert receipt.process_seed > 0
    assert receipt.pid > 0
    assert receipt.executable.startswith("/"), (
        f"the attempt recorded a non-absolute interpreter: {receipt.executable!r}"
    )
    assert receipt.exit_kind is ExitKind.EARNED
    assert receipt.outcome.kind is OutcomeKind.FINAL
    assert receipt.trace, "the receipt carries no observable trace to compare"


def test_a_receipt_survives_a_raising_attempt_and_reports_it_transient(
    tmp_path,
) -> None:
    """A handled failure records itself; the exit status it reports is EARNED.

    This is the contrast case for the non-unwinding deaths in
    ``test_generation_preservation.py``, which write no receipt at all. The
    difference between "a receipt saying it failed" and "no receipt" is the
    difference between an error and a death, and both are outcomes a chain
    controller has to tell apart.
    """

    from tests.helpers.chain_resume_spike.faults import (
        FaultAction,
        FaultPlan,
        FaultPoint,
    )

    launch = spawn_attempt(
        tmp_path / "attempt",
        root=tmp_path / "root",
        run_id="raiser",
        attempt_index=0,
        steps=2,
        total_target=4,
        checkpoint_every=2,
        fault=FaultPlan(FaultPoint.AFTER_MANIFEST_BEFORE_COMPLETE, FaultAction.RAISE),
    )
    assert launch.exit_code == 1

    receipt = read_receipt(launch.receipt_path)
    assert receipt.outcome.kind is OutcomeKind.TRANSIENT
    assert receipt.outcome.consumes_retry_budget
    assert not receipt.outcome.is_scientific_success
    assert receipt.exit_kind is ExitKind.EARNED
    assert receipt.fault is not None
    assert receipt.fault["point"] == FaultPoint.AFTER_MANIFEST_BEFORE_COMPLETE.value
