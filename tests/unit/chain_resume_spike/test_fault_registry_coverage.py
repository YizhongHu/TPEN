"""Every registered fault point is REACHED when injected. REACHABILITY, NOT EFFECT.

## WHAT THIS MODULE ESTABLISHES, AND WHAT IT DOES NOT

It establishes that **AN INJECTION WAS REQUESTED UNDER THAT IDENTITY**. It does
NOT establish effect completion, and it does NOT establish the *site*: the
record derives from the fault PLAN, not from where the injection landed.

The earlier wording here said "injection-site entry", which is too strong --
*entry* implies the site, and the site is precisely what is unobserved. The
claim is re-scoped rather than annotated, because an artefact that says
"exercised" while proving only that a request was made is a false claim in the
thing downstream lanes read.

MEASURED, not argued: a mutant that keeps ``record_fire(fault.point)`` and
RETURNS instead of raising ``InjectedFault`` passes every node in this module.
That mutant causes no damage at all. So these nodes are a reachability
instrument and are named as one.

## WHERE EFFECT DELIVERY *IS* ESTABLISHED

For the seven RAISE-boundary points, by these nodes in
``test_generation_preservation.py`` -- all inside the standard suite, and all
measured to FAIL under that same no-effect mutant because ``raised`` was
``None``:

* ``test_a_precommit_fault_leaves_generation_one_selectable`` for
  ``during_payload_write``, ``after_payload_before_manifest``,
  ``after_manifest_before_complete``, ``after_complete_before_rename``
* ``test_a_committed_but_unacknowledged_generation_is_valid_and_reconcilable``
  for ``after_rename_before_catalog``, ``after_catalog_before_latest``,
  ``after_latest_before_receipt``

A downstream reader wanting effect evidence for a point should follow those
node ids, not this module.

## THREE NAMED LIMITS

**1. BOUNDARY ATTRIBUTION IS NOT OBSERVED, AND AMONG THE PRECOMMIT POINTS IT IS
UNACHIEVABLE UNDER ``RAISE``.** Not merely unmeasured -- unachievable, and the
reason is structural. Every point here is driven with ``FaultAction.RAISE``. The
four precommit points all abort before the rename (save.py:211), and the
replicated ``finally`` (save.py:245) removes ``tmp_dir`` on unwind, so all four
leave an **identical end state**: no committed generation and no residue. No
observation after the fact can tell them apart. Distinguishing them would need a
non-unwinding action, which this module does not use.

**2. GENERATION ATTRIBUTION IS NOT OBSERVED.** Tearing an EARLIER generation's
catalog row while the final generation's row stands passes every node here.
Nothing in this module binds a recorded point to the generation it damaged.

**3. ``TORN_CATALOG_ROW``'s EFFECT DELIVERY IS UNMEASURED**, and this is now
confirmed by a measured mutant rather than only disclosed: suppressing the
actual torn-row write while leaving the recorder and ``RAISE`` intact passes all
of this module's nodes and the whole unit arm. The suppressed-RAISE mutant
cannot reach that point either --
``test_a_torn_final_catalog_row_is_diagnosed_and_repairable`` truncates the
catalog directly rather than injecting through ``FaultAction.RAISE``, so that
mutant is INAPPLICABLE to it rather than defeated by it.

``MEASURED_POINTS`` has **EIGHT** members: the seven RAISE-boundary points, all
of which have effect evidence in the nodes named above, plus
``TORN_CATALOG_ROW``, which has none.

## A RESIDUAL IN NAMING, DISCLOSED RATHER THAN FIXED

The constant is still called ``MEASURED_POINTS`` in ``faults.py``. That name
overstates what this module proves, and it is left alone deliberately:
``faults.py`` sits inside the verified surface of the torch-free import clause,
and renaming it there would void that clause's independent PASS to fix a word.
Read ``MEASURED_POINTS`` as "points this lane registers and reaches", and take
effect evidence from the nodes named above.

The mechanism itself is sound for what it now claims: a record written and
``fsync``-ed at the moment of injection cannot be produced by a mere mention,
and is produced identically by an equivalent set alias -- the two defects that
sank the previous source-scanning version.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.chain_resume_spike.faults import (
    COMMITTED_BUT_UNACKNOWLEDGED_POINTS,
    FIRE_LOG_ENV,
    INSTRUMENT_ONLY_POINTS,
    MEASURED_POINTS,
    FaultAction,
    FaultPlan,
    FaultPoint,
    InjectedFault,
    fired_points,
    record_fire,
)
from tests.helpers.chain_resume_spike.fixture import (
    ContinuationSystem,
    publish_generation,
)
from tests.helpers.chain_resume_spike.identity import new_attempt

CHECKPOINT_EVERY = 2


def _drive(point: FaultPoint, root: Path) -> None:
    """Actually inject ``point`` into a real publish sequence.

    Generation 1 is committed cleanly first, so that every post-commit point has
    a predecessor to be interrupted after, and so the torn-row point has a
    catalog with more than one row to tear.
    """

    system = ContinuationSystem.fresh(seed=2718)
    identity = new_attempt("coverage", 0, None)
    system.run(CHECKPOINT_EVERY)
    publish_generation(root, system, identity, credited_steps=(0, 1))
    system.run(CHECKPOINT_EVERY)
    try:
        publish_generation(
            root,
            system,
            identity,
            credited_steps=(0, 1, 2, 3),
            fault=FaultPlan(point, FaultAction.RAISE),
        )
    except InjectedFault:
        # Expected for every point that aborts the sequence. TORN_CATALOG_ROW
        # damages the index after a clean commit and therefore does not raise.
        pass


def test_the_registry_is_partitioned_with_nothing_left_over() -> None:
    """Measured and instrument-only must cover the enum, disjointly.

    A point in neither set would be silently uncovered AND silently undeclared:
    no test would drive it and no document would admit that.
    """

    assert MEASURED_POINTS | INSTRUMENT_ONLY_POINTS == set(FaultPoint)
    assert not (MEASURED_POINTS & INSTRUMENT_ONLY_POINTS)
    assert COMMITTED_BUT_UNACKNOWLEDGED_POINTS <= MEASURED_POINTS


def test_the_fire_recorder_distinguishes_fired_from_not_fired(
    tmp_path, monkeypatch
) -> None:
    """INSTRUMENT CHECK, run before the coverage assertion is believed.

    Both directions. A recorder that recorded everything, or nothing, would make
    the coverage result below meaningless in opposite ways.
    """

    log = tmp_path / "fires.log"
    monkeypatch.setenv(FIRE_LOG_ENV, str(log))

    assert fired_points(log) == set(), "a fresh log already reports fires"
    record_fire(FaultPoint.TORN_CATALOG_ROW)
    assert fired_points(log) == {FaultPoint.TORN_CATALOG_ROW}
    assert FaultPoint.DURING_PAYLOAD_WRITE not in fired_points(log), (
        "the recorder reported a point that never fired"
    )
    record_fire(FaultPoint.DURING_PAYLOAD_WRITE)
    assert fired_points(log) == {
        FaultPoint.TORN_CATALOG_ROW,
        FaultPoint.DURING_PAYLOAD_WRITE,
    }


def test_a_mere_mention_of_a_fault_point_records_nothing(
    tmp_path, monkeypatch
) -> None:
    """The defect the old scanner had, pinned as a property of the new one.

    Naming a member is not injecting it. Under the source-scanning mechanism the
    statement below counted as coverage; under this one it records nothing,
    which is the whole reason the mechanism was replaced.
    """

    log = tmp_path / "fires.log"
    monkeypatch.setenv(FIRE_LOG_ENV, str(log))

    FaultPoint.AFTER_RENAME_BEFORE_CATALOG  # noqa: B018 - a mention, deliberately

    assert fired_points(log) == set(), (
        "merely naming a FaultPoint produced a coverage record"
    )


@pytest.mark.parametrize(
    "point", sorted(MEASURED_POINTS, key=lambda point: point.value), ids=lambda p: p.value
)
def test_every_registered_fault_point_is_reached_when_injected(
    tmp_path, monkeypatch, point: FaultPoint
) -> None:
    """REACHABILITY, one node per point so a gap names itself.

    Set equality establishes that the RECORDED name matches the REQUESTED name.
    **IT DOES NOT ESTABLISH THAT THE INJECTION OCCURRED AT THE SITE THAT NAME
    DENOTES**, because the record derives from the plan: ``_apply_fault`` calls
    ``record_fire(fault.point)``, which is the requested identity. Exchanging
    two real injection sites while leaving the plan intact leaves this set
    unchanged, and every node here passes.

    It does not establish that the injection had an effect either. See the
    module docstring for where effect delivery is established, and for the two
    attribution limits.
    """

    log = tmp_path / "fires.log"
    monkeypatch.setenv(FIRE_LOG_ENV, str(log))

    _drive(point, tmp_path / "root")

    assert fired_points(log) == {point}, (
        f"injecting {point.value} reached {sorted(f.value for f in fired_points(log))}"
    )


def test_the_registry_is_exactly_the_set_of_reachable_points(tmp_path, monkeypatch) -> None:
    """Set equality across the whole registry, in one shared log.

    The per-point nodes prove each point CAN be reached. This proves the
    registry has no member that is never reached and no reached member that is
    unregistered -- the two halves a per-point loop cannot establish alone.

    Still reachability. No effect is asserted here.
    """

    log = tmp_path / "fires.log"
    monkeypatch.setenv(FIRE_LOG_ENV, str(log))

    for index, point in enumerate(sorted(MEASURED_POINTS, key=lambda p: p.value)):
        _drive(point, tmp_path / f"root-{index}")

    assert fired_points(log) == set(MEASURED_POINTS)


@pytest.mark.parametrize(
    "point",
    sorted(INSTRUMENT_ONLY_POINTS, key=lambda point: point.value),
    ids=lambda p: p.value,
)
def test_an_instrument_only_point_is_not_claimed_as_measured(point: FaultPoint) -> None:
    """UNMEASURED is not PASS, and must not drift into the measured set.

    These three exist so R1/R2/R3 can drive cancellation, budget exhaustion and
    a lost submit acknowledgement against a real backend. This lane provides
    them and measures none of them.
    """

    assert point not in MEASURED_POINTS
    assert point in INSTRUMENT_ONLY_POINTS


def test_a_fault_plan_round_trips_without_reflective_dispatch() -> None:
    """Enum members cross the process boundary by value, against the closed set."""

    for point in FaultPoint:
        for action in FaultAction:
            plan = FaultPlan(point, action, exit_code=71)
            assert FaultPlan.from_mapping(plan.to_dict()) == plan


def test_a_fault_plan_rejects_a_string_where_a_member_belongs() -> None:
    """Typeguard does not instrument this package, so the check must be explicit."""

    with pytest.raises(TypeError, match="FaultPoint"):
        FaultPlan("torn_catalog_row", FaultAction.RAISE)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FaultAction"):
        FaultPlan(FaultPoint.TORN_CATALOG_ROW, "raise")  # type: ignore[arg-type]
