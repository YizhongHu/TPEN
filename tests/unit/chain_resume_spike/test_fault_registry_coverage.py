"""Every registered fault point is OBSERVED FIRING, or declared UNMEASURED.

THE MECHANISM CHANGED, AND THE REASON MATTERS MORE THAN THE CHANGE. The previous
version of this module scanned test sources for ``FaultPoint.MEMBER`` attributes
and treated a reference as coverage. That is a proxy, and it failed in both
directions when it was measured:

* a test containing only an unused ``FaultPoint.DURING_PAYLOAD_WRITE``
  expression COUNTED AS AN EXERCISE while injecting nothing;
* replacing three explicit members with an equivalent sorted set alias made the
  registry nodes FAIL while the real fault tests still passed.

A mention cannot fire, and an alias fires identically. So coverage is now
established by RECORDING WHICH POINT ACTUALLY FIRED AT INJECTION TIME
(:func:`tests.helpers.chain_resume_spike.faults.record_fire`) and asserting the
set of fired points equals the registry. Neither defect survives that.

The record is written and ``fsync``-ed BEFORE the fault's effect, so it survives
``os._exit`` and ``SIGKILL`` -- actions whose entire purpose is that nothing runs
afterwards.
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
def test_every_measured_fault_point_actually_fires_when_injected(
    tmp_path, monkeypatch, point: FaultPoint
) -> None:
    """The coverage clause: each registered point is driven and observed firing.

    One node per point, so a gap names itself. Asserting the fired set EQUALS
    ``{point}`` rather than merely containing it also catches a fault dispatched
    at the wrong boundary -- a point that fired when a different one was
    requested is a defect the previous mechanism could not see at all.
    """

    log = tmp_path / "fires.log"
    monkeypatch.setenv(FIRE_LOG_ENV, str(log))

    _drive(point, tmp_path / "root")

    assert fired_points(log) == {point}, (
        f"injecting {point.value} recorded {sorted(f.value for f in fired_points(log))}"
    )


def test_the_measured_registry_is_exactly_what_fires(tmp_path, monkeypatch) -> None:
    """Set equality across the whole registry, in one shared log.

    The per-point nodes above prove each point CAN fire. This proves the
    registry has no member that never fires and no firing member that is
    unregistered -- the two halves a per-point loop cannot establish on its own.
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
