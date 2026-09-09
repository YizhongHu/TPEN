"""Every registered fault point is either exercised or declared UNMEASURED.

A named fault point nobody drives rots: it keeps looking like coverage in the
registry while testing nothing. So the registry is closed, and it is partitioned
into exactly two sets whose members are checked differently:

* :data:`MEASURED_POINTS` -- each must be referenced by at least one test module
  in this lane, and the scan deliberately EXCLUDES this module so the coverage
  test cannot satisfy itself by naming the points it is checking.
* :data:`INSTRUMENT_ONLY_POINTS` -- provided for R1/R2/R3 and not measured here.
  Providing an instrument is not covering a gate, and these must NOT appear in
  the measured set.

The scanner is shown to detect an absence before its clean result is believed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.helpers.chain_resume_spike.faults import (
    COMMITTED_BUT_UNACKNOWLEDGED_POINTS,
    INSTRUMENT_ONLY_POINTS,
    MEASURED_POINTS,
    FaultAction,
    FaultPlan,
    FaultPoint,
)

#: Directories whose test modules count as exercising a point.
TEST_DIRS = (
    Path("tests/unit/chain_resume_spike"),
    Path("tests/integration/chain_resume_spike"),
)

#: This module is excluded from the scan. It necessarily names every point it
#: checks, so including it would make the coverage assertion self-satisfying.
_THIS_MODULE = Path(__file__).name


def _referenced_points(sources: dict[str, str]) -> set[str]:
    """Return every ``FaultPoint.MEMBER`` attribute named in ``sources``.

    Parsed rather than grepped: a substring search would count a member named
    inside a docstring or a comment as coverage, and a point that is only
    talked about is exactly the rot this test exists to catch.
    """

    referenced: set[str] = set()
    for label, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "FaultPoint"
            ):
                referenced.add(node.attr)
        del label
    return referenced


def _lane_test_sources() -> dict[str, str]:
    """Return the lane's test modules, excluding this one."""

    sources: dict[str, str] = {}
    for directory in TEST_DIRS:
        for path in sorted(directory.glob("test_*.py")):
            if path.name == _THIS_MODULE:
                continue
            sources[str(path)] = path.read_text(encoding="utf-8")
    assert sources, "no lane test modules found; the coverage scan would be vacuous"
    return sources


def test_the_registry_is_partitioned_with_nothing_left_over() -> None:
    """Measured and instrument-only must cover the enum, disjointly.

    A point in neither set would be silently uncovered AND silently
    undeclared -- the worst of both, since no test would drive it and no
    document would admit that.
    """

    assert MEASURED_POINTS | INSTRUMENT_ONLY_POINTS == set(FaultPoint)
    assert not (MEASURED_POINTS & INSTRUMENT_ONLY_POINTS)
    assert COMMITTED_BUT_UNACKNOWLEDGED_POINTS <= MEASURED_POINTS


def test_the_scanner_detects_an_absence() -> None:
    """Instrument check. A scanner that never reports a gap cannot prove one absent.

    Three separate false-zero defects in this repository came from harnesses
    that could not see the thing they were counting.
    """

    referenced = _referenced_points({"synthetic": "FaultPoint.TORN_CATALOG_ROW\n"})
    assert referenced == {"TORN_CATALOG_ROW"}
    assert "DURING_PAYLOAD_WRITE" not in referenced, (
        "the scanner reported a point that the source never named"
    )


def test_the_scanner_ignores_a_point_named_only_in_prose() -> None:
    """A mention in a docstring is not an exercise.

    This is the difference between the AST scan and a grep, and it is the whole
    reason the scan is worth writing.
    """

    prose_only = '"""This module talks about FaultPoint.BUDGET_EXHAUSTED."""\n'
    assert _referenced_points({"prose": prose_only}) == set()


@pytest.mark.parametrize(
    "point", sorted(MEASURED_POINTS, key=lambda point: point.value), ids=lambda p: p.value
)
def test_every_measured_fault_point_is_exercised_by_a_lane_test(point: FaultPoint) -> None:
    """The coverage clause itself, one node per point so a gap names itself."""

    referenced = _referenced_points(_lane_test_sources())
    assert point.name in referenced, (
        f"{point.name} is registered as MEASURED but no lane test module outside "
        f"{_THIS_MODULE} references it"
    )


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


def test_a_fault_plan_round_trips_without_reflective_dispatch(tmp_path) -> None:
    """Enum members cross the process boundary by value, against the closed set."""

    for point in FaultPoint:
        for action in FaultAction:
            plan = FaultPlan(point, action, exit_code=71)
            assert FaultPlan.from_mapping(plan.to_dict()) == plan
    del tmp_path


def test_a_fault_plan_rejects_a_string_where_a_member_belongs() -> None:
    """Typeguard does not instrument this package, so the check must be explicit."""

    with pytest.raises(TypeError, match="FaultPoint"):
        FaultPlan("torn_catalog_row", FaultAction.RAISE)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FaultAction"):
        FaultPlan(FaultPoint.TORN_CATALOG_ROW, "raise")  # type: ignore[arg-type]
