"""Multi-rank run-identity agreement over a real CPU/Gloo process group.

THE VACUOUS TEST THIS DELIBERATELY IS NOT. "every rank produced a run id"
passes against the broken code -- that is what the defect DID. Every assertion
here is on ranks producing the SAME id, and on that id being rank 0's, so
removing the broadcast turns these red.

Ramp: world size 1, then 2, then 3. Each rung is a separate test rather than a
parametrized sweep so a failure names the rung.

Skips are capability-based and attributable (``tests/helpers/ddp_capability.py``),
never facility-named: a torch build without Gloo, or an environment that denies
subprocess creation, says so.
"""

from __future__ import annotations

import pytest

from tests.helpers.ddp_capability import missing_capability_reason, probe_gloo_capability
from tests.helpers.run_id_agreement_harness import run_agreement_group, run_root_for
from tests.helpers.run_id_agreement_worker import expected_derived_run_id


@pytest.fixture(scope="module", autouse=True)
def _require_gloo_subprocesses() -> None:
    capability = probe_gloo_capability()
    if not capability.gloo_available:
        pytest.skip(missing_capability_reason(capability, "gloo_available"))
    if not capability.subprocess_spawn_available:
        pytest.skip(missing_capability_reason(capability, "subprocess_spawn_available"))


def _assert_no_rank_hung(result) -> None:
    """Assert the invocation ended on its own, with a receipt from every rank.

    Checked in every case including the negative ones: the failure mode the
    unconditional resolution exists to prevent is a HANG, so a test that only
    inspected ids would pass while a rank blocked to the process-group timeout.
    """

    assert result.watchdog_fired is False, f"a rank hung; logs in {result.invocation_dir}"
    assert all(receipt is not None for receipt in result.receipts), (
        f"a rank wrote no receipt; logs in {result.invocation_dir}"
    )
    # A receipt is written BEFORE `destroy_process_group`, so a rank that wrote
    # one and then died on the way out would otherwise pass every assertion in
    # these tests. The exit status is the only thing that sees it.
    assert all(code == 0 for code in result.exit_codes), (
        f"a rank exited nonzero {result.exit_codes}; logs in {result.invocation_dir}"
    )


def test_world_size_one_derives_its_own_id(tmp_path) -> None:
    """Ramp rung 1: a one-rank group takes the SERIAL branch, not the collective.

    Stated precisely because the distinction matters for what the ramp proves:
    with a world size of 1 the resolution short-circuits before
    ``_agree_on_run_id`` is ever called, so this rung exercises the same branch
    the serial test file covers -- reached through a real process group and a
    real subprocess. The first rung that enters a collective is world size 2.
    """

    result = run_agreement_group(1, [None], tmp_path)

    _assert_no_rank_hung(result)
    assert result.run_ids() == (expected_derived_run_id(0),)


def test_world_size_two_null_ids_converge_on_the_coordinator_id(tmp_path) -> None:
    """C2, rung 2: both ranks return rank 0's id, and rank 1's is discarded.

    The equality against ``expected_derived_run_id(0)`` is what makes this
    witness the DERIVATION SITE. Asserting only that the two ranks agree would
    also pass if rank 1 had derived the id and rank 0 accepted it, which is a
    different mechanism from the decided one.
    """

    result = run_agreement_group(2, [None, None], tmp_path)

    _assert_no_rank_hung(result)
    coordinator_id = expected_derived_run_id(0)
    assert result.run_ids() == (coordinator_id, coordinator_id)
    # The rank-derived factories guarantee rank 1 WOULD have produced this had
    # it derived its own id, so its absence is evidence rather than luck.
    assert expected_derived_run_id(1) not in result.run_ids()


def test_world_size_three_null_ids_converge_on_the_coordinator_id(tmp_path) -> None:
    """C2, rung 3: agreement is not an artifact of there being only two ranks."""

    result = run_agreement_group(3, [None, None, None], tmp_path)

    _assert_no_rank_hung(result)
    coordinator_id = expected_derived_run_id(0)
    assert result.run_ids() == (coordinator_id, coordinator_id, coordinator_id)
    assert expected_derived_run_id(1) not in result.run_ids()
    assert expected_derived_run_id(2) not in result.run_ids()


def test_agreeing_explicit_ids_are_accepted_without_derivation(tmp_path) -> None:
    """C3 under a real group: an explicit id common to every rank is honored."""

    result = run_agreement_group(2, ["2026-01-01_000000_explicit_aaaaaa"] * 2, tmp_path)

    _assert_no_rank_hung(result)
    assert result.run_ids() == ("2026-01-01_000000_explicit_aaaaaa",) * 2


def test_mixed_null_and_explicit_ids_fail_on_every_rank(tmp_path) -> None:
    """C4: the asymmetric launch fails loudly instead of hanging or scattering.

    This is the case that governed the design. Had the resolution been entered
    only on the ``run_id is None`` branch, rank 0 would have called into a
    collective that rank 1 never reached, and the launch would have blocked to
    the process-group timeout rather than reporting anything.
    """

    result = run_agreement_group(2, [None, "2026-01-01_000000_explicit_aaaaaa"], tmp_path)

    _assert_no_rank_hung(result)
    assert result.errors() == ("RunIdentityError", "RunIdentityError")
    assert result.run_ids() == (None, None)
    # WHICH refusal fired, not merely that one did: every refusal in this module
    # raises RunIdentityError, so the type alone would accept the broadcast
    # failure or the backend refusal just as happily as the right one.
    for message in result.error_messages():
        assert "ranks disagree on run.run_id" in message, message
        assert "no run directory was created" in message


def test_disagreeing_explicit_ids_fail_on_every_rank(tmp_path) -> None:
    """C4: two ranks configured with different explicit ids cannot proceed."""

    result = run_agreement_group(
        2, ["2026-01-01_000000_left_aaaaaa", "2026-01-01_000000_right_bbbbbb"], tmp_path
    )

    _assert_no_rank_hung(result)
    assert result.errors() == ("RunIdentityError", "RunIdentityError")
    assert result.run_ids() == (None, None)
    for message in result.error_messages():
        assert "ranks disagree on run.run_id" in message, message


def test_a_launcher_that_contradicts_its_own_process_group_is_refused(tmp_path) -> None:
    """A topology declaring 3 ranks over a 2-rank group is refused on its own terms.

    Before this refusal existed the same shape produced a message reading
    "cannot agree on a run id across 1 ranks: ... the process group covers 4
    ranks" for the null case, and was silently ACCEPTED for the explicit case --
    a launcher self-contradiction ratified rather than reported. Neither answer
    is trustworthy, so it is refused before the null/explicit split.
    """

    result = run_agreement_group(2, [None, None], tmp_path, declared_world_size=3)

    _assert_no_rank_hung(result)
    assert result.errors() == ("RunIdentityError", "RunIdentityError")
    for message in result.error_messages():
        assert "launcher disagrees with itself" in message, message
        assert "declares 3 ranks" in message
        assert "covers 2" in message


def test_a_contradicting_launcher_is_refused_even_with_an_explicit_id(tmp_path) -> None:
    """The same refusal fires for an explicit id, which used to be accepted."""

    result = run_agreement_group(
        2, ["2026-01-01_000000_explicit_aaaaaa"] * 2, tmp_path, declared_world_size=3
    )

    _assert_no_rank_hung(result)
    assert result.errors() == ("RunIdentityError", "RunIdentityError")
    assert result.run_ids() == (None, None)


def test_ranks_converge_on_one_artifact_root(tmp_path) -> None:
    """The defect stated in its own terms: one run root, not one per rank.

    Asserted on the run DIRECTORY, because scattered artifacts were the harm --
    an agreed id that still produced two roots would satisfy an id-only test.

    Deliberately NOT asserted: the contents of the common files under that root.
    Every rank writes ``run_start.json`` and the metadata artifacts, and
    ``write_json`` truncates and rewrites rather than replacing atomically, so
    common publication is contended until it becomes coordinator-only. That
    hazard is inherited by unifying the root and is tracked separately; pinning
    contended content here would be pinning a race.
    """

    result = run_agreement_group(2, [None, None], tmp_path, mode="context")

    _assert_no_rank_hung(result)
    coordinator_id = expected_derived_run_id(0)
    assert result.run_ids() == (coordinator_id, coordinator_id)

    run_dirs = {receipt["run_dir"] for receipt in result.receipts}
    assert len(run_dirs) == 1, run_dirs

    sector_root = run_root_for(result.invocation_dir) / "run-identity" / "agreement"
    assert [path.name for path in sorted(sector_root.iterdir())] == [coordinator_id]


def test_a_refused_launch_creates_no_run_directory(tmp_path) -> None:
    """C4 through the construction path: refuse while the tree is still absent.

    A refusal that fired after ``make_dirs`` would leave behind the per-rank
    directories the agreement exists to prevent, so the evidence is the absent
    artifact root rather than the raise alone.
    """

    result = run_agreement_group(
        2, [None, "2026-01-01_000000_explicit_aaaaaa"], tmp_path, mode="context"
    )

    _assert_no_rank_hung(result)
    assert result.errors() == ("RunIdentityError", "RunIdentityError")
    for message in result.error_messages():
        assert "ranks disagree on run.run_id" in message, message
    assert not run_root_for(result.invocation_dir).exists()
