"""The ARM N bypass witness, and a control proving the witness itself works.

WHY THIS EXISTS. ARM N's ``apply_rng_state`` arm asserts that skipping that
restore seam leaves the native trajectory UNCHANGED. An assertion of
non-divergence passes for free if the monkeypatch never applied, so the witness
that records the no-op's invocation is what makes that arm mean anything.

WHICH LEAVES THE WITNESS'S OWN CORRECTNESS UNWITNESSED -- the same recursion the
disable-verification problem has. So the witness is tested here in both
directions, and the second direction is the one that matters: a bypass that is
NOT the instrumented one must leave the witness EMPTY, because that is what
makes an empty witness mean "my bypass did not run" rather than "nothing
happened to notice".

Torch-free: ``tpen.checkpoint.restore`` imports torch only inside its functions,
so the seam attributes can be rebound and inspected on a host with no torch.
These tests never call the real seams.
"""

from __future__ import annotations

import pytest

from tests.helpers.chain_resume_spike import native_probe
from tests.helpers.chain_resume_spike.native_probe import DISABLEABLE_SEAMS, disable_seams
from tpen.checkpoint import restore as _restore_module

#: The pristine seam callables, captured at import BEFORE any test patches
#: anything. Identity against these is the only exact check: ``__module__`` does
#: not work, because ``apply_rng_state`` is imported into ``restore`` from
#: ``tpen.checkpoint.rng`` and so reports that module, while ``_load_sampler`` is
#: defined locally and reports ``restore``. A test asserting one module name for
#: both passes for the wrong reason on one of them.
_PRISTINE_SEAMS = {name: getattr(_restore_module, name) for name in DISABLEABLE_SEAMS}


@pytest.fixture(autouse=True)
def _restore_seams_and_witness():
    """Give each test a clean witness and put the real seams back afterwards.

    Restoring the ORIGINAL attributes matters beyond hygiene: a leaked no-op
    would silently disable a restore seam for every later test in the session,
    and those tests would go green while restoring nothing.
    """

    from tpen.checkpoint import restore as restore_module

    saved = {name: getattr(restore_module, name) for name in DISABLEABLE_SEAMS}
    native_probe._SEAM_INVOCATIONS.clear()
    try:
        yield restore_module
    finally:
        for name, original in saved.items():
            setattr(restore_module, name, original)
        native_probe._SEAM_INVOCATIONS.clear()


def test_the_witness_starts_empty_and_stays_empty_if_no_seam_is_called(
    _restore_seams_and_witness,
) -> None:
    """Installing a bypass is not the same as reaching it.

    A seam that is patched but never invoked must leave the witness empty --
    otherwise ARM N could report a bypass on a code path the restore never took.
    """

    assert native_probe._SEAM_INVOCATIONS == []
    disable_seams(("apply_rng_state",))
    assert native_probe._SEAM_INVOCATIONS == [], (
        "the witness recorded an invocation that never happened"
    )


@pytest.mark.parametrize("seam", DISABLEABLE_SEAMS)
def test_the_witness_records_the_instrumented_bypass_when_it_is_called(
    _restore_seams_and_witness, seam: str
) -> None:
    """Positive direction: the installed no-op reports itself, by name."""

    restore_module = _restore_seams_and_witness
    disable_seams((seam,))
    getattr(restore_module, seam)()  # the call the real restore would make

    assert native_probe._SEAM_INVOCATIONS == [seam]


def test_a_bypass_that_is_not_the_instrumented_one_leaves_the_witness_empty(
    _restore_seams_and_witness,
) -> None:
    """THE CONTROL THAT CLOSES THE RECURSION, and the reason this module exists.

    ARM N's inert arm reasons: the trajectory did not change AND the witness
    fired, therefore the seam was genuinely bypassed and genuinely does not
    matter. That inference is only valid if a witness CAN come back empty when
    something other than the instrumented no-op is in place.

    Here the seam is replaced by a plain no-op that is not the witness. If the
    witness reported an invocation anyway -- by echoing the request, or by being
    written to at install time -- then "witness fired" would be unfalsifiable,
    and ARM N's non-divergence assertion would rest on nothing.
    """

    restore_module = _restore_seams_and_witness

    def _uninstrumented_noop(*args, **kwargs):
        del args, kwargs

    setattr(restore_module, "apply_rng_state", _uninstrumented_noop)
    restore_module.apply_rng_state()

    assert native_probe._SEAM_INVOCATIONS == [], (
        "the witness reported an invocation of the instrumented bypass when a "
        "DIFFERENT no-op ran; 'the witness fired' is therefore unfalsifiable and "
        "ARM N's non-divergence arm rests on nothing"
    )


def test_an_unknown_seam_name_is_refused_rather_than_silently_ignored(
    _restore_seams_and_witness,
) -> None:
    """A typo must not read as a seam that was disabled and found harmless.

    Silently accepting an unknown name would let an arm claim it had bypassed a
    seam it never touched -- non-divergence for the most trivial wrong reason.
    """

    with pytest.raises(ValueError, match="unknown restore seam"):
        disable_seams(("apply_rng_stat",))
    assert native_probe._SEAM_INVOCATIONS == []


def test_the_fixture_puts_the_real_seams_back() -> None:
    """The seams are the production functions again once a test finishes.

    Runs after the tests above have installed no-ops. If teardown leaked, this
    node sees a no-op instead of the real callable and fails -- which is the
    only place that leak would be caught before it silently disabled restores
    for the rest of the session.
    """

    for name, pristine in _PRISTINE_SEAMS.items():
        assert getattr(_restore_module, name) is pristine, (
            f"{name} is still patched: {getattr(_restore_module, name)!r}"
        )
