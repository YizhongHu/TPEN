"""R5 native coupling declarations keep attribution falsifiable."""

import pytest

from tpen.execution.chain_resume import (
    ChannelEvent, ChannelParityError, E5_CHANNEL_MAP,
    E5_NATIVE_EXPECTED_MISMATCHES, ProcessEvidence, compare_channel_parity,
)


def _events(step, mismatches=()):
    return tuple(ChannelEvent(spec.channel, step, spec.event_kind,
                              -1 if spec.channel in mismatches else 1)
                 for spec in E5_CHANNEL_MAP)


def test_native_coupling_signatures_distinguish_every_single_omission():
    signatures = tuple(E5_NATIVE_EXPECTED_MISMATCHES.values())
    assert len(set(signatures)) == len(E5_CHANNEL_MAP)
    assert set(E5_NATIVE_EXPECTED_MISMATCHES) == {spec.limb for spec in E5_CHANNEL_MAP}


def test_native_coupled_attribution_passes_before_wrong_attribution_mutant():
    reference = ProcessEvidence("reference", 1, 1, "runtime", (), 0, _events(3))
    mismatches = E5_NATIVE_EXPECTED_MISMATCHES["optimizer"]
    resumed = ProcessEvidence("recovery", 2, 1, "checkpoint", (), 0, _events(3, mismatches))
    result = compare_channel_parity(reference, resumed, resume_step=2,
                                    disabled_limbs=("optimizer",),
                                    expected_mismatches=E5_NATIVE_EXPECTED_MISMATCHES)
    assert {channel for channel, equal in result.items() if not equal} == set(mismatches)
    with pytest.raises(ChannelParityError):
        compare_channel_parity(reference, resumed, resume_step=2,
                               disabled_limbs=("process_rng",),
                               expected_mismatches=E5_NATIVE_EXPECTED_MISMATCHES)


def test_native_cursor_omission_uses_first_restarted_event_and_names_mismatch():
    reference = ProcessEvidence("reference", 1, 1, "runtime", (), 0, _events(3))
    resumed = ProcessEvidence("recovery", 2, 1, "checkpoint", (), 0,
                              _events(0, {"process_random"}) + _events(3))
    result = compare_channel_parity(reference, resumed, resume_step=2,
                                    disabled_limbs=("trainer_cursor",),
                                    expected_mismatches=E5_NATIVE_EXPECTED_MISMATCHES)
    assert result["trainer_cursor"] is False
    # Removing the declaration must preserve the existing fail-closed guard.
    with pytest.raises(ChannelParityError, match="trainer_cursor"):
        compare_channel_parity(reference, resumed, resume_step=2, disabled_limbs=("trainer_cursor",))
