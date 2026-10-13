"""Source-level contract tests for the E5 chain-resume adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tpen.execution import (
    NATIVE_PROFILE_VERSION,
    ChannelMapError,
    ChannelParityError,
    ChainResumeReceipt,
    CheckpointFaultKind,
    UnmeasuredArmError,
    assert_fresh_processes,
    assert_positive_control,
    classify_checkpoint_fault,
    compare_channel_parity,
    e5_channel_map,
    run_fresh_process,
    validate_channel_map,
)
from tpen.execution.native_profile import SUPPORTED_STATE_OWNERS


PROBE = Path(__file__).parents[2] / "helpers" / "chain_resume_probe.py"
COMMAND = (sys.executable, str(PROBE))
RESUME_STEP = 2
SEED = 8663


def _run(
    *,
    phase: str,
    disable: tuple[str, ...] = (),
    suppress: str | None = None,
    omit: str | None = None,
):
    command = [*COMMAND, "--seed", str(SEED), "--resume-step", str(RESUME_STEP)]
    for limb in disable:
        command.extend(("--disable-limb", limb))
    if suppress is not None:
        command.extend(("--suppress-channel", suppress))
    if omit is not None:
        command.extend(("--omit-channel", omit))
    return run_fresh_process(
        command,
        phase=phase,
        seed=SEED,
        seed_source=f"{phase}:runtime.seed={SEED}",
    )


def test_e5_channel_map_is_total_and_directional() -> None:
    mapping = e5_channel_map()
    assert {spec.limb for spec in mapping} == set(SUPPORTED_STATE_OWNERS)
    assert len({spec.channel for spec in mapping}) == len(mapping)
    assert {spec.event_kind for spec in mapping}
    assert validate_channel_map(SUPPORTED_STATE_OWNERS, mapping) == mapping


def test_totality_guard_rejects_supported_limb_without_channel() -> None:
    mapping = tuple(spec for spec in e5_channel_map() if spec.limb != "process_rng")
    with pytest.raises(ChannelMapError, match="no observable channel.*process_rng"):
        validate_channel_map(SUPPORTED_STATE_OWNERS, mapping)


def test_reference_and_resume_are_fresh_processes_with_seed_provenance() -> None:
    reference = _run(phase="reference")
    resumed = _run(phase="recovery")
    assert reference.pid > 0 and resumed.pid > 0
    assert reference.pid != resumed.pid
    assert reference.seed == resumed.seed == SEED
    assert reference.seed_source != resumed.seed_source
    assert_fresh_processes(reference, resumed)
    assert compare_channel_parity(reference, resumed, resume_step=RESUME_STEP) == {
        spec.channel: True for spec in e5_channel_map()
    }


@pytest.mark.parametrize("limb", sorted(SUPPORTED_STATE_OWNERS))
def test_disabled_restore_limb_has_named_channel_mismatch(limb: str) -> None:
    reference = _run(phase="reference")
    resumed = _run(phase="recovery", disable=(limb,))
    result = compare_channel_parity(
        reference,
        resumed,
        resume_step=RESUME_STEP,
        disabled_limbs=(limb,),
    )
    assert result[next(spec.channel for spec in e5_channel_map() if spec.limb == limb)] is False
    assert all(
        value
        for channel, value in result.items()
        if channel != next(spec.channel for spec in e5_channel_map() if spec.limb == limb)
    )


@pytest.mark.parametrize("limb", sorted(SUPPORTED_STATE_OWNERS))
def test_each_channel_positive_control_moves_its_own_observable(limb: str) -> None:
    spec = next(spec for spec in e5_channel_map() if spec.limb == limb)
    baseline = _run(phase="reference")
    perturbed = _run(phase="recovery", disable=(limb,))
    assert_positive_control(baseline, perturbed, spec, resume_step=RESUME_STEP)


def test_observability_mutant_is_rejected_when_limb_is_live_but_channel_suppressed() -> None:
    reference = _run(phase="reference")
    resumed = _run(phase="recovery", suppress="sampler_proposal")
    with pytest.raises(ChannelParityError, match="sampler_proposal"):
        compare_channel_parity(
            reference,
            resumed,
            resume_step=RESUME_STEP,
        )


def test_wrong_channel_attribution_is_rejected() -> None:
    reference = _run(phase="reference")
    # The real disabled limb is sampler_mcmc, but the assertion attributes it
    # to process_rng.  A shared final observable would not catch this error.
    resumed = _run(phase="recovery", disable=("sampler_mcmc",))
    with pytest.raises(ChannelParityError, match="channel"):
        compare_channel_parity(
            reference,
            resumed,
            resume_step=RESUME_STEP,
            disabled_limbs=("process_rng",),
        )


def test_mutual_masking_mutants_are_separately_falsifiable() -> None:
    reference = _run(phase="reference")
    all_disabled = _run(phase="recovery", disable=tuple(sorted(SUPPORTED_STATE_OWNERS)))
    with pytest.raises(ChannelParityError, match="channel"):
        compare_channel_parity(
            reference,
            all_disabled,
            resume_step=RESUME_STEP,
            disabled_limbs=("model",),
        )

    one_disabled = _run(phase="recovery", disable=("model",))
    with pytest.raises(ChannelParityError, match="channel"):
        compare_channel_parity(
            reference,
            one_disabled,
            resume_step=RESUME_STEP,
            disabled_limbs=SUPPORTED_STATE_OWNERS,
        )


def test_missing_observation_is_unmeasured_not_green() -> None:
    reference = _run(phase="reference")
    # A fixture that suppresses a channel entirely has no first event to
    # compare.  The adapter must not turn that absence into parity.
    resumed = _run(phase="recovery", omit="sampler_proposal")
    with pytest.raises(UnmeasuredArmError, match="sampler_proposal"):
        compare_channel_parity(reference, resumed, resume_step=RESUME_STEP)


def test_checkpoint_fault_interface_is_typed_and_separate_from_scheduler() -> None:
    before = classify_checkpoint_fault(
        CheckpointFaultKind.BEFORE_COMMIT, detail="checkpoint directory incomplete"
    )
    after = classify_checkpoint_fault(
        CheckpointFaultKind.CORRUPT_COMMITTED, detail="manifest schema mismatch"
    )
    assert before == {
        "layer": "fixture",
        "kind": "before_commit",
        "detail": "checkpoint directory incomplete",
    }
    assert after["layer"] == "application"
    assert "scheduler" not in before and "wrapper" not in after


def test_receipt_preserves_exact_provenance_and_unmeasured_arm() -> None:
    reference = _run(phase="reference")
    resumed = _run(phase="recovery")
    receipt = ChainResumeReceipt(
        source_sha="a" * 40,
        adapter_sha="b" * 40,
        profile_version=NATIVE_PROFILE_VERSION,
        channel_map_version="e5-native-channel-map-v1",
        commands=(reference.command, resumed.command),
        process_evidence=(reference, resumed),
        channel_results=compare_channel_parity(reference, resumed, resume_step=RESUME_STEP),
        checkpoint_faults=(classify_checkpoint_fault("no_checkpoint", detail="park"),),
        collection_output=("tests/unit/execution/test_chain_resume.py",),
        testcase_ids=("test_reference_and_resume_are_fresh_processes_with_seed_provenance",),
        junit_identities=("tests/unit/execution/test_chain_resume.py::test_reference_and_resume_are_fresh_processes_with_seed_provenance",),
        raw_fault_evidence=({"arm": "observability", "status": "UNMEASURED"},),
        restore_disabled_control={"status": "UNMEASURED", "reason": "not collected in this receipt"},
    )
    encoded = receipt.to_dict()
    assert encoded["profile_version"] == NATIVE_PROFILE_VERSION
    assert encoded["process_evidence"][0]["pid"] == reference.pid
    assert encoded["commands"][1] == list(resumed.command)
    assert encoded["raw_fault_evidence"][0]["status"] == "UNMEASURED"
    assert encoded["restore_disabled_control"]["status"] == "UNMEASURED"
