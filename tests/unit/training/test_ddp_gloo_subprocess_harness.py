"""Stage T1: safe local CPU/Gloo subprocess harness tests (DF1).

Proves the HARNESS's safety properties (isolated rendezvous, bounded
timeouts, deterministic result collection, process-group cleanup, typed
fault injection) using a minimal synthetic worker body. This is
infrastructure for the whole DDP lane and asserts nothing about
``tpen.training.vmc.compute_vmc_objective`` (Stage T2, out of scope here).

Ramp: world_size 1, then 2, then 3.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.helpers.ddp_capability import missing_capability_reason, probe_gloo_capability
from tests.helpers.ddp_fault_injection import FaultKind, FaultPhase, FaultPlan
from tests.helpers.ddp_subprocess_harness import (
    HarnessBounds,
    RankReceipt,
    run_gloo_subprocess_group,
)

_CAPABILITY = probe_gloo_capability()

# Default bounds for tests where the assertion does not depend on precise
# timing: generous enough to absorb interpreter/torch import overhead and
# shared-node scheduling jitter (~2-4s observed for a bare worker import),
# without being so large that a genuine watchdog-kill test takes minutes.
_DEFAULT_PG_TIMEOUT = 6.0
_DEFAULT_WATCHDOG_TIMEOUT = 20.0


def _require_gloo_capability() -> None:
    if not _CAPABILITY.gloo_available:
        pytest.skip(missing_capability_reason(_CAPABILITY, "gloo_available"))
    if not _CAPABILITY.subprocess_spawn_available:
        pytest.skip(missing_capability_reason(_CAPABILITY, "subprocess_spawn_available"))


def _default_bounds() -> HarnessBounds:
    return HarnessBounds(
        process_group_timeout=_DEFAULT_PG_TIMEOUT, watchdog_timeout=_DEFAULT_WATCHDOG_TIMEOUT
    )


# --- T0.1: capability-probe mechanism (ungated -- tests the gate itself) ----


def test_capability_probe_skip_reason_names_missing_capability():
    from tests.helpers.ddp_capability import GlooSubprocessCapability

    capability = GlooSubprocessCapability(
        gloo_available=False,
        subprocess_spawn_available=True,
        reasons={"gloo_available": "synthetic: gloo not built"},
    )
    reason = missing_capability_reason(capability, "gloo_available")
    assert "gloo_available" in reason


# --- World size 1 -------------------------------------------------------


def test_world_size_one_self_test_all_phases_succeed(tmp_path):
    _require_gloo_capability()
    result = run_gloo_subprocess_group(1, None, _default_bounds(), tmp_path)
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert result.publication_observed is True
    assert result.exit_codes == (0,)
    assert len(result.receipts) == 1
    receipt = result.receipts[0]
    assert receipt is not None
    assert receipt.collective_result == pytest.approx(1.0)
    assert receipt.phase_sequence == [
        "BEFORE_COLLECTIVE",
        "AFTER_COLLECTIVE",
        "BEFORE_OPTIMIZER_STEP",
        "AFTER_OPTIMIZER_STEP",
        "BEFORE_STATE_WRITE",
        "AFTER_STATE_WRITE",
        "BEFORE_PUBLICATION",
        "AFTER_PUBLICATION",
    ]


def test_world_size_one_rank_exception_before_optimizer_step_fails_cleanly(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=0, kind=FaultKind.RAISE_BEFORE_BACKWARD, phase=FaultPhase.BEFORE_OPTIMIZER_STEP)
    result = run_gloo_subprocess_group(1, plan, _default_bounds(), tmp_path)
    assert result.publication_observed is False
    assert result.all_reaped is True
    assert result.receipts == (None,)
    assert result.culprit_rank == 0


def test_world_size_one_watchdog_kills_stall_before_process_group_init(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(
        target_rank=0, kind=FaultKind.STALL_BEFORE_COLLECTIVE, phase=None, delay_seconds=20.0
    )
    bounds = HarnessBounds(process_group_timeout=2.0, watchdog_timeout=8.0)
    result = run_gloo_subprocess_group(1, plan, bounds, tmp_path)
    assert result.watchdog_fired is True
    assert result.all_reaped is True
    assert result.publication_observed is False


# --- World size 2 --------------------------------------------------------


def test_world_size_two_self_test_all_phases_succeed(tmp_path):
    _require_gloo_capability()
    result = run_gloo_subprocess_group(2, None, _default_bounds(), tmp_path)
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert result.publication_observed is True
    assert result.exit_codes == (0, 0)
    assert len(result.receipts) == 2
    for receipt in result.receipts:
        assert receipt is not None
        assert receipt.collective_result == pytest.approx(2.0)


@pytest.mark.parametrize(
    "fault_kind", [FaultKind.SKIP_COLLECTIVE, FaultKind.MISMATCH_COLLECTIVE, FaultKind.MISMATCH_SHAPE]
)
def test_collective_fault_does_not_apply_before_configured_phase(tmp_path, fault_kind):
    """A collective fault's effect and attribution must honor its phase.

    Adopted from codex/review/df-round3-contract-evidence at
    75834aa3f06961f34c37c04d937d847f28aa6066 (DF-R3-F2), widened here to all
    three collective-fault kinds per acceptance-contract P1 rather than
    SKIP_COLLECTIVE alone -- the underlying defect was never kind-specific.
    """
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=fault_kind, phase=FaultPhase.AFTER_COLLECTIVE)

    # A helper may reject this unsupported combination before launching
    # workers, but it must not silently reinterpret AFTER_COLLECTIVE as the
    # BEFORE_COLLECTIVE injection point.
    try:
        result = run_gloo_subprocess_group(2, plan, _default_bounds(), tmp_path)
    except ValueError as exc:
        message = str(exc)
        assert fault_kind.name in message
        assert "AFTER_COLLECTIVE" in message
    else:
        assert result.publication_observed is True
        assert result.exit_codes == (0, 0)
        assert result.culprit_rank is None
        assert len(result.receipts) == 2
        for receipt in result.receipts:
            assert receipt is not None
            assert receipt.collective_result == pytest.approx(2.0)


def test_do_collective_self_report_names_actual_phase_not_hardcoded(capsys):
    # Own addition, not from the reviewer's branch. Targets P3 directly and
    # independently of P1/P2: this calls the worker's internal
    # _do_collective directly, bypassing run_gloo_subprocess_group's
    # validate_fault_plan gate entirely, so it is the only test in this
    # file that can observe what _do_collective would report if it were
    # ever reached with a phase other than BEFORE_COLLECTIVE -- something
    # the public harness API can no longer produce once P1/P2 are fixed. A
    # regression to the hardcoded FaultPhase.BEFORE_COLLECTIVE.name would be
    # invisible to every test that goes through the public API, because by
    # the time validate_fault_plan lets a call through, plan.phase is
    # already guaranteed to equal BEFORE_COLLECTIVE -- the hardcoded value
    # and the real one would always agree there. Calling the private
    # function directly is what makes the two divergent and the mutation
    # observable at all. SKIP_COLLECTIVE needs no live process group: it
    # reports and returns None before touching torch.distributed.
    from tests.helpers.ddp_worker_entrypoint import _do_collective

    plan = FaultPlan(target_rank=1, kind=FaultKind.SKIP_COLLECTIVE, phase=FaultPhase.AFTER_COLLECTIVE)
    result = _do_collective(1, 2, plan)
    assert result is None
    captured = capsys.readouterr()
    assert "phase AFTER_COLLECTIVE" in captured.err
    assert "phase BEFORE_COLLECTIVE" not in captured.err


def test_world_size_two_skip_collective_triggers_process_group_timeout_and_group_reap(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=FaultKind.SKIP_COLLECTIVE, phase=FaultPhase.BEFORE_COLLECTIVE)
    result = run_gloo_subprocess_group(2, plan, _default_bounds(), tmp_path)
    assert result.publication_observed is False
    assert result.all_reaped is True
    assert result.culprit_rank == 1


def test_world_size_two_mismatched_collective_type_fails_with_structured_evidence(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=FaultKind.MISMATCH_COLLECTIVE, phase=FaultPhase.BEFORE_COLLECTIVE)
    result = run_gloo_subprocess_group(2, plan, _default_bounds(), tmp_path)
    assert result.publication_observed is False
    assert result.all_reaped is True


def test_world_size_two_mismatched_tensor_shape_fails_with_structured_evidence(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=FaultKind.MISMATCH_SHAPE, phase=FaultPhase.BEFORE_COLLECTIVE)
    result = run_gloo_subprocess_group(2, plan, _default_bounds(), tmp_path)
    assert result.publication_observed is False
    assert result.all_reaped is True


def test_world_size_two_bounded_stall_under_process_group_timeout_still_succeeds(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(
        target_rank=1, kind=FaultKind.STALL_BEFORE_COLLECTIVE, phase=FaultPhase.BEFORE_COLLECTIVE, delay_seconds=1.0
    )
    bounds = HarnessBounds(process_group_timeout=15.0, watchdog_timeout=30.0)
    result = run_gloo_subprocess_group(2, plan, bounds, tmp_path)
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert result.publication_observed is True
    for receipt in result.receipts:
        assert receipt is not None


def test_harness_bounds_rejects_watchdog_not_exceeding_process_group_timeout():
    HarnessBounds(process_group_timeout=1.0, watchdog_timeout=2.0)
    with pytest.raises(ValueError):
        HarnessBounds(process_group_timeout=2.0, watchdog_timeout=2.0)
    with pytest.raises(ValueError):
        HarnessBounds(process_group_timeout=2.0, watchdog_timeout=1.0)


def test_world_size_two_crash_after_local_publish_blocks_global_completion(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=FaultKind.CRASH_AFTER_PUBLISH, phase=FaultPhase.BEFORE_PUBLICATION)
    result = run_gloo_subprocess_group(2, plan, _default_bounds(), tmp_path)
    assert result.publication_observed is False
    assert result.all_reaped is True


def test_world_size_two_receipts_present_for_every_rank_including_culprit(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=FaultKind.SKIP_COLLECTIVE, phase=FaultPhase.BEFORE_COLLECTIVE)
    result = run_gloo_subprocess_group(2, plan, _default_bounds(), tmp_path)
    assert len(result.receipts) == 2


def test_world_size_two_two_consecutive_skip_collective_invocations_use_independent_fresh_rendezvous_and_subprocesses(
    tmp_path,
):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=FaultKind.SKIP_COLLECTIVE, phase=FaultPhase.BEFORE_COLLECTIVE)
    result1 = run_gloo_subprocess_group(2, plan, _default_bounds(), tmp_path)
    result2 = run_gloo_subprocess_group(2, plan, _default_bounds(), tmp_path)
    assert result1.rendezvous_path != result2.rendezvous_path
    assert result1.publication_observed is False
    assert result2.publication_observed is False
    assert result1.all_reaped is True
    assert result2.all_reaped is True


def test_world_size_two_raise_after_optimizer_step_respects_configured_phase(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=FaultKind.RAISE_BEFORE_BACKWARD, phase=FaultPhase.AFTER_OPTIMIZER_STEP)
    result = run_gloo_subprocess_group(2, plan, _default_bounds(), tmp_path)
    assert result.publication_observed is False
    assert result.all_reaped is True
    assert result.culprit_rank == 1
    # Discriminates AFTER_OPTIMIZER_STEP (this plan's phase) from
    # BEFORE_OPTIMIZER_STEP: the marker is written only if the fake
    # optimizer-style update ran before rank 1's raise fired.
    assert (Path(result.invocation_dir) / "state_1.json.optimizer_done").exists()


def test_world_size_two_watchdog_reap_also_kills_worker_spawned_grandchild(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=FaultKind.SKIP_COLLECTIVE, phase=FaultPhase.BEFORE_COLLECTIVE)
    result = run_gloo_subprocess_group(
        2, plan, _default_bounds(), tmp_path, decoy_grandchild_rank=1
    )
    assert result.all_reaped is True
    grandchild_pid_path = Path(result.invocation_dir) / "grandchild_1.pid"
    assert grandchild_pid_path.exists()
    grandchild_pid = int(grandchild_pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild_pid, 0)


# --- World size 3 --------------------------------------------------------


def test_world_size_three_self_test_all_phases_succeed(tmp_path):
    _require_gloo_capability()
    result = run_gloo_subprocess_group(3, None, _default_bounds(), tmp_path)
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert result.publication_observed is True
    assert result.exit_codes == (0, 0, 0)
    assert len(result.receipts) == 3
    for receipt in result.receipts:
        assert receipt is not None
        assert receipt.collective_result == pytest.approx(3.0)
        assert len(receipt.phase_sequence) == 8


def test_world_size_three_middle_rank_fault_identifies_correct_culprit(tmp_path):
    # culprit_rank is DERIVED here from rank 1's own self-reported log entry
    # (see ddp_worker_entrypoint._report_fault_applied), not copied from the
    # plan below -- see test_fault_plan_target_rank_outside_world_size_
    # derives_no_culprit for the case that discriminates the two.
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=FaultKind.RAISE_BEFORE_BACKWARD, phase=FaultPhase.BEFORE_OPTIMIZER_STEP)
    result = run_gloo_subprocess_group(3, plan, _default_bounds(), tmp_path)
    assert result.culprit_rank == 1
    assert result.publication_observed is False
    assert result.all_reaped is True


def test_world_size_three_crash_during_checkpoint_leaves_no_publication_marker(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=2, kind=FaultKind.CRASH_DURING_CHECKPOINT, phase=FaultPhase.AFTER_STATE_WRITE)
    result = run_gloo_subprocess_group(3, plan, _default_bounds(), tmp_path)
    assert result.publication_observed is False
    assert result.all_reaped is True
    assert result.culprit_rank == 2


# --- Round-1 review tests (reviewer-designed; planner 7f1d8dde) -----------
#
# Deterministic stand-in for a torn receipt write. The worker's receipt write
# is a bare, non-atomic ``Path.write_text``, and the harness's ``killpg`` is
# unconditional, so a rank killed mid-write leaves truncated JSON on disk.
# Rather than racing a real kill against a real write, this pre-seeds the
# exact on-disk artefact that race produces, with zero modification to the
# subject.
_TRUNCATED_RECEIPT_JSON = '{"rank": 0, "world_'


def test_malformed_receipt_degrades_to_none_instead_of_crashing_collection(tmp_path):
    _require_gloo_capability()
    # Pre-seed rank 0's receipt path with a truncated write. The fault below
    # fires at BEFORE_STATE_WRITE -- after process-group init, but before the
    # worker's own receipt write -- so this garbage is never overwritten and
    # is exactly what the harness's collection loop reads back. Requires a
    # caller-supplied, pre-known invocation_dir: the harness's own default
    # (a fresh tempfile.mkdtemp per call) is generated only inside the call,
    # too late to pre-seed anything into it.
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()
    receipt_path = invocation_dir / "receipt_0.json"
    receipt_path.write_text(_TRUNCATED_RECEIPT_JSON)

    plan = FaultPlan(
        target_rank=0,
        kind=FaultKind.CRASH_DURING_CHECKPOINT,
        phase=FaultPhase.BEFORE_STATE_WRITE,
    )
    result = run_gloo_subprocess_group(
        1, plan, _default_bounds(), tmp_path, invocation_dir=invocation_dir
    )

    # Premise guards: the crash fired where intended (os._exit(1)) and the
    # pre-seeded truncation survived, so a pass here cannot come from the
    # worker having quietly written a well-formed receipt instead.
    assert result.exit_codes == (1,)
    assert receipt_path.read_text() == _TRUNCATED_RECEIPT_JSON

    # The documented semantic is one slot per rank, never a raise during
    # collection. Asserted as "not a valid RankReceipt" rather than "is None"
    # so the test does not dictate the fix's representation: None and a
    # distinct malformed marker both satisfy it.
    assert len(result.receipts) == 1
    assert not isinstance(result.receipts[0], RankReceipt)

    # Cleanup reporting must survive the malformed receipt, not be skipped
    # by an exception raised earlier in collection.
    assert result.all_reaped is True
    assert result.publication_observed is False


# --- Round-2 review tests (reviewer-designed; durable review
# df-durable-reviewer-20260904, findings F1-F4 on item 37ab3d40) ----------


def test_success_then_fault_in_same_tmp_path_does_not_reuse_success_artifacts(tmp_path):
    _require_gloo_capability()
    success = run_gloo_subprocess_group(1, None, _default_bounds(), tmp_path)
    assert success.publication_observed is True
    plan = FaultPlan(
        target_rank=0,
        kind=FaultKind.CRASH_DURING_CHECKPOINT,
        phase=FaultPhase.BEFORE_STATE_WRITE,
    )
    failed = run_gloo_subprocess_group(1, plan, _default_bounds(), tmp_path)
    assert failed.publication_observed is False
    assert failed.all_reaped is True
    assert failed.exit_codes == (1,)
    assert failed.receipts == (None,)


def test_rank_exception_preserves_attributable_diagnostic_artifact(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(
        target_rank=0,
        kind=FaultKind.RAISE_BEFORE_BACKWARD,
        phase=FaultPhase.BEFORE_OPTIMIZER_STEP,
    )
    result = run_gloo_subprocess_group(1, plan, _default_bounds(), tmp_path)
    assert result.publication_observed is False
    assert result.all_reaped is True
    invocation_dir = Path(result.invocation_dir)
    diagnostics = [
        path.read_bytes().decode("utf-8", errors="replace")
        for path in invocation_dir.iterdir()
        if path.is_file() and path.name != "fault_plan.json"
    ]
    assert any(
        "ddp harness injected fault" in text
        and "rank 0" in text
        and "BEFORE_OPTIMIZER_STEP" in text
        for text in diagnostics
    )


def test_fault_plan_target_rank_outside_world_size_derives_no_culprit(tmp_path):
    # Discriminates a real derivation from a blind copy of the input plan:
    # target_rank=5 never matches any rank in a world_size=2 group, so no
    # rank's code path ever applies the fault and no rank's log ever
    # carries the self-report. The old behavior (culprit_rank read directly
    # from fault_plan.target_rank whenever kind != NONE) would report 5 --
    # a rank that does not exist in this invocation -- instead of None.
    _require_gloo_capability()
    plan = FaultPlan(target_rank=5, kind=FaultKind.RAISE_BEFORE_BACKWARD, phase=FaultPhase.BEFORE_OPTIMIZER_STEP)
    result = run_gloo_subprocess_group(2, plan, _default_bounds(), tmp_path)
    assert result.culprit_rank is None
    assert result.publication_observed is True
    assert result.exit_codes == (0, 0)


@pytest.mark.parametrize("fault_kind", [FaultKind.MISMATCH_COLLECTIVE, FaultKind.MISMATCH_SHAPE])
def test_collective_mismatch_reports_nonzero_exit_evidence(tmp_path, fault_kind):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=fault_kind, phase=FaultPhase.BEFORE_COLLECTIVE)
    result = run_gloo_subprocess_group(2, plan, _default_bounds(), tmp_path)
    assert result.publication_observed is False
    assert result.all_reaped is True
    assert result.culprit_rank == 1
    assert any(code is not None and code != 0 for code in result.exit_codes)


def test_consecutive_successful_invocations_report_disjoint_worker_pids(tmp_path):
    _require_gloo_capability()
    first = run_gloo_subprocess_group(2, None, _default_bounds(), tmp_path)
    second = run_gloo_subprocess_group(2, None, _default_bounds(), tmp_path)
    assert first.publication_observed is True
    assert second.publication_observed is True
    first_pids = {receipt.pid for receipt in first.receipts if isinstance(receipt, RankReceipt)}
    second_pids = {receipt.pid for receipt in second.receipts if isinstance(receipt, RankReceipt)}
    assert len(first_pids) == 2
    assert len(second_pids) == 2
    assert first_pids.isdisjoint(second_pids)


def test_missing_gloo_capability_produces_explicit_attributable_skip():
    from tests.helpers.ddp_capability import GlooSubprocessCapability

    # Force the capability this module's gate actually consults. The gate
    # reads the module-global probed at import time, so patching the probe
    # function itself would not be observed by ``_require_gloo_capability``.
    unavailable = GlooSubprocessCapability(
        gloo_available=False,
        subprocess_spawn_available=True,
        reasons={"gloo_available": "synthetic: gloo backend absent from this torch build"},
    )
    module = sys.modules[__name__]
    original = module._CAPABILITY
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "_CAPABILITY", unavailable)
    try:
        # Execute the real gate, not a reimplementation of it.
        with pytest.raises(pytest.skip.Exception) as excinfo:
            _require_gloo_capability()
    finally:
        monkeypatch.undo()

    # Explicit and attributable: the skip names the missing capability and
    # carries the probe's own reason, so it can never read as a silent pass.
    expected_reason = missing_capability_reason(unavailable, "gloo_available")
    assert str(excinfo.value) == expected_reason
    assert "gloo_available" in str(excinfo.value)
    assert "synthetic: gloo backend absent from this torch build" in str(excinfo.value)

    # Restoration guard, compared against a reference captured before the
    # patch so it cannot pass vacuously: the patch must not leak into the
    # capability gate the other tests in this module depend on.
    assert module._CAPABILITY is original
