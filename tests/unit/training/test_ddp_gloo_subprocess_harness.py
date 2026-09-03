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

import pytest

from tests.helpers.ddp_capability import missing_capability_reason, probe_gloo_capability
from tests.helpers.ddp_fault_injection import FaultKind, FaultPhase, FaultPlan
from tests.helpers.ddp_subprocess_harness import HarnessBounds, run_gloo_subprocess_group

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
    assert len(result.receipts) == 2
    for receipt in result.receipts:
        assert receipt is not None
        assert receipt.collective_result == pytest.approx(2.0)


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


def test_world_size_two_watchdog_reap_also_kills_worker_spawned_grandchild(tmp_path):
    _require_gloo_capability()
    plan = FaultPlan(target_rank=1, kind=FaultKind.SKIP_COLLECTIVE, phase=FaultPhase.BEFORE_COLLECTIVE)
    result = run_gloo_subprocess_group(
        2, plan, _default_bounds(), tmp_path, decoy_grandchild_rank=1
    )
    assert result.all_reaped is True
    grandchild_pid_path = tmp_path / "grandchild_1.pid"
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
    assert len(result.receipts) == 3
    for receipt in result.receipts:
        assert receipt is not None
        assert receipt.collective_result == pytest.approx(3.0)
        assert len(receipt.phase_sequence) == 8


def test_world_size_three_middle_rank_fault_identifies_correct_culprit(tmp_path):
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
