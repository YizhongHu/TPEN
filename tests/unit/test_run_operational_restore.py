"""The typed operational restore entry (S2, item 4023abbb).

``tpen.run.run_from_config`` accepts an optional ``restore_request``: a closed,
versioned ``tpen.execution.recovery.OperationalRestoreRequest`` carrying an
already-selected E1 full-state ``train_resume`` decision. It is injected at
the existing ``Train`` restore boundary -- the runner's ``load`` attribute,
the same seam a concrete configured ``runner.load`` already reaches -- rather
than through a generic mutable config patch. These tests exercise
``tpen._apply_operational_restore`` directly: object-level readback of the
live runner, not config-readback, per
``resume-slices-requirement-object-level-override-readback``.

``OperationalRestoreRequest`` binds a real, on-disk ``CheckpointRef`` (not a
fabricated path string): construction revalidates the exact reference and
derives the closed ``train_resume`` load shape from it (review F2), so every
decision fixture here saves a real checkpoint first.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from tpen.execution.recovery import (
    CheckpointEvidence,
    OperationalRestoreRequest,
    RecoveryDecision,
    RecoveryOutcome,
    RecoveryPhase,
    RecoveryRefusal,
    RefusalCode,
    capture_science_intent,
    select_recovery,
    train_resume_load_overlay,
)
from tpen.run import _apply_operational_restore
from tpen.runner.base import Runner

_TARGET_ITERATIONS = 10


def _save_checkpoint(tmp_path: Path, *, step: int = 3):
    """Write a real, complete, train_resume-eligible checkpoint and return its ref."""

    import torch
    from tpen.checkpoint import TrainResume, save_checkpoint
    from tpen.checkpoint.reference import CheckpointRef

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(
        state_dict=lambda: {"next_iteration": step, "completed_updates": step}
    )
    sampler = SimpleNamespace(mcmc_state_dict=lambda: {"walker": step})
    context = SimpleNamespace(
        cfg=OmegaConf.create({"study": {"name": "s2-restore"}, "trainer": {"max_steps": _TARGET_ITERATIONS}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id=f"s2-restore-{step}", extra={}),
        run_dir=tmp_path,
    )
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=step,
        completed_updates=step,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
        payload=TrainResume(),
        context=context,
    )
    return CheckpointRef.from_directory(checkpoint_dir)


def _recovery_decision(tmp_path: Path, *, ref=None, **overrides) -> RecoveryDecision:
    ref = ref if ref is not None else _save_checkpoint(tmp_path)
    fields = dict(
        phase=RecoveryPhase.RECOVERY,
        outcome=RecoveryOutcome.RECOVERY,
        target_iterations=_TARGET_ITERATIONS,
        next_iteration=ref.next_iteration,
        completed_updates=ref.completed_updates,
        checkpoint=ref,
        load=train_resume_load_overlay(ref),
    )
    fields.update(overrides)
    return RecoveryDecision(**fields)


class _LoadableRunner(Runner):
    """Minimal typed runner fixture exposing the restore boundary attributes."""

    def __init__(self, load=None, *, max_steps=_TARGET_ITERATIONS) -> None:
        self.load = load
        self.trainer = SimpleNamespace(max_steps=max_steps)

class _RunnerWithoutLoad(Runner):
    """Typed runner fixture used to verify that a missing ``load`` is refused."""

    def __init__(self, *, max_steps=_TARGET_ITERATIONS) -> None:
        self.trainer = SimpleNamespace(max_steps=max_steps)


# ---------------------------------------------------------------------------
# OperationalRestoreRequest construction
# ---------------------------------------------------------------------------


def test_request_derives_the_closed_load_shape_from_the_ref(tmp_path):
    ref = _save_checkpoint(tmp_path)
    decision = _recovery_decision(tmp_path, ref=ref)
    request = OperationalRestoreRequest(decision=decision)

    assert request.load == {"mode": "train_resume", "path": str(ref.checkpoint_dir)}
    assert request.target_iterations == _TARGET_ITERATIONS
    assert request.ref == ref
    # The property must not alias a shared mutable mapping.
    request.load["mode"] = "mutated"
    assert request.load == {"mode": "train_resume", "path": str(ref.checkpoint_dir)}


def test_mutating_the_source_decision_after_construction_does_not_change_the_request(tmp_path):
    decision = _recovery_decision(tmp_path)
    request = OperationalRestoreRequest(decision=decision)
    before = request.load

    decision.load["mode"] = "none"
    decision.load["path"] = None

    assert request.load == before
    assert request.load["mode"] == "train_resume"


def test_request_rejects_a_non_recovery_decision():
    with pytest.raises(TypeError):
        OperationalRestoreRequest(decision={"mode": "train_resume"})


def test_request_rejects_a_refused_decision():
    refused = RecoveryDecision(
        phase=RecoveryPhase.RECOVERY,
        outcome=RecoveryOutcome.REFUSED,
        target_iterations=_TARGET_ITERATIONS,
        load=None,
        refusal=RecoveryRefusal(code=RefusalCode.NO_CHECKPOINT, message="no checkpoint"),
    )
    with pytest.raises(ValueError):
        OperationalRestoreRequest(decision=refused)


def test_request_rejects_a_decision_without_a_train_resume_load():
    initial = RecoveryDecision(
        phase=RecoveryPhase.INITIAL,
        outcome=RecoveryOutcome.INITIAL,
        target_iterations=_TARGET_ITERATIONS,
        next_iteration=0,
        completed_updates=0,
        load=None,
    )
    with pytest.raises(ValueError):
        OperationalRestoreRequest(decision=initial)


def test_request_rejects_a_model_only_load(tmp_path):
    ref = _save_checkpoint(tmp_path)
    model_only = _recovery_decision(
        tmp_path, ref=ref, load={"mode": "model_only", "path": str(ref.checkpoint_dir)}
    )
    with pytest.raises(ValueError):
        OperationalRestoreRequest(decision=model_only)


def test_request_rejects_extra_load_options(tmp_path):
    ref = _save_checkpoint(tmp_path)
    overlaid = _recovery_decision(
        tmp_path, ref=ref,
        load={"mode": "train_resume", "path": str(ref.checkpoint_dir), "strict": False},
    )
    with pytest.raises(ValueError):
        OperationalRestoreRequest(decision=overlaid)


def test_request_rejects_a_load_path_disagreeing_with_the_checkpoint_ref(tmp_path):
    ref = _save_checkpoint(tmp_path)
    mismatched = _recovery_decision(
        tmp_path, ref=ref, load={"mode": "train_resume", "path": "/not/the/ref/path"},
    )
    with pytest.raises(ValueError):
        OperationalRestoreRequest(decision=mismatched)


def test_request_rejects_a_decision_without_a_checkpoint_ref():
    bare = RecoveryDecision(
        phase=RecoveryPhase.RECOVERY,
        outcome=RecoveryOutcome.RECOVERY,
        target_iterations=_TARGET_ITERATIONS,
        next_iteration=3,
        completed_updates=3,
        checkpoint=None,
        load={"mode": "train_resume", "path": "/ckpt/step_000003"},
    )
    with pytest.raises(TypeError):
        OperationalRestoreRequest(decision=bare)


def test_request_revalidates_checkpoint_bytes_mutated_after_the_ref_was_built(tmp_path):
    """Construction must call ``ref.validate()``, not just ``read_manifest``.

    ``read_manifest`` alone re-reads the manifest file but never cross-checks
    it against the ref's cached hash fields, so it would accept a checkpoint
    whose ``model.pt`` bytes changed out from under an already-selected ref
    (a TOCTOU-shaped gap between E1 selection and request construction).
    """

    ref = _save_checkpoint(tmp_path)
    decision = _recovery_decision(tmp_path, ref=ref)

    # Green control: untouched bytes still construct successfully.
    OperationalRestoreRequest(decision=decision)

    model_path = ref.checkpoint_dir / "model.pt"
    mutated = bytearray(model_path.read_bytes())
    mutated[0] ^= 0xFF
    model_path.write_bytes(bytes(mutated))

    with pytest.raises(ValueError, match="no longer match"):
        OperationalRestoreRequest(decision=decision)


def _select_recovery_decision(tmp_path: Path, *, ref=None) -> RecoveryDecision:
    """Build a decision through the real E1 ``select_recovery`` selection path."""

    ref = ref if ref is not None else _save_checkpoint(tmp_path)
    intent = capture_science_intent({"study": "s2-restore"}, target_iterations=_TARGET_ITERATIONS)
    evidence = CheckpointEvidence.from_ref(
        ref,
        science_input_digest=intent.science_input_digest,
        target_iterations=_TARGET_ITERATIONS,
    )
    return select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)


def test_request_accepts_an_unmodified_select_recovery_result(tmp_path):
    """Green control: a real, untouched E1 selection must still construct."""

    decision = _select_recovery_decision(tmp_path)
    request = OperationalRestoreRequest(decision=decision)
    assert request.next_iteration == decision.next_iteration
    assert request.completed_updates == decision.completed_updates


@pytest.mark.parametrize("field", ["next_iteration", "completed_updates"])
def test_request_rejects_decision_counters_disagreeing_with_their_ref(tmp_path, field):
    """R2.2: a decision counter altered after selection must refuse, not be
    silently frozen -- construction previously only checked nonnegativity."""

    decision = _select_recovery_decision(tmp_path)
    from dataclasses import replace

    tampered = replace(decision, **{field: getattr(decision, field) + 1})
    with pytest.raises(ValueError, match=field):
        OperationalRestoreRequest(decision=tampered)


def test_request_rejects_a_checkpoint_beyond_the_target_horizon(tmp_path):
    """R2.2: ref.next_iteration must not exceed target_iterations, mirroring
    E1's own gate at ``_checkpoint_refusal`` (recovery.py:517-521)."""

    ref = _save_checkpoint(tmp_path, step=_TARGET_ITERATIONS + 1)
    decision = RecoveryDecision(
        phase=RecoveryPhase.RECOVERY,
        outcome=RecoveryOutcome.RECOVERY,
        target_iterations=_TARGET_ITERATIONS,
        next_iteration=ref.next_iteration,
        completed_updates=ref.completed_updates,
        checkpoint=ref,
        load=train_resume_load_overlay(ref),
    )
    with pytest.raises(ValueError, match="horizon"):
        OperationalRestoreRequest(decision=decision)


def test_request_rejects_completed_updates_beyond_horizon_with_next_iteration_in_range(tmp_path):
    """R3.2: E1's own gate (recovery.py:517-521) refuses when EITHER counter
    exceeds the fixed horizon, but construction previously bounded only
    ``ref.next_iteration``. Vary ``completed_updates`` alone beyond H while
    ``next_iteration`` stays within it, with the manifest and trainer-cursor
    counters mutually consistent, to isolate the omitted check."""

    import torch
    from tpen.checkpoint import TrainResume, save_checkpoint
    from tpen.checkpoint.reference import CheckpointRef

    next_iteration = _TARGET_ITERATIONS
    completed_updates = _TARGET_ITERATIONS + 1
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(
        state_dict=lambda: {
            "next_iteration": next_iteration,
            "completed_updates": completed_updates,
        }
    )
    sampler = SimpleNamespace(mcmc_state_dict=lambda: {"walker": next_iteration})
    context = SimpleNamespace(
        cfg=OmegaConf.create({"study": {"name": "s2-restore"}, "trainer": {"max_steps": _TARGET_ITERATIONS}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id="s2-restore-beyond-h", extra={}),
        run_dir=tmp_path,
    )
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=next_iteration,
        completed_updates=completed_updates,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
        payload=TrainResume(),
        context=context,
    )
    ref = CheckpointRef.from_directory(checkpoint_dir)
    assert ref.next_iteration <= _TARGET_ITERATIONS
    assert ref.completed_updates > _TARGET_ITERATIONS

    decision = RecoveryDecision(
        phase=RecoveryPhase.RECOVERY,
        outcome=RecoveryOutcome.RECOVERY,
        target_iterations=_TARGET_ITERATIONS,
        next_iteration=ref.next_iteration,
        completed_updates=ref.completed_updates,
        checkpoint=ref,
        load=train_resume_load_overlay(ref),
    )
    with pytest.raises(ValueError, match="horizon"):
        OperationalRestoreRequest(decision=decision)


@pytest.mark.parametrize(
    ("phase", "outcome"),
    [
        (RecoveryPhase.RECOVERY, RecoveryOutcome.INITIAL_CHECKPOINT),
        (RecoveryPhase.INITIAL, RecoveryOutcome.RECOVERY),
        (RecoveryPhase.RECOVERY, RecoveryOutcome.NO_CHECKPOINT),
    ],
)
def test_request_rejects_an_invalid_phase_outcome_combination(tmp_path, phase, outcome):
    """R2.2: only (INITIAL, INITIAL_CHECKPOINT) and (RECOVERY, RECOVERY) are
    valid selected-checkpoint combinations."""

    ref = _save_checkpoint(tmp_path)
    decision = RecoveryDecision(
        phase=phase,
        outcome=outcome,
        target_iterations=_TARGET_ITERATIONS,
        next_iteration=ref.next_iteration,
        completed_updates=ref.completed_updates,
        checkpoint=ref,
        load=train_resume_load_overlay(ref),
    )
    with pytest.raises(ValueError, match="phase/outcome"):
        OperationalRestoreRequest(decision=decision)


# ---------------------------------------------------------------------------
# tpen.run._apply_operational_restore
# ---------------------------------------------------------------------------


def test_injects_the_decision_load_into_the_live_runner_when_unconfigured(tmp_path):
    cfg = OmegaConf.create({"runner": {}})
    runner = _LoadableRunner(load=None)
    request = OperationalRestoreRequest(decision=_recovery_decision(tmp_path))

    result = _apply_operational_restore(cfg, runner, request)

    assert result is runner
    assert runner.load == request.load
    # cfg itself is never mutated: the request carries no config patch.
    assert OmegaConf.select(cfg, "runner.load", default=None) is None


def test_agreeing_concrete_configured_load_is_accepted(tmp_path):
    request = OperationalRestoreRequest(decision=_recovery_decision(tmp_path))
    cfg = OmegaConf.create({"runner": {"load": dict(request.load)}})
    runner = _LoadableRunner(load=dict(request.load))

    _apply_operational_restore(cfg, runner, request)

    assert runner.load == request.load


def test_conflicting_concrete_configured_load_is_refused_not_shadowed(tmp_path):
    request = OperationalRestoreRequest(decision=_recovery_decision(tmp_path))
    cfg = OmegaConf.create({"runner": {"load": {"mode": "train_resume", "path": "/elsewhere"}}})
    runner = _LoadableRunner(load={"mode": "train_resume", "path": "/elsewhere"})

    with pytest.raises(ValueError):
        _apply_operational_restore(cfg, runner, request)

    # Refusal happens before mutation: the runner's own configured load survives.
    assert runner.load == {"mode": "train_resume", "path": "/elsewhere"}


@pytest.mark.parametrize(
    "unconfigured_load", [{"mode": "none", "path": None}, {"mode": None, "path": None}, None],
)
def test_supported_no_load_forms_are_treated_as_unconfigured(tmp_path, unconfigured_load):
    """F4: an explicit no-load shape (or ``${load}`` resolving to it) is not a
    competing concrete load; it must not refuse an admissible request."""

    request = OperationalRestoreRequest(decision=_recovery_decision(tmp_path))
    cfg = (
        OmegaConf.create({"runner": {"load": unconfigured_load}})
        if unconfigured_load is not None
        else OmegaConf.create({"runner": {}})
    )
    runner = _LoadableRunner(load=unconfigured_load)

    _apply_operational_restore(cfg, runner, request)

    assert runner.load == request.load


def test_runner_without_a_load_attribute_is_refused(tmp_path):
    cfg = OmegaConf.create({"runner": {}})
    runner = _RunnerWithoutLoad()
    request = OperationalRestoreRequest(decision=_recovery_decision(tmp_path))

    with pytest.raises(TypeError):
        _apply_operational_restore(cfg, runner, request)


def test_non_request_object_is_refused(tmp_path):
    cfg = OmegaConf.create({"runner": {}})
    runner = _LoadableRunner(load=None)

    with pytest.raises(TypeError):
        _apply_operational_restore(cfg, runner, {"mode": "train_resume", "path": "/ckpt"})


def test_mismatched_live_trainer_max_steps_is_refused_before_restore(tmp_path):
    """F3: a live constructed ``runner.trainer.max_steps`` that disagrees with
    the decision's fixed target horizon must refuse before touching ``load``."""

    cfg = OmegaConf.create({"runner": {}})
    runner = _LoadableRunner(load=None, max_steps=_TARGET_ITERATIONS + 1)
    request = OperationalRestoreRequest(decision=_recovery_decision(tmp_path))

    with pytest.raises(ValueError, match="horizon|max_steps"):
        _apply_operational_restore(cfg, runner, request)

    assert runner.load is None


def test_matching_live_trainer_max_steps_is_a_green_control(tmp_path):
    cfg = OmegaConf.create({"runner": {}})
    runner = _LoadableRunner(load=None, max_steps=_TARGET_ITERATIONS)
    request = OperationalRestoreRequest(decision=_recovery_decision(tmp_path))

    _apply_operational_restore(cfg, runner, request)

    assert runner.load == request.load
