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
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from tpen.execution.recovery import (
    OperationalRestoreRequest,
    RecoveryDecision,
    RecoveryOutcome,
    RecoveryPhase,
    RecoveryRefusal,
    RefusalCode,
)
from tpen.run import _apply_operational_restore

_DECISION_LOAD = {"mode": "train_resume", "path": "/ckpt/step_000003"}


def _recovery_decision(**overrides) -> RecoveryDecision:
    fields = dict(
        phase=RecoveryPhase.RECOVERY,
        outcome=RecoveryOutcome.RECOVERY,
        target_iterations=10,
        next_iteration=3,
        completed_updates=3,
        load=dict(_DECISION_LOAD),
    )
    fields.update(overrides)
    return RecoveryDecision(**fields)


class _LoadableRunner:
    """Duck-typed stand-in for ``tpen.runner.train.Train``: exposes ``load``."""

    def __init__(self, load=None) -> None:
        self.load = load


# ---------------------------------------------------------------------------
# OperationalRestoreRequest construction
# ---------------------------------------------------------------------------


def test_request_exposes_the_decisions_load_mapping():
    request = OperationalRestoreRequest(decision=_recovery_decision())
    assert request.load == _DECISION_LOAD
    # The property must not alias the decision's own frozen mapping, so a
    # caller mutating the returned dict cannot reach back into the decision.
    request.load["mode"] = "mutated"
    assert request.decision.load == _DECISION_LOAD


def test_request_rejects_a_non_recovery_decision():
    with pytest.raises(TypeError):
        OperationalRestoreRequest(decision={"mode": "train_resume"})


def test_request_rejects_a_refused_decision():
    refused = _recovery_decision(
        outcome=RecoveryOutcome.REFUSED,
        load=None,
        refusal=RecoveryRefusal(code=RefusalCode.NO_CHECKPOINT, message="no checkpoint"),
    )
    with pytest.raises(ValueError):
        OperationalRestoreRequest(decision=refused)


def test_request_rejects_a_decision_without_a_train_resume_load():
    initial = _recovery_decision(
        phase=RecoveryPhase.INITIAL, outcome=RecoveryOutcome.INITIAL, next_iteration=0,
        completed_updates=0, load=None,
    )
    with pytest.raises(ValueError):
        OperationalRestoreRequest(decision=initial)


def test_request_rejects_a_model_only_load():
    model_only = _recovery_decision(load={"mode": "model_only", "path": "/ckpt"})
    with pytest.raises(ValueError):
        OperationalRestoreRequest(decision=model_only)


# ---------------------------------------------------------------------------
# tpen.run._apply_operational_restore
# ---------------------------------------------------------------------------


def test_injects_the_decision_load_into_the_live_runner_when_unconfigured():
    cfg = OmegaConf.create({"runner": {}})
    runner = _LoadableRunner(load=None)
    request = OperationalRestoreRequest(decision=_recovery_decision())

    result = _apply_operational_restore(cfg, runner, request)

    assert result is runner
    assert runner.load == _DECISION_LOAD
    # cfg itself is never mutated: the request carries no config patch.
    assert OmegaConf.select(cfg, "runner.load", default=None) is None


def test_agreeing_concrete_configured_load_is_accepted():
    cfg = OmegaConf.create({"runner": {"load": dict(_DECISION_LOAD)}})
    runner = _LoadableRunner(load=dict(_DECISION_LOAD))
    request = OperationalRestoreRequest(decision=_recovery_decision())

    _apply_operational_restore(cfg, runner, request)

    assert runner.load == _DECISION_LOAD


def test_conflicting_concrete_configured_load_is_refused_not_shadowed():
    cfg = OmegaConf.create({"runner": {"load": {"mode": "train_resume", "path": "/elsewhere"}}})
    runner = _LoadableRunner(load={"mode": "train_resume", "path": "/elsewhere"})
    request = OperationalRestoreRequest(decision=_recovery_decision())

    with pytest.raises(ValueError):
        _apply_operational_restore(cfg, runner, request)

    # Refusal happens before mutation: the runner's own configured load survives.
    assert runner.load == {"mode": "train_resume", "path": "/elsewhere"}


def test_runner_without_a_load_attribute_is_refused():
    cfg = OmegaConf.create({"runner": {}})
    runner = SimpleNamespace()
    request = OperationalRestoreRequest(decision=_recovery_decision())

    with pytest.raises(TypeError):
        _apply_operational_restore(cfg, runner, request)


def test_non_request_object_is_refused():
    cfg = OmegaConf.create({"runner": {}})
    runner = _LoadableRunner(load=None)

    with pytest.raises(TypeError):
        _apply_operational_restore(cfg, runner, {"mode": "train_resume", "path": "/ckpt"})
