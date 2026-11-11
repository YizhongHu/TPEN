"""Unrun C3 probes for VMCTrainer method-state durability.

These tests intentionally bind ``_resolved_update_method`` directly.  They
exercise the narrow trainer checkpoint seam without requiring a model fit.
Run them in the normal torch environment on Cannon; torch is unavailable in
this review worktree.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import pytest
import torch

import tpen.checkpoint.save as save_module
from tests.unit.callback.test_checkpoint import _write_checkpoint
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    ModelParameterBinding,
    VMCUpdateMethod,
    VMCUpdateResult,
)


class _StatefulVMCUpdateMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """Small persistent method with a counter and convention fingerprint."""

    def __init__(self, parameter: torch.nn.Parameter, *, counter: int, fingerprint: str):
        self.optimizer = torch.optim.SGD((parameter,), lr=0.1)
        self.model_parameters = ModelParameterBinding.from_parameters((parameter,))
        self.counter = counter
        self.fingerprint = fingerprint

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        del update_input
        raise AssertionError("durability probe must not fit a model")

    def update_state(self):
        # The trainer only needs this method's real owned state to be present;
        # no resolve call is needed for this intentionally narrower seam.
        from tpen.training.update import VMCUpdateState

        return VMCUpdateState(
            optimizer=self.optimizer,
            model_parameters=self.model_parameters,
        )

    def method_state_dict(self) -> Mapping[str, object]:
        return {
            "version": 1,
            "counter": self.counter,
            "fingerprint": self.fingerprint,
        }

    def load_method_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported method-state version")
        self.counter = int(state["counter"])
        self.fingerprint = str(state["fingerprint"])


class _StatelessVMCUpdateMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """Base-loader target used to assert non-empty state is refused."""

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        del update_input
        raise AssertionError("durability probe must not fit a model")


def _trainer(method: VMCUpdateMethod[AutogradUpdateInput]) -> VMCTrainer:
    trainer = VMCTrainer(max_steps=1)
    # This is deliberate: C3 targets the state payload seam, not fit setup.
    trainer._resolved_update_method = method
    return trainer


def _stateful_roundtrip() -> tuple[dict, _StatefulVMCUpdateMethod, _StatefulVMCUpdateMethod]:
    source_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    source_method = _StatefulVMCUpdateMethod(
        source_parameter, counter=17, fingerprint="source-fingerprint"
    )
    source = _trainer(source_method)
    encoded = json.dumps(source.state_dict())
    payload = json.loads(encoded)

    target_parameter = torch.nn.Parameter(torch.tensor([99.0]))
    target_method = _StatefulVMCUpdateMethod(
        target_parameter, counter=-4, fingerprint="different-fingerprint"
    )
    _trainer(target_method).load_state_dict(payload)
    return payload, source_method, target_method


def test_stateful_method_payload_is_json_safe_and_restores_different_initial_values() -> None:
    payload, source, target = _stateful_roundtrip()

    assert payload["update_method"] == {
        "version": 1,
        "counter": 17,
        "fingerprint": "source-fingerprint",
    }
    assert (target.counter, target.fingerprint) == (source.counter, source.fingerprint)
    assert (target.counter, target.fingerprint) != (-4, "different-fingerprint")


def test_stateful_target_rejects_checkpoint_missing_method_payload() -> None:
    payload, _, _ = _stateful_roundtrip()
    del payload["update_method"]

    target = _StatefulVMCUpdateMethod(
        torch.nn.Parameter(torch.tensor([99.0])),
        counter=-4,
        fingerprint="different-fingerprint",
    )
    with pytest.raises(ValueError, match="update_method"):
        _trainer(target).load_state_dict(payload)


def test_stateless_target_rejects_nonempty_method_payload_via_real_trainer_loader() -> None:
    payload, _, _ = _stateful_roundtrip()

    with pytest.raises(ValueError, match="stateless VMCUpdateMethod"):
        _trainer(_StatelessVMCUpdateMethod()).load_state_dict(payload)


def test_base_method_loader_refuses_nonempty_method_state() -> None:
    with pytest.raises(ValueError, match="stateless VMCUpdateMethod"):
        VMCUpdateMethod.load_method_state_dict(
            _StatelessVMCUpdateMethod(), {"counter": 3}
        )


def _assert_roundtrip_restores() -> None:
    _, source, target = _stateful_roundtrip()
    assert target.counter == source.counter
    assert target.fingerprint == source.fingerprint


def test_red_control_noop_method_loader_is_discriminated(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_op(self, state):
        del self, state

    monkeypatch.setattr(_StatefulVMCUpdateMethod, "load_method_state_dict", no_op)
    with pytest.raises(AssertionError):
        _assert_roundtrip_restores()


def test_red_control_omitted_trainer_method_payload_is_discriminated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = VMCTrainer.state_dict

    def omit_method_payload(self):
        state = original(self)
        state.pop("update_method", None)
        return state

    monkeypatch.setattr(VMCTrainer, "state_dict", omit_method_payload)
    try:
        _assert_roundtrip_restores()
    except ValueError as caught:
        # The production guard is itself a valid mutant-catching oracle: it
        # refuses a stateful resume before the helper can reach its equality
        # assertion. Keep the check specific so unrelated ValueErrors do not
        # make this control vacuous.
        assert "update_method" in str(caught)
    else:
        raise AssertionError("omitted method payload was not discriminated")


def test_red_control_noop_trainer_restore_path_is_discriminated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(VMCTrainer, "load_state_dict", lambda self, state: None)
    with pytest.raises(AssertionError):
        _assert_roundtrip_restores()


def test_save_oserror_never_commits_a_partial_final_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted component write cannot create ``step_000003``."""

    def fail_save(*args, **kwargs) -> None:
        del args, kwargs
        raise OSError("simulated component-write failure")

    # ``save_checkpoint`` imports torch inside the function; this patches the
    # same torch module object used by ``tpen.checkpoint.save``.
    monkeypatch.setattr(torch, "save", fail_save)
    root = tmp_path / "checkpoints"

    with pytest.raises(OSError, match="simulated component-write failure"):
        _write_checkpoint(tmp_path)

    assert not (root / "step_000003").exists()


def test_save_oserror_cleanup_removes_temporary_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The broad ``Exception`` handler cleans up an ordinary write error."""

    def fail_save(*args, **kwargs) -> None:
        del args, kwargs
        raise OSError("simulated component-write failure")

    monkeypatch.setattr(save_module, "torch", torch, raising=False)
    monkeypatch.setattr(torch, "save", fail_save)
    root = tmp_path / "checkpoints"

    with pytest.raises(OSError, match="simulated component-write failure"):
        _write_checkpoint(tmp_path)

    assert not (root / "step_000003.tmp").exists()


def test_keyboardinterrupt_never_commits_final_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KeyboardInterrupt bypasses cleanup but still cannot commit by rename."""

    def interrupt_save(*args, **kwargs) -> None:
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", interrupt_save)
    root = tmp_path / "checkpoints"

    with pytest.raises(KeyboardInterrupt):
        _write_checkpoint(tmp_path)

    assert not (root / "step_000003").exists()


def test_keyboardinterrupt_cleanup_removes_the_temporary_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was a STRICT XFAIL when handed over. The defect it documented is fixed.

    The lane reviewer wrote this as a deliberate red control: it asserted the
    stronger cleanup claim and was marked ``xfail(strict=True)`` because
    ``save_checkpoint`` used ``except Exception``, which does not catch
    ``KeyboardInterrupt`` -- a ``BaseException``. Round 2 changed that handler
    to a ``finally``, so this now passes, and a strict xfail that passes FAILS
    the suite. The marker is removed rather than the test.

    Kept as a normal expectation instead of deleted: it is the only case
    covering interruption by a BaseException, and it is precisely the case the
    old handler missed.

    NOT a claim of general interruption safety. A ``SIGKILL``, or a ``SIGTERM``
    under Python's default handler, terminates without unwinding and runs no
    ``finally`` -- a scheduler timeout is that case. This covers interruption
    where the interpreter still unwinds. Whether the tmp-then-rename commit
    holds on the deployed Netscratch filesystem is separate and open at item
    3b9b736a.
    """

    def interrupt_save(*args, **kwargs) -> None:
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", interrupt_save)
    root = tmp_path / "checkpoints"

    with pytest.raises(KeyboardInterrupt):
        _write_checkpoint(tmp_path)

    # This assertion is expected to fail against the current save.py shape:
    # the temporary directory remains after BaseException escapes.
    assert not (root / "step_000003.tmp").exists()
