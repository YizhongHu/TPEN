"""Tests for the explicit E5 native mutable-state profile."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.execution import UnsupportedMutableStateError, validate_native_recovery_profile


FIXTURE = Path(__file__).resolve().parents[2] / "integration" / "artifacts" / "execution" / "native_recovery.yaml"


def test_native_fixture_uses_only_supported_state_owners() -> None:
    validate_native_recovery_profile(OmegaConf.load(FIXTURE))


def test_native_profile_refuses_an_unclassified_private_callback() -> None:
    config = OmegaConf.create(
        {"callbacks": [{"_target_": "tests.helpers.PrivateMutableCallback"}], "loggers": []}
    )
    with pytest.raises(UnsupportedMutableStateError, match="not in the native mutable-state profile"):
        validate_native_recovery_profile(config)


def test_native_profile_does_not_resolve_callback_targets() -> None:
    config = OmegaConf.create(
        {
            "callbacks": [{"_target_": "tests.helpers.ThisConstructorMustNotRun"}],
            "loggers": [],
        }
    )
    with pytest.raises(UnsupportedMutableStateError):
        validate_native_recovery_profile(config)
