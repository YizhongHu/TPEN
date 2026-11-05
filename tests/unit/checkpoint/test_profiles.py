"""Unit tests for the declared core recovery profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.checkpoint.profiles import (
    NATIVE_PROFILE,
    NATIVE_RECOVERY_PROFILE,
    PROFILE_CHECK_SCHEMA,
    REFUSAL_CODES,
    SUPPORTED_STATE_OWNERS,
    ProfileCheck,
    check_profile,
)
from tpen.execution import NATIVE_PROFILE_VERSION
from tpen.execution.native_profile import (
    SUPPORTED_CALLBACK_TARGETS,
    SUPPORTED_LOGGER_TARGETS,
    validate_native_recovery_profile,
)


def _config(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "callbacks": [{"_target_": "tpen.callback.Status"}],
        "loggers": [{"_target_": "tpen.logging.JSONL"}],
    }
    value.update(overrides)
    return value


def test_supported_profile_reports_declared_state_owners() -> None:
    result = check_profile(OmegaConf.create(_config()), NATIVE_RECOVERY_PROFILE)
    assert result.schema == PROFILE_CHECK_SCHEMA
    assert result.verdict == "supported"
    assert result.state_owners == SUPPORTED_STATE_OWNERS
    assert result.refusal is None


@pytest.mark.parametrize(
    ("owner", "target", "code"),
    [
        ("callbacks", "tests.helpers.PrivateMutableCallback", "unsupported-callback"),
        ("loggers", "tests.helpers.PrivateLogger", "unsupported-logger"),
    ],
)
def test_unsupported_callback_or_logger_is_typed_refusal(owner: str, target: str, code: str) -> None:
    result = check_profile(_config(**{owner: [{"_target_": target}]}), NATIVE_RECOVERY_PROFILE)
    assert result.verdict == "refused"
    assert result.refusal == {"code": code, "reason": result.refusal["reason"]}


def test_callback_state_negative_control_refuses_unknown_callback() -> None:
    result = check_profile(
        _config(callbacks=[{"_target_": "tests.helpers.CallbackWithState", "state": {"count": 1}}]),
        NATIVE_RECOVERY_PROFILE,
    )
    assert result.refusal is not None
    assert result.refusal["code"] == "unsupported-callback"


def test_unknown_profile_is_refused() -> None:
    result = check_profile(_config(), "tpen.unknown/v1")
    assert result.verdict == "refused"
    assert result.refusal["code"] == "unknown-profile"


@pytest.mark.parametrize(
    "config",
    [None, {"callbacks": "not-a-sequence"}, {"callbacks": [{"state": 1}]}],
)
def test_malformed_config_is_refused(config: object) -> None:
    result = check_profile(config, NATIVE_RECOVERY_PROFILE)
    assert result.refusal is not None
    assert result.refusal["code"] == "malformed-config"


def test_profile_record_is_immutable_and_hash_consistent() -> None:
    first = check_profile(_config(), NATIVE_RECOVERY_PROFILE)
    second = ProfileCheck.from_mapping(first.to_dict())
    assert first == second
    assert hash(first) == hash(second)
    with pytest.raises((TypeError, AttributeError)):
        first.state_owners.add("new-owner")  # type: ignore[attr-defined]


def test_native_constants_are_rehomed_without_value_change() -> None:
    assert NATIVE_PROFILE.name == NATIVE_RECOVERY_PROFILE
    assert NATIVE_PROFILE.state_owners == SUPPORTED_STATE_OWNERS
    assert NATIVE_PROFILE.callbacks == SUPPORTED_CALLBACK_TARGETS
    assert NATIVE_PROFILE.loggers == SUPPORTED_LOGGER_TARGETS
    assert NATIVE_PROFILE.callback_state_supported is False
    assert NATIVE_PROFILE_VERSION == "tpen.execution.native_profile:SUPPORTED_STATE_OWNERS@e5"
    validate_native_recovery_profile(OmegaConf.create(_config()))


def test_cli_output_is_one_json_object(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("callbacks:\n  - _target_: tpen.callback.Status\nloggers: []\n", encoding="utf-8")
    from tpen.checkpoint.profiles import _main

    assert _main([str(config), "--profile", NATIVE_RECOVERY_PROFILE]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "supported"
    assert payload["state_owners"] == sorted(SUPPORTED_STATE_OWNERS)


assert REFUSAL_CODES == {"unknown-profile", "unsupported-callback", "unsupported-logger", "malformed-config"}
