"""Unit tests for the declared core recovery profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.checkpoint.profiles import (
    CALLBACK_CADENCE_REFUSAL_CODE,
    NATIVE_PROFILE,
    NATIVE_RECOVERY_PROFILE,
    Profile,
    PROFILE_CHECK_SCHEMA,
    REFUSAL_CODES,
    SUPPORTED_STATE_OWNERS,
    ProfileCheck,
    check_resume_sensitive_callback_cadence,
    check_profile,
)
from tpen.checkpoint import profiles
from tpen.execution import NATIVE_PROFILE_VERSION
from tpen.execution.native_profile import (
    SUPPORTED_CALLBACK_TARGETS,
    SUPPORTED_LOGGER_TARGETS,
    UnsupportedMutableStateError,
    validate_native_recovery_profile,
)


def _config(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "callbacks": [{"_target_": "tpen.callback.Status"}],
        "loggers": [{"_target_": "tpen.logging.JSONL"}],
    }
    value.update(overrides)
    return value


ROOT = Path(__file__).parents[3]
H_V2_TRAIN_CONFIG = ROOT / "experiments" / "atomistic" / "h-v2" / "configs" / "train.yaml"
NEW_CALLBACK_TARGETS = frozenset(
    {
        "tpen.callback.Checkpoint",
        "tpen.callback.DataIntegrity",
        "tpen.callback.GradientStats",
    }
)


def test_supported_profile_reports_declared_state_owners() -> None:
    result = check_profile(OmegaConf.create(_config()), NATIVE_RECOVERY_PROFILE)
    assert result.schema == PROFILE_CHECK_SCHEMA
    assert result.verdict == "supported"
    assert result.state_owners == SUPPORTED_STATE_OWNERS
    assert result.refusal is None


def test_h_v2_real_train_config_is_admitted_through_profile_loader() -> None:
    """The executor's profile subprocess path accepts the real H-v2 config."""

    result = check_profile(OmegaConf.load(H_V2_TRAIN_CONFIG), NATIVE_RECOVERY_PROFILE)

    assert result.verdict == "supported"
    assert result.refusal is None


def test_unknown_callback_target_remains_refused() -> None:
    result = check_profile(
        OmegaConf.create(
            {
                "callbacks": [{"_target_": "tpen.callback.NotARealCallback"}],
                "loggers": [],
            }
        ),
        NATIVE_RECOVERY_PROFILE,
    )

    assert result.verdict == "refused"
    assert result.refusal["code"] == "unsupported-callback"


@pytest.mark.parametrize("target", sorted(NEW_CALLBACK_TARGETS))
def test_each_new_callback_target_is_admitted(target: str) -> None:
    config = OmegaConf.to_container(OmegaConf.load(H_V2_TRAIN_CONFIG), resolve=False)
    config["callbacks"].append({"_target_": target})

    result = check_profile(config, NATIVE_RECOVERY_PROFILE)

    assert result.verdict == "supported"
    assert result.refusal is None


@pytest.mark.parametrize("target", sorted(NEW_CALLBACK_TARGETS))
@pytest.mark.parametrize(
    ("field", "value"),
    [("max_calls", 1), ("probability", 0.5)],
)
def test_resume_sensitive_cadence_is_refused_by_both_validators(
    target: str, field: str, value: object
) -> None:
    config = _config(callbacks=[{"_target_": target, field: value}])

    result = check_profile(config, NATIVE_RECOVERY_PROFILE)

    assert result.verdict == "refused"
    assert result.refusal is not None
    assert result.refusal["code"] == CALLBACK_CADENCE_REFUSAL_CODE
    assert field in result.refusal["reason"]
    with pytest.raises(UnsupportedMutableStateError, match=field):
        validate_native_recovery_profile(config)


@pytest.mark.parametrize("target", sorted(NEW_CALLBACK_TARGETS))
def test_reconstructible_cadence_is_admitted_by_both_validators(target: str) -> None:
    config = _config(
        callbacks=[
            {
                "_target_": target,
                "every_n_steps": 100,
                "max_calls": None,
                "probability": 1.0,
            }
        ]
    )

    result = check_profile(config, NATIVE_RECOVERY_PROFILE)

    assert result.verdict == "supported"
    assert result.refusal is None
    validate_native_recovery_profile(config)


def test_resume_sensitive_cadence_helper_audits_pop_step_cadence_fields() -> None:
    assert check_resume_sensitive_callback_cadence(
        _config(
            callbacks=[
                {
                    "_target_": "tpen.callback.DataIntegrity",
                    "every_n_steps": 5,
                    "start_step": 2,
                    "seed": 9,
                }
            ]
        )
    ) is None


def test_checkpoint_cadence_is_resume_sensitive() -> None:
    result = check_profile(
        _config(
            callbacks=[
                {"_target_": "tpen.callback.Checkpoint", "max_calls": 1}
            ]
        ),
        NATIVE_RECOVERY_PROFILE,
    )

    assert result.verdict == "refused"
    assert result.refusal is not None
    assert result.refusal["code"] == CALLBACK_CADENCE_REFUSAL_CODE
    assert "max_calls" in result.refusal["reason"]
    with pytest.raises(UnsupportedMutableStateError, match="max_calls"):
        validate_native_recovery_profile(
            _config(
                callbacks=[
                    {"_target_": "tpen.callback.Checkpoint", "max_calls": 1}
                ]
            )
        )


def test_checkpoint_nested_schedule_cadence_is_resume_sensitive() -> None:
    config = _config(
        callbacks=[
            {
                "_target_": "tpen.callback.Checkpoint",
                "schedule": {
                    "_target_": "tpen.callback.cadence.StepCadenceGate",
                    "cadence": {
                        "_target_": "tpen.callback.cadence.StepCadence",
                        "max_calls": 1,
                    },
                },
            }
        ]
    )

    result = check_profile(config, NATIVE_RECOVERY_PROFILE)

    assert result.verdict == "refused"
    assert result.refusal is not None
    assert result.refusal["code"] == CALLBACK_CADENCE_REFUSAL_CODE
    assert "schedule" in result.refusal["reason"]
    with pytest.raises(UnsupportedMutableStateError, match="schedule"):
        validate_native_recovery_profile(config)


def test_checkpoint_plain_schedule_is_admitted_by_both_validators() -> None:
    config = _config(
        callbacks=[
            {
                "_target_": "tpen.callback.Checkpoint",
                "schedule": {
                    "_target_": "tpen.checkpoint.EveryNUpdates",
                    "every_n": 5,
                },
            }
        ]
    )

    result = check_profile(config, NATIVE_RECOVERY_PROFILE)

    assert result.verdict == "supported"
    assert result.refusal is None
    validate_native_recovery_profile(config)


@pytest.mark.parametrize("target", sorted(NEW_CALLBACK_TARGETS))
def test_dropping_each_new_target_refuses_the_h_v2_callback_set(
    target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each newly admitted target is independently covered by the gate."""

    config = OmegaConf.to_container(OmegaConf.load(H_V2_TRAIN_CONFIG), resolve=False)
    callbacks = config["callbacks"]
    if target not in {entry["_target_"] for entry in callbacks}:
        callbacks.append({"_target_": target})
    mutant = Profile(
        name=NATIVE_PROFILE.name,
        state_owners=NATIVE_PROFILE.state_owners,
        callbacks=NATIVE_PROFILE.callbacks - {target},
        loggers=NATIVE_PROFILE.loggers,
        callback_state_supported=NATIVE_PROFILE.callback_state_supported,
    )
    monkeypatch.setattr(profiles, "PROFILES", {NATIVE_RECOVERY_PROFILE: mutant})

    result = check_profile(config, NATIVE_RECOVERY_PROFILE)

    assert result.verdict == "refused"
    assert result.refusal["code"] == "unsupported-callback"
    assert target in result.refusal["reason"]


def test_electron_nucleus_distance_callback_is_profile_admitted() -> None:
    config = _config(
        callbacks=[
            {"_target_": "tpen.callback.ElectronNucleusDistanceQuantiles", "quantiles": [0.5]},
        ]
    )
    result = check_profile(OmegaConf.create(config), NATIVE_RECOVERY_PROFILE)
    assert result.verdict == "supported"
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


assert REFUSAL_CODES == {
    "unknown-profile",
    "unsupported-callback",
    "unsupported-callback-cadence",
    "unsupported-logger",
    "malformed-config",
}
