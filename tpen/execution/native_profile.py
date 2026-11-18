"""Supported mutable-state profile for the bounded native qualification.

The profile is intentionally an allowlist.  A callback or logger not named
here may carry private counters, random generators, or cadence state that is
not reconstructed by ``train_resume``.  Refusing it before ``run_from_config``
is safer than implying that a checkpoint restored more state than it owns.
``SUPPORTED_STATE_OWNERS`` is the definitional allowlist of mutable owners
restored by ``train_resume``; validation below only covers additional
configured callback and logger state.
"""

from __future__ import annotations

from typing import Any

from tpen.checkpoint.profiles import (
    NATIVE_PROFILE,
    check_profile_refusal,
    SUPPORTED_CALLBACK_TARGETS,
    SUPPORTED_LOGGER_TARGETS,
    SUPPORTED_STATE_OWNERS,
)


class UnsupportedMutableStateError(ValueError):
    """Raised when a configured state owner is outside the native profile."""


def validate_native_recovery_profile(config: Any) -> None:
    """Validate the allowlisted mutable-state owners in a config.

    Parameters
    ----------
    config : mapping
        Unresolved OmegaConf-compatible configuration.  Only syntax is read;
        no ``_target_`` is imported or instantiated.

    Raises
    ------
    UnsupportedMutableStateError
        If a callback or logger is not explicitly certified by this profile.
    """

    refusal = check_profile_refusal(config, NATIVE_PROFILE)
    if refusal is not None:
        code, reason = refusal
        if code == "malformed-config":
            message = reason
        elif "is not in the declared profile" in reason:
            message = reason.replace(
                "is not in the declared profile",
                "is not in the native mutable-state profile",
            )
        else:
            message = f"{code}: {reason}"
        raise UnsupportedMutableStateError(message)


__all__ = [
    "SUPPORTED_CALLBACK_TARGETS",
    "SUPPORTED_LOGGER_TARGETS",
    "SUPPORTED_STATE_OWNERS",
    "UnsupportedMutableStateError",
    "validate_native_recovery_profile",
]
