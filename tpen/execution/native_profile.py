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

from collections.abc import Mapping
from typing import Any

from tpen.checkpoint.profiles import (
    check_resume_sensitive_callback_cadence,
    SUPPORTED_CALLBACK_TARGETS,
    SUPPORTED_LOGGER_TARGETS,
    SUPPORTED_STATE_OWNERS,
)


class UnsupportedMutableStateError(ValueError):
    """Raised when a configured state owner is outside the native profile."""


# This allowlist defines the train_resume owners; callback/logger validation
# below only covers additional user-configured mutable state.


def _target(entry: Any, owner: str, index: int) -> str:
    """Read a constructor name without resolving or importing it."""

    if not isinstance(entry, Mapping):
        raise UnsupportedMutableStateError(f"{owner}[{index}] must be a mapping")
    value = entry.get("_target_")
    if not isinstance(value, str) or not value:
        raise UnsupportedMutableStateError(f"{owner}[{index}] has no explicit _target_")
    return value


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

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=False)
    except ImportError:  # pragma: no cover - OmegaConf is a project dependency
        pass
    if not isinstance(config, Mapping):
        raise UnsupportedMutableStateError("native profile requires a mapping config")
    for owner, allowed in (
        ("callbacks", SUPPORTED_CALLBACK_TARGETS),
        ("loggers", SUPPORTED_LOGGER_TARGETS),
    ):
        entries = config.get(owner, ()) or ()
        if not isinstance(entries, (list, tuple)):
            raise UnsupportedMutableStateError(f"{owner} must be a sequence")
        for index, entry in enumerate(entries):
            target = _target(entry, owner, index)
            if target not in allowed:
                raise UnsupportedMutableStateError(
                    f"{owner}[{index}] {target!r} is not in the native mutable-state profile"
                )
    cadence_refusal = check_resume_sensitive_callback_cadence(config)
    if cadence_refusal is not None:
        code, reason = cadence_refusal
        raise UnsupportedMutableStateError(f"{code}: {reason}")


__all__ = [
    "SUPPORTED_CALLBACK_TARGETS",
    "SUPPORTED_LOGGER_TARGETS",
    "SUPPORTED_STATE_OWNERS",
    "UnsupportedMutableStateError",
    "validate_native_recovery_profile",
]
