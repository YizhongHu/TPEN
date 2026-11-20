"""Compatibility exports for the core-owned native recovery profile.

Core validates train-resume admission before construction. The profile and
its typed refusal live in checkpoint so core never imports executor adapters.
"""

from tpen.checkpoint.profiles import (
    SUPPORTED_CALLBACK_TARGETS,
    SUPPORTED_LOGGER_TARGETS,
    SUPPORTED_STATE_OWNERS,
    UnsupportedMutableStateError,
    validate_native_recovery_profile,
)

__all__ = [
    "SUPPORTED_CALLBACK_TARGETS",
    "SUPPORTED_LOGGER_TARGETS",
    "SUPPORTED_STATE_OWNERS",
    "UnsupportedMutableStateError",
    "validate_native_recovery_profile",
]
