"""Boundary evidence for the HI native recovery profile."""

import importlib
from tpen.execution.native_profile import (
    validate_native_recovery_profile,
)


def test_native_profile_admits_hi_callbacks_after_cx07() -> None:
    """CX-07 admits the HI callbacks; study policy still refuses continuation."""

    train_config = importlib.import_module(
        "experiments.atomistic.he-importance.train_config"
    )
    validate_native_recovery_profile(train_config._compose_base_config())
