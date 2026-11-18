"""Boundary evidence for the HI native recovery profile."""

import pytest
import importlib
from tpen.execution.native_profile import (
    UnsupportedMutableStateError,
    validate_native_recovery_profile,
)


def test_native_profile_refuses_hi_train_config_at_expected_entries() -> None:
    """The core profile rejects the known HI callback entries."""

    train_config = importlib.import_module(
        "experiments.atomistic.he-importance.train_config"
    )
    with pytest.raises(UnsupportedMutableStateError, match=r"callbacks\[8\]"):
        validate_native_recovery_profile(train_config._compose_base_config())
    train_config = importlib.import_module(
        "experiments.atomistic.he-importance.train_config"
    )
