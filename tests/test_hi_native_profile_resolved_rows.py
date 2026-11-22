"""Core-owned native profile checks for producer-resolved HI rows."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.checkpoint.profiles import NATIVE_RECOVERY_PROFILE, check_profile
from tpen.execution.native_profile import (
    UnsupportedMutableStateError,
    validate_native_recovery_profile,
)


def _resolved_o1_rows(tmp_path: Path) -> dict[str, object]:
    """Build one producer-resolved Adam row for each O1 context."""

    optimizer_grid = importlib.import_module(
        "experiments.atomistic.he-importance.optimizer_grid"
    )
    stage_coordinate = importlib.import_module(
        "experiments.atomistic.he-importance.stage_coordinate"
    )
    train_config = importlib.import_module(
        "experiments.atomistic.he-importance.train_config"
    )
    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path / "o1",
        availability=optimizer_grid.roster_availability(),
        stages=("O1",),
    )
    rows: dict[str, object] = {}
    for context in ("O-raw", "O-composite"):
        cell = next(
            candidate
            for candidate in cells
            if candidate.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
            and candidate.manifest["scientific_identity"]["context"]["name"] == context
        )
        rows[context] = train_config.resolve_train_config(cell)
    return rows


@pytest.mark.parametrize("context", ["O-raw", "O-composite"])
def test_native_profile_admits_producer_resolved_o1_rows(
    tmp_path: Path, context: str
) -> None:
    """Both producer-resolved O1 contexts pass both core profile checks."""

    config = _resolved_o1_rows(tmp_path)[context]
    result = check_profile(config, NATIVE_RECOVERY_PROFILE)

    assert result.verdict == "supported"
    assert result.refusal is None
    validate_native_recovery_profile(config)


def test_native_profile_core_check_refuses_unsupported_callback_shape(
    tmp_path: Path,
) -> None:
    """The core profile, rather than study code, owns callback-shape refusal."""

    config = _resolved_o1_rows(tmp_path)["O-composite"]
    mutated = copy.deepcopy(OmegaConf.to_container(config, resolve=False))
    runtime = next(
        callback
        for callback in mutated["callbacks"]
        if callback["_target_"] == "tpen.callback.RuntimeEquivariance"
    )
    runtime["probability"] = 0.5

    result = check_profile(mutated, NATIVE_RECOVERY_PROFILE)
    assert result.verdict == "refused"
    assert result.refusal is not None
    assert result.refusal["code"] == "unsupported-callback-cadence"
    with pytest.raises(UnsupportedMutableStateError):
        validate_native_recovery_profile(mutated)
