"""Sealed, study-local admission evidence for future HI continuation."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest
from omegaconf import OmegaConf


inventory = importlib.import_module(
    "experiments.atomistic.he-importance.mutable_state_inventory"
)
optimizer_grid = importlib.import_module(
    "experiments.atomistic.he-importance.optimizer_grid"
)
stage_coordinate = importlib.import_module(
    "experiments.atomistic.he-importance.stage_coordinate"
)
train_config = importlib.import_module(
    "experiments.atomistic.he-importance.train_config"
)
context_mapping = importlib.import_module(
    "experiments.atomistic.he-importance.context_mapping"
)


def _resolved_rows(tmp_path: Path) -> dict[str, tuple[object, object]]:
    """Return one resolved O1 Adam row for each sealed O context."""

    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path / "o1",
        availability=optimizer_grid.roster_availability(),
        stages=("O1",),
    )
    rows = {}
    for context in ("O-raw", "O-composite"):
        cell = next(
            candidate
            for candidate in cells
            if candidate.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
            and candidate.manifest["scientific_identity"]["context"]["name"] == context
        )
        rows[context] = (cell, train_config.resolve_train_config(cell))
    return rows


def _plain(config: object) -> dict[str, object]:
    """Copy a resolved config without retaining OmegaConf mutation state."""

    return copy.deepcopy(OmegaConf.to_container(config, resolve=False))


@pytest.mark.parametrize("context", ["O-raw", "O-composite"])
def test_qualified_o1_rows_have_deterministic_sealed_identity(
    tmp_path: Path, context: str
) -> None:
    cell, config = _resolved_rows(tmp_path)[context]

    first = inventory.validate_hi_continuation_admission(cell, config)
    second = inventory.validate_hi_continuation_admission(cell, config)

    assert first.identity == second.identity
    assert first.identity.stage == "O1"
    assert first.identity.context == context
    assert first.identity.content_hash == cell.content_hash
    assert first.identity.constructor_hash
    assert first.profile == "tpen.native-recovery/v1"


def test_unknown_constructor_refuses_before_identity(tmp_path: Path) -> None:
    cell, config = _resolved_rows(tmp_path)["O-raw"]
    mutated = _plain(config)
    mutated["model"]["injected"] = {"_target_": "tests.NewHIConstructor"}

    with pytest.raises(
        inventory.HIContinuationAdmissionRefusal,
        match="sealed constructor inventory refused",
    ):
        inventory.validate_hi_continuation_admission(cell, mutated)


def test_noncanonical_runner_alias_refuses(tmp_path: Path) -> None:
    cell, config = _resolved_rows(tmp_path)["O-raw"]
    mutated = _plain(config)
    mutated["runner"]["model"]["_target_"] = "tpen.nn.CompositeMixing"

    with pytest.raises(
        inventory.HIContinuationAdmissionRefusal,
        match="sealed constructor inventory refused",
    ):
        inventory.validate_hi_continuation_admission(cell, mutated)


def test_deleted_registry_classification_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell, config = _resolved_rows(tmp_path)["O-raw"]
    key = ("model", "tpen.nn.TPENWaveFunction")
    disposition = inventory._REGISTRY.pop(key)
    try:
        with pytest.raises(
            inventory.HIContinuationAdmissionRefusal,
            match="sealed constructor inventory refused",
        ):
            inventory.validate_hi_continuation_admission(cell, config)
    finally:
        monkeypatch.setitem(inventory._REGISTRY, key, disposition)


def test_o_composite_injection_under_o_raw_identity_refuses(tmp_path: Path) -> None:
    cell, config = _resolved_rows(tmp_path)["O-raw"]
    mutated = _plain(config)
    mutated["model"]["layers"][0] = context_mapping.context_fragment(
        "hybrid/coordinate-neighbor",
        activation="channel-MLP,channel-MLP",
        seed=0,
    )

    with pytest.raises(
        inventory.HIContinuationAdmissionRefusal,
        match="resolved-row context identity",
    ):
        inventory.validate_hi_continuation_admission(cell, mutated)


def test_raw_basis_injection_under_o_raw_identity_refuses(tmp_path: Path) -> None:
    cell, config = _resolved_rows(tmp_path)["O-raw"]
    mutated = _plain(config)
    mutated["model"]["basis"] = {
        "_target_": "tpen.nn.basis.BoundedDistanceBasis",
    }

    with pytest.raises(
        inventory.HIContinuationAdmissionRefusal,
        match="resolved-row context identity",
    ):
        inventory.validate_hi_continuation_admission(cell, mutated)


def test_unsupported_callback_shape_refuses_native_profile(tmp_path: Path) -> None:
    cell, config = _resolved_rows(tmp_path)["O-composite"]
    mutated = _plain(config)
    runtime = next(
        callback
        for callback in mutated["callbacks"]
        if callback["_target_"] == "tpen.callback.RuntimeEquivariance"
    )
    runtime["probability"] = 0.5

    with pytest.raises(
        inventory.HIContinuationAdmissionRefusal,
        match="native profile refused",
    ):
        inventory.validate_hi_continuation_admission(cell, mutated)
