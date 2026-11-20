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
_ROWS: dict[str, dict[str, tuple[object, object]]] | None = None


def _resolved_rows(tmp_path: Path) -> dict[str, dict[str, tuple[object, object]]]:
    """Return two producer-resolved O1 Adam rows for each sealed context."""

    global _ROWS
    if _ROWS is not None:
        return _ROWS
    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path / "o1",
        availability=optimizer_grid.roster_availability(),
        stages=("O1",),
    )
    rows: dict[str, dict[str, tuple[object, object]]] = {}
    for context in ("O-raw", "O-composite"):
        context_cells = [
            candidate
            for candidate in cells
            if candidate.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
            and candidate.manifest["scientific_identity"]["context"]["name"] == context
        ]
        assert len(context_cells) >= 2
        rows[context] = {
            label: (cell, train_config.resolve_train_config(cell))
            for label, cell in zip(("A", "B"), context_cells[:2], strict=True)
        }
        assert rows[context]["A"][1]["optimizer"] != rows[context]["B"][1]["optimizer"]
    _ROWS = rows
    return rows


def _plain(config: object) -> dict[str, object]:
    """Copy a resolved config without retaining OmegaConf mutation state."""

    return copy.deepcopy(OmegaConf.to_container(config, resolve=False))


@pytest.mark.parametrize("context", ["O-raw", "O-composite"])
@pytest.mark.parametrize("row", ["A", "B"])
def test_qualified_o1_rows_have_deterministic_sealed_identity(
    tmp_path: Path, context: str, row: str
) -> None:
    cell, config = _resolved_rows(tmp_path)[context][row]

    first = inventory.validate_hi_continuation_admission(cell, config)
    second = inventory.validate_hi_continuation_admission(cell, config)

    assert first.identity == second.identity
    assert first.identity.stage == "O1"
    assert first.identity.context == context
    assert first.identity.content_hash == cell.content_hash
    assert first.identity.constructor_hash


@pytest.mark.parametrize("context", ["O-raw", "O-composite"])
@pytest.mark.parametrize("source_row,observed_row", [("A", "B"), ("B", "A")])
def test_cross_row_admission_refuses_with_typed_row_identity(
    tmp_path: Path, context: str, source_row: str, observed_row: str
) -> None:
    cell, _ = _resolved_rows(tmp_path)[context][source_row]
    observed = _resolved_rows(tmp_path)[context][observed_row][1]

    with pytest.raises(
        inventory.HIContinuationRowIdentityRefusal,
        match=(
            "hi-continuation-admission-refused: "
            "resolved-row context identity does not match the sealed O1 row"
        ),
    ):
        inventory.validate_hi_continuation_admission(cell, observed)


def test_same_target_runner_optimizer_value_refuses(tmp_path: Path) -> None:
    cell, config = _resolved_rows(tmp_path)["O-raw"]["A"]
    mutated = _plain(config)
    mutated["runner"]["optimizer"]["lr"] = 0.015

    with pytest.raises(
        inventory.HIContinuationRowIdentityRefusal,
        match="resolved-row context identity does not match the sealed O1 row",
    ):
        inventory.validate_hi_continuation_admission(cell, mutated)


def test_noncanonical_runner_alias_refuses(tmp_path: Path) -> None:
    cell, config = _resolved_rows(tmp_path)["O-raw"]["A"]
    mutated = _plain(config)
    mutated["runner"]["model"]["_target_"] = "tpen.nn.CompositeMixing"

    with pytest.raises(
        inventory.HIContinuationRowIdentityRefusal,
        match="resolved-row context identity does not match the sealed O1 row",
    ):
        inventory.validate_hi_continuation_admission(cell, mutated)


def test_deleted_registry_classification_refuses(tmp_path: Path) -> None:
    cell, config = _resolved_rows(tmp_path)["O-raw"]["A"]
    key = ("model", "tpen.nn.TPENWaveFunction")
    disposition = inventory._REGISTRY.pop(key)
    try:
        with pytest.raises(
            inventory.HIContinuationAdmissionRefusal,
            match="sealed constructor inventory refused",
        ):
            inventory.validate_hi_continuation_admission(cell, config)
    finally:
        # Restore the manually removed key directly; monkeypatch would undo
        # this restoration during teardown and poison later admission cases.
        inventory._REGISTRY[key] = disposition


def test_o_composite_injection_under_o_raw_identity_refuses(tmp_path: Path) -> None:
    cell, config = _resolved_rows(tmp_path)["O-raw"]["A"]
    mutated = _plain(config)
    mutated["model"]["layers"][0] = context_mapping.context_fragment(
        "hybrid/coordinate-neighbor",
        activation="channel-MLP,channel-MLP",
        seed=0,
    )

    with pytest.raises(
        inventory.HIContinuationRowIdentityRefusal,
        match="resolved-row context identity",
    ):
        inventory.validate_hi_continuation_admission(cell, mutated)


def test_raw_basis_injection_under_o_raw_identity_refuses(tmp_path: Path) -> None:
    cell, config = _resolved_rows(tmp_path)["O-raw"]["A"]
    mutated = _plain(config)
    mutated["model"]["basis"] = {
        "_target_": "tpen.nn.basis.BoundedDistanceBasis",
    }

    with pytest.raises(
        inventory.HIContinuationRowIdentityRefusal,
        match="resolved-row context identity",
    ):
        inventory.validate_hi_continuation_admission(cell, mutated)
    return None
