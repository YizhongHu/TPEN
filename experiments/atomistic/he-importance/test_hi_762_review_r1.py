"""Follow-up coverage for the closed training-horizon boundary (PR #762)."""

from __future__ import annotations

from dataclasses import replace
import importlib
from typing import Any

import pytest


stage_coordinate = importlib.import_module(
    "experiments.atomistic.he-importance.stage_coordinate"
)
optimizer_grid = importlib.import_module(
    "experiments.atomistic.he-importance.optimizer_grid"
)
train_config = importlib.import_module(
    "experiments.atomistic.he-importance.train_config"
)
run_stage_q = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")


class _Absent:
    pass


ABSENT = _Absent()
PAYLOAD_KEYS_ERROR = (
    "payload keys mismatch: expected=['configuration', 'updates'], "
    "actual=['configuration']"
)


@pytest.fixture(scope="module")
def real_o1_cell(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Use one real available Adam row from the committed materializer."""

    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path_factory.mktemp("materialized-o1"),
        availability=optimizer_grid.roster_availability(),
        stages=("O1",),
    )
    return next(
        cell
        for cell in cells
        if cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.availability["status"] == "available"
    )


def _rebuilt(cell: Any, updates: object) -> Any:
    """Change only payload.updates, then rebind the frozen cell identity."""

    manifest = stage_coordinate._thaw(cell.manifest)
    if updates is ABSENT:
        del manifest["payload"]["updates"]
    else:
        manifest["payload"]["updates"] = updates
    digest = stage_coordinate.content_hash(manifest)
    return replace(
        cell,
        manifest=stage_coordinate._freeze(manifest),
        content_hash=digest,
        output_path=cell.output_path.parent / digest,
    )


def _rebuilt_f(cell: Any, updates: object) -> Any:
    """Rebind a real cell as F while preserving its materialized structure."""

    manifest = stage_coordinate._thaw(cell.manifest)
    manifest["stage"] = "F"
    manifest["seed_identity"]["stage"] = "F"
    manifest["seed_identity"]["label"] = 860_001
    if updates is ABSENT:
        del manifest["payload"]["updates"]
    else:
        manifest["payload"]["updates"] = updates
    digest = stage_coordinate.content_hash(manifest)
    return replace(
        cell,
        manifest=stage_coordinate._freeze(manifest),
        content_hash=digest,
        output_path=cell.output_path.parent.parent / "F" / digest,
    )


def _assert_exact(error_type: type[BaseException], message: str, call: Any) -> None:
    """Assert both the public exception class and its complete message."""

    with pytest.raises(error_type) as raised:
        call()
    assert type(raised.value) is error_type
    assert str(raised.value) == message


def test_real_rebuilt_row_rejects_absent_updates(real_o1_cell: Any) -> None:
    rebuilt = _rebuilt(real_o1_cell, ABSENT)
    _assert_exact(stage_coordinate.ManifestSchemaError, PAYLOAD_KEYS_ERROR, lambda: stage_coordinate.validate_materialized_manifest(rebuilt.manifest))


@pytest.mark.parametrize("updates", [50_000.0, ABSENT], ids=["float", "absent"])
def test_resolver_refuses_rebuilt_float_or_absent_updates(
    real_o1_cell: Any, updates: object
) -> None:
    rebuilt = _rebuilt(real_o1_cell, updates)
    expected = "updates must be int" if updates is not ABSENT else PAYLOAD_KEYS_ERROR
    _assert_exact(
        stage_coordinate.ManifestSchemaError,
        expected,
        lambda: train_config.resolve_train_config(rebuilt),
    )


def test_f_exemption_keeps_structural_validation_separate_from_fixed_horizon(
    real_o1_cell: Any,
) -> None:
    rebuilt = _rebuilt_f(real_o1_cell, 123)
    stage_coordinate.validate_materialized_manifest(rebuilt.manifest)
    _assert_exact(
        train_config.TrainConfigResolutionError,
        "stage 'F' has no fixed training horizon",
        lambda: train_config.resolve_train_config(rebuilt),
    )
    invalid = _rebuilt_f(real_o1_cell, 123.0)
    _assert_exact(
        stage_coordinate.ManifestSchemaError,
        "updates must be int",
        lambda: stage_coordinate.validate_materialized_manifest(invalid.manifest),
    )


@pytest.mark.parametrize(
    "updates",
    [50_000.0, True, "50000", None, ABSENT, 49_999],
    ids=["float", "bool", "string", "none", "absent", "wrong-int"],
)
def test_stage_q_selection_path_refuses_rebuilt_horizon_rows(
    real_o1_cell: Any, updates: object
) -> None:
    rebuilt = _rebuilt(real_o1_cell, updates)
    identity = rebuilt.manifest["scientific_identity"]
    levels = identity["optimizer"]["levels"]
    seed = rebuilt.manifest["seed_identity"]["label"]
    if updates is ABSENT:
        expected = PAYLOAD_KEYS_ERROR
    elif type(updates) is int:
        expected = "payload updates 49999 disagree with stage 'O1' horizon 50000"
    else:
        expected = "updates must be int"
    _assert_exact(
        stage_coordinate.ManifestSchemaError,
        expected,
        lambda: run_stage_q.select_cell(
            [rebuilt],
            stage="O1",
            context=identity["context"]["name"],
            optimizer=identity["optimizer"]["method"],
            lr=levels["lr"],
            beta2=levels["beta2"],
            seed_label=seed,
        ),
    )
