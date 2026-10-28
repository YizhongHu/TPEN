"""Acceptance arms for the materialized Stage-O1 universe (Task Orchestrator 14d596d1).

The O1 universe is science-defining: a wrong or truncated universe produces a
scan that looks complete and is not. These tests therefore do NOT reconcile the
committed inventory against the optimizer-grid authority it was rendered from
(that would be self-consistency). They compare it against the consolidated
design's section 2.7 level tables, section 2.8 contexts and section 2.9 seed
labels, retyped literally below, and each comparison has an in-test mutant
control proving that the comparison turns red on a missing, extra, or altered
cell.

Clause map: C1 universe equality and its mutants; C2 Stage Q unchanged against
the dev base; C3 unavailable rows observable and distinct from an undeclared
stage.
"""

from __future__ import annotations

import copy
import importlib
from itertools import product
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import pytest


stage_coordinate = importlib.import_module("experiments.atomistic.he-importance.stage_coordinate")
optimizer_grid = importlib.import_module("experiments.atomistic.he-importance.optimizer_grid")
train_config = importlib.import_module("experiments.atomistic.he-importance.train_config")

_HERE = Path(__file__).resolve().parent
_INVENTORY = _HERE / "intended_configurations.json"

# dev commit immediately before O1 was added; its inventory is Stage Q only.
_PRE_O1_BASE = "3425ecf7d8c0812c2075f4b8c5aaf2adf8cfba67"

# ---------------------------------------------------------------------------
# Section 2.7 / 2.3 literal tables, retyped from the consolidated design and
# deliberately NOT read from optimizer_grid_authority.json.
# Each family maps to (varied levels, fixed levels).
# ---------------------------------------------------------------------------
_DESIGN_FAMILIES: dict[str, tuple[dict[str, list[Any]], dict[str, Any]]] = {
    "adam": (
        {"lr": [0.0005, 0.0015, 0.005, 0.015], "beta2": [0.99, 0.999]},
        {
            "beta1": 0.9,
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "parameter_groups": 1,
            "schedule": "constant",
        },
    ),
    "sr": (
        {
            "eta": [0.003, 0.01, 0.03, 0.1],
            "relative_damping": [1e-4, 1e-3, 1e-2, 0.1],
            "squared_update_norm_bound": [1e-4, 1e-3],
        },
        {},
    ),
    "kfac": (
        {
            "eta": [0.01, 0.03, 0.05, 0.1],
            "relative_damping": [1e-4, 1e-3, 1e-2, 0.1],
            "factor_ema": [0.95, 0.99],
        },
        {
            "curvature_factor_update_interval": 1,
            "inverse_refresh_interval": 10,
            "separate_momentum": False,
            "predicted_kl_cap": 1e-3,
        },
    ),
    "spring": (
        {
            "eta": [0.002, 0.006, 0.02, 0.06],
            "relative_damping": [1e-4, 1e-3],
            "history_mu": [0.98, 0.99],
            "squared_update_norm_bound": [1e-4, 1e-3],
        },
        {"initial_history": 0.0},
    ),
    "linear_method": (
        {
            "derivative_block_shift": [1e-4, 1e-3, 1e-2, 0.1],
            "overlap_metric_trust_radius": [0.01, 0.03, 0.1, 0.3],
        },
        {},
    ),
}

# Section 2.8 optimizer contexts.
_DESIGN_CONTEXTS: dict[str, dict[str, str]] = {
    "O-raw": {"input_features": "B00", "feature_update": "residual"},
    "O-composite": {
        "producer_policy": "hybrid/coordinate-neighbor",
        "aggregation_activation": "channel-MLP at both activation sites",
        "input_features": "B11",
        "feature_update": "residual",
    },
}

# Section 2.8 O1 horizon and section 2.9 O1 seed labels.
_O1_UPDATES = 50_000
_O1_SEEDS = tuple(range(820_001, 820_009))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _design_o1_rows() -> set[tuple[str, str, str, str, int, int]]:
    """Expand the literal design tables into the expected O1 row keys."""

    rows: set[tuple[str, str, str, str, int, int]] = set()
    for family, (varied, fixed) in _DESIGN_FAMILIES.items():
        names = tuple(varied)
        for combination in product(*(varied[name] for name in names)):
            levels = {**fixed, **dict(zip(names, combination))}
            for context, overrides in _DESIGN_CONTEXTS.items():
                for seed in _O1_SEEDS:
                    rows.add(
                        (
                            family,
                            _canonical(levels),
                            context,
                            _canonical(overrides),
                            seed,
                            _O1_UPDATES,
                        )
                    )
    return rows


def _row_key(cell: Any) -> tuple[str, str, str, str, int, int]:
    manifest = cell.manifest
    identity = manifest["scientific_identity"]
    optimizer = identity["optimizer"]
    context = identity["context"]
    return (
        str(optimizer["method"]),
        _canonical(stage_coordinate._thaw(optimizer["levels"])),
        str(context["name"]),
        _canonical(stage_coordinate._thaw(context["overrides"])),
        int(manifest["seed_identity"]["label"]),
        int(manifest["payload"]["updates"]),
    )


_IDENTITY_KEYS = frozenset({"context", "optimizer", "optimizer_cell"})


def _shape_defects(cell: Any) -> list[str]:
    """Report identity-bearing content outside the row key's coordinates.

    The row key covers method, levels, context, seed and horizon. Anything else
    in the identity, payload or topology would change the content hash without
    changing the key, so it must be absent.
    """

    manifest = stage_coordinate._thaw(cell.manifest)
    defects = []
    if set(manifest["scientific_identity"]) != _IDENTITY_KEYS:
        defects.append(f"identity keys {sorted(manifest['scientific_identity'])}")
    if set(manifest["scientific_identity"]["context"]) != {"name", "overrides"}:
        defects.append(f"context keys {sorted(manifest['scientific_identity']['context'])}")
    if manifest["payload"] != {"updates": _O1_UPDATES, "configuration": {"updates": _O1_UPDATES}}:
        defects.append(f"payload {manifest['payload']!r}")
    if manifest["topology"] != {}:
        defects.append(f"topology {manifest['topology']!r}")
    return defects


def _universe_mismatch(cells: Iterable[Any]) -> tuple[set, set, list, list]:
    """Return (missing, extra, duplicated, shape defects) against the design."""

    cells = tuple(cells)
    keys = [_row_key(cell) for cell in cells]
    expected = _design_o1_rows()
    actual = set(keys)
    duplicated = sorted({key for key in keys if keys.count(key) > 1}) if len(keys) != len(actual) else []
    shape = sorted({defect for cell in cells for defect in _shape_defects(cell)})
    return expected - actual, actual - expected, duplicated, shape


def _availability() -> dict[str, Any]:
    return optimizer_grid.roster_availability()


def _materialize_o1(root: Path) -> tuple[Any, ...]:
    return stage_coordinate.materialize_intended_configurations(
        root, availability=_availability(), stages=("O1",)
    )


def _with_inventory(monkeypatch: pytest.MonkeyPatch, path: Path, entries: list[Any]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(stage_coordinate, "_INTENDED_CONFIGURATIONS_PATH", path)


def _committed_entries() -> list[Any]:
    return json.loads(_INVENTORY.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# C1: the O1 universe is exactly the design grid
# ---------------------------------------------------------------------------


def test_design_tables_themselves_expand_to_the_declared_design_totals() -> None:
    """Guard the literal tables: 120 cells, 1920 O1 lineages (section 2.7/2.8)."""

    per_family = {
        family: len(list(product(*varied.values())))
        for family, (varied, _) in _DESIGN_FAMILIES.items()
    }
    assert per_family == {"adam": 8, "sr": 32, "kfac": 32, "spring": 32, "linear_method": 16}
    assert len(_design_o1_rows()) == 1_920


def test_c1_materialized_o1_universe_equals_the_design_grid(tmp_path: Path) -> None:
    cells = _materialize_o1(tmp_path)
    missing, extra, duplicated, shape = _universe_mismatch(cells)
    assert not missing, f"{len(missing)} design O1 rows were not materialized"
    assert not extra, f"{len(extra)} materialized O1 rows are not in the design"
    assert not duplicated
    assert not shape, shape
    assert len(cells) == 1_920
    assert {cell.manifest["stage"] for cell in cells} == {"O1"}
    assert len({cell.content_hash for cell in cells}) == 1_920
    assert len({cell.output_path for cell in cells}) == 1_920
    assert {cell.output_path.parent.name for cell in cells} == {"O1"}
    # Each O1 seed stream bundle is the O1 namespace's, never another stage's.
    namespaces = stage_coordinate.seed_namespaces("O1")
    for cell in cells:
        label = cell.manifest["seed_identity"]["label"]
        assert dict(cell.seed_streams) == namespaces[label]


def test_c1_o1_horizon_comes_from_the_stage_authority() -> None:
    assert stage_coordinate.stage_definition("O1").updates == _O1_UPDATES
    assert stage_coordinate.seed_labels("O1") == _O1_SEEDS


def _drop_last_o1_entry(entries: list[Any]) -> None:
    index = max(i for i, entry in enumerate(entries) if entry["stage"] == "O1")
    del entries[index]


def _drop_one_o1_context(entries: list[Any]) -> None:
    entry = next(entry for entry in entries if entry["stage"] == "O1")
    entry["configurations"] = [
        configuration
        for configuration in entry["configurations"]
        if configuration["scientific_identity"]["context"]["name"] != "O-composite"
    ]


def _alter_one_o1_level(entries: list[Any]) -> None:
    entry = next(
        entry
        for entry in entries
        if entry["stage"] == "O1" and entry["optimizer"]["method"] == "kfac"
    )
    entry["optimizer"]["levels"]["eta"] = 0.07
    for configuration in entry["configurations"]:
        configuration["scientific_identity"]["optimizer"]["levels"]["eta"] = 0.07


def _alter_one_o1_horizon(entries: list[Any]) -> None:
    entry = next(entry for entry in entries if entry["stage"] == "O1")
    entry["configurations"][0]["payload"]["updates"] = 2_000


def _drop_one_o1_family(entries: list[Any]) -> None:
    entries[:] = [
        entry
        for entry in entries
        if not (entry["stage"] == "O1" and entry["optimizer"]["method"] == "linear_method")
    ]


def _append_duplicate_o1_entry(entries: list[Any]) -> None:
    entries.append(copy.deepcopy(next(entry for entry in entries if entry["stage"] == "O1")))


def _append_undesigned_adam_cell(entries: list[Any]) -> None:
    entry = copy.deepcopy(
        next(e for e in entries if e["stage"] == "O1" and e["optimizer"]["method"] == "adam")
    )
    entry["optimizer"]["levels"]["lr"] = 0.05
    for configuration in entry["configurations"]:
        configuration["scientific_identity"]["optimizer"]["levels"]["lr"] = 0.05
    entries.append(entry)


def _append_third_context(entries: list[Any]) -> None:
    entry = next(e for e in entries if e["stage"] == "O1")
    extra = copy.deepcopy(entry["configurations"][0])
    extra["scientific_identity"]["context"] = {"name": "O-third", "overrides": {"input_features": "B10"}}
    entry["configurations"].append(extra)


def _add_identity_key_to_every_o1_row(entries: list[Any]) -> None:
    for entry in entries:
        if entry["stage"] == "O1":
            for configuration in entry["configurations"]:
                configuration["scientific_identity"]["architecture"] = "control"


def _add_payload_key_to_every_o1_row(entries: list[Any]) -> None:
    for entry in entries:
        if entry["stage"] == "O1":
            for configuration in entry["configurations"]:
                configuration["payload"]["n_walkers"] = 2_048


def _o1_entries_relabelled_as_q(entries: list[Any]) -> None:
    for entry in entries:
        if entry["stage"] == "O1" and entry["optimizer"]["method"] == "spring":
            entry["stage"] = "Q"


# Each mutant names the mismatch component(s) it must light up, so a mutant
# cannot pass merely because some unrelated component happened to be non-empty.
@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_drop_last_o1_entry, {"missing"}),
        (_drop_one_o1_context, {"missing"}),
        (_alter_one_o1_level, {"missing", "extra"}),
        (_drop_one_o1_family, {"missing"}),
        (_o1_entries_relabelled_as_q, {"missing"}),
        (_append_duplicate_o1_entry, {"duplicated"}),
        (_append_undesigned_adam_cell, {"extra"}),
        (_append_third_context, {"extra"}),
        (_add_identity_key_to_every_o1_row, {"shape"}),
        (_add_payload_key_to_every_o1_row, {"shape"}),
    ],
    ids=lambda value: value.__name__ if callable(value) else "-".join(sorted(value)),
)
def test_c1_control_universe_comparison_fails_on_each_mutant_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Any, expected: set[str]
) -> None:
    """The C1 comparison is red for a missing, extra, duplicated or reshaped O1 cell."""

    entries = _committed_entries()
    mutate(entries)
    _with_inventory(monkeypatch, tmp_path / "mutant.json", entries)
    missing, extra, duplicated, shape = _universe_mismatch(_materialize_o1(tmp_path / "out"))
    lit = {
        name
        for name, value in (
            ("missing", missing), ("extra", extra), ("duplicated", duplicated), ("shape", shape)
        )
        if value
    }
    assert lit == expected


def test_c1_control_off_horizon_o1_row_is_refused_at_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identity horizon that would not be the trained horizon never materializes."""

    entries = _committed_entries()
    _alter_one_o1_horizon(entries)
    _with_inventory(monkeypatch, tmp_path / "mutant.json", entries)
    with pytest.raises(stage_coordinate.MaterializationError, match="disagree with stage 'O1' horizon 50000"):
        _materialize_o1(tmp_path / "out")


def test_c1_control_universe_comparison_fails_on_a_truncated_seed_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = stage_coordinate.seed_labels

    def truncated(stage: str) -> tuple[int, ...]:
        labels = original(stage)
        return labels[:-1] if stage == "O1" else labels

    monkeypatch.setattr(stage_coordinate, "seed_labels", truncated)
    missing, extra, _, _ = _universe_mismatch(_materialize_o1(tmp_path))
    assert missing and not extra
    assert {key[4] for key in missing} == {_O1_SEEDS[-1]}


def test_c1_control_unmutated_copy_of_the_inventory_stays_green(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control for the mutants: the copy mechanism alone is not a mutation."""

    _with_inventory(monkeypatch, tmp_path / "copy.json", _committed_entries())
    assert _universe_mismatch(_materialize_o1(tmp_path / "out")) == (set(), set(), [], [])


# ---------------------------------------------------------------------------
# C2: Stage Q is unchanged against the pre-O1 dev base
# ---------------------------------------------------------------------------


def _base_inventory_text() -> str:
    relative = _INVENTORY.relative_to(
        Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=_HERE, text=True
            ).strip()
        ).resolve()
    )
    return subprocess.check_output(
        ["git", "show", f"{_PRE_O1_BASE}:{relative.as_posix()}"], cwd=_HERE, text=True
    )


def _q_identities(
    monkeypatch: pytest.MonkeyPatch, inventory: Path, entries: list[Any], root: Path
) -> set[tuple[str, Path]]:
    _with_inventory(monkeypatch, inventory, entries)
    cells = stage_coordinate.materialize_intended_configurations(
        root, availability=_availability(), stages=("Q",)
    )
    return {(cell.content_hash, cell.output_path) for cell in cells}


def test_c2_stage_q_entries_and_identities_equal_the_pre_o1_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_entries = json.loads(_base_inventory_text())
    assert {entry["stage"] for entry in base_entries} == {"Q"}
    current_q = [entry for entry in _committed_entries() if entry["stage"] == "Q"]
    assert current_q == base_entries

    root = tmp_path / "root"
    base_ids = _q_identities(monkeypatch, tmp_path / "base.json", base_entries, root)
    current_ids = _q_identities(monkeypatch, tmp_path / "current.json", _committed_entries(), root)
    assert len(base_ids) == 480
    assert current_ids == base_ids


def test_c2_control_q_identity_comparison_fails_on_a_perturbed_q_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_entries = json.loads(_base_inventory_text())
    perturbed = copy.deepcopy(_committed_entries())
    q_entry = next(entry for entry in perturbed if entry["stage"] == "Q")
    # Perturb a level, not the horizon: an off-horizon row is refused at
    # materialization before any identity comparison could run.
    q_entry["optimizer"]["levels"]["lr"] = 0.0006
    for configuration in q_entry["configurations"]:
        configuration["scientific_identity"]["optimizer"]["levels"]["lr"] = 0.0006

    root = tmp_path / "root"
    base_ids = _q_identities(monkeypatch, tmp_path / "base.json", base_entries, root)
    perturbed_ids = _q_identities(monkeypatch, tmp_path / "perturbed.json", perturbed, root)
    assert perturbed_ids != base_ids


# ---------------------------------------------------------------------------
# C3: unavailable rows are observable; an undeclared stage is a distinct refusal
# ---------------------------------------------------------------------------


class _Entry:
    """A roster entry with the shape of ``tpen.hi_schema.MethodAvailability``."""

    def __init__(self, method: str, admitted: bool, target: str | None, requires: str) -> None:
        self.method = method
        self.admitted = admitted
        self.target = target
        self.requires = requires


# Known sentinel reasons, including the admitted-without-target case that must
# never be materialized or counted as runnable.
_INJECTED_ROSTER = (
    _Entry("adam", True, "torch.optim.Adam", "admitted"),
    _Entry("sr", False, None, "SR-SENTINEL"),
    _Entry("kfac", True, None, "KFAC-SENTINEL"),
    _Entry("spring", False, None, "SPRING-SENTINEL"),
    _Entry("linear_method", False, None, "LM-SENTINEL"),
)
_INJECTED_EXPECTED = {
    "adam": {"method": "adam", "status": "available"},
    "sr": {"method": "sr", "status": "unavailable", "unavailable_reason": "SR-SENTINEL"},
    "kfac": {
        "method": "kfac",
        "status": "unavailable",
        "unavailable_reason": "admitted but has no constructor target; KFAC-SENTINEL",
    },
    "spring": {"method": "spring", "status": "unavailable", "unavailable_reason": "SPRING-SENTINEL"},
    "linear_method": {
        "method": "linear_method",
        "status": "unavailable",
        "unavailable_reason": "LM-SENTINEL",
    },
}


def test_c3_injected_roster_reasons_reach_every_o1_row_verbatim(tmp_path: Path) -> None:
    availability = optimizer_grid.roster_availability(_INJECTED_ROSTER)
    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path, availability=availability, stages=("O1",)
    )
    per_method: dict[str, int] = {}
    for cell in cells:
        optimizer_cell = dict(cell.manifest["scientific_identity"]["optimizer_cell"])
        method = optimizer_cell["method"]
        assert optimizer_cell == _INJECTED_EXPECTED[method]
        per_method[method] = per_method.get(method, 0) + 1
    # Rows per method = design cells x 2 contexts x 8 seeds; nothing dropped.
    assert per_method == {"adam": 128, "sr": 512, "kfac": 512, "spring": 512, "linear_method": 256}

    report = optimizer_grid.stage_report("O1", roster=_INJECTED_ROSTER)
    assert report["lineages"] == {
        "contexts": 2,
        "seeds_per_point": 8,
        "declared": 1_920,
        "runnable": 128,
        "unavailable": 1_792,
    }
    assert report["families"]["kfac"]["runnable"] == 0
    assert report["families"]["kfac"]["reason"] == _INJECTED_EXPECTED["kfac"]["unavailable_reason"]


def test_c3_live_roster_report_agrees_with_materialized_statuses(tmp_path: Path) -> None:
    """The live report's lineage counts are the materialized status counts."""

    cells = _materialize_o1(tmp_path)
    by_status: dict[str, int] = {}
    for cell in cells:
        status = cell.manifest["scientific_identity"]["optimizer_cell"]["status"]
        by_status[status] = by_status.get(status, 0) + 1
    report = optimizer_grid.stage_report("O1")
    assert report["lineages"]["declared"] == len(cells) == 1_920
    assert report["lineages"]["runnable"] == by_status.get("available", 0)
    assert report["lineages"]["unavailable"] == by_status.get("unavailable", 0)


def test_c3_resolving_an_unavailable_o1_row_is_refused_not_rerouted(tmp_path: Path) -> None:
    cells = _materialize_o1(tmp_path)
    sr_cell = next(
        cell
        for cell in cells
        if cell.manifest["scientific_identity"]["optimizer"]["method"] == "sr"
    )
    with pytest.raises(train_config.UnavailableOptimizerError) as error:
        train_config.resolve_train_config(sr_cell)
    # A refused method on a declared stage must not read as an absent universe.
    assert "no declared intended-configuration universe" not in str(error.value)


@pytest.mark.parametrize("stage_code", ["O2", "A", "B", "R", "F"])
def test_c3_undeclared_stage_is_refused_with_its_named_reason(
    tmp_path: Path, stage_code: str
) -> None:
    reason = optimizer_grid.UNDECLARED_STAGE_REASONS[stage_code]
    assert optimizer_grid.undeclared_stage_reason(stage_code) == reason
    with pytest.raises(stage_coordinate.MaterializationError, match="no declared intended-configuration universe"):
        stage_coordinate.materialize_intended_configurations(
            tmp_path, availability=_availability(), stages=(stage_code,)
        )
    with pytest.raises(optimizer_grid.OptimizerGridError) as error:
        optimizer_grid.stage_report(stage_code)
    assert reason in str(error.value)


def test_c3_declared_and_undeclared_stages_partition_the_stage_authority() -> None:
    codes = {definition.code for definition in stage_coordinate.STAGE_DEFINITIONS}
    declared = set(optimizer_grid.OPTIMIZER_GRID_STAGES)
    undeclared = set(optimizer_grid.UNDECLARED_STAGE_REASONS)
    assert declared == {"Q", "O1"}
    assert not declared & undeclared
    assert declared | undeclared == codes
    with pytest.raises(optimizer_grid.OptimizerGridError, match="unknown HI stage"):
        optimizer_grid.undeclared_stage_reason("Z")


@pytest.mark.parametrize(
    "declared",
    [
        ("Q",),  # O1 would fall into neither set
        ("Q", "O1", "O2"),  # O2 would be in both sets
    ],
    ids=["stage-in-neither-set", "stage-in-both-sets"],
)
def test_c3_control_partition_check_fails_on_a_broken_partition(declared: tuple[str, ...]) -> None:
    with pytest.raises(optimizer_grid.OptimizerGridError, match="partition the stage authority"):
        optimizer_grid._require_stage_partition(declared)


@pytest.mark.parametrize(
    ("stages", "message"),
    [((), "at least one stage"), ("O1", "not a string"), (b"O1", "not a string")],
    ids=["empty", "str", "bytes"],
)
def test_stage_selection_must_be_a_non_empty_iterable_of_codes(
    tmp_path: Path, stages: Any, message: str
) -> None:
    with pytest.raises(stage_coordinate.MaterializationError, match=message):
        stage_coordinate.materialize_intended_configurations(
            tmp_path, availability=_availability(), stages=stages
        )


def test_unknown_stage_selection_is_refused(tmp_path: Path) -> None:
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="unknown HI stage"):
        stage_coordinate.materialize_intended_configurations(
            tmp_path, availability=_availability(), stages=("Z",)
        )
