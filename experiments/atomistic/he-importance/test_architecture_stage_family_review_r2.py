"""Pre-review round 2 tests for PR #580 (AB-4c architecture-stage family).

Written by the independent reviewer against head
3009bc6be3f99a07d0a76f4b12102db7e2e37c36.  Each test states what it
establishes and whether it is expected GREEN or RED at this head.  RED tests
are regression controls for the round-2 findings; they fail with exactly the
failure named in their docstring.
"""

from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace

from omegaconf import OmegaConf
import pytest


stage_coordinate = importlib.import_module(
    "experiments.atomistic.he-importance.stage_coordinate"
)
optimizer_grid = importlib.import_module(
    "experiments.atomistic.he-importance.optimizer_grid"
)
runner = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")
family = importlib.import_module(
    "experiments.atomistic.he-importance.architecture_stage_family"
)


def _fake_adam(
    params: object,
    lr: float = 0.001,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> None:
    del params, lr, betas, eps, weight_decay


@pytest.fixture(autouse=True)
def _runtime_torch_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same guard as test_run_stage_q: keep the route testable without torch."""

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(optim=SimpleNamespace(Adam=_fake_adam)),
        )


_LEVELS = {
    "lr": 0.005,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.0,
    "parameter_groups": 1,
    "schedule": "constant",
}
_X_SEED = 870_001
_X_CONTEXT = {"axis": "X", "literal_configuration": {"label": "x"}}


def _patch_authority_for_x(monkeypatch: pytest.MonkeyPatch, *, updates: int) -> None:
    """Add stage X to BOTH stage_coordinate module objects (optimizer_grid has a private copy)."""

    definition = stage_coordinate.StageDefinition(
        code="X",
        purpose="review fixture r2",
        seeds_per_point=1,
        maximum_lineages=1,
        updates=updates,
        sampler_overrides=MappingProxyType({}),
    )
    copies = {
        id(stage_coordinate): stage_coordinate,
        id(optimizer_grid.stage_coordinate): optimizer_grid.stage_coordinate,
    }
    for module in copies.values():
        monkeypatch.setattr(
            module, "STAGE_DEFINITIONS", module.STAGE_DEFINITIONS + (definition,)
        )
        monkeypatch.setitem(module.STAGES, "X", definition)
        monkeypatch.setitem(module.SEED_NAMESPACE_STARTS, "X", _X_SEED)


def _x_descriptor(*, selector, packet_policy, updates: int) -> family.ArchitectureStageDescriptor:
    """A contract-complete stage X with an A/B-shaped context and one Adam row.

    The resolved view carries root/runner twins and the authority seed streams
    so the runner's stage-independent truthfulness checks (F3) pass; only the
    pieces under test vary.
    """

    slot = family.OptimizerSlot("adam", "available", dict(_LEVELS))
    package = {"optimizer_slots": [slot.as_dict()]}

    def renderer(stage: str):
        return ({
            "stage": stage,
            "package": package,
            "optimizer": slot.as_dict(),
            "configurations": ({
                "scientific_identity": {
                    "context": {
                        "axis": "X",
                        "literal_configuration": dict(_X_CONTEXT["literal_configuration"]),
                    },
                    "optimizer": {"method": "adam", "levels": dict(_LEVELS)},
                },
                "payload": {"updates": updates},
                "topology": {},
            },),
        },)

    def context_validator(context) -> None:
        if dict(context) != _X_CONTEXT:
            raise family.ArchitectureStageFamilyError("X context is closed")

    streams = stage_coordinate.seed_namespace("X", _X_SEED)
    model = {"initializer": {"seed": streams["model_initialization"]}}
    optimizer = {
        "_target_": "torch.optim.Adam",
        "lr": _LEVELS["lr"],
        "betas": [_LEVELS["beta1"], _LEVELS["beta2"]],
        "eps": _LEVELS["epsilon"],
        "weight_decay": _LEVELS["weight_decay"],
    }
    sampler = {"seed": streams["training_sampler"]}
    resolved = OmegaConf.create(
        {
            "model": model,
            "optimizer": optimizer,
            "sampler": sampler,
            "trainer": {"max_steps": updates},
            "hamiltonian_terms": {},
            "runner": {
                "model": model,
                "optimizer": optimizer,
                "sampler": sampler,
                "trainer": {"max_steps": updates},
                "hamiltonian_terms": {},
            },
        }
    )
    return family.ArchitectureStageDescriptor(
        stage="X",
        family="review-fixture-r2",
        renderer=renderer,
        context_validator=context_validator,
        resolver=lambda source: resolved,
        resolved_validator=lambda cell, config: None,
        slot_policy=lambda row: (slot,),
        seed_namespace=lambda stage: stage_coordinate.seed_labels(stage),
        selector=selector,
        packet_policy=packet_policy,
        launch_adapter=lambda *args, **kwargs: 0,
    )


def _select_by_context(cells, *, stage, context, optimizer, seed_label):
    """A selector that declares only the four coordinates it consumes."""

    matches = [
        cell
        for cell in cells
        if cell.manifest["stage"] == stage
        and cell.manifest["scientific_identity"]["context"] == context
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == optimizer
        and cell.manifest["seed_identity"]["label"] == seed_label
    ]
    if len(matches) != 1:
        raise family.ArchitectureStageFamilyError(f"X selection matched {len(matches)} cells")
    return family.FamilySelection(matches[0], family._cell_descriptor(matches[0]), ())


def _select_by_context_and_literal(
    cells, *, stage, context, optimizer, seed_label, literal_configuration=None
):
    """Same selector, but it declares literal_configuration (and still no lr/beta2)."""

    del literal_configuration
    return _select_by_context(
        cells, stage=stage, context=context, optimizer=optimizer, seed_label=seed_label
    )


_GOOD_POLICY = lambda stage: {"updates": 7, "checkpoint_every_n_updates": 1}  # noqa: E731
_LONG_CADENCE_POLICY = lambda stage: {"updates": 7, "checkpoint_every_n_updates": 1000}  # noqa: E731


# ---------------------------------------------------------------------------
# T11 (RED by design -> F11). run_stage_q.select_cell filters the coordinate
# set by the selector's signature (inspect.signature) and silently drops
# whatever the selector does not declare.  A user who asks for lr=0.01 on a
# stage whose selector omits lr gets the lr=0.005 row with no refusal; the
# same for literal_configuration.  Expected failure at this head: DID NOT
# RAISE StageQSelectionError.
# ---------------------------------------------------------------------------
def test_t11a_a_dropped_lr_coordinate_is_refused_not_substituted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """lr=0.01 on a stage whose selector cannot consume lr must refuse by name."""

    _patch_authority_for_x(monkeypatch, updates=7)
    descriptor = _x_descriptor(
        selector=_select_by_context_and_literal, packet_policy=_GOOD_POLICY, updates=7
    )
    family.validate_descriptor(descriptor)
    monkeypatch.setitem(family.ARCHITECTURE_STAGE_REGISTRY, "X", descriptor)
    cells = family.materialize_family_stage("X", tmp_path)
    with pytest.raises(runner.StageQSelectionError, match="lr"):
        runner.select_cell(
            cells,
            stage="X",
            context=dict(_X_CONTEXT),
            optimizer="adam",
            seed_label=_X_SEED,
            lr=0.01,
        )


def test_t11b_a_dropped_literal_configuration_is_refused_not_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A literal_configuration the selector cannot consume must refuse by name."""

    _patch_authority_for_x(monkeypatch, updates=7)
    descriptor = _x_descriptor(selector=_select_by_context, packet_policy=_GOOD_POLICY, updates=7)
    family.validate_descriptor(descriptor)
    monkeypatch.setitem(family.ARCHITECTURE_STAGE_REGISTRY, "X", descriptor)
    cells = family.materialize_family_stage("X", tmp_path)
    with pytest.raises(runner.StageQSelectionError, match="literal_configuration"):
        runner.select_cell(
            cells,
            stage="X",
            context=dict(_X_CONTEXT),
            optimizer="adam",
            seed_label=_X_SEED,
            literal_configuration={"label": "y"},
        )


# ---------------------------------------------------------------------------
# T12 (RED by design -> F12). A packet policy whose cadence is longer than
# its horizon is (a) accepted at registration and (b) surfaces on the route
# as a raw stage_coordinate.MaterializationError ("at least one checkpoint
# observation is required"), which select_cell's handlers do not catch.
# ---------------------------------------------------------------------------
def test_t12a_a_cadence_longer_than_the_horizon_is_refused_at_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_descriptor must refuse checkpoint_every_n_updates > updates by name.

    Expected failure at this head: DID NOT RAISE (registration succeeds).
    """

    _patch_authority_for_x(monkeypatch, updates=7)
    descriptor = _x_descriptor(
        selector=_select_by_context, packet_policy=_LONG_CADENCE_POLICY, updates=7
    )
    with pytest.raises(family.ArchitectureStageFamilyError, match="cadence|checkpoint"):
        family.register_stage_descriptor(descriptor, registry={})


def test_t12b_a_cadence_longer_than_the_horizon_is_a_named_runner_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_training_packet must raise StageQSelectionError, not a materializer error.

    Cells come from a valid descriptor; the registry entry is then swapped for
    the long-cadence one so the route, not registration, is under test.
    Expected failure at this head: stage_coordinate.MaterializationError.
    """

    _patch_authority_for_x(monkeypatch, updates=7)
    good = _x_descriptor(selector=_select_by_context, packet_policy=_GOOD_POLICY, updates=7)
    monkeypatch.setitem(family.ARCHITECTURE_STAGE_REGISTRY, "X", good)
    cells = family.materialize_family_stage("X", tmp_path)
    monkeypatch.setitem(
        family.ARCHITECTURE_STAGE_REGISTRY, "X", replace(good, packet_policy=_LONG_CADENCE_POLICY)
    )
    with pytest.raises(runner.StageQSelectionError, match="X"):
        runner._training_packet(cells[0])


# ---------------------------------------------------------------------------
# T13 (RED by design -> F13). The A/B resolved validator never checks that
# the resolved model reflects the row's literal configuration.  A resolver
# that returns package 0's real resolved config for every A row passes the
# family guard when package 1 is selected.  Expected failure at this head:
# DID NOT RAISE StageQSelectionError.
# ---------------------------------------------------------------------------
def test_t13_a_resolver_that_ignores_the_literal_configuration_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consumed truth for A/B: the resolved model must reflect the selected literal."""

    packages = family.ab_universe.render_a_packages()
    cells = family.materialize_family_stage("A", tmp_path)

    def cell_for(literal):
        return next(
            cell
            for cell in cells
            if dict(cell.manifest["scientific_identity"]["context"]["literal_configuration"]) == dict(literal)
            and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
            and cell.manifest["seed_identity"]["label"] == 830001
        )

    cell_zero = cell_for(packages[0]["literal_configuration"])
    real_zero = family._resolver(runner._training_packet(cell_zero))
    original = family.ARCHITECTURE_STAGE_REGISTRY["A"]
    monkeypatch.setitem(
        family.ARCHITECTURE_STAGE_REGISTRY, "A", replace(original, resolver=lambda source: real_zero)
    )
    selection = runner.select_cell(
        cells,
        stage="A",
        context="A",
        optimizer="adam",
        seed_label=830001,
        literal_configuration=packages[1]["literal_configuration"],
    )
    assert selection.selected.literal_configuration == packages[1]["literal_configuration"]
    with pytest.raises(runner.StageQSelectionError):
        runner._resolve_selected_config(selection)


# ---------------------------------------------------------------------------
# T14 (GREEN expected -> F14). The ORIGINAL package-less clause-8 fixture
# (the one T4 lost when the writer added `package`) is materializable on the
# generic route now that validation and materialization both use
# row.get("package", row).  This is the control for F1(a) that T4 no longer
# provides.
# ---------------------------------------------------------------------------
def test_t14_a_package_less_row_schema_is_materializable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows without a 'package' key pass validation AND materialization."""

    _patch_authority_for_x(monkeypatch, updates=7)
    slot = family.OptimizerSlot("adam", "available", {"lr": 1.0})

    def renderer(stage: str):
        return ({
            "stage": stage,
            "optimizer": slot.as_dict(),
            "configurations": ({
                "scientific_identity": {
                    "context": {"axis": "X", "literal_configuration": {"label": "x"}},
                    "optimizer": {"method": "adam", "levels": {"lr": 1.0}},
                },
                "payload": {"updates": 7},
                "topology": {},
            },),
        },)

    descriptor = family.ArchitectureStageDescriptor(
        stage="X",
        family="review-fixture-r2",
        renderer=renderer,
        context_validator=lambda context: None,
        resolver=lambda cell: object(),
        resolved_validator=lambda cell, resolved: None,
        slot_policy=lambda row: (slot,),
        seed_namespace=lambda stage: stage_coordinate.seed_labels(stage),
        selector=lambda *args, **kwargs: None,
        packet_policy=lambda stage: {"updates": 7},
        launch_adapter=lambda *args, **kwargs: None,
    )
    registry = dict(family.ARCHITECTURE_STAGE_REGISTRY)
    family.register_stage_descriptor(descriptor, registry=registry)
    monkeypatch.setattr(family, "ARCHITECTURE_STAGE_REGISTRY", registry)
    cells = family.materialize_family_stage("X", tmp_path)
    assert len(cells) == 1
    assert "package" not in family.render_family_inventory("X")[0]


# ---------------------------------------------------------------------------
# T15 (GREEN expected). F2 consumption is real on the runner route: the
# registered selector is what select_cell calls, with the full coordinate
# set, and the registered packet policy is what _training_packet consumes.
# ---------------------------------------------------------------------------
def test_t15_the_registered_selector_and_packet_policy_are_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A's selector receives every runner coordinate; A's packet policy sets the cadence."""

    original = family.ARCHITECTURE_STAGE_REGISTRY["A"]
    calls: list[set[str]] = []

    def recording_selector(cells, **kwargs):
        calls.append(set(kwargs))
        return family.select_exact_one(cells, **kwargs)

    monkeypatch.setitem(
        family.ARCHITECTURE_STAGE_REGISTRY,
        "A",
        replace(
            original,
            selector=recording_selector,
            packet_policy=lambda stage: {"updates": 50_000, "checkpoint_every_n_updates": 25_000},
        ),
    )
    package = family.ab_universe.render_a_packages()[0]
    cells = family.materialize_family_stage("A", tmp_path)
    selection = runner.select_cell(
        cells,
        stage="A",
        context="A",
        optimizer="adam",
        seed_label=830001,
        literal_configuration=package["literal_configuration"],
    )
    assert len(calls) == 1
    assert {"stage", "context", "optimizer", "seed_label", "lr", "beta2", "literal_configuration"} <= calls[0]
    packet = runner._training_packet(selection.cell)
    assert packet.checkpoint_cadence.every_n_updates == 25_000
    assert tuple(packet.checkpoint_cadence.selected_updates) == (25_000, 50_000)
