"""Pre-review round 1 tests for PR #580 (AB-4c architecture-stage family).

Written by the independent reviewer against head
3954ba6de035364b61d6f552eff89bf29d10702d.  Each test states what it
establishes and whether it is expected GREEN or RED at this head.  RED tests
are regression controls for findings in the round-1 review comment; they are
not fixture bugs when they fail with the failure named in their docstring.
"""

from __future__ import annotations

import importlib
import json
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
    """Same guard as test_run_stage_q: keep the Q route testable without torch."""

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(optim=SimpleNamespace(Adam=_fake_adam)),
        )


# The exact Adam levels run_stage_q.select_cell injects for the default
# optimizer, so a fixture row is selectable through the runner's own filter.
_ADAM_LEVELS = {
    "lr": 0.005,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.0,
    "parameter_groups": 1,
    "schedule": "constant",
}

_FACTS = {
    "global_rank": 0, "global_size": 1, "local_rank": 0, "local_size": 1,
    "node_rank": 0, "node_size": 1, "host": "host", "pid": 1,
    "device": "cuda:0", "job_id": "job", "device_identity": None,
}


def _args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    values = {
        "output_root": tmp_path,
        "stage": "Q",
        "context": "O-raw",
        "optimizer": "adam",
        "seed_label": 810001,
        "walkers": stage_coordinate.COMMITTED_WALKERS,
        "dry_run": True,
        "literal_configuration": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _first_package(stage_code: str) -> dict:
    return family.ab_universe.render_ab_universe()[stage_code][0]


# ---------------------------------------------------------------------------
# T1 (GREEN expected). Clause 3, S4, G2 with the REAL registry resolver.
# The PR's own tests replace the resolver with a fake; this one does not.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage_code, seed_label", [("A", 830001), ("B", 840001)])
def test_t1_fixed_adam_rows_select_and_resolve_with_the_real_registry_resolver(
    tmp_path: Path, stage_code: str, seed_label: int
) -> None:
    """A/B fixed-Adam rows select and resolve through train_config unpatched.

    Establishes: the family route reaches a real resolved config (clause 3),
    the resolved seed is the authority stream for the selected label (S4),
    and the selected row's levels are the package's own Adam slot (S3).
    """

    descriptor = family.ARCHITECTURE_STAGE_REGISTRY[stage_code]
    assert descriptor.resolver is family._resolver, "registry must be unpatched here"
    package = _first_package(stage_code)
    cells = family.materialize_family_stage(stage_code, tmp_path)
    selection = runner.select_cell(
        cells,
        stage=stage_code,
        context=stage_code,
        optimizer="adam",
        seed_label=seed_label,
        literal_configuration=package["literal_configuration"],
    )
    packet, resolved = runner._resolve_selected_config(selection)
    assert resolved.optimizer._target_ == "torch.optim.Adam"
    assert resolved.trainer.max_steps == stage_coordinate.stage_definition(stage_code).updates
    expected_seed = stage_coordinate.seed_namespace(stage_code, seed_label)["model_initialization"]
    assert resolved.runtime.seed == expected_seed
    assert selection.selected.literal_configuration == package["literal_configuration"]
    assert selection.selected.optimizer["levels"] == package["optimizer_slots"][0]["levels"]
    assert packet.cell.manifest["stage"] == stage_code


# ---------------------------------------------------------------------------
# T2 (GREEN expected). Clause 3 end to end: execute --dry-run with nothing
# monkeypatched (the PR's dry-run test fakes the resolver and the roster
# receipt).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage_code, seed_label", [("A", 830001), ("B", 840001)])
def test_t2_family_dry_run_reaches_launch_construction_with_nothing_patched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], stage_code: str, seed_label: int
) -> None:
    """execute(--dry-run) for an A/B fixed-Adam row returns 0 with the real route."""

    package = _first_package(stage_code)
    args = _args(
        tmp_path,
        stage=stage_code,
        context=stage_code,
        literal_configuration=package["literal_configuration"],
        seed_label=seed_label,
        dry_run=True,
    )
    assert runner.execute(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["selection"]["selected"]["literal_configuration"] == package["literal_configuration"]
    assert report["selection"]["selected"]["optimizer"]["method"] == "adam"
    assert report["resolved_config"]["optimizer"]["_target_"] == "torch.optim.Adam"
    assert report["resolved_config"]["trainer"]["max_steps"] == (
        stage_coordinate.stage_definition(stage_code).updates
    )
    assert report["train_packet"]["content_hash"]
    assert not list(tmp_path.rglob("selection_receipt.json")), "dry run must write nothing"


# ---------------------------------------------------------------------------
# T3 (GREEN expected). G2/G5 on the family LAUNCH route: the selection-time
# and launch-time resolved views agree through the guarded runner, topology
# facts reach the runner, and completion is marked only after the snapshot.
# ---------------------------------------------------------------------------
def test_t3_family_launch_route_agrees_selection_and_launch_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-dry-run A launch through descriptor.launch_adapter with the real resolver."""

    package = _first_package("A")
    observed: dict[str, object] = {}

    def recording_run_from_config(config: object, *, topology: object) -> int:
        observed["topology"] = topology
        output = Path(config.run.dir) / "resolved_config.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(OmegaConf.to_yaml(config), encoding="utf-8")
        observed["config"] = config
        return 0

    monkeypatch.setattr(runner, "_execution_facts", lambda: dict(_FACTS))
    monkeypatch.setattr(runner.launch, "run_from_config", recording_run_from_config)
    args = _args(
        tmp_path,
        stage="A",
        context="A",
        literal_configuration=package["literal_configuration"],
        seed_label=830001,
        dry_run=False,
    )
    assert runner.execute(args) == 0
    report = json.loads(capsys.readouterr().out)
    # The report is the selection-time view after json.dumps(default=str);
    # normalize the launch-time view the same way before comparing.
    launched = json.loads(
        json.dumps(OmegaConf.to_container(observed["config"], resolve=True), default=str)
    )
    assert launched == report["resolved_config"], "launch view must equal the recorded selection view"
    assert observed["topology"] == _FACTS
    run_dir = Path(observed["config"].run.dir)
    assert (run_dir / "COMPLETE").read_text(encoding="utf-8") == "completed\n"
    assert (run_dir / "selection_receipt.json").is_file()
    assert run_dir.parent.name == "A"


# ---------------------------------------------------------------------------
# T4 (GREEN after F1(a)). The original clause-8 fixture is materializable
# through the generic route once validation and materialization agree on the
# package-optional row schema.
# ---------------------------------------------------------------------------
def _clause_8_fixture() -> tuple[dict, family.ArchitectureStageDescriptor]:
    authority = {
        code: SimpleNamespace(updates=7)
        for code in ("Q", "O1", "O2", "A", "B", "R", "F", "X")
    }
    slot = family.OptimizerSlot("adam", "available", {"lr": 1.0})

    def renderer(stage: str):
        package = {"optimizer_slots": [slot.as_dict()]}
        return ({
            "stage": stage,
            "package": package,
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
        family="test-family",
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
    return authority, descriptor


def test_t4_the_pr_clause_8_fixture_is_materializable_on_the_generic_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A descriptor that passes registration must also pass materialization."""

    authority, descriptor = _clause_8_fixture()
    definition = stage_coordinate.StageDefinition(
        code="X",
        purpose="review fixture",
        seeds_per_point=2,
        maximum_lineages=1,
        updates=7,
        sampler_overrides=MappingProxyType({}),
    )
    copies = {id(stage_coordinate): stage_coordinate, id(optimizer_grid.stage_coordinate): optimizer_grid.stage_coordinate}
    for module in copies.values():
        monkeypatch.setattr(module, "STAGE_DEFINITIONS", module.STAGE_DEFINITIONS + (definition,))
        monkeypatch.setitem(module.STAGES, "X", definition)
        monkeypatch.setitem(module.SEED_NAMESPACE_STARTS, "X", 880_001)
    registry = dict(family.ARCHITECTURE_STAGE_REGISTRY)
    family.register_stage_descriptor(descriptor, registry=registry, authority=authority)
    monkeypatch.setattr(family, "ARCHITECTURE_STAGE_REGISTRY", registry)
    monkeypatch.setattr(family, "STAGE_AUTHORITY", authority)
    cells = family.materialize_family_stage("X", tmp_path)
    assert cells


# ---------------------------------------------------------------------------
# T5 (exploratory; result is the evidence). The operator's condition made
# concrete: a COMPLETE fake stage with a test-patched authority and seed
# table, registered by descriptor only, is driven through execute --dry-run.
# No family code is edited.  Whatever stops it first is the answer to
# "could O2/R/F be admitted without editing the family's code".
# ---------------------------------------------------------------------------
def _complete_fake_stage(monkeypatch: pytest.MonkeyPatch) -> family.ArchitectureStageDescriptor:
    definition = stage_coordinate.StageDefinition(
        code="X",
        purpose="review fixture",
        seeds_per_point=2,
        maximum_lineages=2,
        updates=2000,
        sampler_overrides=MappingProxyType({}),
    )
    # optimizer_grid binds a PRIVATE stage_coordinate module object
    # (spec_from_file_location), distinct from the sys.modules one that
    # run_stage_q, train_config and the family use.  Patch every copy.
    copies = {id(stage_coordinate): stage_coordinate}
    copies.setdefault(id(optimizer_grid.stage_coordinate), optimizer_grid.stage_coordinate)
    for module in copies.values():
        monkeypatch.setattr(
            module, "STAGE_DEFINITIONS", module.STAGE_DEFINITIONS + (definition,)
        )
        # family.STAGE_AUTHORITY is the same dict object as stage_coordinate.STAGES.
        monkeypatch.setitem(module.STAGES, "X", definition)
        monkeypatch.setitem(module.SEED_NAMESPACE_STARTS, "X", 870_001)
    slot = family.OptimizerSlot("adam", "available", dict(_ADAM_LEVELS))
    package = {"literal_configuration": {"label": "x"}, "optimizer_slots": [slot.as_dict()]}

    def renderer(stage: str):
        return ({
            "stage": stage,
            "package": package,
            "optimizer": slot.as_dict(),
            "configurations": ({
                "scientific_identity": {
                    "context": {"axis": "X", "literal_configuration": {"label": "x"}},
                    "optimizer": {"method": "adam", "levels": dict(_ADAM_LEVELS)},
                },
                "payload": {"updates": 2000},
                "topology": {},
            },),
        },)

    def context_validator(context) -> None:
        if frozenset(context) != frozenset({"axis", "literal_configuration"}):
            raise family.ArchitectureStageFamilyError("X context keys")
        if context["axis"] != "X" or dict(context["literal_configuration"]) != {"label": "x"}:
            raise family.ArchitectureStageFamilyError("X context values")

    streams = stage_coordinate.seed_namespace("X", 870_001)
    model = {"initializer": {"seed": streams["model_initialization"]}}
    optimizer = {
        "_target_": "torch.optim.Adam",
        "lr": _ADAM_LEVELS["lr"],
        "betas": [_ADAM_LEVELS["beta1"], _ADAM_LEVELS["beta2"]],
        "eps": _ADAM_LEVELS["epsilon"],
        "weight_decay": _ADAM_LEVELS["weight_decay"],
    }
    sampler = {"seed": streams["training_sampler"]}
    resolved = OmegaConf.create(
        {
            "model": model,
            "optimizer": optimizer,
            "sampler": sampler,
            "trainer": {"max_steps": 2000},
            "hamiltonian_terms": {},
            "runner": {
                "model": model,
                "optimizer": optimizer,
                "sampler": sampler,
                "trainer": {"max_steps": 2000},
                "hamiltonian_terms": {},
            },
        }
    )
    descriptor = family.ArchitectureStageDescriptor(
        stage="X",
        family="review-fixture",
        renderer=renderer,
        context_validator=context_validator,
        resolver=lambda source: resolved,
        resolved_validator=lambda cell, config: None,
        slot_policy=lambda row: (slot,),
        seed_namespace=lambda stage: stage_coordinate.seed_labels(stage),
        selector=family.select_exact_one,
        packet_policy=lambda stage: {"checkpoint_every_n_updates": 1_000, "updates": 2000},
        launch_adapter=lambda *args, **kwargs: 0,
    )
    scratch: dict[str, family.ArchitectureStageDescriptor] = {}
    family.register_stage_descriptor(descriptor, registry=scratch)
    monkeypatch.setitem(family.ARCHITECTURE_STAGE_REGISTRY, "X", descriptor)
    return descriptor


def test_t5_a_complete_descriptor_only_stage_reaches_dry_run_through_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fake stage X, admitted by descriptor + patched authority, through execute."""

    _complete_fake_stage(monkeypatch)
    assert optimizer_grid.undeclared_stage_reason("X") is None
    assert family.is_family_stage("X")
    args = _args(
        tmp_path,
        stage="X",
        context="X",
        literal_configuration={"label": "x"},
        seed_label=870_001,
        dry_run=True,
    )
    assert runner.execute(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["selection"]["selected"]["stage"] == "X"
    assert report["selection"]["selected"]["seed_label"] == 870_001
    assert report["resolved_config"]["optimizer"]["_target_"] == "torch.optim.Adam"


# ---------------------------------------------------------------------------
# T6 (RED by design -> finding). The grid route accepts and silently ignores
# a literal_configuration coordinate.  Clause 2 keeps Q/O1 behaviour; a new
# coordinate that is consumed by nothing should be refused by name, not
# dropped.  Expected failure at this head: DID NOT RAISE.
# ---------------------------------------------------------------------------
def test_t6_grid_selection_refuses_a_literal_configuration_coordinate(tmp_path: Path) -> None:
    """select_cell on Q with literal_configuration must refuse, naming the field."""

    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path, availability=optimizer_grid.roster_availability(), stages=("Q",)
    )
    with pytest.raises(runner.StageQSelectionError, match="literal_configuration"):
        runner.select_cell(cells, seed_label=810001, literal_configuration={"bogus": "x"})


# ---------------------------------------------------------------------------
# T7 (RED by design -> finding). Rendering the family inventory must not
# re-render the whole A/B universe per row.  Counted at this head: 1019
# render_ab_universe calls for one render_family_inventory("A") and 1260 at
# import of architecture_stage_family (about 10 s each on a laptop).
# ---------------------------------------------------------------------------
def test_t7_family_inventory_renders_the_universe_a_bounded_number_of_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """render_family_inventory('A') calls ab_universe.render_ab_universe at most 4 times."""

    calls = {"n": 0}
    original = family.ab_universe.render_ab_universe

    def counted():
        calls["n"] += 1
        return original()

    monkeypatch.setattr(family.ab_universe, "render_ab_universe", counted)
    rows = family.render_family_inventory("A")
    assert len(rows) == 508
    assert calls["n"] <= 4, f"render_ab_universe was called {calls['n']} times"


# ---------------------------------------------------------------------------
# T8 (RED by design -> finding). The family truthfulness guard accepts a
# resolved config that consumed nothing (no model, no seed, no horizon); the
# grid guard refuses the same config.  The Q half is the GREEN control.
# Expected failure at this head: DID NOT RAISE on the A half.
# ---------------------------------------------------------------------------
def test_t8_family_truthfulness_guard_refuses_a_config_that_consumed_nothing(
    tmp_path: Path,
) -> None:
    """_require_truthful_cell must refuse a hollow config for A as it does for Q."""

    hollow = OmegaConf.create({"optimizer": {"_target_": "torch.optim.Adam"}})
    q_cells = stage_coordinate.materialize_intended_configurations(
        tmp_path / "q", availability=optimizer_grid.roster_availability(), stages=("Q",)
    )
    with pytest.raises(runner.StageQSelectionError):
        runner._require_truthful_cell(q_cells[0], hollow)
    a_cells = family.materialize_family_stage("A", tmp_path / "a")
    a_cell = next(
        cell for cell in a_cells
        if cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
    )
    with pytest.raises(runner.StageQSelectionError):
        runner._require_truthful_cell(a_cell, hollow)


# ---------------------------------------------------------------------------
# T9 (GREEN expected). S6 / clause 4: identity, never display name.
# ---------------------------------------------------------------------------
def test_t9_selection_is_by_identity_never_display_name(tmp_path: Path) -> None:
    """display_name is refused as a coordinate and absent from receipt and manifest."""

    cells = family.materialize_family_stage("B", tmp_path)
    with pytest.raises(runner.StageQSelectionError, match="closed-vocabulary"):
        runner.select_cell(
            cells,
            stage="B",
            context="B",
            optimizer="adam",
            seed_label=840001,
            literal_configuration={"display_name": "B001"},
        )
    package = _first_package("B")
    selection = runner.select_cell(
        cells,
        stage="B",
        context="B",
        optimizer="adam",
        seed_label=840001,
        literal_configuration=package["literal_configuration"],
    )
    assert "display_name" not in json.dumps(selection.receipt(), default=str)
    assert "display_name" not in json.dumps(selection.cell.manifest, default=str)


# ---------------------------------------------------------------------------
# T10 (GREEN expected). Clause 6 mutant on the ROUTE, not on a helper: a
# universe whose O* slot is relabelled as a second Adam is refused before
# any cell is materialized.
# ---------------------------------------------------------------------------
def test_t10_a_universe_that_relabels_o_star_as_adam_is_refused_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """materialize_family_stage('A') refuses a relabelled O* slot by name."""

    original = family.ab_universe.render_ab_universe

    def relabelled():
        universe = original()
        packages = []
        for package in universe["A"]:
            package = dict(package)
            slots = [dict(slot) for slot in package["optimizer_slots"]]
            slots[1] = {"method": "adam", "status": "available", "levels": dict(slots[0]["levels"])}
            package["optimizer_slots"] = slots
            packages.append(package)
        return {"A": tuple(packages), "B": universe["B"]}

    monkeypatch.setattr(family.ab_universe, "render_ab_universe", relabelled)
    with pytest.raises(family.ArchitectureStageFamilyError, match=r"Adam and O\* exactly once"):
        family.materialize_family_stage("A", tmp_path)
    assert not list(tmp_path.rglob("*")), "nothing may be materialized"
