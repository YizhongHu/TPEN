"""Focused contract tests for the extensible architecture-stage registry."""

from __future__ import annotations

import importlib
import json
from types import MappingProxyType, SimpleNamespace

import pytest


family = importlib.import_module(
    "experiments.atomistic.he-importance.architecture_stage_family"
)
optimizer_grid = importlib.import_module(
    "experiments.atomistic.he-importance.optimizer_grid"
)
runner = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")
stage_coordinate = importlib.import_module(
    "experiments.atomistic.he-importance.stage_coordinate"
)


def test_a_and_b_are_one_family_with_deterministic_literal_rendering() -> None:
    assert family.family_stage_codes() == ("A", "B")
    for stage in family.family_stage_codes():
        first = family.render_family_inventory(stage)
        second = family.render_family_inventory(stage)
        assert first == second
        assert {row["stage"] for row in first} == {stage}
        assert all(
            set(row["configurations"][0]["scientific_identity"]["context"]) ==
            {"axis", "literal_configuration"}
            for row in first
        )


def test_o_star_slot_is_unavailable_with_the_recorded_reason_before_resolution(tmp_path) -> None:
    cells = family.materialize_family_stage("A", tmp_path)
    package = family.ab_universe.render_a_packages()[0]
    with pytest.raises(family.ArchitectureStageFamilyError, match="selection output of O2 is pending"):
        family.select_exact_one(
            cells,
            stage="A",
            context="A",
            literal_configuration=package["literal_configuration"],
            optimizer="O*",
            seed_label=830001,
        )


def test_o_star_to_adam_slot_relabel_mutant_is_rejected() -> None:
    package = dict(family.ab_universe.render_a_packages()[0])
    slots = [dict(slot) for slot in package["optimizer_slots"]]
    slots[1]["method"] = "adam"
    package["optimizer_slots"] = slots
    with pytest.raises(family.ArchitectureStageFamilyError, match=r"Adam and O\* exactly once"):
        family._ab_slot_policy(package)


def test_test_only_descriptor_with_its_own_context_and_selector_uses_the_real_route(
    tmp_path, monkeypatch, capsys
) -> None:
    """A descriptor-only stage reaches runner selection without family edits."""

    definition = stage_coordinate.StageDefinition(
        code="X",
        purpose="review fixture",
        seeds_per_point=1,
        maximum_lineages=1,
        updates=7,
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
        monkeypatch.setitem(module.SEED_NAMESPACE_STARTS, "X", 870_001)

    levels = {
        "lr": 0.005,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "parameter_groups": 1,
        "schedule": "constant",
    }
    slot = family.OptimizerSlot("adam", "available", levels)
    package = {"optimizer_slots": [slot.as_dict()]}
    context = {"basis": "X", "literal": {"label": "x"}}

    def renderer(stage: str):
        return ({
            "stage": stage,
            "package": package,
            "optimizer": slot.as_dict(),
            "configurations": ({
                "scientific_identity": {
                    "context": context,
                    "optimizer": {"method": "adam", "levels": dict(levels)},
                },
                "payload": {"updates": 7},
                "topology": {},
            },),
        },)

    def select_own_context(cells, *, stage, context, optimizer, seed_label, **kwargs):
        del kwargs
        matches = [
            cell
            for cell in cells
            if cell.manifest["scientific_identity"]["context"] == context
            and cell.manifest["scientific_identity"]["optimizer"]["method"] == optimizer
            and cell.manifest["seed_identity"]["label"] == seed_label
        ]
        assert len(matches) == 1
        selected = family._cell_descriptor(matches[0])
        return family.FamilySelection(matches[0], selected, ())

    streams = stage_coordinate.seed_namespace("X", 870_001)
    model = {"initializer": {"seed": streams["model_initialization"]}}
    optimizer = {
        "_target_": "torch.optim.Adam",
        "lr": levels["lr"],
        "betas": [levels["beta1"], levels["beta2"]],
        "eps": levels["epsilon"],
        "weight_decay": levels["weight_decay"],
    }
    sampler = {"seed": streams["training_sampler"]}
    resolved = runner.OmegaConf.create(
        {
            "model": model,
            "optimizer": optimizer,
            "sampler": sampler,
            "trainer": {"max_steps": 7},
            "hamiltonian_terms": {},
            "runner": {
                "model": model,
                "optimizer": optimizer,
                "sampler": sampler,
                "trainer": {"max_steps": 7},
                "hamiltonian_terms": {},
            },
        }
    )

    descriptor = family.ArchitectureStageDescriptor(
        stage="X",
        family="review-fixture",
        renderer=renderer,
        context_validator=lambda value: (
            None
            if value == context
            else (_ for _ in ()).throw(family.ArchitectureStageFamilyError("X context"))
        ),
        resolver=lambda cell: resolved,
        resolved_validator=lambda cell, resolved: None,
        slot_policy=lambda row: (slot,),
        seed_namespace=lambda stage: stage_coordinate.seed_labels(stage),
        selector=select_own_context,
        packet_policy=lambda stage: {
            "updates": 7,
            "checkpoint_every_n_updates": 1,
        },
        launch_adapter=lambda *args, **kwargs: 0,
    )
    monkeypatch.setitem(family.ARCHITECTURE_STAGE_REGISTRY, "X", descriptor)
    args = SimpleNamespace(
        output_root=tmp_path,
        stage="X",
        context=context,
        optimizer="adam",
        lr=levels["lr"],
        beta2=levels["beta2"],
        seed_label=870_001,
        walkers=stage_coordinate.COMMITTED_WALKERS,
        dry_run=True,
        literal_configuration=None,
    )
    assert runner.execute(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["selection"]["selected"]["context"] == context
    assert report["resolved_config"]["trainer"]["max_steps"] == 7
    assert report["train_packet"]["content_hash"]
