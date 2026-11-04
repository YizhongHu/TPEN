"""Focused contract tests for the extensible architecture-stage registry."""

from __future__ import annotations

from types import SimpleNamespace
import importlib

import pytest


family = importlib.import_module(
    "experiments.atomistic.he-importance.architecture_stage_family"
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


def test_test_only_descriptor_can_register_against_a_patched_authority() -> None:
    authority = {
        code: SimpleNamespace(updates=7)
        for code in ("Q", "O1", "O2", "A", "B", "R", "F", "X")
    }
    registry: dict[str, family.ArchitectureStageDescriptor] = {}
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
        family="test-family",
        renderer=renderer,
        context_validator=lambda context: None,
        resolver=lambda cell: object(),
        resolved_validator=lambda cell, resolved: None,
        slot_policy=lambda row: (slot,),
        seed_namespace=lambda stage: (1, 2),
        selector=lambda *args, **kwargs: None,
        packet_policy=lambda stage: {"updates": 7},
        launch_adapter=lambda *args, **kwargs: None,
    )
    family.register_stage_descriptor(descriptor, registry=registry, authority=authority)
    assert tuple(registry) == ("X",)
    family.validate_partition(("Q", "O1"), ("A", "B", "X"), {"O2": "pending", "R": "pending", "F": "pending"}, authority)
