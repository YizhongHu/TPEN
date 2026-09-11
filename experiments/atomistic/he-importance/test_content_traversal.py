"""Structural traversal tests using record objects, not mapping stand-ins."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "content_traversal", Path(__file__).with_name("content_traversal.py")
)
assert _SPEC is not None and _SPEC.loader is not None
traversal = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = traversal
_SPEC.loader.exec_module(traversal)


@dataclass
class OutcomeRow:
    label: str
    payload: object


class SlotRecord:
    __slots__ = ("route",)

    def __init__(self, route: object) -> None:
        self.route = route


def test_walk_reaches_dataclass_and_slots_record_fields_not_just_mappings() -> None:
    value = OutcomeRow("ordinary", SlotRecord({"reference_energy": "named payload"}))
    matches = traversal.walk(value, lambda path, _: path[-1:] == ("reference_energy",))
    assert matches == ((("payload", "route", "reference_energy"), "named payload"),)


def test_walk_reaches_mapping_keys_and_values_through_a_record() -> None:
    value = OutcomeRow("ordinary", {"sampler": ({"label": "nested"},)})
    matches = traversal.walk(value, lambda path, node: path[-1:] == ("label",) and node == "nested")
    assert matches == ((("payload", "sampler", "0", "label"), "nested"),)
