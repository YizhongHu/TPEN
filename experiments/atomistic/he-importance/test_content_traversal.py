"""Focused tests for the declaration-driven two-backend traversal."""
from __future__ import annotations
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
import pytest
import sys

_target = Path(__file__).with_name("content_traversal.py").resolve()
_canonical = sys.modules.get("content_traversal")
if _canonical is not None and Path(getattr(_canonical, "__file__", "")).resolve() == _target:
    content = _canonical
else:
    spec = importlib.util.spec_from_file_location("content_traversal", _target)
    assert spec is not None and spec.loader is not None
    content = importlib.util.module_from_spec(spec)
    sys.modules.update({spec.name: content})
    spec.loader.exec_module(content)
sys.modules.update({"he_importance_content_traversal": content})

def test_freeze_uses_immutable_containers_and_projection_uses_fresh_mutables() -> None:
    source = {"nested": {"values": [1, 2]}}
    frozen = content.freeze_content(source)
    # Anchor the oracle outside the production call: a mutable-dict mutant
    # must fail this assertion instead of moving both sides together.
    assert isinstance(frozen, MappingProxyType)
    assert isinstance(frozen["nested"]["values"], tuple)
    projected = content.project_content(frozen)
    assert isinstance(projected, dict) and isinstance(projected["nested"]["values"], list)
    projected["nested"]["values"].append(3)
    assert frozen["nested"]["values"] == (1, 2)

def test_alias_is_read_once_and_cycle_is_refused() -> None:
    class Viewed(dict):
        reads = 0
        def items(self):
            self.reads += 1
            return super().items()
    child = Viewed(value=1)
    frozen = content.freeze_content({"a": child, "b": child})
    assert child.reads == 1 and frozen["a"] == frozen["b"]
    cycle = {}
    cycle["self"] = cycle
    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content(cycle)
    assert caught.value.refusal is content.ContentRefusal.CONTENT_CYCLE

class EmittedMapping(Mapping):
    """Stable mapping adapter whose first emitted stream is its only stream."""
    def __init__(self, duplicate: bool):
        self.duplicate = duplicate

    def __getitem__(self, key):
        return 1

    def __iter__(self):
        return iter(("a",))

    def __len__(self):
        return 1

    def items(self):
        entries = [("a", 1)]
        if self.duplicate:
            entries.append(("a", 2))
        return entries

def test_stable_unique_mapping_is_admitted() -> None:
    source = EmittedMapping(duplicate=False)
    frozen = content.freeze_content(source)
    assert content.project_content(frozen) == {"a": 1}
    selection_spec = importlib.util.spec_from_file_location(
        "he_importance_selection_commit_control", Path(__file__).with_name("outcome_selection.py")
    )
    assert selection_spec is not None and selection_spec.loader is not None
    selection = importlib.util.module_from_spec(selection_spec)
    sys.modules.update({selection_spec.name: selection})
    selection_spec.loader.exec_module(selection)
    commitment = selection.SelectionCommitment.create(
        ("cell-a",), source, selection.IntervalContract("mean", "two-sided-t", 0.95, "holm")
    )
    commitment.verify()


def test_malformed_mapping_items_are_refused() -> None:
    class Malformed(EmittedMapping):
        def items(self):
            return [("a", 1, "extra")]

    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content(Malformed(duplicate=False))
    assert caught.value.refusal is content.ContentRefusal.MAPPING_ITEM_MALFORMED

def test_declaration_is_closed_before_source_read() -> None:
    class Exploding(dict):
        def items(self):
            raise AssertionError("source was read")
    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content(Exploding(value=1), object())
    assert caught.value.refusal is content.ContentRefusal.DECLARATION_NOT_CLOSED

def test_duplicate_and_non_string_keys_are_refused_without_silent_overwrite() -> None:
    class Duplicate(EmittedMapping):
        def __init__(self):
            super().__init__(duplicate=True)
    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content(Duplicate())
    assert caught.value.refusal is content.ContentRefusal.MAPPING_KEY_REPEATED

    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content({1: "not an admitted key"})
    assert caught.value.refusal is content.ContentRefusal.MAPPING_KEY_NOT_EXACT_STRING


def test_surrogate_mapping_keys_are_refused() -> None:
    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content({chr(0xD800): "value"})
    assert caught.value.refusal is content.ContentRefusal.STRING_HAS_SURROGATE_CODEPOINT


def test_surrogate_pair_key_spelling_is_refused() -> None:
    astral, pair = chr(0x1F600), chr(0xD83D) + chr(0xDE00)
    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content({pair: "value"})
    assert caught.value.refusal is content.ContentRefusal.STRING_HAS_SURROGATE_CODEPOINT
    assert caught.value.path == (("key", pair),)
    admitted = content.freeze_content({astral: "value"})
    assert content.project_content(admitted) == {astral: "value"}


def test_distinct_admitted_unicode_scalars_have_distinct_utf8_and_canonical_bytes() -> None:
    first, second = chr(0x1F600), chr(0x1F601)
    assert first.encode("utf-8") == b"\xf0\x9f\x98\x80"
    assert second.encode("utf-8") == b"\xf0\x9f\x98\x81"
    assert first.encode("utf-8") != second.encode("utf-8")

def test_nonfinite_scalars_are_refused() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(content.ContentTraversalError) as caught:
            content.freeze_content({"x": value})
        assert caught.value.refusal is content.ContentRefusal.CONTENT_NONFINITE


@pytest.mark.parametrize("payload", [{}, [], {"nested": []}])
def test_open_declaration_payloads_are_refused(payload: object) -> None:
    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content(payload, object())
    assert caught.value.refusal is content.ContentRefusal.DECLARATION_NOT_CLOSED


@pytest.mark.parametrize("payload", [{"k": object()}, [object()]])
def test_unknown_carriers_are_refused_on_each_route(payload: object) -> None:
    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content(payload)
    assert caught.value.refusal is content.ContentRefusal.CONTENT_KIND_UNDECLARED


def test_array_ancestor_cycles_are_refused() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content(cycle)
    assert caught.value.refusal is content.ContentRefusal.CONTENT_CYCLE
    assert caught.value.path == (("index", 0),)


def test_refusals_carry_literal_tagged_paths() -> None:
    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content({"outer": [{"inner": object()}]})
    assert caught.value.refusal is content.ContentRefusal.CONTENT_KIND_UNDECLARED
    assert caught.value.path == (("key", "outer"), ("index", 0), ("key", "inner"))


def test_admitted_empty_and_boundary_values_round_trip() -> None:
    value = {"empty-map": {}, "empty-array": [], "boundary": chr(0x10FFFF)}
    frozen = content.freeze_content(value)
    assert content.project_content(frozen) == value


def test_changing_mapping_views_are_read_once_and_refused() -> None:
    class Changing(Mapping):
        reads = 0

        def __getitem__(self, key: str) -> object:
            return 1

        def __iter__(self):
            return iter(("a",))

        def __len__(self) -> int:
            return 1

        def items(self):
            self.reads += 1
            return [("a", self.reads)]

    source = Changing()
    frozen = content.freeze_content(source)
    assert source.reads == 1
    assert content.project_content(frozen) == {"a": 1}


def test_keys_are_refused_for_subclass() -> None:
    class SubclassKey(str):
        pass

    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content({SubclassKey("key"): "value"})
    assert caught.value.refusal is content.ContentRefusal.MAPPING_KEY_NOT_EXACT_STRING
