"""Focused tests for the declaration-driven two-backend traversal."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest
import sys

spec = importlib.util.spec_from_file_location("he_importance_content_traversal", Path(__file__).with_name("content_traversal.py"))
assert spec is not None and spec.loader is not None
content = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = content
spec.loader.exec_module(content)

def test_freeze_uses_immutable_containers_and_projection_uses_fresh_mutables() -> None:
    source = {"nested": {"values": [1, 2]}}
    frozen = content.freeze_content(source)
    assert isinstance(frozen, type(content.freeze_content({})))
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

def test_declaration_is_closed_before_source_read() -> None:
    class Exploding(dict):
        def items(self):
            raise AssertionError("source was read")
    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content(Exploding(value=1), object())
    assert caught.value.refusal is content.ContentRefusal.DECLARATION_NOT_CLOSED

def test_duplicate_and_non_string_keys_are_not_silently_overwritten() -> None:
    class Duplicate:
        def items(self):
            return [("a", 1), ("a", 2)]
    with pytest.raises(content.ContentTraversalError) as caught:
        content.freeze_content(Duplicate())
    assert caught.value.refusal is content.ContentRefusal.MAPPING_KEY_REPEATED

def test_nonfinite_scalars_are_refused() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(content.ContentTraversalError) as caught:
            content.freeze_content({"x": value})
        assert caught.value.refusal is content.ContentRefusal.CONTENT_NONFINITE
