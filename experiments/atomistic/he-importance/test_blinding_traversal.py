"""Contract tests for the shared structural blinding walker."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "he_importance_blinding_traversal", Path(__file__).with_name("blinding_traversal.py")
)
assert _SPEC is not None and _SPEC.loader is not None
walker = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = walker
_SPEC.loader.exec_module(walker)


class ProbeError(ValueError):
    """Structured refusal used to inspect the walker's attribution."""

    def __init__(self, path: str, content: object, reason: str):
        super().__init__(f"{path}: {reason}: {content!r}")
        self.path = path
        self.content = content
        self.reason = reason


def _screen(value: object, label: str = "probe") -> list[str]:
    seen: list[str] = []

    def predicate(text: str) -> bool:
        seen.append(text)
        return "reference" in text.lower()

    def error(path: str, content: object, reason: str) -> Exception:
        return ProbeError(path, content, reason)

    walker.walk_blinding_content(value, label, is_forbidden=predicate, error_factory=error)
    return seen


@dataclass(frozen=True)
class DataclassPayload:
    name: str
    nested: tuple[object, ...]


class SlottedPayload:
    __slots__ = ("name", "nested")

    def __init__(self, name: str, nested: object):
        self.name = name
        self.nested = nested


def test_walks_structural_nodes_and_normalizes_keys_and_leaves() -> None:
    value = DataclassPayload(
        "benign",
        (
            {b"reference_energy": "benign"},
            SlottedPayload("benign", {"leaf": bytearray(b"reference_energy")}),
            Path("reference_energy"),
        ),
    )
    with pytest.raises(ProbeError) as caught:
        _screen(value)
    assert caught.value.path == "probe.nested[0].reference_energy"
    assert caught.value.reason == "key"

    with pytest.raises(ProbeError) as caught:
        _screen(DataclassPayload("benign", ("benign", b"reference_energy")))
    assert caught.value.path == "probe.nested[1]"
    assert caught.value.reason == "value"

    with pytest.raises(ProbeError) as caught:
        _screen(SlottedPayload("benign", Path("reference_energy")))
    assert caught.value.path == "probe.nested"
    assert caught.value.reason == "value"


def test_walks_set_and_plain_object_attributes_with_stable_paths() -> None:
    class Payload:
        def __init__(self) -> None:
            self.items = {"alpha", "reference_energy"}

    with pytest.raises(ProbeError) as caught:
        _screen(Payload())
    assert caught.value.path == "probe.items[1]"
    assert caught.value.content == "reference_energy"


def test_cycle_back_edge_is_skipped_and_later_sibling_is_screened() -> None:
    value: dict[str, object] = {}
    value["self"] = value
    assert _screen(value) == ["self"]

    value["sibling"] = "reference_energy"
    with pytest.raises(ProbeError) as caught:
        _screen(value)
    assert caught.value.path == "probe.sibling"
    assert caught.value.reason == "value"


def test_undecodable_bytes_are_refused() -> None:
    with pytest.raises(ProbeError) as caught:
        _screen(b"\xff")
    assert caught.value.path == "probe"
    assert caught.value.reason == "decode"


def test_benign_normalized_values_reach_the_predicate() -> None:
    seen = _screen({b"key": bytearray(b"value"), Path("path"): ["leaf"]})
    assert seen == ["key", "value", "path", "leaf"]
