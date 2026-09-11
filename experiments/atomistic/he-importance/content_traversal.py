"""Structural content traversal shared by helium-importance contract boundaries.

Callers own their predicates.  This module owns only the structural walk so a
reference-name screen and an inherited-walker-state screen cannot drift into
separate, container-type-specific implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any


Path = tuple[str, ...]
Predicate = Callable[[Path, Any], bool]


def walk(value: Any, predicate: Predicate) -> tuple[tuple[Path, Any], ...]:
    """Return every structural node accepted by ``predicate``.

    The walk reaches mapping keys and values, sequence members, dataclass
    fields, ``__dict__`` attributes, and ``__slots__`` attributes.  It tracks
    object identities to safely traverse cyclic records without treating equal
    but distinct scalar values as cycles.
    """

    matches: list[tuple[Path, Any]] = []
    active: set[int] = set()

    def visit(node: Any, path: Path) -> None:
        if predicate(path, node):
            matches.append((path, node))
        if isinstance(node, (str, bytes, bytearray, int, float, bool, type(None))):
            return
        node_id = id(node)
        if node_id in active:
            return
        active.add(node_id)
        try:
            if isinstance(node, Mapping):
                for key, nested in node.items():
                    key_path = path + (f"key:{key!s}",)
                    visit(key, key_path)
                    visit(nested, path + (str(key),))
            elif isinstance(node, Sequence):
                for index, nested in enumerate(node):
                    visit(nested, path + (str(index),))
            elif is_dataclass(node) and not isinstance(node, type):
                for field in fields(node):
                    visit(getattr(node, field.name), path + (field.name,))
            elif hasattr(node, "__dict__"):
                for name, nested in vars(node).items():
                    visit(nested, path + (name,))
            else:
                for cls in type(node).__mro__:
                    slots = cls.__dict__.get("__slots__", ())
                    if isinstance(slots, str):
                        slots = (slots,)
                    for name in slots:
                        if name in {"__dict__", "__weakref__"} or not hasattr(node, name):
                            continue
                        visit(getattr(node, name), path + (name,))
        finally:
            active.remove(node_id)

    visit(value, ())
    return tuple(matches)
