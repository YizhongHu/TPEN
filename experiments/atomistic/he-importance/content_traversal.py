"""Closed, read-once traversal for content-addressed experiment declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any


class ContentRefusal(str, Enum):
    """Stable identities for failures at the content boundary."""

    DECLARATION_NOT_CLOSED = "declaration_not_closed"
    CONTENT_KIND_UNDECLARED = "content_kind_undeclared"
    MAPPING_ITEM_MALFORMED = "mapping_item_malformed"
    MAPPING_KEY_NOT_EXACT_STRING = "mapping_key_not_exact_string"
    MAPPING_KEY_REPEATED = "mapping_key_repeated"
    CONTENT_NONFINITE = "content_nonfinite"
    CONTENT_CYCLE = "content_cycle"
    STRING_HAS_SURROGATE_CODEPOINT = "string_has_surrogate_codepoint"


class ContentTraversalError(ValueError):
    """A closed declaration rejected content, with an exact path witness."""

    def __init__(self, message: str, *, refusal: ContentRefusal, path: tuple[Any, ...]):
        super().__init__(message)
        self.refusal = refusal
        self.path = path


class _Literal:
    pass


class _MappingDeclaration:
    pass


class _ArrayDeclaration:
    pass


LITERAL_DECLARATION = _Literal()
CRITERIA_DECLARATION = _MappingDeclaration()
_ARRAY = _ArrayDeclaration()


def _error(refusal: ContentRefusal, path: tuple[Any, ...], message: str) -> ContentTraversalError:
    return ContentTraversalError(message, refusal=refusal, path=path)


def _validate_declaration(declaration: Any, seen: set[int] | None = None) -> None:
    """Reject open declarations before touching caller-owned content."""

    if declaration is LITERAL_DECLARATION:
        return
    if declaration is CRITERIA_DECLARATION:
        return
    if declaration is _ARRAY:
        return
    raise _error(ContentRefusal.DECLARATION_NOT_CLOSED, (), "content declaration is not closed")


def _validate_string(value: str, path: tuple[Any, ...]) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _error(ContentRefusal.STRING_HAS_SURROGATE_CODEPOINT, path, "string contains a surrogate code point")
    return value


def _walk(
    value: Any,
    declaration: Any,
    path: tuple[Any, ...],
    *,
    frozen: bool,
    active: set[int],
    memo: dict[int, Any],
    held: list[Any],
) -> Any:
    if declaration is CRITERIA_DECLARATION:
        declaration = _MappingDeclaration
    if declaration is _MappingDeclaration:
        kind = "mapping"
    elif declaration is _ARRAY:
        kind = "array"
    elif declaration is LITERAL_DECLARATION:
        kind = "literal"
    else:
        raise _error(ContentRefusal.DECLARATION_NOT_CLOSED, path, "content declaration is not closed")

    if kind == "mapping":
        if not isinstance(value, Mapping):
            raise _error(ContentRefusal.CONTENT_KIND_UNDECLARED, path, "declared mapping is not a mapping")
        return _walk_mapping(value, path, frozen=frozen, active=active, memo=memo, held=held)
    if kind == "array":
        if type(value) not in (list, tuple):
            raise _error(ContentRefusal.CONTENT_KIND_UNDECLARED, path, "declared array is not an exact list or tuple")
        identity = id(value)
        if identity in active:
            raise _error(ContentRefusal.CONTENT_CYCLE, path, "content contains an ancestor cycle")
        if identity in memo:
            return memo[identity]
        held.append(value)
        active.add(identity)
        try:
            items = [_walk(item, LITERAL_DECLARATION, path + (("index", index),), frozen=frozen, active=active, memo=memo, held=held) for index, item in enumerate(value)]
            result = tuple(items) if frozen else list(items)
            memo[identity] = result
            return result
        finally:
            active.remove(identity)

    if type(value) is str:
        return _validate_string(value, path)
    if type(value) is bool or type(value) is int or value is None:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise _error(ContentRefusal.CONTENT_NONFINITE, path, "content float must be finite")
        return value
    if isinstance(value, Mapping):
        return _walk_mapping(value, path, frozen=frozen, active=active, memo=memo, held=held)
    if type(value) in (list, tuple):
        return _walk(value, _ARRAY, path, frozen=frozen, active=active, memo=memo, held=held)
    raise _error(ContentRefusal.CONTENT_KIND_UNDECLARED, path, "content kind is not declared")


def _walk_mapping(value: Mapping[Any, Any], path: tuple[Any, ...], *, frozen: bool, active: set[int], memo: dict[int, Any], held: list[Any]) -> Any:
    identity = id(value)
    if identity in active:
        raise _error(ContentRefusal.CONTENT_CYCLE, path, "content contains an ancestor cycle")
    if identity in memo:
        return memo[identity]
    held.append(value)
    active.add(identity)
    try:
        result: dict[str, Any] = {}
        for entry_index, item in enumerate(value.items()):
            if type(item) not in (tuple, list) or len(item) != 2:
                raise _error(ContentRefusal.MAPPING_ITEM_MALFORMED, path + (("entry", entry_index),), "mapping item is not a pair")
            key, nested = item
            entry_path = path + (("key", key),) if type(key) is str else path + (("entry", entry_index),)
            if type(key) is not str:
                raise _error(ContentRefusal.MAPPING_KEY_NOT_EXACT_STRING, entry_path, "mapping keys must be exact strings")
            _validate_string(key, entry_path)
            if key in result:
                raise _error(ContentRefusal.MAPPING_KEY_REPEATED, entry_path, "mapping key is repeated")
            result[key] = _walk(nested, LITERAL_DECLARATION, path + (("key", key),), frozen=frozen, active=active, memo=memo, held=held)
        output = MappingProxyType(result) if frozen else result
        memo[identity] = output
        return output
    finally:
        active.remove(identity)


def freeze_content(value: Any, declaration: Any = CRITERIA_DECLARATION) -> Any:
    """Validate and detach content into MappingProxyType/tuple containers."""

    _validate_declaration(declaration)
    return _walk(value, declaration, (), frozen=True, active=set(), memo={}, held=[])


def project_content(value: Any, declaration: Any = CRITERIA_DECLARATION) -> Any:
    """Project a frozen declaration into ordinary dict/list JSON containers."""

    _validate_declaration(declaration)
    return _walk(value, declaration, (), frozen=False, active=set(), memo={}, held=[])


__all__ = [
    "CRITERIA_DECLARATION",
    "ContentRefusal",
    "ContentTraversalError",
    "LITERAL_DECLARATION",
    "freeze_content",
    "project_content",
]
