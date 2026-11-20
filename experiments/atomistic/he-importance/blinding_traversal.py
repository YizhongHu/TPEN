"""Generic structural traversal for study-owned blinding screens.

The walker deliberately owns no token set and makes no admission decision.
Callers provide a predicate for normalized names and leaves, plus an error
factory that preserves the caller's exception type and message policy.  The
L4b declaration freezer is intentionally not a consumer of this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import PurePath
from typing import Any


Predicate = Callable[[str], bool]
ErrorFactory = Callable[[str, Any, str], Exception]


class _Path(str):
    """A string path carrying its explicit parent for error attribution."""

    def __new__(cls, value: str, parent: str) -> "_Path":
        instance = str.__new__(cls, value)
        instance.parent = parent
        return instance


def walk_blinding_content(
    value: Any,
    label: str,
    *,
    is_forbidden: Predicate,
    error_factory: ErrorFactory,
) -> None:
    """Visit names and leaves reachable from ``value``.

    Parameters
    ----------
    value
        Root value to inspect.
    label
        Stable human-readable path label for ``value``.
    is_forbidden
        Caller-owned predicate.  It receives every normalized mapping key,
        dataclass/object attribute name, and supported leaf value.  Returning
        ``True`` refuses that content.
    error_factory
        Caller-owned exception constructor receiving ``(path, content,
        reason)``.  Reasons are ``"key"``, ``"value"``, ``"key-type"``, and
        ``"decode"``.

    Raises
    ------
    Exception
        The exception returned by ``error_factory`` when a predicate refuses
        content or a key/leaf cannot be normalized.

    Notes
    -----
    ``str`` values are passed through unchanged; ``bytes`` and ``bytearray``
    require strict UTF-8; and :class:`~pathlib.PurePath` values are converted
    with :func:`str`.  Lists, tuples, and sets use index-like path components;
    sets are ordered by type and representation to keep labels deterministic.
    """

    _walk(value, label, is_forbidden=is_forbidden, error_factory=error_factory, active=set())


def _walk(
    value: Any,
    label: str,
    *,
    is_forbidden: Predicate,
    error_factory: ErrorFactory,
    active: set[int],
) -> None:
    normalized = _normalize(value, label, error_factory=error_factory)
    if normalized is not None:
        _check(normalized, label, "value", is_forbidden, error_factory)
        return

    if value is None or isinstance(value, (bool, int, float, complex)):
        return

    identity = id(value)
    if identity in active:
        return
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_text = _normalize_key(key, label, error_factory=error_factory)
                _check(
                    key_text,
                    _Path(f"{label}.{key_text}", label),
                    "key",
                    is_forbidden,
                    error_factory,
                )
                _walk(
                    nested,
                    f"{label}.{key_text}",
                    is_forbidden=is_forbidden,
                    error_factory=error_factory,
                    active=active,
                )
            return

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, nested in enumerate(value):
                _walk(
                    nested,
                    f"{label}[{index}]",
                    is_forbidden=is_forbidden,
                    error_factory=error_factory,
                    active=active,
                )
            return

        if isinstance(value, set):
            ordered = sorted(value, key=lambda item: (type(item).__name__, repr(item)))
            for index, nested in enumerate(ordered):
                _walk(
                    nested,
                    f"{label}[{index}]",
                    is_forbidden=is_forbidden,
                    error_factory=error_factory,
                    active=active,
                )
            return

        members = _members(value)
        if members is None:
            return
        for name, nested in members:
            _check(
                name,
                _Path(f"{label}.{name}", label),
                "key",
                is_forbidden,
                error_factory,
            )
            _walk(
                nested,
                f"{label}.{name}",
                is_forbidden=is_forbidden,
                error_factory=error_factory,
                active=active,
            )
    finally:
        active.remove(identity)


def _normalize(value: Any, label: str, *, error_factory: ErrorFactory) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise error_factory(label, value, "decode") from None
    if isinstance(value, PurePath):
        return str(value)
    return None


def _normalize_key(key: Any, label: str, *, error_factory: ErrorFactory) -> str:
    normalized = _normalize(key, _Path(f"{label} key", label), error_factory=error_factory)
    if normalized is None:
        raise error_factory(_Path(f"{label} key", label), key, "key-type")
    return normalized


def _check(
    text: str,
    label: str,
    reason: str,
    is_forbidden: Predicate,
    error_factory: ErrorFactory,
) -> None:
    if is_forbidden(text):
        raise error_factory(label, text, reason)


def _members(value: Any) -> tuple[tuple[str, Any], ...] | None:
    if is_dataclass(value) and not isinstance(value, type):
        return tuple((field.name, getattr(value, field.name)) for field in fields(value))
    members: list[tuple[str, Any]] = []
    if hasattr(value, "__dict__"):
        members.extend(vars(value).items())
    seen_descriptors: set[int] = set()
    for owner in type(value).__mro__:
        declared = owner.__dict__.get("__slots__", ())
        if isinstance(declared, str):
            declared_names = (declared,)
        else:
            declared_names = declared
        for name in declared_names:
            if name in {"__dict__", "__weakref__"}:
                continue
            storage_name = _mangled_slot_name(owner, name)
            descriptor = owner.__dict__.get(storage_name)
            if descriptor is None or not hasattr(descriptor, "__get__"):
                continue
            descriptor_identity = id(descriptor)
            if descriptor_identity in seen_descriptors:
                continue
            seen_descriptors.add(descriptor_identity)
            try:
                nested = descriptor.__get__(value, type(value))
            except AttributeError:
                continue
            members.append((name, nested))
    return tuple(sorted(members, key=lambda member: member[0])) or None


def _mangled_slot_name(owner: type[Any], name: str) -> str:
    """Return the defining class's storage name for a slot declaration."""

    if name.startswith("__") and not name.endswith("__"):
        class_name = owner.__name__.lstrip("_")
        return f"_{class_name}{name}"
    return name


__all__ = ["ErrorFactory", "Predicate", "walk_blinding_content"]
