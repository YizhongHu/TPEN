"""Shared decorators for immutable executor records."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any, TypeVar, cast


_FROZEN_RECORD_MARKER = "__tpen_frozen_record__"
_Record = TypeVar("_Record")


def _reject_setattr(self: object, name: str, value: Any) -> None:
    """Reject every post-construction attribute assignment."""

    del self, value
    raise FrozenInstanceError(f"cannot assign to field {name!r}")


def _reject_delattr(self: object, name: str) -> None:
    """Reject every post-construction attribute deletion."""

    del self
    raise FrozenInstanceError(f"cannot delete field {name!r}")


def frozen_record(cls: type[_Record]) -> type[_Record]:
    """Create one frozen, slotted dataclass with stable 3.12 semantics.

    ``dataclass(..., slots=True)`` rebuilds the class.  On Python 3.12 the
    generated frozen methods retain a closure over the pre-rebuild class, so
    assigning a non-field can raise a misleading ``TypeError``.  Replacing
    those methods on the rebuilt class keeps field and non-field mutation
    failures uniformly ``FrozenInstanceError`` on supported interpreters.

    Parameters
    ----------
    cls : type
        The record class to decorate.

    Returns
    -------
    type
        The rebuilt frozen and slotted dataclass.
    """

    record = dataclass(frozen=True, slots=True)(cls)
    record.__setattr__ = _reject_setattr  # type: ignore[attr-defined]
    record.__delattr__ = _reject_delattr  # type: ignore[attr-defined]
    setattr(record, _FROZEN_RECORD_MARKER, True)
    return cast(type[_Record], record)


def is_frozen_record(cls: type[object]) -> bool:
    """Return whether ``cls`` was produced by :func:`frozen_record`."""

    return cls.__dict__.get(_FROZEN_RECORD_MARKER) is True
