"""Closed, typed mapping validation for execution wire records."""

from collections.abc import Mapping, Sequence
from typing import Any


# Wire entry points intentionally accept Any so decode_fields owns validation.
def decode_fields(
    data: Any,
    required: Mapping[str, type],
    *,
    optional: Mapping[str, type | tuple[type, ...]] | None = None,
    schema: str | None = None,
    name: str = "record",
) -> Mapping[str, Any]:
    """Validate field presence and types without coercing wire values.

    Unknown fields and malformed records raise ``ValueError``. Optional
    defaults remain the caller's responsibility; validation never invents data.
    """

    if not isinstance(data, Mapping):
        raise ValueError(f"{name} must be a mapping")
    fields = {**required, **(optional or {})}
    if schema is not None:
        fields["schema"] = str
    unknown = set(data) - fields.keys()
    if unknown:
        raise ValueError(f"{name} contains unknown keys: {sorted(map(str, unknown))}")
    missing = set(required) - data.keys()
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if schema is not None and data.get("schema") != schema:
        raise ValueError(f"{name} schema is not {schema!r}")
    for key, value in data.items():
        expected = fields[key]
        valid = isinstance(value, expected)
        # bool is an int subclass, and text is a sequence, but neither is the
        # wire representation of an integer counter or an argv/items list.
        if expected in (int, bool):
            valid = type(value) is expected
        elif expected is Sequence:
            valid = valid and not isinstance(value, (str, bytes, bytearray))
        if not valid:
            raise ValueError(f"{name}.{key} has an invalid type")
    return data


def string_argv(value: Any, *, name: str = "argv") -> tuple[str, ...]:
    """Validate a nonempty sequence of nonempty strings without splitting text."""

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(not isinstance(part, str) or not part for part in value)
    ):
        raise ValueError(f"{name} must be a non-empty sequence of non-empty strings")
    return tuple(value)
