"""Small, dependency-free codec helpers for executor records."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any


def _freeze(value: Any, path: str = "value") -> Any:
    """Copy JSON-shaped values into immutable tuples and mapping proxies."""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not __import__("math").isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} keys must be strings")
        return _FrozenMapping({key: _freeze(item, f"{path}.{key}") for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item, f"{path}[]") for item in value)
    raise TypeError(f"{path} is not JSON-shaped")


class _FrozenMapping(dict[str, Any]):
    """Hash-independent immutable mapping with a useful mapping API."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("mapping is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable


def thaw(value: Any) -> Any:
    """Return a JSON-safe mutable copy of a frozen value."""

    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    """Encode JSON-shaped data with stable keys, separators, and UTF-8."""

    return json.dumps(thaw(_freeze(value)), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    """Return the lowercase SHA-256 digest of canonical JSON bytes."""

    return sha256(canonical_bytes(value)).hexdigest()


canonical_payload_bytes = canonical_bytes
payload_digest = digest


def decode_json(data: bytes | str | Mapping[str, Any]) -> Mapping[str, Any]:
    """Decode a mapping or JSON document, rejecting non-object documents."""

    value = data if isinstance(data, Mapping) else json.loads(data)
    if not isinstance(value, Mapping):
        raise ValueError("record must decode to a mapping")
    return value


def fields(data: Any, expected: set[str], schema: str, name: str) -> Mapping[str, Any]:
    """Validate an exact record key set and schema value."""

    if not isinstance(data, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(data) != expected:
        missing, extra = expected - set(data), set(data) - expected
        raise ValueError(f"{name} keys differ (missing={sorted(missing)!r}, extra={sorted(extra)!r})")
    if type(data["schema"]) is not str or data["schema"] != schema:
        raise ValueError(f"{name} schema is not {schema!r}")
    return data


def text(value: Any, name: str) -> str:
    """Validate a non-empty string."""

    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def sha256_text(value: Any, name: str) -> str:
    """Validate a lowercase SHA-256 hex string."""

    value = text(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def sequence(value: Any, name: str) -> tuple[Any, ...]:
    """Validate and freeze a non-string sequence."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)
