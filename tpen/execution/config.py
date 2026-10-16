"""Versioned, backend-neutral execution configuration.

The execution configuration is deliberately separate from scientific intent.
It names the operational backend and recovery lego without carrying Hydra
objects or resolving arbitrary science constructors.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from .recovery import RecoveryDecision, _freeze, _thaw

EXECUTION_CONFIG_SCHEMA = "tpen.execution/config-v1"


def _text(value: Any, name: str) -> str:
    """Return a required non-empty text field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest(value: Any, name: str) -> str:
    """Return a lowercase SHA-256 digest field."""

    value = _text(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _reject_unknown_keys(data: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    """Reject fields not defined by this versioned mapping schema."""

    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"{name} contains unknown keys: {names}")


def canonical_payload_bytes(payload: Any) -> bytes:
    """Encode JSON-shaped data canonically for identity and binding."""

    return json.dumps(
        _thaw(_freeze(payload, path="payload")),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_digest(payload: Any) -> str:
    """Return the stable SHA-256 digest of a JSON-shaped payload."""

    return sha256(canonical_payload_bytes(payload)).hexdigest()


def source_config_digest(config: Any) -> str:
    """Return the stable digest of captured source-configuration data."""

    return payload_digest(config)


def recovery_selection_digest(selection: RecoveryDecision) -> str:
    """Return the stable identity of an E1 recovery selection.

    The selection remains the typed :class:`RecoveryDecision` owned by E1;
    E2 stores only its content digest in operational configuration.
    """

    if not isinstance(selection, RecoveryDecision):
        raise TypeError("selection must be an E1 RecoveryDecision")
    return payload_digest(selection.to_dict())


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Operational execution choices independent of scientific YAML.

    Parameters
    ----------
    lineage_id : str
        Opaque identity of the immutable execution lineage.
    backend : str
        Explicit backend lego name.  It is never merged into science input.
    recovery : str
        Explicit recovery lego name, such as ``"checkpoint"`` or ``"none"``.
    source_config_digest : str
        Digest of the captured source configuration syntax.
    payload_digest : str
        Digest of the exact ready-batch payload bound to this lineage.
    backend_options, recovery_options : mapping, optional
        JSON-shaped operational options.  They are kept in their own sections.
    """

    lineage_id: str
    backend: str
    recovery: str
    source_config_digest: str
    payload_digest: str
    recovery_selection_digest: str | None = None
    backend_options: Mapping[str, Any] = field(default_factory=dict)
    recovery_options: Mapping[str, Any] = field(default_factory=dict)
    schema: str = EXECUTION_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, "lineage_id"))
        object.__setattr__(self, "backend", _text(self.backend, "backend"))
        object.__setattr__(self, "recovery", _text(self.recovery, "recovery"))
        object.__setattr__(
            self,
            "source_config_digest",
            _digest(self.source_config_digest, "source_config_digest"),
        )
        object.__setattr__(self, "payload_digest", _digest(self.payload_digest, "payload_digest"))
        if self.recovery_selection_digest is not None:
            object.__setattr__(
                self,
                "recovery_selection_digest",
                _digest(self.recovery_selection_digest, "recovery_selection_digest"),
            )
        if self.schema != EXECUTION_CONFIG_SCHEMA:
            raise ValueError(f"unsupported execution config schema: {self.schema!r}")
        object.__setattr__(self, "backend_options", _freeze(self.backend_options, path="backend_options"))
        object.__setattr__(self, "recovery_options", _freeze(self.recovery_options, path="recovery_options"))

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-safe configuration mapping."""

        return {
            "schema": self.schema,
            "lineage_id": self.lineage_id,
            "backend": self.backend,
            "recovery": self.recovery,
            "source_config_digest": self.source_config_digest,
            "payload_digest": self.payload_digest,
            "recovery_selection_digest": self.recovery_selection_digest,
            "backend_options": _thaw(self.backend_options),
            "recovery_options": _thaw(self.recovery_options),
        }

    def encode(self) -> bytes:
        """Return canonical UTF-8 JSON bytes."""

        return canonical_payload_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExecutionConfig":
        """Construct and validate a configuration from serialized data."""

        if not isinstance(data, Mapping):
            raise TypeError("execution config must be a mapping")
        _reject_unknown_keys(
            data,
            frozenset(
                {
                    "schema",
                    "lineage_id",
                    "backend",
                    "recovery",
                    "source_config_digest",
                    "payload_digest",
                    "recovery_selection_digest",
                    "backend_options",
                    "recovery_options",
                }
            ),
            "execution config",
        )
        return cls(
            schema=data.get("schema", ""),
            lineage_id=data.get("lineage_id", ""),
            backend=data.get("backend", ""),
            recovery=data.get("recovery", ""),
            source_config_digest=data.get("source_config_digest", ""),
            payload_digest=data.get("payload_digest", ""),
            recovery_selection_digest=data.get("recovery_selection_digest"),
            backend_options=data.get("backend_options", {}),
            recovery_options=data.get("recovery_options", {}),
        )

    @classmethod
    def decode(cls, data: bytes | str | Mapping[str, Any]) -> "ExecutionConfig":
        """Decode canonical JSON or a mapping into an execution config."""

        if isinstance(data, Mapping):
            payload = data
        else:
            payload = json.loads(data)
        return cls.from_mapping(payload)

