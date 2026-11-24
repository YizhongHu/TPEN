"""Versioned ``tpen.run-observability/v1`` artifacts, bounded writer, and pure readers.

This module owns the additive sidecar contract introduced for TPEN core
resource observability and profiling:

* Versioned immutable wire records under ``tpen.run-observability/v1``
  (``manifest``, ``inventory``, ``descriptors``, ``sample``, ``status``,
  ``profile_status``, and optional structured ``profile_summary``).
* Pure, Hydra-configurable :class:`RunArtifactSink` and single-writer bounded
  :class:`ObservabilityBundleWriter` routed through
  :class:`tpen.artifacts.ArtifactManager` with strict containment, no-symlink
  enforcement, complete-line byte capping via :func:`tpen.durable_append.append_record`,
  and an independent status/summary byte budget.
* Pure, strictly read-only inspection (:func:`inspect_observability_bundle`,
  :func:`inspect_run_observability`) and pure statistical reduction that keeps
  gauges, lifetime high-water marks, counters, capacities, and unknown-window
  vendor busy observations semantically separate.

This module is strictly Torch-free and imports nothing from ``experiments`` or
``tpen_executor``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from tpen.artifacts import ArtifactRoutingError
from tpen.durable_append import append_record, ends_without_newline

if TYPE_CHECKING:
    from tpen.artifacts import ArtifactManager
    from tpen.callback.base import ResourceOwnershipClaim

RUN_OBSERVABILITY_SCHEMA = "tpen.run-observability/v1"
RUN_OBSERVABILITY_SCHEMA_PREFIX = "tpen.run-observability/"

OBSERVABILITY_RECORD_KINDS: frozenset[str] = frozenset(
    {
        "manifest",
        "inventory",
        "descriptors",
        "sample",
        "status",
        "profile_status",
    }
)
PROFILE_SUMMARY_RECORD_KIND = "profile_summary"
OBSERVABILITY_ALL_RECORD_KINDS: frozenset[str] = (
    OBSERVABILITY_RECORD_KINDS | {PROFILE_SUMMARY_RECORD_KIND}
)

OBSERVABILITY_LIFECYCLE_STATES: frozenset[str] = frozenset({"open", "closed", "unsafe"})
OBSERVABILITY_DATA_COMPLETENESS: frozenset[str] = frozenset(
    {"unknown", "none", "partial", "complete"}
)
OBSERVABILITY_READING_STATUSES: frozenset[str] = frozenset(
    {"ok", "unavailable", "error"}
)
OBSERVABILITY_SUBJECT_KINDS: frozenset[str] = frozenset(
    {
        "process",
        "process_allocator",
        "physical_gpu",
        "logical_device",
        "descendant_set",
        "cgroup",
    }
)
OBSERVABILITY_STATISTICS: frozenset[str] = frozenset(
    {"counter", "gauge", "high_water", "capacity", "vendor_window_mean"}
)
OBSERVABILITY_WINDOWS: frozenset[str] = frozenset(
    {
        "process_lifetime",
        "read_interval",
        "inventory_or_read_interval",
        "vendor_recent_window_unknown_bounds",
        "allocator_reset_window",
        "cgroup_lifetime",
        "cgroup_read_interval",
        "scan_interval",
        "unknown",
    }
)
OBSERVABILITY_PROFILE_WINDOW_KINDS: frozenset[str] = frozenset({"operation", "elapsed"})

MANIFEST_FILENAME = "manifest.json"
INVENTORY_FILENAME = "inventory.json"
DESCRIPTORS_FILENAME = "descriptors.json"
# Legacy tpen.inspection._read_status scans ``run_dir.rglob("status.json")`` and
# refuses the whole run with "multiple-status" if any nested file is literally
# named "status.json". Core observability therefore writes profile window status
# records to ``profile-status.json`` and forbids ``observer_id == "status"``.
PROFILE_STATUS_FILENAME = "profile-status.json"
LEGACY_PROFILE_STATUS_FILENAME = "status.json"
PROFILE_SUMMARY_FILENAME = "summary.json"

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_OBSERVER_IDS: frozenset[str] = frozenset({"status"})
_ARTIFACT_REF_KINDS: frozenset[str] = OBSERVABILITY_ALL_RECORD_KINDS | {"trace"}


class ObservabilityArtifactError(RuntimeError):
    """Base error for ``tpen.run-observability/v1`` artifact operations."""


class ObservabilitySchemaError(ObservabilityArtifactError, ValueError):
    """Raised when an observability record violates the v1 schema contract."""


class ObservabilityPathError(ObservabilityArtifactError, ArtifactRoutingError):
    """Raised when an observability identifier or path violates containment."""


# ---------------------------------------------------------------------------
# Shared validation, JSON-safety, and path-containment helpers
# ---------------------------------------------------------------------------


def _reject_json_constant(token: str) -> Any:
    """Reject non-finite JSON constants such as ``NaN`` and ``Infinity``."""

    raise ObservabilitySchemaError(
        f"non-finite JSON constant {token!r} is forbidden in observability artifacts"
    )


def _loads_strict_json(text: str, *, context: str) -> Any:
    """Parse JSON while strictly rejecting ``NaN`` and ``Infinity``."""

    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except ObservabilitySchemaError:
        raise
    except ValueError as exc:
        raise ObservabilitySchemaError(f"invalid JSON in {context}: {exc}") from exc


def _dumps_canonical_json(data: Mapping[str, Any], *, indent: int | None = 2) -> str:
    """Serialize a mapping to strict ASCII JSON forbidding non-finite floats."""

    try:
        if indent is None:
            return json.dumps(
                data,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=True,
            )
        return (
            json.dumps(
                data,
                indent=indent,
                sort_keys=True,
                allow_nan=False,
                ensure_ascii=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise ObservabilitySchemaError(
            f"failed to serialize observability payload to strict ASCII JSON: {exc}"
        ) from exc


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ObservabilitySchemaError(
            f"{name} must be a mapping, got {type(value).__name__}"
        )
    return value


def _require_keys(
    mapping: Mapping[str, Any],
    *,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] = frozenset(),
    context: str,
) -> None:
    keys = set(mapping)
    missing = sorted(required - keys)
    extra = sorted(keys - (set(required) | set(optional)))
    if missing or extra:
        raise ObservabilitySchemaError(
            f"{context} keys mismatch: missing={missing}, extra={extra}"
        )


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ObservabilitySchemaError(
            f"{name} must be a bool, got {type(value).__name__}"
        )
    return value


def _require_int(
    value: object,
    name: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    # Reject bool explicitly because isinstance(True, int) is True in Python.
    if type(value) is not int:
        raise ObservabilitySchemaError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    if min_value is not None and value < min_value:
        raise ObservabilitySchemaError(
            f"{name} must be >= {min_value}, got {value}"
        )
    if max_value is not None and value > max_value:
        raise ObservabilitySchemaError(
            f"{name} must be <= {max_value}, got {value}"
        )
    return value


def _require_finite_number(
    value: object,
    name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    strictly_positive: bool = False,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservabilitySchemaError(
            f"{name} must be a finite number (int or float, not bool), "
            f"got {type(value).__name__}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ObservabilitySchemaError(f"{name} must be finite, got {value!r}")
    if strictly_positive and value <= 0:
        raise ObservabilitySchemaError(f"{name} must be > 0, got {value!r}")
    if min_value is not None and value < min_value:
        raise ObservabilitySchemaError(
            f"{name} must be >= {min_value}, got {value!r}"
        )
    if max_value is not None and value > max_value:
        raise ObservabilitySchemaError(
            f"{name} must be <= {max_value}, got {value!r}"
        )
    return value


def _require_nonempty_str(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ObservabilitySchemaError(f"{name} must be a nonempty string")
    return value


def _require_choice(value: object, choices: frozenset[str], name: str) -> str:
    text = _require_nonempty_str(value, name)
    if text not in choices:
        raise ObservabilitySchemaError(
            f"unsupported {name} {text!r}; expected one of {sorted(choices)}"
        )
    return text


def _require_schema(value: object, context: str) -> str:
    text = _require_nonempty_str(value, f"{context}.schema")
    if text != RUN_OBSERVABILITY_SCHEMA:
        raise ObservabilitySchemaError(
            f"unsupported {context} schema {text!r}; expected {RUN_OBSERVABILITY_SCHEMA!r}"
        )
    return text


def _require_kind(value: object, expected: str, context: str) -> str:
    text = _require_nonempty_str(value, f"{context}.kind")
    if text != expected:
        raise ObservabilitySchemaError(
            f"unsupported {context} kind {text!r}; expected {expected!r}"
        )
    return text


def _require_sha256_hex(value: object, name: str) -> str:
    if type(value) is not str or not _SHA256_HEX_RE.match(value):
        raise ObservabilitySchemaError(
            f"{name} must be a 64-character lowercase hexadecimal SHA-256 digest"
        )
    return value


def _has_control_chars(text: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in text)


def validate_identifier(
    value: object,
    name: str,
    *,
    forbid_status_collision: bool = False,
) -> str:
    """Validate a safe observability path-segment identifier.

    Parameters
    ----------
    value : object
        Candidate identifier value.
    name : str
        Field name for error reporting.
    forbid_status_collision : bool, optional
        When ``True``, reject ``"status"`` (case-insensitively) so
        ``status/<observer-id>.json`` can never collide with legacy
        ``status.json`` discovery in :func:`tpen.inspection.inspect_run`.

    Returns
    -------
    str
        Validated identifier string.

    Raises
    ------
    ObservabilityPathError
        If ``value`` is empty, contains unsafe path characters/control bytes,
        traversal segments, or collides with reserved ``"status"``.
    """

    if type(value) is not str or not value or _has_control_chars(value):
        raise ObservabilityPathError(
            f"{name} must be a safe non-empty ASCII identifier without control characters"
        )
    if (
        "/" in value
        or "\\" in value
        or ".." in value
        or value in {".", ".."}
        or not _SAFE_IDENTIFIER_RE.match(value)
    ):
        raise ObservabilityPathError(
            f"unsafe {name} {value!r}; must match {_SAFE_IDENTIFIER_RE.pattern} "
            "without path separators or traversal segments"
        )
    if forbid_status_collision and value.lower() in _FORBIDDEN_OBSERVER_IDS:
        raise ObservabilityPathError(
            f"{name} {value!r} is forbidden because status/{value}.json collides "
            "with legacy run status.json discovery"
        )
    return value


def validate_relative_path(
    value: object,
    name: str,
    *,
    forbid_manifest_self: bool = False,
) -> str:
    """Validate a safe POSIX relative path with no traversal or escape.

    Parameters
    ----------
    value : object
        Candidate relative path string.
    name : str
        Field name for diagnostics.
    forbid_manifest_self : bool, optional
        When ``True``, reject ``"manifest.json"`` so a manifest never
        recursively hashes itself.

    Returns
    -------
    str
        Validated normalized POSIX relative path.
    """

    if type(value) is not str or not value or value != value.strip():
        raise ObservabilityPathError(
            f"{name} must be a nonempty relative path without leading/trailing whitespace"
        )
    if _has_control_chars(value) or "\\" in value:
        raise ObservabilityPathError(
            f"{name} {value!r} contains forbidden control characters or backslashes"
        )
    if value.startswith("/") or value.startswith("~"):
        raise ObservabilityPathError(
            f"{name} {value!r} must be a relative path, not absolute or home-expanded"
        )
    segments = value.split("/")
    if any(seg in {"", ".", ".."} for seg in segments):
        raise ObservabilityPathError(
            f"{name} {value!r} contains empty, current-dir, or parent-traversal segments"
        )
    if ":" in segments[0]:
        raise ObservabilityPathError(
            f"{name} {value!r} contains a forbidden drive or scheme prefix"
        )
    if Path(value).is_absolute():
        raise ObservabilityPathError(f"{name} {value!r} must be relative")
    normalized = PurePosixPath(*segments).as_posix()
    if forbid_manifest_self and normalized == MANIFEST_FILENAME:
        raise ObservabilityPathError(
            "manifest.artifacts must not reference manifest.json recursively"
        )
    return normalized


def _validate_and_copy_json(value: Any, path: str) -> Any:
    """Validate strict JSON safety (forbidding NaN/Infinity) and deep-copy."""

    if value is None or type(value) is bool or type(value) is str or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ObservabilitySchemaError(
                f"{path} contains non-finite float {value!r}"
            )
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ObservabilitySchemaError(
                    f"{path} mapping keys must be strings, got {type(key).__name__}"
                )
            copied[key] = _validate_and_copy_json(item, f"{path}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return [
            _validate_and_copy_json(item, f"{path}[{idx}]")
            for idx, item in enumerate(value)
        ]
    raise ObservabilitySchemaError(
        f"{path} contains non-JSON-safe value of type {type(value).__name__}"
    )


def _validate_tagged_fact(value: object, name: str) -> dict[str, Any]:
    """Validate a tagged fact dict (``ok``/``unavailable``/``error``) used in metadata."""

    mapping = _require_mapping(value, name)
    status = _require_choice(mapping.get("status"), OBSERVABILITY_READING_STATUSES, f"{name}.status")
    if status == "ok":
        _require_keys(mapping, required={"status", "value"}, context=name)
        return {"status": "ok", "value": _validate_and_copy_json(mapping["value"], f"{name}.value")}
    if "value" in mapping:
        raise ObservabilitySchemaError(
            f"{name} with status {status!r} must not carry a 'value' key"
        )
    _require_keys(mapping, required={"status", "reason_code"}, optional={"detail"}, context=name)
    out: dict[str, Any] = {
        "status": status,
        "reason_code": _require_nonempty_str(mapping["reason_code"], f"{name}.reason_code"),
    }
    if "detail" in mapping:
        out["detail"] = _require_nonempty_str(mapping["detail"], f"{name}.detail")
    return out


def _validate_ns_pair(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ObservabilitySchemaError(f"{name} must be a 2-element [start_ns, end_ns] sequence")
    start = _require_int(value[0], f"{name}[0]", min_value=0)
    end = _require_int(value[1], f"{name}[1]", min_value=0)
    if end < start:
        raise ObservabilitySchemaError(
            f"{name} end ({end}) must be >= start ({start})"
        )
    return (start, end)


def _validate_str_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ObservabilitySchemaError(
            f"{name} must be a sequence of strings, got {type(value).__name__}"
        )
    return tuple(
        _require_nonempty_str(item, f"{name}[{idx}]")
        for idx, item in enumerate(value)
    )


def _check_no_symlinks_under(root: Path, relative_parts: Sequence[str]) -> Path:
    """Verify neither ``root`` nor any component under it is a symlink and stay contained.

    Checks ``root.is_symlink()`` and every intermediate/leaf component along
    ``relative_parts`` without flagging OS-level ancestors above ``root``
    (such as ``/var`` -> ``/private/var`` on macOS).
    """

    if root.is_symlink():
        raise ObservabilityPathError(
            f"refusing symlinked observability root {root!s}"
        )
    if root.exists() and not root.is_dir():
        raise ObservabilityPathError(
            f"observability root {root!s} exists and is not a directory"
        )
    current = root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise ObservabilityPathError(
                f"refusing symlinked observability path {current!s}"
            )
    resolved_root = root.resolve(strict=False)
    resolved_target = current.resolve(strict=False)
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ObservabilityPathError(
            f"observability path {current!s} resolves to {resolved_target!s}, "
            f"outside contained root {resolved_root!s}"
        ) from exc
    return current


def _freeze_key(value: Any) -> Any:
    """Convert a JSON-safe structure into a deterministic hashable key."""

    if isinstance(value, Mapping):
        return tuple(
            (key, _freeze_key(val))
            for key, val in sorted(value.items(), key=lambda pair: pair[0])
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_key(item) for item in value)
    return value


def _subject_generation_key(subject: Mapping[str, Any]) -> Any:
    """Return a hashable subject-identity key used to detect generation shifts."""

    filtered = {
        key: val
        for key, val in subject.items()
        if key not in {"mapping_evidence", "evidence"}
    }
    return _freeze_key(filtered)


# ---------------------------------------------------------------------------
# Versioned immutable wire dataclasses (tpen.run-observability/v1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservabilityReading:
    """Tagged reading variant for one observed quantity in a sample packet.

    Parameters
    ----------
    status : {"ok", "unavailable", "error"}
        Explicit reading state. ``"ok"`` requires a finite numeric ``value``;
        ``"unavailable"`` and ``"error"`` carry a ``reason_code`` (and optional
        bounded ``detail``) and must never carry a ``value``.
    value : int or float or None, optional
        Finite numeric observation when ``status == "ok"``. Booleans, ``NaN``,
        and infinities are rejected.
    reason_code : str or None, optional
        Machine-readable unavailability/error code when ``status != "ok"``.
    detail : str or None, optional
        Optional bounded diagnostic message when ``status != "ok"``.
    """

    status: str
    value: int | float | None = None
    reason_code: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        status = _require_choice(
            self.status, OBSERVABILITY_READING_STATUSES, "reading.status"
        )
        if status == "ok":
            _require_finite_number(self.value, "reading.value")
            if self.reason_code is not None or self.detail is not None:
                raise ObservabilitySchemaError(
                    "reading with status 'ok' must not include reason_code or detail"
                )
        else:
            if self.value is not None:
                raise ObservabilitySchemaError(
                    f"reading with status {status!r} must not carry a numeric value"
                )
            _require_nonempty_str(self.reason_code, "reading.reason_code")
            if self.detail is not None:
                _require_nonempty_str(self.detail, "reading.detail")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe representation of this reading."""

        if self.status == "ok":
            return {"status": "ok", "value": self.value}
        data: dict[str, Any] = {
            "status": self.status,
            "reason_code": self.reason_code,
        }
        if self.detail is not None:
            data["detail"] = self.detail
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ObservabilityReading:
        """Validate and construct an :class:`ObservabilityReading` from a mapping."""

        mapping = _require_mapping(data, "reading")
        status = _require_choice(
            mapping.get("status"), OBSERVABILITY_READING_STATUSES, "reading.status"
        )
        if status == "ok":
            _require_keys(mapping, required={"status", "value"}, context="reading")
            return cls(status="ok", value=mapping["value"])
        if "value" in mapping:
            raise ObservabilitySchemaError(
                f"reading with status {status!r} must not contain a 'value' key"
            )
        _require_keys(
            mapping,
            required={"status", "reason_code"},
            optional={"detail"},
            context="reading",
        )
        return cls(
            status=status,
            reason_code=mapping["reason_code"],
            detail=mapping.get("detail"),
        )


@dataclass(frozen=True)
class QuantityDescriptor:
    """Immutable semantic declaration for one observable quantity.

    Parameters
    ----------
    source_id : str
        Identifier of the acquisition source producing this quantity.
    unit : str
        Canonical unit string (for example ``"s"``, ``"B"``, ``"1"``).
    statistic : str
        Closed statistical kind (``counter``, ``gauge``, ``high_water``,
        ``capacity``, ``vendor_window_mean``).
    subject_kind : str
        Closed subject scope kind (``process``, ``process_allocator``,
        ``physical_gpu``, ``logical_device``, ``descendant_set``, ``cgroup``).
    window : str
        Declared observation window semantics.
    native_name : str
        Underlying OS/runtime/vendor metric name.
    native_unit : str
        Native unit prior to conversion.
    conversion : str
        Explicit conversion rule from native unit to canonical unit.
    quality : tuple of str, optional
        Optional quality qualifiers such as ``("approximate_rss",)``.
    """

    source_id: str
    unit: str
    statistic: str
    subject_kind: str
    window: str
    native_name: str
    native_unit: str
    conversion: str
    quality: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.source_id, "descriptor.source_id")
        _require_nonempty_str(self.unit, "descriptor.unit")
        _require_choice(self.statistic, OBSERVABILITY_STATISTICS, "descriptor.statistic")
        _require_choice(
            self.subject_kind, OBSERVABILITY_SUBJECT_KINDS, "descriptor.subject_kind"
        )
        _require_choice(self.window, OBSERVABILITY_WINDOWS, "descriptor.window")
        _require_nonempty_str(self.native_name, "descriptor.native_name")
        _require_nonempty_str(self.native_unit, "descriptor.native_unit")
        _require_nonempty_str(self.conversion, "descriptor.conversion")
        object.__setattr__(
            self, "quality", _validate_str_tuple(self.quality, "descriptor.quality")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary matching the v1 descriptor wire."""

        data: dict[str, Any] = {
            "source_id": self.source_id,
            "unit": self.unit,
            "statistic": self.statistic,
            "subject_kind": self.subject_kind,
            "window": self.window,
            "native_name": self.native_name,
            "native_unit": self.native_unit,
            "conversion": self.conversion,
        }
        if self.quality:
            data["quality"] = list(self.quality)
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> QuantityDescriptor:
        """Validate and construct a :class:`QuantityDescriptor` from a mapping."""

        mapping = _require_mapping(data, "descriptor")
        _require_keys(
            mapping,
            required={
                "source_id",
                "unit",
                "statistic",
                "subject_kind",
                "window",
                "native_name",
                "native_unit",
                "conversion",
            },
            optional={"quality"},
            context="descriptor",
        )
        return cls(
            source_id=mapping["source_id"],
            unit=mapping["unit"],
            statistic=mapping["statistic"],
            subject_kind=mapping["subject_kind"],
            window=mapping["window"],
            native_name=mapping["native_name"],
            native_unit=mapping["native_unit"],
            conversion=mapping["conversion"],
            quality=tuple(mapping.get("quality", ())),
        )


@dataclass(frozen=True)
class ObservabilityArtifactRef:
    """Reference to one bundle-relative artifact inside ``manifest.json``.

    Parameters
    ----------
    path : str
        Safe bundle-relative POSIX path (never ``"manifest.json"``).
    bytes : int or None, optional
        Exact file byte length once closed/finalized.
    sha256 : str or None, optional
        Lowercase 64-hex SHA-256 digest once closed/finalized.
    kind : str or None, optional
        Optional artifact kind label.
    """

    path: str
    bytes: int | None = None
    sha256: str | None = None
    kind: str | None = None

    def __post_init__(self) -> None:
        normalized = validate_relative_path(
            self.path, "artifact.path", forbid_manifest_self=True
        )
        object.__setattr__(self, "path", normalized)
        if (self.bytes is None) != (self.sha256 is None):
            raise ObservabilitySchemaError(
                "artifact ref must supply both bytes and sha256 or neither"
            )
        if self.bytes is not None:
            _require_int(self.bytes, "artifact.bytes", min_value=0)
        if self.sha256 is not None:
            _require_sha256_hex(self.sha256, "artifact.sha256")
        if self.kind is not None:
            _require_choice(self.kind, _ARTIFACT_REF_KINDS, "artifact.kind")

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary for this artifact reference."""

        data: dict[str, Any] = {"path": self.path}
        if self.bytes is not None and self.sha256 is not None:
            data["bytes"] = self.bytes
            data["sha256"] = self.sha256
        if self.kind is not None:
            data["kind"] = self.kind
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ObservabilityArtifactRef:
        """Validate and construct an :class:`ObservabilityArtifactRef`."""

        mapping = _require_mapping(data, "artifact")
        _require_keys(
            mapping,
            required={"path"},
            optional={"bytes", "sha256", "kind"},
            context="artifact",
        )
        return cls(
            path=mapping["path"],
            bytes=mapping.get("bytes"),
            sha256=mapping.get("sha256"),
            kind=mapping.get("kind"),
        )


@dataclass(frozen=True)
class ObservabilityManifest:
    """Top-level ``manifest.json`` record for one observability invocation."""

    invocation_id: str
    lifecycle_state: str
    data_completeness: str
    run_id: str
    producer: Mapping[str, Any]
    configuration_digest: str
    inventory_id: str
    descriptor_set_id: str
    observers: tuple[Mapping[str, Any], ...]
    limits: Mapping[str, Any]
    executor_context: str | Mapping[str, Any] = "not_supplied"
    scheduler_accounting: str | Mapping[str, Any] = "not_supplied"
    artifacts: tuple[ObservabilityArtifactRef, ...] = ()
    schema: str = RUN_OBSERVABILITY_SCHEMA
    kind: str = "manifest"

    def __post_init__(self) -> None:
        _require_schema(self.schema, "manifest")
        _require_kind(self.kind, "manifest", "manifest")
        validate_identifier(self.invocation_id, "manifest.invocation_id")
        _require_choice(
            self.lifecycle_state,
            OBSERVABILITY_LIFECYCLE_STATES,
            "manifest.lifecycle_state",
        )
        _require_choice(
            self.data_completeness,
            OBSERVABILITY_DATA_COMPLETENESS,
            "manifest.data_completeness",
        )
        _require_nonempty_str(self.run_id, "manifest.run_id")
        producer = _validate_and_copy_json(
            _require_mapping(self.producer, "manifest.producer"), "manifest.producer"
        )
        if not producer:
            raise ObservabilitySchemaError("manifest.producer must be a nonempty mapping")
        object.__setattr__(self, "producer", producer)
        _require_sha256_hex(
            self.configuration_digest, "manifest.configuration_digest"
        )
        validate_identifier(self.inventory_id, "manifest.inventory_id")
        validate_identifier(self.descriptor_set_id, "manifest.descriptor_set_id")

        if not isinstance(self.observers, (list, tuple)):
            raise ObservabilitySchemaError("manifest.observers must be a sequence")
        validated_observers: list[dict[str, Any]] = []
        seen_observers: set[str] = set()
        for idx, entry in enumerate(self.observers):
            obs = _validate_and_copy_json(
                _require_mapping(entry, f"manifest.observers[{idx}]"),
                f"manifest.observers[{idx}]",
            )
            observer_id = validate_identifier(
                obs.get("observer_id"),
                f"manifest.observers[{idx}].observer_id",
                forbid_status_collision=True,
            )
            if observer_id in seen_observers:
                raise ObservabilitySchemaError(
                    f"duplicate observer_id {observer_id!r} in manifest.observers"
                )
            seen_observers.add(observer_id)
            if "callback_position" in obs:
                _require_int(
                    obs["callback_position"],
                    f"manifest.observers[{idx}].callback_position",
                    min_value=0,
                )
            if "callback_type" in obs:
                _require_nonempty_str(
                    obs["callback_type"], f"manifest.observers[{idx}].callback_type"
                )
            validated_observers.append(obs)
        object.__setattr__(self, "observers", tuple(validated_observers))

        limits = _validate_and_copy_json(
            _require_mapping(self.limits, "manifest.limits"), "manifest.limits"
        )
        object.__setattr__(self, "limits", limits)

        for field_name in ("executor_context", "scheduler_accounting"):
            val = getattr(self, field_name)
            if val == "not_supplied":
                continue
            if isinstance(val, Mapping):
                object.__setattr__(
                    self,
                    field_name,
                    _validate_and_copy_json(val, f"manifest.{field_name}"),
                )
            else:
                raise ObservabilitySchemaError(
                    f"manifest.{field_name} must be 'not_supplied' or a mapping"
                )

        if not isinstance(self.artifacts, (list, tuple)):
            raise ObservabilitySchemaError("manifest.artifacts must be a sequence")
        refs: list[ObservabilityArtifactRef] = []
        seen_paths: set[str] = set()
        for idx, item in enumerate(self.artifacts):
            ref = (
                item
                if isinstance(item, ObservabilityArtifactRef)
                else ObservabilityArtifactRef.from_mapping(item)
            )
            if ref.path in seen_paths:
                raise ObservabilitySchemaError(
                    f"duplicate artifact path {ref.path!r} in manifest.artifacts"
                )
            seen_paths.add(ref.path)
            if self.lifecycle_state == "closed" and (
                ref.bytes is None or ref.sha256 is None
            ):
                raise ObservabilitySchemaError(
                    f"closed manifest artifact {ref.path!r} must include bytes and sha256"
                )
            refs.append(ref)
        object.__setattr__(self, "artifacts", tuple(refs))

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary matching the v1 manifest wire."""

        return {
            "schema": self.schema,
            "kind": self.kind,
            "invocation_id": self.invocation_id,
            "lifecycle_state": self.lifecycle_state,
            "data_completeness": self.data_completeness,
            "run_id": self.run_id,
            "producer": _validate_and_copy_json(self.producer, "manifest.producer"),
            "configuration_digest": self.configuration_digest,
            "inventory_id": self.inventory_id,
            "descriptor_set_id": self.descriptor_set_id,
            "observers": [
                _validate_and_copy_json(obs, "manifest.observers")
                for obs in self.observers
            ],
            "limits": _validate_and_copy_json(self.limits, "manifest.limits"),
            "executor_context": (
                self.executor_context
                if isinstance(self.executor_context, str)
                else _validate_and_copy_json(
                    self.executor_context, "manifest.executor_context"
                )
            ),
            "scheduler_accounting": (
                self.scheduler_accounting
                if isinstance(self.scheduler_accounting, str)
                else _validate_and_copy_json(
                    self.scheduler_accounting, "manifest.scheduler_accounting"
                )
            ),
            "artifacts": [ref.to_dict() for ref in self.artifacts],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ObservabilityManifest:
        """Validate and construct an :class:`ObservabilityManifest`."""

        mapping = _require_mapping(data, "manifest")
        _require_keys(
            mapping,
            required={
                "schema",
                "kind",
                "invocation_id",
                "lifecycle_state",
                "data_completeness",
                "run_id",
                "producer",
                "configuration_digest",
                "inventory_id",
                "descriptor_set_id",
                "observers",
                "limits",
                "executor_context",
                "scheduler_accounting",
                "artifacts",
            },
            context="manifest",
        )
        return cls(
            schema=mapping["schema"],
            kind=mapping["kind"],
            invocation_id=mapping["invocation_id"],
            lifecycle_state=mapping["lifecycle_state"],
            data_completeness=mapping["data_completeness"],
            run_id=mapping["run_id"],
            producer=mapping["producer"],
            configuration_digest=mapping["configuration_digest"],
            inventory_id=mapping["inventory_id"],
            descriptor_set_id=mapping["descriptor_set_id"],
            observers=tuple(mapping["observers"])
            if isinstance(mapping["observers"], (list, tuple))
            else mapping["observers"],
            limits=mapping["limits"],
            executor_context=mapping["executor_context"],
            scheduler_accounting=mapping["scheduler_accounting"],
            artifacts=tuple(
                ObservabilityArtifactRef.from_mapping(item)
                for item in mapping["artifacts"]
            )
            if isinstance(mapping["artifacts"], (list, tuple))
            else mapping["artifacts"],
        )


@dataclass(frozen=True)
class ObservabilityInventory:
    """Immutable ``inventory.json`` host, clock, hardware, and source snapshot."""

    invocation_id: str
    inventory_id: str
    host: Mapping[str, Any]
    process: Mapping[str, Any]
    clock_domains: tuple[Mapping[str, Any], ...]
    requested: Mapping[str, Any]
    observed: Mapping[str, Any]
    sources: Mapping[str, Mapping[str, Any]]
    executor_context: str | Mapping[str, Any] = field(
        default_factory=lambda: {"status": "unavailable", "reason_code": "not_supplied"}
    )
    scheduler_allocation: str | Mapping[str, Any] = field(
        default_factory=lambda: {"status": "unavailable", "reason_code": "not_supplied"}
    )
    billed_units: str | Mapping[str, Any] = field(
        default_factory=lambda: {"status": "unavailable", "reason_code": "not_supplied"}
    )
    schema: str = RUN_OBSERVABILITY_SCHEMA
    kind: str = "inventory"

    def __post_init__(self) -> None:
        _require_schema(self.schema, "inventory")
        _require_kind(self.kind, "inventory", "inventory")
        validate_identifier(self.invocation_id, "inventory.invocation_id")
        validate_identifier(self.inventory_id, "inventory.inventory_id")

        host = _validate_and_copy_json(
            _require_mapping(self.host, "inventory.host"), "inventory.host"
        )
        _require_nonempty_str(host.get("host_id"), "inventory.host.host_id")
        object.__setattr__(self, "host", host)

        process = _validate_and_copy_json(
            _require_mapping(self.process, "inventory.process"), "inventory.process"
        )
        _require_choice(
            process.get("kind"), frozenset({"process"}), "inventory.process.kind"
        )
        object.__setattr__(self, "process", process)

        if not isinstance(self.clock_domains, (list, tuple)) or not self.clock_domains:
            raise ObservabilitySchemaError(
                "inventory.clock_domains must be a nonempty sequence"
            )
        domains: list[dict[str, Any]] = []
        seen_domains: set[str] = set()
        for idx, item in enumerate(self.clock_domains):
            dom = _validate_and_copy_json(
                _require_mapping(item, f"inventory.clock_domains[{idx}]"),
                f"inventory.clock_domains[{idx}]",
            )
            dom_id = _require_nonempty_str(
                dom.get("clock_domain_id"),
                f"inventory.clock_domains[{idx}].clock_domain_id",
            )
            if dom_id in seen_domains:
                raise ObservabilitySchemaError(
                    f"duplicate clock_domain_id {dom_id!r} in inventory"
                )
            seen_domains.add(dom_id)
            _require_nonempty_str(dom.get("kind"), f"inventory.clock_domains[{idx}].kind")
            _require_int(
                dom.get("anchor_monotonic_ns"),
                f"inventory.clock_domains[{idx}].anchor_monotonic_ns",
                min_value=0,
            )
            if "anchor_read_bracket_ns" in dom:
                _validate_ns_pair(
                    dom["anchor_read_bracket_ns"],
                    f"inventory.clock_domains[{idx}].anchor_read_bracket_ns",
                )
            _require_nonempty_str(
                dom.get("anchor_utc"), f"inventory.clock_domains[{idx}].anchor_utc"
            )
            _require_nonempty_str(
                dom.get("run_timezone"), f"inventory.clock_domains[{idx}].run_timezone"
            )
            domains.append(dom)
        object.__setattr__(self, "clock_domains", tuple(domains))

        object.__setattr__(
            self,
            "requested",
            _validate_and_copy_json(
                _require_mapping(self.requested, "inventory.requested"),
                "inventory.requested",
            ),
        )
        object.__setattr__(
            self,
            "observed",
            _validate_and_copy_json(
                _require_mapping(self.observed, "inventory.observed"),
                "inventory.observed",
            ),
        )

        raw_sources = _require_mapping(self.sources, "inventory.sources")
        validated_sources: dict[str, dict[str, Any]] = {}
        for source_id, meta in raw_sources.items():
            validate_identifier(source_id, "inventory.sources key")
            validated_sources[source_id] = _validate_and_copy_json(
                _require_mapping(meta, f"inventory.sources.{source_id}"),
                f"inventory.sources.{source_id}",
            )
        object.__setattr__(self, "sources", validated_sources)

        for field_name in ("executor_context", "scheduler_allocation", "billed_units"):
            val = getattr(self, field_name)
            if val == "not_supplied":
                continue
            if isinstance(val, Mapping):
                if "status" in val:
                    object.__setattr__(
                        self,
                        field_name,
                        _validate_tagged_fact(val, f"inventory.{field_name}"),
                    )
                else:
                    object.__setattr__(
                        self,
                        field_name,
                        _validate_and_copy_json(val, f"inventory.{field_name}"),
                    )
            else:
                raise ObservabilitySchemaError(
                    f"inventory.{field_name} must be 'not_supplied' or a mapping"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary matching the v1 inventory wire."""

        return {
            "schema": self.schema,
            "kind": self.kind,
            "invocation_id": self.invocation_id,
            "inventory_id": self.inventory_id,
            "host": _validate_and_copy_json(self.host, "inventory.host"),
            "process": _validate_and_copy_json(self.process, "inventory.process"),
            "clock_domains": [
                _validate_and_copy_json(dom, "inventory.clock_domains")
                for dom in self.clock_domains
            ],
            "requested": _validate_and_copy_json(self.requested, "inventory.requested"),
            "observed": _validate_and_copy_json(self.observed, "inventory.observed"),
            "sources": _validate_and_copy_json(self.sources, "inventory.sources"),
            "executor_context": (
                self.executor_context
                if isinstance(self.executor_context, str)
                else _validate_and_copy_json(
                    self.executor_context, "inventory.executor_context"
                )
            ),
            "scheduler_allocation": (
                self.scheduler_allocation
                if isinstance(self.scheduler_allocation, str)
                else _validate_and_copy_json(
                    self.scheduler_allocation, "inventory.scheduler_allocation"
                )
            ),
            "billed_units": (
                self.billed_units
                if isinstance(self.billed_units, str)
                else _validate_and_copy_json(
                    self.billed_units, "inventory.billed_units"
                )
            ),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ObservabilityInventory:
        """Validate and construct an :class:`ObservabilityInventory`."""

        mapping = _require_mapping(data, "inventory")
        _require_keys(
            mapping,
            required={
                "schema",
                "kind",
                "invocation_id",
                "inventory_id",
                "host",
                "process",
                "clock_domains",
                "requested",
                "observed",
                "sources",
                "executor_context",
                "scheduler_allocation",
                "billed_units",
            },
            context="inventory",
        )
        return cls(
            schema=mapping["schema"],
            kind=mapping["kind"],
            invocation_id=mapping["invocation_id"],
            inventory_id=mapping["inventory_id"],
            host=mapping["host"],
            process=mapping["process"],
            clock_domains=tuple(mapping["clock_domains"])
            if isinstance(mapping["clock_domains"], (list, tuple))
            else mapping["clock_domains"],
            requested=mapping["requested"],
            observed=mapping["observed"],
            sources=mapping["sources"],
            executor_context=mapping["executor_context"],
            scheduler_allocation=mapping["scheduler_allocation"],
            billed_units=mapping["billed_units"],
        )


@dataclass(frozen=True)
class ObservabilityDescriptorSet:
    """Immutable ``descriptors.json`` map of quantity IDs to descriptors."""

    invocation_id: str
    descriptor_set_id: str
    quantities: Mapping[str, QuantityDescriptor]
    schema: str = RUN_OBSERVABILITY_SCHEMA
    kind: str = "descriptors"

    def __post_init__(self) -> None:
        _require_schema(self.schema, "descriptors")
        _require_kind(self.kind, "descriptors", "descriptors")
        validate_identifier(self.invocation_id, "descriptors.invocation_id")
        validate_identifier(self.descriptor_set_id, "descriptors.descriptor_set_id")
        raw_quantities = _require_mapping(self.quantities, "descriptors.quantities")
        validated: dict[str, QuantityDescriptor] = {}
        for qty_id, desc in raw_quantities.items():
            validate_identifier(qty_id, "descriptors.quantities key")
            validated[qty_id] = (
                desc
                if isinstance(desc, QuantityDescriptor)
                else QuantityDescriptor.from_mapping(desc)
            )
        object.__setattr__(self, "quantities", validated)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary matching ``descriptors.json``."""

        return {
            "schema": self.schema,
            "kind": self.kind,
            "invocation_id": self.invocation_id,
            "descriptor_set_id": self.descriptor_set_id,
            "quantities": {
                qty_id: desc.to_dict() for qty_id, desc in self.quantities.items()
            },
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ObservabilityDescriptorSet:
        """Validate and construct an :class:`ObservabilityDescriptorSet`."""

        mapping = _require_mapping(data, "descriptors")
        _require_keys(
            mapping,
            required={
                "schema",
                "kind",
                "invocation_id",
                "descriptor_set_id",
                "quantities",
            },
            context="descriptors",
        )
        raw_quantities = _require_mapping(mapping["quantities"], "descriptors.quantities")
        return cls(
            schema=mapping["schema"],
            kind=mapping["kind"],
            invocation_id=mapping["invocation_id"],
            descriptor_set_id=mapping["descriptor_set_id"],
            quantities={
                qty_id: QuantityDescriptor.from_mapping(desc)
                for qty_id, desc in raw_quantities.items()
            },
        )


@dataclass(frozen=True)
class ObservabilitySample:
    """One timestamped sample packet for a single observer and source."""

    invocation_id: str
    observer_id: str
    source_id: str
    descriptor_set_id: str
    sequence: int
    clock_domain_id: str
    subject: Mapping[str, Any]
    read_start_ns: int
    read_end_ns: int
    readings: Mapping[str, ObservabilityReading]
    schema: str = RUN_OBSERVABILITY_SCHEMA
    kind: str = "sample"

    def __post_init__(self) -> None:
        _require_schema(self.schema, "sample")
        _require_kind(self.kind, "sample", "sample")
        validate_identifier(self.invocation_id, "sample.invocation_id")
        validate_identifier(
            self.observer_id, "sample.observer_id", forbid_status_collision=True
        )
        validate_identifier(self.source_id, "sample.source_id")
        validate_identifier(self.descriptor_set_id, "sample.descriptor_set_id")
        _require_int(self.sequence, "sample.sequence", min_value=0)
        _require_nonempty_str(self.clock_domain_id, "sample.clock_domain_id")

        subject = _validate_and_copy_json(
            _require_mapping(self.subject, "sample.subject"), "sample.subject"
        )
        subject_kind = _require_choice(
            subject.get("kind"), OBSERVABILITY_SUBJECT_KINDS, "sample.subject.kind"
        )
        if subject_kind == "descendant_set" and "members" in subject:
            if not isinstance(subject["members"], list) or not all(
                isinstance(member, Mapping) for member in subject["members"]
            ):
                raise ObservabilitySchemaError(
                    "sample.subject.members must be a list of member identity mappings"
                )
        object.__setattr__(self, "subject", subject)

        start = _require_int(self.read_start_ns, "sample.read_start_ns", min_value=0)
        end = _require_int(self.read_end_ns, "sample.read_end_ns", min_value=0)
        if end < start:
            raise ObservabilitySchemaError(
                f"sample.read_end_ns ({end}) must be >= sample.read_start_ns ({start})"
            )

        raw_readings = _require_mapping(self.readings, "sample.readings")
        if not raw_readings:
            raise ObservabilitySchemaError("sample.readings must be nonempty")
        validated_readings: dict[str, ObservabilityReading] = {}
        for qty_id, item in raw_readings.items():
            validate_identifier(qty_id, "sample.readings key")
            if isinstance(item, ObservabilityReading):
                validated_readings[qty_id] = item
            else:
                validated_readings[qty_id] = ObservabilityReading.from_mapping(item)
        object.__setattr__(self, "readings", validated_readings)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary matching one JSONL sample row."""

        return {
            "schema": self.schema,
            "kind": self.kind,
            "invocation_id": self.invocation_id,
            "observer_id": self.observer_id,
            "source_id": self.source_id,
            "descriptor_set_id": self.descriptor_set_id,
            "sequence": self.sequence,
            "clock_domain_id": self.clock_domain_id,
            "subject": _validate_and_copy_json(self.subject, "sample.subject"),
            "read_start_ns": self.read_start_ns,
            "read_end_ns": self.read_end_ns,
            "readings": {
                qty_id: reading.to_dict()
                for qty_id, reading in self.readings.items()
            },
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ObservabilitySample:
        """Validate and construct an :class:`ObservabilitySample`."""

        mapping = _require_mapping(data, "sample")
        _require_keys(
            mapping,
            required={
                "schema",
                "kind",
                "invocation_id",
                "observer_id",
                "source_id",
                "descriptor_set_id",
                "sequence",
                "clock_domain_id",
                "subject",
                "read_start_ns",
                "read_end_ns",
                "readings",
            },
            context="sample",
        )
        raw_readings = _require_mapping(mapping["readings"], "sample.readings")
        return cls(
            schema=mapping["schema"],
            kind=mapping["kind"],
            invocation_id=mapping["invocation_id"],
            observer_id=mapping["observer_id"],
            source_id=mapping["source_id"],
            descriptor_set_id=mapping["descriptor_set_id"],
            sequence=mapping["sequence"],
            clock_domain_id=mapping["clock_domain_id"],
            subject=mapping["subject"],
            read_start_ns=mapping["read_start_ns"],
            read_end_ns=mapping["read_end_ns"],
            readings={
                qty_id: ObservabilityReading.from_mapping(reading)
                for qty_id, reading in raw_readings.items()
            },
        )


@dataclass(frozen=True)
class ObservabilityStatus:
    """Observer lifecycle, coverage, count, and writer/helper final status."""

    invocation_id: str
    observer_id: str
    lifecycle_state: str
    data_completeness: str
    source_status: Mapping[str, str]
    coverage: Mapping[str, Any]
    counts: Mapping[str, int]
    truncation_reasons: tuple[str, ...]
    writer: Mapping[str, Any]
    helper_cleanup: Mapping[str, Any]
    application_outcome: str
    invocation_outcome: str
    export_status: str
    unavailable_fields: tuple[str, ...] | None = None
    diagnostics: tuple[str, ...] | None = None
    schema: str = RUN_OBSERVABILITY_SCHEMA
    kind: str = "status"

    def __post_init__(self) -> None:
        _require_schema(self.schema, "status")
        _require_kind(self.kind, "status", "status")
        validate_identifier(self.invocation_id, "status.invocation_id")
        validate_identifier(
            self.observer_id, "status.observer_id", forbid_status_collision=True
        )
        lifecycle_state = _require_choice(
            self.lifecycle_state,
            OBSERVABILITY_LIFECYCLE_STATES,
            "status.lifecycle_state",
        )
        _require_choice(
            self.data_completeness,
            OBSERVABILITY_DATA_COMPLETENESS,
            "status.data_completeness",
        )

        raw_source_status = _require_mapping(self.source_status, "status.source_status")
        validated_sources: dict[str, str] = {}
        for source_id, state in raw_source_status.items():
            validate_identifier(source_id, "status.source_status key")
            validated_sources[source_id] = _require_nonempty_str(
                state, f"status.source_status.{source_id}"
            )
        object.__setattr__(self, "source_status", validated_sources)

        coverage = _validate_and_copy_json(
            _require_mapping(self.coverage, "status.coverage"), "status.coverage"
        )
        if "activation_ns" in coverage:
            _require_int(coverage["activation_ns"], "status.coverage.activation_ns", min_value=0)
        if "acquisition_end_ns" in coverage:
            _require_int(
                coverage["acquisition_end_ns"],
                "status.coverage.acquisition_end_ns",
                min_value=0,
            )
        if (
            "activation_ns" in coverage
            and "acquisition_end_ns" in coverage
            and coverage["acquisition_end_ns"] < coverage["activation_ns"]
        ):
            raise ObservabilitySchemaError(
                "status.coverage.acquisition_end_ns must be >= activation_ns"
            )
        if "missed_deadlines" in coverage:
            _require_int(
                coverage["missed_deadlines"],
                "status.coverage.missed_deadlines",
                min_value=0,
            )
        if "requested_interval_seconds" in coverage:
            _require_finite_number(
                coverage["requested_interval_seconds"],
                "status.coverage.requested_interval_seconds",
                strictly_positive=True,
            )
        if "source_windows_ns" in coverage:
            windows_map = _require_mapping(
                coverage["source_windows_ns"], "status.coverage.source_windows_ns"
            )
            for source_id, pair in windows_map.items():
                validate_identifier(source_id, "status.coverage.source_windows_ns key")
                _validate_ns_pair(pair, f"status.coverage.source_windows_ns.{source_id}")
        object.__setattr__(self, "coverage", coverage)

        raw_counts = _require_mapping(self.counts, "status.counts")
        _require_keys(
            raw_counts,
            required={"attempted", "written", "dropped", "failed"},
            optional={"pending"},
            context="status.counts",
        )
        attempted = _require_int(raw_counts["attempted"], "status.counts.attempted", min_value=0)
        written = _require_int(raw_counts["written"], "status.counts.written", min_value=0)
        dropped = _require_int(raw_counts["dropped"], "status.counts.dropped", min_value=0)
        failed = _require_int(raw_counts["failed"], "status.counts.failed", min_value=0)
        validated_counts: dict[str, int] = {
            "attempted": attempted,
            "written": written,
            "dropped": dropped,
            "failed": failed,
        }
        pending = 0
        if "pending" in raw_counts:
            pending = _require_int(raw_counts["pending"], "status.counts.pending", min_value=0)
            validated_counts["pending"] = pending
        if lifecycle_state == "closed":
            if pending != 0:
                raise ObservabilitySchemaError(
                    f"closed status cannot have non-zero pending count ({pending})"
                )
            if attempted != written + dropped + failed:
                raise ObservabilitySchemaError(
                    f"closed status counts do not reconcile: attempted={attempted} != "
                    f"written({written}) + dropped({dropped}) + failed({failed})"
                )
        else:
            if "pending" in raw_counts:
                if attempted != written + dropped + failed + pending:
                    raise ObservabilitySchemaError(
                        "status counts with pending must satisfy "
                        "attempted == written + dropped + failed + pending"
                    )
            elif written + dropped + failed > attempted:
                raise ObservabilitySchemaError(
                    "status counts cannot have written + dropped + failed > attempted"
                )
        object.__setattr__(self, "counts", validated_counts)

        object.__setattr__(
            self,
            "truncation_reasons",
            _validate_str_tuple(self.truncation_reasons, "status.truncation_reasons"),
        )
        object.__setattr__(
            self,
            "writer",
            _validate_and_copy_json(
                _require_mapping(self.writer, "status.writer"), "status.writer"
            ),
        )
        object.__setattr__(
            self,
            "helper_cleanup",
            _validate_and_copy_json(
                _require_mapping(self.helper_cleanup, "status.helper_cleanup"),
                "status.helper_cleanup",
            ),
        )
        _require_nonempty_str(self.application_outcome, "status.application_outcome")
        _require_nonempty_str(self.invocation_outcome, "status.invocation_outcome")
        _require_nonempty_str(self.export_status, "status.export_status")

        if self.unavailable_fields is not None:
            object.__setattr__(
                self,
                "unavailable_fields",
                _validate_str_tuple(
                    self.unavailable_fields, "status.unavailable_fields"
                ),
            )
        if self.diagnostics is not None:
            object.__setattr__(
                self,
                "diagnostics",
                _validate_str_tuple(self.diagnostics, "status.diagnostics"),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary matching ``status/<observer-id>.json``."""

        data: dict[str, Any] = {
            "schema": self.schema,
            "kind": self.kind,
            "invocation_id": self.invocation_id,
            "observer_id": self.observer_id,
            "lifecycle_state": self.lifecycle_state,
            "data_completeness": self.data_completeness,
            "source_status": dict(self.source_status),
            "coverage": _validate_and_copy_json(self.coverage, "status.coverage"),
            "counts": dict(self.counts),
            "truncation_reasons": list(self.truncation_reasons),
            "writer": _validate_and_copy_json(self.writer, "status.writer"),
            "helper_cleanup": _validate_and_copy_json(
                self.helper_cleanup, "status.helper_cleanup"
            ),
            "application_outcome": self.application_outcome,
            "invocation_outcome": self.invocation_outcome,
            "export_status": self.export_status,
        }
        if self.unavailable_fields is not None:
            data["unavailable_fields"] = list(self.unavailable_fields)
        if self.diagnostics is not None:
            data["diagnostics"] = list(self.diagnostics)
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ObservabilityStatus:
        """Validate and construct an :class:`ObservabilityStatus`."""

        mapping = _require_mapping(data, "status")
        _require_keys(
            mapping,
            required={
                "schema",
                "kind",
                "invocation_id",
                "observer_id",
                "lifecycle_state",
                "data_completeness",
                "source_status",
                "coverage",
                "counts",
                "truncation_reasons",
                "writer",
                "helper_cleanup",
                "application_outcome",
                "invocation_outcome",
                "export_status",
            },
            optional={"unavailable_fields", "diagnostics"},
            context="status",
        )
        return cls(
            schema=mapping["schema"],
            kind=mapping["kind"],
            invocation_id=mapping["invocation_id"],
            observer_id=mapping["observer_id"],
            lifecycle_state=mapping["lifecycle_state"],
            data_completeness=mapping["data_completeness"],
            source_status=mapping["source_status"],
            coverage=mapping["coverage"],
            counts=mapping["counts"],
            truncation_reasons=tuple(mapping["truncation_reasons"])
            if isinstance(mapping["truncation_reasons"], (list, tuple))
            else mapping["truncation_reasons"],
            writer=mapping["writer"],
            helper_cleanup=mapping["helper_cleanup"],
            application_outcome=mapping["application_outcome"],
            invocation_outcome=mapping["invocation_outcome"],
            export_status=mapping["export_status"],
            unavailable_fields=(
                None
                if "unavailable_fields" not in mapping
                else (
                    tuple(mapping["unavailable_fields"])
                    if isinstance(mapping["unavailable_fields"], (list, tuple))
                    else mapping["unavailable_fields"]
                )
            ),
            diagnostics=(
                None
                if "diagnostics" not in mapping
                else (
                    tuple(mapping["diagnostics"])
                    if isinstance(mapping["diagnostics"], (list, tuple))
                    else mapping["diagnostics"]
                )
            ),
        )


@dataclass(frozen=True)
class ObservabilityProfileStatus:
    """Profiling window status record (``profile-status.json``)."""

    invocation_id: str
    observer_id: str
    window_id: str
    window_kind: str
    requested_activities: tuple[str, ...]
    accepted_activities: tuple[str, ...]
    observed_event_counts: Mapping[str, int]
    clock_domain_id: str
    activation_ns: int
    lifecycle_state: str
    data_completeness: str
    guarantee_class: str
    exporter: str
    export_status: str
    requested_start_ns: int | None = None
    requested_end_ns: int | None = None
    start_call_bracket_ns: tuple[int, int] | None = None
    stop_call_bracket_ns: tuple[int, int] | None = None
    backend_event_bounds: Mapping[str, Any] = field(
        default_factory=lambda: {"status": "unavailable", "reason_code": "not_reported"}
    )
    stop_overshoot_bounds_seconds: tuple[int | float, int | float] | None = None
    longest_checkpoint_gap_seconds: int | float | None = None
    max_rows: int | None = None
    activity_diagnostics: tuple[str, ...] = ()
    operation_id: str | None = None
    operation_occurrences: tuple[int, ...] | None = None
    durable_steps: tuple[int, ...] | None = None
    export_mode: str | None = None
    native_export_seconds: int | float | None = None
    staging_bytes: int | None = None
    artifacts: tuple[ObservabilityArtifactRef, ...] | None = None
    truncation_reasons: tuple[str, ...] | None = None
    schema: str = RUN_OBSERVABILITY_SCHEMA
    kind: str = "profile_status"

    def __post_init__(self) -> None:
        _require_schema(self.schema, "profile_status")
        _require_kind(self.kind, "profile_status", "profile_status")
        validate_identifier(self.invocation_id, "profile_status.invocation_id")
        validate_identifier(
            self.observer_id,
            "profile_status.observer_id",
            forbid_status_collision=True,
        )
        validate_identifier(self.window_id, "profile_status.window_id")
        _require_choice(
            self.window_kind,
            OBSERVABILITY_PROFILE_WINDOW_KINDS,
            "profile_status.window_kind",
        )
        object.__setattr__(
            self,
            "requested_activities",
            _validate_str_tuple(
                self.requested_activities, "profile_status.requested_activities"
            ),
        )
        object.__setattr__(
            self,
            "accepted_activities",
            _validate_str_tuple(
                self.accepted_activities, "profile_status.accepted_activities"
            ),
        )
        raw_counts = _require_mapping(
            self.observed_event_counts, "profile_status.observed_event_counts"
        )
        validated_counts: dict[str, int] = {}
        for act, count in raw_counts.items():
            _require_nonempty_str(act, "profile_status.observed_event_counts key")
            validated_counts[act] = _require_int(
                count, f"profile_status.observed_event_counts.{act}", min_value=0
            )
        object.__setattr__(self, "observed_event_counts", validated_counts)

        _require_nonempty_str(self.clock_domain_id, "profile_status.clock_domain_id")
        _require_int(self.activation_ns, "profile_status.activation_ns", min_value=0)
        if self.requested_start_ns is not None:
            _require_int(
                self.requested_start_ns,
                "profile_status.requested_start_ns",
                min_value=0,
            )
        if self.requested_end_ns is not None:
            _require_int(
                self.requested_end_ns, "profile_status.requested_end_ns", min_value=0
            )
        if (
            self.requested_start_ns is not None
            and self.requested_end_ns is not None
            and self.requested_end_ns < self.requested_start_ns
        ):
            raise ObservabilitySchemaError(
                "profile_status.requested_end_ns must be >= requested_start_ns"
            )
        if self.start_call_bracket_ns is not None:
            object.__setattr__(
                self,
                "start_call_bracket_ns",
                _validate_ns_pair(
                    self.start_call_bracket_ns, "profile_status.start_call_bracket_ns"
                ),
            )
        if self.stop_call_bracket_ns is not None:
            object.__setattr__(
                self,
                "stop_call_bracket_ns",
                _validate_ns_pair(
                    self.stop_call_bracket_ns, "profile_status.stop_call_bracket_ns"
                ),
            )
        object.__setattr__(
            self,
            "backend_event_bounds",
            _validate_tagged_fact(
                self.backend_event_bounds, "profile_status.backend_event_bounds"
            ),
        )
        if self.stop_overshoot_bounds_seconds is not None:
            bounds = self.stop_overshoot_bounds_seconds
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ObservabilitySchemaError(
                    "profile_status.stop_overshoot_bounds_seconds must have 2 elements"
                )
            low = _require_finite_number(
                bounds[0], "profile_status.stop_overshoot_bounds_seconds[0]"
            )
            high = _require_finite_number(
                bounds[1], "profile_status.stop_overshoot_bounds_seconds[1]"
            )
            if high < low:
                raise ObservabilitySchemaError(
                    "profile_status.stop_overshoot_bounds_seconds[1] must be >= [0]"
                )
            object.__setattr__(self, "stop_overshoot_bounds_seconds", (low, high))
        if self.longest_checkpoint_gap_seconds is not None:
            _require_finite_number(
                self.longest_checkpoint_gap_seconds,
                "profile_status.longest_checkpoint_gap_seconds",
                min_value=0.0,
            )
        _require_choice(
            self.lifecycle_state,
            OBSERVABILITY_LIFECYCLE_STATES,
            "profile_status.lifecycle_state",
        )
        _require_choice(
            self.data_completeness,
            OBSERVABILITY_DATA_COMPLETENESS,
            "profile_status.data_completeness",
        )
        _require_nonempty_str(self.guarantee_class, "profile_status.guarantee_class")
        _require_nonempty_str(self.exporter, "profile_status.exporter")
        if self.max_rows is not None:
            _require_int(self.max_rows, "profile_status.max_rows", min_value=1)
        _require_nonempty_str(self.export_status, "profile_status.export_status")
        object.__setattr__(
            self,
            "activity_diagnostics",
            _validate_str_tuple(
                self.activity_diagnostics, "profile_status.activity_diagnostics"
            ),
        )
        if self.operation_id is not None:
            _require_nonempty_str(self.operation_id, "profile_status.operation_id")
        if self.operation_occurrences is not None:
            if not isinstance(self.operation_occurrences, (list, tuple)):
                raise ObservabilitySchemaError(
                    "profile_status.operation_occurrences must be a sequence of ints"
                )
            object.__setattr__(
                self,
                "operation_occurrences",
                tuple(
                    _require_int(
                        item,
                        f"profile_status.operation_occurrences[{idx}]",
                        min_value=1,
                    )
                    for idx, item in enumerate(self.operation_occurrences)
                ),
            )
        if self.durable_steps is not None:
            if not isinstance(self.durable_steps, (list, tuple)):
                raise ObservabilitySchemaError(
                    "profile_status.durable_steps must be a sequence of ints"
                )
            object.__setattr__(
                self,
                "durable_steps",
                tuple(
                    _require_int(
                        item, f"profile_status.durable_steps[{idx}]", min_value=0
                    )
                    for idx, item in enumerate(self.durable_steps)
                ),
            )
        if self.export_mode is not None:
            _require_nonempty_str(self.export_mode, "profile_status.export_mode")
        if self.native_export_seconds is not None:
            _require_finite_number(
                self.native_export_seconds,
                "profile_status.native_export_seconds",
                min_value=0.0,
            )
        if self.staging_bytes is not None:
            _require_int(self.staging_bytes, "profile_status.staging_bytes", min_value=0)
        if self.artifacts is not None:
            if not isinstance(self.artifacts, (list, tuple)):
                raise ObservabilitySchemaError("profile_status.artifacts must be a sequence")
            object.__setattr__(
                self,
                "artifacts",
                tuple(
                    item
                    if isinstance(item, ObservabilityArtifactRef)
                    else ObservabilityArtifactRef.from_mapping(item)
                    for item in self.artifacts
                ),
            )
        if self.truncation_reasons is not None:
            object.__setattr__(
                self,
                "truncation_reasons",
                _validate_str_tuple(
                    self.truncation_reasons, "profile_status.truncation_reasons"
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary matching ``profile_status`` wire."""

        data: dict[str, Any] = {
            "schema": self.schema,
            "kind": self.kind,
            "invocation_id": self.invocation_id,
            "observer_id": self.observer_id,
            "window_id": self.window_id,
            "window_kind": self.window_kind,
            "requested_activities": list(self.requested_activities),
            "accepted_activities": list(self.accepted_activities),
            "observed_event_counts": dict(self.observed_event_counts),
            "clock_domain_id": self.clock_domain_id,
            "activation_ns": self.activation_ns,
        }
        if self.requested_start_ns is not None:
            data["requested_start_ns"] = self.requested_start_ns
        if self.requested_end_ns is not None:
            data["requested_end_ns"] = self.requested_end_ns
        if self.start_call_bracket_ns is not None:
            data["start_call_bracket_ns"] = list(self.start_call_bracket_ns)
        if self.stop_call_bracket_ns is not None:
            data["stop_call_bracket_ns"] = list(self.stop_call_bracket_ns)
        data["backend_event_bounds"] = _validate_and_copy_json(
            self.backend_event_bounds, "profile_status.backend_event_bounds"
        )
        if self.stop_overshoot_bounds_seconds is not None:
            data["stop_overshoot_bounds_seconds"] = list(
                self.stop_overshoot_bounds_seconds
            )
        if self.longest_checkpoint_gap_seconds is not None:
            data["longest_checkpoint_gap_seconds"] = (
                self.longest_checkpoint_gap_seconds
            )
        data["lifecycle_state"] = self.lifecycle_state
        data["data_completeness"] = self.data_completeness
        data["guarantee_class"] = self.guarantee_class
        data["exporter"] = self.exporter
        if self.max_rows is not None:
            data["max_rows"] = self.max_rows
        data["export_status"] = self.export_status
        data["activity_diagnostics"] = list(self.activity_diagnostics)
        if self.operation_id is not None:
            data["operation_id"] = self.operation_id
        if self.operation_occurrences is not None:
            data["operation_occurrences"] = list(self.operation_occurrences)
        if self.durable_steps is not None:
            data["durable_steps"] = list(self.durable_steps)
        if self.export_mode is not None:
            data["export_mode"] = self.export_mode
        if self.native_export_seconds is not None:
            data["native_export_seconds"] = self.native_export_seconds
        if self.staging_bytes is not None:
            data["staging_bytes"] = self.staging_bytes
        if self.artifacts is not None:
            data["artifacts"] = [ref.to_dict() for ref in self.artifacts]
        if self.truncation_reasons is not None:
            data["truncation_reasons"] = list(self.truncation_reasons)
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ObservabilityProfileStatus:
        """Validate and construct an :class:`ObservabilityProfileStatus`."""

        mapping = _require_mapping(data, "profile_status")
        _require_keys(
            mapping,
            required={
                "schema",
                "kind",
                "invocation_id",
                "observer_id",
                "window_id",
                "window_kind",
                "requested_activities",
                "accepted_activities",
                "observed_event_counts",
                "clock_domain_id",
                "activation_ns",
                "lifecycle_state",
                "data_completeness",
                "guarantee_class",
                "exporter",
                "export_status",
            },
            optional={
                "requested_start_ns",
                "requested_end_ns",
                "start_call_bracket_ns",
                "stop_call_bracket_ns",
                "backend_event_bounds",
                "stop_overshoot_bounds_seconds",
                "longest_checkpoint_gap_seconds",
                "max_rows",
                "activity_diagnostics",
                "operation_id",
                "operation_occurrences",
                "durable_steps",
                "export_mode",
                "native_export_seconds",
                "staging_bytes",
                "artifacts",
                "truncation_reasons",
            },
            context="profile_status",
        )
        return cls(
            schema=mapping["schema"],
            kind=mapping["kind"],
            invocation_id=mapping["invocation_id"],
            observer_id=mapping["observer_id"],
            window_id=mapping["window_id"],
            window_kind=mapping["window_kind"],
            requested_activities=tuple(mapping["requested_activities"])
            if isinstance(mapping["requested_activities"], (list, tuple))
            else mapping["requested_activities"],
            accepted_activities=tuple(mapping["accepted_activities"])
            if isinstance(mapping["accepted_activities"], (list, tuple))
            else mapping["accepted_activities"],
            observed_event_counts=mapping["observed_event_counts"],
            clock_domain_id=mapping["clock_domain_id"],
            activation_ns=mapping["activation_ns"],
            requested_start_ns=mapping.get("requested_start_ns"),
            requested_end_ns=mapping.get("requested_end_ns"),
            start_call_bracket_ns=mapping.get("start_call_bracket_ns"),
            stop_call_bracket_ns=mapping.get("stop_call_bracket_ns"),
            backend_event_bounds=mapping.get(
                "backend_event_bounds",
                {"status": "unavailable", "reason_code": "not_reported"},
            ),
            stop_overshoot_bounds_seconds=mapping.get("stop_overshoot_bounds_seconds"),
            longest_checkpoint_gap_seconds=mapping.get(
                "longest_checkpoint_gap_seconds"
            ),
            lifecycle_state=mapping["lifecycle_state"],
            data_completeness=mapping["data_completeness"],
            guarantee_class=mapping["guarantee_class"],
            exporter=mapping["exporter"],
            max_rows=mapping.get("max_rows"),
            export_status=mapping["export_status"],
            activity_diagnostics=tuple(mapping.get("activity_diagnostics", ()))
            if isinstance(mapping.get("activity_diagnostics", ()), (list, tuple))
            else mapping["activity_diagnostics"],
            operation_id=mapping.get("operation_id"),
            operation_occurrences=mapping.get("operation_occurrences"),
            durable_steps=mapping.get("durable_steps"),
            export_mode=mapping.get("export_mode"),
            native_export_seconds=mapping.get("native_export_seconds"),
            staging_bytes=mapping.get("staging_bytes"),
            artifacts=mapping.get("artifacts"),
            truncation_reasons=(
                None
                if "truncation_reasons" not in mapping
                else (
                    tuple(mapping["truncation_reasons"])
                    if isinstance(mapping["truncation_reasons"], (list, tuple))
                    else mapping["truncation_reasons"]
                )
            ),
        )


@dataclass(frozen=True)
class ObservabilityProfileSummary:
    """Bounded structured profile summary artifact (``summary.json``).

    Parameters
    ----------
    invocation_id : str
        Owning invocation identifier.
    observer_id : str
        Owning profile observer identifier.
    window_id : str
        Profile window identifier.
    exporter : str
        Exporter name, such as ``"TorchOperatorSummary"``.
    max_rows : int
        Configured upper bound on serialized rows.
    total_rows : int
        Total available rows before bounding.
    rows : tuple of mapping
        Bounded row payload (at most ``max_rows`` entries).
    truncated : bool
        ``True`` whenever row count or row field values were bounded.
    truncation_reasons : tuple of str, optional
        Explicit reasons describing any applied bounding.
    source_provenance : mapping, optional
        Optional JSON-safe exporter/backend provenance metadata.
    """

    invocation_id: str
    observer_id: str
    window_id: str
    exporter: str
    max_rows: int
    total_rows: int
    rows: tuple[Mapping[str, Any], ...]
    truncated: bool = False
    truncation_reasons: tuple[str, ...] = ()
    source_provenance: Mapping[str, Any] = field(default_factory=dict)
    schema: str = RUN_OBSERVABILITY_SCHEMA
    kind: str = PROFILE_SUMMARY_RECORD_KIND

    def __post_init__(self) -> None:
        _require_schema(self.schema, "profile_summary")
        _require_kind(self.kind, PROFILE_SUMMARY_RECORD_KIND, "profile_summary")
        validate_identifier(self.invocation_id, "profile_summary.invocation_id")
        validate_identifier(
            self.observer_id,
            "profile_summary.observer_id",
            forbid_status_collision=True,
        )
        validate_identifier(self.window_id, "profile_summary.window_id")
        _require_nonempty_str(self.exporter, "profile_summary.exporter")
        max_rows = _require_int(self.max_rows, "profile_summary.max_rows", min_value=1)
        total_rows = _require_int(
            self.total_rows, "profile_summary.total_rows", min_value=0
        )
        truncated = _require_bool(self.truncated, "profile_summary.truncated")
        if not isinstance(self.rows, (list, tuple)):
            raise ObservabilitySchemaError("profile_summary.rows must be a sequence")
        if len(self.rows) > max_rows:
            raise ObservabilitySchemaError(
                f"profile_summary.rows length ({len(self.rows)}) exceeds max_rows ({max_rows})"
            )
        if total_rows < len(self.rows):
            raise ObservabilitySchemaError(
                f"profile_summary.total_rows ({total_rows}) cannot be less than len(rows) ({len(self.rows)})"
            )
        if total_rows > len(self.rows) and not truncated:
            raise ObservabilitySchemaError(
                "profile_summary must set truncated=True when total_rows > len(rows)"
            )
        validated_rows = tuple(
            _validate_and_copy_json(
                _require_mapping(row, f"profile_summary.rows[{idx}]"),
                f"profile_summary.rows[{idx}]",
            )
            for idx, row in enumerate(self.rows)
        )
        object.__setattr__(self, "rows", validated_rows)
        reasons = _validate_str_tuple(
            self.truncation_reasons, "profile_summary.truncation_reasons"
        )
        if reasons and not truncated:
            raise ObservabilitySchemaError(
                "profile_summary must set truncated=True when truncation_reasons is nonempty"
            )
        object.__setattr__(self, "truncation_reasons", reasons)
        object.__setattr__(
            self,
            "source_provenance",
            _validate_and_copy_json(
                _require_mapping(
                    self.source_provenance, "profile_summary.source_provenance"
                ),
                "profile_summary.source_provenance",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary matching ``summary.json``."""

        data: dict[str, Any] = {
            "schema": self.schema,
            "kind": self.kind,
            "invocation_id": self.invocation_id,
            "observer_id": self.observer_id,
            "window_id": self.window_id,
            "exporter": self.exporter,
            "max_rows": self.max_rows,
            "total_rows": self.total_rows,
            "rows": [
                _validate_and_copy_json(row, "profile_summary.rows")
                for row in self.rows
            ],
            "truncated": self.truncated,
            "truncation_reasons": list(self.truncation_reasons),
        }
        if self.source_provenance:
            data["source_provenance"] = _validate_and_copy_json(
                self.source_provenance, "profile_summary.source_provenance"
            )
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ObservabilityProfileSummary:
        """Validate and construct an :class:`ObservabilityProfileSummary`."""

        mapping = _require_mapping(data, "profile_summary")
        _require_keys(
            mapping,
            required={
                "schema",
                "kind",
                "invocation_id",
                "observer_id",
                "window_id",
                "exporter",
                "max_rows",
                "total_rows",
                "rows",
                "truncated",
                "truncation_reasons",
            },
            optional={"source_provenance"},
            context="profile_summary",
        )
        return cls(
            schema=mapping["schema"],
            kind=mapping["kind"],
            invocation_id=mapping["invocation_id"],
            observer_id=mapping["observer_id"],
            window_id=mapping["window_id"],
            exporter=mapping["exporter"],
            max_rows=mapping["max_rows"],
            total_rows=mapping["total_rows"],
            rows=tuple(mapping["rows"])
            if isinstance(mapping["rows"], (list, tuple))
            else mapping["rows"],
            truncated=mapping["truncated"],
            truncation_reasons=tuple(mapping["truncation_reasons"])
            if isinstance(mapping["truncation_reasons"], (list, tuple))
            else mapping["truncation_reasons"],
            source_provenance=mapping.get("source_provenance", {}),
        )


def parse_observability_record(
    data: Mapping[str, Any],
) -> (
    ObservabilityManifest
    | ObservabilityInventory
    | ObservabilityDescriptorSet
    | ObservabilitySample
    | ObservabilityStatus
    | ObservabilityProfileStatus
    | ObservabilityProfileSummary
):
    """Dispatch a raw mapping to its validated ``tpen.run-observability/v1`` type."""

    mapping = _require_mapping(data, "record")
    _require_schema(mapping.get("schema"), "record")
    kind = _require_choice(mapping.get("kind"), OBSERVABILITY_ALL_RECORD_KINDS, "record.kind")
    if kind == "manifest":
        return ObservabilityManifest.from_mapping(mapping)
    if kind == "inventory":
        return ObservabilityInventory.from_mapping(mapping)
    if kind == "descriptors":
        return ObservabilityDescriptorSet.from_mapping(mapping)
    if kind == "sample":
        return ObservabilitySample.from_mapping(mapping)
    if kind == "status":
        return ObservabilityStatus.from_mapping(mapping)
    if kind == "profile_status":
        return ObservabilityProfileStatus.from_mapping(mapping)
    return ObservabilityProfileSummary.from_mapping(mapping)


# ---------------------------------------------------------------------------
# Pure Hydra-configurable sink & single-writer bounded bundle writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunArtifactSink:
    """Pure, Hydra-configurable destination and byte/text limits for observability.

    Performs zero filesystem I/O during construction. Physical resolution and
    containment checks happen only when bound to an
    :class:`tpen.artifacts.ArtifactManager` via :meth:`resolve_bundle_dir`.

    Parameters
    ----------
    relative_dir : str, optional
        Run-relative directory root for observability bundles.
    flush_interval_seconds : float, optional
        Target periodic flush cadence in seconds (strictly positive, finite).
    max_sample_bytes : int, optional
        Hard complete-line byte cap per observer sample stream (default 64 MiB).
    max_pending_bytes : int, optional
        Soft pending in-memory sample buffer cap per observer (default 4 MiB).
    max_status_bytes : int, optional
        Separate reserved byte budget for status and profile summary JSON files
        (default 256 KiB).
    max_text_chars : int, optional
        Maximum length of bounded diagnostic text strings.
    max_list_items : int, optional
        Maximum length of bounded diagnostic lists.
    """

    relative_dir: str = "observability"
    flush_interval_seconds: float = 60.0
    max_sample_bytes: int = 67_108_864
    max_pending_bytes: int = 4_194_304
    max_status_bytes: int = 262_144
    max_text_chars: int = 512
    max_list_items: int = 64

    def __post_init__(self) -> None:
        normalized_dir = validate_relative_path(
            self.relative_dir, "RunArtifactSink.relative_dir"
        )
        object.__setattr__(self, "relative_dir", normalized_dir)
        flush_interval = _require_finite_number(
            self.flush_interval_seconds,
            "RunArtifactSink.flush_interval_seconds",
            strictly_positive=True,
        )
        object.__setattr__(self, "flush_interval_seconds", float(flush_interval))
        _require_int(self.max_sample_bytes, "RunArtifactSink.max_sample_bytes", min_value=1)
        _require_int(
            self.max_pending_bytes, "RunArtifactSink.max_pending_bytes", min_value=1
        )
        _require_int(self.max_status_bytes, "RunArtifactSink.max_status_bytes", min_value=1)
        _require_int(self.max_text_chars, "RunArtifactSink.max_text_chars", min_value=1)
        _require_int(self.max_list_items, "RunArtifactSink.max_list_items", min_value=1)

    def resolve_bundle_dir(
        self, artifact_manager: ArtifactManager, invocation_id: str
    ) -> Path:
        """Resolve ``<run_dir>/<relative_dir>/<invocation_id>`` with containment checks.

        Routes through :meth:`tpen.artifacts.ArtifactManager.path` so bound
        :class:`tpen.artifacts.ArtifactDestination` leaves are honored, and
        refuses any symlink or containment escape along the path.
        """

        valid_invocation = validate_identifier(invocation_id, "invocation_id")
        rel_parts = (*self.relative_dir.split("/"), valid_invocation)
        try:
            candidate = artifact_manager.path(*rel_parts)
        except ArtifactRoutingError as exc:
            raise ObservabilityPathError(str(exc)) from exc
        run_root = artifact_manager.run_dir
        _check_no_symlinks_under(run_root, rel_parts)
        if artifact_manager.destination is not None:
            physical_root = artifact_manager.destination.physical_root
            if physical_root.is_symlink():
                raise ObservabilityPathError(
                    f"refusing symlinked artifact destination {physical_root!s}"
                )
            resolved_physical = physical_root.resolve(strict=False)
            resolved_candidate = candidate.resolve(strict=False)
            try:
                resolved_candidate.relative_to(resolved_physical)
            except ValueError as exc:
                raise ObservabilityPathError(
                    f"observability bundle {candidate!s} resolves to {resolved_candidate!s}, "
                    f"outside physical destination {resolved_physical!s}"
                ) from exc
        return candidate

    def ownership_claim(
        self,
        artifact_manager: ArtifactManager,
        invocation_id: str,
        observer_id: str,
    ) -> ResourceOwnershipClaim:
        """Build an exclusive ``artifact_writer`` claim for one observer stream."""

        from tpen.callback.base import ResourceOwnershipClaim

        valid_observer = validate_identifier(
            observer_id, "observer_id", forbid_status_collision=True
        )
        bundle_dir = self.resolve_bundle_dir(artifact_manager, invocation_id)
        scope_path = bundle_dir / "resources" / f"{valid_observer}.jsonl"
        return ResourceOwnershipClaim(
            kind="artifact_writer",
            scope=str(scope_path.resolve(strict=False)),
        )


@dataclass(frozen=True)
class SampleAppendResult:
    """Typed outcome returned by :meth:`ObservabilityBundleWriter.append_sample`."""

    written: bool
    bytes_written: int
    total_bytes: int
    capped: bool = False
    reason: str | None = None


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _bound_text(
    text: str,
    *,
    max_chars: int,
    reasons: list[str],
    reason_tag: str = "text_truncated",
) -> str:
    if len(text) <= max_chars:
        return text
    _append_unique(reasons, reason_tag)
    if max_chars >= 4:
        return text[: max_chars - 3] + "..."
    return text[:max_chars]


def _bound_reason_list(
    reasons: Sequence[str], *, max_items: int
) -> tuple[str, ...]:
    """Bound a list of canonical truncation reason tags without mangling tag text."""

    deduped: list[str] = []
    for item in reasons:
        _append_unique(deduped, item)
    if len(deduped) <= max_items:
        return tuple(deduped)
    if max_items == 1:
        return ("list_truncated",)
    kept = deduped[: max_items - 1]
    if "list_truncated" not in kept:
        kept.append("list_truncated")
    return tuple(kept)


def _bound_str_sequence(
    values: Sequence[str],
    *,
    max_items: int,
    max_chars: int,
    reasons: list[str],
    list_reason: str = "list_truncated",
    text_reason: str = "text_truncated",
) -> tuple[str, ...]:
    trimmed = list(values)
    if len(trimmed) > max_items:
        _append_unique(reasons, list_reason)
        trimmed = trimmed[:max_items]
    return tuple(
        _bound_text(item, max_chars=max_chars, reasons=reasons, reason_tag=text_reason)
        for item in trimmed
    )


def _bound_json_value(
    value: Any,
    *,
    max_chars: int,
    max_items: int,
    reasons: list[str],
) -> Any:
    if type(value) is str:
        return _bound_text(value, max_chars=max_chars, reasons=reasons)
    if isinstance(value, Mapping):
        return {
            _bound_text(str(k), max_chars=max_chars, reasons=reasons): _bound_json_value(
                v, max_chars=max_chars, max_items=max_items, reasons=reasons
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) > max_items:
            _append_unique(reasons, "list_truncated")
            items = items[:max_items]
        return [
            _bound_json_value(
                item, max_chars=max_chars, max_items=max_items, reasons=reasons
            )
            for item in items
        ]
    return value


class ObservabilityBundleWriter:
    """Single-writer bounded persistence for one ``observability/<invocation-id>/`` bundle.

    Enforces:

    * Safe containment and zero-symlink checks on every write through
      :class:`RunArtifactSink` and :class:`tpen.artifacts.ArtifactManager`.
    * Write-once immutability for ``inventory.json`` and ``descriptors.json``.
    * Strict descriptor, source, subject-kind, and monotonic-sequence validation
      on sample appends, checking exact encoded ASCII line length against
      ``max_sample_bytes`` *before* calling :func:`tpen.durable_append.append_record`.
    * Independent ``max_status_bytes`` budget plus explicit text/list/row
      truncation for status and profile summary files so sample cap saturation
      never starves final diagnostics.
    * Atomic replacement for open and finalized ``manifest.json``, computing
      exact ``bytes`` and ``sha256`` over all closed bundle files without
      recursively hashing ``manifest.json``.
    """

    def __init__(
        self,
        artifact_manager: ArtifactManager,
        invocation_id: str,
        *,
        sink: RunArtifactSink | None = None,
    ) -> None:
        self._artifact_manager = artifact_manager
        self.invocation_id = validate_identifier(invocation_id, "invocation_id")
        self.sink = RunArtifactSink() if sink is None else sink
        if not isinstance(self.sink, RunArtifactSink):
            raise TypeError(
                f"sink must be a RunArtifactSink, got {type(self.sink).__name__}"
            )
        self.bundle_dir = self.sink.resolve_bundle_dir(
            artifact_manager, self.invocation_id
        )
        self._finalized = False
        self._inventory_written = False
        self._descriptors_written = False
        self._descriptor_set: ObservabilityDescriptorSet | None = None
        self._sample_sequences: dict[tuple[str, str], int] = {}
        self._sample_cap_reached: set[str] = set()
        self._writer_truncation_reasons: dict[str, list[str]] = {}
        self._pending_bytes: dict[str, int] = {}
        self._written_artifacts: dict[str, Path] = {}

    @property
    def finalized(self) -> bool:
        """Return whether ``manifest.json`` has been finalized."""

        return self._finalized

    def pending_bytes(self, observer_id: str) -> int:
        """Return currently reserved pending sample bytes for ``observer_id``."""

        valid_observer = validate_identifier(
            observer_id, "observer_id", forbid_status_collision=True
        )
        return self._pending_bytes.get(valid_observer, 0)

    def try_reserve_pending(self, observer_id: str, num_bytes: int) -> bool:
        """Attempt to reserve ``num_bytes`` against ``sink.max_pending_bytes``.

        Never blocks or deletes prior data. Returns ``False`` and records
        ``"pending_bytes_cap_reached"`` when the reservation would exceed
        ``max_pending_bytes``.
        """

        valid_observer = validate_identifier(
            observer_id, "observer_id", forbid_status_collision=True
        )
        size = _require_int(num_bytes, "num_bytes", min_value=0)
        current = self._pending_bytes.get(valid_observer, 0)
        if current + size > self.sink.max_pending_bytes:
            reasons = self._writer_truncation_reasons.setdefault(valid_observer, [])
            _append_unique(reasons, "pending_bytes_cap_reached")
            return False
        self._pending_bytes[valid_observer] = current + size
        return True

    def release_pending(self, observer_id: str, num_bytes: int) -> None:
        """Release previously reserved pending sample bytes for ``observer_id``."""

        valid_observer = validate_identifier(
            observer_id, "observer_id", forbid_status_collision=True
        )
        size = _require_int(num_bytes, "num_bytes", min_value=0)
        current = self._pending_bytes.get(valid_observer, 0)
        self._pending_bytes[valid_observer] = max(0, current - size)

    def observer_truncation_reasons(self, observer_id: str) -> tuple[str, ...]:
        """Return writer-tracked truncation reasons for ``observer_id``."""

        valid_observer = validate_identifier(
            observer_id, "observer_id", forbid_status_collision=True
        )
        return tuple(self._writer_truncation_reasons.get(valid_observer, ()))

    def _require_not_finalized(self, action: str) -> None:
        if self._finalized:
            raise ObservabilityArtifactError(
                f"cannot {action} after manifest.json has been finalized"
            )

    def _safe_artifact_path(
        self, rel_path: str, *, allow_manifest: bool = False
    ) -> Path:
        normalized = validate_relative_path(
            rel_path, "artifact_path", forbid_manifest_self=not allow_manifest
        )
        bundle_dir = self.sink.resolve_bundle_dir(
            self._artifact_manager, self.invocation_id
        )
        rel_parts = (*self.sink.relative_dir.split("/"), self.invocation_id, *normalized.split("/"))
        try:
            candidate = self._artifact_manager.path(*rel_parts)
        except ArtifactRoutingError as exc:
            raise ObservabilityPathError(str(exc)) from exc
        _check_no_symlinks_under(self._artifact_manager.run_dir, rel_parts)
        _check_no_symlinks_under(bundle_dir, normalized.split("/"))
        return candidate

    def _atomic_write_bytes(self, target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ObservabilityPathError(
                f"refusing to overwrite symlinked artifact {target!s}"
            )
        if target.exists() and target.is_dir():
            raise ObservabilityPathError(
                f"refusing to overwrite directory {target!s} with artifact file"
            )
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
            os.replace(tmp_path, target)
            tmp_path = None
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def write_open_manifest(self, manifest: ObservabilityManifest) -> Path:
        """Atomically write the initial ``lifecycle_state="open"`` manifest."""

        self._require_not_finalized("write_open_manifest")
        if not isinstance(manifest, ObservabilityManifest):
            raise TypeError("manifest must be an ObservabilityManifest")
        if manifest.invocation_id != self.invocation_id:
            raise ObservabilitySchemaError(
                f"manifest invocation_id {manifest.invocation_id!r} does not match "
                f"writer invocation_id {self.invocation_id!r}"
            )
        if manifest.lifecycle_state != "open":
            raise ObservabilitySchemaError(
                "write_open_manifest requires lifecycle_state == 'open'"
            )
        target = self._safe_artifact_path(MANIFEST_FILENAME, allow_manifest=True)
        payload = _dumps_canonical_json(manifest.to_dict()).encode("ascii")
        self._atomic_write_bytes(target, payload)
        return target

    def write_inventory(self, inventory: ObservabilityInventory) -> Path:
        """Write immutable ``inventory.json`` once; refuse overwrites."""

        self._require_not_finalized("write_inventory")
        if not isinstance(inventory, ObservabilityInventory):
            raise TypeError("inventory must be an ObservabilityInventory")
        if inventory.invocation_id != self.invocation_id:
            raise ObservabilitySchemaError(
                f"inventory invocation_id {inventory.invocation_id!r} does not match "
                f"writer invocation_id {self.invocation_id!r}"
            )
        target = self._safe_artifact_path(INVENTORY_FILENAME)
        if self._inventory_written or target.exists():
            raise ObservabilityArtifactError(
                f"{INVENTORY_FILENAME} is write-once and already exists at {target!s}"
            )
        payload = _dumps_canonical_json(inventory.to_dict()).encode("ascii")
        self._atomic_write_bytes(target, payload)
        self._inventory_written = True
        self._written_artifacts[INVENTORY_FILENAME] = target
        return target

    def write_descriptors(self, descriptor_set: ObservabilityDescriptorSet) -> Path:
        """Write immutable ``descriptors.json`` once; refuse overwrites."""

        self._require_not_finalized("write_descriptors")
        if not isinstance(descriptor_set, ObservabilityDescriptorSet):
            raise TypeError("descriptor_set must be an ObservabilityDescriptorSet")
        if descriptor_set.invocation_id != self.invocation_id:
            raise ObservabilitySchemaError(
                f"descriptors invocation_id {descriptor_set.invocation_id!r} does not match "
                f"writer invocation_id {self.invocation_id!r}"
            )
        target = self._safe_artifact_path(DESCRIPTORS_FILENAME)
        if self._descriptors_written or target.exists():
            raise ObservabilityArtifactError(
                f"{DESCRIPTORS_FILENAME} is write-once and already exists at {target!s}"
            )
        payload = _dumps_canonical_json(descriptor_set.to_dict()).encode("ascii")
        self._atomic_write_bytes(target, payload)
        self._descriptors_written = True
        self._descriptor_set = descriptor_set
        self._written_artifacts[DESCRIPTORS_FILENAME] = target
        return target

    def append_sample(self, sample: ObservabilitySample) -> SampleAppendResult:
        """Validate and append one complete ASCII JSONL sample line.

        Enforces invocation, descriptor-set, source, subject-kind, and
        strictly monotonic per-``(observer_id, source_id)`` sequence rules.
        Exact encoded line byte length is checked against
        ``sink.max_sample_bytes`` *before* calling
        :func:`tpen.durable_append.append_record`, so cap saturation never
        writes partial JSON or deletes earlier records.
        """

        self._require_not_finalized("append_sample")
        if not isinstance(sample, ObservabilitySample):
            raise TypeError("sample must be an ObservabilitySample")
        if sample.invocation_id != self.invocation_id:
            raise ObservabilitySchemaError(
                f"sample invocation_id {sample.invocation_id!r} does not match "
                f"writer invocation_id {self.invocation_id!r}"
            )
        if self._descriptor_set is not None:
            if sample.descriptor_set_id != self._descriptor_set.descriptor_set_id:
                raise ObservabilitySchemaError(
                    f"sample descriptor_set_id {sample.descriptor_set_id!r} does not match "
                    f"written descriptor_set_id {self._descriptor_set.descriptor_set_id!r}"
                )
            sample_subject_kind = sample.subject["kind"]
            for qty_id in sample.readings:
                descriptor = self._descriptor_set.quantities.get(qty_id)
                if descriptor is None:
                    raise ObservabilitySchemaError(
                        f"sample quantity {qty_id!r} is not declared in descriptors"
                    )
                if descriptor.source_id != sample.source_id:
                    raise ObservabilitySchemaError(
                        f"sample quantity {qty_id!r} belongs to source {descriptor.source_id!r}, "
                        f"not {sample.source_id!r}"
                    )
                if descriptor.subject_kind != sample_subject_kind:
                    raise ObservabilitySchemaError(
                        f"sample subject kind {sample_subject_kind!r} does not match "
                        f"descriptor subject_kind {descriptor.subject_kind!r} for {qty_id!r}"
                    )

        seq_key = (sample.observer_id, sample.source_id)
        if seq_key in self._sample_sequences:
            previous = self._sample_sequences[seq_key]
            if sample.sequence <= previous:
                raise ObservabilitySchemaError(
                    f"sample sequence for {seq_key} must strictly increase: "
                    f"got {sample.sequence} after {previous}"
                )

        rel_path = f"resources/{sample.observer_id}.jsonl"
        target = self._safe_artifact_path(rel_path)
        if target.exists() and target.is_dir():
            raise ObservabilityPathError(
                f"sample target {target!s} is a directory, expected file"
            )
        current_bytes = target.stat().st_size if target.exists() else 0

        line = _dumps_canonical_json(sample.to_dict(), indent=None)
        prefix_bytes = 1 if ends_without_newline(target) else 0
        needed_bytes = prefix_bytes + len(line.encode("ascii")) + 1

        if (
            sample.observer_id in self._sample_cap_reached
            or current_bytes + needed_bytes > self.sink.max_sample_bytes
        ):
            self._sample_cap_reached.add(sample.observer_id)
            reasons = self._writer_truncation_reasons.setdefault(sample.observer_id, [])
            _append_unique(reasons, "sample_bytes_cap_reached")
            return SampleAppendResult(
                written=False,
                bytes_written=0,
                total_bytes=current_bytes,
                capped=True,
                reason="sample_bytes_cap_reached",
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        append_record(target, line)
        self._sample_sequences[seq_key] = sample.sequence
        self._written_artifacts[rel_path] = target
        return SampleAppendResult(
            written=True,
            bytes_written=needed_bytes,
            total_bytes=current_bytes + needed_bytes,
            capped=False,
            reason=None,
        )

    def write_status(self, status: ObservabilityStatus) -> Path:
        """Write bounded ``status/<observer-id>.json`` under ``max_status_bytes``.

        Applies configured ``max_text_chars`` and ``max_list_items`` bounds to
        diagnostic fields, merges any writer-recorded cap truncation reasons,
        and enforces ``sink.max_status_bytes`` independently of sample cap
        saturation.
        """

        self._require_not_finalized("write_status")
        if not isinstance(status, ObservabilityStatus):
            raise TypeError("status must be an ObservabilityStatus")
        if status.invocation_id != self.invocation_id:
            raise ObservabilitySchemaError(
                f"status invocation_id {status.invocation_id!r} does not match "
                f"writer invocation_id {self.invocation_id!r}"
            )

        reasons: list[str] = list(status.truncation_reasons)
        for writer_reason in self._writer_truncation_reasons.get(status.observer_id, ()):
            _append_unique(reasons, writer_reason)

        bounded_unavailable = (
            None
            if status.unavailable_fields is None
            else _bound_str_sequence(
                status.unavailable_fields,
                max_items=self.sink.max_list_items,
                max_chars=self.sink.max_text_chars,
                reasons=reasons,
            )
        )
        bounded_diagnostics = (
            None
            if status.diagnostics is None
            else _bound_str_sequence(
                status.diagnostics,
                max_items=self.sink.max_list_items,
                max_chars=self.sink.max_text_chars,
                reasons=reasons,
            )
        )
        bounded_reasons = _bound_reason_list(
            reasons, max_items=self.sink.max_list_items
        )
        bounded_status = ObservabilityStatus(
            invocation_id=status.invocation_id,
            observer_id=status.observer_id,
            lifecycle_state=status.lifecycle_state,
            data_completeness=status.data_completeness,
            source_status=status.source_status,
            coverage=status.coverage,
            counts=status.counts,
            truncation_reasons=bounded_reasons,
            writer=status.writer,
            helper_cleanup=status.helper_cleanup,
            application_outcome=status.application_outcome,
            invocation_outcome=status.invocation_outcome,
            export_status=status.export_status,
            unavailable_fields=bounded_unavailable,
            diagnostics=bounded_diagnostics,
        )
        payload = _dumps_canonical_json(bounded_status.to_dict()).encode("ascii")
        if len(payload) > self.sink.max_status_bytes:
            raise ObservabilityArtifactError(
                f"encoded status payload ({len(payload)} bytes) exceeds "
                f"max_status_bytes ({self.sink.max_status_bytes})"
            )
        rel_path = f"status/{status.observer_id}.json"
        target = self._safe_artifact_path(rel_path)
        self._atomic_write_bytes(target, payload)
        self._written_artifacts[rel_path] = target
        return target

    def write_profile_status(
        self, profile_status: ObservabilityProfileStatus
    ) -> Path:
        """Write bounded ``profiles/<observer-id>/<window-id>/profile-status.json``.

        Never writes a file literally named ``status.json`` so legacy
        :func:`tpen.inspection.inspect_run` discovery stays unambiguous.
        """

        self._require_not_finalized("write_profile_status")
        if not isinstance(profile_status, ObservabilityProfileStatus):
            raise TypeError("profile_status must be an ObservabilityProfileStatus")
        if profile_status.invocation_id != self.invocation_id:
            raise ObservabilitySchemaError(
                f"profile_status invocation_id {profile_status.invocation_id!r} "
                f"does not match writer invocation_id {self.invocation_id!r}"
            )

        reasons: list[str] = (
            list(profile_status.truncation_reasons)
            if profile_status.truncation_reasons is not None
            else []
        )
        bounded_diag = _bound_str_sequence(
            profile_status.activity_diagnostics,
            max_items=self.sink.max_list_items,
            max_chars=self.sink.max_text_chars,
            reasons=reasons,
        )
        bounded_reasons: tuple[str, ...] | None = (
            None
            if not reasons and profile_status.truncation_reasons is None
            else _bound_reason_list(reasons, max_items=self.sink.max_list_items)
        )
        bounded_profile = ObservabilityProfileStatus(
            invocation_id=profile_status.invocation_id,
            observer_id=profile_status.observer_id,
            window_id=profile_status.window_id,
            window_kind=profile_status.window_kind,
            requested_activities=profile_status.requested_activities,
            accepted_activities=profile_status.accepted_activities,
            observed_event_counts=profile_status.observed_event_counts,
            clock_domain_id=profile_status.clock_domain_id,
            activation_ns=profile_status.activation_ns,
            requested_start_ns=profile_status.requested_start_ns,
            requested_end_ns=profile_status.requested_end_ns,
            start_call_bracket_ns=profile_status.start_call_bracket_ns,
            stop_call_bracket_ns=profile_status.stop_call_bracket_ns,
            backend_event_bounds=profile_status.backend_event_bounds,
            stop_overshoot_bounds_seconds=profile_status.stop_overshoot_bounds_seconds,
            longest_checkpoint_gap_seconds=profile_status.longest_checkpoint_gap_seconds,
            lifecycle_state=profile_status.lifecycle_state,
            data_completeness=profile_status.data_completeness,
            guarantee_class=profile_status.guarantee_class,
            exporter=profile_status.exporter,
            max_rows=profile_status.max_rows,
            export_status=profile_status.export_status,
            activity_diagnostics=bounded_diag,
            operation_id=profile_status.operation_id,
            operation_occurrences=profile_status.operation_occurrences,
            durable_steps=profile_status.durable_steps,
            export_mode=profile_status.export_mode,
            native_export_seconds=profile_status.native_export_seconds,
            staging_bytes=profile_status.staging_bytes,
            artifacts=profile_status.artifacts,
            truncation_reasons=bounded_reasons,
        )
        payload = _dumps_canonical_json(bounded_profile.to_dict()).encode("ascii")
        if len(payload) > self.sink.max_status_bytes:
            raise ObservabilityArtifactError(
                f"encoded profile_status payload ({len(payload)} bytes) exceeds "
                f"max_status_bytes ({self.sink.max_status_bytes})"
            )
        rel_path = (
            f"profiles/{profile_status.observer_id}/{profile_status.window_id}/"
            f"{PROFILE_STATUS_FILENAME}"
        )
        target = self._safe_artifact_path(rel_path)
        self._atomic_write_bytes(target, payload)
        self._written_artifacts[rel_path] = target
        return target

    def write_profile_summary(
        self,
        summary: ObservabilityProfileSummary,
        *,
        max_rows: int | None = None,
    ) -> Path:
        """Write bounded ``profiles/<observer-id>/<window-id>/summary.json``."""

        self._require_not_finalized("write_profile_summary")
        if not isinstance(summary, ObservabilityProfileSummary):
            raise TypeError("summary must be an ObservabilityProfileSummary")
        if summary.invocation_id != self.invocation_id:
            raise ObservabilitySchemaError(
                f"profile_summary invocation_id {summary.invocation_id!r} "
                f"does not match writer invocation_id {self.invocation_id!r}"
            )
        effective_max_rows = summary.max_rows
        if max_rows is not None:
            requested_max = _require_int(max_rows, "max_rows", min_value=1)
            effective_max_rows = min(effective_max_rows, requested_max)

        reasons: list[str] = list(summary.truncation_reasons)
        raw_rows = list(summary.rows)
        if len(raw_rows) > effective_max_rows:
            _append_unique(reasons, "max_rows_exceeded")
            raw_rows = raw_rows[:effective_max_rows]
        bounded_rows = tuple(
            _bound_json_value(
                row,
                max_chars=self.sink.max_text_chars,
                max_items=self.sink.max_list_items,
                reasons=reasons,
            )
            for row in raw_rows
        )
        truncated = summary.truncated or bool(reasons) or (
            summary.total_rows > len(bounded_rows)
        )
        bounded_reasons = _bound_reason_list(
            reasons, max_items=self.sink.max_list_items
        )
        bounded_summary = ObservabilityProfileSummary(
            invocation_id=summary.invocation_id,
            observer_id=summary.observer_id,
            window_id=summary.window_id,
            exporter=summary.exporter,
            max_rows=effective_max_rows,
            total_rows=summary.total_rows,
            rows=bounded_rows,
            truncated=truncated,
            truncation_reasons=bounded_reasons,
            source_provenance=summary.source_provenance,
        )
        payload = _dumps_canonical_json(bounded_summary.to_dict()).encode("ascii")
        if len(payload) > self.sink.max_status_bytes:
            raise ObservabilityArtifactError(
                f"encoded profile_summary payload ({len(payload)} bytes) exceeds "
                f"max_status_bytes ({self.sink.max_status_bytes})"
            )
        rel_path = (
            f"profiles/{summary.observer_id}/{summary.window_id}/"
            f"{PROFILE_SUMMARY_FILENAME}"
        )
        target = self._safe_artifact_path(rel_path)
        self._atomic_write_bytes(target, payload)
        self._written_artifacts[rel_path] = target
        return target

    def finalize_manifest(
        self,
        manifest: ObservabilityManifest,
        *,
        artifact_paths: Sequence[str] | None = None,
    ) -> ObservabilityManifest:
        """Compute actual ``bytes``/``sha256`` over bundle artifacts and atomically close ``manifest.json``.

        Never recursively hashes ``manifest.json`` itself.
        """

        self._require_not_finalized("finalize_manifest")
        if not isinstance(manifest, ObservabilityManifest):
            raise TypeError("manifest must be an ObservabilityManifest")
        if manifest.invocation_id != self.invocation_id:
            raise ObservabilitySchemaError(
                f"manifest invocation_id {manifest.invocation_id!r} does not match "
                f"writer invocation_id {self.invocation_id!r}"
            )
        if manifest.lifecycle_state == "open":
            raise ObservabilitySchemaError(
                "finalize_manifest requires lifecycle_state 'closed' or 'unsafe', not 'open'"
            )

        if artifact_paths is not None:
            ordered_paths: list[str] = []
            for raw in artifact_paths:
                normalized = validate_relative_path(
                    raw, "artifact_paths", forbid_manifest_self=True
                )
                if normalized not in ordered_paths:
                    ordered_paths.append(normalized)
        else:
            all_paths = set(self._written_artifacts) | {
                ref.path for ref in manifest.artifacts
            }
            ordered_paths = sorted(all_paths)

        refs: list[ObservabilityArtifactRef] = []
        for rel_path in ordered_paths:
            target = self._safe_artifact_path(rel_path)
            if not target.exists() or not target.is_file():
                raise ObservabilityArtifactError(
                    f"referenced bundle artifact {rel_path!r} is missing or not a regular file"
                )
            raw_bytes = target.read_bytes()
            digest = hashlib.sha256(raw_bytes).hexdigest()
            refs.append(
                ObservabilityArtifactRef(
                    path=rel_path, bytes=len(raw_bytes), sha256=digest
                )
            )

        finalized_manifest = ObservabilityManifest(
            invocation_id=manifest.invocation_id,
            lifecycle_state=manifest.lifecycle_state,
            data_completeness=manifest.data_completeness,
            run_id=manifest.run_id,
            producer=manifest.producer,
            configuration_digest=manifest.configuration_digest,
            inventory_id=manifest.inventory_id,
            descriptor_set_id=manifest.descriptor_set_id,
            observers=manifest.observers,
            limits=manifest.limits,
            executor_context=manifest.executor_context,
            scheduler_accounting=manifest.scheduler_accounting,
            artifacts=tuple(refs),
        )
        manifest_target = self._safe_artifact_path(
            MANIFEST_FILENAME, allow_manifest=True
        )
        payload = _dumps_canonical_json(finalized_manifest.to_dict()).encode("ascii")
        self._atomic_write_bytes(manifest_target, payload)
        self._finalized = True
        return finalized_manifest


# ---------------------------------------------------------------------------
# Pure read-only inspection models and pure statistical reducer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservabilityDiagnostic:
    """One non-fatal diagnostic emitted by read-only observability inspection."""

    code: str
    message: str
    path: str | None = None
    line_number: int | None = None
    observer_id: str | None = None
    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this diagnostic."""

        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "line_number": self.line_number,
            "observer_id": self.observer_id,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class QuantityObservationSummary:
    """Reduced summary for one ``(observer_id, source_id, quantity_id)`` series.

    Keeps gauge sampled maxima (marked as lower bounds), lifetime high-water
    marks, counter interval deltas/rates, capacities, and unknown-window vendor
    busy means strictly distinct. Never fabricates zero when readings are
    unavailable or errored.
    """

    observer_id: str
    source_id: str
    quantity_id: str
    unit: str
    statistic: str
    subject_kind: str
    window: str
    clock_domains: tuple[str, ...]
    sample_count: int
    ok_count: int
    unavailable_count: int
    error_count: int
    reason_codes: tuple[str, ...]
    # Gauge fields
    sampled_min: int | float | None = None
    sampled_max: int | float | None = None
    sampled_max_is_lower_bound: bool = False
    # High-water fields
    observed_max: int | float | None = None
    # Counter fields
    valid_interval_count: int = 0
    broken_interval_count: int = 0
    valid_delta_total: float | None = None
    valid_elapsed_seconds: float | None = None
    valid_rate_per_second: float | None = None
    # Capacity fields
    latest_value: int | float | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    # Vendor-window-mean fields
    observation_min: int | float | None = None
    observation_max: int | float | None = None
    observation_mean: float | None = None
    continuous_window_integrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe summary dictionary."""

        return {
            "observer_id": self.observer_id,
            "source_id": self.source_id,
            "quantity_id": self.quantity_id,
            "unit": self.unit,
            "statistic": self.statistic,
            "subject_kind": self.subject_kind,
            "window": self.window,
            "clock_domains": list(self.clock_domains),
            "sample_count": self.sample_count,
            "ok_count": self.ok_count,
            "unavailable_count": self.unavailable_count,
            "error_count": self.error_count,
            "reason_codes": list(self.reason_codes),
            "sampled_min": self.sampled_min,
            "sampled_max": self.sampled_max,
            "sampled_max_is_lower_bound": self.sampled_max_is_lower_bound,
            "observed_max": self.observed_max,
            "valid_interval_count": self.valid_interval_count,
            "broken_interval_count": self.broken_interval_count,
            "valid_delta_total": self.valid_delta_total,
            "valid_elapsed_seconds": self.valid_elapsed_seconds,
            "valid_rate_per_second": self.valid_rate_per_second,
            "latest_value": self.latest_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "observation_min": self.observation_min,
            "observation_max": self.observation_max,
            "observation_mean": self.observation_mean,
            "continuous_window_integrated": self.continuous_window_integrated,
        }


@dataclass(frozen=True)
class ObservabilityBundleSummary:
    """Pure reduced summary of one invocation's observability bundle."""

    invocation_id: str
    lifecycle_state: str
    data_completeness: str
    finalization: str
    assessment_reason: str | None
    clock_domains: tuple[str, ...]
    quantities: tuple[QuantityObservationSummary, ...]
    profile_statuses: tuple[ObservabilityProfileStatus, ...]
    profile_summaries: tuple[ObservabilityProfileSummary, ...]

    def quantity(
        self,
        source_id: str,
        quantity_id: str,
        *,
        observer_id: str | None = None,
    ) -> QuantityObservationSummary | None:
        """Look up one reduced quantity summary by source and quantity ID."""

        for item in self.quantities:
            if item.source_id != source_id or item.quantity_id != quantity_id:
                continue
            if observer_id is not None and item.observer_id != observer_id:
                continue
            return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe summary dictionary."""

        return {
            "invocation_id": self.invocation_id,
            "lifecycle_state": self.lifecycle_state,
            "data_completeness": self.data_completeness,
            "finalization": self.finalization,
            "assessment_reason": self.assessment_reason,
            "clock_domains": list(self.clock_domains),
            "quantities": [item.to_dict() for item in self.quantities],
            "profile_statuses": [item.to_dict() for item in self.profile_statuses],
            "profile_summaries": [item.to_dict() for item in self.profile_summaries],
        }


@dataclass(frozen=True)
class ObservabilityBundleInspection:
    """Full pure read-only inspection result for one invocation bundle."""

    bundle_dir: str
    invocation_id: str
    manifest: ObservabilityManifest
    inventory: ObservabilityInventory | None
    descriptors: ObservabilityDescriptorSet | None
    samples: Mapping[str, tuple[ObservabilitySample, ...]]
    statuses: Mapping[str, ObservabilityStatus]
    profile_statuses: tuple[ObservabilityProfileStatus, ...]
    profile_summaries: tuple[ObservabilityProfileSummary, ...]
    derived_lifecycle_state: str
    derived_data_completeness: str
    derived_finalization: str
    derived_reason: str | None
    torn_tails: tuple[ObservabilityDiagnostic, ...]
    interior_errors: tuple[ObservabilityDiagnostic, ...]
    diagnostics: tuple[ObservabilityDiagnostic, ...]
    summary: ObservabilityBundleSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe representation of this bundle inspection."""

        return {
            "bundle_dir": self.bundle_dir,
            "invocation_id": self.invocation_id,
            "manifest": self.manifest.to_dict(),
            "inventory": None if self.inventory is None else self.inventory.to_dict(),
            "descriptors": (
                None if self.descriptors is None else self.descriptors.to_dict()
            ),
            "samples": {
                obs_id: [sample.to_dict() for sample in stream]
                for obs_id, stream in sorted(self.samples.items())
            },
            "statuses": {
                obs_id: status.to_dict()
                for obs_id, status in sorted(self.statuses.items())
            },
            "profile_statuses": [item.to_dict() for item in self.profile_statuses],
            "profile_summaries": [item.to_dict() for item in self.profile_summaries],
            "derived_lifecycle_state": self.derived_lifecycle_state,
            "derived_data_completeness": self.derived_data_completeness,
            "derived_finalization": self.derived_finalization,
            "derived_reason": self.derived_reason,
            "torn_tails": [item.to_dict() for item in self.torn_tails],
            "interior_errors": [item.to_dict() for item in self.interior_errors],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True)
class RunObservabilityInspection:
    """Read-only observability inspection result across a TPEN run directory."""

    run_dir: str
    status: str
    invocations: tuple[ObservabilityBundleInspection, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe representation of this run inspection."""

        return {
            "run_dir": self.run_dir,
            "status": self.status,
            "invocations": [bundle.to_dict() for bundle in self.invocations],
        }


# ---------------------------------------------------------------------------
# Pure read-only bundle & run inspection implementation
# ---------------------------------------------------------------------------


def _read_strict_json_file(path: Path, *, rel_path: str) -> dict[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ObservabilityArtifactError(
            f"failed to read {rel_path}: {exc}"
        ) from exc
    parsed = _loads_strict_json(raw_text, context=rel_path)
    if not isinstance(parsed, dict):
        raise ObservabilitySchemaError(
            f"{rel_path} must contain a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def _check_closed_manifest_artifacts(
    bundle_dir: Path, manifest: ObservabilityManifest
) -> None:
    """Verify containment, no symlinks, and exact bytes/sha256 for manifest artifacts."""

    for ref in manifest.artifacts:
        target = _check_no_symlinks_under(bundle_dir, ref.path.split("/"))
        if manifest.lifecycle_state == "closed":
            if not target.exists() or not target.is_file():
                raise ObservabilityArtifactError(
                    f"closed manifest artifact {ref.path!r} is missing or not a file"
                )
            raw_bytes = target.read_bytes()
            if len(raw_bytes) != ref.bytes:
                raise ObservabilityArtifactError(
                    f"closed manifest artifact {ref.path!r} byte size mismatch: "
                    f"expected {ref.bytes}, observed {len(raw_bytes)}"
                )
            digest = hashlib.sha256(raw_bytes).hexdigest()
            if digest != ref.sha256:
                raise ObservabilityArtifactError(
                    f"closed manifest artifact {ref.path!r} sha256 mismatch: "
                    f"expected {ref.sha256}, observed {digest}"
                )


def _read_sample_stream(
    path: Path,
    *,
    rel_path: str,
    expected_invocation_id: str,
    expected_observer_id: str,
    descriptors: ObservabilityDescriptorSet | None,
    torn_tails: list[ObservabilityDiagnostic],
    interior_errors: list[ObservabilityDiagnostic],
    diagnostics: list[ObservabilityDiagnostic],
) -> tuple[ObservabilitySample, ...]:
    """Recover complete valid sample rows while isolating torn tails and interior errors."""

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ObservabilityArtifactError(f"failed to read {rel_path}: {exc}") from exc

    if not raw_bytes:
        return ()

    has_unterminated_tail = not raw_bytes.endswith(b"\n")
    raw_lines = raw_bytes.split(b"\n")
    if has_unterminated_tail:
        tail_line_number = len(raw_lines)
        tail_diag = ObservabilityDiagnostic(
            code="torn_tail",
            message=(
                f"unterminated trailing line in {rel_path} at line {tail_line_number}; "
                "ignored without modifying file"
            ),
            path=rel_path,
            line_number=tail_line_number,
            observer_id=expected_observer_id,
        )
        torn_tails.append(tail_diag)
        diagnostics.append(tail_diag)
        complete_lines = raw_lines[:-1]
    else:
        # Final element after trailing newline is always empty.
        complete_lines = raw_lines[:-1]

    recovered: list[ObservabilitySample] = []
    last_sequence: dict[str, int] = {}

    for line_idx, raw_line in enumerate(complete_lines, start=1):
        if not raw_line.strip():
            err = ObservabilityDiagnostic(
                code="invalid_interior_record",
                message=f"blank interior line in {rel_path} at line {line_idx}",
                path=rel_path,
                line_number=line_idx,
                observer_id=expected_observer_id,
            )
            interior_errors.append(err)
            diagnostics.append(err)
            continue
        try:
            text = raw_line.decode("utf-8")
            parsed = _loads_strict_json(text, context=f"{rel_path}:{line_idx}")
        except (UnicodeDecodeError, ObservabilitySchemaError) as exc:
            err = ObservabilityDiagnostic(
                code="invalid_interior_record",
                message=f"corrupt interior JSON at {rel_path}:{line_idx}: {exc}",
                path=rel_path,
                line_number=line_idx,
                observer_id=expected_observer_id,
            )
            interior_errors.append(err)
            diagnostics.append(err)
            continue

        if isinstance(parsed, Mapping):
            raw_schema = parsed.get("schema")
            if isinstance(raw_schema, str) and raw_schema != RUN_OBSERVABILITY_SCHEMA:
                raise ObservabilitySchemaError(
                    f"unsupported schema {raw_schema!r} in {rel_path}:{line_idx}; "
                    f"expected {RUN_OBSERVABILITY_SCHEMA!r}"
                )
            raw_kind = parsed.get("kind")
            if isinstance(raw_kind, str) and raw_kind != "sample":
                raise ObservabilitySchemaError(
                    f"unsupported record kind {raw_kind!r} in sample stream {rel_path}:{line_idx}"
                )

        try:
            sample = ObservabilitySample.from_mapping(parsed)
        except (ObservabilitySchemaError, ObservabilityPathError) as exc:
            err = ObservabilityDiagnostic(
                code="invalid_interior_record",
                message=f"invalid sample record at {rel_path}:{line_idx}: {exc}",
                path=rel_path,
                line_number=line_idx,
                observer_id=expected_observer_id,
            )
            interior_errors.append(err)
            diagnostics.append(err)
            continue

        if sample.invocation_id != expected_invocation_id:
            err = ObservabilityDiagnostic(
                code="invalid_interior_record",
                message=(
                    f"sample invocation_id {sample.invocation_id!r} does not match "
                    f"manifest invocation_id {expected_invocation_id!r}"
                ),
                path=rel_path,
                line_number=line_idx,
                observer_id=expected_observer_id,
                source_id=sample.source_id,
            )
            interior_errors.append(err)
            diagnostics.append(err)
            continue

        if sample.observer_id != expected_observer_id:
            err = ObservabilityDiagnostic(
                code="invalid_interior_record",
                message=(
                    f"sample observer_id {sample.observer_id!r} does not match "
                    f"stream observer_id {expected_observer_id!r}"
                ),
                path=rel_path,
                line_number=line_idx,
                observer_id=expected_observer_id,
                source_id=sample.source_id,
            )
            interior_errors.append(err)
            diagnostics.append(err)
            continue

        if descriptors is not None:
            if sample.descriptor_set_id != descriptors.descriptor_set_id:
                err = ObservabilityDiagnostic(
                    code="invalid_interior_record",
                    message=(
                        f"sample descriptor_set_id {sample.descriptor_set_id!r} does not match "
                        f"bundle descriptors {descriptors.descriptor_set_id!r}"
                    ),
                    path=rel_path,
                    line_number=line_idx,
                    observer_id=expected_observer_id,
                    source_id=sample.source_id,
                )
                interior_errors.append(err)
                diagnostics.append(err)
                continue
            mismatch_reason: str | None = None
            sample_subject_kind = sample.subject["kind"]
            for qty_id in sample.readings:
                desc = descriptors.quantities.get(qty_id)
                if desc is None:
                    mismatch_reason = f"quantity {qty_id!r} missing from descriptors"
                    break
                if desc.source_id != sample.source_id:
                    mismatch_reason = (
                        f"quantity {qty_id!r} source {desc.source_id!r} != {sample.source_id!r}"
                    )
                    break
                if desc.subject_kind != sample_subject_kind:
                    mismatch_reason = (
                        f"subject kind {sample_subject_kind!r} != descriptor {desc.subject_kind!r}"
                    )
                    break
            if mismatch_reason is not None:
                err = ObservabilityDiagnostic(
                    code="invalid_interior_record",
                    message=f"descriptor mismatch at {rel_path}:{line_idx}: {mismatch_reason}",
                    path=rel_path,
                    line_number=line_idx,
                    observer_id=expected_observer_id,
                    source_id=sample.source_id,
                )
                interior_errors.append(err)
                diagnostics.append(err)
                continue

        if sample.source_id in last_sequence and sample.sequence <= last_sequence[sample.source_id]:
            err = ObservabilityDiagnostic(
                code="invalid_interior_record",
                message=(
                    f"non-monotonic sequence {sample.sequence} after "
                    f"{last_sequence[sample.source_id]} for source {sample.source_id!r}"
                ),
                path=rel_path,
                line_number=line_idx,
                observer_id=expected_observer_id,
                source_id=sample.source_id,
            )
            interior_errors.append(err)
            diagnostics.append(err)
            continue

        last_sequence[sample.source_id] = sample.sequence
        recovered.append(sample)

    return tuple(recovered)


def _reduce_bundle_summary(
    *,
    invocation_id: str,
    derived_lifecycle_state: str,
    derived_data_completeness: str,
    derived_finalization: str,
    derived_reason: str | None,
    inventory: ObservabilityInventory | None,
    descriptors: ObservabilityDescriptorSet | None,
    samples_by_observer: Mapping[str, tuple[ObservabilitySample, ...]],
    profile_statuses: tuple[ObservabilityProfileStatus, ...],
    profile_summaries: tuple[ObservabilityProfileSummary, ...],
    diagnostics: list[ObservabilityDiagnostic],
) -> ObservabilityBundleSummary:
    """Pure statistical reducer preserving statistic, window, and generation boundaries."""

    bundle_clock_domains: set[str] = set()
    if inventory is not None:
        for dom in inventory.clock_domains:
            bundle_clock_domains.add(str(dom["clock_domain_id"]))
    for profile_status in profile_statuses:
        bundle_clock_domains.add(profile_status.clock_domain_id)

    # Group valid samples by (observer_id, source_id) preserving sequence order.
    grouped_samples: dict[tuple[str, str], list[ObservabilitySample]] = {}
    for observer_id, stream in sorted(samples_by_observer.items()):
        for sample in stream:
            bundle_clock_domains.add(sample.clock_domain_id)
            grouped_samples.setdefault((observer_id, sample.source_id), []).append(sample)

    summaries: list[QuantityObservationSummary] = []

    if descriptors is not None:
        for (observer_id, source_id), source_samples in sorted(grouped_samples.items()):
            ordered = sorted(source_samples, key=lambda item: item.sequence)
            # Determine quantities observed for this (observer_id, source_id).
            qty_ids: set[str] = set()
            for sample in ordered:
                qty_ids.update(sample.readings.keys())

            for qty_id in sorted(qty_ids):
                desc = descriptors.quantities.get(qty_id)
                if desc is None:
                    continue
                qty_samples: list[tuple[ObservabilitySample, ObservabilityReading]] = []
                for sample in ordered:
                    reading = sample.readings.get(qty_id)
                    if reading is not None:
                        qty_samples.append((sample, reading))

                ok_values: list[int | float] = []
                unavailable_count = 0
                error_count = 0
                reason_codes: set[str] = set()
                qty_clocks: set[str] = set()

                for sample, reading in qty_samples:
                    qty_clocks.add(sample.clock_domain_id)
                    if reading.status == "ok":
                        assert reading.value is not None
                        ok_values.append(reading.value)
                    elif reading.status == "unavailable":
                        unavailable_count += 1
                        if reading.reason_code is not None:
                            reason_codes.add(reading.reason_code)
                    else:
                        error_count += 1
                        if reading.reason_code is not None:
                            reason_codes.add(reading.reason_code)

                ok_count = len(ok_values)
                sample_count = len(qty_samples)

                sampled_min: int | float | None = None
                sampled_max: int | float | None = None
                sampled_max_is_lower_bound = False
                observed_max: int | float | None = None
                valid_interval_count = 0
                broken_interval_count = 0
                valid_delta_total: float | None = None
                valid_elapsed_seconds: float | None = None
                valid_rate_per_second: float | None = None
                latest_value: int | float | None = None
                min_value: int | float | None = None
                max_value: int | float | None = None
                observation_min: int | float | None = None
                observation_max: int | float | None = None
                observation_mean: float | None = None

                if desc.statistic == "gauge":
                    if ok_values:
                        sampled_min = min(ok_values)
                        sampled_max = max(ok_values)
                        sampled_max_is_lower_bound = True
                elif desc.statistic == "high_water":
                    if ok_values:
                        observed_max = max(ok_values)
                elif desc.statistic == "capacity":
                    if ok_values:
                        latest_value = ok_values[-1]
                        min_value = min(ok_values)
                        max_value = max(ok_values)
                elif desc.statistic == "vendor_window_mean":
                    if ok_values:
                        observation_min = min(ok_values)
                        observation_max = max(ok_values)
                        observation_mean = sum(float(v) for v in ok_values) / ok_count
                elif desc.statistic == "counter":
                    delta_sum = 0.0
                    elapsed_sum = 0.0
                    for (prev_sample, prev_reading), (curr_sample, curr_reading) in zip(
                        qty_samples, qty_samples[1:]
                    ):
                        if prev_reading.status != "ok" or curr_reading.status != "ok":
                            broken_interval_count += 1
                            continue
                        if prev_sample.clock_domain_id != curr_sample.clock_domain_id:
                            broken_interval_count += 1
                            diagnostics.append(
                                ObservabilityDiagnostic(
                                    code="clock_domain_mismatch",
                                    message=(
                                        f"clock domain changed between sequence "
                                        f"{prev_sample.sequence} and {curr_sample.sequence} "
                                        f"for {source_id}/{qty_id}"
                                    ),
                                    observer_id=observer_id,
                                    source_id=source_id,
                                )
                            )
                            continue
                        if _subject_generation_key(
                            prev_sample.subject
                        ) != _subject_generation_key(curr_sample.subject):
                            broken_interval_count += 1
                            diagnostics.append(
                                ObservabilityDiagnostic(
                                    code="generation_change",
                                    message=(
                                        f"subject generation changed between sequence "
                                        f"{prev_sample.sequence} and {curr_sample.sequence} "
                                        f"for {source_id}/{qty_id}"
                                    ),
                                    observer_id=observer_id,
                                    source_id=source_id,
                                )
                            )
                            continue
                        assert prev_reading.value is not None
                        assert curr_reading.value is not None
                        delta = float(curr_reading.value - prev_reading.value)
                        if delta < 0.0:
                            broken_interval_count += 1
                            diagnostics.append(
                                ObservabilityDiagnostic(
                                    code="counter_reset",
                                    message=(
                                        f"counter reset detected between sequence "
                                        f"{prev_sample.sequence} ({prev_reading.value}) and "
                                        f"{curr_sample.sequence} ({curr_reading.value}) "
                                        f"for {source_id}/{qty_id}"
                                    ),
                                    observer_id=observer_id,
                                    source_id=source_id,
                                )
                            )
                            continue
                        elapsed_ns = curr_sample.read_end_ns - prev_sample.read_end_ns
                        if elapsed_ns <= 0:
                            broken_interval_count += 1
                            continue
                        valid_interval_count += 1
                        delta_sum += delta
                        elapsed_sum += elapsed_ns / 1e9
                    if valid_interval_count > 0:
                        valid_delta_total = delta_sum
                        valid_elapsed_seconds = elapsed_sum
                        if elapsed_sum > 0.0:
                            valid_rate_per_second = delta_sum / elapsed_sum

                summaries.append(
                    QuantityObservationSummary(
                        observer_id=observer_id,
                        source_id=source_id,
                        quantity_id=qty_id,
                        unit=desc.unit,
                        statistic=desc.statistic,
                        subject_kind=desc.subject_kind,
                        window=desc.window,
                        clock_domains=tuple(sorted(qty_clocks)),
                        sample_count=sample_count,
                        ok_count=ok_count,
                        unavailable_count=unavailable_count,
                        error_count=error_count,
                        reason_codes=tuple(sorted(reason_codes)),
                        sampled_min=sampled_min,
                        sampled_max=sampled_max,
                        sampled_max_is_lower_bound=sampled_max_is_lower_bound,
                        observed_max=observed_max,
                        valid_interval_count=valid_interval_count,
                        broken_interval_count=broken_interval_count,
                        valid_delta_total=valid_delta_total,
                        valid_elapsed_seconds=valid_elapsed_seconds,
                        valid_rate_per_second=valid_rate_per_second,
                        latest_value=latest_value,
                        min_value=min_value,
                        max_value=max_value,
                        observation_min=observation_min,
                        observation_max=observation_max,
                        observation_mean=observation_mean,
                        continuous_window_integrated=False,
                    )
                )

    return ObservabilityBundleSummary(
        invocation_id=invocation_id,
        lifecycle_state=derived_lifecycle_state,
        data_completeness=derived_data_completeness,
        finalization=derived_finalization,
        assessment_reason=derived_reason,
        clock_domains=tuple(sorted(bundle_clock_domains)),
        quantities=tuple(summaries),
        profile_statuses=profile_statuses,
        profile_summaries=profile_summaries,
    )


def inspect_observability_bundle(bundle_dir: str | Path) -> ObservabilityBundleInspection:
    """Pure, read-only inspection and reduction of one observability bundle directory.

    Performs zero filesystem writes, repairs, or deletions. Refuses symlinked
    directories/files, escaping paths, unsupported major schemas/kinds, and
    byte/SHA-256 mismatches against closed manifests. Recovers valid complete
    JSONL lines while diagnosing torn tails and invalid interior rows separately.

    Parameters
    ----------
    bundle_dir : str or pathlib.Path
        Path to one ``observability/<invocation-id>/`` directory.

    Returns
    -------
    ObservabilityBundleInspection
        Validated bundle artifacts, diagnostics, derived assessment, and pure
        quantity/profile summary.
    """

    root = Path(bundle_dir)
    if root.is_symlink():
        raise ObservabilityPathError(
            f"refusing symlinked observability bundle directory {root!s}"
        )
    if not root.exists() or not root.is_dir():
        raise ObservabilityArtifactError(
            f"observability bundle directory not found: {root!s}"
        )
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        _check_no_symlinks_under(root, [entry.name])

    manifest_path = _check_no_symlinks_under(root, [MANIFEST_FILENAME])
    if not manifest_path.exists() or not manifest_path.is_file():
        raise ObservabilityArtifactError(
            f"missing {MANIFEST_FILENAME} in observability bundle {root!s}"
        )
    manifest = ObservabilityManifest.from_mapping(
        _read_strict_json_file(manifest_path, rel_path=MANIFEST_FILENAME)
    )
    _check_closed_manifest_artifacts(root, manifest)

    diagnostics: list[ObservabilityDiagnostic] = []
    torn_tails: list[ObservabilityDiagnostic] = []
    interior_errors: list[ObservabilityDiagnostic] = []

    # Optional immutable inventory.json
    inventory: ObservabilityInventory | None = None
    inv_path = _check_no_symlinks_under(root, [INVENTORY_FILENAME])
    if inv_path.exists():
        if not inv_path.is_file():
            raise ObservabilityPathError(f"{INVENTORY_FILENAME} must be a regular file")
        inventory = ObservabilityInventory.from_mapping(
            _read_strict_json_file(inv_path, rel_path=INVENTORY_FILENAME)
        )
        if inventory.invocation_id != manifest.invocation_id:
            raise ObservabilitySchemaError(
                f"inventory invocation_id {inventory.invocation_id!r} does not match "
                f"manifest invocation_id {manifest.invocation_id!r}"
            )
        if inventory.inventory_id != manifest.inventory_id:
            raise ObservabilitySchemaError(
                f"inventory_id {inventory.inventory_id!r} does not match "
                f"manifest inventory_id {manifest.inventory_id!r}"
            )
    else:
        diagnostics.append(
            ObservabilityDiagnostic(
                code="missing_inventory",
                message=f"{INVENTORY_FILENAME} is absent in bundle",
                path=INVENTORY_FILENAME,
            )
        )

    # Optional immutable descriptors.json
    descriptors: ObservabilityDescriptorSet | None = None
    desc_path = _check_no_symlinks_under(root, [DESCRIPTORS_FILENAME])
    if desc_path.exists():
        if not desc_path.is_file():
            raise ObservabilityPathError(
                f"{DESCRIPTORS_FILENAME} must be a regular file"
            )
        descriptors = ObservabilityDescriptorSet.from_mapping(
            _read_strict_json_file(desc_path, rel_path=DESCRIPTORS_FILENAME)
        )
        if descriptors.invocation_id != manifest.invocation_id:
            raise ObservabilitySchemaError(
                f"descriptors invocation_id {descriptors.invocation_id!r} does not match "
                f"manifest invocation_id {manifest.invocation_id!r}"
            )
        if descriptors.descriptor_set_id != manifest.descriptor_set_id:
            raise ObservabilitySchemaError(
                f"descriptor_set_id {descriptors.descriptor_set_id!r} does not match "
                f"manifest descriptor_set_id {manifest.descriptor_set_id!r}"
            )
    else:
        diagnostics.append(
            ObservabilityDiagnostic(
                code="missing_descriptors",
                message=f"{DESCRIPTORS_FILENAME} is absent in bundle",
                path=DESCRIPTORS_FILENAME,
            )
        )

    # Discover sample streams under resources/*.jsonl
    samples_by_observer: dict[str, tuple[ObservabilitySample, ...]] = {}
    resources_dir = _check_no_symlinks_under(root, ["resources"])
    if resources_dir.exists():
        if not resources_dir.is_dir():
            raise ObservabilityPathError("resources entry must be a directory")
        for entry in sorted(resources_dir.iterdir(), key=lambda p: p.name):
            _check_no_symlinks_under(root, ["resources", entry.name])
            if not entry.is_file() or not entry.name.endswith(".jsonl"):
                continue
            observer_id = validate_identifier(
                entry.name[: -len(".jsonl")],
                "resources sample filename",
                forbid_status_collision=True,
            )
            rel_path = f"resources/{entry.name}"
            samples_by_observer[observer_id] = _read_sample_stream(
                entry,
                rel_path=rel_path,
                expected_invocation_id=manifest.invocation_id,
                expected_observer_id=observer_id,
                descriptors=descriptors,
                torn_tails=torn_tails,
                interior_errors=interior_errors,
                diagnostics=diagnostics,
            )

    # Discover observer statuses under status/*.json
    statuses: dict[str, ObservabilityStatus] = {}
    status_dir = _check_no_symlinks_under(root, ["status"])
    if status_dir.exists():
        if not status_dir.is_dir():
            raise ObservabilityPathError("status entry must be a directory")
        for entry in sorted(status_dir.iterdir(), key=lambda p: p.name):
            _check_no_symlinks_under(root, ["status", entry.name])
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            observer_id = validate_identifier(
                entry.name[: -len(".json")],
                "status filename",
                forbid_status_collision=True,
            )
            rel_path = f"status/{entry.name}"
            parsed_status = ObservabilityStatus.from_mapping(
                _read_strict_json_file(entry, rel_path=rel_path)
            )
            if parsed_status.invocation_id != manifest.invocation_id:
                raise ObservabilitySchemaError(
                    f"status invocation_id {parsed_status.invocation_id!r} does not match "
                    f"manifest invocation_id {manifest.invocation_id!r}"
                )
            if parsed_status.observer_id != observer_id:
                raise ObservabilitySchemaError(
                    f"status observer_id {parsed_status.observer_id!r} does not match "
                    f"filename {entry.name!r}"
                )
            statuses[observer_id] = parsed_status

    # Discover profile window statuses and summaries under profiles/<observer>/<window>/
    profile_statuses: list[ObservabilityProfileStatus] = []
    profile_summaries: list[ObservabilityProfileSummary] = []
    profiles_dir = _check_no_symlinks_under(root, ["profiles"])
    if profiles_dir.exists():
        if not profiles_dir.is_dir():
            raise ObservabilityPathError("profiles entry must be a directory")
        for obs_entry in sorted(profiles_dir.iterdir(), key=lambda p: p.name):
            _check_no_symlinks_under(root, ["profiles", obs_entry.name])
            if not obs_entry.is_dir():
                raise ObservabilityPathError(
                    f"unexpected non-directory entry under profiles/: {obs_entry.name!r}"
                )
            observer_id = validate_identifier(
                obs_entry.name,
                "profiles observer directory",
                forbid_status_collision=True,
            )
            for win_entry in sorted(obs_entry.iterdir(), key=lambda p: p.name):
                _check_no_symlinks_under(
                    root, ["profiles", obs_entry.name, win_entry.name]
                )
                if not win_entry.is_dir():
                    raise ObservabilityPathError(
                        f"unexpected non-directory entry under profiles/{obs_entry.name}/: "
                        f"{win_entry.name!r}"
                    )
                window_id = validate_identifier(
                    win_entry.name, "profiles window directory"
                )
                # Inspect every child for symlinks before reading.
                for child in sorted(win_entry.iterdir(), key=lambda p: p.name):
                    _check_no_symlinks_under(
                        root, ["profiles", obs_entry.name, win_entry.name, child.name]
                    )

                canonical_status = win_entry / PROFILE_STATUS_FILENAME
                legacy_status = win_entry / LEGACY_PROFILE_STATUS_FILENAME
                if canonical_status.exists() and legacy_status.exists():
                    raise ObservabilityArtifactError(
                        f"ambiguous profile status files in profiles/{observer_id}/{window_id}/: "
                        f"both {PROFILE_STATUS_FILENAME} and {LEGACY_PROFILE_STATUS_FILENAME} exist"
                    )
                chosen_status = (
                    canonical_status
                    if canonical_status.exists()
                    else (legacy_status if legacy_status.exists() else None)
                )
                if chosen_status is not None:
                    rel_status = (
                        f"profiles/{observer_id}/{window_id}/{chosen_status.name}"
                    )
                    pstatus = ObservabilityProfileStatus.from_mapping(
                        _read_strict_json_file(chosen_status, rel_path=rel_status)
                    )
                    if pstatus.invocation_id != manifest.invocation_id:
                        raise ObservabilitySchemaError(
                            f"profile_status invocation_id {pstatus.invocation_id!r} != "
                            f"manifest invocation_id {manifest.invocation_id!r}"
                        )
                    if pstatus.observer_id != observer_id or pstatus.window_id != window_id:
                        raise ObservabilitySchemaError(
                            f"profile_status ({pstatus.observer_id!r}, {pstatus.window_id!r}) "
                            f"does not match path profiles/{observer_id}/{window_id}/"
                        )
                    profile_statuses.append(pstatus)

                summary_path = win_entry / PROFILE_SUMMARY_FILENAME
                if summary_path.exists():
                    rel_summary = (
                        f"profiles/{observer_id}/{window_id}/{PROFILE_SUMMARY_FILENAME}"
                    )
                    psummary = ObservabilityProfileSummary.from_mapping(
                        _read_strict_json_file(summary_path, rel_path=rel_summary)
                    )
                    if psummary.invocation_id != manifest.invocation_id:
                        raise ObservabilitySchemaError(
                            f"profile_summary invocation_id {psummary.invocation_id!r} != "
                            f"manifest invocation_id {manifest.invocation_id!r}"
                        )
                    if psummary.observer_id != observer_id or psummary.window_id != window_id:
                        raise ObservabilitySchemaError(
                            f"profile_summary ({psummary.observer_id!r}, {psummary.window_id!r}) "
                            f"does not match path profiles/{observer_id}/{window_id}/"
                        )
                    profile_summaries.append(psummary)

    # Derive lifecycle, completeness, finalization, and assessment reason purely from evidence.
    profile_observer_ids = {ps.observer_id for ps in profile_statuses}
    missing_final_status = False
    for obs in manifest.observers:
        obs_id = str(obs["observer_id"])
        has_status = obs_id in statuses or obs_id in profile_observer_ids
        if not has_status:
            missing_final_status = True
            diagnostics.append(
                ObservabilityDiagnostic(
                    code="missing_final_status",
                    message=f"observer {obs_id!r} has no final status artifact",
                    observer_id=obs_id,
                )
            )
    for obs_id in samples_by_observer:
        if obs_id not in statuses:
            missing_final_status = True
            if not any(
                d.code == "missing_final_status" and d.observer_id == obs_id
                for d in diagnostics
            ):
                diagnostics.append(
                    ObservabilityDiagnostic(
                        code="missing_final_status",
                        message=f"sample stream observer {obs_id!r} has no status file",
                        observer_id=obs_id,
                    )
                )

    any_unsafe = manifest.lifecycle_state == "unsafe" or any(
        st.lifecycle_state == "unsafe" for st in statuses.values()
    ) or any(ps.lifecycle_state == "unsafe" for ps in profile_statuses)
    any_open_status = any(
        st.lifecycle_state == "open" for st in statuses.values()
    ) or any(ps.lifecycle_state == "open" for ps in profile_statuses)
    has_recovered_data = any(bool(stream) for stream in samples_by_observer.values()) or bool(
        profile_statuses
    )

    if any_unsafe:
        derived_lifecycle_state = "unsafe"
        derived_finalization = "unsafe"
        derived_data_completeness = "partial" if has_recovered_data else "unknown"
        derived_reason: str | None = "unsafe_lifecycle_state"
    elif (
        manifest.lifecycle_state == "open"
        or bool(torn_tails)
        or missing_final_status
        or any_open_status
    ):
        derived_lifecycle_state = "open"
        derived_finalization = "unknown"
        derived_data_completeness = "partial" if has_recovered_data else "unknown"
        derived_reason = "torn_tail_or_missing_final_status"
    else:
        derived_lifecycle_state = "closed"
        derived_finalization = "confirmed"
        completeness_values = [
            manifest.data_completeness,
            *(st.data_completeness for st in statuses.values()),
            *(ps.data_completeness for ps in profile_statuses),
        ]
        if interior_errors:
            derived_data_completeness = "partial" if has_recovered_data else "unknown"
            derived_reason = "invalid_interior_records"
        elif "unknown" in completeness_values:
            derived_data_completeness = "unknown"
            derived_reason = None
        elif "partial" in completeness_values:
            derived_data_completeness = "partial"
            derived_reason = None
        elif all(val == "none" for val in completeness_values):
            derived_data_completeness = "none"
            derived_reason = None
        elif "none" in completeness_values and "complete" in completeness_values:
            derived_data_completeness = "partial"
            derived_reason = None
        else:
            derived_data_completeness = "complete"
            derived_reason = None

    profile_statuses_tuple = tuple(profile_statuses)
    profile_summaries_tuple = tuple(profile_summaries)
    summary = _reduce_bundle_summary(
        invocation_id=manifest.invocation_id,
        derived_lifecycle_state=derived_lifecycle_state,
        derived_data_completeness=derived_data_completeness,
        derived_finalization=derived_finalization,
        derived_reason=derived_reason,
        inventory=inventory,
        descriptors=descriptors,
        samples_by_observer=samples_by_observer,
        profile_statuses=profile_statuses_tuple,
        profile_summaries=profile_summaries_tuple,
        diagnostics=diagnostics,
    )

    return ObservabilityBundleInspection(
        bundle_dir=str(root.resolve()),
        invocation_id=manifest.invocation_id,
        manifest=manifest,
        inventory=inventory,
        descriptors=descriptors,
        samples=samples_by_observer,
        statuses=statuses,
        profile_statuses=profile_statuses_tuple,
        profile_summaries=profile_summaries_tuple,
        derived_lifecycle_state=derived_lifecycle_state,
        derived_data_completeness=derived_data_completeness,
        derived_finalization=derived_finalization,
        derived_reason=derived_reason,
        torn_tails=tuple(torn_tails),
        interior_errors=tuple(interior_errors),
        diagnostics=tuple(diagnostics),
        summary=summary,
    )


def inspect_run_observability(
    run_dir: str | Path,
    *,
    relative_dir: str = "observability",
) -> RunObservabilityInspection:
    """Pure, read-only inspection of all observability bundles under a TPEN run directory.

    Absent ``<run_dir>/<relative_dir>`` returns
    ``RunObservabilityInspection(run_dir=..., status="absent", invocations=())``
    without creating any files or fabricating zero utilization. Multiple
    invocation bundles remain strictly isolated; duplicate ``invocation_id``
    collisions and symlinked directories/files are refused.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        TPEN run directory to inspect.
    relative_dir : str, optional
        Run-relative directory holding per-invocation bundles
        (default ``"observability"``).

    Returns
    -------
    RunObservabilityInspection
        Absent marker or ordered tuple of inspected invocation bundles.
    """

    normalized_rel = validate_relative_path(relative_dir, "relative_dir")
    run_root = Path(run_dir)
    if run_root.is_symlink():
        raise ObservabilityPathError(
            f"refusing symlinked run directory {run_root!s}"
        )
    if not run_root.exists() or not run_root.is_dir():
        raise ObservabilityArtifactError(f"run directory not found: {run_root!s}")

    obs_root = _check_no_symlinks_under(run_root, normalized_rel.split("/"))
    if not obs_root.exists():
        return RunObservabilityInspection(
            run_dir=str(run_root.resolve()),
            status="absent",
            invocations=(),
        )
    if not obs_root.is_dir():
        raise ObservabilityPathError(
            f"observability path {obs_root!s} exists and is not a directory"
        )

    bundles: list[ObservabilityBundleInspection] = []
    seen_invocations: dict[str, str] = {}
    for child in sorted(obs_root.iterdir(), key=lambda p: p.name):
        _check_no_symlinks_under(obs_root, [child.name])
        if not child.is_dir():
            raise ObservabilityArtifactError(
                f"unexpected non-directory entry under {obs_root!s}: {child.name!r}"
            )
        bundle = inspect_observability_bundle(child)
        if bundle.invocation_id in seen_invocations:
            raise ObservabilityArtifactError(
                f"duplicate invocation_id {bundle.invocation_id!r} found in both "
                f"{seen_invocations[bundle.invocation_id]!r} and {child.name!r}"
            )
        seen_invocations[bundle.invocation_id] = child.name
        bundles.append(bundle)

    bundles.sort(key=lambda item: item.invocation_id)
    return RunObservabilityInspection(
        run_dir=str(run_root.resolve()),
        status="present",
        invocations=tuple(bundles),
    )


__all__ = [
    "DESCRIPTORS_FILENAME",
    "INVENTORY_FILENAME",
    "LEGACY_PROFILE_STATUS_FILENAME",
    "MANIFEST_FILENAME",
    "OBSERVABILITY_ALL_RECORD_KINDS",
    "OBSERVABILITY_DATA_COMPLETENESS",
    "OBSERVABILITY_LIFECYCLE_STATES",
    "OBSERVABILITY_PROFILE_WINDOW_KINDS",
    "OBSERVABILITY_READING_STATUSES",
    "OBSERVABILITY_RECORD_KINDS",
    "OBSERVABILITY_STATISTICS",
    "OBSERVABILITY_SUBJECT_KINDS",
    "OBSERVABILITY_WINDOWS",
    "PROFILE_STATUS_FILENAME",
    "PROFILE_SUMMARY_FILENAME",
    "PROFILE_SUMMARY_RECORD_KIND",
    "RUN_OBSERVABILITY_SCHEMA",
    "ObservabilityArtifactError",
    "ObservabilityArtifactRef",
    "ObservabilityBundleInspection",
    "ObservabilityBundleSummary",
    "ObservabilityBundleWriter",
    "ObservabilityDescriptorSet",
    "ObservabilityDiagnostic",
    "ObservabilityInventory",
    "ObservabilityManifest",
    "ObservabilityPathError",
    "ObservabilityProfileStatus",
    "ObservabilityProfileSummary",
    "ObservabilityReading",
    "ObservabilitySample",
    "ObservabilitySchemaError",
    "ObservabilityStatus",
    "QuantityDescriptor",
    "QuantityObservationSummary",
    "RunArtifactSink",
    "RunObservabilityInspection",
    "SampleAppendResult",
    "inspect_observability_bundle",
    "inspect_run_observability",
    "parse_observability_record",
    "validate_identifier",
    "validate_relative_path",
]
