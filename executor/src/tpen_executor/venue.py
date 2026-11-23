"""Strict, facility-neutral VenueProfile data and JSON loader.

Venue profiles are descriptive operator data only.  Loading a profile never
authorizes submission, chooses a queue, or consults a scheduler.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import json
from pathlib import Path
import re
from typing import Any, ClassVar

from ._codec import canonical_bytes, digest
from ._records import frozen_record


VENUE_PROFILE_SCHEMA = "tpen-executor/venue-profile-v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z", re.ASCII)
_PROFILE_KEYS = {"schema", "venue", "scheduler", "queues", "gpu_strata"}
_STRATUM_KEYS = {"name", "accelerator", "gpu_name_contains", "memory_mib", "mig_mode"}


class VenueProfileRefusal(ValueError):
    """Raised when a VenueProfile document is not exactly the V1 schema."""


def _refuse(message: str) -> VenueProfileRefusal:
    return VenueProfileRefusal(message)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _refuse(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise _refuse(f"{name} keys must be strings")
    return value


def _require_exact(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        missing = sorted(expected - set(data))
        extra = sorted(set(data) - expected)
        raise _refuse(f"{name} keys differ (missing={missing!r}, extra={extra!r})")


def _require_identifier(value: Any, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise _refuse(f"{name} must be an ASCII identifier-like string")
    return value


def _require_string(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise _refuse(f"{name} must be a non-empty string")
    return value


def _ensure_unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise _refuse(f"{name} names must be unique")


@frozen_record
class GpuStratumV1:
    """One immutable CPU or CUDA allocation description."""

    name: str
    accelerator: str
    gpu_name_contains: tuple[str, ...]
    memory_mib: int | None
    mig_mode: str

    def __post_init__(self) -> None:
        _require_identifier(self.name, "gpu stratum name")
        if type(self.accelerator) is not str or self.accelerator not in {"cpu", "cuda"}:
            raise _refuse("accelerator must be 'cpu' or 'cuda'")
        if type(self.gpu_name_contains) is not tuple or any(
            type(fragment) is not str or not fragment for fragment in self.gpu_name_contains
        ):
            raise _refuse("gpu_name_contains must contain non-empty strings")
        if type(self.mig_mode) is not str or self.mig_mode not in {"any", "disabled"}:
            raise _refuse("mig_mode must be 'any' or 'disabled'")
        if self.accelerator == "cpu":
            if self.gpu_name_contains or self.memory_mib is not None or self.mig_mode != "any":
                raise _refuse("cpu strata require empty GPU constraints and mig_mode='any'")
            return
        if not self.gpu_name_contains:
            raise _refuse("cuda strata require gpu_name_contains")
        if type(self.memory_mib) is not int or self.memory_mib <= 0:
            raise _refuse("cuda strata require positive integer memory_mib")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical mutable mapping for this stratum."""

        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_mapping(cls, data: Any) -> "GpuStratumV1":
        """Decode one strict stratum mapping."""

        mapping = _require_mapping(data, "gpu stratum")
        _require_exact(mapping, _STRATUM_KEYS, "gpu stratum")
        fragments = mapping["gpu_name_contains"]
        if isinstance(fragments, (str, bytes, bytearray)) or not isinstance(fragments, list):
            raise _refuse("gpu_name_contains must be a JSON array")
        return cls(
            _require_identifier(mapping["name"], "gpu stratum name"),
            _require_string(mapping["accelerator"], "accelerator"),
            tuple(fragments),
            mapping["memory_mib"],
            _require_string(mapping["mig_mode"], "mig_mode"),
        )


@frozen_record
class VenueProfileV1:
    """Immutable descriptive venue data; it is not submission authorization."""

    schema: ClassVar[str] = VENUE_PROFILE_SCHEMA
    venue: str
    scheduler: str
    queues: tuple[str, ...]
    gpu_strata: tuple[GpuStratumV1, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.venue, "venue")
        if type(self.scheduler) is not str or self.scheduler not in {"slurm", "pbs"}:
            raise _refuse("scheduler must be 'slurm' or 'pbs'")
        if type(self.queues) is not tuple or not self.queues:
            raise _refuse("queues must be a non-empty tuple")
        queue_names = tuple(_require_identifier(name, "queue name") for name in self.queues)
        if queue_names != self.queues:
            raise _refuse("queue names must be strings")
        _ensure_unique(queue_names, "queue")
        if type(self.gpu_strata) is not tuple or not self.gpu_strata:
            raise _refuse("gpu_strata must be a non-empty tuple")
        if any(type(stratum) is not GpuStratumV1 for stratum in self.gpu_strata):
            raise _refuse("gpu_strata must contain GpuStratumV1 records")
        _ensure_unique(tuple(stratum.name for stratum in self.gpu_strata), "gpu stratum")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical mutable mapping for this profile."""

        return {
            "schema": self.schema,
            "venue": self.venue,
            "scheduler": self.scheduler,
            "queues": list(self.queues),
            "gpu_strata": [stratum.to_dict() for stratum in self.gpu_strata],
        }

    def canonical_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON bytes for the profile."""

        return canonical_bytes(self.to_dict())

    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical profile mapping."""

        return digest(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "VenueProfileV1":
        """Decode one strict V1 mapping."""

        mapping = _require_mapping(data, "venue profile")
        if mapping.get("schema") != cls.schema:
            raise _refuse(f"venue profile schema must be {cls.schema!r}")
        _require_exact(mapping, _PROFILE_KEYS, "venue profile")
        queues = mapping["queues"]
        strata = mapping["gpu_strata"]
        if isinstance(queues, (str, bytes, bytearray)) or not isinstance(queues, list):
            raise _refuse("queues must be a JSON array")
        if isinstance(strata, (str, bytes, bytearray)) or not isinstance(strata, list):
            raise _refuse("gpu_strata must be a JSON array")
        return cls(
            _require_identifier(mapping["venue"], "venue"),
            _require_string(mapping["scheduler"], "scheduler"),
            tuple(_require_identifier(name, "queue name") for name in queues),
            tuple(GpuStratumV1.from_mapping(item) for item in strata),
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while refusing duplicate member names."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _refuse(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _refuse(f"invalid JSON constant: {value}")


def load_venue_profile(path: str | Path) -> VenueProfileV1:
    """Load one strict JSON VenueProfile document from a regular file."""

    try:
        candidate = Path(path)
        if candidate.suffix != ".json":
            raise _refuse("venue profile path must use the .json suffix")
        if candidate.is_symlink() or not candidate.is_file():
            raise _refuse("venue profile path must name a regular file")
        try:
            text = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise _refuse(f"venue profile cannot be read: {error}") from error
        try:
            data = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except VenueProfileRefusal:
            raise
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise _refuse(f"venue profile JSON is invalid: {error}") from error
        return VenueProfileV1.from_mapping(data)
    except VenueProfileRefusal:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise _refuse(f"venue profile path is not readable: {error}") from error


load_venue_profile_v1 = load_venue_profile
