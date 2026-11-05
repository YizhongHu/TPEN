"""Executor client for the core-declared checkpoint-recovery profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from ._codec import canonical_bytes, decode_json, fields, sequence
from .environments import PreparedPythonEnvironment
from .workers import SubprocessWorker

PROFILE_CHECK_SCHEMA = "tpen.profile-check/v1"
REFUSAL_CODES = frozenset(
    {"unknown-profile", "unsupported-callback", "unsupported-logger", "malformed-config"}
)
PROFILE_REFUSAL_CODES = REFUSAL_CODES


class ProfileUnavailable(RuntimeError):
    """Raised when core cannot provide a trustworthy profile capability."""


@dataclass(frozen=True, slots=True)
class ProfileCheck:
    """Strict immutable executor mirror of core's profile-check record."""

    schema: ClassVar[str] = PROFILE_CHECK_SCHEMA
    profile: str
    verdict: str
    state_owners: frozenset[str]
    refusal: Mapping[str, str] | None

    def __post_init__(self) -> None:
        if type(self.profile) is not str or not self.profile:
            raise TypeError("profile must be a non-empty string")
        if self.verdict not in {"supported", "refused"}:
            raise ValueError("verdict must be supported or refused")
        if any(type(owner) is not str or not owner for owner in self.state_owners):
            raise TypeError("state_owners must contain non-empty strings")
        object.__setattr__(self, "state_owners", frozenset(self.state_owners))
        if self.refusal is None:
            if self.verdict == "refused":
                raise ValueError("refused profile check requires refusal")
            return
        if self.verdict != "refused":
            raise ValueError("supported profile check cannot contain refusal")
        if not isinstance(self.refusal, Mapping) or set(self.refusal) != {"code", "reason"}:
            raise TypeError("refusal must contain code and reason")
        if type(self.refusal["code"]) is not str or self.refusal["code"] not in REFUSAL_CODES:
            raise ValueError("unknown refusal code")
        if type(self.refusal["reason"]) is not str:
            raise TypeError("refusal reason must be a string")
        object.__setattr__(self, "refusal", MappingProxyType(dict(self.refusal)))

    @classmethod
    def from_mapping(cls, value: Any) -> "ProfileCheck":
        """Decode one strict v1 core profile-check mapping."""

        if type(value) is not dict:
            raise ProfileUnavailable("profile check is not a JSON object")
        try:
            fields(value, {"schema", "profile", "verdict", "state_owners", "refusal"}, cls.schema, "ProfileCheck")
            if type(value["profile"]) is not str or not value["profile"]:
                raise TypeError("profile must be a non-empty string")
            if type(value["verdict"]) is not str or value["verdict"] not in {"supported", "refused"}:
                raise ValueError("unknown profile verdict")
            owners = sequence(value["state_owners"], "state_owners")
            if any(type(owner) is not str or not owner for owner in owners):
                raise TypeError("state_owners must contain non-empty strings")
            if len(set(owners)) != len(owners):
                raise ValueError("state_owners contains duplicates")
            refusal = value["refusal"]
            if value["verdict"] == "supported" and refusal is not None:
                raise ValueError("supported profile check cannot contain refusal")
            if value["verdict"] == "refused" and refusal is None:
                raise ValueError("refused profile check requires refusal")
            return cls(value["profile"], value["verdict"], frozenset(owners), refusal)
        except (TypeError, ValueError, KeyError) as exc:
            raise ProfileUnavailable("profile check failed strict v1 decode") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-shaped record."""

        return {
            "schema": self.schema,
            "profile": self.profile,
            "verdict": self.verdict,
            "state_owners": sorted(self.state_owners),
            "refusal": None if self.refusal is None else dict(self.refusal),
        }

    def __hash__(self) -> int:
        """Hash canonical bytes so equality and hashing agree."""

        return hash(canonical_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class CoreProfileReference:
    """Reference to a core config and declared recovery profile.

    Parameters
    ----------
    config : str
        Absolute path to the unresolved core configuration.
    profile : str
        Declared profile identifier.
    """

    config: str
    profile: str

    def __post_init__(self) -> None:
        if type(self.config) is not str or not os.path.isabs(self.config):
            raise ValueError("config must be an absolute path")
        if type(self.profile) is not str or not self.profile:
            raise ValueError("profile must be a non-empty string")

    def read(
        self,
        worker: SubprocessWorker,
        environment: PreparedPythonEnvironment,
        *,
        cwd: str | os.PathLike[str] | None = None,
        stdout_path: str | os.PathLike[str],
        stderr_path: str | os.PathLike[str],
        timeout: float | None = None,
    ) -> ProfileCheck:
        """Launch core's profile CLI and decode one bound profile record.

        Returns
        -------
        ProfileCheck
            The decoded supported or refused result for ``self.profile``. A
            refused result is a valid, typed capability response and is
            returned to the caller.

        Raises
        ------
        ProfileUnavailable
            If the process, transport, JSON payload, strict v1 record, or
            requested-profile binding cannot be trusted. A response naming a
            different profile is unavailable rather than a refusal for this
            reference.
        """

        try:
            outcome = worker.run(
                environment,
                ("-m", "tpen.checkpoint.profiles", self.config, "--profile", self.profile),
                cwd=cwd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout=timeout,
            )
        except OSError as exc:
            raise ProfileUnavailable("profile process could not be launched") from exc
        if outcome.timed_out or outcome.returncode != 0:
            raise ProfileUnavailable("profile process did not complete successfully")
        try:
            payload = Path(outcome.stdout_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ProfileUnavailable("profile stdout could not be read") from exc
        if not payload.strip():
            raise ProfileUnavailable("profile stdout is empty")
        try:
            decoded = decode_json(payload)
        except (TypeError, ValueError) as exc:
            raise ProfileUnavailable("profile stdout is not one JSON object") from exc
        result = ProfileCheck.from_mapping(decoded)
        if result.profile != self.profile:
            raise ProfileUnavailable("profile response does not match requested profile")
        return result


__all__ = [
    "CoreProfileReference",
    "PROFILE_REFUSAL_CODES",
    "ProfileCheck",
    "ProfileUnavailable",
    "REFUSAL_CODES",
]
