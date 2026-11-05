"""Declared core checkpoint-recovery profile capabilities.

The declaration is intentionally syntax-only.  A profile check reads an
unresolved configuration and never imports or instantiates a Hydra target.
The executor consumes the JSON record emitted by this module's CLI.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

PROFILE_CHECK_SCHEMA = "tpen.profile-check/v1"
NATIVE_RECOVERY_PROFILE = "tpen.native-recovery/v1"
NATIVE_PROFILE_NAME = NATIVE_RECOVERY_PROFILE
REFUSAL_CODES = frozenset(
    {"unknown-profile", "unsupported-callback", "unsupported-logger", "malformed-config"}
)


@dataclass(frozen=True, slots=True)
class Profile:
    """One declared core recovery profile.

    Parameters
    ----------
    name : str
        Stable profile identifier.
    state_owners : frozenset of str
        Mutable state owners restored by the profile.
    callbacks, loggers : frozenset of str
        Explicitly allowlisted callback and logger targets.
    callback_state_supported : bool
        Whether callback-owned mutable state is restored by the profile.
    """

    name: str
    state_owners: frozenset[str]
    callbacks: frozenset[str]
    loggers: frozenset[str]
    callback_state_supported: bool

    @property
    def callback_targets(self) -> frozenset[str]:
        """Return the callback target allowlist."""

        return self.callbacks

    @property
    def logger_targets(self) -> frozenset[str]:
        """Return the logger target allowlist."""

        return self.loggers

    @property
    def callback_allowlist(self) -> frozenset[str]:
        """Return the callback allowlist under its declaration name."""

        return self.callbacks

    @property
    def logger_allowlist(self) -> frozenset[str]:
        """Return the logger allowlist under its declaration name."""

        return self.loggers

    def to_dict(self) -> dict[str, Any]:
        """Return the declared profile as a stable JSON-shaped mapping."""

        return {
            "name": self.name,
            "state_owners": sorted(self.state_owners),
            "callbacks": sorted(self.callbacks),
            "loggers": sorted(self.loggers),
            "callback_state_supported": self.callback_state_supported,
        }


SUPPORTED_CALLBACK_TARGETS = frozenset(
    {
        "tpen.callback.ConfigSnapshot",
        "tpen.callback.ResolvedConfigSnapshot",
        "tpen.callback.Metadata",
        "tpen.callback.RunTiming",
        "tpen.callback.TrainStepTiming",
        "tpen.callback.TrainPhaseTiming",
        "tpen.callback.Status",
        "tpen.callback.Checkpoint",
    }
)
SUPPORTED_LOGGER_TARGETS = frozenset({"tpen.logging.CSV", "tpen.logging.JSONL"})
SUPPORTED_STATE_OWNERS = frozenset(
    {"model", "optimizer", "trainer_cursor", "sampler_mcmc", "process_rng"}
)

NATIVE_PROFILE = Profile(
    name=NATIVE_RECOVERY_PROFILE,
    state_owners=SUPPORTED_STATE_OWNERS,
    callbacks=SUPPORTED_CALLBACK_TARGETS,
    loggers=SUPPORTED_LOGGER_TARGETS,
    callback_state_supported=False,
)
PROFILES: Mapping[str, Profile] = MappingProxyType({NATIVE_PROFILE.name: NATIVE_PROFILE})
PROFILE_REGISTRY = PROFILES
DECLARED_PROFILES = PROFILES
PROFILE_REFUSAL_CODES = REFUSAL_CODES


@dataclass(frozen=True, slots=True)
class ProfileCheck:
    """Immutable result of checking one declared profile against a config."""

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
        object.__setattr__(self, "state_owners", frozenset(self.state_owners))
        if self.refusal is None:
            if self.verdict == "refused":
                raise ValueError("refused profile check requires refusal")
            return
        if self.verdict != "refused":
            raise ValueError("supported profile check cannot contain refusal")
        if not isinstance(self.refusal, Mapping) or set(self.refusal) != {"code", "reason"}:
            raise TypeError("refusal must contain code and reason")
        if self.refusal["code"] not in REFUSAL_CODES or type(self.refusal["reason"]) is not str:
            raise ValueError("refusal has an invalid code or reason")
        object.__setattr__(self, "refusal", MappingProxyType(dict(self.refusal)))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation."""

        return {
            "schema": self.schema,
            "profile": self.profile,
            "verdict": self.verdict,
            "state_owners": sorted(self.state_owners),
            "refusal": None if self.refusal is None else dict(self.refusal),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "ProfileCheck":
        """Decode one strict v1 profile-check mapping."""

        if type(value) is not dict:
            raise TypeError("profile check must be a mapping")
        expected = {"schema", "profile", "verdict", "state_owners", "refusal"}
        if set(value) != expected or value["schema"] != cls.schema:
            raise ValueError("profile check has an unexpected schema or key set")
        if type(value["profile"]) is not str or not value["profile"]:
            raise TypeError("profile must be a non-empty string")
        owners = value["state_owners"]
        if isinstance(owners, (str, bytes, bytearray)) or not isinstance(owners, Sequence):
            raise TypeError("state_owners must be a sequence")
        if any(type(owner) is not str or not owner for owner in owners):
            raise TypeError("state_owners must contain non-empty strings")
        if len(set(owners)) != len(owners):
            raise ValueError("state_owners contains duplicates")
        return cls(value["profile"], value["verdict"], frozenset(owners), value["refusal"])

    def __hash__(self) -> int:
        """Hash the canonical JSON representation used by equality."""

        return hash(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")))


def _plain_config(config: Any) -> Any:
    """Copy an OmegaConf value without resolving or constructing targets."""

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=False)
    except (ImportError, TypeError, ValueError):
        return config
    return config


def _syntax_refusal(config: Any, profile: Profile) -> tuple[str, str] | None:
    """Find malformed syntax or the first disallowed mutable target."""

    if not isinstance(config, Mapping):
        return "malformed-config", "profile config must be a mapping"
    for owner, allowed, code in (
        ("callbacks", profile.callbacks, "unsupported-callback"),
        ("loggers", profile.loggers, "unsupported-logger"),
    ):
        entries = config.get(owner, ())
        if entries is None:
            entries = ()
        if isinstance(entries, (str, bytes, bytearray)) or not isinstance(entries, Sequence):
            return "malformed-config", f"{owner} must be a sequence"
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                return "malformed-config", f"{owner}[{index}] must be a mapping"
            target = entry.get("_target_")
            if type(target) is not str or not target:
                return "malformed-config", f"{owner}[{index}] has no explicit _target_"
            if target not in allowed:
                return code, f"{owner}[{index}] {target!r} is not in the declared profile"
    return None


def _check_refusal(config: Any, profile: Profile) -> tuple[str, str] | None:
    """Validate with the existing native-profile logic and classify refusal."""

    from tpen.execution.native_profile import validate_native_recovery_profile

    try:
        validate_native_recovery_profile(config)
    except Exception:
        return _syntax_refusal(_plain_config(config), profile) or (
            "malformed-config",
            "profile config failed native profile validation",
        )
    return _syntax_refusal(_plain_config(config), profile)


def check_profile(config: Any, profile_name: str) -> ProfileCheck:
    """Check unresolved configuration syntax against a declared profile.

    Parameters
    ----------
    config : mapping
        Unresolved OmegaConf-compatible configuration.  Targets are not
        imported or instantiated.
    profile_name : str
        Declared profile identifier.

    Returns
    -------
    ProfileCheck
        Supported or typed refusal result.  Refusal is a result, not an
        exception, so callers can make the capability gate explicit.
    """

    try:
        profile = PROFILES.get(profile_name)
    except TypeError:
        profile = None
    if profile is None:
        profile_value = profile_name if type(profile_name) is str and profile_name else "<unknown>"
        return ProfileCheck(
            profile_value,
            "refused",
            frozenset(),
            {"code": "unknown-profile", "reason": "profile is not declared by core"},
        )
    refusal = _check_refusal(config, profile)
    return ProfileCheck(
        profile.name,
        "refused" if refusal is not None else "supported",
        profile.state_owners,
        None if refusal is None else {"code": refusal[0], "reason": refusal[1]},
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args(argv)
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.load(args.config)
    except Exception as exc:  # CLI contract reports malformed input as a result.
        config = None
        reason = f"cannot load profile config: {exc}"
    else:
        reason = ""
    result = check_profile(config, args.profile)
    if reason and result.refusal is None:
        result = ProfileCheck(
            result.profile,
            "refused",
            result.state_owners,
            {"code": "malformed-config", "reason": reason},
        )
    print(json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests
    raise SystemExit(_main())


__all__ = [
    "NATIVE_PROFILE",
    "NATIVE_PROFILE_NAME",
    "NATIVE_RECOVERY_PROFILE",
    "DECLARED_PROFILES",
    "PROFILE_CHECK_SCHEMA",
    "PROFILE_REGISTRY",
    "PROFILES",
    "Profile",
    "ProfileCheck",
    "REFUSAL_CODES",
    "PROFILE_REFUSAL_CODES",
    "SUPPORTED_CALLBACK_TARGETS",
    "SUPPORTED_LOGGER_TARGETS",
    "SUPPORTED_STATE_OWNERS",
    "check_profile",
]
