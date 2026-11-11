"""Stable checkpoint payload profiles.

Payload profiles describe what a checkpoint contains and which restore intent
may consume it.  They are deliberately trainable-free and callback-free: a
checkpoint writer can use the same contract from a callback, a direct save,
or a future storage tool without importing event or training code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

PAYLOAD_MANIFEST_SCHEMA: Final = "tpen.checkpoint-payload/v1"
MODEL_ONLY_PROFILE: Final = "model_only"
TRAIN_RESUME_PROFILE: Final = "train_resume"

RestoreIntent: TypeAlias = Literal["model_only", "train_resume"]
PayloadProfile: TypeAlias = Literal["model_only", "train_resume"]

_PAYLOAD_COMPONENTS = ("model", "optimizer", "trainer", "sampler", "rng")
_TRAINER_PROGRESS = ("next_iteration", "completed_updates")


@dataclass(frozen=True, slots=True)
class CheckpointPayload:
    """Immutable contract for one checkpoint payload profile.

    Parameters
    ----------
    profile : {"model_only", "train_resume"}
        Stable profile name recorded in a checkpoint manifest.
    required_files : tuple of str
        Manifest component keys that must be present for this profile.
    required_state : tuple of str
        Named trainer state fields required for this profile.  These are
        intentionally distinct from manifest counters: the fields are read
        from ``trainer.json`` during a train resume, while manifest counters
        identify the published artifact.
    restore_intents : tuple of str
        Restore modes that may consume this payload.  A train-resume payload
        is a superset and may therefore also serve model-only evaluation.

    Notes
    -----
    The two concrete profiles below are the only payloads written by TPEN.
    Keeping this base object data-only makes its manifest stable and keeps
    callback scheduling outside the payload contract.
    """

    profile: PayloadProfile
    required_files: tuple[str, ...]
    required_state: tuple[str, ...]
    restore_intents: tuple[RestoreIntent, ...]

    def __post_init__(self) -> None:
        """Reject malformed profile definitions at the value boundary."""

        if not self.profile.strip():
            raise ValueError("payload profile must be non-empty")
        _validate_unique_nonempty(self.required_files, "required_files")
        _validate_unique_nonempty(self.required_state, "required_state", allow_empty=True)
        _validate_unique_nonempty(self.restore_intents, "restore_intents")

    @property
    def name(self) -> str:
        """Return the stable profile name."""

        return self.profile

    @property
    def required_components(self) -> tuple[str, ...]:
        """Alias for manifest component keys required by this payload."""

        return self.required_files

    @property
    def allowed_restore_modes(self) -> tuple[RestoreIntent, ...]:
        """Return restore intents admitted by this payload."""

        return self.restore_intents

    def validate_restore_intent(self, mode: RestoreIntent) -> None:
        """Ensure this payload is valid for the requested restore intent.

        Raises
        ------
        ValueError
            If the payload profile cannot satisfy ``mode``.
        """

        if mode not in self.restore_intents:
            raise ValueError(
                f"checkpoint payload profile {self.profile!r} cannot satisfy "
                f"restore mode {mode!r}; allowed modes: {list(self.restore_intents)!r}"
            )

    def validate_files(self, files: Mapping[str, str]) -> None:
        """Ensure a manifest names every file required by this profile."""

        missing = [name for name in self.required_files if not files.get(name)]
        if missing:
            raise ValueError(
                f"checkpoint payload profile {self.profile!r} is missing required "
                f"manifest files: {missing!r}"
            )

    def validate_state(self, state: Mapping[str, Any]) -> None:
        """Ensure trainer state carries valid fields required by this profile."""

        missing = [name for name in self.required_state if name not in state]
        if missing:
            raise ValueError(
                f"checkpoint payload profile {self.profile!r} is missing required "
                f"trainer state: {missing!r}"
            )
        for name in self.required_state:
            value = state[name]
            # JSON booleans pass an ``int`` annotation because bool subclasses
            # int.  Exact type comparison is intentional: progress counters
            # must be integer values, not bools or silently coerced strings.
            if type(value) is not int:
                raise ValueError(
                    f"checkpoint payload profile {self.profile!r} requires trainer "
                    f"state {name!r} to be an int"
                )
            if value < 0:
                raise ValueError(
                    f"checkpoint payload profile {self.profile!r} requires trainer "
                    f"state {name!r} to be non-negative"
                )

    def validate_save_flags(self, flags: Mapping[str, bool]) -> None:
        """Ensure save options exactly produce this profile's components."""

        for component in _PAYLOAD_COMPONENTS:
            expected = component in self.required_files
            if bool(flags[component]) != expected:
                raise ValueError(
                    f"payload profile {self.profile!r} requires save_{component}="
                    f"{expected}"
                )

    def to_manifest(self) -> dict[str, Any]:
        """Return the canonical, JSON-safe payload manifest mapping."""

        return {
            "schema": PAYLOAD_MANIFEST_SCHEMA,
            "profile": self.profile,
            "required_files": list(self.required_files),
            "required_state": list(self.required_state),
            "restore_intents": list(self.restore_intents),
        }

    @property
    def manifest(self) -> dict[str, Any]:
        """Return the canonical payload manifest as a fresh mapping."""

        return self.to_manifest()

    to_dict = to_manifest

    @classmethod
    def from_manifest(cls, data: Mapping[str, Any]) -> "CheckpointPayload":
        """Construct the canonical profile represented by a manifest."""

        if data.get("schema") != PAYLOAD_MANIFEST_SCHEMA:
            raise ValueError(
                f"unsupported checkpoint payload schema {data.get('schema')!r}; "
                f"expected {PAYLOAD_MANIFEST_SCHEMA!r}"
            )
        profile = data.get("profile")
        if profile == MODEL_ONLY_PROFILE:
            payload: CheckpointPayload = ModelOnly()
        elif profile == TRAIN_RESUME_PROFILE:
            payload = TrainResume()
        else:
            raise ValueError(f"unsupported checkpoint payload profile {profile!r}")

        if data != payload.to_manifest():
            raise ValueError(
                f"checkpoint payload manifest for {profile!r} is not canonical"
            )
        return payload

    @classmethod
    def for_profile(cls, profile: PayloadProfile) -> "CheckpointPayload":
        """Return the canonical payload for a stable profile name."""

        if profile == MODEL_ONLY_PROFILE:
            return ModelOnly()
        if profile == TRAIN_RESUME_PROFILE:
            return TrainResume()
        raise ValueError(f"unsupported checkpoint payload profile {profile!r}")

    @classmethod
    def model_only(cls) -> "CheckpointPayload":
        """Return the weights-only evaluation profile."""

        return ModelOnly()

    @classmethod
    def train_resume(cls) -> "CheckpointPayload":
        """Return the full train-resume profile."""

        return TrainResume()


class ModelOnly(CheckpointPayload):
    """Weights-only payload suitable for evaluation, never for resuming train."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            profile=MODEL_ONLY_PROFILE,
            required_files=("model",),
            required_state=(),
            restore_intents=("model_only",),
        )


class TrainResume(CheckpointPayload):
    """Full payload preserving optimizer, sampler, RNG, and progress state."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            profile=TRAIN_RESUME_PROFILE,
            required_files=_PAYLOAD_COMPONENTS,
            required_state=_TRAINER_PROGRESS,
            restore_intents=("model_only", "train_resume"),
        )


def _validate_unique_nonempty(
    values: tuple[str, ...], label: str, *, allow_empty: bool = False
) -> None:
    """Validate stable string sequences without runtime type duplication."""

    if not values and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} entries must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} entries must be unique")


MODEL_ONLY_PAYLOAD: Final[ModelOnly] = ModelOnly()
TRAIN_RESUME_PAYLOAD: Final[TrainResume] = TrainResume()


__all__ = [
    "MODEL_ONLY_PAYLOAD",
    "MODEL_ONLY_PROFILE",
    "PAYLOAD_MANIFEST_SCHEMA",
    "TRAIN_RESUME_PAYLOAD",
    "TRAIN_RESUME_PROFILE",
    "CheckpointPayload",
    "ModelOnly",
    "PayloadProfile",
    "RestoreIntent",
    "TrainResume",
]
