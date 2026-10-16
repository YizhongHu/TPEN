"""Core checkpoint-recovery selection and load-overlay contracts.

This module owns the small, backend-independent boundary between an immutable
scientific intent and the existing configured-run ``load`` mapping.  It does
not construct a model, sampler, optimizer, trainer, callback, or backend job.
Checkpoint acceptance remains delegated to :mod:`tpen.checkpoint`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tpen.checkpoint.payload import TRAIN_RESUME_PAYLOAD
from tpen.checkpoint.artifact import require_complete_checkpoint_dir
from tpen.checkpoint.reference import (
    CheckpointRef,
    deserialize_checkpoint_ref,
    serialize_checkpoint_ref,
)
from tpen.checkpoint.schema import (
    KIND_BY_SCHEMA_VERSION,
    SCHEMA_VERSIONS_BY_RESTORE_MODE,
    read_manifest,
)
from tpen.checkpoint.hashing import stable_config_hash


class RecoveryPhase(str, Enum):
    """Invocation phase whose checkpoint-selection policy is being applied."""

    INITIAL = "initial"
    RECOVERY = "recovery"


class RecoveryOutcome(str, Enum):
    """Typed result of selecting the initial or recovery load path."""

    INITIAL = "initial"
    INITIAL_CHECKPOINT = "initial_checkpoint"
    RECOVERY = "recovery"
    NO_CHECKPOINT = "no_checkpoint"
    REFUSED = "refused"


class RefusalCode(str, Enum):
    """Typed refusal reasons exposed to later execution layers."""

    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"
    MODEL_ONLY = "model_only"
    INCOMPATIBLE = "incompatible"
    NO_CHECKPOINT = "no_checkpoint"


def _freeze(value: Any, *, path: str = "value") -> Any:
    """Deep-freeze JSON-shaped configuration without resolving constructors."""

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            frozen[key] = _freeze(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{path} must be JSON-shaped, got {type(value).__name__}")


def _thaw(value: Any) -> Any:
    """Return a JSON-shaped mutable copy for durable serialization."""

    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _capture_config(config: Any) -> Any:
    """Copy config syntax without invoking Hydra or resolving ``_target_``."""

    # ``resolve=False`` is intentional.  OmegaConf interpolation resolution is
    # not science construction, but source capture must preserve the submitter
    # syntax and must never call Hydra's arbitrary-object resolver.
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=False)
    except ImportError:
        pass
    return _freeze(config, path="source_config")


def _require_nonnegative_int(value: Any, name: str) -> int:
    """Validate an explicit integer contract field without bool coercion."""

    if type(value) is not int:
        raise TypeError(f"{name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class ScienceIntent:
    """Immutable science input and fixed total target horizon.

    Parameters
    ----------
    science_input : mapping
        JSON-shaped, immutable scientific identity.  It is hashed for binding
        checkpoint evidence and is never instantiated as a science object.
    target_iterations : int
        Fixed total iteration horizon for every invocation in this lineage.
    source_config : mapping, optional
        Captured source configuration syntax.  ``_target_`` values remain
        opaque strings and are never resolved here.
    overrides : sequence of str, optional
        Captured command-line override strings in their original order.
    """

    science_input: Mapping[str, Any]
    target_iterations: int
    source_config: Mapping[str, Any] = field(default_factory=dict)
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "science_input", _freeze(self.science_input, path="science_input"))
        object.__setattr__(self, "target_iterations", _require_nonnegative_int(self.target_iterations, "target_iterations"))
        object.__setattr__(self, "source_config", _capture_config(self.source_config))
        if isinstance(self.overrides, str):
            overrides = (self.overrides,)
        else:
            overrides = tuple(self.overrides)
        if any(not isinstance(value, str) for value in overrides):
            raise TypeError("overrides must contain only strings")
        object.__setattr__(self, "overrides", overrides)

    @property
    def science_input_digest(self) -> str:
        """Return the stable digest used to bind checkpoint evidence."""

        return stable_config_hash(self.science_input)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe immutable-intent representation."""

        return {
            "science_input": _thaw(self.science_input),
            "science_input_digest": self.science_input_digest,
            "target_iterations": self.target_iterations,
            "source_config": _thaw(self.source_config),
            "overrides": list(self.overrides),
        }


def capture_science_intent(
    science_input: Mapping[str, Any],
    *,
    target_iterations: int,
    source_config: Any = None,
    overrides: Sequence[str] = (),
) -> ScienceIntent:
    """Capture source syntax and immutable intent without science instantiation.

    The function deliberately accepts config-shaped data rather than a Hydra
    config object with an instantiation request.  It performs no import or
    resolution of arbitrary ``_target_`` values.
    """

    return ScienceIntent(
        science_input=science_input,
        target_iterations=target_iterations,
        source_config={} if source_config is None else source_config,
        overrides=overrides,
    )


@dataclass(frozen=True, slots=True)
class CheckpointEvidence:
    """Checkpoint reference plus the immutable intent binding it claims."""

    ref: CheckpointRef | None
    science_input_digest: str
    target_iterations: int
    committed: bool = True
    source: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.science_input_digest, str) or len(self.science_input_digest) != 64:
            raise ValueError("science_input_digest must be a 64-character digest")
        _require_nonnegative_int(self.target_iterations, "target_iterations")
        if self.source is not None:
            object.__setattr__(self, "source", Path(self.source))

    @classmethod
    def from_ref(
        cls,
        ref: CheckpointRef,
        *,
        science_input: Mapping[str, Any] | None = None,
        science_input_digest: str | None = None,
        target_iterations: int,
    ) -> "CheckpointEvidence":
        """Bind a validated reference to an immutable input and target."""

        if not isinstance(ref, CheckpointRef):
            raise TypeError("ref must be a CheckpointRef")
        if (science_input is None) == (science_input_digest is None):
            raise TypeError("provide exactly one of science_input or science_input_digest")
        digest = (
            stable_config_hash(science_input)
            if science_input is not None
            else str(science_input_digest)
        )
        return cls(ref=ref, science_input_digest=digest, target_iterations=target_iterations)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        science_input: Mapping[str, Any] | None = None,
        science_input_digest: str | None = None,
        target_iterations: int,
    ) -> "CheckpointEvidence":
        """Capture a path as evidence, retaining an invalid path for refusal."""

        try:
            ref = CheckpointRef.from_directory(path)
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            ref = None
        if (science_input is None) == (science_input_digest is None):
            raise TypeError("provide exactly one of science_input or science_input_digest")
        digest = (
            stable_config_hash(science_input)
            if science_input is not None
            else str(science_input_digest)
        )
        return cls(
            ref=ref,
            science_input_digest=digest,
            target_iterations=target_iterations,
            source=Path(path),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe durable evidence record."""

        if self.ref is None:
            raise ValueError("cannot serialize evidence without a CheckpointRef")
        return {
            "ref": serialize_checkpoint_ref(self.ref),
            "science_input_digest": self.science_input_digest,
            "target_iterations": self.target_iterations,
            "committed": self.committed,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CheckpointEvidence":
        """Deserialize evidence and verify the embedded CheckpointRef identity."""

        return cls(
            ref=deserialize_checkpoint_ref(data["ref"]),
            science_input_digest=str(data["science_input_digest"]),
            target_iterations=data["target_iterations"],
            committed=bool(data.get("committed", True)),
        )


@dataclass(frozen=True, slots=True)
class RecoveryRefusal:
    """Typed refusal that lets later layers park without retrying blindly."""

    code: RefusalCode
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe refusal record."""

        return {"code": self.code.value, "message": self.message}


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Selection result and the exact existing-run load overlay, if any."""

    phase: RecoveryPhase
    outcome: RecoveryOutcome
    target_iterations: int
    next_iteration: int = 0
    completed_updates: int = 0
    checkpoint: CheckpointRef | None = None
    load: dict[str, str] | None = None
    refusal: RecoveryRefusal | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe decision record."""

        return {
            "phase": self.phase.value,
            "outcome": self.outcome.value,
            "target_iterations": self.target_iterations,
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
            "checkpoint": None if self.checkpoint is None else self.checkpoint.to_dict(),
            "load": None if self.load is None else dict(self.load),
            "refusal": None if self.refusal is None else self.refusal.to_dict(),
        }


def train_resume_load_overlay(ref: CheckpointRef) -> dict[str, str]:
    """Return the existing configured-run ``train_resume`` load shape.

    No additional execution fields are inserted: ``tpen.runner.Train`` and
    ``tpen.checkpoint.restore`` continue to consume the exact mapping already
    accepted by configured runs.
    """

    if not isinstance(ref, CheckpointRef):
        raise TypeError("ref must be a CheckpointRef")
    return {"mode": "train_resume", "path": str(ref.checkpoint_dir)}


def select_recovery(
    intent: ScienceIntent,
    *,
    phase: RecoveryPhase | str,
    evidence: CheckpointEvidence | CheckpointRef | Mapping[str, Any] | str | Path | None = None,
) -> RecoveryDecision:
    """Select initial or recovery state without constructing science objects.

    Initial selection with no evidence starts from immutable intent. Recovery
    selection with no evidence returns the explicit ``no_checkpoint`` outcome;
    it never silently becomes an initial run. A supplied reference must be a
    committed, complete, full-state checkpoint bound to the same input digest
    and fixed target horizon.
    """

    phase = RecoveryPhase(phase)
    if not isinstance(intent, ScienceIntent):
        raise TypeError("intent must be a ScienceIntent")
    if evidence is None:
        if phase is RecoveryPhase.INITIAL:
            return RecoveryDecision(
                phase=phase,
                outcome=RecoveryOutcome.INITIAL,
                target_iterations=intent.target_iterations,
            )
        return _refused(
            intent,
            phase,
            RefusalCode.NO_CHECKPOINT,
            "no committed checkpoint is available for recovery; park this intent",
            outcome=RecoveryOutcome.NO_CHECKPOINT,
        )

    if phase is RecoveryPhase.RECOVERY and isinstance(evidence, (CheckpointRef, str, Path)):
        return _refused(
            intent,
            phase,
            RefusalCode.INCOMPATIBLE,
            "recovery requires checkpoint evidence with a durable immutable binding",
        )

    evidence = _coerce_evidence(evidence, intent)
    if not evidence.committed:
        return _refused(intent, phase, RefusalCode.INCOMPLETE, "checkpoint evidence is not committed")
    if evidence.science_input_digest != intent.science_input_digest:
        return _refused(
            intent,
            phase,
            RefusalCode.INCOMPATIBLE,
            "checkpoint evidence is bound to different immutable science input",
        )
    if evidence.target_iterations != intent.target_iterations:
        return _refused(
            intent,
            phase,
            RefusalCode.INCOMPATIBLE,
            "checkpoint evidence uses a different fixed target horizon",
        )
    if evidence.ref is None:
        source_refusal = _source_manifest_refusal(evidence.source)
        if source_refusal is not None:
            return _refused(intent, phase, source_refusal.code, source_refusal.message)
        return _refused(
            intent,
            phase,
            RefusalCode.INCOMPLETE,
            "checkpoint evidence does not identify a committed complete checkpoint",
        )

    refusal = _checkpoint_refusal(evidence.ref, target_iterations=intent.target_iterations)
    if refusal is not None:
        return _refused(intent, phase, refusal.code, refusal.message)

    ref = evidence.ref
    outcome = (
        RecoveryOutcome.INITIAL_CHECKPOINT
        if phase is RecoveryPhase.INITIAL
        else RecoveryOutcome.RECOVERY
    )
    return RecoveryDecision(
        phase=phase,
        outcome=outcome,
        target_iterations=intent.target_iterations,
        next_iteration=ref.next_iteration,
        completed_updates=ref.completed_updates,
        checkpoint=ref,
        load=train_resume_load_overlay(ref),
    )


def _coerce_evidence(
    evidence: CheckpointEvidence | CheckpointRef | Mapping[str, Any] | str | Path,
    intent: ScienceIntent,
) -> CheckpointEvidence:
    """Normalize supported evidence forms while keeping binding explicit."""

    if isinstance(evidence, CheckpointEvidence):
        return evidence
    if isinstance(evidence, CheckpointRef):
        # A direct ref supplied to this function is bound to the current intent
        # at this boundary. Durable callers should prefer CheckpointEvidence so
        # the binding is preserved across process starts.
        return CheckpointEvidence.from_ref(
            evidence,
            science_input_digest=intent.science_input_digest,
            target_iterations=intent.target_iterations,
        )
    if isinstance(evidence, Mapping):
        try:
            return CheckpointEvidence.from_mapping(evidence)
        except (KeyError, TypeError, ValueError):
            return CheckpointEvidence(
                ref=None,
                science_input_digest=intent.science_input_digest,
                target_iterations=intent.target_iterations,
            )
    return CheckpointEvidence.from_path(
        evidence,
        science_input_digest=intent.science_input_digest,
        target_iterations=intent.target_iterations,
    )


def _checkpoint_refusal(ref: CheckpointRef, *, target_iterations: int) -> RecoveryRefusal | None:
    """Apply core completeness and train-resume schema gates to one ref."""

    try:
        checkpoint_dir = require_complete_checkpoint_dir(ref.checkpoint_dir)
        with (checkpoint_dir / "manifest.json").open("r", encoding="utf-8") as handle:
            raw_manifest = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return RecoveryRefusal(RefusalCode.INCOMPLETE, f"checkpoint is not complete: {exc}")

    # Inspect the raw version before CheckpointRef validation so a committed
    # legacy artifact gets the typed ``unsupported`` outcome rather than being
    # collapsed into generic incompleteness. The schema module remains the
    # authority for the actual mode gate below.
    raw_refusal = _raw_manifest_refusal(raw_manifest)
    if raw_refusal is not None:
        return raw_refusal

    try:
        ref.validate()
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        return RecoveryRefusal(RefusalCode.INCOMPLETE, f"checkpoint bytes are not a committed ref: {exc}")

    manifest_path = checkpoint_dir / "manifest.json"
    try:
        model_manifest = read_manifest(manifest_path, mode="model_only")
    except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return RecoveryRefusal(RefusalCode.UNSUPPORTED, f"checkpoint manifest is unsupported: {exc}")

    profile = None
    if model_manifest.payload is not None:
        profile = model_manifest.payload.get("profile")
    if profile == "model_only":
        return RecoveryRefusal(
            RefusalCode.MODEL_ONLY,
            "model-only checkpoint evidence cannot satisfy train_resume recovery",
        )
    try:
        read_manifest(manifest_path, mode="train_resume")
    except ValueError as exc:
        return RecoveryRefusal(RefusalCode.UNSUPPORTED, f"checkpoint cannot satisfy train_resume: {exc}")
    except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return RecoveryRefusal(RefusalCode.INCOMPLETE, f"checkpoint manifest is incomplete: {exc}")

    missing = [
        name
        for name in TRAIN_RESUME_PAYLOAD.required_files
        if not (ref.checkpoint_dir / str(model_manifest.files.get(name, ""))).is_file()
    ]
    if missing:
        return RecoveryRefusal(
            RefusalCode.INCOMPLETE,
            f"full-state checkpoint is missing committed files: {missing!r}",
        )
    if ref.next_iteration > target_iterations or ref.completed_updates > target_iterations:
        return RecoveryRefusal(
            RefusalCode.INCOMPATIBLE,
            "checkpoint counters exceed the fixed total target horizon",
        )
    return None


def _source_manifest_refusal(source: Path | None) -> RecoveryRefusal | None:
    """Classify a source manifest when no immutable ref could be built."""

    if source is None:
        return None
    try:
        checkpoint_dir = require_complete_checkpoint_dir(source)
        with (checkpoint_dir / "manifest.json").open("r", encoding="utf-8") as handle:
            raw_manifest = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return RecoveryRefusal(RefusalCode.INCOMPLETE, f"checkpoint is not complete: {exc}")
    return _raw_manifest_refusal(raw_manifest)


def _raw_manifest_refusal(raw_manifest: Any) -> RecoveryRefusal | None:
    """Classify raw manifest schema metadata without allowing malformed values to escape."""

    if not isinstance(raw_manifest, Mapping):
        return RecoveryRefusal(RefusalCode.UNSUPPORTED, "checkpoint manifest is not a mapping")
    schema_version = raw_manifest.get("schema_version")
    if type(schema_version) is not int:
        return RecoveryRefusal(
            RefusalCode.UNSUPPORTED,
            "checkpoint manifest schema_version is not an integer",
        )
    accepted_versions = SCHEMA_VERSIONS_BY_RESTORE_MODE["train_resume"]
    if schema_version not in accepted_versions or raw_manifest.get("kind") != KIND_BY_SCHEMA_VERSION.get(schema_version):
        return RecoveryRefusal(
            RefusalCode.UNSUPPORTED,
            "checkpoint manifest is not the current train-resume schema",
        )
    return None


def _refused(
    intent: ScienceIntent,
    phase: RecoveryPhase,
    code: RefusalCode,
    message: str,
    *,
    outcome: RecoveryOutcome = RecoveryOutcome.REFUSED,
) -> RecoveryDecision:
    """Build a refusal decision with no load overlay."""

    return RecoveryDecision(
        phase=phase,
        outcome=outcome,
        target_iterations=intent.target_iterations,
        refusal=RecoveryRefusal(code=code, message=message),
    )


__all__ = [
    "CheckpointEvidence",
    "RecoveryDecision",
    "RecoveryOutcome",
    "RecoveryPhase",
    "RecoveryRefusal",
    "RefusalCode",
    "ScienceIntent",
    "capture_science_intent",
    "select_recovery",
    "train_resume_load_overlay",
]
