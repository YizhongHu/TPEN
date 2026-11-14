"""Core checkpoint-recovery selection and load-overlay contracts.

This module owns the small, backend-independent boundary between an immutable
scientific intent and the existing configured-run ``load`` mapping.  It does
not construct a model, sampler, optimizer, trainer, callback, or backend job.
Checkpoint acceptance remains delegated to :mod:`tpen.checkpoint`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from ._codec import decode_fields

from tpen.checkpoint.payload import TRAIN_RESUME_PAYLOAD
from tpen.checkpoint.artifact import require_complete_checkpoint_dir
from tpen.checkpoint.catalog import publication_catalog_path, read_publications
from tpen.checkpoint.receipt import has_publication_receipt, publication_receipt_path
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
from tpen.inspection import RUN_EVIDENCE_SCHEMA_V3, RunEvidence


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


class AdmissionRefusalCode(str, Enum):
    """Stable refusal reasons for successor continuation admission."""

    FORK = "fork"
    MISSING_PUBLICATION = "missing-publication"
    MODEL_ONLY = "model-only"
    STALE_REF = "stale-ref"
    CORRUPT_REF = "corrupt-ref"
    AMBIGUOUS_YIELD = "ambiguous-yield"
    NON_ELIGIBLE_OUTCOME = "non-eligible-outcome"
    IDENTITY_MISMATCH = "identity-mismatch"


@dataclass(frozen=True, slots=True)
class SuccessorAdmissionRefusal:
    """Typed refusal returned when a successor cannot be admitted."""

    code: AdmissionRefusalCode
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe refusal mapping."""

        return {"code": self.code.value, "message": self.message}


def _admission_text(value: Any, name: str) -> str:
    """Validate one non-empty text field in an admission request."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _admission_digest(value: Any, name: str) -> str:
    """Validate one lowercase SHA-256 digest in an admission request."""

    value = _admission_text(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _admission_identity(value: Any, name: str) -> dict[str, str]:
    """Validate the closed resolved-configuration identity mapping."""

    decode_fields(value, {"schema": str, "sha256": str}, name=name)
    if set(value) != {"schema", "sha256"}:
        raise ValueError(f"{name} must contain exactly schema and sha256")
    return {
        "schema": _admission_text(value["schema"], f"{name}.schema"),
        "sha256": _admission_digest(value["sha256"], f"{name}.sha256"),
    }


@dataclass(frozen=True, slots=True)
class SuccessorAdmissionPredecessor:
    """Immutable predecessor identity carried by a continuation request."""

    task_id: str
    attempt_id: str
    core_outcome_digest: str
    resolved_config_identity: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _admission_text(self.task_id, "predecessor.task_id"))
        object.__setattr__(self, "attempt_id", _admission_text(self.attempt_id, "predecessor.attempt_id"))
        object.__setattr__(
            self,
            "core_outcome_digest",
            _admission_digest(self.core_outcome_digest, "predecessor.core_outcome_digest"),
        )
        object.__setattr__(
            self,
            "resolved_config_identity",
            MappingProxyType(
                _admission_identity(
                    self.resolved_config_identity,
                    "predecessor.resolved_config_identity",
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized predecessor identity."""

        return {
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "core_outcome_digest": self.core_outcome_digest,
            "resolved_config_identity": dict(self.resolved_config_identity),
        }


@dataclass(frozen=True, slots=True)
class SuccessorAdmissionSuccessor:
    """Immutable reserved successor identity carried by a continuation request."""

    task_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _admission_text(self.task_id, "successor.task_id"))
        object.__setattr__(self, "attempt_id", _admission_text(self.attempt_id, "successor.attempt_id"))

    def to_dict(self) -> dict[str, str]:
        """Return the serialized successor identity."""

        return {"task_id": self.task_id, "attempt_id": self.attempt_id}


SUCCESSOR_ADMISSION_SCHEMA = "tpen.execution/successor-admission-request-v1"


@dataclass(frozen=True, slots=True)
class SuccessorAdmissionRequest:
    """Closed request to admit one typed planned-yield successor."""

    predecessor: SuccessorAdmissionPredecessor
    successor: SuccessorAdmissionSuccessor
    science_input_digest: str
    target_iterations: int
    core_evidence: RunEvidence
    schema: str = SUCCESSOR_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SUCCESSOR_ADMISSION_SCHEMA:
            raise ValueError(f"unsupported successor admission schema {self.schema!r}")
        if not isinstance(self.predecessor, SuccessorAdmissionPredecessor):
            raise TypeError("predecessor must be a SuccessorAdmissionPredecessor")
        if not isinstance(self.successor, SuccessorAdmissionSuccessor):
            raise TypeError("successor must be a SuccessorAdmissionSuccessor")
        object.__setattr__(
            self,
            "science_input_digest",
            _admission_digest(self.science_input_digest, "science_input_digest"),
        )
        object.__setattr__(
            self,
            "target_iterations",
            _require_nonnegative_int(self.target_iterations, "target_iterations"),
        )
        if not isinstance(self.core_evidence, RunEvidence):
            raise TypeError("core_evidence must be RunEvidence")

    @classmethod
    def from_mapping(cls, data: Any) -> "SuccessorAdmissionRequest":
        """Decode a closed successor-admission request mapping."""

        decode_fields(
            data,
            {
                "predecessor": Mapping,
                "successor": Mapping,
                "science_input_digest": str,
                "target_iterations": int,
                "core_evidence": Mapping,
            },
            schema=SUCCESSOR_ADMISSION_SCHEMA,
            name="successor admission request",
        )
        decode_fields(
            data["predecessor"],
            {
                "task_id": str,
                "attempt_id": str,
                "core_outcome_digest": str,
                "resolved_config_identity": Mapping,
            },
            name="successor admission predecessor",
        )
        decode_fields(
            data["successor"],
            {"task_id": str, "attempt_id": str},
            name="successor admission successor",
        )
        try:
            evidence = RunEvidence.from_mapping(data["core_evidence"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("core_evidence is not a valid run-evidence mapping") from exc
        return cls(
            predecessor=SuccessorAdmissionPredecessor(**dict(data["predecessor"])),
            successor=SuccessorAdmissionSuccessor(**dict(data["successor"])),
            science_input_digest=data["science_input_digest"],
            target_iterations=data["target_iterations"],
            core_evidence=evidence,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-shaped request mapping."""

        return {
            "schema": self.schema,
            "predecessor": self.predecessor.to_dict(),
            "successor": self.successor.to_dict(),
            "science_input_digest": self.science_input_digest,
            "target_iterations": self.target_iterations,
            "core_evidence": self.core_evidence.to_dict(),
        }

    def encode(self) -> bytes:
        """Return canonical UTF-8 JSON bytes for durable transport."""

        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical request mapping."""

        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def decode_successor_admission_request(
    data: bytes | str | Mapping[str, Any],
) -> SuccessorAdmissionRequest:
    """Decode one closed successor-admission request mapping or JSON payload."""

    if isinstance(data, (bytes, str)):
        try:
            data = json.loads(data)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("successor admission request is not valid JSON") from exc

    return SuccessorAdmissionRequest.from_mapping(data)


@dataclass(frozen=True, slots=True)
class SuccessorAdmissionResult:
    """Result of core admission, carrying either restore or refusal."""

    restore_request: "OperationalRestoreRequest | None" = None
    refusal: SuccessorAdmissionRefusal | None = None

    def __post_init__(self) -> None:
        if (self.restore_request is None) == (self.refusal is None):
            raise ValueError("admission result requires exactly one restore request or refusal")

    @property
    def accepted(self) -> bool:
        """Return whether a successor restore request was admitted."""

        return self.restore_request is not None

    @property
    def request(self) -> "OperationalRestoreRequest | None":
        """Return the admitted operational restore request, if any."""

        return self.restore_request


class SuccessorAdmissionError(ValueError):
    """Raised by the strict adapter when admission is refused."""

    def __init__(self, refusal: SuccessorAdmissionRefusal) -> None:
        super().__init__(f"{refusal.code.value}: {refusal.message}")
        self.refusal = refusal
        self.code = refusal.code


# These descriptive aliases keep the public boundary discoverable without
# introducing a second set of values for the same refusal codes.
SuccessorAdmissionRefusalCode = AdmissionRefusalCode
SuccessorAdmissionDecision = SuccessorAdmissionResult


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
    def from_mapping(cls, data: Any) -> "CheckpointEvidence":
        """Deserialize typed evidence and verify its CheckpointRef identity."""

        decode_fields(data, {
            "ref": Mapping, "science_input_digest": str, "target_iterations": int,
        }, optional={"committed": bool}, name="checkpoint evidence")
        try:
            return cls(
                ref=deserialize_checkpoint_ref(data["ref"]),
                science_input_digest=data["science_input_digest"],
                target_iterations=data["target_iterations"], committed=data.get("committed", True),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("malformed checkpoint evidence") from exc


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

    # No in-lane configured caller supplies an unbound path or ref for an
    # initial checkpoint.  Refuse those forms in both phases so this boundary
    # cannot self-stamp a checkpoint binding; durable evidence mappings remain
    # the explicit form for callers that need an initial checkpoint.
    if isinstance(evidence, (CheckpointRef, str, Path)):
        return _refused(
            intent,
            phase,
            RefusalCode.INCOMPATIBLE,
            "checkpoint selection requires evidence with a durable immutable binding",
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
    evidence: CheckpointEvidence | Mapping[str, Any],
    intent: ScienceIntent,
) -> CheckpointEvidence:
    """Normalize supported evidence forms while keeping binding explicit."""

    if isinstance(evidence, CheckpointEvidence):
        return evidence
    if isinstance(evidence, Mapping):
        try:
            return CheckpointEvidence.from_mapping(evidence)
        except (KeyError, TypeError, ValueError):
            return CheckpointEvidence(
                ref=None,
                science_input_digest=intent.science_input_digest,
                target_iterations=intent.target_iterations,
            )
    raise TypeError("evidence must be CheckpointEvidence or a durable evidence mapping")


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
    except (FileNotFoundError, OSError) as exc:
        return RecoveryRefusal(RefusalCode.INCOMPLETE, f"checkpoint manifest is incomplete: {exc}")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
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


@dataclass(frozen=True, slots=True)
class OperationalRestoreRequest:
    """Closed, versioned, immutable snapshot binding an E1 decision.

    Carries an already-selected full-state ``train_resume``
    :class:`RecoveryDecision`.  Construction freezes a validated snapshot of
    the decision's checkpoint reference and counters, derives the closed
    ``train_resume`` load shape directly from that reference (never trusting
    ``decision.load`` as a mutable dictionary), and revalidates the exact
    reference. Later mutation of the originating ``decision`` (for example
    ``decision.load["mode"] = "none"``) cannot change what this request
    carries, since every field consumed by :attr:`load` and :attr:`ref` is
    copied at construction time, not re-read from ``decision``.

    ``tpen.run.run_from_config`` consumes this at the existing ``Train``
    restore boundary (the constructor's ``load`` argument) and refuses a
    concrete configured ``runner.load`` that conflicts with it, rather than
    guessing precedence between the two.
    """

    SCHEMA: ClassVar[str] = "tpen.execution/operational-restore-request-v1"

    decision: RecoveryDecision
    ref: CheckpointRef = field(init=False)
    target_iterations: int = field(init=False)
    next_iteration: int = field(init=False)
    completed_updates: int = field(init=False)
    _load: Mapping[str, str] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, RecoveryDecision):
            raise TypeError("decision must be a RecoveryDecision")
        if self.decision.refusal is not None:
            raise ValueError("OperationalRestoreRequest requires a non-refused recovery decision")
        configured_load = self.decision.load
        if configured_load is None or dict(configured_load).get("mode") != "train_resume":
            raise ValueError("OperationalRestoreRequest requires a train_resume load decision")
        configured_load = dict(configured_load)
        if set(configured_load) != {"mode", "path"}:
            raise ValueError(
                "OperationalRestoreRequest rejects restore options beyond the "
                f"closed train_resume load shape: {sorted(configured_load)!r}"
            )
        ref = self.decision.checkpoint
        if not isinstance(ref, CheckpointRef):
            raise TypeError("decision must carry a validated CheckpointRef")
        valid_phase_outcome = (
            (self.decision.phase is RecoveryPhase.INITIAL and self.decision.outcome is RecoveryOutcome.INITIAL_CHECKPOINT)
            or (self.decision.phase is RecoveryPhase.RECOVERY and self.decision.outcome is RecoveryOutcome.RECOVERY)
        )
        if not valid_phase_outcome:
            raise ValueError(
                "OperationalRestoreRequest requires phase/outcome "
                f"{(RecoveryPhase.INITIAL, RecoveryOutcome.INITIAL_CHECKPOINT)!r} or "
                f"{(RecoveryPhase.RECOVERY, RecoveryOutcome.RECOVERY)!r}; got "
                f"({self.decision.phase!r}, {self.decision.outcome!r})"
            )
        # Revalidate the exact reference (not whatever ``decision.load`` claims)
        # before binding, so a request can never be built from a path the
        # decision's own checkpoint reference does not identify.
        ref.validate()
        read_manifest(ref.checkpoint_dir / "manifest.json", mode="train_resume")
        closed_load = train_resume_load_overlay(ref)
        if closed_load["path"] != configured_load["path"]:
            raise ValueError(
                "decision load path does not match decision.checkpoint's reference"
            )
        # The decision's counters are a copy made at selection time; require
        # them to still agree with the ref's own counters, and the ref to
        # remain within the fixed horizon (mirrors E1's gate at
        # ``_checkpoint_refusal``, recovery.py:517-521) rather than trusting
        # a decision whose counters could have been altered after selection.
        if self.decision.next_iteration != ref.next_iteration:
            raise ValueError(
                "decision.next_iteration does not match its checkpoint reference "
                f"({self.decision.next_iteration!r} != {ref.next_iteration!r})"
            )
        if self.decision.completed_updates != ref.completed_updates:
            raise ValueError(
                "decision.completed_updates does not match its checkpoint reference "
                f"({self.decision.completed_updates!r} != {ref.completed_updates!r})"
            )
        if ref.next_iteration > self.decision.target_iterations:
            raise ValueError(
                "checkpoint reference exceeds the fixed total target horizon "
                f"({ref.next_iteration!r} > {self.decision.target_iterations!r})"
            )
        if ref.completed_updates > self.decision.target_iterations:
            raise ValueError(
                "checkpoint reference exceeds the fixed total target horizon "
                f"({ref.completed_updates!r} > {self.decision.target_iterations!r})"
            )
        object.__setattr__(self, "ref", ref)
        object.__setattr__(
            self, "target_iterations",
            _require_nonnegative_int(self.decision.target_iterations, "target_iterations"),
        )
        object.__setattr__(
            self, "next_iteration",
            _require_nonnegative_int(self.decision.next_iteration, "next_iteration"),
        )
        object.__setattr__(
            self, "completed_updates",
            _require_nonnegative_int(self.decision.completed_updates, "completed_updates"),
        )
        object.__setattr__(self, "_load", MappingProxyType(dict(closed_load)))

    @property
    def load(self) -> dict[str, str]:
        """Return the closed ``train_resume`` load shape bound at construction."""

        return dict(self._load)


def _admission_refusal(code: AdmissionRefusalCode, message: str) -> SuccessorAdmissionResult:
    """Build a refused successor-admission result."""

    return SuccessorAdmissionResult(refusal=SuccessorAdmissionRefusal(code, message))


def _request_and_intent(
    first: ScienceIntent | SuccessorAdmissionRequest | Mapping[str, Any],
    second: ScienceIntent | SuccessorAdmissionRequest | Mapping[str, Any],
) -> tuple[ScienceIntent, SuccessorAdmissionRequest]:
    """Accept either intent-first or request-first adapter call order."""

    if isinstance(first, ScienceIntent):
        intent, request_value = first, second
    elif isinstance(second, ScienceIntent):
        intent, request_value = second, first
    else:
        raise TypeError("one admission adapter argument must be a ScienceIntent")
    request = (
        request_value
        if isinstance(request_value, SuccessorAdmissionRequest)
        else SuccessorAdmissionRequest.from_mapping(request_value)
    )
    return intent, request


def select_successor_admission(
    intent: ScienceIntent | SuccessorAdmissionRequest | Mapping[str, Any],
    request: ScienceIntent | SuccessorAdmissionRequest | Mapping[str, Any],
) -> SuccessorAdmissionResult:
    """Validate a successor request and select its existing restore boundary.

    Parameters
    ----------
    intent : ScienceIntent
        Immutable science identity supplied by the caller.
    request : SuccessorAdmissionRequest or mapping
        Closed request carrying the predecessor evidence and reserved successor.

    Returns
    -------
    SuccessorAdmissionResult
        Either a typed :class:`OperationalRestoreRequest` or one stable
        continuation-admission refusal.  This function does not launch work,
        create a successor directory, or write any ledger.
    """

    intent, request = _request_and_intent(intent, request)
    predecessor = request.predecessor
    successor = request.successor
    evidence = request.core_evidence

    if (
        evidence.schema == RUN_EVIDENCE_SCHEMA_V3
        and evidence.outcome == "planned_yield"
        and (evidence.checkpoint is None or evidence.checkpoint_ref is None)
    ):
        return _admission_refusal(
            AdmissionRefusalCode.AMBIGUOUS_YIELD,
            "planned-yield evidence does not carry both checkpoint bindings",
        )
    if (
        evidence.schema == RUN_EVIDENCE_SCHEMA_V3
        and evidence.checkpoint_ref is not None
        and evidence.checkpoint_ref.get("profile") == "model_only"
    ):
        return _admission_refusal(
            AdmissionRefusalCode.MODEL_ONLY,
            "successor admission requires a train_resume checkpoint profile",
        )
    if (
        evidence.schema != RUN_EVIDENCE_SCHEMA_V3
        or evidence.verdict != "incomplete"
        or evidence.run_status != "planned_yield"
        or evidence.refusal is not None
        or evidence.outcome != "planned_yield"
        or not evidence.continuation_eligible
    ):
        return _admission_refusal(
            AdmissionRefusalCode.NON_ELIGIBLE_OUTCOME,
            "core evidence is not a non-refused continuation-eligible planned yield",
        )

    if predecessor.task_id != successor.task_id:
        return _admission_refusal(
            AdmissionRefusalCode.IDENTITY_MISMATCH,
            "predecessor and successor task identities differ",
        )
    if predecessor.resolved_config_identity != evidence.resolved_config_identity:
        return _admission_refusal(
            AdmissionRefusalCode.IDENTITY_MISMATCH,
            "request config identity differs from core evidence",
        )
    if request.science_input_digest != intent.science_input_digest:
        return _admission_refusal(
            AdmissionRefusalCode.IDENTITY_MISMATCH,
            "request science input digest differs from the supplied intent",
        )
    if request.target_iterations != intent.target_iterations:
        return _admission_refusal(
            AdmissionRefusalCode.IDENTITY_MISMATCH,
            "request target horizon differs from the supplied intent",
        )

    ref_mapping = dict(evidence.checkpoint_ref)
    if ref_mapping.get("profile") != "train_resume":
        return _admission_refusal(
            AdmissionRefusalCode.MODEL_ONLY,
            "successor admission requires a train_resume checkpoint profile",
        )
    ref_fields = {key: value for key, value in ref_mapping.items() if key != "profile"}
    try:
        ref = CheckpointRef.from_mapping(ref_fields)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return _admission_refusal(
            AdmissionRefusalCode.CORRUPT_REF,
            f"checkpoint ref is malformed or has an invalid content_id: {exc}",
        )

    summary = evidence.checkpoint
    for key in ("checkpoint_dir", "next_iteration", "completed_updates"):
        if summary.get(key) != ref.to_dict()[key]:
            return _admission_refusal(
                AdmissionRefusalCode.IDENTITY_MISMATCH,
                f"core checkpoint summary disagrees with checkpoint_ref.{key}",
            )
    try:
        ref.validate()
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        return _admission_refusal(
            AdmissionRefusalCode.STALE_REF,
            f"checkpoint bytes no longer match checkpoint ref: {exc}",
        )
    if (
        ref.next_iteration > request.target_iterations
        or ref.completed_updates > request.target_iterations
    ):
        return _admission_refusal(
            AdmissionRefusalCode.IDENTITY_MISMATCH,
            "checkpoint counters exceed the fixed target horizon",
        )

    publication_root = ref.checkpoint_dir.parent
    try:
        publications = read_publications(publication_catalog_path(publication_root))
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        return _admission_refusal(
            AdmissionRefusalCode.MISSING_PUBLICATION,
            f"checkpoint publication catalog is unavailable or invalid: {exc}",
        )
    if not any(
        candidate.content_id == ref.content_id
        and candidate.checkpoint_dir.resolve() == ref.checkpoint_dir.resolve()
        for candidate in publications
    ):
        return _admission_refusal(
            AdmissionRefusalCode.MISSING_PUBLICATION,
            "checkpoint ref is not present in the publication catalog",
        )
    if not has_publication_receipt(
        publication_receipt_path(publication_root), ref.content_id
    ):
        return _admission_refusal(
            AdmissionRefusalCode.MISSING_PUBLICATION,
            "checkpoint ref has no complete publication receipt",
        )

    evidence_binding = CheckpointEvidence.from_ref(
        ref,
        science_input_digest=request.science_input_digest,
        target_iterations=request.target_iterations,
    )
    decision = select_recovery(
        intent,
        phase=RecoveryPhase.RECOVERY,
        evidence=evidence_binding,
    )
    if decision.refusal is not None:
        return _admission_refusal(
            AdmissionRefusalCode.AMBIGUOUS_YIELD,
            f"existing recovery selection refused the admitted checkpoint: {decision.refusal.message}",
        )
    try:
        restore_request = OperationalRestoreRequest(decision)
    except (TypeError, ValueError) as exc:
        return _admission_refusal(
            AdmissionRefusalCode.AMBIGUOUS_YIELD,
            f"existing operational restore boundary refused the admitted checkpoint: {exc}",
        )
    return SuccessorAdmissionResult(restore_request=restore_request)


def admit_successor(
    first: ScienceIntent | SuccessorAdmissionRequest | Mapping[str, Any],
    second: ScienceIntent | SuccessorAdmissionRequest | Mapping[str, Any],
) -> SuccessorAdmissionResult:
    """Alias for :func:`select_successor_admission`."""

    return select_successor_admission(first, second)


def operational_restore_request_from_admission(
    first: ScienceIntent | SuccessorAdmissionRequest | Mapping[str, Any],
    second: ScienceIntent | SuccessorAdmissionRequest | Mapping[str, Any],
) -> OperationalRestoreRequest:
    """Return the admitted restore request or raise its typed refusal."""

    result = select_successor_admission(first, second)
    if result.restore_request is None:
        assert result.refusal is not None
        raise SuccessorAdmissionError(result.refusal)
    return result.restore_request


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
    "AdmissionRefusalCode",
    "CheckpointEvidence",
    "OperationalRestoreRequest",
    "RecoveryDecision",
    "RecoveryOutcome",
    "RecoveryPhase",
    "RecoveryRefusal",
    "RefusalCode",
    "ScienceIntent",
    "SUCCESSOR_ADMISSION_SCHEMA",
    "SuccessorAdmissionError",
    "SuccessorAdmissionDecision",
    "SuccessorAdmissionPredecessor",
    "SuccessorAdmissionRefusal",
    "SuccessorAdmissionRefusalCode",
    "SuccessorAdmissionRequest",
    "SuccessorAdmissionResult",
    "SuccessorAdmissionSuccessor",
    "admit_successor",
    "capture_science_intent",
    "decode_successor_admission_request",
    "operational_restore_request_from_admission",
    "select_successor_admission",
    "select_recovery",
    "train_resume_load_overlay",
]
