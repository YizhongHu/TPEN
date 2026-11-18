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
from typing import Any, Callable, ClassVar

PROFILE_CHECK_SCHEMA = "tpen.profile-check/v1"
NATIVE_RECOVERY_PROFILE = "tpen.native-recovery/v1"
NATIVE_PROFILE_NAME = NATIVE_RECOVERY_PROFILE
REFUSAL_CODES = frozenset(
    {
        "unknown-profile",
        "unsupported-callback",
        "unsupported-callback-cadence",
        "unsupported-logger",
        "malformed-config",
    }
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


OBSERVATIONAL_CALLBACK_TARGETS = frozenset(
    {
        "tpen.callback.ConfigSnapshot",
        "tpen.callback.ResolvedConfigSnapshot",
        "tpen.callback.Metadata",
        "tpen.callback.RunTiming",
        "tpen.callback.TrainStepTiming",
        "tpen.callback.TrainPhaseTiming",
        "tpen.callback.Status",
        "tpen.callback.ElectronNucleusDistanceQuantiles",
        "tpen.callback.ModuleNorms",
        "tpen.callback.OptimizerState",
    }
)
# These callbacks are admitted explicitly because their resume behavior is
# part of the callback's contract.  This is deliberately not inferred from
# ``StatefulCallback``: that generic describes the domain state a callback
# observes, not callback-owned state restored by ``train_resume``.
#
# #614 audit for OptimizerState: its state source is ``TrainerState.update_state``
# and the live model's Adam/AdamW carrier. It is observational: validation can
# reject an incompatible state, but the callback has no run-aborting policy of
# its own. Its ``StepCadenceGate`` (including ``num_calls`` and its RNG) is
# callback-owned and not restored by ``train_resume``, but it gates only metric
# logging, so cadence remains admitted.
# #614 audit for SamplerHealth: its state source is the typed
# ``TrainerState.sampler_stats`` record. ``fail_fast`` can abort the run, so it
# belongs with the resume-sensitive health checks. Its callback-owned
# ``StepCadenceGate`` (including ``num_calls`` and its RNG) is not restored;
# ``max_calls`` or ``probability < 1`` must therefore refuse.
# #614 audit for FactorScalars: its state source is each model factor's
# ``scalar_diagnostics`` contract. Its ``fail_fast`` path can also abort the
# run, so it is resume-sensitive. Its callback-owned ``StepCadenceGate``
# (including ``num_calls`` and its RNG) is not restored; the same
# ``max_calls``/``probability < 1`` refusal applies.
# ``InitialCheckpoint`` (tpen/callback/initial_checkpoint.py) and
# ``ParameterActivity`` (tpen/callback/parameter_activity.py) each become a
# VERIFIED no-op on a resumed lineage at step > 0: each checks that its own
# prior artifact already exists and is well-formed at the same configured
# path, and refuses rather than writing late (or mislabeling a later
# update's data as the lineage's first) when it does not. Neither consumes
# any ``pop_step_cadence`` field, so neither is cadence-sensitive.
RESUME_SENSITIVE_CALLBACK_TARGETS = frozenset(
    {
        "tpen.callback.Checkpoint",
        "tpen.callback.DataIntegrity",
        "tpen.callback.GradientStats",
        "tpen.callback.AmplitudeRiskFraction",
        "tpen.callback.InitialCheckpoint",
        "tpen.callback.ParameterActivity",
        "tpen.callback.SamplerHealth",
        "tpen.callback.FactorScalars",
        "tpen.callback.RawTrainingBatchRecorder",
        "tpen.callback.TrainingBatchRecorder",
    }
)
# ``pop_step_cadence`` consumes five fields from these callbacks:
# ``every_n_steps``, ``start_step``, ``max_calls``, ``probability``, and
# ``seed``. The first two are immutable step-window values. ``seed`` only
# initializes the local RNG, which is never consumed when probability is one.
# ``max_calls`` and a probability below one are different: they advance the
# callback-local call counter or RNG, neither of which train_resume restores.
# Checkpoint consumes these fields at ``tpen/callback/checkpoint.py:162`` and
# uses the resulting gate for periodic publication at line 208, so its
# published-checkpoint set is recovery-sensitive. Status consumes the same
# fields at ``tpen/callback/status.py:127``, but gates only its optional
# training-line logging (lines 133-140). TrainStepTiming consumes them at
# ``tpen/callback/timing/train_step_timing.py:41-57`` and TrainPhaseTiming at
# ``tpen/callback/timing/train_phase_timing.py:64-83``; both gate only
# observational ``train/perf`` logging. Those three observational callbacks
# therefore remain admitted. OptimizerState validates its update method at
# iteration entry, then uses its non-restored gate only for completed-update
# optimizer metrics; it therefore remains admitted despite owning cadence
# state. SamplerHealth and FactorScalars are listed above because their same
# gate can suppress a configured run-aborting check. AmplitudeRiskFraction
# consumes the same five fields at ``tpen/callback/amplitude_risk.py`` (via
# ``pop_step_cadence``) and gates which completed-iteration batches get a
# published descriptive occupancy fraction; a gap or duplicate in that
# published set is exactly the resume-fidelity failure this allowlist exists
# to catch, so it is held to the same cadence-sensitive standard as
# DataIntegrity and GradientStats rather than the three purely-timing
# callbacks above.
CALLBACK_CADENCE_REFUSAL_CODE = "unsupported-callback-cadence"
CADENCE_SENSITIVE_CALLBACK_TARGETS = frozenset(
    {
        "tpen.callback.Checkpoint",
        "tpen.callback.DataIntegrity",
        "tpen.callback.GradientStats",
        "tpen.callback.AmplitudeRiskFraction",
        "tpen.callback.SamplerHealth",
        "tpen.callback.FactorScalars",
        "tpen.callback.RawTrainingBatchRecorder",
        "tpen.callback.TrainingBatchRecorder",
        "tpen.callback.RuntimeEquivariance",
    }
)
# These schedules make decisions only from the durable completed-update
# coordinate.  A CheckpointSchedule is a protocol, so an arbitrary Hydra
# target is not evidence that its private counters or RNG are reconstructed by
# train_resume.  Keep the admission boundary explicit until such a schedule
# supplies its own recovery contract.
AUDITED_CHECKPOINT_SCHEDULE_TARGETS = frozenset(
    {
        "tpen.checkpoint.TerminalOnly",
        "tpen.checkpoint.EveryNUpdates",
        "tpen.checkpoint.ExplicitUpdates",
    }
)
RECONSTRUCTED_CALLBACK_TARGETS = frozenset({"tpen.callback.RuntimeEquivariance"})
ALLOCATION_LOCAL_CALLBACK_TARGETS = frozenset({"tpen.callback.ResourceUsage"})
SUPPORTED_CALLBACK_TARGETS = (
    OBSERVATIONAL_CALLBACK_TARGETS
    | RESUME_SENSITIVE_CALLBACK_TARGETS
    | RECONSTRUCTED_CALLBACK_TARGETS
    | ALLOCATION_LOCAL_CALLBACK_TARGETS
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


_RUNTIME_EQUIVARIANCE_TARGET = "tpen.callback.RuntimeEquivariance"
_FULL_EQUIVARIANCE_CHECKER_TARGET = "tpen.equivariance.checks.FullModelEquivarianceChecker"
_TRACE_EQUIVARIANCE_CHECKER_TARGET = "tpen.equivariance.checks.TraceEquivarianceChecker"
_RESOURCE_USAGE_TARGET = "tpen.callback.ResourceUsage"
_TORCH_ALLOCATOR_PROBE_TARGET = "tpen.accelerator.TorchAllocatorPeakProbe"


def _is_unresolved_scalar(value: Any) -> bool:
    """Return whether ``value`` is a literal scalar or unresolved interpolation."""

    if type(value) in (bool, int, float):
        return True
    return (
        type(value) is str
        and value.startswith("${")
        and value.endswith("}")
    )


def _numeric_scalar(value: Any, *, integer: bool = False) -> bool:
    """Validate a numeric constructor scalar without resolving Hydra values."""

    if not _is_unresolved_scalar(value):
        return False
    if isinstance(value, str):
        return True
    if integer:
        return type(value) is int
    return type(value) in (int, float)


def _runtime_equivariance_contract_refusal(
    entry: Mapping[str, Any], path: str
) -> tuple[str, str] | None:
    """Refuse RuntimeEquivariance constructors outside the audited HI shape."""

    expected = {
        "_target_",
        "every_n_steps",
        "start_step",
        "max_calls",
        "probability",
        "seed",
        "fail_fast",
        "artifact_dir",
        "checkers",
    }
    unknown = set(entry) - expected
    if unknown:
        return (
            "unsupported-callback",
            f"{path} {_RUNTIME_EQUIVARIANCE_TARGET!r} has unknown constructor "
            f"field(s): {sorted(unknown)!r}",
        )
    if entry.get("fail_fast") is not True:
        return (
            "unsupported-callback",
            f"{path} {_RUNTIME_EQUIVARIANCE_TARGET!r} requires fail_fast=true",
        )
    if "artifact_dir" not in entry or not isinstance(entry["artifact_dir"], str) or not entry["artifact_dir"]:
        return (
            "unsupported-callback",
            f"{path} {_RUNTIME_EQUIVARIANCE_TARGET!r} requires a routed artifact_dir",
        )
    if "every_n_steps" in entry and not _numeric_scalar(entry["every_n_steps"], integer=True):
        return (
            "unsupported-callback",
            f"{path}.every_n_steps must be an integer or unresolved interpolation",
        )
    if "every_n_steps" in entry and not isinstance(entry["every_n_steps"], str) and entry["every_n_steps"] < 1:
        return (
            "unsupported-callback",
            f"{path}.every_n_steps must be positive",
        )
    if "start_step" in entry and not _numeric_scalar(entry["start_step"], integer=True):
        return (
            "unsupported-callback",
            f"{path}.start_step must be an integer or unresolved interpolation",
        )
    if "start_step" in entry and not isinstance(entry["start_step"], str) and entry["start_step"] < 0:
        return (
            "unsupported-callback",
            f"{path}.start_step must be nonnegative",
        )
    if "seed" in entry and entry["seed"] is not None and not _numeric_scalar(
        entry["seed"], integer=True
    ):
        return (
            "unsupported-callback",
            f"{path}.seed must be an integer, null, or unresolved interpolation",
        )
    checkers = entry.get("checkers")
    if not isinstance(checkers, (list, tuple)) or len(checkers) != 2:
        return (
            "unsupported-callback",
            f"{path}.checkers must contain exactly the ordered full-model and trace checkers",
        )
    expected_targets = (_FULL_EQUIVARIANCE_CHECKER_TARGET, _TRACE_EQUIVARIANCE_CHECKER_TARGET)
    for index, (checker, expected_target) in enumerate(zip(checkers, expected_targets, strict=True)):
        checker_path = f"{path}.checkers[{index}]"
        if not isinstance(checker, Mapping) or checker.get("_target_") != expected_target:
            return (
                "unsupported-callback",
                f"{checker_path} must use the exact audited target {expected_target!r}",
            )
        checker_expected = {
            "_target_",
            "permutation_fraction",
            "max_permutations",
            "seed",
            "atol",
            "rtol",
        }
        if expected_target == _TRACE_EQUIVARIANCE_CHECKER_TARGET:
            checker_expected.add("compare_output")
        unknown_checker = set(checker) - checker_expected
        if unknown_checker:
            return (
                "unsupported-callback",
                f"{checker_path} has unknown constructor field(s): {sorted(unknown_checker)!r}",
            )
        if not _numeric_scalar(checker.get("permutation_fraction")):
            return (
                "unsupported-callback",
                f"{checker_path}.permutation_fraction must be an audited numeric scalar",
            )
        if not isinstance(checker["permutation_fraction"], str) and not 0 <= checker[
            "permutation_fraction"
        ] <= 1:
            return (
                "unsupported-callback",
                f"{checker_path}.permutation_fraction must be between 0 and 1",
            )
        if not _numeric_scalar(checker.get("max_permutations"), integer=True):
            return (
                "unsupported-callback",
                f"{checker_path}.max_permutations must be an audited integer scalar",
            )
        if not isinstance(checker["max_permutations"], str) and checker["max_permutations"] < 1:
            return (
                "unsupported-callback",
                f"{checker_path}.max_permutations must be positive",
            )
        if checker.get("seed") is not None and not _numeric_scalar(checker["seed"], integer=True):
            return (
                "unsupported-callback",
                f"{checker_path}.seed must be an integer, null, or unresolved interpolation",
            )
        if not _numeric_scalar(checker.get("atol")) or not _numeric_scalar(checker.get("rtol")):
            return (
                "unsupported-callback",
                f"{checker_path} requires audited atol and rtol scalars",
            )
        if any(
            not isinstance(checker[key], str) and checker[key] < 0
            for key in ("atol", "rtol")
        ):
            return (
                "unsupported-callback",
                f"{checker_path}.atol and rtol must be nonnegative",
            )
        if expected_target == _TRACE_EQUIVARIANCE_CHECKER_TARGET and checker.get("compare_output") is not False:
            return (
                "unsupported-callback",
                f"{checker_path}.compare_output must be false in the audited profile",
            )
    return None


def _resource_usage_contract_refusal(
    entry: Mapping[str, Any], path: str
) -> tuple[str, str] | None:
    """Refuse ResourceUsage constructor state outside allocation-local telemetry."""

    unknown = set(entry) - {"_target_", "allocator_probe"}
    if unknown:
        return (
            "unsupported-callback",
            f"{path} {_RESOURCE_USAGE_TARGET!r} has unknown constructor field(s): "
            f"{sorted(unknown)!r}",
        )
    if "allocator_probe" not in entry or entry["allocator_probe"] is None:
        return None
    probe = entry["allocator_probe"]
    if not isinstance(probe, Mapping) or set(probe) != {"_target_", "device"}:
        return (
            "unsupported-callback",
            f"{path}.allocator_probe must be the exact audited TorchAllocatorPeakProbe shape",
        )
    if (
        probe.get("_target_") != _TORCH_ALLOCATOR_PROBE_TARGET
        or not isinstance(probe.get("device"), str)
        or not probe["device"]
    ):
        return (
            "unsupported-callback",
            f"{path}.allocator_probe must target {_TORCH_ALLOCATOR_PROBE_TARGET!r} with a device",
        )
    return None


@dataclass(frozen=True, slots=True)
class _CallbackContract:
    """Sealed syntax-only contract for one admitted callback target."""

    category: str
    validate: Callable[[Mapping[str, Any], str], tuple[str, str] | None]


CALLBACK_CONTRACT_REGISTRY: Mapping[str, _CallbackContract] = MappingProxyType(
    {
        _RUNTIME_EQUIVARIANCE_TARGET: _CallbackContract(
            "RECONSTRUCTED", _runtime_equivariance_contract_refusal
        ),
        _RESOURCE_USAGE_TARGET: _CallbackContract(
            "ALLOCATION_LOCAL", _resource_usage_contract_refusal
        ),
    }
)


def check_resume_sensitive_callback_cadence(
    config: Any,
) -> tuple[str, str] | None:
    """Find cadence options whose callback-local state is not restored.

    Parameters
    ----------
    config : mapping
        Unresolved OmegaConf-compatible configuration. Only callback target
        names and cadence option values are inspected.

    Returns
    -------
    tuple[str, str] or None
        A typed refusal code and reason, or ``None`` when every configured
        resume-sensitive callback uses reconstructible cadence options.

    Notes
    -----
    ``max_calls`` is safe only when omitted or explicitly ``None``. A
    ``probability`` is safe only when omitted or numerically equal to one;
    accepting any lower value would admit a callback-local ``random.Random``
    stream whose position is not part of the checkpoint.
    """

    plain = _plain_config(config)
    if not isinstance(plain, Mapping):
        return None
    callbacks = plain.get("callbacks", ())
    if callbacks is None or isinstance(callbacks, (str, bytes, bytearray)):
        return None
    if not isinstance(callbacks, Sequence):
        return None
    for index, entry in enumerate(callbacks):
        if not isinstance(entry, Mapping):
            continue
        target = entry.get("_target_")
        if target not in CADENCE_SENSITIVE_CALLBACK_TARGETS:
            continue
        if "max_calls" in entry and entry["max_calls"] is not None:
            return (
                CALLBACK_CADENCE_REFUSAL_CODE,
                f"callbacks[{index}] {target!r} sets cadence field 'max_calls'; "
                "the callback-local call count is not restored",
            )
        probability = entry.get("probability", 1)
        if type(probability) not in (int, float) or probability != 1:
            return (
                CALLBACK_CADENCE_REFUSAL_CODE,
                f"callbacks[{index}] {target!r} sets cadence field 'probability' "
                "to a value other than 1.0; the callback-local RNG state is not restored",
            )
        if target == "tpen.callback.Checkpoint" and "schedule" in entry:
            schedule_refusal = _checkpoint_schedule_cadence_refusal(
                entry["schedule"], f"callbacks[{index}].schedule"
            )
            if schedule_refusal is not None:
                return schedule_refusal
    return None


def _checkpoint_schedule_cadence_refusal(
    schedule: Any, path: str
) -> tuple[str, str] | None:
    """Refuse unaudited or locally stateful nested checkpoint schedules.

    Hydra may construct a ``StepCadenceGate`` below ``Checkpoint.schedule``.
    Its call counter and RNG are callback-local state, just like the scalar
    cadence fields handled above, and train_resume does not restore either.
    The in-tree schedule implementations are immutable functions of durable
    ``completed_updates`` and are therefore the only admitted schedule forms.
    """

    if schedule is None:
        return None
    if not isinstance(schedule, Mapping):
        return (
            CALLBACK_CADENCE_REFUSAL_CODE,
            f"{path} is not an audited durable-coordinate CheckpointSchedule; "
            "nested callback-local state cannot be certified",
        )
    target = schedule.get("_target_")
    nested_refusal = _nested_cadence_refusal(schedule, path)
    if nested_refusal is not None:
        return nested_refusal
    if target not in AUDITED_CHECKPOINT_SCHEDULE_TARGETS:
        return (
            CALLBACK_CADENCE_REFUSAL_CODE,
            f"{path} target {target!r} is not an audited durable-coordinate "
            "CheckpointSchedule; nested callback-local state is not restored",
        )
    return None


def _nested_cadence_refusal(value: Any, path: str) -> tuple[str, str] | None:
    """Find unsafe cadence fields in a recursively instantiated mapping."""

    if isinstance(value, Mapping):
        if "max_calls" in value and value["max_calls"] is not None:
            return (
                CALLBACK_CADENCE_REFUSAL_CODE,
                f"{path}.max_calls is callback-local state; its call count is not restored",
            )
        if "probability" in value:
            probability = value["probability"]
            if type(probability) not in (int, float) or probability != 1:
                return (
                    CALLBACK_CADENCE_REFUSAL_CODE,
                    f"{path}.probability is callback-local state; its RNG state is not restored",
                )
        for key, child in value.items():
            if key != "_target_":
                refusal = _nested_cadence_refusal(child, f"{path}.{key}")
                if refusal is not None:
                    return refusal
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            refusal = _nested_cadence_refusal(child, f"{path}[{index}]")
            if refusal is not None:
                return refusal
    return None


def _callback_contract_refusal(config: Any) -> tuple[str, str] | None:
    """Validate the sealed constructor contracts of newly admitted callbacks."""

    if not isinstance(config, Mapping):
        return None
    callbacks = config.get("callbacks", ()) or ()
    if not isinstance(callbacks, Sequence) or isinstance(callbacks, (str, bytes, bytearray)):
        return None
    for index, entry in enumerate(callbacks):
        if not isinstance(entry, Mapping):
            continue
        target = entry.get("_target_")
        contract = CALLBACK_CONTRACT_REGISTRY.get(target)
        if contract is None:
            continue
        refusal = contract.validate(entry, f"callbacks[{index}]")
        if refusal is not None:
            return refusal
    return None


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


def check_profile_refusal(config: Any, profile: Profile) -> tuple[str, str] | None:
    """Return the shared syntax, shape, and cadence refusal for a profile."""

    plain = _plain_config(config)
    syntax_refusal = _syntax_refusal(plain, profile)
    if syntax_refusal is not None:
        return syntax_refusal
    contract_refusal = _callback_contract_refusal(plain)
    if contract_refusal is not None:
        return contract_refusal
    return check_resume_sensitive_callback_cadence(plain)


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
    refusal = check_profile_refusal(config, profile)
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
    "CALLBACK_CADENCE_REFUSAL_CODE",
    "ALLOCATION_LOCAL_CALLBACK_TARGETS",
    "AUDITED_CHECKPOINT_SCHEDULE_TARGETS",
    "CALLBACK_CONTRACT_REGISTRY",
    "CADENCE_SENSITIVE_CALLBACK_TARGETS",
    "RECONSTRUCTED_CALLBACK_TARGETS",
    "check_resume_sensitive_callback_cadence",
    "check_profile_refusal",
    "SUPPORTED_CALLBACK_TARGETS",
    "SUPPORTED_LOGGER_TARGETS",
    "SUPPORTED_STATE_OWNERS",
    "check_profile",
]
