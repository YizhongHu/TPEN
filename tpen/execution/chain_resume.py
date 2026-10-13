"""Observable chain-resume checks for the bounded E5 native profile.

This module is a test adapter, not a scheduler or a second checkpoint
implementation.  It consumes the state-owner allowlist from
``tpen.execution.native_profile`` and gives each owner one named observable
channel.  The adapter deliberately compares the first event after the resume
cursor; comparing only final tensors would allow two wrong streams to mask one
another.

The adapter does not claim scheduler, allocation-loss, Balsam, IRI, Globus,
or callback-state coverage.  Its process helper is intentionally small and
backend-neutral so the E5 fixture can be driven by a launcher without making
the core package depend on a launcher.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .native_profile import SUPPORTED_STATE_OWNERS


CHANNEL_MAP_VERSION = "e5-native-channel-map-v1"
NATIVE_PROFILE_VERSION = "tpen.execution.native_profile:SUPPORTED_STATE_OWNERS@e5"


class ChannelMapError(ValueError):
    """Raised when a supported mutable owner has no observable channel."""


class ChannelParityError(AssertionError):
    """Raised when a channel's first post-resume event violates its contract."""


class UnmeasuredArmError(AssertionError):
    """Raised instead of treating a missing or uncollected arm as a pass."""


class CheckpointFaultKind(str, Enum):
    """Typed checkpoint-boundary faults exposed by the adapter."""

    NO_CHECKPOINT = "no_checkpoint"
    BEFORE_COMMIT = "before_commit"
    AFTER_COMMIT = "after_commit"
    CORRUPT_COMMITTED = "corrupt_committed"


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """One supported mutable owner and the observable it feeds.

    Parameters
    ----------
    limb : str
        E5 profile state owner.
    channel : str
        Stable name emitted by the fixture trace.
    event_kind : str
        First-event kind used to reject an end-state-only observation.
    """

    limb: str
    channel: str
    event_kind: str


# These channels are fixture-level observables, not new checkpoint owners.
# ``trainer_cursor`` is observed at the first resumed step; all other channels
# observe the random-dependent work performed at that step.
_E5_CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("model", "model_logabs", "first_model_evaluation"),
    ChannelSpec("optimizer", "optimizer_update", "first_optimizer_update"),
    ChannelSpec("trainer_cursor", "trainer_cursor", "first_resumed_step"),
    ChannelSpec("sampler_mcmc", "sampler_proposal", "first_sampler_proposal"),
    ChannelSpec("process_rng", "process_random", "first_process_random"),
)


def e5_channel_map(
    supported_owners: Iterable[str] = SUPPORTED_STATE_OWNERS,
) -> tuple[ChannelSpec, ...]:
    """Return the complete channel map for the consumed E5 profile.

    Parameters
    ----------
    supported_owners : iterable of str, optional
        The allowlist exported by E5's native profile.  The default reads that
        allowlist directly; no callback or state owner is inferred here.

    Returns
    -------
    tuple of ChannelSpec
        One observable channel for every supported owner.

    Raises
    ------
    ChannelMapError
        If the profile contains an owner for which this adapter has no channel.
    """

    owners = frozenset(supported_owners)
    by_limb = {spec.limb: spec for spec in _E5_CHANNELS}
    missing = owners - by_limb.keys()
    if missing:
        raise ChannelMapError(
            "supported mutable limb has no observable channel: "
            + ", ".join(sorted(missing))
        )
    unknown = by_limb.keys() - owners
    if unknown:
        # A channel for an owner outside the declared E5 profile would silently
        # expand the qualification boundary, so reject it too.
        raise ChannelMapError(
            "channel map expands the E5 profile: " + ", ".join(sorted(unknown))
        )
    return tuple(by_limb[owner] for owner in sorted(owners))


def validate_channel_map(
    supported_owners: Iterable[str], channel_map: Iterable[ChannelSpec]
) -> tuple[ChannelSpec, ...]:
    """Validate totality, uniqueness, and direction of an explicit map.

    Parameters
    ----------
    supported_owners : iterable of str
        The exact E5 allowlist being consumed.
    channel_map : iterable of ChannelSpec
        Candidate owner-to-observable declarations.

    Returns
    -------
    tuple of ChannelSpec
        The normalized map.

    Raises
    ------
    ChannelMapError
        If a limb is absent, duplicated, or points to an empty/duplicate
        channel.  This is deliberately a measured totality guard rather than
        a declaration-only check.
    """

    owners = frozenset(supported_owners)
    specs = tuple(channel_map)
    if any(not isinstance(spec, ChannelSpec) for spec in specs):
        raise ChannelMapError("channel map entries must be ChannelSpec values")
    limbs = [spec.limb for spec in specs]
    channels = [spec.channel for spec in specs]
    if len(limbs) != len(set(limbs)):
        raise ChannelMapError("channel map contains a duplicate mutable limb")
    if len(channels) != len(set(channels)):
        raise ChannelMapError("channel map contains a duplicate observable channel")
    if any(not spec.channel or not spec.event_kind for spec in specs):
        raise ChannelMapError("every mutable limb must name a channel and first event")
    missing = owners - set(limbs)
    if missing:
        raise ChannelMapError(
            "supported mutable limb has no observable channel: "
            + ", ".join(sorted(missing))
        )
    extra = set(limbs) - owners
    if extra:
        raise ChannelMapError(
            "channel map names owners outside the E5 profile: "
            + ", ".join(sorted(extra))
        )
    return tuple(sorted(specs, key=lambda spec: spec.limb))


@dataclass(frozen=True, slots=True)
class ChannelEvent:
    """One observable emitted by a fixture process."""

    channel: str
    step: int
    event_kind: str
    value: Any
    random_dependent: bool = True


@dataclass(frozen=True, slots=True)
class ProcessEvidence:
    """Actual process and seed provenance for one fixture execution."""

    phase: str
    pid: int
    seed: int
    seed_source: str
    command: tuple[str, ...]
    returncode: int
    events: tuple[ChannelEvent, ...]

    @classmethod
    def from_json_output(
        cls,
        *,
        phase: str,
        pid: int,
        seed: int,
        seed_source: str,
        command: Sequence[str],
        returncode: int,
        stdout: str,
    ) -> "ProcessEvidence":
        """Decode a fixture's JSON event stream without importing its code."""

        records = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        events: list[ChannelEvent] = []
        for record in records:
            if not isinstance(record, Mapping) or record.get("record") != "channel_event":
                continue
            events.append(
                ChannelEvent(
                    channel=str(record["channel"]),
                    step=int(record["step"]),
                    event_kind=str(record["event_kind"]),
                    value=record.get("value"),
                    random_dependent=bool(record.get("random_dependent", True)),
                )
            )
        return cls(
            phase=phase,
            pid=pid,
            seed=seed,
            seed_source=seed_source,
            command=tuple(command),
            returncode=returncode,
            events=tuple(events),
        )


def run_fresh_process(
    command: Sequence[str],
    *,
    phase: str,
    seed: int,
    seed_source: str,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ProcessEvidence:
    """Run one fresh fixture process and retain its real PID and trace.

    The adapter does not reuse a Python callable in the parent process.  The
    launcher receives the seed through an environment variable so the receipt
    records provenance without importing or constructing science in the
    adapter.
    """

    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env["TPEN_CHAIN_RESUME_SEED"] = str(seed)
    child_env["TPEN_CHAIN_RESUME_PHASE"] = phase
    process = subprocess.Popen(
        tuple(command),
        cwd=None if cwd is None else str(cwd),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"chain-resume fixture process failed ({phase}, rc={process.returncode}): {stderr}"
        )
    return ProcessEvidence.from_json_output(
        phase=phase,
        pid=process.pid,
        seed=seed,
        seed_source=seed_source,
        command=tuple(command),
        returncode=process.returncode,
        stdout=stdout,
    )


def assert_fresh_processes(reference: ProcessEvidence, resumed: ProcessEvidence) -> None:
    """Require distinct actual processes and explicit distinct seed provenance."""

    if reference.pid == resumed.pid:
        raise ChannelParityError("reference and resumed execution reused the same PID")
    if reference.phase == resumed.phase:
        raise ChannelParityError("reference and resumed evidence have the same phase")
    if not reference.seed_source or not resumed.seed_source:
        raise ChannelParityError("seed provenance is missing")


def first_subsequent_event(
    evidence: ProcessEvidence,
    spec: ChannelSpec,
    *,
    resume_step: int,
) -> ChannelEvent:
    """Return the first random-dependent event after a resume cursor."""

    for event in evidence.events:
        if (
            event.channel == spec.channel
            and event.step > resume_step
            and event.random_dependent
        ):
            if event.event_kind != spec.event_kind:
                raise ChannelParityError(
                    f"{spec.limb} emitted {event.event_kind!r}, expected {spec.event_kind!r}"
                )
            return event
    raise UnmeasuredArmError(
        f"no first subsequent random-dependent event for channel {spec.channel!r}"
    )


def compare_channel_parity(
    reference: ProcessEvidence,
    resumed: ProcessEvidence,
    *,
    resume_step: int,
    channel_map: Iterable[ChannelSpec] | None = None,
    disabled_limbs: Iterable[str] = (),
) -> dict[str, bool]:
    """Compare first post-cursor events, naming expected disabled mismatches.

    Returns
    -------
    dict of str to bool
        Per-channel parity booleans.  Disabled limbs must be false; enabled
        limbs must be true.  A missing event raises ``UnmeasuredArmError``.
    """

    specs = validate_channel_map(SUPPORTED_STATE_OWNERS, channel_map or e5_channel_map())
    disabled = frozenset(disabled_limbs)
    unknown_disabled = disabled - {spec.limb for spec in specs}
    if unknown_disabled:
        raise ChannelMapError("disabled limb is outside the E5 profile")
    result: dict[str, bool] = {}
    for spec in specs:
        left = first_subsequent_event(reference, spec, resume_step=resume_step)
        right = first_subsequent_event(resumed, spec, resume_step=resume_step)
        equal = left.value == right.value
        result[spec.channel] = equal
        expected = spec.limb not in disabled
        if equal != expected:
            expected_text = "parity" if expected else "a mismatch"
            raise ChannelParityError(
                f"channel {spec.channel!r} expected {expected_text}, observed {equal}"
            )
    return result


def assert_positive_control(
    baseline: ProcessEvidence,
    perturbed: ProcessEvidence,
    spec: ChannelSpec,
    *,
    resume_step: int,
) -> None:
    """Prove that a channel's own positive control can move its observable."""

    left = first_subsequent_event(baseline, spec, resume_step=resume_step)
    right = first_subsequent_event(perturbed, spec, resume_step=resume_step)
    if left.value == right.value:
        raise ChannelParityError(f"positive control did not move {spec.channel}")


def classify_checkpoint_fault(
    kind: CheckpointFaultKind | str,
    *,
    detail: str,
) -> dict[str, str]:
    """Return a typed fixture/application outcome for a checkpoint fault.

    Scheduler and wrapper outcomes are intentionally not accepted as inputs;
    they belong to E3/E4 receipts.  This function only records what the
    adapter's fixture/application interface observed at a checkpoint boundary.
    """

    fault = CheckpointFaultKind(kind)
    if not isinstance(detail, str) or not detail:
        raise ValueError("checkpoint fault detail must be non-empty")
    layer = "fixture" if fault in {
        CheckpointFaultKind.NO_CHECKPOINT,
        CheckpointFaultKind.BEFORE_COMMIT,
    } else "application"
    return {"layer": layer, "kind": fault.value, "detail": detail}


@dataclass(frozen=True, slots=True)
class ChainResumeReceipt:
    """Machine-readable evidence retained by the adapter qualification."""

    source_sha: str
    adapter_sha: str
    profile_version: str
    channel_map_version: str
    commands: tuple[tuple[str, ...], ...]
    process_evidence: tuple[ProcessEvidence, ...]
    channel_results: Mapping[str, bool]
    checkpoint_faults: tuple[Mapping[str, str], ...]
    collection_output: tuple[str, ...]
    testcase_ids: tuple[str, ...]
    junit_identities: tuple[str, ...]
    raw_fault_evidence: tuple[Mapping[str, Any], ...]
    restore_disabled_control: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize all provenance fields without dropping unmeasured arms."""

        return {
            "schema": "tpen.execution/chain-resume-adapter-receipt-v1",
            "source_sha": self.source_sha,
            "adapter_sha": self.adapter_sha,
            "profile_version": self.profile_version,
            "channel_map_version": self.channel_map_version,
            "commands": [list(command) for command in self.commands],
            "process_evidence": [
                {
                    "phase": evidence.phase,
                    "pid": evidence.pid,
                    "seed": evidence.seed,
                    "seed_source": evidence.seed_source,
                    "command": list(evidence.command),
                    "returncode": evidence.returncode,
                    "events": [
                        {
                            "channel": event.channel,
                            "step": event.step,
                            "event_kind": event.event_kind,
                            "value": event.value,
                            "random_dependent": event.random_dependent,
                        }
                        for event in evidence.events
                    ],
                }
                for evidence in self.process_evidence
            ],
            "channel_results": dict(self.channel_results),
            "checkpoint_faults": [dict(fault) for fault in self.checkpoint_faults],
            "collection_output": list(self.collection_output),
            "testcase_ids": list(self.testcase_ids),
            "junit_identities": list(self.junit_identities),
            "raw_fault_evidence": [dict(fault) for fault in self.raw_fault_evidence],
            "restore_disabled_control": dict(self.restore_disabled_control),
        }


__all__ = [
    "CHANNEL_MAP_VERSION",
    "NATIVE_PROFILE_VERSION",
    "ChannelEvent",
    "ChannelMapError",
    "ChannelParityError",
    "ChannelSpec",
    "CheckpointFaultKind",
    "ChainResumeReceipt",
    "ProcessEvidence",
    "UnmeasuredArmError",
    "assert_fresh_processes",
    "assert_positive_control",
    "classify_checkpoint_fault",
    "compare_channel_parity",
    "e5_channel_map",
    "first_subsequent_event",
    "run_fresh_process",
    "validate_channel_map",
]
