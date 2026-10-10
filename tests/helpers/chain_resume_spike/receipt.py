"""Durable evidence schema for one chain-resume attempt.

Every field here exists because some prior spike or verifier lost the
corresponding fact:

``process_seed`` and ``pid``
    Recorded so that "the resumed process seeded itself differently from its
    parent" is auditable rather than assumed. Distinct fresh seeds are what
    prevent a reinitialization from accidentally matching.
``executable``
    The absolute interpreter the attempt actually ran under. A receipt that
    does not name its interpreter cannot support any claim about which
    environment produced it.
``limb_application``
    Observed, not requested. Carries the fingerprint comparison that proves a
    disabled limb was genuinely not consumed.
``exit_kind``
    Whether the attempt's exit status was EARNED (the code path really
    succeeded or really failed) or ENGINEERED (the harness chose the status
    because a failure is the datum under test). A receipt that omits this makes
    an arranged exit 0 indistinguishable from a real one.

Writes are flushed and ``fsync``-ed. A ``SIGKILL`` arm otherwise loses the
buffered tail, and a lost tail turns a recorded outcome into silence that looks
exactly like a hang.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .identity import ChainIdentity
from .outcomes import AttemptOutcome

#: Schema token for the attempt receipt. Versioned, so a reader can refuse a
#: shape it does not understand rather than silently misreading fields.
ATTEMPT_RECEIPT_SCHEMA = "tpen.chain-resume-spike.attempt-receipt/v1"


class ExitKind(Enum):
    """Whether an exit status was earned by the code or arranged by the harness.

    ``EARNED``
        The status reflects what the code path actually did.
    ``ENGINEERED``
        The harness chose the status. Used where a test failure is the datum
        and a non-zero exit would be read as harness breakage.
    """

    EARNED = "earned"
    ENGINEERED = "engineered"


@dataclass(frozen=True)
class AttemptReceipt:
    """Durable record of one execution attempt.

    Parameters
    ----------
    identity : ChainIdentity
        Run, attempt and parent-generation identity.
    outcome : AttemptOutcome
        Typed continuation outcome.
    process_seed : int
        Seed this process drew from OS entropy, before any restore.
    pid : int
        Process id, for cross-referencing scheduler and OS records.
    executable : str
        Absolute path of the interpreter that ran this attempt.
    trace : list of dict
        Observable channel trace of the steps this attempt executed.
    credited_steps : tuple of int
        Logical steps this attempt credited.
    replayed_steps : tuple of int
        Logical steps this attempt re-executed but did NOT credit, because a
        previous attempt already had. Recorded separately so the difference
        between replay and double counting is visible in the evidence.
    disabled_limbs : tuple of str
        Limbs whose restore was deliberately skipped.
    limb_application : dict or None
        Observed limb-consumption record. ``None`` on a cold start.
    fault : dict or None
        The injected fault plan, if any.
    exit_kind : ExitKind
        Whether the exit status was earned or engineered.
    notes : dict, optional
        Free-form additional evidence. Never parsed by policy code.
    """

    identity: ChainIdentity
    outcome: AttemptOutcome
    process_seed: int
    pid: int
    executable: str
    trace: list[dict[str, Any]]
    credited_steps: tuple[int, ...]
    replayed_steps: tuple[int, ...]
    disabled_limbs: tuple[str, ...]
    limb_application: dict[str, Any] | None
    fault: dict[str, Any] | None
    exit_kind: ExitKind
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable mapping form."""

        return {
            "schema": ATTEMPT_RECEIPT_SCHEMA,
            "identity": self.identity.to_dict(),
            "outcome": self.outcome.to_dict(),
            "process_seed": int(self.process_seed),
            "pid": int(self.pid),
            "executable": self.executable,
            "trace": self.trace,
            "credited_steps": list(self.credited_steps),
            "replayed_steps": list(self.replayed_steps),
            "disabled_limbs": list(self.disabled_limbs),
            "limb_application": self.limb_application,
            "fault": self.fault,
            "exit_kind": self.exit_kind.value,
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AttemptReceipt":
        """Rebuild an :class:`AttemptReceipt`, refusing an unknown schema."""

        schema = data.get("schema")
        if schema != ATTEMPT_RECEIPT_SCHEMA:
            raise ValueError(
                f"unsupported attempt receipt schema {schema!r}; "
                f"expected {ATTEMPT_RECEIPT_SCHEMA!r}"
            )
        return cls(
            identity=ChainIdentity.from_mapping(data["identity"]),
            outcome=AttemptOutcome.from_mapping(data["outcome"]),
            process_seed=int(data["process_seed"]),
            pid=int(data["pid"]),
            executable=str(data["executable"]),
            trace=list(data["trace"]),
            credited_steps=tuple(int(value) for value in data["credited_steps"]),
            replayed_steps=tuple(int(value) for value in data["replayed_steps"]),
            disabled_limbs=tuple(str(value) for value in data["disabled_limbs"]),
            limb_application=data["limb_application"],
            fault=data["fault"],
            exit_kind=ExitKind(data["exit_kind"]),
            notes=dict(data.get("notes", {})),
        )


def write_receipt(path: Path, receipt: AttemptReceipt) -> None:
    """Write ``receipt`` to ``path`` durably.

    Written to a sibling ``.tmp`` and renamed, then the *directory* is
    ``fsync``-ed, so a reader never observes a half-written receipt and the
    rename itself survives. The same rename-is-the-commit discipline
    ``tpen.checkpoint.save`` uses for a checkpoint directory.
    """

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(receipt.to_dict(), handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_receipt(path: Path) -> AttemptReceipt:
    """Read a :func:`write_receipt` payload back."""

    return AttemptReceipt.from_mapping(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


__all__ = [
    "ATTEMPT_RECEIPT_SCHEMA",
    "AttemptReceipt",
    "ExitKind",
    "read_receipt",
    "write_receipt",
]
