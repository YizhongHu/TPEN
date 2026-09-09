"""Closed fault-point registry, named against TPEN's real commit boundaries.

Every point below is measured against ``tpen/checkpoint/save.py`` as it exists
at this revision, not against a guess. The real post-``mkdir`` sequence is::

    tmp_dir.mkdir(parents=True)                     save.py:157
    resolved_config.yaml                            save.py:158
    model / optimizer / trainer / sampler / rng     save.py:161,167,174,179,186
    component content hashes                        save.py:194
    manifest.write(tmp_dir/"manifest.json")         save.py:209
    (tmp_dir/"COMPLETE").write_text(...)            save.py:210
    tmp_dir.rename(final_dir)                       save.py:211   <-- THE COMMIT
    CheckpointRef.from_directory(final_dir)         save.py:216
    catalog.publish(ref)                            save.py:226
    write_latest(root, final_dir, ...)              save.py:229
    record_publication_receipt(...)                 save.py:237

CORRECTION, recorded deliberately. The lane design note originally named a
point ``after_latest_before_catalog``. Production writes the **catalog before
latest.json**, so that point names a state ``save_checkpoint`` never passes
through and a test against it could not exercise real code. It is replaced here
by the three points that do occur:
:attr:`FaultPoint.AFTER_RENAME_BEFORE_CATALOG`,
:attr:`FaultPoint.AFTER_CATALOG_BEFORE_LATEST` and
:attr:`FaultPoint.AFTER_LATEST_BEFORE_RECEIPT`.

Those three together are the *committed-but-unacknowledged* family: the rename
committed the checkpoint, but one or more of the three separate durable
acknowledgements is missing. ``tpen.checkpoint.catalog.reconcile_publication``
(catalog.py:186) is production's repair primitive for exactly that family, and
its docstring at lines 189-208 states the semantics these tests assert against
rather than guessing them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FaultPoint(Enum):
    """Closed set of named interruption points.

    The first eight are MEASURED by this lane. The last three are INSTRUMENTS
    ONLY: they exist so R1/R2/R3 can drive them against a real backend, and
    this lane deliberately does not measure them. Providing an instrument is
    not covering a gate.
    """

    # --- Pre-commit: the checkpoint is still a ``.tmp`` directory. ---
    DURING_PAYLOAD_WRITE = "during_payload_write"
    AFTER_PAYLOAD_BEFORE_MANIFEST = "after_payload_before_manifest"
    AFTER_MANIFEST_BEFORE_COMPLETE = "after_manifest_before_complete"
    AFTER_COMPLETE_BEFORE_RENAME = "after_complete_before_rename"

    # --- Post-commit: the rename happened; acknowledgements may be missing. ---
    AFTER_RENAME_BEFORE_CATALOG = "after_rename_before_catalog"
    AFTER_CATALOG_BEFORE_LATEST = "after_catalog_before_latest"
    AFTER_LATEST_BEFORE_RECEIPT = "after_latest_before_receipt"

    # --- Catalog durability. ---
    TORN_CATALOG_ROW = "torn_catalog_row"

    # --- Instruments for the candidate lanes. Not measured here. ---
    CANCELLATION_REQUESTED = "cancellation_requested"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ACCEPTED_SUBMIT_LOST_ACK = "accepted_submit_lost_ack"


#: Points this lane exercises with a real test.
MEASURED_POINTS = frozenset(
    {
        FaultPoint.DURING_PAYLOAD_WRITE,
        FaultPoint.AFTER_PAYLOAD_BEFORE_MANIFEST,
        FaultPoint.AFTER_MANIFEST_BEFORE_COMPLETE,
        FaultPoint.AFTER_COMPLETE_BEFORE_RENAME,
        FaultPoint.AFTER_RENAME_BEFORE_CATALOG,
        FaultPoint.AFTER_CATALOG_BEFORE_LATEST,
        FaultPoint.AFTER_LATEST_BEFORE_RECEIPT,
        FaultPoint.TORN_CATALOG_ROW,
    }
)

#: Points provided as instruments for R1/R2/R3 and deliberately UNMEASURED here.
#: Unmeasured is not PASS. The README says so in those words.
INSTRUMENT_ONLY_POINTS = frozenset(
    {
        FaultPoint.CANCELLATION_REQUESTED,
        FaultPoint.BUDGET_EXHAUSTED,
        FaultPoint.ACCEPTED_SUBMIT_LOST_ACK,
    }
)

#: Points that occur after ``tmp_dir.rename`` has committed the checkpoint.
#: A fault at any of these leaves a valid, selectable generation on disk whose
#: durable acknowledgements are incomplete.
COMMITTED_BUT_UNACKNOWLEDGED_POINTS = frozenset(
    {
        FaultPoint.AFTER_RENAME_BEFORE_CATALOG,
        FaultPoint.AFTER_CATALOG_BEFORE_LATEST,
        FaultPoint.AFTER_LATEST_BEFORE_RECEIPT,
    }
)


class FaultAction(Enum):
    """How the fault terminates the attempt.

    ``RAISE``
        A Python exception. The interpreter unwinds, so ``save_checkpoint``'s
        ``finally`` (save.py:245) would run and clear a stale tmp directory.
    ``OS_EXIT``
        ``os._exit``: immediate, no unwinding, no ``finally``, no atexit.
    ``SIGTERM_SELF``
        The process signals itself. Under Python's default handler this
        terminates without unwinding, which is what a scheduler timeout looks
        like -- see the ``WHAT THIS STILL DOES NOT COVER`` note at
        save.py:255-260.
    ``SIGKILL_FROM_PARENT``
        The parent kills the child's process group. No handler can run at all.
        Requested by the child writing a ready marker and then blocking; the
        parent, not the child, delivers the signal.
    """

    RAISE = "raise"
    OS_EXIT = "os_exit"
    SIGTERM_SELF = "sigterm_self"
    SIGKILL_FROM_PARENT = "sigkill_from_parent"


@dataclass(frozen=True)
class FaultPlan:
    """Immutable plan for one injected fault.

    Parameters
    ----------
    point : FaultPoint
        Where to interrupt.
    action : FaultAction
        How to terminate.
    exit_code : int, optional
        Status used by :attr:`FaultAction.OS_EXIT`. Ignored otherwise.
    """

    point: FaultPoint
    action: FaultAction
    exit_code: int = 70

    def __post_init__(self) -> None:
        # Explicit: typeguard is scoped to ``tpen`` and does not instrument
        # this package, so the annotations above are documentation only.
        if not isinstance(self.point, FaultPoint):
            raise TypeError(f"point must be FaultPoint, got {type(self.point).__name__}")
        if not isinstance(self.action, FaultAction):
            raise TypeError(f"action must be FaultAction, got {type(self.action).__name__}")
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise TypeError("exit_code must be an integer")

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serializable mapping form."""

        return {
            "point": self.point.value,
            "action": self.action.value,
            "exit_code": self.exit_code,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "FaultPlan":
        """Rebuild a :class:`FaultPlan`, looking members up by value only.

        No ``getattr``, no reflective dispatch: the same closed-enum discipline
        :mod:`tests.helpers.ddp_fault_injection` applies.
        """

        return cls(
            point=FaultPoint(data["point"]),
            action=FaultAction(data["action"]),
            exit_code=int(data.get("exit_code", 70)),  # type: ignore[arg-type]
        )


def write_fault_plan(plan: FaultPlan, path: Path) -> None:
    """Write ``plan`` to ``path`` as JSON."""

    Path(path).write_text(json.dumps(plan.to_dict(), sort_keys=True), encoding="utf-8")


def read_fault_plan(path: Path) -> FaultPlan:
    """Read a :func:`write_fault_plan` payload back."""

    return FaultPlan.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


class InjectedFault(RuntimeError):
    """Raised by :attr:`FaultAction.RAISE`, so a real bug is never mistaken for one."""


__all__ = [
    "COMMITTED_BUT_UNACKNOWLEDGED_POINTS",
    "INSTRUMENT_ONLY_POINTS",
    "MEASURED_POINTS",
    "FaultAction",
    "FaultPlan",
    "FaultPoint",
    "InjectedFault",
    "read_fault_plan",
    "write_fault_plan",
]
