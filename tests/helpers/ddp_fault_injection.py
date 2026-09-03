"""Typed fault injection for the CPU/Gloo subprocess harness (DF1).

Test-only. Immutable ``FaultPlan``, closed ``FaultKind``/``FaultPhase``
enums, explicit target rank, explicit trigger phase, bounded delay. No
string member lookup via ``getattr``, no reflection, no arbitrary dict
dispatch (``ddp-fault-injection-observability-2026-08-31``).

SCOPE NOTE, load-bearing: several member names below (``CRASH_AFTER_PUBLISH``,
``CRASH_DURING_CHECKPOINT``, ``BEFORE_STATE_WRITE``/``AFTER_STATE_WRITE``,
``BEFORE_PUBLICATION``/``AFTER_PUBLICATION``) echo names that also exist for
real in TPEN's checkpoint machinery under ``tpen/checkpoint/`` (owned by a
different, concurrent lane). This module does not touch that code and
injects nothing into it. Every phase and fault here targets ONLY the
synthetic step sequence run by :mod:`tests.helpers.ddp_worker_entrypoint`:
"publication" there is a ``COMPLETE`` marker file written behind a
``dist.barrier()`` in a throwaway worker process, not a TPEN checkpoint
publish. No test in this slice exercises the real checkpoint publication
path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class FaultKind(Enum):
    """Closed set of faults this harness self-test can inject.

    A subset of the design note's full taxonomy: the VMC-semantic faults
    (``NONFINITE_*``, ``DUPLICATE_RNG_STREAM``, ``DIVERGENT_MODEL_PATH``)
    and ``WRONG_DEVICE`` need a real reducer or a real accelerator, neither
    of which exists in this harness-only slice. ``MISMATCH_SHAPE`` is
    included because it is a pure collective-level fault, needs neither.

    ``CRASH_AFTER_PUBLISH`` and ``CRASH_DURING_CHECKPOINT`` name the
    synthetic worker's own fake publication/state-write steps ONLY -- see
    the module-level SCOPE NOTE. Neither reaches, nor is named after
    coverage of, TPEN's real checkpoint machinery under
    ``tpen/checkpoint/``.
    """

    NONE = "none"
    RAISE_BEFORE_BACKWARD = "raise_before_backward"
    SKIP_COLLECTIVE = "skip_collective"
    MISMATCH_COLLECTIVE = "mismatch_collective"
    MISMATCH_SHAPE = "mismatch_shape"
    STALL_BEFORE_COLLECTIVE = "stall_before_collective"
    CRASH_AFTER_PUBLISH = "crash_after_publish"
    CRASH_DURING_CHECKPOINT = "crash_during_checkpoint"


class FaultPhase(Enum):
    """Typed hook points inside one worker's synthetic step sequence.

    ``BEFORE_STATE_WRITE``/``AFTER_STATE_WRITE`` and
    ``BEFORE_PUBLICATION``/``AFTER_PUBLICATION`` name points in THIS
    module's synthetic self-test only -- a throwaway per-rank JSON file and
    a ``COMPLETE`` marker behind a ``dist.barrier()``, both in ``tmp_path``.
    They are not TPEN checkpoint events; see the module-level SCOPE NOTE.
    """

    BEFORE_COLLECTIVE = "before_collective"
    AFTER_COLLECTIVE = "after_collective"
    BEFORE_OPTIMIZER_STEP = "before_optimizer_step"
    AFTER_OPTIMIZER_STEP = "after_optimizer_step"
    BEFORE_STATE_WRITE = "before_state_write"
    AFTER_STATE_WRITE = "after_state_write"
    BEFORE_PUBLICATION = "before_publication"
    AFTER_PUBLICATION = "after_publication"


@dataclass(frozen=True)
class FaultPlan:
    """Immutable fault-injection plan for one subprocess-group invocation.

    Parameters
    ----------
    target_rank : int
        The single rank the fault applies to. Other ranks run unfaulted.
    kind : FaultKind
        Which fault to apply.
    phase : FaultPhase or None
        Where to apply it. ``None`` is a documented sentinel meaning
        "before process-group init, before any in-group phase begins" --
        used only by ``STALL_BEFORE_COLLECTIVE`` to prove the harness
        watchdog acts independently of the (not-yet-existing) process-group
        timeout.
    delay_seconds : float
        Bounded delay used by ``STALL_BEFORE_COLLECTIVE``; ignored by every
        other kind.
    """

    target_rank: int
    kind: FaultKind
    phase: FaultPhase | None
    delay_seconds: float = 0.0


def serialize_fault_plan(plan: FaultPlan) -> str:
    """Serialize ``plan`` to JSON (never pickle)."""

    payload = asdict(plan)
    payload["kind"] = plan.kind.name
    payload["phase"] = plan.phase.name if plan.phase is not None else None
    return json.dumps(payload)


def deserialize_fault_plan(text: str) -> FaultPlan:
    """Deserialize a :func:`serialize_fault_plan` payload.

    Enum members are looked up by name against the fixed enum
    (``FaultKind[name]``/``FaultPhase[name]``), never via ``getattr`` or any
    other reflective/string-dispatch mechanism.
    """

    payload = json.loads(text)
    kind = FaultKind[payload["kind"]]
    phase = FaultPhase[payload["phase"]] if payload["phase"] is not None else None
    return FaultPlan(
        target_rank=int(payload["target_rank"]),
        kind=kind,
        phase=phase,
        delay_seconds=float(payload["delay_seconds"]),
    )


def write_fault_plan(plan: FaultPlan, path: Path) -> None:
    """Write ``plan`` as JSON to ``path``."""

    path.write_text(serialize_fault_plan(plan))


def read_fault_plan(path: Path) -> FaultPlan:
    """Read a :class:`FaultPlan` previously written by :func:`write_fault_plan`."""

    return deserialize_fault_plan(path.read_text())


NO_FAULT = FaultPlan(target_rank=-1, kind=FaultKind.NONE, phase=None)


__all__ = [
    "NO_FAULT",
    "FaultKind",
    "FaultPhase",
    "FaultPlan",
    "deserialize_fault_plan",
    "read_fault_plan",
    "serialize_fault_plan",
    "write_fault_plan",
]
