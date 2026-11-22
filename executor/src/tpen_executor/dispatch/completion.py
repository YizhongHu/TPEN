"""Controller-side publication of validated leaf transport results."""

from __future__ import annotations

from typing import Any

from ..contracts import ReadyGroup
from ..placement import PlacementRefusal
from ..storage import AttemptFence
from ..tasks import ConfiguredRun
from ..workers import ProcessOutcome
from .contract import DispatcherContract
from .transport import LeafRequest, LeafResult


def _process_outcome_from_leaf_result(result: LeafResult) -> ProcessOutcome | None:
    """Adapt validated leaf science facts for placement receipt generation."""

    science = result.science
    if science is None:
        return None
    return ProcessOutcome(
        tuple(science["argv"]),
        science["returncode"],
        science["stdout_path"],
        science["stderr_path"],
        science["timed_out"],
        science["pid"],
        science["proc_start_ticks"],
    )


def complete_leaf_result(
    lifecycle_or_store: Any,
    fence: AttemptFence,
    request: LeafRequest,
    result: LeafResult,
    *,
    group: ReadyGroup,
    member: Any,
    configured: ConfiguredRun,
    environment: Any | None = None,
    binding: Any = None,
    inspection_error: BaseException | None = None,
    placement_error: PlacementRefusal | None = None,
) -> Exception | None:
    """Publish one validated local or remote leaf result.

    The transport module owns decoding and validation.  This adapter derives
    the same status, digest, evidence, process outcome, and typed child
    refusal facts used by the remote D1/D2 path, then delegates the ordered
    receipt writes to :class:`DispatcherContract`.
    """

    del binding
    if type(request) is not LeafRequest or type(result) is not LeafResult:
        raise TypeError("complete_leaf_result requires a LeafRequest and LeafResult")
    if environment is None:
        environment = request.environment
    status = result.status()
    result_digest = result.result_digest()
    segment_evidence = result.core_evidence()
    outcome = _process_outcome_from_leaf_result(result)
    child_refusal = None
    if result.child_refusal is not None and placement_error is None:
        refusal = result.child_refusal
        child_refusal = PlacementRefusal(refusal["code"], refusal["reason"])
    return DispatcherContract._publish_completion(
        lifecycle_or_store,
        fence,
        group,
        member,
        configured,
        environment,
        outcome,
        status,
        result_digest,
        segment_evidence,
        inspection_error,
        error=placement_error,
        child_refusal=child_refusal,
    )


__all__ = ["complete_leaf_result"]
