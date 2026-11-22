"""T2 proposals: preserve the remote adapter's pre-science refusal facts.

Not run by the reviewer. At PR head 37b98bac the public/known-code cases
are expected to fail; legacy cases and public/unknown-code controls pass.
No leaf process or scheduler is needed by these tests.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tpen_executor.dispatch import contract
from tpen_executor.dispatch.completion import complete_leaf_result
from tpen_executor.dispatch.parsl import _RemoteLeaf, RemoteLeafLaunchError
from tpen_executor.dispatch.transport import (
    LeafTeardownReceipt,
    validate_leaf_result,
)
from tpen_executor.placement import PLACEMENT_OUTPUT_ENV, PlacementRefusal

from test_s3_transport_wire import _request, _synthetic_result


@pytest.mark.parametrize("route", ["legacy", "public"])
@pytest.mark.parametrize(
    ("error_type", "message", "expected_code"),
    [
        (
            "LeafPlacementRefused",
            "placement-cpu-outside-worker-affinity: declared CPUs unavailable",
            "placement-cpu-outside-worker-affinity",
        ),
        (
            "LeafPlacementRefused",
            "placement-worker-affinity-unavailable: affinity query failed",
            "placement-worker-affinity-unavailable",
        ),
        (
            "LeafPlacementRefused",
            "unknown-code: do not promote arbitrary wire text",
            None,
        ),
        (
            "RuntimeError",
            "placement-cpu-outside-worker-affinity: wrong exception type",
            None,
        ),
    ],
)
def test_leaf_launch_refusal_preserves_remote_completion_semantics(
    tmp_path: Path, route: str, error_type: str, message: str,
    expected_code: str | None,
) -> None:
    """Known launch refusals retain their code without caller re-decoding."""

    request, parts = _request(tmp_path)
    group, _allocation, lifecycle, configured, fence = parts
    # No child ran, so there is intentionally no sidecar at this target.
    environment = dataclasses.replace(
        request.environment,
        env={
            **request.environment.env,
            PLACEMENT_OUTPUT_ENV: str(Path(fence.output_dir) / "child-placement.json"),
        },
    )
    request = dataclasses.replace(request, environment=environment)
    result = _synthetic_result(
        request,
        phases=(),
        science=None,
        launch_error={"type": error_type, "message": message},
        inspection="not-run",
        evidence=None,
        teardown=LeafTeardownReceipt.for_request(
            request, inspection="not-run", child_group_reaped=True,
        ).to_dict(),
    )
    result = validate_leaf_result(request, result)

    if route == "legacy":
        # Replay the real resolve -> refusal-overlay -> _complete_member
        # sequence from ParslDispatcher; no dispatcher/worker is launched.
        leaf = _RemoteLeaf(request)
        expected_exception = PlacementRefusal if expected_code else RemoteLeafLaunchError
        with pytest.raises(expected_exception) as caught:
            leaf.resolve(result.encode())
        refusal = caught.value
        if isinstance(refusal, PlacementRefusal):
            environment = contract._with_placement_refusal(environment, refusal)
        error = contract.DispatcherContract._complete_member(
            contract.DispatcherContract.__new__(contract.DispatcherContract),
            group, group.members[0], configured, lifecycle, fence, {},
            environment, None, timeout=None, error=refusal,
        )
    else:
        error = complete_leaf_result(
            lifecycle, fence, request, result,
            group=group, member=group.members[0], configured=configured,
        )

    assert error is None
    manifest = lifecycle.store.manifest(group)
    [attempt] = manifest.attempts
    assert attempt.outcome_receipt.status == "launch_failed"
    assert attempt.outcome_receipt.result_digest is None
    assert manifest.segments == ()
    receipt = attempt.placement_receipt
    assert receipt is not None
    if expected_code is not None:
        assert receipt.refusal["code"] == expected_code
        assert receipt.refusal["reason"] == message.partition(": ")[2]
    else:
        # Match the established missing-sidecar diagnosis for ordinary
        # launch failures; never manufacture a typed affinity refusal.
        assert receipt.refusal["code"] == "placement-missing"
