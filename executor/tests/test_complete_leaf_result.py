"""Direct contract tests for controller-side LeafResult publication."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpen_executor.dispatch import contract
from tpen_executor.dispatch.completion import complete_leaf_result
from tpen_executor.dispatch.transport import LeafResult
from tpen_executor.placement import PLACEMENT_OUTPUT_ENV, PlacementRefusal

from test_s3_transport_wire import _begin, _request, _synthetic_result


CONTRACT_SOURCE = Path(__file__).parents[1] / "src" / "tpen_executor" / "dispatch" / "contract.py"


COMPLETE_LEAF_RESULT_MUTANTS = (
    {
        "id": "skip-placement-write",
        "source": CONTRACT_SOURCE,
        "old": "                lifecycle.store.publish_placement(fence, receipt)\n",
        "new": "                pass\n",
        "killer": "test_complete_leaf_result_publishes_placement",
    },
    {
        "id": "accept-noncompleted-outcome",
        "source": CONTRACT_SOURCE,
        "old": '            status == "completed"\n',
        "new": "            True\n",
        "killer": "test_complete_leaf_result_does_not_accept_noneligible_segment",
    },
    {
        "id": "duplicate-outcome-receipt",
        "source": CONTRACT_SOURCE,
        "old": "            lifecycle.finish(fence, status, _utc_now(), result_digest)\n",
        "new": "            lifecycle.finish(fence, status, _utc_now(), result_digest)\n            lifecycle.finish(fence, status, _utc_now(), result_digest)\n",
        "killer": "test_complete_leaf_result_writes_one_outcome_receipt",
    },
    {
        "id": "adapter-stops-delegating",
        "source": CONTRACT_SOURCE,
        "old": "        return self._publish_completion(\n            lifecycle,\n            fence,\n            group,\n            member,\n            configured,\n            environment,\n            outcome,\n            status,\n            result_digest,\n            segment_evidence,\n            inspection_error,\n            error=error,\n            child_refusal=child_refusal,\n        )\n",
        "new": "        return None\n",
        "killer": "test_complete_member_delegates_to_shared_writer",
    },
)


def test_complete_leaf_result_mutants_are_unique_and_compile() -> None:
    for mutant in COMPLETE_LEAF_RESULT_MUTANTS:
        source = mutant["source"].read_text(encoding="utf-8")
        assert source.count(mutant["old"]) == 1, mutant["id"]
        ast.parse(source.replace(mutant["old"], mutant["new"], 1))


def _publish(tmp_path: Path, result: LeafResult, request, parts) -> object:
    group, _allocation_value, lifecycle, configured, fence = parts
    (Path(fence.output_dir) / "core").mkdir()
    error = complete_leaf_result(
        lifecycle,
        fence,
        request,
        result,
        group=group,
        member=group.members[0],
        configured=configured,
    )
    assert error is None
    return lifecycle.store.manifest(group)


@pytest.mark.parametrize(
    ("name", "changes", "expected_status", "accepts"),
    [
        ("completed", {}, "completed", True),
        (
            "failed",
            {"science": {"returncode": 7}},
            "failed",
            False,
        ),
        (
            "timed_out",
            {"science": {"timed_out": True}},
            "timed_out",
            False,
        ),
        (
            "launch_failed",
            {
                "phases": (),
                "science": None,
                "launch_error": {"type": "RuntimeError", "message": "submit failed"},
                "inspection": "not-run",
                "evidence": None,
            },
            "launch_failed",
            False,
        ),
    ],
)
def test_complete_leaf_result_publishes_exact_outcome_and_segment_records(
    tmp_path: Path, name: str, changes: dict, expected_status: str, accepts: bool,
) -> None:
    request, parts = _request(tmp_path)
    if name in {"failed", "timed_out"}:
        science = dict(_synthetic_result(request).science)
        science.update(changes["science"])
        result = dataclasses.replace(_synthetic_result(request), science=science)
    elif name == "launch_failed":
        result = dataclasses.replace(_synthetic_result(request), **changes)
    else:
        result = _synthetic_result(request)
    manifest = _publish(tmp_path, result, request, parts)
    attempt = manifest.attempts[0]
    assert attempt.outcome_receipt is not None
    assert attempt.outcome_receipt.status == expected_status
    assert attempt.outcome_receipt.result_digest == result.result_digest()
    assert bool(manifest.segments) is accepts


def test_complete_leaf_result_does_not_accept_noneligible_segment(tmp_path: Path) -> None:
    request, parts = _request(tmp_path)
    base = _synthetic_result(request)
    science = dict(base.science)
    science["returncode"] = 7
    result = dataclasses.replace(base, science=science)
    manifest = _publish(tmp_path, result, request, parts)
    assert manifest.segments == ()


def test_complete_leaf_result_publishes_placement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, parts = _request(tmp_path)
    group, allocation, lifecycle, configured, fence = parts
    environment = dataclasses.replace(
        request.environment,
        env={**request.environment.env, PLACEMENT_OUTPUT_ENV: str(Path(fence.output_dir) / "child-placement.json")},
    )
    request = dataclasses.replace(request, environment=environment)
    result = _synthetic_result(request)
    (Path(fence.output_dir) / "core").mkdir()

    def placement_builder(*args):
        _observation, allocation_arg, fence_arg, group_arg, member_arg, configured_arg, environment_arg, outcome_arg = args
        return contract._placement_refusal_receipt(
            PlacementRefusal("placement-receipt-invalid", "test placement"),
            allocation_arg,
            fence_arg,
            group_arg,
            member_arg,
            configured_arg,
            environment_arg,
            outcome_arg,
        )

    monkeypatch.setattr(contract, "read_child_placement", lambda _path: object())
    monkeypatch.setattr(contract, "_placement_receipt", placement_builder)
    error = complete_leaf_result(
        lifecycle,
        fence,
        request,
        result,
        group=group,
        member=group.members[0],
        configured=configured,
    )
    assert error is None
    assert (Path(fence.output_dir) / "placement.json").is_file()
    assert allocation.allocation_id.value == result.allocation_id


def test_complete_leaf_result_writes_one_outcome_receipt(tmp_path: Path) -> None:
    request, parts = _request(tmp_path)
    result = _synthetic_result(request)
    manifest = _publish(tmp_path, result, request, parts)
    output = Path(manifest.attempts[0].start_receipt.output_dir)
    assert sorted(path.name for path in output.glob("outcome.json")) == ["outcome.json"]


def test_complete_member_delegates_to_shared_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, parts = _request(tmp_path)
    group, _allocation_value, lifecycle, configured, fence = parts
    called: list[tuple] = []

    def stub(*args, **kwargs):
        called.append((args, kwargs))
        return SimpleNamespace(name="delegated")

    monkeypatch.setattr(contract.DispatcherContract, "_publish_completion", staticmethod(stub))
    result = contract.DispatcherContract._complete_member(
        contract.DispatcherContract.__new__(contract.DispatcherContract),
        group, group.members[0], configured, lifecycle, fence, {}, None, None,
        timeout=None, error=RuntimeError("launch"),
    )
    assert result.name == "delegated"
    assert called and called[0][0][0] is lifecycle and called[0][0][1] is fence
