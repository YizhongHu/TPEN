"""T2 proposals for #807: mapping failures must not authorize fresh launches.

Expected outcomes at 866ca246 are source predictions, not measured results.
Use real controller/reconciliation/storage code and a recording launch port.
No core import, real process launch, or local test execution is needed.
"""

from pathlib import Path

import pytest

from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.lifecycle.controller import GroupControllerFactsV1, GroupControllerOutcome, run_group_controller
from tpen_executor.recovery import MappingCoreAdmissionPort
from tpen_executor.results import ReconciliationPublicationV2

from test_b1a_b_publication import _fixture
from test_b1d_admission_port import _wire
from test_review_b1b2_controller import _real_fresh
from test_review_b1b2_t2_r1 import RecordingExecutor, _binding


@pytest.mark.parametrize(
    "failure", ["evaluator-oserror", "unknown-schema", "bad-profile", "non-mapping"]
)
def test_mapping_failure_cannot_authorize_fresh_launch(tmp_path, failure):
    """Expected FAIL: swallowed failures currently launch a fresh attempt."""
    _, allocation, facts, binding, store = _real_fresh(tmp_path)
    calls = []

    def evaluate(member):
        calls.append(member)
        if failure == "evaluator-oserror":
            raise OSError("core evidence could not be read")
        if failure == "non-mapping":
            return []
        payload = _wire()
        if failure == "unknown-schema":
            payload["schema"] = "unsupported-admission-version"
        else:
            payload["profile"] = {"schema": "tpen.profile-check/v1"}
        return payload

    executor = RecordingExecutor()
    result = run_group_controller(
        binding, allocation.backend_job_id, facts, store,
        MappingCoreAdmissionPort(evaluate), executor,
    )

    assert calls == list(facts.member_inputs)
    assert isinstance(result, GroupControllerOutcome)
    # Unavailable evidence is not the producer's affirmative absent-leaf None.
    assert executor.calls == [], "adapter failure was converted into a fresh launch"
    assert result.reconciliation.members[0].action == "park"
    assert not list(Path(store.root).rglob("launch-intent.json"))


@pytest.mark.parametrize("evidence", ["absent", "unknown", "direct-exception"])
def test_controller_distinguishes_absence_from_uncertainty(tmp_path, evidence):
    """Expected PASS: true absence launches; unknown and raw port errors park."""
    _, allocation, facts, binding, store = _real_fresh(tmp_path)
    calls = []

    def evaluate(member):
        calls.append(member)
        if evidence == "absent":
            return None
        if evidence == "direct-exception":
            raise OSError("core evidence could not be read")
        return _wire(outcome="unknown", admission="unknown", checkpoint="unknown")

    class DirectExceptionPort:
        def admit(self, member):
            return evaluate(member)

    port = DirectExceptionPort() if evidence == "direct-exception" else MappingCoreAdmissionPort(evaluate)
    executor = RecordingExecutor()
    result = run_group_controller(
        binding, allocation.backend_job_id, facts, store, port, executor,
    )

    assert calls[0] == facts.member_inputs[0]
    if evidence == "absent":
        assert len(calls) == 2
        assert calls[1].task_id == facts.member_inputs[0].task_id
        assert calls[1].start_receipt is not None and calls[1].outcome_receipt is not None
    else:
        assert calls == list(facts.member_inputs)
    assert isinstance(result, GroupControllerOutcome)
    publication_path = next(Path(store.root).rglob(".reconciliation-publications/*.json"))
    publication = ReconciliationPublicationV2.decode(publication_path.read_bytes())
    expected_initial = "start_fresh" if evidence == "absent" else "park"
    assert publication.reconciliation.members[0].action == expected_initial
    # Final reconciliation is receipt-driven: a fresh leaf without core
    # admission evidence is parked even though the initial action was fresh.
    expected_action = "park"
    assert result.reconciliation.members[0].action == expected_action
    assert len(executor.calls) == (1 if evidence == "absent" else 0)
    assert len(list(Path(store.root).rglob("launch-intent.json"))) == len(executor.calls)


def test_mapping_successor_reaches_real_controller_once(tmp_path):
    """Expected PASS: decoded success fills quiescence and launches once.

    This is an executor-only composition control, not real-core qualification.
    """
    store, group, _, inputs, _, successors, _ = _fixture(
        tmp_path, ("successor",), remote=False,
    )
    proposed = successors["successor"][2]
    allocation = ConfiguredAllocation(
        proposed.allocation_attempt.allocation_id,
        proposed.allocation_attempt.backend_job_id,
        {},
    )
    facts = GroupControllerFactsV1(
        group, allocation, "a" * 64, "mapping-successor-review", "utc",
        inputs, successors,
    )
    calls = []

    def evaluate(member):
        calls.append(member)
        # Preserve the existing fixture's profile facts; change only the wire
        # envelope and core's deliberately non-authoritative quiescence.
        payload = member.core_verdict.to_dict()
        payload["schema"] = "tpen-core-admission-v1"
        payload["quiescence"] = "unknown"
        return payload

    executor = RecordingExecutor()
    result = run_group_controller(
        _binding(group, allocation.backend_job_id), allocation.backend_job_id,
        facts, store, MappingCoreAdmissionPort(evaluate), executor,
    )

    assert calls[0] == inputs[0]
    assert len(calls) == 2
    assert calls[1].task_id == inputs[0].task_id
    assert calls[1].start_receipt is not None and calls[1].outcome_receipt is not None
    assert isinstance(result, GroupControllerOutcome)
    assert result.reconciliation.members[0].action == "admit_successor"
    assert result.member_outcomes == (("successor", "collected"),)
    assert len(executor.collected) == 1
    assert len(executor.calls) == 1
    assert executor.calls[0][1].attempt_id == proposed.attempt_id
    assert len(list(Path(store.root).rglob("launch-intent.json"))) == 1
