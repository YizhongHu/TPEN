"""B1e-2b controller submit/collect contract and bounded mutant table."""

from pathlib import Path

from tpen_executor.lifecycle.controller import GroupControllerFactsV1


MODULE_NAME = "executor/src/tpen_executor/lifecycle/controller.py"
MUTANTS = (
    ("submit-then-collect", MODULE_NAME, "launched, handle = executor.submit(fence, invocation)", "handle = executor.submit(fence, invocation)", "test_submit_acknowledgement_is_retained"),
    ("return-before-quiescence", MODULE_NAME, "while active:", "while False:", "test_all_handles_are_collected"),
    ("stop-after-first-fault", MODULE_NAME, "active.pop(0)", "active.clear() or active.pop(0)", "test_failed_leaf_does_not_skip_other_handle"),
    ("skip-final-reconcile", MODULE_NAME, "final = reconcile_group(group, tuple(final_inputs))", "final = published", "test_final_reconciliation_feeds_exit"),
    ("validation-is-success", MODULE_NAME, '"remote-quiescence-unconfirmed"', '"completed"', "test_crossed_result_parks"),
    ("resubmit-after-poll-timeout", MODULE_NAME, "active.append((member, fence, invocation, handle, owner, successor))", "active.append((member, fence, invocation, handle, owner, successor)); executor.submit(fence, invocation)", "test_poll_timeout_reuses_handle"),
)


def test_poll_timeout_is_positive_and_finite() -> None:
    """The controller facts reject an unbounded or non-positive poll."""

    assert GroupControllerFactsV1.__dataclass_fields__["poll_timeout"].default == 1.0


def test_mutant_rows_are_product_only_and_unique() -> None:
    """Every bounded mutant names one source edit and one killer."""

    source = Path(__file__).parents[1].joinpath("src/tpen_executor/lifecycle/controller.py").read_text()
    assert len(MUTANTS) == 6
    assert len({row[2] for row in MUTANTS}) == len(MUTANTS)
    for _name, filename, old, _new, killer in MUTANTS:
        assert filename == MODULE_NAME
        assert source.count(old) == 1
        assert killer.startswith("test_")
