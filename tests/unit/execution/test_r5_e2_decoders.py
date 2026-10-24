"""R5 E2 decoder regressions; FreshOutputReceipt retired under issue 9."""

import pytest


def test_r5_unused_fresh_output_receipt_is_retired():
    import tpen.execution as execution
    from tpen.execution import receipts

    assert not hasattr(execution, "FreshOutputReceipt")
    assert not hasattr(receipts, "FreshOutputReceipt")


def test_r5_allocation_observation_decoder_is_closed_and_typed():
    from tpen.execution import AllocationObservation

    receipt = AllocationObservation("l", "a", "2026-10-24T00:00:00Z", "host", "test", {})
    assert AllocationObservation.from_mapping(receipt.to_dict()) == receipt
    for data in (None, {**receipt.to_dict(), "unexpected": 1}, {**receipt.to_dict(), "metadata": []}):
        with pytest.raises(ValueError):
            AllocationObservation.from_mapping(data)


@pytest.mark.parametrize("value", [None, 1, "python", {"argv": "python"}])
def test_r5_ready_item_bad_wire_values_raise_value_error(value):
    from tpen.execution import ReadyItem

    data = dict(invocation_id="i", logical_task_id="l", fresh_output_id="o", result_dir="r", argv=value)
    with pytest.raises(ValueError):
        ReadyItem.from_mapping(data)


def _receipts():
    from tpen.execution.receipts import BackendJobReceipt, InvocationReceipt, ResultReceipt

    d = "a" * 64
    common = dict(lineage_id="l", batch_id="b", invocation_id="i")
    return (
        BackendJobReceipt(**common, backend="direct", job_id="42", ready_batch_digest=d, source_config_digest=d, payload_digest=d),
        InvocationReceipt(**common, job_id="42", fresh_output_id="o", ready_batch_digest=d, status="accepted"),
        ResultReceipt(**common, fresh_output_id="o", result_receipt_id="r", payload_digest=d, result_digest=d, status="completed"),
    )



@pytest.mark.parametrize("instance", _receipts(), ids=lambda x: type(x).__name__)
# PREDICTED: RED at E3, E4, E5, CR (issue 9)
def test_r5_receipt_decoders_refuse_unknown_keys(instance):
    data = instance.to_dict()
    data["unexpected_field"] = "x"
    with pytest.raises(ValueError):
        type(instance).from_mapping(data)



@pytest.mark.parametrize("instance", _receipts(), ids=lambda x: type(x).__name__)
# PREDICTED: GREEN at E3, E4, E5, CR (issue 9 control)
def test_r5_receipt_decoders_round_trip_valid_data(instance):
    assert type(instance).from_mapping(instance.to_dict()) == instance, "round trip differed: observed value"



def test_r5_ready_item_constructor_refuses_string_argv():
    from tpen.execution.batch import ReadyItem

    try:
        item = ReadyItem("i", "l", "o", "python run.py", "r")
    except (TypeError, ValueError):
        return
    raise AssertionError(f"string argv was accepted and became {item.argv!r}")



def test_r5_ready_item_from_mapping_refuses_string_argv():
    from tpen.execution.batch import ReadyItem

    try:
        item = ReadyItem.from_mapping({"invocation_id": "i", "logical_task_id": "l", "fresh_output_id": "o", "argv": "python", "result_dir": "r"})
    except (TypeError, ValueError):
        return
    raise AssertionError(f"string argv mapping was accepted and became {item.argv!r}")



def test_r5_ready_item_accepts_sequence_argv():
    from tpen.execution.batch import ReadyItem

    item = ReadyItem("i", "l", "o", ("python", "run.py"), "r")
    mapped = ReadyItem.from_mapping({"invocation_id": "i", "logical_task_id": "l", "fresh_output_id": "o", "argv": ["python", "run.py"], "result_dir": "r"})
    assert item.argv == ("python", "run.py"), f"tuple argv observed as {item.argv!r}"
    assert mapped.argv == ("python", "run.py"), f"mapped argv observed as {mapped.argv!r}"



def _evidence(tmp_path):
    from tpen.checkpoint.reference import CheckpointRef
    from tpen.execution.recovery import CheckpointEvidence

    ref = CheckpointRef(tmp_path / "step_000001", 2, "tpen.checkpoint", 1, 1, "a" * 64, "b" * 64, {})
    return CheckpointEvidence(ref=ref, science_input_digest="c" * 64, target_iterations=2)



def test_r5_checkpoint_evidence_string_false_is_not_committed(tmp_path):
    evidence = _evidence(tmp_path)
    data = evidence.to_dict()
    data["committed"] = "false"
    try:
        restored = type(evidence).from_mapping(data)
    except (TypeError, ValueError):
        return
    assert restored.committed is False, f"string false was observed as committed={restored.committed!r}"



def test_r5_checkpoint_evidence_boolean_false_is_not_committed(tmp_path):
    evidence = _evidence(tmp_path)
    data = evidence.to_dict()
    data["committed"] = False
    restored = type(evidence).from_mapping(data)
    assert restored.committed is False, f"boolean false was observed as committed={restored.committed!r}"



def test_r5_v2_row_without_argv_is_a_value_error():
    from experiments.toolkit.execution_v2 import ready_batch_v2

    kwargs = dict(
        lineage_id="l", batch_id="b", source_config_digest="a" * 64,
        science_input_digest="b" * 64, target_iterations=2, backend="direct",
        recovery="checkpoint", payload={"x": 1},
    )
    try:
        ready_batch_v2(**kwargs, rows=[{"invocation_id": "i", "logical_task_id": "l", "fresh_output_id": "o", "result_dir": "r"}])
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(f"missing argv raised {type(exc).__name__}, not ValueError") from exc
    raise AssertionError("missing argv was accepted")



def test_r5_v2_row_with_empty_argv_is_a_value_error():
    from experiments.toolkit.execution_v2 import ready_batch_v2

    kwargs = dict(
        lineage_id="l", batch_id="b", source_config_digest="a" * 64,
        science_input_digest="b" * 64, target_iterations=2, backend="direct",
        recovery="checkpoint", payload={"x": 1},
    )
    with pytest.raises(ValueError):
        ready_batch_v2(**kwargs, rows=[{"invocation_id": "i", "logical_task_id": "l", "fresh_output_id": "o", "result_dir": "r", "argv": ()}])

