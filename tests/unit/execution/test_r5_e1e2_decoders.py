"""Formal review round 5 reviewer tests (E1/E2): issue 9 (decoders accept data they should refuse) and issue 13 (unreachable E1 branches)."""

from pathlib import Path

import pytest


def _receipts():
    from tpen.execution.receipts import BackendJobReceipt, FreshOutputReceipt, InvocationReceipt, ResultReceipt

    d = "a" * 64
    common = dict(lineage_id="l", batch_id="b", invocation_id="i")
    return (
        BackendJobReceipt(**common, backend="direct", job_id="42", ready_batch_digest=d, source_config_digest=d, payload_digest=d),
        InvocationReceipt(**common, job_id="42", fresh_output_id="o", ready_batch_digest=d, status="accepted"),
        FreshOutputReceipt(**common, fresh_output_id="o", output_path="out", payload_digest=d),
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


# PREDICTED: RED at E3, E4, E5, CR (issue 9)
def test_r5_ready_item_constructor_refuses_string_argv():
    from tpen.execution.batch import ReadyItem

    try:
        item = ReadyItem("i", "l", "o", "python run.py", "r")
    except (TypeError, ValueError):
        return
    raise AssertionError(f"string argv was accepted and became {item.argv!r}")


# PREDICTED: RED at E3, E4, E5, CR (issue 9)
def test_r5_ready_item_from_mapping_refuses_string_argv():
    from tpen.execution.batch import ReadyItem

    try:
        item = ReadyItem.from_mapping({"invocation_id": "i", "logical_task_id": "l", "fresh_output_id": "o", "argv": "python", "result_dir": "r"})
    except (TypeError, ValueError):
        return
    raise AssertionError(f"string argv mapping was accepted and became {item.argv!r}")


# PREDICTED: GREEN at E3, E4, E5, CR (issue 9 control)
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


# PREDICTED: RED at E3, E4, E5, CR (issue 9)
def test_r5_checkpoint_evidence_string_false_is_not_committed(tmp_path):
    evidence = _evidence(tmp_path)
    data = evidence.to_dict()
    data["committed"] = "false"
    try:
        restored = type(evidence).from_mapping(data)
    except (TypeError, ValueError):
        return
    assert restored.committed is False, f"string false was observed as committed={restored.committed!r}"


# PREDICTED: GREEN at E3, E4, E5, CR (issue 9 control)
def test_r5_checkpoint_evidence_boolean_false_is_not_committed(tmp_path):
    evidence = _evidence(tmp_path)
    data = evidence.to_dict()
    data["committed"] = False
    restored = type(evidence).from_mapping(data)
    assert restored.committed is False, f"boolean false was observed as committed={restored.committed!r}"


# PREDICTED: RED at E3, E4, E5, CR (issue 9)
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


# PREDICTED: GREEN at E3, E4, E5, CR (issue 9 control)
def test_r5_v2_row_with_empty_argv_is_a_value_error():
    from experiments.toolkit.execution_v2 import ready_batch_v2

    kwargs = dict(
        lineage_id="l", batch_id="b", source_config_digest="a" * 64,
        science_input_digest="b" * 64, target_iterations=2, backend="direct",
        recovery="checkpoint", payload={"x": 1},
    )
    with pytest.raises(ValueError):
        ready_batch_v2(**kwargs, rows=[{"invocation_id": "i", "logical_task_id": "l", "fresh_output_id": "o", "result_dir": "r", "argv": ()}])


# PREDICTED: GREEN at E3, E4, E5, CR (issue 13 unreachable-code control)
def test_r5_unbound_raw_evidence_forms_never_reach_coerce(monkeypatch):
    import tpen.execution.recovery as recovery
    from tpen.execution import RecoveryPhase, RefusalCode, capture_science_intent, select_recovery

    original = recovery._coerce_evidence
    seen = []

    def spy(evidence, *args, **kwargs):
        seen.append(type(evidence))
        return original(evidence, *args, **kwargs)

    monkeypatch.setattr(recovery, "_coerce_evidence", spy)
    intent = capture_science_intent({"x": 1}, target_iterations=2)
    from tpen.checkpoint.reference import CheckpointRef

    ref = CheckpointRef(Path("some/step_000001"), 2, "tpen.checkpoint", 1, 1, "a" * 64, "b" * 64, {})
    for phase in (RecoveryPhase.INITIAL, RecoveryPhase.RECOVERY):
        for value in ("some/step_000001", Path("some/step_000001"), ref):
            decision = select_recovery(intent, phase=phase, evidence=value)
            assert decision.refusal.code is RefusalCode.INCOMPATIBLE, f"observed refusal={decision.refusal!r}"
    assert seen == [], f"_coerce_evidence observed calls: {seen!r}"
