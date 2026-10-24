"""Formal review round 5 reviewer tests (E4): issue 9 (Balsam record decoders drop unknown keys)."""

import pytest


def _records():
    from tpen.execution.balsam import BalsamRegistrationIntent, BalsamRegistrationReceipt

    d = "a" * 64
    return (
        BalsamRegistrationIntent("l", "b", d, "b" * 64, "c" * 64, "d" * 64),
        BalsamRegistrationReceipt("l", "b", "42", d, "b" * 64, "c" * 64, "d" * 64),
    )


@pytest.mark.parametrize("record", _records(), ids=lambda x: type(x).__name__)
# PREDICTED: RED at E4, E5, CR (issue 9)
def test_r5_balsam_records_refuse_unknown_keys(record):
    data = record.to_dict()
    data["unexpected_field"] = "x"
    with pytest.raises(ValueError):
        type(record).from_mapping(data)


@pytest.mark.parametrize("record", _records(), ids=lambda x: type(x).__name__)
# PREDICTED: GREEN at E4, E5, CR (issue 9 control)
def test_r5_balsam_records_round_trip(record):
    assert type(record).from_mapping(record.to_dict()) == record, f"round trip differed: {record!r}"


@pytest.mark.parametrize("record", _records(), ids=lambda x: type(x).__name__)
@pytest.mark.parametrize("malformed", ["non-mapping", "missing", "wrong-type"])
def test_balsam_record_decoders_own_malformed_wire_refusal(record, malformed):
    data = record.to_dict()
    if malformed == "non-mapping":
        data = None
    elif malformed == "missing":
        del data["lineage_id"]
    else:
        data["lineage_id"] = 42
    with pytest.raises(ValueError):
        type(record).from_mapping(data)
