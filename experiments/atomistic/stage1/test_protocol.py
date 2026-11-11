"""Contract tests for the system-neutral Stage-1 manifest."""

from experiments.atomistic.stage1.protocol import Protocol, ProtocolError, build_manifest, manifest_hash
from experiments.atomistic.stage1.launch import command_for_row


def test_manifest_has_three_systems_and_four_fresh_streams() -> None:
    rows = build_manifest(Protocol(), checkpoints=[1000], seeds=[1, 2, 3])
    assert len([row for row in rows if row["kind"] == "smoke"]) == 3
    assert len([row for row in rows if row["kind"] == "train"]) == 9
    assert len([row for row in rows if row["kind"] == "eval"]) == 36
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert manifest_hash(rows) == manifest_hash(rows)


def test_protocol_rejects_collapsed_independence() -> None:
    try:
        Protocol(lineages=2).validate()
    except ProtocolError as exc:
        assert "three lineages" in str(exc)
    else:
        raise AssertionError("two lineages must be rejected")


def test_manifest_rows_are_runnable_configs_with_explicit_overrides() -> None:
    rows = build_manifest(Protocol(), checkpoints=[1000], seeds=[1, 2, 3])
    for row in rows:
        command = command_for_row(row, python="/opt/python", driver="train.py")
        assert row["config"].startswith("configs/")
        assert command[:4] == ["/opt/python", "train.py", "--config", row["config"]]
        assert row["overrides"]
