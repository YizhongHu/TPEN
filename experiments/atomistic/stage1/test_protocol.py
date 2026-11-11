"""Contract tests for the system-neutral Stage-1 manifest."""

from protocol import Protocol, ProtocolError, build_manifest, manifest_hash


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
