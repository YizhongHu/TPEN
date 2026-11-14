"""Contract tests for Stage-1 protocol identity and seed injectivity."""

from __future__ import annotations

import pytest

from experiments.atomistic.stage1.protocol import (
    Protocol,
    ProtocolError,
    STAGE1_PROTOCOL,
    build_manifest,
    evaluation_seed,
    identity_expectations,
    stable_row_id,
    training_seed,
)


SHA1 = "a" * 40
SHA256 = "b" * 64


def _manifest(checkpoints: list[int] = [7, 101]) -> list[dict[str, object]]:
    bindings = {(system, lineage, checkpoint): SHA256
                for system in ("hooke1", "he", "h2")
                for lineage in range(3)
                for checkpoint in checkpoints}
    return build_manifest(protocol=STAGE1_PROTOCOL, source_sha=SHA1,
                          config_sha256=SHA256, checkpoints=checkpoints,
                          lineage_seeds=[11, 12, 13], checkpoint_sha256=bindings,
                          results_root="/external/stage1")


def test_protocol_has_no_implicit_partial_defaults() -> None:
    with pytest.raises(TypeError):
        Protocol()  # type: ignore[call-arg]


def test_seed_and_row_identity_are_injective_for_non_default_coordinates() -> None:
    coords = [(system, lineage, checkpoint, stream)
              for system in ("hooke1", "he", "h2")
              for lineage in (0, 1, 7)
              for checkpoint in (7, 101, 100003)
              for stream in (0, 3)]
    assert len({evaluation_seed(system=s, lineage=l, checkpoint=c, stream=t)
                for s, l, c, t in coords}) == len(coords)
    assert len({stable_row_id(system=s, kind="eval", lineage=l, checkpoint=c, stream=t)
                for s, l, c, t in coords}) == len(coords)
    assert len({training_seed(system=s, lineage=l, lineage_seed=17)
                for s in ("hooke1", "he", "h2") for l in (0, 1, 7)}) == 9


def test_manifest_binds_real_identity_and_v2_expectations() -> None:
    rows = _manifest()
    eval_row = next(row for row in rows if row["kind"] == "eval")
    assert eval_row["source_sha"] == SHA1
    assert eval_row["config_sha256"] == SHA256
    expected = identity_expectations(eval_row)
    assert expected["expected_checkpoint_sha256"] == SHA256
    assert expected["expected_resolved_config_identity"]["row_id"] == eval_row["row_id"]


@pytest.mark.parametrize("field,value", [("source_sha", "tiny"), ("config_sha256", "tiny")])
def test_manifest_rejects_placeholder_identity(field: str, value: str) -> None:
    kwargs = {"source_sha": SHA1, "config_sha256": SHA256, field: value}
    with pytest.raises(ProtocolError):
        build_manifest(protocol=STAGE1_PROTOCOL, checkpoints=[7], lineage_seeds=[11, 12, 13],
                       checkpoint_sha256={(s, l, 7): SHA256 for s in ("hooke1", "he", "h2")
                                          for l in range(3)},
                       results_root="/external/stage1", **kwargs)
