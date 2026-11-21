"""Contract tests for Stage-1 protocol identity and seed injectivity."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import experiments.atomistic.stage1.protocol as protocol_module

from dataclasses import replace

from experiments.atomistic.stage1.protocol import (
    _encode_seed,
    _TORCH_SEED_MODULUS,
    Protocol,
    ProtocolError,
    SYSTEMS,
    STAGE1_PROTOCOL,
    build_manifest,
    bind_evaluation_manifest,
    config_identity,
    evaluation_seed,
    identity_expectations,
    smoke_seed,
    physical_system_id_from_config,
    stable_row_id,
    training_seed,
)

TEST_PROTOCOL = replace(STAGE1_PROTOCOL, lineage_updates=101, retention_cadence=None)


SHA1 = "a" * 40
SHA256 = "b" * 64
REPO_ROOT = Path(__file__).resolve().parents[4]


def _manifest(checkpoints: list[int] = [0, 7, 101]) -> list[dict[str, object]]:
    bindings = {(system, lineage, checkpoint): SHA256
                for system in SYSTEMS
                for lineage in range(3)
                for checkpoint in checkpoints}
    return build_manifest(protocol=TEST_PROTOCOL, source_sha=SHA1,
                          config_sha256=SHA256, checkpoints=checkpoints,
                          lineage_seeds=[11, 12, 13], checkpoint_sha256=bindings,
                          results_root="/external/stage1")


def test_protocol_has_no_implicit_partial_defaults() -> None:
    with pytest.raises(TypeError):
        Protocol()  # type: ignore[call-arg]


def test_stage1_system_roster_includes_the_hooke_pair() -> None:
    assert SYSTEMS == ("hooke1", "hooke2", "he", "h2")


def test_seed_and_row_identity_are_injective_for_non_default_coordinates() -> None:
    coords = [(system, lineage, checkpoint, stream)
              for system in SYSTEMS
              for lineage in (0, 1, 7)
              for checkpoint in (7, 101, 100003)
              for stream in (0, 3)]
    assert len({evaluation_seed(system=s, lineage=l, checkpoint=c, stream=t)
                for s, l, c, t in coords}) == len(coords)
    assert len({stable_row_id(system=s, kind="eval", lineage=l, checkpoint=c, stream=t)
                for s, l, c, t in coords}) == len(coords)
    assert len({training_seed(system=s, lineage=l, lineage_seed=17)
                for s in SYSTEMS for l in (0, 1, 7)}) == len(SYSTEMS) * 3


def test_seed_encoding_covers_field_maxima_and_old_collision_boundaries() -> None:
    rows = [
        (kind, system, 0, 0, 0, 0)
        for kind in ("smoke", "train", "eval")
        for system in SYSTEMS
    ] + [
        (kind, system, 999_999, 999_999, 999_999, 999_999_999)
        for kind in ("train", "eval")
        for system in SYSTEMS
    ]
    seeds = {
        (smoke_seed(system=system) if kind == "smoke" else
         training_seed(system=system, lineage=lineage, lineage_seed=lineage_seed)
         if kind == "train" else
         evaluation_seed(system=system, lineage=lineage, checkpoint=checkpoint, stream=stream))
        for kind, system, lineage, checkpoint, stream, lineage_seed in rows
    }
    assert len(seeds) == len(rows)
    for system in SYSTEMS:
        maxima = {
            _encode_seed(kind=kind, system=system, lineage=999_999,
                         checkpoint=999_999, stream=999_999,
                         lineage_seed=999_999_999)
            for kind in ("smoke", "train", "eval")
        }
        assert len(maxima) == 3
    assert evaluation_seed(system="hooke1", lineage=0, checkpoint=400_000, stream=0) != smoke_seed(system="hooke1")
    assert evaluation_seed(system="hooke1", lineage=0, checkpoint=500_000, stream=0) != smoke_seed(system="he")


def test_seed_known_vectors_are_cross_process_stable_across_hash_seeds() -> None:
    """The protocol vectors do not depend on Python's randomized hash salt."""

    code = (
        "import json; "
        "from experiments.atomistic.stage1.protocol import evaluation_seed, smoke_seed, training_seed; "
        "print(json.dumps([smoke_seed(system='hooke1'), "
        "training_seed(system='he', lineage=7, lineage_seed=17), "
        "evaluation_seed(system='h2', lineage=12, checkpoint=101, stream=3)]))"
    )
    expected = [3275376356878631123, 1864976247629386342, 6989498304342109654]
    outputs = []
    for hash_seed in ("1", "987654321"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        outputs.append(subprocess.check_output(
            [sys.executable, "-c", code], cwd=REPO_ROOT, env=env, text=True,
        ).strip())
    assert [json.loads(output) for output in outputs] == [expected, expected]


def test_manifest_refuses_forced_seed_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    original_training_seed = protocol_module.training_seed
    training_calls = 0

    def collide_across_training_lineages(*, system: str, lineage: int,
                                          lineage_seed: int) -> int:
        nonlocal training_calls
        training_calls += 1
        # The first 12 calls are the complete-plan audit.  Collide only when
        # the row loop reaches hooke1 lineage 1, proving its own guard is live.
        if training_calls == 14:
            return original_training_seed(
                system="hooke1", lineage=0, lineage_seed=11,
            )
        return original_training_seed(
            system=system, lineage=lineage, lineage_seed=lineage_seed,
        )

    monkeypatch.setattr(protocol_module, "training_seed", collide_across_training_lineages)
    with pytest.raises(ProtocolError, match="seed collision"):
        _manifest(checkpoints=[0, 7, 101])
    monkeypatch.setattr(protocol_module, "training_seed", original_training_seed)

    original_evaluation_seed = protocol_module.evaluation_seed
    evaluation_calls = 0

    def collide_across_evaluation_lineages(*, system: str, lineage: int,
                                            checkpoint: int, stream: int) -> int:
        nonlocal evaluation_calls
        evaluation_calls += 1
        # The complete-plan audit emits 144 eval seeds first.  The row loop
        # reaches hooke1 lineage 1 after 12 rows for lineage 0.
        if evaluation_calls == 157:
            return original_evaluation_seed(
                system="hooke1", lineage=0, checkpoint=0, stream=0,
            )
        return original_evaluation_seed(
            system=system, lineage=lineage, checkpoint=checkpoint, stream=stream,
        )

    monkeypatch.setattr(protocol_module, "evaluation_seed", collide_across_evaluation_lineages)
    with pytest.raises(ProtocolError, match="seed collision"):
        _manifest(checkpoints=[0, 7, 101])


def test_manifest_rejects_duplicate_checkpoint() -> None:
    with pytest.raises(ProtocolError, match="duplicates"):
        build_manifest(
            protocol=TEST_PROTOCOL, source_sha=SHA1, config_sha256=SHA256,
            checkpoints=[7, 7], lineage_seeds=[11, 12, 13],
            checkpoint_sha256={(system, lineage, 7): SHA256 for system in SYSTEMS
                               for lineage in range(3)},
            results_root="/external/stage1",
        )


def test_seeds_fit_torch_at_coordinate_maxima_and_reject_overflow() -> None:
    seeds = [
        smoke_seed(system="h2"),
        training_seed(system="h2", lineage=999_999, lineage_seed=999_999_999),
        evaluation_seed(system="h2", lineage=999_999, checkpoint=999_999, stream=999_999),
    ]
    assert all(0 <= seed < _TORCH_SEED_MODULUS for seed in seeds)
    with pytest.raises(ProtocolError, match="invalid training seed coordinates"):
        training_seed(system="h2", lineage=1_000_000, lineage_seed=0)
    with pytest.raises(ProtocolError, match="invalid evaluation seed coordinates"):
        evaluation_seed(system="h2", lineage=0, checkpoint=1_000_000, stream=0)


def test_manifest_binds_real_identity_and_checkpoint_expectations() -> None:
    rows = _manifest()
    expected_rows = len(SYSTEMS) * (1 + 3 + 3 * 3 * 4)
    assert len(rows) == expected_rows
    assert len({row["row_id"] for row in rows}) == expected_rows
    assert all("seed" in row for row in rows)
    assert len({row["seed"] for row in rows}) == expected_rows
    assert all(0 <= int(row["seed"]) < _TORCH_SEED_MODULUS for row in rows)
    eval_row = next(row for row in rows if row["kind"] == "eval")
    assert eval_row["source_sha"] == SHA1
    assert eval_row["config_sha256"] == SHA256
    expected = identity_expectations(eval_row)
    assert set(expected) == {"expected_checkpoint_sha256"}
    assert expected["expected_checkpoint_sha256"] == SHA256
    assert "resolved_config_identity" not in eval_row


def test_smoke_rows_use_real_train_config_and_final_checkpoint_step() -> None:
    rows = _manifest()
    smoke_rows = [row for row in rows if row["kind"] == "smoke"]
    assert {row["config"] for row in smoke_rows} == {
        f"configs/{system}/train.yaml" for system in SYSTEMS
    }
    assert all(row["checkpoint_steps"] == [STAGE1_PROTOCOL.smoke_updates]
               for row in smoke_rows)


def test_manifest_pins_configured_physical_identity_on_eval_rows() -> None:
    rows = _manifest(checkpoints=[0, 7, 101])
    config_root = Path(__file__).parents[1] / "configs"
    for row in rows:
        if row["kind"] == "eval":
            expected = row["expected_physical_system_id"]
            assert isinstance(expected, str) and len(expected) == 64
            assert all(character in "0123456789abcdef" for character in expected)
            assert expected == physical_system_id_from_config(
                config_root / str(row["system"]) / "eval.yaml"
            )
        else:
            assert "expected_physical_system_id" not in row


@pytest.mark.parametrize(
    "ids",
    [
        {"hooke1": "a" * 64},
        {system: "not-a-digest" for system in SYSTEMS},
    ],
)
def test_manifest_rejects_missing_or_invalid_explicit_physical_identity(ids: dict[str, str]) -> None:
    bindings = {(system, lineage, 7): SHA256
                for system in SYSTEMS for lineage in range(3)}
    with pytest.raises(ProtocolError, match="expected_physical_system_id"):
        build_manifest(
            protocol=TEST_PROTOCOL, source_sha=SHA1, config_sha256=SHA256,
            checkpoints=[0, 7, 101], lineage_seeds=[11, 12, 13], checkpoint_sha256=bindings,
            results_root="/external/stage1", expected_physical_system_id=ids,
        )


def _complete_checkpoint_bindings(
    checkpoints: list[int],
) -> dict[tuple[str, int, int], str]:
    """Provide a digest for every requested system/lineage/checkpoint tuple."""

    return {
        (system, lineage, checkpoint): SHA256
        for system in SYSTEMS
        for lineage in range(3)
        for checkpoint in checkpoints
    }


@pytest.mark.parametrize("field,value", [("source_sha", "tiny"), ("config_sha256", "tiny")])
def test_manifest_rejects_placeholder_identity(field: str, value: str) -> None:
    kwargs = {"source_sha": SHA1, "config_sha256": SHA256, field: value}
    with pytest.raises(ProtocolError, match=field):
        build_manifest(
            protocol=TEST_PROTOCOL,
            checkpoints=[0, 7, 101],
            lineage_seeds=[11, 12, 13],
            checkpoint_sha256=_complete_checkpoint_bindings([0, 7, 101]),
            results_root="/external/stage1",
            **kwargs,
        )


def test_manifest_accepts_complete_identity_bindings() -> None:
    """A valid identity and complete checkpoint packet is admitted."""

    rows = build_manifest(
        protocol=TEST_PROTOCOL,
        source_sha=SHA1,
        config_sha256=SHA256,
        checkpoints=[0, 7, 101],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256=_complete_checkpoint_bindings([0, 7, 101]),
        results_root="/external/stage1",
    )
    assert len([row for row in rows if row["kind"] == "eval"]) == len(SYSTEMS) * 3 * 3 * 4


def test_manifest_rejects_placeholder_config_sha256() -> None:
    with pytest.raises(ProtocolError, match="config_sha256"):
        build_manifest(protocol=TEST_PROTOCOL, source_sha=SHA1, config_sha256="tiny",
                       checkpoints=[0, 7, 101], lineage_seeds=[11, 12, 13],
                       checkpoint_sha256={(s, l, 7): SHA256 for s in SYSTEMS
                                          for l in range(3)},
                       results_root="/external/stage1")


def test_manifest_rejects_relative_results_root() -> None:
    with pytest.raises(ProtocolError, match="results_root must be an absolute"):
        build_manifest(
            protocol=TEST_PROTOCOL, source_sha=SHA1, config_sha256=SHA256,
            checkpoints=[0, 7, 101], lineage_seeds=[11, 12, 13],
            checkpoint_sha256={(s, l, 7): SHA256 for s in SYSTEMS
                               for l in range(3)},
            results_root="relative/stage1",
        )


def _phase1_rows() -> list[dict[str, object]]:
    return build_manifest(
        protocol=TEST_PROTOCOL, source_sha=SHA1,
        config_sha256={system: SHA256 for system in SYSTEMS},
        checkpoints=[0, 7, TEST_PROTOCOL.lineage_updates], lineage_seeds=[11, 12, 13], checkpoint_sha256={},
        results_root="/external/stage1", systems=("he",), phase=1,
    )


def _phase2_evidence(phase1: list[dict[str, object]] | None = None) -> dict[str, object]:
    phase1 = _phase1_rows() if phase1 is None else phase1
    return {
        "schema": protocol_module.CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": protocol_module.manifest_hash(phase1),
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": protocol_module.manifest_hash([row])
            for row in phase1 if row["kind"] == "train"
        },
        "checkpoints": {f"he:{lineage}:7": SHA256 for lineage in range(3)},
    }


def test_phase1_selected_system_preserves_full_manifest_identity() -> None:
    selected = _phase1_rows()
    full = build_manifest(
        protocol=TEST_PROTOCOL, source_sha=SHA1,
        config_sha256={system: SHA256 for system in SYSTEMS},
        checkpoints=[0, 7, TEST_PROTOCOL.lineage_updates], lineage_seeds=[11, 12, 13],
            checkpoint_sha256={(system, lineage, checkpoint): SHA256
                               for system in SYSTEMS for lineage in range(3)
                               for checkpoint in (0, 7, TEST_PROTOCOL.lineage_updates)},
        results_root="/external/stage1",
        expected_physical_system_id={system: "c" * 64 for system in SYSTEMS},
    )
    full_selected = [row for row in full if row["system"] == "he" and row["kind"] in {"smoke", "train"}]
    assert selected == full_selected


def test_phase2_rejects_changed_source_sha() -> None:
    with pytest.raises(ProtocolError, match="source_sha"):
        bind_evaluation_manifest(
            phase1_rows=_phase1_rows(), checkpoints=[7], checkpoint_evidence=_phase2_evidence(),
            expected_physical_system_id={"he": "c" * 64}, source_sha="d" * 40,
        )


def test_phase2_requires_checkpoint_binding() -> None:
    with pytest.raises(ProtocolError, match="structured checkpoint evidence"):
        bind_evaluation_manifest(
            phase1_rows=_phase1_rows(), checkpoints=[7], checkpoint_evidence={},
            expected_physical_system_id={"he": "c" * 64},
        )


def test_phase2_rejects_a_duplicate_checkpoint() -> None:
    with pytest.raises(ProtocolError, match="duplicates"):
        bind_evaluation_manifest(
            phase1_rows=_phase1_rows(), checkpoints=[7, 7], checkpoint_evidence=_phase2_evidence(),
            expected_physical_system_id={"he": "c" * 64},
        )


def test_phase2_requires_the_stored_phase1_manifest_hash() -> None:
    rows = _phase1_rows()
    with pytest.raises(ProtocolError, match="manifest_hash"):
        bind_evaluation_manifest(
            phase1_rows=rows, checkpoints=[7], checkpoint_evidence=_phase2_evidence(rows),
            expected_physical_system_id={"he": "c" * 64}, phase1_manifest_hash="a" * 64,
        )


def test_phase2_rows_carry_phase1_manifest_and_train_identity() -> None:
    phase1 = _phase1_rows()
    phase1_hash = protocol_module.manifest_hash(phase1)
    bound = bind_evaluation_manifest(
        phase1_rows=phase1, checkpoints=[7], checkpoint_evidence=_phase2_evidence(phase1),
        expected_physical_system_id={"he": "c" * 64}, phase1_manifest_hash=phase1_hash,
    )
    train_hashes = {
        protocol_module.manifest_hash([row])
        for row in phase1
        if row["kind"] == "train"
    }
    assert {row["phase1_manifest_hash"] for row in bound} == {phase1_hash}
    assert {row["phase1_train_row_hash"] for row in bound} == train_hashes


def test_phase2_rejects_bare_checkpoint_digest_map() -> None:
    with pytest.raises(ProtocolError, match="structured checkpoint evidence"):
        bind_evaluation_manifest(
            phase1_rows=_phase1_rows(), checkpoints=[7],
            checkpoint_evidence={"he:0:7": SHA256},
            expected_physical_system_id={"he": "c" * 64},
        )


def test_phase2_rejects_checkpoint_evidence_from_a_foreign_phase1_manifest() -> None:
    phase1_a = _phase1_rows()
    phase1_b = build_manifest(
        protocol=TEST_PROTOCOL, source_sha=SHA1,
        config_sha256={system: SHA256 for system in SYSTEMS},
        checkpoints=[0, 7, TEST_PROTOCOL.lineage_updates], lineage_seeds=[21, 22, 23], checkpoint_sha256={},
        results_root="/external/stage1", systems=("he",), phase=1,
    )
    with pytest.raises(ProtocolError, match="checkpoint evidence"):
        bind_evaluation_manifest(
            phase1_rows=phase1_b, checkpoints=[7],
            checkpoint_evidence=_phase2_evidence(phase1_a),
            expected_physical_system_id={"he": "c" * 64},
        )


def test_phase2_rejects_evidence_with_a_forged_manifest_hash() -> None:
    phase1 = _phase1_rows()
    evidence = _phase2_evidence(phase1)
    evidence["phase1_manifest_hash"] = "e" * 64
    with pytest.raises(ProtocolError, match="does not match phase-1 manifest"):
        bind_evaluation_manifest(
            phase1_rows=phase1, checkpoints=[7], checkpoint_evidence=evidence,
            expected_physical_system_id={"he": "c" * 64},
        )


def test_phase2_rejects_checkpoint_outside_phase1_steps() -> None:
    evidence = _phase2_evidence()
    evidence["checkpoints"] = {
        **evidence["checkpoints"], "he:0:99": SHA256,
        "he:1:99": SHA256, "he:2:99": SHA256,
    }
    with pytest.raises(ProtocolError, match="checkpoint_steps"):
        bind_evaluation_manifest(
            phase1_rows=_phase1_rows(), checkpoints=[99], checkpoint_evidence=evidence,
            expected_physical_system_id={"he": "c" * 64},
        )


def test_phase2_checks_combined_row_and_seed_uniqueness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase1 = _phase1_rows()
    evidence = _phase2_evidence(phase1)
    monkeypatch.setattr(
        protocol_module, "evaluation_seed",
        lambda **_: int(next(row["seed"] for row in phase1 if row["kind"] == "train")),
    )
    with pytest.raises(ProtocolError, match="jointly unique"):
        bind_evaluation_manifest(
            phase1_rows=phase1, checkpoints=[7], checkpoint_evidence=evidence,
            expected_physical_system_id={"he": "c" * 64},
        )


def test_phase2_rejects_a_conflicting_config_identity() -> None:
    with pytest.raises(ProtocolError, match="config_sha256"):
        bind_evaluation_manifest(
            phase1_rows=_phase1_rows(), checkpoints=[7], checkpoint_evidence=_phase2_evidence(),
            expected_physical_system_id={"he": "c" * 64},
            config_sha256={"he": "d" * 64},
        )


def test_phase2_inherits_phase1_device_when_not_overridden() -> None:
    rows = _phase1_rows()
    for row in rows:
        row["device"] = "cuda"
    bound = bind_evaluation_manifest(
        phase1_rows=rows, checkpoints=[7], checkpoint_evidence=_phase2_evidence(rows),
        expected_physical_system_id={"he": "c" * 64},
    )
    assert all(row["device"] == "cuda" for row in bound)


def test_phase2_rejects_missing_identity_rule() -> None:
    rows = _phase1_rows()
    next(row for row in rows if row["kind"] == "train").pop("config_identity_rule")
    with pytest.raises(ProtocolError, match="identity rule"):
        bind_evaluation_manifest(
            phase1_rows=rows, checkpoints=[7], checkpoint_evidence=_phase2_evidence(rows),
            expected_physical_system_id={"he": "c" * 64},
        )


def test_config_identity_changes_for_eval_only_and_ignores_unrelated_file(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    shutil.copytree(REPO_ROOT / "experiments/atomistic/stage1/configs", config_root)
    before = config_identity(config_root=config_root, systems=("he",))
    (config_root / "unrelated.txt").write_text("unrelated", encoding="utf-8")
    assert config_identity(config_root=config_root, systems=("he",)) == before
    eval_path = config_root / "he" / "eval.yaml"
    eval_path.write_bytes(eval_path.read_bytes() + b"\n")
    assert config_identity(config_root=config_root, systems=("he",))["sha256"] != before["sha256"]
