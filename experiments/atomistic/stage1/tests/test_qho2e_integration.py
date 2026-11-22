"""Acceptance checks for the QHO-2e control."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA,
    HOOKE_A37,
    HOOKE_EVAL_SELECTION,
    HOOKE_SCIENCE_PROFILE,
    ProtocolError,
    QHO_EVAL_SELECTION,
    QHO_Q21,
    STAGE1_PROTOCOL,
    bind_evaluation_manifest,
    build_manifest,
    config_identities,
    manifest_hash,
    overlay_sha256,
    physical_system_id_from_config,
)
from experiments.baselines.systems import load_registry
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.physics.hamiltonian import local_energy
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import HarmonicTrap


ROOT = Path(__file__).parents[1]
CONFIG_ROOT = ROOT / "configs"
QHO2E = CONFIG_ROOT / "qho2e"
HOOKE_OVERLAY = CONFIG_ROOT / "overlays" / "hooke-diagnostics-v1.yaml"


class _ExactNoninteractingGaussian(torch.nn.Module):
    """Exact product-state envelope for the omega=0.5 control Hamiltonian."""

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = -0.25 * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def test_qho2e_has_the_analytic_constant_local_energy() -> None:
    """The product Gaussian is exact away from and near electron coalescence."""

    positions = torch.tensor(
        [
            [[0.75, -0.25, 0.5], [-1.0, 0.25, -0.75]],
            [[0.0, 0.0, 0.0], [1.0e-8, -1.0e-8, 2.0e-8]],
        ],
        dtype=torch.float64,
    )
    energy = local_energy(
        {"kinetic": KineticEnergy(), "harmonic_trap": HarmonicTrap(omega=0.5)},
        _ExactNoninteractingGaussian(),
        ElectronBatch(positions=positions),
    )
    assert torch.all(torch.isfinite(energy))
    assert torch.allclose(energy, torch.full_like(energy, 1.5), atol=1.0e-10, rtol=0.0)


def test_qho2e_identity_and_registry_reference_are_distinct() -> None:
    """The new physics hash and 1.5 Ha registry reference are not inherited."""

    qho2e_id = physical_system_id_from_config(
        QHO2E / "eval.yaml", train_config_path=QHO2E / "train.yaml", system_name="qho2e"
    )
    hooke1_id = physical_system_id_from_config(CONFIG_ROOT / "hooke1" / "eval.yaml")
    hooke2_id = physical_system_id_from_config(CONFIG_ROOT / "hooke2" / "eval.yaml")
    assert qho2e_id not in {hooke1_id, hooke2_id}

    entry = next(item for item in load_registry()["systems"] if item["id"] == "qho_pair_noninteracting_singlet_omega0.5")
    assert entry["reference_energy_hartree"] == 1.5
    assert "3*omega" in entry["reference_source"]
    assert {term["term"] for term in entry["hamiltonian"]["terms"]} == {
        "kinetic", "harmonic_trap",
    }


Q21_LITERAL = (0, 1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000,
               9_000, 10_000, 15_000, 20_000, 30_000, 40_000, 50_000,
               60_000, 70_000, 80_000, 90_000, 100_000)


def _phase1_manifest(systems: tuple[str, ...]) -> tuple[list[dict], dict[str, str], str]:
    config_sha256 = config_identities(config_root=CONFIG_ROOT, systems=systems)
    physical_ids = {
        system: physical_system_id_from_config(
            CONFIG_ROOT / system / "eval.yaml",
            train_config_path=CONFIG_ROOT / system / "train.yaml",
            system_name=system,
        )
        for system in systems
    }
    rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=config_sha256,
        checkpoints=list(range(0, 100001, 1000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/external/stage1",
        systems=systems,
        phase=1,
        science_profile=HOOKE_SCIENCE_PROFILE,
        evaluation_selection=(
            QHO_EVAL_SELECTION if systems == ("qho2e",) else HOOKE_EVAL_SELECTION
        ),
        overlay=str(HOOKE_OVERLAY),
        overlay_sha256=overlay_sha256(HOOKE_OVERLAY),
        expected_physical_system_id=physical_ids,
    )
    return rows, physical_ids, config_sha256


def test_qho2e_q21_literal_selection_is_exact() -> None:
    """Q21 is a frozen explicit tuple, not an accidental range derivation."""

    assert QHO_Q21 == Q21_LITERAL
    assert len(Q21_LITERAL) == 21


def test_qho2e_admits_252_eval_rows() -> None:
    """QHO-2e admits exactly 21 checkpoints x 4 streams x 3 lineages."""

    phase1_rows, physical_ids, config_sha256 = _phase1_manifest(("qho2e",))
    phase1_hash = manifest_hash(phase1_rows)
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"qho2e:{lineage}:{checkpoint}": "b" * 64
            for lineage in range(STAGE1_PROTOCOL.lineages)
            for checkpoint in Q21_LITERAL
        },
    }
    eval_rows = bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=list(Q21_LITERAL),
        checkpoint_evidence=evidence,
        expected_physical_system_id=physical_ids,
        source_sha="a" * 40,
        config_sha256=config_sha256,
        protocol=STAGE1_PROTOCOL,
        phase1_manifest_hash=phase1_hash,
    )
    assert len(eval_rows) == 252
    assert {row["checkpoint"] for row in eval_rows} == set(Q21_LITERAL)
    assert {row["evaluation_selection"] for row in eval_rows} == {QHO_EVAL_SELECTION}


@pytest.mark.parametrize("bad_checkpoints", [
    Q21_LITERAL[:-1],
    Q21_LITERAL + (11_000,),
    tuple(12_000 if checkpoint == 15_000 else checkpoint for checkpoint in Q21_LITERAL),
])
def test_qho2e_refuses_missing_extra_or_non_q21_checkpoints(
    bad_checkpoints: tuple[int, ...],
) -> None:
    """QHO-2e rejects any roster other than the literal Q21 tuple."""

    phase1_rows, physical_ids, config_sha256 = _phase1_manifest(("qho2e",))
    phase1_hash = manifest_hash(phase1_rows)
    with pytest.raises(ProtocolError, match="Q21"):
        bind_evaluation_manifest(
            phase1_rows=phase1_rows,
            checkpoints=list(bad_checkpoints),
            checkpoint_evidence={},
            expected_physical_system_id=physical_ids,
            source_sha="a" * 40,
            config_sha256=config_sha256,
            protocol=STAGE1_PROTOCOL,
            phase1_manifest_hash=phase1_hash,
        )


def test_qho2e_and_interacting_hooke_keep_their_distinct_rosters() -> None:
    """A mixed Hooke phase retains A37 for old systems and Q21 for QHO-2e."""

    systems = ("hooke1", "hooke2", "qho2e")
    phase1_rows, physical_ids, config_sha256 = _phase1_manifest(systems)
    phase1_hash = manifest_hash(phase1_rows)
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"{system}:{lineage}:{checkpoint}": "b" * 64
            for system in systems
            for lineage in range(STAGE1_PROTOCOL.lineages)
            for checkpoint in (Q21_LITERAL if system == "qho2e" else HOOKE_A37)
        },
    }
    eval_rows = bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=list(HOOKE_A37),
        checkpoint_evidence=evidence,
        expected_physical_system_id=physical_ids,
        source_sha="a" * 40,
        config_sha256=config_sha256,
        protocol=STAGE1_PROTOCOL,
        phase1_manifest_hash=phase1_hash,
    )
    assert len(eval_rows) == (2 * 3 * len(HOOKE_A37) * 4) + (3 * len(Q21_LITERAL) * 4)
    assert {row["system"] for row in eval_rows} == set(systems)
    assert all(row["science_profile"] == HOOKE_SCIENCE_PROFILE for row in eval_rows)
