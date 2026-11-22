"""Acceptance checks for the non-interacting two-electron Hooke control."""

from __future__ import annotations

from pathlib import Path

import torch

from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA,
    HOOKE_A37,
    HOOKE_EVAL_SELECTION,
    HOOKE_SCIENCE_PROFILE,
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


def test_qho2e_admits_the_complete_three_system_a37_grid() -> None:
    """The new label participates in the same 444-row Hooke admission."""

    systems = ("hooke1", "hooke2", "qho2e")
    config_sha256 = config_identities(config_root=CONFIG_ROOT, systems=systems)
    physical_ids = {
        system: physical_system_id_from_config(
            CONFIG_ROOT / system / "eval.yaml",
            train_config_path=CONFIG_ROOT / system / "train.yaml",
            system_name=system,
        )
        for system in systems
    }
    phase1_rows = build_manifest(
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
        evaluation_selection=HOOKE_EVAL_SELECTION,
        overlay=str(HOOKE_OVERLAY),
        overlay_sha256=overlay_sha256(HOOKE_OVERLAY),
        expected_physical_system_id=physical_ids,
    )
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
            for checkpoint in HOOKE_A37
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
    assert len(eval_rows) == 3 * 3 * len(HOOKE_A37) * 4
    assert {row["system"] for row in eval_rows} == set(systems)
    assert all(row["science_profile"] == HOOKE_SCIENCE_PROFILE for row in eval_rows)
