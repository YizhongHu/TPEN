"""Delta probes for Stage-1 review findings F1 and F3."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import protocol as stage1_protocol
from experiments.atomistic.stage1.launch import _binding_token, row_overrides
from experiments.atomistic.stage1.protocol import (
    Protocol,
    ProtocolError,
    SYSTEMS,
    build_manifest,
    physical_system_id_from_config,
    validate_gateway_binding,
)
from tpen.run import load_config


REPO_ROOT = Path.cwd()
CONFIG_ROOT = REPO_ROOT / "experiments" / "atomistic" / "stage1" / "configs"
@pytest.fixture(autouse=True)
def _assert_external_packet_imports_reviewed_checkout() -> None:
    """Ensure an externally staged test exercises the exact cloned source."""

    assert (REPO_ROOT / ".git").exists()
    assert Path(stage1_protocol.__file__).resolve().is_relative_to(REPO_ROOT)


def _tiny_train_rows(tmp_path: Path) -> list[dict[str, object]]:
    protocol = Protocol(
        smoke_updates=1,
        smoke_walkers=1,
        lineage_updates=101,
        lineage_walkers=2,
        lineages=3,
        eval_streams=4,
        eval_walkers=2,
        eval_draws=2,
    )
    checkpoints = [7, 101]
    checkpoint_hashes = {
        (system, lineage, checkpoint): "d" * 64
        for system in SYSTEMS
        for lineage in range(protocol.lineages)
        for checkpoint in checkpoints
    }
    rows = build_manifest(
        protocol=protocol,
        source_sha="a" * 40,
        config_sha256="b" * 64,
        checkpoints=checkpoints,
        lineage_seeds=[11, 22, 33],
        checkpoint_sha256=checkpoint_hashes,
        results_root=str(tmp_path / "results"),
    )
    return [row for row in rows if row["kind"] == "train"]


@pytest.mark.parametrize("system", SYSTEMS)
def test_real_train_row_overrides_load_and_bind_checkpoint_schedule(
    system: str, tmp_path: Path
) -> None:
    """The launcher's real overrides must survive public OmegaConf loading."""

    row = next(row for row in _tiny_train_rows(tmp_path) if row["system"] == system)
    overrides = row_overrides(row)
    assert validate_gateway_binding(
        binding=_binding_token(row),
        config=row["config"],
        overrides=overrides,
        kind="train",
    ) == row

    config = CONFIG_ROOT / system / "train.yaml"
    resolved = load_config(str(config), overrides=overrides)
    assert list(resolved.callbacks[4].schedule.updates) == [7, 101]

    with pytest.raises(ProtocolError):
        validate_gateway_binding(
            binding=_binding_token(row),
            config=row["config"],
            overrides=overrides[:-1],
            kind="train",
        )
    bad_overrides = [
        "checkpoint_steps=[7,100]" if item.startswith(
            "checkpoint_steps="
        ) else item
        for item in overrides
    ]
    with pytest.raises(ProtocolError):
        validate_gateway_binding(
            binding=_binding_token(row),
            config=row["config"],
            overrides=bad_overrides,
            kind="train",
        )


def test_physical_id_rejects_wrong_operator_under_existing_kinetic_slot(
    tmp_path: Path,
) -> None:
    """A slot-preserving operator substitution must not pass registry checks."""

    source = CONFIG_ROOT / "he" / "eval.yaml"
    cfg = OmegaConf.load(source)
    cfg.hamiltonian_terms.kinetic = {
        "_target_": "tpen.physics.potential.HarmonicTrap",
        "omega": 0.5,
    }
    mutated = tmp_path / "he-wrong-kinetic-operator.yaml"
    OmegaConf.save(cfg, mutated)

    descriptor = instantiate(load_config(str(mutated)).system_descriptor)
    assert {term.slot for term in descriptor.terms} == {
        "kinetic",
        "electron_nucleus",
        "electron_electron",
    }
    with pytest.raises(ProtocolError, match="operator"):
        physical_system_id_from_config(mutated)


def test_registry_geometry_control_accepts_baseline_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    """The geometry control distinguishes the unchanged He config from drift."""

    source = CONFIG_ROOT / "he" / "eval.yaml"
    assert physical_system_id_from_config(source)

    cfg = OmegaConf.load(source)
    cfg.system.nuclei.positions[0][0] = 0.25
    mutated = tmp_path / "he-wrong-geometry.yaml"
    OmegaConf.save(cfg, mutated)
    with pytest.raises(ProtocolError, match="geometry"):
        physical_system_id_from_config(mutated)


def test_registry_charge_control_remains_exact_within_position_tolerance(
    tmp_path: Path,
) -> None:
    """Nuclear charges are exact integers even when the position tolerance is loose."""

    source = CONFIG_ROOT / "he" / "eval.yaml"
    cfg = OmegaConf.load(source)
    cfg.hamiltonian_terms.electron_nucleus.atoms = {
        "_target_": "tpen.data.atomic_configuration.AtomicConfiguration",
        "positions": {
            "_target_": "torch.tensor",
            "data": [[0.0, 0.0, 0.0]],
        },
        "charges": {"_target_": "torch.tensor", "data": [2.0000005]},
    }
    mutated = tmp_path / "he-nearby-nuclear-charge.yaml"
    OmegaConf.save(cfg, mutated)
    with pytest.raises(ProtocolError, match="geometry"):
        physical_system_id_from_config(mutated)
