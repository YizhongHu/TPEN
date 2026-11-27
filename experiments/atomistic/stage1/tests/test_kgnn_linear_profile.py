"""Contracts for the opt-in Hooke-2e linear interaction profile."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate

from experiments.atomistic.stage1 import protocol
from experiments.atomistic.stage1.protocol import (
    HOOKE2_LINEAR_PROFILE,
    HOOKE2_LINEAR_MAX_ORDER,
    HOOKE2_LINEAR_MAX_VIRTUAL_ORDER,
    HOOKE2_LINEAR_PATH_POLICY,
    STAGE1_PROTOCOL,
    build_manifest,
    config_identities,
)
from experiments.atomistic.stage1.tests.test_act50_profiles import _rows
from experiments.atomistic.stage1.train import build_train_config
from tpen.data.batch import ElectronBatch
from tpen.nn import CompositeMixing, EquivariantMixing, LinearEquivariantMixing
from tpen.run import run_from_config


CONFIG_ROOT = Path(__file__).parents[1] / "configs"
LINEAR_CONFIG = CONFIG_ROOT / "hooke2" / "linear.yaml"
NATIVE_HOOKE2_CONFIG_SHA256 = "a7296e0c52a082ab1949557b00732cf88468a7489e854e430182dcb02f8a698c"
N2_LINEAR_GRADIENT_FREE_PARAMETER_NAMES = frozenset({
    "stack.layers.0.mixing.producers.0.weights.3",
    "stack.layers.0.mixing.producers.0.weights.4",
})


def test_linear_profile_is_opt_in_and_native_hooke_configs_are_unchanged() -> None:
    """The profile marker changes identity; the default pair recipe does not."""

    native = config_identities(config_root=CONFIG_ROOT, systems=("hooke2",))
    profiled = config_identities(
        config_root=CONFIG_ROOT, systems=("hooke2",), science_profile=HOOKE2_LINEAR_PROFILE,
    )
    assert native == {"hooke2": NATIVE_HOOKE2_CONFIG_SHA256}
    assert profiled["hooke2"] != native["hooke2"]
    assert HOOKE2_LINEAR_PROFILE not in protocol.HOOKE2_ACT50_PROFILES
    assert HOOKE2_LINEAR_PROFILE in protocol.HOOKE2_ACT50_ALL_PROFILES


def test_linear_profile_header_records_the_n2_virtual3_caveat() -> None:
    header = LINEAR_CONFIG.read_text(encoding="utf-8")
    assert "max_order=2" in header
    assert "virtual-order-3 supports are empty" in header
    assert "product removal with nonlinear pair-self" in header
    assert "not third-particle messaging" in header


def test_linear_profile_resolves_one_linear_producer_and_shared_layout() -> None:
    row = _rows(HOOKE2_LINEAR_PROFILE)[0]
    cfg = build_train_config(row)
    layer_cfg = cfg.model.layers[0]
    assert cfg.stage1_linear_profile.interaction_mode == "linear"
    assert cfg.stage1_linear_profile.max_order == HOOKE2_LINEAR_MAX_ORDER
    assert cfg.stage1_linear_profile.max_virtual_order == HOOKE2_LINEAR_MAX_VIRTUAL_ORDER
    assert cfg.stage1_linear_profile.max_virtual_order == 3
    assert cfg.stage1_linear_profile.path_policy == HOOKE2_LINEAR_PATH_POLICY
    assert layer_cfg.mixing._target_ == "tpen.nn.CompositeMixing"
    assert layer_cfg.mixing.producers[0]._target_ == "tpen.nn.LinearEquivariantMixing"
    assert layer_cfg.layout.max_virtual_order == HOOKE2_LINEAR_MAX_VIRTUAL_ORDER
    assert layer_cfg.layout.max_virtual_order == 3

    model = instantiate(cfg.model).to(dtype=torch.float64)
    layer = model.stack.layers[0]
    assert isinstance(layer.mixing, CompositeMixing)
    assert len(layer.mixing.producers) == 1
    assert isinstance(layer.mixing.producers[0], LinearEquivariantMixing)
    assert not isinstance(layer.mixing.producers[0], EquivariantMixing)
    assert layer.mixing.layout.fingerprint == layer.path_aggregation.layout.fingerprint
    assert layer.mixing.layout.fingerprint == layer.layout.fingerprint


def test_linear_profile_n2_empty_support_order3_and_finite_forward() -> None:
    row = _rows(HOOKE2_LINEAR_PROFILE)[0]
    cfg = build_train_config(row)
    model = instantiate(cfg.model).to(dtype=torch.float64)
    positions = torch.tensor(
        [[[0.3, -0.2, 0.8], [-0.1, 0.5, 0.3]],
         [[1.0, 0.0, 0.5], [0.0, -0.5, 1.5]]],
        dtype=torch.float64,
    )
    batch = ElectronBatch(
        positions=positions,
        spins=torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64),
    )
    layer = model.stack.layers[0]
    feature = model.embedding(batch)
    interaction = layer.mixing(feature)
    order2 = interaction.blocks[2]
    # coordinate_neighbor order-2 has identity then two replacement paths;
    # N=2 cannot enumerate the support-order-3 replacements.
    assert order2.shape[2] == 3
    assert torch.isfinite(order2).all()
    assert torch.count_nonzero(order2[:, :, 1:]) == 0
    output = model(batch)
    assert torch.isfinite(output.logabs).all()
    output.logabs.square().mean().backward()
    gradient_free = {
        name for name, parameter in model.named_parameters()
        if parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    }
    assert gradient_free == N2_LINEAR_GRADIENT_FREE_PARAMETER_NAMES
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for name, parameter in model.named_parameters()
        if name not in N2_LINEAR_GRADIENT_FREE_PARAMETER_NAMES
    )


def test_linear_profile_stage1_diagnostics_cover_inactive_paths() -> None:
    """Missing N=2 path gradients remain observable, not silently ungrouped."""

    cfg = build_train_config(_rows(HOOKE2_LINEAR_PROFILE)[0])
    callback_by_target = {entry._target_: entry for entry in cfg.callbacks}
    for target in ("tpen.callback.ModuleNorms", "tpen.callback.OptimizerState"):
        groups = callback_by_target[target].groups
        assert groups.layer_0_mixing == ["stack.layers.0.mixing"]
        assert groups.layer_0_path_aggregation == ["stack.layers.0.path_aggregation"]
    assert "tpen.callback.ParameterActivity" in callback_by_target
    assert "tpen.callback.RawTrainingBatchRecorder" in callback_by_target


def test_linear_profile_runs_one_tiny_cpu_vmc_step(tmp_path: Path) -> None:
    """The composed profile reaches one real VMC update on a tiny CPU batch."""

    row = _rows(HOOKE2_LINEAR_PROFILE)[0]
    cfg = build_train_config(row)
    cfg.run.root = str(tmp_path)
    cfg.run.run_id = "linear-one-step"
    cfg.experiment.name = "kgnn_linear"
    cfg.experiment.sector = "stage1"
    cfg.runtime.device = "cpu"
    cfg.sampler.n_walkers = 2
    cfg.sampler.burn_in = 1
    cfg.sampler.n_steps = 1
    cfg.trainer.max_steps = 1
    assert run_from_config(
        cfg, config_path=str(CONFIG_ROOT / "hooke2" / "train.yaml"),
        command="pytest", raise_exceptions=True,
    ) == 0


@pytest.mark.parametrize("profile", [HOOKE2_LINEAR_PROFILE])
def test_linear_profile_keeps_stage1_manifest_contract(profile: str) -> None:
    rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=config_identities(
            config_root=CONFIG_ROOT, systems=("hooke2",), science_profile=profile,
        ),
        checkpoints=list(range(0, 50_001, 1_000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/external/kgnn-linear-test",
        systems=("hooke2",),
        phase=1,
        device="cpu",
        science_profile=profile,
        include_smoke=False,
        auto_overlay=True,
    )
    train_rows = [row for row in rows if row["kind"] == "train"]
    assert len(train_rows) == 3
    assert {row["science_profile"] for row in train_rows} == {profile}
    assert {row["variant"] for row in train_rows} == {"linear"}
    assert {row["updates"] for row in train_rows} == {50_000}
    assert {len(row["checkpoint_steps"]) for row in train_rows} == {51}
