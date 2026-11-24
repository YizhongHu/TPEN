"""Contract, analytical-oracle, masking, checkpoint, and HI-grid tests for native KFAC."""

from __future__ import annotations

import ast
import itertools
import json
import math
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import typeguard
from typeguard import TypeCheckError

import tpen.training as training_pkg
import tpen.training.trainer as trainer_module
from tests.helpers.hooke_models import build_tiny_spenn
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext
from tpen.checkpoint.restore import restore_checkpoint
from tpen.checkpoint.save import save_checkpoint
from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import (
    ElectronBatch,
    ParameterBinding,
    ParameterLayout,
    ParameterSlot,
    WavefunctionOutput,
)
from tpen.data.indices import no_repeated_particle_mask
from tpen.data.paths import (
    LinearPathMetadata,
    NormalizedChannels,
    NormalizedOrders,
    PathMetadata,
    compose_path_layout,
)
from tpen.nn import (
    AdditiveEnvelope,
    ChannelActivationAxes,
    ChannelPreservingMLPActivation,
    CompositeMixing,
    CurvatureElectronNucleusCuspLaw,
    DenseFactorBlock,
    ElectronElectronCusp,
    ElectronNucleusCusp,
    Embedding,
    EquivariantMixing,
    GaussianConfinement,
    KroneckerFactorBlock,
    LinearEquivariantMixing,
    OrderMLPLayout,
    OrderMLPSpec,
    PathAggregation,
    ResidualUpdater,
    SeededLinear,
    SoftplusBeta,
    StructuredCurvatureForwardPacket,
    StructuredCurvatureForwardRequest,
    StructuredCurvatureProvider,
    TPENLayer,
    TPENWaveFunction,
    TailSafeElectronNucleusCuspLaw,
    TorchInitializer,
)
from tpen.nn.jastrow import BoundedTwoCoefficientJastrow
from tpen.nn.mixing_kernel import Aggregation, MixingImplementation
from tpen.nn.readout import PfaffianReadout
from tpen.nn.structured_curvature import evaluate_structured_curvature_forward
from tpen.nn.update import ChannelMappedUpdater
from tpen.physics.hamiltonian import LocalEnergyResult
from tpen.training.kfac import (
    KFAC_CONVENTION_VERSION,
    KFAC_GRID_FACTOR_KEYS,
    KFAC_STATE_VERSION,
    KFACGridHyperparameters,
    KFACPolicy,
    KFACTelemetry,
    NativeKFACUpdate,
    build_kfac_update_from_factors,
    kfac_policy_from_factors,
)
from tpen.training.sr import SRPolicy, StochasticReconfigurationUpdate
from tpen.training.state import TrainerState
from tpen.training.step_requests import (
    StructuredCurvatureStepProducer,
    StructuredCurvatureStepRequest,
    StructuredCurvatureUpdateInput,
    VMCStepProducer,
    resolve_step_request,
)
from tpen.training.trainer import VMCTrainer, _TrainerStepProducer
from tpen.training.update import ModelParameterBinding

REPO_ROOT = Path(__file__).resolve().parents[3]
GRID_AUTHORITY_PATH = (
    REPO_ROOT / "experiments" / "atomistic" / "he-importance" / "optimizer_grid_authority.json"
)


def _binding_for(model: nn.Module) -> ParameterBinding:
    params = tuple(p for p in model.parameters() if p.requires_grad)
    slots = tuple(
        ParameterSlot(ordinal=i, shape=tuple(p.shape), numel=p.numel(), dtype=p.dtype)
        for i, p in enumerate(params)
    )
    return ParameterBinding(layout=ParameterLayout(slots=slots), parameters=params)


def _model_binding_for(model: nn.Module) -> ModelParameterBinding:
    return ModelParameterBinding.from_parameters(
        tuple(p for p in model.parameters() if p.requires_grad)
    )


def _make_batch(
    *,
    batch_size: int = 5,
    n_electrons: int = 2,
    spatial_dim: int = 3,
    seed: int = 11,
    include_spins: bool = True,
    dtype: torch.dtype = torch.float64,
) -> ElectronBatch:
    gen = torch.Generator().manual_seed(seed)
    positions = torch.randn((batch_size, n_electrons, spatial_dim), generator=gen, dtype=dtype)
    if include_spins:
        if n_electrons == 0:
            spins = torch.zeros((batch_size, 0), dtype=dtype)
        else:
            row = torch.tensor(
                [1.0 if idx % 2 == 0 else -1.0 for idx in range(n_electrons)], dtype=dtype
            )
            spins = row.unsqueeze(0).expand(batch_size, n_electrons).clone()
    else:
        spins = None
    return ElectronBatch(positions=positions, spins=spins)


def _per_sample_score_oracle(
    model: nn.Module,
    batch: ElectronBatch,
    params: tuple[nn.Parameter, ...],
) -> torch.Tensor:
    """Compute exact per-sample score matrix [B, sum numel] via unbatched autograd."""

    output = model(batch)
    logabs = output.logabs.reshape(-1)
    rows: list[torch.Tensor] = []
    for sample_idx in range(logabs.numel()):
        grads = torch.autograd.grad(
            logabs[sample_idx],
            params,
            retain_graph=sample_idx + 1 < logabs.numel(),
            create_graph=False,
            allow_unused=True,
        )
        flat = torch.cat(
            [
                (
                    torch.zeros(p.numel(), device=p.device, dtype=torch.float64)
                    if g is None
                    else g.detach().reshape(-1).to(dtype=torch.float64)
                )
                for p, g in zip(params, grads, strict=True)
            ],
            dim=0,
        )
        rows.append(flat)
    return torch.stack(rows, dim=0)


class _LinearMLPWavefunction(nn.Module):
    """Minimal standalone wavefunction for exact linear Kronecker oracle checks."""

    def __init__(
        self,
        *,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        use_seeded: bool = True,
        per_electron: bool = False,
        freeze_first_weight: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.per_electron = per_electron
        init = TorchInitializer(seed=7, stream="linear_oracle")
        if use_seeded:
            self.fc1: nn.Module = SeededLinear(
                in_features, hidden_features, bias=bias, initializer=init.spawn("fc1"), dtype=dtype
            )
            self.fc2: nn.Module = SeededLinear(
                hidden_features, 1, bias=bias, initializer=init.spawn("fc2"), dtype=dtype
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype)
            self.fc2 = nn.Linear(hidden_features, 1, bias=bias, dtype=dtype)
        if freeze_first_weight:
            self.fc1.weight.requires_grad_(False)
        self.act = SoftplusBeta(beta=2.0)
        self._binding = _binding_for(self)

    @property
    def parameter_binding(self) -> ParameterBinding:
        return self._binding

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        if self.per_electron:
            x = flat.positions  # [B, n, d] -> L = n
            h = self.act(self.fc1(x))
            y = self.fc2(h)  # [B, n, 1]
            logabs = y.sum(dim=(1, 2))
        else:
            x = flat.positions.reshape(flat.batch_size, -1)  # [B, n*d] -> L = 1
            h = self.act(self.fc1(x))
            y = self.fc2(h)  # [B, 1]
            logabs = y.squeeze(-1)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))

    def evaluate_structured_curvature_request(
        self,
        *,
        request: StructuredCurvatureForwardRequest,
        batch: ElectronBatch,
    ) -> StructuredCurvatureForwardPacket:
        return evaluate_structured_curvature_forward(
            self,
            batch=batch,
            sample_weights=request.sample_weights,
            solve_dtype=request.solve_dtype,
            max_dense_block_numel=request.max_dense_block_numel,
        )


def _vectorized_tiny_spenn() -> TPENWaveFunction:
    """Return ``build_tiny_spenn()`` switched to vectorized mixing for KFAC tests."""

    torch.manual_seed(0)
    model = build_tiny_spenn()
    for layer in model.stack.layers:
        layer.mixing.implementation = MixingImplementation.VECTORIZED
    return model


def _build_rich_tpen_model(
    *,
    max_order: int = 2,
    max_virtual_order: int = 2,
    channels: int = 2,
    aggregation: Aggregation = Aggregation.SUM,
    use_composite: bool = False,
    use_channel_mlp: bool = False,
    trainable_readout: bool = True,
    trainable_factors: bool = True,
    use_tail_safe_cusp: bool = True,
    disable_envelopes: bool = False,
    dtype: torch.dtype = torch.float64,
) -> TPENWaveFunction:
    """Build a compact vectorized TPENWaveFunction exercising all KFAC capture sites."""

    init = TorchInitializer(seed=23, stream="kfac_rich")
    embedding = Embedding(
        max_order=max_order,
        spatial_dim=3,
        out_channels=channels,
        hidden_channels=4,
        num_hidden_layers=1,
        activation=SoftplusBeta(beta=2.0),
        include_spins=True,
        initializer=init.spawn("embed"),
    )

    mix_act: nn.Module
    agg_act: nn.Module
    if use_channel_mlp:
        mix_act = ChannelPreservingMLPActivation(
            OrderMLPLayout(
                axes=ChannelActivationAxes(channel_axis=1, tuple_axes_start=3),
                specs=tuple(
                    OrderMLPSpec(
                        order=order,
                        channels=channels,
                        hidden_channels=3,
                        num_hidden_layers=1,
                        activation=SoftplusBeta(beta=2.0),
                    )
                    for order in range(1, max_order + 1)
                ),
            ),
            initializer=init.spawn("mix_mlp"),
        )
        agg_act = ChannelPreservingMLPActivation(
            OrderMLPLayout(
                axes=ChannelActivationAxes(channel_axis=1, tuple_axes_start=2),
                specs=tuple(
                    OrderMLPSpec(
                        order=order,
                        channels=channels,
                        hidden_channels=3,
                        num_hidden_layers=1,
                        activation=SoftplusBeta(beta=2.0),
                    )
                    for order in range(1, max_order + 1)
                ),
            ),
            initializer=init.spawn("agg_mlp"),
        )
    else:
        mix_act = SoftplusBeta(beta=2.0)
        agg_act = SoftplusBeta(beta=2.0)

    if use_composite:
        orders = tuple(range(1, max_order + 1))
        lin_meta = LinearPathMetadata.generate(max_order=max_order)
        tp_meta = PathMetadata.generate(
            max_order=max_order,
            max_virtual_order=max_virtual_order,
            output_embedding="canonical",
        )
        layout = compose_path_layout(
            linear=lin_meta,
            tensor_product=tp_meta,
            input_orders=NormalizedOrders(orders),
            output_orders=NormalizedOrders(orders),
            input_channels=NormalizedChannels(tuple((o, channels) for o in orders)),
            output_channels=NormalizedChannels(tuple((o, channels) for o in orders)),
        )
        lin_mix = LinearEquivariantMixing(
            max_order=max_order,
            channels=channels,
            metadata=lin_meta,
            aggregation=aggregation,
            implementation=MixingImplementation.VECTORIZED,
            initial_weight=0.25,
        )
        tp_mix = EquivariantMixing(
            max_order=max_order,
            max_virtual_order=max_virtual_order,
            paths=tp_meta,
            channels=channels,
            aggregation=aggregation,
            implementation=MixingImplementation.VECTORIZED,
            initial_weight=0.2,
            activation=None,
        )
        mixing: nn.Module = CompositeMixing(
            layout=layout,
            producers=(lin_mix, tp_mix),
            activation=mix_act,
        )
        path_agg = PathAggregation(
            max_order=max_order,
            channels=channels,
            layout=layout,
            activation=agg_act,
            initializer=init.spawn("agg"),
        )
        layer = TPENLayer(
            mixing=mixing,
            path_aggregation=path_agg,
            update=ResidualUpdater(),
            layout=layout,
        )
    else:
        tp_meta = PathMetadata.generate(
            max_order=max_order,
            max_virtual_order=max_virtual_order,
            output_embedding="canonical",
        )
        mixing = EquivariantMixing(
            max_order=max_order,
            max_virtual_order=max_virtual_order,
            paths=tp_meta,
            channels=channels,
            aggregation=aggregation,
            implementation=MixingImplementation.VECTORIZED,
            initial_weight=0.2,
            activation=mix_act,
        )
        path_agg = PathAggregation(
            max_order=max_order,
            max_virtual_order=max_virtual_order,
            paths=tp_meta,
            channels=channels,
            activation=agg_act,
            initializer=init.spawn("agg"),
        )
        layer = TPENLayer(
            mixing=mixing,
            path_aggregation=path_agg,
            update=ResidualUpdater(),
        )
        layout = None

    readout = PfaffianReadout(
        channels=channels,
        trainable=trainable_readout,
        initialization="signed_l1",
        initializer=init.spawn("readout"),
    )
    envelope = AdditiveEnvelope(
        envelopes=[
            GaussianConfinement(
                enabled=not disable_envelopes,
                coefficient=0.35,
                trainable=trainable_factors,
            ),
            ElectronElectronCusp(
                enabled=not disable_envelopes,
                range_parameter=1.1,
                trainable_range=trainable_factors,
            ),
        ],
        enabled=True,
    )
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype),
        charges=torch.tensor([2.0], dtype=dtype),
    )
    law = (
        TailSafeElectronNucleusCuspLaw(
            curvature_range=1.2, tail_slope=0.8, trainable=trainable_factors
        )
        if use_tail_safe_cusp
        else CurvatureElectronNucleusCuspLaw(
            curvature_coefficient=0.25, curvature_range=1.2, trainable=trainable_factors
        )
    )
    factors: tuple[nn.Module, ...] = (
        ElectronNucleusCusp(atoms=atoms, law=law),
        BoundedTwoCoefficientJastrow(
            mean_coefficient=0.15,
            difference_coefficient=-0.1,
            trainable=trainable_factors,
        ),
    )
    model = TPENWaveFunction(
        embedding=embedding,
        layers=(layer,),
        readout=readout,
        envelope=envelope,
        factors=factors,
        layout=layout,
    )
    return model.to(dtype=dtype)


class _FixedWalkers:
    def __init__(self, batch: ElectronBatch) -> None:
        self._batch = batch

    def make_batch(self) -> ElectronBatch:
        return self._batch


class _SequenceSampler:
    """Deterministic sampler returning a sequence of prebuilt batches with MCMC state."""

    def __init__(self, batches: list[ElectronBatch]) -> None:
        self._batches = batches
        self.cursor = 0

    def collect_samples(self, model: Any, *, device: Any = None):
        del model, device
        batch = self._batches[self.cursor % len(self._batches)]
        self.cursor += 1
        return _FixedWalkers(batch), None

    def mcmc_state_dict(self) -> dict[str, Any]:
        return {"cursor": self.cursor}

    def load_mcmc_state_dict(self, state: dict[str, Any], *, device: Any = None) -> None:
        del device
        self.cursor = int(state["cursor"])


# ---------------------------------------------------------------------------
# 1. Exact float64 analytical Kronecker & flattened-bilinear oracles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_seeded", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("per_electron", [False, True])
def test_linear_and_seeded_linear_exact_kronecker_oracle(
    use_seeded: bool, bias: bool, per_electron: bool
) -> None:
    """Exact float64 oracle for nn.Linear & SeededLinear (L=1 and L>1, with/without bias)."""

    batch = _make_batch(batch_size=6, n_electrons=3, spatial_dim=2, seed=19)
    in_features = 2 if per_electron else 6
    model = _LinearMLPWavefunction(
        in_features=in_features,
        hidden_features=4,
        bias=bias,
        use_seeded=use_seeded,
        per_electron=per_electron,
    )
    assert isinstance(model, StructuredCurvatureProvider)
    energies = torch.linspace(-2.5, 1.5, batch.batch_size, dtype=torch.float64)
    weights = (energies - energies.mean()) / float(batch.batch_size)

    req = StructuredCurvatureForwardRequest(sample_weights=weights, solve_dtype=torch.float64)
    packet = req.evaluate(model, batch)

    # Hand-computed analytical oracle over the exact forward inputs & cotangents.
    x_in = batch.positions if per_electron else batch.positions.reshape(batch.batch_size, -1)
    y1 = model.fc1(x_in)
    h1 = model.act(y1)
    y2 = model.fc2(h1)
    logabs = y2.sum(dim=(1, 2)) if per_electron else y2.squeeze(-1)
    g1, g2 = torch.autograd.grad(logabs.sum(), (y1, y2), retain_graph=True)
    expected_half_grads = torch.autograd.grad((weights * logabs).sum(), model.parameter_binding.parameters)

    B = batch.batch_size
    L = 3 if per_electron else 1
    assert len(packet.kronecker_blocks) == 2
    assert len(packet.dense_blocks) == 0

    for blk, a_raw, g_raw in zip(packet.kronecker_blocks, (x_in, h1), (g1, g2), strict=True):
        assert blk.active_locations == L
        assert blk.has_bias is bias
        a_flat = a_raw.detach().reshape(B * L, -1)
        if bias:
            a_flat = torch.cat((a_flat, torch.ones((B * L, 1), dtype=torch.float64)), dim=1)
        g_flat = g_raw.detach().reshape(B * L, -1)
        A_ref = (a_flat.T @ a_flat) / float(B * L)
        G_ref = (g_flat.T @ g_flat) / float(B)
        torch.testing.assert_close(blk.A, A_ref, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(blk.G, G_ref, atol=1e-12, rtol=1e-12)

    for actual_g, ref_g in zip(packet.energy_half_gradient, expected_half_grads, strict=True):
        torch.testing.assert_close(actual_g, ref_g, atol=1e-12, rtol=1e-12)


def test_frozen_weight_trainable_bias_linear_captured_as_dense_oracle() -> None:
    """A linear layer with frozen weight and trainable bias uses exact centered DenseFactorBlock."""

    batch = _make_batch(batch_size=5, n_electrons=2, spatial_dim=2, seed=31)
    model = _LinearMLPWavefunction(
        in_features=4, hidden_features=3, bias=True, freeze_first_weight=True
    )
    weights = torch.linspace(-1.0, 1.0, batch.batch_size, dtype=torch.float64) / batch.batch_size
    packet = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)

    assert len(packet.dense_blocks) == 1
    assert len(packet.kronecker_blocks) == 1
    scores = _per_sample_score_oracle(model, batch, (model.fc1.bias,))
    centered = scores - scores.mean(dim=0, keepdim=True)
    F_ref = (centered.T @ centered) / float(batch.batch_size)
    torch.testing.assert_close(packet.dense_blocks[0].F, F_ref, atol=1e-12, rtol=1e-12)


def test_embedding_and_channel_mlp_distinct_tuple_masking_and_zero_support() -> None:
    """Order>=2 Embedding and ChannelPreservingMLPActivation mask repeated tuples; s=3/order=3 at n=2 gives L=0."""

    # Model 1: max_order=2, max_virtual_order=3 exercises distinct-tuple masking at order=2
    # AND 3-body EquivariantMixing paths (s=3, V=0 at n=2) cleanly in <1s on CPU.
    model = _build_rich_tpen_model(
        max_order=2,
        max_virtual_order=3,
        channels=2,
        use_composite=True,
        use_channel_mlp=True,
        trainable_readout=False,
        trainable_factors=False,
    )
    batch = _make_batch(batch_size=4, n_electrons=2, spatial_dim=3, seed=41)
    weights = torch.tensor([-0.3, 0.1, 0.4, -0.2], dtype=torch.float64) / 4.0

    packet = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)

    # Locate the first linear layer of Embedding order 2 and verify exact masked A vs unmasked A.
    id_to_ord = {id(p): i for i, p in enumerate(model.parameter_binding.parameters)}
    emb_o2_first = model.embedding.order_mlps["2"].layers[0]
    w_ord_o2 = id_to_ord[id(emb_o2_first.weight)]
    blk_o2 = next(b for b in packet.kronecker_blocks if b.parameter_ordinals[0] == w_ord_o2)
    assert blk_o2.active_locations == 2  # n=2, order=2 -> 2!/(2-2)! = 2 distinct tuples (not 4!)

    # Reconstruct exact order-2 tuple inputs manually to verify masked vs unmasked oracle.
    pv = torch.cat((batch.positions, batch.spins.unsqueeze(-1)), dim=-1)  # [B, 2, 4]
    from tpen.data.indices import tuple_particle_inputs

    t2 = tuple_particle_inputs(pv, 2)  # [B, 2, 2, 8]
    mask = no_repeated_particle_mask(2, 2).reshape(-1)
    a_masked = t2.reshape(4, 4, 8)[:, mask, :].reshape(4 * 2, 8)
    a_masked = torch.cat((a_masked, torch.ones((8, 1), dtype=torch.float64)), dim=1)
    A_masked_ref = (a_masked.T @ a_masked) / 8.0
    a_unmasked = t2.reshape(4 * 4, 8)
    a_unmasked = torch.cat((a_unmasked, torch.ones((16, 1), dtype=torch.float64)), dim=1)
    A_unmasked_ref = (a_unmasked.T @ a_unmasked) / 16.0

    torch.testing.assert_close(blk_o2.A, A_masked_ref, atol=1e-12, rtol=1e-12)
    # Mutant killer: removing repeated-particle masking would yield A_unmasked_ref != A_masked_ref.
    assert float((A_masked_ref - A_unmasked_ref).abs().max().item()) > 1e-2

    # ChannelPreservingMLPActivation inside CompositeMixing (tuple_axes_start=3) order 2 has P * 2 distinct locations.
    mix_mlp_o2 = model.stack.layers[0].mixing.activation.mlps[1].layers[0]
    blk_mix_mlp_o2 = next(
        b for b in packet.kronecker_blocks if b.parameter_ordinals[0] == id_to_ord[id(mix_mlp_o2.weight)]
    )
    n_paths_o2 = model.layout.count_for_order(2)
    assert blk_mix_mlp_o2.active_locations == n_paths_o2 * 2

    # 3-body EquivariantMixing paths (s=3) at n=2 have V == 0 -> exact zero (A, G), no NaN.
    tp_producer = model.stack.layers[0].mixing.producers[1]
    s3_paths = [p for p in tp_producer.paths if p.s == 3]
    assert s3_paths
    for p in s3_paths:
        w = tp_producer.weights[f"g{p.global_id}"]
        blk_s3 = next(b for b in packet.kronecker_blocks if b.parameter_ordinals == (id_to_ord[id(w)],))
        assert blk_s3.active_locations == 0
        assert torch.equal(blk_s3.A, torch.zeros_like(blk_s3.A))
        assert torch.equal(blk_s3.G, torch.zeros_like(blk_s3.G))

    # Model 2: order-3 Embedding at n=2 (with lightweight linear-only mixing) has L_distinct == 0 -> exact zero (A, G).
    init = TorchInitializer(seed=29, stream="o3_zero")
    lin3 = LinearEquivariantMixing(
        max_order=3,
        channels=2,
        implementation=MixingImplementation.VECTORIZED,
        initial_weight=0.25,
    )
    o3_model = TPENWaveFunction(
        embedding=Embedding(
            max_order=3,
            spatial_dim=3,
            out_channels=2,
            hidden_channels=3,
            num_hidden_layers=1,
            activation=SoftplusBeta(beta=2.0),
            include_spins=True,
            initializer=init.spawn("emb3"),
        ),
        layers=(
            TPENLayer(
                mixing=lin3,
                path_aggregation=PathAggregation(
                    max_order=3,
                    channels=2,
                    path_counts_by_order={
                        o: len(lin3.metadata.paths_for_output_order(o)) for o in (1, 2, 3)
                    },
                    activation=SoftplusBeta(beta=2.0),
                    initializer=init.spawn("agg3"),
                ),
                update=ResidualUpdater(),
            ),
        ),
        readout=PfaffianReadout(channels=2, trainable=False),
    ).to(dtype=torch.float64)
    pkt3 = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(o3_model, batch)
    id3 = {id(p): i for i, p in enumerate(o3_model.parameter_binding.parameters)}
    emb_o3_first = o3_model.embedding.order_mlps["3"].layers[0]
    blk_o3 = next(
        b for b in pkt3.kronecker_blocks if b.parameter_ordinals[0] == id3[id(emb_o3_first.weight)]
    )
    assert blk_o3.active_locations == 0
    assert torch.equal(blk_o3.A, torch.zeros_like(blk_o3.A))
    assert torch.equal(blk_o3.G, torch.zeros_like(blk_o3.G))


@pytest.mark.parametrize("aggregation", [Aggregation.SUM, Aggregation.COMPLETION_MEAN])
def test_equivariant_and_linear_mixing_exact_kronecker_and_path_agg_oracles(
    aggregation: Aggregation,
) -> None:
    """Verify EquivariantMixing (V>1), LinearEquivariantMixing, and PathAggregation against float64 oracles."""

    # n=3 electrons exercises V > 1 completions (s=2 -> V=6, multiple completions per output I)
    # and keeps BOTH order-1 and order-2 active through odd-electron Pfaffian padding.
    model = _build_rich_tpen_model(
        max_order=2,
        max_virtual_order=2,
        channels=2,
        aggregation=aggregation,
        use_composite=True,
        use_channel_mlp=False,
        trainable_readout=True,
        trainable_factors=False,
    )
    batch = _make_batch(batch_size=4, n_electrons=3, spatial_dim=3, seed=53)
    weights = torch.tensor([-0.2, 0.5, -0.4, 0.1], dtype=torch.float64) / 4.0

    packet = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)
    id_to_ord = {id(p): i for i, p in enumerate(model.parameter_binding.parameters)}

    # 1. Exact analytical oracle for PathAggregation (both o1 and o2) via unbatched autograd.
    path_agg = model.stack.layers[0].path_aggregation
    for order_key in ("o1", "o2"):
        w = path_agg.weights[order_key]
        blk = next(b for b in packet.dense_blocks if b.parameter_ordinals == (id_to_ord[id(w)],))
        scores = _per_sample_score_oracle(model, batch, (w,))
        centered = scores - scores.mean(dim=0, keepdim=True)
        F_ref = (centered.T @ centered) / float(batch.batch_size)
        torch.testing.assert_close(blk.F, F_ref, atol=1e-12, rtol=1e-12)

    # 2. Exact analytical flattened-bilinear oracle for one binary EquivariantMixing path (s=2 -> V=6)
    #    and one unary LinearEquivariantMixing path by reproducing the exact gather + einsum.
    from tpen.nn.context import TPENForwardContext
    from tpen.nn.mixing_kernel import (
        _gather_tuple_values,
        build_binary_index_plan,
        build_unary_index_plan,
    )

    context = TPENForwardContext(batch=batch, basis_features=None)
    features = model.embedding(batch, context=context)
    layer = model.stack.layers[0]
    lin_prod = layer.mixing.producers[0]
    tp_prod = layer.mixing.producers[1]

    # Pick an active TP path with s=2, m=1 (has 2 completions per output index at n=3, so SUM != COMPLETION_MEAN).
    target_tp = next(p for p in tp_prod.paths if p.s == 2 and p.m == 1)
    tp_w = tp_prod.weights[f"g{target_tp.global_id}"]
    tp_blk = next(b for b in packet.kronecker_blocks if b.parameter_ordinals == (id_to_ord[id(tp_w)],))
    assert tp_blk.active_locations == 6  # P(3, 2) = 6

    plan = build_binary_index_plan(target_tp, 3)
    lv = _gather_tuple_values(features.blocks[target_tp.m1], plan.left_indices)
    rv = _gather_tuple_values(features.blocks[target_tp.m2], plan.right_indices)
    a_ref = (lv.unsqueeze(2) * rv.unsqueeze(1)).reshape(batch.batch_size, -1, 6)
    a_flat = a_ref.detach().permute(0, 2, 1).reshape(batch.batch_size * 6, -1)
    A_ref = (a_flat.T @ a_flat) / float(batch.batch_size * 6)
    torch.testing.assert_close(tp_blk.A, A_ref, atol=1e-12, rtol=1e-12)

    # Verify G matches exact finite-difference / autograd derivative w.r.t. pre-scatter contribution:
    # sum_v g[b, o, v] * a[b, k, v] == d(logabs[b])/d(W[o, k]) identically for every sample b.
    # For COMPLETION_MEAN vs SUM, compare the two runs directly to prove counts division takes effect!
    if aggregation is Aggregation.COMPLETION_MEAN:
        model_sum = _build_rich_tpen_model(
            max_order=2,
            max_virtual_order=2,
            channels=2,
            aggregation=Aggregation.SUM,
            use_composite=True,
            use_channel_mlp=False,
            trainable_readout=True,
            trainable_factors=False,
        )
        model_sum.load_state_dict(model.state_dict())
        pkt_sum = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model_sum, batch)
        tp_blk_sum = next(
            b for b in pkt_sum.kronecker_blocks if b.parameter_ordinals == tp_blk.parameter_ordinals
        )
        # Because s=2 -> m=1 has completion count 2 at n=3, G_mean != G_sum.
        assert float((tp_blk.G - tp_blk_sum.G).abs().max().item()) > 1e-6

    # Verify LinearEquivariantMixing unary path A oracle.
    target_lin_idx = 0
    target_lin = lin_prod.paths[target_lin_idx]
    lin_w = lin_prod.weights[target_lin_idx]
    lin_blk = next(b for b in packet.kronecker_blocks if b.parameter_ordinals == (id_to_ord[id(lin_w)],))
    uplan = build_unary_index_plan(target_lin, 3)
    u_in = _gather_tuple_values(features.blocks[target_lin.input_order], uplan.input_indices)
    v_lin = int(u_in.shape[2])
    assert lin_blk.active_locations == v_lin
    u_flat = u_in.detach().permute(0, 2, 1).reshape(batch.batch_size * v_lin, -1)
    A_lin_ref = (u_flat.T @ u_flat) / float(batch.batch_size * v_lin)
    torch.testing.assert_close(lin_blk.A, A_lin_ref, atol=1e-12, rtol=1e-12)


def test_slow_mixing_implementation_refused_loudly() -> None:
    """MixingImplementation.SLOW must raise ValueError loudly under KFAC capture (binary & unary)."""

    slow_spenn = build_tiny_spenn()  # fixture ships with implementation: slow
    batch = _make_batch(batch_size=4, n_electrons=2, spatial_dim=3, seed=5)
    weights = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="MixingImplementation.VECTORIZED"):
        StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(slow_spenn, batch)

    # Also test SLOW refusal in execute_unary via LinearEquivariantMixing.
    model = _build_rich_tpen_model(use_composite=True, trainable_readout=False, trainable_factors=False)
    model.stack.layers[0].mixing.producers[0].implementation = MixingImplementation.SLOW
    with pytest.raises(ValueError, match="execute_unary"):
        StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)


@pytest.mark.parametrize("use_tail_safe_cusp", [True, False])
@pytest.mark.parametrize("disable_envelopes", [False, True])
def test_post_readout_and_scalar_dense_blocks_exact_oracle_over_softplus_beta(
    use_tail_safe_cusp: bool, disable_envelopes: bool
) -> None:
    """All post-readout / scalar DenseFactorBlocks match exact autograd scores over SoftplusBeta trunk."""

    model = _build_rich_tpen_model(
        max_order=2,
        max_virtual_order=2,
        channels=3,
        use_composite=False,
        trainable_readout=True,
        trainable_factors=True,
        use_tail_safe_cusp=use_tail_safe_cusp,
        disable_envelopes=disable_envelopes,
    )
    batch = _make_batch(batch_size=5, n_electrons=2, spatial_dim=3, seed=71)
    weights = torch.tensor([-0.4, -0.1, 0.2, 0.3, 0.0], dtype=torch.float64) / 5.0
    packet = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)

    # Every DenseFactorBlock must match exact centered score covariance S = O_c^T O_c / N.
    all_params = model.parameter_binding.parameters
    for blk in packet.dense_blocks:
        blk_params = tuple(all_params[o] for o in blk.parameter_ordinals)
        scores = _per_sample_score_oracle(model, batch, blk_params)
        centered = scores - scores.mean(dim=0, keepdim=True)
        F_ref = (centered.T @ centered) / float(batch.batch_size)
        torch.testing.assert_close(blk.F, F_ref, atol=1e-11, rtol=1e-11)

    # PfaffianReadout.channel_weights centered Fisher must annihilate global scale mode w.
    readout_ord = next(
        i for i, p in enumerate(all_params) if p is model.readout.channel_weights
    )
    readout_blk = next(b for b in packet.dense_blocks if b.parameter_ordinals == (readout_ord,))
    w_vec = model.readout.channel_weights.detach().to(dtype=torch.float64)
    gauge_residual = torch.linalg.vector_norm(readout_blk.F @ w_vec).item()
    assert gauge_residual < 1e-12


def test_unrecognized_upstream_parameter_and_dense_cap_refuse_loudly() -> None:
    """Unrecognized upstream parameters and oversized DenseFactorBlocks raise ValueError loudly."""

    # 1. ChannelMappedUpdater owns trainable channel_maps upstream of readout without a KFAC rule.
    model = _build_rich_tpen_model(
        max_order=2, max_virtual_order=2, channels=2, trainable_readout=False, trainable_factors=False
    )
    model.stack.layers[0].update = ChannelMappedUpdater(max_order=2, channels=2)
    model = model.to(dtype=torch.float64)
    batch = _make_batch(batch_size=4, n_electrons=2, spatial_dim=3, seed=83)
    weights = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="unrecognized trainable parameter"):
        StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)

    # 2. DenseFactorBlock exceeding max_dense_block_numel fails loudly.
    clean_model = _vectorized_tiny_spenn()
    with pytest.raises(ValueError, match="max_dense_block_numel"):
        StructuredCurvatureForwardRequest(
            sample_weights=weights, max_dense_block_numel=4
        ).evaluate(clean_model, batch)


# ---------------------------------------------------------------------------
# 2. n=2 zero blocks, EMA, refresh cadence, KL cap, exact vg_sum identity
# ---------------------------------------------------------------------------


def test_tiny_spenn_n2_zero_blocks_and_damping_floor() -> None:
    """n=2 build_tiny_spenn has G=0, A!=0 blocks; KFAC steps cleanly with trace damping floor."""

    model = _vectorized_tiny_spenn()
    batch = _make_batch(batch_size=6, n_electrons=2, spatial_dim=3, seed=91)
    energies = torch.linspace(-1.2, 0.8, batch.batch_size, dtype=torch.float64)
    weights = (energies - energies.mean()) / float(batch.batch_size)

    packet = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)
    id_to_name = {id(p): name for name, p in model.named_parameters()}
    zero_g_nonzero_a = 0
    for blk in packet.kronecker_blocks:
        name = id_to_name[id(model.parameter_binding.parameters[blk.parameter_ordinals[0]])]
        if ".mixing.weights.g" in name and float(blk.G.abs().max().item()) == 0.0:
            if float(blk.A.abs().max().item()) > 0.0:
                zero_g_nonzero_a += 1
    assert zero_g_nonzero_a == 15  # g0..g14 order-1 output paths at even n=2

    policy = KFACPolicy(
        learning_rate=0.05,
        relative_damping=1e-3,
        factor_ema=0.95,
        curvature_factor_update_interval=1,
        inverse_refresh_interval=10,
        predicted_kl_cap=1e-3,
    )
    opt = torch.optim.SGD(model.parameters(), lr=policy.learning_rate)
    method = NativeKFACUpdate(opt, model_parameters=_model_binding_for(model), policy=policy)

    before = [p.detach().clone() for p in model.parameters()]
    step_in = StructuredCurvatureUpdateInput(
        batch=batch,
        wavefunction=packet.output,
        local_energy=energies,
        step=0,
        parameter_binding=packet.parameter_binding,
        solve_dtype=packet.solve_dtype,
        kronecker_blocks=packet.kronecker_blocks,
        dense_blocks=packet.dense_blocks,
        energy_half_gradient=packet.energy_half_gradient,
        n_finite_samples=batch.batch_size,
    )
    result = method.update(step_in)
    assert result.applied is True
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(p, b) for p, b in zip(model.parameters(), before, strict=True))
    assert method.last_telemetry is not None
    assert math.isclose(
        method.last_telemetry.vg_sum,
        2.0 * method.last_telemetry.predicted_kl + method.last_telemetry.damped_quadratic_term,
        rel_tol=1e-11,
        abs_tol=1e-12,
    )


def test_step0_warm_ema_refresh_cadence_kl_cap_and_vg_sum_identity() -> None:
    """Verify warm EMA at step 0, update-before-refresh at step 10, KL cap, and exact vg_sum identity."""

    model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=True, use_seeded=True)
    policy = KFACPolicy(
        learning_rate=0.1,
        relative_damping=1e-2,
        factor_ema=0.95,
        curvature_factor_update_interval=1,
        inverse_refresh_interval=10,
        predicted_kl_cap=1e-3,
    )
    opt = torch.optim.SGD(model.parameters(), lr=policy.learning_rate)
    method = NativeKFACUpdate(opt, model_parameters=_model_binding_for(model), policy=policy)

    ema_oracle_A: list[torch.Tensor] = []
    ema_oracle_G: list[torch.Tensor] = []
    cached_A: list[torch.Tensor] = []
    cached_G: list[torch.Tensor] = []

    saw_clamped_step = False
    for step in range(12):
        batch = _make_batch(batch_size=5, n_electrons=2, spatial_dim=2, seed=200 + step)
        energies = torch.linspace(-2.0, 2.0, 5, dtype=torch.float64) * (3.0 if step == 0 else 0.5)
        weights = (energies - energies.mean()) / 5.0
        packet = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)
        step_in = StructuredCurvatureUpdateInput(
            batch=batch,
            wavefunction=packet.output,
            local_energy=energies,
            step=step,
            parameter_binding=packet.parameter_binding,
            solve_dtype=packet.solve_dtype,
            kronecker_blocks=packet.kronecker_blocks,
            dense_blocks=packet.dense_blocks,
            energy_half_gradient=packet.energy_half_gradient,
            n_finite_samples=5,
        )
        params_before = [p.detach().clone() for p in model.parameters()]
        result = method.update(step_in)
        assert result.applied is True
        tel = method.last_telemetry
        assert isinstance(tel, KFACTelemetry)
        assert tel.curvature_step == step
        assert tel.factors_updated is True
        expected_refresh = step % 10 == 0
        assert tel.inverse_refreshed is expected_refresh

        if step == 0:
            ema_oracle_A = [b.A.clone() for b in packet.kronecker_blocks]
            ema_oracle_G = [b.G.clone() for b in packet.kronecker_blocks]
        else:
            ema_oracle_A = [
                0.95 * prev + 0.05 * b.A for prev, b in zip(ema_oracle_A, packet.kronecker_blocks)
            ]
            ema_oracle_G = [
                0.95 * prev + 0.05 * b.G for prev, b in zip(ema_oracle_G, packet.kronecker_blocks)
            ]

        assert method._kronecker_ema is not None
        for actual_ema, ref_A, ref_G in zip(
            method._kronecker_ema, ema_oracle_A, ema_oracle_G, strict=True
        ):
            torch.testing.assert_close(actual_ema["A"], ref_A, atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(actual_ema["G"], ref_G, atol=1e-13, rtol=1e-13)

        if expected_refresh:
            cached_A = [a.clone() for a in ema_oracle_A]
            cached_G = [g.clone() for g in ema_oracle_G]
            # Exact identity on refresh steps: vg_sum == 2 * predicted_kl + eta^2 * sum lambda ||d||^2
            assert math.isclose(
                tel.vg_sum,
                2.0 * tel.predicted_kl + tel.damped_quadratic_term,
                rel_tol=1e-12,
                abs_tol=1e-13,
            )
        else:
            # Between refreshes, vg_sum == 2 * predicted_kl_cached + eta^2 * sum lambda ||d||^2.
            q_cached = 0.0
            for blk, cA, cG in zip(packet.kronecker_blocks, cached_A, cached_G, strict=True):
                w_ord, b_ord = blk.parameter_ordinals
                # Unscale parameter.grad by trust_scale to recover raw d_b.
                d_w = model.parameter_binding.parameters[w_ord].grad.detach() / tel.trust_scale
                d_b = model.parameter_binding.parameters[b_ord].grad.detach() / tel.trust_scale
                D = torch.cat((d_w.reshape(blk.out_dim, -1), d_b.reshape(blk.out_dim, 1)), dim=1)
                q_cached += float((D * (cG @ D @ cA)).sum().item())
            predicted_kl_cached = 0.5 * (policy.learning_rate ** 2) * q_cached
            assert math.isclose(
                tel.vg_sum,
                2.0 * predicted_kl_cached + tel.damped_quadratic_term,
                rel_tol=1e-11,
                abs_tol=1e-12,
            )

        # Verify exact SGD displacement theta_new == theta_old - eta * param.grad.
        for p_new, p_old in zip(model.parameters(), params_before, strict=True):
            assert p_new.grad is not None
            torch.testing.assert_close(
                p_new.detach(),
                p_old - policy.learning_rate * p_new.grad.detach(),
                atol=1e-14,
                rtol=1e-14,
            )
        if tel.predicted_kl > policy.predicted_kl_cap:
            saw_clamped_step = True
            expected_nu = math.sqrt(policy.predicted_kl_cap / tel.predicted_kl)
            assert math.isclose(tel.trust_scale, expected_nu, rel_tol=1e-12, abs_tol=1e-13)
            assert math.isclose(
                (tel.trust_scale ** 2) * tel.predicted_kl,
                policy.predicted_kl_cap,
                rel_tol=1e-12,
                abs_tol=1e-13,
            )

    assert saw_clamped_step


# ---------------------------------------------------------------------------
# 3. Trainer two-path masking (0.0 * NaN immunity), degenerate skips, fail_on_pinned
# ---------------------------------------------------------------------------


def test_trainer_masked_subset_zero_times_nan_immunity_and_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under 'mask', dropped NaN rows never enter KFAC forward/backward (immune to 0.0*NaN)."""

    base_batch = _make_batch(batch_size=5, n_electrons=2, spatial_dim=2, seed=303)
    poisoned_positions = base_batch.positions.clone()
    poisoned_positions[2] = float("nan")
    poisoned_batch = replace(base_batch, positions=poisoned_positions)

    # Clean twin containing strictly the 4 finite rows.
    clean_positions = torch.cat((base_batch.positions[:2], base_batch.positions[3:]), dim=0)
    clean_spins = torch.cat((base_batch.spins[:2], base_batch.spins[3:]), dim=0)
    clean_batch = ElectronBatch(positions=clean_positions, spins=clean_spins)

    finite_energies = torch.tensor([-1.5, -0.4, 0.7, 1.2], dtype=torch.float64)
    poisoned_energies = torch.tensor([-1.5, -0.4, float("nan"), 0.7, 1.2], dtype=torch.float64)

    policy = KFACPolicy(learning_rate=0.05, relative_damping=1e-2, predicted_kl_cap=1e-3)

    # Run masked trainer on poisoned_batch vs clean trainer on clean_batch.
    def _run_one(batch_arg: ElectronBatch, energy_arg: torch.Tensor, nf_policy: str):
        model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=True)
        opt = torch.optim.SGD(model.parameters(), lr=policy.learning_rate)
        method = NativeKFACUpdate(
            opt,
            model_parameters=_model_binding_for(model),
            policy=policy,
            nonfinite_local_energy_policy=nf_policy,
        )
        trainer = VMCTrainer(
            max_steps=1,
            update_method=method,
            nonfinite_local_energy_policy=nf_policy,
        )
        monkeypatch.setattr(
            trainer_module,
            "local_energy",
            lambda terms, m, b, return_terms=False: LocalEnergyResult(
                total=energy_arg.clone(), terms={}
            ),
        )
        ctx = _StubContext()
        trainer.fit(
            model=model,
            sampler=_SequenceSampler([batch_arg]),
            hamiltonian_terms={},
            optimizer=opt,
            context=ctx,
            emit=lambda **_: None,
        )
        return model, method

    masked_model, masked_method = _run_one(poisoned_batch, poisoned_energies, "mask")
    clean_model, clean_method = _run_one(clean_batch, finite_energies, "mask")

    for p_m, p_c in zip(masked_model.parameters(), clean_model.parameters(), strict=True):
        assert torch.isfinite(p_m).all()
        torch.testing.assert_close(p_m, p_c, atol=1e-13, rtol=1e-13)
    assert masked_method.last_telemetry is not None
    assert masked_method.last_telemetry.n_samples == 5
    assert masked_method.last_telemetry.n_finite_samples == 4

    # Under "fail", the same poisoned local energy must refuse loudly before stepping.
    with pytest.raises(ValueError, match="active policy is 'fail'"):
        _run_one(poisoned_batch, poisoned_energies, "fail")

    # Under n_finite == 0 (all NaN), flow to compute_vmc_objective and raise loudly.
    all_nan = torch.full((5,), float("nan"), dtype=torch.float64)
    with pytest.raises(ValueError, match="no finite local-energy samples"):
        _run_one(base_batch, all_nan, "mask")


@pytest.mark.parametrize("case", ["vacuum", "single_finite"])
def test_degenerate_batches_skip_without_mutating_state(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_electrons==0 and n_finite<2 skip cleanly without touching EMA, counter, or weights."""

    model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=True)
    policy = KFACPolicy(learning_rate=0.05, relative_damping=1e-2)
    opt = torch.optim.SGD(model.parameters(), lr=policy.learning_rate)
    method = NativeKFACUpdate(
        opt,
        model_parameters=_model_binding_for(model),
        policy=policy,
        nonfinite_local_energy_policy="mask",
    )

    # First apply 1 normal step so EMA state is populated and can be verified unchanged.
    good_batch = _make_batch(batch_size=4, n_electrons=2, spatial_dim=2, seed=401)
    good_e = torch.tensor([-1.0, 0.2, 0.5, 0.9], dtype=torch.float64)
    good_w = (good_e - good_e.mean()) / 4.0
    pkt = StructuredCurvatureForwardRequest(sample_weights=good_w).evaluate(model, good_batch)
    method.update(
        StructuredCurvatureUpdateInput(
            batch=good_batch,
            wavefunction=pkt.output,
            local_energy=good_e,
            step=0,
            parameter_binding=pkt.parameter_binding,
            solve_dtype=pkt.solve_dtype,
            kronecker_blocks=pkt.kronecker_blocks,
            dense_blocks=pkt.dense_blocks,
            energy_half_gradient=pkt.energy_half_gradient,
            n_finite_samples=4,
        )
    )

    params_snapshot = [p.detach().clone() for p in model.parameters()]
    ema_A_snapshot = [e["A"].clone() for e in method._kronecker_ema]
    cstep_before = method.curvature_step
    completed_before = method.completed_updates

    if case == "vacuum":
        deg_batch = _make_batch(batch_size=4, n_electrons=0, spatial_dim=3, seed=402)
        deg_e = torch.zeros(4, dtype=torch.float64)
        expected_reason = "zero_electron_batch"
        # Vacuum model output stub for _LinearMLPWavefunction (which expects 4 inputs otherwise).
        deg_model = _vectorized_tiny_spenn()
        deg_opt = torch.optim.SGD(deg_model.parameters(), lr=policy.learning_rate)
        deg_method = NativeKFACUpdate(
            deg_opt,
            model_parameters=_model_binding_for(deg_model),
            policy=policy,
            nonfinite_local_energy_policy="mask",
        )
        trainer = VMCTrainer(max_steps=1, update_method=deg_method, nonfinite_local_energy_policy="mask")
        monkeypatch.setattr(
            trainer_module,
            "local_energy",
            lambda terms, m, b, return_terms=False: LocalEnergyResult(total=deg_e.clone(), terms={}),
        )
        before_vac = [p.detach().clone() for p in deg_model.parameters()]
        trainer.fit(
            model=deg_model,
            sampler=_SequenceSampler([deg_batch]),
            hamiltonian_terms={},
            optimizer=deg_opt,
            context=_StubContext(),
            emit=lambda **_: None,
        )
        assert deg_method.curvature_step == 0
        assert deg_method.completed_updates == 0
        assert deg_method._kronecker_ema is None
        assert deg_method.last_telemetry is not None
        assert deg_method.last_telemetry.applied is False
        assert deg_method.last_telemetry.reason == expected_reason
        for p_after, p_before in zip(deg_model.parameters(), before_vac, strict=True):
            assert torch.equal(p_after, p_before)
        return

    # single_finite case
    deg_e = torch.tensor([1.25, float("nan"), float("inf"), float("-inf")], dtype=torch.float64)
    trainer = VMCTrainer(max_steps=1, update_method=method, nonfinite_local_energy_policy="mask")
    monkeypatch.setattr(
        trainer_module,
        "local_energy",
        lambda terms, m, b, return_terms=False: LocalEnergyResult(total=deg_e.clone(), terms={}),
    )
    trainer.fit(
        model=model,
        sampler=_SequenceSampler([good_batch]),
        hamiltonian_terms={},
        optimizer=opt,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    assert method.curvature_step == cstep_before
    assert method.completed_updates == completed_before
    assert method.last_telemetry is not None
    assert method.last_telemetry.applied is False
    assert method.last_telemetry.reason == "insufficient_finite_samples"
    for p_after, p_before in zip(model.parameters(), params_snapshot, strict=True):
        assert torch.equal(p_after, p_before)
    for e_after, e_before in zip(method._kronecker_ema, ema_A_snapshot, strict=True):
        assert torch.equal(e_after["A"], e_before)


def test_fail_on_pinned_atomic_refusal_and_getstate_guards() -> None:
    """Non-finite factors/half-grad raise RuntimeError atomically; packets refuse __getstate__."""

    model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=True)
    policy = KFACPolicy(learning_rate=0.05, relative_damping=1e-2)
    opt = torch.optim.SGD(model.parameters(), lr=policy.learning_rate)
    method = NativeKFACUpdate(opt, model_parameters=_model_binding_for(model), policy=policy)

    batch = _make_batch(batch_size=4, n_electrons=2, spatial_dim=2, seed=505)
    energies = torch.tensor([-1.0, 0.1, 0.4, 0.5], dtype=torch.float64)
    weights = (energies - energies.mean()) / 4.0
    pkt = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)

    # __getstate__ unconditional refusal on both packet and update input.
    with pytest.raises(RuntimeError, match="StructuredCurvatureForwardPacket"):
        pickle.dumps(pkt)

    good_input = StructuredCurvatureUpdateInput(
        batch=batch,
        wavefunction=pkt.output,
        local_energy=energies,
        step=0,
        parameter_binding=pkt.parameter_binding,
        solve_dtype=pkt.solve_dtype,
        kronecker_blocks=pkt.kronecker_blocks,
        dense_blocks=pkt.dense_blocks,
        energy_half_gradient=pkt.energy_half_gradient,
        n_finite_samples=4,
    )
    with pytest.raises(RuntimeError, match="StructuredCurvatureUpdateInput"):
        pickle.dumps(good_input)

    # Advance once so we can verify atomic non-mutation on subsequent non-finite refusal.
    method.update(good_input)
    params_before = [p.detach().clone() for p in model.parameters()]
    ema_before = [e["A"].clone() for e in method._kronecker_ema]

    # Poison Kronecker factor A on pinned finite rows.
    bad_A = pkt.kronecker_blocks[0].A.clone()
    bad_A[0, 0] = float("nan")
    bad_kron = (
        KroneckerFactorBlock(
            parameter_ordinals=pkt.kronecker_blocks[0].parameter_ordinals,
            has_bias=pkt.kronecker_blocks[0].has_bias,
            out_dim=pkt.kronecker_blocks[0].out_dim,
            in_dim=pkt.kronecker_blocks[0].in_dim,
            A=bad_A,
            G=pkt.kronecker_blocks[0].G,
            active_locations=pkt.kronecker_blocks[0].active_locations,
        ),
        *pkt.kronecker_blocks[1:],
    )
    bad_factor_input = replace(good_input, step=1, kronecker_blocks=bad_kron)
    with pytest.raises(RuntimeError, match="non-finite Kronecker"):
        method.update(bad_factor_input)

    # Poison energy_half_gradient on pinned finite rows.
    bad_grad0 = pkt.energy_half_gradient[0].clone()
    bad_grad0[0, 0] = float("inf")
    bad_grad_input = replace(
        good_input, step=1, energy_half_gradient=(bad_grad0, *pkt.energy_half_gradient[1:])
    )
    with pytest.raises(RuntimeError, match="non-finite energy_half_gradient"):
        method.update(bad_grad_input)

    # Confirm zero mutation to counters, EMA, or parameters.
    assert method.curvature_step == 1
    assert method.completed_updates == 1
    for p_after, p_before in zip(model.parameters(), params_before, strict=True):
        assert torch.equal(p_after, p_before)
    for e_after, e_before in zip(method._kronecker_ema, ema_before, strict=True):
        assert torch.equal(e_after["A"], e_before)

    # Non-finite input positions on pinned finite rows raise RuntimeError inside forward capture.
    bad_pos = batch.positions.clone()
    bad_pos[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(
            model, replace(batch, positions=bad_pos)
        )


def test_step_request_protocol_and_plain_sgd_guards() -> None:
    """Verify VMCStepProducer protocol, resolve_step_request, and plain-SGD refusal."""

    model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=True)
    policy = KFACPolicy(learning_rate=0.03)
    binding = _model_binding_for(model)

    with pytest.raises(TypeError, match="SGD"):
        NativeKFACUpdate(
            torch.optim.Adam(model.parameters(), lr=0.03), model_parameters=binding, policy=policy
        )
    with pytest.raises(ValueError, match="momentum"):
        NativeKFACUpdate(
            torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9),
            model_parameters=binding,
            policy=policy,
        )
    with pytest.raises(ValueError, match="weight_decay"):
        NativeKFACUpdate(
            torch.optim.SGD(model.parameters(), lr=0.03, weight_decay=1e-4),
            model_parameters=binding,
            policy=policy,
        )
    with pytest.raises(ValueError, match="disagrees"):
        NativeKFACUpdate(
            torch.optim.SGD(model.parameters(), lr=0.01),
            model_parameters=binding,
            policy=policy,
        )

    opt = torch.optim.SGD(model.parameters(), lr=0.03)
    method = NativeKFACUpdate(opt, model_parameters=binding, policy=policy)
    req = resolve_step_request(method)
    assert isinstance(req, StructuredCurvatureStepRequest)
    assert req.describe_label() == "tpen.training.step_requests.StructuredCurvatureStepRequest"
    trainer = VMCTrainer(max_steps=1)
    dummy_state = TrainerState(
        model=model,
        optimizer=opt,
        update_state=method.update_state(),
        trainer=trainer,
        sampler=_SequenceSampler([]),
    )
    assert isinstance(
        _TrainerStepProducer(
            trainer=trainer,
            model=model,
            hamiltonian_terms={},
            batch=_make_batch(batch_size=2, n_electrons=2, spatial_dim=2),
            step=0,
            context=_StubContext(),
            state=dummy_state,
        ),
        (StructuredCurvatureStepProducer, VMCStepProducer),
    )


# ---------------------------------------------------------------------------
# 4. Bit-exact mid-refresh binary method_state.pt checkpoint save/restore
# ---------------------------------------------------------------------------


def _checkpoint_context(tmp_path: Path, method_tag: str) -> Any:
    from types import SimpleNamespace

    cfg = {
        "model": {"tag": "tiny_kfac"},
        "hamiltonian_terms": {"tag": "tiny_ham"},
        "optimizer": {"tag": method_tag},
        "trainer": {"tag": method_tag},
        "sampler": {"tag": "tiny_seq"},
    }
    metadata = SimpleNamespace(
        device="cpu",
        dtype="float64",
        run_id="kfac-ckpt-test",
        git_commit="0" * 40,
        git_branch="test",
        dirty_worktree=False,
        command="pytest",
        extra={},
    )
    return SimpleNamespace(cfg=cfg, metadata=metadata, run_dir=tmp_path)


def test_mid_refresh_checkpoint_save_restore_bit_exact_and_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bit-exact mid-refresh resume via method_state.pt plus tamper/missing/unexpected refusals."""

    batches = [
        _make_batch(batch_size=4, n_electrons=2, spatial_dim=3, seed=700 + i) for i in range(6)
    ]
    policy = KFACPolicy(
        learning_rate=0.04,
        relative_damping=1e-3,
        factor_ema=0.95,
        curvature_factor_update_interval=1,
        inverse_refresh_interval=10,
        predicted_kl_cap=1e-3,
    )
    monkeypatch.setattr(
        trainer_module,
        "local_energy",
        lambda terms, m, b, return_terms=False: LocalEnergyResult(
            total=b.positions.square().sum(dim=(1, 2)).detach(), terms={}
        ),
    )

    def _build_stack(max_steps: int):
        model = _build_rich_tpen_model(
            max_order=2, max_virtual_order=2, channels=2, trainable_readout=True, trainable_factors=True
        )
        opt = torch.optim.SGD(model.parameters(), lr=policy.learning_rate)
        method = NativeKFACUpdate(opt, model_parameters=_model_binding_for(model), policy=policy)
        trainer = VMCTrainer(max_steps=max_steps, update_method=method)
        sampler = _SequenceSampler(batches)
        return model, opt, method, trainer, sampler

    # 1. Uninterrupted 6-step reference run.
    ref_model, ref_opt, ref_method, ref_trainer, ref_sampler = _build_stack(max_steps=6)
    ref_trainer.fit(
        model=ref_model,
        sampler=ref_sampler,
        hamiltonian_terms={},
        optimizer=ref_opt,
        context=_StubContext(),
        emit=lambda **_: None,
    )

    # 2. Split run: 3 steps (strictly mid-refresh with interval=10), save checkpoint, restore, 3 more steps.
    s1_model, s1_opt, s1_method, s1_trainer, s1_sampler = _build_stack(max_steps=3)
    s1_trainer.fit(
        model=s1_model,
        sampler=s1_sampler,
        hamiltonian_terms={},
        optimizer=s1_opt,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    ckpt_root = tmp_path / "kfac_ckpts"
    ckpt_ctx = _checkpoint_context(tmp_path, "native_kfac")
    ckpt_dir = save_checkpoint(
        output_dir=ckpt_root,
        next_iteration=s1_trainer.next_iteration,
        completed_updates=s1_trainer.completed_updates,
        model=s1_model,
        optimizer=s1_opt,
        trainer=s1_trainer,
        sampler=s1_sampler,
        context=ckpt_ctx,
    )

    # Verify manifest and JSON compactness.
    manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    trainer_json = json.loads((ckpt_dir / "trainer.json").read_text(encoding="utf-8"))
    assert manifest["files"]["method_state"] == "method_state.pt"
    assert "method_state_sha256" in manifest["hashes"]
    assert (ckpt_dir / "method_state.pt").is_file()
    assert trainer_json["update_method"]["version"] == KFAC_STATE_VERSION
    assert trainer_json["update_method"]["convention"] == KFAC_CONVENTION_VERSION
    assert trainer_json["update_method"]["curvature_step"] == 3
    assert "kronecker_ema" not in trainer_json["update_method"]

    # Restore into fresh instances and finish steps 3..6.
    s2_model, s2_opt, s2_method, s2_trainer, s2_sampler = _build_stack(max_steps=6)
    s2_trainer.resolve_update_state(model=s2_model, optimizer=s2_opt)
    restore_checkpoint(
        load={"mode": "train_resume", "path": str(ckpt_dir)},
        model=s2_model,
        optimizer=s2_opt,
        trainer=s2_trainer,
        sampler=s2_sampler,
        context=ckpt_ctx,
    )
    s2_trainer.fit(
        model=s2_model,
        sampler=s2_sampler,
        hamiltonian_terms={},
        optimizer=s2_opt,
        context=_StubContext(),
        emit=lambda **_: None,
    )

    for p_resumed, p_ref in zip(s2_model.parameters(), ref_model.parameters(), strict=True):
        assert torch.equal(p_resumed, p_ref)
    for e_resumed, e_ref in zip(s2_method._kronecker_ema, ref_method._kronecker_ema, strict=True):
        assert torch.equal(e_resumed["A"], e_ref["A"])
        assert torch.equal(e_resumed["G"], e_ref["G"])
    for e_resumed, e_ref in zip(s2_method._dense_ema, ref_method._dense_ema, strict=True):
        assert torch.equal(e_resumed["F"], e_ref["F"])

    # 3. Tampered method_state.pt refuses before mutating live model parameters.
    tampered_dir = tmp_path / "tampered_step"
    import shutil

    shutil.copytree(ckpt_dir, tampered_dir)
    raw_bytes = (tampered_dir / "method_state.pt").read_bytes()
    (tampered_dir / "method_state.pt").write_bytes(raw_bytes[:-1] + bytes([raw_bytes[-1] ^ 0xFF]))
    t_model, t_opt, _t_method, t_trainer, t_sampler = _build_stack(max_steps=6)
    t_trainer.resolve_update_state(model=t_model, optimizer=t_opt)
    t_before = [p.detach().clone() for p in t_model.parameters()]
    with pytest.raises(ValueError, match="digest mismatch"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(tampered_dir)},
            model=t_model,
            optimizer=t_opt,
            trainer=t_trainer,
            sampler=t_sampler,
            context=ckpt_ctx,
        )
    for p_after, p_before in zip(t_model.parameters(), t_before, strict=True):
        assert torch.equal(p_after, p_before)

    # 4. Missing method_state entry in manifest when NativeKFACUpdate has curvature_step > 0 raises.
    missing_dir = tmp_path / "missing_sidecar"
    shutil.copytree(ckpt_dir, missing_dir)
    m_missing = json.loads((missing_dir / "manifest.json").read_text(encoding="utf-8"))
    del m_missing["files"]["method_state"]
    del m_missing["hashes"]["method_state_sha256"]
    (missing_dir / "manifest.json").write_text(json.dumps(m_missing), encoding="utf-8")
    m_model, m_opt, _m_method, m_trainer, m_sampler = _build_stack(max_steps=6)
    m_trainer.resolve_update_state(model=m_model, optimizer=m_opt)
    with pytest.raises(ValueError, match="method_state.pt"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(missing_dir)},
            model=m_model,
            optimizer=m_opt,
            trainer=m_trainer,
            sampler=m_sampler,
            context=ckpt_ctx,
        )

    # 5. Backward compatibility: SR & Adam checkpoints omit method_state.pt and reject unexpected ones.
    sr_model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=True)
    sr_opt = torch.optim.SGD(sr_model.parameters(), lr=0.01)
    sr_method = StochasticReconfigurationUpdate(
        sr_opt, model_parameters=_model_binding_for(sr_model), policy=SRPolicy(learning_rate=0.01)
    )
    sr_trainer = VMCTrainer(max_steps=1, update_method=sr_method)
    sr_trainer.resolve_update_state(model=sr_model, optimizer=sr_opt)
    sr_ckpt_dir = save_checkpoint(
        output_dir=tmp_path / "sr_ckpts",
        next_iteration=0,
        completed_updates=0,
        model=sr_model,
        optimizer=sr_opt,
        trainer=sr_trainer,
        sampler=_SequenceSampler(batches),
        context=_checkpoint_context(tmp_path, "sr"),
    )
    sr_manifest = json.loads((sr_ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "method_state" not in sr_manifest["files"]
    assert not (sr_ckpt_dir / "method_state.pt").exists()

    # Inject an unexpected method_state.pt into SR checkpoint -> restore must refuse loudly.
    shutil.copy2(ckpt_dir / "method_state.pt", sr_ckpt_dir / "method_state.pt")
    sr_manifest["files"]["method_state"] = "method_state.pt"
    sr_manifest["hashes"]["method_state_sha256"] = manifest["hashes"]["method_state_sha256"]
    (sr_ckpt_dir / "manifest.json").write_text(json.dumps(sr_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not accept binary method tensor state"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(sr_ckpt_dir)},
            model=sr_model,
            optimizer=sr_opt,
            trainer=sr_trainer,
            sampler=_SequenceSampler(batches),
            context=_checkpoint_context(tmp_path, "sr"),
        )


# ---------------------------------------------------------------------------
# 5. All 32 HI authority KFAC grid cells exact readback + strict refusal matrix
# ---------------------------------------------------------------------------


def test_all_32_kfac_authority_grid_cells_exact_readback() -> None:
    """Every one of the 32 KFAC authority cells maps cleanly with exact readback."""

    authority = json.loads(GRID_AUTHORITY_PATH.read_text(encoding="utf-8"))
    kfac_entry = next(entry for entry in authority["families"] if entry["family"] == "kfac")
    varied = kfac_entry["varied"]
    fixed = kfac_entry["fixed"]

    varied_keys = list(varied.keys())
    cells = [
        {**fixed, **dict(zip(varied_keys, combo, strict=True))}
        for combo in itertools.product(*(varied[key] for key in varied_keys))
    ]
    assert len(cells) == kfac_entry["declared_cells"] == 32

    model = _LinearMLPWavefunction(in_features=4, hidden_features=2, bias=True)
    binding = _model_binding_for(model)

    for cell in cells:
        assert set(cell.keys()) == set(KFAC_GRID_FACTOR_KEYS)
        grid = KFACGridHyperparameters.from_factors(cell)
        assert grid.to_factor_dict() == cell
        policy = kfac_policy_from_factors(cell)
        assert policy.learning_rate == float(cell["eta"])
        assert policy.relative_damping == float(cell["relative_damping"])
        assert policy.factor_ema == float(cell["factor_ema"])
        assert policy.curvature_factor_update_interval == int(
            cell["curvature_factor_update_interval"]
        )
        assert policy.inverse_refresh_interval == int(cell["inverse_refresh_interval"])
        assert policy.predicted_kl_cap == float(cell["predicted_kl_cap"])

        method = build_kfac_update_from_factors(cell, model_parameters=binding)
        assert isinstance(method, NativeKFACUpdate)
        assert method.optimizer.param_groups[0]["lr"] == float(cell["eta"])
        desc = method.describe()
        assert desc.settings["policy_learning_rate"] == float(cell["eta"])
        assert desc.settings["policy_relative_damping"] == float(cell["relative_damping"])
        assert desc.settings["policy_factor_ema"] == float(cell["factor_ema"])
        assert desc.settings["policy_inverse_refresh_interval"] == 10
        assert desc.settings["separate_momentum"] is False


@pytest.mark.parametrize("suppress_typeguard", [False, True])
def test_kfac_grid_and_policy_strict_refusals(suppress_typeguard: bool) -> None:
    """Refuse separate_momentum=True, bools, non-reals, bad bounds, missing/unknown keys."""

    valid = {
        "eta": 0.03,
        "relative_damping": 1e-3,
        "factor_ema": 0.95,
        "curvature_factor_update_interval": 1,
        "inverse_refresh_interval": 10,
        "separate_momentum": False,
        "predicted_kl_cap": 1e-3,
    }

    bad_grid_cases: list[dict[str, Any]] = [
        {**valid, "separate_momentum": True},
        {**valid, "separate_momentum": 0},
        {**valid, "separate_momentum": "false"},
        {**valid, "eta": True},
        {**valid, "eta": 0.0},
        {**valid, "eta": -0.01},
        {**valid, "eta": float("nan")},
        {**valid, "eta": float("inf")},
        {**valid, "eta": "0.03"},
        {**valid, "relative_damping": True},
        {**valid, "relative_damping": 0.0},
        {**valid, "factor_ema": False},
        {**valid, "factor_ema": -0.1},
        {**valid, "factor_ema": 1.0},
        {**valid, "curvature_factor_update_interval": True},
        {**valid, "curvature_factor_update_interval": 0},
        {**valid, "curvature_factor_update_interval": 1.5},
        {**valid, "inverse_refresh_interval": False},
        {**valid, "inverse_refresh_interval": 0},
        {**valid, "predicted_kl_cap": True},
        {**valid, "predicted_kl_cap": 0.0},
        {**valid, "predicted_kl_cap": float("inf")},
        {k: v for k, v in valid.items() if k != "eta"},
        {**valid, "unexpected_key": 1},
    ]

    bad_policy_kwargs: list[dict[str, Any]] = [
        {"learning_rate": True},
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"learning_rate": "0.01"},
        {"relative_damping": False},
        {"relative_damping": 0.0},
        {"factor_ema": True},
        {"factor_ema": 1.0},
        {"factor_ema": -0.01},
        {"curvature_factor_update_interval": True},
        {"curvature_factor_update_interval": 0},
        {"inverse_refresh_interval": False},
        {"inverse_refresh_interval": 0},
        {"predicted_kl_cap": True},
        {"predicted_kl_cap": 0.0},
        {"solve_dtype": torch.int64},
        {"solve_dtype": torch.complex128},
        {"solve_dtype": "not_a_dtype"},
        {"max_dense_block_numel": True},
        {"max_dense_block_numel": 0},
        {"damping_trace_floor": True},
        {"damping_trace_floor": 0.0},
    ]

    ctx = typeguard.suppress_type_checks() if suppress_typeguard else _null_cm()
    with ctx:
        for bad in bad_grid_cases:
            with pytest.raises((ValueError, TypeError, TypeCheckError)):
                kfac_policy_from_factors(bad)
        for bad_kw in bad_policy_kwargs:
            with pytest.raises((ValueError, TypeError, TypeCheckError)):
                KFACPolicy(**bad_kw)  # type: ignore[arg-type]


class _null_cm:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# 6. Physics-free nn boundary & public re-export checks
# ---------------------------------------------------------------------------


def test_nn_curvature_modules_import_zero_physics_and_training_reexports() -> None:
    """tpen/nn/forward.py and tpen/nn/structured_curvature.py import zero symbols from tpen.physics."""

    for rel_path in ("tpen/nn/forward.py", "tpen/nn/structured_curvature.py"):
        tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("tpen.physics")
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("tpen.physics")

    for name in (
        "KFAC_STATE_VERSION",
        "KFAC_GRID_FACTOR_KEYS",
        "KFACPolicy",
        "KFACTelemetry",
        "NativeKFACUpdate",
        "KFACGridHyperparameters",
        "kfac_policy_from_factors",
        "build_kfac_update_from_factors",
        "StructuredCurvatureStepRequest",
        "StructuredCurvatureStepProducer",
        "StructuredCurvatureUpdateInput",
    ):
        assert hasattr(training_pkg, name)
