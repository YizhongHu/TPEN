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
from omegaconf import OmegaConf
from typeguard import TypeCheckError

import tpen.training as training_pkg
import tpen.training.trainer as trainer_module
from tests.helpers.hooke_models import build_tiny_spenn
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext
from tpen.checkpoint.hashing import file_sha256
from tpen.checkpoint.receipt import PAYLOAD_COMPONENT_NAMES, publication_receipt_path
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
        tp_paths = tuple(
            path
            for order in orders
            for path in tp_meta.paths_for_output_order(order)
        )
        tp_mix = EquivariantMixing(
            max_order=max_order,
            max_virtual_order=max_virtual_order,
            paths=tp_paths,
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

    def collect_samples(self, model: Any, *, reset: bool = False, device: Any = None, emitter=None):
        del model, reset, device, emitter
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

    # ChannelPreservingMLPActivation inside CompositeMixing (tuple_axes_start=3) order 2 has P * 2 distinct locations,
    # and inside PathAggregation (tuple_axes_start=2) order 2 has 2 distinct locations.
    mix_mlp_o2 = model.stack.layers[0].mixing.activation.mlps[1].layers[0]
    blk_mix_mlp_o2 = next(
        b for b in packet.kronecker_blocks if b.parameter_ordinals[0] == id_to_ord[id(mix_mlp_o2.weight)]
    )
    n_paths_o2 = model.layout.count_for_order(2)
    assert blk_mix_mlp_o2.active_locations == n_paths_o2 * 2

    # Independent float64 cotangent oracle G_masked_ref and per-sample score identity on masked
    # order-2 Embedding (n_extra=1) and order-2 ChannelPreservingMLPActivation (n_extra=P).
    captured_o2_outputs: dict[str, torch.Tensor] = {}
    emb_hook = emb_o2_first.register_forward_hook(
        lambda _mod, _inp, out: captured_o2_outputs.__setitem__("emb_o2", out)
    )
    mix_hook = mix_mlp_o2.register_forward_hook(
        lambda _mod, _inp, out: captured_o2_outputs.__setitem__("mix_o2", out)
    )
    try:
        masked_oracle_out = model(batch)
    finally:
        emb_hook.remove()
        mix_hook.remove()

    y_emb_o2 = captured_o2_outputs["emb_o2"]  # [B=4, n=2, n=2, hidden=4]
    y_mix_o2 = captured_o2_outputs["mix_o2"]  # [B=4, P=n_paths_o2, n=2, n=2, hidden=3]
    g_emb_o2_full = torch.autograd.grad(
        masked_oracle_out.logabs.sum(), y_emb_o2, retain_graph=True
    )[0].detach()
    g_mix_o2_full = torch.autograd.grad(
        masked_oracle_out.logabs.sum(), y_mix_o2, retain_graph=False
    )[0].detach()

    # Repeated-particle diagonal entries carry zero cotangent because Embedding masks them out.
    assert torch.equal(
        g_emb_o2_full.reshape(4, 4, 4)[:, ~mask, :],
        torch.zeros((4, 2, 4), dtype=torch.float64),
    )
    g_emb_o2_masked_3d = g_emb_o2_full.reshape(4, 4, 4)[:, mask, :]  # [B=4, L=2, out=4]
    g_emb_o2_flat = g_emb_o2_masked_3d.reshape(4 * 2, 4)
    G_masked_ref = (g_emb_o2_flat.T @ g_emb_o2_flat) / 4.0
    assert float(G_masked_ref.abs().max().item()) > 1e-8
    torch.testing.assert_close(blk_o2.G, G_masked_ref, atol=1e-12, rtol=1e-12)

    a_emb_o2_masked_3d = torch.cat(
        (t2.reshape(4, 4, 8)[:, mask, :], torch.ones((4, 2, 1), dtype=torch.float64)), dim=-1
    )
    emb_scores_from_ga = torch.einsum("blo,blk->bok", g_emb_o2_masked_3d, a_emb_o2_masked_3d)
    emb_w_scores_ref = _per_sample_score_oracle(model, batch, (emb_o2_first.weight,)).reshape(4, 4, 8)
    emb_b_scores_ref = _per_sample_score_oracle(model, batch, (emb_o2_first.bias,)).reshape(4, 4)
    torch.testing.assert_close(emb_scores_from_ga[:, :, :-1], emb_w_scores_ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(emb_scores_from_ga[:, :, -1], emb_b_scores_ref, atol=1e-12, rtol=1e-12)

    g_mix_o2_masked = g_mix_o2_full.reshape(4, n_paths_o2, 4, 3)[:, :, mask, :].reshape(
        4 * n_paths_o2 * 2, 3
    )
    G_mix_o2_ref = (g_mix_o2_masked.T @ g_mix_o2_masked) / 4.0
    assert float(G_mix_o2_ref.abs().max().item()) > 1e-8
    torch.testing.assert_close(blk_mix_mlp_o2.G, G_mix_o2_ref, atol=1e-12, rtol=1e-12)

    agg_mlp_o2 = model.stack.layers[0].path_aggregation.activation.mlps[1].layers[0]
    blk_agg_mlp_o2 = next(
        b for b in packet.kronecker_blocks if b.parameter_ordinals[0] == id_to_ord[id(agg_mlp_o2.weight)]
    )
    assert blk_agg_mlp_o2.active_locations == 2

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

    # 2. Exact analytical flattened-bilinear & unary oracles for binary EquivariantMixing (s=2 -> V=6)
    #    and unary LinearEquivariantMixing paths: independent float64 A_ref, G_ref, and exact
    #    per-sample score identity sum_v g_ref[b, o, v] * a_ref[b, k, v] == d(logabs[b]) / d(W[o, k]).
    from tpen.data.real import Interaction
    from tpen.nn.context import TPENForwardContext
    from tpen.nn.mixing_kernel import (
        _gather_tuple_values,
        build_binary_index_plan,
        build_unary_index_plan,
    )

    def _manual_scatter_path_block(
        contrib: torch.Tensor,
        flat_output_indices: torch.Tensor,
        *,
        n_particles: int,
        output_order: int,
        agg: Aggregation,
    ) -> torch.Tensor:
        """Scatter [B, O, V] contributions to [B, O, *(n,)*output_order] independently of mixing_kernel."""

        bsz, out_ch, _n_sup = contrib.shape
        n_out = n_particles ** output_order
        idx_list = flat_output_indices.tolist()
        slots: list[torch.Tensor] = []
        for flat_pos in range(n_out):
            matching = [v for v, dst in enumerate(idx_list) if dst == flat_pos]
            if not matching:
                slots.append(contrib.new_zeros((bsz, out_ch)))
            else:
                slot_sum = torch.stack([contrib[:, :, v] for v in matching], dim=0).sum(dim=0)
                if agg is Aggregation.COMPLETION_MEAN:
                    slot_sum = slot_sum / float(len(matching))
                slots.append(slot_sum)
        return torch.stack(slots, dim=-1).reshape(bsz, out_ch, *((n_particles,) * output_order))

    def _replace_path_slice(
        block: torch.Tensor, path_idx: int, new_slice: torch.Tensor
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if path_idx > 0:
            parts.append(block[:, :, :path_idx])
        parts.append(new_slice.unsqueeze(2))
        if path_idx + 1 < int(block.shape[2]):
            parts.append(block[:, :, path_idx + 1 :])
        return torch.cat(parts, dim=2)

    def _finish_logabs_from_raw_producers(
        feats: Any,
        raw_lin_blocks: list[torch.Tensor],
        raw_tp_blocks: list[torch.Tensor],
    ) -> torch.Tensor:
        blocks = [raw_lin_blocks[0]]
        for order in layer.mixing.layout.output_orders.values:
            cat_blk = torch.cat((raw_lin_blocks[order], raw_tp_blocks[order]), dim=2)
            if layer.mixing.activation is not None:
                cat_blk = layer.mixing.activation(cat_blk)
            blocks.append(cat_blk)
        interaction = Interaction(blocks, layer.mixing.layout)
        real_update = layer.path_aggregation(interaction)
        features_out = layer.update(feats, real_update)
        out = model.readout(features_out, batch)
        logabs_val = out.logabs
        if model.envelope is not None:
            logabs_val = logabs_val + model.envelope(batch)
        for factor in model.factors:
            logabs_val = logabs_val + factor(batch)
        return logabs_val

    context = TPENForwardContext(batch=batch, basis_features=None)
    layer = model.stack.layers[0]
    lin_prod = layer.mixing.producers[0]
    tp_prod = layer.mixing.producers[1]

    # Pick an active TP path with s=2, m=1 (has 2 completions per output index at n=3, so SUM != COMPLETION_MEAN).
    target_tp = next(p for p in tp_prod.paths if p.s == 2 and p.m == 1)
    tp_w = tp_prod.weights[f"g{target_tp.global_id}"]
    tp_blk = next(b for b in packet.kronecker_blocks if b.parameter_ordinals == (id_to_ord[id(tp_w)],))
    assert tp_blk.active_locations == 6  # P(3, 2) = 6

    features_tp = model.embedding(batch, context=context)
    plan = build_binary_index_plan(target_tp, 3)
    lv = _gather_tuple_values(features_tp.blocks[target_tp.m1], plan.left_indices)
    rv = _gather_tuple_values(features_tp.blocks[target_tp.m2], plan.right_indices)
    a_ref = (lv.detach().unsqueeze(2) * rv.detach().unsqueeze(1)).reshape(batch.batch_size, -1, 6)
    a_flat = a_ref.permute(0, 2, 1).reshape(batch.batch_size * 6, -1)
    A_ref = (a_flat.T @ a_flat) / float(batch.batch_size * 6)
    torch.testing.assert_close(tp_blk.A, A_ref, atol=1e-12, rtol=1e-12)

    raw_lin_tp = lin_prod.forward_pre_activation(features_tp)
    raw_tp_tp = tp_prod.forward_pre_activation(features_tp)
    tp_order_paths = [p for p in tp_prod.paths if p.m == target_tp.m]
    tp_path_idx = tp_order_paths.index(target_tp)
    out_dim_tp = int(tp_w.shape[0])
    in_dim_tp = int(a_ref.shape[1])
    contrib_tp = torch.einsum("ok,bkv->bov", tp_w.reshape(out_dim_tp, in_dim_tp), a_ref)
    scattered_tp = _manual_scatter_path_block(
        contrib_tp,
        plan.flat_output_indices,
        n_particles=3,
        output_order=target_tp.m,
        agg=aggregation,
    )
    raw_tp_blocks_override = list(raw_tp_tp.blocks)
    raw_tp_blocks_override[target_tp.m] = _replace_path_slice(
        raw_tp_tp.blocks[target_tp.m], tp_path_idx, scattered_tp
    )
    logabs_tp_ref = _finish_logabs_from_raw_producers(
        features_tp, list(raw_lin_tp.blocks), raw_tp_blocks_override
    )
    torch.testing.assert_close(logabs_tp_ref.detach(), packet.output.logabs, atol=1e-12, rtol=1e-12)
    g_tp_ref = torch.autograd.grad(logabs_tp_ref.sum(), contrib_tp)[0].detach()  # [B, O, V]
    tp_score_from_ga = torch.einsum("bov,bkv->bok", g_tp_ref, a_ref)
    tp_score_oracle = _per_sample_score_oracle(model, batch, (tp_w,)).reshape(
        batch.batch_size, out_dim_tp, in_dim_tp
    )
    torch.testing.assert_close(tp_score_from_ga, tp_score_oracle, atol=1e-12, rtol=1e-12)
    g_tp_flat = g_tp_ref.permute(0, 2, 1).reshape(batch.batch_size * 6, out_dim_tp)
    G_tp_ref = (g_tp_flat.T @ g_tp_flat) / float(batch.batch_size)
    # Non-vacuous check + exact KFAC-expand normalization (kills / (B * V) mutant at structured_curvature.py:1062).
    assert float(G_tp_ref.abs().max().item()) > 1e-8
    torch.testing.assert_close(tp_blk.G, G_tp_ref, atol=1e-12, rtol=1e-12)

    # Verify LinearEquivariantMixing unary path A_ref, G_ref, and score identity for both:
    #   - index 0: (1->1) identity path (support_order=1, V=3, 1 completion per output)
    #   - index 1: (1->1) neighbor-replacement path (support_order=2, V=6, 2 completions per output)
    lin_blocks_by_idx: dict[int, KroneckerFactorBlock] = {}
    for target_lin_idx in (0, 1):
        target_lin = lin_prod.paths[target_lin_idx]
        lin_w = lin_prod.weights[target_lin_idx]
        lin_blk = next(
            b for b in packet.kronecker_blocks if b.parameter_ordinals == (id_to_ord[id(lin_w)],)
        )
        lin_blocks_by_idx[target_lin_idx] = lin_blk
        features_lin = model.embedding(batch, context=context)
        uplan = build_unary_index_plan(target_lin, 3)
        u_in = _gather_tuple_values(
            features_lin.blocks[target_lin.input_order], uplan.input_indices
        ).detach()
        v_lin = int(u_in.shape[2])
        assert lin_blk.active_locations == v_lin
        u_flat = u_in.permute(0, 2, 1).reshape(batch.batch_size * v_lin, -1)
        A_lin_ref = (u_flat.T @ u_flat) / float(batch.batch_size * v_lin)
        torch.testing.assert_close(lin_blk.A, A_lin_ref, atol=1e-12, rtol=1e-12)

        raw_lin_u = lin_prod.forward_pre_activation(features_lin)
        raw_tp_u = tp_prod.forward_pre_activation(features_lin)
        lin_order_paths = list(lin_prod.metadata.paths_for_output_order(target_lin.output_order))
        lin_path_idx = lin_order_paths.index(target_lin)
        out_dim_lin = int(lin_w.shape[0])
        in_dim_lin = int(lin_w.shape[1])
        contrib_lin = torch.einsum("oc,bcv->bov", lin_w, u_in)
        scattered_lin = _manual_scatter_path_block(
            contrib_lin,
            uplan.flat_output_indices,
            n_particles=3,
            output_order=target_lin.output_order,
            agg=aggregation,
        )
        raw_lin_blocks_override = list(raw_lin_u.blocks)
        raw_lin_blocks_override[target_lin.output_order] = _replace_path_slice(
            raw_lin_u.blocks[target_lin.output_order], lin_path_idx, scattered_lin
        )
        logabs_lin_ref = _finish_logabs_from_raw_producers(
            features_lin, raw_lin_blocks_override, list(raw_tp_u.blocks)
        )
        torch.testing.assert_close(
            logabs_lin_ref.detach(), packet.output.logabs, atol=1e-12, rtol=1e-12
        )
        g_lin_ref = torch.autograd.grad(logabs_lin_ref.sum(), contrib_lin)[0].detach()  # [B, O, V]
        lin_score_from_ga = torch.einsum("bov,bcv->boc", g_lin_ref, u_in)
        lin_score_oracle = _per_sample_score_oracle(model, batch, (lin_w,)).reshape(
            batch.batch_size, out_dim_lin, in_dim_lin
        )
        torch.testing.assert_close(lin_score_from_ga, lin_score_oracle, atol=1e-12, rtol=1e-12)
        g_lin_flat = g_lin_ref.permute(0, 2, 1).reshape(batch.batch_size * v_lin, out_dim_lin)
        G_lin_ref = (g_lin_flat.T @ g_lin_flat) / float(batch.batch_size)
        assert float(G_lin_ref.abs().max().item()) > 1e-8
        torch.testing.assert_close(lin_blk.G, G_lin_ref, atol=1e-12, rtol=1e-12)

    # For COMPLETION_MEAN vs SUM, compare the two runs directly on both binary (s=2 -> m=1)
    # and unary (support_order=2 -> output_order=1) paths to prove completion counts take effect.
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
        assert float((tp_blk.G - tp_blk_sum.G).abs().max().item()) > 1e-6
        lin_blk1_sum = next(
            b
            for b in pkt_sum.kronecker_blocks
            if b.parameter_ordinals == lin_blocks_by_idx[1].parameter_ordinals
        )
        assert float((lin_blocks_by_idx[1].G - lin_blk1_sum.G).abs().max().item()) > 1e-6


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

    # Post-init / post-restore optimizer mutation must be refused inside update()
    # before mutating EMA, counters, or parameters.
    batch_sgd = _make_batch(batch_size=4, n_electrons=2, spatial_dim=2, seed=621)
    e_sgd = torch.tensor([-1.0, 0.2, 0.5, 0.7], dtype=torch.float64)
    w_sgd = (e_sgd - e_sgd.mean()) / 4.0
    pkt_sgd = StructuredCurvatureForwardRequest(sample_weights=w_sgd).evaluate(model, batch_sgd)
    step_sgd = StructuredCurvatureUpdateInput(
        batch=batch_sgd,
        wavefunction=pkt_sgd.output,
        local_energy=e_sgd,
        step=0,
        parameter_binding=pkt_sgd.parameter_binding,
        solve_dtype=pkt_sgd.solve_dtype,
        kronecker_blocks=pkt_sgd.kronecker_blocks,
        dense_blocks=pkt_sgd.dense_blocks,
        energy_half_gradient=pkt_sgd.energy_half_gradient,
        n_finite_samples=4,
    )
    assert method.update(step_sgd).applied is True
    params_snap = [p.detach().clone() for p in model.parameters()]
    assert method._kronecker_ema is not None
    ema_snap = [e["A"].clone() for e in method._kronecker_ema]

    opt.param_groups[0]["lr"] = 0.5
    with pytest.raises(ValueError, match="disagrees"):
        method.update(replace(step_sgd, step=1))

    opt.param_groups[0]["lr"] = policy.learning_rate
    opt.param_groups[0]["momentum"] = 0.9
    with pytest.raises(ValueError, match="momentum"):
        method.update(replace(step_sgd, step=1))

    assert method.curvature_step == 1
    assert method.completed_updates == 1
    for p_after, p_before in zip(model.parameters(), params_snap, strict=True):
        assert torch.equal(p_after, p_before)
    for e_after, e_before in zip(method._kronecker_ema, ema_snap, strict=True):
        assert torch.equal(e_after["A"], e_before)


def test_kfac_route_receives_the_psiformer_clipped_energy(monkeypatch) -> None:
    """Native KFAC consumes the same clipped score residual as other score routes."""

    model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    method = NativeKFACUpdate(
        optimizer,
        model_parameters=_model_binding_for(model),
        policy=KFACPolicy(learning_rate=0.03),
    )
    batch = _make_batch(batch_size=4, n_electrons=2, spatial_dim=2, seed=733)
    raw_energy = torch.tensor([0.0, 1.0, 2.0, 100.0], dtype=torch.float64)
    monkeypatch.setattr(
        trainer_module,
        "local_energy",
        lambda *args, **kwargs: raw_energy.clone(),
    )
    trainer = VMCTrainer(
        max_steps=1,
        update_method=method,
        energy_clipping_mode="psiformer",
        energy_clipping_k=1.0,
    )
    state = TrainerState(model=model, optimizer=optimizer, trainer=trainer)
    producer = _TrainerStepProducer(
        trainer=trainer,
        model=model,
        hamiltonian_terms={},
        batch=batch,
        step=0,
        context=_StubContext(),
        state=state,
    )

    produced = producer.produce_structured_curvature_step(resolve_step_request(method))
    assert torch.equal(
        produced.update_input.local_energy,
        torch.tensor([0.0, 1.0, 2.0, 26.75], dtype=torch.float64),
    )

def test_structured_curvature_update_input_validates_block_dims_and_device() -> None:
    """StructuredCurvatureUpdateInput.validate() enforces block dims and uniform parameter device."""

    # 1. Transposed Kronecker dims matching weight.numel() must be rejected.
    model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=False, use_seeded=True)
    batch = _make_batch(batch_size=4, n_electrons=2, spatial_dim=2, seed=611)
    energies = torch.linspace(-1.0, 1.0, 4, dtype=torch.float64)
    weights = (energies - energies.mean()) / 4.0
    pkt = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)

    blk0 = pkt.kronecker_blocks[0]
    assert (blk0.out_dim, blk0.in_dim) == (3, 4)
    transposed_blk0 = KroneckerFactorBlock(
        parameter_ordinals=blk0.parameter_ordinals,
        has_bias=False,
        out_dim=4,
        in_dim=3,
        A=torch.eye(3, dtype=torch.float64),
        G=torch.eye(4, dtype=torch.float64),
        active_locations=blk0.active_locations,
    )
    with pytest.raises(ValueError, match="disagree with weight slot shape"):
        StructuredCurvatureUpdateInput(
            batch=batch,
            wavefunction=pkt.output,
            local_energy=energies,
            step=0,
            parameter_binding=pkt.parameter_binding,
            solve_dtype=pkt.solve_dtype,
            kronecker_blocks=(transposed_blk0, *pkt.kronecker_blocks[1:]),
            dense_blocks=pkt.dense_blocks,
            energy_half_gradient=pkt.energy_half_gradient,
            n_finite_samples=4,
        )

    # 2. Bias slot shape mismatch on a bias-augmented Kronecker block must be rejected.
    bias_model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=True, use_seeded=True)
    bias_pkt = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(bias_model, batch)
    b0, b1 = bias_pkt.kronecker_blocks
    swapped_b0 = KroneckerFactorBlock(
        parameter_ordinals=(b0.parameter_ordinals[0], b1.parameter_ordinals[1]),
        has_bias=True,
        out_dim=b0.out_dim,
        in_dim=b0.in_dim,
        A=b0.A,
        G=b0.G,
        active_locations=b0.active_locations,
    )
    swapped_b1 = KroneckerFactorBlock(
        parameter_ordinals=(b1.parameter_ordinals[0], b0.parameter_ordinals[1]),
        has_bias=True,
        out_dim=b1.out_dim,
        in_dim=b1.in_dim,
        A=b1.A,
        G=b1.G,
        active_locations=b1.active_locations,
    )
    with pytest.raises(ValueError, match="bias slot shape"):
        StructuredCurvatureUpdateInput(
            batch=batch,
            wavefunction=bias_pkt.output,
            local_energy=energies,
            step=0,
            parameter_binding=bias_pkt.parameter_binding,
            solve_dtype=bias_pkt.solve_dtype,
            kronecker_blocks=(swapped_b0, swapped_b1),
            dense_blocks=(),
            energy_half_gradient=bias_pkt.energy_half_gradient,
            n_finite_samples=4,
        )

    # 3. DenseFactorBlock dim mismatch against covered slot numel must be rejected.
    frozen_model = _LinearMLPWavefunction(
        in_features=4, hidden_features=3, bias=True, freeze_first_weight=True
    )
    frozen_pkt = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(
        frozen_model, batch
    )
    bad_dense = DenseFactorBlock(
        parameter_ordinals=frozen_pkt.dense_blocks[0].parameter_ordinals,
        dim=frozen_pkt.dense_blocks[0].dim + 1,
        F=torch.eye(frozen_pkt.dense_blocks[0].dim + 1, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="disagrees with covered numel"):
        StructuredCurvatureUpdateInput(
            batch=batch,
            wavefunction=frozen_pkt.output,
            local_energy=energies,
            step=0,
            parameter_binding=frozen_pkt.parameter_binding,
            solve_dtype=frozen_pkt.solve_dtype,
            kronecker_blocks=frozen_pkt.kronecker_blocks,
            dense_blocks=(bad_dense,),
            energy_half_gradient=frozen_pkt.energy_half_gradient,
            n_finite_samples=4,
        )

    # 4. Mixed parameter devices in parameter_binding must be rejected.
    p_cpu = nn.Parameter(torch.ones((3, 4), dtype=torch.float64))
    p_meta = nn.Parameter(torch.empty((3,), device="meta", dtype=torch.float64))
    mixed_binding = ParameterBinding(
        layout=ParameterLayout(
            slots=(
                ParameterSlot(ordinal=0, shape=(3, 4), numel=12, dtype=torch.float64),
                ParameterSlot(ordinal=1, shape=(3,), numel=3, dtype=torch.float64),
            )
        ),
        parameters=(p_cpu, p_meta),
    )
    with pytest.raises(ValueError, match="share one device"):
        StructuredCurvatureUpdateInput(
            batch=batch,
            wavefunction=pkt.output,
            local_energy=energies,
            step=0,
            parameter_binding=mixed_binding,
            solve_dtype=torch.float64,
            kronecker_blocks=(),
            dense_blocks=(),
            energy_half_gradient=(),
            n_finite_samples=4,
            curvature_available=False,
        )


def test_curvature_factor_update_interval_and_float32_solve_dtype() -> None:
    """Exercise curvature_factor_update_interval > 1 (factors_updated=False) and solve_dtype=float32."""

    # 1. curvature_factor_update_interval=2: odd steps skip EMA updates cleanly.
    model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=True, use_seeded=True)
    policy = KFACPolicy(
        learning_rate=0.05,
        relative_damping=1e-2,
        factor_ema=0.9,
        curvature_factor_update_interval=2,
        inverse_refresh_interval=4,
        predicted_kl_cap=1e-3,
    )
    opt = torch.optim.SGD(model.parameters(), lr=policy.learning_rate)
    method = NativeKFACUpdate(opt, model_parameters=_model_binding_for(model), policy=policy)

    alpha = policy.factor_ema
    beta = 1.0 - alpha
    oracle_ema_A: list[torch.Tensor] = []
    oracle_ema_G: list[torch.Tensor] = []
    last_step_ema_A: list[torch.Tensor] = []
    last_step_ema_G: list[torch.Tensor] = []

    for step in range(5):
        batch = _make_batch(batch_size=4, n_electrons=2, spatial_dim=2, seed=650 + step)
        energies = torch.linspace(-1.5, 1.5, 4, dtype=torch.float64)
        weights = (energies - energies.mean()) / 4.0
        pkt = StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)
        step_in = StructuredCurvatureUpdateInput(
            batch=batch,
            wavefunction=pkt.output,
            local_energy=energies,
            step=step,
            parameter_binding=pkt.parameter_binding,
            solve_dtype=pkt.solve_dtype,
            kronecker_blocks=pkt.kronecker_blocks,
            dense_blocks=pkt.dense_blocks,
            energy_half_gradient=pkt.energy_half_gradient,
            n_finite_samples=4,
        )
        res = method.update(step_in)
        assert res.applied is True
        tel = method.last_telemetry
        assert isinstance(tel, KFACTelemetry)
        expected_factors_updated = step == 0 or (step % 2 == 0)
        expected_refreshed = step % 4 == 0
        assert tel.factors_updated is expected_factors_updated
        assert tel.inverse_refreshed is expected_refreshed
        assert method._kronecker_ema is not None

        if step == 0:
            oracle_ema_A = [b.A.clone() for b in pkt.kronecker_blocks]
            oracle_ema_G = [b.G.clone() for b in pkt.kronecker_blocks]
        elif expected_factors_updated:
            oracle_ema_A = [
                alpha * old + beta * b.A
                for old, b in zip(oracle_ema_A, pkt.kronecker_blocks, strict=True)
            ]
            oracle_ema_G = [
                alpha * old + beta * b.G
                for old, b in zip(oracle_ema_G, pkt.kronecker_blocks, strict=True)
            ]
        for idx, (actual, exp_A, exp_G) in enumerate(
            zip(method._kronecker_ema, oracle_ema_A, oracle_ema_G, strict=True)
        ):
            torch.testing.assert_close(actual["A"], exp_A, atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(actual["G"], exp_G, atol=1e-13, rtol=1e-13)
            if not expected_factors_updated:
                assert torch.equal(actual["A"], last_step_ema_A[idx])
                assert torch.equal(actual["G"], last_step_ema_G[idx])

        last_step_ema_A = [e["A"].clone() for e in method._kronecker_ema]
        last_step_ema_G = [e["G"].clone() for e in method._kronecker_ema]
        if expected_refreshed:
            assert math.isclose(
                tel.vg_sum,
                2.0 * tel.predicted_kl + tel.damped_quadratic_term,
                rel_tol=1e-12,
                abs_tol=1e-13,
            )

    # 2. solve_dtype=torch.float32 end-to-end capture, update, and state round-trip.
    f32_model = _LinearMLPWavefunction(
        in_features=4, hidden_features=3, bias=True, use_seeded=True, dtype=torch.float32
    )
    f32_policy = KFACPolicy(
        learning_rate=0.05,
        relative_damping=1e-2,
        factor_ema=0.95,
        curvature_factor_update_interval=1,
        inverse_refresh_interval=2,
        predicted_kl_cap=1e-3,
        solve_dtype=torch.float32,
    )
    f32_opt = torch.optim.SGD(f32_model.parameters(), lr=f32_policy.learning_rate)
    f32_method = NativeKFACUpdate(
        f32_opt, model_parameters=_model_binding_for(f32_model), policy=f32_policy
    )
    f32_batch = _make_batch(
        batch_size=4, n_electrons=2, spatial_dim=2, seed=670, dtype=torch.float32
    )
    f32_e = torch.linspace(-1.2, 1.2, 4, dtype=torch.float32)
    f32_w = (f32_e - f32_e.mean()) / 4.0
    f32_pkt = StructuredCurvatureForwardRequest(
        sample_weights=f32_w, solve_dtype=torch.float32
    ).evaluate(f32_model, f32_batch)
    assert all(
        b.A.dtype == torch.float32 and b.G.dtype == torch.float32 for b in f32_pkt.kronecker_blocks
    )
    assert all(g.dtype == torch.float32 for g in f32_pkt.energy_half_gradient)
    f32_step = StructuredCurvatureUpdateInput(
        batch=f32_batch,
        wavefunction=f32_pkt.output,
        local_energy=f32_e,
        step=0,
        parameter_binding=f32_pkt.parameter_binding,
        solve_dtype=torch.float32,
        kronecker_blocks=f32_pkt.kronecker_blocks,
        dense_blocks=f32_pkt.dense_blocks,
        energy_half_gradient=f32_pkt.energy_half_gradient,
        n_finite_samples=4,
    )
    assert f32_method.update(f32_step).applied is True
    assert f32_method.last_telemetry is not None
    assert f32_method.last_telemetry.solve_dtype == "torch.float32"
    assert math.isclose(
        f32_method.last_telemetry.vg_sum,
        2.0 * f32_method.last_telemetry.predicted_kl + f32_method.last_telemetry.damped_quadratic_term,
        rel_tol=1e-5,
        abs_tol=1e-6,
    )
    assert f32_method._kronecker_ema is not None
    assert all(
        e["A"].dtype == torch.float32 and e["G"].dtype == torch.float32
        for e in f32_method._kronecker_ema
    )
    f32_meta = f32_method.method_state_dict()
    f32_tensors = f32_method.method_tensor_state_dict()
    assert f32_tensors is not None and f32_tensors["solve_dtype"] == "torch.float32"
    f32_clone_model = _LinearMLPWavefunction(
        in_features=4, hidden_features=3, bias=True, use_seeded=True, dtype=torch.float32
    )
    f32_clone_opt = torch.optim.SGD(f32_clone_model.parameters(), lr=f32_policy.learning_rate)
    f32_clone_method = NativeKFACUpdate(
        f32_clone_opt, model_parameters=_model_binding_for(f32_clone_model), policy=f32_policy
    )
    f32_clone_method.load_method_state_and_tensors(f32_meta, f32_tensors)
    assert f32_clone_method._kronecker_ema is not None
    for cur, ref in zip(f32_clone_method._kronecker_ema, f32_method._kronecker_ema, strict=True):
        assert torch.equal(cur["A"], ref["A"])
        assert torch.equal(cur["G"], ref["G"])


def test_mixing_kfac_capture_detaches_flattened_input_before_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute_binary and execute_unary pass detached flattened_input (no transient MulBackward0)."""

    import tpen.nn.structured_curvature as sc_module

    model = _build_rich_tpen_model(
        max_order=2,
        max_virtual_order=2,
        channels=2,
        use_composite=True,
        trainable_readout=False,
        trainable_factors=False,
    )
    batch = _make_batch(batch_size=4, n_electrons=3, spatial_dim=3, seed=681)
    weights = torch.tensor([-0.2, 0.3, -0.1, 0.0], dtype=torch.float64) / 4.0

    recorded: list[tuple[str, bool, Any]] = []
    orig_record = sc_module._StructuredCurvatureCaptureSession.record_mixing_kronecker

    def _spy(self, *, weight, flattened_input, contribution, label):
        recorded.append((label, bool(flattened_input.requires_grad), flattened_input.grad_fn))
        return orig_record(
            self,
            weight=weight,
            flattened_input=flattened_input,
            contribution=contribution,
            label=label,
        )

    monkeypatch.setattr(
        sc_module._StructuredCurvatureCaptureSession, "record_mixing_kronecker", _spy
    )
    StructuredCurvatureForwardRequest(sample_weights=weights).evaluate(model, batch)

    assert any(label.startswith("execute_binary") for label, _, _ in recorded)
    assert any(label.startswith("execute_unary") for label, _, _ in recorded)
    for label, req_grad, grad_fn in recorded:
        assert req_grad is False, f"{label} passed requires_grad=True flattened_input"
        assert grad_fn is None, f"{label} allocated transient autograd node {grad_fn}"


# ---------------------------------------------------------------------------
# 4. Bit-exact mid-refresh binary method_state.pt checkpoint save/restore
# ---------------------------------------------------------------------------


def _checkpoint_context(tmp_path: Path, method_tag: str) -> Any:
    from types import SimpleNamespace

    cfg = OmegaConf.create(
        {
            "model": {"tag": "tiny_kfac"},
            "hamiltonian_terms": {"tag": "tiny_ham"},
            "optimizer": {"tag": method_tag},
            "trainer": {"tag": method_tag},
            "sampler": {"tag": "tiny_seq"},
        }
    )
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

    # Verify manifest, JSON compactness, and publication receipt payload accounting (B1).
    manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    trainer_json = json.loads((ckpt_dir / "trainer.json").read_text(encoding="utf-8"))
    assert manifest["files"]["method_state"] == "method_state.pt"
    assert "method_state_sha256" in manifest["hashes"]
    assert (ckpt_dir / "method_state.pt").is_file()
    assert trainer_json["update_method"]["version"] == KFAC_STATE_VERSION
    assert trainer_json["update_method"]["convention"] == KFAC_CONVENTION_VERSION
    assert trainer_json["update_method"]["curvature_step"] == 3
    assert "kronecker_ema" not in trainer_json["update_method"]

    assert "method_state" in PAYLOAD_COMPONENT_NAMES
    receipt_path = publication_receipt_path(ckpt_root)
    receipt_rows = [
        json.loads(line)
        for line in receipt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(receipt_rows) == 1
    receipt_row = receipt_rows[0]
    receipt_file_sizes = {entry["component"]: entry["size_bytes"] for entry in receipt_row["files"]}
    method_state_bytes = (ckpt_dir / "method_state.pt").stat().st_size
    assert method_state_bytes > 0
    assert receipt_file_sizes["method_state"] == method_state_bytes
    expected_payload_bytes = sum(
        (ckpt_dir / manifest["files"][comp]).stat().st_size
        for comp in ("model", "optimizer", "trainer", "method_state", "sampler", "rng")
    )
    expected_metadata_bytes = (
        (ckpt_dir / "resolved_config.yaml").stat().st_size
        + (ckpt_dir / "manifest.json").stat().st_size
        + (ckpt_dir / "COMPLETE").stat().st_size
    )
    receipt_summary = receipt_row["summary"]
    assert receipt_summary["payload_bytes"] == expected_payload_bytes
    assert receipt_summary["metadata_bytes"] == expected_metadata_bytes
    assert receipt_summary["total_bytes"] == (
        receipt_summary["payload_bytes"] + receipt_summary["metadata_bytes"]
    )
    assert receipt_summary["file_count"] == len(receipt_row["files"])

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

    # Helpers for strict atomic-refusal checks against live step-1 KFAC state (distinct from step-3 ckpt).
    import shutil

    def _build_live_step1_stack():
        m, o, meth, tr, samp = _build_stack(max_steps=1)
        tr.fit(
            model=m,
            sampler=samp,
            hamiltonian_terms={},
            optimizer=o,
            context=_StubContext(),
            emit=lambda **_: None,
        )
        assert tr.next_iteration == 1
        assert tr.completed_updates == 1
        assert meth.curvature_step == 1
        assert meth._kronecker_ema is not None
        assert meth._kronecker_eigs is not None
        assert meth._dense_ema is not None
        assert meth._dense_eigs is not None
        return m, o, meth, tr, samp

    def _snapshot_kfac_stack(m, o, meth, tr) -> dict[str, Any]:
        return {
            "params": [p.detach().clone() for p in m.parameters()],
            "opt_groups": [
                {k: v for k, v in grp.items() if k != "params"} for grp in o.param_groups
            ],
            "opt_state_keys": list(o.state_dict()["state"].keys()),
            "next_iteration": tr.next_iteration,
            "completed_updates": tr.completed_updates,
            "method_completed_updates": meth.completed_updates,
            "curvature_step": meth.curvature_step,
            "block_signature": meth._block_signature,
            "kron_ema": [
                {"A": e["A"].clone(), "G": e["G"].clone()} for e in meth._kronecker_ema
            ],
            "kron_eigs": [
                {
                    "d_A": e["d_A"].clone(),
                    "Q_A": e["Q_A"].clone(),
                    "d_G": e["d_G"].clone(),
                    "Q_G": e["Q_G"].clone(),
                    "damping": e["damping"],
                }
                for e in meth._kronecker_eigs
            ],
            "dense_ema": [{"F": e["F"].clone()} for e in meth._dense_ema],
            "dense_eigs": [
                {
                    "d_F": e["d_F"].clone(),
                    "Q_F": e["Q_F"].clone(),
                    "damping": e["damping"],
                }
                for e in meth._dense_eigs
            ],
        }

    def _assert_kfac_stack_unchanged(m, o, meth, tr, snap: dict[str, Any]) -> None:
        for p_after, p_before in zip(m.parameters(), snap["params"], strict=True):
            assert torch.equal(p_after, p_before)
        assert [
            {k: v for k, v in grp.items() if k != "params"} for grp in o.param_groups
        ] == snap["opt_groups"]
        assert list(o.state_dict()["state"].keys()) == snap["opt_state_keys"]
        assert tr.next_iteration == snap["next_iteration"]
        assert tr.completed_updates == snap["completed_updates"]
        assert meth.completed_updates == snap["method_completed_updates"]
        assert meth.curvature_step == snap["curvature_step"]
        assert meth._block_signature == snap["block_signature"]
        assert meth._kronecker_ema is not None
        assert meth._kronecker_eigs is not None
        assert meth._dense_ema is not None
        assert meth._dense_eigs is not None
        for cur, prev in zip(meth._kronecker_ema, snap["kron_ema"], strict=True):
            assert torch.equal(cur["A"], prev["A"])
            assert torch.equal(cur["G"], prev["G"])
        for cur, prev in zip(meth._kronecker_eigs, snap["kron_eigs"], strict=True):
            assert torch.equal(cur["d_A"], prev["d_A"])
            assert torch.equal(cur["Q_A"], prev["Q_A"])
            assert torch.equal(cur["d_G"], prev["d_G"])
            assert torch.equal(cur["Q_G"], prev["Q_G"])
            assert cur["damping"] == prev["damping"]
        for cur, prev in zip(meth._dense_ema, snap["dense_ema"], strict=True):
            assert torch.equal(cur["F"], prev["F"])
        for cur, prev in zip(meth._dense_eigs, snap["dense_eigs"], strict=True):
            assert torch.equal(cur["d_F"], prev["d_F"])
            assert torch.equal(cur["Q_F"], prev["Q_F"])
            assert cur["damping"] == prev["damping"]

    # 3. Tampered method_state.pt refuses atomically before mutating live state.
    tampered_dir = tmp_path / "tampered_step"
    shutil.copytree(ckpt_dir, tampered_dir)
    raw_bytes = (tampered_dir / "method_state.pt").read_bytes()
    (tampered_dir / "method_state.pt").write_bytes(raw_bytes[:-1] + bytes([raw_bytes[-1] ^ 0xFF]))
    t_model, t_opt, t_method, t_trainer, t_sampler = _build_live_step1_stack()
    t_snap = _snapshot_kfac_stack(t_model, t_opt, t_method, t_trainer)
    with pytest.raises(ValueError, match="digest mismatch"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(tampered_dir)},
            model=t_model,
            optimizer=t_opt,
            trainer=t_trainer,
            sampler=t_sampler,
            context=ckpt_ctx,
        )
    _assert_kfac_stack_unchanged(t_model, t_opt, t_method, t_trainer, t_snap)

    # 4. Missing method_state entry when trainer.json records curvature_step > 0 refuses atomically.
    missing_dir = tmp_path / "missing_sidecar"
    shutil.copytree(ckpt_dir, missing_dir)
    m_missing = json.loads((missing_dir / "manifest.json").read_text(encoding="utf-8"))
    del m_missing["files"]["method_state"]
    del m_missing["hashes"]["method_state_sha256"]
    (missing_dir / "manifest.json").write_text(json.dumps(m_missing), encoding="utf-8")
    m_model, m_opt, m_method, m_trainer, m_sampler = _build_live_step1_stack()
    # Give target optimizer a distinct lr (vs ckpt's 0.04) so any _load_optimizer mutation is caught.
    m_opt.param_groups[0]["lr"] = 0.88
    m_snap = _snapshot_kfac_stack(m_model, m_opt, m_method, m_trainer)
    with pytest.raises(ValueError, match="method_state.pt"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(missing_dir)},
            model=m_model,
            optimizer=m_opt,
            trainer=m_trainer,
            sampler=m_sampler,
            context=ckpt_ctx,
        )
    _assert_kfac_stack_unchanged(m_model, m_opt, m_method, m_trainer, m_snap)

    # Direct VMCTrainer.load_checkpoint_state refusal also leaves live trainer + KFAC untouched.
    with pytest.raises(ValueError, match="method_state.pt"):
        m_trainer.load_checkpoint_state(trainer_json, method_tensor_state=None)
    _assert_kfac_stack_unchanged(m_model, m_opt, m_method, m_trainer, m_snap)

    # 5. Valid-digest semantic drift in method_state.pt (mismatched curvature_step, NaN entry,
    #    and negative eigenvalues d_A / d_G / d_F) refuses atomically with zero live mutation.
    def _write_drifted_sidecar(tag: str, mutate_fn) -> Path:
        drift_dir = tmp_path / f"drift_{tag}"
        shutil.copytree(ckpt_dir, drift_dir)
        payload = torch.load(drift_dir / "method_state.pt", map_location="cpu", weights_only=False)
        mutate_fn(payload)
        torch.save(payload, drift_dir / "method_state.pt")
        drift_manifest = json.loads((drift_dir / "manifest.json").read_text(encoding="utf-8"))
        drift_manifest["hashes"]["method_state_sha256"] = file_sha256(drift_dir / "method_state.pt")
        (drift_dir / "manifest.json").write_text(json.dumps(drift_manifest), encoding="utf-8")
        return drift_dir

    drift_cases = [
        ("cstep", lambda p: p.__setitem__("curvature_step", 999), "curvature_step"),
        ("nan_A", lambda p: p["kronecker_ema"][0]["A"].__setitem__((0, 0), float("nan")), "non-finite"),
        ("neg_dA", lambda p: p["kronecker_eigs"][0]["d_A"].__setitem__(0, -1e-6), "non-negative"),
        ("neg_dG", lambda p: p["kronecker_eigs"][0]["d_G"].__setitem__(0, -1e-6), "non-negative"),
        ("neg_dF", lambda p: p["dense_eigs"][0]["d_F"].__setitem__(0, -1e-6), "non-negative"),
    ]
    d_model, d_opt, d_method, d_trainer, d_sampler = _build_live_step1_stack()
    d_opt.param_groups[0]["lr"] = 0.88
    d_snap = _snapshot_kfac_stack(d_model, d_opt, d_method, d_trainer)
    for tag, mutate_fn, expected_match in drift_cases:
        drift_dir = _write_drifted_sidecar(tag, mutate_fn)
        with pytest.raises(ValueError, match=expected_match):
            restore_checkpoint(
                load={"mode": "train_resume", "path": str(drift_dir)},
                model=d_model,
                optimizer=d_opt,
                trainer=d_trainer,
                sampler=d_sampler,
                context=ckpt_ctx,
            )
        _assert_kfac_stack_unchanged(d_model, d_opt, d_method, d_trainer, d_snap)

    # 6. Unexpected valid-digest method_state.pt on KFAC curvature_step == 0 refuses atomically.
    s0_model, s0_opt, _s0_method, s0_trainer, s0_sampler = _build_stack(max_steps=1)
    s0_trainer.resolve_update_state(model=s0_model, optimizer=s0_opt)
    s0_ckpt_dir = save_checkpoint(
        output_dir=tmp_path / "kfac_step0_ckpts",
        next_iteration=0,
        completed_updates=0,
        model=s0_model,
        optimizer=s0_opt,
        trainer=s0_trainer,
        sampler=s0_sampler,
        context=ckpt_ctx,
    )
    s0_manifest = json.loads((s0_ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "method_state" not in s0_manifest["files"]
    assert not (s0_ckpt_dir / "method_state.pt").exists()
    shutil.copy2(ckpt_dir / "method_state.pt", s0_ckpt_dir / "method_state.pt")
    s0_manifest["files"]["method_state"] = "method_state.pt"
    s0_manifest["hashes"]["method_state_sha256"] = manifest["hashes"]["method_state_sha256"]
    (s0_ckpt_dir / "manifest.json").write_text(json.dumps(s0_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected binary method_state.pt"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(s0_ckpt_dir)},
            model=d_model,
            optimizer=d_opt,
            trainer=d_trainer,
            sampler=d_sampler,
            context=ckpt_ctx,
        )
    _assert_kfac_stack_unchanged(d_model, d_opt, d_method, d_trainer, d_snap)

    # 7. Backward compatibility & atomic refusal on SR checkpoint carrying unexpected method_state.pt.
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

    # Inject an unexpected method_state.pt into SR checkpoint and mutate live SR state so
    # any partial restore before refusal is immediately observable.
    shutil.copy2(ckpt_dir / "method_state.pt", sr_ckpt_dir / "method_state.pt")
    sr_manifest["files"]["method_state"] = "method_state.pt"
    sr_manifest["hashes"]["method_state_sha256"] = manifest["hashes"]["method_state_sha256"]
    (sr_ckpt_dir / "manifest.json").write_text(json.dumps(sr_manifest), encoding="utf-8")
    with torch.no_grad():
        for p in sr_model.parameters():
            p.add_(0.75)
    sr_opt.param_groups[0]["lr"] = 0.77
    sr_trainer.next_iteration = 7
    sr_trainer.completed_updates = 5
    sr_method.completed_updates = 5
    sr_params_before = [p.detach().clone() for p in sr_model.parameters()]
    with pytest.raises(ValueError, match="does not accept binary method tensor state"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(sr_ckpt_dir)},
            model=sr_model,
            optimizer=sr_opt,
            trainer=sr_trainer,
            sampler=_SequenceSampler(batches),
            context=_checkpoint_context(tmp_path, "sr"),
        )
    for p_after, p_before in zip(sr_model.parameters(), sr_params_before, strict=True):
        assert torch.equal(p_after, p_before)
    assert sr_opt.param_groups[0]["lr"] == 0.77
    assert sr_trainer.next_iteration == 7
    assert sr_trainer.completed_updates == 5
    assert sr_method.completed_updates == 5

    # 8. Backward compatibility & atomic refusal on Adam checkpoint carrying unexpected method_state.pt
    #    (proves populated Adam optimizer state tensors are also completely unmutated).
    adam_ctx = _checkpoint_context(tmp_path, "adam")
    adam_save_model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=True)
    adam_save_opt = torch.optim.Adam(adam_save_model.parameters(), lr=0.01)
    sum(p.square().sum() for p in adam_save_model.parameters()).backward()
    adam_save_opt.step()
    adam_save_opt.zero_grad()
    adam_save_trainer = VMCTrainer(max_steps=1)
    adam_save_trainer.next_iteration = 1
    adam_save_trainer.completed_updates = 1
    adam_save_trainer.resolve_update_state(model=adam_save_model, optimizer=adam_save_opt)
    adam_ckpt_dir = save_checkpoint(
        output_dir=tmp_path / "adam_ckpts",
        next_iteration=1,
        completed_updates=1,
        model=adam_save_model,
        optimizer=adam_save_opt,
        trainer=adam_save_trainer,
        sampler=_SequenceSampler(batches),
        context=adam_ctx,
    )
    adam_manifest = json.loads((adam_ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "method_state" not in adam_manifest["files"]
    assert not (adam_ckpt_dir / "method_state.pt").exists()
    shutil.copy2(ckpt_dir / "method_state.pt", adam_ckpt_dir / "method_state.pt")
    adam_manifest["files"]["method_state"] = "method_state.pt"
    adam_manifest["hashes"]["method_state_sha256"] = manifest["hashes"]["method_state_sha256"]
    (adam_ckpt_dir / "manifest.json").write_text(json.dumps(adam_manifest), encoding="utf-8")

    adam_live_model = _LinearMLPWavefunction(in_features=4, hidden_features=3, bias=True)
    with torch.no_grad():
        for p in adam_live_model.parameters():
            p.add_(1.5)
    adam_live_opt = torch.optim.Adam(adam_live_model.parameters(), lr=0.01)
    sum((2.0 * p).square().sum() for p in adam_live_model.parameters()).backward()
    adam_live_opt.step()
    adam_live_opt.zero_grad()
    adam_live_trainer = VMCTrainer(max_steps=10)
    adam_live_trainer.next_iteration = 9
    adam_live_trainer.completed_updates = 8
    adam_live_trainer.resolve_update_state(model=adam_live_model, optimizer=adam_live_opt)
    adam_params_before = [p.detach().clone() for p in adam_live_model.parameters()]
    adam_opt_state_before = {
        idx: {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in entry.items()
        }
        for idx, entry in adam_live_opt.state_dict()["state"].items()
    }
    assert adam_opt_state_before  # non-vacuous populated Adam moments
    with pytest.raises(ValueError, match="does not accept binary method tensor state"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(adam_ckpt_dir)},
            model=adam_live_model,
            optimizer=adam_live_opt,
            trainer=adam_live_trainer,
            sampler=_SequenceSampler(batches),
            context=adam_ctx,
        )
    for p_after, p_before in zip(adam_live_model.parameters(), adam_params_before, strict=True):
        assert torch.equal(p_after, p_before)
    assert adam_live_trainer.next_iteration == 9
    assert adam_live_trainer.completed_updates == 8
    adam_opt_state_after = adam_live_opt.state_dict()["state"]
    assert set(adam_opt_state_after.keys()) == set(adam_opt_state_before.keys())
    for idx, entry_before in adam_opt_state_before.items():
        for k, v_before in entry_before.items():
            v_after = adam_opt_state_after[idx][k]
            if isinstance(v_before, torch.Tensor):
                assert torch.equal(v_after, v_before)
            else:
                assert v_after == v_before

    # 9. Trainer lacking load_method_tensor_state_dict() given a KFAC checkpoint carrying
    #    method_state.pt refuses atomically before mutating model, optimizer, or trainer.
    class _PlainTrainer:
        def __init__(self) -> None:
            self.loaded: dict[str, Any] | None = None

        def load_state_dict(self, state: dict[str, Any]) -> None:
            self.loaded = dict(state)

    p_model, p_opt, _p_method, _p_tr, p_sampler = _build_stack(max_steps=6)
    with torch.no_grad():
        for p in p_model.parameters():
            p.add_(0.25)
    p_opt.param_groups[0]["lr"] = 0.99
    p_params_before = [p.detach().clone() for p in p_model.parameters()]
    plain_trainer = _PlainTrainer()
    with pytest.raises(ValueError, match="does not implement load_method_tensor_state_dict"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(ckpt_dir)},
            model=p_model,
            optimizer=p_opt,
            trainer=plain_trainer,
            sampler=p_sampler,
            context=ckpt_ctx,
        )
    assert plain_trainer.loaded is None
    assert p_opt.param_groups[0]["lr"] == 0.99
    for p_after, p_before in zip(p_model.parameters(), p_params_before, strict=True):
        assert torch.equal(p_after, p_before)


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
