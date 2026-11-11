"""Full-model gates for the three static interaction producer presets.

These tests intentionally exercise the public model state/checkpoint surface;
they do not reach into checkpoint implementation modules.
"""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tpen.data.batch import ElectronBatch
from tpen.checkpoint import restore_checkpoint, save_checkpoint
from tpen.data.paths import (
    LinearPathMetadata,
    NormalizedChannels,
    NormalizedOrders,
    PathMetadata,
    compose_path_layout,
)
from tpen.nn import (
    CompositeMixing,
    Embedding,
    EquivariantMixing,
    InteractionMode,
    LinearEquivariantMixing,
    PathAggregation,
    ResidualUpdater,
    TPENLayer,
    TPENWaveFunction,
)
from tpen.nn.readout import PfaffianReadout


def _build_model(mode: InteractionMode) -> TPENWaveFunction:
    input_orders = NormalizedOrders((1, 2))
    channels = NormalizedChannels(((1, 1), (2, 1)))
    linear_metadata = LinearPathMetadata.generate(max_order=2)
    tensor_metadata = PathMetadata.generate(max_order=2, max_virtual_order=2, output_embedding="canonical")
    layout = compose_path_layout(
        linear=linear_metadata if mode is not InteractionMode.TENSOR_PRODUCT else None,
        tensor_product=tensor_metadata if mode is not InteractionMode.LINEAR else None,
        input_orders=input_orders,
        output_orders=input_orders,
        input_channels=channels,
        output_channels=channels,
    )
    producers = []
    if mode is not InteractionMode.TENSOR_PRODUCT:
        producers.append(LinearEquivariantMixing(max_order=2, channels=1, metadata=linear_metadata))
    if mode is not InteractionMode.LINEAR:
        producers.append(EquivariantMixing(max_order=2, channels=1, paths=tensor_metadata, activation=None))
    mixing = CompositeMixing(layout=layout, producers=tuple(producers), activation=torch.nn.SiLU())
    aggregation = PathAggregation(max_order=2, channels=1, layout=layout, activation=torch.nn.SiLU())
    layer = TPENLayer(mixing=mixing, path_aggregation=aggregation, update=ResidualUpdater(), layout=layout)
    return TPENWaveFunction(
        embedding=Embedding(max_order=2, spatial_dim=3, out_channels=1, hidden_channels=4, num_hidden_layers=1),
        layers=(layer,),
        readout=PfaffianReadout(channels=1),
        layout=layout,
    ).to(dtype=torch.float64)


def _batch(n_electrons: int = 3) -> ElectronBatch:
    generator = torch.Generator().manual_seed(31)
    return ElectronBatch(
        positions=torch.randn(2, n_electrons, 3, generator=generator, dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0, 1.0]] * 2, dtype=torch.float64)[:, :n_electrons],
    )


@pytest.mark.parametrize("mode", tuple(InteractionMode))
def test_full_model_forward_backward_optimizer_and_strict_roundtrip(mode: InteractionMode) -> None:
    model = _build_model(mode)
    batch = _batch()
    before_keys = tuple(model.state_dict())
    output = model(batch)
    output.validate(batch_size=batch.batch_size)
    loss = output.logabs.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    optimizer.step()
    assert tuple(model.state_dict()) == before_keys

    restored = _build_model(mode)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored_output = restored(batch)
    torch.testing.assert_close(restored_output.logabs, model(batch).logabs)


@pytest.mark.parametrize("mode", tuple(InteractionMode))
def test_runtime_particle_count_does_not_change_eager_model_layout(mode: InteractionMode) -> None:
    model = _build_model(mode)
    keys_before = tuple(model.state_dict())
    model(_batch(n_electrons=3))
    model(_batch(n_electrons=2))
    assert tuple(model.state_dict()) == keys_before


@pytest.mark.parametrize("mode", tuple(InteractionMode))
def test_public_checkpoint_api_round_trips_model_only(mode: InteractionMode, tmp_path) -> None:
    """Save/restore is tested through the public API as a black box."""

    model = _build_model(mode)
    # The public checkpoint writer passes cfg to OmegaConf.select; keep this
    # fixture unambiguously on the same DictConfig boundary as production.
    config = OmegaConf.create({"model": {"interaction_mode": mode.value}, "hamiltonian_terms": {}})
    context = SimpleNamespace(
        cfg=config,
        run_dir=tmp_path,
        metadata=SimpleNamespace(device="cpu", dtype="float64"),
    )
    checkpoint = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=1,
        completed_updates=1,
        model=model,
        context=context,
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )
    restored = _build_model(mode)
    report = restore_checkpoint(
        load={"mode": "model_only", "path": str(checkpoint)},
        model=restored,
        context=context,
        strict=True,
    )
    assert report.loaded_model
    torch.testing.assert_close(restored.state_dict()["_tpen_layout_fingerprint"], model.state_dict()["_tpen_layout_fingerprint"])


@pytest.mark.parametrize("mode", (InteractionMode.LINEAR, InteractionMode.HYBRID))
def test_legacy_tp_state_is_rejected_by_non_tp_models(mode: InteractionMode) -> None:
    """Legacy omission is not a migration path into changed layouts."""

    legacy_source = _build_model(InteractionMode.TENSOR_PRODUCT).state_dict()
    legacy_source.pop("_tpen_layout_fingerprint")
    target = _build_model(mode)
    before = OrderedDict((key, value.detach().clone()) for key, value in target.state_dict().items())
    with pytest.raises(ValueError, match="no layout fingerprint"):
        target.load_state_dict(legacy_source, strict=True)
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0.0, atol=0.0)


def test_tp_only_model_round_trips_full_legacy_weight_namespace() -> None:
    """The model-level public state surface preserves L6's TP compatibility."""

    source = _build_model(InteractionMode.TENSOR_PRODUCT)
    legacy_state = source.state_dict()
    assert any(".mixing.weights.g" in key for key in legacy_state)
    legacy_state.pop("_tpen_layout_fingerprint")
    restored = _build_model(InteractionMode.TENSOR_PRODUCT)
    restored.load_state_dict(legacy_state, strict=True)


def test_wrong_layout_restore_is_rejected_before_model_mutation() -> None:
    source = _build_model(InteractionMode.HYBRID)
    target = _build_model(InteractionMode.LINEAR)
    before = OrderedDict((key, value.detach().clone()) for key, value in target.state_dict().items())
    with pytest.raises(ValueError, match="layout fingerprint"):
        target.load_state_dict(source.state_dict(), strict=True)
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0.0, atol=0.0)
