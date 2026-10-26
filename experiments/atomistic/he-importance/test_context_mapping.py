"""Tests for the literal HI context-to-Hydra mapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "he_importance_context_mapping_test", Path(__file__).with_name("context_mapping.py")
)
assert _SPEC is not None and _SPEC.loader is not None
context_mapping = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = context_mapping
_SPEC.loader.exec_module(context_mapping)


PATH_LABELS = (
    "tensor",
    "linear/coordinate-neighbor",
    "linear/orbit-complete",
    "hybrid/coordinate-neighbor",
    "hybrid/orbit-complete",
)
ACTIVATION_LABELS = (
    "SiLU,SiLU",
    "tanh,tanh",
    "exact GELU,exact GELU",
    "Softplus beta2,Softplus beta2",
    "channel-MLP,SiLU",
    "SiLU,channel-MLP",
    "channel-MLP,channel-MLP",
)
ADMISSIBLE_ACTIVATION_LABELS = tuple(
    label for label in ACTIVATION_LABELS if "Softplus" not in label
)


def test_unknown_labels_refuse_without_a_default() -> None:
    with pytest.raises(context_mapping.UnknownContextLabel):
        context_mapping.context_fragment("linear/unknown")
    with pytest.raises(context_mapping.UnknownContextLabel):
        context_mapping.context_fragment("tensor", activation="unknown")


def test_linear_oracles_are_independent_literal_records() -> None:
    coordinate = context_mapping.context_fragment("linear/coordinate-neighbor")
    orbit = context_mapping.context_fragment("linear/orbit-complete")
    coordinate_paths = coordinate["layout"]["linear"]["explicit"]["_args_"][0]
    orbit_paths = orbit["layout"]["linear"]["explicit"]["_args_"][0]

    assert [
        (p["output_order"], p["input_order"], p["tau_out"], p["tau_in"])
        for p in coordinate_paths
    ] == [
        (1, 1, [0], [0]),
        (1, 1, [0], [1]),
        (2, 2, [0, 1], [0, 1]),
    ]
    assert [
        (p["output_order"], p["input_order"], p["tau_out"], p["tau_in"])
        for p in orbit_paths
    ] == [
        (1, 1, [0], [0]), (1, 1, [0], [1]),
        (1, 2, [0], [0, 1]), (1, 2, [0], [1, 0]),
        (2, 1, [0, 1], [0]), (2, 1, [0, 1], [1]),
        (2, 2, [0, 1], [0, 1]), (2, 2, [0, 1], [1, 0]),
    ]


def test_unfiltered_linear_generators_emit_support_three() -> None:
    from tpen.data.paths import LinearPathMetadata

    for policy in ("coordinate_neighbor", "orbit_complete"):
        metadata = LinearPathMetadata.generate(max_order=2, policy=policy)
        assert any(path.support_order == 3 for path in metadata.all_paths())


def test_layout_and_explicit_execution_controls_are_written() -> None:
    for label in PATH_LABELS:
        fragment = context_mapping.context_fragment(label)
        assert fragment["layout"]["input_orders"]["values"] == [1, 2]
        assert fragment["mixing"]["producers"][-1].get("activation") is None
        assert fragment["path_aggregation"]["initializer"]["stream"] == "layer0/path_aggregation"
        for producer in fragment["mixing"]["producers"]:
            assert producer["implementation"] == "vectorized"
            if producer["_target_"].endswith("LinearEquivariantMixing"):
                assert producer["aggregation"] == "sum"
        if label.startswith("hybrid/"):
            assert [p["_target_"] for p in fragment["mixing"]["producers"]] == [
                "tpen.nn.LinearEquivariantMixing", "tpen.nn.EquivariantMixing"
            ]
            assert fragment["mixing"]["activation"] is not None


def test_composite_owns_gamma_once() -> None:
    for label in PATH_LABELS:
        fragment = context_mapping.context_fragment(label)
        assert fragment["mixing"]["activation"]["_target_"] == "torch.nn.SiLU"
        assert all(producer.get("activation") is None for producer in fragment["mixing"]["producers"])


@pytest.mark.parametrize("path_label", PATH_LABELS)
@pytest.mark.parametrize("activation_label", ADMISSIBLE_ACTIVATION_LABELS)
def test_all_35_fragments_have_declared_targets(path_label: str, activation_label: str) -> None:
    fragment = context_mapping.context_fragment(path_label, activation=activation_label)
    assert fragment["mixing"]["_target_"] == "tpen.nn.CompositeMixing"
    assert fragment["path_aggregation"]["_target_"] == "tpen.nn.PathAggregation"
    assert fragment["update"]["_target_"] == "tpen.nn.ResidualUpdater"


def test_activation_parameters_and_named_streams() -> None:
    fragment = context_mapping.context_fragment("hybrid/orbit-complete", activation="channel-MLP,channel-MLP")
    producer = fragment["mixing"]["activation"]
    aggregation = fragment["path_aggregation"]["activation"]
    assert producer["layout"]["axes"]["tuple_axes_start"] == 3
    assert aggregation["layout"]["axes"]["tuple_axes_start"] == 2
    for activation in (producer, aggregation):
        specs = activation["layout"]["specs"]
        assert [(spec["channels"], spec["hidden_channels"], spec["num_hidden_layers"]) for spec in specs] == [
            (32, 128, 1),
            (32, 128, 1),
        ]
        assert all(spec["activation"]["_target_"] == "torch.nn.SiLU" for spec in specs)
        assert all(spec["bias"] is True for spec in specs)
    assert producer["initializer"]["stream"] != aggregation["initializer"]["stream"]


def test_exact_gelu_and_softplus_beta_two_are_explicit() -> None:
    gelu = context_mapping.context_fragment("tensor", activation="exact GELU,exact GELU")
    assert gelu["mixing"]["activation"] == {"_target_": "torch.nn.GELU", "approximate": "none"}
    with pytest.raises(context_mapping.UnsupportedActivationLabel):
        context_mapping.context_fragment("tensor", activation="Softplus beta2,Softplus beta2")


@pytest.mark.parametrize("path_label", PATH_LABELS)
@pytest.mark.parametrize("activation_label", ADMISSIBLE_ACTIVATION_LABELS)
def test_every_fragment_instantiates(path_label: str, activation_label: str) -> None:
    config = OmegaConf.create(
        context_mapping.context_fragment(path_label, activation=activation_label)
    )
    layout = instantiate(config.layout)
    mixing = instantiate(config.mixing)
    aggregation = instantiate(config.path_aggregation)
    layer = instantiate(config)
    assert mixing.layout.fingerprint == layout.fingerprint
    assert aggregation.layout.fingerprint == layout.fingerprint
    assert layer.layout.fingerprint == layout.fingerprint


@pytest.mark.parametrize("path_label", PATH_LABELS)
@pytest.mark.parametrize("activation_label", ADMISSIBLE_ACTIVATION_LABELS)
def test_fragment_splices_into_control_model(path_label: str, activation_label: str) -> None:
    """Resolve the fragment at the real ``model.layers[0]`` location."""

    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model.layers[0] = OmegaConf.create(
        context_mapping.context_fragment(path_label, activation=activation_label)
    )
    OmegaConf.resolve(train)
    model = instantiate(train.model)
    assert model.stack.layers[0].layout.fingerprint == model.stack.layers[0].mixing.layout.fingerprint


def _he_batch() -> object:
    from tpen.data.batch import ElectronBatch

    generator = torch.Generator().manual_seed(1729)
    positions = torch.randn((2, 2, 3), generator=generator, dtype=torch.float64)
    spins = torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64)
    nuclei = torch.zeros((1, 3), dtype=torch.float64)
    charges = torch.tensor([2.0], dtype=torch.float64)
    return ElectronBatch(positions, nuclear_positions=nuclei, nuclear_charges=charges, spins=spins)


def _inactive_order_one_parameters(model: object) -> set[int]:
    """Return parameter identities in the deliberately inactive order-1 branch."""

    layer = model.stack.layers[0]
    inactive: set[int] = set()
    for producer in layer.mixing.producers:
        if hasattr(producer, "metadata"):
            for index, path in enumerate(producer.paths):
                if path.output_order == 1:
                    inactive.add(id(producer.weights[index]))
        else:
            for path in producer.paths:
                if path.m == 1:
                    inactive.add(id(producer.weights[f"g{path.global_id}"]))
    inactive.add(id(layer.path_aggregation.weights["o1"]))
    for activation in (layer.mixing.activation, layer.path_aggregation.activation):
        if hasattr(activation, "mlps"):
            inactive.update(id(parameter) for parameter in activation.mlps[0].parameters())
    return inactive


def _pfaffian_skew_zero_parameters(model: object) -> set[int]:
    """Return the aggregation output bias that cancels in the skew kernel."""

    activation = model.stack.layers[0].path_aggregation.activation
    if not hasattr(activation, "mlps"):
        return set()
    # The order-2 final bias is added identically to entries (i,j) and (j,i).
    # PfaffianReadout antisymmetrizes those entries, so its gradient is exactly
    # zero. Hidden biases, producer-site biases, and all weights remain tested.
    return {id(activation.mlps[1].layers[-1].bias)}


@pytest.mark.parametrize("path_label", PATH_LABELS)
@pytest.mark.parametrize("activation_label", ADMISSIBLE_ACTIVATION_LABELS)
def test_all_admissible_fragments_have_finite_he_value_and_gradients(path_label: str, activation_label: str) -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model.layers[0] = OmegaConf.create(
        context_mapping.context_fragment(path_label, activation=activation_label)
    )
    OmegaConf.resolve(train)
    model = instantiate(train.model).to(dtype=torch.float64)
    output = model(_he_batch())
    assert torch.isfinite(output.logabs).all()
    # A raw sum has an exact Pfaffian cancellation for one order-2 bias;
    # squared seeded values probe reachability without changing the model.
    (output.logabs.square()).sum().backward()
    exact_zero = _pfaffian_skew_zero_parameters(model)
    for name, parameter in model.named_parameters():
        # C7 qualifies parameters introduced by the context fragment; control
        # factors and readout parameters are outside this fragment's scope.
        if not name.startswith("stack.layers.0."):
            continue
        if id(parameter) in exact_zero:
            assert parameter.grad is not None, name
            assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad)), name
            continue
        if id(parameter) in _inactive_order_one_parameters(model):
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.any(parameter.grad != 0), name


def test_tensor_silu_matches_train_control_state_and_value() -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    torch.manual_seed(41)
    control = instantiate(train.model).to(dtype=torch.float64)
    train.model.layers[0] = OmegaConf.create(context_mapping.context_fragment("tensor"))
    OmegaConf.resolve(train)
    torch.manual_seed(41)
    mapped = instantiate(train.model).to(dtype=torch.float64)
    control_state = {key: value for key, value in control.state_dict().items() if key != "_tpen_layout_fingerprint"}
    mapped_state = {key: value for key, value in mapped.state_dict().items() if key != "_tpen_layout_fingerprint"}
    assert tuple(control_state) == tuple(mapped_state)
    assert all(torch.equal(control_state[key], mapped_state[key]) for key in control_state)
    control_value = control(_he_batch()).logabs
    mapped_value = mapped(_he_batch()).logabs
    assert torch.equal(control_value, mapped_value)


def _zero_tuple_entries(*, diagonal: bool, distinct: bool):
    """Build a hook that zeros selected post-Gamma order-2 entries."""

    def hook(_module: object, _inputs: tuple[object, ...], output: torch.Tensor) -> torch.Tensor:
        if output.ndim != 5:
            return output
        masked = output.clone()
        if diagonal:
            indices = torch.arange(output.shape[-1], device=output.device)
            masked[..., indices, indices] = 0
        if distinct:
            masked[..., 0, 1] = 0
        return masked

    return hook


def test_biased_producer_channel_mlp_non_distinct_leakage_is_measured() -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model.layers[0] = OmegaConf.create(
        context_mapping.context_fragment("tensor", activation="channel-MLP,SiLU")
    )
    OmegaConf.resolve(train)
    torch.manual_seed(73)
    model = instantiate(train.model).to(dtype=torch.float64)
    batch = _he_batch()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "mixing.activation" in name and name.endswith("bias"):
                parameter.fill_(0.25)
    full = model(batch).logabs.detach()
    activation = model.stack.layers[0].mixing.activation
    diagonal_handle = activation.register_forward_hook(
        _zero_tuple_entries(diagonal=True, distinct=False)
    )
    try:
        diagonal_masked = model(batch).logabs.detach()
    finally:
        diagonal_handle.remove()
    distinct_handle = activation.register_forward_hook(
        _zero_tuple_entries(diagonal=False, distinct=True)
    )
    try:
        distinct_masked = model(batch).logabs.detach()
    finally:
        distinct_handle.remove()
    non_distinct_delta = float((full - diagonal_masked).abs().max())
    distinct_delta = float((full - distinct_masked).abs().max())
    print(f"C8 non_distinct_post_gamma_zero max_delta={non_distinct_delta:.17g}")
    print(f"C8 distinct_post_gamma_zero_control max_delta={distinct_delta:.17g}")
    assert torch.isfinite(diagonal_masked).all()
    assert torch.isfinite(distinct_masked).all()
    assert distinct_delta > 0.0
