"""Tests for the literal HI context-to-Hydra mapping."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from fractions import Fraction

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from tests.helpers.run_context import make_run_context


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
    with pytest.raises(context_mapping.UnknownContextLabel):
        context_mapping.input_feature_fragment("bounded distances")


def test_raw_input_feature_package_preserves_the_literal_control() -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    before = OmegaConf.to_container(train.model, resolve=True)
    train.model = OmegaConf.merge(
        train.model,
        OmegaConf.create(context_mapping.input_feature_fragment("raw")),
    )
    OmegaConf.resolve(train)
    assert OmegaConf.to_container(train.model, resolve=True) == before


def test_package_labels_are_the_declared_two_axis_grid() -> None:
    assert context_mapping.package_fragment("B00") == ({}, None)
    assert context_mapping.package_fragment("B10") == (
        context_mapping.input_feature_fragment("raw plus bounded distances"), None
    )
    for package in ("B01", "B11"):
        fragment, factor = context_mapping.package_fragment(package)
        assert fragment in ({}, context_mapping.input_feature_fragment("raw plus bounded distances"))
        assert factor == {
            "_target_": "tpen.nn.jastrow.BoundedTwoCoefficientJastrow",
            "trainable": True,
            "length": 1.0,
            "mean_coefficient": 0.0,
            "difference_coefficient": 0.0,
        }
    with pytest.raises(context_mapping.UnknownContextLabel):
        context_mapping.package_fragment("B99")
    with pytest.raises(context_mapping.UnknownContextLabel):
        context_mapping.jastrow_fragment("implicit")


def test_jastrow_is_appended_without_changing_existing_factors() -> None:
    original = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    model = OmegaConf.create(context_mapping.compose_package(original.model, "B01"))
    assert model.factors[:2] == original.model.factors
    assert model.factors[2] == {
        "_target_": "tpen.nn.jastrow.BoundedTwoCoefficientJastrow",
        "trainable": True,
        "length": 1.0,
        "mean_coefficient": 0.0,
        "difference_coefficient": 0.0,
    }


def test_composed_mapping_preserves_unresolved_root_interpolations() -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    composed = context_mapping.compose_package(train.model, "B01")
    def assert_plain(value: object) -> None:
        assert not OmegaConf.is_config(value)
        if isinstance(value, dict):
            for child in value.values():
                assert_plain(child)
        elif isinstance(value, list):
            for child in value:
                assert_plain(child)

    assert_plain(composed)
    assert composed["embedding"]["spatial_dim"] == "${system.spatial_dim}"
    assert composed["embedding"]["initializer"]["seed"] == "${runtime.seed}"
    assert composed["factors"][1]["atoms"] == "${atoms}"


@pytest.mark.parametrize("package", ("B00", "B10", "B01", "B11"))
def test_spliced_package_preserves_all_nonpackage_config_values(package: str) -> None:
    control = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    expected = OmegaConf.to_container(control, resolve=False)
    cfg = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    cfg.model = context_mapping.compose_package(cfg.model, package)
    actual_unresolved = OmegaConf.to_container(cfg, resolve=False)
    assert {key: actual_unresolved[key] for key in actual_unresolved if key != "model"} == {
        key: expected[key] for key in expected if key != "model"
    }
    OmegaConf.resolve(cfg)
    actual = OmegaConf.to_container(cfg, resolve=True)
    assert actual["runner"]["model"] == actual["model"]


def test_standalone_composed_mapping_exposes_the_root_resolution_hazard() -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    standalone = OmegaConf.create(context_mapping.compose_package(train.model, "B01"))
    with pytest.raises(Exception, match="Interpolation key"):
        OmegaConf.to_container(standalone, resolve=True)


def test_naive_package_merge_would_replace_the_control_factors() -> None:
    original = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    fragment, factor = context_mapping.package_fragment("B01")
    naive = OmegaConf.merge(
        original.model,
        OmegaConf.create({**fragment, "factors": [factor]}),
    )
    assert list(naive.factors) == [factor]
    assert list(original.model.factors) != list(naive.factors)


def _validate_through_hydra(train: object) -> None:
    """Reach the closed HI validator only through its declared Hydra target."""

    OmegaConf.resolve(train)
    instantiate(
        {
            "_target_": "tpen.hi_schema.validate_hi_train_config",
            "_recursive_": False,
            "cfg": train,
            "env": {},
        }
    )


@pytest.mark.parametrize("package", ("B01", "B11"))
def test_two_coefficient_jastrow_packages_pass_the_closed_hi_schema(package: str) -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model = context_mapping.compose_package(train.model, package)
    _validate_through_hydra(train)


@pytest.mark.parametrize("package", ("B01", "B11"))
def test_closed_hi_schema_refuses_a_frozen_two_coefficient_jastrow(package: str) -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model = context_mapping.compose_package(train.model, package)
    train.model.factors[-1].trainable = False
    with pytest.raises(Exception, match="undeclared-trainability"):
        _validate_through_hydra(train)


@pytest.mark.parametrize(("jastrow_package", "control_package"), [("B01", "B00"), ("B11", "B10")])
def test_zero_jastrow_is_exactly_the_corresponding_control(
    jastrow_package: str, control_package: str
) -> None:
    mapped = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    mapped.model = context_mapping.compose_package(mapped.model, jastrow_package)
    control = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    control.model = context_mapping.compose_package(control.model, control_package)
    OmegaConf.resolve(mapped)
    OmegaConf.resolve(control)
    torch.manual_seed(1729)
    mapped_model = instantiate(mapped.model).to(dtype=torch.float64)
    torch.manual_seed(1729)
    control_model = instantiate(control.model).to(dtype=torch.float64)
    mapped_output = mapped_model(_he_batch())
    control_output = control_model(_he_batch())
    assert torch.equal(mapped_output.logabs, control_output.logabs)
    assert torch.equal(mapped_output.sign, control_output.sign)
    factor = mapped_model.factors[-1]
    assert torch.equal(factor.mean_coefficient, torch.zeros_like(factor.mean_coefficient))
    assert torch.equal(factor.difference_coefficient, torch.zeros_like(factor.difference_coefficient))


def test_all_four_matched_starts_have_exactly_the_same_initial_function() -> None:
    """The B grid starts from one function, including the O-composite cells."""

    outputs = []
    for package in ("B00", "B10", "B01", "B11"):
        train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
        train.model = context_mapping.compose_package(train.model, package)
        OmegaConf.resolve(train)
        torch.manual_seed(1729)
        model = instantiate(train.model).to(dtype=torch.float64)
        outputs.append(model(_he_batch()))

    for output in outputs[1:]:
        assert torch.equal(output.logabs, outputs[0].logabs)
        assert torch.equal(output.sign, outputs[0].sign)


@pytest.mark.parametrize("package", ("B10", "B11"))
def test_matched_start_added_columns_have_the_right_first_layer_gradient(
    package: str,
) -> None:
    """Check the outer-product identity for each order's first layer."""

    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model = context_mapping.compose_package(train.model, package)
    OmegaConf.resolve(train)
    torch.manual_seed(1729)
    model = instantiate(train.model).to(dtype=torch.float64)
    batch = instantiate(
        {
            "_target_": "tpen.data.batch.ElectronBatch",
            "positions": torch.tensor(
                [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=torch.float64
            ),
            "nuclear_positions": torch.zeros((1, 3), dtype=torch.float64),
            "nuclear_charges": torch.tensor([2.0], dtype=torch.float64),
            "spins": torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        }
    )
    captured = {}
    outputs = {}
    hooks = []
    for order in (1, 2):
        layer = model.embedding.order_mlps[str(order)].layers[0]
        hooks.append(
            layer.register_forward_pre_hook(
                lambda module, inputs, order=order: captured.setdefault(order, inputs[0])
            )
        )
        hooks.append(
            layer.register_forward_hook(
                lambda module, inputs, output, order=order: (
                    output.retain_grad(),
                    outputs.setdefault(order, output),
                )[-1]
            )
        )
    try:
        logabs = model(batch).logabs[0]
        expected_all = {
            1: torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0, 1 / 2],
                    [0.0, 2.0, 0.0, -1.0, 4 / 5],
                ],
                dtype=torch.float64,
            ),
            2: torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0, 1 / 2, 1.0, 0.0, 0.0, 1.0, 1 / 2, 1 / 2, 1 / 2, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 1 / 2, 0.0, 2.0, 0.0, -1.0, 4 / 5, 1 / 2, 4 / 5, 5 / 6],
                    [0.0, 2.0, 0.0, -1.0, 4 / 5, 1.0, 0.0, 0.0, 1.0, 1 / 2, 4 / 5, 1 / 2, 5 / 6],
                    [0.0, 2.0, 0.0, -1.0, 4 / 5, 0.0, 2.0, 0.0, -1.0, 4 / 5, 4 / 5, 4 / 5, 0.0],
                ],
                dtype=torch.float64,
            ),
        }
        weights = {
            order: model.embedding.order_mlps[str(order)].layers[0].weight
            for order in (1, 2)
        }
        g = torch.autograd.grad(
            logabs,
            [outputs[1], outputs[2], weights[1], weights[2]],
            retain_graph=True,
        )
        assert len(captured) == 2
        assert len(outputs) == 2
        for order, delta_gradient, weight_gradient in (
            (1, g[0], g[2]),
            (2, g[1], g[3]),
        ):
            expected = expected_all[order]
            inputs = captured[order][0]
            delta = delta_gradient[0]
            hidden = weights[order].shape[0]
            if order == 1:
                assert inputs.shape == (2, 5)
                assert delta.shape == (2, hidden)
            else:
                assert inputs.shape == (2, 2, 13)
                assert delta.shape == (2, 2, hidden)
            inputs = inputs.reshape(-1, inputs.shape[-1])
            delta = delta.reshape(-1, delta.shape[-1])
            torch.testing.assert_close(inputs, expected, rtol=1e-12, atol=1e-12)
            assert inputs.shape == expected.shape
            gradient = weight_gradient
            expected_gradient = delta.transpose(0, 1) @ inputs
            added = (4,) if order == 1 else (4, 9, 10, 11, 12)
            for added_index in added:
                torch.testing.assert_close(
                    gradient[:, added_index],
                    expected_gradient[:, added_index],
                    rtol=1e-12,
                    atol=1e-12,
                )
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("package", ("B01", "B11"))
def test_jastrow_coefficient_derivatives_match_the_explicit_two_electron_oracle(
    package: str,
) -> None:
    """Check each sample's coefficient derivatives against the stated formula.

    The coordinates and q values are literal rational values, so this oracle
    does not share the factor's distance implementation.  Float64 arithmetic
    makes 1e-12 tight enough to distinguish an incorrect pair, coefficient, or
    factor of two while allowing only roundoff in the explicit fractions.
    """

    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model = context_mapping.compose_package(train.model, package)
    OmegaConf.resolve(train)
    model = instantiate(train.model).to(dtype=torch.float64)
    factor = model.factors[-1]
    positions = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[1.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    batch = instantiate(
        {
            "_target_": "tpen.data.batch.ElectronBatch",
            "positions": positions,
            "nuclear_positions": torch.zeros((1, 3), dtype=torch.float64),
            "nuclear_charges": torch.tensor([2.0], dtype=torch.float64),
            "spins": torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64),
        }
    )
    logabs = model(batch).logabs
    mean_gradients = []
    difference_gradients = []
    for sample in range(2):
        mean_gradients.append(
            torch.autograd.grad(logabs[sample], factor.theta_s, retain_graph=True)[0]
        )
        difference_gradients.append(
            torch.autograd.grad(logabs[sample], factor.theta_t, retain_graph=True)[0]
        )
    q_one = (Fraction(1, 2), Fraction(2, 3))
    q_two = (Fraction(4, 5), Fraction(4, 5))
    q_pair = (Fraction(5, 6), Fraction(2, 3))
    expected_mean = torch.tensor(
        [float(pair * (first + second) / 2) for first, second, pair in zip(q_one, q_two, q_pair)],
        dtype=torch.float64,
    )
    expected_difference = torch.tensor(
        [float(pair * (first - second) ** 2) for first, second, pair in zip(q_one, q_two, q_pair)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        torch.stack(mean_gradients), expected_mean, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        torch.stack(difference_gradients), expected_difference, rtol=1e-12, atol=1e-12
    )


def test_bounded_distance_input_fragment_is_explicit() -> None:
    assert context_mapping.input_feature_fragment("raw plus bounded distances") == {
        "basis": {
            "_target_": "tpen.nn.basis.BoundedDistanceBasis",
            "inner": {
                "_target_": "tpen.nn.RawCoordinateBasis",
                "spatial_dim": 3,
                "include_spin": True,
            },
            "length": 1.0,
        },
        "embedding": {"in_features": 5, "pair_input_channels": 3},
    }


def test_linear_oracles_are_independent_literal_records() -> None:
    coordinate = context_mapping.context_fragment("linear/coordinate-neighbor")
    orbit = context_mapping.context_fragment("linear/orbit-complete")
    coordinate_paths = coordinate["layout"]["linear"]["explicit"]
    orbit_paths = orbit["layout"]["linear"]["explicit"]

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
    for policy in ("coordinate_neighbor", "orbit_complete"):
        metadata = instantiate(
            {
                "_target_": "tpen.data.paths.LinearPathMetadata.generate",
                "max_order": 2,
                "policy": policy,
            }
        )
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


@pytest.mark.parametrize("label", ("(16,16)", "(24,24)", "(32,32)", "(48,48)", "(40,24)", "(24,40)"))
def test_a4_a5_channel_fragments_reach_embedding_and_readout(label: str) -> None:
    fragment = context_mapping.channel_fragment(label)
    expected = tuple(int(value) for value in label.strip("()").split(","))
    assert fragment["embedding"]["out_channels"] == {1: expected[0], 2: expected[1]}
    assert fragment["readout"]["pair_channels"] == expected[1]
    # A stale control readout is a deliberate falsifier for the A4/A5 route.
    assert fragment["readout"]["pair_channels"] != 32 or expected[1] == 32


def test_a4_a5_channel_vocabulary_refuses_near_miss() -> None:
    with pytest.raises(context_mapping.UnknownContextLabel):
        context_mapping.channel_fragment("(32,24)")


@pytest.mark.parametrize("label", ("(64,1)", "(128,1)", "(128,2)", "(256,2)"))
def test_a7_embedding_fragments_are_exact(label: str) -> None:
    hidden, depth = (int(value) for value in label.strip("()").split(","))
    assert context_mapping.embedding_fragment(label) == {
        "embedding": {"hidden_channels": hidden, "num_hidden_layers": depth}
    }
    with pytest.raises(context_mapping.UnknownContextLabel):
        context_mapping.embedding_fragment("(128,3)")


def test_a8_fragments_select_the_update_seam_and_only_gaussian_adds_an_envelope() -> None:
    residual = context_mapping.update_fragment("residual")
    replacement = context_mapping.update_fragment("replacement")
    gaussian = context_mapping.update_fragment("Gaussian residual")
    assert residual == {"update": {"_target_": "tpen.nn.ResidualUpdater"}}
    assert replacement == {"update": {"_target_": "tpen.nn.ReplaceUpdater"}}
    assert gaussian["update"] == residual["update"]
    assert gaussian["update_envelope"] == {
        "_target_": "tpen.nn.coordinate_envelopes.GaussianCoordinateEnvelope",
        "sigma": 1.0,
    }
    with pytest.raises(context_mapping.UnknownContextLabel):
        context_mapping.update_fragment("Gaussian residual sigma=2")


@pytest.mark.parametrize("label", ("residual", "replacement", "Gaussian residual"))
def test_a8_fragments_instantiate_a_layer_with_the_declared_update(label: str) -> None:
    layer_config = context_mapping.architecture_fragment(update=label)["layers"][0]
    layer = instantiate(OmegaConf.create(layer_config))
    assert layer.update.__class__.__name__ == (
        "ReplaceUpdater" if label == "replacement" else "ResidualUpdater"
    )
    if label == "Gaussian residual":
        assert layer.update_envelope.__class__.__name__ == "GaussianCoordinateEnvelope"
        assert layer.update_envelope.sigma == 1.0
    else:
        assert layer.update_envelope is None


@pytest.mark.parametrize("label", ("residual", "replacement", "Gaussian residual"))
def test_a8_instantiated_update_has_the_declared_zero_and_nonzero_falsifier(label: str) -> None:
    layer = instantiate(OmegaConf.create(context_mapping.architecture_fragment(update=label)["layers"][0]))
    blocks = [
        torch.zeros(1, 0, dtype=torch.float64),
        torch.full((1, 2, 2), 3.0, dtype=torch.float64),
        torch.full((1, 2, 2, 2), 3.0, dtype=torch.float64),
    ]
    x = instantiate({"_target_": "tpen.data.real.Feature", "blocks": blocks})
    zero = instantiate({
        "_target_": "tpen.data.real.Update",
        "blocks": [torch.zeros_like(block) for block in blocks],
    })
    zero_output = layer.update(x, zero)
    if label == "replacement":
        assert all(torch.equal(block, torch.zeros_like(block)) for block in zero_output.blocks)
    else:
        assert all(torch.equal(left, right) for left, right in zip(x.blocks, zero_output.blocks))

    update = instantiate({
        "_target_": "tpen.data.real.Update",
        "blocks": [torch.ones_like(block) for block in blocks],
    })
    if layer.update_envelope is not None:
        context = instantiate({
            "_target_": "tpen.nn.context.TPENForwardContext",
            "_convert_": "object",
            "batch": {
                "_target_": "tpen.data.batch.ElectronBatch",
                "positions": torch.tensor(
                    [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=torch.float64
                ),
                "nuclear_positions": torch.zeros((1, 3), dtype=torch.float64),
                "nuclear_charges": torch.tensor([2.0], dtype=torch.float64),
            },
        })
        assert context.batch.__class__.__name__ == "ElectronBatch"
        update = layer.update_envelope(update, context)
    output = layer.update(x, update)
    if label == "replacement":
        assert all(torch.equal(block, torch.ones_like(block)) for block in output.blocks)
    elif label == "residual":
        assert all(torch.equal(block, expected) for block, expected in zip(output.blocks, [
            torch.zeros(1, 0, dtype=torch.float64),
            torch.full((1, 2, 2), 4.0, dtype=torch.float64),
            torch.full((1, 2, 2, 2), 4.0, dtype=torch.float64),
        ]))
    else:
        gate = torch.exp(torch.tensor(-5.0 / 2.0, dtype=torch.float64))
        assert torch.allclose(output.blocks[1], torch.full_like(output.blocks[1], 3.0 + gate))
        assert torch.allclose(output.blocks[2], torch.full_like(output.blocks[2], 3.0 + gate))


@pytest.mark.parametrize(("builder", "labels"), [
    (context_mapping.producer_initialization_fragment, (".25", ".50", "1.0")),
    (context_mapping.electron_electron_fragment, (".25", "1", "4")),
    (context_mapping.electron_nucleus_fragment, ("(.0025,.25)", "(.01,1)", "(.04,4)")),
    (context_mapping.readout_initialization_fragment, ("equal", "signed L1", "random-direction L2")),
])
def test_a9_to_a12_vocabularies_refuse_undeclared_levels(builder: object, labels: tuple[str, ...]) -> None:
    for label in labels:
        builder(label)  # type: ignore[operator]
    with pytest.raises(context_mapping.UnknownContextLabel):
        builder("not-a-declared-level")  # type: ignore[operator]


def test_a10_and_a11_fragments_preserve_trainability_and_invert_at_helium() -> None:
    e_e = context_mapping.electron_electron_fragment(".25")
    assert e_e["same_range_parameter"] == 1.0
    assert e_e["opposite_range_parameter"] == 0.25
    assert e_e["trainable_range"] is True
    for label, (curvature, radius) in {
        "(.0025,.25)": (0.0025, 0.25),
        "(.01,1)": (0.01, 1.0),
        "(.04,4)": (0.04, 4.0),
    }.items():
        law = context_mapping.electron_nucleus_fragment(label)
        assert law["law"]["curvature_range"] == radius
        assert law["law"]["tail_slope"] == pytest.approx(2.0 - curvature / radius)
        assert law["law"]["trainable"] is True


@pytest.mark.parametrize("label", ("(.0025,.25)", "(.01,1)", "(.04,4)"))
def test_a11_laws_instantiate_with_trainable_raw_coordinates(label: str) -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model = context_mapping.compose_architecture_package(
        train.model, electron_nucleus=label
    )
    OmegaConf.resolve(train)
    factor = instantiate(train.model.factors[1])
    law = factor.law
    assert law.raw_range.requires_grad
    assert law.raw_kappa.requires_grad
    curvature, radius = {
        "(.0025,.25)": (0.0025, 0.25),
        "(.01,1)": (0.01, 1.0),
        "(.04,4)": (0.04, 4.0),
    }[label]
    charges = torch.tensor([2.0], dtype=torch.float64)
    torch.testing.assert_close(law.curvature_range, torch.tensor(radius, dtype=torch.float64))
    torch.testing.assert_close(
        law.curvature_coefficient(charges),
        torch.tensor([curvature], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_a12_random_modes_use_the_model_seed_named_stream() -> None:
    for label, mode in (("equal", "equal"), ("signed L1", "signed_l1"), ("random-direction L2", "random_direction_l2")):
        fragment = context_mapping.readout_initialization_fragment(label, seed=830001)
        assert fragment["initialization"] == mode
        if mode == "equal":
            assert "initializer" not in fragment
        else:
            assert fragment["initializer"] == {
                "_target_": "tpen.nn.initialization.TorchInitializer",
                "seed": 830001,
                "stream": "readout/a12",
            }


def test_full_architecture_composition_is_merge_shaped_and_preserves_factor_list() -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    original_factors = OmegaConf.to_container(train.model.factors, resolve=False)
    composed = context_mapping.compose_architecture_package(
        train.model,
        channels="(40,24)",
        embedding="(256,2)",
        update="Gaussian residual",
        producer_initialization=".25",
        electron_electron="4",
        electron_nucleus="(.04,4)",
        readout_initialization="signed L1",
        seed=830001,
    )
    assert len(composed["factors"]) == len(original_factors)
    assert composed["factors"][0]["opposite_range_parameter"] == 4.0
    assert composed["factors"][1]["law"]["curvature_range"] == 4.0
    assert composed["readout"]["pair_channels"] == 24
    assert composed["layers"][0]["update_envelope"]["sigma"] == 1.0
    assert composed["embedding"]["spatial_dim"] == "${system.spatial_dim}"
    assert composed["embedding"]["initializer"]["seed"] == "${runtime.seed}"
    assert "factors" not in context_mapping.architecture_fragment()


def test_existing_context_fragment_output_is_pinned_to_the_ab2_head() -> None:
    """Compare the current public output with AB-2's mechanically loaded output."""

    source = subprocess.check_output(
        [
            "git",
            "show",
            "fd91de8cb7af7ecbff27536ff9dc474e8832df66:experiments/atomistic/he-importance/context_mapping.py",
        ],
        text=True,
    )
    legacy = importlib.util.module_from_spec(importlib.util.spec_from_loader("legacy_mapping", loader=None))
    exec(compile(source, "legacy_context_mapping.py", "exec"), legacy.__dict__)
    cases = (
        ("tensor", "SiLU,SiLU", 32, 0),
        ("linear/coordinate-neighbor", "tanh,tanh", 32, 17),
        ("hybrid/orbit-complete", "channel-MLP,channel-MLP", 32, 99),
    )
    for path, activation, channels, seed in cases:
        current = context_mapping.context_fragment(
            path, activation=activation, channels=channels, seed=seed
        )
        previous = legacy.context_fragment(
            path, activation=activation, channels=channels, seed=seed
        )
        current_bytes = json.dumps(current, sort_keys=True, separators=(",", ":")).encode()
        previous_bytes = json.dumps(previous, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(current_bytes).digest() == hashlib.sha256(previous_bytes).digest()


@pytest.mark.parametrize("path_label", PATH_LABELS)
@pytest.mark.parametrize("initialization", (".25", ".50", "1.0"))
def test_a9_instantiated_constant_reaches_every_producer(
    path_label: str, initialization: str
) -> None:
    config = OmegaConf.create(
        context_mapping.architecture_fragment(
            producer_path=path_label,
            producer_initialization=initialization,
        )["layers"][0]
    )
    layer = instantiate(config)
    expected = float(initialization)
    for producer in layer.mixing.producers:
        weights = tuple(producer.parameters())
        assert weights
        for weight in weights:
            assert torch.equal(weight.detach(), torch.full_like(weight.detach(), expected))


@pytest.mark.parametrize("path_label", PATH_LABELS)
@pytest.mark.parametrize("update", ("residual", "replacement", "Gaussian residual"))
@pytest.mark.parametrize("package", ("B00", "B10", "B01", "B11"))
def test_b_composition_instantiates_every_producer_update_and_package(
    path_label: str, update: str, package: str
) -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model = context_mapping.compose_architecture_package(
        train.model,
        package=package,
        producer_path=path_label,
        update=update,
    )
    OmegaConf.resolve(train)
    model = instantiate(train.model).to(dtype=torch.float64)
    assert len(model.factors) == (3 if package in {"B01", "B11"} else 2)
    jastrow_count = sum(
        factor.__class__.__name__ == "BoundedTwoCoefficientJastrow"
        for factor in model.factors
    )
    assert jastrow_count == (1 if package in {"B01", "B11"} else 0)
    if package in {"B01", "B11"}:
        jastrow = model.factors[-1]
        assert jastrow.trainable is True
        assert torch.equal(jastrow.mean_coefficient, torch.zeros_like(jastrow.mean_coefficient))
        assert torch.equal(jastrow.difference_coefficient, torch.zeros_like(jastrow.difference_coefficient))
        assert jastrow.length == 1.0


@pytest.mark.parametrize("path_label", PATH_LABELS)
@pytest.mark.parametrize("label", ("(40,24)", "(24,40)"))
def test_instantiated_channel_package_reaches_every_order_specific_consumer(
    path_label: str, label: str
) -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model = context_mapping.compose_architecture_package(
        train.model,
        producer_path=path_label,
        activation="channel-MLP,channel-MLP",
        channels=label,
    )
    OmegaConf.resolve(train)
    model = instantiate(train.model).to(dtype=torch.float64)
    expected = tuple(int(value) for value in label.strip("()").split(","))
    features = model.embedding(_he_batch())
    layer = model.stack.layers[0]
    interaction = layer.mixing(features)
    update = layer.path_aggregation(interaction)
    layer.update(features, update)
    assert features.blocks[1].shape[1] == expected[0]
    assert features.blocks[2].shape[1] == expected[1]
    assert interaction.blocks[1].shape[1] == expected[0]
    assert interaction.blocks[2].shape[1] == expected[1]
    assert update.blocks[1].shape[1] == expected[0]
    assert update.blocks[2].shape[1] == expected[1]
    assert model.readout.pair_channels == expected[1]
    for producer in model.stack.layers[0].mixing.producers:
        output_channels = getattr(producer, "output_channels", None)
        if output_channels is not None:
            assert output_channels.for_order(1) == expected[0]
            assert output_channels.for_order(2) == expected[1]
        else:
            assert producer.out_channels[1] == expected[0]
            assert producer.out_channels[2] == expected[1]
    for activation_module in (layer.mixing.activation, layer.path_aggregation.activation):
        assert activation_module.mlps[0].layers[0].out_features == 4 * expected[0]
        assert activation_module.mlps[1].layers[0].out_features == 4 * expected[1]


@pytest.mark.parametrize("label", ("signed L1", "random-direction L2"))
@pytest.mark.parametrize("seed", (830001, 830002))
def test_a12_mapped_random_modes_are_named_streamed_and_paired(
    label: str, seed: int
) -> None:
    fragment = context_mapping.readout_initialization_fragment(label, seed=seed)
    config = {
        "_target_": "tpen.nn.readout.PfaffianReadout",
        "pair_channels": 8,
        "trainable": True,
        **fragment,
    }
    torch.manual_seed(7)
    first = instantiate(OmegaConf.create(config))
    torch.manual_seed(7000007)
    second = instantiate(OmegaConf.create(config))
    assert torch.equal(first.channel_weights.detach(), second.channel_weights.detach())
    other_seed_config = {
        "_target_": "tpen.nn.readout.PfaffianReadout",
        "pair_channels": 8,
        "trainable": True,
        **context_mapping.readout_initialization_fragment(label, seed=seed + 1),
    }
    other_seed = instantiate(OmegaConf.create(other_seed_config))
    assert not torch.equal(first.channel_weights.detach(), other_seed.channel_weights.detach())
    l2_config = {
        "_target_": "tpen.nn.readout.PfaffianReadout",
        "pair_channels": 8,
        "trainable": True,
        **context_mapping.readout_initialization_fragment("random-direction L2", seed=seed),
    }
    l2 = instantiate(OmegaConf.create(l2_config))
    l1 = instantiate(OmegaConf.create({
        "_target_": "tpen.nn.readout.PfaffianReadout",
        "pair_channels": 8,
        "trainable": True,
        **context_mapping.readout_initialization_fragment("signed L1", seed=seed),
    }))
    ratio = l1.channel_weights.detach() / l2.channel_weights.detach()
    assert torch.all(ratio > 0)
    assert torch.allclose(ratio, torch.full_like(ratio, ratio[0]))
    assert torch.equal(l1.channel_weights.detach(), first.channel_weights.detach()) if label == "signed L1" else True
    assert torch.isclose(l1.channel_weights.detach().abs().sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(
        torch.sqrt(torch.tensor(8.0, dtype=torch.float64))
        * torch.linalg.vector_norm(l2.channel_weights.detach()),
        torch.tensor(1.0, dtype=torch.float64),
    )
    nonuniform = l1.channel_weights.detach() * torch.linspace(
        1.0, 2.0, 8, dtype=torch.float64
    )
    nonuniform = nonuniform / nonuniform.abs().sum()
    mutant_ratio = nonuniform / l2.channel_weights.detach()
    assert not torch.allclose(mutant_ratio, torch.full_like(mutant_ratio, mutant_ratio.mean()))


@pytest.mark.parametrize("package", ("B00", "B10", "B01", "B11"))
def test_full_architecture_composes_each_b_package_without_factor_replacement(package: str) -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    composed = context_mapping.compose_architecture_package(train.model, package=package)
    assert len(composed["factors"]) == (3 if package in {"B01", "B11"} else 2)
    assert [factor["_target_"] for factor in composed["factors"]].count(
        "tpen.nn.jastrow.BoundedTwoCoefficientJastrow"
    ) == (1 if package in {"B01", "B11"} else 0)


def test_literal_control_architecture_has_same_model_state_as_base_config() -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    seed = 830001
    mapped_root = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    mapped_root.model = context_mapping.compose_architecture_package(
        mapped_root.model, seed=seed
    )
    train.runtime.seed = seed
    mapped_root.runtime.seed = seed
    OmegaConf.resolve(train)
    OmegaConf.resolve(mapped_root)
    torch.manual_seed(1729)
    control_model = instantiate(train.model).to(dtype=torch.float64)
    torch.manual_seed(7000007)
    mapped_model = instantiate(mapped_root.model).to(dtype=torch.float64)
    mapped_state = {key: value for key, value in mapped_model.state_dict().items() if key != "_tpen_layout_fingerprint"}
    control_state = {key: value for key, value in control_model.state_dict().items() if key != "_tpen_layout_fingerprint"}
    assert tuple(mapped_state) == tuple(control_state)
    assert all(torch.equal(mapped_state[key], control_state[key]) for key in control_state)
    assert sum(parameter.numel() for parameter in mapped_model.parameters()) == sum(
        parameter.numel() for parameter in control_model.parameters()
    )
    mapped_output = mapped_model(_he_batch())
    control_output = control_model(_he_batch())
    assert torch.equal(mapped_output.logabs, control_output.logabs)
    assert torch.equal(mapped_output.sign, control_output.sign)


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
    generator = torch.Generator().manual_seed(1729)
    positions = torch.randn((2, 2, 3), generator=generator, dtype=torch.float64)
    spins = torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64)
    nuclei = torch.zeros((1, 3), dtype=torch.float64)
    charges = torch.tensor([2.0], dtype=torch.float64)
    return instantiate(
        {
            "_target_": "tpen.data.batch.ElectronBatch",
            "positions": positions,
            "nuclear_positions": nuclei,
            "nuclear_charges": charges,
            "spins": spins,
        }
    )


def _run_short_real_training(
    package: str,
    tmp_path: Path,
    *,
    trainable: bool = True,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, list[torch.Tensor | None]]:
    """Run one reduced update through the runner composed by ``train.yaml``."""

    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model = context_mapping.compose_package(train.model, package)
    if not trainable:
        train.model.factors[-1].trainable = False
    train.runtime.device = "cpu"
    train.sampler.n_walkers = 4
    train.sampler.burn_in = 1
    train.sampler.n_steps = 1
    train.trainer.max_steps = 1
    OmegaConf.resolve(train)
    runner = instantiate(train.runner)
    factor = runner.model.factors[-1]
    initial = (
        factor.mean_coefficient.detach().clone(),
        factor.difference_coefficient.detach().clone(),
    )
    assert torch.equal(initial[0], torch.zeros_like(initial[0]))
    assert torch.equal(initial[1], torch.zeros_like(initial[1]))
    captured: list[torch.Tensor | None] = []
    original_step = torch.optim.Adam.step

    def capture_step(optimizer: torch.optim.Adam, *args: object, **kwargs: object) -> object:
        for parameter in (factor.theta_s, factor.theta_t) if trainable else ():
            captured.append(None if parameter.grad is None else parameter.grad.detach().clone())
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", capture_step)
    runner.run(make_run_context(tmp_path, run_id=f"{package}-{trainable}"))
    return factor, captured


_ADDED_INPUT_COLUMNS = {1: (4,), 2: (4, 9, 10, 11, 12)}


def _run_short_embedding_training(
    package: str,
    tmp_path: Path,
    *,
    freeze_added: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[dict[int, torch.Tensor]]]:
    """Run two real training updates while observing embedding first-layer columns."""

    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model = context_mapping.compose_package(train.model, package)
    train.runtime.device = "cpu"
    train.sampler.n_walkers = 4
    train.sampler.burn_in = 1
    train.sampler.n_steps = 1
    train.trainer.max_steps = 2
    OmegaConf.resolve(train)
    runner = instantiate(train.runner)
    weights = {
        order: runner.model.embedding.order_mlps[str(order)].layers[0].weight
        for order in (1, 2)
    }
    initial = {order: weight.detach().clone() for order, weight in weights.items()}
    for order, weight in weights.items():
        added = _ADDED_INPUT_COLUMNS[order]
        torch.testing.assert_close(weight[:, added], torch.zeros_like(weight[:, added]))

    captured_gradients: list[dict[int, torch.Tensor]] = []
    original_step = torch.optim.Adam.step

    def capture_step(optimizer: torch.optim.Adam, *args: object, **kwargs: object) -> object:
        step_gradients = {}
        for order, weight in weights.items():
            assert weight.grad is not None
            step_gradients[order] = weight.grad.detach().clone()
            if freeze_added:
                weight.grad[:, _ADDED_INPUT_COLUMNS[order]] = 0
        captured_gradients.append(step_gradients)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", capture_step)
    runner.run(make_run_context(tmp_path, run_id=f"{package}-embedding-{freeze_added}"))
    return weights, initial, captured_gradients


@pytest.mark.parametrize("package", ("B10", "B11"))
def test_added_embedding_columns_move_in_a_real_short_training_run(
    package: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights, initial, gradients = _run_short_embedding_training(
        package, tmp_path, freeze_added=False, monkeypatch=monkeypatch
    )
    assert len(gradients) == 2
    for order, weight in weights.items():
        added = _ADDED_INPUT_COLUMNS[order]
        step_zero_gradient = gradients[0][order][:, added]
        assert torch.isfinite(step_zero_gradient).all()
        assert torch.any(step_zero_gradient != 0)
        final_added = weight[:, added].detach()
        assert torch.isfinite(final_added).all()
        assert torch.any(final_added != 0)


@pytest.mark.parametrize("package", ("B10", "B11"))
def test_frozen_added_embedding_columns_stay_zero_while_raw_weights_move(
    package: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights, initial, gradients = _run_short_embedding_training(
        package, tmp_path, freeze_added=True, monkeypatch=monkeypatch
    )
    assert len(gradients) == 2
    for order, weight in weights.items():
        added = _ADDED_INPUT_COLUMNS[order]
        step_zero_gradient = gradients[0][order][:, added]
        assert torch.isfinite(step_zero_gradient).all()
        assert torch.any(step_zero_gradient != 0)
        raw = tuple(index for index in range(weight.shape[1]) if index not in added)
        assert torch.equal(weight[:, added].detach(), torch.zeros_like(weight[:, added]))
        assert torch.isfinite(weight[:, raw]).all()
        assert torch.any(weight[:, raw].detach() != initial[order][:, raw])


@pytest.mark.parametrize("package", ("B01", "B11"))
def test_jastrow_coefficients_move_in_a_real_short_training_run(
    package: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factor, gradients = _run_short_real_training(package, tmp_path, monkeypatch=monkeypatch)
    assert len(gradients) == 2
    assert torch.equal(factor.mean_coefficient.detach(), torch.zeros_like(factor.mean_coefficient)) is False
    assert torch.equal(factor.difference_coefficient.detach(), torch.zeros_like(factor.difference_coefficient)) is False
    assert all(gradient is not None and torch.isfinite(gradient).all() and torch.any(gradient != 0) for gradient in gradients)
    assert torch.isfinite(factor.mean_coefficient).all()
    assert torch.isfinite(factor.difference_coefficient).all()


def test_nontrainable_jastrow_is_inert_in_the_same_real_training_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factor, gradients = _run_short_real_training(
        "B01", tmp_path, trainable=False, monkeypatch=monkeypatch
    )
    assert gradients == []
    assert torch.equal(factor.mean_coefficient, torch.zeros_like(factor.mean_coefficient))
    assert torch.equal(factor.difference_coefficient, torch.zeros_like(factor.difference_coefficient))


def test_bounded_distance_features_match_independent_ordered_oracle() -> None:
    fragment = context_mapping.input_feature_fragment("raw plus bounded distances")
    basis = instantiate(OmegaConf.create(fragment["basis"]))
    batch = _he_batch()
    features = basis(batch)

    positions = batch.positions
    radius_squared = positions.square().sum(dim=-1)
    separation_squared = (
        positions[:, :, None, :] - positions[:, None, :, :]
    ).square().sum(dim=-1)
    expected_one_body_q = radius_squared / (1.0 + radius_squared)
    expected_pair_q = separation_squared / (1.0 + separation_squared)
    expected_one_body = torch.cat(
        [positions, batch.spins.unsqueeze(-1), expected_one_body_q.unsqueeze(-1)],
        dim=-1,
    )
    expected_pair = torch.stack(
        [
            expected_one_body_q[:, :, None].expand_as(expected_pair_q),
            expected_one_body_q[:, None, :].expand_as(expected_pair_q),
            expected_pair_q,
        ],
        dim=-1,
    )
    assert torch.allclose(features.one_body, expected_one_body)
    assert torch.allclose(features.pair, expected_pair)
    assert torch.allclose(features.pair[..., :2], features.pair.transpose(1, 2)[..., :2].flip(-1))
    assert torch.allclose(features.pair[..., 2], features.pair[..., 2].transpose(1, 2))


def test_bounded_distance_features_match_literal_coordinate_oracle() -> None:
    """Check the basis ingress against hand-evaluated, non-random coordinates."""

    fragment = context_mapping.input_feature_fragment("raw plus bounded distances")
    basis = instantiate(OmegaConf.create(fragment["basis"]))
    positions = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[1.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    batch = instantiate(
        {
            "_target_": "tpen.data.batch.ElectronBatch",
            "positions": positions,
            "nuclear_positions": torch.zeros((1, 3), dtype=torch.float64),
            "nuclear_charges": torch.tensor([2.0], dtype=torch.float64),
            "spins": torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64),
        }
    )

    features = basis(batch)
    expected_one_body = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 1.0, 1 / 2], [0.0, 2.0, 0.0, -1.0, 4 / 5]],
            [[1.0, 1.0, 0.0, 1.0, 2 / 3], [2.0, 0.0, 0.0, -1.0, 4 / 5]],
        ],
        dtype=torch.float64,
    )
    expected_pair = torch.tensor(
        [
            [
                [[1 / 2, 1 / 2, 0.0], [1 / 2, 4 / 5, 5 / 6]],
                [[4 / 5, 1 / 2, 5 / 6], [4 / 5, 4 / 5, 0.0]],
            ],
            [
                [[2 / 3, 2 / 3, 0.0], [2 / 3, 4 / 5, 2 / 3]],
                [[4 / 5, 2 / 3, 2 / 3], [4 / 5, 4 / 5, 0.0]],
            ],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(features.one_body, expected_one_body, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(features.pair, expected_pair, rtol=1e-15, atol=1e-15)


def test_bounded_distance_fragment_splices_into_real_control_and_reaches_added_columns() -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model = OmegaConf.merge(
        train.model,
        OmegaConf.create(context_mapping.input_feature_fragment("raw plus bounded distances")),
    )
    OmegaConf.resolve(train)
    model = instantiate(train.model).to(dtype=torch.float64)
    output = model(_he_batch())
    assert torch.isfinite(output.logabs).all()
    output.logabs.square().sum().backward()

    order_one = model.embedding.order_mlps["1"].layers[0].weight.grad[:, 4]
    order_two = model.embedding.order_mlps["2"].layers[0].weight.grad[:, [4, 9, 10, 11, 12]]
    assert torch.isfinite(order_one).all()
    assert torch.isfinite(order_two).all()
    assert torch.any(order_one != 0)
    assert torch.all(torch.any(order_two != 0, dim=0))


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
