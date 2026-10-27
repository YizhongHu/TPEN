"""Tests for the literal HI context-to-Hydra mapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path
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
    assert composed.embedding.spatial_dim == "${system.spatial_dim}"
    assert composed.embedding.initializer.seed == "${runtime.seed}"
    assert composed.factors[1].atoms == "${atoms}"


@pytest.mark.parametrize("package", ("B00", "B10", "B01", "B11"))
def test_spliced_package_preserves_all_nonpackage_config_values(package: str) -> None:
    control = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    expected = OmegaConf.to_container(control, resolve=True)
    cfg = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    cfg.model = OmegaConf.merge(
        cfg.model,
        context_mapping.compose_package(cfg.model, package),
    )
    OmegaConf.resolve(cfg)
    actual = OmegaConf.to_container(cfg, resolve=True)
    assert {key: actual[key] for key in actual if key != "model"} == {
        key: expected[key] for key in expected if key != "model"
    }
    for key in expected["model"]:
        if key not in {"basis", "embedding", "factors"}:
            assert actual["model"][key] == expected["model"][key]


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
    train.model = OmegaConf.create(context_mapping.compose_package(train.model, package))
    _validate_through_hydra(train)


@pytest.mark.parametrize("package", ("B01", "B11"))
def test_closed_hi_schema_refuses_a_frozen_two_coefficient_jastrow(package: str) -> None:
    train = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    train.model = OmegaConf.create(context_mapping.compose_package(train.model, package))
    train.model.factors[-1].trainable = False
    with pytest.raises(Exception, match="undeclared-trainability"):
        _validate_through_hydra(train)


@pytest.mark.parametrize(("jastrow_package", "control_package"), [("B01", "B00"), ("B11", "B10")])
def test_zero_jastrow_is_exactly_the_corresponding_control(
    jastrow_package: str, control_package: str
) -> None:
    mapped = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    mapped.model = OmegaConf.merge(
        mapped.model,
        context_mapping.compose_package(mapped.model, jastrow_package),
    )
    control = OmegaConf.load(Path(__file__).with_name("configs") / "train.yaml")
    control.model = OmegaConf.merge(
        control.model,
        context_mapping.compose_package(control.model, control_package),
    )
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
    train.model = OmegaConf.merge(
        train.model,
        context_mapping.compose_package(train.model, package),
    )
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
    train.model = OmegaConf.create(context_mapping.compose_package(train.model, package))
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
