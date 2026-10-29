"""Unit tests for TPEN's pointwise Gaussian and Softplus-beta activations.

Pins the D3 elementwise activation form ``f(x) = exp(-x**2 / (2 sigma**2))``
with a fixed, non-trainable ``sigma``: closed-form values, ``f(0) == 1``,
evenness, boundedness, dtype/device preservation, finite gradients, and the
fact that an elementwise ``Gamma`` leaves :class:`tpen.nn.EquivariantMixing`
and :class:`tpen.nn.PathAggregation` permutation-equivariant under the
production checkers in :mod:`tpen.equivariance.checks`.

Also records the masked-entry characterization required by the downstream
activation scan: both Gaussian and Softplus beta2 are ``Gamma(0) != 0``
activations, so they write an invariant constant onto the non-distinct tuple
entries that mixing never writes (``EquivariantMixing`` docstring; ``Embedding``
deliberately zeroes those entries). The tests measure that contamination and
check whether it reaches the wavefunction through
:class:`tpen.nn.readout.PfaffianReadout`.

Logged times in this suite use UTC per repository convention.
"""

from __future__ import annotations

import math

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.batch import ElectronBatch
from tpen.data.indices import no_repeated_particle_mask
from tpen.data.real import Feature, Interaction, zero_block
from tpen.equivariance import EquivariantMap
from tpen.equivariance.checks import (
    FullModelEquivarianceChecker,
    TraceEquivarianceChecker,
)
from tpen.nn import (
    EquivariantMixing,
    GaussianActivation,
    PathAggregation,
    SoftplusBeta,
    TorchInitializer,
)
from tpen.nn.readout import PfaffianReadout
from tests.helpers.equivariance import assert_equivariant_all

_DTYPE = torch.float64

# Sigmas spanning below/at/above one, so any implementation that hard-codes the
# width (or folds it in with the wrong power) disagrees with the reference.
_SIGMAS = (0.25, 0.5, 1.0, 2.0, 3.75)

# Deliberately asymmetric around zero and free of the fixed points x in {0, 1}
# where dropping the square would be invisible.
_SAMPLE_POINTS = (-4.0, -2.5, -1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0, 2.5, 4.0)


def _reference(x: float, sigma: float) -> float:
    """Closed-form scalar Gaussian bump, written independently of the module."""

    return math.exp(-(x**2) / (2.0 * sigma**2))


class _State:
    """Minimal TrainerState stand-in accepted by the runtime checkers."""

    def __init__(self, model, batch, step: int = 1) -> None:
        self.model = model
        self.batch = batch
        self.step = step


def _random_feature(
    n_particles: int,
    channels: int,
    max_order: int,
    *,
    seed: int,
    batch: int = 2,
) -> Feature:
    generator = torch.Generator().manual_seed(seed)
    blocks: list[torch.Tensor] = [zero_block(batch_size=batch, dtype=_DTYPE)]
    for order in range(1, max_order + 1):
        shape = (batch, channels, *((n_particles,) * order))
        blocks.append(torch.randn(shape, generator=generator, dtype=_DTYPE))
    return Feature(blocks)


def _mixing(activation, *, max_order: int = 2, channels: int = 2) -> EquivariantMixing:
    return EquivariantMixing(
        max_order=max_order,
        channels=channels,
        implementation="slow",
        activation=activation,
    ).to(dtype=_DTYPE)


_PATHS_BY_ORDER = {1: 3, 2: 4}


def _aggregation(activation, *, seed: int = 53) -> PathAggregation:
    return PathAggregation(
        max_order=2,
        channels=2,
        path_counts_by_order=_PATHS_BY_ORDER,
        activation=activation,
        initializer=TorchInitializer(seed=seed),
    ).to(dtype=_DTYPE)


def _chained_aggregation(activation, *, seed: int = 61) -> PathAggregation:
    """Aggregation whose path counts come from the same canonical metadata as mixing."""

    return PathAggregation(
        max_order=2,
        channels=2,
        activation=activation,
        initializer=TorchInitializer(seed=seed),
    ).to(dtype=_DTYPE)


def _masked_interaction(
    n_particles: int, *, seed: int, channels: int = 2, batch: int = 2
) -> Interaction:
    """Random interaction with non-distinct tuple entries zeroed, as mixing leaves them."""

    generator = torch.Generator().manual_seed(seed)
    max_paths = max(_PATHS_BY_ORDER.values())
    blocks: list[torch.Tensor] = [
        zero_block(batch_size=batch, paths=max_paths, dtype=_DTYPE)
    ]
    for order, paths in sorted(_PATHS_BY_ORDER.items()):
        shape = (batch, channels, paths, *((n_particles,) * order))
        block = torch.randn(shape, generator=generator, dtype=_DTYPE)
        mask = no_repeated_particle_mask(n_particles, order).reshape(
            1, 1, 1, *((n_particles,) * order)
        )
        blocks.append(block * mask.to(dtype=_DTYPE))
    return Interaction(blocks)


def _zero_masked_entries(block: torch.Tensor, n_particles: int, order: int, *, path_axis: bool) -> torch.Tensor:
    """Re-zero the non-distinct tuple entries of one real block."""

    leading = (1, 1, 1) if path_axis else (1, 1)
    mask = no_repeated_particle_mask(n_particles, order).reshape(
        *leading, *((n_particles,) * order)
    )
    return block * mask.to(dtype=block.dtype, device=block.device)


# --------------------------------------------------------------------------
# Closed-form value contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", _SIGMAS)
def test_matches_closed_form_at_several_sigmas(sigma: float) -> None:
    activation = GaussianActivation(sigma=sigma)
    x = torch.tensor(_SAMPLE_POINTS, dtype=_DTYPE)

    actual = activation(x)

    expected = torch.tensor([_reference(value, sigma) for value in _SAMPLE_POINTS], dtype=_DTYPE)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=1e-15)


def test_default_sigma_is_one() -> None:
    # The default must be the unit-width bump, not merely "some" width.
    default = GaussianActivation()
    explicit = GaussianActivation(sigma=1.0)
    x = torch.tensor(_SAMPLE_POINTS, dtype=_DTYPE)

    assert default.sigma == 1.0
    torch.testing.assert_close(default(x), explicit(x), atol=0.0, rtol=0.0)


def test_value_at_zero_is_exactly_one() -> None:
    # Gamma(0) == 1 exactly (not approximately): this is the property the
    # masked-entry characterization below depends on, and it must hold for
    # every width because the exponent is exactly zero there.
    for sigma in _SIGMAS:
        activation = GaussianActivation(sigma=sigma)
        at_zero = activation(torch.zeros(3, 4, dtype=_DTYPE))
        assert torch.equal(at_zero, torch.ones(3, 4, dtype=_DTYPE)), f"f(0) != 1 at sigma={sigma}"


@pytest.mark.parametrize("sigma", _SIGMAS)
def test_is_even(sigma: float) -> None:
    activation = GaussianActivation(sigma=sigma)
    x = torch.tensor(_SAMPLE_POINTS, dtype=_DTYPE)

    torch.testing.assert_close(activation(x), activation(-x), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("sigma", _SIGMAS)
def test_large_inputs_stay_finite_and_in_unit_interval(sigma: float) -> None:
    # Representable-but-large magnitudes: still strictly positive in float64
    # (exp underflows only past |x| / sigma ~ 38.6), always <= 1.
    magnitudes = torch.tensor([5.0, 10.0, 20.0, 30.0], dtype=_DTYPE) * sigma
    x = torch.cat([-magnitudes, magnitudes])

    values = GaussianActivation(sigma=sigma)(x)

    assert torch.isfinite(values).all(), f"non-finite output at sigma={sigma}: {values}"
    assert (values > 0.0).all(), f"output left (0, 1] at sigma={sigma}: {values}"
    assert (values <= 1.0).all(), f"output exceeded 1 at sigma={sigma}: {values}"


def test_extreme_inputs_underflow_to_zero_without_overflowing() -> None:
    # Past the exponent range exp underflows to exactly 0.0. The documented
    # deviation from the open interval (0, 1] is underflow, never inf or nan.
    x = torch.tensor([-1.0e100, -1.0e3, 1.0e3, 1.0e100], dtype=_DTYPE)

    values = GaussianActivation()(x)

    assert torch.isfinite(values).all(), f"extreme inputs produced {values}"
    assert (values >= 0.0).all() and (values <= 1.0).all(), f"extreme inputs produced {values}"


# --------------------------------------------------------------------------
# Tensor-contract preservation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_preserves_shape_dtype_and_device(dtype: torch.dtype) -> None:
    activation = GaussianActivation(sigma=1.5)
    x = torch.randn(2, 3, 4, 4, dtype=dtype)

    values = activation(x)

    assert values.shape == x.shape
    assert values.dtype == x.dtype, f"dtype changed {x.dtype} -> {values.dtype}"
    assert values.device == x.device, f"device changed {x.device} -> {values.device}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_preserves_cuda_device() -> None:
    x = torch.randn(2, 3, dtype=_DTYPE, device="cuda")

    values = GaussianActivation()(x)

    assert values.device == x.device
    assert values.dtype == x.dtype


def test_gradient_is_finite_everywhere_including_zero() -> None:
    x = torch.tensor(
        [-1.0e100, -50.0, -4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 50.0, 1.0e100],
        dtype=_DTYPE,
        requires_grad=True,
    )

    GaussianActivation(sigma=1.5)(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), f"non-finite gradient: {x.grad}"


def test_gradient_at_zero_is_zero_and_matches_closed_form() -> None:
    # f'(x) = -x / sigma^2 * f(x); the stationary point at x == 0 is the
    # signature of the squared argument.
    sigma = 0.75
    points = [-1.3, -0.4, 0.0, 0.4, 1.3]
    x = torch.tensor(points, dtype=_DTYPE, requires_grad=True)

    GaussianActivation(sigma=sigma)(x).sum().backward()

    expected = torch.tensor(
        [-value / sigma**2 * _reference(value, sigma) for value in points], dtype=_DTYPE
    )
    torch.testing.assert_close(x.grad, expected, atol=0.0, rtol=1e-13)
    assert x.grad[points.index(0.0)].item() == 0.0


# --------------------------------------------------------------------------
# Module contract: fixed sigma, plain nn.Module, exported and configurable
# --------------------------------------------------------------------------


def test_sigma_is_fixed_and_module_has_no_learnable_state() -> None:
    activation = GaussianActivation(sigma=2.0)

    assert list(activation.parameters()) == []
    assert list(activation.buffers()) == []
    assert isinstance(activation.sigma, float)


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_non_positive_or_non_finite_sigma(sigma: float) -> None:
    with pytest.raises(ValueError, match="sigma"):
        GaussianActivation(sigma=sigma)


def test_is_a_plain_module_not_an_equivariant_map() -> None:
    # D2/D3: the activation is a pointwise function owned by the equivariant
    # stage; it carries no typed real-state contract of its own.
    activation = GaussianActivation()

    assert isinstance(activation, torch.nn.Module)
    assert not isinstance(activation, EquivariantMap)
    assert not hasattr(activation, "forward_impl")


def test_is_exported_from_the_nn_namespace() -> None:
    import tpen.nn as tpen_nn
    from tpen.nn.activation import GaussianActivation as Direct

    assert tpen_nn.GaussianActivation is Direct
    assert "GaussianActivation" in tpen_nn.__all__


def test_is_hydra_instantiable() -> None:
    cfg = OmegaConf.create({"_target_": "tpen.nn.GaussianActivation", "sigma": 0.5})

    activation = instantiate(cfg)

    assert isinstance(activation, GaussianActivation)
    assert activation.sigma == 0.5
    torch.testing.assert_close(
        activation(torch.tensor([0.5], dtype=_DTYPE)),
        torch.tensor([_reference(0.5, 0.5)], dtype=_DTYPE),
        atol=0.0,
        rtol=1e-15,
    )


def test_is_a_new_activation_and_not_the_squared_radius_decay_gate() -> None:
    # GaussianDecayGate is exp(-x / 2 sigma^2) on non-negative squared radii and
    # diverges for x < 0; this activation must not be that function.
    from tpen.nn import GaussianDecayGate

    x = torch.tensor([-3.0, -1.0, 2.0], dtype=_DTYPE)
    activation_values = GaussianActivation(sigma=1.0)(x)
    gate_values = GaussianDecayGate(sigma=1.0)(x)

    assert not torch.allclose(activation_values, gate_values)
    assert torch.isfinite(activation_values).all()


# --------------------------------------------------------------------------
# Stable Softplus-beta contract (helium-importance design section 2.4)
# --------------------------------------------------------------------------


def _softplus_oracle(
    x: torch.Tensor, beta: float = 2.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent closed-form value, first derivative, and second derivative."""

    scaled = beta * x
    sigmoid = torch.sigmoid(scaled)
    value = (torch.maximum(scaled, torch.zeros_like(scaled)) + torch.log1p(
        torch.exp(-torch.abs(scaled))
    )) / beta
    first = sigmoid
    second = beta * sigmoid * torch.sigmoid(-scaled)
    return value, first, second


def _assert_softplus_contract(activation) -> None:
    points = torch.tensor(
        [-1000.0, -400.0, -100.0, -20.0, -11.0, 0.0, 11.0, 20.0, 100.0, 1000.0],
        dtype=_DTYPE,
    )
    x = points.clone().requires_grad_()
    values = activation(x)
    first = torch.autograd.grad(values.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    expected_value, expected_first, expected_second = _softplus_oracle(points)
    torch.testing.assert_close(values, expected_value, rtol=1e-13, atol=0.0)
    torch.testing.assert_close(first, expected_first, rtol=1e-12, atol=1e-300)
    torch.testing.assert_close(second, expected_second, rtol=1e-12, atol=1e-300)
    assert torch.isfinite(values).all()
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_softplus_beta_matches_closed_form_value_and_derivatives_at_extremes() -> None:
    _assert_softplus_contract(SoftplusBeta())


def test_softplus_oracle_second_derivative_stays_positive_in_finite_range() -> None:
    points = torch.tensor([-100.0, -20.0, 0.0, 20.0, 100.0], dtype=_DTYPE)
    _, _, second = _softplus_oracle(points)
    assert (second[points.abs() <= 100.0] > 0.0).all()


def test_softplus_beta_acceptance_arm_catches_library_mutant() -> None:
    _assert_softplus_contract(SoftplusBeta())
    with pytest.raises(AssertionError):
        _assert_softplus_contract(torch.nn.Softplus(beta=2.0))


def test_softplus_beta_second_derivative_is_finite_past_exp_overflow() -> None:
    x = torch.tensor([-400.0, -1000.0], dtype=_DTYPE, requires_grad=True)
    first = torch.autograd.grad(SoftplusBeta()(x).sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    assert torch.isfinite(second).all()
    torch.testing.assert_close(second, torch.zeros_like(second), rtol=0.0, atol=1e-300)


def test_softplus_beta_value_at_zero_is_log_two_over_beta() -> None:
    actual = SoftplusBeta()(torch.zeros(1, dtype=_DTYPE))
    expected = torch.full_like(actual, math.log(2.0) / 2.0)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_softplus_beta_threshold_regime_catches_torch_linear_mutant() -> None:
    x = torch.tensor([11.0], dtype=_DTYPE, requires_grad=True)
    library = torch.nn.Softplus(beta=2.0)(x)
    library_first = torch.autograd.grad(library.sum(), x, create_graph=True)[0]
    library_second = torch.autograd.grad(library_first.sum(), x)[0]
    _, _, expected_second = _softplus_oracle(x.detach())
    assert not torch.allclose(library_second, expected_second, rtol=1e-6, atol=1e-12)
    assert library_second.item() == 0.0
    assert expected_second.item() > 0.0


def test_softplus_beta_rejects_invalid_beta_and_is_exported() -> None:
    with pytest.raises(ValueError, match="beta"):
        SoftplusBeta(beta=0.0)
    with pytest.raises(ValueError, match="beta"):
        SoftplusBeta(beta=float("inf"))
    import tpen.nn as tpen_nn

    assert tpen_nn.SoftplusBeta is SoftplusBeta
    assert "SoftplusBeta" in tpen_nn.__all__


def test_softplus_beta_gradcheck_and_gradgradcheck() -> None:
    x = torch.tensor([-1.25, -0.125, 0.375, 1.5], dtype=_DTYPE, requires_grad=True)
    activation = SoftplusBeta().to(dtype=_DTYPE)
    assert torch.autograd.gradcheck(activation, (x,), eps=1e-6, atol=1e-8, rtol=1e-6)
    assert torch.autograd.gradgradcheck(activation, (x,), eps=1e-6, atol=1e-8, rtol=1e-6)


def test_naive_softplus_control_overflows_on_the_same_grid() -> None:
    points = torch.tensor([-1000.0, 1000.0], dtype=_DTYPE)
    naive = torch.log1p(torch.exp(2.0 * points)) / 2.0
    assert not torch.isfinite(naive).all()


# --------------------------------------------------------------------------
# Equivariance: elementwise Gamma must not break the owning stages
# --------------------------------------------------------------------------


def test_equivariant_mixing_with_gaussian_activation_passes_full_model_checker() -> None:
    mixing = _mixing(GaussianActivation(sigma=0.8))
    feature = _random_feature(3, channels=2, max_order=2, seed=101)

    result = FullModelEquivarianceChecker(atol=1e-12, rtol=1e-12).run(_State(mixing, feature))

    assert result.n_comparisons > 0, "checker compared nothing; the pass is vacuous"
    assert result.passed, result.failures


def test_path_aggregation_with_gaussian_activation_passes_full_model_checker() -> None:
    aggregation = _aggregation(GaussianActivation(sigma=0.8))
    interaction = _masked_interaction(3, seed=103)

    result = FullModelEquivarianceChecker(atol=1e-12, rtol=1e-12).run(
        _State(aggregation, interaction)
    )

    assert result.n_comparisons > 0, "checker compared nothing; the pass is vacuous"
    assert result.passed, result.failures


@pytest.mark.parametrize("stage", ["mixing", "aggregation"])
def test_stages_with_gaussian_activation_pass_trace_checker(stage: str) -> None:
    if stage == "mixing":
        module = _mixing(GaussianActivation(sigma=1.25))
        inputs = _random_feature(3, channels=2, max_order=2, seed=107)
    else:
        module = _aggregation(GaussianActivation(sigma=1.25))
        inputs = _masked_interaction(3, seed=109)

    result = TraceEquivarianceChecker(atol=1e-12, rtol=1e-12, compare_output=True).run(
        _State(module, inputs)
    )

    assert result.n_comparisons > 0, "checker compared nothing; the pass is vacuous"
    assert result.passed, result.failures


@pytest.mark.parametrize("n_particles", [2, 3, 4])
def test_stages_with_gaussian_activation_are_equivariant_for_all_permutations(
    n_particles: int,
) -> None:
    mixing = _mixing(GaussianActivation(sigma=0.6))
    aggregation = _aggregation(GaussianActivation(sigma=0.6), seed=59)

    assert_equivariant_all(
        mixing, _random_feature(n_particles, channels=2, max_order=2, seed=113),
        atol=1e-12, rtol=1e-12,
    )
    assert_equivariant_all(
        aggregation, _masked_interaction(n_particles, seed=127), atol=1e-12, rtol=1e-12
    )


# --------------------------------------------------------------------------
# Masked-entry characterization: nonzero Gamma(0) writes onto non-distinct tuples
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("activation", "gamma_zero"),
    [(GaussianActivation(), 1.0), (SoftplusBeta(), math.log(2.0) / 2.0)],
)
def test_mixing_activation_writes_gamma_zero_onto_non_distinct_entries(
    activation, gamma_zero: float
) -> None:
    # EquivariantMixing applies Gamma to the full block, and never writes the
    # non-distinct tuple entries, so they enter Gamma as exact zeros. The
    # nonzero f(0) value is written onto every order-2 diagonal entry.
    n_particles = 3
    activated = _mixing(activation)(
        _random_feature(n_particles, channels=2, max_order=2, seed=131)
    )

    diagonal = torch.diagonal(activated.blocks[2], dim1=-2, dim2=-1)

    torch.testing.assert_close(
        diagonal, torch.full_like(diagonal, gamma_zero), atol=0.0, rtol=0.0
    )


def test_mixing_masked_contamination_magnitude_is_exactly_one() -> None:
    # Magnitude of the block contamination relative to an explicitly re-zeroed
    # forward: exactly Gamma(0) = 1 on every masked entry, zero elsewhere.
    n_particles = 3
    activated = _mixing(GaussianActivation())(
        _random_feature(n_particles, channels=2, max_order=2, seed=137)
    )
    block = activated.blocks[2]
    rezeroed = _zero_masked_entries(block, n_particles, 2, path_axis=True)

    difference = (block - rezeroed).abs()

    assert difference.max().item() == 1.0
    # Only the masked entries differ, and every one of them differs by 1.
    n_masked = int((~no_repeated_particle_mask(n_particles, 2).bool()).sum().item())
    assert int((difference > 0).sum().item()) == block.shape[0] * block.shape[1] * block.shape[2] * n_masked


@pytest.mark.parametrize(
    ("activation", "gamma_zero"),
    [(GaussianActivation(), 1.0), (SoftplusBeta(), math.log(2.0) / 2.0)],
)
def test_path_aggregation_activation_writes_gamma_zero_onto_non_distinct_entries(
    activation, gamma_zero: float
) -> None:
    # Same statement one stage later: the path contraction of masked-zero input
    # entries is zero, so the nonzero Gamma(0) lands on them again.
    n_particles = 3
    aggregated = _aggregation(activation)(_masked_interaction(n_particles, seed=139))
    block = aggregated.blocks[2]
    rezeroed = _zero_masked_entries(block, n_particles, 2, path_axis=False)

    diagonal = torch.diagonal(block, dim1=-2, dim2=-1)
    torch.testing.assert_close(
        diagonal, torch.full_like(diagonal, gamma_zero), atol=0.0, rtol=0.0
    )
    assert (block - rezeroed).abs().max().item() == gamma_zero


@pytest.mark.parametrize("n_particles", [2, 3])
@pytest.mark.parametrize(
    ("activation", "gamma_zero"),
    [(GaussianActivation(sigma=0.9), 1.0), (SoftplusBeta(), math.log(2.0) / 2.0)],
)
def test_masked_contamination_does_not_reach_the_pfaffian_wavefunction(
    n_particles: int, activation, gamma_zero: float
) -> None:
    # The readout antisymmetrizes K = 0.5 * (pair - pair^T), which annihilates
    # the order-2 diagonal exactly, so the contamination measured above cannot
    # change logabs or sign. This is the null the downstream activation scan
    # relies on.
    mixing = _mixing(activation)
    aggregation = _chained_aggregation(activation, seed=61)
    feature = _random_feature(n_particles, channels=2, max_order=2, seed=149, batch=2)

    update = aggregation(mixing(feature))
    contaminated = Feature(list(update.blocks))
    clean_blocks = list(update.blocks)
    clean_blocks[2] = _zero_masked_entries(clean_blocks[2], n_particles, 2, path_axis=False)
    clean = Feature(clean_blocks)

    # The contamination really is present in the readout input. After the full
    # mixing -> aggregation chain the masked entries are no longer Gamma(0)
    # itself: mixing writes Gamma(0) there, the path contraction turns that
    # into sum_p U[c, p], and the aggregation Gamma maps it to a channel-wise
    # invariant constant. It is still an invariant constant, which is why it
    # cannot break equivariance -- and it is still nonzero.
    difference = (contaminated.blocks[2] - clean.blocks[2]).abs()
    assert difference.max().item() > 0.0
    # It sits exactly on the masked entries and nowhere else.
    assert torch.equal(
        difference > 0.0,
        (~no_repeated_particle_mask(n_particles, 2)).reshape(1, 1, n_particles, n_particles)
        .expand_as(difference),
    )
    # And it is constant over the tuple indices, per batch element and channel.
    masked_values = torch.diagonal(contaminated.blocks[2], dim1=-2, dim2=-1)
    assert torch.equal(masked_values, masked_values[..., :1].expand_as(masked_values))

    readout = PfaffianReadout(channels=2).to(dtype=_DTYPE)
    batch = ElectronBatch(positions=torch.zeros(2, n_particles, 1, dtype=_DTYPE))
    contaminated_output = readout(contaminated, batch)
    clean_output = readout(clean, batch)

    # Mechanism: the skew kernel has an exactly zero diagonal either way.
    kernel = contaminated_output.aux["K"]
    diagonal = torch.diagonal(kernel, dim1=-2, dim2=-1)
    assert torch.equal(diagonal, torch.zeros_like(diagonal))

    # Consequence: the wavefunction is bit-for-bit unchanged.
    assert torch.equal(contaminated_output.logabs, clean_output.logabs)
    assert torch.equal(contaminated_output.sign, clean_output.sign)
    assert torch.equal(contaminated_output.aux["pfaffian"], clean_output.aux["pfaffian"])

    # Positive control. The null above is doubly protected -- the
    # antisymmetrization zeroes the diagonal, and the Pfaffian expansion reads
    # only off-diagonal entries -- so no single change to the readout can make
    # a diagonal perturbation visible. Instead, show that a perturbation of the
    # same magnitude on an off-diagonal entry does move the wavefunction, which
    # is what makes the equalities above a measurement rather than a tautology.
    control_blocks = list(clean_blocks)
    control_pair = clean_blocks[2].clone()
    control_pair[:, :, 0, 1] += 1.0
    control_blocks[2] = control_pair
    control_output = readout(Feature(control_blocks), batch)

    assert not torch.equal(control_output.aux["pfaffian"], clean_output.aux["pfaffian"])
