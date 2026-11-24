"""Contract and mutation-sensitive tests for SR/minSR grid hyperparameter mapping.

These tests pin the exact bijection between the three SR authority factors
(``eta``, ``relative_damping``, ``squared_update_norm_bound``) and concrete
``SRPolicy`` / ``StochasticReconfigurationUpdate`` construction, including
true squared displacement-norm capping (``||Delta theta||_2^2 <= C``, never
the unsquared substitute ``||Delta theta||_2 <= C``) and loud refusal of
malformed or incompatible inputs both with Typeguard enabled and with
``typeguard.suppress_type_checks()``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import itertools
import math
from typing import Any

import numpy as np
import pytest
import torch
import typeguard
from typeguard import TypeCheckError

from tests.helpers.spring_dense_oracle import spring_trajectory
from tests.helpers.sr_dense_oracle import sr_direction
from tpen.data.batch import (
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    ParameterSlot,
    WavefunctionOutput,
)
import tpen.training as training_pkg
from tpen.training.qgt import DampingPolicy
from tpen.training.score_geometry import ScoreConventions
from tpen.training.spring import SPRINGPolicy, SPRINGUpdate
from tpen.training.sr import (
    SR_GRID_FACTOR_KEYS,
    SRGridHyperparameters,
    SRPolicy,
    StochasticReconfigurationUpdate,
    build_sr_update_from_factors,
    sr_policy_from_factors,
)
from tpen.training.update import ModelParameterBinding, ScoreUpdateInput

AUTHORITY_ETA = (0.003, 0.01, 0.03, 0.1)
AUTHORITY_RELATIVE_DAMPING = (1.0e-4, 1.0e-3, 1.0e-2, 0.1)
AUTHORITY_SQUARED_UPDATE_NORM_BOUND = (1.0e-4, 1.0e-3)
AUTHORITY_CELLS = tuple(
    {
        "eta": eta,
        "relative_damping": relative_damping,
        "squared_update_norm_bound": bound,
    }
    for eta, relative_damping, bound in itertools.product(
        AUTHORITY_ETA,
        AUTHORITY_RELATIVE_DAMPING,
        AUTHORITY_SQUARED_UPDATE_NORM_BOUND,
    )
)

DEFAULT_FACTORS: dict[str, float] = {
    "eta": 0.01,
    "relative_damping": 1.0e-3,
    "squared_update_norm_bound": 1.0e-4,
}


@contextmanager
def _typecheck_mode(suppressed: bool) -> Iterator[None]:
    """Run assertions either under active Typeguard or with runtime checks suppressed."""

    if suppressed:
        with typeguard.suppress_type_checks():
            yield
    else:
        yield


def _layout(n_parameters: int) -> ParameterLayout:
    """Build a single-slot float64 parameter layout."""

    slot = ParameterSlot(
        ordinal=0,
        shape=(n_parameters,),
        numel=n_parameters,
        dtype=torch.float64,
    )
    return ParameterLayout(slots=(slot,))


def _step_input(
    parameter: torch.nn.Parameter,
    features: torch.Tensor,
    energies: torch.Tensor,
    *,
    step: int = 0,
) -> ScoreUpdateInput:
    """Build a linear log-amplitude score packet whose exact scores equal ``features``."""

    n_samples = int(features.shape[0])
    positions = torch.zeros((n_samples, 2, 1), dtype=torch.float64)
    spins = torch.ones((n_samples, 2), dtype=torch.float64)
    batch = ElectronBatch(positions=positions, spins=spins)
    output = WavefunctionOutput(
        logabs=(features @ parameter).detach(),
        sign=torch.ones(n_samples, dtype=torch.float64),
    )
    layout = _layout(int(features.shape[1]))
    return ScoreUpdateInput(
        batch=batch,
        wavefunction=output,
        local_energy=energies,
        step=step,
        parameter_scores=MaterializedParameterLogScores(layout=layout, blocks=(features,)),
        parameter_binding=ParameterBinding(layout=layout, parameters=(parameter,)),
    )


def _problem(
    *,
    n_samples: int,
    n_parameters: int,
    seed: int,
) -> tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]:
    """Generate a deterministic parameter vector, score matrix, and local energies."""

    generator = torch.Generator().manual_seed(seed)
    parameter = torch.nn.Parameter(
        torch.randn(n_parameters, generator=generator, dtype=torch.float64)
    )
    features = torch.randn(
        (n_samples, n_parameters), generator=generator, dtype=torch.float64
    )
    energies = torch.randn(n_samples, generator=generator, dtype=torch.float64)
    return parameter, features, energies


def _scale_energies_for_target_sr_displacement(
    features: torch.Tensor,
    energies: torch.Tensor,
    *,
    eta: float,
    relative_damping: float,
    damping_minimum: float,
    target_displacement_norm: float,
) -> tuple[torch.Tensor, np.ndarray]:
    """Scale ``energies`` so the uncapped SR displacement has exact Euclidean norm.

    Because the regularized QGT and its scalar damping shift depend only on
    ``features`` while the energy residual is strictly linear in ``energies``,
    rescaling ``energies`` scales the uncapped SR direction proportionally.
    """

    oracle = sr_direction(
        features.numpy(),
        energies.numpy(),
        absolute=0.0,
        relative=relative_damping,
        minimum=damping_minimum,
    )
    uncapped_norm = eta * float(np.linalg.norm(oracle.direction))
    assert uncapped_norm > 0.0
    scale = target_displacement_norm / uncapped_norm
    scaled_energies = energies * scale
    expected_uncapped_step = (eta * scale) * oracle.direction
    assert float(np.linalg.norm(expected_uncapped_step)) == pytest.approx(
        target_displacement_norm, rel=1.0e-12
    )
    return scaled_energies, expected_uncapped_step


# --------------------------------------------------------------------------
# Public exports & 32-cell exact mapping / readback
# --------------------------------------------------------------------------


def test_training_namespace_reexports_sr_grid_mapping_api() -> None:
    """Public mapping helpers are re-exported identically from ``tpen.training``."""

    assert training_pkg.SR_GRID_FACTOR_KEYS is SR_GRID_FACTOR_KEYS
    assert training_pkg.SRGridHyperparameters is SRGridHyperparameters
    assert training_pkg.sr_policy_from_factors is sr_policy_from_factors
    assert training_pkg.build_sr_update_from_factors is build_sr_update_from_factors
    assert SR_GRID_FACTOR_KEYS == frozenset(
        {"eta", "relative_damping", "squared_update_norm_bound"}
    )


def test_all_32_authority_cells_map_with_exact_readback() -> None:
    """Every SR authority cell maps cleanly with exact readback across all surfaces."""

    assert len(AUTHORITY_CELLS) == 32

    for cell in AUTHORITY_CELLS:
        parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
        binding = ModelParameterBinding(parameters=(parameter,))

        parsed = SRGridHyperparameters.from_mapping(cell)
        assert parsed.to_factors() == cell

        policy_from_dict = sr_policy_from_factors(cell)
        policy_from_record = sr_policy_from_factors(parsed)
        assert policy_from_dict == policy_from_record

        optimizer, method = build_sr_update_from_factors(cell, model_parameters=binding)
        optimizer_rec, method_rec = build_sr_update_from_factors(
            parsed, model_parameters=binding
        )
        assert method.policy == method_rec.policy
        assert optimizer_rec.param_groups[0]["lr"] == cell["eta"]

        # Plain-SGD optimizer readback.
        assert isinstance(optimizer, torch.optim.SGD)
        assert method.optimizer is optimizer
        assert len(optimizer.param_groups) == 1
        group = optimizer.param_groups[0]
        assert group["lr"] == cell["eta"]
        assert float(group["momentum"]) == 0.0
        assert float(group["dampening"]) == 0.0
        assert float(group["weight_decay"]) == 0.0
        assert bool(group["nesterov"]) is False
        assert bool(group["maximize"]) is False

        # Policy object readback.
        assert method.policy.learning_rate == cell["eta"]
        assert method.policy.damping == DampingPolicy(
            absolute=0.0,
            relative=cell["relative_damping"],
            minimum=1.0e-12,
        )
        assert method.policy.max_update_norm is None
        assert method.policy.squared_update_norm_bound == cell["squared_update_norm_bound"]
        assert method.policy.solve_space == "auto"
        assert method.policy.rank_cutoff == 0.0
        assert method.policy.score_chunk_size is None

        # Policy fingerprint exact readback.
        assert method.policy.fingerprint() == {
            "solve_space": "auto",
            "damping": {
                "absolute": 0.0,
                "relative": cell["relative_damping"],
                "minimum": 1.0e-12,
            },
            "rank_cutoff": 0.0,
            "learning_rate": cell["eta"],
            "max_update_norm": None,
            "squared_update_norm_bound": cell["squared_update_norm_bound"],
            "score_chunk_size": None,
        }

        # Flattened description settings exact readback.
        settings = method.describe().settings
        assert settings["policy_learning_rate"] == cell["eta"]
        assert settings["policy_damping_absolute"] == 0.0
        assert settings["policy_damping_relative"] == cell["relative_damping"]
        assert settings["policy_damping_minimum"] == 1.0e-12
        assert settings["policy_max_update_norm"] is None
        assert (
            settings["policy_squared_update_norm_bound"]
            == cell["squared_update_norm_bound"]
        )
        assert settings["policy_solve_space"] == "auto"
        assert settings["policy_rank_cutoff"] == 0.0
        assert settings["policy_score_chunk_size"] is None


def test_build_sr_update_from_factors_forwards_optional_controls() -> None:
    """Explicit optional policy and update controls propagate without distortion."""

    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    binding = ModelParameterBinding(parameters=(parameter,))
    conventions = ScoreConventions(solve_dtype=torch.float64)

    optimizer, method = build_sr_update_from_factors(
        DEFAULT_FACTORS,
        model_parameters=binding,
        solve_space="sample",
        damping_minimum=0.0,
        rank_cutoff=1.0e-6,
        score_chunk_size=8,
        conventions=conventions,
        nonfinite_local_energy_policy="fail",
    )

    assert optimizer.param_groups[0]["lr"] == DEFAULT_FACTORS["eta"]
    assert method.policy.solve_space == "sample"
    assert method.policy.damping == DampingPolicy(
        absolute=0.0,
        relative=DEFAULT_FACTORS["relative_damping"],
        minimum=0.0,
    )
    assert method.policy.rank_cutoff == 1.0e-6
    assert method.policy.score_chunk_size == 8
    assert method.conventions is conventions
    assert method.nonfinite_local_energy_policy == "fail"
    settings = method.describe().settings
    assert settings["policy_solve_space"] == "sample"
    assert settings["policy_damping_minimum"] == 0.0
    assert settings["policy_rank_cutoff"] == 1.0e-6
    assert settings["policy_score_chunk_size"] == 8
    assert settings["nonfinite_local_energy_policy"] == "fail"


# --------------------------------------------------------------------------
# Decisive squared vs unsquared displacement cap semantics
# --------------------------------------------------------------------------


def test_squared_displacement_cap_rescales_proportionally_to_sqrt_bound() -> None:
    """When ``||Delta theta||_2 = 0.05`` (``2.5e-3 > 1e-4``), cap lands at ``1e-2`` not ``1e-4``."""

    bound = 1.0e-4
    eta = 0.01
    relative_damping = 1.0e-3
    damping_minimum = 1.0e-12
    factors = {
        "eta": eta,
        "relative_damping": relative_damping,
        "squared_update_norm_bound": bound,
    }

    parameter, features, raw_energies = _problem(n_samples=10, n_parameters=4, seed=41)
    energies, expected_uncapped = _scale_energies_for_target_sr_displacement(
        features,
        raw_energies,
        eta=eta,
        relative_damping=relative_damping,
        damping_minimum=damping_minimum,
        target_displacement_norm=0.05,
    )
    before = parameter.detach().clone()

    _, method = build_sr_update_from_factors(
        factors,
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
        damping_minimum=damping_minimum,
    )
    result = method.update(_step_input(parameter, features, energies))
    applied_step = (before - parameter.detach()).numpy()

    applied_norm = float(np.linalg.norm(applied_step))
    applied_sq_norm = float(np.dot(applied_step, applied_step))
    direction_cosine = float(
        np.dot(applied_step, expected_uncapped)
        / (applied_norm * float(np.linalg.norm(expected_uncapped)))
    )

    assert result.applied is True
    assert applied_sq_norm == pytest.approx(bound, rel=1.0e-12)
    assert applied_norm == pytest.approx(math.sqrt(bound), rel=1.0e-12)
    # Explicitly rule out an unsquared substitute (which would clamp norm to 1e-4, sq to 1e-8).
    assert applied_norm > 5.0 * bound
    assert direction_cosine == pytest.approx(1.0, abs=1.0e-12)
    assert method.last_telemetry is not None
    assert method.last_telemetry.trust_scale == pytest.approx(
        math.sqrt(bound) / 0.05, rel=1.0e-12
    )
    assert method.last_telemetry.applied_update_norm == pytest.approx(
        math.sqrt(bound), rel=1.0e-12
    )


def test_squared_displacement_cap_does_not_cap_below_sqrt_bound() -> None:
    """When ``||Delta theta||_2 = 5e-3`` (``2.5e-5 < 1e-4``), ``trust_scale`` stays ``1.0``.

    An unsquared ``||Delta theta||_2 <= 1e-4`` substitute would falsely cap this
    step by 50x (``trust_scale == 0.02``).
    """

    bound = 1.0e-4
    eta = 0.01
    relative_damping = 1.0e-3
    damping_minimum = 1.0e-12
    factors = {
        "eta": eta,
        "relative_damping": relative_damping,
        "squared_update_norm_bound": bound,
    }

    parameter, features, raw_energies = _problem(n_samples=10, n_parameters=4, seed=42)
    energies, expected_uncapped = _scale_energies_for_target_sr_displacement(
        features,
        raw_energies,
        eta=eta,
        relative_damping=relative_damping,
        damping_minimum=damping_minimum,
        target_displacement_norm=5.0e-3,
    )
    before = parameter.detach().clone()

    _, method = build_sr_update_from_factors(
        factors,
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
        damping_minimum=damping_minimum,
    )
    result = method.update(_step_input(parameter, features, energies))
    applied_step = (before - parameter.detach()).numpy()

    assert result.applied is True
    assert method.last_telemetry is not None
    assert method.last_telemetry.trust_scale == 1.0
    assert method.last_telemetry.applied_update_norm == pytest.approx(5.0e-3, rel=1.0e-12)
    np.testing.assert_allclose(applied_step, expected_uncapped, rtol=1.0e-12, atol=1.0e-12)


def test_sr_policy_trust_scale_boundaries_and_legacy_max_update_norm() -> None:
    """``SRPolicy.trust_scale`` handles zero, exact boundary, squared cap, and legacy cap."""

    squared_policy = sr_policy_from_factors(
        {"eta": 0.01, "relative_damping": 1.0e-3, "squared_update_norm_bound": 1.0e-4}
    )
    assert squared_policy.trust_scale(direction_norm=0.0) == 1.0
    assert squared_policy.trust_scale(direction_norm=0.5) == 1.0  # proposed = 5e-3
    assert squared_policy.trust_scale(direction_norm=1.0) == 1.0  # proposed = 1e-2 == sqrt(1e-4)
    assert squared_policy.trust_scale(direction_norm=5.0) == pytest.approx(
        0.2, rel=1.0e-12
    )  # proposed = 0.05 -> 0.01

    legacy_policy = SRPolicy(learning_rate=0.01, max_update_norm=1.0e-4)
    assert legacy_policy.trust_scale(direction_norm=0.0) == 1.0
    assert legacy_policy.trust_scale(direction_norm=0.5) == pytest.approx(
        0.02, rel=1.0e-12
    )  # legacy unsquared cap on proposed = 5e-3


def test_spring_update_honors_base_squared_update_norm_bound() -> None:
    """SPRINGUpdate delegates trust scaling to ``base.trust_scale`` and keeps history unscaled."""

    bound = 1.0e-4
    eta = 0.01
    relative_damping = 1.0e-3
    base = sr_policy_from_factors(
        {
            "eta": eta,
            "relative_damping": relative_damping,
            "squared_update_norm_bound": bound,
        },
        solve_space="sample",
        damping_minimum=0.0,
    )
    policy = SPRINGPolicy(base=base, history_decay=0.35)

    for target_norm, expect_capped in ((0.05, True), (5.0e-3, False)):
        parameter, features, raw_energies = _problem(n_samples=6, n_parameters=10, seed=43)
        oracle_raw = spring_trajectory(
            (features.numpy(),),
            (raw_energies.numpy(),),
            history_decay=0.35,
            relative=relative_damping,
        )[0]
        scale = target_norm / (eta * float(np.linalg.norm(oracle_raw.history)))
        energies = raw_energies * scale
        expected_history = oracle_raw.history * scale
        before = parameter.detach().clone()

        method = SPRINGUpdate(
            torch.optim.SGD([parameter], lr=eta),
            model_parameters=ModelParameterBinding(parameters=(parameter,)),
            policy=policy,
            conventions=ScoreConventions(),
        )
        result = method.update(_step_input(parameter, features, energies))
        applied_step = (before - parameter.detach()).numpy()

        assert result.applied is True
        assert method.last_telemetry is not None
        if expect_capped:
            assert float(np.dot(applied_step, applied_step)) == pytest.approx(
                bound, rel=1.0e-10
            )
            assert method.last_telemetry.trust_scale == pytest.approx(
                math.sqrt(bound) / target_norm, rel=1.0e-10
            )
        else:
            assert method.last_telemetry.trust_scale == 1.0
            np.testing.assert_allclose(
                applied_step, eta * expected_history, rtol=1.0e-10, atol=1.0e-10
            )
        # Canonical persisted SPRING history remains unscaled regardless of cap.
        np.testing.assert_allclose(
            method.history.detach().numpy(),
            expected_history,
            rtol=1.0e-9,
            atol=1.0e-9,
        )


@pytest.mark.parametrize(("n_samples", "n_parameters"), [(12, 4), (4, 12)])
@pytest.mark.parametrize("target_norm", [0.05, 5.0e-3], ids=["capped", "uncapped"])
def test_parameter_and_sample_routes_agree_under_mapped_factors(
    n_samples: int,
    n_parameters: int,
    target_norm: float,
) -> None:
    """Mapped ``relative_damping`` and ``squared_update_norm_bound`` agree across SR/minSR."""

    factors = {
        "eta": 0.03,
        "relative_damping": 1.0e-2,
        "squared_update_norm_bound": 1.0e-4,
    }
    parameter_a, features, raw_energies = _problem(
        n_samples=n_samples, n_parameters=n_parameters, seed=44
    )
    energies, _ = _scale_energies_for_target_sr_displacement(
        features,
        raw_energies,
        eta=factors["eta"],
        relative_damping=factors["relative_damping"],
        damping_minimum=1.0e-12,
        target_displacement_norm=target_norm,
    )
    parameter_b = torch.nn.Parameter(parameter_a.detach().clone())

    _, method_param = build_sr_update_from_factors(
        factors,
        model_parameters=ModelParameterBinding(parameters=(parameter_a,)),
        solve_space="parameter",
    )
    _, method_sample = build_sr_update_from_factors(
        factors,
        model_parameters=ModelParameterBinding(parameters=(parameter_b,)),
        solve_space="sample",
    )

    method_param.update(_step_input(parameter_a, features, energies))
    method_sample.update(_step_input(parameter_b, features, energies))

    np.testing.assert_allclose(
        parameter_a.detach().numpy(),
        parameter_b.detach().numpy(),
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    assert method_param.last_telemetry is not None
    assert method_sample.last_telemetry is not None
    assert method_param.last_telemetry.diagnostics is not None
    assert method_sample.last_telemetry.diagnostics is not None
    assert method_param.last_telemetry.diagnostics.space == "parameter"
    assert method_sample.last_telemetry.diagnostics.space == "sample"
    assert method_param.last_telemetry.diagnostics.shift == pytest.approx(
        method_sample.last_telemetry.diagnostics.shift, rel=1.0e-12
    )
    assert method_param.last_telemetry.trust_scale == pytest.approx(
        method_sample.last_telemetry.trust_scale, rel=1.0e-10
    )
    assert method_param.last_telemetry.applied_update_norm == pytest.approx(
        method_sample.last_telemetry.applied_update_norm, rel=1.0e-10
    )


def test_max_update_norm_and_squared_update_norm_bound_are_mutually_exclusive() -> None:
    """Setting both displacement caps on ``SRPolicy`` fails loudly."""

    with pytest.raises(
        ValueError, match="cannot set both max_update_norm and squared_update_norm_bound"
    ):
        SRPolicy(max_update_norm=1.0e-2, squared_update_norm_bound=1.0e-4)

    # Also verify validate() directly if a caller bypasses __post_init__.
    policy = SRPolicy(max_update_norm=1.0e-2)
    object.__setattr__(policy, "squared_update_norm_bound", 1.0e-4)
    with pytest.raises(
        ValueError, match="cannot set both max_update_norm and squared_update_norm_bound"
    ):
        policy.validate()


# --------------------------------------------------------------------------
# Loud refusal tests (both Typeguard-ON and suppressed-typecheck execution)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("suppressed", [False, True], ids=["typeguard_on", "typeguard_off"])
@pytest.mark.parametrize(
    "unknown_key",
    ["lr", "max_update_norm", "damping", "history_mu", "weight_decay", "absolute_damping"],
)
def test_unknown_factor_keys_are_refused(suppressed: bool, unknown_key: str) -> None:
    """Extra or non-isomorphic authority factor names fail loudly."""

    bad = {**DEFAULT_FACTORS, unknown_key: 1.0e-3}
    with _typecheck_mode(suppressed):
        with pytest.raises(ValueError, match="unknown keys"):
            SRGridHyperparameters.from_mapping(bad)
        with pytest.raises(ValueError, match="unknown keys"):
            sr_policy_from_factors(bad)


@pytest.mark.parametrize("suppressed", [False, True], ids=["typeguard_on", "typeguard_off"])
@pytest.mark.parametrize(
    "missing_keys",
    [
        ("eta",),
        ("relative_damping",),
        ("squared_update_norm_bound",),
        ("eta", "relative_damping", "squared_update_norm_bound"),
    ],
)
def test_missing_factor_keys_are_refused(
    suppressed: bool,
    missing_keys: tuple[str, ...],
) -> None:
    """Omitting any required SR authority factor fails loudly."""

    bad = {k: v for k, v in DEFAULT_FACTORS.items() if k not in missing_keys}
    with _typecheck_mode(suppressed):
        with pytest.raises(ValueError, match="missing keys"):
            SRGridHyperparameters.from_mapping(bad)
        with pytest.raises(ValueError, match="missing keys"):
            sr_policy_from_factors(bad)


@pytest.mark.parametrize("suppressed", [False, True], ids=["typeguard_on", "typeguard_off"])
@pytest.mark.parametrize(
    "non_mapping",
    [None, ["eta", "relative_damping", "squared_update_norm_bound"], 42, "eta=0.01"],
)
def test_non_mapping_factor_inputs_are_refused(
    suppressed: bool,
    non_mapping: Any,
) -> None:
    """Non-mapping factor containers fail loudly with and without Typeguard."""

    expected_exc: type[Exception] | tuple[type[Exception], ...] = (
        TypeError if suppressed else (TypeError, TypeCheckError)
    )
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    binding = ModelParameterBinding(parameters=(parameter,))

    with _typecheck_mode(suppressed):
        with pytest.raises(expected_exc):
            SRGridHyperparameters.from_mapping(non_mapping)
        with pytest.raises(expected_exc):
            sr_policy_from_factors(non_mapping)
        with pytest.raises(expected_exc):
            build_sr_update_from_factors(non_mapping, model_parameters=binding)


@pytest.mark.parametrize("suppressed", [False, True], ids=["typeguard_on", "typeguard_off"])
@pytest.mark.parametrize("field", ["eta", "relative_damping", "squared_update_norm_bound"])
@pytest.mark.parametrize("bad_value", [True, False, None, "0.01"])
def test_bool_none_and_non_real_factor_values_are_refused(
    suppressed: bool,
    field: str,
    bad_value: Any,
) -> None:
    """Booleans, ``None``, and strings are rejected for every authority factor."""

    bad_mapping = {**DEFAULT_FACTORS, field: bad_value}
    expected_ctor_exc: type[Exception] | tuple[type[Exception], ...] = (
        TypeError if suppressed else (TypeError, TypeCheckError)
    )
    with _typecheck_mode(suppressed):
        # from_mapping validates values explicitly, so TypeError is raised in both modes.
        with pytest.raises(TypeError, match="real number"):
            SRGridHyperparameters.from_mapping(bad_mapping)
        with pytest.raises(TypeError, match="real number"):
            sr_policy_from_factors(bad_mapping)
        with pytest.raises(expected_ctor_exc):
            SRGridHyperparameters(**bad_mapping)


@pytest.mark.parametrize("suppressed", [False, True], ids=["typeguard_on", "typeguard_off"])
@pytest.mark.parametrize("field", ["eta", "relative_damping", "squared_update_norm_bound"])
@pytest.mark.parametrize(
    "bad_value",
    [0.0, -1.0e-4, float("nan"), float("inf"), float("-inf")],
)
def test_non_positive_or_non_finite_factor_values_are_refused(
    suppressed: bool,
    field: str,
    bad_value: float,
) -> None:
    """Zero, negative, NaN, and infinite values are refused for every factor."""

    bad_mapping = {**DEFAULT_FACTORS, field: bad_value}
    with _typecheck_mode(suppressed):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            SRGridHyperparameters.from_mapping(bad_mapping)
        with pytest.raises(ValueError, match="finite and strictly positive"):
            SRGridHyperparameters(**bad_mapping)
        with pytest.raises(ValueError, match="finite and strictly positive"):
            sr_policy_from_factors(bad_mapping)


@pytest.mark.parametrize("suppressed", [False, True], ids=["typeguard_on", "typeguard_off"])
@pytest.mark.parametrize("bad_bool", [True, False])
@pytest.mark.parametrize(
    "policy_field",
    ["learning_rate", "max_update_norm", "squared_update_norm_bound", "rank_cutoff"],
)
def test_sr_policy_rejects_booleans_across_numeric_fields(
    suppressed: bool,
    bad_bool: bool,
    policy_field: str,
) -> None:
    """``SRPolicy`` rejects booleans rather than coercing ``True``/``False`` to ``1.0``/``0.0``."""

    expected_exc: type[Exception] | tuple[type[Exception], ...] = (
        TypeError if suppressed else (TypeError, TypeCheckError)
    )
    with _typecheck_mode(suppressed):
        with pytest.raises(expected_exc):
            SRPolicy(**{policy_field: bad_bool})


@pytest.mark.parametrize("suppressed", [False, True], ids=["typeguard_on", "typeguard_off"])
@pytest.mark.parametrize(
    "bad_bound",
    [0.0, -1.0e-4, float("nan"), float("inf"), float("-inf")],
)
def test_sr_policy_rejects_invalid_squared_update_norm_bound(
    suppressed: bool,
    bad_bound: float,
) -> None:
    """``SRPolicy.squared_update_norm_bound`` must be finite and strictly positive."""

    with _typecheck_mode(suppressed):
        with pytest.raises(
            ValueError, match="squared_update_norm_bound must be finite and positive"
        ):
            SRPolicy(squared_update_norm_bound=bad_bound)


@pytest.mark.parametrize("suppressed", [False, True], ids=["typeguard_on", "typeguard_off"])
@pytest.mark.parametrize("bad_space", ["invalid", "minsr", 123])
def test_invalid_solve_space_is_refused(suppressed: bool, bad_space: Any) -> None:
    """Invalid ``solve_space`` fails loudly under both Typeguard-ON and suppressed checks."""

    expected_exc: type[Exception] | tuple[type[Exception], ...] = (
        ValueError if suppressed else (ValueError, TypeCheckError)
    )
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    binding = ModelParameterBinding(parameters=(parameter,))

    with _typecheck_mode(suppressed):
        with pytest.raises(expected_exc):
            sr_policy_from_factors(DEFAULT_FACTORS, solve_space=bad_space)
        with pytest.raises(expected_exc):
            build_sr_update_from_factors(
                DEFAULT_FACTORS,
                model_parameters=binding,
                solve_space=bad_space,
            )


@pytest.mark.parametrize("suppressed", [False, True], ids=["typeguard_on", "typeguard_off"])
@pytest.mark.parametrize(
    "bad_minimum",
    [-1.0e-12, float("nan"), float("inf"), float("-inf")],
)
def test_negative_or_non_finite_damping_minimum_is_refused(
    suppressed: bool,
    bad_minimum: float,
) -> None:
    """Negative or non-finite ``damping_minimum`` fails loudly."""

    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    binding = ModelParameterBinding(parameters=(parameter,))

    with _typecheck_mode(suppressed):
        with pytest.raises(ValueError, match="damping_minimum must be finite and non-negative"):
            sr_policy_from_factors(DEFAULT_FACTORS, damping_minimum=bad_minimum)
        with pytest.raises(ValueError, match="damping_minimum must be finite and non-negative"):
            build_sr_update_from_factors(
                DEFAULT_FACTORS,
                model_parameters=binding,
                damping_minimum=bad_minimum,
            )


@pytest.mark.parametrize("suppressed", [False, True], ids=["typeguard_on", "typeguard_off"])
def test_bool_damping_minimum_and_bad_model_parameters_are_refused(
    suppressed: bool,
) -> None:
    """Boolean ``damping_minimum`` and non-binding ``model_parameters`` fail loudly."""

    expected_exc: type[Exception] | tuple[type[Exception], ...] = (
        TypeError if suppressed else (TypeError, TypeCheckError)
    )
    with _typecheck_mode(suppressed):
        with pytest.raises(expected_exc):
            sr_policy_from_factors(DEFAULT_FACTORS, damping_minimum=True)
        with pytest.raises(expected_exc):
            build_sr_update_from_factors(
                DEFAULT_FACTORS,
                model_parameters="not-a-binding",  # type: ignore[arg-type]
            )


@pytest.mark.parametrize("suppressed", [False, True], ids=["typeguard_on", "typeguard_off"])
def test_trust_scale_refuses_bool_negative_and_non_finite_direction_norm(
    suppressed: bool,
) -> None:
    """``SRPolicy.trust_scale`` validates ``direction_norm`` loudly."""

    policy = sr_policy_from_factors(DEFAULT_FACTORS)
    expected_bool_exc: type[Exception] | tuple[type[Exception], ...] = (
        TypeError if suppressed else (TypeError, TypeCheckError)
    )
    with _typecheck_mode(suppressed):
        with pytest.raises(expected_bool_exc):
            policy.trust_scale(direction_norm=True)
        for bad_norm in (-1.0e-3, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="direction_norm must be finite and non-negative"):
                policy.trust_scale(direction_norm=bad_norm)
