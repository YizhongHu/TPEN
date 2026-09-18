"""Acceptance tests for the native SPRING projected-history update."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from tests.helpers.spring_dense_oracle import spring_trajectory
from tpen.data.batch import (
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    ParameterSlot,
    WavefunctionOutput,
)
from tpen.training import spring as spring_module
from tpen.training.qgt import DampingPolicy, SolveDiagnostics
from tpen.training.score_geometry import ScoreConventions
from tpen.training.sr import SRPolicy, StochasticReconfigurationUpdate
from tpen.training.spring import (
    SPRING_STATE_VERSION,
    SPRINGPolicy,
    SPRINGUpdate,
)
from tpen.training.update import ModelParameterBinding, ScoreUpdateInput


LEARNING_RATE = 0.05
ORACLE_RTOL = 1.0e-9
ORACLE_ATOL = 1.0e-9


def _layout(n_parameters: int) -> ParameterLayout:
    """Build a one-slot float64 layout for an analytic linear model."""

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
    n_electrons: int = 2,
) -> ScoreUpdateInput:
    """Build a score-bearing step for ``log|psi| = features @ parameter``."""

    n_samples = int(features.shape[0])
    positions = torch.zeros((n_samples, n_electrons, 1), dtype=torch.float64)
    spins = torch.ones((n_samples, n_electrons), dtype=torch.float64)
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
        parameter_scores=MaterializedParameterLogScores(
            layout=layout,
            blocks=(features,),
        ),
        parameter_binding=ParameterBinding(layout=layout, parameters=(parameter,)),
    )


def _base_policy(*, max_update_norm: float | None = None) -> SRPolicy:
    """Use an explicit sample-space base so SPRING is unambiguously minSR."""

    return SRPolicy(
        solve_space="sample",
        damping=DampingPolicy(absolute=0.0, relative=1.0e-2),
        learning_rate=LEARNING_RATE,
        max_update_norm=max_update_norm,
    )


def _method(
    parameter: torch.nn.Parameter,
    *,
    history_decay: float = 0.35,
    max_update_norm: float | None = None,
) -> SPRINGUpdate:
    """Build one SPRING method with the same base policy used by SR parity arms."""

    policy = SPRINGPolicy(
        base=_base_policy(max_update_norm=max_update_norm),
        history_decay=history_decay,
    )
    optimizer = torch.optim.SGD([parameter], lr=policy.base.learning_rate)
    return SPRINGUpdate(
        optimizer,
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
        policy=policy,
        conventions=ScoreConventions(),
    )


def _sr_method_from_base(
    parameter: torch.nn.Parameter,
    base: SRPolicy,
) -> StochasticReconfigurationUpdate:
    """Build the comparison arm from the exact SPRING policy base object."""

    return StochasticReconfigurationUpdate(
        torch.optim.SGD([parameter], lr=base.learning_rate),
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
        policy=base,
        conventions=ScoreConventions(),
    )


def _problem(
    *,
    n_samples: int = 5,
    n_parameters: int = 8,
    seed: int = 41,
) -> tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]:
    """Return deterministic float64 model parameters, scores, and energies."""

    generator = torch.Generator().manual_seed(seed)
    parameter = torch.nn.Parameter(
        torch.randn(n_parameters, generator=generator, dtype=torch.float64)
    )
    features = torch.randn(
        (n_samples, n_parameters), generator=generator, dtype=torch.float64
    )
    energies = torch.randn(n_samples, generator=generator, dtype=torch.float64)
    return parameter, features, energies


def _sequence(seed: int = 42) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Return three deterministic score/energy steps with a common layout."""

    generator = torch.Generator().manual_seed(seed)
    features = tuple(
        torch.randn((5, 8), generator=generator, dtype=torch.float64)
        for _ in range(3)
    )
    energies = tuple(
        torch.randn(5, generator=generator, dtype=torch.float64)
        for _ in range(3)
    )
    return features, energies


def test_spring_first_step_is_bitwise_equal_to_minsr_policy_base() -> None:
    """The zero-history first step is exact minSR, not merely close."""

    parameter, features, energies = _problem(seed=51)
    spring_parameter = torch.nn.Parameter(parameter.detach().clone())
    sr_parameter = torch.nn.Parameter(parameter.detach().clone())
    spring = _method(spring_parameter, history_decay=0.37)
    sr = _sr_method_from_base(sr_parameter, spring.policy.base)

    spring.update(_step_input(spring_parameter, features, energies))
    sr.update(_step_input(sr_parameter, features, energies))

    assert torch.equal(spring_parameter.detach(), sr_parameter.detach())
    assert torch.equal(spring.history, sr_parameter.grad.detach())
    assert spring.last_telemetry is not None
    assert spring.last_telemetry.history_advanced is True


def test_history_decay_zero_matches_minsr_each_step_and_positive_diverges() -> None:
    """Zero decay is SR at every step; positive decay must affect step two."""

    features_steps, energy_steps = _sequence(seed=52)
    initial, _, _ = _problem(seed=53)

    zero_parameter = torch.nn.Parameter(initial.detach().clone())
    zero_sr_parameter = torch.nn.Parameter(initial.detach().clone())
    zero = _method(zero_parameter, history_decay=0.0)
    zero_sr = _sr_method_from_base(zero_sr_parameter, zero.policy.base)
    for step, (features, energies) in enumerate(zip(features_steps, energy_steps, strict=True)):
        zero.update(_step_input(zero_parameter, features, energies, step=step))
        zero_sr.update(_step_input(zero_sr_parameter, features, energies, step=step))
        assert torch.equal(zero_parameter.detach(), zero_sr_parameter.detach())

    positive_parameter = torch.nn.Parameter(initial.detach().clone())
    positive_sr_parameter = torch.nn.Parameter(initial.detach().clone())
    positive = _method(positive_parameter, history_decay=0.35)
    positive_sr = _sr_method_from_base(positive_sr_parameter, positive.policy.base)
    positive.update(_step_input(positive_parameter, features_steps[0], energy_steps[0], step=0))
    positive_sr.update(
        _step_input(positive_sr_parameter, features_steps[0], energy_steps[0], step=0)
    )
    assert torch.equal(positive_parameter.detach(), positive_sr_parameter.detach())
    positive.update(_step_input(positive_parameter, features_steps[1], energy_steps[1], step=1))
    positive_sr.update(
        _step_input(positive_sr_parameter, features_steps[1], energy_steps[1], step=1)
    )
    assert not torch.equal(positive_parameter.detach(), positive_sr_parameter.detach())


def test_two_step_spring_matches_independent_parameter_space_oracle() -> None:
    """Two public updates agree with an independent NumPy parameter solve."""

    features_steps, energy_steps = _sequence(seed=54)
    parameter, _, _ = _problem(seed=55)
    method = _method(parameter, history_decay=0.31)
    initial = parameter.detach().clone().numpy()
    oracle = spring_trajectory(
        tuple(features.numpy() for features in features_steps[:2]),
        tuple(energies.numpy() for energies in energy_steps[:2]),
        history_decay=0.31,
        relative=1.0e-2,
    )

    expected_parameter = initial.copy()
    for step, (features, energies, expected) in enumerate(
        zip(features_steps[:2], energy_steps[:2], oracle, strict=True)
    ):
        method.update(_step_input(parameter, features, energies, step=step))
        expected_parameter -= LEARNING_RATE * expected.history
        # 1e-9 is explicit headroom for Torch eigh versus NumPy LU in float64.
        np.testing.assert_allclose(
            parameter.detach().numpy(),
            expected_parameter,
            rtol=ORACLE_RTOL,
            atol=ORACLE_ATOL,
        )
        # The persisted vector is unscaled, so compare it directly to z_t.
        np.testing.assert_allclose(
            method.history.detach().numpy(),
            expected.history,
            rtol=ORACLE_RTOL,
            atol=ORACLE_ATOL,
        )


def test_history_state_resume_matches_uninterrupted_third_step() -> None:
    """A nonzero-history split resume is exactly the uninterrupted trajectory."""

    features_steps, energy_steps = _sequence(seed=56)
    initial, _, _ = _problem(seed=57)

    straight_parameter = torch.nn.Parameter(initial.detach().clone())
    straight = _method(straight_parameter, history_decay=0.4)
    for step in range(3):
        straight.update(
            _step_input(straight_parameter, features_steps[step], energy_steps[step], step=step)
        )

    split_parameter = torch.nn.Parameter(initial.detach().clone())
    split = _method(split_parameter, history_decay=0.4)
    for step in range(2):
        split.update(
            _step_input(split_parameter, features_steps[step], energy_steps[step], step=step)
        )
    assert torch.linalg.vector_norm(split.history).item() > 0.0
    saved = json.loads(json.dumps(dict(split.method_state_dict())))

    resumed_parameter = torch.nn.Parameter(split_parameter.detach().clone())
    resumed = _method(resumed_parameter, history_decay=0.4)
    resumed.load_method_state_dict(saved)
    resumed.update(
        _step_input(resumed_parameter, features_steps[2], energy_steps[2], step=2)
    )

    assert torch.equal(resumed_parameter.detach(), straight_parameter.detach())
    assert torch.equal(resumed.history, straight.history)
    assert resumed.completed_updates == straight.completed_updates == 3
    assert resumed.method_state_dict() == straight.method_state_dict()


def test_cap_applies_uncapped_history_is_persisted() -> None:
    """A firing trust cap changes application, never the persisted projected history."""

    parameter, features, energies = _problem(seed=58)
    oracle = spring_trajectory(
        (features.numpy(),),
        (energies.numpy(),),
        history_decay=0.35,
        relative=1.0e-2,
    )[0]
    cap = 0.25 * LEARNING_RATE * float(np.linalg.norm(oracle.history))
    method = _method(parameter, history_decay=0.35, max_update_norm=cap)
    before = parameter.detach().clone()

    method.update(_step_input(parameter, features, energies))
    applied = (before - parameter.detach()).numpy()
    # Explicit tolerance covers one float64 norm and the plain-SGD subtraction.
    assert float(np.linalg.norm(applied)) == pytest.approx(cap, rel=1.0e-10, abs=1.0e-12)
    assert method.last_telemetry is not None
    assert method.last_telemetry.trust_scale == pytest.approx(
        0.25, rel=1.0e-10, abs=1.0e-12
    )
    np.testing.assert_allclose(
        np.asarray(method.method_state_dict()["history"], dtype=np.float64),
        oracle.history,
        rtol=ORACLE_RTOL,
        atol=ORACLE_ATOL,
    )


def test_skips_leave_history_unchanged_and_nonfinite_direction_not_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every SR skip reason preserves history, including a non-finite direction."""

    parameter, features, energies = _problem(seed=59)
    method = _method(parameter, history_decay=0.4)
    initial = method.update(_step_input(parameter, features, energies, step=0))
    assert initial.applied is True
    prior = method.history.clone()

    zero = method.update(
        _step_input(parameter, features, energies, step=1, n_electrons=0)
    )
    assert zero.applied is False
    assert method.last_telemetry is not None
    assert method.last_telemetry.reason == "zero_electron_batch"
    assert method.last_telemetry.history_advanced is False
    assert torch.equal(method.history, prior)

    one_finite = torch.full_like(energies, float("nan"))
    one_finite[0] = energies[0]
    insufficient = method.update(
        _step_input(parameter, features, one_finite, step=2)
    )
    assert insufficient.applied is False
    assert method.last_telemetry is not None
    assert method.last_telemetry.reason == "insufficient_finite_samples"
    assert torch.equal(method.history, prior)

    diagnostics = method.last_telemetry.diagnostics

    def nonfinite_solver(*args: object, **kwargs: object) -> tuple[torch.Tensor, SolveDiagnostics | None]:
        del kwargs
        operator = args[0]
        assert hasattr(operator, "n_parameters")
        return torch.full(
            (operator.n_parameters,),
            float("nan"),
            dtype=operator.geometry.dtype,
            device=operator.geometry.device,
        ), diagnostics

    monkeypatch.setattr(spring_module, "solve_sample_space", nonfinite_solver)
    nonfinite = method.update(_step_input(parameter, features, energies, step=3))
    assert nonfinite.applied is False
    assert method.last_telemetry is not None
    assert method.last_telemetry.reason == "nonfinite_update_direction"
    assert method.last_telemetry.history_advanced is False
    assert torch.equal(method.history, prior)


def test_state_and_policy_validation_reject_mismatch_and_bounds() -> None:
    """State identity, history shape, and required decay bounds are enforced."""

    parameter, features, energies = _problem(seed=60)
    method = _method(parameter, history_decay=0.3)
    method.update(_step_input(parameter, features, energies))
    state = dict(method.method_state_dict())

    for invalid in (-1.0e-6, 1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="history_decay"):
            SPRINGPolicy(base=_base_policy(), history_decay=invalid)

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = _method(restored_parameter, history_decay=0.3)
    restored.load_method_state_dict(json.loads(json.dumps(state)))
    assert torch.equal(restored.history, method.history)
    assert restored.completed_updates == method.completed_updates

    wrong_version = copy.deepcopy(state)
    wrong_version["version"] = "spring-state-0"
    with pytest.raises(ValueError, match="unsupported SPRING state version"):
        restored.load_method_state_dict(wrong_version)

    wrong_fingerprint = copy.deepcopy(state)
    wrong_fingerprint["fingerprint"]["digest"] = "not-the-live-digest"
    with pytest.raises(ValueError, match="fingerprint does not match"):
        restored.load_method_state_dict(wrong_fingerprint)

    wrong_policy = copy.deepcopy(state)
    wrong_policy["policy"]["history_decay"] = 0.2
    with pytest.raises(ValueError, match="policy does not match"):
        restored.load_method_state_dict(wrong_policy)

    wrong_shape = copy.deepcopy(state)
    wrong_shape["history"] = wrong_shape["history"][:-1]
    with pytest.raises(ValueError, match="history length"):
        restored.load_method_state_dict(wrong_shape)


def test_spring_consumes_one_score_packet_and_one_solve_per_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public score seam is called once and production solves once."""

    from tests.unit.nn.test_tpen_wavefunction_parameter_scores import _build_model
    from tpen.nn import InteractionMode, MaterializedParameterScoreRequest

    torch.manual_seed(61)
    model = _build_model(InteractionMode.TENSOR_PRODUCT)
    batch = ElectronBatch(
        positions=torch.randn((5, 3, 3), dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0, 1.0]] * 5, dtype=torch.float64),
    )
    method = SPRINGUpdate(
        torch.optim.SGD(model.parameters(), lr=LEARNING_RATE),
        model_parameters=ModelParameterBinding(parameters=tuple(model.parameters())),
        policy=SPRINGPolicy(base=_base_policy(), history_decay=0.25),
    )
    original_request = model.evaluate_materialized_parameter_score_request
    request_count = 0

    def counted_request(*, request: MaterializedParameterScoreRequest, batch: ElectronBatch):
        nonlocal request_count
        request_count += 1
        return original_request(request=request, batch=batch)

    monkeypatch.setattr(model, "evaluate_materialized_parameter_score_request", counted_request)
    original_solver = spring_module.solve_sample_space
    solve_count = 0

    def counted_solver(*args: object, **kwargs: object):
        nonlocal solve_count
        solve_count += 1
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(spring_module, "solve_sample_space", counted_solver)
    packet = model.evaluate_materialized_parameter_score_request(
        request=method.forward_request(), batch=batch
    )
    energies = torch.randn(5, generator=torch.Generator().manual_seed(62), dtype=torch.float64)
    update_input = ScoreUpdateInput(
        batch=batch,
        wavefunction=packet.output,
        local_energy=energies,
        step=0,
        parameter_scores=packet.parameter_scores,
        parameter_binding=model.parameter_binding,
    )
    assert method.update(update_input).applied
    assert request_count == 1
    assert solve_count == 1
