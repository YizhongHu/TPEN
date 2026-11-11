"""Contract tests for the SR/minSR VMC update method.

The step model used throughout is a linear log amplitude,
``log|psi(x_k)| = X[k] . theta``.  Its exact per-sample score is the feature
row ``X[k]``, so the score matrix is known analytically and no TPEN score
machinery is involved in producing the reference.  That keeps these tests
about the update method rather than about the model that would normally feed
it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers.sr_dense_oracle import sr_direction
from tpen.data.batch import (
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    ParameterSlot,
    WavefunctionOutput,
)
from tpen.training.qgt import DampingPolicy
from tpen.training.score_geometry import ScoreConventions
from tpen.training.sr import (
    SR_STATE_VERSION,
    SRPolicy,
    StochasticReconfigurationUpdate,
)
from tpen.training.update import ModelParameterBinding, ScoreUpdateInput
from tpen.training.vmc import compute_vmc_objective

SOLVE_TOLERANCE = 1.0e-9
LEARNING_RATE = 0.05


def _layout(n_parameters: int) -> ParameterLayout:
    """Build a single-slot float64 layout of the requested width."""

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
    phase: torch.Tensor | None = None,
) -> ScoreUpdateInput:
    """Build a score-bearing step record for the linear log-amplitude model."""

    n_samples = int(features.shape[0])
    positions = torch.zeros((n_samples, n_electrons, 1), dtype=torch.float64)
    spins = torch.ones((n_samples, n_electrons), dtype=torch.float64)
    batch = ElectronBatch(positions=positions, spins=spins)
    logabs = (features @ parameter).detach()
    output = WavefunctionOutput(
        logabs=logabs,
        sign=torch.ones(n_samples, dtype=torch.float64),
        phase=phase,
    )
    layout = _layout(int(features.shape[1]))
    return ScoreUpdateInput(
        batch=batch,
        wavefunction=output,
        local_energy=energies,
        step=step,
        # The exact score of a linear log amplitude is the feature row.
        parameter_scores=MaterializedParameterLogScores(layout=layout, blocks=(features,)),
        parameter_binding=ParameterBinding(layout=layout, parameters=(parameter,)),
    )


def _method(
    parameter: torch.nn.Parameter,
    *,
    policy: SRPolicy | None = None,
    learning_rate: float = LEARNING_RATE,
) -> StochasticReconfigurationUpdate:
    """Build an SR method owning a plain SGD over one parameter."""

    resolved_policy = policy or SRPolicy(
        damping=DampingPolicy(absolute=0.0, relative=1.0e-2),
        learning_rate=learning_rate,
    )
    optimizer = torch.optim.SGD([parameter], lr=resolved_policy.learning_rate)
    return StochasticReconfigurationUpdate(
        optimizer,
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
        policy=resolved_policy,
        conventions=ScoreConventions(),
    )


def _problem(
    *,
    n_samples: int,
    n_parameters: int,
    seed: int,
) -> tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]:
    """Return a live parameter, feature/score rows, and local energies."""

    generator = torch.Generator().manual_seed(seed)
    parameter = torch.nn.Parameter(
        torch.randn(n_parameters, generator=generator, dtype=torch.float64)
    )
    features = torch.randn(
        (n_samples, n_parameters), generator=generator, dtype=torch.float64
    )
    energies = torch.randn(n_samples, generator=generator, dtype=torch.float64)
    return parameter, features, energies


def test_applied_step_is_exactly_minus_lr_times_the_reference_direction() -> None:
    """The parameter displacement equals ``-lr * delta`` from the NumPy oracle."""

    parameter, features, energies = _problem(n_samples=9, n_parameters=4, seed=1)
    before = parameter.detach().clone()
    method = _method(parameter)

    result = method.update(_step_input(parameter, features, energies))

    expected = sr_direction(
        features.numpy(), energies.numpy(), absolute=0.0, relative=1.0e-2
    )
    np.testing.assert_allclose(
        (before - parameter.detach()).numpy(),
        LEARNING_RATE * expected.direction,
        rtol=SOLVE_TOLERANCE,
        atol=SOLVE_TOLERANCE,
    )
    assert result.applied is True
    # grad_norm is the Euclidean energy-gradient norm, not the natural one, so
    # it stays comparable with an Adam run's headline metric.
    assert result.grad_norm == pytest.approx(
        float(np.linalg.norm(expected.gradient)), rel=1.0e-10
    )


def test_sr_and_minsr_routes_produce_identical_parameter_updates() -> None:
    """Forcing either route gives the same step, in both shape regimes.

    This is the acceptance criterion that SR and minSR agree where they are
    mathematically equivalent, checked end to end through the update method
    rather than only at the solver.
    """

    for n_samples, n_parameters in ((12, 4), (4, 12)):
        parameter_a, features, energies = _problem(
            n_samples=n_samples, n_parameters=n_parameters, seed=2
        )
        parameter_b = torch.nn.Parameter(parameter_a.detach().clone())

        method_a = _method(
            parameter_a,
            policy=SRPolicy(
                solve_space="parameter",
                damping=DampingPolicy(absolute=0.0, relative=1.0e-2),
                learning_rate=LEARNING_RATE,
            ),
        )
        method_b = _method(
            parameter_b,
            policy=SRPolicy(
                solve_space="sample",
                damping=DampingPolicy(absolute=0.0, relative=1.0e-2),
                learning_rate=LEARNING_RATE,
            ),
        )

        method_a.update(_step_input(parameter_a, features, energies))
        method_b.update(_step_input(parameter_b, features, energies))

        np.testing.assert_allclose(
            parameter_a.detach().numpy(),
            parameter_b.detach().numpy(),
            rtol=SOLVE_TOLERANCE,
            atol=SOLVE_TOLERANCE,
        )
        assert method_a.last_telemetry.diagnostics.space == "parameter"
        assert method_b.last_telemetry.diagnostics.space == "sample"


def test_auto_route_picks_the_smaller_matrix() -> None:
    """``auto`` is a cost choice: sample space only when ``B < P``."""

    wide_parameter, wide_features, wide_energies = _problem(
        n_samples=4, n_parameters=12, seed=3
    )
    tall_parameter, tall_features, tall_energies = _problem(
        n_samples=12, n_parameters=4, seed=3
    )

    wide_method = _method(wide_parameter)
    tall_method = _method(tall_parameter)
    wide_method.update(_step_input(wide_parameter, wide_features, wide_energies))
    tall_method.update(_step_input(tall_parameter, tall_features, tall_energies))

    assert wide_method.last_telemetry.diagnostics.space == "sample"
    assert tall_method.last_telemetry.diagnostics.space == "parameter"


def test_euclidean_limit_recovers_the_ordinary_vmc_gradient_direction() -> None:
    """As damping dominates the QGT, the step aligns with the autograd gradient.

    The reference gradient comes from ``compute_vmc_objective`` through
    autograd -- TPEN's existing, independently tested objective -- so this
    pins the SR engine to the ordinary VMC path rather than to itself.
    """

    parameter, features, energies = _problem(n_samples=10, n_parameters=5, seed=4)
    autograd_parameter = torch.nn.Parameter(parameter.detach().clone())
    compute_vmc_objective(features @ autograd_parameter, energies).loss.backward()
    reference = autograd_parameter.grad.detach().clone()

    before = parameter.detach().clone()
    # A damping term vastly larger than the QGT spectrum makes
    # (S + lambda I)^{-1} g approach g / lambda, i.e. plain gradient descent.
    method = _method(
        parameter,
        policy=SRPolicy(
            damping=DampingPolicy(absolute=0.0, relative=1.0e10),
            learning_rate=LEARNING_RATE,
        ),
    )
    result = method.update(_step_input(parameter, features, energies))

    # Read the direction from `.grad` rather than by differencing parameters.
    # Differencing is the wrong instrument at this damping: the displacement is
    # ~3e-12 against O(1) parameters, so recovering it from a subtraction loses
    # about five significant digits to cancellation and reports a ~1e-5
    # direction error that is an artefact of the measurement, not of the step.
    # `.grad` holds the preconditioned direction exactly, and plain SGD does not
    # modify it. The separate claim that parameters move by exactly
    # `-lr * direction` is covered at ordinary damping by
    # ``test_applied_step_is_exactly_minus_lr_times_the_reference_direction``.
    direction = parameter.grad.detach().numpy()
    reference_direction = reference.numpy() / np.linalg.norm(reference.numpy())
    step_direction = direction / np.linalg.norm(direction)

    np.testing.assert_allclose(
        step_direction, reference_direction, rtol=1.0e-9, atol=1.0e-9
    )
    assert result.grad_norm == pytest.approx(
        float(np.linalg.norm(reference.numpy())), rel=1.0e-10
    )
    # The step must still have moved the parameters in that direction.
    assert not torch.equal(parameter.detach(), before)
    assert float(
        np.dot(
            (before - parameter.detach()).numpy() / np.linalg.norm(
                (before - parameter.detach()).numpy()
            ),
            reference_direction,
        )
    ) == pytest.approx(1.0, abs=1.0e-8)


def test_nonfinite_energy_samples_are_excluded_exactly_as_the_objective_does() -> None:
    """A NaN local energy is dropped, and centering uses only the survivors.

    ``compute_vmc_objective`` excludes non-finite local energies, so SR must
    too; otherwise the two disagree precisely on the steps where a sample
    diverged. Dropping the row *before* centering matters -- centering first
    would build the mean from a sample that is then discarded.
    """

    parameter, features, energies = _problem(n_samples=8, n_parameters=3, seed=5)
    poisoned = energies.clone()
    poisoned[3] = float("nan")
    poisoned[6] = float("inf")
    keep = [0, 1, 2, 4, 5, 7]

    before = parameter.detach().clone()
    method = _method(parameter)
    result = method.update(_step_input(parameter, features, poisoned))

    expected = sr_direction(
        features.numpy()[keep],
        energies.numpy()[keep],
        absolute=0.0,
        relative=1.0e-2,
    )
    np.testing.assert_allclose(
        (before - parameter.detach()).numpy(),
        LEARNING_RATE * expected.direction,
        rtol=SOLVE_TOLERANCE,
        atol=SOLVE_TOLERANCE,
    )
    assert result.applied is True
    assert method.last_telemetry.n_samples == 8
    assert method.last_telemetry.n_finite_samples == 6


def test_a_nonfinite_score_row_is_dropped_with_its_sample() -> None:
    """One exploded score row cannot poison the whole QGT through its outer product."""

    parameter, features, energies = _problem(n_samples=7, n_parameters=3, seed=6)
    poisoned = features.clone()
    poisoned[2, 1] = float("nan")
    keep = [0, 1, 3, 4, 5, 6]

    before = parameter.detach().clone()
    method = _method(parameter)
    method.update(_step_input(parameter, poisoned, energies))

    expected = sr_direction(
        features.numpy()[keep],
        energies.numpy()[keep],
        absolute=0.0,
        relative=1.0e-2,
    )
    np.testing.assert_allclose(
        (before - parameter.detach()).numpy(),
        LEARNING_RATE * expected.direction,
        rtol=SOLVE_TOLERANCE,
        atol=SOLVE_TOLERANCE,
    )
    assert method.last_telemetry.n_finite_samples == 6


def test_trust_cap_rescales_the_step_without_turning_it() -> None:
    """The cap bounds the displacement norm and preserves the direction."""

    parameter, features, energies = _problem(n_samples=9, n_parameters=4, seed=7)
    uncapped_parameter = torch.nn.Parameter(parameter.detach().clone())
    before = parameter.detach().clone()

    uncapped = _method(uncapped_parameter)
    uncapped.update(_step_input(uncapped_parameter, features, energies))
    uncapped_step = (before - uncapped_parameter.detach()).numpy()
    cap = 0.25 * float(np.linalg.norm(uncapped_step))

    capped = _method(
        parameter,
        policy=SRPolicy(
            damping=DampingPolicy(absolute=0.0, relative=1.0e-2),
            learning_rate=LEARNING_RATE,
            max_update_norm=cap,
        ),
    )
    capped.update(_step_input(parameter, features, energies))
    capped_step = (before - parameter.detach()).numpy()

    assert float(np.linalg.norm(capped_step)) == pytest.approx(cap, rel=1.0e-10)
    np.testing.assert_allclose(
        capped_step / np.linalg.norm(capped_step),
        uncapped_step / np.linalg.norm(uncapped_step),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert capped.last_telemetry.trust_scale == pytest.approx(0.25, rel=1.0e-10)
    assert uncapped.last_telemetry.trust_scale == 1.0


def test_adam_and_momentum_are_rejected_rather_than_silently_accepted() -> None:
    """There is no configuration in which SR quietly trains with Adam."""

    parameter, _, _ = _problem(n_samples=5, n_parameters=3, seed=8)
    binding = ModelParameterBinding(parameters=(parameter,))
    policy = SRPolicy(learning_rate=LEARNING_RATE)

    with pytest.raises(TypeError, match="Adam"):
        StochasticReconfigurationUpdate(
            torch.optim.Adam([parameter], lr=LEARNING_RATE),
            model_parameters=binding,
            policy=policy,
        )
    with pytest.raises(ValueError, match="momentum"):
        StochasticReconfigurationUpdate(
            torch.optim.SGD([parameter], lr=LEARNING_RATE, momentum=0.9),
            model_parameters=binding,
            policy=policy,
        )
    with pytest.raises(ValueError, match="weight_decay"):
        StochasticReconfigurationUpdate(
            torch.optim.SGD([parameter], lr=LEARNING_RATE, weight_decay=1.0e-4),
            model_parameters=binding,
            policy=policy,
        )
    with pytest.raises(ValueError, match="disagrees with SRPolicy.learning_rate"):
        StochasticReconfigurationUpdate(
            torch.optim.SGD([parameter], lr=LEARNING_RATE * 2.0),
            model_parameters=binding,
            policy=policy,
        )


def test_complex_wavefunction_is_rejected() -> None:
    """A phase-bearing step must fail rather than silently drop the phase."""

    parameter, features, energies = _problem(n_samples=6, n_parameters=3, seed=9)
    method = _method(parameter)
    phase = torch.zeros(6, dtype=torch.float64)

    with pytest.raises(ValueError, match="real wavefunctions only"):
        method.update(_step_input(parameter, features, energies, phase=phase))


def test_zero_electron_batch_reports_a_skip_instead_of_stepping() -> None:
    """The vacuum has no coordinate degrees of freedom, so nothing is applied."""

    parameter, features, energies = _problem(n_samples=4, n_parameters=3, seed=10)
    before = parameter.detach().clone()
    method = _method(parameter)

    result = method.update(
        _step_input(parameter, features, energies, n_electrons=0)
    )

    assert result.applied is False
    assert method.last_telemetry.reason == "zero_electron_batch"
    assert torch.equal(parameter.detach(), before)


def test_all_nonfinite_energies_raise_for_a_nonzero_electron_batch() -> None:
    """No finite sample is a loud failure, never a quiet no-op step."""

    parameter, features, _ = _problem(n_samples=5, n_parameters=3, seed=11)
    method = _method(parameter)
    energies = torch.full((5,), float("nan"), dtype=torch.float64)

    with pytest.raises(RuntimeError, match="no finite local-energy sample"):
        method.update(_step_input(parameter, features, energies))


def test_a_single_finite_sample_is_skipped_with_a_reason() -> None:
    """One sample centers to zero, so the step is provably zero and is reported."""

    parameter, features, energies = _problem(n_samples=5, n_parameters=3, seed=12)
    poisoned = torch.full((5,), float("nan"), dtype=torch.float64)
    poisoned[1] = float(energies[1].item())
    before = parameter.detach().clone()
    method = _method(parameter)

    result = method.update(_step_input(parameter, features, poisoned))

    assert result.applied is False
    assert method.last_telemetry.reason == "insufficient_finite_samples"
    assert torch.equal(parameter.detach(), before)


def test_state_envelope_round_trips_and_rejects_layout_drift() -> None:
    """Restore carries the step counter and refuses a mismatched fingerprint.

    The envelope lives on `method_state_dict`, not `state_dict`: the latter has
    an established meaning for the legacy adapter, where it is the raw
    optimizer payload the checkpoint already persists separately.
    """

    parameter, features, energies = _problem(n_samples=8, n_parameters=4, seed=13)
    method = _method(parameter)
    method.update(_step_input(parameter, features, energies))
    state = method.method_state_dict()

    assert state["version"] == SR_STATE_VERSION
    assert state["completed_updates"] == 1

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = _method(restored_parameter)
    restored.load_method_state_dict(state)
    assert restored.completed_updates == 1

    # A model with a different parameter layout must be refused, or a resumed
    # run would reuse state describing a different geometry.
    other_parameter = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
    other = _method(other_parameter)
    with pytest.raises(ValueError, match="fingerprint does not match"):
        other.load_method_state_dict(state)

    with pytest.raises(ValueError, match="unsupported SR state version"):
        restored.load_method_state_dict({**state, "version": "sr-state-0"})


def test_binding_drift_is_rejected_rather_than_updating_a_stale_reference() -> None:
    """Scores must reference the parameters this method actually owns."""

    parameter, features, energies = _problem(n_samples=6, n_parameters=3, seed=14)
    method = _method(parameter)
    impostor = torch.nn.Parameter(parameter.detach().clone())

    with pytest.raises(ValueError, match="does not reference the bound live"):
        method.update(_step_input(impostor, features, energies))


def test_telemetry_reports_both_norms_and_the_solve_route() -> None:
    """The natural and Euclidean norms are both observable, and differ."""

    parameter, features, energies = _problem(n_samples=10, n_parameters=4, seed=15)
    method = _method(parameter)

    method.update(_step_input(parameter, features, energies, step=7))
    metrics = method.last_telemetry.as_metrics()

    assert metrics["sr_applied"] is True
    assert metrics["sr_reason"] == "applied"
    assert metrics["sr_step"] == 7
    assert metrics["sr_qgt_space"] == "parameter"
    assert metrics["sr_qgt_shift"] > 0.0
    # Preconditioning must actually change the vector, or the reported natural
    # gradient would be the Euclidean one under another name.
    assert metrics["sr_update_direction_norm"] != pytest.approx(
        metrics["sr_energy_gradient_norm"], rel=1.0e-3
    )
    assert metrics["sr_applied_update_norm"] == pytest.approx(
        LEARNING_RATE * metrics["sr_update_direction_norm"], rel=1.0e-12
    )
