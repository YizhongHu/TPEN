"""End-to-end contracts for SR/minSR driven through `VMCTrainer`.

These cover the N3 acceptance items that only appear once the method is wired
into the loop: that a real SR and a real minSR update run on a landed TPEN
model, that legacy behaviour is untouched, that method state survives a
checkpoint round trip, and that a declined step no longer looks to the trainer
like a disconnected loss.

The model is a real `TPENWaveFunction` driven through the real score-request
provider, not a stub that could agree with a wrong contract.  Most of these
tests use a THREE-electron model, which keeps the odd-electron Pfaffian path in
the loop; the two-electron Hooke fixture -- the shape of the helium target -- is
driven end to end by
`test_sr_trains_a_two_electron_tpen_model_through_the_trainer` at the bottom.
That test used to assert the opposite, because the score seam refused any even
electron count outright.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tpen.training.qgt import DampingPolicy
from tpen.training.score_geometry import ScoreConventions
from tpen.training.sr import SRPolicy, StochasticReconfigurationUpdate
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    LegacyAutogradUpdate,
    ModelParameterBinding,
    ScoreUpdateInput,
    VMCUpdateMethod,
    VMCUpdateResult,
)
from tests.helpers.hooke_models import (
    INACTIVE_PAIR_PARAMETERS,
    build_tiny_hamiltonian_terms,
    build_tiny_sampler,
    build_tiny_spenn,
)
from tests.helpers.score_reachability import disconnected_parameter_names
from tests.unit.nn.test_tpen_wavefunction_parameter_scores import _build_model
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext
from tpen.data.batch import ElectronBatch
from tpen.nn import InteractionMode

LEARNING_RATE = 1.0e-3


def build_connected_model():
    """Build a real TPENWaveFunction whose parameters are ALL score-connected.

    Three electrons, so the odd-electron Pfaffian padding block is present and
    every parameter is autograd-reachable. That WAS a workaround -- the seam
    refused even counts outright -- and is now a deliberate choice: with no
    substituted zeros anywhere in the packet, a failure in these tests is a
    failure of the SR integration and cannot be a failure of the substitution.
    The two-electron fixture is covered separately at the bottom of this file.

    It is still a genuine `TPENWaveFunction` driven through the genuine score
    provider, not a stub that could agree with a wrong contract.
    """

    return _build_model(InteractionMode.TENSOR_PRODUCT)


class _FixedWalkers:
    """Walkers wrapping one fixed 3-electron batch."""

    def __init__(self, batch: ElectronBatch) -> None:
        self._batch = batch
        self.n_walkers = int(batch.batch_size)

    def make_batch(self) -> ElectronBatch:
        return self._batch


class _FixedSampler:
    """Deterministic sampler: the loop under test is the updater, not the MCMC."""

    def __init__(self, *, n_walkers: int = 6, seed: int = 5) -> None:
        generator = torch.Generator().manual_seed(seed)
        positions = torch.randn(n_walkers, 3, 3, generator=generator, dtype=torch.float64)
        spins = torch.tensor([[1.0, -1.0, 1.0]] * n_walkers, dtype=torch.float64)
        self._batch = ElectronBatch(positions=positions, spins=spins)

    def collect_samples(self, model, *, device=None):
        del model, device
        return _FixedWalkers(self._batch), None


def _sr_method(
    model,
    *,
    solve_space: str = "auto",
    max_update_norm: float | None = None,
) -> StochasticReconfigurationUpdate:
    """Build an SR method owning plain SGD over the model's parameters."""

    parameters = tuple(model.parameters())
    policy = SRPolicy(
        solve_space=solve_space,
        damping=DampingPolicy(absolute=0.0, relative=1.0e-2, minimum=1.0e-12),
        learning_rate=LEARNING_RATE,
        max_update_norm=max_update_norm,
    )
    return StochasticReconfigurationUpdate(
        torch.optim.SGD(parameters, lr=LEARNING_RATE),
        model_parameters=ModelParameterBinding(parameters=parameters),
        policy=policy,
        conventions=ScoreConventions(solve_dtype=torch.float64),
    )


def _fit(
    *,
    solve_space: str | None,
    max_steps: int = 1,
    seed: int = 0,
) -> tuple[VMCTrainer, Any, _StubContext, Any]:
    """Run the trainer for a few steps, seeded so two runs are comparable."""

    torch.manual_seed(seed)
    model = build_connected_model()
    sampler = _FixedSampler()
    context = _StubContext()
    method = None if solve_space is None else _sr_method(model, solve_space=solve_space)
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if method is None
        else method.optimizer
    )
    trainer = VMCTrainer(max_steps=max_steps, log_every_n_steps=1, update_method=method)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=optimizer,
        context=context,
        emit=lambda **_: None,
    )
    return trainer, model, context, method


def _train_metrics(context: _StubContext) -> dict:
    """Return the last logged `train` metric record.

    The smoke module's own helper insists on exactly one record; these tests
    also run multi-step fits, so this takes the most recent one.
    """

    records = [metrics for namespace, metrics in context.records if namespace == "train"]
    assert records, "the trainer logged no train metrics"
    return records[-1]


@pytest.mark.parametrize("solve_space", ["parameter", "sample"])
def test_a_real_sr_update_runs_on_a_landed_tpen_model(solve_space: str) -> None:
    """One SR and one minSR update run end to end and move the parameters.

    This is the N3 acceptance item "at least one SR and one minSR update runs
    on representative landed TPEN parameter modes", exercised through the real
    score-request provider rather than a stub.
    """

    torch.manual_seed(0)
    reference = build_connected_model()
    before = [p.detach().clone() for p in reference.parameters()]

    trainer, model, context, method = _fit(solve_space=solve_space)

    assert trainer.completed_updates == 1
    assert method.last_telemetry is not None
    assert method.last_telemetry.applied is True
    assert method.last_telemetry.diagnostics.space == solve_space
    after = [p.detach() for p in model.parameters()]
    assert any(
        not torch.equal(b, a) for b, a in zip(before, after, strict=True)
    ), "an applied SR update must move at least one parameter"
    assert all(torch.isfinite(p).all() for p in after)


def test_the_score_forward_happens_once_per_step() -> None:
    """The score-bearing packet is produced by the single forward, not a second one.

    A design that ran an ordinary forward and then recomputed derivatives would
    double the per-step forward and derivative work. Counting provider calls is
    the only way to observe the difference from outside.
    """

    torch.manual_seed(0)
    model = build_connected_model()
    method = _sr_method(model)
    calls = {"score": 0}
    original = model.evaluate_materialized_parameter_score_request

    def counting(*args: Any, **kwargs: Any):
        calls["score"] += 1
        return original(*args, **kwargs)

    model.evaluate_materialized_parameter_score_request = counting  # type: ignore[method-assign]

    VMCTrainer(max_steps=3, log_every_n_steps=1, update_method=method).fit(
        model=model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )

    assert calls["score"] == 3, "exactly one score-bearing forward per step"


def test_sr_and_minsr_reach_the_same_parameters_through_the_trainer() -> None:
    """The two routes are one algorithm, checked end to end rather than at the solver."""

    _, dense_model, _, _ = _fit(solve_space="parameter", seed=3)
    _, sample_model, _, _ = _fit(solve_space="sample", seed=3)

    for dense, sample in zip(dense_model.parameters(), sample_model.parameters(), strict=True):
        torch.testing.assert_close(
            dense.detach(), sample.detach(), rtol=1.0e-8, atol=1.0e-8
        )


def test_sr_telemetry_reaches_the_train_metrics_record() -> None:
    """Bounded solver telemetry is logged, and the method owns its metric names."""

    _, _, context, _ = _fit(solve_space="parameter")
    metrics = _train_metrics(context)

    for key in (
        "sr_applied",
        "sr_reason",
        "sr_energy_gradient_norm",
        "sr_update_direction_norm",
        "sr_applied_update_norm",
        "sr_qgt_space",
        "sr_qgt_shift",
        "sr_qgt_retained_modes",
    ):
        assert key in metrics, f"missing SR telemetry key {key}"
    assert metrics["sr_qgt_space"] == "parameter"
    assert metrics["sr_qgt_shift"] > 0.0
    # The trainer's own keys must survive alongside the method's.
    assert "energy" in metrics and "grad_norm" in metrics and "optimizer_step" in metrics


def test_legacy_run_logs_no_solver_telemetry() -> None:
    """A method that reports nothing adds nothing, so Adam records are unchanged."""

    _, _, context, method = _fit(solve_space=None)
    metrics = _train_metrics(context)

    assert method is None
    assert not [key for key in metrics if key.startswith("sr_")]
    assert "energy" in metrics and "grad_norm" in metrics


def test_method_state_round_trips_through_the_trainer_checkpoint_state() -> None:
    """Method state is a first-class payload, and a wrong layout is refused."""

    trainer, model, _, method = _fit(solve_space="parameter", max_steps=2)

    state = trainer.state_dict()
    assert "update_method" in state, "a stateful method must appear in trainer state"
    assert state["update_method"]["completed_updates"] == 2

    # A fresh trainer over the same model restores the counter.
    resumed_method = _sr_method(model)
    resumed = VMCTrainer(max_steps=2, update_method=resumed_method)
    resumed.resolve_update_state(model=model, optimizer=resumed_method.optimizer)
    resumed.load_state_dict(state)
    assert resumed_method.completed_updates == 2
    assert resumed.next_iteration == trainer.next_iteration

    # A model with a different parameter layout must be refused rather than
    # resumed against state describing a different geometry.
    torch.manual_seed(99)
    other_model = torch.nn.Linear(4, 3, dtype=torch.float64)
    other_method = _sr_method(other_model)
    other = VMCTrainer(max_steps=1, update_method=other_method)
    other.resolve_update_state(model=other_model, optimizer=other_method.optimizer)
    with pytest.raises(ValueError, match="fingerprint does not match"):
        other.load_state_dict(state)


def test_legacy_run_adds_no_update_method_key_to_trainer_state() -> None:
    """A stateless method leaves existing checkpoint payloads byte-identical."""

    trainer, _, _, _ = _fit(solve_space=None)

    assert "update_method" not in trainer.state_dict()


def test_a_factory_is_constructed_once_across_resolve_and_fit() -> None:
    """Resolve and fit must share one method instance, or resume loads the wrong one.

    `_select_update_method` runs twice in a resumed run. If a Hydra `_partial_`
    factory produced a new instance each time, the checkpoint would restore
    into the instance that is then discarded, and the run would silently
    continue with fresh method state.
    """

    torch.manual_seed(0)
    model = build_connected_model()
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE)

    def factory(opt, *, model_parameters):
        return StochasticReconfigurationUpdate(
            opt,
            model_parameters=model_parameters,
            policy=SRPolicy(learning_rate=LEARNING_RATE),
        )

    trainer = VMCTrainer(max_steps=1, update_method=factory)
    first = trainer._select_update_method(
        model=model, optimizer=optimizer, update_method=None
    )
    second = trainer._select_update_method(
        model=model, optimizer=optimizer, update_method=None
    )

    assert isinstance(first, StochasticReconfigurationUpdate)
    assert first is second


class _AlwaysDecliningScoreMethod(StochasticReconfigurationUpdate):
    """An SR method that declines every step, for the skip-path contract."""

    def update(self, update_input: ScoreUpdateInput) -> VMCUpdateResult:
        """Decline without touching parameters, as a finite guard would."""

        return self._skip(
            reason="declined_for_test",
            step=update_input.step,
            n_samples=int(update_input.local_energy.numel()),
            n_finite=0,
            n_parameters=update_input.parameter_scores.layout.total_numel,
        )


def test_a_declined_score_update_is_reported_not_mistaken_for_a_broken_loss() -> None:
    """A score method may legitimately decline a step; the trainer must not raise.

    Before the score seam existed the trainer raised "VMC loss is disconnected
    from model parameters" for ANY non-applied update on a nonzero-electron
    batch. That check was unreachable for the legacy adapter, which raises the
    same condition itself, but it would have turned every legitimate SR decline
    into a misleading crash.
    """

    torch.manual_seed(0)
    model = build_connected_model()
    parameters = tuple(model.parameters())
    method = _AlwaysDecliningScoreMethod(
        torch.optim.SGD(parameters, lr=LEARNING_RATE),
        model_parameters=ModelParameterBinding(parameters=parameters),
        policy=SRPolicy(learning_rate=LEARNING_RATE),
    )
    before = [p.detach().clone() for p in model.parameters()]
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    trainer.fit(
        model=model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )

    assert trainer.completed_updates == 0
    assert method.last_telemetry.reason == "declined_for_test"
    for original, current in zip(before, model.parameters(), strict=True):
        assert torch.equal(original, current.detach())


def test_legacy_still_raises_on_a_genuinely_disconnected_loss() -> None:
    """Relaxing the trainer's branch must not lose the real disconnected-loss guard.

    The guard lives in `LegacyAutogradUpdate.update`, which is why removing the
    trainer's duplicate was safe. Pinning it here means a future edit cannot
    quietly delete both.
    """

    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    adapter = LegacyAutogradUpdate(
        torch.optim.Adam([parameter], lr=LEARNING_RATE),
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
    )
    from tests.unit.training.test_vmc_update import _batch, _output

    batch = _batch(n_electrons=2)
    detached_objective = torch.tensor(1.0, dtype=torch.float64)
    from tpen.training.update import AutogradUpdateInput

    disconnected = AutogradUpdateInput(
        batch=batch,
        wavefunction=_output(batch),
        local_energy=torch.zeros(1, dtype=torch.float64),
        step=0,
        objective=detached_objective,
    )

    with pytest.raises(RuntimeError, match="disconnected from model parameters"):
        adapter.update(disconnected)


def test_trust_cap_bounds_the_applied_step_through_the_trainer() -> None:
    """The trust cap is honoured on a real model, not only in isolation."""

    torch.manual_seed(11)
    uncapped_model = build_connected_model()
    uncapped = _sr_method(uncapped_model, solve_space="parameter")
    VMCTrainer(max_steps=1, update_method=uncapped).fit(
        model=uncapped_model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=uncapped.optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    free_norm = uncapped.last_telemetry.applied_update_norm
    assert free_norm > 0.0

    cap = 0.25 * free_norm
    torch.manual_seed(11)
    capped_model = build_connected_model()
    capped = _sr_method(capped_model, solve_space="parameter", max_update_norm=cap)
    VMCTrainer(max_steps=1, update_method=capped).fit(
        model=capped_model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=capped.optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )

    assert capped.last_telemetry.applied_update_norm == pytest.approx(cap, rel=1.0e-9)
    assert capped.last_telemetry.trust_scale == pytest.approx(0.25, rel=1.0e-9)


def test_score_methods_declare_a_request_and_legacy_does_not() -> None:
    """The forward-request seam is what the trainer dispatches on."""

    torch.manual_seed(0)
    model = build_connected_model()
    parameters = tuple(model.parameters())

    assert VMCUpdateMethod.forward_request(object()) is None  # type: ignore[arg-type]
    assert (
        LegacyAutogradUpdate(
            torch.optim.Adam(parameters, lr=LEARNING_RATE),
            model_parameters=ModelParameterBinding(parameters=parameters),
        ).forward_request()
        is None
    )
    assert _sr_method(model).forward_request() is not None


def test_sr_trains_a_two_electron_tpen_model_through_the_trainer() -> None:
    """WAS A PINNED BLOCKER: SR now runs end to end on the helium-shaped fixture.

    This test previously required `VMCTrainer.fit` to RAISE. The score seam
    passed ``allow_unused=False``, and 16 of this model's 39 parameters have no
    autograd path into ``logabs`` at an even electron count, so the whole score
    request failed and SR could not run on any two-electron system. Helium is
    two electrons, so that was the blocker for the target programme.

    The remaining risk after the fix is not a crash but a silent one: an update
    that quietly moves the structurally inactive coordinates on damping or
    solver noise. So completing the fit is only half of what is asserted. The
    inactive parameters must be BITWISE unchanged, and something else must have
    moved, or "unchanged" would be satisfied by an update that did nothing.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    method = _sr_method(model)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    VMCTrainer(max_steps=1, update_method=method).fit(
        model=model,
        sampler=build_tiny_sampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )

    for name, parameter in model.named_parameters():
        if name in INACTIVE_PAIR_PARAMETERS:
            torch.testing.assert_close(parameter.detach(), before[name], rtol=0.0, atol=0.0)
    moved = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    ]
    assert moved, "the fit completed but moved nothing, so the check above is vacuous"


def test_the_pair_disconnection_is_exactly_the_order_one_output_weights() -> None:
    """Name the cause, so the test above cannot keep passing for a new reason.

    Without this, the test above would pass for any reason the fit happened to
    succeed, and would go on passing if the substituted set silently grew to
    swallow parameters that ought to be trained. Comparing the reachability
    probe to a literal set owned by the fixture module names the cause directly.

    The second half is the parity claim: the SAME architecture at an ODD count
    has nothing disconnected, so this is not a property of the number two, and
    the strict CI guard is correctly placed at odd n.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    batch = build_tiny_sampler().collect_samples(model, device=torch.device("cpu"))[0].make_batch()
    assert batch.n_electrons == 2
    assert disconnected_parameter_names(model, batch) == INACTIVE_PAIR_PARAMETERS

    connected = build_connected_model()
    connected_batch = _FixedSampler().collect_samples(connected, device=None)[0].make_batch()
    assert connected_batch.n_electrons % 2 == 1
    assert disconnected_parameter_names(connected, connected_batch) == frozenset()
