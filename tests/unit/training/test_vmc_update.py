"""Contract tests for typed VMC update inputs and the legacy adapter."""

from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from tpen.checkpoint import TrainResume, restore_checkpoint, save_checkpoint
from tpen.checkpoint.hashing import file_sha256
from tpen.data.batch import (
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    ParameterSlot,
    WavefunctionOutput,
)
from tpen.physics.kinetic import KineticEnergy
from tpen.sampling import MetropolisSampler
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ModelParameterBinding,
    ObjectiveReevaluation,
    ScoreUpdateInput,
    VMCStepData,
    VMCUpdateMethod,
    VMCUpdateResult,
)
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext


def _batch(*, n_electrons: int = 1, sample_shape: tuple[int, ...] = (1,)) -> ElectronBatch:
    """Build a small float64 batch with explicit spin labels."""

    positions = torch.zeros((*sample_shape, n_electrons, 1), dtype=torch.float64)
    spins = torch.ones((*sample_shape, n_electrons), dtype=torch.float64)
    return ElectronBatch(positions=positions, spins=spins)


def _output(batch: ElectronBatch, value: torch.Tensor | None = None) -> WavefunctionOutput:
    """Build an output using the readout's flattened sample convention."""

    shape = (batch.batch_size,)
    logabs = torch.zeros(shape, dtype=batch.dtype) if value is None else value.reshape(shape)
    return WavefunctionOutput(logabs=logabs, sign=torch.ones(shape, dtype=batch.dtype))


def _reevaluation(
    batch: ElectronBatch,
    *,
    recompute: Callable[[], torch.Tensor] | None = None,
) -> ObjectiveReevaluation:
    """Build a re-evaluation for a SYNTHETIC update input.

    The adapter contracts in this module form objectives directly from a bare
    parameter rather than from a model and a Hamiltonian, so they need the seam
    record without the trainer's construction path.  The properties of the real
    construction path -- fixed sample, RNG inertness, policy continuity, true
    closure semantics -- are tested in ``test_update_reevaluation.py`` against
    the real ``VMCTrainer``; nothing here should be read as covering them.

    Parameters
    ----------
    batch : ElectronBatch
        The step's batch, which fixes the required dtype and device.
    recompute : callable, optional
        Recompute to install.  Defaults to a detached zero, which is enough for
        an adapter contract that never calls the re-evaluation.
    """

    if recompute is None:

        def recompute() -> torch.Tensor:
            return torch.zeros((), dtype=batch.dtype, device=batch.device)

    return ObjectiveReevaluation(
        recompute=recompute, dtype=batch.dtype, device=batch.device
    )


def _optimizer_update_input(parameter: torch.nn.Parameter, *, step: int) -> AutogradUpdateInput:
    """Build the same differentiable update input at a given model state."""

    objective = (parameter.square() * 3.0).sum()
    batch = _batch()
    return AutogradUpdateInput(
        batch=batch,
        wavefunction=_output(batch, objective.reshape(1)),
        local_energy=torch.zeros(1, dtype=torch.float64),
        step=step,
        objective=objective,
        reevaluate=_reevaluation(batch),
    )


def _binding(parameter: torch.nn.Parameter) -> ParameterBinding:
    """Build the direct one-slot binding used by score-input tests."""

    slot = ParameterSlot(
        ordinal=0,
        shape=tuple(parameter.shape),
        numel=parameter.numel(),
        dtype=parameter.dtype,
    )
    return ParameterBinding(
        layout=ParameterLayout(slots=(slot,)),
        parameters=(parameter,),
    )


def _assert_nested_equal(left: Any, right: Any) -> None:
    """Compare an optimizer state payload without relying on object identity."""

    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def test_step_records_are_frozen_and_accept_flattened_sample_axes() -> None:
    """The common record follows flattened primal output, not raw sample rank."""

    batch = _batch(sample_shape=(2, 3))
    output = _output(batch)
    data = VMCStepData(
        batch=batch,
        wavefunction=output,
        local_energy=torch.zeros(6, dtype=torch.float64),
    )
    assert data.validate() is data

    with pytest.raises(FrozenInstanceError):
        data.batch = batch  # type: ignore[misc]

    objective = torch.tensor(0.0, dtype=torch.float64)
    autograd_input = AutogradUpdateInput(
        batch=batch,
        wavefunction=output,
        local_energy=data.local_energy,
        step=4,
        objective=objective,
        reevaluate=_reevaluation(batch),
    )
    assert autograd_input.step == 4

    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    binding = _binding(parameter)
    scores = MaterializedParameterLogScores(
        layout=binding.layout,
        blocks=(torch.zeros(6, 2, dtype=torch.float64),),
    )
    score_input = ScoreUpdateInput(
        batch=batch,
        wavefunction=output,
        local_energy=data.local_energy,
        step=4,
        parameter_scores=scores,
        parameter_binding=binding,
    )
    assert score_input.parameter_binding.parameters[0] is parameter


def test_step_records_reject_shape_and_layout_drift() -> None:
    batch = _batch(sample_shape=(2, 3))
    output = _output(batch)

    with pytest.raises(ValueError, match="same shape"):
        VMCStepData(
            batch=batch,
            wavefunction=output,
            local_energy=torch.zeros(2, 3, dtype=torch.float64),
        )

    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    binding = _binding(parameter)
    mismatched_slot = ParameterSlot(
        ordinal=0,
        shape=(3,),
        numel=3,
        dtype=torch.float64,
    )
    mismatched_scores = MaterializedParameterLogScores(
        layout=ParameterLayout(slots=(mismatched_slot,)),
        blocks=(torch.zeros(6, 3, dtype=torch.float64),),
    )
    with pytest.raises(ValueError, match="layouts do not match"):
        ScoreUpdateInput(
            batch=batch,
            wavefunction=output,
            local_energy=torch.zeros(6, dtype=torch.float64),
            step=0,
            parameter_scores=mismatched_scores,
            parameter_binding=binding,
        )


def test_legacy_adapter_matches_current_zero_grad_backward_clip_step_sequence() -> None:
    """An independent current-style sequence pins the adapter's arithmetic."""

    control = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    adapted = torch.nn.Parameter(control.detach().clone())
    control_optimizer = torch.optim.Adam([control], lr=0.1)
    adapted_optimizer = torch.optim.Adam([adapted], lr=0.1)

    control_objective = (control.square() * 3.0).sum()
    control_optimizer.zero_grad(set_to_none=True)
    control_objective.backward()
    torch.nn.utils.clip_grad_norm_([control], 0.5)
    control_grad = control.grad.detach().clone()
    control_optimizer.step()
    control_state = control_optimizer.state_dict()

    adapted_objective = (adapted.square() * 3.0).sum()
    batch = _batch()
    update_input = AutogradUpdateInput(
        batch=batch,
        wavefunction=_output(batch, adapted_objective.reshape(1)),
        local_energy=torch.zeros(1, dtype=torch.float64),
        step=0,
        objective=adapted_objective,
        reevaluate=_reevaluation(batch),
    )
    result = LegacyAutogradUpdate(
        adapted_optimizer,
        gradient_clip_norm=0.5,
        model_parameters=ModelParameterBinding(parameters=(adapted,)),
    ).update(update_input)

    assert result == VMCUpdateResult(applied=True, grad_norm=float(control_grad.norm().item()))
    assert torch.equal(adapted, control)
    assert adapted.grad is not None
    assert torch.equal(adapted.grad, control_grad)
    _assert_nested_equal(adapted_optimizer.state_dict(), control_state)


def test_legacy_adapter_matches_pre_f4_model_gradient_domain_for_subset_optimizer() -> None:
    """A subset optimizer must retain the predecessor model-wide gradient domain."""

    clipped_selected = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    clipped_unselected = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    clipped_optimizer = torch.optim.SGD([clipped_selected], lr=0.1)
    clipped_objective = (clipped_selected + 2.0 * clipped_unselected).sum()
    clipped_batch = _batch()
    clipped_input = AutogradUpdateInput(
        batch=clipped_batch,
        wavefunction=_output(clipped_batch, clipped_objective.reshape(1)),
        local_energy=torch.zeros(1, dtype=torch.float64),
        step=0,
        objective=clipped_objective,
        reevaluate=_reevaluation(clipped_batch),
    )
    clipped_result = LegacyAutogradUpdate(
        clipped_optimizer,
        gradient_clip_norm=0.5,
        model_parameters=ModelParameterBinding(
            parameters=(clipped_selected, clipped_unselected)
        ),
    ).update(clipped_input)

    # These values are the observed pre-F4 trainer result for this fixed
    # objective at the F3 predecessor b2c01970: clip_grad_norm_ operated on
    # both model parameters before the optimizer stepped only ``selected``.
    assert clipped_result.grad_norm == pytest.approx(0.5)
    assert clipped_selected.grad is not None
    assert clipped_unselected.grad is not None
    assert clipped_selected.grad.item() == pytest.approx(0.22360679774997896)
    assert clipped_unselected.grad.item() == pytest.approx(0.4472135954999579)
    assert clipped_selected.item() == pytest.approx(0.9776393202250021)
    assert clipped_unselected.item() == pytest.approx(2.0)

    unclipped_selected = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    unclipped_unselected = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    unclipped_optimizer = torch.optim.SGD([unclipped_selected], lr=0.1)
    unclipped_objective = (unclipped_selected + 2.0 * unclipped_unselected).sum()
    unclipped_batch = _batch()
    unclipped_input = AutogradUpdateInput(
        batch=unclipped_batch,
        wavefunction=_output(unclipped_batch, unclipped_objective.reshape(1)),
        local_energy=torch.zeros(1, dtype=torch.float64),
        step=0,
        objective=unclipped_objective,
        reevaluate=_reevaluation(unclipped_batch),
    )
    unclipped_result = LegacyAutogradUpdate(
        unclipped_optimizer,
        model_parameters=ModelParameterBinding(
            parameters=(unclipped_selected, unclipped_unselected)
        ),
    ).update(unclipped_input)

    # The unclipped norm is also model-wide in the predecessor; a
    # subset-only implementation reports 1.0 here instead of sqrt(5).
    assert unclipped_result.grad_norm == pytest.approx(2.23606797749979)


def test_legacy_adapter_loads_raw_checkpoint_and_preserves_next_update() -> None:
    """Loading an on-disk optimizer payload must affect the next Adam update."""

    control = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    control_optimizer = torch.optim.Adam([control], lr=0.1)
    control_adapter = LegacyAutogradUpdate(
        control_optimizer,
        gradient_clip_norm=0.5,
        model_parameters=ModelParameterBinding(parameters=(control,)),
    )
    control_adapter.update(_optimizer_update_input(control, step=0))

    # This is the raw payload written by the legacy checkpoint callback.
    raw_optimizer_checkpoint = copy.deepcopy(control_optimizer.state_dict())
    assert raw_optimizer_checkpoint["state"]
    first_state = next(iter(raw_optimizer_checkpoint["state"].values()))
    assert first_state["step"].item() == 1.0
    assert first_state["exp_avg"].abs().sum().item() > 0.0
    _assert_nested_equal(control_adapter.state_dict(), raw_optimizer_checkpoint)

    restored = torch.nn.Parameter(control.detach().clone())
    restored_optimizer = torch.optim.Adam([restored], lr=0.1)
    restored_adapter = LegacyAutogradUpdate(
        restored_optimizer,
        gradient_clip_norm=0.5,
        model_parameters=ModelParameterBinding(parameters=(restored,)),
    )
    restored_adapter.load_state_dict(raw_optimizer_checkpoint)
    _assert_nested_equal(restored_adapter.state_dict(), raw_optimizer_checkpoint)

    control_result = control_adapter.update(_optimizer_update_input(control, step=1))
    restored_result = restored_adapter.update(_optimizer_update_input(restored, step=1))

    assert restored_result == control_result
    assert torch.equal(restored, control)
    assert restored.grad is not None
    assert control.grad is not None
    assert torch.equal(restored.grad, control.grad)
    _assert_nested_equal(restored_adapter.state_dict(), control_adapter.state_dict())


def test_legacy_adapter_skips_vacuum_and_errors_for_disconnected_nonvacuum() -> None:
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    adapter = LegacyAutogradUpdate(
        optimizer,
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
    )

    vacuum = _batch(n_electrons=0)
    skipped = adapter.update(
        AutogradUpdateInput(
            batch=vacuum,
            wavefunction=_output(vacuum),
            local_energy=torch.zeros(1, dtype=torch.float64),
            step=0,
            objective=torch.tensor(1.0, dtype=torch.float64),
            reevaluate=_reevaluation(vacuum),
        )
    )
    assert skipped == VMCUpdateResult(applied=False, grad_norm=0.0)
    assert parameter.grad is None

    nonvacuum = _batch()
    with pytest.raises(RuntimeError, match="disconnected"):
        adapter.update(
            AutogradUpdateInput(
                batch=nonvacuum,
                wavefunction=_output(nonvacuum),
                local_energy=torch.zeros(1, dtype=torch.float64),
                step=1,
                objective=torch.tensor(1.0, dtype=torch.float64),
                reevaluate=_reevaluation(nonvacuum),
            )
        )
    assert parameter.grad is None


def test_legacy_adapter_requires_model_binding_before_mutation() -> None:
    """An omitted model binding fails before the adapter can mutate anything."""

    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    optimizer_state_before = copy.deepcopy(optimizer.state_dict())

    with pytest.raises(TypeError, match="model_parameters"):
        LegacyAutogradUpdate(optimizer)

    assert parameter.grad is None
    _assert_nested_equal(optimizer.state_dict(), optimizer_state_before)


def test_trainer_rejects_mismatched_legacy_optimizer_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy ownership mismatch fails before the loop or optimizer can mutate."""

    import tpen.training.trainer as trainer_module

    monkeypatch.setattr(
        trainer_module,
        "local_energy",
        lambda terms, model, batch, return_terms: torch.zeros(batch.batch_size, dtype=batch.dtype),
    )
    model = _OneParameterWavefunction()
    trainer_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    owned_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    update = LegacyAutogradUpdate(
        owned_optimizer,
        model_parameters=ModelParameterBinding(parameters=tuple(model.parameters())),
    )
    trainer = VMCTrainer(max_steps=1, update_method=update)
    context = _StubContext()
    parameter_before = model.weight.detach().clone()
    trainer_optimizer_before = copy.deepcopy(trainer_optimizer.state_dict())
    owned_optimizer_before = copy.deepcopy(owned_optimizer.state_dict())

    with pytest.raises(ValueError, match="ownership"):
        trainer.fit(
            model=model,
            sampler=_OneElectronSampler(),
            hamiltonian_terms=[KineticEnergy()],
            optimizer=trainer_optimizer,
            context=context,
            emit=lambda **_: None,
        )

    assert torch.equal(model.weight, parameter_before)
    assert model.weight.grad is None
    _assert_nested_equal(trainer_optimizer.state_dict(), trainer_optimizer_before)
    _assert_nested_equal(owned_optimizer.state_dict(), owned_optimizer_before)
    # The active non-finite local-energy policy is recorded beside the
    # progress counters, so a reader holding an energy can tell which
    # estimator produced it. Asserted by exact equality on purpose: an
    # unexpected NEW key in trainer state is exactly the kind of thing this
    # test should notice, so it is updated rather than loosened to a subset
    # check.
    assert trainer.state_dict() == {
        "next_iteration": 0,
        "completed_updates": 0,
        "nonfinite_local_energy_policy": "mask",
    }
    assert context.occurrences == []


def test_trainer_state_publishes_the_same_update_state_optimizer() -> None:
    model = _OneParameterWavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    update = LegacyAutogradUpdate(
        optimizer,
        model_parameters=ModelParameterBinding(parameters=tuple(model.parameters())),
    )
    trainer = VMCTrainer(max_steps=0, update_method=update)
    context = _StubContext()

    state = trainer.fit(
        model=model,
        sampler=_OneElectronSampler(),
        hamiltonian_terms=[KineticEnergy()],
        optimizer=optimizer,
        context=context,
        emit=lambda **_: None,
    )

    assert state.update_state is not None
    assert state.update_state.optimizer is optimizer
    assert state.optimizer is state.update_state.optimizer
    assert all(
        left is right
        for left, right in zip(
            state.update_state.model_parameters.parameters,
            tuple(model.parameters()),
            strict=True,
        )
    )


def _checkpoint_context(tmp_path) -> SimpleNamespace:
    """Build the same public context boundary used by checkpoint tests."""

    return SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "model": {"name": "linear"},
                "optimizer": {"name": "adam"},
                "trainer": {"name": "vmc"},
                "sampler": {"name": "metropolis"},
                "hamiltonian_terms": {"constant": {}},
            }
        ),
        metadata=SimpleNamespace(device="cpu", dtype="float64"),
        run_dir=tmp_path,
    )


def _resume_checkpoint(tmp_path):
    """Save a real public train-resume checkpoint with updater layout state."""

    model = torch.nn.Linear(1, 1).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    update = LegacyAutogradUpdate(
        optimizer,
        model_parameters=ModelParameterBinding.from_parameters(tuple(model.parameters())),
    )
    trainer = VMCTrainer(max_steps=2, update_method=update)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    trainer.next_iteration = 1
    trainer.completed_updates = 1
    sampler = MetropolisSampler(
        n_walkers=2,
        burn_in=0,
        n_steps=1,
        n_electrons=1,
        spatial_dim=1,
        seed=7,
        dtype=torch.float64,
    )
    checkpoint = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=1,
        completed_updates=1,
        model=model,
        context=_checkpoint_context(tmp_path),
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
        payload=TrainResume(),
    )
    return checkpoint


def test_public_save_restore_rebuilds_direct_binding_after_model_restore(tmp_path) -> None:
    """Public save/restore retains layout and rebinds the updater to live params."""

    checkpoint = _resume_checkpoint(tmp_path)
    model = torch.nn.Linear(1, 1).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    class _RebindingLegacyUpdate(LegacyAutogradUpdate):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rebind_calls = 0

        def rebind_model_parameters(self, model_parameters):
            self.rebind_calls += 1
            super().rebind_model_parameters(model_parameters)

    update = _RebindingLegacyUpdate(
        optimizer,
        model_parameters=ModelParameterBinding.from_parameters(tuple(model.parameters())),
    )
    trainer = VMCTrainer(max_steps=2, update_method=update)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    sampler = MetropolisSampler(
        n_walkers=2,
        burn_in=0,
        n_steps=1,
        n_electrons=1,
        spatial_dim=1,
        seed=11,
        dtype=torch.float64,
    )

    report = restore_checkpoint(
        load={"mode": "train_resume", "path": str(checkpoint)},
        model=model,
        context=_checkpoint_context(tmp_path),
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
    )

    assert report.loaded_model and report.loaded_optimizer and report.loaded_trainer
    assert update.rebind_calls == 1
    assert all(
        left is right
        for left, right in zip(
            update.model_parameters.parameters,
            tuple(model.parameters()),
            strict=True,
        )
    )
    assert trainer.state_dict()["parameter_layout"] == json.loads(
        (checkpoint / "trainer.json").read_text()
    )["parameter_layout"]


def test_restore_rejects_incompatible_parameter_layout(tmp_path) -> None:
    """A recorded layout mismatch fails rather than silently resuming."""

    checkpoint = _resume_checkpoint(tmp_path)
    trainer_path = checkpoint / "trainer.json"
    trainer_state = json.loads(trainer_path.read_text())
    trainer_state["parameter_layout"]["slots"][0]["dtype"] = "torch.float32"
    trainer_path.write_text(json.dumps(trainer_state), encoding="utf-8")
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"]["trainer_sha256"] = file_sha256(trainer_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    model = torch.nn.Linear(1, 1).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    update = LegacyAutogradUpdate(
        optimizer,
        model_parameters=ModelParameterBinding.from_parameters(tuple(model.parameters())),
    )
    trainer = VMCTrainer(max_steps=2, update_method=update)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    sampler = MetropolisSampler(
        n_walkers=2,
        burn_in=0,
        n_steps=1,
        n_electrons=1,
        spatial_dim=1,
        seed=11,
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="parameter layout"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(checkpoint)},
            model=model,
            context=_checkpoint_context(tmp_path),
            optimizer=optimizer,
            trainer=trainer,
            sampler=sampler,
        )

    assert trainer.state_dict()["next_iteration"] == 0
    assert trainer.state_dict()["completed_updates"] == 0


def test_live_update_inputs_cannot_be_serialized() -> None:
    batch = _batch()
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    objective = parameter.square().sum()
    update_input = AutogradUpdateInput(
        batch=batch,
        wavefunction=_output(batch, objective.reshape(1)),
        local_energy=torch.zeros(1, dtype=torch.float64),
        step=0,
        objective=objective,
        reevaluate=_reevaluation(batch),
    )

    with pytest.raises(RuntimeError, match="graph-bearing"):
        pickle.dumps(update_input)

    with pytest.raises(RuntimeError, match="graph-bearing"):
        torch.save(update_input, BytesIO())


class _RecordingUpdate(VMCUpdateMethod[AutogradUpdateInput]):
    """Test updater proving that the trainer delegates an exact input record."""

    def __init__(self) -> None:
        self.received: AutogradUpdateInput | None = None

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        self.received = update_input
        return VMCUpdateResult(applied=True, grad_norm=2.5)


class _OneElectronWalkers:
    def make_batch(self) -> ElectronBatch:
        return _batch()


class _OneElectronSampler:
    def collect_samples(self, model, *, device=None):
        del model, device
        return _OneElectronWalkers(), None


class _OneParameterWavefunction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        value = self.weight.expand(batch.batch_size)
        return WavefunctionOutput(logabs=value, sign=torch.ones_like(value))


def test_trainer_delegates_without_publishing_live_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """The update record stays local while counters and completion events remain."""

    import tpen.training.trainer as trainer_module

    monkeypatch.setattr(
        trainer_module,
        "local_energy",
        lambda terms, model, batch, return_terms: torch.zeros(batch.batch_size, dtype=batch.dtype),
    )
    update = _RecordingUpdate()
    trainer = VMCTrainer(max_steps=1, update_method=update)
    context = _StubContext()
    state = trainer.fit(
        model=_OneParameterWavefunction(),
        sampler=_OneElectronSampler(),
        hamiltonian_terms=[KineticEnergy()],
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))], lr=0.1),
        context=context,
        emit=lambda **_: None,
    )

    assert update.received is not None
    assert trainer.state_dict()["next_iteration"] == 1
    assert trainer.state_dict()["completed_updates"] == 1
    assert "parameter_layout" in trainer.state_dict()
    assert all(value is not update.received for value in vars(state).values())
    assert state.wavefunction_output is update.received.wavefunction
