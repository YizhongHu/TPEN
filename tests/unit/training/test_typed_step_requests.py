"""Contract, oracle, and negative tests for typed VMC step requests."""

from __future__ import annotations

import math
import pickle
from dataclasses import FrozenInstanceError
from io import BytesIO
from typing import Any

import pytest
import torch
import typeguard
from typeguard import TypeCheckError

import tpen.training as training_pkg
import tpen.training.step_requests as step_requests_module
import tpen.training.trainer as trainer_module
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext
from tpen.data.batch import (
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    ParameterScoreForwardPacket,
    ParameterSlot,
    WavefunctionOutput,
)
from tpen.data.permutation import Permutation
from tpen.events import Ended, Started
from tpen.nn.activation import SoftplusBeta
from tpen.nn.forward import (
    CoordinateGradientRequest,
    MaterializedParameterScoreRequest,
)
from tpen.physics.hamiltonian import LocalEnergyResult, local_energy
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import HarmonicTrap
from tpen.training.events import Forward, LocalEnergy, Objective
from tpen.training.block_ng import BlockDiagonalNaturalGradientUpdate, BlockNGPolicy
from tpen.training.qgt import DampingPolicy
from tpen.training.score_geometry import ScoreConventions
from tpen.training.spring import SPRINGPolicy, SPRINGUpdate
from tpen.training.sr import SRPolicy, StochasticReconfigurationUpdate
from tpen.training.state import TrainerState
from tpen.training.step_requests import (
    AutogradStepProducer,
    AutogradStepRequest,
    HamiltonianTangentStepProducer,
    HamiltonianTangentStepRequest,
    HamiltonianTangentUpdateInput,
    MaterializedHamiltonianTangentStepRequest,
    MaterializedLocalEnergyDerivatives,
    ParameterScoreStepProducer,
    ParameterScoreStepRequest,
    ProducedVMCStep,
    VMCStepProducer,
    VMCStepRequest,
    evaluate_hamiltonian_tangent_step,
    resolve_step_request,
)
from tpen.training.trainer import VMCTrainer, _TrainerStepProducer
from tpen.training.update import (
    AutogradUpdateInput,
    ModelParameterBinding,
    ScoreUpdateInput,
    VMCUpdateMethod,
    VMCUpdateResult,
)
from tpen.training.vmc import VMCObjectiveResult


def _scoped_operation_types(context: _StubContext) -> set[type[object]]:
    """Return concrete operation types observed across Started/Ended boundaries."""

    return {
        type(occ.event.operation)
        for occ in context.occurrences
        if isinstance(occ.event, (Started, Ended))
    }


def _build_parameter_binding(parameters: tuple[torch.nn.Parameter, ...]) -> ParameterBinding:
    """Build a direct live ParameterBinding in exact tuple order."""

    slots = tuple(
        ParameterSlot(
            ordinal=ordinal,
            shape=tuple(parameter.shape),
            numel=parameter.numel(),
            dtype=parameter.dtype,
        )
        for ordinal, parameter in enumerate(parameters)
    )
    return ParameterBinding(layout=ParameterLayout(slots=slots), parameters=parameters)


def _sample_batch(
    *,
    sample_shape: tuple[int, ...] = (4,),
    n_electrons: int = 2,
    spatial_dim: int = 2,
    seed: int = 17,
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> ElectronBatch:
    """Build a deterministic electron batch with explicit spin labels."""

    generator = torch.Generator().manual_seed(seed)
    positions = torch.randn(
        (*sample_shape, n_electrons, spatial_dim),
        generator=generator,
        dtype=dtype,
    )
    if requires_grad:
        positions = positions.requires_grad_(True)
    if n_electrons == 0:
        spins = torch.zeros((*sample_shape, 0), dtype=dtype)
    else:
        spin_row = torch.tensor(
            [1.0 if idx % 2 == 0 else -1.0 for idx in range(n_electrons)],
            dtype=dtype,
        )
        spins = spin_row.expand(*sample_shape, n_electrons).clone()
    return ElectronBatch(positions=positions, spins=spins)


class _FixedWalkers:
    def __init__(self, batch: ElectronBatch) -> None:
        self._batch = batch

    def make_batch(self) -> ElectronBatch:
        return self._batch


class _FixedSampler:
    def __init__(self, batch: ElectronBatch) -> None:
        self._batch = batch
        self.collect_calls = 0

    def collect_samples(self, model: Any, *, reset: bool = False, device: Any = None, emitter=None):
        del model, reset, device, emitter
        self.collect_calls += 1
        return _FixedWalkers(self._batch), None


class _SoftplusWavefunction(torch.nn.Module):
    """Smooth float64 wavefunction exercising SoftplusBeta triple-backward."""

    def __init__(
        self,
        *,
        spatial_dim: int = 2,
        include_disconnected: bool = False,
        include_logabs_only_shift: bool = False,
        return_phase: bool = False,
        constant_logabs: bool = False,
    ) -> None:
        super().__init__()
        self.linear_weight = torch.nn.Parameter(
            torch.tensor([[0.45, -0.30], [0.25, 0.55]], dtype=torch.float64)[:, :spatial_dim]
        )
        self.linear_bias = torch.nn.Parameter(torch.tensor([0.15, -0.20], dtype=torch.float64))
        self.envelope_scale = torch.nn.Parameter(torch.tensor(0.35, dtype=torch.float64))
        self.shift = (
            torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
            if include_logabs_only_shift
            else None
        )
        self.disconnected = (
            torch.nn.Parameter(torch.tensor([[1.0, -2.0]], dtype=torch.float64))
            if include_disconnected
            else None
        )
        self.activation = SoftplusBeta(beta=2.0)
        self.return_phase = return_phase
        self.constant_logabs = constant_logabs
        self.forward_calls = 0
        self._parameter_binding = _build_parameter_binding(
            tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
        )

    @property
    def parameter_binding(self) -> ParameterBinding:
        return self._parameter_binding

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        self.forward_calls += 1
        flat = batch.flatten_samples()
        if self.constant_logabs:
            logabs = torch.zeros((flat.batch_size,), dtype=flat.dtype, device=flat.device)
        elif flat.n_electrons == 0:
            # Depend on parameters and zero-electron positions so KineticEnergy
            # autograd differentiation sees a connected graph in the vacuum sector.
            logabs = (self.envelope_scale + self.linear_bias.sum()).expand(
                flat.batch_size
            ) + 0.0 * flat.positions.sum(dim=(1, 2))
        else:
            # Shape: [B, N_e, 2]
            projected = torch.matmul(flat.positions, self.linear_weight.transpose(0, 1)) + self.linear_bias
            activated = self.activation(projected)
            radial = flat.positions.square().sum(dim=(1, 2))
            logabs = activated.sum(dim=(1, 2)) - self.envelope_scale * radial
            if self.shift is not None:
                logabs = logabs + self.shift
        sign = torch.ones_like(logabs)
        phase = torch.zeros_like(logabs) if self.return_phase else None
        return WavefunctionOutput(logabs=logabs, sign=sign, phase=phase)

    def evaluate_materialized_parameter_score_request(
        self,
        *,
        request: MaterializedParameterScoreRequest,
        batch: ElectronBatch,
        graph_bearing: bool = False,
    ) -> ParameterScoreForwardPacket:
        del request
        output = self.forward(batch)
        parameters = self.parameter_binding.parameters
        blocks_per_param: list[list[torch.Tensor]] = [[] for _ in parameters]
        for idx in range(output.logabs.shape[0]):
            grads = torch.autograd.grad(
                output.logabs[idx],
                parameters,
                retain_graph=(idx + 1 < output.logabs.shape[0]) or graph_bearing,
                create_graph=graph_bearing,
                allow_unused=True,
            )
            for p_idx, (param, grad) in enumerate(zip(parameters, grads, strict=True)):
                blocks_per_param[p_idx].append(
                    param.new_zeros(param.shape)
                    if grad is None
                    else (grad if graph_bearing else grad.detach())
                )
        scores = MaterializedParameterLogScores(
            layout=self.parameter_binding.layout,
            blocks=tuple(torch.stack(rows, dim=0) for rows in blocks_per_param),
        )
        if not graph_bearing:
            output = WavefunctionOutput(
                logabs=output.logabs.detach(),
                sign=output.sign.detach(),
                phase=None if output.phase is None else output.phase.detach(),
            )
        return ParameterScoreForwardPacket(output=output, parameter_scores=scores)


class _RecordingAutogradMethod(VMCUpdateMethod[AutogradUpdateInput]):
    def __init__(self) -> None:
        self.received: AutogradUpdateInput | None = None

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        assert isinstance(update_input, AutogradUpdateInput)
        self.received = update_input
        return VMCUpdateResult(applied=True, grad_norm=0.0)


class _RecordingScoreMethod(VMCUpdateMethod[ScoreUpdateInput]):
    def __init__(self, *, chunk_size: int | None = None) -> None:
        self._chunk_size = chunk_size
        self.received: ScoreUpdateInput | None = None

    def forward_request(self) -> MaterializedParameterScoreRequest:
        return MaterializedParameterScoreRequest(chunk_size=self._chunk_size)

    def update(self, update_input: ScoreUpdateInput) -> VMCUpdateResult:
        assert isinstance(update_input, ScoreUpdateInput)
        self.received = update_input
        return VMCUpdateResult(applied=True, grad_norm=0.0)


class _RecordingTangentMethod(VMCUpdateMethod[HamiltonianTangentUpdateInput]):
    def __init__(self, *, chunk_size: int | None = None) -> None:
        self._chunk_size = chunk_size
        self.received: HamiltonianTangentUpdateInput | None = None

    def step_request(self) -> MaterializedHamiltonianTangentStepRequest:
        return MaterializedHamiltonianTangentStepRequest(chunk_size=self._chunk_size)

    def update(self, update_input: HamiltonianTangentUpdateInput) -> VMCUpdateResult:
        assert isinstance(update_input, HamiltonianTangentUpdateInput)
        self.received = update_input
        return VMCUpdateResult(applied=True, grad_norm=0.0)


def test_public_exports_and_nominal_request_contracts() -> None:
    """All step-request symbols are exported and satisfy nominal/protocol contracts."""

    for name in step_requests_module.__all__:
        assert hasattr(training_pkg, name), f"tpen.training must export {name}"

    autograd_req = AutogradStepRequest()
    score_fwd = MaterializedParameterScoreRequest(chunk_size=3)
    score_req = ParameterScoreStepRequest(forward_request=score_fwd)
    tangent_req = MaterializedHamiltonianTangentStepRequest(chunk_size=2)

    assert isinstance(autograd_req, VMCStepRequest)
    assert isinstance(score_req, VMCStepRequest)
    assert isinstance(tangent_req, VMCStepRequest)
    assert isinstance(tangent_req, HamiltonianTangentStepRequest)

    assert autograd_req.describe_label() == "value"
    assert (
        score_req.describe_label()
        == "tpen.nn.forward.MaterializedParameterScoreRequest"
    )
    assert (
        tangent_req.describe_label()
        == "tpen.training.step_requests.MaterializedHamiltonianTangentStepRequest"
    )

    with pytest.raises(FrozenInstanceError):
        tangent_req.chunk_size = 4  # type: ignore[misc]

    assert _RecordingAutogradMethod().describe().forward_request == "value"
    assert (
        _RecordingScoreMethod().describe().forward_request
        == "tpen.nn.forward.MaterializedParameterScoreRequest"
    )
    assert (
        _RecordingTangentMethod().describe().forward_request
        == "tpen.training.step_requests.MaterializedHamiltonianTangentStepRequest"
    )


def test_narrow_protocols_and_double_dispatch() -> None:
    """Each request dispatches strictly to its narrow method-only producer protocol."""

    dummy_output = WavefunctionOutput(logabs=torch.zeros(1), sign=torch.ones(1))
    dummy_obj = VMCObjectiveResult(loss=torch.tensor(0.0), metrics={})
    dummy_step: ProducedVMCStep[Any] = ProducedVMCStep(
        update_input="payload",
        objective=dummy_obj,
        local_energy=torch.zeros(1),
        term_energies=None,
        wavefunction_output=dummy_output,
    )

    class _OnlyAutograd:
        def __init__(self) -> None:
            self.seen = None

        def produce_autograd_step(self, request: AutogradStepRequest):
            self.seen = request
            return dummy_step

    class _OnlyScore:
        def __init__(self) -> None:
            self.seen = None

        def produce_parameter_score_step(self, request: ParameterScoreStepRequest):
            self.seen = request
            return dummy_step

    class _OnlyTangent:
        def __init__(self) -> None:
            self.seen = None

        def produce_materialized_hamiltonian_tangent_step(
            self, request: MaterializedHamiltonianTangentStepRequest
        ):
            self.seen = request
            return dummy_step

    p_auto = _OnlyAutograd()
    p_score = _OnlyScore()
    p_tangent = _OnlyTangent()

    assert isinstance(p_auto, AutogradStepProducer)
    assert not isinstance(p_auto, VMCStepProducer)
    assert isinstance(p_score, ParameterScoreStepProducer)
    assert isinstance(p_tangent, HamiltonianTangentStepProducer)

    req_auto = AutogradStepRequest()
    req_score = ParameterScoreStepRequest(forward_request=MaterializedParameterScoreRequest())
    req_tangent = MaterializedHamiltonianTangentStepRequest()

    assert req_auto.produce(p_auto) is dummy_step
    assert p_auto.seen is req_auto
    assert req_score.produce(p_score) is dummy_step
    assert p_score.seen is req_score
    assert req_tangent.produce(p_tangent) is dummy_step
    assert p_tangent.seen is req_tangent

    trainer = VMCTrainer(max_steps=1)
    producer_model = _SoftplusWavefunction()
    producer_optimizer = torch.optim.SGD(producer_model.parameters(), lr=0.01)
    producer_state = TrainerState(
        model=producer_model,
        optimizer=producer_optimizer,
        trainer=trainer,
    )
    trainer_producer = _TrainerStepProducer(
        trainer=trainer,
        model=producer_model,
        hamiltonian_terms=[KineticEnergy()],
        batch=_sample_batch(sample_shape=(2,)),
        step=0,
        context=_StubContext(),
        state=producer_state,
    )
    assert isinstance(trainer_producer, VMCStepProducer)


def test_autograd_step_request_liveness_identity_and_bitwise_parity() -> None:
    """AutogradStepRequest preserves live loss_has_grad, wavefunction is-identity, and bitwise steps."""

    batch = _sample_batch(sample_shape=(3,), n_electrons=2, spatial_dim=2, seed=21)

    # Recording method: inspect liveness and exact object identity.
    model = _SoftplusWavefunction()
    recorder = _RecordingAutogradMethod()
    context = _StubContext()
    state = VMCTrainer(max_steps=1, return_terms=True, update_method=recorder).fit(
        model=model,
        sampler=_FixedSampler(batch),
        hamiltonian_terms=[KineticEnergy(), HarmonicTrap(omega=0.5)],
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        context=context,
        emit=lambda **_: None,
    )
    assert recorder.received is not None
    assert recorder.received.objective.requires_grad is True
    assert recorder.received.local_energy.requires_grad is False
    assert state.wavefunction_output is recorder.received.wavefunction
    assert state.loss.requires_grad is False
    assert state.local_energy.requires_grad is False
    train_metrics = [m for ns, m in context.records if ns == "train"][-1]
    assert train_metrics["loss_has_grad"] is True
    phase_types = _scoped_operation_types(context)
    assert {LocalEnergy, Forward, Objective}.issubset(phase_types)

    # Bitwise parity against manual LegacyAutogradUpdate step.
    model_a = _SoftplusWavefunction()
    model_b = _SoftplusWavefunction()
    model_b.load_state_dict(model_a.state_dict())
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.05)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.05)
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5)]

    VMCTrainer(max_steps=1).fit(
        model=model_a,
        sampler=_FixedSampler(batch),
        hamiltonian_terms=terms,
        optimizer=opt_a,
        context=_StubContext(),
        emit=lambda **_: None,
    )

    eloc = local_energy(terms, model_b, batch, return_terms=False).detach()
    out_b = model_b(batch)
    centered = eloc - eloc.mean()
    loss_b = 2.0 * torch.mean(centered * out_b.logabs)
    opt_b.zero_grad(set_to_none=True)
    loss_b.backward()
    opt_b.step()

    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(pa.detach(), pb.detach())
        assert pa.grad is not None and pb.grad is not None
        assert torch.equal(pa.grad, pb.grad)


def test_parameter_score_step_request_detaches_all_tensors_and_preserves_parity() -> None:
    """ParameterScoreStepRequest detaches even graph-bearing packets and preserves values."""

    batch = _sample_batch(sample_shape=(3,), n_electrons=2, spatial_dim=2, seed=29)
    model = _SoftplusWavefunction()
    original_eval = model.evaluate_materialized_parameter_score_request

    # Force a graph-bearing packet to verify producer detaches everything.
    def graph_bearing_eval(*, request: MaterializedParameterScoreRequest, batch: ElectronBatch):
        return original_eval(request=request, batch=batch, graph_bearing=True)

    model.evaluate_materialized_parameter_score_request = graph_bearing_eval  # type: ignore[method-assign]

    recorder = _RecordingScoreMethod(chunk_size=2)
    context = _StubContext()
    state = VMCTrainer(max_steps=1, return_terms=True, update_method=recorder).fit(
        model=model,
        sampler=_FixedSampler(batch),
        hamiltonian_terms=[KineticEnergy(), HarmonicTrap(omega=0.5)],
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        context=context,
        emit=lambda **_: None,
    )

    assert recorder.received is not None
    received = recorder.received
    assert not received.local_energy.requires_grad
    assert not received.wavefunction.logabs.requires_grad
    assert not received.wavefunction.sign.requires_grad
    assert all(not block.requires_grad for block in received.parameter_scores.blocks)
    assert not state.loss.requires_grad
    train_metrics = [m for ns, m in context.records if ns == "train"][-1]
    assert train_metrics["loss_has_grad"] is False

    # Compare against direct reference evaluation.
    ref_packet = original_eval(
        request=MaterializedParameterScoreRequest(chunk_size=2),
        batch=batch,
        graph_bearing=False,
    )
    ok, _ = received.parameter_scores.compare(ref_packet.parameter_scores, atol=0.0, rtol=0.0)
    assert ok
    assert torch.equal(received.wavefunction.logabs, ref_packet.as_output().logabs)


@pytest.mark.parametrize("route", ["sr", "spring", "block_ng"])
def test_parameter_score_routes_receive_the_psiformer_clipped_energy(
    route: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SR-family routes all consume the trainer's one transformed score payload.

    SR, SPRING, and block-NG intentionally share ``ParameterScoreStepRequest``.
    Exercising each concrete method's request keeps that ownership boundary
    explicit: a route cannot silently fall back to the raw local energy.
    """

    model = _SoftplusWavefunction()
    parameters = tuple(model.parameters())
    binding = ModelParameterBinding.from_parameters(parameters)
    optimizer = torch.optim.SGD(parameters, lr=0.05)
    if route == "sr":
        method = StochasticReconfigurationUpdate(
            optimizer,
            model_parameters=binding,
            policy=SRPolicy(
                solve_space="parameter",
                damping=DampingPolicy(absolute=0.0, relative=1.0e-2),
                learning_rate=0.05,
            ),
            conventions=ScoreConventions(solve_dtype=torch.float64),
        )
    elif route == "spring":
        method = SPRINGUpdate(
            optimizer,
            model_parameters=binding,
            policy=SPRINGPolicy(
                base=SRPolicy(
                    solve_space="sample",
                    damping=DampingPolicy(absolute=0.0, relative=1.0e-2),
                    learning_rate=0.05,
                ),
                history_decay=0.5,
            ),
            conventions=ScoreConventions(solve_dtype=torch.float64),
        )
    else:
        method = BlockDiagonalNaturalGradientUpdate(
            optimizer,
            model_parameters=binding,
            policy=BlockNGPolicy(damping=1.0e-2, learning_rate=0.05),
        )

    batch = _sample_batch(sample_shape=(4,), seed=73)
    raw_energy = torch.tensor([0.0, 1.0, 2.0, 100.0], dtype=torch.float64)
    monkeypatch.setattr(
        trainer_module,
        "local_energy",
        lambda *args, **kwargs: raw_energy.clone(),
    )
    trainer = VMCTrainer(
        max_steps=1,
        update_method=method,
        energy_clipping_mode="psiformer",
        energy_clipping_k=1.0,
    )
    state = TrainerState(model=model, optimizer=optimizer, trainer=trainer)
    producer = _TrainerStepProducer(
        trainer=trainer,
        model=model,
        hamiltonian_terms=[],
        batch=batch,
        step=0,
        context=_StubContext(),
        state=state,
    )

    produced = producer.produce_parameter_score_step(resolve_step_request(method))
    assert torch.equal(
        produced.update_input.local_energy,
        torch.tensor([0.0, 1.0, 2.0, 26.75], dtype=torch.float64),
    )


def test_linear_method_route_refuses_opt_in_energy_clipping() -> None:
    """The Hamiltonian-tangent route rejects clipping before it can diverge."""

    model = _SoftplusWavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    trainer = VMCTrainer(max_steps=1, energy_clipping_mode="psiformer")
    state = TrainerState(model=model, optimizer=optimizer, trainer=trainer)
    producer = _TrainerStepProducer(
        trainer=trainer,
        model=model,
        hamiltonian_terms=[],
        batch=_sample_batch(sample_shape=(2,), seed=79),
        step=0,
        context=_StubContext(),
        state=state,
    )

    with pytest.raises(ValueError, match="energy clipping is unsupported for the Linear Method"):
        producer.produce_materialized_hamiltonian_tangent_step(
            MaterializedHamiltonianTangentStepRequest()
        )


def test_tangent_flattened_1d_shapes_chunking_return_terms_and_edge_batches() -> None:
    """1D flattened shapes hold across single/multi-chunk, return_terms, empty, and vacuum batches."""

    model = _SoftplusWavefunction()
    terms = [KineticEnergy(), HarmonicTrap(omega=0.6)]
    batch_2d = _sample_batch(sample_shape=(2, 3), n_electrons=2, spatial_dim=2, seed=31)

    single_input, single_terms = evaluate_hamiltonian_tangent_step(
        model=model,
        hamiltonian_terms=terms,
        batch=batch_2d,
        step=3,
        chunk_size=None,
        return_terms=False,
    )
    assert single_terms is None
    assert single_input.step == 3
    assert single_input.wavefunction.logabs.shape == (6,)
    assert single_input.wavefunction.sign.shape == (6,)
    assert single_input.wavefunction.aux == {}
    assert single_input.local_energy.shape == (6,)
    assert single_input.parameter_scores.sample_shape == (6,)
    assert single_input.local_energy_derivatives.sample_shape == (6,)
    assert not single_input.local_energy.requires_grad
    assert not single_input.wavefunction.logabs.requires_grad
    assert all(not b.requires_grad for b in single_input.parameter_scores.blocks)
    assert all(not b.requires_grad for b in single_input.local_energy_derivatives.blocks)

    # Multi-chunk with both request and trainer chunk sizes (effective = min(4, 2) = 2).
    chunk_sizes_seen: list[int] = []

    def tracking_local_energy(h_terms, wf, chunk, *, return_terms=False):
        chunk_sizes_seen.append(chunk.batch_size)
        return local_energy(h_terms, wf, chunk, return_terms=return_terms)

    multi_input, multi_terms = evaluate_hamiltonian_tangent_step(
        model=model,
        hamiltonian_terms=terms,
        batch=batch_2d,
        step=3,
        chunk_size=4,
        trainer_chunk_size=2,
        return_terms=True,
        local_energy_fn=tracking_local_energy,
    )
    assert chunk_sizes_seen == [2, 2, 2]
    assert multi_terms is not None
    assert set(multi_terms) == {"kinetic_energy", "harmonic_trap"}
    for tensor in multi_terms.values():
        assert tensor.shape == (6,)
        assert not tensor.requires_grad
    assert torch.allclose(multi_input.local_energy, single_input.local_energy, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        multi_input.wavefunction.logabs, single_input.wavefunction.logabs, atol=1e-12, rtol=1e-12
    )
    scores_ok, _ = multi_input.parameter_scores.compare(
        single_input.parameter_scores, atol=1e-12, rtol=1e-12
    )
    derivs_ok, _ = multi_input.local_energy_derivatives.compare(
        single_input.local_energy_derivatives, atol=1e-12, rtol=1e-12
    )
    assert scores_ok and derivs_ok

    # Empty batch (batch_size == 0) preserves per-slot dtype and 0-row shapes.
    empty_batch = _sample_batch(sample_shape=(0, 3), n_electrons=2, spatial_dim=2)
    empty_input, empty_terms = evaluate_hamiltonian_tangent_step(
        model=model,
        hamiltonian_terms=terms,
        batch=empty_batch,
        return_terms=True,
    )
    assert empty_terms == {}
    assert empty_input.wavefunction.logabs.shape == (0,)
    assert empty_input.local_energy.shape == (0,)
    assert empty_input.parameter_scores.sample_shape == (0,)
    assert empty_input.local_energy_derivatives.sample_shape == (0,)
    for slot, s_block, d_block in zip(
        model.parameter_binding.layout.slots,
        empty_input.parameter_scores.blocks,
        empty_input.local_energy_derivatives.blocks,
        strict=True,
    ):
        assert s_block.shape == (0, *slot.shape)
        assert d_block.shape == (0, *slot.shape)
        assert s_block.dtype == slot.dtype
        assert d_block.dtype == slot.dtype

    # Through VMCTrainer, empty batch reaches compute_vmc_objective and raises loudly.
    with pytest.raises(ValueError, match="no finite local-energy samples"):
        VMCTrainer(max_steps=1, update_method=_RecordingTangentMethod()).fit(
            model=model,
            sampler=_FixedSampler(empty_batch),
            hamiltonian_terms=terms,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            context=_StubContext(),
            emit=lambda **_: None,
        )

    # Vacuum batch (n_electrons == 0, batch_size == 3).
    vacuum_batch = _sample_batch(sample_shape=(3,), n_electrons=0, spatial_dim=2)
    vac_input, vac_terms = evaluate_hamiltonian_tangent_step(
        model=model,
        hamiltonian_terms=terms,
        batch=vacuum_batch,
        return_terms=False,
    )
    assert vac_terms is None
    assert vac_input.wavefunction.logabs.shape == (3,)
    assert vac_input.local_energy.shape == (3,)
    assert vac_input.parameter_scores.sample_shape == (3,)
    assert vac_input.local_energy_derivatives.sample_shape == (3,)
    for d_block in vac_input.local_energy_derivatives.blocks:
        assert torch.equal(d_block, torch.zeros_like(d_block))

    # Incoming positions with requires_grad=True under pure potential must be detached.
    grad_batch = _sample_batch(sample_shape=(3,), n_electrons=2, spatial_dim=2, requires_grad=True)
    pot_input, _ = evaluate_hamiltonian_tangent_step(
        model=model,
        hamiltonian_terms=[HarmonicTrap(omega=0.8)],
        batch=grad_batch,
    )
    assert not pot_input.batch.positions.requires_grad
    for d_block in pot_input.local_energy_derivatives.blocks:
        assert torch.equal(d_block, torch.zeros_like(d_block))


def test_softplus_beta_triple_backward_central_finite_difference_oracle() -> None:
    """Scores and local-energy derivatives match float64 central finite differences."""

    model = _SoftplusWavefunction(spatial_dim=2)
    terms = [KineticEnergy(), HarmonicTrap(omega=0.7)]
    batch = _sample_batch(sample_shape=(3,), n_electrons=2, spatial_dim=2, seed=41)

    analytic_input, _ = evaluate_hamiltonian_tangent_step(
        model=model,
        hamiltonian_terms=terms,
        batch=batch,
        chunk_size=2,
        return_terms=True,
    )

    eps = 1.0e-6
    parameters = model.parameter_binding.parameters
    fd_score_blocks: list[torch.Tensor] = []
    fd_eloc_blocks: list[torch.Tensor] = []

    for parameter in parameters:
        fd_score = torch.empty((batch.batch_size, *parameter.shape), dtype=parameter.dtype)
        fd_eloc = torch.empty((batch.batch_size, *parameter.shape), dtype=parameter.dtype)
        flat_param = parameter.data.view(-1)
        flat_score = fd_score.view(batch.batch_size, -1)
        flat_eloc = fd_eloc.view(batch.batch_size, -1)
        for index in range(flat_param.numel()):
            original = float(flat_param[index].item())
            with torch.no_grad():
                flat_param[index] = original + eps
            res_plus = local_energy(terms, model, batch, return_terms=True)
            assert isinstance(res_plus, LocalEnergyResult)
            assert res_plus.wavefunction_output is not None
            eloc_plus = res_plus.total.detach()
            logabs_plus = res_plus.wavefunction_output.logabs.detach()

            with torch.no_grad():
                flat_param[index] = original - eps
            res_minus = local_energy(terms, model, batch, return_terms=True)
            assert isinstance(res_minus, LocalEnergyResult)
            assert res_minus.wavefunction_output is not None
            eloc_minus = res_minus.total.detach()
            logabs_minus = res_minus.wavefunction_output.logabs.detach()

            with torch.no_grad():
                flat_param[index] = original

            flat_score[:, index] = (logabs_plus - logabs_minus) / (2.0 * eps)
            flat_eloc[:, index] = (eloc_plus - eloc_minus) / (2.0 * eps)

        fd_score_blocks.append(fd_score)
        fd_eloc_blocks.append(fd_eloc)

    fd_scores = MaterializedParameterLogScores(
        layout=model.parameter_binding.layout,
        blocks=tuple(fd_score_blocks),
    )
    fd_derivs = MaterializedLocalEnergyDerivatives(
        layout=model.parameter_binding.layout,
        blocks=tuple(fd_eloc_blocks),
    )

    scores_close, score_metrics = analytic_input.parameter_scores.compare(
        fd_scores, atol=1.0e-7, rtol=1.0e-6
    )
    derivs_close, deriv_metrics = analytic_input.local_energy_derivatives.compare(
        fd_derivs, atol=1.0e-7, rtol=1.0e-6
    )
    assert scores_close, f"score FD mismatch: {score_metrics}"
    assert derivs_close, f"local-energy derivative FD mismatch: {deriv_metrics}"

    # Verify end-to-end through VMCTrainer with no synthetic Forward scope.
    recorder = _RecordingTangentMethod(chunk_size=2)
    context = _StubContext()
    VMCTrainer(max_steps=1, return_terms=True, update_method=recorder).fit(
        model=model,
        sampler=_FixedSampler(batch),
        hamiltonian_terms=terms,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        context=context,
        emit=lambda **_: None,
    )
    assert recorder.received is not None
    phase_types = _scoped_operation_types(context)
    assert LocalEnergy in phase_types
    assert Objective in phase_types
    assert Forward not in phase_types
    train_metrics = [m for ns, m in context.records if ns == "train"][-1]
    assert "energy_term_kinetic_energy" in train_metrics
    assert "energy_term_harmonic_trap" in train_metrics
    assert train_metrics["loss_has_grad"] is False


def test_zero_duplicate_forward_and_pure_potential_single_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tangent path calls model forward once per chunk with KineticEnergy or pure potential."""

    batch = _sample_batch(sample_shape=(5,), n_electrons=2, spatial_dim=2, seed=53)

    # Case 1: KineticEnergy + HarmonicTrap, single chunk -> exactly 1 model forward & 1 kinetic call.
    model = _SoftplusWavefunction()
    kinetic = KineticEnergy()
    kinetic_calls = 0
    original_kinetic = kinetic.local_energy

    def counted_kinetic(wf, b):
        nonlocal kinetic_calls
        kinetic_calls += 1
        return original_kinetic(wf, b)

    kinetic.local_energy = counted_kinetic  # type: ignore[method-assign]

    VMCTrainer(max_steps=1, update_method=_RecordingTangentMethod(chunk_size=None)).fit(
        model=model,
        sampler=_FixedSampler(batch),
        hamiltonian_terms=[kinetic, HarmonicTrap(omega=0.5)],
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        context=_StubContext(),
        emit=lambda **_: None,
    )
    assert model.forward_calls == 1
    assert kinetic_calls == 1

    # Case 2: Multi-chunk (N=5, chunk_size=2 -> 3 chunks) -> exactly 3 model forwards & 3 kinetic calls.
    model.forward_calls = 0
    kinetic_calls = 0
    VMCTrainer(max_steps=1, update_method=_RecordingTangentMethod(chunk_size=2)).fit(
        model=model,
        sampler=_FixedSampler(batch),
        hamiltonian_terms=[kinetic, HarmonicTrap(omega=0.5)],
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        context=_StubContext(),
        emit=lambda **_: None,
    )
    expected_chunks = math.ceil(5 / 2)
    assert model.forward_calls == expected_chunks
    assert kinetic_calls == expected_chunks

    # Case 3: Pure coordinate potential (wavefunction_output is None) -> single fallback forward per chunk.
    pot_model = _SoftplusWavefunction()
    recorder = _RecordingTangentMethod(chunk_size=2)
    VMCTrainer(max_steps=1, update_method=recorder).fit(
        model=pot_model,
        sampler=_FixedSampler(batch),
        hamiltonian_terms=[HarmonicTrap(omega=0.5)],
        optimizer=torch.optim.SGD(pot_model.parameters(), lr=0.01),
        context=_StubContext(),
        emit=lambda **_: None,
    )
    assert pot_model.forward_calls == expected_chunks
    assert recorder.received is not None
    assert any(block.abs().sum().item() > 0.0 for block in recorder.received.parameter_scores.blocks)
    for block in recorder.received.local_energy_derivatives.blocks:
        assert torch.equal(block, torch.zeros_like(block))

    # Case 4: trainer_module.local_energy monkeypatch seam is respected by tangent producer.
    patched_calls = 0
    orig_trainer_le = trainer_module.local_energy

    def spy_local_energy(*args, **kwargs):
        nonlocal patched_calls
        patched_calls += 1
        return orig_trainer_le(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "local_energy", spy_local_energy)
    spy_model = _SoftplusWavefunction()
    VMCTrainer(max_steps=1, update_method=_RecordingTangentMethod(chunk_size=3)).fit(
        model=spy_model,
        sampler=_FixedSampler(batch),
        hamiltonian_terms=[KineticEnergy()],
        optimizer=torch.optim.SGD(spy_model.parameters(), lr=0.01),
        context=_StubContext(),
        emit=lambda **_: None,
    )
    assert patched_calls == math.ceil(5 / 3)


def test_loud_refusals_inference_binding_requests_stateless_and_runtime_guards() -> None:
    """Every unsupported runtime mode, binding defect, request conflict, and guard fails loudly."""

    batch = _sample_batch(sample_shape=(4,), n_electrons=2, spatial_dim=2, seed=61)
    terms = [KineticEnergy()]

    # 1. Inference mode guard fires before any model forward.
    model = _SoftplusWavefunction()
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="inference mode"):
            evaluate_hamiltonian_tangent_step(
                model=model,
                hamiltonian_terms=terms,
                batch=batch,
            )
    assert model.forward_calls == 0

    # 2. Missing / invalid / frozen / replaced / reordered / empty ParameterBinding.
    class _NoBindingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))

        def forward(self, b: ElectronBatch) -> WavefunctionOutput:
            raise AssertionError("forward must not run when binding is invalid")

    with pytest.raises(TypeError, match="parameter_binding"):
        evaluate_hamiltonian_tangent_step(
            model=_NoBindingModel(), hamiltonian_terms=terms, batch=batch
        )

    frozen_model = _SoftplusWavefunction()
    frozen_model.envelope_scale.requires_grad_(False)
    with pytest.raises(ValueError, match="binding/layout"):
        evaluate_hamiltonian_tangent_step(
            model=frozen_model, hamiltonian_terms=terms, batch=batch
        )
    assert frozen_model.forward_calls == 0

    replaced_model = _SoftplusWavefunction()
    replaced_model.envelope_scale = torch.nn.Parameter(
        replaced_model.envelope_scale.detach().clone()
    )
    with pytest.raises(ValueError, match="mismatch"):
        evaluate_hamiltonian_tangent_step(
            model=replaced_model, hamiltonian_terms=terms, batch=batch
        )
    assert replaced_model.forward_calls == 0

    reordered_model = _SoftplusWavefunction()
    params = tuple(reordered_model.parameters())
    reordered_model._parameter_binding = _build_parameter_binding(tuple(reversed(params)))
    with pytest.raises(ValueError, match="reordering"):
        evaluate_hamiltonian_tangent_step(
            model=reordered_model, hamiltonian_terms=terms, batch=batch
        )
    assert reordered_model.forward_calls == 0

    empty_binding_model = _SoftplusWavefunction()
    empty_binding_model._parameter_binding = ParameterBinding(
        layout=ParameterLayout(slots=()), parameters=()
    )
    with pytest.raises(ValueError, match="at least one trainable parameter"):
        evaluate_hamiltonian_tangent_step(
            model=empty_binding_model, hamiltonian_terms=terms, batch=batch
        )
    assert empty_binding_model.forward_calls == 0

    # 3. Invalid request constructors (active Typeguard + suppressed runtime guards).
    for bad_chunk in (0, -1, True, False):
        with pytest.raises(ValueError, match="positive integer or None"):
            MaterializedHamiltonianTangentStepRequest(chunk_size=bad_chunk)  # type: ignore[arg-type]
    for bad_chunk_type in (1.5, "2"):
        with pytest.raises((ValueError, TypeCheckError)):
            MaterializedHamiltonianTangentStepRequest(chunk_size=bad_chunk_type)  # type: ignore[arg-type]
    with typeguard.suppress_type_checks():
        for bad_chunk in (0, -1, True, False, 1.5, "2"):
            with pytest.raises(ValueError, match="positive integer or None"):
                MaterializedHamiltonianTangentStepRequest(chunk_size=bad_chunk)  # type: ignore[arg-type]

    for bad_fwd in (CoordinateGradientRequest(), "value", None):
        with pytest.raises((TypeError, TypeCheckError)):
            ParameterScoreStepRequest(forward_request=bad_fwd)  # type: ignore[arg-type]
    with typeguard.suppress_type_checks():
        for bad_fwd in (CoordinateGradientRequest(), "value", None):
            with pytest.raises(TypeError, match="ParameterScoreRequest"):
                ParameterScoreStepRequest(forward_request=bad_fwd)  # type: ignore[arg-type]

    # 4. Conflicting / invalid step_request() vs forward_request() and stateless bypass guard.
    class _NonRequestMethod(VMCUpdateMethod[Any]):
        def step_request(self) -> Any:
            return "not-a-request"

        def update(self, update_input: Any) -> VMCUpdateResult:
            raise AssertionError("update must not run")

    with pytest.raises(TypeError, match="VMCStepRequest"):
        resolve_step_request(_NonRequestMethod())

    class _UnsupportedLegacyForwardMethod(VMCUpdateMethod[Any]):
        def forward_request(self) -> Any:
            return CoordinateGradientRequest()

        def update(self, update_input: Any) -> VMCUpdateResult:
            raise AssertionError("update must not run")

    with pytest.raises(TypeError, match="ParameterScoreRequest or None"):
        resolve_step_request(_UnsupportedLegacyForwardMethod())

    class _ConflictNonScoreLegacyMethod(VMCUpdateMethod[Any]):
        def step_request(self) -> Any:
            return AutogradStepRequest()

        def forward_request(self) -> Any:
            return CoordinateGradientRequest()

        def update(self, update_input: Any) -> VMCUpdateResult:
            raise AssertionError("update must not run")

    with pytest.raises(ValueError, match="conflicting"):
        resolve_step_request(_ConflictNonScoreLegacyMethod())

    class _ConflictScoreMismatchMethod(VMCUpdateMethod[Any]):
        def step_request(self) -> Any:
            return MaterializedHamiltonianTangentStepRequest()

        def forward_request(self) -> Any:
            return MaterializedParameterScoreRequest()

        def update(self, update_input: Any) -> VMCUpdateResult:
            raise AssertionError("update must not run")

    with pytest.raises(ValueError, match="conflicting"):
        resolve_step_request(_ConflictScoreMismatchMethod())
    with pytest.raises(ValueError, match="conflicting"):
        _ConflictScoreMismatchMethod().describe()

    class _ConsistentOverrideMethod(VMCUpdateMethod[ScoreUpdateInput]):
        def step_request(self) -> ParameterScoreStepRequest:
            return ParameterScoreStepRequest(
                forward_request=MaterializedParameterScoreRequest(chunk_size=2)
            )

        def forward_request(self) -> MaterializedParameterScoreRequest:
            return MaterializedParameterScoreRequest(chunk_size=2)

        def update(self, update_input: ScoreUpdateInput) -> VMCUpdateResult:
            del update_input
            return VMCUpdateResult(applied=True, grad_norm=0.0)

    resolved_consistent = resolve_step_request(_ConsistentOverrideMethod())
    assert resolved_consistent == ParameterScoreStepRequest(
        forward_request=MaterializedParameterScoreRequest(chunk_size=2)
    )

    # Stateless method (update_state() is None) must fail at resolve_update_state
    # and in fit() BEFORE sampler.collect_samples.
    stateless_bad = _ConflictScoreMismatchMethod()
    assert stateless_bad.update_state() is None
    with pytest.raises(ValueError, match="conflicting"):
        VMCTrainer(max_steps=1, update_method=stateless_bad).resolve_update_state(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        )
    sampler = _FixedSampler(batch)
    with pytest.raises(ValueError, match="conflicting"):
        VMCTrainer(max_steps=1, update_method=stateless_bad).fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=terms,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            context=_StubContext(),
            emit=lambda **_: None,
        )
    assert sampler.collect_calls == 0

    # 5. Complex wavefunction (phase is not None) refused loudly.
    complex_model = _SoftplusWavefunction(return_phase=True)
    with pytest.raises(ValueError, match="complex wavefunctions"):
        evaluate_hamiltonian_tangent_step(
            model=complex_model,
            hamiltonian_terms=[HarmonicTrap()],
            batch=batch,
        )

    # 6. Non-differentiable logabs refused loudly.
    const_model = _SoftplusWavefunction(constant_logabs=True)
    with pytest.raises(RuntimeError, match="differentiable logabs"):
        evaluate_hamiltonian_tangent_step(
            model=const_model,
            hamiltonian_terms=[HarmonicTrap()],
            batch=batch,
        )

    # 7. Chunk term-key mismatch across chunks refused loudly.
    chunk_counter = 0

    def drifting_terms_le(h_terms, wf, chunk, *, return_terms=False):
        nonlocal chunk_counter
        del h_terms, return_terms
        chunk_counter += 1
        out = wf(chunk)
        total = torch.zeros((chunk.batch_size,), dtype=chunk.dtype, device=chunk.device)
        key = "term_a" if chunk_counter == 1 else "term_b"
        return LocalEnergyResult(total=total, terms={key: total}, wavefunction_output=out)

    with pytest.raises(ValueError, match="terms changed between chunks"):
        evaluate_hamiltonian_tangent_step(
            model=model,
            hamiltonian_terms=terms,
            batch=batch,
            chunk_size=2,
            local_energy_fn=drifting_terms_le,
        )


def test_serialization_refusal_order_and_detached_derivatives_roundtrip() -> None:
    """__getstate__ enforces exact graph-bearing-before-live refusal order."""

    batch = _sample_batch(sample_shape=(2,), n_electrons=2, spatial_dim=2, seed=71)
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    binding = _build_parameter_binding((p,))

    live_block = p.unsqueeze(0).expand(2, 2) * 3.0
    detached_block = live_block.detach()

    # MaterializedLocalEnergyDerivatives: refuses graph-bearing, round-trips detached.
    live_derivs = MaterializedLocalEnergyDerivatives(layout=binding.layout, blocks=(live_block,))
    with pytest.raises(RuntimeError, match="graph-bearing MaterializedLocalEnergyDerivatives"):
        pickle.dumps(live_derivs)
    with pytest.raises(RuntimeError, match="graph-bearing MaterializedLocalEnergyDerivatives"):
        torch.save(live_derivs, BytesIO())

    detached_derivs = live_derivs.detach()
    restored = pickle.loads(pickle.dumps(detached_derivs))
    ok, _ = restored.compare(detached_derivs, atol=0.0, rtol=0.0)
    assert ok

    # HamiltonianTangentUpdateInput refusal priority:
    # (a) VMCStepData graph-bearing -> super().__getstate__() first.
    live_output = WavefunctionOutput(
        logabs=(p.sum()).expand(2),
        sign=torch.ones(2, dtype=torch.float64),
    )
    detached_output = WavefunctionOutput(
        logabs=torch.zeros(2, dtype=torch.float64),
        sign=torch.ones(2, dtype=torch.float64),
    )
    detached_eloc = torch.zeros(2, dtype=torch.float64)
    live_scores = MaterializedParameterLogScores(layout=binding.layout, blocks=(live_block,))
    detached_scores = live_scores.detach()

    case_a = HamiltonianTangentUpdateInput(
        batch=batch,
        wavefunction=live_output,
        local_energy=detached_eloc,
        step=0,
        parameter_scores=live_scores,
        local_energy_derivatives=live_derivs,
        parameter_binding=binding,
    )
    with pytest.raises(RuntimeError, match="graph-bearing VMCStepData"):
        pickle.dumps(case_a)

    # (b) parameter_scores graph-bearing -> specific graph-bearing error before live error.
    case_b = HamiltonianTangentUpdateInput(
        batch=batch,
        wavefunction=detached_output,
        local_energy=detached_eloc,
        step=0,
        parameter_scores=live_scores,
        local_energy_derivatives=detached_derivs,
        parameter_binding=binding,
    )
    with pytest.raises(RuntimeError, match="graph-bearing HamiltonianTangentUpdateInput"):
        pickle.dumps(case_b)

    # (c) local_energy_derivatives graph-bearing -> specific graph-bearing error before live error.
    case_c = HamiltonianTangentUpdateInput(
        batch=batch,
        wavefunction=detached_output,
        local_energy=detached_eloc,
        step=0,
        parameter_scores=detached_scores,
        local_energy_derivatives=live_derivs,
        parameter_binding=binding,
    )
    with pytest.raises(RuntimeError, match="graph-bearing HamiltonianTangentUpdateInput"):
        pickle.dumps(case_c)

    # (d) All tensors detached -> live binding refusal.
    case_d = HamiltonianTangentUpdateInput(
        batch=batch,
        wavefunction=detached_output,
        local_energy=detached_eloc,
        step=0,
        parameter_scores=detached_scores,
        local_energy_derivatives=detached_derivs,
        parameter_binding=binding,
    )
    with pytest.raises(RuntimeError, match="live HamiltonianTangentUpdateInput"):
        pickle.dumps(case_d)
    with pytest.raises(RuntimeError, match="live HamiltonianTangentUpdateInput"):
        torch.save(case_d, BytesIO())

    # ProducedVMCStep unconditionally refuses serialization even when fully detached.
    produced = ProducedVMCStep(
        update_input=case_d,
        objective=VMCObjectiveResult(loss=torch.tensor(0.0, dtype=torch.float64), metrics={}),
        local_energy=detached_eloc,
        term_energies=None,
        wavefunction_output=detached_output,
    )
    with pytest.raises(RuntimeError, match="live ProducedVMCStep"):
        pickle.dumps(produced)
    with pytest.raises(RuntimeError, match="live ProducedVMCStep"):
        torch.save(produced, BytesIO())


def test_disconnected_parameter_zero_blocks_and_derivatives_operations() -> None:
    """Disconnected/eloc-invariant parameters yield exact zero blocks; container ops work cleanly."""

    model = _SoftplusWavefunction(
        spatial_dim=2,
        include_disconnected=True,
        include_logabs_only_shift=True,
    )
    batch = _sample_batch(sample_shape=(3,), n_electrons=2, spatial_dim=2, seed=83)
    update_input, _ = evaluate_hamiltonian_tangent_step(
        model=model,
        hamiltonian_terms=[KineticEnergy(), HarmonicTrap(omega=0.5)],
        batch=batch,
    )

    params = model.parameter_binding.parameters
    shift_idx = next(i for i, p in enumerate(params) if p is model.shift)
    disc_idx = next(i for i, p in enumerate(params) if p is model.disconnected)

    # Disconnected parameter: exact zeros in both scores and local-energy derivatives.
    disc_score = update_input.parameter_scores.blocks[disc_idx]
    disc_deriv = update_input.local_energy_derivatives.blocks[disc_idx]
    assert disc_score.shape == (3, 1, 2)
    assert disc_deriv.shape == (3, 1, 2)
    assert disc_score.dtype == model.disconnected.dtype
    assert disc_deriv.dtype == model.disconnected.dtype
    assert torch.equal(disc_score, torch.zeros_like(disc_score))
    assert torch.equal(disc_deriv, torch.zeros_like(disc_deriv))

    # Additive constant on logabs: score == 1.0, local-energy derivative == 0.0.
    shift_score = update_input.parameter_scores.blocks[shift_idx]
    shift_deriv = update_input.local_energy_derivatives.blocks[shift_idx]
    assert torch.allclose(shift_score, torch.ones_like(shift_score), atol=1e-12, rtol=1e-12)
    assert torch.equal(shift_deriv, torch.zeros_like(shift_deriv))

    # MaterializedLocalEnergyDerivatives container operations: permute, to, detach, compare, validate.
    derivs = update_input.local_energy_derivatives
    assert derivs.sample_shape == (3,)
    assert derivs.device == torch.device("cpu")
    assert derivs.validate(sample_shape=(3,)) is derivs

    permuted = derivs.permute(Permutation((1, 0)))
    ok, metrics = derivs.compare(permuted, atol=0.0, rtol=0.0)
    assert ok and metrics["max_abs_error"] == 0.0
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(derivs.blocks, permuted.blocks, strict=True))

    casted = derivs.to(dtype=torch.float32)
    assert all(slot.dtype == torch.float32 for slot in casted.layout.slots)
    assert all(block.dtype == torch.float32 for block in casted.blocks)
    same_layout, _ = derivs.compare(casted)
    assert not same_layout

    with pytest.raises(ValueError, match="expected sample shape"):
        derivs.validate(sample_shape=(4,))

    # Also verify non-VMCUpdateMethod passed to resolve_step_request fails loudly.
    with pytest.raises((TypeError, TypeCheckError)):
        resolve_step_request(object())  # type: ignore[arg-type]
    with typeguard.suppress_type_checks():
        with pytest.raises(TypeError, match="VMCUpdateMethod"):
            resolve_step_request(object())  # type: ignore[arg-type]
