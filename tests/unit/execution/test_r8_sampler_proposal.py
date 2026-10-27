"""CR-r8 A1 tests for the real pre-acceptance sampler proposal event."""

from __future__ import annotations

import torch
from torch import nn

from tpen.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from tpen.execution.chain_resume import ChannelEvent, E5_CHANNEL_MAP, ProcessEvidence, compare_channel_parity
from tpen.sampling.metropolis import MetropolisSampler
from tpen.sampling.moves import GaussianMove
from tests.helpers.native_chain_resume_trace import NativeTrace, install_native_trace


class _FixedMove(nn.Module):
    """Deterministic move that independently captures every real proposal."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta
        self.calls = 0
        self.proposals: list[torch.Tensor] = []

    def propose(self, walkers: Walkers, *, generator=None):
        del generator
        self.calls += 1
        proposal = walkers.positions + self.delta
        self.proposals.append(proposal.detach().clone())
        ratio = torch.zeros(walkers.batch_size, dtype=walkers.dtype, device=walkers.device)
        return proposal, ratio


class _RecordingGaussianMove(GaussianMove):
    def __init__(self) -> None:
        super().__init__(step_size=0.25, move_all=True)
        self.proposals: list[torch.Tensor] = []

    def propose(self, walkers: Walkers, *, generator=None):
        proposal, ratio = super().propose(walkers, generator=generator)
        self.proposals.append(proposal.detach().clone())
        return proposal, ratio


class _AcceptEveryProposal(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = torch.zeros(batch.batch_size, dtype=batch.positions.dtype, device=batch.positions.device)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _RejectNonzeroProposal(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        is_initial = batch.positions.eq(0).all(dim=(1, 2))
        logabs = torch.where(
            is_initial,
            torch.zeros(batch.batch_size, dtype=batch.positions.dtype, device=batch.positions.device),
            torch.full((batch.batch_size,), -torch.inf, dtype=batch.positions.dtype, device=batch.positions.device),
        )
        sign = torch.where(is_initial, torch.ones_like(logabs), torch.zeros_like(logabs))
        return WavefunctionOutput(logabs=logabs, sign=sign)


def _sampler(move, *, burn_in=0, n_steps=1, seed=17) -> MetropolisSampler:
    sampler = MetropolisSampler(
        move=move,
        n_walkers=2,
        burn_in=burn_in,
        n_steps=n_steps,
        n_electrons=1,
        spatial_dim=1,
        dtype=torch.float64,
        seed=seed,
    )
    sampler._walkers = Walkers(positions=torch.zeros((2, 1, 1), dtype=torch.float64))
    sampler._has_burned_in = burn_in == 0
    return sampler


def _first_event(trace: NativeTrace, channel: str):
    return next(event for event in trace.events if event.channel == channel)


def _run_traced(model, *, burn_in=0, n_steps=1):
    move = _FixedMove()
    sampler = _sampler(move, burn_in=burn_in, n_steps=n_steps)
    trace = NativeTrace(phase="recovery", seed=17, resume_step=3)
    with install_native_trace(trace):
        walkers, _stats = sampler.collect_samples(model)
    return move, walkers, trace


def test_accepted_and_rejected_positions_are_separate_green_controls() -> None:
    accepted_move, accepted_walkers, accepted_trace = _run_traced(_AcceptEveryProposal())
    rejected_move, rejected_walkers, rejected_trace = _run_traced(_RejectNonzeroProposal())

    # Both real sampler calls make the same first proposal before model
    # evaluation.  The independent move spy is the advertised proposal value.
    assert accepted_move.calls == rejected_move.calls == 1
    assert torch.equal(accepted_move.proposals[0], rejected_move.proposals[0])
    assert torch.equal(accepted_move.proposals[0], torch.ones_like(accepted_move.proposals[0]))

    # Accepted positions are intentionally different, so they are a separate
    # green control and cannot serve as the proposal channel.
    assert torch.equal(accepted_walkers.positions, accepted_move.proposals[0])
    assert torch.equal(rejected_walkers.positions, torch.zeros_like(rejected_walkers.positions))
    assert not torch.equal(accepted_walkers.positions, rejected_walkers.positions)

    # The proposal channel is checked independently below; this control only
    # proves that acceptance changes positions without changing the proposal.
    assert _first_event(accepted_trace, "sampler_proposal").value is not None
    assert _first_event(rejected_trace, "sampler_proposal").value is not None


def test_sampler_proposal_channel_emits_the_first_pre_acceptance_proposal() -> None:
    accepted_move, _accepted_walkers, accepted_trace = _run_traced(_AcceptEveryProposal())
    rejected_move, _rejected_walkers, rejected_trace = _run_traced(_RejectNonzeroProposal())
    assert float(accepted_move.proposals[0].mean()) == 1.0
    assert float(rejected_move.proposals[0].mean()) == 1.0
    assert _first_event(accepted_trace, "sampler_proposal").value == float(
        accepted_move.proposals[0].mean()
    )
    assert _first_event(rejected_trace, "sampler_proposal").value == float(
        rejected_move.proposals[0].mean()
    )


def test_two_acceptance_regimes_have_equal_proposal_channel_after_helper_fix() -> None:
    accepted_move, _accepted_walkers, accepted_trace = _run_traced(_AcceptEveryProposal())
    rejected_move, _rejected_walkers, rejected_trace = _run_traced(_RejectNonzeroProposal())
    assert torch.equal(accepted_move.proposals[0], rejected_move.proposals[0])

    # This is the parity requirement for the proposed _propose-only spy.
    assert _first_event(accepted_trace, "sampler_proposal").value == _first_event(
        rejected_trace, "sampler_proposal"
    ).value


def test_real_sampler_state_restore_preserves_first_proposal_and_omission_diverges() -> None:
    baseline_move = _RecordingGaussianMove()
    baseline = _sampler(baseline_move, seed=17)
    state = baseline.mcmc_state_dict()
    baseline.collect_samples(_AcceptEveryProposal())

    restored_move = _RecordingGaussianMove()
    restored = _sampler(restored_move, seed=91)
    restored.load_mcmc_state_dict(state, device="cpu")
    restored.collect_samples(_AcceptEveryProposal())

    omitted_move = _RecordingGaussianMove()
    omitted = _sampler(omitted_move, seed=91)
    omitted.collect_samples(_AcceptEveryProposal())

    assert torch.equal(baseline_move.proposals[0], restored_move.proposals[0])
    assert not torch.equal(baseline_move.proposals[0], omitted_move.proposals[0])


def test_multiple_real_proposals_advance_trace_instrumentation_once_per_collect() -> None:
    move, _walkers, trace = _run_traced(_AcceptEveryProposal(), burn_in=2, n_steps=2)
    assert move.calls == 4
    assert len([event for event in trace.events if event.channel == "trainer_cursor"]) == 1
    assert len([event for event in trace.events if event.channel == "process_random"]) == 1
    assert len([event for event in trace.events if event.channel == "sampler_proposal"]) == 1
    assert _first_event(trace, "sampler_proposal").value == float(move.proposals[0].mean())


def test_synthetic_full_parity_comparator_control() -> None:
    events = tuple(ChannelEvent(spec.channel, 3, spec.event_kind, 1.0) for spec in E5_CHANNEL_MAP)
    reference = ProcessEvidence(
        phase="reference", pid=1, seed=17, seed_source="reference", command=(), returncode=0, events=events
    )
    full = ProcessEvidence(
        phase="recovery", pid=2, seed=17, seed_source="recovery", command=(), returncode=0, events=events
    )
    assert all(compare_channel_parity(reference, full, resume_step=2).values())
