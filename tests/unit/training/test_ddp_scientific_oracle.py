"""DF0: process-free VMC scientific oracle and world-size-one identity tests.

Groups, per note ``df0-test-plan`` on Task Orchestrator item ``03375c09``:

- Group A: oracle self-consistency and metamorphic invariants (pure data;
  never touches production ``tpen.training.vmc`` or ``VMCTrainer``).
- Group B: world-size-one identity against production (single shard = whole
  batch); must pass on today's code.
- Group C: closure/custom optimizer shape, and event/callback/checkpoint
  identity across both optimizer shapes.
- Divergence characterization: pins today's observed behavior for a gap
  between ``ddp-scientific-oracle-2026-08-31`` and today's one-process code
  (see note ``df0-oracle-vs-production-divergences``); asserts no requirement.

No test in this module opens a process group, touches an accelerator, or
names a facility. There is no capability gate and this module has a zero
skip count by construction.
"""

from __future__ import annotations

import copy
import json
import math

import pytest
import torch

from tpen.callback import Checkpoint, DataIntegrity
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, HarmonicTrap
from tpen.training.events import (
    Backward,
    BuildBatch,
    CollectSamples,
    Forward,
    LocalEnergy,
    Metrics,
    Objective,
    OptimizerUpdate,
    TrainingIteration,
    TrainingIterationCompleted,
    UpdateCompleted,
)
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ModelParameterBinding,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
)
from tpen.training.vmc import compute_vmc_objective
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn
from tests.helpers.run_context import RecordingLogger, make_run_context
from tests.helpers.vmc_scientific_oracle import (
    SufficientStatistics,
    loss_tolerance_envelope,
    naive_per_shard_clip_then_sum,
    oracle_global_clip,
    oracle_vmc_objective,
    reduce_energy_shards,
)
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext, _occurrence_labels


def _hooke_terms() -> list:
    return [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]


def _gradient_l2_norm(parameters) -> float:
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().pow(2).sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).item()) if total is not None else 0.0


def _parameter_tolerance_envelope(terms: torch.Tensor) -> float:
    """Parameter-space tolerance: the loss envelope, widened for backprop.

    Comparing MODEL PARAMETERS after a real backward pass adds the tiny
    network's own chain of operations on top of the score-function reduction
    ``loss_tolerance_envelope`` bounds. 10x is a documented, generous multiple
    of the already-derived loss envelope -- not a fresh unexplained epsilon --
    chosen because the Hooke fixture's forward/backward is a short, shallow
    op chain (a handful of layers), so one order of magnitude comfortably
    covers its additional rounding steps without hiding a real divergence.
    """

    return 10.0 * loss_tolerance_envelope(terms)


def _adam_recursion_tolerance_envelope(terms: torch.Tensor, *, adam_eps: float = 1e-8) -> float:
    """Tolerance across TWO chained Adam updates (state_dict round trip, B5).

    Adam divides by ``sqrt(v_hat) + adam_eps``; a base rounding perturbation
    can be amplified by up to ``1 / adam_eps`` in the worst case where
    ``v_hat`` is near zero. Applying that amplification once (not compounded
    per step) already exceeds the empirically observed two-step discrepancy
    (~1e-10) by several orders of magnitude, so this is a conservative bound
    derived from Adam's own epsilon floor, not a value tuned to just pass.
    """

    return loss_tolerance_envelope(terms) / adam_eps


def _assert_tensor_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float) -> None:
    assert torch.allclose(actual, expected, atol=atol, rtol=0.0), (
        f"max abs diff {(actual - expected).abs().max().item()} exceeds atol={atol}"
    )


def _assert_nested_close(left, right, *, atol: float) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        if left.dtype.is_floating_point:
            _assert_tensor_close(left, right, atol=atol)
        else:
            assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_close(left[key], right[key], atol=atol)
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_close(left_value, right_value, atol=atol)
    else:
        assert left == right


# =============================================================================
# Group A -- oracle self-consistency and metamorphic invariants (pure data)
# =============================================================================


def test_sufficient_statistics_merge_matches_batch_welford_on_equal_shards() -> None:
    shard_a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    shard_b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)

    merged = SufficientStatistics.from_values(shard_a).merge(SufficientStatistics.from_values(shard_b))
    whole = torch.cat([shard_a, shard_b])

    assert merged.count == 6
    atol = loss_tolerance_envelope(whole.abs())
    assert abs(merged.mean - float(whole.mean().item())) <= atol
    assert abs(merged.variance - float(whole.var(unbiased=False).item())) <= atol


def test_sufficient_statistics_zero_count_packet_is_merge_identity() -> None:
    packet = SufficientStatistics.from_values(torch.tensor([1.0, -2.0, 3.5], dtype=torch.float64))

    assert SufficientStatistics.EMPTY.merge(packet) == packet
    assert packet.merge(SufficientStatistics.EMPTY) == packet


def test_global_energy_statistics_matches_naive_concatenation_on_unequal_shards() -> None:
    """Oracle-note M2: raw ``[5, 3, 7]``, finite ``[3, 0, 5]``."""

    torch.manual_seed(0)
    shard1 = torch.cat(
        [torch.randn(3, dtype=torch.float64), torch.full((2,), float("nan"), dtype=torch.float64)]
    )
    shard2 = torch.full((3,), float("inf"), dtype=torch.float64)
    shard3 = torch.cat(
        [torch.randn(5, dtype=torch.float64), torch.full((2,), float("nan"), dtype=torch.float64)]
    )
    assert (shard1.numel(), shard2.numel(), shard3.numel()) == (5, 3, 7)

    stats = reduce_energy_shards([shard1, shard2, shard3])
    all_finite = torch.cat([shard1[torch.isfinite(shard1)], shard3[torch.isfinite(shard3)]])

    assert stats.n_total == 15
    assert stats.n_finite == 8
    atol = loss_tolerance_envelope(all_finite.abs())
    assert abs(stats.mean - float(all_finite.mean().item())) <= atol
    assert abs(stats.variance - float(all_finite.var(unbiased=False).item())) <= atol


def test_oracle_partition_invariance_across_shard_splits() -> None:
    """Oracle-note M7: the same sample order re-partitioned as [15]/[5,3,7]/[1,13,1]."""

    torch.manual_seed(0)
    whole = torch.randn(15, dtype=torch.float64)
    whole[2] = float("nan")
    whole[9] = float("nan")

    def split(sizes: list[int]) -> list[torch.Tensor]:
        shards, offset = [], 0
        for size in sizes:
            shards.append(whole[offset : offset + size])
            offset += size
        return shards

    single = reduce_energy_shards(split([15]))
    three = reduce_energy_shards(split([5, 3, 7]))
    uneven = reduce_energy_shards(split([1, 13, 1]))

    atol = loss_tolerance_envelope(whole[torch.isfinite(whole)].abs())
    for candidate in (three, uneven):
        assert candidate.n_total == single.n_total
        assert candidate.n_finite == single.n_finite
        assert abs(candidate.mean - single.mean) <= atol
        assert abs(candidate.variance - single.variance) <= atol


def test_oracle_invariance_under_shard_merge_order() -> None:
    """Oracle-note M7 rank (shard-order) permutation, split from the within-shard
    sample-permutation case below (R5): a combined test cannot tell a reviewer
    which invariance broke."""

    torch.manual_seed(1)
    shard_a = torch.randn(4, dtype=torch.float64)
    shard_b = torch.randn(3, dtype=torch.float64)
    shard_c = torch.randn(5, dtype=torch.float64)
    all_values = torch.cat([shard_a, shard_b, shard_c])
    atol = loss_tolerance_envelope(all_values.abs())

    forward = reduce_energy_shards([shard_a, shard_b, shard_c])
    reordered = reduce_energy_shards([shard_c, shard_a, shard_b])
    assert reordered.n_total == forward.n_total
    assert reordered.n_finite == forward.n_finite
    assert abs(reordered.mean - forward.mean) <= atol
    assert abs(reordered.variance - forward.variance) <= atol


def test_oracle_invariance_under_within_shard_sample_permutation() -> None:
    """Oracle-note M8 sample permutation, split from the shard-order case above (R5)."""

    torch.manual_seed(1)
    shard_a = torch.randn(4, dtype=torch.float64)
    shard_b = torch.randn(3, dtype=torch.float64)
    shard_c = torch.randn(5, dtype=torch.float64)
    all_values = torch.cat([shard_a, shard_b, shard_c])
    atol = loss_tolerance_envelope(all_values.abs())

    forward = reduce_energy_shards([shard_a, shard_b, shard_c])
    # A fixed reversal, not `torch.randperm`: a random permutation of a
    # length-3 tensor has a 1-in-6 chance of landing on the identity, which
    # would make this case invariant to any mutation by pure luck rather
    # than by construction.
    permuted_b = shard_b.flip(0)
    within_shard = reduce_energy_shards([shard_a, permuted_b, shard_c])
    assert abs(within_shard.mean - forward.mean) <= atol
    assert abs(within_shard.variance - forward.variance) <= atol


def test_oracle_duplicate_data_invariance_halves_stderr() -> None:
    """Oracle-note M8 duplication.

    Duplication preserves mean/variance/loss and halves stderr by sqrt(2), at
    the level of the global scalar metrics. Per-sample ``d(loss)/d(logabs_i)``
    is NOT asserted equal here: each duplicated leaf independently carries
    half the original per-sample gradient magnitude (M is doubled), which is
    correct, not a bug -- the M8 "gradient unchanged" claim is about a real
    model parameter shared by both copies, exercised at the full-trainer
    level in Group B/C, not about independent per-sample leaves here.
    """

    torch.manual_seed(2)
    logabs = torch.randn(6, dtype=torch.float64, requires_grad=True)
    energy = torch.randn(6, dtype=torch.float64)

    single = oracle_vmc_objective([logabs], [energy])
    doubled = oracle_vmc_objective(
        [logabs, logabs.detach().clone().requires_grad_(True)],
        [energy, energy.clone()],
    )

    atol = loss_tolerance_envelope(energy.abs())
    assert doubled.metrics["local_energy_n_finite"] == 2 * single.metrics["local_energy_n_finite"]
    assert doubled.metrics["local_energy_n_total"] == 2 * single.metrics["local_energy_n_total"]
    assert abs(doubled.metrics["energy"] - single.metrics["energy"]) <= atol
    assert abs(doubled.metrics["energy_variance"] - single.metrics["energy_variance"]) <= atol
    assert abs(doubled.loss.item() - single.loss.item()) <= atol
    assert abs(doubled.metrics["energy_stderr"] - single.metrics["energy_stderr"] / (2**0.5)) <= atol


def test_oracle_affine_energy_transform_matches_analytic_prediction() -> None:
    """Oracle-note M8: ``E' = aE + b``."""

    torch.manual_seed(3)
    logabs = torch.randn(8, dtype=torch.float64, requires_grad=True)
    energy = torch.randn(8, dtype=torch.float64)
    a, b = 2.5, -1.3

    base = oracle_vmc_objective([logabs], [energy])
    transformed = oracle_vmc_objective(
        [logabs.detach().clone().requires_grad_(True)], [a * energy + b]
    )

    atol = loss_tolerance_envelope(energy.abs())
    assert abs(transformed.metrics["energy"] - (a * base.metrics["energy"] + b)) <= abs(a) * atol
    assert abs(transformed.metrics["energy_variance"] - (a**2) * base.metrics["energy_variance"]) <= (a**2) * atol
    assert abs(transformed.loss.item() - a * base.loss.item()) <= abs(a) * atol


def test_oracle_constant_logabs_shift_leaves_loss_and_gradient_unchanged() -> None:
    """Oracle-note M8: adding a constant to every logabs."""

    torch.manual_seed(4)
    logabs = torch.randn(6, dtype=torch.float64, requires_grad=True)
    energy = torch.randn(6, dtype=torch.float64)
    shift = 7.25

    base = oracle_vmc_objective([logabs], [energy])
    base.loss.backward()
    base_grad = logabs.grad.clone()

    shifted_logabs = (logabs.detach() + shift).requires_grad_(True)
    shifted = oracle_vmc_objective([shifted_logabs], [energy])
    shifted.loss.backward()

    atol = loss_tolerance_envelope(energy.abs())
    assert abs(shifted.loss.item() - base.loss.item()) <= atol
    _assert_tensor_close(shifted_logabs.grad, base_grad, atol=atol)


def test_oracle_minimal_local_centering_counterexample_diverges_from_naive_local() -> None:
    """Oracle-note M6: one distinct sample per shard, two shards.

    Per-shard local centering (each shard subtracts ITS OWN one-sample mean)
    zeroes every residual identically -- the counterexample the oracle's
    GLOBAL centering exists to rule out.
    """

    shard_a_energy = torch.tensor([1.0], dtype=torch.float64)
    shard_b_energy = torch.tensor([5.0], dtype=torch.float64)
    shard_a_logabs = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    shard_b_logabs = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)

    result = oracle_vmc_objective(
        [shard_a_logabs, shard_b_logabs], [shard_a_energy, shard_b_energy]
    )
    result.loss.backward()
    assert shard_a_logabs.grad.abs().item() > 0.0
    assert shard_b_logabs.grad.abs().item() > 0.0

    # The wrong algorithm: with exactly one sample per shard, centering on the
    # shard's OWN mean subtracts the sample from itself, so both residuals are
    # identically zero regardless of the actual energy values.
    naive_local_residual_a = shard_a_energy.item() - shard_a_energy.item()
    naive_local_residual_b = shard_b_energy.item() - shard_b_energy.item()
    assert naive_local_residual_a == 0.0
    assert naive_local_residual_b == 0.0
    assert shard_a_logabs.grad.item() != naive_local_residual_a
    assert shard_b_logabs.grad.item() != naive_local_residual_b


def test_oracle_rejects_finite_energy_with_nonfinite_logabs() -> None:
    logabs = torch.tensor([0.1, float("nan"), 0.3], dtype=torch.float64, requires_grad=True)
    energy = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="poisons the objective"):
        oracle_vmc_objective([logabs], [energy])


def test_oracle_raises_when_global_finite_count_is_zero() -> None:
    logabs = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    energy = torch.tensor([float("nan"), float("inf")], dtype=torch.float64)

    with pytest.raises(ValueError, match="finite-energy count is zero"):
        oracle_vmc_objective([logabs], [energy])


def test_oracle_analytic_gradient_matches_autograd_gradient() -> None:
    torch.manual_seed(5)
    logabs = torch.randn(10, dtype=torch.float64, requires_grad=True)
    energy = torch.randn(10, dtype=torch.float64)
    energy[3] = float("nan")

    result = oracle_vmc_objective([logabs], [energy])
    result.loss.backward()

    atol = loss_tolerance_envelope(energy[torch.isfinite(energy)].abs())
    _assert_tensor_close(logabs.grad, result.per_sample_gradients[0], atol=atol)


def test_oracle_single_valid_sample_variance_is_exactly_zero() -> None:
    """Oracle-note M5: one globally valid sample across a two-shard split."""

    shard1 = torch.tensor([float("nan"), 3.5, float("inf")], dtype=torch.float64)
    shard2 = torch.tensor([float("nan")], dtype=torch.float64)

    stats = reduce_energy_shards([shard1, shard2])

    assert stats.n_finite == 1
    assert stats.variance == 0.0
    assert stats.mean == pytest.approx(3.5)


def test_oracle_appending_nonfinite_energy_rows_changes_only_counts_not_moments_or_gradient() -> None:
    """Oracle-note M8 appended-invalid-rows clause; direct test of contract break 3."""

    torch.manual_seed(6)
    logabs = torch.randn(5, dtype=torch.float64, requires_grad=True)
    energy = torch.randn(5, dtype=torch.float64)

    base = oracle_vmc_objective([logabs], [energy])
    base.loss.backward()
    base_grad = logabs.grad.clone()

    appended_logabs = torch.cat(
        [logabs.detach(), torch.tensor([0.5, -0.2], dtype=torch.float64)]
    ).requires_grad_(True)
    appended_energy = torch.cat(
        [energy, torch.tensor([float("nan"), float("inf")], dtype=torch.float64)]
    )

    appended = oracle_vmc_objective([appended_logabs], [appended_energy])
    appended.loss.backward()

    atol = loss_tolerance_envelope(energy.abs())
    assert abs(appended.metrics["energy"] - base.metrics["energy"]) <= atol
    assert abs(appended.metrics["energy_variance"] - base.metrics["energy_variance"]) <= atol
    assert abs(appended.loss.item() - base.loss.item()) <= atol
    _assert_tensor_close(appended_logabs.grad[:5], base_grad, atol=atol)
    assert appended_logabs.grad[5].item() == 0.0
    assert appended_logabs.grad[6].item() == 0.0
    assert appended.metrics["local_energy_n_total"] == base.metrics["local_energy_n_total"] + 2
    assert (
        appended.metrics["local_energy_nonfinite_count"]
        == base.metrics["local_energy_nonfinite_count"] + 2
    )


def test_global_gradient_clip_differs_from_per_shard_clip_on_large_opposing_partial_gradients() -> None:
    """Oracle-note M9's decisive clipping case; the process-free half of contract break 3."""

    shard_a = torch.tensor([10.0, 10.0], dtype=torch.float64)
    shard_b = torch.tensor([-10.05, -10.05], dtype=torch.float64)
    clip_norm = 1.0

    global_clipped = oracle_global_clip([shard_a, shard_b], clip_norm)
    naive = naive_per_shard_clip_then_sum([shard_a, shard_b], clip_norm)
    raw_sum = shard_a + shard_b

    assert float(raw_sum.norm().item()) < clip_norm
    _assert_tensor_close(global_clipped, raw_sum, atol=1e-12)
    assert not torch.allclose(global_clipped, naive)


# =============================================================================
# Group B -- world-size-one identity against production
# =============================================================================


def test_oracle_matches_compute_vmc_objective_loss_and_metrics_on_synthetic_batch() -> None:
    """Forward outputs / local-energy statistics / loss, at world_size=1.

    Excludes the D1 pathological combination (finite energy paired with a
    nonfinite logabs) so the identity bar stays achievable on today's code;
    see divergence D1 in note ``df0-oracle-vs-production-divergences``.
    """

    torch.manual_seed(7)
    logabs = torch.randn(9, dtype=torch.float64, requires_grad=True)
    energy = torch.randn(9, dtype=torch.float64)
    energy[2] = float("nan")
    energy[5] = float("nan")

    production = compute_vmc_objective(logabs.detach().clone().requires_grad_(True), energy.clone())
    oracle = oracle_vmc_objective([logabs.detach().clone().requires_grad_(True)], [energy.clone()])

    assert production.metrics["local_energy_n_finite"] == oracle.metrics["local_energy_n_finite"]
    assert production.metrics["local_energy_n_total"] == oracle.metrics["local_energy_n_total"]
    assert (
        production.metrics["local_energy_nonfinite_count"]
        == oracle.metrics["local_energy_nonfinite_count"]
    )

    finite_energy = energy[torch.isfinite(energy)]
    atol = loss_tolerance_envelope(finite_energy.abs())
    assert abs(production.metrics["energy"] - oracle.metrics["energy"]) <= atol
    assert abs(production.metrics["energy_variance"] - oracle.metrics["energy_variance"]) <= atol
    assert abs(production.loss.item() - oracle.loss.item()) <= atol


def test_oracle_gradient_matches_compute_vmc_objective_autograd_gradient() -> None:
    torch.manual_seed(8)
    energy = torch.randn(9, dtype=torch.float64)
    energy[1] = float("nan")
    energy[6] = float("nan")

    logabs_prod = torch.randn(9, dtype=torch.float64, requires_grad=True)
    logabs_oracle = logabs_prod.detach().clone().requires_grad_(True)

    production = compute_vmc_objective(logabs_prod, energy.clone())
    production.loss.backward()
    oracle = oracle_vmc_objective([logabs_oracle], [energy.clone()])
    oracle.loss.backward()

    finite_energy = energy[torch.isfinite(energy)]
    atol = loss_tolerance_envelope(finite_energy.abs())
    _assert_tensor_close(logabs_prod.grad, logabs_oracle.grad, atol=atol)


def _run_control(
    *,
    optimizer_factory,
    gradient_clip_norm: float | None = None,
    update_method_factory=None,
    seed: int = 0,
):
    """Run the real ``VMCTrainer`` one step over the tiny Hooke stack.

    Returns ``(pre_step_model, post_step_model, optimizer, state, context)``.
    ``pre_step_model`` is a deep copy taken BEFORE ``fit`` runs, so a caller
    can replay the same forward pass independently on ``state.batch`` /
    ``state.local_energy`` without needing a second sampler draw.
    """

    torch.manual_seed(seed)
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    assert sampler.seed is not None, (
        "the smoke fixture stopped supplying a sampler seed: two independent "
        "forward passes over the same captured batch is the whole point here"
    )
    pre_step_model = copy.deepcopy(model)
    optimizer = optimizer_factory(model.parameters())
    trainer_kwargs: dict = {"max_steps": 1}
    if update_method_factory is not None:
        trainer_kwargs["update_method"] = update_method_factory(optimizer, model)
    else:
        trainer_kwargs["gradient_clip_norm"] = gradient_clip_norm
    trainer = VMCTrainer(**trainer_kwargs)
    context = _StubContext()
    state = trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=_hooke_terms(),
        optimizer=optimizer,
        context=context,
        emit=lambda name, *, state=None, payload=None, step=None: None,
    )
    return pre_step_model, model, optimizer, state, context


def _run_oracle_shadow_ordinary(pre_step_model, state, *, optimizer_factory, clip_norm=None):
    """Independently replay one ordinary-shape optimizer update from the oracle."""

    shadow_model = copy.deepcopy(pre_step_model)
    output = shadow_model(state.batch)
    result = oracle_vmc_objective([output.logabs], [state.local_energy])
    result.loss.backward()
    raw_norm = _gradient_l2_norm(shadow_model.parameters())
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(shadow_model.parameters(), clip_norm)
    shadow_optimizer = optimizer_factory(shadow_model.parameters())
    shadow_optimizer.step()
    return shadow_model, shadow_optimizer, raw_norm


def test_full_trainer_step_matches_oracle_ordinary_optimizer_shape() -> None:
    """Loss, gradients, optimizer update; ORDINARY shape, on the real Hooke stack.

    Comparison surface (per plan R3): every named parameter after the step,
    and the FULL ``optimizer.state_dict()`` compared recursively. Covers
    three optimizer configurations so a single one cannot hide a state key it
    never creates.
    """

    factories = [
        lambda params: torch.optim.Adam(params, lr=0.01, amsgrad=False),
        lambda params: torch.optim.Adam(params, lr=0.01, amsgrad=True),
        lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
    ]
    for optimizer_factory in factories:
        pre_step_model, model, optimizer, state, _ = _run_control(optimizer_factory=optimizer_factory)
        shadow_model, shadow_optimizer, _ = _run_oracle_shadow_ordinary(
            pre_step_model, state, optimizer_factory=optimizer_factory
        )

        atol = _parameter_tolerance_envelope(state.local_energy.abs())
        for control_param, shadow_param in zip(
            model.parameters(), shadow_model.parameters(), strict=True
        ):
            _assert_tensor_close(control_param.detach(), shadow_param.detach(), atol=atol)
        _assert_nested_close(optimizer.state_dict(), shadow_optimizer.state_dict(), atol=atol)


def test_full_trainer_step_clip_below_above_near_threshold_matches_oracle() -> None:
    """Optimizer update under clipping; M9 below/above/near-threshold cases.

    ``tpen/training/update.py`` discards ``clip_grad_norm_``'s return value
    and computes the published ``grad_norm`` metric from the ALREADY-CLIPPED
    parameters (post-clip, not pre-clip). This test therefore computes the
    PRE-clip norm itself from the oracle-driven shadow and asserts only
    against optimizer state and parameters -- never against a "logged
    pre-clip norm" metric, because production publishes none.
    """

    optimizer_factory = lambda params: torch.optim.Adam(params, lr=0.01)

    probe_pre_step_model, _, _, probe_state, _ = _run_control(
        optimizer_factory=optimizer_factory, gradient_clip_norm=None
    )
    _, _, raw_norm = _run_oracle_shadow_ordinary(
        probe_pre_step_model, probe_state, optimizer_factory=optimizer_factory, clip_norm=None
    )
    assert raw_norm > 0.0

    atol = _parameter_tolerance_envelope(probe_state.local_energy.abs())
    for clip_norm in (raw_norm * 10.0, raw_norm * 0.1, raw_norm * (1 - 1e-6)):
        pre_step_model, model, optimizer, state, _ = _run_control(
            optimizer_factory=optimizer_factory, gradient_clip_norm=clip_norm
        )
        shadow_model, shadow_optimizer, _ = _run_oracle_shadow_ordinary(
            pre_step_model, state, optimizer_factory=optimizer_factory, clip_norm=clip_norm
        )
        for control_param, shadow_param in zip(
            model.parameters(), shadow_model.parameters(), strict=True
        ):
            _assert_tensor_close(control_param.detach(), shadow_param.detach(), atol=atol)
        _assert_nested_close(optimizer.state_dict(), shadow_optimizer.state_dict(), atol=atol)


class _ToyWavefunction(torch.nn.Module):
    """One-parameter stand-in used only for B5's state_dict continuation."""

    def __init__(self, initial: float = 1.5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([initial], dtype=torch.float64))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.weight.expand(batch_size)


def test_legacy_update_method_state_dict_round_trip_matches_oracle_continued_optimizer_state() -> None:
    """The ``VMCUpdateMethod.state_dict``/``load_state_dict`` seam DF1 will lean on."""

    energy_step1 = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    energy_step2 = torch.tensor([0.2, -1.0, 2.5, -0.5], dtype=torch.float64)

    def _oracle_step(model: _ToyWavefunction, update: LegacyAutogradUpdate, energy: torch.Tensor) -> None:
        logabs = model(energy.shape[0])
        result = oracle_vmc_objective([logabs], [energy])
        update.optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        update.optimizer.step()

    control_model = _ToyWavefunction()
    control_optimizer = torch.optim.Adam(control_model.parameters(), lr=0.05)
    control_update = LegacyAutogradUpdate(
        control_optimizer,
        model_parameters=ModelParameterBinding(parameters=tuple(control_model.parameters())),
    )
    _oracle_step(control_model, control_update, energy_step1)
    saved_state = control_update.state_dict()

    reloaded_model = _ToyWavefunction()
    with torch.no_grad():
        reloaded_model.weight.copy_(control_model.weight)
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=0.05)
    reloaded_update = LegacyAutogradUpdate(
        reloaded_optimizer,
        model_parameters=ModelParameterBinding(parameters=tuple(reloaded_model.parameters())),
    )
    reloaded_update.load_state_dict(saved_state)

    _oracle_step(control_model, control_update, energy_step2)
    _oracle_step(reloaded_model, reloaded_update, energy_step2)

    atol = _adam_recursion_tolerance_envelope(energy_step2.abs())
    _assert_tensor_close(control_model.weight.detach(), reloaded_model.weight.detach(), atol=atol)
    _assert_nested_close(control_optimizer.state_dict(), reloaded_optimizer.state_dict(), atol=atol)


# =============================================================================
# Group C -- closure/custom optimizer shape; event/callback/checkpoint identity
# =============================================================================


class _ClosureLBFGSUpdate(VMCUpdateMethod):
    """Test-authored CLOSURE/CUSTOM optimizer shape: real ``torch.optim.LBFGS``.

    LBFGS genuinely requires a closure (``optimizer.step(closure)``), so this
    exercises the ``VMCUpdateMethod`` extensibility seam against real PyTorch
    closure semantics rather than a synthetic stand-in.
    """

    def __init__(self, optimizer: torch.optim.LBFGS, model_parameters: ModelParameterBinding) -> None:
        self.optimizer = optimizer
        self.model_parameters = model_parameters
        self._backward_scope = None
        self._optimizer_scope = None

    def update_state(self) -> VMCUpdateState:
        return VMCUpdateState(optimizer=self.optimizer, model_parameters=self.model_parameters)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self.model_parameters = model_parameters

    def set_step_scopes(self, *, backward_scope=None, optimizer_scope=None) -> None:
        self._backward_scope = backward_scope
        self._optimizer_scope = optimizer_scope

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        objective = update_input.objective
        if not objective.requires_grad:
            if update_input.batch.n_electrons == 0:
                return VMCUpdateResult(applied=False, grad_norm=0.0)
            raise RuntimeError(
                "VMC loss is disconnected from model parameters for a nonzero-electron batch"
            )

        # LBFGS calls its closure repeatedly PER `step()` call, re-evaluating
        # loss/gradient at each internal quasi-Newton iterate. LBFGS also
        # mutates parameters IN PLACE between those calls, so re-running
        # `objective.backward()` on the same retained graph after such a
        # mutation raises "modified by an in-place operation" -- the graph's
        # saved tensors are versioned, and the in-place step bumps that
        # version out from under the retained backward. Gradient is therefore
        # computed exactly ONCE here; the closure replays the same already-
        # computed loss value on every subsequent call. This tests the
        # trainer's compatibility with a closure-consuming optimizer, not a
        # claim that this reproduces textbook LBFGS's fresh-gradient history.
        self.optimizer.zero_grad(set_to_none=True)
        if self._backward_scope is None:
            objective.backward()
        else:
            with self._backward_scope(update_input.step):
                objective.backward()
        cached_loss = objective.detach()

        def closure():
            return cached_loss

        if self._optimizer_scope is None:
            self.optimizer.step(closure)
        else:
            with self._optimizer_scope(update_input.step):
                self.optimizer.step(closure)
        return VMCUpdateResult(
            applied=True, grad_norm=_gradient_l2_norm(self.model_parameters.parameters)
        )


def _lbfgs_optimizer_factory(params) -> torch.optim.LBFGS:
    return torch.optim.LBFGS(params, lr=0.5, max_iter=5, history_size=5)


def _closure_update_method_factory(optimizer, model) -> _ClosureLBFGSUpdate:
    return _ClosureLBFGSUpdate(optimizer, ModelParameterBinding(parameters=tuple(model.parameters())))


def test_full_trainer_step_matches_oracle_closure_based_custom_update_method() -> None:
    """Optimizer update; CLOSURE/CUSTOM shape. Comparison surface as in the ordinary test."""

    pre_step_model, model, optimizer, state, _ = _run_control(
        optimizer_factory=_lbfgs_optimizer_factory,
        update_method_factory=_closure_update_method_factory,
    )

    shadow_model = copy.deepcopy(pre_step_model)
    output = shadow_model(state.batch)
    result = oracle_vmc_objective([output.logabs], [state.local_energy])
    shadow_optimizer = _lbfgs_optimizer_factory(shadow_model.parameters())

    shadow_optimizer.zero_grad(set_to_none=True)
    result.loss.backward()
    cached_loss = result.loss.detach()

    def closure():
        return cached_loss

    shadow_optimizer.step(closure)

    atol = _parameter_tolerance_envelope(state.local_energy.abs())
    for control_param, shadow_param in zip(model.parameters(), shadow_model.parameters(), strict=True):
        _assert_tensor_close(control_param.detach(), shadow_param.detach(), atol=atol)


def test_typed_event_sequence_identical_between_ordinary_and_closure_update_methods() -> None:
    """Typed event sequence; both shapes, compared to each other (RELATIVE)."""

    _, _, _, _, ordinary_context = _run_control(
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.01)
    )
    _, _, _, _, closure_context = _run_control(
        optimizer_factory=_lbfgs_optimizer_factory,
        update_method_factory=_closure_update_method_factory,
    )

    assert _occurrence_labels(closure_context.occurrences) == _occurrence_labels(
        ordinary_context.occurrences
    )


def test_ordinary_shape_typed_event_sequence_matches_explicit_expected_order() -> None:
    """Typed event sequence; ABSOLUTE order, so two implementations wrong the
    same way cannot both pass the relative comparison above silently.

    ``Backward``/``OptimizerUpdate`` scopes are injected by
    ``set_step_scopes`` (``update.py:404``, consumed at ``:468-481``) rather
    than opened by the trainer directly, so this is the test that would
    notice either shape silently dropping both events.
    """

    _, _, _, _, context = _run_control(optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.01))

    expected = [
        ("started", TrainingIteration),
        ("started", CollectSamples),
        ("ended", CollectSamples),
        ("started", BuildBatch),
        ("ended", BuildBatch),
        ("started", LocalEnergy),
        ("ended", LocalEnergy),
        ("started", Forward),
        ("ended", Forward),
        ("started", Objective),
        ("ended", Objective),
        ("started", Backward),
        ("ended", Backward),
        ("started", OptimizerUpdate),
        ("ended", OptimizerUpdate),
        ("event", UpdateCompleted),
        ("started", Metrics),
        ("ended", Metrics),
        ("event", TrainingIterationCompleted),
        ("ended", TrainingIteration),
    ]
    assert _occurrence_labels(context.occurrences) == expected


def test_data_integrity_and_checkpoint_callback_effects_agree_across_update_method_shapes(
    tmp_path,
) -> None:
    """Callback effects, checkpoint outcomes; both shapes."""

    def _run_with_callbacks(*, subdir: str, optimizer_factory, update_method_factory=None):
        torch.manual_seed(0)
        model = build_tiny_spenn()
        sampler = build_tiny_sampler()
        optimizer = optimizer_factory(model.parameters())
        trainer_kwargs: dict = {"max_steps": 1}
        if update_method_factory is not None:
            trainer_kwargs["update_method"] = update_method_factory(optimizer, model)
        trainer = VMCTrainer(**trainer_kwargs)
        checkpoint_dir = tmp_path / subdir / "checkpoints"
        data_integrity = DataIntegrity(fail_fast=True)
        checkpoint = Checkpoint(output_dir=checkpoint_dir, every_n_steps=1)
        logger = RecordingLogger()
        context = make_run_context(
            tmp_path / subdir / "run",
            callbacks=[data_integrity, checkpoint],
            loggers=[logger],
            run_id=subdir,
        )
        trainer.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=_hooke_terms(),
            optimizer=optimizer,
            context=context,
            emit=lambda name, *, state=None, payload=None, step=None: None,
        )
        return checkpoint_dir, logger

    ordinary_dir, ordinary_logger = _run_with_callbacks(
        subdir="ordinary", optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.01)
    )
    closure_dir, closure_logger = _run_with_callbacks(
        subdir="closure",
        optimizer_factory=_lbfgs_optimizer_factory,
        update_method_factory=_closure_update_method_factory,
    )

    for logger in (ordinary_logger, closure_logger):
        records = logger.by_namespace("checks/data_integrity")
        assert len(records) == 1
        assert records[0].metrics["passed"] is True

    ordinary_manifest = json.loads((ordinary_dir / "step_000001" / "manifest.json").read_text())
    closure_manifest = json.loads((closure_dir / "step_000001" / "manifest.json").read_text())
    assert set(ordinary_manifest["files"]) == set(closure_manifest["files"])
    assert ordinary_manifest["next_iteration"] == closure_manifest["next_iteration"] == 1
    assert ordinary_manifest["completed_updates"] == closure_manifest["completed_updates"] == 1


# =============================================================================
# Divergence characterization (D1): observed, not required; no production change
# =============================================================================


def test_production_compute_vmc_objective_admits_nonfinite_logabs_paired_with_finite_energy_today() -> None:
    """Pins D1 (note ``df0-oracle-vs-production-divergences``): NOT a requirement.

    ``compute_vmc_objective``'s finite mask (``vmc.py:95``) is built from
    ``local_energy`` alone; a finite-energy row with a nonfinite ``logabs``
    is not excluded, and floods ``loss``/gradient with a nonfinite value
    instead of raising a coordinated failure. This slice makes no production
    change; it only pins the observed behavior.
    """

    logabs = torch.tensor([0.1, float("nan"), 0.3], dtype=torch.float64, requires_grad=True)
    local_energy = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    result = compute_vmc_objective(logabs, local_energy)

    assert result.metrics["local_energy_n_finite"] == 3
    assert not math.isfinite(result.loss.item())
