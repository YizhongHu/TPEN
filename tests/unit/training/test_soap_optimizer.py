"""Unit and integration tests for the pure-PyTorch multi-mode SOAP optimizer.

Covers every item in the Task Orchestrator acceptance contract:
- Hyperparameter validation, Hydra ``make_optimizer`` construction, and
  ``UpdateMethodDescription`` CSV/JSONL sink safety.
- Independent reference numerical oracles for 2D matrices and 3D rank-3 tensors
  (including off-diagonal correlation, mixed preconditioned/unpreconditioned
  axes, step-1 ``eigh`` initialization, and subspace-iteration QR refreshes).
- 0D/1D parity with ``torch.optim.AdamW`` and opt-in ``precondition_1d=True``.
- Float64 preservation on ``(32, 32, 32)`` TPEN tensor-product weights and real
  ``build_tiny_spenn()`` models (including disconnected even-electron
  parameters with ``grad is None``).
- Honest first-step update accounting across real and vacuum ``VMCTrainer``
  steps.
- Permanently-zero, zero-to-nonzero, nonzero-to-zero, degenerate factors, and
  atomic refusal of non-finite or sparse gradients.
- Controlled gradient-shock displacement, basis rotation, and recovery probe.
- Real ``save_checkpoint`` / ``restore_checkpoint(mode="train_resume")`` parity
  across both pre-refresh and post-refresh boundaries, plus ``load_state_dict``
  validation and atomicity.
- ``OptimizerState`` callback compatibility with ``SOAP`` across real CSV and
  JSONL sinks while continuing to refuse unsupported optimizers/methods.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from typeguard import TypeCheckError, suppress_type_checks

from tpen.callback import OptimizerState
from tpen.checkpoint import restore_checkpoint, save_checkpoint
from tpen.events import Started
from tpen.logging import CSV, JSONL, LogRecord
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, HarmonicTrap
from tpen.training import SOAP, VMCTrainer, make_optimizer
from tpen.training.events import TrainingIteration
from tpen.training.soap import _project, _project_back, _qr_refresh_basis
from tpen.training.state import TrainerState
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ModelParameterBinding,
    SETTING_NAME_COLLISIONS_KEY,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
)
from tests.helpers.hooke_models import (
    INACTIVE_PAIR_PARAMETERS,
    PAIR_TRAIN_CONFIG,
    build_tiny_hamiltonian_terms,
    build_tiny_sampler,
    build_tiny_spenn,
    pair_parameters_by_name,
    tiny_pair_batch,
)
from tests.unit.callback.support import RecordingContext, deliver_completed_iteration
from tests.unit.callback.test_optimizer_state import GroupedModel, LoggerContext
from tests.unit.training.test_vmc_trainer_tpen_smoke import (
    _ConstantWavefunction,
    _StubContext,
    _VacuumSampler,
)


# ---------------------------------------------------------------------------
# Independent matricized reference oracle (does not call SOAP's internals)
# ---------------------------------------------------------------------------


def _matricize(tensor: torch.Tensor, axis: int) -> torch.Tensor:
    """Unfold ``tensor`` along ``axis`` into a 2D matrix ``(shape[axis], -1)``."""

    moved = torch.movedim(tensor, axis, 0)
    return moved.reshape(tensor.shape[axis], -1)


def _mode_matmul(tensor: torch.Tensor, matrix: torch.Tensor, axis: int) -> torch.Tensor:
    """Apply ``matrix`` along ``axis`` via explicit unfolding and folding."""

    moved = torch.movedim(tensor, axis, 0)
    tail_shape = moved.shape[1:]
    folded = (matrix @ moved.reshape(moved.shape[0], -1)).reshape(matrix.shape[0], *tail_shape)
    return torch.movedim(folded, 0, axis)


def _oracle_eigh(sym: torch.Tensor) -> torch.Tensor:
    _, vecs = torch.linalg.eigh(sym)
    vecs = vecs.flip(1)
    for col in range(vecs.shape[1]):
        pivot = int(torch.argmax(vecs[:, col].abs()).item())
        if float(vecs[pivot, col].item()) < 0.0:
            vecs[:, col] = -vecs[:, col]
    return vecs


def _oracle_soap_trajectory(
    init_param: torch.Tensor,
    grads: list[torch.Tensor],
    *,
    lr: float,
    betas: tuple[float, float],
    shampoo_beta: float,
    eps: float,
    weight_decay: float,
    precondition_frequency: int,
    max_precond_dim: int,
    precondition_1d: bool = False,
    correct_bias: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run an independent mode-unfolding SOAP oracle over a sequence of gradients."""

    beta1, beta2 = betas
    s_beta = beta2 if shampoo_beta < 0.0 else shampoo_beta
    param = init_param.detach().clone()
    shape = tuple(param.shape)
    ndim = param.ndim
    if ndim == 0 or 0 in shape:
        precond = [False] * ndim
    elif ndim == 1:
        precond = [bool(precondition_1d and 1 < shape[0] <= max_precond_dim)]
    else:
        precond = [1 < d <= max_precond_dim for d in shape]

    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)
    gg = [torch.zeros((d, d), dtype=param.dtype) if flag else None for d, flag in zip(shape, precond)]
    q_mats = [torch.eye(d, dtype=param.dtype) if flag else None for d, flag in zip(shape, precond)]
    eigh_done = [False] * ndim

    step_int = 0
    for step_int, grad in enumerate(grads, start=1):
        # Explicit mode-by-mode forward projection using the active eigenbasis Q
        # (initially I on step 1, per Vyas et al., 2024, Algorithm 1).
        grad_proj = grad.clone()
        for axis in range(ndim):
            if q_mats[axis] is not None:
                grad_proj = _mode_matmul(grad_proj, q_mats[axis].T, axis)

        exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (grad_proj * grad_proj)

        exp_avg_proj = exp_avg.clone()
        for axis in range(ndim):
            if q_mats[axis] is not None:
                exp_avg_proj = _mode_matmul(exp_avg_proj, q_mats[axis].T, axis)

        if correct_bias:
            bc1 = 1.0 - beta1**step_int
            bc2 = 1.0 - beta2**step_int
            step_dir_proj = (exp_avg_proj / bc1) / ((exp_avg_sq / bc2).sqrt() + eps)
        else:
            step_dir_proj = exp_avg_proj / (exp_avg_sq.sqrt() + eps)

        # Explicit mode-by-mode backward projection: apply Q_k along each axis k
        step_dir = step_dir_proj
        for axis in range(ndim):
            if q_mats[axis] is not None:
                step_dir = _mode_matmul(step_dir, q_mats[axis], axis)

        if weight_decay != 0.0:
            param = param * (1.0 - lr * weight_decay)
        param = param - lr * step_dir

        # Post-step preconditioner update: update GG EMAs, initialize Q on the
        # first nonzero step for each mode, and run QR refreshes on multiples
        # of precondition_frequency.
        for axis in range(ndim):
            if gg[axis] is None:
                continue
            unfolded = _matricize(grad, axis)
            gram = unfolded @ unfolded.T
            gram = 0.5 * (gram + gram.T)
            gg[axis] = s_beta * gg[axis] + (1.0 - s_beta) * gram

        for axis in range(ndim):
            if gg[axis] is None:
                continue
            sym = 0.5 * (gg[axis] + gg[axis].T)
            if float(sym.abs().max().item()) == 0.0:
                continue
            if not eigh_done[axis]:
                q_mats[axis] = _oracle_eigh(sym)
                eigh_done[axis] = True
            elif step_int % precondition_frequency == 0:
                q_old = q_mats[axis]
                assert q_old is not None
                est_eig = torch.diag(q_old.T @ sym @ q_old)
                sort_idx = torch.argsort(est_eig, descending=True, stable=True)
                q_sorted = q_old[:, sort_idx]
                exp_avg_sq = exp_avg_sq.index_select(axis, sort_idx)
                max_eig = torch.clamp(est_eig.max(), min=0.0)
                if float(max_eig.item()) == 0.0:
                    q_mats[axis] = q_sorted
                else:
                    finfo = torch.finfo(sym.dtype)
                    shift = torch.clamp(
                        max_eig * math.sqrt(finfo.eps), min=math.sqrt(finfo.tiny)
                    )
                    eye = torch.eye(sym.shape[0], device=sym.device, dtype=sym.dtype)
                    q_new, r_mat = torch.linalg.qr((sym + shift * eye) @ q_sorted)
                    signs = torch.where(torch.diagonal(r_mat) < 0, -1.0, 1.0).to(
                        dtype=q_new.dtype
                    )
                    q_mats[axis] = q_new * signs.unsqueeze(0)

    return param, {
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
        "GG": gg,
        "Q": q_mats,
        "mode_eigh_done": eigh_done,
        "step": step_int,
    }


class _ResumeContext(_StubContext):
    """Minimal ``RunContext`` supporting both ``VMCTrainer`` and real checkpoints."""

    def __init__(self, cfg: Any, run_dir: Path) -> None:
        super().__init__()
        self.cfg = cfg
        self._run_dir = Path(run_dir)
        self.metadata = SimpleNamespace(
            device="cpu",
            dtype="float64",
            run_id="soap-resume-test",
            git_commit="0" * 40,
            git_branch="agent/opt-d5d3521b-soap-optimizer",
            dirty_worktree=False,
            command="pytest",
            extra={},
        )

    @property
    def run_dir(self) -> Path:
        return self._run_dir


# ---------------------------------------------------------------------------
# 1. Hyperparameter validation, Hydra construction, and description sinks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": 0.0},
        {"lr": -1e-3},
        {"lr": float("nan")},
        {"lr": float("inf")},
        {"lr": True},
        {"betas": (0.9,)},
        {"betas": (-0.1, 0.9)},
        {"betas": (0.9, 1.0)},
        {"betas": (float("nan"), 0.9)},
        {"betas": (True, 0.9)},
        {"shampoo_beta": -0.5},
        {"shampoo_beta": 1.0},
        {"shampoo_beta": float("inf")},
        {"shampoo_beta": False},
        {"eps": 0.0},
        {"eps": -1e-8},
        {"eps": float("nan")},
        {"weight_decay": -0.01},
        {"weight_decay": float("inf")},
        {"precondition_frequency": 0},
        {"precondition_frequency": -2},
        {"precondition_frequency": True},
        {"max_precond_dim": 0},
        {"max_precond_dim": False},
        {"precondition_1d": 1},
        {"correct_bias": 0},
    ],
)
def test_invalid_hyperparameters_raise_value_error(kwargs: dict[str, Any]) -> None:
    param = nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
    with pytest.raises((ValueError, TypeCheckError)):
        SOAP([param], **kwargs)
    # Also verify SOAP's own runtime guards raise ValueError without typeguard.
    with suppress_type_checks(), pytest.raises(ValueError):
        SOAP([param], **kwargs)


def test_invalid_per_group_override_rolls_back_param_group() -> None:
    p1 = nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
    p2 = nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
    opt = SOAP([p1], lr=1e-3)
    with pytest.raises(ValueError, match="precondition_frequency"):
        opt.add_param_group({"params": [p2], "precondition_frequency": 0})
    assert len(opt.param_groups) == 1


def test_hydra_make_optimizer_and_update_method_description_sink_safety(tmp_path: Path) -> None:
    model = GroupedModel().double()
    cfg = OmegaConf.create(
        {
            "_target_": "tpen.training.SOAP",
            "_partial_": True,
            "lr": 0.003,
            "betas": [0.9, 0.99],
            "shampoo_beta": 0.99,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "precondition_frequency": 10,
            "max_precond_dim": 128,
            "precondition_1d": False,
            "correct_bias": True,
        }
    )
    optimizer = make_optimizer(cfg, model.parameters())
    assert isinstance(optimizer, SOAP)
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.99)
    assert optimizer.param_groups[0]["shampoo_beta"] == 0.99

    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))
    adapter = LegacyAutogradUpdate(optimizer, model_parameters=binding)
    description = adapter.describe()
    assert description.carrier_class == "tpen.training.soap.SOAP"

    metrics = description.as_metrics()
    assert f"update_method_{SETTING_NAME_COLLISIONS_KEY}" not in metrics
    assert metrics["update_method_setting_g0_betas1"] == 0.9
    assert metrics["update_method_setting_g0_betas2"] == 0.99
    assert metrics["update_method_setting_g0_shampoo_beta"] == 0.99
    assert metrics["update_method_setting_g0_precondition_frequency"] == 10
    assert metrics["update_method_setting_g0_max_precond_dim"] == 128

    csv_path = tmp_path / "desc.csv"
    jsonl_path = tmp_path / "desc.jsonl"
    record = LogRecord(step=0, namespace="train/update_method", metrics=metrics)
    CSV(csv_path).log(record)
    JSONL(jsonl_path).log(record)

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(None not in row for row in rows)
    with jsonl_path.open(encoding="utf-8") as stream:
        parsed = json.loads(stream.readline())
    assert parsed["metrics"]["update_method_carrier_class"] == "tpen.training.soap.SOAP"


# ---------------------------------------------------------------------------
# 2. Independent numerical oracles (2D matrix, 3D rank-3, mixed axes)
# ---------------------------------------------------------------------------


def test_project_and_project_back_are_exact_orthogonal_inverses_rank3() -> None:
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(4, 3, 5, generator=gen, dtype=torch.float64)
    q0, _ = torch.linalg.qr(torch.randn(4, 4, generator=gen, dtype=torch.float64))
    q2, _ = torch.linalg.qr(torch.randn(5, 5, generator=gen, dtype=torch.float64))
    q_mats = [q0, None, q2]
    roundtrip = _project_back(_project(x, q_mats), q_mats)
    assert torch.allclose(roundtrip, x, atol=1e-13, rtol=1e-13)


def test_2d_matrix_matches_independent_oracle_and_differs_from_adamw() -> None:
    gen = torch.Generator().manual_seed(11)
    init = torch.randn(4, 3, generator=gen, dtype=torch.float64)
    # Build strongly correlated off-diagonal gradients so eigenbasis rotation matters.
    mix_l = torch.tensor(
        [[1.0, 0.7, -0.4, 0.2], [0.7, 1.2, 0.5, -0.3], [-0.4, 0.5, 0.9, 0.6], [0.2, -0.3, 0.6, 1.1]],
        dtype=torch.float64,
    )
    mix_r = torch.tensor([[1.0, -0.6, 0.3], [-0.6, 1.3, 0.5], [0.3, 0.5, 0.8]], dtype=torch.float64)
    grads = [
        mix_l @ torch.randn(4, 3, generator=gen, dtype=torch.float64) @ mix_r
        for _ in range(6)
    ]

    kwargs = dict(
        lr=2e-2,
        betas=(0.9, 0.95),
        shampoo_beta=0.85,
        eps=1e-8,
        weight_decay=0.01,
        precondition_frequency=2,
        max_precond_dim=16,
    )
    p_soap = nn.Parameter(init.clone())
    opt_soap = SOAP([p_soap], **kwargs)
    p_adamw = nn.Parameter(init.clone())
    opt_adamw = torch.optim.AdamW([p_adamw], lr=2e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    for idx, g in enumerate(grads, start=1):
        p_soap.grad = g.clone()
        opt_soap.step()
        p_adamw.grad = g.clone()
        opt_adamw.step()
        if idx == 1:
            # Post-step preconditioner order (Vyas et al. 2024 Algorithm 1):
            # step 1 executes in Q = I (matching AdamW without 0 / eps SVD
            # roundoff amplification) and initializes Q post-step for step 2+.
            assert torch.allclose(p_soap.detach(), p_adamw.detach(), atol=1e-13, rtol=1e-13)

    ref_param, ref_state = _oracle_soap_trajectory(init, grads, **kwargs)
    st = opt_soap.state[p_soap]
    assert int(st["step"].item()) == 6
    assert torch.allclose(p_soap.detach(), ref_param, atol=1e-12, rtol=1e-12)
    assert torch.allclose(st["exp_avg"], ref_state["exp_avg"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(st["exp_avg_sq"], ref_state["exp_avg_sq"], atol=1e-12, rtol=1e-12)
    for axis in range(2):
        assert torch.allclose(st["GG"][axis], ref_state["GG"][axis], atol=1e-12, rtol=1e-12)
        assert torch.allclose(st["Q"][axis], ref_state["Q"][axis], atol=1e-12, rtol=1e-12)

    # Discriminating control: rotated SOAP steps 2..6 materially differ from coordinatewise AdamW.
    assert float((p_soap.detach() - p_adamw.detach()).abs().max().item()) > 1e-3


def test_rank3_tensor_unmerged_modes_and_mixed_precond_dim_match_oracle() -> None:
    gen = torch.Generator().manual_seed(23)
    # Full unmerged rank-3 tensor: all 3 axes preconditioned
    init3 = torch.randn(4, 3, 5, generator=gen, dtype=torch.float64)
    a0 = torch.randn(4, 4, generator=gen, dtype=torch.float64)
    a1 = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    a2 = torch.randn(5, 5, generator=gen, dtype=torch.float64)
    grads3 = []
    for _ in range(5):
        raw = torch.randn(4, 3, 5, generator=gen, dtype=torch.float64)
        correlated = _mode_matmul(_mode_matmul(_mode_matmul(raw, a0, 0), a1, 1), a2, 2)
        grads3.append(correlated)

    kwargs3 = dict(
        lr=1e-2,
        betas=(0.9, 0.98),
        shampoo_beta=-1.0,
        eps=1e-8,
        weight_decay=0.0,
        precondition_frequency=2,
        max_precond_dim=16,
    )
    p3 = nn.Parameter(init3.clone())
    opt3 = SOAP([p3], **kwargs3)
    for g in grads3:
        p3.grad = g.clone()
        opt3.step()

    ref3, ref3_state = _oracle_soap_trajectory(init3, grads3, **kwargs3)
    st3 = opt3.state[p3]
    assert [tuple(m.shape) for m in st3["GG"]] == [(4, 4), (3, 3), (5, 5)]
    assert [tuple(m.shape) for m in st3["Q"]] == [(4, 4), (3, 3), (5, 5)]
    assert torch.allclose(p3.detach(), ref3, atol=1e-12, rtol=1e-12)
    assert torch.allclose(st3["exp_avg_sq"], ref3_state["exp_avg_sq"], atol=1e-12, rtol=1e-12)

    # Mixed axes: middle axis exceeds max_precond_dim=6 and stays None
    init_mixed = torch.randn(4, 9, 3, generator=gen, dtype=torch.float64)
    grads_mixed = [torch.randn(4, 9, 3, generator=gen, dtype=torch.float64) for _ in range(4)]
    kwargs_mixed = dict(
        lr=1e-2,
        betas=(0.9, 0.95),
        shampoo_beta=0.9,
        eps=1e-8,
        weight_decay=0.02,
        precondition_frequency=2,
        max_precond_dim=6,
    )
    p_mixed = nn.Parameter(init_mixed.clone())
    opt_mixed = SOAP([p_mixed], **kwargs_mixed)
    for g in grads_mixed:
        p_mixed.grad = g.clone()
        opt_mixed.step()

    ref_mixed, _ = _oracle_soap_trajectory(init_mixed, grads_mixed, **kwargs_mixed)
    st_mixed = opt_mixed.state[p_mixed]
    assert st_mixed["GG"][0] is not None and tuple(st_mixed["GG"][0].shape) == (4, 4)
    assert st_mixed["GG"][1] is None and st_mixed["Q"][1] is None
    assert st_mixed["GG"][2] is not None and tuple(st_mixed["GG"][2].shape) == (3, 3)
    assert torch.allclose(p_mixed.detach(), ref_mixed, atol=1e-12, rtol=1e-12)


def test_step1_float32_2d_does_not_amplify_svd_zero_roundoff() -> None:
    """Step 1 in float32 stays close to float64 with no 0 / eps SVD noise blowup."""

    gen = torch.Generator().manual_seed(7)
    g32 = torch.randn(32, 32, generator=gen, dtype=torch.float32)

    p64 = nn.Parameter(torch.zeros(32, 32, dtype=torch.float64))
    opt64 = SOAP([p64], lr=1e-2, betas=(0.95, 0.95), eps=1e-8)
    p64.grad = g32.to(torch.float64)
    opt64.step()
    delta64 = p64.detach().clone()

    p32 = nn.Parameter(torch.zeros(32, 32, dtype=torch.float32))
    opt32 = SOAP([p32], lr=1e-2, betas=(0.95, 0.95), eps=1e-8)
    p32.grad = g32.clone()
    opt32.step()
    delta32 = p32.detach().to(torch.float64)

    rel_err = float((delta32 - delta64).norm() / delta64.norm())
    assert rel_err < 1e-4

    # Step 1 matches AdamW cleanly and initializes orthonormal float32 Q post-step.
    p32_adam = nn.Parameter(torch.zeros(32, 32, dtype=torch.float32))
    opt32_adam = torch.optim.AdamW(
        [p32_adam], lr=1e-2, betas=(0.95, 0.95), eps=1e-8, weight_decay=0.0
    )
    p32_adam.grad = g32.clone()
    opt32_adam.step()
    assert torch.allclose(p32.detach(), p32_adam.detach(), atol=1e-6, rtol=1e-6)
    assert opt32.state[p32]["mode_eigh_done"] == [True, True]
    eye32 = torch.eye(32, dtype=torch.float32)
    for q in opt32.state[p32]["Q"]:
        assert q is not None and q.dtype == torch.float32
        assert torch.allclose(q.T @ q, eye32, atol=1e-5, rtol=1e-5)


def test_step1_and_step2_ill_conditioned_float64_no_spurious_offdiag_step() -> None:
    """Ill-conditioned float64 G_1 = U diag([1e3, 1e-2, 1e-3, 1e-3]) V^T avoids O(1) noise."""

    h = (
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        / 2.0
    )
    c, s = math.cos(0.3), math.sin(0.3)
    rot = torch.tensor(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, c, s], [0.0, 0.0, -s, c]],
        dtype=torch.float64,
    )
    u = rot @ h
    v = rot.T @ h
    sing = torch.tensor([1e3, 1e-2, 1e-3, 1e-3], dtype=torch.float64)
    g1 = u @ torch.diag(sing) @ v.T

    lr = 1e-2
    eps = 1e-8
    p = nn.Parameter(torch.zeros(4, 4, dtype=torch.float64))
    opt = SOAP([p], lr=lr, betas=(0.95, 0.95), eps=eps, weight_decay=0.0, precondition_frequency=10)

    # Step 1 executes in Q = I (exact AdamW, zero 0 / eps SVD amplification) and
    # initializes orthonormal Q_1 post-step.
    p.grad = g1.clone()
    opt.step()
    expected_step1 = -lr * (g1 / (g1.abs() + eps))
    assert torch.allclose(p.detach(), expected_step1, atol=1e-13, rtol=1e-13)
    q1 = [q.clone() for q in opt.state[p]["Q"]]
    assert opt.state[p]["mode_eigh_done"] == [True, True]
    for q in q1:
        assert torch.allclose(q.T @ q, torch.eye(4, dtype=torch.float64), atol=1e-12)

    # Step 2 runs in the rotated Q_1 basis; because exp_avg_sq already carries
    # non-zero step-1 variance, off-diagonal step_dir_proj stays << 1e-2 rather
    # than saturating at O(1).
    p_after_1 = p.detach().clone()
    p.grad = g1.clone()
    opt.step()
    step_dir_2 = -(p.detach() - p_after_1) / lr
    step_dir_proj_2 = _project(step_dir_2, q1)
    offdiag_mask = ~torch.eye(4, dtype=torch.bool)
    assert float(step_dir_proj_2[offdiag_mask].abs().max().item()) < 1e-2
    assert float(step_dir_proj_2[0, 0].abs().item()) > 1.0

    # Diagonal ill-conditioned G_1 also produces exact zero off-diagonal step 1.
    p_diag = nn.Parameter(torch.zeros(4, 4, dtype=torch.float64))
    opt_diag = SOAP([p_diag], lr=lr, betas=(0.95, 0.95), eps=eps, weight_decay=0.0)
    p_diag.grad = torch.diag(sing)
    opt_diag.step()
    step_dir_diag = -p_diag.detach() / lr
    assert float(step_dir_diag[offdiag_mask].abs().max().item()) == 0.0


def test_eigenvalue_crossing_permutes_exp_avg_sq_along_each_axis() -> None:
    """Asymmetric multi-mode eigenvalue crossing permutes exp_avg_sq via index_select."""

    init = torch.zeros(3, 3, dtype=torch.float64)
    # Step 1: GG_0 = GG_1 = diag([25, 4, 0.25]) -> Q_0 = Q_1 = I_3.
    g1 = torch.diag(torch.tensor([5.0, 2.0, 0.5], dtype=torch.float64))
    # Step 2 (refresh): GG_0 = diag([0.04, 9, 100]) -> sort_idx_0 = [2, 1, 0];
    #                   GG_1 = diag([100, 0.04, 9]) -> sort_idx_1 = [0, 2, 1].
    g2 = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 3.0], [10.0, 0.0, 0.0]], dtype=torch.float64)
    g3 = torch.tensor([[1.0, -0.5, 0.2], [0.3, 2.0, -0.4], [-0.1, 0.6, 1.5]], dtype=torch.float64)

    kwargs = dict(
        lr=1e-2,
        betas=(0.9, 0.95),
        shampoo_beta=0.0,
        eps=1e-8,
        weight_decay=0.0,
        precondition_frequency=2,
        max_precond_dim=16,
    )
    p = nn.Parameter(init.clone())
    opt = SOAP([p], **kwargs)

    p.grad = g1.clone()
    opt.step()
    q_before = [q.clone() for q in opt.state[p]["Q"]]
    assert opt.state[p]["mode_eigh_done"] == [True, True]

    # Step 2 triggers post-step QR refresh with distinct non-identity permutations on both axes.
    p.grad = g2.clone()
    opt.step()
    st_after_2 = opt.state[p]
    expected_sorts = [torch.tensor([2, 1, 0]), torch.tensor([0, 2, 1])]
    for axis, expected_sort in enumerate(expected_sorts):
        gg = st_after_2["GG"][axis]
        assert gg is not None
        _, sort_idx = _qr_refresh_basis(0.5 * (gg + gg.T), q_before[axis])
        assert not torch.equal(sort_idx, torch.arange(3))
        assert torch.equal(sort_idx, expected_sort)

    ref_p2, ref_st2 = _oracle_soap_trajectory(init, [g1, g2], **kwargs)
    assert torch.allclose(p.detach(), ref_p2, atol=1e-12, rtol=1e-12)
    assert torch.allclose(st_after_2["exp_avg_sq"], ref_st2["exp_avg_sq"], atol=1e-12, rtol=1e-12)
    # Pin concrete permuted entries (unpermuted exp_avg_sq would keep ~1.1875 at (0, 0) instead of 5.0).
    assert st_after_2["exp_avg_sq"][0, 0].item() == pytest.approx(5.0, abs=1e-12)
    assert st_after_2["exp_avg_sq"][1, 1].item() == pytest.approx(0.45, abs=1e-12)
    assert st_after_2["exp_avg_sq"][2, 0].item() == pytest.approx(1.1875, abs=1e-12)

    # Step 3 consumes the permuted exp_avg_sq in the refreshed eigenbasis.
    p.grad = g3.clone()
    opt.step()
    ref_p3, ref_st3 = _oracle_soap_trajectory(init, [g1, g2, g3], **kwargs)
    assert torch.allclose(p.detach(), ref_p3, atol=1e-12, rtol=1e-12)
    assert torch.allclose(opt.state[p]["exp_avg_sq"], ref_st3["exp_avg_sq"], atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. 0D / 1D AdamW parity and opt-in precondition_1d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_0d_and_1d_reduce_to_adamw_when_precondition_1d_false(weight_decay: float) -> None:
    gen = torch.Generator().manual_seed(31)
    init0 = torch.tensor(0.42, dtype=torch.float64)
    init1 = torch.randn(7, generator=gen, dtype=torch.float64)
    grads0 = [torch.randn((), generator=gen, dtype=torch.float64) for _ in range(5)]
    grads1 = [torch.randn(7, generator=gen, dtype=torch.float64) for _ in range(5)]

    p0_soap, p1_soap = nn.Parameter(init0.clone()), nn.Parameter(init1.clone())
    p0_adam, p1_adam = nn.Parameter(init0.clone()), nn.Parameter(init1.clone())
    opt_soap = SOAP(
        [p0_soap, p1_soap],
        lr=1e-2,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=weight_decay,
        precondition_1d=False,
        correct_bias=True,
    )
    opt_adam = torch.optim.AdamW(
        [p0_adam, p1_adam],
        lr=1e-2,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    for g0, g1 in zip(grads0, grads1, strict=True):
        p0_soap.grad, p1_soap.grad = g0.clone(), g1.clone()
        p0_adam.grad, p1_adam.grad = g0.clone(), g1.clone()
        opt_soap.step()
        opt_adam.step()

    assert opt_soap.state[p0_soap]["GG"] == []
    assert opt_soap.state[p1_soap]["GG"] == [None]
    assert torch.allclose(p0_soap.detach(), p0_adam.detach(), atol=1e-14, rtol=1e-13)
    assert torch.allclose(p1_soap.detach(), p1_adam.detach(), atol=1e-14, rtol=1e-13)


def test_precondition_1d_true_preconditions_vector() -> None:
    gen = torch.Generator().manual_seed(37)
    init = torch.randn(5, generator=gen, dtype=torch.float64)
    grads = [torch.randn(5, generator=gen, dtype=torch.float64) for _ in range(4)]
    p = nn.Parameter(init.clone())
    opt = SOAP([p], lr=1e-2, betas=(0.9, 0.95), precondition_frequency=2, precondition_1d=True)
    for g in grads:
        p.grad = g.clone()
        opt.step()
    ref, ref_state = _oracle_soap_trajectory(
        init,
        grads,
        lr=1e-2,
        betas=(0.9, 0.95),
        shampoo_beta=-1.0,
        eps=1e-8,
        weight_decay=0.0,
        precondition_frequency=2,
        max_precond_dim=128,
        precondition_1d=True,
    )
    st = opt.state[p]
    assert st["GG"][0] is not None and tuple(st["GG"][0].shape) == (5, 5)
    assert st["Q"][0].dtype == torch.float64
    assert torch.allclose(st["Q"][0].T @ st["Q"][0], torch.eye(5, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(st["exp_avg"], ref_state["exp_avg"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(st["GG"][0], ref_state["GG"][0], atol=1e-12, rtol=1e-12)
    # A 1D length-5 vector has rank-2/rank-4 GG on post-step QR refresh steps
    # 2/4 (shift-stabilized QR condition number 1 / sqrt(eps_mach) ~= 6.7e7 in
    # float64), giving O(sqrt(eps_mach)) nullspace sensitivity between
    # independent contraction paths vs O(lr) ~ 6.5e-3 drift without shift
    # stabilization.
    assert torch.allclose(st["Q"][0], ref_state["Q"][0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(p.detach(), ref, atol=1e-7, rtol=1e-7)


def test_qr_refresh_rank_deficient_stability_and_nullspace_preservation() -> None:
    """Shift-stabilized QR refresh preserves nullspace vectors and resists ULP noise."""

    gen = torch.Generator().manual_seed(61)
    q_old, r0 = torch.linalg.qr(torch.randn(5, 5, generator=gen, dtype=torch.float64))
    q_old = q_old * torch.where(torch.diagonal(r0) < 0, -1.0, 1.0).unsqueeze(0)
    eye5 = torch.eye(5, dtype=torch.float64)

    # 1. Rank-1 matrix aligned with q_old[:, 0]: exact nullspace preservation
    #    (modulo permutation of tied zero-eigenvalue columns via sort_idx) and
    #    stability under 1e-15 symmetric perturbation (simulating ULP noise).
    sym_rank1 = 3.5 * torch.outer(q_old[:, 0], q_old[:, 0])
    sym_rank1 = 0.5 * (sym_rank1 + sym_rank1.T)
    q_new1, sort_idx1 = _qr_refresh_basis(sym_rank1, q_old)
    assert int(sort_idx1[0].item()) == 0
    assert torch.allclose(q_new1.T @ q_new1, eye5, atol=1e-13, rtol=1e-13)
    assert torch.allclose(q_new1[:, 0], q_old[:, 0], atol=1e-12, rtol=1e-12)
    # Shift c = max_eig * sqrt(eps_f64) gives kappa(M) = 1 / sqrt(eps_f64) ~= 6.7e7;
    # O(max_eig * eps_f64) ULP roundoff in sym @ q_sorted yields O(sqrt(eps_f64))
    # ~= 1.5e-8 backward error in nullspace columns (vs O(1) drift without shift).
    assert torch.allclose(q_new1, q_old.index_select(1, sort_idx1), atol=1e-6, rtol=1e-6)

    noise = torch.randn(5, 5, generator=gen, dtype=torch.float64)
    sym_noise = 0.5 * (noise + noise.T)
    sym_rank1_pert = sym_rank1 + 1e-15 * sym_noise
    q_pert1, sort_idx_pert1 = _qr_refresh_basis(sym_rank1_pert, q_old)
    assert int(sort_idx_pert1[0].item()) == 0
    assert torch.allclose(q_pert1.T @ q_pert1, eye5, atol=1e-13, rtol=1e-13)
    assert torch.allclose(q_pert1[:, 0], q_old[:, 0], atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        q_pert1, q_old.index_select(1, sort_idx_pert1), atol=1e-6, rtol=1e-6
    )

    # 2. Rank-2 matrix in 5D (3D nullspace) with distinct Rayleigh quotients:
    #    min gap ~ 0.034 >> 1e-15 preserves sort order, and 1e-15 symmetric
    #    noise produces O(1e-15 / shift) ~ 1.5e-7 drift instead of O(1) arbitrary
    #    nullspace rotation (atol=1e-5 gives ~60x margin).
    coeffs = torch.tensor([0.2, 0.7, 0.5, 0.4, 0.3], dtype=torch.float64)
    u2 = q_old @ (coeffs / coeffs.norm())
    sym_rank2 = 2.0 * torch.outer(q_old[:, 0], q_old[:, 0]) + 0.5 * torch.outer(u2, u2)
    sym_rank2 = 0.5 * (sym_rank2 + sym_rank2.T)
    est_r2 = (q_old * (sym_rank2 @ q_old)).sum(dim=0)
    assert float((est_r2[:-1] - est_r2[1:]).min().item()) > 0.03
    q_r2, sort_r2 = _qr_refresh_basis(sym_rank2, q_old)
    q_r2_pert, sort_r2_pert = _qr_refresh_basis(sym_rank2 + 1e-15 * sym_noise, q_old)
    assert torch.equal(sort_r2, sort_r2_pert)
    assert torch.allclose(q_r2.T @ q_r2, eye5, atol=1e-13, rtol=1e-13)
    assert torch.allclose(q_r2_pert.T @ q_r2_pert, eye5, atol=1e-13, rtol=1e-13)
    assert torch.allclose(q_r2, q_r2_pert, atol=1e-5, rtol=1e-5)

    # 3. Float32 small-gradient scale invariance: max_eig=1e-8 must NOT be swamped
    #    by float32 eps (~1.19e-7), so active subspace tracking and scale invariance
    #    both hold across 4+ orders of magnitude (condition number ~ 1/sqrt(eps_f32)
    #    gives O(sqrt(eps_f32)) ~ 3.5e-4 nullspace sensitivity vs ~0.56 if swamped).
    q_old_f32 = torch.eye(4, dtype=torch.float32)
    v_f32 = torch.tensor([0.6, 0.8, 0.0, 0.0], dtype=torch.float32)
    sym_small = 1e-8 * torch.outer(v_f32, v_f32)
    sym_small = 0.5 * (sym_small + sym_small.T)
    sym_scaled = (2.0**14) * sym_small
    q_small, sort_small = _qr_refresh_basis(sym_small, q_old_f32)
    q_scaled, sort_scaled = _qr_refresh_basis(sym_scaled, q_old_f32)
    assert torch.equal(sort_small, sort_scaled)
    assert abs(float((q_small[:, 0] @ v_f32).item())) > 0.999
    assert torch.allclose(q_small.T @ q_small, torch.eye(4, dtype=torch.float32), atol=1e-5)
    assert torch.allclose(q_small, q_scaled, atol=1e-3, rtol=1e-3)

    # 4. Exact-zero matrix preserves q_old and identity ordering cleanly.
    q_old_4 = q_old[:4, :4].clone()
    q_old_4, _ = torch.linalg.qr(q_old_4)
    q_zero, sort_zero = _qr_refresh_basis(torch.zeros(4, 4, dtype=torch.float64), q_old_4)
    assert torch.equal(sort_zero, torch.arange(4))
    assert torch.equal(q_zero, q_old_4)


# ---------------------------------------------------------------------------
# 4. Float64 (32, 32, 32) TPEN weight and real build_tiny_spenn()
# ---------------------------------------------------------------------------


def test_float64_preserved_on_32x32x32_tensor_product_weight() -> None:
    gen = torch.Generator().manual_seed(41)
    p = nn.Parameter(torch.randn(32, 32, 32, generator=gen, dtype=torch.float64))
    opt = SOAP([p], lr=1e-3, betas=(0.9, 0.99), shampoo_beta=0.99, precondition_frequency=2)
    for _ in range(3):
        p.grad = torch.randn(32, 32, 32, generator=gen, dtype=torch.float64)
        opt.step()

    st = opt.state[p]
    assert p.dtype == torch.float64
    assert st["exp_avg"].dtype == torch.float64
    assert st["exp_avg_sq"].dtype == torch.float64
    assert len(st["GG"]) == 3 and len(st["Q"]) == 3
    eye = torch.eye(32, dtype=torch.float64)
    for axis in range(3):
        gg = st["GG"][axis]
        q = st["Q"][axis]
        assert gg is not None and q is not None
        assert gg.dtype == torch.float64 and tuple(gg.shape) == (32, 32)
        assert q.dtype == torch.float64 and tuple(q.shape) == (32, 32)
        assert torch.allclose(q.T @ q, eye, atol=1e-12, rtol=1e-12)


def test_real_tiny_spenn_float64_skips_disconnected_even_electron_parameters() -> None:
    model = build_tiny_spenn()
    named = pair_parameters_by_name(model)
    initial_inactive = {name: named[name].detach().clone() for name in INACTIVE_PAIR_PARAMETERS}

    optimizer = SOAP(model.parameters(), lr=1e-3, betas=(0.9, 0.99), precondition_frequency=2)
    batch = tiny_pair_batch(n_walkers=4)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = out.logabs.square().mean()
        loss.backward()
        optimizer.step()

    # All 16 disconnected n=2 parameters must have no allocated optimizer state
    # and remain bitwise unchanged.
    for name in INACTIVE_PAIR_PARAMETERS:
        param = named[name]
        assert param not in optimizer.state
        assert torch.equal(param.detach(), initial_inactive[name])

    # Active rank-3 mixing weights must have 3 unmerged float64 factor matrices.
    active_rank3 = [
        (name, p)
        for name, p in named.items()
        if p.ndim == 3 and name not in INACTIVE_PAIR_PARAMETERS
    ]
    assert len(active_rank3) == 14
    for _, param in active_rank3:
        st = optimizer.state[param]
        assert int(st["step"].item()) == 2
        assert len(st["GG"]) == 3
        for dim, gg, q in zip(param.shape, st["GG"], st["Q"], strict=True):
            assert gg is not None and gg.dtype == torch.float64 and tuple(gg.shape) == (dim, dim)
            assert q is not None and q.dtype == torch.float64 and tuple(q.shape) == (dim, dim)


# ---------------------------------------------------------------------------
# 5. Honest first-step update accounting & VMCTrainer real vs vacuum batches
# ---------------------------------------------------------------------------


def test_first_step_updates_parameters_and_trainer_accounting_matches_real_and_vacuum() -> None:
    # Direct step-1 check: no silent skipped step 1
    p = nn.Parameter(torch.ones(3, 3, dtype=torch.float64))
    before = p.detach().clone()
    opt = SOAP([p], lr=1e-2)
    p.grad = torch.tensor([[1.0, 0.2, -0.5], [0.3, -0.8, 0.4], [-0.1, 0.6, 0.9]], dtype=torch.float64)
    opt.step()
    assert int(opt.state[p]["step"].item()) == 1
    assert not torch.equal(p.detach(), before)

    # Real VMCTrainer step: completed_updates == 1 and matches active param step
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    soap_opt = SOAP(model.parameters(), lr=1e-3, precondition_frequency=2)
    ctx = _StubContext()
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=soap_opt,
        context=ctx,
        emit=lambda *_, **__: None,
    )
    assert trainer.next_iteration == 1
    assert trainer.completed_updates == 1
    active_steps = {int(st["step"].item()) for st in soap_opt.state.values()}
    assert active_steps == {1}

    # Vacuum VMCTrainer step: update skipped cleanly, completed_updates == 0, no state allocated
    vac_model = _ConstantWavefunction()
    vac_opt = SOAP(vac_model.parameters(), lr=1e-3)
    vac_ctx = _StubContext()
    vac_trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    vac_trainer.fit(
        model=vac_model,
        sampler=_VacuumSampler(),
        hamiltonian_terms=[],
        optimizer=vac_opt,
        context=vac_ctx,
        emit=lambda *_, **__: None,
    )
    assert vac_trainer.next_iteration == 1
    assert vac_trainer.completed_updates == 0
    assert len(vac_opt.state) == 0


# ---------------------------------------------------------------------------
# 6. Zero / degenerate / transition gradients & atomic non-finite/sparse refusal
# ---------------------------------------------------------------------------


def test_zero_transitions_and_degenerate_factors() -> None:
    # Permanently zero gradient with weight_decay=0 leaves parameter untouched
    p_zero = nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, 0.5]], dtype=torch.float64))
    init_zero = p_zero.detach().clone()
    opt_zero = SOAP([p_zero], lr=1e-2, weight_decay=0.0, precondition_frequency=2)
    for _ in range(4):
        p_zero.grad = torch.zeros_like(p_zero)
        opt_zero.step()
    assert torch.equal(p_zero.detach(), init_zero)
    assert opt_zero.state[p_zero]["mode_eigh_done"] == [False, False]
    for q in opt_zero.state[p_zero]["Q"]:
        assert torch.equal(q, torch.eye(2, dtype=torch.float64))

    # Zero on step 1 -> nonzero on step 2..4: eigh initializes on refresh step 2
    p_z2n = nn.Parameter(torch.ones(3, 3, dtype=torch.float64))
    opt_z2n = SOAP([p_z2n], lr=1e-2, precondition_frequency=2)
    p_z2n.grad = torch.zeros(3, 3, dtype=torch.float64)
    opt_z2n.step()
    assert opt_z2n.state[p_z2n]["mode_eigh_done"] == [False, False]
    g_nz = torch.tensor([[1.0, 0.5, -0.2], [0.5, 2.0, 0.1], [-0.2, 0.1, 0.4]], dtype=torch.float64)
    for _ in range(3):
        p_z2n.grad = g_nz.clone()
        opt_z2n.step()
    assert opt_z2n.state[p_z2n]["mode_eigh_done"] == [True, True]
    assert torch.isfinite(p_z2n).all()
    for q in opt_z2n.state[p_z2n]["Q"]:
        assert torch.allclose(q.T @ q, torch.eye(3, dtype=torch.float64), atol=1e-12)

    # Nonzero on step 1 -> zero on steps 2..5: momentum decays smoothly, Q stays orthonormal
    p_n2z = nn.Parameter(torch.ones(3, 3, dtype=torch.float64))
    opt_n2z = SOAP([p_n2z], lr=1e-2, precondition_frequency=2)
    p_n2z.grad = g_nz.clone()
    opt_n2z.step()
    for _ in range(4):
        p_n2z.grad = torch.zeros(3, 3, dtype=torch.float64)
        opt_n2z.step()
    assert torch.isfinite(p_n2z).all()
    for q in opt_n2z.state[p_n2z]["Q"]:
        assert torch.allclose(q.T @ q, torch.eye(3, dtype=torch.float64), atol=1e-12)

    # Degenerate factor (GG proportional to identity c * I)
    p_deg = nn.Parameter(torch.ones(3, 3, dtype=torch.float64))
    opt_deg = SOAP([p_deg], lr=1e-2, precondition_frequency=2)
    for _ in range(4):
        p_deg.grad = 2.0 * torch.eye(3, dtype=torch.float64)
        opt_deg.step()
    assert torch.isfinite(p_deg).all()
    for q in opt_deg.state[p_deg]["Q"]:
        assert torch.allclose(q.T @ q, torch.eye(3, dtype=torch.float64), atol=1e-12)

    # Empty tensor (numel == 0) is handled safely without allocating factors
    p_empty = nn.Parameter(torch.empty(0, 4, dtype=torch.float64))
    opt_empty = SOAP([p_empty], lr=1e-2, precondition_frequency=1)
    p_empty.grad = torch.empty(0, 4, dtype=torch.float64)
    opt_empty.step()
    assert opt_empty.state[p_empty]["GG"] == [None, None]
    assert opt_empty.state[p_empty]["Q"] == [None, None]
    assert int(opt_empty.state[p_empty]["step"].item()) == 1

    # Uncorrected bias mode (correct_bias=False) matches independent oracle
    gen_nb = torch.Generator().manual_seed(43)
    init_nb = torch.randn(3, 3, generator=gen_nb, dtype=torch.float64)
    grads_nb = [torch.randn(3, 3, generator=gen_nb, dtype=torch.float64) for _ in range(3)]
    p_nb = nn.Parameter(init_nb.clone())
    opt_nb = SOAP([p_nb], lr=1e-2, betas=(0.9, 0.95), precondition_frequency=2, correct_bias=False)
    for g in grads_nb:
        p_nb.grad = g.clone()
        opt_nb.step()
    ref_nb, ref_nb_state = _oracle_soap_trajectory(
        init_nb,
        grads_nb,
        lr=1e-2,
        betas=(0.9, 0.95),
        shampoo_beta=-1.0,
        eps=1e-8,
        weight_decay=0.0,
        precondition_frequency=2,
        max_precond_dim=128,
        correct_bias=False,
    )
    st_nb = opt_nb.state[p_nb]
    assert torch.allclose(p_nb.detach(), ref_nb, atol=1e-12, rtol=1e-12)
    assert torch.allclose(st_nb["exp_avg"], ref_nb_state["exp_avg"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(st_nb["exp_avg_sq"], ref_nb_state["exp_avg_sq"], atol=1e-12, rtol=1e-12)
    for axis in range(2):
        assert torch.allclose(st_nb["GG"][axis], ref_nb_state["GG"][axis], atol=1e-12, rtol=1e-12)
        assert torch.allclose(st_nb["Q"][axis], ref_nb_state["Q"][axis], atol=1e-12, rtol=1e-12)


def test_delayed_zero_to_nonzero_initializes_q_immediately_with_large_precond_freq() -> None:
    """Zero on step 1 -> nonzero on step 2 initializes Q immediately even at freq=10."""

    init = torch.tensor(
        [[0.4, -0.2, 0.7], [-0.5, 0.9, -0.1], [0.3, 0.6, -0.8]], dtype=torch.float64
    )
    g1 = torch.zeros(3, 3, dtype=torch.float64)
    g2 = torch.tensor(
        [[1.0, 0.8, -0.4], [0.8, 2.0, 0.3], [-0.4, 0.3, 0.5]], dtype=torch.float64
    )
    g3 = torch.tensor(
        [[-0.6, 1.2, 0.5], [0.9, -0.4, 1.1], [0.2, -0.7, 0.8]], dtype=torch.float64
    )

    kwargs = dict(
        lr=1e-2,
        betas=(0.9, 0.95),
        shampoo_beta=0.9,
        eps=1e-8,
        weight_decay=0.0,
        precondition_frequency=10,
        max_precond_dim=16,
    )
    p = nn.Parameter(init.clone())
    opt = SOAP([p], **kwargs)
    p_adam = nn.Parameter(init.clone())
    opt_adam = torch.optim.AdamW([p_adam], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    # Step 1: zero gradient keeps Q = I and mode_eigh_done = [False, False].
    p.grad = g1.clone()
    opt.step()
    p_adam.grad = g1.clone()
    opt_adam.step()
    assert opt.state[p]["mode_eigh_done"] == [False, False]
    eye3 = torch.eye(3, dtype=torch.float64)
    for q in opt.state[p]["Q"]:
        assert torch.equal(q, eye3)

    # Step 2: first nonzero gradient executes in Q = I (matching AdamW) and
    # immediately initializes Q post-step instead of waiting for step 10.
    p.grad = g2.clone()
    opt.step()
    p_adam.grad = g2.clone()
    opt_adam.step()
    assert torch.allclose(p.detach(), p_adam.detach(), atol=1e-13, rtol=1e-13)
    assert opt.state[p]["mode_eigh_done"] == [True, True]
    for q in opt.state[p]["Q"]:
        assert q is not None
        assert torch.allclose(q.T @ q, eye3, atol=1e-12, rtol=1e-12)
        assert float((q - eye3).abs().max().item()) > 1e-2

    # Step 3: runs in the rotated eigenbasis Q (matches oracle, differs from AdamW).
    p.grad = g3.clone()
    opt.step()
    p_adam.grad = g3.clone()
    opt_adam.step()
    ref_p, ref_st = _oracle_soap_trajectory(init, [g1, g2, g3], **kwargs)
    assert torch.allclose(p.detach(), ref_p, atol=1e-12, rtol=1e-12)
    assert torch.allclose(opt.state[p]["exp_avg_sq"], ref_st["exp_avg_sq"], atol=1e-12, rtol=1e-12)
    assert float((p.detach() - p_adam.detach()).abs().max().item()) > 1e-4


@pytest.mark.parametrize("bad_kind", ["nan", "pos_inf", "neg_inf", "sparse", "sparse_csr"])
def test_atomic_refusal_on_nonfinite_or_sparse_gradient(bad_kind: str) -> None:
    p1 = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64))
    p2 = nn.Parameter(torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64))
    opt = SOAP([p1, p2], lr=1e-2, precondition_frequency=2)

    # Take one valid step so state is populated before the bad step
    p1.grad = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float64)
    p2.grad = torch.tensor([[-0.2, 0.1], [0.5, -0.3]], dtype=torch.float64)
    opt.step()

    p1_before = p1.detach().clone()
    p2_before = p2.detach().clone()
    state_before = opt.state_dict()

    p1.grad = torch.tensor([[0.2, 0.1], [-0.1, 0.3]], dtype=torch.float64)
    if bad_kind == "nan":
        p2.grad = torch.tensor([[1.0, float("nan")], [0.0, 0.0]], dtype=torch.float64)
    elif bad_kind == "pos_inf":
        p2.grad = torch.tensor([[1.0, float("inf")], [0.0, 0.0]], dtype=torch.float64)
    elif bad_kind == "neg_inf":
        p2.grad = torch.tensor([[1.0, float("-inf")], [0.0, 0.0]], dtype=torch.float64)
    elif bad_kind == "sparse":
        p2.grad = torch.sparse_coo_tensor(
            torch.tensor([[0], [1]]),
            torch.tensor([1.0], dtype=torch.float64),
            size=(2, 2),
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p2.grad = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64).to_sparse_csr()

    with pytest.raises(RuntimeError):
        opt.step()

    assert torch.equal(p1.detach(), p1_before)
    assert torch.equal(p2.detach(), p2_before)
    state_after = opt.state_dict()
    assert set(state_after["state"].keys()) == set(state_before["state"].keys())
    for idx in state_before["state"]:
        s_old = state_before["state"][idx]
        s_new = state_after["state"][idx]
        assert torch.equal(s_old["step"], s_new["step"])
        assert torch.equal(s_old["exp_avg"], s_new["exp_avg"])
        assert torch.equal(s_old["exp_avg_sq"], s_new["exp_avg_sq"])
        for g_old, g_new in zip(s_old["GG"], s_new["GG"], strict=True):
            assert torch.equal(g_old, g_new)
        for q_old, q_new in zip(s_old["Q"], s_new["Q"], strict=True):
            assert torch.equal(q_old, q_new)


# ---------------------------------------------------------------------------
# 7. Controlled gradient shock and recovery probe
# ---------------------------------------------------------------------------


def test_controlled_gradient_shock_bounded_displacement_and_recovery() -> None:
    gen = torch.Generator().manual_seed(53)
    p = nn.Parameter(torch.zeros(4, 4, dtype=torch.float64))
    lr = 1e-2
    opt = SOAP([p], lr=lr, betas=(0.9, 0.99), shampoo_beta=0.95, precondition_frequency=2)

    # Steady baseline steps
    for _ in range(4):
        p.grad = 0.1 * torch.randn(4, 4, generator=gen, dtype=torch.float64)
        opt.step()

    q_before_shock = [q.clone() for q in opt.state[p]["Q"]]
    p_before_shock = p.detach().clone()

    # Inject a 1000x rank-1 off-diagonal shock on a non-refresh step (step 5)
    u = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64) / 2.0
    v = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64) / 2.0
    p.grad = 100.0 * torch.outer(u, v)
    opt.step()
    shock_disp = float((p.detach() - p_before_shock).norm().item())

    # Adam-style rotated normalization keeps immediate displacement bounded O(lr),
    # nowhere near proportional to the 1000x shock norm.
    assert shock_disp < 10.0 * lr * (p.numel() ** 0.5)
    assert torch.isfinite(p).all()

    # Next step (step 6) triggers QR refresh: basis must adapt cleanly and stay orthonormal
    recovery_disps: list[float] = []
    for _ in range(6):
        prev = p.detach().clone()
        p.grad = 0.1 * torch.randn(4, 4, generator=gen, dtype=torch.float64)
        opt.step()
        recovery_disps.append(float((p.detach() - prev).norm().item()))

    q_after_refresh = opt.state[p]["Q"]
    basis_shift = max(
        float((q_new - q_old).norm().item())
        for q_new, q_old in zip(q_after_refresh, q_before_shock, strict=True)
    )
    assert basis_shift > 1e-3
    eye = torch.eye(4, dtype=torch.float64)
    for q in q_after_refresh:
        assert torch.isfinite(q).all()
        assert torch.allclose(q.T @ q, eye, atol=1e-12, rtol=1e-12)
    assert all(d < 10.0 * lr * (p.numel() ** 0.5) for d in recovery_disps)
    assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# 8. Real checkpoint / resume parity (pre-refresh and post-refresh)
# ---------------------------------------------------------------------------


def _checkpoint_cfg(precondition_frequency: int) -> Any:
    base = OmegaConf.load(PAIR_TRAIN_CONFIG)
    return OmegaConf.create(
        {
            "model": OmegaConf.to_container(base.model, resolve=True),
            "sampler": OmegaConf.to_container(base.sampler, resolve=True),
            "hamiltonian_terms": OmegaConf.to_container(base.hamiltonian_terms, resolve=True),
            "optimizer": {
                "_target_": "tpen.training.SOAP",
                "_partial_": True,
                "lr": 1e-3,
                "betas": [0.9, 0.99],
                "shampoo_beta": 0.95,
                "precondition_frequency": precondition_frequency,
                "max_precond_dim": 128,
            },
            "trainer": {"max_steps": 5, "log_every_n_steps": 1},
        }
    )


def _assert_soap_state_equal(
    model_a: nn.Module,
    opt_a: SOAP,
    model_b: nn.Module,
    opt_b: SOAP,
) -> None:
    params_a = list(model_a.parameters())
    params_b = list(model_b.parameters())
    assert len(params_a) == len(params_b)
    for pa, pb in zip(params_a, params_b, strict=True):
        assert torch.allclose(pa.detach(), pb.detach(), atol=1e-14, rtol=1e-13)
        in_a = pa in opt_a.state and len(opt_a.state[pa]) > 0
        in_b = pb in opt_b.state and len(opt_b.state[pb]) > 0
        assert in_a == in_b
        if not in_a:
            continue
        sa, sb = opt_a.state[pa], opt_b.state[pb]
        assert torch.equal(sa["step"], sb["step"])
        assert sa["mode_eigh_done"] == sb["mode_eigh_done"]
        assert torch.allclose(sa["exp_avg"], sb["exp_avg"], atol=1e-14, rtol=1e-13)
        assert torch.allclose(sa["exp_avg_sq"], sb["exp_avg_sq"], atol=1e-14, rtol=1e-13)
        for ga, gb in zip(sa["GG"], sb["GG"], strict=True):
            if ga is None or gb is None:
                assert ga is None and gb is None
            else:
                assert torch.allclose(ga, gb, atol=1e-14, rtol=1e-13)
        for qa, qb in zip(sa["Q"], sb["Q"], strict=True):
            if qa is None or qb is None:
                assert qa is None and qb is None
            else:
                assert torch.allclose(qa, qb, atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize("checkpoint_step", [2, 3])
def test_real_checkpoint_resume_parity_pre_and_post_refresh(
    tmp_path: Path, checkpoint_step: int
) -> None:
    """Resume immediately before (step 2) and after (step 3) QR refresh (freq=3)."""

    precond_freq = 3
    total_steps = 5
    cfg = _checkpoint_cfg(precond_freq)

    # Uninterrupted reference run (5 steps)
    torch.manual_seed(101)
    ref_model = build_tiny_spenn()
    ref_sampler = build_tiny_sampler()
    ref_terms = list(build_tiny_hamiltonian_terms().values())
    ref_opt = make_optimizer(cfg.optimizer, ref_model.parameters())
    ref_ctx = _ResumeContext(cfg, tmp_path / "ref_run")
    ref_trainer = VMCTrainer(max_steps=total_steps, log_every_n_steps=1)
    ref_trainer.fit(
        model=ref_model,
        sampler=ref_sampler,
        hamiltonian_terms=ref_terms,
        optimizer=ref_opt,
        context=ref_ctx,
        emit=lambda *_, **__: None,
    )

    # Split run: run to checkpoint_step, save real checkpoint, restore into fresh
    # model/optimizer/trainer/sampler, and finish remaining steps.
    torch.manual_seed(101)
    split_model = build_tiny_spenn()
    split_sampler = build_tiny_sampler()
    split_terms = list(build_tiny_hamiltonian_terms().values())
    split_opt = make_optimizer(cfg.optimizer, split_model.parameters())
    split_ctx = _ResumeContext(cfg, tmp_path / f"split_run_{checkpoint_step}")
    split_trainer = VMCTrainer(max_steps=checkpoint_step, log_every_n_steps=1)
    split_trainer.fit(
        model=split_model,
        sampler=split_sampler,
        hamiltonian_terms=split_terms,
        optimizer=split_opt,
        context=split_ctx,
        emit=lambda *_, **__: None,
    )

    ckpt_root = tmp_path / f"ckpts_{checkpoint_step}"
    ckpt_dir = save_checkpoint(
        output_dir=ckpt_root,
        next_iteration=split_trainer.next_iteration,
        completed_updates=split_trainer.completed_updates,
        model=split_model,
        context=split_ctx,
        optimizer=split_opt,
        trainer=split_trainer,
        sampler=split_sampler,
    )

    # Perturb global RNG before restore to prove RNG + optimizer + sampler state
    # actually came from the checkpoint rather than accidental process state.
    torch.manual_seed(9999)
    resumed_model = build_tiny_spenn()
    resumed_sampler = build_tiny_sampler()
    resumed_terms = list(build_tiny_hamiltonian_terms().values())
    resumed_opt = make_optimizer(cfg.optimizer, resumed_model.parameters())
    resumed_ctx = _ResumeContext(cfg, tmp_path / f"resumed_run_{checkpoint_step}")
    resumed_trainer = VMCTrainer(max_steps=total_steps, log_every_n_steps=1)
    resumed_trainer.resolve_update_state(model=resumed_model, optimizer=resumed_opt)

    report = restore_checkpoint(
        load={"mode": "train_resume", "path": str(ckpt_dir)},
        model=resumed_model,
        context=resumed_ctx,
        optimizer=resumed_opt,
        trainer=resumed_trainer,
        sampler=resumed_sampler,
    )
    assert report.loaded_optimizer is True
    assert report.next_iteration == checkpoint_step
    assert report.completed_updates == checkpoint_step

    resumed_trainer.fit(
        model=resumed_model,
        sampler=resumed_sampler,
        hamiltonian_terms=resumed_terms,
        optimizer=resumed_opt,
        context=resumed_ctx,
        emit=lambda *_, **__: None,
    )

    assert resumed_trainer.next_iteration == total_steps
    assert resumed_trainer.completed_updates == total_steps
    _assert_soap_state_equal(ref_model, ref_opt, resumed_model, resumed_opt)


def test_load_state_dict_validation_atomicity_and_deepcopy_isolation() -> None:
    p = nn.Parameter(torch.ones(3, 3, dtype=torch.float64))
    opt = SOAP([p], lr=1e-2, precondition_frequency=2)
    p.grad = torch.tensor([[1.0, 0.2, -0.3], [0.4, 0.8, 0.1], [-0.2, 0.5, 0.7]], dtype=torch.float64)
    opt.step()

    saved = opt.state_dict()
    # Mutating the returned state_dict must not affect live optimizer state
    saved["state"][0]["Q"][0].zero_()
    assert torch.allclose(
        opt.state[p]["Q"][0].T @ opt.state[p]["Q"][0],
        torch.eye(3, dtype=torch.float64),
        atol=1e-12,
    )

    # Malformed state_dict (bad Q shape) must raise and preserve prior state atomically
    snapshot_q0 = opt.state[p]["Q"][0].clone()
    bad = opt.state_dict()
    bad["state"][0]["Q"][0] = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="shape"):
        opt.load_state_dict(bad)
    assert torch.equal(opt.state[p]["Q"][0], snapshot_q0)

    # Missing required key must raise and preserve prior state
    missing_key = copy.deepcopy(opt.state_dict())
    del missing_key["state"][0]["mode_eigh_done"]
    with pytest.raises(ValueError, match="missing required key"):
        opt.load_state_dict(missing_key)
    assert torch.equal(opt.state[p]["Q"][0], snapshot_q0)


def test_load_state_dict_preserves_float32_factors_for_bfloat16_and_float16() -> None:
    """GG/Q float32 factors survive load_state_dict bitwise on bfloat16 and float16."""

    gen = torch.Generator().manual_seed(17)
    p_bf16 = nn.Parameter(torch.randn(4, 4, generator=gen, dtype=torch.bfloat16))
    opt = SOAP([p_bf16], lr=1e-2, precondition_frequency=2)
    g1_bf16 = torch.randn(4, 4, generator=gen, dtype=torch.bfloat16)
    p_bf16.grad = g1_bf16.clone()
    opt.step()
    sd = opt.state_dict()

    p_bf16_res = nn.Parameter(p_bf16.detach().clone())
    opt_res = SOAP([p_bf16_res], lr=1e-2, precondition_frequency=2)
    opt_res.load_state_dict(sd)

    eye4_f32 = torch.eye(4, dtype=torch.float32)
    for axis in range(2):
        q_orig = opt.state[p_bf16]["Q"][axis]
        q_res = opt_res.state[p_bf16_res]["Q"][axis]
        gg_orig = opt.state[p_bf16]["GG"][axis]
        gg_res = opt_res.state[p_bf16_res]["GG"][axis]
        assert q_res is not None and gg_res is not None
        assert q_res.dtype == torch.float32 and gg_res.dtype == torch.float32
        assert torch.equal(q_res, q_orig)
        assert torch.equal(gg_res, gg_orig)
        assert float((q_res.T @ q_res - eye4_f32).abs().max().item()) < 1e-6

    # Verify bitwise resume parity on step 2 (in rotated Q) and caller state_dict isolation.
    sd_q0_before = sd["state"][0]["Q"][0].clone()
    g2_bf16 = torch.randn(4, 4, generator=gen, dtype=torch.bfloat16)
    p_bf16.grad = g2_bf16.clone()
    p_bf16_res.grad = g2_bf16.clone()
    opt.step()
    opt_res.step()
    assert torch.equal(p_bf16.detach(), p_bf16_res.detach())
    assert torch.equal(opt.state[p_bf16]["Q"][0], opt_res.state[p_bf16_res]["Q"][0])
    assert torch.equal(sd["state"][0]["Q"][0], sd_q0_before)

    # float16 overflow check: finite float32 GG (160000 > 65504) must not overflow to inf.
    p_f16 = nn.Parameter(torch.randn(4, 4, generator=gen, dtype=torch.float16))
    opt16 = SOAP([p_f16], lr=1e-2, shampoo_beta=0.0, precondition_frequency=2)
    p_f16.grad = torch.full((4, 4), 200.0, dtype=torch.float16)
    opt16.step()
    assert float(opt16.state[p_f16]["GG"][0].max().item()) > 65504.0

    p_f16_res = nn.Parameter(p_f16.detach().clone())
    opt16_res = SOAP([p_f16_res], lr=1e-2, shampoo_beta=0.0, precondition_frequency=2)
    opt16_res.load_state_dict(opt16.state_dict())
    for axis in range(2):
        gg16 = opt16_res.state[p_f16_res]["GG"][axis]
        q16 = opt16_res.state[p_f16_res]["Q"][axis]
        assert gg16 is not None and q16 is not None
        assert gg16.dtype == torch.float32 and q16.dtype == torch.float32
        assert torch.isfinite(gg16).all()
        assert torch.equal(gg16, opt16.state[p_f16]["GG"][axis])
        assert torch.equal(q16, opt16.state[p_f16]["Q"][axis])


def test_load_state_dict_hardening_refuses_malformed_inputs_atomically() -> None:
    """Reject malformed state_dict shapes, layouts, dtypes, and stray/empty entries."""

    # Shape (3, 1): axis 0 is preconditioned, axis 1 is unpreconditioned (None).
    p = nn.Parameter(torch.ones(3, 1, dtype=torch.float64))
    opt = SOAP([p], lr=1e-2, precondition_frequency=2)
    p.grad = torch.tensor([[1.0], [-0.5], [0.8]], dtype=torch.float64)
    opt.step()

    snapshot = opt.state_dict()

    def _assert_prior_state_untouched() -> None:
        current = opt.state_dict()
        assert current["param_groups"] == snapshot["param_groups"]
        assert set(current["state"].keys()) == set(snapshot["state"].keys())
        s0, s1 = snapshot["state"][0], current["state"][0]
        assert torch.equal(s0["step"], s1["step"])
        assert torch.equal(s0["exp_avg"], s1["exp_avg"])
        assert torch.equal(s0["exp_avg_sq"], s1["exp_avg_sq"])
        assert torch.equal(s0["GG"][0], s1["GG"][0])
        assert torch.equal(s0["Q"][0], s1["Q"][0])
        assert s1["GG"][1] is None and s1["Q"][1] is None
        assert s1["mode_eigh_done"] == s0["mode_eigh_done"]

    # Non-mapping state_dict raises TypeError (checked without typeguard interception).
    with suppress_type_checks(), pytest.raises(TypeError):
        opt.load_state_dict("not-a-mapping")  # type: ignore[arg-type]
    _assert_prior_state_untouched()

    # Missing top-level keys or invalid container types.
    for malformed in (
        {"state": {}},
        {"param_groups": snapshot["param_groups"]},
        {"state": [], "param_groups": snapshot["param_groups"]},
        {"state": {}, "param_groups": "invalid"},
    ):
        with pytest.raises(ValueError):
            opt.load_state_dict(malformed)
        _assert_prior_state_untouched()

    # Empty {} per-parameter state entry.
    empty_entry = copy.deepcopy(snapshot)
    empty_entry["state"][0] = {}
    with pytest.raises(ValueError, match="must not be empty"):
        opt.load_state_dict(empty_entry)
    _assert_prior_state_untouched()

    # Stray unmapped key in state_dict['state'].
    stray_key = copy.deepcopy(snapshot)
    stray_key["state"][999] = copy.deepcopy(snapshot["state"][0])
    with pytest.raises(ValueError, match="unmapped"):
        opt.load_state_dict(stray_key)
    _assert_prior_state_untouched()

    # Non-integer / bool / non-0D / complex / sparse step tensors.
    sparse_step = torch.sparse_coo_tensor(
        torch.tensor([[0]], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        size=(1,),
    )
    for bad_step in (
        torch.tensor(2.7, dtype=torch.float64),
        torch.tensor(True, dtype=torch.bool),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor(1 + 0j, dtype=torch.complex128),
        sparse_step,
        True,
        -1,
    ):
        bad = copy.deepcopy(snapshot)
        bad["state"][0]["step"] = bad_step
        with pytest.raises(ValueError):
            opt.load_state_dict(bad)
        _assert_prior_state_untouched()

    # Complex / non-floating / sparse COO / sparse CSR moment and factor tensors.
    bad_complex = copy.deepcopy(snapshot)
    bad_complex["state"][0]["exp_avg"] = torch.zeros(3, 1, dtype=torch.complex128)
    with pytest.raises(ValueError):
        opt.load_state_dict(bad_complex)
    _assert_prior_state_untouched()

    bad_int_moment = copy.deepcopy(snapshot)
    bad_int_moment["state"][0]["exp_avg_sq"] = torch.zeros(3, 1, dtype=torch.int64)
    with pytest.raises(ValueError):
        opt.load_state_dict(bad_int_moment)
    _assert_prior_state_untouched()

    bad_sparse_gg = copy.deepcopy(snapshot)
    bad_sparse_gg["state"][0]["GG"][0] = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([1.0], dtype=torch.float64), size=(3, 3)
    )
    with pytest.raises(ValueError):
        opt.load_state_dict(bad_sparse_gg)
    _assert_prior_state_untouched()

    bad_sparse_q = copy.deepcopy(snapshot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bad_sparse_q["state"][0]["Q"][0] = torch.eye(3, dtype=torch.float64).to_sparse_csr()
    with pytest.raises(ValueError):
        opt.load_state_dict(bad_sparse_q)
    _assert_prior_state_untouched()

    # mode_eigh_done[axis]=True on non-preconditioned (None) axis 1.
    bad_eigh_done = copy.deepcopy(snapshot)
    bad_eigh_done["state"][0]["mode_eigh_done"][1] = True
    with pytest.raises(ValueError, match="mode_eigh_done"):
        opt.load_state_dict(bad_eigh_done)
    _assert_prior_state_untouched()

    # Invalid param_group hyperparameter rolls back cleanly without mutating live groups.
    bad_hparam = copy.deepcopy(snapshot)
    bad_hparam["param_groups"][0]["lr"] = -1.0
    with pytest.raises(ValueError, match="lr"):
        opt.load_state_dict(bad_hparam)
    _assert_prior_state_untouched()


# ---------------------------------------------------------------------------
# 9. OptimizerState callback compatibility with SOAP & refusal controls
# ---------------------------------------------------------------------------


def _bound_trainer_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> TrainerState:
    trainer = VMCTrainer(max_steps=1)
    method = trainer.bind_update_method(model=model, optimizer=optimizer)
    assert isinstance(method, LegacyAutogradUpdate)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))
    return TrainerState(
        model=model,
        optimizer=optimizer,
        update_state=VMCUpdateState(optimizer=optimizer, model_parameters=binding),
        trainer=trainer,
    )


def test_optimizer_state_callback_supports_soap_with_csv_and_jsonl_sinks(tmp_path: Path) -> None:
    model = GroupedModel().double()
    optimizer = SOAP(model.parameters(), lr=0.01, precondition_frequency=2)
    csv_path = tmp_path / "soap_opt_state.csv"
    jsonl_path = tmp_path / "soap_opt_state.jsonl"
    context = LoggerContext(CSV(csv_path), JSONL(jsonl_path))
    callback = OptimizerState({"embed": ["embedding"]})
    state = _bound_trainer_state(model, optimizer)

    # Before first step: missing state counted honestly
    deliver_completed_iteration(callback, context, state, step=0)
    before = context.latest("train/optimizer_state/embed")
    assert before["n_no_optimizer_state"] == 1
    assert before["step"] == 0

    # After 2 SOAP steps (crossing refresh boundary): metrics match live SOAP moments
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(param.square().sum() for param in model.parameters())
        loss.backward()
        optimizer.step()
        deliver_completed_iteration(callback, context, state, step=step)

    metrics = context.latest("train/optimizer_state/embed")
    soap_st = optimizer.state[model.embedding.weight]
    assert metrics["n_no_optimizer_state"] == 0
    assert metrics["step"] == 2
    assert metrics["step_mismatch"] is False
    assert metrics["exp_avg_l2"] == pytest.approx(float(soap_st["exp_avg"].norm().item()))
    assert metrics["exp_avg_sq_rms"] == pytest.approx(
        float(soap_st["exp_avg_sq"].square().mean().sqrt().item())
    )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(None not in row for row in rows)
    with jsonl_path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert any(r["namespace"] == "train/optimizer_state/embed" and r["step"] == 2 for r in records)


def test_optimizer_state_callback_still_refuses_sgd_lbfgs_and_non_legacy_method() -> None:
    model = GroupedModel().double()
    for unsupported_opt in (
        torch.optim.SGD(model.parameters(), lr=0.01),
        torch.optim.LBFGS(model.parameters(), lr=0.01),
    ):
        with pytest.raises(ValueError, match="Adam, AdamW, or SOAP"):
            deliver_completed_iteration(
                OptimizerState({"embed": ["embedding"]}),
                RecordingContext(),
                _bound_trainer_state(model, unsupported_opt),
                step=0,
            )

    class _CustomMethod(VMCUpdateMethod[AutogradUpdateInput]):
        def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
            del update_input
            return VMCUpdateResult(applied=False, grad_norm=0.0)

    soap_opt = SOAP(model.parameters(), lr=0.01)
    trainer = type("Trainer", (), {"_resolved_update_method": _CustomMethod()})()
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))
    bad_state = TrainerState(
        model=model,
        optimizer=soap_opt,
        update_state=VMCUpdateState(optimizer=soap_opt, model_parameters=binding),
        trainer=trainer,
    )
    with pytest.raises(ValueError, match="LegacyAutogradUpdate"):
        deliver_completed_iteration(
            OptimizerState({"embed": ["embedding"]}),
            RecordingContext(),
            bad_state,
            step=0,
        )


def test_optimizer_state_callback_soap_in_real_vmc_trainer_fit() -> None:
    """Real VMCTrainer.fit() exercises Started(TrainingIteration) and disconnected counts."""

    class _DispatchingContext(_StubContext):
        def __init__(self) -> None:
            super().__init__()
            self.logged: list[tuple[int | None, str, dict[str, Any]]] = []

        def log(self, metrics, *, step=None, namespace="run") -> None:
            super().log(metrics, step=step, namespace=namespace)
            self.logged.append((step, namespace, dict(metrics)))

        def _dispatch_occurrence(self, occurrence, *, state=None) -> None:
            super()._dispatch_occurrence(occurrence, state=state)
            for callback in self.callbacks:
                callback.handle_occurrence(occurrence, self, state)

    model = build_tiny_spenn()
    named = pair_parameters_by_name(model)
    mixing_names = [name for name in named if name.startswith("stack.layers.0.mixing.")]
    expected_mixing_inactive = sum(1 for name in mixing_names if name in INACTIVE_PAIR_PARAMETERS)
    expected_ungrouped_inactive = len(INACTIVE_PAIR_PARAMETERS) - expected_mixing_inactive
    assert expected_mixing_inactive == 15
    assert expected_ungrouped_inactive == 1

    sampler = build_tiny_sampler()
    terms = list(build_tiny_hamiltonian_terms().values())
    optimizer = SOAP(model.parameters(), lr=1e-3, precondition_frequency=2)
    context = _DispatchingContext()
    context.callbacks = [OptimizerState({"mixing": ["stack.layers.0.mixing"]}, every_n_steps=1)]

    trainer = VMCTrainer(max_steps=2, log_every_n_steps=1)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=optimizer,
        context=context,
        emit=lambda *_, **__: None,
    )

    assert trainer.completed_updates == 2
    assert any(
        isinstance(occ.event, Started) and isinstance(occ.event.operation, TrainingIteration)
        for occ in context.occurrences
    )

    mixing_step1 = [
        metrics
        for step, ns, metrics in context.logged
        if ns == "train/optimizer_state/mixing" and step == 1
    ]
    ungrouped_step1 = [
        metrics
        for step, ns, metrics in context.logged
        if ns == "train/optimizer_state/ungrouped" and step == 1
    ]
    assert len(mixing_step1) == 1
    assert len(ungrouped_step1) == 1

    mix = mixing_step1[0]
    assert mix["n_parameters"] == len(mixing_names)
    assert mix["n_no_optimizer_state"] == expected_mixing_inactive
    assert mix["step"] == 2
    assert mix["step_mismatch"] is False
    assert mix["exp_avg_nonfinite_count"] == 0
    assert mix["exp_avg_sq_nonfinite_count"] == 0
    assert mix["exp_avg_l2"] > 0.0
    assert mix["exp_avg_sq_rms"] > 0.0

    ung = ungrouped_step1[0]
    assert ung["n_no_optimizer_state"] == expected_ungrouped_inactive
    assert ung["step"] == 2
    assert ung["step_mismatch"] is False

    # Control: unsupported optimizer is refused at Started(TrainingIteration) before sampling.
    bad_model = build_tiny_spenn()
    bad_opt = torch.optim.SGD(bad_model.parameters(), lr=1e-3)
    bad_sampler = build_tiny_sampler()
    sampler_calls = 0
    orig_collect = bad_sampler.collect_samples

    def _counting_collect(*args: Any, **kwargs: Any) -> Any:
        nonlocal sampler_calls
        sampler_calls += 1
        return orig_collect(*args, **kwargs)

    bad_sampler.collect_samples = _counting_collect  # type: ignore[method-assign]
    bad_ctx = _DispatchingContext()
    bad_ctx.callbacks = [OptimizerState({"mixing": ["stack.layers.0.mixing"]}, every_n_steps=1)]
    bad_trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    with pytest.raises(ValueError, match="Adam, AdamW, or SOAP"):
        bad_trainer.fit(
            model=bad_model,
            sampler=bad_sampler,
            hamiltonian_terms=list(build_tiny_hamiltonian_terms().values()),
            optimizer=bad_opt,
            context=bad_ctx,
            emit=lambda *_, **__: None,
        )
    assert sampler_calls == 0
