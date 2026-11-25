"""Comprehensive unit, oracle, memory-admission, and TPEN smoke tests for dense LM.

Covers all five verification items in the approved P4 design gate (`5d3ab78c`,
Revision 2, note `54ab0807` §7.4) plus PR #854 NB1/NB2 carry-forwards:

1. Analytic Oracles A--F (`54ab0807` §5.1--§5.6): exact float64 matrices,
   non-symmetry invariants (`h_r != h_c`, `dH11 != dH11.T`), all 6 Table 5.1
   `(c_diag, c_ovlp, xi)` configurations, steepest-descent limit, overlap and
   squared/linear Euclidean trust caps, `step_scale != 1.0` pre/post telemetry
   norms, constant-shift `+1000.0` invariance, end-to-end autograd tangent
   verification, Oracle B rank-2 gauge+disconnected projection, Oracle C
   4-point zero-variance theorem + symmetrized-H / drop-gE / flip-gE mutants,
   Oracle D exact ground-state stationary point, Oracle E refusal oracles
   (complex pair, non-descent, `min_c0`, constant non-binary `O=0.3` + tiny
   non-constant controls, `eigensolve_failed` / `nonfinite_eigensolve` /
   `nonfinite_update_direction`, nullable `None` skip diagnostics under
   Typeguard), and Oracle F1/F2 `N < P, c_ovlp > 0` nullspace regressions.
2. Real tiny `TPENWaveFunction` smoke test (PR #854 NB2): `VMCTrainer` +
   `MaterializedHamiltonianTangentStepRequest` + `LinearMethodUpdate` on tiny
   2-electron TPEN (`C=2, H=4, P=366`), multi-chunk parity, `return_terms=True`,
   bitwise-zero movement of the 16 structurally disconnected order-1-output
   parameters, non-vacuous movement of live order-2 parameters, `norm_semantics`
   / `describe` / `last_telemetry` CSV+JSONL sink safety, `TrainResume`
   checkpoint save/restore split-run parity, and `lm-state-1` drift refusals.
3. PR #854 NB1 graph-release verification: weakref check confirming chunk-k
   `term_value` graphs are released before chunk k+1 when `return_terms=True`,
   plus staged `S_param`/`dH11` weakref release before `torch.linalg.eig(M)`.
4. Non-finite row policies (`fail`, `mask`, `fail_on_pinned`), degenerate batch
   guards, `declared_nonfinite_local_energy_policy()`, complex-phase refusal,
   graph-bearing refusal, binding drift refusal, and plain-SGD validation.
5. Dense memory admission formulas (Table 6.2 & mixed-dtype promotion),
   preflight & per-step `LinearMethodMemoryLimitError` refusal BEFORE
   `flatten_*` allocation (including synthetic HI `O-raw` `P=961,284`), and
   all 16 HI `optimizer_grid_authority.json` `linear_method` cells.
"""

from __future__ import annotations

import copy
import csv
import gc
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import weakref

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch
import typeguard
from typeguard import TypeCheckError

import tpen.training as training_pkg
import tpen.training.linear_method as lm_module
from tests.helpers.hooke_models import (
    INACTIVE_PAIR_PARAMETERS,
    PAIR_TRAIN_CONFIG,
    pair_parameters_by_name,
)
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext
from tpen.checkpoint import restore_checkpoint, save_checkpoint
from tpen.data.batch import (
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    ParameterSlot,
    WavefunctionOutput,
)
from tpen.logging import CSV, JSONL, LogRecord
from tpen.physics.hamiltonian import LocalEnergyResult, local_energy
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import HarmonicTrap
from tpen.training.linear_method import (
    DEFAULT_MAX_DENSE_BYTES,
    DENSE_LM_WORKING_SET_FACTOR,
    LM_GRID_FACTOR_KEYS,
    LM_STATE_VERSION,
    NONFINITE_TANGENT_ROW_POLICY,
    DenseLinearMethodMatrices,
    LMGridHyperparameters,
    LinearMethodMemoryLimitError,
    LinearMethodPolicy,
    LinearMethodSolveDiagnostics,
    LinearMethodSolveOutcome,
    LinearMethodTelemetry,
    LinearMethodUpdate,
    TangentRowSelection,
    assemble_dense_linear_method_matrices,
    build_linear_method_update_from_factors,
    dense_matrix_working_bytes,
    dense_step_working_bytes,
    flatten_local_energy_derivative_blocks,
    lm_policy_from_factors,
    select_tangent_update_rows,
    solve_dense_linear_method,
    validate_dense_memory_budget,
)
from tpen.training.score_geometry import ScoreConventions
from tpen.training.statistics import StatisticsReducer
from tpen.training.step_requests import (
    HamiltonianTangentUpdateInput,
    MaterializedHamiltonianTangentStepRequest,
    MaterializedLocalEnergyDerivatives,
    evaluate_hamiltonian_tangent_step,
    resolve_step_request,
)
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    ModelParameterBinding,
    SETTING_NAME_COLLISIONS_KEY,
    VMCUpdateMethod,
)

# Fixed 12 deterministic 3D coordinates for Oracle System A & D (54ab0807 §5.1).
ORACLE_A_COORDS: tuple[tuple[float, float, float], ...] = (
    (1.65, 0.95, 0.85),
    (-0.40, -1.05, 0.70),
    (0.75, 0.25, -1.25),
    (2.05, -0.50, -0.55),
    (-0.75, 1.25, 0.30),
    (1.05, -1.20, 1.10),
    (0.15, 0.80, -0.95),
    (1.40, -0.30, 1.30),
    (-0.15, -0.70, -1.10),
    (0.55, 1.40, 0.35),
    (1.80, 0.10, -0.40),
    (-0.25, -0.90, 0.45),
)

_SELECTED_12_OPTIONAL_FIELDS: tuple[str, ...] = (
    "selected_delta_eigenvalue",
    "selected_eigenvalue",
    "selected_eigenvalue_imag",
    "selected_eigenvector_imag_norm",
    "selected_c0",
    "rescaling_denominator",
    "raw_overlap_step_norm",
    "rescaled_overlap_step_norm",
    "reg_overlap_step_norm",
    "applied_overlap_step_norm",
    "applied_reg_overlap_step_norm",
    "directional_derivative",
)

_ALL_13_OPTIONAL_FIELDS: tuple[str, ...] = (
    "overlap_min_retained_eigenvalue",
    *_SELECTED_12_OPTIONAL_FIELDS,
)


def _layout(n_parameters: int, *, dtype: torch.dtype = torch.float64) -> ParameterLayout:
    slot = ParameterSlot(
        ordinal=0,
        shape=(n_parameters,),
        numel=n_parameters,
        dtype=dtype,
    )
    return ParameterLayout(slots=(slot,))


def _tangent_input(
    parameter: torch.nn.Parameter,
    scores: torch.Tensor,
    derivatives: torch.Tensor,
    energies: torch.Tensor,
    *,
    step: int = 0,
    n_electrons: int = 1,
    positions: torch.Tensor | None = None,
    phase: torch.Tensor | None = None,
    logabs: torch.Tensor | None = None,
) -> HamiltonianTangentUpdateInput:
    """Build a detached HamiltonianTangentUpdateInput for a single flat parameter."""

    n_samples = int(energies.shape[0])
    dtype = energies.dtype
    if positions is None:
        positions = torch.zeros((n_samples, n_electrons, 3), dtype=dtype)
    spins = torch.ones((n_samples, n_electrons), dtype=dtype)
    batch = ElectronBatch(positions=positions, spins=spins)
    resolved_logabs = (
        torch.zeros((n_samples,), dtype=dtype) if logabs is None else logabs
    )
    output = WavefunctionOutput(
        logabs=resolved_logabs,
        sign=torch.ones((n_samples,), dtype=dtype),
        phase=phase,
    )
    layout = _layout(int(parameter.numel()), dtype=parameter.dtype)
    return HamiltonianTangentUpdateInput(
        batch=batch,
        wavefunction=output,
        local_energy=energies,
        step=step,
        parameter_scores=MaterializedParameterLogScores(layout=layout, blocks=(scores,)),
        local_energy_derivatives=MaterializedLocalEnergyDerivatives(
            layout=layout,
            blocks=(derivatives,),
        ),
        parameter_binding=ParameterBinding(layout=layout, parameters=(parameter,)),
    )


def _lm_method(
    parameter: torch.nn.Parameter,
    *,
    policy: LinearMethodPolicy | None = None,
    conventions: ScoreConventions | None = None,
    nonfinite_local_energy_policy: str = "mask",
) -> LinearMethodUpdate:
    resolved_policy = LinearMethodPolicy() if policy is None else policy
    optimizer = torch.optim.SGD([parameter], lr=resolved_policy.step_scale)
    return LinearMethodUpdate(
        optimizer,
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
        policy=resolved_policy,
        conventions=conventions,
        nonfinite_local_energy_policy=nonfinite_local_energy_policy,
    )


def _oracle_a_arrays(
    *,
    alpha: float = 0.35,
    beta: float = 0.40,
    omega: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(positions [12, 1, 3], E_L [12], O [12, 2], gE [12, 2])`` for Oracle A."""

    coords = torch.tensor(ORACLE_A_COORDS, dtype=torch.float64)
    r2 = (coords * coords).sum(dim=1)
    x = coords[:, 0]
    o_mat = torch.stack((-r2, x), dim=1)
    e_loc = (
        3.0 * alpha
        - 0.5 * (beta * beta)
        + 2.0 * alpha * beta * x
        + (0.5 * omega * omega - 2.0 * alpha * alpha) * r2
    )
    ge_mat = torch.stack(
        (3.0 + 2.0 * beta * x - 4.0 * alpha * r2, -beta + 2.0 * alpha * x),
        dim=1,
    )
    positions = coords.unsqueeze(1)
    return positions, e_loc, o_mat, ge_mat


class _OracleAHarmonicWavefunction(torch.nn.Module):
    """Exact 3D 1-electron HO ansatz ``log|psi(r)| = -alpha ||r||^2 + beta x``."""

    def __init__(self, alpha: float = 0.35, beta: float = 0.40) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([alpha, beta], dtype=torch.float64))
        slot = ParameterSlot(ordinal=0, shape=(2,), numel=2, dtype=torch.float64)
        self._binding = ParameterBinding(
            layout=ParameterLayout(slots=(slot,)),
            parameters=(self.theta,),
        )

    @property
    def parameter_binding(self) -> ParameterBinding:
        return self._binding

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        r = flat.positions[:, 0, :]
        r2 = (r * r).sum(dim=-1)
        x = r[:, 0]
        logabs = -self.theta[0] * r2 + self.theta[1] * x
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _FixedWalkers:
    def __init__(self, batch: ElectronBatch) -> None:
        self._batch = batch

    def make_batch(self) -> ElectronBatch:
        return self._batch


class _FixedSampler:
    def __init__(self, batch: ElectronBatch) -> None:
        self._batch = batch

    def collect_samples(self, model: Any, *, device: Any = None):
        del model, device
        return _FixedWalkers(self._batch), None


class _ResumeContext(_StubContext):
    """Minimal ``RunContext`` supporting both ``VMCTrainer`` and real checkpoints."""

    def __init__(self, cfg: Any, run_dir: Path) -> None:
        super().__init__()
        self.cfg = cfg
        self._run_dir = Path(run_dir)
        self.metadata = SimpleNamespace(
            device="cpu",
            dtype="float64",
            run_id="lm-resume-test",
            git_commit="0" * 40,
            git_branch="agent/opt-889cd4d1-linear-method",
            dirty_worktree=False,
            command="pytest",
            extra={},
        )

    @property
    def run_dir(self) -> Path:
        return self._run_dir


def _assert_csv_and_jsonl_roundtrip(
    tmp_path: Path,
    *,
    name: str,
    namespace: str,
    metrics: dict[str, Any],
) -> None:
    """Verify flat metric mapping writes cleanly to strict CSV and JSONL sinks."""

    csv_path = tmp_path / f"{name}.csv"
    jsonl_path = tmp_path / f"{name}.jsonl"
    record = LogRecord(step=0, namespace=namespace, metrics=metrics)
    CSV(csv_path).log(record)
    JSONL(jsonl_path).log(record)

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(None not in row for row in rows)
    with jsonl_path.open(encoding="utf-8") as stream:
        parsed = json.loads(stream.readline())
    assert parsed["namespace"] == namespace


def _tiny_tpen_config() -> Any:
    """Build tiny ``C=2, H=4`` (``P=366``) 2-electron Hooke config from ``pair_train.yaml``."""

    base = OmegaConf.load(PAIR_TRAIN_CONFIG)
    cfg = OmegaConf.create(
        {
            "model": OmegaConf.to_container(base.model, resolve=True),
            "sampler": OmegaConf.to_container(base.sampler, resolve=True),
            "hamiltonian_terms": OmegaConf.to_container(base.hamiltonian_terms, resolve=True),
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "_partial_": True,
                "lr": 1.0,
            },
            "trainer": {"max_steps": 2, "log_every_n_steps": 1, "return_terms": True},
        }
    )
    cfg.model.embedding.out_channels = 2
    cfg.model.embedding.hidden_channels = 4
    cfg.model.layers[0].mixing.channels = 2
    cfg.model.layers[0].path_aggregation.channels = 2
    cfg.model.readout.channels = 2
    cfg.sampler.n_walkers = 12
    cfg.sampler.burn_in = 2
    cfg.sampler.n_steps = 2
    return cfg


def _instantiate_tiny_tpen(cfg: Any, *, seed: int = 19) -> tuple[Any, Any, dict[str, Any]]:
    torch.manual_seed(seed)
    model = instantiate(cfg.model).to(device=torch.device("cpu"), dtype=torch.float64)
    sampler = instantiate(cfg.sampler)
    terms = dict(instantiate(cfg.hamiltonian_terms))
    return model, sampler, terms


# ---------------------------------------------------------------------------
# 1. Public exports, interface contracts, policy & SGD guards
# ---------------------------------------------------------------------------


def test_public_exports_and_method_interface_contracts(tmp_path: Path) -> None:
    """All public LM symbols re-export cleanly and satisfy VMCUpdateMethod contracts."""

    for symbol in lm_module.__all__:
        assert hasattr(training_pkg, symbol), f"tpen.training must export {symbol}"
        assert symbol in training_pkg.__all__, f"tpen.training.__all__ must list {symbol}"

    param = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    policy = LinearMethodPolicy(
        derivative_block_shift=1.0e-2,
        overlap_metric_trust_radius=0.1,
        tangent_chunk_size=4,
    )
    method_fail = _lm_method(param, policy=policy, nonfinite_local_energy_policy="fail")
    method_mask = _lm_method(param, policy=policy, nonfinite_local_energy_policy="mask")

    assert method_fail.declared_nonfinite_local_energy_policy() == "fail"
    assert method_mask.declared_nonfinite_local_energy_policy() == "mask"
    assert method_fail.nonfinite_tangent_row_policy == NONFINITE_TANGENT_ROW_POLICY == "fail_on_pinned"

    assert method_fail.forward_request() is None
    step_req = method_fail.step_request()
    assert step_req == MaterializedHamiltonianTangentStepRequest(chunk_size=4)
    assert resolve_step_request(method_fail) == step_req
    assert method_fail.last_telemetry is None
    assert method_fail.norm_semantics() == "energy_gradient_l2_preconditioned_direction_in_grad"

    desc = method_fail.describe()
    assert desc.method_class == "tpen.training.linear_method.LinearMethodUpdate"
    assert desc.carrier_class == "torch.optim.sgd.SGD"
    assert desc.forward_request == "tpen.training.step_requests.MaterializedHamiltonianTangentStepRequest"
    assert desc.norm_semantics == "energy_gradient_l2_preconditioned_direction_in_grad"
    assert desc.settings["nonfinite_local_energy_policy"] == "fail"
    assert desc.settings["nonfinite_tangent_row_policy"] == "fail_on_pinned"
    assert desc.settings["reducer_class"] == "IdentityStatisticsReducer"
    assert desc.settings["policy_derivative_block_shift"] == 1.0e-2
    assert desc.settings["policy_overlap_metric_trust_radius"] == 0.1
    assert desc.settings["conventions_solve_dtype"] == "torch.float64"

    desc_metrics = desc.as_metrics()
    assert f"update_method_{SETTING_NAME_COLLISIONS_KEY}" not in desc_metrics
    _assert_csv_and_jsonl_roundtrip(
        tmp_path,
        name="lm_desc",
        namespace="train/update_method",
        metrics=desc_metrics,
    )


def test_policy_validation_and_plain_sgd_and_constructor_guards() -> None:
    """LinearMethodPolicy, plain-SGD, conventions, and empty-binding guards fail loudly."""

    bad_policy_cases: list[tuple[dict[str, Any], type[Exception]]] = [
        ({"derivative_block_shift": -1.0e-6}, ValueError),
        ({"derivative_block_shift": float("nan")}, ValueError),
        ({"derivative_block_shift": float("inf")}, ValueError),
        ({"derivative_block_shift": True}, TypeError),
        ({"overlap_shift": -0.1}, ValueError),
        ({"overlap_shift": False}, TypeError),
        ({"rank_cutoff": -1.0e-12}, ValueError),
        ({"rank_cutoff": 1.0}, ValueError),
        ({"rank_cutoff": True}, TypeError),
        ({"rescaling_xi": -0.01}, ValueError),
        ({"rescaling_xi": 1.01}, ValueError),
        ({"rescaling_xi": False}, TypeError),
        ({"min_c0": 0.0}, ValueError),
        ({"min_c0": 1.0}, ValueError),
        ({"min_c0": True}, TypeError),
        ({"imag_eigenvalue_tol": -1.0e-9}, ValueError),
        ({"imag_eigenvalue_tol": True}, TypeError),
        ({"imag_eigenvector_tol": -1.0e-9}, ValueError),
        ({"imag_eigenvector_tol": False}, TypeError),
        ({"require_descent": 1}, TypeError),
        ({"step_scale": 0.0}, ValueError),
        ({"step_scale": -1.0}, ValueError),
        ({"step_scale": True}, TypeError),
        ({"overlap_metric_trust_radius": 0.0}, ValueError),
        ({"overlap_metric_trust_radius": True}, TypeError),
        ({"max_update_norm": 0.0}, ValueError),
        ({"max_update_norm": False}, TypeError),
        ({"squared_update_norm_bound": 0.0}, ValueError),
        ({"squared_update_norm_bound": True}, TypeError),
        ({"max_update_norm": 0.1, "squared_update_norm_bound": 0.01}, ValueError),
        ({"tangent_chunk_size": 0}, ValueError),
        ({"tangent_chunk_size": True}, TypeError),
        ({"max_dense_bytes": 0}, ValueError),
        ({"max_dense_bytes": False}, TypeError),
        ({"max_parameters": 0}, ValueError),
        ({"max_parameters": True}, TypeError),
    ]
    for kwargs, expected_exc in bad_policy_cases:
        with pytest.raises((expected_exc, TypeCheckError)):
            LinearMethodPolicy(**kwargs)
        with typeguard.suppress_type_checks(), pytest.raises(expected_exc):
            LinearMethodPolicy(**kwargs)

    # Exact boundary values that ARE valid.
    valid_boundary = LinearMethodPolicy(
        derivative_block_shift=0.0,
        overlap_shift=0.0,
        rank_cutoff=0.0,
        rescaling_xi=0.0,
        imag_eigenvalue_tol=0.0,
        imag_eigenvector_tol=0.0,
    )
    assert valid_boundary.derivative_block_shift == 0.0
    assert LinearMethodPolicy(rescaling_xi=1.0).rescaling_xi == 1.0

    # Plain-SGD carrier validation.
    param = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    binding = ModelParameterBinding(parameters=(param,))
    policy = LinearMethodPolicy(step_scale=1.0)
    with pytest.raises(TypeError, match="Adam"):
        LinearMethodUpdate(
            torch.optim.Adam([param], lr=1.0),
            model_parameters=binding,
            policy=policy,
        )
    with pytest.raises(ValueError, match="momentum"):
        LinearMethodUpdate(
            torch.optim.SGD([param], lr=1.0, momentum=0.9),
            model_parameters=binding,
            policy=policy,
        )
    with pytest.raises(ValueError, match="dampening"):
        LinearMethodUpdate(
            torch.optim.SGD([param], lr=1.0, momentum=0.0, dampening=0.1),
            model_parameters=binding,
            policy=policy,
        )
    with pytest.raises(ValueError, match="weight_decay"):
        LinearMethodUpdate(
            torch.optim.SGD([param], lr=1.0, weight_decay=1.0e-4),
            model_parameters=binding,
            policy=policy,
        )
    with pytest.raises(ValueError, match="maximize"):
        LinearMethodUpdate(
            torch.optim.SGD([param], lr=1.0, maximize=True),
            model_parameters=binding,
            policy=policy,
        )
    with pytest.raises(ValueError, match="disagrees with LinearMethodPolicy.step_scale"):
        LinearMethodUpdate(
            torch.optim.SGD([param], lr=0.5),
            model_parameters=binding,
            policy=policy,
        )
    with pytest.raises(ValueError, match="energy_gradient_scale > 0.0"):
        LinearMethodUpdate(
            torch.optim.SGD([param], lr=1.0),
            model_parameters=binding,
            policy=policy,
            conventions=ScoreConventions(energy_gradient_scale=-2.0, solve_dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="non-empty parameter layout"):
        LinearMethodUpdate(
            torch.optim.SGD([param], lr=1.0),
            model_parameters=ModelParameterBinding(parameters=()),
            policy=policy,
        )


# ---------------------------------------------------------------------------
# 2. Oracle System A: 3D 1e HO (P=2, N=12) — matrices, 6 configs, limits,
#    trust caps, step_scale telemetry, shift invariance, autograd parity
# ---------------------------------------------------------------------------


def test_oracle_a_matrices_six_parameter_cases_limits_trust_and_shift_invariance() -> None:
    """Oracle System A matches 54ab0807 §5.1 matrices and all 6 solve rows to 1e-11."""

    positions, e_loc, o_mat, ge_mat = _oracle_a_arrays()
    matrices = assemble_dense_linear_method_matrices(
        e_loc, o_mat, ge_mat, solve_dtype=torch.float64
    )

    assert matrices.count == 12
    assert matrices.dtype == "torch.float64"
    assert matrices.energy_mean == pytest.approx(1.852451041666666, abs=1.0e-12, rel=1.0e-12)

    expected_s = torch.tensor(
        [
            [1.332374435763889, -0.917167534722222],
            [-0.917167534722222, 0.815607638888889],
        ],
        dtype=torch.float64,
    )
    expected_hc = torch.tensor([-0.596562390842014, 0.462247860243056], dtype=torch.float64)
    expected_hr = torch.tensor([-0.912437390842013, 0.520164526909722], dtype=torch.float64)
    expected_dh11 = torch.tensor(
        [
            [1.361245033483706, -0.797423736640987],
            [-0.786954899835431, 0.635293481589988],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(matrices.S_param, expected_s, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(matrices.h_c, expected_hc, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(matrices.h_r, expected_hr, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(matrices.dH11, expected_dh11, atol=1.0e-12, rtol=1.0e-12)
    # Explicit non-symmetry assertions (h_r != h_c and dH11 != dH11.T).
    torch.testing.assert_close(
        matrices.h_r - matrices.h_c,
        torch.tensor([-0.315875, 0.057916666666667], dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert float((matrices.dH11[0, 1] - matrices.dH11[1, 0]).item()) == pytest.approx(
        -1.046883680555577e-2, abs=1.0e-12
    )
    s_eigs = torch.linalg.eigvalsh(matrices.S_param)
    torch.testing.assert_close(
        s_eigs,
        torch.tensor([0.121122595548328, 2.026859479104450], dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    # All 6 Table 5.1 (c_diag, c_ovlp, xi) configurations through LinearMethodUpdate.
    table_cases = [
        (0.0, 0.0, 1.0, -0.294731415196905, 1.557719626469761, 0.903967757693709,
         [0.070850650803380, -0.442330494171401], 0.473026410140131, -0.493466316173921),
        (0.0, 0.0, 0.5, -0.294731415196905, 1.557719626469761, 0.903967757693709,
         [0.064046703937871, -0.399852504975672], 0.427600623304279, -0.446077639329114),
        (0.01, 0.0, 0.5, -0.294857677953108, 1.557593363713558, 0.905598833404980,
         [0.072722351501727, -0.385778280012415], 0.424135300270489, -0.443417208687018),
        (0.01, 0.05, 0.5, -0.294834671209192, 1.557616370457474, 0.904512612191523,
         [0.083426078399435, -0.366346791176260], 0.426446871704397, -0.438223762233351),
        (0.1, 0.0, 0.0, -0.292380158547706, 1.560070883118960, 0.915521616974894,
         [0.109228457889514, -0.279532754115445], 0.368285761066716, -0.388750014888534),
        (1.0e6, 0.0, 1.0, -7.84769297e-07, 1.852450256897370, 0.999999999999423,
         [5.96561210e-07, -4.62247097e-07], 1.074373818e-06, -1.139117426731e-06),
    ]
    for c_diag, c_ovlp, xi, exp_dlam, exp_lam, exp_c0, exp_dt, exp_ovlp, exp_dd in table_cases:
        param = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
        before = param.detach().clone()
        policy = LinearMethodPolicy(
            derivative_block_shift=c_diag,
            overlap_shift=c_ovlp,
            rescaling_xi=xi,
        )
        method = _lm_method(param, policy=policy)
        result = method.update(_tangent_input(param, o_mat, ge_mat, e_loc, positions=positions))

        assert result.applied is True
        assert result.reason == "applied"
        tel = method.last_telemetry
        assert isinstance(tel, LinearMethodTelemetry)
        assert isinstance(tel.diagnostics, LinearMethodSolveDiagnostics)
        diag = tel.diagnostics
        tol = 1.0e-8 if c_diag >= 1.0e6 else 1.0e-11
        assert diag.selected_delta_eigenvalue == pytest.approx(exp_dlam, abs=tol, rel=tol)
        assert diag.selected_eigenvalue == pytest.approx(exp_lam, abs=tol, rel=tol)
        assert diag.selected_c0 == pytest.approx(exp_c0, abs=tol, rel=tol)
        assert diag.reg_overlap_step_norm == pytest.approx(exp_ovlp, abs=tol, rel=tol)
        assert diag.applied_reg_overlap_step_norm == pytest.approx(exp_ovlp, abs=tol, rel=tol)
        assert diag.directional_derivative == pytest.approx(exp_dd, abs=tol, rel=tol)
        if c_ovlp == 0.05:
            assert diag.rescaled_overlap_step_norm == pytest.approx(
                0.418089033925017, abs=1.0e-11, rel=1.0e-11
            )
            assert diag.applied_overlap_step_norm == pytest.approx(
                0.418089033925017, abs=1.0e-11, rel=1.0e-11
            )
        actual_dt = param.detach() - before
        torch.testing.assert_close(
            actual_dt,
            torch.tensor(exp_dt, dtype=torch.float64),
            atol=tol,
            rtol=tol,
        )
        assert result.grad_norm == pytest.approx(
            2.0 * float(torch.linalg.vector_norm(expected_hc).item()),
            abs=1.0e-12,
            rel=1.0e-12,
        )

    # Steepest-descent limit (c_diag = 1e6, c_ovlp = 0, xi = 1.0) vs -h_c / 1e6.
    sd_outcome = solve_dense_linear_method(
        matrices,
        policy=LinearMethodPolicy(derivative_block_shift=1.0e6, overlap_shift=0.0, rescaling_xi=1.0),
    )
    assert isinstance(sd_outcome, LinearMethodSolveOutcome)
    assert sd_outcome.accepted and sd_outcome.delta_theta is not None
    sd_ref = -expected_hc / 1.0e6
    sd_rel_err = float(
        torch.linalg.vector_norm(sd_outcome.delta_theta - sd_ref).item()
        / torch.linalg.vector_norm(sd_ref).item()
    )
    assert sd_rel_err < 5.0e-6

    # Active overlap-metric trust radius (0.1) at (0.01, 0.0, 0.5).
    p_ovlp = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    before_ovlp = p_ovlp.detach().clone()
    m_ovlp = _lm_method(
        p_ovlp,
        policy=LinearMethodPolicy(
            derivative_block_shift=0.01,
            overlap_shift=0.0,
            rescaling_xi=0.5,
            overlap_metric_trust_radius=0.1,
        ),
    )
    m_ovlp.update(_tangent_input(p_ovlp, o_mat, ge_mat, e_loc, positions=positions))
    assert m_ovlp.last_telemetry is not None and m_ovlp.last_telemetry.diagnostics is not None
    assert m_ovlp.last_telemetry.overlap_trust_scale == pytest.approx(
        0.235773820137644, abs=1.0e-12, rel=1.0e-12
    )
    assert m_ovlp.last_telemetry.euclidean_trust_scale == 1.0
    assert m_ovlp.last_telemetry.trust_scale == pytest.approx(
        0.235773820137644, abs=1.0e-12, rel=1.0e-12
    )
    assert m_ovlp.last_telemetry.diagnostics.applied_overlap_step_norm == pytest.approx(
        0.1, abs=1.0e-12
    )
    torch.testing.assert_close(
        p_ovlp.detach() - before_ovlp,
        torch.tensor([0.017146026622955, -0.090956418804657], dtype=torch.float64),
        atol=1.0e-11,
        rtol=1.0e-11,
    )

    # Active squared Euclidean update-norm bound (1e-3) AND linear max_update_norm (sqrt(1e-3)).
    p_sq = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    before_sq = p_sq.detach().clone()
    m_sq = _lm_method(
        p_sq,
        policy=LinearMethodPolicy(
            derivative_block_shift=0.01,
            overlap_shift=0.0,
            rescaling_xi=0.5,
            squared_update_norm_bound=1.0e-3,
        ),
    )
    m_sq.update(_tangent_input(p_sq, o_mat, ge_mat, e_loc, positions=positions))
    assert m_sq.last_telemetry is not None
    assert m_sq.last_telemetry.euclidean_trust_scale == pytest.approx(
        0.080552638160566, abs=1.0e-12, rel=1.0e-12
    )
    dt_sq = p_sq.detach() - before_sq
    assert float(torch.dot(dt_sq, dt_sq).item()) == pytest.approx(1.0e-3, abs=1.0e-12)

    p_lin = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    before_lin = p_lin.detach().clone()
    m_lin = _lm_method(
        p_lin,
        policy=LinearMethodPolicy(
            derivative_block_shift=0.01,
            overlap_shift=0.0,
            rescaling_xi=0.5,
            max_update_norm=math.sqrt(1.0e-3),
        ),
    )
    m_lin.update(_tangent_input(p_lin, o_mat, ge_mat, e_loc, positions=positions))
    torch.testing.assert_close(p_lin.detach() - before_lin, dt_sq, atol=1.0e-12, rtol=1.0e-12)

    # step_scale != 1.0 pre- vs post-scaling telemetry norm semantics (54ab0807 NB4).
    p_sc = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    before_sc = p_sc.detach().clone()
    m_sc = _lm_method(
        p_sc,
        policy=LinearMethodPolicy(
            derivative_block_shift=0.01,
            overlap_shift=0.05,
            rescaling_xi=0.5,
            step_scale=0.5,
            overlap_metric_trust_radius=0.1,
        ),
    )
    m_sc.update(_tangent_input(p_sc, o_mat, ge_mat, e_loc, positions=positions))
    tel_sc = m_sc.last_telemetry
    assert tel_sc is not None and tel_sc.diagnostics is not None
    diag_sc = tel_sc.diagnostics
    assert diag_sc.rescaled_overlap_step_norm == pytest.approx(0.418089033925017, abs=1.0e-11)
    assert diag_sc.reg_overlap_step_norm == pytest.approx(0.426446871704397, abs=1.0e-11)
    assert tel_sc.applied_update_norm == pytest.approx(
        0.5 * tel_sc.trust_scale * tel_sc.update_direction_norm, abs=1.0e-14
    )
    assert diag_sc.applied_overlap_step_norm == pytest.approx(
        0.5 * tel_sc.trust_scale * diag_sc.rescaled_overlap_step_norm, abs=1.0e-14
    )
    assert diag_sc.applied_reg_overlap_step_norm == pytest.approx(
        0.5 * tel_sc.trust_scale * diag_sc.reg_overlap_step_norm, abs=1.0e-14
    )
    assert diag_sc.applied_reg_overlap_step_norm == pytest.approx(0.1, abs=1.0e-12)
    assert float(torch.linalg.vector_norm(p_sc.detach() - before_sc).item()) == pytest.approx(
        tel_sc.applied_update_norm, abs=1.0e-14
    )

    # Constant energy-shift invariance (E_L -> E_L + 1000.0) at c_ovlp=0.0 and c_ovlp=0.05.
    for c_ovlp in (0.0, 0.05):
        pol = LinearMethodPolicy(derivative_block_shift=0.01, overlap_shift=c_ovlp, rescaling_xi=0.5)
        base_out = solve_dense_linear_method(matrices, policy=pol)
        shifted_mat = assemble_dense_linear_method_matrices(
            e_loc + 1000.0, o_mat, ge_mat, solve_dtype=torch.float64
        )
        shifted_out = solve_dense_linear_method(shifted_mat, policy=pol)
        assert base_out.accepted and shifted_out.accepted
        assert base_out.delta_theta is not None and shifted_out.delta_theta is not None
        assert shifted_out.diagnostics.selected_delta_eigenvalue == pytest.approx(
            base_out.diagnostics.selected_delta_eigenvalue, abs=1.0e-12
        )
        assert shifted_out.diagnostics.selected_eigenvalue == pytest.approx(
            base_out.diagnostics.selected_eigenvalue + 1000.0, abs=1.0e-10
        )
        torch.testing.assert_close(
            shifted_out.delta_theta, base_out.delta_theta, atol=1.0e-12, rtol=1.0e-12
        )


def test_oracle_a_end_to_end_autograd_tangent_step_and_trainer() -> None:
    """Real KineticEnergy + HarmonicTrap through evaluate_hamiltonian_tangent_step matches Oracle A."""

    positions, exp_e_loc, exp_o, exp_ge = _oracle_a_arrays()
    batch = ElectronBatch(
        positions=positions,
        spins=torch.ones((12, 1), dtype=torch.float64),
    )
    terms = [KineticEnergy(), HarmonicTrap(omega=1.0)]

    model = _OracleAHarmonicWavefunction(alpha=0.35, beta=0.40)
    update_input, term_map = evaluate_hamiltonian_tangent_step(
        model=model,
        hamiltonian_terms=terms,
        batch=batch,
        chunk_size=5,
        return_terms=True,
    )
    assert term_map is not None
    torch.testing.assert_close(update_input.local_energy, exp_e_loc, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(update_input.parameter_scores.blocks[0], exp_o, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(
        update_input.local_energy_derivatives.blocks[0], exp_ge, atol=1.0e-11, rtol=1.0e-11
    )

    # Full VMCTrainer step matches Table 5.1 (0.01, 0.0, 0.5).
    policy = LinearMethodPolicy(derivative_block_shift=0.01, overlap_shift=0.0, rescaling_xi=0.5)
    method = LinearMethodUpdate(
        torch.optim.SGD(model.parameters(), lr=1.0),
        model_parameters=ModelParameterBinding(parameters=tuple(model.parameters())),
        policy=policy,
    )
    before = model.theta.detach().clone()
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, return_terms=True, update_method=method)
    context = _StubContext()
    trainer.fit(
        model=model,
        sampler=_FixedSampler(batch),
        hamiltonian_terms=terms,
        optimizer=method.optimizer,
        context=context,
        emit=lambda **_: None,
    )
    assert trainer.completed_updates == 1
    torch.testing.assert_close(
        model.theta.detach() - before,
        torch.tensor([0.072722351501727, -0.385778280012415], dtype=torch.float64),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    train_metrics = [m for ns, m in context.records if ns == "train"][-1]
    assert train_metrics["lm_applied"] is True
    assert train_metrics["lm_solve_selected_delta_eigenvalue"] == pytest.approx(
        -0.294857677953108, abs=1.0e-11
    )


# ---------------------------------------------------------------------------
# 3. Oracle System B: Rank-Deficient Gauge + Disconnected Extension (P=4, r=2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("c_diag", "c_ovlp", "exp_dlam", "exp_active_dt"),
    [
        (0.0, 0.0, -0.294731415196905, [0.064046703937871, -0.399852504975672]),
        (0.01, 0.05, -0.294834671209192, [0.083426078399435, -0.366346791176260]),
        (0.0, 0.05, -0.294875089822993, [0.076190161418772, -0.378216126232356]),
    ],
)
def test_oracle_b_rank_deficient_gauge_and_disconnected_projection(
    c_diag: float,
    c_ovlp: float,
    exp_dlam: float,
    exp_active_dt: list[float],
) -> None:
    """Pre-shift rank_cutoff projects out gauge+disconnected modes for all c_ovlp >= 0."""

    positions, e_loc, o_a, ge_a = _oracle_a_arrays()
    ones_col = torch.ones((12, 1), dtype=torch.float64)
    zeros_col = torch.zeros((12, 1), dtype=torch.float64)
    # Append constant-score gauge mode gamma (O=1, gE=0) and disconnected mode delta (O=0, gE=0).
    o_b = torch.cat((o_a, ones_col, zeros_col), dim=1)
    ge_b = torch.cat((ge_a, zeros_col, zeros_col), dim=1)

    param = torch.nn.Parameter(torch.tensor([0.35, 0.40, 0.10, -0.20], dtype=torch.float64))
    before = param.detach().clone()
    policy = LinearMethodPolicy(
        derivative_block_shift=c_diag,
        overlap_shift=c_ovlp,
        rescaling_xi=0.5,
        rank_cutoff=1.0e-12,
    )
    method = _lm_method(param, policy=policy)
    result = method.update(_tangent_input(param, o_b, ge_b, e_loc, positions=positions))

    assert result.applied is True
    assert method.last_telemetry is not None and method.last_telemetry.diagnostics is not None
    diag = method.last_telemetry.diagnostics
    assert diag.overlap_modes == 4
    assert diag.overlap_retained_modes == 2
    assert diag.overlap_truncated_modes == 2
    # Raw UNSHIFTED minimum retained overlap eigenvalue.
    assert diag.overlap_min_retained_eigenvalue == pytest.approx(
        0.121122595548328, abs=1.0e-12, rel=1.0e-12
    )
    assert diag.selected_delta_eigenvalue == pytest.approx(exp_dlam, abs=1.0e-11, rel=1.0e-11)

    # Also compare directly against the 2-parameter Oracle A solve with identical policy.
    oracle_a_out = solve_dense_linear_method(
        assemble_dense_linear_method_matrices(e_loc, o_a, ge_a, solve_dtype=torch.float64),
        policy=policy,
    )
    assert oracle_a_out.accepted and oracle_a_out.delta_theta is not None
    dt = param.detach() - before
    torch.testing.assert_close(dt[:2], oracle_a_out.delta_theta, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(
        dt[:2],
        torch.tensor(exp_active_dt, dtype=torch.float64),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    assert abs(float(dt[2].item())) <= 1.0e-14
    assert abs(float(dt[3].item())) <= 1.0e-14


# ---------------------------------------------------------------------------
# 4. Oracle System C: Finite-Sample Zero-Variance Theorem + Mutants
# ---------------------------------------------------------------------------


def test_oracle_c_finite_sample_zero_variance_theorem_and_asymmetry_mutants() -> None:
    """4-point linear-span ansatz recovers exact (0.5, 1.5); asymmetry mutants fail > 0.04 Ha."""

    theta1 = 0.3
    xs = torch.tensor([-1.1, -0.2, 0.5, 1.3], dtype=torch.float64)
    denom = 1.0 + theta1 * xs
    o_mat = (xs / denom).unsqueeze(1)
    e_loc = (0.5 + 1.5 * theta1 * xs) / denom
    ge_mat = (xs / (denom * denom)).unsqueeze(1)

    matrices = assemble_dense_linear_method_matrices(
        e_loc, o_mat, ge_mat, solve_dtype=torch.float64
    )
    assert matrices.energy_mean == pytest.approx(0.463660805377541, abs=1.0e-12)
    assert float(matrices.S_param[0, 0].item()) == pytest.approx(0.936447123658302, abs=1.0e-12)
    assert float(matrices.h_c[0].item()) == pytest.approx(0.280934137097490, abs=1.0e-12)
    assert float(matrices.h_r[0].item()) == pytest.approx(-0.125532438960895, abs=1.0e-12)
    assert float((matrices.h_r[0] - matrices.h_c[0]).item()) == pytest.approx(
        -0.406466576058385, abs=1.0e-12
    )
    assert float(matrices.dH11[0, 0].item()) == pytest.approx(1.004506592218825, abs=1.0e-12)

    # Full 2x2 spectrum of (H, S) must equal exact {0.5, 1.5}.
    s_inv_sqrt = float(matrices.S_param[0, 0].item()) ** -0.5
    m_mat = torch.tensor(
        [
            [0.0, float(matrices.h_r[0].item()) * s_inv_sqrt],
            [float(matrices.h_c[0].item()) * s_inv_sqrt, float(matrices.dH11[0, 0].item()) * s_inv_sqrt**2],
        ],
        dtype=torch.float64,
    )
    evals = torch.linalg.eigvals(m_mat).real.sort().values + matrices.energy_mean
    torch.testing.assert_close(
        evals, torch.tensor([0.5, 1.5], dtype=torch.float64), atol=1.0e-12, rtol=1.0e-12
    )

    policy = LinearMethodPolicy(derivative_block_shift=0.0, overlap_shift=0.0, rescaling_xi=1.0)
    outcome = solve_dense_linear_method(matrices, policy=policy)
    assert outcome.accepted is True and outcome.delta_theta is not None
    assert outcome.diagnostics.selected_eigenvalue == pytest.approx(0.5, abs=1.0e-12)
    assert outcome.diagnostics.selected_delta_eigenvalue is not None
    assert outcome.diagnostics.selected_delta_eigenvalue > 0.0
    assert outcome.diagnostics.directional_derivative is not None
    assert outcome.diagnostics.directional_derivative < 0.0

    raw_delta = float(outcome.delta_theta[0].item())
    assert raw_delta == pytest.approx(-0.289480511358340, abs=1.0e-12)
    mean_o1 = float(o_mat.mean().item())
    # Exact linear-parameter jump maps theta1 = 0.3 -> 0.0 in one step.
    assert raw_delta / (1.0 - mean_o1 * raw_delta) == pytest.approx(-0.3, abs=1.0e-12)

    # Mutant-killer checks (54ab0807 §5.3 & NB2):
    # 1. Symmetrized H breaks lambda_0 = 0.5 by 0.042316 Ha (> 0.04 Ha).
    h_sym = 0.5 * (matrices.h_c + matrices.h_r)
    sym_matrices = DenseLinearMethodMatrices(
        count=matrices.count,
        energy_mean=matrices.energy_mean,
        S_param=matrices.S_param,
        h_c=h_sym,
        h_r=h_sym,
        dH11=0.5 * (matrices.dH11 + matrices.dH11.transpose(0, 1)),
        dtype=matrices.dtype,
    )
    sym_out = solve_dense_linear_method(sym_matrices, policy=policy)
    assert sym_out.accepted and sym_out.diagnostics.selected_eigenvalue is not None
    assert sym_out.diagnostics.selected_eigenvalue == pytest.approx(0.457684, abs=1.0e-5)
    assert abs(sym_out.diagnostics.selected_eigenvalue - 0.5) > 0.04

    # 2. Dropping <g^E> from H_01 breaks lambda_0 by 0.109869 Ha.
    drop_ge_matrices = DenseLinearMethodMatrices(
        count=matrices.count,
        energy_mean=matrices.energy_mean,
        S_param=matrices.S_param,
        h_c=matrices.h_c,
        h_r=matrices.h_c.clone(),
        dH11=matrices.dH11,
        dtype=matrices.dtype,
    )
    drop_out = solve_dense_linear_method(drop_ge_matrices, policy=policy)
    assert drop_out.accepted and drop_out.diagnostics.selected_eigenvalue is not None
    assert drop_out.diagnostics.selected_eigenvalue == pytest.approx(0.390131, abs=1.0e-5)
    assert abs(drop_out.diagnostics.selected_eigenvalue - 0.5) > 0.10

    # 3. Flipping sign of g^E breaks lambda_0 by 1.588874 Ha.
    flip_matrices = assemble_dense_linear_method_matrices(
        e_loc, o_mat, -ge_mat, solve_dtype=torch.float64
    )
    flip_out = solve_dense_linear_method(flip_matrices, policy=policy)
    assert flip_out.accepted and flip_out.diagnostics.selected_eigenvalue is not None
    assert flip_out.diagnostics.selected_eigenvalue == pytest.approx(-1.088874, abs=1.0e-5)
    assert abs(flip_out.diagnostics.selected_eigenvalue - 0.5) > 1.5


# ---------------------------------------------------------------------------
# 5. Oracle System D: Exact Ground-State Zero-Variance Stationary Point
# ---------------------------------------------------------------------------


def test_oracle_d_exact_ground_state_stationary_point() -> None:
    """At exact eigenstate (0.5, 0.0), h_c == 0, h_r != 0, dlam == 0, and delta_theta == 0."""

    positions, e_loc, o_mat, ge_mat = _oracle_a_arrays(alpha=0.5, beta=0.0, omega=1.0)
    matrices = assemble_dense_linear_method_matrices(
        e_loc, o_mat, ge_mat, solve_dtype=torch.float64
    )
    assert matrices.energy_mean == pytest.approx(1.5, abs=1.0e-14)
    assert float(torch.linalg.vector_norm(matrices.h_c).item()) <= 1.0e-14
    assert float(torch.linalg.vector_norm(matrices.h_r).item()) > 1.0e-2

    param = torch.nn.Parameter(torch.tensor([0.5, 0.0], dtype=torch.float64))
    before = param.detach().clone()
    method = _lm_method(
        param,
        policy=LinearMethodPolicy(derivative_block_shift=0.0, overlap_shift=0.0, rescaling_xi=0.5),
    )
    result = method.update(_tangent_input(param, o_mat, ge_mat, e_loc, positions=positions))
    assert result.applied is True
    assert method.last_telemetry is not None and method.last_telemetry.diagnostics is not None
    diag = method.last_telemetry.diagnostics
    assert diag.selected_delta_eigenvalue == pytest.approx(0.0, abs=1.0e-13)
    assert diag.selected_eigenvalue == pytest.approx(1.5, abs=1.0e-13)
    assert float(torch.linalg.vector_norm(param.detach() - before).item()) <= 1.0e-13


# ---------------------------------------------------------------------------
# 6. Oracle System E: Refusal Oracles, Nullable Skip Diagnostics, Sink Safety
# ---------------------------------------------------------------------------


def test_oracle_e_refusal_oracles_nullable_diagnostics_and_sink_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oracle E.1--E.4 plus eigensolve/nonfinite skips populate exact None fields atomically."""

    # E.1: Complex conjugate pair on 1D 1e HO, alpha=0.05, xs=[0.1, 0.5, 1.0, 1.5].
    alpha = 0.05
    xs = torch.tensor([0.1, 0.5, 1.0, 1.5], dtype=torch.float64)
    r2 = xs * xs
    o_e1 = (-r2).unsqueeze(1)
    e_e1 = alpha + (0.5 - 2.0 * alpha * alpha) * r2
    ge_e1 = (1.0 - 4.0 * alpha * r2).unsqueeze(1)

    p_e1 = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.float64))
    p_e1.grad = torch.tensor([7.0], dtype=torch.float64)
    before_e1 = p_e1.detach().clone()
    m_e1 = _lm_method(
        p_e1,
        policy=LinearMethodPolicy(derivative_block_shift=0.0, overlap_shift=0.0, rescaling_xi=0.5),
    )
    res_e1 = m_e1.update(_tangent_input(p_e1, o_e1, ge_e1, e_e1))
    assert res_e1.applied is False
    assert res_e1.reason == "no_admissible_real_eigenpair"
    assert torch.equal(p_e1.detach(), before_e1)
    assert p_e1.grad is not None and torch.equal(p_e1.grad, torch.tensor([7.0], dtype=torch.float64))
    assert m_e1.completed_updates == 0
    tel_e1 = m_e1.last_telemetry
    assert tel_e1 is not None and tel_e1.diagnostics is not None
    assert tel_e1.diagnostics.n_complex_eigenpairs == 2
    assert tel_e1.diagnostics.n_admissible_real_eigenpairs == 0
    assert tel_e1.diagnostics.overlap_min_retained_eigenvalue is not None
    for field_name in _SELECTED_12_OPTIONAL_FIELDS:
        assert getattr(tel_e1.diagnostics, field_name) is None, f"{field_name} must be None"
    _assert_csv_and_jsonl_roundtrip(
        tmp_path, name="oracle_e1_skip", namespace="train", metrics=tel_e1.as_metrics()
    )

    # E.1 stabilized at c_diag=0.5 -> applied=True with exact dlam = 0.252105605595.
    m_e1_stab = _lm_method(
        p_e1,
        policy=LinearMethodPolicy(derivative_block_shift=0.5, overlap_shift=0.0, rescaling_xi=0.5),
    )
    res_e1_stab = m_e1_stab.update(_tangent_input(p_e1, o_e1, ge_e1, e_e1))
    assert res_e1_stab.applied is True
    assert m_e1_stab.last_telemetry is not None and m_e1_stab.last_telemetry.diagnostics is not None
    assert m_e1_stab.last_telemetry.diagnostics.selected_delta_eigenvalue == pytest.approx(
        0.252105605595, abs=1.0e-9
    )
    assert m_e1_stab.last_telemetry.diagnostics.directional_derivative is not None
    assert m_e1_stab.last_telemetry.diagnostics.directional_derivative < 0.0

    # E.2: Non-descent step (S=[[1]], h_c=[-1], h_r=[+2], dH11=[[-3]]).
    e2_matrices = DenseLinearMethodMatrices(
        count=4,
        energy_mean=0.0,
        S_param=torch.tensor([[1.0]], dtype=torch.float64),
        h_c=torch.tensor([-1.0], dtype=torch.float64),
        h_r=torch.tensor([2.0], dtype=torch.float64),
        dH11=torch.tensor([[-3.0]], dtype=torch.float64),
        dtype="torch.float64",
    )
    e2_skip = solve_dense_linear_method(
        e2_matrices,
        policy=LinearMethodPolicy(
            derivative_block_shift=0.0, overlap_shift=0.0, rescaling_xi=1.0, require_descent=True
        ),
    )
    assert e2_skip.accepted is False
    assert e2_skip.reason == "non_descent_step"
    assert e2_skip.delta_theta is None
    assert e2_skip.diagnostics.directional_derivative == pytest.approx(2.0, abs=1.0e-12)
    assert e2_skip.diagnostics.applied_overlap_step_norm is None
    assert e2_skip.diagnostics.applied_reg_overlap_step_norm is None
    assert e2_skip.update_direction_norm == pytest.approx(1.0, abs=1.0e-12)

    e2_allow = solve_dense_linear_method(
        e2_matrices,
        policy=LinearMethodPolicy(
            derivative_block_shift=0.0, overlap_shift=0.0, rescaling_xi=0.5, require_descent=False
        ),
    )
    assert e2_allow.accepted is True and e2_allow.delta_theta is not None
    assert float(e2_allow.delta_theta[0].item()) == pytest.approx(-1.0 / math.sqrt(2.0), abs=1.0e-12)

    # E.3: min_c0 admissibility filter.
    e3a_matrices = DenseLinearMethodMatrices(
        count=4,
        energy_mean=0.0,
        S_param=torch.tensor([[1.0]], dtype=torch.float64),
        h_c=torch.tensor([0.0], dtype=torch.float64),
        h_r=torch.tensor([0.0], dtype=torch.float64),
        dH11=torch.tensor([[-1.0]], dtype=torch.float64),
        dtype="torch.float64",
    )
    e3a_out = solve_dense_linear_method(
        e3a_matrices,
        policy=LinearMethodPolicy(derivative_block_shift=0.0, overlap_shift=0.0, rescaling_xi=0.5),
    )
    assert e3a_out.accepted is True and e3a_out.delta_theta is not None
    assert e3a_out.diagnostics.selected_delta_eigenvalue == pytest.approx(0.0, abs=1.0e-13)
    assert float(e3a_out.delta_theta[0].item()) == pytest.approx(0.0, abs=1.0e-13)

    e3b_matrices = DenseLinearMethodMatrices(
        count=4,
        energy_mean=0.0,
        S_param=torch.tensor([[1.0]], dtype=torch.float64),
        h_c=torch.tensor([-1.0], dtype=torch.float64),
        h_r=torch.tensor([-1.0], dtype=torch.float64),
        dH11=torch.tensor([[0.0]], dtype=torch.float64),
        dtype="torch.float64",
    )
    e3b_out = solve_dense_linear_method(
        e3b_matrices,
        policy=LinearMethodPolicy(
            derivative_block_shift=0.0, overlap_shift=0.0, min_c0=0.99
        ),
    )
    assert e3b_out.accepted is False
    assert e3b_out.reason == "no_admissible_real_eigenpair"
    assert e3b_out.diagnostics.n_complex_eigenpairs == 0
    assert e3b_out.diagnostics.n_admissible_real_eigenpairs == 0
    for field_name in _SELECTED_12_OPTIONAL_FIELDS:
        assert getattr(e3b_out.diagnostics, field_name) is None

    # E.4: Constant non-binary score O=0.3, N=10, P=1 at c_diag in {0.0, 0.01}.
    o_const = torch.full((10, 1), 0.3, dtype=torch.float64)
    ge_const = torch.zeros((10, 1), dtype=torch.float64)
    e_const = torch.sin(torch.arange(1, 11, dtype=torch.float64))
    for c_diag in (0.0, 0.01):
        p_const = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        before_const = p_const.detach().clone()
        m_const = _lm_method(
            p_const,
            policy=LinearMethodPolicy(derivative_block_shift=c_diag, overlap_shift=0.0),
        )
        res_const = m_const.update(_tangent_input(p_const, o_const, ge_const, e_const))
        assert res_const.applied is False
        assert res_const.reason == "degenerate_overlap_subspace"
        assert torch.equal(p_const.detach(), before_const)
        assert p_const.grad is None
        tel_c = m_const.last_telemetry
        assert tel_c is not None and tel_c.diagnostics is not None
        assert tel_c.diagnostics.overlap_retained_modes == 0
        assert tel_c.diagnostics.overlap_truncated_modes == 1
        for field_name in _ALL_13_OPTIONAL_FIELDS:
            assert getattr(tel_c.diagnostics, field_name) is None, f"{field_name} must be None"
        _assert_csv_and_jsonl_roundtrip(
            tmp_path,
            name=f"oracle_e4_const_{c_diag}",
            namespace="train",
            metrics=tel_c.as_metrics(),
        )

    # E.4 Tiny non-constant control (a): 1e-50 reparametrization of Oracle A.
    _, e_a, o_a, ge_a = _oracle_a_arrays()
    scale = 1.0e-50
    pol_a = LinearMethodPolicy(derivative_block_shift=0.0, overlap_shift=0.0, rescaling_xi=0.5)
    out_a = solve_dense_linear_method(
        assemble_dense_linear_method_matrices(e_a, o_a, ge_a, solve_dtype=torch.float64),
        policy=pol_a,
    )
    out_a50 = solve_dense_linear_method(
        assemble_dense_linear_method_matrices(
            e_a, o_a * scale, ge_a * scale, solve_dtype=torch.float64
        ),
        policy=pol_a,
    )
    assert out_a.accepted and out_a50.accepted
    assert out_a.delta_theta is not None and out_a50.delta_theta is not None
    assert out_a50.diagnostics.overlap_retained_modes == 2
    assert out_a50.diagnostics.selected_delta_eigenvalue == pytest.approx(
        out_a.diagnostics.selected_delta_eigenvalue, abs=1.0e-12
    )
    torch.testing.assert_close(
        out_a50.delta_theta * scale, out_a.delta_theta, atol=1.0e-11, rtol=1.0e-11
    )

    # E.4 Tiny non-constant control (b): O = 1.0 + 1e-10 * (-1)^b at c_diag in {0.0, 0.01}.
    alt_sign = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(10)], dtype=torch.float64)
    o_tiny = (1.0 + 1.0e-10 * alt_sign).unsqueeze(1)
    ge_tiny = torch.zeros((10, 1), dtype=torch.float64)
    # Scale energy correlation so M = [[0, -1e8], [-1e8, c_diag / 1e-20]] stays
    # well above float64 machine precision at both c_diag = 0.0 and c_diag = 0.01.
    e_tiny = -1.0e8 * alt_sign
    for c_diag in (0.0, 0.01):
        out_tiny = solve_dense_linear_method(
            assemble_dense_linear_method_matrices(
                e_tiny, o_tiny, ge_tiny, solve_dtype=torch.float64
            ),
            policy=LinearMethodPolicy(derivative_block_shift=c_diag, overlap_shift=0.0),
        )
        assert out_tiny.accepted is True
        assert out_tiny.diagnostics.overlap_retained_modes == 1

    # Additional numerical skips: eigensolve_failed, nonfinite_eigensolve, nonfinite_update_direction.
    mat_a = assemble_dense_linear_method_matrices(e_a, o_a, ge_a, solve_dtype=torch.float64)

    def _raise_linalg(_mat: torch.Tensor):
        raise torch.linalg.LinAlgError("synthetic convergence failure")

    with monkeypatch.context() as mp:
        mp.setattr(torch.linalg, "eig", _raise_linalg)
        fail_out = solve_dense_linear_method(mat_a, policy=pol_a)
    assert fail_out.accepted is False and fail_out.reason == "eigensolve_failed"
    assert fail_out.diagnostics.overlap_min_retained_eigenvalue is not None
    for field_name in _SELECTED_12_OPTIONAL_FIELDS:
        assert getattr(fail_out.diagnostics, field_name) is None

    def _nan_eig(mat: torch.Tensor):
        n = int(mat.shape[0])
        return (
            torch.full((n,), float("nan"), dtype=torch.complex128),
            torch.eye(n, dtype=torch.complex128),
        )

    with monkeypatch.context() as mp:
        mp.setattr(torch.linalg, "eig", _nan_eig)
        nan_out = solve_dense_linear_method(mat_a, policy=pol_a)
    assert nan_out.accepted is False and nan_out.reason == "nonfinite_eigensolve"
    assert nan_out.diagnostics.overlap_min_retained_eigenvalue is not None
    for field_name in _SELECTED_12_OPTIONAL_FIELDS:
        assert getattr(nan_out.diagnostics, field_name) is None

    # Overflowing step_scale triggers nonfinite_update_direction and stays JSONL/CSV safe.
    # Scaling (O, gE) by 1e-2 yields ||delta_rescaled|| ~ 39, so 1e308 * delta_rescaled -> inf.
    p_ovf = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    before_ovf = p_ovf.detach().clone()
    m_ovf = _lm_method(
        p_ovf,
        policy=LinearMethodPolicy(
            derivative_block_shift=0.0, overlap_shift=0.0, step_scale=1.0e308
        ),
    )
    res_ovf = m_ovf.update(_tangent_input(p_ovf, o_a * 1.0e-2, ge_a * 1.0e-2, e_a))
    assert res_ovf.applied is False
    assert res_ovf.reason == "nonfinite_update_direction"
    assert torch.equal(p_ovf.detach(), before_ovf)
    assert m_ovf.last_telemetry is not None and m_ovf.last_telemetry.diagnostics is not None
    assert m_ovf.last_telemetry.diagnostics.applied_overlap_step_norm is None
    assert m_ovf.last_telemetry.diagnostics.applied_reg_overlap_step_norm is None
    assert m_ovf.last_telemetry.diagnostics.directional_derivative is None
    _assert_csv_and_jsonl_roundtrip(
        tmp_path,
        name="nonfinite_dir_skip",
        namespace="train",
        metrics=m_ovf.last_telemetry.as_metrics(),
    )


# ---------------------------------------------------------------------------
# 7. Oracle System F1 & F2: N < P, c_ovlp > 0 Nullspace Regression Oracles
# ---------------------------------------------------------------------------


def test_oracle_f1_and_f2_n_less_than_p_nullspace_regression() -> None:
    """Pre-shift rank_cutoff eliminates N < P spurious mu = c_diag/c_ovlp null modes."""

    # Oracle F1: psi(x) = (1 + th1*x + th2*x^2) exp(-x^2/2) at (0.3, 0.0), xs = [-1.1, 0.5].
    th1, th2 = 0.3, 0.0
    xs1 = torch.tensor([-1.1, 0.5], dtype=torch.float64)
    f1 = 1.0 + th1 * xs1 + th2 * xs1 * xs1
    o_f1 = torch.stack((xs1 / f1, (xs1 * xs1) / f1), dim=1)
    e_f1 = (0.5 - th2 + 1.5 * th1 * xs1 + 2.5 * th2 * xs1 * xs1) / f1
    ge_f1 = torch.stack(
        (
            xs1 * (1.0 + th2 - th2 * xs1 * xs1) / (f1 * f1),
            (2.0 * xs1 * xs1 - 1.0 + th1 * xs1 * (xs1 * xs1 - 1.0)) / (f1 * f1),
        ),
        dim=1,
    )
    mat_f1 = assemble_dense_linear_method_matrices(e_f1, o_f1, ge_f1, solve_dtype=torch.float64)
    w_f1, u_f1 = torch.linalg.eigh(mat_f1.S_param)
    v_null = u_f1[:, 0]
    assert abs(float(w_f1[0].item())) <= 1.0e-14
    assert float(w_f1[1].item()) == pytest.approx(1.708935221193921, abs=1.0e-12)
    assert abs(float(torch.dot(mat_f1.h_r, v_null).item())) == pytest.approx(
        0.381402578329430, abs=1.0e-12
    )

    p_f1 = torch.nn.Parameter(torch.tensor([th1, th2], dtype=torch.float64))
    before_f1 = p_f1.detach().clone()
    m_f1 = _lm_method(
        p_f1,
        policy=LinearMethodPolicy(
            derivative_block_shift=0.005,
            overlap_shift=0.05,
            rescaling_xi=0.5,
            rank_cutoff=1.0e-12,
        ),
    )
    res_f1 = m_f1.update(_tangent_input(p_f1, o_f1, ge_f1, e_f1))
    assert res_f1.applied is True
    assert m_f1.last_telemetry is not None and m_f1.last_telemetry.diagnostics is not None
    d_f1 = m_f1.last_telemetry.diagnostics
    assert d_f1.overlap_modes == 2
    assert d_f1.overlap_retained_modes == 1
    assert d_f1.overlap_truncated_modes == 1
    assert d_f1.overlap_min_retained_eigenvalue == pytest.approx(1.708935221193921, abs=1.0e-12)
    assert d_f1.selected_delta_eigenvalue == pytest.approx(0.192657185133408, abs=1.0e-11)
    assert d_f1.selected_eigenvalue == pytest.approx(0.511605919721338, abs=1.0e-11)
    assert d_f1.selected_c0 == pytest.approx(0.977686920050676, abs=1.0e-11)
    assert d_f1.directional_derivative == pytest.approx(-0.128992661422628, abs=1.0e-11)
    dt_f1 = p_f1.detach() - before_f1
    torch.testing.assert_close(
        dt_f1,
        torch.tensor([-0.125802175747490, 0.096238664446830], dtype=torch.float64),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    # Zero leakage into ker(S_param).
    assert abs(float(torch.dot(dt_f1, v_null).item())) <= 1.0e-14

    # Oracle F2: log|psi(x)| = -alpha*x^2 + beta*x at (0.05, 0.0), xs = [0.1, 1.0].
    alpha, beta = 0.05, 0.0
    xs2 = torch.tensor([0.1, 1.0], dtype=torch.float64)
    o_f2 = torch.stack((-(xs2 * xs2), xs2), dim=1)
    e_f2 = alpha - 0.5 * beta * beta + 2.0 * alpha * beta * xs2 + (0.5 - 2.0 * alpha * alpha) * xs2 * xs2
    ge_f2 = torch.stack((1.0 + 2.0 * beta * xs2 - 4.0 * alpha * xs2 * xs2, -beta + 2.0 * alpha * xs2), dim=1)

    p_f2 = torch.nn.Parameter(torch.tensor([alpha, beta], dtype=torch.float64))
    before_f2 = p_f2.detach().clone()
    m_f2 = _lm_method(
        p_f2,
        policy=LinearMethodPolicy(
            derivative_block_shift=0.001,
            overlap_shift=0.05,
            rescaling_xi=0.5,
            rank_cutoff=1.0e-12,
        ),
    )
    res_f2 = m_f2.update(_tangent_input(p_f2, o_f2, ge_f2, e_f2))
    assert res_f2.applied is False
    assert res_f2.reason == "no_admissible_real_eigenpair"
    assert torch.equal(p_f2.detach(), before_f2)
    assert m_f2.last_telemetry is not None and m_f2.last_telemetry.diagnostics is not None
    d_f2 = m_f2.last_telemetry.diagnostics
    assert d_f2.overlap_retained_modes == 1
    assert d_f2.overlap_truncated_modes == 1
    assert d_f2.overlap_min_retained_eigenvalue == pytest.approx(0.447525, abs=1.0e-12)
    assert d_f2.n_complex_eigenpairs == 2
    assert d_f2.n_admissible_real_eigenpairs == 0
    for field_name in _SELECTED_12_OPTIONAL_FIELDS:
        assert getattr(d_f2, field_name) is None


# ---------------------------------------------------------------------------
# 8. PR #854 NB1 weakref graph-release & staged S_param/dH11 release tests
# ---------------------------------------------------------------------------


def test_pr854_nb1_chunk_term_value_graph_released_before_next_chunk() -> None:
    """evaluate_hamiltonian_tangent_step releases chunk-k term_value graph before chunk k+1."""

    model = _OracleAHarmonicWavefunction(alpha=0.35, beta=0.40)
    batch = ElectronBatch(
        positions=torch.tensor(ORACLE_A_COORDS[:2], dtype=torch.float64).unsqueeze(1),
        spins=torch.ones((2, 1), dtype=torch.float64),
    )

    chunk0_refs: dict[str, weakref.ReferenceType[Any]] = {}
    call_index = 0

    def tracking_local_energy(h_terms: Any, wf: Any, chunk: ElectronBatch, *, return_terms: bool = False):
        nonlocal call_index
        del h_terms
        assert return_terms is True
        if call_index == 1:
            # Before chunk 1 executes, every graph-bearing tensor from chunk 0
            # (including the loop variable `term_value`) must already be dead.
            gc.collect()
            for label, ref in chunk0_refs.items():
                assert ref() is None, f"chunk 0 graph-bearing {label} leaked into chunk 1"

        out = wf(chunk)
        # Distinct graph-bearing term tensor (requires_grad=True) so `term_value`
        # holds a live autograd node if not deleted at the end of the chunk loop.
        term_tensor = out.logabs * 1.25 + 0.5
        total = term_tensor + 0.25
        assert term_tensor.requires_grad and total.requires_grad
        if call_index == 0:
            chunk0_refs["term_value"] = weakref.ref(term_tensor)
            chunk0_refs["total"] = weakref.ref(total)
            chunk0_refs["logabs"] = weakref.ref(out.logabs)
        call_index += 1
        return LocalEnergyResult(
            total=total,
            terms={"synthetic_term": term_tensor},
            wavefunction_output=out,
        )

    update_input, assembled_terms = evaluate_hamiltonian_tangent_step(
        model=model,
        hamiltonian_terms=[],
        batch=batch,
        chunk_size=1,
        return_terms=True,
        local_energy_fn=tracking_local_energy,
    )
    assert call_index == 2
    assert assembled_terms is not None and "synthetic_term" in assembled_terms
    assert not assembled_terms["synthetic_term"].requires_grad
    assert not update_input.local_energy.requires_grad


def test_staged_matrix_lifetime_releases_s_param_and_dh11_before_eig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LinearMethodUpdate.update releases S_param and dH11 before torch.linalg.eig(M)."""

    positions, e_loc, o_mat, ge_mat = _oracle_a_arrays()
    param = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    method = _lm_method(param)

    refs: dict[str, weakref.ReferenceType[Any]] = {}
    orig_assemble = lm_module.assemble_dense_linear_method_matrices
    orig_eig = torch.linalg.eig
    checked = {"eig_called": False}

    def spy_assemble(*args: Any, **kwargs: Any) -> DenseLinearMethodMatrices:
        mats = orig_assemble(*args, **kwargs)
        refs["S_param"] = weakref.ref(mats.S_param)
        refs["dH11"] = weakref.ref(mats.dH11)
        return mats

    def spy_eig(m_mat: torch.Tensor):
        gc.collect()
        checked["eig_called"] = True
        assert refs["S_param"]() is None, "S_param must be freed before torch.linalg.eig(M)"
        assert refs["dH11"]() is None, "dH11 must be freed before torch.linalg.eig(M)"
        return orig_eig(m_mat)

    monkeypatch.setattr(lm_module, "assemble_dense_linear_method_matrices", spy_assemble)
    monkeypatch.setattr(torch.linalg, "eig", spy_eig)

    result = method.update(_tangent_input(param, o_mat, ge_mat, e_loc, positions=positions))
    assert result.applied is True
    assert checked["eig_called"] is True


# ---------------------------------------------------------------------------
# 9. Non-finite row policies, degenerate batches, & loud contract refusals
# ---------------------------------------------------------------------------


def test_nonfinite_row_policies_degenerate_batches_and_loud_refusals() -> None:
    """fail/mask/fail_on_pinned, degenerate skips, phase/graph/binding refusals hold."""

    positions, e_loc, o_mat, ge_mat = _oracle_a_arrays()

    # Direct helper checks for select_tangent_update_rows and flatten_local_energy_derivative_blocks.
    sel = select_tangent_update_rows(
        e_loc, o_mat, ge_mat, nonfinite_local_energy_policy="fail"
    )
    assert isinstance(sel, TangentRowSelection)
    assert sel.n_samples == 12 and sel.n_finite_samples == 12
    assert sel.nonfinite_tangent_row_policy == NONFINITE_TANGENT_ROW_POLICY

    two_slot_layout = ParameterLayout(
        slots=(
            ParameterSlot(ordinal=0, shape=(1,), numel=1, dtype=torch.float64),
            ParameterSlot(ordinal=1, shape=(1,), numel=1, dtype=torch.float64),
        )
    )
    derivs_obj = MaterializedLocalEnergyDerivatives(
        layout=two_slot_layout,
        blocks=(ge_mat[:, :1], ge_mat[:, 1:]),
    )
    torch.testing.assert_close(
        flatten_local_energy_derivative_blocks(derivs_obj, sample_shape=(12,)),
        ge_mat,
        atol=0.0,
        rtol=0.0,
    )

    # 1. nonfinite_local_energy_policy="fail" raises on bad E_L.
    bad_e = e_loc.clone()
    bad_e[3] = float("nan")
    p_fail = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    m_fail = _lm_method(p_fail, nonfinite_local_energy_policy="fail")
    with pytest.raises(RuntimeError, match="nonfinite_local_energy_policy is 'fail'"):
        m_fail.update(_tangent_input(p_fail, o_mat, ge_mat, bad_e, positions=positions))

    # 2. nonfinite_local_energy_policy="mask" drops unpinned non-finite E_L rows
    #    (even if that dropped row also has NaN score/derivative) and matches clean solve.
    keep = [i for i in range(12) if i != 3]
    bad_o_unpinned = o_mat.clone()
    bad_o_unpinned[3, 0] = float("nan")
    bad_ge_unpinned = ge_mat.clone()
    bad_ge_unpinned[3, 1] = float("inf")

    p_mask = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    p_clean = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    m_mask = _lm_method(p_mask, nonfinite_local_energy_policy="mask")
    m_clean = _lm_method(p_clean, nonfinite_local_energy_policy="mask")
    res_mask = m_mask.update(
        _tangent_input(p_mask, bad_o_unpinned, bad_ge_unpinned, bad_e, positions=positions)
    )
    res_clean = m_clean.update(
        _tangent_input(
            p_clean, o_mat[keep], ge_mat[keep], e_loc[keep], positions=positions[keep]
        )
    )
    assert res_mask.applied is True and res_clean.applied is True
    torch.testing.assert_close(p_mask.detach(), p_clean.detach(), atol=1.0e-12, rtol=1.0e-12)
    assert m_mask.last_telemetry is not None
    assert m_mask.last_telemetry.n_samples == 12
    assert m_mask.last_telemetry.n_finite_samples == 11
    assert m_mask.last_telemetry.n_nonfinite_energy_samples == 1
    assert m_mask.last_telemetry.n_nonfinite_score_samples == 1
    assert m_mask.last_telemetry.n_nonfinite_derivative_samples == 1

    # 3. fail_on_pinned refuses non-finite O or gE on any pinned finite-E_L row
    #    under BOTH "fail" and "mask".
    for le_policy in ("fail", "mask"):
        p_pin = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
        before_pin = p_pin.detach().clone()
        m_pin = _lm_method(p_pin, nonfinite_local_energy_policy=le_policy)

        bad_o_pinned = o_mat.clone()
        bad_o_pinned[5, 0] = float("nan")
        with pytest.raises(RuntimeError, match="fail_on_pinned.*parameter scores.*\\[5\\]"):
            m_pin.update(_tangent_input(p_pin, bad_o_pinned, ge_mat, e_loc, positions=positions))
        assert torch.equal(p_pin.detach(), before_pin)

        bad_ge_pinned = ge_mat.clone()
        bad_ge_pinned[7, 1] = float("-inf")
        with pytest.raises(
            RuntimeError, match="fail_on_pinned.*local-energy derivatives.*\\[7\\]"
        ):
            m_pin.update(_tangent_input(p_pin, o_mat, bad_ge_pinned, e_loc, positions=positions))
        assert torch.equal(p_pin.detach(), before_pin)

    # 4. Guarded VMCTrainer policy-mismatch check (activates once PR #852 lands on dev).
    if hasattr(VMCUpdateMethod, "declared_nonfinite_local_energy_policy"):
        model_ho = _OracleAHarmonicWavefunction()
        m_mismatch = LinearMethodUpdate(
            torch.optim.SGD(model_ho.parameters(), lr=1.0),
            model_parameters=ModelParameterBinding(parameters=tuple(model_ho.parameters())),
            nonfinite_local_energy_policy="fail",
        )
        trainer_mismatch = VMCTrainer(
            max_steps=1,
            update_method=m_mismatch,
            nonfinite_local_energy_policy="mask",
        )
        with pytest.raises(ValueError, match="nonfinite"):
            trainer_mismatch.resolve_update_state(
                model=model_ho,
                optimizer=m_mismatch.optimizer,
            )

    # 5. Degenerate batch guards: zero_electron_batch, n_finite==0, n_finite==1.
    p_deg = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    before_deg = p_deg.detach().clone()
    m_deg = _lm_method(p_deg)

    res_vac = m_deg.update(_tangent_input(p_deg, o_mat[:4], ge_mat[:4], e_loc[:4], n_electrons=0))
    assert res_vac.applied is False
    assert res_vac.reason == "zero_electron_batch"
    assert res_vac.grad_norm == 0.0
    assert m_deg.last_telemetry is not None and m_deg.last_telemetry.diagnostics is None
    assert torch.equal(p_deg.detach(), before_deg)

    all_nan = torch.full((4,), float("nan"), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="no finite local-energy sample remains"):
        m_deg.update(_tangent_input(p_deg, o_mat[:4], ge_mat[:4], all_nan, n_electrons=1))

    one_finite = torch.full((4,), float("nan"), dtype=torch.float64)
    one_finite[2] = float(e_loc[2].item())
    res_one = m_deg.update(_tangent_input(p_deg, o_mat[:4], ge_mat[:4], one_finite, n_electrons=1))
    assert res_one.applied is False
    assert res_one.reason == "insufficient_finite_samples"
    assert res_one.grad_norm == 0.0
    assert m_deg.last_telemetry is not None and m_deg.last_telemetry.diagnostics is None
    assert torch.equal(p_deg.detach(), before_deg)

    # 6. Complex phase refusal.
    with pytest.raises(ValueError, match="real wavefunctions only"):
        m_deg.update(
            _tangent_input(
                p_deg,
                o_mat,
                ge_mat,
                e_loc,
                positions=positions,
                phase=torch.zeros((12,), dtype=torch.float64),
            )
        )

    # 7. Graph-bearing step inputs refused loudly across logabs, local_energy, scores, derivatives.
    with pytest.raises(ValueError, match="detached"):
        m_deg.update(
            _tangent_input(
                p_deg,
                o_mat,
                ge_mat,
                e_loc,
                positions=positions,
                logabs=(p_deg.sum()).expand(12),
            )
        )
    with pytest.raises(ValueError, match="detached"):
        m_deg.update(
            _tangent_input(
                p_deg,
                o_mat,
                ge_mat,
                e_loc + 0.0 * p_deg.sum(),
                positions=positions,
            )
        )
    with pytest.raises(ValueError, match="detached"):
        m_deg.update(
            _tangent_input(
                p_deg,
                o_mat + 0.0 * p_deg.sum(),
                ge_mat,
                e_loc,
                positions=positions,
            )
        )
    with pytest.raises(ValueError, match="detached"):
        m_deg.update(
            _tangent_input(
                p_deg,
                o_mat,
                ge_mat + 0.0 * p_deg.sum(),
                e_loc,
                positions=positions,
            )
        )

    # 8. Binding drift (impostor parameter identity & wrong layout) refused loudly.
    impostor = torch.nn.Parameter(p_deg.detach().clone())
    with pytest.raises(ValueError, match="does not reference the bound live"):
        m_deg.update(_tangent_input(impostor, o_mat, ge_mat, e_loc, positions=positions))

    other_p = torch.nn.Parameter(torch.zeros((3,), dtype=torch.float64))
    with pytest.raises(ValueError, match="does not match the bound parameter layout"):
        m_deg.update(
            _tangent_input(
                other_p,
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 3), dtype=torch.float64),
                e_loc[:4],
            )
        )


# ---------------------------------------------------------------------------
# 10. Dense memory admission formulas, preflight, per-step, HI O-raw refusal
# ---------------------------------------------------------------------------


def test_memory_admission_formulas_preflight_per_step_and_hi_o_raw_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory formulas match Table 6.2 and refuse over-budget layouts BEFORE allocation."""

    # Table 6.2 Row 1 (P=2, N=12, float64): B_dense=576 B, B_step=1,024 B (NB1 fix).
    assert DENSE_LM_WORKING_SET_FACTOR == 8
    assert DEFAULT_MAX_DENSE_BYTES == 1024**3
    assert dense_matrix_working_bytes(2, torch.float64) == 576
    assert dense_step_working_bytes(2, 12, torch.float64, torch.float64) == 1024
    assert (
        validate_dense_memory_budget(
            n_parameters=2,
            n_samples=12,
            solve_dtype=torch.float64,
            input_dtype=torch.float64,
            policy=LinearMethodPolicy(max_dense_bytes=1024),
        )
        == 1024
    )
    # Table 6.2 Row 2 (P=64, N=64, float64) & Row 4 (P=2156, N=16) & HI O-raw (P=961,284).
    assert dense_matrix_working_bytes(64, torch.float64) == 8 * 65 * 65 * 8
    assert dense_step_working_bytes(64, 64, torch.float64, torch.float64) == 335936
    assert dense_matrix_working_bytes(2156, torch.float64) == 8 * 2157 * 2157 * 8
    hi_o_raw_dense = dense_matrix_working_bytes(961_284, torch.float64)
    assert hi_o_raw_dense == 8 * 961_285 * 961_285 * 8
    assert hi_o_raw_dense > 50 * (1024**4)  # 53.79 TiB
    # Mixed input_dtype (float32) vs solve_dtype (float64).
    assert dense_step_working_bytes(2, 12, torch.float64, torch.float32) == (
        2 * 12 * 2 * 4 + max(3 * 12 * 2 * 8 + 2 * 4 * 8, 576)
    )

    # Preflight refusal on synthetic HI O-raw P=961,284 layout (zero dense matrix allocation!).
    hi_param = torch.nn.Parameter(torch.empty((961_284,), dtype=torch.float64))
    hi_binding = ModelParameterBinding(parameters=(hi_param,))
    hi_opt = torch.optim.SGD([hi_param], lr=1.0)
    with pytest.raises(LinearMethodMemoryLimitError, match="961284"):
        LinearMethodUpdate(hi_opt, model_parameters=hi_binding)

    # Preflight refusal on max_parameters in both __init__ and rebind_model_parameters.
    p2 = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    p4 = torch.nn.Parameter(torch.zeros((4,), dtype=torch.float64))
    with pytest.raises(LinearMethodMemoryLimitError, match="max_parameters"):
        _lm_method(p4, policy=LinearMethodPolicy(max_parameters=2))

    m_small = _lm_method(p2, policy=LinearMethodPolicy(max_parameters=2))
    with pytest.raises(LinearMethodMemoryLimitError, match="max_parameters"):
        m_small.rebind_model_parameters(ModelParameterBinding(parameters=(p4,)))
    # Original binding stays intact after refused rebind.
    assert m_small.model_parameters.layout is not None
    assert m_small.model_parameters.layout.total_numel == 2

    # Mixed-dtype layout (float32 + float64) promotes input_dtype to float64 when solve_dtype=None.
    p32 = torch.nn.Parameter(torch.ones((2,), dtype=torch.float32))
    p64 = torch.nn.Parameter(torch.ones((2,), dtype=torch.float64))
    mixed_binding = ModelParameterBinding(parameters=(p32, p64))
    # P=4 -> float32 needs 8*25*4=800 B, float64 needs 8*25*8=1600 B.
    # Budget of 1000 B MUST reject because promoted input/solve dtype is float64 (1600 B).
    with pytest.raises(LinearMethodMemoryLimitError, match="1600"):
        LinearMethodUpdate(
            torch.optim.SGD([p32, p64], lr=1.0),
            model_parameters=mixed_binding,
            policy=LinearMethodPolicy(max_dense_bytes=1000),
            conventions=ScoreConventions(solve_dtype=None),
        )

    # Per-step refusal BEFORE flatten_parameter_score_blocks / flatten_local_energy_derivative_blocks.
    positions, e_loc, o_mat, ge_mat = _oracle_a_arrays()
    # P=2, N=12, float64: matrix-only preflight = 576 B, full step = 1024 B.
    # Budget of 800 B passes __init__ (576 <= 800) and MUST fail in update() (1024 > 800)
    # BEFORE any flatten_* helper runs.
    p_step = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    m_step = _lm_method(p_step, policy=LinearMethodPolicy(max_dense_bytes=800))

    def _forbid_flatten(*args: Any, **kwargs: Any):
        raise AssertionError("flatten_* must not be called before validate_dense_memory_budget")

    monkeypatch.setattr(lm_module, "flatten_parameter_score_blocks", _forbid_flatten)
    monkeypatch.setattr(lm_module, "flatten_local_energy_derivative_blocks", _forbid_flatten)

    with pytest.raises(LinearMethodMemoryLimitError, match="1024"):
        m_step.update(_tangent_input(p_step, o_mat, ge_mat, e_loc, positions=positions))


# ---------------------------------------------------------------------------
# 11. HI authority grid: all 16 linear_method cells + invalid mappings
# ---------------------------------------------------------------------------


def test_hi_authority_grid_all_16_cells_and_invalid_mappings() -> None:
    """All 16 linear_method cells in optimizer_grid_authority.json map cleanly."""

    authority_path = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "atomistic"
        / "he-importance"
        / "optimizer_grid_authority.json"
    )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    lm_family = next(fam for fam in authority["families"] if fam["family"] == "linear_method")
    assert lm_family["declared_cells"] == 16
    assert set(lm_family["varied"].keys()) == set(LM_GRID_FACTOR_KEYS)

    shifts = lm_family["varied"]["derivative_block_shift"]
    radii = lm_family["varied"]["overlap_metric_trust_radius"]
    cells = [
        {"derivative_block_shift": shift, "overlap_metric_trust_radius": radius}
        for shift, radius in itertools.product(shifts, radii)
    ]
    assert len(cells) == 16

    param = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
    binding = ModelParameterBinding(parameters=(param,))
    for factors in cells:
        hyper = LMGridHyperparameters.from_mapping(factors)
        assert hyper.to_factors() == factors

        policy = lm_policy_from_factors(factors)
        assert policy.derivative_block_shift == float(factors["derivative_block_shift"])
        assert policy.overlap_metric_trust_radius == float(factors["overlap_metric_trust_radius"])
        assert policy.overlap_shift == 0.0
        assert policy.rank_cutoff == 1.0e-12
        assert policy.rescaling_xi == 0.5
        assert policy.step_scale == 1.0
        assert policy.max_update_norm is None
        assert policy.squared_update_norm_bound is None

        update_method = build_linear_method_update_from_factors(
            factors,
            model_parameters=binding,
            nonfinite_local_energy_policy="fail",
        )
        assert isinstance(update_method, LinearMethodUpdate)
        assert update_method.policy == policy
        assert update_method.declared_nonfinite_local_energy_policy() == "fail"
        assert float(update_method.optimizer.param_groups[0]["lr"]) == 1.0

    # Invalid factor mappings.
    with pytest.raises(TypeError, match="Mapping"):
        LMGridHyperparameters.from_mapping([("derivative_block_shift", 0.01)])
    with pytest.raises(ValueError, match="missing"):
        LMGridHyperparameters.from_mapping({"derivative_block_shift": 0.01})
    with pytest.raises(ValueError, match="extra"):
        LMGridHyperparameters.from_mapping(
            {
                "derivative_block_shift": 0.01,
                "overlap_metric_trust_radius": 0.1,
                "overlap_shift": 0.0,
            }
        )
    with pytest.raises(TypeError):
        LMGridHyperparameters.from_mapping(
            {"derivative_block_shift": True, "overlap_metric_trust_radius": 0.1}
        )
    with pytest.raises(TypeError):
        LMGridHyperparameters.from_mapping(
            {"derivative_block_shift": "0.01", "overlap_metric_trust_radius": 0.1}
        )
    for bad_val in (0.0, -1.0e-4, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            LMGridHyperparameters.from_mapping(
                {"derivative_block_shift": bad_val, "overlap_metric_trust_radius": 0.1}
            )
        with pytest.raises(ValueError):
            LMGridHyperparameters.from_mapping(
                {"derivative_block_shift": 0.01, "overlap_metric_trust_radius": bad_val}
            )


# ---------------------------------------------------------------------------
# 12. Real tiny TPENWaveFunction central finite-difference tangent parity,
#     smoke (PR #854 NB2) + TrainResume parity + lm-state-1 drift refusals
# ---------------------------------------------------------------------------


def test_real_tiny_tpen_hamiltonian_tangent_central_finite_difference_parity() -> None:
    """Real TPEN second-order autograd (O, g^E) matches central FD across all parameter groups."""

    cfg = _tiny_tpen_config()
    cfg.sampler.n_walkers = 4
    cfg.model.readout.trainable = True
    cfg.model.envelope.envelopes[0].trainable = True

    model, sampler, terms = _instantiate_tiny_tpen(cfg, seed=19)
    pair_parameters_by_name(model)
    walkers, _ = sampler.collect_samples(model)
    batch = walkers.make_batch()

    update_input, term_map = evaluate_hamiltonian_tangent_step(
        model=model,
        hamiltonian_terms=terms,
        batch=batch,
        chunk_size=2,
        return_terms=True,
    )
    assert term_map is not None
    assert "kinetic" in term_map

    named_slots = {
        name: (slot_idx, parameter)
        for slot_idx, (name, parameter) in enumerate(model.named_parameters())
    }
    # Deterministic subset covering all five live parameter groups plus one
    # structurally disconnected order-1 parameter.
    live_targets: tuple[tuple[str, str, int], ...] = (
        ("embedding", "embedding.order_mlps.1.layers.0.weight", 0),
        ("embedding", "embedding.order_mlps.2.layers.2.weight", 0),
        ("mixing", "stack.layers.0.mixing.weights.g15", 0),
        ("mixing", "stack.layers.0.mixing.weights.g20", 3),
        ("path_aggregation", "stack.layers.0.path_aggregation.weights.o2", 0),
        ("readout", "readout.channel_weights", 0),
        ("envelope", "envelope.envelopes.0.raw_coefficient", 0),
    )
    inactive_target = ("stack.layers.0.mixing.weights.g0", 0)
    assert inactive_target[0] in INACTIVE_PAIR_PARAMETERS

    eps = 1.0e-6

    checked_groups: set[str] = set()
    for group_name, param_name, flat_idx in live_targets:
        assert param_name in named_slots, f"missing expected live parameter {param_name!r}"
        assert param_name not in INACTIVE_PAIR_PARAMETERS
        slot_idx, parameter = named_slots[param_name]
        ag_o = update_input.parameter_scores.blocks[slot_idx].reshape(batch.batch_size, -1)[
            :, flat_idx
        ]
        ag_ge = update_input.local_energy_derivatives.blocks[slot_idx].reshape(
            batch.batch_size, -1
        )[:, flat_idx]
        assert torch.isfinite(ag_o).all()
        assert torch.isfinite(ag_ge).all()

        flat_view = parameter.data.reshape(-1)
        orig = float(flat_view[flat_idx].item())
        try:
            flat_view[flat_idx] = orig + eps
            res_plus = local_energy(terms, model, batch, return_terms=True)
            assert res_plus.wavefunction_output is not None
            logabs_plus = res_plus.wavefunction_output.logabs.detach()
            eloc_plus = res_plus.total.detach()

            flat_view[flat_idx] = orig - eps
            res_minus = local_energy(terms, model, batch, return_terms=True)
            assert res_minus.wavefunction_output is not None
            logabs_minus = res_minus.wavefunction_output.logabs.detach()
            eloc_minus = res_minus.total.detach()
        finally:
            flat_view[flat_idx] = orig

        fd_o = (logabs_plus - logabs_minus) / (2.0 * eps)
        fd_ge = (eloc_plus - eloc_minus) / (2.0 * eps)

        assert float(fd_o.abs().max().item()) > 1.0e-6, f"vacuous score FD for {param_name}"
        assert float(fd_ge.abs().max().item()) > 1.0e-6, f"vacuous E_L FD for {param_name}"
        torch.testing.assert_close(ag_o, fd_o, atol=1.0e-6, rtol=1.0e-5)
        torch.testing.assert_close(ag_ge, fd_ge, atol=1.0e-6, rtol=1.0e-5)
        checked_groups.add(group_name)

    assert checked_groups == {"embedding", "mixing", "path_aggregation", "readout", "envelope"}

    # Disconnected order-1 parameter must have exact zero autograd and FD tangents.
    inactive_name, inactive_idx = inactive_target
    slot_idx, inactive_param = named_slots[inactive_name]
    ag_o_zero = update_input.parameter_scores.blocks[slot_idx].reshape(batch.batch_size, -1)[
        :, inactive_idx
    ]
    ag_ge_zero = update_input.local_energy_derivatives.blocks[slot_idx].reshape(
        batch.batch_size, -1
    )[:, inactive_idx]
    assert torch.equal(ag_o_zero, torch.zeros_like(ag_o_zero))
    assert torch.equal(ag_ge_zero, torch.zeros_like(ag_ge_zero))

    flat_view = inactive_param.data.reshape(-1)
    orig = float(flat_view[inactive_idx].item())
    try:
        flat_view[inactive_idx] = orig + eps
        res_plus = local_energy(terms, model, batch, return_terms=True)
        assert res_plus.wavefunction_output is not None
        logabs_plus = res_plus.wavefunction_output.logabs.detach()
        eloc_plus = res_plus.total.detach()

        flat_view[inactive_idx] = orig - eps
        res_minus = local_energy(terms, model, batch, return_terms=True)
        assert res_minus.wavefunction_output is not None
        logabs_minus = res_minus.wavefunction_output.logabs.detach()
        eloc_minus = res_minus.total.detach()
    finally:
        flat_view[inactive_idx] = orig

    assert torch.equal(logabs_plus, logabs_minus)
    assert torch.equal(eloc_plus, eloc_minus)


def test_real_tiny_tpen_smoke_chunk_parity_disconnected_zero_and_train_resume(
    tmp_path: Path,
) -> None:
    """Tiny 2e TPEN (C=2, H=4, P=366) runs end-to-end with chunk parity and TrainResume parity."""

    cfg = _tiny_tpen_config()
    # At random tiny-TPEN initialization with N=12 walkers << P=366 parameters,
    # ||dH11||_F ~ 60 and w_max ~ 139, so c_diag=100.0 stabilizes
    # Sym(dH11) + c_diag * I > 0 into the guaranteed-descent regime while still
    # exercising the N < P (r <= 11) nullspace projection and staying distinct
    # from the lm-state-1 policy-drift probe (0.1) below.
    policy_chunked = LinearMethodPolicy(
        derivative_block_shift=100.0,
        overlap_shift=0.0,
        overlap_metric_trust_radius=0.1,
        tangent_chunk_size=4,
    )
    policy_full = LinearMethodPolicy(
        derivative_block_shift=100.0,
        overlap_shift=0.0,
        overlap_metric_trust_radius=0.1,
        tangent_chunk_size=None,
    )

    # 1. Single-chunk vs multi-chunk parity & disconnected parameter invariance (step 1).
    model_full, sampler_full, terms_full = _instantiate_tiny_tpen(cfg, seed=19)
    total_params = sum(p.numel() for p in model_full.parameters())
    assert 0 < total_params < 1000
    pair_parameters_by_name(model_full)

    before_named = {
        name: parameter.detach().clone()
        for name, parameter in model_full.named_parameters()
    }
    method_full = LinearMethodUpdate(
        torch.optim.SGD(model_full.parameters(), lr=1.0),
        model_parameters=ModelParameterBinding(parameters=tuple(model_full.parameters())),
        policy=policy_full,
        nonfinite_local_energy_policy="fail",
    )
    ctx_full = _ResumeContext(cfg, tmp_path / "full_chunk")
    VMCTrainer(
        max_steps=1,
        log_every_n_steps=1,
        return_terms=True,
        update_method=method_full,
        nonfinite_local_energy_policy="fail",
    ).fit(
        model=model_full,
        sampler=sampler_full,
        hamiltonian_terms=terms_full,
        optimizer=method_full.optimizer,
        context=ctx_full,
        emit=lambda **_: None,
    )

    model_chunk, sampler_chunk, terms_chunk = _instantiate_tiny_tpen(cfg, seed=19)
    method_chunk = LinearMethodUpdate(
        torch.optim.SGD(model_chunk.parameters(), lr=1.0),
        model_parameters=ModelParameterBinding(parameters=tuple(model_chunk.parameters())),
        policy=policy_chunked,
        nonfinite_local_energy_policy="fail",
    )
    ctx_chunk = _ResumeContext(cfg, tmp_path / "multi_chunk")
    VMCTrainer(
        max_steps=1,
        log_every_n_steps=1,
        return_terms=True,
        update_method=method_chunk,
        nonfinite_local_energy_policy="fail",
    ).fit(
        model=model_chunk,
        sampler=sampler_chunk,
        hamiltonian_terms=terms_chunk,
        optimizer=method_chunk.optimizer,
        context=ctx_chunk,
        emit=lambda **_: None,
    )

    assert method_full.last_telemetry is not None and method_full.last_telemetry.applied is True
    assert method_chunk.last_telemetry is not None and method_chunk.last_telemetry.applied is True
    assert method_chunk.last_telemetry.diagnostics is not None
    # N=12 walkers -> rank(S_param) <= N - 1 = 11 << P.
    assert 1 <= method_chunk.last_telemetry.diagnostics.overlap_retained_modes <= 11

    for p_full, p_chunk in zip(model_full.parameters(), model_chunk.parameters(), strict=True):
        torch.testing.assert_close(p_full.detach(), p_chunk.detach(), atol=1.0e-11, rtol=1.0e-11)

    # All 16 structurally disconnected order-1-output parameters must be BITWISE unchanged,
    # while live order-2 parameters must have moved non-vacuously.
    moved_live: list[str] = []
    for name, parameter in model_chunk.named_parameters():
        assert torch.isfinite(parameter).all()
        if name in INACTIVE_PAIR_PARAMETERS:
            assert torch.equal(parameter.detach(), before_named[name]), (
                f"disconnected parameter {name} must stay bitwise unchanged"
            )
        elif not torch.equal(parameter.detach(), before_named[name]):
            moved_live.append(name)
    assert moved_live, "live TPEN parameters must move on an applied LM step"

    train_metrics = [m for ns, m in ctx_chunk.records if ns == "train"][-1]
    assert train_metrics["lm_applied"] is True
    assert "lm_solve_selected_delta_eigenvalue" in train_metrics
    assert "energy_term_kinetic" in train_metrics
    _assert_csv_and_jsonl_roundtrip(
        tmp_path, name="tiny_tpen_train", namespace="train", metrics=train_metrics
    )

    # 2. Uninterrupted 2-step run vs TrainResume 1+1 split-run parity.
    ref_model, ref_sampler, ref_terms = _instantiate_tiny_tpen(cfg, seed=23)
    ref_method = LinearMethodUpdate(
        torch.optim.SGD(ref_model.parameters(), lr=1.0),
        model_parameters=ModelParameterBinding(parameters=tuple(ref_model.parameters())),
        policy=policy_chunked,
        nonfinite_local_energy_policy="fail",
    )
    ref_ctx = _ResumeContext(cfg, tmp_path / "ref_2step")
    ref_trainer = VMCTrainer(
        max_steps=2,
        log_every_n_steps=1,
        return_terms=True,
        update_method=ref_method,
        nonfinite_local_energy_policy="fail",
    )
    ref_trainer.fit(
        model=ref_model,
        sampler=ref_sampler,
        hamiltonian_terms=ref_terms,
        optimizer=ref_method.optimizer,
        context=ref_ctx,
        emit=lambda **_: None,
    )

    split_model, split_sampler, split_terms = _instantiate_tiny_tpen(cfg, seed=23)
    split_method = LinearMethodUpdate(
        torch.optim.SGD(split_model.parameters(), lr=1.0),
        model_parameters=ModelParameterBinding(parameters=tuple(split_model.parameters())),
        policy=policy_chunked,
        nonfinite_local_energy_policy="fail",
    )
    split_ctx = _ResumeContext(cfg, tmp_path / "split_part1")
    split_trainer = VMCTrainer(
        max_steps=1,
        log_every_n_steps=1,
        return_terms=True,
        update_method=split_method,
        nonfinite_local_energy_policy="fail",
    )
    split_trainer.fit(
        model=split_model,
        sampler=split_sampler,
        hamiltonian_terms=split_terms,
        optimizer=split_method.optimizer,
        context=split_ctx,
        emit=lambda **_: None,
    )

    ckpt_dir = save_checkpoint(
        output_dir=tmp_path / "lm_ckpts",
        next_iteration=split_trainer.next_iteration,
        completed_updates=split_trainer.completed_updates,
        model=split_model,
        context=split_ctx,
        optimizer=split_method.optimizer,
        trainer=split_trainer,
        sampler=split_sampler,
    )

    # Perturb global RNG before restore to prove state came from the checkpoint.
    torch.manual_seed(9999)
    res_model, res_sampler, res_terms = _instantiate_tiny_tpen(cfg, seed=99)
    res_method = LinearMethodUpdate(
        torch.optim.SGD(res_model.parameters(), lr=1.0),
        model_parameters=ModelParameterBinding(parameters=tuple(res_model.parameters())),
        policy=policy_chunked,
        nonfinite_local_energy_policy="fail",
    )
    res_ctx = _ResumeContext(cfg, tmp_path / "split_part2")
    res_trainer = VMCTrainer(
        max_steps=2,
        log_every_n_steps=1,
        return_terms=True,
        update_method=res_method,
        nonfinite_local_energy_policy="fail",
    )
    res_trainer.resolve_update_state(model=res_model, optimizer=res_method.optimizer)
    report = restore_checkpoint(
        load={"mode": "train_resume", "path": str(ckpt_dir)},
        model=res_model,
        context=res_ctx,
        optimizer=res_method.optimizer,
        trainer=res_trainer,
        sampler=res_sampler,
    )
    assert report.loaded_trainer is True
    assert report.next_iteration == 1
    assert report.completed_updates == 1
    assert res_method.completed_updates == 1

    res_trainer.fit(
        model=res_model,
        sampler=res_sampler,
        hamiltonian_terms=res_terms,
        optimizer=res_method.optimizer,
        context=res_ctx,
        emit=lambda **_: None,
    )
    assert res_trainer.next_iteration == ref_trainer.next_iteration == 2
    assert res_trainer.completed_updates == ref_trainer.completed_updates == 2
    assert res_method.completed_updates == ref_method.completed_updates == 2
    for p_ref, p_res in zip(ref_model.parameters(), res_model.parameters(), strict=True):
        torch.testing.assert_close(p_ref.detach(), p_res.detach(), atol=1.0e-12, rtol=1.0e-12)

    # 3. Strict lm-state-1 fail-closed drift refusals (atomically preserving completed_updates).
    valid_state = json.loads(json.dumps(res_method.method_state_dict()))
    assert valid_state["version"] == LM_STATE_VERSION == "lm-state-1"
    assert valid_state["completed_updates"] == 2

    with pytest.raises((TypeError, TypeCheckError)):
        res_method.load_method_state_dict(["not-a-mapping"])  # type: ignore[arg-type]
    with typeguard.suppress_type_checks(), pytest.raises(TypeError, match="mapping"):
        res_method.load_method_state_dict(["not-a-mapping"])  # type: ignore[arg-type]

    drift_cases: list[tuple[dict[str, Any], str]] = [
        ({**valid_state, "version": "lm-state-0"}, "version"),
        (
            {**valid_state, "fingerprint": {**valid_state["fingerprint"], "digest": "bad"}},
            "fingerprint",
        ),
        (
            {
                **valid_state,
                "policy": {**valid_state["policy"], "derivative_block_shift": 0.1},
            },
            "policy does not match",
        ),
        (
            {**valid_state, "nonfinite_local_energy_policy": "mask"},
            "nonfinite_local_energy_policy",
        ),
        (
            {**valid_state, "nonfinite_tangent_row_policy": "drop_quietly"},
            "nonfinite_tangent_row_policy",
        ),
        ({k: v for k, v in valid_state.items() if k != "completed_updates"}, "completed_updates"),
        ({**valid_state, "completed_updates": -1}, "completed_updates"),
        ({**valid_state, "completed_updates": True}, "completed_updates"),
    ]
    for mutated_state, pattern in drift_cases:
        with pytest.raises(ValueError, match=pattern):
            res_method.load_method_state_dict(copy.deepcopy(mutated_state))
        assert res_method.completed_updates == 2


# ---------------------------------------------------------------------------
# 13. Review Round-1 Fix Regressions (F1--F4)
# ---------------------------------------------------------------------------


class _TwoParamOracleAHarmonicWavefunction(torch.nn.Module):
    """Two-parameter variant of Oracle A for partial-subset SGD carrier probes."""

    def __init__(self, alpha: float = 0.35, beta: float = 0.40) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.float64))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.float64))
        self._binding = ParameterBinding(
            layout=ParameterLayout(
                slots=(
                    ParameterSlot(ordinal=0, shape=(1,), numel=1, dtype=torch.float64),
                    ParameterSlot(ordinal=1, shape=(1,), numel=1, dtype=torch.float64),
                )
            ),
            parameters=(self.alpha, self.beta),
        )

    @property
    def parameter_binding(self) -> ParameterBinding:
        return self._binding

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        r = flat.positions[:, 0, :]
        r2 = (r * r).sum(dim=-1)
        x = r[:, 0]
        logabs = -self.alpha[0] * r2 + self.beta[0] * x
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _FixedCheckpointedSampler(_FixedSampler):
    """Minimal fixed sampler supporting TrainResume checkpoint save and restore."""

    def mcmc_state_dict(self) -> dict[str, Any]:
        return {"step": 0}

    def load_mcmc_state_dict(self, state: dict[str, Any], *, device: Any = None) -> None:
        del state, device


def test_f1_plain_sgd_carrier_validation_and_runtime_resume_revalidation(
    tmp_path: Path,
) -> None:
    """Plain-SGD validation enforces exact coverage, non-bool lr, and runtime/resume atomicity."""

    positions, e_loc, o_mat, ge_mat = _oracle_a_arrays()
    policy = LinearMethodPolicy(derivative_block_shift=0.01, overlap_shift=0.0, step_scale=1.0)

    # 1. Constructor guards: empty param_groups, empty params, bool/None/non-finite lr,
    #    strict-subset carrier, foreign/extra carrier, duplicate parameter, multi-group OK.
    p1 = torch.nn.Parameter(torch.tensor([0.35], dtype=torch.float64))
    p2 = torch.nn.Parameter(torch.tensor([0.40], dtype=torch.float64))
    p_extra = torch.nn.Parameter(torch.tensor([0.10], dtype=torch.float64))
    binding_12 = ModelParameterBinding(parameters=(p1, p2))

    opt_empty_groups = torch.optim.SGD([p1, p2], lr=1.0)
    opt_empty_groups.param_groups.clear()
    with pytest.raises(ValueError, match="non-empty"):
        LinearMethodUpdate(opt_empty_groups, model_parameters=binding_12, policy=policy)

    opt_empty_params = torch.optim.SGD([p1, p2], lr=1.0)
    opt_empty_params.param_groups[0]["params"].clear()
    with pytest.raises(ValueError, match="empty"):
        LinearMethodUpdate(opt_empty_params, model_parameters=binding_12, policy=policy)

    for bool_lr in (True, False):
        with pytest.raises(TypeError, match="real number"):
            LinearMethodUpdate(
                torch.optim.SGD([p1, p2], lr=bool_lr),  # type: ignore[arg-type]
                model_parameters=binding_12,
                policy=policy,
            )

    opt_none_lr = torch.optim.SGD([p1, p2], lr=1.0)
    opt_none_lr.param_groups[0]["lr"] = None
    with pytest.raises(TypeError, match="real number"):
        LinearMethodUpdate(opt_none_lr, model_parameters=binding_12, policy=policy)

    for bad_lr in (float("nan"), float("inf")):
        opt_nonfinite_lr = torch.optim.SGD([p1, p2], lr=1.0)
        opt_nonfinite_lr.param_groups[0]["lr"] = bad_lr
        with pytest.raises(ValueError, match="finite"):
            LinearMethodUpdate(opt_nonfinite_lr, model_parameters=binding_12, policy=policy)

    # Strict-subset SGD (missing p2) refused loudly at construction.
    with pytest.raises(ValueError, match="partial parameter-subset"):
        LinearMethodUpdate(
            torch.optim.SGD([p1], lr=1.0),
            model_parameters=binding_12,
            policy=policy,
        )

    # Extra/foreign parameter SGD refused loudly at construction.
    with pytest.raises(ValueError, match="foreign"):
        LinearMethodUpdate(
            torch.optim.SGD([p1, p2, p_extra], lr=1.0),
            model_parameters=binding_12,
            policy=policy,
        )
    with pytest.raises(ValueError, match="foreign"):
        LinearMethodUpdate(
            torch.optim.SGD([p_extra], lr=1.0),
            model_parameters=ModelParameterBinding(parameters=(p1,)),
            policy=policy,
        )

    # Duplicate parameter across groups or inside a group refused loudly.
    opt_dup = torch.optim.SGD([p1, p2], lr=1.0)
    opt_dup.param_groups[0]["params"].append(p1)
    with pytest.raises(ValueError, match="duplicate"):
        LinearMethodUpdate(opt_dup, model_parameters=binding_12, policy=policy)

    # Positive control: multi-group plain SGD with exact disjoint coverage succeeds.
    opt_multi = torch.optim.SGD([{"params": [p1], "lr": 1.0}, {"params": [p2], "lr": 1.0}])
    method_multi = LinearMethodUpdate(opt_multi, model_parameters=binding_12, policy=policy)
    assert method_multi.completed_updates == 0

    # 2. Post-init param_groups drift before update(): atomic zero-mutation refusal.
    drift_mutations: list[tuple[Any, type[Exception], str]] = [
        (lambda opt, _p: opt.param_groups[0].__setitem__("lr", 0.5), ValueError, "step_scale"),
        (lambda opt, _p: opt.param_groups[0].__setitem__("lr", True), TypeError, "real number"),
        (lambda opt, _p: opt.param_groups[0].__setitem__("lr", None), TypeError, "real number"),
        (
            lambda opt, _p: opt.param_groups[0].__setitem__("lr", float("nan")),
            ValueError,
            "finite",
        ),
        (
            lambda opt, _p: opt.param_groups[0].__setitem__("momentum", 0.9),
            ValueError,
            "momentum",
        ),
        (
            lambda opt, _p: opt.param_groups[0].__setitem__("dampening", 0.1),
            ValueError,
            "dampening",
        ),
        (
            lambda opt, _p: opt.param_groups[0].__setitem__("weight_decay", 1.0e-4),
            ValueError,
            "weight_decay",
        ),
        (
            lambda opt, _p: opt.param_groups[0].__setitem__("nesterov", True),
            ValueError,
            "nesterov",
        ),
        (
            lambda opt, _p: opt.param_groups[0].__setitem__("maximize", True),
            ValueError,
            "maximize",
        ),
        (lambda opt, _p: opt.param_groups.clear(), ValueError, "non-empty"),
        (
            lambda opt, _p: opt.param_groups[0].__setitem__("params", [p_extra]),
            ValueError,
            "foreign",
        ),
        (
            lambda opt, p: opt.param_groups[0]["params"].append(p),
            ValueError,
            "duplicate",
        ),
    ]
    for mutate, expected_exc, pattern in drift_mutations:
        param = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
        method = _lm_method(param, policy=policy)
        # Apply one clean step first so completed_updates and _last_result are non-trivial.
        first_res = method.update(_tangent_input(param, o_mat, ge_mat, e_loc, positions=positions))
        assert first_res.applied is True
        assert method.completed_updates == 1
        sentinel_grad = torch.tensor([11.0, -13.0], dtype=torch.float64)
        param.grad = sentinel_grad.clone()
        before_param = param.detach().clone()
        before_last_result = method._last_result

        mutate(method.optimizer, param)
        with pytest.raises(expected_exc, match=pattern):
            method.update(_tangent_input(param, o_mat, ge_mat, e_loc, positions=positions))

        assert torch.equal(param.detach(), before_param)
        assert param.grad is not None and torch.equal(param.grad, sentinel_grad)
        assert method.completed_updates == 1
        assert method._last_result is before_last_result

    # 3. Post-init strict-subset mutation routed through VMCTrainer.fit() fails closed
    #    even though VMCTrainer's validate_optimizer_parameter_containment admits subsets.
    two_param_model = _TwoParamOracleAHarmonicWavefunction(alpha=0.35, beta=0.40)
    with pytest.raises(ValueError, match="partial parameter-subset"):
        LinearMethodUpdate(
            torch.optim.SGD([two_param_model.alpha], lr=1.0),
            model_parameters=ModelParameterBinding(parameters=tuple(two_param_model.parameters())),
            policy=policy,
        )

    subset_method = LinearMethodUpdate(
        torch.optim.SGD(two_param_model.parameters(), lr=1.0),
        model_parameters=ModelParameterBinding(parameters=tuple(two_param_model.parameters())),
        policy=policy,
    )
    subset_method.optimizer.param_groups[0]["params"] = [two_param_model.alpha]
    before_alpha = two_param_model.alpha.detach().clone()
    before_beta = two_param_model.beta.detach().clone()
    batch = ElectronBatch(
        positions=positions,
        spins=torch.ones((12, 1), dtype=torch.float64),
    )
    subset_trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=subset_method)
    with pytest.raises(ValueError, match="partial parameter-subset"):
        subset_trainer.fit(
            model=two_param_model,
            sampler=_FixedSampler(batch),
            hamiltonian_terms=[KineticEnergy(), HarmonicTrap(omega=1.0)],
            optimizer=subset_method.optimizer,
            context=_StubContext(),
            emit=lambda **_: None,
        )
    assert torch.equal(two_param_model.alpha.detach(), before_alpha)
    assert torch.equal(two_param_model.beta.detach(), before_beta)
    assert two_param_model.alpha.grad is None
    assert two_param_model.beta.grad is None
    assert subset_method.completed_updates == 0
    assert subset_method._last_result is None
    assert subset_trainer.completed_updates == 0

    # 4. Direct load_method_state_dict() fails closed atomically on drifted optimizer.
    for mutate, expected_exc, pattern in (
        (lambda opt, _p: opt.param_groups[0].__setitem__("lr", 0.5), ValueError, "step_scale"),
        (lambda opt, _p: opt.param_groups[0].__setitem__("lr", True), TypeError, "real number"),
        (
            lambda opt, _p: opt.param_groups[0].__setitem__("momentum", 0.9),
            ValueError,
            "momentum",
        ),
        (
            lambda opt, _p: opt.param_groups[0].__setitem__("weight_decay", 1.0e-3),
            ValueError,
            "weight_decay",
        ),
        (
            lambda opt, _p: opt.param_groups[0].__setitem__("maximize", True),
            ValueError,
            "maximize",
        ),
        (lambda opt, _p: opt.param_groups.clear(), ValueError, "non-empty"),
    ):
        param_load = torch.nn.Parameter(torch.tensor([0.35, 0.40], dtype=torch.float64))
        method_load = _lm_method(param_load, policy=policy)
        method_load.update(_tangent_input(param_load, o_mat, ge_mat, e_loc, positions=positions))
        assert method_load.completed_updates == 1
        target_state = {**method_load.method_state_dict(), "completed_updates": 7}
        sentinel_grad = torch.tensor([5.0, -6.0], dtype=torch.float64)
        param_load.grad = sentinel_grad.clone()
        before_param = param_load.detach().clone()
        before_last_result = method_load._last_result

        mutate(method_load.optimizer, param_load)
        with pytest.raises(expected_exc, match=pattern):
            method_load.load_method_state_dict(target_state)

        assert method_load.completed_updates == 1
        assert torch.equal(param_load.detach(), before_param)
        assert param_load.grad is not None and torch.equal(param_load.grad, sentinel_grad)
        assert method_load._last_result is before_last_result

    # 5. Real TrainResume checkpoint restore refuses drifted optimizer.pt atomically
    #    inside load_method_state_dict() before mutating method/trainer counters.
    ho_cfg = OmegaConf.create(
        {
            "model": {"name": "oracle_a_ho"},
            "sampler": {"name": "fixed"},
            "hamiltonian_terms": {"kinetic": True, "harmonic_trap": 1.0},
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 1.0},
            "trainer": {"max_steps": 2, "log_every_n_steps": 1},
        }
    )
    terms = [KineticEnergy(), HarmonicTrap(omega=1.0)]
    for label, drift_key, drift_val, pattern in (
        ("lr_drift", "lr", 0.5, "step_scale"),
        ("momentum_drift", "momentum", 0.9, "momentum"),
        ("maximize_drift", "maximize", True, "maximize"),
    ):
        src_model = _OracleAHarmonicWavefunction(alpha=0.35, beta=0.40)
        src_method = LinearMethodUpdate(
            torch.optim.SGD(src_model.parameters(), lr=1.0),
            model_parameters=ModelParameterBinding(parameters=tuple(src_model.parameters())),
            policy=policy,
        )
        src_sampler = _FixedCheckpointedSampler(batch)
        src_ctx = _ResumeContext(ho_cfg, tmp_path / f"{label}_src")
        src_trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=src_method)
        src_trainer.fit(
            model=src_model,
            sampler=src_sampler,
            hamiltonian_terms=terms,
            optimizer=src_method.optimizer,
            context=src_ctx,
            emit=lambda **_: None,
        )
        assert src_method.completed_updates == 1

        # Simulate scheduler/carrier mutation persisted into optimizer.pt with valid manifest hashes.
        src_method.optimizer.param_groups[0][drift_key] = drift_val
        ckpt_dir = save_checkpoint(
            output_dir=tmp_path / f"{label}_ckpts",
            next_iteration=src_trainer.next_iteration,
            completed_updates=src_trainer.completed_updates,
            model=src_model,
            context=src_ctx,
            optimizer=src_method.optimizer,
            trainer=src_trainer,
            sampler=src_sampler,
        )

        res_model = _OracleAHarmonicWavefunction(alpha=0.35, beta=0.40)
        res_method = LinearMethodUpdate(
            torch.optim.SGD(res_model.parameters(), lr=1.0),
            model_parameters=ModelParameterBinding(parameters=tuple(res_model.parameters())),
            policy=policy,
        )
        res_sampler = _FixedCheckpointedSampler(batch)
        res_ctx = _ResumeContext(ho_cfg, tmp_path / f"{label}_res")
        res_trainer = VMCTrainer(max_steps=2, log_every_n_steps=1, update_method=res_method)
        res_trainer.resolve_update_state(model=res_model, optimizer=res_method.optimizer)

        with pytest.raises(ValueError, match=pattern):
            restore_checkpoint(
                load={"mode": "train_resume", "path": str(ckpt_dir)},
                model=res_model,
                context=res_ctx,
                optimizer=res_method.optimizer,
                trainer=res_trainer,
                sampler=res_sampler,
            )
        assert res_method.completed_updates == 0
        assert res_method._last_result is None
        assert res_trainer.completed_updates == 0
        assert res_trainer.next_iteration == 0


class _Pass1Done(Exception):
    """Sentinel stopping Stage-1 capture immediately after Pass-1 reduction."""


class _Pass1CaptureReducer(StatisticsReducer):
    """Capture one rank's Pass-1 local count and sums emitted by assembly."""

    def __init__(self) -> None:
        self.count: int | None = None
        self.sums: list[torch.Tensor] = []

    def reduce_count(self, count: int) -> int:
        self.count = count
        return count

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        self.sums.append(value.detach().clone())
        if len(self.sums) == 3:
            raise _Pass1Done()
        return value


class _Pass2CaptureReducer(StatisticsReducer):
    """Replay global Pass-1 sums and capture one rank's Pass-2 local sums."""

    def __init__(
        self,
        *,
        global_count: int,
        global_first: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        self._global_count = global_count
        self._global_first = global_first
        self._sum_calls = 0
        self.pass2_sums: list[torch.Tensor] = []

    def reduce_count(self, count: int) -> int:
        del count
        return self._global_count

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        idx = self._sum_calls
        self._sum_calls += 1
        if idx < 3:
            return self._global_first[idx].clone()
        self.pass2_sums.append(value.detach().clone())
        return value


class _ReplayAllReduceReducer(StatisticsReducer):
    """Replay exact all-reduced Pass-1 and Pass-2 sums and assert reducer call protocol."""

    def __init__(
        self,
        *,
        expected_count: int,
        expected_locals: tuple[torch.Tensor, ...],
        global_count: int,
        global_sums: tuple[torch.Tensor, ...],
    ) -> None:
        self._expected_count = expected_count
        self._expected_locals = expected_locals
        self._global_count = global_count
        self._global_sums = global_sums
        self.count_calls = 0
        self.sum_calls = 0

    def reduce_count(self, count: int) -> int:
        assert count == self._expected_count
        self.count_calls += 1
        return self._global_count

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        idx = self.sum_calls
        assert idx < len(self._global_sums)
        torch.testing.assert_close(value, self._expected_locals[idx], atol=0.0, rtol=0.0)
        self.sum_calls += 1
        return self._global_sums[idx].clone()


class _AsymmetricReducer(StatisticsReducer):
    """Doubles count and triples sums so reduced mean is 1.5x local mean."""

    def reduce_count(self, count: int) -> int:
        return 2 * count

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        return 3.0 * value


def _simulate_multirank_assembly(
    shards: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    solve_dtype: torch.dtype = torch.float64,
) -> tuple[list[DenseLinearMethodMatrices], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Deterministically execute two-pass multi-rank reduction across rank shards."""

    pass1_counts: list[int] = []
    pass1_locals: list[list[torch.Tensor]] = []
    for e_r, o_r, ge_r in shards:
        cap1 = _Pass1CaptureReducer()
        try:
            assemble_dense_linear_method_matrices(
                e_r, o_r, ge_r, solve_dtype=solve_dtype, reducer=cap1
            )
        except _Pass1Done:
            pass
        assert cap1.count == int(e_r.shape[0])
        assert len(cap1.sums) == 3
        pass1_counts.append(cap1.count)
        pass1_locals.append(cap1.sums)

    global_count = sum(pass1_counts)
    global_first = (
        sum((sums[0] for sums in pass1_locals[1:]), pass1_locals[0][0].clone()),
        sum((sums[1] for sums in pass1_locals[1:]), pass1_locals[0][1].clone()),
        sum((sums[2] for sums in pass1_locals[1:]), pass1_locals[0][2].clone()),
    )

    pass2_locals: list[list[torch.Tensor]] = []
    for e_r, o_r, ge_r in shards:
        cap2 = _Pass2CaptureReducer(global_count=global_count, global_first=global_first)
        assemble_dense_linear_method_matrices(
            e_r, o_r, ge_r, solve_dtype=solve_dtype, reducer=cap2
        )
        assert len(cap2.pass2_sums) == 3
        pass2_locals.append(cap2.pass2_sums)

    global_second = (
        sum((sums[0] for sums in pass2_locals[1:]), pass2_locals[0][0].clone()),
        sum((sums[1] for sums in pass2_locals[1:]), pass2_locals[0][1].clone()),
        sum((sums[2] for sums in pass2_locals[1:]), pass2_locals[0][2].clone()),
    )
    global_all = global_first + global_second

    rank_outputs: list[DenseLinearMethodMatrices] = []
    for rank_idx, (e_r, o_r, ge_r) in enumerate(shards):
        replay = _ReplayAllReduceReducer(
            expected_count=pass1_counts[rank_idx],
            expected_locals=tuple(pass1_locals[rank_idx]) + tuple(pass2_locals[rank_idx]),
            global_count=global_count,
            global_sums=global_all,
        )
        mats = assemble_dense_linear_method_matrices(
            e_r, o_r, ge_r, solve_dtype=solve_dtype, reducer=replay
        )
        assert replay.count_calls == 1
        assert replay.sum_calls == 6
        rank_outputs.append(mats)

    return rank_outputs, pass1_locals, pass2_locals


def test_f2_multirank_statistics_reducer_splits_empty_rank_and_asymmetric_centering() -> None:
    """Two-pass StatisticsReducer assembly matches single-rank on 5+7, 5+0+7, and skewed means."""

    _, e_loc, o_mat, ge_mat = _oracle_a_arrays()
    ref_mats = assemble_dense_linear_method_matrices(
        e_loc, o_mat, ge_mat, solve_dtype=torch.float64
    )
    policy = LinearMethodPolicy(derivative_block_shift=0.01, overlap_shift=0.05, rescaling_xi=0.5)
    ref_out = solve_dense_linear_method(ref_mats, policy=policy)
    assert ref_out.accepted and ref_out.delta_theta is not None

    def _assert_matches_ref(mats: DenseLinearMethodMatrices) -> None:
        assert mats.count == ref_mats.count == 12
        assert mats.dtype == ref_mats.dtype == "torch.float64"
        assert mats.energy_mean == pytest.approx(ref_mats.energy_mean, abs=1.0e-12, rel=1.0e-12)
        assert mats.score_mean_norm_sq == pytest.approx(
            ref_mats.score_mean_norm_sq, abs=1.0e-12, rel=1.0e-12
        )
        torch.testing.assert_close(mats.S_param, ref_mats.S_param, atol=1.0e-12, rtol=1.0e-12)
        torch.testing.assert_close(mats.h_c, ref_mats.h_c, atol=1.0e-12, rtol=1.0e-12)
        torch.testing.assert_close(mats.h_r, ref_mats.h_r, atol=1.0e-12, rtol=1.0e-12)
        torch.testing.assert_close(mats.dH11, ref_mats.dH11, atol=1.0e-12, rtol=1.0e-12)
        out = solve_dense_linear_method(mats, policy=policy)
        assert out.accepted and out.delta_theta is not None
        assert out.diagnostics.selected_delta_eigenvalue == pytest.approx(
            ref_out.diagnostics.selected_delta_eigenvalue, abs=1.0e-11, rel=1.0e-11
        )
        assert out.diagnostics.selected_eigenvalue == pytest.approx(
            ref_out.diagnostics.selected_eigenvalue, abs=1.0e-11, rel=1.0e-11
        )
        assert out.diagnostics.selected_c0 == pytest.approx(
            ref_out.diagnostics.selected_c0, abs=1.0e-11, rel=1.0e-11
        )
        assert out.diagnostics.directional_derivative == pytest.approx(
            ref_out.diagnostics.directional_derivative, abs=1.0e-11, rel=1.0e-11
        )
        torch.testing.assert_close(out.delta_theta, ref_out.delta_theta, atol=1.0e-11, rtol=1.0e-11)

    # (a) Uneven 5 + 7 multi-rank split.
    shards_5_7 = [
        (e_loc[:5], o_mat[:5], ge_mat[:5]),
        (e_loc[5:], o_mat[5:], ge_mat[5:]),
    ]
    outputs_5_7, _, _ = _simulate_multirank_assembly(shards_5_7)
    for mats in outputs_5_7:
        _assert_matches_ref(mats)

    # (b) 5 + 0 + 7 multi-rank split exercising n_local == 0 Pass-1 and Pass-2 zero branches.
    shards_5_0_7 = [
        (e_loc[:5], o_mat[:5], ge_mat[:5]),
        (
            torch.empty((0,), dtype=torch.float64),
            torch.empty((0, 2), dtype=torch.float64),
            torch.empty((0, 2), dtype=torch.float64),
        ),
        (e_loc[5:], o_mat[5:], ge_mat[5:]),
    ]
    outputs_5_0_7, pass1_5_0_7, pass2_5_0_7 = _simulate_multirank_assembly(shards_5_0_7)
    # Directly verify empty rank emitted exact zero tensors of exact shapes.
    assert pass1_5_0_7[1][0].shape == () and torch.equal(
        pass1_5_0_7[1][0], torch.zeros((), dtype=torch.float64)
    )
    assert pass1_5_0_7[1][1].shape == (2,) and torch.equal(
        pass1_5_0_7[1][1], torch.zeros((2,), dtype=torch.float64)
    )
    assert pass1_5_0_7[1][2].shape == (2,) and torch.equal(
        pass1_5_0_7[1][2], torch.zeros((2,), dtype=torch.float64)
    )
    assert pass2_5_0_7[1][0].shape == (2,) and torch.equal(
        pass2_5_0_7[1][0], torch.zeros((2,), dtype=torch.float64)
    )
    assert pass2_5_0_7[1][1].shape == (2, 2) and torch.equal(
        pass2_5_0_7[1][1], torch.zeros((2, 2), dtype=torch.float64)
    )
    assert pass2_5_0_7[1][2].shape == (2, 2) and torch.equal(
        pass2_5_0_7[1][2], torch.zeros((2, 2), dtype=torch.float64)
    )
    for mats in outputs_5_0_7:
        _assert_matches_ref(mats)

    # (c.1) Strongly skewed rank-0 vs rank-1 means prove centering uses global reduced totals.
    e_skew0 = e_loc[:5] + 15.0
    o_skew0 = o_mat[:5] + torch.tensor([8.0, -6.0], dtype=torch.float64)
    ge_skew0 = ge_mat[:5] + torch.tensor([1.5, -2.0], dtype=torch.float64)
    e_skew1 = e_loc[5:] - 11.0
    o_skew1 = o_mat[5:] + torch.tensor([-7.0, 5.0], dtype=torch.float64)
    ge_skew1 = ge_mat[5:] + torch.tensor([-1.0, 3.0], dtype=torch.float64)
    assert abs(float((e_skew0.mean() - e_skew1.mean()).item())) > 20.0
    assert float(torch.linalg.vector_norm(o_skew0.mean(dim=0) - o_skew1.mean(dim=0)).item()) > 15.0

    e_skew_all = torch.cat((e_skew0, e_skew1), dim=0)
    o_skew_all = torch.cat((o_skew0, o_skew1), dim=0)
    ge_skew_all = torch.cat((ge_skew0, ge_skew1), dim=0)
    skew_ref = assemble_dense_linear_method_matrices(
        e_skew_all, o_skew_all, ge_skew_all, solve_dtype=torch.float64
    )
    skew_ranks, _, _ = _simulate_multirank_assembly(
        [(e_skew0, o_skew0, ge_skew0), (e_skew1, o_skew1, ge_skew1)]
    )
    for mats in skew_ranks:
        assert mats.count == 12
        assert mats.energy_mean == pytest.approx(skew_ref.energy_mean, abs=1.0e-12, rel=1.0e-12)
        assert mats.score_mean_norm_sq == pytest.approx(
            skew_ref.score_mean_norm_sq, abs=1.0e-12, rel=1.0e-12
        )
        torch.testing.assert_close(mats.S_param, skew_ref.S_param, atol=1.0e-12, rtol=1.0e-12)
        torch.testing.assert_close(mats.h_c, skew_ref.h_c, atol=1.0e-12, rtol=1.0e-12)
        torch.testing.assert_close(mats.h_r, skew_ref.h_r, atol=1.0e-12, rtol=1.0e-12)
        torch.testing.assert_close(mats.dH11, skew_ref.dH11, atol=1.0e-12, rtol=1.0e-12)

    # Naive rank-local centering disagrees sharply (> 10.0 in Frobenius norm).
    local_s_wrong = (
        (o_skew0 - o_skew0.mean(dim=0)).transpose(0, 1) @ (o_skew0 - o_skew0.mean(dim=0))
        + (o_skew1 - o_skew1.mean(dim=0)).transpose(0, 1) @ (o_skew1 - o_skew1.mean(dim=0))
    ) / 12.0
    assert float(torch.linalg.matrix_norm(local_s_wrong - skew_ref.S_param).item()) > 10.0

    # (c.2) Asymmetric reducer probe (count * 2, sum * 3 => reduced mean = 1.5x local mean).
    asym_mats = assemble_dense_linear_method_matrices(
        e_loc, o_mat, ge_mat, solve_dtype=torch.float64, reducer=_AsymmetricReducer()
    )
    n_reduced = 2 * int(e_loc.shape[0])
    total_e = 3.0 * e_loc.sum()
    total_o = 3.0 * o_mat.sum(dim=0)
    total_ge = 3.0 * ge_mat.sum(dim=0)
    exp_e_mean = float((total_e / n_reduced).item())
    exp_o_mean = total_o / n_reduced
    exp_ge_mean = total_ge / n_reduced
    exp_score_mean_sq = float((exp_o_mean * exp_o_mean).sum().item())
    e_bar_asym = e_loc - total_e / n_reduced
    o_bar_asym = o_mat - total_o / n_reduced
    h_tan_asym = e_bar_asym.unsqueeze(1) * o_bar_asym + ge_mat
    exp_hc = (3.0 * (o_bar_asym.transpose(0, 1) @ e_bar_asym)) / n_reduced
    exp_hr = exp_hc + exp_ge_mean
    exp_s_raw = (3.0 * (o_bar_asym.transpose(0, 1) @ o_bar_asym)) / n_reduced
    exp_s = 0.5 * (exp_s_raw + exp_s_raw.transpose(0, 1))
    exp_dh11 = (3.0 * (o_bar_asym.transpose(0, 1) @ h_tan_asym)) / n_reduced

    assert asym_mats.count == n_reduced
    assert asym_mats.energy_mean == pytest.approx(exp_e_mean, abs=1.0e-12, rel=1.0e-12)
    assert asym_mats.score_mean_norm_sq == pytest.approx(
        exp_score_mean_sq, abs=1.0e-12, rel=1.0e-12
    )
    torch.testing.assert_close(asym_mats.S_param, exp_s, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(asym_mats.h_c, exp_hc, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(asym_mats.h_r, exp_hr, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(asym_mats.dH11, exp_dh11, atol=1.0e-12, rtol=1.0e-12)
    # Must strictly disagree with local-mean centering.
    local_o_bar = o_mat - o_mat.mean(dim=0)
    local_s_only = (3.0 * (local_o_bar.transpose(0, 1) @ local_o_bar)) / n_reduced
    assert float(torch.linalg.matrix_norm(asym_mats.S_param - local_s_only).item()) > 1.0


def test_f3_pre_eig_finiteness_guard_float32_overflow_atomic_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Whitened/shifted M overflow in float32 skips atomically before entering torch.linalg.eig."""

    def _forbid_eig(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("torch.linalg.eig must not be entered when M is non-finite")

    # 1. Direct solve_dense_linear_method float32 overflow: 100.0 / 1e-37 -> +inf in M.
    ovf_matrices = DenseLinearMethodMatrices(
        count=4,
        energy_mean=1.5,
        S_param=torch.tensor([[1.0e-37]], dtype=torch.float32),
        h_c=torch.tensor([-1.5811388e-19], dtype=torch.float32),
        h_r=torch.tensor([-1.5811388e-19], dtype=torch.float32),
        dH11=torch.tensor([[0.0]], dtype=torch.float32),
        dtype="torch.float32",
    )
    ovf_policy = LinearMethodPolicy(
        derivative_block_shift=100.0,
        overlap_shift=0.0,
        rescaling_xi=0.5,
    )
    with monkeypatch.context() as mp:
        mp.setattr(torch.linalg, "eig", _forbid_eig)
        outcome = solve_dense_linear_method(ovf_matrices, policy=ovf_policy)

    assert outcome.accepted is False
    assert outcome.reason == "nonfinite_eigensolve"
    assert outcome.delta_theta is None
    assert outcome.update_direction_norm is None
    assert outcome.diagnostics.dtype == "torch.float32"
    assert outcome.diagnostics.overlap_retained_modes == 1
    assert outcome.diagnostics.overlap_truncated_modes == 0
    assert outcome.diagnostics.overlap_min_retained_eigenvalue == pytest.approx(
        1.0e-37, rel=1.0e-4
    )
    for field_name in _SELECTED_12_OPTIONAL_FIELDS:
        assert getattr(outcome.diagnostics, field_name) is None, f"{field_name} must be None"

    # 2. Full LinearMethodUpdate.update() in float32: zero mutation and JSONL/CSV-safe telemetry.
    param = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
    sentinel_grad = torch.tensor([3.25], dtype=torch.float32)
    param.grad = sentinel_grad.clone()
    before_param = param.detach().clone()
    method = _lm_method(
        param,
        policy=ovf_policy,
        conventions=ScoreConventions(solve_dtype=torch.float32),
    )
    o_rows = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]], dtype=torch.float32) * 3.1622777e-19
    ge_rows = torch.zeros((4, 1), dtype=torch.float32)
    e_rows = torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32)

    with monkeypatch.context() as mp:
        mp.setattr(torch.linalg, "eig", _forbid_eig)
        result = method.update(_tangent_input(param, o_rows, ge_rows, e_rows))

    assert result.applied is False
    assert result.reason == "nonfinite_eigensolve"
    assert torch.equal(param.detach(), before_param)
    assert param.grad is not None and torch.equal(param.grad, sentinel_grad)
    assert method.completed_updates == 0
    tel = method.last_telemetry
    assert tel is not None and tel.diagnostics is not None
    assert tel.diagnostics.dtype == "torch.float32"
    assert tel.diagnostics.overlap_retained_modes == 1
    assert tel.diagnostics.overlap_min_retained_eigenvalue == pytest.approx(1.0e-37, rel=1.0e-4)
    for field_name in _SELECTED_12_OPTIONAL_FIELDS:
        assert getattr(tel.diagnostics, field_name) is None, f"{field_name} must be None"
    _assert_csv_and_jsonl_roundtrip(
        tmp_path, name="f3_float32_m_overflow", namespace="train", metrics=tel.as_metrics()
    )


def test_f4_assembly_releases_promoted_inputs_before_gemm_and_preserves_caller_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """float32 -> float64 assembly frees e_loc/o_loc early and never mutates caller inputs."""

    _, e_f64, o_f64, ge_f64 = _oracle_a_arrays()
    e_f32 = e_f64.to(dtype=torch.float32)
    o_f32 = o_f64.to(dtype=torch.float32)
    ge_f32 = ge_f64.to(dtype=torch.float32)

    e_f32_snap = e_f32.clone()
    o_f32_snap = o_f32.clone()
    ge_f32_snap = ge_f32.clone()
    e_f64_snap = e_f64.clone()
    o_f64_snap = o_f64.clone()
    ge_f64_snap = ge_f64.clone()

    refs: dict[str, weakref.ReferenceType[Any]] = {}
    orig_center = lm_module.center_statistics
    center_calls = 0

    def spy_center(values: torch.Tensor, *, count: int, total: torch.Tensor) -> torch.Tensor:
        nonlocal center_calls
        out = orig_center(values, count=count, total=total)
        if center_calls == 0:
            refs["e_loc"] = weakref.ref(values)
            refs["e_bar"] = weakref.ref(out)
        elif center_calls == 1:
            gc.collect()
            assert refs["e_loc"]() is None, "promoted e_loc must be freed right after e_bar"
            refs["o_loc"] = weakref.ref(values)
            refs["o_bar"] = weakref.ref(out)
        center_calls += 1
        return out

    class _LivenessCheckReducer(StatisticsReducer):
        def __init__(self) -> None:
            self.sum_calls = 0

        def reduce_count(self, count: int) -> int:
            return count

        def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
            if self.sum_calls == 3:
                # Entering Pass-2 reductions: e_loc, o_loc, e_bar, and o_bar must all be freed.
                gc.collect()
                for name in ("e_loc", "o_loc", "e_bar", "o_bar"):
                    assert refs[name]() is None, f"{name} must be released before Pass-2 reduce"
            self.sum_calls += 1
            return value

    liveness_reducer = _LivenessCheckReducer()
    with monkeypatch.context() as mp:
        mp.setattr(lm_module, "center_statistics", spy_center)
        mats_promoted = assemble_dense_linear_method_matrices(
            e_f32, o_f32, ge_f32, solve_dtype=torch.float64, reducer=liveness_reducer
        )

    assert center_calls == 2
    assert liveness_reducer.sum_calls == 6
    assert torch.equal(e_f32, e_f32_snap)
    assert torch.equal(o_f32, o_f32_snap)
    assert torch.equal(ge_f32, ge_f32_snap)

    # Matches explicit pre-cast float64 assembly identically and also preserves float64 inputs.
    mats_precast = assemble_dense_linear_method_matrices(
        e_f32.to(torch.float64),
        o_f32.to(torch.float64),
        ge_f32.to(torch.float64),
        solve_dtype=torch.float64,
    )
    torch.testing.assert_close(mats_promoted.S_param, mats_precast.S_param, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(mats_promoted.h_c, mats_precast.h_c, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(mats_promoted.h_r, mats_precast.h_r, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(mats_promoted.dH11, mats_precast.dH11, atol=1.0e-12, rtol=1.0e-12)

    assemble_dense_linear_method_matrices(e_f64, o_f64, ge_f64, solve_dtype=torch.float64)
    assert torch.equal(e_f64, e_f64_snap)
    assert torch.equal(o_f64, o_f64_snap)
    assert torch.equal(ge_f64, ge_f64_snap)
