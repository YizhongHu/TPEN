#!/usr/bin/env python3
"""Measure a bounded-row SR replay prototype against dense sample-space SR.

This is a disposable experiment harness.  It deliberately does not add a
producer, consumer, policy, trainer, or checkpoint interface to TPEN.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import resource
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tpen.data.batch import ElectronBatch, ParameterScoreForwardPacket
from tpen.nn.forward import MaterializedParameterScoreRequest
from tpen.training.qgt import (
    QGTOperator,
    _eigh_solve,
    solve_sample_space,
)
from tpen.training.score_geometry import (
    ScoreConventions,
    build_energy_residual,
    build_score_geometry_from_rows,
    flatten_parameter_score_blocks,
)
from tpen.training.sr import SRPolicy, SRTelemetry, StochasticReconfigurationUpdate
from tpen.training.update import ModelParameterBinding, ScoreUpdateInput


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/atomistic/stage1/configs/he/train.yaml"
GI_B = 1024**3
MIB_B = 1024**2
DIAGNOSTIC_K = 32
DIAGNOSTIC_RANDOM = 32


class HarnessInvalid(RuntimeError):
    """A parity, completeness, resource, or timing failure in the harness."""


@dataclass
class Case:
    model: Any
    batch: ElectronBatch
    local_energy: torch.Tensor
    policy: SRPolicy
    model_state: dict[str, torch.Tensor]
    channel_width: int
    n_walkers: int


@dataclass
class Trial:
    seconds: float
    event_seconds: float | None
    host_crosscheck_ok: bool
    displacement: torch.Tensor
    direction: torch.Tensor
    gram: torch.Tensor | None
    trace: float | None
    damping: float | None
    diagnostics: dict[str, Any]
    memory: dict[str, int | None]
    design_cpu: torch.Tensor | None
    centering_mean_cpu: torch.Tensor | None


def _require_type(value: Any, expected: type, path: str) -> None:
    if type(value) is not expected:
        raise HarnessInvalid(f"config audit: {path} must be {expected.__name__}, got {type(value).__name__}")


def audit_config(config_path: Path) -> DictConfig:
    """Audit the real Stage-1 He config before any model construction.

    The Slice 1 plan requires a static audit of config paths/types against the
    real config before a cluster run.  These checks intentionally inspect the
    config tree, rather than copying expected parameter counts into the harness.
    """

    if config_path != DEFAULT_CONFIG:
        raise HarnessInvalid(f"config audit: only the actual Stage-1 He config is allowed: {DEFAULT_CONFIG}")
    if not config_path.is_file():
        raise HarnessInvalid(f"config audit: missing config {config_path}")
    cfg = OmegaConf.load(config_path)
    required_targets = {
        "model": "tpen.nn.TPENWaveFunction",
        "model.embedding": "tpen.nn.Embedding",
        "model.layers[0]": "tpen.nn.TPENLayer",
        "model.layers[0].mixing": "tpen.nn.EquivariantMixing",
        "model.layers[0].path_aggregation": "tpen.nn.PathAggregation",
        "model.readout": "tpen.nn.readout.PfaffianReadout",
        "sampler": "tpen.sampling.metropolis.MetropolisSampler",
    }
    for path, target in required_targets.items():
        node: Any = cfg
        for part in path.replace("[0]", ".0").split("."):
            node = node[part] if not part.isdigit() else node[int(part)]
        if node.get("_target_") != target:
            raise HarnessInvalid(f"config audit: {path}._target_ must be {target!r}, got {node.get('_target_')!r}")
    _require_type(cfg.system.n_particles, int, "system.n_particles")
    _require_type(cfg.system.spatial_dim, int, "system.spatial_dim")
    _require_type(cfg.system.spin.n_up, int, "system.spin.n_up")
    _require_type(cfg.system.spin.n_down, int, "system.spin.n_down")
    _require_type(cfg.model.embedding.out_channels, int, "model.embedding.out_channels")
    _require_type(cfg.model.embedding.hidden_channels, int, "model.embedding.hidden_channels")
    _require_type(cfg.model.layers[0].mixing.channels, int, "model.layers[0].mixing.channels")
    _require_type(cfg.model.layers[0].path_aggregation.channels, int, "model.layers[0].path_aggregation.channels")
    _require_type(cfg.model.readout.channels, int, "model.readout.channels")
    if str(cfg.runtime.dtype) != "float64":
        raise HarnessInvalid(f"config audit: runtime.dtype must be float64, got {cfg.runtime.dtype!r}")
    if cfg.model.embedding.hidden_channels != 4 * cfg.model.embedding.out_channels:
        raise HarnessInvalid("config audit: Stage-1 He hidden_channels must be 4 * out_channels")
    if cfg.model.layers[0].mixing.channels != cfg.model.embedding.out_channels:
        raise HarnessInvalid("config audit: embedding and mixing channel widths disagree")
    if cfg.model.layers[0].path_aggregation.channels != cfg.model.embedding.out_channels:
        raise HarnessInvalid("config audit: embedding and path-aggregation channel widths disagree")
    if cfg.model.readout.channels != cfg.model.embedding.out_channels:
        raise HarnessInvalid("config audit: embedding and readout channel widths disagree")
    if cfg.sampler.n_electrons != cfg.system.n_particles or cfg.sampler.spatial_dim != cfg.system.spatial_dim:
        raise HarnessInvalid("config audit: sampler and system dimensions disagree")
    return cfg


def _set_width(cfg: DictConfig, width: int) -> None:
    """Apply the Stage-1 width spelling to a copied real config."""

    if type(width) is not int or width < 1:
        raise HarnessInvalid(f"channel width must be a positive int, got {width!r}")
    cfg.model.embedding.out_channels = width
    cfg.model.embedding.hidden_channels = 4 * width
    cfg.model.layers[0].mixing.channels = width
    cfg.model.layers[0].path_aggregation.channels = width
    cfg.model.readout.channels = width


def _make_case(cfg: DictConfig, *, width: int, n_walkers: int, device: torch.device, seed: int) -> Case:
    torch.manual_seed(seed)
    local_cfg = copy.deepcopy(cfg)
    _set_width(local_cfg, width)
    model = instantiate(local_cfg.model)
    model = model.to(device=device, dtype=torch.float64)
    model.eval()
    sampler = instantiate(local_cfg.sampler)
    walkers = sampler.reset(n_walkers=n_walkers, device=device)
    batch = walkers.make_batch()
    positions = batch.positions.detach()
    # Frozen finite local energies keep this harness independent of a physics
    # evaluator while preserving the real TPEN score/VJP producer and update.
    local_energy = (
        positions.square().sum(dim=(-2, -1))
        + 0.03125 * positions.sin().sum(dim=(-2, -1))
    ).detach()
    if not torch.isfinite(local_energy).all():
        raise HarnessInvalid("frozen local-energy fixture is non-finite")
    policy = SRPolicy(solve_space="sample", score_chunk_size=16)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return Case(
        model=model,
        batch=batch,
        local_energy=local_energy,
        policy=policy,
        model_state=state,
        channel_width=width,
        n_walkers=n_walkers,
    )


def _restore(case: Case) -> None:
    case.model.load_state_dict(case.model_state)
    case.model.zero_grad(set_to_none=True)


def _panel_batch(batch: ElectronBatch, indices: torch.Tensor) -> ElectronBatch:
    kwargs: dict[str, Any] = {
        "positions": batch.positions.index_select(0, indices),
        "system": batch.system,
        "nuclear_positions": batch.nuclear_positions,
        "nuclear_charges": batch.nuclear_charges,
        "atomic_configuration": batch.atomic_configuration,
        "spins": None if batch.spins is None else batch.spins.index_select(0, indices),
        "aux": dict(batch.aux),
    }
    return ElectronBatch(**kwargs)


def _score_rows(model: Any, batch: ElectronBatch, indices: torch.Tensor, *, chunk_size: int | None) -> torch.Tensor:
    panel = _panel_batch(batch, indices)
    packet = model.evaluate_materialized_parameter_score_request(
        request=MaterializedParameterScoreRequest(chunk_size=chunk_size),
        batch=panel,
    )
    if not isinstance(packet, ParameterScoreForwardPacket):
        raise HarnessInvalid(f"score producer returned {type(packet).__name__}, not ParameterScoreForwardPacket")
    rows = flatten_parameter_score_blocks(packet.parameter_scores, sample_shape=tuple(packet.output.logabs.shape))
    rows = rows.detach()
    if rows.dtype != torch.float64 or rows.shape[0] != indices.numel():
        raise HarnessInvalid(f"score producer shape/dtype mismatch: {tuple(rows.shape)} {rows.dtype}")
    if not torch.isfinite(rows).all():
        raise HarnessInvalid("real TPEN score/VJP producer returned non-finite scores")
    return rows


def _reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _memory_snapshot(device: torch.device) -> dict[str, int | None]:
    process_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        process_peak *= 1
    else:
        process_peak *= 1024
    if device.type != "cuda":
        return {
            "max_memory_allocated_bytes": None,
            "max_memory_reserved_bytes": None,
            "device_capacity_bytes": None,
            "process_peak_bytes": process_peak,
        }
    return {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device_capacity_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "process_peak_bytes": process_peak,
    }


def _timed(device: torch.device, fn: Any) -> tuple[float, float | None, bool, Any, dict[str, int | None]]:
    """Run a body with CUDA events plus a synchronized host cross-check."""

    _reset_memory(device)
    if device.type != "cuda":
        started = time.perf_counter()
        value = fn()
        elapsed = time.perf_counter() - started
        return elapsed, None, True, value, _memory_snapshot(device)
    # Slice 1 plan: "Synchronize CUDA immediately before starting and immediately after completing each timed body; use CUDA events (and record a synchronized host cross-check)."
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    host_start = time.perf_counter()
    start.record()
    value = fn()
    end.record()
    torch.cuda.synchronize(device)
    host_elapsed = time.perf_counter() - host_start
    event_elapsed = start.elapsed_time(end) / 1000.0
    crosscheck = abs(host_elapsed - event_elapsed) <= 0.25 * max(host_elapsed, event_elapsed, 1.0e-12)
    if not crosscheck or event_elapsed <= 0.0 or host_elapsed <= 0.0:
        raise HarnessInvalid(
            f"unsynchronized timing: CUDA event={event_elapsed:.6g}s host={host_elapsed:.6g}s"
        )
    return event_elapsed, event_elapsed, True, value, _memory_snapshot(device)


def _dense_trial(case: Case, *, include_reference: bool, retain_design: bool = False) -> Trial:
    model = case.model
    request = MaterializedParameterScoreRequest(chunk_size=case.policy.score_chunk_size)
    packet = model.evaluate_materialized_parameter_score_request(request=request, batch=case.batch).detach()
    reference_gram = None
    reference_trace = None
    reference_damping = None
    design_cpu = None
    centering_mean_cpu = None
    if include_reference:
        # The dense reference is parity-only; it must not be part of the timed current path.
        rows = flatten_parameter_score_blocks(packet.parameter_scores, sample_shape=tuple(packet.output.logabs.shape))
        geometry = build_score_geometry_from_rows(rows, layout=packet.parameter_scores.layout)
        if retain_design:
            design_cpu = geometry.design.detach().to("cpu", dtype=torch.float64)
            # Keep the O(P) mean paired with the already-retained diagnostic design.
            centering_mean_cpu = (rows.sum(dim=0) / float(rows.shape[0])).detach().to("cpu", dtype=torch.float64)
        residual = build_energy_residual(case.local_energy, geometry=geometry)
        operator = QGTOperator(geometry)
        _, dense_diag = solve_sample_space(
            operator,
            residual,
            damping=case.policy.damping,
            rank_cutoff=case.policy.rank_cutoff,
        )
        reference_gram = operator.dense_sample_gram().detach()
        reference_trace = float(operator.trace().item())
        reference_damping = float(dense_diag.shift)
    optimizer = torch.optim.SGD(model.parameters(), lr=case.policy.learning_rate)
    model_parameters = ModelParameterBinding(
        parameters=tuple(model.parameters()),
        layout=model.parameter_binding.layout,
    )
    method = StochasticReconfigurationUpdate(
        optimizer,
        model_parameters=model_parameters,
        policy=case.policy,
    )
    update_input = ScoreUpdateInput(
        batch=case.batch,
        wavefunction=packet.output,
        local_energy=case.local_energy,
        step=0,
        parameter_scores=packet.parameter_scores,
        parameter_binding=model.parameter_binding,
    )
    before = torch.nn.utils.parameters_to_vector(tuple(model.parameters())).detach().clone()
    result = method.update(update_input)
    after = torch.nn.utils.parameters_to_vector(tuple(model.parameters())).detach().clone()
    if not result.applied or not isinstance(result.diagnostics, SRTelemetry):
        raise HarnessInvalid(f"dense current path did not apply: {result.reason!r}")
    telemetry = result.diagnostics
    if telemetry.diagnostics is None:
        raise HarnessInvalid("dense current path omitted solve diagnostics")
    trust = telemetry.trust_scale
    direction = -((after - before) / (case.policy.learning_rate * trust))
    return Trial(
        seconds=0.0,
        event_seconds=None,
        host_crosscheck_ok=True,
        displacement=after - before,
        direction=direction,
        gram=reference_gram,
        trace=reference_trace,
        damping=reference_damping if include_reference else float(telemetry.diagnostics.shift),
        diagnostics={
            "applied": telemetry.applied,
            "reason": telemetry.reason,
            "space": telemetry.diagnostics.space,
            "n_modes": telemetry.diagnostics.n_modes,
            "retained_modes": telemetry.diagnostics.retained_modes,
            "dtype": telemetry.diagnostics.dtype,
        },
        memory={},
        design_cpu=design_cpu,
        centering_mean_cpu=centering_mean_cpu,
    )


def _replay_trial(case: Case, *, panel_size: int, retain_centering_mean: bool = False) -> Trial:
    n = case.n_walkers
    p = case.model.parameter_binding.layout.total_numel
    all_indices = torch.arange(n, device=case.batch.positions.device)
    panels = [all_indices[start : start + panel_size] for start in range(0, n, panel_size)]
    coverage = torch.bincount(torch.cat(panels).cpu(), minlength=n)
    if not torch.equal(coverage, torch.ones(n, dtype=coverage.dtype)):
        raise HarnessInvalid(f"replay row coverage is not exactly once: {coverage.tolist()}")
    score_sum = torch.zeros(p, dtype=torch.float64, device=case.batch.positions.device)
    for indices in panels:
        score_sum += _score_rows(case.model, case.batch, indices, chunk_size=indices.numel()).sum(dim=0)
    mean = score_sum / float(n)
    gram = torch.zeros((n, n), dtype=torch.float64, device=case.batch.positions.device)
    # Slice 1 plan: "it constructs the centered sample Gram, solves and maps back once."
    # The replay schedule holds only the active b=16 row panels, never an [N,P] table.
    for outer in range(len(panels)):
        outer_rows = _score_rows(case.model, case.batch, panels[outer], chunk_size=panels[outer].numel()) - mean
        for inner in range(outer, len(panels)):
            inner_rows = (
                outer_rows
                if inner == outer
                else _score_rows(case.model, case.batch, panels[inner], chunk_size=panels[inner].numel()) - mean
            )
            i0 = int(panels[outer][0].item())
            j0 = int(panels[inner][0].item())
            block = outer_rows @ inner_rows.transpose(0, 1) / float(n)
            gram[i0 : i0 + outer_rows.shape[0], j0 : j0 + inner_rows.shape[0]] = block
            if inner != outer:
                gram[j0 : j0 + inner_rows.shape[0], i0 : i0 + outer_rows.shape[0]] = block.transpose(0, 1)
    gram = 0.5 * (gram + gram.transpose(0, 1))
    energy = case.local_energy.reshape(-1)
    conventions = ScoreConventions()
    epsilon = (energy - energy.mean()) * (conventions.energy_gradient_scale / float(n) ** 0.5)
    trace = float(torch.diagonal(gram).sum().item())
    damping = case.policy.damping.shift(trace=trace, n_parameters=p)
    y, retained, max_eigenvalue, min_retained = _eigh_solve(
        gram,
        epsilon,
        shift=damping,
        rank_cutoff=case.policy.rank_cutoff,
    )
    direction = torch.zeros(p, dtype=torch.float64, device=case.batch.positions.device)
    # Slice 1 plan: "one final score replay" for the map-back after the solve.
    for indices in panels:
        rows = _score_rows(case.model, case.batch, indices, chunk_size=indices.numel()) - mean
        direction += rows.transpose(0, 1) @ y.index_select(0, indices) / float(n) ** 0.5
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    trust = case.policy.trust_scale(direction_norm=direction_norm)
    displacement = -case.policy.learning_rate * trust * direction
    return Trial(
        seconds=0.0,
        event_seconds=None,
        host_crosscheck_ok=True,
        displacement=displacement,
        direction=direction,
        gram=gram,
        trace=trace,
        damping=float(damping),
        diagnostics={
            "applied": True,
            "reason": "applied",
            "space": "sample",
            "n_modes": int(gram.shape[0]),
            "retained_modes": int(retained),
            "max_eigenvalue": float(max_eigenvalue),
            "min_retained_eigenvalue": float(min_retained),
            "dtype": str(gram.dtype),
            "row_coverage_counts": [int(value) for value in coverage.tolist()],
            "row_panel_count": len(panels),
            "row_panel_sizes": [int(panel.numel()) for panel in panels],
        },
        memory={},
        design_cpu=None,
        centering_mean_cpu=mean.detach().to("cpu", dtype=torch.float64) if retain_centering_mean else None,
    )


def _assert_parity(dense: Trial, replay: Trial) -> None:
    # Slice 1 plan: "Before timing, it must match the current dense sample-space direction/displacement ... (float64: direction/displacement rtol=atol=1e-9; Gram/trace/lambda rtol=atol=1e-11; discrete route/skip fields exact)."
    try:
        torch.testing.assert_close(replay.direction, dense.direction, rtol=1.0e-9, atol=1.0e-9)
        torch.testing.assert_close(replay.displacement, dense.displacement, rtol=1.0e-9, atol=1.0e-9)
        torch.testing.assert_close(replay.gram, dense.gram, rtol=1.0e-11, atol=1.0e-11)
        if abs(float(replay.trace) - float(dense.trace)) > 1.0e-11 * max(abs(float(dense.trace)), 1.0):
            raise AssertionError("trace mismatch")
        if abs(float(replay.damping) - float(dense.damping)) > 1.0e-11 * max(abs(float(dense.damping)), 1.0):
            raise AssertionError("lambda mismatch")
    except AssertionError as exc:
        raise HarnessInvalid(f"parity failed: {exc}") from exc
    for field in ("space", "n_modes", "retained_modes", "dtype", "applied", "reason"):
        if replay.diagnostics.get(field) != dense.diagnostics.get(field):
            raise HarnessInvalid(
                f"discrete parity failed for {field}: {replay.diagnostics.get(field)!r} != {dense.diagnostics.get(field)!r}"
            )


def _diagnostic_entries(
    dense_gram: torch.Tensor,
    replay_gram: torch.Tensor,
    design_cpu: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    """Compare selected Gram entries with a CPU ``math.fsum`` reference."""

    dense_cpu = dense_gram.detach().to("cpu", dtype=torch.float64)
    replay_cpu = replay_gram.detach().to("cpu", dtype=torch.float64)
    mismatch = (dense_cpu - replay_cpu).abs()
    n, p = design_cpu.shape
    flat_order = torch.argsort(mismatch.reshape(-1), descending=True).tolist()
    worst = flat_order[: min(DIAGNOSTIC_K, n * n)]
    rng = random.Random(seed)
    random_indices = rng.sample(range(n * n), min(DIAGNOSTIC_RANDOM, n * n))
    selected: dict[int, set[str]] = {}
    for index in worst:
        selected.setdefault(int(index), set()).add("worst")
    for index in random_indices:
        selected.setdefault(int(index), set()).add("random")

    eps = torch.finfo(torch.float64).eps
    entries: list[dict[str, Any]] = []
    for flat_index, kinds in selected.items():
        i, j = divmod(flat_index, n)
        row_i = design_cpu[i].tolist()
        row_j = design_cpu[j].tolist()
        g_ref = math.fsum(float(a) * float(b) for a, b in zip(row_i, row_j))
        s_ij = math.fsum(abs(float(a)) * abs(float(b)) for a, b in zip(row_i, row_j))
        scale = float(p) * eps * s_ij
        dense_error = float(dense_cpu[i, j].item()) - g_ref
        replay_error = float(replay_cpu[i, j].item()) - g_ref
        entries.append(
            {
                "i": i,
                "j": j,
                "selection": sorted(kinds),
                "dense_minus_reference": dense_error,
                "replay_minus_reference": replay_error,
                "abs_g_reference": abs(g_ref),
                "S_ij": s_ij,
                "P_times_eps_times_S": scale,
                "dense_error_over_P_eps_S": dense_error / scale if scale else (0.0 if dense_error == 0.0 else math.inf),
                "replay_error_over_P_eps_S": replay_error / scale if scale else (0.0 if replay_error == 0.0 else math.inf),
                "farther_from_reference": "dense" if abs(dense_error) > abs(replay_error) else "replay" if abs(replay_error) > abs(dense_error) else "tie",
            }
        )

    dense_errors = [abs(entry["dense_minus_reference"]) for entry in entries]
    replay_errors = [abs(entry["replay_minus_reference"]) for entry in entries]
    dense_ratios = [abs(entry["dense_error_over_P_eps_S"]) for entry in entries]
    replay_ratios = [abs(entry["replay_error_over_P_eps_S"]) for entry in entries]
    farther_counts = {"dense": 0, "replay": 0, "tie": 0}
    for entry in entries:
        farther_counts[entry["farther_from_reference"]] += 1
    return {
        "K_worst": DIAGNOSTIC_K,
        "random_count": DIAGNOSTIC_RANDOM,
        "selected_count": len(entries),
        "reference": "math.fsum(float(x_i[k]) * float(x_j[k]) for k in range(P))",
        "eps": float(eps),
        "entries": entries,
        "summary": {
            "max_abs_dense_error": max(dense_errors, default=0.0),
            "max_abs_replay_error": max(replay_errors, default=0.0),
            "max_abs_dense_error_over_P_eps_S": max(dense_ratios, default=0.0),
            "max_abs_replay_error_over_P_eps_S": max(replay_ratios, default=0.0),
            "mean_abs_dense_error": sum(dense_errors) / len(dense_errors) if dense_errors else 0.0,
            "mean_abs_replay_error": sum(replay_errors) / len(replay_errors) if replay_errors else 0.0,
            "farther_from_reference_counts": farther_counts,
        },
    }


def _bound_check(
    dense_gram: torch.Tensor,
    replay_gram: torch.Tensor,
    design_cpu: torch.Tensor,
) -> dict[str, Any]:
    """Check the non-gating first-order summation bound for every entry."""

    dense_cpu = dense_gram.detach().to("cpu", dtype=torch.float64)
    replay_cpu = replay_gram.detach().to("cpu", dtype=torch.float64)
    abs_design = design_cpu.abs()
    s_matrix = abs_design @ abs_design.transpose(0, 1)
    eps = torch.finfo(torch.float64).eps
    p = design_cpu.shape[1]
    difference = (dense_cpu - replay_cpu).abs()
    denominator = 4.0 * float(p) * eps * s_matrix
    ratio = torch.zeros_like(difference)
    positive = denominator > 0.0
    ratio[positive] = difference[positive] / denominator[positive]
    ratio[~positive] = torch.where(difference[~positive] == 0.0, torch.zeros_like(difference[~positive]), torch.full_like(difference[~positive], math.inf))
    return {
        "bound": "4 * P * eps * S_ij",
        "pass": bool(torch.all(difference <= denominator).item()),
        "max_ratio": float(torch.max(ratio).item()),
        "violating_entries": int(torch.count_nonzero(difference > denominator).item()),
        "P": int(p),
        "eps": float(eps),
    }


def _diagnostic_report(case: Case, dense: Trial, replay: Trial, *, seed: int) -> dict[str, Any]:
    design_cpu = dense.design_cpu
    if dense.gram is None or replay.gram is None or design_cpu is None:
        raise HarnessInvalid("diagnostic requires dense and replay Gram matrices")
    if dense.centering_mean_cpu is None or replay.centering_mean_cpu is None:
        raise HarnessInvalid("diagnostic requires dense and replay centering means")
    trace_diff = float(replay.trace) - float(dense.trace)
    lambda_diff = float(replay.damping) - float(dense.damping)
    mean_difference = dense.centering_mean_cpu - replay.centering_mean_cpu
    max_abs_mean_difference = float(mean_difference.abs().max().item())
    max_abs_mean = float(
        torch.maximum(dense.centering_mean_cpu.abs(), replay.centering_mean_cpu.abs()).max().item()
    )
    eps = torch.finfo(torch.float64).eps
    eps_max_abs_mean = eps * max_abs_mean
    return {
        "cell": {"N": case.n_walkers, "P": int(design_cpu.shape[1]), "channel_width": case.channel_width},
        "row_structure": {
            "coverage_asserted": True,
            "coverage_counts": replay.diagnostics.get("row_coverage_counts"),
            "panel_count": replay.diagnostics.get("row_panel_count"),
            "panel_sizes": replay.diagnostics.get("row_panel_sizes"),
        },
        "trace_lambda_compare": {
            "trace_dense": float(dense.trace),
            "trace_replay": float(replay.trace),
            "trace_difference": trace_diff,
            "lambda_dense": float(dense.damping),
            "lambda_replay": float(replay.damping),
            "lambda_difference": lambda_diff,
            "pass": abs(trace_diff) <= 1.0e-11 * max(abs(float(dense.trace)), 1.0) and abs(lambda_diff) <= 1.0e-11 * max(abs(float(dense.damping)), 1.0),
        },
        "centering_mean_compare": {
            "max_abs_mean_difference": max_abs_mean_difference,
            "max_abs_mean": max_abs_mean,
            "eps": float(eps),
            "eps_times_max_abs_mean": eps_max_abs_mean,
            "ratio_to_eps_times_max_abs_mean": (
                max_abs_mean_difference / eps_max_abs_mean
                if eps_max_abs_mean
                else (0.0 if max_abs_mean_difference == 0.0 else math.inf)
            ),
        },
        "entry_reference": _diagnostic_entries(dense.gram, replay.gram, design_cpu, seed=seed),
        "bound_check": _bound_check(dense.gram, replay.gram, design_cpu),
    }


def _json_tensor_stats(values: list[float]) -> dict[str, Any]:
    return {
        "samples_seconds": values,
        "mean_seconds": sum(values) / len(values),
        "median_seconds": sorted(values)[len(values) // 2],
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _run_cell(
    cfg: DictConfig,
    *,
    width: int,
    n_walkers: int,
    device: torch.device,
    panel_size: int,
    warmups: int,
    repetitions: int,
    seed: int,
    diagnostic: bool = False,
) -> dict[str, Any]:
    case = _make_case(cfg, width=width, n_walkers=n_walkers, device=device, seed=seed)
    p = case.model.parameter_binding.layout.total_numel
    _restore(case)
    dense_parity = _dense_trial(case, include_reference=True, retain_design=diagnostic)
    _restore(case)
    replay_parity = _replay_trial(case, panel_size=panel_size, retain_centering_mean=diagnostic)
    parity_error: str | None = None
    try:
        _assert_parity(dense_parity, replay_parity)
    except HarnessInvalid as exc:
        parity_error = str(exc)
    diagnostic_report = _diagnostic_report(case, dense_parity, replay_parity, seed=seed) if diagnostic else None
    if parity_error is not None:
        if not diagnostic:
            raise HarnessInvalid(parity_error)
        return {
            "N": n_walkers,
            "channel_width": width,
            "P": p,
            "panel_size": panel_size,
            "parity": {"status": "FAIL", "error": parity_error, "gram_trace_lambda_rtol": 1.0e-11, "gram_trace_lambda_atol": 1.0e-11},
            "diagnostic": diagnostic_report,
            "timing": {"status": "UNMEASURED_PARITY_FAILURE"},
        }
    dense_times: list[float] = []
    replay_times: list[float] = []
    dense_memory: list[dict[str, int | None]] = []
    replay_memory: list[dict[str, int | None]] = []
    timing_ok = True
    for _ in range(warmups):
        for arm in ("dense", "replay"):
            _restore(case)
            trial_fn = (lambda: _dense_trial(case, include_reference=False)) if arm == "dense" else (lambda: _replay_trial(case, panel_size=panel_size))
            _, _, _, _, _ = _timed(device, trial_fn)
    for pair in range(repetitions):
        order = ("dense", "replay") if pair % 2 == 0 else ("replay", "dense")
        for arm in order:
            _restore(case)
            trial_fn = (lambda: _dense_trial(case, include_reference=False)) if arm == "dense" else (lambda: _replay_trial(case, panel_size=panel_size))
            seconds, event_seconds, host_ok, trial, memory = _timed(device, trial_fn)
            if not host_ok:
                raise HarnessInvalid("unsynchronized timing")
            trial.seconds = seconds
            trial.event_seconds = event_seconds
            trial.host_crosscheck_ok = host_ok
            trial.memory = memory
            if arm == "dense":
                dense_times.append(seconds)
                dense_memory.append(memory)
            else:
                replay_times.append(seconds)
                replay_memory.append(memory)
    dense_stats = _json_tensor_stats(dense_times)
    replay_stats = _json_tensor_stats(replay_times)
    ratio = replay_stats["mean_seconds"] / dense_stats["mean_seconds"]
    cell = {
        "N": n_walkers,
        "channel_width": width,
        "P": p,
        "panel_size": panel_size,
        "dense": {**dense_stats, "memory": dense_memory},
        "replay": {**replay_stats, "memory": replay_memory},
        "slowdown_replay_over_dense": ratio,
        "parity": {
            "status": "PASS",
            "direction_rtol": 1.0e-9,
            "direction_atol": 1.0e-9,
            "gram_trace_lambda_rtol": 1.0e-11,
            "gram_trace_lambda_atol": 1.0e-11,
            "discrete_fields": "exact",
        },
        "structure": {
            "row_coverage_counts": replay_parity.diagnostics.get("row_coverage_counts"),
            "row_panel_count": replay_parity.diagnostics.get("row_panel_count"),
            "row_panel_sizes": replay_parity.diagnostics.get("row_panel_sizes"),
            "trace": {"dense": float(dense_parity.trace), "replay": float(replay_parity.trace)},
            "lambda": {"dense": float(dense_parity.damping), "replay": float(replay_parity.damping)},
        },
        "timing": {
            "cuda_events": device.type == "cuda",
            "host_crosscheck": True,
            "synchronized": True,
            "warmups": warmups,
            "paired_repetitions": repetitions,
        },
    }
    if diagnostic_report is not None:
        cell["diagnostic"] = diagnostic_report
    return cell


def _memory_gate(
    cells: list[dict[str, Any]], *, device: torch.device, diagnostic: bool = False
) -> dict[str, Any]:
    if diagnostic:
        return {"status": "NOT_APPLICABLE_DIAGNOSTIC_SINGLE_CELL"}
    if device.type != "cuda":
        return {"status": "UNMEASURED_CPU_NON_EVIDENCE"}
    # Slice 1 plan: "accept the prototype result only if all four allocator peaks fit below 18 GiB ... and the measured increment in (N) is invariant ... and the measured increment in (P) is invariant ..."
    replay_peaks: dict[tuple[int, int], int] = {}
    for cell in cells:
        values = [entry["max_memory_allocated_bytes"] for entry in cell["replay"]["memory"]]
        if not values or any(value is None for value in values):
            raise HarnessInvalid("missing prototype allocator peak")
        replay_peaks[(cell["N"], cell["channel_width"])] = max(int(value) for value in values)
    maximum = max(replay_peaks.values())
    ns = sorted({key[0] for key in replay_peaks})
    widths = sorted({key[1] for key in replay_peaks})
    if len(ns) != 2 or len(widths) != 2:
        raise HarnessInvalid("missing cell for measured memory gate")
    n_deltas = [replay_peaks[(ns[1], width)] - replay_peaks[(ns[0], width)] for width in widths]
    p_deltas = [replay_peaks[(ns, widths[1])] - replay_peaks[(ns, widths[0])] for ns in ns]
    tolerance = max(256 * MIB_B, int(0.10 * maximum))
    invariant = max(n_deltas) - min(n_deltas) <= tolerance and max(p_deltas) - min(p_deltas) <= tolerance
    under_ceiling = all(value < 18 * GI_B for value in replay_peaks.values())
    return {
        "status": "PASS" if invariant and under_ceiling else "FAIL",
        "prototype_peak_bytes": {f"N{n}_C{w}": v for (n, w), v in replay_peaks.items()},
        "maximum_peak_bytes": maximum,
        "ceiling_bytes": 18 * GI_B,
        "n_increment_bytes": n_deltas,
        "p_increment_bytes": p_deltas,
        "interaction_tolerance_bytes": tolerance,
        "under_ceiling": under_ceiling,
        "increments_invariant": invariant,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--panel-size", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--walkers", type=int, nargs=2, default=(64, 128))
    parser.add_argument("--channels", type=int, nargs=2, default=(16, 32))
    parser.add_argument("--smoke", action="store_true", help="tiny CPU-only correctness/timing smoke; not evidence")
    parser.add_argument("--diagnostic", action="store_true", help="report exact-reference Gram diagnostics for one cell")
    parser.add_argument("--diagnostic-cell", type=int, nargs=2, metavar=("N", "CHANNELS"), help="N and channel width for --diagnostic")
    return parser.parse_args()


def _human_summary(result: dict[str, Any]) -> str:
    lines = [
        "SR bounded-row replay benchmark",
        f"status={result['status']} device={result.get('device', 'unknown')} cells={len(result['cells'])}",
    ]
    for cell in result["cells"]:
        if "dense" in cell and "replay" in cell:
            lines.append(
                "N={N} P={P} channels={channel_width}: dense={dense:.6g}s replay={replay:.6g}s ratio={ratio:.6g} parity={parity}".format(
                    N=cell["N"],
                    P=cell["P"],
                    channel_width=cell["channel_width"],
                    dense=cell["dense"]["mean_seconds"],
                    replay=cell["replay"]["mean_seconds"],
                    ratio=cell["slowdown_replay_over_dense"],
                    parity=cell["parity"]["status"],
                )
            )
        else:
            lines.append(
                f"N={cell['N']} P={cell['P']} channels={cell['channel_width']}: timing=UNMEASURED parity={cell['parity']['status']}"
            )
    gate = result.get("memory_gate")
    if gate:
        lines.append(f"memory_gate={gate.get('status', 'unknown')}")
    if "error" in result:
        lines.append(f"error={result['error']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output = args.output or ROOT / "scripts/cannon/results/sr-replay-benchmark.json"
    result: dict[str, Any] = {
        "schema": "sr-replay-benchmark-v1",
        "purpose": "benchmark",
        "status": "UNMEASURED",
        "source_commit": os.environ.get("TPEN_SOURCE_SHA", "unknown"),
        "config": str(args.config),
        "typecheck": "on (pyproject addopts, no override)",
        "plan_contract": {
            "dense_score_chunk_size": 16,
            "prototype_panel_size": args.panel_size,
            "matrix": "N in {64,128} x actual P from binding.layout.total_numel for Stage-1 He channels {16,32}",
        },
        "cells": [],
    }
    try:
        cfg = audit_config(args.config)
        if args.panel_size != 16 and not args.smoke:
            raise HarnessInvalid("production benchmark requires panel b=16")
        if args.smoke:
            if args.device not in ("auto", "cpu"):
                raise HarnessInvalid("CPU smoke must use --device cpu or auto")
            walkers = (4, 8)
            channels = (4, 8)
            warmups = 1
            repetitions = 2
            device = torch.device("cpu")
            result["smoke"] = True
        else:
            walkers = tuple(args.walkers)
            channels = tuple(args.channels)
            warmups = args.warmups
            repetitions = args.repetitions
            device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
            if device.type != "cuda":
                raise HarnessInvalid("production benchmark requires CUDA; use --smoke for CPU")
        if args.diagnostic:
            if args.diagnostic_cell is None:
                raise HarnessInvalid("--diagnostic requires --diagnostic-cell N CHANNELS")
            walkers = (args.diagnostic_cell[0],)
            channels = (args.diagnostic_cell[1],)
            result["diagnostic"] = {"cell": {"N": walkers[0], "channel_width": channels[0]}, "K_worst": DIAGNOSTIC_K, "random_count": DIAGNOSTIC_RANDOM}
        if args.panel_size < 1 or warmups < 0 or repetitions < 1:
            raise HarnessInvalid("panel size, warmups, and repetitions must be positive where applicable")
        for n in walkers:
            if type(n) is not int or n < 2:
                raise HarnessInvalid(f"walker count must be an int >= 2, got {n!r}")
        for width in channels:
            if type(width) is not int or width < 1:
                raise HarnessInvalid(f"channel width must be a positive int, got {width!r}")
        for n in walkers:
            for width in channels:
                result["cells"].append(
                    _run_cell(
                        cfg,
                        width=width,
                        n_walkers=n,
                        device=device,
                        panel_size=args.panel_size,
                        warmups=warmups,
                        repetitions=repetitions,
                        seed=args.seed,
                        diagnostic=args.diagnostic,
                    )
                )
        result["device"] = str(device)
        result["memory_gate"] = _memory_gate(result["cells"], device=device, diagnostic=args.diagnostic)
        if args.diagnostic:
            parity_status = "PASS" if all(cell["parity"]["status"] == "PASS" for cell in result["cells"]) else "FAIL"
            result["status"] = f"DIAGNOSTIC_COMPLETE_PARITY_{parity_status}"
        else:
            result["status"] = "PASS" if result["memory_gate"]["status"] in ("PASS", "UNMEASURED_CPU_NON_EVIDENCE") else "FAIL"
            if args.smoke:
                result["status"] = "CPU_SMOKE_PASS"
    except (HarnessInvalid, RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as exc:
        result["status"] = "HARNESS_INVALID"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["memory_gate"] = {"status": "UNMEASURED"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = output.with_suffix(".summary.txt")
    summary = _human_summary(result)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(json.dumps({"status": result["status"], "output": str(output), "summary": str(summary_path), "cells": len(result["cells"])}, sort_keys=True))
    return 0 if result["status"] in ("PASS", "CPU_SMOKE_PASS", "DIAGNOSTIC_COMPLETE_PARITY_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
