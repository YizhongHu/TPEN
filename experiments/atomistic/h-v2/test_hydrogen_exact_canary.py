"""End-to-end evaluation-route canary for the exact one-electron hydrogen state.

Drives `tpen.physics.hydrogen.HydrogenExact` through the real evaluation
route: `tpen.run.run_from_config` into the real `Evaluate` runner,
`TrajectoryMCMCGenerator`, `LocalEnergyCalculator`/`WavefunctionCalculator`,
`SampledRecordWriter`/`LocalEnergySummary`, and finally the real H-v2
collector (`collect.py`).

Neither `test_oracle_vs_tpen.py` nor `test_stub_oracle_contract.py` touches
this route: both feed a stub wavefunction straight to the Hamiltonian terms
and sampler, never through `run_from_config`, the trajectory record writer,
the summaries, or the collector (see container note
``canary-overlap-assessment-20261106``). An evaluation-path defect -- a
wrong aggregation, a dropped or duplicated record, or a summary computed on
a subset -- cannot be told apart from model behavior on a learned model,
whose true local energy is unknown. On the exact hydrogen ground state
(alpha=1) every local energy must equal -0.5 Ha exactly, so any such defect
is directly visible.

**Experiment boundary.** Per the operator ruling (root note
``operator-decisions-20261106-imports-mirflow-readonly``, amended by
``acceptance-contract-amendment-20261106``), this file reaches core only
through configuration, `tpen.run.run_from_config`, or the installed `tpen`
command -- never a direct `tpen.*` import. The wrong-kinetic-prefactor and
flipped-potential-sign mutants, which need a direct monkeypatch of
`tpen.physics.kinetic.KineticEnergy`/`tpen.physics.potential.
ElectronNucleusPotential`, therefore moved to the core integration test
(`tests/integration/evaluation/test_hydrogen_exact_evaluation.py`), which is
not subject to this boundary. `experiments/atomistic/h-v2/
test_import_boundary.py` enforces this boundary for every file in this
directory, including this one.

**Restore scope.** `HydrogenExact` carries no trainable state, so there is
no optimizer/trainer/sampler state a *training*-resume checkpoint would add
(an `nn.Module` with zero parameters cannot even build an optimizer --
`torch.optim.SGD([])` raises "optimizer got an empty parameter list"). The
only configured route that writes a checkpoint is `tpen.runner.Train`, and
`Train.run` unconditionally builds a real optimizer over `model.parameters()`
before touching the trainer, sampler, or `load` mode
(`tpen/runner/train.py`), so there is no configured route -- `run_from_config`
or the `tpen` command -- that can produce a checkpoint for this module.
`test_no_configured_checkpoint_route_exists_for_the_parameterless_model`
below exercises this directly rather than assuming it, per the amendment's
requirement to report the exact error (see `implementation-receipt`). This
canary therefore uses no checkpoint and does not cover restore; the core
integration test covers that honestly instead, with a genuine checkpoint
built directly via `tpen.checkpoint.save.save_checkpoint` (core test code may
do that freely; this file may not).

Because there is no checkpoint, this canary cannot call the collector's
composite `collect_records` (it requires a restored checkpoint directory and
a `model_only` resolved restore mode). It instead calls the real collector
functions that do not depend on a checkpoint directly --
`collect.validate_run_identity`, `collect.validate_manifest`, and
`collect.validate_rows` -- so source-commit identity, resolved-config
identity, manifest completeness, and full per-row validation are still
genuinely exercised against the real collector code. This is a disclosed
gap, not a silently dropped one: `collect.validate_checkpoint_identity` and
`collect.validate_restore_mode` are exercised by `test_collect.py`'s own
unit tests against synthetic checkpoint artifacts instead.

**Full-record-set bookkeeping.** The streamed trajectory manifest
(`sampled_eval_table.metadata.json`, written by
`tpen.evaluation.trajectory_records`) accumulates `mean`/`variance`/
`row_count`/`finite_count` across every retained draw, not just the final
one. `TrajectoryMCMCGenerator`'s evaluation batch, by contrast, is
explicitly "its final retained draw" (see that generator's own docstring),
so the JSONL `local_energy_mean`/`local_energy_variance` metrics that
`LocalEnergySummary` publishes from that batch are a final-draw snapshot,
not a full-grid aggregate -- a real, documented distinction, not a defect.
Both tests below check the manifest's own aggregates against an independent
recomputation from every row in `sampled_eval_table.csv`, to tight float64
tolerance, separately from the Monte Carlo bands on the JSONL snapshot
metric. They also check the JSONL metric itself against a recomputation
from only the rows of its own final draw (`_final_draw_energies`) -- the
same population it claims to summarize, so this must agree to tight
tolerance too. A summary silently aggregated over a further subset of that
same draw (e.g. half its walkers) moves this second recomputation
measurably at alpha=1.2, where the local energy is position-dependent, even
when the raw metric still falls inside the predeclared Monte Carlo band. At
alpha=1 every value is the same constant, so subsetting a constant signal
cannot change its mean by construction -- a mathematical limit on this
family of check, not a gap specific to this canary, and the alpha=1.2 test
is what actually exercises it.

**Record-path precision.** `HydrogenExact`'s local energy is the closed
form `-alpha^2/2 + (alpha-1)/r` exactly at every walker, independent of
position. At alpha=1.2 the imperfect-state test recomputes this pointwise
from each record's own recorded position and requires float64-level
agreement (`rel=1e-10`), which a float32 downcast anywhere in the record
path (~1e-7 relative error) would fail. The trajectory manifest exposes no
separate dtype/column-precision field to assert directly; this pointwise
check is the precision gate.

**Calibrated moment band.** The imperfect-state (alpha=1.2) final-draw mean
and variance were previously checked only for internal bookkeeping (agreement
with their own recomputation), never for accuracy against a reference. Both
are now also required to fall inside a moment band calibrated by
`moment_bands.calibrate_moment_band` (see `test_moment_bands.py` for that
module's own unit tests) at `sample_count=N_WALKERS`, matching this draw's
independent walker count exactly -- walkers are independent after burn-in,
while successive draws of one walker are not, so only a walker count is a
valid calibration sample size here. The full-grid comparisons below are
unaffected and keep their existing whole-walker uncertainty.

**CUDA parametrization.** Both canary tests are parametrized over
`runtime.device` (`cpu`, `cuda`), set purely through configuration. The
`cuda` case is skipped with an explicit reason when no CUDA device is
available; a GPU verification run is expected to collect it unskipped.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from tpen.run import run_from_config

HERE = Path(__file__).parent
CONFIG_PATH = HERE / "configs" / "exact_canary_eval.yaml"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"h_v2_{name}", HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collect = _module("collect")
moment_bands = _module("moment_bands")

#: Parametrized pytest ids come out as ``[cuda-...]``/``[cpu-...]`` (the
#: device parameter, not a guessed ``[..-cuda]`` suffix) -- select tests by
#: their collected id, not an assumed string shape.
_CUDA_DEVICE = pytest.param(
    "cuda",
    marks=pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA exact canary requires an allocated CUDA device",
    ),
)
_DEVICES = ("cpu", _CUDA_DEVICE)

pytestmark = pytest.mark.integration

# Contract analytic controls (`contract_inputs.json` `analytic_controls`).
# These bands apply to the full-grid manifest aggregates (all 65536 rows),
# not the final-draw JSONL metric: `E[r^-4]` diverges under the exact
# ground-state radial distribution (r ~ Gamma(3, 2.4) under |psi|^2), so the
# *sample* variance of E_L has infinite variance and is dominated by
# whichever walker lands closest to the nucleus. A ten-seed sweep (PR 592
# review round 1, Cannon job 58746188) measured the final-draw variance
# ranging 0.0395 to 0.0717 (-31% to +24% of 0.0576) and the full-grid
# variance ranging 0.0498 to 0.0649 (-14% to +13%) -- both too wide for a
# tight band. The full-grid mean (-0.4783 to -0.4812) and median (-0.5393
# to -0.5412, against the analytic ALPHA_1_2_MEDIAN) were robust across the
# same sweep, so those are the statistics pinned below; the final-draw
# JSONL metric is kept only as an exact recomputation from its own rows
# (see `test_imperfect_state_matches_closed_form_analytic_controls`), never
# compared to the analytic target directly. The W statistic (rel=0.3) was
# also robust (2.25 to 2.47) and is unaffected by this change.
#
# Exposure: these bands were chosen after the ten-seed sweep in review job
# 58746188, which also certifies them, while the analytic targets were
# known before measuring -- the bands are post hoc with respect to that
# sweep.
ALPHA_1_2_MEAN = -0.48
ALPHA_1_2_VARIANCE = 0.0576
ALPHA_1_2_MEDIAN = -0.540498
ALPHA_1_2_W = 2.339741

N_WALKERS = 1024
N_DRAWS = 64


def _load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _run_canary(tmp_path: Path, *, alpha: float, run_id: str, device: str = "cpu") -> dict[str, Any]:
    config = _load_config()
    config["model"]["alpha"] = alpha
    config["run"]["root"] = str(tmp_path)
    config["run"]["run_id"] = run_id
    config["runtime"]["device"] = device

    status = run_from_config(
        OmegaConf.create(config), command=f"h-v2 exact canary {run_id}", raise_exceptions=True
    )

    run_dir = Path(config["run"]["root"]) / config["experiment"]["name"] / config["experiment"]["sector"] / run_id
    return {"status": status, "run_dir": run_dir, "config": config}


def _read_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "trajectory" / "sampled_eval_table.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _assert_resolved_device(result: dict[str, Any], device: str) -> None:
    """Require the parametrized device to reach the real configured runner.

    A passing [cuda] id alone cannot detect a dropped device override: the
    YAML default is CPU, where all the same numerical assertions also pass.
    This checks the configuration artifact, not the location of every tensor.
    """

    resolved = yaml.safe_load((result["run_dir"] / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["runtime"]["device"] == device


def _read_trajectory_manifest(run_dir: Path) -> dict[str, Any]:
    """Read the streamed trajectory's own manifest (`_write_metadata` in
    `tpen.evaluation.trajectory_records`), which the collector also validates.

    Its ``mean``/``variance``/``row_count``/``finite_count`` are accumulated
    across every draw as the trajectory streams (`ObservableTrajectoryAccumulator`),
    not just the final-draw snapshot the JSONL `local_energy_mean`/`variance`
    metrics carry (`TrajectoryMCMCGenerator`'s evaluation batch is explicitly
    "its final retained draw" -- see that module's docstring), so this is the
    full-record-set aggregate to check bookkeeping against.
    """

    path = run_dir / "trajectory" / "sampled_eval_table.metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _full_grid_mean_and_variance(energies: list[float]) -> tuple[float, float]:
    """Population mean/variance over every record, matching
    `ObservableTrajectoryAccumulator`'s `total/n` and `total_square/n - mean**2`."""

    n = len(energies)
    mean = sum(energies) / n
    variance = sum((value - mean) ** 2 for value in energies) / n
    return mean, variance


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    midpoint = n // 2
    if n % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _final_draw_energies(rows: list[dict[str, str]], *, n_draws: int) -> list[float]:
    """Return `local_energy` for the rows `LocalEnergySummary`'s JSONL metric
    actually summarizes.

    `TrajectoryMCMCGenerator.generate` exposes the evaluation batch as "its
    final retained draw" (see that module's docstring); every calculator and
    summary in the task, including `LocalEnergySummary`, runs on that batch
    alone. So the JSONL `local_energy_mean`/`local_energy_variance` metrics
    are a snapshot over this one draw's records, not the full draw-by-walker
    grid -- a real, documented distinction, not a defect. Recomputing from
    exactly this same row subset (not the full grid) gives a tight,
    noise-free check: unlike the full grid vs. a single draw, this is the
    same population the metric claims to summarize, so it must agree to
    float64 tolerance, and a mutant that further subsets what
    `LocalEnergySummary` sees changes this recomputation even though it may
    hide inside the Monte Carlo band on the raw metric value.
    """

    final_draw_index = n_draws - 1
    return [float(row["local_energy"]) for row in rows if int(row["draw_index"]) == final_draw_index]


def _read_metric(run_dir: Path, *, namespace: str, key: str) -> float:
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    matches = [record for record in records if record["namespace"] == namespace and key in record["metrics"]]
    assert matches, f"no record under namespace={namespace!r} carries metric {key!r}"
    return float(matches[-1]["metrics"][key])


def _w_statistic(energies: list[float]) -> float:
    """Contract `W = P99.9(abs(E_L - median(E_L)))`, no interpolation."""

    sorted_energies = sorted(energies)
    n = len(sorted_energies)
    midpoint = n // 2
    median = sorted_energies[midpoint] if n % 2 else (sorted_energies[midpoint - 1] + sorted_energies[midpoint]) / 2.0
    deviations = sorted(abs(value - median) for value in energies)
    index = -(-999 * len(deviations) // 1000)  # ceil(0.999 * n), no interpolation
    index = min(index, len(deviations)) - 1
    return deviations[index]


def _collect_without_checkpoint(result: dict[str, Any]) -> dict[str, Any]:
    """Exercise the real collector's checkpoint-independent envelope checks.

    This canary carries no checkpoint (see module docstring and
    `test_no_configured_checkpoint_route_exists_for_the_parameterless_model`),
    so the composite `collect.collect_records` -- which requires a restored
    checkpoint directory and a `model_only` resolved restore mode -- cannot
    be called honestly here. Instead this calls the real, unmodified
    `collect.validate_run_identity`, `collect.validate_manifest`, and
    `collect.validate_rows` directly: source-commit identity, resolved-config
    identity, manifest completeness, and full per-row validation are all
    still genuinely exercised against the production collector code.
    `collect.validate_checkpoint_identity`/`collect.validate_restore_mode`
    are NOT exercised by this call -- a disclosed gap, covered instead by
    `test_collect.py`'s own unit tests against synthetic checkpoint
    artifacts.
    """

    run_dir = result["run_dir"]
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=HERE).strip()
    resolved_config_path = run_dir / "resolved_config.yaml"
    spec = collect.RecordSpec(
        source_sha=source_sha,
        checkpoint_model_sha256="not-applicable-no-configured-checkpoint-route",
        resolved_config_sha256=collect.expected_resolved_config_identity_from_path(resolved_config_path)["sha256"],
        walkers=N_WALKERS,
        draws=N_DRAWS,
    )
    run_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    collect.validate_run_identity(run_metadata, resolved_config_path, spec)

    csv_path = run_dir / "trajectory" / "sampled_eval_table.csv"
    manifest = _read_trajectory_manifest(result["run_dir"])
    collect.validate_manifest(manifest, csv_path, spec)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ordered = collect.validate_rows(rows, spec)
    return {"status": "complete_without_checkpoint_admission", "row_count": len(ordered)}


def test_no_configured_checkpoint_route_exists_for_the_parameterless_model(tmp_path: Path) -> None:
    """Confirm, by executing it, that no configured route can checkpoint
    `HydrogenExact`.

    The only configured route that writes a checkpoint is `tpen.runner.Train`.
    This builds a real `Train` config -- the same model, sampler, and
    Hamiltonian terms as the canary's own evaluation config, with a trivial
    partial `torch.optim.SGD` factory -- and drives it through
    `run_from_config`. `Train.run` unconditionally builds a real optimizer
    over `model.parameters()` before touching the trainer, sampler, or `load`
    mode (`tpen/runner/train.py`), and `HydrogenExact` has zero trainable
    parameters, so every standard `torch.optim` optimizer refuses on an empty
    parameter list, independent of `max_steps` or any other training
    configuration. This is the "exact error" the acceptance-contract
    amendment requires reporting rather than assuming (see
    `implementation-receipt`): there is no configured route -- `run_from_config`
    or the `tpen` command -- that can produce a checkpoint for this module.
    """

    config = _load_config()
    config["run"]["root"] = str(tmp_path)
    config["run"]["run_id"] = "configured-route-probe"
    config.pop("evaluator", None)
    config["runner"] = {
        "_target_": "tpen.runner.Train",
        "model": "${model}",
        "sampler": "${sampler}",
        "hamiltonian_terms": "${hamiltonian_terms}",
        "optimizer": {"_target_": "torch.optim.SGD", "_partial_": True, "lr": 0.01},
        "trainer": None,
    }

    cfg = OmegaConf.create(config)
    with pytest.raises(ValueError, match="empty parameter list"):
        run_from_config(cfg, command="h-v2 exact canary configured-route probe", raise_exceptions=True)


@pytest.mark.parametrize("device", _DEVICES)
def test_exact_ground_state_every_record_is_minus_half_hartree(tmp_path: Path, device: str) -> None:
    result = _run_canary(tmp_path, alpha=1.0, run_id=f"exact-alpha-1-{device}", device=device)
    assert result["status"] == 0
    _assert_resolved_device(result, device)
    rows = _read_rows(result["run_dir"])
    assert len(rows) == 1024 * 64
    energies = [float(row["local_energy"]) for row in rows]
    assert all(row["finite"] == "True" for row in rows)
    for energy in energies:
        assert energy == pytest.approx(-0.5, abs=1e-6)

    mean = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_mean")
    variance = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_variance")
    assert mean == pytest.approx(-0.5, abs=1e-6)
    assert variance == pytest.approx(0.0, abs=1e-8)
    assert _w_statistic(energies) == pytest.approx(0.0, abs=1e-6)

    # Exact bookkeeping against the exact same population the JSONL metric
    # claims to summarize (the final retained draw only, see
    # `_final_draw_energies`). At alpha=1 every value is the same constant,
    # so this specific check cannot by itself distinguish a subset of this
    # draw from the whole of it -- that is a mathematical fact about a
    # constant signal, not a gap in the check -- but it still catches any
    # other aggregation defect (wrong draw selected, wrong row count, wrong
    # formula), and the alpha=1.2 test below closes the subset gap.
    final_draw_energies = _final_draw_energies(rows, n_draws=N_DRAWS)
    assert len(final_draw_energies) == N_WALKERS
    final_draw_mean, final_draw_variance = _full_grid_mean_and_variance(final_draw_energies)
    assert mean == pytest.approx(final_draw_mean, rel=1e-10, abs=1e-12)
    assert variance == pytest.approx(final_draw_variance, rel=1e-10, abs=1e-12)

    # Exact bookkeeping, not a Monte Carlo band: the trajectory manifest's own
    # mean/variance/row_count/finite_count are accumulated across every draw
    # as the grid streams, so they must equal an independent recomputation
    # from the full set of recorded rows to tight float64 tolerance. A
    # summary aggregated on a subset of the records (e.g. one draw instead of
    # all 64) would not match this.
    manifest = _read_trajectory_manifest(result["run_dir"])
    assert manifest["row_count"] == 1024 * 64
    assert manifest["finite_count"] == 1024 * 64
    full_mean, full_variance = _full_grid_mean_and_variance(energies)
    assert manifest["mean"] == pytest.approx(full_mean, rel=1e-10, abs=1e-12)
    assert manifest["variance"] == pytest.approx(full_variance, rel=1e-10, abs=1e-12)

    receipt = _collect_without_checkpoint(result)
    assert receipt["status"] == "complete_without_checkpoint_admission"
    assert receipt["row_count"] == 1024 * 64


@pytest.fixture(scope="module")
def _alpha_1_2_moment_band() -> Any:
    """Calibrate once and reuse across the cpu/cuda parametrizations below.

    `sample_count=N_WALKERS` matches the canary's final retained draw's
    independent walker count exactly, per the acceptance contract.
    """

    return moment_bands.calibrate_moment_band(alpha=1.2, sample_count=N_WALKERS, seed=120001)


@pytest.mark.parametrize("device", _DEVICES)
def test_imperfect_state_matches_closed_form_analytic_controls(
    tmp_path: Path, device: str, _alpha_1_2_moment_band: Any,
) -> None:
    alpha = 1.2
    result = _run_canary(tmp_path, alpha=alpha, run_id=f"exact-alpha-1-2-{device}", device=device)
    assert result["status"] == 0
    _assert_resolved_device(result, device)
    rows = _read_rows(result["run_dir"])
    energies = [float(row["local_energy"]) for row in rows]
    assert all(row["finite"] == "True" for row in rows)

    # The JSONL trajectory metric is a final-draw snapshot (see module
    # docstring). Its sample variance has no Gaussian error bar at this
    # alpha (see the comment above `ALPHA_1_2_MEAN`); below it is checked
    # both as an exact recomputation from its own rows and against the
    # calibrated moment band, never against the analytic target directly.
    mean = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_mean")
    variance = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_variance")
    assert _w_statistic(energies) == pytest.approx(ALPHA_1_2_W, rel=0.3)

    # Exact bookkeeping against the exact same population the JSONL metric
    # summarizes (the final retained draw, see `_final_draw_energies`). Here
    # alpha=1.2 makes the local energy position-dependent, so unlike the
    # alpha=1 case this recomputation IS sensitive to a summary silently
    # aggregated over a further subset of that draw (e.g. half its walkers):
    # such a mutant can still fall inside a Monte Carlo band on the raw
    # metric, but changes this recomputed value measurably, well outside
    # float64 rounding.
    final_draw_energies = _final_draw_energies(rows, n_draws=N_DRAWS)
    assert len(final_draw_energies) == N_WALKERS
    final_draw_mean, final_draw_variance = _full_grid_mean_and_variance(final_draw_energies)
    assert mean == pytest.approx(final_draw_mean, rel=1e-10, abs=1e-12)
    assert variance == pytest.approx(final_draw_variance, rel=1e-10, abs=1e-12)

    # Calibrated moment band (contract f5068c1a): the final-draw mean and
    # variance checked above only against their own recomputation, with no
    # band on their accuracy. `_alpha_1_2_moment_band` is calibrated at
    # `sample_count=N_WALKERS`, matching this draw's independent walker
    # count exactly (walkers are independent after burn-in; successive
    # draws of one walker are not, so only a walker count is a valid
    # calibration sample size here). The closed-form moments must lie
    # inside the same band.
    assert _alpha_1_2_moment_band[0, 0] <= ALPHA_1_2_MEAN <= _alpha_1_2_moment_band[1, 0]
    assert _alpha_1_2_moment_band[0, 1] <= ALPHA_1_2_VARIANCE <= _alpha_1_2_moment_band[1, 1]
    assert _alpha_1_2_moment_band[0, 0] <= final_draw_mean <= _alpha_1_2_moment_band[1, 0]
    assert _alpha_1_2_moment_band[0, 1] <= final_draw_variance <= _alpha_1_2_moment_band[1, 1]

    # Exact bookkeeping, separate from the Monte Carlo bands: the
    # summary mean/variance must equal the mean/variance of the recorded
    # rows, recomputed independently from the full record set. The analytic
    # bands below apply here, to the full-grid aggregate, which the seed
    # sweep showed to be robust (see the comment above `ALPHA_1_2_MEAN`).
    # These whole-grid comparisons keep their existing whole-walker
    # uncertainty and are unchanged by the calibrated band above, which
    # applies only to the final-draw moments.
    manifest = _read_trajectory_manifest(result["run_dir"])
    assert manifest["row_count"] == 1024 * 64
    assert manifest["finite_count"] == 1024 * 64
    full_mean, full_variance = _full_grid_mean_and_variance(energies)
    assert manifest["mean"] == pytest.approx(full_mean, rel=1e-10, abs=1e-12)
    assert manifest["variance"] == pytest.approx(full_variance, rel=1e-10, abs=1e-12)
    assert full_mean == pytest.approx(ALPHA_1_2_MEAN, abs=0.01)
    assert full_variance == pytest.approx(ALPHA_1_2_VARIANCE, rel=0.25)
    assert _median(energies) == pytest.approx(ALPHA_1_2_MEDIAN, abs=0.005)

    # Precision: recompute the oracle local energy pointwise from each
    # record's own recorded position (`-alpha^2/2 + (alpha-1)/r`) and require
    # float64-level agreement. `HydrogenExact`'s local energy is this closed
    # form exactly at every walker, independent of position, so any
    # deviation beyond float64 rounding -- e.g. a float32 downcast of
    # `local_energy` anywhere in the record path, which would show up as a
    # ~1e-7 relative discrepancy -- must fail this check.
    for row in rows:
        coordinates = [float(row[f"position/electron_0/axis_{axis}"]) for axis in range(3)]
        radius = math.sqrt(sum(coordinate**2 for coordinate in coordinates))
        oracle_energy = -(alpha**2) / 2.0 + (alpha - 1.0) / radius
        assert float(row["local_energy"]) == pytest.approx(oracle_energy, rel=1e-10, abs=1e-10)

    receipt = _collect_without_checkpoint(result)
    assert receipt["status"] == "complete_without_checkpoint_admission"
    assert receipt["row_count"] == 1024 * 64


@pytest.fixture(scope="module")
def _alpha_one_rows(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, str]]:
    """One alpha=1 canary run, shared by the collector-mutant tests below.

    Both tests hand-edit a copy of these rows with identities the collector
    rejects regardless of the underlying local-energy values, so the
    assertions do not depend on a fresh evaluation per test; a single
    module-scoped run suffices (F9).
    """

    tmp_path = tmp_path_factory.mktemp("alpha-one-collector-source")
    result = _run_canary(tmp_path, alpha=1.0, run_id="mutant-source")
    return _read_rows(result["run_dir"])


def test_dropped_record_mutant_is_refused_by_the_collector(_alpha_one_rows: list[dict[str, str]]) -> None:
    rows = [dict(row) for row in _alpha_one_rows]
    rows.pop()
    with pytest.raises(collect.RecordIntegrityError, match="missing draw"):
        collect.validate_rows(
            rows,
            collect.RecordSpec(
                source_sha="irrelevant",
                checkpoint_model_sha256="irrelevant",
                resolved_config_sha256="irrelevant",
                walkers=1024,
                draws=64,
            ),
        )


def test_duplicated_record_mutant_is_refused_by_the_collector(_alpha_one_rows: list[dict[str, str]]) -> None:
    rows = [dict(row) for row in _alpha_one_rows]
    rows[1] = dict(rows[0])
    with pytest.raises(collect.RecordIntegrityError, match="duplicate"):
        collect.validate_rows(
            rows,
            collect.RecordSpec(
                source_sha="irrelevant",
                checkpoint_model_sha256="irrelevant",
                resolved_config_sha256="irrelevant",
                walkers=1024,
                draws=64,
            ),
        )
