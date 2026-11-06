"""End-to-end evaluation-route canary for the exact one-electron hydrogen state.

Drives `tpen.physics.hydrogen.HydrogenExact` through the real evaluation
route: `tpen.run.run_from_config` into the real `Evaluate` runner, a genuine
model-only checkpoint restore, `TrajectoryMCMCGenerator`,
`LocalEnergyCalculator`/`WavefunctionCalculator`, `SampledRecordWriter` and
`LocalEnergySummary`, and finally the real H-v2 collector (`collect.py`).

Neither `test_oracle_vs_tpen.py` nor `test_stub_oracle_contract.py` touches
this route: both feed a stub wavefunction straight to the Hamiltonian terms
and sampler, never through `run_from_config`, the trajectory record writer,
the summaries, or the collector (see container note
``canary-overlap-assessment-20261106``). An evaluation-path defect -- a
wrong aggregation, a dropped or duplicated record, a term mis-assigned
between kinetic and potential, or a summary computed on a subset -- cannot be
told apart from model behavior on a learned model, whose true local energy is
unknown. On the exact hydrogen ground state (alpha=1) every local energy must
equal -0.5 Ha exactly, so any such defect is directly visible.

**Restore scope.** `HydrogenExact` carries no trainable state, so there is no
optimizer/trainer/sampler state a *training*-resume checkpoint would add (an
``nn.Module`` with zero parameters cannot even build an optimizer --
``torch.optim.Adam([])`` raises). The canary therefore cannot go through the
`Train` runner the way a learned checkpoint does. What it does instead, and
what the real collector (`collect.py`) requires regardless, is build a
genuine model-only checkpoint directly with
`tpen.checkpoint.save.save_checkpoint` -- the same function, directory
format, and manifest a learned run's checkpoint uses -- and restore it
through the same `load:` block and the real `Evaluate` runner
(`tpen.checkpoint.restore.restore_checkpoint`). This exercises the real
model-only restore path honestly; it does not exercise optimizer/trainer/
sampler/RNG restore, because this module has none of that state to restore.

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
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.checkpoint.save import save_checkpoint
from tpen.physics.hamiltonian import LocalEnergyResult
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronNucleusPotential
from tpen.run import run_from_config

HERE = Path(__file__).parent
CONFIG_PATH = HERE / "configs" / "exact_canary_eval.yaml"

_collect_spec = importlib.util.spec_from_file_location("h_v2_exact_canary_collect", HERE / "collect.py")
assert _collect_spec is not None and _collect_spec.loader is not None
collect = importlib.util.module_from_spec(_collect_spec)
sys.modules[_collect_spec.name] = collect
_collect_spec.loader.exec_module(collect)

pytestmark = pytest.mark.integration

# Contract analytic controls (`contract_inputs.json` `analytic_controls`).
ALPHA_1_2_MEAN = -0.48
ALPHA_1_2_VARIANCE = 0.0576
ALPHA_1_2_W = 2.339741

N_WALKERS = 1024
N_DRAWS = 64


def _load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _build_model_only_checkpoint(tmp_path: Path, cfg: Any) -> Path:
    """Save a genuine model-only checkpoint of the exact, parameterless module."""

    model = instantiate(cfg.model)
    artifact_manager = ArtifactManager(
        tmp_path / "checkpoint-source",
        experiment="h_v2_exact_canary",
        sector="atomistic",
        run_id="checkpoint-source",
        layout="flat",
    )
    artifact_manager.make_dirs()
    metadata = RunMetadata(
        run_id="checkpoint-source",
        run_name="checkpoint-source",
        timestamp="2026-11-06T00:00:00+00:00",
        timezone="UTC",
        git_commit="checkpoint-source",
        git_branch="agent/h-v2-exact-canary",
        dirty_worktree=False,
        command="pytest",
        config_path=str(CONFIG_PATH),
        resolved_config_path=str(artifact_manager.path("resolved_config.yaml")),
        run_dir=str(artifact_manager.run_dir),
        device="cpu",
        dtype="float64",
    )
    context = RunContext(
        cfg=cfg,
        source_cfg=cfg,
        artifact_manager=artifact_manager,
        metadata=metadata,
        clock=RunClock(timezone="UTC", tzinfo=UTC),
    )
    return save_checkpoint(
        output_dir=tmp_path / "checkpoint-source" / "checkpoints",
        next_iteration=0,
        completed_updates=0,
        model=model,
        context=context,
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )


def _run_canary(tmp_path: Path, *, alpha: float, run_id: str) -> dict[str, Any]:
    config = _load_config()
    config["model"]["alpha"] = alpha
    config["run"]["root"] = str(tmp_path)
    config["run"]["run_id"] = run_id

    cfg = OmegaConf.create(config)
    checkpoint_dir = _build_model_only_checkpoint(tmp_path, cfg)
    config["load"]["path"] = str(checkpoint_dir)

    status = run_from_config(
        OmegaConf.create(config), command=f"h-v2 exact canary {run_id}", raise_exceptions=True
    )

    run_dir = Path(config["run"]["root"]) / config["experiment"]["name"] / config["experiment"]["sector"] / run_id
    return {
        "status": status,
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "config": config,
    }


def _read_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "trajectory" / "sampled_eval_table.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _collect(tmp_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    run_dir = result["run_dir"]
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=HERE).strip()
    spec = collect.RecordSpec(
        source_sha=source_sha,
        checkpoint_model_sha256=collect.file_sha256(result["checkpoint_dir"] / "model.pt"),
        resolved_config_sha256=collect.file_sha256(run_dir / "resolved_config.yaml"),
        walkers=1024,
        draws=64,
    )
    return collect.collect_records(
        csv_path=run_dir / "trajectory" / "sampled_eval_table.csv",
        metadata_path=run_dir / "trajectory" / "sampled_eval_table.metadata.json",
        resolved_config_path=run_dir / "resolved_config.yaml",
        run_metadata_path=run_dir / "metadata.json",
        checkpoint_dir=result["checkpoint_dir"],
        spec=spec,
    )


def test_exact_ground_state_every_record_is_minus_half_hartree(tmp_path: Path) -> None:
    result = _run_canary(tmp_path, alpha=1.0, run_id="exact-alpha-1")
    assert result["status"] == 0
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

    receipt = _collect(tmp_path, result)
    assert receipt["status"] == "complete"
    assert receipt["row_count"] == 1024 * 64


def test_imperfect_state_matches_closed_form_analytic_controls(tmp_path: Path) -> None:
    alpha = 1.2
    result = _run_canary(tmp_path, alpha=alpha, run_id="exact-alpha-1-2")
    assert result["status"] == 0
    rows = _read_rows(result["run_dir"])
    energies = [float(row["local_energy"]) for row in rows]
    assert all(row["finite"] == "True" for row in rows)

    mean = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_mean")
    variance = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_variance")
    # Predeclared Monte Carlo bands: generous relative to the serially
    # correlated 1024-walker x 64-draw MCMC sample (not IID), looser than the
    # IID-sample tolerances in `test_stub_oracle_contract.py`.
    assert mean == pytest.approx(ALPHA_1_2_MEAN, abs=0.02)
    assert variance == pytest.approx(ALPHA_1_2_VARIANCE, rel=0.15)
    assert _w_statistic(energies) == pytest.approx(ALPHA_1_2_W, rel=0.3)

    # Exact bookkeeping against the exact same population the JSONL metric
    # summarizes (the final retained draw, see `_final_draw_energies`). Here
    # alpha=1.2 makes the local energy position-dependent, so unlike the
    # alpha=1 case this recomputation IS sensitive to a summary silently
    # aggregated over a further subset of that draw (e.g. half its walkers):
    # such a mutant can still fall inside the Monte Carlo band on the raw
    # metric above, but changes this recomputed value measurably, well
    # outside float64 rounding.
    final_draw_energies = _final_draw_energies(rows, n_draws=N_DRAWS)
    assert len(final_draw_energies) == N_WALKERS
    final_draw_mean, final_draw_variance = _full_grid_mean_and_variance(final_draw_energies)
    assert mean == pytest.approx(final_draw_mean, rel=1e-10, abs=1e-12)
    assert variance == pytest.approx(final_draw_variance, rel=1e-10, abs=1e-12)

    # Exact bookkeeping, separate from the Monte Carlo band above: the
    # summary mean/variance must equal the mean/variance of the recorded
    # rows, recomputed independently from the full record set.
    manifest = _read_trajectory_manifest(result["run_dir"])
    assert manifest["row_count"] == 1024 * 64
    assert manifest["finite_count"] == 1024 * 64
    full_mean, full_variance = _full_grid_mean_and_variance(energies)
    assert manifest["mean"] == pytest.approx(full_mean, rel=1e-10, abs=1e-12)
    assert manifest["variance"] == pytest.approx(full_variance, rel=1e-10, abs=1e-12)

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

    receipt = _collect(tmp_path, result)
    assert receipt["status"] == "complete"
    assert receipt["row_count"] == 1024 * 64


def test_wrong_kinetic_prefactor_mutant_breaks_the_canary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong kinetic prefactor must be visible on the exact ground state."""

    real_local_energy = KineticEnergy.local_energy

    def _wrong_prefactor_local_energy(self, wavefunction, batch):
        result = real_local_energy(self, wavefunction, batch)
        # Wrong kinetic prefactor: -1.0 instead of the real -0.5. Preserve
        # `wavefunction_output`/`per_electron_kinetic`: the trajectory
        # generator requires exactly one term to supply the signed-log
        # output (it differentiates the model once and reuses the byproduct
        # rather than re-evaluating it), so dropping it here would fail the
        # generator itself rather than exercising the intended mutation.
        mutated = 2.0 * result.total
        return LocalEnergyResult(
            total=mutated,
            terms={self.name: mutated},
            wavefunction_output=result.wavefunction_output,
            per_electron_kinetic=result.per_electron_kinetic,
        )

    monkeypatch.setattr(KineticEnergy, "local_energy", _wrong_prefactor_local_energy)
    result = _run_canary(tmp_path, alpha=1.0, run_id="mutant-kinetic-prefactor")
    assert result["status"] == 0
    mean = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_mean")
    assert mean != pytest.approx(-0.5, abs=1e-6)
    assert abs(mean - (-0.5)) > 0.1


def test_sign_flipped_potential_mutant_breaks_the_canary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sign-flipped nuclear potential must be visible on the exact ground state."""

    real_local_energy = ElectronNucleusPotential.local_energy

    def _flipped_sign_local_energy(self, wavefunction, batch):
        result = real_local_energy(self, wavefunction, batch)
        mutated = -result.total
        return LocalEnergyResult(total=mutated, terms={self.name: mutated})

    monkeypatch.setattr(ElectronNucleusPotential, "local_energy", _flipped_sign_local_energy)
    result = _run_canary(tmp_path, alpha=1.0, run_id="mutant-potential-sign")
    assert result["status"] == 0
    mean = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_mean")
    assert mean != pytest.approx(-0.5, abs=1e-6)
    assert abs(mean - (-0.5)) > 0.1


def test_dropped_record_mutant_is_refused_by_the_collector(tmp_path: Path) -> None:
    result = _run_canary(tmp_path, alpha=1.0, run_id="mutant-dropped-record")
    rows = _read_rows(result["run_dir"])
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


def test_duplicated_record_mutant_is_refused_by_the_collector(tmp_path: Path) -> None:
    result = _run_canary(tmp_path, alpha=1.0, run_id="mutant-duplicated-record")
    rows = _read_rows(result["run_dir"])
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
