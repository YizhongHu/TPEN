"""Core integration test: exact one-electron hydrogen through the real
evaluation route.

Drives `tpen.physics.hydrogen.HydrogenExact` through `tpen.run.run_from_config`
into the real `Evaluate` runner, a genuine model-only checkpoint restore,
`TrajectoryMCMCGenerator`, `LocalEnergyCalculator`/`WavefunctionCalculator`,
`SampledRecordWriter`, and `LocalEnergySummary`. On the exact hydrogen ground
state (`alpha=1`) every local energy must equal -0.5 Ha exactly, so an
evaluation-path defect -- a wrong aggregation, a wrong kinetic prefactor, or a
flipped potential sign -- is directly visible, independent of model behavior.

This file owns the model-only restore coverage for the H-v2 exact-hydrogen
canary: `HydrogenExact` carries no trainable state, so there is no
optimizer/trainer/sampler state a *training*-resume checkpoint would add (an
`nn.Module` with zero parameters cannot even build an optimizer --
`torch.optim.SGD([])` raises). This test builds a genuine model-only
checkpoint directly with `tpen.checkpoint.save.save_checkpoint` -- the same
function, directory format, and manifest a learned run's checkpoint uses --
and restores it through the same `load:` block and the real `Evaluate` runner
(`tpen.checkpoint.restore.restore_checkpoint`). This exercises the real
model-only restore path honestly; it does not exercise optimizer/trainer/
sampler/RNG restore, because this module has none of that state to restore.
The colocated H-v2 study canary (`experiments/atomistic/h-v2/
test_hydrogen_exact_canary.py`) cannot do this itself: it may reach core only
through configuration, `run_from_config`, or the `tpen` command, and no
configured route can produce a checkpoint for a zero-parameter model (see that
file's own probe test for the exact error).

Core integration tests may import tpen internals freely (unlike the
experiment code under `experiments/`), so this file is not subject to the
H-v2 experiment import boundary. It imports no file under `experiments/` and
never reads or calls the study collector (`collect.py`).

**Full-record-set bookkeeping.** The streamed trajectory manifest
(`sampled_eval_table.metadata.json`) accumulates `mean`/`variance`/
`row_count`/`finite_count` across every retained draw, not just the final
one. `TrajectoryMCMCGenerator`'s evaluation batch, by contrast, is explicitly
"its final retained draw" (see that generator's own docstring), so the JSONL
`local_energy_mean`/`local_energy_variance` metrics `LocalEnergySummary`
publishes are a final-draw snapshot, not a full-grid aggregate -- a real,
documented distinction, not a defect. Both tests below check the manifest's
own aggregates against an independent recomputation from every row in
`sampled_eval_table.csv`, and the JSONL metric against a recomputation from
only the rows of its own final draw (`_final_draw_energies`) -- the same
population it claims to summarize.

**Record-path precision.** `HydrogenExact`'s local energy is the closed form
`-alpha^2/2 + (alpha-1)/r` exactly at every walker, independent of position.
At `alpha=1.2` the imperfect-state test recomputes this pointwise from each
record's own recorded position and requires float64-level agreement
(`rel=1e-10`), which a float32 downcast anywhere in the record path (~1e-7
relative error) would fail.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.checkpoint.save import save_checkpoint
from tpen.physics.hamiltonian import LocalEnergyResult
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronNucleusPotential
from tpen.run import run_from_config

HERE = Path(__file__).parent
FIXTURES = HERE.resolve().parents[1] / "artifacts" / "hydrogen"
CONFIG_PATH = FIXTURES / "exact_eval.yaml"

# Analytic controls for the `alpha=1.2` imperfect state:
# E_L = -alpha^2/2 + (alpha-1)/r, with r ~ the exact-ground-state radial
# distribution. Mirrors the H-v2 contract's `analytic_controls`.
ALPHA_1_2_MEAN = -0.48
ALPHA_1_2_VARIANCE = 0.0576

N_WALKERS = 1024
N_DRAWS = 64


def _build_model_only_checkpoint(tmp_path: Path, cfg: Any) -> Path:
    """Save a genuine model-only checkpoint of the exact, parameterless module."""

    model = instantiate(cfg.model)
    artifact_manager = ArtifactManager(
        tmp_path / "checkpoint-source",
        experiment="hydrogen_exact",
        sector="atomistic",
        run_id="checkpoint-source",
        layout="flat",
    )
    artifact_manager.make_dirs()
    metadata = RunMetadata(
        run_id="checkpoint-source",
        run_name="checkpoint-source",
        timestamp="2026-11-07T00:00:00+00:00",
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


def _load_config() -> dict[str, Any]:
    cfg = OmegaConf.load(CONFIG_PATH)
    value = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(value, dict)
    return value


def _run(tmp_path: Path, *, alpha: float, run_id: str) -> dict[str, Any]:
    config = _load_config()
    config["model"]["alpha"] = alpha
    config["run"]["root"] = str(tmp_path)
    config["run"]["run_id"] = run_id

    cfg = OmegaConf.create(config)
    checkpoint_dir = _build_model_only_checkpoint(tmp_path, cfg)
    config["load"]["path"] = str(checkpoint_dir)

    status = run_from_config(
        OmegaConf.create(config), command=f"hydrogen exact eval {run_id}", raise_exceptions=True
    )

    run_dir = Path(config["run"]["root"]) / config["experiment"]["name"] / config["experiment"]["sector"] / run_id
    return {"status": status, "run_dir": run_dir}


def _read_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "trajectory" / "sampled_eval_table.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_trajectory_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "trajectory" / "sampled_eval_table.metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _full_grid_mean_and_variance(energies: list[float]) -> tuple[float, float]:
    n = len(energies)
    mean = sum(energies) / n
    variance = sum((value - mean) ** 2 for value in energies) / n
    return mean, variance


def _final_draw_energies(rows: list[dict[str, str]], *, n_draws: int) -> list[float]:
    final_draw_index = n_draws - 1
    return [float(row["local_energy"]) for row in rows if int(row["draw_index"]) == final_draw_index]


def _read_metric(run_dir: Path, *, namespace: str, key: str) -> float:
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    matches = [record for record in records if record["namespace"] == namespace and key in record["metrics"]]
    assert matches, f"no record under namespace={namespace!r} carries metric {key!r}"
    return float(matches[-1]["metrics"][key])


def test_exact_ground_state_every_record_is_minus_half_hartree(tmp_path: Path) -> None:
    result = _run(tmp_path, alpha=1.0, run_id="exact-alpha-1")
    assert result["status"] == 0
    rows = _read_rows(result["run_dir"])
    assert len(rows) == N_WALKERS * N_DRAWS
    energies = [float(row["local_energy"]) for row in rows]
    assert all(row["finite"] == "True" for row in rows)
    for energy in energies:
        assert energy == pytest.approx(-0.5, abs=1e-6)

    mean = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_mean")
    variance = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_variance")
    assert mean == pytest.approx(-0.5, abs=1e-6)
    assert variance == pytest.approx(0.0, abs=1e-8)

    final_draw_energies = _final_draw_energies(rows, n_draws=N_DRAWS)
    assert len(final_draw_energies) == N_WALKERS
    final_draw_mean, final_draw_variance = _full_grid_mean_and_variance(final_draw_energies)
    assert mean == pytest.approx(final_draw_mean, rel=1e-10, abs=1e-12)
    assert variance == pytest.approx(final_draw_variance, rel=1e-10, abs=1e-12)

    manifest = _read_trajectory_manifest(result["run_dir"])
    assert manifest["row_count"] == N_WALKERS * N_DRAWS
    assert manifest["finite_count"] == N_WALKERS * N_DRAWS
    full_mean, full_variance = _full_grid_mean_and_variance(energies)
    assert manifest["mean"] == pytest.approx(full_mean, rel=1e-10, abs=1e-12)
    assert manifest["variance"] == pytest.approx(full_variance, rel=1e-10, abs=1e-12)


def test_imperfect_state_matches_closed_form_analytic_controls(tmp_path: Path) -> None:
    alpha = 1.2
    result = _run(tmp_path, alpha=alpha, run_id="exact-alpha-1-2")
    assert result["status"] == 0
    rows = _read_rows(result["run_dir"])
    energies = [float(row["local_energy"]) for row in rows]
    assert all(row["finite"] == "True" for row in rows)

    mean = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_mean")
    variance = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_variance")
    assert mean == pytest.approx(ALPHA_1_2_MEAN, abs=0.02)
    assert variance == pytest.approx(ALPHA_1_2_VARIANCE, rel=0.15)

    final_draw_energies = _final_draw_energies(rows, n_draws=N_DRAWS)
    assert len(final_draw_energies) == N_WALKERS
    final_draw_mean, final_draw_variance = _full_grid_mean_and_variance(final_draw_energies)
    assert mean == pytest.approx(final_draw_mean, rel=1e-10, abs=1e-12)
    assert variance == pytest.approx(final_draw_variance, rel=1e-10, abs=1e-12)

    manifest = _read_trajectory_manifest(result["run_dir"])
    assert manifest["row_count"] == N_WALKERS * N_DRAWS
    assert manifest["finite_count"] == N_WALKERS * N_DRAWS
    full_mean, full_variance = _full_grid_mean_and_variance(energies)
    assert manifest["mean"] == pytest.approx(full_mean, rel=1e-10, abs=1e-12)
    assert manifest["variance"] == pytest.approx(full_variance, rel=1e-10, abs=1e-12)

    # Precision: recompute the oracle local energy pointwise from each
    # record's own recorded position and require float64-level agreement. A
    # float32 downcast anywhere in the record path (~1e-7 relative error)
    # would fail this.
    for row in rows:
        coordinates = [float(row[f"position/electron_0/axis_{axis}"]) for axis in range(3)]
        radius = math.sqrt(sum(coordinate**2 for coordinate in coordinates))
        oracle_energy = -(alpha**2) / 2.0 + (alpha - 1.0) / radius
        assert float(row["local_energy"]) == pytest.approx(oracle_energy, rel=1e-10, abs=1e-10)


def test_wrong_kinetic_prefactor_mutant_breaks_the_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong kinetic prefactor must be visible on the exact ground state."""

    real_local_energy = KineticEnergy.local_energy

    def _wrong_prefactor_local_energy(self, wavefunction, batch):
        result = real_local_energy(self, wavefunction, batch)
        # Wrong kinetic prefactor: -1.0 instead of the real -0.5. Preserve
        # `wavefunction_output`/`per_electron_kinetic`: the trajectory
        # generator requires exactly one term to supply the signed-log output
        # (it differentiates the model once and reuses the byproduct rather
        # than re-evaluating it), so dropping it here would fail the
        # generator itself rather than exercising the intended mutation.
        mutated = 2.0 * result.total
        return LocalEnergyResult(
            total=mutated,
            terms={self.name: mutated},
            wavefunction_output=result.wavefunction_output,
            per_electron_kinetic=result.per_electron_kinetic,
        )

    monkeypatch.setattr(KineticEnergy, "local_energy", _wrong_prefactor_local_energy)
    result = _run(tmp_path, alpha=1.0, run_id="mutant-kinetic-prefactor")
    assert result["status"] == 0
    mean = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_mean")
    assert abs(mean - (-0.5)) > 0.1


def test_sign_flipped_potential_mutant_breaks_the_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sign-flipped nuclear potential must be visible on the exact ground state."""

    real_local_energy = ElectronNucleusPotential.local_energy

    def _flipped_sign_local_energy(self, wavefunction, batch):
        result = real_local_energy(self, wavefunction, batch)
        mutated = -result.total
        return LocalEnergyResult(total=mutated, terms={self.name: mutated})

    monkeypatch.setattr(ElectronNucleusPotential, "local_energy", _flipped_sign_local_energy)
    result = _run(tmp_path, alpha=1.0, run_id="mutant-potential-sign")
    assert result["status"] == 0
    mean = _read_metric(result["run_dir"], namespace="trajectory", key="local_energy_mean")
    assert abs(mean - (-0.5)) > 0.1
