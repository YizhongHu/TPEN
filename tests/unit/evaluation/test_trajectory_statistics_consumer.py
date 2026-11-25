"""Contracts for the trajectory-statistics consumer wiring.

Every test that claims an end-to-end result drives a *real* gradient-requiring
local energy: a real `KineticEnergy` term differentiating a real model's
``logabs`` twice with respect to positions, over a real serially-correlated
walker chain. Nothing here asserts against a hand-built receipt, because a
hand-built record would prove only that the assertion matches the fixture.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import pytest
import torch

from tpen.checkpoint.hashing import file_sha256
from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.batch.geometry import electron_nuclear_displacements, electron_nuclear_distances, pairwise_distances
from tpen.data.batch.walkers import Walkers
from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations
from tpen.evaluation.calculators import LocalEnergyCalculator, WavefunctionCalculator
from tpen.evaluation.generators import (
    SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY,
    TRAJECTORY_METADATA_KEY,
    TrajectoryMCMCGenerator,
)
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.summaries import (
    LocalEnergySummary,
    SampledRecordWriter,
    SamplerStatsSummary,
    TrajectoryStatisticsSummary,
)
from tpen.evaluation.summaries.metadata import SAMPLER_TRAJECTORY_DIAGNOSTICS_FILENAME
from tpen.evaluation.summaries.trajectory_statistics import DEFAULT_SIDECAR_NAME
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, ElectronNucleusInteraction
from tpen.sampling.stats import SamplerStats
from tpen.statistics import (
    ObservableTrajectory,
    TrajectoryStatisticsIdentity,
    TrajectoryStatisticsSidecar,
)

STAGE = "eval"
RUN_ID = "he-v1-test"
ATTEMPT_ID = "attempt-1"
EVALUATOR_ID = "he_mcse/v1"
CONFIG = {"system": {"name": "helium"}, "evaluation": {"namespace": "eval"}}


class HeliumSlaterModel(torch.nn.Module):
    """``log|psi| = -alpha * sum_i r_i`` with a trainable ``alpha``.

    Real and twice-differentiable: the kinetic term takes two autograd
    derivatives of ``logabs`` with respect to positions, so this model exercises
    the gradient path rather than standing in for it. The parameter exists so
    the collector's freeze-and-fingerprint contract has something to act on.
    """

    def __init__(self, alpha: float = 1.8) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float64))
        self.forward_count = 0

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        self.forward_count += 1
        distance = electron_nuclear_displacements(batch).norm(dim=-1)
        logabs = -self.alpha * distance.sum(dim=(-2, -1))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class CorrelatedHeliumSampler:
    """Persistent AR(1) walker chain around the helium nucleus.

    Positions carry over between calls and move by a small increment, so
    consecutive draws are genuinely serially correlated -- which is the whole
    property the autocorrelation estimate is supposed to detect. A sampler that
    redrew independently every call would make ``tau_int`` approximately one and
    the test would pass without exercising anything.
    """

    def __init__(
        self,
        *,
        n_walkers: int = 4,
        n_steps: int = 2,
        rho: float = 0.85,
        seed: int = 11,
        acceptance_rates: tuple[float, ...] | None = None,
    ) -> None:
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.rho = rho
        self.generator = torch.Generator().manual_seed(seed)
        # A2 replaced the partial nuclear_positions/nuclear_charges pair with one
        # typed value. AtomicConfiguration is immutable and carried by reference,
        # so it is built once here and never reconstructed per draw.
        self.atomic_configuration = AtomicConfiguration(
            positions=torch.zeros(1, 3, dtype=torch.float64),
            charges=torch.tensor([2.0], dtype=torch.float64),
        )
        # Start well away from the nucleus: the Coulomb terms diverge at r = 0
        # and a non-finite draw would be reported as `unresolved`, masking the
        # behaviour under test.
        self.positions = 0.9 + 0.2 * torch.rand(
            (n_walkers, 2, 3), generator=self.generator, dtype=torch.float64
        )
        self.call_count = 0
        self.acceptance_rates = acceptance_rates

    def collect_samples(
        self,
        model: torch.nn.Module,
        *,
        reset: bool = False,
        device: torch.device | str | None = None,
        emitter=None,
    ) -> tuple[Walkers, SamplerStats]:
        self.call_count += 1
        del emitter
        noise = 0.05 * torch.randn(
            self.positions.shape, generator=self.generator, dtype=torch.float64
        )
        # Mean-reverting around unit radius so the chain neither collapses onto
        # the nucleus nor wanders off.
        self.positions = 1.0 + self.rho * (self.positions - 1.0) + noise
        walkers = Walkers(
            positions=self.positions.clone(),
            atomic_configuration=self.atomic_configuration,
        )
        acceptance_rate = (
            0.6
            if self.acceptance_rates is None
            else self.acceptance_rates[self.call_count - 1]
        )
        stats = SamplerStats(
            acceptance_rate,
            self.n_walkers,
            0,
            self.n_steps,
            0.1,
            seed=self.call_count,
        )
        return walkers, stats


def _terms() -> tuple[object, ...]:
    return (KineticEnergy(), ElectronElectronInteraction(), ElectronNucleusInteraction())


def _context(tmp_path: Path) -> EvaluationContext:
    return EvaluationContext(
        namespace="eval/mcmc_energy",
        artifact_level="summaries",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=3,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )


def _checkpoint(tmp_path: Path, *, payload: bytes = b"he-v1-checkpoint-bytes") -> Path:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(payload)
    return path


def _summary(tmp_path: Path, checkpoint: Path, **overrides) -> TrajectoryStatisticsSummary:
    kwargs = {
        "stage": STAGE,
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "checkpoint_path": checkpoint,
        "evaluator_id": EVALUATOR_ID,
        "config": CONFIG,
    }
    kwargs.update(overrides)
    return TrajectoryStatisticsSummary(**kwargs)


def _generate(tmp_path: Path, *, n_draws: int = 48, model: HeliumSlaterModel | None = None):
    generator = TrajectoryMCMCGenerator(
        sampler=CorrelatedHeliumSampler(),
        hamiltonian_terms=_terms(),
        n_draws=n_draws,
        discard_draws=2,
        chunk_size=2,
    )
    return generator.generate(model=model or HeliumSlaterModel(), context=_context(tmp_path))


def test_real_local_energy_trajectory_yields_available_mcse(tmp_path: Path) -> None:
    """Drive the whole path on a real autograd local energy, end to end."""

    checkpoint = _checkpoint(tmp_path)
    generated = _generate(tmp_path)
    trajectory = generated.metadata[TRAJECTORY_METADATA_KEY]

    # The trajectory is a real time series of real local energies, not a stub.
    assert trajectory.observable == "local_energy"
    assert trajectory.values.shape == (48, 4)
    assert trajectory.draw_stride == 2
    assert trajectory.burn_in_draws == 2
    assert torch.isfinite(trajectory.values).all()
    assert trajectory.values.std() > 0.0

    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )

    assert result.metrics["local_energy_trajectory_statistics_available"] is True
    assert result.metrics["local_energy_mcse"] > 0.0
    assert result.metrics["local_energy_ess"] > 0.0
    assert result.metrics["local_energy_tau_int"] > 0.0
    assert result.metrics["local_energy_trajectory_walkers"] == 4
    assert result.metrics["local_energy_trajectory_draws_per_walker"] == 48
    assert result.metrics["local_energy_trajectory_total_draws"] == 192

    (artifact,) = result.artifacts
    assert artifact.metadata["status"] == "available"
    assert artifact.metadata["reason"] is None
    assert artifact.metadata["estimator_id"] == "pooled_geyer_ips"


def test_records_stream_the_complete_grid_from_the_single_trajectory_evaluation(tmp_path: Path) -> None:
    """Records retain accepted draw states and need neither another draw nor forward pass."""

    n_draws = 3
    discard_draws = 2
    sampler = CorrelatedHeliumSampler(n_walkers=4)
    model = HeliumSlaterModel()
    context = EvaluationContext(
        namespace="eval/mcmc_energy",
        artifact_level="records",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=3,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )
    generator = TrajectoryMCMCGenerator(
        sampler=sampler,
        hamiltonian_terms=_terms(),
        n_draws=n_draws,
        discard_draws=discard_draws,
        chunk_size=2,
    )

    generated = generator.generate(model=model, context=context)
    records = generated.trajectory_records
    assert records is not None
    assert sampler.call_count == discard_draws + n_draws
    # Four walkers, chunked in pairs, one kinetic forward per chunk and draw.
    assert model.forward_count == (discard_draws + n_draws) * 2
    assert records.row_count == n_draws * sampler.n_walkers
    assert records.final_draw.retained_draw_index == n_draws - 1
    assert torch.equal(generated.batch.positions, records.final_draw.positions.to(generated.batch.dtype))
    reconciliation = generated.metadata[TRAJECTORY_METADATA_KEY].reconciliation()
    assert records.observable_values_content_id == reconciliation.values_content_id
    assert records.mean == reconciliation.mean
    assert records.variance == reconciliation.variance

    calculated = LocalEnergyCalculator(
        hamiltonian_terms=_terms(), return_terms=True, chunk_size=2
    ).calculate(model=model, bundle=EvaluationBundle(generated=generated), context=context)
    calculated = WavefunctionCalculator(chunk_size=2).calculate(
        model=model, bundle=calculated, context=context
    )
    # Calculators reuse the captured final retained draw; records never trigger
    # a second per-draw/snapshot model evaluation.
    assert model.forward_count == (discard_draws + n_draws) * 2
    result = SampledRecordWriter(
        max_samples=n_draws * sampler.n_walkers,
        include_term_energies=True,
    ).summarize(bundle=calculated, context=context, namespace="eval/mcmc_energy")
    (artifact,) = result.artifacts
    assert artifact.metadata["selection"] == "complete_draw_walker_grid"
    assert artifact.metadata["rows"] == n_draws * sampler.n_walkers

    with records.path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(row["draw_index"]), int(row["walker_index"])) for row in rows] == [
        (draw, walker)
        for draw in range(n_draws)
        for walker in range(sampler.n_walkers)
    ]
    assert all(
        float(row["local_energy"])
        == pytest.approx(sum(float(row[f"term/{name}"]) for name in records.term_names))
        for row in rows
    )

    # The CSV stores raw coordinates only. Existing typed geometry helpers can
    # reconstruct all radial and pairwise quantities without duplicating them.
    first = rows[0]
    positions = torch.tensor(
        [
            [
                float(first[f"position/electron_{electron}/axis_{axis}"])
                for axis in range(records.spatial_dim)
            ]
            for electron in range(records.n_electrons)
        ],
        dtype=torch.float64,
    ).unsqueeze(0)
    batch = ElectronBatch(
        positions=positions,
        atomic_configuration=records.atomic_configuration,
        nuclear_positions=records.atomic_configuration.positions,
        nuclear_charges=records.atomic_configuration.charges,
    )
    assert electron_nuclear_distances(batch).shape == (1, records.n_electrons, 1)
    assert pairwise_distances(positions, eps=0.0).shape == (1, records.n_electrons, records.n_electrons, 1)
    metadata = json.loads(records.metadata_path.read_text(encoding="utf-8"))
    assert metadata["row_semantics"] == "complete_draw_walker_grid"
    assert metadata["atomic_configuration_id"] == records.atomic_configuration.content_id()

    with pytest.raises(ValueError, match="would truncate a complete trajectory grid"):
        SampledRecordWriter(max_samples=1, include_term_energies=True).summarize(
            bundle=calculated,
            context=context,
            namespace="eval/mcmc_energy",
        )

    with pytest.raises(ValueError, match="would truncate the final retained draw"):
        TrajectoryMCMCGenerator(
            sampler=CorrelatedHeliumSampler(n_walkers=4),
            hamiltonian_terms=_terms(),
            n_draws=1,
            max_samples=1,
        ).generate(model=HeliumSlaterModel(), context=context)


def test_chunked_records_and_sampler_diagnostics_reconcile_retained_geometry(
    tmp_path: Path,
) -> None:
    """The retained minimum spans every chunk and draw, not the final batch."""

    context = EvaluationContext(
        namespace="eval/mcmc_energy",
        artifact_level="records",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=3,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )
    generated = TrajectoryMCMCGenerator(
        sampler=CorrelatedHeliumSampler(
            n_walkers=4,
            acceptance_rates=(0.1, 0.2, 0.4, 0.6, 0.8),
        ),
        hamiltonian_terms=_terms(),
        n_draws=3,
        discard_draws=2,
        chunk_size=2,
    ).generate(model=HeliumSlaterModel(), context=context)
    records = generated.trajectory_records
    assert records is not None
    diagnostics = generated.metadata[SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY]

    with records.path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    retained_radii = []
    for row in rows:
        for electron in range(records.n_electrons):
            coordinates = torch.tensor(
                [
                    float(row[f"position/electron_{electron}/axis_{axis}"])
                    for axis in range(records.spatial_dim)
                ],
                dtype=torch.float64,
            )
            retained_radii.append(float(coordinates.norm().item()))
    expected_minimum = min(retained_radii)
    assert diagnostics.as_metrics()[
        "trajectory_retained_draw_minimum_electron_nucleus_radius"
    ] == pytest.approx(expected_minimum)
    assert len(diagnostics.retained_draws) == 3
    assert len(diagnostics.discarded_draws) == 2

    summary = SamplerStatsSummary().summarize(
        bundle=EvaluationBundle(generated=generated),
        context=context,
        namespace=context.namespace,
    )
    # The established scalar remains the final sampler call (0.8), while the
    # retained draw aggregate and series have distinct names and shapes.
    assert summary.metrics["sampler_acceptance_rate"] == pytest.approx(0.8)
    assert summary.metrics[
        "sampler_trajectory_retained_draw_acceptance_rate_mean"
    ] == pytest.approx(0.6)
    assert summary.metrics[
        "sampler_trajectory_retained_draw_minimum_electron_nucleus_radius"
    ] == pytest.approx(expected_minimum)
    (artifact,) = summary.artifacts
    assert artifact.path.name == SAMPLER_TRAJECTORY_DIAGNOSTICS_FILENAME
    payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    assert payload["retained_draw_acceptance_rate_series"] == [0.4, 0.6, 0.8]
    assert payload["discarded_draw_acceptance_rate_series"] == [0.1, 0.2]
    assert payload["intermediate_sampler_steps_observed"] is False


def test_kinetic_term_genuinely_requires_grad(tmp_path: Path) -> None:
    """Prove the observable needs autograd rather than merely tolerating it.

    Without this, a future `torch.no_grad()` wrapper around collection could be
    added and every other test here would still pass -- the collected values
    would just be wrong or the failure would surface somewhere unrelated. Under
    `no_grad` the second derivative cannot be taken at all, so the path raises.
    """

    with pytest.raises(RuntimeError):
        with torch.no_grad():
            _generate(tmp_path, n_draws=8)


def test_mcse_and_iid_stderr_are_both_published_and_distinct(tmp_path: Path) -> None:
    """Both estimators survive, labelled, and the IID bar is never relabelled."""

    checkpoint = _checkpoint(tmp_path)
    generated = _generate(tmp_path)
    context = _context(tmp_path)

    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=context,
        namespace="eval/mcmc_energy",
    )

    mcse = result.metrics["local_energy_mcse"]
    stderr_iid = result.metrics["local_energy_stderr_iid"]
    assert stderr_iid > 0.0
    # Both are published under distinct names; neither is the other.
    assert "local_energy_mcse" in result.metrics
    assert "local_energy_stderr_iid" in result.metrics
    assert mcse != stderr_iid
    # A serially correlated chain must inflate the honest bar, never shrink it
    # below the naive one -- that direction would flatter the result.
    assert result.metrics["local_energy_mcse_inflation"] == pytest.approx(mcse / stderr_iid)
    assert result.metrics["local_energy_mcse_inflation"] > 1.0

    # The final retained draw remains available to snapshot summaries without
    # being an extra sampler draw; its IID stderr keeps its own name.
    calculated = LocalEnergyCalculator(hamiltonian_terms=_terms(), chunk_size=2).calculate(
        model=HeliumSlaterModel(),
        bundle=EvaluationBundle(generated=generated),
        context=context,
    )
    snapshot = LocalEnergySummary().summarize(
        bundle=calculated,
        context=context,
        namespace="eval/mcmc_energy",
    )
    assert "local_energy_stderr" in snapshot.metrics
    assert "local_energy_stderr" not in result.metrics
    assert "local_energy_mcse" not in snapshot.metrics


def test_exactly_one_receipt_is_written_and_matches_the_identity(tmp_path: Path) -> None:
    """One sidecar line, keyed by the seven-part content-addressed identity."""

    checkpoint = _checkpoint(tmp_path)
    generated = _generate(tmp_path)

    _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )

    sidecar_path = tmp_path / DEFAULT_SIDECAR_NAME
    lines = [line for line in sidecar_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1

    # The sidecar record is flat: identity fields sit at the top level.
    record = json.loads(lines[0])
    assert record["checkpoint_sha256"] == file_sha256(checkpoint)
    assert record["stage"] == STAGE
    assert record["run_id"] == RUN_ID
    assert record["attempt_id"] == ATTEMPT_ID
    assert record["observable"] == "local_energy"
    assert record["evaluator_id"] == EVALUATOR_ID
    assert record["status"] == "available"
    assert record["statistics"]["mcse"] > 0.0
    # Content-addressed, never path-derived: no part of the checkpoint location
    # may appear in the key.
    identity_values = {field: record[field] for field in ("stage", "run_id", "attempt_id", "checkpoint_sha256", "config_sha256", "observable", "evaluator_id")}
    assert str(checkpoint) not in json.dumps(identity_values)


def test_moved_run_tree_still_joins(tmp_path: Path) -> None:
    """A relocated run tree joins on the same key; the hash is over contents."""

    original = tmp_path / "original"
    original.mkdir()
    checkpoint = _checkpoint(original)
    generated = _generate(tmp_path)
    context = EvaluationContext(
        namespace="eval/mcmc_energy",
        artifact_level="summaries",
        task_failure_policy="continue",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=3,
        run_dir=original,
        task_output_dir=original,
        metadata={},
    )
    _summary(original, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=context,
        namespace="eval/mcmc_energy",
    )

    moved = tmp_path / "moved"
    shutil.move(str(original), str(moved))

    # Rebuild the join key from the checkpoint at its NEW location.
    identity = TrajectoryStatisticsIdentity(
        stage=STAGE,
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        checkpoint_sha256=file_sha256(moved / "checkpoint.pt"),
        config_sha256=json.loads((moved / DEFAULT_SIDECAR_NAME).read_text().splitlines()[0])[
            "config_sha256"
        ],
        observable="local_energy",
        evaluator_id=EVALUATOR_ID,
    )
    receipt = TrajectoryStatisticsSidecar(moved / DEFAULT_SIDECAR_NAME).get(identity)
    assert receipt is not None
    assert receipt.status == "available"


def test_absent_trajectory_reports_reason_and_omits_statistics(tmp_path: Path) -> None:
    """A snapshot generator yields `absent` with a reason, never a zero."""

    checkpoint = _checkpoint(tmp_path)
    generated = GeneratedConfigurations(
        batch=_generate(tmp_path, n_draws=8).batch,
        metadata={},
    )

    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )

    assert result.metrics["local_energy_trajectory_statistics_available"] is False
    # Omitted, not zero-filled: a zero MCSE would read as "no correlation".
    for key in ("local_energy_mcse", "local_energy_ess", "local_energy_tau_int", "local_energy_stderr_iid"):
        assert key not in result.metrics

    (artifact,) = result.artifacts
    assert artifact.metadata["status"] == "absent"
    assert artifact.metadata["reason"]
    assert TRAJECTORY_METADATA_KEY in str(artifact.metadata["reason"])

    (receipt,) = TrajectoryStatisticsSidecar(tmp_path / DEFAULT_SIDECAR_NAME).read()
    assert receipt.status == "absent"
    assert receipt.payload is None


def test_too_few_draws_reports_unresolved_with_reason(tmp_path: Path) -> None:
    """An unresolvable real trajectory stays `unresolved`, never imputed."""

    checkpoint = _checkpoint(tmp_path)
    generated = _generate(tmp_path, n_draws=3)

    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )

    assert result.metrics["local_energy_trajectory_statistics_available"] is False
    assert "local_energy_mcse" not in result.metrics
    # Shape is still reported: three draws is exactly what explains the outcome.
    assert result.metrics["local_energy_trajectory_draws_per_walker"] == 3

    (artifact,) = result.artifacts
    assert artifact.metadata["status"] == "unresolved"
    assert artifact.metadata["reason"]

    (receipt,) = TrajectoryStatisticsSidecar(tmp_path / DEFAULT_SIDECAR_NAME).read()
    assert receipt.status == "unresolved"
    assert receipt.payload is None
    assert receipt.reason


def test_resolved_receipt_projects_all_four_geyer_diagnostics(tmp_path: Path) -> None:
    """The four truncation fields reach the row, carrying the receipt's values.

    Asserted against the receipt rather than against literals, so this checks
    PROJECTION FIDELITY and not that someone transcribed four numbers correctly
    on one fixture.
    """

    checkpoint = _checkpoint(tmp_path)
    generated = _generate(tmp_path)

    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )

    (receipt,) = TrajectoryStatisticsSidecar(tmp_path / DEFAULT_SIDECAR_NAME).read()
    assert receipt.status == "available"
    assert receipt.plateau is not None

    assert result.metrics["local_energy_plateau_reached"] == receipt.plateau.plateau_reached
    assert result.metrics["local_energy_truncation_lag"] == receipt.plateau.truncation_lag
    assert result.metrics["local_energy_geyer_pair_count"] == receipt.plateau.pair_count
    assert result.metrics["local_energy_max_lag"] == receipt.plateau.max_lag

    # `truncation_lag` is only interpretable against the window that was
    # available, which is why `max_lag` travels with it: 2*pairs - 1 lags were
    # summed out of max_lag possible.
    assert receipt.plateau.truncation_lag == 2 * receipt.plateau.pair_count - 1
    assert result.metrics["local_energy_max_lag"] == 47


def test_unresolved_receipt_still_projects_its_truncation_diagnostics(tmp_path: Path) -> None:
    """The diagnostics survive the case they exist to explain.

    THIS IS THE TEST THAT PROVES THE HOIST. `_receipt_metrics` returns early on
    ``payload is None``, but `producer.py` builds `PlateauDiagnostics`
    unconditionally at line 223 and passes it into every ``unresolved(...)``
    return, so an unresolved receipt genuinely carries these fields. A
    projection placed below that early return drops them precisely when the
    producer withheld tau and ESS -- which is exactly when a reader needs to
    know whether an unterminated Geyer sequence is the reason.

    Three draws against a minimum of eight is the deterministic way to reach an
    unresolved receipt; the ``no plateau within N lags`` outcome is the same
    code path and the same early return.
    """

    checkpoint = _checkpoint(tmp_path)
    generated = _generate(tmp_path, n_draws=3)

    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )

    (receipt,) = TrajectoryStatisticsSidecar(tmp_path / DEFAULT_SIDECAR_NAME).read()
    assert receipt.status == "unresolved"
    assert receipt.payload is None
    assert receipt.plateau is not None

    # No payload, so no tau and no MCSE -- and yet the reason the estimator
    # refused is still on the row.
    assert "local_energy_mcse" not in result.metrics
    assert result.metrics["local_energy_plateau_reached"] is False
    assert result.metrics["local_energy_max_lag"] == 2

    # Absent rather than zero-filled: no pair was ever summed, and a 0 here
    # would read as "truncated at lag zero" instead of "never got that far".
    assert "local_energy_truncation_lag" not in result.metrics
    assert "local_energy_geyer_pair_count" not in result.metrics


#: Relative tolerance for the homogeneous MCSE/tau identity.
#:
#: STATED IN THE UNIT THAT SETS THE ROUNDING ERROR, which is not the order-one
#: `inflation` in view. On four BITWISE-IDENTICAL columns the four per-chain
#: variances are bitwise equal to each other, yet `Var_pooled` -- their mean --
#: differs from them in the last place, because mean-of-C-identical-floats is
#: not the identity in IEEE. That ~1 ulp perturbation of a VARIANCE is halved by
#: the square root and lands in `inflation`. A bound expressed in ulps of
#: `inflation` would therefore have been the wrong unit by construction.
#:
#: MEASURED, not guessed (Cannon jobs 39555148 and 39555915, partition `test`):
#: 1.245 eps at n=64 and 0.520 eps at n=1024. Pinned at 16 eps, roughly 13x the
#: worst observed, leaving room for a different summation order on another
#: platform without admitting anything structural. A systematic offset would
#: show up near C(n-1)/(Cn-1) = 0.98824 at C=4, n=64 -- a 1.18% effect that this
#: tolerance is deliberately far too tight to swallow.
_HOMOGENEOUS_REL_TOLERANCE = 16 * 2.220446049250313e-16


def _ar1(n_draws: int, *, rho: float, scale: float, seed: int) -> torch.Tensor:
    """One deterministic zero-mean AR(1) column.

    Synthetic on purpose. The homogeneity and heterogeneity of the chains is the
    property under test, and a real sampler cannot be asked for chains with
    prescribed per-chain variance and tau. Note this builds a synthetic
    TRAJECTORY, never a synthetic RECEIPT: the receipt is produced by the real
    estimator from these values, so the relations asserted below are the
    implementation's, not the fixture's.
    """

    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(n_draws, generator=generator, dtype=torch.float64) * scale
    out = torch.empty(n_draws, dtype=torch.float64)
    value = 0.0
    for index in range(n_draws):
        value = rho * value + float(noise[index])
        out[index] = value
    return out - out.mean()


def _receipt_for(values: torch.Tensor, tmp_path: Path) -> tuple[dict, object]:
    """Run a trajectory through the real summary and return (metrics, receipt)."""

    checkpoint = _checkpoint(tmp_path)
    trajectory = ObservableTrajectory(
        observable="local_energy", values=values, draw_stride=1, burn_in_draws=0
    )
    generated = GeneratedConfigurations(
        batch=_generate(tmp_path, n_draws=8).batch,
        metadata={TRAJECTORY_METADATA_KEY: trajectory},
    )
    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )
    (receipt,) = TrajectoryStatisticsSidecar(tmp_path / DEFAULT_SIDECAR_NAME).read()
    return result.metrics, receipt


def test_inflation_equals_root_tau_when_the_chains_are_homogeneous(tmp_path: Path) -> None:
    """The shortcut is CORRECT exactly when its assumptions hold, and this pins that.

    ``mcse = stderr_iid * sqrt(tau_int)`` assumes every chain shares one variance
    and one tau. Four bitwise-identical columns satisfy that by shared code path
    rather than by numerical coincidence -- `per_chain_integrated_autocorrelation`
    hands each column over as its own ``[draw, 1]`` trajectory, so identical input
    gives identical output. C = 4 is a power of two, so the ``1/C`` weighting is
    exact.

    This is half a pair. Its partner asserts the two DIVERGE on heterogeneous
    chains. Asserting only the divergence would be false, and asserting only the
    agreement would license the very "fix" the producer rejects.
    """

    column = _ar1(64, rho=0.6, scale=1.0, seed=17)
    metrics, receipt = _receipt_for(column.unsqueeze(1).repeat(1, 4), tmp_path)
    assert receipt.status == "available"

    # The fixture IS homogeneous, asserted rather than assumed: if a future
    # change made the per-chain estimates differ, the identity below would fail
    # for a reason that has nothing to do with the estimator.
    variances = {chain.variance for chain in receipt.chains}
    taus = {chain.tau_int for chain in receipt.chains}
    assert len(variances) == 1
    assert len(taus) == 1

    inflation = metrics["local_energy_mcse_inflation"]
    root_tau = math.sqrt(metrics["local_energy_tau_int"])
    assert inflation == pytest.approx(root_tau, rel=_HOMOGENEOUS_REL_TOLERANCE)


def test_inflation_diverges_from_root_tau_when_the_chains_are_heterogeneous(
    tmp_path: Path,
) -> None:
    """Heterogeneous chains give ``tau_int < 1`` beside ``inflation > 1``.

    That pairing looks like a contradiction on an eval row and is not. `tau_int`
    is an N-weighted HARMONIC mean of the per-chain tau, so it is dominated by
    the best-mixed chains and can fall below one; `mcse` sums per-chain
    ``s_i^2 * tau_i`` terms and is dominated by the worst ones; and `stderr_iid`
    uses the POOLED variance, which includes between-chain spread that `mcse`
    excludes. Two slow high-amplitude chains against two fast low-amplitude ones
    separate all three.

    THIS IS THE MUTATION-SENSITIVE HALF. The dangerous future edit is someone
    "fixing" the estimator so the two numbers agree -- which would silently
    substitute the shortcut `producer.py` rejects by name. That edit passes the
    homogeneous test above and fails here.

    1024 draws, not 64. At 64 the rho=0.90 chains span only ~3.4 of their own
    tau, split-Rhat reached 1.16, and the producer correctly returned
    ``unresolved`` rather than publish a bar around a disputed mean. That was a
    degenerate fixture, not a finding, and the fix is a fixture long enough to
    mix -- never a widened ``r_hat_threshold``.
    """

    stacked = torch.stack(
        [
            _ar1(1024, rho=0.90, scale=3.0, seed=101),
            _ar1(1024, rho=0.88, scale=3.0, seed=102),
            _ar1(1024, rho=-0.5, scale=0.2, seed=103),
            _ar1(1024, rho=-0.5, scale=0.2, seed=104),
        ],
        dim=1,
    )
    metrics, receipt = _receipt_for(stacked, tmp_path)
    assert receipt.status == "available"

    tau_int = metrics["local_energy_tau_int"]
    inflation = metrics["local_energy_mcse_inflation"]
    root_tau = math.sqrt(tau_int)

    # The signature that prompted this test: both true at once.
    assert tau_int < 1.0
    assert inflation > 1.0

    # Divergence far above the ~1 eps floor the homogeneous case sits at.
    # Measured on Cannon job 39555915: tau_int 0.6633, inflation 4.5172,
    # sqrt(tau_int) 0.8144, a ratio of 5.55.
    assert inflation / root_tau > 3.0

    # ...and the per-chain form is the one actually emitted, not the shortcut.
    total = receipt.shape.total_draws
    per_chain_mcse = (
        sum(
            (chain.n_draws / total) ** 2 * chain.variance * chain.tau_int / chain.n_draws
            for chain in receipt.chains
        )
        ** 0.5
    )
    shortcut_mcse = metrics["local_energy_stderr_iid"] * root_tau
    assert metrics["local_energy_mcse"] == per_chain_mcse
    assert abs(metrics["local_energy_mcse"] - shortcut_mcse) / metrics["local_energy_mcse"] > 0.5


def test_every_published_relation_is_recomputable_from_the_receipt(tmp_path: Path) -> None:
    """Assert each implemented relation positively, on a real emitted row.

    Exact ``==`` is legitimate here and is not a floating-point hope: each right
    side executes the IDENTICAL expression on the IDENTICAL inputs the producer
    used, so this is bitwise equality by shared code path. Measured residual on
    Cannon jobs 39555148 and 39555915 was 0.000 ulp for all three on every
    fixture, homogeneous and heterogeneous.
    """

    checkpoint = _checkpoint(tmp_path)
    generated = _generate(tmp_path)
    result = _summary(tmp_path, checkpoint).summarize(
        bundle=EvaluationBundle(generated=generated),
        context=_context(tmp_path),
        namespace="eval/mcmc_energy",
    )
    (receipt,) = TrajectoryStatisticsSidecar(tmp_path / DEFAULT_SIDECAR_NAME).read()
    metrics = result.metrics
    total = receipt.shape.total_draws

    # ess = sum_i (N_i / tau_i)
    assert metrics["local_energy_ess"] == sum(
        chain.n_draws / chain.tau_int for chain in receipt.chains
    )
    # tau_int = N / ess -- reported as the value consistent with the pooled ESS,
    # never independently estimated.
    assert metrics["local_energy_tau_int"] == total / metrics["local_energy_ess"]
    # mcse^2 = sum_i (N_i/N)^2 s_i^2 tau_i / N_i
    assert metrics["local_energy_mcse"] == (
        sum(
            (chain.n_draws / total) ** 2 * chain.variance * chain.tau_int / chain.n_draws
            for chain in receipt.chains
        )
        ** 0.5
    )
    # stderr_iid = sqrt(Var_pooled / N), from the POOLED variance
    assert metrics["local_energy_stderr_iid"] == (receipt.payload.variance / total) ** 0.5
    # inflation = mcse / stderr_iid, and nothing else
    assert (
        metrics["local_energy_mcse_inflation"]
        == metrics["local_energy_mcse"] / metrics["local_energy_stderr_iid"]
    )


def test_summary_requires_a_config_identity(tmp_path: Path) -> None:
    """The join key admits no blanks, so a missing config hash fails loudly."""

    with pytest.raises(ValueError, match="config_sha256 or config"):
        TrajectoryStatisticsSummary(
            stage=STAGE,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
            checkpoint_path=_checkpoint(tmp_path),
            evaluator_id=EVALUATOR_ID,
        )


def test_trajectory_metadata_must_be_typed(tmp_path: Path) -> None:
    """A wrong-typed trajectory is a defect, not an `absent` receipt."""

    generated = GeneratedConfigurations(
        batch=_generate(tmp_path, n_draws=8).batch,
        metadata={TRAJECTORY_METADATA_KEY: torch.zeros(4, 2)},
    )
    with pytest.raises(TypeError, match="ObservableTrajectory"):
        _summary(tmp_path, _checkpoint(tmp_path)).summarize(
            bundle=EvaluationBundle(generated=generated),
            context=_context(tmp_path),
            namespace="eval/mcmc_energy",
        )
