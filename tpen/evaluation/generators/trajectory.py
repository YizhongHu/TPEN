"""MCMC generator that retains a ``[draw, walker]`` observable trajectory.

:class:`tpen.evaluation.generators.MCMCGenerator` takes a single snapshot of the
sampler's walkers. A snapshot has no time axis, so nothing downstream can
estimate an autocorrelation time from it, and the only error bar available is
the IID ``sigma / sqrt(N)`` that
:func:`tpen.evaluation.summaries.local_energy.summarize_values` publishes.

This generator adds the missing draw axis by delegating to
:func:`tpen.sampling.trajectory.collect_observable_trajectory`, which advances
the sampler's persistent chains and evaluates one observable per draw. The
resulting :class:`~tpen.statistics.ObservableTrajectory` is carried in the
generated metadata for
:class:`~tpen.evaluation.summaries.trajectory_statistics.TrajectoryStatisticsSummary`
to hand to the statistics producer.

Autograd stays live throughout. The local energy differentiates ``logabs``
twice with respect to positions, so a ``torch.no_grad()`` wrapper would make the
contract's first observable raise rather than merely run slower. "Fixed model"
is a claim about *parameters*, and the collector enforces it by freezing them
and fingerprinting them before and after -- not by disabling autograd.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from tpen.data.batch.walkers import Walkers
from tpen.evaluation.bundle import GeneratedConfigurations
from tpen.evaluation.calculators.local_energy import (
    evaluate_local_energy_in_chunks,
    slice_flat_batch,
)
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.trajectory_records import (
    TRAJECTORY_RECORD_FILENAME,
    TrajectoryRecordBatch,
    TrajectoryRecordStreamWriter,
)
from tpen.physics.hamiltonian import (
    HamiltonianTerm,
    LocalEnergyEvaluator,
    LocalEnergyResult,
    NaiveLocalEnergyEvaluator,
    RecordingLocalEnergyEvaluator,
    normalize_hamiltonian_terms,
)
from tpen.physics.system import SystemDescriptor
from tpen.physics.receipt import ExecutionReceipt, decode_ordered_contribution_sequence
from tpen.sampling.trajectory import collect_observable_trajectory_with_diagnostics

TRAJECTORY_METADATA_KEY = "observable_trajectory"
"""Metadata key under which the collected trajectory is published."""

SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY = "sampler_trajectory_diagnostics"
"""Metadata key for typed draw-resolved sampler health."""


class TrajectoryMCMCGenerator:
    """Collect a fixed-model ``[draw, walker]`` local-energy trajectory.

    The evaluation batch returned alongside the trajectory is its final
    retained draw, so snapshot summaries remain available without advancing the
    sampler one more time.

    Parameters
    ----------
    sampler : object
        Sampler exposing ``collect_samples(model, *, reset=..., device=...)``.
    hamiltonian_terms : sequence or mapping of HamiltonianTerm
        Terms defining the local energy evaluated once per draw.
    n_draws : int
        Draws retained per walker. Each draw advances the sampler by its own
        configured ``n_steps``, so this multiplies sampling cost.
    discard_draws : int, optional
        Draws collected and discarded before retention begins, on top of the
        sampler's own burn-in.
    observable_name : str, optional
        Name recorded on the trajectory and used as the receipt's ``observable``
        identity field.
    chunk_size : int, optional
        Local-energy evaluation chunk size, forwarded per draw.
    seed : int, optional
        Recorded in metadata for bookkeeping parity with `MCMCGenerator`.
    max_samples : int, optional
        Cap on the final-draw evaluation batch only. It does **not** truncate
        the trajectory records: dropping those rows would redefine the
        draw-by-walker artifact.
    record_filename : str, optional
        Streamed CSV filename used when ``artifact_level == "records"``.
    reset : bool, optional
        Force a fresh, re-burned-in chain before the first draw.
    verify_model_unchanged : bool, optional
        Fingerprint parameters before and after collection. Defaults to ``True``.
    system : SystemDescriptor or None, optional
        Explicit opt-in to ``trajectory_records/v2``. When ``None`` (the
        default), the legacy ``trajectory_records/v1`` path is used unchanged
        and `Walkers.atomic_configuration` is required, exactly as before this
        parameter existed. When supplied, `evaluator` is wrapped in
        `~tpen.physics.hamiltonian.RecordingLocalEnergyEvaluator` bound to
        `system`, and a complete-grid record requires no nuclear resource at
        all unless a term actually declares one (see
        ``system-context-records-design-r3.md``, "Explicit selection reaches
        the producer").
    evaluator : LocalEnergyEvaluator or None, optional
        Evaluator applied to every per-draw local-energy evaluation; defaults
        to the naive reference. Ignored unless it is the delegate wrapped when
        `system` is supplied -- selection never silently falls back to naive
        for an explicitly requested evaluator.

    Notes
    -----
    The v2 writer receives the run-derived identity from ``EvaluationContext``.
    A checkpoint-backed run must provide the model-file digest; constructed
    controls carry ``model_provenance="constructed"`` and no checkpoint hash.

    Raises
    ------
    ValueError
        If `n_draws` is below one or `discard_draws` is negative.
    """

    name = "trajectory_mcmc"

    def __init__(
        self,
        *,
        sampler: object,
        hamiltonian_terms: Sequence[HamiltonianTerm] | Mapping[str, HamiltonianTerm],
        n_draws: int,
        discard_draws: int = 0,
        observable_name: str = "local_energy",
        chunk_size: int | None = None,
        seed: int | None = None,
        max_samples: int | None = None,
        record_filename: str = TRAJECTORY_RECORD_FILENAME,
        reset: bool = False,
        verify_model_unchanged: bool = True,
        system: SystemDescriptor | None = None,
        evaluator: LocalEnergyEvaluator | None = None,
    ) -> None:
        self.sampler = sampler
        self.hamiltonian_terms = normalize_hamiltonian_terms(hamiltonian_terms)
        self.n_draws = int(n_draws)
        if self.n_draws < 1:
            raise ValueError(f"n_draws must be at least 1, got {self.n_draws}")
        self.discard_draws = int(discard_draws)
        if self.discard_draws < 0:
            raise ValueError(f"discard_draws must be non-negative, got {self.discard_draws}")
        self.observable_name = str(observable_name).strip()
        if not self.observable_name:
            raise ValueError("observable_name must be non-empty")
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.seed = None if seed is None else int(seed)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.record_filename = str(record_filename).strip()
        if not self.record_filename:
            raise ValueError("record_filename must be non-empty")
        self.reset = bool(reset)
        self.verify_model_unchanged = bool(verify_model_unchanged)
        self.system = system
        raw_evaluator: LocalEnergyEvaluator = (
            NaiveLocalEnergyEvaluator() if evaluator is None else evaluator
        )
        self._evaluator: LocalEnergyEvaluator = (
            raw_evaluator
            if self.system is None
            else RecordingLocalEnergyEvaluator(raw_evaluator, system=self.system)
        )

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Collect the trajectory and expose its final retained draw as the batch."""

        if model is None:
            raise TypeError("TrajectoryMCMCGenerator requires a model to evaluate the observable")
        collect = getattr(self.sampler, "collect_samples", None)
        if not callable(collect):
            raise TypeError(
                "TrajectoryMCMCGenerator sampler must expose "
                "collect_samples(model, *, reset=..., device=...)"
            )

        evaluation_index = 0
        final_walkers: Walkers | None = None
        record_stream: TrajectoryRecordStreamWriter | None = None
        first_receipt: ExecutionReceipt | None = None

        def observable(walkers: Walkers) -> torch.Tensor:
            nonlocal evaluation_index, final_walkers, record_stream, first_receipt
            retained_draw = evaluation_index - self.discard_draws
            evaluation_index += 1
            result = self._local_energy_per_walker(model=model, walkers=walkers)
            if self.system is not None:
                if result.receipt is None:
                    raise ValueError("trajectory_records/v2 requires an execution receipt")
                if first_receipt is None:
                    first_receipt = result.receipt
                elif (
                    result.receipt.system_id != first_receipt.system_id
                    or result.receipt.method_id() != first_receipt.method_id()
                ):
                    raise ValueError(
                        "trajectory record execution receipt changed between draws: "
                        "system_id and method_id must agree with the first draw"
                    )
            if retained_draw >= 0:
                final_walkers = walkers.clone().detach()
                if context.artifact_level == "records":
                    record = self._record_from_result(
                        walkers=walkers,
                        result=result,
                        draw_index=retained_draw,
                    )
                    if record_stream is None:
                        if self.max_samples is not None and self.max_samples < walkers.batch_size:
                            raise ValueError(
                                "TrajectoryMCMCGenerator max_samples would truncate the final "
                                "retained draw required by trajectory records: "
                                f"capacity={self.max_samples}, required={walkers.batch_size}"
                            )
                        if self.system is None:
                            atoms = walkers.atomic_configuration
                            if atoms is None:
                                raise ValueError(
                                    "trajectory records require Walkers.atomic_configuration "
                                    "so geometry is retained once as a typed reference"
                                )
                            record_stream = TrajectoryRecordStreamWriter(
                                context.task_output_dir / self.record_filename,
                                observable=self.observable_name,
                                n_draws=self.n_draws,
                                n_walkers=walkers.batch_size,
                                term_names=tuple(record.term_energies),
                                atomic_configuration=atoms,
                                first_draw=record,
                            )
                        else:
                            if result.receipt is None:
                                raise ValueError(
                                    "trajectory_records/v2 requires an execution receipt; "
                                    "the bound evaluator did not declare one"
                                )
                            record_stream = TrajectoryRecordStreamWriter(
                                context.task_output_dir / self.record_filename,
                                observable=self.observable_name,
                                n_draws=self.n_draws,
                                n_walkers=walkers.batch_size,
                                term_names=tuple(record.term_energies),
                                system_descriptor=self.system,
                                execution_receipt=result.receipt,
                                resolved_config_identity=context.resolved_config_identity,
                                model_provenance=context.model_provenance,
                                checkpoint_sha256=context.checkpoint_sha256,
                                first_draw=record,
                            )
                    else:
                        record_stream.append(record)
            return result.total

        try:
            trajectory, sampler_stats, sampler_diagnostics = (
                collect_observable_trajectory_with_diagnostics(
                    self.sampler,
                    model,
                    observable,
                    observable_name=self.observable_name,
                    n_draws=self.n_draws,
                    discard_draws=self.discard_draws,
                    device=context.device,
                    reset=self.reset,
                    verify_model_unchanged=self.verify_model_unchanged,
                )
            )
            trajectory_records = (
                None if record_stream is None else record_stream.finalize(trajectory)
            )
        except Exception:
            if record_stream is not None:
                record_stream.close()
            raise

        expected_evaluations = self.discard_draws + self.n_draws
        if evaluation_index != expected_evaluations or final_walkers is None:
            raise RuntimeError(
                "trajectory collection did not expose the expected final retained draw"
            )
        batch = final_walkers.make_batch().flatten_samples()
        if self.max_samples is not None and self.max_samples >= 0:
            batch = slice_flat_batch(batch, 0, min(self.max_samples, batch.batch_size))
        sample_index = torch.arange(batch.batch_size, device=batch.device)
        metadata: dict[str, Any] = {
            "sample_index": sample_index,
            "walker_index": sample_index,
            "draw_index": self.n_draws - 1,
            "snapshot_artifact": "final_retained_trajectory_draw",
            "sampler_stats": sampler_stats,
            SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY: sampler_diagnostics,
            TRAJECTORY_METADATA_KEY: trajectory,
        }
        if self.seed is not None:
            metadata["seed"] = self.seed
        return GeneratedConfigurations(
            batch=batch,
            metadata=metadata,
            trajectory_records=trajectory_records,
        )

    def _local_energy_per_walker(
        self,
        *,
        model: torch.nn.Module,
        walkers: Walkers,
    ) -> LocalEnergyResult:
        """Evaluate every row primitive in one chunked local-energy sweep."""

        batch = walkers.make_batch().flatten_samples()
        result = evaluate_local_energy_in_chunks(
            self.hamiltonian_terms,
            model,
            batch,
            return_terms=True,
            chunk_size=self.chunk_size,
            evaluator=self._evaluator,
        )
        if not isinstance(result, LocalEnergyResult):
            raise TypeError("trajectory local-energy evaluation must return LocalEnergyResult")
        if result.wavefunction_output is None:
            raise ValueError(
                "trajectory local-energy evaluation did not expose the wavefunction output "
                "used by its kinetic term; a fallback model pass is forbidden"
            )
        if result.receipt is not None:
            try:
                contribution_names = decode_ordered_contribution_sequence(
                    result.receipt.numerical_policy["contribution_order"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "trajectory local-energy receipt must declare its ordered contribution sequence"
                ) from exc
        else:
            contribution_names = tuple(self.hamiltonian_terms)
        if set(result.terms) != set(contribution_names) or len(result.terms) != len(contribution_names):
            raise ValueError(
                "trajectory local-energy result contributions disagree with its execution receipt"
            )
        reconstructed: torch.Tensor | None = None
        for name in contribution_names:
            value = result.terms[name]
            reconstructed = value if reconstructed is None else reconstructed + value
        if reconstructed is None or not torch.equal(reconstructed, result.total):
            raise ValueError("trajectory local-energy total does not equal its configured term sum")
        return result

    def _record_from_result(
        self,
        *,
        walkers: Walkers,
        result: LocalEnergyResult,
        draw_index: int,
    ) -> TrajectoryRecordBatch:
        """Build one owned CPU record draw from the exact accepted walker state."""

        output = result.wavefunction_output
        if output is None:
            raise ValueError("trajectory record requires the captured wavefunction output")
        n_walkers = walkers.batch_size
        return TrajectoryRecordBatch(
            draw_index=torch.full(
                (n_walkers,),
                int(draw_index),
                device=walkers.device,
                dtype=torch.int64,
            ),
            walker_index=torch.arange(n_walkers, device=walkers.device),
            positions=walkers.positions,
            local_energy=result.total,
            term_energies=result.terms,
            logabs=output.logabs,
            sign=output.sign,
            finite_mask=torch.isfinite(result.total),
        )


__all__ = [
    "SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY",
    "TRAJECTORY_METADATA_KEY",
    "TrajectoryMCMCGenerator",
]
