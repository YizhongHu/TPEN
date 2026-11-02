"""Local-energy evaluation calculator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch

# Re-exported for existing evaluation callers. The bounded executor is
# physics-owned and the batch slice is data-owned, so training can use both
# without depending on the evaluation package.
from tpen.data.batch import slice_flat_batch
from tpen.evaluation.bundle import EvaluationBundle, LocalEnergyValues
from tpen.evaluation.protocols import EvaluationContext
from tpen.physics.hamiltonian import (
    HamiltonianTerm,
    LocalEnergyEvaluator,
    LocalEnergyResult,
    NaiveLocalEnergyEvaluator,
    evaluate_local_energy_in_chunks,
    normalize_hamiltonian_terms,
)

_DEFAULT_EVALUATOR = NaiveLocalEnergyEvaluator()


class LocalEnergyCalculator:
    """Compute raw local-energy values for generated configurations."""

    name = "local_energy"

    def __init__(
        self,
        *,
        hamiltonian_terms: Sequence[HamiltonianTerm] | Mapping[str, HamiltonianTerm],
        return_terms: bool = False,
        chunk_size: int | None = None,
        evaluator: LocalEnergyEvaluator | None = None,
    ) -> None:
        self.hamiltonian_terms = normalize_hamiltonian_terms(hamiltonian_terms)
        self.return_terms = bool(return_terms)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.evaluator = _DEFAULT_EVALUATOR if evaluator is None else evaluator

    def validate(self, *, model: object, generator: object) -> None:
        """Validate the explicitly selected evaluator before sampling."""

        self.evaluator.validate_for_generator(self.hamiltonian_terms, model, generator)

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Evaluate local energy and return a bundle with raw values."""

        records = bundle.generated.trajectory_records
        if records is not None:
            records.validate(check_files=False)
            if tuple(self.hamiltonian_terms) != records.term_names:
                raise ValueError(
                    "LocalEnergyCalculator terms disagree with streamed trajectory records"
                )
            flat = bundle.generated.batch.flatten_samples()
            n_rows = records.validate_snapshot_batch(flat)
            total = records.final_draw.local_energy[:n_rows].to(
                device=flat.device,
                dtype=flat.dtype,
            )
            terms = None
            if self.return_terms:
                terms = {
                    name: records.final_draw.term_energies[name][:n_rows].to(
                        device=flat.device,
                        dtype=flat.dtype,
                    )
                    for name in records.term_names
                }
            return replace(
                bundle,
                local_energy=LocalEnergyValues(
                    local_energy=total,
                    finite_mask=torch.isfinite(total),
                    term_energies=terms,
                ),
            )

        result = evaluate_local_energy_in_chunks(
            self.hamiltonian_terms,
            model,
            bundle.generated.batch,
            return_terms=self.return_terms,
            chunk_size=self.chunk_size,
            evaluator=self.evaluator,
        )
        total, term_energies = split_local_energy_result(result)
        total = total.detach()
        terms = None if term_energies is None else {name: value.detach() for name, value in term_energies.items()}
        local = LocalEnergyValues(
            local_energy=total,
            finite_mask=torch.isfinite(total),
            term_energies=terms,
        )
        return replace(bundle, local_energy=local)


def split_local_energy_result(
    result: LocalEnergyResult | torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor] | None]:
    """Return ``(total, terms_or_none)`` from a local-energy result."""

    if isinstance(result, LocalEnergyResult):
        return result.total, result.terms
    return result, None


__all__ = [
    "LocalEnergyCalculator",
    "evaluate_local_energy_in_chunks",
    "slice_flat_batch",
    "split_local_energy_result",
]
