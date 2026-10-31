"""Hamiltonian terms, local-energy results, and typed evaluation.

A Hamiltonian is a collection of `HamiltonianTerm`s, given either as a sequence
or as a ``dict[str, HamiltonianTerm]`` that names each term explicitly. Dict
keys are the public, authoritative term names: they must be non-empty strings,
and the values must expose ``local_energy(wavefunction, batch)``.

Evaluation goes through a typed evaluator seam: `NaiveLocalEnergyEvaluator`
consumes a `NaiveLocalEnergyContext` and owns term normalization (see
`normalize_hamiltonian_terms`), ordered per-term evaluation, validation, and
summation, optionally returning the per-term decomposition keyed by the
resolved term names. The `local_energy` helper is an explicit delegate to the
naive evaluator and remains the single default path; the naive evaluator is
the permanent numerical reference for any future optimized evaluator behind
the `LocalEnergyEvaluator` protocol. Evaluation summaries use canonical flat
metric keys such as ``energy`` and ``energy_term_{name}``; hierarchy belongs
in the logging namespace.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import torch

from tpen.data.atomic_configuration import strict_equal_atomic_configurations
from tpen.data.batch import ElectronBatch, WavefunctionOutput, slice_flat_batch
from tpen.naming import camel_to_snake
from tpen.physics.operators import (
    ELECTRON_NUCLEUS_COULOMB,
    KINETIC_ENERGY,
    OperatorId,
    declared_operator_geometry,
    is_registered_operator,
)


@dataclass
class LocalEnergyResult:
    """Container for a decomposed local-energy evaluation.

    Parameters
    ----------
    total : torch.Tensor
        Summed local energy across all contributions, shape ``[batch]``.
    terms : dict[str, torch.Tensor]
        Per-term local energies keyed by the resolved term name. When produced
        by `local_energy`, names come from the ``dict`` key (named form) or the
        snake-case class name (sequence form), and are guaranteed unique.
    wavefunction_output : WavefunctionOutput or None, optional
        Exact wavefunction output used by a term while evaluating this local
        energy. The aggregate evaluator permits at most one producing term so
        consumers can retain signed-log values without a second model pass.
    per_electron_kinetic : torch.Tensor or None, optional
        Explicit per-electron kinetic attribution with shape
        [batch, n_electrons]. The aggregate evaluator permits at most one
        producing term and carries it alongside the wavefunction output so
        diagnostics reuse the same differentiated model evaluation.
    """

    total: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    wavefunction_output: WavefunctionOutput | None = None
    per_electron_kinetic: torch.Tensor | None = None


def normalize_hamiltonian_terms(
    terms: Mapping[Any, Any] | Sequence[Any],
) -> dict[str, Any]:
    """Return an ordered ``{name: term}`` mapping from a dict or sequence.

    A ``dict[str, HamiltonianTerm]`` is used directly: its keys are the
    explicit, authoritative term names used in `LocalEnergyResult.terms` and
    downstream metrics. A sequence falls back to the snake-case class name of
    each term, suffixed with the term index when a class name repeats, so the
    resulting names are always unique.

    Names are enforced as non-empty strings and each value must expose a
    callable ``local_energy``; invalid specifications fail loudly here rather
    than later during evaluation.

    Parameters
    ----------
    terms : Mapping or Sequence of HamiltonianTerm
        Configured Hamiltonian terms, named (dict) or unnamed (sequence).

    Returns
    -------
    dict[str, HamiltonianTerm]
        Ordered mapping from resolved name to term.
    """

    if isinstance(terms, Mapping):
        normalized: dict[str, HamiltonianTerm] = {}
        for key, term in terms.items():
            if not isinstance(key, str):
                raise TypeError(f"hamiltonian term names must be strings, got {type(key).__name__}")
            _validate_hamiltonian_term(key, term)
            normalized[key] = term
        return normalized

    sequence = list(terms)
    base_names = [camel_to_snake(type(term).__name__) for term in sequence]
    counts = Counter(base_names)
    normalized = {}
    for index, (term, base) in enumerate(zip(sequence, base_names)):
        name = base if counts[base] == 1 else f"{base}_{index}"
        _validate_hamiltonian_term(name, term)
        normalized[name] = term
    return normalized


def _validate_hamiltonian_term(name: str, term: object) -> None:
    """Fail loudly on an empty term name or an invalid term specification."""

    if not name or not name.strip():
        raise ValueError("hamiltonian term names must be non-empty strings")
    if not callable(getattr(term, "local_energy", None)):
        raise TypeError(
            f"hamiltonian term {name!r} ({type(term).__name__}) must expose a callable "
            "local_energy(wavefunction, batch)"
        )


def _declared_operator_id(term: object) -> OperatorId | None:
    """Return a registered term's declared operator identity, if present."""

    term_type = type(term)
    if not is_registered_operator(term_type):
        return None
    return term_type.operator_id


@runtime_checkable
class HamiltonianTerm(Protocol):
    """Protocol for a single Hamiltonian term.

    A term reports its contribution to the local energy as a
    `LocalEnergyResult` so that decompositions compose under summation.
    """

    name: str

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        """Return this term's local-energy contribution."""
        ...


ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class NaiveLocalEnergyContext:
    """Typed input context for the naive local-energy evaluator.

    Parameters
    ----------
    wavefunction : callable
        Wavefunction model or exact reference returning ``WavefunctionOutput``.
        The narrowest existing callable contract is reused deliberately; no
        ansatz-specific base class exists solely for this interface.
    batch : ElectronBatch
        Electron configuration batch.
    """

    wavefunction: object
    batch: ElectronBatch


@dataclass(frozen=True)
class AnalyticCuspContext:
    """Typed input context for the analytic electron-nucleus evaluator."""

    wavefunction: object
    batch: ElectronBatch


class LocalEnergyEvaluator(Protocol, Generic[ContextT]):
    """Protocol for local-energy evaluators over an open set of terms.

    Evaluators consume an explicit typed context, never an arbitrary mapping.
    Each evaluator declares the narrow context type it requires; the naive
    evaluator below is the permanent numerical reference for every future
    optimized evaluator.
    """

    def make_context(self, wavefunction: object, batch: ElectronBatch) -> ContextT:
        """Build the typed context consumed by this evaluator."""
        ...

    def validate_for_generator(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        wavefunction: object,
        generator: object,
    ) -> None:
        """Validate evaluator eligibility before a generator starts sampling."""
        ...

    def evaluate(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: ContextT,
        *,
        return_terms: bool = False,
    ) -> torch.Tensor | LocalEnergyResult:
        """Evaluate the local energy for one typed context."""
        ...


class NaiveLocalEnergyEvaluator(LocalEnergyEvaluator[NaiveLocalEnergyContext]):
    """Reference evaluator: ordered per-term autodiff evaluation and summation.

    Owns the aggregation loop previously inlined in `local_energy`: term
    normalization, per-term evaluation through the open ``HamiltonianTerm``
    protocol, result validation, named decomposition, and summation — with
    unchanged arithmetic and ordering. This path remains the independent
    numerical reference for any later optimized evaluator.
    """

    def make_context(self, wavefunction: object, batch: ElectronBatch) -> NaiveLocalEnergyContext:
        """Build the typed context consumed by the naive evaluator."""

        return NaiveLocalEnergyContext(wavefunction=wavefunction, batch=batch)

    def validate_for_generator(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        wavefunction: object,
        generator: object,
    ) -> None:
        """Declare that the reference evaluator has no preflight requirements."""

        del terms, wavefunction, generator

    def evaluate(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: NaiveLocalEnergyContext,
        *,
        return_terms: bool = False,
    ) -> torch.Tensor | LocalEnergyResult:
        """Evaluate the summed (optionally decomposed) local energy."""

        if not isinstance(context, NaiveLocalEnergyContext):
            raise TypeError(
                "NaiveLocalEnergyEvaluator requires a NaiveLocalEnergyContext, "
                f"got {type(context).__name__}"
            )
        normalized = normalize_hamiltonian_terms(terms)
        batch = context.batch
        flat = batch.flatten_samples()
        batch_size = flat.batch_size
        total: torch.Tensor | None = None
        decomposition: dict[str, torch.Tensor] = {}
        wavefunction_output: WavefunctionOutput | None = None
        per_electron_kinetic: torch.Tensor | None = None
        for name, term in normalized.items():
            result = term.local_energy(context.wavefunction, batch)
            result = _validate_local_energy_result(
                name,
                result,
                batch_size=batch_size,
                n_electrons=flat.n_electrons,
            )
            decomposition[name] = result.total
            total = result.total if total is None else total + result.total
            if result.wavefunction_output is not None:
                if wavefunction_output is not None:
                    raise ValueError(
                        "local-energy evaluation produced more than one wavefunction output; "
                        "trajectory records require one model evaluation source"
                    )
                wavefunction_output = result.wavefunction_output
            if result.per_electron_kinetic is not None:
                if per_electron_kinetic is not None:
                    raise ValueError(
                        "local-energy evaluation produced more than one per-electron "
                        "kinetic attribution"
                    )
                per_electron_kinetic = result.per_electron_kinetic
        if total is None:
            flat = batch.flatten_samples()
            total = torch.zeros(flat.batch_size, device=flat.device, dtype=flat.dtype)
        if return_terms:
            return LocalEnergyResult(
                total=total,
                terms=decomposition,
                wavefunction_output=wavefunction_output,
                per_electron_kinetic=per_electron_kinetic,
            )
        return total


class AnalyticCuspEvaluator(LocalEnergyEvaluator[AnalyticCuspContext]):
    """Evaluate the fused analytic electron-nucleus local-energy contribution.

    The vectorized :meth:`evaluate` implementation and the deliberately slow
    :meth:`evaluate_reference` implementation share only typed setup and
    validation. Keeping the arithmetic separate makes the reference an
    executable oracle for the optimized kernel.
    """

    fused_term_name = "kinetic_plus_electron_nucleus"

    def make_context(self, wavefunction: object, batch: ElectronBatch) -> AnalyticCuspContext:
        """Build the typed context consumed by the analytic evaluator."""

        return AnalyticCuspContext(wavefunction=wavefunction, batch=batch)

    def validate_for_generator(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        wavefunction: object,
        generator: object,
    ) -> None:
        """Validate evaluator eligibility before a generator starts sampling."""

        del generator
        self._validate_configuration(terms, wavefunction)

    def _validate_configuration(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        wavefunction: object,
    ) -> tuple[dict[str, Any], object]:
        """Validate configuration-only requirements and return participants."""

        from tpen.nn.cusp import ElectronNucleusCusp

        normalized = normalize_hamiltonian_terms(terms)
        kinetic = [term for term in normalized.values() if _declared_operator_id(term) == KINETIC_ENERGY]
        potentials = [
            term for term in normalized.values() if _declared_operator_id(term) == ELECTRON_NUCLEUS_COULOMB
        ]
        if len(kinetic) != 1:
            raise ValueError(
                "analytic cusp evaluation requires exactly one term declaring operator "
                f"{KINETIC_ENERGY}"
            )
        if len(potentials) != 1:
            raise ValueError(
                "analytic cusp evaluation requires exactly one term declaring operator "
                f"{ELECTRON_NUCLEUS_COULOMB}"
            )
        potential = potentials[0]
        if potential.eps != 0:
            raise ValueError(
                "analytic cusp evaluation requires the electron-nucleus Coulomb term eps == 0"
            )

        provider = wavefunction.analytic_cusp_provider
        if not isinstance(provider, ElectronNucleusCusp):
            raise ValueError(
                "analytic cusp evaluation requires one explicitly bound ElectronNucleusCusp provider"
            )
        factorized = wavefunction.factorized_local_energy_input
        if not callable(factorized):
            raise ValueError("analytic cusp evaluation requires factorized_local_energy_input capability")
        if provider.atoms.spatial_dim != 3:
            raise ValueError("analytic cusp evaluation requires spatial dimension 3")
        declared_geometry = declared_operator_geometry(potential)
        if declared_geometry is not None and not strict_equal_atomic_configurations(
            provider.atoms,
            declared_geometry,
            left_label="provider atoms",
            right_label="declared geometry",
        ):
            raise ValueError(
                "analytic cusp electron-nucleus term geometry must agree exactly "
                "with the ElectronNucleusCusp provider atoms"
            )
        return normalized, provider

    def evaluate(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: AnalyticCuspContext,
        *,
        return_terms: bool = False,
    ) -> torch.Tensor | LocalEnergyResult:
        """Evaluate with the batched analytic kernel."""

        return self._evaluate(terms, context, return_terms=return_terms, vectorized=True)

    def evaluate_reference(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: AnalyticCuspContext,
        *,
        return_terms: bool = False,
    ) -> torch.Tensor | LocalEnergyResult:
        """Evaluate with the independent loop-based oracle."""

        return self._evaluate(terms, context, return_terms=return_terms, vectorized=False)

    def _evaluate(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: AnalyticCuspContext,
        *,
        return_terms: bool = False,
        vectorized: bool,
    ) -> torch.Tensor | LocalEnergyResult:
        if not isinstance(context, AnalyticCuspContext):
            raise TypeError(
                "AnalyticCuspEvaluator requires an AnalyticCuspContext, "
                f"got {type(context).__name__}"
            )
        from tpen.nn.cusp import ElectronNucleusCusp
        from tpen.physics.potential import _validate_batch_atoms_context

        normalized, provider = self._validate_configuration(terms, context.wavefunction)
        if not isinstance(provider, ElectronNucleusCusp):
            raise TypeError("analytic cusp provider validation returned an invalid provider")

        flat = context.batch.flatten_samples()
        if flat.spatial_dim != 3:
            raise ValueError("analytic cusp evaluation requires spatial dimension 3")
        _validate_batch_atoms_context(provider.atoms, flat, term_name="analytic cusp provider")
        positions = flat.positions.detach().clone().requires_grad_(True)
        probe = ElectronBatch(
            positions=positions,
            system=flat.system,
            nuclear_positions=flat.nuclear_positions,
            nuclear_charges=flat.nuclear_charges,
            atomic_configuration=flat.atomic_configuration,
            spins=flat.spins,
            aux=dict(flat.aux),
        )
        factorized_input = context.wavefunction.factorized_local_energy_input(probe)
        from tpen.data.batch import FactorizedLocalEnergyInput

        if not isinstance(factorized_input, FactorizedLocalEnergyInput):
            raise TypeError("factorized_local_energy_input must return FactorizedLocalEnergyInput")
        factorized_input.validate(batch_size=flat.batch_size)
        evaluation = factorized_input.electron_nucleus_cusp_evaluation
        if evaluation.n_electrons != flat.n_electrons or evaluation.n_nuclei != provider.atoms.n_nuclei:
            raise ValueError("analytic cusp evaluation geometry does not match the electron batch")
        if evaluation.displacement.shape[-1] != 3:
            raise ValueError("analytic cusp evaluation requires spatial dimension 3")
        atoms_on_device = provider.atoms.to(device=positions.device, dtype=positions.dtype)
        expected_displacement = positions.unsqueeze(2) - atoms_on_device.positions.view(1, 1, -1, 3)
        if not torch.equal(evaluation.displacement, expected_displacement):
            raise ValueError("analytic cusp evaluation displacement does not match the bound atoms")
        if not torch.equal(evaluation.distance, expected_displacement.norm(dim=-1)):
            raise ValueError("analytic cusp evaluation distance does not match its displacement")
        if torch.any(evaluation.distance <= 0):
            raise ValueError("analytic cusp evaluation does not support electron-nucleus coalescence")
        if not torch.equal(evaluation.nuclear_charges, provider.atoms.charges.to(evaluation.displacement)):
            raise ValueError("analytic cusp evaluation provider charges do not match its atoms")
        expected_slope = -evaluation.nuclear_charges
        if not torch.equal(evaluation.origin_radial_slope, expected_slope):
            raise ValueError("analytic cusp evaluation provider origin slopes do not match nuclear charges")

        regular = factorized_input.regular_wavefunction_output
        if regular.phase is not None:
            raise ValueError("analytic cusp evaluation requires a real wavefunction")
        if (
            not torch.isfinite(regular.logabs).all()
            or not torch.isfinite(regular.sign).all()
            or torch.any(regular.sign == 0)
        ):
            raise ValueError("analytic cusp evaluation requires an off-node real wavefunction")
        # Keep constant and affine regular factors in a second-derivative graph.
        differentiable_logabs = regular.logabs + positions.square().sum(dim=(1, 2)) * 0.0
        gradient = torch.autograd.grad(differentiable_logabs.sum(), positions, create_graph=True)[0]
        if gradient.shape != positions.shape:
            raise ValueError("regular wavefunction gradient must match electron positions")

        if vectorized:
            n_coordinates = flat.n_electrons * 3
            if n_coordinates:
                basis = torch.eye(n_coordinates, device=positions.device, dtype=positions.dtype)
                basis = basis.reshape(n_coordinates, 1, flat.n_electrons, 3)
                basis = basis.expand(-1, flat.batch_size, -1, -1)
                hessian = torch.autograd.grad(
                    gradient,
                    positions,
                    grad_outputs=basis,
                    is_grads_batched=True,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                electron_indices = torch.arange(n_coordinates, device=positions.device) // 3
                coordinate_indices = torch.arange(n_coordinates, device=positions.device) % 3
                diagonal = hessian[
                    torch.arange(n_coordinates, device=positions.device),
                    :,
                    electron_indices,
                    coordinate_indices,
                ]
                laplacian = diagonal.transpose(0, 1).reshape(flat.batch_size, flat.n_electrons, 3).sum(dim=-1)
            else:
                laplacian = torch.zeros(
                    (flat.batch_size, flat.n_electrons),
                    device=positions.device,
                    dtype=positions.dtype,
                )
            cusp_gradient = (
                evaluation.radial_first_derivative.unsqueeze(-1)
                * evaluation.displacement
                / evaluation.distance.unsqueeze(-1)
            ).sum(dim=2)
            total_gradient = gradient + cusp_gradient
            fused = -0.5 * (
                laplacian.sum(dim=-1) + total_gradient.square().sum(dim=-1).sum(dim=-1)
            )
            fused = fused + evaluation.local_energy_pair().sum(dim=(1, 2))
        else:
            # Independent executable oracle: keep sample/electron/coordinate/
            # nucleus loops so the fast kernel has a real reference.
            fused = torch.zeros(flat.batch_size, device=positions.device, dtype=positions.dtype)
            for sample in range(flat.batch_size):
                for electron in range(flat.n_electrons):
                    laplacian = torch.zeros((), device=positions.device, dtype=positions.dtype)
                    for coordinate in range(3):
                        second = torch.autograd.grad(
                            gradient[:, electron, coordinate].sum(),
                            positions,
                            create_graph=True,
                            retain_graph=True,
                        )[0]
                        laplacian = laplacian + second[sample, electron, coordinate]
                    cusp_gradient = torch.zeros(3, device=positions.device, dtype=positions.dtype)
                    for nucleus in range(evaluation.n_nuclei):
                        cusp_gradient = cusp_gradient + (
                            evaluation.radial_first_derivative[sample, electron, nucleus]
                            * evaluation.displacement[sample, electron, nucleus]
                            / evaluation.distance[sample, electron, nucleus]
                        )
                    total_gradient = gradient[sample, electron] + cusp_gradient
                    fused[sample] = fused[sample] - 0.5 * (
                        laplacian + total_gradient.square().sum()
                    )
                fused[sample] = fused[sample] + evaluation.local_energy_pair()[sample].sum()

        if fused.shape != (flat.batch_size,):
            raise ValueError(
                "analytic cusp fused energy must have shape "
                f"{(flat.batch_size,)}, got {tuple(fused.shape)}"
            )

        full_output = WavefunctionOutput(
            logabs=regular.logabs + evaluation.pair_value.sum(dim=(1, 2)),
            sign=regular.sign,
            phase=regular.phase,
            aux=dict(regular.aux),
        )
        return self._aggregate(
            normalized,
            context.batch,
            context.wavefunction,
            fused,
            full_output,
            return_terms=return_terms,
        )

    def _aggregate(
        self,
        normalized: Mapping[str, Any],
        batch: ElectronBatch,
        wavefunction: object,
        fused: torch.Tensor,
        full_output: WavefunctionOutput,
        *,
        return_terms: bool,
    ) -> torch.Tensor | LocalEnergyResult:
        flat = batch.flatten_samples()
        decomposition: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        inserted = False
        per_electron: torch.Tensor | None = None
        for name, term in normalized.items():
            if _declared_operator_id(term) in (KINETIC_ENERGY, ELECTRON_NUCLEUS_COULOMB):
                if not inserted:
                    decomposition[self.fused_term_name] = fused
                    total = fused if total is None else total + fused
                    inserted = True
                continue
            result = _validate_local_energy_result(
                name,
                term.local_energy(wavefunction, batch),
                batch_size=flat.batch_size,
                n_electrons=flat.n_electrons,
            )
            decomposition[name] = result.total
            total = result.total if total is None else total + result.total
            if result.wavefunction_output is not None:
                raise ValueError("analytic cusp evaluation cannot combine another wavefunction output")
            if result.per_electron_kinetic is not None:
                if per_electron is not None:
                    raise ValueError(
                        "local-energy evaluation produced more than one per-electron kinetic attribution"
                    )
                per_electron = result.per_electron_kinetic
        if total is None:
            total = fused
        if return_terms:
            return LocalEnergyResult(
                total=total,
                terms=decomposition,
                wavefunction_output=full_output,
                per_electron_kinetic=per_electron,
            )
        return total


_NAIVE_EVALUATOR = NaiveLocalEnergyEvaluator()


def local_energy(
    terms: Mapping[Any, Any] | Sequence[Any],
    wavefunction,
    batch: ElectronBatch,
    *,
    return_terms: bool = False,
) -> torch.Tensor | LocalEnergyResult:
    """Evaluate the local energy of a collection of Hamiltonian terms.

    Parameters
    ----------
    terms : Mapping or Sequence of HamiltonianTerm
        Hamiltonian contributions to sum. A ``dict[str, HamiltonianTerm]`` names
        terms by its non-empty string keys; a sequence falls back to snake-case
        class names (see `normalize_hamiltonian_terms`).
    wavefunction : callable
        Wavefunction model or exact reference returning ``WavefunctionOutput``.
    batch : ElectronBatch
        Electron configuration batch.
    return_terms : bool, optional
        If ``True``, return a ``LocalEnergyResult`` whose ``terms`` decomposition
        is keyed by the resolved (unique) term names; otherwise return the summed
        tensor directly.

    Returns
    -------
    torch.Tensor or LocalEnergyResult
        Summed local energy with shape ``[batch]``, or a decomposed result when
        ``return_terms=True``.
    """

    # Explicit delegate: this entry point stays the single default path used
    # by training and evaluation; the naive evaluator owns the aggregation.
    context = NaiveLocalEnergyContext(wavefunction=wavefunction, batch=batch)
    return _NAIVE_EVALUATOR.evaluate(terms, context, return_terms=return_terms)


def evaluate_local_energy_in_chunks(
    terms: Mapping[str, HamiltonianTerm],
    wavefunction,
    batch: ElectronBatch,
    *,
    return_terms: bool = False,
    chunk_size: int | None = None,
    evaluator: LocalEnergyEvaluator | None = None,
) -> torch.Tensor | LocalEnergyResult:
    """Evaluate local energy on bounded flattened batches.

    Peak device memory of the autodiff kinetic term grows with the number of
    configurations evaluated at once. This executor bounds it by evaluating
    consecutive slices of the flattened sample axis and concatenating the
    detached per-configuration values in order.

    Parameters
    ----------
    terms : Mapping[str, HamiltonianTerm]
        Hamiltonian contributions, as accepted by `normalize_hamiltonian_terms`.
    wavefunction : callable
        Wavefunction model or exact reference returning ``WavefunctionOutput``.
    batch : ElectronBatch
        Electron configuration batch; all sample axes are flattened first.
    return_terms : bool, optional
        If ``True``, return a ``LocalEnergyResult`` with the ordered named
        decomposition, the signed-log values (``logabs`` and ``sign`` only),
        and any per-electron kinetic attribution.
    chunk_size : int or None, optional
        Maximum configurations per evaluation. ``None`` or a non-positive value
        evaluates the whole flattened batch as one chunk.
    evaluator : LocalEnergyEvaluator or None, optional
        Evaluator applied to each chunk; defaults to the naive reference.

    Returns
    -------
    torch.Tensor or LocalEnergyResult
        Detached local energy with shape ``[n_configurations]``, or the
        detached decomposition when ``return_terms=True``.

    Notes
    -----
    Every returned tensor is detached, so no gradient flows through local
    energy. That is the VMC score-function contract: the objective
    differentiates only ``logabs`` from a separate forward pass.
    """

    flat = batch.flatten_samples()
    batch_size = flat.batch_size
    if batch_size == 0:
        total = torch.empty(0, device=flat.device, dtype=flat.dtype)
        return LocalEnergyResult(total=total, terms={}) if return_terms else total

    size = batch_size if chunk_size is None or int(chunk_size) <= 0 else int(chunk_size)
    total_chunks: list[torch.Tensor] = []
    term_chunks: dict[str, list[torch.Tensor]] = {}
    logabs_chunks: list[torch.Tensor] = []
    sign_chunks: list[torch.Tensor] = []
    per_electron_kinetic_chunks: list[torch.Tensor] = []
    term_order: tuple[str, ...] | None = None
    captured_wavefunction = False
    captured_per_electron_kinetic = False
    selected_evaluator = _NAIVE_EVALUATOR if evaluator is None else evaluator
    for start in range(0, batch_size, size):
        chunk = slice_flat_batch(flat, start, min(start + size, batch_size))
        result = selected_evaluator.evaluate(
            terms,
            selected_evaluator.make_context(wavefunction, chunk),
            return_terms=return_terms,
        )
        value = None
        if return_terms:
            if not isinstance(result, LocalEnergyResult):
                raise TypeError("local_energy(return_terms=True) must return LocalEnergyResult")
            chunk_terms = tuple(result.terms)
            if term_order is None:
                term_order = chunk_terms
            elif chunk_terms != term_order:
                raise ValueError("chunked local-energy terms changed between chunks")
            total_chunks.append(result.total.detach())
            for name, value in result.terms.items():
                term_chunks.setdefault(name, []).append(value.detach())
            output = result.wavefunction_output
            if output is not None:
                captured_wavefunction = True
                # Keep only the signed-log primitives. ``aux`` and the original
                # output may retain the derivative graph used by the kinetic
                # term, so neither may escape the current chunk.
                logabs_chunks.append(output.logabs.detach().reshape(-1))
                sign_chunks.append(output.sign.detach().reshape(-1))
            if result.per_electron_kinetic is not None:
                captured_per_electron_kinetic = True
                per_electron_kinetic_chunks.append(
                    result.per_electron_kinetic.detach()
                )
        else:
            if not isinstance(result, torch.Tensor):
                raise TypeError("local_energy(return_terms=False) must return a torch.Tensor")
            total_chunks.append(result.detach())
            output = None

        # The evaluator's result, output, and term tensors can own the
        # derivative graph.  Return only detached tensors above, then end all
        # bindings to graph-bearing values before the next evaluator call.
        del result, output, value, chunk
    total = torch.cat(total_chunks, dim=0)
    if not return_terms:
        return total
    wavefunction_output = None
    if captured_wavefunction:
        if len(logabs_chunks) != len(total_chunks):
            raise ValueError("wavefunction output was not produced for every local-energy chunk")
        wavefunction_output = WavefunctionOutput(
            logabs=torch.cat(logabs_chunks, dim=0),
            sign=torch.cat(sign_chunks, dim=0),
        )
    per_electron_kinetic = None
    if captured_per_electron_kinetic:
        if len(per_electron_kinetic_chunks) != len(total_chunks):
            raise ValueError(
                "per-electron kinetic attribution was not produced for every "
                "local-energy chunk"
            )
        per_electron_kinetic = torch.cat(
            per_electron_kinetic_chunks,
            dim=0,
        )
    return LocalEnergyResult(
        total=total,
        terms={name: torch.cat(chunks, dim=0) for name, chunks in term_chunks.items()},
        wavefunction_output=wavefunction_output,
        per_electron_kinetic=per_electron_kinetic,
    )


def _validate_local_energy_result(
    name: str,
    result: object,
    *,
    batch_size: int,
    n_electrons: int,
) -> LocalEnergyResult:
    """Validate the object returned by one Hamiltonian term."""

    if not isinstance(result, LocalEnergyResult):
        raise TypeError(
            f"hamiltonian term {name!r} must return LocalEnergyResult, got {type(result).__name__}"
        )
    if not isinstance(result.total, torch.Tensor):
        raise TypeError(f"hamiltonian term {name!r} total must be a torch.Tensor")
    expected_shape = (batch_size,)
    if tuple(result.total.shape) != expected_shape:
        raise ValueError(
            f"hamiltonian term {name!r} total must have shape {expected_shape}, "
            f"got {tuple(result.total.shape)}"
        )
    if not isinstance(result.terms, Mapping):
        raise TypeError(f"hamiltonian term {name!r} terms must be a mapping")
    for term_name, value in result.terms.items():
        if not isinstance(term_name, str) or not term_name.strip():
            raise ValueError(f"hamiltonian term {name!r} returned an empty decomposition name")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"hamiltonian term {name!r} decomposition {term_name!r} must be a torch.Tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"hamiltonian term {name!r} decomposition {term_name!r} must have shape "
                f"{expected_shape}, got {tuple(value.shape)}"
            )
    if result.wavefunction_output is not None:
        if not isinstance(result.wavefunction_output, WavefunctionOutput):
            raise TypeError(
                f"hamiltonian term {name!r} wavefunction_output must be a WavefunctionOutput"
            )
        result.wavefunction_output.validate(batch_size=batch_size)
    if result.per_electron_kinetic is not None:
        attribution = result.per_electron_kinetic
        if not isinstance(attribution, torch.Tensor):
            raise TypeError(
                f"hamiltonian term {name!r} per_electron_kinetic must be a torch.Tensor"
            )
        expected_attribution_shape = (batch_size, n_electrons)
        if tuple(attribution.shape) != expected_attribution_shape:
            raise ValueError(
                f"hamiltonian term {name!r} per_electron_kinetic must have shape "
                f"{expected_attribution_shape}, got {tuple(attribution.shape)}"
            )
        if attribution.device != result.total.device or attribution.dtype != result.total.dtype:
            raise ValueError(
                f"hamiltonian term {name!r} per_electron_kinetic must match total "
                "dtype/device"
            )
    return result


def summarize_local_energy(
    result: LocalEnergyResult | torch.Tensor,
) -> dict[str, Any]:
    """Summarize a sampled local energy into scalar logging metrics.

    Handles all-finite, partially-nonfinite, all-nonfinite, and empty inputs,
    and per-term decompositions. Returned keys follow the metric naming
    convention: callers provide the logging namespace, while this helper emits
    flat leaf keys such as ``energy``, ``local_energy_n_finite``, and
    ``energy_term_{name}``. This summary is reference-free; benchmark
    comparison belongs to evaluation diagnostics.

    Parameters
    ----------
    result : LocalEnergyResult or torch.Tensor
        Per-sample local energy, optionally with a per-term decomposition.

    Returns
    -------
    dict
        Scalar metrics. When no finite samples exist, ``energy`` and
        ``energy_variance`` are ``nan`` and ``energy_stderr`` is ``inf``.

    Notes
    -----
    ``energy_stderr`` and every ``{prefix}_stderr`` here is an **IID-only**
    standard error, ``sigma / sqrt(N)``, identical in meaning to the metric of
    the same name from `tpen.training.vmc.compute_vmc_objective`. It ignores serial
    correlation between MCMC walkers and so understates the true uncertainty by
    roughly ``sqrt(tau_int)``.

    The correlation-aware quantity is the MCSE from
    :func:`tpen.statistics.produce_trajectory_statistics`, which needs a
    ``[draw, walker]`` trajectory rather than the flat sample tensor available
    here. Do not reinterpret this metric as an MCSE.
    """

    if isinstance(result, LocalEnergyResult):
        eloc, terms = result.total, result.terms
    else:
        eloc, terms = result, {}

    metrics = _summarize_values("", eloc)
    for name, value in terms.items():
        metrics.update(_summarize_values(f"energy_term_{name}", value))
    return metrics


def _summarize_values(prefix: str, values: torch.Tensor) -> dict[str, Any]:
    """Return canonical finite-aware energy metrics for one value tensor.

    ``{prefix}_stderr`` is an **IID-only** standard error; see the Notes on
    `summarize_local_energy`. It is never an MCSE.
    """

    n_total = int(values.numel())
    finite_mask = torch.isfinite(values)
    n_finite = int(finite_mask.sum().item())
    if n_finite > 0:
        finite = values[finite_mask]
        energy = float(finite.mean().item())
        variance = float(finite.var(unbiased=False).item()) if n_finite > 1 else 0.0
        std = math.sqrt(variance)
        stderr = std / math.sqrt(n_finite)
    else:
        energy = float("nan")
        variance = float("nan")
        std = float("nan")
        stderr = float("inf")

    if prefix:
        return {
            prefix: energy,
            f"{prefix}_variance": variance,
            f"{prefix}_std": std,
            f"{prefix}_stderr": stderr,
            f"{prefix}_n_finite": n_finite,
            f"{prefix}_n_total": n_total,
            f"{prefix}_finite_fraction": float(n_finite / n_total) if n_total else 0.0,
            f"{prefix}_nonfinite_count": n_total - n_finite,
        }

    return {
        "energy": energy,
        "energy_variance": variance,
        "energy_std": std,
        "energy_stderr": stderr,
        "local_energy_n_finite": n_finite,
        "local_energy_n_total": n_total,
        "local_energy_finite_fraction": float(n_finite / n_total) if n_total else 0.0,
        "local_energy_nonfinite_count": n_total - n_finite,
    }


__all__ = [
    "HamiltonianTerm",
    "LocalEnergyEvaluator",
    "LocalEnergyResult",
    "AnalyticCuspContext",
    "AnalyticCuspEvaluator",
    "NaiveLocalEnergyContext",
    "NaiveLocalEnergyEvaluator",
    "evaluate_local_energy_in_chunks",
    "local_energy",
    "normalize_hamiltonian_terms",
    "summarize_local_energy",
]
