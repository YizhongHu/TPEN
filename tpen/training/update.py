"""Typed VMC update inputs and the behavior-preserving legacy adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from tpen.data.batch import (
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    ParameterSlot,
    WavefunctionOutput,
)
from tpen.dependencies import require_torch
from tpen.physics.hamiltonian import local_energy
from tpen.training.vmc import (
    DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    compute_vmc_objective,
    resolve_nonfinite_local_energy_policy,
)

torch = require_torch(feature="VMC update methods")


InputT = TypeVar("InputT")
ScopeFactory = Callable[[int], AbstractContextManager[Any]]


@dataclass(frozen=True, kw_only=True)
class VMCStepData:
    """Common live data produced for one VMC iteration.

    Parameters
    ----------
    batch : ElectronBatch
        Electron configurations used to produce the wavefunction and energy.
    wavefunction : WavefunctionOutput
        The model output for ``batch``.  Its leading shape may be the batch's
        flattened size because the TPEN readout flattens multidimensional
        sample axes.
    local_energy : torch.Tensor
        Per-sample total local energies with the same shape, dtype, and device
        as ``wavefunction.logabs``.

    Notes
    -----
    This record is deliberately live for exactly one update call.  It is never
    assigned to :class:`tpen.training.state.TrainerState`; its serialization
    guard also prevents a graph-bearing record from crossing an artifact
    boundary by accident.
    """

    batch: ElectronBatch
    wavefunction: WavefunctionOutput
    local_energy: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "VMCStepData":
        """Validate the common batch, output, and local-energy contract."""

        if not isinstance(self.batch, ElectronBatch):
            raise TypeError("VMCStepData.batch must be an ElectronBatch")
        if not isinstance(self.wavefunction, WavefunctionOutput):
            raise TypeError("VMCStepData.wavefunction must be a WavefunctionOutput")
        if not isinstance(self.local_energy, torch.Tensor):
            raise TypeError("VMCStepData.local_energy must be a torch.Tensor")
        self.batch.validate()
        # ``batch_size`` rather than ``sample_shape`` is intentional: the
        # Pfaffian readout returns a flat primal output for a multidimensional
        # sample shape, while local energy follows that output shape.
        self.wavefunction.validate(batch_size=self.batch.batch_size)
        if self.local_energy.shape != self.wavefunction.logabs.shape:
            raise ValueError(
                "VMCStepData.local_energy must have the same shape as wavefunction.logabs, "
                f"got {tuple(self.local_energy.shape)} and {tuple(self.wavefunction.logabs.shape)}"
            )
        if not self.local_energy.is_floating_point():
            raise TypeError("VMCStepData.local_energy must have a real floating dtype")
        if self.local_energy.device != self.wavefunction.logabs.device:
            raise ValueError("VMCStepData local_energy and wavefunction must share one device")
        if self.local_energy.dtype != self.wavefunction.logabs.dtype:
            raise ValueError("VMCStepData local_energy and wavefunction must share one dtype")
        if self.batch.device != self.wavefunction.logabs.device:
            raise ValueError("VMCStepData batch and wavefunction must share one device")
        if self.batch.dtype != self.wavefunction.logabs.dtype:
            raise ValueError("VMCStepData batch and wavefunction must share one dtype")
        return self

    def __getstate__(self) -> dict[str, Any]:
        """Reject serialization while any common step value is graph-live."""

        if _batch_requires_grad(self.batch) or _output_requires_grad(self.wavefunction):
            raise RuntimeError("graph-bearing VMCStepData cannot be serialized")
        if self.local_energy.requires_grad:
            raise RuntimeError("graph-bearing VMCStepData cannot be serialized")
        return self.__dict__


@dataclass(frozen=True, kw_only=True)
class ObjectiveReevaluation:
    """Recompute one step's objective at the CURRENT parameter values.

    A TRUE closure optimizer -- ``torch.optim.LBFGS`` is the archetype -- calls
    its closure repeatedly inside a single ``step()`` and mutates parameters in
    place between those calls.  Calling ``backward()`` a second time on the
    retained graph of an already-materialized objective raises after such a
    mutation: the graph's saved tensors are versioned, and the in-place step
    bumps that version out from under the retained backward.  A materialized
    scalar objective with no means to recompute it is therefore not merely
    inconvenient for such an optimizer, it is unusable by one.  This record is
    the seam that removes the limit; every call builds a NEW graph from the
    step's fixed sample.

    Parameters
    ----------
    recompute : callable
        Zero-argument callable returning a fresh scalar objective for this
        step's fixed sample at the parameters live at call time.
    dtype : torch.dtype
        Real floating dtype every recomputed objective must have.
    device : torch.device
        Device every recomputed objective must be on.

    Notes
    -----
    WHAT THIS DELIBERATELY DOES NOT DO.  It does not zero gradients, call
    ``backward()``, or step an optimizer.  Those belong to the update method,
    which is where :class:`LegacyAutogradUpdate` already keeps them; a seam
    that ran them would move optimizer policy into the trainer.  A method
    builds its own closure around this callable.

    STATED PROPERTIES.  These are properties of the seam, not incidental
    behavior, and each is covered by a named test in
    ``tests/unit/training/test_update_reevaluation.py``:

    FIXED-SAMPLE.  Every re-evaluation within one step uses the identical
    sample.  This record holds no sampler and no walker source, so it cannot
    resample even by mistake -- the guarantee is structural rather than a rule
    a caller must remember.

    RNG-INERT.  A re-evaluation advances neither the sampler-local generator
    nor the global torch RNG.  This matters beyond tidiness: TPEN's rank-local
    sampler/walker/RNG resume policy is tested for bitwise reproducibility, so
    a seam that advanced RNG per re-evaluation would make a resumed run diverge
    from the uninterrupted one purely because an optimizer looked at the
    objective twice.

    SAME-FUNCTION, SCOPED TO UNCHANGED PARAMETERS.  At unchanged parameters,
    repeated re-evaluations return bitwise-equal objectives.  The scope is not
    a hedge; see the mask caveat below for why the property is not claimed at
    mutated parameters.

    POLICY-CONTINUOUS.  The recomputed objective uses the SAME resolved
    non-finite local-energy policy as the step's primary objective.  Falling
    back to the module default here would run one step under two estimators,
    which is exactly what the checkpoint continuity check in
    :class:`~tpen.training.trainer.VMCTrainer` exists to prevent.

    DECLARED PRECONDITION -- RNG-INERTNESS HOLDS BY ABSENCE, NOT BY A GUARD.
    Nothing on the recompute path (``model(batch)``,
    :func:`~tpen.physics.hamiltonian.local_energy`,
    :func:`~tpen.training.vmc.compute_vmc_objective`) draws from an RNG today,
    because ``tpen/physics/`` contains no generator at all.  The named tests
    are the enforcement.  TRIGGER: the day ``tpen/physics/`` acquires a
    generator -- a stochastic kinetic estimator being the obvious candidate --
    the same-function test is the one that must go red.

    ``torch.random.fork_rng()`` was considered and rejected.  It would make
    RNG-inertness true by construction, and would thereby HIDE the day that
    premise stops holding: a future stochastic local-energy estimator should
    fail the same-function test loudly rather than be silently pinned.  A loud
    failure is chosen over a silent correctness.

    DECLARED PRECONDITION -- UNDER POLICY ``"mask"`` THE ESTIMATOR'S SUBSAMPLE
    IS PARAMETER-DEPENDENT.  :func:`~tpen.training.vmc.compute_vmc_objective`
    excludes non-finite local-energy rows, and which rows are non-finite
    depends on the parameter values.  A closure optimizer re-evaluates at
    DIFFERENT parameters, so the finite-row set can SHIFT between closure
    calls and successive calls may return values drawn from different
    subsamples -- meaning a line search would be comparing different
    functions.  That is not a defect of this seam: it is the known-biased
    estimator that ``"mask"`` explicitly selects, meeting a line search.

    WHETHER A LINE SEARCH OVER A MASK-BIASED ESTIMATOR IS SOUND AT ALL IS AN
    OPEN QUESTION THIS SEAM SURFACES AND DOES NOT ANSWER.  It is tracked as
    item ``377f6b0f``, and :func:`select_reevaluation_rows` is the ONE place a
    ruling on it changes -- deliberately a named decision rather than a
    property that falls out of how the recompute is wired.  Scoping the
    same-function property to unchanged parameters records the boundary; it
    does not dispose of the question.  The decision belongs to whoever admits
    the optimizers that consume this seam (the staged optimizer-admission lane
    covering SPRING, KFAC, and the linear method).  Under ``"fail"`` the
    situation is different and simpler: a re-evaluation that meets a
    non-finite row raises out of the closure, from inside ``step(closure)``,
    and the step does not apply.
    """

    recompute: Callable[[], torch.Tensor]
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        if not callable(self.recompute):
            raise TypeError("ObjectiveReevaluation.recompute must be callable")
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            raise TypeError("ObjectiveReevaluation.dtype must be a real floating torch.dtype")
        if not isinstance(self.device, torch.device):
            raise TypeError("ObjectiveReevaluation.device must be a torch.device")

    def __call__(self) -> torch.Tensor:
        """Return a fresh scalar objective at the parameters live right now.

        Returns
        -------
        torch.Tensor
            Scalar objective on a newly built graph.

        Notes
        -----
        ``requires_grad`` is deliberately NOT required here.  A vacuum batch
        legitimately produces a disconnected objective, and the update method
        already distinguishes that case from a genuinely disconnected loss;
        requiring a live graph at the seam would turn the method's considered
        decision into a seam-level crash.
        """

        objective = self.recompute()
        if not isinstance(objective, torch.Tensor):
            raise TypeError("ObjectiveReevaluation must recompute a torch.Tensor")
        if objective.ndim != 0:
            raise ValueError(
                "ObjectiveReevaluation must recompute a scalar objective, "
                f"got shape {tuple(objective.shape)}"
            )
        if objective.dtype != self.dtype:
            raise ValueError(
                "ObjectiveReevaluation must recompute an objective with dtype "
                f"{self.dtype}, got {objective.dtype}"
            )
        if objective.device != self.device:
            raise ValueError(
                "ObjectiveReevaluation must recompute an objective on device "
                f"{self.device}, got {objective.device}"
            )
        return objective

    def __getstate__(self) -> dict[str, Any]:
        """Reject serialization of a callable closed over a live model."""

        raise RuntimeError("live ObjectiveReevaluation cannot be serialized")


def select_reevaluation_rows(
    *,
    primary_finite_mask: torch.Tensor,
    recomputed_local_energy: torch.Tensor,
    policy: str,
) -> torch.Tensor | None:
    """THE MASK DECISION POINT for re-evaluation.  One line owns it.

    Which finite-row set a re-evaluation uses is a DESIGN DECISION, not a
    consequence of how the recompute happens to be wired, and this function is
    the single place it is made.  It exists so that implementing a ruling on
    open question ``377f6b0f`` is a small change here rather than a rederivation
    of the seam.

    The question it answers: under policy ``"mask"``,
    :func:`~tpen.training.vmc.compute_vmc_objective` excludes non-finite
    local-energy rows, and which rows are non-finite depends on the parameter
    values.  A closure optimizer re-evaluates at DIFFERENT parameters, so the
    excluded set can shift between closure calls and a line search may compare
    values drawn from different subsamples -- that is, different functions.

    Parameters
    ----------
    primary_finite_mask : torch.Tensor
        Boolean mask of the rows that were finite when the step's PRIMARY
        objective was formed.  Detached at construction, so it carries no
        graph and cannot be moved by a re-evaluation.
    recomputed_local_energy : torch.Tensor
        Local energy just recomputed at the current parameters.
    policy : str
        The step's resolved non-finite local-energy policy.

    Returns
    -------
    torch.Tensor or None
        ``None`` -- the current behaviour -- means make no selection here and
        let `compute_vmc_objective` apply `policy` to the recomputed rows,
        per call.  A boolean mask means restrict this re-evaluation to exactly
        those rows.

    Notes
    -----
    HOW EACH POSSIBLE RULING LANDS HERE, so an implementer does not have to
    re-derive it:

    - FIXED ROW MASK PER STEP (the recommendation before the operator at time
      of writing): ``return primary_finite_mask``.  Every re-evaluation within
      one step then scores the same rows, and the objective is one function of
      the parameters for the whole step.  Note the consequence that must be
      stated alongside it: rows that go non-finite mid-step are then INCLUDED
      by the mask and their non-finite values reach the reduction, so a fixed
      mask needs its own answer for that case -- most likely raising, since a
      pinned row that has become non-finite is a genuine failure rather than a
      row to quietly drop.
    - ``"fail"``-ONLY STRICT MODE: no change here.  ``policy == "fail"`` already
      raises inside `compute_vmc_objective` the moment a recomputed row is
      non-finite, which is the refusal, and it surfaces from inside
      ``optimizer.step(closure)``.
    - ACCEPT THE BIAS: no change here.  ``None`` is exactly that answer.

    Returning ``None`` today is deliberately the status quo and NOT an implicit
    ruling: it preserves #483's declared semantics unchanged while the question
    is open.  Tracked as item ``377f6b0f`` under the SR/minSR program,
    cross-linked to ``02859027`` (the parameter-score row drop, which has the
    same selection-bias character and a different cost).
    """

    del primary_finite_mask, recomputed_local_energy, policy
    return None


def vmc_objective_reevaluation(
    *,
    model,
    hamiltonian_terms: Any,
    batch: ElectronBatch,
    primary_local_energy: torch.Tensor,
    nonfinite_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
) -> ObjectiveReevaluation:
    """Build the canonical VMC re-evaluation for one step's fixed sample.

    Parameters
    ----------
    model : callable
        Wavefunction model returning a
        :class:`~tpen.data.batch.WavefunctionOutput`.
    hamiltonian_terms : Mapping or Sequence of HamiltonianTerm
        The step's Hamiltonian contributions, passed through unchanged.
    batch : ElectronBatch
        The step's FIXED sample.  The same object is reused by every call.
    primary_local_energy : torch.Tensor
        Local energy of the step's PRIMARY objective.  Only its finite mask is
        retained, detached, so this argument contributes no graph and cannot
        keep the primary step's graph alive.  It exists to give
        :func:`select_reevaluation_rows` the one input it cannot recompute.
    nonfinite_policy : str, optional
        Resolved non-finite local-energy policy, which must be the one the
        step's primary objective used.

    Returns
    -------
    ObjectiveReevaluation
        Callable recomputing this step's objective at current parameters.

    Notes
    -----
    ``return_terms=False`` is used deliberately.  Per-term local energies are
    metrics, never objective components, and a re-evaluation exists to feed an
    optimizer rather than the metrics record; asking for the decomposition
    would do extra work whose result nothing reads.  The summed total is
    identical either way.
    """

    if not isinstance(batch, ElectronBatch):
        raise TypeError("vmc_objective_reevaluation requires an ElectronBatch")
    if not callable(model):
        raise TypeError("vmc_objective_reevaluation requires a callable model")
    if not isinstance(primary_local_energy, torch.Tensor):
        raise TypeError("vmc_objective_reevaluation requires a primary_local_energy tensor")
    resolved_policy = resolve_nonfinite_local_energy_policy(nonfinite_policy)
    primary_finite_mask = torch.isfinite(primary_local_energy).detach().clone()

    def recompute() -> torch.Tensor:
        # Both the forward and the local energy are recomputed, because a
        # closure optimizer has moved the parameters since the primary
        # objective was formed and a stale factor would silently mix two
        # parameter versions into one scalar.
        output = model(batch)
        total_local_energy = local_energy(
            hamiltonian_terms,
            model,
            batch,
            return_terms=False,
        )
        # The row-set decision is delegated, never inlined: see
        # `select_reevaluation_rows` for why it is a decision and how a ruling
        # on it lands.  Looked up through the module so the decision has one
        # definition and one call site.
        logabs = output.logabs
        selection = select_reevaluation_rows(
            primary_finite_mask=primary_finite_mask,
            recomputed_local_energy=total_local_energy,
            policy=resolved_policy,
        )
        if selection is not None:
            logabs = logabs[selection]
            total_local_energy = total_local_energy[selection]
        return compute_vmc_objective(
            logabs,
            total_local_energy,
            nonfinite_policy=resolved_policy,
        ).loss

    return ObjectiveReevaluation(
        recompute=recompute,
        dtype=batch.dtype,
        device=batch.device,
    )


@dataclass(frozen=True, kw_only=True)
class AutogradUpdateInput(VMCStepData):
    """Typed input for an autograd-backed VMC update."""

    step: int
    objective: torch.Tensor
    # REQUIRED, deliberately not ``| None``.  A materialized objective with no
    # means to recompute it is the exact defect this field exists to close, and
    # an optional field re-admits it: every caller could once again hand an
    # update method an objective no closure optimizer can use.  The cost is a
    # keyword at each construction site; the benefit is that the unusable
    # shape is no longer expressible.
    reevaluate: ObjectiveReevaluation

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "AutogradUpdateInput":
        """Validate the common step data and differentiable scalar objective."""

        super().validate()
        _validate_step(self.step)
        if not isinstance(self.objective, torch.Tensor):
            raise TypeError("AutogradUpdateInput.objective must be a torch.Tensor")
        if self.objective.ndim != 0:
            raise ValueError(
                f"AutogradUpdateInput.objective must be scalar, got shape {tuple(self.objective.shape)}"
            )
        if not self.objective.is_floating_point():
            raise TypeError("AutogradUpdateInput.objective must have a real floating dtype")
        if self.objective.device != self.wavefunction.logabs.device:
            raise ValueError("AutogradUpdateInput objective and wavefunction must share one device")
        if self.objective.dtype != self.wavefunction.logabs.dtype:
            raise ValueError("AutogradUpdateInput objective and wavefunction must share one dtype")
        if not isinstance(self.reevaluate, ObjectiveReevaluation):
            raise TypeError("AutogradUpdateInput.reevaluate must be an ObjectiveReevaluation")
        # The re-evaluation must agree with the primary objective it replaces.
        # A re-evaluation returning float32 where the step is float64 would not
        # fail loudly; it would quietly degrade an optimizer's curvature
        # history one closure call at a time.
        if self.reevaluate.dtype != self.objective.dtype:
            raise ValueError("AutogradUpdateInput reevaluation and objective must share one dtype")
        if self.reevaluate.device != self.objective.device:
            raise ValueError("AutogradUpdateInput reevaluation and objective must share one device")
        return self

    def __getstate__(self) -> dict[str, Any]:
        """Reject serialization: the record holds a live re-evaluation.

        The graph-bearing checks run first so a graph-live record still
        reports that more specific reason, which is the one a caller can act
        on.  The unconditional refusal that follows matches
        :class:`ScoreUpdateInput`: this record now closes over a live model,
        so no instance of it may cross an artifact boundary.
        """

        super().__getstate__()
        if self.objective.requires_grad:
            raise RuntimeError("graph-bearing AutogradUpdateInput cannot be serialized")
        raise RuntimeError("live AutogradUpdateInput cannot be serialized")


@dataclass(frozen=True, kw_only=True)
class ScoreUpdateInput(VMCStepData):
    """Typed input reserved for score-based VMC update methods.

    The first score consumer is intentionally not implemented in this slice.
    Keeping its exact input record here prevents a future SR implementation from
    inventing an optional capability bag or a string-keyed parameter lookup.
    """

    step: int
    parameter_scores: MaterializedParameterLogScores
    parameter_binding: ParameterBinding

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "ScoreUpdateInput":
        """Validate score blocks against the direct model parameter binding."""

        super().validate()
        _validate_step(self.step)
        if not isinstance(self.parameter_scores, MaterializedParameterLogScores):
            raise TypeError(
                "ScoreUpdateInput.parameter_scores must be MaterializedParameterLogScores"
            )
        if not isinstance(self.parameter_binding, ParameterBinding):
            raise TypeError("ScoreUpdateInput.parameter_binding must be a ParameterBinding")
        self.parameter_binding.validate()
        self.parameter_scores.validate(sample_shape=tuple(self.wavefunction.logabs.shape))
        if not self.parameter_scores.layout.compare(self.parameter_binding.layout)[0]:
            raise ValueError("ScoreUpdateInput parameter scores and binding layouts do not match")
        if self.parameter_binding.parameters:
            binding_device = self.parameter_binding.parameters[0].device
            if self.parameter_scores.device != binding_device:
                raise ValueError("ScoreUpdateInput parameter scores and binding must share one device")
        return self

    def __getstate__(self) -> dict[str, Any]:
        """Reject serialization of a record holding direct live parameters."""

        super().__getstate__()
        raise RuntimeError("live ScoreUpdateInput cannot be serialized")


@dataclass(frozen=True, kw_only=True)
class VMCUpdateResult:
    """Result of one update-method invocation."""

    applied: bool
    grad_norm: float

    def __post_init__(self) -> None:
        if type(self.applied) is not bool:
            raise TypeError("VMCUpdateResult.applied must be a bool")
        object.__setattr__(self, "grad_norm", float(self.grad_norm))


@dataclass(frozen=True, kw_only=True)
class ModelParameterBinding:
    """Bind the legacy gradient domain to direct model parameters.

    The static layout is retained separately from the live parameter
    references.  Checkpoint restore can therefore compare the recorded layout
    before rebuilding the binding against the model objects that are live
    after ``load_state_dict``.
    """

    parameters: tuple[torch.nn.Parameter, ...]
    layout: ParameterLayout | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if any(not isinstance(parameter, torch.nn.Parameter) for parameter in self.parameters):
            raise TypeError("ModelParameterBinding.parameters must contain direct parameters")
        if self.layout is None:
            object.__setattr__(self, "layout", _parameter_layout(self.parameters))
        if not isinstance(self.layout, ParameterLayout):
            raise TypeError("ModelParameterBinding.layout must be a ParameterLayout")
        self.validate()

    def validate(self) -> "ModelParameterBinding":
        """Validate that the live references have the recorded layout."""

        assert self.layout is not None
        self.layout.validate()
        if len(self.parameters) != len(self.layout.slots):
            raise ValueError(
                "ModelParameterBinding.parameters must have one reference per layout slot"
            )
        for slot, parameter in zip(self.layout.slots, self.parameters, strict=True):
            if tuple(parameter.shape) != slot.shape:
                raise ValueError(
                    f"ModelParameterBinding slot {slot.ordinal} expected shape {slot.shape}, "
                    f"got {tuple(parameter.shape)}"
                )
            if parameter.numel() != slot.numel:
                raise ValueError(
                    f"ModelParameterBinding slot {slot.ordinal} expected numel {slot.numel}, "
                    f"got {parameter.numel()}"
                )
            if parameter.dtype != slot.dtype:
                raise ValueError(
                    f"ModelParameterBinding slot {slot.ordinal} expected dtype {slot.dtype}, "
                    f"got {parameter.dtype}"
                )
        return self

    def compare(
        self,
        other: "ModelParameterBinding",
    ) -> tuple[bool, dict[str, float]]:
        """Compare layout metadata and direct parameter-reference identity."""

        if type(self) is not type(other) or not self.layout.compare(other.layout)[0]:
            return False, {"max_abs_error": float("inf")}
        close = len(self.parameters) == len(other.parameters) and all(
            left is right for left, right in zip(self.parameters, other.parameters, strict=True)
        )
        return close, {"max_abs_error": 0.0 if close else float("inf")}

    @classmethod
    def from_parameters(
        cls,
        parameters: tuple[torch.nn.Parameter, ...],
    ) -> "ModelParameterBinding":
        """Build a binding whose layout is derived from live parameters."""

        return cls(parameters=tuple(parameters))

    def rebind(
        self,
        parameters: tuple[torch.nn.Parameter, ...],
        *,
        layout: ParameterLayout | None = None,
    ) -> "ModelParameterBinding":
        """Rebuild direct references after checking the expected layout.

        Parameters
        ----------
        parameters : tuple of torch.nn.Parameter
            The current model-owned parameter objects.
        layout : ParameterLayout, optional
            Recorded layout to enforce.  Defaults to this binding's layout.
        """

        parameters = tuple(parameters)
        expected = self.layout if layout is None else layout
        current = _parameter_layout(parameters)
        if not expected.compare(current)[0]:
            layout_mismatch_message = "checkpoint parameter layout does not match live model"
            raise ValueError(layout_mismatch_message)
        return type(self)(layout=current, parameters=parameters)


@dataclass(frozen=True, kw_only=True)
class VMCUpdateState:
    """The single optimizer and parameter binding owned by an update method."""

    optimizer: torch.optim.Optimizer
    model_parameters: ModelParameterBinding

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("VMCUpdateState.optimizer must be a torch.optim.Optimizer")
        if not isinstance(self.model_parameters, ModelParameterBinding):
            raise TypeError("VMCUpdateState.model_parameters must be a ModelParameterBinding")


class VMCUpdateMethod(Generic[InputT], ABC):
    """Nominal typed contract for VMC update strategies.

    Concrete methods own their optimizer/preconditioner state.  The default
    state surface is empty, which lets stateless future methods satisfy the
    contract without inventing checkpoint payloads.  ``set_step_scopes`` is a
    narrow trainer integration hook: it preserves typed Backward and
    OptimizerUpdate event boundaries without adding a context or callback bag
    to the live update input.
    """

    @abstractmethod
    def update(self, update_input: InputT) -> VMCUpdateResult:
        """Apply one update and report whether it returned an applied step."""

    def state_dict(self) -> Mapping[str, Any]:
        """Return this method's checkpointable state, empty by default."""

        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore this method's state, empty by default."""

        if not isinstance(state, Mapping):
            raise TypeError("VMCUpdateMethod state must be a mapping")
        if state:
            raise ValueError("stateless VMCUpdateMethod cannot load non-empty state")

    def update_state(self) -> VMCUpdateState | None:
        """Return owned optimizer state, or ``None`` for a stateless method.

        Returning ``None`` delegates the authority to the optimizer supplied
        to ``VMCTrainer.fit``.  A stateful method must return its one typed
        authority so the trainer can reject an ambiguous legacy optimizer
        before restore or update work begins.
        """

        return None

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        """Accept a rebuilt direct parameter binding after checkpoint restore.

        Stateless methods do not retain a binding.  A stateful method that
        keeps direct parameter references must override this hook so the
        references used by its next update are the restored model's live
        objects.
        """

        del model_parameters

    def method_state_dict(self) -> Mapping[str, Any]:
        """Return JSON-safe PERSISTENT METHOD state, empty by default.

        This is deliberately separate from :meth:`state_dict`. That one has an
        established meaning for the legacy adapter, where it is the raw
        PyTorch optimizer payload kept for checkpoint compatibility -- tensors,
        not JSON, and already persisted independently as ``optimizer.pt``.
        Reusing it for method state would put the optimizer's tensors into the
        trainer's JSON record and duplicate a payload that already has an owner.

        What belongs here is state the METHOD owns and the optimizer does not:
        schedule counters, a layout/convention fingerprint, and any canonical
        warm-start vector. Returning ``{}`` -- the default, and what the legacy
        adapter does -- adds no key to the trainer's checkpoint state, so
        existing checkpoints are unchanged.
        """

        return {}

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore persistent method state, empty by default."""

        if not isinstance(state, Mapping):
            raise TypeError("VMCUpdateMethod method state must be a mapping")
        if state:
            raise ValueError("stateless VMCUpdateMethod cannot load non-empty method state")

    def forward_request(self) -> Any | None:
        """Return the typed forward request this method's input requires.

        Returns
        -------
        WavefunctionForwardRequest or None
            ``None``, the default, means an ordinary value forward
            (``model(batch)``) supplies everything the method needs.  A method
            consuming derivative payloads returns the exact request describing
            them, for example a
            :class:`~tpen.nn.forward.MaterializedParameterScoreRequest`.

        Notes
        -----
        This exists so the trainer can produce the score-bearing forward packet
        **once**, in the single forward it already performs, rather than
        running an ordinary forward and then recomputing derivatives.  A design
        that did the latter would double the forward and derivative work of
        every step.

        The return type is deliberately the request object itself rather than a
        capability flag or a name.  The trainer dispatches on the request's own
        type, so adding a future derivative payload cannot be done by inventing
        a string, and a method cannot claim a capability whose payload it does
        not then receive.
        """

        return None

    def set_step_scopes(
        self,
        *,
        backward_scope: ScopeFactory | None = None,
        optimizer_scope: ScopeFactory | None = None,
    ) -> None:
        """Install optional typed event scopes for one trainer invocation."""

        del backward_scope, optimizer_scope


class LegacyAutogradUpdate(VMCUpdateMethod[AutogradUpdateInput]):
    """Exact adapter for TPEN's historical zero-grad/backward/step sequence."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_norm: float | None = None,
        *,
        model_parameters: ModelParameterBinding,
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("LegacyAutogradUpdate.optimizer must be a torch.optim.Optimizer")
        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError(
                "LegacyAutogradUpdate.model_parameters must be a ModelParameterBinding"
            )
        self.optimizer = optimizer
        self.gradient_clip_norm = None if gradient_clip_norm is None else float(gradient_clip_norm)
        self.model_parameters = model_parameters
        self._backward_scope: ScopeFactory | None = None
        self._optimizer_scope: ScopeFactory | None = None

    def update_state(self) -> VMCUpdateState:
        """Return the optimizer and direct gradient binding owned by the adapter."""

        return VMCUpdateState(
            optimizer=self.optimizer,
            model_parameters=self.model_parameters,
        )

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        """Replace the legacy gradient domain with restored model references."""

        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError("LegacyAutogradUpdate model_parameters must be a ModelParameterBinding")
        self.model_parameters = model_parameters

    def set_step_scopes(
        self,
        *,
        backward_scope: ScopeFactory | None = None,
        optimizer_scope: ScopeFactory | None = None,
    ) -> None:
        """Preserve the trainer's typed phase boundaries around legacy work."""

        self._backward_scope = backward_scope
        self._optimizer_scope = optimizer_scope

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        """Run the historical update sequence without clearing post-step grads."""

        if not isinstance(update_input, AutogradUpdateInput):
            raise TypeError("LegacyAutogradUpdate requires AutogradUpdateInput")

        # This ordering is observable: GradientStats reads the gradients after
        # optimizer.step(), so there is deliberately no post-step zero_grad.
        self.optimizer.zero_grad(set_to_none=True)
        objective = update_input.objective
        if not objective.requires_grad:
            if update_input.batch.n_electrons == 0:
                return VMCUpdateResult(applied=False, grad_norm=0.0)
            raise RuntimeError(
                "VMC loss is disconnected from model parameters for a "
                "nonzero-electron batch"
            )

        self._run_backward(update_input)
        gradient_parameters = self.gradient_params()
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                gradient_parameters, self.gradient_clip_norm
            )
        grad_norm = _gradient_norm(gradient_parameters)
        self._run_optimizer_step(update_input)
        return VMCUpdateResult(applied=True, grad_norm=grad_norm)

    def optimizer_params(self) -> tuple[torch.nn.Parameter, ...]:
        """Return the optimizer's direct parameter references in group order."""

        parameters: list[torch.nn.Parameter] = []
        for group in self.optimizer.param_groups:
            parameters.extend(group["params"])
        return tuple(parameters)

    def gradient_params(self) -> tuple[torch.nn.Parameter, ...]:
        """Return the exact model parameter domain used by legacy gradients."""

        return self.model_parameters.parameters

    def state_dict(self) -> Mapping[str, Any]:
        """Return the raw PyTorch optimizer payload unchanged."""

        return self.optimizer.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load a raw PyTorch optimizer payload for checkpoint compatibility."""

        if not isinstance(state, Mapping):
            raise TypeError("LegacyAutogradUpdate state must be a mapping")
        self.optimizer.load_state_dict(state)

    def _run_backward(self, update_input: AutogradUpdateInput) -> None:
        if self._backward_scope is None:
            update_input.objective.backward()
            return
        with self._backward_scope(update_input.step):
            update_input.objective.backward()

    def _run_optimizer_step(self, update_input: AutogradUpdateInput) -> None:
        if self._optimizer_scope is None:
            self.optimizer.step()
            return
        with self._optimizer_scope(update_input.step):
            self.optimizer.step()


def _validate_step(step: int) -> None:
    if type(step) is not int or step < 0:
        raise ValueError("VMC update step must be a non-negative integer")


def _parameter_layout(
    parameters: tuple[torch.nn.Parameter, ...],
) -> ParameterLayout:
    """Derive static layout metadata from direct live parameters."""

    return ParameterLayout(
        slots=tuple(
            ParameterSlot(
                ordinal=ordinal,
                shape=tuple(parameter.shape),
                numel=parameter.numel(),
                dtype=parameter.dtype,
            )
            for ordinal, parameter in enumerate(parameters)
        )
    )


def serialize_parameter_layout(layout: ParameterLayout) -> dict[str, Any]:
    """Return JSON-safe immutable metadata for a parameter layout."""

    if not isinstance(layout, ParameterLayout):
        raise TypeError("parameter layout must be a ParameterLayout")
    layout.validate()
    return {
        "slots": [
            {
                "ordinal": slot.ordinal,
                "shape": list(slot.shape),
                "numel": slot.numel,
                "dtype": str(slot.dtype),
            }
            for slot in layout.slots
        ]
    }


def deserialize_parameter_layout(state: Mapping[str, Any]) -> ParameterLayout:
    """Parse strict JSON-safe parameter-layout metadata."""

    if not isinstance(state, Mapping):
        raise TypeError("parameter layout state must be a mapping")
    slots = state.get("slots")
    if not isinstance(slots, list):
        raise ValueError("parameter layout state must contain a slots list")
    parsed: list[ParameterSlot] = []
    for raw in slots:
        if not isinstance(raw, Mapping):
            raise TypeError("parameter layout slots must be mappings")
        dtype_name = raw.get("dtype")
        if not isinstance(dtype_name, str) or not dtype_name.startswith("torch."):
            raise ValueError("parameter layout slot dtype must be a torch dtype name")
        dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("parameter layout slot dtype must be a real floating torch.dtype")
        shape = raw.get("shape")
        if not isinstance(shape, list):
            raise TypeError("parameter layout slot shape must be a list")
        parsed.append(
            ParameterSlot(
                ordinal=raw.get("ordinal"),
                shape=tuple(shape),
                numel=raw.get("numel"),
                dtype=dtype,
            )
        )
    return ParameterLayout(slots=tuple(parsed))


def _batch_requires_grad(batch: ElectronBatch) -> bool:
    tensors = (batch.positions, batch.nuclear_positions, batch.nuclear_charges, batch.spins)
    return any(tensor is not None and tensor.requires_grad for tensor in tensors)


def _output_requires_grad(output: WavefunctionOutput) -> bool:
    tensors = (output.logabs, output.sign, output.phase, *output.aux.values())
    return any(isinstance(tensor, torch.Tensor) and tensor.requires_grad for tensor in tensors)


def _gradient_norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().pow(2).sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).item()) if total is not None else 0.0


__all__ = [
    "AutogradUpdateInput",
    "LegacyAutogradUpdate",
    "ModelParameterBinding",
    "ObjectiveReevaluation",
    "ScoreUpdateInput",
    "VMCStepData",
    "VMCUpdateMethod",
    "VMCUpdateResult",
    "VMCUpdateState",
    "deserialize_parameter_layout",
    "select_reevaluation_rows",
    "serialize_parameter_layout",
    "vmc_objective_reevaluation",
]
