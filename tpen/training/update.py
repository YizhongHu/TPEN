"""Typed VMC update inputs and the behavior-preserving legacy adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
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
    Nothing on the recompute path draws from an RNG today.  The precondition
    is stated as the property's actual dependency rather than as a directory
    being empty, because the recompute path spans two trees:

    (i) NO STOCHASTIC LAYER IN ANY FORWARD reached by ``model(batch)``.  The
        model is a ``tpen/nn/`` object, so a claim scoped to ``tpen/physics/``
        could not see a stochastic layer added here at all.  Measured: the
        only RNG in ``tpen/nn/`` is in ``initialization.py``, resolved by AST
        to ``uniform_``, ``xavier_uniform_``, and ``linear_kaiming_uniform_``
        -- all initialization, none a forward -- and there is no ``Dropout``
        anywhere in the package.
    (ii) NO RNG DRAW ON THE LOCAL-ENERGY PATH
        (:func:`~tpen.physics.hamiltonian.local_energy`,
        :func:`~tpen.training.vmc.compute_vmc_objective`).  Measured: zero
        draws in ``tpen/physics/``.

    INITIALIZATION-TIME RNG IS OUT OF SCOPE, deliberately: it runs before the
    step, so no re-evaluation can reach it.  That distinction is why (i) is
    worded as "in a forward" and not "in the package".

    A NOTE ON HOW THIS WAS GOT WRONG ONCE, because the failure is reusable.  An
    earlier version of this precondition claimed ``tpen/physics/`` "contains no
    generator".  That tree contains the word ``generator`` ten times, in
    ``validate_for_generator`` methods, where generator means a SAMPLER and not
    an RNG -- so the literal claim was false while the property was true.
    Stating a precondition as "directory contains no X" invites exactly that
    error when X is an overloaded word.  Read the hits you dismissed before
    publishing an absence, and resolve matches to their enclosing function
    before classifying them.

    The named tests are the enforcement.  TRIGGER: the day a forward acquires a
    stochastic layer, or the local-energy path acquires a draw -- a stochastic
    kinetic estimator being the obvious candidate for the latter -- the
    same-function test is the one that must go red.

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

    THAT QUESTION IS NOW ANSWERED.  Operator ruling ``377f6b0f`` settles it:
    the finite-row mask is FIXED PER STEP, pinned at the step's first
    evaluation, and a pinned row going non-finite at a later trial point
    RAISES rather than being re-masked.  Silent re-mask is rejected.  So the
    subsample no longer shifts between closure calls, and a line search
    compares one function for the whole step.
    :func:`select_reevaluation_rows` is where that lands, which is what the
    seam was shaped for.  See the ``operator-ruling`` note on ``377f6b0f`` for
    the ruling itself rather than any paraphrase.  Under ``"fail"`` the
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
    ``377f6b0f`` would be a small change here rather than a rederivation of the
    seam -- and that is how it played out: the ruling landed as eight lines in
    this body and nothing else.

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
    torch.Tensor
        Boolean mask restricting this re-evaluation to exactly the rows that
        were finite when the step's mask was pinned.  ``None`` is no longer
        returned; the signature keeps the optional type so a future ruling can
        reintroduce a no-selection mode without a signature change.

    Raises
    ------
    ValueError
        If a row that was finite when the mask was pinned is non-finite at this
        trial point.  The message names the offending row indices.

    Notes
    -----
    NO OPT-OUT IS SHIPPED, AND THAT IS THE RULING HONOURED RATHER THAN
    NARROWED.  ``377f6b0f`` permits ``propagate-non-finite`` only where an
    optimizer's admission carries a test proving its search rejects non-finite
    trial values and terminates.  There is no channel from an optimizer or an
    update method to this function -- ``policy`` arrives from the TRAINER --
    so building one now could only end in a boolean any optimizer could set,
    which is the flag the ruling forbids.  With no channel, "only where
    proven" means NO CHANNEL UNTIL SOMETHING PROVES IT, which makes the
    obligation structural rather than documented.  When SPRING or the linear
    method needs it, it builds the channel TOGETHER WITH its proof test, and
    the thing that travels it should be a typed capability carrying the node
    id of that test -- never a boolean.

    CONSEQUENCE, stated so it is not discovered at first use: this is a
    behaviour change under the DEFAULT ``"mask"`` policy, not only under
    ``"fail"``.  A mid-search non-finite row used to be dropped silently and
    the step completed; it now refuses.  Every closure optimizer feels it,
    including stock ``torch.optim.LBFGS``, which has no escape hatch until an
    admission builds one.  That is intended: LBFGS does not robustly reject
    non-finite trial values, and a refusal is recoverable where a silently
    shifted objective is not.

    A SIDE EFFECT WORTH KNOWING: ``policy`` is now inert at compute time in
    the ordinary case.  The selection contains exactly the rows finite at
    pinning, and any of those going bad raises, so the tensors reaching
    `compute_vmc_objective` are always all-finite and neither policy branch
    can fire.  It is still threaded through and still asserted to arrive
    unchanged, because a future ruling could make it load-bearing again and a
    recompute quietly substituting the module default would then be running
    one step under two estimators.

    The ruling itself lives in the ``operator-ruling`` note on ``377f6b0f``,
    cross-linked from ``02859027``.  Read it there rather than any paraphrase;
    a third copy is a third thing that can decay.
    """

    del policy
    # PART 1 of ruling 377f6b0f: a FIXED ROW MASK PER STEP, pinned at the step's
    # FIRST evaluation. `primary_finite_mask` already IS that first evaluation,
    # so pinning needs no new input.
    #
    # PART 2: a row that was finite when the mask was pinned and is non-finite
    # at a later trial point is a genuine failure, not a row to drop quietly.
    # Re-masking here would change the estimator mid-step, so a line search
    # would compare different functions and return a plausible number -- the
    # expensive kind of error. A refusal is recoverable; a silently shifted
    # objective is not. SILENT RE-MASK IS REJECTED by the ruling, and this
    # raise is what makes it unreachable rather than merely discouraged.
    pinned_gone_bad = primary_finite_mask & ~torch.isfinite(recomputed_local_energy)
    if bool(pinned_gone_bad.any()):
        rows = pinned_gone_bad.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            "re-evaluation produced a non-finite local energy at row(s) "
            f"{rows}, which were finite when this step's mask was pinned. "
            "The step is refused rather than re-masked: dropping them now "
            "would change the estimator mid-search, so a line search would "
            "compare different functions"
        )
    return primary_finite_mask


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
    # No ``.detach().clone()``: ``torch.isfinite`` returns a FRESH bool tensor
    # (measured ``_base is None``), and a bool tensor can never require grad --
    # even from a grad-requiring input.  Both calls were provable no-ops.  The
    # property they appeared to provide, that this mask is fixed at
    # construction and immune to later in-place mutation of the caller's
    # tensor, is now asserted by a test that can actually fail instead.
    primary_finite_mask = torch.isfinite(primary_local_energy)

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
        #
        # THESE TWO GUARD NON-FACTORY CONSTRUCTION, which is a real route and
        # not a hypothetical: `vmc_objective_reevaluation` derives dtype and
        # device from the batch, and `VMCStepData.validate` already forces
        # batch/logabs/objective agreement, so through the factory they cannot
        # fire.  `ObjectiveReevaluation` is directly constructible, and the
        # test suite constructs it that way (see `_reevaluation` in
        # tests/unit/training/test_vmc_update.py, which never touches the
        # factory).  Anyone re-deriving the reachability question through the
        # factory alone will conclude these are dead and delete real
        # protection -- stated here so that conclusion is not reached twice.
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


# Reason vocabulary shared by the built-in methods. These are DISPLAY and
# RECORD values, never behavior selectors: nothing in the trainer or in any
# method branches on a reason string. SR and SPRING already spell the vacuum
# skip ``zero_electron_batch``; the legacy adapter reuses that exact spelling
# rather than inventing a second name for one condition.
UPDATE_REASON_APPLIED = "applied"
UPDATE_REASON_ZERO_ELECTRON_BATCH = "zero_electron_batch"
# What a result reports when a method returned no reason at all. A custom
# method predating the reason contract is the case this exists for. It is
# deliberately a distinct token, NOT "applied" and NOT an empty string, so a
# record never claims a reason its method did not give.
UPDATE_REASON_UNREPORTED = "unreported"


def json_safe_scalar(value: Any) -> Any:
    """Return a value a JSON and CSV metrics sink can both carry.

    WHY THIS EXISTS, as a mechanism rather than a convention. The two sinks
    this project writes impose two different hard limits, and a naive float
    violates both:

    - ``tpen.logging.jsonl`` calls ``json.dumps(..., allow_nan=False)``. A NaN
      or infinity therefore RAISES in the sink, turning a diagnostic nobody
      asked to be fatal into a failed run.
    - ``tpen.logging.csv`` writes ``f"{step},{namespace},{key},{value}"`` with
      no quoting. A value whose text contains a comma silently becomes two
      columns, which is worse than an exception because the file still parses.

    The representation here is explicit in both directions: a finite real
    number passes through as a ``float``; anything else becomes a SHORT
    COMMA-FREE STRING naming what was actually observed. A reader can tell the
    two apart by type alone.

    NEVER FABRICATE ZERO. Returning ``0.0`` for an unavailable or non-finite
    observation is the specific failure this helper exists to prevent: zero is
    a legitimate value for every norm reported here, so a fabricated zero is
    indistinguishable from a real measurement and silently corrupts any
    downstream average.

    Parameters
    ----------
    value : Any
        Observation to represent. ``None`` means the method had nothing to
        report for this key.

    Returns
    -------
    float or int or bool or str or None
        ``None`` passes through (JSON ``null``, empty CSV cell). ``bool`` and
        ``int`` pass through. A finite ``float`` passes through. A non-finite
        float becomes ``"nan"``, ``"inf"``, or ``"-inf"``. Any other object
        becomes ``repr``-free ``str`` text with commas replaced by ``;``.
    """

    if value is None or type(value) is bool or type(value) is int:
        return value
    if isinstance(value, torch.Tensor):
        # A TENSOR-VALUED SETTING IS REAL, not hypothetical: PyTorch's
        # capturable and fused optimizer paths put a tensor in `lr`. Falling
        # through to `str()` would yield "tensor(0.0010)" -- DISPLAY TEXT,
        # already rounded -- so the one setting a reader most wants to trust
        # would be silently truncated. Take the exact Python scalar instead,
        # and recurse so the finiteness handling below still applies.
        if value.numel() == 1:
            return json_safe_scalar(value.item())
        # A multi-element tensor is not a scalar setting. Name it rather than
        # serializing it: its text would carry commas and could be unbounded.
        return f"tensor_numel{value.numel()}"
    if isinstance(value, float) or isinstance(value, int):
        numeric = float(value)
        if numeric != numeric:
            return "nan"
        if numeric == float("inf"):
            return "inf"
        if numeric == float("-inf"):
            return "-inf"
        return numeric
    text = str(value)
    # UNQUOTED CSV ROWS IMPOSE THREE SEPARATE CONSTRAINTS. The sink writes
    # `f"{step},{ns},{key},{value}\n"` with no quoting and no escaping:
    #   - a COMMA creates a phantom column, and the file still parses;
    #   - a NEWLINE or CARRIAGE RETURN TERMINATES THE ROW, turning the
    #     remainder of the value into a bogus record;
    #   - a DOUBLE QUOTE is the csv module's quotechar, so a value containing
    #     one makes a STRICT READER fail with "unexpected end of data" when it
    #     looks for a closing quote that never comes. The file is then not
    #     merely wrong, it is unreadable from that row on.
    # This guard was built up one reported symptom at a time -- first the
    # comma, then the newline, then the quote -- which is exactly how a guard
    # ends up covering less than its name promises. The test for it reads a
    # row back with a STRICT csv reader rather than splitting on commas, so a
    # future hostile character fails the reader instead of passing a
    # hand-rolled check.
    for hostile, replacement in ((",", ";"), ("\r", " "), ("\n", " "), ('"', "'")):
        text = text.replace(hostile, replacement)
    return text


class UpdateDiagnostics(ABC):
    """Nominal DETACHED record of what one update attempt actually did.

    Nominal rather than structural on purpose. An update method cannot satisfy
    this by happening to own an attribute of the right name; it has to declare
    the record as this contract, which is what lets
    :class:`VMCUpdateResult` state the type it carries instead of accepting
    any object with an ``as_metrics``.

    DETACHED is the load-bearing word. Implementations hold plain Python
    scalars and already-detached sub-records only. No tensor that participates
    in a graph, and no model or optimizer reference, belongs in a diagnostic:
    the record outlives the step that produced it and is written to artifacts,
    so a graph-bearing field would both retain the step's memory and fail at
    the serialization boundary.
    """

    @abstractmethod
    def as_metrics(self) -> dict[str, Any]:
        """Return this record as flat, JSON-safe, comma-free metric entries.

        Flat because the CSV sink writes one row per key and cannot express
        nesting; JSON-safe because the JSONL sink rejects non-finite floats.
        Implementations compose their OWN key names, so the trainer never
        re-spells a method-owned key.
        """


@dataclass(frozen=True, kw_only=True)
class MinimalUpdateDiagnostics(UpdateDiagnostics):
    """The smallest honest diagnostic: what every method can always report.

    This is the compatibility path the update contract promises to methods
    that own no solver telemetry -- the legacy autograd adapter, and any
    custom or stateless method outside this package. It reports only what is
    knowable without a solver, and reports absence as absence.

    Using this record is an explicit choice by a method, not a fallback the
    trainer applies. That distinction matters: a trainer-side fallback would
    be attribute probing again, which is exactly what this slice removes.
    """

    method: str
    applied: bool
    reason: str
    step: int
    grad_norm: float | None = None
    # NOT "update": the trainer emits a generic `update_reason` for every
    # attempt, so a record prefixed "update" would collide with it and one key
    # would silently overwrite the other in the metrics dict. The collision was
    # real and invisible -- both happened to carry the same value -- which is
    # exactly why the prefix is pinned here and asserted by a test rather than
    # left to chance.
    prefix: str = "update_record"

    def as_metrics(self) -> dict[str, Any]:
        """Return the minimal record under this method's own key prefix."""

        return {
            f"{self.prefix}_method": json_safe_scalar(self.method),
            f"{self.prefix}_applied": bool(self.applied),
            f"{self.prefix}_reason": json_safe_scalar(self.reason),
            f"{self.prefix}_step": int(self.step),
            f"{self.prefix}_grad_norm": json_safe_scalar(self.grad_norm),
        }


@dataclass(frozen=True, kw_only=True)
class VMCUpdateResult:
    """Result of one update-method invocation.

    Parameters
    ----------
    applied : bool
        Whether the attempt actually stepped the parameters.
    grad_norm : float
        The norm this method reports for the step. ITS MEANING IS METHOD-OWNED
        and is not uniform across methods: for the legacy adapter it is the
        post-clip Euclidean norm of ``.grad`` over the gradient domain, while
        for SR, SPRING, and block NG it is the ENERGY-GRADIENT norm, which is
        not the norm of the preconditioned direction those methods write into
        ``.grad``. Read the method's
        :meth:`VMCUpdateMethod.describe` record for the actual meaning rather
        than assuming one; this is precisely the ambiguity the description
        contract exists to resolve.
    reason : str or None, optional
        Explicit, method-owned reason for the outcome. ``None`` means the
        method reported none -- the shape a custom method written before this
        field existed has. Use :attr:`reported_reason` to read it without
        having to re-handle ``None`` at every call site.
    diagnostics : UpdateDiagnostics or None, optional
        The detached record for THIS attempt. ``None`` means the method
        reported none.

    Notes
    -----
    BOTH NEW FIELDS DEFAULT. That is a compatibility requirement, not an
    oversight: existing custom ``VMCUpdateMethod`` implementations construct
    this record positionally by keyword with two fields, and making either new
    field required would break every one of them at once. A method that
    reports nothing is a supported shape; a method that reports a FABRICATED
    reason would not be.

    THE RESULT IS THE SINGLE AUTHORITY for one attempt. The built-in methods
    retain a ``last_telemetry`` property for compatibility, but it is derived
    from the result they returned and is not a second place where an attempt
    is recorded.
    """

    applied: bool
    grad_norm: float
    reason: str | None = None
    diagnostics: UpdateDiagnostics | None = None

    def __post_init__(self) -> None:
        if type(self.applied) is not bool:
            raise TypeError("VMCUpdateResult.applied must be a bool")
        object.__setattr__(self, "grad_norm", float(self.grad_norm))
        if self.reason is not None:
            if not isinstance(self.reason, str) or not self.reason:
                raise TypeError("VMCUpdateResult.reason must be a non-empty str or None")
        if self.diagnostics is not None and not isinstance(self.diagnostics, UpdateDiagnostics):
            raise TypeError("VMCUpdateResult.diagnostics must be an UpdateDiagnostics or None")

    @property
    def reported_reason(self) -> str:
        """Return a SINK-SAFE reason, or the explicit unreported token.

        SANITIZED HERE RATHER THAN REJECTED AT CONSTRUCTION, and that reversal
        is deliberate. An earlier version RAISED on a reason containing a
        comma. That made the observation layer able to abort the run it was
        only supposed to describe -- a method returning an awkward string
        crashed training rather than producing a slightly ugly log line. It is
        the same mistake, in the opposite direction, as letting a non-finite
        float reach a sink that raises on it: in both cases the record written
        to explain something destroyed the thing it was explaining.

        The stored :attr:`reason` is left VERBATIM, because it is the method's
        own datum and a caller reading the result object should see exactly
        what the method said. Only the EMITTED form is sanitized, and only at
        this boundary.

        Custom methods are the realistic source: a reason assembled from an
        exception message can carry a newline, and the unquoted CSV sink
        terminates its row on one.
        """

        if self.reason is None:
            return UPDATE_REASON_UNREPORTED
        return json_safe_scalar(self.reason)


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


def carrier_settings(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Return the carrier optimizer's EFFECTIVE settings as flat scalars.

    "Effective" means read off the live ``param_groups`` at call time, not
    recalled from the config that built the optimizer. After a checkpoint
    restore those two can differ, and the restored values are the ones the
    next step will actually use.

    COMPOUND SETTINGS ARE SPLIT, NOT REPR'D. Adam's ``betas`` is the motivating
    case: writing ``(0.9, 0.999)`` into the unquoted CSV sink produces a row
    with two extra columns that still parses. Each element becomes its own
    named scalar key instead.

    KEYS ARE SANITIZED, NOT ONLY VALUES. A param_group is an ordinary dict and
    its keys become METRIC NAMES, so a key carrying a comma or a newline
    corrupts the row exactly as a hostile value would. An earlier version
    guarded only values, which is half a channel.

    EVERY GROUP IS PREFIXED, INCLUDING THE FIRST. Leaving group 0 unprefixed
    read as tidier and was wrong: a group-0 key literally named ``group1_lr``
    then produced the same name as group 1's real learning rate, and a group-0
    key named ``n_param_groups`` overwrote the count this function itself
    seeds. Uniform ``g<i>_`` prefixing puts every group-derived name in a space
    no unprefixed name can reach.

    COLLISIONS ARE REPORTED, NOT RESOLVED SILENTLY. Prefixing removes the
    collisions structure can remove, but it CANNOT remove all of them: keys are
    arbitrary strings, so a group-0 key named ``betas1`` still lands on the
    name produced by splitting a ``betas`` pair. Rather than letting the later
    write win -- which is how a description comes to report a value nobody set
    -- the FIRST value is kept and every collided name is listed under
    ``setting_name_collisions``. A reader then sees that the map is incomplete
    instead of trusting a silently overwritten entry.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The live carrier to describe.

    Returns
    -------
    dict
        Flat JSON-safe entries. ``n_param_groups`` is always present. Every
        group's settings appear under ``g<i>_<key>``, so a per-layer learning
        rate is visible rather than hidden behind group 0.
    """

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("carrier_settings requires a torch.optim.Optimizer")
    groups = list(optimizer.param_groups)
    settings: dict[str, Any] = {}
    collisions: list[str] = []

    def put(name: Any, value: Any) -> None:
        """Record one entry, keeping the FIRST writer and noting any clash."""

        safe_name = str(json_safe_scalar(name))
        if safe_name in settings:
            collisions.append(safe_name)
            return
        settings[safe_name] = json_safe_scalar(value)

    # Seeded first so that no group key can displace it.
    put("n_param_groups", len(groups))
    for index, group in enumerate(groups):
        prefix = f"g{index}_"
        # Sorted by the key's TEXT: a param_group may hold non-comparable key
        # types, and sorting those directly raises.
        for key, value in sorted(group.items(), key=lambda item: str(item[0])):
            # `params` is the live parameter list -- unbounded, and it holds
            # the model's tensors. It must never enter a description.
            if key == "params":
                continue
            if isinstance(value, (tuple, list)):
                for ordinal, element in enumerate(value, start=1):
                    put(f"{prefix}{key}{ordinal}", element)
                continue
            put(f"{prefix}{key}", value)
    if collisions:
        put("setting_name_collisions", ";".join(sorted(set(collisions))))
    return settings


def flatten_settings(settings: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested policy fingerprint into flat JSON-safe scalar entries.

    A policy fingerprint is allowed to nest -- ``SRPolicy.fingerprint`` carries
    a whole ``damping`` sub-mapping -- but neither metrics sink can express
    nesting: the CSV sink writes one row per key, and a dict written into it
    would be a brace-and-comma string that silently becomes several columns.
    Nested keys are therefore joined with ``_`` into one flat namespace.

    Parameters
    ----------
    settings : Mapping
        Possibly nested settings mapping.
    prefix : str, optional
        Key prefix for recursion; callers normally leave it empty.

    Returns
    -------
    dict
        Flat mapping whose values are all :func:`json_safe_scalar` outputs.
    """

    flat: dict[str, Any] = {}
    for key, value in settings.items():
        # Sanitized here too: these names reach the sinks as metric names.
        composed = f"{prefix}{json_safe_scalar(key)}"
        if isinstance(value, Mapping):
            flat.update(flatten_settings(value, prefix=f"{composed}_"))
        elif isinstance(value, (tuple, list)):
            for ordinal, element in enumerate(value, start=1):
                flat[f"{composed}{ordinal}"] = json_safe_scalar(element)
        else:
            flat[composed] = json_safe_scalar(value)
    return flat


def parameter_layout_fingerprint(layout: ParameterLayout) -> str:
    """Return a stable, sink-safe identity for a parameter LAYOUT.

    WHY COUNTS ARE NOT ENOUGH. The binding design requires the description to
    carry "parameter count/layout identity", and a count alone does not supply
    it: one parameter shaped ``(4,)`` and one shaped ``(2, 2)`` have identical
    scalar and tensor counts, so two genuinely different models produced
    identical descriptions. That defeats the purpose of describing what was
    actually constructed.

    STABLE ACROSS VALUES AND ADDRESSES: built only from shape and dtype, never
    from parameter values or object identity, so the same architecture
    fingerprints identically across runs, processes and restores -- which is
    what makes it usable for comparing a resumed run against its original.

    Comma-free and newline-free by construction: dimensions are joined with
    ``x`` and slots with ``|``, so the result passes through the unquoted CSV
    sink without needing sanitization.
    """

    slots = []
    for slot in layout.slots:
        dtype = str(slot.dtype).replace("torch.", "")
        shape = "x".join(str(dim) for dim in slot.shape) or "scalar"
        slots.append(f"{dtype}:{shape}")
    return "|".join(slots)


@dataclass(frozen=True, kw_only=True)
class UpdateMethodDescription:
    """What was ACTUALLY constructed or restored, for one fit invocation.

    THE PROBLEM THIS SOLVES. SR, SPRING, and block NG all drive a plain
    ``torch.optim.SGD`` as a carrier: they compute a preconditioned direction,
    write it into ``.grad``, and let SGD apply it. Anything that identifies the
    update by looking at the optimizer therefore reports every one of those
    runs as plain SGD, which is wrong in the most expensive possible way --
    the run looks like a baseline nobody ran. :attr:`method_class` and
    :attr:`carrier_class` are separate fields for exactly this reason.

    CLASS NAMES ARE FOR DISPLAY ONLY. Nothing selects behavior from these
    strings. A config's label for a run is likewise never consulted here: every
    field is read off the live constructed or restored objects, so a run
    mislabelled in YAML still describes itself truthfully.

    Parameters
    ----------
    method_class : str
        Qualified name of the actual :class:`VMCUpdateMethod` instance.
    carrier_class : str
        Qualified name of the actual optimizer applying the step, or
        ``"none"`` for a method that owns no carrier.
    n_parameters : int or None
        Total scalar parameter count in the bound domain. ``None`` when this
        method owns no carrier and therefore no parameter domain -- never 0,
        which is a legitimate count and would be indistinguishable from a
        measurement.
    n_parameter_tensors : int or None
        Number of tensors in the bound domain, or ``None`` on the same terms.
    layout_fingerprint : str or None
        Stable shape-and-dtype identity of the bound domain, or ``None`` when
        there is none. Counts alone cannot distinguish ``(4,)`` from
        ``(2, 2)``; this can.
    forward_request : str
        The requested derivative convention -- the qualified type of the typed
        forward request this method returns, or ``"value"`` for an ordinary
        ``model(batch)`` forward.
    norm_semantics : str
        What this method's reported ``grad_norm`` actually measures.
    settings : Mapping
        Flat scalar effective settings: carrier settings, plus any
        method-owned policy fields.
    """

    method_class: str
    carrier_class: str
    # OPTIONAL, and NOT defaulted to 0. A method owning no carrier has no
    # parameter domain to count, and zero is a legitimate count -- so reporting
    # 0 there would be a fabricated value indistinguishable from a real
    # measurement, which is precisely what `json_safe_scalar` exists to
    # prevent. `None` means "not reported" and stays distinguishable.
    n_parameters: int | None
    n_parameter_tensors: int | None
    # Shape-and-dtype identity, NOT just a count. Two layouts with equal
    # counts but different shapes must be distinguishable; see
    # `parameter_layout_fingerprint`.
    layout_fingerprint: str | None
    forward_request: str
    norm_semantics: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def as_metrics(self, *, prefix: str = "update_method") -> dict[str, Any]:
        """Return the description as flat, JSON-safe, comma-free entries.

        SETTINGS ARE NAMESPACED AWAY FROM THE AUTHORITATIVE FIELDS, and that
        separation is the whole point of this method rather than a tidiness
        preference.

        THE ATTACK IT CLOSES. Settings are read off the LIVE optimizer
        ``param_groups``, and a param_group is an ordinary dict that anything
        may add keys to. When settings were merged into the same ``prefix_``
        namespace as the identity fields, a group carrying a key named
        ``class`` or ``n_parameters`` OVERWROTE the authoritative value. The
        description OBJECT stayed truthful while the emitted METRICS lied --
        reporting, in the observed case, ``update_method_class =
        'misleading-run-label'`` for a real ``LegacyAutogradUpdate``.

        That is precisely the failure this whole description contract exists
        to prevent. The contract's promise is that a misleading external label
        cannot change what is reported; a test asserting that promise passed,
        because it set an ATTRIBUTE on the method, and the reachable route was
        through the carrier's group dict instead. A guard is only as good as
        the channel it watches.

        Two independent mechanisms now hold, so neither alone is load-bearing:
        settings go under ``{prefix}_setting_``, which cannot collide with any
        authoritative name because no authoritative name begins with
        ``setting_``; and the authoritative fields are written LAST, so even a
        hypothetical collision resolves in favour of the truth.
        """

        metrics: dict[str, Any] = {}
        # Settings FIRST, under their own sub-namespace. The KEY is sanitized
        # as well as the value: a setting name becomes a METRIC NAME, so a
        # comma or newline in it corrupts the row exactly as a value would.
        for key, value in self.settings.items():
            safe_key = str(json_safe_scalar(key))
            metrics[f"{prefix}_setting_{safe_key}"] = json_safe_scalar(value)
        # Authoritative identity LAST, so it always wins.
        metrics.update(
            {
                f"{prefix}_class": json_safe_scalar(self.method_class),
                f"{prefix}_carrier_class": json_safe_scalar(self.carrier_class),
                f"{prefix}_n_parameters": json_safe_scalar(self.n_parameters),
                f"{prefix}_n_parameter_tensors": json_safe_scalar(self.n_parameter_tensors),
                f"{prefix}_layout_fingerprint": json_safe_scalar(self.layout_fingerprint),
                f"{prefix}_forward_request": json_safe_scalar(self.forward_request),
                f"{prefix}_norm_semantics": json_safe_scalar(self.norm_semantics),
            }
        )
        return metrics


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

    @property
    def last_result(self) -> VMCUpdateResult | None:
        """Return the result of this method's most recent attempt.

        ``None`` before the first attempt. This is the ONE place an attempt is
        remembered; the built-in methods' ``last_telemetry`` property reads
        the diagnostics out of this record rather than keeping a parallel one.
        """

        return getattr(self, "_last_result", None)

    def _record_result(
        self,
        *,
        applied: bool,
        grad_norm: float,
        reason: str,
        diagnostics: UpdateDiagnostics | None = None,
    ) -> VMCUpdateResult:
        """Build, remember, and return the result for one attempt.

        Remembering is assignment to a single slot, so a fresh attempt always
        REPLACES the previous record rather than merging with it. That is what
        makes "applied, then skipped" report the skip's own diagnostics
        instead of leaking the earlier solve's.
        """

        result = VMCUpdateResult(
            applied=applied,
            grad_norm=grad_norm,
            reason=reason,
            diagnostics=diagnostics,
        )
        self._last_result = result
        return result

    def describe(self) -> UpdateMethodDescription:
        """Describe the object that will actually perform the next update.

        The default reads everything off ``self`` and off the owned
        :class:`VMCUpdateState`, so a method inherits a truthful description
        without writing one. A method with policy worth recording overrides
        this, calls ``super().describe()``, and returns a copy with its policy
        merged into ``settings`` and its own ``norm_semantics``.

        CALL THIS AFTER RESTORE. Settings are read from the live carrier, so
        calling it before ``load_state_dict`` would describe the pre-restore
        object and the description would disagree with the run.

        Returns
        -------
        UpdateMethodDescription
            Flat, JSON-safe description of the live method and carrier.
        """

        update_state = self.update_state()
        if update_state is None:
            # A stateless method delegates the carrier to the trainer, so
            # there is no owned optimizer to read. Report that honestly
            # rather than naming an optimizer this method does not own.
            carrier_class = "none"
            settings: dict[str, Any] = {}
            # NOT 0: this method owns no parameter domain, so the count is
            # UNKNOWN rather than empty. See the field comment above.
            n_parameters = None
            n_parameter_tensors = None
            layout_fingerprint = None
        else:
            optimizer = update_state.optimizer
            carrier_class = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
            settings = carrier_settings(optimizer)
            layout = update_state.model_parameters.layout
            assert layout is not None
            n_parameters = int(layout.total_numel)
            n_parameter_tensors = len(layout.slots)
            layout_fingerprint = parameter_layout_fingerprint(layout)
        request = self.forward_request()
        forward_request = (
            "value"
            if request is None
            else f"{type(request).__module__}.{type(request).__qualname__}"
        )
        return UpdateMethodDescription(
            method_class=f"{type(self).__module__}.{type(self).__qualname__}",
            carrier_class=carrier_class,
            n_parameters=n_parameters,
            n_parameter_tensors=n_parameter_tensors,
            layout_fingerprint=layout_fingerprint,
            forward_request=forward_request,
            norm_semantics=self.norm_semantics(),
            settings=settings,
        )

    def norm_semantics(self) -> str:
        """Return what this method's reported ``grad_norm`` actually measures.

        The default is deliberately non-committal: a method that has not said
        what its norm means must not have a specific meaning invented for it.
        """

        return "method_defined"

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
                # THE VACUUM SKIP. A zero-electron batch has no sampled
                # coordinate degree of freedom, so there is nothing to
                # differentiate and declining is correct. The grad_norm of 0.0
                # here is a REAL measurement -- no backward ran, so the
                # gradient domain genuinely holds no gradient -- and it is an
                # established compatibility field, so it keeps its value and
                # its meaning. The reason field is what newly distinguishes
                # this from the raise below; the two conditions were always
                # distinct in behavior and are now distinct in the record.
                return self._result(
                    applied=False,
                    grad_norm=0.0,
                    reason=UPDATE_REASON_ZERO_ELECTRON_BATCH,
                    step=update_input.step,
                )
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
        return self._result(
            applied=True,
            grad_norm=grad_norm,
            reason=UPDATE_REASON_APPLIED,
            step=update_input.step,
        )

    def _result(
        self,
        *,
        applied: bool,
        grad_norm: float,
        reason: str,
        step: int,
    ) -> VMCUpdateResult:
        """Build this attempt's result with its explicit minimal record.

        Routed through :meth:`VMCUpdateMethod._record_result` rather than
        constructing the record directly, so ``last_result`` reports this
        adapter's attempts like any other method's. Building it here instead
        would leave ``last_result`` permanently ``None`` on the one method
        every default configuration uses.
        """

        return self._record_result(
            applied=applied,
            grad_norm=grad_norm,
            reason=reason,
            diagnostics=MinimalUpdateDiagnostics(
                method=type(self).__qualname__,
                applied=applied,
                reason=reason,
                step=step,
                grad_norm=grad_norm,
            ),
        )

    def norm_semantics(self) -> str:
        """Report the legacy adapter's norm as the POST-CLIP gradient norm.

        Stated precisely because the ordering is observable: the norm is taken
        after ``clip_grad_norm_`` and before ``optimizer.step()``, so with
        clipping configured it is bounded by ``gradient_clip_norm`` and is NOT
        the norm of the raw backward gradient.
        """

        return "post_clip_grad_l2_over_gradient_domain"

    def describe(self) -> UpdateMethodDescription:
        """Add the adapter's clipping setting to the generic description."""

        base = super().describe()
        settings = dict(base.settings)
        settings["gradient_clip_norm"] = json_safe_scalar(self.gradient_clip_norm)
        return replace(base, settings=settings)

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
    "MinimalUpdateDiagnostics",
    "ModelParameterBinding",
    "ObjectiveReevaluation",
    "ScoreUpdateInput",
    "UPDATE_REASON_APPLIED",
    "UPDATE_REASON_UNREPORTED",
    "UPDATE_REASON_ZERO_ELECTRON_BATCH",
    "UpdateDiagnostics",
    "UpdateMethodDescription",
    "VMCStepData",
    "VMCUpdateMethod",
    "VMCUpdateResult",
    "VMCUpdateState",
    "carrier_settings",
    "deserialize_parameter_layout",
    "flatten_settings",
    "json_safe_scalar",
    "parameter_layout_fingerprint",
    "select_reevaluation_rows",
    "serialize_parameter_layout",
    "vmc_objective_reevaluation",
]
