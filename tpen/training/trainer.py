"""Minimal event-driven VMC trainer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.dependencies import require_torch
from tpen.physics.hamiltonian import LocalEnergyResult, local_energy
from tpen.training.events import (
    Backward,
    BuildBatch,
    CollectSamples,
    Forward,
    LocalEnergy,
    Metrics,
    Objective,
    OptimizerUpdate,
    TrainingIteration,
    TrainingIterationCompleted,
    UpdateCompleted,
    UpdateSkipped,
)
from tpen.training.state import TrainerState
from tpen.nn.forward import ParameterScoreRequest
from tpen.training.optim import UpdateMethodSpec, make_update_method
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ModelParameterBinding,
    ScoreUpdateInput,
    deserialize_parameter_layout,
    serialize_parameter_layout,
    VMCUpdateMethod,
    VMCUpdateState,
    vmc_objective_reevaluation,
)
from tpen.training.vmc import (
    DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    compute_vmc_objective,
    resolve_nonfinite_local_energy_policy,
    summarize_local_energy_terms,
    summarize_logabs,
)

torch = require_torch(feature="VMC training")


def _parameter_norm(model) -> float:
    """Return the L2 norm of trainable initialized parameters."""

    total = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        value = param.detach().pow(2).sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).item()) if total is not None else 0.0


def _gradient_norm(model) -> float:
    """Return the L2 norm of available gradients."""

    total = None
    for param in model.parameters():
        if param.grad is None:
            continue
        value = param.grad.detach().pow(2).sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).item()) if total is not None else 0.0


class VMCTrainer:
    """Run a fixed number of VMC optimization steps over an event stream.

    The trainer is configuration-only: ``fit`` receives the model, sampler,
    Hamiltonian terms, optimizer, run context, and an ``emit`` callable, and
    drives the sample -> local-energy -> surrogate-loss -> step loop while
    logging metrics and emitting lifecycle events.

    Parameters
    ----------
    max_steps : int
        Number of optimization steps to run.
    log_every_n_steps : int, optional
        Log metrics every ``log_every_n_steps`` steps.
    return_terms : bool, optional
        Whether to request and summarize the per-term local-energy decomposition.
    gradient_clip_norm : float or None, optional
        Maximum gradient norm. When ``None``, gradients are not clipped.

    Notes
    -----
    Progress uses two independent counters. ``next_iteration`` is the durable
    resume cursor and is assigned near the end of the loop body, so it advances
    once per iteration that *ran to completion*: an iteration that raises
    part-way through (say inside ``local_energy`` or ``optimizer.step()``) never
    advances it and is retried from the same cursor on resume. It advances
    whether or not that iteration applied an optimizer update.
    ``completed_updates`` counts optimizer updates that actually returned, so the
    two diverge whenever a completed iteration skips its update (the
    zero-electron vacuum).

    Every key in the ``train`` record logged at loop step ``k`` describes the
    *pre-update* model -- the one that produced step ``k``'s samples. That
    includes ``param_norm``, which is therefore read before ``optimizer.step()``
    rather than after it. Keeping the record single-version is a contract: a new
    trainer-owned metric must be computed before the update, or it does not
    belong in this record.
    """

    def __init__(
        self,
        max_steps: int,
        log_every_n_steps: int = 1,
        return_terms: bool = False,
        gradient_clip_norm: float | None = None,
        update_method: UpdateMethodSpec = None,
        nonfinite_local_energy_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    ) -> None:
        self.max_steps = int(max_steps)
        self.log_every_n_steps = int(log_every_n_steps)
        self.return_terms = bool(return_terms)
        self.gradient_clip_norm = None if gradient_clip_norm is None else float(gradient_clip_norm)
        self.update_method = update_method
        # Validated at construction rather than at the first non-finite row, so
        # a misspelled policy fails before a run starts instead of after it has
        # produced hours of samples under an estimator nobody chose.
        self.nonfinite_local_energy_policy = resolve_nonfinite_local_energy_policy(
            nonfinite_local_energy_policy
        )
        # These are populated at the invocation boundary.  Keeping the
        # resolved method and model here gives checkpoint restore one owner for
        # rebuilding direct parameter references after model weights load.
        self._resolved_model = None
        self._resolved_update_method: VMCUpdateMethod[AutogradUpdateInput] | None = None
        # The spec that produced `_resolved_update_method`, kept so a repeated
        # selection from the same spec reuses one instance. See F5 above.
        self._update_method_spec: UpdateMethodSpec = None
        self._resolved_update_state: VMCUpdateState | None = None
        self._checkpoint_parameter_layout = None
        # Durable resume cursor: the next iteration this trainer will attempt.
        self.next_iteration = 0
        # Optimizer updates that actually returned; skipped updates never count.
        self.completed_updates = 0

    def state_dict(self) -> dict[str, Any]:
        """Return checkpointable trainer progress state.

        Returns
        -------
        dict
            ``next_iteration`` (durable resume cursor) and
            ``completed_updates`` (applied optimizer updates). The two diverge
            whenever a completed iteration skipped its optimizer update.
        """

        state: dict[str, Any] = {
            "next_iteration": int(self.next_iteration),
            "completed_updates": int(self.completed_updates),
            # The ACTIVE estimator, recorded beside the numbers it produced.
            # Config records intent; this records what actually ran, so a
            # reader holding an energy can tell whether it came from the full
            # sample or from a masked -- and therefore biased -- subsample.
            # One authoritative location, checked on resume; deliberately NOT
            # duplicated into run metadata, because two locations for one fact
            # is how they drift apart.
            "nonfinite_local_energy_policy": self.nonfinite_local_energy_policy,
        }
        if self._resolved_update_state is not None:
            state["parameter_layout"] = serialize_parameter_layout(
                self._resolved_update_state.model_parameters.layout
            )
        # Method state is a first-class payload, not an optimizer detail. A
        # method that owns a schedule counter, a convention fingerprint, or a
        # warm-start vector must be able to round-trip it; the default
        # `VMCUpdateMethod.state_dict()` is empty, so a stateless method adds
        # no key and existing checkpoints are unchanged.
        if self._resolved_update_method is not None:
            method_state = self._resolved_update_method.method_state_dict()
            if method_state:
                state["update_method"] = dict(method_state)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore trainer progress state for ``train_resume``.

        Parameters
        ----------
        state : Mapping
            Trainer state read from a checkpoint's ``trainer.json``. Both
            ``next_iteration`` and ``completed_updates`` are required.

        Raises
        ------
        KeyError
            If either required key is missing. Checkpoints written before the
            rename (``global_step``/``completed_steps``) are unsupported and
            fail here rather than silently resuming from step 0.
        ValueError or TypeError
            If either value cannot be coerced to ``int``. Both coercions run
            before either assignment, so a rejected state leaves the trainer's
            progress counters unmutated.
        """

        for key in ("next_iteration", "completed_updates"):
            if key not in state:
                raise KeyError(
                    f"trainer state is missing required key {key!r}; checkpoints "
                    "written with 'global_step'/'completed_steps' are unsupported"
                )
        # Coerce both counters before assigning either one, so a malformed value
        # cannot leave the trainer half-restored at a bogus resume cursor.
        next_iteration = int(state["next_iteration"])
        completed_updates = int(state["completed_updates"])

        # ESTIMATOR CONTINUITY. A checkpoint written under one non-finite policy
        # and resumed under another would produce, in a single run, numbers from
        # two different estimators -- one over the full sample and one over a
        # systematically selected subsample. Recording the policy without
        # checking it would leave a fact that nothing verifies, and a silent
        # override is exactly how a run ends up reporting under an estimator
        # nobody chose.
        #
        # Follows the precedent set by the update-method restore below, which
        # raises in BOTH directions rather than skipping quietly. Absence is
        # treated as the historical "mask" because every checkpoint written
        # before this key existed used it -- so a legacy checkpoint resumed
        # under "fail" is a real estimator change and is reported as one, rather
        # than being waved through because the old file happened to be silent.
        checkpoint_policy = state.get(
            "nonfinite_local_energy_policy", DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY
        )
        if checkpoint_policy != self.nonfinite_local_energy_policy:
            raise ValueError(
                "non-finite local-energy policy disagrees with the checkpoint: "
                f"checkpoint recorded {checkpoint_policy!r}"
                + (
                    " (absent, so the historical 'mask' is assumed)"
                    if "nonfinite_local_energy_policy" not in state
                    else ""
                )
                + f", this run is configured for {self.nonfinite_local_energy_policy!r}. "
                "Resuming would continue one run under two different estimators. "
                "Match the configuration to the checkpoint, or start a new run"
            )

        restored_layout = None
        if "parameter_layout" in state:
            restored_layout = deserialize_parameter_layout(state["parameter_layout"])
            current_model = self._resolved_model
            if current_model is None:
                raise ValueError(
                    "parameter layout restore requires update-state resolution before restore"
                )

        # Restore method state before the layout rebind below, so a method
        # whose fingerprint disagrees with this checkpoint fails while the
        # trainer's own counters are still unmutated.
        if "update_method" in state:
            if self._resolved_update_method is None:
                raise ValueError(
                    "update-method state restore requires update-state resolution "
                    "before restore"
                )
            self._resolved_update_method.load_method_state_dict(state["update_method"])
        elif (
            self._resolved_update_method is not None
            and self._resolved_update_method.method_state_dict()
        ):
            # A method that OWNS persistent state is being resumed from a
            # checkpoint that carries none of it. Skipping quietly would restore
            # the trainer's counters while leaving the method's own counters,
            # fingerprint and any warm start at their fresh values -- a resume
            # that looks clean and is not, which is precisely what "strict
            # updater/layout/damping/warm-start resume parity" forbids.
            #
            # This is loud in the reverse direction already: SR state handed to
            # a stateless method raises. Making the missing direction loud too
            # is what closes the asymmetry. It also catches the realistic
            # mistake of resuming a legacy Adam checkpoint under an SR config,
            # where silence would train from a half-restored state.
            raise ValueError(
                "checkpoint state has no 'update_method' payload, but the resolved "
                f"update method {type(self._resolved_update_method).__name__} owns "
                "persistent method state; refusing a partial resume"
            )

        self._checkpoint_parameter_layout = restored_layout
        if restored_layout is not None:
            # `_load_trainer` runs after the checkpoint model and optimizer have
            # loaded. Rebinding here also makes the public restore API correct
            # when callers use `restore_checkpoint` directly rather than the
            # config runner.
            self.rebuild_update_state(model=self._resolved_model)
        self.next_iteration = next_iteration
        self.completed_updates = completed_updates

    def resolve_update_state(
        self,
        *,
        model,
        optimizer: torch.optim.Optimizer,
        update_method: UpdateMethodSpec = None,
    ) -> VMCUpdateState:
        """Resolve the one typed update-state authority before any mutation.

        Stateful update methods expose their owned optimizer through
        ``VMCUpdateState``.  The legacy optimizer argument is accepted only
        when it is that same object; otherwise runner restore or the training
        loop could mutate one optimizer while publishing another.
        Stateless methods use the supplied optimizer as their authority.
        """

        selected_update_method = self._select_update_method(
            model=model,
            optimizer=optimizer,
            update_method=update_method,
        )
        resolved_state = self._resolve_method_state(
            model=model,
            optimizer=optimizer,
            update_method=selected_update_method,
        )
        self._resolved_model = model
        self._resolved_update_method = selected_update_method
        self._resolved_update_state = resolved_state
        return resolved_state

    def rebuild_update_state(
        self,
        *,
        model,
    ) -> VMCUpdateState:
        """Rebuild the direct update binding against restored model objects.

        The checkpoint's layout is compared with the model after its weights
        have been loaded.  Only then are the update method's direct references
        replaced, so a resumed update cannot retain references to a pre-load
        model or silently accept a different parameter layout.
        """

        update_method = self._resolved_update_method
        update_state = self._resolved_update_state
        if update_method is None or update_state is None:
            raise RuntimeError("update state must be resolved before it can be rebuilt")
        expected_layout = self._checkpoint_parameter_layout or update_state.model_parameters.layout
        rebuilt_binding = update_state.model_parameters.rebind(
            tuple(model.parameters()),
            layout=expected_layout,
        )
        update_method.rebind_model_parameters(rebuilt_binding)
        rebuilt_state = update_method.update_state()
        if rebuilt_state is None:
            raise TypeError("stateful update method returned no update state after rebind")
        if rebuilt_state.optimizer is not update_state.optimizer:
            raise ValueError("mismatched legacy optimizer ownership")
        if not rebuilt_state.model_parameters.compare(rebuilt_binding)[0]:
            raise ValueError("update method did not retain the rebuilt model binding")
        self._resolved_update_state = rebuilt_state
        return rebuilt_state

    def _select_update_method(
        self,
        *,
        model,
        optimizer: torch.optim.Optimizer,
        update_method: UpdateMethodSpec,
    ) -> VMCUpdateMethod[AutogradUpdateInput]:
        """Select or construct the update method for one fit invocation.

        Notes
        -----
        The memo below is keyed on the SPEC's identity, not on the model or the
        optimizer. Fitting the SAME trainer again with a DIFFERENT model or
        optimizer under that same spec therefore returns the instance built for
        the first one, rather than rebuilding against the second.

        That is not silent today, and the reviewer judged disposal defensible:
        a stateful method validates its parameter binding at its first update
        (`_validate_binding` on the SR method) and raises when the scores no
        longer reference the parameters it holds, so the mismatch surfaces
        loudly rather than training the wrong tensors. It is recorded here
        rather than guarded because a guard would add a branch to buy an error
        message for a case that already fails loudly, and one trainer instance
        driving two different models is not a shape this project uses.

        If that ever becomes a real usage, key the memo on the resolved
        `VMCUpdateState` identity instead of on the spec alone.
        """

        spec = self.update_method if update_method is None else update_method
        # Reuse the instance already built from THIS spec. Selection runs twice
        # in a resumed run -- once from `resolve_update_state` before restore,
        # once from `fit` -- and a factory would otherwise yield two different
        # instances, so the checkpoint would load into the one then discarded.
        # Keying on the spec's identity covers a factory passed EXPLICITLY to
        # both calls, which an earlier version keyed on `self` alone and missed.
        if (
            spec is not None
            and self._resolved_update_method is not None
            and spec is self._update_method_spec
        ):
            return self._resolved_update_method
        selected_update_method = spec
        if selected_update_method is None:
            return LegacyAutogradUpdate(
                optimizer=optimizer,
                gradient_clip_norm=self.gradient_clip_norm,
                model_parameters=ModelParameterBinding(parameters=tuple(model.parameters())),
            )
        # A Hydra `_partial_` block resolves to a factory rather than a method,
        # because a stateful method needs the optimizer and the live parameter
        # binding, neither of which exists at config time. Completing it here
        # keeps the one place that already resolves the method as the only
        # place that knows how it is built.
        selected_update_method = make_update_method(
            selected_update_method,
            optimizer=optimizer,
            model_parameters=ModelParameterBinding(parameters=tuple(model.parameters())),
        )
        if not isinstance(selected_update_method, VMCUpdateMethod):
            raise TypeError("VMCTrainer update_method must be a VMCUpdateMethod")
        # Remember the spec alongside the instance it produced, so the guard at
        # the top of this method can reuse it on the next selection from the
        # same spec. The caller's constructor argument is deliberately left
        # unmutated; an earlier version overwrote `self.update_method` with the
        # constructed instance, which memoized the self path only.
        self._update_method_spec = spec
        self._resolved_update_method = selected_update_method
        return selected_update_method

    def _resolve_method_state(
        self,
        *,
        model,
        optimizer: torch.optim.Optimizer,
        update_method: VMCUpdateMethod[AutogradUpdateInput],
    ) -> VMCUpdateState:
        """Validate a selected method and return its single authority."""

        owned_state = update_method.update_state()
        if owned_state is None:
            resolved_state = VMCUpdateState(
                optimizer=optimizer,
                model_parameters=ModelParameterBinding.from_parameters(tuple(model.parameters())),
            )
            return resolved_state
        if not isinstance(owned_state, VMCUpdateState):
            raise TypeError("VMCUpdateMethod.update_state must return VMCUpdateState or None")
        if owned_state.optimizer is not optimizer:
            ownership_mismatch_message = "mismatched legacy optimizer ownership"
            raise ValueError(ownership_mismatch_message)
        expected_binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))
        if not owned_state.model_parameters.compare(expected_binding)[0]:
            raise ValueError("update method parameter binding does not match live model")
        return owned_state

    def fit(
        self,
        *,
        model,
        sampler,
        hamiltonian_terms,
        optimizer: torch.optim.Optimizer,
        context: RunContext,
        emit: Callable[..., None],
        update_method: UpdateMethodSpec = None,
    ) -> TrainerState:
        """Run the training loop and return the final `TrainerState`."""

        selected_update_method = self._select_update_method(
            model=model,
            optimizer=optimizer,
            update_method=update_method,
        )
        update_state = self._resolve_method_state(
            model=model,
            optimizer=optimizer,
            update_method=selected_update_method,
        )
        self._resolved_model = model
        self._resolved_update_method = selected_update_method
        self._resolved_update_state = update_state
        state = TrainerState(
            model=model,
            optimizer=update_state.optimizer,
            update_state=update_state,
            trainer=self,
            sampler=sampler,
        )
        # The hooks are owned by the adapter rather than by the live input
        # record. This keeps VMCStepData exact and ephemeral while preserving
        # the historical typed phase boundaries around backward and step.
        selected_update_method.set_step_scopes(
            backward_scope=lambda step: context.scope(Backward(step=step), state=state),
            optimizer_scope=lambda step: context.scope(OptimizerUpdate(step=step), state=state),
        )
        # One `TrainerState` instance is passed beside every typed occurrence
        # this loop emits, so a typed handler reads it at the moment it is
        # delivered. Its reference fields (`model`, `optimizer`, `trainer`,
        # `sampler`) are live everywhere; its value fields are assigned near the
        # end of the body, so a handler at a boundary *above* that assignment
        # reads the previous iteration's values -- and the constructor defaults,
        # including `step == -1`, on the first iteration.
        # Training-loop steps are 0-indexed for metrics and most callbacks.
        # Checkpoint callbacks reuse the resume cursor `next_iteration`.
        for step in range(self.next_iteration, self.max_steps):
            iteration = TrainingIteration(step=step)
            with context.scope(iteration, state=state):

                with context.scope(CollectSamples(step=step), state=state):
                    walkers, sampler_stats = sampler.collect_samples(
                        model, device=context.metadata.device
                    )
                with context.scope(BuildBatch(step=step), state=state):
                    batch = walkers.make_batch()
                with context.scope(LocalEnergy(step=step), state=state):
                    result = local_energy(
                        hamiltonian_terms,
                        model,
                        batch,
                        return_terms=self.return_terms,
                    )
                if isinstance(result, LocalEnergyResult):
                    total_local_energy = result.total
                    term_energies = result.terms
                else:
                    total_local_energy = result
                    term_energies = None

                # One forward, shaped by what the update method actually needs.
                # A score method receives its raw per-sample score blocks in
                # this same packet; running an ordinary forward and then
                # recomputing derivatives would double the step's forward and
                # derivative work.
                forward_request = selected_update_method.forward_request()
                with context.scope(Forward(step=step), state=state):
                    if forward_request is None:
                        output = model(batch)
                        parameter_scores = None
                    else:
                        packet = forward_request.evaluate(model, batch)
                        output = packet.as_output()
                        parameter_scores = packet.parameter_scores
                with context.scope(Objective(step=step), state=state):
                    objective = compute_vmc_objective(
                        output.logabs,
                        total_local_energy,
                        nonfinite_policy=self.nonfinite_local_energy_policy,
                    )
                loss = objective.loss

                # Read the parameter norm here, before any update, so the whole
                # `train` record describes exactly one model version: the model
                # that produced these samples, this loss, and this gradient.
                # Reading it after `optimizer.step()` would make `param_norm`
                # the sole post-update key in an otherwise pre-update record.
                # Both branches below reach the metrics block, so the skip path
                # reports this same pre-update value.
                param_norm = _parameter_norm(model)

                # Dispatch on the typed request the method already declared,
                # never on a name or a capability flag: a method cannot receive
                # a score input without having asked for the score payload.
                update_input: AutogradUpdateInput | ScoreUpdateInput
                if isinstance(forward_request, ParameterScoreRequest):
                    assert parameter_scores is not None
                    update_input = ScoreUpdateInput(
                        batch=batch,
                        wavefunction=output,
                        local_energy=total_local_energy,
                        step=step,
                        parameter_scores=parameter_scores,
                        parameter_binding=model.parameter_binding,
                    )
                else:
                    update_input = AutogradUpdateInput(
                        batch=batch,
                        wavefunction=output,
                        local_energy=total_local_energy,
                        step=step,
                        objective=loss,
                        # A TRUE closure optimizer re-evaluates the objective
                        # one or more times per step at parameters it has
                        # already mutated in place, so it cannot use `loss`:
                        # backward on that retained graph raises once the
                        # saved-tensor versions have moved. The re-evaluation
                        # is built over THIS step's fixed `batch` and carries
                        # the SAME resolved non-finite policy, so a second
                        # evaluation cannot resample, cannot advance the
                        # sampler RNG the resume path depends on, and cannot
                        # switch estimators midway through one step.
                        reevaluate=vmc_objective_reevaluation(
                            model=model,
                            hamiltonian_terms=hamiltonian_terms,
                            batch=batch,
                            primary_local_energy=total_local_energy,
                            nonfinite_policy=self.nonfinite_local_energy_policy,
                        ),
                    )
                update_result = selected_update_method.update(update_input)
                optimizer_step = False
                if update_result.applied:
                    # The update counts only once `optimizer.step()` has
                    # returned, so this always follows Ended[OptimizerUpdate].
                    self.completed_updates += 1
                    context.emit(UpdateCompleted(iteration=iteration), state=state)
                    optimizer_step = True
                    grad_norm = update_result.grad_norm
                else:
                    # A method that declines a step reports it; the trainer does
                    # not second-guess the reason. The zero-electron vacuum has
                    # no sampled coordinate degrees of freedom, so a no-op is
                    # correct there; a score method may also decline a step it
                    # cannot form, for example when too few samples survive the
                    # finite guard. No OptimizerUpdate scope opens on this path.
                    #
                    # There is deliberately no disconnected-loss check here.
                    # LegacyAutogradUpdate.update already raises for exactly
                    # that case before it can return, so a second check in the
                    # trainer added no protection while making every legitimate
                    # decline by a score method look like a disconnected loss.
                    grad_norm = update_result.grad_norm
                    context.emit(UpdateSkipped(iteration=iteration), state=state)

                # Canonical VMC-native metrics come from the objective helper;
                # the trainer only adds trainer-owned mechanics and optional
                # per-term local-energy metrics (metrics only, never part of the
                # objective).
                with context.scope(Metrics(step=step), state=state):
                    metrics: dict[str, Any] = dict(objective.metrics)
                    metrics.update(summarize_logabs(output.logabs))
                    if term_energies is not None:
                        metrics.update(summarize_local_energy_terms(term_energies))
                    metrics["grad_norm"] = grad_norm
                    metrics["param_norm"] = param_norm
                    metrics["loss_has_grad"] = bool(loss.requires_grad)
                    metrics["optimizer_step"] = optimizer_step
                    # Bounded, method-owned telemetry. The update method
                    # composes its own metric names, so the trainer never
                    # re-spells a solver key and a method that reports nothing
                    # adds nothing.
                    update_metrics = getattr(selected_update_method, "last_telemetry", None)
                    if update_metrics is not None:
                        metrics.update(update_metrics.as_metrics())

                state.step = step
                state.metrics = metrics
                # Published as `train/optimizer_step` above and carried here as
                # a typed field for the same reason: a callback observing update
                # by-products (gradients) must be able to tell "this iteration
                # applied no update, so there is nothing to see" from "an update
                # ran and I still saw nothing", which is a broken observation.
                state.optimizer_step = optimizer_step
                state.samples = walkers
                state.batch = batch
                state.local_energy = total_local_energy.detach()
                state.loss = loss.detach()
                state.wavefunction_output = output
                state.sampler_stats = sampler_stats

                if self.log_every_n_steps and step % self.log_every_n_steps == 0:
                    context.log(metrics, step=step, namespace="train")
                    # Explicit None check: SamplerStats defines no __bool__, so a
                    # truthiness test would always pass and could not express
                    # "this sampler produced no diagnostics". The typed record
                    # owns the metric-name composition, so the trainer never
                    # re-spells a train/sampler key.
                    if sampler_stats is not None:
                        context.log(
                            sampler_stats.as_metrics(),
                            step=step,
                            namespace="train/sampler",
                        )

                # Assigned here, near the end of the body, so only an iteration
                # that ran to completion advances the resume cursor: a crash
                # above retries this same step on resume. An iteration that
                # applied no optimizer update still advances it.
                self.next_iteration = step + 1
                context.emit(TrainingIterationCompleted(iteration=iteration), state=state)

        return state


__all__ = ["VMCTrainer"]
