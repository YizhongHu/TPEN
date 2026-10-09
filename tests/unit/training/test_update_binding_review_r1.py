"""Round-1 review probes for the one-bound-update-method lifecycle (PR 524).

ADOPTED BY THE WRITER, unchanged except where a disposition required it.
These are the reviewer's probes, not the writer's, and they are kept under the
reviewer's names so the provenance stays legible: every one of them found
something the writer's own suite did not.

Original state: all five GREEN at head 4175fc17c79a51e62d9d886ee3f18f1980aed280,
measured on FASRC-Cannon jobs 51674866 and 51675986.

Changed on adoption, and only these:
  - R1-1's probe pinned the HOLE (a second carrier accepted silently). The
    writer accepted the issue and closed it, so the probe is INVERTED to
    require the refusal. Its control changed message, because the binding
    boundary now raises its own distinguishable string.
  - R1-5's pin test is ADDED here rather than in the writer's file, because it
    is the reviewer's design.
R1-2, R1-3 and R1-4 are untouched: the writer kept the behavior R1-2 pins and
corrected the documentation instead, and R1-3/R1-4 were always acceptance arms
rather than disputes.

R1-1  A stateless method bound through the public
      `VMCTrainer.bind_update_method` accepts a SECOND carrier silently. The
      acceptance contract says a wrong model/optimizer binding fails before
      sampling, and the inline comment beside the carrier check claims the
      check exists precisely so "a second carrier would [not] be accepted
      silently" -- but the check derives the bound carrier from
      `_resolved_update_state` or `bound.update_state()`, both of which are
      `None` for a stateless method that was bound directly, so the check is
      skipped for exactly the method class it was added for. The control test
      shows the default adapter refuses the same call sequence, so the hole is
      specific to stateless methods.

R1-2  `bind_update_method`'s Parameters doc says an explicit
      ``update_method=None`` "never" conflicts, and the inline comment says it
      means "use what is already bound". The code implements neither: with an
      explicit ``None`` it re-reads the CONSTRUCTOR spec and conflict-checks
      that, so a trainer configured with one method and bound from an explicit
      override raises on a plain ``fit``. Raising may be the right call, but
      then both docstrings are wrong; one of code or doc must move.

R1-3  The acceptance contract names "SPRING history/trajectory match
      uninterrupted execution" and the writer's design note promised it as new
      test #5; the delivered suite covers Adam exactly and SPRING only at
      method level (pre-existing `test_spring_update`). This drives the real
      SPRING method through the full bind -> save_checkpoint -> restore ->
      rebuild -> fit lifecycle and requires exact history and parameter
      equality.

R1-4  The runner-driven construction-count test covers a FRESH run only. The
      resumed runner path -- the path where the original double-construction
      did its damage -- had no single-construction pin.
"""

from __future__ import annotations

import pytest
import torch

from tpen.checkpoint import restore_checkpoint
from tpen.runner import Train
from tpen.training.qgt import DampingPolicy
from tpen.training.score_geometry import ScoreConventions
from tpen.training.spring import SPRINGPolicy, SPRINGUpdate
from tpen.training.sr import SRPolicy
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ModelParameterBinding,
    VMCUpdateMethod,
    VMCUpdateResult,
)
from tests.helpers.hooke_models import (
    build_tiny_hamiltonian_terms,
    build_tiny_spenn,
)
from tests.helpers.run_context import make_run_context
from tests.unit.training.test_update_binding import (
    _checkpoint_context,
    _count_legacy_constructions,
    _fit,
    _FixedSampler,
    _save_resume_checkpoint,
)
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext

LEARNING_RATE = 0.01
SR_LEARNING_RATE = 1.0e-3


class _StatelessProbeMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """A stateless custom method: `update_state()` is the base-class `None`."""

    def __init__(self) -> None:
        self.update_calls = 0

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        self.update_calls += 1
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_probe")


# ---------------------------------------------------------------------------
# R1-1: the carrier check does not cover a directly-bound stateless method
# ---------------------------------------------------------------------------


def test_r1_1_stateless_direct_bind_refuses_a_second_carrier() -> None:
    """Bind carrier A, rebind carrier B: REFUSED.

    INVERTED ON ADOPTION. As written by the reviewer this probe pinned the
    HOLE -- it asserted that no refusal happened and that `fit` went on to
    publish carrier B. The writer accepted R1-1 and closed it by retaining the
    first-bind optimizer on the trainer, so the probe now requires the
    refusal it was built to expose.

    This is the path the carrier check was written for and did not cover: a
    stateless method bound DIRECTLY has no `VMCUpdateState` anywhere --
    `update_state()` is `None` and nothing has been resolved -- so a check that
    derived the bound carrier from update state had nothing to compare and
    passed silently.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    carrier_a = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    carrier_b = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    method = _StatelessProbeMethod()
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    first = trainer.bind_update_method(model=model, optimizer=carrier_a)
    assert first is method
    assert method.update_state() is None, "the probe must be genuinely stateless"

    with pytest.raises(ValueError, match="already bound to a different optimizer"):
        trainer.bind_update_method(model=model, optimizer=carrier_b)


def test_r1_1_control_the_default_adapter_refuses_the_same_sequence() -> None:
    """The identical call sequence IS refused when the method owns its state.

    Before the fix this control passed through `_resolve_method_state`'s older
    "mismatched legacy optimizer ownership" check, which is exactly why the
    hole above was invisible: the two guards raised the SAME string, so no test
    could tell which one refused. Both now refuse, and this one asserts the
    binding boundary's own message.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    carrier_a = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    carrier_b = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)

    trainer.bind_update_method(model=model, optimizer=carrier_a)
    with pytest.raises(ValueError, match="already bound to a different optimizer"):
        trainer.bind_update_method(model=model, optimizer=carrier_b)


# ---------------------------------------------------------------------------
# R1-2: explicit None is not "use what is bound", despite both docstrings
# ---------------------------------------------------------------------------


def test_r1_2_explicit_none_after_an_explicit_override_raises_despite_the_doc() -> None:
    """Configured spec F, bound from explicit override G, then a plain call.

    `bind_update_method`'s Parameters doc: ``None`` "never [conflicts]". Its
    inline comment: explicit ``None`` "means 'use what is already bound'".
    The code instead resurrects the constructor spec F and refuses. The raise
    may be the safer semantic -- but then the documentation is wrong twice.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def configured_factory(opt, *, model_parameters):  # never called
        raise AssertionError("the configured factory must not be constructed")

    trainer = VMCTrainer(
        max_steps=1, log_every_n_steps=1, update_method=configured_factory
    )
    override = LegacyAutogradUpdate(
        optimizer,
        model_parameters=ModelParameterBinding.from_parameters(tuple(model.parameters())),
    )

    bound = trainer.bind_update_method(
        model=model, optimizer=optimizer, update_method=override
    )
    assert bound is override

    with pytest.raises(ValueError, match="already bound from a different specification"):
        trainer.bind_update_method(model=model, optimizer=optimizer, update_method=None)


# ---------------------------------------------------------------------------
# R1-3: SPRING through the full checkpoint lifecycle, exactly
# ---------------------------------------------------------------------------


def _spring_factory(opt, *, model_parameters):
    """The configured-factory shape a Hydra ``_partial_`` block resolves to."""

    return SPRINGUpdate(
        opt,
        model_parameters=model_parameters,
        policy=SPRINGPolicy(
            base=SRPolicy(
                solve_space="auto",
                damping=DampingPolicy(absolute=0.0, relative=1.0e-2, minimum=1.0e-12),
                learning_rate=SR_LEARNING_RATE,
                max_update_norm=None,
            ),
            history_decay=0.4,
        ),
        conventions=ScoreConventions(solve_dtype=torch.float64),
    )


def _spring_run(*, max_steps: int, seed: int):
    """One seeded model/carrier/trainer triple configured for SPRING."""

    torch.manual_seed(seed)
    model = build_tiny_spenn()
    optimizer = torch.optim.SGD(model.parameters(), lr=SR_LEARNING_RATE)
    trainer = VMCTrainer(
        max_steps=max_steps, log_every_n_steps=1, update_method=_spring_factory
    )
    return model, optimizer, trainer


def test_r1_3_spring_history_and_trajectory_survive_a_real_checkpoint_resume(
    tmp_path,
) -> None:
    """Four SPRING steps straight equal two, a real checkpoint, and two more.

    Exact equality on both the parameters and the projected history, with a
    non-vacuity control that the history actually accumulated.
    """

    terms = build_tiny_hamiltonian_terms()

    straight_model, straight_optimizer, straight_trainer = _spring_run(max_steps=4, seed=0)
    straight_sampler = _FixedSampler()
    straight_trainer.resolve_update_state(
        model=straight_model, optimizer=straight_optimizer
    )
    straight_trainer.fit(
        model=straight_model,
        sampler=straight_sampler,
        hamiltonian_terms=terms,
        optimizer=straight_optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    assert straight_trainer.completed_updates == 4
    straight_method = straight_trainer.bind_update_method(
        model=straight_model, optimizer=straight_optimizer
    )
    assert isinstance(straight_method, SPRINGUpdate)
    # Non-vacuity: the history must have accumulated something to preserve.
    assert torch.linalg.vector_norm(straight_method.history).item() > 0.0
    expected = [p.detach().clone() for p in straight_model.parameters()]

    model, optimizer, trainer = _spring_run(max_steps=2, seed=0)
    sampler = _FixedSampler()
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    checkpoint = _save_resume_checkpoint(
        tmp_path, model=model, optimizer=optimizer, trainer=trainer, sampler=sampler
    )

    resumed_model, resumed_optimizer, resumed_trainer = _spring_run(max_steps=4, seed=99)
    resumed_sampler = _FixedSampler()
    # The runner's order: bind BEFORE restore.
    resumed_trainer.resolve_update_state(model=resumed_model, optimizer=resumed_optimizer)
    restore_checkpoint(
        load={"mode": "train_resume", "path": str(checkpoint)},
        model=resumed_model,
        context=_checkpoint_context(tmp_path),
        optimizer=resumed_optimizer,
        trainer=resumed_trainer,
        sampler=resumed_sampler,
    )
    resumed_trainer.rebuild_update_state(model=resumed_model)
    resumed_trainer.fit(
        model=resumed_model,
        sampler=resumed_sampler,
        hamiltonian_terms=terms,
        optimizer=resumed_optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )

    assert resumed_trainer.completed_updates == 4
    resumed_method = resumed_trainer.bind_update_method(
        model=resumed_model, optimizer=resumed_optimizer
    )
    assert isinstance(resumed_method, SPRINGUpdate)
    assert torch.equal(resumed_method.history, straight_method.history)
    for reference, resumed in zip(
        expected, resumed_model.parameters(), strict=True
    ):
        assert torch.equal(reference, resumed.detach())


# ---------------------------------------------------------------------------
# R1-4: the resumed RUNNER path constructs the default adapter exactly once
# ---------------------------------------------------------------------------


def test_r1_4_a_runner_driven_resume_constructs_the_default_adapter_once(
    tmp_path, monkeypatch
) -> None:
    """`Train.run` with ``load.mode='train_resume'`` builds ONE default method.

    The fresh-run counterpart exists in the writer's suite; the resumed path is
    the one the original defect damaged (restore landed on the discarded
    instance), and it was unpinned.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sampler = _FixedSampler()
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    _fit(trainer, model, optimizer, sampler=sampler)
    checkpoint = _save_resume_checkpoint(
        tmp_path, model=model, optimizer=optimizer, trainer=trainer, sampler=sampler
    )

    calls = _count_legacy_constructions(monkeypatch)
    torch.manual_seed(555)
    # `make_run_context` builds its `RunContext` with an empty `cfg`, but
    # `_save_resume_checkpoint` saved through `_checkpoint_context`'s populated
    # one. Restore hashes `context.cfg` against the manifest, so an empty cfg
    # is refused at the hash gate before the construction count this test
    # pins is ever exercised. `RunContext` is a plain (non-frozen) dataclass,
    # so align the two sides post-construction rather than hand-building a
    # second `RunContext`.
    resume_context = make_run_context(tmp_path / "resume-run")
    resume_context.cfg = _checkpoint_context(tmp_path).cfg
    resume_context.source_cfg = resume_context.cfg
    Train(
        model=build_tiny_spenn(),
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=lambda params: torch.optim.Adam(params, lr=LEARNING_RATE),
        trainer=VMCTrainer(max_steps=2, log_every_n_steps=1),
        load={"mode": "train_resume", "path": str(checkpoint)},
    ).run(resume_context)

    assert calls["n"] == 1, (
        "the resumed runner path constructed the default adapter more than "
        "once; restore landed on an instance the loop then discarded"
    )


# ---------------------------------------------------------------------------
# R1-5: the duck-typed runner contract must not narrow
# ---------------------------------------------------------------------------


def test_r1_5_a_trainer_with_a_foreign_resolve_contract_keeps_the_runner_path(
    tmp_path,
) -> None:
    """A non-VMC trainer returning a truthy object without ``.optimizer`` runs.

    `Train.run` DISCARDED `resolve_update_state`'s return value before PR 524,
    so any trainer implementing the duck-typed contract was free to return
    anything at all. Reading `.optimizer` off it unguarded would turn that into
    an `AttributeError` and break the runner path the acceptance contract
    requires preserving -- "Preserve other configured trainers' existing runner
    path".

    The reviewer raised this as an observation and the writer upgraded it to an
    accepted issue: nothing in this repository breaks today only because the
    one other implementer happens to return `None`, which is an accident rather
    than a guarantee. This test makes it a guarantee.
    """

    seen = {}
    built = {}

    class _ForeignTrainer:
        """Implements the duck-typed runner contract and nothing else."""

        next_iteration = 0

        def resolve_update_state(self, *, model, optimizer):
            del model, optimizer
            # Truthy, and deliberately WITHOUT `.optimizer`.
            return object()

        def fit(self, *, model, sampler, hamiltonian_terms, optimizer, context, emit):
            del model, sampler, hamiltonian_terms, context, emit
            seen["optimizer"] = optimizer
            return None

    def build_adam(params):
        # Capture the instance, so the assertion below can pin IDENTITY rather
        # than type. Strengthened after reviewer round 2 (R2-4): an `isinstance`
        # check passes for ANY Adam, including one the runner did not build,
        # so it could not distinguish "the runner kept its own carrier" from
        # "something else supplied a different Adam".
        built["optimizer"] = torch.optim.Adam(params, lr=LEARNING_RATE)
        return built["optimizer"]

    context = make_run_context(tmp_path / "foreign-run")
    torch.manual_seed(0)

    result = Train(
        model=build_tiny_spenn(),
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=build_adam,
        trainer=_ForeignTrainer(),
    ).run(context)

    assert result.status == "completed"
    assert seen["optimizer"] is built["optimizer"], (
        "the runner must hand `fit` the optimizer it built when the trainer's "
        "resolve contract returns something without a carrier"
    )
