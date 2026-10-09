# PR 524 round-1 review — DRAFT (pending worker job results)

Reviewer: fable-5, agent 8d064e60, worktree vmc-if-b-review-r1, branch
claude/review/vmc-if-b-r1-4175fc17. Head reviewed:
4175fc17c79a51e62d9d886ee3f18f1980aed280 (verified = local worktree HEAD).
Review tests: tests/unit/training/test_update_binding_review_r1.py at commit
d09eeb6a on the reviewer branch, md5 bb20355366f8c5b4d99c14d9a532ff3d (5 tests).
Worker agent d821671a (claude-sonnet-5[1m], bypassPermissions) running them on
Cannon, partition test, at the exact head + the review file.

## Verdict shape (fill counts after worker returns)

4 issues: 1 medium (R1-1), 3 low (R1-2, R1-3, R1-4). None invalidates the core
lifecycle fix. The five writer claims: 1, 2, 5 CONFIRMED; 4 confirmed
analytically (empirical arm is the verifier's M1); 3 REFUTED AS STATED.

## Writer-claim audit

1. **Runner change (`optimizer = update_state.optimizer`) is a behavioural
   no-op today — TRUE.** `_resolve_method_state` has exactly two return
   branches: stateless wraps the supplied optimizer; stateful passes only when
   `owned_state.optimizer is optimizer`. Either way `.optimizer is optimizer`.
   Only VMCTrainer defines `resolve_update_state` in-repo (grep). One
   observation, not an issue: the duck-typed runner contract narrowed — a
   third-party trainer whose `resolve_update_state` returns a non-None object
   without `.optimizer` now crashes with AttributeError where its return value
   was previously discarded. No in-repo implementer; worth one sentence in the
   runner comment at most.

2. **Removed `raise TypeError(...)` was unreachable — TRUE.**
   `VMCUpdateMethod` is a nominal ABC (update.py:1445), not a runtime
   Protocol, so isinstance is nominal. Every non-None return path of
   `make_update_method` is isinstance-checked inside that function (instance
   passthrough checked at entry; post-instantiate instance checked; callable
   result checked; everything else raises). The None path returned early in
   the old selector before `make_update_method` was called. The guard could
   never fire.

3. **Explicit `update_method=None` treated as "use what is bound" — REFUTED AS
   STATED (issue R1-2).** The code does not implement that semantic: with an
   explicit None it falls back to `self.update_method` and conflict-checks
   THAT against the bound spec. When the first bind used an explicit override
   on a trainer configured with a different method, a later plain
   `fit`/`bind` RAISES "already bound from a different specification" — while
   the Parameters doc says None is "never a conflict" and the inline comment
   says None means "use what is already bound". Raising is arguably the safer
   semantic (the old code silently rebuilt from the constructor spec and ran a
   DIFFERENT method than restore had landed on, which was worse), but code
   and both docstrings currently say three different things.

4. **New tests fail if the fix is reverted — TRUE analytically.** Under the
   old selector, the runner-driven counter test reads 2 constructions
   (resolve + fit) vs asserted 1; the resolve/fit identity test compares two
   different adapters' bindings with `is` and fails; the three refusal tests
   get no raise. Empirical confirmation is the verifier's M1 arm, not mine.

5. **Exact-equality Adam resume test deterministic, not lucky — TRUE.** All
   three arms run in one process (platform/BLAS variation cancels);
   `_FixedSampler` removes sampling randomness; both seeded arms perform
   identical op sequences; checkpoint round-trip of tensors is bit-exact; and
   if any fit-path code consumes global RNG, the checkpoint carries and
   restores RNG state (restore refuses checkpoints without it —
   tests/integration/training/test_train_runner.py:396), so the resumed arm
   still replays the straight arm's stream. Deterministic by construction.

## Issues

### R1-1 (medium) — stateless method bound via public `bind_update_method` accepts a second carrier silently

trainer.py, already-bound branch: `bound_state = self._resolved_update_state`
falls back to `bound.update_state()`. For a STATELESS method bound directly
through the new public API (no resolve/fit yet), both are None, so the carrier
check is SKIPPED — for exactly the method class the inline comment says the
check was added for ("for a stateless method it wraps whatever optimizer it is
handed, so without this a second carrier would be accepted silently"). The
sequence bind(model, optA) -> bind(model, optB) -> fit(optB) runs without
refusal and publishes optB. Acceptance clause violated: "Conflicting late
selector or wrong model/optimizer binding fails before sampling or mutable
restore." The default adapter refuses the identical sequence (owns its state),
so the hole is stateless-specific. Fix is trainer-local and in-scope: retain
the first-bind optimizer (e.g. `_bound_optimizer`) and validate against it
instead of deriving the carrier from state that may not exist.
Tests: test_r1_1_stateless_direct_bind_accepts_a_second_carrier_silently
(pins the hole; invert to expect refusal when fixed) +
test_r1_1_control_the_default_adapter_refuses_the_same_sequence.

### R1-2 (low) — explicit-None semantics: code contradicts both of its own docstrings

See claim 3 above. Concrete trigger: VMCTrainer(update_method=F);
bind(model, opt, update_method=G_instance); then bind/fit with
update_method=None -> ValueError. Pick one semantic: either late_spec should
not resurrect `self.update_method` when the caller passed None (then the
refusal disappears and None truly means use-what-is-bound), or the raise stays
and both docstrings change to say None re-reads the configured spec. I lean
keep-the-raise + fix docs: silent divergence between configured and bound is
the class of defect this slice exists to kill.
Test: test_r1_2_explicit_none_after_an_explicit_override_raises_despite_the_doc
(green either way once docs/code agree; today it documents the raise).

### R1-3 (low) — SPRING-through-checkpoint acceptance arm untested

Acceptance contract: "Adam moments/trajectory and SPRING history/trajectory
match uninterrupted execution." Design note promised it as new test #5.
Delivered: Adam exact-equality only; SPRING resume equality exists only at
METHOD level (test_spring_update.py:311, pre-existing, no trainer, no real
checkpoint). No test anywhere drives the real SPRINGUpdate through
bind -> save_checkpoint -> restore -> rebuild -> fit. The composite is
probably green (parts are covered), but the clause is a composite claim and
the binding slice is exactly where composition could break.
Test: test_r1_3_spring_history_and_trajectory_survive_a_real_checkpoint_resume
(exact history + parameter equality, 4 straight == 2 + checkpoint + 2).

### R1-4 (low) — the resumed RUNNER path has no single-construction pin

test_the_default_adapter_is_constructed_once_per_runner_driven_run drives
Train.run FRESH only. The resumed runner path is where the original defect did
its damage (restore landed on the discarded instance), and the pre-existing
runner-resume bitwise test cannot catch a reintroduced double-construction —
it passed throughout the defect's lifetime (the stateless adapter made double
construction numerically invisible, which is the writer's own point).
Test: test_r1_4_a_runner_driven_resume_constructs_the_default_adapter_once.

## Out-of-scope notes (not issues)

- Error message "mismatched legacy optimizer ownership" now also fires for
  non-legacy carriers at the binding boundary; cosmetic.
- Duck-type narrowing of the runner's resolve contract (claim-1 observation).

## Worker results — FILL IN

- Job id / partition delivered / node / elapsed / exit:
- JUnit arm 1 (review file, expect 5):
- JUnit arm 2 (writer file, expect 15):
- Edits needed to R1-3 / R1-4 fixtures:
- Final file md5:
