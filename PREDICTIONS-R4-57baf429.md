# Round-4 review: registered predictions BEFORE results

Reviewer: fable-5, worktree vmc-if-b-review-r4. Head under review:
`57baf42907d32432023b512634eab4c9d6c38935` (PR 524). Written and committed
BEFORE any Cannon job was submitted; the commit timestamp is the registration
time. Facility: FASRC-Cannon, partition `test`, in-allocation only.

Focused set (4 files, 24 tests expected):
- tests/unit/training/test_update_binding.py (15)
- tests/unit/training/test_update_binding_review_r1.py (6)
- tests/unit/training/test_update_binding_review_r2.py (2)
- tests/unit/training/test_update_binding_review_r3.py (1)

All counts to be read from raw JUnit `<testsuite>` attributes, errors
separately from failures. Every mutant must be proven ACTIVE (sha256 differs
from pristine + mutated region printed in the log) and restored to pristine
sha256 with tracked porcelain clean before the next arm. `__pycache__` purged
and the remaining count printed before every arm.

## Arm predictions

### G-FOCUSED (green, 4 files)
tests=24 failures=0 errors=0.

### G-FULL (green, tests/unit)
tests=3053 failures=0 errors=0 skipped=5 (base 3029 + 24 exactly).
Bounded by the disclosed pre-existing flake: if a failure appears in
`tests/unit/test_run_id_rank_agreement.py` with signature
"AssertionError: a rank wrote no receipt", disposition is flake-family
(file-scoped per round 3), to be rerun once in isolation and reported either
way, not absorbed.

### M1 = MUT-DOUBLEBUILD (tpen/runner/train.py: the `fit` call's
`optimizer=optimizer` replaced by a second
`make_optimizer(self.optimizer, self.model.parameters())`)
Round 3 measured the then-current pins r1_5/r2_1 GREEN under this exact mutant
shape one commit earlier; the head under review claims to close that (R3-2).
Focused: failures=5, BY NAME:
1. test_r1_5_a_trainer_with_a_foreign_resolve_contract_keeps_the_runner_path
   (seen is builds[0] now fails: fit receives the second build)
2. test_r2_1_a_foreign_resolve_return_with_a_carrier_keeps_the_runner_optimizer
   (same mechanism)
3. test_r3_the_runner_builds_exactly_one_carrier_for_a_foreign_trainer
   (len(builds)==2)
4. test_the_default_adapter_is_constructed_once_per_runner_driven_run
   (VMC path: fit's second carrier hits the binding boundary refusal
   "already bound to a different optimizer")
5. test_r1_4_a_runner_driven_resume_constructs_the_default_adapter_once
   (same refusal on the resumed path)
Predicted SURVIVORS a careless reader might assume fail:
test_a_configured_factory_is_constructed_once (trainer-level, runner not
involved); test_adam_moments_and_trajectory_match_uninterrupted_execution,
test_r1_3_spring_history_and_trajectory_survive_a_real_checkpoint_resume,
test_the_default_adapter_survives_restore_as_one_instance (all drive the
trainer/restore directly, never Train.run).

### M2 = MUT-DOUBLEFIRST (tpen/runner/train.py: immediately after the one
sanctioned `make_optimizer` call, add a SECOND
`make_optimizer(self.optimizer, self.model.parameters())` whose result is
DISCARDED; fit and restore keep receiving the FIRST build)
This is the round-4 question: does `builds[0]` close R3-2 or relocate it?
Direction: double-build where fit gets the FIRST carrier.
Focused: failures=1, BY NAME:
1. test_r3_the_runner_builds_exactly_one_carrier_for_a_foreign_trainer
   (len(builds)==2)
Predicted SURVIVORS, registered because a careless reader would assume they
fail: test_r1_5_* and test_r2_1_* (both assert `seen is builds[0]`, which the
mutant SATISFIES — they carry no count assertion; the count pin alone guards
this direction); test_the_default_adapter_is_constructed_once_per_runner_driven_run
and test_r1_4_* (binding boundary only ever sees the first carrier).
If anything OTHER than the count pin fails, my model of the capture design is
wrong and the verdict must say so.

### M3 = MUT-SPECCHECK (tpen/training/trainer.py: the collapsed late-selector
conflict check's first conjunct `spec is not None` replaced by `False`, so the
refusal can never fire)
Attacks the R3-3 `late_spec`->`spec` collapse: proves the collapsed check is
live and pinned at this head.
Focused: failures=2, BY NAME:
1. test_r1_2_explicit_none_after_an_explicit_override_raises_as_the_doc_states
   (DID NOT RAISE)
2. test_a_conflicting_late_selector_is_refused_after_binding (DID NOT RAISE)
Predicted SURVIVORS: test_the_bound_instance_itself_is_not_a_conflicting_selector
(asserts no raise), test_an_explicit_none_late_selector_rereads_the_constructor_spec
(default-spec case, no conflict), test_a_refusal_happens_before_any_update_runs
(fires the CARRIER guard, not the spec guard).

### M4 = MUT-RESTORE-CARRIER (tpen/runner/train.py: the restore call's
`optimizer=optimizer` replaced by a fresh
`make_optimizer(self.optimizer, self.model.parameters())`, so checkpoint
restore mutates a carrier the loop never uses)
This attacks the claim "checkpoint/resume contracts preserved AND PINNED" on
the runner-driven path. Static analysis says tests/unit cannot see it:
- test_r1_4 pins the CONSTRUCTION COUNT only (the adapter count stays 1);
- every exactness resume test in the focused files
  (test_adam_moments_*, test_r1_3_*, test_the_default_adapter_survives_restore_*)
  calls restore_checkpoint DIRECTLY, bypassing Train.run;
- block-NG's runner-driven bitwise resume uses a stateless SGD carrier, so a
  lost optimizer state dict is numerically invisible there;
- the one test that WOULD catch it
  (tests/integration/training/test_train_runner.py::test_resume_reproduces_the_uninterrupted_run_bitwise,
  Adam, run_from_config -> Train) is in the INTEGRATION tree, which no
  verification receipt in this PR runs.
Focused: failures=0 (all 24 GREEN — the blind-spot demonstration).
Full tests/unit: failures=0 predicted (same 3053/5 as green), with the stated
uncertainty that an unknown runner-driven resume assertion somewhere in
tests/unit falsifies this; any failure must be decomposed by name, not
absorbed.
If BOTH M4 arms come back green, the finding is: the runner-driven
restore-carrier axis of the resume contract is unpinned in tests/unit at this
head; adoptable fix is a runner-driven Adam resume exactness pin (or extending
test_r1_4 with trajectory equality against an uninterrupted arm).

## Elapsed-time controls, pre-announced
Job A (green): fresh env, `uv run --extra cpu --locked` sync ~1-2 min
(prescribed sync, NOT an install), focused ~40-90 s, full tests/unit
~420-670 s. Total ~12-18 min.
Job B (mutants): REUSES job A's env, so per-focused-arm elapsed of ~60-120 s
is legitimate and pre-announced here; the M4 full arm adds ~420-670 s.
Total ~15-20 min.

## Static findings already registered before the run (independent of arms)
R4-1 (low): tests/unit/training/test_update_binding.py:284 docstring
cross-references `test_r1_2_explicit_none_after_an_explicit_override_raises_despite_the_doc`,
a name this very commit DELETED via the R3-1 rename (now
`..._raises_as_the_doc_states`). The stale pointer reintroduces the refuted
"despite the doc" wording and names a test that no longer exists.
R4-nit: tests/unit/training/test_update_binding_review_r1.py:175-176 claims
the two refuted strings "NEITHER EXISTS AT THIS HEAD"; the string
"use what is already bound" does exist at tpen/training/trainer.py:519, as a
quoted historical reference. The CLAIMS are gone; the string is not.
