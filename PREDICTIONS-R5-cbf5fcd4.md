# Round-5 predictions — REGISTERED BEFORE ANY SUBMISSION

Head under review: `cbf5fcd48a414c89690a506160df874d56f6a658` (PR 524).
Reviewer: fable-5, worktree `vmc-if-b-review-r5`, branch `claude/review/vmc-if-b-r5-cbf5fcd4`.
Nothing below is written after a result. This file is committed and pushed before
the first `sbatch`.

## Exposure declaration (per the Cannon read-first chain's standing rule)

Numbers seen before measuring, and where: writer's red/green at this head
(`writer-redgreen-cbf5fcd4-compact`): green-focused 25/0; green-full 3054/0/5
(base 3029 + 25); MUT-RESTORE-CARRIER focused 1 and full 1 (the R4-2 pin);
MUT-DOUBLEBUILD 6 (writer's registered prediction was 5; the miss is the R4-2
pin); MUT-A 4 by name. Round-4 figures (3053, 24, five M1 names) from the PR
thread. My green counts below are those figures plus exactly my one added test;
agreement with them is therefore NOT independent corroboration of the totals,
only of the delta. My mutant-arm FAILURE SETS are my own analysis from source;
the MUT-DB failure-message prediction is mine alone and contradicts the
writer's stated mechanism.

## Test inventory

Focused set = 6 files: `tests/unit/training/test_update_binding.py` (15),
`..._review_r1.py` (6), `..._review_r2.py` (2), `..._review_r3.py` (1),
`..._review_r4.py` (1), `..._review_r5.py` (1, mine) = **26 tests**.

## Arms

### G-FOCUSED (pristine tree + my probe)
- tests=26 failures=0 errors=0.

### G-FULL (pristine, all of tests/unit)
- tests=3055 failures=0 errors=0 skipped=5 (writer's 3054 + exactly my 1).
- Bounded by the disclosed file-scoped Gloo flake in
  `tests/unit/test_run_id_rank_agreement.py` (`a rank wrote no receipt`); if it
  fires it is decomposed against that signature, not absorbed and not a catch.

### M-RC — MUT-RESTORE-CARRIER (train.py:125 restore arg -> fresh `make_optimizer` build; r4's M4 payload, NO witness)
- Focused: failures=1, exactly
  `test_r4_2_a_runner_driven_adam_resume_is_bitwise_identical`, and its failure
  text contains "diverged from the uninterrupted run" (the equality assert).
- Full: tests=3055, failures=1, same single name.
- SURVIVORS a careless reader would expect to fail:
  `test_r1_4_a_runner_driven_resume_constructs_the_default_adapter_once` (pins
  adapter-construction COUNT, not carrier identity; the extra build is an
  optimizer, not an adapter), `test_adam_moments_and_trajectory_match_uninterrupted_execution`
  (bypasses `Train.run` entirely), and my
  `test_r5_the_hash_gate_is_live_on_the_runner_resume_path` (the gate refuses
  before `_load_optimizer`; the mutated argument is evaluated but its product
  never reached).

### M-DB — MUT-DOUBLEBUILD (train.py:144 fit arg -> second `make_optimizer` build)
- Focused: failures=6 errors=0, exactly:
  1. `test_the_default_adapter_is_constructed_once_per_runner_driven_run`
  2. `test_r1_4_a_runner_driven_resume_constructs_the_default_adapter_once`
  3. `test_r1_5_a_trainer_with_a_foreign_resolve_contract_keeps_the_runner_path`
  4. `test_r2_1_a_foreign_resolve_return_with_a_carrier_keeps_the_runner_optimizer`
  5. `test_r3_the_runner_builds_exactly_one_carrier_for_a_foreign_trainer`
  6. `test_r4_2_a_runner_driven_adam_resume_is_bitwise_identical`
- **THE POINT OF THIS ARM — the R4-2 pin's failure MECHANISM.** The writer's
  miss decomposition at this head says the pin fires because "a fresh Adam has
  no moments, so the trajectory diverges and the pin fires". My reading of the
  source says that is NOT what happens: in the pin's straight arm,
  `fit(optimizer=<second build>)` reaches `bind_update_method` on a trainer
  already bound to the first build, and the BINDING BOUNDARY refuses. The
  ValueError propagates out of `Train.run` before any trajectory exists.
  PREDICTION: the pin's failure text under this mutant contains
  "update method is already bound to a different optimizer" and does NOT
  contain "diverged from the uninterrupted run". If instead the equality
  message appears, the writer's account is right and mine is wrong; either
  way the arm decides it.
- SURVIVOR: my r5 control (restore's refusal fires before `fit`; the mutated
  line never executes on that path).

### M-RS — MUT-RESUME-SKIP (train.py:121 -> `if False and mode == "train_resume":`; NOVEL this round)
- Attacks the R4-2 pin's non-vacuity on the RESTORE leg: a pin that stayed
  green while the runner never restored would be decorative.
- Focused: failures=2 errors=0, exactly:
  1. `test_r4_2_a_runner_driven_adam_resume_is_bitwise_identical` — this time
     via the EQUALITY assert ("diverged from the uninterrupted run"): the
     resumed arm trains 4 fresh steps from a seed-999 init.
  2. `test_r5_the_hash_gate_is_live_on_the_runner_resume_path` — DID NOT RAISE.
- SURVIVORS a careless reader would expect to fail:
  `test_r1_4_a_runner_driven_resume_constructs_the_default_adapter_once`
  (restore skipped still builds exactly one adapter; the count assert cannot
  see a missing restore) and every direct-`restore_checkpoint` exactness test
  (they never enter `Train.run`).

### M-A — MUT-BINDCHECK-OFF (trainer.py:492 -> `if False and ...`; regression guard, writer ran it)
- Focused: failures=4 errors=0, exactly:
  1. `test_r1_1_stateless_direct_bind_refuses_a_second_carrier` (DID NOT RAISE)
  2. `test_r1_1_control_the_default_adapter_refuses_the_same_sequence` (DID NOT RAISE)
  3. `test_a_second_carrier_is_refused_after_binding` (raises, but
     `_resolve_method_state`'s "mismatched legacy optimizer ownership", so the
     match on the binding boundary's message fails)
  4. `test_a_refusal_happens_before_any_update_runs` (same mechanism as 3)
- SURVIVORS: `test_the_bound_instance_itself_is_not_a_conflicting_selector`,
  `test_r4_2_*`, my r5 control, and every test that never offers a second
  carrier.

## Elapsed bands (pre-announced)
- Focused arms: 40–120 s each (26 tests, two of them runner-resume).
- Full arms: 420–800 s.
- Any full arm materially outside the band is reported, not absorbed.

## Integrity plan
- In-job: assert `git rev-parse HEAD` == my branch tip; assert
  `git diff --name-only cbf5fcd4..HEAD` touches ONLY
  `tests/unit/training/test_update_binding_review_r5.py`,
  `PREDICTIONS-R5-cbf5fcd4.md`, and round-5 harness files — i.e. the
  production tree under test is byte-identical to the reviewed head.
- Pristine sha256 recorded locally before submission:
  `tpen/runner/train.py` = `a13eb986c27ddb140692018773fbf9c441c86fb30f6f818bf306ff5d6ddf48f7`
  (matches rounds 3/4), `tpen/training/trainer.py` =
  `6e6723820a533d5092409df12aef1987f4f1948d090b01accd6e62c39a0e92c5` (matches
  round 4's recorded pristine at 57baf429; the last commit touches no
  production file).
- Every mutant: exact-content assertion on the target line before writing,
  sha256 difference + printed mutated line as the activity proof, restore to
  pristine sha256 with porcelain clean after each arm, `__pycache__` purged
  and the remaining count printed to 0 before every arm.
- Counts read from raw JUnit `<testsuite>` attributes, errors separate from
  failures; artefacts job-scoped `junit/<arm>-<jobid>.xml` and
  `logs/<arm>-<jobid>.log`.
