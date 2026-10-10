# Round-6 predictions — head f2006235a81df8655fe723fe6f9309b4afd714b2

Reviewer fable-5, worktree vmc-if-b-review-r6, 2026-10-10. Registered and PUSHED
BEFORE any job submission. One Claude child per lane: probes designed,
implemented and run by this reviewer alone; no second party checked these
fixtures.

## Exposure declaration — every number seen before measuring

From the writer's receipts and rounds 1-5 (PR 524 thread + Task Orchestrator
item b6b5b4f5): focused=25, full tests/unit=3054/0/skip 5, base count 3029,
round-4 24/3053, round-4 M4-full failures=1 (sole failure
`test_durable_append...append_handle`, instrument-attributed), round-5 all arm
results INCLUDING mechanisms (M-RC fails exactly the r4_2 pin via equality;
M-DB fails 6 with the pin failing via the binding refusal; M-A fails 4 as
2 DID-NOT-RAISE + 2 message-mismatch), r5 pristine/mutated sha256 values
(recovered from the r5 run root logs, listed below), round-2 M-A 4/17,
round-3 25/3054. For the INTEGRATION arms no prior number exists anywhere in
this PR's record — zero exposure; those arms are the least anchored
measurements this lane has produced since round 2's base arm.

## Payload identity, pinned ahead of time

Mutants are byte-identical reconstructions of round 5's, content-asserted
before writing. The job must reproduce these sha256 transitions exactly, or
the arm is measuring a different mutant and its result is void:

- pristine `tpen/runner/train.py` = `a13eb986c27ddb140692018773fbf9c441c86fb30f6f818bf306ff5d6ddf48f7`
- pristine `tpen/training/trainer.py` = `6e6723820a533d5092409df12aef1987f4f1948d090b01accd6e62c39a0e92c5`
- M-RC mutated train.py = `332f1eaec858af31794efe2f8a47c5a83b1db9f71131f724c28a883038bc9734`
- M-DB mutated train.py = `f36ba3fd12dfd902f0c82c62250d61afda6cfc22140012496f2b18edfd3ea8cd`
- M-A mutated trainer.py = `f19f7227c56a19901b1fe65aee653b59ca08ffe22705afdcc43350e4779a7d44`

(Verified locally before submission: both pristine values match this head, so
the writer's byte-identity claim for skipping mutant arms holds at the file
level. The arms below convert that argument into a measurement anyway: mutant
results are a joint function of production AND test trees, and the docs-only
commit changed test-module `__doc__` constants. I checked that nothing in
tests/unit asserts on a test module's `__doc__` — the only `__doc__`
assertions target `tpen` modules — so I predict full carryover; the arms are
what make that a fact rather than an argument.)

## Registered positions on the contested claims

1. `correction-my-doublebuild-decomposition-was-wrong` (the corrected
   mechanism): I AGREE with the corrected account and will re-measure it. My
   independent derivation from source: under M-DB the straight arm's
   `Train.run` binds the run carrier in `resolve_update_state`, then hands
   `fit` a fresh `make_optimizer` build; `bind_update_method` sees
   `optimizer is not self._bound_optimizer` and raises the binding refusal
   before any step, restore, or comparison exists. Prediction: the r4_2 pin's
   m-db failure text contains "already bound to a different optimizer" and
   does NOT contain "runner-driven resume diverged".
2. The writer's claim that `tests/integration/training/test_train_runner.py::
   test_resume_reproduces_the_uninterrupted_run_bitwise` would catch R4-2's
   payload: I predict TRUE, mechanism below (arm m-rc-int). Never measured by
   anyone; if it survives M-RC, round 4's central coverage claim and the
   carried-gap-#1 disposition both need rewriting, and that is a finding
   worth the round.
3. Integration tree at this head: never run by any receipt in this PR. I
   predict GREEN (6/6) at head AND at base dbb486ea — the slice binds the
   method pre-restore and restores into the bound instance, which my static
   trace says preserves the real resume path. A head-only red here is a
   production finding outranking everything else in this round.

## Arms, in run order — expected counts, named failures, mechanisms, survivors

Focused set = 5 files: test_update_binding.py + review_r1 + review_r2 +
review_r3 + review_r4 (15+6+2+1+1 = 25 tests).

| # | arm | tree | expected |
|---|-----|------|----------|
| 1 | g-focused | head, pristine | tests=25 failures=0 errors=0 |
| 2 | g-full | head, pristine, tests/unit | tests=3054 failures=0 errors=0 skipped=5 |
| 3 | g-int-head | head, pristine, integration file | tests=6 failures=0 errors=0 |
| 4 | g-int-base | base dbb486ea, pristine, same file (byte-identical at both SHAs) | tests=6 failures=0 errors=0 |
| 5 | m-rc-focused | head + M-RC | failures=1: `test_r4_2_a_runner_driven_adam_resume_is_bitwise_identical`, via its EQUALITY assert ("runner-driven resume diverged") |
| 6 | m-rc-full | head + M-RC, tests/unit | failures=1, same single name, tests=3054 |
| 7 | m-rc-int | head + M-RC, integration file | failures=1: `test_resume_reproduces_the_uninterrupted_run_bitwise` |
| 8 | m-db-focused | head + M-DB | failures=6, names below |
| 9 | m-a-focused | head + M-A | failures=4, names below |

### Arm 5/6 (M-RC) mechanism and survivors
The mutant hands `restore_checkpoint_with_events` a fresh factory build while
`fit` keeps the run's bound carrier; restored Adam moments land in the
discarded object. Named survivors I expect, because a careless reader would
expect them to fail: `test_r1_4_*` (counts ADAPTER constructions — the mutant
builds a second OPTIMIZER, not a second adapter), `test_adam_moments_*` and
`test_r1_3_*` and `test_the_default_adapter_survives_restore_as_one_instance`
(all call `restore_checkpoint` directly, bypassing the mutated runner line),
both r1_1 probes, both r2 probes, the r3 count pin (the mutated line runs only
under `mode == "train_resume"`, and those tests never resume).

### Arm 7 (m-rc-int) — the never-measured claim
`uninterrupted_run` fixture: no `load`, mutated line not executed, fixture
green. `test_resume_reproduces_the_uninterrupted_run_bitwise` FAILS at
`assert _diverged_parameters(final_a, final_b) == []`
(test_train_runner.py:332): arm B's restore loads optimizer state into the
fresh build, the loop steps a cold Adam from step 3, trajectories diverge,
the assert reports a NON-EMPTY diverged-parameter list. The run itself exits 0
— no refusal fires, because the bound method and the run carrier still agree.
Named survivors: `test_train_runner_writes_standard_artifacts` and
`test_train_runner_logs_finite_train_metrics` (no resume),
`test_training_resume_records_restored_checkpoint_identity` (restore succeeds
and truthfully reports loaded components; it checks identity fields, not
arithmetic), `test_resume_is_refused_when_the_checkpoint_carries_no_rng_state`
(refused at the manifest gate before the mutated line matters),
`test_resume_diverges_when_the_restored_sampler_stream_is_perturbed` (asserts
divergence, which the mutant only makes more true).
If instead the bitwise test SURVIVES, the writer's R4-2 coverage claim is
refuted and carried gap #1's disposition is wrong — finding of the round.

### Arm 8 (M-DB) — six failures, by name and mechanism
1. `test_the_default_adapter_is_constructed_once_per_runner_driven_run` — binding refusal ValueError propagates out of `Train.run`.
2. `test_r1_4_a_runner_driven_resume_constructs_the_default_adapter_once` — same refusal on the resumed path.
3. `test_r1_5_a_trainer_with_a_foreign_resolve_contract_keeps_the_runner_path` — identity: `seen["optimizer"] is builds[0]` false (fit fed the second build).
4. `test_r2_1_a_foreign_resolve_return_with_a_carrier_keeps_the_runner_optimizer` — same identity failure.
5. `test_r3_the_runner_builds_exactly_one_carrier_for_a_foreign_trainer` — `len(builds) == 1` sees 2.
6. `test_r4_2_a_runner_driven_adam_resume_is_bitwise_identical` — the binding REFUSAL from the straight arm's fit (see registered position 1), NOT the equality assert.
Survivors: everything else, notably both r1_1 probes and all direct-restore
exactness tests (no runner involvement).

### Arm 9 (M-A) — four failures, by name and mechanism
1. `test_r1_1_stateless_direct_bind_refuses_a_second_carrier` — DID NOT RAISE (bind-only path; the disabled check was the only guard there).
2. `test_r1_1_control_the_default_adapter_refuses_the_same_sequence` — DID NOT RAISE (also bind-only; `_resolve_method_state`'s older guard is never reached by `bind_update_method`).
3. `test_a_second_carrier_is_refused_after_binding` — raises, but the OLDER guard's string "mismatched legacy optimizer ownership" → regex mismatch against "already bound to a different optimizer".
4. `test_a_refusal_happens_before_any_update_runs` — same message mismatch, reached through `fit` → `_resolve_method_state`.
Survivors: `test_r4_2_*` (fresh trainer per `Train`, no second-carrier event),
`test_a_second_model_is_refused_after_binding` (model check untouched),
`test_a_conflicting_late_selector_is_refused_after_binding` (spec check untouched).

## Elapsed bands, pre-announced
Fresh locked env sync 40-150 s per checkout (two checkouts). g-focused and all
focused mutant arms 15-70 s each (fail-fast refusals keep mutant arms short).
g-full and m-rc-full 420-700 s each. Integration arms: FIRST MEASUREMENT, wide
band 60-900 s each (six configured 6-step runs with Metropolis sampling at 256
walkers, float64 CPU). Whole job expected 25-55 min; wall requested 02:30:00.

## Disposition pre-registrations
- If `tests/unit/test_run_id_rank_agreement.py` fails with "a rank wrote no
  receipt" in a full arm: the disclosed pre-existing file-scoped Gloo flake
  (carried gap #3) — report as flake-fired with the count decomposed, not a
  new finding, and do not rerun to make it green.
- If g-int-base and g-int-head are BOTH red with the same signature: harness
  or environment artefact (both-sides discriminator), investigate before
  attributing anything to the head.
- If any mutated sha256 differs from the pinned values above: the arm is void;
  say so rather than reading its counts.
