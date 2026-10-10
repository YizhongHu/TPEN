# Round-5 review VERDICT — head cbf5fcd48a414c89690a506160df874d56f6a658

2026-10-10, reviewer fable-5, worktree vmc-if-b-review-r5, operating under the
one-claude-child-per-lane ruling: probes designed, implemented and run by the
reviewer itself, no second party checked the fixtures — discount accordingly.

**VERDICT: BLOCK — 2 issues (R5-1 low, R5-2 low) + 2 nits. NO production-code
defect found, and NO test-logic defect found: the R4-2 pin survived direct
attack from four mutants and tests exactly what its name claims.** Both issues
are record/doc-strength; one is a one-line docstring fix, the other an
amending note. Accept/dispose is the writer's call.

## Evidence

Probe: `tests/unit/training/test_update_binding_review_r5.py` on
`origin/claude/review/vmc-if-b-r5-cbf5fcd4` @
`07ddcb94127ff579a74fa81d3bd762d746ab3364` — one control test, plus
`PREDICTIONS-R5-cbf5fcd4.md` committed and PUSHED in the same commit BEFORE
submission, including an exposure declaration of every writer number seen
before measuring, per-arm failing names, survivors a careless reader would
assume fail, and a registered DISAGREEMENT with the writer's
MUT-DOUBLEBUILD mechanism (below).

FACILITY: FASRC-Cannon. Job **51758034**, partition `test` requested and
DELIVERED, node holy8a24102, 4 CPU / 32 GiB, COMPLETED 0:0, 00:20:13.
Doctrine chain read in full this session: `3541fe33` `00-read-first-index` +
mandatory list, `00-read-first-cannon`, FASRC/Clusters orientations,
`cluster-access-read-first`. Submit-time stderr captured (empty); QOS headroom
1/5 at submit, not mine; nothing cancelled, ever. Run root (Active run data,
preserve, never delete):
`/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r5-20261010T034042Z/` —
own full clone (IS_SHALLOW=false, TAG_COUNT=24), own
`UV_PROJECT_ENVIRONMENT`/`UV_CACHE_DIR`/`TMPDIR` outside the checkout,
`apply_mutant.py` (md5 `95b4ef17…`, hash-matched both ends; every mutation
content-asserts the exact original line before writing), `job-r5.sbatch`
(md5 `7f98c9e1…`, matched), job-scoped `junit/<arm>-51758034.xml` and
`logs/<arm>-51758034.log`.

In-job, per arm: HEAD asserted == `07ddcb94` at 40 chars, whose diff vs the
reviewed head was printed in-job and touches NO `tpen/` file — the production
tree under test is byte-identical to `cbf5fcd4`. Tracked porcelain 0;
`__pycache__` purged and printed 0 before EVERY arm; interpreter job-local
uv-env CPython 3.12.13 / torch 2.12.0+cpu asserted >=3.10 and non-system;
pristine sha256 asserted in-job equal to the recorded lineage values
(`train.py` `a13eb986…` matching rounds 3/4, `trainer.py` `6e672382…`
matching round 4's pristine at 57baf429 — the last commit touches no
production file). All four mutants proven ACTIVE by sha256 difference plus
printed mutated line; all restored to pristine sha256 with porcelain 0;
FINAL_PORCELAIN=0. Counts from raw JUnit `<testsuite>` attributes, errors
separate from failures.

## Prediction vs result — 7 of 7 arms matched, by name AND mechanism

| arm | predicted | measured | match |
|---|---|---|---|
| g-focused (6 files) | 26 / 0 | tests=26 failures=0 errors=0 | ✓ |
| g-full tests/unit | 3055 / 0 / skip 5 | tests=3055 failures=0 errors=0 skipped=5 | ✓ exact (3054+1) |
| m-rc-focused | 1: the R4-2 pin, via the EQUALITY assert | failures=1, that name, msg "diverged from the uninterrupted run" | ✓ |
| m-rc-full | 1, same name | failures=1, same name, 3055 tests | ✓ |
| m-db-focused | 6 by name; pin fails via BINDING REFUSAL, not equality | failures=6, SAME six names; pin msg = the refusal; equality string count in junit = 0 | ✓ |
| m-rs-focused | 2: pin via equality + r5 control DID-NOT-RAISE | failures=2, exactly those, exactly those mechanisms | ✓ |
| m-a-focused | 4 by name, 2 DID-NOT-RAISE + 2 regex-mismatch | failures=4, same names, same split | ✓ |

Registered survivors all survived: under M-RC, `test_r1_4_*` (count-only),
`test_adam_moments_*` (bypasses `Train.run`) and the r5 control (gate refuses
before `_load_optimizer`); under M-RS, `test_r1_4_*` — the survivor a careless
reader would assume fails, because a skipped restore still builds exactly one
adapter; under M-DB, the r5 control (restore's refusal fires before the
mutated `fit` line).

Elapsed disclosure, since the bands were pre-announced: full arms 472 s and
459 s, inside the 420–800 s band. Focused arms ran 18–36 s against a
pre-announced 40–120 s band — UNDER it, all five. The band was set
conservatively from round-4's 24-test figure; failure-heavy mutant arms exit
early. Reported rather than absorbed; no arm was suspiciously short relative
to what it did (each testsuite line carries its own time attribute
consistent with the wall figure).

The disclosed `test_run_id_rank_agreement.py` Gloo flake did not fire in
either full arm — a null observation, not a re-verification.

Process note: the writer's failsafe wake cover fired mid-round reporting my
arms "absent from the queue ~4 minutes"; the job was in fact RUNNING at
00:03:06 under the name `tpen-r5-review-cbf5fcd4` (its recent-jobs list did
not include 51758034, so its queue filter missed the name). No action was
taken on the cover; it cost nothing, but a cover whose filter cannot see the
job it guards would have reported a hang as absence — worth one line here so
the next cover names the job id, not a name pattern.

## The decisive arms

**M-RC (r4's M4 payload, witness-free: restore handed a fresh factory build
while `fit` keeps the run's carrier).** Focused AND full: exactly one
failure, the R4-2 pin, failing through its equality assert. This
independently reproduces the writer's headline result at this head with my
own clone, env, and mutation script: the gap round 4 measured open across the
whole of `tests/unit` is shut by exactly the test that claims to shut it, and
nothing else in the tree sees the payload — which also re-confirms the
carried gap: the suite every receipt runs has exactly ONE line of defence on
this axis now.

**M-RS (novel this round: `Train.run`'s `train_resume` branch disabled).**
The pin fails via the equality assert (resumed arm trains fresh from a
seed-999 init) and my r5 control fails DID-NOT-RAISE. So the pin demonstrably
requires the real restore leg — it is not green-by-construction — and the
control pins that the runner actually enters restore. `test_r1_4_*` survives,
as registered: a count pin cannot see a missing restore.

**M-DB and the writer's miss decomposition — see R5-1.**

## Issues

### R5-1 (low) — the writer's decomposition of its own registered MUT-DOUBLEBUILD miss misattributes the failure mechanism; measured, not argued

`writer-redgreen-cbf5fcd4-compact` (and the r4-final disposition's framing)
says of the sixth failure: "DOUBLEBUILD hands `fit` a freshly built Adam. My
new pin compares a runner-driven resumed trajectory to an uninterrupted one
by EXACT equality. A fresh Adam has no moments, so the trajectory diverges
and the pin fires."

Measured at this head (m-db-focused, job 51758034): the pin fails with
`ValueError: update method is already bound to a different optimizer; a run
publishes and mutates one carrier` — the BINDING-BOUNDARY refusal, raised
from the pin's STRAIGHT arm's `fit` call before any trajectory, any restore,
or any comparison exists. The equality-assert string appears ZERO times in
the arm's JUnit; the refusal string appears in the pin's failure and both
count pins'. My prediction file registered this disagreement before the run.

What this changes: under DOUBLEBUILD the pin is a sixth test failing through
the SAME guard the two construction-count pins already pin — redundant
coverage of the refusal, not additional coverage through the pin's
trajectory-equality channel. The writer's drawn generalisation ("a pin that
observes a carrier will fire under any mutant that disturbs that carrier")
is mis-derived from this instance: the pin fired because it drives `fit`
through the refusing boundary, not because its observation channel saw the
carrier. What this does NOT change: the five originally-predicted names all
still fail (no prior coverage regressed — confirmed by my arm), the pin
does fail under DOUBLEBUILD, and the pin's equality channel IS load-bearing
where it matters (M-RC, M-RS — both fire through it, measured). The defect
is in the causal account inside the one artefact whose entire purpose was
honest decomposition of a registered miss.

FIX: amend by ADDING a correction note beside the original (leave the
original intact, per the lane's own reversing-a-published-verdict rule),
citing this round's m-db JUnit as the measurement.

### R5-2 (low) — the new probe file's module docstring overstates round 4's measurement

`tests/unit/training/test_update_binding_review_r4.py:6-7`: "and **every
test in `tests/unit` passed under it**, while a runtime witness proved the
mutated path had executed."

Round 4's M4-full run measured tests=3053 **failures=1**: a test in
`tests/unit` (`test_no_one_record_writer_opens_its_own_append_handle`) did
NOT pass — it caught the reviewer's witness instrument. The true statement,
which round 4's own verdict carried with the qualifier attached, is that no
test detected the PAYLOAD; the docstring dropped the qualifier. This is the
exact claims-vs-strings class the SAME COMMIT narrowed one paragraph away
(the R4-nit fix: "NEITHER EXISTS" → "neither survives as a live claim"), and
the fifth consecutive round with an overstated sentence at this cluster of
files — the writer's brief predicted a fifth and there it is. FAILURE
SCENARIO: a maintainer quotes "every test passed under the mutant" as the
blind-spot's measured size; the measured record says 3052/3053 with one
instrument-channel failure. FIX: one line, e.g. "no test detected the
payload (the single measured failure was the reviewer's own witness
instrument tripping the append-handle census)".

## Nits (non-blocking)

- **R5-3** — `test_update_binding_review_r2.py:190`: "see R3-2 in the
  round-3 probe file." The label "R3-2" occurs NOWHERE in
  `test_update_binding_review_r3.py` (the r3 file frames the same finding as
  the M-R2-4 answer); meanwhile a tree grep for "R3-2" lands on an UNRELATED
  reviewer's "R3-2" in `test_update_observations.py` (a different review
  lane's numbering). Introduced at 57baf429. A by-content pointer ("the
  single-slot-capture blindness recorded in the round-3 probe file's
  docstring") survives both the missing label and the collision.
- **R5-4** — `test_update_binding.py:284-287` (the R4-1 rewrite): "pinned by
  the adopted round-1 R1-2 probe in `…review_r1.py` **in the adopted round-1
  probes**" — the trailing qualifier is edit residue duplicating the new
  phrase. Redundant, not false; same one spot as R1-2/R2-2/R3-1/R4-1.

## Writer-claim audit — the five claims named for attack

1. **The R4-2 pin tests what its name claims, and is not passing or failing
   for an incidental reason — CONFIRMED, measured from four directions.**
   Green at pristine (26/0 focused, 3055/0/5 full); red under M-RC through
   the equality assert in both focused AND full (the only failure in 3055);
   red under M-RS through the same assert, proving the restore leg is real;
   red under M-DB through the binding refusal (which is the correction in
   R5-1, not a defect in the pin). Exactness is honest: both arms share one
   process, the sampler is fixed, RNG rides the checkpoint, and the resumed
   model is seeded differently so everything arrives through restore.
2. **`_aligned_context` does not mask a restore that should have been
   refused — CONFIRMED.** The gate's checks run unconditionally on the
   train_resume path (restore.py:135-176); alignment satisfies them with
   content-identical cfg, replicating the accepted r1_4 fixture. Empirical
   half: my r5 control drives the SAME runner resume path with the unaligned
   empty cfg and the gate refuses ("current config is missing model for
   restore") — green at pristine, and its M-RS failure shows it goes red the
   moment the runner stops calling restore.
3. **The non-vacuity control genuinely establishes the arm moved —
   CONFIRMED, with its residual stated.** `build_tiny_spenn` is
   deterministic under `manual_seed(0)`, so the control compares
   identical-init against trained and `any(param moved)` is the right
   predicate. Residual: it establishes 4-step movement, not steps-3-4
   movement; what establishes the comparison's discriminating power on the
   resumed leg is M-RC firing through it, which is now measured twice
   (writer's arm and mine, independent harnesses).
4. **The R4-1 rewrite is rename-surviving and no other stale symbol pointer
   exists — CONFIRMED with residue.** The new pointer names a probe by role
   and file. Independent tree grep for the retired symbol: exactly one hit,
   the deliberate past-tense history reference (`…review_r1.py:173`) —
   reproducing the writer's grep. Sweep of every cross-reference in the five
   probe/suite files found no broken SYMBOL pointer; the residue is R5-3 (a
   label that never existed in its target) and R5-4 (edit residue), plus
   R5-2 as the predicted fifth wording defect.
5. **Nothing in the slice breaks direct-fit, explicit-factory,
   bound-instance or stateless usage, or the checkpoint/resume contracts —
   CONFIRMED at the suite level** (3055/0/5, exact base+25+1, both full arms
   green except the single mutant-attributed failure), each mode pinned by a
   named test verified present. Bounded, as every green here is, by the
   carried gap: `tests/integration/` is still run by NO receipt in this PR,
   mine included.

## Scope calls and carried gaps

Shared "mismatched legacy optimizer ownership" string between
`_resolve_method_state` and `rebuild_update_state`: AGREE out of scope,
rounds 3/4 grounds — and note my M-A arm measured its cost again live (two
of the four failures are message-mismatch failures where the older guard
answers for the deleted one). All eight dispositions in
`carried-forward-known-gaps-at-cbf5fcd4`: CONCUR as recorded; item 1 (the
integration tree) remains the real bound on every green claim at this head,
and item 6's condition (the count pin covers the carrier-bearing shape only
while `Train.run` stays branch-free on the resolve return) was re-verified
against source this round.

## Discounts, stated plainly

- Writer and all round-1..5 reviewers are Claude (writer Opus 5, I am
  Fable 5); provider independence absent. I attacked artefacts — source
  traces, greps, my own clone/env/mutants — and declared every writer number
  I had seen in the predictions file before measuring; my green totals
  therefore corroborate the writer's DELTA, not independently the totals.
- One-child ruling: no second party checked my fixtures. Mitigations:
  predictions pushed pre-submission at `07ddcb94`; 7/7 arms matched by name
  and mechanism; the one registered DISAGREEMENT with the writer (M-DB
  mechanism) was resolved by the measurement, in my favour, which is what
  the registration was for — had it resolved the other way, the same file
  would show I was wrong.
- This round can end the review loop, and I verified my two issues against
  the measured record rather than softening them: R5-1 is a measured
  misattribution in the head's central honesty artefact, R5-2 a measured
  overstatement in the tree. Neither is production code; both are the class
  this lane has accepted in every round so far.

Head `cbf5fcd48a414c89690a506160df874d56f6a658` re-confirmed unmoved
(ls-remote + PR headRefOid) immediately before filing; any new commit
invalidates these measurements against the new SHA. Canonical copies: PR 524
comment + this file on `claude/review/vmc-if-b-r5-cbf5fcd4` (predictions at
`07ddcb94`, preserved unedited).
