# Round-4 review VERDICT (PARTIAL) — head 57baf42907d32432023b512634eab4c9d6c38935

2026-10-09/10, reviewer fable-5, worktree vmc-if-b-review-r4, operating under
the one-claude-child-per-lane ruling: probes designed and implemented by the
reviewer itself, no second party checked the fixtures — discount accordingly.

**VERDICT: BLOCK — 1 issue (R4-1, low) + 1 nit, both STATIC, standing on no
cluster evidence. The round's empirical arms are NOT MEASURED (infrastructure
loss, below), so this round additionally cannot supply the independent
verification the head currently lacks — a NO-BLOCK was not available from this
round even with zero findings.** Accept/dispose is the writer's call.

CANONICAL COPY: PR 524 comment + this file on
`claude/review/vmc-if-b-r4-57baf429`. Task Orchestrator was DOWN the entire
round: ECONNRESET at round start, then the Docker daemon died and every call
now returns ECONNREFUSED — observed by this session's own MCP connection both
times, not only reported. A TO note was attempted and failed; outcome recorded
in the PR comment.

## Infrastructure record, so the partial scope is auditable

- Cannon SSH worked at round start: clone, both job scripts staged, and green
  job 51728679 submitted (submission output read, submit-time stderr empty,
  test partition, headroom 1/5 at submit).
- A session restart then killed the multiplexed control socket. FASRC requires
  keyboard-interactive 2FA that cannot be supplied non-interactively.
- My own probe count: EXACTLY ONE, `BatchMode=yes` + `ConnectTimeout=8`,
  result `Permission denied (keyboard-interactive)` — the same signature the
  writer reported from its own two probes. Per `cluster-access` rule 5 I
  stopped there: no retries, reported, human assistance required.
- Consequence applied throughout: an arm whose logs and raw JUnit I cannot
  retrieve is NOT MEASURED — not failed, not passed, not evidence in either
  direction.

## Arm status

Predictions for every arm were registered to disk and pushed BEFORE any
submission: `PREDICTIONS-R4-57baf429.md` at commit
`f27937b0df2bd71d23b2e5f3ddecffb8e723d3ad`. Per-arm failure counts AND the
named tests expected to fail AND the named survivors a careless reader would
assume fail are all in that file; none of it was written after any result.

| arm | status |
|---|---|
| G-FOCUSED (4 files, predict 24/0) | SUBMITTED in Cannon job 51728679, UNREADABLE → NOT MEASURED |
| G-FULL (tests/unit, predict 3053/0/5) | same job, NOT MEASURED |
| M1 MUT-DOUBLEBUILD focused (predict 5 named failures) | NEVER SUBMITTED |
| M2 MUT-DOUBLEFIRST focused (predict exactly 1: the r3 count pin) | NEVER SUBMITTED |
| M3 MUT-SPECCHECK focused (predict 2 named DID-NOT-RAISE) | NEVER SUBMITTED |
| M4 MUT-RESTORE-CARRIER focused + full (predict 0 failures — blind-spot demonstration, witness-file activity proof) | NEVER SUBMITTED |

Job 51728679 may complete on its own; Slurm does not need our SSH. Its
artefacts (job-scoped junit/logs under the round root) remain valid evidence
FOR THE REGISTERED PREDICTIONS if read later by any party, because the
predictions are commit-timestamped before submission.

Netscratch run data, preserve, never delete:
`/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r4-20261010T003554Z/`
(checkout at the exact head, `job-a-green.sbatch`, `job-b-mutants.sbatch`,
`apply_mutant.py` with content-asserted mutations, `submit-stderr-a.txt`
(empty), and whatever job 51728679 wrote).

## Issues — static, complete, standing on no cluster evidence

### R4-1 (low) — a cross-reference to the R3-1-renamed test went stale IN THE SAME COMMIT that renamed it

`tests/unit/training/test_update_binding.py:284` — the docstring of
`test_an_explicit_none_late_selector_rereads_the_constructor_spec` (itself the
R2-2 fix) says the override-then-plain-call raise "is pinned by
`test_r1_2_explicit_none_after_an_explicit_override_raises_despite_the_doc`
in the adopted round-1 probes."

That name was DELETED at this head: the R3-1 fix renamed the probe to
`test_r1_2_explicit_none_after_an_explicit_override_raises_as_the_doc_states`
(`test_update_binding_review_r1.py:169`). The pointer was added at `dd1fa146`
(R2-2 fix) and went stale one commit later, in the commit under review —
`grep -rn "despite_the_doc" tests/ tpen/` at this head returns exactly two
hits: the legitimate past-tense history note in the renamed test's own
docstring (`..._review_r1.py:173`), and this live, present-tense reference.

FAILURE SCENARIO: a reader follows the named pin, finds no such test,
concludes the pin was dropped. Worse, the stale name itself carries the
refuted "raises despite the doc" claim back into the tree — the precise
wording class R3-1 was accepted to remove, now surviving in the writer's own
suite, which is where R2-2 established it does the most damage. This is the
fourth consecutive round with a stale-wording defect at this one spot (R1-2,
R2-2, R3-1, now this); the writer's brief predicted a fourth and named the
spot. FIX: one line — update the referenced name (and consider whether a
bare test name is the right pointer at all, given it has now gone stale once;
"the adopted round-1 R1-2 probe" survives renames).

Found statically; needs no cluster arm. The behaviour on both sides of the
pointer is correct and (per prior rounds' measurements at dd1fa146) tested;
this is a documentation-integrity defect only.

### R4-nit (non-blocking) — "NEITHER EXISTS AT THIS HEAD" is true of the claims, false of the strings

`tests/unit/training/test_update_binding_review_r1.py:175-176` (the R3-1
rewrite) says of the two refuted doc strings: "Both strings were real when
round 1 found them and NEITHER EXISTS AT THIS HEAD." The string
"use what is already bound" DOES exist at this head — `trainer.py:519`, inside
the R1-2 body comment, as a quoted historical reference ("an earlier version
of this comment claimed..."). The CLAIMS are gone; the string is not. A
maintainer grepping the refuted wording lands on trainer.py:519 holding a
docstring that says the string does not exist. Suggested wording: "neither
survives as a live claim; one is quoted, as history, at the check itself."
Filed as a nit, not an issue: the referent is unambiguous in context, and the
defect cannot mislead about behaviour, only about a grep.

## Writer-claim audit — what the static half established

The six claims named for attack, with the attack mode that applies:

1. **`builds[0]` closes R3-2 rather than relocating it — STATICALLY SOUND,
   EMPIRICALLY NOT MEASURED.** Design verified: all three foreign-path pins
   now share the list capture; `seen is builds[0]` kills feed-the-second-build
   on both foreign shapes, and the one direction it cannot see (build twice,
   feed the FIRST) is carried by the adopted count pin's `len(builds) == 1`.
   At this head the runner discards the resolve return unconditionally — there
   is no branch on the return's shape — so a count pin on the carrier-free
   foreign shape covers the carrier-bearing one too; that coverage claim is
   conditional on the runner staying branch-free there, and I verified the
   current `Train.run` is. No remaining single-slot capture exists
   (`grep 'built\['` over tests/ is empty). The kill predictions are M1/M2 in
   the registered file; they were not run.
2. **`late_spec` → `spec` collapse is behaviour-preserving — CONFIRMED
   STATICALLY, and this one needs nothing more.** `spec` (trainer.py:469,
   `self.update_method if update_method is None else update_method`) and the
   deleted `late_spec` (`update_method if update_method is not None else
   self.update_method`) are the same expression. Both were computed INSIDE the
   same call, so "`self.update_method` mutated between calls" cannot
   distinguish them: each call re-reads the attribute at line 469 exactly as
   the old code re-read it at the check site, and nothing on the path between
   lines 469 and 529 assigns `update_method` or `self.update_method`
   (verified line-by-line; `update_method` is a plain `__init__` attribute,
   trainer.py:137, not a property, so evaluation count cannot matter either).
   A revert-the-collapse mutant is equivalent by construction, which is why
   none was scheduled; M3 instead attacks whether the collapsed check is
   ALIVE, and was not run.
3. **Renamed R1-2 probe and docstring describe this head — LARGELY CONFIRMED,
   residue above.** The new name `..._raises_as_the_doc_states` is accurate:
   the head's Raises section (trainer.py:434-440) documents exactly that
   raise. The rewritten docstring and module-header paragraph are past-tense
   and accurate — except the R4-nit sentence. The predicted "fourth" doc error
   at this spot is real but landed in the NEIGHBOURING file: R4-1.
4. **Dropping the weakened-isinstance instrument lost nothing — CONFIRMED.**
   Diffed `5d2b569e:tests/unit/training/test_update_binding_review_r3.py`
   against the head file: the adopted body is byte-identical to the original
   minus exactly the instrument test; the header rewrite discloses the
   deletion, records the measurement the instrument existed to make, and the
   deletion follows the reviewer's own written instruction in the original
   header ("keep the docstring's verdict with it or delete the test"). The
   quoted sentence "a weakened twin that outlives its measurement reads as a
   second pin and is not one" is verbatim from the original (checked against
   the round-3 commit, not retyped from memory). The instrument remains
   recoverable at `5d2b569e` and the head file names round 3 and the head it
   ran at, which suffices to find it.
5. **The `assert builds` guards report the right diagnosis — CONFIRMED
   STATICALLY.** Ordering is right in all three tests: `result.status`
   is asserted first, so a run that crashed cannot reach the builds assert;
   an empty `builds` with a completed run means exactly what the message says
   (configured factory never invoked); without the guard the next line's
   `builds[0]` would raise IndexError and mask it. The r3 count pin's
   f-string reports the observed count.
6. **Usage modes and checkpoint/resume contracts preserved — NOT MEASURED at
   this head by this round** (the green arms are unread). One ANALYTIC
   finding, explicitly flagged as analysis awaiting its registered arm (M4):
   the runner-driven restore-carrier axis of the resume contract appears
   unpinned in `tests/unit`. `test_r1_4_*` pins the construction COUNT only;
   every exactness resume test in the focused files calls
   `restore_checkpoint` directly, bypassing `Train.run`; block-NG's
   runner-driven bitwise resume uses a stateless SGD carrier. The one test
   that would catch a restore-fed-the-wrong-carrier regression
   (`tests/integration/training/test_train_runner.py::test_resume_reproduces_the_uninterrupted_run_bitwise`,
   Adam, full config path) lives in the INTEGRATION tree, which no
   verification receipt in this PR runs. M4's registered prediction is that
   its mutant survives all of `tests/unit`; until an arm runs, this is a
   reading of the tree, not a measurement, and I am deliberately NOT filing
   it as an issue on analysis alone — it is the first thing round 4's
   completion (or round 5) should measure.

SCOPE CALLS, as asked: the shared "mismatched legacy optimizer ownership"
string stays a follow-up — AGREE, same grounds as round 3. No other scope
disagreement. The known `test_run_id_rank_agreement.py` flake: CONCUR with
pre-existing/disclose-not-chase on the record as it stands; I could not add a
measurement either way this round, so the writer's disposition is inherited,
not independently re-verified.

## Standing of the head after this round

- No production-code defect found by the static half. R4-1 and the nit are
  test/doc-integrity, one line each.
- There is still NO independent verification at this head: the writer's own
  job 51724017 results are the only measurements at 57baf429, and
  `tip-verification-dd1fa146` covers a superseded SHA. Whoever next has
  cluster access should run the registered arms (predictions at `f27937b0`
  are binding — they were pushed before job submission and must not be
  edited); job 51728679's artefacts, if it completed, satisfy the two green
  arms against those same predictions.

## Discounts, stated plainly

- Writer, prior reviewers and I are all Claude; I attacked artefacts (diffs,
  greps, byte comparisons against the round-3 commits) rather than intent,
  but provider diversity remains absent.
- One-child ruling: nobody checked my fixtures. The mutation scripts assert
  the exact original line content before mutating, so a drifted line fails
  loudly rather than mutating the wrong thing — but no arm ran, so this round
  the discount applies to the ANALYSIS, which has no empirical corroboration
  of its own yet.
- Everything empirical in this verdict is inherited from prior rounds'
  receipts at SUPERSEDED SHAs and is labelled as such where used.
