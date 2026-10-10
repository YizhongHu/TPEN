# Round-7 review VERDICT — head 1132e910324d12d75d73403f0fb2762b04209118

**BLOCK — 3 issues (R7-1 low, R7-2 low, R7-3 low) + 1 non-blocking observation.
NO production-code defect (fifth consecutive round), NO test-logic defect
(third consecutive round). All three issues are doc/comment-strength, in the
same family the last two commits set out to remove — two of them survive those
commits, one is introduced by them.**

All 5/5 empirical arms matched the pre-registered predictions by NAME and
MECHANISM (predictions pushed BEFORE submission at
`claude/review/vmc-if-b-r7-1132e910` @ fb65f3bc, with exposure declaration and
a pre-registered VOID condition that was checked first and passed).

## Facility and identity

FASRC-Cannon, job **51813678**, partition `test` requested and DELIVERED
(`PART=test`), node holy8a24102, COMPLETED 0:0, Elapsed 00:11:00. Run root
(Active run data, preserve):
`/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r7-20261010T095301Z/`.
Doctrine chain read in full before access: TPEN root
`cluster-access-read-first`, Clusters `00-orientation`, FASRC `00-orientation`,
Cannon `00-read-first-cannon`, policy item 3541fe33 `00-read-first-index` plus
its mandatory list. Full clone (IS_SHALLOW=false, TAGCOUNT=24); in-job asserts
all passed: HEAD=1132e910…, PORCELAIN_LINES=0, interpreter uv-env CPython
3.12.13 / torch 2.12.0+cpu non-system, `__pycache__` purged and printed 0
before every arm; sbatch md5 e4387880ec87af022a238d1f2f570175 matched both
ends; test-QOS 1 of 5 in use at submit (foreign `storage-report`); nothing
cancelled.

**VOID condition checked FIRST and passed:** in-job
TRAINER_PRISTINE=b44aaa0962f2e4e046294fe50a65e2140b680c64008bd2f3d7283755dbb1417e
(NOT the rounds-3–6 value 6e672382…), TRAIN_PRISTINE=a13eb986… — both
re-derived in-job, neither inherited. Both mutants proven active by sha256
difference (m-A 4d7eff61…, m-RC 3a1551f2…) plus the printed mutated line; both
restored to the exact pristine hash with porcelain 0; FINAL_PORCELAIN=0,
FINAL_HEAD unchanged.

## Arms vs registered predictions — 5/5 matched, raw JUnit attributes

| arm | predicted | measured | match |
|---|---|---|---|
| g-focused (5 binding files) | 25 / 0 fail | tests=25 failures=0 errors=0 skipped=0 | yes |
| g-unit `tests/unit` | 3054 / 0 / skip 5 | tests=3054 failures=0 errors=0 skipped=5 | yes |
| g-int `test_train_runner.py` | 6 / 0 | tests=6 failures=0 errors=0 | yes |
| m-a-focused | 4, by name, split 2 DNR + 2 mismatch | failures=4, exactly those names, exactly that split | yes |
| m-rc-int | 1: the bitwise-resume test via diverged-parameters | failures=1, that name, `['raw_alpha','raw_beta'] == []` fails | yes |

m-a-focused failure messages, read from the JUnit (not inferred):
`test_r1_1_stateless_direct_bind_refuses_a_second_carrier` and
`test_r1_1_control_the_default_adapter_refuses_the_same_sequence` are genuine
`DID NOT RAISE <ValueError>`; `test_a_second_carrier_is_refused_after_binding`
and `test_a_refusal_happens_before_any_update_runs` are regex mismatches with
actual message `'mismatched legacy optimizer ownership'` — the older guard
answering with the shared string, exactly the carried gap #5 mechanism.

The Gloo flake (`test_run_id_rank_agreement.py`) did NOT fire — a null
observation per the pre-registered disposition, not a re-verification.

Elapsed disclosures, not absorbed: env sync 92 s against the announced
~40–90 s; m-rc-int JUnit time 10.9 s against the announced 12–60 s band (the
mutant fails the resume test fast); every other arm inside its band; whole job
11:00 inside 10–20 min.

## What the two commits under review claimed, and what held

**The trainer.py sentence this head rewrote is MEASURED TRUE at this head.**
"At that time this check could have been deleted outright without a single
test failing. That is no longer so" — m-a-focused disables exactly that check
and kills 4 tests, two of them DID-NOT-RAISE on the R1-1 probes that bind
directly and pin the refusal by its own message, which is the comment's precise
claim.

**review_r4.py's new present-tense sentence is MEASURED TRUE at this head.**
"`tests/integration/training/test_train_runner.py` does catch this payload" —
m-rc-int re-measured it at 1132e910 rather than inheriting round 6's result at
f2006235: the witness-free R4-2 payload kills exactly
`test_resume_reproduces_the_uninterrupted_run_bitwise` via the
diverged-parameters assert, same signature round 6 recorded. (The integration
file's blob is byte-identical at head and base — 18fb6d98… at both — so round
6's base-side green covers these bytes.)

**Doc-only, reproduced with MY OWN method, not the writer's.** For each of
review_r1/r2/r4 and trainer.py between f2006235 and 1132e910: (a) AST with all
docstrings stripped identical; (b) token stream minus COMMENT/NL identical for
trainer.py — strictly STRONGER than the writer's changed-lines-are-comments
check, since it also rules out comment edits that merge or split code lines —
and different for the three probe files exactly as docstring edits must be.
Controls: both comparators flag an injected statement; AST passes a synthetic
docstring edit; token passes a synthetic comment edit. The challenge "construct
a change that passes both checks and still alters behaviour" FAILS at this
head, and each escape channel was measured absent rather than assumed: no
`__doc__` read targets a test module (6 reads in the tree, all tpen modules or
argparse), no doctest configuration (`addopts` is typeguard only), no
getsource/linecache/lineno use under the training test trees, and stale
bytecode is excluded by the per-arm purge.

## Issues

**R7-1 (low) — review_r1.py:7-9: the new header's own history claim is false
against the record.** "Six consecutive review rounds found exactly one such
sentence false, each time at this cluster of files" fails in both directions.
Round 6 alone found TWO accepted instances of the stated class (R6-1
file-change claim, R6-2 past-measurement claim) plus an observation — and the
writer's own 1132e910 grep found two more unreported ones, which the commit
message itself says. And rounds 1–4's instances (R1-2, R2-2, R3-1, R4-1) were
live wording or cross-reference defects CHECKABLE against the tree — not
members of the class the sentence defines ("a claim about a past run, or about
how the surrounding file has changed, cannot be checked against the tree").
3eb6cbb1's commit message carries the same mischaracterisation ("Every
instance rounds 1-6 found was the same kind of statement… Neither can be
checked against the tree"); the commit message is a pushed record amended only
by addition, so the fix is to the header sentence. A false historical claim in
the paragraph that abolishes false historical claims is the class
demonstrating it survives its own removal; the honest form is "at least one
per round", or no count at all.

**R7-2 (low) — trainer.py:504-507: the retained mechanism clause is false at
the head it describes.** "Every test that asserted on the shared string
reached it through `resolve_update_state`, where the older check fires first"
— traced at 4175fc17: `test_a_refusal_happens_before_any_update_runs` reached
the string through `fit`, not `resolve_update_state`; and on BOTH paths the
BINDING-BOUNDARY check itself raised first (`bound_state =
self._resolved_update_state` was populated by the first resolve, so its
comparison fired before `_resolve_method_state` was ever called). The older
check never fired first while both existed — it was the identical-string
BACKSTOP that fired only under deletion, which is the actual mechanism that
made round 1's M2 survive. The clause was born at 47989a12, survived rounds
2–6 unflagged, and was preserved verbatim through 1132e910's rewrite of this
exact block. The sentence's conclusion (deletability then, caught now) is
measured true — by M2 then and by my m-a arm now — only the mechanism is
wrong, and a confident wrong mechanism in a comment a maintainer will reason
from is this lane's own named failure mode
(`correction-my-doublebuild-decomposition-was-wrong`).

**R7-3 (low) — review_r3.py:29-35: false present-tense description of sibling
fixtures, false since the commit that adopted it.** "The R1-5/R2-1 capture
fixtures record only the LAST optimizer the factory built, so their identity
assertions are blind to a runner that builds the carrier TWICE … `built` is
overwritten and `seen is built` passes." The SAME commit that adopted this
file (57baf429) strengthened both captures to builds-lists asserting
`builds[0]` — the sentence was false the moment it entered the tree, no
variable `built` exists in either sibling, and rounds 5–6's M-DB arms measured
both tests FAILING under double-build, the exact blindness the sentence
asserts. This is the seventh instance the writer told this round to assume
exists, sitting in the one probe file 3eb6cbb1 did not touch, invisible to the
writer's signature-word grep because it is a structural present-tense claim
rather than a measurement sentence. Secondary site, same file, same fix:
line 3 "These two tests exist to settle M-R2-4" — the file contains ONE test
(the omission is explained six lines later, but the opening sentence reads
false alone). Fix is a retense ("recorded only … were blind"), scoped to the
forms round 3 measured.

## Non-blocking observation

**R7-obs** — the same commit pair that removed measurement history from the
r1/r2 headers WROTE new measurement history into review_r4.py (round 6's
head/base/mutant results, and the present-tense "does catch this payload").
It is accurate today — my m-rc-int re-measured it at this head — and it is
round-anchored, but it is the class 3eb6cbb1's own rationale evicts, and the
present-tense half will go stale with the next change to the payload, the
test, or the file. Writer's call: keep it as the finding's documentation, or
point at the record as the r1/r2 headers now do.

## The six attack points, answered

1. **Information destroyed: NONE.** Every fact the rewrite deleted maps to a
   durable location I verified exists: jobs 51674866/51675986 and the
   per-probe results — `review-round-1-verdict-4175fc17` +
   `review-r1-empirical-arm-corroborated…` + PR comment 6089212913 (and the
   deleted sentence was FALSE as written, per R6-2; the true decomposition
   lives only in the record); base-green/head-red for R2-1 —
   `review-round-2-verdict-47989a12` + PR comment 6090447926; the
   adoption-edit list for r1 — round-2 verdict claim-audit #2 and
   `tip-verification-dd1fa146`'s static diff section; "no receipt runs
   tests/integration" — superseded by round 6's measurement, now in
   `carried-forward-known-gaps-at-1132e910`. The new headers' pointer claim
   (verdict and red/green notes carry heads, job IDs, per-arm counts) is TRUE
   — I checked it against all 48 notes and the 12 PR comments.
2. **A seventh instance exists: R7-3**, plus the meta-instance R7-1 inside the
   new header itself.
3. **The deliberately-left sentence** (review_r1.py:189 "The behaviour
   assertion below is unchanged") **is TRUE at this head** — the diff
   57baf429..1132e910 touches only docstring/header text in that file, the
   assertion body is untouched since adoption. I CONCUR with leaving it; it is
   accurate, and reopening the paragraph for one true sentence is the widening
   the writer rightly refused. It joins the class's watch-list, nothing more.
4. **The docstring-only evidence holds under an independent method**, and the
   trainer.py weakness the writer disclosed is CLOSED by a stronger check that
   also passed (token stream minus comments, identical). No bypassing
   construction exists at this head; every candidate channel measured absent.
5. **Skipping a cluster arm at f2006235→1132e910 was the right call** — with
   the explicit caveat that my concurrence is cheap, since this round ran the
   real arms that bound the residual risk. `ast.parse` on every edited file
   plus docstring/comment-only proof strictly dominates the collection-safety
   inference round 5 spent an arm refuting; nothing in my five arms contradicts
   the choice.
6. **The rewritten trainer.py claim is TRUE at this head**, measured by
   m-a-focused with the exact predicted decomposition (see above).

## Discounts and scope

All-Claude lane (writer/orchestrator Opus 5-family, reviewer Fable 5),
one-child serialization, no second party checked my fixtures or my mutants;
mitigations are the pushed pre-registration, the in-job VOID condition, and
re-derived (never inherited) hashes. Not run: whole-suite `pytest -q` (PR 522
not in ancestry — `merge-base --is-ancestor be3d0a5a 1132e910` false; no arm
spent, per brief), `tests/integration/` beyond the one named file, GPU,
multi-process, `experiments/`. `tip-verification` at this head remains ABSENT
(last PASS `dd1fa146`, five heads stale) — this round is a reviewer's
measurement, not the independent verification gate.

## Gate

BLOCK stands unless the writer disputes the three findings. All three are
doc/comment-only; none requires a cluster to fix; a one-commit disposition
(two retenses and one count correction) plus this round's arms would leave the
next head in the strongest recorded state of this PR. If the writer accepts
and fixes, round 8 reviews the disposition; if the writer disputes R7-2 or
R7-3 on the source trace, the trace is in this file and the predictions file,
both pushed before the arms ran.
