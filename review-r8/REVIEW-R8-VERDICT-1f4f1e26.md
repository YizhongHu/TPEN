# Round-8 review VERDICT — head 1f4f1e267c118ceef58125209dce68a801dea7f0

**3 findings (R8-1 low, R8-2 low, R8-3 low), ALL in the prose/doc class — NO
production-code defect (6th consecutive round), NO test-logic defect (4th
consecutive), nothing that changes what the suite measures. None of the three
is introduced by the commit under review: all pre-date it and survive it. The
commit itself is CLEAN on every axis it was attacked on: its deletions destroy
no fact, its replacements are verified true, and it is behaviour-neutral by
two comparators and five cluster arms that matched the pushed predictions 5/5
by name and mechanism.**

By the writer's pre-registered stopping rule
(`stopping-rule-for-the-prose-defect-class-registered-before-round-7-results`),
findings of exactly this class at round 8 are recorded on a follow-up item
rather than fixed in this PR. The rule binds the writer, not this review;
everything below is reported at full strength regardless. My position on the
rule, since the brief invited it: it is SOUND — three more instances found by
one fresh read is evidence the class is dense and that per-round fixing
converges on luck, not on a condition — with one requirement: the follow-up
item must EXIST and carry these findings by exact location before
`tip-verification` at this head is treated as the slice's completion gate. A
filed finding nobody can find from the filing is not filed
(`correction-the-follow-up-this-lane-kept-citing-did-not-exist` is this lane's
own precedent).

Canonical copies: PR 524 comment (posted with this text) +
`review-r8/REVIEW-R8-VERDICT-1f4f1e26.md` on `claude/review/vmc-if-b-r8-1f4f1e26`.
Predictions pushed BEFORE any job existed, at
`1b1f97f2a6e382fc3b3d55c9705be402bf42bfb7`, with exposure declaration and a
pre-registered VOID condition — checked FIRST in-job, never triggered.

## FACILITY: FASRC-Cannon — two jobs, both reported

- **51817046 FAILED 73:0, 00:00:24** (Submit 2026-10-10T06:29:25 local) — MY
  script defect: `uv sync --locked` without `--extra cpu` resolves this
  project's env WITHOUT torch (torch is optional-dependency-only, extras
  cpu/cu126/cu128/cu130/rocm71). The job's own fail-closed interpreter gate
  caught it; ZERO arms ran, nothing was measured, nothing contaminated. Run
  root preserved: `/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r8-20261010T102855Z/`.
  FILING FAILURE, per the Cannon index's own instruction: the fix was already
  recorded in `per-checkout-uv-sync-locked-works-on-cannon-2026-09-03` (item
  3541fe33), but `00-read-first-index` on the same item — modified AFTER that
  note's creation — does not name its key, so the mandatory chain could not
  deliver it; the note even declares its own unreachability. The note from the
  index that should have caught this: `00-read-first-index` itself
  (environment section). The script defect is mine; the unreachability is not,
  and any lane reaching Cannon through the chain is exposed to the same
  24-second death. Reported to the orchestrator; no index edit attempted (a
  15 KB inline re-emit through an agent's generation retypes rather than
  copies).
- **51817263 COMPLETED 0:0, 00:11:01**, partition `test` requested and
  DELIVERED, node holy8a24101, Submit 06:31:53 / Start 06:31:55 / End 06:42:56
  local. Run root (Active run data, preserve):
  `/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r8-20261010T103129Z/`.
  The two Submit timestamps bracket a single unchanged pre-registration
  (`1b1f97f2`, pushed 10:28Z): the predictions were not touched between
  failure and retry.

Doctrine chain cited: 3541fe33 `00-read-first-index` + mandatory list,
`00-read-first-cannon`, FASRC/Clusters `00-orientation`,
`cluster-access-read-first`. Netscratch quota read with the supported call
before submission: 33.6Ti/50.0Ti (table 06:20:04, read 10:27:54Z). Full clone,
`is-shallow=false`, TAGS=24; in-job HEAD/porcelain/interpreter asserts passed
(torch 2.12.0+cpu, interpreter inside the run-root uv-env, subprocess `python`
resolves to it); pristine hashes RE-DERIVED in-job — trainer.py
`b9b59f8b7…` (the THIS-head value; the circulating `b44aaa09…`/`6e672382…` are
stale), train.py `a13eb986…`; both mutants printed their mutated line and an
in-job-derived mutated sha256 (m-A `980185a6…`, m-RC `332f1eae…` — m-RC
differs from round 7's `3a1551f2…` because the byte-level construction
differs; the semantic payload and the kill are identical); both restored to
pristine by hash; FINAL_PORCELAIN=0. Script md5 identical both ends
(`4dda64ae9c2cb7edeaccdf0c3e965ac7`).

## 5/5 arms matched the registered predictions by NAME and MECHANISM

- **g-focused** 25/0/0/0, 35 s (collected 25, independently cross-checked).
- **g-unit** 3054/0/0/skip5, 453 s. The Gloo flake did NOT fire — null
  observation; pre-registered disposition unused.
- **g-int** `test_train_runner.py` 6/0/0, 17 s.
- **m-a-focused** fails EXACTLY 4 of 25, split exactly 2/2 as registered:
  `test_r1_1_stateless_direct_bind_refuses_a_second_carrier` and
  `test_r1_1_control_the_default_adapter_refuses_the_same_sequence` genuine
  DID-NOT-RAISE; `test_a_second_carrier_is_refused_after_binding` and
  `test_a_refusal_happens_before_any_update_runs` regex-mismatch (the older
  guard answers with 'mismatched legacy optimizer ownership' — carried gap
  #5's mechanism). All named survivors survived. This MEASURES TRUE the one
  live claim the commit left at the check in trainer.py: "The distinct message
  is what lets the review probes pin THIS check by its own text, and disabling
  it is what makes them fail."
- **m-rc-int** fails EXACTLY
  `test_resume_reproduces_the_uninterrupted_run_bitwise` via
  `['raw_alpha','raw_beta'] == []` — so the pointer that replaced
  review_r4.py's measurement sentence points at a record whose property still
  holds at THIS head, re-measured rather than inherited.

Elapsed disclosures: every arm inside its announced band (sync 87 s of 40–95;
whole job 11:01 of 10–20 min). No arm rerun. Exposure: every count above was a
number I had seen before measuring (declared in the predictions file); counts
are reported as raw JUnit `<testsuite>` attributes, the same decomposition
every prior round used, so agreement cannot hide in a changed form.

## The brief's four questions, answered

**(1) Does any deletion remove a fact now reachable from nowhere? NO —
checked independently.** review_r4.py's deleted round-6 integration
measurement (green at head AND stack base, red under M-RC, failing test and
diverged list) lives in `review-round-6-verdict-f2006235`, PR comment
6096119065, and `carried-forward-known-gaps-at-1132e910` #1 — which carries
MORE than the docstring did (job 51808607, blob
`18fb6d98916f9ecba3e45e058dea6287323a151a`). review_r3.py's deleted head SHA
`dd1fa146…` lives in PR comment 6091362333 and three dd1fa146-keyed notes.
trainer.py's deleted paragraph: its true content (deletable-then; M2 survival)
lives in the round-2 and round-7 verdicts and in 1f4f1e26's own commit
message; its false content (the ordering mechanism) is preserved AS A FINDING
in R7-2, which is the right place for it. review_r1.py's deleted count claim
was false (R7-1) — deleting a false claim destroys no evidence — and the
deleted "structural cause" analysis survives, stronger, in the stopping-rule
note.

**(2) Is there an eighth instance? YES — three, all verified by my own
reading of the tree and of `git show 4175fc17:tpen/training/trainer.py`, and
all invisible to a signature grep (R8-2's key phrase even wraps across a line
break, so a literal grep for it misses; the search that worked was reading
every prose claim about a sibling file or a past mechanism and checking it
against the named tree).** Locations exact, for a reader with none of this
context:

- **R8-1 (low)** — `tests/unit/training/test_update_binding_review_r3.py`,
  function docstring of
  `test_r3_the_runner_builds_exactly_one_carrier_for_a_foreign_trainer`,
  lines 79–81, the parenthetical "(the case the last-build capture in
  R1-5/R2-1 cannot see)". FALSE at this head and false SINCE BIRTH: both
  captures are builds-LISTS asserting `builds[0]` (review_r1.py:425–436,454;
  review_r2.py:193–203,222), and the commit that introduced this file —
  **57baf429** — is the SAME commit that converted both captures
  (`git log -S'builds.append'` names only it). Same substance as R7-3 one
  scope level down in the same file; survived round 7's deliberately precise
  fix, which retensed the module header only. Sharpened by this commit: the
  new header sentence "this header does not describe them" is literally true
  of the header while this function docstring below still describes them,
  falsely.
- **R8-2 (low)** — `tests/unit/training/test_update_binding.py`, docstring of
  `test_a_second_carrier_is_refused_after_binding`, lines 565–568: "the older
  check fires first on this path, so the binding check could be deleted
  outright and no test noticed". The MECHANISM is false at 4175fc17, the head
  it describes: the binding check fired FIRST on every live path (via
  `_resolved_update_state` after a resolve; via `bound.update_state()` on a
  direct bind — `LegacyAutogradUpdate.update_state()` returns its owned
  state); the older check in `_resolve_method_state` was the identical-string
  BACKSTOP that answered only UNDER deletion. The conclusion (M2 survived
  deletion) is true. This is R7-2's exact defect, surviving in the writer's
  own suite file, born **47989a12** — the same commit as the trainer.py clause
  R7-2 killed.
- **R8-3 (low)** — `tests/unit/training/test_update_binding_review_r1.py`,
  docstring of `test_r1_1_control_the_default_adapter_refuses_the_same_sequence`,
  lines 146–148: "Before the fix this control passed through
  `_resolve_method_state`'s older 'mismatched legacy optimizer ownership'
  check". FALSE under either reading: the control's bind→bind sequence never
  reaches `_resolve_method_state` (nothing in `bind_update_method` calls it);
  at 4175fc17 the refusal came from the binding boundary's own state-derived
  carrier check. The shared-string halves of the sentence are true. Born
  **47989a12**.

Provenance matters because it splits the three into two authoring events with
different characters: 47989a12 (R8-2, R8-3) propagated a mechanism claim that
was plausible when written and was only refuted by round-7's analysis;
57baf429 (R8-1, and R7-3 before it) wrote a claim its own diff contradicted
the same day. A remedy aimed at one event will miss the other. (I initially
mis-grouped R8-1 under 47989a12; the orchestrator's blame and mine agree it is
57baf429 — correction adopted from measurement, recorded here.)

**(3) Did the commit introduce anything new? NO false claim found.** Checked
every replacement, not just the deletions: review_r1.py's general claim and
retained pointer (verdict/red-green notes carry heads, job IDs, per-arm counts
— spot-verified against rounds 6–7); review_r3.py's hypothetical shape
(matches the test body) and train.py quote (verbatim at train.py:73–74);
review_r4.py's pointer (the record holds, and m-rc-int re-measured the
property at this head); trainer.py's surviving mechanism sentence (measured
true by m-a, above). One sharp edge, filed under R8-1: the new "this header
does not describe them" sentence is header-scoped-true while the file still
describes siblings falsely below.

**(4) Behaviour-neutrality, MY method.** Independent comparator pair on `git
show` extracts at 1132e910 vs 1f4f1e26: AST-with-docstrings-stripped
(docstring→`pass`, never deleted) IDENTICAL for all four files;
token-stream-minus-COMMENT/NL IDENTICAL for trainer.py and DIFFERENT for the
three probe files exactly as docstring edits must be. Controls both
directions: injected statement caught by BOTH; synthetic comment-only edit
passes BOTH (the declared blind spot, which is what makes trainer.py's
pass/pass honest); synthetic docstring edit passes AST, caught by tokens;
non-trivial inputs (313–2554 tokens). Bypass construction attempted and
failed: a change passing both comparators is confined to comments/blank
lines, one passing only AST to docstrings; every consumption channel measured
ABSENT at this head — no doctest config (`addopts` is typeguard-only), no
`__doc__` read of any touched module, no
getsource/getdoc/linecache/`.lineno`/`co_firstlineno` under the touched test
trees or the two production files, no encoding-declaration comments, per-arm
`__pycache__` purge printed 0 in-job. The five arms are the empirical
confirmation: identical counts everywhere, mutants red exactly as at prior
heads.

## Also verified true (so the next reader need not re-fear them)

review_r2.py's two checkable pointer claims ("R3-2" absent from the r3 file —
count 0; present in `test_update_observations.py`:1450,1479);
review_r4.py's enumeration (all three named exactness tests call
`restore_checkpoint` directly); review_r4.py's block-NG claim (SGD at
`test_block_ng_trainer_integration.py:49`, runner-driven bitwise resume);
review_r1.py's R1-2 "one of the two strings is still present" (exactly one
verbatim string survives at the check; the other appears only as a
non-greppable paraphrase); train.py's `_load_trainer` reference exists
(checkpoint/restore.py:348); carried gap #5's shared string still shared
(trainer.py:393 vs :561).

## Discounts and gate

All-Claude lane, one-child serialization, no second party checked my
comparators; mitigations: pushed pre-registration, in-job VOID condition,
hashes re-derived in-job, and the attempt-1 failure handled fail-closed with
both jobs reported. Not run: whole-suite `pytest -q` (PR 522 not in ancestry —
`git merge-base --is-ancestor be3d0a5a 1f4f1e26` is false, verified),
tests/integration beyond the one file, GPU, multi-process, experiments/.
`tip-verification` at this head remains ABSENT — seven heads now — and stays
the single largest open item on the slice; this round is a reviewer's
measurement, not that gate.

**Disposition this verdict asks for: per the stopping rule — record R8-1/2/3
by the exact locations above on the follow-up item, do NOT fix them in this
PR, and send the independent verifier at THIS head, which five green/red-true
arms and two comparators say is safe to verify.**
