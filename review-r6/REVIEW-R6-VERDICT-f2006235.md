# Round-6 review VERDICT — head f2006235a81df8655fe723fe6f9309b4afd714b2

2026-10-10, reviewer fable-5, worktree vmc-if-b-review-r6, operating under the
one-claude-child-per-lane ruling: probes designed, implemented and run by the
reviewer itself, no second party checked the fixtures — discount accordingly.
Dispatched by writer `15ea70cc`, completed under adopting orchestrator
`e069d455` after a session-limit interruption mid-read (no deliverable
existed from the interrupted turn; everything below was produced after it).

**VERDICT: BLOCK — 2 issues (R6-1 low, R6-2 low) + 1 non-blocking
observation. NO production-code defect found (fourth consecutive round), NO
test-logic defect found (second consecutive round). Both issues are
doc/provenance-strength, in the module headers of adopted probe files; one
commit disposes of both.** Accept/dispose is the writer's call.

The writer's brief said to assume a sixth wording defect exists at the
`test_update_binding*` cluster. It does, and a seventh. Both were found by
independent greps and measured against git history and the PR thread, not by
reproducing the writer's searches.

CANONICAL COPY: PR 524 comment + this file on
`claude/review/vmc-if-b-r6-f2006235`. Predictions committed and PUSHED before
submission at `f8ea2c06b66928e339b3e6cfbb3747a7438b4726`
(`review-r6/PREDICTIONS-R6-f2006235.md`), including an exposure declaration of
every writer/reviewer number seen before measuring, per-arm failing names AND
mechanisms, named survivors, and pre-registered dispositions — one of which
(the Gloo flake) was needed.

## Evidence

FACILITY: FASRC-Cannon. Job **51808607**, partition `test` requested and
DELIVERED, node holy8a24101, 4 CPU / 32 GiB, **COMPLETED 0:0, 00:21:42**.
Doctrine chain read in full this session: `3541fe33` `00-read-first-index` +
mandatory list, `00-read-first-cannon`, FASRC/Clusters orientations,
`cluster-access-read-first`. QOS headroom 1/5 before submit (not mine; nothing
cancelled, ever); submit-time stderr captured (empty). Run root (Active run
data, preserve, never delete):
`/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r6-20261010T090006Z/` —
own full clone (IS_SHALLOW=false, TAG_COUNT=24), head checkout at the exact
SHA plus a base worktree at `dbb486ea`, own
`UV_PROJECT_ENVIRONMENT`(×2)/`UV_CACHE_DIR`/`TMPDIR` outside the checkouts,
`apply_mutant_r6.py` (md5 `b9d86edd…`, hash-matched both ends) and
`job-r6.sbatch` (md5 `9e4599e5…`, matched), job-scoped
`junit/<arm>-51808607.xml` and `logs/<arm>-51808607.log`.

In-job, per arm: HEAD asserted at 40 chars in BOTH checkouts, tracked
porcelain 0, `__pycache__` purged and printed 0 before EVERY arm in the
checkout that arm uses, interpreter job-local uv-env CPython 3.12.13 / torch
2.12.0+cpu asserted >=3.10 and non-system (both envs), counts from raw JUnit
`<testsuite>` attributes with errors separate from failures.

MUTANT IDENTITY, the strongest form this lane has used: every mutant
content-asserted the exact original line before writing, AND its mutated-file
sha256 was asserted equal to round 5's recorded value — all three matched
(`M-RC` → `332f1eae…`, `M-DB` → `f36ba3fd…`, `M-A` → `f19f7227…`), so the
payloads are byte-identical to the ones rounds 4-5 measured, not
reconstructions that merely reproduce a number. Pristine
`train.py`/`trainer.py` sha256 asserted equal to rounds 3-5's lineage values
before any arm; every restore re-asserted pristine sha256 with porcelain 0;
FINAL_PORCELAIN=0 in both checkouts.

## Prediction vs result — 8 of 9 arms matched by name AND mechanism; the one
## miss resolved by a pre-registered disposition

| arm | predicted | measured | match |
|---|---|---|---|
| g-focused | 25/0 | tests=25 failures=0 errors=0 | yes |
| g-full | 3054/0/skip 5 | tests=3054 **failures=1** skip=5 | **NO — the disclosed flake, below** |
| g-int-head | 6/0 | tests=6 failures=0 | yes |
| g-int-base | 6/0 | tests=6 failures=0 | yes |
| m-rc-focused | 1: the r4_2 pin via EQUALITY | exactly that, "runner-driven resume diverged" | yes |
| m-rc-full | 1, same name, tests=3054 | exactly that | yes |
| m-rc-int | 1: the bitwise resume test via the diverged-parameters assert | exactly that: `assert ['raw_alpha', 'raw_beta'] == []` | yes |
| m-db-focused | 6, by name; r4_2 via REFUSAL not equality | failures=6, those names; refusal string ×9 in JUnit, equality string ×0 | yes |
| m-a-focused | 4: two DID-NOT-RAISE + two regex-mismatch on the old string | exactly those, verbatim | yes |

### The g-full miss, decomposed
The single failure is
`tests/unit/test_run_id_rank_agreement.py::test_a_contradicting_launcher_is_refused_even_with_an_explicit_id`,
message `AssertionError: a rank wrote no receipt` — the EXACT signature of the
pre-existing file-scoped Gloo flake carried as gap #3, in the file the gap
names, with the per-rank logs preserved under the run root's `tmp/`. My
predictions file pre-registered precisely this disposition before the run:
report as flake-fired with the count decomposed, do not rerun to green. Two
facts that keep the decomposition honest rather than convenient: (1) the
m-rc-full arm ran the same file minutes later and it PASSED — intermittence,
consistent with a flake and not with a head defect; (2) the flake had NOT
fired in any arm since the gap was written, so this round converts gap #3
from a null observation into a live, confirmed one. The writer's "3054/0" at
this head is bounded by this flake exactly as its own disclosure said.

### THE ROUND'S CENTRAL NEW MEASUREMENT — carried gap #1's untested inference
## is now TRUE, measured
Round 4 claimed
`tests/integration/training/test_train_runner.py::test_resume_reproduces_the_uninterrupted_run_bitwise`
"would have caught" the M4/R4-2 payload, and NO receipt in this PR had ever
run any integration test. Measured this round, all three legs: the file is
GREEN at the head (6/0 — the first integration receipt in this PR at any
SHA), GREEN at the stack base `dbb486ea` (6/0, both-sides control), and under
the byte-identical witness-free M4 payload (M-RC) it fails EXACTLY the
predicted test through EXACTLY the predicted assertion — the
`_diverged_parameters(final_a, final_b) == []` bitwise compare at
`test_train_runner.py:332`, reporting `['raw_alpha', 'raw_beta']` diverged,
with the five predicted survivors (including the restored-identity test,
which truthfully reports a successful restore while the arithmetic is wrong —
identity fields are not arithmetic) all surviving. The integration arms also
had ZERO prior numbers anywhere in the record: no anchoring was possible.
Gap #1 is not closed — the rest of `tests/integration/` remains unrun — but
its sharpest edge (an inference standing in for a measurement) is gone, and
this PR now has runner-resume coverage receipts at BOTH suite levels.

## Issues

### R6-1 (low) — the r2 probe header's provenance claim is falsified by two
### later commits, one of them a CODE change

`tests/unit/training/test_update_binding_review_r2.py:29-32`: "becomes the
permanent pin against the alias being reintroduced -- **unmodified except for
this paragraph**." Measured against git history: at `57baf429` the probe's
CODE changed — the single-slot `built` dict became the `builds` list, a new
`assert builds` non-vacuity guard was added, and the identity assert moved to
`builds[0]` (the R3-2 strengthening); at `f2006235` the `build_adam` comment
was rewritten again (the R5-3 fix). The header therefore attributes round 2's
base-GREEN/head-RED measurement to a probe body that no longer exists: the
form that was measured on both sides of the slice is the single-slot one, and
the strengthened form now in the tree has never been run at the base SHA
(statically it would also be green there, but that is an inference, not the
measurement the header claims). Same family as R1-2/R2-2/R3-1/R4-1/R5-2:
the site was strengthened and its self-description three paragraphs up was
not. SECONDARY SITE, writer's call: `test_update_binding_review_r1.py:3,18`
("unchanged except where a disposition required it", "R1-2, R1-3 and R1-4 are
untouched") — defensible as adoption-scoped, but R1-2 has since been renamed
(R3-1) and R1-5's fixture strengthened (R3-2), so a reader comparing against
the reviewer's original artefact is misled by the present tense. FIX: one
sentence each — name the later edits (57baf429 strengthening, f2006235
comment rewrite) or drop the freshness clause in favour of pointing at git
history.

### R6-2 (low) — the r1 probe header attributes "all five GREEN" to a job
### that measured failures=1

`tests/unit/training/test_update_binding_review_r1.py:8-9`: "Original state:
all five GREEN at head 4175fc17…, measured on FASRC-Cannon jobs 51674866 and
51675986." Round 1's canonical table (PR comment 6089212913): job 51674866
measured tests=5 **failures=1** — the reviewer's own R1-4 fixture defect,
"fixed file-locally, rerun green" — and only job 51675986 measured the
committed file's five tests all green. So the committed probes were never run
in 51674866 at all; a different (pre-fix) version was, and it was red. Same
dropped-qualifier class as R5-2 ("every test passed" vs failures=1), one file
over. FIX: one line — attribute all-five-green to 51675986, and 51674866 as
the run whose single failure was the probe file's own pre-fix fixture.

### R6-obs (non-blocking, writer's call) — the r4 header's "Every exactness
### resume test in the focused files … calls `restore_checkpoint` DIRECTLY"
`test_update_binding_review_r4.py:21-24`. At this head the same file contains
`test_r4_2_*`, an exactness resume test that goes through `Train.run`. The
sentence sits under "Why the existing coverage missed it, from the reviewer's
analysis" and closes with an explicit three-name enumeration, so the
historical scope is recoverable — but the universal quantifier is now false
read against the tree, which is the same trap shape as the R4-nit. A
two-word edit ("Every PRE-EXISTING exactness resume test") retires it.

## Writer-claim audit — the five claims named for attack

1. **The three round-5 text fixes are accurate, not merely different —
   CONFIRMED against the primary records, not the writer's summaries.**
   R5-2's replacement text was checked against round 4's FINAL verdict
   (comment 6092350102): M4-full measured 3053 tests / failures=1, sole
   failure `test_durable_append…append_handle` census catching the reviewer's
   own `open(..., "a")` witness; round 4 did carry the qualifier; "no test
   detected the PAYLOAD" is round 4's own substantive claim. All four facts in
   the new docstring match. R5-3's replacement: verified "R3-2" occurs
   nowhere in the r3 probe file, and a tree grep does land on the unrelated
   lane's identically-numbered finding in `test_update_observations.py`
   (plus this lane's own legitimate round-reference at `review_r1.py:429`,
   which names the finding rather than pointing a grep at a file — not a
   defect). R5-4's residue removal verified; the surviving pointer names the
   adopted R1-2 probe by role and file, and that probe exists and pins the
   claimed raise.
2. **No OTHER overstated/broken pointer survives — REFUTED, twice: R6-1 and
   R6-2 above.** Both found by independent sweep (quoted-string greps against
   production source, claim-by-claim comparison of module headers against git
   history and the PR thread's measured tables).
3. **Skipping mutant arms at this head on the sha256 identity argument —
   CHALLENGED, then CONFIRMED BY MEASUREMENT.** The argument as stated was
   incomplete: mutant results are a joint function of the production AND test
   trees, and the docs-only commit changed test-module `__doc__` constants,
   which the file-level sha assertion on two production files cannot see. The
   missing half holds — the only `__doc__` assertions in tests/unit target
   `tpen` modules, and comments compile to nothing — and the whole-tree
   production identity follows from the git diff touching only `tests/`. I
   then ran the three decisive mutants anyway, byte-pinned to round 5's
   payloads: M-RC (focused 1 / full 1 / integration 1), M-DB (6), M-A (4) —
   every count and every mechanism carried over exactly. The skip hid
   nothing.
4. **`correction-my-doublebuild-decomposition-was-wrong` states the corrected
   mechanism accurately — CONFIRMED, re-measured on an independent harness.**
   I registered agreement with the corrected account before running
   (predictions §1) and derived it from source rather than from the note.
   Measured: under M-DB the r4_2 pin fails with `ValueError: update method is
   already bound to a different optimizer` raised from the straight arm's
   `fit`; the refusal string appears 9 times in the m-db JUnit and the
   equality-assert string ZERO times. The note's narrower lesson — a test
   fires through whichever guard it reaches first — is exactly what the
   measurement shows, and its claim that the equality channel is load-bearing
   under M-RC is confirmed by both m-rc arms failing through the equality
   assert alone.
5. **Nothing breaks direct-fit, explicit-factory, bound-instance, stateless
   usage, or the checkpoint/resume contracts — CONFIRMED at the strongest
   level any round has had:** focused 25/0; full suite 3054 with the sole
   failure the disclosed pre-existing flake in a file this PR does not touch;
   and the runner-resume contract now measured at the integration level,
   green at head AND base with the discriminating mutant red. Static trace of
   `bind_update_method`'s already-bound path (model `is`, retained-carrier
   `is`, effective-spec conflict with the bound-instance escape) found no
   uncovered edge; the bind-then-restore-without-resolve edge fails closed at
   `rebuild_update_state`'s RuntimeError.

## Scope calls and carried-gap dispositions

Shared "mismatched legacy optimizer ownership" string between
`_resolve_method_state` and `rebuild_update_state`: AGREE it stays a
follow-up — rounds 3-5's grounds unchanged, and my m-a arm shows the
binding-boundary message distinctness doing its job meanwhile. All eight
carried dispositions at `carried-forward-known-gaps-at-cbf5fcd4`: CONCUR,
with two amendments this round's evidence compels. Gap #1: no longer "run by
NO receipt" — `test_train_runner.py` (one file, not the tree) now has green
head, green base, and mutant-red receipts; the note should be amended rather
than left overtaking the record. Gap #3: the flake FIRED (first time since
the gap was carried), same signature, different test in the same file —
"file-scoped" is now measured, not inferred; per-rank logs preserved under my
run root.

## Elapsed disclosures, and my own misses stated plainly

g-focused 34 s, m-db 15 s, m-a 35 s, m-rc-focused 33 s — inside the announced
15-70 s band. g-full 472 s and m-rc-full 456 s — inside 420-700 s. The three
integration arms ran 15 s / 30 s / 14 s against a pre-announced 60-900 s
band: UNDER-band ×3. Decomposed: I padded a first-measurement band upward for
MCMC sampling cost and overestimated it; the shared module-scoped fixture and
the tiny 6-step config make the file cheap. Reported, not absorbed — a band
that cannot be missed low is not a control. And the g-full prediction itself
was wrong (0 predicted, 1 measured): the pre-registered flake disposition
covered it, but the registration standard this lane runs on counts it as a
miss either way.

## Discounts, stated so the reader can apply them

Writer, all prior reviewers and I are Claude (the writer is Opus 5, I am
Fable 5; model diversity only — provider independence absent since codex went
down fleet-wide 2026-10-09). One-child ruling: no second party checked my
fixtures; mitigations are the pushed pre-registration (one prediction missed
and decomposed, three bands missed low and reported), byte-pinned mutant
payloads verified against round 5's recorded hashes rather than
reconstructions, and a base-side control on the only never-before-run arm.

## Gate status

Review: BLOCK with two accepted-or-disputed-by-writer doc findings — a
further round is required under the standing rule unless the writer disputes
both and the dispute stands. Verification: `tip-verification` at this head
remains ABSENT — this round's arms are a reviewer's measurements, not the
independent clean verification CLAUDE.md requires; the last tip-verification
PASS (`dd1fa146`) is four heads stale. Whoever runs it should note the
integration file is now cheap (≤30 s) and worth including.
