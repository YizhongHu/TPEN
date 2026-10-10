# Round-3 review VERDICT — head dd1fa1464b38a71782e4dea29408a92ea89538f8

2026-10-09/10, reviewer fable-5, worktree vmc-if-b-review-r3, operating under the
one-claude-child-per-lane ruling: probes designed, implemented and run by the
reviewer itself, no second party checked the fixtures — discount accordingly.

**VERDICT: BLOCK — 2 issues (R3-1 low, R3-2 low) + 1 nit + 2 non-blocking
observations. NO production-code defect found.** The production diff held under
attack: the lifecycle fix, the retained-carrier guard, and the alias removal all
survived static tracing and two new mutants. Both issues are test/doc-strength;
one commit disposes of both. Accept/dispose is the writer's call.

CANONICAL COPY: PR 524 comment (Task Orchestrator was DOWN at filing time —
ECONNRESET on every call, independently observed by this session's own MCP
connection, not only reported; this file and the PR comment are the durable
record, a TO note was attempted and its outcome recorded in the PR comment).

## Evidence

Probes: `tests/unit/training/test_update_binding_review_r3.py` on
`origin/claude/review/vmc-if-b-r3-dd1fa146` @ `5d2b569e8df313cdf774f837ecbe47200582a8f1`
(parent dd1fa146; diff touches ONLY the probe file; md5
`211b558022e27b1807fe2f021b549c1d`, hash-matched local vs Cannon). Two tests:
a weakened-isinstance INSTRUMENT (the pre-R2-4 form of the R1-5 assertion) and
an adoptable build-count pin.

FACILITY: FASRC-Cannon, partition `test` requested AND DELIVERED both jobs,
node holy8a24101, 4 CPU / 32 GiB. Doctrine chain read in full this session:
item `3541fe33` `00-read-first-index` + mandatory list, `00-read-first-cannon`,
FASRC/Clusters orientations, `cluster-access-read-first`. Root
`/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r3-20261009T234003Z/`
(run data, preserve): own full clone (`is-shallow` false, TAG_COUNT=24), own
`UV_PROJECT_ENVIRONMENT`/`UV_CACHE_DIR`/`TMPDIR` outside the checkout. In-job
per arm: HEAD asserted at 40 chars, porcelain 0, `__pycache__` purged and
printed 0, interpreter job-local uv-env CPython 3.12.13 / torch 2.12.0+cpu
asserted >=3.10 and non-system, scripts md5-matched both ends, counts from raw
JUnit `<testsuite>` attributes, artefacts job-scoped: `junit/<arm>-<jobid>.xml`
AND `logs/<arm>-<jobid>.log`. QOS headroom via `squeue -u $USER -h -p test |
wc -l` (1 of 5 in use throughout, not mine; nothing cancelled, ever).
Pristine `tpen/runner/train.py` sha256 `a13eb986…` matches the writer's own
recorded pristine hash. Both mutants proven ACTIVE by sha256 difference +
printed mutated lines; both restored to pristine sha256 with porcelain 0.

| arm | job | state | tests | fail | predicted |
|---|---|---|---|---|---|
| green-focused (4 files) | 51717669 | COMPLETED 0:0 | 25 | 0 | 0 ✓ |
| green-full tests/unit | 51717669 | COMPLETED 0:0 | 3054 | 0 (skip 5) | 0 ✓ (3052+2 exact) |
| MUT-BYPASS focused | 51720341 | COMPLETED 0:0 | 25 | 3 | 3 ✓, by name |
| MUT-BYPASS full | 51720341 | COMPLETED 0:0 | 3054 | 5 | 3 ✗ — 2 extra, resolved below |
| MUT-DOUBLEBUILD focused | 51720341 | COMPLETED 0:0 | 25 | 3 | 3 ✓, by name |

All predictions were REGISTERED BEFORE RESULTS, including the prediction that
the reviewer's own target pins would FAIL to catch MUT-DOUBLEBUILD.

HARNESS DEFECT, disclosed: job 51717669 died FAILED 1:0 after its two green
arms — the mutant-application heredoc called BARE `python`, which is not on
PATH on a compute node (the documented exit-127 family;
`uv-not-on-path-exit-127-is-script-defect-2026-08-15` names exactly this and I
tripped it anyway). My own `mutant inactive` sha-guard caught it before any
false arm ran. v2 (51720341) routes the heredocs through uv; the green arms
were not re-run — their job-scoped artefacts from 51717669 stand.

## M-R2-4, the handed question — ANSWERED

The mutants:

- **MUT-BYPASS**: `Train.run` line 75 replaced by direct
  `torch.optim.Adam(self.model.parameters(), lr=0.01)` — same type, same lr as
  the test fixtures' configured factory, different instance, factory never
  invoked.
- **MUT-DOUBLEBUILD**: the `fit` call's `optimizer=optimizer` replaced by a
  second `make_optimizer(self.optimizer, self.model.parameters())` — fit fed a
  second factory build.

Measured, at head+probes:

1. **The R2-4 strengthening is REAL, not decorative.** Under MUT-BYPASS the
   weakened-isinstance instrument (the pre-R2-4 assertion) PASSES — the
   slip-through — while the strengthened identity pin FAILS. A
   same-type-same-hyperparameter-different-instance carrier is exactly what
   `isinstance` cannot see and identity can.
2. **But both identity pins have a measured blind spot** (R3-2 below): under
   MUT-DOUBLEBUILD, `test_r1_5_*` AND `test_r2_1_*` both stay GREEN. Their
   capture fixtures record only the LAST factory build (`built["optimizer"]`
   is overwritten), so `seen is built` is satisfied by the WRONG build. The
   suite catches MUT-DOUBLEBUILD anyway — but only via the VMC binding
   boundary (`test_the_default_adapter_is_constructed_once_per_runner_driven_run`
   and `test_r1_4_*`, both failing on the distinct "already bound to a
   different optimizer" refusal), which does not exist on the foreign path the
   R1-5/R2-1 pins govern.
3. **Suite-level, the identity pins carry nearly all of the bypass coverage.**
   MUT-BYPASS full tests/unit: 5 failures = the 3 identity/count pins + 1
   genuine extra catch (`test_block_ng_resume_after_sampler_draw_is_bitwise`,
   which refuses because Block-NG REQUIRES `torch.optim.SGD` — a type-level
   accident, not identity coverage) + 1 flake-family failure (below, not a
   catch). Nothing else in tests/unit notices the runner ignoring its
   configured optimizer spec.

So: carry R2-4 as a real improvement, narrow; and the blind spot is R3-2.

## Issues

### R3-1 (low) — the adopted R1-2 probe asserts, in present tense, a doc contradiction this head FIXED

`tests/unit/training/test_update_binding_review_r1.py`:

- Test name `test_r1_2_explicit_none_after_an_explicit_override_raises_despite_the_doc`:
  at dd1fa146 the raise is DOCUMENTED — `bind_update_method`'s Raises section
  says explicitly that an explicit `None` "re-reads the CONSTRUCTOR's spec, so
  a trainer configured with one method and bound from an explicit override
  raises here on a plain call". The raise is now *per* the doc, not *despite*
  it. The name travels alone in pytest/JUnit output and asserts a live
  contradiction that does not exist.
- Its docstring quotes the refuted wording in present tense: "`bind_update_method`'s
  Parameters doc: ``None`` 'never [conflicts]'. Its inline comment: explicit
  ``None`` 'means use what is already bound'." Neither string exists at this
  head. Closing line "then the documentation is wrong twice" is a present-tense
  false claim about the current tree.
- The module header's R1-2 paragraph repeats the same present-tense account.

FAILURE SCENARIO: a reader at this head (a round-4 reviewer, or a maintainer
grepping the raise message) reads the pinning test and concludes
`bind_update_method`'s current docs are wrong, re-files R1-2, or "fixes" docs
that are correct. This is the exact class as R2-2, which the writer accepted
one round ago: refuted/stale wording surviving in the test meant to pin the
behaviour. The header's "R1-2 … untouched" provenance note does not travel
with the test name. FIX: rename (e.g. `…_raises_as_the_corrected_doc_states`)
and re-scope docstring + header paragraph to past tense. Behaviour assertion
unchanged; zero code.

### R3-2 (low) — the R1-5/R2-1 capture fixtures pin the LAST build, so their identity assertions are blind to a second factory build reaching `fit`

Demonstrated: MUT-DOUBLEBUILD, both pins GREEN (prediction registered before
the run). The capture `built["optimizer"] = …` in `build_adam` is overwritten
by the mutant's second invocation, so `seen is built["optimizer"]` compares
`fit`'s optimizer against itself. On the foreign-trainer path there is no VMC
binding boundary to refuse the second carrier, so within those two tests the
runner's own claim at `train.py:73` — "ONE carrier is constructed for this
run, here, and nothing below builds a second" — is unpinned in the direction
a duck-typed consumer would be hurt by (restore and fit silently handed a
carrier with fresh state).

FAILURE SCENARIO: a future edit reintroduces a second `make_optimizer` call on
any path (e.g. "defensively" rebuilding at the fit call); R1-5 and R2-1 stay
green; the regression is caught only by VMC-path refusals, and a reader of the
foreign-path pins wrongly concludes the foreign path was guarded.

ADOPTABLE FIX, measured:
`test_r3_the_runner_builds_exactly_one_carrier_for_a_foreign_trainer` (on my
branch) captures EVERY build in a list and asserts `len(builds) == 1` and
`seen is builds[0]`. GREEN at head; RED under BOTH mutants (count=0 for
BYPASS, count=2 for DOUBLEBUILD). Adopt it (or fold the list-capture into the
two existing pins). If adopting my probe file, either drop or keep-with-verdict
the weakened-isinstance INSTRUMENT test per its own docstring — it is a
measuring stick, not a pin, and should not outlive its measurement unlabelled.

### R3-3 (nit, non-blocking) — doc residue in `bind_update_method`

- Parameters doc says the explicit-`None` semantics are "deliberate (see
  Notes)", but the Notes section never addresses explicit-`None`; the
  rationale lives in the inline body comment (the R1-2 paragraph). Dangling
  internal pointer at the exact spot with a two-round doc history.
- Cosmetic: `spec` computed at the top of `bind_update_method` is unused on
  the already-bound branch; `late_spec` recomputes the identical expression 56
  lines later. One of the two can go.

## Non-blocking observations

- **The rank-agreement flake is FILE-scoped, not test-scoped.** In
  MUT-BYPASS-full, `test_a_contradicting_launcher_is_refused_even_with_an_explicit_id`
  — a DIFFERENT test in `tests/unit/test_run_id_rank_agreement.py` than the
  disclosed flake — failed once with the IDENTICAL signature
  (`AssertionError: a rank wrote no receipt`, watchdog not fired, peer rank's
  receipt carrying the EXPECTED refusal text). Basis for calling it the same
  pre-existing Gloo family and not a mutant catch: the refusal under test
  fires at launcher-topology validation, before any optimizer is constructed;
  the signature matches `gloo-peer-close-before-rank-receipt-2026-10-07`
  (filed two days before this branch existed); the same test passed in this
  session's own green-full and in every prior full-suite run at this head.
  Residual, stated: I did not re-run it in isolation under the mutant, so
  "mutant-independent" is supported, not proven. CONSEQUENCE for the
  disclosure the writer is carrying: broaden it from one named test to the
  file's multi-rank receipt mechanism.
- **Writer's flake disposition: CONCUR, verified independently.** I read
  `tip-verification-dbb486ea` (created 2026-10-09T19:21Z, "no Gloo flake on
  this run") and the 2026-10-07 Gloo note myself; both predate this branch.
  Pre-existing, not this slice's, disclose-not-chase is right.

## Writer-claim audit (the five claims named for attack)

1. **`_bound_optimizer` lifecycle hazard — NO HAZARD FOUND (second look).**
   Set once at first bind; never reset (correct: new run = fresh trainer, and
   the refusal message says so); not serialized (identity is meaningless
   cross-process; `state_dict`/`load_state_dict` never touch it); checkpoint
   restore mutates the SAME optimizer object in place and never swaps the
   carrier, so the identity comparison survives resume (pinned by r1_4 and the
   Adam-equivalence test); no production path assigns `_resolved_update_method`
   directly (grep: the only direct poke is the documented
   test_method_state_persistence one, which never reaches the boundary).
2. **Corrected docstrings describe the code exactly — CONFIRMED at the
   corrected sites**, traced against all three validation branches, both
   first-bind retentions, and the runner's actual call shapes (runner passes
   no `update_method` anywhere — the Parameters claim holds). Residue is R3-1
   (stale claims in the ADOPTED probe file, not the product docstrings) and
   the R3-3 pointer nit.
3. **Adopted probe files undamaged — CONFIRMED independently**: diffed both
   files against the reviewers' own commits (`40a9e69a`, `96ab1c9d`); R1 file
   = header + disclosed R1-1 inversion + R1-5 addition with R2-4 hunk,
   R1-2/3/4 bodies byte-identical; R2 file = docstring-only hunks. Each
   adopted test still tests what its name claims — except R3-1's name.
4. **Alias removal broke nothing that legitimately depended on it —
   CONFIRMED**: no production consumer of `resolve_update_state`'s return
   anywhere in `tpen/` (grep), trainer-level test callers unaffected
   (suite 3054 = 3052 + exactly my 2, same 5 skips), and the removal is
   pinned behaviourally (prior verifier's ALIAS2; my MUT-BYPASS/DOUBLEBUILD
   add that the surrounding carrier plumbing is now guarded from two more
   directions).
5. **Four usage modes + checkpoint/resume contracts preserved — CONFIRMED**:
   direct-fit, explicit-factory, bound-instance, stateless each pinned by a
   named test; Adam and SPRING resume-equivalence green; the one semantic
   asymmetry I could construct (unconfigured trainer + explicit-override bind
   + later plain call returns the override silently) is the SUPPORTED
   bound-instance flow, correctly scoped by the current docs — raising there
   would break that mode. Not a defect.

SCOPE CALLS, as asked: the shared "mismatched legacy optimizer ownership"
string between `_resolve_method_state` and `rebuild_update_state` — AGREE
out of scope here (pre-existing, on lines this slice does not otherwise
touch, already filed). No other scope disagreement.

## Discounts, stated plainly

- Writer, verifier and reviewer are all Claude; I am the same model family as
  the author of the text I audited. I attacked artefacts and ran mutants
  rather than reasoning about writer intent, but the provider-independence
  this process was designed around remains absent.
- One-child ruling: no second party checked my fixtures. Mitigation:
  predictions registered before results, 5/5 focused-arm outcomes matched by
  name, and the one aggregate miss (bypass-full 5 vs 3) decomposed into one
  genuine extra catch and one documented flake signature rather than being
  absorbed.
- The green claims at this head are bounded by the known rank-agreement
  flake, per the writer's standing disclosure — now file-scoped per above.

Artefacts (Active Netscratch run data, NOT to be deleted):
`/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r3-20261009T234003Z/` —
`checkout/`, `uv-env/`, `uv-cache/`, `junit/*-{51717669,51720341}.xml`,
`logs/*-{51717669,51720341}.log`, `logs/job-{51717669,51720341}.{out,err}`,
both sbatch scripts (v1 md5 `d3f2113e…` with the disclosed bare-python defect,
v2 md5 `9c8bd4e2…`), `submit-stderr*.txt` (both empty).

Head `dd1fa1464b38a71782e4dea29408a92ea89538f8` re-confirmed unmoved
immediately before filing. Any new commit invalidates these measurements
against the new SHA.
