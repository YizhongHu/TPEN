# Round-4 review VERDICT (FINAL) — head 57baf42907d32432023b512634eab4c9d6c38935

2026-10-09/10, reviewer fable-5, worktree vmc-if-b-review-r4, operating under
the one-claude-child-per-lane ruling: probes designed, implemented and run by
the reviewer itself, no second party checked the fixtures — discount
accordingly. This SUPERSEDES the PARTIAL verdict
(`REVIEW-R4-VERDICT-57baf429-PARTIAL.md`, PR comment 6091974169), which is
left intact as the record of the mid-round infrastructure loss; everything in
it stands, and the empirical half is now complete.

**VERDICT: BLOCK — 2 issues (R4-1 low, R4-2 low) + 1 nit + 1 process note.
NO production-code defect found.** The production diff at this head held under
static tracing and four mutants; both issues are test/doc-strength.
Accept/dispose is the writer's call.

CANONICAL COPY: PR 524 comment + this file on
`claude/review/vmc-if-b-r4-57baf429`. A Task Orchestrator note was impossible
for this session throughout: the MCP connection failed at session start
(ECONNRESET, later ECONNREFUSED while the Docker daemon was down) and does not
reconnect mid-session; the writer reports the service healthy again, and
prefers the PR comment canonical regardless — so it is.

## Evidence

Predictions for every arm were registered and PUSHED BEFORE the first
submission: `PREDICTIONS-R4-57baf429.md` at
`f27937b0df2bd71d23b2e5f3ddecffb8e723d3ad` — per-arm counts, failing tests BY
NAME, and the survivors a careless reader would assume fail. Nothing below
was reinterpreted after the fact; the one miss is decomposed, not absorbed.

FACILITY: FASRC-Cannon, partition `test` requested AND DELIVERED both jobs,
node holy8a24101, 4 CPU / 32 GiB. The mid-round outage: Cannon SSH died with
a session restart (keyboard-interactive 2FA; my probe count: exactly one,
BatchMode, stopped per `cluster-access`), later restored by the operator.
Round root (run data, preserve, never delete):
`/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r4-20261010T003554Z/` —
own full clone at the exact head, own `UV_PROJECT_ENVIRONMENT`/`UV_CACHE_DIR`/
`TMPDIR` outside the checkout, `job-a-green.sbatch`
(md5 `4ca72dfda030e3152fe609f38b4f612f`), `job-b-mutants.sbatch`
(md5 `ffdf67426307c15a5d08ac9664231c22`), `apply_mutant.py`
(md5 `ec195bf84badab7a3e7cedfd80f2fe2f`; every mutation asserts the exact
original line content before writing, so drift fails loudly), submit-time
stderr captures (both empty), job-scoped `junit/<arm>-<jobid>.xml` and
`logs/<arm>-<jobid>.log`.

In-job, per arm: HEAD asserted at 40 chars, tracked porcelain 0,
`__pycache__` purged and printed 0, interpreter asserted job-local uv-env
CPython 3.12.13 / torch 2.12.0+cpu (non-system, >=3.10), mutants applied
through `uv run` (never bare python), counts read from raw JUnit
`<testsuite>` attributes with errors separate from failures. QOS headroom
checked before each submit (1/5 in use both times, not mine; nothing
cancelled, ever). Integrity cross-check: pristine `tpen/runner/train.py`
sha256 `a13eb986…` MATCHES round 3's recorded pristine at `dd1fa146` — the
diff under review does not touch that file; pristine `tpen/training/trainer.py`
is `6e672382…`, new at this head, as the diff requires. Every mutant proven
ACTIVE (sha256 differs + mutated region printed; M4 additionally by a runtime
witness, below) and restored to pristine sha256 with porcelain 0.

## Prediction vs. result — say it by name

| arm | job | state | predicted | measured | match |
|---|---|---|---|---|---|
| G-FOCUSED (4 files) | 51728679 | COMPLETED 0:0, arm 49.8 s | 24/0 | tests=24 failures=0 errors=0 | ✓ |
| G-FULL tests/unit | 51728679 | COMPLETED 0:0, arm 667.5 s | 3053/0/skip 5 | tests=3053 failures=0 errors=0 skipped=5 | ✓ exact (3029+24) |
| M1 DOUBLEBUILD focused | 51742735 | COMPLETED 0:0 | 5, by name | failures=5, SAME five names | ✓ |
| M2 DOUBLEFIRST focused | 51742735 | 〃 | exactly 1: the r3 count pin | failures=1: `test_r3_the_runner_builds_exactly_one_carrier_for_a_foreign_trainer` (`assert 2 == 1`) | ✓ |
| M3 SPECCHECK focused | 51742735 | 〃 | 2, by name | failures=2, SAME two names (both `DID NOT RAISE`) | ✓ |
| M4 RESTORE-CARRIER focused | 51742735 | 〃 | 0 | failures=0, witness hits=1 | ✓ |
| M4 RESTORE-CARRIER full | 51742735 | 〃 | 0 | **failures=1** — MISMATCH, decomposed below | ✗ |

Elapsed controls: job A 19:57 total, job B 17:17 total, both inside the
pre-announced bands except the M4-full arm at 779.7 s against an announced
420–670 s band — a ~16% overrun under mutation load, noted, not material.
The disclosed `test_run_id_rank_agreement.py` flake did NOT fire in either
full arm; the pre-registered disposition was never needed.

M1's five: `test_the_default_adapter_is_constructed_once_per_runner_driven_run`
and `test_r1_4_*` (both on the distinct binding-boundary refusal "already
bound to a different optimizer"), `test_r1_5_*`, `test_r2_1_*` (both identity
failures — **these two were GREEN under this same mutant shape at `dd1fa146`
in round 3's measurement; they are RED now. R3-2 is CLOSED, measured, not
relocated**), and the r3 count pin. M2's survivors matched too: r1_5 and
r2_1 stayed green under build-twice-feed-first, exactly as registered — that
direction is carried by the count pin alone, by design, and the design held.
M3 proves the R3-3 `late_spec`→`spec` collapse left the conflict check alive
and pinned from two directions (the adopted R1-2 probe and the writer's own
refusal test).

## The M4-full mismatch, decomposed — instrument channel, not payload

The single failure is
`tests/unit/test_durable_append.py::test_no_one_record_writer_opens_its_own_append_handle`:
"append-handle owners changed… Unexpected: {PosixPath('tpen/runner/train.py')}".

That test is an AST census of `tpen/` for the append-handle WRITE PATTERN. It
did not see the carrier swap: it saw MY WITNESS — the activity probe I built
into M4 (`open(os.environ["MUTANT_WITNESS"], "a")` at the mutated line, so a
predicted-survive mutant cannot pass vacuously) is itself an append-handle in
`tpen/`, and the census correctly refused to let an unclassified one appear.
The mutation payload (`make_optimizer(...)` handed to restore in place of the
run's carrier) adds no append handle and is invisible to that test.

So the SUBSTANTIVE prediction — no test in `tests/unit` detects the payload —
is CONFIRMED: 3052/3053 pass, the one failure is fully attributed to the
instrument, and the witness proves the mutated restore path EXECUTED (2 hits
in full — the two runner-driven resume call sites; 1 hit in focused — r1_4).
The LITERAL prediction (failures=0) was WRONG, and the error was mine: I
registered the witness design and failed to ask what else in the suite
observes the channel the witness writes on. Same-observable-is-not-same-
channel, in my own fixture. Recorded as a process note because the registered
number missed; the miss changes nothing about what the arm establishes, and
decomposing it is what the pre-registration was for. (Incidentally measured:
the append-handle census is sensitive enough to catch a one-line instrument —
a useful fact about that guard's reach.)

## Issues

### R4-1 (low) — a cross-reference to the R3-1-renamed test went stale IN THE SAME COMMIT that renamed it

`tests/unit/training/test_update_binding.py:284` — the R2-2 fix's docstring
says the override-then-plain-call raise "is pinned by
`test_r1_2_explicit_none_after_an_explicit_override_raises_despite_the_doc`".
That name was deleted at this head by the R3-1 rename
(`..._raises_as_the_doc_states`, `test_update_binding_review_r1.py:169`); the
pointer is broken as well as stale, and the dead name carries the refuted
"raises despite the doc" claim back into the writer's own suite — the fourth
consecutive round with stale/refuted wording at this one spot (R1-2, R2-2,
R3-1, now this). FIX: one line; and prefer a rename-surviving pointer ("the
adopted round-1 R1-2 probe"). Found statically; the writer has verified and
ACCEPTED this finding and the nit below during the round, holding the fix
until this verdict so the head stayed fixed under my registered predictions —
which is the right order.

### R4-2 (low) — the runner-driven restore-carrier axis of the resume contract is unpinned in `tests/unit` — now MEASURED, not inferred

M4 hands `restore_checkpoint_with_events` a fresh factory build while `fit`
keeps the run's real carrier — the exact lifecycle class this slice exists to
remove (state restored into an object the loop never uses). Measured: every
test in `tests/unit` passes under it (the one failure above is my
instrument's, not the mutation's), while the witness proves the mutated path
ran. Why: `test_r1_4_*` pins the construction COUNT only; every exactness
resume test in the focused files (`test_adam_moments_*`, `test_r1_3_*`,
`test_the_default_adapter_survives_restore_as_one_instance`) calls
`restore_checkpoint` directly, bypassing `Train.run`; block-NG's runner-driven
bitwise resume rides a stateless SGD carrier, so a lost optimizer state dict
is numerically invisible there. The one test that would catch it —
`tests/integration/training/test_train_runner.py::test_resume_reproduces_the_uninterrupted_run_bitwise`
(Adam, full `run_from_config` path; support for "would catch" is static: its
config is stateful and its comparison exact — I did not run integration) —
is in a tree NO verification receipt in this PR runs, the writer's included.
FAILURE SCENARIO: a future edit reroutes the restore call's carrier (a
"defensive" rebuild, an argument shuffle); every receipt this PR's process
produces stays green; resumed runs silently lose optimizer state on the
default path. ADOPTABLE FIX, same family as accepted R1-4/R3-2: a
runner-driven Adam resume exactness pin in `tests/unit` (extend `test_r1_4_*`
with trajectory equality against an uninterrupted arm, or a dedicated test);
M4's mutant is the red-arm proof for whoever lands it. Filed at low because
the property is covered in-repo (integration) — the gap is in the suite every
receipt in this PR's verification process actually runs.

### R4-nit (non-blocking) — "NEITHER EXISTS AT THIS HEAD" is true of the claims, false of the strings

`test_update_binding_review_r1.py:175-176` says the two refuted doc strings
"NEITHER EXISTS AT THIS HEAD"; `trainer.py:519` carries "use what is already
bound" as a quoted historical reference. The claims are gone; the string is
not. Suggested: "neither survives as a live claim; one is quoted, as history,
at the check itself." Cannot mislead about behaviour, only about a grep.
Accepted by the writer alongside R4-1.

## Writer-claim audit — the six claims named for attack, final

1. **`builds[0]` closes R3-2 rather than relocating it — CONFIRMED, measured.**
   The two pins round 3 measured GREEN under DOUBLEBUILD are RED under it at
   this head (M1); the one direction the identity assert cannot see
   (build-twice-feed-FIRST) fails the adopted count pin and only it (M2,
   exactly as registered). No single-slot capture remains anywhere in tests/
   (grep). Residual, stated: the count pin lives on the carrier-FREE foreign
   shape; it covers the carrier-bearing one because `Train.run` discards the
   resolve return unconditionally with no branch on its shape — verified at
   this head, and any future branch there reopens the question.
2. **`late_spec`→`spec` collapse behaviour-preserving — CONFIRMED statically
   (same expression, same call, plain `__init__` attribute, nothing assigns
   it on the path between; a revert mutant is equivalent by construction),
   and the collapsed check is ALIVE and pinned (M3: both registered tests
   fail by DID-NOT-RAISE when it is disabled, registered survivors all
   survived).**
3. **Renamed R1-2 probe + docstring describe this head — LARGELY CONFIRMED.**
   Name accurate against the head's Raises section; rewritten docstring and
   header past-tense and accurate. The predicted fourth doc error is real and
   is R4-1 — in the NEIGHBOURING file; plus the R4-nit.
4. **Dropping the weakened-isinstance instrument lost nothing — CONFIRMED.**
   Adopted file is byte-identical to the reviewer's original minus exactly
   the instrument (diffed against `5d2b569e`); the deletion follows the
   instrument's own docstring instruction; the header records the measurement
   and quotes the reviewer VERBATIM (hash-level check against the round-3
   commit, not a retype); the instrument stays recoverable at `5d2b569e`.
5. **`assert builds` guards diagnose correctly — CONFIRMED** statically
   (status asserted first; empty-builds-with-completed-run means exactly what
   the message says; without the guard `builds[0]` raises IndexError and
   masks it) and exercised: under M1/M2 the failures land on the identity and
   count asserts with their own messages, never on the guard.
6. **Four usage modes + checkpoint/resume contracts preserved — CONFIRMED at
   the suite level** (3053/0/5, exact base+24, both full arms) **with one
   measured evidence gap on the resume side: R4-2.** Preserved, yes;
   fully pinned by the suite this process runs, no.

SCOPE CALLS, as asked: shared "mismatched legacy optimizer ownership" string
stays a follow-up — AGREE, round 3's grounds. Flake disposition: CONCUR as
carried (file-scoped, pre-existing, disclose-not-chase); it did not fire in
any of my seven arms, so this round adds a null observation, not a
re-verification.

## Discounts, stated plainly

- Writer, prior reviewers and I are all Claude (model diversity only; the
  writer is Opus 5, I am Fable 5). I attacked artefacts and ran mutants
  rather than reasoning about intent; provider independence remains absent.
- One-child ruling: no second party checked my fixtures — and the M4-full
  miss shows exactly what that costs: my own instrument contaminated a
  channel nobody else would have had to be the one to notice. Mitigations
  that held: predictions registered and pushed pre-submission, 6 of 7 arms
  matched by name, the miss decomposed to a named cause with the payload
  conclusion intact.
- The round spans an infrastructure outage; the PARTIAL verdict records
  what was and was not knowable at that point, and job 51728679's artefacts
  were read only after, against predictions pushed before its submission.
