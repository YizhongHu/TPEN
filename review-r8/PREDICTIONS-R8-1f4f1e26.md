# Round-8 review predictions — REGISTERED BEFORE ANY SUBMISSION

Head under review: `1f4f1e267c118ceef58125209dce68a801dea7f0` (PR 524, base
`claude/vmc-interface/a-observations`, stack base
`dbb486ea10744810dc95a6d3d288c47efed012fb`). Reviewer r8, actor
`claude-vmc-if-b-reviewer-r8`. Written and pushed BEFORE any Cannon job exists
for this round. Facility for all arms: FASRC-Cannon, partition `test`, inside
an allocation, never a login node. The round-8 brief disclosed the stopping
rule registered before round 7; per that brief it binds the writer, not this
review, and nothing below is softened by it.

## Exposure declaration — every number seen before measuring

From the item's notes, the PR thread, and round 7's pushed artefacts, read in
full before this file was written: focused five-file arm = 25/0 (15+6+2+1+1,
re-derived locally by `grep -c "def test_"`); `tests/unit` full = 3054 tests /
5 skipped (base 3029 = 3024 passed / 5 skipped); integration file
`test_train_runner.py` = 6 tests, green at head and base in round 6, green at
1132e910 in round 7, blob byte-identical at head and stack base
(`18fb6d98916f9ecba3e45e058dea6287323a151a`, round 6); M-A at 1132e910 =
exactly 4 failures, split 2 DID-NOT-RAISE + 2 message-mismatch via the older
guard; m-rc-int at f2006235 and 1132e910 = exactly
`test_resume_reproduces_the_uninterrupted_run_bitwise`, r6/r7 printed
`['raw_alpha','raw_beta'] != []`. Elapsed bands seen: focused/mutant arms
10–45 s, full arms 400–560 s, integration arms 10.9–30 s, fresh
`uv sync --locked` ~40–95 s. Hashes seen: trainer.py pristine
`b9b59f8b763bab3e1cd59272d78fb0c2066906414fc4e0a30546326f4d3d7388` (THIS head
— re-derived locally by me from `git show 1f4f1e26:tpen/training/trainer.py`),
`b44aaa0962f2e4e046294fe50a65e2140b680c64008bd2f3d7283755dbb1417e` (1132e910,
STALE here), `6e6723820a533d5092409df12aef1987f4f1948d090b01accd6e62c39a0e92c5`
(f2006235 and rounds 3–6, STALE here); train.py
`a13eb986c27ddb140692018773fbf9c441c86fb30f6f818bf306ff5d6ddf48f7` (unchanged,
re-derived locally). Round-7 mutated hashes m-A `4d7eff61…` / m-RC `3a1551f2…`
were seen; m-A's CANNOT recur here (pristine trainer.py changed) and NO mutated
hash below is inherited — both are derived in-job. Prior job ids
51667528–51813678. Local tag count 25; rounds 6–7 asserted TAGS=24 in-job —
the in-job count is recorded, not forced to either value. I will report raw
JUnit `<testsuite>` attributes (tests/failures/errors/skipped), the same form
every prior round used, so an agreement cannot hide in a changed decomposition.

## Hash discipline (pre-registered VOID condition)

In-job, before any arm: `git rev-parse HEAD` must equal
`1f4f1e267c118ceef58125209dce68a801dea7f0`; `git status --porcelain` must be
empty; re-derived pristine sha256 of `tpen/training/trainer.py` must equal
`b9b59f8b…` above and of `tpen/runner/train.py` must equal `a13eb986…`. A
pristine trainer.py reading `b44aaa09…` or `6e672382…` means the checkout is
not at this head and EVERY arm in the job is VOID, not readable. Each mutant
must print the mutated region, its in-job-derived mutated sha256, and be
restored to the pristine sha256 with porcelain 0 before the next arm. A job
that cannot be read is NOT MEASURED — reported as such, never back-filled.

## Arms, with named failures, named survivors, and mechanisms

One Slurm job, 4 CPU / 32 GiB / 45 min, partition `test` (CPU verification,
operator-sanctioned per `cpu-partition-test-is-allowed-operator-2026-08-31`;
small immediate-response job per `fairshare-zero-is-the-steady-state-2026-08-31`
— the stated reason for not naming the production partitions).

### Arm 1 — g-focused: the five binding files
`test_update_binding.py` + `_review_r1/_r2/_r3/_r4.py`.
PREDICT: tests=25 failures=0 errors=0 skipped=0.

### Arm 2 — g-unit: all of `tests/unit`
PREDICT: tests=3054 failures=0 errors=0 skipped=5.
PRE-REGISTERED DISPOSITION: if failures=1 and the failing test lives in
`tests/unit/test_run_id_rank_agreement.py` with signature
`AssertionError: a rank wrote no receipt`, that is the carried file-scoped
Gloo flake (fired round 6, silent round 7) — reported, decomposed, counted as
a prediction miss, and NOT rerun to make the arm green. Any other failure is a
finding at this head.

### Arm 3 — g-int: `tests/integration/training/test_train_runner.py`
PREDICT: tests=6 failures=0 errors=0. The file is outside the PR diff and the
commit under review is doc/comment-only by two comparators (below). A red here
means the doc-only evidence is wrong and outranks everything else in the round.

### Arm 4 — m-a-focused: disable the retained-carrier check (trainer.py)
Mutation: `if optimizer is not self._bound_optimizer:` in `bind_update_method`
becomes `if optimizer is not self._bound_optimizer and False:` (unique
substring, replacement count asserted = 1). This arm is the empirical test of
the comment this commit LEFT in trainer.py: "The distinct message is what lets
the review probes pin THIS check by its own text, and disabling it is what
makes them fail." It matters more than usual this round because that sentence
is now the only live claim at the check.
PREDICT: failures=4 of 25, BY NAME AND MECHANISM:
1. `test_r1_1_stateless_direct_bind_refuses_a_second_carrier` — genuine
   DID-NOT-RAISE (direct bind; stateless; no other guard on the path).
2. `test_r1_1_control_the_default_adapter_refuses_the_same_sequence` — genuine
   DID-NOT-RAISE (bind-bind sequence; `_resolve_method_state` is never invoked
   by `bind_update_method`, so the older guard cannot answer).
3. `test_a_second_carrier_is_refused_after_binding` — MESSAGE-MISMATCH: the
   older guard raises "mismatched legacy optimizer ownership" from
   `_resolve_method_state`, reached through the second `resolve_update_state`;
   the test matches "already bound to a different optimizer".
4. `test_a_refusal_happens_before_any_update_runs` — MESSAGE-MISMATCH, same
   older-guard mechanism, reached through `fit`.
NAMED SURVIVORS a careless reader might expect to fail:
`test_a_second_model_is_refused_after_binding` (model check untouched),
`test_a_conflicting_late_selector_is_refused_after_binding` and `test_r1_2_*`
(spec guard untouched), `test_r2_1_*`/`test_r3_*`/`test_r4_2_*`
(foreign/runner paths never reach this guard). 21 of 25 pass.
If failures != 4, or the DNR/mismatch split differs from 2/2, either the
surviving trainer.py sentence or my reading of it is wrong — a finding either
way.

### Arm 5 — m-rc-int: the R4-2 payload against the integration file
Mutation (witness-free M-RC, `tpen/runner/train.py`): in the `train_resume`
branch only, `restore_checkpoint_with_events(... optimizer=optimizer ...)`
becomes `optimizer=make_optimizer(self.optimizer, self.model.parameters())` —
restore is handed a FRESH build while `fit` keeps the run's real carrier. The
unique multi-line context is asserted (count = 1) so the `fit` call's
`optimizer=optimizer` cannot be hit.
PREDICT: failures=1 of 6, exactly
`test_resume_reproduces_the_uninterrupted_run_bitwise`, via the
diverged-parameters assert (non-empty diverged list; r6/r7 printed
`['raw_alpha','raw_beta']`). The other 5 survive. If this mutant SURVIVES, the
pointer that replaced review_r4.py's measurement sentence points at a record
whose property no longer holds at this head, and that outranks everything else
in the round.

## Elapsed bands, announced in advance

Fresh env sync 40–95 s once; g-focused 12–45 s; g-unit 400–560 s; g-int and
m-rc-int 10–60 s; m-a 10–45 s. Whole job 10–20 min. An arm far outside its
band is reported, not absorbed.

## What was already established statically, BEFORE this run

### Behaviour-neutrality, MY OWN implementation (written blind to round 7's)

Two comparators with controls in BOTH directions, run on `git show` extracts
of all four changed files at 1132e910 and 1f4f1e26 (pristine hashes above
derived from these extracts):
- AST-with-docstrings-stripped (docstring replaced by `pass`, never deleted,
  so a docstring-only body cannot vanish): IDENTICAL for all four files.
- Token-stream-minus-COMMENT/NL: IDENTICAL for trainer.py (comment-only edit),
  DIFFERENT for the three probe files exactly as docstring edits must be.
- Controls: an injected statement (`int(max_steps) + 1`) is caught by BOTH; a
  synthetic comment-only edit passes BOTH (the declared blind spot, which is
  what makes trainer.py's pass/pass reading honest); a synthetic
  docstring-only edit passes AST and is caught by the token stream. Token
  streams are non-trivial (313–2554 tokens per file), so the comparison is not
  vacuously green.
- Bypass construction attempted: a change passing both comparators is confined
  to comments and blank lines; one passing only AST is confined to docstrings.
  Channels by which those alter behaviour, each measured ABSENT at this head:
  no doctest configuration (`addopts` is typeguard-only, no doctest flag
  anywhere), no `__doc__` read targets any touched module (the one `__doc__`
  use in `tpen/` is `tpen/run.py`'s own argparse description), no
  getsource/getdoc/linecache/`.lineno`/`co_firstlineno` under
  `tests/unit/training/` or `tests/integration/training/` or in the two
  production files, no encoding-declaration comments in any of the four files,
  and per-arm `__pycache__` purge in-job closes stale-bytecode shadowing. I
  could not construct a bypass.

### Deleted-fact reachability — NONE destroyed (checked independently)

Every fact this commit deletes maps to a durable location I verified exists:
the review_r4.py round-6 integration measurement (green head+base, red under
M-RC, failing test and diverged list) lives in `review-round-6-verdict-f2006235`
+ PR comment 6096119065 + `carried-forward-known-gaps-at-1132e910` #1 (which
also carries job 51808607 and the blob sha the docstring never had); the
review_r3.py head SHA `dd1fa146…` lives in PR comment 6091362333 and three
dd1fa146-keyed notes; trainer.py's true historical content (deletable-then,
M2 survival) lives in the round-2 and round-7 verdicts and in 1f4f1e26's own
commit message; review_r1.py's deleted count claim was FALSE (R7-1) and its
"structural cause" analysis survives, stronger, in the stopping-rule note.

### Three static findings in hand (dispositions deferred to the verdict)

All three verified by MY OWN reading of the tree and of `git show
4175fc17:tpen/training/trainer.py`; none moves with the arms; all three were
born at 47989a12 and SURVIVE this commit (none is introduced by it):
- R8-1 — `test_update_binding_review_r3.py:81` (function docstring): "the case
  the last-build capture in R1-5/R2-1 cannot see", present tense, is FALSE at
  this head: both captures are builds-lists asserting `builds[0]` since
  57baf429 (review_r1.py:425-436,454; review_r2.py:193-203,222). Same
  substance as R7-3 one scope level down, in the same file whose module header
  the fix retensed — and the commit's new header sentence "this header does
  not describe them" sits one screen above a function docstring that still
  does, falsely.
- R8-2 — `test_update_binding.py:563-569`: "the older check fires first on
  this path, so the binding check could be deleted outright and no test
  noticed" repeats R7-2's false mechanism in the writer's own suite file: at
  4175fc17 the BINDING check fired first on every live path (via
  `_resolved_update_state` after a resolve, via `bound.update_state()` on a
  direct bind); the older check was the identical-string backstop that
  answered only UNDER deletion. Conclusion (M2 survived) true; mechanism
  false.
- R8-3 — `test_update_binding_review_r1.py:146-148`: "Before the fix this
  control passed through `_resolve_method_state`'s older 'mismatched legacy
  optimizer ownership' check" is FALSE: the control's bind-bind sequence never
  reaches `_resolve_method_state` (nothing in `bind_update_method` calls it);
  at 4175fc17 the refusal came from the binding boundary's own state-derived
  carrier check, which covered the stateful default adapter via
  `bound.update_state()`. The shared-string halves of the sentence are true.

## Other pre-registrations

- The rank-agreement flake: see Arm 2. Never rerun-to-green; per-rank evidence
  preserved under the run root.
- If g-int and m-rc-int are BOTH red with the same signature, that is an
  environment artefact (both-sides rule), not a head finding.
- No `scancel` of any job under any circumstances; the two long-PENDING
  foreign jobs (`rclone-backup` 40911608, `storage-report` 40911609) are not
  mine and not touched.
- Netscratch group quota is read with the supported call (`quota -g
  kozinsky_lab /n/netscratch/kozinsky_lab/Lab/rhu`) on the login node before
  submission and recorded with its read time; if at limit, I stop and report
  rather than submit a job that will die EDQUOT mid-write.
- Run root is unique per attempt
  (`/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r8-<UTC>Z/`), with
  `UV_PROJECT_ENVIRONMENT` and `UV_CACHE_DIR` inside the run root and OUTSIDE
  the checkout; `__pycache__` purged per arm with the remaining count printed.
