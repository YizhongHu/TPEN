# Round-7 review predictions — REGISTERED BEFORE ANY SUBMISSION

Head under review: `1132e910324d12d75d73403f0fb2762b04209118` (PR 524, base
`claude/vmc-interface/a-observations` @ `dbb486ea10744810dc95a6d3d288c47efed012fb`).
Reviewer r7, actor `claude-vmc-if-b-reviewer-r7`. Written and pushed BEFORE any
Cannon job exists for this round. Facility for all arms: FASRC-Cannon, partition
`test`, inside an allocation, never a login node.

## Exposure declaration — every number seen before measuring

From the item's notes and PR thread, read in full before this file was written:
base `tests/unit` = 3029 (3024 passed / 5 skipped); head counts at f2006235 =
3054 tests / 5 skipped full, 25/0 focused; integration file
`test_train_runner.py` = 6 tests, green at head f2006235 AND base dbb486ea
(round 6); M-A at f2006235 = 4 failures, split 2 DID-NOT-RAISE (both R1-1
probes) + 2 message-mismatch via the older guard (rounds 2v, 5, 6); M-DB = 6
failures; M-RC full = exactly `test_r4_2_a_runner_driven_adam_resume_is_
bitwise_identical`; m-rc-int at f2006235 = exactly
`test_resume_reproduces_the_uninterrupted_run_bitwise` via the
diverged-parameters assert, r6 printed `['raw_alpha','raw_beta'] != []`.
Elapsed bands seen: focused/mutant arms 12–40 s, full arms 420–520 s,
integration arms 14–30 s, fresh `uv sync --locked` ~50 s. Hashes seen:
trainer.py pristine b44aaa0962f2e4e046294fe50a65e2140b680c64008bd2f3d7283755dbb1417e
(this head) and 6e6723820a533d5092409df12aef1987f4f1948d090b01accd6e62c39a0e92c5
(rounds 3–6, STALE at this head); train.py
a13eb986c27ddb140692018773fbf9c441c86fb30f6f818bf306ff5d6ddf48f7 (unchanged).
Prior job ids 51667528–51808607 range. I will report raw JUnit `<testsuite>`
attributes (tests/failures/errors/skipped), the same form every prior round
used, so an agreement cannot hide in a changed decomposition.

## Hash discipline (pre-registered VOID condition)

Pristine sha256 values are RE-DERIVED IN-JOB and must equal the two values
above for trainer.py and train.py. Mutated hashes are derived in-job and
recorded; no mutated hash is inherited from any prior round's receipt. If an
in-job pristine hash mismatches — in particular if trainer.py reads
`6e672382…`, the rounds-3–6 value — the checkout is not at this head and every
arm in the job is VOID, not readable. Every mutant must print the mutated line
and be restored to the pristine sha256 with porcelain 0.

## Arms, with named failures, named survivors, and mechanisms

One Slurm job, 4 CPU / 32 GiB / 45 min, partition `test`.

### Arm 1 — g-focused: the five binding files
`test_update_binding.py` + `_review_r1/_r2/_r3/_r4.py`.
PREDICT: tests=25 failures=0 errors=0 skipped=0. (15+6+2+1+1.)

### Arm 2 — g-unit: all of `tests/unit`
PREDICT: tests=3054 failures=0 errors=0 skipped=5 (base 3029 + 25).
PRE-REGISTERED DISPOSITION: if failures=1 and the failing test lives in
`tests/unit/test_run_id_rank_agreement.py` with signature
`AssertionError: a rank wrote no receipt`, that is the carried file-scoped
Gloo flake — reported, decomposed, counted as a prediction miss, and NOT
rerun to make the arm green. Any other failure is a finding at this head.

### Arm 3 — g-int: `tests/integration/training/test_train_runner.py`
PREDICT: tests=6 failures=0 errors=0. This file is outside the PR's diff and
was green at head f2006235 and base dbb486ea in round 6; the two commits since
are doc/comment-only (independently verified below). A red here would mean the
doc-only evidence is wrong and would outrank everything else in the round.

### Arm 4 — m-a-focused: disable the retained-carrier check (trainer.py)
Mutation: the `optimizer is not self._bound_optimizer` condition in
`bind_update_method` gets `and False` appended — the check can never fire.
This arm is the empirical test of the trainer.py comment this head rewrote:
"at that time this check could have been deleted outright without a single
test failing. That is no longer so."
PREDICT: failures=4 of 25, BY NAME AND MECHANISM:
1. `test_r1_1_stateless_direct_bind_refuses_a_second_carrier` — genuine
   DID-NOT-RAISE (direct bind; no resolve; no other guard on the path).
2. `test_r1_1_control_the_default_adapter_refuses_the_same_sequence` —
   genuine DID-NOT-RAISE (direct bind-bind; `_resolve_method_state` is never
   invoked on this path, so the older guard cannot catch it).
3. `test_a_second_carrier_is_refused_after_binding` — MESSAGE-MISMATCH: the
   older guard raises "mismatched legacy optimizer ownership" from
   `_resolve_method_state` (reached through the second `resolve_update_state`),
   while the test matches "already bound to a different optimizer".
4. `test_a_refusal_happens_before_any_update_runs` — MESSAGE-MISMATCH, same
   older-guard mechanism, reached through `fit`'s `_resolve_method_state`.
NAMED SURVIVORS a careless reader might expect to fail: `test_a_second_model_
is_refused_after_binding` (model check untouched), `test_a_conflicting_late_
selector_is_refused_after_binding` and `test_r1_2_*` (spec guard untouched),
`test_r2_1_*`/`test_r3_*`/`test_r4_2_*` (foreign/runner paths never reach this
guard). 21 of 25 pass.
If failures != 4, or the DNR/mismatch split differs from 2/2, the trainer.py
sentence or my reading of it is wrong — a finding either way.

### Arm 5 — m-rc-int: the R4-2 payload against the integration file
Mutation (witness-free M-RC, in `tpen/runner/train.py`): the `train_resume`
branch hands `restore_checkpoint_with_events` a FRESH `make_optimizer` build
while `fit` keeps the run's real carrier. This arm re-measures, AT THIS HEAD,
the present-tense sentence 3eb6cbb1 wrote into review_r4.py:
"`tests/integration/training/test_train_runner.py` does catch this payload".
PREDICT: failures=1 of 6, exactly
`test_resume_reproduces_the_uninterrupted_run_bitwise`, via the
diverged-parameters assert (a non-empty diverged list; r6 printed
`['raw_alpha','raw_beta']`). The other 5 tests survive.
If this mutant SURVIVES, the review_r4.py sentence is false at this head and
the finding outranks everything else in the round.

## Elapsed bands, announced in advance

Fresh env sync ~40–90 s once; g-focused 12–45 s; g-unit 400–560 s; g-int and
m-rc-int 12–60 s; m-a 10–45 s. Whole job 10–20 min. An arm far outside its
band is reported, not absorbed.

## What was already established statically, BEFORE this run

Doc-only verification, MY OWN method, not the writer's: for each of
review_r1/r2/r4 and trainer.py between f2006235 and 1132e910, (a) AST with all
docstrings stripped is identical, (b) token stream minus COMMENT/NL tokens is
identical for trainer.py (strictly stronger than the writer's changed-lines-
are-comments check: it also rules out comment edits that merge or split code
lines) and differs for the three probe files exactly as docstring edits must.
Controls: both comparators flag an injected statement; the AST comparator
passes a synthetic docstring-only edit; the token comparator passes a
synthetic comment-only edit. Behaviour channels measured ABSENT at this head:
no `__doc__` read targets a test module (6 reads, all tpen modules/argparse),
no doctest configuration (addopts is typeguard only), no
getsource/linecache/lineno use under tests/unit/training or
tests/integration/training.

Three static findings are in hand (dispositions deferred to the verdict so the
head cannot move under these predictions): R7-1 — review_r1.py:7-9's "Six
consecutive review rounds found exactly one such sentence false" is wrong
against the record (round 6 alone found two accepted in-class instances, and
rounds 1–4's instances were tree-checkable wording defects, not members of the
stated class); R7-2 — trainer.py's retained clause "every test that asserted
on the shared string reached it through `resolve_update_state`, where the
older check fires first" is false at the head it describes (4175fc17:
`test_a_refusal_*` reached the string through `fit`, and on both paths the
BINDING check itself raised first; the older check was the identical-string
backstop that fired only under deletion — which is the actual deletability
mechanism); R7-3 — review_r3.py:29-35 asserts in the present tense that the
R1-5/R2-1 captures "record only the LAST optimizer … `built` is overwritten
and `seen is built` passes", which 57baf429 falsified in the SAME COMMIT that
adopted the file (both captures became builds-lists asserting `builds[0]`),
and which rounds 5–6's M-DB arms measured false (both tests FAIL under
double-build). None of these is empirical; none moves with the arms above.

## Gloo-flake disposition and other pre-registrations

- The rank-agreement flake: see Arm 2. Never rerun-to-green; per-rank evidence
  preserved under the run root.
- If g-int and m-rc-int are BOTH red with the same signature, that is an
  environment artefact (both-sides rule), not a head finding.
- No `scancel` of any job under any circumstances; the two long-PENDING
  foreign jobs (`rclone-backup`, `storage-report`) are not mine and not
  touched.
- A job that cannot be read is NOT MEASURED — reported as such, never
  back-filled.
