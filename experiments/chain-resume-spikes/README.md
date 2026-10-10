# Chain-resume comparative spikes — R0 shared harness

Recipe document for TPEN chain-resume spike **R0** (Task Orchestrator item
`ab757fd8-cf77-4860-99cc-e99f903f8a53`), under plan
`polaris-chain-resume-comparative-spikes-2026-09-09`.

R0 owns the **shared continuation fixture** that the three candidate lanes —
R1 (Balsam), R2 (IRI/PBS), R3 (Globus Compute) — inherit, plus the
generation-preservation and evidence-schema tests. R0 selects no backend.

---

## Gate coverage map

This section is the load-bearing part of the document. Read it before quoting
any result from this lane.

> **THE SINGLE MOST CONSEQUENTIAL FACT FOR R1, R2 AND R3, STATED HERE RATHER
> THAN LEFT TO BE INFERRED FROM FIVE SEPARATE `UNMEASURED` ROWS: THIS SHARED
> ORACLE IS SCOPED TO LOCAL FAULT INJECTION AND DOES NOT COVER THE
> SCHEDULER-INTERACTION GATES THE CANDIDATE LANES EXIST TO TEST.**
>
> G2 real scheduler-walltime exhaustion, G4 controller restart, G5
> accepted-submit-lost-ack, G7 cancellation and budgets, and G9 parallel
> placement are **all UNMEASURED**, and they are the gates that distinguish
> Balsam from PBS from Globus Compute. What R0 delivers is a stochastic
> chain-resume oracle plus a local fault harness: **it will tell a candidate
> lane whether its resume reproduces the trajectory, and it will not tell that
> lane anything about how its scheduler behaves.** Reading the table row by row
> makes each absence look like a detail; together they are the shape of the
> deliverable, and **a docs amendment does not change it.**

| Gate | Status in R0 | What backs it |
|---|---|---|
| **G0** stochastic parity with restore-disabled mutation | **COVERED** | `tests/unit/chain_resume_spike/test_fixture_stream_sensitivity.py` (ARM T) and `tests/integration/chain_resume_spike/test_native_reference_arm.py` (ARM N) |
| **G1** clean yield at a coherent boundary | **COVERED IN ITS LOCAL SHAPE ONLY** | `test_typed_outcomes_and_ledger.py`. The typed yield/final/fatal/transient outcome and automatic continuation to the same total target are exercised. What is *not* exercised is a yield triggered by a real scheduler signal. |
| **G2** actual scheduler walltime exhaustion | **UNMEASURED** | No PBS/Slurm walltime event is driven here. The `SIGKILL_FROM_PARENT` and `SIGTERM_SELF` arms are *not* substitutes, and the shared contract says so explicitly. |
| **G3** transient retry cap, prompt fatal stop | **PARTIAL — instruments only** | The four outcome kinds and the retry-budget policy exist and are tested; no retry *controller* is built here, so the cap itself is unmeasured. |
| **G4** controller restart | **UNMEASURED** | R0 builds no controller. |
| **G5** accepted submit, lost acknowledgement | **UNMEASURED** | `FaultPoint.ACCEPTED_SUBMIT_LOST_ACK` is registered as an instrument for the candidate lanes and is deliberately not driven. |
| **G6** generation preservation | **COVERED** | `test_generation_preservation.py`, against production's own selection primitives. |
| **G7** cancellation and budgets | **UNMEASURED** | `FaultPoint.CANCELLATION_REQUESTED` and `FaultPoint.BUDGET_EXHAUSTED` are instruments only. |
| **G8** identity and credited-work accounting | **COVERED** | `test_typed_outcomes_and_ledger.py`. |
| **G9** parallel placement | **UNMEASURED** | Out of scope for R0. |

**Providing an instrument is not covering the gate.** R0 registers
`CANCELLATION_REQUESTED`, `BUDGET_EXHAUSTED` and `ACCEPTED_SUBMIT_LOST_ACK` so
that R1/R2/R3 can drive them against a real backend. R0 drives none of them.

**Any skipped or unexecuted required arm is UNMEASURED, never PASS.** ARM N
skips wholesale on any host without `torch`, which includes the development
workstation, so a green LOCAL run establishes nothing about the native surfaces.
It has now been EXECUTED in Cannon job 45645515 (5/5, no skips) — see the
measured section below. Read the local skip and the Cannon result as two
different facts.

---

## The two arms

### ARM T — torch-free, runs anywhere

`tests/helpers/chain_resume_spike/` plus `tests/unit/chain_resume_spike/`.

Runs on a host with no `torch` and no GPU. That is the point beyond R0: the
shared fixture executes on a facility worker without importing `torch`, which
decouples backend-mechanics testing from runtime qualification. R1/R2/R3 can
drive a real backend against this fixture before their TPEN runtime is
qualified.

**How much of ARM T is production code.** More than the word "fixture"
suggests. `tpen/checkpoint/artifact.py`, `catalog.py`, `reference.py`,
`manifest.py`, `receipt.py`, `hashing.py` and `schema.py` were **measured**
torch-free at runtime on 2026-09-09 (each imported under the project
interpreter with `torch not in sys.modules` asserted afterwards). So generation
selection, completeness, `.tmp` rejection, `CheckpointRef` identity, catalog
publication, `latest.json` and the publication receipt are all exercised as
**production code**, locally, on a host with no torch installed.

**What is a disclosed replica.** `save_checkpoint` writes its payload files with
`torch.save` (save.py lines 161, 167, 179, 186). Those four calls, and only
those, are replaced by a torch-free byte writer.
`tests/helpers/chain_resume_spike/fixture.py::BOUNDARY_ORDER` declares the
sequence the replica performs, and
`test_generation_preservation.py::test_the_replica_boundary_order_matches_production_save_py`
re-derives the order from the real `save.py` source. A production reorder turns
that test red rather than silently invalidating the claim.

**Disclosed fixture simplification.** ARM T's parameter update depends on the
global-RNG limb only; it does not read the walkers. Real VMC couples them, and
under coupling a disabled sampler limb would perturb every downstream channel at
once. The decoupling is what makes the limb→channel map disjoint and therefore
what makes per-limb attribution meaningful. **The attribution claim is a claim
about this fixture, not about production TPEN.**

### ARM N — native TPEN, Cannon allocation only

`tests/helpers/chain_resume_spike/native_probe.py` plus
`tests/integration/chain_resume_spike/test_native_reference_arm.py`.

Drives the shipped VMC smoke configuration through `run_from_config` with the
**real `VMCTrainer`, real `MetropolisSampler`**, a real optimizer, and a
**TEST-HELPER ansatz** (qualified below) —
`max_steps=6`, a checkpoint at step 3, and a resumed arm continuing from it.

**WHAT THIS ARM DOES NOT RE-PROVE, stated so it claims no credit for existing
coverage.** `tests/integration/training/test_train_runner.py` already
establishes native bitwise resume equivalence on real components, compared on
values and on byte-identical `train` metric lines. ARM N adds the three
dimensions that test does not cover:

1. **Every resume is a fresh OS process.** That test runs all of its arms inside
   one pytest process, where a live sampler and its generator survive in memory.
   A chain link is a new process on a new allocation.
2. **The disable is at the apply seam.** That test perturbs the *saved* sampler
   bytes; ARM N skips the *restore call itself*, which is what a code omission
   does and what DS-A0 actually did.
3. **Which stream is load-bearing**, below.

#### Which stream has to survive a job boundary

Derived from the source **before** any run, not fitted to a measurement. Every
draw in the sampling path passes the sampler's own private generator —
`metropolis.py` **188** (initial positions), **255** (the `move.propose` call)
and **298** (acceptance uniforms) — and `moves.py` 39/93/98 take the generator as
a parameter, that module owning no RNG at all.

The **step-level** claim is production's own, not an inference from the sampler.
`tpen/training/update.py` **166-185** declares it as a precondition, AST-resolved:
no forward reached by `model(batch)` is stochastic (the only RNG in `tpen/nn/` is
`initialization.py`, and there is no `Dropout` in the package), and the
local-energy path draws nothing. That docstring also names the scoping trap this
arm would otherwise have fallen into — a claim scoped to `tpen/physics/` could
not see a stochastic layer added in `tpen/nn/`.

**Precision — `path_aggregation.py:211` is INAPPLICABLE to this arm, not merely
configuration-dependent.** The distinction matters: *configuration-dependent*
suggests a setting that could flip, while **inapplicable** says the code path is
not in this arm's program at all. An earlier version of this section said the
global stream *is* consumed at construction, citing `path_aggregation.py:211`'s
bare
`nn.init.xavier_uniform_(weight)`. **That call is never reached by this arm.** It
sits inside `if self.initializer is None`, in a module this arm never loads: the
configured model is
`tests.helpers.trainable_wavefunctions.TrainableHookeSingletAnsatz`, which
imports only `tpen.data.batch` from `tpen` and contains no `xavier`, `nn.init`,
`initializer`, `PathAggregation` or global-RNG use at all.

`native_probe` loads `vmc_smoke.yaml`, whose `_target_` is that test ansatz, and
**no configuration of this arm reaches `PathAggregation`** — so there is no
setting under which line 211 becomes live here.

So for this arm, **nothing consumes the process-global stream at construction OR
during a step** — which is why the inert result is what it is, and it is a fact
about the fixture rather than a measurement of production.

**THE FLIP CONDITIONS, both real, neither exercised here.** A configuration that
*does* reach a bare initializer consumes the globals at construction — harmless
for trajectory, since construction precedes restore and the restored
`state_dict` overwrites those weights, but it moves the stream's position. And
if any future code draws from the globals *after* construction, the asymmetry
inverts outright. The finding is a latent property held in place by the absence
of a post-construction global consumer, not a guarantee.

**REAL-MODEL QUALIFICATION.** The ansatz above is a **test helper**, not a
`tpen/nn` production model. ARM N therefore establishes native **trainer,
sampler, optimizer and checkpoint replay**; it does **not** establish
production-model replay. The basis is still the repo's own native footing —
`test_train_runner.py` loads the same fixture — but no reader should infer a
production ansatz was replayed.

| Seam skipped | Expected | Why |
|---|---|---|
| `_load_sampler` (restore.py:207) | **diverges** | restores the operative stream and the walkers |
| `apply_rng_state` (restore.py:208) | **inert** | no training step consumes the globals it restores |

**The second row is a finding, not a nuisance.** `rng.pt` is load-bearing for
the restore-*refusal* gate (`require_restorable_rng_state`, and
`test_train_runner`'s missing-RNG arm) but **not** for trajectory in this
configuration. A chain backend must preserve the sampler's own state; restoring
the process globals alone would not be enough. If that arm ever goes red,
something in the training path has begun drawing from the globals — the test
says so in its own failure message, and that is a finding about production
rather than something to silence.

Both mutation arms assert a **witness that the no-op was actually invoked**.
Without it, a genuinely inert seam and a patch that never applied are
indistinguishable, and the `inert` arm would pass for entirely the wrong reason.

#### The C10 residual, in two parts, with the bound that produced it

The order machinery is two checks with different exposures, and conflating them
overstates both:

* The **order assertion** derives its record from **inside a wrapper on the real
  production callable**, so a reorder of the wrapped calls moves the record with
  it. It is **wrapper-dependent and evadable** — a write that does not route
  through the wrapped binding is invisible to it.
* The **no-unbacked-pointer assertion** installs nothing and reads only
  `latest.json` and `publications.jsonl`. Nothing to bypass — but it is
  **end-state-only**, sampling after a completed publish and after each
  post-commit interruption.

**THE BOUND, stated as a bound rather than as "C10 is broken": the checks
discriminate for operations routed through the wrapped bindings, and are blind
to out-of-path writes.** Positive controls establish that they *do*
discriminate — a **duplicate** catalog call fails the order node; an **omitted**
catalog call fails the order node *and* the pointer-state node. A verifier then
produced a transient unbacked pointer via an early write on a different import
path, and both nodes passed: out of path for the first, gone before the second
looked.

**What closes the gap is the instantaneous window**, observed with no installed
instrument *during* a publish. That arm is designed in note
`c10-observable-order-design` and is **NOT BUILT** — a budget decision, recorded
here so it is a named residual rather than a silent hole.

**THE FULL NAMED-AND-NOT-BUILT INVENTORY, in the repo artefact rather than only
in the tracker.** Three arms are designed and unbuilt, and a residual that lives
only in a tracker note is a residual R1/R2/R3 will not find:

1. **The instantaneous-window arm** above — closes C10's transient gap.
2. **The globals-channel control** for ARM N — moves the observable *through*
   the channel the inert arm disables, and is what would convert that arm's
   silence from a declaration into a measurement.
3. **CRASH SAMPLING.** Sampling durable state *from outside the process* while a
   publish is in flight, rather than at chosen moments inside it. Every
   process-death arm here kills at a **fixture-chosen boundary**, so the deaths
   are sampled from a set this lane picked; a real crash is not. **This arm
   appeared nowhere in this README until the round-5 sweep put it here** — it
   was named in the implementation receipt and in tracker notes, which is
   exactly the replacement-without-retirement failure the sweep exists to catch.

None of the three is a defect in what was built. Each is a bound on what the
built thing establishes, and each is stated so that a downstream lane inherits
the bound rather than the impression.

**And the witness has its own control**, because otherwise its correctness is
itself unwitnessed — the same recursion as the disable-verification problem.
`tests/unit/chain_resume_spike/test_native_probe_witness.py` proves the witness
stays **empty** when a bypass that is *not* the instrumented one runs. That is
what makes "the witness fired" falsifiable, and therefore what makes the
non-divergence arm mean anything.

**MONKEYPATCH DISCLOSURE.** Production exposes no flag to skip one restore limb,
so the arms rebind `tpen.checkpoint.restore.apply_rng_state` and
`tpen.checkpoint.restore._load_sampler` at their call sites, from test code. **A
monkeypatch is not a production seam.** It reproduces what a code omission would
do; it does not show that production has, or should have, a switch there. No
file under `tpen/` is modified.

#### MEASURED ON CANNON — ARM N IS NOW COVERED FOR G0

Slurm job **45645515**, partition `test` (requested and delivered), node
`holy8a24401`, 4 CPU / 32 GiB / 45 min wall, elapsed **00:04:50**. Scheduler
state **COMPLETED**, `ExitCode 0:0`; inner `NATIVE_PYTEST_RC=0` and
`ARMT_PYTEST_RC=0`. Scheduler state and inner exit are recorded as separate
fields; both are **EARNED** — the script exits with the test status and nothing
exits 0 deliberately.

In-job provenance, asserted inside the allocation rather than inferred from the
submitting shell: `CHECKOUT_SHA=179eb0b965860b217be39670c3cd49e041e627fc`,
Python 3.12.13, torch 2.12.0+cpu, CUDA unavailable, numpy 2.4.6, interpreter
under the job-local Netscratch venv with no interpreter pinned and `uv` invoked
by absolute path.

**Native arm: `tests=5 failures=0 errors=0 skipped=0`** (junitxml). All five
nodes executed — no skips, because torch is present:

| Node | Result |
|---|---|
| in-job interpreter and torch provenance | PASS |
| seam expectation map covers every seam | PASS |
| **fresh-process native resume reproduces the uninterrupted trajectory** | **PASS** |
| **skipping `_load_sampler` moves the trajectory** | **PASS** |
| **skipping `apply_rng_state` leaves it unchanged** | **PASS** |

**The source-derived prediction held in both directions.** The sampler seam is
the operative stream and the global-RNG seam is inert for trajectory — measured,
not assumed, and predicted before the run rather than fitted to it. Each
mutation arm's witness confirmed the seam was genuinely reached and bypassed, so
the inert result is not a patch that failed to apply.

**ARM T re-run in the same allocation under torch PRESENT:
`tests=112 failures=0 errors=0 skipped=0`.** This matters specifically for the
torch-free import test: on the workstation torch is absent, so a stray
`import torch` in a shared module would fail as a *missing module*; here it
would land in `sys.modules` and be caught as the *property violation* it is.
That condition cannot be reproduced locally, and it cost no extra allocation.

Evidence preserved under the job's Netscratch log directory (`.out`, `.err`, and
both junitxml files) and under the superseded job 45615268's directory. No
cleanup without an explicit lifecycle disposition.

#### What ARM N establishes, decomposed into four buckets

A single verdict hides which clauses are load-bearing. Read the arm as four
kinds of statement, not one:

**MEASURED — with a witness.** Skipping `_load_sampler` moves the trajectory,
and the bypass witness proves the seam was genuinely reached and skipped. The
green arm reproduces the uninterrupted trajectory byte-identically on train
metric lines, plus model digests and both durable counters. **This is the half
R1/R2/R3 need: the sampler's own state must survive a job boundary.**

**DECLARATION — production's own, to which this arm lends no evidence.** "No
global draw occurs during a training step" rests on `update.py` 166-185, which
declares it as an AST-resolved precondition. The arm does not test it. It was
previously reported as though the measured arm supported it; it does not.

**NEVER EXERCISED in this fixture.** Construction consuming the globals
(`path_aggregation.py:211`, the default-initializer path). This arm's configured
model never reaches that module, so the case is untouched here — neither
confirmed nor contradicted.

**NAMED AND NOT BUILT.** A control that moves the observable *through the channel
the inert arm disables*, **inside ARM N**. That is what would convert the inert
result from a declaration into a measurement. Its absence is why the bucket
exists. **Inside ARM N the arm could not detect a global-RNG omission even if
one existed**, because there is no consumer and so no channel through which the
observable could move. That is a statement about *this arm*; the next paragraph
draws the line against the lane.

**AND THE REASON THE FOURTH BUCKET IS NOT EMPTY IS WORTH THE SPACE: SHARING AN
OBSERVABLE IS NOT SHARING A CHANNEL.** All three arms compare the same
observable, which is necessary and was verified by three parties. But the
diverging arm moves that observable through the **sampler's private generator**,
while the inert arm makes a claim about the **process globals**.

**WHAT ARM N LACKS IS A CONSUMER, NOT A DETECTOR.** State this precisely,
because the loose version understates the lane and an understatement is as false
a record as an overstatement. The globals channel **does** have a demonstrated
mover — in ARM T. `test_disabling_one_limb_diverges_in_that_limb_own_channel`
runs `GLOBAL_RNG` as one of its arms and requires the divergence to land in a
channel that limb actually feeds, and
`test_the_channel_map_partitions_every_observable_channel` establishes the map
is disjoint and total, so `GLOBAL_RNG`'s `{noise_term, parameters}` cannot be
mistaken for another limb's redness. **The lane can detect a globals omission
wherever a consumer exists.** ARM N's configured ansatz simply never draws from
the globals, so in *that* arm there is nothing for an omission to perturb.

Both halves have to be said, because each alone is misleading. Its inert result
is **fixture-guaranteed** — a fact about this arm's model, **not** a limit on
the lane's ability to detect a globals omission. And **the same fact that
explains that result disqualifies it as evidence**: an absence can only fail to
contradict a capability claim, never support one. So dropping the
`path_aggregation.py:211` caveat removes a distractor and **adds no sensitivity
evidence**. On the globals channel, ARM T carries the weight and ARM N
contributes nothing.

Three independent verifications all confirmed the stated requirement and all were
correct. **Independence of party is not independence of question**: each checked
the same condition, so concurrence measured execution rather than sufficiency.

#### The record-binding audit: can the action move without moving the record?

The claims sweep is a **prose** instrument. This is a **structural** one, and it
finds things prose cannot, so it is run separately over every record-or-witness
mechanism in the lane rather than only over the one whose defect was being
chased. That scoping error is itself the finding that prompted it: the question
was asked of the order mechanism and **not** of the coverage mechanism, in the
same file, in the same round.

**A record is supposed to witness an act, and that relationship has two ends.**
Mutating the **record** — log a different name, suppress the entry — proves the
record is *consumed*. Mutating the **act** — move the real operation, leave the
record correct — proves the record is *bound to it*. **Either result alone is
consistent with a working mechanism**, so value-level bidirectionality, however
rigorously run, cannot separate them: present/absent and on/off both live at the
record end.

| mechanism | can the act move without the record? | status |
|---|---|---|
| `record_fire` (`faults.py`) | **YES** — it records the *plan's* identity, so exchanging two real injection sites leaves the set unchanged | known limit, stated at the instrument |
| `boundary_log` (`fixture.py`) | **YES** — a self-report emitted beside the operation | known limit; this is the original C10 finding |
| `_observe_real_operation_order` wrappers | **NO** for wrapped calls — the record is *inside* a wrapper on the real callable | bound, with the out-of-path write named as its bound |
| `_SEAM_INVOCATIONS` (`native_probe.py`) | **NO** — the record is emitted *by* the patched no-op, so the record **is** the act | bound; a missed patch fails loudly rather than silently |
| `LimbApplication.consumed` (`restore_limbs.py`) | **NO** — derived by fingerprinting the live object against the checkpointed fingerprint, never by echoing the requested flag | bound by construction, and separately tested |
| ledger `credited` (`entrypoint.py`) | **N/A** — the record *is* the credit; there is no separate act to displace | not a witness relationship |

**RESULT: NO NEW MECHANISM DEFECT.** Two mechanisms are separable and both were
already carrying that limit at the instrument; three are bound, one by
construction; one is not a witness relationship at all. **Nothing here required
a mechanism change** — consistent with the claims sweep, which also found only
claims wider than correct checks.

**FOR R1/R2/R3, AS A CHECK TO RUN RATHER THAN A RESULT TO INHERIT:** for any
check of the shape *X records or witnesses Y*, write the mutant **pair**
explicitly — one that moves X, one that moves Y — and state what each
establishes. Then ask it of **every** mechanism in the file, not just the one
under suspicion.

#### The claims sweep: every assertive claim in the lane, classified

Four universals were attacked in verification and **all four were false as
written**. That is a population problem, not four incidents, so the response was
to sweep the population rather than patch the found ones. A fifth fell out of the
sweep, in a file nobody had flagged.

**COVERAGE, stated as examined-over-total, because a partial audit that does not
say so IS the defect.**

**SWEPT BY CLAIM SHAPE, NOT BY KEYWORD.** A keyword vocabulary is one
instrument, and it was measured to miss a real find: of the two retired wordings
still live in this file at `c73ad66`, the regex caught one via `proves` and was
**blind to the other**, which carries no assertive keyword at all. So the
keyword count is a **floor, not the population**, and bounding the sweep by it
would inherit the vocabulary instead of sweeping the corpus. The shapes swept
are the ones that have actually produced false claims in this lane:

| claim shape | lines matched |
|---|---|
| instantaneous-vs-sampled (`at the moment`, `while`, `during`, `when`) | 97 |
| entry-vs-completion (`reached`, `entered`, `invoked`, `takes effect`) | 96 |
| requested-vs-occurred (evidence from a plan, label, record or name) | 234 |
| declared-vs-measured (resting on production's own statement) | 32 |
| universal vocabulary (the original keyword floor) | 578 |
| **union — the population** | **881** |

| population | total | examined |
|---|---|---|
| lane files, Python **and Markdown** | 25 | **25 (100%)** |
| prose lines scanned by shape | 3001 | **3001 (100%)** |
| node names carrying a universal | 17 | **17 (100%), each body opened** |
| claim lines individually read | 129 | **129 (100%)** |

**AND THE LIMIT, because this census has one.** The corpus holds **433 claim
sites** (224 Python docstrings plus 209 Markdown paragraphs), of which **142
make a claim** (79 docstrings, 63 paragraphs). Every one was shape-scanned;
**not every one received an individual four-way verdict.** The individual
verdicts cover the 17 universal-bearing node names in full, the 129 flagged
claim lines, and every site the shape scan raised in the two highest-risk
categories. **A reader should treat the four counts below as complete for the
node-name population and as a floor for the prose.**

**CLASSIFICATION OF THE 17 NODE NAMES** — the high-yield class, since every
confirmed defect lived there rather than in body prose:

- **ESTABLISHED — 8.** The universal is true over a closed, enumerated
  population that the body actually iterates: the fault-point registry, the
  channel map, `RestoreLimb`, the outcome-kind enum, `DISABLEABLE_SEAMS`, the
  empty-witness node, the SIGKILL survival node, and the no-temp-file node whose
  end-state claim matches its end-state check.
- **RE-SCOPED — 4.** Universal claimed, weaker property checked. The C10 pointer
  node; `test_the_committed_receipt_parses_as_one_whole_document`, formerly
  asserting a torn read is *never observable* while writing and reading at one
  instant with no partial state and no concurrent reader;
  `test_reconciliation_does_not_rewrite_the_committed_payload_bytes`, formerly
  claiming a committed generation is *never* destructively edited while
  exercising exactly one operation; and the C8 node, whose **name** still carried
  the universal **its own docstring had already retracted one round earlier**.
- **BOUNDED — 2.** The universal holds over a named population the docstring now
  states: `any random stream` means the three `RestoreLimb` members, and
  `every shared module` means the hand-maintained `SHARED_MODULES` list, not a
  computed set.
- **NOT CLAIMS — 3.** `credit_all`, `all_enabled`, `all_disabled` are helper
  operation names.
- **UNESTABLISHED and deleted — 0.**

**THREE FINDS THAT LIVED OUTSIDE EVERY EARLIER DIFF RANGE.** Two retired
wordings were still asserted in this file while their corrected versions were
asserted elsewhere in it — *at the moment `latest.json` is published*, and *the
fire record proves injection-site entry* — so **the artefact stated both the
correction and the thing it retired**. Replacement without retirement. Both are
corrected above, in place, with the retired wording named so the diff is
legible. The third was an absence: **crash sampling appeared nowhere in this
README**, living only in the implementation receipt and tracker notes. A
residual recorded only in a tracker is a residual a downstream lane never finds,
so the full named-and-not-built inventory is now in this file.

**NO MECHANISM CHANGE WAS REQUIRED, and that is a finding rather than a
convenience.** Every defect the sweep found was a claim wider than a correct
check. Not one instrument was measuring the wrong thing; each was measuring a
real property and describing it too broadly.

**THE PATTERN, for R1/R2/R3: the name outlives the correction.** Three of the
five re-scopes were nodes whose prose had already been fixed while the
identifier kept asserting the retracted universal — the C8 node did it inside the
file this same round corrected. **A node id is an interface**; a reader who
never opens the body sees only the name, and a suite summary quotes nothing else.
Re-scope both, in the same change, or the artefact still lies.

#### A limit of how this lane was built, which R1/R2/R3 will inherit

Three times in this lane a finding was recorded and then **not applied one file
over**: ARM T's limb→channel map was not carried into ARM N; the C10
record-separable-from-action fix was not turned on C9's attribution; and a design
note saying precommit damage assertions cannot use `RAISE` was not turned on the
coverage instrument in that same file. Each time the knowledge existed and was
written down.

**A finding stays scoped to the artefact you were looking at when you made it.**
It is not landed when it is recorded — only when it has been applied to every
instrument already in scope, and this lane had no step that did that. Each
candidate lane will build two arms; the gap will be between them.

#### The earlier stand-in arm, and why it was replaced rather than repaired

The first ARM N used minimal stand-ins and executed on Cannon as job 45615268
(delivered `kozinsky`, scheduler `FAILED 1:0`, inner `PYTEST_RC=1`, 2 passed / 3
failed). **The two that passed were the provenance and seam-map nodes — neither
is a parity claim, so "2 passed" was never evidence about native resume.**

Root cause, and it **exonerates production**: `restore.py:137` calls
`_verify_hash` for `model_config` unconditionally, `hashing.py:94-99` builds that
hash from the model config, and the shared test run context supplies an *empty*
config — so no `model_config` hash was written and restore correctly refused. **A
restore path that fails closed on an under-specified manifest is exactly the
behaviour the candidate lanes will want to rely on**, and it is recorded here as
a positive result rather than buried as the cause of a red.

That arm was replaced rather than patched: a real config supplies `model_config`
naturally, and patching the stand-in would have produced a green arm whose
greenness meant nothing about native trajectory exactness — the toy-only claim
the shared contract forbids.

---

## Why the gate is shaped the way it is: the DS-A0 failure

Spike DS-A0 (`8f0bdfcd`, tip `eaa89605`, Cannon job 45092878) saved RNG state,
never restored it, and **its deterministic parity assertion passed anyway**. Its
non-vacuity control inherited the same blindness. Every rule below exists to
make that specific failure impossible to repeat:

1. **Parity is a function of subsequent draws.** Next proposals, acceptance
   uniforms, walker positions, the parameter noise term, cadence measurements —
   never serialized state bytes, never deterministic outputs alone.
2. **Every resume attempt is a fresh OS process,** spawned with an absolute
   `sys.executable`. In-process resume is forbidden as vacuous: a live RNG
   object surviving in memory makes the gate pass for free. This applies to
   ARM N too.
3. **The fresh process seeds itself from OS entropy and records the seed,** so
   "the resume did not accidentally reproduce the parent" is a checked fact.
4. **Three independent entropy consumers,** each landing in the compared trace.
   A limb whose output never reaches the trace has a vacuous mutation arm.
5. **The disable is a flag at the apply seam,** skipping exactly that limb's
   apply call — what a code omission does. Not an edit-and-revert, which is not
   re-runnable and can be left half-restored.
6. **The disable is verified.** `apply_limb_states` returns a `consumed` map
   derived by fingerprinting the *live* objects after the seam, never by echoing
   the flag. An unverified disable can invert into a strengthening and go green.
7. **One limb per arm, plus an all-disabled arm.** Two clauses sharing one
   falsifier mask each other.
8. **Each arm asserts the channel, not just redness.** Redness is produced by
   every limb and by unrelated bugs; only the channel says which mutant ran.
9. **The green arm runs first.** A red green-arm invalidates every mutation arm.

### The three limbs and the channels they feed

| Limb | Real analogue | Channels |
|---|---|---|
| `SAMPLER_GENERATOR` | `MetropolisSampler`'s private generator, in `sampler.pt` | `proposal`, `acceptance_uniform`, `walkers` |
| `GLOBAL_RNG` | Python + NumPy legacy globals, restored by `apply_rng_state` | `noise_term`, `parameters` |
| `CADENCE_RNG` | `StepCadenceGate`'s `_rng` **and** `num_calls` — absent from TPEN's checkpoint inventory entirely | `measurement_gate`, `measurement_value` |

The map is asserted disjoint and total by a test, because disjointness is what
makes attribution possible and totality is what makes every limb observable.

---

## G6: fault points at the real commit boundaries

**Correction recorded deliberately.** The lane design note originally named a
fault point `after_latest_before_catalog`. Production writes the **catalog
before `latest.json`**. Measured order in `tpen/checkpoint/save.py`:

```
tmp_dir.rename(final_dir)              save.py:211   <-- THE COMMIT
CheckpointRef.from_directory(...)      save.py:216
catalog.publish(ref)                   save.py:226
write_latest(root, final_dir, ...)     save.py:229
record_publication_receipt(...)        save.py:237
```

So `after_latest_before_catalog` named a state `save_checkpoint` never passes
through, and a test against it could not have exercised real code. It is
replaced by the three points that do occur: `after_rename_before_catalog`,
`after_catalog_before_latest`, `after_latest_before_receipt`. Together these are
the **committed-but-unacknowledged** family, and
`tpen.checkpoint.catalog.reconcile_publication` (catalog.py:186) is production's
repair primitive for exactly it — the tests assert its documented semantics
(catalog.py:189-208) rather than a guess.

The registry is closed and partitioned. A coverage test asserts every
`MEASURED` point is referenced by at least one lane test module, scanning the
AST so a mention in a docstring does not count as coverage, and **excluding the
coverage module itself** so it cannot satisfy itself.

### The non-unwinding gap, measured

`save_checkpoint`'s `finally` (save.py:245) clears the temporary directory when
the interpreter unwinds. Its own note at save.py:255-260 says what it does not
cover. Measured at this revision, per fault action:

| Action | Child exit | `.tmp` residue | Receipt written |
|---|---|---|---|
| `RAISE` | 1 | no — `finally` ran | yes (`TRANSIENT`) |
| `OS_EXIT` | 70 | **yes** | no |
| `SIGTERM_SELF` | -15 | **yes** | no |
| `SIGKILL_FROM_PARENT` | -9 | **yes** | no |

The residue is guaranteed rather than incidental for the three non-unwinding
actions, so the tests assert it unconditionally. The distinction between "a
receipt saying it failed" and "no receipt at all" is the distinction between an
error and a death, and a chain controller has to tell them apart. Race-sensitive
arms are repeated three times; that bounds an anecdote, it is not an
availability figure.

---

## The spawn hard clause

Every subprocess this package starts passes an **absolute `sys.executable`**.
`tests/unit/chain_resume_spike/test_spawn_uses_absolute_interpreter.py` enforces
it structurally rather than by writer discipline, with three rules covering
*passing* a bare name, *accepting* one as a parameter default, and reaching the
program through a shell.

This exists because `experiments/baselines/test_scaling_probe.py` builds
`["python", "-c", ...]` (line 167 at this revision) and spawns it, inheriting
the job `PATH`. That is the one **pre-existing trunk red**, red at both base and
tip, owned by the `experiments/baselines` owner. This lane **discloses it,
excludes it from new-failure comparison, and does not fix it.** Its presence in
a base arm is the cheap instrument check proving a failure extractor can see
failures at all.

The detector was pointed at that real file and correctly reported
`experiments/baselines/test_scaling_probe.py:167: a command sequence starts with
the bare interpreter name 'python'`. That check was run once by hand and is
**not** committed as a test, because the file belongs to another owner and
fixing it must not break this lane.

---

## Running the arms

ARM T, locally, with the project interpreter:

```bash
uv run pytest tests/unit/chain_resume_spike
```

ARM N requires `torch` and therefore a Cannon allocation. `pytest` is compute
and must not run on a login node. Read the `cluster-access` skill and its full
current note chain before any facility action; interpreter, workspace root and
cache locations are runtime inputs and are never hardcoded in this repository.
Assert `sys.executable` and the torch version **in-job** — `native_probe`
records both into its evidence file for exactly that reason.

---

## The selection surface, and why it is the one to assert on

**Production resume follows `latest.json` and nothing else.**
`restore_checkpoint` reaches its checkpoint through exactly one path:
`restore.py:118` calls `resolve_checkpoint_dir`, which reads `latest.json`
(artifact.py:68 and :71). Measured at this revision: `list_complete_checkpoints`
has **no caller** anywhere in `tpen/checkpoint` outside its own definition and
the package re-export. **The production resume path never lists directories.**

That matters because the two surfaces **disagree** in the two pre-`latest.json`
fault windows. After a fault in `after_rename_before_catalog` or
`after_catalog_before_latest`, generation 2 is complete on disk — it has its
`COMPLETE` marker and no `.tmp` suffix — so a directory listing returns it,
while `latest.json` still names generation 1 and resume therefore takes
generation 1 and replays the tail.

The outcome is safe, but **it is safe because of which surface is consulted, not
because of the directory state.** The fixture therefore selects through
`resume_generation` (the pointer), keeps `newest_listed_generation` as a
separate named function, and pins the disagreement as an explicit property.

**USAGE RULE FOR R1/R2/R3.** A controller that picks up "the newest complete
directory" selects a generation the production resume path would not. Read the
pointer.

## Two hazards found by this lane. Disclosed, pinned, NOT fixed

Both are in `tpen/checkpoint`, which is outside this lane's write surface. No
`resolve()` and no monotonicity guard were added to `tpen/`.

### 1. An orphaned post-rename generation deadlocks the chain

After an interruption between the rename and the catalog append, resume does the
right thing at every individual step and still cannot make progress:

1. it follows `latest.json` to generation 1, correctly;
2. it replays the tail, correctly, reproducing the same stream;
3. it then tries to commit generation 2 — **whose directory already exists**,
   because the interrupted attempt renamed it into place.

`save_checkpoint` refuses with `FileExistsError: checkpoint already exists`
(save.py:142-143). The direction is **fail-closed** — loud, nothing
overwritten — but a chain that cannot advance a single further generation is
exactly the failure this program exists to prevent, and **nothing performs the
recovery automatically.**

**USAGE RULE FOR R1/R2/R3.** Before resuming, reconcile any complete generation
newer than the one `latest.json` names. Otherwise the chain restores, replays,
and then dies on the collision.

#### This hazard defeats its own owner's acceptance predicates

The interruption-safety item (`3b9b736a`) specifies its layer-3 acceptance
experiment as: **no `step_` directory lacks `COMPLETE` or `manifest.json`, and
no `.tmp` residue survives.**

**Both predicates pass on the orphan.** Measured, by constructing the state and
evaluating them:

```
step dirs: ['step_000002', 'step_000004']
PREDICATE 1 (no malformed step dir): violations = []  -> PASSES
PREDICATE 2 (no .tmp residue):       residue    = []  -> PASSES
pointer still names step_000002, newest complete dir is step_000004
```

The reason is structural. `save.py` writes `manifest.json` (209) and `COMPLETE`
(210) **before** the rename (211), and `write_latest` does not run until 229. So
the orphan is a **fully formed final directory carrying both files**, and there
is no `.tmp` anywhere because the tmp directory *was* the thing renamed.

**That item would have run its own acceptance experiment, reported clean, and
shipped the deadlock.** Its predicates look for a *malformed* directory and for
*residue*. The orphan is neither — it is a well-formed directory that is simply
unreferenced. An acceptance contract that enumerates the damage shapes its
author imagined cannot see a shape that is not damage at all.

**And the remedy that looks like it covers this does not.** That item's layer-2
work on `SIGTERM` unwinding **does not narrow this class at all**: the `finally`
at save.py:245 removes `tmp_dir`, and post-rename `tmp_dir` does not exist —
save.py's own comment at 252 says so, *"the rename above moves the directory, so
`exists()` is already False"*. Making unwinding reliable is a **no-op** for this
state. Stated explicitly because a reader who knows a `SIGTERM` fix is planned
would otherwise assume this class is already covered by it.

For citation accuracy: that `finally` **is** present at `dev` and the item's
layer-1 fix **is landed** — the comment at 246-250 records that it was changed
*from* `except Exception` precisely because `KeyboardInterrupt` and `SystemExit`
derive from `BaseException`. Layer 1 is not open.

#### Two routes to "cannot advance", enumerated rather than counted

Evidenced at `4a580e6` **by state construction**. Completeness is **UNMEASURED**.
Stated as an enumeration with discriminators, not a tally: a count cannot be
checked by a reader, so they can spot neither a missing route nor an over-folded
one.

**ROUTE 1 — PUBLISH CONFLICT.** catalog.py:77-80, `ValueError conflicting
checkpoint publication for content_id`. Fires ONLY when `existing.content_id ==
ref.content_id` **and** `existing.to_dict() != serialized_ref`. Trigger:
republishing the same content identity under a **different path spelling**. It
never consults the filesystem.

**ROUTE 2 — ORPHAN NAME COLLISION.** save.py:142-143, `FileExistsError checkpoint
already exists`. Fires on `final_dir.exists()` **alone** — a filesystem name
collision with **no content comparison**. Trigger: an interrupt between the
rename at 211 and the pointer update at 229, with no path weirdness at all.

**They are not one mechanism at two altitudes**, and the discriminator is a
control-flow fact rather than a judgement: **line 142 precedes 211, 226 and
229**, so in the orphan state `save_checkpoint` raises at 143 and **never
reaches** `catalog.publish` at 226. Route 1 is structurally unreachable from
route 2's state. Two failures cannot be the same mechanism when one preempts the
other.

The reconcile **rewind** is deliberately **not** counted here: it does not fail
to advance, it **advances from the wrong place**, re-running committed science.
Different signature, different cost.

**How to look for a third**, because this is the part that transfers: **both
routes were found by BUILDING REAL STATES, not by reading the source.** Route 2
surfaced only because a repair item forced construction of a genuine
overlapping-generation state — the hand-sliced ledger test it replaced could not
have produced it. A reader who wants to know whether a third route exists must
construct states, not grep.

#### Bounds on these pins — limits, not defects

* The orphan-deadlock node establishes **failure to advance at the collision**.
  It does **not** compare the failed replay's trace, and it does **not**
  byte-hash both generations.
* The recovery node establishes **manual reconciliation recovery, NOT automatic
  recovery** — it calls production `reconcile_publication` on the orphan
  explicitly. **Do not read it as the chain self-healing.** For a candidate lane
  deciding whether its controller must do this itself, that distinction is the
  whole decision.
* The rewind node does assert complete before/after byte maps of **both**
  generations unchanged.

### 2. `reconcile_publication` rewinds the resume pointer

`reconcile_publication` builds its expected pointer from the directory it is
**handed** (catalog.py:216-233), compares `read_latest` against it, and on any
mismatch writes the pointer at that directory — **with no monotonicity guard on
step.** Handed an older generation while `latest.json` names a newer one, it
moves the pointer backwards, and since resume follows the pointer, that means
re-running committed science.

**This is not a defect in current usage** — today's only caller reaches it with
the newest directory, so the hazard is held off by an accident of usage rather
than by a guard. **The trigger is the usage pattern this program is
evaluating:** a chain controller that restarts and reconciles the generation it
*remembers* handing off can hand it an older one and silently rewind past a
newer committed generation.

The subsystem documents this about itself. `CheckpointCatalog.iter_publications`'
torn-row repair message says, in production code, that `reconcile_publication`
*"rewrites latest.json unconditionally and would point it at an older
checkpoint"*, and tells operators **"Do NOT reconcile older directories to be
safe."**

**USAGE RULE FOR R1/R2/R3.** Never call `reconcile_publication` with a
generation older than the one `latest.json` currently names. A controller that
must reconcile a remembered generation has to compare it against `read_latest`
first.

## Path spelling: restore is spelling-independent, publish is not

Nothing in `tpen/checkpoint` canonicalises a path — no `resolve`, `realpath`,
`samefile`, `abspath` or `absolute` in any of its seven modules. (Naming trap:
`artifact.resolve_checkpoint_dir` does *pointer* resolution, not path
canonicalisation.)

Equivalent spellings open the same file, so spelling is harmless wherever a path
is merely opened or stat-ed — and `write_latest` stores a **basename** only, so
**the resume pointer is spelling-independent**. But `CheckpointCatalog.publish`
compares the **full serialized mapping**, which includes `checkpoint_dir`, so a
differently-spelled path to the same directory conflicts. The guard is
deliberate — a relocation must not create two locations for one catalog
identity — but it **conflates a genuine relocation with the same directory
reached by an equivalent spelling**, because nothing resolves.

**The asymmetry is the finding: a chain attempt reaching the same root by a
different spelling restores fine and then refuses to commit its next
generation.** Fresh processes are exactly where spelling diverges — different
cwd, relative versus absolute, symlinked scratch roots, a mount presenting
differently in a later allocation — and every resume attempt here is a fresh OS
process.

Measured once, deliberately, in `test_path_spelling.py`, which **exercises
production `publish`, `reference` and pointer code directly** — those modules
are torch-free at runtime, so no replica is in the path. The arm asserts the
exact `ValueError` naming the actual `content_id`, asserts the pointer still
resolves under the same spelling, and includes a control showing that
republishing the *identical* spelling is idempotent — without that control the
refusal would be equally consistent with "republishing anything conflicts".
**Every other arm supplies absolute, already-resolved roots** so this fragility
cannot contaminate the parity or G6 results.

## The torch-free property is enforced by a test, not a convention

`test_torch_free_import.py` imports each shared module **in a subprocess with a
clean `sys.modules`** and asserts torch is absent afterwards. An in-process
check would be vacuous exactly when it matters, because the native arm imports
torch into the same pytest process — the same failure class as an in-process
resume. `native_probe` is exempt and keeps its torch imports function-local,
matching the `tpen/checkpoint/rng.py` 100/150/222/271 precedent.

This is a test rather than a note because the property dies **silently**:
measured in the previous round, an `import torch` inserted into shared
`fixture.py` left all 88 unit tests green.

**LIMITATION OF THE LOCAL ARM, and it inverts the usual assumption: the
torch-free property is only properly tested WHERE TORCH EXISTS.** On a host
without torch, a stray `import torch` in a shared module fails as a **missing
module** — so the test goes red for the *wrong reason*, and would go red
identically for a typo. Only where torch is **present** does the import land in
`sys.modules` and get caught as the property violation it is. The development
workstation cannot reproduce the condition this test exists for; Cannon job
45645515 re-ran the arm under torch 2.12.0+cpu for exactly that reason
(112/112, 0 skipped).

## How the fixture's own checks are kept honest

Three of this lane's checks previously observed a **proxy** rather than the
property, and all three were replaced with runtime observation:

| Check | Old mechanism | Why it failed | Now |
|---|---|---|---|
| Fault coverage | scan test sources for `FaultPoint.MEMBER` | an unused enum expression counted as an exercise; an equivalent set alias exercised a point invisibly | each point is **driven and observed firing**, recorded at injection time |
| Replica order | compare the declaration against `save.py` source | a mutant reordering the replica's **actual** publish/latest calls survived | `publish_generation` **records each boundary as it executes**; observed == declared == production-derived |
| Torch-free | (absent) | — | subprocess import with a clean `sys.modules` |

A mention cannot fire, and an alias fires identically, so runtime observation
closes both coverage defects at once.

**AND THEN TWO OF THOSE THREE REPAIRS WERE THEMSELVES STILL PROXIES**, which is
the more useful lesson: replacing a declaration with a self-report is not the
same as observing the thing.

* **Order.** The runtime pin recorded boundaries the fixture *emitted about
  itself*, and those emissions are separable from the operations: a mutant that
  moved the real `catalog.publish` past `write_latest` while leaving the labels
  in place passed the static pin, the runtime pin, and all 112 nodes. The order
  is now derived from **inside a wrapper around the real production callable**,
  so displacing the call necessarily displaces its record, and separately from
  **observable filesystem state** — sampled immediately before the pointer write
  executes, at which instant the catalog row must already be on disk. Neither
  reads a label. On that same mutant the two self-report checks still pass while
  both new checks fail.

  **PROVENANCE, BECAUSE THIS BULLET OTHERWISE READS AS ONE EXECUTION AND IS
  THREE.** The operation-only reorder above was measured at the earlier revision
  — this lane's own run and the verifier's independent `52c84fe` pair, which are
  **separately attributed evidence, not one result.** It was **NOT re-run** at
  the `d22ef8f` pin. What *was* independently measured at that pin is a
  different family: the **out-of-path bypass**, the **duplicate** catalog call,
  and the **dropped** catalog call. Both sets are real; **combining them into a
  single implied d22 reorder execution would be a fourth false universal of the
  provenance kind**, and the count of "112 nodes" belongs to the revision where
  it was taken, not to the current suite.

  **RETIRED WORDING, CORRECTED HERE:** this bullet used to
  say *at the moment `latest.json` is published*, the instantaneous claim C10
  gave up. The wrapper-derived order establishes catalog COMPLETION before
  pointer-call ENTRY, and the state check is END-STATE ONLY, so a transient
  unbacked pointer between samples is **not** excluded.
* **Fault coverage.** The fire record establishes only that **an injection was
  REQUESTED under that identity**: a mutant keeping `record_fire` and returning
  instead of raising passes every coverage node. That claim is **re-scoped**
  rather than annotated, because an artefact saying "exercised" while proving
  something weaker is false in the thing downstream lanes read. **RETIRED
  WORDING, CORRECTED HERE:** this bullet used to say the record proves
  *injection-site entry*, which item 2 retired — **entry implies the site, and
  the site is the unobserved thing.** Nor is BOUNDARY attribution among the
  precommit points merely unobserved: it is **UNACHIEVABLE under `RAISE`**, all
  four aborting before the rename while the replicated `finally` erases
  `tmp_dir`, leaving an identical end state. Effect delivery for
  the seven RAISE-boundary points is established by named nodes in
  `test_generation_preservation.py`; `TORN_CATALOG_ROW`'s effect delivery is a
  **named UNMEASURED limit**, since only a suppressed-RAISE mutant exists and it
  cannot reach a node that truncates the catalog directly.

---

## Attribution, not absorption

Native production gaps found by this lane are **disclosed and attributed to
their existing owners, never fixed here**:

- **HI L5a `e2e512eb`** — factor-rich method and callback replay.
- **`3b9b736a`** — interruption safety. The non-unwinding residue table above is
  evidence for that item, not a defect this lane repairs.
- **`72bc554b`** — semantics-key forward compatibility.

Callback state classification is in `native-state-classification.md`.

No scientific behaviour was altered to obtain a green anywhere in this lane.
