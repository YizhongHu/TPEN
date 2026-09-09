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
workstation. A green local run establishes nothing about the native surfaces.

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

Drives the real `save_checkpoint` and the real `restore_checkpoint` in
`train_resume` mode, including the `torch.save` payload writes and the two
production apply seams `restore.py:207` (`_load_sampler`) and `restore.py:208`
(`apply_rng_state`).

**ARM N may not claim toy-level exact resume of a real TPEN run.** Its model,
optimizer, trainer and sampler are minimal stand-ins satisfying the interfaces
the checkpoint path requires — not `VMCTrainer`, `MetropolisSampler` or a real
wavefunction. Standing those up would make a failure ambiguous between the
checkpoint path and the physics setup.

**MONKEYPATCH DISCLOSURE.** Production exposes no flag to skip one restore limb.
The ARM N mutation arms rebind `tpen.checkpoint.restore.apply_rng_state` and
`tpen.checkpoint.restore._load_sampler` at their call sites, from test code.
**A monkeypatch is not a production seam.** It reproduces what a code omission
would do; it does not show that production has, or should have, a switch there.
No file under `tpen/` is modified by this lane.

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

## Attribution, not absorption

Native production gaps found by this lane are **disclosed and attributed to
their existing owners, never fixed here**:

- **HI L5a `e2e512eb`** — factor-rich method and callback replay.
- **`3b9b736a`** — interruption safety. The non-unwinding residue table above is
  evidence for that item, not a defect this lane repairs.
- **`72bc554b`** — semantics-key forward compatibility.

Callback state classification is in `native-state-classification.md`.

No scientific behaviour was altered to obtain a green anywhere in this lane.
