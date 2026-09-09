# Native mutable-state classification for chain resume

Deliverable of TPEN chain-resume spike **R0** (`ab757fd8`). Classifies every
mutable-state owner outside TPEN's checkpoint inventory as
**scientifically-active**, **reconstructible**, or **per-attempt-diagnostic**,
with `file:line` and the consequence if it is not restored.

**This is a disclosure, not a work item for this lane.** Nothing here is fixed
by R0. Owners are named in the last section.

Measured 2026-09-09 at `4a580e642046a1756afb041a3b8f95a871201389`.

---

## What the checkpoint actually persists

`tpen/checkpoint/save.py` writes `model.pt`, `optimizer.pt`, `trainer.json`,
`sampler.pt` and `rng.pt`. Between them these cover model parameters, optimizer
state, trainer progress and update-method state, the sampler's walkers and
private generator, and the Python / NumPy / Torch process-global RNG streams.

**No callback state is persisted anywhere.** This is not an oversight to be
inferred from a grep; it is stated in the source.
`tpen/checkpoint/payload.py:4-5` declares checkpoint payloads *"deliberately
trainable-free and callback-free"*. A search of the whole `tpen/checkpoint/`
package for `callback`, `cadence` and `num_calls` returns only prose in
docstrings and comments — no field, no file entry, no restore path.

---

## Classification

### Scientifically active — an unrestored value changes what the run does

| Owner | Location | Consequence if unrestored |
|---|---|---|
| `CadenceGate.num_calls` | `tpen/callback/cadence.py:74`, incremented at `:91` | Under a `max_calls` cap, the counter resets to 0 on resume and a callback that had exhausted its budget **fires again**. The admitted set of occurrences across a resumed run differs from an uninterrupted one. |
| `CadenceGate._rng` | `tpen/callback/cadence.py:75`, drawn at `:89` | For `probability < 1`, the admission sequence after resume is drawn from a fresh stream, so a *different subset* of occurrences is admitted. |
| `StepCadenceGate.num_calls` | `tpen/callback/cadence.py:184`, incremented at `:207` | Same as `CadenceGate.num_calls`. Note `StepCadenceGate` deliberately has **no `reset()`** (`:167-172`), so nothing resets it either. |
| `StepCadenceGate._rng` | `tpen/callback/cadence.py:185`, drawn at `:220` | Same as `CadenceGate._rng`. |
| `_CallbackCore._rng` / `num_calls` | `tpen/callback/base.py:109-110` | The legacy scalar scheduling path, same two failure shapes. |

**The two halves fail independently, and this is the sharp point.**
`StepCadenceGate._draw` (`:216-217`) returns early **without touching the RNG**
when `probability >= 1.0`, while `num_calls` advances on **every** admission
(`:207`). So:

- a `probability = 1`, **uncapped** cadence owns private RNG state that is never
  consumed. It is *not* a science defect merely because it owns unused state.
- a `probability = 1`, **capped** cadence has *no* live RNG dependence and a
  *fully live* counter dependence. Restoring the RNG alone would not help it.
- a `probability < 1` cadence depends on both.

Classifying the whole gate on the strength of the RNG alone gets the capped case
exactly backwards. R0's fixture reproduces both halves:
`test_restore_limb_mutations.py::test_a_cadence_with_probability_one_depends_only_on_its_counter`
isolates the counter dependence, and the main parity fixture runs at
`probability < 1` so the RNG half lands in the compared trace.

**Which shipped configurations are affected is not established by this lane.**
The mechanism is measured; the exposure is not.

### Per-attempt diagnostic — correctly not restored

Restoring any of these across a resume would be *wrong*: they measure the
attempt, not the science.

| Owner | Location |
|---|---|
| `ResourceUsage._process_baseline`, `_reported`, `_allocator_reset_failure` | `tpen/callback/resource_usage.py:96-97, 119, 131` |
| `RunTiming._start_perf` | `tpen/callback/timing/run_timing.py:97, 119` |
| `EvaluationTiming._start` | `tpen/callback/timing/evaluation_timing.py:103, 121, 126` |
| `EvaluationComponentTiming._task`, `_starts` | `tpen/callback/timing/evaluation_component_timing.py:105-108, 159, 173` |
| `TrainPhaseTiming._phase_starts` | `tpen/callback/timing/train_phase_timing.py:142` |
| `TerminalStatus.start_time` | `tpen/callback/status.py:188` |
| `Checkpoint._updated_iteration_step` | `tpen/callback/checkpoint.py:228, 343`, cleared per event |

A wall-clock baseline carried across an interruption would report a duration
that includes time the process did not exist.

### Reconstructible — rebuilt from configuration on every attempt

Constructor-assigned, config-derived, and identical on any attempt built from
the same config: `fail_fast`, threshold scalars, output paths, `checkers`,
`accelerator_synchronize`, injected `clock` callables, cached type objects
(`_task_run_type`, `_phase_type`, and siblings), and `_checker_log_names`.
Enumerated across `tpen/callback/{checkpoint,equivariance,evaluation,factor_scalars,metadata,snapshot,status}.py`,
`tpen/callback/health/*.py` and `tpen/callback/timing/*.py`.

These need no persistence. They do need the **configuration** to be identical
between links, which is a separate admission question and belongs with
`72bc554b`.

---

## Consequence for a chain-resumed run

A chain whose configuration includes a cadence-gated callback that is either
capped or probabilistic does **not** reproduce an uninterrupted run's callback
firing schedule, because the gate's counter and stream restart at every link.
Whether that changes a *scientific* result depends on what the callback does:

- a **diagnostic** callback (timing, resource usage, terminal status) firing on a
  different schedule changes only the diagnostics;
- a callback whose output enters the scientific record, or whose `max_calls` cap
  is load-bearing for cost control, changes the record or the cost.

R0 does not enumerate which shipped configurations fall in which category. That
enumeration is part of the production admission question, not of this spike.

**The honest options for production**, neither chosen here: persist the
scientifically-active cadence state alongside the trainer state, or **refuse**
configurations whose callback state cannot be replayed. TPEN already refuses a
`train_resume` whose checkpoint lacks RNG state
(`require_restorable_rng_state`), so refusal has precedent as a design.

---

## Checkpoint-index hazards found by this lane

These are not callback state, but they are mutable state outside the payload
that a chain depends on, and they belong in the same disclosure.

### The resume pointer has no monotonicity guard

`reconcile_publication` builds its expected pointer from the directory it is
**handed** (catalog.py:216-233) and, on any mismatch with `read_latest`, writes
the pointer at that directory. There is **no comparison of step numbers**, so
handing it an older generation while `latest.json` names a newer one moves the
pointer backwards. Resume follows the pointer (`restore.py:118` ->
`resolve_checkpoint_dir` -> `read_latest`), so the chain re-runs committed
science.

Classification: **scientifically active**. The pointer decides which state a
continued run starts from.

Not a defect in current usage — the only caller today passes the newest
directory — but the hazard is held off by an **accident of usage, not a guard**,
and the trigger is precisely the pattern this program is evaluating: a
controller that restarts and reconciles the generation it *remembers* handing
off. The subsystem documents the hazard about itself in
`iter_publications`' torn-row repair message, which warns that
`reconcile_publication` rewrites `latest.json` unconditionally and would point
it at an older checkpoint.

Pinned non-destructively for DIRECTION in
`test_typed_outcomes_and_ledger.py::test_reconciling_an_older_generation_rewinds_the_resume_pointer`,
which also asserts both committed payloads are byte-unchanged. **No guard was
added to `tpen/`.**

### An orphaned post-rename generation blocks the next commit

An interruption between `tmp_dir.rename` (save.py:211) and the catalog append
leaves a complete generation on disk that `latest.json` does not name. Resume
correctly takes the older generation and replays the tail — and then
`save_checkpoint` refuses to commit, because the directory name is already
occupied (`FileExistsError`, save.py:142-143). The chain advances zero further
generations.

Classification: **scientifically active**, in the sense that it halts the
science. Direction is fail-closed; nothing is corrupted.

### Two selection surfaces that disagree

`list_complete_checkpoints` and the `latest.json` pointer return **different**
generations in both pre-`latest.json` fault windows. Production resume consults
only the pointer. Classification: **reconstructible**, but only if a controller
knows which surface is authoritative — which is why it is written down here.

Owners for all three: the `tpen/checkpoint` owner, and `3b9b736a` for
interruption safety. The planner has the path-spelling sibling filed as a gated
input to R4.

---

## Provenance of the validation boundary the G6 assertions rest on

Recorded because a misstated provenance claims a guarantee in a form the code
does not give. `CheckpointRef` validation is closed in **both** directions, but
the two directions get their closure from **different places**:

- **CONSTRUCT — LOCAL closure.** `reference.py` `__post_init__` (72-86) runs
  `_nonnegative_int`, `_nonempty_text`, `_require_sha256` and `_freeze` directly
  on the fields. `_freeze` (322-336) traverses `Mapping` and `list`/`tuple` to
  scalars and raises `TypeError` on anything that is not `None`, `str`, `bool`,
  `int` or `float`. The check is at the point of use.
- **DESERIALIZE — DOWNSTREAM closure.** `_thaw` (338-343) **validates nothing
  itself**. The path is safe only because `deserialize_checkpoint_ref` (296)
  routes through `CheckpointRef.from_mapping`, which constructs the dataclass
  and re-runs the same `__post_init__`.

A downstream closure holds only **while the routing holds**. A future caller
reaching `_thaw` directly, or a deserialize path refactored to build a
`CheckpointRef` by any route bypassing `__post_init__`, removes the validation
**silently** — no validator is edited and no test of the validators goes red.
The regression would appear as a change in call routing, which is not where
anyone looks for a validation failure.

**`_freeze` is NOT the caller-open-leaf pattern.** Its `None` is a legitimate
terminal JSON scalar — `None` is valid JSON and there is nothing beneath it to
descend into — not a position left open. Accepting a terminal `None` is correct
there, and a shared remedy phrased as "a screen must not stop at `None`" would
break correct allowlists like this one. The discriminator is whether anything
remains **below** the leaf to traverse to, not the token.

---

## Owners

- **HI L5a `e2e512eb`** — factor-rich method and callback replay. Owns the
  cadence-state question above.
- **`3b9b736a`** — interruption safety. Owns the non-unwinding residue measured
  in `README.md`: `save_checkpoint`'s `finally` (save.py:245) does not run for
  `os._exit`, a default-handler `SIGTERM`, or a `SIGKILL`, and the temporary
  directory survives in all three cases. save.py:255-260 already states this;
  R0 provides the measurement.
- **`72bc554b`** — semantics-key forward compatibility. Owns whether
  configuration may differ between links.

R0 absorbs none of these and changes no scientific behaviour.
