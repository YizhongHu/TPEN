# Stage 1 shared protocol and launch surface

This directory owns the Stage 1 manifest, guarded Cannon launcher, and the
science-default Hooke-1e, Hooke-2e, the QHO-2e non-interacting control, He,
and H2 train/eval configs.

Every manifest row carries the exact source SHA, the planned config SHA256, an
external results root, and the TPEN run identity (`experiment_name`, `sector`,
and `run_id`). Evaluation rows additionally carry separate absolute
`checkpoint_dir` and `checkpoint_file` fields plus the checkpoint SHA256. The
run's own core resolved-config identity is joined by the collection slice; this
protocol layer does not fabricate one. Evaluation launches bind
`load.path` to the checkpoint step directory and
`trajectory_identity.checkpoint_file` to its `model.pt`, so the evaluator
records checkpoint provenance. S1-A3 system
eval configs must define a `trajectory_identity` block with `checkpoint_file`
for this override to be accepted by Hydra.

Planning is two-phase: phase 1 may select systems and emits smoke/train rows;
phase 2 consumes that manifest and binds checkpoint digests to eval rows without
replanning seeds or source/config identity. For He/H2, phase 1 retains every
0-to-100,000 update at 1,000-update cadence, while phase 2 selects only roster
B (`0, 1000, 2000, 4000, 8000, 16000, 25000, 40000, 60000, 80000, 100000`).
That is 3 lineages x 11 checkpoints x 4 streams = 132 eval rows per system,
plus one smoke and three train rows (136 total); selection never replaces the
full phase-1 retention set. `config_sha256` is computed by the
planner under `stage1-config-v1`: sorted `configs/<system>/{train,eval}.yaml`
paths and their canonical bytes are hashed. The rule id is recorded in each row;
GPU intent is recorded as a row `device` only when explicitly requested.
Overlay-backed train rows carry a separate exact-byte `overlay_sha256` and
canonical overlay path. The base train/eval pair digest remains unchanged when
only diagnostics bytes change, so `stage1-config-v1` alone never identifies an
overlay-backed execution.
The one-shot compatibility `--config-sha256` input is recorded as
`legacy-config-sha256`; two-phase planning always records the computed rule.

Phase-2 checkpoint input is a `stage1-checkpoint-evidence/v1` envelope, not a
bare digest map. It contains `phase1_manifest_hash`, a
`phase1_train_row_hashes` map keyed by `system:lineage`, and a `checkpoints` map
keyed by `system:lineage:checkpoint`. The planner compares both identities
before reading checkpoint digests, and phase-2 checkpoints must be among the
phase-1 train rows' declared `checkpoint_steps`. Two-phase CLI modes always
compute config identity from this checkout; a supplied legacy scalar is only a
comparison value. The one-shot compatibility mode may still consume the old
scalar/map inputs.

He/H2 roster-B rows carry the versioned selection id `he-h2-b-v1`. Collection
checks the complete coordinate multiset before opening any evaluation output:
missing checkpoint groups, duplicate or gapped streams, and 128/131-row
variants are invalid. Every accepted row retains the phase-1 manifest and
train-row joins, source/config identity, and checkpoint digest. The terminal
100,000 checkpoint is the sole primary; the other roster-B checkpoints are
comparative evaluations. This semantic contract does not approve paid
capacity, storage expansion, or a release, and the executable launch/PBS
consumer blocks remain owned by the launch-redesign follow-up.

Each evaluation config builds one `SystemDescriptor` from its instantiated
`ParticleSpec` and `hamiltonian_terms`, and passes that descriptor plus an
explicit `NaiveLocalEnergyEvaluator` to `TrajectoryMCMCGenerator`. The
descriptor content ID is computed during planning and stored as
`expected_physical_system_id` in every v3 eval row. Collection refuses a
record whose observed ID differs from that value; registry IDs are only a
cross-check, never the source of the physical hash.

Known deviations from the source studies are the shared Stage-1 defaults:
four channels/one layer, CPU device, 256 train walkers, 4096 eval walkers,
and 256 retained draws. Run-slice overrides may select approved science
budgets. The one-electron Hooke route uses the h-v2-style one-electron model
at four channels/one layer and adds the required fixed GaussianConfinement
with alpha=0.25 (omega/2); it has no electron-electron cusp. He retains the
he-v1 trainable electron-electron ranges, curvature cusp (coefficient 0.01,
range 1.0), and trainable Pfaffian readout. H2 follows h2-v1's two-electron
cusp/readout route and retains its width, while using the shared Stage-1
walker defaults. The Hooke-2e route uses the registry's Taut-solvable singlet
pair at omega=0.5 with two electrons, no nuclear resource, kinetic + harmonic
trap + electron-electron terms, fixed GaussianConfinement coefficient 0.25,
and a fixed-default ElectronElectronCusp (opposite-spin coefficient 0.5). H2
retains the two-center R=1.4-bohr geometry, and He retains the two-electron
Coulomb Hamiltonian. Train checkpoint callbacks use ExplicitUpdates through the declared
top-level `checkpoint_steps`; launch overrides bind that parameter to exactly
the manifest row's planned checkpoint steps.

The following table records every retained source-to-Stage-1 deviation and
its reason. Lengths are in bohr where stated; callback frequencies are
training updates.

| System / surface | Source value | Stage-1 value | Reason |
| --- | --- | --- | --- |
| Hooke-1e readout | h-v2: Pfaffian readout `trainable: true` | `trainable: false` in train and eval | Deliberate one-electron Stage-1 default; the Gaussian confinement remains the fixed envelope and the train/eval model stays strict-restore compatible. |
| Hooke-2e envelope and Hamiltonian | `experiments/hooke/tpen-pair-v1/configs/train.yaml`: Gaussian coefficient 0.25, electron-electron cusp, kinetic + harmonic trap + electron-electron interaction | Nucleus-free v2 descriptor with fixed GaussianConfinement coefficient 0.25, fixed-default ElectronElectronCusp, and `omega: 0.5` | Preserve the approved two-electron singlet physics through the current system-context/dispatch route; no legacy atomic placeholder or stale pair launcher is used. |
| Hooke-1e eval sampler | h-v2: 1024 walkers, burn-in 1000, steps 20 | 4096 walkers, burn-in 100, steps 10 | Use the shared Stage-1 evaluation budget and cadence; this is a protocol default, not a claim that the source burn-in study transfers unchanged. |
| He eval sampler | he-v1: 4096 walkers, burn-in 100, steps 20 | 4096 walkers, burn-in 1000, steps 20 | Bind the approved He/H2 burn-in and stride to the semantic evaluation contract. |
| H2 eval sampler | h2-v1: 4096 walkers, burn-in 100, steps 20 | 4096 walkers, burn-in 1000, steps 20 | Keep He/H2 evaluation leaves identical for comparable four-stream rows. |
| H2 model width | h2-v1: 4 output / 16 hidden channels | 4 output / 16 hidden channels | No width change is made; this row documents the retained source value and corrects the prior “width reduction” wording. |
| H2 train/eval walkers | h2-v1: 128 train / 1024 eval | 256 train / 4096 eval | Use the shared Stage-1 train/eval walker defaults across all four systems. |
| Hooke-1e train observability | h-v2: `DataIntegrity`/`GradientStats` every 100 updates; JSONL logger | Both callbacks and logger omitted | Keep the science-default config small and portable; Stage-1 collection relies on its run/record provenance contract. |
| He train observability | he-v1: `DataIntegrity` every 1, `GradientStats` every 5; CSV and JSONL loggers | Both callbacks and both loggers omitted | Same bounded Stage-1 config surface; these diagnostics are outside the Stage-1 manifest contract. |
| H2 train observability | h2-v1: `DataIntegrity` every 1, `GradientStats` every 5; CSV and JSONL loggers | Both callbacks and both loggers omitted | Same bounded Stage-1 config surface; these diagnostics are outside the Stage-1 manifest contract. |
| Evaluation loggers | h-v2, he-v1, and h2-v1 each write CSV/JSONL or JSONL metrics | No metric logger in the Stage-1 eval configs | Stage-1 evaluation retains trajectory records and summaries as its required evidence; study-specific metric logging is not part of this layer. |
| Manifest seed encoding | Prior mixed-radix coordinates could exceed Torch's signed int64 seed range | Deterministic SHA-256 digest mapped to `[0, 2**63-2]`; manifest-wide collision refusal and boundary tests | The real producer exposed Torch overflow. Retaining unbounded mixed-radix values was rejected because every run seeds Torch generators. |
| Stage-1 evaluation statistics summary | `TrajectoryStatisticsSummary` without its required identity fields | Removed from Stage-1 eval configs; provenance remains from `Evaluator._evaluation_identity`, and collection computes statistics from the inspected grid | Supplying `stage`, `run_id`, `attempt_id`, `evaluator_id`, and `config_sha256` from `launch.py` was rejected because it would broaden the launcher contract without an owned evaluator identity. |
| Stage-1 record calculators | Eval configs supplied only `LocalEnergyCalculator` | Add `WavefunctionCalculator` beside it | `SampledRecordWriter` requires the wavefunction bundle field for durable records; the canonical producer wiring supplies both calculators. |

`plan.py` writes a manifest only. `launch.py` renders a script only; submission
is an explicit operator action. The generated script requires Slurm, checks the
delivered partition, uses node-local per-job scratch, and records job identity.
The supported submission directory is the repository root. `render` and the
CLI require an explicit absolute `checkout_root`/`--checkout-root`: this is the
exact execution checkout staged for the job, and it owns the driver,
`experiments` protocol, configs, and source identity. The checkout used to
render the command may be different. The row's kind selects the fixed driver;
the encoded row binding supplies config, identity, overlays, and checkpoint
provenance:

```text
/absolute/path/to/python /absolute/execution/checkout/experiments/atomistic/stage1/train.py --binding <row token>
```

The Stage-1 drivers resolve the bound config against this directory, not the
process cwd. The eval driver applies its strict model-only checkpoint binding
from the same row. A CLI caller rendering from the repository root for a
staged checkout passes `--checkout-root /path/to/staged/checkout`; it must not
rely on the launch module's own location or the submission cwd.

Smoke rows use the same `configs/<system>/train.yaml` files as train rows. Their
smoke-ness is only the bounded budget supplied by `launch.row_overrides`:
`sampler.n_walkers`, `trainer.max_steps`, and `checkpoint_steps` are set from
the manifest row, with the final checkpoint at `smoke_updates`. A smoke checks
that the launch path, gateway, real train config, and final checkpoint can run
to `COMPLETE` at the planned cluster directory. It does not measure energy
accuracy and does not run evaluation or collection.

The real-producer integration test is marked `stage1_real_producer` and remains
in default collection. Mutant packets select their test paths explicitly so
this producer chain is not multiplied across mutants; the base-on Stage-1
verification must include it explicitly, for example:

```text
pytest -q experiments/atomistic/stage1/tests/test_real_producer_e2e.py
```
