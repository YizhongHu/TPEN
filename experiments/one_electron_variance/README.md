# One-electron variance experiment

This study asks whether energy remains stable while local-energy dispersion
falls and later rises without any electron-electron interaction. It trains
three independent one-electron Hooke lineages and three atomic-hydrogen
lineages for 50,000 completed optimizer updates. The full specification is
Task Orchestrator plan `one-electron-variance-design-20261028`, document hash
`9c7c3bf998e06502ea40814fbfdb3d1ed641e0002e581a609e797e2eaa5ddc51`.

The motivating approximately eightyfold contrast belongs to the documented
**He-v1** lineage `train-seed0000`, run
`tpen-f4-eval42-78152c40-20260826T194602--0400`, corrected collection
`f4-collect-20260828T0100--0400`, between 25,000 and 50,000 updates. The manager
selected that anchor on evidence while the operator was away; it remains
subject to operator correction. No separate He-v2 dataset was identified.
The approximately 14,139-fold contrast on 5,751 selected diagnostic coordinates
is a different quantity and does not determine this experiment's outcome.

## Fixed physics and independent controls

Atomic units and infinite nuclear mass are used throughout. Hooke has
frequency 0.5, potential `r²/8`, normalized state proportional to
`exp(-r²/4)`, and exact energy 0.75. Hydrogen has charge one, potential `-1/r`,
state proportional to `exp(-r)`, and exact energy -0.5 for positive radius.
Neither Hamiltonian nor learned model contains an electron-electron term or
cusp. The existing odd-electron readout is retained, with its squared-amplitude
floor fixed at `1e-12`. A border in that readout is not a physical electron.

`tpen.physics.one_electron` supplies analytic functions, not energy answers or
learned forwards. Separate scalar formulas in the tests supply the expected
kinetic, potential, and total energies. The existing two-electron Hooke
references and their 2.0 ground energy remain regression controls for two
electrons; that energy is never reused for one electron. Wrong kinetic factors,
missing potentials, and deliberately imperfect normalizable states test that
the instrument can fail. Hydrogen at the singular point `r=0` is undefined;
neither the reference nor the Coulomb potential regularizes it.

Sampler qualification is separate. Hooke uses independent normal-coordinate
draws; hydrogen uses a radius from three independent exponential draws and a
uniform direction. Before looking at any sampler output, fixed-seed simulations
calibrate simultaneous radial-distribution bands including the 99% and 99.9%
tails. Both unmodified streams must pass; the amplitude-target mutant must
fail. These bands assess the final draw's independent walkers, with all 64
draws' occupancy counts retained. They do not pretend correlated draws are
independent and do not replace the primary whole-trajectory energy estimator.

## Training and measurement

`base.yaml` freezes the common learned architecture and optimizer. `protocol.py`
adds each external potential and factor, six initialization/sampling seed
identities, and 108 fresh evaluation streams. No exact reference replaces a
learned model. Initial state and every 2,500-update checkpoint are retained.
The first-update gradient inventory distinguishes absent, zero, and nonzero
gradients; it does not prove a parameter is structurally inactive.

Each checkpoint stream retains 64 draws of 4,096 walkers after 1,000 burn-in
proposals, with 20 proposals between draws. Evaluation checkpoints are 2,500;
5,000; 10,000; 12,500; 15,000; 25,000; 40,000; 45,000; 50,000. Raw variance
uses all 262,144 local energies, without clipping or finite-only filtering.
The raw channel-summed readout amplitude is replayed on every retained
coordinate in bounded chunks. Signed log amplitudes must reconcile. This
diagnostic shares the learned forward; it is not another independent oracle.
Hooke records explicitly use `trajectory_records/v2` with null nuclear geometry.
This is opt-in; nuclear systems retain their existing version-one records.
The atomic-configuration type still requires real, positive-charge nuclei.

The intended population is each checkpoint's learned squared wavefunction.
Exact-ground-state draws, selected grids, and the final draw alone cannot
satisfy or negate the claim. Finite empirical variance is not a proof that a
population moment exists.

The strong variance signature requires both streams to show an early-to-middle
decrease of at least twofold, a middle-to-late increase of at least tenfold and
`1e-4` in squared energy units, sustained at two of three late checkpoints.
The floor is frozen before training as the larger of `1e-10` and one hundred
times the largest exact-state ordinary-grid variance. The historical 25,000 to
50,000 ratio is reported separately. Replication means two of three independent
training lineages in one system, not two evaluations of the same lineage.

Energy equivalence uses 1,000 whole-walker bootstrap resamples and a 90%
interval wholly within 0.005 of zero for both specified energy differences,
and within 0.005 of the exact energy for both windows. Prefix and stream
checks are required. The collector defaults to unresolved energy stability:
an analyst must supply a recorded, reasoned assessment that tail behavior and
sampling support these intervals. Merely passing an interval calculation does
not establish that premise. A clean null also requires adequate initial
variation, actual parameter gradients, and all 50,000 completed updates.

## Qualification and execution

All tests run inside cluster allocations. Study qualification tests live beside
the experiment, separately from the core package test suite:

```
experiments/one_electron_variance
```

`test_reference_controls.py` contains the independent scalar and imperfect-state
controls, `test_sampler_controls.py` contains the full-budget radial and
wrong-target controls, and `test_readout_records.py` checks the study's complete
readout records. The protocol, admission, and configured integration tests also
remain in this experiment directory.

Also run the unchanged Hooke exact-energy, local-energy, sampler,
electron-count training, envelope, checkpoint, and trajectory regression tests.
Preserve collected identities, runtime results, source and dependency identities,
control artifacts, and scheduler records. An independent clean verifier must
test the exact proposed source. No new dependencies are required.

Prepare configurations without launching anything:

```sh
python -m experiments.one_electron_variance.protocol \
  --root "$STUDY_ROOT" --source-sha "$RELEASED_SOURCE"
```

The planner writes the existing toolkit's attempt-free plans. The allocation
launcher uses toolkit admission and separate child processes, never submits to
a scheduler, and never retries. It requires an exclusive output location.
An excluded-seed pilot keeps the full 4,096-walker batch and measures 1,000
updates and one complete evaluation stream for each system:

```sh
python -m experiments.one_electron_variance.launch pilot \
  --root "$PILOT_ROOT" --source-sha "$SOURCE" --admission-id "$ADMISSION"
```

The batch wrapper must verify one full A100, record delivered hardware,
software and peak memory, and use the current cluster-access instructions.
The pilot's time projection includes process setup and checkpoint writing
conservatively. Both lineages must fit six hours with 20% spare capacity.
Missing memory evidence is unresolved admission, not a pass.

Science is permitted only on qualified released-main source. Both launcher and
collector require the same `one-electron-admission/v1` document, identifying
`source_sha`, equal `released_main_sha`, `design_sha256`, and `writer_id`.
Its `evidence` mapping must contain `instrument_controls`, `gpu_correctness`,
`independent_verification`, `timing`, `memory`, and `release`. Each reference has
a `path` and `sha256`. Each referenced receipt names its `kind`, exact
`source_sha`, `status: passed`, and a nonempty `artifacts` list of hashed raw
evidence references. Relative paths resolve beside the referring document.

Instrument receipts require every check named in `admission.CONTROL_CHECKS`
and the measured `max_exact_ordinary_grid_variance`. Independent verification
names a distinct `verifier_id`. Release names `ref: refs/heads/main`; timing
records each system's `fits_six_hours_with_twenty_percent_margin`; memory records
`full_a100_verified` and `full_4096_batch_and_stream_verified`. These checks bind
and preserve the inspected evidence; they do not authenticate the people making
release or review decisions. Development tests and pilots do not authorize
science. Then one allocation runs one lineage and its 18 evaluations:

```sh
python -m experiments.one_electron_variance.launch lineage \
  --root "$STUDY_ROOT" --source-sha "$RELEASED_SOURCE" \
  --admission-id "$ADMISSION" --system hooke --lineage 0 --gate "$GATE"
```

No source changes, retries, replacement seeds, shortened horizon, or hidden
extensions are part of the protocol. Budget ceiling: six six-hour full-A100
allocations plus a half-hour full-card pilot; separately a quarter-hour
partitioned-card smoke and 1.5 four-core processor-node hours. The previously
written 28.2-hour scenario assumes 0.30 seconds per update and 90 seconds per
evaluation stream; those are planning assumptions, not measured performance.

## Collection and interpretation

`analysis.py` checks source, completed-update records, fresh evaluation seeds,
strict restore identities, complete coordinate grids and checksums. Missing
or invalid lineages remain incomplete. It also reconciles observed sampler
settings and the emitted model-only restore receipt. The qualification argument
is the same complete admission document used for launch. An optional interpretation document maps
`hooke:0` through `hydrogen:2` to `energy_uncertainty_adequate` and a nonempty
`rationale`. Preserve its hash and the evidence behind every assessment.

```sh
python -m experiments.one_electron_variance.analysis --root "$STUDY_ROOT" \
  --qualification "$QUALIFICATION" --interpretation "$INTERPRETATION" \
  --output "$ANALYSIS_RESULT"
```

The 25,000 and 50,000 profiles also name the fixed 32 evenly indexed and 32
largest-residual records for the existing optional finite-difference derivative
check. Preserve agreement, disagreement or abstention and error estimates;
these selected records cannot measure prevalence. That check shares the
learned forward and requires the already-approved, locked optional dependency.

A replicated Hooke recurrence shows that neither electron-electron interaction
nor a Coulomb singularity is necessary for this measured signature. A hydrogen
recurrence also removes electron-electron interaction as necessary, without
isolating a nuclear-cusp cause. A qualified null is bounded to this protocol
and horizon. It does not establish helium causality: particle count also changes
geometry, active paths, and readout behavior. Three seeds give limited evidence
about rare failures. The historical helium result had only one lineage.
