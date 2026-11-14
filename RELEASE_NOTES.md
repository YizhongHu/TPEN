# Release Notes

## v0.5.2 - H-v2 science code, core-executor contract start, Stage 1 records, executor teardown

44 commits since `v0.5.1` (`76c35bc6`), merged into `dev` through 26 pull
requests and released to `main` by #709 as `14235a0a` (tree identical to dev
`fce3de22`). The tagged code still reports package version `0.5.1`; the
version sources move to `0.5.2` on `dev` in #712.

```
git log --format='%H' 76c35bc63bcf2fee0cc2b3dee7587c5f1358b86c..fce3de22ccb88538f704ad329c94ea868b7b2e11 | wc -l
```

Release gate: full default suite at `fce3de22` on FASRC Cannon (`test`
partition, typecheck on, executor arms included), job 60478238: 8586 passed,
0 failed, 0 errors, 20 skipped.

### Changed

- **`scripts/cannon/verify.sh` type-checks by default** (#683). Runs now use
  typeguard unless `--typecheck off` is passed. The overlay installs the
  executor as an editable package and binds only the whole-word
  `*_(EXECUTOR|ROOT|SCIENCE)_PYTHON` interpreter selectors (#696).
- **Implementation lifecycle** (#705): an item advances to `review` only once
  its PR is ready for human merge, so `unblockAt: review` fires at ready. The
  review checklist asks any new test slower than 30 s to state and justify its
  runtime.

### Added

- **H-v2 science code**: science horizon extended to 100k updates (#687);
  `AmplitudeRiskFraction` enabled in the science A1 modules every 100 updates
  (#690); evaluation generations packed per GPU job (#675); 100k pilot
  consumer probes (#684).
- **Core-executor contract**: boundary contract document and versioned
  fixtures (CX-00, #693); published-boundary planned yield for resumable runs
  (S4, #691).
- **Executor**: bounded remote teardown acknowledgement (EX-S3.C, #668).
- **Physics and evaluation (Stage 1)**: physical system descriptor and semantic
  term declarations (#674); shared local-energy execution receipt and result
  transport (#677); version-aware historical trajectory reader with immutable
  v1 fixtures (#678); opt-in `trajectory_records/v2` writer and active-producer
  binding (#682); one-electron Hooke oracle and reference registry (#695).
- **Operational artifact routing** with constant science config (S3, #663).
- **Baselines**: seed-spread summary for collected baseline records (#701).
- **Cannon submit-lint** shared-environment prefix split (#676).

### Fixed

- `tpen.run` no longer imports `torch.nn` at module import; the planned-yield
  and checkpoint names load lazily (#703, regression from #691).
- Baseline rung builder uses checkout imports instead of a hard-coded home
  path (#700).

### Tests and documentation

- Test-only fixes: S2 restore-test fakes are real `Runner` instances (#688);
  chain-resume tests treat the R2.3 `trainer_cursor` refusal as an observed
  outcome (#698); H-v2 store reservation probe stabilised (#685); named skip
  when review-test interpreters are unset (#697); sampler-draw receipt with a
  corrupted `transition_count` is refused (#699).
- README release version set to 0.5.1 (#694).

## v0.5.1 - H-v2 100k pilot payload, shared Holystore environments, resume and executor layers

33 commits since `v0.5.0` (`db001085`), merged into `dev` through 11 pull
requests: #653, #656, #657, #659, #661, #664, #666, #670, #660, #671, #673.

```
git log --format='%H' db00108526b894f4f3dd4644bd25298cea91cc14..<release dev head> | wc -l
```

### Breaking

- **`scripts/cannon/verify.sh` no longer builds environments** (#671). It
  reuses only a complete shared environment under the new required
  `SHARED_ENV_PREFIX` (the Holystore `envs/` root); the old `SHARED_ENV_ROOT`
  input now fails with `PRECHECK_FAILED` (exit 2). Callers must pass the new
  prefix and build an environment for a new `uv.lock` hash with the shared-env
  builder first.

### Added

- **H-v2 100k capacity-pilot payload** (#673): a fresh 0-100,000-update
  lineage payload with a generalized checkpoint inventory and stage-1/SEL
  window. The released 20k pilot payload is unchanged.
- **Capped shared-environment builder** (#670): immutable lock-scoped uv
  environments on Holystore with a committed cap
  (`scripts/cannon/shared-env-policy.json`) and hard refusal at the cap.
- **Typed operational restore with read-only predecessor discovery** (#660)
  and its measurement against a sealed predecessor leaf (#659).
- **Executor S3 layers**: remote-leaf transport wire and worker-owned execution
  (#661); attached HTEX transport selection and submit (#664).

### Changed

- **HI launch/resolver**: pre-resolve overrides are forwarded (#657) and
  bracket-segment override paths are refused (#656).
- **Cannon submit-lint** soft-requirement wiring and false-block fixes (#653).
- **Design document** `main.typ` updated to the 2026-11-10 version with its
  bibliography (#666).

### Environment note

The version bump changes `uv.lock` (sha12 `a0d83e31a6a2` -> `2d11a1c38872`).
Shared environments are named by that hash, so CPU and GPU environments for
`2d11a1c38872` must be built before lock-keyed jobs run at this release.

## v0.5.0 - Atomic hydrogen, checkpoint admissions, and durable execution

186 commits since `origin/main` (`5d920226`). The release brings the H-v2
atomic-hydrogen experiment through science-config routing, pilot/TRAIN
payloads, study-side evaluation, and contract-cadence logging; admits
checkpoint-profile callbacks with fail-closed resume behavior; and lands the
independent `tpen_executor` project through placement v2, the L1b GPU sampler,
and resolved-config identity fences.

The census below covers every commit in `origin/main..ca0f93b3` (186 total),
measured with:

```
git log --format='%H' origin/main..ca0f93b3bdd6761399a2506f7829adc06532b7c6 | wc -l  # 186
```

No commits are excluded. Entries without a pull-request number are cited by
commit hash; the hash/PR list is the complete census for the range.

### Breaking

- **Checkpoint publication is no longer task-owned cleanup** (#450). The
  checkpoint stream retains all checkpoints and stops TPEN-initiated deletion,
  changing retention and storage expectations for existing runs.
- **The HI `tpen.hi.train.v1` import facade is removed** (#517).
- **VMC training now uses the native SR/minSR engine and explicit method
  state** (#472, #476), with a declared non-finite estimator policy (#483)
  and foreign optimizer-carrier rejection at both binding boundaries (#537).

### Added

- **H-v2 atomic hydrogen experiment.** The science contract, atomic-hydrogen
  pipeline, exact evaluation canary, and executor integration land in #587,
  #590, #592, and #613. Approved per-lineage science configs route through the
  attached builder (#629); pilot and part-(g) TRAIN payloads (#637), study-side
  evaluation fan-out and SEL (#641), and contract-cadence training logging
  (#651) complete the release path. Collector integrity checks from #631
  enforce restore, sampler, checkpoint, log-completeness, clean-run, and
  variance evidence.
- **Checkpoint-profile admissions.** InitialCheckpoint and ParameterActivity
  are admitted with resume safety (#624); the raw training-batch recorder is
  fail-closed on resume and lineage/cursor mismatch (#632); OptimizerState
  (#633), SamplerHealth and FactorScalars (#647), and AmplitudeRiskFraction
  (#630) receive native-profile admission and callback coverage.
- **Independent executor subproject.** The public run-evidence seam and
  `tpen_executor` boundary (#576, #577) grow through immutable contracts and
  run manifests (#578), staged Hydra program specifications (#579), prepared
  environments and payload delivery (#581), durable stores and receipts
  (#583), the attached Parsl dispatcher (#588), and the public dispatch CLI
  (#593).
- **Scientific and operational infrastructure.** Placement v2 with physical
  host maps and verdict semantics (#627), the P8 placement reader (#628), and
  the in-child L1b GPU PID-to-UUID sampler (#640) establish placement
  evidence. The fast Cannon runner (#636), pre-submit contract linter (#639),
  Polaris CPU/GPU verification harnesses (#554, #557), and P10 smoke template
  (#625) improve execution checks.
- **Model, training, and analysis capabilities.** The coordinate-gradient
  request (#437), typed wavefunction packets (#435), activation-owned MLP
  (#453), activation presets (#458), VMC statistics/resume authority (#451),
  resource artifacts and profiling (#452, #454, #459, #460), and the HI
  stage/evaluator architecture (#491, #492, #494, #497, #499, #503, #510)
  become available.

### Changed

- **Resolved-config identity is now carried and enforced end to end.** Accepted
  segments record the identity (#618), L1/L2 consumers stamp and verify it
  (#618 x #628), and callback output slots are classified as operational
  (#635), preventing observational artifacts from changing scientific config
  identity.
- **The repository edit guard refuses unsupported Python interpreters**
  (#518).
- **Training configs no longer carry reference energies** for He-v1/H2-v1
  (#638); the historical He configuration remains documented separately.
- **The H-v2 executor path records child placement evidence** (#605), resolves
  fenced evidence paths (#608), and preserves tiled worker visibility (#601).
- **HI execution and checkpoint recovery are now explicit.** Stage-Q carries
  literal context coordinates (#527), architecture and O-composite contexts
  are admitted through the stage runner (#552, #555, #573, #580), and the
  native recovery chain gains E1/E2 contracts plus Balsam and native adapters
  (#529, #530, #533, #534, #535).

### Fixed

- **H-v2 review-test modules register under their spec name** (#648, #655),
  as the study-module identity rule requires.
- **H-v2 store-parent creation is ordered before dispatch** (#648, #650),
  preventing pilot runs from reserving a child store without its parent.
- **Checkpoint and callback lifecycle holes are closed.** Raw recorder resume
  is refused fail-closed (#632); InitialCheckpoint and ParameterActivity
  profile/resume admission is guarded (#624); callback output identity is
  classified correctly (#635); and trajectory metadata, checkpoint retention,
  and publication receipts are validated (#609, #450, #464, #477).
- **Executor and study routing boundaries are hardened.** P5/P6/P7/P8/P9/P10
  routes, slot pools, attached allocations, and placement evidence refuse
  malformed or ambiguous inputs (#582, #584, #585, #586, #588, #589, #593,
  #595, #596, #597, #599, #600, #601, #605, #608); the resolved-config fence
  prevents cross-lineage identity joins (#618, #628).
- **Training correctness and resource safety improve** through chunk-bounded
  local-energy graphs (#565), duplicate experiment-module collection fixes
  (#522), torn-tail-safe JSONL append (#481), rank-consistent run IDs (#487),
  and preservation of the latest checkpoint during older reconciliation
  (#509).

Complete census, in chronological order (`short-sha` followed by PR number
when present):

```
79e37332#418, a125e1af#419, 8921f6cb#420, 813c9974#422, b7858ad2#423, 98a8d6eb#425, fa68f89b#426, 534b7441#427, 633f9a74#430, 8976b932, 623eec69, 2790fb32,
4c95175a#424, 90022009#428, 0eb40797#429, 01055bab#450, f58ef39c#435, 75bae018#437, ff5aff29#441, 323da175#443, 09c950e1#451, d1fe29f8#453, 95a0771a#456, 0cc82e27#458,
b39d5269#452, 5ea3da07#454, 85d81b1e#459, 01c8712d#460, 73016557#464, 77317556#462, 23a16afd, 7cd2a897, f05a7c65, b1bfcfdf, 49c82663, 561ff743,
b806d0e3, 0e7b2040, 838e5ca8, aafa4a03, 66bf630d, 7ee49320#468, 4ec7c8cb#472, 56308c8b#476, 72128573#477, 8d1ca6fd#481, c16afd0e#473, bf011cf2#474, e8d637f6#479, 4798b7dd#480, 4d67d5b0#482, 7d86d090#483, c2ce426e#484, 223b81fb#486, 0f622481#487, 80f6249a#488, 1bfeac1a#489, fe497899#491, 28ab5446#492, 36cfddc5#494,
8e2bef6d#490, 4a580e64#495, ae20c11a#496, a1f76b92#497, de7a5677#505, a37d7c62#509, d39aa2cd#510, 71ac61ed#511, 5fc56983#499, 9c415376#503, d04a50cd#512, e9204d68#513, 3143e43a#515, be3d0a5a#522, 842698e8#518, edc0228e#517, 7735ac80#519, b972a023#521, 29a41df1#528, 46ab6585#538, 8dc25f00#523, df700150#524, 4f8ebb0f#537, 1677bf6f#536, d727375f#527, a757ed76#540, 2f5b8d56#542, 3994845f#543, 1afffde9#544, f1419267#547, d7649753#546, 56d95c4b#529, b3ed0406#530, e9c67890#532, 29ffe95d#533, 5b42a8f8#534, a6761c3e#535, e27b29ae#545, 62ba7ebb#549, 58445152#551, bf2f208f#550, 3425ecf7#552, 1ed309f0#556, 8a18cd14#555, d55c6d11#558, 3a54cf85#559, 07305a9b#561, 7a04317a#554, 98da8760#557, 1db9e4b0#563, cde285a6#564, 6514b1b7#565, 241fe16f#566, 19a22ed5#568, fbbebe69#569, a0dd9b52#571, 67beab20#575, 1ae776c9#572, 1ead406c#573, efc33bb3#580, b80b0428#594, 8d148c47#587, 0b5ba7d5#590, ae9de462#592, 2a7e5fb3#576, e4d45e69#577, 019469f3#578, 2b3926e9#579, 184c3fd4#581, 2629a3a4#582, 9fe71f9e#583, 56f16bc9#584, 33ef1ce3#585, 98ce5e8b#586, ec826201#588, 0f7e8b3c#589, 12d1ad73#591, d2a58495#593, 060044d8#596, 8535bc05#597, 043d365b#599, 179240d5#601, 7029de22#595, 8d3e81af#600, 2630ea7b#609, d48e60e2#610, f6b7c398#611, b3a4ccac#605, 389c5ce4#608, 8222b5cd#598, d31747ce#612, 29acb155#614, fbed0263#613, e2d4a449#619, cf146d39#615, 6323db7d#624, 637aa5c5#616, aef23069#618, 78390364#622, 15aebfb1#621, c284e29f#620, deca8d44#626, aba217ad#634, 8a142f84#625, 08e54ea8#635, 152792af#631, b9ba7c2d#629, f71e0a5e#637, 096d724d#632, d8ca479a#633, fd061d73#627, f8a0f40d#628, aa53b1f0#638, df46e2db#645, 672926c9#630, 5b441878#646, b60c6771#647, 7f1e0432#636, 1d295253#639, 26285862#649, f1116084#640, 47e342a2#648, 4970ba1d#650, 9091b678#641, 294d7ee0#651, ca0f93b3#655
```

## v0.4.0 - Analytic electron-nucleus cusp local energy

20 substantive commits since v0.3.4 (`ed09f85`), excluding the #411 release
chore. The headline is an analytic evaluator for
the electron-nucleus cusp; the rest is the typed operator machinery it needed
and two compatibility changes.

The census for the immutable `origin/main..887988c816ae19d306c7bd21f7116cc1174185b7`
examined 21 commits: 20 are cited in this section (including the release bump,
#411), and one is deliberately excluded. The heading's release count excludes
that release chore; the census cites it so every commit in the range is
accounted for. The count was measured with:

```
git log --format='%H' origin/main..887988c816ae19d306c7bd21f7116cc1174185b7 | wc -l       # 21, including #411
git log --format='%H' origin/main..887988c816ae19d306c7bd21f7116cc1174185b7 | grep -v '^fe569dd' | wc -l  # 20 substantive
git log --format='%H' origin/main..fe569dd | wc -l                                           # 19, including #411
git log --format='%H' origin/main..fe569dd^ | wc -l                                          # 18 substantive
```

Commit census: `72be473` (#383), `e75243d` (#385), `8193d9c` (#392),
`7faf370` (#386), `9c65cdb` (#388), `c54a916` (#389), `c2fb091` (#391),
`287c96d` (#403), `4a33b6a` (#404), `84a0a52` (#405), `e6063b5` (#406),
`25342b6` (#407), `c761a4c` (#387), `06e2d17`, `dedf761` (#410),
`27af226` (#413), `a1f8938` (#414), `fe569dd` (#411), `7a8edf2` (#415),
and `887988c` are represented below. `c585534` is deliberately excluded:
it removes a stale documentation comment without changing user-facing
behaviour, so it does not warrant a release-note headline.

### Breaking

- **Minimum Python is now 3.12** (#404). The declaration was the only thing
  lagging: `.python-version`, the docs workflow, and the `h5py` pin comment
  were already 3.12. `uv.lock` loses its sub-3.12 resolution forks, which is
  not a dependency upgrade - every dropped entry carried a
  `python_full_version < '3.11'` or `== '3.11.*'` marker and was already
  unselectable.
- **The Coulomb distance floor now defaults to `0.0`** (#403). The `eps`
  keyword is retained for backwards compatibility and documented as unproven.
  A non-zero floor removes the potential divergence and introduces one into
  the total; measurement found `E(r; eps) = Z/r - Z/eps - Z^2/2` inside the
  floor. Runs that relied on a non-zero default will change.
- **Vestigial `SpENN` module names are retired from the core package** (#414).
  `tpen/nn/spenn_layer.py` and `tpen/nn/spenn_wave_function.py` become
  `tpen_layer.py` and `tpen_wave_function.py`; the classes they export were
  renamed to `TPENLayer`/`TPENWaveFunction` long ago. Import paths change, so
  this is breaking, and it is batched into this release rather than deferred
  so consumers absorb one break instead of two.

  Legacy compatibility identifiers are deliberately preserved:
  `LEGACY_BASIS_FEATURE_DIM_RESOLVER` (`"spenn.basis_feature_dim"`) and
  `LEGACY_CHECKPOINT_KIND` (`"spenn.checkpoint"`) still carry their historical
  spellings, because checkpoints and configs on disk reference them. The v1
  record-format documentation in `tpen/metrics_naming.md` is likewise
  unchanged. A test now pins the checkpoint literal, which previously had no
  effective coverage.

### Added

- **Analytic electron-nucleus cusp local-energy evaluator** (#410). `T` and
  `V_en` each diverge as `1/r` at coalescence and only their sum is finite, so
  they are evaluated together as
  `T + V_en = -1/2 sum_i [Lap_i f + |grad_i f + grad_i U|^2] + sum_iA g_A(r_iA)`.
  `g_A` is taken from the cusp provider's pre-cancelled `slope_residual`;
  nothing reconstructs `(u' + Z)/r` by subtraction, which would reintroduce
  the genuine `0/0`. Selected by one explicit config override. A slow
  reference implementation ships alongside as an independent executable
  oracle, matching the vectorised kernel in values and parameter gradients at
  `rtol = atol = 1e-12`.
- Typed analytic electron-nucleus cusp capability (#389) and model-owned
  regular/cusp factorization input (#391), which the evaluator consumes.
- Operators declare which physics they compute via a typed `OperatorId`
  (#405); typed context keys, providers, and a read-only context (#406); an
  operator lowering registry and local-energy planner (#407). The planner
  enforces exactly-once operator consumption structurally rather than by a
  downstream check.
- Physical-term and virial-residual metrics owned in `tpen/` (#386) and an
  optional Numdifftools finite-difference kinetic oracle (#388).
- Baseline matrix plan builder (#387), `StagePlanV2` dispatch via
  `ParslAttachExecutor` (`06e2d17`), DeepQMC environment ported to Polaris
  (#385), and a reusable multi-GPU scaling probe (#392).
- DeepQMC run seeds are recorded (#383).

### Changed

- The documented `pytest` console-script invocation now collects the
  `experiments/` namespace package (#413), and the He-cutover runbook test
  isolates its environment sync under `tmp_path` (#415).

### Fixed

- Parsl attach now accepts any row width that tiles a node, rather than only
  1-GPU rows (`887988c`). Previously the only dispatchable multi-node shape was
  1 GPU per row. For the H2 seed spread, that cost 118 node-hours on Polaris
  job 7575420 arm B versus 39 node-hours for the same work, with 80 of 100
  GPUs idle. The wrapper now admits 2- and 4-GPU row bindings when they cover
  each node without overlap or gaps.

### Known limitations

- Cross-device geometry comparison is untested; `CUDA_AVAILABLE` was false on
  every verification node.
- `torch.equal` raises for `chalf`, `float8_e5m2` and `float8_e8m0fnu` even at
  matching dtype, so the geometry comparator can raise from a path annotated
  `-> bool`. These dtypes have no plausible use for nuclear geometry.

## v0.3.4

### Added

- OneQMC/Orbformer results adapter, emitting baseline records from `result.h5` (#290)

### Changed

- Baseline records reported against published values (#335)
- OneQMC adapter: `--allow-short-tail` help no longer asserts a resolution rule that was removed (#294)
- Timing callback stack (#311, #312) was reverted before release after its cadence test conflicted with the configured callback cadence.

### Fixed

- **Config identity is semantic rather than byte-based** (#379). A YAML reorder whose parsed form is identical previously changed the config hash, which failed all 42 identity joins of a completed, archived study and took it to `n_pass=0`. Identity is now computed over the canonicalised document, so an inert reformat leaves it unchanged while a genuine semantic edit still changes it. Receipts written under the old byte-based scheme stay joinable through an explicitly classified legacy path that refuses an identity it cannot account for, and a missing historical revision now fails one row instead of aborting the whole collection. `PyYAML` is declared directly rather than relied on transitively.

## v0.3.3 - Trajectory energy estimand correction

34 commits since v0.3.2 (`dd49cc1`). Two are `dev`/`main` merges; the rest
fall into three areas: the He-v1 42-row evaluation study, the He-cutover
multi-node Polaris port, and agent/cluster infrastructure.

### Added

**He-v1 evaluation study**

- Factor-response evaluation: scale the trained envelope parameters (`b_ee`,
  `c_en`, `d_en`) in a bounded context with byte-verified restore (#337)
- Evaluation rows declare their own task graph; the 42-row evaluation grid is
  frozen (#339)
- The frozen grid's metric-namespace bindings are carried into the plan
  manifest, so a tolerance cannot land on a different task that happens to
  emit the same metric name (#348)
- Production evaluation rows are placed on `kozinsky_gpu` and `seas_gpu`
  together in one job (#349)

**He-cutover: multi-node execution on Polaris**

- Parsl attach executor for allocation-attached batches, with immutable
  dispatch receipts and completion verification (#354)
- He cutover study: two-facility smoke plan, runtime admission,
  allocation-local Parsl pipeline, facility templates, and contract tests
  (#358)
- Attempt-free dispatch seam (#352)
- Full He cutover plan, production grid, and Polaris production template
  (`4410890`; this commit has no PR number in its subject or body)
- Optional `nodes_per_block` on the dispatch context (#367)
- Placement and topology evidence for multi-node runs: which nodes and which
  physical GPUs actually ran (#369)
- Proof manifest of 1 train + 40 eval rows (#371)

### Changed

- **The He-v1 study's canonical energy is now the whole-trajectory estimate
  with its MCSE.** The final-draw snapshot is retained beside it and explicitly
  labelled as a snapshot; `plateau_reached` travels with the MCSE, because a
  Geyer sequence truncated at the window edge understates it. A row whose
  trajectory statistics never resolved renders `absent` and never falls back
  to the snapshot (#373)
- Duplicate row submission is refused before the allocation rather than
  inside it (#357)
- Legacy Hooke studies marked stale with provenance banners; legacy toolkit
  deprecation recorded (#351)
- Running the authoritative-edit guard outside the agent sandbox is sanctioned
  in AGENTS.md (#338)

### Fixed

**He-v1 collector and planner**

- Collection completeness is derived from the plan, not a row-count literal
  (#342)
- Term energies are included in re-equilibrated records, and the planner no
  longer claims two rows (#343)
- Metrics-only evaluation tasks are no longer required to emit an artifact
  (#345)
- The summary-field oracle unwraps delegating calculators (#346)
- The collector's numeric oracles are boundary-sensitive (#341)
- Evaluation task order restored (#375)
- Test files excluded from the import-boundary check (#376)

**He-cutover and Parsl**

- `AllocationContext` accepts empty `visibility_values` in inherit mode (#355)
- he-cutover entrypoints are importable (#359)
- Polaris Torch bootstrap paths repaired (#360)
- Polaris operator overlay protected (#361)
- Parsl dataflow kernel reused across batches (#362)
- Virtualenv interpreter preserved in dispatches (#363)
- The plan owns `result_dir`, so completion probes the path the runner
  actually writes (#364)
- A bad `PBS_NODEFILE` is rejected before Parsl loads, removing a localhost
  false-green (#368)
- Multi-node `LocalProvider` branch; GPUs are no longer assigned by dispatch
  index (#370)

**Cluster tooling**

- Cannon `uv` resolved outside `PATH` (#374)

## v0.3.2 - Current-dev baseline release

Covers the exact 29-commit `origin/main..origin/dev` range ending at
`d04a96fbd882d5c101b1bcadf28e57fc51ec3e11`.

### Added

- Baseline-comparison coverage now includes DeepQMC and Neural Pfaffian
  adapters, shared estimator statistics, an explicit B/N/N2 system registry,
  and the declared `h5py` dependency needed to execute the DeepQMC adapter
  tests.
- Artifact lifecycle schema and lifecycle-gated run-data-cleanup guidance,
  together with the Cannon archive-experiment skill.
- Row-aligned trajectory records; conditioned-energy, sampler-health, and
  cost diagnostics; and expanded symmetry trace records with repaired trace
  identity contracts.
- He-v1 checkpoint replay semantics, reconciled helium atlases, and a minimal
  real-checkpoint evaluation canary with its collection-contract repair.

### Changed and fixed

- Baseline emission now rejects unmeasured, zero, or otherwise unassessable
  error bars; protects short-tail handling; reports measured tail windows; and
  supports an explicit operator caveat for DeepQMC records.
- The below-floor fallback decision and FermiNet attribution guidance are
  documented, and baseline fixtures exercise executable DeepQMC coverage and
  measurable Neural Pfaffian tail variation.

### Excluded

- Closed, unmerged diagnostics I1/#325 and I2/#326 are not included in this
  release, nor is their V1 canary/gate. This section inventories only the
  current `origin/main..origin/dev` baseline above; it makes no claim about
  those excluded diagnostics, generated results, tags, or merges.

## v0.3.1 - Generic atom API, helium study infrastructure, and correlation-aware statistics

Covers the full `v0.3.0..a9f9f7fa` delta of 55 commits.

### Added

- Generic atom API and nuclear geometry, landed as the eleven-PR atom stack
  (#258-#274): an immutable `AtomicConfiguration`, its transport through the
  sampler and checkpoints, batch-owned electron-nucleus data, an explicit
  nuclear factorization seam, generic Coulomb potentials, nucleus-nucleus
  repulsion, and the cusp relocation into `tpen/nn/cusp.py`.
- Helium study infrastructure: the He v1 config (#247), radial diagnostics
  (#249), tolerance gates over emitted cusp and tail metrics (#263), the study
  driver (#264), the trajectory consumer (#265), and the log-amplitude oracle
  with a singlet-purity diagnostic (#262).
- Fixed-model statistics: the trajectory IAT/ESS/MCSE producer (#257), which is
  the correlation-aware uncertainty machinery the He study's headline bar depends
  on, with multi-timescale and heterogeneous-chain coverage for the estimator
  (#278).
- Timing and cost accounting: emitted boundary stamps (#240), the device event
  timer (#243), cost provenance (#246), delivered-allocation GPU-seconds (#250),
  warmup exclusion from throughput projection (#254), and per-axis GPU-second
  rollups (#256).
- NN-QMC baseline comparison surface: the survey (#216), the comparison-system
  registry with provenance (#217), the common results record and run collector
  (#218), the transcribed He reference energy (#220), the reference-energy
  correction with H2 (#232), and the FermiNet run-directory adapter (#233).
- Hooke scan work: the four-level basis choice library (#225), the scan
  train/eval slot surface (#226), the `tpen-pair-scan-v1` fork with an enforced
  choice-library merge (#227), and checked-in scan grids with split-sample
  champion selection (#228).
- The frozen He-v1 production arm is predeclared and checkable rather than
  described (#279).
- `tests/unit/test_version_agreement.py` asserts that all four version sources
  agree, reading each from its own file at runtime.

### Changed

- The cusp law is named by its functional form rather than by its trainability
  (#280).
- The unused `AsymptoticDecay` interface is removed (#281).

### Fixed

- Batched Pfaffian evaluation over walkers and channels (#234).
- Antisymmetry is reported from summary metrics at `artifact_level: summaries`
  (#230).
- The inverse softplus is stable above the float64 overflow boundary (#277).
- Shipped he-v1 artifacts no longer claim a byte identity the check never
  established (#282).

### Validation

- Commit delta measured at the release-prep SHA rather than reused: `v0.3.0..a9f9f7fa`
  = 55 commits, which is 50 at the earlier `dev` tip `515231a1` plus exactly the
  five merged layers #278-#282. `origin/main..dev` = 37 was NOT used as the
  release figure: `main` already carries two post-tag merges, so 37 would
  understate the release by 18 commits. `dev..origin/main` = 0, so `main` remains
  a strict ancestor and the promotion stays fast-forwardable.
- Package metadata, runtime metadata, the `uv.lock` root `tpen` entry, and the
  README current-release text all agree on `0.3.1`, asserted executably rather
  than by inspection. `uv.lock` is load-bearing here, not generated noise: the
  cluster environment is built with `uv sync --extra cpu --locked`, which fails
  closed, so a stale lock entry stops every scheduler job from starting.
- `uv.lock` was regenerated with `uv lock`; the resulting diff touches only the
  root `tpen` version line and no dependency, hash, or resolution marker.
- Five-stage he-v1 smoke on merged `dev`: Cannon Slurm `39726909`, partition
  `test`, `COMPLETED 0:0`, `00:05:48`, pinned in-job to `a9f9f7fa` and tree
  `a246b88f`. Composed `2283 passed`, `8 skipped`, `6 subtests passed`; the
  study-local suite gave `227 passed`. Restore was exercised across both the
  renamed Hydra `_target_` of #280 and the added `readout.channel_weights` key
  of #279, the cross-check mutant went RED and restored clean, and
  `assess_convergence` refused with verdict `indeterminate` without narrowing
  windows.
- That smoke installed `tpen` as `0.3.0` in-job, so it is a PRE-BUMP
  measurement. It establishes that the tree being bumped was releasable; it does
  not validate the version change itself, and it predates the version-agreement
  test.

## v0.3.0 - TPEN public API, typed event clock, and four-facility verification

### Added

- Typed occurrence foundation: a typed callback cadence, typed occurrence
  records, and delivery of a domain's own state object to typed callbacks, with
  the four health observers moved onto typed state delivery.
- Typed training phases with explicit update outcomes, and a typed
  `SamplerStats` record in place of dictionary probing.
- Durable checkpoint update state on `tpen.checkpoint` schema v2, with an
  update-driven checkpoint cadence.
- Portable pair-v1 allocation launcher
  (`experiments/hooke/tpen-pair-v1/launch.py`) that runs one task per explicitly
  bound accelerator through `AllocationPoolExecutor`, using the scheduler job ID
  as the allocation identifier and pass-scoped row claims that are never
  released.
- Pinned `rocm71` Torch extra so OLCF Frontier resolves an AMD ROCm stack.
- Authoritative-edit guard (`tools/check_authoritative_edit.py`) and a native
  unbounded linear pull-request stack workflow.

### Changed

- **Breaking:** the public runtime surface completes the SpENN to TPEN rename.
  Importers must use `tpen`, and the default run name no longer falls back to
  the former branding.
- The accelerator backend is resolved from the runtime instead of naming CUDA
  directly, so CUDA, Intel XPU, and ROCm share one code path.
- A whole training record is attributed to a single model version.
- The shell stack launcher is removed in favour of the tracked Python launcher.

### Fixed

- Launcher and receipt truthfulness: allocation receipts are required, task
  failures propagate to the process exit code, malformed execution receipts are
  rejected instead of silently accepted, evaluation failures are reported
  truthfully, completion truth is enforced, and any nested dataclass field is
  serialized rather than only `Event` and `Operation`.
- Durable run identifiers are normalized and required to be non-empty before
  they reach artifact paths.
- Test-time logger isolation: `tpen.bootstrap` no longer leaks its private
  terminal handler or `propagate=False` between tests, so the bootstrap-channel
  assertion no longer depends on test order. Runtime logging is unchanged.

### Validation

- `uv lock --check`
- Package and runtime metadata agree on `0.3.0`.
- Four-facility same-source hardware gate at tree
  `585df3187b14a017e23b6cf0d1cdb23970218125`, one fresh detached checkout and
  exactly one scheduler job per facility: Cannon CPU `38807293` (`1004 passed`,
  `2 skipped`) and GPU `38807314` on one A100 MIG; Polaris PBS `7438353`
  (A100-SXM4-40GB); Aurora PBS `8751664` (Intel Max 1550 tile, flat device
  hierarchy); Frontier Slurm `5250416` (AMD Instinct MI250X). Every accelerator
  lane produced identical science: `95` metric records, `2310` recursively
  finite numeric leaves with zero non-finite, `25/25` data-integrity checks,
  five COMPLETE `tpen.checkpoint` v2 manifests, `505` typed `tpen.*`
  occurrences, exactly one `run_start` and one `run_end`, and no `spenn`
  leakage.
- Merged integration tree `41004485cb61d5d60584134bc8650314ebcd34fa`
  re-verified on Cannon as Slurm `38838638` (`COMPLETED 0:0`): the accepted
  baseline invocation reproduced `1004 passed`, `2 skipped`; the full `tests`
  tree gave `1014 passed`, `2 skipped`, `6 subtests passed`; and the
  authoritative-edit guard alone gave `10 passed`, `6 subtests passed`.
- This release commit adds only version metadata and these notes on top of the
  verified tree above.

## v0.2.3 - Pair Stability v3 diagnostics and durable execution

### Added

- Sampled evaluation records can retain named local energies for individual
  Hamiltonian terms.
- Pair Stability v3 can archive a bounded, checkpoint-free study snapshot from
  final-report provenance.

### Changed

- Hooke singlet opposite-spin cusp range is fixed at `b=0.25` in the training
  and validation model configurations, with an explicit named override for
  controlled ablations.
- Stack workers retain their source checkout paths, and concurrent Submitit
  chunks serialize `uv sync` outside the checkout to preserve clean run
  provenance.

### Validation

- `uv lock --check`
- Package and runtime metadata agree on `0.2.3`.
- Focused local-energy record tests: `10 passed`.
- Pair Stability v3 GPU smoke stack controller `31658932` completed scan train
  and validation (`64/64` each) plus final train and evaluation (`8/8` each).


## v0.2.2 - Pair Stability v3 operations and checkpointing

### Added

- Final checkpoints are written at completed training steps.
- Pair Stability v3 stack runner (`submit_stack.sh`).
- Pair Stability v3 smoke gate reports submitted Slurm job IDs and uses
  estimated run time for smoke and full commands.

### Changed

- Pair Stability v3 launchers use per-CPU Slurm memory, and Submitit memory
  handling is corrected.
- Pair Stability collectors scope results to latest planned grids.
- Pair Stability 1B plotting and final reports are refined.
- Pair Stability documentation, smoke configuration, metric documentation, and
  test coverage align with the updated workflow.
- Local environment files are ignored.
- Contributor branch ownership and pull-request guidance are updated.

### Validation

- `uv lock --check`
- Package and runtime metadata agree on `0.2.2`.
- `uv run --extra cpu python -m compileall spenn run.py typechecked.py`
- Focused Pair Stability v3 and checkpoint callback tests: `66 passed`.

## v0.1.0 - Hooke benchmark integration

### Added

- Hooke two-electron benchmark physics, exact-solution fixtures, and smoke
  training examples.
- VMC runner infrastructure with config-root callbacks, logging, checkpoints,
  status files, health checks, and runtime equivariance diagnostics.
- Evaluation task infrastructure for Hooke, local-energy, trace, orbit, and
  sampler-based diagnostics.
- Hooke pair-validation and pair-stability study machinery, including
  planning, selection, final train/eval/collect/report stages, and Slurm/local
  launch support.
- Experiment documentation for the Hooke study workflows and reusable execution
  planning notes under `experiments/`.

### Changed

- The top-level run path now goes through `run.py` and
  `spenn.run.run_from_config`, with runner-owned training/evaluation logic.
- Hydra config compatibility is not guaranteed with pre-`v0.1.0` configs. The
  active contract is the `v0.1.0` config-root ownership model for callbacks and
  loggers.
- Package metadata now reports `spenn.__version__ == "0.1.0"`.

### Validation

- `uv run --extra cpu python -m compileall spenn run.py typechecked.py`
- `uv run --extra cpu pytest -q`

### Deferred

- A reusable experiment toolkit that fully separates planning, execution, and
  study-specific analysis remains future work.
- Dynamic heterogeneous job assignment is intentionally left out of this
  release; current pair-stability runs use static task tables plus claim-based
  execution where needed.
- Further modular scale-control generalization should stay in focused follow-up
  PRs rather than this integration PR.
