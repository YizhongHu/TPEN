# H-v2 atomic-hydrogen scientific contract

H-v2 studies **atomic hydrogen**, with one electron, nuclear charge `Z = 1`,
and exact nonrelativistic infinite-nuclear-mass ground-state energy `-0.5 Ha`.
It is not molecular hydrogen (`H2`) or Hooke's atom. The exact state is an
instrument control, not a substitute for a trained-state result.

This directory is a documentation and admission slice. It contains no
launcher, training code, science output, or qualification result.

## Evidence boundary

The comparison source is pinned to `origin/dev` commit
`67beab2095cc987eb044abceeecfdb549a5793fb`: the He-v1 README, production
grid, collector, and report. Historical run identity and corrections are
listed in `contract_inputs.json`. The current `production_grid.yaml` at that
commit names a 100k/200k/300k plan; it is not the executed historical
25k/50k plan.

The historical helium basin is a **5k-block median of within-batch variance
on a changing training model**, approximately `0.00173 Ha^2`. It is training
history, not a frozen-checkpoint population estimate. The reported `s10=26570`
and `s90=44982` values are definition-dependent summaries; the definition is
only in the unopened Netscratch report, so neither value is the point where a
basin was first left. The retained fresh evaluations show a finite-range
25k-to-50k tail contrast, but do not establish a population moment class or a
common-estimand decrease-then-increase trajectory.

## Lane proposals pending operator review

The eight entries in `contract_inputs.json` are deliberately lane proposals,
not operator decisions. Each has a rationale and alternatives. The live
contention is visible: percentile `W` versus event probability as the primary
axis; fixed versus trainable nuclear curvature; three versus five lineages;
the factor-two/`0.01 Ha` and `0.002` effect margins; the model/optimizer and
training sampler; and Cannon single-card science versus efficiently packed
Polaris science.

The proposed science design is 50,000 completed updates, retaining update zero
and every 1,000 updates (51 retained checkpoints) while fresh-evaluating a
separate 27-checkpoint set. The checkpoint counts themselves are **decided**,
2026-11-06: the operator confirmed "27 makes sense. especially if evaluation
is expensive". All 51 retained checkpoints are kept; fresh evaluation stays at
27. The rest of the science plan (lineage count, walker budget, venue) remains
a lane proposal pending operator review. It proposes three independent
lineages with
initialization seeds `73001..73003` and sampler seeds `83001..83003`, four
fresh streams per evaluated checkpoint, 4,096 walkers, 256 draws, burn-in
1,000, and stride 20. The training sampler uses burn-in 100, 10 Metropolis
steps per update, and proposal scale 0.5 bohr. Evaluation seeds use the
formula recorded in the JSON;
the non-science pipeline and capacity-pilot seed namespaces are excluded from
science and are globally disjoint from one another and from the science seeds.

The fresh-evaluation estimator is machine-readable in `contract_inputs.json`:
compute each stream's W on its complete 4,096 by 256 grid, use that stream's
median, and take the `ceil(0.999*N)` sorted deviation with no interpolation.
Streams 0/1 select only the basin-triple index; streams 2/3 independently
re-measure early, basin and late W. Bounded-null inference uses all four
streams. The zero-event safeguard is a one-sided 97.5% Clopper--Pearson upper
bound on the walker-level any-exceedance indicator.

The proposed primary statistic is

`W = P99.9(abs(E_L - median(E_L)))`.

The contract also reports high quantiles, separate positive/negative survival
curves, and fixed-threshold exceedance probabilities at `0.01`, `0.05`,
`0.1`, `0.5`, and `1 Ha`. MAD, IQR, ordinary variance, and conventional
MCSE are secondary/descriptive only. The `W` recurrence rule requires a factor
of two and `0.01 Ha` change with simultaneous uncertainty support; the bounded
null additionally requires every fixed-threshold exceedance probability to
exclude a late absolute increase of `0.002`. Its W clause applies to every
evaluated reference checkpoint through update 20,000 and excludes the declared
late count reaching `max(2 W(k), W(k) + 0.01 Ha)`. A flat MAD is not
equivalence.

The gating instrument is W, the high quantiles, and the exceedance
probabilities, provided complete records and qualified dependence are present.
Held-out beta survival-slope curvature and calibrated generalized-Pareto `xi`
are non-gating interpretation diagnostics; Hill is context only. Their drift
is measured across the declared held-out ordered draw prefixes and supported
tail-fraction intervals. An unresolved ultimate tail class does not by itself
make the trajectory label `unresolved`; an absent or inconclusive gating
diagnostic is never a pass. No finite-prefix diagnostic
proves an ultimate finite or infinite variance.

## Closed outcomes

The only result categories are `recurrence`, `bounded_null`, and
`unresolved`. Recurrence requires complete records, qualified uncertainty,
the predeclared basin search, late persistence, and at least two of three
independent lineages. Bounded null requires all three complete lineages,
simultaneous inference over the full predeclared search, and both the `W` and
exceedance safeguards. Missing identities, records, controls, dependence
qualification, operator decisions, or supported margins produce `unresolved`.

## Two admission scopes

The non-science one-electron **PIPELINE CHECK** is admitted under the
operator's 2026-11-05 direction. Parts (a)-(f), including the independent
analytic oracle, imperfect-state rejection, short learned route, fresh restore,
and complete-record refusal controls, belong to slice `5cfcca3e` on Cannon CPU.
For part (e), the training sampler is `256` walkers with burn-in `100`, `10`
Metropolis steps per update, and proposal scale `0.5` bohr; each fresh
evaluation uses `1,024` walkers, `64` draws, `2` streams, burn-in `1,000`, and
stride `20`. Part (g), the excluded-seed GPU route smoke, belongs to slice
`406194c5` on Polaris. Slice `2f1b2921` reruns the integrated check on the exact
candidate. The JSON gives each part's pass criteria and its disjoint non-science
seed namespace. This check is not a science result and does not authorize the
executor or a lineage.

The **SCIENCE LINEAGES** remain prohibited until the operator has reviewed the
science plan and explicitly approves it. No pipeline pass, scheduler state,
pilot, or capacity estimate can waive that hold.

The executable controls must separately enforce the raw `-1/r` potential and
`-1/2` Laplacian, absence of an electron-electron term, the intended nuclear
cusp and decay, the odd-electron readout as a readout device rather than a
second electron, rejection at undefined radius zero without regularization,
an independently coded scalar hydrogen oracle, sampler-target/mutation
rejection, and source/resolved-input identity checks.

The proposed nuclear law has a normalizability gate at every checkpoint:
outer slope `-1 + c/d <= -kappa_min`, with `kappa_min = 0.5` proposed for
operator review. A violation makes that checkpoint `unresolved`; it is never
filtered out. `c = 0` is the exact hydrogen answer, so upward curvature drift
is itself a reportable model signal. The independently recomputed analytic
controls are `alpha=1.2`: mean `-0.48 Ha`, variance `0.0576 Ha^2`, median
local energy `-0.540498 Ha`, `W=2.339741 Ha`, with exceedance probabilities
`0.008304` at 1 Ha and `0.926514` at 0.01 Ha; `alpha=1` has constant `-0.5 Ha`
and zero W/exceedances. The W/bootstrap instrument must reproduce these
controls on independent Gamma draws.

The early comparison uses updates 0 and 1,000; the contract records that this
decrease component may have little dynamic range because update zero is
untrained. A post-transient secondary early point remains an operator
alternative and does not silently change the rule.

## Logging and diagnostics

Two observation classes are never pooled. **Training history** comes from a
changing model and is diagnostic only. **Fresh evaluation** of a frozen
checkpoint is the primary evidence.

Every new measurement beyond what already exists is an **optional module**
that a run's configuration adds, following the established callback and
logger pattern: a callback class listed under `callbacks:` with an
`every_n_steps` cadence, writing its metrics through `context.log` to the
configured `loggers:`. Core does not compute or log any of them
unconditionally (operator, 2026-11-06: "remember it is added as a module not
a core metric that always gets logged. This is an old, established design.").
The modules, and the stack layer that provides each one, are listed
machine-readably in `contract_inputs.json` under `logging.modules`.

Cadence: scalar diagnostics log every completed update. Raw training batches
are captured every 50th update (operator: "50 makes more sense here"), in
partitions of 1,000 updates per lineage, written as NumPy column bundles with
a JSON integrity sidecar (operator: "numpy seems to make more sense as long
as it does not cause OOM events"). The out-of-memory condition is checked
twice: a bounded-memory test in the training-batch recorder layer, and a
logging-on measurement in the full-card pilot. A recorded training batch is
the pre-update sample; each row carries the iteration step, the
completed-update count after the iteration, and whether the update was
applied. Skipped updates are recorded, not dropped.

All 51 checkpoints per lineage are retained (update 0 and every 1,000 through
50,000). Fresh evaluation stays at the predeclared 27 per lineage (operator,
2026-11-06: "27 makes sense. especially if evaluation is expensive"). Both
counts are now **decided**, not proposed.

Storage lifecycle: the working location is Cannon netscratch, not scratch,
which is purged after 90 days, so the HPSS archive must be made inside that
window. The Holystore set is chosen retroactively, after the analysis; about
26 to 28 GiB is acceptable for now. The procedures are the `archive-experiment`
and `backup-experiment-cannon` skills. Nothing on netscratch is removed
without a Task Orchestrator lifecycle disposition naming the exact paths and
a passed recovery verification of the archive.

Storage figures, recomputed at the science scale of 4,096 training walkers:
fresh evaluation is `3 x 27 x 4 x 4,096 x 256 = 339,738,624` rows, about 28.1
GiB compact; raw training batches every 50th update is `3 x 1,000 x 4,096 =
12,288,000` rows, about 0.93 GiB compact before container and compression.
Compression ratios are unmeasured and are labelled so in
`contract_inputs.json`.

## Tail characterization

The operator's concern that "only a few percentile values may not be enough"
is answered in two parts, recorded machine-readably under
`tail_characterization` in `contract_inputs.json`.

First, the sampler itself is qualified over full survival curves, not a few
percentiles. The exact state (`alpha=1`) has radius `R ~ Gamma(3, 2)`; the
imperfect state (`alpha=1.2`), sampled from its own `|psi|^2`, has `R ~
Gamma(3, 2.4)` and local energy `E_L = -0.72 + 0.2/r`. The large positive
local-energy tail for `alpha=1.2` comes from the **inner** radial tail (small
`r`), which the superseded stub check never examined: it compared the CDF
only up to `r=3`, about the 94th percentile of the outer radius. The new
grid covers lower-tail CDF and upper-tail survival levels down to `0.0001`
for both states, plus fixed-threshold and graded exceedance probabilities of
the local-energy deviation for `alpha=1.2`. A level gates only if its
expected count is at least 100 per stream at the contract's fresh-evaluation
shape; deeper levels are reported but never gated. Each gated level is
checked with a whole-walker bootstrap z-score against a Bonferroni-corrected
critical value computed from the declared grid (`K` comparisons, false-fail
rate 0.001).

Second, the primary science evidence for the tail is the **full** positive
and negative survival and exceedance curves computed from the retained raw
rows, on the same level grid, down to its supported depth. No reported
percentile, `W`, variance, MAD, IQR, or fitted tail index replaces the raw
rows; this is the lane's scientific response to the operator's concern.

## Pull request 560 incorporation

Pull request 560 (`origin/codex/one-electron-variance-20261028`, tip
preserved at `0da2b23d628a15b18ac2031ff5f72dff0c010280`) is superseded by
H-v2 for the hydrogen physics controls, but several of its capabilities are
worth carrying forward (operator: "the others look good to incorporate").
The mapping from each finding to its destination layer is machine-readable
under `pr_560_incorporation` in `contract_inputs.json`: tail-calibrated
sampler bands, science-collector integrity checks, the `InitialCheckpoint`
and `ParameterActivity` callbacks, the readout trajectory generator,
stricter trajectory-record metadata validation, and moment-band calibration
of the imperfect state with a CUDA exact canary. Items with no H-v2 use
(nucleus-free trajectory records, `OneElectronGaussian`, the Hooke one-electron
facts, `GaussianConfinement.scalar_diagnostics`) are handed to the manager for
a possible Hooke lane, not planned here.

## Experiment boundary

Experiment code and tests reach core only through configuration,
`tpen.run.run_from_config`, or the installed `tpen` command (operator,
2026-11-06: "well it can also run the TPEN command? that is allowed as
well."). An integration test that needs core internals for fixtures or
mutation hooks lives in the core `tests/` tree, not under `experiments/`
(operator: "integration tests that need core access elements should be in
the core. otherwise the interface is only through the config").

**Trap:** `python -m tpen.run ...` exits 0 and does nothing. `tpen/run.py`
has a `main` function but no `if __name__ == "__main__":` guard, and the
operator has decided no guard will be added (2026-11-06). Use the installed
`tpen` entrypoint, or call `tpen.run.run_from_config`/`tpen.run.main`
explicitly. A green exit code from `python -m tpen.run` proves nothing.

## Still open

The amplitude-risk definition and threshold remain explicitly **open**: the
lane's proposal is the fraction of walkers whose `logabs` falls below the
batch-median `logabs` minus a threshold, plus nonfinite-`logabs` and sign
counts, never called a Pfaffian or readout floor unless the pre-floor readout
amplitude is itself measured. The science venue and resource ceiling, and
every other `lane_proposals` entry, remain pending operator review. Science
lineages remain prohibited.
