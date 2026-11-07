# H-v2 atomic-hydrogen scientific contract

H-v2 studies **atomic hydrogen**, with one electron, nuclear charge `Z = 1`,
and exact nonrelativistic infinite-nuclear-mass ground-state energy `-0.5 Ha`.
It is not molecular hydrogen (`H2`) or Hooke's atom. The exact state is an
instrument control, not a substitute for a trained-state result.

This directory contains the non-science CPU pipeline check. It owns frozen
Hydra configurations, an independent standard-library scalar oracle, strict
record collection, and tests beside those sources. It does not contain a
science launcher, science output, or qualification result.

The executable route is the normal TPEN entrypoint (the ``tpen`` console
script, or ``tpen.run.run_from_config`` from a test). Do not use
``python -m tpen.run``: that module has no ``__main__`` guard.

The short learned route is deliberately separate from the scalar oracle:
``configs/train.yaml`` uses 256 training walkers and distinct initialization
and sampler seeds, while ``configs/eval.yaml`` restores ``model_only`` into
fresh 1,024-walker, 64-draw streams. Both configs contain exactly one
electron, one charge-one nucleus, raw ``eps: 0`` Coulomb potential, and no
electron-electron factor or term.

``collect.py`` consumes TPEN's real per-run artifacts directly: the
``sampled_eval_table.csv``/``.metadata.json`` trajectory envelope, the run's
own ``resolved_config.yaml`` and ``metadata.json``, and the restored
checkpoint directory. It admits only a complete draw-by-walker grid after
source, checkpoint, and resolved-config identities reconcile at that
envelope. Missing draws, duplicate walker records, nonfinite data, altered
source/checkpoint/config identity, a missing checkpoint commit marker, an
uncommitted tail, and a training-state restore request are all refusals; no
clipped or final-draw statistic is produced.

TPEN's real kinetic, electron-nucleus, and sampler accept-ratio/accept-decision
code is checked against independently-coded references (central finite
differences with a Richardson-extrapolated tolerance, a raw ``-Z/r`` formula,
an independent rederivation of both the |psi|^2 accept exponent and the
resulting accept/reject decision, and the closed-form oracle on an exact
``psi = exp(-alpha r)`` amplitude stub) by
``tests/unit/physics/test_hydrogen_oracle_controls.py`` and
``tests/unit/sampling/test_hydrogen_sampler_controls.py`` in the core tests
tree, not under this directory: the experiment boundary (see "Experiment
boundary" below) restricts files here to configuration,
``tpen.run.run_from_config``, or the ``tpen`` command, and these checks reach
``tpen.physics``/``tpen.sampling`` internals directly. The charge-one hydrogen
setup those core tests use is built from literals in
``tests/helpers/hydrogen_contract.py``, duplicated from (and pinned against,
by ``test_protocol.test_frozen_training_config_selects_the_contract_sampler_literals``)
this directory's frozen ``configs/train.yaml``, not read from it. Measured on
Cannon (job 58147449, head ``a74bc22c``, pre-move): a kinetic-prefactor mutant
and a sign-flipped nuclear potential, both previously uncaught, are now KILLED
by these checks; a sampler |psi|-vs-|psi|^2 mutant is KILLED by both the
behavioral check and the pre-existing grep. The oracle checks cover contract
parts (b), (c) and (d) behaviorally (see "Known gap" below for what this does
not close).
Part (e)'s ``0.02`` Ha energy-error (the pooled mean local energy, not a
median) and ``0.25`` median-absolute-error-improvement criteria are evaluated
at contract scale, with the complete draw-by-walker grid asserted, by
``test_pipeline.test_part_e_contract_scale_training_and_evaluation_energy_criteria``,
marked ``h_v2_pipeline``. That test is skipped by default by
``experiments/atomistic/h-v2/conftest.py`` -- a verifier must pass an explicit
option to run it, once per head rather than once per mutant::

    uv run pytest -q --run-h-v2-pipeline -m h_v2_pipeline experiments/atomistic/h-v2/test_pipeline.py

Without ``--run-h-v2-pipeline``, the standard
``pytest -q experiments/atomistic/h-v2 experiments/atomistic/he-v1`` run skips
it (measured: part (e) alone adds several minutes on top of the rest of the
suite, inside the contract's 2h/4-core CPU budget, which is why verifiers only
pay that cost once per head). The opt-in flag only works with this documented
invocation naming ``test_pipeline.py`` directly; a broader invocation (e.g.
rooted at ``experiments/atomistic``) can only skip the test, not run it.

**Scope, not a gap.** ``tests/unit/physics/test_hydrogen_oracle_controls.py``
already compares TPEN's real kinetic and potential code, and
``tests/unit/sampling/test_hydrogen_sampler_controls.py`` the real sampler
code, against the closed-form oracle on an exact ``psi = exp(-alpha r)``
amplitude stub, covering contract parts (b), (c)
and (d) through the real operators. What is not done, and is not required by
the contract, is pinning a Hydra-constructed ``TPENWaveFunction`` to that exact
state: `tpen.nn.embedding.Embedding` only ever builds
`tpen.nn.mlp.MLP`/`tpen.nn.initialization.SeededLinear` layers, and
`TorchInitializer` only exposes random (Kaiming/Xavier/normal) initializers
with no zero/constant scheme; `tpen.nn.readout.pfaffian.PfaffianReadout`
always folds a learned order-1 feature into the odd-electron border, even
with ``trainable=False``. So no frozen config can make the model's
embedding/readout contribute an exact, input-independent constant. No core
slice is pending for this protocol-only slice.

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

## Lane proposals: operator-reviewed 2026-11-07

All eight entries in `contract_inputs.json` are now **decided** (see
`decided_by` on each, except `one_electron_pipeline_check`, which was already
`admitted_by_operator_direction_2026-11-05`). The operator reviewed them in
TO note `operator-review-contract-proposals-20261107` on item `fffc313a`,
with an item-3 correction and an item-8 answer appended the same day. Items 1
and 2 (percentile `W` versus event probability as the primary axis, and the
recurrence-rule margins) were delegated to the lane, quoted "decide based on
He-v1 analysis"/"same as above", and resolved from a read-only He-v1
fresh-evaluation analysis (FASRC Cannon job `58568904`; note
`primary-measure-and-margins-analysis-20261107` on item `8a67671f`). Items 3
to 7 (nuclear curvature, lineage count, model/optimizer/sampler, horizon and
walker budget, science venue) were decided directly by the operator. Item 8
(the four-stream design) was decided after one round of clarification.

The science design is 50,000 completed updates, retaining update zero and
every 1,000 updates (51 retained checkpoints) while fresh-evaluating a
separate 27-checkpoint set; both the checkpoint counts (2026-11-06, "27 makes
sense. especially if evaluation is expensive") and the horizon itself
(2026-11-07, "you already decided") are decided. It uses three independent
lineages (operator, quoted "3") with initialization seeds `73001..73003` and
sampler seeds `83001..83003`, four fresh streams per evaluated checkpoint,
4,096 walkers, 256 draws, burn-in 1,000, and stride 20 (operator, "you
already decided"). The training sampler uses burn-in 100, 10 Metropolis
steps per update, and proposal scale 0.5 bohr. Evaluation seeds use the
formula recorded in the JSON;
the non-science pipeline and capacity-pilot seed namespaces are excluded from
science and are globally disjoint from one another and from the science seeds.

### Primary measure and margins (items 1-2)

`W` stays primary; exceedance probabilities stay safeguards. The analysis
found that, for an He-v1-sized effect, `W` and the TAIL exceedance thresholds
(`0.5`, `1 Ha`) agree -- He-v1's own 25k-to-50k fresh-evaluation contrast is a
22-25x `W` rise, with both tail thresholds rising too. The BULK exceedance
thresholds (`0.01`, `0.05`, `0.1 Ha`) FALL over the same contrast (`-0.07`,
`-0.24`, `-0.15`); an exceedance-primary rule on those thresholds would have
read He-v1's broken checkpoint as an improvement. This is why `decision_rules`
now states the tail-safeguard contradiction DIRECTIONALLY, and restricts it
to the TAIL thresholds only -- `0.5 Ha`, `1 Ha`, and a third, pilot-frozen
`hydrogen_equivalent_tail_threshold`, quantile-matched before science so the
pilot's basin-window exceedance there is about `0.003`, matching He-v1's
`p(>0.5 Ha) = 0.003` at its `25k` fresh evaluation, its earliest
fresh-evaluated checkpoint, and never retuned once frozen (round-1
pre-review finding F4, option (B), accepted 2026-11-07; wording corrected
round-2, finding G4): a bulk-threshold
decrease that accompanies a `W` rise is expected, not a contradiction, and a
bulk-threshold exceedance that fails to confirm a `W` rise is reported as a
qualification, never a contradiction. Option (A), making the rule an active
contradiction only, was considered and not chosen: He-v1's own bulk
exceedances fell while its `W` rose, so option (A) would make He-v1 itself
`unresolved`, the exact failure the directional rule was meant to fix. The `W`
margins (factor `2` and `0.01 Ha`) are unchanged; the measured noise ceiling
between null stream pairs never exceeds a ratio of about `1.15`. The
exceedance margin is now per-threshold in `exceedance_margins_hartree`:
`0.002 Ha` at the tail thresholds `0.5`, `1 Ha` and the hydrogen-equivalent
threshold (unchanged at the first two, and achievable under the pilot gate
there), and `0.01 Ha` at the bulk thresholds `0.01`, `0.05` and `0.1 Ha`
(widened from `0.002 Ha`, because the He-v1 analysis half-widths there,
`0.0010`-`0.0022 Ha` under a 6-member Bonferroni family -- narrower than this
contract's family, about `1.4x` -- fail the contract's own
half-width-at-most-half-the-margin gate at `0.002 Ha`; no H pilot has run).
This is the `operator_alternative` extended to all three bulk thresholds, not
adopted verbatim. An "increase of at least its margin", for both the tail and
bulk clauses, means the simultaneous LOWER confidence bound of the increase
reaches the margin (round-1 pre-review finding F5(a)). Each confirmation
stream (2 and 3) passes the `W` clause independently; for the exceedance
safeguard, the recurrence confirmation streams (2 and 3) are pooled,
`n=8192` walkers, and the bounded-null check pools all four streams,
`n=16384` walkers (finding F5(b); wording corrected round-2, finding G1). The
zero-event safeguard's half-width is the one-sided 97.5% Clopper-Pearson
upper bound `U`, computed with the pooled `n` of the rule that uses it and
family-adjusted by that rule's own explicit comparison count, not the
evaluated-checkpoint count (round-2 correction to finding F5(c)): the
recurrence family is the three tail thresholds times the three late-checkpoint
comparisons made against the basin (`family_k_recurrence=9`), and the
bounded-null family is the six fixed exceedance thresholds -- the five in
`fixed_exceedance_thresholds_hartree` plus the pilot-frozen
hydrogen-equivalent threshold, which the bounded-null exceedance sentence now
names explicitly -- times every reference-checkpoint (the 21 evaluated
checkpoints at or below `20,000` updates) by late-checkpoint comparison
(`family_k_bounded_null=378`, round-2 correction, finding G2). Recomputed
mechanically by script: `U=0.000718` at `n=8192` and `U=0.000587` at
`n=16384`, both still inside half the tightest margin (`0.001`), so the
pilot gate passes without loosening.

### Nuclear curvature and model/optimizer/sampler (items 3 and 5)

Nuclear curvature is **whatever He-v1 did**, not an independently chosen
value (the operator corrected an initial "trainable" answer to "whatever
He-v1 did we do in Hydrogen experiment"). The primary source is the He-v1
**resolved config of the executed run `train-seed0000` itself**
(`/Users/yizhonghu/he-v1-independent-report-20260823/receipts/resolved_config.yaml`,
sha256 `b9234c469ceadb36291833a2f8ef8a02f29feeba73cdd330949eebbd90d2e8c1`,
listed in `BUNDLE_MANIFEST.json`; round-1 pre-review finding F2, accepted
2026-11-07). It agrees with He-v1's training config at the executed source
commit `418accf153368aab45586dc2a2cc97c18472691c`
(`experiments/atomistic/he-v1/configs/train.yaml`) for the model, optimizer
and sampler blocks, and records an executed horizon of **50,000 updates**,
which differs from the `300000` in `train.yaml` and from the
`production_grid.yaml` plan. The nuclear law is a
`CurvatureElectronNucleusCuspLaw` with `curvature_coefficient=0.01`
(`c`), `curvature_range=1.0` (`d`), `trainable=true` -- this already matched
what the lane had proposed, so no numeric value changed; only the citation
and the "matches He-v1" framing are new. The same source fixes model,
optimizer and training sampler (item 5): one `TPENLayer`, `Embedding`
`out_channels=32`/`hidden_channels=128`/`include_spins=true`,
`EquivariantMixing`/`PathAggregation` `channels=32`/`initial_weight=0.5`/
`implementation=vectorized`/SiLU, `PfaffianReadout(channels=32,
trainable=true)`, `torch.optim.Adam(lr=0.005)`, training runtime `dtype=
float64`, and the training `MetropolisSampler(n_walkers=4096, burn_in=100,
n_steps=10, proposal_scale=0.5)` (the four non-default fields above were
missing from the prior enumeration; round-1 pre-review finding F3). The full
model/optimizer/sampler/runtime blocks of the resolved config govern, apart
from the recorded one-electron-forced deviations; the enumeration is
illustrative, not exhaustive. One-electron-forced deviations from He-v1 are
recorded in `contract_inputs.json`: the `ElectronElectronCusp` factor is
**omitted** from the H-v2 model configuration -- no electron pair exists to
carry a range gradient, so the one-electron system forces its omission (not
retained for parsed-config agreement; round-1 pre-review finding F1,
accepted 2026-11-07); the spin configuration is `n_up=1, n_down=0` against
He-v1's `n_up=1, n_down=1`; the nuclear charge is `1` against He-v1's `2`;
and the `PfaffianReadout`'s odd-electron border stays a readout device, not a
second electron.

### Science venue (item 7)

Cannon, using long jobs, not packed Polaris (operator, "cannon with long
jobs"). Science submits to both FASRC Cannon production GPU partitions in one
job: `kozinsky_gpu` (wall limit `7-00:00:00`, `gpu:nvidia_a100-sxm4-80gb:4`)
and `seas_gpu` (wall limit `2-00:00:00`, mixed `gpu:nvidia_h200:4` and
`gpu:nvidia_a100-sxm4-80gb:4`), pinned with `--constraint=a100` for a
comparable stratum, per the Cannon partition-naming policy (node counts as of
the 2026-08-15 hardware note; they drift -- re-measure with `sinfo` rather
than citing this date). At most 32
GPU-hours per lineage fits comfortably inside even the tighter `seas_gpu`
ceiling. Polaris is not used for H-v2 science; its only role stays the
pipeline check's excluded-seed GPU smoke, part (g) of slice `406194c5`,
which is not a science venue. Slice `406194c5`'s other part, the Cannon
capacity-and-freeze pilot (`admission.pilot`), runs on Cannon, not Polaris --
see "Two admission scopes" below.

### Four-stream design (item 8)

The four-stream select/confirm design (streams 0/1 select the basin, streams
2/3 confirm it) is kept, unchanged in the machine-readable fields below. The
operator: "1 if it is not too difficult. it is ok to be a little scrappy
since we are likely only doing this once. 2 otherwise. in that case I would
do 5 streams for better data." A two-stream fallback is permitted only if the
four-stream design proves too difficult to implement, and its exact layout
(the operator suggested five streams for better data) must be confirmed with
the operator through the manager before use -- it is not predecided here.

The fresh-evaluation estimator is machine-readable in `contract_inputs.json`:
compute each stream's W on its complete 4,096 by 256 grid, use that stream's
median, and take the `ceil(0.999*N)` sorted deviation with no interpolation.
Streams 0/1 select only the basin-triple index; streams 2/3 independently
re-measure early, basin and late W. Bounded-null inference uses all four
streams. The zero-event safeguard is a one-sided 97.5% Clopper--Pearson upper
bound on the walker-level any-exceedance indicator.

The primary statistic, decided 2026-11-07 (see "Primary measure and margins"
above), is

`W = P99.9(abs(E_L - median(E_L)))`.

The contract also reports high quantiles, separate positive/negative survival
curves, and fixed-threshold exceedance probabilities at `0.01`, `0.05`,
`0.1`, `0.5`, and `1 Ha`. MAD, IQR, ordinary variance, and conventional
MCSE are secondary/descriptive only. The `W` recurrence rule requires a factor
of two and `0.01 Ha` change with simultaneous uncertainty support; the bounded
null additionally requires every fixed-threshold exceedance probability to
exclude a late absolute increase of at least its own margin in
`exceedance_margins_hartree`: `0.002 Ha` at `0.5` and `1 Ha`, `0.01 Ha` at
`0.01`, `0.05` and `0.1 Ha`. Its W clause applies to every
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
executor or a lineage. The pipeline check (parts a-g, including the
integrated rerun) runs the science model's nuclear law,
`CurvatureElectronNucleusCuspLaw(0.01, 1.0, trainable)`, with scale-only
overrides, following He-v1's "same file with scale-only overrides" rule; PR
590's configs are adapted to this at its rebase onto this amendment (round-1
pre-review finding F8, accepted 2026-11-07).

The **SCIENCE LINEAGES** remain prohibited until the operator has reviewed the
science plan and explicitly approves it. No pipeline pass, scheduler state,
pilot, or capacity estimate can waive that hold.

The executable controls must separately enforce the raw `-1/r` potential and
`-1/2` Laplacian, absence of an electron-electron term, the intended nuclear
cusp and decay, the odd-electron readout as a readout device rather than a
second electron, rejection at undefined radius zero without regularization,
an independently coded scalar hydrogen oracle, sampler-target/mutation
rejection, and source/resolved-input identity checks.

The nuclear law (decided 2026-11-07 to match He-v1; see "Nuclear curvature and
model/optimizer/sampler" above) has a normalizability gate at every
checkpoint: outer slope `-1 + c/d <= -kappa_min`, with `kappa_min = 0.5`. A
violation makes that checkpoint `unresolved`; it is never
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
logger pattern: for training callbacks, a callback class listed under
`callbacks:` with an `every_n_steps` cadence, writing its metrics through
`context.log` to the configured `loggers:`. The one-shot `initial_checkpoint`
callback fires once rather than on a cadence, and
`readout_trajectory_generator` is an evaluation-side generator, not a
training callback; each module carries a `kind` field
(`training_callback`/`evaluation_generator`/`one_shot_callback`) in
`contract_inputs.json`. Core does not compute or log any of them
unconditionally (operator, 2026-11-06: "remember it is added as a module not
a core metric that always gets logged. This is an old, established design.").
The modules, and the stack layer that provides each one, are listed
machine-readably in `contract_inputs.json` under `logging.modules`.

Cadence: scalar diagnostics log every completed update. Raw training batches
are captured on the iteration-step coordinate, every 50th step including step
0 (operator: "50 makes more sense here"), in partitions of 1,000 iteration
steps per lineage (the field is named `partition_size_steps`, matching the
iteration-step capture coordinate rather than the completed-update count),
written as NumPy column bundles with a JSON integrity sidecar
(operator: "numpy seems to make more sense as long as it does not cause OOM
events"). The out-of-memory condition is checked twice: a bounded-memory test
in the training-batch recorder layer, and a logging-on measurement in the
full-card pilot. A recorded training batch is the pre-update sample; each row
carries the iteration step (the same coordinate used for the capture cadence
and the partition ranges), the completed-update count after the iteration,
and whether the update was applied. Skipped updates are recorded, not
dropped.

All 51 checkpoints per lineage are retained (update 0 and every 1,000 through
50,000). Fresh evaluation stays at the predeclared 27 per lineage (operator,
2026-11-06: "27 makes sense. especially if evaluation is expensive"). Both
counts are now **decided**, not proposed.

Storage lifecycle: the working location is Cannon netscratch, not scratch,
which is purged after 90 days, so the HPSS archive must be made inside that
window. The Holystore set is chosen retroactively, after the analysis; about
26 to 28 GiB is acceptable for now, and excludes the checkpoints themselves
(operator: "archive everything leading up to the conclusion except the
checkpoints themselves into holystore"). The HPSS scope is the whole run,
checkpoints included, compressed. The procedures are the `archive-experiment`
and `backup-experiment-cannon` skills. Nothing on netscratch is removed
without a Task Orchestrator lifecycle disposition naming the exact paths and
a passed recovery verification of the archive.

Storage figures, recomputed at the science scale of 4,096 training walkers:
fresh evaluation is `3 x 27 x 4 x 4,096 x 256 = 339,738,624` rows, about 28.1
GiB compact at 89 bytes per row (the conservative envelope); raw training
batches, captured every 50th iteration step including step 0 through 50,000
(1,001 batches per lineage), is `3 x 1,001 x 4,096 = 12,300,288` rows, about
0.928 GiB compact at 81 primitive bytes per row, before container and
compression. At 20 batches per 1,000-step partition, 1,001 batches per
lineage gives 50 full partitions plus one single-batch partition (holding
only step 50,000), 51 partitions per lineage, `3 x 51 = 153` data objects
total, not the 150 the storage proposal costed. Both byte-per-row figures and
their source are recorded in `contract_inputs.json`. Compression ratios are
unmeasured and are labelled so in `contract_inputs.json`.

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
critical value computed from the declared grid, counted once per stream
(`K = 122` comparisons across both streams, false-fail rate 0.001, giving
`c = 4.45998023125495`). The comparison centers `A` on the per-stream sample
median of `E_L`, as the science estimator does; the median is recomputed
inside every whole-walker resample, and the exact survival value at the fixed
thresholds is the oracle value centered on the exact median `-0.540498`.

Second, the primary science evidence for the tail is the **full** positive
and negative survival and exceedance curves computed from the retained raw
rows, on the same level grid, down to its supported depth. No reported
percentile, `W`, variance, MAD, IQR, or fitted tail index replaces the raw
rows; this is the lane's scientific response to the operator's concern.

## Pull request 560 incorporation

Pull request 560 (`origin/codex/one-electron-variance-20261028`, tip
`0da2b23d628a15b18ac2031ff5f72dff0c010280`, preserved at the local-only ref
`refs/preserved/codex-one-electron-variance-20261028-0da2b23d` in the main
checkout and at `refs/pull/560/head` on GitHub) is superseded by
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
operator has decided no guard will be added (2026-11-06). Experiment code
must use the installed `tpen` entrypoint or call `tpen.run.run_from_config`
explicitly; `tpen.run.main` is not an experiment-facing surface. A green exit
code from `python -m tpen.run` proves nothing.

## Still open

The amplitude-risk **definition and thresholds are both decided**. The
operator (2026-11-06, "Tell the orchestrator to make a call that makes sense
from previous runs of exact models and trained models. If it needs to spike
something so be it.") handed the call to the lane; it is no longer held for
the operator. The accepted definition is the fraction of samples whose
`logabs` falls below the median of the finite `logabs` values minus a
threshold, computed per training batch and per fresh-evaluation draw
separately, plus the nonfinite-`logabs` count and the invalid-sign count,
never called a Pfaffian or readout floor unless the pre-floor readout
amplitude is itself measured. A bounded Cannon CPU spike (branch
`agent/h-v2-amplitude-spike` at `39135fda1388ef319c2325c8ad934de3b3c513db`,
Cannon jobs `58383577`, `58388053`, `58388109`) froze the study threshold list
`tau in {-3, -4, -5, -7, -10}`: `-5` is the primary descriptive scale, `-3`/
`-4` are resolved sensitivities, `-7` is "marginal resolution at the spike's
evaluation scale (4096 x 64 rows per stream, 17 of 20 streams resolved);
unmeasured at the contract's 4096 x 256", and `-10` is "rare or unresolved; a
zero is not a healthy zero". The
spike found the exact, imperfect, untrained and trained states all fall in
the same narrow band, confirming the module cannot classify state quality.
It remains descriptive only: no alarm and no science gate. Implementation
(layer `fa3c35a8`) and independent verification against the frozen list
still remain.

Every `lane_proposals` entry is now decided (2026-11-07 operator review; see
"Lane proposals: operator-reviewed 2026-11-07" above). That does not lift the
science hold: **science lineages remain prohibited** pending this amendment's
own Cannon verification and the human merge of this pull request, pending
the remaining pipeline-check (`5cfcca3e`), excluded-seed Polaris smoke
(`406194c5`), and integrated-rerun (`2f1b2921`) gates already described above,
and pending a third admission gate added round-2 (finding G3):
`admission.pilot`.

Slice `406194c5` has two parts with separate placements and ceilings (round-3
pre-review finding H1, accepted 2026-11-07): part (g), the excluded-seed
Polaris GPU route smoke, keeps its current caps (1,000 updates, one
node-hour including setup) and runs first; then, after part (g) and
qualification, the Cannon capacity-and-freeze pilot runs on one A100
(`--constraint=a100`) with its own ceiling of **16 GPU-hours** including
setup, checkpoint writes and fresh evaluation -- flagged in the contract as
the lane's proposal for this pull request's operator review, separate from
and not counted against the 96 GPU-hour science ceiling. If the pilot
exceeds its ceiling it stops and reports; the ceiling is not extended.

`admission.pilot` is the contract definition of the Cannon pilot run: the
science model/sampler shape at the excluded `capacity_pilot` seeds, run
through a **20,000-update target** (4,000-update hard minimum, round-3
pre-review finding H3, accepted 2026-11-07) so its own basin search covers
the same centers (`3000..19000`) the science recurrence rule searches. It
fresh-evaluates the contract's evaluated checkpoints it reaches, at most
20,000, seeded by a dedicated `capacity_pilot.evaluation_formula`
(`996000 + 10*pilot_checkpoint_ordinal + stream_index`, 21 checkpoints x 4
streams = 84 seeds, disjoint from every other namespace; round-3 pre-review
finding H2, accepted 2026-11-07). If the ceiling stops the pilot before
20,000 updates, the shortfall is reported. The freeze uses the pilot's own
smallest-mean-W consecutive triple over the centers it reached, by the
recurrence rule's selection, not necessarily the earliest triple. It freezes
`measurement.hydrogen_equivalent_tail_threshold` and measures the pilot-gate
simultaneous-band half-widths at all six exceedance margins; `admission.pilot`
is the contract definition of the run, implemented by slice `406194c5`'s
Cannon pilot part. None of these gates is closed by this amendment. If the
pilot gate fails at the hydrogen-equivalent threshold, the F5 stop rule
applies and the decision returns to the lane and the operator with the
measured value as evidence; a failure is never grounds to retune the
threshold, its margin, or its quantile after the number is seen (finding
G5).
