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

``collect.py`` admits only a complete committed draw-by-walker grid after
source, checkpoint, and resolved-config identities reconcile. Missing,
duplicate, nonfinite, wrong-identity, training-state, and uncommitted records
are refusals; no clipped or final-draw statistic is produced.

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
separate 27-checkpoint set. It proposes three independent lineages with
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
